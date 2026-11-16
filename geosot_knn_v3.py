#!/usr/bin/env python3
# coding: utf-8
"""
Legacy KNN V3 helpers: adaptive grid pruning, prefix queries, and optional geometry
reranking. Find the finest safe level meeting the margin (default 100 m), query its prefix,
widen to a 3x3 neighborhood or coarsen when needed, and rank centroids with cosine-weighted
planar distance. This planar metric is an approximation, not a global ellipsoidal ordering
guarantee. Optional edge/vertex reranking supports nonpoint geometries.

V1 used level-digit graph traversal; V2 used circular quadtree filling; V3 uses a smaller
prefix neighborhood. The published main KNN entry is V3.2, which depends on these helpers.

Neo4j prefix queries require an appropriate main_code index. The default PG reference uses
an innermost GiST KNN candidate query, outer deduplication, and geography reranking; --pg-
dedup additionally runs the slower DISTINCT ON alternative.

Usage:
    python geosot_knn_v3.py --queries 100
    python geosot_knn_v3.py --queries 10 --k 10 --margin 200
    python geosot_knn_v3.py --queries 50 --k 1,3,5,10
    python geosot_knn_v3.py --queries 50 --k 5 --seed 7
    python geosot_knn_v3.py --queries 50 --no-geometry
    python geosot_knn_v3.py --queries 10 --verbose
    python geosot_knn_v3.py --queries 100 --pg-dedup
"""

import bisect as _bisect
import csv
import heapq
import math
import os
import random
import shutil
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import psycopg2
from neo4j import GraphDatabase

from geosot_spatial_reasoning import (
    code_to_bbox,
    fast_code_to_center,
    is_valid_g0_code,
    GRID_SIZES,
    PG_CONFIG,
)
from geosot_geometry_reasoning import (
    fast_estimate_vertices,
    geometry_min_distance_m,
    geometry_min_distance_m_np,
    _build_cos_ref_matrix,
)
try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


# ============================================================================
# Neo4j config
# ============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

DEG_TO_M = 111_320.0

# Parallel geometry rerank: after LCPC pruning, rerank volume is usually far below candidates; single-thread is enough.
# Keep a parallel API for ablation; set a huge default to disable.
_GEO_PARALLEL_MIN = 100_000

# Safety margin (m) on LCPC bbox lower bound: compensates GeoSOT quantization error (~half cell step)
_LCPC_SAFETY_M = 10.0

# NumPy batch size: 128-256 is usually best for real workloads
# (large batch → more continue-pruning inside batch; small batch → earlier break between batches)
# Override at test time via env: GEO_BATCH_SIZE=128 python ...
import os as _os
_GEO_BATCH_SIZE: int = int(_os.environ.get("GEO_BATCH_SIZE", "256"))

_FEATURE_INFO_CYPHER = (
    "UNWIND $codes AS code\n"
    "MATCH (c:GeoSOTCentroidCode {main_code: code})"
    "<-[:HAS_CENTROID_CODE]-(g:SpatialFeature)\n"
    "OPTIONAL MATCH (c)-[:HAS_GEOMETRY_TYPE]->(gt:GeoFeatureType)\n"
    "OPTIONAL MATCH (g)-[:HAS_COMMON_PREFIX_CODE]->(cp:GeoSOTCommonPrefixCode)\n"
    "OPTIONAL MATCH (g)-[hv:HAS_VERTEX_CODE]->(v:GeoSOTVertexCode)\n"
    "WITH c.main_code AS mc, gt.type_name AS geo_type,\n"
    "     cp.prefix_code AS prefix_code,\n"
    "     v.vertex_code AS vc, hv.seq AS vi\n"
    "ORDER BY mc, vi\n"
    "WITH mc, geo_type, prefix_code,\n"
    "     collect(CASE WHEN vc IS NOT NULL"
    "             THEN vc ELSE null END) AS raw\n"
    "RETURN mc, geo_type, prefix_code,\n"
    "       [x IN raw WHERE x IS NOT NULL] AS vertex_codes"
)


def _geo_score_one_prepared_task(
    task: Tuple[
        str,
        List[str],
        List[Tuple[float, float]],
        str,
        List[str],
        List[Tuple[float, float]],
        str,
        float,
    ],
) -> Tuple[str, float]:
    """
    Rerank one geometry candidate in a ProcessPool worker using predecoded vertices.

    @param task (mc, q_codes, q_coords, q_type, c_codes, c_coords, c_type, fallback_sq_deg2).
    @returns (main_code, distance_m).
    """
    import math as _math

    (mc, q_codes, q_coords, q_type,
     c_codes, c_coords, c_type, fallback_sq) = task
    try:
        cos_mx = _build_cos_ref_matrix(q_codes, c_codes)
        dist, _, _ = geometry_min_distance_m(
            q_coords, q_type, c_coords, c_type, cos_mx)
        return mc, dist
    except Exception:
        return mc, _math.sqrt(fallback_sq) * DEG_TO_M


def _geo_score_chunk_task(
    task: Tuple[
        List[str],
        List[Tuple[float, float]],
        str,
        List[Tuple[str, List[str], List[Tuple[float, float]], str, float]],
    ],
) -> List[Tuple[str, float]]:
    """
    Rerank a batch in a ProcessPool worker, transmitting query vertices only once.

    @param task (q_codes, q_coords, q_type, [(mc, c_codes, c_coords, c_type, fallback_sq), ...]).
    @returns [(main_code, distance_m), ...].
    """
    q_codes, q_coords, q_type, items = task
    return [
        _geo_score_one_prepared_task(
            (mc, q_codes, q_coords, q_type,
             c_codes, c_coords, c_type, fallback_sq))
        for mc, c_codes, c_coords, c_type, fallback_sq in items
    ]


# ============================================================================
# GR-side KNN candidate prefix client cache (counterpart to PG pg_prewarm)
# ============================================================================
#
# Benchmark protocol: disclose the same setup/measurement boundary as the other operators.
# PG uses pg_prewarm to load poi_new + GiST into shared_buffers; GR
# pulls all main_codes for a geom_type (typically Point) from Neo4j once
# into Python memory, sorts lexicographically, and does local prefix match via bisect
# (O(log N + k)). This is the graph-native warm-cache protocol for the KNN operator,
# Corresponding implementation; follows the same client-side index design as forward-overlay MainCodeIndex
# and reverse-overlay CommonPrefixIndex.
#
# Codes stay as strings; no lon/lat decode. RAM ~30 bytes × N_POI; 5M POI
# ≈ 150 MB.

class POIPrefixCache:
    """
    Client-side KNN prefix index built during explicit setup. Fetch main_codes by geometry type,
    sort them, and match prefixes locally with bisect. This setup/materialization is reported
    separately from measured queries.

    @attr _sorted Sorted codes keyed by geometry type.
    @attr elapsed_sec Load time in seconds.
    """

    def __init__(self, session, geom_types: Iterable[str],
                 verbose: bool = True):
        """
        @param session Neo4j session.
        @param geom_types Geometry types to preload, for example ["Point"].
        @param verbose Whether to print progress.
        """
        self._sorted: Dict[str, List[str]] = {}
        # Centroid vectorization (main point-KNN speedup): precompute centroids for all main_codes,
        # aligned with the sorted array. _centers[gt] is (N,2) (lon,lat);
        # _code_idx[gt] maps main_code -> row. Candidate ranking uses
        # NumPy slice + argpartition for top-m, avoiding per-candidate Python decode+full sort.
        self._centers: Dict[str, "_np.ndarray"] = {}
        self._code_idx: Dict[str, Dict[str, int]] = {}
        t0 = time.perf_counter()
        for gt in geom_types:
            t_g = time.perf_counter()
            try:
                rows = list(session.run(
                    "MATCH (c:GeoSOTCentroidCode)"
                    "-[:HAS_GEOMETRY_TYPE]->"
                    "(:GeoFeatureType {type_name: $gt}) "
                    "RETURN c.main_code AS mc",
                    gt=gt))
            except Exception as e:
                if verbose:
                    print(f"  [poi_cache] {gt} load failed: {e}")
                self._sorted[gt] = []
                continue
            codes = sorted({
                str(r["mc"]).strip().upper()
                for r in rows
                if r.get("mc") is not None
                and is_valid_g0_code(str(r["mc"]).strip().upper())
            })
            self._sorted[gt] = codes
            if verbose:
                dt = (time.perf_counter() - t_g) * 1000
                print(f"  [poi_cache] {gt:<12s}: "
                      f"{len(codes):>9,} codes  ({dt:.0f} ms)")
            self._precompute_centers(gt, codes, verbose)
        self.elapsed_sec = time.perf_counter() - t0
        if verbose:
            total = sum(len(c) for c in self._sorted.values())
            print(f"  [poi_cache] 完成: 共 {total:,} 个 codes, "
                  f"耗时 {self.elapsed_sec * 1000:.0f} ms")

    def _precompute_centers(
        self, geom_type: str, codes: List[str], verbose: bool = False,
    ) -> None:
        """
        Precompute centroid (lon,lat) arrays and code-to-row indexes for vectorized ranking. Invalid
        codes receive a large sentinel and sort last.

        @param geom_type Geometry-type key.
        @param codes Sorted codes aligned with centroid rows.
        @param verbose Whether to show progress.
        """
        if not _HAS_NP or not codes:
            return
        n = len(codes)
        arr = _np.empty((n, 2), dtype=_np.float64)
        SENT = 1.0e9  # Invalid-coordinate sentinel: impossible lon/lat values yield an extremely large distance.
        t_c = time.perf_counter()
        it = range(n)
        pbar = None
        if verbose:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=n, desc=f"  [poi_cache] {geom_type} centers",
                            unit="code", leave=False)
            except Exception:
                pbar = None
        for i in it:
            try:
                lon, lat = fast_code_to_center(codes[i])
                arr[i, 0] = lon
                arr[i, 1] = lat
            except Exception:
                arr[i, 0] = SENT
                arr[i, 1] = SENT
            if pbar is not None and (i & 0x3FFF) == 0:
                pbar.update(min(0x4000, n - i))
        if pbar is not None:
            pbar.n = n
            pbar.close()
        self._centers[geom_type] = arr
        self._code_idx[geom_type] = {c: i for i, c in enumerate(codes)}
        if verbose:
            dt = (time.perf_counter() - t_c) * 1000
            print(f"  [poi_cache] {geom_type:<12s}: centers precomputed "
                  f"({n:,} pts, {dt:.0f} ms, {arr.nbytes / 1e6:.0f} MB)")

    def rank_candidates(
        self,
        geom_type: str,
        codes: Iterable[str],
        q_lon: float,
        q_lat: float,
        cos2: float,
        m: int,
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Rank candidates by gathering precomputed centroid rows and using NumPy argpartition for
        Top-m, avoiding repeated decoding and a full Python sort.

        @param geom_type Geometry-type key.
        @param codes Candidate codes present in _sorted[geom_type].
        @param q_lon Query longitude.
        @param q_lat Query latitude.
        @param cos2 Longitude weight cos(latitude)^2.
        @param m Number of coarse leaders, usually max(coarse_top_n,k).
        @returns Ascending [(code, squared_planar_degrees), ...], length <=m, or None when the NumPy
            path is unavailable.
        """
        if not _HAS_NP:
            return None
        centers = self._centers.get(geom_type)
        cidx = self._code_idx.get(geom_type)
        sorted_codes = self._sorted.get(geom_type)
        if centers is None or cidx is None or not sorted_codes:
            return None
        idx = _np.fromiter(
            (cidx[c] for c in codes if c in cidx), dtype=_np.int64)
        n = int(idx.size)
        if n == 0:
            return None
        dlon = centers[idx, 0] - q_lon
        dlat = centers[idx, 1] - q_lat
        d = dlon * dlon * cos2 + dlat * dlat
        mm = m if m < n else n
        if mm < n:
            part = _np.argpartition(d, mm - 1)[:mm]
        else:
            part = _np.arange(n)
        order = part[_np.argsort(d[part], kind="stable")]
        sel_idx = idx[order].tolist()
        sel_d = d[order].tolist()
        return [(sorted_codes[i], dv) for i, dv in zip(sel_idx, sel_d)]

    def find_indices_batch(
        self,
        geom_type: str,
        prefixes: Iterable[str],
        exclude_code: Optional[str] = None,
    ) -> "Optional[_np.ndarray]":
        """
        Index-native prefix retrieval: return deduplicated row indexes in _sorted[geom_type], not
        code strings. Match the same candidates as find_by_prefixes_batch, using bisect ranges and
        np.arange for direct rank_indices input.

        @returns Unordered np.int64 row indexes, or None when NumPy/type data is unavailable.
        """
        if not _HAS_NP:
            return None
        codes = self._sorted.get(geom_type)
        if not codes:
            return _np.empty(0, dtype=_np.int64)
        parts: List["_np.ndarray"] = []
        for prefix in prefixes:
            lo = _bisect.bisect_left(codes, prefix)
            hi = _bisect.bisect_right(codes, prefix + '\uffff')
            if hi > lo:
                parts.append(_np.arange(lo, hi, dtype=_np.int64))
        if not parts:
            return _np.empty(0, dtype=_np.int64)
        # Same-length neighbor cell prefixes are not prefixes of each other; [lo,hi) ranges are disjoint,
        # so concatenate directly — no np.unique sort (cross-round dedup is caller's
        # np.union1d).
        idx = parts[0] if len(parts) == 1 else _np.concatenate(parts)
        if exclude_code:
            ec = exclude_code.strip().upper()
            pos = _bisect.bisect_left(codes, ec)
            if pos < len(codes) and codes[pos] == ec:
                idx = idx[idx != pos]
        return idx

    def rank_indices(
        self,
        geom_type: str,
        idx: "_np.ndarray",
        q_lon: float,
        q_lat: float,
        cos2: float,
        m: int,
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Gather precomputed centroids by row index and use argpartition for Top-m, avoiding code-to-
        row dictionary lookups.

        @param idx np.int64 candidate row indexes valid for this geometry type.
        @returns Ascending [(code, squared_planar_degrees), ...], length <=m, or None if unavailable.
        """
        if not _HAS_NP:
            return None
        centers = self._centers.get(geom_type)
        sorted_codes = self._sorted.get(geom_type)
        if centers is None or not sorted_codes:
            return None
        idx = _np.asarray(idx, dtype=_np.int64)
        n = int(idx.size)
        if n == 0:
            return None
        dlon = centers[idx, 0] - q_lon
        dlat = centers[idx, 1] - q_lat
        d = dlon * dlon * cos2 + dlat * dlat
        mm = m if m < n else n
        if mm < n:
            part = _np.argpartition(d, mm - 1)[:mm]
        else:
            part = _np.arange(n)
        order = part[_np.argsort(d[part], kind="stable")]
        sel_idx = idx[order].tolist()
        sel_d = d[order].tolist()
        return [(sorted_codes[i], dv) for i, dv in zip(sel_idx, sel_d)]

    def find_by_prefixes_batch(
        self,
        geom_type: str,
        prefixes: Iterable[str],
        exclude_code: Optional[str] = None,
    ) -> Set[str]:
        """
        Local equivalent of UNWIND prefixes with STARTS WITH.

        @param geom_type Geometry-type filter key.
        @param prefixes Prefix strings.
        @param exclude_code Optional query code to omit, matching Cypher's exc parameter.
        @returns Matching code set.
        """
        codes = self._sorted.get(geom_type)
        if not codes:
            return set()
        result: Set[str] = set()
        for prefix in prefixes:
            lo = _bisect.bisect_left(codes, prefix)
            hi = _bisect.bisect_right(codes, prefix + '\uffff')
            result.update(codes[lo:hi])
        if exclude_code:
            result.discard(exclude_code.strip().upper())
        return result

    def n_codes(self, geom_type: str) -> int:
        return len(self._sorted.get(geom_type, []))

    def has_geom_type(self, geom_type: str) -> bool:
        return (geom_type in self._sorted
                and len(self._sorted[geom_type]) > 0)

    @classmethod
    def from_codes(
        cls,
        geom_type: str,
        codes: Iterable[str],
        verbose: bool = True,
    ) -> "POIPrefixCache":
        """
        Build a prefix cache from an existing code list without Neo4j access.

        @param geom_type Geometry-type key; Point for the paper's KNN experiment.
        @param codes Centroid codes; validate G0, deduplicate, and sort internally.
        @param verbose Whether to print count/timing.
        @returns Cache suitable for set_poi_prefix_index.
        """
        obj = cls.__new__(cls)
        obj._sorted = {}
        obj._centers = {}
        obj._code_idx = {}
        t0 = time.perf_counter()
        uniq = sorted({
            str(c).strip().upper()
            for c in codes
            if c is not None
            and is_valid_g0_code(str(c).strip().upper())
        })
        obj._sorted[geom_type] = uniq
        if verbose:
            print(f"  [poi_cache] {geom_type:<12s}: "
                  f"{len(uniq):>9,} codes  (from list, no Neo4j)")
        obj._precompute_centers(geom_type, uniq, verbose)
        obj.elapsed_sec = time.perf_counter() - t0
        if verbose:
            print(f"  [poi_cache] 完成: 共 {len(uniq):,} 个 codes, "
                  f"耗时 {obj.elapsed_sec * 1000:.0f} ms")
        return obj


# Module-level singleton injected by main() at setup; None falls back to Cypher
_poi_prefix_index: Optional[POIPrefixCache] = None


def set_poi_prefix_index(idx: Optional[POIPrefixCache]) -> None:
    """Inject POI prefix client index (None → cold-cache mode)."""
    global _poi_prefix_index
    _poi_prefix_index = idx


def get_poi_prefix_index() -> Optional[POIPrefixCache]:
    return _poi_prefix_index


# ============================================================================
# Helper functions
# ============================================================================

def pctile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def run_status_line(
    done: int,
    total: int,
    k: int,
    last_recall: float,
    mean_recall: float,
    t_gr_us: float,
    t_pgs_us: float,
    elapsed: float,
    last_len: List[int],
    bar_width: int = 22,
) -> None:
    """
    Single-line 
    refresh: progress + batch Recall + mean Recall + time.
    """
    pct = done / total if total else 0
    filled = int(bar_width * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
    rate = done / elapsed if elapsed > 0.001 else 0
    eta = (total - done) / rate if rate > 0 else 0
    try:
        tw = max(40, shutil.get_terminal_size(fallback=(100, 24)).columns - 1)
    except Exception:
        tw = 100
    line = (
        f"\r  [{bar}] {pct * 100:5.1f}%  {done:,}/{total:,}"
        f"  R@{k}={last_recall:.0%}  avg={mean_recall:.0%}"
        f"  GR={t_gr_us / 1000:.0f}ms  PGs={t_pgs_us / 1000:.0f}ms"
        f"  {rate:.1f}/s  ETA {eta:.0f}s"
    )
    pad = max(0, last_len[0] - len(line))
    last_len[0] = len(line)
    out = line + (" " * min(pad, max(0, tw - len(line) - 2)))
    if len(out) > tw:
        out = out[:tw]
    sys.stdout.write(out)
    sys.stdout.flush()


# ============================================================================
# Grid utils: safe level + Z-order 3×3 neighborhood
# ============================================================================

def _min_margin_m(
    point_lon: float, point_lat: float,
    bbox: Tuple[float, float, float, float],
) -> float:
    """Min distance from query point to the four cell borders (meters, approx)."""
    min_lon, max_lon, min_lat, max_lat = bbox
    cos_lat = math.cos(math.radians(point_lat))
    return min(
        (point_lat - min_lat) * DEG_TO_M,
        (max_lat - point_lat) * DEG_TO_M,
        (point_lon - min_lon) * DEG_TO_M * cos_lat,
        (max_lon - point_lon) * DEG_TO_M * cos_lat,
    )


def find_safe_level(code: str, min_margin_m: float = 100.0) -> int:
    """
    Search from the finest level toward coarser levels for the first cell whose minimum query-
    to-boundary margin is at least min_margin_m. This defines the local coverage cell for the
    requested margin.
    """
    code = code.strip().upper()
    max_level = len(code) - 2
    point = fast_code_to_center(code)

    for level in range(max_level, 0, -1):
        prefix = code[:level + 2]
        bbox = code_to_bbox(prefix)
        margin = _min_margin_m(point[0], point[1], bbox)
        if margin >= min_margin_m:
            return level
    return 1


def _prefix_to_rowcol(prefix: str) -> Tuple[int, int, int]:
    """GeoSOT prefix → (lat_val, lon_val, level)."""
    refine = prefix[2:]
    level = len(refine)
    lat_val = lon_val = 0
    for ch in refine:
        d = ord(ch) - 48
        lat_val = (lat_val << 1) | (d >> 1)
        lon_val = (lon_val << 1) | (d & 1)
    return lat_val, lon_val, level


def _rowcol_to_prefix(
    lat_val: int, lon_val: int, level: int, hemisphere: str = "0",
) -> str:
    """(lat_val, lon_val, level) → GeoSOT prefix."""
    digits = []
    for i in range(level - 1, -1, -1):
        lat_bit = (lat_val >> i) & 1
        lon_bit = (lon_val >> i) & 1
        digits.append(str(lat_bit * 2 + lon_bit))
    return "G" + hemisphere + "".join(digits)


def get_3x3_neighbors(prefix: str) -> List[str]:
    """Get 3×3 cell neighborhood (includes self; at most 9; fewer near borders)."""
    lat_val, lon_val, level = _prefix_to_rowcol(prefix)
    hemisphere = prefix[1]
    max_val = (1 << level) - 1
    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr = lat_val + dr
            nc = lon_val + dc
            if 0 <= nr <= max_val and 0 <= nc <= max_val:
                neighbors.append(
                    _rowcol_to_prefix(nr, nc, level, hemisphere))
    return neighbors


# ============================================================================
# Module-level geometry-rerank helpers (callable without a Searcher instance)
# ============================================================================

def _lcp_of_codes(codes: List[str]) -> str:
    """Longest common prefix of a list of GeoSOT codes."""
    if not codes:
        return ""
    prefix = codes[0]
    for c in codes[1:]:
        while not c.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _lcpc_bbox_from_vcodes(
    vcodes: List[str],
    prefix_code: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """Compute LCPC bbox from vertex codes (or a precomputed common prefix)."""
    if prefix_code and len(prefix_code) >= 3:
        return code_to_bbox(prefix_code)
    if vcodes:
        lcp = _lcp_of_codes(list(dict.fromkeys(vcodes)))
        if len(lcp) >= 3:
            return code_to_bbox(lcp)
        bboxes = [code_to_bbox(c) for c in vcodes]
        return (
            min(b[0] for b in bboxes), max(b[1] for b in bboxes),
            min(b[2] for b in bboxes), max(b[3] for b in bboxes),
        )
    return (0.0, 0.0, 0.0, 0.0)


def _bbox_lb_m(
    bq: Tuple[float, float, float, float],
    bc: Tuple[float, float, float, float],
) -> float:
    """
    Conservative lower bound distance (m) between two bboxes;
    bbox=(min_lon,max_lon,min_lat,max_lat).
    """
    lon_gap = max(0.0, max(bq[0], bc[0]) - min(bq[1], bc[1]))
    lat_gap = max(0.0, max(bq[2], bc[2]) - min(bq[3], bc[3]))
    if lon_gap == 0.0 and lat_gap == 0.0:
        return 0.0
    ref_lat = max(abs(bq[2]), abs(bq[3]), abs(bc[2]), abs(bc[3]))
    cos_lb = math.cos(math.radians(ref_lat))
    return math.sqrt((lat_gap * DEG_TO_M) ** 2
                     + (lon_gap * cos_lb * DEG_TO_M) ** 2)


def geometry_rerank_bb(
    q_vcodes: List[str],
    q_type: str,
    candidates: List[Tuple[str, List[str], str]],
    k: int,
    lcpc_cache: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> Tuple[List[Tuple[str, float]], int]:
    """
    Module-level LCPC-guided branch-and-bound geometry reranking, usable without a
    GeoSotKNNSearcherV3 instance when vertex codes are already available.

    @param q_vcodes Query vertex codes.
    @param q_type Query geometry type.
    @param candidates [(main_code, vertex_codes, geometry_type), ...].
    @param k Requested Top-K.
    @param lcpc_cache Shared LCPC bbox cache; None creates a temporary cache.
    @returns (distance-sorted [(main_code, dist_m), ...], n_evaluated).
    """
    if lcpc_cache is None:
        lcpc_cache = {}

    # Decode query coords & compute query LCPC bbox
    q_coords = [fast_code_to_center(c) for c in q_vcodes]
    q_bbox = _lcpc_bbox_from_vcodes(q_vcodes)

    # Compute lb for each candidate and sort
    items: List[Tuple[float, str, List[Tuple[float, float]], str]] = []
    for mc, vcodes, geo_type in candidates:
        c_coords = [fast_code_to_center(c) for c in vcodes]
        c_bbox = lcpc_cache.get(mc)
        if c_bbox is None:
            c_bbox = _lcpc_bbox_from_vcodes(vcodes)
            lcpc_cache[mc] = c_bbox
        lb = max(0.0, _bbox_lb_m(q_bbox, c_bbox) - _LCPC_SAFETY_M)
        items.append((lb, mc, c_coords, geo_type))
    items.sort(key=lambda x: x[0])

    q_lat_mean = (sum(c[1] for c in q_coords) / len(q_coords)
                  if q_coords else 0.0)
    _q_cos = math.cos(math.radians(q_lat_mean))
    _BATCH = _GEO_BATCH_SIZE if _HAS_NP else 1

    top_k_heap: List[Tuple[float, str]] = []
    computed: List[Tuple[str, float]] = []

    for batch_start in range(0, len(items), _BATCH):
        if (len(top_k_heap) >= k
                and items[batch_start][0] > -top_k_heap[0][0]):
            break
        batch = items[batch_start: batch_start + _BATCH]

        if _HAS_NP and q_type == "Polygon":
            import numpy as np
            try:
                q_arr = np.array(q_coords, dtype=np.float64)
                n_b = len(batch)
                max_nb = max(len(item[2]) for item in batch)
                B_batch = np.full((n_b, max_nb, 2), np.nan)
                for ii, (_, _, cc, _) in enumerate(batch):
                    nb_ = len(cc)
                    if nb_ > 0:
                        B_batch[ii, :nb_, :] = cc
                dlon = (q_arr[:, None, None, 0]
                        - B_batch[None, :, :, 0]) * _q_cos
                dlat = q_arr[:, None, None, 1] - B_batch[None, :, :, 1]
                sq_min_vv = np.nanmin(dlon ** 2 + dlat ** 2, axis=(0, 2))
                order = np.argsort(sq_min_vv).tolist()
            except Exception:
                order = list(range(len(batch)))
        else:
            order = list(range(len(batch)))

        for ii in order:
            lb, mc, c_coords, c_type_c = batch[ii]
            if len(top_k_heap) >= k and lb > -top_k_heap[0][0]:
                continue
            try:
                dist, _, _ = geometry_min_distance_m(
                    q_coords, q_type, c_coords, c_type_c, None)
            except Exception:
                dist = float("inf")
            computed.append((mc, dist))
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (-dist, mc))
            elif dist < -top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, (-dist, mc))

    computed.sort(key=lambda x: x[1])
    return computed, len(computed)


# ============================================================================
# KNN Searcher V3
# ============================================================================

class GeoSotKNNSearcherV3:
    """
    V3 KNN searcher: choose a safe level, query its prefix and 3x3 neighbors, then coarsen as
    needed. Coarse ranking uses cosine-weighted decoded centroids; optional edge/vertex
    reranking handles nonpoint geometry.
    """

    def __init__(self, session, geom_type: str = "Point",
                 verbose: bool = False):
        self._session = session
        self._geom_type = geom_type
        self._verbose = verbose
        # In-process VC + geometry-type cache (reused across queries; does not change algorithm results)
        self._feat_cache: Dict[str, Dict] = {}
        self._coord_cache: Dict[Tuple[str, ...], List[Tuple[float, float]]] = {}
        # LCPC bbox cache: main_code → (min_lon, max_lon, min_lat, max_lat)
        self._lcpc_bbox_cache: Dict[str, Tuple[float, float, float, float]] = {}
        self._geo_pool = None
        self._geo_pool_workers = 0

    # ---------- Neo4j queries ----------

    def _query_candidates(
        self, prefixes: Set[str], exclude_code: str,
    ) -> Set[str]:
        """
        Batch prefix retrieval filtered by geometry type. Prefer an injected POIPrefixCache for
        local in-memory matching; otherwise issue Cypher queries through Bolt. Cache materialization
        and raw traversal are distinct execution modes.
        """
        cache = get_poi_prefix_index()
        if cache is not None and cache.has_geom_type(self._geom_type):
            return cache.find_by_prefixes_batch(
                self._geom_type, prefixes, exclude_code=exclude_code)

        if self._session is None:
            raise RuntimeError(
                "POI prefix cache is empty and no Neo4j session is bound")

        cypher = (
            "UNWIND $prefixes AS prefix\n"
            "MATCH (c:GeoSOTCentroidCode)"
            "-[:HAS_GEOMETRY_TYPE]->"
            "(:GeoFeatureType {type_name: $gtype})\n"
            "WHERE c.main_code STARTS WITH prefix\n"
            "  AND c.main_code <> $exc\n"
            "RETURN DISTINCT c.main_code AS mc"
        )
        try:
            rows = self._session.run(
                cypher, prefixes=list(prefixes),
                exc=exclude_code, gtype=self._geom_type)
            return {str(r["mc"]) for r in rows}
        except Exception:
            return set()

    def _fetch_feature_info_from_db(
        self, codes: List[str],
    ) -> Dict[str, Dict]:
        """
        Fetch vertex codes and geometry types from Neo4j without a cache.

        @returns {main_code: {"geometry_type": ..., "vertex_codes": [...]}}.
        """
        if not codes:
            return {}
        if self._session is None:
            return {}
        try:
            rows = self._session.run(_FEATURE_INFO_CYPHER, codes=codes)
            result: Dict[str, Dict] = {}
            for r in rows:
                mc = str(r["mc"]).strip().upper()
                geo_type = r.get("geo_type") or "Point"
                verts = [str(v).strip().upper()
                         for v in (r.get("vertex_codes") or []) if v]
                prefix_code = r.get("prefix_code") or None
                if prefix_code:
                    prefix_code = str(prefix_code).strip().upper()
                result[mc] = {
                    "geometry_type": geo_type,
                    "vertex_codes": verts,
                    "prefix_code": prefix_code,
                }
            return result
        except Exception:
            return {}

    def _batch_feature_info(
        self, codes: List[str],
    ) -> Dict[str, Dict]:
        """
        Fetch vertex codes and geometry types with a process-local cache.

        @returns {main_code: {"geometry_type": ..., "vertex_codes": [...]}}.
        """
        if not codes:
            return {}
        norm = [str(c).strip().upper() for c in codes]
        result: Dict[str, Dict] = {}
        missing: List[str] = []
        for mc in norm:
            if mc in self._feat_cache:
                result[mc] = self._feat_cache[mc]
            elif mc not in missing:
                missing.append(mc)
        if missing:
            fetched = self._fetch_feature_info_from_db(missing)
            for mc, info in fetched.items():
                self._feat_cache[mc] = info
                result[mc] = info
            default = {"geometry_type": "Point", "vertex_codes": []}
            for mc in missing:
                if mc not in result:
                    self._feat_cache[mc] = default
                    result[mc] = default
        return result

    def prefetch_feature_info(
        self,
        codes,
        batch_size: int = 5000,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Prefetch vertex codes and geometry types into the process cache during setup, without
        changing KNN outputs.

        @param codes Iterable of main_codes.
        @param batch_size UNWIND batch size.
        @returns {n_total, n_hit, n_fetch, n_cached, elapsed_sec}.
        """
        norm = sorted({str(c).strip().upper() for c in codes if c})
        missing = [c for c in norm if c not in self._feat_cache]
        n_hit = len(norm) - len(missing)
        t0 = time.perf_counter()
        n_fetch = 0
        default = {"geometry_type": "Point", "vertex_codes": []}
        pbar = None
        if verbose and missing:
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    total=len(missing),
                    desc="feat_cache",
                    unit="code",
                    ncols=100,
                    dynamic_ncols=False,
                    leave=True,
                    mininterval=1.0,
                )
            except ImportError:
                pbar = None
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            fetched = self._fetch_feature_info_from_db(chunk)
            for mc, info in fetched.items():
                self._feat_cache[mc] = info
            n_fetch += len(fetched)
            for mc in chunk:
                if mc not in self._feat_cache:
                    self._feat_cache[mc] = default
            if pbar is not None:
                pbar.update(len(chunk))
            elif verbose:
                done = min(i + batch_size, len(missing))
                print(f"  [feat_cache] {done:,}/{len(missing):,} fetched")
        if pbar is not None:
            pbar.close()
        # Also warm LCPC bbox cache (code_to_bbox is pure math, ~3–5s)
        n_bbox = 0
        for mc, info in self._feat_cache.items():
            if mc not in self._lcpc_bbox_cache:
                vc = info.get("vertex_codes") or []
                pc = info.get("prefix_code") or None
                self._get_lcpc_bbox(mc, vc, pc)
                n_bbox += 1
        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  [feat_cache] done: {len(self._feat_cache):,} cached, "
                  f"{n_hit:,} already warm, {elapsed:.1f}s"
                  f"  (bbox prewarm: {n_bbox:,})")
        return {
            "n_total": float(len(norm)),
            "n_hit": float(n_hit),
            "n_fetch": float(n_fetch),
            "n_cached": float(len(self._feat_cache)),
            "elapsed_sec": elapsed,
        }

    def close(self) -> None:
        """Close engineering resources held by the searcher (e.g. geometry-rerank pool)."""
        if self._geo_pool is not None:
            self._geo_pool.shutdown(wait=True)
            self._geo_pool = None
            self._geo_pool_workers = 0

    def _get_geo_pool(self, workers: int):
        """Reuse the geometry-rerank process pool; avoid respawning per query on Windows."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        workers = min(max(1, workers), multiprocessing.cpu_count())
        if (self._geo_pool is None
                or self._geo_pool_workers != workers):
            self.close()
            self._geo_pool = ProcessPoolExecutor(max_workers=workers)
            self._geo_pool_workers = workers
        return self._geo_pool

    @staticmethod
    def _get_all_codes(mc: str, info: Dict) -> List[str]:
        """Get feature code list (Point uses main_code; others use vertex codes)."""
        if info.get("geometry_type") == "Point" or not info.get("vertex_codes"):
            return [mc.strip().upper()]
        return info["vertex_codes"]

    def _coords_for_codes(
        self, codes: List[str],
    ) -> List[Tuple[float, float]]:
        """
        Decode and cache deterministic grid-center coordinates for vertex codes. Candidate-specific
        cos_ref_matrix weighting is still applied later, preserving the reranking metric.
        """
        key = tuple(codes)
        cached = self._coord_cache.get(key)
        if cached is not None:
            return cached
        coords = [fast_code_to_center(c) for c in codes]
        self._coord_cache[key] = coords
        return coords

    @staticmethod
    def _lcp_of(codes: List[str]) -> str:
        """Longest common prefix (LCPC) of a GeoSOT code list."""
        if not codes:
            return ""
        prefix = codes[0]
        for c in codes[1:]:
            while not c.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix

    def _get_lcpc_bbox(
        self,
        main_code: str,
        codes: List[str],
        prefix_code: Optional[str] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Compute/cache LCPC bbox (min_lon,max_lon,min_lat,max_lat). Prefer the graph's precomputed
        prefix_code; otherwise derive an LCP from vertex codes. For prefixes shorter than three
        characters, use the union of vertex-cell boxes as the fallback extent.
        """
        cached = self._lcpc_bbox_cache.get(main_code)
        if cached is not None:
            return cached

        if prefix_code and len(prefix_code) >= 3:
            # Precomputed common prefix in the graph; most reliable
            bbox = code_to_bbox(prefix_code)
        elif codes:
            lcpc = self._lcp_of(list(dict.fromkeys(codes)))
            if len(lcpc) >= 3:
                bbox = code_to_bbox(lcpc)
            else:
                # LCPC too short: use union of vertex cells; always covers the whole polygon
                bboxes = [code_to_bbox(c) for c in codes]
                bbox = (
                    min(b[0] for b in bboxes),
                    max(b[1] for b in bboxes),
                    min(b[2] for b in bboxes),
                    max(b[3] for b in bboxes),
                )
        else:
            bbox = code_to_bbox(main_code)

        self._lcpc_bbox_cache[main_code] = bbox
        return bbox

    @staticmethod
    def _bbox_lb_dist_m(
        bq: Tuple[float, float, float, float],
        bc: Tuple[float, float, float, float],
    ) -> float:
        """
        Conservative inter-bbox distance lower bound in meters. Boxes are
        (min_lon,max_lon,min_lat,max_lat); use the smallest longitude cosine at the highest latitude
        to retain the lower-bound property.
        """
        lon_gap = max(0.0, max(bq[0], bc[0]) - min(bq[1], bc[1]))
        lat_gap = max(0.0, max(bq[2], bc[2]) - min(bq[3], bc[3]))
        if lon_gap == 0.0 and lat_gap == 0.0:
            return 0.0
        ref_lat = max(abs(bq[2]), abs(bq[3]), abs(bc[2]), abs(bc[3]))
        cos_lb = math.cos(math.radians(ref_lat))
        return math.sqrt((lat_gap * DEG_TO_M) ** 2
                         + (lon_gap * cos_lb * DEG_TO_M) ** 2)

    def _geometry_rerank(
        self,
        query_code: str,
        q_info: Dict,
        top_codes: List[str],
        c_raw: Dict[str, Dict],
        rank_list: List[Tuple[str, float]],
        k: int,
        geo_workers: int = 0,
    ) -> Tuple[List[Tuple[str, float]], int, Dict[str, float]]:
        """
        LCPC-guided branch-and-bound geometry reranking. Process candidates in increasing bbox
        lower-bound distance and stop when the next bound exceeds the current Kth geometry distance.

        @param k Top-K for the stopping condition.
        @param geo_workers Worker interface; sufficiently large candidate sets can use parallel
            processing.
        @returns (ascending [(code,distance_m), ...], n_evaluated, timing), with t_lcpc_prune_ms,
            t_exact_ms, and t_decode_ms.
        """
        _t = time.perf_counter
        t_decode = 0.0

        q_type = q_info["geometry_type"]
        q_codes = self._get_all_codes(query_code, q_info)
        _td0 = _t()
        q_coords = self._coords_for_codes(q_codes)
        t_decode += (_t() - _td0) * 1000
        q_bbox = self._get_lcpc_bbox(
            query_code, q_codes, q_info.get("prefix_code"))

        sq_map = {mc: sq for mc, sq in rank_list}
        default_ci = {"geometry_type": "Point", "vertex_codes": []}

        _t_prune0 = _t()

        # Compute LCPC bbox lower bound per candidate (code space only; no vertex decode).
        # Sort key = (lb, centroid_dist): lb primary; centroid as zero-decode secondary key,
        # replacing old NumPy reordering that decoded all vertices.
        items: List[Tuple[float, float, str, List[str], str]] = []
        for mc in top_codes:
            ci = c_raw.get(mc) or default_ci
            c_codes = self._get_all_codes(mc, ci)
            c_type = ci.get("geometry_type", "Point")
            c_bbox = self._get_lcpc_bbox(
                mc, c_codes, ci.get("prefix_code"))
            lb = max(0.0,
                     self._bbox_lb_dist_m(q_bbox, c_bbox) - _LCPC_SAFETY_M)
            items.append((lb, sq_map.get(mc, float("inf")),
                          mc, c_codes, c_type))
        items.sort(key=lambda x: (x[0], x[1]))

        # LCPC prune prep timing (pure codes/grid; no decode)
        t_lcpc_prune = (_t() - _t_prune0) * 1000

        top_k_heap: List[Tuple[float, str]] = []  # max-heap: (-dist, mc)
        computed: List[Tuple[str, float]] = []

        # branch-and-bound + lazy decode: candidates sorted by (lb, centroid); once lb exceeds
        # current K-th distance, all later are larger — break. Only candidates that pass lb
        # and need exact distance get coords decoded (lazy decode); pruned ones stay undecoded.
        _decode_before = t_decode
        _t_exact0 = _t()
        for lb, fallback_sq, mc, c_codes, c_type_c in items:
            if len(top_k_heap) >= k and lb > -top_k_heap[0][0]:
                break  # Lower bounds are sorted; all subsequent lower bounds are at least this large.

            _td0 = _t()
            c_coords = self._coords_for_codes(c_codes)
            t_decode += (_t() - _td0) * 1000

            try:
                dist, _, _ = geometry_min_distance_m(
                    q_coords, q_type, c_coords, c_type_c, None)
            except Exception:
                dist = math.sqrt(fallback_sq) * DEG_TO_M
            computed.append((mc, dist))
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (-dist, mc))
            elif dist < -top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, (-dist, mc))

        # Exact-distance loop timing (coord space): loop wall clock minus in-loop decode time
        t_exact = (_t() - _t_exact0) * 1000 - (t_decode - _decode_before)

        computed.sort(key=lambda x: x[1])
        timing = {
            "t_lcpc_prune_ms": round(max(0.0, t_lcpc_prune), 3),
            "t_exact_ms": round(t_exact, 3),
            "t_decode_ms": round(t_decode, 3),
        }
        return computed, len(computed), timing

    # ---------- KNN search ----------

    def knn_search(
        self,
        query_code: str,
        k: int = 5,
        min_margin_m: float = 100.0,
        min_level: int = 8,
        coarse_top_n: int = 50,
        geometry_scoring: bool = True,
        geo_workers: int = 0,
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        """
        Run legacy V3 KNN.

        @param query_code Query GeoSOT code.
        @param k Requested Top-K.
        @param min_margin_m Safe-level minimum boundary margin in meters.
        @param min_level Coarsest search level.
        @param coarse_top_n Maximum coarse candidates for geometry reranking.
        @param geometry_scoring Enable geometry-type lookup and edge/vertex reranking for nonpoints.
        @returns ([(code, approximate_distance_m), ...], info).
        """
        _t = time.perf_counter
        query_code = query_code.strip().upper()
        query_center = fast_code_to_center(query_code)

        # ---- 1. Determine safe level ----
        t0 = _t()
        safe_level = find_safe_level(query_code, min_margin_m)
        t_level = (_t() - t0) * 1000

        # ---- 2. Progressive candidate search + boundary check ----
        t0 = _t()
        cos_qlat = math.cos(math.radians(query_center[1]))
        cos2 = cos_qlat * cos_qlat

        candidates: Set[str] = set()
        level = safe_level
        n_prefixes = 0

        # Step A: always start from 3×3 (covers at least margin + one cell width)
        prefix = query_code[:level + 2]
        neighbors = get_3x3_neighbors(prefix)
        cands = self._query_candidates(set(neighbors), query_code)
        candidates.update(cands)
        n_prefixes += len(neighbors)

        # Step B: rank → boundary check → expand as needed
        # If K-th candidate distance ≥ 70% of search boundary, true Top-K may be truncated;
        # expand toward a coarser level.
        max_retries = 6

        def _rank_candidates() -> List[Tuple[str, float]]:
            ranked: List[Tuple[str, float]] = []
            for mc in candidates:
                try:
                    cc = fast_code_to_center(mc)
                    dlon = cc[0] - query_center[0]
                    dlat = cc[1] - query_center[1]
                    ranked.append((mc, dlon * dlon * cos2 + dlat * dlat))
                except Exception:
                    pass
            ranked.sort(key=lambda x: x[1])
            return ranked

        def _coverage_ok(ranked: List[Tuple[str, float]]) -> bool:
            if len(ranked) < k:
                return False
            kth_m = math.sqrt(ranked[k - 1][1]) * DEG_TO_M
            cell_margin = _min_margin_m(
                query_center[0], query_center[1],
                code_to_bbox(query_code[:level + 2]))
            grid_step = GRID_SIZES.get(level, (0.0, 0.0))
            grid_m = grid_step[0] * DEG_TO_M * cos_qlat
            coverage_m = cell_margin + grid_m
            return kth_m < coverage_m * 0.7

        rank_list = _rank_candidates()
        retries = 0

        while not _coverage_ok(rank_list) and level > min_level \
                and retries < max_retries:
            level -= 1
            prefix = query_code[:level + 2]
            neighbors = get_3x3_neighbors(prefix)
            new_cands = (
                self._query_candidates(set(neighbors), query_code)
                - candidates)
            candidates.update(new_cands)
            n_prefixes += len(neighbors)
            if new_cands:
                rank_list = _rank_candidates()
            retries += 1

        t_search = (_t() - t0) * 1000
        final_level = level

        if not rank_list:
            return [], {
                "safe_level": safe_level, "final_level": final_level,
                "total_candidates": 0, "prefixes": n_prefixes,
                "t_level_ms": round(t_level, 3),
                "t_search_ms": round(t_search, 3),
                "t_rank_ms": 0,
                "t_total_ms": round(t_level + t_search, 3),
            }

        # ---- 3. Ranking finished in search; extract timings ----
        t_rank = 0.0  # Included in t_search.

        # ---- 4. (optional) geometry rerank: only when non-Point types exist ----
        t_geo = 0.0
        final_scored = rank_list

        if geometry_scoring:
            t0 = _t()
            top_codes = [mc for mc, _ in rank_list[:coarse_top_n]]

            q_raw = self._batch_feature_info([query_code])
            q_info = q_raw.get(query_code) or {
                "geometry_type": "Point", "vertex_codes": [],
            }
            q_type = q_info["geometry_type"]
            c_raw = self._batch_feature_info(top_codes)

            need_geo = (
                q_type != "Point" or
                any((c_raw.get(m) or {}).get("geometry_type", "Point")
                    != "Point" for m in top_codes))

            if need_geo:
                geo_scored, n_geo_eval, geo_timing = self._geometry_rerank(
                    query_code, q_info, top_codes, c_raw, rank_list,
                    k=k, geo_workers=geo_workers)
                t_geo = (_t() - t0) * 1000

                t_total = t_level + t_search + t_rank + t_geo
                return geo_scored[:k], {
                    "safe_level": safe_level,
                    "final_level": final_level,
                    "total_candidates": len(candidates),
                    "coarse_top_n": len(top_codes),
                    "geo_evaluated": n_geo_eval,
                    "prefixes": n_prefixes,
                    "has_geometry": True,
                    "t_level_ms": round(t_level, 3),
                    "t_search_ms": round(t_search, 3),
                    "t_rank_ms": round(t_rank + t_geo, 3),
                    "t_geo_ms": round(t_geo, 3),
                    "t_lcpc_prune_ms": geo_timing["t_lcpc_prune_ms"],
                    "t_exact_ms": geo_timing["t_exact_ms"],
                    "t_decode_ms": geo_timing["t_decode_ms"],
                    "t_total_ms": round(t_total, 3),
                }

            t_geo = (_t() - t0) * 1000

        # ---- output: sqrt + equirectangular approx → approx meters ----
        results = [(mc, math.sqrt(sq) * DEG_TO_M)
                   for mc, sq in final_scored[:k]]

        t_total = t_level + t_search + t_rank + t_geo
        info = {
            "safe_level": safe_level,
            "final_level": final_level,
            "total_candidates": len(candidates),
            "prefixes": n_prefixes,
            "t_level_ms": round(t_level, 3),
            "t_search_ms": round(t_search, 3),
            "t_rank_ms": round(t_rank + t_geo, 3),
            "t_total_ms": round(t_total, 3),
        }
        return results, info


# ============================================================================
# PostGIS KNN
# ============================================================================

def postgis_knn(
    pg_conn, query_code: str, k: int,
    oversample: int = 8,
) -> List[Tuple[str, float]]:
    """
    PostGIS KNN with DISTINCT ON deduplication and geography reranking. Returns [(main_code,
    distance_m), ...]. Like postgis_knn_standard, the inner query limits KNN candidates before
    deduplication; only the outer deduplication mechanism differs.
    """
    cur = pg_conn.cursor()
    # Fetch ref-point geometry (EWKB with SRID) as a literal param so <-> right operand is constant
    cur.execute(
        "SELECT ST_AsEWKB(geometry) FROM public.poi_new "
        "WHERE main_code = %s LIMIT 1",
        (query_code,))
    row = cur.fetchone()
    if not row or row[0] is None:
        cur.close()
        return []
    q_geom = bytes(row[0])
    fetch_n = k * oversample

    cur.execute(
        "SELECT sub.main_code, sub.dist_m FROM ("
        "  SELECT DISTINCT ON (main_code) "
        "    main_code, "
        "    ST_Distance(geometry::geography, %s::geometry::geography)"
        "      AS dist_m, "
        "    sort_dist "
        "  FROM ("
        "    SELECT main_code, geometry, "
        "      geometry <-> %s::geometry AS sort_dist "
        "    FROM public.poi_new "
        "    WHERE main_code <> %s "
        "    ORDER BY geometry <-> %s::geometry "
        "    LIMIT %s"
        "  ) knn "
        "  ORDER BY main_code, sort_dist "
        ") sub ORDER BY sub.dist_m LIMIT %s",
        (q_geom, q_geom, query_code, q_geom, fetch_n, k))
    results = [(str(r[0]).strip().upper(), float(r[1]))
               for r in cur.fetchall()]
    cur.close()
    return results


def postgis_knn_standard(
    pg_conn, query_code: str, k: int,
    oversample: int = 8,
) -> List[Tuple[str, float]]:
    """
    PostGIS GiST KNN with bounded overfetch and deduplication. Keep ORDER BY geometry <->
    reference LIMIT k*oversample in the innermost query without windows or DISTINCT, allowing
    index early termination. Apply ROW_NUMBER deduplication and geography reranking only to
    those candidates. Combining the window and LIMIT at the same level previously forced large
    scans/sorts. Bind the reference as a constant %s::geometry parameter to support the KNN
    index plan. Finite overfetch remains a reference protocol, not an exhaustive accuracy
    guarantee.
    """
    cur = pg_conn.cursor()
    # Fetch ref-point geometry (EWKB with SRID) as a literal param
    cur.execute(
        "SELECT ST_AsEWKB(geometry) FROM public.poi_new "
        "WHERE main_code = %s LIMIT 1",
        (query_code,))
    row = cur.fetchone()
    if not row or row[0] is None:
        cur.close()
        return []
    q_geom = bytes(row[0])
    fetch_n = k * oversample

    cur.execute(
        "SELECT main_code, dist_m FROM ("
        "  SELECT main_code, "
        "    ST_Distance(geometry::geography, %s::geometry::geography)"
        "      AS dist_m, "
        "    ROW_NUMBER() OVER ("
        "      PARTITION BY main_code ORDER BY sort_dist"
        "    ) AS rn "
        "  FROM ("
        "    SELECT main_code, geometry, "
        "      geometry <-> %s::geometry AS sort_dist "
        "    FROM public.poi_new "
        "    WHERE main_code <> %s "
        "    ORDER BY geometry <-> %s::geometry "
        "    LIMIT %s"
        "  ) knn "
        ") sub WHERE rn = 1 "
        "ORDER BY dist_m LIMIT %s",
        (q_geom, q_geom, query_code, q_geom, fetch_n, k))
    results = [(str(r[0]).strip().upper(), float(r[1]))
               for r in cur.fetchall()]
    cur.close()
    return results


# ============================================================================
# Main flow
# ============================================================================

def main():
    num_queries = 100
    k_list: List[int] = [5]
    seed = 42
    margin = 100.0
    min_level = 8
    coarse_top_n = 50
    geom_type = "Point"
    output_csv = "knn_v3_benchmark_results.csv"
    save_csv = True
    use_geometry = True
    verbose_detail = False
    run_pg_dedup = False

    i = 0
    args = sys.argv[1:]
    while i < len(args):
        if args[i] == "--queries" and i + 1 < len(args):
            num_queries = int(args[i + 1]); i += 2
        elif args[i] == "--k" and i + 1 < len(args):
            k_list = sorted(set(int(x) for x in args[i + 1].split(",")))
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1]); i += 2
        elif args[i] == "--margin" and i + 1 < len(args):
            margin = float(args[i + 1]); i += 2
        elif args[i] == "--min-level" and i + 1 < len(args):
            min_level = int(args[i + 1]); i += 2
        elif args[i] == "--coarse-top" and i + 1 < len(args):
            coarse_top_n = int(args[i + 1]); i += 2
        elif args[i] == "--geom-type" and i + 1 < len(args):
            geom_type = args[i + 1]; i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_csv = args[i + 1]; i += 2
        elif args[i] == "--no-csv":
            save_csv = False; i += 1
        elif args[i] == "--no-geometry":
            use_geometry = False; i += 1
        elif args[i] in ("--verbose", "-v"):
            verbose_detail = True; i += 1
        elif args[i] == "--pg-dedup":
            run_pg_dedup = True; i += 1
        else:
            i += 1

    k_max = max(k_list)
    k_str = ",".join(str(x) for x in k_list)

    random.seed(seed)

    W = 76
    print("=" * W)
    print("GeoSOT KNN V3: Adaptive Grid Pruning + Geometry Scoring")
    print("=" * W)
    print(f"  Queries: {num_queries},  K: {k_str},  Seed: {seed}")
    print(f"  Margin: {margin:.0f} m,  Min level: {min_level}")
    print(f"  Geometry scoring: {'ON' if use_geometry else 'OFF'}"
          f"  (coarse_top_n: {coarse_top_n})")
    console_mode = ("per-query detail (--verbose)" if verbose_detail
                    else "single-line progress (add --verbose for detail)")
    print(f"  Console: {console_mode}")
    print(
        "  PostGIS:  benchmark = Standard "
        "(GiST <-> KNN + ROW_NUMBER 按 main_code 去重);  "
        "慢速 DISTINCT_ON 对照查询: "
        f"{'ON (--pg-dedup)' if run_pg_dedup else 'OFF'}"
    )

    # ================================================================
    # 1. Load POI codes from PostGIS
    # ================================================================
    print(f"\n[1/4] Loading POI codes from PostGIS ...")
    t0 = time.perf_counter()
    pg_conn = psycopg2.connect(**PG_CONFIG)
    cur = pg_conn.cursor()
    cur.execute("SELECT main_code FROM public.poi_new")
    all_codes = []
    for row in cur.fetchall():
        mc = str(row[0]).strip().upper() if row[0] else ""
        if is_valid_g0_code(mc):
            all_codes.append(mc)
    cur.close()
    print(f"  Valid POIs: {len(all_codes):,}  "
          f"({(time.perf_counter() - t0) * 1000:.0f} ms)")

    if len(all_codes) < k_max + 1:
        print("[ERROR] Not enough POIs")
        pg_conn.close()
        return

    # ================================================================
    # 2. Sample query points
    # ================================================================
    query_codes = random.sample(all_codes, min(num_queries, len(all_codes)))
    print(f"  Sampled {len(query_codes):,} query points")

    # ================================================================
    # 3. Connect Neo4j + warmup
    # ================================================================
    print(f"\n[2/4] Connecting to Neo4j ...")
    t0 = time.perf_counter()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    session = driver.session(database=NEO4J_DATABASE)
    session.run("RETURN 1").consume()

    searcher = GeoSotKNNSearcherV3(session, geom_type=geom_type,
                                    verbose=False)
    print(f"  Connected ({(time.perf_counter() - t0) * 1000:.0f} ms)"
          f"  geom_type filter: {geom_type}")

    # ---- fairness setup: PG warmup + dual-side throwaway ----
    # Benchmark protocol: time both G3R and PostGIS with warm data/plan caches,
    # consistent with G3R's lru_cache and the Neo4j page cache. Report setup separately.
    from bench_setup import prepare_pg_warm
    prepare_pg_warm(
        pg_conn,
        tables=["public.poi_new"],
        throwaway_queries=[
            ("SELECT main_code FROM public.poi_new LIMIT 200", None),
        ],
    )

    # GR+PG dual throwaway: warm both sides with 5 dummy queries; discard results
    n_throwaway = min(5, len(query_codes))
    if n_throwaway > 0:
        print(f"  Throwaway: GR/PG 双侧丢弃前 {n_throwaway} 条 query")
        for tw_qc in query_codes[:n_throwaway]:
            try:
                searcher.knn_search(
                    tw_qc, k=k_max, min_margin_m=margin,
                    min_level=min_level, coarse_top_n=coarse_top_n,
                    geometry_scoring=use_geometry)
            except Exception:
                pass
            try:
                postgis_knn_standard(pg_conn, tw_qc, k_max)
            except Exception:
                pass
    print("  Warmup done")

    # ================================================================
    # 4. Run queries (supports resume: query_codes already in CSV are skipped)
    # ================================================================

    CSV_FIELDS = [
        "query_code", "k", "recall_at_k", "top1_match", "pg_std_agree",
        "pgs_top1_dist_m", "gr_top1_dist_m", "top1_err_m", "top1_err_pct",
        "gr_total_candidates", "gr_safe_level", "gr_final_level",
        "gr_prefixes", "t_gr_us", "t_pg_us", "t_pgs_us",
        "gr_top_k", "pgs_top_k",
    ]

    done_keys: set = set()
    rows_out: List[Dict] = []
    csv_file = None
    csv_writer = None

    if save_csv:
        if os.path.exists(output_csv):
            with open(output_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_keys.add((row["query_code"], int(row["k"])))
                    rows_out.append(row)
            csv_file = open(output_csv, "a", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        else:
            csv_file = open(output_csv, "w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            csv_file.flush()

    done_qcodes = {qc for qc, _ in done_keys}
    all_k_done = {qc for qc in done_qcodes
                  if all((qc, kv) in done_keys for kv in k_list)}
    remaining = [qc for qc in query_codes if qc not in all_k_done]
    skipped = len(query_codes) - len(remaining)
    if skipped > 0:
        print(f"\n  Resuming: {skipped:,} queries already in {output_csv}, "
              f"{len(remaining):,} remaining")

    print(f"\n[3/4] Running {len(remaining):,} queries "
          f"(K={k_str}, query with k_max={k_max}) ...")

    per_k_stats: Dict[int, Dict[str, List]] = {}
    for kv in k_list:
        per_k_stats[kv] = {
            "recall": [], "t_gr": [], "t_pg": [], "t_pgs": [],
            "top1_err": [], "top1_pct": [], "cand": [],
            "safe_lv": [], "final_lv": [], "prefixes": [],
            "pg_agree": [], "rows": [],
        }

    for prev in rows_out:
        kv = int(prev["k"])
        if kv not in per_k_stats:
            continue
        s = per_k_stats[kv]
        s["recall"].append(float(prev["recall_at_k"]))
        s["t_gr"].append(float(prev["t_gr_us"]))
        s["t_pgs"].append(float(prev["t_pgs_us"]))
        s["top1_err"].append(float(prev["top1_err_m"]))
        s["top1_pct"].append(float(prev["top1_err_pct"]))
        s["cand"].append(int(prev["gr_total_candidates"]))
        s["safe_lv"].append(int(prev["gr_safe_level"]))
        s["final_lv"].append(int(prev["gr_final_level"]))
        s["prefixes"].append(int(prev["gr_prefixes"]))
        s["pg_agree"].append(int(prev["pg_std_agree"]))
        s["rows"].append(prev)
        pg_us = prev.get("t_pg_us", "")
        if pg_us not in ("", None):
            s["t_pg"].append(float(pg_us))

    total_n = len(query_codes)
    t_start = time.perf_counter()
    status_pad = [0]

    for r_idx, qc in enumerate(remaining):
        idx = skipped + r_idx
        # --- Graph V3 KNN (query once at k_max) ---
        tg0 = time.perf_counter()
        gr_results, gr_info = searcher.knn_search(
            qc, k=k_max, min_margin_m=margin, min_level=min_level,
            coarse_top_n=coarse_top_n, geometry_scoring=use_geometry)
        tg1 = time.perf_counter()
        t_gr_us = (tg1 - tg0) * 1e6

        # --- PostGIS KNN Standard (query once at k_max) ---
        tps0 = time.perf_counter()
        pgs_results = postgis_knn_standard(pg_conn, qc, k_max)
        tps1 = time.perf_counter()
        t_pgs_us = (tps1 - tps0) * 1e6

        if not pgs_results:
            continue

        t_pg_us = 0.0
        pg_dedup_results = []
        if run_pg_dedup:
            tp0 = time.perf_counter()
            pg_dedup_results = postgis_knn(pg_conn, qc, k_max)
            tp1 = time.perf_counter()
            t_pg_us = (tp1 - tp0) * 1e6
            if not pg_dedup_results:
                continue

        # --- Evaluate and record for each K ---
        last_recall = 0.0
        for kv in k_list:
            if (qc, kv) in done_keys:
                continue

            pgs_set = {r[0] for r in pgs_results[:kv]}
            gr_set = {r[0] for r in gr_results[:kv]}
            recall = len(pgs_set & gr_set) / kv if kv > 0 else 0.0
            matched = pgs_set & gr_set
            pg_only = pgs_set - gr_set
            gr_only = gr_set - pgs_set

            pg_top1_dist = pgs_results[0][1] if pgs_results else 0
            gr_top1_dist = gr_results[0][1] if gr_results else 0
            top1_err = abs(gr_top1_dist - pg_top1_dist)
            top1_p = ((top1_err / pg_top1_dist * 100)
                      if pg_top1_dist > 0.01 else 0)
            top1_match = 1 if (gr_results and pgs_results and
                               gr_results[0][0] == pgs_results[0][0]) else 0

            std_agree_val = -1
            if run_pg_dedup and pg_dedup_results:
                pg_set_d = {r[0] for r in pg_dedup_results[:kv]}
                std_agree_val = 1 if pg_set_d == pgs_set else 0

            s = per_k_stats[kv]
            s["recall"].append(recall)
            s["t_gr"].append(t_gr_us)
            s["t_pgs"].append(t_pgs_us)
            s["top1_err"].append(top1_err)
            s["top1_pct"].append(top1_p)
            s["cand"].append(gr_info.get("total_candidates", 0))
            s["safe_lv"].append(gr_info.get("safe_level", 0))
            s["final_lv"].append(gr_info.get("final_level", 0))
            s["prefixes"].append(gr_info.get("prefixes", 0))
            s["pg_agree"].append(std_agree_val)
            if run_pg_dedup:
                s["t_pg"].append(t_pg_us)

            row_dict = {
                "query_code": qc,
                "k": kv,
                "recall_at_k": round(recall, 4),
                "top1_match": top1_match,
                "pg_std_agree": std_agree_val,
                "pgs_top1_dist_m": round(pg_top1_dist, 4),
                "gr_top1_dist_m": round(gr_top1_dist, 4),
                "top1_err_m": round(top1_err, 6),
                "top1_err_pct": round(top1_p, 6),
                "gr_total_candidates": gr_info.get("total_candidates", 0),
                "gr_safe_level": gr_info.get("safe_level", 0),
                "gr_final_level": gr_info.get("final_level", 0),
                "gr_prefixes": gr_info.get("prefixes", 0),
                "t_gr_us": round(t_gr_us, 1),
                "t_pg_us": (round(t_pg_us, 1) if run_pg_dedup else ""),
                "t_pgs_us": round(t_pgs_us, 1),
                "gr_top_k": ";".join(r[0] for r in gr_results[:kv]),
                "pgs_top_k": ";".join(r[0] for r in pgs_results[:kv]),
            }
            rows_out.append(row_dict)
            s["rows"].append(row_dict)
            if csv_writer is not None:
                csv_writer.writerow(row_dict)
                csv_file.flush()

            last_recall = recall

        # --- Status line (recall at k_max) ---
        kmax_s = per_k_stats[k_max]
        mean_r = (statistics.mean(kmax_s["recall"])
                  if kmax_s["recall"] else 0)
        run_status_line(
            idx + 1, total_n, k_max, last_recall, mean_r,
            t_gr_us, t_pgs_us, time.perf_counter() - t_start, status_pad)

        if verbose_detail:
            sys.stdout.write("\n")
            print(f"  -- Query {idx + 1}/{total_n}: {qc}")
            print(f"     safe_level={gr_info.get('safe_level')}, "
                  f"final_level={gr_info.get('final_level')}, "
                  f"candidates={gr_info.get('total_candidates')}, "
                  f"prefixes={gr_info.get('prefixes')}")
            for kv in k_list:
                pgs_set_v = {r[0] for r in pgs_results[:kv]}
                gr_set_v = {r[0] for r in gr_results[:kv]}
                rv = len(pgs_set_v & gr_set_v) / kv if kv > 0 else 0
                print(f"     Recall@{kv} = {rv:.0%}")
            print(f"     Graph V3: {t_gr_us / 1000:.1f} ms  "
                  f"(search {gr_info.get('t_search_ms', 0):.1f}, "
                  f"rank {gr_info.get('t_rank_ms', 0):.1f})")
            if run_pg_dedup:
                print(f"     PostGIS (dedup):    {t_pg_us / 1000:.1f} ms")
            print(f"     PostGIS (standard): {t_pgs_us / 1000:.1f} ms")

            max_rows = max(len(pgs_results[:k_max]),
                           len(gr_results[:k_max]))
            for ri in range(max_rows):
                pc = pgs_results[ri][0] if ri < len(pgs_results) else ""
                gc = gr_results[ri][0] if ri < len(gr_results) else ""
                pd = pgs_results[ri][1] if ri < len(pgs_results) else 0
                gd = gr_results[ri][1] if ri < len(gr_results) else 0
                pgs_set_max = {r[0] for r in pgs_results[:k_max]}
                flag = ("OK" if pc == gc
                        else ("~" if gc in pgs_set_max else "MISS"))
                print(f"     #{ri + 1:<3d} {flag:>4s}  "
                      f"PG: {pd:>9.2f}m  GR: {gd:>9.2f}m")

    if total_n > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if csv_file is not None:
        csv_file.close()

    session.close()
    driver.close()
    pg_conn.close()

    elapsed = time.perf_counter() - t_start
    n_rows = len(rows_out)
    new_n = len(remaining)
    print(f"\n  Done: {new_n:,} new queries + {skipped:,} resumed, "
          f"{n_rows:,} rows total, {elapsed:.1f}s")

    if save_csv:
        print(f"  Results incrementally saved to: {output_csv}")

    # ================================================================
    # Summary (per K)
    # ================================================================
    for kv in k_list:
        s = per_k_stats[kv]
        n = len(s["recall"])
        if n == 0:
            continue
        print_summary(
            W, kv, n,
            s["recall"], s["t_gr"], s["t_pg"], s["t_pgs"],
            s["top1_err"], s["top1_pct"], s["cand"],
            s["safe_lv"], s["final_lv"], s["prefixes"],
            s["pg_agree"], s["rows"], use_geometry, run_pg_dedup)

    print()


# ============================================================================
# Statistics output
# ============================================================================

def print_summary(W, k, n, recall_list, t_gr, t_pg, t_pgs,
                  top1_err, top1_pct, cand_counts,
                  safe_levels, final_levels, prefix_counts,
                  pg_agree, rows, use_geometry, run_pg_dedup: bool):
    SEP = f"  {'\u2500' * (W - 4)}"

    scoring = "Planar+Geometry" if use_geometry else "Planar-only"
    print(f"\n{'=' * W}")
    print(f"KNN V3 Benchmark Results  (K={k}, N={n:,}, scoring={scoring})")
    print(f"{'=' * W}")

    # --- Recall (vs PostGIS Standard) ---
    mean_recall = statistics.mean(recall_list)
    recall_1 = sum(1 for r in recall_list if r >= 1.0) / n * 100
    top1_match_rate = sum(1 for r in rows if r["top1_match"] == 1) / n * 100

    print(f"\n  Recall & Accuracy  (baseline: PostGIS Standard KNN):")
    print(SEP)
    print(f"    Mean Recall@{k}:      {mean_recall:.4f}  "
          f"({mean_recall * 100:.1f}%)")
    print(f"    Perfect Recall:      "
          f"{sum(1 for r in recall_list if r >= 1.0):,}/{n}  "
          f"({recall_1:.1f}%)")
    print(f"    Top-1 Match Rate:    {top1_match_rate:.1f}%")
    print(f"    Recall Distribution: "
          f"P25={pctile(recall_list, 25):.2f}  "
          f"P50={pctile(recall_list, 50):.2f}  "
          f"P75={pctile(recall_list, 75):.2f}  "
          f"P99={pctile(recall_list, 99):.2f}")

    # --- PostGIS dedup vs standard agreement (--pg-dedup only) ---
    if run_pg_dedup:
        agree_n = sum(1 for a in pg_agree if a == 1)
        print(f"\n  PostGIS Dedup vs Standard Agreement:")
        print(SEP)
        print(f"    Identical Top-K:     "
              f"{agree_n}/{n}  ({agree_n / n * 100:.1f}%)")

    # --- Top-1 distance error ---
    print(f"\n  Top-1 Distance Error  (vs PostGIS Standard):")
    print(SEP)
    print(f"    Mean |Error|:   {statistics.mean(top1_err):>12.4f} m")
    print(f"    Median |Error|: {statistics.median(top1_err):>12.4f} m")
    print(f"    P99 |Error|:    {pctile(top1_err, 99):>12.4f} m")
    print(f"    Max |Error|:    {max(top1_err):>12.4f} m")
    print(f"    Mean |Error%|:  {statistics.mean(top1_pct):>12.6f} %")

    # --- Timing ---
    total_gr = sum(t_gr)
    total_pgs = sum(t_pgs)

    print(f"\n  Query Timing:")
    print(SEP)
    if run_pg_dedup and t_pg:
        total_pg = sum(t_pg)
        hdr = (f"  {'':22s}  {'Graph V3':>12s}"
               f"  {'PG Standard':>12s}  {'PG Dedup':>12s}")
        print(hdr)
        print(SEP)

        def _row3(label, gr_v, pgs_v, pg_v):
            print(f"  {label:22s}  {gr_v:>12.1f}  {pgs_v:>12.1f}  {pg_v:>12.1f}")

        _row3("Total (ms)",
              total_gr / 1000, total_pgs / 1000, total_pg / 1000)
        _row3("Mean per-query (ms)",
              statistics.mean(t_gr) / 1000,
              statistics.mean(t_pgs) / 1000,
              statistics.mean(t_pg) / 1000)
        _row3("Median per-query (ms)",
              statistics.median(t_gr) / 1000,
              statistics.median(t_pgs) / 1000,
              statistics.median(t_pg) / 1000)
        _row3("P99 per-query (ms)",
              pctile(t_gr, 99) / 1000,
              pctile(t_pgs, 99) / 1000,
              pctile(t_pg, 99) / 1000)
        _row3("Max per-query (ms)",
              max(t_gr) / 1000,
              max(t_pgs) / 1000,
              max(t_pg) / 1000)

        speedup_std = (statistics.mean(t_pgs) / statistics.mean(t_gr)
                       if statistics.mean(t_gr) > 0 else 0)
        speedup_ded = (statistics.mean(t_pg) / statistics.mean(t_gr)
                       if statistics.mean(t_gr) > 0 else 0)
        print(f"\n  Speedup vs Graph:  "
              f"PG Standard {speedup_std:.1f}x,  PG Dedup {speedup_ded:.1f}x")
    else:
        hdr = (f"  {'':22s}  {'Graph V3':>12s}"
               f"  {'PG Standard':>12s}")
        print(hdr)
        print(SEP)

        def _row2(label, gr_v, pgs_v):
            print(f"  {label:22s}  {gr_v:>12.1f}  {pgs_v:>12.1f}")

        _row2("Total (ms)", total_gr / 1000, total_pgs / 1000)
        _row2("Mean per-query (ms)",
              statistics.mean(t_gr) / 1000,
              statistics.mean(t_pgs) / 1000)
        _row2("Median per-query (ms)",
              statistics.median(t_gr) / 1000,
              statistics.median(t_pgs) / 1000)
        _row2("P99 per-query (ms)",
              pctile(t_gr, 99) / 1000,
              pctile(t_pgs, 99) / 1000)
        _row2("Max per-query (ms)",
              max(t_gr) / 1000,
              max(t_pgs) / 1000)

        speedup_std = (statistics.mean(t_pgs) / statistics.mean(t_gr)
                       if statistics.mean(t_gr) > 0 else 0)
        print(f"\n  Speedup vs Graph (PG Standard): {speedup_std:.1f}x")

    # --- Candidate & search stats ---
    print(f"\n  Search Statistics:")
    print(SEP)
    print(f"    Candidates per query:  "
          f"Mean {statistics.mean(cand_counts):.1f}  "
          f"Median {statistics.median(cand_counts):.0f}  "
          f"Min {min(cand_counts)}  Max {max(cand_counts)}")
    print(f"    Safe level:            "
          f"Mean {statistics.mean(safe_levels):.1f}  "
          f"Median {statistics.median(safe_levels):.0f}  "
          f"Min {min(safe_levels)}  Max {max(safe_levels)}")
    print(f"    Final level:           "
          f"Mean {statistics.mean(final_levels):.1f}  "
          f"Median {statistics.median(final_levels):.0f}  "
          f"Min {min(final_levels)}  Max {max(final_levels)}")
    print(f"    Prefixes queried:      "
          f"Mean {statistics.mean(prefix_counts):.1f}  "
          f"Median {statistics.median(prefix_counts):.0f}  "
          f"Min {min(prefix_counts)}  Max {max(prefix_counts)}")

    # --- Recall by PostGIS top-1 distance ---
    print(f"\n  Recall by PostGIS Standard Top-1 Distance:")
    print(SEP)
    ranges = [
        ("< 100m",     0,      100),
        ("100m-1km",   100,    1000),
        ("1-10km",     1000,   10000),
        ("> 10km",     10000,  float("inf")),
    ]
    print(f"  {'Range':<14s} {'Queries':>8s} {'Mean Recall':>12s} "
          f"{'Top1 Match%':>12s}")
    print(f"  {'\u2500' * 14} {'\u2500' * 8} {'\u2500' * 12} {'\u2500' * 12}")
    for label, lo, hi in ranges:
        sub = [r for r in rows
               if lo <= float(r["pgs_top1_dist_m"]) < hi]
        cnt = len(sub)
        if cnt == 0:
            print(f"  {label:<14s} {0:>8d}")
            continue
        mr = statistics.mean([float(r["recall_at_k"]) for r in sub])
        tm = (sum(1 for r in sub if int(r["top1_match"]) == 1)
              / cnt * 100)
        print(f"  {label:<14s} {cnt:>8d} {mr:>11.4f}  {tm:>11.1f}%")

    print(f"\n{'=' * W}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding: utf-8
"""
KNN V3.2: planar window termination with local-ellipsoid ranking and final Vincenty
refinement. Reuse V3.1's expansion/brakes. The default reranker selects the leading 8K
planar candidates, ranks them using query-latitude meridian/prime-vertical radii, then
refines Top-K with Vincenty. --max-ellipsoid-n 0 selects the full window; combine it with
--rank-kernel vincenty for full-window iterative Vincenty comparison.

The point-centroid path uses this reranker; line/polygon distance retains V3.1's
_geometry_rerank. Default setup loads Point codes from Neo4j into POIPrefixCache. --from-csv
is the no-graph candidate-source comparison.

Usage:
    python geosot_knn_v3_2.py --queries 10000 --k 3,5,10,20
    python geosot_knn_v3_2.py --from-csv csv/neo4j_poi_point_codes.csv --queries 10000 --k 3,5,10,20 --rank-kernel local --reuse-pg-csv csv/knn_v31_10k.csv --output csv/knn_v32_local8k_n10k.csv
    python geosot_knn_v3_2.py --rank-kernel vincenty --max-ellipsoid-n 0 --queries 50 --k 20
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

from geosot_spatial_reasoning import (
    GRID_SIZES,
    HAS_NUMBA,
    code_to_bbox,
    fast_code_to_center,
    is_valid_g0_code,
    vincenty_fast,
)
from neo4j import GraphDatabase

from geosot_knn_v3 import (
    DEG_TO_M,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    POIPrefixCache,
    _min_margin_m,
    find_safe_level,
    get_poi_prefix_index,
    set_poi_prefix_index,
)
from geosot_knn_v3_1 import GeoSotKNNSearcherV31, get_nxn_neighbors

CodeDist = Tuple[str, float]
_BATCH_VINCENTY_M = None

# CGCS2000 / EPSG:4490: the same ellipsoid as vincenty_fast.
_A_CGCS = 6378137.0
_F_CGCS = 1.0 / 298.257222101
_E2_CGCS = _F_CGCS * (2.0 - _F_CGCS)


def _build_batch_vincenty_m():
    """
    Build a batch Vincenty kernel in meters, using a Numba nopython loop when available.

    @returns Callable (q_lat,q_lon,lats,lons) -> float64 array.
    """
    if _HAS_NP and HAS_NUMBA:
        from numba import njit

        @njit(cache=True)
        def _batch_nb(q_lat, q_lon, lats, lons):
            n = lats.shape[0]
            out = _np.empty(n, dtype=_np.float64)
            for i in range(n):
                out[i] = vincenty_fast(
                    q_lat, q_lon, lats[i], lons[i]) * 1000.0
            return out

        return _batch_nb

    def _batch_py(q_lat, q_lon, lats, lons):
        n = int(lats.shape[0])
        out = _np.empty(n, dtype=_np.float64)
        for i in range(n):
            out[i] = vincenty_fast(
                q_lat, q_lon, float(lats[i]), float(lons[i])) * 1000.0
        return out

    return _batch_py


def _batch_vincenty_m(q_lat: float, q_lon: float, lats, lons):
    """
    @param q_lat Query latitude in degrees.
    @param q_lon Query longitude in degrees.
    @param lats Candidate latitudes.
    @param lons Candidate longitudes.
    @returns Ellipsoidal distances in meters.
    """
    global _BATCH_VINCENTY_M
    if _BATCH_VINCENTY_M is None:
        _BATCH_VINCENTY_M = _build_batch_vincenty_m()
        _BATCH_VINCENTY_M(
            22.5, 114.0,
            _np.array([22.51], dtype=_np.float64),
            _np.array([114.01], dtype=_np.float64))
    return _BATCH_VINCENTY_M(q_lat, q_lon, lats, lons)


def _batch_local_ellipsoid_m(q_lat: float, q_lon: float, lats, lons):
    """
    Local-ellipsoid distance using meridian and prime-vertical radii at the query latitude to
    scale angular offsets into meters, without iterative Vincenty.

    @param q_lat Query latitude in degrees.
    @param q_lon Query longitude in degrees.
    @param lats Candidate latitudes.
    @param lons Candidate longitudes.
    @returns Local-ellipsoid distances in meters.
    """
    phi = math.radians(q_lat)
    s = math.sin(phi)
    one_e2s = 1.0 - _E2_CGCS * s * s
    n_rad = _A_CGCS / math.sqrt(one_e2s)
    m_rad = _A_CGCS * (1.0 - _E2_CGCS) / (one_e2s ** 1.5)
    east = n_rad * math.cos(phi) * math.radians(1.0)
    north = m_rad * math.radians(1.0)
    dlon = lons - q_lon
    dlat = lats - q_lat
    return _np.hypot(east * dlon, north * dlat)


def _batch_andoyer_m(q_lat: float, q_lon: float, lats, lons):
    """
    Vectorized, noniterative Andoyer-Lambert ellipsoidal distance in meters.

    @param q_lat Query latitude in degrees.
    @param q_lon Query longitude in degrees.
    @param lats Candidate latitudes.
    @param lons Candidate longitudes.
    @returns Ellipsoidal distances in meters.
    """
    lat1 = math.radians(q_lat)
    lat2 = _np.radians(lats)
    lam = _np.radians(lons - q_lon) * 0.5
    fmid = (lat1 + lat2) * 0.5
    g = (lat1 - lat2) * 0.5
    sin_g = _np.sin(g)
    cos_g = _np.cos(g)
    sin_f = _np.sin(fmid)
    cos_f = _np.cos(fmid)
    sin_l = _np.sin(lam)
    cos_l = _np.cos(lam)
    s = sin_g * sin_g * cos_l * cos_l + cos_f * cos_f * sin_l * sin_l
    c = cos_g * cos_g * cos_l * cos_l + sin_f * sin_f * sin_l * sin_l
    s = _np.maximum(s, 0.0)
    c = _np.maximum(c, 1e-30)
    w = _np.arctan2(_np.sqrt(s), _np.sqrt(c))
    r = _np.sqrt(s * c) / _np.maximum(w, 1e-30)
    d = 2.0 * w * _A_CGCS
    h1 = (3.0 * r - 1.0) / (2.0 * c)
    h2 = (3.0 * r + 1.0) / (2.0 * _np.maximum(s, 1e-30))
    return d * (1.0 + _F_CGCS * h1 * sin_f * sin_f * cos_g * cos_g
                - _F_CGCS * h2 * cos_f * cos_f * sin_g * sin_g)


def _kernel_dists(kernel: str, q_lat: float, q_lon: float, lats, lons):
    """
    @param kernel vincenty, local, or andoyer.
    @param q_lat Query latitude.
    @param q_lon Query longitude.
    @param lats Candidate latitudes.
    @param lons Candidate longitudes.
    @returns Distances in meters.
    """
    if kernel == "local":
        return _batch_local_ellipsoid_m(q_lat, q_lon, lats, lons)
    if kernel == "andoyer":
        return _batch_andoyer_m(q_lat, q_lon, lats, lons)
    return _batch_vincenty_m(q_lat, q_lon, lats, lons)


def _resolve_shortlist(max_ellipsoid_n: int, k: int) -> int:
    """
    @param max_ellipsoid_n Negative selects an 8K shortlist; zero selects the full window.
    @param k Requested Top-K.
    @returns Planar leaders to rerank; zero means all.
    """
    if max_ellipsoid_n < 0:
        return max(8 * k, k)
    return max_ellipsoid_n


def ellipsoid_rank_window(
    query_center: Tuple[float, float],
    geom_type: str,
    use_idx: bool,
    cand_arr,
    candidates: Set[str],
    cache,
    k: int,
    max_ellipsoid_n: int,
    cos2: float,
    rank_kernel: str = "local",
    refine_k: bool = True,
    planar_ranked: Optional[List[CodeDist]] = None,
) -> Tuple[List[CodeDist], int]:
    """
    Rank candidates inside the planar search window under the selected ellipsoid distance
    kernel. By default, rerank the leading 8K planar candidates locally and refine Top-K with
    Vincenty; zero max_ellipsoid_n means the full window.

    @param query_center (lon,lat).
    @param geom_type Cache key, Point for the paper.
    @param use_idx Whether candidates are cache row indexes.
    @param cand_arr Window row indexes, None in code mode.
    @param candidates Window codes; may be empty in index mode.
    @param cache POIPrefixCache or None.
    @param k Top-K.
    @param max_ellipsoid_n Negative=8K, zero=all, positive=explicit shortlist size.
    @param cos2 Planar prefilter longitude weight.
    @param rank_kernel local, andoyer, or vincenty.
    @param refine_k Refine selected Top-K with Vincenty meters.
    @param planar_ranked Reuse an existing planar shortlist, especially without an index.
    @returns (ascending [(code,meters), ...], number of distance-kernel evaluations).
    """
    q_lon, q_lat = query_center
    kernel = (rank_kernel or "local").strip().lower()
    take_n = _resolve_shortlist(max_ellipsoid_n, k)

    def _top_from_dists(codes_or_sel, dists, from_idx: bool) -> List[CodeDist]:
        """
        @param codes_or_sel Codes or row indexes.
        @param dists Aligned distances.
        @param from_idx True when codes_or_sel contains cache indexes.
        @returns Top-K.
        """
        n_eval = int(dists.shape[0])
        mm = min(k, n_eval)
        if mm <= 0:
            return []
        part = _np.argpartition(dists, mm - 1)[:mm]
        order = part[_np.argsort(dists[part], kind="stable")]
        if from_idx:
            sorted_codes = cache._sorted[geom_type]
            out = [
                (sorted_codes[int(codes_or_sel[j])], float(dists[j]))
                for j in order
                if _np.isfinite(dists[j])
            ]
        else:
            out = [
                (codes_or_sel[int(j)], float(dists[j]))
                for j in order
                if _np.isfinite(dists[j])
            ]
        return out[:k]

    def _refine_vincenty(ranked: List[CodeDist]) -> List[CodeDist]:
        """Refine the same selected Top-K set using Vincenty meters, then sort it by the refined distances."""
        if not refine_k or not ranked:
            return ranked
        n = len(ranked)
        lats = _np.empty(n, dtype=_np.float64)
        lons = _np.empty(n, dtype=_np.float64)
        cidx = None
        centers = None
        if cache is not None:
            cmap = getattr(cache, "_code_idx", None)
            carr = getattr(cache, "_centers", None)
            if isinstance(cmap, dict):
                cidx = cmap.get(geom_type)
            if isinstance(carr, dict):
                centers = carr.get(geom_type)
        if cidx is not None and centers is not None:
            for i, (mc, _) in enumerate(ranked):
                j = cidx.get(mc)
                if j is None:
                    lon, lat = fast_code_to_center(mc)
                    lons[i], lats[i] = lon, lat
                else:
                    lons[i], lats[i] = centers[j, 0], centers[j, 1]
        else:
            for i, (mc, _) in enumerate(ranked):
                lon, lat = fast_code_to_center(mc)
                lons[i], lats[i] = lon, lat
        d = _batch_vincenty_m(q_lat, q_lon, lats, lons)
        pairs = [(ranked[i][0], float(d[i])) for i in range(n)]
        pairs.sort(key=lambda x: x[1])
        return pairs

    if use_idx and cache is not None and cand_arr is not None:
        n_all = int(cand_arr.size)
        if n_all == 0:
            return [], 0
        take = n_all if take_n <= 0 else min(take_n, n_all)
        if take < n_all:
            cidx = cache._code_idx[geom_type]
            if planar_ranked and len(planar_ranked) >= take:
                sel = _np.array(
                    [cidx[mc] for mc, _ in planar_ranked[:take]
                     if mc in cidx],
                    dtype=_np.int64)
            else:
                planar = cache.rank_indices(
                    geom_type, cand_arr, q_lon, q_lat, cos2, take)
                if not planar:
                    return [], 0
                sel = _np.array(
                    [cidx[mc] for mc, _ in planar if mc in cidx],
                    dtype=_np.int64)
            if sel.size == 0:
                return [], 0
        else:
            sel = _np.asarray(cand_arr, dtype=_np.int64)
        centers = cache._centers[geom_type]
        lons = centers[sel, 0]
        lats = centers[sel, 1]
        dists = _kernel_dists(kernel, q_lat, q_lon, lats, lons)
        bad = (
            (lats < -90.0) | (lats > 90.0)
            | (lons < -180.0) | (lons > 180.0)
            | ~_np.isfinite(dists))
        if _np.any(bad):
            dists = dists.copy()
            dists[bad] = _np.inf
        n_eval = int(sel.size)
        out = _top_from_dists(sel, dists, True)
        if kernel != "vincenty":
            out = _refine_vincenty(out)
        return out, n_eval

    pool: Iterable[str]
    if take_n > 0 and planar_ranked:
        pool = [mc for mc, _ in planar_ranked[:take_n]]
    elif take_n > 0 and cache is not None and candidates:
        rank_fn = getattr(cache, "rank_candidates", None)
        planar = (
            rank_fn(geom_type, candidates, q_lon, q_lat, cos2, take_n)
            if callable(rank_fn) else None)
        pool = [mc for mc, _ in planar] if planar else list(candidates)
    else:
        pool = list(candidates)
    codes: List[str] = []
    lons_list: List[float] = []
    lats_list: List[float] = []
    for mc in pool:
        try:
            lon, lat = fast_code_to_center(mc)
        except Exception:
            continue
        codes.append(mc)
        lons_list.append(lon)
        lats_list.append(lat)
    if not codes:
        return [], 0
    lons = _np.asarray(lons_list, dtype=_np.float64)
    lats = _np.asarray(lats_list, dtype=_np.float64)
    dists = _kernel_dists(kernel, q_lat, q_lon, lats, lons)
    n_eval = len(codes)
    out = _top_from_dists(codes, dists, False)
    if kernel != "vincenty":
        out = _refine_vincenty(out)
    return out, n_eval


class GeoSotKNNSearcherV32(GeoSotKNNSearcherV31):
    """
    V3.2 KNN: inherit V3.1's planar coverage test and refine the selected window shortlist with
    the configured ellipsoid kernel and optional Vincenty step.
    """

    def knn_search(
        self,
        query_code: str,
        k: int = 5,
        min_margin_m: float = 100.0,
        min_level: int = 8,
        coarse_top_n: int = 50,
        geometry_scoring: bool = False,
        max_widen: int = 5,
        max_candidates_per_level: int = 20000,
        start_level_max: int = 0,
        geo_workers: int = 0,
        max_ellipsoid_n: int = -1,
        rank_kernel: str = "local",
        refine_k: bool = True,
    ) -> Tuple[List[CodeDist], Dict]:
        """
        Search with planar coverage and rerank the resulting window. The default local kernel uses
        query-latitude meridian/prime-vertical radii, then Vincenty-refines Top-K. Full-window
        iterative Vincenty uses rank_kernel=vincenty and max_ellipsoid_n=0.

        @param query_code Query GeoSOT code.
        @param k Top-K.
        @param max_ellipsoid_n Negative=leading 8K planar candidates; zero=full window;
            positive=explicit shortlist.
        @param rank_kernel local, andoyer, or vincenty.
        @param refine_k Vincenty-refine Top-K after local/Andoyer ranking.
        @returns ([(code,meters), ...], info).
        """
        del geometry_scoring, geo_workers
        _t = time.perf_counter
        query_code = query_code.strip().upper()
        query_center = fast_code_to_center(query_code)
        cos_qlat = math.cos(math.radians(query_center[1]))
        cos2 = cos_qlat * cos_qlat

        t0 = _t()
        safe_level_raw = find_safe_level(query_code, min_margin_m)
        if start_level_max > 0 and safe_level_raw < start_level_max:
            safe_level = start_level_max
        else:
            safe_level = safe_level_raw
        t_level_ms = (_t() - t0) * 1000

        t0 = _t()
        level = safe_level
        w = 3
        n_prefixes = 0
        n_rounds = 0
        braked = False
        t_cand_query = 0.0
        t_coarse_rank = 0.0
        short_n = _resolve_shortlist(max_ellipsoid_n, k)
        rank_m = max(coarse_top_n, k, short_n)

        cache = get_poi_prefix_index()
        use_idx = (
            _HAS_NP and cache is not None
            and hasattr(cache, "find_indices_batch")
            and hasattr(cache, "rank_indices")
            and cache.has_geom_type(self._geom_type))

        candidates: Set[str] = set()
        cand_arr = None

        def _collect(level_: int, w_: int) -> int:
            """
            @param level_ Current GeoSOT level.
            @param w_ Neighborhood width.
            @returns Number of newly added candidates.
            """
            nonlocal candidates, cand_arr, n_prefixes
            prefix = query_code[:level_ + 2]
            cells = get_nxn_neighbors(prefix, w_)
            n_prefixes += len(cells)
            if use_idx:
                new_arr = cache.find_indices_batch(
                    self._geom_type, cells, query_code)
                if cand_arr is None:
                    prev = 0
                    cand_arr = new_arr
                else:
                    prev = int(cand_arr.size)
                    cand_arr = _np.union1d(cand_arr, new_arr)
                return int(cand_arr.size) - prev
            new_c = self._query_candidates(set(cells), query_code)
            prev = len(candidates)
            candidates.update(new_c)
            return len(candidates) - prev

        def _count() -> int:
            """@returns Number of candidates in the current window."""
            if use_idx:
                return 0 if cand_arr is None else int(cand_arr.size)
            return len(candidates)

        def _rank_all() -> List[CodeDist]:
            """@returns Ascending squared planar distances, with at most rank_m entries."""
            if use_idx:
                if cand_arr is None or cand_arr.size == 0:
                    return []
                return cache.rank_indices(
                    self._geom_type, cand_arr,
                    query_center[0], query_center[1], cos2, rank_m)
            if not candidates:
                return []
            if cache is not None and hasattr(cache, "rank_candidates"):
                vec = cache.rank_candidates(
                    self._geom_type, candidates,
                    query_center[0], query_center[1], cos2, rank_m)
                if vec is not None:
                    return vec
            ranked: List[CodeDist] = []
            for mc in candidates:
                try:
                    cc = fast_code_to_center(mc)
                    dlon = cc[0] - query_center[0]
                    dlat = cc[1] - query_center[1]
                    ranked.append(
                        (mc, dlon * dlon * cos2 + dlat * dlat))
                except Exception:
                    pass
            ranked.sort(key=lambda x: x[1])
            return ranked[:rank_m]

        def _r_safe_m(level_: int, w_: int) -> float:
            """@returns Inscribed-circle radius of the w-by-w neighborhood, in meters."""
            cell_margin = _min_margin_m(
                query_center[0], query_center[1],
                code_to_bbox(query_code[:level_ + 2]))
            grid_step = GRID_SIZES.get(level_, (0.0, 0.0))
            grid_m = grid_step[0] * DEG_TO_M * cos_qlat
            return cell_margin + ((w_ - 1) / 2.0) * grid_m

        def _coverage_ok(ranked: List[CodeDist],
                         level_: int, w_: int) -> bool:
            """@returns Whether planar r_K is at most r_safe."""
            if len(ranked) < k:
                return False
            r_k_m = math.sqrt(ranked[k - 1][1]) * DEG_TO_M
            return r_k_m <= _r_safe_m(level_, w_)

        _tq0 = _t()
        _collect(level, w)
        t_cand_query += (_t() - _tq0) * 1000
        n_rounds += 1
        _tr0 = _t()
        rank_list = _rank_all()
        t_coarse_rank += (_t() - _tr0) * 1000

        hard_cap_rounds = (max_widen // 2) + (safe_level - min_level) + 4
        while (not _coverage_ok(rank_list, level, w)
               and n_rounds < hard_cap_rounds):
            if w + 2 <= max_widen and _count() >= k:
                w += 2
            elif level > min_level:
                level -= 1
                w = 3
            else:
                break
            _tq0 = _t()
            added = _collect(level, w)
            t_cand_query += (_t() - _tq0) * 1000
            n_rounds += 1
            if added:
                _tr0 = _t()
                rank_list = _rank_all()
                t_coarse_rank += (_t() - _tr0) * 1000
            if added > max_candidates_per_level and len(rank_list) >= k:
                braked = True
                break

        t_search_ms = (_t() - t0) * 1000
        final_level = level
        total_candidates = _count()
        planar_top = [
            (mc, math.sqrt(sq) * DEG_TO_M) for mc, sq in rank_list[:k]
        ]

        empty_info = {
            "safe_level": safe_level, "final_level": final_level,
            "total_candidates": total_candidates, "prefixes": n_prefixes,
            "final_width": w, "rounds": n_rounds,
            "braked": int(braked),
            "t_level_ms": round(t_level_ms, 3),
            "t_search_ms": round(t_search_ms, 3),
            "t_cand_query_ms": round(t_cand_query, 3),
            "t_coarse_rank_ms": round(t_coarse_rank, 3),
            "t_featinfo_ms": 0.0,
            "t_rank_ms": 0.0,
            "t_ellipsoid_ms": 0.0,
            "n_ellipsoid_eval": 0,
            "rank_metric": rank_kernel,
            "planar_top_k": "",
            "t_total_ms": round(t_level_ms + t_search_ms, 3),
        }
        if not rank_list:
            return [], empty_info

        t_e0 = _t()
        results, n_eval = ellipsoid_rank_window(
            query_center, self._geom_type, use_idx, cand_arr,
            candidates, cache, k, max_ellipsoid_n, cos2,
            rank_kernel=rank_kernel, refine_k=refine_k,
            planar_ranked=rank_list)
        t_ell = (_t() - t_e0) * 1000
        t_total = t_level_ms + t_search_ms + t_ell
        info = {
            "safe_level": safe_level,
            "final_level": final_level,
            "total_candidates": total_candidates,
            "prefixes": n_prefixes,
            "final_width": w,
            "rounds": n_rounds,
            "braked": int(braked),
            "t_level_ms": round(t_level_ms, 3),
            "t_search_ms": round(t_search_ms, 3),
            "t_cand_query_ms": round(t_cand_query, 3),
            "t_coarse_rank_ms": round(t_coarse_rank, 3),
            "t_featinfo_ms": 0.0,
            "t_rank_ms": round(t_ell, 3),
            "t_ellipsoid_ms": round(t_ell, 3),
            "n_ellipsoid_eval": n_eval,
            "rank_metric": rank_kernel,
            "planar_top_k": ";".join(mc for mc, _ in planar_top),
            "t_total_ms": round(t_total, 3),
        }
        return results[:k], info


def _resolve(path: str) -> str:
    """Resolve a user path; if a relative path does not exist under cwd, try the script directory."""
    if not path:
        return path
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    alt = os.path.join(_ROOT, path)
    return alt if os.path.isfile(alt) else path


def _load_poi_codes(path: str, geom_type: str) -> List[str]:
    """
    @param path Exported code CSV.
    @param geom_type Filter geom_type when the column exists.
    @returns Valid deduplicated main_codes.
    """
    codes = []
    seen = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt = (row.get("geom_type") or "").strip()
            if gt and gt != geom_type:
                continue
            mc = (row.get("main_code") or "").strip().upper()
            if not mc or mc in seen or not is_valid_g0_code(mc):
                continue
            seen.add(mc)
            codes.append(mc)
    return codes


def _split_codes(text: str) -> List[str]:
    """Parse semicolon-separated codes and return uppercase codes."""
    return [c.strip().upper() for c in (text or "").split(";") if c.strip()]


def _set_recall(pred: List[str], gold: List[str], k: int) -> Optional[float]:
    """
    @param pred Predicted Top-K.
    @param gold Reference Top-K.
    @param k K.
    @returns Set agreement, or None when gold has fewer than K entries.
    """
    if k <= 0 or len(gold) < k:
        return None
    return len(set(pred[:k]) & set(gold[:k])) / k


def _parse_k_list(text: str) -> List[int]:
    """Parse comma-separated positive K values, for example 3,5,10,20, and return a sorted unique list."""
    try:
        vals = sorted(set(int(x) for x in str(text).split(",") if x.strip()))
    except ValueError:
        vals = []
    if not vals or any(v <= 0 for v in vals):
        raise SystemExit("--k must be positive integers, e.g. 3,5,10,20")
    return vals


def _load_paper(
        path: str, k_set: Set[int]
        ) -> Tuple[List[str], Dict[str, Dict[int, Dict]]]:
    """
    @param path Paper's knn_v31 reference CSV.
    @param k_set K values to retain.
    @returns (query appearance order, query_code -> k -> G3R/PostGIS data).
    """
    order: List[str] = []
    seen = set()
    by_qc: Dict[str, Dict[int, Dict]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qc = (row.get("query_code") or "").strip().upper()
            if not qc:
                continue
            try:
                kr = int(row.get("k") or 0)
            except ValueError:
                continue
            if kr not in k_set:
                continue
            if qc not in seen:
                seen.add(qc)
                order.append(qc)
            by_qc.setdefault(qc, {})[kr] = {
                "gr_top": _split_codes(row.get("gr_top_k") or ""),
                "pgs_top": _split_codes(row.get("pgs_top_k") or ""),
                "pgs_top1_dist_m": float(row.get("pgs_top1_dist_m") or 0.0),
            }
    return order, by_qc


def _load_true(path: str) -> Dict[str, List[str]]:
    """
    @param path Exhaustive-reference CSV.
    @returns query_code -> complete reference list; callers truncate at K.
    """
    out: Dict[str, List[str]] = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qc = (row.get("query_code") or "").strip().upper()
            if not qc:
                continue
            codes = _split_codes(row.get("true_top_k") or "")
            if codes:
                out[qc] = codes
    return out


def _connect_neo4j():
    """@returns (driver,session); the caller must close both."""
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    session = driver.session(database=NEO4J_DATABASE)
    session.run("RETURN 1").consume()
    return driver, session


def _parse_args() -> argparse.Namespace:
    """@returns Parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description="KNN V3.2: planar stop, Vincenty rank in the window.")
    p.add_argument(
        "--from-csv", default="",
        help="optional exported POI CSV; empty = pull codes from Neo4j")
    p.add_argument(
        "--reuse-pg-csv",
        default=os.path.join(_ROOT, "csv", "knn_v31_10k.csv"))
    p.add_argument(
        "--true-csv",
        default=os.path.join(_ROOT, "csv", "knn_r14_true_n500.csv"))
    p.add_argument(
        "--output",
        default=os.path.join(_ROOT, "csv", "knn_v32_ellipsoid_rank.csv"))
    p.add_argument("--geom-type", default="Point")
    p.add_argument("--queries", type=int, default=50)
    p.add_argument(
        "--k", default="3,5,10,20",
        help="comma-separated K list, same as V3.1 (each K is timed independently)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=100.0)
    p.add_argument("--min-level", type=int, default=8)
    p.add_argument("--max-widen", type=int, default=5)
    p.add_argument("--max-cand-per-level", type=int, default=20000)
    p.add_argument("--start-level-max", type=int, default=13)
    p.add_argument("--coarse-top", type=int, default=50)
    p.add_argument(
        "--max-ellipsoid-n", type=int, default=-1,
        help="-1: 8K 平面短名单再精排; 0=整窗; >0=指定短名单")
    p.add_argument(
        "--rank-kernel", default="local",
        choices=("local", "andoyer", "vincenty"),
        help="local=距离推理局部椭球半径; andoyer=闭式椭球; "
             "vincenty=迭代精排 (旧 V3.2)")
    p.add_argument(
        "--no-refine-k", action="store_true",
        help="local/andoyer 排序后不对 Top-K 再跑 Vincenty")
    p.add_argument("--throwaway", type=int, default=5)
    p.add_argument(
        "--no-poi-cache", action="store_true",
        help="query-time Cypher STARTS WITH, no client prefix index")
    return p.parse_args()


def _mean(xs: List[float]) -> Optional[float]:
    """
    @param xs Samples.
    @returns Mean, or None for an empty sequence.
    """
    return statistics.mean(xs) if xs else None


def main() -> None:
    """
    Connect to Neo4j, build a prefix index, and evaluate V3.2 against the planar reference,
    PostGIS, and exhaustive ellipsoid reference where available.
    """
    args = _parse_args()
    k_list = _parse_k_list(args.k)
    k_max = max(k_list)
    k_str = ",".join(str(x) for x in k_list)
    from_csv = _resolve(args.from_csv) if args.from_csv else ""
    paper_csv = _resolve(args.reuse_pg_csv) if args.reuse_pg_csv else ""
    true_csv = _resolve(args.true_csv) if args.true_csv else ""
    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    print("=" * 72)
    print("GeoSOT KNN V3.2  planar-stop + distance-reasoning rank")
    print("=" * 72)
    print(f"  HAS_NUMBA={HAS_NUMBA}  kernel={args.rank_kernel}  "
          f"max_ellipsoid_n={args.max_ellipsoid_n}  "
          f"refine_k={not args.no_refine_k}")
    print(f"  K={k_str}  (each K independently, same as V3.1)")

    paper_map: Dict[str, Dict[int, Dict]] = {}
    paper_order: List[str] = []
    if paper_csv:
        if not os.path.isfile(paper_csv):
            raise SystemExit(f"missing paper KNN CSV: {paper_csv}")
        paper_order, paper_map = _load_paper(paper_csv, set(k_list))
        print(f"  reuse queries from {paper_csv}: {len(paper_order):,}")

    true_map = _load_true(true_csv) if true_csv else {}

    driver = None
    session = None
    try:
        if from_csv:
            if not os.path.isfile(from_csv):
                raise SystemExit(f"missing POI CSV: {from_csv}")
            codes = _load_poi_codes(from_csv, args.geom_type)
            print(f"  POI codes: {len(codes):,}  from CSV {from_csv}")
            print("  Neo4j: skipped (--from-csv)")
            cache = POIPrefixCache.from_codes(
                args.geom_type, codes, verbose=True)
            set_poi_prefix_index(cache)
            searcher = GeoSotKNNSearcherV32(
                None, geom_type=args.geom_type, verbose=False)
            pool = set(codes)
        else:
            print(f"  Connecting Neo4j {NEO4J_URI}  db={NEO4J_DATABASE}")
            t0 = time.perf_counter()
            driver, session = _connect_neo4j()
            print(f"  Connected ({(time.perf_counter() - t0) * 1000:.0f} ms)")
            searcher = GeoSotKNNSearcherV32(
                session, geom_type=args.geom_type, verbose=False)
            if args.no_poi_cache:
                set_poi_prefix_index(None)
                print("  [poi_cache] skipped (--no-poi-cache, Cypher)")
                if not paper_order:
                    raise SystemExit(
                        "--no-poi-cache needs --reuse-pg-csv for queries")
                pool = set(paper_order)
            else:
                print("  [bench_setup] POI prefix cache from Neo4j")
                cache = POIPrefixCache(
                    session, geom_types=[args.geom_type], verbose=True)
                set_poi_prefix_index(cache)
                pool = set(cache._sorted.get(args.geom_type) or [])
                print(f"  POI codes in graph: {len(pool):,}")

        if paper_order:
            kept = [q for q in paper_order if q in pool]
            dropped = len(paper_order) - len(kept)
            query_codes = kept[: args.queries]
            print(f"  paper queries kept: {len(query_codes):,} / "
                  f"{len(paper_order):,}  (dropped not in index: {dropped:,})")
        else:
            random.seed(args.seed)
            query_codes = random.sample(
                list(pool), min(args.queries, len(pool)))
            print(f"  sampled queries: {len(query_codes):,}  seed={args.seed}")
        if not query_codes:
            raise SystemExit("no overlapping queries")

        if true_map:
            n_true = sum(1 for q in query_codes if q in true_map)
            print(f"  true-CSV overlap in this run: {n_true:,} / "
                  f"{len(query_codes):,}")

        knn_kw = dict(
            min_margin_m=args.margin,
            min_level=args.min_level, coarse_top_n=args.coarse_top,
            geometry_scoring=False, max_widen=args.max_widen,
            max_candidates_per_level=args.max_cand_per_level,
            start_level_max=args.start_level_max,
            max_ellipsoid_n=args.max_ellipsoid_n,
            rank_kernel=args.rank_kernel,
            refine_k=not args.no_refine_k,
        )
        n_tw = min(args.throwaway, len(query_codes))
        for qc in query_codes[:n_tw]:
            searcher.knn_search(qc, k=k_max, **knn_kw)
        print(f"  Throwaway {n_tw} queries done (k={k_max})")

        fields = [
            "query_code", "k",
            "recall_vs_planar", "recall_vs_paper_gr", "recall_vs_pgs",
            "recall_vs_true", "top1_vs_planar", "top1_vs_pgs",
            "v32_top1_dist_m", "n_ellipsoid_eval", "t_ellipsoid_ms",
            "t_gr_us", "gr_total_candidates",
            "v32_top_k", "planar_top_k", "paper_gr_top_k", "pgs_top_k",
        ]
        stats: Dict[int, Dict[str, list]] = {
            kv: {
                "rec_planar": [], "rec_paper": [], "rec_pgs": [],
                "rec_true": [], "n_rank_diff": 0,
                "t_ell": [], "t_us": [], "n_eval": [],
            }
            for kv in k_list
        }
        n_rows = 0

        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            t_run = time.perf_counter()
            n_jobs = len(query_codes) * len(k_list)
            done = 0
            for i, qc in enumerate(query_codes):
                paper_by_k = paper_map.get(qc) or {}
                true_full = true_map.get(qc) or []
                for kv in k_list:
                    t0 = time.perf_counter()
                    gr, info = searcher.knn_search(qc, k=kv, **knn_kw)
                    t_us = (time.perf_counter() - t0) * 1e6
                    v32_top = [r[0] for r in gr[:kv]]
                    planar_top = _split_codes(info.get("planar_top_k") or "")
                    paper = paper_by_k.get(kv) or {}
                    paper_gr = paper.get("gr_top") or []
                    pgs_top = paper.get("pgs_top") or []
                    true_top = true_full[:kv]

                    r_pl = _set_recall(v32_top, planar_top, kv)
                    r_gr = _set_recall(v32_top, paper_gr, kv)
                    r_pg = _set_recall(v32_top, pgs_top, kv)
                    r_tr = _set_recall(v32_top, true_top, kv)
                    st = stats[kv]
                    if r_pl is not None:
                        st["rec_planar"].append(r_pl)
                        if r_pl < 1.0:
                            st["n_rank_diff"] += 1
                    if r_gr is not None:
                        st["rec_paper"].append(r_gr)
                    if r_pg is not None:
                        st["rec_pgs"].append(r_pg)
                    if r_tr is not None:
                        st["rec_true"].append(r_tr)
                    t_ell = float(info.get("t_ellipsoid_ms") or 0.0)
                    st["t_ell"].append(t_ell)
                    st["t_us"].append(t_us)
                    st["n_eval"].append(int(info.get("n_ellipsoid_eval") or 0))

                    w.writerow({
                        "query_code": qc,
                        "k": kv,
                        "recall_vs_planar": (
                            "" if r_pl is None else round(r_pl, 4)),
                        "recall_vs_paper_gr": (
                            "" if r_gr is None else round(r_gr, 4)),
                        "recall_vs_pgs": (
                            "" if r_pg is None else round(r_pg, 4)),
                        "recall_vs_true": (
                            "" if r_tr is None else round(r_tr, 4)),
                        "top1_vs_planar": (
                            1 if v32_top and planar_top
                            and v32_top[0] == planar_top[0] else 0),
                        "top1_vs_pgs": (
                            1 if v32_top and pgs_top
                            and v32_top[0] == pgs_top[0] else 0),
                        "v32_top1_dist_m": (
                            round(gr[0][1], 6) if gr else ""),
                        "n_ellipsoid_eval": info.get("n_ellipsoid_eval", 0),
                        "t_ellipsoid_ms": round(t_ell, 3),
                        "t_gr_us": round(t_us, 1),
                        "gr_total_candidates": info.get(
                            "total_candidates", 0),
                        "v32_top_k": ";".join(v32_top),
                        "planar_top_k": ";".join(planar_top),
                        "paper_gr_top_k": ";".join(paper_gr[:kv]),
                        "pgs_top_k": ";".join(pgs_top[:kv]),
                    })
                    n_rows += 1
                    done += 1
                if (i + 1) % 10 == 0 or i + 1 == len(query_codes):
                    elapsed = time.perf_counter() - t_run
                    rate = done / elapsed if elapsed else 0
                    print(f"\r  {i + 1:,}/{len(query_codes):,} queries  "
                          f"{done:,}/{n_jobs:,}  {rate:.2f} jobs/s",
                          end="", flush=True)
            print()

        print(f"wrote {n_rows:,} rows -> {out}")

        def _pct(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:.4%}"

        for kv in k_list:
            st = stats[kv]
            n_pl = len(st["rec_planar"])
            print(f"  ---- K={kv}  n={n_pl:,} ----")
            print(f"    mean set-agree vs same-window planar: "
                  f"{_pct(_mean(st['rec_planar']))}  "
                  f"(set differs on {st['n_rank_diff']}/{n_pl})")
            print(f"    mean set-agree vs paper V3.1 G3R:     "
                  f"{_pct(_mean(st['rec_paper']))}")
            print(f"    mean set-agree vs paper 8x PostGIS:   "
                  f"{_pct(_mean(st['rec_pgs']))}")
            print(f"    mean set-agree vs R1-4 ellipsoid:     "
                  f"{_pct(_mean(st['rec_true']))}  "
                  f"(n={len(st['rec_true'])})")
            if st["t_ell"]:
                print(f"    median Vincenty stage: "
                      f"{statistics.median(st['t_ell']):.3f} ms")
            if st["t_us"]:
                print(f"    median end-to-end:     "
                      f"{statistics.median(st['t_us']) / 1000.0:.3f} ms")
            if st["n_eval"]:
                print(f"    median window eval:    "
                      f"{statistics.median(st['n_eval']):.0f} candidates")
    finally:
        if session is not None:
            session.close()
        if driver is not None:
            driver.close()
            print("  Neo4j session closed")


if __name__ == "__main__":
    main()

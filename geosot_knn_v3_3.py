#!/usr/bin/env python3
# coding: utf-8
"""
KNN V3.3 coverage-metric comparison, not the main paper default. Retain V3.2's starting
level, expansion, and brakes, but compute r_K and r_safe with the same local-ellipsoid
meters-per-degree factors at the query. Refine selected Top-K with Vincenty after stopping.
--stop-kernel vincenty evaluates the whole window iteratively at every expansion for a more
expensive comparison; local is the default.

Usage:
    python geosot_knn_v3_3.py --queries 300 --k 3,5,10,20 --output csv/knn_v33_ellstop_n300.csv
    python geosot_knn_v3_3.py --from-csv csv/neo4j_poi_point_codes.csv --queries 10000 --k 3,5,10,20 --output csv/knn_v33_ellstop_n10k.csv
    python geosot_knn_v3_3.py --stop-kernel vincenty --queries 50 --k 20
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
from typing import Dict, List, Optional, Set, Tuple

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
)

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
from geosot_knn_v3_1 import get_nxn_neighbors
from geosot_knn_v3_2 import (
    _A_CGCS,
    _E2_CGCS,
    _batch_vincenty_m,
    _connect_neo4j,
    _kernel_dists,
    _load_paper,
    _load_poi_codes,
    _load_true,
    _mean,
    _parse_k_list,
    _resolve,
    _set_recall,
    _split_codes,
    CodeDist,
    GeoSotKNNSearcherV32,
)


def _mn_meters_per_deg(lat_deg: float) -> Tuple[float, float]:
    """
    Return prime-vertical/meridian scaling at the query latitude.

    @param lat_deg Latitude in degrees.
    @returns (east_m_per_deg_lon,north_m_per_deg_lat).
    """
    phi = math.radians(lat_deg)
    s = math.sin(phi)
    one_e2s = 1.0 - _E2_CGCS * s * s
    n_rad = _A_CGCS / math.sqrt(one_e2s)
    m_rad = _A_CGCS * (1.0 - _E2_CGCS) / (one_e2s ** 1.5)
    east = n_rad * math.cos(phi) * math.radians(1.0)
    north = m_rad * math.radians(1.0)
    return east, north


def _min_margin_ell_m(
        lon: float, lat: float,
        bbox: Tuple[float, float, float, float],
        east: float, north: float) -> float:
    """
    @param lon Query longitude.
    @param lat Query latitude.
    @param bbox (min_lon,max_lon,min_lat,max_lat).
    @param east Meters per longitude degree.
    @param north Meters per latitude degree.
    @returns Minimum local-ellipsoid point-to-boundary distance in meters.
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    return min(
        (lat - min_lat) * north,
        (max_lat - lat) * north,
        (lon - min_lon) * east,
        (max_lon - lon) * east,
    )


def _r_safe_ell_m(
        query_code: str, query_center: Tuple[float, float],
        level: int, w: int, east: float, north: float) -> float:
    """
    Compute cell_margin+((w-1)/2)*min(delta_EW,delta_NS), using local-ellipsoid side lengths as
    in the paper's inscribed-window construction.

    @param query_code Query code.
    @param query_center (lon,lat).
    @param level Current level.
    @param w Neighborhood width.
    @param east Meters per longitude degree.
    @param north Meters per latitude degree.
    @returns r_safe in meters.
    """
    bbox = code_to_bbox(query_code[:level + 2])
    cell_margin = _min_margin_ell_m(
        query_center[0], query_center[1], bbox, east, north)
    dlon, dlat = GRID_SIZES.get(level, (0.0, 0.0))
    grid_m = min(dlon * east, dlat * north)
    return cell_margin + ((w - 1) / 2.0) * grid_m


class GeoSotKNNSearcherV33(GeoSotKNNSearcherV32):
    """V3.3: use the same local-ellipsoid metric for window coverage and candidate ranking."""

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
        stop_kernel: str = "local",
    ) -> Tuple[List[CodeDist], Dict]:
        """
        Stop with local-ellipsoid r_K/r_safe, rank in the same metric, then optionally refine Top-K
        with Vincenty. Starting-level selection retains the V3.1 planar margin; stop_kernel=vincenty
        iterates over the full window at every expansion.

        @param query_code Query GeoSOT code.
        @param k Top-K.
        @param stop_kernel local, andoyer, or vincenty.
        @param rank_kernel Defaults to stop_kernel; configurable for comparisons.
        @param refine_k Vincenty-refine Top-K after non-Vincenty stopping.
        @returns ([(code,meters), ...], info).
        """
        del geometry_scoring, geo_workers, max_ellipsoid_n
        _t = time.perf_counter
        query_code = query_code.strip().upper()
        query_center = fast_code_to_center(query_code)
        q_lon, q_lat = query_center
        cos_qlat = math.cos(math.radians(q_lat))
        cos2 = cos_qlat * cos_qlat
        east, north = _mn_meters_per_deg(q_lat)
        stop_k = (stop_kernel or "local").strip().lower()
        rank_k = (rank_kernel or stop_k).strip().lower()

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
        t_stop = 0.0
        n_stop_eval = 0
        planar_stop_round: Optional[int] = None
        last_dists = None
        last_sel = None
        last_from_idx = False
        last_r_k = None
        last_r_safe = None

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
            @param level_ Current level.
            @param w_ Neighborhood width.
            @returns Newly added candidate count.
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

        def _rank_planar() -> List[CodeDist]:
            """
            @returns Ascending squared planar distances, used only to compare possible earlier planar
                termination.
            """
            if use_idx:
                if cand_arr is None or cand_arr.size == 0:
                    return []
                return cache.rank_indices(
                    self._geom_type, cand_arr,
                    q_lon, q_lat, cos2, max(k, 1))
            if not candidates:
                return []
            ranked: List[CodeDist] = []
            for mc in candidates:
                try:
                    cc = fast_code_to_center(mc)
                    dlon = cc[0] - q_lon
                    dlat = cc[1] - q_lat
                    ranked.append((mc, dlon * dlon * cos2 + dlat * dlat))
                except Exception:
                    pass
            ranked.sort(key=lambda x: x[1])
            return ranked[:k]

        def _window_coords():
            """@returns (lats,lons,sel,from_idx); lats is empty for an empty window."""
            if use_idx and cache is not None and cand_arr is not None:
                n = int(cand_arr.size)
                if n == 0:
                    return (_np.empty(0), _np.empty(0), cand_arr, True)
                centers = cache._centers[self._geom_type]
                return (
                    centers[cand_arr, 1],
                    centers[cand_arr, 0],
                    cand_arr,
                    True,
                )
            codes = list(candidates)
            if not codes:
                return (_np.empty(0), _np.empty(0), codes, False)
            lats = _np.empty(len(codes), dtype=_np.float64)
            lons = _np.empty(len(codes), dtype=_np.float64)
            for i, mc in enumerate(codes):
                lon, lat = fast_code_to_center(mc)
                lons[i], lats[i] = lon, lat
            return lats, lons, codes, False

        def _coverage_ell() -> bool:
            """@returns Whether the Kth local/closed-form/Vincenty distance is <=r_safe."""
            nonlocal t_stop, n_stop_eval, last_dists, last_sel
            nonlocal last_from_idx, last_r_k, last_r_safe
            n = _count()
            r_safe = _r_safe_ell_m(
                query_code, query_center, level, w, east, north)
            last_r_safe = r_safe
            if n < k:
                last_r_k = None
                return False
            ts0 = _t()
            lats, lons, sel, from_idx = _window_coords()
            dists = _kernel_dists(stop_k, q_lat, q_lon, lats, lons)
            t_stop += (_t() - ts0) * 1000
            n_stop_eval = int(dists.shape[0])
            last_dists, last_sel, last_from_idx = dists, sel, from_idx
            kth = float(_np.partition(dists, k - 1)[k - 1])
            last_r_k = kth
            return kth <= r_safe

        def _coverage_planar(ranked: List[CodeDist]) -> bool:
            """
            @param ranked Squared planar distance list.
            @returns Whether planar r_K<=planar r_safe, for the V3.1 comparison only.
            """
            if len(ranked) < k:
                return False
            r_k_m = math.sqrt(ranked[k - 1][1]) * DEG_TO_M
            bbox = code_to_bbox(query_code[:level + 2])
            cell_margin = _min_margin_m(q_lon, q_lat, bbox)
            grid_step = GRID_SIZES.get(level, (0.0, 0.0))
            grid_m = grid_step[0] * DEG_TO_M * cos_qlat
            r_safe = cell_margin + ((w - 1) / 2.0) * grid_m
            return r_k_m <= r_safe

        def _topk_from_last() -> List[CodeDist]:
            """@returns Top-K under the selected stopping kernel."""
            dists = last_dists
            sel = last_sel
            if dists is None or sel is None or int(dists.shape[0]) == 0:
                return []
            mm = min(k, int(dists.shape[0]))
            part = _np.argpartition(dists, mm - 1)[:mm]
            order = part[_np.argsort(dists[part], kind="stable")]
            if last_from_idx:
                sorted_codes = cache._sorted[self._geom_type]
                out = [
                    (sorted_codes[int(sel[j])], float(dists[j]))
                    for j in order if _np.isfinite(dists[j])
                ]
            else:
                out = [
                    (sel[int(j)], float(dists[j]))
                    for j in order if _np.isfinite(dists[j])
                ]
            return out[:k]

        def _refine_vincenty(ranked: List[CodeDist]) -> List[CodeDist]:
            """
            @param ranked Selected Top-K.
            @returns The same set with Vincenty-meter distances, sorted again.
            """
            if not refine_k or stop_k == "vincenty" or not ranked:
                return ranked
            n = len(ranked)
            lats = _np.empty(n, dtype=_np.float64)
            lons = _np.empty(n, dtype=_np.float64)
            cidx = cache._code_idx.get(self._geom_type) if cache else None
            centers = cache._centers.get(self._geom_type) if cache else None
            for i, (mc, _) in enumerate(ranked):
                if cidx is not None and centers is not None:
                    j = cidx.get(mc)
                    if j is not None:
                        lons[i], lats[i] = centers[j, 0], centers[j, 1]
                        continue
                lon, lat = fast_code_to_center(mc)
                lons[i], lats[i] = lon, lat
            d = _batch_vincenty_m(q_lat, q_lon, lats, lons)
            pairs = [(ranked[i][0], float(d[i])) for i in range(n)]
            pairs.sort(key=lambda x: x[1])
            return pairs

        _tq0 = _t()
        _collect(level, w)
        t_cand_query += (_t() - _tq0) * 1000
        n_rounds += 1
        _tr0 = _t()
        rank_list = _rank_planar()
        t_coarse_rank += (_t() - _tr0) * 1000
        if _coverage_planar(rank_list) and planar_stop_round is None:
            planar_stop_round = n_rounds
        ell_ok = _coverage_ell()

        hard_cap_rounds = (max_widen // 2) + (safe_level - min_level) + 4
        while (not ell_ok) and n_rounds < hard_cap_rounds:
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
                rank_list = _rank_planar()
                t_coarse_rank += (_t() - _tr0) * 1000
            if (_coverage_planar(rank_list)
                    and planar_stop_round is None):
                planar_stop_round = n_rounds
            ell_ok = _coverage_ell()
            if added > max_candidates_per_level and _count() >= k:
                braked = True
                break

        t_search_ms = (_t() - t0) * 1000
        final_level = level
        total_candidates = _count()
        planar_top = [
            (mc, math.sqrt(sq) * DEG_TO_M) for mc, sq in rank_list[:k]
        ]
        extra_expand = 0
        earlier_stop = 0
        extra_rounds = 0
        if planar_stop_round is None:
            earlier_stop = int(ell_ok)
        else:
            extra_rounds = n_rounds - planar_stop_round
            extra_expand = int(extra_rounds > 0)
            earlier_stop = int(extra_rounds < 0)

        empty_info = {
            "safe_level": safe_level, "final_level": final_level,
            "total_candidates": total_candidates, "prefixes": n_prefixes,
            "final_width": w, "rounds": n_rounds,
            "braked": int(braked),
            "t_level_ms": round(t_level_ms, 3),
            "t_search_ms": round(t_search_ms, 3),
            "t_cand_query_ms": round(t_cand_query, 3),
            "t_coarse_rank_ms": round(t_coarse_rank, 3),
            "t_stop_ms": round(t_stop, 3),
            "t_featinfo_ms": 0.0,
            "t_rank_ms": 0.0,
            "t_ellipsoid_ms": round(t_stop, 3),
            "n_ellipsoid_eval": n_stop_eval,
            "rank_metric": rank_k,
            "stop_metric": stop_k,
            "planar_top_k": "",
            "extra_expand": extra_expand,
            "earlier_stop": earlier_stop,
            "extra_rounds": extra_rounds,
            "r_k_ell_m": last_r_k if last_r_k is not None else "",
            "r_safe_ell_m": last_r_safe if last_r_safe is not None else "",
            "t_total_ms": round(t_level_ms + t_search_ms, 3),
        }
        if last_dists is None or int(getattr(last_dists, "shape", [0])[0]) == 0:
            return [], empty_info

        t_e0 = _t()
        results = _refine_vincenty(_topk_from_last())
        t_ell = (_t() - t_e0) * 1000
        t_total = t_level_ms + t_search_ms + t_ell
        info = dict(empty_info)
        info.update({
            "t_rank_ms": round(t_ell, 3),
            "t_ellipsoid_ms": round(t_stop + t_ell, 3),
            "planar_top_k": ";".join(mc for mc, _ in planar_top),
            "t_total_ms": round(t_total, 3),
        })
        return results[:k], info


def _parse_args() -> argparse.Namespace:
    """@returns Parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description="KNN V3.3: ellipsoid-metric stop + same-metric rank.")
    p.add_argument("--from-csv", default="")
    p.add_argument(
        "--reuse-pg-csv",
        default=os.path.join(_ROOT, "csv", "knn_v31_10k.csv"))
    p.add_argument(
        "--true-csv",
        default=os.path.join(_ROOT, "csv", "knn_r14_true_n500.csv"))
    p.add_argument(
        "--output",
        default=os.path.join(_ROOT, "csv", "knn_v33_ellstop.csv"))
    p.add_argument("--geom-type", default="Point")
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--k", default="3,5,10,20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=100.0)
    p.add_argument("--min-level", type=int, default=8)
    p.add_argument("--max-widen", type=int, default=5)
    p.add_argument("--max-cand-per-level", type=int, default=20000)
    p.add_argument("--start-level-max", type=int, default=13)
    p.add_argument("--coarse-top", type=int, default=50)
    p.add_argument(
        "--stop-kernel", default="local",
        choices=("local", "andoyer", "vincenty"),
        help="coverage metric; vincenty = full-window iterative each round")
    p.add_argument(
        "--rank-kernel", default="local",
        choices=("local", "andoyer", "vincenty"))
    p.add_argument("--no-refine-k", action="store_true")
    p.add_argument("--throwaway", type=int, default=5)
    p.add_argument("--no-poi-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    """
    Build the Neo4j prefix index and evaluate V3.3 against planar, PostGIS, and exhaustive
    references where available.
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
    print("GeoSOT KNN V3.3  ellipsoid-stop + same-metric rank")
    print("=" * 72)
    print(f"  HAS_NUMBA={HAS_NUMBA}  stop={args.stop_kernel}  "
          f"rank={args.rank_kernel}  refine_k={not args.no_refine_k}")
    print(f"  K={k_str}  (each K independently)")

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
            searcher = GeoSotKNNSearcherV33(
                None, geom_type=args.geom_type, verbose=False)
            pool = set(codes)
        else:
            print(f"  Connecting Neo4j {NEO4J_URI}  db={NEO4J_DATABASE}")
            t0 = time.perf_counter()
            driver, session = _connect_neo4j()
            print(f"  Connected ({(time.perf_counter() - t0) * 1000:.0f} ms)")
            searcher = GeoSotKNNSearcherV33(
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
            rank_kernel=args.rank_kernel,
            refine_k=not args.no_refine_k,
            stop_kernel=args.stop_kernel,
        )
        n_tw = min(args.throwaway, len(query_codes))
        for qc in query_codes[:n_tw]:
            searcher.knn_search(qc, k=k_max, **knn_kw)
        print(f"  Throwaway {n_tw} queries done (k={k_max})")

        fields = [
            "query_code", "k",
            "recall_vs_planar", "recall_vs_paper_gr", "recall_vs_pgs",
            "recall_vs_true", "top1_vs_planar", "top1_vs_pgs",
            "v33_top1_dist_m", "n_ellipsoid_eval", "t_ellipsoid_ms",
            "t_stop_ms", "t_gr_us", "gr_total_candidates", "gr_rounds",
            "extra_expand", "earlier_stop", "extra_rounds",
            "v33_top_k", "planar_top_k", "paper_gr_top_k", "pgs_top_k",
        ]
        stats: Dict[int, Dict[str, list]] = {
            kv: {
                "rec_planar": [], "rec_paper": [], "rec_pgs": [],
                "rec_true": [], "n_rank_diff": 0,
                "t_ell": [], "t_us": [], "n_eval": [],
                "t_stop": [], "n_cand": [], "rounds": [],
                "extra_expand": 0, "earlier_stop": 0,
            }
            for kv in k_list
        }
        n_rows = 0

        with open(out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
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
                    v33_top = [r[0] for r in gr[:kv]]
                    planar_top = _split_codes(info.get("planar_top_k") or "")
                    paper = paper_by_k.get(kv) or {}
                    paper_gr = paper.get("gr_top") or []
                    pgs_top = paper.get("pgs_top") or []
                    true_top = true_full[:kv]

                    r_pl = _set_recall(v33_top, planar_top, kv)
                    r_gr = _set_recall(v33_top, paper_gr, kv)
                    r_pg = _set_recall(v33_top, pgs_top, kv)
                    r_tr = _set_recall(v33_top, true_top, kv)
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
                    st["t_stop"].append(float(info.get("t_stop_ms") or 0.0))
                    st["n_cand"].append(int(info.get("total_candidates") or 0))
                    st["rounds"].append(int(info.get("rounds") or 0))
                    st["extra_expand"] += int(info.get("extra_expand") or 0)
                    st["earlier_stop"] += int(info.get("earlier_stop") or 0)

                    wr.writerow({
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
                            1 if v33_top and planar_top
                            and v33_top[0] == planar_top[0] else 0),
                        "top1_vs_pgs": (
                            1 if v33_top and pgs_top
                            and v33_top[0] == pgs_top[0] else 0),
                        "v33_top1_dist_m": (
                            round(gr[0][1], 6) if gr else ""),
                        "n_ellipsoid_eval": info.get("n_ellipsoid_eval", 0),
                        "t_ellipsoid_ms": round(t_ell, 3),
                        "t_stop_ms": round(float(info.get("t_stop_ms") or 0), 3),
                        "t_gr_us": round(t_us, 1),
                        "gr_total_candidates": info.get(
                            "total_candidates", 0),
                        "gr_rounds": info.get("rounds", 0),
                        "extra_expand": info.get("extra_expand", 0),
                        "earlier_stop": info.get("earlier_stop", 0),
                        "extra_rounds": info.get("extra_rounds", 0),
                        "v33_top_k": ";".join(v33_top),
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
            print(f"    mean set-agree vs freeze-window planar: "
                  f"{_pct(_mean(st['rec_planar']))}  "
                  f"(set differs on {st['n_rank_diff']}/{n_pl})")
            print(f"    mean set-agree vs paper V3.1 G3R:     "
                  f"{_pct(_mean(st['rec_paper']))}")
            print(f"    mean set-agree vs paper 8x PostGIS:   "
                  f"{_pct(_mean(st['rec_pgs']))}")
            print(f"    mean set-agree vs R1-4 ellipsoid:     "
                  f"{_pct(_mean(st['rec_true']))}  "
                  f"(n={len(st['rec_true'])})")
            if st["t_us"]:
                print(f"    median end-to-end:     "
                      f"{statistics.median(st['t_us']) / 1000.0:.3f} ms")
            if st["t_stop"]:
                print(f"    median stop-kernel:    "
                      f"{statistics.median(st['t_stop']):.3f} ms")
            if st["t_ell"]:
                print(f"    median stop+refine:    "
                      f"{statistics.median(st['t_ell']):.3f} ms")
            if st["n_cand"]:
                print(f"    median window size:    "
                      f"{statistics.median(st['n_cand']):.0f} candidates")
            if st["n_eval"]:
                print(f"    median stop eval:      "
                      f"{statistics.median(st['n_eval']):.0f} candidates")
            if st["rounds"]:
                print(f"    median rounds:         "
                      f"{statistics.median(st['rounds']):.0f}")
            print(f"    extra expand vs planar stop: "
                  f"{st['extra_expand']}/{n_pl}")
            print(f"    earlier stop than planar:    "
                  f"{st['earlier_stop']}/{n_pl}")
    finally:
        if session is not None:
            session.close()
        if driver is not None:
            driver.close()
            print("  Neo4j session closed")


if __name__ == "__main__":
    main()

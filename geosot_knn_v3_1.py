#!/usr/bin/env python3
# coding: utf-8
"""
KNN V3.1 search skeleton: planar L2 coverage, adaptive neighborhoods, and candidate-growth
brakes. V3.2 inherits this search logic for the paper's default KNN.

Replace V3's 0.7 heuristic with r_safe(L,w)=cell_margin(L)+((w-1)/2)*grid_m(L). Widen by odd
widths 3,5,7,... before coarsening, remove the fixed retry count, and stop excessive
candidate growth when enough candidates already exist. Coverage applies to the local planar
model; brakes and finite search limits can prevent complete recovery, so do not infer an
unconditional exact ellipsoidal Top-K guarantee.

CSV results remain compatible with V3. Important options: --max-widen (odd maximum width),
--max-cand-per-level (new-candidate brake), --start-level-max (optional forced minimum
initial level, typically 13..15), and --reuse-pg-csv (reuse reference results for the same
seed/query pool instead of executing PG). See --help for actual defaults.

Usage:
    python geosot_knn_v3_1.py --queries 100
    python geosot_knn_v3_1.py --queries 10 --k 3,5,10,20 --margin 200
    python geosot_knn_v3_1.py --queries 1000 --max-widen 7
    python geosot_knn_v3_1.py --queries 100 --max-cand-per-level 50000
"""

import csv
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Set, Tuple

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
    _build_cos_ref_matrix,
)
try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

from geosot_knn_v3 import (
    DEG_TO_M,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    _prefix_to_rowcol, _rowcol_to_prefix, _min_margin_m,
    find_safe_level,
    GeoSotKNNSearcherV3,
    POIPrefixCache,
    set_poi_prefix_index,
    get_poi_prefix_index,
    postgis_knn, postgis_knn_standard,
    print_summary, pctile, run_status_line,
)


# ============================================================================
# Utilities: w×w neighborhood (w odd >= 1)
# ============================================================================

def get_nxn_neighbors(prefix: str, n: int = 3) -> List[str]:
    """
    Return an n-by-n neighborhood including the center; domain boundaries may yield fewer than
    n^2 cells. n=1 selects the center, n=3 the standard 3x3 neighborhood, and n=5 a 5x5
    neighborhood.
    """
    if n < 1 or n % 2 == 0:
        raise ValueError(f"n must be a positive odd integer, got {n}")
    half = (n - 1) // 2
    lat_val, lon_val, level = _prefix_to_rowcol(prefix)
    hemisphere = prefix[1]
    max_val = (1 << level) - 1
    neighbors: List[str] = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            nr = lat_val + dr
            nc = lon_val + dc
            if 0 <= nr <= max_val and 0 <= nc <= max_val:
                neighbors.append(
                    _rowcol_to_prefix(nr, nc, level, hemisphere))
    return neighbors


# ============================================================================
# KNN Searcher V3.1
# ============================================================================

class GeoSotKNNSearcherV31(GeoSotKNNSearcherV3):
    """
    V3.1 searcher inheriting V3's Neo4j retrieval and geometry reranking. Start at
    find_safe_level with a 3x3 neighborhood; widen by two while allowed and enough candidates
    exist, otherwise coarsen and reset width to three. Test planar r_K<=r_safe; a per-round
    candidate-growth brake can terminate expansion when at least K candidates already exist.
    """

    def knn_search(
        self,
        query_code: str,
        k: int = 5,
        min_margin_m: float = 100.0,
        min_level: int = 8,
        coarse_top_n: int = 50,
        geometry_scoring: bool = True,
        max_widen: int = 5,
        max_candidates_per_level: int = 20000,
        start_level_max: int = 0,
        geo_workers: int = 0,
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        _t = time.perf_counter
        query_code = query_code.strip().upper()
        query_center = fast_code_to_center(query_code)
        cos_qlat = math.cos(math.radians(query_center[1]))
        cos2 = cos_qlat * cos_qlat

        # ---- 1. Determine safe level ----
        t0 = _t()
        safe_level_raw = find_safe_level(query_code, min_margin_m)
        # If find_safe_level returns too coarse (often near multi-level grid borders),
        # force start at start_level_max and coarsen via v3.1 adaptive expansion,
        # avoiding one 3×3 pull of millions of candidates from a very coarse level.
        if start_level_max > 0 and safe_level_raw < start_level_max:
            safe_level = start_level_max
        else:
            safe_level = safe_level_raw
        t_level_ms = (_t() - t0) * 1000

        # ---- 2. Adaptive expansion search ----
        t0 = _t()
        level = safe_level
        w = 3                        # Odd neighborhood width.
        n_prefixes = 0
        n_rounds = 0
        braked = False
        # Accumulate stage times (ms): prefix candidate retrieval and vectorized centroid coarse ranking.
        t_cand_query = 0.0
        t_coarse_rank = 0.0

        # sort only top-m: coverage needs k-th nearest; geometry rerank needs first coarse_top_n.
        rank_m = max(coarse_top_n, k)

        # --- candidate representation mode ---
        # index-native: when injected POIPrefixCache supports find_indices_batch /
        # rank_indices, candidates are row indices (ints) into the sorted code table. Prefix match
        # bisect [lo,hi) becomes np.arange; cross-round dedup via np.union1d; centroid
        # rank uses NumPy gather + argpartition — no string set, no
        # code->index dict lookup; avoids cache misses from million-scale dict/string hashing.
        # Representation-only optimization: candidate sets, coverage tests, and rankings match code mode.
        # code mode: bridge indexes (e.g. building _BuildingPOIBridge) without those methods fall back;
        # candidates use main_code string sets; behavior identical to before the change.
        cache = get_poi_prefix_index()
        use_idx = (
            _HAS_NP and cache is not None
            and hasattr(cache, "find_indices_batch")
            and hasattr(cache, "rank_indices")
            and cache.has_geom_type(self._geom_type))

        # candidates container: idx mode = np.int64 index array; code mode = str set.
        candidates: Set[str] = set()
        cand_arr = None  # type: ignore

        def _collect(level_: int, w_: int) -> int:
            """Query w×w neighborhood candidates into the set; return newly added count."""
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
            if use_idx:
                return 0 if cand_arr is None else int(cand_arr.size)
            return len(candidates)

        def _rank_all() -> List[Tuple[str, float]]:
            if use_idx:
                if cand_arr is None or cand_arr.size == 0:
                    return []
                return cache.rank_indices(
                    self._geom_type, cand_arr,
                    query_center[0], query_center[1], cos2, rank_m)
            if not candidates:
                return []
            # vectorized path (precomputed centroids + argpartition); bridges without rank_candidates
            # return None and fall back to pure Python.
            if cache is not None and hasattr(cache, "rank_candidates"):
                vec = cache.rank_candidates(
                    self._geom_type, candidates,
                    query_center[0], query_center[1], cos2, rank_m)
                if vec is not None:
                    return vec
            ranked: List[Tuple[str, float]] = []
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
            return ranked

        def _r_safe_m(level_: int, w_: int) -> float:
            """w×w inscribed-circle radius (m): strict geometric safe radius."""
            cell_margin = _min_margin_m(
                query_center[0], query_center[1],
                code_to_bbox(query_code[:level_ + 2]))
            grid_step = GRID_SIZES.get(level_, (0.0, 0.0))
            grid_m = grid_step[0] * DEG_TO_M * cos_qlat
            return cell_margin + ((w_ - 1) / 2.0) * grid_m

        def _coverage_ok(ranked: List[Tuple[str, float]],
                         level_: int, w_: int) -> bool:
            if len(ranked) < k:
                return False
            r_k_m = math.sqrt(ranked[k - 1][1]) * DEG_TO_M
            return r_k_m <= _r_safe_m(level_, w_)

        # --- Step A: initial 3×3 @safe_level ---
        _tq0 = _t()
        _collect(level, w)
        t_cand_query += (_t() - _tq0) * 1000
        n_rounds += 1
        _tr0 = _t()
        rank_list = _rank_all()
        t_coarse_rank += (_t() - _tr0) * 1000

        # --- Step B: adaptive expansion ---
        # hard cap: avoid pathological infinite loops (should not trigger)
        hard_cap_rounds = (max_widen // 2) + (safe_level - min_level) + 4

        while (not _coverage_ok(rank_list, level, w)
               and n_rounds < hard_cap_rounds):

            # decision: widen or drop level
            if w + 2 <= max_widen and _count() >= k:
                # widen at current level (cheap; avoid jumping to overly coarse levels)
                w += 2
            elif level > min_level:
                # drop level and reset w
                level -= 1
                w = 3
            else:
                # already at coarsest level; cannot expand further
                break

            _tq0 = _t()
            added = _collect(level, w)
            t_cand_query += (_t() - _tq0) * 1000
            n_rounds += 1

            if added:
                _tr0 = _t()
                rank_list = _rank_all()
                t_coarse_rank += (_t() - _tr0) * 1000

            # --- candidate-explosion brake ---
            # If this round adds candidates > threshold and we already have >= K,
            # coarse-level density is extreme; further expansion only worsens latency — stop.
            # (still record when strict coverage fails, but always return K results)
            if added > max_candidates_per_level and len(rank_list) >= k:
                braked = True
                break

        t_search_ms = (_t() - t0) * 1000
        final_level = level
        total_candidates = _count()

        if not rank_list:
            return [], {
                "safe_level": safe_level, "final_level": final_level,
                "total_candidates": 0, "prefixes": n_prefixes,
                "final_width": w, "rounds": n_rounds,
                "braked": int(braked),
                "t_level_ms": round(t_level_ms, 3),
                "t_search_ms": round(t_search_ms, 3),
                "t_cand_query_ms": round(t_cand_query, 3),
                "t_coarse_rank_ms": round(t_coarse_rank, 3),
                "t_featinfo_ms": 0.0,
                "t_rank_ms": 0.0,
                "t_total_ms": round(t_level_ms + t_search_ms, 3),
            }

        # ---- 3. (optional) geometry rerank ----
        t_rank = 0.0
        t_geo = 0.0
        t_featinfo = 0.0
        final_scored = rank_list

        if geometry_scoring:
            t0 = _t()
            top_codes = [mc for mc, _ in rank_list[:coarse_top_n]]

            _tf0 = _t()
            q_raw = self._batch_feature_info([query_code])
            q_info = q_raw.get(query_code) or {
                "geometry_type": "Point", "vertex_codes": [],
            }
            q_type = q_info["geometry_type"]
            c_raw = self._batch_feature_info(top_codes)
            t_featinfo = (_t() - _tf0) * 1000

            need_geo = (
                q_type != "Point" or
                any((c_raw.get(m) or {}).get("geometry_type", "Point")
                    != "Point" for m in top_codes))

            if need_geo:
                geo_scored, n_geo_eval, geo_timing = self._geometry_rerank(
                    query_code, q_info, top_codes, c_raw, rank_list,
                    k=k, geo_workers=geo_workers)
                t_geo = (_t() - t0) * 1000

                t_total = t_level_ms + t_search_ms + t_rank + t_geo
                return geo_scored[:k], {
                    "safe_level": safe_level,
                    "final_level": final_level,
                    "total_candidates": total_candidates,
                    "coarse_top_n": len(top_codes),
                    "geo_evaluated": n_geo_eval,
                    "prefixes": n_prefixes,
                    "final_width": w,
                    "rounds": n_rounds,
                    "braked": int(braked),
                    "has_geometry": True,
                    "t_level_ms": round(t_level_ms, 3),
                    "t_search_ms": round(t_search_ms, 3),
                    "t_cand_query_ms": round(t_cand_query, 3),
                    "t_coarse_rank_ms": round(t_coarse_rank, 3),
                    "t_featinfo_ms": round(t_featinfo, 3),
                    "t_rank_ms": round(t_rank + t_geo, 3),
                    "t_geo_ms": round(t_geo, 3),
                    "t_lcpc_prune_ms": geo_timing["t_lcpc_prune_ms"],
                    "t_exact_ms": geo_timing["t_exact_ms"],
                    "t_decode_ms": geo_timing["t_decode_ms"],
                    "t_total_ms": round(t_total, 3),
                }

            t_geo = (_t() - t0) * 1000

        # ---- output ----
        results = [(mc, math.sqrt(sq) * DEG_TO_M)
                   for mc, sq in final_scored[:k]]

        t_total = t_level_ms + t_search_ms + t_rank + t_geo
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
            "t_featinfo_ms": round(t_featinfo, 3),
            "t_rank_ms": round(t_rank + t_geo, 3),
            "t_total_ms": round(t_total, 3),
        }
        return results, info


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
    max_widen = 5
    max_cand_per_level = 20000
    start_level_max = 13  # Enabled by default to avoid million-candidate tails; set to 0 to disable.
    reuse_pg_csv = ""     # Reuse an existing PG CSV instead of rerunning PostGIS.
    geom_type = "Point"
    output_csv = "knn_v31_benchmark_results.csv"
    save_csv = True
    use_geometry = True
    verbose_detail = False
    run_pg_dedup = False
    skip_poi_cache = False  # --no-poi-cache disables the POI cache and retrieves candidates through Cypher.
    geo_workers = 0         # --geo-workers N: geometry reranking processes; 0 selects serial execution.
    flush_every = 50        # --flush-every N: CSV flush interval.
    skip_prefetch = False   # --no-prefetch: skip vertex-code prefetch.

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
        elif args[i] == "--max-widen" and i + 1 < len(args):
            max_widen = int(args[i + 1]); i += 2
        elif args[i] == "--max-cand-per-level" and i + 1 < len(args):
            max_cand_per_level = int(args[i + 1]); i += 2
        elif args[i] == "--start-level-max" and i + 1 < len(args):
            start_level_max = int(args[i + 1]); i += 2
        elif args[i] == "--reuse-pg-csv" and i + 1 < len(args):
            reuse_pg_csv = args[i + 1]; i += 2
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
        elif args[i] == "--no-poi-cache":
            skip_poi_cache = True; i += 1
        elif args[i] == "--geo-workers" and i + 1 < len(args):
            geo_workers = int(args[i + 1]); i += 2
        elif args[i] == "--flush-every" and i + 1 < len(args):
            flush_every = max(1, int(args[i + 1])); i += 2
        elif args[i] == "--no-prefetch":
            skip_prefetch = True; i += 1
        else:
            i += 1

    if max_widen < 3 or max_widen % 2 == 0:
        raise SystemExit("--max-widen must be odd and >= 3")

    k_max = max(k_list)
    k_str = ",".join(str(x) for x in k_list)

    random.seed(seed)

    W = 76
    print("=" * W)
    print("GeoSOT KNN V3.1: Strict L2 Coverage + Adaptive Neighborhood")
    print("=" * W)
    print(f"  Queries: {num_queries},  K: {k_str},  Seed: {seed}")
    print(f"  Margin: {margin:.0f} m,  Min level: {min_level}")
    print(f"  Geometry scoring: {'ON' if use_geometry else 'OFF'}"
          f"  (coarse_top_n: {coarse_top_n})")
    print(f"  V3.1 params: max_widen={max_widen}, "
          f"max_cand_per_level={max_cand_per_level:,}, "
          f"start_level_max={start_level_max}"
          + (" (disabled)" if start_level_max <= 0 else ""))
    print(f"  Engineering: geo_workers={geo_workers or 'OFF'}, "
          f"feat_prefetch={'OFF' if skip_prefetch else 'ON'}, "
          f"flush_every={flush_every}")
    console_mode = ("per-query detail (--verbose)" if verbose_detail
                    else "single-line progress (add --verbose for detail)")
    print(f"  Console: {console_mode}")
    print(
        "  PostGIS:  benchmark = Standard "
        "(GiST <-> KNN + ROW_NUMBER 按 main_code 去重);  "
        "慢速 DISTINCT_ON 对照查询: "
        f"{'ON (--pg-dedup)' if run_pg_dedup else 'OFF'}"
    )
    if reuse_pg_csv:
        print(f"  Reuse PG CSV: {reuse_pg_csv}  (PG not queried, "
              "pgs_* / t_pgs_us / t_pg_us from file)")

    # 1. Load POI codes from PostGIS
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

    # 2. Sample query points
    query_codes = random.sample(all_codes, min(num_queries, len(all_codes)))
    print(f"  Sampled {len(query_codes):,} query points")

    # 2.5 (optional) Load reuse-PG CSV and filter to covered queries
    pg_reuse_map: Dict[str, Dict] = {}
    if reuse_pg_csv:
        if not os.path.exists(reuse_pg_csv):
            print(f"[ERROR] reuse-pg-csv not found: {reuse_pg_csv}")
            pg_conn.close()
            return
        print(f"\n  Loading PG results from reuse CSV ...")
        t0 = time.perf_counter()
        # per (qc, k) row. We keep the row for k = k_max (full top list);
        # smaller k re-use the prefix because PG top-k is a prefix of top-K.
        n_rows_read = 0
        with open(reuse_pg_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_rows_read += 1
                qc_r = (row.get("query_code") or "").strip().upper()
                if not qc_r:
                    continue
                try:
                    k_r = int(row.get("k") or 0)
                except ValueError:
                    continue
                # Prefer the k_max row; fall back to the largest k seen.
                cur = pg_reuse_map.get(qc_r)
                if cur is None or k_r > cur["k"]:
                    top_codes = (row.get("pgs_top_k") or "").split(";")
                    top_codes = [c for c in top_codes if c]
                    try:
                        agree_val = int(row.get("pg_std_agree") or -1)
                    except ValueError:
                        agree_val = -1
                    pg_reuse_map[qc_r] = {
                        "k": k_r,
                        "pgs_top_codes": top_codes,
                        "pgs_top1_dist_m": float(row.get("pgs_top1_dist_m")
                                                  or 0.0),
                        "t_pgs_us": float(row.get("t_pgs_us") or 0.0),
                        "t_pg_us": (float(row["t_pg_us"])
                                    if row.get("t_pg_us") not in ("", None)
                                    else 0.0),
                        "pg_std_agree": agree_val,
                    }
        dt = (time.perf_counter() - t0)
        print(f"    Read {n_rows_read:,} rows → {len(pg_reuse_map):,} "
              f"unique queries  ({dt*1000:.0f} ms)")

        # Filter sampled queries to those present in reuse map, keeping
        # the original sampling order (seed-reproducible).
        before = len(query_codes)
        query_codes = [qc for qc in query_codes if qc in pg_reuse_map]
        after = len(query_codes)
        if after < before:
            print(f"    Filtered queries to reuse-covered subset: "
                  f"{after:,}/{before:,}  "
                  f"(dropped {before - after:,} not found in CSV)")
        if after == 0:
            print("[ERROR] No overlap between sampled queries and reuse CSV. "
                  "Check --seed / --queries match the reuse run.")
            pg_conn.close()
            return
        # Validate that each kept query has all top-k codes (>= k_max).
        short = sum(1 for qc in query_codes
                    if len(pg_reuse_map[qc]["pgs_top_codes"]) < k_max)
        if short > 0:
            print(f"    WARN: {short:,} queries have < k_max={k_max} "
                  "codes in reuse CSV — their recall@k_max will be biased.")

    # 3. Connect Neo4j + warmup
    print(f"\n[2/4] Connecting to Neo4j ...")
    t0 = time.perf_counter()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    session = driver.session(database=NEO4J_DATABASE)
    session.run("RETURN 1").consume()

    searcher = GeoSotKNNSearcherV31(session, geom_type=geom_type,
                                     verbose=False)
    print(f"  Connected ({(time.perf_counter() - t0) * 1000:.0f} ms)"
          f"  geom_type filter: {geom_type}")

    # ---- fairness setup: both sides RAM-resident + warm plan cache ----
    # Benchmark protocol: disclose that both sides use warm caches and resident data during measurement.
    # timed under this state; dual-side setup excluded from official latency. See G3R_Chinese.md §3.1.3.
    from bench_setup import prepare_pg_warm
    if not reuse_pg_csv:
        prepare_pg_warm(
            pg_conn,
            tables=["public.poi_new"],
            throwaway_queries=[
                ("SELECT main_code FROM public.poi_new LIMIT 200", None),
            ],
        )

    # GR client-side POI prefix cache (counterpart to PG pg_prewarm).
    # Pull all main_codes for the given geom_type into Python memory once,
    # removing Cypher/Bolt round-trips during KNN candidate generation; same protocol as
    # Use the same setup boundary as the V7 and reverse-overlay client-side indexes.
    if not skip_poi_cache:
        print()
        print("─" * W)
        print("  [bench_setup] GR client-side POI prefix cache "
              "(对应 PG pg_prewarm)")
        print("─" * W)
        poi_cache = POIPrefixCache(
            session, geom_types=[geom_type], verbose=True)
        set_poi_prefix_index(poi_cache)
        print("─" * W)
    else:
        poi_cache = None
        set_poi_prefix_index(None)
        print("  [poi_cache] 跳过 (--no-poi-cache, cold-cache 对照模式)")

    if not skip_prefetch:
        prefetch_codes = (
            poi_cache._sorted.get(geom_type, all_codes)
            if poi_cache is not None else all_codes)
        print(f"\n  [feat_cache] prefetching VC for {len(prefetch_codes):,} "
              f"{geom_type} codes ...")
        pf = searcher.prefetch_feature_info(prefetch_codes, verbose=True)
        print(f"  [feat_cache] {int(pf['n_cached']):,} cached in "
              f"{pf['elapsed_sec']:.1f}s")

    # GR+PG dual throwaway: warm both sides with 5 dummy queries; discard results
    n_throwaway = min(5, len(query_codes))
    if n_throwaway > 0:
        print(f"  Throwaway: GR/PG 双侧丢弃前 {n_throwaway} 条 query")
        for tw_qc in query_codes[:n_throwaway]:
            try:
                searcher.knn_search(
                    tw_qc, k=k_max, min_margin_m=margin,
                    min_level=min_level, coarse_top_n=coarse_top_n,
                    geometry_scoring=use_geometry,
                    max_widen=max_widen,
                    max_candidates_per_level=max_cand_per_level,
                    start_level_max=start_level_max,
                    geo_workers=geo_workers)
            except Exception:
                pass
            if not reuse_pg_csv:
                try:
                    postgis_knn_standard(pg_conn, tw_qc, k_max)
                except Exception:
                    pass
    print("  Warmup done")

    # 4. Run queries
    CSV_FIELDS = [
        "query_code", "k", "recall_at_k", "reciprocal_rank",
        "top1_match", "pg_std_agree",
        "pgs_top1_dist_m", "gr_top1_dist_m", "top1_err_m", "top1_err_pct",
        "gr_total_candidates", "gr_safe_level", "gr_final_level",
        "gr_prefixes", "t_gr_us", "t_pg_us", "t_pgs_us",
        "gr_top_k", "pgs_top_k",
        # V3.1 new fields (ignored by V3-compatible readers)
        "gr_final_width", "gr_rounds", "gr_braked",
        # Stage times (ms): safe level, candidate retrieval, centroid coarse ranking, feature metadata, search wall time.
        "t_gr_safelevel_ms", "t_gr_cand_query_ms", "t_gr_coarse_rank_ms",
        "t_gr_featinfo_ms", "t_gr_search_ms", "t_gr_total_ms",
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
            "recall": [], "rr": [], "t_gr": [], "t_pg": [], "t_pgs": [],
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
        s["rr"].append(float(prev.get("reciprocal_rank", 0.0) or 0.0))
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
    rows_since_flush = 0

    # extra V3.1 field stats
    brake_count = 0
    width_hist: Dict[int, int] = {}

    for r_idx, qc in enumerate(remaining):
        idx = skipped + r_idx

        # time each K independently: old code ran once at k_max then sliced, so all K latency
        # fields were identical; now measure true end-to-end G3R and PostGIS latency per K to
        # Report latency against K; smaller K can satisfy the safe-radius test earlier and require less overfetch.
        # and thus completes sooner).
        last_recall = 0.0
        last_t_gr_us = 0.0
        last_t_pgs_us = 0.0
        recall_by_k: Dict[int, float] = {}
        detail_by_k: Dict[int, Dict] = {}

        for kv in k_list:
            if (qc, kv) in done_keys:
                continue

            # --- G3R: time independently for current K ---
            tg0 = time.perf_counter()
            gr_results, gr_info = searcher.knn_search(
                qc, k=kv, min_margin_m=margin, min_level=min_level,
                coarse_top_n=coarse_top_n, geometry_scoring=use_geometry,
                max_widen=max_widen,
                max_candidates_per_level=max_cand_per_level,
                start_level_max=start_level_max,
                geo_workers=geo_workers)
            tg1 = time.perf_counter()
            t_gr_us = (tg1 - tg0) * 1e6

            # --- PostGIS: time independently for current K ---
            if reuse_pg_csv:
                # reuse mode only has k_max top codes and latency; cannot retime per K;
                # here only slice-reuse (G3R-only accuracy check, not timing comparison).
                info_pg = pg_reuse_map.get(qc)
                if info_pg is None:
                    continue
                top_codes = info_pg["pgs_top_codes"][:kv]
                top1_dist = info_pg["pgs_top1_dist_m"]
                pgs_results = [
                    (code, top1_dist if i == 0 else 0.0)
                    for i, code in enumerate(top_codes)
                ]
                t_pgs_us = info_pg["t_pgs_us"]
                t_pg_us = info_pg["t_pg_us"]
                pg_dedup_results = []
            else:
                tps0 = time.perf_counter()
                pgs_results = postgis_knn_standard(pg_conn, qc, kv)
                tps1 = time.perf_counter()
                t_pgs_us = (tps1 - tps0) * 1e6
                if not pgs_results:
                    continue
                t_pg_us = 0.0
                pg_dedup_results = []
                if run_pg_dedup:
                    tp0 = time.perf_counter()
                    pg_dedup_results = postgis_knn(pg_conn, qc, kv)
                    tp1 = time.perf_counter()
                    t_pg_us = (tp1 - tp0) * 1e6
                    if not pg_dedup_results:
                        continue

            # metrics for this K (already top-kv; no further slicing)
            pgs_set = {r[0] for r in pgs_results[:kv]}
            gr_set = {r[0] for r in gr_results[:kv]}
            recall = len(pgs_set & gr_set) / kv if kv > 0 else 0.0

            pg_top1_dist = pgs_results[0][1] if pgs_results else 0
            gr_top1_dist = gr_results[0][1] if gr_results else 0
            top1_err = abs(gr_top1_dist - pg_top1_dist)
            top1_p = ((top1_err / pg_top1_dist * 100)
                      if pg_top1_dist > 0.01 else 0)
            top1_match = 1 if (gr_results and pgs_results and
                               gr_results[0][0] == pgs_results[0][0]) else 0

            # MRR: reciprocal rank of PG true NN (top-1) in the ordered G3R list.
            # rank is 1-based; hit top-1 → RR=1; at position r → 1/r;
            # not in G3R top-k → RR=0. MRR = mean RR over all queries.
            reciprocal_rank = 0.0
            if pgs_results:
                pg_true_nn = pgs_results[0][0]
                for rank_i, (code, _) in enumerate(gr_results[:kv], start=1):
                    if code == pg_true_nn:
                        reciprocal_rank = 1.0 / rank_i
                        break

            std_agree_val = -1
            if run_pg_dedup and pg_dedup_results:
                pg_set_d = {r[0] for r in pg_dedup_results[:kv]}
                std_agree_val = 1 if pg_set_d == pgs_set else 0
            elif reuse_pg_csv:
                info_pg_agree = pg_reuse_map.get(qc)
                if info_pg_agree is not None:
                    std_agree_val = info_pg_agree.get("pg_std_agree", -1)

            s = per_k_stats[kv]
            s["recall"].append(recall)
            s["rr"].append(reciprocal_rank)
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

            fw = int(gr_info.get("final_width", 3))
            row_dict = {
                "query_code": qc,
                "k": kv,
                "recall_at_k": round(recall, 4),
                "reciprocal_rank": round(reciprocal_rank, 6),
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
                "t_pg_us": (round(t_pg_us, 1)
                            if (run_pg_dedup
                                or (reuse_pg_csv and t_pg_us > 0))
                            else ""),
                "t_pgs_us": round(t_pgs_us, 1),
                "gr_top_k": ";".join(r[0] for r in gr_results[:kv]),
                "pgs_top_k": ";".join(r[0] for r in pgs_results[:kv]),
                "gr_final_width": fw,
                "gr_rounds": int(gr_info.get("rounds", 0)),
                "gr_braked": int(gr_info.get("braked", 0)),
                "t_gr_safelevel_ms": gr_info.get("t_level_ms", 0.0),
                "t_gr_cand_query_ms": gr_info.get("t_cand_query_ms", 0.0),
                "t_gr_coarse_rank_ms": gr_info.get("t_coarse_rank_ms", 0.0),
                "t_gr_featinfo_ms": gr_info.get("t_featinfo_ms", 0.0),
                "t_gr_search_ms": gr_info.get("t_search_ms", 0.0),
                "t_gr_total_ms": gr_info.get("t_total_ms", 0.0),
            }
            rows_out.append(row_dict)
            s["rows"].append(row_dict)
            if csv_writer is not None:
                csv_writer.writerow(row_dict)
                rows_since_flush += 1
                if rows_since_flush >= flush_every:
                    csv_file.flush()
                    rows_since_flush = 0

            # V3.1 summary fields: accumulate once per query on k_max row to avoid double-counting
            if kv == k_max:
                brake_count += gr_info.get("braked", 0)
                width_hist[fw] = width_hist.get(fw, 0) + 1

            recall_by_k[kv] = recall
            detail_by_k[kv] = {
                "gr_info": gr_info, "t_gr_us": t_gr_us,
                "t_pgs_us": t_pgs_us, "t_pg_us": t_pg_us,
            }
            last_recall = recall
            last_t_gr_us = t_gr_us
            last_t_pgs_us = t_pgs_us

        kmax_s = per_k_stats[k_max]
        mean_r = (statistics.mean(kmax_s["recall"])
                  if kmax_s["recall"] else 0)
        run_status_line(
            idx + 1, total_n, k_max, last_recall, mean_r,
            last_t_gr_us, last_t_pgs_us, time.perf_counter() - t_start,
            status_pad)

        if verbose_detail and detail_by_k:
            sys.stdout.write("\n")
            print(f"  -- Query {idx + 1}/{total_n}: {qc}")
            for kv in k_list:
                if kv not in detail_by_k:
                    continue
                di = detail_by_k[kv]
                gi = di["gr_info"]
                line = (f"     [K={kv}] Recall={recall_by_k.get(kv, 0):.0%}  "
                        f"safe={gi.get('safe_level')}, "
                        f"final={gi.get('final_level')}, "
                        f"width={int(gi.get('final_width', 3))}, "
                        f"rounds={gi.get('rounds')}, "
                        f"braked={gi.get('braked')}, "
                        f"cand={gi.get('total_candidates')}  "
                        f"G3R {di['t_gr_us'] / 1000:.1f}ms / "
                        f"PG {di['t_pgs_us'] / 1000:.1f}ms")
                if run_pg_dedup:
                    line += f" / PGdedup {di['t_pg_us'] / 1000:.1f}ms"
                print(line)

    if total_n > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if csv_file is not None:
        if rows_since_flush > 0:
            csv_file.flush()
        csv_file.close()

    searcher.close()
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

    # ---- V3.1-specific stats ----
    new_runs = len(remaining)
    if new_runs > 0:
        print(f"\n  V3.1 Adaptive Stats:")
        SEP = "  " + "\u2500" * (W - 4)
        print(SEP)
        print(f"    Candidate brake triggered: "
              f"{brake_count:,}/{new_runs}  "
              f"({brake_count / new_runs * 100:.2f}%)")
        wh_line = "    Final neighborhood width: "
        for wv in sorted(width_hist):
            frac = width_hist[wv] / new_runs * 100
            wh_line += f"w={wv}: {width_hist[wv]:,} ({frac:.1f}%)   "
        print(wh_line.rstrip())

    # ---- Summary (per K) ----
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

        # ---- MRR (mean reciprocal rank of PG true NN in G3R list) ----
        if s["rr"]:
            mrr = statistics.mean(s["rr"])
            print(f"  MRR@{kv} (PG真NN在G3R排名): {mrr:.4f}")

        # ---- G3R stage timings: median, mean, P95 ----
        def _col(field):
            vals = []
            for r in s["rows"]:
                v = r.get(field)
                if v not in (None, ""):
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            return vals

        steps = [
            ("找安全层级  safelevel", "t_gr_safelevel_ms"),
            ("候选检索    cand_query", "t_gr_cand_query_ms"),
            ("粗排(质心)  coarse_rank", "t_gr_coarse_rank_ms"),
            ("特征信息    featinfo", "t_gr_featinfo_ms"),
            ("搜索墙钟    search", "t_gr_search_ms"),
        ]
        if any(_col(f) for _, f in steps):
            print("  G3R 分步 (median / mean / P95, ms):")
            for label, field in steps:
                vals = _col(field)
                if not vals:
                    continue
                med = statistics.median(vals)
                mean = statistics.mean(vals)
                p95 = pctile(vals, 95)
                print(f"      {label:<24} = {med:8.3f} / "
                      f"{mean:8.3f} / {p95:8.3f}")

    print()


if __name__ == "__main__":
    main()

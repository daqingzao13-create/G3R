#!/usr/bin/env python3
# coding: utf-8
"""
CSV linear-scan KNN with the paper's V3.2 reranker: planar stop, leading 8K local-ellipsoid
candidates, and Top-K Vincenty. Retrieve candidates by startswith over an unsorted list,
without a prefix index or Neo4j.

Usage:
    python knn_csv_scan_v32.py --k 20 --queries 0 --output csv/knn_csv_scan_v32_n10k.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from geosot_knn_v3 import set_poi_prefix_index
from geosot_knn_v3_2 import GeoSotKNNSearcherV32
from knn_csv_memory_baseline import _resolve
from knn_csv_scan_baseline import (
    CSVLinearCodeStore,
    _load_reuse_multi,
    _parse_k_list,
    _print_summary,
)
from knn_csv_memory_baseline import _load_poi_codes


def _parse_args() -> argparse.Namespace:
    """@returns Parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description="CSV linear-scan KNN with V3.2 ellipsoid rank.")
    p.add_argument(
        "--poi-csv",
        default=os.path.join(_ROOT, "csv", "neo4j_poi_point_codes.csv"))
    p.add_argument(
        "--output",
        default=os.path.join(_ROOT, "csv", "knn_csv_scan_v32_n10k.csv"))
    p.add_argument("--geom-type", default="Point")
    p.add_argument("--queries", type=int, default=0)
    p.add_argument("--k", default="20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=100.0)
    p.add_argument("--min-level", type=int, default=8)
    p.add_argument("--max-widen", type=int, default=5)
    p.add_argument("--max-cand-per-level", type=int, default=20000)
    p.add_argument("--start-level-max", type=int, default=13)
    p.add_argument("--coarse-top", type=int, default=50)
    p.add_argument(
        "--reuse-pg-csv",
        default=os.path.join(_ROOT, "csv", "knn_v31_10k.csv"))
    p.add_argument("--throwaway", type=int, default=1)
    p.add_argument("--flush-every", type=int, default=1)
    return p.parse_args()


def _search(searcher: GeoSotKNNSearcherV32, qc: str, k: int,
            args: argparse.Namespace):
    """
    @param searcher V3.2 searcher.
    @param qc Query code.
    @param k Top-K.
    @param args Expansion parameters.
    @returns (topk,info).
    """
    return searcher.knn_search(
        qc, k=k, min_margin_m=args.margin,
        min_level=args.min_level, coarse_top_n=args.coarse_top,
        geometry_scoring=False, max_widen=args.max_widen,
        max_candidates_per_level=args.max_cand_per_level,
        start_level_max=args.start_level_max, geo_workers=0,
        max_ellipsoid_n=-1, rank_kernel="local", refine_k=True)


def main() -> None:
    """Collect candidates by linear scan, rank with V3.2, and compare with PostGIS."""
    args = _parse_args()
    k_list = _parse_k_list(args.k)
    k_max = max(k_list)
    poi_csv = _resolve(args.poi_csv)
    reuse_path = _resolve(args.reuse_pg_csv)
    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if not os.path.isfile(poi_csv):
        raise SystemExit(f"missing POI CSV: {poi_csv}")
    if not os.path.isfile(reuse_path):
        raise SystemExit(f"missing reuse CSV: {reuse_path}")

    print("=" * 72)
    print("CSV linear-scan KNN  V3.2 rank (no Neo4j, no prefix index)")
    print("=" * 72)
    codes = _load_poi_codes(poi_csv, args.geom_type)
    print(f"  POI codes: {len(codes):,}")
    order, pg_map = _load_reuse_multi(reuse_path, set(k_list))
    pool = set(codes)
    kept = [q for q in order if q in pool]
    n_take = len(kept) if args.queries <= 0 else min(args.queries, len(kept))
    query_codes = kept[:n_take]
    print(f"  queries: {len(query_codes):,}  K={','.join(str(x) for x in k_list)}")
    n_jobs = len(query_codes) * len(k_list)

    store = CSVLinearCodeStore(args.geom_type, codes)
    set_poi_prefix_index(store)
    searcher = GeoSotKNNSearcherV32(
        None, geom_type=args.geom_type, verbose=False)

    fields = [
        "query_code", "k", "recall_at_k", "top1_match",
        "gr_top1_dist_m", "gr_total_candidates",
        "t_gr_us", "gr_top_k", "pgs_top_k", "mode",
    ]
    done_keys = set()
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        with open(out, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qc = (row.get("query_code") or "").strip().upper()
                try:
                    kr = int(row.get("k") or 0)
                except ValueError:
                    continue
                if qc:
                    done_keys.add((qc, kr))
        print(f"  resume: {len(done_keys):,} rows in {out}")

    n_tw = min(args.throwaway, len(query_codes))
    for qc in query_codes[:n_tw]:
        _search(searcher, qc, k_max, args)
    print(f"  throwaway {n_tw} x K={k_max}")

    t_by_k: Dict[int, List[float]] = defaultdict(list)
    rec_by_k: Dict[int, List[float]] = defaultdict(list)
    top1_by_k: Dict[int, List[int]] = defaultdict(list)
    t_pgs_by_k: Dict[int, List[float]] = defaultdict(list)
    n_ok = 0
    mode = "a" if done_keys else "w"
    with open(out, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        t_run = time.perf_counter()
        done_work = len(done_keys)
        for qc in query_codes:
            for kv in k_list:
                if (qc, kv) in done_keys:
                    continue
                t0 = time.perf_counter()
                gr, info = _search(searcher, qc, kv, args)
                t_us = (time.perf_counter() - t0) * 1e6
                t_by_k[kv].append(t_us)
                gr_top = [r[0] for r in gr[:kv]]
                pg = pg_map.get((qc, kv), {})
                pgs_top = (pg.get("pgs_top") or [])[:kv]
                recall = ""
                top1 = ""
                if pgs_top:
                    rec = len(set(gr_top) & set(pgs_top)) / kv
                    rec_by_k[kv].append(rec)
                    t_pgs_by_k[kv].append(float(pg.get("t_pgs_us") or 0.0))
                    hit1 = 1 if (gr_top and pgs_top
                                 and gr_top[0] == pgs_top[0]) else 0
                    top1_by_k[kv].append(hit1)
                    recall = round(rec, 4)
                    top1 = hit1
                w.writerow({
                    "query_code": qc, "k": kv,
                    "recall_at_k": recall, "top1_match": top1,
                    "gr_top1_dist_m": round(gr[0][1], 4) if gr else "",
                    "gr_total_candidates": info.get("total_candidates", 0),
                    "t_gr_us": round(t_us, 1),
                    "gr_top_k": ";".join(gr_top),
                    "pgs_top_k": ";".join(pgs_top),
                    "mode": "csv_scan_v32",
                })
                n_ok += 1
                done_work += 1
                if n_ok % max(1, args.flush_every) == 0:
                    f.flush()
                elapsed = time.perf_counter() - t_run
                rate = n_ok / elapsed if elapsed else 0
                eta = (n_jobs - done_work) / rate if rate else 0
                med = (statistics.median(t_by_k[k_max]) / 1000.0
                       if t_by_k.get(k_max) else 0.0)
                print(f"\r  {done_work:,}/{n_jobs:,}  {rate:.2f}/s  "
                      f"med {med:.1f} ms  ETA {eta / 60.0:.1f} min",
                      end="", flush=True)
        print()
    set_poi_prefix_index(None)
    print(f"wrote {n_ok:,} new rows -> {out}")
    _print_summary(k_list, t_by_k, rec_by_k, top1_by_k, t_pgs_by_k)


if __name__ == "__main__":
    main()

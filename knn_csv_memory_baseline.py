#!/usr/bin/env python3
# coding: utf-8
"""
No-graph V3.1 memory baseline and CSV helper module. Build POIPrefixCache.from_codes from
exported centroid codes using sorted-prefix bisect and centroid arrays; query with
session=None so an unavailable cache path cannot fall back to Neo4j. For the final V3.2
indexed CSV arm, use geosot_knn_v3_2.py --from-csv.

Usage:
    python export_neo4j_poi_codes.py
    python knn_csv_memory_baseline.py --poi-csv csv/neo4j_poi_point_codes.csv --queries 10000 --k 20 --seed 42 --reuse-pg-csv csv/knn_v31_10k.csv --output csv/knn_csv_memory_n10k.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Tuple

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from geosot_knn_v3 import (
    POIPrefixCache,
    set_poi_prefix_index,
)
from geosot_knn_v3_1 import GeoSotKNNSearcherV31
from geosot_spatial_reasoning import is_valid_g0_code


def _parse_args() -> argparse.Namespace:
    """@returns Parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description="KNN from exported POI codes, no Neo4j at query time.")
    p.add_argument(
        "--poi-csv",
        default=os.path.join(_ROOT, "csv", "neo4j_poi_point_codes.csv"),
        help="CSV from export_neo4j_poi_codes.py")
    p.add_argument(
        "--output",
        default=os.path.join(_ROOT, "csv", "knn_csv_memory_n10k.csv"))
    p.add_argument("--geom-type", default="Point")
    p.add_argument("--queries", type=int, default=10000)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=100.0)
    p.add_argument("--min-level", type=int, default=8)
    p.add_argument("--max-widen", type=int, default=5)
    p.add_argument("--max-cand-per-level", type=int, default=20000)
    p.add_argument("--start-level-max", type=int, default=13)
    p.add_argument("--coarse-top", type=int, default=50)
    p.add_argument(
        "--reuse-pg-csv", default="",
        help="optional warm-run CSV; reuse query_code list and PostGIS top-K")
    p.add_argument("--throwaway", type=int, default=5)
    return p.parse_args()


def _load_poi_codes(path: str, geom_type: str) -> List[str]:
    """
    @param path Exported code CSV.
    @param geom_type Filter by geom_type when present.
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


def _load_reuse(path: str, k: int) -> Tuple[List[str], Dict[str, Dict]]:
    """
    @param path Existing knn_v31-format CSV.
    @param k K value to select.
    @returns (query order, query_code -> PG row).
    """
    order: List[str] = []
    seen = set()
    by_qc: Dict[str, Dict] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qc = (row.get("query_code") or "").strip().upper()
            if not qc:
                continue
            try:
                kr = int(row.get("k") or 0)
            except ValueError:
                continue
            if kr != k:
                continue
            if qc not in seen:
                seen.add(qc)
                order.append(qc)
            pgs = [c for c in (row.get("pgs_top_k") or "").split(";") if c]
            by_qc[qc] = {
                "pgs_top": pgs,
                "t_pgs_us": float(row.get("t_pgs_us") or 0.0),
                "pgs_top1_dist_m": float(row.get("pgs_top1_dist_m") or 0.0),
            }
    return order, by_qc


def _resolve(path: str) -> str:
    """Resolve a user path relative to cwd; if absent, try the script directory."""
    if not path:
        return path
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    alt = os.path.join(_ROOT, path)
    return alt if os.path.isfile(alt) else path


def main() -> None:
    """Build a memory prefix index and run KNN without Neo4j."""
    args = _parse_args()
    poi_csv = _resolve(args.poi_csv)
    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if not os.path.isfile(poi_csv):
        raise SystemExit(
            f"missing POI CSV: {poi_csv}\n"
            "run export_neo4j_poi_codes.py first")

    print("=" * 72)
    print("CSV memory KNN baseline (no Neo4j at query time)")
    print("=" * 72)
    codes = _load_poi_codes(poi_csv, args.geom_type)
    print(f"  POI codes: {len(codes):,}  from {poi_csv}")
    if len(codes) < args.k + 1:
        raise SystemExit("not enough POI codes")

    random.seed(args.seed)
    pg_map: Dict[str, Dict] = {}
    if args.reuse_pg_csv:
        reuse_path = _resolve(args.reuse_pg_csv)
        order, pg_map = _load_reuse(reuse_path, args.k)
        pool = set(codes)
        kept = [q for q in order if q in pool]
        dropped = len(order) - len(kept)
        query_codes = kept[: args.queries]
        print(f"  reuse CSV: {reuse_path}")
        print(f"  reuse queries kept: {len(query_codes):,} / {len(order):,} "
              f"(dropped not in dump: {dropped:,})")
        if not query_codes:
            raise SystemExit("reuse CSV queries not in exported POI codes")
    else:
        query_codes = random.sample(codes, min(args.queries, len(codes)))
        print(f"  sampled queries: {len(query_codes):,}  seed={args.seed}")

    cache = POIPrefixCache.from_codes(args.geom_type, codes, verbose=True)
    set_poi_prefix_index(cache)

    searcher = GeoSotKNNSearcherV31(
        None, geom_type=args.geom_type, verbose=False)
    default_info = {"geometry_type": args.geom_type, "vertex_codes": []}
    for mc in codes:
        searcher._feat_cache[mc] = default_info

    n_tw = min(args.throwaway, len(query_codes))
    for qc in query_codes[:n_tw]:
        searcher.knn_search(
            qc, k=args.k, min_margin_m=args.margin,
            min_level=args.min_level, coarse_top_n=args.coarse_top,
            geometry_scoring=False, max_widen=args.max_widen,
            max_candidates_per_level=args.max_cand_per_level,
            start_level_max=args.start_level_max, geo_workers=0)

    fields = [
        "query_code", "k", "recall_at_k", "top1_match",
        "gr_top1_dist_m", "gr_total_candidates",
        "gr_safe_level", "gr_final_level", "t_gr_us", "t_gr_total_ms",
        "gr_top_k", "pgs_top_k", "mode",
    ]
    t_list: List[float] = []
    rec_list: List[float] = []
    n_ok = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        t_run = time.perf_counter()
        for i, qc in enumerate(query_codes):
            t0 = time.perf_counter()
            gr, info = searcher.knn_search(
                qc, k=args.k, min_margin_m=args.margin,
                min_level=args.min_level, coarse_top_n=args.coarse_top,
                geometry_scoring=False, max_widen=args.max_widen,
                max_candidates_per_level=args.max_cand_per_level,
                start_level_max=args.start_level_max, geo_workers=0)
            t_us = (time.perf_counter() - t0) * 1e6
            t_list.append(t_us)
            gr_top = [r[0] for r in gr[: args.k]]
            pgs_top = []
            recall = ""
            top1 = ""
            pg = pg_map.get(qc)
            if pg and pg["pgs_top"]:
                pgs_top = pg["pgs_top"][: args.k]
                rec = (len(set(gr_top) & set(pgs_top)) / args.k
                       if args.k else 0.0)
                rec_list.append(rec)
                recall = round(rec, 4)
                top1 = 1 if (gr_top and pgs_top and gr_top[0] == pgs_top[0]
                             ) else 0
            w.writerow({
                "query_code": qc,
                "k": args.k,
                "recall_at_k": recall,
                "top1_match": top1,
                "gr_top1_dist_m": round(gr[0][1], 4) if gr else "",
                "gr_total_candidates": info.get("total_candidates", 0),
                "gr_safe_level": info.get("safe_level", 0),
                "gr_final_level": info.get("final_level", 0),
                "t_gr_us": round(t_us, 1),
                "t_gr_total_ms": round(info.get("t_total_ms", 0.0), 4),
                "gr_top_k": ";".join(gr_top),
                "pgs_top_k": ";".join(pgs_top),
                "mode": "csv_memory",
            })
            n_ok += 1
            if (i + 1) % 200 == 0 or i + 1 == len(query_codes):
                elapsed = time.perf_counter() - t_run
                rate = (i + 1) / elapsed if elapsed else 0
                print(f"\r  {i + 1:,}/{len(query_codes):,}  "
                      f"{rate:.1f} q/s", end="", flush=True)
        print()

    print(f"wrote {n_ok:,} rows -> {out}")
    if t_list:
        med = statistics.median(t_list) / 1000.0
        print(f"  median latency: {med:.3f} ms  "
              f"(warm table D.1 K=20 was 0.63 ms)")
    if rec_list:
        print(f"  mean Recall@{args.k} vs reused PostGIS: "
              f"{statistics.mean(rec_list):.4%}")


if __name__ == "__main__":
    main()

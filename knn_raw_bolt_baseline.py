#!/usr/bin/env python3
# coding: utf-8
"""
Unmaterialized Bolt KNN using the V3.1 kernel. Do not construct POIPrefixCache or centroid
arrays; query each neighborhood prefix with Neo4j STARTS WITH. Disable vertex prefetch and
geometry reranking to match the corresponding V3.1/CSV protocol. This raw arm differs in
ranking kernel from the final V3.2 warm arm, as disclosed in the appendix.

Historical 10k-query runs at about 270 ms median imply tens of minutes of wall time; actual
performance depends on the environment.

Usage:
    python knn_raw_bolt_baseline.py --queries 20 --k 20 --reuse-pg-csv csv/knn_v31_10k.csv --output csv/knn_raw_bolt_smoke.csv
    python knn_raw_bolt_baseline.py --queries 10000 --k 20 --reuse-pg-csv csv/knn_v31_10k.csv --output csv/knn_raw_bolt_n10k.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from typing import List

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from neo4j import GraphDatabase

from geosot_knn_v3 import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    set_poi_prefix_index,
)
from geosot_knn_v3_1 import GeoSotKNNSearcherV31
from knn_csv_memory_baseline import _load_reuse, _resolve


def _parse_args() -> argparse.Namespace:
    """@returns Parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description="KNN with Neo4j STARTS WITH, no prefix-index materialization.")
    p.add_argument(
        "--output",
        default=os.path.join(_ROOT, "csv", "knn_raw_bolt_n10k.csv"))
    p.add_argument("--geom-type", default="Point")
    p.add_argument("--queries", type=int, default=10000)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--margin", type=float, default=100.0)
    p.add_argument("--min-level", type=int, default=8)
    p.add_argument("--max-widen", type=int, default=5)
    p.add_argument("--max-cand-per-level", type=int, default=20000)
    p.add_argument("--start-level-max", type=int, default=13)
    p.add_argument("--coarse-top", type=int, default=50)
    p.add_argument(
        "--reuse-pg-csv",
        default=os.path.join(_ROOT, "csv", "knn_v31_10k.csv"),
        help="warm-run CSV; reuse query_code order and PostGIS top-K")
    p.add_argument("--throwaway", type=int, default=2)
    p.add_argument("--flush-every", type=int, default=10)
    return p.parse_args()


def _search(searcher: GeoSotKNNSearcherV31, qc: str, args: argparse.Namespace):
    """
    @param searcher V3.1 searcher.
    @param qc Query centroid code.
    @param args Expansion settings shared with CSV arms.
    @returns (topk,info).
    """
    return searcher.knn_search(
        qc, k=args.k, min_margin_m=args.margin,
        min_level=args.min_level, coarse_top_n=args.coarse_top,
        geometry_scoring=False, max_widen=args.max_widen,
        max_candidates_per_level=args.max_cand_per_level,
        start_level_max=args.start_level_max, geo_workers=0)


def main() -> None:
    """Disable prefix materialization and run the same query sequence as the CSV arms through Neo4j."""
    args = _parse_args()
    reuse_path = _resolve(args.reuse_pg_csv)
    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if not os.path.isfile(reuse_path):
        raise SystemExit(f"missing reuse CSV: {reuse_path}")

    order, pg_map = _load_reuse(reuse_path, args.k)
    query_codes = order[: args.queries]
    if not query_codes:
        raise SystemExit("no queries in reuse CSV")

    print("=" * 72)
    print("Raw Bolt KNN (no POIPrefixCache, Cypher STARTS WITH)")
    print("=" * 72)
    print(f"  Neo4j {NEO4J_URI}  db={NEO4J_DATABASE}")
    print(f"  queries: {len(query_codes):,}  from {reuse_path}")
    print("  prefix index: OFF")
    print("  feat prefetch: OFF")
    print("  geometry scoring: OFF")
    print("  Table D.1 raw K=20 median was 270.79 ms "
          "(~45-90 min for 10k)")

    set_poi_prefix_index(None)
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    fields = [
        "query_code", "k", "recall_at_k", "top1_match",
        "gr_top1_dist_m", "gr_total_candidates",
        "gr_safe_level", "gr_final_level",
        "t_gr_us", "t_gr_total_ms", "t_gr_cand_query_ms",
        "gr_top_k", "pgs_top_k", "mode",
    ]
    t_list: List[float] = []
    rec_list: List[float] = []
    n_ok = 0
    try:
        with drv.session(database=NEO4J_DATABASE) as sess:
            sess.run("RETURN 1").consume()
            searcher = GeoSotKNNSearcherV31(
                sess, geom_type=args.geom_type, verbose=False)

            n_tw = min(args.throwaway, len(query_codes))
            for qc in query_codes[:n_tw]:
                _search(searcher, qc, args)
            print(f"  throwaway done ({n_tw})")

            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                t_run = time.perf_counter()
                for i, qc in enumerate(query_codes):
                    t0 = time.perf_counter()
                    gr, info = _search(searcher, qc, args)
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
                        top1 = 1 if (
                            gr_top and pgs_top and gr_top[0] == pgs_top[0]
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
                        "t_gr_cand_query_ms": round(
                            info.get("t_cand_query_ms", 0.0), 4),
                        "gr_top_k": ";".join(gr_top),
                        "pgs_top_k": ";".join(pgs_top),
                        "mode": "raw_bolt",
                    })
                    n_ok += 1
                    if (i + 1) % max(1, args.flush_every) == 0:
                        f.flush()
                    elapsed = time.perf_counter() - t_run
                    rate = (i + 1) / elapsed if elapsed else 0
                    eta = (len(query_codes) - i - 1) / rate if rate else 0
                    med_so_far = statistics.median(t_list) / 1000.0
                    print(
                        f"\r  {i + 1:,}/{len(query_codes):,}  "
                        f"{rate:.2f} q/s  median {med_so_far:.1f} ms  "
                        f"ETA {eta:.0f}s",
                        end="", flush=True)
                print()
    finally:
        drv.close()
        set_poi_prefix_index(None)

    print(f"wrote {n_ok:,} rows -> {out}")
    if t_list:
        med = statistics.median(t_list) / 1000.0
        mean = statistics.mean(t_list) / 1000.0
        print(f"  median latency: {med:.2f} ms")
        print(f"  mean latency:   {mean:.2f} ms")
        print("  compare: CSV memory 0.51 ms | G3R-warm 0.63 ms | "
              "Table D.1 raw 270.79 ms | PostGIS 2.75 ms")
    if rec_list:
        print(f"  mean Recall@{args.k} vs reused PostGIS: "
              f"{statistics.mean(rec_list):.4%}")


if __name__ == "__main__":
    main()

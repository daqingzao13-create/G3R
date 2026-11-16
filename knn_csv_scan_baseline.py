#!/usr/bin/env python3
# coding: utf-8
"""
Legacy V3.1 CSV linear-scan baseline without Neo4j or prefix materialization. Scan an
unsorted code list with startswith, without bisect or precomputed centroids. Call knn_search
independently for each K because coverage depends on K. Reuse PostGIS lists/timings from the
same-query reference CSV. The final matching-ranker arm is knn_csv_scan_v32.py.

Usage:
    python knn_csv_scan_baseline.py --poi-csv csv/neo4j_poi_point_codes.csv --reuse-pg-csv csv/knn_v31_10k.csv --k 3,5,10,20 --queries 0 --output csv/knn_csv_scan_n10k.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from geosot_knn_v3 import set_poi_prefix_index
from geosot_knn_v3_1 import GeoSotKNNSearcherV31
from knn_csv_memory_baseline import _load_poi_codes, _resolve


class CSVLinearCodeStore:
    """
    Linear STARTS WITH over unsorted centroid codes. Implement
    has_geom_type/find_by_prefixes_batch only; intentionally omit
    find_indices_batch/rank_candidates so V3.1 cannot use the index-native path.
    """

    def __init__(self, geom_type: str, codes: Iterable[str]) -> None:
        """
        @param geom_type Geometry-type key.
        @param codes Codes in input order; deduplicate without sorting.
        """
        self._geom_type = geom_type
        self._codes = list(codes)

    def has_geom_type(self, geom_type: str) -> bool:
        """
        @param geom_type Requested geometry type.
        @returns Whether that type has codes.
        """
        return geom_type == self._geom_type and bool(self._codes)

    def find_by_prefixes_batch(
        self,
        geom_type: str,
        prefixes: Iterable[str],
        exclude_code: Optional[str] = None,
    ) -> Set[str]:
        """
        Scan once and retain codes matching any neighborhood prefix.

        @param geom_type Geometry type.
        @param prefixes Neighborhood prefixes for this round.
        @param exclude_code Query code to exclude.
        @returns Matching main_codes.
        """
        if geom_type != self._geom_type:
            return set()
        prefs = tuple(str(p) for p in prefixes if p)
        if not prefs:
            return set()
        exc = (exclude_code or "").strip().upper()
        hit: Set[str] = set()
        for code in self._codes:
            if code == exc:
                continue
            if code.startswith(prefs):
                hit.add(code)
        return hit


def _parse_k_list(text: str) -> List[int]:
    """Parse comma-separated K values such as 3,5,10,20 and return a sorted unique list."""
    ks = sorted({int(x) for x in text.split(",") if x.strip()})
    if not ks or min(ks) < 1:
        raise SystemExit("--k must be positive integers, e.g. 3,5,10,20")
    return ks


def _parse_args() -> argparse.Namespace:
    """@returns Parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description="CSV linear-scan KNN, all K, vs PostGIS from reuse CSV.")
    p.add_argument(
        "--poi-csv",
        default=os.path.join(_ROOT, "csv", "neo4j_poi_point_codes.csv"))
    p.add_argument(
        "--output",
        default=os.path.join(_ROOT, "csv", "knn_csv_scan_n10k.csv"))
    p.add_argument("--geom-type", default="Point")
    p.add_argument(
        "--queries", type=int, default=0,
        help="max unique queries; 0 = all queries in the reuse CSV")
    p.add_argument(
        "--k", default="3,5,10,20",
        help="comma-separated K list (paper default 3,5,10,20)")
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


def _load_reuse_multi(
    path: str, k_set: Set[int],
) -> Tuple[List[str], Dict[Tuple[str, int], Dict]]:
    """
    @param path Paper KNN CSV with pgs_top_k/t_pgs_us/gr_top_k.
    @param k_set Required K values.
    @returns (query appearance order, (query_code,k) -> PostGIS/warm row).
    """
    order: List[str] = []
    seen: Set[str] = set()
    by: Dict[Tuple[str, int], Dict] = {}
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
            pgs = [c for c in (row.get("pgs_top_k") or "").split(";") if c]
            grw = [c for c in (row.get("gr_top_k") or "").split(";") if c]
            by[(qc, kr)] = {
                "pgs_top": pgs,
                "warm_top": grw,
                "t_pgs_us": float(row.get("t_pgs_us") or 0.0),
                "t_gr_warm_us": float(row.get("t_gr_us") or 0.0),
            }
    return order, by


def _search(
    searcher: GeoSotKNNSearcherV31,
    qc: str,
    k: int,
    args: argparse.Namespace,
):
    """
    @param searcher V3.1 searcher.
    @param qc Query code.
    @param k Current K.
    @param args Expansion parameters.
    @returns (topk,info).
    """
    return searcher.knn_search(
        qc, k=k, min_margin_m=args.margin,
        min_level=args.min_level, coarse_top_n=args.coarse_top,
        geometry_scoring=False, max_widen=args.max_widen,
        max_candidates_per_level=args.max_cand_per_level,
        start_level_max=args.start_level_max, geo_workers=0)


def _print_summary(
    k_list: List[int],
    t_by_k: Dict[int, List[float]],
    rec_by_k: Dict[int, List[float]],
    top1_by_k: Dict[int, List[int]],
    t_pgs_by_k: Dict[int, List[float]],
) -> None:
    """
    @param k_list K values.
    @param t_by_k CSV-scan microseconds.
    @param rec_by_k Recall@K against PG.
    @param top1_by_k Top-1 agreement with PG.
    @param t_pgs_by_k Reused PG microseconds.
    """
    print()
    print("=== per-K vs PostGIS (reuse csv/knn_v31_10k.csv) ===")
    print(f"  {'K':>4}  {'n':>6}  {'scan_med_ms':>12}  {'scan_mean_ms':>12}  "
          f"{'pgs_med_ms':>11}  {'Recall@K':>10}  {'Top1':>8}")
    for k in k_list:
        ts = t_by_k.get(k) or []
        rs = rec_by_k.get(k) or []
        t1 = top1_by_k.get(k) or []
        pg = t_pgs_by_k.get(k) or []
        if not ts:
            continue
        rec_s = f"{statistics.mean(rs):.4%}" if rs else "—"
        t1_s = f"{statistics.mean(t1):.4%}" if t1 else "—"
        pg_s = (f"{statistics.median(pg) / 1000.0:11.2f}"
                if pg else f"{'—':>11}")
        print(
            f"  {k:4d}  {len(ts):6,d}  "
            f"{statistics.median(ts) / 1000.0:12.2f}  "
            f"{statistics.mean(ts) / 1000.0:12.2f}  "
            f"{pg_s}  {rec_s:>10}  {t1_s:>8}")


def main() -> None:
    """
    Read codes from CSV, evaluate each K by linear scanning, and compare with the PostGIS Top-K
    reference.
    """
    args = _parse_args()
    k_list = _parse_k_list(args.k)
    k_max = max(k_list)
    k_set = set(k_list)
    poi_csv = _resolve(args.poi_csv)
    reuse_path = _resolve(args.reuse_pg_csv)
    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if not os.path.isfile(poi_csv):
        raise SystemExit(
            f"missing POI CSV: {poi_csv}\n"
            "run export_neo4j_poi_codes.py first")
    if not os.path.isfile(reuse_path):
        raise SystemExit(f"missing reuse CSV: {reuse_path}")

    print("=" * 72)
    print("CSV linear-scan KNN (no Neo4j, no prefix index)")
    print("=" * 72)
    t_load = time.perf_counter()
    codes = _load_poi_codes(poi_csv, args.geom_type)
    print(f"  POI codes: {len(codes):,}  from {poi_csv}  "
          f"(load {(time.perf_counter() - t_load) * 1000:.0f} ms, unsorted)")
    if len(codes) < k_max + 1:
        raise SystemExit("not enough POI codes")

    random.seed(args.seed)
    order, pg_map = _load_reuse_multi(reuse_path, k_set)
    pool = set(codes)
    kept = [q for q in order if q in pool]
    n_take = len(kept) if args.queries <= 0 else min(args.queries, len(kept))
    query_codes = kept[:n_take]
    print(f"  reuse CSV: {reuse_path}")
    print(f"  queries: {len(query_codes):,} / {len(order):,} unique  "
          f"K={','.join(str(x) for x in k_list)}")
    if not query_codes:
        raise SystemExit("reuse CSV queries not in exported POI codes")

    print("  prefix index: OFF (linear startswith over the CSV list)")
    print("  Neo4j: OFF")
    print("  PostGIS: reuse Top-K + t_pgs_us from the warm CSV")
    n_jobs = len(query_codes) * len(k_list)
    print(f"  planned searches: {n_jobs:,}  "
          f"(~hours if each scan is hundreds of ms)")

    store = CSVLinearCodeStore(args.geom_type, codes)
    set_poi_prefix_index(store)
    searcher = GeoSotKNNSearcherV31(
        None, geom_type=args.geom_type, verbose=False)

    fields = [
        "query_code", "k", "recall_at_k", "top1_match",
        "gr_top1_dist_m", "gr_total_candidates",
        "gr_safe_level", "gr_final_level",
        "t_gr_us", "t_gr_total_ms", "t_gr_cand_query_ms", "t_pgs_us",
        "gr_top_k", "pgs_top_k", "mode",
    ]
    done_keys: Set[Tuple[str, int]] = set()
    file_exists = os.path.isfile(out) and os.path.getsize(out) > 0
    if file_exists:
        with open(out, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qc = (row.get("query_code") or "").strip().upper()
                try:
                    kr = int(row.get("k") or 0)
                except ValueError:
                    continue
                if qc:
                    done_keys.add((qc, kr))
        print(f"  resume: {len(done_keys):,} rows already in {out}")

    n_tw = min(args.throwaway, len(query_codes))
    for qc in query_codes[:n_tw]:
        _search(searcher, qc, k_max, args)
    print(f"  throwaway done ({n_tw} x K={k_max})")

    t_by_k: Dict[int, List[float]] = defaultdict(list)
    rec_by_k: Dict[int, List[float]] = defaultdict(list)
    top1_by_k: Dict[int, List[int]] = defaultdict(list)
    t_pgs_by_k: Dict[int, List[float]] = defaultdict(list)
    n_ok = 0
    n_skip = 0
    mode = "a" if done_keys else "w"
    with open(out, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        t_run = time.perf_counter()
        total = n_jobs
        done_work = len(done_keys)
        for i, qc in enumerate(query_codes):
            for kv in k_list:
                if (qc, kv) in done_keys:
                    n_skip += 1
                    continue
                t0 = time.perf_counter()
                gr, info = _search(searcher, qc, kv, args)
                t_us = (time.perf_counter() - t0) * 1e6
                t_by_k[kv].append(t_us)
                gr_top = [r[0] for r in gr[:kv]]
                pg = pg_map.get((qc, kv), {})
                pgs_top = (pg.get("pgs_top") or [])[:kv]
                t_pgs = float(pg.get("t_pgs_us") or 0.0)
                recall = ""
                top1 = ""
                if pgs_top:
                    rec = len(set(gr_top) & set(pgs_top)) / kv if kv else 0.0
                    rec_by_k[kv].append(rec)
                    t_pgs_by_k[kv].append(t_pgs)
                    hit1 = 1 if (gr_top and pgs_top
                                 and gr_top[0] == pgs_top[0]) else 0
                    top1_by_k[kv].append(hit1)
                    recall = round(rec, 4)
                    top1 = hit1
                w.writerow({
                    "query_code": qc,
                    "k": kv,
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
                    "t_pgs_us": round(t_pgs, 1) if t_pgs else "",
                    "gr_top_k": ";".join(gr_top),
                    "pgs_top_k": ";".join(pgs_top),
                    "mode": "csv_scan",
                })
                n_ok += 1
                done_work += 1
                if n_ok % max(1, args.flush_every) == 0:
                    f.flush()
                elapsed = time.perf_counter() - t_run
                rate = n_ok / elapsed if elapsed else 0
                remain = total - done_work
                eta = remain / rate if rate else 0
                med20 = (
                    statistics.median(t_by_k[k_max]) / 1000.0
                    if t_by_k.get(k_max) else 0.0)
                line = (
                    f"  {done_work:,}/{total:,}  "
                    f"{rate:.2f} search/s  K={kv}  "
                    f"K{k_max} med {med20:.1f} ms  ETA {eta / 60.0:.1f} min")
                print(f"\r{line}", end="", flush=True)
                if n_ok % 50 == 0:
                    print()
                    print(f"  progress {done_work:,}/{total:,}  "
                          f"new {n_ok:,}  skipped {n_skip:,}")
        print()

    set_poi_prefix_index(None)
    print(f"wrote {n_ok:,} new rows -> {out}  (skipped {n_skip:,})")
    _print_summary(k_list, t_by_k, rec_by_k, top1_by_k, t_pgs_by_k)
    print("  (above: this process only; full file summary follows)")
    t_all: Dict[int, List[float]] = defaultdict(list)
    rec_all: Dict[int, List[float]] = defaultdict(list)
    top1_all: Dict[int, List[int]] = defaultdict(list)
    pgs_all: Dict[int, List[float]] = defaultdict(list)
    with open(out, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                kr = int(row.get("k") or 0)
                t_all[kr].append(float(row["t_gr_us"]))
            except (TypeError, ValueError, KeyError):
                continue
            rec_s = row.get("recall_at_k")
            if rec_s not in ("", None):
                rec_all[kr].append(float(rec_s))
            t1_s = row.get("top1_match")
            if t1_s not in ("", None):
                top1_all[kr].append(int(float(t1_s)))
            pg_s = row.get("t_pgs_us")
            if pg_s not in ("", None):
                pgs_all[kr].append(float(pg_s))
    print("=== full output file vs PostGIS ===")
    _print_summary(k_list, t_all, rec_all, top1_all, pgs_all)
    print("  note: pgs_med_ms is PostGIS latency copied from the warm CSV, "
          "not re-timed in this process.")
    print("  compare also: CSV memory K=20 median 0.51 ms | "
          "G3R-warm 0.63 ms | G3R-raw 270.79 ms")


if __name__ == "__main__":
    main()

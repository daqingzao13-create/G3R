#!/usr/bin/env python3
# coding: utf-8
"""
Exhaustive ellipsoid Top-K reference and PostGIS overfetch sensitivity. Sample unique
query_codes from the existing paper CSV using --seed, reuse its G3R and 8x PG lists, and do
not rerun Neo4j/G3R. Deduplicate POIs by main_code, then rank ST_Distance(geography). The
default uses a geometry-KNN seed distance as a ST_DWithin upper bound before ellipsoid
sorting; equivalence to a full scan depends on that bound containing true Top-K. --verify-
first checks the first query by full scan; --full-scan checks every query at higher cost.
Overfetch factors default to 4/8/16.

Usage:
    python knn_r14_overfetch_ablation.py --queries 5 --full-scan
    python knn_r14_overfetch_ablation.py --queries 500
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
from typing import Dict, List, Optional, Sequence, Set, Tuple

import psycopg2

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from geosot_knn_v3 import postgis_knn_standard
from geosot_spatial_reasoning import PG_CONFIG

CodeDist = Tuple[str, float]


def _parse_int_list(text: str, name: str) -> List[int]:
    """
    @param text Comma-separated positive integers.
    @param name Argument name for errors.
    @returns Sorted unique values.
    """
    vals = sorted({int(x) for x in text.split(",") if x.strip()})
    if not vals or min(vals) < 1:
        raise SystemExit(f"{name} must be positive integers")
    return vals


def _parse_args() -> argparse.Namespace:
    """@returns Parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description="R1-4 exhaustive ellipsoid Top-K and overfetch sweep.")
    p.add_argument(
        "--reuse-pg-csv",
        default=os.path.join(_ROOT, "csv", "knn_v31_10k.csv"),
        help="paper KNN CSV (query order, G3R and 8x PostGIS lists)")
    p.add_argument(
        "--queries", type=int, default=500,
        help="subset size from unique paper queries; 0 = all unique")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", default="3,5,10,20")
    p.add_argument("--overfetch", default="4,8,16")
    p.add_argument(
        "--seed-overfetch", type=int, default=2000,
        help="geometry-KNN seed size for ST_DWithin exact ranking")
    p.add_argument(
        "--output-true",
        default=os.path.join(_ROOT, "csv", "knn_r14_true_n500.csv"))
    p.add_argument(
        "--output-overfetch",
        default=os.path.join(_ROOT, "csv", "knn_r14_overfetch_n500.csv"))
    p.add_argument(
        "--full-scan", action="store_true",
        help="exact Top-K by scanning every distinct code (slow)")
    p.add_argument(
        "--verify-first", action="store_true", default=True,
        help="full-scan the first query and compare with ST_DWithin")
    p.add_argument(
        "--no-verify-first", dest="verify_first", action="store_false")
    p.add_argument("--flush-every", type=int, default=1)
    return p.parse_args()


def _load_paper_maps(
    path: str, k_set: Set[int],
) -> Tuple[List[str], Dict[Tuple[str, int], Dict]]:
    """
    @param path Paper KNN reference CSV.
    @param k_set Required K values.
    @returns (unique query appearance order, (query_code,k) -> G3R/PostGIS row).
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
            pgs = [c.strip().upper()
                   for c in (row.get("pgs_top_k") or "").split(";") if c]
            grw = [c.strip().upper()
                   for c in (row.get("gr_top_k") or "").split(";") if c]
            by[(qc, kr)] = {
                "pgs_top": pgs,
                "gr_top": grw,
                "t_pgs_us": float(row.get("t_pgs_us") or 0.0),
            }
    if not order:
        raise SystemExit(f"no queries in {path}")
    return order, by


def _sample_queries(order: List[str], n: int, seed: int) -> List[str]:
    """
    @param order Unique query sequence from the paper CSV.
    @param n Subset size; zero or at least the pool size selects all.
    @param seed random.sample seed.
    @returns Sampled query codes.
    """
    if n <= 0 or n >= len(order):
        return list(order)
    rng = random.Random(seed)
    return rng.sample(order, n)


def _recall(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    """
    @param pred Predicted Top list.
    @param gold Reference Top list.
    @param k Truncation length.
    @returns |pred[:k] intersect gold[:k]| / k.
    """
    if k <= 0:
        return 0.0
    return len(set(pred[:k]) & set(gold[:k])) / float(k)


def _fmt_codes(rows: Sequence[CodeDist], k: int) -> str:
    """
    @param rows (code,distance) pairs.
    @param k Truncation length.
    @returns Semicolon-separated codes.
    """
    return ";".join(c for c, _ in rows[:k])


def _fmt_dists(rows: Sequence[CodeDist], k: int) -> str:
    """
    @param rows (code,distance) pairs.
    @param k Truncation length.
    @returns Semicolon-separated distances.
    """
    return ";".join(f"{d:.6f}" for _, d in rows[:k])


def _fetch_ewkb(cur, query_code: str) -> Optional[bytes]:
    """
    @param cur Database cursor.
    @param query_code Query main_code.
    @returns EWKB, or None if absent.
    """
    cur.execute(
        "SELECT ST_AsEWKB(geometry) FROM public.poi_new "
        "WHERE main_code = %s LIMIT 1",
        (query_code,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return bytes(row[0])


def _prepare_poi_cc(cur) -> int:
    """
    Build a main_code-deduplicated temporary point table with geometry/geography GiST indexes.

    @param cur Database cursor.
    @returns Number of deduplicated points.
    """
    print("  building temp poi_cc (distinct main_code) ...")
    t0 = time.perf_counter()
    cur.execute("DROP TABLE IF EXISTS poi_cc")
    cur.execute(
        "CREATE TEMP TABLE poi_cc AS "
        "SELECT DISTINCT ON (main_code) main_code, geometry, "
        "       geometry::geography AS geog "
        "FROM public.poi_new "
        "ORDER BY main_code")
    cur.execute("CREATE INDEX poi_cc_gix ON poi_cc USING gist (geometry)")
    cur.execute("CREATE INDEX poi_cc_geog ON poi_cc USING gist (geog)")
    cur.execute("ANALYZE poi_cc")
    cur.execute("SELECT count(*) FROM poi_cc")
    n = int(cur.fetchone()[0])
    print(f"  poi_cc n={n:,} in {time.perf_counter() - t0:.1f}s")
    return n


def _rank_full_scan(cur, q_geom: bytes, query_code: str, k: int) -> List[CodeDist]:
    """
    Rank the complete point table by ST_Distance(geography).

    @param cur Cursor.
    @param q_geom Query EWKB.
    @param query_code Self code to exclude.
    @param k Top-K.
    @returns [(code,dist_m), ...].
    """
    cur.execute(
        "SELECT main_code, dist_m FROM ("
        "  SELECT main_code,"
        "         ST_Distance(geog, %s::geometry::geography) AS dist_m "
        "    FROM poi_cc "
        "   WHERE main_code <> %s"
        ") t "
        "ORDER BY dist_m, main_code "
        "LIMIT %s",
        (q_geom, query_code, k))
    return [(str(r[0]).strip().upper(), float(r[1])) for r in cur.fetchall()]


def _rank_dwithin(
    cur, q_geom: bytes, query_code: str, k: int, seed_n: int,
) -> List[CodeDist]:
    """
    Use geometry-KNN seeds to obtain a radius upper bound, then apply ST_DWithin and ellipsoid
    ranking.

    @param cur Cursor.
    @param q_geom Query EWKB.
    @param query_code Self code to exclude.
    @param k Top-K.
    @param seed_n Geometry overfetch count.
    @returns [(code,dist_m), ...].
    """
    fetch_n = max(int(seed_n), k)
    cur.execute(
        "SELECT main_code, dist_m FROM ("
        "  SELECT main_code,"
        "         ST_Distance(geog, %s::geometry::geography) AS dist_m "
        "    FROM ("
        "      SELECT main_code, geog "
        "        FROM poi_cc "
        "       WHERE main_code <> %s "
        "       ORDER BY geometry <-> %s::geometry "
        "       LIMIT %s"
        "    ) s"
        ") t "
        "ORDER BY dist_m, main_code "
        "LIMIT %s",
        (q_geom, query_code, q_geom, fetch_n, k))
    seed = [(str(r[0]).strip().upper(), float(r[1])) for r in cur.fetchall()]
    if len(seed) < k:
        fetch_n *= 8
        cur.execute(
            "SELECT main_code, dist_m FROM ("
            "  SELECT main_code,"
            "         ST_Distance(geog, %s::geometry::geography) AS dist_m "
            "    FROM ("
            "      SELECT main_code, geog "
            "        FROM poi_cc "
            "       WHERE main_code <> %s "
            "       ORDER BY geometry <-> %s::geometry "
            "       LIMIT %s"
            "    ) s"
            ") t "
            "ORDER BY dist_m, main_code "
            "LIMIT %s",
            (q_geom, query_code, q_geom, fetch_n, k))
        seed = [(str(r[0]).strip().upper(), float(r[1]))
                for r in cur.fetchall()]
    if not seed:
        return []
    radius = seed[min(k, len(seed)) - 1][1]
    for extra in (0.0, 0.01, 1.0):
        cur.execute(
            "SELECT main_code, dist_m FROM ("
            "  SELECT main_code,"
            "         ST_Distance(geog, %s::geometry::geography) AS dist_m "
            "    FROM poi_cc "
            "   WHERE main_code <> %s "
            "     AND ST_DWithin(geog, %s::geometry::geography, %s)"
            ") t "
            "ORDER BY dist_m, main_code "
            "LIMIT %s",
            (q_geom, query_code, q_geom, radius + extra, k))
        rows = [(str(r[0]).strip().upper(), float(r[1]))
                for r in cur.fetchall()]
        if len(rows) >= min(k, len(seed)):
            return rows
    return seed


def _load_done_true(path: str) -> Dict[str, Dict]:
    """
    @param path Reference CSV.
    @returns query_code -> previously written row.
    """
    out: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qc = (row.get("query_code") or "").strip().upper()
            if qc:
                out[qc] = row
    return out


def _load_done_overfetch(path: str) -> Set[Tuple[str, int, int]]:
    """
    @param path Overfetch CSV.
    @returns Completed (query_code,k,factor) keys.
    """
    done: Set[Tuple[str, int, int]] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qc = (row.get("query_code") or "").strip().upper()
            try:
                k = int(row.get("k") or 0)
                fac = int(row.get("overfetch") or 0)
            except ValueError:
                continue
            if qc and k > 0 and fac > 0:
                done.add((qc, k, fac))
    return done


def _summarize_overfetch_csv(path: str) -> None:
    """
    Recompute a summary from the complete overfetch CSV, including rows written before a resumed
    run.

    @param path Overfetch CSV.
    """
    rec_of: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    rec_gr: Dict[int, List[float]] = defaultdict(list)
    rec_pgs8: Dict[int, List[float]] = defaultdict(list)
    rec_gr_pgs8: Dict[int, List[float]] = defaultdict(list)
    t_of: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    t_true: List[float] = []
    k_seen: Set[int] = set()
    fac_seen: Set[int] = set()
    seen_true: Set[str] = set()
    seen_agree: Set[Tuple[str, int]] = set()
    if not os.path.exists(path):
        print("  no overfetch CSV to summarize")
        return
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                k = int(row.get("k") or 0)
                fac = int(row.get("overfetch") or 0)
            except ValueError:
                continue
            if k < 1 or fac < 1:
                continue
            k_seen.add(k)
            fac_seen.add(fac)
            qc = (row.get("query_code") or "").strip().upper()
            if qc and qc not in seen_true:
                seen_true.add(qc)
                t_true.append(float(row.get("t_true_us") or 0.0))
            rec_of[(k, fac)].append(float(row.get("recall_pgs_vs_true") or 0.0))
            t_of[(k, fac)].append(float(row.get("t_pgs_us") or 0.0))
            key = (qc, k)
            if key in seen_agree:
                continue
            seen_agree.add(key)
            gr = row.get("recall_gr_vs_true")
            p8 = row.get("recall_paper8_vs_true")
            g8 = row.get("recall_gr_vs_paper8")
            if gr not in (None, ""):
                rec_gr[k].append(float(gr))
            if p8 not in (None, ""):
                rec_pgs8[k].append(float(p8))
            if g8 not in (None, ""):
                rec_gr_pgs8[k].append(float(g8))
    _print_summary(
        sorted(k_seen), sorted(fac_seen), rec_of, rec_gr, rec_pgs8,
        rec_gr_pgs8, t_of, t_true)


def _print_summary(
    k_list: List[int],
    factors: List[int],
    rec_of: Dict[Tuple[int, int], List[float]],
    rec_gr: Dict[int, List[float]],
    rec_pgs8: Dict[int, List[float]],
    rec_gr_pgs8: Dict[int, List[float]],
    t_of: Dict[Tuple[int, int], List[float]],
    t_true: List[float],
) -> None:
    """
    Print mean agreement against the exhaustive reference for each K/overfetch factor.

    @param k_list K values.
    @param factors Overfetch factors.
    @param rec_of (k,factor) -> PG/reference agreements.
    @param rec_gr k -> G3R/reference agreements.
    @param rec_pgs8 k -> paper 8x PG/reference agreements.
    @param rec_gr_pgs8 k -> original G3R/8x PG agreements.
    @param t_of (k,factor) -> microseconds.
    @param t_true Reference times in microseconds.
    """
    print()
    print("  summary (mean set agreement)")
    if t_true:
        print(
            f"  true-rank median {statistics.median(t_true) / 1000.0:.1f} ms  "
            f"n={len(t_true)}")
    hdr = (
        f"  {'K':>3}  {'arm':<22}  {'n':>5}  "
        f"{'agree vs true':>13}  {'median ms':>10}")
    print(hdr)
    for k in k_list:
        for fac in factors:
            xs = rec_of.get((k, fac), [])
            ts = t_of.get((k, fac), [])
            agree = (100.0 * statistics.mean(xs)) if xs else float("nan")
            med = (statistics.median(ts) / 1000.0) if ts else float("nan")
            print(
                f"  {k:>3}  {('PostGIS ' + str(fac) + 'x'):<22}  "
                f"{len(xs):>5}  {agree:12.4f}%  {med:10.2f}")
        xs = rec_pgs8.get(k, [])
        agree = (100.0 * statistics.mean(xs)) if xs else float("nan")
        print(
            f"  {k:>3}  {'paper 8x CSV':<22}  "
            f"{len(xs):>5}  {agree:12.4f}%  {'(csv)':>10}")
        xs = rec_gr.get(k, [])
        agree = (100.0 * statistics.mean(xs)) if xs else float("nan")
        print(
            f"  {k:>3}  {'G3R vs true':<22}  "
            f"{len(xs):>5}  {agree:12.4f}%  {'(csv)':>10}")
        xs = rec_gr_pgs8.get(k, [])
        agree = (100.0 * statistics.mean(xs)) if xs else float("nan")
        print(
            f"  {k:>3}  {'G3R vs paper 8x':<22}  "
            f"{len(xs):>5}  {agree:12.4f}%  {'(csv)':>10}")


def main() -> None:
    """Sample queries, write exhaustive-reference and overfetch CSVs, and print summaries."""
    args = _parse_args()
    k_list = _parse_int_list(args.k, "--k")
    factors = _parse_int_list(args.overfetch, "--overfetch")
    k_max = max(k_list)
    k_set = set(k_list)

    order, paper = _load_paper_maps(args.reuse_pg_csv, k_set)
    queries = _sample_queries(order, args.queries, args.seed)
    print(
        f"  paper unique queries {len(order):,}  "
        f"subset {len(queries):,}  seed {args.seed}  "
        f"K={k_list}  overfetch={factors}")
    print(f"  true -> {args.output_true}")
    print(f"  overfetch -> {args.output_overfetch}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_true)) or ".",
                exist_ok=True)
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_overfetch)) or ".",
        exist_ok=True)

    done_true = _load_done_true(args.output_true)
    done_of = _load_done_overfetch(args.output_overfetch)
    print(f"  resume true={len(done_true):,}  overfetch={len(done_of):,}")

    conn = psycopg2.connect(**PG_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET jit = off")
    cur.execute("SET work_mem = '256MB'")
    _prepare_poi_cc(cur)

    true_fields = [
        "query_code", "k_max", "n_true", "method",
        "t_true_us", "true_top_k", "true_top_k_dist_m",
    ]
    of_fields = [
        "query_code", "k", "overfetch",
        "t_true_us", "t_pgs_us",
        "recall_pgs_vs_true", "recall_gr_vs_true",
        "recall_paper8_vs_true", "recall_gr_vs_paper8",
        "top1_pgs_vs_true", "top1_gr_vs_true",
        "n_true", "true_top_k", "pgs_top_k", "gr_top_k", "paper8_top_k",
    ]

    true_new = not os.path.exists(args.output_true)
    of_new = not os.path.exists(args.output_overfetch)
    f_true = open(args.output_true, "a", newline="", encoding="utf-8")
    f_of = open(args.output_overfetch, "a", newline="", encoding="utf-8")
    w_true = csv.DictWriter(f_true, fieldnames=true_fields)
    w_of = csv.DictWriter(f_of, fieldnames=of_fields)
    if true_new:
        w_true.writeheader()
    if of_new:
        w_of.writeheader()

    rec_of: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    rec_gr: Dict[int, List[float]] = defaultdict(list)
    rec_pgs8: Dict[int, List[float]] = defaultdict(list)
    rec_gr_pgs8: Dict[int, List[float]] = defaultdict(list)
    t_of: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    t_true_all: List[float] = []

    verified = False
    t_run = time.perf_counter()
    n_ok = 0
    try:
        for i, qc in enumerate(queries, start=1):
            if qc in done_true:
                gold_codes = [
                    c for c in (done_true[qc].get("true_top_k") or "").split(";")
                    if c]
                gold = [(c, 0.0) for c in gold_codes]
                t_true_us = float(done_true[qc].get("t_true_us") or 0.0)
            else:
                q_geom = _fetch_ewkb(cur, qc)
                if q_geom is None:
                    print(f"\n  skip {qc}: no geometry")
                    continue
                method = "full_scan" if args.full_scan else "dwithin"
                t0 = time.perf_counter()
                if args.full_scan:
                    gold = _rank_full_scan(cur, q_geom, qc, k_max)
                else:
                    gold = _rank_dwithin(
                        cur, q_geom, qc, k_max, args.seed_overfetch)
                    if args.verify_first and not verified:
                        print("  verifying first query: DWithin vs full scan")
                        gold_full = _rank_full_scan(cur, q_geom, qc, k_max)
                        dwith_codes = [c for c, _ in gold]
                        full_codes = [c for c, _ in gold_full]
                        if dwith_codes != full_codes:
                            raise SystemExit(
                                "DWithin Top-K differs from full scan on "
                                f"{qc}\n  dwithin={dwith_codes}\n"
                                f"  full={full_codes}")
                        gold = gold_full
                        method = "full_scan_verified"
                        verified = True
                        print("  verify ok")
                t_true_us = (time.perf_counter() - t0) * 1e6
                row_t = {
                    "query_code": qc,
                    "k_max": k_max,
                    "n_true": len(gold),
                    "method": method,
                    "t_true_us": round(t_true_us, 1),
                    "true_top_k": _fmt_codes(gold, k_max),
                    "true_top_k_dist_m": _fmt_dists(gold, k_max),
                }
                w_true.writerow(row_t)
                done_true[qc] = row_t
                f_true.flush()

            gold_codes = [c for c, _ in gold]
            if not gold_codes:
                print(f"\n  skip {qc}: empty true Top-K")
                continue
            t_true_all.append(t_true_us)

            for kv in k_list:
                paper_row = paper.get((qc, kv), {})
                gr_top = (paper_row.get("gr_top") or [])[:kv]
                paper8 = (paper_row.get("pgs_top") or [])[:kv]
                gold_k = gold_codes[:kv]
                r_gr = _recall(gr_top, gold_k, kv) if gr_top else ""
                r_p8 = _recall(paper8, gold_k, kv) if paper8 else ""
                r_g8 = (
                    _recall(gr_top, paper8, kv)
                    if gr_top and paper8 else "")
                if isinstance(r_gr, float):
                    rec_gr[kv].append(r_gr)
                if isinstance(r_p8, float):
                    rec_pgs8[kv].append(r_p8)
                if isinstance(r_g8, float):
                    rec_gr_pgs8[kv].append(r_g8)

                for fac in factors:
                    if (qc, kv, fac) in done_of:
                        continue
                    t1 = time.perf_counter()
                    pgs = postgis_knn_standard(conn, qc, kv, oversample=fac)
                    t_pgs_us = (time.perf_counter() - t1) * 1e6
                    pgs_codes = [c for c, _ in pgs[:kv]]
                    r_pgs = _recall(pgs_codes, gold_k, kv) if pgs_codes else 0.0
                    rec_of[(kv, fac)].append(r_pgs)
                    t_of[(kv, fac)].append(t_pgs_us)
                    top1_pgs = int(
                        bool(pgs_codes and gold_k and pgs_codes[0] == gold_k[0]))
                    top1_gr = int(
                        bool(gr_top and gold_k and gr_top[0] == gold_k[0]))
                    w_of.writerow({
                        "query_code": qc,
                        "k": kv,
                        "overfetch": fac,
                        "t_true_us": round(t_true_us, 1),
                        "t_pgs_us": round(t_pgs_us, 1),
                        "recall_pgs_vs_true": round(r_pgs, 4),
                        "recall_gr_vs_true": (
                            round(r_gr, 4) if isinstance(r_gr, float) else ""),
                        "recall_paper8_vs_true": (
                            round(r_p8, 4) if isinstance(r_p8, float) else ""),
                        "recall_gr_vs_paper8": (
                            round(r_g8, 4) if isinstance(r_g8, float) else ""),
                        "top1_pgs_vs_true": top1_pgs,
                        "top1_gr_vs_true": top1_gr,
                        "n_true": len(gold),
                        "true_top_k": ";".join(gold_k),
                        "pgs_top_k": ";".join(pgs_codes),
                        "gr_top_k": ";".join(gr_top),
                        "paper8_top_k": ";".join(paper8),
                    })
                    done_of.add((qc, kv, fac))
                    n_ok += 1
                    if n_ok % max(1, args.flush_every) == 0:
                        f_of.flush()

            elapsed = time.perf_counter() - t_run
            rate = i / elapsed if elapsed else 0.0
            remain = (len(queries) - i) / rate if rate else 0.0
            print(
                f"\r  {i}/{len(queries)}  {qc[:18]}  "
                f"true {t_true_us / 1000.0:.0f} ms  "
                f"ETA {remain / 60.0:.1f} min",
                end="", flush=True)
        print()
    finally:
        f_true.flush()
        f_of.flush()
        f_true.close()
        f_of.close()
        cur.close()
        conn.close()

    _summarize_overfetch_csv(args.output_overfetch)
    print(f"  wrote true -> {args.output_true}")
    print(f"  wrote overfetch -> {args.output_overfetch}")


if __name__ == "__main__":
    main()

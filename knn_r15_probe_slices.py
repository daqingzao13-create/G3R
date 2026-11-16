#!/usr/bin/env python3
# coding: utf-8
"""
Report grid-boundary, sparse-candidate, and near-tie KNN slices from a compatible paper CSV
without database access. Boundary margin is measured at level 16; sparse queries use the
bottom 5% of gr_total_candidates; near ties have different Top-1 codes with distance
difference <=0.05 m. Optionally join exhaustive agreement from an overfetch-reference CSV.
The historical default input is V3.1; use the V3.2-compatible joined CSV for the final paper
protocol.

Usage:
    python knn_r15_probe_slices.py
    python knn_r15_probe_slices.py --k 20 --true-csv csv/knn_r14_overfetch_n500.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from geosot_knn_v3 import _min_margin_m, find_safe_level
from geosot_spatial_reasoning import code_to_bbox, fast_code_to_center


def _parse_args() -> argparse.Namespace:
    """@returns Parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description="KNN boundary/sparse/near-tie slices from paper CSV.")
    p.add_argument(
        "--knn-csv",
        default=os.path.join(_ROOT, "csv", "knn_v31_10k.csv"))
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--margin-level", type=int, default=16)
    p.add_argument("--pct", type=float, default=5.0)
    p.add_argument("--tie-m", type=float, default=0.05)
    p.add_argument(
        "--true-csv",
        default=os.path.join(_ROOT, "csv", "knn_r14_overfetch_n500.csv"),
        help="optional R1-4 overfetch CSV; empty string skips")
    return p.parse_args()


def _margin(code: str, level: int) -> float:
    """
    @param code Query main_code.
    @param level GeoSOT level.
    @returns Minimum point-to-cell-boundary distance in meters.
    """
    pt = fast_code_to_center(code)
    bbox = code_to_bbox(code[:level + 2])
    return _min_margin_m(pt[0], pt[1], bbox)


def _load_k(path: str, k: int) -> List[Dict]:
    """
    @param path Compatible paper KNN CSV.
    @param k K value to read.
    @returns Matching rows.
    """
    out = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("k") or 0) == k:
                out.append(row)
    if not out:
        raise SystemExit(f"no rows with k={k} in {path}")
    return out


def _load_true(path: str, k: int) -> Dict[str, float]:
    """
    @param path Overfetch-reference CSV.
    @param k K value; select overfetch=8 G3R/reference agreement.
    @returns query_code -> recall_gr_vs_true.
    """
    out: Dict[str, float] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("k") or 0) != k:
                continue
            if int(row.get("overfetch") or 0) != 8:
                continue
            qc = (row.get("query_code") or "").strip().upper()
            val = row.get("recall_gr_vs_true")
            if qc and val not in (None, ""):
                out[qc] = float(val)
    return out


def _mean(xs: List[float]) -> str:
    """
    @param xs Numeric values.
    @returns Percentage text, or an em dash if unavailable.
    """
    if not xs:
        return "—"
    return f"{100.0 * statistics.mean(xs):.2f}%"


def _slice_stats(
    name: str,
    rows: List[Dict],
    true_map: Dict[str, float],
) -> None:
    """
    @param name Slice name.
    @param rows Slice records.
    @param true_map Exhaustive-reference agreements.
    """
    rec = [float(r["recall_at_k"]) for r in rows]
    brk = sum(int(r.get("gr_braked") or 0) for r in rows)
    tvals = [true_map[r["query_code"]]
             for r in rows if r["query_code"] in true_map]
    print(
        f"  {name:<22} n={len(rows):5d}  "
        f"agree vs PG {_mean(rec):>8}  min {100.0 * min(rec):.2f}%  "
        f"braked {brk}  "
        f"vs true n={len(tvals)} {_mean(tvals)}")


def main() -> None:
    """Print brake/expansion statistics and the three diagnostic slices."""
    args = _parse_args()
    rows = _load_k(args.knn_csv, args.k)
    true_map = _load_true(args.true_csv, args.k)
    print(f"  k={args.k}  n={len(rows):,}  from {args.knn_csv}")

    braked = sum(int(r.get("gr_braked") or 0) for r in rows)
    widen = sum(1 for r in rows if int(r.get("gr_final_width") or 3) > 3)
    expand = [r for r in rows
              if int(r.get("gr_final_level") or 0)
              < int(r.get("gr_safe_level") or 0)]
    print(f"  braked {braked}/{len(rows)}")
    print(f"  width>3 {widen}/{len(rows)}")
    print(
        f"  final_level<safe_level {len(expand)}/"
        f"{len(rows)}  agree {_mean([float(r['recall_at_k']) for r in expand])}")

    raw_lt13 = 0
    margins: List[float] = []
    for r in rows:
        qc = r["query_code"]
        if find_safe_level(qc, 100.0) < 13:
            raw_lt13 += 1
        margins.append(_margin(qc, args.margin_level))
    print(f"  raw L0 < 13 (before start_level_max) {raw_lt13}/{len(rows)}")

    pct = args.pct
    m_sorted = sorted(margins)
    m_thr = m_sorted[max(0, int(len(m_sorted) * pct / 100.0) - 1)]
    cand = [int(r["gr_total_candidates"]) for r in rows]
    c_sorted = sorted(cand)
    c_thr = c_sorted[max(0, int(len(c_sorted) * pct / 100.0) - 1)]

    all_rec = [float(r["recall_at_k"]) for r in rows]
    _slice_stats("all", rows, true_map)
    boundary = [r for r, m in zip(rows, margins) if m <= m_thr]
    _slice_stats(f"boundary L{args.margin_level} p{pct:.0f}",
                 boundary, true_map)
    sparse = [r for r in rows if int(r["gr_total_candidates"]) <= c_thr]
    _slice_stats(f"sparse cand p{pct:.0f}", sparse, true_map)
    near = [
        r for r in rows
        if r.get("top1_match") == "0"
        and float(r.get("top1_err_m") or 0.0) <= args.tie_m
    ]
    _slice_stats(f"near-tie Top-1 <={args.tie_m} m", near, true_map)
    print("  safe_level", Counter(r["gr_safe_level"] for r in rows).most_common())
    print(f"  all agree vs PG {_mean(all_rec)}")


if __name__ == "__main__":
    main()

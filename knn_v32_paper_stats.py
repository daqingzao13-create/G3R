#!/usr/bin/env python3
# coding: utf-8
"""
Summarize Recall, Top-1 agreement, MRR, and median latency from knn_v32_local8k_n10k.csv;
join pgs_top1_dist_m from knn_v31_10k.csv for MAE.

Usage:
    python knn_v32_paper_stats.py
"""
from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict

_ROOT = os.path.dirname(os.path.abspath(__file__))
V32 = os.path.join(_ROOT, "csv", "knn_v32_local8k_n10k.csv")
V31 = os.path.join(_ROOT, "csv", "knn_v31_10k.csv")


def _codes(text: str):
    """Parse semicolon-separated codes and return uppercase codes."""
    return [c.strip().upper() for c in (text or "").split(";") if c.strip()]


def _mrr(pred, gold):
    """
    @param pred G3R Top-K.
    @param gold PostGIS Top-K.
    @returns Reciprocal rank of gold[0] in pred, or zero if absent.
    """
    if not gold or not pred:
        return 0.0
    nn = gold[0]
    for i, c in enumerate(pred, start=1):
        if c == nn:
            return 1.0 / i
    return 0.0


def main() -> None:
    """Print summaries grouped by K."""
    pg_d = {}
    with open(V31, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qc = (row.get("query_code") or "").strip().upper()
            try:
                kv = int(row.get("k") or 0)
            except ValueError:
                continue
            try:
                pg_d[(qc, kv)] = float(row.get("pgs_top1_dist_m") or 0.0)
            except ValueError:
                pass
            if (qc, kv) not in pg_d:
                continue

    by_k = defaultdict(lambda: {
        "rec": [], "top1": [], "rr": [], "t": [], "mae": [],
        "t_pgs": [],
    })
    with open(V32, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qc = (row.get("query_code") or "").strip().upper()
            kv = int(row["k"])
            rec = float(row["recall_vs_pgs"]) if row.get("recall_vs_pgs") else None
            if rec is not None:
                by_k[kv]["rec"].append(rec)
            by_k[kv]["top1"].append(int(row.get("top1_vs_pgs") or 0))
            pred = _codes(row.get("v32_top_k") or "")
            gold = _codes(row.get("pgs_top_k") or "")
            by_k[kv]["rr"].append(_mrr(pred, gold))
            by_k[kv]["t"].append(float(row["t_gr_us"]))
            try:
                g3 = float(row.get("v32_top1_dist_m") or "")
            except ValueError:
                g3 = None
            p1 = pg_d.get((qc, kv))
            if g3 is not None and p1 is not None:
                by_k[kv]["mae"].append(abs(g3 - p1))

    with open(V31, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kv = int(row["k"])
            try:
                by_k[kv]["t_pgs"].append(float(row["t_pgs_us"]))
            except (ValueError, KeyError):
                pass

    print("K  n  recall  top1  MRR  MAE_m  med_g3r_ms  med_pg_ms  speedup")
    for kv in sorted(by_k):
        s = by_k[kv]
        n = len(s["rec"])
        rec = 100.0 * statistics.mean(s["rec"]) if s["rec"] else None
        top1 = 100.0 * statistics.mean(s["top1"]) if s["top1"] else None
        mrr = statistics.mean(s["rr"]) if s["rr"] else None
        mae = statistics.mean(s["mae"]) if s["mae"] else None
        med = statistics.median(s["t"]) / 1000.0 if s["t"] else None
        pg = statistics.median(s["t_pgs"]) / 1000.0 if s["t_pgs"] else None
        sp = (pg / med) if (pg and med) else None
        print(
            f"{kv:2d} {n:5d}  {rec:.4f}%  {top1:.4f}%  {mrr:.4f}  "
            f"{mae:.4f}  {med:.3f}  {pg:.3f}  {sp:.2f}x"
        )


if __name__ == "__main__":
    main()

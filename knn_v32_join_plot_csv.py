#!/usr/bin/env python3
# coding: utf-8
"""
Join V3.2 results to the V3.1-compatible plotting schema for plot_knn_v3_sci.py and slice
analyses.

Usage:
    python knn_v32_join_plot_csv.py
"""
from __future__ import annotations

import csv
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
V32 = os.path.join(_ROOT, "csv", "knn_v32_local8k_n10k.csv")
V31 = os.path.join(_ROOT, "csv", "knn_v31_10k.csv")
OUT = os.path.join(_ROOT, "csv", "knn_v32_plot_n10k.csv")

FIELDS = [
    "query_code", "k", "recall_at_k", "reciprocal_rank", "top1_match",
    "pg_std_agree", "pgs_top1_dist_m", "gr_top1_dist_m", "top1_err_m",
    "top1_err_pct", "gr_total_candidates", "gr_safe_level", "gr_final_level",
    "gr_prefixes", "t_gr_us", "t_pg_us", "t_pgs_us", "gr_top_k", "pgs_top_k",
    "gr_final_width", "gr_rounds", "gr_braked",
]


def main() -> None:
    """
    Join PostGIS timing/level reference fields from V3.1; overwrite G3R latency and recall with
    V3.2 values.
    """
    v31 = {}
    with open(V31, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qc = (row.get("query_code") or "").strip().upper()
            kv = int(row["k"])
            v31[(qc, kv)] = row

    n = 0
    miss = 0
    with open(V32, "r", newline="", encoding="utf-8") as fin, \
            open(OUT, "w", newline="", encoding="utf-8") as fout:
        wr = csv.DictWriter(fout, fieldnames=FIELDS)
        wr.writeheader()
        for row in csv.DictReader(fin):
            qc = (row.get("query_code") or "").strip().upper()
            kv = int(row["k"])
            old = v31.get((qc, kv))
            if old is None:
                miss += 1
                continue
            rec = row.get("recall_vs_pgs") or ""
            try:
                gr_d = float(row.get("v32_top1_dist_m") or "")
                pg_d = float(old.get("pgs_top1_dist_m") or "")
                err = abs(gr_d - pg_d)
                pct = (100.0 * err / pg_d) if pg_d > 0 else ""
            except ValueError:
                gr_d, pg_d, err, pct = "", "", "", ""
            wr.writerow({
                "query_code": qc,
                "k": kv,
                "recall_at_k": rec,
                "reciprocal_rank": old.get("reciprocal_rank", ""),
                "top1_match": row.get("top1_vs_pgs", ""),
                "pg_std_agree": old.get("pg_std_agree", ""),
                "pgs_top1_dist_m": old.get("pgs_top1_dist_m", ""),
                "gr_top1_dist_m": row.get("v32_top1_dist_m", ""),
                "top1_err_m": ("" if err == "" else round(err, 6)),
                "top1_err_pct": ("" if pct == "" else round(pct, 6)),
                "gr_total_candidates": row.get("gr_total_candidates")
                or old.get("gr_total_candidates", ""),
                "gr_safe_level": old.get("gr_safe_level", ""),
                "gr_final_level": old.get("gr_final_level", ""),
                "gr_prefixes": old.get("gr_prefixes", ""),
                "t_gr_us": row.get("t_gr_us", ""),
                "t_pg_us": old.get("t_pg_us", ""),
                "t_pgs_us": old.get("t_pgs_us", ""),
                "gr_top_k": row.get("v32_top_k", ""),
                "pgs_top_k": row.get("pgs_top_k") or old.get("pgs_top_k", ""),
                "gr_final_width": old.get("gr_final_width", ""),
                "gr_rounds": old.get("gr_rounds", ""),
                "gr_braked": old.get("gr_braked", "0"),
            })
            n += 1
    print(f"wrote {n:,} rows -> {OUT}  (v31 miss {miss})")


if __name__ == "__main__":
    main()

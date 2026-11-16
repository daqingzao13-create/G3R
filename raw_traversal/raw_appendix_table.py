#!/usr/bin/env python3
# coding: utf-8
"""
Aggregate warm-cache vs raw-traversal CSVs into a single appendix table.

Reads four (warm, raw) CSV pairs — one per operator — and emits both a
Markdown table (for the LaTeX appendix) and a machine-readable CSV.

The table reports, for each operator and subset:
    * mean / median per-query latency on Neo4j (G3R)
    * the warm-vs-raw ratio (how much extra wall-clock time the
      runtime traversal incurs relative to the warm-cache lookups)
    * mean / median latency on PostGIS (kept fixed across rows so
      readers can re-derive the warm and raw speedups)
    * set-level F1 / Recall@K agreement with PostGIS (unchanged
      between warm and raw — same algorithm, same candidate semantics)

A typical invocation (run from ``./``)::

    python raw_traversal/raw_appendix_table.py \\
        --distance-warm benchmark_v9/summary.csv \\
        --distance-raw  raw_benchmark_v9/summary.csv \\
        --forward-warm  csv/overlay_v7_warm_check2.csv \\
        --forward-raw   csv/raw_overlay_forward.csv \\
        --reverse-warm  csv/reverse_warm.csv \\
        --reverse-raw   csv/raw_reverse_overlay.csv \\
        --knn-warm      knn_v31_100k.csv \\
        --knn-raw       csv/raw_knn_n100k.csv \\
        --out-md        appendix_warm_vs_raw.md \\
        --out-csv       appendix_warm_vs_raw.csv

Any of the eight ``--*-warm`` / ``--*-raw`` flags may be omitted; the
corresponding rows are dropped from the table. This lets you generate
a partial appendix when only some of the raw runs have completed.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Callable, Dict, Iterable, List, Optional, Tuple


# ----------------------------------------------------------------------
# generic CSV helpers
# ----------------------------------------------------------------------

def _read_csv(path: str) -> List[Dict[str, str]]:
    """Read one CSV file, or every per-combo CSV in a directory.

    When ``path`` is a directory we glob ``*_x_*.csv`` (the per-combo
    files written by ``geometry_benchmark.py`` /
    ``raw_geometry_benchmark.py``) and skip ``summary.csv`` because
    that one is pre-aggregated (one row per combination, with column
    names ending in ``_mean``/``_median``/``_count`` etc.) and would
    require a different aggregation path.
    """
    if not path:
        return []
    if os.path.isdir(path):
        recs: List[Dict[str, str]] = []
        for name in sorted(os.listdir(path)):
            if not name.lower().endswith(".csv"):
                continue
            if name.lower() == "summary.csv":
                continue
            if "_x_" not in name:
                continue
            with open(os.path.join(path, name),
                      "r", encoding="utf-8-sig", newline="") as f:
                recs.extend(csv.DictReader(f))
        return recs
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(s) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _stat(vals: Iterable[Optional[float]]) -> Dict[str, float]:
    arr = [v for v in vals if v is not None]
    if not arr:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p95": float("nan")}
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    # P95 by linear interpolation
    k = 0.95 * (n - 1)
    lo, hi = int(k), min(int(k) + 1, n - 1)
    p95 = arr_sorted[lo] + (k - lo) * (arr_sorted[hi] - arr_sorted[lo])
    return {
        "n": n,
        "mean": statistics.fmean(arr_sorted),
        "median": statistics.median(arr_sorted),
        "p95": p95,
    }


def _fmt(x: float, unit: str = "ms", prec: int = 2) -> str:
    if x != x:  # NaN
        return "—"
    if unit == "%":
        return f"{x * 100:.{prec}f}%"
    if unit == "x":
        return f"{x:.{prec}f}×"
    return f"{x:.{prec}f} {unit}".rstrip()


# ----------------------------------------------------------------------
# per-operator row construction
# ----------------------------------------------------------------------

def _row(label: str, warm: Dict, raw: Dict, pg: Dict,
         accuracy_label: str = "F1",
         accuracy_warm: Optional[float] = None,
         accuracy_raw: Optional[float] = None) -> Dict[str, str]:
    """Build one Markdown / CSV row from per-mode stats dicts.

    Accuracy is stored in two generic columns plus an ``accuracy_metric``
    column naming the metric for that row. Doing it this way keeps the
    fieldnames stable across operators (distance uses ``dist_match``,
    overlay uses ``F1``, KNN uses ``Recall@K``) so the aggregated CSV
    has a single header.
    """
    ratio = (raw["median"] / warm["median"]
             if warm["median"] and warm["median"] > 0
             and raw["median"] == raw["median"]
             else float("nan"))
    speedup_warm = (pg["median"] / warm["median"]
                    if warm["median"] and warm["median"] > 0
                    and pg["median"] == pg["median"]
                    else float("nan"))
    speedup_raw = (pg["median"] / raw["median"]
                   if raw["median"] and raw["median"] > 0
                   and pg["median"] == pg["median"]
                   else float("nan"))
    return {
        "subset": label,
        "n_warm": str(warm.get("n", 0)),
        "n_raw": str(raw.get("n", 0)),
        "g3r_warm_mean_ms": _fmt(warm["mean"]),
        "g3r_warm_median_ms": _fmt(warm["median"]),
        "g3r_raw_mean_ms": _fmt(raw["mean"]),
        "g3r_raw_median_ms": _fmt(raw["median"]),
        "raw_to_warm_ratio": _fmt(ratio, unit="x"),
        "pg_mean_ms": _fmt(pg["mean"]),
        "pg_median_ms": _fmt(pg["median"]),
        "warm_speedup_vs_pg": _fmt(speedup_warm, unit="x"),
        "raw_speedup_vs_pg": _fmt(speedup_raw, unit="x"),
        "accuracy_metric": accuracy_label,
        "accuracy_warm": (_fmt(accuracy_warm, unit="%")
                          if accuracy_warm is not None else "—"),
        "accuracy_raw": (_fmt(accuracy_raw, unit="%")
                         if accuracy_raw is not None else "—"),
    }


# ----------------------------------------------------------------------
# operator-specific readers
# ----------------------------------------------------------------------

def _aggregate_distance(warm_path: str, raw_path: str) -> List[Dict]:
    """Per-pair schema: ``gr_total_ms``, ``pg_query_ms``,
    ``err_gr_vs_pg_m``.

    Accepts three input shapes per side:
      1. a directory (e.g. ``benchmark_v9/``) — globs per-combo
         ``*_x_*.csv`` files and concatenates them; this is the
         recommended form because both warm (``geometry_benchmark.py``)
         and raw (``raw_geometry_benchmark.py``) write per-combo CSVs;
      2. a single per-pair flat CSV (e.g. the raw
         ``raw_benchmark_smoke/summary.csv``) — used as-is;
      3. a pre-aggregated per-combo summary (e.g.
         ``benchmark_v9/summary.csv`` produced by
         ``geometry_benchmark_analysis.py``) — falls back to
         per-combo ``_median`` / ``_mean`` aggregation.
    """
    rows: List[Dict] = []
    if not (warm_path or raw_path):
        return rows
    warm_recs = _read_csv(warm_path)
    raw_recs = _read_csv(raw_path)

    def _is_preagg(recs: List[Dict]) -> bool:
        return bool(recs) and "gr_total_ms_median" in recs[0]

    def _stat_perpair(recs, key):
        return _stat(_to_float(r.get(key)) for r in recs)

    def _stat_preagg(recs, key):
        """Pool across combinations using per-combo aggregates.

        We approximate the pooled mean as the count-weighted mean of
        per-combo means, and the pooled median as the median of
        per-combo medians (good enough for the appendix table; the
        per-pair path remains the reference when available).
        """
        ws = [_to_float(r.get(f"{key}_count")) or 0 for r in recs]
        ms = [_to_float(r.get(f"{key}_mean")) for r in recs]
        meds = [_to_float(r.get(f"{key}_median"))
                for r in recs]
        meds = [v for v in meds if v is not None]
        wsum = sum(w for w, m in zip(ws, ms) if m is not None)
        msum = sum(w * m for w, m in zip(ws, ms) if m is not None)
        pooled_mean = (msum / wsum) if wsum > 0 else float("nan")
        pooled_median = statistics.median(meds) if meds else float("nan")
        return {
            "n": int(sum(ws)),
            "mean": pooled_mean,
            "median": pooled_median,
            "p95": float("nan"),
        }

    def _acc_perpair(recs):
        errs = [_to_float(r.get("err_gr_vs_pg_m")) for r in recs]
        errs = [abs(e) for e in errs if e is not None]
        if not errs:
            return None
        return sum(1 for e in errs if e < 0.05) / len(errs)

    def _acc_preagg(recs):
        # the warm summary keeps |err| stats but not the raw values;
        # we report the mean of per-combo error medians as a proxy and
        # convert <0.05m into a 1.0 indicator if mean is below 0.05.
        meds = [_to_float(r.get("err_gr_vs_pg_m_median"))
                for r in recs]
        meds = [abs(v) for v in meds if v is not None]
        if not meds:
            return None
        med = statistics.fmean(meds)
        return 1.0 if med < 0.05 else med

    if _is_preagg(warm_recs):
        warm_stat = _stat_preagg(warm_recs, "gr_total_ms")
        pg_warm = _stat_preagg(warm_recs, "pg_query_ms")
        warm_acc = _acc_preagg(warm_recs)
    else:
        warm_stat = _stat_perpair(warm_recs, "gr_total_ms")
        pg_warm = _stat_perpair(warm_recs, "pg_query_ms")
        warm_acc = _acc_perpair(warm_recs)

    if _is_preagg(raw_recs):
        raw_stat = _stat_preagg(raw_recs, "gr_total_ms")
        pg_raw = _stat_preagg(raw_recs, "pg_query_ms")
        raw_acc = _acc_preagg(raw_recs)
    else:
        raw_stat = _stat_perpair(raw_recs, "gr_total_ms")
        pg_raw = _stat_perpair(raw_recs, "pg_query_ms")
        raw_acc = _acc_perpair(raw_recs)

    # PG row: prefer warm side (kept warm in both modes), fall back to raw.
    pg_stat = pg_warm if pg_warm.get("n", 0) > 0 else pg_raw

    rows.append(_row(
        "Distance (all 9 P/L/Po combinations pooled)",
        warm_stat, raw_stat, pg_stat,
        accuracy_label="dist_match",
        accuracy_warm=warm_acc,
        accuracy_raw=raw_acc,
    ))
    return rows


def _aggregate_forward(warm_path: str, raw_path: str) -> List[Dict]:
    """Forward overlay: split by ``strategy`` (``fast_path`` vs
    ``polygon_fill``). Columns ``gr_time_ms`` (or ``t_gr_total_ms``),
    ``pg_time_ms``, ``f1``.
    """
    rows: List[Dict] = []
    if not (warm_path or raw_path):
        return rows
    warm_recs = _read_csv(warm_path)
    raw_recs = _read_csv(raw_path)

    def _gr_ms(r):
        return _to_float(r.get("gr_time_ms")) or _to_float(
            r.get("t_gr_total_ms"))

    def _pg_ms(r):
        return _to_float(r.get("pg_time_ms")) or _to_float(
            r.get("t_pg_ms"))

    strategies = sorted({(r.get("strategy") or "").strip()
                         for r in warm_recs + raw_recs
                         if (r.get("strategy") or "").strip()})
    # also produce a pooled row
    plans: List[Tuple[str, Callable[[Dict], bool]]] = [
        ("fast_path (LCP >= threshold)",
         lambda r: (r.get("strategy") or "").strip() == "fast_path"),
        ("polygon_fill (LCP < threshold)",
         lambda r: (r.get("strategy") or "").strip() == "polygon_fill"),
        ("pooled (all queries)",
         lambda r: True),
    ]
    # drop rows whose strategy is neither
    for label, pred in plans:
        if label.startswith("fast_path") and "fast_path" not in strategies:
            continue
        if label.startswith("polygon_fill") and "polygon_fill" not in strategies:
            continue
        w = [r for r in warm_recs if pred(r)]
        x = [r for r in raw_recs if pred(r)]
        warm_stat = _stat(_gr_ms(r) for r in w)
        raw_stat = _stat(_gr_ms(r) for r in x)
        pg_stat = _stat(_pg_ms(r) for r in (w + x))
        f1_w = [_to_float(r.get("f1")) for r in w]
        f1_w = [v for v in f1_w if v is not None]
        f1_x = [_to_float(r.get("f1")) for r in x]
        f1_x = [v for v in f1_x if v is not None]
        rows.append(_row(
            "Forward overlay — " + label,
            warm_stat, raw_stat, pg_stat,
            accuracy_label="F1",
            accuracy_warm=(statistics.fmean(f1_w) if f1_w else None),
            accuracy_raw=(statistics.fmean(f1_x) if f1_x else None),
        ))
    return rows


def _aggregate_reverse(warm_path: str, raw_path: str) -> List[Dict]:
    """Reverse overlay: split by hit status (``pip_hits > 0`` => Hit).
    Columns ``t_gr_total_ms``, ``t_pg_ms``, ``f1``, ``pip_hits``.
    """
    rows: List[Dict] = []
    if not (warm_path or raw_path):
        return rows
    warm_recs = _read_csv(warm_path)
    raw_recs = _read_csv(raw_path)

    def _hit(r):
        h = _to_float(r.get("pip_hits"))
        return h is not None and h > 0

    splits: List[Tuple[str, Callable[[Dict], bool]]] = [
        ("Hit (POI inside a parcel)", _hit),
        ("No-Hit (POI outside all parcels)",
         lambda r: not _hit(r)),
        ("pooled (all queries)", lambda r: True),
    ]
    for label, pred in splits:
        w = [r for r in warm_recs if pred(r)]
        x = [r for r in raw_recs if pred(r)]
        warm_stat = _stat(_to_float(r.get("t_gr_total_ms")) for r in w)
        raw_stat = _stat(_to_float(r.get("t_gr_total_ms")) for r in x)
        pg_stat = _stat(_to_float(r.get("t_pg_ms")) for r in (w + x))

        # F1: empty f1 on No-Hit => treat as 1.0 (consensus empty set)
        def _f1_filled(r):
            v = _to_float(r.get("f1"))
            if v is not None:
                return v
            m = _to_float(r.get("match"))
            return 1.0 if m == 1 else None
        f1_w = [v for v in (_f1_filled(r) for r in w) if v is not None]
        f1_x = [v for v in (_f1_filled(r) for r in x) if v is not None]
        rows.append(_row(
            "Reverse overlay — " + label,
            warm_stat, raw_stat, pg_stat,
            accuracy_label="F1",
            accuracy_warm=(statistics.fmean(f1_w) if f1_w else None),
            accuracy_raw=(statistics.fmean(f1_x) if f1_x else None),
        ))
    return rows


def _aggregate_knn(warm_path: str, raw_path: str) -> List[Dict]:
    """KNN: one row per (query, k). Columns ``k``, ``t_gr_us``,
    ``t_pgs_us``, ``recall_at_k``, ``query_code``.

    Auto-fairness filter: if warm has many more queries than raw (e.g.
    warm ran on 100k random POIs, raw on 10k), we restrict warm to the
    intersection of ``query_code`` values so the two columns describe
    exactly the same query set. Otherwise random-sample variance can
    make raw appear *faster* than warm just because the smaller raw
    sample happens to draw easier neighborhoods.
    """
    rows: List[Dict] = []
    if not (warm_path or raw_path):
        return rows
    warm_recs = _read_csv(warm_path)
    raw_recs = _read_csv(raw_path)

    # We report only K=20 (the headline number used in §4.5.3).
    ks_to_report = ["20"]
    for kv in ks_to_report:
        w = [r for r in warm_recs if str(r.get("k", "")).strip() == kv]
        x = [r for r in raw_recs if str(r.get("k", "")).strip() == kv]

        # Apples-to-apples filter: only keep warm queries that also
        # appear in raw (and vice versa). Skip if either side lacks
        # the query_code column, or if the intersection is empty.
        label_suffix = ""
        if w and x and "query_code" in w[0] and "query_code" in x[0]:
            raw_keys = {str(r.get("query_code") or "").strip()
                        for r in x}
            warm_keys = {str(r.get("query_code") or "").strip()
                         for r in w}
            inter = raw_keys & warm_keys
            if inter and len(inter) < max(len(raw_keys), len(warm_keys)):
                w_filtered = [r for r in w
                              if str(r.get("query_code") or "").strip()
                              in inter]
                x_filtered = [r for r in x
                              if str(r.get("query_code") or "").strip()
                              in inter]
                w = w_filtered
                x = x_filtered
                label_suffix = (
                    f"; warm filtered to {len(inter):,}-query "
                    f"intersection for fair comparison")

        # values are in microseconds; convert to ms for consistency
        warm_stat = _stat(
            ((_to_float(r.get("t_gr_us")) or 0) / 1000.0) for r in w)
        raw_stat = _stat(
            ((_to_float(r.get("t_gr_us")) or 0) / 1000.0) for r in x)
        pg_stat = _stat(
            ((_to_float(r.get("t_pgs_us")) or 0) / 1000.0)
            for r in (w + x))
        rec_w = [_to_float(r.get("recall_at_k")) for r in w]
        rec_w = [v for v in rec_w if v is not None]
        rec_x = [_to_float(r.get("recall_at_k")) for r in x]
        rec_x = [v for v in rec_x if v is not None]
        rows.append(_row(
            f"KNN (K={kv}, POI query points{label_suffix})",
            warm_stat, raw_stat, pg_stat,
            accuracy_label="Recall@K",
            accuracy_warm=(statistics.fmean(rec_w) if rec_w else None),
            accuracy_raw=(statistics.fmean(rec_x) if rec_x else None),
        ))
    return rows


# ----------------------------------------------------------------------
# Markdown table writer
# ----------------------------------------------------------------------

_MD_HEADER = (
    "| Operator / subset | n (warm/raw) | "
    "G3R warm median (ms) | G3R raw median (ms) | "
    "Raw / Warm | PG median (ms) | "
    "Warm speedup PG/G3R | Raw speedup PG/G3R | "
    "Accuracy metric | Accuracy (warm / raw) |"
)
_MD_SEP = (
    "|---|---|---|---|---|---|---|---|---|---|"
)


def _to_markdown(rows: List[Dict]) -> str:
    lines = [_MD_HEADER, _MD_SEP]
    for r in rows:
        n_field = f"{r['n_warm']} / {r['n_raw']}"
        acc_field = f"{r.get('accuracy_warm', '—')} / {r.get('accuracy_raw', '—')}"
        lines.append(
            f"| {r['subset']} | {n_field} | "
            f"{r['g3r_warm_median_ms']} | {r['g3r_raw_median_ms']} | "
            f"{r['raw_to_warm_ratio']} | {r['pg_median_ms']} | "
            f"{r['warm_speedup_vs_pg']} | {r['raw_speedup_vs_pg']} | "
            f"{r.get('accuracy_metric', '—')} | {acc_field} |"
        )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate warm-cache vs raw-traversal CSVs into a single "
            "appendix table (Markdown + CSV)."),
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--distance-warm", default="",
                   help="benchmark_v9/summary.csv (or similar)")
    p.add_argument("--distance-raw", default="",
                   help="raw_benchmark_v9/summary.csv (or similar)")
    p.add_argument("--forward-warm", default="",
                   help="csv/overlay_v7_warm_check2.csv")
    p.add_argument("--forward-raw", default="",
                   help="csv/raw_overlay_forward.csv")
    p.add_argument("--reverse-warm", default="",
                   help="csv/reverse_warm.csv")
    p.add_argument("--reverse-raw", default="",
                   help="csv/raw_reverse_overlay.csv")
    p.add_argument("--knn-warm", default="",
                   help="knn_v31_100k.csv")
    p.add_argument("--knn-raw", default="",
                   help="csv/raw_knn_n100k.csv")
    p.add_argument("--out-md", default="appendix_warm_vs_raw.md")
    p.add_argument("--out-csv", default="appendix_warm_vs_raw.csv")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    rows: List[Dict] = []
    rows.extend(_aggregate_distance(args.distance_warm, args.distance_raw))
    rows.extend(_aggregate_forward(args.forward_warm, args.forward_raw))
    rows.extend(_aggregate_reverse(args.reverse_warm, args.reverse_raw))
    rows.extend(_aggregate_knn(args.knn_warm, args.knn_raw))

    if not rows:
        sys.exit("error: no CSVs provided. Pass at least one "
                 "--*-warm / --*-raw pair.")

    md = _to_markdown(rows)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(
            "# Appendix Table — warm-cache lookups vs runtime "
            "graph traversal on Neo4j\n\n"
            "*Both modes execute the same candidate-retrieval and "
            "verification pattern (Section 2.6). The warm mode uses "
            "the traversal-equivalent in-memory materializations "
            "(centroid-code index, LCPC inverted index, parcel "
            "polygon cache); the raw mode disables all of them and "
            "issues a fresh Cypher round-trip per per-query operation. "
            "PostGIS is kept under `pg_prewarm` in both modes. Accuracy "
            "(F1 for overlay; Recall@K for KNN; |err| < 0.05 m for "
            "distance) is unchanged because the candidate semantics "
            "are identical.*\n\n"
        )
        f.write(md)
    print(f"[md ] {args.out_md} ({len(rows)} rows)")

    # Collect fieldnames across ALL rows (preserve row[0] order, append
    # any extra keys later rows introduce). Otherwise DictWriter would
    # raise ValueError on the first row that has an extra key.
    fieldnames: List[str] = list(rows[0].keys())
    seen = set(fieldnames)
    for r in rows[1:]:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] {args.out_csv} ({len(rows)} rows)")

    # also echo to stdout for quick eyeballing
    print()
    print(md)


if __name__ == "__main__":
    main()

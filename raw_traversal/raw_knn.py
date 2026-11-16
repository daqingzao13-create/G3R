#!/usr/bin/env python3
# coding: utf-8
"""
Raw Neo4j traversal — KNN retrieval.

Disables the ``POIPrefixCache`` client-side warm cache so that every
``_query_candidates`` call inside the searcher issues a Cypher
``UNWIND $prefixes AS prefix MATCH (c:GeoSOTCentroidCode) ... STARTS WITH``
round-trip — the genuine runtime graph traversal pattern from Table 4.

Implementation strategy: this driver re-uses the already-supported
``--no-poi-cache`` flag of ``geosot_knn_v3_1.py`` (which calls
``set_poi_prefix_index(None)``), then post-processes the CSV to append
the ``mode='raw_traversal'`` column expected by the appendix
aggregator. We do not modify the original script.

PostGIS is left in its usual warm state. Set
``GEOSOT_SKIP_PG_PREWARM=1`` to evaluate cold PG.

Usage::

    python raw_traversal/raw_knn.py --queries 10000 --k 20 \\
        --output csv/raw_knn_n10k.csv --seed 42

    # 100k queries (paper-scale corpus):
    python raw_traversal/raw_knn.py --queries 100000 --k 20 \\
        --output csv/raw_knn_n100k.csv --seed 42

Passthrough flags (everything below ``--`` is forwarded to
``geosot_knn_v3_1.py``)::

    python raw_traversal/raw_knn.py --queries 10000 \\
        --output csv/raw_knn_n10k.csv -- \\
        --reuse-pg-csv knn_v31_100k.csv
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import subprocess
import sys
import time
from typing import List

from _path_setup import ensure_geosot_on_path

PARENT = ensure_geosot_on_path()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Raw Neo4j traversal — KNN. Wraps geosot_knn_v3_1.py with "
            "--no-poi-cache so every prefix lookup is a fresh Cypher "
            "round-trip. Output CSV gets a mode='raw_traversal' column."),
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--queries", type=int, default=10000,
                   help="number of POI query points (default 10000)")
    p.add_argument("--k", default="20",
                   help="comma-separated K values (default '20')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True,
                   help="output CSV path")
    p.add_argument("--margin", type=float, default=100.0)
    p.add_argument("--min-level", type=int, default=8)
    p.add_argument("--max-widen", type=int, default=5)
    p.add_argument("--max-cand-per-level", type=int, default=20000)
    p.add_argument("--start-level-max", type=int, default=13)
    p.add_argument("--no-geometry", action="store_true")
    p.add_argument("--pg-dedup", action="store_true")
    p.add_argument(
        "--reuse-pg-csv", default="",
        help="reuse PostGIS top-K + latencies from a prior warm-run CSV; "
             "PG will not be re-queried (recommended to isolate the "
             "G3R-side warm-vs-raw delta).")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    script = os.path.join(PARENT, "geosot_knn_v3_1.py")
    if not os.path.exists(script):
        sys.exit(f"error: cannot find {script}")

    cmd: List[str] = [
        sys.executable, script,
        "--queries", str(args.queries),
        "--k", args.k,
        "--seed", str(args.seed),
        "--margin", f"{args.margin:g}",
        "--min-level", str(args.min_level),
        "--max-widen", str(args.max_widen),
        "--max-cand-per-level", str(args.max_cand_per_level),
        "--start-level-max", str(args.start_level_max),
        "--output", args.output,
        "--no-poi-cache",                # <-- key flag: disable warm cache
    ]
    if args.no_geometry:
        cmd.append("--no-geometry")
    if args.pg_dedup:
        cmd.append("--pg-dedup")
    if args.reuse_pg_csv:
        cmd.extend(["--reuse-pg-csv", args.reuse_pg_csv])

    print("=" * 76)
    print("G3R RAW Neo4j traversal — KNN")
    print("  (POIPrefixCache disabled; per-prefix STARTS WITH via Cypher)")
    print("=" * 76)
    print("  delegating to: geosot_knn_v3_1.py --no-poi-cache ...")
    print("  full command:")
    print("    " + " ".join(cmd))
    print()

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=PARENT)
    dt = time.perf_counter() - t0
    if rc != 0:
        sys.exit(f"\nKNN runner exited with code {rc}")
    print(f"\nKNN runner finished in {dt:.1f} s.")

    # Post-process: append `mode='raw_traversal'` column.
    if not os.path.exists(args.output):
        sys.exit(f"error: expected output CSV not found: {args.output}")

    with open(args.output, "r", encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        fields = (reader.fieldnames or [])
        if "mode" not in fields:
            fields = fields + ["mode"]
        data = list(reader)
    for r in data:
        r["mode"] = "raw_traversal"
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in data:
            writer.writerow(r)
    print(f"  CSV finalized with mode='raw_traversal': "
          f"{args.output} ({len(data)} rows)")


if __name__ == "__main__":
    main()

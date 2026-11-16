#!/usr/bin/env python3
# coding: utf-8
"""
Raw Neo4j traversal — reverse overlay (POI -> containing parcel).

Reproduces the §4.5.2 reverse overlay experiment with all G3R warm-cache
materializations (Section 2.6) disabled:

    * ``PlanPolygonCache`` is NOT preloaded — every candidate parcel's
      vertex codes are fetched on-demand through
      ``_fetch_polygons_for_codes`` (UNWIND + ``MATCH HAS_VERTEX_CODE``).
    * ``CommonPrefixIndex`` is NOT injected — the LCPC-based prefix
      lookup goes through the original Cypher pattern
      (``MATCH (p:GeoSOTCommonPrefixCode {prefix_code: $pf}) ...
        -[:HAS_COMMON_PREFIX_CODE]- (parcel)``) for every POI, every tier.

PostGIS is kept warm to match the manuscript baseline; set
``GEOSOT_SKIP_PG_PREWARM=1`` to evaluate cold PG as well.

CSV schema matches ``geosot_reverse_query.py`` (so existing analysis
scripts can read it) plus a trailing ``mode='raw_traversal'`` column.

Usage::

    python raw_traversal/raw_overlay_reverse.py \\
        --batch 10000 --seed 42 --strategy prefix \\
        --output csv/raw_reverse_overlay_n10k.csv

    python raw_traversal/raw_overlay_reverse.py \\
        --all --output csv/raw_reverse_overlay_all.csv
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import random
import sys
import time

import psycopg2
from neo4j import GraphDatabase

from _path_setup import ensure_geosot_on_path

ensure_geosot_on_path()

import geosot_reverse_query as rq  # noqa: E402
from geosot_reverse_query import (  # noqa: E402
    CSV_FIELDS,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    SOURCE_TO_PG_TABLE,
    _fetch_poi_codes,
    run_batch,
    set_common_prefix_index,
)
from geosot_spatial_reasoning import PG_CONFIG  # noqa: E402
from bench_setup import prepare_pg_warm  # noqa: E402


def _read_codes_from_csv(path: str) -> list:
    """
    Read poi_code from a previous run's CSV while preserving order and duplicates to reproduce
    the query sequence.
    """
    codes = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in _csv.DictReader(f):
            c = (r.get("poi_code") or "").strip().upper()
            if c:
                codes.append(c)
    return codes


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Raw Neo4j traversal — reverse overlay (POI -> parcel). "
            "No PlanPolygonCache, no CommonPrefixIndex. PG kept warm."),
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--poi-source", default="POI")
    p.add_argument("--parcel-source", default="Statutory Plan")
    p.add_argument("--batch", type=int, default=0,
                   help="random POI sample size; ignored when --all "
                        "or --codes-from-csv")
    p.add_argument("--all", action="store_true",
                   help="iterate over every POI")
    p.add_argument("--codes-from-csv", default="",
                   help="read the exact POI query set (poi_code column, in "
                        "order) from a prior run's CSV, e.g. "
                        "csv/reverse_warm.csv. Guarantees raw evaluates the "
                        "identical query set as warm -> Hit/No-Hit counts "
                        "become directly comparable. Overrides --batch/--seed.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategy", default="prefix",
                   choices=["prefix", "located_in", "neighbor"],
                   help="cascade prefix (default; matches main paper)")
    p.add_argument("--output", default="",
                   help="output CSV path (default: timestamped)")
    p.add_argument("--no-postgis", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    compare_pg = not args.no_postgis
    if args.batch <= 0 and not args.all and not args.codes_from_csv:
        sys.exit("error: --all, --batch N, or --codes-from-csv required")

    W = 76
    print("=" * W)
    print("G3R RAW Neo4j traversal — reverse overlay")
    print("  (no PlanPolygonCache, no CommonPrefixIndex; per-query")
    print("   prefix lookups + per-batch HAS_VERTEX_CODE fetch via Cypher)")
    print("=" * W)
    print(f"  POI source:     {args.poi_source}")
    print(f"  parcel source:  {args.parcel_source}")
    print(f"  strategy:       {args.strategy}")

    # [1] Connect
    print("\n[1] connecting ...")
    t0 = time.perf_counter()
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    session = driver.session(database=NEO4J_DATABASE)
    session.run("RETURN 1").consume()
    t_n = (time.perf_counter() - t0) * 1000
    print(f"  Neo4j:   {NEO4J_URI} / {NEO4J_DATABASE} ({t_n:.0f} ms)")

    pg_conn = None
    if compare_pg:
        pg_conn = psycopg2.connect(**PG_CONFIG)
        print(f"  PostGIS: {PG_CONFIG['host']}:{PG_CONFIG['port']}/"
              f"{PG_CONFIG['dbname']}")
        if os.getenv("GEOSOT_SKIP_PG_PREWARM") != "1":
            used_tables = sorted({
                tbl for tbl in (
                    SOURCE_TO_PG_TABLE.get(args.poi_source),
                    SOURCE_TO_PG_TABLE.get(args.parcel_source),
                ) if tbl})
            pt = SOURCE_TO_PG_TABLE.get(args.poi_source, "")
            tt = SOURCE_TO_PG_TABLE.get(args.parcel_source, "")
            tw_sqls = [(f"SELECT main_code FROM {t} LIMIT 50", None)
                       for t in used_tables]
            if pt and tt:
                tw_sqls.append((
                    f"SELECT p.main_code FROM {tt} p, {pt} t "
                    f"WHERE t.main_code = "
                    f"(SELECT main_code FROM {pt} LIMIT 1) "
                    f"AND ST_Contains(p.geometry, t.geometry) LIMIT 1",
                    None))
            prepare_pg_warm(pg_conn, used_tables, tw_sqls)

    # CRITICAL: defeat warm-cache materialization on the G3R side.
    set_common_prefix_index(None)
    # Strategy 'neighbor' would want EAST_OF/NORTH_OF cache; we leave it
    # unbuilt as well so the comparison is symmetric.

    # [2] Build POI query list
    if args.codes_from_csv:
        # Fixed query set: exactly reproduce a prior run (e.g. reverse_warm.csv)
        # so Hit/No-Hit counts are directly comparable (option 1 of the
        # appendix-D consistency fix). No sampling, no dependence on the
        # (non-deterministic) SELECT ... row order.
        print(f"\n[2] reading fixed POI query set from "
              f"{args.codes_from_csv} ...")
        poi_codes = _read_codes_from_csv(args.codes_from_csv)
        if not poi_codes:
            sys.exit(f"error: no poi_code found in {args.codes_from_csv}")
        print(f"  fixed query set: {len(poi_codes):,} POIs "
              f"(same as source CSV)")
    else:
        print(f"\n[2] fetching {args.poi_source} POI codes ...")
        all_codes = _fetch_poi_codes(session, pg_conn, args.poi_source)
        print(f"  total POIs: {len(all_codes):,}")
        if args.all:
            poi_codes = all_codes
        else:
            random.seed(args.seed)
            poi_codes = random.sample(
                all_codes, min(args.batch, len(all_codes)))
            print(f"  sampled: {len(poi_codes)}")

    output_csv = args.output
    if not output_csv:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = (
            f"raw_reverse_overlay_{args.strategy}_"
            f"N{len(poi_codes)}_{ts}.csv")

    # [3] Run — polygon_cache=None forces per-query polygon fetch via
    # _fetch_polygons_for_codes (batched HAS_VERTEX_CODE UNWIND, but per
    # candidate batch only, not the 40k all-parcel preload).
    print(f"\n[3] running {len(poi_codes):,} queries "
          f"(polygon_cache=None, cp_index=None) ...")
    rows = run_batch(
        session, pg_conn,
        poi_codes,
        args.poi_source, args.parcel_source,
        compare_pg, strategy=args.strategy,
        output_csv=output_csv,
        polygon_cache=None,    # <-- raw mode
        nb_cache=None,
        hit_only=0,
    )

    # Append `mode` column for the appendix aggregator
    if output_csv and rows:
        with open(output_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = _csv.DictReader(f)
            fields = (reader.fieldnames or []) + ["mode"]
            data = list(reader)
        for r in data:
            r["mode"] = "raw_traversal"
        with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in data:
                writer.writerow(r)
        print(f"  CSV finalized with mode='raw_traversal': "
              f"{output_csv} ({len(data)} rows)")

    session.close()
    driver.close()
    if pg_conn:
        pg_conn.close()
    print("\ndone.")


if __name__ == "__main__":
    main()

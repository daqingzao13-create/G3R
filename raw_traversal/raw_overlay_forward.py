#!/usr/bin/env python3
# coding: utf-8
"""
Raw Neo4j traversal — forward overlay (statutory plan parcel -> POI).

Runs the same forward overlay algorithm as ``geosot_overlay_query_v7.py``
(adaptive strategy: ``fast_path`` when LCP is deep, ``polygon_fill``
otherwise; winding-number PIP refinement on boundary candidates), but
DISABLES every G3R warm-cache materialization listed in Section 2.6 of
the manuscript:

    * ``PlanPreloaderV6`` is NOT instantiated — every query fetches its
      own plan polygon's vertex codes through a fresh Cypher round-trip
      (``_fetch_polygon_vcodes_from_neo4j``).
    * ``MainCodeIndex`` is NOT injected — every per-prefix candidate
      lookup goes through ``UNWIND $prefixes ... MATCH ... STARTS WITH``
      against Neo4j (i.e. the genuine graph traversal pattern from
      Table 4, instead of a client-side bisect over a sorted in-memory
      array of POI codes).
    * The module-level ``_prefix_cache`` is cleared **after every
      query** so cross-query memoization cannot leak into per-query
      latencies. (Within a single query, repeated lookups for the same
      prefix still hit the in-flight Python dict — this is just call
      sharing, not cross-query warm cache.)

PostGIS is kept warm (``pg_prewarm`` + ``GiST/B-tree`` resident) so the
appendix table isolates the G3R-side overhead and matches the
manuscript's PG warm baseline.

CSV schema matches ``geosot_overlay_query_v7.py``'s output exactly,
plus a trailing ``mode='raw_traversal'`` column so the warm CSV
(``csv/overlay_v7_warm_check2.csv``) and the raw CSV can be loaded
together by ``raw_appendix_table.py``.

Usage::

    python raw_traversal/raw_overlay_forward.py \\
        --source POI --query-source "Statutory Plan" \\
        --output csv/raw_overlay_forward.csv --all

    # quick smoke (200 random parcels):
    python raw_traversal/raw_overlay_forward.py \\
        --source POI --batch 200 --seed 42 \\
        --output csv/raw_overlay_forward_n200.csv
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import psycopg2
from neo4j import GraphDatabase

from _path_setup import ensure_geosot_on_path

ensure_geosot_on_path()

import geosot_overlay_query_v7 as ovq  # noqa: E402
from geosot_overlay_query_v7 import (  # noqa: E402
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    REFINE_LEVELS,
    SOURCE_TO_PG_TABLE,
    _fetch_all_plan_codes,
    prefix_cache_clear,
    run_batch,
    set_main_code_index,
)
from geosot_spatial_reasoning import PG_CONFIG, is_valid_g0_code  # noqa: E402
from bench_setup import prepare_pg_warm  # noqa: E402


def _install_per_query_cache_purge() -> None:
    """Monkey-patch the prefix-cache helpers so each ``_query_neo4j_*``
    call only sees an empty cache.

    The implementation wraps the existing ``_query_neo4j_by_cells`` and
    ``_query_neo4j_split`` functions. Both are called at most once per
    query in ``run_batch``; clearing the module-level ``_prefix_cache``
    immediately before each invocation guarantees every per-prefix
    ``STARTS WITH`` lookup goes through a fresh Cypher round-trip.

    We keep the in-call dict (so multiple prefixes inside one query
    still de-duplicate). The point of "raw" mode is to forbid cross-
    query memoization, not to disable intra-query batching.
    """
    orig_by_cells = ovq._query_neo4j_by_cells
    orig_split = ovq._query_neo4j_split

    def _wrapped_by_cells(session, cells, source, batch_size=1024):
        prefix_cache_clear()
        return orig_by_cells(session, cells, source, batch_size=batch_size)

    def _wrapped_split(session, inside_cells, boundary_cells, source,
                       batch_size=1024):
        prefix_cache_clear()
        return orig_split(
            session, inside_cells, boundary_cells, source,
            batch_size=batch_size)

    ovq._query_neo4j_by_cells = _wrapped_by_cells
    ovq._query_neo4j_split = _wrapped_split


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Raw Neo4j traversal — forward overlay (parcel -> POI). "
            "No PlanPreloaderV6, no MainCodeIndex, no cross-query "
            "prefix cache. PG kept warm."),
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", default="POI",
                   help="target source (default: POI)")
    p.add_argument("--query-source", default="Statutory Plan",
                   help="query polygon source (default: Statutory Plan)")
    p.add_argument("--batch", type=int, default=0,
                   help="random sample size; ignored when --all is set")
    p.add_argument("--all", action="store_true",
                   help="iterate over every query polygon (full corpus)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="",
                   help="output CSV path (default: raw_overlay_forward_<ts>.csv)")
    p.add_argument("--no-postgis", action="store_true")
    p.add_argument("--strategy", default="adaptive",
                   help="must stay 'adaptive' to mirror the warm run")
    p.add_argument("--max-cells", type=int, default=256)
    p.add_argument("--refine", type=int, default=REFINE_LEVELS)
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    compare_pg = not args.no_postgis
    if args.batch <= 0 and not args.all:
        sys.exit("error: --all or --batch N required")

    W = 76
    print("=" * W)
    print("G3R RAW Neo4j traversal — forward overlay (V7 adaptive)")
    print("  (no PlanPreloaderV6, no MainCodeIndex, "
          "_prefix_cache cleared per query;")
    print("   PG side prewarmed unless GEOSOT_SKIP_PG_PREWARM=1)")
    print("=" * W)
    print(f"  target source:  {args.source}")
    print(f"  query source:   {args.query_source}")
    print(f"  max_cells:      {args.max_cells}")

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
                    SOURCE_TO_PG_TABLE.get(args.source),
                    SOURCE_TO_PG_TABLE.get(args.query_source),
                ) if tbl})
            qt = SOURCE_TO_PG_TABLE.get(args.query_source, "")
            tt = SOURCE_TO_PG_TABLE.get(args.source, "")
            tw_sqls = [(f"SELECT main_code FROM {t} LIMIT 50", None)
                       for t in used_tables]
            if qt and tt:
                tw_sqls.append((
                    f"SELECT t.main_code FROM {tt} t, {qt} q "
                    f"WHERE q.main_code = "
                    f"(SELECT main_code FROM {qt} LIMIT 1) "
                    f"AND ST_Intersects(t.geometry, q.geometry) LIMIT 1",
                    None))
            prepare_pg_warm(pg_conn, used_tables, tw_sqls)

    # CRITICAL: ensure no warm cache is materialized.
    set_main_code_index(None)
    prefix_cache_clear()
    _install_per_query_cache_purge()

    # [2] Get query polygons
    print(f"\n[2] fetching {args.query_source} main_codes ...")
    all_codes = _fetch_all_plan_codes(
        session, pg_conn, args.query_source)
    all_codes = [c for c in all_codes if is_valid_g0_code(c)]
    print(f"  total: {len(all_codes):,}")

    if args.all:
        plan_codes = all_codes
    else:
        random.seed(args.seed)
        plan_codes = random.sample(
            all_codes, min(args.batch, len(all_codes)))
        print(f"  sampled: {len(plan_codes)}")

    output_csv = args.output
    if not output_csv:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_lbl = len(plan_codes)
        output_csv = f"raw_overlay_forward_N{n_lbl}_{ts}.csv"

    # [3] Run — plan_preloader=None forces per-query polygon fetch.
    # The wrapped _query_neo4j_* helpers also clear the module-level
    # cache before each query, so cross-query memoization cannot
    # contaminate the timings.
    print(f"\n[3] running {len(plan_codes):,} queries "
          f"(raw runtime traversal, plan_preloader=None) ...")
    rows = run_batch(
        session, pg_conn,
        plan_codes,
        args.source, args.query_source,
        compare_pg, strategy=args.strategy,
        max_cells=args.max_cells,
        refine_levels=args.refine,
        force_fast_min_prefix=False,
        enable_final_pg_filter=False,
        enable_final_strict_filter=False,
        all_candidates_pip=False,
        mbr_cover_pip=False,
        mbr_cover_safety_shrink=0,
        output_csv=output_csv,
        plan_preloader=None,   # <-- raw mode: NO warm polygon dict
        poi_cache=None,
    )

    # Append the mode column (after the standard schema) so the appendix
    # aggregator can stack warm and raw CSVs.
    if output_csv and rows:
        import csv as _csv
        # Read the freshly-written CSV and rewrite it with a `mode` column
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            fields = (reader.fieldnames or []) + ["mode"]
            data = list(reader)
        for r in data:
            r["mode"] = "raw_traversal"
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
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

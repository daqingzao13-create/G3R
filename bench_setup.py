#!/usr/bin/env python3
# coding: utf-8
"""
Shared PostGIS setup protocol for geometry, forward/reverse overlay, and KNN benchmarks. G3R
preloads vertex/neighbor/code indexes into client memory; PG prewarms tables plus
GiST/B-tree indexes into shared_buffers and runs disposable queries before measurement.
Report setup costs separately and run warmup queries on both sides.

API: prepare_pg_warm, pg_prewarm_tables, discover_table_indexes. GEOSOT_SKIP_PG_PREWARM=1
skips prewarming; GEOSOT_PREWARM_VERBOSE=0 suppresses progress. These controls do not
themselves flush database/OS caches.

Example:
    from bench_setup import prepare_pg_warm
    stats = prepare_pg_warm(pg_conn, tables=["public.poi_new", "public.statutory_plan"], throwaway_queries=[("SELECT main_code FROM public.poi_new LIMIT 100", None)])
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence, Tuple

__all__ = [
    "prepare_pg_warm",
    "pg_prewarm_tables",
    "discover_table_indexes",
    "ensure_pg_prewarm_extension",
    "split_table_name",
]


def _verbose_default() -> bool:
    """Read GEOSOT_PREWARM_VERBOSE; default to True."""
    val = os.getenv("GEOSOT_PREWARM_VERBOSE", "1")
    return val not in ("0", "false", "False", "")


def split_table_name(table_full_name: str) -> Tuple[str, str]:
    """
    Split schema.table into (schema,table), defaulting to schema public.

    @param table_full_name Qualified or unqualified table name.
    @returns (schema,table).
    """
    parts = table_full_name.split(".")
    if len(parts) == 1:
        return "public", parts[0]
    return parts[0], parts[1]


def ensure_pg_prewarm_extension(pg_conn) -> bool:
    """
    Ensure pg_prewarm is available.

    @param pg_conn psycopg2 connection.
    @returns True if available, otherwise False, for example when privileges or extension files
        are missing.
    """
    try:
        cur = pg_conn.cursor()
        cur.execute(
            "SELECT 1 FROM pg_extension "
            "WHERE extname = 'pg_prewarm' LIMIT 1"
        )
        already = cur.fetchone() is not None
        if not already:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
            pg_conn.commit()
        cur.close()
        return True
    except Exception as e:
        try:
            pg_conn.rollback()
        except Exception:
            pass
        print(f"[bench_setup] pg_prewarm 扩展不可用: {e}")
        return False


def discover_table_indexes(
    pg_conn, table_full_name: str,
) -> list:
    """
    List schema-qualified indexes on a table for prewarming.

    @param pg_conn psycopg2 connection.
    @param table_full_name Qualified table name.
    @returns Qualified index names.
    """
    schema, table = split_table_name(table_full_name)
    try:
        cur = pg_conn.cursor()
        cur.execute(
            "SELECT i.relname "
            "FROM pg_class t "
            "JOIN pg_namespace n ON n.oid = t.relnamespace "
            "JOIN pg_index ix ON t.oid = ix.indrelid "
            "JOIN pg_class i ON i.oid = ix.indexrelid "
            "WHERE n.nspname = %s AND t.relname = %s",
            (schema, table),
        )
        names = [f"{schema}.{r[0]}" for r in cur.fetchall()]
        cur.close()
        return names
    except Exception as e:
        try:
            pg_conn.rollback()
        except Exception:
            pass
        print(f"[bench_setup] 索引发现失败 {table_full_name}: {e}")
        return []


def _table_exists(pg_conn, table_full_name: str) -> bool:
    """Check whether a regular table or materialized view exists."""
    schema, table = split_table_name(table_full_name)
    try:
        cur = pg_conn.cursor()
        cur.execute(
            "SELECT 1 "
            "FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s AND c.relname = %s "
            "  AND c.relkind IN ('r', 'm') "  # r=table, m=matview
            "LIMIT 1",
            (schema, table),
        )
        ok = cur.fetchone() is not None
        cur.close()
        return ok
    except Exception:
        try:
            pg_conn.rollback()
        except Exception:
            pass
        return False


def pg_prewarm_tables(
    pg_conn,
    tables: Sequence[str],
    include_indexes: bool = True,
    verbose: Optional[bool] = None,
) -> Tuple[float, int, int, int]:
    """
    Call pg_prewarm for tables and optionally their indexes.

    @param pg_conn psycopg2 connection.
    @param tables Qualified table names.
    @param include_indexes Whether to prewarm indexes; default True.
    @param verbose Progress setting; None reads GEOSOT_PREWARM_VERBOSE.
    @returns (elapsed seconds, table count, index count, loaded page count).
    """
    if verbose is None:
        verbose = _verbose_default()

    if not ensure_pg_prewarm_extension(pg_conn):
        return 0.0, 0, 0, 0

    t0 = time.perf_counter()
    n_tables = 0
    n_indexes = 0
    total_pages = 0

    for tbl in tables:
        if not _table_exists(pg_conn, tbl):
            if verbose:
                print(f"  [pg_prewarm] SKIP {tbl} (表不存在)")
            continue

        try:
            cur = pg_conn.cursor()
            cur.execute("SELECT pg_prewarm(%s)", (tbl,))
            n_pages = int(cur.fetchone()[0])
            cur.close()
            n_tables += 1
            total_pages += n_pages
            if verbose:
                print(f"  [pg_prewarm] {tbl:<40s}  pages={n_pages:>9,}")
        except Exception as e:
            try:
                pg_conn.rollback()
            except Exception:
                pass
            if verbose:
                print(f"  [pg_prewarm] SKIP {tbl}: {e}")
            continue

        if include_indexes:
            for idx in discover_table_indexes(pg_conn, tbl):
                try:
                    cur = pg_conn.cursor()
                    cur.execute("SELECT pg_prewarm(%s)", (idx,))
                    n_pages = int(cur.fetchone()[0])
                    cur.close()
                    n_indexes += 1
                    total_pages += n_pages
                    if verbose:
                        print(f"    [idx] {idx:<38s}  pages={n_pages:>9,}")
                except Exception as e:
                    try:
                        pg_conn.rollback()
                    except Exception:
                        pass
                    if verbose:
                        print(f"    [idx] SKIP {idx}: {e}")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  [pg_prewarm] 完成: {n_tables} 表 + {n_indexes} 索引, "
              f"共 {total_pages:,} 页, 耗时 {elapsed * 1000:.0f} ms")
    return elapsed, n_tables, n_indexes, total_pages


def _run_throwaway(
    pg_conn,
    queries: Sequence[Tuple[str, Optional[Tuple]]],
    verbose: bool,
) -> Tuple[float, int]:
    """
    Run disposable queries for plan/JIT warmup.

    @returns (total seconds, successful query count).
    """
    if not queries:
        return 0.0, 0
    t0 = time.perf_counter()
    n_ok = 0
    for sql, params in queries:
        try:
            cur = pg_conn.cursor()
            if params is None:
                cur.execute(sql)
            else:
                cur.execute(sql, params)
            if cur.description is not None:
                _ = cur.fetchall()
            cur.close()
            n_ok += 1
        except Exception as e:
            try:
                pg_conn.rollback()
            except Exception:
                pass
            if verbose:
                short = sql.replace("\n", " ")[:60]
                print(f"  [throwaway] SKIP ({short}...): {e}")
    return time.perf_counter() - t0, n_ok


def prepare_pg_warm(
    pg_conn,
    tables: Sequence[str],
    throwaway_queries: Optional[Sequence[Tuple[str, Optional[Tuple]]]] = None,
    include_indexes: bool = True,
    verbose: Optional[bool] = None,
) -> dict:
    """
    Prewarm PostGIS relations/indexes, then execute disposable warmup queries before
    measurement. Report these costs separately alongside G3R materialization; prewarming does
    not make the two architectures identical.

    @param pg_conn psycopg2 connection.
    @param tables All relevant schema-qualified tables.
    @param throwaway_queries [(sql,params), ...], representative of benchmark queries.
    @param include_indexes Whether to prewarm indexes.
    @param verbose Progress setting; None reads GEOSOT_PREWARM_VERBOSE.
    @returns {prewarm_sec, throwaway_sec, total_sec, n_tables, n_indexes, n_throwaway, n_pages,
        skipped}.
    """
    if verbose is None:
        verbose = _verbose_default()

    if os.getenv("GEOSOT_SKIP_PG_PREWARM") == "1":
        if verbose:
            print("[bench_setup] GEOSOT_SKIP_PG_PREWARM=1, "
                  "跳过 prewarm (cold-cache 对照模式)")
        return {
            "prewarm_sec": 0.0,
            "throwaway_sec": 0.0,
            "total_sec": 0.0,
            "n_tables": 0,
            "n_indexes": 0,
            "n_throwaway": 0,
            "n_pages": 0,
            "skipped": True,
        }

    if verbose:
        print()
        print("─" * 76)
        print("  [bench_setup] PostGIS 预热 (pg_prewarm + throwaway)")
        print("─" * 76)

    prewarm_sec, n_t, n_i, n_pages = pg_prewarm_tables(
        pg_conn, tables, include_indexes=include_indexes, verbose=verbose)

    throwaway_sec, n_throwaway = _run_throwaway(
        pg_conn, throwaway_queries or [], verbose=verbose)
    if verbose and n_throwaway > 0:
        print(f"  [throwaway] 已跑 {n_throwaway} 条 dummy SQL, "
              f"耗时 {throwaway_sec * 1000:.0f} ms")

    total_sec = prewarm_sec + throwaway_sec
    if verbose:
        print(f"  [bench_setup] PostGIS 预热完成: 总耗时 "
              f"{total_sec * 1000:.0f} ms")
        print("─" * 76)
        print()

    return {
        "prewarm_sec": prewarm_sec,
        "throwaway_sec": throwaway_sec,
        "total_sec": total_sec,
        "n_tables": n_t,
        "n_indexes": n_i,
        "n_throwaway": n_throwaway,
        "n_pages": n_pages,
        "skipped": False,
    }


# ============================================================================
# CLI: run one manual prewarm pass for inspection.
# ============================================================================

def _cli_main():
    import argparse

    from geosot_spatial_reasoning import PG_CONFIG  # noqa: WPS433
    import psycopg2  # noqa: WPS433

    parser = argparse.ArgumentParser(
        description="GeoSOT benchmark PostGIS 预热工具")
    parser.add_argument(
        "--tables", nargs="+",
        default=[
            "public.poi_new", "public.building_new", "public.road",
            "public.statutory_plan",
        ],
        help="要预热的表 (含 schema); 默认仅正文实验四表")
    parser.add_argument(
        "--no-indexes", action="store_true",
        help="跳过索引预热")
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式")
    args = parser.parse_args()

    pg_conn = psycopg2.connect(**PG_CONFIG)
    try:
        stats = prepare_pg_warm(
            pg_conn, args.tables,
            throwaway_queries=[
                ("SELECT count(*) FROM public.poi_new", None),
            ],
            include_indexes=not args.no_indexes,
            verbose=not args.quiet,
        )
        print()
        print(f"  prewarm:    {stats['prewarm_sec'] * 1000:>8.0f} ms")
        print(f"  throwaway:  {stats['throwaway_sec'] * 1000:>8.0f} ms")
        print(f"  total:      {stats['total_sec'] * 1000:>8.0f} ms")
        print(f"  tables:     {stats['n_tables']}")
        print(f"  indexes:    {stats['n_indexes']}")
        print(f"  pages:      {stats['n_pages']:,}")
    finally:
        pg_conn.close()


if __name__ == "__main__":
    _cli_main()

#!/usr/bin/env python3
# coding: utf-8
"""
Raw Neo4j traversal — distance & direction reasoning.

Re-runs the §4.4 distance/direction experiment but DISABLES every G3R
warm-cache materialization listed in Section 2.6 of the manuscript:

    * ``GeoSotGeometryReasoner._feat_cache``
        — the client-side {code -> (vertex_codes, source, geom_type)}
          dict that is normally pre-filled by ``prefetch_features``.
        — under this driver, every per-pair query issues a fresh Cypher
          ``MATCH (g:SpatialFeature) ... HAS_VERTEX_CODE ... RETURN ...`` call,
          i.e. one round-trip per code, two per pair, so 90,000 pairs
          incur 180,000 Cypher round-trips.

    * Throwaway warmup of Numba-compiled kernels is also skipped by
      default (``--no-throwaway``); we want the *measured* path to
      reflect a cold first-touch on every query.

PostGIS is left in its usual warm state (``pg_prewarm`` + index pages
resident) so the comparison in the appendix isolates the G3R side: it
shows what the manuscript numbers would look like if the G3R warm-cache
materializations were removed but PostGIS were not also degraded.

The CSV schema mirrors ``geometry_benchmark.py`` (so existing analysis
scripts can read it), but with one extra column ``mode='raw'`` for
joining with the warm CSV in ``raw_appendix_table.py``.

Usage::

    python raw_traversal/raw_geometry_benchmark.py raw_benchmark_v9 \\
        --count 200 --seed 42
    python raw_traversal/raw_geometry_benchmark.py raw_benchmark_v9 \\
        --count 200 --seed 42 --no-postgis

Environment variables honored::

    GEOSOT_SKIP_PG_PREWARM=1   skip pg_prewarm (cold PG; usually leave
                               unset so PG stays warm, isolating the
                               G3R-side cost)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time
from itertools import product
from typing import Dict, List, Optional

from _path_setup import ensure_geosot_on_path

ensure_geosot_on_path()

from geosot_geometry_reasoning import (  # noqa: E402
    GeoSotGeometryReasoner,
    SOURCE_TO_PG_TABLE,
)
from geosot_spatial_reasoning import HAS_NUMBA  # noqa: E402
from geometry_benchmark import (  # noqa: E402
    CSV_HEADERS,
    TYPES,
    _fetch_features_by_type,
    _make_pairs,
    _run_single_benchmark,
)
from bench_setup import prepare_pg_warm  # noqa: E402


class RawNeo4jReasoner(GeoSotGeometryReasoner):
    """``GeoSotGeometryReasoner`` with the feature-info cache disabled.

    Every ``get_feature_info`` call goes through the parent implementation
    but the cache dict is cleared immediately afterwards, so a follow-up
    lookup for the same ``code_id`` re-issues the Cypher query — the
    intended ``runtime graph traversal`` regime.
    """

    def get_feature_info(self, code_id):  # type: ignore[override]
        info = super().get_feature_info(code_id)
        self._feat_cache.clear()
        return info

    def prefetch_features(self, code_ids):  # type: ignore[override]
        # Disable bulk prefetch in raw mode — silently skip and return
        # an empty stats dict so downstream code does not break.
        return {
            "n_total": 0, "n_hit": 0, "n_fetch": 0,
            "n_miss": 0, "elapsed_sec": 0.0,
        }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Raw Neo4j traversal — distance & direction reasoning "
            "(no _feat_cache, no prefetch). PG stays warm by default."),
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("output_dir",
                   help="folder for the 9 per-combination CSVs")
    p.add_argument("-n", "--count", type=int, default=200,
                   help="pairs per combination (default 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed (default 42)")
    p.add_argument("--no-postgis", action="store_true",
                   help="skip PostGIS (G3R-only timing)")
    p.add_argument("--fetch-limit", type=int, default=2000,
                   help="candidate pool size per geometry type")
    p.add_argument("--no-throwaway", action="store_true", default=True,
                   help="(default ON in raw mode) skip Numba JIT warmup")
    p.add_argument("--with-throwaway", dest="no_throwaway",
                   action="store_false",
                   help="explicitly include a 5-pair JIT warmup; "
                        "results then mix cold first call + warm JIT")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    out_dir = args.output_dir
    n_per_combo = args.count
    seed = args.seed
    compare_pg = not args.no_postgis
    rng = random.Random(seed)

    os.makedirs(out_dir, exist_ok=True)

    W = 76
    print("=" * W)
    print("G3R RAW Neo4j traversal — distance & direction")
    print("  (warm-cache materialization disabled: _feat_cache OFF,")
    print("   prefetch_features OFF; PG kept warm for fair contrast)")
    print("=" * W)
    print(f"  output dir:   {os.path.abspath(out_dir)}")
    print(f"  pairs/combo:  {n_per_combo}")
    print(f"  seed:         {seed}")
    print(f"  compare PG:   {'yes' if compare_pg else 'no'}")
    print(f"  Numba JIT:    {'available' if HAS_NUMBA else 'missing'}")
    print(f"  throwaway:    {'OFF (cold first call)' if args.no_throwaway else 'ON'}")
    print("=" * W)

    reasoner = RawNeo4jReasoner(verbose=True, enable_postgis=compare_pg)

    print("\n[1/3] Sampling candidate features from Neo4j ...")
    features: Dict[str, List[Dict]] = {}
    for t in TYPES:
        features[t] = _fetch_features_by_type(
            reasoner, t, limit=args.fetch_limit, seed=seed)
        print(f"  {t:<12s}: {len(features[t]):>5d} features")

    combos = list(product(TYPES, TYPES))
    active_combos = [(a, b) for a, b in combos
                     if features[a] and features[b]]

    # PG side: keep warm (the comparison isolates G3R's loss of warm cache)
    if compare_pg and reasoner._pg_conn is not None and \
            os.getenv("GEOSOT_SKIP_PG_PREWARM") != "1":
        used_sources = {
            (f.get("source") or "").strip()
            for t in TYPES for f in features.get(t, [])}
        used_tables = sorted({
            SOURCE_TO_PG_TABLE[s] for s in used_sources
            if s in SOURCE_TO_PG_TABLE})
        prepare_pg_warm(
            reasoner._pg_conn,
            tables=used_tables,
            throwaway_queries=[
                (f"SELECT main_code FROM {tbl} LIMIT 50", None)
                for tbl in used_tables],
        )

    # Optional JIT warmup. Off by default in raw mode.
    if not args.no_throwaway and active_combos:
        print("\n[setup] WITH-throwaway: heating Numba JIT only ...")
        for ta, tb in active_combos:
            for fa, fb in _make_pairs(features[ta], features[tb], 5, rng):
                _run_single_benchmark(
                    reasoner,
                    fa.get("code_id", fa.get("main_code", "")),
                    fb.get("code_id", fb.get("main_code", "")),
                    compare_pg)
                # Force cache clear so even the warmup pairs do not
                # leak into per-query timings.
                reasoner._feat_cache.clear()

    total_pairs = len(active_combos) * n_per_combo
    print(f"\n[2/3] Running {len(active_combos)} combinations × "
          f"{n_per_combo} pairs = {total_pairs} measurements ...")

    all_rows_flat: List[Dict] = []
    t_start = time.perf_counter()

    for combo_idx, (ta, tb) in enumerate(active_combos, start=1):
        pairs = _make_pairs(features[ta], features[tb], n_per_combo, rng)
        rows: List[Dict] = []
        for fa, fb in pairs:
            row = _run_single_benchmark(
                reasoner,
                fa.get("code_id", fa.get("main_code", "")),
                fb.get("code_id", fb.get("main_code", "")),
                compare_pg)
            if row is None:
                continue
            row["mode"] = "raw"
            rows.append(row)
            # CRITICAL: clear the feature-info cache after every pair
            # so the next pair re-issues two fresh Cypher round-trips.
            reasoner._feat_cache.clear()

        all_rows_flat.extend(rows)
        fname = f"{ta}_x_{tb}.csv"
        path = os.path.join(out_dir, fname)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=CSV_HEADERS + ["mode"])
            writer.writeheader()
            for r in rows:
                r.setdefault("pair_idx", "")
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

        if rows:
            gr_med = statistics.median(
                r["gr_total_ms"] for r in rows
                if isinstance(r.get("gr_total_ms"), (int, float)))
            pg_vals = [r["pg_query_ms"] for r in rows
                       if isinstance(r.get("pg_query_ms"), (int, float))]
            pg_med = statistics.median(pg_vals) if pg_vals else float("nan")
            print(f"  [{combo_idx}/{len(active_combos)}] "
                  f"{ta:<10s} x {tb:<10s}  n={len(rows):>4d}  "
                  f"GR_med={gr_med:>8.3f} ms  PG_med={pg_med:>8.3f} ms")

    # Write a single pooled summary CSV (used by raw_appendix_table.py)
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_HEADERS + ["mode"])
        writer.writeheader()
        for r in all_rows_flat:
            r.setdefault("pair_idx", "")
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    elapsed = time.perf_counter() - t_start
    n_ok = len(all_rows_flat)
    print(f"\n[3/3] Done. {n_ok:,} valid measurements in "
          f"{elapsed:.1f} s ({n_ok / elapsed:.1f} pair/s).")
    print(f"  Pooled summary: {summary_path}")

    # ── pooled statistics for the appendix table ──
    if all_rows_flat:
        gr = [r["gr_total_ms"] for r in all_rows_flat
              if isinstance(r.get("gr_total_ms"), (int, float))]
        pg = [r["pg_query_ms"] for r in all_rows_flat
              if isinstance(r.get("pg_query_ms"), (int, float))]
        if gr and pg:
            sp = [p / g for p, g in zip(pg, gr) if g > 0]
            print("\n  Pooled medians (raw runtime traversal):")
            print(f"    G3R median:  {statistics.median(gr):>9.3f} ms")
            print(f"    PG median:   {statistics.median(pg):>9.3f} ms")
            if sp:
                print(f"    Speedup (PG/G3R) median: "
                      f"{statistics.median(sp):>6.2f}x")


if __name__ == "__main__":
    sys.exit(main() or 0)

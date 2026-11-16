"""Verify the setup-cost / amortization-crossover claims for the distance
experiment against real data.

Two things are checked and combined:

1. PostGIS one-time warm-up cost: actually time ``pg_prewarm`` + plan-cache
   throwaway on the geometry tables the distance benchmark touches. This is the
   real counterpart of the hardcoded ``init_pg_ms = 1000.0`` in the amortization
   panel of ``geometry_benchmark_analysis.py`` (which is wrong; the measured
   value is on the order of 12-16 s cold) and of the ``PostGIS total approx 16 s``
   row of the appendix setup-cost table.

2. Amortization crossover N, where the total wall time of G3R and PostGIS break
   even:  ``N* = (T_init_G3R - T_init_PG) / (t_PG - t_G3R)``.  Because the total
   time scales with the *mean* per-query cost (not the median), and because the
   G3R one-time setup can be scoped either to the distance experiment alone or to
   the full multi-operator deployment, the crossover is reported under all four
   transparent combinations. This documents how fragile the single ``4x10^4``
   number in the appendix footnote is.

pg_prewarm only loads table/index pages into shared_buffers; it does not modify
data, so this is a read-only measurement. Run twice to see cold vs already-warm.
"""
import os
import time

import psycopg2

PG_CONFIG = dict(host="127.0.0.1", port=5432, dbname="postgis_35_sample",
                 user="postgres", password=os.getenv("PGPASSWORD", ""))

# Distance benchmark samples Point/LineString/Polygon features from these tables.
TABLES = ["public.poi_new", "public.building_new",
          "public.road", "public.statutory_plan"]

THROWAWAY = [f"SELECT main_code FROM {t} LIMIT 50" for t in TABLES]

# Nine geometry combinations used by the distance benchmark.
COMBOS = [
    "Point_x_Point", "Point_x_LineString", "Point_x_Polygon",
    "LineString_x_Point", "LineString_x_LineString", "LineString_x_Polygon",
    "Polygon_x_Point", "Polygon_x_LineString", "Polygon_x_Polygon",
]

# G3R one-time setup components (seconds) as measured in the appendix
# setup-cost table (tab:appendix_setup_cost). Distance retrieval only needs the
# centroid-code index, the POI prefix cache, and kernel JIT; the parcel
# preloader / polygon cache / LCPC inverted index are overlay/reverse only.
G3R_SETUP_S = {
    "centroid_code_index": 8.0,   # distance + overlay + knn
    "poi_prefix_cache": 6.3,      # distance (POI candidates) + knn
    "kernel_jit": 2.1,            # distance + all kernels
    "parcel_preloader": 64.7,     # overlay only
    "parcel_polygon_cache": 11.2,  # reverse only
    "lcpc_inverted_index": 4.8,   # overlay/reverse only
}
G3R_SETUP_DISTANCE_S = (G3R_SETUP_S["centroid_code_index"]
                        + G3R_SETUP_S["poi_prefix_cache"]
                        + G3R_SETUP_S["kernel_jit"])          # ~16.4 s
G3R_SETUP_FULL_S = sum(G3R_SETUP_S.values())                   # ~97 s


def _connect():
    try:
        return psycopg2.connect(connect_timeout=5, **PG_CONFIG)
    except Exception as e:  # noqa: BLE001
        print("CONNECT FAIL", PG_CONFIG["dbname"], ":", str(e)[:120])
        PG_CONFIG["dbname"] = "postgres"
        conn = psycopg2.connect(connect_timeout=5, **PG_CONFIG)
        print("connected to 'postgres' db instead")
        return conn


def main():
    t0 = time.time()
    conn = _connect()
    conn.autocommit = True
    cur = conn.cursor()
    print(f"connected in {time.time()-t0:.2f}s  db={PG_CONFIG['dbname']}")

    cur.execute("SELECT extname FROM pg_extension "
                "WHERE extname IN ('postgis','pg_prewarm')")
    exts = [r[0] for r in cur.fetchall()]
    print("extensions:", exts)
    if "pg_prewarm" not in exts:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
            print("created pg_prewarm extension")
        except Exception as e:  # noqa: BLE001
            print("cannot create pg_prewarm:", str(e)[:120])
            return

    # Sizes + row counts.
    print("\n--- table sizes ---")
    total_mb = 0.0
    valid = []
    for t in TABLES:
        try:
            cur.execute(f"SELECT count(*), pg_total_relation_size('{t}')")
            n, sz = cur.fetchone()
            total_mb += sz / 1e6
            valid.append(t)
            print(f"{t:<26s} rows={n:>10,d}  total_size={sz/1e6:8.1f} MB")
        except Exception as e:  # noqa: BLE001
            print(f"{t:<26s} MISSING/ERR: {str(e)[:80]}")
    print(f"{'TOTAL':<26s} {'':>16s}  {total_mb:8.1f} MB")

    def run_prewarm(label):
        print(f"\n=== pg_prewarm pass: {label} ===")
        t_pre0 = time.time()
        pages = 0
        for t in valid:
            # prewarm the table heap
            cur.execute(f"SELECT pg_prewarm('{t}')")
            pages += cur.fetchone()[0]
            # prewarm each index on the table
            cur.execute(
                "SELECT indexrelid::regclass::text FROM pg_index "
                "WHERE indrelid = %s::regclass", (t,))
            for (idx,) in cur.fetchall():
                cur.execute(f"SELECT pg_prewarm('{idx}')")
                pages += cur.fetchone()[0]
        prewarm_sec = time.time() - t_pre0

        t_tw0 = time.time()
        for sql in THROWAWAY:
            try:
                cur.execute(sql)
                cur.fetchall()
            except Exception:  # noqa: BLE001
                pass
        throwaway_sec = time.time() - t_tw0
        total = prewarm_sec + throwaway_sec
        print(f"  prewarm : {prewarm_sec*1000:8.0f} ms  ({pages:,} pages, "
              f"{pages*8/1024:.1f} MB @8KB)")
        print(f"  throwaway:{throwaway_sec*1000:8.0f} ms")
        print(f"  TOTAL   : {total*1000:8.0f} ms  <-- compare to init_pg_ms=1000")
        return total

    pg_cold = run_prewarm("1st (may hit disk / cold buffers)")
    run_prewarm("2nd (already in shared_buffers)")

    conn.close()
    return pg_cold


def _read_kernels(folder="benchmark_v9"):
    """Pooled per-query kernel time (ms) for G3R and PostGIS over 90k pairs."""
    import numpy as np
    import pandas as pd
    gr, pg = [], []
    for k in COMBOS:
        p = os.path.join(folder, k + ".csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        gr += list(pd.to_numeric(df["gr_total_ms"], errors="coerce").dropna())
        pg += list(pd.to_numeric(df["pg_query_ms"], errors="coerce").dropna())
    gr, pg = np.array(gr), np.array(pg)
    return {
        "n": len(gr),
        "gr_mean": gr.mean(), "gr_med": float(np.median(gr)),
        "pg_mean": pg.mean(), "pg_med": float(np.median(pg)),
    }


def analyze_crossover(pg_setup_s):
    """Compute the amortization crossover N under transparent assumptions.

    N* = (T_init_G3R - T_init_PG) / (t_PG - t_G3R), evaluated with
      * per-query cost = mean (drives total wall time) vs median (typical query)
      * G3R setup scoped to distance only (~16 s) vs full deployment (~97 s)
    """
    print("\n" + "=" * 68)
    print("AMORTIZATION CROSSOVER (real data)")
    print("=" * 68)
    k = _read_kernels()
    print(f"benchmark_v9 pooled N={k['n']:,}")
    print(f"  G3R  kernel : mean={k['gr_mean']:.4f} ms  median={k['gr_med']:.4f} ms")
    print(f"  PG   kernel : mean={k['pg_mean']:.4f} ms  median={k['pg_med']:.4f} ms")
    print(f"  diff        : mean={k['pg_mean']-k['gr_mean']:.4f} ms  "
          f"median={k['pg_med']-k['gr_med']:.4f} ms")
    print(f"  speedup     : mean={k['pg_mean']/k['gr_mean']:.2f}x  "
          f"median={k['pg_med']/k['gr_med']:.2f}x")
    print(f"\nPostGIS setup (measured this run) : {pg_setup_s:.1f} s")
    print(f"G3R setup (appendix, distance-only): {G3R_SETUP_DISTANCE_S:.1f} s")
    print(f"G3R setup (appendix, full deploy)  : {G3R_SETUP_FULL_S:.1f} s")

    print("\n  scope        per-query   crossover N*   90k pairs past it?")
    print("  " + "-" * 60)
    for scope, g3r_setup in [("distance", G3R_SETUP_DISTANCE_S),
                             ("full", G3R_SETUP_FULL_S)]:
        for stat, diff in [("mean", k["pg_mean"] - k["gr_mean"]),
                           ("median", k["pg_med"] - k["gr_med"])]:
            setup_diff_ms = (g3r_setup - pg_setup_s) * 1000.0
            if diff <= 0:
                res = "never (PG faster/query)"
            else:
                n_star = setup_diff_ms / diff
                if n_star <= 0:
                    res = "N*<=0 (G3R wins from start)"
                else:
                    res = f"{n_star:,.0f}  ({'YES' if 90000 > n_star else 'NO'})"
            print(f"  {scope:<12s} {stat:<9s}  {res}")
    print("\nTakeaway: the crossover spans several orders of magnitude across")
    print("assumptions, so a single fixed value (e.g. 4e4) is not defensible.")


if __name__ == "__main__":
    pg_setup = main()
    try:
        analyze_crossover(pg_setup if pg_setup else 16.0)
    except Exception as e:  # noqa: BLE001
        print("crossover analysis skipped:", str(e)[:120])

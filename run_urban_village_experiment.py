#!/usr/bin/env python3
# coding: utf-8
"""
Exploratory urban-village identification: paired G3R and PostGIS pipelines sharing the same
normalization and equal-weight score.

A: aggregate semantic, morphological, origin-diversity, and population attributes. B: find
nearest other buildings through GeoSOT KNN or the PG spatial baseline. C: normalize and
combine the same five signals. Compare top-fraction building sets/Jaccard, main_code-aligned
semantic attributes, spacing error quantiles, composite score differences, and
execution/transfer costs. The current analysis baseline uses geography distances; projected
coordinates are for clustering/plotting intermediates.

--limit aligns both implementations to the G3R building sample; full runs execute independently.

Usage:
    python run_urban_village_experiment.py
    python run_urban_village_experiment.py --limit 6000
    python run_urban_village_experiment.py --top 0.15
    python run_urban_village_experiment.py --prewarm --workers 8 --repeat 3
"""
from __future__ import annotations
import csv, json, os, sys, time
from typing import Dict, List, Optional

import numpy as np
from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mismatch_experiment import (
    NEO4J_URI, NEO4J_USER, NEO4J_PWD, NEO4J_DB, PG_MAIN, PG_MIG, SOURCE_SRID,
    StepLog, _print_substeps, _prewarm, _bench_summary)
from urban_village_g3r import (
    stage_a, stage_b, stage_c, score_threshold, write_scores,
    SIGNAL_KEYS, SIGNAL_CN, Heartbeat, Phases)

_t = time.perf_counter
W = 78
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════════
# G3R in-graph pipeline (reuse 3 stages)
# ══════════════════════════════════════════════════════════════════════
def run_g3r(limit: int, outdir: str, ph: Phases, steps: StepLog = None,
            workers: int = 1):
    steps = steps if steps is not None else StepLog()
    metrics = {"engine": "g3r", "limit": limit, "stages": {},
               "cross_system_transfers": 0}
    print("=" * W)
    print("[G3R] 图内统一执行流水线 (Neo4j + GeoSOT, 语义/形态/人口同图)")
    print("=" * W)
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    sess = drv.session(database=NEO4J_DB)
    t_all = _t()
    ph.step("G3R·A 图内聚合 语义+形态+人口")
    t0 = _t(); rows = stage_a(sess, limit, steps=steps)
    metrics["stages"]["A_aggregate_ingraph"] = round(_t() - t0, 2)
    ph.step("G3R·B 空间桥·握手楼 KNN")
    t0 = _t(); stage_b(sess, rows, steps=steps, workers=workers)
    metrics["stages"]["B_handshake_knn_ingraph"] = round(_t() - t0, 2)
    ph.step("G3R·C 归一化+复合评分")
    t0 = _t(); stage_c(rows)
    steps.add("g3r", "C", "归一化+等权复合评分", _t() - t0, rows=len(rows))
    metrics["stages"]["C_score"] = round(_t() - t0, 2)
    metrics["stages"]["total_end_to_end"] = round(_t() - t_all, 2)
    metrics["result"] = {"resident_buildings": len(rows)}
    write_scores(os.path.join(outdir, "g3r_uv_building_score.csv"), rows)
    with open(os.path.join(outdir, "g3r_uv_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    sess.close(); drv.close()
    print(f"[G3R] 端到端 {metrics['stages']['total_end_to_end']:.1f}s "
          f"跨系统搬运 0 次\n")
    return metrics, rows


# ══════════════════════════════════════════════════════════════════════
# PostGIS external pipeline
# ══════════════════════════════════════════════════════════════════════
# Resolve resident origin strings through a normalized administrative dictionary.
# county_origin_dict(name -> province/city) mirrors G3R's CountyOrigin -> Province/AdminCity traversal.
# Province/city resolution matched STARTS WITH in historical checks, leaving per-building scores unchanged.
_AGG_PG = """
    SELECT h.buildingid, count(*),
      count(*) FILTER (WHERE d.province IS NULL
                       OR NOT (d.province = '广东省' AND d.city = '深圳市')),
      count(*) FILTER (WHERE d.province IS NULL OR d.province <> '广东省'),
      count(DISTINCT r.county_origin),
      max(b.layers), max(b.buildtype), max(b.buildingarea), max(b.address)
    FROM oracle_src.resident r
    JOIN oracle_src.house h ON h.id = r.housecode
    JOIN oracle_src.building b ON b.id = h.buildingid
    LEFT JOIN oracle_src.county_origin_dict d ON d.name = r.county_origin
    {where}
    GROUP BY h.buildingid
"""


def run_pg(limit: int, outdir: str, restrict_codes: Optional[List[str]],
           ph: Phases, steps: StepLog = None):
    import psycopg2
    from psycopg2.extras import execute_values

    steps = steps if steps is not None else StepLog()
    metrics = {"engine": "postgis", "limit": limit, "stages": {},
               "cross_system_transfers": []}
    xfer = metrics["cross_system_transfers"]
    print("=" * W)
    print("[PG] PostGIS 外置基线流水线 (几何库 + oracle_mig 跨库居民/建筑)")
    print("=" * W)
    t_all = _t()
    m = psycopg2.connect(**PG_MAIN)
    g = psycopg2.connect(**PG_MIG)

    # Align main_code → BLDG_NO
    restrict_bno = None
    if restrict_codes:
        with m.cursor() as c:
            c.execute('SELECT main_code, "BLDG_NO" FROM public.building_new '
                      'WHERE main_code = ANY(%s)', (list(restrict_codes),))
            restrict_bno = [str(b) for _, b in c.fetchall()]
        print(f"  限定建筑集: {len(restrict_codes):,} main_code → "
              f"{len(restrict_bno):,} BLDG_NO (与 G3R 同批)")

    # A' cross-DB aggregate semantics + form
    ph.step("PG·A 跨库聚合 外省/籍贯/人口+形态")
    print("[A] 跨库聚合 外省/籍贯/人口 + 建筑形态 (oracle_mig → app) ...")
    t0 = _t()
    with g.cursor() as c, Heartbeat("跨库聚合(扫居民表)"):
        if restrict_bno is not None:
            c.execute(_AGG_PG.format(where="WHERE h.buildingid = ANY(%s)"),
                      (restrict_bno,))
        else:
            c.execute(_AGG_PG.format(where=""))
        demo = {}
        for (bid, tot, fcnt, ocnt, no, ly, bt, ba, addr) in c.fetchall():
            if bid is None:
                continue
            demo[str(bid)] = (int(tot), int(fcnt), int(ocnt), int(no),
                              ly, bt, ba, addr)
    metrics["stages"]["A_aggregate_crossdb"] = round(_t() - t0, 2)
    steps.add("postgis", "A", "跨库聚合(join county_origin_dict 解析省/市)",
              _t() - t0, rows=len(demo),
              note="oracle_mig→app 跨库搬运1; 籍贯→行政区划字典 join")
    xfer.append({"dir": "oracle_mig→app", "what": "居民语义+建筑形态聚合",
                 "rows": len(demo)})
    print(f"    有居民建筑 {len(demo):,}  {_t()-t0:.1f}s (跨库搬运1)")

    # Map BLDG_NO to main_code and retain buildings with geometry, matching G3R's HAS_GEOSOT requirement.
    with m.cursor() as c:
        c.execute('SELECT "BLDG_NO", main_code FROM public.building_new '
                  'WHERE "BLDG_NO" = ANY(%s) AND main_code IS NOT NULL '
                  'AND geometry IS NOT NULL', (list(demo.keys()),))
        bno2cc = {str(b): str(mc).strip().upper() for b, mc in c.fetchall()}
    rows = []
    for bid, (tot, fcnt, ocnt, no, ly, bt, ba, addr) in demo.items():
        cc = bno2cc.get(bid)
        if not cc:
            continue
        try:
            layers = int(ly) if ly is not None else None
        except (ValueError, TypeError):
            layers = None
        try:
            bldarea = float(ba) if ba is not None else None
        except (ValueError, TypeError):
            bldarea = None
        rows.append({"cc": cc, "res_total": tot, "foreign_cnt": fcnt,
                     "outprov_cnt": ocnt, "n_origin": no, "layers": layers,
                     "buildtype": str(bt), "usage": "", "bldarea": bldarea,
                     "address": (addr or "").strip()})

    # B' handshake nearest-building distance
    ph.step("PG·B 握手楼 KNN (GIST)")
    print("[B] PostGIS 握手楼 KNN: 最近邻建筑距离 "
          "(EPSG:4490 geography + GIST) ...")
    t0 = _t()
    with m.cursor() as c, Heartbeat("PostGIS 握手楼自连接"):
        t_build = _t()
        c.execute("DROP TABLE IF EXISTS uv_pts")
        c.execute("CREATE TEMP TABLE uv_pts(cc text PRIMARY KEY, c geography)")
        execute_values(c, "INSERT INTO uv_pts(cc) VALUES %s "
                          "ON CONFLICT DO NOTHING",
                       [(r["cc"],) for r in rows], page_size=10000)
        c.execute("""UPDATE uv_pts t SET c = ST_Centroid(b.geometry)::geography
                      FROM public.building_new b WHERE b.main_code = t.cc""")
        c.execute("DELETE FROM uv_pts WHERE c IS NULL")
        c.execute("CREATE INDEX ix_uv_pts ON uv_pts USING GIST(c)")
        c.execute("ANALYZE uv_pts")
        steps.add("postgis", "B",
                  "建筑质心 geography(EPSG:4490)+GIST(app→PG 搬运)",
                  _t() - t_build, rows=len(rows))
        t_join = _t()
        # <-> on geography is spherical; over-fetch 8, then ellipsoidal
        # ST_Distance picks the true nearest (K=1 → 8× over-fetch).
        c.execute("""SELECT t.cc, k.d
          FROM uv_pts t
          CROSS JOIN LATERAL (
            SELECT c.d
              FROM (
                SELECT p.cc,
                       ST_Distance(t.c, p.c) AS d
                  FROM uv_pts p
                 WHERE p.cc <> t.cc
                 ORDER BY t.c <-> p.c
                 LIMIT 8
              ) c
             ORDER BY c.d
             LIMIT 1
          ) k""")
        nn = {str(cc).strip().upper(): float(d)
              for cc, d in c.fetchall() if d is not None}
        steps.add("postgis", "B",
                  "握手楼最近邻(球面KNN×8→椭球精排)",
                  _t() - t_join, rows=len(nn))
    m.commit()
    rows = [r for r in rows if r["cc"] in nn]     # Retain only buildings with geometry and a neighbor.
    for r in rows:
        r["nn_dist"] = nn[r["cc"]]
    xfer.append({"dir": "app→postgis", "what": "目标建筑集(握手KNN)",
                 "rows": len(rows)})
    xfer.append({"dir": "postgis→app", "what": "握手楼最近距离", "rows": len(rows)})
    metrics["stages"]["B_handshake_knn"] = round(_t() - t0, 2)
    print(f"    握手 {len(rows):,}  {_t()-t0:.1f}s (往返2)")

    # C' same stage_c as G3R (fair)
    ph.step("PG·C 归一化+复合评分")
    t0 = _t(); stage_c(rows)
    metrics["stages"]["C_score"] = round(_t() - t0, 2)
    steps.add("postgis", "C", "归一化+等权复合评分", _t() - t0, rows=len(rows))

    metrics["stages"]["total_end_to_end"] = round(_t() - t_all, 2)
    metrics["result"] = {"resident_buildings": len(rows)}
    write_scores(os.path.join(outdir, "pg_uv_building_score.csv"), rows)
    with open(os.path.join(outdir, "pg_uv_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    m.close(); g.close()
    print(f"[PG] 端到端 {metrics['stages']['total_end_to_end']:.1f}s "
          f"跨系统搬运 {len(xfer)} 次\n")
    return metrics, rows


# ══════════════════════════════════════════════════════════════════════
# Compare + diagnose
# ══════════════════════════════════════════════════════════════════════
def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def compare(mg, rg, mp, rp, top_frac, ph):
    ph.step("对比 & 诊断")
    print("=" * W)
    print("  对比 & 诊断   G3R (图内)  vs  PostGIS (外置)")
    print("=" * W)
    gd = {r["cc"]: r for r in rg}
    pd = {r["cc"]: r for r in rp}
    common = sorted(set(gd) & set(pd))

    # [1] High-score building-set consistency
    thg = score_threshold(rg, top_frac)
    thp = score_threshold(rp, top_frac)
    hg = {r["cc"] for r in rg if r["uv_score"] >= thg}
    hp = {r["cc"] for r in rp if r["uv_score"] >= thp}
    inter = len(hg & hp); uni = len(hg | hp)
    jac = inter / uni if uni else 0.0
    print(f"\n[1] 高分建筑集合 (top {top_frac:.0%})")
    print(f"  G3R {len(hg):,}  PG {len(hp):,}  交集 {inter:,}  Jaccard {jac:.3f}")

    if not common:
        print("\n[!] 无可对齐建筑 (main_code 交集为空)"); return
    print(f"\n[2] 逐建筑语义信号一致性 (对齐 main_code, n={len(common):,})")
    print(f"  {'信号':<12}{'完全一致率':>10}{'中位|Δ|':>10}{'均值|Δ|':>10}")
    for key, cn in [("res_total", "居民总数"), ("outprov_cnt", "外省人数"),
                    ("n_origin", "籍贯种数")]:
        dg = np.array([gd[c][key] for c in common], dtype=float)
        dp = np.array([pd[c][key] for c in common], dtype=float)
        ad = np.abs(dg - dp)
        eq = float(np.mean(ad == 0) * 100)
        print(f"  {cn:<12}{eq:>9.1f}%{np.median(ad):>10.1f}{np.mean(ad):>10.1f}")

    # [3] Handshake distance fidelity
    hg_d = np.array([gd[c].get("nn_dist", 0.0) for c in common])
    hp_d = np.array([pd[c].get("nn_dist", 0.0) for c in common])
    ad = np.abs(hg_d - hp_d)
    corr = float(np.corrcoef(hg_d, hp_d)[0, 1]) if len(common) > 2 else 0.0
    print(f"\n[3] 握手楼最近邻距离保真 (G3R 网格KNN vs PG 投影ST_Distance)")
    print(f"  中位|Δ| {np.median(ad):.2f}m  均值|Δ| {np.mean(ad):.2f}m  "
          f"P95 {_pct(ad,95):.2f}m  P99 {_pct(ad,99):.2f}m  相关 {corr:.4f}")
    print(f"  G3R 中位 {np.median(hg_d):.1f}m   PG 中位 {np.median(hp_d):.1f}m")

    # [4] Composite-score consistency
    sg = np.array([gd[c]["uv_score"] for c in common])
    sp = np.array([pd[c]["uv_score"] for c in common])
    ads = np.abs(sg - sp)
    scorr = float(np.corrcoef(sg, sp)[0, 1]) if len(common) > 2 else 0.0
    print(f"\n[4] 城中村复合评分一致性 (uv_score)")
    print(f"  中位|Δ| {np.median(ads):.4f}  均值|Δ| {np.mean(ads):.4f}  "
          f"P95 {_pct(ads,95):.4f}  相关 {scorr:.4f}")

    # architecture/cost (timing + cross-system xfers)
    print(f"\n[5] 架构 / 成本")
    gs, ps = mg["stages"], mp["stages"]
    print(f"  {'阶段':<22}{'G3R(s)':>10}{'PG(s)':>10}")
    print("  " + "-" * 42)
    pairs = [("A 聚合(语义+形态+人口)", "A_aggregate_ingraph",
              "A_aggregate_crossdb"),
             ("B 握手楼 KNN", "B_handshake_knn_ingraph", "B_handshake_knn"),
             ("C 评分", "C_score", "C_score"),
             ("端到端", "total_end_to_end", "total_end_to_end")]
    for label, gk, pk in pairs:
        print(f"  {label:<22}{gs.get(gk,0):>10.1f}{ps.get(pk,0):>10.1f}")
    ntx = len(mp["cross_system_transfers"])
    print(f"  {'跨系统搬运':<22}{'0':>10}{ntx:>10}")
    for x in mp["cross_system_transfers"]:
        print(f"      PG [{x['dir']:<14}] {x['what']:<22} rows={x['rows']:,}")
    print("\n[完成]")


def main():
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print('Usage: python run_urban_village_experiment.py [options]\n\nExploratory urban-village candidate comparison (G3R vs PostGIS).\nRequires prepared databases. This help command does not connect to them.\n\n  -h, --help          Show this message and exit\n  --limit N           Align both methods to a sample of N buildings; 0=full\n  --top FRACTION      High-score fraction (default 0.15)\n  --outdir DIR        CSV/metric output directory (default csv)\n  --workers N         Worker processes; 1=serial\n  --repeat N          Number of measured repeats (default 1)\n  --prewarm           Warm relevant data before measured runs\n')
        return
    limit = 0; outdir = "csv"; top_frac = 0.15
    only = None
    workers = min(os.cpu_count() or 1, 16)      # Building-spacing KNN workers; 1 selects serial execution.
    prewarm = False                             # Symmetric warmup to reduce the first-run cold-cache spike.
    repeat = 1                                  # For repeats above 1, report stage minimum/median/maximum.
    a = sys.argv[1:]; i = 0
    while i < len(a):
        if a[i] == "--limit" and i + 1 < len(a):
            limit = int(a[i+1]); i += 2
        elif a[i] == "--top" and i + 1 < len(a):
            top_frac = float(a[i+1]); i += 2
        elif a[i] == "--outdir" and i + 1 < len(a):
            outdir = a[i+1]; i += 2
        elif a[i] == "--only" and i + 1 < len(a):
            only = a[i+1]; i += 2
        elif a[i] == "--workers" and i + 1 < len(a):
            workers = max(int(a[i+1]), 1); i += 2
        elif a[i] == "--repeat" and i + 1 < len(a):
            repeat = max(int(a[i+1]), 1); i += 2
        elif a[i] == "--prewarm":
            prewarm = True; i += 1
        else:
            i += 1
    os.makedirs(outdir, exist_ok=True)
    print("#" * W)
    print('#  Exploratory urban-village workflow (G3R vs PostGIS)')
    print(f"#  limit={limit or '全量'}   top={top_frac:.0%}   outdir={outdir}"
          f"   workers={workers}")
    print("#" * W)

    if prewarm:
        _prewarm()

    g_runs, p_runs = [], []                     # Per-run stage times for --repeat summaries.
    mg = rg = mp = rp = steps = None
    for it in range(1, repeat + 1):
        if repeat > 1:
            print("\n" + "#" * W)
            print(f"#  REPEAT {it}/{repeat}")
            print("#" * W)
        ph = Phases(7)   # G3R stages A/B/C, PG stages A/B/C, and comparison.
        steps = StepLog()                # Detailed timings include origin administrative-hierarchy resolution.
        mg, rg = run_g3r(limit, outdir, ph, steps=steps, workers=workers)
        g_runs.append(mg["stages"])
        # smoke: align PG to G3R; full: independent
        restrict = [r["cc"] for r in rg] if limit > 0 else None
        mp, rp = run_pg(limit, outdir, restrict, ph, steps=steps)
        p_runs.append(mp["stages"])
        if steps.rows:
            sp = os.path.join(outdir, "uv_step_timings.csv")
            steps.write(sp)
            print(f"\n[细粒度耗时] {sp}  ({len(steps.rows)} 子步)")

    compare(mg, rg, mp, rp, top_frac, ph)        # Use the final run's results.
    if steps and steps.rows:
        _print_substeps(os.path.join(outdir, "uv_step_timings.csv"))

    if repeat > 1:
        print("\n" + "=" * W)
        print(f"  多次基准汇总 (N={repeat}, 预热{'开' if prewarm else '关'}, "
              f"workers={workers})")
        print("  取数建议: 每阶段用「中位」列; 极差% 大者受本机负载影响, 需减背景干扰")
        print("=" * W)
        if g_runs:
            _bench_summary(g_runs, "G3R")
        if p_runs:
            _bench_summary(p_runs, "PostGIS")


if __name__ == "__main__":
    main()

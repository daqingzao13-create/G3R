#!/usr/bin/env python3
# coding: utf-8
"""
深圳城中村探查 —— 单脚本一键实验 (G3R 图内统一执行  vs  PostGIS 外置基线)。
Shenzhen urban-village exploration — one-script experiment (G3R in-graph vs PostGIS baseline).

与 run_mismatch_experiment.py 同结构、同严谨度; 复用 urban_village_g3r.py 的
评分/聚类算子, 保证两侧走**同一套归一化 + 等权评分**(公平)。
Same structure/rigor as run_mismatch_experiment.py; reuses urban_village_g3r scoring
so both sides share the **same normalize + equal-weight score** (fair).

三阶段镜像 (评分口径完全一致) / Three-stage mirror (identical scoring):
  A 聚合语义+形态+人口   G3R 图内语义链(跨库0)      | PG oracle_mig 跨库聚合(搬运1)
     Aggregate semantics+form+pop | G3R in-graph (0 xfers) | PG cross-DB (xfer 1)
  B 握手楼最近邻建筑距离  G3R 空间桥·KNN(GeoSOT 网格) | PG 投影米制 GIST <-> 自连接
     Handshake nearest-building dist | G3R spatial-bridge KNN | PG geography GIST <->
  C 归一化+等权复合评分   两侧同一 stage_c (公平)
     Normalize + equal-weight score | same stage_c both sides (fair)

对比 & 诊断 / Compare & diagnose:
  [1] 高分建筑集合一致性 (top-N%, Jaccard) / high-score set consistency
  [2] 逐建筑语义信号一致性 / per-building semantic-signal consistency
  [3] 握手楼距离保真 / handshake distance fidelity
  [4] 逐建筑复合评分一致性 / per-building composite-score consistency
  [5] 架构/成本 / architecture/cost (timing + cross-system xfers)

--limit 语义: 冒烟时两侧强制同一批建筑(G3R 先跑, PG 按其 main_code 对齐) → 可比;
全量(不加 --limit)时两侧各自独立跑 → 保留"两套独立实现互证"。
--limit: smoke forces same building batch; full run keeps independent implementations.

用法 / Usage:
  python run_urban_village_experiment.py                # 全量 / full
  python run_urban_village_experiment.py --limit 6000   # 冒烟, 两侧同批 / smoke, same batch
  python run_urban_village_experiment.py --top 0.15     # 高分集取 top 15% / top 15% set
  python run_urban_village_experiment.py --prewarm --workers 8 --repeat 3
      # 正式基准: 对称预热 + 握手楼 KNN 并行 + 3 次取中位 (消首跑冷缓存尖峰)
      # Formal bench: symmetric prewarm + parallel handshake KNN + median of 3
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
# G3R 图内流水线 (复用 urban_village_g3r 三阶段) / G3R in-graph pipeline (reuse 3 stages)
# ══════════════════════════════════════════════════════════════════════
def run_g3r(limit: int, outdir: str, ph: Phases, steps: StepLog = None,
            workers: int = 1):
    steps = steps if steps is not None else StepLog()
    metrics = {"engine": "g3r", "limit": limit, "stages": {},
               "cross_system_transfers": 0}
    print("=" * W)
    print("[G3R] 图内统一执行流水线 (Neo4j + GeoSOT, 语义/形态/人口同图) / In-graph unified pipeline")
    print("=" * W)
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    sess = drv.session(database=NEO4J_DB)
    t_all = _t()
    ph.step("G3R·A 图内聚合 语义+形态+人口 / in-graph aggregate")
    t0 = _t(); rows = stage_a(sess, limit, steps=steps)
    metrics["stages"]["A_aggregate_ingraph"] = round(_t() - t0, 2)
    ph.step("G3R·B 空间桥·握手楼 KNN / handshake KNN")
    t0 = _t(); stage_b(sess, rows, steps=steps, workers=workers)
    metrics["stages"]["B_handshake_knn_ingraph"] = round(_t() - t0, 2)
    ph.step("G3R·C 归一化+复合评分 / normalize+score")
    t0 = _t(); stage_c(rows)
    steps.add("g3r", "C", "归一化+等权复合评分 / normalize + equal-weight score", _t() - t0, rows=len(rows))
    metrics["stages"]["C_score"] = round(_t() - t0, 2)
    metrics["stages"]["total_end_to_end"] = round(_t() - t_all, 2)
    metrics["result"] = {"resident_buildings": len(rows)}
    write_scores(os.path.join(outdir, "g3r_uv_building_score.csv"), rows)
    with open(os.path.join(outdir, "g3r_uv_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    sess.close(); drv.close()
    print(f"[G3R] 端到端 / end-to-end {metrics['stages']['total_end_to_end']:.1f}s "
          f"跨系统搬运 / cross-system xfers 0\n")
    return metrics, rows


# ══════════════════════════════════════════════════════════════════════
# PostGIS 外置流水线 (键: building_new.main_code, 与 G3R 对齐) / PostGIS external pipeline
# ══════════════════════════════════════════════════════════════════════
# 籍贯: 假设 resident 只有籍贯全串(叶子), 省/市须 join 规范化行政区划字典
# Origin: assume resident has full origin string (leaf) only; province/city need dict join
# county_origin_dict(name→province/city) 解析 → 与 G3R 走 CountyOrigin→Province/City 对称。
# Resolve via county_origin_dict(name→province/city) → symmetric to G3R CountyOrigin→Province/City.
# (province/city 与 STARTS WITH 前缀逐位一致 → 城中村评分逐建筑不变)
# (province/city bit-identical to STARTS WITH prefix → per-building UV scores unchanged)
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
    print("[PG] PostGIS 外置基线流水线 (几何库 + oracle_mig 跨库居民/建筑) / external baseline (geometry + cross-DB)")
    print("=" * W)
    t_all = _t()
    m = psycopg2.connect(**PG_MAIN)
    g = psycopg2.connect(**PG_MIG)

    # 对齐: main_code → BLDG_NO (= building.id = house.buildingid) / Align main_code → BLDG_NO
    restrict_bno = None
    if restrict_codes:
        with m.cursor() as c:
            c.execute('SELECT main_code, "BLDG_NO" FROM public.building_new '
                      'WHERE main_code = ANY(%s)', (list(restrict_codes),))
            restrict_bno = [str(b) for _, b in c.fetchall()]
        print(f"  限定建筑集 / restricted set: {len(restrict_codes):,} main_code → "
              f"{len(restrict_bno):,} BLDG_NO (与 G3R 同批 / same batch as G3R)")

    # A' 跨库聚合 居民语义 + 建筑形态 / A' cross-DB aggregate semantics + form
    ph.step("PG·A 跨库聚合 外省/籍贯/人口+形态 / cross-DB aggregate")
    print("[A] 跨库聚合 外省/籍贯/人口 + 建筑形态 (oracle_mig → app) ... / Cross-DB aggregate ...")
    t0 = _t()
    with g.cursor() as c, Heartbeat("跨库聚合(扫居民表) / cross-DB aggregate (scan residents)"):
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
    steps.add("postgis", "A", "跨库聚合(join county_origin_dict 解析省/市) / cross-DB aggregate",
              _t() - t0, rows=len(demo),
              note="oracle_mig→app 跨库搬运1; 籍贯→行政区划字典 join / xfer 1; origin dict join")
    xfer.append({"dir": "oracle_mig→app", "what": "居民语义+建筑形态聚合 / resident semantics+form",
                 "rows": len(demo)})
    print(f"    有居民建筑 / resident bldgs {len(demo):,}  {_t()-t0:.1f}s (跨库搬运1 / xfer 1)")

    # BLDG_NO → main_code, 组装行 (仅保留有几何的建筑, 与 G3R 有 HAS_GEOSOT 对齐)
# Assemble rows (keep buildings with geometry; align with G3R HAS_GEOSOT)
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

    # B' 握手楼最近邻建筑距离 (EPSG:4490 → geography + GIST <->) / B' handshake nearest-building distance
    ph.step("PG·B 握手楼 KNN (GIST) / handshake KNN")
    print("[B] PostGIS 握手楼 KNN: 最近邻建筑距离 / nearest-building distance "
          "(EPSG:4490 geography + GIST) ...")
    t0 = _t()
    with m.cursor() as c, Heartbeat("PostGIS 握手楼自连接 / handshake self-join"):
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
                  "建筑质心 geography(EPSG:4490)+GIST(app→PG 搬运) / building centroids+GIST",
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
                  "握手楼最近邻(球面KNN×8→椭球精排) / handshake nn (spherical×8 → ellipsoid refine)",
                  _t() - t_join, rows=len(nn))
    m.commit()
    rows = [r for r in rows if r["cc"] in nn]     # 仅保留有几何/邻居的 / keep those with geom/neighbor
    for r in rows:
        r["nn_dist"] = nn[r["cc"]]
    xfer.append({"dir": "app→postgis", "what": "目标建筑集(握手KNN) / target buildings (handshake KNN)",
                 "rows": len(rows)})
    xfer.append({"dir": "postgis→app", "what": "握手楼最近距离 / handshake nearest distance", "rows": len(rows)})
    metrics["stages"]["B_handshake_knn"] = round(_t() - t0, 2)
    print(f"    握手 / handshake {len(rows):,}  {_t()-t0:.1f}s (往返2 / round-trips 2)")

    # C' 归一化 + 等权评分 (与 G3R 同一 stage_c → 公平) / C' same stage_c as G3R (fair)
    ph.step("PG·C 归一化+复合评分 / normalize+score")
    t0 = _t(); stage_c(rows)
    metrics["stages"]["C_score"] = round(_t() - t0, 2)
    steps.add("postgis", "C", "归一化+等权复合评分 / normalize + equal-weight score", _t() - t0, rows=len(rows))

    metrics["stages"]["total_end_to_end"] = round(_t() - t_all, 2)
    metrics["result"] = {"resident_buildings": len(rows)}
    write_scores(os.path.join(outdir, "pg_uv_building_score.csv"), rows)
    with open(os.path.join(outdir, "pg_uv_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    m.close(); g.close()
    print(f"[PG] 端到端 / end-to-end {metrics['stages']['total_end_to_end']:.1f}s "
          f"跨系统搬运 / cross-system xfers {len(xfer)}\n")
    return metrics, rows


# ══════════════════════════════════════════════════════════════════════
# 对比 + 诊断 / Compare + diagnose
# ══════════════════════════════════════════════════════════════════════
def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def compare(mg, rg, mp, rp, top_frac, ph):
    ph.step("对比 & 诊断 / Compare & diagnose")
    print("=" * W)
    print("  对比 & 诊断 / Compare & diagnose   G3R (图内/in-graph)  vs  PostGIS (外置/external)")
    print("=" * W)
    gd = {r["cc"]: r for r in rg}
    pd = {r["cc"]: r for r in rp}
    common = sorted(set(gd) & set(pd))

    # [1] 高分建筑集合一致性 / [1] High-score building-set consistency
    thg = score_threshold(rg, top_frac)
    thp = score_threshold(rp, top_frac)
    hg = {r["cc"] for r in rg if r["uv_score"] >= thg}
    hp = {r["cc"] for r in rp if r["uv_score"] >= thp}
    inter = len(hg & hp); uni = len(hg | hp)
    jac = inter / uni if uni else 0.0
    print(f"\n[1] 高分建筑集合 / high-score set (top {top_frac:.0%})")
    print(f"  G3R {len(hg):,}  PG {len(hp):,}  交集 / ∩ {inter:,}  Jaccard {jac:.3f}")

    if not common:
        print("\n[!] 无可对齐建筑 (main_code 交集为空) / no alignable buildings"); return
    print(f"\n[2] 逐建筑语义信号一致性 / per-building semantic-signal consistency (main_code, n={len(common):,})")
    print(f"  {'信号/signal':<12}{'完全一致率/eq%':>10}{'中位|Δ|/med':>10}{'均值|Δ|/mean':>10}")
    for key, cn in [("res_total", "居民总数 / res_total"), ("outprov_cnt", "外省人数 / outprov"),
                    ("n_origin", "籍贯种数 / n_origin")]:
        dg = np.array([gd[c][key] for c in common], dtype=float)
        dp = np.array([pd[c][key] for c in common], dtype=float)
        ad = np.abs(dg - dp)
        eq = float(np.mean(ad == 0) * 100)
        print(f"  {cn:<12}{eq:>9.1f}%{np.median(ad):>10.1f}{np.mean(ad):>10.1f}")

    # [3] 握手楼距离保真 / [3] Handshake distance fidelity
    hg_d = np.array([gd[c].get("nn_dist", 0.0) for c in common])
    hp_d = np.array([pd[c].get("nn_dist", 0.0) for c in common])
    ad = np.abs(hg_d - hp_d)
    corr = float(np.corrcoef(hg_d, hp_d)[0, 1]) if len(common) > 2 else 0.0
    print(f"\n[3] 握手楼最近邻距离保真 / handshake nn fidelity (G3R grid KNN vs PG ST_Distance)")
    print(f"  中位|Δ|/med {np.median(ad):.2f}m  均值|Δ|/mean {np.mean(ad):.2f}m  "
          f"P95 {_pct(ad,95):.2f}m  P99 {_pct(ad,99):.2f}m  相关/corr {corr:.4f}")
    print(f"  G3R 中位/med {np.median(hg_d):.1f}m   PG 中位/med {np.median(hp_d):.1f}m")

    # [4] 复合评分一致性 / [4] Composite-score consistency
    sg = np.array([gd[c]["uv_score"] for c in common])
    sp = np.array([pd[c]["uv_score"] for c in common])
    ads = np.abs(sg - sp)
    scorr = float(np.corrcoef(sg, sp)[0, 1]) if len(common) > 2 else 0.0
    print(f"\n[4] 城中村复合评分一致性 / UV composite-score consistency (uv_score)")
    print(f"  中位|Δ|/med {np.median(ads):.4f}  均值|Δ|/mean {np.mean(ads):.4f}  "
          f"P95 {_pct(ads,95):.4f}  相关/corr {scorr:.4f}")

    # [5] 架构/成本 / [5] Architecture / cost
    print(f"\n[5] 架构 / 成本 / Architecture / cost")
    gs, ps = mg["stages"], mp["stages"]
    print(f"  {'阶段/stage':<22}{'G3R(s)':>10}{'PG(s)':>10}")
    print("  " + "-" * 42)
    pairs = [("A 聚合(语义+形态+人口) / aggregate", "A_aggregate_ingraph",
              "A_aggregate_crossdb"),
             ("B 握手楼 KNN / handshake KNN", "B_handshake_knn_ingraph", "B_handshake_knn"),
             ("C 评分 / score", "C_score", "C_score"),
             ("端到端 / e2e", "total_end_to_end", "total_end_to_end")]
    for label, gk, pk in pairs:
        print(f"  {label:<22}{gs.get(gk,0):>10.1f}{ps.get(pk,0):>10.1f}")
    ntx = len(mp["cross_system_transfers"])
    print(f"  {'跨系统搬运/xfers':<22}{'0':>10}{ntx:>10}")
    for x in mp["cross_system_transfers"]:
        print(f"      PG [{x['dir']:<14}] {x['what']:<22} rows={x['rows']:,}")
    print("\n[完成 / done]")


def main():
    limit = 0; outdir = "csv"; top_frac = 0.15
    only = None
    workers = min(os.cpu_count() or 1, 16)      # 握手楼 KNN 并行度; 1=串行 / handshake KNN workers
    prewarm = False                             # 对称预热 (消首跑冷缓存尖峰) / symmetric prewarm
    repeat = 1                                  # >1: 多次跑取每阶段 min/中位/max / multi-run stats
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
    print("#  城中村探查 单脚本实验  (G3R vs PostGIS) / Urban-village one-script experiment")
    print(f"#  limit={limit or '全量/full'}   top={top_frac:.0%}   outdir={outdir}"
          f"   workers={workers}")
    print("#" * W)

    if prewarm:
        _prewarm()

    g_runs, p_runs = [], []                     # 各次每阶段耗时 (供 --repeat 汇总) / per-run stage timings
    mg = rg = mp = rp = steps = None
    for it in range(1, repeat + 1):
        if repeat > 1:
            print("\n" + "#" * W)
            print(f"#  REPEAT {it}/{repeat}")
            print("#" * W)
        ph = Phases(7)   # G3R A/B/C + PG A/B/C + 对比 / compare
        steps = StepLog()                # 细粒度分步耗时 (含籍贯行政区划树解析) / fine timings
        mg, rg = run_g3r(limit, outdir, ph, steps=steps, workers=workers)
        g_runs.append(mg["stages"])
        # 冒烟(limit>0): PG 对齐 G3R 建筑集; 全量: 两侧独立 / smoke: align PG to G3R; full: independent
        restrict = [r["cc"] for r in rg] if limit > 0 else None
        mp, rp = run_pg(limit, outdir, restrict, ph, steps=steps)
        p_runs.append(mp["stages"])
        if steps.rows:
            sp = os.path.join(outdir, "uv_step_timings.csv")
            steps.write(sp)
            print(f"\n[细粒度耗时 / fine timings] {sp}  ({len(steps.rows)} 子步 / substeps)")

    compare(mg, rg, mp, rp, top_frac, ph)        # 用最后一次结果 / use last-run results
    if steps and steps.rows:
        _print_substeps(os.path.join(outdir, "uv_step_timings.csv"))

    if repeat > 1:
        print("\n" + "=" * W)
        print(f"  多次基准汇总 / multi-run bench summary (N={repeat}, 预热/prewarm {'开/on' if prewarm else '关/off'}, "
              f"workers={workers})")
        print("  取数建议: 每阶段用「中位」列; 极差% 大者受本机负载影响, 需减背景干扰 / Prefer median; large spread% ⇒ reduce background load")
        print("=" * W)
        if g_runs:
            _bench_summary(g_runs, "G3R")
        if p_runs:
            _bench_summary(p_runs, "PostGIS")


if __name__ == "__main__":
    main()

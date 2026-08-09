#!/usr/bin/env python3
# coding: utf-8
"""
深圳城中村探查 (G3R 图内统一执行) —— 无监督多信号复合评分。
Shenzhen urban-village exploration (G3R in-graph) — unsupervised multi-signal scoring.

定位 (重要, 防"标准不科学"质疑) / Positioning (important):
  * 本脚本是**方法能力示范**: 展示 G3R 能在一张图里融合「语义属性(外来户籍/籍贯) +
    空间形态(握手楼间距/密度)」做复合推理; **不是**声称给出权威城中村地图。
    Method capability demo: fuse semantic attrs + spatial form in one graph; NOT an
    authoritative urban-village map claim.
  * 判别信号沿用城中村识别文献的通用准则 (流动人口聚居 / 来源混杂 / 中低多层自建 /
    高密度紧凑肌理), 非自定; 权重等权 + 敏感性分析 (见 Part2)。
    Signals follow common literature criteria; equal weights + sensitivity (Part2).
  * 无官方标签 → exploratory; 结果用公认城中村 (白石洲/岗厦/上下沙…) 做定性叠图核验。
    No official labels → exploratory; qualitative overlay on known villages.

信号 (每栋居住建筑, 均图内可得) / Signals (per residential building, all in-graph):
  1 外省占比 / out-of-province share     ↑
  2 籍贯多样性 / origin diversity        ↑
  3 中低多层形态 / mid-rise form         ↑
  4 居住密度 / residential density       ↑
  5 握手楼间距 / handshake spacing       ↑(inverted)

Part1 (本文件当前 / this file): Stage A aggregate → B handshake KNN → C score
                    → csv/uv_building_score.csv
Part2 (后续追加 / later): grid connected components → candidates → name check + sensitivity.

用法 / Usage:
  python urban_village_g3r.py --limit 4000     # 冒烟 / smoke
  python urban_village_g3r.py                  # 全量 / full
"""
from __future__ import annotations
import csv, math, os, sys, threading, time
from typing import Dict, List

import numpy as np
from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geosot_overlay_query_v7 import GRID_LEVEL as GL, _code_to_xy
from geosot_knn_v3 import set_poi_prefix_index
from geosot_knn_v3_1 import GeoSotKNNSearcherV31
from run_mismatch_experiment import (
    NEO4J_URI, NEO4J_USER, NEO4J_PWD, NEO4J_DB, SubsetPOICache, _bar)

_t = time.perf_counter
W = 78
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Heartbeat:
    """单条阻塞查询(Cypher/SQL 大聚合)期间的存活心跳: 每 2s 打印已运行秒数,
    让长时间无输出的阶段不像卡死。用作上下文管理器。
    Liveness heartbeat during one blocking query: print elapsed every 2s."""
    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t0 = _t()
        while not self._stop.wait(2.0):
            sys.stdout.write(f"\r    {self.label} … 运行中 / running {_t()-t0:.0f}s")
            sys.stdout.flush()

    def __enter__(self):
        self._th.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self._th.join(timeout=0.2)
        sys.stdout.write("\r" + " " * 64 + "\r"); sys.stdout.flush()


class Phases:
    """整体阶段进度条: 每进入一个阶段调用 .step(title)。 / Overall stage progress: call .step(title) on each stage entry."""
    def __init__(self, n: int):
        self.i = 0; self.n = max(n, 1)

    def step(self, title: str):
        self.i += 1
        done = int(24 * self.i / self.n)
        bar = "█" * done + "░" * (24 - done)
        print(f"\n[总进度 / progress {bar} {self.i}/{self.n}]  {title}")

# 握手楼 KNN: k=2 (第1个是自身 dist0, 取第1个非自身邻居)。
# Handshake KNN: k=2 (1st is self at dist0; take first non-self neighbor).
# 邻居中位仅 ~13m → 无需 300m 保证半径的召回强化档: min_margin_m 300→50、
# Median nn is only ~13m → no need for 300m guaranteed-radius recall boost: min_margin_m 300→50,
# max_widen 11→8, 大幅减少每栋候选扫描 (城区 300m 半径含大量建筑) → 提速,
# max_widen 11→8, much fewer candidates scanned per building (300m urban radius is dense) → faster,
# 对"近邻"精度无损 (真最近邻在几米内即命中); 仅极稀疏区(nn>50m)可能提前停,
# near-neighbor precision intact (true nn hits within metres); only very sparse nn>50m may early-stop,
# 而那些正是高 nn/低握手分建筑, 精确值不影响其(低)得分。
# and those are high-nn / low-handshake buildings whose exact value does not affect (low) score.
_HS_KW = dict(k=2, min_margin_m=50.0, min_level=8, coarse_top_n=50,
              geometry_scoring=False, max_widen=8,
              max_candidates_per_level=500000, start_level_max=0)
_HS_MISS = 1000.0   # KNN 未找到邻居(真孤立)兜底: 大值 → 低握手分(不是 0.0=极紧凑) / isolated fallback: large → low handshake score

# 信号权重 (等权; Part2 做敏感性)。键须与 _signals 输出一致。 / Signal weights (equal; Part2 sensitivity). Keys must match signal outputs.
SIGNAL_KEYS = ["foreign", "diversity", "midrise", "density", "handshake"]
SIGNAL_CN = {"foreign": "外省占比 / out-of-province", "diversity": "籍贯多样性 / origin diversity",
             "midrise": "中低多层 / mid-rise", "density": "居住密度 / density",
             "handshake": "握手楼近 / handshake-near"}


# ── Stage A: 图内聚合 / in-graph aggregate ───────────────────────────
# 籍贯: 走行政区划树 (Resident)-[:HAS_COUNTY_ORIGIN]->(CountyOrigin)
# Origin: walk admin hierarchy (Resident)-[:HAS_COUNTY_ORIGIN]->(CountyOrigin)
#   -[:IN_PROVINCE]->(Province) / -[:IN_CITY]->(AdminCity), 由层级推理省/市,
#   -[:IN_PROVINCE]->(Province) / -[:IN_CITY]->(AdminCity), infer province/city from hierarchy,
# 而非在属性串上截前缀 → 与 PG 端 join county_origin_dict 对称。
# not string-prefix on the attribute → symmetric to PG join county_origin_dict.
# 已验证: 外省/非深两口径与 STARTS WITH 前缀在全量 18.5M 居民上 Δ=0。
# Verified: out-of-province / non-SZ scopes match STARTS WITH on 18.5M residents (Δ=0).
_AGG_TAIL = (
    "MATCH (b)<-[:LOCATED_IN]-(:House)<-[:LIVE_IN]-(r:Resident)\n"
    "OPTIONAL MATCH (r)-[:HAS_COUNTY_ORIGIN]->(co:CountyOrigin)\n"
    "OPTIONAL MATCH (co)-[:IN_PROVINCE]->(pr:Province)\n"
    "OPTIONAL MATCH (co)-[:IN_CITY]->(ci:AdminCity)\n"
    "WITH g.main_code AS cc, b.layers AS layers, b.buildtype AS buildtype,\n"
    "     b.buildingusage AS usage, b.buildingarea AS bldarea,\n"
    "     b.address AS address,\n"
    "     count(r) AS res_total,\n"
    "     sum(CASE WHEN pr IS NULL "
    "          OR NOT (pr.name = '广东省' AND ci.name = '深圳市') "
    "          THEN 1 ELSE 0 END) AS foreign_cnt,\n"
    "     sum(CASE WHEN pr IS NULL OR pr.name <> '广东省' "
    "          THEN 1 ELSE 0 END) AS outprov_cnt,\n"
    "     count(DISTINCT co) AS n_origin\n"
    "RETURN cc, layers, buildtype, usage, bldarea, address, res_total, "
    "foreign_cnt, outprov_cnt, n_origin")


def _row_from(r) -> dict:
    cc = str(r["cc"]).strip().upper()
    if not cc:
        return None
    try:
        layers = int(r["layers"]) if r["layers"] is not None else None
    except (ValueError, TypeError):
        layers = None
    try:
        bldarea = float(r["bldarea"]) if r["bldarea"] is not None else None
    except (ValueError, TypeError):
        bldarea = None
    return {"cc": cc, "layers": layers, "buildtype": str(r["buildtype"]),
            "usage": str(r["usage"]), "bldarea": bldarea,
            "address": (r["address"] or "").strip(),
            "res_total": int(r["res_total"]),
            "foreign_cnt": int(r["foreign_cnt"]),
            "outprov_cnt": int(r["outprov_cnt"]),
            "n_origin": int(r["n_origin"])}


def stage_a(sess, limit: int, steps=None) -> List[dict]:
    print("[A] 图内聚合: 每栋居住建筑 语义(外来/籍贯) + 形态(层数/占地) + 人口 ... / In-graph aggregate: semantics + form + population ...")
    t0 = _t()
    rows = []
    if limit > 0:
        # 冒烟: 先取 Z-order 前 N 栋(空间连续块, 快), 再仅对这些建筑聚合居民 →
        # Smoke: take first N buildings by Z-order (spatial contiguous block, fast), then aggregate residents only for them →
        # 不扫全图, 真正随 --limit 缩小。
        # do not scan the full graph; truly shrinks with --limit.
        ccs = [str(r["cc"]).strip().upper() for r in sess.run(
            "MATCH (b:Building)-[:HAS_GEOSOT]->(g:GeoSOTCode)\n"
            "WHERE EXISTS { (b)<-[:LOCATED_IN]-(:House)<-[:LIVE_IN]-(:Resident) }\n"
            "RETURN g.main_code AS cc ORDER BY cc LIMIT $lim", lim=limit)]
        q = ("MATCH (b:Building)-[:HAS_GEOSOT]->(g:GeoSOTCode)\n"
             "WHERE g.main_code IN $ccs\n" + _AGG_TAIL)
        with Heartbeat("图内聚合居民/形态 / in-graph aggregate residents/form"):
            for r in sess.run(q, ccs=ccs):
                row = _row_from(r)
                if row:
                    rows.append(row)
    else:
        # 全量: 单趟聚合 (与 mismatch StageA 同高效范式)。 / Full: single-pass aggregate (same as mismatch StageA).
        q = ("MATCH (b:Building)-[:HAS_GEOSOT]->(g:GeoSOTCode)\n"
             "WHERE EXISTS { (b)<-[:LOCATED_IN]-(:House)<-[:LIVE_IN]-(:Resident) }\n"
             + _AGG_TAIL + " ORDER BY cc")
        with Heartbeat("图内聚合居民/形态(全量) / in-graph aggregate (full)"):
            for r in sess.run(q):
                row = _row_from(r)
                if row:
                    rows.append(row)
    print(f"    有居民建筑 / resident bldgs {len(rows):,}  {_t()-t0:.1f}s (跨库0 / 0 xfers)")
    if steps is not None:
        steps.add("g3r", "A", "籍贯行政区划树遍历+形态/人口聚合 / origin-tree + form/pop aggregate", _t() - t0,
                  rows=len(rows), note="CountyOrigin→Province/City 图内, 跨库0 / in-graph, 0 xfers")
    return rows


# ── Stage B: 握手楼楼间距 (GeoSOT 建筑↔建筑 KNN, 空间桥) ─────────────
# Stage B: handshake spacing (GeoSOT building↔building KNN, spatial bridge)
# 多进程并行: 握手楼 KNN 是逐建筑独立检索, 与 mismatch StageD KNN 同构 →
# Multiprocess: handshake KNN is per-building independent, isomorphic to mismatch StageD KNN →
# 复用 run_mismatch_experiment._parallel_map。目标集缓存(全体建筑 CC)由主
# reuse run_mismatch_experiment._parallel_map. Target-set cache (all building CCs) is built by parent
# 进程建一次, pickle 传入各 worker; searcher 构造极轻(仅存 session)。
# once and pickled into workers; searcher construction is tiny (session only).
_MP_HS: dict = {}


def _mp_hs_init(uri, auth, db, cache):
    """worker 初始化: 各建 Neo4j session + searcher, 装载共享握手楼目标集缓存。 / Worker init: Neo4j session + searcher; load shared handshake target cache."""
    drv = GraphDatabase.driver(uri, auth=tuple(auth))
    sess = drv.session(database=db)
    set_poi_prefix_index(cache)
    _MP_HS.update(sess=sess, cache=cache,
                  searcher=GeoSotKNNSearcherV31(sess, geom_type="Point",
                                                verbose=False))


def _mp_hs_work(chunk):
    """一个 CC 块: 逐栋取第 1 个非自身最近邻距离; 孤立兜底 _HS_MISS。 / One CC chunk: first non-self nearest distance; isolated → _HS_MISS."""
    searcher = _MP_HS["searcher"]
    set_poi_prefix_index(_MP_HS["cache"])
    out = {}
    for cc in chunk:
        nn = None
        try:
            res, _ = searcher.knn_search(cc, **_HS_KW)
            for item in res:
                code, dist = item[0], item[1]
                if str(code).strip().upper() != cc:
                    nn = float(dist); break
        except Exception:
            nn = None
        out[cc] = nn if nn is not None else _HS_MISS
    return out


def stage_b(sess, rows: List[dict], steps=None, workers: int = 1) -> None:
    print(f"[B] 空间桥·KNN: 每栋最近邻建筑距离 (握手楼肌理) ... / nearest-building distance (handshake fabric) ... (workers={workers})")
    t0 = _t()
    ccs = [r["cc"] for r in rows]
    t_cache = _t()
    cache = SubsetPOICache(ccs)                 # 全体建筑 CC 作 KNN 目标集 / all building CCs as KNN targets
    set_poi_prefix_index(cache)
    if steps is not None:
        steps.add("g3r", "B", "握手楼 KNN 目标集缓存构建 / handshake KNN target cache", _t() - t_cache,
                  rows=len(ccs))
    t_search = _t()
    N = len(rows)
    if workers > 1:
        from run_mismatch_experiment import _parallel_map
        dist_map: Dict[str, float] = {}
        for part in _parallel_map(
                _mp_hs_init, (NEO4J_URI, [NEO4J_USER, NEO4J_PWD],
                              NEO4J_DB, cache),
                _mp_hs_work, ccs, workers, t0):
            dist_map.update(part)
        for r in rows:
            r["nn_dist"] = dist_map.get(r["cc"], _HS_MISS)
    else:
        searcher = GeoSotKNNSearcherV31(sess, geom_type="Point", verbose=False)
        step = max(N // 100, 1)
        for i, r in enumerate(rows):
            if i % step == 0:
                _bar(i, N, t0)
            nn = None
            try:
                res, _ = searcher.knn_search(r["cc"], **_HS_KW)
                for item in res:
                    code, dist = item[0], item[1]
                    if str(code).strip().upper() != r["cc"]:
                        nn = float(dist); break
            except Exception:
                nn = None
            # 找到邻居(含同格 dist≈0)→ 用真值; 真找不到(孤立)→ 大值(低握手分) / neighbor found → true dist; isolated → large (low score)
            r["nn_dist"] = nn if nn is not None else _HS_MISS
        sys.stdout.write(f"\r    [{'#'*30}] 100.0%  {N:,}/{N:,}\n")
        searcher.close()
    if steps is not None:
        steps.add("g3r", "B", "空间桥·握手楼最近邻逐栋检索 / per-building handshake nn", _t() - t_search,
                  rows=N, note=f"workers={workers}")
    print(f"    完成 / done {N:,}  {_t()-t0:.1f}s")


# ── Stage C: 归一化 + 等权复合评分 / normalize + equal-weight score ──
def _rnorm(vals):
    """robust min-max: p5/p95 裁剪后线性到 [0,1]; 返回映射函数。 / robust min-max: clip p5/p95 then linear map to [0,1]; return mapper."""
    arr = np.array([v for v in vals if v is not None and v == v], dtype=float)
    if arr.size == 0:
        return lambda v: 0.5
    lo, hi = np.percentile(arr, [5, 95])
    if hi <= lo:
        hi = lo + 1e-9
    return lambda v: (0.5 if (v is None or v != v)
                      else min(1.0, max(0.0, (v - lo) / (hi - lo))))


def stage_c(rows: List[dict], weights: Dict[str, float] = None) -> None:
    print("[C] 归一化 + 等权复合评分 ... / normalize + equal-weight composite score ...")
    w = weights or {k: 1.0 for k in SIGNAL_KEYS}
    wsum = sum(w[k] for k in SIGNAL_KEYS)
    # 原始信号 / raw signals
    for r in rows:
        rt = max(r["res_total"], 1)
        # 外省(非广东)占比: 比"非深圳户籍"更有区分力(深圳本就近全民非深籍) / out-of-province share (more discriminative than non-SZ hukou)
        r["s_foreign"] = r["outprov_cnt"] / rt
        r["s_diversity"] = float(r["n_origin"])
        ly = r["layers"]
        r["s_midrise"] = (0.0 if ly is None
                          else max(0.0, 1.0 - abs(ly - 6.5) / 6.5))
        area = r["bldarea"]
        area = min(max(area, 30.0), 5000.0) if (area and area > 0) else 200.0
        r["s_density"] = r["res_total"] / area
        r["s_handshake"] = r.get("nn_dist", 0.0)  # 越小越像; 下方取反归一 / smaller → more alike; invert below
    # 归一化器 / normalizers
    nf = _rnorm([r["s_foreign"] for r in rows])
    nd = _rnorm([r["s_diversity"] for r in rows])
    ng = _rnorm([r["s_density"] for r in rows])
    nh = _rnorm([r["s_handshake"] for r in rows])
    for r in rows:
        f = nf(r["s_foreign"])
        dv = nd(r["s_diversity"])
        mr = r["s_midrise"]                       # 已在 [0,1] / already in [0,1]
        de = ng(r["s_density"])
        hs = 1.0 - nh(r["s_handshake"])           # 间距小 → 分高 / small spacing → high score
        r["n_foreign"], r["n_diversity"] = f, dv
        r["n_midrise"], r["n_density"], r["n_handshake"] = mr, de, hs
        r["uv_score"] = (w["foreign"]*f + w["diversity"]*dv + w["midrise"]*mr
                         + w["density"]*de + w["handshake"]*hs) / wsum
    scores = sorted((r["uv_score"] for r in rows), reverse=True)
    n = len(scores)
    if n:
        p = lambda q: scores[min(int(q*n), n-1)]
        print(f"    评分分布 / score dist  p50 {p(0.5):.3f}  p80 {p(0.2):.3f}  "
              f"p90 {p(0.1):.3f}  p95 {p(0.05):.3f}  max {scores[0]:.3f}")


def write_scores(path: str, rows: List[dict]) -> None:
    cols = (["cc", "res_total", "foreign_cnt", "outprov_cnt", "n_origin",
             "layers", "buildtype", "usage", "bldarea", "nn_dist"]
            + [f"n_{k}" for k in SIGNAL_KEYS] + ["uv_score"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in rows:
            wr.writerow([r.get(c, "") for c in cols])
    print(f"[写出 / write] {path}  ({len(rows):,} 行 / rows)")


# ── Part2 Stage D: GeoSOT/米制网格 连通分量聚类 / metric-grid connected components ──
def _pull_coords_pg(ccs):
    """从 PostGIS building_new 取高分建筑质心，供米制网格聚类/出图。
    Pull high-score building centroids from PostGIS for metric-grid clustering/plots.

    库内几何为 EPSG:4490。分析距离一律用 geography；此处仅把质心临时投到
    CLUSTER_METRIC_SRID（深圳 3° GK）以便按米网格做连通聚类，不是距离基线。
    Geometry is EPSG:4490; analysis distances use geography. Centroids are temporarily
    projected to CLUSTER_METRIC_SRID (Shenzhen 3° GK) for metre-grid clustering only.
    """
    import psycopg2
    from run_mismatch_experiment import PG_MAIN, CLUSTER_METRIC_SRID
    out = {}
    conn = psycopg2.connect(**PG_MAIN)
    ccl = sorted(set(ccs))
    with conn.cursor() as c:
        for i in range(0, len(ccl), 20000):
            chunk = ccl[i:i+20000]
            c.execute(
                "SELECT main_code, "
                " ST_X(ST_Transform(ST_Centroid(geometry),%s)), "
                " ST_Y(ST_Transform(ST_Centroid(geometry),%s)) "
                "FROM building_new WHERE main_code = ANY(%s)",
                (CLUSTER_METRIC_SRID, CLUSTER_METRIC_SRID, chunk))
            for mc, x, y in c.fetchall():
                if mc and x is not None:
                    out[str(mc).strip().upper()] = (float(x), float(y))
    conn.close()
    return out


def _find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]; x = p[x]
    return x


def stage_d(rows, score_th, cell, min_bldg):
    from collections import defaultdict
    print(f"[D] 网格连通聚类 / grid connected clustering: score>={score_th:.3f}  cell={cell:.0f}m  "
          f"min_bldg={min_bldg} ...")
    t0 = _t()
    hi = [r for r in rows if r["uv_score"] >= score_th]
    coords = _pull_coords_pg([r["cc"] for r in hi])
    for r in hi:
        r["_xy"] = coords.get(r["cc"])
    hi = [r for r in hi if r.get("_xy")]
    cells = defaultdict(list)
    for r in hi:
        x, y = r["_xy"]
        cells[(int(x // cell), int(y // cell))].append(r)
    occ = list(cells.keys())
    idx = {k: i for i, k in enumerate(occ)}
    parent = list(range(len(occ)))
    occset = set(occ)
    for (cx, cy) in occ:
        a = idx[(cx, cy)]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (cx + dx, cy + dy)
                if nb in occset:
                    ra, rb = _find(parent, a), _find(parent, idx[nb])
                    if ra != rb:
                        parent[ra] = rb
    comp = defaultdict(list)
    for k in occ:
        comp[_find(parent, idx[k])].append(k)
    clusters = []
    for keys in comp.values():
        bs = [r for k in keys for r in cells[k]]
        if len(bs) >= min_bldg:
            clusters.append(bs)
    clusters.sort(key=lambda bs: sum(r["res_total"] for r in bs), reverse=True)
    print(f"    高分建筑 / high-score bldgs {len(hi):,}  → 候选连片区 / candidate clusters {len(clusters):,} "
          f"(>={min_bldg}栋/bldgs)  {_t()-t0:.1f}s")
    return clusters


# ── Part2 Stage E: 社区名核验 + 敏感性 / name check + sensitivity ─────
import re
_DIGIT = re.compile(r"[0-9０-９]")
# 公认城中村关键词 (定性叠图核验用; 可扩) / Known urban-village keywords (qualitative overlay)
KNOWN_UV = ["白石洲", "岗厦", "上沙", "下沙", "沙嘴", "大冲", "蔡屋围", "水围",
            "皇岗", "新洲", "石厦", "上梅林", "下梅林", "白石龙", "民治", "龙华",
            "清湖", "南头", "向南", "大新", "南山村", "北头", "田面", "湖贝",
            "罗芳", "渔民", "湖南", "笋岗", "清水河", "布吉", "坂田", "平湖",
            "湾厦", "商乐", "渔一", "渔二", "后海", "登良", "白石", "岗贝",
            "上步", "赤尾", "福田村", "沙尾", "沙嘴", "新沙", "冈厦"]
# "村/旧村/新村/围/坊"等 名字自证村落 (深圳城中村多此命名) / Name self-evidence of village (common SZ UV naming)
_VILLAGE_WORD = re.compile(r"(旧村|新村|.村|.围$|.坊$)")


def _addr_base(a):
    """地址→村名标签: 切到首个数字前, 去尾部'巷/号/栋/幢/座'等。 / Address→village label: cut before first digit; strip trailing unit markers."""
    if not a:
        return ""
    m = _DIGIT.search(a)
    s = a[:m.start()] if m else a
    for mk in ("巷", "号", "栋", "幢", "座", "村委", "居委"):
        j = s.find(mk)
        if j > 0:
            s = s[:j]
    return s.strip()


def stage_e(clusters, outdir, weights_label="等权 / equal-weight"):
    from collections import Counter
    print("[E] 候选连片区命名 + 公认城中村核验 ... / name candidates + known-UV check ...")
    recs = []
    for i, bs in enumerate(clusters):
        names = Counter(_addr_base(r["address"]) for r in bs if r["address"])
        name = names.most_common(1)[0][0] if names else "(无地址 / no address)"
        xs = [r["_xy"][0] for r in bs]; ys = [r["_xy"][1] for r in bs]
        rec = {
            "cluster_id": i + 1, "name": name, "n_bldg": len(bs),
            "res_total": sum(r["res_total"] for r in bs),
            "mean_score": round(sum(r["uv_score"] for r in bs) / len(bs), 3),
            "cx": round(sum(xs) / len(xs), 1), "cy": round(sum(ys) / len(ys), 1),
            "span_m": round(max(max(xs)-min(xs), max(ys)-min(ys)), 0),
            "hit_known": any(k in name for k in KNOWN_UV),
            "is_village_name": bool(_VILLAGE_WORD.search(name))}
        recs.append(rec)
    hit = sum(1 for r in recs if r["hit_known"])
    vil = sum(1 for r in recs if r["hit_known"] or r["is_village_name"])
    print(f"    Top 候选连片区 (按人口, 前20) / Top candidate clusters (by pop, top 20):")
    print(f"    {'#':>3} {'名称/name':<16} {'栋/n':>5} {'人口/pop':>8} {'均分/mean':>6} "
          f"{'跨度m/span':>7}  证据/evidence")
    for r in recs[:20]:
        mark = ("公认✓/known" if r["hit_known"]
                else ("村名/village-name" if r["is_village_name"] else ""))
        print(f"    {r['cluster_id']:>3} {r['name'][:16]:<16} {r['n_bldg']:>5} "
              f"{r['res_total']:>8,} {r['mean_score']:>6.3f} "
              f"{r['span_m']:>7.0f}  {mark}")
    print(f"    命中公认城中村关键词 / hit known UV keywords: {hit}/{len(recs)}   "
          f"名字自证村落(含公认) / self-evident village names: {vil}/{len(recs)}")
    path = os.path.join(outdir, "uv_clusters.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0].keys()) if recs
                            else ["cluster_id"])
        wr.writeheader()
        for r in recs:
            wr.writerow(r)
    print(f"[写出 / write] {path}  ({len(recs)} 个候选连片区 / candidate clusters)")
    return recs


def score_threshold(rows, top_frac):
    s = sorted((r["uv_score"] for r in rows), reverse=True)
    if not s:
        return 1.0
    return s[min(int(top_frac * len(s)), len(s) - 1)]


def main():
    limit = 0
    outdir = "csv"
    top_frac = 0.15
    cell = 150.0
    min_bldg = 10
    part2 = True
    workers = min(os.cpu_count() or 1, 16)      # 握手楼 KNN 并行度; 1=串行 / handshake KNN workers; 1=serial
    a = sys.argv[1:]; i = 0
    while i < len(a):
        if a[i] == "--limit" and i + 1 < len(a):
            limit = int(a[i+1]); i += 2
        elif a[i] == "--outdir" and i + 1 < len(a):
            outdir = a[i+1]; i += 2
        elif a[i] == "--top" and i + 1 < len(a):
            top_frac = float(a[i+1]); i += 2
        elif a[i] == "--cell" and i + 1 < len(a):
            cell = float(a[i+1]); i += 2
        elif a[i] == "--min-bldg" and i + 1 < len(a):
            min_bldg = int(a[i+1]); i += 2
        elif a[i] == "--workers" and i + 1 < len(a):
            workers = max(int(a[i+1]), 1); i += 2
        elif a[i] == "--no-part2":
            part2 = False; i += 1
        else:
            i += 1
    os.makedirs(outdir, exist_ok=True)
    print("#" * W)
    print("#  深圳城中村探查 (G3R 图内, 无监督复合评分)  Part1: 评分 / SZ urban-village (G3R) Part1: score")
    print(f"#  limit={limit or '全量/full'}   信号/signals={'/'.join(SIGNAL_CN[k] for k in SIGNAL_KEYS)}")
    print("#" * W)

    ph = Phases(5 if part2 else 3)
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    sess = drv.session(database=NEO4J_DB)
    t_all = _t()
    ph.step("A 图内聚合 语义+形态+人口 / in-graph aggregate"); rows = stage_a(sess, limit)
    ph.step("B 空间桥·握手楼 KNN / handshake KNN"); stage_b(sess, rows, workers=workers)
    ph.step("C 归一化+复合评分 / normalize+score"); stage_c(rows)
    write_scores(os.path.join(outdir, "uv_building_score.csv"), rows)
    sess.close(); drv.close()
    if part2 and rows:
        ph.step("D 网格连通聚类 / grid clustering")
        th = score_threshold(rows, top_frac)
        clusters = stage_d(rows, th, cell, min_bldg)
        ph.step("E 命名+公认城中村核验 / name + known-UV check")
        if clusters:
            stage_e(clusters, outdir)
    print(f"\n[完成 / done] 端到端 / end-to-end {_t()-t_all:.1f}s")


if __name__ == "__main__":
    main()

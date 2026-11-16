#!/usr/bin/env python3
# coding: utf-8
"""
Exploratory urban-village candidate identification through unsupervised multi-signal
scoring. This demonstrates semantic/spatial integration rather than an authoritative village
map. No official labels are assumed; known village names support qualitative inspection,
with equal weights and sensitivity analysis.

Per residential building, combine: out-of-province resident share; number of distinct
origins; mid/low-rise form near 4..9 floors; population per clipped footprint area; and
inverse nearest-building spacing.

Stage A aggregates graph attributes, B performs GeoSOT building KNN, and C produces equal-
weight scores in csv/uv_building_score.csv. Subsequent stages cluster candidates and inspect
neighborhood names/sensitivity. Individual-level source records are not distributed.

Usage:
    python urban_village_g3r.py --limit 4000
    python urban_village_g3r.py
"""
from __future__ import annotations
import csv, math, os, sys, threading, time
from typing import Dict, List

import numpy as np
from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geosot_overlay_query_v7 import GRID_LEVEL as GL, _code_to_xy
from geosot_knn_v3 import set_poi_prefix_index
from geosot_knn_v3_2 import GeoSotKNNSearcherV32
from run_mismatch_experiment import (
    NEO4J_URI, NEO4J_USER, NEO4J_PWD, NEO4J_DB, SubsetPOICache, _bar)

_t = time.perf_counter
W = 78
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Heartbeat:
    """
    Context manager printing elapsed seconds every two seconds while a blocking Cypher/SQL
    aggregation runs, so long silent stages remain observable.
    """
    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t0 = _t()
        while not self._stop.wait(2.0):
            sys.stdout.write(f"\r    {self.label} … 运行中 {_t()-t0:.0f}s")
            sys.stdout.flush()

    def __enter__(self):
        self._th.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self._th.join(timeout=0.2)
        sys.stdout.write("\r" + " " * 64 + "\r"); sys.stdout.flush()


class Phases:
    """Overall stage progress: call .step(title) on each stage entry."""
    def __init__(self, n: int):
        self.i = 0; self.n = max(n, 1)

    def step(self, title: str):
        self.i += 1
        done = int(24 * self.i / self.n)
        bar = "█" * done + "░" * (24 - done)
        print(f"\n[总进度 {bar} {self.i}/{self.n}]  {title}")

# Building-spacing KNN uses k=2: skip the self-match at zero distance and take the first other building.
# Historical median spacing was about 13 m, so reduce min_margin_m from 300 to 50
# and max_widen from 11 to 8 to reduce dense urban candidate scans.
# This preserves nearby hits but can stop early in very sparse areas with nearest spacing above 50 m;
# those buildings receive low proximity scores, although their exact spacing can differ.
# Same Point reranker as the paper: planar stop, leading 8K local-ellipsoid candidates, Top-K Vincenty (K=2: shortlist 16).
_HS_KW = dict(k=2, min_margin_m=50.0, min_level=8, coarse_top_n=50,
              geometry_scoring=False, max_widen=8,
              max_candidates_per_level=500000, start_level_max=0,
              max_ellipsoid_n=-1, rank_kernel="local", refine_k=True)
_HS_MISS = 1000.0   # No-neighbor fallback: use a large spacing and low proximity score, never zero spacing.

# Signal weights (equal; Part2 sensitivity). Keys must match signal outputs.
SIGNAL_KEYS = ["foreign", "diversity", "midrise", "density", "handshake"]
SIGNAL_CN = {"foreign": "外省占比", "diversity": "籍贯多样性",
             "midrise": "中低多层", "density": "居住密度",
             "handshake": "握手楼近"}


# ---- Stage A: graph aggregation ----
# Infer origin through Resident-[:HAS_COUNTY_ORIGIN]->CountyOrigin
# -[:IN_PROVINCE]->Province / -[:IN_CITY]->AdminCity,
# matching the PG county_origin_dict join rather than parsing string prefixes.
# Historical validation on 18.5M residents found zero count differences for nonlocal/non-Shenzhen definitions.
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
    print("[A] 图内聚合: 每栋居住建筑 语义(外来/籍贯) + 形态(层数/占地) + 人口 ...")
    t0 = _t()
    rows = []
    if limit > 0:
        # Smoke runs select the first N buildings in Z-order, then aggregate residents only for that block,
        # so --limit avoids scanning the full graph.
        ccs = [str(r["cc"]).strip().upper() for r in sess.run(
            "MATCH (b:Building)-[:HAS_GEOSOT]->(g:GeoSOTCode)\n"
            "WHERE EXISTS { (b)<-[:LOCATED_IN]-(:House)<-[:LIVE_IN]-(:Resident) }\n"
            "RETURN g.main_code AS cc ORDER BY cc LIMIT $lim", lim=limit)]
        q = ("MATCH (b:Building)-[:HAS_GEOSOT]->(g:GeoSOTCode)\n"
             "WHERE g.main_code IN $ccs\n" + _AGG_TAIL)
        with Heartbeat("图内聚合居民/形态"):
            for r in sess.run(q, ccs=ccs):
                row = _row_from(r)
                if row:
                    rows.append(row)
    else:
        # Full: single-pass aggregate (same as mismatch StageA).
        q = ("MATCH (b:Building)-[:HAS_GEOSOT]->(g:GeoSOTCode)\n"
             "WHERE EXISTS { (b)<-[:LOCATED_IN]-(:House)<-[:LIVE_IN]-(:Resident) }\n"
             + _AGG_TAIL + " ORDER BY cc")
        with Heartbeat("图内聚合居民/形态(全量)"):
            for r in sess.run(q):
                row = _row_from(r)
                if row:
                    rows.append(row)
    print(f"    有居民建筑 {len(rows):,}  {_t()-t0:.1f}s (跨库0)")
    if steps is not None:
        steps.add("g3r", "A", "籍贯行政区划树遍历+形态/人口聚合", _t() - t0,
                  rows=len(rows), note="CountyOrigin→Province/City 图内, 跨库0")
    return rows


# ---- Stage B: interbuilding spacing through GeoSOT building-to-building KNN ----
# Each building's spacing query is independent, as in mismatch Stage D;
# reuse run_mismatch_experiment._parallel_map and build the full building-CC target cache
# once in the parent, then serialize it to workers. Searcher construction stores only a session.
_MP_HS: dict = {}


def _mp_hs_init(uri, auth, db, cache):
    """Worker init: Neo4j session + searcher; load shared handshake target cache."""
    drv = GraphDatabase.driver(uri, auth=tuple(auth))
    sess = drv.session(database=db)
    set_poi_prefix_index(cache)
    _MP_HS.update(sess=sess, cache=cache,
                  searcher=GeoSotKNNSearcherV32(sess, geom_type="Point",
                                                verbose=False))


def _mp_hs_work(chunk):
    """One CC chunk: first non-self nearest distance; isolated → _HS_MISS."""
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
    print(f"[B] 空间桥·KNN: 每栋最近邻建筑距离 (握手楼肌理) ... (workers={workers})")
    t0 = _t()
    ccs = [r["cc"] for r in rows]
    t_cache = _t()
    cache = SubsetPOICache(ccs)                 # Use all building centroid codes as the KNN target set.
    set_poi_prefix_index(cache)
    if steps is not None:
        steps.add("g3r", "B", "握手楼 KNN 目标集缓存构建", _t() - t_cache,
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
        searcher = GeoSotKNNSearcherV32(sess, geom_type="Point", verbose=False)
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
            # neighbor found → true dist; isolated → large (low score)
            r["nn_dist"] = nn if nn is not None else _HS_MISS
        sys.stdout.write(f"\r    [{'#'*30}] 100.0%  {N:,}/{N:,}\n")
        searcher.close()
    if steps is not None:
        steps.add("g3r", "B", "空间桥·握手楼最近邻逐栋检索", _t() - t_search,
                  rows=N, note=f"workers={workers}")
    print(f"    完成 {N:,}  {_t()-t0:.1f}s")


# ---- Stage C: normalization and equal-weight composite score ----
def _rnorm(vals):
    """robust min-max: clip p5/p95 then linear map to [0,1]; return mapper."""
    arr = np.array([v for v in vals if v is not None and v == v], dtype=float)
    if arr.size == 0:
        return lambda v: 0.5
    lo, hi = np.percentile(arr, [5, 95])
    if hi <= lo:
        hi = lo + 1e-9
    return lambda v: (0.5 if (v is None or v != v)
                      else min(1.0, max(0.0, (v - lo) / (hi - lo))))


def stage_c(rows: List[dict], weights: Dict[str, float] = None) -> None:
    print("[C] 归一化 + 等权复合评分 ...")
    w = weights or {k: 1.0 for k in SIGNAL_KEYS}
    wsum = sum(w[k] for k in SIGNAL_KEYS)
    # raw signals
    for r in rows:
        rt = max(r["res_total"], 1)
        # out-of-province share (more discriminative than non-SZ hukou)
        r["s_foreign"] = r["outprov_cnt"] / rt
        r["s_diversity"] = float(r["n_origin"])
        ly = r["layers"]
        r["s_midrise"] = (0.0 if ly is None
                          else max(0.0, 1.0 - abs(ly - 6.5) / 6.5))
        area = r["bldarea"]
        area = min(max(area, 30.0), 5000.0) if (area and area > 0) else 200.0
        r["s_density"] = r["res_total"] / area
        r["s_handshake"] = r.get("nn_dist", 0.0)  # Smaller values indicate stronger evidence; invert during normalization below.
    # normalizers
    nf = _rnorm([r["s_foreign"] for r in rows])
    nd = _rnorm([r["s_diversity"] for r in rows])
    ng = _rnorm([r["s_density"] for r in rows])
    nh = _rnorm([r["s_handshake"] for r in rows])
    for r in rows:
        f = nf(r["s_foreign"])
        dv = nd(r["s_diversity"])
        mr = r["s_midrise"]                       # Already in [0,1].
        de = ng(r["s_density"])
        hs = 1.0 - nh(r["s_handshake"])           # Smaller spacing produces a higher score.
        r["n_foreign"], r["n_diversity"] = f, dv
        r["n_midrise"], r["n_density"], r["n_handshake"] = mr, de, hs
        r["uv_score"] = (w["foreign"]*f + w["diversity"]*dv + w["midrise"]*mr
                         + w["density"]*de + w["handshake"]*hs) / wsum
    scores = sorted((r["uv_score"] for r in rows), reverse=True)
    n = len(scores)
    if n:
        p = lambda q: scores[min(int(q*n), n-1)]
        print(f"    评分分布  p50 {p(0.5):.3f}  p80 {p(0.2):.3f}  "
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
    print(f"[写出] {path}  ({len(rows):,} 行)")


# ---- Part 2, Stage D: connected-component clustering on GeoSOT/metric grids ----
def _pull_coords_pg(ccs):
    """
    Fetch high-score building centroids from PostGIS for metric-grid clustering/plotting. Stored
    geometries use EPSG:4490; analysis distances use geography. This function temporarily
    projects centroids to CLUSTER_METRIC_SRID for connected components on a meter grid, not the
    distance baseline.
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
    print(f"[D] 网格连通聚类: score>={score_th:.3f}  cell={cell:.0f}m  "
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
    print(f"    高分建筑 {len(hi):,}  → 候选连片区 {len(clusters):,} "
          f"(>={min_bldg}栋)  {_t()-t0:.1f}s")
    return clusters


# ---- Part 2, Stage E: neighborhood-name checks and sensitivity ----
import re
_DIGIT = re.compile(r"[0-9０-９]")
# Known urban-village keywords (qualitative overlay)
KNOWN_UV = ["白石洲", "岗厦", "上沙", "下沙", "沙嘴", "大冲", "蔡屋围", "水围",
            "皇岗", "新洲", "石厦", "上梅林", "下梅林", "白石龙", "民治", "龙华",
            "清湖", "南头", "向南", "大新", "南山村", "北头", "田面", "湖贝",
            "罗芳", "渔民", "湖南", "笋岗", "清水河", "布吉", "坂田", "平湖",
            "湾厦", "商乐", "渔一", "渔二", "后海", "登良", "白石", "岗贝",
            "上步", "赤尾", "福田村", "沙尾", "沙嘴", "新沙", "冈厦"]
# Name self-evidence of village (common SZ UV naming)
_VILLAGE_WORD = re.compile(r"(旧村|新村|.村|.围$|.坊$)")


def _addr_base(a):
    """Address→village label: cut before first digit; strip trailing unit markers."""
    if not a:
        return ""
    m = _DIGIT.search(a)
    s = a[:m.start()] if m else a
    for mk in ("巷", "号", "栋", "幢", "座", "村委", "居委"):
        j = s.find(mk)
        if j > 0:
            s = s[:j]
    return s.strip()


def stage_e(clusters, outdir, weights_label="等权"):
    from collections import Counter
    print("[E] 候选连片区命名 + 公认城中村核验 ...")
    recs = []
    for i, bs in enumerate(clusters):
        names = Counter(_addr_base(r["address"]) for r in bs if r["address"])
        name = names.most_common(1)[0][0] if names else "(无地址)"
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
    print(f"    Top 候选连片区 (按人口, 前20):")
    print(f"    {'#':>3} {'名称':<16} {'栋':>5} {'人口':>8} {'均分':>6} "
          f"{'跨度m':>7}  证据")
    for r in recs[:20]:
        mark = ("公认✓" if r["hit_known"]
                else ("村名" if r["is_village_name"] else ""))
        print(f"    {r['cluster_id']:>3} {r['name'][:16]:<16} {r['n_bldg']:>5} "
              f"{r['res_total']:>8,} {r['mean_score']:>6.3f} "
              f"{r['span_m']:>7.0f}  {mark}")
    print(f"    命中公认城中村关键词: {hit}/{len(recs)}   "
          f"名字自证村落(含公认): {vil}/{len(recs)}")
    path = os.path.join(outdir, "uv_clusters.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0].keys()) if recs
                            else ["cluster_id"])
        wr.writeheader()
        for r in recs:
            wr.writerow(r)
    print(f"[写出] {path}  ({len(recs)} 个候选连片区)")
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
    workers = min(os.cpu_count() or 1, 16)      # Building-spacing KNN workers; 1 selects serial execution.
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
    print('#  Exploratory urban-village identification (G3R, unsupervised), Part 1: scoring')
    print(f"#  limit={limit or '全量'}   信号={'/'.join(SIGNAL_CN[k] for k in SIGNAL_KEYS)}")
    print("#" * W)

    ph = Phases(5 if part2 else 3)
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    sess = drv.session(database=NEO4J_DB)
    t_all = _t()
    ph.step("A 图内聚合 语义+形态+人口"); rows = stage_a(sess, limit)
    ph.step("B 空间桥·握手楼 KNN"); stage_b(sess, rows, workers=workers)
    ph.step("C 归一化+复合评分"); stage_c(rows)
    write_scores(os.path.join(outdir, "uv_building_score.csv"), rows)
    sess.close(); drv.close()
    if part2 and rows:
        ph.step("D 网格连通聚类")
        th = score_threshold(rows, top_frac)
        clusters = stage_d(rows, th, cell, min_bldg)
        ph.step("E 命名+公认城中村核验")
        if clusters:
            stage_e(clusters, outdir)
    print(f"\n[完成] 端到端 {_t()-t_all:.1f}s")


if __name__ == "__main__":
    main()

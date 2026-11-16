#!/usr/bin/env python3
# coding: utf-8
"""
Public-service proximity under land-use mismatch: paired G3R and PostGIS pipelines.

Aggregate resident structure, assign building centroids to parcels, resolve/filter land-use
classes, and find nearest facilities in six categories. G3R uses semantic graph chains and
reverse-overlay/KNN operators; the PG baseline uses cross-database semantic aggregation,
ST_Contains, normalized dictionaries, and GiST/geography queries. Compare parcel sets,
accessibility, per-building nearest facilities, timings, and cross-system transfers.

With --limit, both sides use the same main_code-aligned building sample. Full runs execute
independently.

Usage:
    python run_mismatch_experiment.py
    python run_mismatch_experiment.py --limit 8000
    python run_mismatch_experiment.py --threshold 100
    python run_mismatch_experiment.py --skip-g3r
    python run_mismatch_experiment.py --only compare
"""
from __future__ import annotations
import csv, json, os, sys, time
from statistics import median
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from landuse_norm import classify, normalize
from neo4j import GraphDatabase
from geosot_reverse_query import (
    PlanPolygonCache, CommonPrefixIndex, set_common_prefix_index,
    _cascade_prefix_query, _pip_single, _point_in_bbox)
from geosot_overlay_query_v7 import GRID_LEVEL as GL, _code_to_xy, _xy_dms_to_linear
from service_categories import load_category_codes
from geosot_knn_v3 import POIPrefixCache, set_poi_prefix_index
from geosot_knn_v3_2 import GeoSotKNNSearcherV32

# ---- Configuration ----
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DB", "geocode")
PG_MAIN = {"host": "127.0.0.1", "port": 5432, "dbname": "postgis_35_sample",
           "user": "postgres", "password": os.getenv("PGPASSWORD", "")}
PG_MIG = {**PG_MAIN, "dbname": "oracle_mig"}
# All analysis geometries use CGCS2000 geographic coordinates, consistent with G3R/GeoSOT encoding.
# PostGIS distance/KNN baselines use geometry::geography in ellipsoidal meters, without a local GK projection.
SOURCE_SRID = 4490
# This projected CRS is only for metric-grid clustering and plotting intermediates, never the analysis distance path.
CLUSTER_METRIC_SRID = 4547  # CGCS2000 / 3-degree GK central meridian 114E (Shenzhen).
# Legacy import alias (do not use for ST_Distance/KNN)
METRIC_SRID = CLUSTER_METRIC_SRID
PARCEL_SRC = "Statutory Plan"
CHILD_BIRTH_MIN = "2001-01-01"
ELDER_BIRTH_MAX = "1956-01-01"

# Legacy six-category conditions on redundant poi_new f1/f2 fields; retained for reference, not used by the current pipeline.
PG_CAT_SQL = {
    "edu":       "f2 = '学校'",
    "medical":   "f2 IN ('综合医院','专科医院','诊所')",
    "sport":     "f1 = '体育休闲服务'",
    "scenic":    "f1 = '风景名胜'",
    "public":    "f1 = '公共设施'",
    "transport": "f1 = '交通设施服务' AND (f2 IS NULL OR f2 <> '停车场')",
}

# Normalized hierarchy resolution, symmetric to G3R's leaf-to-SUBCLASS_OF traversal:
# Assume poi_new stores only leaf code poi (equivalent to f3); broader classes require a dictionary join.
# The predicates below apply to gaode_poi_category: first resolve the target class to leaf codes,
# then filter poi_new. Historical checks found identical counts for all six categories versus f1/f2.
GAODE_CAT_SQL = {
    "edu":       "中类 = '学校'",
    "medical":   "中类 IN ('综合医院','专科医院','诊所')",
    "sport":     "大类 = '体育休闲服务'",
    "scenic":    "大类 = '风景名胜'",
    "public":    "大类 = '公共设施'",
    "transport": "大类 = '交通设施服务' AND 中类 <> '停车场'",
}
ORDER = ["edu", "medical", "sport", "scenic", "public", "transport"]
CAT_CN = {"edu": "教育-学校", "medical": "医疗", "sport": "体育休闲",
          "scenic": "风景名胜", "public": "公共设施", "transport": "交通设施"}

_t = time.perf_counter
W = 78
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

G3R_COLOR = "#2166AC"
PG_COLOR = "#B2182B"


# ---- Shared utilities ----
def _bar(k, n, t0, tag=""):
    pct = k / n if n else 1.0
    el = _t() - t0
    rate = k / el if el > 0.1 else 0
    eta = (n - k) / rate if rate > 0 else 0
    sys.stdout.write(
        f"\r    {tag}[{int(30*pct)*'#'}{(30-int(30*pct))*'.'}] "
        f"{pct*100:5.1f}%  {k:,}/{n:,}  {rate:,.0f}/s  ETA {eta:.0f}s")
    sys.stdout.flush()


def _poly_area(poly) -> float:
    """Shoelace area in integer Z-order coords (relative compare; no real metres)."""
    if poly is None or len(poly) < 3:
        return float("inf")
    n = len(poly)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += float(poly[i][0]) * float(poly[j][1]) \
           - float(poly[j][0]) * float(poly[i][1])
    return abs(a) / 2.0


def _combine_classes(pairs: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Combine each parcel's resolved (class,subtype) values under the mixed-use rules. This
    parcel-level aggregation is shared by G3R hierarchy traversal and the PG dictionary join,
    independently of per-name semantic resolution.
    """
    pairs = [(c, s) for c, s in pairs if c]
    if not pairs:
        return ("地块无用地功能", "地块无用地功能")
    classes = {c for c, _ in pairs}
    if "混合" in classes or ("住宅" in classes and "非住宅" in classes):
        return ("混合", "混合")
    if classes == {"住宅"}:
        return ("住宅", "住宅")
    st = next((s for c, s in pairs if c == "非住宅"), "其他非住宅")
    return ("非住宅", st)


def _classify_parcel_from_lus(lus: List[str]) -> Tuple[str, str]:
    """Per-name classify then combine (fallback; canonical path uses dict/graph hierarchy)."""
    return _combine_classes([classify(nm) for nm in lus if nm])


class StepLog:
    """Fine-grained step timing: one CSV row per substep (e.g. semantic hierarchy resolve)."""
    def __init__(self):
        self.rows: List[dict] = []

    def add(self, engine: str, stage: str, substep: str, sec: float,
            rows="", note=""):
        self.rows.append({"engine": engine, "stage": stage, "substep": substep,
                          "sec": round(sec, 4), "rows": rows, "note": note})

    def write(self, path: str):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["engine", "stage", "substep", "sec", "rows", "note"])
            for r in self.rows:
                wr.writerow([r["engine"], r["stage"], r["substep"],
                             r["sec"], r["rows"], r["note"]])


def _dump_parcels(path: str, parcels: Dict[str, Dict]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["parcel_main_code", "land_class", "land_subtype",
                     "res_total", "res_child", "res_elderly", "n_buildings"])
        for pmc, pa in parcels.items():
            wr.writerow([pmc, pa["class"], pa["subtype"], pa["res"],
                         pa["child"], pa["elder"], pa["nb"]])


def _select_parcels(parcel_agg, threshold):
    mismatch, baseline = {}, {}
    for pmc, pa in parcel_agg.items():
        if pa["res"] < threshold:
            continue
        if pa["class"] == "非住宅":
            mismatch[pmc] = pa
        elif pa["class"] == "住宅":
            baseline[pmc] = pa
    return mismatch, baseline


# Stage D scope: city=citywide (default, three groups); mb=mismatch+baseline; mismatch=mismatch only.
DTARGET_LABEL = {"city": "全市", "mb": "错配+基准", "mismatch": "仅错配"}


def _filter_targets(targets, dtarget):
    """
    Restrict Stage D targets by dtarget, using targets[i][2] as the group label. Stages A/B/C
    remain complete because mismatch identification needs full attribution and population; only
    per-building Stage D KNN follows the reduced target scope.
    """
    if dtarget == "mb":
        return [t for t in targets if t[2] in ("mismatch", "baseline")]
    if dtarget == "mismatch":
        return [t for t in targets if t[2] == "mismatch"]
    return targets                                     # city: all rows.


def _write_attr(path, records):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["building_cc", "parcel_main_code", "land_class",
                     "land_subtype", "res_total", "res_child",
                     "res_elderly", "n_parcel_hits"])
        wr.writerows(records)


def _write_acc(path, targets, dist):
    """(distance_m, nearest POI main_code); legacy scalar distance still accepted."""
    def _d(cc, k):
        v = dist.get(cc, {}).get(k)
        return v[0] if isinstance(v, tuple) else (v if v is not None else "")

    def _p(cc, k):
        v = dist.get(cc, {}).get(k)
        return v[1] if isinstance(v, tuple) else ""

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["building_cc", "parcel_main_code", "group", "land_subtype",
                     "res_total", "res_child", "res_elderly"]
                    + [f"d_{k}" for k in ORDER] + [f"p_{k}" for k in ORDER])
        for (cc, pmc, grp, sub, tot, ch, el) in targets:
            wr.writerow([cc, pmc, grp, sub, tot, ch, el]
                        + [_d(cc, k) for k in ORDER]
                        + [_p(cc, k) for k in ORDER])


class SubsetPOICache(POIPrefixCache):
    """Subset cache of one POI category main_codes (for per-class KNN)."""
    def __init__(self, codes):
        self._sorted, self._centers, self._code_idx = {}, {}, {}
        codes = sorted({c.strip().upper() for c in codes if c})
        self._sorted["Point"] = codes
        self._precompute_centers("Point", codes, verbose=False)
        self.elapsed_sec = 0.0


# Enhanced-recall settings: min_margin_m controls the guaranteed coverage radius.
# Increasing it forces a wider coverage search, reducing missed nearest neighbors in sparse
# public/scenic/education subsets. Historical parameter probes reduced misses
# from about 0.6% to about 0.05%, without a measured per-query latency increase in that run.
# Same Point reranker as the paper: planar stop, leading 8K local-ellipsoid candidates, Top-K Vincenty (K=1: shortlist 8).
_KNN_KW = dict(k=1, min_margin_m=300.0, min_level=8, coarse_top_n=50,
               geometry_scoring=False, max_widen=11,
               max_candidates_per_level=500000, start_level_max=0,
               max_ellipsoid_n=-1, rank_kernel="local", refine_k=True)


def _query_xy(cc: str):
    """
    Map a centroid code to the same linear integer space as PlanPolygonCache. Polygon vertices
    pass through _xy_dms_to_linear during setup; using packed DMS points against those vertices
    would incorrectly reject bbox/PIP matches.

    @param cc GeoSOT centroid code.
    @returns Linear integer (x,y).
    """
    lin = _xy_dms_to_linear(_code_to_xy(cc, GL))
    return (int(lin[0]), int(lin[1]))


def _attr_one(sess, pc, parcel_cls, src, cc, res, ch, el):
    """Reverse-overlay one building to its parcel; nh=-1 means decode failure (drop)."""
    try:
        xy = _query_xy(cc)
    except Exception:
        return (cc, "", "无所属地块", "无所属地块", res, ch, el, -1)
    hits = _cascade_prefix_query(sess, cc, xy, src, pc, tier_size=1)["results"]
    if not hits:
        return (cc, "", "无所属地块", "无所属地块", res, ch, el, 0)
    nh = len(hits)
    if nh > 1:
        pmc = min((str(h["main_code"]).strip().upper() for h in hits),
                  key=lambda m: _poly_area(pc.get_polygon(m)))
    else:
        pmc = str(hits[0]["main_code"]).strip().upper()
    cls, sub = parcel_cls.get(pmc, ("地块无用地功能", "地块无用地功能"))
    return (cc, pmc, cls, sub, res, ch, el, nh)


def _attr_one_lean(cc, res, ch, el, pc, cp_map, plens, parcel_cls):
    """
    Streamlined reverse overlay, preserving _attr_one semantics without intermediate objects.
    Iterate only existing prefix lengths in descending order, skip empty tiers, and access
    cp_map/get_bbox/get_polygon directly. Stop at the finest tier with a PIP hit; among multiple
    hits at that tier, select the smallest-area parcel, matching cascade tier_size=1.
    """
    try:
        xy = _query_xy(cc)
    except Exception:
        return (cc, "", "无所属地块", "无所属地块", res, ch, el, -1)
    L = len(cc)
    for k in plens:
        if k > L:
            continue
        mcs = cp_map.get(cc[:k])
        if not mcs:
            continue
        hits = []
        for mc in mcs:
            bb = pc.get_bbox(mc)
            if bb is None or not _point_in_bbox(xy, bb):
                continue
            poly = pc.get_polygon(mc)
            if poly is not None and len(poly) >= 3 and _pip_single(xy, poly):
                hits.append(mc)
        if hits:
            nh = len(hits)
            pmc = (hits[0] if nh == 1
                   else min(hits, key=lambda m: _poly_area(pc.get_polygon(m))))
            cls, sub = parcel_cls.get(pmc, ("地块无用地功能", "地块无用地功能"))
            return (cc, pmc, cls, sub, res, ch, el, nh)
    return (cc, "", "无所属地块", "无所属地块", res, ch, el, 0)


def _agg_parcels(records):
    """Aggregate parcel_agg from attribution records."""
    agg = {}
    for (cc, pmc, cls, sub, res, ch, el, nh) in records:
        if pmc:
            pa = agg.setdefault(pmc, {"class": cls, "subtype": sub,
                 "res": 0, "child": 0, "elder": 0, "nb": 0})
            pa["res"] += res; pa["child"] += ch; pa["elder"] += el; pa["nb"] += 1
    return agg


# ---- Process workers: partition buildings for stages B/D; share read-only cache state ----
_MP: Dict = {}


def _mp_b_init(uri, auth, db, parcel_src, pc, cp_map, plens, parcel_cls):
    """
    Receive polygon cache, prefix-to-parcel map, descending existing prefix lengths, and parcel
    classes serialized from the parent. This local path does not query Neo4j and needs no
    session.
    """
    _MP.update(pc=pc, cp_map=cp_map, plens=plens, parcel_cls=parcel_cls,
               src=parcel_src)


def _mp_b_work(chunk):
    pc, cp_map, plens, cls = (_MP["pc"], _MP["cp_map"], _MP["plens"],
                              _MP["parcel_cls"])
    return [_attr_one_lean(cc, res, ch, el, pc, cp_map, plens, cls)
            for (cc, res, ch, el) in chunk]


def _mp_d_init(uri, auth, db, caches):
    """
    Receive six serialized SubsetPOICache instances from the parent. Each worker creates a
    lightweight searcher that only stores its session reference.
    """
    drv = GraphDatabase.driver(uri, auth=tuple(auth))
    sess = drv.session(database=db)
    _MP.update(sess=sess, caches=caches,
               searcher=GeoSotKNNSearcherV32(sess, geom_type="Point",
                                             verbose=False))


def _mp_d_work(chunk):
    searcher, caches = _MP["searcher"], _MP["caches"]
    out = {}
    for cat in ORDER:
        set_poi_prefix_index(caches[cat])
        for cc in chunk:
            try:
                res, _ = searcher.knn_search(cc, **_KNN_KW)
                if res:
                    out.setdefault(cc, {})[cat] = (
                        round(res[0][1], 1), str(res[0][0]).strip().upper())
            except Exception:
                pass
    return out


def _chunks(items, nparts):
    k = max(1, (len(items) + nparts - 1) // nparts)
    return [items[i:i + k] for i in range(0, len(items), k)]


def _parallel_map(init_fn, initargs, work_fn, items, workers, t0):
    """Chunk by workers*8, parallel map, preserve order, with progress bar."""
    from concurrent.futures import ProcessPoolExecutor
    parts = _chunks(items, max(workers * 8, 1))
    N = len(items); done = 0; out = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_fn,
                             initargs=initargs) as ex:
        for i, res in enumerate(ex.map(work_fn, parts)):
            out.append(res); done += len(parts[i]); _bar(done, N, t0)
    sys.stdout.write(f"\r    [{'#'*30}] 100.0%  {N:,}/{N:,}\n")
    return out


# ══════════════════════════════════════════════════════════════════════
# G3R pipeline
# ══════════════════════════════════════════════════════════════════════
def run_g3r(limit: int, threshold: int, outdir: str,
            workers: int = 1, dtarget: str = "city",
            steps: "StepLog" = None) -> Tuple[Dict, List[str]]:
    steps = steps if steps is not None else StepLog()
    metrics = {"engine": "g3r", "threshold": threshold, "limit": limit,
               "dtarget": dtarget, "stages": {}, "cross_system_transfers": []}
    print("=" * W)
    print("[G3R] 图内统一执行流水线 (Neo4j + GeoSOT, 语义/几何/人口同图)")
    print("=" * W)
    t_all = _t()
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    sess = drv.session(database=NEO4J_DB)

    # A in-graph pull demographics
    print("[A] 图内拉有居民建筑 + 人口结构 ...")
    t0 = _t()
    lim = f"LIMIT {limit}" if limit > 0 else ""
    q = ("MATCH (b:Building)-[:HAS_GEOSOT]->(g:GeoSOTCode)\n"
         "WHERE EXISTS { (b)<-[:LOCATED_IN]-(:House)<-[:LIVE_IN]-(:Resident) }\n"
         "MATCH (b)<-[:LOCATED_IN]-(:House)<-[:LIVE_IN]-(r:Resident)\n"
         "WITH g.main_code AS cc, count(r) AS res,\n"
         "  sum(CASE WHEN r.birthday >= $c THEN 1 ELSE 0 END) AS child,\n"
         "  sum(CASE WHEN r.birthday <> '' AND r.birthday IS NOT NULL "
         "       AND r.birthday < $e THEN 1 ELSE 0 END) AS elder\n"
         f"RETURN cc, res, child, elder ORDER BY cc {lim}")
    buildings = [(str(r["cc"]).strip().upper(), int(r["res"]),
                  int(r["child"]), int(r["elder"]))
                 for r in sess.run(q, c=CHILD_BIRTH_MIN, e=ELDER_BIRTH_MAX)]
    metrics["stages"]["A_pull_residents_ingraph"] = round(_t() - t0, 2)
    steps.add("g3r", "A", "图内语义链聚合人口结构", _t() - t0,
              rows=len(buildings), note="Building←House←Resident 单趟, 跨库0")
    print(f"    有居民建筑 {len(buildings):,}  {_t()-t0:.1f}s (跨库0)")
    used_codes = [b[0] for b in buildings]

    # B: reverse overlay, tier_size=1; resolve multiple containing parcels by smallest area, as in PostGIS.
    print(f"[B] 空间桥·反向叠置 (CC 前缀+PIP) 归属地块 ... (workers={workers})")
    t0 = _t()
    auth = [NEO4J_USER, NEO4J_PWD]
    # Build the cache once in the parent and pass serialized state to workers,
    # avoiding repeated concurrent Neo4j loads and cache-construction contention.
    # Infer land-use classes through LUFunctionGroup-[:IN_LU_CLASS]->LUClass,
    # symmetric to the PG lu_function_dict join, rather than application-side keyword classification.
    t_lu = _t()
    parcel_cls = {}
    for r in sess.run(
        "MATCH (g:GeoSOTCode)-[:FROM_SOURCE]->(:Source {name:$s})\n"
        "OPTIONAL MATCH (g)-[:HAS_LU_FUNCTION_GROUP]->(t:LUFunctionGroup)\n"
        "OPTIONAL MATCH (t)-[:IN_LU_CLASS]->(lc:LUClass)\n"
        "OPTIONAL MATCH (t)-[:HAS_LU_SUBTYPE]->(ls:LUSubtype)\n"
        "RETURN g.main_code AS mc, collect([lc.name, ls.name]) AS pairs",
            s=PARCEL_SRC):
        pairs = [(a, b) for a, b in r["pairs"] if a]
        parcel_cls[str(r["mc"]).strip().upper()] = _combine_classes(pairs)
    steps.add("g3r", "B", "用地类型层级解析(IN_LU_CLASS 图内)", _t() - t_lu,
              rows=len(parcel_cls), note="图层级推理, 非应用层关键词")
    t_poly = _t()
    pc = PlanPolygonCache(sess, verbose=False); pc.preload(PARCEL_SRC)
    steps.add("g3r", "B", "地块多边形缓存预载", _t() - t_poly, rows=pc.count)
    t_cp = _t()
    cp = CommonPrefixIndex(sess, parcel_source=PARCEL_SRC, verbose=False)
    steps.add("g3r", "B", "GeoSOT 前缀倒排索引", _t() - t_cp)
    # Use an inverted prefix dictionary and descending existing prefix lengths to skip empty tiers.
    cp_map = cp._index
    plens = sorted({len(p) for p in cp_map}, reverse=True)
    print(f"    缓存就绪: {pc.count:,} 地块多边形  {len(plens)} 前缀长度  "
          f"{_t()-t0:.1f}s")
    t_ov = _t()
    raw = []
    if workers > 1:
        pc._session = None                       # Remove session references before pickling cache state.
        parts = _parallel_map(
            _mp_b_init,
            (NEO4J_URI, auth, NEO4J_DB, PARCEL_SRC, pc, cp_map, plens, parcel_cls),
            _mp_b_work, buildings, workers, t0)
        for p in parts:
            raw.extend(p)
    else:
        N = len(buildings); step = max(N // 100, 1)
        for k, (cc, res, ch, el) in enumerate(buildings):
            if k % step == 0:
                _bar(k, N, t0)
            raw.append(_attr_one_lean(cc, res, ch, el, pc, cp_map, plens,
                                      parcel_cls))
        sys.stdout.write(f"\r    [{'#'*30}] 100.0%  {N:,}/{N:,}\n")
    steps.add("g3r", "B", "空间桥·反向叠置归属(CC前缀+PIP)", _t() - t_ov,
              rows=len(raw), note=f"workers={workers}")
    derr = sum(1 for r in raw if r[7] == -1)
    records = [r for r in raw if r[7] != -1]
    parcel_agg = _agg_parcels(records)
    _write_attr(os.path.join(outdir, "g3r_parcel_attribution.csv"), records)
    metrics["stages"]["B_reverse_overlay_ingraph"] = round(_t() - t0, 2)
    print(f"    归属 {len(records):,} 解码失败 {derr}  {_t()-t0:.1f}s")

    # C filter
    t0 = _t()
    mismatch, baseline = _select_parcels(parcel_agg, threshold)
    _dump_parcels(os.path.join(outdir, "g3r_mismatch_parcels.csv"), mismatch)
    _dump_parcels(os.path.join(outdir, "g3r_baseline_parcels.csv"), baseline)
    metrics["stages"]["C_select"] = round(_t() - t0, 2)
    steps.add("g3r", "C", "筛选错配/基准地块(人口阈值)", _t() - t0,
              rows=len(mismatch) + len(baseline))
    print(f"[C] 错配 {len(mismatch):,}  基准 {len(baseline):,}")

    # D: KNN; dtarget selects citywide, mismatch+baseline, or mismatch-only buildings.
    print(f"[D] 空间桥·KNN 六类设施最近距离 ({DTARGET_LABEL[dtarget]}) ...")
    t0 = _t()
    # Resolve each POI category from HAS_POI_TYPE_3 leaves through SUBCLASS_OF; time each category separately.
    poi_tim: Dict[str, float] = {}
    cat_codes = load_category_codes(sess, timings=poi_tim)
    for kk in ORDER:
        steps.add("g3r", "D", f"POI类树解析:{CAT_CN[kk]}", poi_tim.get(kk, 0.0),
                  rows=len(cat_codes[kk]), note="叶子→SUBCLASS_OF 上溯")
    t_cache = _t()
    caches = {kk: SubsetPOICache(cat_codes[kk]) for kk in ORDER}  # Build once in the parent process.
    steps.add("g3r", "D", "六类 POI 子集缓存构建", _t() - t_cache)
    grp = {}
    for pmc in mismatch:
        grp[pmc] = "mismatch"
    for pmc in baseline:
        grp[pmc] = "baseline"
    targets = [(cc, pmc, grp.get(pmc, "other"), sub, tot, c2, e2)
               for (cc, pmc, cls, sub, tot, c2, e2, nh) in records]
    targets = _filter_targets(targets, dtarget)
    ccs = [t[0] for t in targets]
    print(f"    目标建筑 {len(targets):,} ({DTARGET_LABEL[dtarget]}) "
          f"错配 {sum(1 for t in targets if t[2]=='mismatch'):,} "
          f"基准 {sum(1 for t in targets if t[2]=='baseline'):,}")
    t_knn = _t()
    dist = {}
    if workers > 1:
        for d in _parallel_map(_mp_d_init, (NEO4J_URI, auth, NEO4J_DB, caches),
                               _mp_d_work, ccs, workers, t0):
            dist.update(d)
    else:
        searcher = GeoSotKNNSearcherV32(sess, geom_type="Point", verbose=False)
        dist = {cc: {} for cc in ccs}
        TOT = len(ORDER) * len(ccs); step = max(TOT // 100, 1); done = 0
        for cat in ORDER:
            set_poi_prefix_index(caches[cat])
            for cc in ccs:
                if done % step == 0:
                    _bar(done, TOT, t0)
                done += 1
                try:
                    res, _ = searcher.knn_search(cc, **_KNN_KW)
                    if res:
                        dist[cc][cat] = (round(res[0][1], 1),
                                         str(res[0][0]).strip().upper())
                except Exception:
                    pass
        sys.stdout.write(f"\r    [{'#'*30}] 100.0%  {TOT:,}/{TOT:,}\n")
        searcher.close()
    steps.add("g3r", "D", "空间桥·KNN 六类逐栋检索", _t() - t_knn,
              rows=len(ccs) * len(ORDER), note=f"workers={workers}")
    NT = len(targets)
    _write_acc(os.path.join(outdir, "g3r_service_accessibility.csv"),
               targets, dist)
    metrics["stages"]["D_knn_accessibility_ingraph"] = round(_t() - t0, 2)
    print(f"    完成 {NT:,}  {_t()-t0:.1f}s")

    metrics["stages"]["total_end_to_end"] = round(_t() - t_all, 2)
    metrics["result"] = {"resident_buildings": len(buildings),
                         "mismatch_parcels": len(mismatch),
                         "baseline_parcels": len(baseline),
                         "target_buildings": len(targets)}
    with open(os.path.join(outdir, "g3r_pipeline_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    sess.close(); drv.close()
    print(f"[G3R] 端到端 {metrics['stages']['total_end_to_end']:.1f}s "
          f"跨系统搬运 0 次\n")
    return metrics, used_codes


# ══════════════════════════════════════════════════════════════════════
# PostGIS pipeline (keyed by main_code)
# ══════════════════════════════════════════════════════════════════════
def run_pg(limit: int, threshold: int, outdir: str,
           restrict_codes: Optional[List[str]], dtarget: str = "city",
           steps: "StepLog" = None) -> Dict:
    import psycopg2
    from psycopg2.extras import execute_values

    steps = steps if steps is not None else StepLog()
    metrics = {"engine": "postgis", "threshold": threshold, "limit": limit,
               "dtarget": dtarget, "stages": {}, "cross_system_transfers": []}
    xfer = metrics["cross_system_transfers"]
    print("=" * W)
    print("[PG] PostGIS 外置基线流水线 (几何库 + oracle_mig 跨库居民)")
    print("=" * W)
    t_all = _t()
    m = psycopg2.connect(**PG_MAIN)
    g = psycopg2.connect(**PG_MIG)

    # If restricted set: map main_code → BLDG_NO
    restrict_bno = None
    code2bno, bno2code = {}, {}
    if restrict_codes:
        with m.cursor() as c:
            c.execute('SELECT main_code, "BLDG_NO" FROM public.building_new '
                      'WHERE main_code = ANY(%s)', (list(restrict_codes),))
            for mc, bno in c.fetchall():
                code2bno[str(mc).strip().upper()] = str(bno)
                bno2code[str(bno)] = str(mc).strip().upper()
        restrict_bno = list(bno2code.keys())
        print(f"  限定建筑集: {len(restrict_codes):,} main_code → "
              f"{len(restrict_bno):,} BLDG_NO (与 G3R 同批)")

    # A cross-DB pull demographics
    print("[A] 跨库拉居民人口结构 (oracle_mig → app) ...")
    t0 = _t()
    with g.cursor() as c:
        if restrict_bno is not None:
            c.execute("""
                SELECT h.buildingid, count(*),
                  count(*) FILTER (WHERE r.birthday >= %s),
                  count(*) FILTER (WHERE r.birthday IS NOT NULL
                                   AND r.birthday < %s)
                FROM oracle_src.house h
                JOIN oracle_src.resident r ON r.housecode = h.id
                WHERE h.buildingid = ANY(%s)
                GROUP BY h.buildingid
            """, (CHILD_BIRTH_MIN, ELDER_BIRTH_MAX, restrict_bno))
        else:
            res_from = ("oracle_src.resident" if limit <= 0
                        else f"(SELECT * FROM oracle_src.resident LIMIT {limit})")
            c.execute(f"""
                SELECT h.buildingid, count(*),
                  count(*) FILTER (WHERE r.birthday >= %s),
                  count(*) FILTER (WHERE r.birthday IS NOT NULL
                                   AND r.birthday < %s)
                FROM {res_from} r
                JOIN oracle_src.house h ON h.id = r.housecode
                GROUP BY h.buildingid
            """, (CHILD_BIRTH_MIN, ELDER_BIRTH_MAX))
        demo = {str(r[0]): (int(r[1]), int(r[2]), int(r[3]))
                for r in c.fetchall() if r[0]}
    metrics["stages"]["A_pull_residents_crossdb"] = round(_t() - t0, 2)
    steps.add("postgis", "A", "跨库聚合居民人口结构", _t() - t0,
              rows=len(demo), note="oracle_mig→app 跨库搬运1")
    xfer.append({"dir": "oracle_mig→app", "what": "居民人口结构聚合",
                 "rows": len(demo)})
    print(f"    有居民建筑 {len(demo):,}  {_t()-t0:.1f}s (跨库搬运1)")

    # B spatial join PIP (key: main_code)
    print("[B] PostGIS 空间连接 ST_Contains(质心∈地块) ...")
    t0 = _t()
    with m.cursor() as c:
        t_ins = _t()
        c.execute("DROP TABLE IF EXISTS tgt_bldg")
        c.execute("CREATE TEMP TABLE tgt_bldg(bno text PRIMARY KEY)")
        execute_values(c, "INSERT INTO tgt_bldg(bno) VALUES %s "
                          "ON CONFLICT DO NOTHING",
                       [(k,) for k in demo.keys()], page_size=10000)
        c.execute("ANALYZE tgt_bldg")
        steps.add("postgis", "B", "上传目标建筑集(app→PG 搬运)", _t() - t_ins,
                  rows=len(demo))
        # DISTINCT ON selects the smallest containing parcel for an overlapping building,
        # matching G3R and preventing duplicate rows from multiple containment.
        t_join = _t()
        c.execute("""
            SELECT DISTINCT ON (b.main_code)
                   b.main_code, b."BLDG_NO", p.main_code, p.lu_functio
            FROM tgt_bldg t
            JOIN public.building_new b ON b."BLDG_NO" = t.bno
            LEFT JOIN public.statutory_plan p
                   ON ST_Contains(p.geometry, ST_Centroid(b.geometry))
            ORDER BY b.main_code, ST_Area(p.geometry) ASC NULLS LAST
        """)
        pip = c.fetchall()
        steps.add("postgis", "B", "ST_Contains 空间连接(质心∈地块)", _t() - t_join,
                  rows=len(pip))
    m.commit()
    xfer.append({"dir": "app→postgis", "what": "有居民建筑 id", "rows": len(demo)})
    xfer.append({"dir": "postgis→app", "what": "建筑→地块归属", "rows": len(pip)})
    metrics["stages"]["B_pip_spatial_join"] = round(_t() - t0, 2)
    print(f"    归属 {len(pip):,}  {_t()-t0:.1f}s (往返2)")

    # C classify + filter (key: building main_code)
    t0 = _t()
    # Resolve land use through the normalized lu_function_dict, rather than application keyword rules.
    # Its lu_name -> land_class/land_subtype mapping mirrors G3R's IN_LU_CLASS traversal.
    t_lu = _t()
    lu_dict: Dict[str, Tuple[str, str]] = {}
    with m.cursor() as c:
        c.execute("SELECT lu_name, land_class, land_subtype "
                  "FROM public.lu_function_dict")
        for nm, cls_, sub_ in c.fetchall():
            lu_dict[nm] = (cls_, sub_)
    steps.add("postgis", "C", "用地类型字典解析(查 lu_function_dict)", _t() - t_lu,
              rows=len(lu_dict), note="规范化字典表, 非应用层关键词")
    t_agg = _t()
    parcel_lu_cls, parcel_agg, records = {}, {}, []
    for bmc, bno, parcel, lu in pip:
        cc = str(bmc).strip().upper()
        tot, ch, el = demo.get(str(bno), (0, 0, 0))
        if not parcel:
            pmc, cls, sub = "", "无所属地块", "无所属地块"
        else:
            pmc = str(parcel).strip().upper()
            if pmc not in parcel_lu_cls:
                # Split mixed land use on '+', matching G3R's collection of all LUFunctionGroup nodes.
                # G3R's LUFunctionGroup nodes are lu_functio values pre-split on '+'.
                parts = [p for p in normalize(lu).split("+") if p] if lu else []
                # Per-name dict lookup (classify fallback)
                pairs = [lu_dict.get(p) or classify(p) for p in parts]
                parcel_lu_cls[pmc] = _combine_classes(pairs)
            cls, sub = parcel_lu_cls[pmc]
        records.append((cc, pmc, cls, sub, tot, ch, el, 1 if parcel else 0))
        if pmc:
            pa = parcel_agg.setdefault(pmc, {"class": cls, "subtype": sub,
                 "res": 0, "child": 0, "elder": 0, "nb": 0})
            pa["res"] += tot; pa["child"] += ch; pa["elder"] += el; pa["nb"] += 1
    _write_attr(os.path.join(outdir, "pg_parcel_attribution.csv"), records)
    mismatch, baseline = _select_parcels(parcel_agg, threshold)
    _dump_parcels(os.path.join(outdir, "pg_mismatch_parcels.csv"), mismatch)
    _dump_parcels(os.path.join(outdir, "pg_baseline_parcels.csv"), baseline)
    steps.add("postgis", "C", "地块聚合+筛选错配/基准", _t() - t_agg,
              rows=len(mismatch) + len(baseline))
    metrics["stages"]["C_landuse_classify_select"] = round(_t() - t0, 2)
    print(f"[C] 错配 {len(mismatch):,}  基准 {len(baseline):,}")

    # D KNN — geography on EPSG:4490 (matches G3R ellipsoidal metres)
    print("[D] PostGIS KNN 六类设施最近距离 "
          f"(EPSG:{SOURCE_SRID} → geography + GIST) ...")
    t0 = _t()
    grp = {}
    for pmc in mismatch:
        grp[pmc] = "mismatch"
    for pmc in baseline:
        grp[pmc] = "baseline"
    # dtarget controls KNN coverage (same scope as G3R)
    targets = [(cc, pmc, grp.get(pmc, "other"), sub, tot, ch, el)
               for (cc, pmc, cls, sub, tot, ch, el, nh) in records]
    targets = _filter_targets(targets, dtarget)
    print(f"    目标建筑 {len(targets):,} ({DTARGET_LABEL[dtarget]}) "
          f"错配 {sum(1 for t in targets if t[2]=='mismatch'):,} "
          f"基准 {sum(1 for t in targets if t[2]=='baseline'):,}")
    t_pts = _t()
    with m.cursor() as c:
        c.execute("DROP TABLE IF EXISTS tgt_pts")
        c.execute("CREATE TEMP TABLE tgt_pts(cc text PRIMARY KEY, c geography)")
        execute_values(c, "INSERT INTO tgt_pts(cc) VALUES %s "
                          "ON CONFLICT DO NOTHING",
                       [(b[0],) for b in targets], page_size=10000)
        c.execute("""UPDATE tgt_pts t SET c = ST_Centroid(b.geometry)::geography
                      FROM public.building_new b WHERE b.main_code = t.cc""")
        c.execute("DELETE FROM tgt_pts WHERE c IS NULL")
        c.execute("CREATE INDEX ix_tgt_pts ON tgt_pts USING GIST(c)")
        c.execute("ANALYZE tgt_pts")
    m.commit()
    steps.add("postgis", "D",
              f"目标建筑质心 geography(EPSG:{SOURCE_SRID})+GIST(app→PG 搬运)",
              _t() - t_pts, rows=len(targets))
    xfer.append({"dir": "app→postgis", "what": "目标建筑集(KNN)",
                 "rows": len(targets)})
    dist = {b[0]: {} for b in targets}
    for cat in ORDER:
        tc = _t()
        with m.cursor() as c:
            # D1: poi_new stores leaf poi codes; join gaode_poi_category to resolve broader classes,
            # producing the same leaf-code sets as G3R's leaf-to-SUBCLASS_OF traversal.
            t_res = _t()
            c.execute(f"""SELECT ("poi编码")::double precision
                          FROM public.gaode_poi_category
                          WHERE {GAODE_CAT_SQL[cat]}""")
            leaf_codes = [r[0] for r in c.fetchall()]
            steps.add("postgis", "D", f"POI字典解析:{CAT_CN[cat]}", _t() - t_res,
                      rows=len(leaf_codes), note="join gaode_poi_category 展开叶子码")
            # D2: filter poi_new by leaf codes, then use geography meters on the same CGCS2000 ellipsoid as G3R.
            t_build = _t()
            c.execute("DROP TABLE IF EXISTS poi_cat")
            c.execute("""CREATE TEMP TABLE poi_cat AS
                SELECT main_code AS mc, geometry::geography AS g
                FROM public.poi_new
                WHERE poi = ANY(%s) AND geometry IS NOT NULL""", (leaf_codes,))
            c.execute("CREATE INDEX ix_poi_cat ON poi_cat USING GIST(g)")
            c.execute("ANALYZE poi_cat")
            c.execute("SELECT count(*) FROM poi_cat")
            npoi = c.fetchone()[0]
            steps.add("postgis", "D",
                      f"POI子集 geography+GIST:{CAT_CN[cat]}",
                      _t() - t_build, rows=npoi)
            # D3 KNN LATERAL: geography <-> is spherical; over-fetch 8 then
            # pick true nearest by ellipsoidal ST_Distance (same 8× rule as
            # the single-operator KNN baseline with K=1).
            t_knn = _t()
            c.execute("""SELECT t.cc, k.mc, k.d
              FROM tgt_pts t
              CROSS JOIN LATERAL (
                SELECT c.mc, c.d
                  FROM (
                    SELECT p.mc,
                           ST_Distance(t.c, p.g) AS d
                      FROM poi_cat p
                     ORDER BY t.c <-> p.g
                     LIMIT 8
                  ) c
                 ORDER BY c.d
                 LIMIT 1
              ) k""")
            for cc, mc, d in c.fetchall():
                if d is not None:
                    dist[cc][cat] = (round(float(d), 1),
                                     str(mc).strip().upper() if mc else "")
            steps.add("postgis", "D", f"KNN LATERAL:{CAT_CN[cat]}", _t() - t_knn,
                      rows=len(targets))
        print(f"    {cat:<10} POI {npoi:>8,}  KNN {_t()-tc:.1f}s")
    _write_acc(os.path.join(outdir, "pg_service_accessibility.csv"),
               targets, dist)
    xfer.append({"dir": "postgis→app", "what": "KNN 最近距离",
                 "rows": len(targets) * len(ORDER)})
    metrics["stages"]["D_knn_accessibility"] = round(_t() - t0, 2)

    metrics["stages"]["total_end_to_end"] = round(_t() - t_all, 2)
    metrics["result"] = {"resident_buildings": len(demo),
                         "mismatch_parcels": len(mismatch),
                         "baseline_parcels": len(baseline),
                         "target_buildings": len(targets)}
    with open(os.path.join(outdir, "pg_pipeline_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    m.close(); g.close()
    print(f"[PG] 端到端 {metrics['stages']['total_end_to_end']:.1f}s "
          f"跨系统搬运 {len(xfer)} 次\n")
    return metrics


# ══════════════════════════════════════════════════════════════════════
# Compare + diagnose
# ══════════════════════════════════════════════════════════════════════
def _read_json(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _read_parcels(p):
    s = set()
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            s.add(r["parcel_main_code"].strip().upper())
    return s


def _read_acc(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig"))) \
        if os.path.exists(p) else []


def _pct(vals, p):
    """Unweighted percentile (linear interpolation)."""
    xs = sorted(v for v in vals if v == v)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    idx = p / 100 * (len(xs) - 1)
    lo = int(idx); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)


def _wmedian(pairs):
    pairs = [(v, w) for v, w in pairs if w > 0]
    if not pairs:
        return float("nan")
    pairs.sort()
    tot = sum(w for _, w in pairs); acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2:
            return v
    return pairs[-1][0]


def _collect(rows, cat, group=None):
    """group=None → citywide (all rows); else filter by group."""
    out = []
    for r in rows:
        if group is not None and r.get("group") != group:
            continue
        d = r.get(f"d_{cat}", "")
        if d in ("", None):
            continue
        try:
            out.append((float(d), float(r.get("res_total") or 0)))
        except ValueError:
            pass
    return out


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def compare_and_diagnose(outdir, figdir, limit):
    gj = _read_json(os.path.join(outdir, "g3r_pipeline_metrics.json"))
    pj = _read_json(os.path.join(outdir, "pg_pipeline_metrics.json"))
    g_mm = _read_parcels(os.path.join(outdir, "g3r_mismatch_parcels.csv"))
    p_mm = _read_parcels(os.path.join(outdir, "pg_mismatch_parcels.csv"))
    g_bl = _read_parcels(os.path.join(outdir, "g3r_baseline_parcels.csv"))
    p_bl = _read_parcels(os.path.join(outdir, "pg_baseline_parcels.csv"))
    g_acc = _read_acc(os.path.join(outdir, "g3r_service_accessibility.csv"))
    p_acc = _read_acc(os.path.join(outdir, "pg_service_accessibility.csv"))

    print("=" * W)
    print("  对比 & 诊断   G3R (图内)  vs  PostGIS (外置)")
    print("=" * W)

    # (1) Parcel-set consistency
    print("\n[1] 结果一致性 — 错配/基准地块集合 (同 statutory_plan.main_code 键)")
    print(f"  错配: G3R {len(g_mm):,}  PG {len(p_mm):,}  交集 {len(g_mm & p_mm):,}"
          f"  Jaccard {_jaccard(g_mm, p_mm):.3f}")
    print(f"  基准: G3R {len(g_bl):,}  PG {len(p_bl):,}  交集 {len(g_bl & p_bl):,}"
          f"  Jaccard {_jaccard(g_bl, p_bl):.3f}")

    # (2a) G3R service-accessibility reference groups follow the Stage D scope:
    # city: mismatch/citywide/baseline; mb: mismatch/baseline; mismatch: mismatch only.
    dt = gj.get("dtarget", "city")
    print(f"\n[2a] 服务可达性 (G3R, 总人口加权中位最近距离 m)  "
          f"阶段D={DTARGET_LABEL.get(dt, dt)}")
    if dt == "city":
        print(f"  {'类别':<12} {'错配':>8} {'全市':>8} {'基准':>8} "
              f"{'错配/全市':>9} {'错配/基准':>9}")
        print("  " + "-" * 60)
        for cat in ORDER:
            mm = _wmedian(_collect(g_acc, cat, "mismatch"))
            cy = _wmedian(_collect(g_acc, cat))          # Citywide includes all rows.
            bl = _wmedian(_collect(g_acc, cat, "baseline"))
            r_cy = mm / cy if cy and cy == cy else float("nan")
            r_bl = mm / bl if bl and bl == bl else float("nan")
            print(f"  {CAT_CN[cat]:<12} {mm:>7.0f}m {cy:>7.0f}m {bl:>7.0f}m "
                  f"{r_cy:>9.2f} {r_bl:>9.2f}")
    elif dt == "mb":
        print(f"  {'类别':<12} {'错配':>8} {'基准':>8} {'错配/基准':>9}")
        print("  " + "-" * 42)
        for cat in ORDER:
            mm = _wmedian(_collect(g_acc, cat, "mismatch"))
            bl = _wmedian(_collect(g_acc, cat, "baseline"))
            r_bl = mm / bl if bl and bl == bl else float("nan")
            print(f"  {CAT_CN[cat]:<12} {mm:>7.0f}m {bl:>7.0f}m {r_bl:>9.2f}")
    else:                                                # Mismatch-only scope has one series.
        print(f"  {'类别':<12} {'错配':>8}")
        print("  " + "-" * 22)
        for cat in ORDER:
            mm = _wmedian(_collect(g_acc, cat, "mismatch"))
            print(f"  {CAT_CN[cat]:<12} {mm:>7.0f}m")
        print("  [注] 本次仅算错配, 无基准/全市参照; 需比值请用 --dtarget mb/city")

    # (2b) G3R/PostGIS agreement: six category medians within the current Stage D scope.
    print(f"\n[2b] 结果一致性 — 六类总人口加权中位 (G3R vs PG, "
          f"{DTARGET_LABEL.get(dt, dt)})")
    print(f"  {'类别':<12} {'G3R':>8} {'PG':>8} {'|Δ|':>7} {'相对差':>7}")
    print("  " + "-" * 46)
    acc_rows = []
    for cat in ORDER:
        gm = _wmedian(_collect(g_acc, cat)); pm = _wmedian(_collect(p_acc, cat))
        diff = abs(gm - pm) if gm == gm and pm == pm else float("nan")
        rel = diff / pm if pm and pm == pm else float("nan")
        print(f"  {CAT_CN[cat]:<12} {gm:>7.0f}m {pm:>7.0f}m {diff:>6.0f}m "
              f"{rel*100:>6.1f}%")
        acc_rows.append((cat, gm, pm))

    # (3) Per-building diagnostics aligned by main_code, using three paper metrics:
    # 1. Nearest-facility agreement: both methods select the same POI main_code.
    # 2. Distance fidelity among agreeing hits: MAE and median of signed G3R-PG errors.
    # 3. Absolute-error P95/P99, matching the accuracy/ECDF-tail measures in the spatial experiments.
    print("\n[3] 逐建筑诊断 — 命中一致率 + 距离保真 (仅同一POI子集) + P95/P99")
    gmap = {r["building_cc"]: r for r in g_acc}
    common = [(gmap[r["building_cc"]], r) for r in p_acc
              if r["building_cc"] in gmap]
    print(f"  可对齐建筑: {len(common):,} (G3R {len(g_acc):,} ∩ PG {len(p_acc):,})")
    if common:
        print(f"  {'类别':<12} {'n':>7} {'命中一致率':>9} {'同POI-MAE':>10} "
              f"{'同POI中位':>9} {'|Δ|P95':>8} {'|Δ|P99':>8}")
        print("  " + "-" * 66)
        allsame_abs = []
        for cat in ORDER:
            n = same = 0
            signed, absd = [], []          # Restrict to agreeing nearest-facility hits.
            for gr, pr in common:
                dg, dp = gr.get(f"d_{cat}", ""), pr.get(f"d_{cat}", "")
                if dg in ("", None) or dp in ("", None):
                    continue
                n += 1
                pg_poi = (gr.get(f"p_{cat}", "") or "").upper()
                pp_poi = (pr.get(f"p_{cat}", "") or "").upper()
                same_poi = pg_poi and pp_poi and pg_poi == pp_poi
                if same_poi:
                    same += 1
                    e = float(dg) - float(dp)
                    signed.append(e); absd.append(abs(e))
            if n:
                rate = same / n * 100
                mae = sum(absd) / len(absd) if absd else float("nan")
                med = median(absd) if absd else float("nan")
                p95 = _pct(absd, 95); p99 = _pct(absd, 99)
                allsame_abs += absd
                print(f"  {CAT_CN[cat]:<12} {n:>7,} {rate:>8.1f}% "
                      f"{mae:>9.2f}m {med:>8.2f}m {p95:>7.2f}m {p99:>7.2f}m")
        if allsame_abs:
            print("  " + "-" * 66)
            print(f"  {'六类合计(同POI)':<12} {len(allsame_abs):>7,} "
                  f"{'':>9} {sum(allsame_abs)/len(allsame_abs):>9.2f}m "
                  f"{median(allsame_abs):>8.2f}m {_pct(allsame_abs,95):>7.2f}m "
                  f"{_pct(allsame_abs,99):>7.2f}m")
        print("  说明: 命中一致=两侧选同一POI; 该子集 MAE/P95/P99 才是可与 §4.4 距离核\n"
              "        (亚厘米) / §4.5.3 KNN (MAE 0.026m) 对齐的精度; 命中不一致\n"
              "        多为近等距 POI 平局翻转(与几何/CRS微差有关), 非算错。")

    # (4) Architecture and cost.
    print("\n[4] 架构 / 成本")
    if gj and pj:
        gs, ps = gj["stages"], pj["stages"]
        rows = [("A 人口", "A_pull_residents_ingraph", "A_pull_residents_crossdb"),
                ("B 归属", "B_reverse_overlay_ingraph", "B_pip_spatial_join"),
                ("C 分类", "C_select", "C_landuse_classify_select"),
                ("D KNN", "D_knn_accessibility_ingraph", "D_knn_accessibility")]
        print(f"  {'阶段':<10} {'G3R(s)':>10} {'PG(s)':>10}")
        print("  " + "-" * 32)
        for lab, gk, pk in rows:
            print(f"  {lab:<10} {gs.get(gk, 0):>10.1f} {ps.get(pk, 0):>10.1f}")
        print("  " + "-" * 32)
        print(f"  {'端到端':<10} {gs.get('total_end_to_end', 0):>10.1f} "
              f"{ps.get('total_end_to_end', 0):>10.1f}")
        print(f"  {'跨系统搬运':<8} {len(gj['cross_system_transfers']):>10} "
              f"{len(pj['cross_system_transfers']):>10}")
        for x in pj["cross_system_transfers"]:
            print(f"      PG [{x['dir']:<14}] {x['what']:<18} rows={x['rows']:,}")

    # (4b) Detailed stage timings, including semantic hierarchy resolution.
    _print_substeps(os.path.join(outdir, "mismatch_step_timings.csv"))

    if limit > 0:
        print("\n  [注] --limit 冒烟: 两侧已强制同一批建筑, 结果/时间可比; "
              "全量请去掉 --limit。")

    try:
        _plots(figdir, gj, pj, acc_rows, common)
    except Exception as e:
        print(f"\n[!] 出图跳过: {e}")
    print("\n[完成]")


def _print_substeps(path):
    """Read fine-grained timing CSV; print by engine/stage."""
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        return
    print("\n[4b] 细粒度分步耗时 (s)  —— 逐小步, 语义层级解析可见")
    for eng, tag in (("g3r", "G3R (图内)"), ("postgis", "PostGIS (外置)")):
        er = [r for r in rows if r["engine"] == eng]
        if not er:
            continue
        tot = sum(float(r["sec"]) for r in er)
        print(f"  {tag}  合计 {tot:.2f}s")
        for r in er:
            note = f"  ({r['note']})" if r["note"] else ""
            rr = f"  rows={r['rows']}" if r["rows"] not in ("", None) else ""
            print(f"    [{r['stage']}] {r['substep']:<34} {float(r['sec']):>8.3f}s"
                  f"{rr}{note}")


def _plots(figdir, gj, pj, acc_rows, common):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei",
                         "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 9, "savefig.dpi": 300,
                         "savefig.bbox": "tight"})
    os.makedirs(figdir, exist_ok=True)
    if gj and pj:
        gs, ps = gj["stages"], pj["stages"]
        labels = ["A 人口", "B 归属", "C 分类", "D KNN"]
        gv = [gs.get("A_pull_residents_ingraph", 0),
              gs.get("B_reverse_overlay_ingraph", 0), gs.get("C_select", 0),
              gs.get("D_knn_accessibility_ingraph", 0)]
        pv = [ps.get("A_pull_residents_crossdb", 0),
              ps.get("B_pip_spatial_join", 0),
              ps.get("C_landuse_classify_select", 0),
              ps.get("D_knn_accessibility", 0)]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
        x = np.arange(len(labels)); w = 0.38
        a1.bar(x - w/2, gv, w, color=G3R_COLOR, label="G3R (图内)")
        a1.bar(x + w/2, pv, w, color=PG_COLOR, label="PostGIS (外置)")
        a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_ylabel("耗时 (s)")
        a1.set_title("分阶段耗时", fontweight="bold"); a1.legend()
        a1.grid(axis="y", alpha=0.3)
        e2e = [gs.get("total_end_to_end", 0), ps.get("total_end_to_end", 0)]
        xf = [len(gj["cross_system_transfers"]),
              len(pj["cross_system_transfers"])]
        bars = a2.bar(["G3R", "PostGIS"], e2e, color=[G3R_COLOR, PG_COLOR],
                      width=0.55)
        a2.set_ylabel("端到端 (s)")
        a2.set_title("端到端 + 跨系统搬运", fontweight="bold")
        for b, x0 in zip(bars, xf):
            a2.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{b.get_height():.0f}s\n搬运{x0}次", ha="center",
                    va="bottom", fontsize=8)
        a2.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p1 = os.path.join(figdir, "pipeline_compare_cost.png")
        fig.savefig(p1); plt.close(fig); print(f"\n  [图1] {p1}")

    if acc_rows:
        cns = [CAT_CN[r[0]] for r in acc_rows]
        gm = [r[1] if r[1] == r[1] else 0 for r in acc_rows]
        pm = [r[2] if r[2] == r[2] else 0 for r in acc_rows]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        x = np.arange(len(cns)); w = 0.38
        ax.bar(x - w/2, gm, w, color=G3R_COLOR, label="G3R")
        ax.bar(x + w/2, pm, w, color=PG_COLOR, label="PostGIS")
        ax.set_xticks(x); ax.set_xticklabels(cns, rotation=20, ha="right")
        ax.set_ylabel("人口加权中位最近距离 (m)")
        ax.set_title("六类设施可达性一致性", fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        p2 = os.path.join(figdir, "pipeline_compare_accessibility.png")
        fig.savefig(p2); plt.close(fig); print(f"  [图2] {p2}")

    # Per-building scatter (diagnosis)
    if common:
        import numpy as np
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
        for i, cat in enumerate(ORDER):
            ax = axes[i // 3][i % 3]
            xs, ys = [], []
            for gr, pr in common:
                dg, dp = gr.get(f"d_{cat}", ""), pr.get(f"d_{cat}", "")
                if dg in ("", None) or dp in ("", None):
                    continue
                xs.append(float(dp)); ys.append(float(dg))
            if xs:
                ax.scatter(xs, ys, s=3, alpha=0.25, color=G3R_COLOR,
                           edgecolors="none")
                mx = max(max(xs), max(ys))
                ax.plot([0, mx], [0, mx], "--", color="#888", lw=0.8)
            ax.set_title(CAT_CN[cat], fontsize=9)
            ax.set_xlabel("PostGIS (m)", fontsize=7)
            ax.set_ylabel("G3R (m)", fontsize=7)
        fig.suptitle("逐建筑最近距离对齐 (y=x 即完全一致)", fontweight="bold")
        fig.tight_layout()
        p3 = os.path.join(figdir, "pipeline_diagnose_scatter.png")
        fig.savefig(p3); plt.close(fig); print(f"  [图3] {p3}")


# ══════════════════════════════════════════════════════════════════════
def _prewarm():
    """
    Warm the relevant PG tables and Neo4j label/relationship data before steady-state pipeline
    measurement, following the spatial benchmark protocol. PG uses pg_prewarm; Neo4j scans
    relevant labels in pages. Exclude warmup time from pipeline timing and report it separately;
    this is cache preparation, not a guarantee of permanently pinned pages.
    """
    import psycopg2
    print("=" * W)
    print("[预热] 对称载入热数据 (PG pg_prewarm + Neo4j 分页扫描) ...")
    t0 = _t()
    for tag, conn, tabs in (
        ("PG几何库", PG_MAIN, ("public.building_new", "public.poi_new",
                               "public.statutory_plan")),
        ("PG居民库", PG_MIG, ("oracle_src.resident", "oracle_src.house"))):
        try:
            cn = psycopg2.connect(**conn)
            with cn.cursor() as c:
                c.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
                warmed = 0
                for t in tabs:
                    try:
                        c.execute("SELECT pg_prewarm(%s)", (t,))
                        warmed += int(c.fetchone()[0] or 0)
                    except Exception as e:
                        print(f"    [{tag}] {t} 跳过: {e}")
            cn.commit(); cn.close()
            print(f"    [{tag}] 预热 {warmed:,} 页")
        except Exception as e:
            print(f"    [{tag}] 连接失败, 跳过: {e}")
    try:
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
        with drv.session(database=NEO4J_DB) as s:
            for lbl in ("Resident", "House", "Building", "GeoSOTCode"):
                s.run(f"MATCH (n:{lbl}) RETURN count(n)").consume()
            print("    [Neo4j] 全标签扫描分页完成")
            # Relationship-store warmup: label counts warm node pages but Stage A also traverses
            # HAS_GEOSOT / LOCATED_IN / LIVE_IN / HAS_COUNTY_ORIGIN /
            # IN_PROVINCE/IN_CITY relationship pages. Cold first access caused the historical
            # first-run spike. Mirror the Stage A traversal once to warm these relationship pages
            # before measuring the pipeline, bringing the first measured run closer to steady state.
            s.run("MATCH (:Building)-[:HAS_GEOSOT]->(:GeoSOTCode) "
                  "RETURN count(*)").consume()
            s.run(
                "MATCH (b:Building)<-[:LOCATED_IN]-(:House)"
                "<-[:LIVE_IN]-(r:Resident)\n"
                "OPTIONAL MATCH (r)-[:HAS_COUNTY_ORIGIN]->(co:CountyOrigin)\n"
                "OPTIONAL MATCH (co)-[:IN_PROVINCE]->(:Province)\n"
                "OPTIONAL MATCH (co)-[:IN_CITY]->(:AdminCity)\n"
                "RETURN count(r)").consume()
            print("    [Neo4j] 关系存储遍历预热完成 (镜像阶段A, 消首跑尖峰)")
        drv.close()
    except Exception as e:
        print(f"    [Neo4j] 跳过: {e}")
    print(f"    预热完成 {_t()-t0:.1f}s (不计入流水线计时)")
    print("=" * W)


def _med(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _bench_summary(runs, engine):
    """Per-stage min/median/max over repeats; median is the formal figure."""
    keys, seen = [], set()
    for r in runs:                              # Collect all stage keys with order-preserving deduplication.
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    print(f"\n  [{engine}]  (n={len(runs)})    {'min':>9} {'中位':>9} {'max':>9}"
          f" {'极差%':>7}")
    print("  " + "-" * 62)
    for k in keys:
        xs = [r[k] for r in runs if k in r]
        if not xs:
            continue
        md = _med(xs)
        spread = (max(xs) - min(xs)) / md * 100 if md else 0.0
        print(f"  {k:<28} {min(xs):>9.1f} {md:>9.1f} {max(xs):>9.1f} "
              f"{spread:>6.0f}%")


def main():
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print('Usage: python run_mismatch_experiment.py [options]\n\nPublic-service proximity under land-use mismatch (G3R vs PostGIS).\nRequires prepared databases. This help command does not connect to them.\n\n  -h, --help          Show this message and exit\n  --limit N           Align both methods to a sample of N buildings; 0=full\n  --threshold N       Resident-count threshold (default 100)\n  --outdir DIR        CSV/metric output directory (default csv)\n  --figdir DIR        Figure output directory (default paper_figures)\n  --only STAGE        g3r, pg, run, compare, or diag\n  --dtarget SCOPE     city, mb, or mismatch (default city)\n  --workers N         Worker processes; 1=serial\n  --repeat N          Number of measured repeats (default 1)\n  --prewarm           Warm relevant data before measured runs\n  --skip-g3r          Reuse existing G3R outputs\n  --skip-pg           Reuse existing PostGIS outputs\n')
        return
    limit, threshold, outdir, figdir = 0, 100, "csv", "paper_figures"
    only, skip_g3r, skip_pg, prewarm = "", False, False, False
    dtarget = "city"                        # Stage D scope: city/mb/mismatch.
    repeat = 1                              # For repeat counts above 1, report stage minimum/median/maximum.
    workers = min(os.cpu_count() or 1, 16)  # Default workers = min(logical CPU count, 16); 1 selects serial execution.
                                            # Shared cache state avoids repeated worker-load contention.
    a = sys.argv[1:]; i = 0
    while i < len(a):
        if a[i] == "--limit" and i + 1 < len(a):
            limit = int(a[i + 1]); i += 2
        elif a[i] == "--threshold" and i + 1 < len(a):
            threshold = int(a[i + 1]); i += 2
        elif a[i] == "--outdir" and i + 1 < len(a):
            outdir = a[i + 1]; i += 2
        elif a[i] == "--figdir" and i + 1 < len(a):
            figdir = a[i + 1]; i += 2
        elif a[i] == "--only" and i + 1 < len(a):
            only = a[i + 1]; i += 2
        elif a[i] == "--dtarget" and i + 1 < len(a):
            dtarget = a[i + 1]; i += 2      # city | mb | mismatch
        elif a[i] == "--repeat" and i + 1 < len(a):
            repeat = max(int(a[i + 1]), 1); i += 2
        elif a[i] == "--workers" and i + 1 < len(a):
            workers = max(int(a[i + 1]), 1); i += 2
        elif a[i] == "--prewarm":
            prewarm = True; i += 1
        elif a[i] == "--skip-g3r":
            skip_g3r = True; i += 1
        elif a[i] == "--skip-pg":
            skip_pg = True; i += 1
        else:
            i += 1
    if dtarget not in DTARGET_LABEL:
        print(f"[警告] --dtarget {dtarget} 无效, 回退 city (可选 city/mb/mismatch)")
        dtarget = "city"
    os.makedirs(outdir, exist_ok=True)

    print("#" * W)
    print('#  Land-use mismatch workflow (G3R vs PostGIS)')
    print(f"#  Resident-count threshold >={threshold}   limit={limit or '全量'}   outdir={outdir}   workers={workers}   阶段D={DTARGET_LABEL[dtarget]}")
    if limit > 0:
        print('#  Smoke mode: both sides use the same main_code-aligned buildings')
    if dtarget != "city":
        print(f'#  Stage D scope: {DTARGET_LABEL[dtarget]}」建筑 (A/B/C 仍全量; 提速来自 D 目标缩小)')
    print("#" * W)

    if prewarm and only in ("", "g3r", "pg", "run"):
        _prewarm()

    g_runs, p_runs = [], []                # Per-run stage times for --repeat summaries.
    for it in range(1, repeat + 1):
        if repeat > 1:
            print("\n" + "#" * W)
            print(f"#  REPEAT {it}/{repeat}")
            print("#" * W)
        shared_codes = None
        steps = StepLog()                  # Detailed substeps include semantic hierarchy resolution.
        if only in ("", "g3r", "run") and not skip_g3r:
            _gm, used = run_g3r(limit, threshold, outdir, workers=workers,
                                dtarget=dtarget, steps=steps)
            g_runs.append(_gm["stages"])
            if limit > 0:
                shared_codes = used        # For a smoke run, restrict PG to the same buildings selected by G3R.
        if only in ("", "pg", "run") and not skip_pg:
            _pm = run_pg(limit, threshold, outdir, restrict_codes=shared_codes,
                         dtarget=dtarget, steps=steps)
            p_runs.append(_pm["stages"])
        if steps.rows:
            sp = os.path.join(outdir, "mismatch_step_timings.csv")
            steps.write(sp)
            print(f"\n[细粒度耗时] {sp}  ({len(steps.rows)} 子步)")
    if only in ("", "compare", "diag", "run"):
        compare_and_diagnose(outdir, figdir, limit)   # Use the final run's CSV.

    if repeat > 1:
        print("\n" + "=" * W)
        print(f"  多次基准汇总 (N={repeat}, 预热{'开' if prewarm else '关'}, "
              f"workers={workers}, 阶段D={DTARGET_LABEL[dtarget]})")
        print("  取数建议: 每阶段用「中位」列; 极差% 大者受本机负载影响, 需减背景干扰")
        print("=" * W)
        if g_runs:
            _bench_summary(g_runs, "G3R")
        if p_runs:
            _bench_summary(p_runs, "PostGIS")


if __name__ == "__main__":
    main()

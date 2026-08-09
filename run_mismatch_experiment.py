#!/usr/bin/env python3
# coding: utf-8
"""
错配复合查询 —— 单脚本一键实验 (G3R 图内统一执行  vs  PostGIS 外置基线)。
Mismatch composite query — one-script end-to-end experiment (G3R in-graph unified
execution vs PostGIS external baseline).

一条命令跑完全流程 / One command runs the full flow:
  1) G3R  流水线 (Neo4j + GeoSOT 编码, 图内统一执行)
     G3R pipeline (Neo4j + GeoSOT codes, unified in-graph execution)
  2) PostGIS 流水线 (完全镜像的五阶段, 外置几何库 + 跨库居民)
     PostGIS pipeline (fully mirrored five stages, external geometry + cross-DB residents)
  3) 对比: 结果一致性 (地块集合 + 可达性) + 架构/成本 (计时 + 跨系统搬运)
     Compare: result consistency (parcel sets + accessibility) + architecture/cost
  4) 诊断: 逐建筑 main_code 对齐, 六类最近距离逐栋比对 (证明空间桥算的是同一设施)
     Diagnose: per-building main_code alignment; six-category nearest-distance check

五阶段镜像 / Five-stage mirror:
  A 拉居民人口结构   G3R 图内语义链(跨库0)   | PG oracle_mig 跨库(搬运1)
     Pull resident demographics | G3R in-graph (0 xfers) | PG cross-DB (xfer 1)
  B 建筑质心→地块    G3R 空间桥·反向叠置     | PG ST_Contains 空间连接(搬运2)
     Building centroid→parcel | G3R reverse overlay | PG ST_Contains (xfer 2)
  C 用地分类+筛选    landuse_norm (两侧同一分类器, 公平)
     Land-use classify+filter | same classifier both sides (fair)
  D 六类设施 KNN     G3R 空间桥·KNN(复用CC)  | PG 投影子集+GIST LATERAL(搬运2)
     Six-category facility KNN | G3R spatial-bridge KNN | PG GIST LATERAL (xfer 2)

--limit 语义 (关键): 冒烟时**两侧强制用同一批建筑** (按 main_code 对齐) → 时间/结果
直接可比; 全量 (不加 --limit) 时两侧各自独立跑 → 保留"两套独立实现互证"说服力。
--limit semantics (critical): smoke tests force **the same building batch** on both
sides (aligned by main_code) so time/results are comparable; full runs (no --limit)
keep independent implementations for mutual validation.

用法 (PyCharm 建议) / Usage (PyCharm-friendly):
  python run_mismatch_experiment.py                  # 全量, 跑全部 / full run
  python run_mismatch_experiment.py --limit 8000     # 冒烟, 两侧同 8000 栋 / smoke, same 8000
  python run_mismatch_experiment.py --threshold 100
  python run_mismatch_experiment.py --skip-g3r       # 复用已有 g3r_*.csv, 只跑 PG+对比
  python run_mismatch_experiment.py --only compare    # 只读已有产物做对比+诊断
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
from geosot_overlay_query_v7 import GRID_LEVEL as GL, _code_to_xy
from service_categories import load_category_codes
from geosot_knn_v3 import POIPrefixCache, set_poi_prefix_index
from geosot_knn_v3_1 import GeoSotKNNSearcherV31

# ── 配置 / Config ────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DB", "geocode")
PG_MAIN = {"host": "127.0.0.1", "port": 5432, "dbname": "postgis_35_sample",
           "user": "postgres", "password": os.getenv("PGPASSWORD", "")}
PG_MIG = {**PG_MAIN, "dbname": "oracle_mig"}
# 全部分析几何的存储/语义 CRS：CGCS2000 经纬度（与 G3R / GeoSOT 编码一致）。
# Storage/semantic CRS for all analysis geometry: CGCS2000 lon/lat (same as G3R/GeoSOT).
# PostGIS 距离与 KNN 基线在 geometry::geography 上算椭球米，不再投影到局部 GK。
# PostGIS distance/KNN baseline uses ellipsoidal metres on geometry::geography, not local GK.
SOURCE_SRID = 4490
# 仅用于米制网格聚类 / 出图凸包等非分析中间坐标；分析路径禁止依赖此 SRID。
# Metric-grid clustering / plot hulls only; analysis path must not depend on this SRID.
CLUSTER_METRIC_SRID = 4547  # CGCS2000 / 3° GK CM 114E (深圳 / Shenzhen)
# 兼容旧 import 名（勿再用于 ST_Distance / KNN 基线） / Legacy import alias (do not use for ST_Distance/KNN)
METRIC_SRID = CLUSTER_METRIC_SRID
PARCEL_SRC = "Statutory Plan"
CHILD_BIRTH_MIN = "2001-01-01"
ELDER_BIRTH_MAX = "1956-01-01"

# 六类服务 POI (旧: 直接用 poi_new 冗余列 f1/f2; 保留仅作参照, 现流程已弃用)
# Six service POI categories (legacy f1/f2 columns; kept for reference, unused now)
PG_CAT_SQL = {
    "edu":       "f2 = '学校'",
    "medical":   "f2 IN ('综合医院','专科医院','诊所')",
    "sport":     "f1 = '体育休闲服务'",
    "scenic":    "f1 = '风景名胜'",
    "public":    "f1 = '公共设施'",
    "transport": "f1 = '交通设施服务' AND (f2 IS NULL OR f2 <> '停车场')",
}

# 规范化层级解析 (V0.9, 与 G3R 叶子→SUBCLASS_OF 对称):
# Canonical hierarchy resolution (V0.9, symmetric to G3R leaf→SUBCLASS_OF):
# 假设 poi_new 只保留叶子码 poi (相当于 f3), 粗类必须 join 分类字典
# Assume poi_new keeps only leaf code poi (=f3); coarse class must join the category dict
# gaode_poi_category(poi编码/大类/中类) 展开。下列条件作用在**字典表**上,
# Expand via gaode_poi_category (poi code / L1 / L2). Predicates apply on the **dict table**;
# 先解析出目标类的叶子码集合, 再据此过滤 poi_new。已验证六类计数与 f1/f2 逐类 Δ=0。
# Resolve target leaf-code sets first, then filter poi_new. Six-cat counts match f1/f2 (Δ=0).
GAODE_CAT_SQL = {
    "edu":       "中类 = '学校'",
    "medical":   "中类 IN ('综合医院','专科医院','诊所')",
    "sport":     "大类 = '体育休闲服务'",
    "scenic":    "大类 = '风景名胜'",
    "public":    "大类 = '公共设施'",
    "transport": "大类 = '交通设施服务' AND 中类 <> '停车场'",
}
ORDER = ["edu", "medical", "sport", "scenic", "public", "transport"]
CAT_CN = {"edu": "教育-学校 / Education-School", "medical": "医疗 / Medical",
          "sport": "体育休闲 / Sports-Leisure",
          "scenic": "风景名胜 / Scenic", "public": "公共设施 / Public",
          "transport": "交通设施 / Transport"}

_t = time.perf_counter
W = 78
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

G3R_COLOR = "#2166AC"
PG_COLOR = "#B2182B"


# ── 通用 / Commons ───────────────────────────────────────────────────
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
    """整数 Z-order 坐标下的鞋带公式面积 (相对比较用, 无需真实米制)。 / Shoelace area in integer Z-order coords (relative compare; no real metres)."""
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
    """由地块内各用地功能的 (class, subtype) 组合出地块级类别 (混合逻辑)。
    Combine per-parcel (class, subtype) into a parcel-level category (mixed logic).
    此为**地块级聚合**, 与"逐名语义解析"解耦 → 两侧(G3R 走图层级 / PG join 字典)
    共用同一组合规则, 保证一致。
    Parcel-level aggregation, decoupled from per-name parsing; both sides share the same rule."""
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
    """逐名用 landuse_norm.classify 解析后组合 (兜底路径; 规范化流程改走字典/图层级)。 / Per-name classify then combine (fallback; canonical path uses dict/graph hierarchy)."""
    return _combine_classes([classify(nm) for nm in lus if nm])


class StepLog:
    """细粒度分步耗时记录: 每个子步一行, 落 CSV, 让"语义层级解析"等小步耗时可见。 / Fine-grained step timing: one CSV row per substep (e.g. semantic hierarchy resolve)."""
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


# 阶段D 覆盖面: city=全市(默认,三线) | mb=错配+基准(自洽双线) | mismatch=仅错配
# Stage-D coverage: city=citywide (default, 3 lines) | mb=mismatch+baseline | mismatch=only
DTARGET_LABEL = {"city": "全市 / citywide", "mb": "错配+基准 / mismatch+baseline",
                 "mismatch": "仅错配 / mismatch-only"}


def _filter_targets(targets, dtarget):
    """按 dtarget 收窄阶段D 目标集 (targets[i][2] 为 group 标签)。
    Narrow Stage-D targets by dtarget (targets[i][2] is the group label).
    A/B/C 仍全量(错配判定需全量归属+人口), 只有 D 逐栋 KNN 随目标缩小而提速。
    A/B/C stay full-scale; only per-building KNN in D shrinks with the target set."""
    if dtarget == "mb":
        return [t for t in targets if t[2] in ("mismatch", "baseline")]
    if dtarget == "mismatch":
        return [t for t in targets if t[2] == "mismatch"]
    return targets                                     # city: 全量 / citywide full set


def _write_attr(path, records):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["building_cc", "parcel_main_code", "land_class",
                     "land_subtype", "res_total", "res_child",
                     "res_elderly", "n_parcel_hits"])
        wr.writerows(records)


def _write_acc(path, targets, dist):
    """dist[cc][cat] = (距离m, 最近POI main_code); 兼容旧标量(只距离)。 / (distance_m, nearest POI main_code); legacy scalar distance still accepted."""
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
    """只含某一类 POI 主码的子集缓存 (供 KNN 按类检索)。 / Subset cache of one POI category main_codes (for per-class KNN)."""
    def __init__(self, codes):
        self._sorted, self._centers, self._code_idx = {}, {}, {}
        codes = sorted({c.strip().upper() for c in codes if c})
        self._sorted["Point"] = codes
        self._precompute_centers("Point", codes, verbose=False)
        self.elapsed_sec = 0.0


# 召回强化档 (子集缓存小, 深搜零额外耗时): min_margin_m 是主控旋钮 —
# Recall boost (small subset cache; deep search has no extra cost): min_margin_m is the knob —
# 它是 KNN 的"保证正确半径", 调大迫使覆盖判据搜更大范围, 消除稀疏区
# It is KNN's "guaranteed-correct radius"; larger forces wider coverage search in sparse areas
# (公共/风景/教育) 漏检真最近的问题。经 _tmp_recall_probe 扫参: 漏检
# (public/scenic/edu) true-nearest misses. After _tmp_recall_probe sweep: miss rate
# 由 ~0.6% 降到 ~0.05%, ms/查询不变。
# dropped ~0.6% → ~0.05%, ms/query unchanged.
_KNN_KW = dict(k=1, min_margin_m=300.0, min_level=8, coarse_top_n=50,
               geometry_scoring=False, max_widen=11,
               max_candidates_per_level=500000, start_level_max=0)


def _attr_one(sess, pc, parcel_cls, src, cc, res, ch, el):
    """单栋建筑反向叠置归属地块; nh=-1 表示编码解码失败(主程序丢弃)。 / Reverse-overlay one building to its parcel; nh=-1 means decode failure (drop)."""
    try:
        xy = _code_to_xy(cc, GL)
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
    """精简版反向叠置 (行为与 _attr_one 逐位一致, 但零对象开销):
    Lean reverse overlay (bit-identical to _attr_one, zero object overhead):
    - 只扫索引里"真实存在的前缀长度"(plens 降序), 跳过全部空 tier 的字符串切片
      + dict 查询 (原级联 tier_size=1 从 len 到 2 逐长度扫, 绝大多数为空);
      Only scan prefix lengths that exist (plens desc); skip empty-tier slices/lookups.
    - 直接命中 cp_map/get_bbox/get_polygon, 不建 row-dict/seen_mc/候选 list;
      Hit cp_map/get_bbox/get_polygon directly; no row-dict/seen_mc/candidate lists.
    - 语义严格保持: 最细前缀 tier 优先(降序首个有 PIP 命中的 tier 即停), tier 内
      多命中取最小面积 —— 与级联 tier_size=1 完全等价。
      Finest prefix tier first; min area on multi-hit — equivalent to cascade tier_size=1."""
    try:
        xy = _code_to_xy(cc, GL)
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
    """由归属 records 聚合 parcel_agg。 / Aggregate parcel_agg from attribution records."""
    agg = {}
    for (cc, pmc, cls, sub, res, ch, el, nh) in records:
        if pmc:
            pa = agg.setdefault(pmc, {"class": cls, "subtype": sub,
                 "res": 0, "child": 0, "elder": 0, "nb": 0})
            pa["res"] += res; pa["child"] += ch; pa["elder"] += el; pa["nb"] += 1
    return agg


# ── 多进程 worker (阶段B/D 按建筑切分; 每 worker 进程各建只读缓存一次) ──
# Multiprocess workers (stages B/D split by building; each worker builds RO cache once)
_MP: Dict = {}


def _mp_b_init(uri, auth, db, parcel_src, pc, cp_map, plens, parcel_cls):
    """共享缓存: pc(多边形)/cp_map(前缀→地块)/plens(存在的前缀长度降序)/
    parcel_cls 由主进程建好后 pickle 传入。精简路径不查 Neo4j, 无需 session。
    Shared cache pickled from parent; lean path needs no Neo4j session."""
    _MP.update(pc=pc, cp_map=cp_map, plens=plens, parcel_cls=parcel_cls,
               src=parcel_src)


def _mp_b_work(chunk):
    pc, cp_map, plens, cls = (_MP["pc"], _MP["cp_map"], _MP["plens"],
                              _MP["parcel_cls"])
    return [_attr_one_lean(cc, res, ch, el, pc, cp_map, plens, cls)
            for (cc, res, ch, el) in chunk]


def _mp_d_init(uri, auth, db, caches):
    """共享缓存: 六类 SubsetPOICache 由主进程建好后 pickle 传入; searcher
    构造极轻(仅存 session, 无预载), 各 worker 各建一个即可。
    Shared six-category SubsetPOICache from parent; each worker builds a light searcher."""
    drv = GraphDatabase.driver(uri, auth=tuple(auth))
    sess = drv.session(database=db)
    _MP.update(sess=sess, caches=caches,
               searcher=GeoSotKNNSearcherV31(sess, geom_type="Point",
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
    """按 worker*8 分块并行, 保序返回各块结果, 带进度条。 / Chunk by workers*8, parallel map, preserve order, with progress bar."""
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
# G3R 流水线 / G3R pipeline
# ══════════════════════════════════════════════════════════════════════
def run_g3r(limit: int, threshold: int, outdir: str,
            workers: int = 1, dtarget: str = "city",
            steps: "StepLog" = None) -> Tuple[Dict, List[str]]:
    steps = steps if steps is not None else StepLog()
    metrics = {"engine": "g3r", "threshold": threshold, "limit": limit,
               "dtarget": dtarget, "stages": {}, "cross_system_transfers": []}
    print("=" * W)
    print("[G3R] 图内统一执行流水线 (Neo4j + GeoSOT, 语义/几何/人口同图) / In-graph unified pipeline (semantics/geometry/population)")
    print("=" * W)
    t_all = _t()
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    sess = drv.session(database=NEO4J_DB)

    # A 图内拉人口结构 / A in-graph pull demographics
    print("[A] 图内拉有居民建筑 + 人口结构 ... / Pull resident buildings + demographics ...")
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
    steps.add("g3r", "A", "图内语义链聚合人口结构 / in-graph semantic-chain demographics", _t() - t0,
              rows=len(buildings), note="Building←House←Resident 单趟, 跨库0 / single pass, 0 xfers")
    print(f"    有居民建筑 / resident bldgs {len(buildings):,}  {_t()-t0:.1f}s (跨库0 / 0 xfers)")
    used_codes = [b[0] for b in buildings]

    # B 空间桥·反向叠置 (tier_size=1; 多重包含取最小面积地块, 与 PG 同规则)
# B spatial-bridge reverse overlay (tier_size=1; min-area on multi-contain; same as PG)
    print(f"[B] 空间桥·反向叠置 (CC 前缀+PIP) 归属地块 ... / reverse overlay attribute parcels ... (workers={workers})")
    t0 = _t()
    auth = [NEO4J_USER, NEO4J_PWD]
    # 主进程建一次缓存 (串行直接用; 并行 pickle 共享给各 worker → 避免 N 次
    # Parent builds cache once (serial use; pickle-share to workers → avoid N
    # 并发 Neo4j 重载, 消除 worker 数越多建缓存越慢的争用)
    # concurrent Neo4j reloads; more workers no longer slow cache build via contention).
    # 用地类型: 走图层级 (LUFunctionGroup)-[:IN_LU_CLASS]->(LUClass) 推理地块类别,
    # Land-use: infer parcel class via graph hierarchy (LUFunctionGroup)-[:IN_LU_CLASS]->(LUClass),
    # 而非在应用层跑关键词分类器 → 与 PG 端 join lu_function_dict 对称。
    # not an app-layer keyword classifier → symmetric to PG join lu_function_dict.
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
    steps.add("g3r", "B", "用地类型层级解析(IN_LU_CLASS 图内) / LU hierarchy resolve", _t() - t_lu,
              rows=len(parcel_cls), note="图层级推理, 非应用层关键词 / graph infer, not keywords")
    t_poly = _t()
    pc = PlanPolygonCache(sess, verbose=False); pc.preload(PARCEL_SRC)
    steps.add("g3r", "B", "地块多边形缓存预载 / parcel polygon cache preload", _t() - t_poly, rows=pc.count)
    t_cp = _t()
    cp = CommonPrefixIndex(sess, parcel_source=PARCEL_SRC, verbose=False)
    steps.add("g3r", "B", "GeoSOT 前缀倒排索引 / GeoSOT prefix inverted index", _t() - t_cp)
    # 精简路径: 直接用前缀倒排 dict + "真实存在的前缀长度"降序表, 跳过空 tier
# Lean path: prefix inverted dict + existing prefix lengths (desc); skip empty tiers
    cp_map = cp._index
    plens = sorted({len(p) for p in cp_map}, reverse=True)
    print(f"    缓存就绪 / cache ready: {pc.count:,} 地块多边形/parcels  {len(plens)} 前缀长度/prefix lens  "
          f"{_t()-t0:.1f}s")
    t_ov = _t()
    raw = []
    if workers > 1:
        pc._session = None                       # 置空会话引用以便 pickle / drop session ref for pickle
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
    steps.add("g3r", "B", "空间桥·反向叠置归属(CC前缀+PIP) / reverse-overlay attribute", _t() - t_ov,
              rows=len(raw), note=f"workers={workers}")
    derr = sum(1 for r in raw if r[7] == -1)
    records = [r for r in raw if r[7] != -1]
    parcel_agg = _agg_parcels(records)
    _write_attr(os.path.join(outdir, "g3r_parcel_attribution.csv"), records)
    metrics["stages"]["B_reverse_overlay_ingraph"] = round(_t() - t0, 2)
    print(f"    归属 / attributed {len(records):,} 解码失败 / decode-fail {derr}  {_t()-t0:.1f}s")

    # C 筛选 / C filter
    t0 = _t()
    mismatch, baseline = _select_parcels(parcel_agg, threshold)
    _dump_parcels(os.path.join(outdir, "g3r_mismatch_parcels.csv"), mismatch)
    _dump_parcels(os.path.join(outdir, "g3r_baseline_parcels.csv"), baseline)
    metrics["stages"]["C_select"] = round(_t() - t0, 2)
    steps.add("g3r", "C", "筛选错配/基准地块(人口阈值) / filter mismatch/baseline parcels", _t() - t0,
              rows=len(mismatch) + len(baseline))
    print(f"[C] 错配 / mismatch {len(mismatch):,}  基准 / baseline {len(baseline):,}")

    # D 空间桥·KNN (dtarget 控制覆盖面: city 全市 / mb 错配+基准 / mismatch 仅错配)
# D spatial-bridge KNN (dtarget controls coverage)
    print(f"[D] 空间桥·KNN 六类设施最近距离 / six-category nearest dist ({DTARGET_LABEL[dtarget]}) ...")
    t0 = _t()
    # POI 类树解析: 叶子(HAS_POI_TYPE_3)→SUBCLASS_OF 上溯到目标大/中类 (逐类计时)
# POI class-tree resolve: leaf→SUBCLASS_OF up to target L1/L2 (timed per class)
    poi_tim: Dict[str, float] = {}
    cat_codes = load_category_codes(sess, timings=poi_tim)
    for kk in ORDER:
        steps.add("g3r", "D", f"POI类树解析 / class-tree resolve:{CAT_CN[kk]}", poi_tim.get(kk, 0.0),
                  rows=len(cat_codes[kk]), note="叶子→SUBCLASS_OF 上溯 / leaf→SUBCLASS_OF")
    t_cache = _t()
    caches = {kk: SubsetPOICache(cat_codes[kk]) for kk in ORDER}  # 主进程建一次 / parent builds once
    steps.add("g3r", "D", "六类 POI 子集缓存构建 / six-category POI subset caches", _t() - t_cache)
    grp = {}
    for pmc in mismatch:
        grp[pmc] = "mismatch"
    for pmc in baseline:
        grp[pmc] = "baseline"
    targets = [(cc, pmc, grp.get(pmc, "other"), sub, tot, c2, e2)
               for (cc, pmc, cls, sub, tot, c2, e2, nh) in records]
    targets = _filter_targets(targets, dtarget)
    ccs = [t[0] for t in targets]
    print(f"    目标建筑 / targets {len(targets):,} ({DTARGET_LABEL[dtarget]}) "
          f"错配 / mismatch {sum(1 for t in targets if t[2]=='mismatch'):,} "
          f"基准 / baseline {sum(1 for t in targets if t[2]=='baseline'):,}")
    t_knn = _t()
    dist = {}
    if workers > 1:
        for d in _parallel_map(_mp_d_init, (NEO4J_URI, auth, NEO4J_DB, caches),
                               _mp_d_work, ccs, workers, t0):
            dist.update(d)
    else:
        searcher = GeoSotKNNSearcherV31(sess, geom_type="Point", verbose=False)
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
    steps.add("g3r", "D", "空间桥·KNN 六类逐栋检索 / per-building six-cat KNN", _t() - t_knn,
              rows=len(ccs) * len(ORDER), note=f"workers={workers}")
    NT = len(targets)
    _write_acc(os.path.join(outdir, "g3r_service_accessibility.csv"),
               targets, dist)
    metrics["stages"]["D_knn_accessibility_ingraph"] = round(_t() - t0, 2)
    print(f"    完成 / done {NT:,}  {_t()-t0:.1f}s")

    metrics["stages"]["total_end_to_end"] = round(_t() - t_all, 2)
    metrics["result"] = {"resident_buildings": len(buildings),
                         "mismatch_parcels": len(mismatch),
                         "baseline_parcels": len(baseline),
                         "target_buildings": len(targets)}
    with open(os.path.join(outdir, "g3r_pipeline_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    sess.close(); drv.close()
    print(f"[G3R] 端到端 / end-to-end {metrics['stages']['total_end_to_end']:.1f}s "
          f"跨系统搬运 / cross-system xfers 0\n")
    return metrics, used_codes


# ══════════════════════════════════════════════════════════════════════
# PostGIS 流水线 (键: building_new.main_code, 与 G3R 对齐) / PostGIS pipeline (keyed by main_code)
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
    print("[PG] PostGIS 外置基线流水线 (几何库 + oracle_mig 跨库居民) / external baseline (geometry DB + cross-DB residents)")
    print("=" * W)
    t_all = _t()
    m = psycopg2.connect(**PG_MAIN)
    g = psycopg2.connect(**PG_MIG)

    # 若限定建筑集: 先由 main_code 映射 BLDG_NO / If restricted set: map main_code → BLDG_NO
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
        print(f"  限定建筑集 / restricted set: {len(restrict_codes):,} main_code → "
              f"{len(restrict_bno):,} BLDG_NO (与 G3R 同批 / same batch as G3R)")

    # A 跨库拉人口结构 / A cross-DB pull demographics
    print("[A] 跨库拉居民人口结构 (oracle_mig → app) ... / Cross-DB resident demographics ...")
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
    steps.add("postgis", "A", "跨库聚合居民人口结构 / cross-DB aggregate demographics", _t() - t0,
              rows=len(demo), note="oracle_mig→app 跨库搬运1 / cross-DB xfer 1")
    xfer.append({"dir": "oracle_mig→app", "what": "居民人口结构聚合 / resident demographics aggregate",
                 "rows": len(demo)})
    print(f"    有居民建筑 / resident bldgs {len(demo):,}  {_t()-t0:.1f}s (跨库搬运1 / xfer 1)")

    # B 空间连接 PIP (键: main_code) / B spatial join PIP (key: main_code)
    print("[B] PostGIS 空间连接 ST_Contains(质心∈地块) ... / ST_Contains(centroid ∈ parcel) ...")
    t0 = _t()
    with m.cursor() as c:
        t_ins = _t()
        c.execute("DROP TABLE IF EXISTS tgt_bldg")
        c.execute("CREATE TEMP TABLE tgt_bldg(bno text PRIMARY KEY)")
        execute_values(c, "INSERT INTO tgt_bldg(bno) VALUES %s "
                          "ON CONFLICT DO NOTHING",
                       [(k,) for k in demo.keys()], page_size=10000)
        c.execute("ANALYZE tgt_bldg")
        steps.add("postgis", "B", "上传目标建筑集(app→PG 搬运) / upload target buildings", _t() - t_ins,
                  rows=len(demo))
        # DISTINCT ON: 一栋建筑落在多重叠地块时只取最小面积地块 (与 G3R 同规则),
        # DISTINCT ON: when a building hits overlapping parcels, keep min-area one (same as G3R),
        # 消除多重包含造成的重复行 → 与 G3R 逐栋一一对应
        # drop duplicate rows from multi-contain → 1:1 with G3R per building.
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
        steps.add("postgis", "B", "ST_Contains 空间连接(质心∈地块) / spatial join", _t() - t_join,
                  rows=len(pip))
    m.commit()
    xfer.append({"dir": "app→postgis", "what": "有居民建筑 id / resident-building ids", "rows": len(demo)})
    xfer.append({"dir": "postgis→app", "what": "建筑→地块归属 / building→parcel attribution", "rows": len(pip)})
    metrics["stages"]["B_pip_spatial_join"] = round(_t() - t0, 2)
    print(f"    归属 / attributed {len(pip):,}  {_t()-t0:.1f}s (往返2 / round-trips 2)")

    # C 分类 + 筛选 (键: 建筑 main_code) / C classify + filter (key: building main_code)
    t0 = _t()
    # 用地类型: 假设不在应用层跑关键词分类器, 而是查规范化字典表 lu_function_dict
    # Land-use: look up canonical dict lu_function_dict (not app-layer keyword classifier)
    # (lu_name→land_class/land_subtype) 解析每个用地功能名 → 与 G3R 走 IN_LU_CLASS 对称。
    # (lu_name→land_class/land_subtype) to resolve each LU name → symmetric to G3R IN_LU_CLASS.
    t_lu = _t()
    lu_dict: Dict[str, Tuple[str, str]] = {}
    with m.cursor() as c:
        c.execute("SELECT lu_name, land_class, land_subtype "
                  "FROM public.lu_function_dict")
        for nm, cls_, sub_ in c.fetchall():
            lu_dict[nm] = (cls_, sub_)
    steps.add("postgis", "C", "用地类型字典解析(查 lu_function_dict) / LU dict resolve", _t() - t_lu,
              rows=len(lu_dict), note="规范化字典表, 非应用层关键词 / canonical dict, not keywords")
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
                # 按 '+' 拆多用地, 与 G3R collect(所有LUFunctionGroup) 同口径
                # Split multi-LU on '+'; same scope as G3R collect(all LUFunctionGroup)
                # (G3R 的 LUFunctionGroup 即 lu_functio 按 + 预拆的节点)
                # (G3R LUFunctionGroup nodes are lu_functio pre-split on '+')
                parts = [p for p in normalize(lu).split("+") if p] if lu else []
                # 逐名查字典 (缺项兜底 classify, 正常不触发) / Per-name dict lookup (classify fallback)
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
    steps.add("postgis", "C", "地块聚合+筛选错配/基准 / parcel agg + filter mismatch/baseline", _t() - t_agg,
              rows=len(mismatch) + len(baseline))
    metrics["stages"]["C_landuse_classify_select"] = round(_t() - t0, 2)
    print(f"[C] 错配 / mismatch {len(mismatch):,}  基准 / baseline {len(baseline):,}")

    # D KNN — geography on EPSG:4490 (matches G3R ellipsoidal metres)
    print("[D] PostGIS KNN 六类设施最近距离 / six-category nearest distance "
          f"(EPSG:{SOURCE_SRID} → geography + GIST) ...")
    t0 = _t()
    grp = {}
    for pmc in mismatch:
        grp[pmc] = "mismatch"
    for pmc in baseline:
        grp[pmc] = "baseline"
    # dtarget 控制 KNN 覆盖面 (与 G3R 同口径, 保证两侧计时可比) / dtarget controls KNN coverage (same scope as G3R)
    targets = [(cc, pmc, grp.get(pmc, "other"), sub, tot, ch, el)
               for (cc, pmc, cls, sub, tot, ch, el, nh) in records]
    targets = _filter_targets(targets, dtarget)
    print(f"    目标建筑 / targets {len(targets):,} ({DTARGET_LABEL[dtarget]}) "
          f"错配 / mismatch {sum(1 for t in targets if t[2]=='mismatch'):,} "
          f"基准 / baseline {sum(1 for t in targets if t[2]=='baseline'):,}")
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
              f"目标建筑质心 geography(EPSG:{SOURCE_SRID})+GIST(app→PG 搬运) / target centroids+GIST",
              _t() - t_pts, rows=len(targets))
    xfer.append({"dir": "app→postgis", "what": "目标建筑集(KNN) / target buildings (KNN)",
                 "rows": len(targets)})
    dist = {b[0]: {} for b in targets}
    for cat in ORDER:
        tc = _t()
        with m.cursor() as c:
            # D1 语义层级解析: 假设 poi_new 只有叶子码 poi, 粗类须 join 分类字典
            # D1 semantic hierarchy: assume poi_new has leaf code poi only; join category dict for coarse class
            # gaode_poi_category 展开为叶子码集合 (与 G3R 叶子→SUBCLASS_OF 对称)
            # Expand leaf-code set via gaode_poi_category (symmetric to G3R leaf→SUBCLASS_OF)
            t_res = _t()
            c.execute(f"""SELECT ("poi编码")::double precision
                          FROM public.gaode_poi_category
                          WHERE {GAODE_CAT_SQL[cat]}""")
            leaf_codes = [r[0] for r in c.fetchall()]
            steps.add("postgis", "D", f"POI字典解析 / POI dict resolve:{CAT_CN[cat]}", _t() - t_res,
                      rows=len(leaf_codes), note="join gaode_poi_category 展开叶子码 / expand leaf codes")
            # D2 叶子码过滤 poi_new → geography（与 G3R Vincenty/4490 同椭球米口径）
            # D2 filter poi_new by leaf codes → geography (same ellipsoidal metres as G3R)
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
                      f"POI子集 geography+GIST / POI subset:{CAT_CN[cat]}",
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
    xfer.append({"dir": "postgis→app", "what": "KNN 最近距离 / KNN nearest distances",
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
    print(f"[PG] 端到端 / end-to-end {metrics['stages']['total_end_to_end']:.1f}s "
          f"跨系统搬运 / cross-system xfers {len(xfer)}\n")
    return metrics


# ══════════════════════════════════════════════════════════════════════
# 对比 + 诊断 / Compare + diagnose
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
    """无权重分位数 (线性插值)。 / Unweighted percentile (linear interpolation)."""
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
    """group=None → 全市(全部行); 否则按 group 过滤。 / group=None → citywide (all rows); else filter by group."""
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
    print("  对比 & 诊断 / Compare & diagnose   G3R (图内/in-graph)  vs  PostGIS (外置/external)")
    print("=" * W)

    # (1) 地块集合一致性 / (1) Parcel-set consistency
    print("\n[1] 结果一致性 — 错配/基准地块集合 / Result consistency — mismatch/baseline parcel sets (statutory_plan.main_code)")
    print(f"  错配 / mismatch: G3R {len(g_mm):,}  PG {len(p_mm):,}  交集 / ∩ {len(g_mm & p_mm):,}"
          f"  Jaccard {_jaccard(g_mm, p_mm):.3f}")
    print(f"  基准 / baseline: G3R {len(g_bl):,}  PG {len(p_bl):,}  交集 / ∩ {len(g_bl & p_bl):,}"
          f"  Jaccard {_jaccard(g_bl, p_bl):.3f}")

    # (2a) 服务可达性 (G3R) — 参照线随阶段D 覆盖面 (dtarget) 自适应:
    # (2a) Service accessibility (G3R) — reference lines adapt to Stage-D coverage (dtarget):
    #   city → 错配/全市/基准 三线; mb → 错配/基准 双线; mismatch → 仅错配单线
    #   city → mismatch/citywide/baseline; mb → mismatch/baseline; mismatch → mismatch only
    dt = gj.get("dtarget", "city")
    print(f"\n[2a] 服务可达性 / accessibility (G3R, pop-weighted median nearest m)  "
          f"阶段D / Stage-D={DTARGET_LABEL.get(dt, dt)}")
    if dt == "city":
        print(f"  {'类别/cat':<12} {'错配/mm':>8} {'全市/city':>8} {'基准/bl':>8} "
              f"{'错配/全市':>9} {'错配/基准':>9}")
        print("  " + "-" * 60)
        for cat in ORDER:
            mm = _wmedian(_collect(g_acc, cat, "mismatch"))
            cy = _wmedian(_collect(g_acc, cat))          # 全市 = 全部行 / citywide = all rows
            bl = _wmedian(_collect(g_acc, cat, "baseline"))
            r_cy = mm / cy if cy and cy == cy else float("nan")
            r_bl = mm / bl if bl and bl == bl else float("nan")
            print(f"  {CAT_CN[cat]:<12} {mm:>7.0f}m {cy:>7.0f}m {bl:>7.0f}m "
                  f"{r_cy:>9.2f} {r_bl:>9.2f}")
    elif dt == "mb":
        print(f"  {'类别/cat':<12} {'错配/mm':>8} {'基准/bl':>8} {'错配/基准':>9}")
        print("  " + "-" * 42)
        for cat in ORDER:
            mm = _wmedian(_collect(g_acc, cat, "mismatch"))
            bl = _wmedian(_collect(g_acc, cat, "baseline"))
            r_bl = mm / bl if bl and bl == bl else float("nan")
            print(f"  {CAT_CN[cat]:<12} {mm:>7.0f}m {bl:>7.0f}m {r_bl:>9.2f}")
    else:                                                # mismatch: 仅单线 / single line
        print(f"  {'类别/cat':<12} {'错配/mm':>8}")
        print("  " + "-" * 22)
        for cat in ORDER:
            mm = _wmedian(_collect(g_acc, cat, "mismatch"))
            print(f"  {CAT_CN[cat]:<12} {mm:>7.0f}m")
        print("  [注] 本次仅算错配, 无基准/全市参照; 需比值请用 --dtarget mb/city / Note: mismatch-only; use --dtarget mb/city for ratios")

    # (2b) G3R vs PG 一致性 (当前阶段D 覆盖面下六类中位) — 正确性核对
    # (2b) G3R vs PG consistency (six-cat medians under Stage-D coverage) — correctness check
    print(f"\n[2b] 结果一致性 — 六类总人口加权中位 / Consistency — six-cat pop-weighted median (G3R vs PG, "
          f"{DTARGET_LABEL.get(dt, dt)})")
    print(f"  {'类别/cat':<12} {'G3R':>8} {'PG':>8} {'|Δ|':>7} {'相对差/rel':>7}")
    print("  " + "-" * 46)
    acc_rows = []
    for cat in ORDER:
        gm = _wmedian(_collect(g_acc, cat)); pm = _wmedian(_collect(p_acc, cat))
        diff = abs(gm - pm) if gm == gm and pm == pm else float("nan")
        rel = diff / pm if pm and pm == pm else float("nan")
        print(f"  {CAT_CN[cat]:<12} {gm:>7.0f}m {pm:>7.0f}m {diff:>6.0f}m "
              f"{rel*100:>6.1f}%")
        acc_rows.append((cat, gm, pm))

    # (3) 逐建筑诊断 (main_code 对齐) — 论文口径三指标
    # (3) Per-building diagnosis (main_code align) — three paper metrics:
    #   ① 最近设施命中一致率: G3R 与 PG 是否选中同一个 POI (同 main_code)
    #      Hit agreement: same nearest POI main_code on both sides
    #   ② 距离保真 (仅命中一致子集): 带符号误差 G3R-PG 的 MAE / 中位
    #      Distance fidelity (agree subset): signed G3R-PG MAE / median
    #   ③ |误差| 的 P95 / P99 (ECDF 尾部) —— 与 §4.4/§4.5.3 精度口径对齐
    #      |error| P95/P99 (ECDF tails) — aligned with §4.4/§4.5.3 precision
    print("\n[3] 逐建筑诊断 — 命中一致率 + 距离保真 (仅同一POI子集) + P95/P99 / Per-building: hit rate + distance fidelity + P95/P99")
    gmap = {r["building_cc"]: r for r in g_acc}
    common = [(gmap[r["building_cc"]], r) for r in p_acc
              if r["building_cc"] in gmap]
    print(f"  可对齐建筑 / alignable: {len(common):,} (G3R {len(g_acc):,} ∩ PG {len(p_acc):,})")
    if common:
        print(f"  {'类别/cat':<12} {'n':>7} {'命中一致率/hit%':>9} {'同POI-MAE':>10} "
              f"{'同POI中位/med':>9} {'|Δ|P95':>8} {'|Δ|P99':>8}")
        print("  " + "-" * 66)
        allsame_abs = []
        for cat in ORDER:
            n = same = 0
            signed, absd = [], []          # 仅命中一致子集 / agree-on-same-POI subset
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
            print(f"  {'六类合计(同POI)/6-cat':<12} {len(allsame_abs):>7,} "
                  f"{'':>9} {sum(allsame_abs)/len(allsame_abs):>9.2f}m "
                  f"{median(allsame_abs):>8.2f}m {_pct(allsame_abs,95):>7.2f}m "
                  f"{_pct(allsame_abs,99):>7.2f}m")
        print("  说明: 命中一致=两侧选同一POI; 该子集 MAE/P95/P99 才是可与 §4.4 距离核\n"
              "        (亚厘米) / §4.5.3 KNN (MAE 0.026m) 对齐的精度; 命中不一致\n"
              "        多为近等距 POI 平局翻转(与几何/CRS微差有关), 非算错。\n"
              "  Note: hit-agree = same POI; MAE/P95/P99 on that subset align with §4.4/§4.5.3;\n"
              "        disagreements are usually near-tie flips from geometry/CRS micro-diff.")

    # (4) 架构/成本 / (4) Architecture / cost
    print("\n[4] 架构 / 成本")
    if gj and pj:
        gs, ps = gj["stages"], pj["stages"]
        rows = [("A 人口 / pop", "A_pull_residents_ingraph", "A_pull_residents_crossdb"),
                ("B 归属 / attr", "B_reverse_overlay_ingraph", "B_pip_spatial_join"),
                ("C 分类 / class", "C_select", "C_landuse_classify_select"),
                ("D KNN", "D_knn_accessibility_ingraph", "D_knn_accessibility")]
        print(f"  {'阶段/stage':<10} {'G3R(s)':>10} {'PG(s)':>10}")
        print("  " + "-" * 32)
        for lab, gk, pk in rows:
            print(f"  {lab:<10} {gs.get(gk, 0):>10.1f} {ps.get(pk, 0):>10.1f}")
        print("  " + "-" * 32)
        print(f"  {'端到端/e2e':<10} {gs.get('total_end_to_end', 0):>10.1f} "
              f"{ps.get('total_end_to_end', 0):>10.1f}")
        print(f"  {'跨系统搬运/xfers':<8} {len(gj['cross_system_transfers']):>10} "
              f"{len(pj['cross_system_transfers']):>10}")
        for x in pj["cross_system_transfers"]:
            print(f"      PG [{x['dir']:<14}] {x['what']:<18} rows={x['rows']:,}")

    # (4b) 细粒度分步耗时 (含语义层级解析子步) — 让 G3R 优势可逐步看清
    # (4b) Fine-grained substep timings (incl. hierarchy resolve)
    _print_substeps(os.path.join(outdir, "mismatch_step_timings.csv"))

    if limit > 0:
        print("\n  [注] --limit 冒烟: 两侧已强制同一批建筑, 结果/时间可比; "
              "全量请去掉 --limit。 / Note: smoke mode uses same building batch; drop --limit for full.")

    try:
        _plots(figdir, gj, pj, acc_rows, common)
    except Exception as e:
        print(f"\n[!] 出图跳过 / plot skipped: {e}")
    print("\n[完成 / done]")


def _print_substeps(path):
    """读取细粒度分步耗时 CSV, 按 引擎/阶段 汇总打印 (语义解析子步单列)。 / Read fine-grained timing CSV; print by engine/stage."""
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        return
    print("\n[4b] 细粒度分步耗时 (s) —— 逐小步, 语义层级解析可见 / Fine-grained substep timings (s)")
    for eng, tag in (("g3r", "G3R (图内/in-graph)"), ("postgis", "PostGIS (外置/external)")):
        er = [r for r in rows if r["engine"] == eng]
        if not er:
            continue
        tot = sum(float(r["sec"]) for r in er)
        print(f"  {tag}  合计 / total {tot:.2f}s")
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
        labels = ["A 人口/pop", "B 归属/attr", "C 分类/class", "D KNN"]
        gv = [gs.get("A_pull_residents_ingraph", 0),
              gs.get("B_reverse_overlay_ingraph", 0), gs.get("C_select", 0),
              gs.get("D_knn_accessibility_ingraph", 0)]
        pv = [ps.get("A_pull_residents_crossdb", 0),
              ps.get("B_pip_spatial_join", 0),
              ps.get("C_landuse_classify_select", 0),
              ps.get("D_knn_accessibility", 0)]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
        x = np.arange(len(labels)); w = 0.38
        a1.bar(x - w/2, gv, w, color=G3R_COLOR, label="G3R (图内/in-graph)")
        a1.bar(x + w/2, pv, w, color=PG_COLOR, label="PostGIS (外置/external)")
        a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_ylabel("耗时 (s) / time (s)")
        a1.set_title("分阶段耗时 / Stage timings", fontweight="bold"); a1.legend()
        a1.grid(axis="y", alpha=0.3)
        e2e = [gs.get("total_end_to_end", 0), ps.get("total_end_to_end", 0)]
        xf = [len(gj["cross_system_transfers"]),
              len(pj["cross_system_transfers"])]
        bars = a2.bar(["G3R", "PostGIS"], e2e, color=[G3R_COLOR, PG_COLOR],
                      width=0.55)
        a2.set_ylabel("端到端 (s) / end-to-end (s)")
        a2.set_title("端到端 + 跨系统搬运 / E2E + cross-system xfers", fontweight="bold")
        for b, x0 in zip(bars, xf):
            a2.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{b.get_height():.0f}s\n搬运/xfers {x0}", ha="center",
                    va="bottom", fontsize=8)
        a2.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p1 = os.path.join(figdir, "pipeline_compare_cost.png")
        fig.savefig(p1); plt.close(fig); print(f"\n  [图1 / fig1] {p1}")

    if acc_rows:
        cns = [CAT_CN[r[0]] for r in acc_rows]
        gm = [r[1] if r[1] == r[1] else 0 for r in acc_rows]
        pm = [r[2] if r[2] == r[2] else 0 for r in acc_rows]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        x = np.arange(len(cns)); w = 0.38
        ax.bar(x - w/2, gm, w, color=G3R_COLOR, label="G3R")
        ax.bar(x + w/2, pm, w, color=PG_COLOR, label="PostGIS")
        ax.set_xticks(x); ax.set_xticklabels(cns, rotation=20, ha="right")
        ax.set_ylabel("人口加权中位最近距离 (m) / pop-weighted median nearest (m)")
        ax.set_title("六类设施可达性一致性 / Six-cat accessibility consistency", fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        p2 = os.path.join(figdir, "pipeline_compare_accessibility.png")
        fig.savefig(p2); plt.close(fig); print(f"  [图2 / fig2] {p2}")

    # 逐建筑散点 (诊断) / Per-building scatter (diagnosis)
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
        fig.suptitle("逐建筑最近距离对齐 (y=x 即完全一致) / Per-building nearest-dist align (y=x = perfect)", fontweight="bold")
        fig.tight_layout()
        p3 = os.path.join(figdir, "pipeline_diagnose_scatter.png")
        fig.savefig(p3); plt.close(fig); print(f"  [图3 / fig3] {p3}")


# ══════════════════════════════════════════════════════════════════════
def _prewarm():
    """对称把两侧热数据钉进内存, 消除冷/热 IO 波动 (与 §4.4/§4.5 prewarm 协议
    一致: 计时段测的是稳态计算, 不是首访磁盘 IO)。PG 两库用 pg_prewarm 载入
    表堆, Neo4j 全标签扫描分页。预热本身不计入流水线计时。
    Symmetrically pin hot data on both sides (steady-state compute, not cold IO).
    PG uses pg_prewarm; Neo4j full-label scans. Prewarm excluded from pipeline timing."""
    import psycopg2
    print("=" * W)
    print("[预热] 对称载入热数据 (PG pg_prewarm + Neo4j 分页扫描) ... / Prewarm: pin hot data both sides ...")
    t0 = _t()
    for tag, conn, tabs in (
        ("PG几何库 / geometry DB", PG_MAIN, ("public.building_new", "public.poi_new",
                               "public.statutory_plan")),
        ("PG居民库 / resident DB", PG_MIG, ("oracle_src.resident", "oracle_src.house"))):
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
                        print(f"    [{tag}] {t} 跳过 / skip: {e}")
            cn.commit(); cn.close()
            print(f"    [{tag}] 预热 / warmed {warmed:,} 页/pages")
        except Exception as e:
            print(f"    [{tag}] 连接失败, 跳过 / connect failed, skip: {e}")
    try:
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
        with drv.session(database=NEO4J_DB) as s:
            for lbl in ("Resident", "House", "Building", "GeoSOTCode"):
                s.run(f"MATCH (n:{lbl}) RETURN count(n)").consume()
            print("    [Neo4j] 全标签扫描分页完成 / full-label scan done")
            # 关系存储预热: 标签 count 只焐节点存储, 但阶段A那条大聚合要遍历
            # Rel-store prewarm: label count only warms node store, but Stage-A aggregate must traverse
            # HAS_GEOSOT / LOCATED_IN / LIVE_IN / HAS_COUNTY_ORIGIN /
            # IN_PROVINCE / IN_CITY 关系存储 —— 首访这些冷页正是首跑阶段A
            # IN_PROVINCE / IN_CITY relationship store — first-touch cold pages are why Stage-A
            # 异常慢(千秒级尖峰)的根因。这里镜像阶段A遍历一次把关系页焐热,
            # first run is abnormally slow (ks-scale spike). Mirror Stage-A once to warm rel pages,
            # 使首跑与稳态一致 (本段不计入流水线计时)。
            # so first timed run matches steady state (this block excluded from pipeline timing).
            s.run("MATCH (:Building)-[:HAS_GEOSOT]->(:GeoSOTCode) "
                  "RETURN count(*)").consume()
            s.run(
                "MATCH (b:Building)<-[:LOCATED_IN]-(:House)"
                "<-[:LIVE_IN]-(r:Resident)\n"
                "OPTIONAL MATCH (r)-[:HAS_COUNTY_ORIGIN]->(co:CountyOrigin)\n"
                "OPTIONAL MATCH (co)-[:IN_PROVINCE]->(:Province)\n"
                "OPTIONAL MATCH (co)-[:IN_CITY]->(:AdminCity)\n"
                "RETURN count(r)").consume()
            print("    [Neo4j] 关系存储遍历预热完成 (镜像阶段A, 消首跑尖峰) / rel-store prewarm done (mirror Stage A)")
        drv.close()
    except Exception as e:
        print(f"    [Neo4j] 跳过 / skip: {e}")
    print(f"    预热完成 / prewarm done {_t()-t0:.1f}s (不计入流水线计时 / excluded from pipeline)")
    print("=" * W)


def _med(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _bench_summary(runs, engine):
    """多次跑的每阶段 min/中位/max。中位数即抗 OS 调度噪声的正式取数。 / Per-stage min/median/max over repeats; median is the formal figure."""
    keys, seen = [], set()
    for r in runs:                              # 保序去重收集所有阶段键 / ordered unique stage keys
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    print(f"\n  [{engine}]  (n={len(runs)})    {'min':>9} {'中位/med':>9} {'max':>9}"
          f" {'极差%/spread':>7}")
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
    limit, threshold, outdir, figdir = 0, 100, "csv", "paper_figures"
    only, skip_g3r, skip_pg, prewarm = "", False, False, False
    dtarget = "city"                        # 阶段D 覆盖: city/mb/mismatch / Stage-D coverage
    repeat = 1                              # >1: 多次跑取每阶段 min/中位/max / multi-run min/med/max
    workers = min(os.cpu_count() or 1, 16)  # 默认 = min(逻辑核, 16); 1 = 串行
                                            # default = min(logical cores, 16); 1 = serial
                                            # (共享缓存后并行不再受重载争用制约)
                                            # (with shared cache, parallelism no longer reload-bound)
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
        print(f"[警告 / warn] --dtarget {dtarget} 无效, 回退 city / invalid, fallback city (city/mb/mismatch)")
        dtarget = "city"
    os.makedirs(outdir, exist_ok=True)

    print("#" * W)
    print("#  错配复合查询 单脚本实验  (G3R vs PostGIS) / Mismatch composite one-script experiment")
    print(f"#  阈值居民>={threshold} / res threshold   limit={limit or '全量/full'}   "
          f"outdir={outdir}   workers={workers}   "
          f"阶段D/Stage-D={DTARGET_LABEL[dtarget]}")
    if limit > 0:
        print("#  冒烟模式: 两侧强制用同一批建筑 (main_code 对齐), 时间/结果可比 / Smoke: same building batch both sides")
    if dtarget != "city":
        print(f"#  阶段D 仅算「{DTARGET_LABEL[dtarget]}」建筑 (A/B/C 仍全量; "
              f"提速来自 D 目标缩小) / Stage-D only those buildings; A/B/C still full")
    print("#" * W)

    if prewarm and only in ("", "g3r", "pg", "run"):
        _prewarm()

    g_runs, p_runs = [], []                # 各次每阶段耗时 (供 --repeat 汇总) / per-run stage timings
    for it in range(1, repeat + 1):
        if repeat > 1:
            print("\n" + "#" * W)
            print(f"#  REPEAT {it}/{repeat}")
            print("#" * W)
        shared_codes = None
        steps = StepLog()                  # 细粒度分步耗时 (含语义层级解析子步) / fine-grained substep timings
        if only in ("", "g3r", "run") and not skip_g3r:
            _gm, used = run_g3r(limit, threshold, outdir, workers=workers,
                                dtarget=dtarget, steps=steps)
            g_runs.append(_gm["stages"])
            if limit > 0:
                shared_codes = used        # 冒烟: 用 G3R 同批建筑约束 PG / smoke: constrain PG to G3R batch
        if only in ("", "pg", "run") and not skip_pg:
            _pm = run_pg(limit, threshold, outdir, restrict_codes=shared_codes,
                         dtarget=dtarget, steps=steps)
            p_runs.append(_pm["stages"])
        if steps.rows:
            sp = os.path.join(outdir, "mismatch_step_timings.csv")
            steps.write(sp)
            print(f"\n[细粒度耗时 / fine timings] {sp}  ({len(steps.rows)} 子步 / substeps)")
    if only in ("", "compare", "diag", "run"):
        compare_and_diagnose(outdir, figdir, limit)   # 用最后一次 CSV / use last-run CSVs

    if repeat > 1:
        print("\n" + "=" * W)
        print(f"  多次基准汇总 / multi-run bench summary (N={repeat}, 预热/prewarm {'开/on' if prewarm else '关/off'}, "
              f"workers={workers}, 阶段D/Stage-D={DTARGET_LABEL[dtarget]})")
        print("  取数建议: 每阶段用「中位」列; 极差% 大者受本机负载影响, 需减背景干扰 / Prefer median; large spread% ⇒ reduce background load")
        print("=" * W)
        if g_runs:
            _bench_summary(g_runs, "G3R")
        if p_runs:
            _bench_summary(p_runs, "PostGIS")


if __name__ == "__main__":
    main()

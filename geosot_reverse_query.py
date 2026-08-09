#!/usr/bin/env python3
# coding: utf-8
"""
GeoSOT 反向叠置查询 — 给定要素, 找所属地块
GeoSOT reverse overlay query — given a feature, find its containing parcel

给定一个 POI (或其他点/面要素) 的编码 (main_code), 通过 GeoSOTCommonPrefixCode
前缀匹配筛选候选地块 (如 Statutory Plan), 再用编码-only 绕数法 PIP
精确判定.
Given a POI (or other point/area feature) main_code, filter candidate parcels
(e.g. Statutory Plan) via GeoSOTCommonPrefixCode prefix match, then exact-test
with code-only winding-number PIP.

与 V3 的区别 / vs V3:
  - V3: 给定地块 → 找地块内的要素 (正向覆盖 + 图推理)
  - V3: given parcel → find features inside (forward cover + graph reasoning)
  - 反向: 给定要素 → 找该要素所属的地块 (前缀匹配 + PIP)
  - Reverse: given feature → find containing parcel (prefix match + PIP)

算法 (与正向叠置 v7 / 几何推理共享同一编码-only 几何原理):
Algorithm (same code-only geometry principles as forward overlay v7 / geometry reasoning):
  1. 读取 POI 的 main_code, 通过位运算 _code_to_xy 直接映射到层级 32
     的 Z-order 整数网格坐标 (无经纬度解码).
     Read POI main_code; map via _code_to_xy bit ops to level-32 Z-order integer grid (no lon/lat decode).
  2. 通过 GeoSOTCommonPrefixCode 前缀匹配找候选地块
     策略 A: poi_code STARTS WITH prefix_code (Neo4j 索引)
     策略 B: LCPC_HAS_LEVEL_k_DIGIT 层级数字链回退
     Find candidate parcels via GeoSOTCommonPrefixCode prefix match
     Strategy A: poi_code STARTS WITH prefix_code (Neo4j index)
     Strategy B: LCPC_HAS_LEVEL_k_DIGIT digit-chain fallback
  3. 候选地块的多边形顶点用同样位运算映射到 Z-order 整数网格
     (PlanPolygonCache 在 setup 阶段一次性构建, 缓存为
      Dict[main_code, (N, 2) int64 数组]).
     Map candidate polygon vertices the same way (PlanPolygonCache built once at setup
     as Dict[main_code, (N,2) int64 arrays]).
  4. 绕数法 (Winding Number) PIP, 全链路整数运算, 无浮点误差,
     无经纬度参与, 与 geosot_geometry_reasoning._cross /
     geosot_overlay_query_v7._winding_number_batch 同一原理.
     Winding-number PIP, all-integer, no float error, no lon/lat; same principle as
     geosot_geometry_reasoning._cross / geosot_overlay_query_v7._winding_number_batch.
  5. (可选) PostGIS ST_Contains 对比验证.
     Optional PostGIS ST_Contains comparison.

使用方法 / Usage:
    python geosot_reverse_query.py --code G0xxx
    python geosot_reverse_query.py --batch 50 --seed 42
"""

import csv
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import psycopg2
from neo4j import GraphDatabase

from geosot_spatial_reasoning import (
    PG_CONFIG,
    fast_code_to_center,
    is_valid_g0_code,
)
from geosot_overlay_query import (
    SOURCE_TO_PG_TABLE,
)
from geosot_overlay_query_v7 import (
    GRID_LEVEL as _GRID_LEVEL,
    _code_to_xy,
    _xy_dms_to_linear,
    _winding_number_batch,
)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

_t = time.perf_counter


# ============================================================================
# 几何辅助 / Geometry helpers
# ============================================================================

def _pip_single(
    poi_xy: Tuple[int, int],
    polygon_xy: Optional[np.ndarray],
) -> bool:
    """
    编码-only 绕数法 (Winding Number) PIP 单点测试.
    Code-only winding-number PIP for a single point.

    输入坐标为 GeoSOT 编码经位运算 _code_to_xy 直接映射得到的
    Z-order 整数网格坐标 (level=32), 全链路 int64 运算, 无浮点误差,
    无经纬度参与. 与 geosot_overlay_query_v7._winding_number_batch
    使用同一几何原理.
    Inputs are Z-order integer grid coords (level=32) from _code_to_xy; all-int64,
    no float error, no lon/lat. Same geometry principle as
    geosot_overlay_query_v7._winding_number_batch.

    @param poi_xy     POI 的 (x, y) Z-order 整数坐标 / POI (x,y) Z-order integers
    @param polygon_xy (N, 2) int64 多边形顶点数组; None 或顶点 < 3 则为 False
                      / (N,2) int64 polygon verts; None or <3 verts → False
    @returns True 当且仅当 POI 严格落在多边形内部 (绕数 ≠ 0)
             / True iff POI is strictly inside (winding ≠ 0)
    """
    if polygon_xy is None or len(polygon_xy) < 3:
        return False
    pts = np.asarray([poi_xy], dtype=np.int64)
    return bool(_winding_number_batch(pts, polygon_xy)[0])


def _pip_batch(
    pts_xy: np.ndarray,
    polygon_xy: Optional[np.ndarray],
) -> np.ndarray:
    """
    编码-only 绕数法批量 PIP. 直接复用 v7 的 _winding_number_batch.
    Code-only batch winding PIP. Reuses v7 _winding_number_batch.

    @param pts_xy     (M, 2) int64 候选点 / candidate points
    @param polygon_xy (N, 2) int64 多边形顶点 / polygon vertices
    @returns bool 数组 (M,) / bool array (M,)
    """
    if polygon_xy is None or len(polygon_xy) < 3 or len(pts_xy) == 0:
        return np.zeros(len(pts_xy), dtype=bool)
    return _winding_number_batch(pts_xy, polygon_xy)


def _bbox_of(polygon_xy: np.ndarray) -> Tuple[int, int, int, int]:
    """多边形整数轴对齐包围盒 (min_x, max_x, min_y, max_y)。 / Integer axis-aligned polygon bbox (min_x, max_x, min_y, max_y)."""
    xs = polygon_xy[:, 0]
    ys = polygon_xy[:, 1]
    return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))


def _point_in_bbox(
    poi_xy: Tuple[int, int],
    bbox: Optional[Tuple[int, int, int, int]],
) -> bool:
    """
    点是否落在整数包围盒内 (含边界)。bbox 预筛: 盒外的点必然在多边形外,
    可在昂贵的绕数法 PIP 之前以纯整数比较剔除绝大多数候选, 结果不变。
    Whether point is inside the integer bbox (inclusive). bbox prefilter: out-of-box
    points are outside the polygon; drop most candidates with pure integer compares
    before expensive winding PIP; results unchanged.

    @param x,y 点坐标 / point coords
    @param bbox (min_x, max_x, min_y, max_y)
    """
    if bbox is None:
        return True
    x, y = poi_xy
    return bbox[0] <= x <= bbox[1] and bbox[2] <= y <= bbox[3]


# ============================================================================
# 候选地块查询 — 策略 A: 前缀索引 / Candidate parcel query — strategy A: prefix index
# ============================================================================

def _query_parcels_by_prefix(
    session,
    poi_code: str,
    parcel_source: str = "Statutory Plan",
) -> Tuple[List[Dict], float]:
    """
    通过 GeoSOTCommonPrefixCode 前缀匹配查找候选地块。
    Find candidate parcels via GeoSOTCommonPrefixCode prefix match.

    将 POI 编码的所有前缀字符串作为精确匹配条件，
    利用 prefix_code 索引做高效查询。
    Match all prefixes of the POI code exactly, using the prefix_code index.
    """
    # 从 POI 编码生成所有可能的前缀 (从最长到最短，最短取 "G0") / Generate all prefixes of the POI code (longest to shortest; shortest is "G0")
    prefixes = [poi_code[:i] for i in range(len(poi_code), 1, -1)]

    t0 = _t()
    cp_idx = get_common_prefix_index()
    if cp_idx is not None and cp_idx.has_source(parcel_source):
        # 本地倒排索引: 取每个 parcel 命中的最长 prefix / Local inverted index: take the longest hit prefix per parcel
        # (与 Cypher 中 ORDER BY size(prefix) DESC + collect[0] 等价) / (equivalent to Cypher ORDER BY size(prefix) DESC + collect[0])
        best_per_mc: Dict[str, str] = {}
        for pf in prefixes:
            for hit in cp_idx.query_prefixes([pf]):
                mc = hit["mc"]
                pref = hit["prefix"]
                cur = best_per_mc.get(mc)
                if cur is None or len(pref) > len(cur):
                    best_per_mc[mc] = pref
        rows = [{"mc": mc, "prefix": pf}
                for mc, pf in best_per_mc.items()]
    else:
        rows = [dict(r) for r in session.run(
            "UNWIND $prefixes AS pf\n"
            "MATCH (p:GeoSOTCommonPrefixCode {prefix_code: pf})\n"
            "MATCH (parcel:SpatialFeature)"
            "-[:HAS_COMMON_PREFIX_CODE]->(p)\n"
            "MATCH (parcel)-[:HAS_CENTROID_CODE]->"
            "(c:GeoSOTCentroidCode)\n"
            "MATCH (c)-[:FROM_SOURCE]->"
            "(:DataSource {name: $src})\n"
            "MATCH (c)-[:HAS_GEOMETRY_TYPE]->"
            "(:GeoFeatureType {type_name: 'Polygon'})\n"
            "WITH parcel.main_code AS mc, "
            "p.prefix_code AS prefix\n"
            "ORDER BY size(prefix) DESC\n"
            "WITH mc, collect(prefix)[0] AS prefix\n"
            "RETURN DISTINCT mc, prefix",
            prefixes=prefixes,
            src=parcel_source,
        )]
    t_ms = (_t() - t0) * 1000
    return rows, t_ms



# ============================================================================
# 候选地块查询 — 策略 B: LCPC_HAS_LEVEL 层级数字链 / Candidate parcel query — strategy B: LCPC_HAS_LEVEL digit chain
# ============================================================================

def _build_reverse_located_in_cypher(level: int) -> str:
    """
    构建 LCPC_HAS_LEVEL 层级数字链反向查询 Cypher。
    Build Cypher for reverse query via LCPC_HAS_LEVEL digit chain.

    通过公共前缀码 (GeoSOTCommonPrefixCode) 的 LCPC_HAS_LEVEL_k_DIGIT
    关系逐层缩小候选。
    Narrow candidates level-by-level via GeoSOTCommonPrefixCode LCPC_HAS_LEVEL_k_DIGIT.
    """
    parts = [
        "MATCH (p:GeoSOTCommonPrefixCode)"
    ]
    for lv in range(1, level + 1):
        parts.append(
            f"MATCH (p)-[:LCPC_HAS_LEVEL_{lv}_DIGIT]->"
            f"(:GeoSOTLevelDigit{lv} "
            f"{{digit: $digits[{lv - 1}]}})"
        )
    parts.append(
        "MATCH (parcel:SpatialFeature)"
        "-[:HAS_COMMON_PREFIX_CODE]->(p)\n"
        "MATCH (parcel)-[:HAS_CENTROID_CODE]->"
        "(c:GeoSOTCentroidCode)\n"
        "MATCH (c)-[:FROM_SOURCE]->"
        "(:DataSource {name: $src})\n"
        "MATCH (c)-[:HAS_GEOMETRY_TYPE]->"
        "(:GeoFeatureType {type_name: 'Polygon'})\n"
        "OPTIONAL MATCH (parcel)-[hv:HAS_VERTEX_CODE]->"
        "(v:GeoSOTVertexCode)\n"
        "WITH parcel, p, v, hv "
        "ORDER BY parcel.feature_id, hv.seq\n"
        "WITH parcel.main_code AS mc,\n"
        "     p.prefix_code AS prefix,\n"
        "     collect(CASE WHEN v IS NOT NULL "
        "THEN v.vertex_code ELSE null END) AS raw_verts\n"
        "WITH mc, prefix,\n"
        "     [x IN raw_verts WHERE x IS NOT NULL] "
        "AS vertex_codes\n"
        "WHERE size(vertex_codes) >= 3\n"
        "RETURN DISTINCT mc, prefix, vertex_codes\n"
        "ORDER BY size(prefix) DESC"
    )
    return "\n".join(parts)


def _query_parcels_by_grid(
    session,
    poi_code: str,
    parcel_source: str = "Statutory Plan",
) -> Tuple[List[Dict], float]:
    """
    通过 LCPC_HAS_LEVEL_k_DIGIT 层级数字链逐层缩小范围查找候选地块。
    Find candidate parcels by narrowing via LCPC_HAS_LEVEL_k_DIGIT digit chain.

    从最粗层级开始，逐层细化直到找到候选或层级用尽。
    Start coarse and refine until candidates are found or levels are exhausted.

    @returns (候选列表, 耗时ms) / (candidates, time_ms)
    """
    body = poi_code[1:]  # 去掉 G / strip leading G
    max_level = min(32, len(body))
    digits = list(body)

    t0 = _t()
    all_candidates: Dict[str, Dict] = {}

    for level in range(max_level, 0, -1):
        cypher = _build_reverse_located_in_cypher(level)
        rows = [dict(r) for r in session.run(
            cypher, digits=digits[:level], src=parcel_source)]
        for r in rows:
            mc = str(r["mc"]).strip().upper()
            if mc not in all_candidates:
                all_candidates[mc] = r

        if all_candidates:
            break

    t_ms = (_t() - t0) * 1000
    return list(all_candidates.values()), t_ms


def _query_parcels(
    session,
    poi_code: str,
    parcel_source: str,
    strategy: str = "prefix",
) -> Tuple[List[Dict], float]:
    """统一入口。 / Unified entry point."""
    if strategy == "located_in":
        return _query_parcels_by_grid(
            session, poi_code, parcel_source)
    else:
        return _query_parcels_by_prefix(
            session, poi_code, parcel_source)


# ============================================================================
# PostGIS 对比 / PostGIS comparison
# ============================================================================

def _postgis_reverse_query(
    pg_conn,
    poi_code: str,
    poi_table: str,
    parcel_table: str,
) -> Tuple[List[str], float]:
    """PostGIS ST_Contains 查找包含该要素的地块。 / Find parcels containing the feature via PostGIS ST_Contains."""
    sql = (
        f"SELECT p.main_code "
        f"FROM {parcel_table} p, {poi_table} t "
        f"WHERE t.main_code = %s "
        f"AND ST_Contains(p.geometry, t.geometry)"
    )
    t0 = _t()
    cur = pg_conn.cursor()
    cur.execute(sql, (poi_code,))
    results = [str(r[0]).strip().upper()
               for r in cur.fetchall() if r[0]]
    cur.close()
    t_ms = (_t() - t0) * 1000
    return results, t_ms


# ============================================================================
# 按需获取多边形 (单条查询模式) / Fetch polygons on demand (single-query mode)
# ============================================================================

def _fetch_polygons_for_codes(
    session,
    main_codes: List[str],
) -> Dict[str, np.ndarray]:
    """
    批量按需获取指定 main_code 列表的多边形顶点 Z-order 整数坐标.
    Batch on-demand fetch of polygon vertex Z-order integer coords for main_codes.

    顶点编码经位运算 _code_to_xy 直接映射到层级 32 网格 (无经纬度解码),
    Vertex codes map via _code_to_xy to level-32 grid (no lon/lat decode),
    """
    if not main_codes:
        return {}
    rows = [dict(r) for r in session.run(
        "UNWIND $codes AS mc\n"
        "MATCH (c:GeoSOTCentroidCode {main_code: mc})"
        "<-[:HAS_CENTROID_CODE]-(p:SpatialFeature)\n"
        "OPTIONAL MATCH (p)-[hv:HAS_VERTEX_CODE]->"
        "(v:GeoSOTVertexCode)\n"
        "WITH p, v, hv ORDER BY p.feature_id, "
        "hv.seq\n"
        "WITH p.main_code AS mc,\n"
        "     collect(CASE WHEN v IS NOT NULL "
        "THEN v.vertex_code ELSE null END) AS raw_verts\n"
        "WITH mc, [x IN raw_verts WHERE x IS NOT NULL] "
        "AS vertex_codes\n"
        "WHERE size(vertex_codes) >= 3\n"
        "RETURN mc, vertex_codes",
        codes=main_codes,
    )]
    result: Dict[str, np.ndarray] = {}
    for r in rows:
        mc = str(r["mc"]).strip().upper()
        vcodes = [str(v).strip().upper() for v in r["vertex_codes"]]
        xy_list: List[Tuple[int, int]] = []
        for vc in vcodes:
            try:
                xy_list.append(_code_to_xy(vc, _GRID_LEVEL))
            except Exception:
                pass
        if len(xy_list) >= 3:
            # DMS 位打包 -> 线性欧式整数空间, 消除分/秒进位空洞对绕数法的扭曲 / DMS bit-pack → linear Euclidean integer space; remove deg/min/sec carry holes that warp winding PIP
            result[mc] = _xy_dms_to_linear(
                np.asarray(xy_list, dtype=np.int64))
    return result


# ============================================================================
# Plan 多边形预加载器 / Plan polygon preloader
# ============================================================================

class PlanPolygonCache:
    """
    预加载所有地块多边形顶点的 Z-order 整数坐标缓存.
    Preload Z-order integer vertex caches for all parcel polygons.

    与之前版本 (缓存 List[(lon, lat)]) 不同, 此实现完全沿用与正向叠置 v7
    相同的"编码-only"整数几何空间.
    Unlike the prior version (List[(lon,lat)]), this uses the same code-only integer
    geometry space as forward overlay v7.
    """

    def __init__(self, session, verbose: bool = True):
        self._session = session
        self._verbose = verbose
        self._cache: Dict[str, np.ndarray] = {}
        # 预计算整数包围盒 (min_x, max_x, min_y, max_y), 供 PIP bbox 预筛 / Precompute integer bbox (min_x, max_x, min_y, max_y) for PIP bbox prefilter
        self._bbox: Dict[str, Tuple[int, int, int, int]] = {}

    @property
    def count(self) -> int:
        return len(self._cache)

    def preload(self, parcel_source: str = "Statutory Plan") -> int:
        t0 = _t()
        rows = [dict(r) for r in self._session.run(
            "MATCH (c:GeoSOTCentroidCode)"
            "-[:FROM_SOURCE]->(:DataSource {name: $src})\n"
            "MATCH (c)-[:HAS_GEOMETRY_TYPE]->"
            "(:GeoFeatureType {type_name: 'Polygon'})\n"
            "MATCH (c)<-[:HAS_CENTROID_CODE]-(p:SpatialFeature)\n"
            "OPTIONAL MATCH (p)-[hv:HAS_VERTEX_CODE]->"
            "(v:GeoSOTVertexCode)\n"
            "WITH p, v, hv ORDER BY p.feature_id, "
            "hv.seq\n"
            "WITH p.main_code AS mc,\n"
            "     collect(CASE WHEN v IS NOT NULL "
            "THEN v.vertex_code ELSE null END) AS raw_verts\n"
            "WITH mc, [x IN raw_verts WHERE x IS NOT NULL] "
            "AS vertex_codes\n"
            "WHERE size(vertex_codes) >= 3\n"
            "RETURN mc, vertex_codes",
            src=parcel_source,
        )]
        t_query = (_t() - t0) * 1000

        t0 = _t()
        for r in rows:
            mc = str(r["mc"]).strip().upper()
            vcodes = [str(v).strip().upper() for v in r["vertex_codes"]]
            xy_list: List[Tuple[int, int]] = []
            for vc in vcodes:
                try:
                    xy_list.append(_code_to_xy(vc, _GRID_LEVEL))
                except Exception:
                    pass
            if len(xy_list) >= 3:
                # DMS 位打包 -> 线性欧式整数空间, 消除分/秒进位空洞对绕数法的扭曲 / DMS bit-pack → linear Euclidean integer space; remove deg/min/sec carry holes that warp winding PIP
                arr = _xy_dms_to_linear(
                    np.asarray(xy_list, dtype=np.int64))
                self._cache[mc] = arr
                self._bbox[mc] = _bbox_of(arr)
        t_xy = (_t() - t0) * 1000

        if self._verbose:
            print(f"    Neo4j 查询: {t_query:.0f} ms ({len(rows):,} 条)")
            print(f"    Z-order 映射: {t_xy:.0f} ms "
                  f"({len(self._cache):,} 个有效多边形, 位运算, 无经纬度解码)")
        return len(self._cache)

    def get_polygon(self, main_code: str) -> Optional[np.ndarray]:
        """返回 (N, 2) int64 顶点 Z-order 数组; 不存在则 None. / Return (N,2) int64 Z-order vertex array; None if missing."""
        return self._cache.get(main_code.strip().upper())

    def get_bbox(
        self, main_code: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """返回预计算整数包围盒 (min_x, max_x, min_y, max_y); 不存在则 None. / Return precomputed integer bbox; None if missing."""
        return self._bbox.get(main_code.strip().upper())


# ============================================================================
# Plan parcel 共有前缀本地倒排索引 (与 PG pg_prewarm 等价对照) / Plan-parcel common-prefix local inverted index (counterpart to PG pg_prewarm)
# ============================================================================

class CommonPrefixIndex:
    """
    GR 端的 plan parcel 共有前缀本地倒排索引.
    GR-side plan-parcel common-prefix local inverted index.

    setup 阶段一次性从 Neo4j 拉取所有 (parcel.main_code, GeoSOTCommonPrefixCode)
    对, 构建 prefix -> [parcel_codes] 倒排。
    At setup, pull all (parcel.main_code, GeoSOTCommonPrefixCode) pairs from Neo4j
    and build prefix -> [parcel_codes] inverted lists.
    """

    def __init__(self, session,
                 parcel_source: str = "Statutory Plan",
                 verbose: bool = True):
        """
        @param session Neo4j session
        @param parcel_source Source 名称 (默认 Statutory Plan) / Source name (default Statutory Plan)
        @param verbose 是否打印 / whether to print
        """
        self._index: Dict[str, Set[str]] = {}
        self._parcel_source = parcel_source
        self.n_pairs = 0
        self.elapsed_sec = 0.0

        t0 = _t()
        try:
            rows = list(session.run(
                "MATCH (c:GeoSOTCentroidCode)-[:FROM_SOURCE]->"
                "(:DataSource {name: $src})\n"
                "MATCH (c)-[:HAS_GEOMETRY_TYPE]->"
                "(:GeoFeatureType {type_name: 'Polygon'})\n"
                "MATCH (c)<-[:HAS_CENTROID_CODE]-(parcel:SpatialFeature)\n"
                "MATCH (parcel)-[:HAS_COMMON_PREFIX_CODE]->"
                "(p:GeoSOTCommonPrefixCode)\n"
                "RETURN parcel.main_code AS mc, "
                "p.prefix_code AS prefix",
                src=parcel_source))
        except Exception as e:
            if verbose:
                print(f"    [common_prefix_index] 加载失败: {e}")
            self.elapsed_sec = _t() - t0
            return

        n_pairs = 0
        for r in rows:
            mc = r.get("mc")
            pf = r.get("prefix")
            if mc is None or pf is None:
                continue
            mc_s = str(mc).strip().upper()
            pf_s = str(pf).strip().upper()
            if not mc_s or not pf_s:
                continue
            self._index.setdefault(pf_s, set()).add(mc_s)
            n_pairs += 1

        self.n_pairs = n_pairs
        self.elapsed_sec = _t() - t0

        if verbose:
            n_parcels = len({
                m for s in self._index.values() for m in s
            })
            print(f"    [common_prefix_index] {parcel_source}: "
                  f"{len(self._index):,} 个 prefix, "
                  f"{n_parcels:,} 个 parcel, "
                  f"{n_pairs:,} 个 (parcel, prefix) 对, "
                  f"耗时 {self.elapsed_sec * 1000:.0f} ms")

    def query_prefixes(self, prefixes: List[str]) -> List[Dict]:
        """
        本地批量前缀查询 (dict lookup).
        Local batch prefix query (dict lookup).

        @param prefixes POI 编码切片得到的一批前缀字符串 / prefixes sliced from POI code
        @returns [{"main_code": ..., "prefix": ...}, ...]
        """
        results: List[Dict] = []
        for pf in prefixes:
            pf_u = str(pf).strip().upper()
            mcs = self._index.get(pf_u)
            if not mcs:
                continue
            for mc in mcs:
                results.append({"mc": mc, "prefix": pf_u})
        return results

    def is_loaded(self) -> bool:
        return bool(self._index)

    def n_prefixes(self) -> int:
        return len(self._index)

    def has_source(self, source: str) -> bool:
        return self._parcel_source == source and self.is_loaded()


# 模块级单例: setup 阶段注入, 查询期间所有路径透明使用. / Module singleton: injected at setup; used transparently on all query paths.
# 为 None 时所有查询回退到原 Cypher 路径 (cold-cache 对照模式). / When None, all queries fall back to Cypher (cold-cache control mode).
_common_prefix_index: Optional[CommonPrefixIndex] = None


def set_common_prefix_index(idx: Optional[CommonPrefixIndex]):
    """注入全局共有前缀索引 (None 时回到 cold-cache Cypher 模式). / Inject global common-prefix index (None → cold-cache Cypher)."""
    global _common_prefix_index
    _common_prefix_index = idx


def get_common_prefix_index() -> Optional[CommonPrefixIndex]:
    return _common_prefix_index


# ============================================================================
# 递进前缀匹配 + PIP (命中即停) / Progressive prefix match + PIP (stop on first hit)
# ============================================================================

def _cascade_prefix_query(
    session,
    poi_code: str,
    poi_xy: Tuple[int, int],
    parcel_source: str = "Statutory Plan",
    polygon_cache: Optional[PlanPolygonCache] = None,
    tier_size: int = 5,
) -> Dict:
    """
    逐层递进前缀匹配 + PIP，命中即停。
    Progressive prefix match + PIP; stop on first hit.

    从最长前缀开始，每次查询 tier_size 个前缀级别，
    查完立即做 PIP 测试。一旦命中就终止，
    避免短前缀产生过多候选。
    Start from the longest prefix; query tier_size prefix levels at a time;
    PIP immediately after each tier. Stop on hit to avoid too many short-prefix candidates.
    """
    prefixes = [poi_code[:i] for i in range(len(poi_code), 1, -1)]

    all_candidates: List[Dict] = []
    pip_results: List[Dict] = []
    seen_mc: Set[str] = set()
    n_decoded = 0
    t_query_total = 0.0
    t_pip_total = 0.0

    _CYPHER = (
        "UNWIND $prefixes AS pf\n"
        "MATCH (p:GeoSOTCommonPrefixCode {prefix_code: pf})\n"
        "MATCH (parcel:SpatialFeature)"
        "-[:HAS_COMMON_PREFIX_CODE]->(p)\n"
        "MATCH (parcel)-[:HAS_CENTROID_CODE]->"
        "(c:GeoSOTCentroidCode)\n"
        "MATCH (c)-[:FROM_SOURCE]->"
        "(:DataSource {name: $src})\n"
        "MATCH (c)-[:HAS_GEOMETRY_TYPE]->"
        "(:GeoFeatureType {type_name: 'Polygon'})\n"
        "RETURN DISTINCT parcel.main_code AS mc, "
        "p.prefix_code AS prefix"
    )

    # 优先走本地倒排索引 (与 PG pg_prewarm 等价对照). / Prefer local inverted index (counterpart to PG pg_prewarm).
    # 索引未注入时回退到 Cypher (cold-cache 对照模式). / If index not injected, fall back to Cypher (cold-cache control).
    cp_idx = get_common_prefix_index()

    for tier_start in range(0, len(prefixes), tier_size):
        batch = prefixes[tier_start:tier_start + tier_size]

        t0 = _t()
        if cp_idx is not None and cp_idx.has_source(parcel_source):
            rows = cp_idx.query_prefixes(batch)
        else:
            rows = [dict(r) for r in session.run(
                _CYPHER, prefixes=batch, src=parcel_source)]
        t_query_total += (_t() - t0) * 1000

        new_candidates = []
        for r in rows:
            mc = str(r["mc"]).strip().upper()
            if mc not in seen_mc:
                seen_mc.add(mc)
                r["mc"] = mc
                new_candidates.append(r)
                all_candidates.append(r)

        if not new_candidates:
            continue

        t0 = _t()
        if polygon_cache:
            for c in new_candidates:
                # bbox 预筛: 整数比较剔除盒外候选, 避免昂贵绕数法 PIP / bbox prefilter: integer compares drop out-of-box candidates before expensive winding PIP
                if not _point_in_bbox(
                        poi_xy, polygon_cache.get_bbox(c["mc"])):
                    continue
                polygon_xy = polygon_cache.get_polygon(c["mc"])
                if polygon_xy is not None and len(polygon_xy) >= 3:
                    n_decoded += 1
                    if _pip_single(poi_xy, polygon_xy):
                        pip_results.append({
                            "main_code": c["mc"],
                            "prefix": str(c.get("prefix", "")),
                            "n_vertices": int(len(polygon_xy)),
                        })
        else:
            codes = [c["mc"] for c in new_candidates]
            local_polys = _fetch_polygons_for_codes(
                session, codes)
            for c in new_candidates:
                polygon_xy = local_polys.get(c["mc"])
                if polygon_xy is not None and len(polygon_xy) >= 3:
                    # bbox 预筛 (即时计算包围盒) / bbox prefilter (compute bbox on the fly)
                    if not _point_in_bbox(poi_xy, _bbox_of(polygon_xy)):
                        continue
                    n_decoded += 1
                    if _pip_single(poi_xy, polygon_xy):
                        pip_results.append({
                            "main_code": c["mc"],
                            "prefix": str(c.get("prefix", "")),
                            "n_vertices": int(len(polygon_xy)),
                        })
        t_pip_total += (_t() - t0) * 1000

        if pip_results:
            break

    return {
        "candidates": all_candidates,
        "results": pip_results,
        "n_decoded": n_decoded,
        "t_query_ms": t_query_total,
        "t_pip_ms": t_pip_total,
    }


# ============================================================================
# 邻域投票策略 — 基于知识图谱 EAST_OF / NORTH_OF 空间推理
# ============================================================================

_DIGIT_XY = {"0": (0, 0), "1": (1, 0), "2": (0, 1), "3": (1, 1)}
_XY_DIGIT = {v: k for k, v in _DIGIT_XY.items()}


def _preload_neighbor_cache(session) -> Dict:
    """
    从知识图谱预加载 EAST_OF / NORTH_OF 邻居关系。
    Preload EAST_OF / NORTH_OF neighbor relations from the knowledge graph.

    缓存格式: (level, digit, direction) -> neighbor_digit
    Cache format: (level, digit, direction) -> neighbor_digit
    """
    cache: Dict[Tuple, str] = {}
    for level in range(1, 33):
        label = f"GeoSOTLevelDigit{level}"
        prop = "digit"
        for r in session.run(
            f"MATCH (a:`{label}`)-[:EAST_OF]->(b:`{label}`) "
            f"RETURN a.{prop} AS east, b.{prop} AS west"
        ):
            e, w = str(r["east"]), str(r["west"])
            cache[(level, w, "east")] = e
            cache[(level, e, "west")] = w
        for r in session.run(
            f"MATCH (a:`{label}`)-[:NORTH_OF]->(b:`{label}`) "
            f"RETURN a.{prop} AS north, b.{prop} AS south"
        ):
            n, s = str(r["north"]), str(r["south"])
            cache[(level, s, "north")] = n
            cache[(level, n, "south")] = s
    return cache


def _graph_find_neighbor(
    prefix: str,
    direction: str,
    nb_cache: Dict,
) -> Optional[str]:
    """
    通过知识图谱 EAST_OF/NORTH_OF 关系递归查找邻居前缀编码。
    Recursively find neighbor prefix codes via EAST_OF/NORTH_OF relations.

    同层有邻居时直接返回；跨父网格边界时递归上溯，
    找到父级邻居后再回到本层选择对侧子格。
    Return same-level neighbors directly; if crossing a parent border, recurse up,
    then pick the opposite child cell at this level.
    """
    body = prefix[1:]
    level = len(body)
    if level < 2:
        return None

    digit = body[-1]
    parent = prefix[:-1]

    nb = nb_cache.get((level, digit, direction))
    if nb is not None:
        return parent + nb

    parent_nb = _graph_find_neighbor(parent, direction, nb_cache)
    if parent_nb is None:
        return None

    x, y = _DIGIT_XY[digit]
    if direction == "east":
        child = _XY_DIGIT[(0, y)]
    elif direction == "west":
        child = _XY_DIGIT[(1, y)]
    elif direction == "north":
        child = _XY_DIGIT[(x, 0)]
    elif direction == "south":
        child = _XY_DIGIT[(x, 1)]
    else:
        return None
    return parent_nb + child


def _find_8_neighbors(
    prefix: str, nb_cache: Dict,
) -> Dict[str, str]:
    """
    通过知识图谱找 8 邻域前缀编码。
    Find 8-neighborhood prefix codes via the knowledge graph.

    4 基本方向直接查 EAST_OF/NORTH_OF；
    4 对角方向组合两次基本方向（先纵后横）。
    4 cardinal directions via EAST_OF/NORTH_OF;
    4 diagonals by composing two cardinal steps (vertical then horizontal).
    """
    result: Dict[str, str] = {}

    for d in ("east", "west", "north", "south"):
        nb = _graph_find_neighbor(prefix, d, nb_cache)
        if nb:
            result[d] = nb

    if "north" in result:
        ne = _graph_find_neighbor(result["north"], "east", nb_cache)
        if ne:
            result["northeast"] = ne
        nw = _graph_find_neighbor(result["north"], "west", nb_cache)
        if nw:
            result["northwest"] = nw
    if "south" in result:
        se = _graph_find_neighbor(result["south"], "east", nb_cache)
        if se:
            result["southeast"] = se
        sw = _graph_find_neighbor(result["south"], "west", nb_cache)
        if sw:
            result["southwest"] = sw

    return result


def _lcp2(a: str, b: str) -> str:
    """两个字符串的最长公共前缀。 / Longest common prefix of two strings."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return a[:i]
    return a[:n]


def _neighbor_voting_query(
    session,
    poi_code: str,
    poi_xy: Tuple[int, int],
    parcel_source: str = "Statutory Plan",
    polygon_cache: Optional[PlanPolygonCache] = None,
    nb_cache: Optional[Dict] = None,
    tier_size: int = 5,
) -> Dict:
    """
    邻域投票策略: 前缀递进 + 图推理 8 邻域 + 投票排序。
    Neighborhood voting: progressive prefixes + graph 8-neighborhood + vote ranking.

    Phase 1: 从最长前缀开始递进查找候选，PIP 测试
    Phase 1: progressive candidate search from longest prefix + PIP
    Phase 2: 未命中时，通过 EAST_OF/NORTH_OF 找 8 邻域并投票
    Phase 2: on miss, find 8-neighborhood via EAST_OF/NORTH_OF and vote
    """
    prefixes = [poi_code[:i] for i in range(len(poi_code), 1, -1)]

    all_candidates: List[Dict] = []
    pip_results: List[Dict] = []
    seen_mc: Set[str] = set()
    n_decoded = 0
    t_query = 0.0
    t_pip = 0.0
    method = "no_match"
    neighbors: Dict[str, str] = {}
    neighbor_lcps: List[str] = []
    deepest_prefix = ""

    _CYP = (
        "UNWIND $prefixes AS pf\n"
        "MATCH (p:GeoSOTCommonPrefixCode {prefix_code: pf})\n"
        "MATCH (parcel:SpatialFeature)"
        "-[:HAS_COMMON_PREFIX_CODE]->(p)\n"
        "MATCH (parcel)-[:HAS_CENTROID_CODE]->"
        "(c:GeoSOTCentroidCode)\n"
        "MATCH (c)-[:FROM_SOURCE]->"
        "(:DataSource {name: $src})\n"
        "MATCH (c)-[:HAS_GEOMETRY_TYPE]->"
        "(:GeoFeatureType {type_name: 'Polygon'})\n"
        "RETURN DISTINCT parcel.main_code AS mc, "
        "p.prefix_code AS prefix"
    )

    cp_idx_local = get_common_prefix_index()

    def _q(batch):
        nonlocal t_query
        t0 = _t()
        if cp_idx_local is not None and \
                cp_idx_local.has_source(parcel_source):
            rows = cp_idx_local.query_prefixes(batch)
        else:
            rows = [dict(r) for r in session.run(
                _CYP, prefixes=batch, src=parcel_source)]
        t_query += (_t() - t0) * 1000
        return rows

    def _dedup(rows):
        new = []
        for r in rows:
            mc = str(r["mc"]).strip().upper()
            if mc not in seen_mc:
                seen_mc.add(mc)
                r["mc"] = mc
                new.append(r)
                all_candidates.append(r)
        return new

    def _pip(cands):
        nonlocal n_decoded, t_pip
        hits = []
        t0 = _t()
        if polygon_cache:
            for c in cands:
                # bbox 预筛: 整数比较剔除盒外候选, 避免昂贵绕数法 PIP / bbox prefilter: integer compares drop out-of-box candidates before expensive winding PIP
                if not _point_in_bbox(
                        poi_xy, polygon_cache.get_bbox(c["mc"])):
                    continue
                poly_xy = polygon_cache.get_polygon(c["mc"])
                if poly_xy is not None and len(poly_xy) >= 3:
                    n_decoded += 1
                    if _pip_single(poi_xy, poly_xy):
                        hits.append({
                            "main_code": c["mc"],
                            "prefix": str(c.get("prefix", "")),
                            "n_vertices": int(len(poly_xy)),
                        })
        else:
            codes = [c["mc"] for c in cands]
            lp = _fetch_polygons_for_codes(session, codes)
            for c in cands:
                poly_xy = lp.get(c["mc"])
                if poly_xy is not None and len(poly_xy) >= 3:
                    # bbox 预筛 (即时计算包围盒) / bbox prefilter (compute bbox on the fly)
                    if not _point_in_bbox(poi_xy, _bbox_of(poly_xy)):
                        continue
                    n_decoded += 1
                    if _pip_single(poi_xy, poly_xy):
                        hits.append({
                            "main_code": c["mc"],
                            "prefix": str(c.get("prefix", "")),
                            "n_vertices": int(len(poly_xy)),
                        })
        t_pip += (_t() - t0) * 1000
        return hits

    # ── Phase 1: 递进前缀匹配 ── / ── Phase 1: progressive prefix match ──
    for tier_start in range(0, len(prefixes), tier_size):
        batch = prefixes[tier_start:tier_start + tier_size]
        new_cands = _dedup(_q(batch))
        if not new_cands:
            continue

        for c in new_cands:
            pf = str(c.get("prefix", ""))
            if len(pf) > len(deepest_prefix):
                deepest_prefix = pf

        hits = _pip(new_cands)
        pip_results.extend(hits)
        if pip_results:
            method = "prefix_direct"
            break
        break  # 有候选无 PIP 命中 → 转入邻域投票 / candidates but no PIP hit → neighborhood voting

    # ── Phase 2: 图推理 8 邻域 + 投票 ── / ── Phase 2: graph 8-neighborhood + voting ──
    if not pip_results and deepest_prefix and nb_cache:
        t0_nb = _t()
        neighbors = _find_8_neighbors(deepest_prefix, nb_cache)
        t_neighbor = (_t() - t0_nb) * 1000

        lcp_set: Set[str] = set()
        for nb_code in neighbors.values():
            lcp = _lcp2(poi_code, nb_code)
            if lcp and 1 < len(lcp) < len(deepest_prefix):
                lcp_set.add(lcp)
        neighbor_lcps = sorted(lcp_set, key=len, reverse=True)

        if neighbor_lcps:
            new_cands = _dedup(_q(neighbor_lcps))
            if new_cands:
                all_codes = [poi_code] + list(neighbors.values())
                for c in new_cands:
                    pf = str(c.get("prefix", ""))
                    c["vote"] = sum(
                        1 for code in all_codes
                        if code.startswith(pf))
                new_cands.sort(
                    key=lambda c: c.get("vote", 0),
                    reverse=True)

                hits = _pip(new_cands)
                pip_results.extend(hits)
                if pip_results:
                    method = "neighbor_voting"

    # ── Phase 3: 回退 cascade ── / ── Phase 3: fallback cascade ──
    if not pip_results:
        tried = {str(c.get("prefix", ""))
                 for c in all_candidates}
        tried.update(neighbor_lcps)
        remaining = [p for p in prefixes
                     if p not in tried
                     and len(p) < len(deepest_prefix)]

        for tier_start in range(0, len(remaining), tier_size):
            batch = remaining[tier_start:tier_start + tier_size]
            new_cands = _dedup(_q(batch))
            if not new_cands:
                continue
            hits = _pip(new_cands)
            pip_results.extend(hits)
            if pip_results:
                method = "cascade_fallback"
                break

    return {
        "candidates": all_candidates,
        "results": pip_results,
        "n_decoded": n_decoded,
        "t_query_ms": t_query,
        "t_pip_ms": t_pip,
        "method": method,
        "neighbors": neighbors,
        "neighbor_lcps": neighbor_lcps,
        "deepest_prefix": deepest_prefix,
    }


# ============================================================================
# 单条查询 / Single query
# ============================================================================

def run_single_query(
    session, pg_conn,
    poi_code: str,
    poi_source: str,
    parcel_source: str,
    compare_pg: bool,
    strategy: str = "prefix",
    polygon_cache: Optional[PlanPolygonCache] = None,
    nb_cache: Optional[Dict] = None,
):
    W = 76
    SEP = f"  {'─' * (W - 4)}"
    qcode = poi_code.strip().upper()

    print(f"\n{'=' * W}")
    print("GeoSOT 反向叠置查询 — 给定要素，找所属地块")
    print(f"{'=' * W}")
    print(f"  要素编码:   {qcode}")
    print(f"  要素来源:   {poi_source}")
    print(f"  地块来源:   {parcel_source}")
    print(f"  查询策略:   {strategy}")

    # ── Step 1: 获取要素坐标 (编码-only Z-order 映射) ── / ── Step 1: get feature coords (code-only Z-order map) ──
    # 计时阶段仅做 _code_to_xy 位运算; lon/lat 仅作为诊断列在计时之外解码. / Timed stage is only _code_to_xy bit ops; lon/lat decoded outside timing as diagnostics.
    print(f"\n{SEP}")
    print(f"  Step 1: 获取要素坐标 (Z-order 整数映射)")
    print(f"{SEP}")

    t0 = _t()
    try:
        # DMS 位打包 -> 线性欧式整数空间 (与多边形顶点同坐标系), 保证绕数法 PIP 正确 / DMS bit-pack → linear Euclidean int space (same frame as polygon verts) so winding PIP is correct
        poi_xy = _xy_dms_to_linear(
            np.asarray(_code_to_xy(qcode, _GRID_LEVEL), dtype=np.int64))
    except Exception as e:
        print(f"  [ERROR] 无法映射 {qcode}: {e}")
        return
    t_decode_poi = (_t() - t0) * 1000

    try:
        poi_lon, poi_lat = fast_code_to_center(qcode)
    except Exception:
        poi_lon, poi_lat = float("nan"), float("nan")

    print(f"  Z-order:    {poi_xy}")
    print(f"  (诊断 lon/lat, 不计入时延): "
          f"({poi_lon:.8f}, {poi_lat:.8f})")
    print(f"  耗时:       {t_decode_poi:.4f} ms")

    # ── Step 2: 递进前缀匹配 + PIP ── / ── Step 2: progressive prefix match + PIP ──
    print(f"\n{SEP}")
    print(f"  Step 2: 递进匹配 + 编码-only 绕数法 PIP (策略: {strategy})")
    print(f"{SEP}")

    nb_method = ""
    nb_neighbors: Dict[str, str] = {}
    nb_lcps: List[str] = []
    nb_deepest = ""

    if strategy == "neighbor":
        nv = _neighbor_voting_query(
            session, qcode, poi_xy,
            parcel_source, polygon_cache, nb_cache)
        candidates = nv["candidates"]
        results = nv["results"]
        n_decoded = nv["n_decoded"]
        t_query = nv["t_query_ms"]
        t_pip = nv["t_pip_ms"]
        nb_method = nv.get("method", "")
        nb_neighbors = nv.get("neighbors", {})
        nb_lcps = nv.get("neighbor_lcps", [])
        nb_deepest = nv.get("deepest_prefix", "")
    elif strategy == "prefix":
        cascade = _cascade_prefix_query(
            session, qcode, poi_xy,
            parcel_source, polygon_cache)
        candidates = cascade["candidates"]
        results = cascade["results"]
        n_decoded = cascade["n_decoded"]
        t_query = cascade["t_query_ms"]
        t_pip = cascade["t_pip_ms"]
    else:
        candidates, t_query = _query_parcels(
            session, qcode, parcel_source, strategy)
        t0 = _t()
        results: List[Dict] = []
        n_decoded = 0
        if not polygon_cache and candidates:
            cand_codes = [str(c["mc"]).strip().upper()
                          for c in candidates]
            _local_cache = _fetch_polygons_for_codes(
                session, cand_codes)
        else:
            _local_cache = None
        for cand in candidates:
            mc = str(cand["mc"]).strip().upper()
            polygon_xy = None
            if polygon_cache:
                polygon_xy = polygon_cache.get_polygon(mc)
            elif _local_cache:
                polygon_xy = _local_cache.get(mc)
            if polygon_xy is None or len(polygon_xy) < 3:
                continue
            # bbox 预筛: 盒外候选必然在多边形外, 跳过昂贵绕数法 PIP / bbox prefilter: out-of-box candidates are outside the polygon; skip expensive winding PIP
            bbox = (polygon_cache.get_bbox(mc) if polygon_cache
                    else _bbox_of(polygon_xy))
            if not _point_in_bbox(poi_xy, bbox):
                continue
            n_decoded += 1
            if _pip_single(poi_xy, polygon_xy):
                results.append({
                    "main_code": mc,
                    "prefix": str(cand.get("prefix", "")),
                    "n_vertices": int(len(polygon_xy)),
                })
        t_pip = (_t() - t0) * 1000

    print(f"  候选地块:   {len(candidates)} 个")
    print(f"  查询耗时:   {t_query:.1f} ms")
    if candidates:
        for c in candidates[:10]:
            print(f"    {c['mc']}  (前缀: {c['prefix']})")
        if len(candidates) > 10:
            print(f"    ... 还有 {len(candidates) - 10} 个")

    print(f"  解码+PIP:   {n_decoded} 个多边形, "
          f"{len(results)} 命中, {t_pip:.3f} ms")

    if strategy == "neighbor" and nb_deepest:
        print(f"\n  [邻域投票详情]")
        print(f"    最深匹配前缀:  {nb_deepest}  "
              f"(level {len(nb_deepest) - 1})")
        print(f"    命中方法:      {nb_method}")
        if nb_neighbors:
            print(f"    8 邻域 ({len(nb_neighbors)} 个):")
            _dir_cn = {
                "east": "东", "west": "西",
                "north": "北", "south": "南",
                "northeast": "东北", "northwest": "西北",
                "southeast": "东南", "southwest": "西南",
            }
            for d, code in sorted(nb_neighbors.items()):
                cn = _dir_cn.get(d, d)
                lcp = _lcp2(qcode, code)
                print(f"      {cn:>4s}: {code}  "
                      f"(LCP level {len(lcp) - 1})")
        if nb_lcps:
            print(f"    邻域 LCP 前缀 ({len(nb_lcps)} 个):")
            for lp in nb_lcps[:5]:
                print(f"      {lp}  (level {len(lp) - 1})")

    # ── 总结 ── / ── Summary ──
    t_total = t_decode_poi + t_query + t_pip
    print(f"\n  图推理总耗时: {t_total:.1f} ms")
    print(f"    ① 解码要素坐标:     {t_decode_poi:>8.4f} ms")
    print(f"    ② 前缀匹配候选:     {t_query:>8.1f} ms  "
          f"({len(candidates)} 候选)")
    print(f"    ③ 解码+PIP:         {t_pip:>8.3f} ms  "
          f"({len(results)} 命中)")

    # ── Step 4: PostGIS 对比 ── / ── Step 4: PostGIS comparison ──
    if compare_pg and pg_conn:
        print(f"\n{SEP}")
        print(f"  Step 4: PostGIS ST_Contains")
        print(f"{SEP}")

        pt = SOURCE_TO_PG_TABLE.get(poi_source, "")
        tt = SOURCE_TO_PG_TABLE.get(parcel_source, "")
        if pt and tt:
            pg_results, t_pg = _postgis_reverse_query(
                pg_conn, qcode, pt, tt)
            print(f"  找到地块:   {len(pg_results)} 个")
            print(f"  PostGIS 耗时: {t_pg:.1f} ms")

            gr_set = {r["main_code"] for r in results}
            pg_set = set(pg_results)
            common = gr_set & pg_set
            gr_only = gr_set - pg_set
            pg_only = pg_set - gr_set

            print(f"\n{SEP}")
            print(f"  结果对比")
            print(f"{SEP}")
            print(f"  图推理:    {len(gr_set)} 个")
            print(f"  PostGIS:   {len(pg_set)} 个")
            print(f"  共同:      {len(common)} 个")
            print(f"  仅图推理:  {len(gr_only)} 个")
            print(f"  仅 PostGIS: {len(pg_only)} 个")

            print(f"\n  耗时对比:")
            print(f"    图推理:  {t_total:>10.1f} ms")
            print(f"    PostGIS: {t_pg:>10.1f} ms")
            if t_total > 0:
                print(f"    比率:    {t_pg / t_total:.2f}x")

    # ── 结果展示 ── / ── Result display ──
    if results:
        print(f"\n{SEP}")
        print(f"  所属地块 ({len(results)} 个)")
        print(f"{SEP}")
        for i, r in enumerate(results):
            print(f"  {i + 1:>4d}. {r['main_code']}  "
                  f"(前缀: {r['prefix']}, "
                  f"顶点: {r['n_vertices']})")
    else:
        print(f"\n  该要素不属于任何 {parcel_source} 地块。")

    print(f"\n{'=' * W}")


# ============================================================================
# 批量测试 / Batch test
# ============================================================================

CSV_FIELDS = [
    # 基本信息 / Basic info
    "poi_idx", "poi_code", "poi_lon", "poi_lat",
    # 图推理结果 / Graph-reasoning results
    "n_candidates", "n_decoded", "pip_hits",
    "gr_matched_codes",
    # 分步耗时 (ms) / Step timings (ms)
    "t_decode_poi_ms", "t_query_ms", "t_pip_ms",
    "t_gr_total_ms",
    # 邻域投票详情 / Neighborhood-voting details
    "method", "deepest_prefix", "deepest_prefix_level",
    "n_neighbors", "n_neighbor_lcps",
    # PostGIS 对比 / PostGIS comparison
    "pg_count", "pg_matched_codes", "t_pg_ms",
    "speedup",
    # 精度指标 / Accuracy metrics
    "recall", "precision", "f1", "match",
    "gr_only_count", "gr_only_codes",
    "pg_only_count", "pg_only_codes",
    # 元信息 / Metadata
    "strategy",
]


def run_batch(
    session, pg_conn,
    poi_codes: List[str],
    poi_source: str,
    parcel_source: str,
    compare_pg: bool,
    strategy: str = "prefix",
    output_csv: str = "",
    polygon_cache: Optional[PlanPolygonCache] = None,
    nb_cache: Optional[Dict] = None,
    hit_only: int = 0,
) -> List[Dict]:
    W = 76
    SEP = f"  {'─' * (W - 4)}"
    n = len(poi_codes)
    target_hits = hit_only

    pt = SOURCE_TO_PG_TABLE.get(poi_source, "")
    tt = SOURCE_TO_PG_TABLE.get(parcel_source, "")

    rows_out: List[Dict] = []
    t_start = _t()
    bar_update = max(n // 100, 1)
    n_scanned = 0

    for idx, pc in enumerate(poi_codes):
        if target_hits > 0 and len(rows_out) >= target_hits:
            break

        if idx % bar_update == 0:
            elapsed = _t() - t_start
            if target_hits > 0:
                pct = len(rows_out) / target_hits
                label = (f"{len(rows_out):,}/{target_hits:,} hits  "
                         f"scanned {idx:,}")
            else:
                pct = idx / n if n else 0
                label = f"{idx:,}/{n:,}"
            rate = idx / elapsed if elapsed > 0.001 else 0
            eta_remain = ((target_hits - len(rows_out))
                          / (len(rows_out) / elapsed)
                          if target_hits > 0 and len(rows_out) > 0
                          else (n - idx) / rate if rate > 0 else 0)
            sys.stdout.write(
                f"\r  [{int(30 * min(pct, 1)) * '#'}"
                f"{(30 - int(30 * min(pct, 1))) * '.'}] "
                f"{min(pct, 1) * 100:5.1f}%  {label}  "
                f"{rate:,.1f}/s  ETA {eta_remain:.0f}s")
            sys.stdout.flush()

        n_scanned = idx + 1

        qcode = pc.strip().upper()

        # 1. 编码-only Z-order 映射 (计时); lon/lat 仅作 CSV 诊断列, 不计时 / 1. Code-only Z-order map (timed); lon/lat only as CSV diagnostics, not timed
        t0 = _t()
        try:
            # DMS 位打包 -> 线性欧式整数空间 (与多边形顶点同坐标系), 保证绕数法 PIP 正确 / DMS bit-pack → linear Euclidean int space (same frame as polygon verts) so winding PIP is correct
            poi_xy = _xy_dms_to_linear(
                np.asarray(_code_to_xy(qcode, _GRID_LEVEL), dtype=np.int64))
        except Exception:
            rows_out.append({k: "" for k in CSV_FIELDS})
            rows_out[-1].update({
                "poi_idx": idx, "poi_code": qcode,
                "n_candidates": 0, "n_decoded": 0,
                "pip_hits": 0, "t_gr_total_ms": 0,
                "strategy": strategy, "method": "decode_error",
            })
            continue
        t_decode_poi = (_t() - t0) * 1000

        try:
            poi_lon, poi_lat = fast_code_to_center(qcode)
        except Exception:
            poi_lon, poi_lat = float("nan"), float("nan")

        # 2+3. 递进前缀匹配 + 编码-only 绕数法 PIP (命中即停) / 2+3. Progressive prefix match + code-only winding PIP (stop on hit)
        nb_method = ""
        nb_deepest = ""
        nb_n_neighbors = 0
        nb_n_lcps = 0

        if strategy == "neighbor":
            nv = _neighbor_voting_query(
                session, qcode, poi_xy,
                parcel_source, polygon_cache, nb_cache)
            candidates = nv["candidates"]
            gr_results = [r["main_code"]
                          for r in nv["results"]]
            n_decoded = nv["n_decoded"]
            t_query = nv["t_query_ms"]
            t_pip = nv["t_pip_ms"]
            nb_method = nv.get("method", "")
            nb_deepest = nv.get("deepest_prefix", "")
            nb_n_neighbors = len(nv.get("neighbors", {}))
            nb_n_lcps = len(nv.get("neighbor_lcps", []))
        elif strategy == "prefix":
            cascade = _cascade_prefix_query(
                session, qcode, poi_xy,
                parcel_source, polygon_cache)
            candidates = cascade["candidates"]
            gr_results = [r["main_code"]
                          for r in cascade["results"]]
            n_decoded = cascade["n_decoded"]
            t_query = cascade["t_query_ms"]
            t_pip = cascade["t_pip_ms"]
            nb_method = "cascade"
        else:
            candidates, t_query = _query_parcels(
                session, qcode, parcel_source, strategy)
            t0 = _t()
            gr_results: List[str] = []
            n_decoded = 0
            for cand in candidates:
                mc = str(cand["mc"]).strip().upper()
                polygon_xy = None
                if polygon_cache:
                    polygon_xy = polygon_cache.get_polygon(mc)
                if polygon_xy is None or len(polygon_xy) < 3:
                    continue
                n_decoded += 1
                if _pip_single(poi_xy, polygon_xy):
                    gr_results.append(mc)
            t_pip = (_t() - t0) * 1000
            nb_method = "located_in"

        t_gr_total = t_decode_poi + t_query + t_pip
        gr_set = set(gr_results)

        # 4. PostGIS 对比 / 4. PostGIS comparison
        pg_count = ""
        pg_matched_str = ""
        t_pg_val = ""
        recall_v = ""
        precision_v = ""
        f1_v = ""
        match_flag = ""
        gr_only_count = ""
        gr_only_str = ""
        pg_only_count = ""
        pg_only_str = ""
        speedup_v = ""

        if compare_pg and pg_conn and pt and tt:
            pg_results, t_pg = _postgis_reverse_query(
                pg_conn, qcode, pt, tt)
            pg_set = set(pg_results)
            pg_count = len(pg_set)
            pg_matched_str = ";".join(sorted(pg_set))
            t_pg_val = round(t_pg, 3)
            common = gr_set & pg_set
            gr_only = gr_set - pg_set
            pg_only = pg_set - gr_set
            gr_only_count = len(gr_only)
            gr_only_str = ";".join(sorted(gr_only))
            pg_only_count = len(pg_only)
            pg_only_str = ";".join(sorted(pg_only))
            recall_v = (round(len(common) / len(pg_set), 6)
                        if pg_set else "")
            precision_v = (round(len(common) / len(gr_set), 6)
                           if gr_set else "")
            if (isinstance(recall_v, float)
                    and isinstance(precision_v, float)
                    and (recall_v + precision_v) > 0):
                f1_v = round(
                    2 * recall_v * precision_v
                    / (recall_v + precision_v), 6)
            match_flag = 1 if gr_set == pg_set else 0
            if t_gr_total > 0:
                speedup_v = round(t_pg / t_gr_total, 4)

        if hit_only > 0 and len(gr_results) == 0:
            continue

        rows_out.append({
            "poi_idx": idx,
            "poi_code": qcode,
            "poi_lon": round(poi_lon, 8),
            "poi_lat": round(poi_lat, 8),
            "n_candidates": len(candidates),
            "n_decoded": n_decoded,
            "pip_hits": len(gr_results),
            "gr_matched_codes": (
                ";".join(sorted(gr_set)) if gr_set else ""),
            "t_decode_poi_ms": round(t_decode_poi, 4),
            "t_query_ms": round(t_query, 3),
            "t_pip_ms": round(t_pip, 4),
            "t_gr_total_ms": round(t_gr_total, 3),
            "method": nb_method,
            "deepest_prefix": nb_deepest,
            "deepest_prefix_level": (
                len(nb_deepest) - 1 if nb_deepest else ""),
            "n_neighbors": nb_n_neighbors,
            "n_neighbor_lcps": nb_n_lcps,
            "pg_count": pg_count,
            "pg_matched_codes": pg_matched_str,
            "t_pg_ms": t_pg_val,
            "speedup": speedup_v,
            "recall": recall_v,
            "precision": precision_v,
            "f1": f1_v,
            "match": match_flag,
            "gr_only_count": gr_only_count,
            "gr_only_codes": gr_only_str,
            "pg_only_count": pg_only_count,
            "pg_only_codes": pg_only_str,
            "strategy": strategy,
        })

    if target_hits > 0:
        sys.stdout.write(
            f"\r  [{'#' * 30}] 100.0%  "
            f"{len(rows_out):,}/{target_hits:,} hits  "
            f"scanned {n_scanned:,}\n")
    else:
        sys.stdout.write(
            f"\r  [{'#' * 30}] 100.0%  {n:,}/{n:,}\n")
    sys.stdout.flush()

    # CSV 输出 / CSV output
    if output_csv and rows_out:
        with open(output_csv, "w", newline="",
                  encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in rows_out:
                writer.writerow(
                    {k: r.get(k, "") for k in CSV_FIELDS})
        print(f"\n  CSV 已保存: {output_csv}  "
              f"({len(rows_out)} 行, "
              f"{len(CSV_FIELDS)} 列)")

    # ================================================================
    # 统计汇总 / Statistics summary
    # ================================================================
    valid_rows = [r for r in rows_out
                  if r.get("poi_lon", "") != ""]
    valid = len(valid_rows)
    print(f"\n{'=' * W}")
    print(f"  反向叠置查询批量测试汇总  (N={valid})")
    print(f"{'=' * W}")

    if valid == 0:
        print("  无有效结果。")
        return rows_out

    def _pct(vals, p):
        return float(np.percentile(vals, p))

    def _stat5(vals):
        a = np.array(vals, dtype=np.float64)
        return (f"Mean={np.mean(a):>10.3f}  "
                f"Med={np.median(a):>10.3f}  "
                f"P5={_pct(a, 5):>10.3f}  "
                f"P95={_pct(a, 95):>10.3f}  "
                f"Max={np.max(a):>10.3f}")

    def _stat3(vals):
        return (f"Mean={statistics.mean(vals):>10.3f}  "
                f"Med={statistics.median(vals):>10.3f}  "
                f"Max={max(vals):>10.3f}")

    # ── 候选与命中统计 ── / ── Candidate & hit stats ──
    cand_list = [r["n_candidates"] for r in valid_rows]
    decoded_list = [r["n_decoded"] for r in valid_rows]
    hits_list = [r["pip_hits"] for r in valid_rows]

    print(f"\n  ── 候选与命中统计 ──")
    print(SEP)
    print(f"    候选地块数: {_stat3(cand_list)}")
    print(f"    解码成功数: {_stat3(decoded_list)}")
    print(f"    PIP 命中数: {_stat3(hits_list)}")

    n_has_hit = sum(1 for h in hits_list if h > 0)
    print(f"    有命中的查询: {n_has_hit}/{valid} "
          f"({n_has_hit / valid:.1%})")

    # ── 分步耗时 ── / ── Step timings ──
    t_decode_list = [r["t_decode_poi_ms"] for r in valid_rows]
    t_query_list = [r["t_query_ms"] for r in valid_rows]
    t_pip_list = [r["t_pip_ms"] for r in valid_rows]
    t_total_list = [r["t_gr_total_ms"] for r in valid_rows]

    print(f"\n  ── 图推理分步耗时 (ms) ──")
    print(SEP)
    print(f"    ① 解码要素坐标:     {_stat5(t_decode_list)}")
    print(f"    ② 前缀匹配候选:     {_stat5(t_query_list)}")
    print(f"    ③ 解码+PIP:         {_stat5(t_pip_list)}")
    print(f"    合计:               {_stat5(t_total_list)}")

    mean_total = statistics.mean(t_total_list)
    if mean_total > 0:
        print(f"\n    各步骤平均占比:")
        for label, vals in [
            ("解码坐标", t_decode_list),
            ("前缀匹配", t_query_list),
            ("解码+PIP", t_pip_list),
        ]:
            pct_ = statistics.mean(vals) / mean_total * 100
            print(f"      {label}: {pct_:>6.2f}%")

    # ── PostGIS 对比 ── / ── PostGIS comparison ──
    if compare_pg:
        pg_rows = [r for r in valid_rows
                   if r.get("t_pg_ms", "") != ""]
        if pg_rows:
            t_pg_list = [r["t_pg_ms"] for r in pg_rows]
            pg_count_list = [r["pg_count"] for r in pg_rows]
            gr_hits_pg = [r["pip_hits"] for r in pg_rows]

            print(f"\n  ── PostGIS 耗时 (ms) ──")
            print(SEP)
            print(f"    ST_Contains:        {_stat5(t_pg_list)}")

            speedup_list = [r["speedup"] for r in pg_rows
                            if isinstance(r.get("speedup"), float)]
            if speedup_list:
                print(f"\n  ── 加速比 (PostGIS / 图推理) ──")
                print(SEP)
                print(f"    {_stat5(speedup_list)}")

            print(f"\n  ── 命中数量对比 ──")
            print(SEP)
            hdr = f"    {'指标':>10s}  {'图推理':>12s}  {'PostGIS':>12s}"
            print(hdr)
            print(f"    {'Mean':>10s}  "
                  f"{statistics.mean(gr_hits_pg):>12.1f}  "
                  f"{statistics.mean(pg_count_list):>12.1f}")
            print(f"    {'Median':>10s}  "
                  f"{statistics.median(gr_hits_pg):>12.0f}  "
                  f"{statistics.median(pg_count_list):>12.0f}")

            recall_vals = [r["recall"] for r in pg_rows
                           if isinstance(r.get("recall"), float)]
            precision_vals = [r["precision"] for r in pg_rows
                              if isinstance(r.get("precision"), float)]
            f1_vals = [r["f1"] for r in pg_rows
                       if isinstance(r.get("f1"), float)]

            print(f"\n  ── 精度指标 (vs PostGIS 基准) ──")
            print(SEP)
            if recall_vals:
                print(f"    召回率:  Mean="
                      f"{statistics.mean(recall_vals):.4%}  "
                      f"Med={statistics.median(recall_vals):.4%}  "
                      f"Min={min(recall_vals):.4%}")
            if precision_vals:
                print(f"    精确率:  Mean="
                      f"{statistics.mean(precision_vals):.4%}  "
                      f"Med={statistics.median(precision_vals):.4%}  "
                      f"Min={min(precision_vals):.4%}")
            if f1_vals:
                print(f"    F1:      Mean="
                      f"{statistics.mean(f1_vals):.4%}  "
                      f"Med={statistics.median(f1_vals):.4%}  "
                      f"Min={min(f1_vals):.4%}")

            match_rows = [r for r in pg_rows
                          if r.get("match", "") != ""]
            if match_rows:
                n_match = sum(1 for r in match_rows
                              if r["match"] == 1)
                print(f"    完全一致: "
                      f"{n_match}/{len(match_rows)} "
                      f"({n_match / len(match_rows):.2%})")

    # ── 论文引用摘要 ── / ── Paper citation summary ──
    print(f"\n  ── 论文引用摘要 ──")
    print(SEP)
    print(f"    查询总数:     {valid}")
    print(f"    查询策略:     {strategy}")
    print(f"    Mean 总耗时:  {statistics.mean(t_total_list):.2f} ms")
    print(f"    Med  总耗时:  {statistics.median(t_total_list):.2f} ms")
    if compare_pg:
        pg_rows2 = [r for r in valid_rows
                    if r.get("t_pg_ms", "") != ""]
        if pg_rows2:
            tp = [r["t_pg_ms"] for r in pg_rows2]
            tg = [r["t_gr_total_ms"] for r in pg_rows2]
            print(f"    Mean PostGIS:  {statistics.mean(tp):.2f} ms")
            if statistics.mean(tg) > 0:
                print(f"    Mean 加速比:   "
                      f"{statistics.mean(tp) / statistics.mean(tg):.2f}x")

    print(f"\n{'=' * W}")
    return rows_out


# ============================================================================
# 主流程 / Main flow
# ============================================================================

def _read_codes_from_csv(path: str) -> List[str]:
    """读取先前 run 的 CSV 中 poi_code 列 (保持顺序与重复), 复现同一查询集。
Read poi_code column from a prior-run CSV (keep order/duplicates) to reproduce the same query set.

    用于 --codes-from-csv: warm 与 raw 喂同一份固定 POI 列表。
    For --codes-from-csv: feed warm and raw the same fixed POI list.
    """
    codes: List[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            c = (r.get("poi_code") or "").strip().upper()
            if c:
                codes.append(c)
    return codes


def _fetch_poi_codes(
    session, pg_conn, poi_source: str,
) -> List[str]:
    """获取指定 source 的所有要素编码。 / Get all feature codes for a given source."""
    pt = SOURCE_TO_PG_TABLE.get(poi_source, "")
    if pg_conn and pt:
        cur = pg_conn.cursor()
        cur.execute(f"SELECT main_code FROM {pt}")
        codes = [
            str(r[0]).strip().upper()
            for r in cur.fetchall()
            if r[0] and is_valid_g0_code(
                str(r[0]).strip().upper())]
        cur.close()
        return codes

    code_rows = [dict(r) for r in session.run(
        "MATCH (c:GeoSOTCentroidCode)"
        "-[:FROM_SOURCE]->(:DataSource {name: $src})\n"
        "RETURN c.main_code AS mc",
        src=poi_source)]
    return [
        str(r["mc"]).strip().upper()
        for r in code_rows
        if is_valid_g0_code(str(r["mc"]).strip().upper())]


def main():
    single_code: Optional[str] = None
    poi_source = "POI"
    parcel_source = "Statutory Plan"
    batch_n = 0
    run_all = False
    seed = 42
    no_postgis = False
    output_csv = ""
    strategy = "prefix"
    hit_only = 0
    codes_from_csv = ""

    i = 0
    args = sys.argv[1:]
    while i < len(args):
        if args[i] == "--code" and i + 1 < len(args):
            single_code = args[i + 1]; i += 2
        elif args[i] == "--poi-source" and i + 1 < len(args):
            poi_source = args[i + 1]; i += 2
        elif args[i] == "--parcel-source" and i + 1 < len(args):
            parcel_source = args[i + 1]; i += 2
        elif args[i] == "--batch" and i + 1 < len(args):
            batch_n = int(args[i + 1]); i += 2
        elif args[i] == "--all":
            run_all = True; i += 1
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1]); i += 2
        elif args[i] == "--no-postgis":
            no_postgis = True; i += 1
        elif args[i] == "--hit-only" and i + 1 < len(args):
            hit_only = int(args[i + 1]); i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_csv = args[i + 1]; i += 2
        elif args[i] == "--strategy" and i + 1 < len(args):
            strategy = args[i + 1]; i += 2
        elif args[i] == "--codes-from-csv" and i + 1 < len(args):
            codes_from_csv = args[i + 1]; i += 2
        else:
            i += 1

    compare_pg = not no_postgis

    if not single_code and batch_n <= 0 and not run_all and not codes_from_csv:
        print("用法:")
        print("  python geosot_reverse_query.py "
              "--code G0xxx")
        print("  python geosot_reverse_query.py "
              "--batch 50 --seed 42")
        print("  python geosot_reverse_query.py "
              "--all --output result.csv")
        print()
        print("参数:")
        print("  --code CODE            要素的 main_code")
        print(f"  --poi-source SRC       要素 source "
              f"(默认: {poi_source})")
        print(f"  --parcel-source SRC    地块 source "
              f"(默认: {parcel_source})")
        print("  --batch N              批量测试 N 个")
        print("  --all                  遍历所有要素")
        print("  --output FILE          CSV 输出路径")
        print("  --no-postgis           不对比 PostGIS")
        print("  --hit-only N           仅保留命中记录，"
              "补足至 N 条")
        print("  --strategy STR         查询策略: "
              "prefix (默认) / located_in")
        print("  --codes-from-csv FILE  从已有 CSV 的 poi_code 列读取固定"
              "查询集 (与 raw 对齐同一批 POI)")
        return

    mode_desc = ("单条查询" if single_code
                 else "全部遍历" if run_all
                 else f"固定查询集 ({codes_from_csv})" if codes_from_csv
                 else f"随机采样 {batch_n} 个")

    W = 76
    print("=" * W)
    print("GeoSOT 反向叠置查询 — 给定要素，找所属地块")
    print("=" * W)
    print(f"  模式:       {mode_desc}")
    print(f"  要素来源:   {poi_source}")
    print(f"  地块来源:   {parcel_source}")
    print(f"  查询策略:   {strategy}")
    print(f"  PostGIS 对比: {'是' if compare_pg else '否'}")
    if hit_only > 0:
        print(f"  CSV 模式:   仅命中记录 (目标 {hit_only} 条)")
    if output_csv:
        print(f"  输出文件:   {output_csv}")

    # [1] 连接数据库 / [1] Connect databases
    print(f"\n[1] 连接数据库 ...")
    t0 = time.perf_counter()
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    session = driver.session(database=NEO4J_DATABASE)
    session.run("RETURN 1").consume()
    t_neo4j = (time.perf_counter() - t0) * 1000
    print(f"  Neo4j:   {NEO4J_URI} / {NEO4J_DATABASE}  "
          f"({t_neo4j:.0f} ms)")

    pg_conn = None
    if compare_pg:
        t0 = time.perf_counter()
        pg_conn = psycopg2.connect(**PG_CONFIG)
        t_pg = (time.perf_counter() - t0) * 1000
        print(f"  PostGIS: {PG_CONFIG['host']}:{PG_CONFIG['port']}"
              f"/{PG_CONFIG['dbname']}  ({t_pg:.0f} ms)")

        # ---- 公平性 setup: PG 预热 (pg_prewarm + ST_Contains throwaway) ---- / ---- fairness setup: PG warmup (pg_prewarm + ST_Contains throwaway) ----
        # 论文 Methodology 应披露: 与 GR 端 PlanPolygonCache (内存常驻) / Methodology should disclose: counterpart to GR PlanPolygonCache (RAM-resident);
        # 等价对照, PG 这边把 ST_Contains 涉及的表 + GiST + B-tree 全部 / on PG, pg_prewarm all tables + GiST + B-tree used by ST_Contains
        # pg_prewarm 进 shared_buffers, 并跑一条 ST_Contains 形态的 / into shared_buffers, and run one ST_Contains-shaped
        # dummy SQL 热 plan cache. / dummy SQL to warm the plan cache.
        from bench_setup import prepare_pg_warm
        used_tables = sorted({
            tbl for tbl in (
                SOURCE_TO_PG_TABLE.get(poi_source),
                SOURCE_TO_PG_TABLE.get(parcel_source),
            ) if tbl
        })
        pt_w = SOURCE_TO_PG_TABLE.get(poi_source, "")
        tt_w = SOURCE_TO_PG_TABLE.get(parcel_source, "")
        throwaway_sqls = [
            (f"SELECT main_code FROM {tbl} LIMIT 50", None)
            for tbl in used_tables
        ]
        if pt_w and tt_w:
            throwaway_sqls.append((
                f"SELECT p.main_code FROM {tt_w} p, {pt_w} t "
                f"WHERE t.main_code = (SELECT main_code FROM {pt_w} LIMIT 1) "
                f"AND ST_Contains(p.geometry, t.geometry) LIMIT 1",
                None,
            ))
        prepare_pg_warm(pg_conn, used_tables, throwaway_sqls)

    # 预加载邻域缓存 (neighbor 策略需要) / Preload neighborhood cache (needed by neighbor strategy)
    nb_cache = None
    if strategy == "neighbor":
        print(f"\n[1.5] 预加载 EAST_OF/NORTH_OF 邻居关系 ...")
        t0 = time.perf_counter()
        nb_cache = _preload_neighbor_cache(session)
        t_nb = (time.perf_counter() - t0) * 1000
        print(f"  缓存: {len(nb_cache):,} 条关系, "
              f"耗时 {t_nb:.0f} ms")

    # [2] 查询 / [2] Query
    if single_code:
        run_single_query(
            session, pg_conn,
            single_code, poi_source, parcel_source,
            compare_pg, strategy=strategy,
            nb_cache=nb_cache)
    elif run_all or batch_n > 0 or codes_from_csv:
        # 预加载多边形缓存 / Preload polygon cache
        print(f"\n[2] 预加载 {parcel_source} 多边形 ...")
        pc = PlanPolygonCache(session, verbose=True)
        t0 = time.perf_counter()
        n_plans = pc.preload(parcel_source)
        t_pp = (time.perf_counter() - t0) * 1000
        print(f"    合计: {t_pp:.0f} ms, "
              f"{n_plans:,} 个地块已就绪")

        # 预加载 plan parcel 共有前缀本地倒排索引 / Preload plan-parcel common-prefix local inverted index
        # (与 PG pg_prewarm + GiST 等价对照: setup 阶段一次性 O(N) / (counterpart to PG pg_prewarm + GiST: one-time O(N) load at setup;
        # 装载, 单条查询 O(1) hash 命中, 取代每条 POI 都打一次 / O(1) hash hit per query, replacing a Cypher prefix round-trip
        # Cypher prefix 查询的高频网络往返) / on every POI)
        print(f"\n[2.5] 预加载 {parcel_source} 共有前缀倒排索引 ...")
        cp_idx = CommonPrefixIndex(
            session, parcel_source=parcel_source, verbose=True)
        if cp_idx.is_loaded():
            set_common_prefix_index(cp_idx)
        else:
            print(f"    [警告] 未能加载共有前缀索引, "
                  f"回退到 Cypher 查询路径")

        # 获取要素列表 / Fetch feature list
        if codes_from_csv:
            # 固定查询集: 精确复现既有 run (如 raw_reverse_overlay.csv), / Fixed query set: exactly reproduce a prior run (e.g. raw_reverse_overlay.csv),
            # 使 warm 与 raw 评估严格相同的一批 POI; 不做抽样, 不依赖 / so warm and raw evaluate the same POI batch; no sampling; no reliance on
            # (不确定的) SELECT 行序。 / (unstable) SELECT row order.
            print(f"\n[3] 读取固定 POI 查询集: {codes_from_csv} ...")
            poi_codes = _read_codes_from_csv(codes_from_csv)
            if not poi_codes:
                print(f"    [错误] 未从 {codes_from_csv} 读到 poi_code, 退出")
                return
            print(f"    固定查询集: {len(poi_codes):,} 个 (与源 CSV 一致)")
        else:
            print(f"\n[3] 获取 {poi_source} 要素列表 ...")
            all_codes = _fetch_poi_codes(
                session, pg_conn, poi_source)
            print(f"    总数: {len(all_codes):,}")

            if run_all:
                poi_codes = all_codes
            elif hit_only > 0:
                random.seed(seed)
                random.shuffle(all_codes)
                poi_codes = all_codes
                print(f"    目标命中: {hit_only} 条 (从 "
                      f"{len(all_codes):,} 个中采样)")
            else:
                random.seed(seed)
                poi_codes = random.sample(
                    all_codes,
                    min(batch_n, len(all_codes)))
                print(f"    采样: {len(poi_codes)} 个")

        if not output_csv:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            n_label = hit_only if hit_only > 0 else len(poi_codes)
            output_csv = (
                f"reverse_query_{strategy}"
                f"_N{n_label}_{ts}.csv")

        # ---- 公平性 setup: 双侧 throwaway ---- / ---- fairness setup: dual-side throwaway ----
        # 与距离 / 正向叠置实验保持一致: 跑前 5 条真实 query 丢弃,
        # 同时热化 GR 端 (前缀 Cypher 候选 + Numba JIT PIP 编译) 与 / Warm GR (prefix Cypher candidates + Numba JIT PIP compile) and
        # PG 端 (ST_Contains plan cache + GiST/B-tree page cache), / PG (ST_Contains plan cache + GiST/B-tree page cache) together,
        # 之后才进入正式计时阶段. 论文 Methodology 段需披露此协议. / then enter the timed phase. Methodology must disclose this protocol.
        n_throwaway = min(5, len(poi_codes))
        if n_throwaway > 0:
            print(f"\n[3.5] 双侧 throwaway: 丢弃前 {n_throwaway} 条 query "
                  f"(热 GR JIT / Cypher 候选 + PG plan cache)")
            import contextlib
            import io
            tw_codes = poi_codes[:n_throwaway]
            for tw_qc in tw_codes:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        run_single_query(
                            session, pg_conn,
                            tw_qc, poi_source, parcel_source,
                            compare_pg, strategy=strategy,
                            polygon_cache=pc,
                            nb_cache=nb_cache)
                except Exception:
                    pass
            print(f"      throwaway 完成, 进入正式计时阶段")

        print(f"\n[4] 运行反向叠置查询 "
              f"({len(poi_codes):,} 次) ...")
        run_batch(
            session, pg_conn,
            poi_codes, poi_source, parcel_source,
            compare_pg, strategy=strategy,
            output_csv=output_csv,
            polygon_cache=pc,
            nb_cache=nb_cache,
            hit_only=hit_only)

    session.close()
    driver.close()
    if pg_conn:
        pg_conn.close()
    print("\n完成。")


if __name__ == "__main__":
    main()

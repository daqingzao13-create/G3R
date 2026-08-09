#!/usr/bin/env python3
# coding: utf-8
"""
GeoSOT 通用几何要素距离推理模块
/ GeoSOT general geometry-feature distance reasoning module

支持 Point / LineString / Polygon 任意两种几何类型之间的最短距离计算（3×3 = 9 种组合）。
/ Shortest distance between any two of Point / LineString / Polygon (3×3 = 9 combos).

数据模型 (论文 schema) / Data model (paper schema):
  (:SpatialFeature)-[:HAS_CENTROID_CODE]->(:GeoSOTCentroidCode)
  (:GeoSOTCentroidCode)-[:HAS_GEOMETRY_TYPE]->(:GeoFeatureType {type_name})
  (:SpatialFeature)-[:HAS_VERTEX_CODE {seq}]->(:GeoSOTVertexCode)
  (:GeoSOTCentroidCode)-[:FROM_SOURCE]->(:DataSource {name})

算法 / Algorithm:
  1. 从 Neo4j 查询两个要素的几何类型和顶点编码
     / Query geometry types and vertex codes from Neo4j
  2. 根据几何类型构建坐标和边集 / Build coords and edge sets by type:
     - Point:      1 坐标, 0 条边 / 1 coord, 0 edges
     - LineString:  n 坐标, n-1 条边（不闭合） / n coords, n-1 edges (open)
     - Polygon:     n 坐标, n 条边（闭合环） / n coords, n edges (closed ring)
  3. 三种方法计算几何最短距离 / Three methods for geometric min distance:
     a) 网格解码:  fast_code_to_center → 浮点经纬度 → 局部平面距离
        / grid decode: fast_code_to_center → float lon/lat → local planar distance
     b) 图推理:    fast_estimate_vertices_int → 整数偏移 (1/U 度单位)
                  → 整数谓词扫描 (相交 / 垂足 t∈[0,1] / 序保 argmin)
                  → 仅对胜出点对合成经纬度 + 1 次 Vincenty
        / graph: integer offsets (1/U deg) → integer predicates
          (intersect / foot t∈[0,1] / order-preserving argmin)
          → synthesize lon/lat for winner pair + 1× Vincenty
     c) PostGIS:   ST_Distance (geography) 作为精度基准
        / PostGIS ST_Distance(geography) as accuracy baseline
  4. 网格解码法在浮点经纬度上做几何; 图推理法所有谓词为整数运算,
     仅末端做 1 次坐标合成 + 1 次 Vincenty。
     / Decode path uses float lon/lat geometry; graph path uses integer
       predicates only, with 1× coord synthesis + 1× Vincenty at the end.
  5. 与 §2.3.1 算子叙事一致: G3R 距离推理"除末端 Vincenty 外
     全部判定均在编码侧整数偏移上完成"。
     / Aligned with §2.3.1: G3R distance "all predicates on code-side
       integer offsets except terminal Vincenty".

使用方法 / Usage:
    python geosot_geometry_reasoning.py --list                           # 列出所有要素 / list features
    python geosot_geometry_reasoning.py --list --type Polygon            # 按几何类型筛选 / filter by type
    python geosot_geometry_reasoning.py --list --source Buildings        # 按数据源筛选 / filter by source
    python geosot_geometry_reasoning.py <id_a> <id_b>                   # 计算距离 / compute distance
    python geosot_geometry_reasoning.py --no-postgis <id_a> <id_b>      # 不对比 PostGIS / skip PostGIS
    python geosot_geometry_reasoning.py --json <id_a> <id_b>            # JSON 输出 / JSON output
"""

import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

from neo4j import GraphDatabase

from geosot_spatial_reasoning import (
    PG_CONFIG,
    _GS_LAT,
    _GS_LON,
    _OFF_EW,
    _OFF_NS,
    _bits31_to_degree_int,
    compute_bearing,
    fast_code_to_center,
    vincenty_fast,
)

# ============================================================================
# Neo4j 配置 / Neo4j config
# ============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

VALID_TYPES = ("Point", "LineString", "Polygon")

# Neo4j Source.name → PostGIS 表名映射 / Neo4j Source.name → PostGIS table map
SOURCE_TO_PG_TABLE: Dict[str, str] = {
    "POI": "public.poi_new",
    "Buildings": "public.building_new",
    "Road": "public.road",
    "Community": "public.admin_community",
    "Subdistrict": "public.admin_subdistrict",
    "District": "public.admin_district",
    "Statutory Plan": "public.statutory_plan",
}


# ============================================================================
# 高性能顶点坐标估算（图推理） / Fast vertex coordinate estimation (graph reasoning)
# ============================================================================

def fast_estimate_vertices(
    codes_a: List[str],
    codes_b: List[str],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
           float, float]:
    """
    高性能图推理顶点坐标估算，使用整数位运算替代 chain dict。
    / Fast graph-reasoning vertex estimation via integer bit ops (no chain dict).

    所有顶点共用同一参考坐标系（公共祖先网格中心），保证两组坐标
    可直接比较。同时返回公共祖先参考纬度，供后续几何计算复用。
    / All vertices share one reference frame (common-ancestor grid center) so
      both sets are comparable. Also returns ancestor reference latitude for
      later geometry reuse.

    @param codes_a 第一组 GeoSOT 编码（已大写，无空白） / first code group (upper, stripped)
    @param codes_b 第二组 GeoSOT 编码 / second GeoSOT code group
    @returns (coords_a, coords_b, ref_lon, ref_lat)
             coords 各为 [(lon, lat), ...] / each coords is [(lon, lat), ...]
             ref_lon/ref_lat 为公共祖先网格中心经纬度 / common-ancestor center lon/lat
    """
    all_codes = list(codes_a) + list(codes_b)
    if not all_codes:
        return [], [], 0.0, 0.0

    bodies = [c[1:] for c in all_codes]
    min_len = min(len(b) for b in bodies)
    max_body = max(len(b) for b in bodies)

    diff = -1
    for i in range(min_len):
        chars = set(b[i] for b in bodies)
        if len(chars) > 1:
            diff = i
            break

    if diff < 0:
        coords = [fast_code_to_center(c) for c in all_codes]
        n_a = len(codes_a)
        ref_lon = coords[0][0]
        ref_lat = coords[0][1]
        return coords[:n_a], coords[n_a:], ref_lon, ref_lat

    pf = bodies[0][1:max(diff, 1)]
    n_pf = len(pf)
    lat_v = lon_v = 0
    for ch in pf:
        d = ord(ch) - 48
        lat_v = (lat_v << 1) | (d >> 1)
        lon_v = (lon_v << 1) | (d & 1)
    remain = 31 - n_pf
    lat_lo = lat_v << remain
    lon_lo = lon_v << remain
    mask = (1 << remain) - 1
    ref_lon = (_bits31_to_degree_int(lon_lo)
               + _bits31_to_degree_int(lon_lo | mask)) * 0.5
    ref_lat = (_bits31_to_degree_int(lat_lo)
               + _bits31_to_degree_int(lat_lo | mask)) * 0.5

    coords: List[Tuple[float, float]] = []
    for body in bodies:
        oa_ew = oa_ns = 0.0
        upper = min(len(body), 32)
        for i in range(diff, upper):
            d = ord(body[i]) - 48
            oa_ew += _OFF_EW[d] * _GS_LON[i]
            oa_ns += _OFF_NS[d] * _GS_LAT[i]
        coords.append((ref_lon + oa_ew, ref_lat + oa_ns))

    n_a = len(codes_a)
    return coords[:n_a], coords[n_a:], ref_lon, ref_lat


# ============================================================================
# 通用几何要素最短距离计算 / General geometry-feature min-distance
# ============================================================================

def _build_cos_ref_matrix(
    codes_a: List[str], codes_b: List[str],
) -> List[List[float]]:
    """
    为每一对顶点编码计算其公共祖先网格参考纬度的 cos 值。
    / Cosine of common-ancestor reference latitude for each vertex-code pair.

    与点–点推理一致：每对顶点 (codes_a[i], codes_b[j]) 独立求
    公共前缀，取公共祖先网格中心纬度的 cos 作为该对的经度缩放因子。
    / Same as point–point: each pair finds its own common prefix; use
      cos(ancestor center lat) as that pair's longitude scale factor.

    @param codes_a 要素 A 的编码列表 / feature A code list
    @param codes_b 要素 B 的编码列表 / feature B code list
    @returns na×nb 矩阵，matrix[i][j] = cos(φ_ref) for codes_a[i] vs codes_b[j]
             / na×nb matrix of cos(φ_ref) per pair
    """
    na, nb = len(codes_a), len(codes_b)
    matrix: List[List[float]] = []
    for i in range(na):
        ba = codes_a[i][1:]
        row: List[float] = []
        for j in range(nb):
            bb = codes_b[j][1:]
            min_len = min(len(ba), len(bb))
            diff = -1
            for k in range(min_len):
                if ba[k] != bb[k]:
                    diff = k
                    break
            if diff < 0:
                ref_lat = fast_code_to_center(codes_a[i])[1]
            else:
                pf = ba[1:max(diff, 1)]
                n_pf = len(pf)
                lat_v = 0
                for ch in pf:
                    d = ord(ch) - 48
                    lat_v = (lat_v << 1) | (d >> 1)
                remain = 31 - n_pf
                lat_lo = lat_v << remain
                mask = (1 << remain) - 1
                ref_lat = (_bits31_to_degree_int(lat_lo)
                           + _bits31_to_degree_int(lat_lo | mask)) * 0.5
            row.append(math.cos(math.radians(ref_lat)))
        matrix.append(row)
    return matrix


def _seg_foot_lonlat(
    p_lon: float, p_lat: float,
    a_lon: float, a_lat: float,
    b_lon: float, b_lat: float,
    cos_ref: float,
) -> Tuple[float, float]:
    """
    在经纬度空间计算点 P 到线段 AB 的垂足经纬度。
    / Foot of perpendicular from P to segment AB in lon/lat space.

    使用 cos(参考纬度) 对经度方向进行等距缩放以求得正确的投影参数 t，
    垂足通过在经纬度空间线性插值获得。
    / Scale longitude by cos(ref lat) for correct projection parameter t;
      interpolate the foot in lon/lat space.

    @param p_lon, p_lat 点 P 的经纬度 / point P lon/lat
    @param a_lon, a_lat 线段端点 A 的经纬度 / segment endpoint A
    @param b_lon, b_lat 线段端点 B 的经纬度 / segment endpoint B
    @param cos_ref      cos(参考纬度)，用于经度缩放 / cos(ref lat) for lon scaling
    @returns (foot_lon, foot_lat)
    """
    dax = (p_lon - a_lon) * cos_ref
    day = p_lat - a_lat
    dbx = (b_lon - a_lon) * cos_ref
    dby = b_lat - a_lat
    len_sq = dbx * dbx + dby * dby
    if len_sq < 1e-20:
        return a_lon, a_lat
    t = max(0.0, min(1.0, (dax * dbx + day * dby) / len_sq))
    return (a_lon + t * (b_lon - a_lon),
            a_lat + t * (b_lat - a_lat))


def _seg_intersect_lonlat(
    a1_lon: float, a1_lat: float, a2_lon: float, a2_lat: float,
    b1_lon: float, b1_lat: float, b2_lon: float, b2_lat: float,
) -> bool:
    """两线段是否严格相交（经纬度空间，叉积符号在正缩放下不变）。 / Strict segment intersection in lon/lat (cross-product sign invariant under positive axis scaling)."""
    def _cross(ox, oy, ax, ay, bx, by):
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
    d1 = _cross(b1_lon, b1_lat, b2_lon, b2_lat, a1_lon, a1_lat)
    d2 = _cross(b1_lon, b1_lat, b2_lon, b2_lat, a2_lon, a2_lat)
    d3 = _cross(a1_lon, a1_lat, a2_lon, a2_lat, b1_lon, b1_lat)
    d4 = _cross(a1_lon, a1_lat, a2_lon, a2_lat, b2_lon, b2_lat)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _build_edge_indices(n: int, geo_type: str) -> List[Tuple[int, int]]:
    """
    根据几何类型构建边的索引对。
    / Build edge index pairs from geometry type.

    @param n 顶点数 / vertex count
    @param geo_type "Point" / "LineString" / "Polygon"
    @returns [(start_idx, end_idx), ...]
    """
    if geo_type == "Point" or n < 2:
        return []
    if geo_type == "LineString":
        return [(i, i + 1) for i in range(n - 1)]
    return [(i, (i + 1) % n) for i in range(n)]


def geometry_min_distance_m(
    coords_a: List[Tuple[float, float]], type_a: str,
    coords_b: List[Tuple[float, float]], type_b: str,
    cos_ref_matrix: Optional[List[List[float]]] = None,
) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """
    通用几何要素最短距离（米），支持 Point/LineString/Polygon 的全部 3×3 组合。
    / General geometry min distance (m); all Point/LineString/Polygon 3×3 combos.

    与点–点推理一致：每对顶点使用各自编码的公共祖先网格参考纬度进行
    经度缩放，以 cos(φ_ref) 加权的经纬度差平方和筛选候选最近点对，
    仅对最终胜出的一对调用 Vincenty 椭球公式。
    / Same as point–point: per-pair ancestor-lat cos scales longitude; pick
      nearest pair by cos-weighted squared lon/lat delta; Vincenty once on winner.

    @param coords_a        要素 A 的坐标列表 [(lon, lat), ...] / feature A coords
    @param type_a          要素 A 的几何类型 / feature A geometry type
    @param coords_b        要素 B 的坐标列表 / feature B coords
    @param type_b          要素 B 的几何类型 / feature B geometry type
    @param cos_ref_matrix  na×nb 矩阵，[i][j] = cos(公共祖先纬度) for
                           codes_a[i] vs codes_b[j]；None 时回退到均值纬度
                           / na×nb cos(ancestor lat); None → mean-lat fallback
    @returns (distance_m, nearest_point_a, nearest_point_b)
    """
    na, nb = len(coords_a), len(coords_b)
    if na == 0 or nb == 0:
        return float("inf"), (0, 0), (0, 0)

    # Point × Point: 直接 Vincenty / direct Vincenty
    if na == 1 and nb == 1:
        d = vincenty_fast(
            coords_a[0][1], coords_a[0][0],
            coords_b[0][1], coords_b[0][0]) * 1000
        return d, coords_a[0], coords_b[0]

    # 逐对 cos(φ_ref) 或回退到统一均值 / per-pair cos(φ_ref) or unified mean fallback
    if cos_ref_matrix is not None:
        def _cos(ia: int, jb: int) -> float:
            return cos_ref_matrix[ia][jb]
    else:
        avg_lat = sum(p[1] for p in coords_a + coords_b) / (na + nb)
        _fallback = math.cos(math.radians(avg_lat))
        def _cos(ia: int, jb: int) -> float:
            return _fallback

    ea = _build_edge_indices(na, type_a)
    eb = _build_edge_indices(nb, type_b)

    best_sq = float("inf")
    pt_a = coords_a[0]
    pt_b = coords_b[0]

    def _try(pa: Tuple[float, float], pb: Tuple[float, float],
             cos_r: float):
        nonlocal best_sq, pt_a, pt_b
        dlon = pa[0] - pb[0]
        dlat = pa[1] - pb[1]
        sq = dlon * dlon * cos_r * cos_r + dlat * dlat
        if sq < best_sq:
            best_sq, pt_a, pt_b = sq, pa, pb

    # 边 × 边：相交检测 + 端点到对方边的垂足 / edge×edge: intersect + endpoint feet
    for i1, i2 in ea:
        for j1, j2 in eb:
            a1 = coords_a[i1]
            a2 = coords_a[i2]
            b1 = coords_b[j1]
            b2 = coords_b[j2]

            if _seg_intersect_lonlat(a1[0], a1[1], a2[0], a2[1],
                                     b1[0], b1[1], b2[0], b2[1]):
                mid = ((a1[0] + a2[0] + b1[0] + b2[0]) * 0.25,
                       (a1[1] + a2[1] + b1[1] + b2[1]) * 0.25)
                return 0.0, mid, mid

            for ep, seg_s, seg_e, a_is_ep, ri, rj in (
                (a1, b1, b2, True,  i1, j1),
                (a2, b1, b2, True,  i2, j1),
                (b1, a1, a2, False, i1, j1),
                (b2, a1, a2, False, i1, j2),
            ):
                cr = _cos(ri, rj)
                foot = _seg_foot_lonlat(
                    ep[0], ep[1],
                    seg_s[0], seg_s[1],
                    seg_e[0], seg_e[1],
                    cr)
                if a_is_ep:
                    _try(ep, foot, cr)
                else:
                    _try(foot, ep, cr)

    # 点 × 边（一方为 Point，无自身边） / point×edge (one side is Point)
    if not ea:
        p = coords_a[0]
        for j1, j2 in eb:
            cr = _cos(0, j1)
            foot = _seg_foot_lonlat(
                p[0], p[1],
                coords_b[j1][0], coords_b[j1][1],
                coords_b[j2][0], coords_b[j2][1],
                cr)
            _try(p, foot, cr)

    if not eb:
        p = coords_b[0]
        for i1, i2 in ea:
            cr = _cos(i1, 0)
            foot = _seg_foot_lonlat(
                p[0], p[1],
                coords_a[i1][0], coords_a[i1][1],
                coords_a[i2][0], coords_a[i2][1],
                cr)
            _try(foot, p, cr)

    # 全顶点对校验 / all vertex–vertex pairs
    for i, ca in enumerate(coords_a):
        for j, cb in enumerate(coords_b):
            _try(ca, cb, _cos(i, j))

    # 仅对最终最近点对调用一次 Vincenty 获得椭球距离 / one Vincenty on final nearest pair
    dist_m = vincenty_fast(pt_a[1], pt_a[0], pt_b[1], pt_b[0]) * 1000
    return dist_m, pt_a, pt_b


# ============================================================================
# NumPy 向量化版本 / NumPy vectorized variant
# ============================================================================

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _points_to_edges_sq_np(
    P: "np.ndarray",   # (np_, 2)  lon/lat
    S: "np.ndarray",   # (ne, 2)   segment starts
    E: "np.ndarray",   # (ne, 2)   segment ends
    cos: float,
):
    """
    向量化: 每个点 P[i] 到每条线段 S[j]→E[j] 的最短距离平方。
    / Vectorized: squared min distance from each P[i] to each segment S[j]→E[j].

    @returns sq   (np_, ne) 每点到每段的距离平方 (度²，经向已乘 cos)
                  / per-point-per-segment squared distance (deg², lon scaled by cos)
             foot (np_, ne, 2) 对应垂足坐标 (lon, lat) / corresponding foot lon/lat
    """
    import numpy as np
    D = E - S                            # (ne, 2)
    Dx = D[:, 0] * cos                   # (ne,)
    Dy = D[:, 1]                         # (ne,)
    len_sq = Dx ** 2 + Dy ** 2           # (ne,)

    # diff: P[i] - S[j], shape (np_, ne, 2)
    diff = P[:, None, :] - S[None, :, :]
    dot = diff[:, :, 0] * cos * Dx[None, :] + diff[:, :, 1] * Dy[None, :]  # (np_, ne)

    safe_len = np.where(len_sq > 1e-20, len_sq, 1.0)
    t = np.clip(dot / safe_len[None, :], 0.0, 1.0)
    t = np.where(len_sq[None, :] > 1e-20, t, 0.0)

    # foot = S + t * D, shape (np_, ne, 2)
    foot = S[None, :, :] + t[:, :, None] * D[None, :, :]

    df = P[:, None, :] - foot            # (np_, ne, 2)
    sq = (df[:, :, 0] * cos) ** 2 + df[:, :, 1] ** 2  # (np_, ne)
    return sq, foot


def geometry_min_distance_m_np(
    coords_a: List[Tuple[float, float]], type_a: str,
    coords_b: List[Tuple[float, float]], type_b: str,
    cos_ref: float,
) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """
    NumPy 向量化版几何最短距离。算法与 geometry_min_distance_m 完全等价,
    用矩阵运算代替 Python 嵌套循环，对多顶点多边形有显著加速。
    / NumPy vectorized min distance; equivalent to geometry_min_distance_m,
      matrix ops instead of nested loops; faster on multi-vertex polygons.

    numpy 不可用时自动回退到标量版本。
    / Falls back to scalar version when numpy is unavailable.

    @param cos_ref  统一 cos 参考值 (同城内用查询中心纬度的 cos 即可)
                    / unified cos reference (query-center lat cos is fine in-city)
    @returns (distance_m, nearest_point_a, nearest_point_b)
    """
    if not _HAS_NUMPY:
        avg_lat = (sum(c[1] for c in coords_a) + sum(c[1] for c in coords_b)
                   ) / (len(coords_a) + len(coords_b))
        fixed_cos = math.cos(math.radians(avg_lat))
        m = [[fixed_cos] * len(coords_b)] * len(coords_a)
        return geometry_min_distance_m(coords_a, type_a, coords_b, type_b, m)

    import numpy as np
    A = np.array(coords_a, dtype=np.float64)  # (na, 2)
    B = np.array(coords_b, dtype=np.float64)  # (nb, 2)
    na, nb = len(A), len(B)

    if na == 0 or nb == 0:
        return float("inf"), (0.0, 0.0), (0.0, 0.0)

    # Point × Point: Vincenty 直接 / Vincenty directly
    if na == 1 and nb == 1:
        d = vincenty_fast(float(A[0, 1]), float(A[0, 0]),
                          float(B[0, 1]), float(B[0, 0])) * 1000
        return d, (float(A[0, 0]), float(A[0, 1])), (float(B[0, 0]), float(B[0, 1]))

    # 构建边索引 / build edge indices
    def _edge_idx(n, geo_type):
        if geo_type == "Point" or n < 2:
            return None, None
        idx = np.arange(n)
        if geo_type == "Polygon":
            return idx, np.roll(idx, -1)
        return idx[:-1], idx[1:]   # LineString

    ai1, ai2 = _edge_idx(na, type_a)
    bi1, bi2 = _edge_idx(nb, type_b)

    AS = A[ai1] if ai1 is not None else None  # (ne_a, 2)
    AE = A[ai2] if ai2 is not None else None
    BS = B[bi1] if bi1 is not None else None  # (ne_b, 2)
    BE = B[bi2] if bi2 is not None else None

    # 检测边-边相交 (bbox 快速排除后才进入精确检测) / edge-edge intersect (after bbox reject)
    def _bboxes_overlap_1d(lo1, hi1, lo2, hi2):
        return lo1 <= hi2 and lo2 <= hi1

    a_lo = A.min(axis=0); a_hi = A.max(axis=0)
    b_lo = B.min(axis=0); b_hi = B.max(axis=0)
    if (_bboxes_overlap_1d(a_lo[0], a_hi[0], b_lo[0], b_hi[0]) and
            _bboxes_overlap_1d(a_lo[1], a_hi[1], b_lo[1], b_hi[1]) and
            AS is not None and BS is not None):
        for i in range(len(AS)):
            for j in range(len(BS)):
                if _seg_intersect_lonlat(
                    float(AS[i, 0]), float(AS[i, 1]), float(AE[i, 0]), float(AE[i, 1]),
                    float(BS[j, 0]), float(BS[j, 1]), float(BE[j, 0]), float(BE[j, 1])
                ):
                    mid = ((float(AS[i, 0]) + float(AE[i, 0]) +
                            float(BS[j, 0]) + float(BE[j, 0])) * 0.25,
                           (float(AS[i, 1]) + float(AE[i, 1]) +
                            float(BS[j, 1]) + float(BE[j, 1])) * 0.25)
                    return 0.0, mid, mid

    # 全顶点对: 一次矩阵运算 / all vertex pairs: one matrix op
    dlon = (A[:, None, 0] - B[None, :, 0]) * cos_ref   # (na, nb)
    dlat = A[:, None, 1] - B[None, :, 1]               # (na, nb)
    sq_vv = dlon ** 2 + dlat ** 2                       # (na, nb)

    best_sq: float = float(sq_vv.min())
    idx_vv = np.unravel_index(sq_vv.argmin(), sq_vv.shape)
    best_pa = (float(A[idx_vv[0], 0]), float(A[idx_vv[0], 1]))
    best_pb = (float(B[idx_vv[1], 0]), float(B[idx_vv[1], 1]))

    # A 顶点 → B 各边垂足 / A vertices → feet on B edges
    if BS is not None:
        sq_ve, foot_ve = _points_to_edges_sq_np(A, BS, BE, cos_ref)  # (na, ne_b)
        m = float(sq_ve.min())
        if m < best_sq:
            best_sq = m
            idx = np.unravel_index(sq_ve.argmin(), sq_ve.shape)
            best_pa = (float(A[idx[0], 0]), float(A[idx[0], 1]))
            best_pb = (float(foot_ve[idx[0], idx[1], 0]),
                       float(foot_ve[idx[0], idx[1], 1]))

    # B 顶点 → A 各边垂足 / B vertices → feet on A edges
    if AS is not None:
        sq_ev, foot_ev = _points_to_edges_sq_np(B, AS, AE, cos_ref)  # (nb, ne_a)
        m = float(sq_ev.min())
        if m < best_sq:
            best_sq = m
            idx = np.unravel_index(sq_ev.argmin(), sq_ev.shape)
            best_pb = (float(B[idx[0], 0]), float(B[idx[0], 1]))
            best_pa = (float(foot_ev[idx[0], idx[1], 0]),
                       float(foot_ev[idx[0], idx[1], 1]))

    # 只对胜出点对调一次 Vincenty / one Vincenty on winning pair
    dist_m = vincenty_fast(best_pa[1], best_pa[0], best_pb[1], best_pb[0]) * 1000
    return dist_m, best_pa, best_pb


# ============================================================================
# 整数谓词版图推理几何距离 (G3R 编码代数下的几何谓词)
# / Integer-predicate graph geometry distance (G3R code-algebra predicates)
# ============================================================================
#
# 设计动机：与 §2.3.1 算子叙事对齐——"距离与方向推理仅在末端做一次坐标
# / Design: align with §2.3.1 — "distance/direction decode coords
# 解码 + 一次 Vincenty"，其余判定（线段相交、垂足落点、序保 argmin）全部
# / + Vincenty only at end"; other predicates (intersect, foot, order-preserving argmin) all
# 在 GeoSOT 公共祖先参考系下的整数偏移上完成，遵循
# / run on integer offsets in the GeoSOT common-ancestor frame, following
# "exact predicates, inexact constructions" 范式。
# / "exact predicates, inexact constructions" paradigm.
#
# 整数化方案 / Integerization:
#   1. 公共最小单位 U = 2 * 3600 * 2048 = 14_745_600 (1/U 度)
#      / Common min unit U = 14_745_600 (1/U deg)
#      使每一层级 GeoSOT 半格步长 ±0.5·S(k) 都是整数。
#      / so every half-cell step ±0.5·S(k) is an integer.
#   2. cos(φ_ref) 用定点表示 Cx = round(cos(φ_ref) * Ky), Ky = 2**30。
#      / Fixed-point cos: Cx = round(cos(φ_ref) * Ky), Ky = 2**30.
#      预乘到 X 轴: X' = X * Cx, Y' = Y * Ky。现在度量空间内 X 与 Y
#      / Pre-scale X: X'=X*Cx, Y'=Y*Ky; then X and Y are
#      等距，所有谓词在 (X', Y') 整数空间执行欧氏算术。
#      / isometric; all predicates use Euclidean arithmetic in (X', Y').
#   3. 相交检测: 整数叉积符号 (符号在正缩放下不变, 与是否乘 cos 无关)。
#      / Intersection via integer cross-product signs.
#   4. 垂足参数 t ∈ [0,1]: 整数比较 0 ≤ N ≤ D, 不计算 t 的浮点值。
#      / Foot parameter t∈[0,1] via integer bounds 0≤N≤D (no float t).
#   5. 序保 argmin: 顶点-顶点对距离平方为整数; 顶点-垂足对距离平方为
#      / Order-preserving argmin: vv squared dist is int; vertex-foot squared dist is
#      有理数 num/den², 通过整数交叉相乘 num1·den2² < num2·den1² 比较。
#      / a rational num/den², compared by integer cross-multiply.
#   6. 最终仅对胜出对 (P*, Q*) 一次合成经纬度并调用 1 次 Vincenty。
#      / Synthesize lon/lat + 1× Vincenty only for the winning pair.

# 单位换算: 1 度 = U 个 "1/U 度" 单位 / 1 degree = U units of 1/U degree
_U_PER_DEG = 2 * 3600 * 2048  # 14_745_600
# 定点 cos 缩放: 选 2**30 ~ 1.07e9, 平衡精度与整数位宽
# / Fixed-point cos scale 2**30 ~ 1.07e9 (precision vs integer width)
_K_COS = 1 << 30


def _half_cell_int_units(level: int) -> Tuple[int, int]:
    """
    第 level 层 GeoSOT 网格的半格步长, 单位 1/U 度 (整数)。
    / Half-cell step at GeoSOT level, in 1/U-degree integer units.

    @returns (ew_units, ns_units)
             层级 ≥ 2 时两者相等; level=1 因 EW=180°/NS=90° 不等。
             / Equal for level≥2; level=1 differs (EW=180° / NS=90°).
    """
    if level == 1:
        return (90 * _U_PER_DEG, 45 * _U_PER_DEG)
    bit_pos = level - 2
    if bit_pos < 8:
        v = (1 << (7 - bit_pos)) * 3600 * 2048
    elif bit_pos < 14:
        v = (1 << (13 - bit_pos)) * 60 * 2048
    elif bit_pos < 20:
        v = (1 << (19 - bit_pos)) * 2048
    else:
        v = 1 << (30 - bit_pos)
    return (v, v)


_HALF_INT_EW = tuple(_half_cell_int_units(lv)[0] for lv in range(1, 33))
_HALF_INT_NS = tuple(_half_cell_int_units(lv)[1] for lv in range(1, 33))


def fast_estimate_vertices_int(
    codes_a: List[str],
    codes_b: List[str],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]],
           float, float, float]:
    """
    G3R 整数偏移版顶点估算, 不解码到经纬度。
    / G3R integer-offset vertex estimation (no lon/lat decode).

    所有顶点共用公共祖先参考点 (ref_lon, ref_lat). 每个顶点的相对偏移
    以 1/U 度 (U=14_745_600) 为单位的整数 (X, Y) 表示, X=东向, Y=北向。
    与 fast_estimate_vertices 不同: 整个过程不出现浮点经纬度, 仅在
    最终需要时才反向合成。
    / Shared ancestor reference; offsets as integer (X,Y) in 1/U-degree units
      (X east, Y north). Unlike fast_estimate_vertices, no float lon/lat until
      reverse synthesis is needed.

    @param codes_a 第一组 GeoSOT 编码 (已大写, 无空白) / first codes (upper, stripped)
    @param codes_b 第二组 GeoSOT 编码 / second GeoSOT codes
    @returns (coords_int_a, coords_int_b, ref_lon, ref_lat, cos_ref)
             coords_int_*: [(X, Y), ...] 公共祖先下整数偏移 (1/U 度单位)
             / integer offsets under common ancestor (1/U-degree units)
             ref_lon/ref_lat: 公共祖先网格中心经纬度 (度, float)
             / ancestor center lon/lat (degrees, float)
             cos_ref: cos(ref_lat) (用于度量空间的 X 轴等距加权)
             / cos(ref_lat) for isometric X weighting
    """
    all_codes = list(codes_a) + list(codes_b)
    if not all_codes:
        return [], [], 0.0, 0.0, 1.0

    bodies = [c[1:] for c in all_codes]
    min_len = min(len(b) for b in bodies)

    diff = -1
    for i in range(min_len):
        chars = set(b[i] for b in bodies)
        if len(chars) > 1:
            diff = i
            break

    if diff < 0:
        ref_lon, ref_lat = fast_code_to_center(all_codes[0])
        n_a = len(codes_a)
        zeros = [(0, 0)] * len(all_codes)
        return zeros[:n_a], zeros[n_a:], ref_lon, ref_lat, math.cos(
            math.radians(ref_lat))

    pf = bodies[0][1:max(diff, 1)]
    n_pf = len(pf)
    lat_v = lon_v = 0
    for ch in pf:
        d = ord(ch) - 48
        lat_v = (lat_v << 1) | (d >> 1)
        lon_v = (lon_v << 1) | (d & 1)
    remain = 31 - n_pf
    lat_lo = lat_v << remain
    lon_lo = lon_v << remain
    mask = (1 << remain) - 1
    ref_lon = (_bits31_to_degree_int(lon_lo)
               + _bits31_to_degree_int(lon_lo | mask)) * 0.5
    ref_lat = (_bits31_to_degree_int(lat_lo)
               + _bits31_to_degree_int(lat_lo | mask)) * 0.5
    cos_ref = math.cos(math.radians(ref_lat))

    coords_int: List[Tuple[int, int]] = []
    for body in bodies:
        x = y = 0
        upper = min(len(body), 32)
        for i in range(diff, upper):
            d = ord(body[i]) - 48
            sx = 1 if (d & 1) else -1
            sy = 1 if (d >> 1) else -1
            x += sx * _HALF_INT_EW[i]
            y += sy * _HALF_INT_NS[i]
        coords_int.append((x, y))

    n_a = len(codes_a)
    return coords_int[:n_a], coords_int[n_a:], ref_lon, ref_lat, cos_ref


def _seg_intersect_int(
    a1x: int, a1y: int, a2x: int, a2y: int,
    b1x: int, b1y: int, b2x: int, b2y: int,
) -> bool:
    """
    两线段是否严格相交; 整数叉积谓词, 无浮点。
    / Strict segment intersection; integer cross-product predicate, no floats.

    叉积符号在沿任一轴的正缩放下不变, 故等价于在原始 (X,Y) 整数偏移
    或在 cos 加权空间 (X', Y') 上判定。
    / Cross-product sign is invariant under positive axis scaling, so equivalent
      on raw (X,Y) offsets or cos-weighted (X', Y').
    """
    d1 = (b2x - b1x) * (a1y - b1y) - (b2y - b1y) * (a1x - b1x)
    d2 = (b2x - b1x) * (a2y - b1y) - (b2y - b1y) * (a2x - b1x)
    d3 = (a2x - a1x) * (b1y - a1y) - (a2y - a1y) * (b1x - a1x)
    d4 = (a2x - a1x) * (b2y - a1y) - (a2y - a1y) * (b2x - a1x)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def geometry_min_distance_m_int(
    coords_int_a: List[Tuple[int, int]], type_a: str,
    coords_int_b: List[Tuple[int, int]], type_b: str,
    ref_lon: float, ref_lat: float, cos_ref: float,
) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """
    G3R 整数谓词版几何最短距离 (m), 单一公共祖先参考系 + 末端 1 次 Vincenty。
    / G3R integer-predicate min distance (m): one ancestor frame + terminal Vincenty.

    所有相交检测 / 垂足落点 t ∈ [0,1] 判定 / 序保 argmin 比较均在
    cos 加权后的整数空间 (X', Y') = (X*Cx, Y*Ky) 上完成; 其中
    Cx = round(cos(φ_ref)*Ky), Ky = 2**30。仅在最后对胜出点对反向
    合成经纬度并调用 1 次 Vincenty 计算椭球距离。
    / Intersect / foot t∈[0,1] / order-preserving argmin all run in
      cos-weighted integer space (X',Y')=(X*Cx,Y*Ky); synthesize lon/lat and
      call Vincenty once for the winning pair.

    与 geometry_min_distance_m (浮点版) 在数学上等价 (差异仅来自
    cos_ref 单一参考 vs 逐对 cos_ref_matrix 的近似), 但谓词全部为
    整数运算, 无浮点累计误差。
    / Mathematically equivalent to float geometry_min_distance_m (aside from
      single cos_ref vs per-pair matrix); predicates are pure integer.

    @param coords_int_a 整数偏移 [(X, Y), ...] (1/U 度单位) / integer offsets
    @param type_a       "Point" / "LineString" / "Polygon"
    @param coords_int_b 同上 / same for B
    @param type_b       同上 / same
    @param ref_lon      公共祖先中心经度 (度) / ancestor center lon (deg)
    @param ref_lat      公共祖先中心纬度 (度) / ancestor center lat (deg)
    @param cos_ref      cos(ref_lat); 用于 X 轴等距加权 / for isometric X weighting
    @returns (distance_m, nearest_a_lonlat, nearest_b_lonlat)
    """
    na, nb = len(coords_int_a), len(coords_int_b)
    if na == 0 or nb == 0:
        return float("inf"), (ref_lon, ref_lat), (ref_lon, ref_lat)

    Ky = _K_COS
    Cx = round(cos_ref * Ky)
    if Cx <= 0:
        Cx = 1

    # 预乘 cos: 在 (X', Y') 等距整数空间执行所有几何谓词 / pre-scale cos; predicates in isometric (X',Y')
    A = [(x * Cx, y * Ky) for x, y in coords_int_a]
    B = [(x * Cx, y * Ky) for x, y in coords_int_b]

    inv_dx = 1.0 / (Cx * _U_PER_DEG)
    inv_dy = 1.0 / (Ky * _U_PER_DEG)

    def _to_lonlat(xi: int, yi: int) -> Tuple[float, float]:
        return (ref_lon + xi * inv_dx, ref_lat + yi * inv_dy)

    def _foot_to_lonlat(xn: int, yn: int, d: int) -> Tuple[float, float]:
        return (ref_lon + xn / d * inv_dx,
                ref_lat + yn / d * inv_dy)

    if na == 1 and nb == 1:
        pa = _to_lonlat(*A[0])
        pb = _to_lonlat(*B[0])
        d = vincenty_fast(pa[1], pa[0], pb[1], pb[0]) * 1000
        return d, pa, pb

    ea = _build_edge_indices(na, type_a)
    eb = _build_edge_indices(nb, type_b)

    # argmin 状态: best_num / best_den_sq = 当前最优距离平方 (cos 加权空间)
    # / argmin state: best_num / best_den_sq = best squared distance (cos-weighted)
    # 候选描述符 best_desc 用于末端反向合成 lat/lon:
    # / best_desc describes winner for terminal lon/lat synthesis:
    #   ("vv", a_int, b_int)              顶点 - 顶点 / vertex–vertex
    #   ("vf", a_int, fx_n, fy_n, fd)     A 顶点 - B 上垂足 (foot=fxn/fd, fyn/fd)
    #                                     / A vertex – foot on B
    #   ("fv", fx_n, fy_n, fd, b_int)     A 上垂足 - B 顶点 / foot on A – B vertex
    state = {
        "init": False,
        "num": 0,
        "den_sq": 1,
        "desc": ("vv", A[0], B[0]),
    }

    def _better(num: int, den_sq: int) -> bool:
        if not state["init"]:
            return True
        return num * state["den_sq"] < state["num"] * den_sq

    def _update(num: int, den_sq: int, desc: tuple):
        state["num"] = num
        state["den_sq"] = den_sq
        state["desc"] = desc
        state["init"] = True

    def _try_vv(pa: Tuple[int, int], pb: Tuple[int, int],
                desc: Optional[tuple] = None):
        dx = pa[0] - pb[0]
        dy = pa[1] - pb[1]
        sq = dx * dx + dy * dy
        if _better(sq, 1):
            _update(sq, 1, desc if desc is not None else ("vv", pa, pb))

    def _try_foot(ep: Tuple[int, int],
                  sa: Tuple[int, int], sb: Tuple[int, int],
                  ep_is_a: bool):
        """端点 ep 到线段 (sa, sb) 的垂足候选 (整数 t-bound 判定)。 / Foot candidate from endpoint ep to segment (sa, sb) (integer t-bound)."""
        abx = sb[0] - sa[0]
        aby = sb[1] - sa[1]
        den = abx * abx + aby * aby
        if den == 0:
            if ep_is_a:
                _try_vv(ep, sa)
            else:
                _try_vv(sa, ep)
            return
        apx = ep[0] - sa[0]
        apy = ep[1] - sa[1]
        num = apx * abx + apy * aby
        if num <= 0:
            if ep_is_a:
                _try_vv(ep, sa)
            else:
                _try_vv(sa, ep)
            return
        if num >= den:
            if ep_is_a:
                _try_vv(ep, sb)
            else:
                _try_vv(sb, ep)
            return
        # t ∈ (0, 1): 垂足坐标为 (fx_n / den, fy_n / den), 整数 / foot as integer rationals
        fx_n = sa[0] * den + num * abx
        fy_n = sa[1] * den + num * aby
        dx_n = ep[0] * den - fx_n
        dy_n = ep[1] * den - fy_n
        sq_num = dx_n * dx_n + dy_n * dy_n
        sq_den = den * den
        if _better(sq_num, sq_den):
            if ep_is_a:
                _update(sq_num, sq_den, ("vf", ep, fx_n, fy_n, den))
            else:
                _update(sq_num, sq_den, ("fv", fx_n, fy_n, den, ep))

    # 边 × 边: 整数相交检测 (短路返回 0) + 端点-垂足候选 / edge×edge: int intersect (short-circuit 0) + feet
    for i1, i2 in ea:
        for j1, j2 in eb:
            a1, a2 = A[i1], A[i2]
            b1, b2 = B[j1], B[j2]
            if _seg_intersect_int(a1[0], a1[1], a2[0], a2[1],
                                  b1[0], b1[1], b2[0], b2[1]):
                mx = (a1[0] + a2[0] + b1[0] + b2[0]) // 4
                my = (a1[1] + a2[1] + b1[1] + b2[1]) // 4
                pt = _to_lonlat(mx, my)
                return 0.0, pt, pt
            _try_foot(a1, b1, b2, True)
            _try_foot(a2, b1, b2, True)
            _try_foot(b1, a1, a2, False)
            _try_foot(b2, a1, a2, False)

    # 点 × 边 (一方为 Point) / point×edge (one side Point)
    if not ea:
        p = A[0]
        for j1, j2 in eb:
            _try_foot(p, B[j1], B[j2], True)
    if not eb:
        p = B[0]
        for i1, i2 in ea:
            _try_foot(p, A[i1], A[i2], False)

    # 全顶点-顶点对兜底 / all vertex–vertex fallback
    for ca in A:
        for cb in B:
            _try_vv(ca, cb)

    # 末端: 反向合成胜出对 (P*, Q*) 经纬度 + 1 次 Vincenty / terminal: synthesize winner lon/lat + 1× Vincenty
    desc = state["desc"]
    kind = desc[0]
    if kind == "vv":
        pa_ll = _to_lonlat(*desc[1])
        pb_ll = _to_lonlat(*desc[2])
    elif kind == "vf":
        pa_ll = _to_lonlat(*desc[1])
        pb_ll = _foot_to_lonlat(desc[2], desc[3], desc[4])
    else:  # "fv"
        pa_ll = _foot_to_lonlat(desc[1], desc[2], desc[3])
        pb_ll = _to_lonlat(*desc[4])

    dist_m = vincenty_fast(pa_ll[1], pa_ll[0], pb_ll[1], pb_ll[0]) * 1000
    return dist_m, pa_ll, pb_ll


# ============================================================================
# 通用几何要素推理器 / General geometry-feature reasoner
# ============================================================================

class GeoSotGeometryReasoner:
    """
    GeoSOT 通用几何要素距离推理器。
    / GeoSOT general geometry-feature distance reasoner.

    自动检测 Point/LineString/Polygon 类型，支持全部 3×3 距离组合。
    / Auto-detect Point/LineString/Polygon; all 3×3 distance combinations.
    """

    def __init__(self, verbose: bool = True, enable_postgis: bool = True):
        self.verbose = verbose
        self._pg_conn = None
        self._driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._database = NEO4J_DATABASE
        self._session = self._driver.session(database=self._database)
        self._session.run("RETURN 1").consume()
        # 客户端要素信息缓存. 与 PG 端 pg_prewarm 等价对照: 把 GR 端
        # / Client feature-info cache; PG pg_prewarm counterpart: preload GR-side
        # 常驻的"顶点编码 + 数据源"信息预先拉到 Python 进程内存,
        # / resident "vertex codes + source" into Python process memory,
        # 让正式计时阶段两侧都是 RAM-resident + warm cache.
        # / so timed runs are RAM-resident + warm-cache on both sides.
        self._feat_cache: Dict[str, Optional[Dict]] = {}
        if verbose:
            self._log(f"[Neo4j] {NEO4J_URI} / {self._database}")
        if enable_postgis:
            self._init_postgis()

    def _log(self, msg: str):
        if self.verbose:
            try:
                print(msg)
            except (UnicodeEncodeError, UnicodeDecodeError):
                print(msg.encode("utf-8", errors="replace").decode("utf-8"))

    def _run(self, cypher: str, **params) -> List[Dict]:
        try:
            if self._session is None:
                self._session = self._driver.session(database=self._database)
            return [dict(r) for r in self._session.run(cypher, **params)]
        except Exception as e:
            self._log(f"[ERROR] Cypher: {e}")
            return []

    def _init_postgis(self):
        try:
            import psycopg2
            self._pg_conn = psycopg2.connect(**PG_CONFIG)
            if self.verbose:
                self._log(
                    f"[PostGIS] {PG_CONFIG['host']}:{PG_CONFIG['port']}/"
                    f"{PG_CONFIG['dbname']}")
        except ImportError:
            self._log("[WARN] psycopg2 未安装")
        except Exception as e:
            self._log(f"[WARN] PostGIS 连接失败: {e}")

    # ------------------------------------------------------------------
    # Neo4j 查询 / Neo4j queries
    # ------------------------------------------------------------------

    def get_feature_info(self, code_id: str) -> Optional[Dict]:
        """
        查询要素的几何类型、顶点编码、数据源 (含客户端缓存).
        / Query geometry type, vertex codes, source (with client cache).

        缓存策略与 PG 端 pg_prewarm 对等: 同一 code_id 第二次访问直接
        命中 self._feat_cache, 不再走 Bolt 协议+Cypher; 与 PG buffer
        cache + plan cache 形成同口径的"warm cache"对照.
        / Cache mirrors PG pg_prewarm: second hit on code_id uses
          self._feat_cache (no Bolt/Cypher); pairs with PG buffer+plan cache
          as a same-baseline warm-cache control.

        @param code_id feature_id 或 main_code / feature_id or main_code
        @returns {code_id, main_code, geometry_type, source, vertex_codes}
        """
        cached = self._feat_cache.get(code_id)
        if cached is not None or code_id in self._feat_cache:
            return cached

        rows = self._run(
            """
            MATCH (g:SpatialFeature)-[:HAS_CENTROID_CODE]->(c:GeoSOTCentroidCode)
            WHERE g.feature_id = $cid OR c.main_code = $cid
            OPTIONAL MATCH (c)-[:HAS_GEOMETRY_TYPE]->(gt:GeoFeatureType)
            OPTIONAL MATCH (c)-[:FROM_SOURCE]->(s:DataSource)
            OPTIONAL MATCH (g)-[hv:HAS_VERTEX_CODE]->(v:GeoSOTVertexCode)
            WITH g, c, gt, s, v, hv
            ORDER BY hv.seq
            WITH g, c,
                 gt.type_name AS geometry_type,
                 s.name AS source,
                 collect(CASE WHEN v IS NOT NULL
                         THEN {code: v.vertex_code, idx: hv.seq}
                         ELSE null END) AS raw_verts
            RETURN g.feature_id AS code_id,
                   c.main_code AS main_code,
                   geometry_type,
                   source,
                   [x IN raw_verts WHERE x IS NOT NULL | x.code]
                       AS vertex_codes
            """,
            cid=code_id)

        if not rows:
            self._feat_cache[code_id] = None
            return None
        r = rows[0]
        geo_type = r.get("geometry_type") or "Point"
        vcodes = r.get("vertex_codes") or []

        info = {
            "code_id": str(r.get("code_id", "")),
            "main_code": str(r.get("main_code", "")),
            "geometry_type": geo_type,
            "source": str(r.get("source", "") or ""),
            "vertex_codes": [str(v).strip().upper() for v in vcodes],
        }
        # 同时把别名 (code_id 与 main_code) 写入缓存, 避免下次以另一种
        # / Also cache aliases (code_id and main_code) to avoid miss when
        # 标识查询时 cache miss
        # / queried by the other identifier
        self._feat_cache[code_id] = info
        if info["code_id"]:
            self._feat_cache.setdefault(info["code_id"], info)
        if info["main_code"]:
            self._feat_cache.setdefault(info["main_code"], info)
        return info

    def prefetch_features(self, code_ids) -> Dict:
        """
        批量预拉取要素信息进客户端缓存 (与 pg_prewarm 等价对照).
        / Batch-prefetch feature info into client cache (pg_prewarm counterpart).

        论文 Methodology 中应披露: GR 端在正式计时之前一次性把要素
        顶点编码 + 数据源信息加载到 Python 进程内存, 与 PG 端 pg_prewarm
        把表+索引拉进 shared_buffers 形成对等的"warm cache"基线.
        / Methodology note: before timed runs, GR loads vertex codes + source
          into Python RAM once; PG pg_prewarm loads tables/indexes into
          shared_buffers — equivalent warm-cache baseline.

        @param code_ids 可迭代的 code_id / main_code / iterable of ids
        @returns {n_total, n_hit, n_fetch, n_miss, elapsed_sec}
        """
        import time as _time
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:  # pragma: no cover
            _tqdm = None
        t0 = _time.perf_counter()
        n_hit = 0
        n_fetch = 0
        n_miss = 0
        n_total = 0
        # 去重保序: 抽样阶段可能已写入部分 cache, 避免重复 Bolt 往返 / dedupe preserving order; avoid repeat Bolt
        seen = set()
        unique_ids = []
        for cid in code_ids:
            if cid in seen:
                continue
            seen.add(cid)
            unique_ids.append(cid)
        iterator = unique_ids
        if _tqdm is not None and self.verbose:
            iterator = _tqdm(
                unique_ids, desc="  prefetch features",
                unit="feat", ncols=90, leave=False)
        for cid in iterator:
            n_total += 1
            if cid in self._feat_cache:
                n_hit += 1
                continue
            info = self.get_feature_info(cid)
            if info is None:
                n_miss += 1
            else:
                n_fetch += 1
            # 每 200 条刷新一次, 避免长时间无输出被误认为卡死 / progress every 200 items
            if self.verbose and _tqdm is None and n_total % 200 == 0:
                elapsed = _time.perf_counter() - t0
                rate = n_total / elapsed if elapsed > 0 else 0.0
                self._log(
                    f"  prefetch {n_total}/{len(unique_ids)} "
                    f"({rate:.1f} feat/s, fetch={n_fetch}, hit={n_hit})")
        return {
            "n_total": n_total,
            "n_hit": n_hit,
            "n_fetch": n_fetch,
            "n_miss": n_miss,
            "elapsed_sec": _time.perf_counter() - t0,
        }

    def list_features(
        self,
        geo_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 50,
        random_sample: bool = False,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """
        列出要素，支持按几何类型和数据源筛选。
        / List features; filter by geometry type and data source.

        @param geo_type      几何类型过滤 ("Point" / "LineString" / "Polygon")
                             / geometry type filter
        @param source        数据源名称过滤 / data-source name filter
        @param limit         最大返回数 / max rows
        @param random_sample 是否做无偏随机抽样.
                             / whether to do unbiased random sampling.
                             - False (默认): 按 (geo_type, vc) 升序返回头 N 条,
                               适合人工浏览, 但因 ORDER BY vc 会优先返回顶点
                               数最少的简单几何, 不可用于 benchmark 候选池.
                               / Default: first N by (geo_type, vc); fine for browsing,
                                 biased to simple geometries — not for benchmarks.
                             - True: 先在 Neo4j 端取出全集, 再在 Python 端做
                               (可 seed 控制的) `random.sample`, 保证候选池
                               在顶点数等几何复杂度维度上不带选择偏差,
                               且与 --seed 一起使用时跑多次完全可复现.
                               benchmark 必须使用此模式.
                               / Fetch full set then Python random.sample (seedable);
                                 unbiased in complexity; required for benchmarks.
        @param seed          random_sample=True 时的随机种子;
                             None = 用全局 random (不可复现).
                             / RNG seed when random_sample=True; None = global random.
        """
        where_parts = []
        params: Dict = {}

        if geo_type:
            where_parts.append("gt.type_name = $geo_type")
            params["geo_type"] = geo_type
        if source:
            where_parts.append("s.name = $source")
            params["source"] = source

        type_match = ("MATCH" if geo_type
                      else "OPTIONAL MATCH")
        source_match = ("MATCH" if source
                        else "OPTIONAL MATCH")

        if random_sample:
            # 客户端 seeded sample: 先取全集 (不带 ORDER BY 避免按 vc 偏置),
            # / Client seeded sample: fetch full set (no ORDER BY to avoid vc bias),
            # 再用 Python-side random.Random(seed).sample 做无偏均匀抽样.
            # / then Python-side random.Random(seed).sample for unbiased sampling.
            # 与 Cypher 的 ORDER BY rand() 相比, 这种方式可被 --seed 完全
            # / Unlike Cypher ORDER BY rand(), this is fully reproducible with --seed
            # 复现, 是 benchmark 的正确做法.
            # / and is the correct approach for benchmarks.
            cypher = f"""
            MATCH (c:GeoSOTCentroidCode)<-[:HAS_CENTROID_CODE]-(g:SpatialFeature)
            {type_match} (c)-[:HAS_GEOMETRY_TYPE]->(gt:GeoFeatureType)
            {source_match} (c)-[:FROM_SOURCE]->(s:DataSource)
            OPTIONAL MATCH (g)-[hv:HAS_VERTEX_CODE]->(v:GeoSOTVertexCode)
            WITH c, g, gt.type_name AS geo_type, s.name AS source, count(v) AS vc
            {"WHERE " + " AND ".join(where_parts) if where_parts else ""}
            RETURN g.feature_id AS code_id,
                   c.main_code AS main_code,
                   geo_type,
                   source,
                   vc AS vertex_count
            """
            all_rows = self._run(cypher, **params)
            if len(all_rows) <= limit:
                return all_rows
            rng = random.Random(seed) if seed is not None else random
            return rng.sample(all_rows, limit)

        params["limit"] = limit
        cypher = f"""
        MATCH (c:GeoSOTCentroidCode)<-[:HAS_CENTROID_CODE]-(g:SpatialFeature)
        {type_match} (c)-[:HAS_GEOMETRY_TYPE]->(gt:GeoFeatureType)
        {source_match} (c)-[:FROM_SOURCE]->(s:DataSource)
        OPTIONAL MATCH (g)-[hv:HAS_VERTEX_CODE]->(v:GeoSOTVertexCode)
        WITH c, g, gt.type_name AS geo_type, s.name AS source, count(v) AS vc
        {"WHERE " + " AND ".join(where_parts) if where_parts else ""}
        RETURN g.feature_id AS code_id,
               c.main_code AS main_code,
               geo_type,
               source,
               vc AS vertex_count
        ORDER BY geo_type, vc
        LIMIT $limit
        """
        return self._run(cypher, **params)

    # ------------------------------------------------------------------
    # 坐标提取 / Coordinate extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_coords_decode(
        info: Dict,
    ) -> List[Tuple[float, float]]:
        """网格解码法获取坐标 / Get coordinates via grid decode"""
        if info["geometry_type"] == "Point" or not info["vertex_codes"]:
            mc = info["main_code"].strip().upper()
            return [fast_code_to_center(mc)]
        return [fast_code_to_center(vc) for vc in info["vertex_codes"]]

    @staticmethod
    def _get_all_codes(info: Dict) -> List[str]:
        """获取要素的所有相关编码（Point 用 main_code，其余用顶点编码） / All related codes (Point→main_code, else vertex codes)"""
        if info["geometry_type"] == "Point" or not info["vertex_codes"]:
            return [info["main_code"].strip().upper()]
        return info["vertex_codes"]

    # ------------------------------------------------------------------
    # PostGIS 对比 / PostGIS comparison
    # ------------------------------------------------------------------

    def _postgis_distance_and_bearing(
        self, main_code_a: str, source_a: str,
        main_code_b: str, source_b: str,
    ) -> Optional[Tuple[float, float]]:
        """
        PostGIS 精度基准：距离 + 方位角（同一 geography 最近点对）。
        / PostGIS accuracy baseline: distance + bearing (same geography nearest pair).

        库内几何为 EPSG:4490（CGCS2000 经纬度）。距离与最近点均在
        geography 上计算：ST_Distance(ga, gb) 与 ST_ShortestLine(ga, gb)
        共用椭球最短路径；线的起终点再交 ST_Azimuth(geography)。
        要求 PostGIS ≥ 3.4（geography 版 ST_ShortestLine）。
        / Geometries are EPSG:4490 (CGCS2000 lon/lat). Distance and nearest
          points use geography: ST_Distance and ST_ShortestLine share the
          ellipsoidal shortest path; endpoints feed ST_Azimuth(geography).
          Requires PostGIS ≥ 3.4 (geography ST_ShortestLine).

        @param main_code_a 要素 A 的 main_code / feature A main_code
        @param source_a    要素 A 的 Source 名称（映射到 PostGIS 表） / A Source → PG table
        @param main_code_b 要素 B 的 main_code / feature B main_code
        @param source_b    要素 B 的 Source 名称 / B Source name
        @returns (distance_m, bearing_deg) 或 None / or None
        """
        if self._pg_conn is None:
            return None
        table_a = SOURCE_TO_PG_TABLE.get(source_a)
        table_b = SOURCE_TO_PG_TABLE.get(source_b)
        if not table_a or not table_b:
            self._log(
                f"[WARN] PostGIS 表映射缺失: "
                f"source_a={source_a!r} → {table_a}, "
                f"source_b={source_b!r} → {table_b}")
            return None
        try:
            cur = self._pg_conn.cursor()
            # 一次 ST_ShortestLine(geography)：长度 = ST_Distance，端点 = 最近点对 / one ShortestLine: length=distance, ends=nearest pair
            cur.execute(
                f"WITH pair AS ("
                f"  SELECT a.geometry::geography AS ga,"
                f"         b.geometry::geography AS gb"
                f"    FROM {table_a} a, {table_b} b"
                f"   WHERE a.main_code = %s AND b.main_code = %s"
                f"   LIMIT 1"
                f"), sl AS ("
                f"  SELECT ST_Distance(ga, gb) AS d_m,"
                f"         ST_ShortestLine(ga, gb) AS line"
                f"    FROM pair"
                f")"
                f"SELECT d_m,"
                f"       degrees(ST_Azimuth("
                f"         ST_StartPoint(line::geometry)::geography,"
                f"         ST_EndPoint(line::geometry)::geography))"
                f"  FROM sl",
                (main_code_a, main_code_b))
            row = cur.fetchone()
            cur.close()
            if row and row[0] is not None:
                dist = float(row[0])
                brg = float(row[1]) if row[1] is not None else 0.0
                return dist, brg
            return None
        except Exception as e:
            self._log(f"[WARN] PostGIS 查询失败: {e}")
            try:
                self._pg_conn.rollback()
            except Exception:
                pass
            return None

    # ------------------------------------------------------------------
    # 核心: 通用距离计算 / Core: general distance
    # ------------------------------------------------------------------

    def calculate_distance(
        self, code_a: str, code_b: str,
        compare_postgis: bool = True,
    ) -> Dict:
        """
        计算两个任意几何类型要素之间的最短距离。
        / Shortest distance between two features of any geometry types.

        自动检测 Point/LineString/Polygon 类型，使用三种方法:
        / Auto-detect Point/LineString/Polygon; three methods:
        1. 网格解码 (fast_code_to_center → 局部平面最短距离)
           / grid decode (fast_code_to_center → local planar min distance)
        2. 图推理 (公共参考坐标 + 逐级偏移 → 局部平面最短距离)
           / graph reasoning (shared ref + level offsets → local planar min)
        3. PostGIS ST_Distance (可选精度基准) / optional accuracy baseline

        @param code_a 要素 A 标识 (feature_id 或 main_code) / feature A id
        @param code_b 要素 B 标识 / feature B id
        @param compare_postgis 是否对比 PostGIS / whether to compare PostGIS
        @returns 结果字典 / result dict
        """
        result: Dict = {
            "code_a": code_a, "code_b": code_b, "success": False,
        }

        _t = time.perf_counter

        # ---- 1) 获取要素信息 (Neo4j) / fetch feature info ----
        t0 = _t()
        info_a = self.get_feature_info(code_a)
        t_neo4j_a = (_t() - t0) * 1000

        t0 = _t()
        info_b = self.get_feature_info(code_b)
        t_neo4j_b = (_t() - t0) * 1000

        if not info_a:
            result["error"] = f"要素 A 未找到: {code_a}"
            return result
        if not info_b:
            result["error"] = f"要素 B 未找到: {code_b}"
            return result

        type_a = info_a["geometry_type"]
        type_b = info_b["geometry_type"]
        min_verts = {"Point": 0, "LineString": 2, "Polygon": 3}
        vc_a = len(info_a["vertex_codes"])
        vc_b = len(info_b["vertex_codes"])

        if type_a != "Point" and vc_a < min_verts.get(type_a, 0):
            result["error"] = (
                f"要素 A ({type_a}) 顶点不足: {vc_a} < "
                f"{min_verts.get(type_a, 0)}")
            return result
        if type_b != "Point" and vc_b < min_verts.get(type_b, 0):
            result["error"] = (
                f"要素 B ({type_b}) 顶点不足: {vc_b} < "
                f"{min_verts.get(type_b, 0)}")
            return result

        result.update({
            "feature_a": {
                "code_id": info_a["code_id"],
                "main_code": info_a["main_code"],
                "geometry_type": type_a,
                "source": info_a["source"],
                "vertex_count": vc_a,
            },
            "feature_b": {
                "code_id": info_b["code_id"],
                "main_code": info_b["main_code"],
                "geometry_type": type_b,
                "source": info_b["source"],
                "vertex_count": vc_b,
            },
            "combination": f"{type_a} × {type_b}",
        })

        if self.verbose:
            self._log(
                f"\n[推理] A: {type_a} ({info_a['code_id']}, "
                f"{vc_a} 顶点, 源={info_a['source'] or 'N/A'})")
            self._log(
                f"[推理] B: {type_b} ({info_b['code_id']}, "
                f"{vc_b} 顶点, 源={info_b['source'] or 'N/A'})")
            self._log(f"[推理] 组合: {type_a} × {type_b}")

        # ---- 提取编码 + 构建逐对公共祖先参考矩阵 / codes + per-pair ancestor cos matrix ----
        codes_a = self._get_all_codes(info_a)
        codes_b = self._get_all_codes(info_b)
        cos_matrix = _build_cos_ref_matrix(codes_a, codes_b)

        # ---- 2) 网格解码法 (分步计时) / grid decode (step timing) ----
        t0 = _t()
        dec_a = self._get_coords_decode(info_a)
        t_dec_coord_a = (_t() - t0) * 1000

        t0 = _t()
        dec_b = self._get_coords_decode(info_b)
        t_dec_coord_b = (_t() - t0) * 1000

        t0 = _t()
        dec_dist, dec_near_a, dec_near_b = geometry_min_distance_m(
            dec_a, type_a, dec_b, type_b,
            cos_ref_matrix=cos_matrix)
        t_dec_geom = (_t() - t0) * 1000

        t0 = _t()
        dec_brg = compute_bearing(
            dec_near_a[0], dec_near_a[1],
            dec_near_b[0], dec_near_b[1]) if dec_dist > 0.001 else 0.0
        t_dec_bearing = (_t() - t0) * 1000

        t_dec_total = t_dec_coord_a + t_dec_coord_b + t_dec_geom + t_dec_bearing

        result["decode"] = {
            "distance_m": round(dec_dist, 4),
            "bearing": round(dec_brg, 4),
            "nearest_a": tuple(round(x, 8) for x in dec_near_a),
            "nearest_b": tuple(round(x, 8) for x in dec_near_b),
            "timing": {
                "coord_a_ms": round(t_dec_coord_a, 3),
                "coord_b_ms": round(t_dec_coord_b, 3),
                "geometry_distance_ms": round(t_dec_geom, 3),
                "bearing_ms": round(t_dec_bearing, 3),
                "total_ms": round(t_dec_total, 3),
            },
        }

        # ---- 3) 图推理法 (整数谓词版, 分步计时) / graph (integer predicates, step timing) ----
        # G3R 算子叙事: 图遍历输出整数偏移 -> 整数谓词扫描 (相交/垂足/argmin)
        # / G3R narrative: graph → integer offsets → integer predicates (intersect/foot/argmin)
        # -> 末端 1 次坐标合成 + 1 次 Vincenty。除最后一步外不出现浮点经纬度。
        # / → terminal 1× coord synthesis + 1× Vincenty; no float lon/lat before the end.
        t_gr_prepare = 0.0

        t0 = _t()
        gr_a_int, gr_b_int, gr_ref_lon, gr_ref_lat, gr_cos_ref = (
            fast_estimate_vertices_int(codes_a, codes_b))
        t_gr_estimate = (_t() - t0) * 1000

        t0 = _t()
        gr_dist, gr_near_a, gr_near_b = geometry_min_distance_m_int(
            gr_a_int, type_a, gr_b_int, type_b,
            gr_ref_lon, gr_ref_lat, gr_cos_ref)
        t_gr_geom = (_t() - t0) * 1000

        t0 = _t()
        gr_brg = compute_bearing(
            gr_near_a[0], gr_near_a[1],
            gr_near_b[0], gr_near_b[1]) if gr_dist > 0.001 else 0.0
        t_gr_bearing = (_t() - t0) * 1000

        t_gr_total = t_gr_prepare + t_gr_estimate + t_gr_geom + t_gr_bearing

        result["graph"] = {
            "distance_m": round(gr_dist, 4),
            "bearing": round(gr_brg, 4),
            "nearest_a": tuple(round(x, 8) for x in gr_near_a),
            "nearest_b": tuple(round(x, 8) for x in gr_near_b),
            "timing": {
                "prepare_codes_ms": round(t_gr_prepare, 3),
                "vertex_estimate_ms": round(t_gr_estimate, 3),
                "geometry_distance_ms": round(t_gr_geom, 3),
                "bearing_ms": round(t_gr_bearing, 3),
                "total_ms": round(t_gr_total, 3),
            },
        }

        # ---- 4) PostGIS 精度基准 (分步计时) / PostGIS baseline (step timing) ----
        if compare_postgis and self._pg_conn:
            t0 = _t()
            pg_result = self._postgis_distance_and_bearing(
                info_a["main_code"], info_a["source"],
                info_b["main_code"], info_b["source"])
            t_pg_query = (_t() - t0) * 1000
            if pg_result is not None:
                pg_dist, pg_brg = pg_result
                result["postgis"] = {
                    "distance_m": round(pg_dist, 4),
                    "bearing": round(pg_brg, 4),
                    "timing": {
                        "st_distance_ms": round(t_pg_query, 3),
                    },
                }

        # ---- 5) 汇总计时 / timing summary ----
        result["timing_summary"] = {
            "neo4j_info_a_ms": round(t_neo4j_a, 3),
            "neo4j_info_b_ms": round(t_neo4j_b, 3),
            "neo4j_total_ms": round(t_neo4j_a + t_neo4j_b, 3),
            "decode_total_ms": round(t_dec_total, 3),
            "graph_total_ms": round(t_gr_total, 3),
        }
        pg_res = result.get("postgis")
        if pg_res:
            result["timing_summary"]["postgis_total_ms"] = (
                pg_res["timing"]["st_distance_ms"])

        # ---- 6) 误差对比 / error comparison ----
        comp: Dict = {}
        comp["decode_vs_graph_m"] = round(dec_dist - gr_dist, 6)

        def _bearing_diff(a: float, b: float) -> float:
            d = a - b
            if d > 180:
                d -= 360
            elif d < -180:
                d += 360
            return d

        dec_brg = result["decode"]["bearing"]
        gr_brg = result["graph"]["bearing"]
        comp["decode_vs_graph_bearing"] = round(
            _bearing_diff(dec_brg, gr_brg), 6)

        if pg_res and pg_res["distance_m"] > 0.01:
            pg_d = pg_res["distance_m"]
            pg_brg = pg_res["bearing"]
            comp["decode_vs_postgis_m"] = round(dec_dist - pg_d, 6)
            comp["graph_vs_postgis_m"] = round(gr_dist - pg_d, 6)
            comp["decode_pct"] = round(
                (dec_dist - pg_d) / pg_d * 100, 8)
            comp["graph_pct"] = round(
                (gr_dist - pg_d) / pg_d * 100, 8)
            comp["decode_vs_postgis_bearing"] = round(
                _bearing_diff(dec_brg, pg_brg), 6)
            comp["graph_vs_postgis_bearing"] = round(
                _bearing_diff(gr_brg, pg_brg), 6)
        result["comparison"] = comp

        result["success"] = True
        return result

    # ------------------------------------------------------------------
    # 格式化输出 / Formatted output
    # ------------------------------------------------------------------

    def format_result(self, result: Dict) -> str:
        """
        格式化推理结果，包含每个方法的分步耗时明细。
        / Format reasoning result with per-method step timing.

        @param result calculate_distance 返回的结果字典 / result from calculate_distance
        @returns 格式化的字符串 / formatted string
        """
        W = 76
        lines = ["=" * W, "GeoSOT 通用几何要素距离推理", "=" * W]

        if not result.get("success"):
            lines.append(f"  [ERROR] {result.get('error', '未知错误')}")
            return "\n".join(lines)

        for tag, key in (("要素 A", "feature_a"), ("要素 B", "feature_b")):
            f = result[key]
            lines.append(
                f"  {tag}: {f['geometry_type']} "
                f"({f['code_id']})")
            lines.append(
                f"    main_code: {f['main_code']}  "
                f"顶点: {f['vertex_count']}  "
                f"源: {f['source'] or 'N/A'}")

        lines.append(f"  组合: {result['combination']}")
        lines.append("")

        SEP = f"  {'─' * (W - 4)}"
        TSEP = f"  {'·' * (W - 4)}"

        # ---- Neo4j 查询耗时 / Neo4j query timing ----
        ts = result.get("timing_summary", {})
        lines.append(SEP)
        lines.append("  [Neo4j] 要素信息查询")
        lines.append(
            f"    查询要素 A:       {ts.get('neo4j_info_a_ms', 0):>10.3f} ms")
        lines.append(
            f"    查询要素 B:       {ts.get('neo4j_info_b_ms', 0):>10.3f} ms")
        lines.append(
            f"    合计:             {ts.get('neo4j_total_ms', 0):>10.3f} ms")

        # ---- 网格解码法 / grid decode method ----
        dec = result.get("decode", {})
        dt = dec.get("timing", {})
        lines.append(SEP)
        lines.append("  [网格解码] fast_code_to_center + Vincenty")
        lines.append(TSEP)
        lines.append("    分步耗时:")
        lines.append(
            f"      ① 解码坐标 A:   {dt.get('coord_a_ms', 0):>10.3f} ms")
        lines.append(
            f"      ② 解码坐标 B:   {dt.get('coord_b_ms', 0):>10.3f} ms")
        lines.append(
            f"      ③ 几何距离计算: {dt.get('geometry_distance_ms', 0):>10.3f} ms")
        lines.append(
            f"      ④ 方位角计算:   {dt.get('bearing_ms', 0):>10.3f} ms")
        lines.append(TSEP)
        lines.append(
            f"    合计:             {dt.get('total_ms', 0):>10.3f} ms")
        lines.append(
            f"    距离:             {dec.get('distance_m', 0):>14,.4f} m")
        lines.append(
            f"    方位角:                  {dec.get('bearing', 0):.4f}°")
        na = dec.get("nearest_a", (0, 0))
        nb = dec.get("nearest_b", (0, 0))
        lines.append(f"    最近点 A: ({na[0]:.8f}, {na[1]:.8f})")
        lines.append(f"    最近点 B: ({nb[0]:.8f}, {nb[1]:.8f})")

        # ---- 图推理法 / graph reasoning method ----
        gr = result.get("graph", {})
        gt = gr.get("timing", {})
        lines.append(SEP)
        lines.append("  [图推理] 整数偏移 + 整数谓词扫描 + 末端 1 次 Vincenty")
        lines.append(TSEP)
        lines.append("    分步耗时:")
        lines.append(
            f"      ① 准备编码列表: {gt.get('prepare_codes_ms', 0):>10.3f} ms")
        lines.append(
            f"      ② 顶点坐标估算: {gt.get('vertex_estimate_ms', 0):>10.3f} ms")
        lines.append(
            f"      ③ 几何距离计算: {gt.get('geometry_distance_ms', 0):>10.3f} ms")
        lines.append(
            f"      ④ 方位角计算:   {gt.get('bearing_ms', 0):>10.3f} ms")
        lines.append(TSEP)
        lines.append(
            f"    合计:             {gt.get('total_ms', 0):>10.3f} ms")
        lines.append(
            f"    距离:             {gr.get('distance_m', 0):>14,.4f} m")
        lines.append(
            f"    方位角:                  {gr.get('bearing', 0):.4f}°")
        na = gr.get("nearest_a", (0, 0))
        nb = gr.get("nearest_b", (0, 0))
        lines.append(f"    最近点 A: ({na[0]:.8f}, {na[1]:.8f})")
        lines.append(f"    最近点 B: ({nb[0]:.8f}, {nb[1]:.8f})")

        # ---- PostGIS ----
        pg = result.get("postgis")
        if pg:
            pt = pg.get("timing", {})
            lines.append(SEP)
            lines.append(
                "  [PostGIS] ST_Distance + ST_ShortestLine/Azimuth "
                "(geography, EPSG:4490)")
            lines.append(TSEP)
            lines.append("    分步耗时:")
            lines.append(
                f"      ① SQL 查询:    {pt.get('st_distance_ms', 0):>10.3f} ms")
            lines.append(TSEP)
            lines.append(
                f"    合计:             {pt.get('st_distance_ms', 0):>10.3f} ms")
            lines.append(
                f"    距离:             {pg['distance_m']:>14,.4f} m")
            lines.append(
                f"    方位角:                  {pg.get('bearing', 0):.4f}°")

        # ---- 总耗时汇总 / total timing summary ----
        lines.append(SEP)
        lines.append("  耗时汇总 (不含 Neo4j 查询):")
        lines.append(
            f"    网格解码:         {dt.get('total_ms', 0):>10.3f} ms")
        lines.append(
            f"    图推理:           {gt.get('total_ms', 0):>10.3f} ms")
        if pg:
            pt = pg.get("timing", {})
            lines.append(
                f"    PostGIS:          "
                f"{pt.get('st_distance_ms', 0):>10.3f} ms")

        # ---- 误差对比 / error comparison ----
        comp = result.get("comparison", {})
        if comp:
            lines.append(SEP)
            lines.append("  距离误差:")
            lines.append(
                f"    解码 − 图推理: "
                f"{comp.get('decode_vs_graph_m', 0):+.6f} m")
            if "decode_vs_postgis_m" in comp:
                lines.append(
                    f"    解码 − PostGIS: "
                    f"{comp.get('decode_vs_postgis_m', 0):+.6f} m "
                    f"({comp.get('decode_pct', 0):+.6f}%)")
                lines.append(
                    f"    图推理 − PostGIS: "
                    f"{comp.get('graph_vs_postgis_m', 0):+.6f} m "
                    f"({comp.get('graph_pct', 0):+.6f}%)")
            lines.append("  方位角误差:")
            lines.append(
                f"    解码 − 图推理: "
                f"{comp.get('decode_vs_graph_bearing', 0):+.6f}°")
            if "decode_vs_postgis_bearing" in comp:
                lines.append(
                    f"    解码 − PostGIS: "
                    f"{comp.get('decode_vs_postgis_bearing', 0):+.6f}°")
                lines.append(
                    f"    图推理 − PostGIS: "
                    f"{comp.get('graph_vs_postgis_bearing', 0):+.6f}°")

        lines.append("=" * W)
        return "\n".join(lines)

    def close(self):
        for res in (self._pg_conn, self._session, self._driver):
            if res:
                try:
                    res.close()
                except Exception:
                    pass


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GeoSOT 通用几何要素距离推理 (Point/LineString/Polygon) / GeoSOT general geometry distance reasoning",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "codes", nargs="*",
        help="两个要素标识 (feature_id 或 main_code) / two feature ids")
    parser.add_argument(
        "--list", action="store_true",
        help="列出要素 / list features")
    parser.add_argument(
        "--type", type=str, default=None,
        choices=["Point", "LineString", "Polygon"],
        help="按几何类型筛选 / filter by geometry type")
    parser.add_argument(
        "--source", type=str, default=None,
        help="按数据源筛选 / filter by data source")
    parser.add_argument(
        "--limit", type=int, default=50,
        help="列表最大返回数 (默认 50) / max list rows (default 50)")
    parser.add_argument(
        "--no-postgis", action="store_true",
        help="不对比 PostGIS / skip PostGIS compare")
    parser.add_argument(
        "--json", action="store_true",
        help="JSON 格式输出 / JSON output")
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="安静模式 / quiet mode")

    args = parser.parse_args()
    verbose = not args.quiet
    enable_pg = not args.no_postgis

    reasoner = GeoSotGeometryReasoner(
        verbose=verbose, enable_postgis=enable_pg)

    try:
        if args.list:
            features = reasoner.list_features(
                geo_type=args.type, source=args.source,
                limit=args.limit)
            if not features:
                print("未找到匹配要素。")
                return
            print(f"\n要素列表 ({len(features)} 条):")
            print(f"  {'code_id':<40s} {'main_code':<36s} "
                  f"{'类型':<12s} {'顶点':>4s}  {'数据源'}")
            print(f"  {'─' * 40} {'─' * 36} {'─' * 12} "
                  f"{'─' * 4}  {'─' * 20}")
            for f in features:
                print(
                    f"  {str(f.get('code_id', '')):<40s} "
                    f"{str(f.get('main_code', '')):<36s} "
                    f"{str(f.get('geo_type', '')):<12s} "
                    f"{f.get('vertex_count', 0):>4d}  "
                    f"{str(f.get('source', ''))}")
            return

        if len(args.codes) < 2:
            parser.print_help()
            print("\n示例:")
            print("  python geosot_geometry_reasoning.py --list --type Polygon")
            print("  python geosot_geometry_reasoning.py <id_a> <id_b>")
            return

        result = reasoner.calculate_distance(
            args.codes[0], args.codes[1],
            compare_postgis=enable_pg)

        if args.json:
            import json
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(reasoner.format_result(result))

    finally:
        reasoner.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding: utf-8
"""
GeoSOT 空间推理模块 - 基于 BFS 的点与点空间推理
/ GeoSOT spatial reasoning module — BFS-based point-to-point spatial reasoning

基于 GB/T 40087-2021 编码体系和 Neo4j 知识图谱中的网格方向关系，
通过图遍历（BFS）实现两个 GeoSOT 编码点之间的距离和方向推理。
/ Based on the GB/T 40087-2021 coding system and grid direction relations in a Neo4j
knowledge graph, use graph traversal (BFS) to infer distance and direction between
two GeoSOT-coded points.

算法原理（V2 方案） / Algorithm (V2):
1. 从 Neo4j 查询两个编码的层级数字链（L1~L32，通过 CC_HAS_LEVEL_k_DIGIT 关系）
   / Query level digit chains (L1–L32 via CC_HAS_LEVEL_k_DIGIT) from Neo4j
2. 逐级比较找到首个分异层级 Ld / Find first differing level Ld
3. 在 Ld 及更深层级，通过 BFS 遍历 EAST_OF / NORTH_OF 关系计算网格位置差
   / At Ld and deeper levels, BFS EAST_OF / NORTH_OF to compute grid offsets
4. 将各级网格偏移量转换为经纬度偏移（基于 DMS 编码的网格尺寸）
   / Convert per-level grid offsets to lon/lat offsets (DMS grid sizes)
5. 通过解码共同父网格获取参考坐标，加上图推理偏移量得到估计坐标
   / Decode common parent grid as reference; add graph offsets for estimated coords
6. 使用 Vincenty 公式（CGCS2000 椭球）计算大地测量距离
   / Vincenty formula (CGCS2000 ellipsoid) for geodetic distance
7. 可选：连接 PostGIS (public.poi_new) 获取要素实际坐标作为精度基准
   / Optional: PostGIS (public.poi_new) actual coords as accuracy baseline

精度对比体系 / Accuracy comparison:
- 解码对比：图推理结果 vs 网格中心解码（code_to_center），衡量图推理的数学一致性
  / Decode check: graph vs code_to_center (math consistency of graph reasoning)
- PostGIS 对比：图推理结果 vs PostGIS 实际坐标（精度基准），衡量网格编码的空间精度损失
  / PostGIS check: graph vs real coords (spatial precision loss of grid coding)

使用方法 / Usage:
    python geosot_spatial_reasoning.py --verify                    # 验证图结构 / verify graph
    python geosot_spatial_reasoning.py <编码1> <编码2>             # 两点推理（含PostGIS对比） / pair reasoning (+PostGIS)
    python geosot_spatial_reasoning.py --no-compare <编码1> <编码2> # 不对比直接解码 / skip decode compare
    python geosot_spatial_reasoning.py --no-postgis <编码1> <编码2> # 不对比PostGIS / skip PostGIS
    python geosot_spatial_reasoning.py                             # 交互模式 / interactive
"""

import math
import os
import re
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from neo4j import GraphDatabase

PG_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "dbname": "postgis_35_sample",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", ""),
}


# ============================================================================
# GeoSOT 编码辅助函数 / GeoSOT encoding helpers
# ============================================================================

DIGIT_TO_ROWCOL = {
    "0": (0, 0),
    "1": (0, 1),
    "2": (1, 0),
    "3": (1, 1),
}

GRID_LEVEL_MAP = {
    lv: (f"GeoSOTLevelDigit{lv}", "digit")
    for lv in range(1, 33)
}


def _digit_to_bits(digit: str) -> Tuple[str, str]:
    """@returns (lat_bit, lon_bit)"""
    return {"0": ("0", "0"), "1": ("0", "1"),
            "2": ("1", "0"), "3": ("1", "1")}[digit]


def _bits31_to_degree(bits31: str) -> float:
    """31 位二进制（8度+6分+6秒+11秒以下）→ 十进制度 / 31-bit binary (8°+6′+6″+11 sub-arcsec) → decimal degrees"""
    d = int(bits31[0:8], 2)
    m = int(bits31[8:14], 2)
    s = int(bits31[14:20], 2)
    u = int(bits31[20:31], 2)
    return d + m / 60.0 + (s + u / 2048.0) / 3600.0


def code_to_bbox(code: str) -> Tuple[float, float, float, float]:
    """
    GeoSOT 编码 → (min_lon, max_lon, min_lat, max_lat)
    / GeoSOT code → (min_lon, max_lon, min_lat, max_lat)
    """
    code = code.strip().upper()
    body = code[1:]
    refine = body[1:]
    lat_bits: List[str] = []
    lon_bits: List[str] = []
    for d in refine:
        lb, lob = _digit_to_bits(d)
        lat_bits.append(lb)
        lon_bits.append(lob)
    remain = 31 - len(refine)
    lat_lo = "".join(lat_bits) + "0" * remain
    lon_lo = "".join(lon_bits) + "0" * remain
    lat_hi = "".join(lat_bits) + "1" * remain
    lon_hi = "".join(lon_bits) + "1" * remain
    return (
        _bits31_to_degree(lon_lo), _bits31_to_degree(lon_hi),
        _bits31_to_degree(lat_lo), _bits31_to_degree(lat_hi),
    )


def code_to_center(code: str) -> Tuple[float, float]:
    """GeoSOT 编码 → (center_lon, center_lat) / GeoSOT code → (center_lon, center_lat)"""
    lo_lon, hi_lon, lo_lat, hi_lat = code_to_bbox(code)
    return ((lo_lon + hi_lon) / 2.0, (lo_lat + hi_lat) / 2.0)


def reconstruct_code(chain: Dict[int, str], up_to_level: int) -> str:
    """从网格链重建 GeoSOT 编码前缀 / Rebuild GeoSOT code prefix from grid chain"""
    digits = [chain.get(lv, "0") for lv in range(1, up_to_level + 1)]
    return "G" + "".join(digits)


def is_valid_g0_code(code: str) -> bool:
    """检查是否为合法的东北半球 GeoSOT 编码 / Check valid NE-hemisphere GeoSOT code"""
    if not code or not code.startswith("G"):
        return False
    body = code[1:]
    if len(body) < 1 or len(body) > 32:
        return False
    if body[0] != "0":
        return False
    return all(c in "0123" for c in body)


def _compute_grid_step(level: int) -> Tuple[float, float]:
    """
    计算指定层级网格的精确步长（相邻网格中心距离），单位度。
    / Exact grid step at a level (center-to-center), in degrees.

    GeoSOT 31 位 DMS 编码结构 / GeoSOT 31-bit DMS layout:
      位 0-7   (8位): 度     → 层级 2-9 / bits 0–7 (8): degrees → levels 2–9
      位 8-13  (6位): 分     → 层级 10-15 / bits 8–13 (6): minutes → levels 10–15
      位 14-19 (6位): 秒     → 层级 16-21 / bits 14–19 (6): seconds → levels 16–21
      位 20-30 (11位): 秒以下 → 层级 22-32 / bits 20–30 (11): sub-arcsec → levels 22–32

    使用 code_to_bbox 的 "全填1" 方式会将分/秒字段扩展到 0-63，
    但实际 DMS 只允许 0-59，导致 bbox 方式高估网格尺寸约 6.78%。
    本方法直接根据 DMS 位权计算精确步长，避免该误差。
    / Filling remaining bits with 1 in code_to_bbox expands min/sec to 0–63,
      while DMS allows only 0–59, overestimating size by ~6.78%. This method
      uses DMS bit weights for exact step size.
    """
    if level <= 1:
        return (180.0, 90.0)
    bit_pos = level - 2
    if bit_pos < 8:
        step = float(1 << (7 - bit_pos))
    elif bit_pos < 14:
        step = float(1 << (13 - bit_pos)) / 60.0
    elif bit_pos < 20:
        step = float(1 << (19 - bit_pos)) / 3600.0
    else:
        step = float(1 << (30 - bit_pos)) / (3600.0 * 2048.0)
    return (step, step)


GRID_SIZES: Dict[int, Tuple[float, float]] = {
    lv: _compute_grid_step(lv) for lv in range(1, 33)
}


def vincenty_distance(lat1: float, lon1: float,
                      lat2: float, lon2: float) -> float:
    """Vincenty 椭球体距离（CGCS2000 / EPSG:4490），返回公里 / Vincenty ellipsoidal distance (CGCS2000 / EPSG:4490), km"""
    a = 6378.137
    f = 1 / 298.257222101
    b = a * (1 - f)

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    L = math.radians(lon2 - lon1)
    U1 = math.atan((1 - f) * math.tan(phi1))
    U2 = math.atan((1 - f) * math.tan(phi2))
    sU1, cU1 = math.sin(U1), math.cos(U1)
    sU2, cU2 = math.sin(U2), math.cos(U2)

    lam = L
    for _ in range(200):
        sl, cl = math.sin(lam), math.cos(lam)
        sin_sig = math.sqrt((cU2 * sl) ** 2 +
                            (cU1 * sU2 - sU1 * cU2 * cl) ** 2)
        if sin_sig == 0:
            return 0.0
        cos_sig = sU1 * sU2 + cU1 * cU2 * cl
        sig = math.atan2(sin_sig, cos_sig)
        sin_a = cU1 * cU2 * sl / sin_sig
        cos2a = 1 - sin_a ** 2
        c2sm = (cos_sig - 2 * sU1 * sU2 / cos2a) if cos2a else 0
        C = f / 16 * cos2a * (4 + f * (4 - 3 * cos2a))
        lam_new = L + (1 - C) * f * sin_a * (
            sig + C * sin_sig * (
                c2sm + C * cos_sig * (-1 + 2 * c2sm ** 2)))
        if abs(lam_new - lam) < 1e-12:
            break
        lam = lam_new

    u2 = cos2a * (a ** 2 - b ** 2) / (b ** 2)
    A_ = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B_ = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    dsig = B_ * sin_sig * (
        c2sm + B_ / 4 * (
            cos_sig * (-1 + 2 * c2sm ** 2) -
            B_ / 6 * c2sm *
            (-3 + 4 * sin_sig ** 2) * (-3 + 4 * c2sm ** 2)))
    return b * A_ * (sig - dsig)


# ============================================================================
# 高性能计算函数 / High-performance compute helpers
# ============================================================================

_INV60 = 1.0 / 60.0
_INV3600 = 1.0 / 3600.0
_INV2048 = 1.0 / 2048.0

_GS_LON = tuple(GRID_SIZES[lv][0] for lv in range(1, 33))
_GS_LAT = tuple(GRID_SIZES[lv][1] for lv in range(1, 33))

_OFF_EW = (-0.5, 0.5, -0.5, 0.5)
_OFF_NS = (-0.5, -0.5, 0.5, 0.5)


def _bits31_to_degree_int(val: int) -> float:
    """
    31 位整数 DMS → 十进制度（整数位运算，无字符串解析）。
    / 31-bit integer DMS → decimal degrees (integer bit ops, no string parse).

    @param val 31 位整数（8 位度 + 6 位分 + 6 位秒 + 11 位秒以下）
               / 31-bit int (8° + 6′ + 6″ + 11 sub-arcsec bits)
    @returns 十进制度 / decimal degrees
    """
    return (
        ((val >> 23) & 0xFF)
        + ((val >> 17) & 0x3F) * _INV60
        + (((val >> 11) & 0x3F) + (val & 0x7FF) * _INV2048) * _INV3600
    )


def _spherical_bearing(lon1: float, lat1: float,
                       lon2: float, lat2: float) -> float:
    """球面大圆初始方位角（0 = 正北，顺时针，度），用作椭球反算不收敛时的回退。 / Spherical great-circle initial bearing (0=N, clockwise, deg); fallback when ellipsoidal inverse fails."""
    la1 = math.radians(lat1)
    la2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(la2)
    y = (math.cos(la1) * math.sin(la2)
         - math.sin(la1) * math.cos(la2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def compute_bearing(lon1: float, lat1: float,
                    lon2: float, lat2: float) -> float:
    """
    计算方位角（0 = 正北，顺时针，度）。
    / Compute bearing (0 = north, clockwise, degrees).

    采用 Vincenty 椭球反算（CGCS2000 / EPSG:4490）得到起点处的大地测线正
    方位角，与 vincenty_distance 及 PostGIS ST_Azimuth 的椭球口径一致；对
    重合点或极少数不收敛情形回退到球面大圆方位。
    / Vincenty ellipsoidal inverse (CGCS2000 / EPSG:4490) for forward azimuth
      at the start point, consistent with vincenty_distance and PostGIS
      ST_Azimuth; fall back to spherical bearing for coincident points or rare
      non-convergence.

    @param lon1 起点经度 / start longitude
    @param lat1 起点纬度 / start latitude
    @param lon2 终点经度 / end longitude
    @param lat2 终点纬度 / end latitude
    @returns 方位角（度） / bearing (degrees)
    """
    f = 1.0 / 298.257222101
    L = math.radians(lon2 - lon1)
    U1 = math.atan((1 - f) * math.tan(math.radians(lat1)))
    U2 = math.atan((1 - f) * math.tan(math.radians(lat2)))
    sin_u1, cos_u1 = math.sin(U1), math.cos(U1)
    sin_u2, cos_u2 = math.sin(U2), math.cos(U2)

    lam = L
    sin_sigma = 0.0
    for _ in range(100):
        sin_lam, cos_lam = math.sin(lam), math.cos(lam)
        sin_sigma = math.sqrt(
            (cos_u2 * sin_lam) ** 2
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam) ** 2)
        if sin_sigma == 0.0:
            return 0.0  # 重合点，方位无定义 / coincident points; bearing undefined
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2
        cos_2sm = (cos_sigma - 2 * sin_u1 * sin_u2 / cos_sq_alpha
                   if cos_sq_alpha != 0 else 0.0)
        c = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        lam_prev = lam
        lam = L + (1 - c) * f * sin_alpha * (
            sigma + c * sin_sigma * (
                cos_2sm + c * cos_sigma * (-1 + 2 * cos_2sm ** 2)))
        if abs(lam - lam_prev) < 1e-12:
            break
    else:
        return _spherical_bearing(lon1, lat1, lon2, lat2)

    alpha1 = math.atan2(
        cos_u2 * math.sin(lam),
        cos_u1 * sin_u2 - sin_u1 * cos_u2 * math.cos(lam))
    return (math.degrees(alpha1) + 360) % 360


def fast_code_to_center(code: str) -> Tuple[float, float]:
    """
    高性能 GeoSOT 编码 → (center_lon, center_lat)。
    / Fast GeoSOT code → (center_lon, center_lat).

    使用整数移位/掩码替代字符串拼接和 int(str, 2) 解析。
    全精度编码（31 位细化）只需 2 次 DMS 转换而非 4 次。
    / Integer shift/mask instead of string concat and int(str, 2).
      Full-precision codes (31 refine bits) need 2 DMS conversions, not 4.

    调用者需确保 code 已大写且无前后空白。
    / Caller must ensure code is uppercased and stripped.

    @param code GeoSOT 编码字符串（如 "G0123..."） / GeoSOT code string
    @returns (center_lon, center_lat) 十进制度 / decimal degrees
    """
    refine = code[2:]
    n = len(refine)
    lat_v = lon_v = 0
    for ch in refine:
        d = ord(ch) - 48
        lat_v = (lat_v << 1) | (d >> 1)
        lon_v = (lon_v << 1) | (d & 1)
    if n >= 31:
        return (_bits31_to_degree_int(lon_v),
                _bits31_to_degree_int(lat_v))
    remain = 31 - n
    lat_lo = lat_v << remain
    lon_lo = lon_v << remain
    mask = (1 << remain) - 1
    return (
        (_bits31_to_degree_int(lon_lo)
         + _bits31_to_degree_int(lon_lo | mask)) * 0.5,
        (_bits31_to_degree_int(lat_lo)
         + _bits31_to_degree_int(lat_lo | mask)) * 0.5,
    )


# Vincenty 加速：优先 Numba JIT，回退到纯 Python / Vincenty accel: Numba JIT preferred, pure Python fallback
vincenty_fast = vincenty_distance
HAS_NUMBA = False

try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _vincenty_numba(lat1, lon1, lat2, lon2):
        """Numba JIT 编译的 Vincenty 椭球距离（CGCS2000），返回 km / Numba-JIT Vincenty ellipsoidal distance (CGCS2000), km"""
        a = 6378.137
        f = 1.0 / 298.257222101
        b_ax = a * (1.0 - f)
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        L = math.radians(lon2 - lon1)
        U1 = math.atan((1.0 - f) * math.tan(phi1))
        U2 = math.atan((1.0 - f) * math.tan(phi2))
        sU1 = math.sin(U1)
        cU1 = math.cos(U1)
        sU2 = math.sin(U2)
        cU2 = math.cos(U2)
        lam = L
        cos2a = 0.0
        sin_sig = 0.0
        cos_sig = 0.0
        sig = 0.0
        c2sm = 0.0
        for _ in range(200):
            sl = math.sin(lam)
            cl = math.cos(lam)
            sin_sig = math.sqrt(
                (cU2 * sl) ** 2 + (cU1 * sU2 - sU1 * cU2 * cl) ** 2)
            if sin_sig == 0.0:
                return 0.0
            cos_sig = sU1 * sU2 + cU1 * cU2 * cl
            sig = math.atan2(sin_sig, cos_sig)
            sin_a = cU1 * cU2 * sl / sin_sig
            cos2a = 1.0 - sin_a ** 2
            if cos2a != 0.0:
                c2sm = cos_sig - 2.0 * sU1 * sU2 / cos2a
            else:
                c2sm = 0.0
            C = f / 16.0 * cos2a * (4.0 + f * (4.0 - 3.0 * cos2a))
            lam_new = L + (1.0 - C) * f * sin_a * (
                sig + C * sin_sig * (
                    c2sm + C * cos_sig * (
                        -1.0 + 2.0 * c2sm ** 2)))
            if abs(lam_new - lam) < 1e-12:
                break
            lam = lam_new
        u2 = cos2a * (a ** 2 - b_ax ** 2) / (b_ax ** 2)
        A_ = 1.0 + u2 / 16384.0 * (
            4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
        B_ = u2 / 1024.0 * (
            256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
        dsig = B_ * sin_sig * (
            c2sm + B_ / 4.0 * (
                cos_sig * (-1.0 + 2.0 * c2sm ** 2)
                - B_ / 6.0 * c2sm
                * (-3.0 + 4.0 * sin_sig ** 2)
                * (-3.0 + 4.0 * c2sm ** 2)))
        return b_ax * A_ * (sig - dsig)

    _vincenty_numba(0.0, 0.0, 1.0, 1.0)
    vincenty_fast = _vincenty_numba
    HAS_NUMBA = True
except (ImportError, Exception):
    pass


def fast_graph_reasoning_pair(
    code_a: str, code_b: str,
) -> Tuple[float, float]:
    """
    融合式图推理距离计算，直接从编码字符串到 (distance_m, bearing_deg)。
    / Fused graph-reasoning distance: codes → (distance_m, bearing_deg).

    相比 graph_reasoning_pair 的优化 / vs graph_reasoning_pair:
      - 跳过 chain dict 构建，直接比较编码字符 / skip chain dict; compare code chars
      - 内联参考点坐标计算，避免 reconstruct_code → code_to_center 往返
        / inline reference coords; avoid reconstruct_code → code_to_center
      - 预计算元组索引替代 dict.get() / precomputed tuple index vs dict.get()
      - 使用 vincenty_fast（Numba 可用时自动加速） / vincenty_fast (Numba when available)

    调用者需确保 code 已大写且无前后空白。
    / Caller must ensure code is uppercased and stripped.

    @param code_a 第一个 GeoSOT 编码 / first GeoSOT code
    @param code_b 第二个 GeoSOT 编码 / second GeoSOT code
    @returns (distance_m, bearing_deg)
    """
    body_a = code_a[1:]
    body_b = code_b[1:]
    len_a = len(body_a)
    len_b = len(body_b)

    diff = -1
    for i in range(min(len_a, len_b)):
        if body_a[i] != body_b[i]:
            diff = i
            break
    if diff < 0:
        return 0.0, 0.0

    pf = body_a[1:max(diff, 1)]
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

    oa_ew = oa_ns = ob_ew = ob_ns = 0.0
    upper = min(max(len_a, len_b), 32)
    for i in range(diff, upper):
        lsz = _GS_LON[i]
        asz = _GS_LAT[i]
        if i < len_a:
            d = ord(body_a[i]) - 48
            oa_ew += _OFF_EW[d] * lsz
            oa_ns += _OFF_NS[d] * asz
        if i < len_b:
            d = ord(body_b[i]) - 48
            ob_ew += _OFF_EW[d] * lsz
            ob_ns += _OFF_NS[d] * asz

    dist_m = vincenty_fast(
        ref_lat + oa_ns, ref_lon + oa_ew,
        ref_lat + ob_ns, ref_lon + ob_ew) * 1000
    if dist_m > 0.001:
        brg = compute_bearing(
            ref_lon + oa_ew, ref_lat + oa_ns,
            ref_lon + ob_ew, ref_lat + ob_ns)
    else:
        brg = 0.0
    return dist_m, brg


# ============================================================================
# GeoSOT 空间推理器 / GeoSOT spatial reasoner
# ============================================================================

class GeoSotSpatialReasoner:
    """
    GeoSOT 空间推理器 - 基于图遍历的空间推理
    / GeoSOT spatial reasoner — graph-traversal spatial reasoning

    核心算法 / Core algorithm:
    1. 提取两个 GeoSOT 编码的网格链（L1~L32） / Extract grid chains (L1–L32)
    2. 找到首个分异层级 diff_level / Find first differing level
    3. 在 diff_level 及更深层级，通过 BFS 遍历 EAST_OF/NORTH_OF
       关系计算各级网格的位置差
       / BFS EAST_OF/NORTH_OF at diff_level+ for per-level position deltas
    4. 将位置差转换为经纬度偏移 / Convert position deltas to lon/lat offsets
    5. 解码共同父网格获取参考坐标，加上推理偏移得到估计坐标
       / Decode common parent as reference; add offsets for estimates
    6. Vincenty 公式计算椭球体距离 / Vincenty ellipsoidal distance
    """

    def __init__(self, verbose: bool = True, use_cache: bool = True,
                 enable_postgis: bool = True):
        self.verbose = verbose
        self._neighbor_cache: Dict[Tuple, str] = {}
        self._cache_loaded = False
        self._pg_conn = None

        uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        database = os.getenv("NEO4J_DB", "geocode")

        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._session = self._driver.session(database=database)
        self._session.run("RETURN 1").consume()
        if verbose:
            self._print(f"[GeoSotReasoner] 已连接 Neo4j: {uri}, 数据库: {database}")

        if enable_postgis:
            self._init_postgis()

        if use_cache:
            self._preload_neighbor_cache()

    # ------------------------------------------------------------------
    # 基础设施 / Infrastructure
    # ------------------------------------------------------------------

    @staticmethod
    def _print(msg: str):
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
            self._print(f"[ERROR] Cypher 查询失败: {e}")
            return []

    def _init_postgis(self):
        """连接 PostGIS 数据库（public.poi_new） / Connect to PostGIS (public.poi_new)"""
        try:
            import psycopg2
            self._pg_conn = psycopg2.connect(**PG_CONFIG)
            if self.verbose:
                self._print(
                    f"[GeoSotReasoner] 已连接 PostGIS: "
                    f"{PG_CONFIG['host']}:{PG_CONFIG['port']}/"
                    f"{PG_CONFIG['dbname']}")
        except ImportError:
            self._print("[警告] psycopg2 未安装，PostGIS 精度对比不可用")
        except Exception as e:
            self._print(f"[警告] PostGIS 连接失败: {e}")

    def _query_postgis_coordinates(
        self, main_code: str,
    ) -> Optional[Tuple[float, float]]:
        """通过 main_code 从 PostGIS 查询要素的实际坐标 (lon, lat) / Query real feature coords (lon, lat) from PostGIS by main_code"""
        if self._pg_conn is None:
            return None
        try:
            cur = self._pg_conn.cursor()
            cur.execute(
                "SELECT ST_X(geometry) AS lon, ST_Y(geometry) AS lat "
                "FROM public.poi_new "
                "WHERE main_code = %s LIMIT 1",
                (main_code,),
            )
            row = cur.fetchone()
            cur.close()
            if row:
                return (float(row[0]), float(row[1]))
        except Exception as e:
            self._print(f"[警告] PostGIS 查询 {main_code} 失败: {e}")
            try:
                self._pg_conn.rollback()
            except Exception:
                pass
        return None

    def _query_postgis_distance(
        self, main_code_a: str, main_code_b: str,
    ) -> Optional[float]:
        """利用 PostGIS ST_Distance (geography) 计算两个要素的椭球体距离(米) / Ellipsoidal distance (m) via PostGIS ST_Distance(geography)"""
        if self._pg_conn is None:
            return None
        try:
            cur = self._pg_conn.cursor()
            cur.execute(
                "SELECT ST_Distance("
                "  a.geometry::geography, b.geometry::geography"
                ") AS dist_m "
                "FROM public.poi_new a, public.poi_new b "
                "WHERE a.main_code = %s AND b.main_code = %s "
                "LIMIT 1",
                (main_code_a, main_code_b),
            )
            row = cur.fetchone()
            cur.close()
            if row and row[0] is not None:
                return float(row[0])
        except Exception as e:
            self._print(f"[警告] PostGIS 距离计算失败: {e}")
            try:
                self._pg_conn.rollback()
            except Exception:
                pass
        return None

    def close(self):
        if self._pg_conn:
            try:
                self._pg_conn.close()
            except Exception:
                pass
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 邻居缓存 / Neighbor cache
    # ------------------------------------------------------------------

    def _preload_neighbor_cache(self):
        """预加载所有层级的 EAST_OF / NORTH_OF 关系到内存 / Preload all-level EAST_OF / NORTH_OF relations into memory"""
        t0 = time.perf_counter()
        total = 0
        for level in range(1, 33):
            label, prop = GRID_LEVEL_MAP[level]
            for row in self._run(
                f"MATCH (a:`{label}`)-[:EAST_OF]->(b:`{label}`) "
                f"RETURN a.{prop} AS east, b.{prop} AS west"
            ):
                e, w = str(row["east"]), str(row["west"])
                self._neighbor_cache[(level, w, "east")] = e
                self._neighbor_cache[(level, e, "west")] = w
                total += 1
            for row in self._run(
                f"MATCH (a:`{label}`)-[:NORTH_OF]->(b:`{label}`) "
                f"RETURN a.{prop} AS north, b.{prop} AS south"
            ):
                n, s = str(row["north"]), str(row["south"])
                self._neighbor_cache[(level, s, "north")] = n
                self._neighbor_cache[(level, n, "south")] = s
                total += 1
        self._cache_loaded = True
        elapsed = (time.perf_counter() - t0) * 1000
        if self.verbose:
            self._print(
                f"[GeoSotReasoner] 邻居缓存加载完成: {total} 条关系, "
                f"{len(self._neighbor_cache)} 条缓存项, 耗时 {elapsed:.1f}ms"
            )

    def _get_neighbor(self, level: int, code: str,
                      direction: str) -> Optional[str]:
        if self._cache_loaded:
            return self._neighbor_cache.get((level, code, direction))
        label, prop = GRID_LEVEL_MAP[level]
        if direction == "east":
            cypher = (f"MATCH (x:`{label}`)-[:EAST_OF]->"
                      f"(g:`{label}` {{{prop}: $d}}) "
                      f"RETURN x.{prop} AS nc LIMIT 1")
        elif direction == "west":
            cypher = (f"MATCH (g:`{label}` {{{prop}: $d}})-[:EAST_OF]->"
                      f"(x:`{label}`) "
                      f"RETURN x.{prop} AS nc LIMIT 1")
        elif direction == "north":
            cypher = (f"MATCH (x:`{label}`)-[:NORTH_OF]->"
                      f"(g:`{label}` {{{prop}: $d}}) "
                      f"RETURN x.{prop} AS nc LIMIT 1")
        elif direction == "south":
            cypher = (f"MATCH (g:`{label}` {{{prop}: $d}})-[:NORTH_OF]->"
                      f"(x:`{label}`) "
                      f"RETURN x.{prop} AS nc LIMIT 1")
        else:
            return None
        rows = self._run(cypher, d=code)
        return str(rows[0]["nc"]) if rows else None

    # ------------------------------------------------------------------
    # 网格链获取 / Grid-chain retrieval
    # ------------------------------------------------------------------

    def get_grid_chain_from_neo4j(self, main_code: str) -> Dict[int, str]:
        """从 Neo4j 查询质心码 (GeoSOTCentroidCode) 的层级数字链 / Query level digit chain of a centroid code from Neo4j"""
        rows = self._run(
            "MATCH (c:GeoSOTCentroidCode {main_code: $code})-[r]->(d) "
            "WHERE type(r) STARTS WITH 'CC_HAS_LEVEL_' "
            "RETURN type(r) AS rel_type, d.digit AS digit "
            "LIMIT 100",
            code=main_code,
        )
        chain: Dict[int, str] = {}
        for row in rows:
            m = re.match(r"^CC_HAS_LEVEL_(\d+)_DIGIT$", row.get("rel_type", ""))
            if not m:
                continue
            lvl = int(m.group(1))
            val = row.get("digit")
            if val is not None:
                chain[lvl] = str(val)
        return chain

    @staticmethod
    def get_grid_chain_from_code(geosot_code: str) -> Dict[int, str]:
        """直接从 GeoSOT 编码字符串解析网格链（不查数据库） / Parse grid chain from GeoSOT code string (no DB)"""
        code = geosot_code.strip().upper()
        if not is_valid_g0_code(code):
            return {}
        body = code[1:]
        return {i + 1: d for i, d in enumerate(body)}

    # ------------------------------------------------------------------
    # BFS 位置定位 / BFS position locating
    # ------------------------------------------------------------------

    def find_position_by_bfs(self, level: int,
                             code: str) -> Tuple[int, int, int, int]:
        """
        BFS 遍历确定网格在父网格内的行列位置
        / BFS to locate row/col of a grid within its parent

        策略 / Strategy:
        - 向西 BFS 直到边界 → 跳数 = col / west BFS to boundary → hops = col
        - 向南 BFS 直到边界 → 跳数 = row / south BFS to boundary → hops = row
        - 从最西端向东 → total_cols / from westmost eastward → total_cols
        - 从西南角向北 → total_rows / from SW corner northward → total_rows

        @returns (row, col, total_rows, total_cols)
                 row=0 南, col=0 西 / row=0 south, col=0 west
        """
        # 向西 → col / west → col
        col = 0
        cur = code
        while True:
            west = self._get_neighbor(level, cur, "west")
            if west is None:
                break
            col += 1
            cur = west
        westmost = cur

        # 向南 → row / south → row
        row = 0
        cur = code
        while True:
            south = self._get_neighbor(level, cur, "south")
            if south is None:
                break
            row += 1
            cur = south

        # 从最西端向东 → total_cols / westmost eastward → total_cols
        total_cols = 1
        cur = westmost
        while True:
            east = self._get_neighbor(level, cur, "east")
            if east is None:
                break
            total_cols += 1
            cur = east

        # 从最西端向南找到西南角 / from westmost south to SW corner
        southwest = westmost
        cur = westmost
        while True:
            south = self._get_neighbor(level, cur, "south")
            if south is None:
                break
            southwest = south
            cur = south

        # 从西南角向北 → total_rows / from SW corner north → total_rows
        total_rows = 1
        cur = southwest
        while True:
            north = self._get_neighbor(level, cur, "north")
            if north is None:
                break
            total_rows += 1
            cur = north

        return row, col, total_rows, total_cols

    # ------------------------------------------------------------------
    # 图结构验证 / Graph structure verification
    # ------------------------------------------------------------------

    def verify_graph_structure(self) -> Dict:
        """
        验证 Neo4j 中的 GeoSOT 网格结构
        / Verify GeoSOT grid structure in Neo4j

        检查项 / Checks:
        1. 各层级网格节点数量和编码值 / per-level node counts and digit values
        2. EAST_OF / NORTH_OF 关系数量及正确性 / EAST_OF / NORTH_OF counts and correctness
        3. GeoSOTCentroidCode 节点和 CC_HAS_LEVEL_k_DIGIT 关系
           / GeoSOTCentroidCode nodes and CC_HAS_LEVEL_k_DIGIT relations
        4. 抽样 BFS 验证 / sample BFS checks
        """
        report: Dict = {"levels": {}, "geosot_codes": {}, "issues": []}

        print("\n" + "=" * 72)
        print("GeoSOT 知识图谱网格结构验证")
        print("=" * 72)

        # ---- 1. 各层级网格节点和关系 / per-level grid nodes and relations ----
        print(f"\n{'层级':<6} {'节点数':<8} {'EAST_OF':<10} "
              f"{'NORTH_OF':<10} {'编码值':<16} {'状态'}")
        print("-" * 66)

        levels_with_nodes = []
        for level in range(1, 33):
            label, prop = GRID_LEVEL_MAP[level]
            rows = self._run(
                f"MATCH (n:`{label}`) RETURN n.{prop} AS code")
            if not rows:
                continue
            codes = sorted(str(r["code"]) for r in rows)
            node_count = len(codes)
            levels_with_nodes.append(level)

            rows_e = self._run(
                f"MATCH (:`{label}`)-[r:EAST_OF]->(:`{label}`) "
                f"RETURN count(r) AS cnt")
            east_count = rows_e[0]["cnt"] if rows_e else 0

            rows_n = self._run(
                f"MATCH (:`{label}`)-[r:NORTH_OF]->(:`{label}`) "
                f"RETURN count(r) AS cnt")
            north_count = rows_n[0]["cnt"] if rows_n else 0

            issues: List[str] = []
            is_template = set(codes) == {"0", "1", "2", "3"}

            if is_template:
                if east_count != 2:
                    issues.append(f"EAST_OF 期望 2, 实际 {east_count}")
                if north_count != 2:
                    issues.append(f"NORTH_OF 期望 2, 实际 {north_count}")

                # 验证具体关系对 / verify concrete relation pairs
                east_pairs = set()
                for r in self._run(
                    f"MATCH (a:`{label}`)-[:EAST_OF]->(b:`{label}`) "
                    f"RETURN a.{prop} AS e, b.{prop} AS w"
                ):
                    east_pairs.add((str(r["e"]), str(r["w"])))
                north_pairs = set()
                for r in self._run(
                    f"MATCH (a:`{label}`)-[:NORTH_OF]->(b:`{label}`) "
                    f"RETURN a.{prop} AS n, b.{prop} AS s"
                ):
                    north_pairs.add((str(r["n"]), str(r["s"])))

                expected_east = {("1", "0"), ("3", "2")}
                expected_north = {("2", "0"), ("3", "1")}

                if east_pairs != expected_east:
                    issues.append(
                        f"EAST_OF 关系不匹配: 期望{expected_east}, "
                        f"实际{east_pairs}")
                if north_pairs != expected_north:
                    issues.append(
                        f"NORTH_OF 关系不匹配: 期望{expected_north}, "
                        f"实际{north_pairs}")

            status = "OK" if not issues else "ERROR"
            codes_str = ",".join(codes[:6])
            if len(codes) > 6:
                codes_str += "..."
            print(f"L{level:<5} {node_count:<8} {east_count:<10} "
                  f"{north_count:<10} {codes_str:<16} {status}")
            for iss in issues:
                print(f"       -> {iss}")
                report["issues"].append(f"L{level}: {iss}")

            report["levels"][level] = {
                "nodes": node_count, "codes": codes,
                "east_of": east_count, "north_of": north_count,
                "is_template": is_template, "issues": issues,
            }

        # ---- 2. GeoSOTCentroidCode 节点 / GeoSOTCentroidCode nodes ----
        print(f"\n{'='*40}")
        print("GeoSOTCentroidCode 节点统计")
        print(f"{'='*40}")
        rows = self._run("MATCH (n:GeoSOTCentroidCode) RETURN count(n) AS cnt")
        code_count = rows[0]["cnt"] if rows else 0
        print(f"  总数: {code_count}")
        report["geosot_codes"]["total"] = code_count

        if code_count > 0:
            rows = self._run(
                "MATCH (c:GeoSOTCentroidCode)-[r]->() "
                "WHERE type(r) STARTS WITH 'CC_HAS_LEVEL_' "
                "RETURN type(r) AS rt, count(r) AS cnt ORDER BY rt")
            if rows:
                print("  CC_HAS_LEVEL 关系:")
                for r in rows:
                    print(f"    {r['rt']}: {r['cnt']}")

            samples = self._run(
                "MATCH (n:GeoSOTCentroidCode) RETURN n.main_code AS c LIMIT 5")
            sample_codes = [r["c"] for r in samples]
            print(f"  样例: {sample_codes}")
            report["geosot_codes"]["samples"] = sample_codes

        # ---- 3. BFS 抽样验证 / BFS sample verification ----
        if levels_with_nodes:
            print(f"\n{'='*40}")
            print("BFS 抽样验证")
            print(f"{'='*40}")
            test_level = levels_with_nodes[0]
            for digit in ["0", "1", "2", "3"]:
                r, c, tr, tc = self.find_position_by_bfs(test_level, digit)
                expected = DIGIT_TO_ROWCOL.get(digit, (-1, -1))
                match = (r == expected[0] and c == expected[1]
                         and tr == 2 and tc == 2)
                status = "OK" if match else "MISMATCH"
                print(f"  L{test_level} 码元'{digit}': "
                      f"BFS=(r{r},c{c},{tr}x{tc}) "
                      f"期望=(r{expected[0]},c{expected[1]},2x2) {status}")
                if not match:
                    report["issues"].append(
                        f"L{test_level} BFS 码元'{digit}' 位置不匹配")

        # ---- 总结 / summary ----
        print(f"\n{'='*72}")
        if report["issues"]:
            print(f"发现 {len(report['issues'])} 个问题:")
            for iss in report["issues"]:
                print(f"  X {iss}")
        else:
            print("所有验证通过，网格结构正确")
        print(f"有效层级: L{min(levels_with_nodes)}-L{max(levels_with_nodes)}"
              if levels_with_nodes else "无网格节点")
        print("=" * 72)
        return report

    # ------------------------------------------------------------------
    # 主推理方法 / Main reasoning methods
    # ------------------------------------------------------------------

    def find_first_different_level(
        self, chain_a: Dict[int, str], chain_b: Dict[int, str],
    ) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """找到首个分异层级 / Find first differing level"""
        all_levels = sorted(set(chain_a.keys()) | set(chain_b.keys()))
        for lvl in all_levels:
            da, db = chain_a.get(lvl), chain_b.get(lvl)
            if da is None or db is None:
                continue
            if da != db:
                return lvl, da, db
        return None, None, None

    def calculate_distance_and_direction(
        self,
        code_a: str,
        code_b: str,
        compare_decode: bool = True,
        compare_postgis: bool = True,
    ) -> Dict:
        """
        计算两个 GeoSOT 编码之间的距离和相对方向
        / Compute distance and relative direction between two GeoSOT codes

        算法流程 / Pipeline:
        1. 获取网格链（优先 Neo4j，回退到编码解析）
           / Get grid chains (Neo4j first, code parse fallback)
        2. 逐级比较找首个分异层级 / Find first differing level
        3. BFS 遍历计算各级偏移 / BFS for per-level offsets
        4. 累加偏移，解码参考坐标，估计两点坐标
           / Accumulate offsets; decode reference; estimate both points
        5. Vincenty 距离 + 方向 / Vincenty distance + direction
        6. 可选：与直接解码结果对比 / Optional: compare with direct decode
        """
        code_a = code_a.strip().upper()
        code_b = code_b.strip().upper()

        result: Dict = {
            "success": False,
            "message": "",
            "code_a": code_a,
            "code_b": code_b,
            "different_level": None,
            "ew_offset_deg": 0.0,
            "ns_offset_deg": 0.0,
            "distance_km": 0.0,
            "direction": "",
            "level_details": [],
            "estimated_a": None,
            "estimated_b": None,
            "common_parent_level": None,
            "reference_point": None,
            "decode_comparison": None,
            "postgis_comparison": None,
            "timing": {},
        }

        t_total = time.perf_counter()

        # ---- 1. 获取网格链 / get grid chains ----
        t0 = time.perf_counter()
        chain_a_data = self.get_grid_chain_from_neo4j(code_a)
        chain_b_data = self.get_grid_chain_from_neo4j(code_b)

        source_a = "Neo4j"
        source_b = "Neo4j"
        if not chain_a_data:
            chain_a_data = self.get_grid_chain_from_code(code_a)
            source_a = "编码解析"
        if not chain_b_data:
            chain_b_data = self.get_grid_chain_from_code(code_b)
            source_b = "编码解析"
        result["timing"]["get_chain_ms"] = (
            time.perf_counter() - t0) * 1000

        if not chain_a_data:
            result["message"] = f"无法获取 {code_a} 的网格链"
            return result
        if not chain_b_data:
            result["message"] = f"无法获取 {code_b} 的网格链"
            return result

        if self.verbose:
            max_show = 16
            ca_show = {k: v for k, v in sorted(chain_a_data.items())
                       if k <= max_show}
            cb_show = {k: v for k, v in sorted(chain_b_data.items())
                       if k <= max_show}
            suffix_a = "..." if len(chain_a_data) > max_show else ""
            suffix_b = "..." if len(chain_b_data) > max_show else ""
            self._print(
                f"\n[推理] A 网格链({source_a}, {len(chain_a_data)}层): "
                f"{ca_show}{suffix_a}")
            self._print(
                f"[推理] B 网格链({source_b}, {len(chain_b_data)}层): "
                f"{cb_show}{suffix_b}")

        # ---- 2. 找首个分异层级 / find first differing level ----
        t0 = time.perf_counter()
        diff_level, da, db = self.find_first_different_level(
            chain_a_data, chain_b_data)
        result["timing"]["find_diff_ms"] = (
            time.perf_counter() - t0) * 1000

        if diff_level is None:
            result["success"] = True
            result["message"] = "两个编码在所有层级上完全相同"
            result["direction"] = "同一位置"
            result["timing"]["total_ms"] = (
                time.perf_counter() - t_total) * 1000
            return result

        result["different_level"] = diff_level
        if self.verbose:
            self._print(
                f"[推理] 首个分异层级: L{diff_level}, "
                f"A='{da}', B='{db}'")

        # ---- 3. 逐级计算偏移量 / per-level offset accumulation ----
        t0 = time.perf_counter()
        total_ew = 0.0
        total_ns = 0.0
        level_details: List[Dict] = []
        all_levels = sorted(
            set(chain_a_data.keys()) | set(chain_b_data.keys()))
        max_level = max(all_levels) if all_levels else 1

        for level in all_levels:
            digit_a = chain_a_data.get(level)
            digit_b = chain_b_data.get(level)
            lon_size, lat_size = GRID_SIZES.get(level, (0.0, 0.0))

            info: Dict = {
                "level": level,
                "digit_a": digit_a,
                "digit_b": digit_b,
                "grid_size_lon": lon_size,
                "grid_size_lat": lat_size,
                "ew_offset_deg": 0.0,
                "ns_offset_deg": 0.0,
                "method": "",
            }

            if level < diff_level:
                info["method"] = "相同"
                if self.verbose:
                    self._print(
                        f"  L{level:2d}: A='{digit_a}', B='{digit_b}'"
                        f" -> 相同")

            elif digit_a is None or digit_b is None:
                info["method"] = "数据缺失"
                if self.verbose:
                    self._print(
                        f"  L{level:2d}: A='{digit_a}', B='{digit_b}'"
                        f" -> 数据缺失")

            elif digit_a == digit_b:
                info["method"] = "相同"
                if self.verbose:
                    self._print(
                        f"  L{level:2d}: A='{digit_a}', B='{digit_b}'"
                        f" -> 相同")

            else:
                # BFS 定位 / BFS locate
                r_a, c_a, tr_a, tc_a = self.find_position_by_bfs(
                    level, digit_a)
                r_b, c_b, tr_b, tc_b = self.find_position_by_bfs(
                    level, digit_b)

                bfs_ok = (tr_a > 1 or tc_a > 1) and (
                    tr_b > 1 or tc_b > 1)

                if bfs_ok:
                    method = "BFS"
                else:
                    rc_a = DIGIT_TO_ROWCOL.get(digit_a, (0, 0))
                    rc_b = DIGIT_TO_ROWCOL.get(digit_b, (0, 0))
                    r_a, c_a = rc_a
                    r_b, c_b = rc_b
                    method = "编码规则(BFS不可用)"

                ew_grids = c_b - c_a
                ns_grids = r_b - r_a
                ew_deg = ew_grids * lon_size
                ns_deg = ns_grids * lat_size

                total_ew += ew_deg
                total_ns += ns_deg

                info["method"] = method
                info["ew_grids"] = ew_grids
                info["ns_grids"] = ns_grids
                info["ew_offset_deg"] = ew_deg
                info["ns_offset_deg"] = ns_deg

                if self.verbose:
                    self._print(
                        f"  L{level:2d}: A='{digit_a}'(r{r_a},c{c_a}), "
                        f"B='{digit_b}'(r{r_b},c{c_b}) -> {method}")
                    self._print(
                        f"        网格偏移: EW={ew_grids:+d}格, "
                        f"NS={ns_grids:+d}格")
                    self._print(
                        f"        网格尺寸: {lon_size:.10f} x "
                        f"{lat_size:.10f} 度")
                    self._print(
                        f"        度数偏移: EW={ew_deg:+.10f}, "
                        f"NS={ns_deg:+.10f} 度")

            level_details.append(info)

        result["timing"]["level_calc_ms"] = (
            time.perf_counter() - t0) * 1000
        result["level_details"] = level_details
        result["ew_offset_deg"] = total_ew
        result["ns_offset_deg"] = total_ns

        if self.verbose:
            self._print(
                f"\n[推理] 总偏移: EW={total_ew:+.10f} 度, "
                f"NS={total_ns:+.10f} 度")

        # ---- 4. 计算距离 / compute distance ----
        t0 = time.perf_counter()
        common_parent_level = max(diff_level - 1, 1)
        parent_code = reconstruct_code(chain_a_data, common_parent_level)
        ref_lon, ref_lat = code_to_center(parent_code)

        result["common_parent_level"] = common_parent_level
        result["reference_point"] = {"lon": ref_lon, "lat": ref_lat}

        # 各点从共同父网格中心的累计偏移 / cumulative offsets from common parent center
        off_a_ew, off_a_ns = 0.0, 0.0
        off_b_ew, off_b_ns = 0.0, 0.0
        for level in range(diff_level, max_level + 1):
            lon_sz, lat_sz = GRID_SIZES.get(level, (0.0, 0.0))
            d_a = chain_a_data.get(level)
            if d_a is not None:
                r, c = DIGIT_TO_ROWCOL.get(d_a, (0, 0))
                off_a_ew += (c - 0.5) * lon_sz
                off_a_ns += (r - 0.5) * lat_sz
            d_b = chain_b_data.get(level)
            if d_b is not None:
                r, c = DIGIT_TO_ROWCOL.get(d_b, (0, 0))
                off_b_ew += (c - 0.5) * lon_sz
                off_b_ns += (r - 0.5) * lat_sz

        est_a_lon = ref_lon + off_a_ew
        est_a_lat = ref_lat + off_a_ns
        est_b_lon = ref_lon + off_b_ew
        est_b_lat = ref_lat + off_b_ns

        result["estimated_a"] = {"lon": est_a_lon, "lat": est_a_lat}
        result["estimated_b"] = {"lon": est_b_lon, "lat": est_b_lat}

        dist_km = vincenty_distance(est_a_lat, est_a_lon,
                                    est_b_lat, est_b_lon)
        result["distance_km"] = dist_km
        result["timing"]["distance_ms"] = (
            time.perf_counter() - t0) * 1000

        if self.verbose:
            self._print(
                f"[推理] 共同父网格: L{common_parent_level} "
                f"({parent_code})")
            self._print(
                f"[推理] 参考点: ({ref_lon:.8f}, {ref_lat:.8f})")
            self._print(
                f"[推理] A 估计坐标: ({est_a_lon:.8f}, {est_a_lat:.8f})")
            self._print(
                f"[推理] B 估计坐标: ({est_b_lon:.8f}, {est_b_lat:.8f})")
            self._print(
                f"[推理] Vincenty 距离: {dist_km:.6f} km "
                f"({dist_km * 1000:.2f} m)")

        # ---- 5. 方向 / direction ----
        t0 = time.perf_counter()
        result["direction"] = self._compute_direction(total_ew, total_ns)
        result["timing"]["direction_ms"] = (
            time.perf_counter() - t0) * 1000

        result["success"] = True
        result["message"] = "计算成功"
        result["timing"]["total_ms"] = (
            time.perf_counter() - t_total) * 1000

        # ---- 6. 对比直接解码 / compare with direct decode ----
        if compare_decode:
            self._add_decode_comparison(result, code_a, code_b)

        # ---- 7. 对比 PostGIS 实际坐标（精度基准） / compare PostGIS real coords (baseline) ----
        if compare_postgis and self._pg_conn is not None:
            self._add_postgis_comparison(result, code_a, code_b)

        return result

    @staticmethod
    def _compute_direction(ew_deg: float, ns_deg: float) -> str:
        if abs(ew_deg) < 1e-12 and abs(ns_deg) < 1e-12:
            return "同一位置"
        angle = math.degrees(math.atan2(ew_deg, ns_deg))
        if angle < 0:
            angle += 360
        if abs(angle) < 1 or abs(angle - 360) < 1:
            return "正北"
        if abs(angle - 90) < 1:
            return "正东"
        if abs(angle - 180) < 1:
            return "正南"
        if abs(angle - 270) < 1:
            return "正西"
        if 0 < angle < 90:
            return f"北偏东 {angle:.1f}"
        if 90 < angle < 180:
            return f"南偏东 {180 - angle:.1f}"
        if 180 < angle < 270:
            return f"南偏西 {angle - 180:.1f}"
        return f"北偏西 {360 - angle:.1f}"

    def _add_decode_comparison(self, result: Dict,
                               code_a: str, code_b: str):
        """对比直接解码坐标的结果 / Compare against direct-decoded coordinates"""
        try:
            center_a = code_to_center(code_a)
            center_b = code_to_center(code_b)
            decode_dist = vincenty_distance(
                center_a[1], center_a[0], center_b[1], center_b[0])

            delta_lon = center_b[0] - center_a[0]
            delta_lat = center_b[1] - center_a[1]
            dist_err_m = (result["distance_km"] - decode_dist) * 1000
            dist_err_pct = (
                (dist_err_m / (decode_dist * 1000) * 100)
                if decode_dist > 0 else 0)
            ew_err = result["ew_offset_deg"] - delta_lon
            ns_err = result["ns_offset_deg"] - delta_lat

            pos_err_m = math.sqrt(
                (ew_err * 111000
                 * math.cos(math.radians(center_a[1]))) ** 2
                + (ns_err * 111000) ** 2
            )

            comparison = {
                "decode_a": {"lon": center_a[0], "lat": center_a[1]},
                "decode_b": {"lon": center_b[0], "lat": center_b[1]},
                "decode_distance_km": decode_dist,
                "distance_error_m": dist_err_m,
                "distance_error_pct": dist_err_pct,
                "ew_error_deg": ew_err,
                "ns_error_deg": ns_err,
                "ew_error_arcsec": ew_err * 3600,
                "ns_error_arcsec": ns_err * 3600,
                "position_error_m": pos_err_m,
            }
            result["decode_comparison"] = comparison

            if self.verbose:
                self._print(f"\n{'=' * 60}")
                self._print("直接解码对比验证")
                self._print("=" * 60)
                self._print(
                    f"  解码 A: ({center_a[0]:.8f}, {center_a[1]:.8f})")
                self._print(
                    f"  解码 B: ({center_b[0]:.8f}, {center_b[1]:.8f})")
                self._print(
                    f"  解码距离: {decode_dist:.6f} km "
                    f"({decode_dist * 1000:.2f} m)")
                self._print(
                    f"\n  {'项目':<16} {'图推理':<20} "
                    f"{'直接解码':<20} {'误差'}")
                self._print("  " + "-" * 72)
                self._print(
                    f"  {'距离':<16} "
                    f"{result['distance_km'] * 1000:.2f} m{'':<10} "
                    f"{decode_dist * 1000:.2f} m{'':<10} "
                    f"{dist_err_m:.4f} m ({dist_err_pct:.6f}%)")
                self._print(
                    f"  {'EW偏移':<16} "
                    f"{result['ew_offset_deg']:.8f}{'':<6} "
                    f"{delta_lon:.8f}{'':<6} "
                    f"{ew_err * 3600:.4f}\"")
                self._print(
                    f"  {'NS偏移':<16} "
                    f"{result['ns_offset_deg']:.8f}{'':<6} "
                    f"{delta_lat:.8f}{'':<6} "
                    f"{ns_err * 3600:.4f}\"")
                self._print(
                    f"\n  位置综合误差: {pos_err_m:.4f} m")
                self._print("=" * 60)

        except Exception as e:
            if self.verbose:
                self._print(f"[警告] 解码对比失败: {e}")

    def _add_postgis_comparison(self, result: Dict,
                                code_a: str, code_b: str):
        """对比 PostGIS 中实际要素坐标（精度基准） / Compare against PostGIS real feature coords (baseline)"""
        try:
            t0 = time.perf_counter()
            coord_a = self._query_postgis_coordinates(code_a)
            coord_b = self._query_postgis_coordinates(code_b)

            if coord_a is None or coord_b is None:
                missing = []
                if coord_a is None:
                    missing.append(code_a)
                if coord_b is None:
                    missing.append(code_b)
                if self.verbose:
                    self._print(
                        f"[PostGIS] 未找到要素: {', '.join(missing)}")
                return

            pg_dist_m = self._query_postgis_distance(code_a, code_b)
            elapsed = (time.perf_counter() - t0) * 1000
            result["timing"]["postgis_ms"] = elapsed

            real_lon_a, real_lat_a = coord_a
            real_lon_b, real_lat_b = coord_b
            real_delta_lon = real_lon_b - real_lon_a
            real_delta_lat = real_lat_b - real_lat_a

            vincenty_real = vincenty_distance(
                real_lat_a, real_lon_a, real_lat_b, real_lon_b)

            graph_dist_m = result["distance_km"] * 1000
            graph_ew = result["ew_offset_deg"]
            graph_ns = result["ns_offset_deg"]

            # 推理距离 vs PostGIS 距离 / reasoned distance vs PostGIS
            if pg_dist_m is not None:
                postgis_dist_m = pg_dist_m
            else:
                postgis_dist_m = vincenty_real * 1000

            dist_err_m = graph_dist_m - postgis_dist_m
            dist_err_pct = (
                (dist_err_m / postgis_dist_m * 100)
                if postgis_dist_m > 0 else 0)

            ew_err = graph_ew - real_delta_lon
            ns_err = graph_ns - real_delta_lat

            cos_lat = math.cos(math.radians(real_lat_a))
            pos_err_m = math.sqrt(
                (ew_err * 111000 * cos_lat) ** 2
                + (ns_err * 111000) ** 2)

            # 估计坐标 vs 实际坐标 / estimated coords vs real
            est_a = result.get("estimated_a", {})
            est_b = result.get("estimated_b", {})
            coord_err_a_m = None
            coord_err_b_m = None
            if est_a and est_b:
                d_a_lon = est_a["lon"] - real_lon_a
                d_a_lat = est_a["lat"] - real_lat_a
                coord_err_a_m = math.sqrt(
                    (d_a_lon * 111000 * cos_lat) ** 2
                    + (d_a_lat * 111000) ** 2)
                d_b_lon = est_b["lon"] - real_lon_b
                d_b_lat = est_b["lat"] - real_lat_b
                cos_lat_b = math.cos(math.radians(real_lat_b))
                coord_err_b_m = math.sqrt(
                    (d_b_lon * 111000 * cos_lat_b) ** 2
                    + (d_b_lat * 111000) ** 2)

            # 解码坐标 vs 实际坐标 / decoded coords vs real
            decode_err_a_m = None
            decode_err_b_m = None
            decode_comp = result.get("decode_comparison")
            if decode_comp:
                dec_a = decode_comp["decode_a"]
                dec_b = decode_comp["decode_b"]
                d_a_lon = dec_a["lon"] - real_lon_a
                d_a_lat = dec_a["lat"] - real_lat_a
                decode_err_a_m = math.sqrt(
                    (d_a_lon * 111000 * cos_lat) ** 2
                    + (d_a_lat * 111000) ** 2)
                d_b_lon = dec_b["lon"] - real_lon_b
                d_b_lat = dec_b["lat"] - real_lat_b
                cos_lat_b = math.cos(math.radians(real_lat_b))
                decode_err_b_m = math.sqrt(
                    (d_b_lon * 111000 * cos_lat_b) ** 2
                    + (d_b_lat * 111000) ** 2)

            comparison = {
                "real_a": {"lon": real_lon_a, "lat": real_lat_a},
                "real_b": {"lon": real_lon_b, "lat": real_lat_b},
                "real_delta_lon": real_delta_lon,
                "real_delta_lat": real_delta_lat,
                "postgis_distance_m": postgis_dist_m,
                "vincenty_real_km": vincenty_real,
                "graph_distance_error_m": dist_err_m,
                "graph_distance_error_pct": dist_err_pct,
                "graph_ew_error_deg": ew_err,
                "graph_ns_error_deg": ns_err,
                "graph_ew_error_arcsec": ew_err * 3600,
                "graph_ns_error_arcsec": ns_err * 3600,
                "graph_offset_error_m": pos_err_m,
                "coord_err_a_m": coord_err_a_m,
                "coord_err_b_m": coord_err_b_m,
                "decode_err_a_m": decode_err_a_m,
                "decode_err_b_m": decode_err_b_m,
            }
            result["postgis_comparison"] = comparison

            if self.verbose:
                self._print(f"\n{'=' * 60}")
                self._print("PostGIS 实际坐标精度对比（精度基准）")
                self._print("=" * 60)
                self._print(
                    f"  实际 A: ({real_lon_a:.8f}, {real_lat_a:.8f})")
                self._print(
                    f"  实际 B: ({real_lon_b:.8f}, {real_lat_b:.8f})")
                self._print(
                    f"  PostGIS 距离: {postgis_dist_m:.2f} m "
                    f"({postgis_dist_m / 1000:.6f} km)")

                self._print(
                    f"\n  {'项目':<18} {'图推理':<20} "
                    f"{'PostGIS实际':<20} {'误差'}")
                self._print("  " + "-" * 74)
                self._print(
                    f"  {'距离':<18} "
                    f"{graph_dist_m:.2f} m{'':<10} "
                    f"{postgis_dist_m:.2f} m{'':<10} "
                    f"{dist_err_m:.4f} m ({dist_err_pct:.4f}%)")
                self._print(
                    f"  {'EW偏移(°)':<18} "
                    f"{graph_ew:.8f}{'':<6} "
                    f"{real_delta_lon:.8f}{'':<6} "
                    f"{ew_err * 3600:.4f}\"")
                self._print(
                    f"  {'NS偏移(°)':<18} "
                    f"{graph_ns:.8f}{'':<6} "
                    f"{real_delta_lat:.8f}{'':<6} "
                    f"{ns_err * 3600:.4f}\"")
                self._print(
                    f"\n  偏移综合误差: {pos_err_m:.4f} m")

                if coord_err_a_m is not None:
                    self._print(
                        f"  图推理坐标误差: "
                        f"A={coord_err_a_m:.4f} m, "
                        f"B={coord_err_b_m:.4f} m")
                if decode_err_a_m is not None:
                    self._print(
                        f"  网格解码坐标误差: "
                        f"A={decode_err_a_m:.4f} m, "
                        f"B={decode_err_b_m:.4f} m")

                self._print(f"  PostGIS 查询耗时: {elapsed:.1f} ms")
                self._print("=" * 60)

        except Exception as e:
            if self.verbose:
                self._print(f"[警告] PostGIS 对比失败: {e}")

    # ------------------------------------------------------------------
    # 格式化输出 / Formatted output
    # ------------------------------------------------------------------

    def format_result(self, result: Dict) -> str:
        """格式化推理结果 / Format reasoning result"""
        lines = [
            "=" * 70,
            "GeoSOT 空间推理结果",
            "=" * 70,
        ]

        if not result["success"]:
            lines.append(f"状态: 失败 - {result['message']}")
            lines.append("=" * 70)
            return "\n".join(lines)

        lines.append(f"编码 A: {result['code_a']}")
        lines.append(f"编码 B: {result['code_b']}")
        lines.append(f"首个分异层级: L{result['different_level']}")
        lines.append("")

        # 逐级偏移 / per-level offsets
        details = result.get("level_details", [])
        differing = [d for d in details if d["method"] not in
                     ("相同", "数据缺失", "")]
        if differing:
            lines.append("逐级偏移:")
            lines.append(f"  {'层级':<6} {'A':<4} {'B':<4} "
                         f"{'EW偏移':<14} {'NS偏移':<14} {'方法'}")
            lines.append("  " + "-" * 56)
            for d in differing:
                ew_g = d.get("ew_grids", 0)
                ns_g = d.get("ns_grids", 0)
                lines.append(
                    f"  L{d['level']:<5} {d['digit_a']:<4} {d['digit_b']:<4} "
                    f"{ew_g:+d}格={d['ew_offset_deg']:+.8f}"
                    f"  {ns_g:+d}格={d['ns_offset_deg']:+.8f}"
                    f"  {d['method']}")
            lines.append("")

        # 汇总 / summary
        lines.append("汇总:")
        lines.append(
            f"  总偏移: EW={result['ew_offset_deg']:+.10f} 度, "
            f"NS={result['ns_offset_deg']:+.10f} 度")
        lines.append(
            f"  距离: {result['distance_km']:.6f} km "
            f"({result['distance_km'] * 1000:.2f} m)")
        lines.append(
            f"  方向: B 在 A 的{result['direction']}")

        if result.get("estimated_a"):
            ea = result["estimated_a"]
            eb = result["estimated_b"]
            lines.append(
                f"  A 估计坐标: ({ea['lon']:.8f}, {ea['lat']:.8f})")
            lines.append(
                f"  B 估计坐标: ({eb['lon']:.8f}, {eb['lat']:.8f})")

        comp = result.get("decode_comparison")
        if comp:
            lines.append("")
            lines.append("与网格解码对比:")
            lines.append(
                f"  距离误差: {comp['distance_error_m']:.4f} m "
                f"({comp['distance_error_pct']:.6f}%)")
            lines.append(
                f"  位置误差: {comp['position_error_m']:.4f} m")

        pg_comp = result.get("postgis_comparison")
        if pg_comp:
            lines.append("")
            lines.append("与 PostGIS 实际坐标对比（精度基准）:")
            lines.append(
                f"  实际坐标 A: ({pg_comp['real_a']['lon']:.8f}, "
                f"{pg_comp['real_a']['lat']:.8f})")
            lines.append(
                f"  实际坐标 B: ({pg_comp['real_b']['lon']:.8f}, "
                f"{pg_comp['real_b']['lat']:.8f})")
            lines.append(
                f"  PostGIS 距离: {pg_comp['postgis_distance_m']:.2f} m")
            lines.append(
                f"  距离误差（图推理 vs PostGIS）: "
                f"{pg_comp['graph_distance_error_m']:.4f} m "
                f"({pg_comp['graph_distance_error_pct']:.4f}%)")
            lines.append(
                f"  偏移综合误差: {pg_comp['graph_offset_error_m']:.4f} m")
            if pg_comp.get("coord_err_a_m") is not None:
                lines.append(
                    f"  图推理坐标误差: A={pg_comp['coord_err_a_m']:.4f} m, "
                    f"B={pg_comp['coord_err_b_m']:.4f} m")
            if pg_comp.get("decode_err_a_m") is not None:
                lines.append(
                    f"  网格解码坐标误差: "
                    f"A={pg_comp['decode_err_a_m']:.4f} m, "
                    f"B={pg_comp['decode_err_b_m']:.4f} m")

        timing = result.get("timing", {})
        if timing.get("total_ms"):
            lines.append(f"\n  总耗时: {timing['total_ms']:.1f} ms")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# 命令行入口 / CLI entry
# ============================================================================

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--verify":
        reasoner = GeoSotSpatialReasoner(verbose=True)
        reasoner.verify_graph_structure()
        reasoner.close()
        return

    no_compare = "--no-compare" in sys.argv
    no_postgis = "--no-postgis" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if len(args) >= 2:
        code_a, code_b = args[0], args[1]
        if not is_valid_g0_code(code_a.strip().upper()):
            print(f"[ERROR] 非法 GeoSOT 编码: {code_a}")
            return
        if not is_valid_g0_code(code_b.strip().upper()):
            print(f"[ERROR] 非法 GeoSOT 编码: {code_b}")
            return

        reasoner = GeoSotSpatialReasoner(
            verbose=True, enable_postgis=not no_postgis)
        result = reasoner.calculate_distance_and_direction(
            code_a, code_b,
            compare_decode=not no_compare,
            compare_postgis=not no_postgis)
        print("\n" + reasoner.format_result(result))
        reasoner.close()
        return

    # 交互模式 / interactive mode
    print("\nGeoSOT 空间推理 - 交互模式")
    print("输入两个 GeoSOT 编码（空格分隔），计算距离和方向")
    print("输入 'verify' 验证图结构, 'q' 退出")
    print("选项: --no-compare 不对比解码, --no-postgis 不对比PostGIS\n")

    reasoner = GeoSotSpatialReasoner(
        verbose=True, enable_postgis=not no_postgis)

    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break
            if line.lower() == "verify":
                reasoner.verify_graph_structure()
                continue

            parts = line.split()
            if len(parts) < 2:
                print("请输入两个 GeoSOT 编码（空格分隔）")
                continue

            ca, cb = parts[0], parts[1]
            if not is_valid_g0_code(ca.strip().upper()):
                print(f"非法编码: {ca}")
                continue
            if not is_valid_g0_code(cb.strip().upper()):
                print(f"非法编码: {cb}")
                continue

            do_compare = "--no-compare" not in parts
            do_postgis = "--no-postgis" not in parts and not no_postgis
            result = reasoner.calculate_distance_and_direction(
                ca, cb,
                compare_decode=do_compare,
                compare_postgis=do_postgis)
            print("\n" + reasoner.format_result(result))

        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            print(f"错误: {e}")
            traceback.print_exc()

    reasoner.close()
    print("再见！")


if __name__ == "__main__":
    main()

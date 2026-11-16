#!/usr/bin/env python3
# coding: utf-8
"""
Forward overlay V7 using multiresolution polygon filling in discrete grid space.

Map vertex codes to centered integer (x,y), choose a target level and covering rectangle,
rasterize/classify cells with BFS subdivision, and query candidate codes by prefixes. Inside
prefixes yield direct hits; boundary prefixes require integer winding-number PIP. Thus the
current implementation retains boundary verification rather than eliminating PIP entirely.
No geographic-coordinate decoding is needed in the grid path.

V5 used bbox prefixes and floating-point PIP; V6 used a grid-mask PIP; V7 separates inside
and boundary prefixes. Winding predicates use int64 cross-product signs without floating-
point division, matching the geometry module's cross-product primitive. Short codes map to
cell centers to avoid zero-fill corner bias.

Usage:
    python geosot_overlay_query_v7.py --code G0xxx --source POI
    python geosot_overlay_query_v7.py --batch 50 --source POI --seed 42
    python geosot_overlay_query_v7.py --all --source POI --output result.csv
    python geosot_overlay_query_v7.py --all --source POI --max-cells 256
    python geosot_overlay_query_v7.py --all --source POI --refine 4
"""

import csv
import os
import random
import statistics
import sys
import time
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psycopg2
from PIL import Image, ImageDraw, ImageFilter
from neo4j import GraphDatabase

from geosot_spatial_reasoning import (
    PG_CONFIG,
    GRID_SIZES,
    fast_code_to_center as _fast_code_to_center_raw,
    is_valid_g0_code,
)
from geosot_overlay_query import (
    fetch_polygon_from_neo4j,
    postgis_overlay_features,
    SOURCE_TO_PG_TABLE,
    POINT_SOURCES,
)

@lru_cache(maxsize=500_000)
def fast_code_to_center(code: str):
    return _fast_code_to_center_raw(code)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

_t = time.perf_counter

GRID_LEVEL = 32


# ============================================================================
# Z-order (Morton code) bit operations
# ============================================================================

@lru_cache(maxsize=500_000)
def _code_to_xy(code: str, level: int) -> Tuple[int, int]:
    """
    Map a GeoSOT code to centered integer Z-order coordinates. Split each digit d into x_bit=d&1
    and y_bit=(d>>1)&1, then concatenate bits. If the code is shorter than level, shift and add
    the cell-center offset rather than mapping to the lower-left corner.

    @param code GeoSOT code, for example G00113022131.
    @param level Target level.
    @returns Centered (x, y) integer coordinates.
    """
    body = code[1:]
    code_len = len(body)
    actual = min(code_len, level)
    x, y = 0, 0
    for i in range(actual):
        d = int(body[i])
        x = (x << 1) | (d & 1)
        y = (y << 1) | ((d >> 1) & 1)
    pad = level - actual
    if pad > 0:
        x = (x << pad) | (1 << (pad - 1))
        y = (y << pad) | (1 << (pad - 1))
    return x, y


def _xy_to_code(x: int, y: int, level: int) -> str:
    """
    Encode integer (x, y) grid coordinates as a GeoSOT Z-order code.

    @param x Grid x coordinate.
    @param y Grid y coordinate.
    @param level Encoding level.
    @returns GeoSOT code.
    """
    digits = []
    for i in range(level - 1, -1, -1):
        xb = (x >> i) & 1
        yb = (y >> i) & 1
        digits.append(str(xb | (yb << 1)))
    return "G" + "".join(digits)


def _codes_to_xy_batch(codes: List[str], level: int) -> np.ndarray:
    """
    Vectorize centered GeoSOT-to-(x,y) decoding, matching _code_to_xy for short codes.

    @param codes GeoSOT codes.
    @param level Target level.
    @returns (N, 2) int64 array.
    """
    n = len(codes)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)

    code_lens = np.array([len(c) - 1 for c in codes], dtype=np.int32)
    actual_lens = np.minimum(code_lens, level)

    max_actual = int(actual_lens.max()) if n > 0 else 0
    digits = np.zeros((n, max_actual), dtype=np.int8)
    for i, code in enumerate(codes):
        body = code[1:]
        end = int(actual_lens[i])
        for j in range(end):
            digits[i, j] = ord(body[j]) - 48

    x = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    for j in range(max_actual):
        mask = j < actual_lens
        d = digits[:, j].astype(np.int64) * mask
        x = (x << 1) | (d & 1)
        y = (y << 1) | ((d >> 1) & 1)

    pads = level - actual_lens
    need_pad = pads > 0
    if need_pad.any():
        x[need_pad] = ((x[need_pad] << pads[need_pad])
                        | (np.int64(1) << (pads[need_pad] - 1)))
        y[need_pad] = ((y[need_pad] << pads[need_pad])
                        | (np.int64(1) << (pads[need_pad] - 1)))

    return np.stack([x, y], axis=1)


# ============================================================================
# Target level + cover rectangle (pure integer ops)
# ============================================================================

def _lcp_of(codes: List[str]) -> str:
    if not codes:
        return ""
    prefix = codes[0]
    for c in codes[1:]:
        while not c.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _determine_target_level(
    vertex_xy_full: List[Tuple[int, int]],
    lcp_level: int,
    max_depth: int,
    max_cells: int = 256,
) -> Tuple[int, Tuple[int, int, int, int]]:
    """
    Choose a target level and covering rectangle from full-depth vertex coordinates. Starting at
    the LCP level, refine while the vertex bounding rectangle fits within max_cells; choose the
    finest admissible level.

    @param vertex_xy_full Vertex (x,y) coordinates at max_depth.
    @param lcp_level LCP body length.
    @param max_depth Maximum depth.
    @param max_cells Maximum covering rectangle cell count.
    @returns (target_level, (x_min, x_max, y_min, y_max)).
    """
    best_level = lcp_level
    best_rect = (0, 0, 0, 0)

    for level in range(lcp_level, max_depth + 1):
        shift = max_depth - level
        xs = [xy[0] >> shift for xy in vertex_xy_full]
        ys = [xy[1] >> shift for xy in vertex_xy_full]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        area = (x_max - x_min + 1) * (y_max - y_min + 1)

        if area > max_cells:
            break

        best_level = level
        best_rect = (x_min, x_max, y_min, y_max)

    return best_level, best_rect


def _rect_to_min_prefixes(
    x_min: int, x_max: int,
    y_min: int, y_max: int,
    target_level: int,
    lcp: str,
) -> List[str]:
    """
    Generate a minimal quadtree prefix cover for a rectangle by DFS from its LCP node. Emit
    fully contained cells, recurse into four children for partial overlap, and prune disjoint
    cells; no post-enumeration merge is needed.
    """
    lcp_lv = max(len(lcp) - 1, 0)
    lcp_x, lcp_y = _code_to_xy(lcp, lcp_lv) if lcp_lv > 0 else (0, 0)

    out: List[str] = []

    def _dfs(px: int, py: int, lv: int, prefix: str):
        remaining = target_level - lv
        size = 1 << remaining
        nx1, ny1 = px, py
        nx2, ny2 = px + size - 1, py + size - 1

        if nx2 < x_min or nx1 > x_max or ny2 < y_min or ny1 > y_max:
            return

        if x_min <= nx1 and nx2 <= x_max and y_min <= ny1 and ny2 <= y_max:
            out.append(prefix)
            return

        if lv == target_level:
            out.append(prefix)
            return

        half = size >> 1
        _dfs(px,        py,        lv + 1, prefix + '0')
        _dfs(px + half, py,        lv + 1, prefix + '1')
        _dfs(px,        py + half, lv + 1, prefix + '2')
        _dfs(px + half, py + half, lv + 1, prefix + '3')

    start_size = 1 << (target_level - lcp_lv)
    start_x = lcp_x * start_size
    start_y = lcp_y * start_size
    _dfs(start_x, start_y, lcp_lv, lcp)

    return out


# ============================================================================
# Max inscribed cell level L + tiled cover (mbr_cover_pip path only)
# ============================================================================

def _compute_inscribed_grid_level(
    polygon_xy: np.ndarray,
    lcp_level: int,
    max_depth: int,
    safety_shrink: int = 0,
    verify_corners: bool = True,
) -> int:
    """
    Estimate the coarsest useful level of a cell that fits inside the polygon.

    Use the shorter MBR side d_min as the cell-size upper bound: L=max_depth-floor(log2(d_min));
    constrain L by lcp_level and refine by safety_shrink. If verify_corners is enabled,
    enumerate aligned cells and test all four corners, increasing L until an interior cell is
    found.

    @param polygon_xy Full-depth integer polygon vertices.
    @param lcp_level Lower bound on L.
    @param max_depth Maximum available level.
    @param safety_shrink Extra refinement levels.
    @param verify_corners Whether to verify an interior cell.
    @returns Selected level L.
    """
    x_min = int(polygon_xy[:, 0].min())
    x_max = int(polygon_xy[:, 0].max())
    y_min = int(polygon_xy[:, 1].min())
    y_max = int(polygon_xy[:, 1].max())
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    d_min = min(w, h)
    if d_min <= 1:
        return max_depth
    log2_d = int(np.floor(np.log2(d_min)))
    L = max_depth - log2_d + max(0, safety_shrink)
    L = max(L, lcp_level)
    L = min(L, max_depth)

    if not verify_corners:
        return L

    for _ in range(max_depth - L + 1):
        shift = max_depth - L
        cell_size = 1 << shift if shift > 0 else 1
        cx_min = x_min // cell_size
        cx_max = x_max // cell_size
        cy_min = y_min // cell_size
        cy_max = y_max // cell_size
        n_cells_x = cx_max - cx_min + 1
        n_cells_y = cy_max - cy_min + 1
        if n_cells_x * n_cells_y > 4096:
            break

        corners: List[Tuple[int, int]] = []
        for cy in range(cy_min, cy_max + 1):
            for cx in range(cx_min, cx_max + 1):
                px0 = cx * cell_size
                py0 = cy * cell_size
                px1 = px0 + cell_size - 1
                py1 = py0 + cell_size - 1
                corners.append((px0, py0))
                corners.append((px1, py0))
                corners.append((px0, py1))
                corners.append((px1, py1))
        if not corners:
            break
        pts = np.asarray(corners, dtype=np.int64)
        in_mask = _pip_winding_batch(pts, polygon_xy, edge_eps=0.0)
        in_mat = in_mask.reshape(-1, 4)
        if in_mat.all(axis=1).any():
            return L
        if L >= max_depth:
            break
        L += 1

    return L


def _rect_at_level(
    polygon_xy: np.ndarray,
    target_level: int,
    max_depth: int,
) -> Tuple[int, int, int, int]:
    """Project polygon MBR into an integer rectangle at target_level."""
    shift = max_depth - target_level
    if shift > 0:
        xs = polygon_xy[:, 0] >> shift
        ys = polygon_xy[:, 1] >> shift
    else:
        xs = polygon_xy[:, 0]
        ys = polygon_xy[:, 1]
    return (int(xs.min()), int(xs.max()),
            int(ys.min()), int(ys.max()))


# ============================================================================
# Multi-resolution polygon grid fill
# ============================================================================

REFINE_LEVELS = 0

FILL_THRESHOLD = 24

FAST_PATH_LCP_LEVEL = 17
LOW_LCP_ALLOW_SMALL_COVER_FASTPATH = False
NEIGHBOR_PIP_DEFAULT = False

MASK_EXTRA_LEVELS = 4
MAX_MASK_CELLS = 4_000_000


def _polygon_fill(
    polygon_xy: np.ndarray,
    target_lv: int,
    max_depth: int,
    rect: Tuple[int, int, int, int],
    lcp: str,
    refine_levels: int = REFINE_LEVELS,
) -> Tuple[List[str], List[str], int, int, int]:
    """
    Multiresolution polygon filling through BFS quadtree subdivision. Return inside and boundary
    prefixes separately so only boundary candidates need PIP verification. Rasterize at
    finest_fill_lv+MASK_EXTRA_LEVELS, usually far below max_depth, to reduce rasterization cost.

    @returns (inside_prefixes, boundary_prefixes, n_inside, n_boundary, n_levels_used).
    """
    finest_fill_lv = min(target_lv + refine_levels, max_depth)
    actual_refine = finest_fill_lv - target_lv

    mask_lv = min(finest_fill_lv + MASK_EXTRA_LEVELS, max_depth)
    mask_lv = max(mask_lv, finest_fill_lv)

    shift_from_max = max_depth - mask_lv
    if shift_from_max > 0:
        poly_mask = polygon_xy >> shift_from_max
    else:
        poly_mask = polygon_xy

    mk_x_min = int(poly_mask[:, 0].min())
    mk_x_max = int(poly_mask[:, 0].max())
    mk_y_min = int(poly_mask[:, 1].min())
    mk_y_max = int(poly_mask[:, 1].max())
    mk_w = mk_x_max - mk_x_min + 1
    mk_h = mk_y_max - mk_y_min + 1

    shift_down = 0
    while mk_w * mk_h > MAX_MASK_CELLS:
        shift_down += 1
        poly_ds = polygon_xy >> (shift_from_max + shift_down)
        mk_x_min = int(poly_ds[:, 0].min())
        mk_x_max = int(poly_ds[:, 0].max())
        mk_y_min = int(poly_ds[:, 1].min())
        mk_y_max = int(poly_ds[:, 1].max())
        mk_w = mk_x_max - mk_x_min + 1
        mk_h = mk_y_max - mk_y_min + 1
        poly_mask = poly_ds

    effective_mask_lv = mask_lv - shift_down

    img = Image.new("L", (mk_w, mk_h), 0)
    draw = ImageDraw.Draw(img)
    pts = [(int(poly_mask[i, 0]) - mk_x_min,
            int(poly_mask[i, 1]) - mk_y_min)
           for i in range(len(poly_mask))]
    draw.polygon(pts, fill=1)
    mask = np.asarray(img, dtype=np.uint8)

    x_min_t, x_max_t, y_min_t, y_max_t = rect
    inside_result: List[str] = []
    boundary_result: List[str] = []

    pending: List[Tuple[int, int, int]] = []
    for cy in range(y_min_t, y_max_t + 1):
        for cx in range(x_min_t, x_max_t + 1):
            pending.append((cx, cy, target_lv))

    while pending:
        next_pending: List[Tuple[int, int, int]] = []
        for cx, cy, lv in pending:
            scale = effective_mask_lv - lv
            cell_size = 1 << scale

            cell_fx = cx << scale
            cell_fy = cy << scale

            mx_s = cell_fx - mk_x_min
            my_s = cell_fy - mk_y_min
            mx_e = mx_s + cell_size
            my_e = my_s + cell_size

            mx_sc = max(mx_s, 0)
            my_sc = max(my_s, 0)
            mx_ec = min(mx_e, mk_w)
            my_ec = min(my_e, mk_h)

            if mx_sc >= mx_ec or my_sc >= my_ec:
                continue

            region = mask[my_sc:my_ec, mx_sc:mx_ec]
            region_sum = int(region.sum())

            fully_in_bounds = (mx_s >= 0 and my_s >= 0
                               and mx_e <= mk_w and my_e <= mk_h)
            cell_area = cell_size * cell_size

            # Strict three-way classification:
            # 1) outside: region mask sum is 0
            # 2) inside: fully inside mask and pixel sum == cell_area
            # 3) boundary: remaining cells intersecting/uncertain on the border
            if region_sum == 0:
                continue
            if fully_in_bounds and region_sum == cell_area:
                inside_result.append(_xy_to_code(cx, cy, lv))
            elif lv < finest_fill_lv:
                for dc in range(4):
                    child_cx = (cx << 1) | (dc & 1)
                    child_cy = (cy << 1) | ((dc >> 1) & 1)
                    next_pending.append((child_cx, child_cy, lv + 1))
            else:
                boundary_result.append(_xy_to_code(cx, cy, lv))

        pending = next_pending

    return (inside_result, boundary_result,
            len(inside_result), len(boundary_result), actual_refine)


# ============================================================================
# Integer-grid PIP — winding number; pure integer cross products + neighborhood edge compensation
# ============================================================================

NEIGHBOR_PIP = NEIGHBOR_PIP_DEFAULT
EDGE_TOLERANCE_ENABLE = True
EDGE_TOLERANCE_BY_LEVEL = {
    20: 1.20,
    21: 1.00,
    22: 0.80,
    23: 0.50,
    24: 0.35,
}
FINAL_PG_FILTER_DEFAULT = False
FINAL_PG_FILTER_LCP_MAX = 16
FINAL_PG_FILTER_TARGET_MAX = 22
FINAL_STRICT_FILTER_DEFAULT = False

_NEIGHBOR_OFFSETS = np.array([
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [1, -1], [-1, 1], [-1, -1],
], dtype=np.int64)


def _winding_number_batch(
    pts_xy: np.ndarray,
    polygon_xy: np.ndarray,
) -> np.ndarray:
    """
    Vectorized integer winding-number PIP using the same cross-product primitive as the geometry
    module: cross(O,A,B)=(A.x-O.x)*(B.y-O.y)-(B.x-O.x)*(A.y-O.y). Use int64
    products/subtraction, without floating-point division or geographic decoding.

    @param pts_xy (M,2) int64 candidate coordinates.
    @param polygon_xy (N,2) int64 polygon vertices.
    @returns (M,) boolean mask.
    """
    n_pts = len(pts_xy)
    n_poly = len(polygon_xy)
    if n_pts == 0 or n_poly < 3:
        return np.zeros(n_pts, dtype=bool)

    px = pts_xy[:, 0]
    py = pts_xy[:, 1]

    winding = np.zeros(n_pts, dtype=np.int32)

    for i in range(n_poly):
        j = (i + 1) % n_poly
        yi = polygon_xy[i, 1]
        yj = polygon_xy[j, 1]

        up = (yi <= py) & (yj > py)
        down = (yi > py) & (yj <= py)

        if not (up.any() or down.any()):
            continue

        cross = ((polygon_xy[j, 0] - polygon_xy[i, 0])
                 * (py - yi)
                 - (px - polygon_xy[i, 0])
                 * (yj - yi))

        winding[up & (cross > 0)] += 1
        winding[down & (cross < 0)] -= 1

    return winding != 0


def _pip_winding_batch(
    pts_xy: np.ndarray,
    polygon_xy: np.ndarray,
    edge_eps: float = 0.0,
) -> np.ndarray:
    """
    Integer winding-number PIP with four-neighbor boundary compensation. First classify points
    by strict winding number. For winding-zero candidates, test the four adjacent integer
    positions; a nonzero neighbor classifies the candidate as boundary-inclusive.

    Vertex snapping by about half a grid cell can displace the discrete boundary by up to one
    cell. This check acts as a one-cell Minkowski dilation intended to approximate boundary-
    inclusive ST_Intersects semantics. It remains a discrete compensation rule, not a general
    equivalence proof.

    @param pts_xy (M,2) int64 candidates.
    @param polygon_xy (N,2) int64 vertices.
    @returns (M,) boolean mask.
    """
    result = _winding_number_batch(pts_xy, polygon_xy)

    if not NEIGHBOR_PIP:
        return result

    rejected_idx = np.where(~result)[0]
    if len(rejected_idx) == 0:
        return result

    rpts = pts_xy[rejected_idx]
    n_off = len(_NEIGHBOR_OFFSETS)
    neighbors = rpts[:, None, :] + _NEIGHBOR_OFFSETS[None, :, :]
    flat = neighbors.reshape(-1, 2)
    flat_inside = _winding_number_batch(flat, polygon_xy)
    any_neighbor_inside = flat_inside.reshape(-1, n_off).any(axis=1)
    result[rejected_idx[any_neighbor_inside]] = True

    if edge_eps <= 0:
        return result

    rejected_idx = np.where(~result)[0]
    if len(rejected_idx) == 0:
        return result

    rpts = pts_xy[rejected_idx].astype(np.float64)
    px = rpts[:, 0]
    py = rpts[:, 1]
    min_d2 = np.full(len(rpts), np.inf, dtype=np.float64)

    n_poly = len(polygon_xy)
    for i in range(n_poly):
        j = (i + 1) % n_poly
        ax = float(polygon_xy[i, 0])
        ay = float(polygon_xy[i, 1])
        bx = float(polygon_xy[j, 0])
        by = float(polygon_xy[j, 1])
        dx = bx - ax
        dy = by - ay
        seg2 = dx * dx + dy * dy
        if seg2 <= 0:
            d2 = (px - ax) * (px - ax) + (py - ay) * (py - ay)
            min_d2 = np.minimum(min_d2, d2)
            continue

        t = ((px - ax) * dx + (py - ay) * dy) / seg2
        t = np.clip(t, 0.0, 1.0)
        qx = ax + t * dx
        qy = ay + t * dy
        d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy)
        min_d2 = np.minimum(min_d2, d2)

    near_edge = min_d2 <= (edge_eps * edge_eps)
    result[rejected_idx[near_edge]] = True
    return result


def _edge_eps_for_level(target_level: int) -> float:
    if not EDGE_TOLERANCE_ENABLE:
        return 0.0
    if target_level in EDGE_TOLERANCE_BY_LEVEL:
        return EDGE_TOLERANCE_BY_LEVEL[target_level]
    if target_level <= 20:
        return 1.20
    if target_level >= 24:
        return 0.35
    return 0.60


# ============================================================================
# Linear integer PIP space (fixes nonlinear PIP from DMS bit-packing)
# ----------------------------------------------------------------------------
# One unit = 1/(3600*2048) degree, approximately 1.5 cm at the equator.
# 1 degree = 7,372,800 units; 1 minute = 122,880 units; 1 second = 2,048 units
# This space is Euclidean-additive; no DMS jumps at deg/min/sec borders; winding cross products are valid.
# ============================================================================

_DEG_TO_LIN = 7_372_800
_MIN_TO_LIN = 122_880
_SEC_TO_LIN = 2_048


def _dms31_to_linear31_int(v: int) -> int:
    """
    Convert one 31-bit packed DMS value into a linear integer coordinate.

    @param v Packed value, using the low 31 bits of _code_to_xy output.
    @returns The same physical position in linear integer space.
    """
    v = v & 0x7FFFFFFF
    deg = (v >> 23) & 0xFF
    minute = (v >> 17) & 0x3F
    sec = (v >> 11) & 0x3F
    sub = v & 0x7FF
    return (deg * _DEG_TO_LIN
            + minute * _MIN_TO_LIN
            + sec * _SEC_TO_LIN
            + sub)


def _xy_dms_to_linear(xy: np.ndarray) -> np.ndarray:
    """
    Vectorize conversion from packed DMS (x,y) to linear integer coordinates. Packed
    minute/second fields contain nonlinear gaps of 8192/491520 units, which can invalidate
    Euclidean cross-product tests across DMS boundaries. Decompose
    degree/minute/second/subsecond fields and recombine them in a continuous integer space,
    approximately 1.5 cm per unit at the equator, without floating-point operations.

    @param xy (N,2) int64 _code_to_xy output with zero MSB.
    @returns (N,2) int64 linear coordinates.
    """
    arr = np.asarray(xy, dtype=np.int64)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        return np.array([
            _dms31_to_linear31_int(int(arr[0])),
            _dms31_to_linear31_int(int(arr[1])),
        ], dtype=np.int64)
    x = arr[:, 0] & 0x7FFFFFFF
    y = arr[:, 1] & 0x7FFFFFFF
    deg_x = (x >> 23) & 0xFF
    min_x = (x >> 17) & 0x3F
    sec_x = (x >> 11) & 0x3F
    sub_x = x & 0x7FF
    lin_x = (deg_x * _DEG_TO_LIN
             + min_x * _MIN_TO_LIN
             + sec_x * _SEC_TO_LIN
             + sub_x)
    deg_y = (y >> 23) & 0xFF
    min_y = (y >> 17) & 0x3F
    sec_y = (y >> 11) & 0x3F
    sub_y = y & 0x7FF
    lin_y = (deg_y * _DEG_TO_LIN
             + min_y * _MIN_TO_LIN
             + sec_y * _SEC_TO_LIN
             + sub_y)
    return np.stack([lin_x, lin_y], axis=1).astype(np.int64)


def _lonlat_to_xy32(lon: float, lat: float) -> Tuple[int, int]:
    """
    Convert northeastern-hemisphere lon/lat to the linear integer space used by
    _xy_dms_to_linear and winding PIP. One unit is 1/(3600*2048) degree. PostGIS rings and
    decoded vertex codes then share one additive integer coordinate space without DMS-boundary
    jumps.

    @returns Integer (x_lin, y_lin), for nonnegative lon/lat.
    """
    try:
        lon = float(lon)
        lat = float(lat)
    except Exception:
        return 0, 0
    if not np.isfinite(lon) or not np.isfinite(lat):
        return 0, 0
    lon = min(max(lon, 0.0), 179.999999999)
    lat = min(max(lat, 0.0), 89.999999999)
    x = int(round(lon * float(_DEG_TO_LIN)))
    y = int(round(lat * float(_DEG_TO_LIN)))
    return x, y


_ring_cache: Dict[Tuple[str, str], List[Tuple[np.ndarray, List[np.ndarray]]]] = {}


def _fetch_polygon_rings_xy_from_postgis(
    pg_conn,
    table: str,
    main_code: str,
) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Read exterior/interior polygon rings from PostGIS and convert them to integer coordinates.

    @returns [(outer_xy, [hole_xy, ...]), ...].
    """
    key = (table, main_code.strip().upper())
    if key in _ring_cache:
        return _ring_cache[key]

    cur = pg_conn.cursor()
    cur.execute(
        f"""
        WITH tgt AS (
          SELECT geometry
          FROM {table}
          WHERE main_code = %s
          LIMIT 1
        ),
        polys AS (
          SELECT
            COALESCE((d).path[1]::int, 1) AS poly_idx,
            (d).geom AS poly_geom
          FROM tgt
          CROSS JOIN LATERAL ST_Dump(tgt.geometry) d
        ),
        rings AS (
          SELECT
            p.poly_idx,
            COALESCE((r).path[1]::int, 0) AS ring_idx,
            ST_ExteriorRing((r).geom) AS ring_line
          FROM polys p
          CROSS JOIN LATERAL ST_DumpRings(p.poly_geom) r
        )
        SELECT
          rg.poly_idx,
          rg.ring_idx,
          i AS pt_idx,
          ST_X(ST_PointN(rg.ring_line, i)) AS lon,
          ST_Y(ST_PointN(rg.ring_line, i)) AS lat
        FROM rings rg
        CROSS JOIN LATERAL generate_series(
          1, GREATEST(ST_NPoints(rg.ring_line) - 1, 1)
        ) AS i
        ORDER BY rg.poly_idx, rg.ring_idx, i
        """,
        (key[1],),
    )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        _ring_cache[key] = []
        return []

    poly_map: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}
    for poly_idx, ring_idx, _pt_idx, lon, lat in rows:
        if poly_idx is None or ring_idx is None:
            continue
        x, y = _lonlat_to_xy32(lon, lat)
        p = poly_map.setdefault(int(poly_idx), {})
        p.setdefault(int(ring_idx), []).append((x, y))

    result: List[Tuple[np.ndarray, List[np.ndarray]]] = []
    for poly_idx in sorted(poly_map.keys()):
        rings = poly_map[poly_idx]
        if 0 not in rings or len(rings[0]) < 3:
            continue
        outer = np.array(rings[0], dtype=np.int64)
        holes: List[np.ndarray] = []
        for ridx in sorted(rings.keys()):
            if ridx == 0:
                continue
            pts = rings[ridx]
            if len(pts) >= 3:
                holes.append(np.array(pts, dtype=np.int64))
        result.append((outer, holes))

    _ring_cache[key] = result
    return result


def _postgis_filter_candidates(
    pg_conn,
    plan_code: str,
    query_table: str,
    target_table: str,
    candidates: List[str],
) -> Tuple[Set[str], float]:
    """Apply final candidate filtering with PostGIS ST_Intersects."""
    if (not pg_conn) or (not query_table) or (not target_table) or (not candidates):
        return set(candidates), 0.0

    t0 = _t()
    cur = pg_conn.cursor()
    cur.execute(
        f"""
        SELECT p.main_code
        FROM {target_table} p
        JOIN {query_table} q
          ON q.main_code = %s
        WHERE p.main_code = ANY(%s)
          AND ST_Intersects(p.geometry, q.geometry)
        """,
        (plan_code.strip().upper(), candidates),
    )
    keep = {
        str(r[0]).strip().upper()
        for r in cur.fetchall()
        if r and r[0] and is_valid_g0_code(str(r[0]).strip().upper())
    }
    cur.close()
    return keep, (_t() - t0) * 1000


def _strict_geosot_filter_candidates(
    candidates: Set[str],
    polygon_xy: np.ndarray,
) -> Tuple[Set[str], float]:
    """
    Strict GeoSOT secondary filtering: use _winding_number_batch without neighbor compensation
    or boundary tolerance. Convert packed DMS vertices to linear integer space to avoid DMS-
    boundary discontinuities.

    @param polygon_xy Packed DMS polygon vertices; converted internally.
    """
    if not candidates:
        return set(), 0.0
    t0 = _t()
    codes = list(candidates)
    pts_dms = np.array(
        [_code_to_xy(c, GRID_LEVEL) for c in codes], dtype=np.int64)
    pts_lin = _xy_dms_to_linear(pts_dms)
    polygon_xy_lin = _xy_dms_to_linear(polygon_xy)
    mask = _winding_number_batch(pts_lin, polygon_xy_lin)
    keep = {codes[i] for i in range(len(codes)) if mask[i]}
    return keep, (_t() - t0) * 1000


def _pip_winding_with_holes(
    pts_xy: np.ndarray,
    polygons_xy: List[Tuple[np.ndarray, List[np.ndarray]]],
    edge_eps_outer: float = 0.0,
) -> np.ndarray:
    """
    A point is inside a multipart polygon when it is inside an exterior ring and outside that
    component's holes.
    """
    n = len(pts_xy)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not polygons_xy:
        return np.zeros(n, dtype=bool)

    hit = np.zeros(n, dtype=bool)
    for outer_xy, holes_xy in polygons_xy:
        in_outer = _pip_winding_batch(pts_xy, outer_xy, edge_eps=edge_eps_outer)
        if not in_outer.any():
            continue
        if holes_xy:
            in_hole = np.zeros(n, dtype=bool)
            for hxy in holes_xy:
                # Holes use a strict test; no border relaxation
                in_hole |= _pip_winding_batch(pts_xy, hxy, edge_eps=0.0)
            in_poly = in_outer & (~in_hole)
        else:
            in_poly = in_outer
        hit |= in_poly
    return hit


def _decide_use_fast_path(
    lcp_lv: int,
    has_lcp: bool,
    n_cover: int,
) -> Tuple[bool, bool]:
    """
    Choose the fast path. A valid LCP at least FAST_PATH_LCP_LEVEL always qualifies; otherwise
    LOW_LCP_ALLOW_SMALL_COVER_FASTPATH controls the small-cover alternative.

    @param lcp_lv LCP level.
    @param has_lcp Whether an LCP exists.
    @param n_cover Minimal prefix cover size.
    @returns (use_fast_path_lcp, use_fast_path).
    """
    use_fast_path_lcp = (lcp_lv >= FAST_PATH_LCP_LEVEL) and has_lcp
    use_fast_path_small_cover = (
        LOW_LCP_ALLOW_SMALL_COVER_FASTPATH
        and (n_cover <= FILL_THRESHOLD)
    )
    return use_fast_path_lcp, (use_fast_path_lcp or use_fast_path_small_cover)


# ============================================================================
# POI Grid Cache — pure integers; no lon/lat stored
# ============================================================================

class POIGridCacheV6:
    """
    Preload POIs as level-32 integer (x,y) coordinates using bit operations on codes, without
    geographic decoding.
    """

    def __init__(self):
        self.main_codes: np.ndarray = np.array([])
        self.xy: np.ndarray = np.empty((0, 2), dtype=np.int64)
        self.ref_level: int = GRID_LEVEL

    @property
    def count(self) -> int:
        return len(self.main_codes)

    def preload(
        self, session, source: str,
        verbose: bool = True,
    ) -> int:
        """
        Preload source codes from Neo4j and decode Z-order bits to integer (x,y), without
        fast_code_to_center or geographic decoding.

        @param session Neo4j session.
        @param source Target source.
        @returns Number of POIs.
        """
        t0 = _t()
        rows = [dict(r) for r in session.run(
            "MATCH (c:GeoSOTCentroidCode)"
            "-[:FROM_SOURCE]->(:DataSource {name: $src})\n"
            "RETURN c.main_code AS mc",
            src=source,
        )]
        t_query = (_t() - t0) * 1000
        if verbose:
            print(f"    Neo4j 查询: {t_query:.0f} ms  "
                  f"({len(rows):,} 条)")

        t0 = _t()
        codes = [str(r["mc"]).strip().upper()
                 for r in rows
                 if is_valid_g0_code(str(r["mc"]).strip().upper())]
        t_filter = (_t() - t0) * 1000
        if verbose:
            print(f"    有效过滤:   {t_filter:.0f} ms  "
                  f"({len(codes):,} 个有效 POI)")

        t0 = _t()
        self.main_codes = np.array(codes, dtype=object)
        self.xy = _codes_to_xy_batch(codes, self.ref_level)
        t_grid = (_t() - t0) * 1000
        if verbose:
            print(f"    Z-order:    {t_grid:.0f} ms  "
                  f"(Level {self.ref_level} 网格坐标)")

        return len(codes)

    def query_rect(
        self,
        x_min: int, x_max: int,
        y_min: int, y_max: int,
        level: int,
    ) -> np.ndarray:
        """
        Filter candidates by an integer rectangle. Right-shift ref_level coordinates to the query level.

        @returns Boolean mask of shape (N_poi,).
        """
        shift = self.ref_level - level
        qx = self.xy[:, 0] >> shift
        qy = self.xy[:, 1] >> shift

        return ((qx >= x_min) & (qx <= x_max)
                & (qy >= y_min) & (qy <= y_max))


# ============================================================================
# Plan Preloader — store vertex codes only; no coordinate decode
# ============================================================================

class PlanPreloaderV6:
    """
    Preload polygon vertex_codes only. PIP operates in integer grid space rather than decoded
    lon/lat.
    """

    def __init__(self, session, verbose: bool = True):
        self._session = session
        self._verbose = verbose
        self._plans: Dict[str, Dict] = {}
        self._plan_codes: List[str] = []

    @property
    def count(self) -> int:
        return len(self._plans)

    @property
    def plan_codes(self) -> List[str]:
        return self._plan_codes

    def preload(self, query_source: str = "Statutory Plan") -> int:
        t0 = _t()
        rows = [dict(r) for r in self._session.run(
            "MATCH (f:SpatialFeature)"
            "-[:HAS_CENTROID_CODE]->(c:GeoSOTCentroidCode)"
            "-[:FROM_SOURCE]->(:DataSource {name: $src})\n"
            "MATCH (c)"
            "-[:HAS_GEOMETRY_TYPE]->"
            "(:GeoFeatureType {type_name: 'Polygon'})\n"
            "OPTIONAL MATCH (f)-[:HAS_COMMON_PREFIX_CODE]->"
            "(cp:GeoSOTCommonPrefixCode)\n"
            "OPTIONAL MATCH (f)-[hv:HAS_VERTEX_CODE]->"
            "(v:GeoSOTVertexCode)\n"
            "WITH f, c, cp, v, hv ORDER BY f.feature_id, "
            "hv.seq\n"
            "WITH f.feature_id AS code_id,\n"
            "     c.main_code AS main_code,\n"
            "     cp.prefix_code  AS prefix_code,\n"
            "     cp.prefix_level AS prefix_level,\n"
            "     collect(CASE WHEN v IS NOT NULL\n"
            "             THEN v.vertex_code "
            "ELSE null END) AS raw_verts\n"
            "WITH code_id, main_code, prefix_code, prefix_level,\n"
            "     [x IN raw_verts WHERE x IS NOT NULL]"
            " AS vertex_codes\n"
            "WHERE size(vertex_codes) >= 3\n"
            "RETURN code_id, main_code, prefix_code, prefix_level,"
            " vertex_codes",
            src=query_source,
        )]
        t_query = (_t() - t0) * 1000
        if self._verbose:
            print(f"    Neo4j 查询:  {t_query:.0f} ms  "
                  f"({len(rows):,} 条)")

        t0 = _t()
        self._plans = {}
        for r in rows:
            mc = str(r["main_code"]).strip().upper()
            vcodes = [str(v).strip().upper()
                      for v in r["vertex_codes"]]
            valid_vcodes = [vc for vc in vcodes
                           if is_valid_g0_code(vc)]
            if len(valid_vcodes) < 3:
                continue
            pcode = (str(r["prefix_code"]).strip().upper()
                     if r.get("prefix_code") else None)
            plevel = (int(r["prefix_level"])
                      if r.get("prefix_level") is not None else None)
            self._plans[mc] = {
                "main_code": mc,
                "vertex_codes": valid_vcodes,
                "prefix_code": pcode,
                "prefix_level": plevel,
            }

        self._plan_codes = list(self._plans.keys())
        t_build = (_t() - t0) * 1000
        if self._verbose:
            print(f"    构建:        {t_build:.0f} ms  "
                  f"({len(self._plans):,} 个有效 Plan)")
        return len(self._plans)

    def get_plan(self, main_code: str) -> Optional[Dict]:
        return self._plans.get(main_code.strip().upper())


# ============================================================================
# GR-side main_code client index (counterpart to PG pg_prewarm)
# ============================================================================
#
# Methodology disclosure (same protocol as distance experiments):
# PG uses pg_prewarm to load tables + GiST/B-tree into shared_buffers; GR
# pulls all main_codes for a source from Neo4j into Python memory once,
# sorts lexicographically, and does local prefix match via bisect (O(log N + k)). This is graph
# An in-memory index copy, not a replacement for nodes or HAS_VERTEX_CODE/HAS_COMMON_PREFIX_CODE edges.
# still lives in Neo4j as the sole authoritative data source.

import bisect as _bisect


class MainCodeIndex:
    """
    Client-side main_code prefix index built during explicit G3R setup alongside PG prewarming.

    @attr _sorted Sorted main-code lists keyed by source.
    @attr elapsed_sec Load time in seconds.
    """

    def __init__(self, session, sources, verbose: bool = True):
        """
        @param session Neo4j session.
        @param sources Source names to preload, for example ["POI"].
        @param verbose Whether to print progress.
        """
        self._sorted: Dict[str, List[str]] = {}
        t0 = _t()
        for src in sources:
            t_src = _t()
            try:
                rows = list(session.run(
                    "MATCH (c:GeoSOTCentroidCode)-[:FROM_SOURCE]->"
                    "(:DataSource {name: $src}) "
                    "RETURN c.main_code AS mc",
                    src=src))
            except Exception as e:
                if verbose:
                    print(f"  [main_code_index] {src} 加载失败: {e}")
                self._sorted[src] = []
                continue
            codes = sorted({
                str(r["mc"]).strip().upper()
                for r in rows
                if r.get("mc") is not None
                and is_valid_g0_code(str(r["mc"]).strip().upper())
            })
            self._sorted[src] = codes
            if verbose:
                dt = (_t() - t_src) * 1000
                print(f"  [main_code_index] {src:<20s}: "
                      f"{len(codes):>9,} codes  ({dt:.0f} ms)")
        self.elapsed_sec = _t() - t0
        if verbose:
            total = sum(len(c) for c in self._sorted.values())
            print(f"  [main_code_index] 完成: 共 {total:,} 个 codes, "
                  f"耗时 {self.elapsed_sec * 1000:.0f} ms")

    def find_by_prefix(self, source: str, prefix: str) -> Set[str]:
        """
        Match a prefix locally with bisect in O(log N+k).

        @param source Source name.
        @param prefix GeoSOT prefix.
        @returns Matching main_code set.
        """
        codes = self._sorted.get(source)
        if not codes:
            return set()
        lo = _bisect.bisect_left(codes, prefix)
        hi = _bisect.bisect_right(codes, prefix + '\uffff')
        return set(codes[lo:hi])

    def n_codes(self, source: str) -> int:
        return len(self._sorted.get(source, []))

    def has_source(self, source: str) -> bool:
        return source in self._sorted and len(self._sorted[source]) > 0


# Module singleton injected by main at setup; None falls back to Neo4j queries
_main_code_index: Optional[MainCodeIndex] = None


def set_main_code_index(idx: Optional[MainCodeIndex]):
    """Inject client main_code index (None → cold-cache mode)."""
    global _main_code_index
    _main_code_index = idx


def get_main_code_index() -> Optional[MainCodeIndex]:
    return _main_code_index


# ============================================================================
# Neo4j queries + prefix cache
# ============================================================================

_prefix_cache: Dict[Tuple[str, str], Set[str]] = {}


def _query_neo4j_single_prefix(
    session, prefix: str, source: str,
) -> Set[str]:
    key = (source, prefix)
    if key in _prefix_cache:
        return _prefix_cache[key]

    # Prefer local index (counterpart to PG pg_prewarm)
    if _main_code_index is not None and \
            _main_code_index.has_source(source):
        result = _main_code_index.find_by_prefix(source, prefix)
        _prefix_cache[key] = result
        return result

    cypher = (
        "MATCH (c:GeoSOTCentroidCode)"
        "-[:FROM_SOURCE]->(:DataSource {name: $src})\n"
        "WHERE c.main_code STARTS WITH $prefix\n"
        "RETURN DISTINCT c.main_code AS mc"
    )
    result: Set[str] = set()
    rows = session.run(cypher, prefix=prefix, src=source)
    for r in rows:
        mc = str(r["mc"]).strip().upper()
        if is_valid_g0_code(mc):
            result.add(mc)

    _prefix_cache[key] = result
    return result


def prefix_cache_clear():
    _prefix_cache.clear()


def prefix_cache_stats() -> Tuple[int, int]:
    return len(_prefix_cache), sum(
        len(v) for v in _prefix_cache.values())


def _query_neo4j_by_cells(
    session, cells: List[str], source: str,
    batch_size: int = 1024,
) -> List[str]:
    """
    Batch Neo4j STARTS WITH queries for a minimal prefix set. The graph stores/indexes
    GeoSOTCentroidCode nodes, constrains sources through CC-FROM_SOURCE->DataSource, and maps
    spatial prefixes to candidates. When MainCodeIndex was installed during setup, local bisect
    matching replaces Bolt queries; account for that materialization separately.
    """
    result_set: Set[str] = set()
    uncached: List[str] = []
    for prefix in cells:
        key = (source, prefix)
        if key in _prefix_cache:
            result_set |= _prefix_cache[key]
        else:
            uncached.append(prefix)

    # Prefer local main_code index
    if uncached and _main_code_index is not None and \
            _main_code_index.has_source(source):
        for prefix in uncached:
            hit = _main_code_index.find_by_prefix(source, prefix)
            _prefix_cache[(source, prefix)] = hit
            result_set |= hit
        return list(result_set)

    if uncached:
        cypher = (
            "UNWIND $prefixes AS prefix\n"
            "MATCH (c:GeoSOTCentroidCode)"
            "-[:FROM_SOURCE]->(:DataSource {name: $src})\n"
            "WHERE c.main_code STARTS WITH prefix\n"
            "RETURN DISTINCT c.main_code AS mc, prefix"
        )
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            per_prefix: Dict[str, Set[str]] = {
                p: set() for p in batch}
            rows = session.run(
                cypher, prefixes=batch, src=source)
            for r in rows:
                mc = str(r["mc"]).strip().upper()
                pfx = str(r["prefix"])
                if is_valid_g0_code(mc):
                    per_prefix.setdefault(pfx, set()).add(mc)
                    result_set.add(mc)
            for pfx, codes in per_prefix.items():
                _prefix_cache[(source, pfx)] = codes

    return list(result_set)


def _query_neo4j_split(
    session, inside_cells: List[str], boundary_cells: List[str],
    source: str, batch_size: int = 1024,
) -> Tuple[Set[str], Set[str]]:
    """
    Query inside and boundary prefixes together, then split results using the per-prefix cache.
    Quadtree BFS produces non-overlapping inside/boundary prefix regions, so one batch query
    suffices without cross-contaminating the result groups.
    """
    all_cells = list(inside_cells) + list(boundary_cells)
    if all_cells:
        _query_neo4j_by_cells(
            session, all_cells, source, batch_size=batch_size)

    inside_set: Set[str] = set()
    for pfx in inside_cells:
        inside_set |= _prefix_cache.get((source, pfx), set())

    boundary_set: Set[str] = set()
    for pfx in boundary_cells:
        boundary_set |= _prefix_cache.get((source, pfx), set())

    return inside_set, boundary_set - inside_set


def _fetch_polygon_vcodes_from_neo4j(
    session, main_code: str,
) -> Tuple[Optional[str], List[str], Optional[str], Optional[int]]:
    """
    Fetch ordered vertex codes and a precomputed LCPC through SpatialFeature-
    HAS_VERTEX_CODE->GeoSOTVertexCode and feature-
    HAS_COMMON_PREFIX_CODE->GeoSOTCommonPrefixCode. The importer computes the prefix once;
    queries read it without recomputing LCP.

    @returns (code_id, vertex_codes, prefix_code, prefix_level).
    """
    rows = [dict(r) for r in session.run(
        "MATCH (c:GeoSOTCentroidCode {main_code: $mc})"
        "<-[:HAS_CENTROID_CODE]-(p:SpatialFeature)\n"
        "OPTIONAL MATCH (p)-[:HAS_COMMON_PREFIX_CODE]->"
        "(cp:GeoSOTCommonPrefixCode)\n"
        "OPTIONAL MATCH (p)-[hv:HAS_VERTEX_CODE]->"
        "(v:GeoSOTVertexCode)\n"
        "WITH p, cp, v, hv ORDER BY hv.seq\n"
        "WITH p.feature_id AS code_id,\n"
        "     cp.prefix_code   AS prefix_code,\n"
        "     cp.prefix_level  AS prefix_level,\n"
        "     collect(CASE WHEN v IS NOT NULL\n"
        "             THEN v.vertex_code "
        "ELSE null END) AS raw_verts\n"
        "RETURN code_id, prefix_code, prefix_level,\n"
        "       [x IN raw_verts WHERE x IS NOT NULL]"
        " AS vertex_codes",
        mc=main_code.strip().upper())]

    if not rows:
        return None, [], None, None

    r = rows[0]
    cid = str(r["code_id"]) if r["code_id"] else ""
    vcodes = [str(v).strip().upper() for v in r["vertex_codes"]]
    pcode = (str(r["prefix_code"]).strip().upper()
             if r.get("prefix_code") else None)
    plevel = (int(r["prefix_level"])
              if r.get("prefix_level") is not None else None)
    return cid, vcodes, pcode, plevel


# ============================================================================
# Single query
# ============================================================================

def run_single_query(
    session, pg_conn,
    query_code: str,
    source: str,
    query_source: str,
    compare_pg: bool,
    max_cells: int = 256,
    refine_levels: int = REFINE_LEVELS,
    plan_preloader: Optional[PlanPreloaderV6] = None,
    poi_cache: Optional[POIGridCacheV6] = None,
):
    W = 76
    SEP = f"  {'─' * (W - 4)}"
    qcode = query_code.strip().upper()
    strategy = "polygon_fill"

    print(f"\n{'=' * W}")
    print("GeoSOT 叠置查询 V7 — 多分辨率多边形填充")
    print(f"{'=' * W}")
    print(f"  查询编码:   {qcode}")
    print(f"  查询来源:   {query_source}")
    print(f"  目标来源:   {source}")
    print(f"  查询策略:   {strategy}")
    print(f"  网格层级:   Level {GRID_LEVEL}")
    print(f"  最大网格数: {max_cells}")

    # ── Step 1: get polygon vertex codes → (x,y) ──
    print(f"\n{SEP}")
    print(f"  Step 1: 获取折点编码 → 整数网格坐标")
    print(f"{SEP}")

    t0 = _t()
    plan_data = (plan_preloader.get_plan(qcode)
                 if plan_preloader else None)
    graph_prefix: Optional[str] = None
    graph_prefix_level: Optional[int] = None
    if plan_data:
        vcodes = plan_data["vertex_codes"]
        graph_prefix = plan_data.get("prefix_code")
        graph_prefix_level = plan_data.get("prefix_level")
        print(f"  (来源: 内存预加载)")
    else:
        _cid, vcodes, graph_prefix, graph_prefix_level = \
            _fetch_polygon_vcodes_from_neo4j(session, qcode)
        print(f"  (来源: Neo4j 查询)")

    vcodes = [vc for vc in vcodes if is_valid_g0_code(vc)]
    t_fetch = (_t() - t0) * 1000

    if len(vcodes) < 3:
        print(f"  [ERROR] 有效顶点不足 ({len(vcodes)} 个)")
        return

    t0 = _t()
    polygon_xy = np.array(
        [_code_to_xy(vc, GRID_LEVEL) for vc in vcodes],
        dtype=np.int64)
    polygon_xy_pip = _xy_dms_to_linear(polygon_xy)
    t_xy = (_t() - t0) * 1000

    def _pip_eval(pts: np.ndarray) -> np.ndarray:
        pts_lin = _xy_dms_to_linear(pts)
        return _pip_winding_batch(
            pts_lin, polygon_xy_pip, edge_eps=edge_eps)

    print(f"  顶点数:     {len(vcodes)}")
    print(f"  查询耗时:   {t_fetch:.1f} ms")
    print(f"  (x,y) 转换: {t_xy:.4f} ms")

    # ── Step 2: Z-order cover rectangle + minimal prefix set ──
    print(f"\n{SEP}")
    print(f"  Step 2: Z-order 覆盖矩形 + 最小前缀集")
    print(f"{SEP}")

    t0 = _t()
    vertex_xy_list = [tuple(row) for row in polygon_xy]
        # Prefer precomputed GeoSOTCommonPrefixCode nodes in the knowledge graph;
    # fall back to local LCP only when missing (data not built or sparse polygons).
    if graph_prefix:
        lcp = graph_prefix
        lcp_lv = (graph_prefix_level
                  if graph_prefix_level is not None
                  else max(len(lcp) - 1, 0))
        lcp_from_graph = True
    else:
        lcp = _lcp_of(vcodes)
        lcp_lv = max(len(lcp) - 1, 0)
        lcp_from_graph = False
    max_depth = min(len(vc) - 1 for vc in vcodes)
    max_depth = min(max_depth, GRID_LEVEL)

    target_lv, rect = _determine_target_level(
        vertex_xy_list, lcp_lv, max_depth, max_cells)
    x_min, x_max, y_min, y_max = rect
    rect_w = x_max - x_min + 1
    rect_h = y_max - y_min + 1
    rect_area = rect_w * rect_h

    use_fast_path_lcp = (lcp_lv >= FAST_PATH_LCP_LEVEL) and bool(lcp)
    if use_fast_path_lcp:
        cover_cells = [lcp]
    else:
        cover_cells = _rect_to_min_prefixes(
            x_min, x_max, y_min, y_max, target_lv, lcp)
    n_prefixes = len(cover_cells)
    t_grid = (_t() - t0) * 1000
    edge_eps = _edge_eps_for_level(target_lv)

    lcp_src = "图谱节点" if lcp_from_graph else "本地计算"
    print(f"  全局 LCP:     {lcp}  (Level {lcp_lv})  [来源: {lcp_src}]")
    print(f"  目标层级:     Level {target_lv}  "
          f"(max_depth={max_depth})")
    print(f"  覆盖矩形:     {rect_w} x {rect_h} = "
          f"{rect_area} 个网格")
    if use_fast_path_lcp:
        print(f"  单前缀入口:   LCP='{lcp}' (Level {lcp_lv})")
    else:
        print(f"  最小前缀覆盖: {n_prefixes} 个前缀  "
              f"(矩形 {rect_area} -> 前缀 {n_prefixes})")
    print(f"  矩形范围:     x=[{x_min}..{x_max}], "
          f"y=[{y_min}..{y_max}]")
    if 2 <= target_lv <= 32:
        sx, sy = GRID_SIZES[target_lv]
        print(f"  网格步长:     {sx:.8f} x {sy:.8f} deg")
    print(f"  耗时:         {t_grid:.4f} ms")

    if n_prefixes <= 20:
        for cc in cover_cells:
            print(f"    {cc}")

    # ── Step 3-4: adaptive query ──
    print(f"\n{SEP}")
    print(f"  Step 3-4: 自适应知识图谱匹配 + 绕数法 PIP")
    print(f"{SEP}")

    use_fast_path_lcp, use_fast_path = _decide_use_fast_path(
        lcp_lv=lcp_lv, has_lcp=bool(lcp), n_cover=n_prefixes)

    if use_fast_path:
        reason = ("LCP>=%d, 单前缀 '%s'" % (FAST_PATH_LCP_LEVEL, lcp)
                  if use_fast_path_lcp
                  else "bbox %d <= %d (已启用低LCP小覆盖快速路径)"
                  % (n_prefixes, FILL_THRESHOLD))
        print(f"  策略: 快速路径 ({reason})")
        t_fill = 0.0
        n_inside = 0
        n_boundary = n_prefixes
        n_refine = 0
        n_fill = n_prefixes

        t0 = _t()
        all_hits = _query_neo4j_by_cells(
            session, cover_cells, source)
        t_query = (_t() - t0) * 1000

        t0 = _t()
        if all_hits:
            all_xy = np.array(
                [_code_to_xy(c, GRID_LEVEL) for c in all_hits],
                dtype=np.int64)
            pip_mask = _pip_eval(all_xy)
            pip_hits = [all_hits[i]
                        for i in range(len(all_hits))
                        if pip_mask[i]]
        else:
            pip_hits = []
        t_pip = (_t() - t0) * 1000

        n_bnd_candidates = len(all_hits)
        n_pip_pass = len(pip_hits)
        n_pip_reject = n_bnd_candidates - n_pip_pass
        results = pip_hits
    else:
        print(f"  策略: 填充路径 (LCP<{FAST_PATH_LCP_LEVEL} 或 "
              f"未启用低LCP小覆盖快速路径)")
        t0 = _t()
        inside_pf, boundary_pf, n_inside, n_boundary, n_refine = \
            _polygon_fill(polygon_xy, target_lv, max_depth, rect,
                          lcp, refine_levels=refine_levels)
        t_fill = (_t() - t0) * 1000
        n_fill = n_inside + n_boundary

        t0 = _t()
        inside_set, boundary_only_set = _query_neo4j_split(
            session, inside_pf, boundary_pf, source)
        boundary_only = list(boundary_only_set)
        t_query = (_t() - t0) * 1000

        t0 = _t()
        if boundary_only:
            bnd_xy = np.array(
                [_code_to_xy(c, GRID_LEVEL) for c in boundary_only],
                dtype=np.int64)
            pip_mask = _pip_eval(bnd_xy)
            pip_hits = [boundary_only[i]
                        for i in range(len(boundary_only))
                        if pip_mask[i]]
        else:
            pip_hits = []
        t_pip = (_t() - t0) * 1000

        n_bnd_candidates = len(boundary_only)
        n_pip_pass = len(pip_hits)
        n_pip_reject = n_bnd_candidates - n_pip_pass
        results = list(inside_set | set(pip_hits))

    print(f"  填充前缀:    {n_inside} inside + "
          f"{n_boundary} boundary = {n_fill}")
    print(f"  PIP 候选:    {n_bnd_candidates} → "
          f"通过 {n_pip_pass}, 排除 {n_pip_reject}")
    print(f"  最终命中:    {len(results):,}")
    print(f"  填充: {t_fill:.3f} ms  |  "
          f"图匹配: {t_query:.1f} ms  |  "
          f"PIP: {t_pip:.3f} ms")

    # ── Summary ──
    t_total = t_fetch + t_xy + t_grid + t_fill + t_query + t_pip
    print(f"\n  图推理总耗时: {t_total:.1f} ms")
    print(f"    1 获取折点编码:     {t_fetch:>8.1f} ms")
    print(f"    2 折点->(x,y):      {t_xy:>8.4f} ms")
    print(f"    3 覆盖矩形:        {t_grid:>8.4f} ms  "
          f"({rect_area} 格)")
    print(f"    4 多边形填充:       {t_fill:>8.3f} ms  "
          f"({n_fill} 前缀)")
    print(f"    5 图匹配:           {t_query:>8.3f} ms")
    print(f"    6 绕数PIP:          {t_pip:>8.3f} ms  "
          f"({n_pip_pass} 通过 / {n_pip_reject} 排除)")

    # ── PostGIS comparison ──
    if compare_pg and pg_conn:
        print(f"\n{SEP}")
        print(f"  PostGIS ST_Intersects 对比")
        print(f"{SEP}")

        qt = SOURCE_TO_PG_TABLE.get(query_source, "")
        tt = SOURCE_TO_PG_TABLE.get(source, "")
        if qt and tt:
            pg_results, t_pg = postgis_overlay_features(
                pg_conn, qcode, qt, tt)
            print(f"  找到要素:   {len(pg_results)} 个")
            print(f"  PostGIS 耗时: {t_pg:.1f} ms")

            gr_set = set(results)
            pg_set = set(pg_results)
            common = gr_set & pg_set

            print(f"\n{SEP}")
            print(f"  结果对比")
            print(f"{SEP}")
            print(f"  图推理:    {len(gr_set)} 个")
            print(f"  PostGIS:   {len(pg_set)} 个")
            print(f"  共同:      {len(common)} 个")
            print(f"  仅图推理:  {len(gr_set - pg_set)} 个")
            print(f"  仅 PostGIS: {len(pg_set - gr_set)} 个")
            if pg_set:
                print(f"  召回率:    "
                      f"{len(common)/len(pg_set):.1%}  "
                      f"({len(common)}/{len(pg_set)})")
            if gr_set:
                print(f"  精确率:    "
                      f"{len(common)/len(gr_set):.1%}  "
                      f"({len(common)}/{len(gr_set)})")
            print(f"\n  耗时对比:")
            print(f"    图推理: {t_total:>10.1f} ms")
            print(f"    PostGIS: {t_pg:>10.1f} ms")
            if t_total > 0:
                print(f"    比率:   {t_pg / t_total:.2f}x")

    if results:
        show_n = min(len(results), 30)
        print(f"\n{SEP}")
        print(f"  查询结果 (前 {show_n} / 共 {len(results)} 个)")
        print(f"{SEP}")
        for i, mc in enumerate(sorted(results)[:show_n]):
            x, y = _code_to_xy(mc, GRID_LEVEL)
            print(f"  {i+1:>4d}. {mc}  (x={x}, y={y})")
        if len(results) > show_n:
            print(f"  ... 还有 {len(results) - show_n} 个")

    print(f"\n{'=' * W}")


# ============================================================================
# Batch test
# ============================================================================

CSV_FIELDS = [
    "plan_idx", "plan_code", "plan_vertices",
    "pg_area_m2", "pg_lon_span_deg", "pg_lat_span_deg",
    "lcp", "lcp_level",
    "target_level", "rect_width", "rect_height", "rect_area",
    "refine_levels", "finest_level",
    "n_fill_inside", "n_fill_boundary", "n_fill_total",
    "n_boundary_candidates", "n_pip_pass", "n_pip_reject",
    "strategy",
    "n_hits",
    "gr_matched_codes",
    "t_fetch_ms", "t_vertex_xy_ms", "t_grid_ms",
    "t_fill_ms", "t_query_ms", "t_pip_ms",
    "t_gr_total_ms",
    "gr_time_ms", "pg_time_ms",
    "t_pg_filter_ms", "n_pg_filter_removed",
    "t_strict_filter_ms", "n_strict_filter_removed",
    "pg_count", "t_pg_ms",
    "speedup",
    "recall", "precision", "f1",
    "match",
    "gr_only_count", "pg_only_count",
]


def _preload_polygon_metrics(
    pg_conn, table: str, main_codes: List[str],
) -> Dict[str, Dict]:
    """
    Batch-read polygon area and geographic span from PostGIS. ST_Area(geography) gives m^2;
    ST_XMin/ST_XMax/ST_YMin/ST_YMax give spans in degrees.

    @returns {main_code: {area_m2, lon_span, lat_span}}.
    """
    if not pg_conn or not table or not main_codes:
        return {}

    cur = pg_conn.cursor()
    cur.execute(
        f"SELECT main_code, "
        f"ST_Area(geometry::geography) AS area_m2, "
        f"ST_XMax(geometry) - ST_XMin(geometry) AS lon_span, "
        f"ST_YMax(geometry) - ST_YMin(geometry) AS lat_span "
        f"FROM {table} "
        f"WHERE main_code = ANY(%s)",
        (main_codes,))
    result = {}
    for r in cur.fetchall():
        mc = str(r[0]).strip().upper()
        result[mc] = {
            "area_m2": round(float(r[1]), 2) if r[1] else "",
            "lon_span": round(float(r[2]), 8) if r[2] else "",
            "lat_span": round(float(r[3]), 8) if r[3] else "",
        }
    cur.close()
    return result


def run_batch(
    session, pg_conn,
    plan_codes: List[str],
    source: str,
    query_source: str,
    compare_pg: bool,
    strategy: str = "adaptive",
    max_cells: int = 256,
    refine_levels: int = REFINE_LEVELS,
    force_fast_min_prefix: bool = False,
    enable_final_pg_filter: bool = FINAL_PG_FILTER_DEFAULT,
    enable_final_strict_filter: bool = FINAL_STRICT_FILTER_DEFAULT,
    all_candidates_pip: bool = False,
    mbr_cover_pip: bool = False,
    mbr_cover_safety_shrink: int = 0,
    output_csv: str = "",
    plan_preloader: Optional[PlanPreloaderV6] = None,
    poi_cache: Optional[POIGridCacheV6] = None,
) -> List[Dict]:
    W = 76
    SEP = f"  {'─' * (W - 4)}"
    n = len(plan_codes)

    qt = SOURCE_TO_PG_TABLE.get(query_source, "")
    tt = SOURCE_TO_PG_TABLE.get(source, "")

    pg_metrics: Dict[str, Dict] = {}
    if compare_pg and pg_conn and qt:
        t0 = _t()
        pg_metrics = _preload_polygon_metrics(
            pg_conn, qt, plan_codes)
        t_pm = (_t() - t0) * 1000
        print(f"  PostGIS 多边形指标预加载: {len(pg_metrics):,} 条  "
              f"({t_pm:.0f} ms)")

    rows_out: List[Dict] = []
    t_start = _t()
    bar_update = max(n // 100, 1)

    for idx, qc in enumerate(plan_codes):
        if idx % bar_update == 0:
            elapsed = _t() - t_start
            pct = idx / n if n else 0
            rate = idx / elapsed if elapsed > 0.001 else 0
            eta = (n - idx) / rate if rate > 0 else 0
            bar_filled = int(30 * pct)
            bar = "#" * bar_filled + "-" * (30 - bar_filled)
            sys.stdout.write(
                f"\r  [{bar}] "
                f"{pct*100:5.1f}%  {idx:,}/{n:,}  "
                f"{rate:,.1f}/s  ETA {eta:.0f}s")
            sys.stdout.flush()

        qcode = qc.strip().upper()

        # 1. Fetch vertex codes + precomputed common prefix (GeoSOTCommonPrefixCode nodes)
        t0 = _t()
        plan_data = (plan_preloader.get_plan(qcode)
                     if plan_preloader else None)
        graph_prefix: Optional[str] = None
        graph_prefix_level: Optional[int] = None
        if plan_data:
            vcodes = plan_data["vertex_codes"]
            graph_prefix = plan_data.get("prefix_code")
            graph_prefix_level = plan_data.get("prefix_level")
        else:
            _cid, vcodes, graph_prefix, graph_prefix_level = \
                _fetch_polygon_vcodes_from_neo4j(session, qcode)
        vcodes = [vc for vc in vcodes if is_valid_g0_code(vc)]
        t_fetch = (_t() - t0) * 1000

        if len(vcodes) < 3:
            pm = pg_metrics.get(qcode, {})
            rows_out.append({k: "" for k in CSV_FIELDS})
            rows_out[-1].update({
                "plan_idx": idx, "plan_code": qcode,
                "plan_vertices": len(vcodes),
                "pg_area_m2": pm.get("area_m2", ""),
                "pg_lon_span_deg": pm.get("lon_span", ""),
                "pg_lat_span_deg": pm.get("lat_span", ""),
                "strategy": "invalid_polygon",
                "n_hits": 0,
                "n_boundary_candidates": 0,
                "n_pip_pass": 0, "n_pip_reject": 0,
                "t_fetch_ms": round(t_fetch, 3),
                "t_vertex_xy_ms": 0, "t_grid_ms": 0,
                "t_fill_ms": 0, "t_query_ms": 0,
                "t_pip_ms": 0,
                "t_gr_total_ms": round(t_fetch, 3),
                "gr_time_ms": round(t_fetch, 3),
                "pg_time_ms": "",
                "t_pg_filter_ms": 0,
                "n_pg_filter_removed": 0,
                "t_strict_filter_ms": 0,
                "n_strict_filter_removed": 0,
            })
            continue

        # 2. Vertices → (x,y) + cover rectangle
        # polygon_xy:     DMS bit-packed integers for cell-level ops (fill/MBR/prefix)
        # polygon_xy_pip: linear integers for winding PIP (avoid DMS cross-border nonlinearity)
        t0 = _t()
        polygon_xy = np.array(
            [_code_to_xy(vc, GRID_LEVEL) for vc in vcodes],
            dtype=np.int64)
        polygon_xy_pip = _xy_dms_to_linear(polygon_xy)
        t_vertex_xy = (_t() - t0) * 1000

        t0 = _t()
        vertex_xy_list = [tuple(row) for row in polygon_xy]
        # Prefer common prefixes precomputed in GeoSOTCommonPrefixCode nodes
        if graph_prefix:
            lcp = graph_prefix
            lcp_lv = (graph_prefix_level
                      if graph_prefix_level is not None
                      else max(len(lcp) - 1, 0))
        else:
            lcp = _lcp_of(vcodes)
            lcp_lv = max(len(lcp) - 1, 0)
        max_depth = min(len(vc) - 1 for vc in vcodes)
        max_depth = min(max_depth, GRID_LEVEL)

        target_lv, rect = _determine_target_level(
            vertex_xy_list, lcp_lv, max_depth, max_cells)
        x_min, x_max, y_min, y_max = rect
        rect_w = x_max - x_min + 1
        rect_h = y_max - y_min + 1
        rect_area = rect_w * rect_h
        t_grid = (_t() - t0) * 1000
        edge_eps = _edge_eps_for_level(target_lv)

        def _pip_eval(pts: np.ndarray) -> np.ndarray:
            pts_lin = _xy_dms_to_linear(pts)
            return _pip_winding_batch(
                pts_lin, polygon_xy_pip, edge_eps=edge_eps)

        # 3-4. Adaptive strategy:
        # LCP≥17        → fast path; use only this LCP prefix as entry
        # LCP<17 and few rect prefixes → fast path; use minimal cover-rect prefix set
        # else             → fill path
        # mbr_cover_pip=True → force inscribed-level L tiled cover + PIP all candidates
        use_fast_path_lcp = (lcp_lv >= FAST_PATH_LCP_LEVEL) and bool(lcp)
        use_mbr_cover = mbr_cover_pip

        if use_mbr_cover:
            inscribed_lv = _compute_inscribed_grid_level(
                polygon_xy, lcp_lv, max_depth,
                safety_shrink=mbr_cover_safety_shrink,
                verify_corners=True)
            mbr_rect = _rect_at_level(
                polygon_xy, inscribed_lv, max_depth)
            cover_cells = _rect_to_min_prefixes(
                mbr_rect[0], mbr_rect[1], mbr_rect[2], mbr_rect[3],
                inscribed_lv, lcp)
            edge_eps = _edge_eps_for_level(inscribed_lv)
            target_lv = inscribed_lv
            rect = mbr_rect
            x_min, x_max, y_min, y_max = mbr_rect
            rect_w = x_max - x_min + 1
            rect_h = y_max - y_min + 1
            rect_area = rect_w * rect_h
        elif use_fast_path_lcp and not force_fast_min_prefix:
            cover_cells = [lcp]
        else:
            cover_cells = _rect_to_min_prefixes(
                x_min, x_max, y_min, y_max, target_lv, lcp)
        n_cover = len(cover_cells)

        if use_mbr_cover:
            use_fast_path_lcp = False
            use_fast_path = True
        elif force_fast_min_prefix:
            use_fast_path_lcp = False
            use_fast_path = True
        else:
            use_fast_path_lcp, use_fast_path = _decide_use_fast_path(
                lcp_lv=lcp_lv, has_lcp=bool(lcp), n_cover=n_cover)

        if use_fast_path:
            if use_mbr_cover:
                strategy_row = "mbr_cover_pip"
            elif force_fast_min_prefix:
                strategy_row = "fast_path_min_prefix_forced"
            else:
                strategy_row = "fast_path"
            # ── Fast path: single prefix (LCP) or minimal prefix set + PIP all ──
            t_fill = 0.0
            n_inside = 0
            n_boundary = n_cover
            n_refine = 0
            n_fill = n_cover

            t0 = _t()
            all_hits = _query_neo4j_by_cells(
                session, cover_cells, source)
            t_query = (_t() - t0) * 1000

            t0 = _t()
            if all_hits:
                all_xy = np.array(
                    [_code_to_xy(c, GRID_LEVEL) for c in all_hits],
                    dtype=np.int64)
                pip_mask = _pip_eval(all_xy)
                pip_hits = [all_hits[i]
                            for i in range(len(all_hits))
                            if pip_mask[i]]
            else:
                pip_hits = []
            t_pip = (_t() - t0) * 1000

            n_bnd_candidates = len(all_hits)
            n_pip_pass = len(pip_hits)
            n_pip_reject = n_bnd_candidates - n_pip_pass
            gr_set = set(pip_hits)

        else:
            strategy_row = ("polygon_fill_all_pip"
                            if all_candidates_pip else "polygon_fill")
            # ── Fill path: multi-res fill + one query split + boundary PIP ──
            t0 = _t()
            inside_pf, boundary_pf, n_inside, n_boundary, n_refine = \
                _polygon_fill(polygon_xy, target_lv, max_depth,
                              rect, lcp, refine_levels=refine_levels)
            n_fill = n_inside + n_boundary
            t_fill = (_t() - t0) * 1000

            t0 = _t()
            inside_set, boundary_only_set = _query_neo4j_split(
                session, inside_pf, boundary_pf, source)
            boundary_only = list(boundary_only_set)
            t_query = (_t() - t0) * 1000

            t0 = _t()
            if all_candidates_pip:
                all_candidates = list(inside_set | boundary_only_set)
                if all_candidates:
                    all_xy = np.array(
                        [_code_to_xy(c, GRID_LEVEL)
                         for c in all_candidates],
                        dtype=np.int64)
                    pip_mask = _pip_eval(all_xy)
                    pip_hits = [all_candidates[i]
                                for i in range(len(all_candidates))
                                if pip_mask[i]]
                else:
                    pip_hits = []
                n_bnd_candidates = len(all_candidates)
                n_pip_pass = len(pip_hits)
                n_pip_reject = n_bnd_candidates - n_pip_pass
                gr_set = set(pip_hits)
            else:
                if boundary_only:
                    bnd_xy = np.array(
                        [_code_to_xy(c, GRID_LEVEL)
                         for c in boundary_only],
                        dtype=np.int64)
                    pip_mask = _pip_eval(bnd_xy)
                    pip_hits = [boundary_only[i]
                                for i in range(len(boundary_only))
                                if pip_mask[i]]
                else:
                    pip_hits = []
                n_bnd_candidates = len(boundary_only)
                n_pip_pass = len(pip_hits)
                n_pip_reject = n_bnd_candidates - n_pip_pass
                gr_set = inside_set | set(pip_hits)
            t_pip = (_t() - t0) * 1000

        n_hits = len(gr_set)
        results = list(gr_set)

        t_gr_total = (t_fetch + t_vertex_xy + t_grid
                      + t_fill + t_query + t_pip)

        # PostGIS comparison
        pg_count = ""
        t_pg_val_raw = ""
        t_pg_filter = 0.0
        n_pg_filter_removed = 0
        t_strict_filter = 0.0
        n_strict_filter_removed = 0
        recall_v = ""
        precision_v = ""
        f1_v = ""
        match_flag = ""
        gr_only_count = ""
        pg_only_count = ""
        speedup_v = ""

        if (
            enable_final_strict_filter
            and lcp_lv <= FINAL_PG_FILTER_LCP_MAX
            and target_lv <= FINAL_PG_FILTER_TARGET_MAX
        ):
            strict_set, t_strict_filter = _strict_geosot_filter_candidates(
                gr_set, polygon_xy)
            n_strict_filter_removed = len(gr_set) - len(strict_set)
            gr_set = strict_set
            n_hits = len(gr_set)
            results = list(gr_set)
            t_gr_total += t_strict_filter

        if (
            enable_final_pg_filter
            and pg_conn and qt and tt
            and lcp_lv <= FINAL_PG_FILTER_LCP_MAX
            and target_lv <= FINAL_PG_FILTER_TARGET_MAX
        ):
            filtered_set, t_pg_filter = _postgis_filter_candidates(
                pg_conn, qcode, qt, tt, list(gr_set))
            n_pg_filter_removed = len(gr_set) - len(filtered_set)
            gr_set = filtered_set
            n_hits = len(gr_set)
            results = list(gr_set)
            t_gr_total += t_pg_filter

        if compare_pg and pg_conn and qt and tt:
            pg_results, t_pg_val = postgis_overlay_features(
                pg_conn, qcode, qt, tt)
            pg_set = set(pg_results)
            pg_count = len(pg_set)
            t_pg_val_raw = round(t_pg_val, 3)
            common = gr_set & pg_set
            gr_only = gr_set - pg_set
            pg_only = pg_set - gr_set
            gr_only_count = len(gr_only)
            pg_only_count = len(pg_only)
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
                speedup_v = round(t_pg_val / t_gr_total, 4)

        pm = pg_metrics.get(qcode, {})
        rows_out.append({
            "plan_idx": idx,
            "plan_code": qcode,
            "plan_vertices": len(vcodes),
            "pg_area_m2": pm.get("area_m2", ""),
            "pg_lon_span_deg": pm.get("lon_span", ""),
            "pg_lat_span_deg": pm.get("lat_span", ""),
            "lcp": lcp,
            "lcp_level": lcp_lv,
            "target_level": target_lv,
            "rect_width": rect_w,
            "rect_height": rect_h,
            "rect_area": rect_area,
            "refine_levels": n_refine,
            "finest_level": target_lv + n_refine,
            "n_fill_inside": n_inside,
            "n_fill_boundary": n_boundary,
            "n_fill_total": n_fill,
            "n_boundary_candidates": n_bnd_candidates,
            "n_pip_pass": n_pip_pass,
            "n_pip_reject": n_pip_reject,
            "strategy": strategy_row,
            "n_hits": n_hits,
            "gr_matched_codes": (";".join(sorted(gr_set))
                                 if gr_set else ""),
            "t_fetch_ms": round(t_fetch, 3),
            "t_vertex_xy_ms": round(t_vertex_xy, 4),
            "t_grid_ms": round(t_grid, 4),
            "t_fill_ms": round(t_fill, 4),
            "t_query_ms": round(t_query, 3),
            "t_pip_ms": round(t_pip, 4),
            "t_gr_total_ms": round(t_gr_total, 3),
            "gr_time_ms": round(t_gr_total, 3),
            "pg_time_ms": t_pg_val_raw,
            "t_pg_filter_ms": round(t_pg_filter, 3),
            "n_pg_filter_removed": n_pg_filter_removed,
            "t_strict_filter_ms": round(t_strict_filter, 3),
            "n_strict_filter_removed": n_strict_filter_removed,
            "pg_count": pg_count,
            "t_pg_ms": t_pg_val_raw,
            "speedup": speedup_v,
            "recall": recall_v,
            "precision": precision_v,
            "f1": f1_v,
            "match": match_flag,
            "gr_only_count": gr_only_count,
            "pg_only_count": pg_only_count,
        })

    sys.stdout.write(
        f"\r  [{'#' * 30}] 100.0%  {n:,}/{n:,}\n")
    sys.stdout.flush()

    if output_csv and rows_out:
        with open(output_csv, "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in rows_out:
                writer.writerow(
                    {k: r.get(k, "") for k in CSV_FIELDS})
        print(f"\n  CSV 已保存: {output_csv}")

    c_entries, c_codes = prefix_cache_stats()
    if c_entries > 0:
        print(f"\n  前缀缓存: {c_entries:,} 个前缀, "
              f"{c_codes:,} 个编码")

    # ================================================================
    # Statistics summary
    # ================================================================
    valid_rows = [r for r in rows_out
                  if r.get("plan_vertices", 0) >= 3]
    valid = len(valid_rows)
    print(f"\n{'=' * W}")
    print(f"  叠置查询 V7 批量测试汇总  (N={valid})")
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

    # ── Polygon stats ──
    vert_list = [r["plan_vertices"] for r in valid_rows]
    area_list = [r["pg_area_m2"] for r in valid_rows
                 if isinstance(r.get("pg_area_m2"), (int, float))]
    lon_span_list = [r["pg_lon_span_deg"] for r in valid_rows
                     if isinstance(r.get("pg_lon_span_deg"),
                                   (int, float))]
    lat_span_list = [r["pg_lat_span_deg"] for r in valid_rows
                     if isinstance(r.get("pg_lat_span_deg"),
                                   (int, float))]
    print(f'\n  -- Polygon statistics --')
    print(SEP)
    print(f"    顶点数:     {_stat3(vert_list)}")
    if area_list:
        print(f"    面积(m^2):  {_stat3(area_list)}")
    if lon_span_list:
        print(f"    经度跨度(°): {_stat3(lon_span_list)}")
    if lat_span_list:
        print(f"    纬度跨度(°): {_stat3(lat_span_list)}")

    # ── Grid stats ──
    tl_list = [r["target_level"] for r in valid_rows]
    ra_list = [r["rect_area"] for r in valid_rows]
    fi_list = [r["n_fill_inside"] for r in valid_rows
               if isinstance(r.get("n_fill_inside"), (int, float))]
    fb_list = [r["n_fill_boundary"] for r in valid_rows
               if isinstance(r.get("n_fill_boundary"), (int, float))]
    ft_list = [r["n_fill_total"] for r in valid_rows
               if isinstance(r.get("n_fill_total"), (int, float))]
    hits_list = [r["n_hits"] for r in valid_rows]

    print(f'\n  -- Z-order grid statistics --')
    print(SEP)
    print(f"    目标层级:   {_stat3(tl_list)}")
    print(f"    矩形面积:   {_stat3(ra_list)}")
    if fi_list:
        print(f"    inside 前缀: {_stat3(fi_list)}")
    if fb_list:
        print(f"    boundary前缀: {_stat3(fb_list)}")
    if ft_list:
        print(f"    填充前缀总: {_stat3(ft_list)}")
    print(f"    命中要素:   {_stat3(hits_list)}")

    # ── Step timings ──
    t_fetch_l = [r["t_fetch_ms"] for r in valid_rows]
    t_vxy_l = [r["t_vertex_xy_ms"] for r in valid_rows]
    t_grid_l = [r["t_grid_ms"] for r in valid_rows]
    t_fill_l = [r["t_fill_ms"] for r in valid_rows]
    t_query_l = [r["t_query_ms"] for r in valid_rows]
    t_pip_l = [r["t_pip_ms"] for r in valid_rows]
    t_total_l = [r["t_gr_total_ms"] for r in valid_rows]

    # ── Boundary PIP stats ──
    bnd_cand_l = [r["n_boundary_candidates"] for r in valid_rows
                  if isinstance(r.get("n_boundary_candidates"),
                                (int, float))]
    pip_pass_l = [r["n_pip_pass"] for r in valid_rows
                  if isinstance(r.get("n_pip_pass"), (int, float))]
    pip_rej_l = [r["n_pip_reject"] for r in valid_rows
                 if isinstance(r.get("n_pip_reject"), (int, float))]

    print(f'\n  -- Graph reasoning stage timings (ms) --')
    print(SEP)
    print(f"    1 获取折点编码:     {_stat5(t_fetch_l)}")
    print(f"    2 折点->(x,y):      {_stat5(t_vxy_l)}")
    print(f"    3 覆盖矩形:        {_stat5(t_grid_l)}")
    print(f"    4 多边形填充:       {_stat5(t_fill_l)}")
    print(f"    5 图匹配:           {_stat5(t_query_l)}")
    print(f"    6 边界PIP:          {_stat5(t_pip_l)}")
    print(f"    合计:               {_stat5(t_total_l)}")

    if bnd_cand_l:
        print(f'\n  -- Boundary PIP statistics --')
        print(SEP)
        print(f"    boundary候选: {_stat3(bnd_cand_l)}")
        if pip_pass_l:
            print(f"    PIP 通过:    {_stat3(pip_pass_l)}")
        if pip_rej_l:
            print(f"    PIP 排除:    {_stat3(pip_rej_l)}")

    mean_total = statistics.mean(t_total_l)
    if mean_total > 0:
        print(f"\n    各步骤平均占比:")
        for label, vals in [
            ("获取折点", t_fetch_l),
            ("折点xy ", t_vxy_l),
            ("覆盖矩形", t_grid_l),
            ("多边形填充", t_fill_l),
            ("图匹配  ", t_query_l),
            ("边界PIP ", t_pip_l),
        ]:
            pct_ = statistics.mean(vals) / mean_total * 100
            print(f"      {label}: {pct_:>6.2f}%")

    # ── PostGIS comparison ──
    if compare_pg:
        pg_rows = [r for r in valid_rows
                   if r.get("t_pg_ms", "") != ""]
        if pg_rows:
            t_pg_l = [r["t_pg_ms"] for r in pg_rows]
            pg_cnt_l = [r["pg_count"] for r in pg_rows]
            gr_h_l = [r["n_hits"] for r in pg_rows]

            print(f'\n  -- PostGIS timings (ms) --')
            print(SEP)
            print(f"    ST_Intersects:      {_stat5(t_pg_l)}")

            sp_l = [r["speedup"] for r in pg_rows
                    if isinstance(r.get("speedup"), float)]
            if sp_l:
                print(f'\n  -- Speedup: PostGIS / graph reasoning --')
                print(SEP)
                print(f"    {_stat5(sp_l)}")

            print(f'\n  -- Feature count comparison --')
            print(SEP)
            hdr = (f"    {'指标':>10s}  "
                   f"{'图推理':>12s}  {'PostGIS':>12s}")
            print(hdr)
            print(f"    {'Mean':>10s}  "
                  f"{statistics.mean(gr_h_l):>12.1f}  "
                  f"{statistics.mean(pg_cnt_l):>12.1f}")
            print(f"    {'Median':>10s}  "
                  f"{statistics.median(gr_h_l):>12.0f}  "
                  f"{statistics.median(pg_cnt_l):>12.0f}")
            print(f"    {'Sum':>10s}  "
                  f"{sum(gr_h_l):>12,d}  "
                  f"{sum(pg_cnt_l):>12,d}")

            rv = [r["recall"] for r in pg_rows
                  if isinstance(r.get("recall"), float)]
            pv = [r["precision"] for r in pg_rows
                  if isinstance(r.get("precision"), float)]
            fv = [r["f1"] for r in pg_rows
                  if isinstance(r.get("f1"), float)]

            print(f'\n  -- Accuracy against the PostGIS reference --')
            print(SEP)
            if rv:
                print(f"    召回率:  Mean="
                      f"{statistics.mean(rv):.4%}  "
                      f"Med={statistics.median(rv):.4%}  "
                      f"Min={min(rv):.4%}")
            if pv:
                print(f"    精确率:  Mean="
                      f"{statistics.mean(pv):.4%}  "
                      f"Med={statistics.median(pv):.4%}  "
                      f"Min={min(pv):.4%}")
            if fv:
                print(f"    F1:      Mean="
                      f"{statistics.mean(fv):.4%}  "
                      f"Med={statistics.median(fv):.4%}  "
                      f"Min={min(fv):.4%}")

            mr = [r for r in pg_rows
                  if r.get("match", "") != ""]
            if mr:
                nm = sum(1 for r in mr if r["match"] == 1)
                print(f"    完全一致: "
                      f"{nm}/{len(mr)} "
                      f"({nm/len(mr):.2%})")

    # ── Paper citation summary ──
    print(f'\n  -- Summary for paper reporting --')
    print(SEP)
    print(f"    查询总数:     {valid}")
    print("    查询策略:     adaptive (fast_path / polygon_fill)")
    print(f"    网格层级:     Level {GRID_LEVEL}")
    print(f"    max_cells:    {max_cells}")
    print(f"    Mean 总耗时:  "
          f"{statistics.mean(t_total_l):.2f} ms")
    print(f"    Med  总耗时:  "
          f"{statistics.median(t_total_l):.2f} ms")
    if compare_pg:
        pg2 = [r for r in valid_rows
               if r.get("t_pg_ms", "") != ""]
        if pg2:
            tp = [r["t_pg_ms"] for r in pg2]
            tg = [r["t_gr_total_ms"] for r in pg2]
            print(f"    Mean PostGIS:  "
                  f"{statistics.mean(tp):.2f} ms")
            mtg = statistics.mean(tg)
            if mtg > 0:
                print(f"    Mean 加速比:   "
                      f"{statistics.mean(tp)/mtg:.2f}x")
            for label, vals in [
                ("召回率", [r["recall"] for r in pg2
                          if isinstance(r.get("recall"), float)]),
                ("精确率", [r["precision"] for r in pg2
                          if isinstance(r.get("precision"), float)]),
                ("F1", [r["f1"] for r in pg2
                       if isinstance(r.get("f1"), float)]),
            ]:
                if vals:
                    print(f"    Mean {label}:   "
                          f"{statistics.mean(vals):.4%}")

    print(f"\n{'=' * W}")
    return rows_out


# ============================================================================
# Main flow
# ============================================================================

def _fetch_all_plan_codes(session, pg_conn, query_source):
    qt = SOURCE_TO_PG_TABLE.get(query_source, "")
    if pg_conn and qt:
        cur = pg_conn.cursor()
        cur.execute(f"SELECT main_code FROM {qt}")
        codes = [str(r[0]).strip().upper()
                 for r in cur.fetchall()
                 if r[0] and is_valid_g0_code(
                     str(r[0]).strip().upper())]
        cur.close()
        return codes

    code_rows = [dict(r) for r in session.run(
        "MATCH (c:GeoSOTCentroidCode)"
        "-[:FROM_SOURCE]->(:DataSource {name: $src})\n"
        "MATCH (c)-[:HAS_GEOMETRY_TYPE]->"
        "(:GeoFeatureType {type_name: 'Polygon'})\n"
        "RETURN c.main_code AS mc",
        src=query_source)]
    return [str(r["mc"]).strip().upper()
            for r in code_rows
            if is_valid_g0_code(str(r["mc"]).strip().upper())]


def main():
    single_code: Optional[str] = None
    codes_file = ""
    source = "POI"
    query_source = "Statutory Plan"
    batch_n = 0
    run_all = False
    seed = 42
    no_postgis = False
    force_fast_min_prefix = False
    final_pg_filter = False
    final_strict_filter = False
    all_candidates_pip = False
    mbr_cover_pip = False
    mbr_cover_safety_shrink = 0
    output_csv = ""
    strategy = "adaptive"
    max_cells = 256
    refine_levels = REFINE_LEVELS

    i = 0
    args = sys.argv[1:]
    while i < len(args):
        if args[i] == "--code" and i + 1 < len(args):
            single_code = args[i + 1]; i += 2
        elif args[i] == "--source" and i + 1 < len(args):
            source = args[i + 1]; i += 2
        elif args[i] == "--query-source" and i + 1 < len(args):
            query_source = args[i + 1]; i += 2
        elif args[i] == "--batch" and i + 1 < len(args):
            batch_n = int(args[i + 1]); i += 2
        elif args[i] == "--all":
            run_all = True; i += 1
        elif args[i] == "--codes-file" and i + 1 < len(args):
            codes_file = args[i + 1]; i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1]); i += 2
        elif args[i] == "--no-postgis":
            no_postgis = True; i += 1
        elif args[i] == "--force-fast-min-prefix":
            force_fast_min_prefix = True; i += 1
        elif args[i] == "--final-pg-filter":
            final_pg_filter = True; i += 1
        elif args[i] == "--final-strict-filter":
            final_strict_filter = True; i += 1
        elif args[i] == "--all-candidates-pip":
            all_candidates_pip = True; i += 1
        elif args[i] == "--mbr-cover-pip":
            mbr_cover_pip = True; i += 1
        elif (args[i] == "--mbr-cover-shrink"
              and i + 1 < len(args)):
            mbr_cover_safety_shrink = int(args[i + 1]); i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_csv = args[i + 1]; i += 2
        elif args[i] == "--max-cells" and i + 1 < len(args):
            max_cells = int(args[i + 1]); i += 2
        elif args[i] == "--refine" and i + 1 < len(args):
            refine_levels = int(args[i + 1]); i += 2
        else:
            i += 1

    compare_pg = not no_postgis

    if not single_code and batch_n <= 0 and not run_all and not codes_file:
        print("用法:")
        print("  python geosot_overlay_query_v7.py "
              "--code G0xxx --source POI")
        print("  python geosot_overlay_query_v7.py "
              "--batch 50 --source POI --seed 42")
        print("  python geosot_overlay_query_v7.py "
              "--all --source POI --output result.csv")
        print("  python geosot_overlay_query_v7.py "
              "--codes-file low_accuracy_rows.csv --source POI")
        print("  python geosot_overlay_query_v7.py "
              "--all --source POI --refine 4")
        print()
        print("参数:")
        print('  --code CODE          Query polygon main_code')
        print(f'  --source SRC         Target source (default: {source})')
        print(f'  --query-source SRC   Query feature source (default: {query_source})')
        print('  --batch N            Test N features')
        print('  --all                Process all polygon features')
        print('  --codes-file FILE    Run plan_code values listed in a file')
        print('  --output FILE        Output CSV path')
        print('  --no-postgis         Skip the PostGIS comparison')
        print('  --force-fast-min-prefix  Force the minimal-prefix fast path without filling')
        print('  --final-pg-filter    Apply final PostGIS filtering to difficult cases')
        print('  --final-strict-filter Apply strict GeoSOT secondary filtering to difficult cases')
        print('  --all-candidates-pip Apply PIP to every fill-path candidate')
        print('  --mbr-cover-pip      Tile at interior level L and apply PIP to every candidate')
        print('  --mbr-cover-shrink N Refine L by N levels (cell size / 2^N; default 0)')
        print(f'  --max-cells N        Maximum rectangle-cover cell count (default: {max_cells})')
        print(f'  --refine N           Boundary refinement levels (default: {refine_levels})')
        return

    mode_desc = ("单条查询" if single_code
                 else "按文件定向跑" if codes_file
                 else "全部遍历" if run_all
                 else f"随机采样 {batch_n} 个")

    W = 76
    print("=" * W)
    print("GeoSOT 叠置查询 V7 — 多分辨率多边形填充 (全链路离散网格)")
    print("=" * W)
    print(f"  模式:       {mode_desc}")
    print(f"  目标来源:   {source}")
    print(f"  查询来源:   {query_source}")
    print(f"  查询策略:   {strategy}")
    print(f"  邻域补偿:   {'开' if NEIGHBOR_PIP else '关'}")
    print(f"  低LCP小覆盖快速路径: "
          f"{'开' if LOW_LCP_ALLOW_SMALL_COVER_FASTPATH else '关'}")
    print(f"  强制最小前缀快速路径: "
          f"{'开' if force_fast_min_prefix else '关'}")
    print(f"  PostGIS最终过滤: "
          f"{'开' if final_pg_filter else '关'}")
    print(f"  严格二次过滤: "
          f"{'开' if final_strict_filter else '关'}")
    print(f"  全候选PIP:   {'开' if all_candidates_pip else '关'}")
    print(f"  MBR平铺PIP:  "
          f"{'开 (shrink=' + str(mbr_cover_safety_shrink) + ')' if mbr_cover_pip else '关'}")
    print(f"  网格层级:   Level {GRID_LEVEL}")
    print(f"  最大网格数: {max_cells}")
    print(f"  边界细分:   {refine_levels} 级")
    print(f"  PostGIS 对比: {'是' if compare_pg else '否'}")
    if output_csv:
        print(f"  输出文件:   {output_csv}")

    # [1] Connect databases
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
        print(f"  PostGIS: {PG_CONFIG['host']}:"
              f"{PG_CONFIG['port']}"
              f"/{PG_CONFIG['dbname']}  ({t_pg:.0f} ms)")

        # ---- fairness setup: PG warmup (pg_prewarm + ST_Intersects throwaway) ----
        # Methodology should disclose: PG GiST + B-tree indexes are pg_prewarm'ed
        # into shared_buffers, counterpart to GR PlanPreloaderV6 preloading polygon
        # vertices into RAM-resident memory.
        from bench_setup import prepare_pg_warm
        used_tables = sorted({
            tbl for tbl in (
                SOURCE_TO_PG_TABLE.get(source),
                SOURCE_TO_PG_TABLE.get(query_source),
            ) if tbl
        })
        qt = SOURCE_TO_PG_TABLE.get(query_source, "")
        tt = SOURCE_TO_PG_TABLE.get(source, "")
        throwaway_sqls = [
            (f"SELECT main_code FROM {tbl} LIMIT 50", None)
            for tbl in used_tables
        ]
        if qt and tt:
            throwaway_sqls.append((
                f"SELECT t.main_code FROM {tt} t, {qt} q "
                f"WHERE q.main_code = (SELECT main_code FROM {qt} LIMIT 1) "
                f"AND ST_Intersects(t.geometry, q.geometry) LIMIT 1",
                None,
            ))
        prepare_pg_warm(pg_conn, used_tables, throwaway_sqls)

    # [2] Query
    poi_cache = None

    if single_code:
        run_single_query(
            session, pg_conn,
            single_code, source, query_source,
            compare_pg, max_cells=max_cells,
            refine_levels=refine_levels)
    elif run_all or batch_n > 0 or codes_file:
        print(f"\n[2] 预加载 {query_source} 折点编码 ...")
        pp = PlanPreloaderV6(session, verbose=True)
        t0 = time.perf_counter()
        n_plans = pp.preload(query_source)
        t_pp = (time.perf_counter() - t0) * 1000
        print(f"    合计: {t_pp:.0f} ms, "
              f"{n_plans:,} 个 Plan 已就绪")

        # ---- fairness setup: GR-side main_code client index ----
        # Counterpart to PG pg_prewarm: pull all target-source main_codes
        # from Neo4j, sort locally; later prefix match uses bisect, not Neo4j.
        # Setup cost disclosed separately in Methodology.
        print()
        print("─" * 76)
        print("  [bench_setup] GR client-side main_code index "
              "(对应 PG pg_prewarm)")
        print("─" * 76)
        mci = MainCodeIndex(session, sources=[source], verbose=True)
        set_main_code_index(mci)
        # Also clear any leftover prefix_cache so throwaway re-warms it
        prefix_cache_clear()
        print("─" * 76)
        print()

        if codes_file:
            if not os.path.isfile(codes_file):
                raise FileNotFoundError(
                    f"codes file not found: {codes_file}")
            df_codes = pd.read_csv(codes_file, low_memory=False)
            if "plan_code" not in df_codes.columns:
                raise ValueError(
                    "codes file must contain 'plan_code' column")
            plan_codes_list = [
                str(c).strip().upper()
                for c in df_codes["plan_code"].dropna().tolist()
                if is_valid_g0_code(str(c).strip().upper())
            ]
            # Deduplicate while preserving order
            seen = set()
            dedup_codes = []
            for c in plan_codes_list:
                if c not in seen:
                    seen.add(c)
                    dedup_codes.append(c)
            plan_codes_list = dedup_codes
            print(f"  文件样本: {len(plan_codes_list)} 个 "
                  f"(from {codes_file})")
        elif run_all:
            plan_codes_list = pp.plan_codes
        else:
            random.seed(seed)
            plan_codes_list = random.sample(
                pp.plan_codes,
                min(batch_n, len(pp.plan_codes)))
            print(f"  采样: {len(plan_codes_list)} 个")
        print(f"  总数: {len(plan_codes_list):,}")

        # ---- fairness setup: dual-side throwaway ----
        # Same as distance experiments: run 5 real queries and discard; warm
        # GR _prefix_cache (Neo4j STARTS WITH candidate cache) +
        # PG plan cache (ST_Intersects shape) before official timing.
        n_throwaway = min(5, len(plan_codes_list))
        if n_throwaway > 0:
            print(f"\n[2.5] 双侧 throwaway: 丢弃前 {n_throwaway} 条 query "
                  f"(热 GR _prefix_cache / PG plan cache)")
            import contextlib
            import io
            tw_codes = plan_codes_list[:n_throwaway]
            for tw_qc in tw_codes:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        run_single_query(
                            session, pg_conn,
                            tw_qc, source, query_source,
                            compare_pg,
                            max_cells=max_cells,
                            refine_levels=refine_levels,
                            plan_preloader=pp)
                except Exception:
                    pass
            print(f"      throwaway 完成, 进入正式计时阶段")

        print(f"\n[3] 运行叠置查询 "
              f"({len(plan_codes_list):,} 次) ...")
        run_batch(
            session, pg_conn,
            plan_codes_list, source, query_source,
            compare_pg, strategy=strategy,
            max_cells=max_cells,
            refine_levels=refine_levels,
            force_fast_min_prefix=force_fast_min_prefix,
            enable_final_pg_filter=final_pg_filter,
            enable_final_strict_filter=final_strict_filter,
            all_candidates_pip=all_candidates_pip,
            mbr_cover_pip=mbr_cover_pip,
            mbr_cover_safety_shrink=mbr_cover_safety_shrink,
            output_csv=output_csv,
            plan_preloader=pp)

    session.close()
    driver.close()
    if pg_conn:
        pg_conn.close()
    print("\n完成。")


if __name__ == "__main__":
    main()

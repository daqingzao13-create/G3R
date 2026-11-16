#!/usr/bin/env python3
# coding: utf-8
"""
Overlay helpers using preloading, NumPy bbox filtering, and ray-casting PIP.

Given a polygon main_code, retrieve contained features from the target source. Setup fetches
target codes, decodes their grid centers, and builds coordinate arrays/main-code indexes. A
query fetches or reuses polygon vertices, filters candidates by vectorized bbox comparisons,
then applies ray-casting PIP; target features need no additional Neo4j round trips.

Usage:
    python geosot_overlay_query.py --code G0xxx --source POI
    python geosot_overlay_query.py --batch 50 --source POI --seed 42
    python geosot_overlay_query.py --all --source POI --output result.csv
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

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

SOURCE_TO_PG_TABLE: Dict[str, str] = {
    "POI": "public.poi_new",
    "Buildings": "public.building_new",
    "Road": "public.road",
    "Statutory Plan": "public.statutory_plan",
    "Community": "public.admin_community",
    "Subdistrict": "public.admin_subdistrict",
    "District": "public.admin_district",
}

POINT_SOURCES: Set[str] = {"POI"}


# ============================================================================
# Geometry helpers
# ============================================================================

def _point_in_polygon(
    px: float, py: float,
    polygon: List[Tuple[float, float]],
) -> bool:
    """
    Ray-casting point-in-polygon test.

    point x (longitude)
    point y (latitude)
    polygon vertices [(x, y), ...]
    whether inside
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _pip_batch(
    pts: np.ndarray,
    polygon: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Batch ray-casting PIP (NumPy vectorized).

    candidate point coordinates
    polygon vertices
    bool array (M,)
    """
    n_poly = len(polygon)
    if n_poly < 3 or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)

    px = pts[:, 0]
    py = pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)

    poly_arr = np.array(polygon, dtype=np.float64)
    for i in range(n_poly):
        j = (i + n_poly - 1) % n_poly
        xi, yi = poly_arr[i]
        xj, yj = poly_arr[j]

        cond = (yi > py) != (yj > py)
        if yj == yi:
            continue
        x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
        cross = px < x_intersect

        inside ^= (cond & cross)

    return inside


# ============================================================================
# Preloader: load all target source point coords once
# ============================================================================

class FeaturePreloader:
    """
    Preload feature coordinates for one source. For point features such as POI, decode main_code
    into (lon, lat) and store NumPy arrays.
    """

    def __init__(self, session, verbose: bool = True):
        """
        Neo4j session
        whether to print details
        """
        self._session = session
        self._verbose = verbose

        self._codes: List[str] = []
        self._code_ids: List[str] = []
        self._coords: Optional[np.ndarray] = None
        self._loaded = False
        self._source = ""

    @property
    def count(self) -> int:
        return len(self._codes)

    def preload(self, source: str) -> int:
        """
        Load and decode all features from a source.

        @param source Data source name, for example POI.
        @returns Number of loaded features.
        """
        self._source = source
        _t = time.perf_counter

        # query all main_codes from Neo4j
        t0 = _t()
        rows = [dict(r) for r in self._session.run(
            "MATCH (c:GeoSOTCode)"
            "-[:FROM_SOURCE]->(:Source {name: $src})\n"
            "RETURN c.main_code AS mc, "
            "c.geosot_code_id AS cid",
            src=source,
        )]
        t_query = (_t() - t0) * 1000

        if self._verbose:
            print(f"    Neo4j 查询:  {t_query:.0f} ms  "
                  f"({len(rows):,} 条记录)")

        # decode coordinates
        t0 = _t()
        codes: List[str] = []
        code_ids: List[str] = []
        coords: List[Tuple[float, float]] = []
        fail_count = 0

        for r in rows:
            mc = str(r["mc"]).strip().upper()
            cid = str(r["cid"])
            if not is_valid_g0_code(mc):
                fail_count += 1
                continue
            try:
                lon, lat = fast_code_to_center(mc)
                codes.append(mc)
                code_ids.append(cid)
                coords.append((lon, lat))
            except Exception:
                fail_count += 1

        self._codes = codes
        self._code_ids = code_ids
        if coords:
            self._coords = np.array(coords, dtype=np.float64)
        else:
            self._coords = np.empty((0, 2), dtype=np.float64)
        self._loaded = True

        t_decode = (_t() - t0) * 1000

        if self._verbose:
            print(f"    解码坐标:    {t_decode:.0f} ms  "
                  f"({len(codes):,} 个有效, "
                  f"{fail_count} 失败)")
            mem_mb = self._coords.nbytes / 1024 / 1024
            print(f"    内存占用:    {mem_mb:.1f} MB")

        return len(codes)

    def bbox_filter(
        self, min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
    ) -> np.ndarray:
        """
        Vectorized NumPy bbox filter.

        @param min_lon Minimum longitude.
        @param min_lat Minimum latitude.
        @param max_lon Maximum longitude.
        @param max_lat Maximum latitude.
        @returns Candidate indexes into self._coords.
        """
        c = self._coords
        mask = ((c[:, 0] >= min_lon) & (c[:, 0] <= max_lon) &
                (c[:, 1] >= min_lat) & (c[:, 1] <= max_lat))
        return np.nonzero(mask)[0]

    def pip_filter(
        self, idxs: np.ndarray,
        polygon: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Apply ray-casting PIP to bbox candidates.

        @param idxs Indexes returned by bbox_filter.
        @param polygon Polygon vertices.
        @returns Indexes of contained points.
        """
        if len(idxs) == 0:
            return idxs
        pts = self._coords[idxs]
        hit = _pip_batch(pts, polygon)
        return idxs[hit]

    def get_results(self, idxs: np.ndarray) -> List[Dict]:
        """
        final hit indices
        @returns [{main_code, code_id}, ...]
        """
        return [{"main_code": self._codes[i],
                 "code_id": self._code_ids[i]}
                for i in idxs]


# ============================================================================
# Fetch query polygon
# ============================================================================

def fetch_polygon_from_neo4j(
    session, main_code: str,
) -> Tuple[Optional[str], List[str], List[Tuple[float, float]]]:
    """
    Fetch polygon vertex codes from Neo4j and decode their coordinates.

    @param session Neo4j session.
    @param main_code Polygon main code.
    @returns (code_id, vertex_codes, polygon_coords).
    """
    rows = [dict(r) for r in session.run(
        "MATCH (p:GeoSOTCode {main_code: $mc})\n"
        "OPTIONAL MATCH (p)-[hv:HAS_VERTEX]->"
        "(v:GeoSOTVertexCode)\n"
        "WITH p, v, hv ORDER BY hv.vertex_index\n"
        "WITH p.geosot_code_id AS code_id,\n"
        "     collect(CASE WHEN v IS NOT NULL\n"
        "             THEN v.vertex_code "
        "ELSE null END) AS raw_verts\n"
        "RETURN code_id,\n"
        "       [x IN raw_verts WHERE x IS NOT NULL]"
        " AS vertex_codes",
        mc=main_code.strip().upper())]

    if not rows:
        return None, [], []

    r = rows[0]
    cid = str(r["code_id"]) if r["code_id"] else ""
    vcodes = [str(v).strip().upper() for v in r["vertex_codes"]]

    coords = []
    for vc in vcodes:
        try:
            lon, lat = fast_code_to_center(vc)
            coords.append((lon, lat))
        except Exception:
            pass

    return cid, vcodes, coords


def postgis_overlay_features(
    pg_conn, query_main_code: str,
    query_table: str, target_table: str,
) -> Tuple[List[str], float]:
    """
    Find overlay features via PostGIS ST_Intersects.

    PostGIS connection
    query feature main_code
    query feature table name
    target table name
    @returns (main_code_list, time_ms)
    """
    sql = (
        f"SELECT t.main_code "
        f"FROM {target_table} t, {query_table} q "
        f"WHERE q.main_code = %s "
        f"AND ST_Intersects(t.geometry, q.geometry)"
    )
    t0 = time.perf_counter()
    cur = pg_conn.cursor()
    cur.execute(sql, (query_main_code,))
    results = [str(r[0]).strip().upper()
               for r in cur.fetchall() if r[0]]
    cur.close()
    t_ms = (time.perf_counter() - t0) * 1000
    return results, t_ms


# ============================================================================
# Single query
# ============================================================================

def run_single_query(
    session, pg_conn,
    preloader: FeaturePreloader,
    query_code: str,
    source: str,
    query_source: str,
    compare_pg: bool,
):
    """
    Neo4j session
    PostGIS connection
    preloaded feature data
    query polygon main_code
    target source
    query feature source
    whether to compare with PostGIS
    """
    W = 76
    SEP = f"  {'─' * (W - 4)}"
    _t = time.perf_counter
    qcode = query_code.strip().upper()

    print(f"\n{'=' * W}")
    print("GeoSOT 叠置查询 — 预加载 + bbox + 射线法")
    print(f"{'=' * W}")
    print(f"  查询编码:   {qcode}")
    print(f"  查询来源:   {query_source}")
    print(f"  目标来源:   {source}")
    print(f"  预加载要素: {preloader.count:,}")

    # fetch polygon
    print(f"\n{SEP}")
    print(f"  Step 1: 获取多边形顶点")
    print(f"{SEP}")

    t0 = _t()
    cid, vcodes, polygon = fetch_polygon_from_neo4j(session, qcode)
    t_fetch = (_t() - t0) * 1000

    if len(polygon) < 3:
        print(f"  [ERROR] 顶点不足 ({len(polygon)} 个)")
        return

    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    print(f"  顶点数:     {len(polygon)}")
    print(f"  经度范围:   [{min_lon:.8f}, {max_lon:.8f}]")
    print(f"  纬度范围:   [{min_lat:.8f}, {max_lat:.8f}]")
    print(f"  耗时:       {t_fetch:.1f} ms")

    # NumPy bbox filter
    print(f"\n{SEP}")
    print(f"  Step 2: NumPy bbox 快速过滤")
    print(f"{SEP}")

    t0 = _t()
    bbox_idxs = preloader.bbox_filter(
        min_lon, min_lat, max_lon, max_lat)
    t_bbox = (_t() - t0) * 1000

    print(f"  bbox 候选:  {len(bbox_idxs):,} / "
          f"{preloader.count:,}")
    print(f"  耗时:       {t_bbox:.3f} ms")

    # ray-casting PIP
    print(f"\n{SEP}")
    print(f"  Step 3: 射线法精确判断 (PIP)")
    print(f"{SEP}")

    t0 = _t()
    hit_idxs = preloader.pip_filter(bbox_idxs, polygon)
    t_pip = (_t() - t0) * 1000

    results = preloader.get_results(hit_idxs)
    print(f"  命中要素:   {len(results)}")
    print(f"  耗时:       {t_pip:.3f} ms")

    t_total = t_fetch + t_bbox + t_pip
    print(f"\n  图推理总耗时: {t_total:.1f} ms")
    print(f"    ① 获取多边形:     {t_fetch:>8.1f} ms")
    print(f"    ② bbox 过滤:      {t_bbox:>8.3f} ms  "
          f"({len(bbox_idxs)} 候选)")
    print(f"    ③ PIP 精确判断:   {t_pip:>8.3f} ms  "
          f"({len(results)} 命中)")

    # PostGIS comparison
    if compare_pg and pg_conn:
        print(f"\n{SEP}")
        print(f"  Step 4: PostGIS ST_Intersects")
        print(f"{SEP}")

        qt = SOURCE_TO_PG_TABLE.get(query_source, "")
        tt = SOURCE_TO_PG_TABLE.get(source, "")
        if qt and tt:
            pg_results, t_pg = postgis_overlay_features(
                pg_conn, qcode, qt, tt)
            print(f"  找到要素:   {len(pg_results)} 个")
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
            if pg_set:
                print(f"  召回率:    "
                      f"{len(common) / len(pg_set):.1%}  "
                      f"({len(common)}/{len(pg_set)})")
            if gr_set:
                print(f"  精确率:    "
                      f"{len(common) / len(gr_set):.1%}  "
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
        for i, r in enumerate(results[:show_n]):
            mc = r["main_code"]
            try:
                lon, lat = fast_code_to_center(mc)
                print(f"  {i + 1:>4d}. {mc}  "
                      f"({lon:.6f}, {lat:.6f})")
            except Exception:
                print(f"  {i + 1:>4d}. {mc}")
        if len(results) > show_n:
            print(f"  ... 还有 {len(results) - show_n} 个")

    print(f"\n{'=' * W}")


# ============================================================================
# Batch test
# ============================================================================

CSV_FIELDS = [
    "plan_idx", "plan_code", "plan_vertices",
    "bbox_candidates", "pip_hits",
    "gr_matched_codes",
    "t_fetch_ms", "t_bbox_ms", "t_pip_ms", "t_total_ms",
    "pg_count", "t_pg_ms",
    "recall", "precision",
    "match",
    "gr_only_count", "pg_only_count",
]


def run_batch(
    session, pg_conn,
    preloader: FeaturePreloader,
    plan_codes: List[str],
    source: str,
    query_source: str,
    compare_pg: bool,
    output_csv: str = "",
) -> List[Dict]:
    """
    Neo4j session
    PostGIS connection
    preloaded feature data
    list of query polygon codes
    target source
    query feature source
    whether to compare with PostGIS
    CSV output path
    per-row result list
    """
    W = 76
    SEP = f"  {'─' * (W - 4)}"
    _t = time.perf_counter
    n = len(plan_codes)

    qt = SOURCE_TO_PG_TABLE.get(query_source, "")
    tt = SOURCE_TO_PG_TABLE.get(source, "")

    rows_out: List[Dict] = []
    t_start = _t()
    bar_update = max(n // 100, 1)

    for idx, qc in enumerate(plan_codes):
        if idx % bar_update == 0:
            elapsed = _t() - t_start
            pct = idx / n if n else 0
            rate = idx / elapsed if elapsed > 0.001 else 0
            eta = (n - idx) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r  [{int(30 * pct) * '█'}"
                f"{(30 - int(30 * pct)) * '░'}] "
                f"{pct * 100:5.1f}%  {idx:,}/{n:,}  "
                f"{rate:,.1f}/s  ETA {eta:.0f}s")
            sys.stdout.flush()

        qcode = qc.strip().upper()

        # 1. Fetch polygon
        t0 = _t()
        cid, vcodes, polygon = fetch_polygon_from_neo4j(
            session, qcode)
        t_fetch = (_t() - t0) * 1000

        if len(polygon) < 3:
            rows_out.append({
                "plan_idx": idx, "plan_code": qcode,
                "plan_vertices": len(polygon),
                "bbox_candidates": 0, "pip_hits": 0,
                "gr_matched_codes": "",
                "t_fetch_ms": round(t_fetch, 3),
                "t_bbox_ms": 0, "t_pip_ms": 0,
                "t_total_ms": round(t_fetch, 3),
                "pg_count": "", "t_pg_ms": "",
                "recall": "", "precision": "",
                "match": "",
                "gr_only_count": "", "pg_only_count": "",
            })
            continue

        lons = [p[0] for p in polygon]
        lats = [p[1] for p in polygon]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # 2. Bbox filter
        t0 = _t()
        bbox_idxs = preloader.bbox_filter(
            min_lon, min_lat, max_lon, max_lat)
        t_bbox = (_t() - t0) * 1000

        # 3. PIP
        t0 = _t()
        hit_idxs = preloader.pip_filter(bbox_idxs, polygon)
        t_pip = (_t() - t0) * 1000

        results = preloader.get_results(hit_idxs)
        t_total = t_fetch + t_bbox + t_pip
        gr_set = {r["main_code"] for r in results}

        # PostGIS
        pg_count = ""
        t_pg = ""
        recall = ""
        precision = ""
        match_flag = ""
        gr_only_count = ""
        pg_only_count = ""

        if compare_pg and pg_conn and qt and tt:
            pg_results, t_pg_val = postgis_overlay_features(
                pg_conn, qcode, qt, tt)
            pg_set = set(pg_results)
            pg_count = len(pg_set)
            t_pg = round(t_pg_val, 3)
            common = gr_set & pg_set
            gr_only = gr_set - pg_set
            pg_only = pg_set - gr_set
            gr_only_count = len(gr_only)
            pg_only_count = len(pg_only)
            recall = (round(len(common) / len(pg_set), 4)
                      if pg_set else "")
            precision = (round(len(common) / len(gr_set), 4)
                         if gr_set else "")
            match_flag = 1 if gr_set == pg_set else 0

        gr_codes_str = ";".join(sorted(gr_set)) if gr_set else ""

        rows_out.append({
            "plan_idx": idx,
            "plan_code": qcode,
            "plan_vertices": len(polygon),
            "bbox_candidates": len(bbox_idxs),
            "pip_hits": len(results),
            "gr_matched_codes": gr_codes_str,
            "t_fetch_ms": round(t_fetch, 3),
            "t_bbox_ms": round(t_bbox, 3),
            "t_pip_ms": round(t_pip, 3),
            "t_total_ms": round(t_total, 3),
            "pg_count": pg_count,
            "t_pg_ms": t_pg,
            "recall": recall,
            "precision": precision,
            "match": match_flag,
            "gr_only_count": gr_only_count,
            "pg_only_count": pg_only_count,
        })

    # final progress bar
    sys.stdout.write(
        f"\r  [{'█' * 30}] 100.0%  {n:,}/{n:,}\n")
    sys.stdout.flush()

    # CSV output
    if output_csv and rows_out:
        with open(output_csv, "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in rows_out:
                writer.writerow(
                    {k: r.get(k, "") for k in CSV_FIELDS})
        print(f"\n  CSV 已保存: {output_csv}")

    # ================================================================
    # Statistics summary
    # ================================================================
    valid_rows = [r for r in rows_out
                  if r.get("plan_vertices", 0) >= 3]
    valid = len(valid_rows)
    print(f"\n{'=' * W}")
    print(f"  叠置查询批量测试汇总  (N={valid})")
    print(f"{'=' * W}")

    if valid == 0:
        print("  无有效结果。")
        return rows_out

    def _stat(vals):
        return (f"Mean={statistics.mean(vals):>10.3f}  "
                f"Med={statistics.median(vals):>10.3f}  "
                f"Max={max(vals):>10.3f}")

    t_fetch_list = [r["t_fetch_ms"] for r in valid_rows]
    t_bbox_list = [r["t_bbox_ms"] for r in valid_rows]
    t_pip_list = [r["t_pip_ms"] for r in valid_rows]
    t_total_list = [r["t_total_ms"] for r in valid_rows]
    bbox_cand_list = [r["bbox_candidates"] for r in valid_rows]
    pip_hits_list = [r["pip_hits"] for r in valid_rows]

    print(f"\n  图推理分步耗时 (ms):")
    print(SEP)
    print(f"    ① 获取多边形:     {_stat(t_fetch_list)}")
    print(f"    ② bbox 过滤:      {_stat(t_bbox_list)}")
    print(f"    ③ PIP 精确判断:   {_stat(t_pip_list)}")
    print(f"    合计:             {_stat(t_total_list)}")

    print(f"\n  bbox 候选数:")
    print(f"    Mean={statistics.mean(bbox_cand_list):.1f}  "
          f"Med={statistics.median(bbox_cand_list):.0f}  "
          f"Max={max(bbox_cand_list)}")

    print(f"\n  PIP 命中数:")
    print(f"    Mean={statistics.mean(pip_hits_list):.1f}  "
          f"Med={statistics.median(pip_hits_list):.0f}  "
          f"Max={max(pip_hits_list)}")

    if compare_pg:
        pg_rows = [r for r in valid_rows
                   if r.get("t_pg_ms", "") != ""]
        if pg_rows:
            t_pg_list = [r["t_pg_ms"] for r in pg_rows]
            pg_count_list = [r["pg_count"] for r in pg_rows]
            print(f"\n  PostGIS 耗时 (ms):")
            print(f"    {_stat(t_pg_list)}")

            print(f"\n  PostGIS 命中数:")
            print(f"    Mean={statistics.mean(pg_count_list):.1f}  "
                  f"Med={statistics.median(pg_count_list):.0f}  "
                  f"Max={max(pg_count_list)}")

            recall_vals = [r["recall"] for r in pg_rows
                           if r.get("recall", "") != ""]
            precision_vals = [r["precision"] for r in pg_rows
                              if r.get("precision", "") != ""]
            if recall_vals:
                print(f"\n  召回率 (vs PostGIS):")
                print(f"    Mean={statistics.mean(recall_vals):.1%}  "
                      f"Med={statistics.median(recall_vals):.1%}")
            if precision_vals:
                print(f"  精确率:")
                print(f"    Mean="
                      f"{statistics.mean(precision_vals):.1%}  "
                      f"Med="
                      f"{statistics.median(precision_vals):.1%}")

            match_rows = [r for r in pg_rows
                          if r.get("match", "") != ""]
            if match_rows:
                n_match = sum(1 for r in match_rows
                              if r["match"] == 1)
                print(f"\n  完全一致: {n_match}/{len(match_rows)} "
                      f"({n_match / len(match_rows):.1%})")

            if statistics.mean(t_total_list) > 0:
                speedup = (statistics.mean(t_pg_list)
                           / statistics.mean(t_total_list))
                print(f"\n  加速比 (PostGIS / Graph): "
                      f"{speedup:.2f}x")

    print(f"\n{'=' * W}")
    return rows_out


# ============================================================================
# Main flow
# ============================================================================

def _fetch_all_plan_codes(session, pg_conn, query_source: str):
    """
    Get all polygon feature codes for a source
    Neo4j session
    PostGIS connection (may be None)
    query feature source name
    list of valid codes
    """
    qt = SOURCE_TO_PG_TABLE.get(query_source, "")
    if pg_conn and qt:
        cur = pg_conn.cursor()
        cur.execute(f"SELECT main_code FROM {qt}")
        codes = [
            str(r[0]).strip().upper()
            for r in cur.fetchall()
            if r[0] and is_valid_g0_code(
                str(r[0]).strip().upper())]
        cur.close()
        return codes

    code_rows = [dict(r) for r in session.run(
        "MATCH (p:GeoSOTCode)"
        "-[:FROM_SOURCE]->"
        "(:Source {name: $src})\n"
        "MATCH (p)-[:HAS_GEOMETRY_TYPE]->"
        "(:GeoFeatureType {type_name: 'Polygon'})\n"
        "RETURN p.main_code AS mc",
        src=query_source)]
    return [
        str(r["mc"]).strip().upper()
        for r in code_rows
        if is_valid_g0_code(str(r["mc"]).strip().upper())]


def main():
    single_code: Optional[str] = None
    source = "POI"
    query_source = "Statutory Plan"
    batch_n = 0
    run_all = False
    seed = 42
    no_postgis = False
    output_csv = ""

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
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1]); i += 2
        elif args[i] == "--no-postgis":
            no_postgis = True; i += 1
        elif args[i] == "--output" and i + 1 < len(args):
            output_csv = args[i + 1]; i += 2
        else:
            i += 1

    compare_pg = not no_postgis

    if not single_code and batch_n <= 0 and not run_all:
        print("用法:")
        print("  python geosot_overlay_query.py "
              "--code G0xxx --source POI")
        print("  python geosot_overlay_query.py "
              "--batch 50 --source POI --seed 42")
        print("  python geosot_overlay_query.py "
              "--all --source POI --output result.csv")
        print()
        print("参数:")
        print('  --code CODE        Query polygon main_code')
        print('  --source SRC       Target source (POI, Buildings, Road, Statutory Plan)')
        print(f'  --query-source SRC Query feature source (default: {query_source})')
        print('  --batch N          Test N polygon features')
        print('  --all              Process all polygon features')
        print('  --output FILE      Output CSV path')
        print('  --no-postgis       Skip the PostGIS comparison')
        return

    mode_desc = ("单条查询" if single_code
                 else "全部遍历" if run_all
                 else f"随机采样 {batch_n} 个")

    W = 76
    print("=" * W)
    print("GeoSOT 叠置查询 — 预加载 + bbox + 射线法 (方案 A)")
    print("=" * W)
    print(f"  模式:       {mode_desc}")
    print(f"  目标来源:   {source}")
    print(f"  查询来源:   {query_source}")
    print(f"  PostGIS 对比: {'是' if compare_pg else '否'}")
    if output_csv:
        print(f"  输出文件:   {output_csv}")

    # connect
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
        print(f"  PostGIS: {PG_CONFIG['host']}:{PG_CONFIG['port']}/"
              f"{PG_CONFIG['dbname']}  ({t_pg:.0f} ms)")

    # preload target features
    print(f"\n[2] 预加载 {source} 要素 ...")
    preloader = FeaturePreloader(session, verbose=True)
    t0 = time.perf_counter()
    n_features = preloader.preload(source)
    t_preload = (time.perf_counter() - t0) * 1000
    print(f"    合计: {t_preload:.0f} ms, "
          f"{n_features:,} 个要素已就绪")

    # query
    if single_code:
        run_single_query(
            session, pg_conn, preloader,
            single_code, source, query_source,
            compare_pg)
    elif run_all or batch_n > 0:
        print(f"\n[3] 获取 {query_source} 面状要素列表 ...")
        all_codes = _fetch_all_plan_codes(
            session, pg_conn, query_source)
        print(f"  总数: {len(all_codes):,}")

        if run_all:
            plan_codes = all_codes
        else:
            random.seed(seed)
            plan_codes = random.sample(
                all_codes, min(batch_n, len(all_codes)))
            print(f"  采样: {len(plan_codes)} 个")

        print(f"\n[4] 运行叠置查询 "
              f"({len(plan_codes):,} 次) ...")
        run_batch(
            session, pg_conn, preloader,
            plan_codes, source, query_source,
            compare_pg, output_csv=output_csv)

    session.close()
    driver.close()
    if pg_conn:
        pg_conn.close()
    print("\n完成。")


if __name__ == "__main__":
    main()

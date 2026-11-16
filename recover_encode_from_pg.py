#!/usr/bin/env python3
# coding: utf-8
"""
Spatial graph preparation, step 1/3: encode PostGIS points, lines, and polygons as centroid
codes (CC), ordered vertex codes (VC), and longest common prefix codes (LCPC).

Sources: public.poi_new -> POI/Point; public.road -> Road/LineString; public.statutory_plan
-> Statutory Plan/Parcel/Polygon; public.building_new -> Buildings/Building/Polygon.

The kernel matches gbt40087_encoder.py: level 32, EPSG:4490, northeastern-hemisphere G0
quaternary codes. Input geometries must already use geographic SRID 4490; they are read with
ST_AsGeoJSON without reprojection. A point has CC equal to its point code and VC=[CC].
Lines/polygons use the centroid CC and ordered vertex codes; LCPC is the LCP of CC plus all
vertices, generated only for features with more than one vertex.

Each source produces recover_out/<table>_encoded.csv with feature_id, source, feature_label,
geometry_type, main_code, cc_lon, cc_lat, level, vertex_count, lcpc_code, lcpc_level,
vertex_codes_json, and props_json. Review exported business properties before sharing data.

Usage:
    python recover_encode_from_pg.py
    python recover_encode_from_pg.py --tables poi_new
    python recover_encode_from_pg.py --limit 2000
    python recover_encode_from_pg.py --out-dir recover_out
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("需要 psycopg2: pip install psycopg2-binary")
    sys.exit(1)

try:
    from shapely.geometry import shape as shp_shape
except ImportError:
    print("需要 shapely: pip install shapely")
    sys.exit(1)

import math

from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from tqdm import tqdm


# ======================================================================
# Encoding kernel identical to gbt40087_encoder.py; inlined to avoid a GeoPandas dependency.
# ======================================================================
CHINA_LON_MIN, CHINA_LON_MAX = 73.0, 136.0
CHINA_LAT_MIN, CHINA_LAT_MAX = 3.0, 54.0


def _to_dms_parts(value: float) -> Tuple[int, int, int, float]:
    if value < 0:
        raise ValueError("仅支持东北半球 (经纬度必须为非负)")
    degree = int(math.floor(value))
    minutes_float = (value - degree) * 60.0
    minute = int(math.floor(minutes_float))
    seconds_float = (minutes_float - minute) * 60.0
    second = int(math.floor(seconds_float))
    return degree, minute, second, seconds_float - second


def _to_fixed_bits(value: int, width: int) -> str:
    if value < 0 or value >= (1 << width):
        raise ValueError(f"数值 {value} 超出 {width} bit 范围")
    return format(value, f"0{width}b")


def _coord_to_31_bits(value: float) -> str:
    degree, minute, second, sub_second = _to_dms_parts(value)
    if degree > 255:
        raise ValueError("度值超出 8bit 上限")
    if minute > 59 or second > 59:
        raise ValueError("分/秒值非法")
    sub_q = int(sub_second * 2048.0)
    if sub_q > 2047:
        sub_q = 2047
    return (
        _to_fixed_bits(degree, 8)
        + _to_fixed_bits(minute, 6)
        + _to_fixed_bits(second, 6)
        + _to_fixed_bits(sub_q, 11)
    )


def _interleave(lat_bits: str, lon_bits: str) -> str:
    out: List[str] = []
    for i in range(31):
        pair = lat_bits[i] + lon_bits[i]
        out.append({"00": "0", "01": "1", "10": "2", "11": "3"}[pair])
    return "".join(out)


def encode_gbt40087_ne(lon: float, lat: float, level: int = 32) -> str:
    if level < 0 or level > 32:
        raise ValueError("level 必须在 0~32")
    if level == 0:
        return "G"
    if level == 1:
        return "G0"
    if lon < 0 or lat < 0:
        raise ValueError("仅支持东北半球")
    if lon > 180 or lat > 90:
        raise ValueError("经纬度超出地理合法范围")
    q31 = _interleave(_coord_to_31_bits(lat), _coord_to_31_bits(lon))
    return "G0" + q31[: level - 1]


def _check_china_range(lon: float, lat: float, strict: bool) -> None:
    if not strict:
        return
    if not (CHINA_LON_MIN <= lon <= CHINA_LON_MAX and CHINA_LAT_MIN <= lat <= CHINA_LAT_MAX):
        raise ValueError(f"坐标超出中国范围: ({lon}, {lat})")


def _iter_vertices(geom):
    if isinstance(geom, Point):
        yield geom.x, geom.y
        return
    if isinstance(geom, LineString):
        for c in geom.coords:
            yield c[0], c[1]
        return
    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        for c in coords:
            yield c[0], c[1]
        return
    raise ValueError(f"不支持的几何类型: {type(geom)}")


def _explode_geometries(geom, base_name: str):
    if isinstance(geom, (Point, LineString, Polygon)):
        return [(base_name, geom)]
    if isinstance(geom, (MultiPoint, MultiLineString, MultiPolygon)):
        return [(f"{base_name}_part{i}", g) for i, g in enumerate(geom.geoms)]
    return []


PG_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "dbname": "postgis_35_sample",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", ""),
}

# Table -> (schema, source name, feature label).
TABLE_MAP: Dict[str, Tuple[str, str, str]] = {
    "poi_new": ("public", "POI", "POI"),
    "road": ("public", "Road", "Road"),
    "statutory_plan": ("public", "Statutory Plan", "Parcel"),
    "building_new": ("public", "Buildings", "Building"),
}

# Internal encoding columns are recomputed and excluded from business properties.
SKIP_PROP_COLS = {"enc_level", "vtx_count"}

CSV_FIELDS = [
    "feature_id",
    "source",
    "feature_label",
    "geometry_type",
    "main_code",
    "cc_lon",
    "cc_lat",
    "level",
    "vertex_count",
    "lcpc_code",
    "lcpc_level",
    "vertex_codes_json",
    "props_json",
]


def longest_common_prefix(strings: List[str]) -> str:
    """
    @param strings Encoded strings.
    @returns Their longest common prefix.
    """
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for i, ch in enumerate(shortest):
        for s in strings:
            if s[i] != ch:
                return shortest[:i]
    return shortest


def discover_columns(conn, schema: str, table: str) -> Tuple[List[str], str]:
    """@returns (non-geometry business column names, geometry column name)."""
    sql = """
        SELECT column_name, udt_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (schema, table))
        rows = cur.fetchall()
    geom_col = None
    biz_cols: List[str] = []
    for r in rows:
        if r["udt_name"] == "geometry":
            geom_col = r["column_name"]
        else:
            biz_cols.append(r["column_name"])
    if geom_col is None:
        raise ValueError(f"{schema}.{table} 未找到 geometry 列")
    return biz_cols, geom_col


def count_rows(conn, schema: str, table: str, geom_col: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{schema}"."{table}" WHERE "{geom_col}" IS NOT NULL')
        return int(cur.fetchone()[0])


def encode_table(
    conn,
    table: str,
    out_dir: str,
    level: int,
    strict_china: bool,
    limit: Optional[int],
    batch_fetch: int,
) -> str:
    """
    Encode one source table and export a CSV.

    @returns Output CSV path.
    """
    schema, source_name, feature_label = TABLE_MAP[table]
    biz_cols, geom_col = discover_columns(conn, schema, table)

    total = count_rows(conn, schema, table, geom_col)
    if limit:
        total = min(total, limit)

    # Quote column names to preserve mixed case; export geometry with ST_AsGeoJSON.
    quoted_biz = ", ".join(f'"{c}"' for c in biz_cols)
    geo_expr = f'ST_AsGeoJSON("{geom_col}", 12) AS __geojson'
    select_cols = (quoted_biz + ", " if quoted_biz else "") + geo_expr
    limit_clause = f" LIMIT {int(limit)}" if limit else ""
    sql = f'SELECT {select_cols} FROM "{schema}"."{table}" WHERE "{geom_col}" IS NOT NULL{limit_clause}'

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{table}_encoded.csv")

    # Increase the CSV field limit for long vertex_codes_json values.
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10 ** 7)

    n_features = 0
    n_skipped = 0
    seq_counter: Dict[str, int] = {}

    # Stream rows through a server-side cursor instead of loading millions of rows at once.
    named = conn.cursor(name=f"recover_{table}", cursor_factory=RealDictCursor)
    named.itersize = batch_fetch

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        named.execute(sql)
        pbar = tqdm(total=total, desc=f"编码 {table}", file=sys.stdout, mininterval=0.5)
        for row in named:
            pbar.update(1)
            gj = row.get("__geojson")
            if not gj:
                n_skipped += 1
                continue
            try:
                geom = shp_shape(json.loads(gj))
            except Exception:
                n_skipped += 1
                continue
            if geom.is_empty:
                n_skipped += 1
                continue

            # Business properties, excluding geometry and internal encoding fields.
            props = {}
            for c in biz_cols:
                if c in SKIP_PROP_COLS:
                    continue
                v = row.get(c)
                if v is None or v == "":
                    continue
                props[c] = v if isinstance(v, (str, int, float, bool)) else str(v)

            base_gid = str(row.get("geosot_code_id") or "").strip()
            base_main = str(row.get("main_code") or "").strip()

            parts = _explode_geometries(geom, base_name=base_gid or base_main or "feat")
            for part_idx, (_, one_geom) in enumerate(parts):
                try:
                    if one_geom.geom_type == "Point":
                        main_point = one_geom
                    else:
                        main_point = one_geom.centroid
                    cc_lon = float(main_point.x)
                    cc_lat = float(main_point.y)
                    if strict_china:
                        _check_china_range(cc_lon, cc_lat, True)
                    cc_code = encode_gbt40087_ne(cc_lon, cc_lat, level=level)

                    vertex_codes: List[str] = []
                    for vx, vy in _iter_vertices(one_geom):
                        if strict_china:
                            _check_china_range(float(vx), float(vy), True)
                        vertex_codes.append(encode_gbt40087_ne(float(vx), float(vy), level=level))
                except Exception:
                    n_skipped += 1
                    continue

                if not vertex_codes:
                    vertex_codes = [cc_code]

                # LCPC applies only to lines/polygons with more than one vertex.
                lcpc_code = ""
                lcpc_level = ""
                if len(vertex_codes) > 1:
                    lcpc_code = longest_common_prefix([cc_code] + vertex_codes)
                    if lcpc_code:
                        lcpc_level = str(len(lcpc_code) - 1)  # The code body length equals the encoding level.

                # Reuse the PG geosot_code_id when present; otherwise use main_code plus a sequence number.
                if base_gid:
                    feature_id = base_gid if part_idx == 0 else f"{base_gid}_p{part_idx}"
                else:
                    seq = seq_counter.get(cc_code, 0)
                    seq_counter[cc_code] = seq + 1
                    feature_id = f"{cc_code}{seq:04d}"

                writer.writerow(
                    {
                        "feature_id": feature_id,
                        "source": source_name,
                        "feature_label": feature_label,
                        "geometry_type": one_geom.geom_type,
                        "main_code": cc_code,
                        "cc_lon": f"{cc_lon:.12f}",
                        "cc_lat": f"{cc_lat:.12f}",
                        "level": level,
                        "vertex_count": len(vertex_codes),
                        "lcpc_code": lcpc_code,
                        "lcpc_level": lcpc_level,
                        "vertex_codes_json": json.dumps(vertex_codes, ensure_ascii=False),
                        "props_json": json.dumps(props, ensure_ascii=False),
                    }
                )
                n_features += 1
        pbar.close()

    named.close()
    print(f"[OK] {table}: 写出 {n_features:,} 个要素 (跳过 {n_skipped:,}) -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="恢复流程(1/3): PG -> GeoSOT 编码 CSV")
    parser.add_argument("--tables", default="", help="逗号分隔的表名子集 (默认全部 4 张)")
    parser.add_argument("--out-dir", default="recover_out", help="CSV 输出目录 (默认 recover_out)")
    parser.add_argument("--level", type=int, default=32, help="编码级别 (默认 32)")
    parser.add_argument("--strict-china", action="store_true", help="启用中国范围校验")
    parser.add_argument("--limit", type=int, default=0, help="每表限量 (冒烟测试, 0=全量)")
    parser.add_argument("--fetch", type=int, default=5000, help="服务器端游标 itersize (默认 5000)")
    parser.add_argument("--host", default=PG_CONFIG["host"])
    parser.add_argument("--port", type=int, default=PG_CONFIG["port"])
    parser.add_argument("--dbname", default=PG_CONFIG["dbname"])
    parser.add_argument("--user", default=PG_CONFIG["user"])
    parser.add_argument("--password", default=PG_CONFIG["password"])
    args = parser.parse_args()

    if args.tables.strip():
        tables = [t.strip() for t in args.tables.split(",") if t.strip()]
        for t in tables:
            if t not in TABLE_MAP:
                raise ValueError(f"未知表 {t}; 可选: {list(TABLE_MAP)}")
    else:
        tables = list(TABLE_MAP)

    cfg = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "user": args.user,
        "password": args.password,
    }
    conn = psycopg2.connect(**cfg)
    # Named server-side cursors require a transaction. Keep autocommit=False and roll back after each read-only table export.
    conn.autocommit = False

    print("=" * 72)
    print("恢复流程(1/3): PostGIS -> GeoSOT 编码")
    print(f"PG: {cfg['host']}:{cfg['port']}/{cfg['dbname']}")
    print(f"表: {tables}  级别: L{args.level}  输出: {args.out_dir}")
    print("=" * 72)

    try:
        for t in tables:
            try:
                encode_table(
                    conn=conn,
                    table=t,
                    out_dir=args.out_dir,
                    level=args.level,
                    strict_china=args.strict_china,
                    limit=(args.limit or None),
                    batch_fetch=args.fetch,
                )
            finally:
                # End the read-only transaction and reset the connection for the next table.
                conn.rollback()
    finally:
        conn.close()

    print("\n[OK] 编码完成。下一步: recover_build_levdigit_backbone.py 建骨架, 再 recover_import_to_neo4j.py 导入。")


if __name__ == "__main__":
    main()

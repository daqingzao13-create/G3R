"""
GB/T 40087-2021 encoder for the northeastern hemisphere, with an optional China-range check.
Points are encoded directly; lines and polygons use a centroid main code and an ordered
vertex-code sequence.
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from tqdm import tqdm


CHINA_LON_MIN = 73.0
CHINA_LON_MAX = 136.0
CHINA_LAT_MIN = 3.0
CHINA_LAT_MAX = 54.0


@dataclass
class EncodedFeature:
    """Encoded feature result"""

    feature_index: int
    feature_name: str
    geometry_type: str
    main_lon: float
    main_lat: float
    main_code: str
    level: int
    vertex_count: int
    vertex_codes: List[str]
    properties: Dict
    geometry: object  # Shapely geometries used for Shapefile output.


def _to_dms_parts(value: float) -> Tuple[int, int, int, float]:
    """
    Decimal-degree coordinate (non-negative only)
    (degree, minute, second integer, sub-second fraction [0,1))
    """
    if value < 0:
        raise ValueError("仅支持东北半球（经纬度必须为非负）")
    degree = int(math.floor(value))
    minutes_float = (value - degree) * 60.0
    minute = int(math.floor(minutes_float))
    seconds_float = (minutes_float - minute) * 60.0
    second = int(math.floor(seconds_float))
    sub_second = seconds_float - second
    return degree, minute, second, sub_second


def _to_fixed_bits(value: int, width: int) -> str:
    """
    Non-negative integer
    Fixed bit width
    Fixed-width binary string
    """
    if value < 0 or value >= (1 << width):
        raise ValueError(f"数值 {value} 超出 {width} bit 可表示范围")
    return format(value, f"0{width}b")


def _coord_to_31_bits(value: float) -> str:
    """
    Decimal-degree coordinate (lon/lat)
    31-bit binary (8-bit deg + 6-bit min + 6-bit sec + 11-bit sub-sec)
    """
    degree, minute, second, sub_second = _to_dms_parts(value)
    if degree > 255:
        raise ValueError("度值超出 8bit 上限（>255）")
    if minute > 59 or second > 59:
        raise ValueError("分/秒值非法（应在 0~59）")

    # Quantize sub-second to 11 bits with floor and upper-bound clamp
    sub_second_quantized = int(sub_second * 2048.0)
    if sub_second_quantized > 2047:
        sub_second_quantized = 2047

    return (
        _to_fixed_bits(degree, 8)
        + _to_fixed_bits(minute, 6)
        + _to_fixed_bits(second, 6)
        + _to_fixed_bits(sub_second_quantized, 11)
    )


def _interleave_lat_lon_to_quaternary(lat_bits: str, lon_bits: str) -> str:
    """
    31-bit latitude binary
    31-bit longitude binary
    31-digit quaternary string (Morton: lat bit then lon bit)
    """
    if len(lat_bits) != 31 or len(lon_bits) != 31:
        raise ValueError("经纬度 bit 长度必须为 31")

    quaternary_digits: List[str] = []
    for i in range(31):
        pair = lat_bits[i] + lon_bits[i]
        if pair == "00":
            quaternary_digits.append("0")
        elif pair == "01":
            quaternary_digits.append("1")
        elif pair == "10":
            quaternary_digits.append("2")
        else:
            quaternary_digits.append("3")
    return "".join(quaternary_digits)


def encode_gbt40087_ne(lon: float, lat: float, level: int = 32) -> str:
    """
    Longitude (decimal degrees)
    Latitude (decimal degrees)
    Encoding level (0~32)
    GB/T 40087-2021 code (NE Hemisphere: G0...)
    """
    if level < 0 or level > 32:
        raise ValueError("level 必须在 0~32 之间")
    if level == 0:
        return "G"
    if level == 1:
        return "G0"

    if lon < 0 or lat < 0:
        raise ValueError("仅支持东北半球（经纬度必须 >= 0）")
    if lon > 180 or lat > 90:
        raise ValueError("经纬度超出地理合法范围（lon<=180, lat<=90）")

    lat_bits = _coord_to_31_bits(lat)
    lon_bits = _coord_to_31_bits(lon)
    q31 = _interleave_lat_lon_to_quaternary(lat_bits, lon_bits)

    # Level definition:
    # Level 0: G
    # Level 1: G0 (Northeast Hemisphere)
    # Levels 2~32: G0 + (level-1) quaternary refinement digits
    return "G0" + q31[: level - 1]


def _check_china_range(lon: float, lat: float, strict: bool) -> None:
    """China extent check, optional strict mode"""
    if not strict:
        return
    if not (CHINA_LON_MIN <= lon <= CHINA_LON_MAX and CHINA_LAT_MIN <= lat <= CHINA_LAT_MAX):
        raise ValueError(
            f"坐标超出中国范围（{CHINA_LON_MIN}~{CHINA_LON_MAX}, {CHINA_LAT_MIN}~{CHINA_LAT_MAX}）："
            f"({lon}, {lat})"
        )


def _iter_vertices(geom) -> Iterable[Tuple[float, float]]:
    """Extract vertex coordinates (point/line/polygon)"""
    if isinstance(geom, Point):
        yield geom.x, geom.y
        return
    if isinstance(geom, LineString):
        for coord in geom.coords:
            yield coord[0], coord[1]
        return
    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        for coord in coords:
            yield coord[0], coord[1]
        return
    raise ValueError(f"不支持的几何类型: {type(geom)}")


def _explode_geometries(geom, base_name: str) -> List[Tuple[str, object]]:
    """Split Multi geometries"""
    if isinstance(geom, (Point, LineString, Polygon)):
        return [(base_name, geom)]
    if isinstance(geom, MultiPoint):
        return [(f"{base_name}_part{i}", g) for i, g in enumerate(geom.geoms)]
    if isinstance(geom, MultiLineString):
        return [(f"{base_name}_part{i}", g) for i, g in enumerate(geom.geoms)]
    if isinstance(geom, MultiPolygon):
        return [(f"{base_name}_part{i}", g) for i, g in enumerate(geom.geoms)]
    return []


def _feature_name(row, index: int, name_field: str) -> str:
    """Prefer reading the given field as feature name"""
    if name_field and name_field in row and row[name_field] is not None and str(row[name_field]).strip():
        return str(row[name_field]).strip()
    return f"feature_{index}"


def encode_file(
    input_path: str,
    output_csv: str | None,
    level: int,
    strict_china: bool,
    name_field: str,
    show_progress: bool = True,
    output_shp: str | None = None,
) -> str | None:
    """
    Input spatial file (Shapefile/GeoJSON/GPKG, etc.)
    Output CSV path; None means no CSV
    Encoding level
    Whether to strictly enforce China extent
    Feature name field
    Whether to show progress bar
    Optional Shapefile path with encoded attributes
    Final CSV path, or None if CSV not written
    """
    if show_progress:
        print("正在读取文件...", flush=True)
    gdf = gpd.read_file(input_path)
    if gdf.empty:
        raise ValueError("输入文件为空")
    n_total = len(gdf)
    if show_progress:
        print(f"已读取 {n_total} 个要素，正在转换坐标系...", flush=True)
    if gdf.crs is None:
        raise ValueError("输入数据缺少坐标系，请先设置 CRS")
    gdf = gdf.to_crs("EPSG:4490")
    if show_progress:
        print("开始编码...", flush=True)

    results: List[EncodedFeature] = []
    feature_index = 0
    attr_fields = [c for c in gdf.columns if c != "geometry"]

    iterator = tqdm(
        gdf.iterrows(),
        total=len(gdf),
        desc="编码进度",
        disable=not show_progress,
        mininterval=0.5,
        file=sys.stdout,
    )
    for idx, row in iterator:
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        base_name = _feature_name(row, idx, name_field)
        expanded = _explode_geometries(geom, base_name)
        for name, one_geom in expanded:
            if isinstance(one_geom, Point):
                main_point = one_geom
            else:
                main_point = one_geom.centroid

            main_lon = float(main_point.x)
            main_lat = float(main_point.y)
            _check_china_range(main_lon, main_lat, strict_china)
            main_code = encode_gbt40087_ne(main_lon, main_lat, level=level)

            vertex_codes: List[str] = []
            for vx, vy in _iter_vertices(one_geom):
                _check_china_range(float(vx), float(vy), strict_china)
                vertex_codes.append(encode_gbt40087_ne(float(vx), float(vy), level=level))

            results.append(
                EncodedFeature(
                    feature_index=feature_index,
                    feature_name=name,
                    geometry_type=one_geom.geom_type,
                    main_lon=main_lon,
                    main_lat=main_lat,
                    main_code=main_code,
                    level=level,
                    vertex_count=len(vertex_codes),
                    vertex_codes=vertex_codes,
                    properties={k: row[k] for k in attr_fields},
                    geometry=one_geom,
                )
            )
            feature_index += 1

    if not results:
        raise ValueError("没有可编码的有效几何")

    output_csv_path: str | None = None
    if output_csv:
        output_csv_path = output_csv
        if os.path.isdir(output_csv_path):
            output_csv_path = os.path.join(
                output_csv_path,
                f"gbt40087_encoded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = [
                "feature_index",
                "feature_name",
                "geometry_type",
                "main_lon",
                "main_lat",
                "level",
                "main_code",
                "vertex_count",
                "vertex_codes_json",
            ] + sorted(attr_fields)
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            for item in results:
                row_dict = {
                    "feature_index": item.feature_index,
                    "feature_name": item.feature_name,
                    "geometry_type": item.geometry_type,
                    "main_lon": f"{item.main_lon:.12f}",
                    "main_lat": f"{item.main_lat:.12f}",
                    "level": item.level,
                    "main_code": item.main_code,
                    "vertex_count": item.vertex_count,
                    "vertex_codes_json": json.dumps(item.vertex_codes, ensure_ascii=False),
                }
                for key in attr_fields:
                    val = item.properties.get(key)
                    if val is None:
                        row_dict[key] = ""
                    elif isinstance(val, (dict, list)):
                        row_dict[key] = json.dumps(val, ensure_ascii=False)
                    else:
                        row_dict[key] = val
                writer.writerow(row_dict)

    if output_shp:
        if show_progress:
            print("正在写入 Shapefile...", flush=True)
        _write_encoded_shp(results, output_shp, attr_fields)

    return output_csv_path


def _write_encoded_shp(
    results: List[EncodedFeature],
    output_path: str,
    attr_fields: List[str],
) -> None:
    """
    Write encoded features to a Shapefile attribute table. Shapefile field names are limited to
    10 characters and text values to 254 characters, so only main_code, level, and vertex_count
    are written here. The CSV remains the authoritative output for the complete vertex_codes
    sequence.
    """
    geometries = [r.geometry for r in results]

    # Build attribute table: keep original attrs + encoding fields (SHP name ≤10 chars)
    data = {}
    for k in attr_fields:
        data[k] = [r.properties.get(k) for r in results]
    data["main_code"] = [r.main_code for r in results]
    data["enc_level"] = [r.level for r in results]
    data["vtx_count"] = [r.vertex_count for r in results]

    out_gdf = gpd.GeoDataFrame(
        data,
        geometry=geometries,
        crs="EPSG:4490",
    )
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_gdf.to_file(output_path, encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    """CLI argument parser"""
    parser = argparse.ArgumentParser(description="GB/T 40087-2021（东北半球，中国范围）编码工具")
    parser.add_argument("--input", required=True, help="输入空间文件路径")
    parser.add_argument("--output", default="", help="输出 CSV 路径（默认与 SHP 一起输出，可单独指定以覆盖）")
    parser.add_argument("--level", type=int, default=32, help="编码级别（0~32，默认 32）")
    parser.add_argument(
        "--strict-china",
        action="store_true",
        help="启用中国范围严格校验（超出范围报错）",
    )
    parser.add_argument(
        "--name-field",
        default="",
        help="用作要素名称的字段名（可选）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭进度条显示",
    )
    parser.add_argument(
        "--output-shp",
        default="",
        help="输出 Shapefile 路径（默认与 CSV 一起输出，可单独指定以覆盖）",
    )
    return parser


def main() -> None:
    """Program entry point"""
    parser = build_cli()
    args = parser.parse_args()

    output_csv = args.output if args.output else None
    output_shp = args.output_shp if args.output_shp else None
    if not output_csv and not output_shp:
        # Default: write both CSV and SHP based on input path
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = os.path.dirname(os.path.abspath(args.input))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(out_dir, f"{base}_gbt40087_encoded_{ts}.csv")
        output_shp = os.path.join(out_dir, f"{base}_gbt40087_encoded_{ts}.shp")

    output_csv_path = encode_file(
        input_path=args.input,
        output_csv=output_csv,
        level=args.level,
        strict_china=args.strict_china,
        name_field=args.name_field,
        show_progress=not args.quiet,
        output_shp=output_shp,
    )
    if output_csv_path:
        print(f"编码完成，输出 CSV：{output_csv_path}")
    if output_shp:
        print(f"编码完成，输出 Shapefile：{output_shp}")


if __name__ == "__main__":
    main()

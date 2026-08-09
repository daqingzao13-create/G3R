"""
GB/T 40087-2021 解码主程序（东北半球，中国范围）
GB/T 40087-2021 decode main program (Northeast Hemisphere, China extent)
读取编码文件（CSV 或 TXT），解码为 GeoJSON 网格多边形。
Read encoded files (CSV or TXT) and decode to GeoJSON grid polygons.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


class GBT40087Decoder:
    """
    GB/T 40087-2021（东北半球）解码器。
    GB/T 40087-2021 (Northeast Hemisphere) decoder.
    """

    def __init__(self) -> None:
        self.codes: List[str] = []
        self.records: List[Dict] = []
        self.geojson: Optional[Dict] = None

    @staticmethod
    def _normalize_code(code: str) -> str:
        """
        @param code 原始编码字符串 / Raw code string
        @returns 规范化后的编码（去空白并大写） / Normalized code (strip whitespace and uppercase)
        """
        return str(code).strip().upper()

    @staticmethod
    def _is_valid_code(code: str) -> bool:
        """
        @param code 编码字符串 / Code string
        @returns 是否是有效的东北半球编码（G 或 G0...） / Whether it is a valid NE Hemisphere code (G or G0...)
        """
        if not code:
            return False
        if code == "G":
            return True
        if not code.startswith("G"):
            return False
        body = code[1:]
        if len(body) < 1 or len(body) > 32:
            return False
        # 仅支持东北半球编码：1级前缀应为 0，即 G0... / NE Hemisphere only: level-1 prefix must be 0, i.e. G0...
        if body[0] != "0":
            return False
        return all(ch in "0123" for ch in body)

    def load_from_csv(self, csv_path: str, code_column: str = "main_code", verbose: bool = True) -> int:
        """
        @param csv_path CSV 文件路径 / CSV file path
        @param code_column 编码列名 / Code column name
        @returns 成功加载的编码数量 / Number of codes successfully loaded
        """
        if verbose:
            print(f"正在从 CSV 加载编码：{csv_path} / Loading codes from CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        except Exception:
            df = pd.read_csv(csv_path, encoding="gbk", low_memory=False)

        if code_column not in df.columns:
            raise ValueError(f"CSV 中未找到列：{code_column} / Column not found in CSV: {code_column}")

        raw_codes = df[code_column].dropna().astype(str).tolist()
        normalized = [self._normalize_code(c) for c in raw_codes]
        self.codes = [c for c in normalized if self._is_valid_code(c)]
        self.records = []

        for _, row in df.iterrows():
            code_raw = row.get(code_column)
            if pd.isna(code_raw):
                continue
            code = self._normalize_code(code_raw)
            if not self._is_valid_code(code):
                continue
            props = {}
            for col in df.columns:
                if col == code_column:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                props[col] = val
            self.records.append({"code": code, "properties": props})

        if verbose:
            print(f"成功加载 {len(self.codes)} 个有效编码 / Successfully loaded {len(self.codes)} valid codes")
        return len(self.codes)

    def load_from_txt(self, txt_path: str, verbose: bool = True) -> int:
        """
        @param txt_path TXT 文件路径（每行一个编码） / TXT file path (one code per line)
        @returns 成功加载的编码数量 / Number of codes successfully loaded
        """
        if verbose:
            print(f"正在从 TXT 加载编码：{txt_path} / Loading codes from TXT: {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        normalized = [self._normalize_code(line) for line in lines if line.strip()]
        self.codes = [c for c in normalized if self._is_valid_code(c)]
        self.records = [{"code": c, "properties": {}} for c in self.codes]

        if verbose:
            print(f"成功加载 {len(self.codes)} 个有效编码 / Successfully loaded {len(self.codes)} valid codes")
        return len(self.codes)

    def load_from_list(self, codes: List[str], verbose: bool = True) -> int:
        """
        @param codes 编码列表 / List of codes
        @returns 成功加载的编码数量 / Number of codes successfully loaded
        """
        normalized = [self._normalize_code(c) for c in codes]
        self.codes = [c for c in normalized if self._is_valid_code(c)]
        self.records = [{"code": c, "properties": {}} for c in self.codes]
        if verbose:
            print(f"成功加载 {len(self.codes)} 个有效编码 / Successfully loaded {len(self.codes)} valid codes")
        return len(self.codes)

    @staticmethod
    def _digit_to_bits(digit: str) -> Tuple[str, str]:
        """
        @param digit 四进制码元（0~3） / Quaternary digit (0~3)
        @returns (lat_bit, lon_bit)
        """
        if digit == "0":
            return "0", "0"
        if digit == "1":
            return "0", "1"
        if digit == "2":
            return "1", "0"
        if digit == "3":
            return "1", "1"
        raise ValueError(f"非法四进制码元: {digit} / Invalid quaternary digit: {digit}")

    @staticmethod
    def _bits31_to_degree(bits31: str) -> float:
        """
        @param bits31 31 位二进制（8 度 + 6 分 + 6 秒 + 11 秒以下） / 31-bit binary (8 deg + 6 min + 6 sec + 11 sub-sec)
        @returns 十进制度坐标 / Decimal-degree coordinate
        """
        if len(bits31) != 31:
            raise ValueError("bits31 长度必须为 31 / bits31 length must be 31")

        d = int(bits31[0:8], 2)
        m = int(bits31[8:14], 2)
        s = int(bits31[14:20], 2)
        u = int(bits31[20:31], 2)

        return d + m / 60.0 + (s + u / 2048.0) / 3600.0

    def decode_code_to_center(self, code: str) -> Tuple[float, float, int]:
        """
        @param code GB/T 40087 编码 / GB/T 40087 code
        @returns (center_lon, center_lat, level)
        """
        min_lon, min_lat, max_lon, max_lat, level = self.decode_code_to_bbox(code)
        return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0, level

    @staticmethod
    def _parse_vertex_codes(value) -> List[str]:
        """
        @param value CSV 中的 vertex_codes_json 字段 / vertex_codes_json field from CSV
        @returns 折点编码列表 / List of vertex codes
        """
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip().upper() for v in value if str(v).strip()]
        value_str = str(value).strip()
        if not value_str:
            return []
        try:
            arr = json.loads(value_str)
            if isinstance(arr, list):
                return [str(v).strip().upper() for v in arr if str(v).strip()]
        except Exception:
            return []
        return []

    def decode_code_to_bbox(self, code: str) -> Tuple[float, float, float, float, int]:
        """
        @param code GB/T 40087 编码（G 或 G0...） / GB/T 40087 code (G or G0...)
        @returns (min_lon, min_lat, max_lon, max_lat, level)
        """
        code = self._normalize_code(code)
        if not self._is_valid_code(code):
            raise ValueError(f"非法编码: {code} / Invalid code: {code}")

        if code == "G":
            # 全局扩展网格，按现有编码器可表示范围返回 / Global expanded grid; return range representable by current encoder
            return 0.0, 0.0, 256.0, 256.0, 0

        digits = code[1:]
        level = len(digits)
        # 第1位是东北半球标识（0），后续位才是细分码 / First digit is NE Hemisphere flag (0); remaining digits are refinement
        refine_digits = digits[1:]

        lat_prefix_bits: List[str] = []
        lon_prefix_bits: List[str] = []
        for d in refine_digits:
            lb, lob = self._digit_to_bits(d)
            lat_prefix_bits.append(lb)
            lon_prefix_bits.append(lob)

        # 将细分前缀补齐到 31 位，获得当前网格边界（最小值补 0，最大值补 1）
        # Pad refinement prefix to 31 bits for grid bounds (min pads 0, max pads 1)
        # 其中 level=1（G0）时，细分位长度为 0，对应东北半球 1 级网格。
        # At level=1 (G0), refinement length is 0 for the NE Hemisphere level-1 grid.
        remain = 31 - len(refine_digits)
        if remain < 0:
            raise ValueError(
                "细分编码位数超过 31，超出二维椭球面编码位数 / Refinement digits exceed 31, beyond 2D ellipsoid coding length"
            )

        lat_min_bits = "".join(lat_prefix_bits) + ("0" * remain)
        lon_min_bits = "".join(lon_prefix_bits) + ("0" * remain)
        lat_max_bits = "".join(lat_prefix_bits) + ("1" * remain)
        lon_max_bits = "".join(lon_prefix_bits) + ("1" * remain)

        min_lat = self._bits31_to_degree(lat_min_bits)
        min_lon = self._bits31_to_degree(lon_min_bits)
        max_lat = self._bits31_to_degree(lat_max_bits)
        max_lon = self._bits31_to_degree(lon_max_bits)

        return min_lon, min_lat, max_lon, max_lat, level

    def decode_to_geojson(
        self,
        max_features: Optional[int] = None,
        show_progress: bool = True,
        verbose: bool = True,
        decode_mode: str = "geometry",
    ) -> Dict:
        """
        @param max_features 最大解码数量（None 表示全部） / Max features to decode (None = all)
        @param show_progress 是否显示进度条 / Whether to show progress bar
        @param verbose 是否输出日志 / Whether to print logs
        @param decode_mode 解码模式：grid / geometry / Decode mode: grid / geometry
        @returns GeoJSON FeatureCollection
        """
        if not self.codes:
            raise ValueError("没有可解码的编码，请先加载数据 / No codes to decode; load data first")

        records_to_decode = self.records[:max_features] if max_features and max_features > 0 else self.records

        if verbose:
            print(
                f"\n开始批量解码 {len(records_to_decode)} 个编码"
                f"{'（已根据 --max-features 截断）' if max_features else ''}..."
                f" / Start batch decoding {len(records_to_decode)} codes"
                f"{' (truncated by --max-features)' if max_features else ''}..."
            )

        features = []
        success_count = 0
        failure_count = 0

        decode_mode = (decode_mode or "geometry").lower()
        if decode_mode not in ("grid", "geometry"):
            raise ValueError("decode_mode 仅支持 grid 或 geometry / decode_mode supports only grid or geometry")

        for code in tqdm(
            records_to_decode,
            desc="解码进度 / Decoding",
            total=len(records_to_decode),
            disable=not show_progress,
        ):
            code_repr = code["code"] if isinstance(code, dict) and "code" in code else str(code)
            try:
                code_value = code["code"]
                original_properties = code.get("properties", {})
                feature_properties = {
                    "code": code_value,
                }
                feature_properties.update(original_properties)

                if decode_mode == "grid":
                    min_lon, min_lat, max_lon, max_lat, level = self.decode_code_to_bbox(code_value)
                    polygon = [
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]
                    feature_properties.update(
                        {
                            "grid_level": level,
                            "min_lon": min_lon,
                            "min_lat": min_lat,
                            "max_lon": max_lon,
                            "max_lat": max_lat,
                        }
                    )
                    geometry = {"type": "Polygon", "coordinates": [polygon]}
                else:
                    geom_type = str(original_properties.get("geometry_type", "")).strip()
                    vertex_codes = self._parse_vertex_codes(original_properties.get("vertex_codes_json"))

                    # 没有折点信息时，回退为主码网格中心点 / Fallback to main-code grid center when no vertex info
                    if not vertex_codes:
                        center_lon, center_lat, level = self.decode_code_to_center(code_value)
                        feature_properties.update({"grid_level": level, "decode_fallback": "main_code_center"})
                        geometry = {"type": "Point", "coordinates": [center_lon, center_lat]}
                    else:
                        points: List[List[float]] = []
                        max_level = 0
                        for vcode in vertex_codes:
                            lon, lat, lv = self.decode_code_to_center(vcode)
                            points.append([lon, lat])
                            if lv > max_level:
                                max_level = lv
                        feature_properties.update({"grid_level": max_level})

                        if geom_type == "Point" or len(points) == 1:
                            geometry = {"type": "Point", "coordinates": points[0]}
                        elif geom_type == "LineString":
                            geometry = {"type": "LineString", "coordinates": points}
                        elif geom_type == "Polygon":
                            ring = points[:]
                            if ring[0] != ring[-1]:
                                ring.append(ring[0])
                            geometry = {"type": "Polygon", "coordinates": [ring]}
                        else:
                            # 几何类型缺失时，按点数启发式判断 / When geometry type is missing, infer from point count
                            if len(points) >= 3:
                                ring = points[:]
                                if ring[0] != ring[-1]:
                                    ring.append(ring[0])
                                geometry = {"type": "Polygon", "coordinates": [ring]}
                                feature_properties["decode_fallback"] = "geometry_type_missing_assume_polygon"
                            else:
                                geometry = {"type": "LineString", "coordinates": points}
                                feature_properties["decode_fallback"] = "geometry_type_missing_assume_line"

                feature = {
                    "type": "Feature",
                    "properties": feature_properties,
                    "geometry": geometry,
                }
                features.append(feature)
                success_count += 1
            except Exception as e:
                failure_count += 1
                if verbose and failure_count <= 5:
                    print(f"警告：编码 {code_repr} 解码失败：{e} / Warning: decode failed for {code_repr}: {e}")

        self.geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "EPSG:4490"},
            },
            "features": features,
            "properties": {
                "total_count": len(features),
                "success_count": success_count,
                "failure_count": failure_count,
                "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": f"GB/T 40087-2021 解码结果（东北半球，mode={decode_mode}） / GB/T 40087-2021 decode result (NE Hemisphere, mode={decode_mode})",
                "coordinate_system": "CGCS2000 (EPSG:4490)",
                "decode_mode": decode_mode,
            },
        }

        if verbose:
            print("\n解码完成： / Decode finished:")
            print(f"  - 成功：{success_count} 个 / Success: {success_count}")
            print(f"  - 失败：{failure_count} 个 / Failed: {failure_count}")
        return self.geojson

    def save_geojson(self, output_path: str, verbose: bool = True) -> None:
        """
        @param output_path 输出 GeoJSON 路径 / Output GeoJSON path
        """
        if self.geojson is None:
            raise ValueError("没有可保存的 GeoJSON，请先执行解码 / No GeoJSON to save; decode first")

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.geojson, f, ensure_ascii=False, indent=2)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if verbose:
            print(f"\nGeoJSON 文件已保存：{output_path} / GeoJSON saved: {output_path}")
            print(f"文件大小：{size_mb:.2f} MB / File size: {size_mb:.2f} MB")

    def print_summary(self) -> None:
        """
        @description 打印解码摘要 / Print decode summary
        """
        if self.geojson is None:
            print("尚未执行解码 / Decode has not been run")
            return

        props = self.geojson["properties"]
        level_stats: Dict[int, int] = {}
        for feat in self.geojson["features"]:
            lv = int(feat["properties"].get("grid_level", 0))
            level_stats[lv] = level_stats.get(lv, 0) + 1

        print("\n" + "=" * 70)
        print("解码完成摘要 / Decode summary")
        print("=" * 70)
        print(f"生成时间：{props['generated_time']} / Generated at: {props['generated_time']}")
        print(f"坐标系统：{props['coordinate_system']} / CRS: {props['coordinate_system']}")
        print(f"解码模式：{props.get('decode_mode', 'geometry')} / Decode mode: {props.get('decode_mode', 'geometry')}")
        print(f"总计：{props['total_count']} 个要素 / Total: {props['total_count']} features")
        print(f"成功：{props['success_count']} 个 / Success: {props['success_count']}")
        print(f"失败：{props['failure_count']} 个 / Failed: {props['failure_count']}")
        if level_stats:
            print("\n网格级别分布： / Grid level distribution:")
            for lv in sorted(level_stats.keys()):
                print(f"  - 第{lv:2d}级: {level_stats[lv]:6d} 个 / Level {lv:2d}: {level_stats[lv]:6d}")
        print("=" * 70)


def main() -> None:
    """
    @description 命令行入口 / CLI entry point
    """
    use_demo = "--demo" in sys.argv
    quiet = "--quiet" in sys.argv
    max_features: Optional[int] = None
    code_column = "main_code"
    decode_mode = "geometry"

    if not quiet:
        print("\n" + "=" * 70)
        print("GB/T 40087-2021 解码程序（东北半球） / GB/T 40087-2021 decoder (NE Hemisphere)")
        print("将 G0... 编码转换为网格 GeoJSON 多边形 / Convert G0... codes to GeoJSON grid polygons")
        print("=" * 70)

    for i, arg in enumerate(sys.argv):
        if arg == "--max-features" and i + 1 < len(sys.argv):
            max_features = int(sys.argv[i + 1])
        if arg == "--code-column" and i + 1 < len(sys.argv):
            code_column = sys.argv[i + 1]
        if arg == "--decode-mode" and i + 1 < len(sys.argv):
            decode_mode = sys.argv[i + 1].strip().lower()

    input_file = None
    output_file = None
    skip_next = False
    for i, arg in enumerate(sys.argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--max-features", "--code-column", "--decode-mode"):
            skip_next = True
            continue
        if arg in ("--demo", "--quiet"):
            continue
        if not arg.startswith("--"):
            if input_file is None:
                input_file = arg
            elif output_file is None:
                output_file = arg

    if not use_demo and not input_file:
        print("\n用法： / Usage:")
        print("  python gbt40087_decode_main.py <输入文件> [输出geojson] [--code-column main_code] [--decode-mode geometry|grid] [--max-features N] [--quiet]")
        print("  python gbt40087_decode_main.py --demo")
        print("\n支持输入： / Supported inputs:")
        print("  - CSV（默认读取 main_code 列，可用 --code-column 指定） / CSV (default column main_code; override with --code-column)")
        print("  - TXT（每行一个编码） / TXT (one code per line)")
        print("\n示例： / Examples:")
        print("  python gbt40087_decode_main.py encoded_out.csv")
        print("  python gbt40087_decode_main.py encoded_out.csv out_geometry.geojson --decode-mode geometry")
        print("  python gbt40087_decode_main.py encoded_out.csv out_grid.geojson --decode-mode grid --max-features 100")
        return

    try:
        decoder_app = GBT40087Decoder()

        if use_demo:
            demo_codes = [
                "G0",
                "G001",
                "G0013103",
                "G001310321",
                "G00131032123310222031322010122013",
            ]
            decoder_app.load_from_list(demo_codes, verbose=not quiet)
            output_file = output_file or "output/gbt40087_demo_grid.geojson"
        else:
            if not os.path.exists(input_file):
                print("[ERROR] 输入文件不存在 / Input file does not exist")
                sys.exit(1)

            if input_file.lower().endswith(".csv"):
                decoder_app.load_from_csv(input_file, code_column=code_column, verbose=not quiet)
            elif input_file.lower().endswith(".txt"):
                decoder_app.load_from_txt(input_file, verbose=not quiet)
            else:
                print("[ERROR] 仅支持 CSV 或 TXT 输入 / Only CSV or TXT input is supported")
                sys.exit(1)

            if not output_file:
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_file = f"output/{base_name}_gbt40087_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"

        decoder_app.decode_to_geojson(
            max_features=max_features,
            show_progress=not quiet,
            verbose=not quiet,
            decode_mode=decode_mode,
        )
        decoder_app.save_geojson(output_file, verbose=not quiet)
        if not quiet:
            decoder_app.print_summary()
            print("\n[OK] 解码完成 / Decode completed")
        else:
            print("[OK] 解码完成 / Decode completed")
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 用户取消解码 / User cancelled decode")
    except Exception as e:
        print(f"\n[ERROR] 解码失败: {e} / Decode failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

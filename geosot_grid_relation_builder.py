#!/usr/bin/env python3
# coding: utf-8
"""
GeoSOT(G0) 网格节点与方向关系一键构建工具
One-click builder for GeoSOT(G0) grid nodes and directional relationships

功能 / Features:
1. 从 GB/T 40087 编码 CSV 中提取网格前缀（L1~L32）
   Extract grid prefixes (L1~L32) from GB/T 40087 encoded CSV
2. 按 grid01Code~grid32Code 标签创建网格节点
   Create grid nodes under labels grid01Code~grid32Code
3. 自动构建每层 EAST_OF / NORTH_OF 关系
   Automatically build per-level EAST_OF / NORTH_OF relationships

说明 / Notes:
- 仅处理东北半球编码（G0...） / Only process Northeast Hemisphere codes (G0...)
- 节点属性名采用 grid_L{level}_code（如 grid_L1_code, grid_L32_code）
  Node property names use grid_L{level}_code (e.g. grid_L1_code, grid_L32_code)
- 关系定义与现有 QASystem 一致： / Relationship semantics match existing QASystem:
  - A 在 B 东边：A-[:EAST_OF]->B / A is east of B: A-[:EAST_OF]->B
  - A 在 B 北边：A-[:NORTH_OF]->B / A is north of B: A-[:NORTH_OF]->B
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase
from tqdm import tqdm


# 默认连接参数（与 QASystem 现有脚本保持一致） / Default connection params (aligned with existing QASystem scripts)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

BATCH_SIZE = 5000
CHINA_BBOX = (73.0, 136.0, 3.0, 54.0)  # (min_lon, max_lon, min_lat, max_lat)


def normalize_code(code: str) -> str:
    """
    @param code 原始编码 / Raw code
    @returns 规范化编码 / Normalized code
    """
    return str(code).strip().upper()


def is_valid_g0_code(code: str) -> bool:
    """
    @param code 编码 / Code
    @returns 是否为合法 G0 编码（level 1~32） / Whether it is a valid G0 code (level 1~32)
    """
    if not code or not code.startswith("G"):
        return False
    body = code[1:]
    if len(body) < 1 or len(body) > 32:
        return False
    if body[0] != "0":
        return False
    return all(c in "0123" for c in body)


def level_code(full_code: str, level: int) -> str:
    """
    @param full_code 完整编码（如 32 级） / Full code (e.g. level 32)
    @param level 层级（1~32） / Level (1~32)
    @returns 对应层级前缀编码 / Prefix code at the given level
    """
    if level < 1 or level > 32:
        raise ValueError("level 必须在 1~32 / level must be between 1 and 32")
    # 约定：编码总是 "G" + body；body 长度即层级 / Convention: code is always "G" + body; body length is the level
    return full_code[: level + 1]


def code_to_xy(code: str) -> Tuple[int, int, int]:
    """
    @param code G0 层级编码（G + body） / G0 level code (G + body)
    @returns (x, y, k)，k 为细分位数（k=level-1） / (x, y, k) where k is refinement bit count (k=level-1)
    """
    code = normalize_code(code)
    if not is_valid_g0_code(code):
        raise ValueError(f"非法编码: {code} / Invalid code: {code}")

    body = code[1:]
    refine = body[1:]  # 去掉第1位半球标识 0 / Drop first hemisphere digit 0
    k = len(refine)
    x = 0
    y = 0

    for d in refine:
        # digit -> (lat_bit, lon_bit)
        # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        if d == "0":
            lat_bit, lon_bit = 0, 0
        elif d == "1":
            lat_bit, lon_bit = 0, 1
        elif d == "2":
            lat_bit, lon_bit = 1, 0
        else:
            lat_bit, lon_bit = 1, 1

        x = (x << 1) | lon_bit
        y = (y << 1) | lat_bit

    return x, y, k


def digit_to_bits(digit: str) -> Tuple[str, str]:
    """
    @param digit 四进制字符 / Quaternary digit
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


def bits31_to_degree(bits31: str) -> float:
    """
    @param bits31 31 位二进制（8度+6分+6秒+11秒以下） / 31-bit binary (8 deg + 6 min + 6 sec + 11 sub-sec)
    @returns 十进制度 / Decimal degrees
    """
    d = int(bits31[0:8], 2)
    m = int(bits31[8:14], 2)
    s = int(bits31[14:20], 2)
    u = int(bits31[20:31], 2)
    return d + m / 60.0 + (s + u / 2048.0) / 3600.0


def code_to_bbox(code: str) -> Tuple[float, float, float, float]:
    """
    @param code G0 编码 / G0 code
    @returns (min_lon, max_lon, min_lat, max_lat)
    """
    code = normalize_code(code)
    if not is_valid_g0_code(code):
        raise ValueError(f"非法编码: {code} / Invalid code: {code}")

    body = code[1:]
    refine = body[1:]  # 去掉半球位 / Drop hemisphere digit
    lat_prefix: List[str] = []
    lon_prefix: List[str] = []
    for d in refine:
        lb, lob = digit_to_bits(d)
        lat_prefix.append(lb)
        lon_prefix.append(lob)

    remain = 31 - len(refine)
    lat_min_bits = "".join(lat_prefix) + ("0" * remain)
    lon_min_bits = "".join(lon_prefix) + ("0" * remain)
    lat_max_bits = "".join(lat_prefix) + ("1" * remain)
    lon_max_bits = "".join(lon_prefix) + ("1" * remain)

    min_lat = bits31_to_degree(lat_min_bits)
    min_lon = bits31_to_degree(lon_min_bits)
    max_lat = bits31_to_degree(lat_max_bits)
    max_lon = bits31_to_degree(lon_max_bits)
    return min_lon, max_lon, min_lat, max_lat


def bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    """
    @param a (min_lon, max_lon, min_lat, max_lat)
    @param b (min_lon, max_lon, min_lat, max_lat)
    @returns 是否相交 / Whether the boxes intersect
    """
    return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])


def xy_to_code(x: int, y: int, k: int) -> str:
    """
    @param x 经向索引 / Longitude-direction index
    @param y 纬向索引 / Latitude-direction index
    @param k 细分位数 / Refinement bit count
    @returns 编码（G + body，body 长度 = k+1） / Code (G + body, body length = k+1)
    """
    if k < 0:
        raise ValueError("k 不能小于 0 / k cannot be less than 0")

    refine_digits: List[str] = []
    for i in range(k - 1, -1, -1):
        lon_bit = (x >> i) & 1
        lat_bit = (y >> i) & 1
        pair = (lat_bit, lon_bit)
        if pair == (0, 0):
            refine_digits.append("0")
        elif pair == (0, 1):
            refine_digits.append("1")
        elif pair == (1, 0):
            refine_digits.append("2")
        else:
            refine_digits.append("3")

    return "G0" + "".join(refine_digits)


class GeoSotGridRelationBuilder:
    """
    GeoSOT 网格节点与关系构建器 / GeoSOT grid node and relation builder
    """

    def __init__(self, uri: str, user: str, password: str, database: str):
        """
        @param uri Neo4j URI
        @param user 用户名 / Username
        @param password 密码 / Password
        @param database 数据库名 / Database name
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        """@description 关闭连接 / Close connection"""
        self.driver.close()

    def _get_session(self):
        """@returns 数据库会话 / Database session"""
        return self.driver.session(database=self.database)

    def test_connection(self) -> bool:
        """@returns 是否连接成功 / Whether connection succeeded"""
        try:
            with self._get_session() as session:
                rec = session.run("RETURN 1 AS v").single()
                return rec and rec["v"] == 1
        except Exception:
            return False

    def create_constraints(self) -> None:
        """@description 为 grid01Code~grid32Code 创建唯一约束 / Create unique constraints for grid01Code~grid32Code"""
        with self._get_session() as session:
            for level in range(1, 33):
                label = f"grid{level:02d}Code"
                prop = f"grid_L{level}_code"
                cypher = (
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
                session.run(cypher)

    def collect_level_codes_from_csv(self, csv_path: str, code_column: str = "main_code") -> Dict[int, Set[str]]:
        """
        @param csv_path 编码 CSV 路径 / Encoded CSV path
        @param code_column 主码字段名 / Main-code field name
        @returns {level: set(codes)}
        """
        level_codes: Dict[int, Set[str]] = defaultdict(set)

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in tqdm(rows, desc="提取编码前缀 / Extract code prefixes"):
            main_code = normalize_code(row.get(code_column, ""))
            if is_valid_g0_code(main_code):
                max_level = len(main_code) - 1
                for level in range(1, min(32, max_level) + 1):
                    level_codes[level].add(level_code(main_code, level))

            vertex_codes_raw = row.get("vertex_codes_json", "")
            if vertex_codes_raw:
                try:
                    vertex_codes = json.loads(vertex_codes_raw)
                except Exception:
                    vertex_codes = []
                if isinstance(vertex_codes, list):
                    for vc in vertex_codes:
                        vc_norm = normalize_code(vc)
                        if not is_valid_g0_code(vc_norm):
                            continue
                        max_level = len(vc_norm) - 1
                        for level in range(1, min(32, max_level) + 1):
                            level_codes[level].add(level_code(vc_norm, level))

        return level_codes

    def collect_level_codes_from_region(
        self,
        min_level: int = 1,
        max_level: int = 12,
        region_bbox: Tuple[float, float, float, float] = CHINA_BBOX,
    ) -> Dict[int, Set[str]]:
        """
        @param min_level 最小层级（1~32） / Minimum level (1~32)
        @param max_level 最大层级（1~32） / Maximum level (1~32)
        @param region_bbox 目标区域包围盒 (min_lon, max_lon, min_lat, max_lat) / Target region bbox
        @returns {level: set(codes)}
        """
        if min_level < 1 or max_level > 32 or min_level > max_level:
            raise ValueError("层级范围非法 / Invalid level range")

        level_codes: Dict[int, Set[str]] = defaultdict(set)
        current = {"G0"}  # L1

        if min_level <= 1 <= max_level:
            level_codes[1].add("G0")

        for level in range(2, max_level + 1):
            nxt: Set[str] = set()
            for parent in current:
                for d in "0123":
                    child = parent + d
                    if bbox_intersects(code_to_bbox(child), region_bbox):
                        nxt.add(child)

            if min_level <= level <= max_level:
                level_codes[level].update(nxt)
            current = nxt
            if not current:
                break

        return level_codes

    def collect_level_codes_template(
        self,
        min_level: int = 1,
        max_level: int = 32,
    ) -> Dict[int, Set[str]]:
        """
        @param min_level 最小层级 / Minimum level
        @param max_level 最大层级 / Maximum level
        @returns 每层固定码元集合（0/1/2/3） / Fixed digit set per level (0/1/2/3)
        """
        if min_level < 1 or max_level > 32 or min_level > max_level:
            raise ValueError("层级范围非法 / Invalid level range")

        level_codes: Dict[int, Set[str]] = defaultdict(set)
        for level in range(min_level, max_level + 1):
            level_codes[level] = {"0", "1", "2", "3"}
        return level_codes

    def create_grid_nodes(self, level_codes: Dict[int, Set[str]], batch_size: int = BATCH_SIZE) -> None:
        """
        @param level_codes 各层级编码集合 / Code sets per level
        @param batch_size 批量大小 / Batch size
        """
        with self._get_session() as session:
            for level in range(1, 33):
                codes = sorted(level_codes.get(level, set()))
                if not codes:
                    continue

                label = f"grid{level:02d}Code"
                prop = f"grid_L{level}_code"
                total_batches = (len(codes) + batch_size - 1) // batch_size

                for i in tqdm(range(0, len(codes), batch_size), desc=f"创建L{level:02d}节点 / Create L{level:02d} nodes", total=total_batches, leave=False):
                    chunk = codes[i : i + batch_size]
                    data = [{"code": c, "level": level} for c in chunk]
                    session.run(
                        f"""
                        UNWIND $data AS item
                        MERGE (g:{label} {{{prop}: item.code}})
                        ON CREATE SET g.level = item.level, g.created_at = datetime()
                        """,
                        data=data,
                    )

    def create_direction_relations(
        self,
        level_codes: Dict[int, Set[str]],
        batch_size: int = BATCH_SIZE,
        local_digit_mode: bool = False,
    ) -> Dict[int, Dict[str, int]]:
        """
        @param level_codes 各层级编码集合 / Code sets per level
        @param batch_size 批量大小 / Batch size
        """
        relation_stats: Dict[int, Dict[str, int]] = {}

        with self._get_session() as session:
            for level in range(1, 33):
                codes = sorted(level_codes.get(level, set()))
                if not codes:
                    continue

                label = f"grid{level:02d}Code"
                prop = f"grid_L{level}_code"

                # 建立同层编码 -> 坐标索引映射 / Build same-level code -> (x,y) index map
                code_set = set(codes)
                xy_to_code_map: Dict[Tuple[int, int], str] = {}
                for code in codes:
                    if local_digit_mode:
                        d = str(code)
                        if d not in ("0", "1", "2", "3"):
                            continue
                        # digit -> (lat_bit, lon_bit)
                        if d == "0":
                            y, x = 0, 0
                        elif d == "1":
                            y, x = 0, 1
                        elif d == "2":
                            y, x = 1, 0
                        else:
                            y, x = 1, 1
                    else:
                        x, y, _ = code_to_xy(code)
                    xy_to_code_map[(x, y)] = code

                east_edges: List[Dict[str, str]] = []
                north_edges: List[Dict[str, str]] = []

                for code in codes:
                    if local_digit_mode:
                        d = str(code)
                        if d not in ("0", "1", "2", "3"):
                            continue
                        if d == "0":
                            y, x = 0, 0
                        elif d == "1":
                            y, x = 0, 1
                        elif d == "2":
                            y, x = 1, 0
                        else:
                            y, x = 1, 1
                    else:
                        x, y, _ = code_to_xy(code)

                    # west 邻居存在时：当前 code 在其东边 -> code -[:EAST_OF]-> west_code
                    # If west neighbor exists: current code is east of it -> code -[:EAST_OF]-> west_code
                    if x > 0:
                        west_code = xy_to_code_map.get((x - 1, y))
                        if west_code and west_code in code_set:
                            east_edges.append({"from_code": code, "to_code": west_code})

                    # south 邻居存在时：当前 code 在其北边 -> code -[:NORTH_OF]-> south_code
                    # If south neighbor exists: current code is north of it -> code -[:NORTH_OF]-> south_code
                    if y > 0:
                        south_code = xy_to_code_map.get((x, y - 1))
                        if south_code and south_code in code_set:
                            north_edges.append({"from_code": code, "to_code": south_code})

                # 批量写 EAST_OF / Batch write EAST_OF
                if east_edges:
                    total_batches = (len(east_edges) + batch_size - 1) // batch_size
                    for i in tqdm(range(0, len(east_edges), batch_size), desc=f"L{level:02d} EAST_OF", total=total_batches, leave=False):
                        chunk = east_edges[i : i + batch_size]
                        session.run(
                            f"""
                            UNWIND $data AS item
                            MATCH (a:{label} {{{prop}: item.from_code}})
                            MATCH (b:{label} {{{prop}: item.to_code}})
                            MERGE (a)-[:EAST_OF]->(b)
                            """,
                            data=chunk,
                        )

                # 批量写 NORTH_OF / Batch write NORTH_OF
                if north_edges:
                    total_batches = (len(north_edges) + batch_size - 1) // batch_size
                    for i in tqdm(range(0, len(north_edges), batch_size), desc=f"L{level:02d} NORTH_OF", total=total_batches, leave=False):
                        chunk = north_edges[i : i + batch_size]
                        session.run(
                            f"""
                            UNWIND $data AS item
                            MATCH (a:{label} {{{prop}: item.from_code}})
                            MATCH (b:{label} {{{prop}: item.to_code}})
                            MERGE (a)-[:NORTH_OF]->(b)
                            """,
                            data=chunk,
                        )

                relation_stats[level] = {
                    "east_of_expected": len(east_edges),
                    "north_of_expected": len(north_edges),
                }

        return relation_stats

    def print_summary(self, level_codes: Dict[int, Set[str]], relation_stats: Optional[Dict[int, Dict[str, int]]] = None) -> None:
        """
        @param level_codes 层级编码集合 / Level code sets
        """
        print("\n" + "=" * 72)
        print("构建完成摘要 / Build summary")
        print("=" * 72)
        total = 0
        for level in range(1, 33):
            cnt = len(level_codes.get(level, set()))
            total += cnt
            if cnt > 0:
                if relation_stats and level in relation_stats:
                    eo = relation_stats[level]["east_of_expected"]
                    no = relation_stats[level]["north_of_expected"]
                    print(f"L{level:02d}: {cnt:>10} 节点 / nodes, EAST_OF={eo:>10}, NORTH_OF={no:>10}")
                else:
                    print(f"L{level:02d}: {cnt:>10} 个网格节点 / grid nodes")
        print("-" * 72)
        print(f"总网格节点（去重后按层累计）: {total} / Total grid nodes (sum over levels after dedup): {total}")
        print("=" * 72)


def main() -> None:
    """
    @description 主函数 / Main function
    """
    parser = argparse.ArgumentParser(
        description="GeoSOT(G0) 网格节点与 EAST_OF/NORTH_OF 关系一键构建工具 / One-click GeoSOT(G0) grid nodes + EAST_OF/NORTH_OF builder"
    )
    parser.add_argument("--input", help="编码 CSV 文件路径（来自 gbt40087_encoder.py） / Encoded CSV path (from gbt40087_encoder.py)")
    parser.add_argument("--code-column", default="main_code", help="主码列名（默认 main_code） / Main-code column (default main_code)")
    parser.add_argument(
        "--template-level-grids",
        action="store_true",
        help="层内模板模式：每层仅创建 0/1/2/3 四个网格节点（推荐先检查结构） / Per-level template mode: only create digits 0/1/2/3 per level (recommended for structure checks)",
    )
    parser.add_argument("--seed-region-china", action="store_true", help="不依赖 CSV，按中国范围预生成网格结构 / Seed grid structure over China extent without CSV")
    parser.add_argument("--min-level", type=int, default=1, help="生成/处理最小层级（默认1） / Min level to generate/process (default 1)")
    parser.add_argument("--max-level", type=int, default=12, help="生成/处理最大层级（默认12，建议先检查结构） / Max level to generate/process (default 12; check structure first)")
    parser.add_argument("--force-large", action="store_true", help="允许生成较高层级（max-level>16） / Allow higher levels (max-level>16)")
    parser.add_argument("--uri", default=NEO4J_URI, help=f"Neo4j URI（默认 {NEO4J_URI}） / Neo4j URI (default {NEO4J_URI})")
    parser.add_argument("--user", default=NEO4J_USER, help=f"Neo4j 用户名（默认 {NEO4J_USER}） / Neo4j username (default {NEO4J_USER})")
    parser.add_argument("--password", default=NEO4J_PASSWORD, help="Neo4j 密码 / Neo4j password")
    parser.add_argument("--database", default=NEO4J_DATABASE, help=f"Neo4j 数据库名（默认 {NEO4J_DATABASE}） / Neo4j database (default {NEO4J_DATABASE})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help=f"批量大小（默认 {BATCH_SIZE}） / Batch size (default {BATCH_SIZE})")
    parser.add_argument("--skip-constraints", action="store_true", help="跳过约束创建 / Skip constraint creation")
    parser.add_argument("--nodes-only", action="store_true", help="仅创建节点，不创建 EAST_OF/NORTH_OF / Create nodes only; skip EAST_OF/NORTH_OF")

    args = parser.parse_args()

    if args.template_level_grids:
        pass
    elif args.seed_region_china:
        if args.max_level > 16 and not args.force_large:
            raise ValueError("max-level > 16 可能产生海量节点，请确认后添加 --force-large / max-level > 16 may create huge graphs; add --force-large to confirm")
    else:
        if not args.input:
            raise ValueError("非 seed 模式必须提供 --input / Non-seed mode requires --input")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"输入文件不存在: {args.input} / Input file does not exist: {args.input}")

    builder = GeoSotGridRelationBuilder(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
    )

    try:
        print("=" * 72)
        print("GeoSOT 网格节点与方向关系构建 / GeoSOT grid nodes and direction relations build")
        print("=" * 72)
        if args.template_level_grids:
            print(f"模式: 层内模板生成（每层 0/1/2/3，L{args.min_level}~L{args.max_level}） / Mode: per-level template (0/1/2/3, L{args.min_level}~L{args.max_level})")
        elif args.seed_region_china:
            print(f"模式: 中国范围种子生成（L{args.min_level}~L{args.max_level}） / Mode: China-extent seed (L{args.min_level}~L{args.max_level})")
        else:
            print(f"CSV: {args.input}")
        print(f"Neo4j: {args.uri} / {args.database}")
        print(f"主码列: {args.code_column} / Main-code column: {args.code_column}")

        if not builder.test_connection():
            raise ConnectionError("Neo4j 连接失败，请检查 URI/账号/密码 / Neo4j connection failed; check URI/user/password")

        if not args.skip_constraints:
            print("\n创建唯一约束... / Creating unique constraints...")
            builder.create_constraints()
            print("[OK] 约束创建完成 / Constraints created")

        print("\n收集各层级网格编码... / Collecting per-level grid codes...")
        if args.template_level_grids:
            level_codes = builder.collect_level_codes_template(
                min_level=args.min_level,
                max_level=args.max_level,
            )
        elif args.seed_region_china:
            level_codes = builder.collect_level_codes_from_region(
                min_level=args.min_level,
                max_level=args.max_level,
                region_bbox=CHINA_BBOX,
            )
        else:
            level_codes = builder.collect_level_codes_from_csv(args.input, code_column=args.code_column)
            # CSV 模式下只保留指定层级范围，便于结构检查 / In CSV mode keep only selected levels for structure checks
            level_codes = {
                lv: codes for lv, codes in level_codes.items() if args.min_level <= lv <= args.max_level
            }

        print("\n创建网格节点... / Creating grid nodes...")
        builder.create_grid_nodes(level_codes, batch_size=args.batch)

        relation_stats: Optional[Dict[int, Dict[str, int]]] = None
        if not args.nodes_only:
            print("\n创建 EAST_OF / NORTH_OF 关系... / Creating EAST_OF / NORTH_OF relations...")
            relation_stats = builder.create_direction_relations(
                level_codes,
                batch_size=args.batch,
                local_digit_mode=args.template_level_grids,
            )

        builder.print_summary(level_codes, relation_stats=relation_stats)
        print("\n[OK] 一键构建完成 / One-click build completed")

    finally:
        builder.close()


if __name__ == "__main__":
    main()

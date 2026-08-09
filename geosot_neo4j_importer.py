#!/usr/bin/env python3
# coding: utf-8
"""
GeoSOT 编码 CSV 导入 Neo4j 工具
GeoSOT encoded CSV import tool for Neo4j

功能 / Features:
1. 将 CSV 中主码导入为 (:GeoSOTCode {main_code})
   Import main codes from CSV as (:GeoSOTCode {main_code})
2. 基于主码逐层建立 LOCATED_IN_01GRID ~ LOCATED_IN_32GRID
   Build LOCATED_IN_01GRID ~ LOCATED_IN_32GRID layer by layer from main codes
3. 读取 vertex_codes_json，创建顶点节点并建立 HAS_VERTEX / VERTEX_IN_xxGRID
   Read vertex_codes_json, create vertex nodes and HAS_VERTEX / VERTEX_IN_xxGRID
4. 支持手动指定：
   Supports manually specifying:
   - 哪些字段作为 GeoSOTCode 属性 / Which fields become GeoSOTCode properties
   - 哪些字段作为独立标签节点并建立关系 / Which fields become separate label nodes with relations
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase
from tqdm import tqdm


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")
BATCH_SIZE = 5000


def normalize_code(code: str) -> str:
    """
    @param code 原始编码 / Raw code
    @returns 规范化编码 / Normalized code
    """
    return str(code).strip().upper()


def is_valid_g0_code(code: str) -> bool:
    """
    @param code 编码 / Code
    @returns 是否为合法 G0 编码 / Whether it is a valid G0 code
    """
    if not code or not code.startswith("G"):
        return False
    body = code[1:]
    if len(body) < 1 or len(body) > 32:
        return False
    if body[0] != "0":
        return False
    return all(c in "0123" for c in body)


def parse_vertex_codes(value: Any) -> List[str]:
    """
    @param value vertex_codes_json 字段值 / vertex_codes_json field value
    @returns 顶点编码列表 / List of vertex codes
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_code(v) for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    try:
        arr = json.loads(s)
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    return [normalize_code(v) for v in arr if str(v).strip()]


def to_safe_label(name: str) -> str:
    """
    @param name 字段名 / Field name
    @returns 可用作 Neo4j Label 的名称 / Name safe to use as a Neo4j Label
    """
    raw = "".join(ch if ch.isalnum() else "_" for ch in str(name))
    raw = raw.strip("_")
    if not raw:
        raw = "Field"
    if raw[0].isdigit():
        raw = f"F_{raw}"
    return raw[0].upper() + raw[1:]


def to_safe_rel(name: str) -> str:
    """
    @param name 字段名 / Field name
    @returns 可用作关系类型的名称 / Name safe to use as a relationship type
    """
    raw = "".join(ch if ch.isalnum() else "_" for ch in str(name).upper())
    raw = raw.strip("_")
    if not raw:
        raw = "FIELD"
    if raw[0].isdigit():
        raw = f"F_{raw}"
    return raw


def parse_source_spec(spec: str) -> Optional[Dict[str, str]]:
    """
    解析 --source 参数，格式： / Parse --source argument; formats:
      Buildings              -> Label=Source, prop=name, value=Buildings
      Source:name:Buildings  -> Label=Source, prop=name, value=Buildings
    @param spec 源节点规格 / Source node specification
    @returns 若为空则返回 None，否则返回 {label, prop, value} / None if empty, else {label, prop, value}
    """
    if not spec or not str(spec).strip():
        return None
    s = str(spec).strip()
    parts = [p.strip() for p in s.split(":") if p.strip()]
    if len(parts) == 1:
        return {"label": "Source", "prop": "name", "value": parts[0]}
    if len(parts) >= 3:
        return {"label": to_safe_label(parts[0]), "prop": parts[1], "value": parts[2]}
    if len(parts) == 2:
        return {"label": to_safe_label(parts[0]), "prop": "name", "value": parts[1]}
    return None


def parse_label_map(spec: str) -> List[Dict[str, str]]:
    """
    @param spec 字符串，格式： / Spec string formats:
      field:Label:prop,field2:Label2:prop2
      或 field:Label（prop 默认 value） / or field:Label (prop defaults to value)
      或 field（Label 自动推断，prop 默认 value） / or field (Label inferred, prop defaults to value)
    @returns 映射列表 / Mapping list
    """
    items: List[Dict[str, str]] = []
    if not spec:
        return items

    for token in [x.strip() for x in spec.split(",") if x.strip()]:
        parts = [p.strip() for p in token.split(":") if p.strip()]
        if len(parts) == 1:
            field = parts[0]
            label = to_safe_label(field)
            prop = "value"
        elif len(parts) == 2:
            field, label = parts
            prop = "value"
        else:
            field, label, prop = parts[0], parts[1], parts[2]

        items.append(
            {
                "field": field,
                "label": to_safe_label(label),
                "prop": prop,
                "rel_type": f"HAS_{to_safe_rel(field)}",
            }
        )

    return items


class GeoSotNeo4jImporter:
    """
    GeoSOT Neo4j 导入器 / GeoSOT Neo4j importer
    """

    def __init__(self, uri: str, user: str, password: str, database: str):
        """
        @param uri Neo4j URI
        @param user 用户名 / Username
        @param password 密码 / Password
        @param database 数据库 / Database
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        """@description 关闭连接 / Close connection"""
        self.driver.close()

    def _session(self):
        """@returns 会话 / Session"""
        return self.driver.session(database=self.database)

    def test_connection(self) -> bool:
        """@returns 是否连接成功 / Whether connection succeeded"""
        try:
            with self._session() as session:
                r = session.run("RETURN 1 AS v").single()
                return r and r["v"] == 1
        except Exception:
            return False

    def create_constraints(
        self, label_maps: List[Dict[str, str]], source_spec: Optional[Dict[str, str]] = None
    ) -> None:
        """
        @param label_maps 字段到标签节点映射 / Field-to-label-node mappings
        @param source_spec --source 解析结果，若提供则为此节点建唯一约束 / Parsed --source; if set, create unique constraint for it
        """
        with self._session() as session:
            # 主码节点 / Main-code nodes
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (n:GeoSOTCode) REQUIRE n.geosot_code_id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS "
                "FOR (n:GeoSOTCode) ON (n.main_code)"
            )
            # 顶点节点 / Vertex nodes
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (n:GeoSOTVertexCode) REQUIRE n.vertex_code IS UNIQUE"
            )
            # 映射标签节点唯一约束 / Unique constraints for mapped label nodes
            for m in label_maps:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{m['label']}) REQUIRE n.{m['prop']} IS UNIQUE"
                )
            # 源节点（--source）唯一约束 / Unique constraint for source node (--source)
            if source_spec:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{source_spec['label']}) REQUIRE n.{source_spec['prop']} IS UNIQUE"
                )

    def load_rows(self, csv_path: str) -> List[Dict[str, Any]]:
        """
        @param csv_path CSV 路径 / CSV path
        @returns 全部行 / All rows
        """
        # 放宽 CSV 单字段大小限制，避免 vertex_codes_json 等长字段触发 131072 上限
        # Raise CSV field size limit to avoid 131072 cap on long fields like vertex_codes_json
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(10**7)

        encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
        last_error: Optional[Exception] = None

        for enc in encodings:
            try:
                with open(csv_path, "r", encoding=enc, newline="") as f:
                    return list(csv.DictReader(f))
            except Exception as e:
                last_error = e
                continue

        raise ValueError(
            f"CSV 读取失败，尝试编码 {encodings} 均失败。最后错误: {last_error}"
            f" / CSV read failed for encodings {encodings}. Last error: {last_error}"
        )

    def fetch_existing_max_seq(self, main_codes: Set[str]) -> Dict[str, int]:
        """
        @param main_codes 待导入 main_code 集合 / Set of main_code values to import
        @returns 每个 main_code 当前数据库中的最大顺序码 / Max sequence number per main_code in DB
        """
        if not main_codes:
            return {}
        with self._session() as session:
            rows = session.run(
                """
                UNWIND $codes AS code
                MATCH (n:GeoSOTCode {main_code: code})
                WITH code, n, right(n.geosot_code_id, 4) AS sfx
                WITH code, CASE WHEN sfx =~ '\\\\d{4}' THEN toInteger(sfx) ELSE -1 END AS seq
                RETURN code AS main_code, max(seq) AS max_seq
                """,
                codes=list(main_codes),
            )
            result = {}
            for r in rows:
                result[r["main_code"]] = int(r["max_seq"]) if r["max_seq"] is not None else -1
            return result

    def build_import_records(
        self,
        rows: List[Dict[str, Any]],
        code_column: str,
        vertex_column: str,
        props_fields: List[str],
    ) -> List[Dict[str, Any]]:
        """
        @param rows CSV 行数据 / CSV row data
        @param code_column 主码列 / Main-code column
        @param vertex_column 顶点编码列 / Vertex-code column
        @param props_fields 属性字段 / Property fields
        @returns 标准化导入记录 / Normalized import records
        """
        valid_rows: List[Dict[str, Any]] = []
        main_codes: Set[str] = set()
        for row in rows:
            main_code = normalize_code(row.get(code_column, ""))
            if not is_valid_g0_code(main_code):
                continue
            valid_rows.append(row)
            main_codes.add(main_code)

        existing_max = self.fetch_existing_max_seq(main_codes)
        local_counter: Dict[str, int] = {}
        records: List[Dict[str, Any]] = []

        for row in valid_rows:
            main_code = normalize_code(row.get(code_column, ""))
            if main_code not in local_counter:
                local_counter[main_code] = existing_max.get(main_code, -1) + 1
            seq = local_counter[main_code]
            if seq > 9999:
                raise ValueError(
                    f"main_code={main_code} 顺序码超过 9999，无法继续分配"
                    f" / main_code={main_code} sequence exceeds 9999; cannot allocate more"
                )
            local_counter[main_code] += 1

            geosot_code_id = f"{main_code}{seq:04d}"
            props: Dict[str, Any] = {}
            for p in props_fields:
                if p in row and row[p] not in (None, ""):
                    props[p] = row[p]

            records.append(
                {
                    "main_code": main_code,
                    "seq_code": f"{seq:04d}",
                    "geosot_code_id": geosot_code_id,
                    "props": props,
                    "vertex_codes": parse_vertex_codes(row.get(vertex_column)),
                    "row": row,
                }
            )
        return records

    def create_geosot_nodes(
        self,
        records: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """
        @param records 导入记录 / Import records
        @param batch_size 批量大小 / Batch size
        @returns 导入记录（原样返回，便于后续串联） / Import records (returned as-is for chaining)
        """
        items = []
        for r in records:
            node_props = {
                "geosot_code_id": r["geosot_code_id"],
                "main_code": r["main_code"],
                "seq_code": r["seq_code"],
            }
            node_props.update(r["props"])
            items.append(node_props)

        total_batches = (len(items) + batch_size - 1) // batch_size if items else 0
        with self._session() as session:
            for i in tqdm(range(0, len(items), batch_size), desc="创建 GeoSOTCode / Create GeoSOTCode", total=total_batches):
                chunk = items[i : i + batch_size]
                session.run(
                    """
                    UNWIND $data AS item
                    MERGE (n:GeoSOTCode {geosot_code_id: item.geosot_code_id})
                    SET n += item
                    """,
                    data=chunk,
                )

        return records

    def create_field_label_nodes_and_relations(
        self,
        records: List[Dict[str, Any]],
        label_maps: List[Dict[str, str]],
        batch_size: int,
    ) -> None:
        """
        @param records 导入记录 / Import records
        @param label_maps 标签映射 / Label mappings
        @param batch_size 批量大小 / Batch size
        """
        if not label_maps:
            return

        with self._session() as session:
            for m in label_maps:
                pairs: List[Dict[str, Any]] = []
                for r in records:
                    val = r["row"].get(m["field"])
                    if val in (None, ""):
                        continue
                    pairs.append({"geosot_code_id": r["geosot_code_id"], "value": val})

                if not pairs:
                    continue

                # 创建标签节点 / Create label nodes
                uniq_vals = {}
                for p in pairs:
                    uniq_vals[str(p["value"])] = p["value"]
                val_data = [{"value": v} for v in uniq_vals.values()]
                total_batches = (len(val_data) + batch_size - 1) // batch_size
                for i in tqdm(range(0, len(val_data), batch_size), desc=f"创建 {m['label']} 节点 / Create {m['label']} nodes", total=total_batches, leave=False):
                    chunk = val_data[i : i + batch_size]
                    session.run(
                        f"""
                        UNWIND $data AS item
                        MERGE (n:{m['label']} {{{m['prop']}: item.value}})
                        """,
                        data=chunk,
                    )

                # 创建关系 / Create relationships
                total_batches = (len(pairs) + batch_size - 1) // batch_size
                for i in tqdm(range(0, len(pairs), batch_size), desc=f"创建 {m['rel_type']} 关系 / Create {m['rel_type']} rels", total=total_batches, leave=False):
                    chunk = pairs[i : i + batch_size]
                    session.run(
                        f"""
                        UNWIND $data AS item
                        MATCH (g:GeoSOTCode {{geosot_code_id: item.geosot_code_id}})
                        MATCH (n:{m['label']} {{{m['prop']}: item.value}})
                        MERGE (g)-[:{m['rel_type']}]->(n)
                        """,
                        data=chunk,
                    )

    def create_source_relation(
        self,
        records: List[Dict[str, Any]],
        source_spec: Dict[str, str],
        batch_size: int,
    ) -> None:
        """
        创建固定的源节点，并使所有 GeoSOTCode 指向该节点。
        Create a fixed source node and point all GeoSOTCode nodes to it.
        例如 --source Buildings 会创建 (:Source {name: "Buildings"})，并建立 (GeoSOTCode)-[:FROM_SOURCE]->(:Source)。
        e.g. --source Buildings creates (:Source {name: "Buildings"}) and (GeoSOTCode)-[:FROM_SOURCE]->(:Source).
        @param records 导入记录 / Import records
        @param source_spec {label, prop, value} 源节点规格 / Source node specification
        @param batch_size 批量大小 / Batch size
        """
        label = source_spec["label"]
        prop = source_spec["prop"]
        value = source_spec["value"]
        with self._session() as session:
            session.run(
                f"MERGE (s:{label} {{{prop}: $value}})",
                value=value,
            )
        geosot_ids = [r["geosot_code_id"] for r in records]
        with self._session() as session:
            for i in tqdm(range(0, len(geosot_ids), batch_size), desc=f"创建 FROM_SOURCE -> {label} / Create FROM_SOURCE -> {label}", leave=False):
                chunk = geosot_ids[i : i + batch_size]
                session.run(
                    f"""
                    UNWIND $ids AS geosot_code_id
                    MATCH (g:GeoSOTCode {{geosot_code_id: geosot_code_id}})
                    MATCH (s:{label} {{{prop}: $value}})
                    MERGE (g)-[:FROM_SOURCE]->(s)
                    """,
                    ids=chunk,
                    value=value,
                )

    def create_main_grid_relations(
        self,
        records: List[Dict[str, Any]],
        batch_size: int,
    ) -> None:
        """
        @param records 导入记录 / Import records
        @param batch_size 批量大小 / Batch size
        """
        per_level_data: Dict[int, List[Dict[str, str]]] = defaultdict(list)

        for r in records:
            code = r["main_code"]
            body = code[1:]  # 长度=层级 / length = level
            max_level = min(32, len(body))
            for lv in range(1, max_level + 1):
                digit = body[lv - 1]
                per_level_data[lv].append({"geosot_code_id": r["geosot_code_id"], "digit": digit})

        with self._session() as session:
            for lv in range(1, 33):
                rows = per_level_data.get(lv, [])
                if not rows:
                    continue
                label = f"grid{lv:02d}Code"
                prop = f"grid_L{lv}_code"
                rel = f"LOCATED_IN_{lv:02d}GRID"

                total_batches = (len(rows) + batch_size - 1) // batch_size
                for i in tqdm(range(0, len(rows), batch_size), desc=f"创建 {rel} / Create {rel}", total=total_batches, leave=False):
                    chunk = rows[i : i + batch_size]
                    session.run(
                        f"""
                        UNWIND $data AS item
                        MATCH (g:GeoSOTCode {{geosot_code_id: item.geosot_code_id}})
                        MATCH (grid:{label} {{{prop}: item.digit}})
                        MERGE (g)-[:{rel}]->(grid)
                        """,
                        data=chunk,
                    )

    def create_vertex_nodes_and_relations(
        self,
        records: List[Dict[str, Any]],
        batch_size: int,
    ) -> None:
        """
        @param records 导入记录 / Import records
        @param batch_size 批量大小 / Batch size
        """
        vertex_set: Set[str] = set()
        has_vertex: List[Dict[str, Any]] = []

        for r in records:
            vlist = r["vertex_codes"]
            for idx, vc in enumerate(vlist):
                if not is_valid_g0_code(vc):
                    continue
                vertex_set.add(vc)
                has_vertex.append(
                    {
                        "geosot_code_id": r["geosot_code_id"],
                        "vertex_code": vc,
                        "vertex_index": idx,
                    }
                )

        vertex_items = [{"vertex_code": v} for v in sorted(vertex_set)]

        with self._session() as session:
            # 顶点节点 / Vertex nodes
            total_batches = (len(vertex_items) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(vertex_items), batch_size), desc="创建 GeoSOTVertexCode / Create GeoSOTVertexCode", total=total_batches):
                chunk = vertex_items[i : i + batch_size]
                session.run(
                    """
                    UNWIND $data AS item
                    MERGE (v:GeoSOTVertexCode {vertex_code: item.vertex_code})
                    SET v += item
                    """,
                    data=chunk,
                )

            # HAS_VERTEX
            total_batches = (len(has_vertex) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(has_vertex), batch_size), desc="创建 HAS_VERTEX / Create HAS_VERTEX", total=total_batches):
                chunk = has_vertex[i : i + batch_size]
                session.run(
                    """
                    UNWIND $data AS item
                    MATCH (g:GeoSOTCode {geosot_code_id: item.geosot_code_id})
                    MATCH (v:GeoSOTVertexCode {vertex_code: item.vertex_code})
                    MERGE (g)-[r:HAS_VERTEX]->(v)
                    SET r.vertex_index = item.vertex_index
                    """,
                    data=chunk,
                )

        # VERTEX_IN_xxGRID
        # 使用“按层级流式分批写入”方式，避免一次性构建巨大的 per_level_data 导致内存爆炸。
        # Stream writes level-by-level in batches to avoid building a huge per_level_data in memory.
        level_counts: Dict[int, int] = defaultdict(int)
        for vc in vertex_set:
            max_level = min(32, len(vc) - 1)
            for lv in range(1, max_level + 1):
                level_counts[lv] += 1

        with self._session() as session:
            for lv in range(1, 33):
                rows_count = level_counts.get(lv, 0)
                if rows_count <= 0:
                    continue

                label = f"grid{lv:02d}Code"
                prop = f"grid_L{lv}_code"
                rel = f"VERTEX_IN_{lv:02d}GRID"
                total_batches = (rows_count + batch_size - 1) // batch_size

                chunk: List[Dict[str, str]] = []
                pbar = tqdm(total=total_batches, desc=f"创建 {rel} / Create {rel}", leave=False)
                for vc in vertex_set:
                    body = vc[1:]
                    if len(body) < lv:
                        continue
                    chunk.append({"vertex_code": vc, "digit": body[lv - 1]})
                    if len(chunk) >= batch_size:
                        session.run(
                            f"""
                            UNWIND $data AS item
                            MATCH (v:GeoSOTVertexCode {{vertex_code: item.vertex_code}})
                            MATCH (grid:{label} {{{prop}: item.digit}})
                            MERGE (v)-[:{rel}]->(grid)
                            """,
                            data=chunk,
                        )
                        chunk = []
                        pbar.update(1)

                if chunk:
                    session.run(
                        f"""
                        UNWIND $data AS item
                        MATCH (v:GeoSOTVertexCode {{vertex_code: item.vertex_code}})
                        MATCH (grid:{label} {{{prop}: item.digit}})
                        MERGE (v)-[:{rel}]->(grid)
                        """,
                        data=chunk,
                    )
                    pbar.update(1)
                pbar.close()


def main() -> None:
    """
    @description 主函数 / Main function
    """
    parser = argparse.ArgumentParser(description="GeoSOT 编码 CSV 导入 Neo4j / Import GeoSOT encoded CSV into Neo4j")
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径 / Input CSV file path")
    parser.add_argument("--code-column", default="main_code", help="主码列名（默认 main_code） / Main-code column (default main_code)")
    parser.add_argument("--vertex-column", default="vertex_codes_json", help="顶点编码列名（默认 vertex_codes_json） / Vertex-code column (default vertex_codes_json)")
    parser.add_argument(
        "--props",
        default="",
        help="作为 GeoSOTCode 属性写入的字段，逗号分隔。如 feature_name,geometry_type,main_lon,main_lat / Fields written as GeoSOTCode properties, comma-separated",
    )
    parser.add_argument(
        "--label-map",
        default="",
        help=(
            "字段映射为标签节点。格式：field:Label:prop,field2:Label2:prop2 "
            "例如：geometry_type:GeoFeatureType:type_name,source_name:DataSource:source_name"
            " / Map fields to label nodes. Format: field:Label:prop,... "
            "e.g. geometry_type:GeoFeatureType:type_name,source_name:DataSource:source_name"
        ),
    )
    parser.add_argument(
        "--source",
        default="",
        help=(
            "数据源节点：使所有导入的 GeoSOTCode 指向固定的 Source 节点。"
            "格式：Buildings（即 Source:name:Buildings）或 Source:name:Buildings"
            " / Source node: point all imported GeoSOTCode to a fixed Source node. "
            "Format: Buildings (i.e. Source:name:Buildings) or Source:name:Buildings"
        ),
    )
    parser.add_argument("--uri", default=NEO4J_URI, help=f"Neo4j URI（默认 {NEO4J_URI}） / Neo4j URI (default {NEO4J_URI})")
    parser.add_argument("--user", default=NEO4J_USER, help=f"Neo4j 用户名（默认 {NEO4J_USER}） / Neo4j username (default {NEO4J_USER})")
    parser.add_argument("--password", default=NEO4J_PASSWORD, help="Neo4j 密码 / Neo4j password")
    parser.add_argument("--database", default=NEO4J_DATABASE, help=f"数据库名（默认 {NEO4J_DATABASE}） / Database name (default {NEO4J_DATABASE})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help=f"批量大小（默认 {BATCH_SIZE}） / Batch size (default {BATCH_SIZE})")
    parser.add_argument("--skip-constraints", action="store_true", help="跳过约束创建 / Skip constraint creation")
    parser.add_argument("--list-columns", action="store_true", help="仅列出 CSV 列名 / List CSV column names only")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input} / Input file does not exist: {args.input}")

    props_fields = [p.strip() for p in args.props.split(",") if p.strip()]
    label_maps = parse_label_map(args.label_map)
    source_spec = parse_source_spec(args.source)

    importer = GeoSotNeo4jImporter(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
    )

    try:
        if not importer.test_connection():
            raise ConnectionError("Neo4j 连接失败，请检查参数 / Neo4j connection failed; check parameters")

        rows = importer.load_rows(args.input)
        if not rows:
            raise ValueError("CSV 无有效数据 / CSV has no valid data")

        if args.list_columns:
            cols = list(rows[0].keys())
            print("CSV 列名： / CSV columns:")
            for i, c in enumerate(cols, start=1):
                print(f"{i:3d}. {c}")
            return

        print("=" * 72)
        print("GeoSOT CSV -> Neo4j 导入 / GeoSOT CSV -> Neo4j import")
        print("=" * 72)
        print(f"CSV: {args.input}")
        print(f"主码列: {args.code_column} / Main-code column: {args.code_column}")
        print(f"顶点列: {args.vertex_column} / Vertex column: {args.vertex_column}")
        print(f"GeoSOTCode 属性字段: {props_fields if props_fields else '(无)'} / GeoSOTCode prop fields: {props_fields if props_fields else '(none)'}")
        print(f"标签映射: {label_maps if label_maps else '(无)'} / Label maps: {label_maps if label_maps else '(none)'}")
        print(f"数据源节点: {source_spec or '(无)'} / Source node: {source_spec or '(none)'}")
        print(f"Neo4j: {args.uri} / {args.database}")
        print("-" * 72)

        if not args.skip_constraints:
            print("创建约束... / Creating constraints...")
            importer.create_constraints(label_maps, source_spec)
            print("[OK] 约束完成 / Constraints done")

        records = importer.build_import_records(
            rows=rows,
            code_column=args.code_column,
            vertex_column=args.vertex_column,
            props_fields=props_fields,
        )
        if not records:
            raise ValueError("没有可导入的有效 main_code / No valid main_code to import")

        # 1) GeoSOTCode
        records = importer.create_geosot_nodes(
            records=records,
            batch_size=args.batch,
        )
        print(f"[OK] GeoSOTCode 节点: {len(records)} / GeoSOTCode nodes: {len(records)}")

        # 2) 标签映射节点 / Label-mapped nodes
        importer.create_field_label_nodes_and_relations(
            records=records,
            label_maps=label_maps,
            batch_size=args.batch,
        )
        if label_maps:
            print("[OK] 标签映射节点与关系完成 / Label-mapped nodes and relations done")

        # 2b) 数据源节点（--source） / Source node (--source)
        if source_spec:
            importer.create_source_relation(
                records=records,
                source_spec=source_spec,
                batch_size=args.batch,
            )
            print(f"[OK] FROM_SOURCE -> {source_spec['label']} 完成 / FROM_SOURCE -> {source_spec['label']} done")

        # 3) 主码 -> 网格 / Main code -> grid
        importer.create_main_grid_relations(
            records=records,
            batch_size=args.batch,
        )
        print("[OK] LOCATED_IN_01~32GRID 完成 / LOCATED_IN_01~32GRID done")

        # 4) 顶点 / Vertices
        importer.create_vertex_nodes_and_relations(
            records=records,
            batch_size=args.batch,
        )
        print("[OK] 顶点节点与 VERTEX_IN_01~32GRID 完成 / Vertex nodes and VERTEX_IN_01~32GRID done")

        print("\n[OK] 导入完成 / Import completed")

    finally:
        importer.close()


if __name__ == "__main__":
    main()

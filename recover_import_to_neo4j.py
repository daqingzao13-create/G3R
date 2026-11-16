#!/usr/bin/env python3
# coding: utf-8
"""
Spatial graph preparation, step 3/3: import encoded CSVs into Neo4j using the paper's
schema. Run recover_encode_from_pg.py and recover_build_levdigit_backbone.py first.

Feature nodes POI/Road/Parcel/Building share :SpatialFeature. CC nodes are
GeoSOTCentroidCode {cc_id, main_code, lon, lat, level}; VC nodes are GeoSOTVertexCode
{vertex_code}; LCPC nodes are GeoSOTCommonPrefixCode {prefix_code, prefix_level}.
Geometry/source nodes are GeoFeatureType {type_name} and DataSource {name}.

Features connect through HAS_CENTROID_CODE, ordered HAS_VERTEX_CODE {seq}, and
HAS_COMMON_PREFIX_CODE. CC connects to HAS_GEOMETRY_TYPE and FROM_SOURCE. CC/VC/LCPC connect
to GeoSOTLevelDigit{k} via CC_HAS_LEVEL_{k}_DIGIT, VC_HAS_LEVEL_{k}_DIGIT, and
LCPC_HAS_LEVEL_{k}_DIGIT.

Stages selected by --stages use idempotent MERGE: 0 constraints/indexes; 1
feature+CC+type+source; 2 VC and ordered edges; 3 LCPC; 4 CC affiliation; 5 VC affiliation;
6 LCPC affiliation.

Usage:
    python recover_import_to_neo4j.py
    python recover_import_to_neo4j.py --source POI
    python recover_import_to_neo4j.py --stages 0,1
    python recover_import_to_neo4j.py --limit 5000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from neo4j import GraphDatabase
from tqdm import tqdm


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

BATCH = 5000
MAX_LEVEL = 32

# Source name -> (feature label, CSV filename).
SOURCES: Dict[str, Tuple[str, str]] = {
    "POI": ("POI", "poi_new_encoded.csv"),
    "Road": ("Road", "road_encoded.csv"),
    "Statutory Plan": ("Parcel", "statutory_plan_encoded.csv"),
    "Buildings": ("Building", "building_new_encoded.csv"),
}

FEATURE_LABELS = ["POI", "Road", "Parcel", "Building"]


def code_body(code: str) -> str:
    """@returns The code body after removing the leading G; body length equals the level."""
    return code[1:] if code and code[0] in ("G", "g") else code


def iter_csv(path: str, limit: Optional[int]) -> Iterator[Dict[str, str]]:
    """Stream records from an encoded CSV."""
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10 ** 7)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            yield row


class Importer:
    def __init__(self, uri: str, user: str, password: str, database: str, batch: int):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.batch = batch

    def close(self) -> None:
        self.driver.close()

    def _session(self):
        return self.driver.session(database=self.database)

    def test_connection(self) -> bool:
        try:
            with self._session() as s:
                return s.run("RETURN 1 AS v").single()["v"] == 1
        except Exception:
            return False

    # ---------- Stage 0: constraints and indexes ----------
    def stage_constraints(self) -> None:
        stmts = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:GeoSOTCentroidCode) REQUIRE n.cc_id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (n:GeoSOTCentroidCode) ON (n.main_code)",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:GeoSOTVertexCode) REQUIRE n.vertex_code IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:GeoSOTCommonPrefixCode) REQUIRE n.prefix_code IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:GeoFeatureType) REQUIRE n.type_name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DataSource) REQUIRE n.name IS UNIQUE",
        ]
        for lbl in FEATURE_LABELS:
            stmts.append(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{lbl}) REQUIRE n.feature_id IS UNIQUE"
            )
            stmts.append(f"CREATE INDEX IF NOT EXISTS FOR (n:{lbl}) ON (n.main_code)")
        # Index prefix_level for LCPC fallback; prefix_code already has a uniqueness constraint.
        stmts.append(
            "CREATE INDEX IF NOT EXISTS FOR (n:GeoSOTCommonPrefixCode) ON (n.prefix_level)"
        )
        with self._session() as s:
            for st in stmts:
                s.run(st)
        print(f"[OK] 阶段0: 约束/索引 ({len(stmts)} 条)")

    # ---------- Shared batch execution ----------
    def _run_batches(
        self,
        cypher: str,
        rows: Iterable[dict],
        desc: str,
        total: Optional[int] = None,
    ) -> None:
        buf: List[dict] = []
        pbar = tqdm(total=total, desc=desc, file=sys.stdout, mininterval=0.5)
        with self._session() as s:
            for r in rows:
                buf.append(r)
                pbar.update(1)
                if len(buf) >= self.batch:
                    s.run(cypher, data=buf)
                    buf = []
            if buf:
                s.run(cypher, data=buf)
        pbar.close()

    # ---------- Stage 1: feature, CC, geometry type, and source ----------
    def stage_features(self, source: str, label: str, path: str, limit: Optional[int], total: int) -> None:
        cypher = f"""
        UNWIND $data AS item
        MERGE (f:`{label}` {{feature_id: item.feature_id}})
        SET f += item.props
        SET f.main_code = item.main_code
        SET f:SpatialFeature
        MERGE (c:GeoSOTCentroidCode {{cc_id: item.feature_id}})
        SET c.main_code = item.main_code,
            c.lon = item.lon, c.lat = item.lat,
            c.level = item.level, c.code_length = item.code_length
        MERGE (f)-[:HAS_CENTROID_CODE]->(c)
        MERGE (t:GeoFeatureType {{type_name: item.gtype}})
        MERGE (c)-[:HAS_GEOMETRY_TYPE]->(t)
        MERGE (s:DataSource {{name: item.source}})
        MERGE (c)-[:FROM_SOURCE]->(s)
        """

        def gen():
            for row in iter_csv(path, limit):
                mc = row["main_code"]
                props = {}
                pj = row.get("props_json") or ""
                if pj:
                    try:
                        props = json.loads(pj)
                    except Exception:
                        props = {}
                yield {
                    "feature_id": row["feature_id"],
                    "main_code": mc,
                    "props": props,
                    "lon": float(row["cc_lon"]) if row.get("cc_lon") else None,
                    "lat": float(row["cc_lat"]) if row.get("cc_lat") else None,
                    "level": int(row["level"]) if row.get("level") else None,
                    "code_length": len(mc),
                    "gtype": row["geometry_type"],
                    "source": source,
                }

        self._run_batches(cypher, gen(), desc=f"[1]要素+CC {source}", total=total)

    # ---------- Stage 2: VC and ordered HAS_VERTEX_CODE edges ----------
    def stage_vertices(self, label: str, path: str, limit: Optional[int], total: int) -> None:
        cypher = f"""
        UNWIND $data AS item
        MERGE (v:GeoSOTVertexCode {{vertex_code: item.vertex_code}})
        WITH item, v
        MATCH (f:`{label}` {{feature_id: item.feature_id}})
        MERGE (f)-[r:HAS_VERTEX_CODE {{seq: item.seq}}]->(v)
        """

        def gen():
            for row in iter_csv(path, limit):
                fid = row["feature_id"]
                try:
                    vlist = json.loads(row.get("vertex_codes_json") or "[]")
                except Exception:
                    vlist = []
                for idx, vc in enumerate(vlist):
                    yield {"feature_id": fid, "vertex_code": vc, "seq": idx}

        # The expanded row count is unknown; use an unbounded progress bar.
        self._run_batches(cypher, gen(), desc=f"[2]VC {label}", total=None)

    # ---------- Stage 3: LCPC and HAS_COMMON_PREFIX_CODE ----------
    def stage_lcpc(self, label: str, path: str, limit: Optional[int], total: int) -> None:
        cypher = f"""
        UNWIND $data AS item
        MERGE (p:GeoSOTCommonPrefixCode {{prefix_code: item.prefix_code}})
        ON CREATE SET p.prefix_level = item.prefix_level
        WITH item, p
        MATCH (f:`{label}` {{feature_id: item.feature_id}})
        MERGE (f)-[:HAS_COMMON_PREFIX_CODE]->(p)
        """

        def gen():
            for row in iter_csv(path, limit):
                lc = (row.get("lcpc_code") or "").strip()
                if not lc:
                    continue
                lv = row.get("lcpc_level") or ""
                yield {
                    "feature_id": row["feature_id"],
                    "prefix_code": lc,
                    "prefix_level": int(lv) if lv else (len(lc) - 1),
                }

        self._run_batches(cypher, gen(), desc=f"[3]LCPC {label}", total=None)

    # ---------- Stages 4/5/6: level affiliation, batched by level ----------
    def _affiliate(
        self,
        node_match: str,          # Example: "(c:GeoSOTCentroidCode {cc_id: item.key})".
        rel_prefix: str,          # Example: "CC_HAS_LEVEL_" -> CC_HAS_LEVEL_{k}_DIGIT.
        node_alias: str,          # Example node alias: "c".
        code_rows: Iterator[Tuple[str, str]],  # (node_key, code_string)
        desc: str,
        max_level: int,
    ) -> None:
        """
        Generate level affiliations for each (node_key, code) at levels 1..len(body); bucket by
        level and MERGE in batches.
        """
        buckets: Dict[int, List[dict]] = {lv: [] for lv in range(1, max_level + 1)}
        pbar = tqdm(desc=desc, file=sys.stdout, mininterval=0.5)

        def flush(session, lv: int) -> None:
            data = buckets[lv]
            if not data:
                return
            rel = f"{rel_prefix}{lv}_DIGIT"
            lbl = f"GeoSOTLevelDigit{lv}"
            cypher = f"""
            UNWIND $data AS item
            MATCH {node_match}
            MATCH (d:{lbl} {{digit: item.digit}})
            MERGE ({node_alias})-[:{rel}]->(d)
            """
            session.run(cypher, data=data)
            buckets[lv] = []

        with self._session() as s:
            for node_key, code in code_rows:
                body = code_body(code)
                depth = min(max_level, len(body))
                for lv in range(1, depth + 1):
                    buckets[lv].append({"key": node_key, "digit": body[lv - 1]})
                    if len(buckets[lv]) >= self.batch:
                        flush(s, lv)
                pbar.update(1)
            for lv in range(1, max_level + 1):
                flush(s, lv)
        pbar.close()

    def stage_cc_affiliation(self, path: str, limit: Optional[int], max_level: int) -> None:
        def gen() -> Iterator[Tuple[str, str]]:
            for row in iter_csv(path, limit):
                yield row["feature_id"], row["main_code"]

        self._affiliate(
            node_match="(c:GeoSOTCentroidCode {cc_id: item.key})",
            rel_prefix="CC_HAS_LEVEL_",
            node_alias="c",
            code_rows=gen(),
            desc="[4]CC 层级挂接",
            max_level=max_level,
        )

    def stage_vc_affiliation(self, path: str, limit: Optional[int], max_level: int) -> None:
        seen = set()

        def gen() -> Iterator[Tuple[str, str]]:
            for row in iter_csv(path, limit):
                try:
                    vlist = json.loads(row.get("vertex_codes_json") or "[]")
                except Exception:
                    vlist = []
                for vc in vlist:
                    if vc in seen:
                        continue
                    seen.add(vc)
                    yield vc, vc  # key == code == vertex_code

        self._affiliate(
            node_match="(v:GeoSOTVertexCode {vertex_code: item.key})",
            rel_prefix="VC_HAS_LEVEL_",
            node_alias="v",
            code_rows=gen(),
            desc="[5]VC 层级挂接(去重)",
            max_level=max_level,
        )

    def stage_lcpc_affiliation(self, path: str, limit: Optional[int], max_level: int) -> None:
        seen = set()

        def gen() -> Iterator[Tuple[str, str]]:
            for row in iter_csv(path, limit):
                lc = (row.get("lcpc_code") or "").strip()
                if not lc or lc in seen:
                    continue
                seen.add(lc)
                yield lc, lc

        self._affiliate(
            node_match="(p:GeoSOTCommonPrefixCode {prefix_code: item.key})",
            rel_prefix="LCPC_HAS_LEVEL_",
            node_alias="p",
            code_rows=gen(),
            desc="[6]LCPC 层级挂接(去重)",
            max_level=max_level,
        )


def count_csv_rows(path: str, limit: Optional[int]) -> int:
    n = 0
    for _ in iter_csv(path, limit):
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="恢复流程(3/3): 编码 CSV -> Neo4j (论文 schema)")
    parser.add_argument("--in-dir", default="recover_out", help="编码 CSV 目录 (默认 recover_out)")
    parser.add_argument("--source", default="", help="仅导入某源: POI/Road/'Statutory Plan'/Buildings")
    parser.add_argument("--stages", default="0,1,2,3,4,5,6", help="要执行的阶段, 逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="每源限量 (冒烟, 0=全量)")
    parser.add_argument("--max-level", type=int, default=MAX_LEVEL, help="层级挂接最大层 (默认 32)")
    parser.add_argument("--batch", type=int, default=BATCH, help="批量大小 (默认 5000)")
    parser.add_argument("--uri", default=NEO4J_URI)
    parser.add_argument("--user", default=NEO4J_USER)
    parser.add_argument("--password", default=NEO4J_PASSWORD)
    parser.add_argument("--database", default=NEO4J_DATABASE)
    args = parser.parse_args()

    stages = {int(x) for x in args.stages.split(",") if x.strip() != ""}
    limit = args.limit or None

    if args.source:
        if args.source not in SOURCES:
            raise ValueError(f"未知源 {args.source}; 可选 {list(SOURCES)}")
        src_list = [args.source]
    else:
        src_list = list(SOURCES)

    imp = Importer(args.uri, args.user, args.password, args.database, args.batch)
    try:
        print("=" * 72)
        print("恢复流程(3/3): 编码 CSV -> Neo4j (论文 schema)")
        print(f"Neo4j: {args.uri} / {args.database}  阶段: {sorted(stages)}  源: {src_list}")
        print("=" * 72)
        if not imp.test_connection():
            raise ConnectionError("Neo4j 连接失败, 请检查 URI/账号/密码/数据库")

        if 0 in stages:
            imp.stage_constraints()

        for source in src_list:
            label, fname = SOURCES[source]
            path = os.path.join(args.in_dir, fname)
            if not os.path.exists(path):
                print(f"[跳过] 缺少编码 CSV: {path} (先跑 recover_encode_from_pg.py)")
                continue

            print(f'\n--- Source: {source} (标签 {label})  文件: {path} ---')
            total = None
            if stages & {1}:  # Stage 1 uses the row count for an accurate progress bar.
                total = count_csv_rows(path, limit)
                print(f"    要素数: {total:,}")

            if 1 in stages:
                imp.stage_features(source, label, path, limit, total or 0)
            if 2 in stages:
                imp.stage_vertices(label, path, limit, total or 0)
            if 3 in stages:
                imp.stage_lcpc(label, path, limit, total or 0)
            if 4 in stages:
                imp.stage_cc_affiliation(path, limit, args.max_level)
            if 5 in stages:
                imp.stage_vc_affiliation(path, limit, args.max_level)
            if 6 in stages:
                imp.stage_lcpc_affiliation(path, limit, args.max_level)

        print("\n[OK] 导入完成。可用 recover_verify.py 或 QASystem/check_neo4j_data.py 校验。")
    finally:
        imp.close()


if __name__ == "__main__":
    main()

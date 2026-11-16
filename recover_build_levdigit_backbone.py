#!/usr/bin/env python3
# coding: utf-8
"""
Spatial graph preparation, step 2/3: construct the shared level-digit backbone used by the
paper.

For each level 1..32, create four GeoSOTLevelDigit{level} nodes with digit 0,1,2,3: 128
nodes total. Each digit maps to (lat_bit, lon_bit): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).

(A)-[:EAST_OF]->(B) means A.lon=B.lon+1 at the same latitude; (A)-[:NORTH_OF]->(B) means
A.lat=B.lat+1 at the same longitude. Thus 1 EAST_OF 0, 3 EAST_OF 2, 2 NORTH_OF 0, and 3
NORTH_OF 1.

Set Neo4j credentials through environment variables; no password is supplied by this release.
"""
from __future__ import annotations

import argparse
import os

from neo4j import GraphDatabase
from tqdm import tqdm


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

MAX_LEVEL = 32

# digit -> (lat_bit, lon_bit)
DIGIT_BITS = {"0": (0, 0), "1": (0, 1), "2": (1, 0), "3": (1, 1)}


def level_label(level: int) -> str:
    """@returns The GeoSOTLevelDigit{level} node label."""
    return f"GeoSOTLevelDigit{level}"


class BackboneBuilder:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

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

    def create_constraints(self, max_level: int) -> None:
        """Create a unique digit constraint per level; its index supports MATCH during affiliation."""
        with self._session() as s:
            for lv in range(1, max_level + 1):
                s.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{level_label(lv)}) REQUIRE n.digit IS UNIQUE"
                )

    def create_nodes(self, max_level: int) -> None:
        """Create four digit nodes (0/1/2/3) at each level."""
        with self._session() as s:
            for lv in tqdm(range(1, max_level + 1), desc="建 GeoSOTLevelDigit 节点"):
                data = []
                for d, (lat_bit, lon_bit) in DIGIT_BITS.items():
                    data.append({"digit": d, "level": lv, "lat_bit": lat_bit, "lon_bit": lon_bit})
                s.run(
                    f"""
                    UNWIND $data AS item
                    MERGE (n:{level_label(lv)} {{digit: item.digit}})
                    ON CREATE SET n.level = item.level,
                                  n.lat_bit = item.lat_bit,
                                  n.lon_bit = item.lon_bit
                    """,
                    data=data,
                )

    def create_direction_relations(self, max_level: int) -> None:
        """Create within-level EAST_OF and NORTH_OF relationships."""
        # Precompute same-level adjacency; all levels share the same 0/1/2/3 topology.
        digits = list(DIGIT_BITS.keys())
        pos = {d: DIGIT_BITS[d] for d in digits}  # d -> (lat, lon)
        east_edges = []   # (from, to): from lies east of to.
        north_edges = []  # (from, to): from lies north of to.
        for a in digits:
            la, loa = pos[a]
            for b in digits:
                lb, lob = pos[b]
                if la == lb and loa == lob + 1:
                    east_edges.append((a, b))
                if loa == lob and la == lb + 1:
                    north_edges.append((a, b))

        with self._session() as s:
            for lv in tqdm(range(1, max_level + 1), desc="建 EAST_OF/NORTH_OF"):
                lbl = level_label(lv)
                if east_edges:
                    s.run(
                        f"""
                        UNWIND $edges AS e
                        MATCH (a:{lbl} {{digit: e[0]}})
                        MATCH (b:{lbl} {{digit: e[1]}})
                        MERGE (a)-[:EAST_OF]->(b)
                        """,
                        edges=east_edges,
                    )
                if north_edges:
                    s.run(
                        f"""
                        UNWIND $edges AS e
                        MATCH (a:{lbl} {{digit: e[0]}})
                        MATCH (b:{lbl} {{digit: e[1]}})
                        MERGE (a)-[:NORTH_OF]->(b)
                        """,
                        edges=north_edges,
                    )

    def summary(self, max_level: int) -> None:
        with self._session() as s:
            total_nodes = 0
            total_e = 0
            total_n = 0
            for lv in range(1, max_level + 1):
                lbl = level_label(lv)
                c = s.run(f"MATCH (n:{lbl}) RETURN count(n) AS c").single()["c"]
                e = s.run(
                    f"MATCH (:{lbl})-[r:EAST_OF]->(:{lbl}) RETURN count(r) AS c"
                ).single()["c"]
                n = s.run(
                    f"MATCH (:{lbl})-[r:NORTH_OF]->(:{lbl}) RETURN count(r) AS c"
                ).single()["c"]
                total_nodes += c
                total_e += e
                total_n += n
        print("=" * 72)
        print(f"GeoSOTLevelDigit1..{max_level}: 节点 {total_nodes} (每层 4), "
              f"EAST_OF {total_e}, NORTH_OF {total_n}")
        print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="恢复流程(2/3): GeoSOT 层级-码元骨架")
    parser.add_argument("--max-level", type=int, default=MAX_LEVEL, help="最大层级 (默认 32)")
    parser.add_argument("--nodes-only", action="store_true", help="仅建节点, 不建方向关系")
    parser.add_argument("--skip-constraints", action="store_true", help="跳过约束创建")
    parser.add_argument("--uri", default=NEO4J_URI)
    parser.add_argument("--user", default=NEO4J_USER)
    parser.add_argument("--password", default=NEO4J_PASSWORD)
    parser.add_argument("--database", default=NEO4J_DATABASE)
    args = parser.parse_args()

    if args.max_level < 1 or args.max_level > 32:
        raise ValueError("max-level 必须在 1~32")

    b = BackboneBuilder(args.uri, args.user, args.password, args.database)
    try:
        print("=" * 72)
        print("恢复流程(2/3): 构建 GeoSOTLevelDigit 骨架 + EAST_OF/NORTH_OF")
        print(f"Neo4j: {args.uri} / {args.database}  max-level=L{args.max_level}")
        print("=" * 72)
        if not b.test_connection():
            raise ConnectionError("Neo4j 连接失败, 请检查 URI/账号/密码/数据库")

        if not args.skip_constraints:
            print("创建约束...")
            b.create_constraints(args.max_level)
        print("创建节点...")
        b.create_nodes(args.max_level)
        if not args.nodes_only:
            print("创建方向关系...")
            b.create_direction_relations(args.max_level)
        b.summary(args.max_level)
        print("[OK] 骨架构建完成")
    finally:
        b.close()


if __name__ == "__main__":
    main()

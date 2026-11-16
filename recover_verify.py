#!/usr/bin/env python3
# coding: utf-8
"""
Count nodes/relations under the paper's schema. Historical reference scale: approximately
1,660,949 CC, 7,768,659 VC, 237,805 LCPC, and 368 million relations. These are study-data
reference counts, not expected counts for arbitrary or synthetic inputs.

Usage: python recover_verify.py
"""
from __future__ import annotations

import logging
import os

from neo4j import GraphDatabase

# Suppress harmless notifications such as absent relationship types at unused LCPC levels.
logging.getLogger("neo4j").setLevel(logging.ERROR)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

NODE_LABELS = [
    "SpatialFeature", "POI", "Road", "Parcel", "Building",
    "GeoSOTCentroidCode", "GeoSOTVertexCode", "GeoSOTCommonPrefixCode",
    "GeoFeatureType", "DataSource",
]
SIMPLE_RELS = [
    "HAS_CENTROID_CODE", "HAS_VERTEX_CODE", "HAS_COMMON_PREFIX_CODE",
    "HAS_GEOMETRY_TYPE", "FROM_SOURCE", "EAST_OF", "NORTH_OF",
]


def count_node(s, label: str) -> int:
    return s.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()["c"]


def count_rel(s, rel: str) -> int:
    return s.run(f"MATCH ()-[r:`{rel}`]->() RETURN count(r) AS c").single()["c"]


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DATABASE) as s:
        print("=" * 72)
        print(f"恢复校验: {NEO4J_URI} / {NEO4J_DATABASE}")
        print("=" * 72)

        print("\n【节点】")
        for lbl in NODE_LABELS:
            print(f"  {lbl:24s}: {count_node(s, lbl):>14,}")
        digit_total = 0
        for k in range(1, 33):
            digit_total += count_node(s, f"GeoSOTLevelDigit{k}")
        print(f"  {'GeoSOTLevelDigit1..32':24s}: {digit_total:>14,}  (预期 128)")

        print("\n【基础关系】")
        for rel in SIMPLE_RELS:
            print(f"  {rel:26s}: {count_rel(s, rel):>14,}")

        print("\n【层级挂接关系 (按层求和)】")
        for prefix in ("CC_HAS_LEVEL_", "VC_HAS_LEVEL_", "LCPC_HAS_LEVEL_"):
            tot = 0
            for k in range(1, 33):
                tot += count_rel(s, f"{prefix}{k}_DIGIT")
            print(f"  {prefix + '*_DIGIT':26s}: {tot:>14,}")

        print("\n【每源要素数 (FROM_SOURCE)】")
        rows = s.run(
            """
            MATCH (c:GeoSOTCentroidCode)-[:FROM_SOURCE]->(d:DataSource)
            RETURN d.name AS source, count(c) AS c ORDER BY c DESC
            """
        )
        for r in rows:
            print(f"  {r['source']:24s}: {r['c']:>14,}")

        print("\n【完整性抽查】")
        no_cc = s.run(
            "MATCH (f:SpatialFeature) WHERE NOT (f)-[:HAS_CENTROID_CODE]->() "
            "RETURN count(f) AS c"
        ).single()["c"]
        print(f"  无 HAS_CENTROID_CODE 的要素: {no_cc:,} (应为 0)")

    driver.close()
    print("\n" + "=" * 72)
    print("校验完成")
    print("=" * 72)


if __name__ == "__main__":
    main()

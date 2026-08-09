"""
为 GeoSOTCode 与其关联的 GeoSOTVertexCode 计算最长公共前缀，
Compute the longest common prefix for each GeoSOTCode and its related GeoSOTVertexCode nodes,
并在 Neo4j 中创建 GeoSOTCommonPrefix 节点及 HAS_COMMON_PREFIX 关系，
create GeoSOTCommonPrefix nodes and HAS_COMMON_PREFIX relationships in Neo4j,
同时为 GeoSOTCommonPrefix 建立 LOCATED_IN_**GRID 网格层级关系。
and build LOCATED_IN_**GRID hierarchical grid relations for GeoSOTCommonPrefix.

仅处理拥有多于 1 个 HAS_VERTEX 关系的 GeoSOTCode 节点。
Only process GeoSOTCode nodes that have more than one HAS_VERTEX relationship.

图模型 / Graph model:
  (GeoSOTCode)-[:HAS_COMMON_PREFIX]->(GeoSOTCommonPrefix {prefix_code, prefix_level})
  (GeoSOTCommonPrefix)-[:LOCATED_IN_01GRID]->(grid01Code)
  (GeoSOTCommonPrefix)-[:LOCATED_IN_02GRID]->(grid02Code)
  ...
"""

import os
import time
import argparse
from collections import defaultdict
from neo4j import GraphDatabase
from tqdm import tqdm


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")


def longest_common_prefix(strings: list[str]) -> str:
    """计算多个字符串的最长公共前缀。 / Compute the longest common prefix of multiple strings."""
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for i, ch in enumerate(shortest):
        for s in strings:
            if s[i] != ch:
                return shortest[:i]
    return shortest


def fetch_geosotcode_with_vertices(session) -> list[dict]:
    """
    查询所有拥有 >1 个 HAS_VERTEX 关系的 GeoSOTCode，
    Query all GeoSOTCode nodes that have more than one HAS_VERTEX relationship,
    返回 main_code 及其所有 vertex_code 列表。
    and return each main_code with its list of vertex_code values.
    """
    result = session.run(
        """
        MATCH (g:GeoSOTCode)-[:HAS_VERTEX]->(v:GeoSOTVertexCode)
        WITH g, collect(v.vertex_code) AS vertex_codes
        WHERE size(vertex_codes) > 1
        RETURN g.main_code AS main_code, vertex_codes
        """
    )
    records = []
    for record in result:
        main_code = record["main_code"]
        vertex_codes = record["vertex_codes"]
        if main_code and vertex_codes:
            records.append({
                "main_code": main_code,
                "vertex_codes": vertex_codes,
            })
    return records


def compute_prefixes(records: list[dict]) -> list[dict]:
    """对每条记录计算 main_code 与所有 vertex_code 的最长公共前缀。 / For each record, compute LCP of main_code and all vertex_codes."""
    results = []
    for rec in records:
        all_codes = [rec["main_code"]] + rec["vertex_codes"]
        prefix = longest_common_prefix(all_codes)
        if prefix:
            results.append({
                "main_code": rec["main_code"],
                "prefix_code": prefix,
                "prefix_level": len(prefix),
                "vertex_count": len(rec["vertex_codes"]),
            })
    return results


def create_prefix_nodes(session, batch: list[dict]):
    """批量创建 GeoSOTCommonPrefix 节点并建立 HAS_COMMON_PREFIX 关系。 / Batch-create GeoSOTCommonPrefix nodes and HAS_COMMON_PREFIX relations."""
    session.run(
        """
        UNWIND $data AS item
        MATCH (g:GeoSOTCode {main_code: item.main_code})
        MERGE (p:GeoSOTCommonPrefix {prefix_code: item.prefix_code})
        ON CREATE SET p.prefix_level = item.prefix_level
        MERGE (g)-[:HAS_COMMON_PREFIX]->(p)
        """,
        data=batch,
    )


def build_grid_relations(prefix_data: list[dict]) -> dict[int, list[dict]]:
    """
    根据 prefix_code 的 body 部分（去掉首字符 G），按层级收集需要建立的
    From the body of prefix_code (strip leading G), collect per-level data needed to build
    LOCATED_IN_**GRID 关系数据。
    LOCATED_IN_**GRID relationships.
    返回 {level: [{"prefix_code": ..., "digit": ...}, ...]}
    Returns {level: [{"prefix_code": ..., "digit": ...}, ...]}
    """
    per_level: dict[int, list[dict]] = defaultdict(list)
    seen: set[str] = set()

    for rec in prefix_data:
        pc = rec["prefix_code"]
        if pc in seen:
            continue
        seen.add(pc)

        body = pc[1:]  # 去掉首字符 G / Strip leading character G
        max_level = min(32, len(body))
        for lv in range(1, max_level + 1):
            digit = body[lv - 1]
            per_level[lv].append({"prefix_code": pc, "digit": digit})

    return dict(per_level)


def create_grid_relations(session, level: int, batch: list[dict]):
    """批量创建某一层级的 LOCATED_IN_**GRID 关系。 / Batch-create LOCATED_IN_**GRID relations for one level."""
    label = f"grid{level:02d}Code"
    prop = f"grid_L{level}_code"
    rel = f"LOCATED_IN_{level:02d}GRID"
    session.run(
        f"""
        UNWIND $data AS item
        MATCH (p:GeoSOTCommonPrefix {{prefix_code: item.prefix_code}})
        MATCH (grid:`{label}` {{{prop}: item.digit}})
        MERGE (p)-[:`{rel}`]->(grid)
        """,
        data=batch,
    )


def ensure_indexes(session):
    """创建必要的索引以加速查询。 / Create necessary indexes to speed up queries."""
    session.run(
        "CREATE INDEX IF NOT EXISTS FOR (p:GeoSOTCommonPrefix) ON (p.prefix_code)"
    )
    print("[索引] GeoSOTCommonPrefix.prefix_code 索引已就绪 / [Index] GeoSOTCommonPrefix.prefix_code index ready")


def main():
    parser = argparse.ArgumentParser(
        description="为 GeoSOTCode 与 GeoSOTVertexCode 生成最长公共前缀节点 / Build LCP nodes for GeoSOTCode and GeoSOTVertexCode"
    )
    parser.add_argument("--uri", default=NEO4J_URI)
    parser.add_argument("--user", default=NEO4J_USER)
    parser.add_argument("--password", default=NEO4J_PASSWORD)
    parser.add_argument("--database", default=NEO4J_DATABASE)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true",
                        help="仅计算前缀，不写入数据库 / Compute prefixes only; do not write to DB")
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    try:
        with driver.session(database=args.database) as session:
            ensure_indexes(session)

        print("[1/4] 查询拥有多个顶点的 GeoSOTCode ... / Query GeoSOTCode with multiple vertices ...")
        t0 = time.time()
        with driver.session(database=args.database) as session:
            records = fetch_geosotcode_with_vertices(session)
        print(f"      找到 {len(records)} 个 GeoSOTCode (耗时 {time.time() - t0:.1f}s) / Found {len(records)} GeoSOTCode (took {time.time() - t0:.1f}s)")

        if not records:
            print("没有符合条件的 GeoSOTCode，退出。 / No matching GeoSOTCode; exit.")
            return

        print("[2/4] 计算最长公共前缀 ... / Compute longest common prefixes ...")
        t0 = time.time()
        prefix_data = compute_prefixes(records)
        print(f"      生成 {len(prefix_data)} 条前缀记录 (耗时 {time.time() - t0:.3f}s) / Generated {len(prefix_data)} prefix records (took {time.time() - t0:.3f}s)")

        unique_prefixes = set(d["prefix_code"] for d in prefix_data)
        print(f"      去重后共 {len(unique_prefixes)} 个不同的前缀 / {len(unique_prefixes)} unique prefixes after dedup")

        prefix_len_dist = {}
        for d in prefix_data:
            l = d["prefix_level"]
            prefix_len_dist[l] = prefix_len_dist.get(l, 0) + 1
        print("      前缀长度分布: / Prefix length distribution:")
        for length in sorted(prefix_len_dist):
            print(f"        长度 {length}: {prefix_len_dist[length]} 个 / length {length}: {prefix_len_dist[length]}")

        if args.dry_run:
            print("\n[dry-run] 前 10 条示例: / First 10 examples:")
            for d in prefix_data[:10]:
                print(f"  main_code={d['main_code']}")
                print(f"    prefix={d['prefix_code']}  "
                      f"(长度={d['prefix_level']}, 顶点数={d['vertex_count']}) / "
                      f"(length={d['prefix_level']}, vertices={d['vertex_count']})")
            print("\n[dry-run] 未写入数据库。 / Nothing written to database.")
            return

        print("[3/4] 写入 GeoSOTCommonPrefix 节点及 HAS_COMMON_PREFIX 关系 ... / Write GeoSOTCommonPrefix nodes and HAS_COMMON_PREFIX ...")
        t0 = time.time()
        batch_size = args.batch_size
        total_batches = (len(prefix_data) + batch_size - 1) // batch_size
        with driver.session(database=args.database) as session:
            for i in tqdm(range(0, len(prefix_data), batch_size),
                          desc="写入前缀节点 / Write prefix nodes", total=total_batches):
                chunk = prefix_data[i: i + batch_size]
                create_prefix_nodes(session, chunk)
        print(f"      写入完成 (耗时 {time.time() - t0:.1f}s) / Write done (took {time.time() - t0:.1f}s)")

        print("[4/4] 建立 LOCATED_IN_**GRID 网格层级关系 ... / Build LOCATED_IN_**GRID hierarchy ...")
        t0 = time.time()
        per_level_data = build_grid_relations(prefix_data)
        total_rels = sum(len(v) for v in per_level_data.values())
        print(f"      需建立 {total_rels} 条网格关系 (覆盖 {len(per_level_data)} 个层级) / Need {total_rels} grid relations across {len(per_level_data)} levels")

        with driver.session(database=args.database) as session:
            for lv in tqdm(sorted(per_level_data.keys()),
                           desc="建立网格关系 / Build grid relations"):
                rows = per_level_data[lv]
                for i in range(0, len(rows), batch_size):
                    chunk = rows[i: i + batch_size]
                    create_grid_relations(session, lv, chunk)
        print(f"      网格关系写入完成 (耗时 {time.time() - t0:.1f}s) / Grid relations written (took {time.time() - t0:.1f}s)")

        print("\n[验证] 检查写入结果 ... / [Verify] Checking write results ...")
        with driver.session(database=args.database) as session:
            res = session.run(
                "MATCH (p:GeoSOTCommonPrefix) RETURN count(p) AS cnt"
            ).single()
            print(f"  GeoSOTCommonPrefix 节点数: {res['cnt']} / GeoSOTCommonPrefix node count: {res['cnt']}")

            res = session.run(
                "MATCH (:GeoSOTCode)-[r:HAS_COMMON_PREFIX]->(:GeoSOTCommonPrefix) "
                "RETURN count(r) AS cnt"
            ).single()
            print(f"  HAS_COMMON_PREFIX 关系数: {res['cnt']} / HAS_COMMON_PREFIX relation count: {res['cnt']}")

            res = session.run(
                "MATCH (:GeoSOTCommonPrefix)-[r]->() "
                "WHERE type(r) STARTS WITH 'LOCATED_IN_' "
                "RETURN type(r) AS rel_type, count(r) AS cnt "
                "ORDER BY rel_type"
            )
            grid_total = 0
            for row in res:
                cnt = row["cnt"]
                grid_total += cnt
                print(f"  {row['rel_type']}: {cnt}")
            print(f"  LOCATED_IN 关系总数: {grid_total} / LOCATED_IN total: {grid_total}")

        print("\n完成！ / Done!")

    finally:
        driver.close()


if __name__ == "__main__":
    main()

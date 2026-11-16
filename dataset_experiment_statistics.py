#!/usr/bin/env python3
# coding: utf-8
"""
Collect PostGIS dataset counts, environment information, and optional legacy Neo4j graph
counts. Spatial tables include buildings, POIs, statutory-plan parcels, and roads; semantic
counts use oracle_src resident/house tables. The optional graph path still describes
GeoSOTCode/grid01Code..grid32Code and should be checked against the deployed schema.

Configure PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD and NEO4J_URI, NEO4J_USER,
NEO4J_PASSWORD, NEO4J_DB through the environment.

Usage:
    python dataset_experiment_statistics.py
    python dataset_experiment_statistics.py --neo4j --json out.json
    python dataset_experiment_statistics.py --markdown table.md
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extensions import connection as PGConnection

from geosot_overlay_query import SOURCE_TO_PG_TABLE
from geosot_spatial_reasoning import PG_CONFIG

# Fixed source labels for the paper's Table 1; adapt DEFAULT_POI_SOURCE and related values to your data.
DATASET_META = {
    "建筑物": {
        "feature_type": "Polygon",
        "pg_source": "public.building_new",
        "source_note": "深圳市不动产登记数据",
    },
    "POI": {
        "feature_type": "Point",
        "pg_source": "public.poi_new",
        "source_note": os.getenv("DATASET_POI_SOURCE", "（请填写实际 POI 数据来源）"),
    },
    "法定图则地块": {
        "feature_type": "Polygon",
        "pg_source": "public.statutory_plan",
        "source_note": "深圳市规划数据",
    },
    "道路": {
        "feature_type": "LineString",
        "pg_source": "public.road",
        "source_note": os.getenv("DATASET_ROAD_SOURCE", "（请填写实际道路数据来源）"),
    },
    "居民": {
        "feature_type": "关系型记录",
        "pg_source": "oracle_src.resident",
        "source_note": "人口管理数据",
    },
    "房屋": {
        "feature_type": "关系型记录",
        "pg_source": "oracle_src.house",
        "source_note": "不动产关联数据",
    },
}

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")


def _pg_connect(cfg: Dict[str, Any]) -> PGConnection:
    return psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
    )


def _load_pg_config() -> Dict[str, Any]:
    c = dict(PG_CONFIG)
    c["host"] = os.getenv("PGHOST", c["host"])
    c["port"] = int(os.getenv("PGPORT", str(c["port"])))
    c["dbname"] = os.getenv("PGDATABASE", c["dbname"])
    c["user"] = os.getenv("PGUSER", c["user"])
    c["password"] = os.getenv("PGPASSWORD", c["password"])
    return c


def _parse_table(qname: str) -> Tuple[str, str]:
    if "." in qname:
        sch, tbl = qname.split(".", 1)
        return sch.strip(), tbl.strip()
    return "public", qname.strip()


def _geometry_column(conn: PGConnection, schema: str, table: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT f_geometry_column
        FROM geometry_columns
        WHERE f_table_schema = %s AND f_table_name = %s
        ORDER BY f_geometry_column
        LIMIT 1
        """,
        (schema, table),
    )
    row = cur.fetchone()
    cur.close()
    return str(row[0]) if row else None


def _table_exists(conn: PGConnection, schema: str, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
        LIMIT 1
        """,
        (schema, table),
    )
    ok = cur.fetchone() is not None
    cur.close()
    return ok


def _count_rows(conn: PGConnection, schema: str, table: str) -> Optional[int]:
    if not _table_exists(conn, schema, table):
        return None
    cur = conn.cursor()
    cur.execute(
        f'SELECT count(*) FROM "{schema}"."{table}"'
    )
    n = int(cur.fetchone()[0])
    cur.close()
    return n


def _geometry_type_counts(
    conn: PGConnection, schema: str, table: str, geom_col: str,
) -> Dict[str, int]:
    cur = conn.cursor()
    cur.execute(
        f'''
        SELECT lower(ST_GeometryType("{geom_col}")) AS gtype, count(*) AS cnt
        FROM "{schema}"."{table}"
        WHERE "{geom_col}" IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
        ''',
    )
    out = {str(r[0]): int(r[1]) for r in cur.fetchall()}
    cur.close()
    return out


def _postgis_srid(conn: PGConnection, schema: str, table: str, geom_col: str) -> Optional[int]:
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT Find_SRID(%s, %s, %s)",
            (schema, table, geom_col),
        )
        row = cur.fetchone()
    except Exception:
        row = None
    finally:
        cur.close()
    if row is None or row[0] is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def collect_postgis_stats(conn: PGConnection) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT version()")
    pg_version = cur.fetchone()[0]
    postgis_ver = None
    try:
        cur.execute("SELECT PostGIS_Full_Version()")
        postgis_ver = cur.fetchone()[0]
    except Exception:
        pass
    cur.close()

    spatial_layers: Dict[str, Any] = {}
    for name, meta in DATASET_META.items():
        if meta["feature_type"] == "关系型记录":
            continue
        schema, table = _parse_table(meta["pg_source"])
        entry: Dict[str, Any] = {
            "qualified": f"{schema}.{table}",
            "exists": _table_exists(conn, schema, table),
        }
        if not entry["exists"]:
            spatial_layers[name] = entry
            continue
        geom = _geometry_column(conn, schema, table)
        entry["geometry_column"] = geom
        entry["row_count"] = _count_rows(conn, schema, table)
        if geom:
            entry["geometry_types"] = _geometry_type_counts(conn, schema, table, geom)
            entry["srid"] = _postgis_srid(conn, schema, table, geom)
        spatial_layers[name] = entry

    relational: Dict[str, Any] = {}
    for name in ("居民", "房屋"):
        meta = DATASET_META[name]
        schema, table = _parse_table(meta["pg_source"])
        relational[name] = {
            "qualified": f"{schema}.{table}",
            "row_count": _count_rows(conn, schema, table),
        }

    return {
        "postgresql_version": pg_version,
        "postgis_full_version": postgis_ver,
        "spatial_layers": spatial_layers,
        "relational": relational,
        "source_to_pg_map": dict(SOURCE_TO_PG_TABLE),
    }


def collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor() or "",
    }
    try:
        import psutil  # type: ignore

        env["cpu_logical"] = psutil.cpu_count(logical=True)
        env["cpu_physical"] = psutil.cpu_count(logical=False)
        vm = psutil.virtual_memory()
        env["memory_total_gb"] = round(vm.total / (1024**3), 2)
    except Exception:
        env["cpu_logical"] = None
        env["memory_total_gb"] = None
    return env


def collect_neo4j_stats() -> Dict[str, Any]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    out: Dict[str, Any] = {
        "uri": NEO4J_URI,
        "database": NEO4J_DATABASE,
    }
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            comps = session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions, edition",
            ).data()
            out["components"] = comps

            def _one(q: str) -> int:
                rec = session.run(q).single()
                if not rec:
                    return 0
                return int(list(rec.values())[0])

            out["nodes_total"] = _one("MATCH (n) RETURN count(n) AS c")
            out["relationships_total"] = _one("MATCH ()-[r]->() RETURN count(r) AS c")
            out["geosotcode_nodes"] = _one(
                "MATCH (n:GeoSOTCode) RETURN count(n) AS c",
            )

            grid_counts: Dict[str, int] = {}
            grid_sum = 0
            for lv in range(1, 33):
                lab = f"grid{lv:02d}Code"
                try:
                    c = _one(f"MATCH (n:{lab}) RETURN count(n) AS c")
                except Exception:
                    c = 0
                if c:
                    grid_counts[lab] = c
                grid_sum += c
            out["grid_level_counts"] = grid_counts
            out["grid_nodes_sum_L1_L32"] = grid_sum

            out["rel_east_of"] = _one(
                "MATCH ()-[r:EAST_OF]->() RETURN count(r) AS c",
            )
            out["rel_north_of"] = _one(
                "MATCH ()-[r:NORTH_OF]->() RETURN count(r) AS c",
            )
            out["rel_direction_sum"] = (
                out["rel_east_of"] + out["rel_north_of"])
    finally:
        driver.close()
    return out


def _main_count_for_table(stats: Dict[str, Any], cn: str) -> Optional[int]:
    """
    Paper Table 1 count convention: use spatial table row counts, with geometry-type counts
    reported separately for inspection.
    """
    if cn in stats["spatial_layers"]:
        layer = stats["spatial_layers"][cn]
        return layer.get("row_count") if layer.get("exists") else None
    if cn in stats["relational"]:
        return stats["relational"][cn].get("row_count")
    return None


def print_human_report(
    pg_stats: Dict[str, Any],
    env_info: Dict[str, Any],
    neo_stats: Optional[Dict[str, Any]],
) -> None:
    print("=" * 72)
    print("表 1 实验数据集概况（PostGIS 实测数量）")
    print("=" * 72)
    print(
        f"{'数据集':<14} {'要素类型':<12} {'数量':>14} {'来源/库表'}",
    )
    print("-" * 72)
    for cn, meta in DATASET_META.items():
        cnt = _main_count_for_table(pg_stats, cn)
        cnt_s = f"{cnt:,}" if isinstance(cnt, int) else "N/A（表不存在或无权限）"
        ref = meta["source_note"]
        if meta.get("pg_source"):
            ref = f"{ref}  [{meta['pg_source']}]"
        print(
            f"{cn:<14} {meta['feature_type']:<12} {cnt_s:>14}  {ref}",
        )
    print()
    print("空间图层几何类型分布（ST_GeometryType，非空几何）：")
    for cn, layer in pg_stats["spatial_layers"].items():
        if not layer.get("exists"):
            print(f"  [{cn}] 表不存在: {layer.get('qualified')}")
            continue
        gt = layer.get("geometry_types") or {}
        srid = layer.get("srid")
        print(f"  [{cn}] SRID={srid}  {gt}")
    print()
    print("=" * 72)
    print("表 2 实验环境配置（本机探测）")
    print("=" * 72)
    print(f"操作系统: {env_info.get('os')}")
    if env_info.get("memory_total_gb") is not None:
        print(
            f"CPU（逻辑核）: {env_info.get('cpu_logical')}  "
            f"内存: {env_info.get('memory_total_gb')} GiB",
        )
    else:
        print("CPU/内存: 请安装 psutil 后重试，或手工填写")
    print(f"Python: {env_info.get('python')}")
    print(f"PostgreSQL: {pg_stats.get('postgresql_version', '')}")
    if pg_stats.get("postgis_full_version"):
        print(f"PostGIS: {pg_stats['postgis_full_version']}")
    if neo_stats:
        print("Neo4j:")
        for comp in neo_stats.get("components") or []:
            v = comp.get("versions")
            vs = ",".join(v) if isinstance(v, list) else str(v)
            print(f"  {comp.get('name')}: {vs} ({comp.get('edition')})")
        print()
        print("知识图谱规模（Neo4j 当前库）:")
        print(f"  总节点数: {neo_stats.get('nodes_total', 0):,}")
        print(f"  总关系数: {neo_stats.get('relationships_total', 0):,}")
        print(f"  GeoSOTCode 节点: {neo_stats.get('geosotcode_nodes', 0):,}")
        print(
            f"  grid01~grid32 节点数合计: "
            f"{neo_stats.get('grid_nodes_sum_L1_L32', 0):,}",
        )
        print(
            f"  EAST_OF: {neo_stats.get('rel_east_of', 0):,}  "
            f"NORTH_OF: {neo_stats.get('rel_north_of', 0):,}  "
            f"合计: {neo_stats.get('rel_direction_sum', 0):,}",
        )
    else:
        print("Neo4j: 未查询（加 --neo4j 可拉取版本与图谱计数）")
    print("=" * 72)


def build_export_payload(
    pg_stats: Dict[str, Any],
    env_info: Dict[str, Any],
    neo_stats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    table1 = []
    for cn, meta in DATASET_META.items():
        table1.append({
            "dataset": cn,
            "feature_type": meta["feature_type"],
            "count": _main_count_for_table(pg_stats, cn),
            "source": meta["source_note"],
            "pg_table": meta.get("pg_source"),
        })
    return {
        "table_1_dataset_overview": table1,
        "postgis_detail": pg_stats,
        "table_2_environment": env_info,
        "neo4j": neo_stats,
        "notes": {
            "crs": "空间数据应采用 CGCS2000（EPSG:4490）；请核对 postgis_detail 中各图层 SRID。",
            "encoding_level": "论文中编码级别 L=32 为实验设定；本脚本仅统计库内记录数。",
        },
    }


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="2.1.1 数据集与 2.1.2 环境统计",
    )
    parser.add_argument(
        "--neo4j", action="store_true", help="连接 Neo4j 统计图谱规模",
    )
    parser.add_argument(
        "--json", metavar="PATH", help="将完整结果写入 JSON",
    )
    parser.add_argument(
        "--markdown", metavar="PATH", help="将表 1/表 2 摘要写入 Markdown",
    )
    args = parser.parse_args()

    cfg = _load_pg_config()
    try:
        conn = _pg_connect(cfg)
    except Exception as e:
        print(f"[ERROR] PostGIS 连接失败: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        pg_stats = collect_postgis_stats(conn)
    finally:
        conn.close()

    env_info = collect_environment()
    neo_stats = None
    if args.neo4j:
        try:
            neo_stats = collect_neo4j_stats()
        except Exception as e:
            print(f"[WARN] Neo4j 统计失败: {e}", file=sys.stderr)

    print_human_report(pg_stats, env_info, neo_stats)

    payload = build_export_payload(pg_stats, env_info, neo_stats)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 JSON: {args.json}")

    if args.markdown:
        lines = [
            '## Table 1. Experimental dataset overview',
            "",
            "| 数据集 | 要素类型 | 数量 | 来源 |",
            "| --- | --- | ---: | --- |",
        ]
        for row in payload["table_1_dataset_overview"]:
            c = row["count"]
            cs = f"{c:,}" if isinstance(c, int) else "N/A"
            lines.append(
                f"| {row['dataset']} | {row['feature_type']} | {cs} | "
                f"{row['source']} |",
            )
        lines.extend([
            "",
            '## Table 2. Detected experimental environment',
            "",
            f"- 操作系统: {env_info.get('os')}",
            f"- Python: {env_info.get('python')}",
            f"- PostgreSQL: {pg_stats.get('postgresql_version', '')}",
        ])
        if pg_stats.get("postgis_full_version"):
            lines.append(f"- PostGIS: {pg_stats['postgis_full_version']}")
        if env_info.get("memory_total_gb") is not None:
            lines.append(
                f"- CPU（逻辑）: {env_info.get('cpu_logical')} / "
                f"内存: {env_info.get('memory_total_gb')} GiB",
            )
        if neo_stats:
            nv = ""
            for comp in neo_stats.get("components") or []:
                if comp.get("name") == "Neo4j Kernel":
                    v = comp.get("versions")
                    nv = ",".join(v) if isinstance(v, list) else str(v)
            lines.append(f"- Neo4j: {nv or '（见 JSON）'}")
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"已写入 Markdown: {args.markdown}")


if __name__ == "__main__":
    main()

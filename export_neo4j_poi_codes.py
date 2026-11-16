#!/usr/bin/env python3
# coding: utf-8
"""
Read-only export of Point centroid codes for no-graph KNN comparisons. Match the same full
GeoSOTCentroidCode/HAS_GEOMETRY_TYPE population as POIPrefixCache, not only the POI source
in recover_out/poi_new_encoded.csv. --compare-csv audits that coverage difference. Exported
real codes are location-bearing data and are not distributed automatically.

Usage:
    python export_neo4j_poi_codes.py
    python export_neo4j_poi_codes.py --geom-type Point --output csv/neo4j_poi_point_codes.csv --compare-csv recover_out/poi_new_encoded.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter

from neo4j import GraphDatabase

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from geosot_knn_v3 import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)
from geosot_spatial_reasoning import is_valid_g0_code

_CYPHER = (
    "MATCH (c:GeoSOTCentroidCode)-[:HAS_GEOMETRY_TYPE]->"
    "(:GeoFeatureType {type_name: $gt})\n"
    "OPTIONAL MATCH (c)-[:FROM_SOURCE]->(s:DataSource)\n"
    "RETURN c.main_code AS mc, s.name AS src"
)


def _parse_args() -> argparse.Namespace:
    """@returns Parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description="Export Point centroid codes from Neo4j for CSV KNN.")
    p.add_argument("--geom-type", default="Point",
                   help="GeoFeatureType.type_name (default Point)")
    p.add_argument(
        "--output",
        default=os.path.join(_ROOT, "csv", "neo4j_poi_point_codes.csv"),
        help="output CSV path")
    p.add_argument(
        "--compare-csv",
        default=os.path.join(_ROOT, "recover_out", "poi_new_encoded.csv"),
        help="encoded CSV used to print coverage vs the Neo4j dump; "
             "empty string skips")
    return p.parse_args()


def _load_encoded_codes(path: str) -> set:
    """
    @param path Encoded CSV with a main_code column.
    @returns Valid G0 main_code set.
    """
    codes = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "main_code" not in reader.fieldnames:
            raise SystemExit(f"no main_code column in {path}")
        for row in reader:
            mc = (row.get("main_code") or "").strip().upper()
            if mc and is_valid_g0_code(mc):
                codes.add(mc)
    return codes


def _print_compare(neo_codes: set, compare_path: str) -> None:
    """
    @param neo_codes Codes exported from the graph.
    @param compare_path Comparison CSV.
    """
    if not compare_path or not os.path.isfile(compare_path):
        if compare_path:
            print(f"  compare skipped (missing): {compare_path}")
        return
    csv_codes = _load_encoded_codes(compare_path)
    only_graph = neo_codes - csv_codes
    only_csv = csv_codes - neo_codes
    both = neo_codes & csv_codes
    print(f"  compare vs {compare_path}")
    print(f"    encoded CSV unique codes: {len(csv_codes):,}")
    print(f"    intersection:             {len(both):,}")
    print(f"    graph only:               {len(only_graph):,}")
    print(f"    encoded CSV only:         {len(only_csv):,}")


def main() -> None:
    """Connect to Neo4j and export main_code, geom_type, and source."""
    args = _parse_args()
    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    print(f"Neo4j {NEO4J_URI}  db={NEO4J_DATABASE}  geom_type={args.geom_type}")
    t0 = time.perf_counter()
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    rows_out = []
    src_hist: Counter = Counter()
    n_invalid = 0
    n_dup = 0
    seen = set()
    with drv.session(database=NEO4J_DATABASE) as sess:
        for r in sess.run(_CYPHER, gt=args.geom_type):
            raw = r.get("mc")
            mc = str(raw).strip().upper() if raw else ""
            if not mc or not is_valid_g0_code(mc):
                n_invalid += 1
                continue
            if mc in seen:
                n_dup += 1
                continue
            seen.add(mc)
            src = str(r["src"] or "").strip()
            src_hist[src or "(none)"] += 1
            rows_out.append({
                "main_code": mc,
                "geom_type": args.geom_type,
                "source": src,
            })
    drv.close()
    dt = time.perf_counter() - t0

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["main_code", "geom_type", "source"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"wrote {len(rows_out):,} codes -> {out}  ({dt:.1f}s)")
    print(f"  skipped invalid: {n_invalid:,}")
    print(f"  skipped duplicate main_code: {n_dup:,}")
    print("  by FROM_SOURCE:")
    for name, n in src_hist.most_common():
        print(f"    {name}: {n:,}")
    _print_compare(seen, args.compare_csv)


if __name__ == "__main__":
    main()

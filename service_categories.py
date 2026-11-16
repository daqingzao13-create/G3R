#!/usr/bin/env python3
# coding: utf-8
"""
Define/load six facility categories: education (schools); medical (hospitals/clinics);
sports/recreation; scenic sites; public facilities; and transport excluding parking.

Resolve broad categories only from HAS_POI_TYPE_3 leaves through SUBCLASS_OF, rather than
redundant POI-to-parent shortcuts. This mirrors PG leaf poi codes joined to
gaode_poi_category. Historical checks found identical six-category counts to the shortcut-
based definitions. Chinese category labels are query values and are retained.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional

from geosot_spatial_reasoning import is_valid_g0_code

# key -> (Chinese label, level, name list)
CATEGORIES = {
    "edu":       ("教育-学校",       "L2",       ["学校"]),
    "medical":   ("医疗",           "L2",       ["综合医院", "专科医院", "诊所"]),
    "sport":     ("体育休闲",        "L1",       ["体育休闲服务"]),
    "scenic":    ("风景名胜",        "L1",       ["风景名胜"]),
    "public":    ("公共设施",        "L1",       ["公共设施"]),
    "transport": ("交通设施(去停车场)", "L1_minus", ["交通设施服务"]),
}

# Display order for reports
ORDER = ["edu", "medical", "sport", "scenic", "public", "transport"]

# Traverse POIType3 leaves through SUBCLASS_OF: f3->f2 (1 hop), then f2->f1 (up to 2 hops total).
_Q_L2 = (
    "MATCH (p:POI)-[:HAS_POI_TYPE_3]->(:POIType3)-[:SUBCLASS_OF]->(t:POIType2)\n"
    "WHERE t.name IN $names\n"
    "RETURN DISTINCT p.main_code AS mc")
_Q_L1 = (
    "MATCH (p:POI)-[:HAS_POI_TYPE_3]->(:POIType3)-[:SUBCLASS_OF*1..2]->(t:POIType)\n"
    "WHERE t.name IN $names\n"
    "RETURN DISTINCT p.main_code AS mc")
_Q_L1_MINUS_PARK = (
    "MATCH (p:POI)-[:HAS_POI_TYPE_3]->(l:POIType3)"
    "-[:SUBCLASS_OF*1..2]->(:POIType {name:'交通设施服务'})\n"
    "WHERE NOT EXISTS { (l)-[:SUBCLASS_OF]->(:POIType2 {name:'停车场'}) }\n"
    "RETURN DISTINCT p.main_code AS mc")


def _valid_codes(rows) -> List[str]:
    out = []
    for r in rows:
        mc = str(r["mc"]).strip().upper() if r["mc"] else ""
        if mc and is_valid_g0_code(mc):
            out.append(mc)
    return out


def load_category_codes(session,
                        timings: Optional[Dict[str, float]] = None
                        ) -> Dict[str, List[str]]:
    """
    Return {category_key: [main_code, ...]}. When timings is supplied, fill each category's
    leaf-to-SUBCLASS_OF resolution and code-retrieval seconds, matching the PG leaf-code join
    timing boundary.
    """
    result: Dict[str, List[str]] = {}
    for key in ORDER:
        cn, level, names = CATEGORIES[key]
        t0 = time.perf_counter()
        if level == "L2":
            rows = session.run(_Q_L2, names=names)
        elif level == "L1":
            rows = session.run(_Q_L1, names=names)
        elif level == "L1_minus":
            rows = session.run(_Q_L1_MINUS_PARK)
        else:
            rows = []
        result[key] = _valid_codes(rows)
        if timings is not None:
            timings[key] = time.perf_counter() - t0
    return result


if __name__ == "__main__":
    import os, sys
    from neo4j import GraphDatabase
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    USER = os.getenv("NEO4J_USER", "neo4j")
    PWD = os.getenv("NEO4J_PASSWORD", "")
    DB = os.getenv("NEO4J_DB", "geocode")
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    with drv.session(database=DB) as s:
        codes = load_category_codes(s)
    print("六类服务设施 POI 计数 (有效 G0 码):")
    for k in ORDER:
        cn = CATEGORIES[k][0]
        print(f"  {k:<10} {cn:<16} {len(codes[k]):>8,}")
    drv.close()

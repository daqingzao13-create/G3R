#!/usr/bin/env python3
# coding: utf-8
"""
六类服务设施 POI 定义 + 加载 (供 KNN 可达性用)。口径 = 作者 V0.6 锁定:
Six service-facility POI categories + loader (for KNN accessibility). Scope locked to author V0.6:
  edu       教育: POIType2 学校 / Education: POIType2 school
  medical   医疗: POIType2 综合医院/专科医院/诊所 / Medical: general/specialty hospital, clinic
  sport     体育休闲: POIType1 体育休闲服务(全部) / Sports & leisure: POIType1 (all)
  scenic    风景名胜: POIType1 风景名胜(全部) / Scenic spots: POIType1 (all)
  public    公共设施: POIType1 公共设施(全部) / Public facilities: POIType1 (all)
  transport 交通设施: POIType1 交通设施服务(全部) 去 POIType2 停车场
            Transport: POIType1 transport services (all) minus POIType2 parking

规范化层级解析 (V0.9, 与 PostGIS 叶子码 join 对称):
Canonical hierarchy resolution (V0.9, symmetric to PostGIS leaf-code join):
  不再走 POI 的冗余捷径边 (HAS_POI_TYPE / HAS_POI_TYPE_2 直连大/中类),
  而是**只从叶子** (HAS_POI_TYPE_3 → POIType3) 出发, 沿 SUBCLASS_OF 上溯到
  目标大/中类。这样 POI 只需保存最细粒度类型, 粗类由类树推理得到 —— 与 PG 端
  "poi_new 只留叶子码 poi, join gaode_poi_category 展开大/中类"逐条等价。
  Do not use redundant shortcut edges (HAS_POI_TYPE / HAS_POI_TYPE_2 to coarse classes);
  start **only from leaves** (HAS_POI_TYPE_3 → POIType3) and walk SUBCLASS_OF up to the
  target L1/L2 class. POI stores finest type only; coarse class is inferred — equivalent
  to PG keeping leaf code `poi` and joining `gaode_poi_category` for L1/L2.
  已验证: 六类计数与旧捷径边口径逐类 Δ=0 (见 _tmp_probe_poi_cols)。
  Verified: six-category counts match old shortcut-edge scope with per-class Δ=0
  (see _tmp_probe_poi_cols).
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional

from geosot_spatial_reasoning import is_valid_g0_code

# key -> (中文名, 级别, 名称列表) / key -> (Chinese label, level, name list)
CATEGORIES = {
    "edu":       ("教育-学校",       "L2",       ["学校"]),
    "medical":   ("医疗",           "L2",       ["综合医院", "专科医院", "诊所"]),
    "sport":     ("体育休闲",        "L1",       ["体育休闲服务"]),
    "scenic":    ("风景名胜",        "L1",       ["风景名胜"]),
    "public":    ("公共设施",        "L1",       ["公共设施"]),
    "transport": ("交通设施(去停车场)", "L1_minus", ["交通设施服务"]),
}

# 便于报告的显示顺序 / Display order for reports
ORDER = ["edu", "medical", "sport", "scenic", "public", "transport"]

# 叶子 (POIType3) 沿 SUBCLASS_OF 上溯: f3→f2 (1 跳) 到中类; f3→f2→f1 (1..2 跳) 到大类
# Leaf (POIType3) via SUBCLASS_OF: f3→f2 (1 hop) to mid-class; f3→f2→f1 (1..2 hops) to top-class
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
    """返回 {key: [main_code, ...]}。 / Return {key: [main_code, ...]}.

    timings 非空时, 逐类回填该类"叶子→SUBCLASS_OF 上溯"解析 + 取码耗时(秒),
    供细粒度分步计时用 (体现 G3R 类树推理代价, 与 PG 端叶子码 join 对称)。
    When timings is set, fill per-category seconds for leaf→SUBCLASS_OF resolution +
    code fetch (fine-grained timing; mirrors PG leaf-code join cost).
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
    print("六类服务设施 POI 计数 (有效 G0 码) / Six service POI counts (valid G0 codes):")
    for k in ORDER:
        cn = CATEGORIES[k][0]
        print(f"  {k:<10} {cn:<16} {len(codes[k]):>8,}")
    drv.close()

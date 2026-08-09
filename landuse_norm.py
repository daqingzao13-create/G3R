#!/usr/bin/env python3
# coding: utf-8
"""
用地功能(LUFunctionGroup)归一化模块 —— 供错配实验复用。
Land-use function (LUFunctionGroup) normalization — reused by the mismatch experiment.

核心产出 / Core output:
    classify(raw) -> (land_class, land_subtype)
      land_class  ∈ {住宅, 非住宅, 混合}
      land_subtype: 非住宅的细分(工业仓储/商业办公/绿地水域/风景旅游/交通设施/
                    市政公用/教育科研/医疗卫生/文体娱乐/行政特殊/社会福利/
                    配套设施/发展备用控制/其他非住宅); 住宅/混合固定回其类名。
      land_subtype: non-residential subtypes (industrial-storage / commercial-office /
                    green-water / scenic-tourism / transport / municipal / education-R&D /
                    medical / culture-sports / admin-special / social-welfare /
                    supporting / reserve-control / other-non-res); residential/mixed
                    echo their class name.

设计原则(重要) / Design principles (important):
  * 保守偏"住宅": 含糊的居住相关一律归住宅 —— 因为错配实验里把非住宅误判成
    住宅只是漏掉候选(保守), 把住宅误判成非住宅才会制造假错配。
    Bias toward "residential": ambiguous dwelling-related labels → residential —
    misclassifying non-res as res only drops candidates (conservative); the reverse
    fabricates false mismatches.
  * "住宅区/居住区 + 绿地/道路/停车" = 居住区内基础设施, 非居住空间 → 非住宅。
    "Residential zone + green/road/parking" = in-zone infrastructure, not dwelling → non-res.
  * 商住 / 居住+(商业/办公/工业/服务业/医疗) = 混合。
    Mixed commercial-residential / dwelling+(commercial/office/industrial/service/medical) = mixed.
  * 商务/办公/酒店/旅馆公寓 = 商业; 单身公寓/单身宿舍/公寓 = 住宅。
    Business/office/hotel/inn apartments = commercial; bachelor apt/dorm/apartment = residential.
"""
from __future__ import annotations
import re
from typing import Tuple

# 强非住宅信号(与居住共现即判混合) / Strong non-residential signals (co-occur with dwelling → mixed)
NONRES_STRONG = ("商业", "商务", "办公", "工业", "服务业", "医疗", "旅馆",
                 "酒店", "金融", "产业")

# 居住区内基础设施(即使含居住/住宅也算非住宅) / In-zone infrastructure (non-res even if text has dwelling/residential)
_ZONE_PREFIX = ("住宅区", "居住区")
_ZONE_INFRA = ("绿地", "道路", "停车")

# 公寓当住宅的排除前缀(这些是商业性公寓) / Prefixes that exclude apartments from residential (commercial apts)
_APT_COMMERCIAL = ("商务", "办公", "酒店", "旅馆", "商业")

# 截断/缩写的居住类 / Truncated/abbreviated residential labels
_RES_ABBR = {"一类", "二类", "三类", "四类", "五类", "类居住", "二类居"}

# 非住宅细分关键词(有序; 先匹配者胜) / Non-residential subtype keyword rules (ordered; first match wins)
_SUBTYPE_RULES = [
    ("发展备用控制", ("填海", "白色用地", "发展备用", "备用地", "备用", "控制",
                  "限制", "城市建设", "未利用", "发展用地", "综合发展",
                  "特别管制", "拆除整治", "旧村保护", "水源保护",
                  "农田保护", "保留林", "保留")),
    ("绿地水域", ("绿地", "公园", "水域", "水面", "河流", "湖泊", "水库",
              "湿地", "林地", "园地", "耕地", "菜地", "农田", "农用地",
              "牧草", "沙滩", "海水", "海滨", "组团隔离", "退果还林",
              "水源涵养", "生态", "园林生产")),
    ("风景旅游", ("风景", "旅游", "度假", "游乐", "高尔夫", "郊野")),
    ("交通设施", ("交通", "道路", "公路", "铁路", "轨道", "地铁", "地下铁道",
              "停车", "港口", "客运", "货运", "枢纽", "口岸", "快速路",
              "高速", "隧道", "管道运输", "对外交通", "场站", "长途",
              "汽车站", "教练场", "加油", "加气", "广场")),
    ("市政公用", ("市政", "供电", "供水", "供气", "供燃气", "燃气", "邮政",
              "电信", "邮电", "环卫", "环境卫生", "垃圾", "污水", "雨水",
              "粪便", "消防", "广播电视", "施工与维修", "水工", "供应设施",
              "供应燃气", "供气设施")),
    ("工业仓储", ("工业", "产业", "仓库", "仓储", "堆场", "物流", "创储",
              "研发", "生产资料", "园区", "高新")),
    ("教育科研", ("学校", "小学", "中学", "教育", "幼托", "托幼", "幼儿园",
              "院校", "科研")),
    ("医疗卫生", ("医疗", "卫生", "医院", "防疫")),
    ("文体娱乐", ("文化", "图书", "剧院", "影剧院", "娱乐", "文娱", "体育",
              "会议", "宗教", "文物", "休闲")),
    ("社会福利", ("社会福利", "福利", "休疗养", "殡葬")),
    ("行政特殊", ("行政", "政府", "社团", "团体", "公安", "司法", "军事",
              "保安", "特殊用地", "机关")),
    ("商业办公", ("商业", "商务", "办公", "服务业", "金融", "旅馆", "酒店",
              "餐饮", "市场", "会议中心", "商业性", "公寓", "服务")),
    ("配套设施", ("配套", "社区其他设施", "社区其它设施", "社区设施")),
]


def normalize(raw: str) -> str:
    """清洗: 去换行/空白, 全角＋→+, 全角括号统一。 / Clean: strip newlines/whitespace, fullwidth ＋→+, unify fullwidth parentheses."""
    if raw is None:
        return ""
    s = str(raw)
    s = s.replace("\\n", "").replace("\n", "").replace("\r", "")
    s = s.replace("＋", "+").replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    return s.strip()


def _is_zone_infra(s: str) -> bool:
    return any(p in s for p in _ZONE_PREFIX) and any(i in s for i in _ZONE_INFRA)


def _residential_signal(s: str) -> bool:
    if _is_zone_infra(s):
        return False
    if any(t in s for t in ("居住", "住宅", "宿舍")):
        return True
    if "公寓" in s and not any(c in s for c in _APT_COMMERCIAL):
        return True
    if s in _RES_ABBR:
        return True
    return False


def _subtype(s: str) -> str:
    for label, kws in _SUBTYPE_RULES:
        if any(k in s for k in kws):
            return label
    return "其他非住宅"


def classify(raw: str) -> Tuple[str, str]:
    """返回 (land_class, land_subtype)。 / Return (land_class, land_subtype)."""
    s = normalize(raw)
    if not s:
        return ("非住宅", "其他非住宅")

    # 显式混合 / Explicit mixed
    if "商住" in s or s in ("混合用地", "混合"):
        return ("混合", "混合")

    res = _residential_signal(s)
    strong = any(t in s for t in NONRES_STRONG)

    if res and strong:
        return ("混合", "混合")
    if res:
        return ("住宅", "住宅")
    return ("非住宅", _subtype(s))


if __name__ == "__main__":
    tests = [
        "二类居住用地", "商住混合用地", "住宅区绿地", "住宅区道路用地",
        "二类居住及配套设施用地", "工业住宅混合用地", "医疗住宅用地",
        "商务公寓", "单身公寓", "公寓", "单身宿舍用地", "一类工业用地",
        "综合医院用地", "小学用地", "公共绿地", "水域", "社会停车场库用地",
        "二类居住用地＋商业用地", "商业性办公＋三类居住用地", "白色用地",
        "普通创储", "二类", "类居住", "社会福利用地", "填海单元04",
    ]
    for t in tests:
        c, st = classify(t)
        print(f"  {t:<18} -> {c} / {st}")

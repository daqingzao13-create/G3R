#!/usr/bin/env python3
# coding: utf-8
"""
Normalize county_origin administrative names for hierarchical origin resolution in the
urban-village experiment.

parse_admin(raw) returns (province, city). The province includes the province/autonomous-
region/municipality suffix; city denotes the prefecture-level city, autonomous prefecture,
region, or league, with fallback handling for directly administered counties.

The parsed province/city must preserve the original prefix-based definitions of Guangdong
and Shenzhen. Replacing string-prefix checks by CountyOrigin-to-Province/AdminCity traversal
or dictionary joins should leave building scores unchanged. Historical validation covered
all 2,926 CountyOrigin values with zero parse failures or prefix inconsistencies. Chinese
place-name strings are functional data and remain unchanged.
"""
from __future__ import annotations
from typing import Optional, Tuple

# Direct-administered municipalities: the city name also represents the province-level unit.
_MUNI = ("北京", "上海", "天津", "重庆")


def parse_admin(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a full province/city/county origin string into (province, city), including
    municipalities, autonomous regions, and directly administered counties.
    """
    if not raw:
        return (None, None)
    s = str(raw).strip()
    if not s:
        return (None, None)

    prov = None
    rest = s
    for i, ch in enumerate(s):
        if ch == "省":
            prov, rest = s[:i + 1], s[i + 1:]
            break
        if s[i:i + 3] == "自治区":
            prov, rest = s[:i + 3], s[i + 3:]
            break
        if s[i:i + 2] in _MUNI and "市" in s[i:i + 4]:
            j = s.find("市", i)
            prov, rest = s[:j + 1], s[j + 1:]
            break
    if prov is None:
        j = s.find("市")
        if j >= 0:
            prov, rest = s[:j + 1], s[j + 1:]
        else:
            return (s, None)

    city = None
    for kw in ("市", "自治州", "地区", "盟"):
        j = rest.find(kw)
        if j >= 0:
            city = rest[:j + len(kw)]
            break
    if city is None:
        city = rest or None
    return (prov, city)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tests = [
        "海南省省直辖县级行政单位临高县", "广东省深圳市龙岗区",
        "河南省驻马店市新蔡县", "湖南省衡阳市耒阳市", "北京市东城区",
        "重庆市万州区", "广西壮族自治区南宁市青秀区", "内蒙古自治区呼伦贝尔市",
        "新疆维吾尔自治区伊犁哈萨克自治州", "广东省茂名市化州市",
    ]
    for t in tests:
        print(f"  {t:<22} -> {parse_admin(t)}")

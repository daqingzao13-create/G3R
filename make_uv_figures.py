#!/usr/bin/env python3
# coding: utf-8
"""
Optional maps of exploratory urban-village candidate regions over satellite imagery.

--by parcel aggregates building scores to statutory-plan parcels using
building_cc/parcel_main_code attribution from the mismatch experiment. Population-weighted
parcel scores use actual parcel boundaries. --by grid uses legacy 150 m connected components
and building-centroid convex hulls.

Outputs include paper_figures/uv_satellite.png, uv_satellite.html, uv_zoom_*.png, and a
member-building CSV; parcel mode also writes csv/uv_parcels.csv. Do not publish restricted
attributes or downloaded imagery caches automatically.

Usage:
    python make_uv_figures.py --regions --sort-by score
    python make_uv_figures.py --regions --sort-by score --limit-regions 20 --paper-anno
    python make_uv_figures.py --by grid
    python make_uv_figures.py --zoom-top 6
    python make_uv_figures.py --min-bldg 5 --min-res 300 --parcel-th 0.698
"""
from __future__ import annotations
import argparse, csv, math, os, sys

# Clear PROJ_LIB/GDAL_DATA overrides pointing to an older PostgreSQL proj.db; otherwise contextily/
# rasterio may fail EPSG resolution. Let pyproj and rasterio use their bundled PROJ data.
# Do this before importing pyproj or contextily.
for _v in ("PROJ_LIB", "PROJ_DATA", "PROJ_NETWORK", "GDAL_DATA", "GDAL_DRIVER_PATH"):
    os.environ.pop(_v, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neo4j import GraphDatabase
from run_mismatch_experiment import (
    NEO4J_URI, NEO4J_USER, NEO4J_PWD, NEO4J_DB, PG_MAIN)
from urban_village_g3r import (
    score_threshold, stage_d, stage_e, _addr_base, KNOWN_UV, _VILLAGE_WORD)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Three colors distinguish known-name hits, village-like names, and other candidates over dark imagery.
C_HIT = "#FF3B30"     # Known urban-village keyword hit: red.
C_VIL = "#FF9500"     # Village-like naming patterns: orange.
C_OTH = "#FFD60A"     # Other high-scoring regions: yellow.


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(path: str) -> list:
    """Read score CSV rows with cc, uv_score, and res_total fields."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for d in csv.DictReader(f):
            cc = (d.get("cc") or "").strip().upper()
            sc = _to_float(d.get("uv_score"))
            if not cc or sc is None:
                continue
            rows.append({"cc": cc, "uv_score": sc,
                         "res_total": int(_to_float(d.get("res_total")) or 0),
                         "address": ""})
    print(f"[读入] {path}  {len(rows):,} 栋")
    return rows


def _color_for(hit, village):
    return C_HIT if hit else (C_VIL if village else C_OTH)


# ---- Address enrichment: load naming fields from Neo4j because the score CSV has no address ----
def load_parcel_name_cache(csvdir="csv") -> dict:
    """When Neo4j is unavailable, recover parcel names from existing uv_parcels*.csv files."""
    cache = {}
    for fname in ("uv_parcels.csv", "uv_parcels_merged.csv",
                  "uv_parcels_by_score.csv", "uv_parcels_merged_by_score.csv"):
        path = os.path.join(csvdir, fname)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig", newline="") as f:
            for d in csv.DictReader(f):
                name = (d.get("name") or "").strip()
                if not name or name == "(无地址)":
                    continue
                code = (d.get("parcel_main_code") or "").strip().upper()
                if not code:
                    continue
                for pmc in code.split(";"):
                    pmc = pmc.strip().upper()
                    if pmc and pmc not in cache:
                        cache[pmc] = name
    if cache:
        print(f"[命名] 本地缓存地名 {len(cache):,} 个地块")
    return cache


def fetch_address(ccs, csvdir="csv") -> dict:
    ccs = sorted(set(ccs))
    if not ccs:
        return {}
    print(f"[命名] 从 Neo4j 取 {len(ccs):,} 栋地址 ...")
    amap = {}
    try:
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
        with drv.session(database=NEO4J_DB) as sess:
            for i in range(0, len(ccs), 10000):
                chunk = ccs[i:i+10000]
                q = ("MATCH (b:Building)-[:HAS_GEOSOT]->(g:GeoSOTCode)\n"
                     "WHERE g.main_code IN $ccs\n"
                     "RETURN g.main_code AS cc, b.address AS address")
                for rec in sess.run(q, ccs=chunk):
                    a = rec["address"]
                    if a:
                        amap[str(rec["cc"]).strip().upper()] = a.strip()
        drv.close()
        print(f"[命名] Neo4j 命中地址 {len(amap):,} 栋")
        return amap
    except Exception as e:
        print(f"[命名] Neo4j 不可用({e.__class__.__name__}: {e}); "
              f"改用本地 uv_parcels 地名缓存")
        return {"__parcel_name_cache__": load_parcel_name_cache(csvdir)}


# ══════════════════════════════════════════════════════════════════════
# Aggregation A: statutory-plan parcels.
# ══════════════════════════════════════════════════════════════════════
def load_attribution(path: str) -> dict:
    """
    Map building_cc to (parcel_main_code,land_class), retaining only buildings assigned to an
    actual parcel.
    """
    m = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for d in csv.DictReader(f):
            cc = (d.get("building_cc") or "").strip().upper()
            pmc = (d.get("parcel_main_code") or "").strip().upper()
            if cc and pmc:
                m[cc] = (pmc, (d.get("land_class") or "").strip())
    print(f"[读入] {path}  建筑→地块映射 {len(m):,}")
    return m


def aggregate_by_parcel(rows, attr, build_cut, min_bldg, min_res):
    """
    Aggregate building scores by parcel population weight and select high-score parcels.
    build_cut is the building threshold; retain ssum=sum(score*population),
    wsum=sum(population), and high-score counts for lossless regrouping with --merge.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {"ccs": [], "res": 0, "wsum": 0.0,
                               "ssum": 0.0, "ssum_plain": 0.0, "hi": 0,
                               "land": ""})
    for r in rows:
        hit = attr.get(r["cc"])
        if not hit:
            continue
        pmc, land = hit
        a = agg[pmc]
        a["ccs"].append(r["cc"])
        w = max(r["res_total"], 0)
        a["res"] += w
        a["wsum"] += w
        a["ssum"] += r["uv_score"] * w
        a["ssum_plain"] += r["uv_score"]
        if r["uv_score"] >= build_cut:
            a["hi"] += 1
        if not a["land"]:
            a["land"] = land
    parcels = []
    for pmc, a in agg.items():
        n = len(a["ccs"])
        if n < min_bldg or a["res"] < min_res:
            continue
        score_pw = (a["ssum"] / a["wsum"] if a["wsum"] > 0
                    else a["ssum_plain"] / n)
        parcels.append({"pmc": pmc, "ccs": a["ccs"], "n_bldg": n,
                        "res_total": a["res"], "score_pw": score_pw,
                        "ssum": a["ssum"], "wsum": a["wsum"], "hi": a["hi"],
                        "hi_frac": a["hi"] / n, "land_class": a["land"]})
    parcels = [p for p in parcels if p["score_pw"] >= build_cut]
    # finalize_units assigns the final order using --sort-by; use only a stable preliminary order here.
    parcels.sort(key=lambda p: (p["score_pw"], p["res_total"]), reverse=True)
    print(f"[汇聚·地块] 候选高分地块 {len(parcels):,} 个 "
          f"(score_pw≥{build_cut:.3f}, ≥{min_bldg}栋, ≥{min_res}人)")
    return parcels


def fetch_parcel_geoms(pmcs) -> dict:
    """
    Read candidate statutory-plan parcel geometries as Shapely objects in EPSG:3857 for basemaps
    and EPSG:4326 for Folium.
    """
    import json, psycopg2
    from shapely.geometry import shape
    print(f"[几何] 从 statutory_plan 取 {len(pmcs):,} 地块边界 ...")
    out = {}
    conn = psycopg2.connect(**PG_MAIN)
    pl = sorted(set(pmcs))
    with conn.cursor() as c:
        for i in range(0, len(pl), 5000):
            chunk = pl[i:i+5000]
            c.execute(
                "SELECT upper(main_code), "
                " ST_AsGeoJSON(ST_Transform(geometry,4326)), "
                " ST_AsGeoJSON(ST_Transform(geometry,3857)) "
                "FROM public.statutory_plan WHERE upper(main_code) = ANY(%s)",
                (chunk,))
            for mc, gj4326, gj3857 in c.fetchall():
                if not gj4326:
                    continue
                out[str(mc).strip().upper()] = {
                    "g4326": shape(json.loads(gj4326)),
                    "g3857": shape(json.loads(gj3857))}
    conn.close()
    return out


def _geom_display(g3857, g4326) -> dict:
    """
    Convert Shapely EPSG:3857/4326 geometries to plotting rings, centroid, and span, including
    MultiPolygon support.
    """
    def _parts(g):
        return list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
    polys3857 = [list(zip(*gp.exterior.xy)) for gp in _parts(g3857)]
    polys4326 = [[(y, x) for x, y in zip(*gp.exterior.xy)]
                 for gp in _parts(g4326)]
    cen = g3857.centroid
    cll = g4326.centroid
    minx, miny, maxx, maxy = g3857.bounds
    return {"polys3857": polys3857, "polys4326": polys4326,
            "c3857": (cen.x, cen.y), "latlon": (cll.y, cll.x),
            "span_m": max(maxx - minx, maxy - miny)}


def merge_adjacent(parcels, geom_map, gap):
    """
    Group parcels with union-find when their separation is <=gap meters, union each component's
    outline, and recombine population-weighted attributes. Return units matching the single-
    parcel structure, with semicolon-separated member codes.
    """
    from collections import defaultdict
    from shapely.strtree import STRtree
    from shapely.ops import unary_union
    items = [p for p in parcels if p["pmc"] in geom_map]
    g3857 = [geom_map[p["pmc"]]["g3857"] for p in items]
    g4326 = [geom_map[p["pmc"]]["g4326"] for p in items]
    tree = STRtree(g3857)
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for i, g in enumerate(g3857):
        for j in tree.query(g.buffer(gap)):
            j = int(j)
            if j <= i:
                continue
            if g.distance(g3857[j]) <= gap:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)

    units = []
    for idxs in groups.values():
        mem = [items[i] for i in idxs]
        u3857 = unary_union([g3857[i] for i in idxs])
        u4326 = unary_union([g4326[i] for i in idxs])
        ssum = sum(p["ssum"] for p in mem)
        wsum = sum(p["wsum"] for p in mem)
        n_bldg = sum(p["n_bldg"] for p in mem)
        hi = sum(p["hi"] for p in mem)
        # Land-use class: population-weighted mode among members.
        land_w = defaultdict(int)
        for p in mem:
            land_w[p["land_class"]] += p["res_total"]
        land = max(land_w, key=land_w.get) if land_w else ""
        member_polys = []
        for p in mem:
            gm = geom_map[p["pmc"]]["g3857"]
            parts = list(gm.geoms) if gm.geom_type == "MultiPolygon" else [gm]
            for gp in parts:
                member_polys.append(list(zip(*gp.exterior.xy)))
        u = {"code": ";".join(p["pmc"] for p in mem),
             "n_parcel": len(mem),
             "ccs": [cc for p in mem for cc in p["ccs"]],
             "n_bldg": n_bldg, "res_total": sum(p["res_total"] for p in mem),
             "score_pw": (ssum / wsum if wsum > 0 else 0.0),
             "hi_frac": (hi / n_bldg if n_bldg else 0.0), "land_class": land,
             "member_polys3857": member_polys}
        u.update(_geom_display(u3857, u4326))
        units.append(u)
    units.sort(key=lambda u: (u["score_pw"], u["res_total"]), reverse=True)
    print(f"[合并] {len(items):,} 地块 → {len(units):,} 连片村 "
          f"(间距≤{gap:.0f}m 合并)")
    return units


def parcels_to_units(parcels, geom_map):
    """Without merging, map each candidate parcel to one unit."""
    units = []
    for p in parcels:
        gm = geom_map.get(p["pmc"])
        if not gm:
            continue
        u = {"code": p["pmc"], "n_parcel": 1, "ccs": p["ccs"],
             "n_bldg": p["n_bldg"], "res_total": p["res_total"],
             "score_pw": p["score_pw"], "hi_frac": p["hi_frac"],
             "land_class": p["land_class"]}
        u.update(_geom_display(gm["g3857"], gm["g4326"]))
        units.append(u)
    return units


def finalize_units(units, amap, csvdir, by, sort_by="score",
                   parcel_name_cache=None):
    """
    Assign names/known-name checks, assemble plotting geometries, and write csv/uv_parcels*.csv.
    sort_by=score uses weighted score then population; pop reverses that priority.
    """
    from collections import Counter
    parcel_name_cache = parcel_name_cache or {}
    if sort_by == "pop":
        units = sorted(units, key=lambda u: (u["res_total"], u["score_pw"]),
                       reverse=True)
        sort_label = "人口"
    else:
        units = sorted(units, key=lambda u: (u["score_pw"], u["res_total"]),
                       reverse=True)
        sort_label = "加权分"
    geoms, recs = [], []
    for i, u in enumerate(units):
        names = Counter(_addr_base(amap.get(cc, "")) for cc in u["ccs"]
                        if amap.get(cc))
        name = names.most_common(1)[0][0] if names else ""
        if not name:
            for pmc in str(u["code"]).split(";"):
                pmc = pmc.strip().upper()
                if pmc in parcel_name_cache:
                    name = parcel_name_cache[pmc]
                    break
        if not name:
            name = "(无地址)"
        hit = any(k in name for k in KNOWN_UV)
        vil = bool(_VILLAGE_WORD.search(name))
        rec = {"cluster_id": i + 1, "parcel_main_code": u["code"],
               "n_parcel": u["n_parcel"], "name": name,
               "land_class": u["land_class"], "n_bldg": u["n_bldg"],
               "res_total": u["res_total"], "score_pw": round(u["score_pw"], 3),
               "hi_frac": round(u["hi_frac"], 3),
               "span_m": round(u["span_m"], 0),
               "hit_known": hit, "is_village_name": vil}
        recs.append(rec)
        geoms.append({**rec, "mean_score": rec["score_pw"],
                      "polys3857": u["polys3857"], "polys4326": u["polys4326"],
                      "member_polys3857": u.get("member_polys3857") or [],
                      "c3857": u["c3857"], "latlon": u["latlon"],
                      "_color": _color_for(hit, vil), "ccs": u["ccs"]})
    hitn = sum(1 for r in recs if r["hit_known"])
    viln = sum(1 for r in recs if r["hit_known"] or r["is_village_name"])
    hdr = "连片村" if by == "parcel_merge" else "地块"
    print(f"    Top 候选{hdr} (按{sort_label}, 前20):")
    print(f"    {'#':>3} {'名称':<16} {'用地':<5} {'块':>4} {'栋':>5} {'人口':>8} "
          f"{'加权分':>6} {'高分占比':>6}  证据")
    for r in recs[:20]:
        mark = ("公认*" if r["hit_known"]
                else ("村名" if r["is_village_name"] else ""))
        print(f"    {r['cluster_id']:>3} {r['name'][:16]:<16} "
              f"{r['land_class'][:5]:<5} {r['n_parcel']:>4} {r['n_bldg']:>5} "
              f"{r['res_total']:>8,} {r['score_pw']:>6.3f} {r['hi_frac']:>6.2f}"
              f"  {mark}")
    print(f"    命中公认城中村关键词: {hitn}/{len(recs)}   "
          f"名字自证村落(含公认): {viln}/{len(recs)}")
    fname = ("uv_parcels_merged_by_score.csv" if by == "parcel_merge"
             else "uv_parcels_by_score.csv")
    if sort_by == "pop":
        fname = ("uv_parcels_merged.csv" if by == "parcel_merge"
                 else "uv_parcels.csv")
    path = os.path.join(csvdir, fname)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0].keys()) if recs
                            else ["cluster_id"])
        wr.writeheader()
        for r in recs:
            wr.writerow(r)
    print(f"[写出] {path}  ({len(recs)} 个候选{hdr})")
    return geoms


# ══════════════════════════════════════════════════════════════════════
# Aggregation B: legacy connected components on a 150 m grid, reusing stage_d/stage_e.
# ══════════════════════════════════════════════════════════════════════
def build_grid_geoms(rows, csvdir, top_frac, cell, min_bldg):
    from pyproj import Transformer
    from shapely.geometry import MultiPoint
    th = score_threshold(rows, top_frac)
    clusters = stage_d(rows, th, cell, min_bldg)
    if not clusters:
        return []
    ccs = [r["cc"] for bs in clusters for r in bs]
    amap = fetch_address(ccs)
    for bs in clusters:
        for r in bs:
            r["address"] = amap.get(r["cc"], "")
    recs = stage_e(clusters, csvdir)
    t4326 = Transformer.from_crs(4547, 4326, always_xy=True)
    t3857 = Transformer.from_crs(4547, 3857, always_xy=True)
    geoms = []
    for bs, rec in zip(clusters, recs):
        pts = [r["_xy"] for r in bs if r.get("_xy")]
        if not pts:
            continue
        hull = MultiPoint(pts).convex_hull
        hull = hull.buffer(40.0) if hull.geom_type != "Polygon" else hull.buffer(20.0)
        hx, hy = hull.exterior.xy
        ring3857 = [t3857.transform(x, y) for x, y in zip(hx, hy)]
        ring4326 = [(lat, lon) for lon, lat
                    in (t4326.transform(x, y) for x, y in zip(hx, hy))]
        cx3857, cy3857 = t3857.transform(rec["cx"], rec["cy"])
        clon, clat = t4326.transform(rec["cx"], rec["cy"])
        geoms.append({**rec, "mean_score": rec["mean_score"],
                      "polys3857": [ring3857], "polys4326": [ring4326],
                      "c3857": (cx3857, cy3857), "latlon": (clat, clon),
                      "_color": _color_for(rec["hit_known"], rec["is_village_name"])})
    return geoms


# ---- Basemap: contextily ----
# Basemap providers: Esri (global; possible alignment offsets in China) or Tianditu (CGCS2000,
# aligned with official Chinese data; requires a tk key). Set by main.
_PROVIDER = "esri"
_TDT_KEY = None


def _add_basemap(ax, zoom_adjust=0, retries=4, backoff=1.5):
    """
    Add satellite imagery with bounded retries for transient HTTPS failures. Negative
    zoom_adjust requests coarser tiles for lower transfer cost. This historical figure helper
    can suppress in-panel attribution; retain the required imagery-source credit in the figure
    or caption when using its output.
    """
    import time
    import contextily as cx
    kw = dict(attribution=False)
    if zoom_adjust:
        kw["zoom_adjust"] = zoom_adjust
    if _PROVIDER == "tianditu":
        if not _TDT_KEY:
            raise ValueError("tianditu 需 --tdt-key <天地图密钥>")
        url = ("https://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile"
               "&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles"
               "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=" + _TDT_KEY)
        source = url
    else:
        source = cx.providers.Esri.WorldImagery
    last = None
    for attempt in range(1, retries + 1):
        try:
            cx.add_basemap(ax, source=source, **kw)
            return
        except Exception as e:
            last = e
            if attempt >= retries:
                break
            time.sleep(backoff * attempt)
    raise last


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _nice_length_m(span_m):
    """Choose a rounded ground-distance scale length in meters from the map span."""
    target = max(span_m * 0.22, 20.0)
    for L in (20, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if L >= target * 0.55:
            return L
    return 5000


def _draw_scalebar(ax, ground_m, lat, anno_scale=1.0):
    """
    Draw a segmented black/white scale bar. ground_m is true ground length; convert to EPSG:3857
    map units using its 1/cos(latitude) scale factor.
    """
    import math
    from matplotlib.patches import Rectangle
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    w, h = xlim[1] - xlim[0], ylim[1] - ylim[0]
    merc = ground_m / max(math.cos(math.radians(lat)), 1e-6)
    nseg = 2
    seg = merc / nseg
    x0 = xlim[0] + 0.055 * w
    y0 = ylim[0] + 0.075 * h
    bar_h = 0.011 * h * anno_scale
    padx, pady = 0.018 * w, 0.010 * h
    label = f"{int(ground_m)} m" if ground_m < 1000 else f"{ground_m/1000:g} km"
    box_h = bar_h + 0.052 * h * anno_scale
    ax.add_patch(Rectangle((x0 - padx, y0 - pady), merc + 2 * padx, box_h,
                           facecolor="white", edgecolor="none", alpha=0.72,
                           zorder=19))
    for i in range(nseg):
        fc = "black" if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((x0 + i * seg, y0), seg, bar_h, facecolor=fc,
                               edgecolor="black", lw=0.6, zorder=20))
    ax.text(x0 + merc / 2, y0 + bar_h + 0.008 * h, label, ha="center",
            va="bottom", fontsize=7.5 * anno_scale, color="black", zorder=22)


def _draw_north_arrow(ax, anno_scale=1.0):
    """Draw a two-tone north arrow in the upper-right corner."""
    from matplotlib.patches import Polygon
    import matplotlib.patheffects as pe
    cx, cy = 0.94, 0.85
    s = 0.05 * anno_scale
    tip = (cx, cy + s)
    base = cy - s
    mid = (cx, cy - 0.5 * s)
    left = (cx - 0.42 * s, base)
    right = (cx + 0.42 * s, base)
    ax.add_patch(Polygon([tip, mid, left], closed=True, transform=ax.transAxes,
                         facecolor="white", edgecolor="black", lw=0.7,
                         zorder=22))
    ax.add_patch(Polygon([tip, mid, right], closed=True, transform=ax.transAxes,
                         facecolor="black", edgecolor="black", lw=0.7,
                         zorder=22))
    ax.text(cx, cy + s + 0.015, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9 * anno_scale, fontweight="bold",
            color="white", zorder=23,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])


def _render_region(g, outdir, color, dpi, pad_frac, pad_min, zoom_adjust,
                   figsize, line_width, fill_alpha, range_only, save_img,
                   paper_anno=False, show_parcel_lines=False,
                   show_attribution=False, panel_label=None):
    """
    Render one region with Matplotlib's object-oriented API inside a worker. paper_anno adds a
    scale bar/north arrow; panel_label supplies the ASCII panel identifier while place names
    belong in captions. show_parcel_lines overlays member boundaries for merged regions;
    show_attribution adds an imagery-source label, which may instead be supplied in the
    composite caption.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.patheffects as pe
    allx = [p[0] for ring in g["polys3857"] for p in ring]
    ally = [p[1] for ring in g["polys3857"] for p in ring]
    if not allx:
        return False
    w, h = max(allx) - min(allx), max(ally) - min(ally)
    pad = max(w, h) * pad_frac + pad_min
    fig = Figure(figsize=(figsize, figsize)); FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(min(allx) - pad, max(allx) + pad)
    ax.set_ylim(min(ally) - pad, max(ally) + pad)
    try:
        _add_basemap(ax, zoom_adjust=zoom_adjust)
    except Exception as e:
        print(f"    [警告] #{g['cluster_id']} 底图失败({e})")
    safe = "".join(ch for ch in g["name"]
                   if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
                   or ch in "一二三四五六七八九十村围坊新旧")
    base = f"{g['cluster_id']:03d}_{safe[:12] or 'region'}"
    if save_img and not range_only:
        fig.savefig(os.path.join(outdir, base + "_img.png"), dpi=dpi,
                    facecolor="white")
    # If requested, draw thin member-parcel boundaries before the candidate's outer boundary.
    if show_parcel_lines and g.get("member_polys3857") and g.get("n_parcel", 1) > 1:
        for ring in g["member_polys3857"]:
            xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
            ax.plot(xs, ys, color="white", lw=max(line_width * 0.55, 0.4),
                    alpha=0.85, zorder=5)
            ax.plot(xs, ys, color="#FFCC00", lw=max(line_width * 0.4, 0.3),
                    alpha=0.9, zorder=5.5)
    for ring in g["polys3857"]:
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        if fill_alpha > 0:
            ax.fill(xs, ys, color=color, alpha=fill_alpha, zorder=6)
        ax.plot(xs, ys, color=color, lw=line_width, zorder=7)
    anno_scale = max(figsize / 6.0, 0.75)
    if paper_anno:
        anno_scale = max(anno_scale, 1.45)
    if panel_label:
        ax.text(0.035, 0.955, str(panel_label), transform=ax.transAxes,
                ha="left", va="top", fontsize=11 * anno_scale,
                fontweight="bold", color="black", zorder=25,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", linewidth=0.6, alpha=0.88))
    if paper_anno:
        span = max(ax.get_xlim()[1] - ax.get_xlim()[0],
                   ax.get_ylim()[1] - ax.get_ylim()[0])
        lat = (g.get("latlon") or (22.5, 114.0))[0]
        _draw_scalebar(ax, _nice_length_m(span), lat, anno_scale=anno_scale)
        _draw_north_arrow(ax, anno_scale=anno_scale)
    if show_attribution:
        src = ("Esri World Imagery" if _PROVIDER == "esri"
               else "Tianditu imagery")
        ax.text(0.01, 0.015, src, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6 * anno_scale, color="white", alpha=0.9, zorder=25,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])
    fig.savefig(os.path.join(outdir, base + "_range.png"), dpi=dpi,
                facecolor="white")
    return True


def _panel_label(idx, mode):
    """
    Use a zero-based panel index. Modes: none, alphabetic labels (a)..(z),(aa).., or id using
    cluster_id.
    """
    if mode == "none":
        return None
    if mode == "letter":
        s, n = "", idx
        while True:
            s = chr(ord("a") + n % 26) + s
            n = n // 26 - 1
            if n < 0:
                break
        return f"({s})"
    return None  # In id mode, the caller supplies cluster_id.


def make_regions(geoms, outdir, color, dpi=200, pad_frac=0.22, pad_min=90,
                 limit=0, workers=8, zoom_adjust=-1, figsize=6.0,
                 line_width=0.8, fill_alpha=0.12, range_only=False,
                 paper_anno=False, show_parcel_lines=False,
                 show_attribution=False, label_mode="letter"):
    """
    Export two views for every region with the same extent: <id>_img.png contains imagery;
    <id>_range.png adds boundaries and optional panel label/scale/north arrow. label_mode is
    letter, cluster id, or none. Put place names in captions using ranking.csv rather than
    embedding them in the panel.
    """
    import contextily as cx
    from concurrent.futures import ThreadPoolExecutor
    import threading
    _setup_mpl()
    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(outdir, ".tile_cache")
    os.makedirs(cache, exist_ok=True)
    try:
        cx.set_cache_dir(cache)
    except Exception:
        pass
    gs = geoms[:limit] if limit and limit > 0 else geoms
    n = len(gs)
    per = 1 if range_only else 2
    print(f"[出图·全区域] {n} 个区域 × {per}张 → {outdir}  "
          f"(底图:{_PROVIDER}, {workers}线程, zoom{zoom_adjust}, 色{color}, "
          f"线宽{line_width}, paper_anno={paper_anno}, label={label_mode}, "
          f"{'仅range' if range_only else 'img+range'})")
    done = {"k": 0, "ok": 0}
    lock = threading.Lock()
    t0 = __import__("time").perf_counter

    def _task(args):
        idx, g = args
        if label_mode == "id":
            label = f"#{g['cluster_id']}"
        else:
            label = _panel_label(idx, label_mode)
        r = _render_region(g, outdir, color, dpi, pad_frac, pad_min,
                           zoom_adjust, figsize, line_width, fill_alpha,
                           range_only, True, paper_anno=paper_anno,
                           show_parcel_lines=show_parcel_lines,
                           show_attribution=show_attribution,
                           panel_label=label)
        with lock:
            done["k"] += 1
            if r:
                done["ok"] += 1
            k = done["k"]
        if k % 25 == 0 or k == n:
            print(f"    {k}/{n} ...")
        return r

    ts = t0()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_task, list(enumerate(gs))))
    # Write a local ranking with panel labels to support selecting panels and writing captions.
    rank_path = os.path.join(outdir, "ranking.csv")
    with open(rank_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["panel", "cluster_id", "name", "score_pw", "res_total",
                  "n_bldg", "n_parcel", "hit_known", "is_village_name",
                  "land_class"]
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for idx, g in enumerate(gs):
            if label_mode == "id":
                g = {**g, "panel": f"#{g['cluster_id']}"}
            else:
                g = {**g, "panel": _panel_label(idx, label_mode) or ""}
            wr.writerow(g)
    print(f"[写出] {rank_path}")
    print(f"[出图·全区域] 完成 {done['ok']} 个区域 → {outdir}  "
          f"{t0()-ts:.0f}s")


def make_static(geoms, outpath, by, label_top=25, dpi=220):
    plt = _setup_mpl()
    from matplotlib.patches import Patch
    import contextily as cx
    fig, ax = plt.subplots(figsize=(13, 11))
    for g in geoms:
        for ring in g["polys3857"]:
            xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
            ax.fill(xs, ys, color=g["_color"], alpha=0.34, zorder=3)
            ax.plot(xs, ys, color=g["_color"], lw=0.9, alpha=0.95, zorder=4)
    for g in geoms[:label_top]:
        x, y = g["c3857"]
        tag = " *" if g["hit_known"] else ""
        ax.annotate(f"{g['name']}{tag}", (x, y), fontsize=7.0, color="white",
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black",
                              ec=g["_color"], lw=0.8, alpha=0.72))
    try:
        _add_basemap(ax)
    except Exception as e:
        print(f"    [警告] 卫星底图拉取失败({e}); 出无底图版本。")
    ax.set_axis_off()
    hit = sum(1 for g in geoms if g["hit_known"])
    unit = {"parcel": "地块", "parcel_merge": "连片村"}.get(by, "连片区")
    ax.set_title(f"深圳疑似城中村{unit} (G3R 图内评分, 按{unit}汇聚)  "
                 f"共 {len(geoms)} 个 · 命中公认城中村关键词 {hit} 个",
                 fontsize=12, pad=8)
    leg = [Patch(fc=C_HIT, alpha=0.5, label="命中公认城中村 (标 *)"),
           Patch(fc=C_VIL, alpha=0.5, label="名字自证村落(村/围/坊)"),
           Patch(fc=C_OTH, alpha=0.5, label=f"其它高分{unit}")]
    ax.legend(handles=leg, loc="lower left", framealpha=0.85, fontsize=8)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[出图] {outpath}")


def make_zoom(geoms, outdir, n_top, dpi=220):
    plt = _setup_mpl()
    import contextily as cx
    os.makedirs(outdir, exist_ok=True)
    for g in geoms[:n_top]:
        allx = [p[0] for ring in g["polys3857"] for p in ring]
        ally = [p[1] for ring in g["polys3857"] for p in ring]
        fig, ax = plt.subplots(figsize=(7, 7))
        for ring in g["polys3857"]:
            xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
            ax.fill(xs, ys, color=g["_color"], alpha=0.30, zorder=3)
            ax.plot(xs, ys, color=g["_color"], lw=2.0, zorder=4)
        pad = max(max(allx)-min(allx), max(ally)-min(ally)) * 0.35 + 120
        ax.set_xlim(min(allx)-pad, max(allx)+pad)
        ax.set_ylim(min(ally)-pad, max(ally)+pad)
        try:
            _add_basemap(ax)
        except Exception as e:
            print(f"    [警告] 特写底图失败({e})")
        ax.set_axis_off()
        tag = " [公认城中村]" if g["hit_known"] else ""
        ax.set_title(f"#{g['cluster_id']} {g['name']}{tag}\n"
                     f"{g['n_bldg']}栋 · {g['res_total']:,}人 · 分{g['mean_score']:.3f}",
                     fontsize=11)
        safe = "".join(ch for ch in g["name"]
                       if ch.isalnum() or ch in "一二三四五六七八九十村围坊")
        p = os.path.join(outdir, f"uv_zoom_{g['cluster_id']:02d}_{safe[:10]}.png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[出图] {p}")


# ---- Interactive HTML: Folium with Esri imagery ----
def make_html(geoms, outpath, by):
    import folium
    if not geoms:
        return
    lat0 = sum(g["latlon"][0] for g in geoms) / len(geoms)
    lon0 = sum(g["latlon"][1] for g in geoms) / len(geoms)
    m = folium.Map(location=[lat0, lon0], zoom_start=11, tiles=None,
                   control_scale=True)
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        attr="Esri World Imagery", name="Esri 卫星影像",
        overlay=False, control=True).add_to(m)
    unit = {"parcel": "地块", "parcel_merge": "连片村"}.get(by, "连片区")
    for g in geoms:
        extra = (f" · 用地 {g.get('land_class','')}"
                 if by in ("parcel", "parcel_merge") else "")
        popup = folium.Popup(
            f"<b>{g['name']}</b>{' *公认' if g['hit_known'] else ''}<br>"
            f"栋数 {g['n_bldg']} · 人口 {g['res_total']:,}<br>"
            f"分 {g['mean_score']:.3f}{extra}", max_width=240)
        for ring in g["polys4326"]:
            folium.Polygon(
                locations=ring, color=g["_color"], weight=2,
                fill=True, fill_color=g["_color"], fill_opacity=0.32,
                popup=popup, tooltip=f"{g['name']}({unit})").add_to(m)
        folium.CircleMarker(location=g["latlon"], radius=3, color=g["_color"],
                            fill=True, fill_opacity=0.9).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(outpath)
    print(f"[出图] {outpath}")


def write_members(geoms, rows_by_cc, parcels_by_id, outpath, by):
    """
    Export member-building records for reproducibility; parcel mode includes parcel_main_code.
    Review data-sharing restrictions before distributing these outputs.
    """
    with open(outpath, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        if by == "parcel":
            wr.writerow(["cluster_id", "parcel_main_code", "building_cc",
                         "uv_score", "res_total"])
            for g in geoms:
                p = parcels_by_id.get(g["cluster_id"])
                if not p:
                    continue
                for cc in p["ccs"]:
                    r = rows_by_cc.get(cc)
                    if r:
                        wr.writerow([g["cluster_id"], g["parcel_main_code"], cc,
                                     f"{r['uv_score']:.4f}", r["res_total"]])
    print(f"[写出] {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["parcel", "grid"], default="parcel")
    ap.add_argument("--scores", default="csv/g3r_uv_building_score.csv")
    ap.add_argument("--attr", default="csv/g3r_parcel_attribution.csv")
    ap.add_argument("--outdir", default="paper_figures")
    ap.add_argument("--csvdir", default="csv")
    ap.add_argument("--top", type=float, default=0.15)
    ap.add_argument("--parcel-th", type=float, default=None,
                    help="地块加权分阈值; 默认=建筑 top 分位阈值")
    ap.add_argument("--min-bldg", type=int, default=None)
    ap.add_argument("--min-res", type=int, default=300)
    ap.add_argument("--cell", type=float, default=150.0)
    ap.add_argument("--merge", action="store_true",
                    help="按地块时: 相邻候选地块并集合并成整村轮廓")
    ap.add_argument("--merge-gap", type=float, default=20.0,
                    help="合并的地块间距阈值(米), 桥接村内道路/退线; 默认20")
    ap.add_argument("--zoom-top", type=int, default=0)
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--sort-by", choices=["score", "pop"], default="score",
                    help="候选排序: score=加权分(默认); pop=人口(旧口径)")
    # Export an imagery view and a boundary view for each region.
    ap.add_argument("--regions", action="store_true",
                    help="每个区域各出 影像图+范围图 到子文件夹(默认只出这个)")
    ap.add_argument("--regions-dir", default="paper_figures/uv_regions_by_score",
                    help="区域图输出目录(默认按加权分新目录)")
    ap.add_argument("--limit-regions", type=int, default=0,
                    help="仅出前 N 个区域(调试/挑图用; 0=全部)")
    ap.add_argument("--workers", type=int, default=8,
                    help="每区域出图的并行线程数(网络受限, 默认8)")
    ap.add_argument("--zoom-adjust", type=int, default=-1,
                    help="底图瓦片层级调整(负=更少瓦片更快, 默认-1)")
    ap.add_argument("--dpi", type=int, default=200,
                    help="每区域出图 DPI(默认200→1200px; 高DPI建议配 --zoom-adjust 0)")
    ap.add_argument("--figsize", type=float, default=6.0,
                    help="图幅英寸(方形); 像素=figsize×dpi。默认6→(×200)1200px")
    ap.add_argument("--color", default="#FFFF00",
                    help="统一多边形颜色(默认黄); 每区域出图用此色")
    ap.add_argument("--line-width", type=float, default=1.4,
                    help="范围边界线宽(论文默认1.4)")
    ap.add_argument("--fill-alpha", type=float, default=0.12,
                    help="范围填充透明度(默认0.12淡填充; 0=只描边不填充)")
    ap.add_argument("--range-only", action="store_true",
                    help="只出范围图, 不出纯影像图")
    ap.add_argument("--paper-anno", action="store_true", default=True,
                    help="论文标注: 比例尺、指北针、ASCII面板编号(默认开; 图中无中文)")
    ap.add_argument("--no-paper-anno", action="store_true",
                    help="关闭论文标注")
    ap.add_argument("--label-mode", choices=["letter", "id", "none"],
                    default="letter",
                    help="面板编号: letter=(a)(b)..(默认); id=#cluster_id; none=不标")
    ap.add_argument("--show-parcel-lines", action="store_true",
                    help="连片合并时叠加成员地块边界")
    ap.add_argument("--show-attribution", action="store_true",
                    help="图内角标影像来源(多子图时更建议写进总图注)")
    ap.add_argument("--provider", choices=["esri", "tianditu"], default="esri",
                    help="底图: esri(默认) / tianditu(天地图, 与中国官方对齐, 需 --tdt-key)")
    ap.add_argument("--tdt-key", default=None, help="天地图 tk 密钥")
    args = ap.parse_args()

    global _PROVIDER, _TDT_KEY
    _PROVIDER = args.provider
    _TDT_KEY = args.tdt_key
    os.makedirs(args.outdir, exist_ok=True)
    paper_anno = args.paper_anno and not args.no_paper_anno

    rows = load_rows(args.scores)
    rows_by_cc = {r["cc"]: r for r in rows}
    build_cut = score_threshold(rows, args.top)
    parcels_by_id = {}

    by_tag = args.by
    if args.by == "parcel":
        min_bldg = args.min_bldg if args.min_bldg is not None else 5
        th = args.parcel_th if args.parcel_th is not None else build_cut
        attr = load_attribution(args.attr)
        parcels = aggregate_by_parcel(rows, attr, th, min_bldg, args.min_res)
        if not parcels:
            print("[结束] 无候选地块。"); return
        amap = fetch_address([cc for p in parcels for cc in p["ccs"]],
                             csvdir=args.csvdir)
        name_cache = {}
        if "__parcel_name_cache__" in amap:
            name_cache = amap.pop("__parcel_name_cache__")
        elif not amap:
            name_cache = load_parcel_name_cache(args.csvdir)
        geom_map = fetch_parcel_geoms([p["pmc"] for p in parcels])
        if args.merge:
            by_tag = "parcel_merge"
            units = merge_adjacent(parcels, geom_map, args.merge_gap)
        else:
            units = parcels_to_units(parcels, geom_map)
        geoms = finalize_units(units, amap, args.csvdir, by_tag,
                               sort_by=args.sort_by,
                               parcel_name_cache=name_cache)
        for g in geoms:
            parcels_by_id[g["cluster_id"]] = {"ccs": g["ccs"],
                                              "code": g["parcel_main_code"]}
    else:
        min_bldg = args.min_bldg if args.min_bldg is not None else 10
        geoms = build_grid_geoms(rows, args.csvdir, args.top, args.cell, min_bldg)
        if not geoms:
            print("[结束] 无连片区。"); return

    print(f"[几何] {args.by} 汇聚 {len(geoms)} 个已就绪。")
    if args.by == "parcel":
        memp = os.path.join(args.outdir, "uv_parcel_buildings.csv")
        write_members(geoms, rows_by_cc, parcels_by_id, memp, args.by)

    # --regions exports an imagery view and a boundary view for each region.
    if args.regions:
        make_regions(geoms, args.regions_dir, args.color, dpi=args.dpi,
                     limit=args.limit_regions, workers=args.workers,
                     zoom_adjust=args.zoom_adjust, figsize=args.figsize,
                     line_width=args.line_width, fill_alpha=args.fill_alpha,
                     range_only=args.range_only, paper_anno=paper_anno,
                     show_parcel_lines=args.show_parcel_lines,
                     show_attribution=args.show_attribution,
                     label_mode=args.label_mode)
        print("[完成]")
        return

    if not args.no_static:
        make_static(geoms, os.path.join(args.outdir, "uv_satellite.png"), by_tag)
    if args.zoom_top > 0:
        make_zoom(geoms, args.outdir, args.zoom_top)
    if not args.no_html:
        make_html(geoms, os.path.join(args.outdir, "uv_satellite.html"), by_tag)
    print("[完成]")


if __name__ == "__main__":
    main()

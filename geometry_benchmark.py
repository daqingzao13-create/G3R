#!/usr/bin/env python3
# coding: utf-8
"""
Batch distance benchmark for all nine Point/LineString/Polygon type combinations. Record
per-stage timings in nine detailed CSVs and a combined summary.

Usage:
    python geometry_benchmark.py <output_directory> [options]
    python geometry_benchmark.py benchmark_v1 -n 100
    python geometry_benchmark.py benchmark_v1 -n 50 --no-postgis --seed 42
"""

import csv
import math
import os
import random
import statistics
import sys
import time
from itertools import product
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from geosot_geometry_reasoning import (
    GeoSotGeometryReasoner,
    SOURCE_TO_PG_TABLE,
    _build_cos_ref_matrix,
    fast_estimate_vertices_int,
    geometry_min_distance_m,
    geometry_min_distance_m_int,
)
from geosot_spatial_reasoning import (
    compute_bearing,
    fast_code_to_center,
    vincenty_fast,
    HAS_NUMBA,
)
from bench_setup import prepare_pg_warm

TYPES = ("Point", "LineString", "Polygon")

CSV_HEADERS = [
    "pair_idx",
    "code_a", "code_b",
    "type_a", "type_b",
    "source_a", "source_b",
    "vertex_count_a", "vertex_count_b",
    # Neo4j
    "neo4j_info_a_ms", "neo4j_info_b_ms", "neo4j_total_ms",
    # Grid decoding.
    "dec_coord_a_ms", "dec_coord_b_ms",
    "dec_geometry_distance_ms", "dec_bearing_ms", "dec_total_ms",
    "dec_distance_m", "dec_bearing_deg",
    # Graph reasoning.
    "gr_prepare_codes_ms", "gr_vertex_estimate_ms",
    "gr_geometry_distance_ms", "gr_bearing_ms", "gr_total_ms",
    "gr_distance_m", "gr_bearing_deg",
    # PostGIS
    "pg_query_ms", "pg_distance_m", "pg_bearing_deg",
    # Distance error.
    "err_dec_vs_gr_m", "err_dec_vs_pg_m", "err_gr_vs_pg_m",
    "err_dec_dist_pct", "err_gr_dist_pct",
    # Azimuth error.
    "err_dec_vs_gr_bearing_deg",
    "err_dec_vs_pg_bearing_deg", "err_gr_vs_pg_bearing_deg",
]

TIMING_FIELDS = [
    ("neo4j_info_a_ms", "Neo4j 查询 A"),
    ("neo4j_info_b_ms", "Neo4j 查询 B"),
    ("neo4j_total_ms", "Neo4j 合计"),
    ("dec_coord_a_ms", "解码坐标 A"),
    ("dec_coord_b_ms", "解码坐标 B"),
    ("dec_geometry_distance_ms", "解码几何距离"),
    ("dec_bearing_ms", "解码方位角"),
    ("dec_total_ms", "解码合计"),
    ("gr_prepare_codes_ms", "图推理准备编码"),
    ("gr_vertex_estimate_ms", "图推理顶点估算"),
    ("gr_geometry_distance_ms", "图推理几何距离"),
    ("gr_bearing_ms", "图推理方位角"),
    ("gr_total_ms", "图推理合计"),
    ("pg_query_ms", "PostGIS ST_Distance"),
]

ERROR_FIELDS = [
    ("err_dec_vs_gr_m", "解码−图推理 距离差(m)"),
    ("err_dec_vs_pg_m", "解码−PostGIS 距离差(m)"),
    ("err_gr_vs_pg_m", "图推理−PostGIS 距离差(m)"),
    ("err_dec_dist_pct", "解码−PostGIS 距离差(%)"),
    ("err_gr_dist_pct", "图推理−PostGIS 距离差(%)"),
    ("err_dec_vs_gr_bearing_deg", "解码−图推理 方位角差(°)"),
    ("err_dec_vs_pg_bearing_deg", "解码−PostGIS 方位角差(°)"),
    ("err_gr_vs_pg_bearing_deg", "图推理−PostGIS 方位角差(°)"),
]


def _fetch_features_by_type(
    reasoner: GeoSotGeometryReasoner,
    geo_type: str,
    limit: int = 2000,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Draw a uniform, seeded feature pool for one geometry type. Fetch the full pool and use
    random.Random(seed).sample, avoiding the old vertex-count-ordered LIMIT bias toward simpler
    lines/polygons.

    @param reasoner Reasoner instance.
    @param geo_type Point, LineString, or Polygon.
    @param limit Maximum sample size.
    @param seed Reproducible random seed.
    @returns Sampled features.
    """
    return reasoner.list_features(
        geo_type=geo_type, limit=limit,
        random_sample=True, seed=seed,
    )


def _make_pairs(
    features_a: List[Dict],
    features_b: List[Dict],
    n: int,
    rng: random.Random,
) -> List[Tuple[Dict, Dict]]:
    """
    Generate n random feature pairs, excluding a feature paired with itself when types match.

    @param features_a Candidate pool A.
    @param features_b Candidate pool B.
    @param n Number of pairs.
    @param rng Random generator.
    @returns [(feature_a,feature_b), ...].
    """
    pairs = []
    for _ in range(n * 5):
        if len(pairs) >= n:
            break
        a = rng.choice(features_a)
        b = rng.choice(features_b)
        if a.get("code_id") != b.get("code_id"):
            pairs.append((a, b))
    return pairs[:n]


def _run_single_benchmark(
    reasoner: GeoSotGeometryReasoner,
    code_a: str,
    code_b: str,
    compare_postgis: bool,
) -> Optional[Dict]:
    """
    Run one distance comparison with detailed stage timings.

    @param reasoner Reasoner instance.
    @param code_a Feature A identifier.
    @param code_b Feature B identifier.
    @param compare_postgis Whether to run the PG reference.
    @returns Result dictionary, or None on failure.
    """
    _t = time.perf_counter

    # ---- Neo4j ----
    t0 = _t()
    info_a = reasoner.get_feature_info(code_a)
    t_neo4j_a = (_t() - t0) * 1000

    t0 = _t()
    info_b = reasoner.get_feature_info(code_b)
    t_neo4j_b = (_t() - t0) * 1000

    if not info_a or not info_b:
        return None

    type_a = info_a["geometry_type"]
    type_b = info_b["geometry_type"]
    vc_a = len(info_a["vertex_codes"])
    vc_b = len(info_b["vertex_codes"])

    row: Dict = {
        "code_a": code_a,
        "code_b": code_b,
        "type_a": type_a,
        "type_b": type_b,
        "source_a": info_a.get("source", ""),
        "source_b": info_b.get("source", ""),
        "vertex_count_a": vc_a if type_a != "Point" else 1,
        "vertex_count_b": vc_b if type_b != "Point" else 1,
        "neo4j_info_a_ms": round(t_neo4j_a, 3),
        "neo4j_info_b_ms": round(t_neo4j_b, 3),
        "neo4j_total_ms": round(t_neo4j_a + t_neo4j_b, 3),
    }

    # ---- Extract codes and build the pairwise common-ancestor reference matrix ----
    codes_a = reasoner._get_all_codes(info_a)
    codes_b = reasoner._get_all_codes(info_b)
    cos_matrix = _build_cos_ref_matrix(codes_a, codes_b)

    # ---- Grid decoding ----
    t0 = _t()
    dec_a = reasoner._get_coords_decode(info_a)
    t_dec_ca = (_t() - t0) * 1000

    t0 = _t()
    dec_b = reasoner._get_coords_decode(info_b)
    t_dec_cb = (_t() - t0) * 1000

    t0 = _t()
    dec_dist, dec_na, dec_nb = geometry_min_distance_m(
        dec_a, type_a, dec_b, type_b,
        cos_ref_matrix=cos_matrix)
    t_dec_geom = (_t() - t0) * 1000

    t0 = _t()
    dec_brg = (compute_bearing(dec_na[0], dec_na[1], dec_nb[0], dec_nb[1])
               if dec_dist > 0.001 else 0.0)
    t_dec_brg = (_t() - t0) * 1000

    t_dec_total = t_dec_ca + t_dec_cb + t_dec_geom + t_dec_brg

    row.update({
        "dec_coord_a_ms": round(t_dec_ca, 3),
        "dec_coord_b_ms": round(t_dec_cb, 3),
        "dec_geometry_distance_ms": round(t_dec_geom, 3),
        "dec_bearing_ms": round(t_dec_brg, 3),
        "dec_total_ms": round(t_dec_total, 3),
        "dec_distance_m": round(dec_dist, 4),
        "dec_bearing_deg": round(dec_brg, 4),
    })

    # ---- Integer-predicate graph reasoning, following the distance operator ----
    # Integer offsets (1/U degree) -> intersection/projection/argmin predicates
    # -> one final coordinate reconstruction and one Vincenty call.
    t_gr_prep = 0.0

    t0 = _t()
    gr_a_int, gr_b_int, gr_ref_lon, gr_ref_lat, gr_cos_ref = (
        fast_estimate_vertices_int(codes_a, codes_b))
    t_gr_est = (_t() - t0) * 1000

    t0 = _t()
    gr_dist, gr_na, gr_nb = geometry_min_distance_m_int(
        gr_a_int, type_a, gr_b_int, type_b,
        gr_ref_lon, gr_ref_lat, gr_cos_ref)
    t_gr_geom = (_t() - t0) * 1000

    t0 = _t()
    gr_brg = (compute_bearing(gr_na[0], gr_na[1], gr_nb[0], gr_nb[1])
              if gr_dist > 0.001 else 0.0)
    t_gr_brg = (_t() - t0) * 1000

    t_gr_total = t_gr_prep + t_gr_est + t_gr_geom + t_gr_brg

    row.update({
        "gr_prepare_codes_ms": round(t_gr_prep, 3),
        "gr_vertex_estimate_ms": round(t_gr_est, 3),
        "gr_geometry_distance_ms": round(t_gr_geom, 3),
        "gr_bearing_ms": round(t_gr_brg, 3),
        "gr_total_ms": round(t_gr_total, 3),
        "gr_distance_m": round(gr_dist, 4),
        "gr_bearing_deg": round(gr_brg, 4),
    })

    # ---- PostGIS ----
    pg_dist = None
    pg_brg = None
    if compare_postgis:
        t0 = _t()
        pg_result = reasoner._postgis_distance_and_bearing(
            info_a["main_code"], info_a.get("source", ""),
            info_b["main_code"], info_b.get("source", ""))
        t_pg = (_t() - t0) * 1000
        row["pg_query_ms"] = round(t_pg, 3)
        if pg_result is not None:
            pg_dist, pg_brg = pg_result
            row["pg_distance_m"] = round(pg_dist, 4)
            row["pg_bearing_deg"] = round(pg_brg, 4)
        else:
            row["pg_distance_m"] = ""
            row["pg_bearing_deg"] = ""
    else:
        row["pg_query_ms"] = ""
        row["pg_distance_m"] = ""
        row["pg_bearing_deg"] = ""

    # ---- Distance error ----
    row["err_dec_vs_gr_m"] = round(dec_dist - gr_dist, 6)
    if pg_dist is not None and pg_dist > 0.01:
        row["err_dec_vs_pg_m"] = round(dec_dist - pg_dist, 6)
        row["err_gr_vs_pg_m"] = round(gr_dist - pg_dist, 6)
        row["err_dec_dist_pct"] = round(
            (dec_dist - pg_dist) / pg_dist * 100, 8)
        row["err_gr_dist_pct"] = round(
            (gr_dist - pg_dist) / pg_dist * 100, 8)
    else:
        row["err_dec_vs_pg_m"] = ""
        row["err_gr_vs_pg_m"] = ""
        row["err_dec_dist_pct"] = ""
        row["err_gr_dist_pct"] = ""

    # ---- Azimuth error ----
    def _brg_diff(a, b):
        d = a - b
        if d > 180:
            d -= 360
        elif d < -180:
            d += 360
        return d

    if dec_dist > 0.001 and gr_dist > 0.001:
        row["err_dec_vs_gr_bearing_deg"] = round(
            _brg_diff(dec_brg, gr_brg), 6)
    else:
        row["err_dec_vs_gr_bearing_deg"] = ""

    if pg_brg is not None and pg_dist is not None and pg_dist > 0.01:
        if dec_dist > 0.001:
            row["err_dec_vs_pg_bearing_deg"] = round(
                _brg_diff(dec_brg, pg_brg), 6)
        else:
            row["err_dec_vs_pg_bearing_deg"] = ""
        if gr_dist > 0.001:
            row["err_gr_vs_pg_bearing_deg"] = round(
                _brg_diff(gr_brg, pg_brg), 6)
        else:
            row["err_gr_vs_pg_bearing_deg"] = ""
    else:
        row["err_dec_vs_pg_bearing_deg"] = ""
        row["err_gr_vs_pg_bearing_deg"] = ""

    return row


def _collect_stats(rows: List[Dict], key: str) -> Dict:
    """
    Collect statistics for one field.

    @param rows Result rows.
    @param key Field name.
    @returns {count,mean,median,std,min,max}, or an empty dictionary.
    """
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return {}
    return {
        "count": len(vals),
        "mean": round(statistics.mean(vals), 3),
        "median": round(statistics.median(vals), 3),
        "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
    }


def _print_combo_summary(combo_name: str, rows: List[Dict],
                         compare_postgis: bool):
    """Print the statistics for one geometry combination."""
    n = len(rows)
    if n == 0:
        print(f"  {combo_name}: 无有效数据")
        return

    active_timing = [(f, l) for f, l in TIMING_FIELDS
                     if compare_postgis or f != "pg_query_ms"]

    W = 76
    print(f"\n  {'─' * (W - 4)}")
    print(f"  {combo_name}  (n={n})")
    print(f"  {'─' * (W - 4)}")
    print(f"  {'步骤':<26s} {'均值':>10s} {'中位数':>10s} "
          f"{'标准差':>10s} {'最小':>10s} {'最大':>10s}")
    print(f"  {'─' * 26} {'─' * 10} {'─' * 10} "
          f"{'─' * 10} {'─' * 10} {'─' * 10}")

    for field, label in active_timing:
        s = _collect_stats(rows, field)
        if not s:
            continue
        print(f"  {label:<26s} "
              f"{s['mean']:>9.3f}  {s['median']:>9.3f}  "
              f"{s['std']:>9.3f}  {s['min']:>9.3f}  {s['max']:>9.3f}")

    print(f"\n  距离误差:")
    for field, label in ERROR_FIELDS[:5]:
        s = _collect_stats(rows, field)
        if not s:
            continue
        print(f"    {label:<30s}  均值={s['mean']:+.6f}  "
              f"中位={s['median']:+.6f}  "
              f"max=|{max(abs(s['min']), abs(s['max'])):.6f}|")

    print(f"\n  方位角误差:")
    for field, label in ERROR_FIELDS[5:]:
        s = _collect_stats(rows, field)
        if not s:
            continue
        print(f"    {label:<30s}  均值={s['mean']:+.6f}°  "
              f"中位={s['median']:+.6f}°  "
              f"max=|{max(abs(s['min']), abs(s['max'])):.6f}°|")


def _build_summary_rows(
    all_results: Dict[str, List[Dict]],
    combos: List[Tuple[str, str]],
    compare_postgis: bool,
) -> Tuple[List[str], List[Dict]]:
    """
    Build one summary row per geometry combination plus a pooled row.

    @param all_results {combination: [rows]}.
    @param combos Combination list.
    @param compare_postgis Whether PG results are included.
    @returns (headers,rows).
    """
    stat_suffixes = ["_mean", "_median", "_std", "_min", "_max", "_count"]

    active_timing = [(f, l) for f, l in TIMING_FIELDS
                     if compare_postgis or f != "pg_query_ms"]
    active_errors = ERROR_FIELDS if compare_postgis else [
        e for e in ERROR_FIELDS
        if "pg" not in e[0].lower() and "postgis" not in e[1].lower()
    ]

    metric_fields = active_timing + active_errors

    headers = ["combination", "valid_pairs"]
    for field, _ in metric_fields:
        for suf in stat_suffixes:
            headers.append(field + suf)

    summary_rows: List[Dict] = []
    all_flat: List[Dict] = []

    for ta, tb in combos:
        combo_key = f"{ta}_x_{tb}"
        combo_label = f"{ta} × {tb}"
        rows = all_results.get(combo_key, [])
        all_flat.extend(rows)

        sr: Dict = {"combination": combo_label, "valid_pairs": len(rows)}
        for field, _ in metric_fields:
            s = _collect_stats(rows, field)
            for suf in stat_suffixes:
                sr[field + suf] = s.get(suf.lstrip("_"), "")
        summary_rows.append(sr)

    sr_all: Dict = {"combination": "全部组合", "valid_pairs": len(all_flat)}
    for field, _ in metric_fields:
        s = _collect_stats(all_flat, field)
        for suf in stat_suffixes:
            sr_all[field + suf] = s.get(suf.lstrip("_"), "")
    summary_rows.append(sr_all)

    return headers, summary_rows


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GeoSOT 几何要素距离 批量基准测试 (9 种 P/L/Po 组合)",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "output_dir",
        help="输出文件夹名称 (自动创建)")
    parser.add_argument(
        "-n", "--count", type=int, default=100,
        help="每种组合的测试对数 (默认 100)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认 42)")
    parser.add_argument(
        "--no-postgis", action="store_true",
        help="不对比 PostGIS")
    parser.add_argument(
        "--fetch-limit", type=int, default=2000,
        help="每种类型最多取多少候选要素 (默认 2000)")
    parser.add_argument(
        "--no-prewarm", action="store_true",
        help="跳过 pg_prewarm (cold-cache 对照实验; 默认会 prewarm)")
    parser.add_argument(
        "--throwaway", type=int, default=5,
        help="正式计时前丢弃的 throwaway 对数 (默认 5, 同时热双侧)")

    args = parser.parse_args()
    out_dir = args.output_dir
    n_per_combo = args.count
    seed = args.seed
    compare_pg = not args.no_postgis
    rng = random.Random(seed)

    os.makedirs(out_dir, exist_ok=True)

    W = 76
    print("=" * W)
    print("GeoSOT 几何要素距离 批量基准测试")
    print("=" * W)
    print(f"  输出目录:     {os.path.abspath(out_dir)}")
    print(f"  每组测试数:   {n_per_combo}")
    print(f"  随机种子:     {seed}")
    print(f"  PostGIS 对比: {'是' if compare_pg else '否'}")
    print(f"  Numba JIT:    {'已启用' if HAS_NUMBA else '未安装'}")
    print("=" * W)

    # Initialize the reasoner.
    reasoner = GeoSotGeometryReasoner(
        verbose=True, enable_postgis=compare_pg)

    # Fetch candidate features by type; unbiased sampling is reproducible through seed.
    print("\n[1/3] 从 Neo4j 获取候选要素 (random_sample, seed=%d)..."
          % seed)
    features: Dict[str, List[Dict]] = {}
    for t in TYPES:
        feats = _fetch_features_by_type(
            reasoner, t, limit=args.fetch_limit, seed=seed)
        features[t] = feats
        if feats:
            vcs = [int(f.get("vertex_count") or 0) for f in feats]
            vc_min, vc_max = min(vcs), max(vcs)
            vc_mean = sum(vcs) / len(vcs)
            print(f"  {t:<12s}: {len(feats):>5d} 个要素  "
                  f"vc=[{vc_min}, {vc_max}], mean={vc_mean:.1f}")
        else:
            print(f"  {t:<12s}: 0 个要素")

    # Check which geometry combinations are available.
    combos = list(product(TYPES, TYPES))
    skipped_combos = []
    active_combos = []
    for ta, tb in combos:
        if not features[ta] or not features[tb]:
            skipped_combos.append((ta, tb))
        else:
            active_combos.append((ta, tb))

    if skipped_combos:
        print(f"\n  [跳过] 因候选要素不足:")
        for ta, tb in skipped_combos:
            print(f"    {ta} × {tb}")

    # ---- Benchmark setup: prepare warm caches on both sides ----
    # Disclose G3R client-side prefetch and Numba JIT setup,
    # and PostGIS pg_prewarm and plan-cache warmup. Measure both sides after warmup.
    if compare_pg and reasoner._pg_conn is not None and not args.no_prewarm:
        # SOURCE_TO_PG_TABLE keys are Neo4j Source.name values,
        # not Point/LineString/Polygon. Extract sources actually present in the feature pool,
        # then map them to PG tables so prewarm does not silently select zero tables.
        used_sources = set()
        for geo_type in TYPES:
            for feat in features.get(geo_type, []):
                src = (feat.get("source") or "").strip()
                if src:
                    used_sources.add(src)
        used_tables = sorted({
            SOURCE_TO_PG_TABLE[src] for src in used_sources
            if src in SOURCE_TO_PG_TABLE
        })
        prepare_pg_warm(
            reasoner._pg_conn,
            tables=used_tables,
            throwaway_queries=[
                (
                    f"SELECT main_code FROM {tbl} LIMIT 50",
                    None,
                )
                for tbl in used_tables
            ],
        )

    # ---- G3R: prefetch the full feature pool into the client cache once ----
    # As with PG prewarm, fetches during the measured G3R phase are cache hits (approximately microseconds).
    if not args.no_prewarm:
        all_code_ids = []
        for geo_type in TYPES:
            for feat in features.get(geo_type, []):
                cid = feat.get("code_id") or feat.get("main_code")
                if cid:
                    all_code_ids.append(cid)
        if all_code_ids:
            print()
            print("─" * 76)
            print("  [bench_setup] GR client-side prefetch "
                  "(对应 PG pg_prewarm)")
            print("─" * 76)
            prep = reasoner.prefetch_features(all_code_ids)
            print(f"  prefetch 完成: {prep['n_fetch']:,} 条新加载, "
                  f"{prep['n_hit']:,} 已缓存, "
                  f"{prep['n_miss']:,} 未命中, "
                  f"耗时 {prep['elapsed_sec'] * 1000:.0f} ms "
                  f"(此项不计入正式计时, 与 pg_prewarm 同等披露)")
            print("─" * 76)
            print()

    if args.throwaway > 0 and active_combos:
        n_tw = args.throwaway
        print(f"\n[setup] 双侧 throwaway: 每组合丢弃前 {n_tw} 对 "
              f"(热 Neo4j cache / Numba JIT / PG plan cache)")
        for ta, tb in active_combos:
            tw_pairs = _make_pairs(
                features[ta], features[tb], n_tw, rng)
            for fa, fb in tw_pairs:
                cid_a = fa.get("code_id", fa.get("main_code", ""))
                cid_b = fb.get("code_id", fb.get("main_code", ""))
                _run_single_benchmark(
                    reasoner, cid_a, cid_b, compare_pg)
        print(f"  throwaway 完成, 进入正式计时阶段")

    total_pairs = len(active_combos) * n_per_combo
    print(f"\n[2/3] 批量计算 ({len(active_combos)} 种组合 × "
          f"{n_per_combo} 对 = {total_pairs} 次)")
    print()

    all_results: Dict[str, List[Dict]] = {}
    global_bar = tqdm(total=total_pairs, desc="总进度", unit="对",
                      ncols=90, position=0)

    for ta, tb in combos:
        combo_key = f"{ta}_x_{tb}"
        combo_label = f"{ta} × {tb}"

        if (ta, tb) in skipped_combos:
            csv_path = os.path.join(out_dir, f"{combo_key}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=CSV_HEADERS)
                writer.writeheader()
            all_results[combo_key] = []
            continue

        pairs = _make_pairs(features[ta], features[tb], n_per_combo, rng)
        rows: List[Dict] = []

        combo_bar = tqdm(pairs, desc=f"  {combo_label:<20s}",
                         unit="对", ncols=90, position=1, leave=False)

        for idx, (fa, fb) in enumerate(combo_bar):
            cid_a = fa.get("code_id", fa.get("main_code", ""))
            cid_b = fb.get("code_id", fb.get("main_code", ""))
            result = _run_single_benchmark(
                reasoner, cid_a, cid_b, compare_pg)
            if result:
                result["pair_idx"] = idx
                rows.append(result)
            global_bar.update(1)

        combo_bar.close()

        csv_path = os.path.join(out_dir, f"{combo_key}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for r in rows:
                writer.writerow({h: r.get(h, "") for h in CSV_HEADERS})

        all_results[combo_key] = rows
        tqdm.write(f"  ✓ {combo_label:<20s}  "
                   f"{len(rows):>4d} 条  → {csv_path}")

    global_bar.close()

    # ---- Summary ----
    print(f"\n[3/3] 统计汇总")
    print("=" * W)

    total_valid = 0
    for ta, tb in combos:
        combo_key = f"{ta}_x_{tb}"
        combo_label = f"{ta} × {tb}"
        rows = all_results.get(combo_key, [])
        total_valid += len(rows)
        _print_combo_summary(combo_label, rows, compare_pg)

    all_rows = []
    for v in all_results.values():
        all_rows.extend(v)

    if all_rows:
        _print_combo_summary("全部组合 (汇总)", all_rows, compare_pg)

    # ---- Tenth CSV: combined summary ----
    summary_headers, summary_rows = _build_summary_rows(
        all_results, combos, compare_pg)
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_headers)
        writer.writeheader()
        for sr in summary_rows:
            writer.writerow({h: sr.get(h, "") for h in summary_headers})

    print(f"\n{'=' * W}")
    print(f"  有效计算总数: {total_valid}")
    print(f"  输出目录:     {os.path.abspath(out_dir)}")
    print(f"  CSV 文件:     {len(combos) + 1} 个")
    for ta, tb in combos:
        combo_key = f"{ta}_x_{tb}"
        n_rows = len(all_results.get(combo_key, []))
        mark = "✓" if n_rows > 0 else "○"
        print(f"    {mark} {combo_key}.csv  ({n_rows} 条)")
    print(f"    ★ summary.csv  (汇总统计, {len(summary_rows)} 行)")
    print(f"{'=' * W}")

    reasoner.close()
    print("\n完成。")


if __name__ == "__main__":
    main()

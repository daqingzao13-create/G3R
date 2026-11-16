#!/usr/bin/env python3
# coding: utf-8
"""
Forward Overlay V7 Paper Figure Generator

Reads V7 CSV (from geosot_overlay_query_v7.py --all --output ...).

Generates:
  fig_forward_overlay_v7_story.{fmt}      — 1x4 main figure (paper)
    (a) Latency Breakdown: G3R (graph stage) vs Geometric post-processing
    (b) Speedup Distribution (CDF) — long-tail advantage
    (c) Speedup vs Complexity (fill prefix count) — scaling behavior
    (d) Set-level Accuracy (Recall / Precision / F1 CDF)

  fig_forward_overlay_v7_breakdown.{fmt}  — 1x2 latency breakdown
    (a) 6-step detailed bar (fetch / xy / grid / match / fill / PIP)
    (b) 3-group aggregated bar (Data prep / G3R graph stage / Geom)

  fig_forward_overlay_v7.{fmt}            — 1x3 supplementary
    (a) Fill prefix count vs query time (binned median + IQR)
    (b) Empirical CDF of per-query time (V7 vs PostGIS)
    (c) Inside + boundary fill vs final hits (scatter + y=x)

  fig_forward_overlay_v7_area.{fmt}       — 1x2 area / span analysis
    (a) Polygon area (m²) vs total query time
    (b) Lon/Lat span vs total query time

Usage:
    python overlay_v7_analysis.py
    python overlay_v7_analysis.py --csv result_v7_2.csv --format pdf --dpi 600
"""

import argparse
import os
import sys
from matplotlib.transforms import Bbox
from paper_figure_style import export_consistent_panels


def _export_axes_as_standalone(fig, axes_dict, out_dir, fmt, dpi,
                               name_prefix, strip_panel_label=True,
                               pad_inches=0.20):
    """
    Crop composite axes to standalone files for LaTeX. axes_dict maps panel labels to either one
    axis or an axis sequence; use their union bounding box. strip_panel_label removes embedded
    (a)-style titles to avoid duplicate subfloat labels.
    """
    exported = export_consistent_panels(
        fig, axes_dict, out_dir, fmt, dpi, name_prefix)
    axes_dict = {key: ax for key, ax in axes_dict.items() if key not in exported}
    if not axes_dict:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    saved_titles = {}
    if strip_panel_label:
        for letter, ax_or_seq in axes_dict.items():
            seq = ax_or_seq if isinstance(ax_or_seq, (tuple, list)) \
                else (ax_or_seq,)
            saved_titles[letter] = [ax.get_title() for ax in seq]
            for ax in seq:
                ax.set_title("")

    try:
        for letter, ax_or_seq in axes_dict.items():
            seq = ax_or_seq if isinstance(ax_or_seq, (tuple, list)) \
                else (ax_or_seq,)
            keep = set(seq)
            hidden = []
            for o in fig.axes:
                if o not in keep and o.get_visible():
                    hidden.append(o)
                    o.set_visible(False)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bboxes = [
                ax.get_tightbbox(renderer).transformed(
                    fig.dpi_scale_trans.inverted())
                for ax in seq
            ]
            extent = Bbox.union(bboxes).padded(pad_inches)
            path = os.path.join(out_dir,
                                f"{name_prefix}_panel_{letter}.{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches=extent,
                        facecolor="white")
            for o in hidden:
                o.set_visible(True)
    finally:
        if strip_panel_label:
            for letter, ax_or_seq in axes_dict.items():
                seq = ax_or_seq if isinstance(ax_or_seq, (tuple, list)) \
                    else (ax_or_seq,)
                for ax, t in zip(seq, saved_titles[letter]):
                    ax.set_title(t)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.6,
})

# Unified palette aligned with multiscale_analysis / reverse_query_analysis:
#   G3R (overall)        → deep blue (#1565C0)
#   PostGIS              → mid-gray  (#9E9E9E)
#   accent (highlight)   → orange    (#FB8C00)
C_GR = "#1565C0"            # G3R primary color: deep blue.
C_PG = "#9E9E9E"            # PostGIS primary color: medium gray.
C_PREP = "#FB8C00"
C_FILL = "#43A047"
C_MATCH = "#64B5F6"
C_PIP = "#E53935"
C_FAST = "#8E24AA"
C_FILL_ONLY = "#FB8C00"

# Stage breakdown (Data prep / Graph reasoning / Geometric post-proc)
C_GROUP_PREP  = "#BBDEFB"   # Light blue: data preparation.
C_GROUP_GRAPH = "#1565C0"   # Deep blue: graph reasoning, matching C_GR.
C_GROUP_GEOM  = "#FB8C00"   # Orange: geometric post-processing.
C_HIGHLIGHT   = "#43A047"   # Green accent.


def _binned_median_iqr(x, y, n_bins=42, edges=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        m = float(np.median(y))
        return np.array([lo]), np.array([m]), np.array([m]), np.array([m])
    if edges is None:
        edges = np.linspace(lo, hi, n_bins + 1)
    else:
        edges = np.asarray(edges, dtype=float)

    med, _, _ = binned_statistic(x, y, statistic="median", bins=edges)
    q25, _, _ = binned_statistic(
        x, y, statistic=lambda a: np.percentile(a, 25) if len(a) else np.nan,
        bins=edges)
    q75, _, _ = binned_statistic(
        x, y, statistic=lambda a: np.percentile(a, 75) if len(a) else np.nan,
        bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = ~np.isnan(med)
    return (centers[valid], med[valid].astype(float),
            q25[valid].astype(float), q75[valid].astype(float))


def _ecdf(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.array([]), np.array([])
    y = np.sort(y)
    return y, np.arange(1, len(y) + 1, dtype=float) / len(y)


def _clean_ax(ax, grid_axis="y"):
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, alpha=0.12,
            linewidth=0.4, color="#999999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBBB")
        ax.spines[s].set_linewidth(0.5)


def _prepare_df(df):
    df = df.copy()
    if "gr_time_ms" in df.columns:
        df["t_gr"] = pd.to_numeric(df["gr_time_ms"], errors="coerce")
    else:
        df["t_gr"] = pd.to_numeric(df["t_gr_total_ms"], errors="coerce")
    if "pg_time_ms" in df.columns:
        df["t_pg"] = pd.to_numeric(df["pg_time_ms"], errors="coerce")
    else:
        df["t_pg"] = pd.to_numeric(df["t_pg_ms"], errors="coerce")

    if "strategy" not in df.columns:
        df["strategy"] = "unknown"
    else:
        df["strategy"] = (
            df["strategy"].astype(str).str.strip().str.lower()
        )
    return df


# ======================================================================
# Figure 1: 1x3 composite
# ======================================================================

def fig_paper(df, out_dir, fmt, dpi):
    df = _prepare_df(df)
    df["n_pf"] = pd.to_numeric(df["n_fill_total"], errors="coerce")
    df["hits"] = pd.to_numeric(df["n_hits"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) Fill prefix count vs query time
    ax = axes[0]
    _clean_ax(ax, "both")

    tmp = df.dropna(subset=["n_pf", "t_gr", "t_pg"]).copy()
    tmp = tmp[tmp["n_pf"] > 0].copy()
    x_cap = float(np.percentile(tmp["n_pf"], 99))
    plot_df = tmp[tmp["n_pf"] <= x_cap].copy()

    scatter_df = plot_df
    if len(scatter_df) > 15000:
        scatter_df = scatter_df.sample(n=15000, random_state=42)

    ax.scatter(scatter_df["n_pf"], scatter_df["t_pg"],
               s=10, alpha=0.14, color=C_PG, edgecolors="none",
               linewidths=0, rasterized=True, zorder=1)
    fast_scatter = scatter_df[scatter_df["strategy"] == "fast_path"]
    fill_scatter = scatter_df[scatter_df["strategy"] == "polygon_fill"]
    ax.scatter(fill_scatter["n_pf"], fill_scatter["t_gr"],
               s=10, alpha=0.16, color=C_FILL_ONLY, edgecolors="none",
               linewidths=0, rasterized=True, zorder=2)
    ax.scatter(fast_scatter["n_pf"], fast_scatter["t_gr"],
               s=10, alpha=0.16, color=C_FAST, edgecolors="none",
               linewidths=0, rasterized=True, zorder=2)

    x_arr = plot_df["n_pf"].values
    bin_edges = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), 43)

    cx_pg, med_pg, q25_pg, q75_pg = _binned_median_iqr(
        x_arr, plot_df["t_pg"].values, edges=bin_edges)
    fast_df = plot_df[plot_df["strategy"] == "fast_path"]
    fill_df = plot_df[plot_df["strategy"] == "polygon_fill"]
    cx_fast, med_fast, q25_fast, q75_fast = _binned_median_iqr(
        fast_df["n_pf"].values, fast_df["t_gr"].values, edges=bin_edges)
    cx_fill, med_fill, q25_fill, q75_fill = _binned_median_iqr(
        fill_df["n_pf"].values, fill_df["t_gr"].values, edges=bin_edges)
    if len(cx_pg):
        ax.fill_between(cx_pg, q25_pg, q75_pg, color=C_PG, alpha=0.22,
                         linewidth=0, zorder=3)
        ax.plot(cx_pg, med_pg, color=C_PG, linewidth=2.2, zorder=5,
                label="PostGIS")
    if len(cx_fill):
        ax.fill_between(cx_fill, q25_fill, q75_fill, color=C_FILL_ONLY, alpha=0.22,
                         linewidth=0, zorder=4)
        ax.plot(cx_fill, med_fill, color=C_FILL_ONLY, linewidth=2.2, zorder=6,
                label="GeoSOT Polygon Fill")
    if len(cx_fast):
        ax.fill_between(cx_fast, q25_fast, q75_fast, color=C_FAST, alpha=0.22,
                        linewidth=0, zorder=4)
        ax.plot(cx_fast, med_fast, color=C_FAST, linewidth=2.2, zorder=6,
                label="GeoSOT Direct Prefix")

    ax.set_xlabel("Number of Fill Prefixes")
    ax.set_ylabel("Query Time (ms)")
    ax.set_xlim(0, x_cap * 1.02)
    ax.set_ylim(0, min(150, np.percentile(plot_df["t_gr"], 99) * 1.3))
    ax.legend(fontsize=7.5, loc="upper left",
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title("(a) Query Time vs. Fill Prefix Count",
                 fontsize=10, fontweight="bold")

    # (b) Empirical CDF of query time
    ax = axes[1]
    _clean_ax(ax)

    t_fast = df.loc[df["strategy"] == "fast_path", "t_gr"].dropna().values
    t_fill = df.loc[df["strategy"] == "polygon_fill", "t_gr"].dropna().values
    t_pg_all = df["t_pg"].dropna().values
    x_fast, f_fast = _ecdf(t_fast)
    x_fill, f_fill = _ecdf(t_fill)
    x_pg, f_pg = _ecdf(t_pg_all)
    ax.plot(x_pg, f_pg, color=C_PG, linewidth=2.0, label="PostGIS")
    if len(x_fill):
        ax.plot(x_fill, f_fill, color=C_FILL_ONLY, linewidth=2.0,
                label="GeoSOT Polygon Fill")
    if len(x_fast):
        ax.plot(x_fast, f_fast, color=C_FAST, linewidth=2.0,
                label="GeoSOT Direct Prefix")
    gr_all = np.concatenate([t_fast, t_fill]) if (len(t_fast) + len(t_fill)) else np.array([])
    all_for_xlim = np.concatenate([gr_all, t_pg_all]) if len(gr_all) else t_pg_all
    x_hi = float(np.percentile(all_for_xlim, 99.5)) if len(all_for_xlim) else 1.0
    ax.set_xlim(0, min(max(x_hi * 1.02, 1.0), 500))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Query Time (ms)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=7.5, loc="lower right",
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title("(b) Query Time Distribution",
                 fontsize=10, fontweight="bold")

    # (c) Fill prefixes vs final hits
    ax = axes[2]
    _clean_ax(ax, "both")

    pf_v = df["n_pf"].dropna().values
    hits_v = df["hits"].dropna().values
    n_pts = min(len(pf_v), len(hits_v))
    pf_v, hits_v = pf_v[:n_pts], hits_v[:n_pts]

    ax.scatter(pf_v, hits_v, s=4, alpha=0.07, color=C_GR,
               edgecolors="none", rasterized=True)
    lim = max(np.percentile(pf_v, 99),
              np.percentile(hits_v, 99)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.7, alpha=0.25,
            label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, max(np.percentile(hits_v, 99) * 1.1, 1))
    ax.set_xlabel("Fill Prefixes (inside + boundary)")
    ax.set_ylabel("Final Hits (after winding PIP)")
    ax.legend(fontsize=7, loc="upper left",
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title("(c) Fill Efficiency & Accuracy",
                 fontsize=10, fontweight="bold")

    fig.tight_layout(w_pad=2.5)
    p = os.path.join(out_dir, f"fig_forward_overlay_v7.{fmt}")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    print(f"  [Paper 1x3] -> {p}")


# ======================================================================
# Figure 2: Stacked bar — mean latency breakdown
# ======================================================================

def fig_time_breakdown(df, out_dir, fmt, dpi):
    """
    Two-panel timing breakdown: six stages on the left (fetch, vertex_xy, grid, query, fill,
    PIP) and three groups on the right (preparation, graph reasoning, geometric post-
    processing), matching the publication palette.
    """
    df = _prepare_df(df)
    tpg = pd.to_numeric(df["t_pg"], errors="coerce")
    fast = df[df["strategy"] == "fast_path"]
    fill = df[df["strategy"] == "polygon_fill"]

    def _means_detailed(sub_df):
        tf = pd.to_numeric(sub_df["t_fetch_ms"], errors="coerce").fillna(0)
        tvxy = pd.to_numeric(sub_df["t_vertex_xy_ms"], errors="coerce").fillna(0)
        tg = pd.to_numeric(sub_df["t_grid_ms"], errors="coerce").fillna(0)
        tfill = pd.to_numeric(sub_df["t_fill_ms"], errors="coerce").fillna(0)
        tq = pd.to_numeric(sub_df["t_query_ms"], errors="coerce").fillna(0)
        tpip = pd.to_numeric(sub_df["t_pip_ms"], errors="coerce").fillna(0)
        if len(sub_df) == 0:
            return (0.0,) * 6
        return (
            float(np.nanmean(tf.values)),
            float(np.nanmean(tvxy.values)),
            float(np.nanmean(tg.values)),
            float(np.nanmean(tq.values)),
            float(np.nanmean(tfill.values)),
            float(np.nanmean(tpip.values)),
        )

    md_fast = _means_detailed(fast)
    md_fill = _means_detailed(fill)
    m_pg = float(np.nanmean(tpg.values))

    fast_p, fast_g, fast_geom = _means_grouped(fast)
    fill_p, fill_g, fill_geom = _means_grouped(fill)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---------- (a) Six-stage breakdown ----------
    ax = axes[0]
    _clean_ax(ax)

    x = np.array([0, 1, 2])
    width = 0.55

    parts_def = [
        ("Fetch (Neo4j)", C_PREP),
        ("Vertex→XY", "#FFCC80"),
        ("Grid encode", "#90CAF9"),
        ("Graph match (STARTS WITH)", C_MATCH),
        ("Polygon fill (mask+BFS)", C_FILL),
        ("Winding PIP (boundary)", C_PIP),
    ]

    bottom = 0.0
    for v, (lbl, col) in zip(md_fast, parts_def):
        if v > 0:
            ax.bar(x[0], v, width, bottom=bottom, color=col, alpha=0.9,
                   edgecolor="white", linewidth=0.5, label=lbl)
            bottom += v
    total_fast = bottom
    if total_fast > 0:
        ax.text(x[0], total_fast * 1.01, f"{total_fast:.2f}",
                ha="center", va="bottom", fontsize=7.8)

    bottom = 0.0
    for v, (_lbl, col) in zip(md_fill, parts_def):
        if v > 0:
            ax.bar(x[1], v, width, bottom=bottom, color=col, alpha=0.9,
                   edgecolor="white", linewidth=0.5)
            bottom += v
    total_fill = bottom
    if total_fill > 0:
        ax.text(x[1], total_fill * 1.01, f"{total_fill:.2f}",
                ha="center", va="bottom", fontsize=7.8)

    ax.bar(x[2], m_pg, width, color=C_PG, alpha=0.9,
           edgecolor="white", linewidth=0.5,
           label="PostGIS ST_Intersects")
    if m_pg > 0:
        ax.text(x[2], m_pg * 1.01, f"{m_pg:.2f}",
                ha="center", va="bottom", fontsize=7.8)

    ax.set_xticks(x)
    ax.set_xticklabels(["G3R Direct", "G3R Fill", "PostGIS"], fontsize=9)
    ax.set_ylabel("Mean Time (ms)")
    ax.set_title("(a) Detailed Breakdown (6 steps)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.8, loc="upper right",
              framealpha=0.92, edgecolor="#CCCCCC")
    y_max_a = max(total_fast, total_fill, m_pg)
    if y_max_a > 0:
        ax.set_ylim(0, y_max_a * 1.15)

    # ---------- (b) Three aggregated groups ----------
    ax = axes[1]
    _clean_ax(ax)

    bottom = 0.0
    ax.bar(x[0], fast_p, width, bottom=bottom,
           color=C_GROUP_PREP, edgecolor="white", linewidth=0.6,
           label="Data prep (fetch+xy)")
    bottom += fast_p
    ax.bar(x[0], fast_g, width, bottom=bottom,
           color=C_GROUP_GRAPH, edgecolor="white", linewidth=0.6,
           label="G3R (graph stage)")
    bottom += fast_g
    ax.bar(x[0], fast_geom, width, bottom=bottom,
           color=C_GROUP_GEOM, edgecolor="white", linewidth=0.6,
           label="Geometric post-proc (fill+PIP)")
    h0 = bottom + fast_geom
    if h0 > 0:
        ax.text(x[0], h0 * 1.01, f"{h0:.2f}",
                ha="center", va="bottom", fontsize=7.8)

    bottom = 0.0
    ax.bar(x[1], fill_p, width, bottom=bottom,
           color=C_GROUP_PREP, edgecolor="white", linewidth=0.6)
    bottom += fill_p
    ax.bar(x[1], fill_g, width, bottom=bottom,
           color=C_GROUP_GRAPH, edgecolor="white", linewidth=0.6)
    bottom += fill_g
    ax.bar(x[1], fill_geom, width, bottom=bottom,
           color=C_GROUP_GEOM, edgecolor="white", linewidth=0.6)
    h1 = bottom + fill_geom
    if h1 > 0:
        ax.text(x[1], h1 * 1.01, f"{h1:.2f}",
                ha="center", va="bottom", fontsize=7.8)

    ax.bar(x[2], m_pg, width, color=C_PG, edgecolor="white",
           linewidth=0.6, label="PostGIS ST_Intersects")
    if m_pg > 0:
        ax.text(x[2], m_pg * 1.01, f"{m_pg:.2f}",
                ha="center", va="bottom", fontsize=7.8)

    if fast_g > 0 and m_pg > 0:
        ratio = m_pg / max(fast_g, 1e-9)
        geom_share = (fast_geom / max(h0, 1e-9)) * 100
        ax.text(
            0.55, 0.97,
            f"G3R graph stage is {ratio:.0f}× faster than PG\n"
            f"Geometric post-processing dominates "
            f"({geom_share:.0f}% of G3R Direct Prefix)",
            transform=ax.transAxes, fontsize=7.5,
            ha="left", va="top",
            color=C_GROUP_GRAPH, fontweight="bold",
            bbox=dict(facecolor="white",
                      edgecolor=C_GROUP_GRAPH,
                      alpha=0.92, pad=2.5, linewidth=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(["G3R Direct", "G3R Fill", "PostGIS"], fontsize=9)
    ax.set_ylabel("Mean Time (ms)")
    ax.set_title("(b) Grouped: G3R vs Geometric",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.2, loc="upper left",
              framealpha=0.92, edgecolor="#CCCCCC")
    y_max_b = max(h0, h1, m_pg)
    if y_max_b > 0:
        ax.set_ylim(0, y_max_b * 1.18)

    fig.tight_layout(w_pad=2.5)
    p = os.path.join(out_dir, f"fig_forward_overlay_v7_breakdown.{fmt}")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    print(f"  [Breakdown] -> {p}")


# ======================================================================
# Publication figure: four panels.
# (a) Latency Breakdown — G3R (graph stage) vs Geometric post-processing
# (b) Speedup Distribution (CDF) — G3R shines on long-tail queries
# (c) Speedup vs Complexity — G3R's advantage grows with complexity
# (d) Accuracy CDF — F1 / Recall / Precision distribution
# ======================================================================

def _means_grouped(sub):
    """
    Aggregate six G3R stages into three mean timings.

    @returns (prep_mean,graph_mean,geom_mean), where prep=fetch+vertex_xy, graph=grid+query, and
        geom=fill+PIP.
    """
    if len(sub) == 0:
        return 0.0, 0.0, 0.0
    prep = (
        pd.to_numeric(sub["t_fetch_ms"], errors="coerce").fillna(0)
        + pd.to_numeric(sub["t_vertex_xy_ms"], errors="coerce").fillna(0)
    )
    graph = (
        pd.to_numeric(sub["t_grid_ms"], errors="coerce").fillna(0)
        + pd.to_numeric(sub["t_query_ms"], errors="coerce").fillna(0)
    )
    geom = (
        pd.to_numeric(sub["t_fill_ms"], errors="coerce").fillna(0)
        + pd.to_numeric(sub["t_pip_ms"], errors="coerce").fillna(0)
    )
    return (
        float(np.nanmean(prep.values)),
        float(np.nanmean(graph.values)),
        float(np.nanmean(geom.values)),
    )


def fig_paper_story(df, out_dir, fmt, dpi):
    """
    Four-panel forward-overlay figure showing timing breakdown, complexity-dependent speedup,
    latency tails, and set-level accuracy. The plotted data determine the observed advantages
    and limitations.
    """
    df = _prepare_df(df)

    fig = plt.figure(figsize=(11.0, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    ax_c = fig.add_subplot(gs[1, 0])

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = ax_a
    axes[0, 1] = ax_b
    axes[1, 0] = ax_c
    axes[1, 1] = ax_d

    fast = df[df["strategy"] == "fast_path"]
    fill = df[df["strategy"] == "polygon_fill"]
    fast_p, fast_g, fast_geom = _means_grouped(fast)
    fill_p, fill_g, fill_geom = _means_grouped(fill)

    pg_t = float(np.nanmean(
        pd.to_numeric(df["t_pg"], errors="coerce").values))

    pg_query = float(np.nanmean(
        pd.to_numeric(df["t_query_ms"], errors="coerce")
        .fillna(0).values))

    # ----------------- (a) Latency Breakdown -----------------
    ax = axes[0, 0]
    _clean_ax(ax)

    x = np.arange(3)
    width = 0.55

    def _stack(xi, prep, graph, geom, with_label=False):
        kw = dict(width=width, edgecolor="white", linewidth=0.6)
        ax.bar(xi, prep, color=C_GROUP_PREP,
               label="Data preparation" if with_label else None, **kw)
        ax.bar(xi, graph, bottom=prep, color=C_GROUP_GRAPH,
               label="Graph reasoning" if with_label else None, **kw)
        ax.bar(xi, geom, bottom=prep + graph, color=C_GROUP_GEOM,
               label="Geometric verification" if with_label else None, **kw)
        return prep + graph + geom

    h0 = _stack(x[0], fast_p, fast_g, fast_geom, with_label=True)
    h1 = _stack(x[1], fill_p, fill_g, fill_geom)
    ax.bar(x[2], pg_t, width, color=C_PG, edgecolor="white",
           linewidth=0.6, label="PostGIS")

    if fast_g > 0 and pg_t > 0:
        y_max = max(h0, h1, pg_t)
        ax.set_ylim(0, y_max * 1.32)

    ax.set_xticks(x)
    ax.set_xticklabels(["G3R\nDirect Prefix", "G3R\nPolygon Fill", "PostGIS"],
                       fontsize=12)
    ax.set_ylabel("Mean Time (ms)")
    ax.legend(loc="upper center",
              ncol=2, fontsize=11,
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title("(a) End-to-End Latency Breakdown",
                 fontsize=9.5, fontweight="bold")

    # ----------------- (b) Speedup CDF (per strategy) -----------------
    ax = axes[0, 1]
    _clean_ax(ax)

    C_FAST_LINE = "#1565C0"
    C_FILL_LINE = "#E65100"

    fast_sp = pd.to_numeric(
        fast["speedup"], errors="coerce").dropna().values
    fast_sp = fast_sp[(fast_sp > 0) & np.isfinite(fast_sp)]
    fill_sp = pd.to_numeric(
        fill["speedup"], errors="coerce").dropna().values
    fill_sp = fill_sp[(fill_sp > 0) & np.isfinite(fill_sp)]

    all_sp = np.concatenate([fast_sp, fill_sp]) \
        if (len(fast_sp) + len(fill_sp)) > 0 else np.array([])

    if len(fast_sp) > 0:
        xf, ff = _ecdf(fast_sp)
        ax.plot(xf, ff, color=C_FAST_LINE, linewidth=2.2,
                label="Direct Prefix", zorder=4)
    if len(fill_sp) > 0:
        xg, fg = _ecdf(fill_sp)
        ax.plot(xg, fg, color=C_FILL_LINE, linewidth=2.2,
                linestyle="--",
                label="Polygon Fill", zorder=3)

    if len(all_sp) > 0:
        max_sp_all = float(np.max(all_sp))
        ax.axvline(1.0, color="black", linewidth=0.8,
                   linestyle="--", alpha=0.6)
        ax.axvspan(1.0, max_sp_all * 1.5,
                   color="#1565C0", alpha=0.05, zorder=1)

    if len(fast_sp) > 0:
        m_f = float(np.median(fast_sp))
        p95_f = float(np.percentile(fast_sp, 95))
        ax.scatter([m_f], [0.5], s=45, color=C_FAST_LINE,
                   edgecolors="white", linewidths=1.0, zorder=6)
        ax.scatter([p95_f], [0.95], s=45, color=C_FAST_LINE,
                   edgecolors="white", linewidths=1.0, zorder=6)

    if len(fill_sp) > 0:
        m_g = float(np.median(fill_sp))
        ax.scatter([m_g], [0.5], s=45, color=C_FILL_LINE,
                   edgecolors="white", linewidths=1.0,
                   marker="s", zorder=6)

    if len(all_sp) > 0:
        x_lo = max(0.01, float(np.percentile(all_sp, 0.3)) * 0.9)
        x_hi = float(np.max(all_sp)) * 1.5
        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Speedup  (PG / G3R, log scale)")
    ax.set_ylabel("CDF")
    ax.legend(loc="lower right", ncol=1, fontsize=11,
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title("(b) Speedup Distribution by Strategy",
                 fontsize=10, fontweight="bold")

    # ----------------- (c) Speedup vs Complexity (single log axis) ------
    # Direct Prefix has n_fill=1; summarize it by median and IQR at x=1.
    # Polygon Fill has n_fill in [16,255]; show binned median/IQR and P95
    # to illustrate the speedup trend with complexity. The two strategies occupy disjoint complexity ranges.
    ax = axes[1, 0]
    _clean_ax(ax, "both")

    y_clip_lo, y_clip_hi = 0.1, 100.0

    df["n_pf"] = pd.to_numeric(df["n_fill_total"], errors="coerce")
    tmp = df.dropna(subset=["n_pf", "speedup", "strategy"]).copy()
    tmp = tmp[(tmp["n_pf"] > 0) & (tmp["speedup"] > 0)
              & np.isfinite(tmp["speedup"])]

    fast_df = tmp[tmp["strategy"] == "fast_path"]
    fill_df = tmp[tmp["strategy"] == "polygon_fill"]

    x_hi_axis = 300.0

    # ---- Polygon Fill: scatter density, binned median/IQR, and P95 trend ----
    if len(fill_df) > 0:
        x_min_f = float(fill_df["n_pf"].min())
        x_max_f = float(np.percentile(fill_df["n_pf"], 99.7))
        sub = fill_df[fill_df["n_pf"] <= x_max_f]
        yy = sub["speedup"].clip(y_clip_lo, y_clip_hi).values
        ax.scatter(sub["n_pf"].values, yy, s=4, alpha=0.06,
                   color=C_FILL_LINE, edgecolors="none",
                   rasterized=True, zorder=1)

        log_edges = np.logspace(
            np.log10(max(x_min_f, 2.0)),
            np.log10(x_max_f), 18)
        cx_g, med_g, q25_g, q75_g = _binned_median_iqr(
            sub["n_pf"].values, sub["speedup"].values,
            edges=log_edges)
        p95_arr_g, _, _ = binned_statistic(
            sub["n_pf"].values, sub["speedup"].values,
            statistic=lambda a: np.percentile(a, 95)
            if len(a) else np.nan,
            bins=log_edges)
        centers_g = 0.5 * (log_edges[:-1] + log_edges[1:])
        valid_p95_g = ~np.isnan(p95_arr_g)

        if len(cx_g):
            ax.fill_between(cx_g, q25_g, q75_g,
                            color=C_FILL_LINE, alpha=0.18,
                            linewidth=0, zorder=3)
            ax.plot(cx_g, med_g, color=C_FILL_LINE,
                    linewidth=2.0, label="Polygon Fill: median + IQR",
                    zorder=4)
        if valid_p95_g.any():
            ax.plot(centers_g[valid_p95_g],
                    p95_arr_g[valid_p95_g],
                    color="#BF360C", linewidth=1.6,
                    linestyle="--", marker="o", markersize=3,
                    label="Polygon Fill: P95", zorder=4)
        x_hi_axis = x_max_f * 1.3

    # ---- Direct Prefix: median/IQR at n_fill=1 ----
    if len(fast_df) > 0:
        fast_med = float(np.median(fast_df["speedup"]))
        fast_p25 = float(np.percentile(fast_df["speedup"], 25))
        fast_p75 = float(np.percentile(fast_df["speedup"], 75))
        fast_p95 = float(np.percentile(fast_df["speedup"], 95))

        ax.errorbar(
            [1.0], [fast_med],
            yerr=[[fast_med - fast_p25], [fast_p75 - fast_med]],
            fmt='o', color=C_FAST_LINE, ecolor=C_FAST_LINE,
            elinewidth=2.2, capsize=6, capthick=1.5,
            markersize=8.5, markeredgecolor="white",
            markeredgewidth=1.2, zorder=6,
            label="Direct Prefix: median + IQR")
        ax.scatter([1.0], [min(fast_p95, y_clip_hi)], s=70,
                   color="#0D47A1", marker="^", edgecolors="white",
                   linewidths=1.2, zorder=7,
                   label="Direct Prefix: P95")

    ax.axhline(1.0, color="black", linewidth=0.8,
               linestyle="--", alpha=0.6, zorder=1, label="G3R = PG")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, x_hi_axis)
    ax.set_ylim(y_clip_lo, y_clip_hi)
    ax.set_xlabel("Number of Fill Prefixes (log scale)")
    ax.set_ylabel("Speedup  (PG / G3R, log scale)")
    ax.legend(fontsize=11, loc="upper right", ncol=1,
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title("(c) Speedup vs. Complexity",
                 fontsize=10, fontweight="bold")

    # ----------------- (d) Accuracy by strategy (F1 buckets) -----------
    # Most set-level F1 values are 1.00, making the CDF nearly vertical; use bins instead:
    # F1=1.00, [0.95,1.00), and <0.95, with a logarithmic share axis
    # to make both the near-perfect mass and sparse error tail visible.
    ax = axes[1, 1]
    _clean_ax(ax, "y")

    fast_f1 = pd.to_numeric(
        fast["f1"], errors="coerce").dropna().values
    fill_f1 = pd.to_numeric(
        fill["f1"], errors="coerce").dropna().values

    def _bucket_pct(arr):
        if len(arr) == 0:
            return [0.0, 0.0, 0.0]
        perfect = float((arr >= 0.9999).mean()) * 100.0
        mid = float(((arr >= 0.95) & (arr < 0.9999)).mean()) * 100.0
        low = float((arr < 0.95).mean()) * 100.0
        return [perfect, mid, low]

    fast_b = _bucket_pct(fast_f1)
    fill_b = _bucket_pct(fill_f1)

    bucket_labels = ["$F_1 = 1.00$", "$0.95 \\leq F_1 < 1.00$",
                     "$F_1 < 0.95$"]
    xb = np.arange(3)
    wb = 0.38
    y_floor = 0.02

    def _bar(offset, vals, color, label):
        heights = [max(v, y_floor) for v in vals]
        bars = ax.bar(xb + offset, heights, wb, bottom=y_floor,
                      color=color, edgecolor="white", linewidth=0.5,
                      label=label, zorder=3)
        for rect, v in zip(bars, vals):
            txt = f"{v:.1f}" if v >= 0.1 else (f"{v:.2f}" if v > 0 else "0")
            ax.text(rect.get_x() + rect.get_width() / 2,
                    max(v, y_floor) + y_floor,
                    txt, ha="center", va="bottom", fontsize=9,
                    color=color)
        return bars

    _bar(-wb / 2, fast_b, C_FAST_LINE, "Direct Prefix")
    _bar(+wb / 2, fill_b, C_FILL_LINE, "Polygon Fill")

    ax.set_yscale("log")
    ax.set_ylim(y_floor, 200.0)
    ax.set_xticks(xb)
    ax.set_xticklabels(bucket_labels, fontsize=10.5)
    ax.set_ylabel("Queries (%, log scale)")
    ax.legend(fontsize=11, loc="upper right",
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title("(d) Set-Level Accuracy by Strategy",
                 fontsize=10, fontweight="bold")

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    p = os.path.join(out_dir, f"fig_forward_overlay_v7_story.{fmt}")
    fig.savefig(p, dpi=dpi)
    print(f"  [Story 1x4] -> {p}")

    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    _export_axes_as_standalone(
        fig,
        axes_dict={"a": ax_a, "b": ax_b,
                   "c": ax_c,
                   "d": ax_d},
        out_dir=panels_dir,
        fmt=fmt, dpi=dpi,
        name_prefix="fig_forward_overlay",
    )
    print(f"  [Story 1x4] 4 standalone panels -> "
          f"{panels_dir}/fig_forward_overlay_panel_*.{fmt}")
    plt.close(fig)


# ======================================================================
# Figure 3: Polygon area / span vs query time
# ======================================================================

def fig_area_analysis(df, out_dir, fmt, dpi):
    df = _prepare_df(df)
    df["area"] = pd.to_numeric(df["pg_area_m2"], errors="coerce")
    df["lon_sp"] = pd.to_numeric(df["pg_lon_span_deg"], errors="coerce")
    df["lat_sp"] = pd.to_numeric(df["pg_lat_span_deg"], errors="coerce")

    has_area = df["area"].notna().sum() > 10
    if not has_area:
        print("  [Area] skipped (no pg_area_m2 data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) Area vs query time
    ax = axes[0]
    _clean_ax(ax, "both")
    tmp = df.dropna(subset=["area", "t_gr"]).copy()
    tmp = tmp[tmp["area"] > 0]
    area_cap = float(np.percentile(tmp["area"], 99))
    p_df = tmp[tmp["area"] <= area_cap]

    ax.scatter(p_df["area"], p_df["t_gr"],
               s=6, alpha=0.12, color=C_GR, edgecolors="none",
               rasterized=True)

    cx, med, q25, q75 = _binned_median_iqr(
        p_df["area"].values, p_df["t_gr"].values, n_bins=30)
    if len(cx):
        ax.fill_between(cx, q25, q75, color=C_GR, alpha=0.2, linewidth=0)
        ax.plot(cx, med, color=C_GR, linewidth=2.2, label="Median + IQR")

    ax.set_xlabel("Polygon Area (m²)")
    ax.set_ylabel("Query Time (ms)")
    ax.set_xlim(0, area_cap * 1.02)
    ax.set_ylim(0, min(200, np.percentile(p_df["t_gr"], 99) * 1.3))
    ax.legend(fontsize=7.5, loc="upper left",
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title("(a) Query Time vs. Polygon Area",
                 fontsize=10, fontweight="bold")

    # (b) Max span vs query time
    ax = axes[1]
    _clean_ax(ax, "both")
    tmp2 = df.dropna(subset=["lon_sp", "lat_sp", "t_gr"]).copy()
    tmp2["max_span"] = np.maximum(tmp2["lon_sp"], tmp2["lat_sp"])
    tmp2 = tmp2[tmp2["max_span"] > 0]
    span_cap = float(np.percentile(tmp2["max_span"], 99))
    p2 = tmp2[tmp2["max_span"] <= span_cap]

    ax.scatter(p2["max_span"] * 1000, p2["t_gr"],
               s=6, alpha=0.12, color=C_MATCH, edgecolors="none",
               rasterized=True)

    cx2, med2, q25_2, q75_2 = _binned_median_iqr(
        p2["max_span"].values * 1000, p2["t_gr"].values, n_bins=30)
    if len(cx2):
        ax.fill_between(cx2, q25_2, q75_2, color=C_MATCH, alpha=0.2,
                         linewidth=0)
        ax.plot(cx2, med2, color=C_MATCH, linewidth=2.2,
                label="Median + IQR")

    ax.set_xlabel("Max Span (millideg)")
    ax.set_ylabel("Query Time (ms)")
    ax.set_xlim(0, span_cap * 1000 * 1.02)
    ax.set_ylim(0, min(200, np.percentile(p2["t_gr"], 99) * 1.3))
    ax.legend(fontsize=7.5, loc="upper left",
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title("(b) Query Time vs. Polygon Extent",
                 fontsize=10, fontweight="bold")

    fig.tight_layout(w_pad=2.5)
    p = os.path.join(out_dir, f"fig_forward_overlay_v7_area.{fmt}")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    print(f"  [Area] -> {p}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Forward overlay V7 paper figure generator")
    parser.add_argument("--csv", default="result_v7_2.csv")
    parser.add_argument("--out-dir", default="overlay_v7_figures")
    parser.add_argument("--format", default="png",
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--all", action="store_true",
                        help="Also produce supplementary figures "
                             "(time_breakdown, fig_paper, area_analysis). "
                             "Default: only fig_paper_story (canonical "
                             "single figure for paper).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isfile(args.csv):
        print(f"[ERROR] File not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Data:   {args.csv}  ({len(df):,} rows)")
    print(f"Output: {args.out_dir}/")
    print()

    fig_paper_story(df, args.out_dir, args.format, args.dpi)
    if args.all:
        fig_time_breakdown(df, args.out_dir, args.format, args.dpi)
        fig_paper(df, args.out_dir, args.format, args.dpi)
        fig_area_analysis(df, args.out_dir, args.format, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()

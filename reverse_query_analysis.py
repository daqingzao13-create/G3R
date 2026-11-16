#!/usr/bin/env python3
# coding: utf-8
"""
GeoSOT Reverse Overlay Query Result Analyzer

Reads the CSV produced by geosot_reverse_query.py (--batch),
generates publication-quality figures (English labels).
Key focus: impact of hit vs. no-hit on query performance.

Usage:
    python reverse_query_analysis.py result.csv
    python reverse_query_analysis.py result.csv --dpi 600 --format pdf
    python reverse_query_analysis.py result.csv --out-dir my_figures
    python reverse_query_analysis.py result.csv --paper-only
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import Bbox


def _export_axes_as_standalone(fig, axes_dict, out_dir, fmt, dpi,
                               name_prefix, strip_panel_label=True,
                               pad_inches=0.20):
    """
    Crop axes from a composite figure into standalone files for LaTeX. axes_dict maps panel
    labels to one axis or a sequence of axes; crop to the union bounding box for multipart
    panels.
    """
    from paper_figure_style import export_consistent_panels
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
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ============================================================================
# Style (SCI-paper friendly)
# ============================================================================

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
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.6,
})

C_HIT = "#1565C0"
C_MISS = "#E53935"
C_PG = "#9E9E9E"
C_QUERY = "#43A047"
C_PIP = "#8E24AA"
C_DECODE = "#FB8C00"
C_METHODS = {
    "prefix_direct": "#1565C0",
    "neighbor_voting": "#43A047",
    "cascade_fallback": "#FB8C00",
    "no_match": "#E53935",
}

# Match the forward-overlay palette in overlay_v7_analysis.py / fig_paper_story.
# Used by the four-panel publication figure.
C_GR = "#1565C0"          # G3R primary color: deep blue, matching forward overlay.
C_PG_BAR = "#9E9E9E"      # PostGIS primary color: gray.
C_GROUP_PREP = "#BBDEFB"  # Light blue for data preparation, matching forward overlay.
C_GROUP_GRAPH = "#1565C0" # Deep blue for graph reasoning: prefix matching.
C_GROUP_GEOM = "#FB8C00"  # Orange for geometric post-processing: decoding and PIP.
C_FAST_LINE = "#1565C0"   # Blue for Hit queries.
C_FILL_LINE = "#E53935"   # Red for No-Hit queries.
C_HIGHLIGHT = "#43A047"   # Green accent.


def _clean_ax(ax, grid_axis="y"):
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, alpha=0.12,
            linewidth=0.4, color="#999999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBBB")
        ax.spines[s].set_linewidth(0.5)


def _safe_col(df, col):
    if col not in df.columns:
        return np.array([])
    return pd.to_numeric(df[col], errors="coerce").dropna().values


def _split_hit(df):
    hit = df[pd.to_numeric(df["pip_hits"], errors="coerce") > 0]
    miss = df[pd.to_numeric(df["pip_hits"], errors="coerce") == 0]
    return hit, miss


def _f1_filled(sub) -> np.ndarray:
    """
    Read F1 and assign 1.0 to agreeing empty result sets. The raw reverse-overlay CSV leaves F1
    blank when both precision/recall denominators are zero; if match==1, both returned sets are
    empty and equal. Filling this case in analysis retains No-Hit queries in the set-agreement
    CDF without modifying the raw records.
    """
    if "f1" not in sub.columns:
        return np.array([])
    raw = pd.to_numeric(sub["f1"], errors="coerce")
    match = (pd.to_numeric(sub.get("match", pd.Series(dtype=float)),
                           errors="coerce") == 1)
    pg_zero = (pd.to_numeric(sub.get("pg_count", pd.Series(dtype=float)),
                             errors="coerce") == 0)
    consensus_empty = match & pg_zero
    filled = raw.where(~(raw.isna() & consensus_empty), other=1.0)
    return filled.dropna().values


# ============================================================================
# Fig 1: Total Time — Hit vs No-Hit (violin + box)
# ============================================================================

def fig_time_hit_vs_miss(df, out, fmt, dpi):
    hit, miss = _split_hit(df)
    gr_hit = _safe_col(hit, "t_gr_total_ms")
    gr_miss = _safe_col(miss, "t_gr_total_ms")

    if len(gr_hit) == 0 and len(gr_miss) == 0:
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    _clean_ax(ax)

    data, labels, colors = [], [], []
    if len(gr_hit):
        data.append(gr_hit); labels.append(f"Hit\n(n={len(gr_hit):,})")
        colors.append(C_HIT)
    if len(gr_miss):
        data.append(gr_miss); labels.append(f"No Hit\n(n={len(gr_miss):,})")
        colors.append(C_MISS)

    pg = _safe_col(df, "t_pg_ms")
    if len(pg):
        data.append(pg); labels.append("PostGIS")
        colors.append(C_PG)

    pos = range(len(data))
    vp = ax.violinplot(data, positions=pos,
                       showmedians=False, showextrema=False)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(colors[i]); b.set_alpha(0.25)

    bp = ax.boxplot(data, positions=pos, widths=0.18,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2))
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(colors[i]); p.set_alpha(0.7)

    for i, d in enumerate(data):
        med = float(np.median(d))
        ax.text(i, med, f"  {med:.1f}", va="center", ha="left",
                fontsize=7, fontweight="bold", color=colors[i])

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-Query Time (ms)")
    ax.set_title("Reverse Query: Hit vs. No-Hit Latency",
                 fontweight="bold")

    fig.tight_layout()
    p = os.path.join(out, f"fig1_time_hit_vs_miss.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.1] Hit vs No-Hit time   -> {p}")


# ============================================================================
# Fig 2: Step Breakdown by Hit Status (grouped stacked bar)
# ============================================================================

def fig_step_by_hit(df, out, fmt, dpi):
    hit, miss = _split_hit(df)

    steps = [
        ("t_decode_poi_ms", "Coord Decode", C_DECODE),
        ("t_query_ms", "Prefix Match", C_QUERY),
        ("t_pip_ms", "PIP Test", C_PIP),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    _clean_ax(ax)

    groups = []
    group_labels = []
    if len(hit):
        groups.append(hit); group_labels.append("Hit")
    if len(miss):
        groups.append(miss); group_labels.append("No Hit")

    has_pg = len(_safe_col(df, "t_pg_ms")) > 0
    if has_pg:
        group_labels.append("PostGIS")

    x = np.arange(len(group_labels))
    handles = []

    for si, (col, label, color) in enumerate(steps):
        bottoms = []
        heights = []
        for g in groups:
            v = _safe_col(g, col)
            heights.append(float(np.median(v)) if len(v) else 0)
            prev = sum(
                float(np.median(_safe_col(g, c)))
                for j, (c, _, _) in enumerate(steps)
                if j < si and len(_safe_col(g, c))
            )
            bottoms.append(prev)

        if has_pg:
            heights.append(0)
            bottoms.append(0)

        ax.bar(x, heights, bottom=bottoms, width=0.5,
               color=color, alpha=0.85)

        for i, (h, b) in enumerate(zip(heights, bottoms)):
            if h > 0.01:
                ax.text(i, b + h / 2, f"{h:.2f}",
                        ha="center", va="center",
                        fontsize=6.5, color="white",
                        fontweight="bold")

        handles.append(plt.Rectangle((0, 0), 1, 1,
                                     fc=color, alpha=0.85,
                                     label=label))

    if has_pg:
        pg_med = float(np.median(_safe_col(df, "t_pg_ms")))
        idx_pg = len(group_labels) - 1
        ax.bar(idx_pg, pg_med, width=0.5, color=C_PG, alpha=0.85)
        ax.text(idx_pg, pg_med / 2, f"{pg_med:.2f}",
                ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold")
        handles.append(plt.Rectangle((0, 0), 1, 1,
                                     fc=C_PG, alpha=0.85,
                                     label="ST_Contains"))

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Median Time (ms)")
    ax.set_title("Step Breakdown: Hit vs. No-Hit",
                 fontweight="bold")
    ax.legend(handles=handles, loc="upper right", fontsize=7)

    fig.tight_layout()
    p = os.path.join(out, f"fig2_step_by_hit.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.2] Step breakdown       -> {p}")


# ============================================================================
# Fig 3: Method Distribution (pie)
# ============================================================================

def fig_method_pie(df, out, fmt, dpi):
    if "method" not in df.columns:
        return
    counts = df["method"].value_counts()
    if len(counts) == 0:
        return

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    labels, sizes, colors = [], [], []
    label_map = {
        "prefix_direct": "Prefix Direct",
        "neighbor_voting": "Neighbor Voting",
        "cascade_fallback": "Cascade Fallback",
        "no_match": "No Match",
        "cascade": "Cascade",
    }
    for method, cnt in counts.items():
        labels.append(label_map.get(method, method))
        sizes.append(cnt)
        colors.append(C_METHODS.get(method, "#888888"))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p*sum(sizes)/100):,})",
        startangle=90,
        textprops=dict(fontsize=8))
    for t in autotexts:
        t.set_fontsize(7); t.set_fontweight("bold")

    ax.set_title("Resolution Method Distribution",
                 fontweight="bold")

    fig.tight_layout()
    p = os.path.join(out, f"fig3_method_pie.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.3] Method distribution  -> {p}")


# ============================================================================
# Fig 4: Time by Method (box)
# ============================================================================

def fig_time_by_method(df, out, fmt, dpi):
    if "method" not in df.columns:
        return

    methods_order = ["prefix_direct", "neighbor_voting",
                     "cascade_fallback", "no_match", "cascade"]
    label_map = {
        "prefix_direct": "Prefix\nDirect",
        "neighbor_voting": "Neighbor\nVoting",
        "cascade_fallback": "Cascade\nFallback",
        "no_match": "No\nMatch",
        "cascade": "Cascade",
    }

    data, labels, colors = [], [], []
    for m in methods_order:
        sub = df[df["method"] == m]
        vals = _safe_col(sub, "t_gr_total_ms")
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(f"{label_map.get(m, m)}\n(n={len(vals):,})")
        colors.append(C_METHODS.get(m, "#888888"))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    _clean_ax(ax)

    pos = range(len(data))
    vp = ax.violinplot(data, positions=pos,
                       showmedians=False, showextrema=False)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(colors[i]); b.set_alpha(0.25)

    bp = ax.boxplot(data, positions=pos, widths=0.2,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2))
    for i, p_box in enumerate(bp["boxes"]):
        p_box.set_facecolor(colors[i]); p_box.set_alpha(0.7)

    for i, d in enumerate(data):
        med = float(np.median(d))
        ax.text(i, med, f"  {med:.1f}", va="center", ha="left",
                fontsize=7, fontweight="bold", color=colors[i])

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-Query Time (ms)")
    ax.set_title("Query Time by Resolution Method",
                 fontweight="bold")

    fig.tight_layout()
    p = os.path.join(out, f"fig4_time_by_method.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.4] Time by method       -> {p}")


# ============================================================================
# Fig 5: CDF — Hit vs No-Hit
# ============================================================================

def fig_cdf_hit_miss(df, out, fmt, dpi):
    hit, miss = _split_hit(df)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _clean_ax(ax, grid_axis="both")

    for sub, label, color, ls in [
        (hit, "Hit", C_HIT, "-"),
        (miss, "No Hit", C_MISS, "--"),
    ]:
        vals = _safe_col(sub, "t_gr_total_ms")
        if len(vals) == 0:
            continue
        s = np.sort(vals)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, color=color, linewidth=1.5,
                linestyle=ls, label=f"{label} (n={len(s):,})")

    pg = _safe_col(df, "t_pg_ms")
    if len(pg):
        s = np.sort(pg)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, color=C_PG, linewidth=1.2,
                linestyle=":", label=f"PostGIS (n={len(s):,})")

    ax.set_xlabel("Query Time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution: Hit vs. No-Hit",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    p = os.path.join(out, f"fig5_cdf_hit_miss.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.5] CDF hit vs miss      -> {p}")


# ============================================================================
# Fig 6: Candidate Count — Hit vs No-Hit
# ============================================================================

def fig_candidates_hit_miss(df, out, fmt, dpi):
    hit, miss = _split_hit(df)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _clean_ax(ax)

    for sub, label, color in [
        (hit, "Hit", C_HIT), (miss, "No Hit", C_MISS),
    ]:
        vals = _safe_col(sub, "n_candidates")
        if len(vals) == 0:
            continue
        bins = np.linspace(0, min(vals.max(), 1000), 40)
        ax.hist(vals, bins=bins, alpha=0.5, color=color,
                label=f"{label} (med={int(np.median(vals))})",
                edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Number of Candidates")
    ax.set_ylabel("Count")
    ax.set_title("Candidate Count: Hit vs. No-Hit",
                 fontweight="bold")
    ax.legend(fontsize=7)

    fig.tight_layout()
    p = os.path.join(out, f"fig6_candidates_hit_miss.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.6] Candidates dist.     -> {p}")


# ============================================================================
# Fig 7: Prefix Level Distribution
# ============================================================================

def fig_prefix_level(df, out, fmt, dpi):
    vals = _safe_col(df, "deepest_prefix_level")
    if len(vals) == 0:
        return

    hit, miss = _split_hit(df)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _clean_ax(ax)

    all_levels = np.arange(int(vals.min()), int(vals.max()) + 1)
    w = 0.35

    for i, (sub, label, color) in enumerate([
        (hit, "Hit", C_HIT), (miss, "No Hit", C_MISS),
    ]):
        lv = _safe_col(sub, "deepest_prefix_level")
        if len(lv) == 0:
            continue
        levels, counts = np.unique(lv.astype(int),
                                   return_counts=True)
        full_counts = np.zeros(len(all_levels), dtype=int)
        for l, c in zip(levels, counts):
            idx = np.where(all_levels == l)[0]
            if len(idx):
                full_counts[idx[0]] = c
        ax.bar(all_levels + (i - 0.5) * w, full_counts,
               width=w, color=color, alpha=0.8, label=label)

    ax.set_xlabel("Deepest Prefix Level")
    ax.set_ylabel("Count")
    ax.set_title("Prefix Level Distribution: Hit vs. No-Hit",
                 fontweight="bold")
    ax.legend(fontsize=7)

    fig.tight_layout()
    p = os.path.join(out, f"fig7_prefix_level.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.7] Prefix level dist.   -> {p}")


# ============================================================================
# Fig 8: Recall / Precision / F1 (bar)
# ============================================================================

def fig_accuracy(df, out, fmt, dpi):
    metrics = []
    for col, label in [
        ("recall", "Recall"), ("precision", "Precision"),
        ("f1", "F1"),
    ]:
        vals = _safe_col(df, col)
        if len(vals):
            metrics.append((label, float(np.mean(vals)),
                            float(np.median(vals)),
                            float(np.min(vals))))
    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    _clean_ax(ax)

    x = np.arange(len(metrics))
    means = [m[1] for m in metrics]
    meds = [m[2] for m in metrics]
    mins = [m[3] for m in metrics]

    ax.bar(x - 0.15, means, 0.28, color=C_HIT, alpha=0.8,
           label="Mean")
    ax.bar(x + 0.15, meds, 0.28, color=C_QUERY, alpha=0.8,
           label="Median")

    for i, (lbl, mn, md, mi) in enumerate(metrics):
        ax.text(i - 0.15, mn + 0.005, f"{mn:.2%}",
                ha="center", va="bottom", fontsize=6.5)
        ax.text(i + 0.15, md + 0.005, f"{md:.2%}",
                ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(min(0.95, min(mins) - 0.02), 1.01)
    ax.yaxis.set_major_formatter(
        ticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_title("Accuracy vs. PostGIS Baseline",
                 fontweight="bold")
    ax.legend(fontsize=7)

    match_col = _safe_col(df, "match")
    if len(match_col):
        match_rate = float(np.mean(match_col))
        ax.text(0.97, 0.03,
                f"Exact match: {match_rate:.2%}",
                transform=ax.transAxes, fontsize=7,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#E8F5E9", alpha=0.9))

    fig.tight_layout()
    p = os.path.join(out, f"fig8_accuracy.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.8] Accuracy metrics     -> {p}")


# ============================================================================
# Fig 9: Speedup Scatter (GR time vs PostGIS time)
# ============================================================================

def fig_speedup_scatter(df, out, fmt, dpi):
    gr = _safe_col(df, "t_gr_total_ms")
    pg = _safe_col(df, "t_pg_ms")
    if len(gr) == 0 or len(pg) == 0:
        return

    valid = df.dropna(
        subset=["t_gr_total_ms", "t_pg_ms", "pip_hits"])
    gr_v = pd.to_numeric(valid["t_gr_total_ms"]).values
    pg_v = pd.to_numeric(valid["t_pg_ms"]).values
    hits = pd.to_numeric(valid["pip_hits"]).values

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    _clean_ax(ax, grid_axis="both")

    is_hit = hits > 0
    if is_hit.sum():
        ax.scatter(pg_v[is_hit], gr_v[is_hit],
                   s=8, alpha=0.3, color=C_HIT,
                   label=f"Hit (n={is_hit.sum():,})",
                   edgecolors="none")
    if (~is_hit).sum():
        ax.scatter(pg_v[~is_hit], gr_v[~is_hit],
                   s=8, alpha=0.3, color=C_MISS,
                   label=f"No Hit (n={(~is_hit).sum():,})",
                   edgecolors="none")

    lim = max(gr_v.max(), pg_v.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8,
            alpha=0.4, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("PostGIS Time (ms)")
    ax.set_ylabel("G3R Time (ms)")
    ax.set_title("Per-Query Time: G3R vs. PostGIS",
                 fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    p = os.path.join(out, f"fig9_speedup_scatter.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.9] Speedup scatter      -> {p}")


# ============================================================================
# Fig 10: Time Proportion Pie (hit only)
# ============================================================================

def fig_time_pie_hit(df, out, fmt, dpi):
    hit, miss = _split_hit(df)
    if len(hit) == 0 and len(miss) == 0:
        return

    steps = [
        ("t_decode_poi_ms", "Coord Decode", C_DECODE),
        ("t_query_ms", "Prefix Match", C_QUERY),
        ("t_pip_ms", "PIP Test", C_PIP),
    ]
    labels = [s[1] for s in steps]
    colors = [s[2] for s in steps]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax_idx, (sub, title) in enumerate([
        (hit, "Hit Queries"), (miss, "No-Hit Queries"),
    ]):
        ax = axes[ax_idx]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "N/A", ha="center",
                    va="center", fontsize=12)
            ax.set_title(title, fontweight="bold")
            continue

        v = []
        for col, _, _ in steps:
            sv = _safe_col(sub, col)
            v.append(float(np.mean(sv)) if len(sv) else 0)

        if sum(v) < 0.001:
            ax.text(0.5, 0.5, "N/A", ha="center",
                    va="center", fontsize=12)
            ax.set_title(title, fontweight="bold")
            continue

        wedges, texts, autotexts = ax.pie(
            v, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops=dict(fontsize=7))
        for t in autotexts:
            t.set_fontsize(7); t.set_fontweight("bold")
        total = sum(v)
        ax.set_title(f"{title}\n(Mean total: {total:.2f} ms)",
                     fontweight="bold", fontsize=9)

    fig.tight_layout()
    p = os.path.join(out, f"fig10_time_pie.{fmt}")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    print(f"  [Fig.10] Time proportion pie -> {p}")


# ============================================================================
# Fig 11: Overview 2x3 (legacy)
# ============================================================================

def fig_overview(df, out, fmt, dpi):
    fig_paper(df, out, fmt, dpi)


# ============================================================================
# Paper Figure: 2x3 composite aligned with Section 4.3.3 narrative
# ============================================================================

def fig_paper(df, out, fmt, dpi):
    """
    Reverse-overlay publication figure, matching forward-overlay styling in a 2x2 layout: (a)
    stacked latency for G3R Hit/No-Hit and PG; (b) grouped speedup CDFs; (c) log-log G3R/PG time
    scatter with y=x; (d) set-level F1 CDF and an inset for the F1<1 tail.
    """
    hit, miss = _split_hit(df)
    n_total = len(df)
    n_hit = len(hit)
    n_miss = len(miss)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6))

    gr_total = _safe_col(df, "t_gr_total_ms")
    pg_total = _safe_col(df, "t_pg_ms")
    speedup = _safe_col(df, "speedup")

    # ==================================================================
    # (a) Latency Breakdown — GR Hit / GR No-Hit / PostGIS (stacked bar)
    # ==================================================================
    ax = axes[0, 0]
    _clean_ax(ax)

    def _means_breakdown(sub):
        return (
            float(np.mean(_safe_col(sub, "t_decode_poi_ms"))) if len(sub) else 0.0,
            float(np.mean(_safe_col(sub, "t_query_ms"))) if len(sub) else 0.0,
            float(np.mean(_safe_col(sub, "t_pip_ms"))) if len(sub) else 0.0,
        )

    hit_d, hit_q, hit_p = _means_breakdown(hit)
    miss_d, miss_q, miss_p = _means_breakdown(miss)
    pg_mean = float(np.mean(pg_total)) if len(pg_total) else 0.0

    bar_x = np.arange(3)
    bar_w = 0.6
    bar_labels = [
        f"G3R Hit\n(n={n_hit:,})",
        f"G3R No-Hit\n(n={n_miss:,})",
        f"PostGIS\n(n={n_total:,})",
    ]

    # GR Hit (stacked)
    ax.bar(bar_x[0], hit_d, bar_w,
           color=C_GROUP_PREP, edgecolor="white", linewidth=0.5,
           label="Data preparation", zorder=2)
    ax.bar(bar_x[0], hit_q, bar_w, bottom=hit_d,
           color=C_GROUP_GRAPH, edgecolor="white", linewidth=0.5,
           label="Graph reasoning", zorder=2)
    ax.bar(bar_x[0], hit_p, bar_w, bottom=hit_d + hit_q,
           color=C_GROUP_GEOM, edgecolor="white", linewidth=0.5,
           label="Geometric verification", zorder=2)
    # GR No-Hit
    ax.bar(bar_x[1], miss_d, bar_w,
           color=C_GROUP_PREP, edgecolor="white", linewidth=0.5, zorder=2)
    ax.bar(bar_x[1], miss_q, bar_w, bottom=miss_d,
           color=C_GROUP_GRAPH, edgecolor="white", linewidth=0.5, zorder=2)
    ax.bar(bar_x[1], miss_p, bar_w, bottom=miss_d + miss_q,
           color=C_GROUP_GEOM, edgecolor="white", linewidth=0.5, zorder=2)
    # PG
    ax.bar(bar_x[2], pg_mean, bar_w,
           color=C_PG_BAR, edgecolor="white", linewidth=0.5,
           label="PostGIS", zorder=2)

    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylabel("Mean End-to-End Time (ms)")
    ax.set_title("(a) End-to-End Latency Breakdown",
                 fontsize=9.5, fontweight="bold")

    max_total = max(hit_d + hit_q + hit_p,
                    miss_d + miss_q + miss_p, pg_mean)
    if max_total > 0:
        ax.set_ylim(0, max_total * 1.32)

    ax.legend(fontsize=11, loc="upper center",
              ncol=2,
              framealpha=0.9, edgecolor="#CCCCCC")

    # ==================================================================
    # (b) Speedup CDF — Hit vs No-Hit grouped
    # ==================================================================
    ax = axes[0, 1]
    _clean_ax(ax, grid_axis="both")

    def _cdf_speedup(sub, lab, col, ls, lw):
        v = _safe_col(sub, "speedup")
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) == 0:
            return None
        s = np.sort(v)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, color=col, linewidth=lw, linestyle=ls, label=lab)
        return v

    v_hit = _cdf_speedup(hit, "Hit",
                         C_FAST_LINE, "-", 1.8)
    v_miss = _cdf_speedup(miss, "No-Hit",
                          C_FILL_LINE, "--", 1.6)

    ax.axvline(1.0, color="#666666", linewidth=0.8,
               linestyle=":", zorder=1)
    ax.axvspan(1.0, 1e6, color=C_FAST_LINE, alpha=0.05, zorder=0)

    ax.set_xscale("log")
    all_speedup = speedup[np.isfinite(speedup) & (speedup > 0)]
    if len(all_speedup):
        ax.set_xlim(max(all_speedup.min(), 1e-2), all_speedup.max() * 1.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Speedup  (PostGIS / G3R)  [log]")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("(b) Speedup CDF (Hit vs No-Hit)",
                 fontsize=10, fontweight="bold")

    ax.legend(fontsize=11, loc="lower right", ncol=1,
              framealpha=0.9, edgecolor="#CCCCCC")

    # ==================================================================
    # (c) GR vs PostGIS Time — log-log scatter + y=x diagonal
    # ==================================================================
    ax = axes[1, 0]
    _clean_ax(ax, grid_axis="both")

    def _scatter_subset(sub, col, marker, lab):
        x = _safe_col(sub, "t_pg_ms")
        y = _safe_col(sub, "t_gr_total_ms")
        n = min(len(x), len(y))
        if n == 0:
            return
        x = x[:n]
        y = y[:n]
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        # Randomly subsample 5,000 points when there are more than 5,000 to limit overplotting.
        if len(x) > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(x), size=5000, replace=False)
            x = x[idx]; y = y[idx]
        ax.scatter(x, y, s=4, c=col, marker=marker,
                   alpha=0.35, edgecolors="none",
                   label=lab, zorder=2)

    _scatter_subset(hit,  C_FAST_LINE, "o", "Hit")
    _scatter_subset(miss, C_FILL_LINE, "^", "No-Hit")

    # y=x reference diagonal.
    if len(gr_total) and len(pg_total):
        lo = max(min(gr_total[gr_total > 0].min(),
                     pg_total[pg_total > 0].min()), 1e-3)
        hi = max(gr_total.max(), pg_total.max()) * 1.2
        diag = np.array([lo, hi])
        ax.plot(diag, diag, color="#333333", linewidth=1.0,
                linestyle="--", zorder=3, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PostGIS Time (ms)  [log]")
    ax.set_ylabel("G3R Time (ms)  [log]")
    ax.set_title("(c) G3R vs PostGIS Per-Query Time",
                 fontsize=10, fontweight="bold")

    ax.legend(fontsize=11, loc="lower right", ncol=1,
              framealpha=0.9, edgecolor="#CCCCCC")

    # ==================================================================
    # (d) Set-level F1 CDF with an inset for F1<1.
    # ==================================================================
    ax = axes[1, 1]
    _clean_ax(ax, grid_axis="both")

    f1_all = _f1_filled(df)

    def _cdf_f1(sub, lab, col, ls, lw):
        v = _f1_filled(sub)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return None
        s = np.sort(v)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, color=col, linewidth=lw,
                linestyle=ls, label=lab, zorder=3)
        return v

    f1_hit = _cdf_f1(hit, "Hit",
                     C_FAST_LINE, "-", 1.8)
    f1_miss = _cdf_f1(miss, "No-Hit",
                      C_FILL_LINE, "--", 1.6)

    ax.axvspan(0.999, 1.001, color=C_HIGHLIGHT, alpha=0.10, zorder=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Set-Level F1 Score")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("(d) Set-Level Accuracy",
                 fontsize=10, fontweight="bold")

    ax.legend(fontsize=10.5, loc="lower center",
              framealpha=0.9, edgecolor="#CCCCCC", ncol=2)

    # Inset zoom on the F1<1 error tail.
    if len(f1_all) and float(np.min(f1_all)) < 0.999:
        try:
            ax_zoom = inset_axes(
                ax, width="42%", height="42%",
                loc="center right", borderpad=1.6,
            )
            for sub, col, ls, lw in [
                (hit,  C_FAST_LINE, "-",  1.6),
                (miss, C_FILL_LINE, "--", 1.4),
            ]:
                v = _f1_filled(sub)
                v = v[np.isfinite(v) & (v < 0.9999)]
                if len(v) == 0:
                    continue
                s = np.sort(v)
                y = np.arange(1, len(s) + 1) / len(s)
                ax_zoom.plot(s, y, color=col, linewidth=lw, linestyle=ls)
            ax_zoom.set_xlim(-0.02, 1.0)
            ax_zoom.set_ylim(0, 1.02)
            ax_zoom.tick_params(axis="both", labelsize=6)
            for sp in ax_zoom.spines.values():
                sp.set_color("#999999")
                sp.set_linewidth(0.5)
            ax_zoom.grid(True, alpha=0.15, linewidth=0.4)
        except Exception:
            pass

    # Use subplots_adjust instead of tight_layout to avoid inset_axes warnings.
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.94,
                        wspace=0.22, hspace=0.30)
    p = os.path.join(out, f"fig_paper_reverse_overlay.{fmt}")
    fig.savefig(p, dpi=dpi)
    print(f"  [Paper] Reverse overlay composite -> {p}")

    panels_dir = os.path.join(out, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    _export_axes_as_standalone(
        fig,
        axes_dict={"a": axes[0, 0], "b": axes[0, 1],
                   "c": axes[1, 0], "d": axes[1, 1]},
        out_dir=panels_dir,
        fmt=fmt, dpi=dpi,
        name_prefix="fig_reverse_overlay",
    )
    print(f"  [Paper] 4 standalone panels -> "
          f"{panels_dir}/fig_reverse_overlay_panel_*.{fmt}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reverse Overlay Query result analyzer")
    parser.add_argument("csv",
                        help="CSV file from geosot_reverse_query.py")
    parser.add_argument("--out-dir", default="",
                        help="Output directory")
    parser.add_argument("--format", default="png",
                        choices=["png", "pdf", "svg"],
                        help="Image format (default: png)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--paper-only", action="store_true",
                        help="Generate only the paper composite figure")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    out_dir = args.out_dir
    if not out_dir:
        out_dir = "reverse_query_figures"
    os.makedirs(out_dir, exist_ok=True)

    fmt = args.format
    dpi = args.dpi

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Output:  {out_dir}/")
    print()

    fig_paper(df, out_dir, fmt, dpi)

    if not args.paper_only:
        fig_time_hit_vs_miss(df, out_dir, fmt, dpi)
        fig_step_by_hit(df, out_dir, fmt, dpi)
        fig_method_pie(df, out_dir, fmt, dpi)
        fig_time_by_method(df, out_dir, fmt, dpi)
        fig_cdf_hit_miss(df, out_dir, fmt, dpi)
        fig_candidates_hit_miss(df, out_dir, fmt, dpi)
        fig_prefix_level(df, out_dir, fmt, dpi)
        fig_accuracy(df, out_dir, fmt, dpi)
        fig_speedup_scatter(df, out_dir, fmt, dpi)
        fig_time_pie_hit(df, out_dir, fmt, dpi)
        print(f"\nDone. Paper figure + 10 individual figures -> {out_dir}/")
    else:
        print(f"\nDone. Paper figure -> {out_dir}/")


if __name__ == "__main__":
    main()

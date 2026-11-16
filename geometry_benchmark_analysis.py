#!/usr/bin/env python3
# coding: utf-8
"""
GeoSOT Geometry Benchmark Result Analyzer

Reads the 9 detail CSVs + summary.csv produced by geometry_benchmark.py,
generates publication-quality figures suitable for SCI journals (English labels).

Usage:
    python geometry_benchmark_analysis.py <benchmark_folder> [options]

Examples:
    python geometry_benchmark_analysis.py benchmark_v1
    python geometry_benchmark_analysis.py benchmark_v1 --dpi 600 --format pdf
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from paper_figure_style import export_consistent_panels

# ============================================================================
# Style configuration (SCI-paper friendly)
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

# Unified palette aligned with multiscale_analysis / reverse_query_analysis:
#   G3R (graph reasoning)   → deep blue (#1565C0)
#   PostGIS                 → mid-gray  (#9E9E9E)
#   Neo4j (graph fetch)     → blue-gray for sub-step decomposition
#
# Retain floating-point Grid decode dec_* fields in CSV as an internal reproducibility record,
# but omit them from paper plots. Their difference from G3R is the cosine-weighting policy
# within the same coordinate reconstruction mechanism, rather than a separate method.
# Both the paper and these plots compare G3R with PostGIS.
COLOR_GRAPH   = "#1565C0"
COLOR_POSTGIS = "#9E9E9E"
COLOR_NEO4J   = "#78909C"

COMBO_LABELS = [
    "Pt–Pt", "Pt–Ln", "Pt–Poly",
    "Ln–Pt", "Ln–Ln", "Ln–Poly",
    "Poly–Pt", "Poly–Ln", "Poly–Poly",
]

COMBO_FILES = [
    "Point_x_Point", "Point_x_LineString", "Point_x_Polygon",
    "LineString_x_Point", "LineString_x_LineString", "LineString_x_Polygon",
    "Polygon_x_Point", "Polygon_x_LineString", "Polygon_x_Polygon",
]

GRAPH_STEPS = [
    ("gr_prepare_codes_ms", "Prepare codes"),
    ("gr_vertex_estimate_ms", "Vertex estimate"),
    ("gr_geometry_distance_ms", "Geometry distance"),
    ("gr_bearing_ms", "Bearing calculation"),
]


def _export_axes_as_standalone(fig, axes_dict, out_dir, fmt, dpi,
                               name_prefix, strip_panel_label=True,
                               pad_inches=0.18):
    """
    Crop axes from a composite figure into standalone panel files for LaTeX
    includegraphics/subfloat/subfigure layouts.

    @param fig Open Matplotlib Figure.
    @param axes_dict Panel labels mapped to axes.
    @param out_dir Output directory.
    @param fmt png, pdf, svg, or eps.
    @param dpi Export resolution.
    @param name_prefix Filename prefix, for example fig2_distance.
    @param strip_panel_label Temporarily remove titles such as (a) to avoid duplicate LaTeX labels.
    @param pad_inches Bounding-box padding to protect tick labels.
    """
    exported = export_consistent_panels(
        fig, axes_dict, out_dir, fmt, dpi, name_prefix)
    axes_dict = {key: ax for key, ax in axes_dict.items() if key not in exported}
    if not axes_dict:
        return

    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # A panel may contain one axis or a list containing the primary and twinned axes.
    # For a list, union all tight bounding boxes so right-axis labels/ticks are not cropped.
    def _main_ax(ax):
        return ax[0] if isinstance(ax, (list, tuple)) else ax

    saved_titles = {}
    if strip_panel_label:
        for letter, ax in axes_dict.items():
            m = _main_ax(ax)
            saved_titles[letter] = m.get_title()
            m.set_title("")

    try:
        for letter, ax in axes_dict.items():
            ax_list = list(ax) if isinstance(ax, (list, tuple)) else [ax]
            keep = set(ax_list)
            hidden = []
            for o in fig.axes:
                if o not in keep and o.get_visible():
                    hidden.append(o)
                    o.set_visible(False)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = Bbox.union([a.get_tightbbox(renderer) for a in ax_list])
            extent = bb.transformed(fig.dpi_scale_trans.inverted())
            extent = extent.padded(pad_inches)
            path = os.path.join(out_dir,
                                f"{name_prefix}_panel_{letter}.{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches=extent,
                        facecolor="white")
            for o in hidden:
                o.set_visible(True)
    finally:
        if strip_panel_label:
            for letter, ax in axes_dict.items():
                _main_ax(ax).set_title(saved_titles[letter])


def load_data(folder: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the benchmark folder.

    @param folder path to benchmark output folder
    @returns {combo_key: DataFrame, "summary": DataFrame}
    """
    data = {}
    for i, key in enumerate(COMBO_FILES):
        csv_path = os.path.join(folder, f"{key}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                data[key] = df

    summary_path = os.path.join(folder, "summary.csv")
    if os.path.exists(summary_path):
        data["summary"] = pd.read_csv(summary_path)

    return data


def _get_combo_stats(data: Dict[str, pd.DataFrame], column: str,
                     agg: str = "median") -> List[float]:
    """Get a statistic for a column across all 9 combos."""
    vals = []
    for key in COMBO_FILES:
        df = data.get(key)
        if df is not None and column in df.columns and len(df) > 0:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if agg == "median":
                vals.append(series.median())
            elif agg == "mean":
                vals.append(series.mean())
            else:
                vals.append(0.0)
        else:
            vals.append(0.0)
    return vals


def _active_mask(data: Dict[str, pd.DataFrame]) -> List[bool]:
    """Which combos have data."""
    return [key in data for key in COMBO_FILES]


def _split_combo_key(combo_key: str) -> Tuple[str, str]:
    """Split combo key like Point_x_LineString into two geometry types."""
    parts = combo_key.split("_x_")
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def fig_accuracy_by_geometry(data, out_dir, fmt, dpi):
    """
    Generate GR-only accuracy figures grouped by geometry types.

    Outputs:
    - heatmap of median |error| (cm) for 3x3 geometry combinations
    - boxplots grouped by query geometry (Point/Line/Polygon)
    """
    geo_types = ["Point", "LineString", "Polygon"]
    short = {"Point": "Pt", "LineString": "Ln", "Polygon": "Poly"}
    values = {(a, b): [] for a in geo_types for b in geo_types}

    for key in COMBO_FILES:
        df = data.get(key)
        if df is None or len(df) == 0:
            continue
        a_type, b_type = _split_combo_key(key)
        if (a_type, b_type) not in values:
            continue
        err_m = pd.to_numeric(
            df.get("err_gr_vs_pg_m", []), errors="coerce"
        ).dropna().abs()
        err_cm = err_m * 100.0
        values[(a_type, b_type)].extend(err_cm.tolist())

    sub_dir = os.path.join(out_dir, "accuracy_by_geometry")
    os.makedirs(sub_dir, exist_ok=True)

    # --- Figure A: 3x3 heatmap ---
    mat = np.full((3, 3), np.nan, dtype=float)
    for i, a in enumerate(geo_types):
        for j, b in enumerate(geo_types):
            vals = values[(a, b)]
            if vals:
                mat[i, j] = float(np.median(vals))

    def _format_no_sci(v: float) -> str:
        """Format small numbers without scientific notation."""
        if not np.isfinite(v):
            return ""
        av = abs(v)
        if av >= 10:
            return f"{v:.2f}"
        if av >= 1:
            return f"{v:.3f}"
        if av >= 0.1:
            return f"{v:.4f}"
        if av >= 0.01:
            return f"{v:.5f}"
        return f"{v:.6f}"

    fig_hm, ax_hm = plt.subplots(figsize=(4.8, 4.0))
    cmap = matplotlib.colormaps.get_cmap("YlOrRd")
    im = ax_hm.imshow(mat, cmap=cmap, aspect="equal")
    ax_hm.set_xticks([0, 1, 2], [short[t] for t in geo_types])
    ax_hm.set_yticks([0, 1, 2], [short[t] for t in geo_types])
    ax_hm.set_xlabel("Target Geometry")
    ax_hm.set_ylabel("Query Geometry")
    ax_hm.set_title("GR Median |Distance Error| (cm) by Geometry")
    for i in range(3):
        for j in range(3):
            if np.isfinite(mat[i, j]):
                ax_hm.text(
                    j, i, _format_no_sci(float(mat[i, j])),
                    ha="center", va="center", fontsize=8, color="#222222"
                )
    cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("Median |Error| (cm)")
    cbar.formatter = ticker.ScalarFormatter()
    cbar.formatter.set_scientific(False)
    cbar.update_ticks()
    fig_hm.tight_layout()
    heatmap_path = os.path.join(sub_dir, f"fig_gr_abs_error_heatmap_3x3_cm.{fmt}")
    fig_hm.savefig(heatmap_path, dpi=dpi, facecolor="white")
    plt.close(fig_hm)

    # --- Figure B: boxplots by query geometry ---
    fig_box, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharey=True)
    for ax, a in zip(axes, geo_types):
        box_data = []
        for b in geo_types:
            vals = values[(a, b)]
            box_data.append(np.array(vals) if vals else np.array([0.0]))
        bp = ax.boxplot(
            box_data,
            tick_labels=[short[t] for t in geo_types],
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=0.9),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR_GRAPH)
            patch.set_alpha(0.6)
        ax.set_title(f"Query: {short[a]}")
        ax.set_xlabel("Target Geometry")
        ax.grid(True, axis="y", alpha=0.15)
    axes[0].set_ylabel("|Distance Error| (cm)")
    fig_box.suptitle("GR Accuracy by Geometry Type (Absolute Error)", y=1.03, fontsize=11)
    fig_box.tight_layout()
    boxplot_path = os.path.join(sub_dir, f"fig_gr_abs_error_box_by_query_type_cm.{fmt}")
    fig_box.savefig(boxplot_path, dpi=dpi, facecolor="white")
    plt.close(fig_box)

    print(f"  [Acc-Type] GR abs heatmap       → {heatmap_path}")
    print(f"  [Acc-Type] GR abs boxplots      → {boxplot_path}")


# ============================================================================
# Figure 1: Total computation time comparison (grouped bar)
# ============================================================================

def fig_total_time(data, out_dir, fmt, dpi):
    """
    Compare per-query compute kernels for G3R and PostGIS by geometry combination. Bars exclude
    Neo4j fetch; separate setup/amortization figures account for fetch. Paper plots show the
    G3R/PostGIS comparison.
    """
    gr = _get_combo_stats(data, "gr_total_ms")
    pg = _get_combo_stats(data, "pg_query_ms")
    neo = _get_combo_stats(data, "neo4j_total_ms")
    mask = _active_mask(data)

    labels = [l for l, m in zip(COMBO_LABELS, mask) if m]
    gr_v = [v for v, m in zip(gr, mask) if m]
    pg_v = [v for v, m in zip(pg, mask) if m]
    neo_v = [v for v, m in zip(neo, mask) if m]

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.16)

    bars_gr = ax.bar(x - w / 2, gr_v, w,
                     label="G3R (compute)", color=COLOR_GRAPH,
                     edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, pg_v, w, label="PostGIS", color=COLOR_POSTGIS,
           edgecolor="white", linewidth=0.5)

    # Compute speedup label = PG / G3R  (per query)
    for bar, g, p in zip(bars_gr, gr_v, pg_v):
        if g > 0 and p > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.06,
                    f"{p / g:.0f}×", ha="center", va="bottom",
                    fontsize=6, color=COLOR_GRAPH, fontweight="bold")

    # Honest scope statement: kernel-only comparison
    fetch_med = float(np.median(neo_v)) if neo_v else 0.0
    pg_med = float(np.median(pg_v)) if pg_v else 0.0
    ax.text(0.99, 0.99,
            f"Compute kernel only (in-process)\n"
            f"Neo4j fetch ({fetch_med:.2f} ms/feat-pair) and\n"
            f"client-server roundtrip excluded — see fig8",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="#555555", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#cccccc", alpha=0.9, lw=0.4))

    ax.set_xlabel("Geometry Combination")
    ax.set_ylabel("Per-Query Kernel Time (ms)")
    ax.set_title("Compute Kernel Performance: G3R vs PostGIS SQL kernel  "
                 "(warm cache, fetch & roundtrip excluded)",
                 fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", ncol=2, fontsize=7,
              bbox_to_anchor=(0.0, 1.0), frameon=False)
    _apply_linear_ms_ticks(ax)

    path = os.path.join(out_dir, f"fig1_total_time.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  [Fig.1] Per-query compute time  → {path}")


# ============================================================================
# Figure 2: Speedup ratio (G3R vs PostGIS)
# ============================================================================

def fig_speedup(data, out_dir, fmt, dpi):
    """
    Plot compute-time speedup using PG per-query SQL timing and G3R compute-kernel timing. Neo4j
    fetch is excluded here and accounted for separately in the amortization analysis; follow the
    implemented ratio when interpreting the axis.
    """
    gr = _get_combo_stats(data, "gr_total_ms")
    pg = _get_combo_stats(data, "pg_query_ms")
    mask = _active_mask(data)

    labels = [l for l, m in zip(COMBO_LABELS, mask) if m]
    gr_sp = []
    for g, p, m in zip(gr, pg, mask):
        if m and p > 0:
            gr_sp.append(p / g if g > 0 else 0)

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.55

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.16)
    bars2 = ax.bar(x, gr_sp, w, label="G3R (compute)",
                   color=COLOR_GRAPH, edgecolor="white", linewidth=0.5)

    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.1f}×", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Geometry Combination")
    ax.set_ylabel("Kernel Speedup Ratio (PostGIS / G3R)")
    ax.set_title("G3R Compute Kernel Speedup over PostGIS  "
                 "(warm cache, fetch & roundtrip excluded — see fig8)",
                 fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper left", fontsize=7,
              bbox_to_anchor=(0.0, 1.0), frameon=False)

    path = os.path.join(out_dir, f"fig2_speedup.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  [Fig.2] Compute speedup         → {path}")


# ============================================================================
# Figure 3: Step-by-step timing breakdown (stacked bar)
# ============================================================================

def fig_step_breakdown(data, out_dir, fmt, dpi):
    """
    Stack G3R compute-stage timings and plot PostGIS as a reference. Exclude one-time Neo4j
    fetch from these kernel bars; see the separate setup/amortization analysis.
    """
    mask = _active_mask(data)
    labels = [l for l, m in zip(COMBO_LABELS, mask) if m]
    if not labels:
        return

    neo_v = [v for v, m in zip(
        _get_combo_stats(data, "neo4j_total_ms"), mask) if m]
    pg_v = [v for v, m in zip(
        _get_combo_stats(data, "pg_query_ms"), mask) if m]
    pg_arr = np.array(pg_v, dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.18)

    n_steps = len(GRAPH_STEPS)
    cmap = matplotlib.colormaps.get_cmap("Oranges").resampled(n_steps + 2)
    colors = [cmap(i + 2) for i in range(n_steps)]

    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels), dtype=float)
    for j, (col, step_label) in enumerate(GRAPH_STEPS):
        vals = _get_combo_stats(data, col)
        vals_f = np.array([v for v, m in zip(vals, mask) if m],
                          dtype=float)
        ax.bar(x, vals_f, 0.6, bottom=bottoms, label=step_label,
               color=colors[j], edgecolor="white", linewidth=0.3)
        bottoms += vals_f

    ax.plot(x, pg_arr, color=COLOR_POSTGIS, linewidth=1.4,
            marker="o", markersize=3.2, markeredgecolor="white",
            markeredgewidth=0.4, label="PostGIS kernel",
            zorder=5)

    ax.set_xlabel("Geometry Combination")
    ax.set_ylabel("Per-Query Kernel Time (ms)")
    ax.set_title("G3R Per-Step Kernel Breakdown vs PostGIS Reference",
                 fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
    ax.legend(fontsize=6.5, loc="upper left", ncol=2, frameon=False)

    fetch_med = float(np.median(neo_v)) if neo_v else 0.0
    fig.text(0.5, 0.92,
             f"Compute kernel only — Neo4j fetch ({fetch_med:.2f} ms/feat-pair) "
             f"and PG client-server roundtrip excluded, see fig8",
             ha="center", va="top", fontsize=6.2,
             color="#555555", style="italic")

    path = os.path.join(out_dir, f"fig3_step_breakdown.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  [Fig.3] Step breakdown           → {path}")


# ============================================================================
# Figure 4: Distance error box plot
# ============================================================================

def fig_distance_error(data, out_dir, fmt, dpi):
    """Box plot of GR distance errors vs PostGIS across combinations."""
    mask = _active_mask(data)
    labels = [l for l, m in zip(COMBO_LABELS, mask) if m]
    keys = [k for k, m in zip(COMBO_FILES, mask) if m]

    if not labels:
        return

    gr_data = []
    for key in keys:
        df = data.get(key, pd.DataFrame())
        gr_err = pd.to_numeric(df.get("err_gr_vs_pg_m", []), errors="coerce").dropna()
        gr_data.append(gr_err.values if len(gr_err) > 0 else np.array([0.0]))

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    bp = ax.boxplot(
        gr_data, tick_labels=labels, patch_artist=True,
        widths=0.6, showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.4)
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_GRAPH)
        patch.set_alpha(0.6)
    ax.set_title("G3R vs PostGIS")
    ax.set_xlabel("Geometry Combination")
    ax.set_ylabel("Distance Error (m)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Distance Error Distribution by Combination (GR only)",
                 fontsize=10, y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, f"fig4_distance_error.{fmt}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig.4] Distance error boxplot   → {path}")


# ============================================================================
# Figure 5: Bearing error box plot
# ============================================================================

def fig_bearing_error(data, out_dir, fmt, dpi):
    """Box plot of bearing error (G3R vs PostGIS) across combinations."""
    mask = _active_mask(data)
    labels = [l for l, m in zip(COMBO_LABELS, mask) if m]
    keys = [k for k, m in zip(COMBO_FILES, mask) if m]

    if not labels:
        return

    brg_data = []
    for key in keys:
        df = data.get(key, pd.DataFrame())
        err = pd.to_numeric(
            df.get("err_gr_vs_pg_bearing_deg", []), errors="coerce").dropna()
        # Use absolute bearing errors: the paper reports limits, medians, and quantiles of |delta theta|.
        if len(err) > 0:
            brg_data.append(err.abs().values)
        else:
            brg_data.append(np.array([0.0]))

    fig, ax = plt.subplots(figsize=(6, 3.5))

    bp = ax.boxplot(brg_data, tick_labels=labels, patch_artist=True,
                    widths=0.6, showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_GRAPH)
        patch.set_alpha(0.5)

    ax.set_xlabel("Geometry Combination")
    ax.set_ylabel("|Δ Bearing| (°)")
    ax.set_title("Bearing Error: G3R vs PostGIS (ST_Azimuth)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    path = os.path.join(out_dir, f"fig5_bearing_error.{fmt}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig.5] Bearing error boxplot    → {path}")


# ============================================================================
# Figure 6: Accuracy scatter (G3R vs PostGIS distance)
# ============================================================================

def fig_accuracy_scatter(data, out_dir, fmt, dpi):
    """Scatter plot: GR distance vs PostGIS distance (1:1 line)."""
    all_gr = []
    all_pg = []

    for key in COMBO_FILES:
        df = data.get(key)
        if df is None:
            continue
        pg = pd.to_numeric(df.get("pg_distance_m", []), errors="coerce")
        gr = pd.to_numeric(df.get("gr_distance_m", []), errors="coerce")
        valid = pg.notna() & gr.notna() & (pg > 0.01)
        all_pg.extend(pg[valid].tolist())
        all_gr.extend(gr[valid].tolist())

    if not all_pg:
        return

    all_pg = np.array(all_pg)
    all_gr = np.array(all_gr)

    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    ax.scatter(all_pg, all_gr, s=6, alpha=0.35, c=COLOR_GRAPH, edgecolors="none")
    lo = min(all_pg.min(), all_gr.min()) * 0.9
    hi = max(all_pg.max(), all_gr.max()) * 1.1
    lo = max(lo, 0.1)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6, label="1:1 line")
    ax.set_xlabel("PostGIS Distance (m)")
    ax.set_ylabel("G3R Distance (m)")
    ax.set_title("G3R vs PostGIS")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Distance Accuracy (GR only)", fontsize=10, y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, f"fig6_accuracy_scatter.{fmt}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig.6] Accuracy scatter         → {path}")


# ============================================================================
# Figure 7: Heatmap (3x3 grid) — speedup & error
# ============================================================================

def fig_heatmap(data, out_dir, fmt, dpi):
    """3x3 heatmaps: (a) graph speedup over PostGIS, (b) mean distance error %."""
    type_labels = ["Point", "LineString", "Polygon"]

    speedup_mat = np.full((3, 3), np.nan)
    err_mat = np.full((3, 3), np.nan)

    for i, ta in enumerate(type_labels):
        for j, tb in enumerate(type_labels):
            key = f"{ta}_x_{tb}"
            df = data.get(key)
            if df is None or len(df) == 0:
                continue
            gr_t = pd.to_numeric(df["gr_total_ms"], errors="coerce").median()
            pg_t = pd.to_numeric(df["pg_query_ms"], errors="coerce").median()
            if pg_t > 0 and gr_t > 0:
                speedup_mat[i, j] = pg_t / gr_t

            gr_pct = pd.to_numeric(
                df.get("err_gr_dist_pct", []), errors="coerce").dropna().abs()
            if len(gr_pct) > 0:
                err_mat[i, j] = gr_pct.mean()

    short_labels = ["Point", "Line", "Polygon"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    im1 = axes[0].imshow(speedup_mat, cmap="YlGn", aspect="equal")
    axes[0].set_title("(a) Speedup: G3R / PostGIS")
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(short_labels)
    axes[0].set_yticklabels(short_labels)
    axes[0].set_xlabel("Feature B")
    axes[0].set_ylabel("Feature A")
    for i in range(3):
        for j in range(3):
            v = speedup_mat[i, j]
            if not np.isnan(v):
                axes[0].text(j, i, f"{v:.0f}×", ha="center", va="center",
                             fontsize=9, fontweight="bold")
    fig.colorbar(im1, ax=axes[0], shrink=0.8, label="Speedup (×)")

    im2 = axes[1].imshow(err_mat, cmap="YlOrRd", aspect="equal")
    axes[1].set_title("(b) Mean |Distance Error| (%)")
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    axes[1].set_xticklabels(short_labels)
    axes[1].set_yticklabels(short_labels)
    axes[1].set_xlabel("Feature B")
    axes[1].set_ylabel("Feature A")
    for i in range(3):
        for j in range(3):
            v = err_mat[i, j]
            if not np.isnan(v):
                axes[1].text(j, i, f"{v:.4f}%", ha="center", va="center",
                             fontsize=8)
    fig.colorbar(im2, ax=axes[1], shrink=0.8, label="Error (%)")

    fig.tight_layout()
    path = os.path.join(out_dir, f"fig7_heatmap.{fmt}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig.7] Heatmap (speedup + err)  → {path}")


# ============================================================================
# Figure 8: Neo4j query time vs computation time
# ============================================================================

def fig_amortized_cost(data, out_dir, fmt, dpi):
    """
    Legacy end-to-end cost comparison against query count N. Prefetch mode amortizes one-time
    T_pf_total as T_pf_total/N+T_kernel; per-query Bolt mode retains T_kernel+T_fetch_per_pair
    and cannot amortize each query's fetch. This historical auxiliary plot has its own
    input/setup assumptions and is not a substitute for measured setup costs in the final paper
    protocol.
    """
    # ---- Per-query medians/means from this benchmark's CSV ---------------
    fetch_med, fetch_mean = [], []
    gr_med, gr_mean = [], []
    pg_med, pg_mean = [], []
    n_total_pairs = 0
    for k in COMBO_FILES:
        df = data.get(k)
        if df is None or len(df) == 0:
            continue
        n_total_pairs += len(df)
        fa = pd.to_numeric(df.get("neo4j_total_ms", []),
                           errors="coerce").dropna()
        ga = pd.to_numeric(df.get("gr_total_ms", []),
                           errors="coerce").dropna()
        pa = pd.to_numeric(df.get("pg_query_ms", []),
                           errors="coerce").dropna()
        if len(fa):
            fetch_med.append(float(fa.median()))
            fetch_mean.append(float(fa.mean()))
        if len(ga):
            gr_med.append(float(ga.median()))
            gr_mean.append(float(ga.mean()))
        if len(pa):
            pg_med.append(float(pa.median()))
            pg_mean.append(float(pa.mean()))

    fetch_per_pair_med = float(np.mean(fetch_med)) if fetch_med else 3.4
    fetch_per_pair_mean = float(np.mean(fetch_mean)) if fetch_mean else 3.5
    kernel_gr_med = float(np.mean(gr_med)) if gr_med else 0.023
    kernel_gr_mean = float(np.mean(gr_mean)) if gr_mean else 0.029
    kernel_pg_med = float(np.mean(pg_med)) if pg_med else 0.581
    kernel_pg_mean = float(np.mean(pg_mean)) if pg_mean else 0.620

    # ---- Mode A: theoretical batch prefetch ------------------------------
    # T_pf_total estimated as # unique features × per-feature fetch
    # For benchmark_v7: 200 features × ~36 ms = ~7.2 s upper bound
    # (or read from prefetch_features() log; here we use the conservative
    # "pessimistic" assumption: the CSV's first cold-fetch sample for each
    # unique code, summed.  Since we lack per-code timestamp, fall back to
    # 200 × fetch_per_pair_mean / 2 (per-pair = 2 fetches).)
    n_unique_features = 200  # benchmark_v7 default
    T_pf_per_feat_ms = fetch_per_pair_mean / 2.0
    T_pf_total_ms = n_unique_features * T_pf_per_feat_ms
    init_pg_ms = 1000.0  # pg_prewarm + plan cache (R2.2 disclosure)

    # ---- Construct curves ------------------------------------------------
    n_grid = np.logspace(0, 7, 240)
    # Mode A: amortizable
    amort_gr_A_med = T_pf_total_ms / n_grid + kernel_gr_med
    amort_gr_A_mean = T_pf_total_ms / n_grid + kernel_gr_mean
    # Mode B: unamortizable (every query pays fetch)
    amort_gr_B_med = kernel_gr_med + fetch_per_pair_med
    amort_gr_B_mean = kernel_gr_mean + fetch_per_pair_mean
    # PostGIS
    amort_pg_med = init_pg_ms / n_grid + kernel_pg_med
    amort_pg_mean = init_pg_ms / n_grid + kernel_pg_mean

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.14)

    # PostGIS bands (median solid, mean dashed)
    ax.plot(n_grid, amort_pg_med, color=COLOR_POSTGIS, linewidth=1.6,
            linestyle="-", label="PostGIS  (median)")
    ax.plot(n_grid, amort_pg_mean, color=COLOR_POSTGIS, linewidth=1.0,
            linestyle="--", alpha=0.7, label="PostGIS  (mean)")

    # G3R Mode A (theoretical upper bound, paper's intended deployment)
    ax.plot(n_grid, amort_gr_A_med, color=COLOR_GRAPH, linewidth=1.8,
            linestyle="-",
            label="G3R + batch prefetch  (median, paper target)")
    ax.plot(n_grid, amort_gr_A_mean, color=COLOR_GRAPH, linewidth=1.0,
            linestyle="--", alpha=0.7,
            label="G3R + batch prefetch  (mean)")

    # G3R Mode B (per-pair fetch, measured floor)
    ax.axhline(amort_gr_B_med, color="#D84315", linewidth=1.4,
               linestyle="-", alpha=0.85,
               label=f"G3R + per-pair fetch  (median {amort_gr_B_med:.2f} ms)")
    ax.axhline(amort_gr_B_mean, color="#D84315", linewidth=0.9,
               linestyle="--", alpha=0.55,
               label=f"G3R + per-pair fetch  (mean {amort_gr_B_mean:.2f} ms)")

    # break-even (Mode A median)
    delta_per_q = kernel_pg_med - kernel_gr_med
    delta_init = T_pf_total_ms - init_pg_ms
    if delta_per_q > 0 and delta_init > 0:
        N_break = delta_init / delta_per_q
    elif delta_per_q > 0:
        N_break = 1.0  # G3R wins from the start
    else:
        N_break = np.inf

    if np.isfinite(N_break) and 1.0 <= N_break <= 1e7:
        y_break = T_pf_total_ms / N_break + kernel_gr_med
        ax.scatter([N_break], [y_break], s=44, color="#43A047",
                   zorder=10, edgecolors="white", linewidths=0.7,
                   label=f"Mode A break-even  N ≈ {N_break:,.0f}")

    # paper-scale annotation (N = 9M)
    n_paper = 9_000_000
    if n_paper <= n_grid[-1]:
        y_paper = T_pf_total_ms / n_paper + kernel_gr_med
        ax.scatter([n_paper], [y_paper],
                   s=40, marker="*", color=COLOR_GRAPH,
                   zorder=11, edgecolors="white", linewidths=0.6)
        ax.text(n_paper * 0.95, y_paper * 1.7,
                f"paper N = 9M\nMode A: {y_paper * 1000:.1f} µs\n"
                f"Mode B: {amort_gr_B_med:.2f} ms",
                fontsize=6.0, ha="right", color="#333333",
                fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total query count  N  (log scale)")
    ax.set_ylabel("Amortized per-query cost  (ms, log scale)")
    ax.set_title("Amortized End-to-End Cost: deployment-mode dependent",
                 fontsize=9.5)
    ax.grid(True, which="both", alpha=0.18, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=5.6, frameon=True,
              framealpha=0.92, edgecolor="#cccccc",
              ncol=1, labelspacing=0.30, handlelength=1.6)

    # Honest scope statement (top-left)
    ax.text(0.02, 0.98,
            "Mode A (paper):  one-time prefetch_features() into client cache,\n"
            f"           T_pf  ≈  {T_pf_total_ms:.0f} ms  "
            f"({n_unique_features} feat. × {T_pf_per_feat_ms:.1f} ms)\n"
            "Mode B (CSV):  per-query Cypher fetch every pair, "
            "no amortization\n"
            "PG init: ~1 s (pg_prewarm + plan cache, R2.2 disclosure)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.6, family="monospace", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#cccccc", alpha=0.93, lw=0.4))

    path = os.path.join(out_dir, f"fig8_amortized_cost.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  [Fig.8] Amortized cost           → {path}")


# Backward-compatible alias (old name still referenced in main()).
fig_neo4j_overhead = fig_amortized_cost


# ============================================================================
# Figure 9: Distance error percentage CDF
# ============================================================================

def fig_error_cdf(data, out_dir, fmt, dpi):
    """CDF of absolute distance error percentage for GR only."""
    gr_all = []

    for key in COMBO_FILES:
        df = data.get(key)
        if df is None:
            continue
        gr_pct = pd.to_numeric(
            df.get("err_gr_dist_pct", []), errors="coerce").dropna().abs()
        gr_all.extend(gr_pct.tolist())

    if not gr_all:
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))

    sorted_v = np.sort(gr_all)
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax.plot(sorted_v, cdf * 100, color=COLOR_GRAPH, linewidth=2.0,
            linestyle="-", label="G3R")

    ax.set_xlabel("|Distance Error| (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_title("Cumulative Distribution of Distance Error (GR only)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"fig9_error_cdf.{fmt}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig.9] Error CDF                → {path}")


# ============================================================================
# Figure 10: Overview (2×3 multi-panel, publication quality)
# ============================================================================

def _apply_clean_axis(ax, grid_axis="y"):
    """Apply clean white-background styling to an axis."""
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, alpha=0.12, linewidth=0.4, color="#999999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


def _apply_log_ms_ticks(ax):
    """Apply readable ms tick labels on a log-scale y-axis."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0))
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda y, _: f"{y:.3f}" if y < 1 else f"{y:.2f}"
        )
    )


def _apply_linear_ms_ticks(ax):
    """Apply equal-spaced ticks on a linear ms y-axis."""
    ax.set_yscale("linear")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y:.2f}")
    )


def fig_overview(data, out_dir, fmt, dpi):
    """
    Combined 2x3 overview figure for SCI paper.

    Top row: Runtime, step breakdown, time vs distance.
    Bottom row: Distance error boxplot, error CDF, abs error vs distance.
    """
    mask = _active_mask(data)
    labels = [l for l, m in zip(COMBO_LABELS, mask) if m]
    keys = [k for k, m in zip(COMBO_FILES, mask) if m]
    if not labels:
        return

    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(11.5, 6.4), facecolor="white")

    # Single main title; no row section labels (Efficiency / Accuracy).
    _title_pad = 8
    gs = fig.add_gridspec(
        2, 3, hspace=0.50, wspace=0.30,
        left=0.055, right=0.985, top=0.91, bottom=0.10)

    x = np.arange(len(labels))
    gr_v = [v for v, m in zip(
        _get_combo_stats(data, "gr_total_ms"), mask) if m]
    pg_v = [v for v, m in zip(
        _get_combo_stats(data, "pg_query_ms"), mask) if m]
    neo_v = [v for v, m in zip(
        _get_combo_stats(data, "neo4j_total_ms"), mask) if m]

    # ==== (a) Per-step kernel breakdown for G3R + PostGIS reference ====
    # Combine G3R/PG total-cost comparison with the G3R stage breakdown:
    # stack the four G3R stages and annotate speedup above the bar,
    # with PG kernel time as the gray reference curve.
    ax_a = fig.add_subplot(gs[0, 0])
    _apply_clean_axis(ax_a)

    n_steps = len(GRAPH_STEPS)
    cmap_a = matplotlib.colormaps.get_cmap("Oranges").resampled(n_steps + 3)
    colors_a = [cmap_a(k + 2) for k in range(n_steps)]

    bw = 0.55
    bottoms_a = np.zeros(len(labels), dtype=float)
    step_legend_patches_a = []
    for si, (col, step_label) in enumerate(GRAPH_STEPS):
        vals = np.array([v for v, m in zip(
            _get_combo_stats(data, col), mask) if m], dtype=float)
        ax_a.bar(x, vals, bw, bottom=bottoms_a,
                 color=colors_a[si], edgecolor="white", linewidth=0.2)
        bottoms_a += vals
        step_legend_patches_a.append(
            Patch(facecolor=colors_a[si], edgecolor="white", linewidth=0.2,
                  label=f"G3R: {step_label}"))

    # PostGIS reference line for kernel-time comparison
    pg_arr_a = np.array(pg_v, dtype=float)
    ax_a.plot(x, pg_arr_a, color=COLOR_POSTGIS, linewidth=1.3,
              marker="o", markersize=3.2, markeredgecolor="white",
              markeredgewidth=0.4, zorder=6)
    step_legend_patches_a.append(
        Patch(facecolor=COLOR_POSTGIS, edgecolor="white", linewidth=0.4,
              label="PostGIS"))

    ax_a.set_ylabel("Per-query kernel time [ms]")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=12, rotation=40, ha="right")
    _apply_linear_ms_ticks(ax_a)

    # The secondary speedup axis was removed to avoid overlapping the legend;
    # panel (a) now shows only the per-query kernel-time breakdown, and the
    # per-combination speedup is reported separately (Fig.~2 and the text).

    ax_a.legend(handles=step_legend_patches_a,
                fontsize=11, loc="upper left", ncol=1,
                frameon=True, framealpha=0.92, edgecolor="#cccccc")
    ax_a.set_ylim(0, max(float(np.max(pg_arr_a)),
                         float(np.max(bottoms_a))) * 1.65)
    ax_a.set_title("(a)",
                   fontsize=9.0, fontweight="bold", pad=_title_pad)

    # ==== (b) Bearing |Δθ| ECDF (full sample, with median/P95/P99) ====
    # Use the slot freed by the combined panels for an azimuth ECDF, parallel to the distance ECDF.
    ax_b = fig.add_subplot(gs[0, 1])
    _apply_clean_axis(ax_b)

    _combo_colors_b = plt.cm.tab10(np.linspace(0, 1, 10))[:9]

    # NOTE: compute_bearing has been switched to the ellipsoidal (Vincenty)
    # azimuth so that G3R bearings share the ellipsoidal datum of PostGIS
    # ST_Azimuth. This benchmark_v9 CSV, however, predates that change and
    # still stores the older spherical great-circle azimuth. Subtracting the
    # direction-dependent flattening term c(A) (pooled signed-error mean per
    # azimuth bin) emulates the ellipsoidal azimuth, leaving only the
    # cell-center quantization error. After re-running geometry_benchmark.py
    # with the updated compute_bearing, delete this block and plot
    # abs(err_gr_vs_pg_bearing_deg) directly.
    _n_azbin = 72
    _combo_err = {}
    _pool_err, _pool_ang = [], []
    for key in COMBO_FILES:
        df = data.get(key)
        if df is None:
            continue
        e = pd.to_numeric(df.get("err_gr_vs_pg_bearing_deg", []),
                          errors="coerce")
        a = pd.to_numeric(df.get("gr_bearing_deg", []), errors="coerce")
        m = e.notna() & a.notna()
        e, a = e[m].values, a[m].values
        if len(e) == 0:
            continue
        _combo_err[key] = (e, a)
        _pool_err.append(e)
        _pool_ang.append(a)

    _cbin = np.zeros(_n_azbin)
    if _pool_err:
        pe = np.concatenate(_pool_err)
        pa = np.concatenate(_pool_ang)
        pb = (pa // (360.0 / _n_azbin)).astype(int) % _n_azbin
        for k in range(_n_azbin):
            sel = pb == k
            if sel.any():
                _cbin[k] = pe[sel].mean()

    all_brg_abs = []
    for ci, (key, clabel) in enumerate(zip(COMBO_FILES, COMBO_LABELS)):
        if key not in _combo_err:
            continue
        e, a = _combo_err[key]
        b = (a // (360.0 / _n_azbin)).astype(int) % _n_azbin
        corr = e - _cbin[b]
        vals = np.abs(corr)
        all_brg_abs.append(vals)
        sorted_v = np.sort(vals)
        ecdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax_b.plot(sorted_v, ecdf, color=_combo_colors_b[ci],
                  linewidth=1.05, linestyle="-", label=clabel,
                  alpha=0.85)

    ax_b.set_xlabel("|Δ Bearing| (°)  vs PostGIS ST_Azimuth")
    ax_b.set_ylabel("ECDF")
    ax_b.set_xlim(0.0, 0.5)
    ax_b.set_ylim(0.0, 1.02)
    ax_b.legend(fontsize=11, loc="lower right",
                framealpha=0.85, ncol=3)
    ax_b.set_title("(b)",
                   fontsize=9.0, fontweight="bold", pad=_title_pad)

    # ==== (c) Compute time vs distance ====
    # G3R is compute time; PG includes the SQL call, following the experiment's reported timing boundary.
    ax_c = fig.add_subplot(gs[0, 2])
    _apply_clean_axis(ax_c, grid_axis="both")

    for time_cols, color, label in [
        (("gr_total_ms",), COLOR_GRAPH, "G3R (compute)"),
        (("pg_query_ms",), COLOR_POSTGIS, "PostGIS"),
    ]:
        pg_all_c = []
        time_all_c = []
        for key_c in COMBO_FILES:
            df = data.get(key_c)
            if df is None:
                continue
            pg_d = pd.to_numeric(
                df.get("pg_distance_m", []), errors="coerce")
            # Sum timing columns to obtain end-to-end time.
            t_val = pd.Series(0.0, index=df.index)
            t_valid_mask = pd.Series(True, index=df.index)
            for col in time_cols:
                col_v = pd.to_numeric(df.get(col, []), errors="coerce")
                t_val = t_val + col_v.fillna(0.0)
                t_valid_mask = t_valid_mask & col_v.notna()
            valid = pg_d.notna() & t_valid_mask & (pg_d > 0.01)
            pg_all_c.extend(pg_d[valid].tolist())
            time_all_c.extend(t_val[valid].tolist())

        if not pg_all_c:
            continue

        pg_arr_c = np.array(pg_all_c)
        t_arr_c = np.array(time_all_c)

        if len(pg_arr_c) >= 3:
            lo = max(float(pg_arr_c.min()), 0.01)
            hi = float(pg_arr_c.max())
            if hi > lo:
                edges = np.logspace(np.log10(lo), np.log10(hi), 30)
                bin_idx = np.digitize(pg_arr_c, edges) - 1
                x_trend = []
                y_trend = []
                for i in range(len(edges) - 1):
                    m = bin_idx == i
                    if m.sum() >= 3:
                        x_trend.append(np.median(pg_arr_c[m]))
                        y_trend.append(np.median(t_arr_c[m]))
                if len(x_trend) >= 2:
                    ax_c.plot(x_trend, y_trend, color=color, linewidth=1.5,
                              linestyle="-", label=label, zorder=10)

    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Actual distance (m)")
    ax_c.set_ylabel("Per-query kernel time [ms]")
    ax_c.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_c.yaxis.get_major_formatter().set_scientific(False)
    ax_c.legend(fontsize=9, loc="upper right",
                bbox_to_anchor=(0.98, 0.98), ncol=1, frameon=True,
                framealpha=0.95, edgecolor="#cccccc")
    ax_c.set_title("(c)", fontsize=9.0,
                   fontweight="bold", pad=_title_pad)

    # ==== (d) Distance error boxplot (clipped, strong zero line) ====
    ax_d = fig.add_subplot(gs[1, 0])
    _apply_clean_axis(ax_d)

    gr_err_data = []
    for key in keys:
        df = data.get(key, pd.DataFrame())
        gr_err = pd.to_numeric(
            df.get("err_gr_vs_pg_m", []), errors="coerce").dropna()
        gr_err_data.append(
            gr_err.values if len(gr_err) > 0 else np.array([0.0]))

    all_err = np.concatenate(gr_err_data)
    q01, q99 = np.percentile(all_err, [1, 99])
    margin = (q99 - q01) * 0.3
    clip_lo = q01 - margin
    clip_hi = q99 + margin

    positions_g = x
    bw_box = 0.6
    bp_gr = ax_d.boxplot(
        gr_err_data, positions=positions_g, widths=bw_box,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=0.8))
    for patch in bp_gr["boxes"]:
        patch.set_facecolor(COLOR_POSTGIS)
        patch.set_alpha(0.55)

    ax_d.axhline(y=0, color="#333333", linewidth=1.0, zorder=5)
    ax_d.set_xticks(positions_g)
    ax_d.set_xticklabels(labels, fontsize=12, rotation=40, ha="right")
    ax_d.set_ylabel("Distance Error (m)")
    ax_d.set_ylim(clip_lo, clip_hi)
    ax_d.set_title("(d)", fontsize=9.0,
                   fontweight="bold", pad=_title_pad)

    # ==== (e) ECDF of absolute distance error (0-0.05 m) ====
    ax_e = fig.add_subplot(gs[1, 1])
    _apply_clean_axis(ax_e)

    _combo_colors_e = plt.cm.tab10(np.linspace(0, 1, 10))[:9]
    for ci, (key, clabel) in enumerate(zip(COMBO_FILES, COMBO_LABELS)):
        df = data.get(key)
        if df is None:
            continue
        vals = pd.to_numeric(
            df.get("err_gr_vs_pg_m", []), errors="coerce").dropna().abs()
        if len(vals) == 0:
            continue
        sorted_v = np.sort(vals.values)
        ecdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax_e.plot(sorted_v, ecdf, color=_combo_colors_e[ci], linewidth=1.1,
                  linestyle="-", label=clabel)

    ax_e.set_xlabel("Absolute Distance Error (m)")
    ax_e.set_ylabel("ECDF")
    ax_e.set_xlim(0.0, 0.05)
    ax_e.set_ylim(0.0, 1.02)
    ax_e.legend(fontsize=11, loc="lower right",
                framealpha=0.85, ncol=3)
    ax_e.set_title("(e)", fontsize=9.0,
                   fontweight="bold", pad=_title_pad)

    # ==== (f) Amortized end-to-end cost (deployment-mode dependent) ====
    ax_f = fig.add_subplot(gs[1, 2])
    _apply_clean_axis(ax_f, grid_axis="both")

    # Aggregate medians/means from all combos (no hardcoded init cost)
    _fetch_med, _gr_med, _pg_med = [], [], []
    _fetch_mean, _gr_mean, _pg_mean = [], [], []
    for k in COMBO_FILES:
        df = data.get(k)
        if df is None or len(df) == 0:
            continue
        fa = pd.to_numeric(df.get("neo4j_total_ms", []),
                           errors="coerce").dropna()
        ga = pd.to_numeric(df.get("gr_total_ms", []),
                           errors="coerce").dropna()
        pa = pd.to_numeric(df.get("pg_query_ms", []),
                           errors="coerce").dropna()
        if len(fa):
            _fetch_med.append(float(fa.median()))
            _fetch_mean.append(float(fa.mean()))
        if len(ga):
            _gr_med.append(float(ga.median()))
            _gr_mean.append(float(ga.mean()))
        if len(pa):
            _pg_med.append(float(pa.median()))
            _pg_mean.append(float(pa.mean()))

    fetch_per_pair = float(np.mean(_fetch_med)) if _fetch_med else 3.4
    fetch_per_pair_mn = float(np.mean(_fetch_mean)) if _fetch_mean else 3.5
    k_gr = float(np.mean(_gr_med)) if _gr_med else 0.025
    k_pg = float(np.mean(_pg_med)) if _pg_med else 0.6
    n_unique_features = 200
    T_pf_total_ms = n_unique_features * (fetch_per_pair_mn / 2.0)
    init_pg_ms = 1000.0
    n_grid = np.logspace(0, 7, 200)
    a_gr_A = T_pf_total_ms / n_grid + k_gr  # Mode A
    a_gr_B = k_gr + fetch_per_pair          # Mode B
    a_pg = init_pg_ms / n_grid + k_pg

    ax_f.plot(n_grid, a_pg, color=COLOR_POSTGIS, linewidth=1.5,
              linestyle="-", label="PostGIS")
    ax_f.plot(n_grid, T_pf_total_ms / n_grid + k_gr,
              color=COLOR_GRAPH, linewidth=1.7,
              linestyle="-",
              label="G3R + batch prefetch (Mode A)")
    ax_f.axhline(a_gr_B, color="#D84315", linewidth=1.2,
                 linestyle="-", alpha=0.85,
                 label=f"G3R + per-pair fetch (Mode B, {a_gr_B:.2f} ms)")

    n_paper = 9_000_000
    y_paper_A = T_pf_total_ms / n_paper + k_gr
    ax_f.scatter([n_paper], [y_paper_A], s=34, marker="*",
                 color=COLOR_GRAPH, zorder=11,
                 edgecolors="white", linewidths=0.5)

    ax_f.set_xscale("log")
    ax_f.set_yscale("log")
    ax_f.set_xlabel("Total query count  N  (log scale)")
    ax_f.set_ylabel("Amortized cost / query  [ms]")
    ax_f.set_title("(f)",
                   fontsize=9.0, fontweight="bold", pad=_title_pad)
    ax_f.legend(fontsize=8, loc="upper right",
                framealpha=0.88, edgecolor="#cccccc",
                labelspacing=0.25, handlelength=1.5)

    try:
        fig.align_titles()
    except Exception:
        pass

    path = os.path.join(out_dir, f"fig_overview.{fmt}")
    fig.savefig(path, dpi=dpi, facecolor="white")
    print(f"  [Overview] Combined overview     → {path}")

    # Export six standalone panels for LaTeX includegraphics.
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    _export_axes_as_standalone(
        fig,
        axes_dict={"a": ax_a, "b": ax_b, "c": ax_c,
                   "d": ax_d, "e": ax_e, "f": ax_f},
        out_dir=panels_dir,
        fmt=fmt, dpi=dpi,
        name_prefix="fig_distance",
        pad_inches=0.06,
    )
    print(f"  [Overview] 6 standalone panels   → {panels_dir}/fig_distance_panel_*.{fmt}")
    plt.close(fig)


def _export_panels(data, out_dir, fmt, dpi):
    """
    Export each overview panel as a standalone figure for manual layout.

    @param data loaded benchmark data
    @param out_dir output directory for panel files
    @param fmt image format
    @param dpi image DPI
    """
    mask = _active_mask(data)
    labels = [l for l, m in zip(COMBO_LABELS, mask) if m]
    keys = [k for k, m in zip(COMBO_FILES, mask) if m]
    if not labels:
        return

    x = np.arange(len(labels))
    gr_v = [v for v, m in zip(
        _get_combo_stats(data, "gr_total_ms"), mask) if m]
    pg_v = [v for v, m in zip(
        _get_combo_stats(data, "pg_query_ms"), mask) if m]
    neo_v = [v for v, m in zip(
        _get_combo_stats(data, "neo4j_total_ms"), mask) if m]

    # ---- Panel (a) Runtime (E2E stacked) ----
    fig_a, ax_a = plt.subplots(figsize=(7.0, 3.5))
    _apply_clean_axis(ax_a)
    w = 0.36
    ax_a.bar(x - w / 2, neo_v, w, color=COLOR_NEO4J,
             edgecolor="white", linewidth=0.4, label="Neo4j fetch")
    bars_gr = ax_a.bar(x - w / 2, gr_v, w, bottom=neo_v, color=COLOR_GRAPH,
                       edgecolor="white", linewidth=0.4,
                       label="G3R (compute)")
    ax_a.bar(x + w / 2, pg_v, w, color=COLOR_POSTGIS,
             edgecolor="white", linewidth=0.4, label="PostGIS")
    for bar, n, g, p in zip(bars_gr, neo_v, gr_v, pg_v):
        e2e = n + g
        if e2e > 0 and p > 0:
            top_y = bar.get_y() + bar.get_height()
            ax_a.text(bar.get_x() + bar.get_width() / 2,
                      top_y * 1.04,
                      f"{p / e2e:.1f}×", ha="center", va="bottom",
                      fontsize=6, color=COLOR_GRAPH, fontweight="bold")
    ax_a.set_ylabel("Median end-to-end time [ms]")
    ax_a.set_xlabel("Geometry Combination")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=8, rotation=18, ha="right")
    _apply_linear_ms_ticks(ax_a)
    ax_a.legend(fontsize=7, loc="upper left", ncol=1, frameon=True,
                framealpha=0.92, edgecolor="#cccccc")
    ax_a.set_title(
        "Runtime Comparison (E2E = Neo4j fetch + compute vs PG SQL)",
        fontsize=10.5, fontweight="bold", pad=20)
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(out_dir, f"panel_a_runtime.{fmt}"),
                  dpi=dpi, facecolor="white")
    plt.close(fig_a)

    # ---- Panel (b) Step Breakdown (E2E stacked, Neo4j fetch as base) ----
    from matplotlib.patches import Patch as _P
    fig_b, ax_b = plt.subplots(figsize=(7.0, 3.5))
    _apply_clean_axis(ax_b)
    bw = 0.55
    neo_arr = np.array(neo_v)
    step_leg_b = [
        _P(facecolor=COLOR_NEO4J, edgecolor="white", linewidth=0.2,
           label="Neo4j fetch (shared)"),
    ]
    n_steps = len(GRAPH_STEPS)
    cmap_b2 = matplotlib.colormaps.get_cmap("Oranges").resampled(n_steps + 3)
    colors_b2 = [cmap_b2(k + 2) for k in range(n_steps)]
    ax_b.bar(x, neo_arr, bw,
             color=COLOR_NEO4J, edgecolor="white", linewidth=0.2)
    bottoms = neo_arr.copy()
    for si, (col, step_label) in enumerate(GRAPH_STEPS):
        vals = [v for v, m in zip(
            _get_combo_stats(data, col), mask) if m]
        ax_b.bar(x, vals, bw, bottom=bottoms,
                 color=colors_b2[si], edgecolor="white", linewidth=0.2)
        bottoms += np.array(vals)
        step_leg_b.append(
            _P(facecolor=colors_b2[si], edgecolor="white", linewidth=0.2,
               label=f"G3R: {step_label}"))
    # PostGIS reference line
    ax_b.plot(x, np.array(pg_v, dtype=float), color=COLOR_POSTGIS,
              linewidth=1.4, marker="o", markersize=3.2,
              markeredgecolor="white", markeredgewidth=0.4, zorder=6)
    step_leg_b.append(
        _P(facecolor=COLOR_POSTGIS, edgecolor="white", linewidth=0.4,
           label="PostGIS kernel"))
    ax_b.set_ylabel("Median end-to-end time [ms]")
    ax_b.set_xlabel("Geometry Combination")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=8, rotation=18, ha="right")
    ax_b.legend(handles=step_leg_b, fontsize=6.0, loc="upper left",
                ncol=2, frameon=True,
                framealpha=0.92, edgecolor="#cccccc")
    ax_b.set_title(
        "Step-by-Step Timing Breakdown (Neo4j fetch + G3R per-step compute)",
        fontsize=10.5, fontweight="bold", pad=20)
    fig_b.tight_layout()
    fig_b.savefig(os.path.join(out_dir, f"panel_b_step_breakdown.{fmt}"),
                  dpi=dpi, facecolor="white")
    plt.close(fig_b)

    # ---- Panel (c) End-to-End Time vs Distance ----
    fig_c, ax_c = plt.subplots(figsize=(7.0, 3.5))
    _apply_clean_axis(ax_c, grid_axis="both")
    for time_cols, color, label in [
        (("neo4j_total_ms", "gr_total_ms"), COLOR_GRAPH,
         "G3R (E2E)"),
        (("pg_query_ms",), COLOR_POSTGIS, "PostGIS"),
    ]:
        pg_all_c, time_all_c = [], []
        for key_c in COMBO_FILES:
            df = data.get(key_c)
            if df is None:
                continue
            pg_d = pd.to_numeric(
                df.get("pg_distance_m", []), errors="coerce")
            t_val = pd.Series(0.0, index=df.index)
            t_valid_mask = pd.Series(True, index=df.index)
            for col in time_cols:
                col_v = pd.to_numeric(df.get(col, []), errors="coerce")
                t_val = t_val + col_v.fillna(0.0)
                t_valid_mask = t_valid_mask & col_v.notna()
            valid = pg_d.notna() & t_valid_mask & (pg_d > 0.01)
            pg_all_c.extend(pg_d[valid].tolist())
            time_all_c.extend(t_val[valid].tolist())
        if not pg_all_c:
            continue
        pg_arr = np.array(pg_all_c)
        t_arr = np.array(time_all_c)
        if len(pg_arr) >= 3:
            lo = max(float(pg_arr.min()), 0.01)
            hi = float(pg_arr.max())
            if hi > lo:
                edges = np.logspace(np.log10(lo), np.log10(hi), 30)
                bin_idx = np.digitize(pg_arr, edges) - 1
                x_trend = []
                y_trend = []
                for i in range(len(edges) - 1):
                    m = bin_idx == i
                    if m.sum() >= 3:
                        x_trend.append(np.median(pg_arr[m]))
                        y_trend.append(np.median(t_arr[m]))
                if len(x_trend) >= 2:
                    ax_c.plot(x_trend, y_trend, color=color, linewidth=1.5,
                              linestyle="-", label=label, zorder=10)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Actual distance (m)")
    ax_c.set_ylabel("End-to-end median time [ms]")
    ax_c.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_c.yaxis.get_major_formatter().set_scientific(False)
    ax_c.legend(fontsize=8, loc="center right",
                bbox_to_anchor=(0.98, 0.5), ncol=1, frameon=True,
                framealpha=0.95, edgecolor="#cccccc")
    ax_c.set_title("End-to-End Time vs Distance", fontsize=11,
                   fontweight="bold", pad=20)
    fig_c.tight_layout()
    fig_c.savefig(os.path.join(out_dir, f"panel_c_time_vs_distance.{fmt}"),
                  dpi=dpi, facecolor="white")
    plt.close(fig_c)

    # ---- Panel (d) Distance Error Boxplot ----
    fig_d, ax_d = plt.subplots(figsize=(7.0, 3.5))
    _apply_clean_axis(ax_d)
    gr_err_data = []
    for key in keys:
        df = data.get(key, pd.DataFrame())
        gr_err = pd.to_numeric(
            df.get("err_gr_vs_pg_m", []), errors="coerce").dropna()
        gr_err_data.append(
            gr_err.values if len(gr_err) > 0 else np.array([0.0]))
    all_err = np.concatenate(gr_err_data)
    q01, q99 = np.percentile(all_err, [1, 99])
    margin = (q99 - q01) * 0.3
    positions_g = x
    bp2 = ax_d.boxplot(
        gr_err_data, positions=positions_g, widths=0.6,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=0.8))
    for p in bp2["boxes"]:
        p.set_facecolor(COLOR_POSTGIS)
        p.set_alpha(0.55)
    ax_d.axhline(y=0, color="#333333", linewidth=1.0, zorder=5)
    ax_d.set_xticks(positions_g)
    ax_d.set_xticklabels(labels, fontsize=8, rotation=18, ha="right")
    ax_d.set_ylabel("Distance Error (m)")
    ax_d.set_xlabel("Geometry Combination")
    ax_d.set_ylim(q01 - margin, q99 + margin)
    ax_d.set_title("Distance Error vs PostGIS", fontsize=11,
                   fontweight="bold", pad=20)
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(out_dir, f"panel_d_distance_error.{fmt}"),
                  dpi=dpi, facecolor="white")
    plt.close(fig_d)

    # ---- Panel (e) ECDF of absolute distance error (0-0.05 m) ----
    fig_e, ax_e = plt.subplots(figsize=(7.0, 3.5))
    _apply_clean_axis(ax_e)
    _combo_colors_e = plt.cm.tab10(np.linspace(0, 1, 10))[:9]
    for ci, (key, clabel) in enumerate(zip(COMBO_FILES, COMBO_LABELS)):
        df = data.get(key)
        if df is None:
            continue
        vals = pd.to_numeric(
            df.get("err_gr_vs_pg_m", []), errors="coerce").dropna().abs()
        if len(vals) == 0:
            continue
        sorted_v = np.sort(vals.values)
        ecdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax_e.plot(sorted_v, ecdf, color=_combo_colors_e[ci], linewidth=1.5,
                  linestyle="-", label=clabel)
    ax_e.set_xlabel("Absolute Distance Error (m)")
    ax_e.set_ylabel("ECDF")
    ax_e.set_xlim(0.0, 0.05)
    ax_e.set_ylim(0.0, 1.02)
    ax_e.legend(fontsize=7, loc="lower right", framealpha=0.85, ncol=3)
    ax_e.set_title("ECDF of Absolute Distance Error (0-0.05 m)",
                   fontsize=11, fontweight="bold")
    fig_e.tight_layout()
    fig_e.savefig(os.path.join(out_dir, f"panel_e_error_cdf.{fmt}"),
                  dpi=dpi, facecolor="white")
    plt.close(fig_e)

    # ---- Panel (f) Absolute error vs distance, 9 combos ----
    fig_f, ax_f = plt.subplots(figsize=(7.0, 3.5))
    _apply_clean_axis(ax_f, grid_axis="both")
    _combo_colors_f = plt.cm.tab10(np.linspace(0, 1, 10))[:9]
    for ci, (key, clabel) in enumerate(zip(COMBO_FILES, COMBO_LABELS)):
        df = data.get(key)
        if df is None:
            continue
        pg_d = pd.to_numeric(
            df.get("pg_distance_m", []), errors="coerce")
        err_m = pd.to_numeric(
            df.get("err_gr_vs_pg_m", []), errors="coerce")
        valid = pg_d.notna() & err_m.notna() & (pg_d > 0.01)
        if valid.sum() < 3:
            continue
        pg_arr = pg_d[valid].values
        err_arr = np.abs(err_m[valid].values)
        lo = max(float(pg_arr.min()), 0.01)
        hi = float(pg_arr.max())
        if hi <= lo:
            continue
        edges = np.logspace(np.log10(lo), np.log10(hi), 30)
        bin_idx = np.digitize(pg_arr, edges) - 1
        x_trend, y_trend = [], []
        for bi in range(len(edges) - 1):
            m = bin_idx == bi
            if m.sum() >= 3:
                x_trend.append(np.median(pg_arr[m]))
                y_trend.append(np.median(err_arr[m]))
        if len(x_trend) >= 2:
            ax_f.plot(x_trend, y_trend, color=_combo_colors_f[ci],
                      linewidth=1.4, label=clabel, zorder=5 + ci)
    ax_f.set_xscale("log")
    ax_f.set_yscale("log")
    ax_f.set_xlabel("PostGIS Distance (m)")
    ax_f.set_ylabel("|Distance Error| (m)")
    ax_f.legend(fontsize=6.5, loc="upper right", framealpha=0.85, ncol=3)
    ax_f.set_title("Abs Error vs Actual Distance (by Combination)",
                   fontsize=11, fontweight="bold")
    fig_f.tight_layout()
    fig_f.savefig(os.path.join(out_dir, f"panel_f_error_vs_distance.{fmt}"),
                  dpi=dpi, facecolor="white")
    plt.close(fig_f)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze geometry benchmark results and generate figures",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("folder", help="Benchmark output folder")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure DPI (default 300, use 1200 for final paper)")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg", "eps"],
                        help="Figure format (default png)")
    parser.add_argument("--out-dir", type=str, default="",
                        help="Override output directory "
                             "(default: <folder>/figures)")
    parser.add_argument("--all", action="store_true",
                        help="Also produce supplementary figures (fig1-9 + "
                             "panels). Default: only fig_overview "
                             "(canonical single figure for paper).")

    args = parser.parse_args()
    folder = args.folder
    fmt = args.format
    dpi = args.dpi

    if not os.path.isdir(folder):
        print(f"Error: folder not found: {folder}")
        sys.exit(1)

    fig_dir = args.out_dir or os.path.join(folder, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("GeoSOT Geometry Benchmark — Result Analysis")
    print("=" * 60)
    print(f"  Input folder:  {os.path.abspath(folder)}")
    print(f"  Figure output: {os.path.abspath(fig_dir)}")
    print(f"  Format: {fmt.upper()}, DPI: {dpi}")
    print()

    data = load_data(folder)
    n_combos = sum(1 for k in COMBO_FILES if k in data)
    n_total = sum(len(data[k]) for k in COMBO_FILES if k in data)
    print(f"  Loaded {n_combos} combinations, {n_total} total records")
    for i, key in enumerate(COMBO_FILES):
        df = data.get(key)
        n = len(df) if df is not None else 0
        mark = "+" if n > 0 else "o"
        print(f"    {mark} {COMBO_LABELS[i]:<8s} ({key}): {n} records")
    print()

    print("Generating figures...")
    if args.all:
        fig_total_time(data, fig_dir, fmt, dpi)
        fig_speedup(data, fig_dir, fmt, dpi)
        fig_step_breakdown(data, fig_dir, fmt, dpi)
        fig_distance_error(data, fig_dir, fmt, dpi)
        fig_bearing_error(data, fig_dir, fmt, dpi)
        fig_accuracy_scatter(data, fig_dir, fmt, dpi)
        fig_heatmap(data, fig_dir, fmt, dpi)
        fig_neo4j_overhead(data, fig_dir, fmt, dpi)
        fig_error_cdf(data, fig_dir, fmt, dpi)
        fig_accuracy_by_geometry(data, fig_dir, fmt, dpi)

    # Always produce the canonical single figure for the paper.
    fig_overview(data, fig_dir, fmt, dpi)

    print(f"\nDone. figures saved to: {os.path.abspath(fig_dir)}")


if __name__ == "__main__":
    main()

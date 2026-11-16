#!/usr/bin/env python3
# coding: utf-8
"""
G3R KNN — SCI-quality publication figures.

Reads the CSV produced by ``geosot_knn_v3_1.py`` (e.g. ``knn_v31_100k.csv``)
and produces 6 compact, non-redundant figures covering:

    Fig S  Overall performance summary      (single 2x3 figure for the paper)
    Fig 1  Accuracy overview                (Recall@K violin  +  perfect-recall bar)
    Fig 2  Query latency                    (CDF of G3R vs PG  +  per-query speedup)
    Fig 3  Top-1 distance fidelity          (G3R-vs-PG scatter  +  |error| histogram)
    Fig 4  Adaptive grid internals          (safe level  +  prefixes  +  candidates)
    Fig 5  Recall vs PG top-1 distance      (binned lines per K  +  query counts)
    Fig 6  Efficiency-accuracy frontier     (hexbin: latency x Recall@20)

CSV columns expected
--------------------
    query_code, k, recall_at_k, top1_match, pg_std_agree,
    pgs_top1_dist_m, gr_top1_dist_m, top1_err_m, top1_err_pct,
    gr_total_candidates, gr_safe_level, gr_final_level, gr_prefixes,
    t_gr_us, t_pg_us, t_pgs_us, gr_top_k, pgs_top_k

Usage
-----
    python plot_knn_v3_sci.py --csv knn_v3_351020_v2.csv
    python plot_knn_v3_sci.py --csv knn_v3_351020_v2.csv --dpi 600 --format pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

from matplotlib.transforms import Bbox
from paper_figure_style import export_consistent_panels


def _export_axes_as_standalone(fig, axes_dict, out_dir, fmt, dpi,
                               name_prefix, strip_panel_label=True,
                               pad_inches=0.20):
    """
    Crop composite axes to standalone PNG/PDF panels for LaTeX. axes_dict may map labels to a
    single axis or an axis sequence; use the union bounding box for multipart panels.
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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from scipy import stats
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ── Global SCI style ─────────────────────────────────────────────────────────

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
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Unified palette aligned with multiscale_analysis / reverse_query_analysis:
#   GR (G3R)  → deep blue (#1565C0)
#   PG        → mid-gray  (#9E9E9E)
#   accent    → orange    (#FB8C00) for highlights / annotations
GR_COLOR = "#1565C0"        # G3R     (deep blue)
PG_COLOR = "#9E9E9E"        # PostGIS (mid-gray)
ACCENT   = "#FB8C00"        # neutral highlight (orange)
K_PALETTE = ["#1565C0", "#43A047", "#FB8C00", "#E53935"]  # K=3/5/10/20

# Resolved by main() based on --max-dist; consumed by fig5/fig_summary panel(e).
# None  → use the 99.9 percentile of the distance distribution.
# float → use this absolute meter value as the linear x-axis upper bound.
_X_MAX_OVERRIDE: Optional[float] = None
_X_MAX_PCT_DEFAULT: float = 99.9


# ── Helpers ──────────────────────────────────────────────────────────────────

def _annotate_stats(ax, text: str, loc: str = "upper right"):
    """Boxed text annotation (upper-right by default)."""
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    x = 0.97 if ha == "right" else 0.03
    y = 0.97 if va == "top" else 0.03
    ax.text(x, y, text,
            transform=ax.transAxes, ha=ha, va=va, fontsize=7,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92,
                      ec="#bbbbbb", lw=0.5))


def _downsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _fmt_ms(us: np.ndarray) -> np.ndarray:
    """microseconds -> milliseconds."""
    return us.astype(np.float64) / 1e3


def _resolve_xmax(d_pos: np.ndarray) -> Tuple[float, str]:
    """Pick the linear x-axis upper bound for distance plots.

    Returns (xmax, label_suffix).  Uses the module-level override if set,
    otherwise the 99.9 percentile (so a handful of long-tail queries don't
    leave the plot blank on the right).
    """
    if _X_MAX_OVERRIDE is not None and not np.isinf(_X_MAX_OVERRIDE):
        return float(_X_MAX_OVERRIDE), f", capped @ {_X_MAX_OVERRIDE:g} m"
    if len(d_pos) == 0:
        return 1.0, ""
    if _X_MAX_OVERRIDE is not None and np.isinf(_X_MAX_OVERRIDE):
        return float(np.max(d_pos)), ", full range"
    pct = float(np.percentile(d_pos, _X_MAX_PCT_DEFAULT))
    return pct, f", 0 → {_X_MAX_PCT_DEFAULT:g} pct"


def _recall_curve(
    d: np.ndarray,
    r: np.ndarray,
    xmin: float = 0.0,
    xmax: Optional[float] = None,
    n_bins: int = 200,
    min_per_bin: int = 1,
    smooth_win: int = 11,
    min_eff_count: int = 30,
    log_x: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean recall as a continuous function of PG top-1 distance.

    ``log_x=False`` (default): bins are equally spaced on a *linear* axis
    from ``xmin`` to ``xmax``; ``centers`` are bin midpoints.
    ``log_x=True``: bins are log-spaced; ``centers`` are geometric means.

    Smoothing is **count-weighted** over a window of width ``smooth_win``:
    we accumulate Σ recall and Σ samples across the window and divide.
    Output bins where the window's total sample count falls below
    ``min_eff_count`` are returned as NaN -- so sparse-tail noise (e.g.
    a single-query bin spiking the curve down to 0.97) is suppressed
    cleanly instead of producing isolated stubs.
    """
    d = np.asarray(d, dtype=float)
    r = np.asarray(r, dtype=float)
    ok = np.isfinite(d) & np.isfinite(r)
    if log_x:
        ok &= (d > 0)
    else:
        ok &= (d >= 0)
    d, r = d[ok], r[ok]
    if len(d) == 0:
        z = np.zeros(n_bins)
        return z, np.full(n_bins, np.nan), z.astype(int)

    if xmax is None:
        xmax = float(np.max(d))

    if log_x:
        if xmin <= 0:
            xmin = max(float(np.min(d[d > 0])) * 0.9, 1e-3)
        edges = np.logspace(np.log10(xmin), np.log10(xmax), n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(xmin, xmax, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    sum_r  = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(sum_r,  idx, r)
    np.add.at(counts, idx, 1)

    # Count-weighted sliding window: aggregate raw samples, not bin means.
    if smooth_win is None or smooth_win < 1:
        smooth_win = 1
    half = smooth_win // 2
    csum_r = np.concatenate(([0.0], np.cumsum(sum_r)))
    csum_n = np.concatenate(([0],   np.cumsum(counts)))
    means  = np.full(n_bins, np.nan)
    eff_n  = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        lo, hi = max(0, i - half), min(n_bins, i + half + 1)
        win_r = csum_r[hi] - csum_r[lo]
        win_n = int(csum_n[hi] - csum_n[lo])
        eff_n[i] = win_n
        # The bin must itself contain >= min_per_bin samples *and* the
        # smoothing window must contain >= min_eff_count samples.
        if counts[i] >= min_per_bin and win_n >= min_eff_count:
            means[i] = win_r / win_n

    return centers, means, counts


def _recall_curve_quantile(
    d: np.ndarray,
    r: np.ndarray,
    xmin: float = 0.0,
    xmax: Optional[float] = None,
    n_bins: int = 24,
    min_per_bin: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recall-vs-distance with equal-sample (quantile) bins.

    This avoids sparse-tail artifacts from fixed-width bins. Returns
    (centers, means, lo95, hi95, counts), where lo95/hi95 are normal-approx
    confidence bounds and are NaN if a bin has too few samples.
    """
    d = np.asarray(d, dtype=float)
    r = np.asarray(r, dtype=float)
    ok = np.isfinite(d) & np.isfinite(r) & (d >= xmin)
    if xmax is not None:
        ok &= (d <= xmax)
    d, r = d[ok], r[ok]
    if len(d) == 0:
        z = np.zeros(0, dtype=float)
        return z, z, z, z, np.zeros(0, dtype=int)

    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(d, q)
    edges = np.unique(edges)
    if len(edges) < 2:
        z = np.zeros(0, dtype=float)
        return z, z, z, z, np.zeros(0, dtype=int)

    nb = len(edges) - 1
    idx = np.clip(np.digitize(d, edges, right=False) - 1, 0, nb - 1)
    centers = np.full(nb, np.nan, dtype=float)
    means = np.full(nb, np.nan, dtype=float)
    lo95 = np.full(nb, np.nan, dtype=float)
    hi95 = np.full(nb, np.nan, dtype=float)
    counts = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = (idx == b)
        n = int(np.sum(m))
        counts[b] = n
        if n > 0:
            centers[b] = float(np.median(d[m]))
        if n < min_per_bin:
            continue
        rv = r[m]
        mu = float(np.mean(rv))
        sd = float(np.std(rv, ddof=1)) if n > 1 else 0.0
        se = sd / max(np.sqrt(n), 1.0)
        delta = 1.96 * se
        means[b] = mu
        lo95[b] = max(0.0, mu - delta)
        hi95[b] = min(1.0, mu + delta)
    return centers, means, lo95, hi95, counts


def _smooth_xy_by_index(
    x: np.ndarray,
    y: np.ndarray,
    win: int = 5,
) -> np.ndarray:
    """NaN-aware moving average on y (index domain), keep x unchanged."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0 or win <= 1:
        return y.copy()
    half = win // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        v = y[lo:hi]
        v = v[np.isfinite(v)]
        if len(v) > 0:
            out[i] = float(np.mean(v))
    return out


# ── Fig 1: Accuracy Overview ─────────────────────────────────────────────────

def fig1_accuracy(df: pd.DataFrame, out: str, fmt: str, dpi: int):
    k_values = sorted(df["k"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) Recall@K violin -----------------------------------------------------
    ax = axes[0]
    data = [df.loc[df["k"] == k, "recall_at_k"].values for k in k_values]
    parts = ax.violinplot(data, positions=range(len(k_values)),
                          showmedians=False, showextrema=False, widths=0.75)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(K_PALETTE[i % len(K_PALETTE)])
        pc.set_edgecolor("#333333")
        pc.set_alpha(0.75)
        pc.set_linewidth(0.5)

    # Overlay mean ± IQR as simple markers
    for i, arr in enumerate(data):
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        ax.vlines(i, q1, q3, color="black", linewidth=2.0, zorder=5)
        ax.plot(i, med, "o", color="white", markeredgecolor="black",
                markersize=4, zorder=6)
        ax.plot(i, arr.mean(), "D", color="#ffdd00",
                markeredgecolor="black", markersize=3.2, zorder=6)

    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_ylabel("Recall@K")
    # Zoom: show the meaningful range while keeping the 1.0 ceiling visible.
    y_low = max(0.5, float(np.min([np.min(a) for a in data])) - 0.02)
    ax.set_ylim(y_low, 1.012)
    ax.set_title("(a) Recall@K Distribution", fontweight="bold")
    ax.axhline(1.0, color="#999999", linewidth=0.5, linestyle="--", zorder=0)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    # Annotate mean above each violin
    means = [np.mean(a) for a in data]
    for i, m in enumerate(means):
        ax.text(i, 1.005, f"{m:.4f}", ha="center", va="bottom",
                fontsize=7, family="monospace", color="#333333")

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=5, label="median"),
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="#ffdd00", markeredgecolor="black",
                   markersize=4, label="mean"),
        plt.Line2D([0], [0], color="black", lw=2, label="IQR"),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              frameon=True, edgecolor="#cccccc", fancybox=False, fontsize=7)

    # (b) Perfect recall + Top-1 match ---------------------------------------
    ax = axes[1]
    perfect = [(df.loc[df["k"] == k, "recall_at_k"] >= 0.999999).mean() * 100
               for k in k_values]
    top1 = [df.loc[df["k"] == k, "top1_match"].astype(float).mean() * 100
            for k in k_values]

    x = np.arange(len(k_values))
    w = 0.38
    b1 = ax.bar(x - w / 2, perfect, w, label="Perfect Recall (=1.0)",
                color=GR_COLOR, edgecolor="white", linewidth=0.4, zorder=3)
    b2 = ax.bar(x + w / 2, top1, w, label="Top-1 Match",
                color=PG_COLOR, edgecolor="white", linewidth=0.4, zorder=3)

    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 0.4,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_ylabel("Percentage of queries (%)")
    ymax = max(max(perfect), max(top1))
    # Leave headroom for legend and bar labels without overlap
    ax.set_ylim(0, ymax * 1.32)
    ax.set_title("(b) Perfect-Recall & Top-1 Match Rates", fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc",
              fancybox=False, ncol=1)

    fig.tight_layout()
    path = f"{out}_fig1_accuracy.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig 1] {path}")


# ── Fig 2: Query Latency Comparison ──────────────────────────────────────────

def fig2_latency(df: pd.DataFrame, out: str, fmt: str, dpi: int):
    # G3R is called once per query (k_max=20) → dedupe by query_code
    uniq = df.drop_duplicates(subset="query_code").copy()
    gr_ms  = _fmt_ms(uniq["t_gr_us"].values)
    pgs_ms = _fmt_ms(uniq["t_pgs_us"].values)
    gr_ms  = gr_ms[gr_ms  > 0]
    pgs_ms = pgs_ms[pgs_ms > 0]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) Empirical CDF -------------------------------------------------------
    ax = axes[0]
    for arr, color, lbl in [
        (gr_ms,  GR_COLOR, "G3R"),
        (pgs_ms, PG_COLOR, "PostGIS (GiST-KNN)"),
    ]:
        sarr = np.sort(arr)
        y = np.arange(1, len(sarr) + 1) / len(sarr)
        ax.plot(sarr, y, color=color, linewidth=1.3, label=lbl)

    ax.set_xscale("log")
    ax.set_xlabel("Per-query latency (ms, log scale)")
    ax.set_ylabel("Cumulative fraction of queries")
    ax.set_title("(a) Latency CDF", fontweight="bold")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.02)

    # Mark percentiles
    for arr, color in [(gr_ms, GR_COLOR), (pgs_ms, PG_COLOR)]:
        p50 = np.percentile(arr, 50)
        p99 = np.percentile(arr, 99)
        ax.axvline(p50, color=color, linewidth=0.5, linestyle=":",
                   alpha=0.7)
        ax.axvline(p99, color=color, linewidth=0.5, linestyle="--",
                   alpha=0.7)

    stats_text = (
        f"{'':10s}{'G3R':>10s}{'PGs':>10s}\n"
        f"{'median':10s}{np.median(gr_ms):>9.1f} {np.median(pgs_ms):>9.1f}\n"
        f"{'mean':10s}{gr_ms.mean():>9.1f} {pgs_ms.mean():>9.1f}\n"
        f"{'P99':10s}{np.percentile(gr_ms,99):>9.1f} "
        f"{np.percentile(pgs_ms,99):>9.1f}"
    )
    _annotate_stats(ax, stats_text, loc="lower right")
    ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc",
              fancybox=False)

    # (b) Per-query speedup ---------------------------------------------------
    ax = axes[1]
    n = min(len(gr_ms), len(pgs_ms))
    sp = pgs_ms[:n] / np.maximum(gr_ms[:n], 1e-9)
    sp = sp[np.isfinite(sp) & (sp > 0)]

    # Log-space bins
    bins = np.logspace(np.log10(max(sp.min(), 1e-3)),
                       np.log10(sp.max()), 60)
    ax.hist(sp, bins=bins, color=GR_COLOR, alpha=0.85,
            edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Speedup factor  (PostGIS / G3R)")
    ax.set_ylabel("Number of queries")
    ax.set_title("(b) Per-query Speedup Distribution", fontweight="bold")
    ax.axvline(1.0, color=ACCENT, linestyle="--", linewidth=0.7, zorder=4)
    ax.axvline(np.median(sp), color=PG_COLOR, linestyle="-",
               linewidth=0.9, zorder=4)
    ax.axvline(sp.mean(), color="#000000", linestyle=":",
               linewidth=0.9, zorder=4)
    frac_faster = (sp > 1.0).mean() * 100
    _annotate_stats(ax,
                    f"median   = {np.median(sp):7.2f}x\n"
                    f"mean     = {sp.mean():7.2f}x\n"
                    f"P90      = {np.percentile(sp,90):7.2f}x\n"
                    f"P99      = {np.percentile(sp,99):7.2f}x\n"
                    f">PG:      {frac_faster:6.1f}% queries",
                    loc="upper left")
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = f"{out}_fig2_latency.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig 2] {path}")


# ── Fig 3: Top-1 Distance Fidelity ───────────────────────────────────────────

def fig3_distance_error(df: pd.DataFrame, out: str, fmt: str, dpi: int):
    uniq = df.drop_duplicates(subset="query_code").copy()
    uniq = uniq[(uniq["pgs_top1_dist_m"] >= 0) &
                (uniq["gr_top1_dist_m"]  >= 0)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) G3R vs PG scatter ---------------------------------------------------
    ax = axes[0]
    sample = _downsample(uniq, n=8000)
    x = sample["pgs_top1_dist_m"].values
    y = sample["gr_top1_dist_m"].values
    # Use log-log to cover wide range, shift 1e-2 for zeros
    eps = 1e-2
    ax.scatter(x + eps, y + eps, s=1.2, alpha=0.25, color=ACCENT,
               edgecolors="none", rasterized=True)
    hi = max(x.max(), y.max()) + eps
    lo = eps
    ax.plot([lo, hi], [lo, hi], "--", color="#999999", linewidth=0.8,
            zorder=0, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("PostGIS Top-1 distance (m)")
    ax.set_ylabel("G3R Top-1 distance (m)")
    ax.set_title("(a) Top-1 Distance Fidelity", fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="both", length=3)

    xx = uniq["pgs_top1_dist_m"].values
    yy = uniq["gr_top1_dist_m"].values
    rmse = float(np.sqrt(np.mean((xx - yy) ** 2)))
    mae  = float(np.mean(np.abs(xx - yy)))
    if _HAVE_SCIPY and len(xx) > 2:
        r, _ = stats.pearsonr(xx, yy)
    else:
        r = float(np.corrcoef(xx, yy)[0, 1]) if len(xx) > 2 else float("nan")
    _annotate_stats(ax,
                    f"n    = {len(uniq):,}\n"
                    f"r    = {r:.6f}\n"
                    f"MAE  = {mae:.4f} m\n"
                    f"RMSE = {rmse:.4f} m",
                    loc="upper left")
    ax.legend(loc="lower right", frameon=True, edgecolor="#cccccc",
              fancybox=False, fontsize=7)

    # (b) |Error| histogram ---------------------------------------------------
    ax = axes[1]
    err = np.abs(uniq["top1_err_m"].values)
    err = err[np.isfinite(err)]
    # Bucket: 0 exact-match as a separate bar
    exact = (err < 1e-4).sum()
    non_exact = err[err >= 1e-4]

    if len(non_exact) > 0:
        bins = np.logspace(np.log10(max(non_exact.min(), 1e-4)),
                           np.log10(non_exact.max()), 60)
        ax.hist(non_exact, bins=bins, color=PG_COLOR, alpha=0.85,
                edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|Top-1 distance error| (m, log scale)")
    ax.set_ylabel("Number of queries (log)")
    ax.set_title("(b) Top-1 Distance Error Distribution", fontweight="bold")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    mean_e = float(np.mean(err))
    med_e  = float(np.median(err))
    p99_e  = float(np.percentile(err, 99))
    max_e  = float(np.max(err))
    exact_pct = 100.0 * exact / len(err)

    # Draw percentile lines without separate legend (annotate directly)
    for v, c in [(med_e, "#000000"), (mean_e, GR_COLOR),
                 (p99_e, PG_COLOR)]:
        if v > 0:
            ax.axvline(v, color=c, linewidth=0.7, linestyle=":", alpha=0.85)

    _annotate_stats(ax,
                    f"exact (<1e-4): {exact_pct:5.1f}%\n"
                    f"median       : {med_e:8.4f} m   (···)\n"
                    f"mean         : {mean_e:8.4f} m   (···)\n"
                    f"P99          : {p99_e:8.4f} m   (···)\n"
                    f"max          : {max_e:8.3f} m",
                    loc="upper right")

    fig.tight_layout()
    path = f"{out}_fig3_distance_error.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig 3] {path}")


# ── Fig 4: Adaptive-grid internals ───────────────────────────────────────────

def fig4_grid_internals(df: pd.DataFrame, out: str, fmt: str, dpi: int):
    uniq = df.drop_duplicates(subset="query_code").copy()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))

    # (a) Safe level ----------------------------------------------------------
    ax = axes[0]
    safe = uniq["gr_safe_level"].astype(int).values
    finl = uniq["gr_final_level"].astype(int).values
    levels = np.arange(safe.min(), safe.max() + 1)
    counts_s = [(safe == L).sum() for L in levels]
    counts_f = [(finl == L).sum() for L in levels]

    w = 0.4
    x = np.arange(len(levels))
    ax.bar(x - w / 2, counts_s, w, label="safe level",
           color=GR_COLOR, edgecolor="white", linewidth=0.3, zorder=3)
    ax.bar(x + w / 2, counts_f, w, label="final level",
           color=PG_COLOR, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in levels], fontsize=7.5)
    ax.set_xlabel("GeoSOT level")
    ax.set_ylabel("Number of queries")
    ax.set_title("(a) Adaptive Level Selection", fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, edgecolor="#cccccc", fancybox=False, fontsize=7)
    _annotate_stats(ax,
                    f"safe  median = {int(np.median(safe))}\n"
                    f"final median = {int(np.median(finl))}",
                    loc="upper left")

    # (b) Prefixes queried ----------------------------------------------------
    ax = axes[1]
    pref = uniq["gr_prefixes"].astype(int).values
    pmin, pmax = int(pref.min()), int(pref.max())
    bins = np.arange(pmin, pmax + 2) - 0.5
    ax.hist(pref, bins=bins, color=GR_COLOR, alpha=0.85,
            edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Prefixes queried per call")
    ax.set_ylabel("Number of queries (log)")
    ax.set_title("(b) Query Prefix Fan-out", fontweight="bold")
    ax.grid(axis="y", which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    _annotate_stats(ax,
                    f"median = {int(np.median(pref))}\n"
                    f"P90    = {int(np.percentile(pref,90))}\n"
                    f"P99    = {int(np.percentile(pref,99))}\n"
                    f"max    = {pmax}",
                    loc="upper right")

    # (c) Candidates per query ------------------------------------------------
    ax = axes[2]
    cand = uniq["gr_total_candidates"].astype(int).values
    cand_nz = cand[cand > 0]
    bins = np.logspace(np.log10(cand_nz.min()),
                       np.log10(cand_nz.max()), 50)
    ax.hist(cand_nz, bins=bins, color=ACCENT, alpha=0.85,
            edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Candidates per query (log)")
    ax.set_ylabel("Number of queries")
    ax.set_title("(c) Candidate Pool Size", fontweight="bold")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    _annotate_stats(ax,
                    f"median = {int(np.median(cand_nz)):,}\n"
                    f"mean   = {int(cand_nz.mean()):,}\n"
                    f"P99    = {int(np.percentile(cand_nz,99)):,}\n"
                    f"max    = {int(cand_nz.max()):,}",
                    loc="upper right")

    fig.tight_layout()
    path = f"{out}_fig4_grid_internals.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig 4] {path}")


# ── Fig 5: Recall vs PG Top-1 Distance ───────────────────────────────────────

def fig5_recall_vs_distance(df: pd.DataFrame, out: str, fmt: str, dpi: int):
    """Per-query recall vs PostGIS Top-1 distance, scatter + trend per K.

    Top panel: every query is plotted as a translucent dot (one colour per
    K).  Because Recall@K is quantised to multiples of 1/K, the dots
    naturally form horizontal bands -- the upper band at y=1.0 holds the
    overwhelming majority of queries, while the lower bands directly show
    *where* (in distance) the failures concentrate.  A heavier
    count-weighted smoothing curve is overlaid as the trend per K.

    Bottom panel: query density (log-count) over the same linear x-axis.
    """
    k_values = sorted(df["k"].unique().tolist())

    # Shared [0, xmax] across all K. xmax: 99.9-percentile by default so the
    # handful of >1 km long-tail queries don't leave the right half blank.
    d_all = df["pgs_top1_dist_m"].values
    d_pos = d_all[np.isfinite(d_all) & (d_all >= 0)]
    xmin  = 0.0
    xmax, cap_lbl = _resolve_xmax(d_pos)
    n_bins = 80

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.8, 1.0]})
    ax_top, ax_bot = axes

    # Plot in reverse-K order so K=3 (most-likely-to-fail) sits on top of
    # the K=20 layer; legend handles are added separately to keep K=3..20
    # left-to-right reading order.
    legend_handles = []
    for i, k in enumerate(reversed(k_values)):
        idx = list(k_values).index(k)
        sub = df[df["k"] == k]
        d = sub["pgs_top1_dist_m"].values
        r = sub["recall_at_k"].values
        m = np.isfinite(d) & np.isfinite(r) & (d >= xmin) & (d <= xmax)
        d, r = d[m], r[m]

        c = K_PALETTE[idx % len(K_PALETTE)]
        # Per-query scatter -- alpha is the key knob: at 99 k points per K
        # we want individual outliers visible but the y=1.0 band readable.
        ax_top.scatter(d, r, s=3.2, color=c, alpha=0.05,
                       edgecolors="none", rasterized=True, zorder=2)

    # Trend curves on top of all scatter layers (quantile bins + 95% CI).
    x_right_eff = xmin
    for i, k in enumerate(k_values):
        sub = df[df["k"] == k]
        _mpb = max(50, len(sub) // (24 * 2))
        centers, means, lo95, hi95, _ = _recall_curve_quantile(
            sub["pgs_top1_dist_m"].values,
            sub["recall_at_k"].values,
            xmin=xmin, xmax=xmax, n_bins=24, min_per_bin=_mpb,
        )
        c = K_PALETTE[i % len(K_PALETTE)]
        ax_top.fill_between(centers, lo95, hi95, color=c, alpha=0.05,
                            linewidth=0.0, zorder=3)
        line, = ax_top.plot(centers, means, "-", color=c, linewidth=1.8,
                            zorder=4, label=f"K = {k}")
        legend_handles.append(line)
        valid = np.isfinite(means)
        if np.any(valid):
            x_right_eff = max(x_right_eff, float(np.max(centers[valid])))

    if x_right_eff <= xmin:
        x_right_eff = xmax
    # Remove right blank area: clamp x-axis to last effective trend point.
    ax_top.set_xlim(xmin, x_right_eff * 1.02)

    ax_top.set_ylabel("Recall@K")
    ax_top.set_ylim(-0.03, 1.05)
    ax_top.set_xlim(xmin, xmax)
    ax_top.set_title("(a) Per-query Recall vs PostGIS Top-1 Distance  "
                     f"(scatter + trend, linear x{cap_lbl})",
                     fontweight="bold")
    ax_top.axhline(1.0, color="#999999", linewidth=0.5, linestyle="--",
                   zorder=1)
    ax_top.grid(True, linewidth=0.3, alpha=0.4)
    ax_top.set_axisbelow(True)
    leg = ax_top.legend(handles=legend_handles, ncol=len(k_values),
                        frameon=True, edgecolor="#cccccc",
                        fancybox=False, loc="lower left",
                        title="trend (quantile bins, 95% CI)")
    leg.get_title().set_fontsize(7.5)

    # ── Query-count density (bottom), shared linear-x with top panel ───────
    sub0 = df[df["k"] == k_values[0]]
    d0 = sub0["pgs_top1_dist_m"].values
    d0 = d0[np.isfinite(d0) & (d0 >= 0) & (d0 <= x_right_eff * 1.02)]
    edges = np.linspace(xmin, x_right_eff * 1.02, n_bins + 1)
    counts0, _ = np.histogram(d0, bins=edges)
    centers_plot = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1] - edges[0]
    ax_bot.bar(centers_plot, counts0, width=widths,
               color=ACCENT, alpha=0.85,
               edgecolor="none", zorder=3)
    ax_bot.set_yscale("log")
    ax_bot.set_ylabel("Queries (log)")
    ax_bot.set_xlabel("PostGIS Top-1 distance (m)")
    ax_bot.grid(axis="y", which="both", linewidth=0.3, alpha=0.4)
    ax_bot.set_axisbelow(True)

    fig.tight_layout()
    path = f"{out}_fig5_recall_vs_distance.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig 5] {path}")


# ── Fig 6: Efficiency-Accuracy Frontier ──────────────────────────────────────

def fig6_frontier(df: pd.DataFrame, out: str, fmt: str, dpi: int):
    k_values = sorted(df["k"].unique().tolist())
    k_show = max(k_values)
    sub = df[df["k"] == k_show].copy()
    sub = sub[(sub["t_gr_us"] > 0) & (sub["recall_at_k"].notna())]
    gr_ms = _fmt_ms(sub["t_gr_us"].values)
    rec   = sub["recall_at_k"].values

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    # Hexbin density
    hb = ax.hexbin(gr_ms, rec, gridsize=55, xscale="log",
                   cmap="viridis", mincnt=1, bins="log",
                   edgecolors="none")
    cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("log10(query count)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Overlay decile trend (median recall per latency decile)
    order = np.argsort(gr_ms)
    g_sorted, r_sorted = gr_ms[order], rec[order]
    deciles = np.array_split(np.arange(len(g_sorted)), 10)
    xd = [np.median(g_sorted[d]) for d in deciles]
    yd = [np.mean(r_sorted[d])  for d in deciles]
    ax.plot(xd, yd, "o-", color="#ffdd00", markersize=5,
            markeredgecolor="black", markeredgewidth=0.6,
            linewidth=1.4, label="decile mean")

    ax.set_xlabel("Per-query latency (ms, log scale)")
    ax.set_ylabel(f"Recall@{k_show}")
    ax.set_title(f"Efficiency-Accuracy Frontier (K = {k_show})",
                 fontweight="bold")
    ax.set_ylim(-0.02, 1.04)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    _annotate_stats(ax,
                    f"n          = {len(sub):,}\n"
                    f"med lat    = {np.median(gr_ms):7.1f} ms\n"
                    f"mean R@{k_show:<2d}  = {rec.mean():7.4f}\n"
                    f"%R=1.0     = {(rec >= 0.999999).mean()*100:6.2f}%",
                    loc="lower right")
    ax.legend(loc="lower left", frameon=True, edgecolor="#cccccc",
              fancybox=False)

    fig.tight_layout()
    path = f"{out}_fig6_frontier.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [Fig 6] {path}")


# ── Summary Figure (single publication-ready figure) ────────────────────────

def fig_summary(df: pd.DataFrame, out: str, fmt: str, dpi: int):
    """Single 2x2 figure for a paper's main results section.

    The numerical accuracy table (Recall@K / Perfect@K / Top-1 match /
    MAE) is intended to live in the paper text, not in the figure;
    every panel here is a chart that adds information beyond raw numbers.

    Top row -- Accuracy:
        (a) Recall vs PostGIS Top-1 distance (trend per K)
        (b) Top-1 distance fidelity (G3R vs PostGIS scatter)

    Bottom row -- Efficiency / trade-off:
        (c) Latency CDF, G3R vs PostGIS
        (d) Efficiency-accuracy frontier (K = max, hexbin)
    """
    k_values = sorted(df["k"].unique().tolist())
    uniq = df.drop_duplicates(subset="query_code").copy()

    fig = plt.figure(figsize=(11.0, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                          left=0.075, right=0.985, top=0.95, bottom=0.085)

    # ╔════ ACCURACY ROW ═══════════════════════════════════════════════════╗
    # ── (a) Recall vs True Top-1 Distance (trend per K) ─────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    d_all = df["pgs_top1_dist_m"].values
    d_pos = d_all[np.isfinite(d_all) & (d_all >= 0)]
    xmin_a = 0.0
    xmax_a, cap_lbl_a = _resolve_xmax(d_pos)
    y_seen = []
    x_right_eff = xmin_a
    for i, k in enumerate(k_values):
        sub = df[df["k"] == k]
        # Adaptive min_per_bin: require half the expected len/n_bins observations,
        # so small samples (for example 10k per K) still form bins while sparse-tail noise is suppressed.
        _mpb = max(50, len(sub) // (24 * 2))
        centers, means, lo95, hi95, _ = _recall_curve_quantile(
            sub["pgs_top1_dist_m"].values,
            sub["recall_at_k"].values,
            xmin=xmin_a, xmax=xmax_a, n_bins=24, min_per_bin=_mpb,
        )
        # Make panel (a) uniformly smooth across the full distance range.
        means = _smooth_xy_by_index(centers, means, win=5)
        lo95 = _smooth_xy_by_index(centers, lo95, win=5)
        hi95 = _smooth_xy_by_index(centers, hi95, win=5)
        c = K_PALETTE[i % len(K_PALETTE)]
        ax_a.fill_between(centers, lo95, hi95, color=c, alpha=0.05,
                          linewidth=0.0, zorder=2)
        ax_a.plot(centers, means, "-", color=c, linewidth=1.9,
                  label=f"K = {k}", zorder=3)
        band = np.concatenate([
            np.asarray(lo95, dtype=float),
            np.asarray(hi95, dtype=float),
            np.asarray(means, dtype=float),
        ])
        band = band[np.isfinite(band)]
        if band.size:
            y_seen.append(band)
        valid = np.isfinite(means)
        if np.any(valid):
            x_right_eff = max(x_right_eff, float(np.max(centers[valid])))
    if x_right_eff <= xmin_a:
        x_right_eff = xmax_a
    # Remove right blank area: clamp x-axis to last effective trend point.
    ax_a.set_xlim(xmin_a, x_right_eff * 1.02)
    # Fit the axis to the recall band. A fixed floor at 0.90 pinned every
    # curve against the top edge.
    if y_seen:
        y_all = np.concatenate(y_seen)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        span = max(y_max - y_min, 0.0005)
        ax_a.set_ylim(y_min - 0.12 * span, y_max + 0.12 * span)
    else:
        ax_a.set_ylim(0.90, 1.001)
    ax_a.set_ylabel("Mean Recall@K")
    ax_a.set_xlabel(f"PostGIS Top-1 distance (m){cap_lbl_a}")
    ax_a.set_title("(a) Recall vs True Top-1 Distance (quantile bins)",
                   fontweight="bold")
    ax_a.axhline(1.0, color="#999999", linewidth=0.5, linestyle="--",
                 zorder=0)
    ax_a.grid(True, linewidth=0.3, alpha=0.4)
    ax_a.set_axisbelow(True)
    ax_a.legend(ncol=2, frameon=True, edgecolor="#cccccc",
                fancybox=False, fontsize=11, loc="lower right")

    # ── (b) Top-1 Distance Fidelity ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    valid = uniq[(uniq["pgs_top1_dist_m"] >= 0) &
                 (uniq["gr_top1_dist_m"]  >= 0)].copy()
    sample = _downsample(valid, 8000)
    eps = 1e-2
    ax_b.scatter(sample["pgs_top1_dist_m"].values + eps,
                 sample["gr_top1_dist_m"].values + eps,
                 s=1.4, alpha=0.28, color=ACCENT, edgecolors="none",
                 rasterized=True)
    xx = valid["pgs_top1_dist_m"].values
    yy = valid["gr_top1_dist_m"].values
    hi = max(float(np.max(xx)), float(np.max(yy))) + eps
    ax_b.plot([eps, hi], [eps, hi], "--", color="#888888", linewidth=0.8,
              zorder=0, label="y = x")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(eps, hi)
    ax_b.set_ylim(eps, hi)
    ax_b.set_xlabel("PostGIS Top-1 distance (m)")
    ax_b.set_ylabel("G3R Top-1 distance (m)")
    ax_b.set_title("(b) Top-1 Distance Fidelity", fontweight="bold")
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.grid(True, which="both", linewidth=0.3, alpha=0.35)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="upper left",
                ncol=1, frameon=True, edgecolor="#cccccc",
                fancybox=False, fontsize=11)

    # ╔════ EFFICIENCY ROW ═════════════════════════════════════════════════╗
    # ── (c) Latency CDF, G3R vs PostGIS ─────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    gr_ms  = _fmt_ms(uniq["t_gr_us"].values)
    pgs_ms = _fmt_ms(uniq["t_pgs_us"].values)
    gr_ms  = gr_ms[gr_ms  > 0]
    pgs_ms = pgs_ms[pgs_ms > 0]
    for arr, color, lbl in [(gr_ms, GR_COLOR, "G3R"),
                            (pgs_ms, PG_COLOR, "PostGIS")]:
        sarr = np.sort(arr)
        y = np.arange(1, len(sarr) + 1) / len(sarr)
        ax_c.plot(sarr, y, color=color, linewidth=1.6, label=lbl)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Per-query end-to-end latency (ms, log scale)")
    ax_c.set_ylabel("Cumulative fraction of queries")
    ax_c.set_title("(c) End-to-End Latency CDF  "
                   "(all stages incl. fetch & roundtrip)",
                   fontweight="bold")
    ax_c.set_ylim(0, 1.02)
    ax_c.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax_c.set_axisbelow(True)
    ax_c.legend(loc="lower right", ncol=1,
                frameon=True, edgecolor="#cccccc",
                fancybox=False, fontsize=11)

    # ── (d) Efficiency-Accuracy Frontier ────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    k_show = max(k_values)
    sub = df[df["k"] == k_show]
    sub = sub[(sub["t_gr_us"] > 0) & (sub["recall_at_k"].notna())]
    g = _fmt_ms(sub["t_gr_us"].values)
    rc = sub["recall_at_k"].values
    hb = ax_d.hexbin(g, rc, gridsize=42, xscale="log", cmap="viridis",
                     mincnt=1, bins="log", edgecolors="none")
    cb = fig.colorbar(hb, ax=ax_d, pad=0.02, shrink=0.92)
    cb.set_label("log10(query count)", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    # Decile mean recall vs latency, overlay as trend line.
    order = np.argsort(g)
    g_s, r_s = g[order], rc[order]
    deciles = np.array_split(np.arange(len(g_s)), 10)
    xd = [float(np.median(g_s[d]))   for d in deciles]
    yd = [float(np.mean(r_s[d]))     for d in deciles]
    ax_d.plot(xd, yd, "o-", color="#ffdd00", markersize=4,
              markeredgecolor="black", markeredgewidth=0.5,
              linewidth=1.2, label="decile mean")
    ax_d.set_xlabel("Per-query latency (ms, log scale)")
    ax_d.set_ylabel(f"Recall@{k_show}")
    ax_d.set_title(f"(d) Efficiency-Accuracy Frontier  (K = {k_show})",
                   fontweight="bold")
    ax_d.set_ylim(-0.02, 1.04)
    ax_d.grid(True, which="both", linewidth=0.3, alpha=0.35)
    ax_d.set_axisbelow(True)
    ax_d.legend(loc="lower left", frameon=True, edgecolor="#cccccc",
                fancybox=False, fontsize=11)

    path = f"{out}_fig_summary.{fmt}"
    fig.savefig(path, dpi=dpi)
    print(f"  [Fig S] {path}")

    panels_dir = os.path.join(os.path.dirname(out) or ".", "panels")
    os.makedirs(panels_dir, exist_ok=True)
    base = os.path.basename(out) or "knn"
    _export_axes_as_standalone(
        fig,
        axes_dict={"a": ax_a, "b": ax_b, "c": ax_c,
                   "d": (ax_d, cb.ax)},
        out_dir=panels_dir,
        fmt=fmt, dpi=dpi,
        name_prefix=f"{base}_summary",
    )
    print(f"  [Fig S] 4 standalone panels -> "
          f"{panels_dir}/{base}_summary_panel_*.{fmt}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCI publication figures for G3R KNN benchmark CSV")
    parser.add_argument("--csv", default="knn_v3_351020_v2.csv",
                        help="Input benchmark CSV (from geosot_knn_v3.py)")
    parser.add_argument("--output-prefix", default="",
                        help="Output filename prefix (default: derived from CSV)")
    parser.add_argument("--out-dir", default="",
                        help="Output directory; if set, basename of "
                             "--output-prefix (or CSV stem) is used inside this "
                             "directory.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (300 for draft, 1200 for final)")
    parser.add_argument("--format", default="png",
                        choices=["png", "pdf", "svg", "eps"],
                        help="Output format")
    parser.add_argument("--only", default="",
                        help="Comma-separated figure numbers to render (e.g. 1,3,6)")
    parser.add_argument("--max-dist", default="auto",
                        help="Linear x-axis cap for Fig 5 / Fig S panel (e). "
                             "'auto' = 99.9 percentile of PG Top-1 distance "
                             "(default; clips empty long tail), "
                             "'max' = full data max, "
                             "or a number in metres (e.g. 500).")
    args = parser.parse_args()

    global _X_MAX_OVERRIDE
    md = str(args.max_dist).strip().lower()
    if md in ("", "auto"):
        _X_MAX_OVERRIDE = None
    elif md == "max":
        _X_MAX_OVERRIDE = float("inf")  # sentinel; resolved later
    else:
        try:
            _X_MAX_OVERRIDE = float(md)
        except ValueError:
            print(f"[!] --max-dist must be 'auto', 'max', or a number; got {md!r}")
            sys.exit(1)

    if not os.path.isfile(args.csv):
        print(f"[!] CSV not found: {args.csv}")
        sys.exit(1)

    out = args.output_prefix or args.csv.rsplit(".", 1)[0]
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, os.path.basename(out))

    print(f"Loading {args.csv} ...")
    # Skip unused serialized list columns for speed/memory
    usecols = ["query_code", "k", "recall_at_k", "top1_match", "pg_std_agree",
               "pgs_top1_dist_m", "gr_top1_dist_m", "top1_err_m", "top1_err_pct",
               "gr_total_candidates", "gr_safe_level", "gr_final_level",
               "gr_prefixes", "t_gr_us", "t_pg_us", "t_pgs_us"]
    df = pd.read_csv(args.csv, usecols=lambda c: c in usecols,
                     low_memory=False)
    print(f"  {len(df):,} rows, {df['query_code'].nunique():,} unique queries, "
          f"K values = {sorted(df['k'].unique().tolist())}")

    # Coerce numeric (some fields may be blank strings)
    for c in ["recall_at_k", "pgs_top1_dist_m", "gr_top1_dist_m",
              "top1_err_m", "top1_err_pct",
              "gr_total_candidates", "gr_safe_level", "gr_final_level",
              "gr_prefixes", "t_gr_us", "t_pg_us", "t_pgs_us"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["k", "recall_at_k", "t_gr_us", "t_pgs_us"])
    df["k"] = df["k"].astype(int)

    fmt = args.format
    dpi = args.dpi
    print(f"  Output: {out}_fig*.{fmt} @ {dpi} DPI\n")

    # 0 = canonical summary figure (paper); 1..6 = supplementary
    if args.only:
        todo = [int(x) for x in args.only.split(",")
                if x.strip().lstrip("-").isdigit()]
    else:
        todo = [0]  # default: only the canonical single figure

    figs = {
        0: fig_summary,
        1: fig1_accuracy,
        2: fig2_latency,
        3: fig3_distance_error,
        4: fig4_grid_internals,
        5: fig5_recall_vs_distance,
        6: fig6_frontier,
    }
    for i in todo:
        if i in figs:
            figs[i](df, out, fmt, dpi)

    print(f"\nDone. All figures saved with prefix: {out}_fig*.{fmt}")


if __name__ == "__main__":
    main()

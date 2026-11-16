"""Consistent standalone panels for the IJDE second revision.

Only presentation properties are changed: canvas, fonts, spines, grid and
legend placement. The data artists and statistical calculations stay with
their original analysis scripts.
"""

from pathlib import Path
import json
import textwrap

import matplotlib as mpl
from matplotlib import ticker


FIGSIZE = (4.6, 3.6)
AXES_RECT = (0.22, 0.25, 0.73, 0.72)
LABEL_SIZE = 15
TICK_SIZE = 14
LEGEND_SIZE = 12

# Same font and frame everywhere; location follows the empty part of each
# panel. Letters here are source filenames (distance d/e become paper c/d).
PANEL_LEGENDS = {
    "fig_distance": {
        "a": ("upper left", 1),
        "b": ("lower right", 3),
        "d": None,
        "e": ("lower right", 3),
    },
    "fig_forward_overlay": {
        "a": ("upper center", 2),
        "b": ("lower right", 1),
        "c": ("upper right", 1),
    },
    "fig_reverse_overlay": {
        "a": ("upper center", 2),
        "b": ("lower right", 1),
        "c": ("lower right", 1),
    },
    "knn_v32_10k_summary": {
        "a": ("lower right", 2),
        "b": ("upper left", 1),
        "c": ("lower right", 1),
    },
}


def paper_legend(ax, *, loc, ncol=1, handles=None):
    """Put a compact, opaque white legend inside the plotting area."""
    kwargs = {} if handles is None else {"handles": handles}
    legend = ax.legend(
        **kwargs, loc=loc, ncol=ncol, fontsize=LEGEND_SIZE,
        frameon=True, fancybox=False, framealpha=1.0,
        facecolor="white", edgecolor="#BDBDBD",
        borderpad=0.35, borderaxespad=0.45,
        labelspacing=0.30, handlelength=1.0,
        handletextpad=0.35, columnspacing=0.40,
    )
    legend.get_frame().set_linewidth(0.6)
    legend.set_zorder(30)
    for label in legend.get_texts():
        label.set_fontfamily("Times New Roman")
    return legend


def _wrap_label(label, width=37):
    return "\n".join(textwrap.wrap(label.replace("\n", " "), width=width,
                                  break_long_words=False,
                                  break_on_hyphens=False))


def _style_panel(ax, prefix, letter):
    ax.set_title("")
    xlabel = ax.get_xlabel()
    if "PostGIS Top-1 distance (m), " in xlabel:
        xlabel = xlabel.replace("(m), ", "(m)\n", 1)
    elif xlabel == "Per-query end-to-end latency (ms, log scale)":
        xlabel = "Per-query end-to-end latency\n(ms, log scale)"
    elif "vs PostGIS ST_Azimuth" in xlabel:
        xlabel = xlabel.replace("vs PostGIS ST_Azimuth", "\nvs PostGIS ST_Azimuth").strip()
    else:
        xlabel = _wrap_label(xlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_wrap_label(ax.get_ylabel(), width=31))
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontsize(LABEL_SIZE)
        label.set_fontfamily("Times New Roman")
    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 6
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE,
                   direction="out", width=0.6, length=3.5, pad=3)
    ax.tick_params(axis="both", which="minor", direction="out",
                   width=0.4, length=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#9E9E9E")
        ax.spines[side].set_linewidth(0.6)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(False, which="both")
    category_panel = (prefix == "fig_distance" and letter in "ad") or (
        prefix in ("fig_forward_overlay", "fig_reverse_overlay")
        and letter == "a")
    ax.grid(True, which="major", axis="y" if category_panel else "both",
            color="#E5E5E5", linewidth=0.5, alpha=0.70)
    if prefix == "fig_distance" and letter in "ad":
        ax.tick_params(axis="x", labelsize=12.5)
        for label in ax.get_xticklabels():
            label.set_rotation(40)
            label.set_ha("right")
        if letter == "d":
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    if prefix in ("fig_forward_overlay", "fig_reverse_overlay") and letter == "a":
        ax.tick_params(axis="x", labelsize=12.5)
    if prefix == "knn_v32_10k_summary" and letter == "a":
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.001))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    if prefix == "knn_v32_10k_summary" and letter == "c":
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    policy = PANEL_LEGENDS[prefix][letter]
    legend = ax.get_legend()
    if policy is not None and legend is not None:
        # Explicit handles preserve the custom stage patches used by distance.
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.get_texts()]
        for handle, label in zip(handles, labels):
            handle.set_label(label)
        legend.remove()
        paper_legend(ax, loc=policy[0], ncol=policy[1], handles=handles)


def export_consistent_panels(fig, axes_dict, out_dir, fmt, dpi, name_prefix):
    """Save the selected paper panels on equal canvases; return their keys.

    Other panels retain the scripts' existing standalone export behaviour.
    The original figure geometry is restored before returning.
    """
    policies = PANEL_LEGENDS.get(name_prefix, {})
    selected = [key for key in axes_dict if key in policies]
    if not selected:
        return set()

    original_size = fig.get_size_inches().copy()
    original_layout = fig.get_layout_engine()
    original_positions = {ax: ax.get_position().frozen() for ax in fig.axes}
    original_visibility = {ax: ax.get_visible() for ax in fig.axes}
    original_titles = {ax: ax.get_title() for ax in fig.axes}
    fig.set_layout_engine(None)
    report = []
    try:
        for letter in selected:
            ax = axes_dict[letter]
            if isinstance(ax, (tuple, list)):
                raise ValueError("The paper-panel style expects a single axis")
            for other in fig.axes:
                other.set_visible(other is ax)
            fig.set_size_inches(*FIGSIZE, forward=False)
            ax.set_position(AXES_RECT)
            _style_panel(ax, name_prefix, letter)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            canvas_box = fig.bbox
            legend = ax.get_legend()
            legend_box = legend.get_window_extent(renderer) if legend else None
            axis_box = ax.get_window_extent(renderer)
            unclipped = True
            outside = []
            labels = [ax.xaxis.label, ax.yaxis.label]
            # Locators also create ticks just outside the view interval;
            # Matplotlib does not draw those, so exclude them from the check.
            for axis in (ax.xaxis, ax.yaxis):
                lower, upper = sorted(axis.get_view_interval())
                labels.extend(tick.label1 for tick in axis.get_major_ticks()
                              if lower <= tick.get_loc() <= upper)
            for label in labels:
                if not label.get_text() or not label.get_visible():
                    continue
                box = label.get_window_extent(renderer)
                if not (canvas_box.x0 <= box.x0 and box.x1 <= canvas_box.x1
                        and canvas_box.y0 <= box.y0 and box.y1 <= canvas_box.y1):
                    unclipped = False
                    outside.append(label.get_text())
            legend_inside = legend_box is None or (
                axis_box.x0 <= legend_box.x0 and legend_box.x1 <= axis_box.x1
                and axis_box.y0 <= legend_box.y0 and legend_box.y1 <= axis_box.y1)
            if not unclipped or not legend_inside:
                raise ValueError(
                    f"Panel {name_prefix}/{letter}: outside labels={outside}, "
                    f"legend_inside={legend_inside}, "
                    f"legend_width={legend_box.width if legend_box else 0}, "
                    f"axis_width={axis_box.width}")
            output = Path(out_dir) / f"{name_prefix}_panel_{letter}.{fmt}"
            # rcParams['savefig.bbox']='tight' would otherwise resize each
            # canvas independently and undo the common physical font scale.
            with mpl.rc_context({"savefig.bbox": None}):
                fig.savefig(output, dpi=dpi, bbox_inches=None, facecolor="white")
            report.append({
                "panel": letter, "file": output.name,
                "canvas_inches": FIGSIZE, "axis_label_pt": LABEL_SIZE,
                "tick_pt": 12.5 if category_panel_key(name_prefix, letter) else TICK_SIZE,
                "legend_pt": LEGEND_SIZE if legend else None,
                "legend_inside_axes": bool(legend_inside),
                "labels_inside_canvas": bool(unclipped),
                "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
            })
    finally:
        fig.set_size_inches(original_size, forward=False)
        for ax in fig.axes:
            ax.set_position(original_positions[ax])
            ax.set_visible(original_visibility[ax])
            ax.set_title(original_titles[ax])
        fig.set_layout_engine(original_layout)
    report_path = Path(out_dir) / f"{name_prefix}_layout_check.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return set(selected)


def category_panel_key(prefix, letter):
    return (prefix == "fig_distance" and letter in "ad") or (
        prefix in ("fig_forward_overlay", "fig_reverse_overlay") and letter == "a")

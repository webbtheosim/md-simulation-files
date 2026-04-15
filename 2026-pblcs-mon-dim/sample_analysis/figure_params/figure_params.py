"""
Common figure styling constants and helpers for paper_1_figures.
Keeping colors/labels and basic matplotlib defaults in one place makes it
easier to keep panels consistent across scripts.
"""

from __future__ import annotations

import matplotlib as mpl

# System-specific colors and labels reused across figures
SYSTEM_COLORS = {"P_monomer": "#90c0c8", "M_monomer": "#c890c0"}
SYSTEM_LABELS = {"P_monomer": r"$P_1$", "M_monomer": r"$M_1$"}
# Dimer systems
DIMER_COLORS = {"PM": "#90c0c8", "MP": "#c890c8", "P2": "#c8c890", "M2": "#90c8a0"}
DIMER_LABELS = {"PM": r"$\mathrm{PM}$", "MP": r"$\mathrm{MP}$", "P2": r"$\mathrm{P}_2$", "M2": r"$\mathrm{M}_2$"}

# Typography and line settings
AXES_LINEWIDTH = 1.5
TICK_WIDTH = 1.3
TICK_FONTSIZE = 8
TICK_LENGTH = 3.0
LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 8
LEGEND_FONTSIZE = 8
FIG_DPI = 900
MARKER_SIZE = 4.5
MARKER_EDGEWIDTH = 1.2
FILL_ALPHA = 0.25

# ML analysis figure defaults
ML_HIST_BINS = 40
ML_HIST_SMOOTH_KERNEL = (0.25, 0.5, 0.25)
ML_HIST_INTERP_FACTOR = 6

ML_PANEL_W = 1.30
ML_PANEL_H = 1.55
ML_WSPACE = 0.08
ML_HSPACE = 0.18
ML_LEFT_MARGIN = 0.08
ML_RIGHT_MARGIN = 0.92
ML_BOTTOM_MARGIN = 0.10
ML_TOP_MARGIN = 0.90
ML_ROW_LABEL_X = 0.06

ML_RIDGE_GAP = 1.35
ML_RIDGE_SCALE = 1.0
ML_JOY_FIG_W = 1.8
ML_JOY_FIG_H_PER_RIDGE = 0.35
ML_JOY_FIG_H_BASE = 1.3

ML_STD_BAR_FIGSIZE = (3.3075, 1.575)
ML_STD_BAR_COLOR = "#d9d9d9"
ML_STD_BAR_HEIGHT = 0.9
ML_STD_BAR_XMAX = 0.1
ML_STD_BAR_XTICKS = (0.0, 0.05, 0.1)


def use_mpl_defaults() -> None:
    """Apply lightweight rcParams defaults for consistent styling."""
    mpl.rcParams.update(
        {
            "axes.linewidth": AXES_LINEWIDTH,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "legend.fontsize": LEGEND_FONTSIZE,
            "legend.frameon": False,
        }
    )


def style_axis(ax, show_left: bool = True, show_bottom: bool = True) -> None:
    """Consistently style axes ticks and spines."""
    ax.tick_params(
        direction="in",
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        top=True,
        right=True,
        labelleft=show_left,
        labelbottom=show_bottom,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(AXES_LINEWIDTH)

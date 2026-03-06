#!/usr/bin/env python
"""
Stack three torsion energy profiles (torsion 0, 1, 9) in a tall, narrow panel
matching the N_visualize axis box geometry while leaving right-side space for
chemical structures.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DATA PATHS -- adjust these to point to your local data locations
# DATA_ROOT: root of the openff_QM_optimization tree containing visualize_N/
# ---------------------------------------------------------------------------
import os
DATA_ROOT = os.environ.get("QM_DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/openff_QM_optimization")


import sys
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# shared styling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure_params"))
if FIG_PARAM_DIR not in sys.path:
    sys.path.insert(0, FIG_PARAM_DIR)
import figure_params as fp  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# paths and constants
DATA_DIR = os.path.join(DATA_ROOT, "visualize_N")
OUTPUT_PATH = os.path.join(BASE_DIR, "polished_dft_torsion_panel.svg")
KEY_OUTPUT_PATH = os.path.join(BASE_DIR, "key.svg")

PANEL_ORDER = [0, 1, 9]  # top to bottom
FIG_SIZE = (1.5, 4.0)  # inches (width, height)
KEY_FIG_SIZE = (1.6, 0.7)

# Axis box dimensions taken from the original N_visualize single-panel figure
# (figsize 1.5 in with a 0.65x0.65 axis). Keep absolute box size but shift
# slightly right to open room for the shared y-label and avoid overlap.
AX_WIDTH_FRAC = 0.65
AX_HEIGHT_FRAC = 0.65 * (1.5 / FIG_SIZE[1])  # 0.24375 for a 4.0 in tall fig
LEFT_FRAC = 0.22
TOP_FRAC = 0.05
BOTTOM_FRAC = 0.16  # extra bottom margin keeps angled tick labels visible
VERT_GAP_FRAC = (1 - TOP_FRAC - BOTTOM_FRAC - AX_HEIGHT_FRAC * len(PANEL_ORDER)) / (len(PANEL_ORDER) - 1)
YLABEL_X = LEFT_FRAC - 0.205

ANGLE_TICKS = [-180, -90, 0, 90, 180]
COLORS = {"initial": "#1f77b4", "final": "#ff7f0e"}
QM_LABEL = "B3LYP-D3BJ/DZVP"


def _load_energy_file(path: str) -> np.ndarray:
    """Load an energy text file (skip header), returning an empty array if missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing energy file: {path}")
    return np.loadtxt(path, skiprows=1)


def load_curves(torsion: int) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load and process energy curves for a torsion.

    Returns angles (deg) and dict with initial, final, QM energies (kJ/mol),
    each shifted to a minimum of zero and rolled by half the number of points
    to match N_visualize.
    """
    init_path = os.path.join(DATA_DIR, f"torsion-{torsion}-initial.txt")
    final_path = os.path.join(DATA_DIR, f"torsion-{torsion}-final.txt")

    before_data = _load_energy_file(init_path)
    after_data = _load_energy_file(final_path)

    initial_mm = before_data[:, 1] * 4.184
    final_mm = after_data[:, 1] * 4.184
    qm = after_data[:, 0] * 4.184

    for arr in (initial_mm, final_mm, qm):
        arr -= np.min(arr)

    n_points = initial_mm.size
    if n_points == 0 or final_mm.size != n_points or qm.size != n_points:
        raise ValueError(f"Unexpected array sizes for torsion {torsion}")

    shift = n_points // 2
    initial_mm = np.roll(initial_mm, shift)
    final_mm = np.roll(final_mm, shift)
    qm = np.roll(qm, shift)

    angles = np.linspace(-180, 180, n_points)
    return angles, {"initial": initial_mm, "final": final_mm, "qm": qm}


def axis_positions(n_panels: int) -> Iterable[Tuple[float, float, float, float]]:
    """Yield axes positions (left, bottom, width, height) for top-to-bottom panels."""
    for idx in range(n_panels):
        y0 = 1 - TOP_FRAC - AX_HEIGHT_FRAC * (idx + 1) - VERT_GAP_FRAC * idx
        yield (LEFT_FRAC, y0, AX_WIDTH_FRAC, AX_HEIGHT_FRAC)


def plot_panel() -> None:
    """Assemble the stacked torsion plots."""
    fp.use_mpl_defaults()
    fig = plt.figure(figsize=FIG_SIZE)

    positions = list(axis_positions(len(PANEL_ORDER)))
    for idx, torsion in enumerate(PANEL_ORDER):
        angles, curves = load_curves(torsion)
        ax = fig.add_axes(positions[idx])

        ax.plot(angles, curves["initial"], color=COLORS["initial"], linewidth=2.0, alpha=0.8, label="Initial Sage")
        ax.plot(
            angles,
            curves["final"],
            color=COLORS["final"],
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            label="Final Sage",
        )
        ax.scatter(
            angles,
            curves["qm"],
            color=COLORS["final"],
            alpha=0.8,
            s=25,
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
            label=QM_LABEL,
        )

        max_energy = float(max(np.max(curves["initial"]), np.max(curves["final"]), np.max(curves["qm"])))
        tick_step = max(1, int(np.ceil(max_energy / 3.0)))
        tick_max = tick_step * 3

        ax.set_xlim(-180, 180)
        ax.set_xticks(ANGLE_TICKS)
        ax.set_ylim(0, tick_max)
        ax.set_yticks(np.arange(0, tick_max + 1, tick_step))

        show_x_labels = idx == len(PANEL_ORDER) - 1
        fp.style_axis(ax, show_left=True, show_bottom=show_x_labels)
        ax.tick_params(axis="x", labelrotation=45 if show_x_labels else 0)
        if show_x_labels:
            ax.set_xlabel(r"Dihedral Angle, $\phi$ (°)", fontsize=fp.LABEL_FONTSIZE)
        ax.set_ylabel("")

    # shared labels
    mid_idx = len(PANEL_ORDER) // 2
    mid_pos = positions[mid_idx]
    y_label_center = mid_pos[1] + 0.5 * mid_pos[3]
    fig.text(
        YLABEL_X,
        y_label_center,
        r"Energy, $E$ (kJ/mol)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=fp.LABEL_FONTSIZE,
    )

    fig.savefig(OUTPUT_PATH, dpi=fp.FIG_DPI)
    plt.close(fig)
    print(f"[done] wrote figure to {OUTPUT_PATH}")


def save_key_figure() -> None:
    """Write a small legend-only SVG for the torsion panel."""
    fp.use_mpl_defaults()
    fig = plt.figure(figsize=KEY_FIG_SIZE)
    ax = fig.add_subplot(111)

    handles = [
        Line2D([], [], color=COLORS["initial"], linewidth=2.0, alpha=0.8, label="Initial Sage"),
        Line2D(
            [],
            [],
            color=COLORS["final"],
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            label="Final Sage",
        ),
        Line2D(
            [],
            [],
            color=COLORS["final"],
            marker="o",
            markersize=5.5,
            markerfacecolor=COLORS["final"],
            markeredgecolor="black",
            markeredgewidth=1.5,
            linestyle="None",
            label=QM_LABEL,
        ),
    ]

    ax.legend(handles=handles, frameon=False, loc="center left")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(KEY_OUTPUT_PATH, dpi=fp.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote key to {KEY_OUTPUT_PATH}")


def main() -> None:
    plot_panel()
    save_key_figure()


if __name__ == "__main__":
    main()

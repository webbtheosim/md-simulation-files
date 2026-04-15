#!/usr/bin/env python3
"""
Plot the joint distribution of Rg vs bend angle for P2 at 400 K and 460 K.
Generates two square panels side-by-side with a shared bottom colorbar.

Computes bend angle and Rg from trajectories, with .npz caching.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# DATA PATHS -- adjust these to point to your local data locations
# DATA_ROOT: root of the paper_1_figures analysis tree (for cached data)
# SIMULATION_ROOT: root of the raw GROMACS simulation directories
# ---------------------------------------------------------------------------
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/paper_1_figures")
SIMULATION_ROOT = os.environ.get("SIMULATION_ROOT", "/scratch/gpfs/WEBB/jv6139")


import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.colors import LogNorm
from matplotlib import transforms

import MDAnalysis as mda
from MDAnalysis.transformations import unwrap
from rdkit import Chem

# ---- Figure params ----
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

# ---- Paths and constants ----
SYSTEM = "P2"
TEMPERATURES = (400, 460)
FRAME_STRIDE = 10

HIST_BINS = (80, 80)
RG_RANGE = (7.0, 17.0)
BEND_RANGE = (0.0, 180.0)

FIG_SIZE = (3.0, 1.8)
LEFT = 0.12
GAP = 0.09
AX_WIDTH = (1.0 - 2 * LEFT - GAP) / 2.0
AX_HEIGHT = AX_WIDTH * (FIG_SIZE[0] / FIG_SIZE[1])
BOTTOM = 0.26
XLABEL_Y = -0.22

# Cache directory for computed metrics
CACHE_DIR = os.path.join(DATA_ROOT, "bend_angle_distributions", "bend_angles")
os.makedirs(CACHE_DIR, exist_ok=True)

OUTPUT_PATH = SCRIPT_DIR / "p2_rg_bend_joint_400_460.svg"
COLORBAR_PATH = SCRIPT_DIR / "p2_rg_bend_joint_400_460_colorbar.svg"

SMARTS_CORE = Chem.MolFromSmarts("c1ccccc1COc2ccccc2")


def traj_path(system: str, temperature: float) -> str:
    return os.path.join(SIMULATION_ROOT, "dimers", system, "simulations", "tREM", f"{temperature}K", "trem_gpu.xtc")


def top_path(system: str, temperature: float) -> str:
    return os.path.join(SIMULATION_ROOT, "dimers", system, "simulations", "tREM", f"{temperature}K", "trem_gpu.tpr")


# ====================== COMPUTATION CODE ======================
# Adapted from bend_angle_distributions/bend_angles/bend_angle.py

def _prepare_ring_pairs(frag: mda.AtomGroup) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Return a list of (ring_a_indices, ring_b_indices) for the SMARTS directors."""
    if SMARTS_CORE is None:
        raise RuntimeError("SMARTS pattern failed to compile.")

    rdmol = frag.convert_to("RDKIT")
    matches = rdmol.GetSubstructMatches(SMARTS_CORE)
    if not matches:
        raise ValueError("SMARTS pattern not found in fragment.")

    ring_info = rdmol.GetRingInfo().AtomRings()
    ring_pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for match in matches:
        rings = [tuple(r) for r in ring_info if set(r).issubset(match)]
        if len(rings) != 2:
            raise ValueError(f"Expected exactly 2 rings per match, found {len(rings)}.")
        ring_pairs.append((rings[0], rings[1]))

    if len(ring_pairs) != 2:
        raise ValueError(f"Expected two director vectors, got {len(ring_pairs)}.")
    return ring_pairs


def _compute_bend_angle(frag: mda.AtomGroup, ring_pairs: Sequence[Tuple[Sequence[int], Sequence[int]]]) -> float:
    """Calculate the bend angle (degrees) from cached ring index pairs."""
    directors: List[np.ndarray] = []
    for ring_a, ring_b in ring_pairs:
        com_a = frag.atoms[list(ring_a)].center_of_mass()
        com_b = frag.atoms[list(ring_b)].center_of_mass()
        vec = com_b - com_a
        norm = np.linalg.norm(vec)
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError("Degenerate ring director encountered (norm ~ 0).")
        directors.append(vec / norm)

    if len(directors) != 2:
        raise ValueError("Bend angle requires two director vectors.")

    cosang = float(np.clip(np.dot(directors[0], directors[1]), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _compute_rg(frag: mda.AtomGroup) -> float:
    """Compute the (unweighted) radius of gyration for the fragment."""
    coords = frag.atoms.positions
    com = frag.atoms.center_of_mass()
    rel = coords - com
    gyr_tensor = np.einsum("im,in->mn", rel, rel) / rel.shape[0]
    return float(np.sqrt(np.trace(gyr_tensor)))


def _collect_metrics(system: str, temperature: float, frame_stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Gather bend angle and Rg pairs for a single temperature.
    Uses cached .npy files if available, otherwise computes from trajectories."""

    # Check for cached data
    T_label = f"{int(temperature)}K" if float(temperature).is_integer() else f"{temperature}K"
    cache_subdir = os.path.join(CACHE_DIR, system, T_label)
    os.makedirs(cache_subdir, exist_ok=True)
    bend_cache = os.path.join(cache_subdir, f"bend_angle_{T_label}.npy")
    rg_cache = os.path.join(cache_subdir, f"rg_{T_label}.npy")

    if os.path.isfile(bend_cache) and os.path.isfile(rg_cache):
        bend_arr = np.load(bend_cache).ravel()
        rg_arr = np.load(rg_cache).ravel()
        if rg_arr.shape == bend_arr.shape:
            print(f"[cached] {system} {temperature} K")
            return rg_arr, bend_arr
        print(f"[stale cache] {system} {temperature} K (shape mismatch), recomputing")

    # Compute from trajectories
    top = top_path(system, temperature)
    traj = traj_path(system, temperature)
    if not (os.path.isfile(top) and os.path.isfile(traj)):
        raise FileNotFoundError(f"Missing topology/trajectory for {system} {temperature} K.")

    print(f"[compute] {system} {temperature} K")
    u = mda.Universe(top, traj)
    u.trajectory.add_transformations(unwrap(u.atoms))
    fragments = list(u.atoms.fragments)
    if not fragments:
        raise RuntimeError("No fragments detected in topology.")

    ring_pairs_all = [_prepare_ring_pairs(frag) for frag in fragments]

    rg_values: List[float] = []
    bend_values: List[float] = []
    stride = max(1, int(frame_stride))

    for ts in u.trajectory[::stride]:
        for frag_idx, frag in enumerate(fragments):
            try:
                bend = _compute_bend_angle(frag, ring_pairs_all[frag_idx])
            except ValueError as err:
                print(f"[warn] {temperature} K frame {ts.frame} frag {frag_idx}: {err}")
                continue
            rg_values.append(_compute_rg(frag))
            bend_values.append(bend)

    rg_arr = np.asarray(rg_values, dtype=float)
    bend_arr = np.asarray(bend_values, dtype=float)

    # Cache the computed data
    # Reshape to (n_frames, n_frags) for compatibility with bend_angle.py format
    n_frags = len(fragments)
    n_frames = len(rg_arr) // n_frags if n_frags > 0 else 0
    if n_frames > 0 and n_frames * n_frags == len(rg_arr):
        np.save(bend_cache, bend_arr.reshape(n_frames, n_frags))
        np.save(rg_cache, rg_arr.reshape(n_frames, n_frags))
    else:
        np.save(bend_cache, bend_arr)
        np.save(rg_cache, rg_arr)
    print(f"[cached] {bend_cache}")

    return rg_arr, bend_arr


# ====================== PLOTTING CODE ======================

def _histogram(rg: np.ndarray, bend: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, xedges, yedges = np.histogram2d(
        rg,
        bend,
        bins=HIST_BINS,
        range=[RG_RANGE, BEND_RANGE],
    )
    total = float(hist.sum())
    if total > 0:
        hist_prob = hist / total
    else:
        hist_prob = hist.astype(float)
    return hist_prob, xedges, yedges


def _shared_norm(hists: Sequence[np.ndarray]) -> LogNorm:
    positive = np.concatenate([h[h > 0] for h in hists if h.size]) if hists else np.array([])
    if positive.size:
        vmin = float(positive.min())
        vmax = float(positive.max())
        if not np.isfinite(vmin) or vmin <= 0:
            vmin = 1e-6
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 10.0
    else:
        vmin = 1e-6
        vmax = 1.0
    exp_min = float(np.floor(np.log10(vmin)))
    exp_max = float(np.ceil(np.log10(vmax)))
    vmin = 10.0 ** exp_min
    vmax = 10.0 ** exp_max
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def _offset_zero_ytick(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.canvas.draw()
    zero_offset = transforms.ScaledTranslation(0.0, 3.0 / 72.0, fig.dpi_scale_trans)
    for label in ax.get_yticklabels():
        if label.get_text() == "0":
            label.set_transform(label.get_transform() + zero_offset)
            break


def _save_colorbar(norm: LogNorm, cmap: matplotlib.colors.Colormap) -> None:
    ticks = [norm.vmin, norm.vmax]
    fig = plt.figure(figsize=(2.2, 0.24))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.25, top=0.90)
    cbar = ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal",
        ticks=ticks,
        format=LogFormatterMathtext(10),
    )
    cbar.set_label("Prob. (a.u.)", fontsize=fp.LABEL_FONTSIZE)
    cbar.ax.xaxis.label.set_size(fp.LABEL_FONTSIZE)
    exponents = []
    for val in ticks:
        exp = int(round(np.log10(val))) if val > 0 else 0
        exponents.append(rf"$10^{{{exp}}}$")
    cbar.ax.set_xticklabels(exponents)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(
        length=fp.TICK_LENGTH,
        width=fp.TICK_WIDTH,
        labelsize=fp.TICK_FONTSIZE,
        direction="in",
        pad=2.0,
        labeltop=True,
        labelbottom=False,
    )
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(fp.AXES_LINEWIDTH)
    fig.tight_layout()
    fig.savefig(COLORBAR_PATH, dpi=fp.FIG_DPI)
    plt.close(fig)
    print(f"[done] wrote colorbar to {COLORBAR_PATH}")


def main() -> None:
    fp.use_mpl_defaults()

    data = []
    for temp in TEMPERATURES:
        rg_vals, bend_vals = _collect_metrics(SYSTEM, temp, FRAME_STRIDE)
        if rg_vals.size == 0 or bend_vals.size == 0:
            raise RuntimeError(f"No samples for {SYSTEM} at {temp} K.")
        hist, xedges, yedges = _histogram(rg_vals, bend_vals)
        data.append((temp, hist, xedges, yedges))

    norm = _shared_norm([entry[1] for entry in data])
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="white")

    fig = plt.figure(figsize=FIG_SIZE)

    axes = []
    for idx, (temp, hist, xedges, yedges) in enumerate(data):
        left = LEFT + idx * (AX_WIDTH + GAP)
        ax = fig.add_axes([left, BOTTOM, AX_WIDTH, AX_HEIGHT])
        axes.append(ax)

        hist_masked = np.ma.masked_where(hist <= 0, hist)
        mesh = ax.pcolormesh(
            xedges,
            yedges,
            hist_masked.T,
            norm=norm,
            cmap=cmap,
            shading="auto",
            antialiased=False,
            linewidth=0.0,
            edgecolors="face",
        )
        mesh.set_rasterized(False)

        ax.set_title(f"{temp} K", fontsize=fp.TITLE_FONTSIZE, pad=2)
        ax.set_xlim(RG_RANGE)
        ax.set_ylim(BEND_RANGE)
        xticks = [7.0, 9.5, 12.0, 14.5, 17.0]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{tick:g}" for tick in xticks])
        ax.set_yticks([0, 60, 120, 180])
        ax.tick_params(
            length=fp.TICK_LENGTH,
            width=fp.TICK_WIDTH,
            labelsize=fp.TICK_FONTSIZE,
            pad=2.5,
            top=True,
            right=True,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(fp.AXES_LINEWIDTH)

        for x in np.linspace(RG_RANGE[0], RG_RANGE[1], 6):
            ax.axvline(x, color="#d0d0d0", linewidth=0.6, linestyle="-", zorder=0)
        for y in np.linspace(BEND_RANGE[0], BEND_RANGE[1], 7):
            ax.axhline(y, color="#d0d0d0", linewidth=0.6, linestyle="-", zorder=0)
        ax.set_axisbelow(False)

        if idx == 0:
            ax.set_ylabel(r"bend angle ($^\circ$)", fontsize=fp.LABEL_FONTSIZE, labelpad=6)
            ax.yaxis.set_label_coords(-0.30, 0.5)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
            ax.tick_params(labelbottom=True)
        ax.set_xlabel(r"$R_\mathrm{g}$ ($\mathrm{\AA}$)", fontsize=fp.LABEL_FONTSIZE)
        ax.xaxis.set_label_coords(0.5, XLABEL_Y)
        if idx == 0:
            _offset_zero_ytick(fig, ax)

    fig.savefig(OUTPUT_PATH, dpi=fp.FIG_DPI)
    plt.close(fig)
    print(f"[done] wrote figure to {OUTPUT_PATH}")
    _save_colorbar(norm, cmap)


if __name__ == "__main__":
    main()

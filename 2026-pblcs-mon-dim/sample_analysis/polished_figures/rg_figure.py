#!/usr/bin/env python
"""
Create a compact, polished panel with (i) Rg distributions at the six target
temperatures and (ii) a variance-versus-temperature trace, suitable for the
paper-sized layout.

Computes monomer Rg from trajectories if cached data is not found.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DATA PATHS -- adjust these to point to your local data locations
# SIMULATION_ROOT: root of the raw GROMACS simulation directories
# DATA_ROOT: root of the paper_1_figures analysis tree (for cached data)
# ---------------------------------------------------------------------------
import os
SIMULATION_ROOT = os.environ.get("SIMULATION_ROOT", "/scratch/gpfs/WEBB/jv6139")
DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/paper_1_figures")

import math
import sys
import re
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# imports from shared figure formatting file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure_params"))
if FIG_PARAM_DIR not in sys.path:
    sys.path.insert(0, FIG_PARAM_DIR)
import figure_params as fp  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# paths and constants
DATA_DIR = os.path.join(DATA_ROOT, "monomer_fragment_gyration_tensor_analysis", "data")
FIG_DIR = BASE_DIR
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

TEMPS = [300, 325, 350, 375, 400, 425]
SYSTEMS = ("P_monomer", "M_monomer")
HIST_BINS = 40
SMOOTH_KERNEL = np.array([0.25, 0.5, 0.25])
INTERP = 6
FIG_SIZE = (6.7, 1.75)  # (width, height) in inches
RG_XLIM = (6.0, 10.0)
OUTPUT_PATH = os.path.join(FIG_DIR, "polished_rg_panel.svg")
LEGEND_LABELS = {"P_monomer": r"$\mathrm{P}_1$", "M_monomer": r"$\mathrm{M}_1$"}
SYSTEM_MARKERS = {"P_monomer": "o", "M_monomer": "^"}

# Trajectory path templates
SYSTEM_PATHS = {
    "P_monomer": (
        os.path.join(SIMULATION_ROOT, "N_monomer/simulations/tREM/{T}K/trem_gpu.tpr"),
        os.path.join(SIMULATION_ROOT, "N_monomer/simulations/tREM/{T}K/trem_gpu.xtc"),
    ),
    "M_monomer": (
        os.path.join(SIMULATION_ROOT, "M_monomer/simulations/tREM/{T}K/trem_gpu.tpr"),
        os.path.join(SIMULATION_ROOT, "M_monomer/simulations/tREM/{T}K/trem_gpu.xtc"),
    ),
}

SKIP = 5
N_CORES = min(8, os.cpu_count() or 1)

# Fragment SMARTS for the full fragment analysis (needed for npz cache format)
FRAG_SMARTS_SPECS = {
    "frag1": "[O]CCCCCCCCCl",
    "frag2": "[O]CCCCC#C",
    "frag3": "O=C(Oc1ccc(cc1)*)c2ccc(cc2)*",
}
POLYMER_LABEL = "polymer"
FRAG_ORDER = [POLYMER_LABEL] + list(FRAG_SMARTS_SPECS.keys())
PROPS = ["Rg2", "acylindricity", "asphericity", "kappa2"]
ANGLE_PAIRS = [("frag1", "frag3"), ("frag2", "frag3")]


# ====================== COMPUTATION CODE ======================
# Adapted from monomer_fragment_gyration_tensor_analysis/monomer_fragment_analysis.py

def _gyration_tensor(coords: np.ndarray) -> np.ndarray:
    """Compute the gyration tensor from coordinates."""
    com = coords.mean(axis=0)
    centered = coords - com
    return np.einsum("im,in->mn", centered, centered) / centered.shape[0]


def _tensor_props_and_axis(tensor: np.ndarray):
    """Return shape properties and the principal axis (lambda_z eigenvector)."""
    vals, vecs = np.linalg.eigh(tensor)
    vals = np.real(vals)
    vecs = np.real(vecs)
    sort_idx = np.argsort(vals)
    vals = vals[sort_idx]
    vecs = vecs[:, sort_idx]
    vals = np.clip(vals, 0.0, None)
    rg2 = float(np.sum(vals))
    if rg2 <= 0:
        props = {"Rg2": float("nan"), "acylindricity": float("nan"),
                 "asphericity": float("nan"), "kappa2": float("nan")}
        return props, None
    b = float(vals[2] - 0.5 * (vals[0] + vals[1]))
    c = float(vals[1] - vals[0])
    kappa2 = (b * b + 0.75 * c * c) / (rg2 * rg2)
    props = {"Rg2": rg2, "acylindricity": b, "asphericity": c, "kappa2": kappa2}
    vec_z = vecs[:, -1]
    norm = np.linalg.norm(vec_z)
    if norm > 0:
        vec_z = vec_z / norm
    else:
        vec_z = None
    return props, vec_z


def _build_fragment_matches(u):
    """Build per-fragment SMARTS match indices using RDKit."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    frag_smarts = {k: Chem.MolFromSmarts(v) for k, v in FRAG_SMARTS_SPECS.items()}
    matches = []
    for frag in u.atoms.fragments:
        rdmol = frag.convert_to("RDKIT")
        frag_matches = {POLYMER_LABEL: [tuple(range(frag.atoms.n_atoms))]}
        for label, patt in frag_smarts.items():
            frag_matches[label] = list(rdmol.GetSubstructMatches(patt, uniquify=True))
        matches.append(frag_matches)
    return matches


# Multiprocessing globals
_mp_univ = None
_mp_match_map = None


def _mp_init_worker(top_path: str, traj_path: str):
    """Initializer: one Universe per worker + cached matches."""
    global _mp_univ, _mp_match_map
    import MDAnalysis as mda
    from MDAnalysis.transformations import unwrap
    _mp_univ = mda.Universe(top_path, traj_path)
    _mp_univ.trajectory.add_transformations(unwrap(_mp_univ.atoms))
    _mp_match_map = _build_fragment_matches(_mp_univ)


def _mp_process_frame(frame_idx: int):
    """Compute props for one frame across all fragments."""
    global _mp_univ, _mp_match_map
    _mp_univ.trajectory[frame_idx]

    full_local = {label: {p: [] for p in PROPS} for label in FRAG_ORDER}
    angle_local = {pair: [] for pair in ANGLE_PAIRS}

    for frag_idx, frag in enumerate(_mp_univ.atoms.fragments):
        frag_matches = _mp_match_map[frag_idx]
        principal_axes = {}
        com_positions = {}
        ee_vectors = {}

        for label, atom_sets in frag_matches.items():
            for atom_idxs in atom_sets:
                coords = frag.atoms[list(atom_idxs)].positions
                if coords.shape[0] < 2:
                    continue
                tensor = _gyration_tensor(coords)
                props, vec_z = _tensor_props_and_axis(tensor)
                for p in PROPS:
                    full_local[label][p].append(props[p])
                if label not in com_positions:
                    com_positions[label] = coords.mean(axis=0)
                if vec_z is not None and label not in principal_axes:
                    principal_axes[label] = vec_z
                if label in ("frag1", "frag2") and label not in ee_vectors:
                    ee = coords[-1] - coords[0]
                    ee_norm = np.linalg.norm(ee)
                    if ee_norm > 1e-10:
                        ee_vectors[label] = ee / ee_norm

        for a, b in ANGLE_PAIRS:
            if a in ee_vectors and b in principal_axes and b in com_positions and a in com_positions:
                v_tail = ee_vectors[a]
                e_core = principal_axes[b].copy()
                com_dir = com_positions[a] - com_positions[b]
                if np.dot(e_core, com_dir) < 0:
                    e_core = -e_core
                dot = float(np.clip(np.dot(v_tail, e_core), -1.0, 1.0))
                angle_deg = 180.0 - math.degrees(math.acos(dot))
                angle_local[(a, b)].append(angle_deg)

    return full_local, angle_local


def compute_and_cache(system: str, T: int) -> None:
    """Compute fragment gyration properties and cache as .npz."""
    import MDAnalysis as mda

    label = str(int(T))
    cache_path = os.path.join(DATA_DIR, f"{system}_{label}K_gyration_props.npz")

    if system not in SYSTEM_PATHS:
        print(f"[skip] unknown system {system}")
        return

    top_tmpl, xtc_tmpl = SYSTEM_PATHS[system]
    top = top_tmpl.format(T=T)
    xtc = xtc_tmpl.format(T=T)

    if not (os.path.isfile(top) and os.path.isfile(xtc)):
        print(f"[skip] {system} {T} K - missing trajectory files")
        return

    print(f"[compute] {system} {T} K")
    u_tmp = mda.Universe(top, xtc)
    frame_indices = list(range(0, u_tmp.trajectory.n_frames, SKIP))

    full = {label_name: {p: [] for p in PROPS} for label_name in FRAG_ORDER}
    angle_accum = {pair: [] for pair in ANGLE_PAIRS}

    with mp.Pool(processes=N_CORES, initializer=_mp_init_worker, initargs=(top, xtc)) as pool:
        for full_local, angle_local in pool.imap_unordered(_mp_process_frame, frame_indices):
            for label_name in full_local:
                for p in PROPS:
                    full[label_name][p].extend(full_local[label_name][p])
            for pair in ANGLE_PAIRS:
                angle_accum[pair].extend(angle_local[pair])

    cache_data = {}
    for frag_label in FRAG_ORDER:
        for p in PROPS:
            cache_data[f"{frag_label}_{p}"] = np.asarray(full[frag_label][p], dtype=float)
    for a, b in ANGLE_PAIRS:
        cache_data[f"angle_{a}_{b}"] = np.asarray(angle_accum[(a, b)], dtype=float)
    np.savez(cache_path, **cache_data)
    print(f"[cached] {cache_path}")


# ====================== ORIGINAL PLOTTING CODE ======================

def load_rg_values(system: str, temp: int | float) -> np.ndarray:
    """Load polymer Rg values (Ang) for a given system/temperature.
    If cached data is not found, compute from trajectories first."""
    label = f"{int(temp)}" if float(temp).is_integer() else f"{temp:g}"
    npz_path = os.path.join(DATA_DIR, f"{system}_{label}K_gyration_props.npz")

    if not os.path.isfile(npz_path):
        # Try to compute from trajectories
        compute_and_cache(system, int(temp))

    if not os.path.isfile(npz_path):
        return np.array([])
    data = np.load(npz_path)
    if "polymer_Rg2" not in data:
        return np.array([])
    vals = np.asarray(data["polymer_Rg2"], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return np.sqrt(vals)


def density_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Histogram + light smoothing to produce clean outlines."""
    if values.size == 0:
        return np.array([]), np.array([])
    counts, edges = np.histogram(values, bins=HIST_BINS, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(counts, SMOOTH_KERNEL, mode="same")
    if centers.size > 1 and INTERP > 1:
        xs = np.linspace(centers[0], centers[-1], centers.size * INTERP)
        ys = np.interp(xs, centers, smooth)
    else:
        xs, ys = centers, smooth
    return xs, ys


def collect_distributions() -> tuple[dict[int, list[tuple[str, np.ndarray, np.ndarray]]], float]:
    """
    Gather density curves for each temperature/system and return a shared y-max.
    """
    dist_data: dict[int, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    y_samples: list[float] = []
    for temp in TEMPS:
        dist_data[temp] = []
        for system in SYSTEMS:
            vals = load_rg_values(system, temp)
            xs, ys = density_curve(vals)
            dist_data[temp].append((system, xs, ys))
            if ys.size:
                y_samples.extend(ys.tolist())
    if y_samples:
        y_max = float(np.quantile(y_samples, 0.995) * 1.05)
    else:
        y_max = 1.0
    return dist_data, y_max


def collect_variances() -> dict[str, dict[str, list[float]]]:
    """Variance of Rg (Ang^2) by system and temperature."""
    var_data: dict[str, dict[str, list[float]]] = {sys: {"T": [], "var": []} for sys in SYSTEMS}

    def available_temps(system: str) -> list[float]:
        temps: list[float] = []
        prefix = f"{system}_"
        suffix = "_gyration_props.npz"
        if not os.path.isdir(DATA_DIR):
            return list(TEMPS)
        for fname in os.listdir(DATA_DIR):
            if not (fname.startswith(prefix) and fname.endswith(suffix)):
                continue
            label = fname[len(prefix) : -len(suffix)]
            m = re.match(r"([0-9]+(?:\\.[0-9]+)?)K", label)
            if not m:
                continue
            try:
                temps.append(float(m.group(1)))
            except ValueError:
                continue
        if not temps:
            temps = list(TEMPS)
        return sorted(set(temps))

    for system in SYSTEMS:
        for temp in available_temps(system):
            vals = load_rg_values(system, temp)
            if vals.size < 2:
                continue
            var_val = float(np.var(vals, ddof=1))
            var_data[system]["T"].append(temp)
            var_data[system]["var"].append(var_val)
    return var_data


def add_global_labels(fig: plt.Figure, rg_axes: list[plt.Axes]) -> None:
    """Place shared labels for the Rg mini-panels."""
    if not rg_axes:
        return
    left = rg_axes[0].get_position().x0
    right = rg_axes[-1].get_position().x1
    x_center = 0.5 * (left + right)
    fig.text(
        left - 0.055,
        0.52,
        "Prob. (a.u.)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=fp.LABEL_FONTSIZE,
    )
    fig.text(
        x_center,
        0.1,
        r"Radius of Gyration, $R_{\mathrm{g}}$ ($\mathrm{\AA}$)",
        ha="center",
        va="center",
        fontsize=fp.LABEL_FONTSIZE,
    )


def plot_panel(dist_data: dict[int, list[tuple[str, np.ndarray, np.ndarray]]], var_data: dict[str, dict[str, list[float]]], y_max: float) -> None:
    """Make the combined Rg distribution + variance figure."""
    fp.use_mpl_defaults()
    fig = plt.figure(figsize=FIG_SIZE)
    width_ratios = [1.0] * len(TEMPS) + [0.8, 1.35]  # wider spacer + variance panel
    gs = fig.add_gridspec(
        1,
        len(width_ratios),
        width_ratios=width_ratios,
        left=0.08,
        right=0.98,
        bottom=0.24,
        top=0.9,
        wspace=0.22,
    )

    # Rg distributions
    rg_axes: list[plt.Axes] = []
    for idx, temp in enumerate(TEMPS):
        ax = fig.add_subplot(gs[0, idx], sharey=rg_axes[0] if rg_axes else None)
        rg_axes.append(ax)
        for system, xs, ys in dist_data.get(temp, []):
            if xs.size == 0 or ys.size == 0:
                continue
            ax.fill_between(xs, ys, color=fp.SYSTEM_COLORS[system], alpha=0.28, linewidth=0)
            label = LEGEND_LABELS[system] if idx == len(TEMPS) - 1 else None
            ax.plot(xs, ys, color=fp.SYSTEM_COLORS[system], lw=1.4, label=label)
        ax.set_xlim(RG_XLIM)
        ax.set_ylim(0, 2.0)
        ax.set_yticks(np.arange(0, 2.1, 0.5))
        ax.set_xticks([6, 8, 10])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        fp.style_axis(ax, show_left=(idx == 0), show_bottom=True)
        ax.tick_params(labelsize=fp.TICK_FONTSIZE)
        if idx != 0:
            ax.tick_params(labelleft=False)
        ax.set_title(f"{temp} K", pad=4)

    if rg_axes:
        rg_axes[-1].legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=fp.LEGEND_FONTSIZE,
            handlelength=1.8,
        )

    # Variance vs temperature
    var_ax = fig.add_subplot(gs[0, -1])
    for system in SYSTEMS:
        temps = var_data.get(system, {}).get("T", [])
        vars_rg = var_data.get(system, {}).get("var", [])
        if not temps:
            continue
        order = np.argsort(temps)
        temps = np.asarray(temps)[order]
        vars_rg = np.asarray(vars_rg)[order]
        plot_t, plot_v = temps, vars_rg
        var_ax.plot(
            plot_t,
            plot_v,
            color=fp.SYSTEM_COLORS[system],
            marker=SYSTEM_MARKERS.get(system, "o"),
            markersize=fp.MARKER_SIZE,
            linestyle="None",
            linewidth=0,
            markeredgecolor="black",
            markeredgewidth=fp.MARKER_EDGEWIDTH,
            label=fp.SYSTEM_LABELS[system],
        )
    var_ax.axvline(350, color=fp.SYSTEM_COLORS["M_monomer"], linestyle="--", linewidth=1.2, alpha=0.7, zorder=0.5)
    var_ax.axvline(400, color=fp.SYSTEM_COLORS["P_monomer"], linestyle="--", linewidth=1.2, alpha=0.7, zorder=0.5)
    var_ax.set_xlabel("T (K)")
    var_ax.set_ylabel(r"Variance[$R_{\mathrm{g}}$] ($\mathrm{\AA}^{2}$)")
    fp.style_axis(var_ax)
    var_ax.tick_params(labelsize=fp.TICK_FONTSIZE)
    var_ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    desired_ticks = [300, 370, 440]
    var_ax.set_xticks(desired_ticks)
    var_ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    var_ax.set_xlim(300, 440)
    var_ax.set_ylim(0.1, 0.25)
    var_ax.set_yticks(np.linspace(0.1, 0.25, 4))

    add_global_labels(fig, rg_axes)
    fig.savefig(OUTPUT_PATH, dpi=fp.FIG_DPI)
    plt.close(fig)


def main() -> None:
    dist_data, y_max = collect_distributions()
    var_data = collect_variances()
    plot_panel(dist_data, var_data, y_max)
    print(f"[done] wrote figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

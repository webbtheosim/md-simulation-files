#!/usr/bin/env python3
"""
Plot fragment Rg and angle distributions vs temperature using cached data
from monomer_fragment_gyration_tensor_analysis, styled with figure_params.

Computes fragment gyration tensor properties from trajectories if cached
data is not found.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# DATA PATHS -- adjust these to point to your local data locations
# SIMULATION_ROOT: root of the raw GROMACS simulation directories
# DATA_ROOT: root for cached .npz data files
#   The script expects/creates data files in DATA_ROOT/data/ (e.g. *_gyration_props.npz)
# ---------------------------------------------------------------------------
import os
SIMULATION_ROOT = os.environ.get("SIMULATION_ROOT", "/scratch/gpfs/WEBB/jv6139")
DATA_ROOT = os.environ.get("FRAGMENT_RG_DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/paper_1_figures/polished_figures/ml_analysis_fig/fragment_rg_and_angle_analysis")

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import sys
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---- Figure params ----
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

DATA_DIR = Path(DATA_ROOT) / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = SCRIPT_DIR

SYSTEMS = ["P_monomer", "M_monomer"]
TEMPS = [300, 325, 350, 375, 400, 425]
JOY_TEMPS = [325, 350, 375, 400]
JOY_FIG_H = 2.25
FRAGS = ["frag1", "frag2", "frag3"]
FRAG_LABELS = {
    "frag1": "Fragment 1",
    "frag2": "Fragment 2",
    "frag3": "Fragment 3",
}
ANGLE_PAIRS = [("frag1", "frag3"), ("frag2", "frag3")]
ANGLE_LABELS = {
    ("frag1", "frag3"): r"$\theta_{1}$",
    ("frag2", "frag3"): r"$\theta_{2}$",
}

HIST_BINS = fp.ML_HIST_BINS
HIST_SMOOTH_KERNEL = np.array(fp.ML_HIST_SMOOTH_KERNEL, dtype=float)
HIST_INTERP_FACTOR = fp.ML_HIST_INTERP_FACTOR

PANEL_W, PANEL_H = fp.ML_PANEL_W, fp.ML_PANEL_H
WSPACE, HSPACE = fp.ML_WSPACE, fp.ML_HSPACE
LEFT_MARGIN, RIGHT_MARGIN = fp.ML_LEFT_MARGIN, fp.ML_RIGHT_MARGIN
BOTTOM_MARGIN, TOP_MARGIN = fp.ML_BOTTOM_MARGIN, fp.ML_TOP_MARGIN
ROW_LABEL_X = fp.ML_ROW_LABEL_X

RIDGE_GAP = fp.ML_RIDGE_GAP
RIDGE_SCALE = fp.ML_RIDGE_SCALE

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

# Fragment SMARTS specifications
FRAG_SMARTS_SPECS = {
    "frag1": "[O]CCCCCCCCCl",
    "frag2": "[O]CCCCC#C",
    "frag3": "O=C(Oc1ccc(cc1)*)c2ccc(cc2)*",
}
POLYMER_LABEL = "polymer"
FRAG_ORDER = [POLYMER_LABEL] + list(FRAG_SMARTS_SPECS.keys())
PROPS = ["Rg2", "acylindricity", "asphericity", "kappa2"]


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
    """Compute fragment gyration properties from trajectory and cache as .npz."""
    import MDAnalysis as mda

    label = str(int(T))
    cache_path = DATA_DIR / f"{system}_{label}K_gyration_props.npz"

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
    np.savez(str(cache_path), **cache_data)
    print(f"[cached] {cache_path}")


# ====================== ORIGINAL PLOTTING CODE ======================

def _density_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    counts, edges = np.histogram(values, bins=HIST_BINS, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(counts, HIST_SMOOTH_KERNEL, mode="same")
    if centers.size > 1 and HIST_INTERP_FACTOR > 1:
        xs = np.linspace(centers[0], centers[-1], centers.size * HIST_INTERP_FACTOR)
        ys = np.interp(xs, centers, smooth)
    else:
        xs, ys = centers, smooth
    return xs, ys


def _density_curve_fixed(values: np.ndarray, xs: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(xs)
    counts, edges = np.histogram(values, bins=HIST_BINS, range=(x_min, x_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(counts, HIST_SMOOTH_KERNEL, mode="same")
    return np.interp(xs, centers, smooth)


def _load_temp_npz(system: str, T: int) -> tuple[Dict[str, np.ndarray], Dict[tuple[str, str], np.ndarray]]:
    """Load cached data for a system/temperature. Compute if not found."""
    path = DATA_DIR / f"{system}_{T}K_gyration_props.npz"

    if not path.is_file():
        # Try to compute from trajectories
        compute_and_cache(system, T)

    if not path.is_file():
        return {}, {}
    data = np.load(path)
    frag_out = {}
    for frag in FRAGS:
        key = f"{frag}_Rg2"
        if key in data:
            arr = np.asarray(data[key], dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0)]
            frag_out[frag] = np.sqrt(arr)
        else:
            frag_out[frag] = np.array([])
    ang_out = {}
    for pair in ANGLE_PAIRS:
        key = f"angle_{pair[0]}_{pair[1]}"
        ang_out[pair] = np.asarray(data[key], dtype=float) if key in data else np.array([])
    return frag_out, ang_out


def _collect_all_data() -> tuple[
    Dict[str, Dict[str, Dict[int, np.ndarray]]],
    Dict[tuple[str, str], Dict[str, Dict[int, np.ndarray]]],
]:
    frag_all: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {frag: {} for frag in FRAGS}
    ang_all: Dict[tuple[str, str], Dict[str, Dict[int, np.ndarray]]] = {pair: {} for pair in ANGLE_PAIRS}
    for system in SYSTEMS:
        for frag in FRAGS:
            frag_all[frag].setdefault(system, {})
        for pair in ANGLE_PAIRS:
            ang_all[pair].setdefault(system, {})
        for T in TEMPS:
            frag_t, ang_t = _load_temp_npz(system, T)
            for frag in FRAGS:
                frag_all[frag][system][T] = frag_t.get(frag, np.array([]))
            for pair in ANGLE_PAIRS:
                ang_all[pair][system][T] = ang_t.get(pair, np.array([]))
    return frag_all, ang_all


def _range_from_values(values: Dict[str, Dict[int, np.ndarray]], q_low: float, q_high: float) -> tuple[float, float]:
    vals = []
    for system in SYSTEMS:
        for T in TEMPS:
            arr = np.asarray(values.get(system, {}).get(T, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(arr)
    if not vals:
        return 0.0, 1.0
    all_vals = np.concatenate(vals)
    if all_vals.size == 0:
        return 0.0, 1.0
    low = float(np.quantile(all_vals, q_low))
    high = float(np.quantile(all_vals, q_high))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = 0.03 * (high - low)
    return low - pad, high + pad


def _whole_tick_span(x_min: float, x_max: float, tick_count: int = 4) -> tuple[float, float, list[float]]:
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        ticks = np.linspace(0.0, 1.0, tick_count)
        return 0.0, 1.0, ticks.tolist()
    if tick_count < 2:
        base = float(np.floor(x_min))
        return base, base + 1.0, [base]
    start = int(np.floor(x_min))
    end = int(np.ceil(x_max))
    if end <= start:
        end = start + (tick_count - 1)
    span = end - start
    step_mod = tick_count - 1
    if span % step_mod != 0:
        span += step_mod - (span % step_mod)
        end = start + span
    step = max(1, span // step_mod)
    ticks = [float(start + step * i) for i in range(tick_count)]
    return float(start), float(end), ticks


def _whole_tick_formatter(value: float, _pos: int) -> str:
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:.2f}"


def _fragment_distribution_plot(
    frag_data: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    out_path: Path,
) -> None:
    fp.use_mpl_defaults()
    n_rows = len(FRAGS)
    n_cols = len(TEMPS)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(PANEL_W * n_cols, PANEL_H * n_rows),
        squeeze=False,
    )

    y_max = {frag: 1.0 for frag in FRAGS}
    x_lims = {frag: [np.inf, -np.inf] for frag in FRAGS}
    for frag in FRAGS:
        y_samples = []
        for system in SYSTEMS:
            for T in TEMPS:
                arr = np.asarray(frag_data.get(frag, {}).get(system, {}).get(T, []), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                xs, ys = _density_curve(arr)
                if xs.size == 0 or ys.size == 0:
                    continue
                y_samples.extend(ys.tolist())
                x_lims[frag][0] = min(x_lims[frag][0], float(xs.min()))
                x_lims[frag][1] = max(x_lims[frag][1], float(xs.max()))
        if y_samples:
            y_max[frag] = float(np.quantile(y_samples, 0.99) * 1.1)
        if not np.isfinite(x_lims[frag][0]) or not np.isfinite(x_lims[frag][1]):
            x_lims[frag] = [0.0, 1.0]

    for r, frag in enumerate(FRAGS):
        for c, T in enumerate(TEMPS):
            ax = axes[r][c]
            for system in SYSTEMS:
                arr = np.asarray(frag_data.get(frag, {}).get(system, {}).get(T, []), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                xs, ys = _density_curve(arr)
                if xs.size == 0 or ys.size == 0:
                    continue
                fill_x = np.concatenate(([xs[0]], xs, [xs[-1]]))
                fill_y = np.concatenate(([0.0], ys, [0.0]))
                ax.fill_between(fill_x, fill_y, color=fp.SYSTEM_COLORS[system], alpha=fp.FILL_ALPHA, linewidth=0)
                ax.plot(xs, ys, color=fp.SYSTEM_COLORS[system], lw=1.5, label=fp.SYSTEM_LABELS[system])

            ax.set_xlim(*x_lims[frag])
            ax.set_xticks(np.linspace(x_lims[frag][0], x_lims[frag][1], 3))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_ylim(0, y_max[frag])
            fp.style_axis(ax, show_left=(c == 0), show_bottom=(r == n_rows - 1))
            ax.tick_params(labelsize=fp.TICK_FONTSIZE)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            if r == 0:
                ax.set_title(f"{T} K", fontsize=fp.TITLE_FONTSIZE, pad=2)
            if c == 0:
                ax.set_ylabel("Prob. (a.u.)", fontsize=fp.LABEL_FONTSIZE)
            else:
                ax.tick_params(labelleft=False, left=False)

    handles = [plt.Line2D([], [], color=fp.SYSTEM_COLORS[s], lw=1.5, label=fp.SYSTEM_LABELS[s]) for s in SYSTEMS]
    if handles:
        axes[0][-1].legend(handles=handles, frameon=False, fontsize=fp.LEGEND_FONTSIZE, loc="upper right")

    plt.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        bottom=BOTTOM_MARGIN,
        top=TOP_MARGIN,
        wspace=WSPACE,
        hspace=HSPACE,
    )
    for r, frag in enumerate(FRAGS):
        pos = axes[r][0].get_position()
        row_center = pos.y0 + 0.5 * pos.height
        fig.text(
            ROW_LABEL_X,
            row_center,
            FRAG_LABELS.get(frag, frag),
            rotation=90,
            va="center",
            ha="center",
            fontsize=fp.LABEL_FONTSIZE,
        )
    fig.text(0.5, 0.04, r"$R_{\mathrm{g}}$ ($\mathrm{\AA}$)", ha="center", va="center", fontsize=fp.LABEL_FONTSIZE)
    fig.savefig(out_path, dpi=fp.FIG_DPI)
    plt.close(fig)


def _angle_distribution_plot(
    angle_data: Dict[tuple[str, str], Dict[str, Dict[int, np.ndarray]]],
    out_path: Path,
) -> None:
    fp.use_mpl_defaults()
    n_rows = len(ANGLE_PAIRS)
    n_cols = len(TEMPS)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(PANEL_W * n_cols, PANEL_H * n_rows),
        squeeze=False,
    )

    y_max = {pair: 1.0 for pair in ANGLE_PAIRS}
    for pair in ANGLE_PAIRS:
        y_samples = []
        for system in SYSTEMS:
            for T in TEMPS:
                arr = np.asarray(angle_data.get(pair, {}).get(system, {}).get(T, []), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                xs, ys = _density_curve(arr)
                if xs.size == 0 or ys.size == 0:
                    continue
                y_samples.extend(ys.tolist())
        if y_samples:
            y_max[pair] = float(np.quantile(y_samples, 0.99) * 1.1)

    for r, pair in enumerate(ANGLE_PAIRS):
        for c, T in enumerate(TEMPS):
            ax = axes[r][c]
            for system in SYSTEMS:
                arr = np.asarray(angle_data.get(pair, {}).get(system, {}).get(T, []), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                xs, ys = _density_curve(arr)
                if xs.size == 0 or ys.size == 0:
                    continue
                fill_x = np.concatenate(([xs[0]], xs, [xs[-1]]))
                fill_y = np.concatenate(([0.0], ys, [0.0]))
                ax.fill_between(fill_x, fill_y, color=fp.SYSTEM_COLORS[system], alpha=fp.FILL_ALPHA, linewidth=0)
                ax.plot(xs, ys, color=fp.SYSTEM_COLORS[system], lw=1.5, label=fp.SYSTEM_LABELS[system])

            ax.set_xlim(0.0, 180.0)
            ax.set_xticks([0, 90, 180])
            ax.set_ylim(0, y_max[pair])
            fp.style_axis(ax, show_left=(c == 0), show_bottom=(r == n_rows - 1))
            ax.tick_params(labelsize=fp.TICK_FONTSIZE)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            if r == 0:
                ax.set_title(f"{T} K", fontsize=fp.TITLE_FONTSIZE, pad=2)
            if c == 0:
                ax.set_ylabel("Prob. (a.u.)", fontsize=fp.LABEL_FONTSIZE)
            else:
                ax.tick_params(labelleft=False, left=False)

    handles = [plt.Line2D([], [], color=fp.SYSTEM_COLORS[s], lw=1.5, label=fp.SYSTEM_LABELS[s]) for s in SYSTEMS]
    if handles:
        axes[0][-1].legend(handles=handles, frameon=False, fontsize=fp.LEGEND_FONTSIZE, loc="upper right")

    plt.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        bottom=BOTTOM_MARGIN,
        top=TOP_MARGIN,
        wspace=WSPACE,
        hspace=HSPACE,
    )
    for r, pair in enumerate(ANGLE_PAIRS):
        pos = axes[r][0].get_position()
        row_center = pos.y0 + 0.5 * pos.height
        fig.text(
            ROW_LABEL_X,
            row_center,
            ANGLE_LABELS.get(pair, f"{pair[0]} vs {pair[1]}"),
            rotation=90,
            va="center",
            ha="center",
            fontsize=fp.LABEL_FONTSIZE,
        )
    fig.text(0.5, 0.04, "Angle (deg)", ha="center", va="center", fontsize=fp.LABEL_FONTSIZE)
    fig.savefig(out_path, dpi=fp.FIG_DPI)
    plt.close(fig)


def _joyplot(
    values: Dict[str, Dict[int, np.ndarray]],
    xlabel: str,
    out_path: Path,
    xlim: Tuple[float, float] | None = None,
    xticks: list[float] | None = None,
) -> None:
    fp.use_mpl_defaults()
    use_whole_tick_formatter = False
    temps = [
        T
        for T in JOY_TEMPS
        if any(np.asarray(values.get(s, {}).get(T, []), dtype=float).size > 0 for s in SYSTEMS)
    ]
    if not temps:
        return

    if xlim is None:
        x_min, x_max = _range_from_values(values, 0.01, 0.99)
        if xticks is None:
            x_min, x_max, xticks = _whole_tick_span(x_min, x_max, tick_count=4)
            use_whole_tick_formatter = True
    else:
        x_min, x_max = xlim
    xs = np.linspace(x_min, x_max, HIST_BINS * HIST_INTERP_FACTOR)

    n_ridges = len(temps)
    fig = plt.figure(figsize=(fp.ML_JOY_FIG_W, JOY_FIG_H))
    ax = fig.add_subplot(111)

    for i, T in enumerate(temps):
        offset = (n_ridges - 1 - i) * RIDGE_GAP
        curves = {}
        local_max = 0.0
        for system in SYSTEMS:
            arr = np.asarray(values.get(system, {}).get(T, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            ys = _density_curve_fixed(arr, xs, x_min, x_max)
            local_max = max(local_max, float(np.max(ys)))
            curves[system] = ys
        if local_max <= 0:
            continue
        scale = RIDGE_SCALE / local_max
        for system in SYSTEMS:
            ys = curves.get(system)
            if ys is None:
                continue
            y = offset + scale * ys
            ax.fill_between(xs, offset, y, color=fp.SYSTEM_COLORS[system], alpha=fp.FILL_ALPHA, linewidth=0)
            ax.plot(xs, y, color=fp.SYSTEM_COLORS[system], lw=1.2)
        ax.text(
            -0.03,
            offset + RIDGE_GAP * 0.1,
            f"{T} K",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=fp.TICK_FONTSIZE,
            clip_on=False,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5 * RIDGE_GAP, (n_ridges - 1) * RIDGE_GAP + RIDGE_GAP)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=fp.LABEL_FONTSIZE)
    ax.tick_params(
        direction="in",
        width=fp.TICK_WIDTH,
        length=fp.TICK_LENGTH,
        labelsize=fp.TICK_FONTSIZE,
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False,
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    for spine in ax.spines.values():
        spine.set_linewidth(fp.AXES_LINEWIDTH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if xticks is not None:
        ax.set_xticks(xticks)
        if use_whole_tick_formatter:
            ax.xaxis.set_major_formatter(FuncFormatter(_whole_tick_formatter))
    plt.tight_layout()
    fig.savefig(out_path, dpi=fp.FIG_DPI)
    plt.close(fig)


def _save_system_key(out_path: Path) -> None:
    fp.use_mpl_defaults()
    fig = plt.figure(figsize=(1.6, 0.45))
    ax = fig.add_subplot(111)
    handles = [
        plt.Line2D([], [], color=fp.SYSTEM_COLORS[system], lw=1.5, label=fp.SYSTEM_LABELS[system])
        for system in SYSTEMS
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=fp.LEGEND_FONTSIZE,
        loc="center",
        ncol=len(handles),
        handlelength=1.4,
        columnspacing=0.8,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=fp.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frag_data, angle_data = _collect_all_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _fragment_distribution_plot(frag_data, OUTPUT_DIR / "fragment_rg_distributions.svg")
    _angle_distribution_plot(angle_data, OUTPUT_DIR / "fragment_angle_distributions.svg")

    for frag in FRAGS:
        _joyplot(
            frag_data[frag],
            xlabel=r"$R_{\mathrm{g}}$ ($\mathrm{\AA}$)",
            out_path=OUTPUT_DIR / f"fragment_{frag}_rg_joy.svg",
        )

    for pair in ANGLE_PAIRS:
        _joyplot(
            angle_data[pair],
            xlabel="Angle (deg)",
            out_path=OUTPUT_DIR / f"fragment_{pair[0]}_{pair[1]}_theta_joy.svg",
            xlim=(0.0, 180.0),
            xticks=[0.0, 60.0, 120.0, 180.0],
        )
    _save_system_key(OUTPUT_DIR / "fragment_joyplot_key.svg")


if __name__ == "__main__":
    main()

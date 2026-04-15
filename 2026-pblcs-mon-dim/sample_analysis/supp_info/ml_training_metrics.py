#!/usr/bin/env python3
"""
Generate supplemental ML training metrics plots:
- Distribution of std-only features (phi_i_std) as a ridgeline plot
- Parity plot for the std-only regressor using cross-validated predictions

METHODOLOGY
-----------
Feature extraction from MD trajectories (used to produce the aggregated
features consumed here):
  1. Load GROMACS .tpr/.xtc trajectory via MDAnalysis.
  2. Detect rotatable bonds using the SMARTS pattern
       [!#1]~[!$(*#*)&!D1:1]-,=;!@[!$(*#*)&!D1:2]~[!#1]
     applied to the first fragment (molecule) via RDKit.
  3. For each frame (every ``skip``-th), compute all dihedral angles
     (degrees, wrapped to (-180, 180]) and the radius of gyration (Rg)
     for every molecule in the box.
  4. Concatenate results across all temperatures and systems into a
     raw features.csv.
  5. Group molecules into clusters of N=256, compute circular mean and
     std of each dihedral angle per cluster, and average Rg per cluster,
     producing the aggregated features CSV.

The script checks for cached aggregated CSV and model files first, and
computes from raw trajectories if not found.
"""

# ---------------------------------------------------------------------------
# DATA / SIMULATION PATHS
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import hashlib
import os
import sys

DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/paper_1_figures")
SIMULATION_ROOT = os.environ.get("SIMULATION_ROOT", "/scratch/gpfs/WEBB/jv6139")
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ---- Figure params ----
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

DEFAULT_FEATURES = Path(DATA_ROOT) / (
    "polished_figures/ml_analysis_fig/regressor_std_only"
    "/overall_N_256_aggregated_features.csv"
)
DEFAULT_RAW_FEATURES = Path(DATA_ROOT) / (
    "old_ml_conformational_analysis/ML_analysis/previous_all/"
    "feature_extraction/features.csv"
)
DEFAULT_MODEL = Path(DATA_ROOT) / (
    "polished_figures/ml_analysis_fig/regressor_std_only"
    "/overall_N_256_std_only.joblib"
)
DEFAULT_PREFIX = "std_only"
RANDOM_STATE = 42
CLUSTER_SIZE = 256

HIST_BINS = fp.ML_HIST_BINS
HIST_SMOOTH_KERNEL = np.array(fp.ML_HIST_SMOOTH_KERNEL, dtype=float)
HIST_INTERP_FACTOR = fp.ML_HIST_INTERP_FACTOR
RIDGE_GAP = fp.ML_RIDGE_GAP
RIDGE_SCALE = fp.ML_RIDGE_SCALE

# ---------------------------------------------------------------------------
# Trajectory-level configuration for on-the-fly feature extraction
# ---------------------------------------------------------------------------
TRAJECTORY_SYSTEMS = {
    "P_monomer": (
        os.path.join(SIMULATION_ROOT, "N_monomer/simulations/tREM/{T}K/trem_gpu.tpr"),
        os.path.join(SIMULATION_ROOT, "N_monomer/simulations/tREM/{T}K/trem_gpu.xtc"),
    ),
    "M_monomer": (
        os.path.join(SIMULATION_ROOT, "M_monomer/simulations/tREM/{T}K/trem_gpu.tpr"),
        os.path.join(SIMULATION_ROOT, "M_monomer/simulations/tREM/{T}K/trem_gpu.xtc"),
    ),
}
TRAJECTORY_TEMPS = list(range(260, 441, 5))
TRAJECTORY_SKIP = 50


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION FROM TRAJECTORIES
# ---------------------------------------------------------------------------
# Adapted from:
#   paper_1_figures/old_ml_conformational_analysis/ML_analysis/
#       previous_all/feature_extraction/extract_features.py
# ---------------------------------------------------------------------------

_TORSION_CACHE: dict[str, list[tuple[int, int, int, int]]] = {}
_TORSIONS_GLOBAL: list[tuple[int, int, int, int]] = []
_FRAG_TORSION_QUADS_GLOBAL: dict[int, list[tuple[int, int, int, int]]] = {}
_SYSTEM_NAME_GLOBAL: str = ""
_TEMPERATURE_GLOBAL: int = 0
_univ = None


def rotatable_torsions_from_fragment(frag):
    """
    Detect rotatable bonds inside one molecule fragment using the SMARTS:

        [!#1]~[!$(*#*)&!D1:1]-,=;!@[!$(*#*)&!D1:2]~[!#1]

    Returns a sorted list of (a, i, j, b) index tuples (local to the
    fragment, with i < j).
    """
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(
        "[!#1]~[!$(*#*)&!D1:1]-,=;!@[!$(*#*)&!D1:2]~[!#1]"
    )
    rdkit_mol = frag.convert_to("RDKIT")

    torsion_set: set[tuple[int, int, int, int]] = set()
    for match in rdkit_mol.GetSubstructMatches(pattern, uniquify=True):
        idx1 = next(
            k for k, atom in enumerate(pattern.GetAtoms())
            if atom.HasProp("molAtomMapNumber")
            and atom.GetProp("molAtomMapNumber") == "1"
        )
        idx2 = next(
            k for k, atom in enumerate(pattern.GetAtoms())
            if atom.HasProp("molAtomMapNumber")
            and atom.GetProp("molAtomMapNumber") == "2"
        )

        i_atom, j_atom = match[idx1], match[idx2]
        i, j = (i_atom, j_atom) if i_atom < j_atom else (j_atom, i_atom)

        ai = rdkit_mol.GetAtomWithIdx(i)
        aj = rdkit_mol.GetAtomWithIdx(j)

        neigh_i = sorted(
            n.GetIdx() for n in ai.GetNeighbors()
            if n.GetIdx() != j and n.GetAtomicNum() != 1
        )
        neigh_j = sorted(
            n.GetIdx() for n in aj.GetNeighbors()
            if n.GetIdx() != i and n.GetAtomicNum() != 1
        )
        if not (neigh_i and neigh_j):
            continue

        a = neigh_i[0]
        b = neigh_j[0]
        torsion_set.add((a, i, j, b))

    return sorted(torsion_set)


def _init_universe(top: str, traj: str):
    """Initialise a global Universe in each worker and apply unwrap."""
    import MDAnalysis as mda
    from MDAnalysis.transformations import unwrap as mda_unwrap

    global _univ
    _univ = mda.Universe(top, traj)
    _univ.trajectory.add_transformations(mda_unwrap(_univ.atoms))


def _process_frame_global(frame_no: int) -> list[dict]:
    """
    Process a single frame across all fragments, using module-level globals.
    Computes dihedral angles (degrees, wrapped to (-180, 180]) and Rg for
    every molecule (residue) in the box.
    """
    from MDAnalysis.lib.distances import calc_dihedrals

    _univ.trajectory[frame_no]
    rows = []
    for frag_idx, quad_list in _FRAG_TORSION_QUADS_GLOBAL.items():
        pos = _univ.atoms.positions
        p0 = np.array([pos[i0] for (i0, _, _, _) in quad_list])
        p1 = np.array([pos[i1] for (_, i1, _, _) in quad_list])
        p2 = np.array([pos[i2] for (_, _, i2, _) in quad_list])
        p3 = np.array([pos[i3] for (_, _, _, i3) in quad_list])
        angles = np.degrees(calc_dihedrals(p0, p1, p2, p3))
        phi = ((angles + 180.0) % 360.0) - 180.0

        frag = _univ.residues[frag_idx].atoms
        com = frag.center_of_mass()
        r = frag.positions - com
        G = np.einsum("im,in->mn", r, r) / r.shape[0]
        rg = np.sqrt(np.sum(np.linalg.eigvalsh(G)))

        row = {f"phi_{k}": float(phi[k]) for k in range(len(_TORSIONS_GLOBAL))}
        row.update({
            "T": _TEMPERATURE_GLOBAL,
            "system": _SYSTEM_NAME_GLOBAL,
            "fragment": frag_idx,
            "frame": frame_no,
            "R_g": float(rg),
        })
        rows.append(row)
    return rows


def extract_features_for_traj(
    tpr: str,
    xtc: str,
    system_name: str,
    T: int,
    skip: int = 50,
    n_cpus: int = 1,
) -> pd.DataFrame:
    """
    Build a DataFrame of raw dihedral angles (degrees) + Rg + temperature
    for every fragment and every ``skip``-th frame.
    """
    import MDAnalysis as mda
    from multiprocessing import Pool

    global _TORSIONS_GLOBAL, _FRAG_TORSION_QUADS_GLOBAL
    global _SYSTEM_NAME_GLOBAL, _TEMPERATURE_GLOBAL

    u = mda.Universe(tpr, xtc)
    print(f"[{system_name} @ {T}K] Loaded trajectory", flush=True)

    if system_name in _TORSION_CACHE:
        torsions = _TORSION_CACHE[system_name]
    else:
        frag0 = u.atoms.fragments[0]
        torsions = rotatable_torsions_from_fragment(frag0)
        _TORSION_CACHE[system_name] = torsions

    frag_torsion_quads: dict[int, list[tuple[int, int, int, int]]] = {}
    for frag_idx, res in enumerate(u.residues):
        frag_torsion_quads[frag_idx] = [
            (
                res.atoms[a].index,
                res.atoms[i].index,
                res.atoms[j].index,
                res.atoms[b].index,
            )
            for (a, i, j, b) in torsions
        ]

    _TORSIONS_GLOBAL = torsions
    _FRAG_TORSION_QUADS_GLOBAL = frag_torsion_quads
    _SYSTEM_NAME_GLOBAL = system_name
    _TEMPERATURE_GLOBAL = T

    frame_indices = list(range(0, len(u.trajectory), skip))

    with Pool(
        processes=min(n_cpus, len(frame_indices)),
        initializer=_init_universe,
        initargs=(tpr, xtc),
    ) as pool:
        frame_results = pool.map(_process_frame_global, frame_indices)

    data = [row for frag_list in frame_results for row in frag_list]
    return pd.DataFrame(data)


def extract_all_features(
    systems: dict[str, tuple[str, str]] | None = None,
    temps: list[int] | None = None,
    skip: int = TRAJECTORY_SKIP,
    n_cpus: int = 1,
) -> pd.DataFrame:
    """Run feature extraction across all systems and temperatures."""
    if systems is None:
        systems = TRAJECTORY_SYSTEMS
    if temps is None:
        temps = TRAJECTORY_TEMPS

    dfs: list[pd.DataFrame] = []
    for sys_name, (tpr_tpl, xtc_tpl) in systems.items():
        for T in temps:
            tpr = tpr_tpl.format(T=T)
            xtc = xtc_tpl.format(T=T)
            if not (Path(tpr).is_file() and Path(xtc).is_file()):
                print(f"  [SKIP] Missing files for {sys_name} @ {T}K", flush=True)
                continue
            print(f"Processing {sys_name} at {T} K ...", flush=True)
            dfs.append(
                extract_features_for_traj(tpr, xtc, sys_name, T, skip=skip, n_cpus=n_cpus)
            )

    if not dfs:
        raise RuntimeError("No trajectory data found. Check TRAJECTORY_SYSTEMS paths.")
    return pd.concat(dfs, ignore_index=True)


def load_or_compute_raw_features(
    features_csv: Path,
    n_cpus: int = 1,
) -> pd.DataFrame:
    """
    Load cached raw features.csv if it exists; otherwise compute from
    trajectories and save.
    """
    if features_csv.is_file():
        print(f"Loading cached raw features from {features_csv}", flush=True)
        return pd.read_csv(features_csv)

    print(
        f"Cached raw features not found at {features_csv}. "
        "Computing from trajectories ...",
        flush=True,
    )
    df = extract_all_features(n_cpus=n_cpus)
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_csv, index=False)
    print(f"Saved raw features ({df.shape}) to {features_csv}", flush=True)
    return df


# ---------------------------------------------------------------------------
# AGGREGATION: raw features -> clustered circular-std features
# ---------------------------------------------------------------------------

def _deterministic_seed(*parts: object) -> int:
    key = "_".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32 - 1)


def _circular_mean_std(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return 0.0, 0.0
    max_abs = np.nanmax(np.abs(vals))
    if max_abs > np.pi + 1e-6:
        vals = np.deg2rad(vals)
    sin_mean = np.sin(vals).mean()
    cos_mean = np.cos(vals).mean()
    circ_mean = float(np.arctan2(sin_mean, cos_mean))
    R = np.sqrt(cos_mean**2 + sin_mean**2)
    R = float(np.clip(R, 1e-12, 1.0))
    circ_std = float(np.sqrt(-2.0 * np.log(R)))
    return circ_mean, circ_std


def _clean_regression_data(
    df_sub: pd.DataFrame,
    N: int,
    phi_cols: list[str],
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    if df_sub.empty:
        return pd.DataFrame()
    feats = df_sub[phi_cols].copy()
    if "R_g" not in df_sub.columns:
        raise KeyError("Column 'R_g' not found in input DataFrame.")
    rg = df_sub["R_g"].copy()
    if len(feats) < N:
        return pd.DataFrame()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(feats))
    feats = feats.iloc[perm].reset_index(drop=True)
    rg = rg.iloc[perm].reset_index(drop=True)
    remainder = len(feats) % N
    if remainder:
        keep = len(feats) - remainder
        feats = feats.iloc[:keep].reset_index(drop=True)
        rg = rg.iloc[:keep].reset_index(drop=True)
    if feats.empty:
        return pd.DataFrame()
    cluster_ids = np.arange(len(feats)) // N
    n_clusters = len(feats) // N
    records = []
    for cid in range(n_clusters):
        block = feats.iloc[cid * N : (cid + 1) * N]
        stats = {}
        for col in phi_cols:
            vals = block[col].to_numpy(dtype=float, copy=False)
            mu, sig = _circular_mean_std(vals)
            stats[f"{col}_mean"] = mu
            stats[f"{col}_std"] = sig
        records.append(stats)
    grouped_features = pd.DataFrame(records)
    y_mu = rg.groupby(cluster_ids).mean().rename("y").reset_index(drop=True)
    return pd.concat([grouped_features, y_mu], axis=1)


def _clean_by_frame(
    df_all: pd.DataFrame,
    N: int,
    phi_cols: list[str],
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    required = {"T", "system", "frame"}
    if not required.issubset(df_all.columns):
        missing = ", ".join(sorted(required - set(df_all.columns)))
        raise KeyError(
            f"Expected columns {required} in features.csv; missing: {missing}"
        )
    parts = []
    for (T, sys_name, frame), g in df_all.groupby(
        ["T", "system", "frame"], sort=True
    ):
        if g.empty:
            continue
        frame_seed = _deterministic_seed("frame", sys_name, T, frame, N, seed)
        cg = _clean_regression_data(g, N, phi_cols, seed=frame_seed)
        if not cg.empty:
            cg["T"] = T
            cg["system"] = sys_name
            cg["frame"] = frame
            parts.append(cg)
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame()


def load_or_compute_aggregated_features(
    aggregated_csv: Path,
    raw_csv: Path,
    N: int = CLUSTER_SIZE,
    n_cpus: int = 1,
) -> pd.DataFrame:
    """
    Load cached aggregated features if available. Otherwise, load (or
    compute) raw features, aggregate into N-molecule clusters with
    circular std/mean, and save.
    """
    if aggregated_csv.is_file():
        print(f"Loading cached aggregated features from {aggregated_csv}", flush=True)
        return pd.read_csv(aggregated_csv)

    print(
        f"Aggregated features not found at {aggregated_csv}. "
        "Building from raw features ...",
        flush=True,
    )
    raw_df = load_or_compute_raw_features(raw_csv, n_cpus=n_cpus)
    phi_cols = [c for c in raw_df.columns if c.startswith("phi_")]
    if not phi_cols:
        raise ValueError("No phi_* columns in raw features.")
    agg_df = _clean_by_frame(raw_df, N, phi_cols, seed=RANDOM_STATE)
    if agg_df.empty:
        raise ValueError(f"Aggregated dataframe is empty for N={N}.")
    aggregated_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(aggregated_csv, index=False)
    print(f"Saved aggregated features ({agg_df.shape}) to {aggregated_csv}", flush=True)
    return agg_df


# ---------------------------------------------------------------------------
# ANALYSIS HELPERS (unchanged from original)
# ---------------------------------------------------------------------------

def _std_feature_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("phi_") and c.endswith("_std")]


def _phi_index(name: str) -> int | str:
    try:
        return int(name.split("_")[1])
    except Exception:
        return name


def _phi_label(name: str) -> str:
    base = name.replace("_std", "")
    try:
        idx = int(base.split("_")[1])
        return rf"$\sigma_{{\phi_{{{idx}}}}}$"
    except Exception:
        return base


def _range_from_values(values: np.ndarray, q_low: float, q_high: float) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 0.0, 1.0
    low = float(np.quantile(vals, q_low))
    high = float(np.quantile(vals, q_high))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(vals))
        high = float(np.nanmax(vals))
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


def _density_curve_fixed(values: np.ndarray, xs: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(xs)
    counts, edges = np.histogram(values, bins=HIST_BINS, range=(x_min, x_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(counts, HIST_SMOOTH_KERNEL, mode="same")
    return np.interp(xs, centers, smooth)


def _kfold_indices(n: int, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    out = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        out.append((train_idx, val_idx))
        current = stop
    return out


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_y = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_y) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _split_validation_mask(
    df: pd.DataFrame,
    val_frac: float,
    group_cols: list[str],
    random_state: int,
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    use_groups = bool(group_cols) and all(c in df.columns for c in group_cols)
    if use_groups:
        group_keys = df[group_cols].astype(str).agg("|".join, axis=1).to_numpy()
        unique_groups = np.unique(group_keys)
        rng.shuffle(unique_groups)
        n_val = max(1, int(round(val_frac * len(unique_groups))))
        val_groups = set(unique_groups[:n_val])
        val_mask = np.array([g in val_groups for g in group_keys], dtype=bool)
        if not val_mask.all() and val_mask.any():
            return val_mask
    n_val = max(1, int(round(val_frac * len(df))))
    idx = rng.permutation(len(df))
    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[idx[:n_val]] = True
    return val_mask


# ---------------------------------------------------------------------------
# PLOTTING (unchanged from original)
# ---------------------------------------------------------------------------

def plot_std_feature_distributions(df: pd.DataFrame, std_cols: list[str], out_path: Path) -> None:
    fp.use_mpl_defaults()
    std_cols = sorted(std_cols, key=_phi_index)
    all_vals = df[std_cols].to_numpy(dtype=float).ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    y_min, y_max = _range_from_values(all_vals, 0.01, 0.99)
    y_min, y_max, y_ticks = _whole_tick_span(y_min, y_max, tick_count=4)

    systems = [s for s in fp.SYSTEM_COLORS if s in df["system"].unique()]
    if not systems:
        systems = sorted(df["system"].unique())

    max_fig_w = 6.0
    per_feature_w = 0.45
    fig_h = 2.4
    if len(std_cols) > 1:
        target_per_panel = int(np.ceil(len(std_cols) / 2))
    else:
        target_per_panel = len(std_cols)
    chunks = [std_cols[:target_per_panel]]
    remainder = std_cols[target_per_panel:]
    if remainder:
        if len(remainder) < target_per_panel:
            remainder = remainder + [None] * (target_per_panel - len(remainder))
        chunks.append(remainder)

    handles = [
        Patch(
            facecolor=fp.SYSTEM_COLORS.get(system, "#4d4d4d"),
            edgecolor="none",
            alpha=0.6,
            label=fp.SYSTEM_LABELS.get(system, system),
        )
        for system in systems
    ]

    for idx, chunk in enumerate(chunks, start=1):
        n_features = len(chunk)
        fig_w = min(max_fig_w, max(2.5, per_feature_w * target_per_panel))
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)

        positions = np.arange(1, n_features + 1)
        for pos, col in zip(positions, chunk):
            if col is None:
                continue
            for system in systems:
                arr = np.asarray(df.loc[df["system"] == system, col], dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                parts = ax.violinplot(
                    [arr],
                    positions=[pos],
                    vert=True,
                    widths=0.8,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(fp.SYSTEM_COLORS.get(system, "#4d4d4d"))
                    body.set_edgecolor("black")
                    body.set_linewidth(0.4)
                    body.set_alpha(0.6)

        ax.set_xlim(0.5, n_features + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [_phi_label(col) if col is not None else "" for col in chunk],
            rotation=45,
            ha="right",
            fontsize=fp.TICK_FONTSIZE,
        )
        ax.set_yticks(y_ticks)
        ax.set_ylabel(r"circular std $\sigma_{\phi}$ (rad)", fontsize=fp.LABEL_FONTSIZE)
        ax.vlines(positions, y_min, y_max, colors="#d0d0d0", linewidth=0.6, zorder=0)
        ax.tick_params(
            direction="in",
            width=fp.TICK_WIDTH,
            length=fp.TICK_LENGTH,
            labelsize=fp.TICK_FONTSIZE,
            bottom=True,
            top=False,
            labelbottom=True,
            labeltop=False,
            right=False,
            labelright=False,
        )
        ax.yaxis.set_ticks_position("left")
        for spine in ax.spines.values():
            spine.set_linewidth(fp.AXES_LINEWIDTH)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if handles:
            ax.legend(handles=handles, frameon=False, fontsize=fp.LEGEND_FONTSIZE, loc="upper right")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(chunks) == 1:
            panel_path = out_path
        else:
            panel_path = out_path.with_name(f"{out_path.stem}_part{idx}{out_path.suffix}")
        fig.savefig(panel_path, dpi=fp.FIG_DPI)
        plt.close(fig)


def plot_parity(
    df: pd.DataFrame,
    std_cols: list[str],
    model_path: Path,
    out_path: Path,
    n_jobs: int,
    random_state: int,
    val_frac: float,
    val_group_cols: list[str],
) -> None:
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise ImportError("xgboost is required to load the saved model for parity plots.") from exc

    import joblib

    fp.use_mpl_defaults()

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = joblib.load(model_path)
    if hasattr(model, "set_params"):
        model.set_params(n_jobs=n_jobs)

    X = df[std_cols].to_numpy(dtype=float)
    y_true = df["y"].to_numpy(dtype=float)
    val_mask = _split_validation_mask(df, val_frac, val_group_cols, random_state)
    X_val = X[val_mask]
    y_val = y_true[val_mask]
    if X_val.size == 0:
        raise ValueError("Validation split is empty; adjust val_frac or group columns.")
    y_pred = model.predict(X_val)

    r2 = _r2_score(y_val, y_pred)
    metrics = {
        "val_r2": r2,
        "val_n": int(len(y_val)),
    }
    metrics_path = out_path.with_name(f"{out_path.stem}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    val_systems = df.loc[val_mask, "system"].to_numpy()
    for system in sorted(df["system"].unique()):
        mask = val_systems == system
        if not np.any(mask):
            continue
        color = fp.SYSTEM_COLORS.get(system, "#4d4d4d")
        if system == "M_monomer":
            label = r"$\mathrm{M_{1}}$"
        elif system == "P_monomer":
            label = r"$\mathrm{P_{1}}$"
        else:
            label = fp.SYSTEM_LABELS.get(system, system)
        ax.scatter(
            y_val[mask],
            y_pred[mask],
            s=fp.MARKER_SIZE ** 2,
            alpha=0.6,
            color=color,
            edgecolors="none",
            label=label,
        )

    x_min, x_max = 7.5, 8.5
    ticks = [7.5, 8.0, 8.5]
    ax.plot([x_min, x_max], [x_min, x_max], "k--", linewidth=1.2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"Actual $R_g$ ($\AA$)", fontsize=fp.LABEL_FONTSIZE)
    ax.set_ylabel(r"Predicted $R_g$ ($\AA$)", fontsize=fp.LABEL_FONTSIZE)
    ax.text(
        0.04,
        0.96,
        f"$R^2$ (val)={r2:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fp.TICK_FONTSIZE,
    )
    ax.legend(frameon=False, fontsize=fp.LEGEND_FONTSIZE, loc="lower right")
    fp.style_axis(ax)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=fp.FIG_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplemental ML training metric plots")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--raw-features", type=Path, default=DEFAULT_RAW_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--val-group-cols", type=str, default="system,T,frame")
    args = parser.parse_args()

    df = load_or_compute_aggregated_features(
        args.features,
        args.raw_features,
        N=CLUSTER_SIZE,
        n_cpus=args.cores,
    )
    std_cols = _std_feature_cols(df.columns.tolist())
    if not std_cols:
        raise ValueError("No phi_*_std columns found in aggregated features CSV.")

    out_dir = args.out_dir
    if not args.skip_distribution:
        out_path = out_dir / f"{args.prefix}_std_feature_distributions.png"
        plot_std_feature_distributions(df, std_cols, out_path)

    if not args.skip_parity:
        out_path = out_dir / f"{args.prefix}_parity.png"
        group_cols = [c.strip() for c in args.val_group_cols.split(",") if c.strip()]
        plot_parity(
            df,
            std_cols,
            args.model,
            out_path,
            n_jobs=args.cores,
            random_state=RANDOM_STATE,
            val_frac=args.val_frac,
            val_group_cols=group_cols,
        )


if __name__ == "__main__":
    main()

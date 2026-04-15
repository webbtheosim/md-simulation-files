#!/usr/bin/env python3
"""
Train an Rg regressor using ONLY circular std features (N=256) and generate
a std-only SHAP bar plot.

This mirrors the overall pipeline in rg_regressor_overall_256.py, but keeps
only phi_*_std features for training/SHAP.

METHODOLOGY
-----------
Feature extraction from MD trajectories:
  1. Load GROMACS .tpr/.xtc trajectory via MDAnalysis.
  2. Detect rotatable bonds using the SMARTS pattern
       [!#1]~[!$(*#*)&!D1:1]-,=;!@[!$(*#*)&!D1:2]~[!#1]
     applied to the first fragment (molecule) via RDKit.
  3. For each frame (every `skip`-th), compute all dihedral angles
     (degrees, wrapped to (-180, 180]) and the radius of gyration (Rg)
     for every molecule in the box.
  4. Concatenate results across all temperatures and systems into a
     single features.csv.

The script checks for cached features.csv first and only runs the
extraction when the cache is missing.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# DATA / SIMULATION PATHS
# ---------------------------------------------------------------------------
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/paper_1_figures")
SIMULATION_ROOT = os.environ.get("SIMULATION_ROOT", "/scratch/gpfs/WEBB/jv6139")

import argparse
import sys
import hashlib
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Figure params ----
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402
import _patch_shap  # noqa: F401,E402  # fix shap/xgboost base_score compat

# ---- Defaults ----
DEFAULT_FEATURES = Path(DATA_ROOT) / "old_ml_conformational_analysis" / "ML_analysis" / "previous_wrapped_distribution" / "feature_extraction" / "features.csv"
DEFAULT_OVERALL_DIR = Path(DATA_ROOT) / "ml_analysis" / "overall"
DEFAULT_PREFIX = "overall_N_256"
RANDOM_STATE = 42
SHAP_SAMPLE_MAX = 4000
SHAP_BACKGROUND = 1000

XGB_PARAM_GRID = {
    "n_estimators": [400, 600, 800],
    "learning_rate": [0.03, 0.05],
    "max_depth": [5, 7],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
}

# ---------------------------------------------------------------------------
# Trajectory-level configuration for on-the-fly feature extraction
# ---------------------------------------------------------------------------
# Template paths use {T} as placeholder for temperature in Kelvin.
# These point to the original simulation data used in the upstream pipeline
# (paper_1_figures/old_ml_conformational_analysis).
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
TRAJECTORY_SKIP = 50  # analyse every N-th frame

# ---------------------------------------------------------------------------
# FEATURE EXTRACTION FROM TRAJECTORIES
# ---------------------------------------------------------------------------
# The functions below are adapted from the upstream script:
#   paper_1_figures/old_ml_conformational_analysis/ML_analysis/
#       previous_all/feature_extraction/extract_features.py
#
# They use MDAnalysis + RDKit to detect rotatable bonds via SMARTS,
# compute per-molecule dihedral angles and radius of gyration (Rg),
# and return a tidy DataFrame.
# ---------------------------------------------------------------------------

# Global torsion cache so both monomers share the same ordering
_TORSION_CACHE: dict[str, list[tuple[int, int, int, int]]] = {}

# Globals for parallel frame processing (set before Pool.map)
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

        # Radius of gyration from eigenvalues of the gyration tensor
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

    # Build torsion list once from the first fragment (reference molecule)
    if system_name in _TORSION_CACHE:
        torsions = _TORSION_CACHE[system_name]
    else:
        frag0 = u.atoms.fragments[0]
        torsions = rotatable_torsions_from_fragment(frag0)
        _TORSION_CACHE[system_name] = torsions

    # Precompute global atom indices for each residue's torsions
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
    """
    Run feature extraction across all systems and temperatures.
    Returns a single concatenated DataFrame.
    """
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


def load_or_compute_features(
    features_csv: Path,
    n_cpus: int = 1,
) -> pd.DataFrame:
    """
    Load cached features.csv if it exists; otherwise compute from
    trajectories and save.
    """
    if features_csv.is_file():
        print(f"Loading cached features from {features_csv}", flush=True)
        return pd.read_csv(features_csv)

    print(
        f"Cached features not found at {features_csv}. "
        "Computing from trajectories ...",
        flush=True,
    )
    df = extract_all_features(n_cpus=n_cpus)
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_csv, index=False)
    print(f"Saved features ({df.shape}) to {features_csv}", flush=True)
    return df


# ---------------------------------------------------------------------------
# ANALYSIS HELPERS (unchanged from original)
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


def _clean_regression_data(df_sub: pd.DataFrame, N: int, phi_cols: list[str], seed: int = RANDOM_STATE) -> pd.DataFrame:
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


def _clean_by_frame(df_all: pd.DataFrame, N: int, phi_cols: list[str], seed: int = RANDOM_STATE) -> pd.DataFrame:
    required = {"T", "system", "frame"}
    if not required.issubset(df_all.columns):
        missing = ", ".join(sorted(required - set(df_all.columns)))
        raise KeyError(f"Expected columns {required} in features.csv; missing: {missing}")
    parts = []
    for (T, sys_name, frame), g in df_all.groupby(["T", "system", "frame"], sort=True):
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


def _std_feature_cols(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c.startswith("phi_") and c.endswith("_std")]


def _phi_index(name: str) -> int | str:
    try:
        return int(name.split("_")[1])
    except Exception:
        return name


def _phi_label(name: str) -> str:
    base = name.replace("_std", "")
    if base == "other_phi":
        return r"other $\phi$"
    try:
        idx = int(base.split("_")[1])
        return rf"$\phi_{{{idx}}}$"
    except Exception:
        return base


def _load_std_only(features_csv: Path, N: int, n_cpus: int = 1) -> Tuple[pd.DataFrame, list[str]]:
    df = load_or_compute_features(features_csv, n_cpus=n_cpus)
    phi_cols = [c for c in df.columns if c.startswith("phi_")]
    if not phi_cols:
        raise ValueError("No phi_* columns found in features CSV.")
    clean = _clean_by_frame(df, N, phi_cols, seed=RANDOM_STATE)
    if clean.empty:
        raise ValueError(f"Aggregated dataframe is empty for N={N}.")
    std_cols = _std_feature_cols(clean.columns)
    if not std_cols:
        raise ValueError("No phi_*_std columns found after aggregation.")
    return clean, std_cols


def _train_regressor(df: pd.DataFrame, std_cols: list[str], out_path: Path, n_jobs: int) -> object:
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    import xgboost as xgb

    X = df[std_cols].values
    y = df["y"].values
    if len(df) < 3:
        raise ValueError(f"Not enough samples for training (n={len(df)}).")
    kf = KFold(n_splits=min(3, max(2, len(df))), shuffle=True, random_state=RANDOM_STATE)
    base_params = {
        "random_state": RANDOM_STATE,
        "n_jobs": n_jobs,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }

    grid = GridSearchCV(
        estimator=xgb.XGBRegressor(**base_params),
        param_grid=XGB_PARAM_GRID,
        scoring="r2",
        cv=kf,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    grid.fit(X, y)
    best_model = grid.best_estimator_

    fold_r2, fold_rmse = [], []
    y_pred_cv = np.zeros_like(y, dtype=float)
    for tr, va in kf.split(X, y):
        clf = xgb.XGBRegressor(**{**base_params, **grid.best_params_})
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[va])
        y_pred_cv[va] = yp
        fold_r2.append(float(r2_score(y[va], yp)))
        fold_rmse.append(float(np.sqrt(mean_squared_error(y[va], yp))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_path)

    metrics = {
        "mean_r2": float(np.mean(fold_r2)) if fold_r2 else float("nan"),
        "std_r2": float(np.std(fold_r2, ddof=1)) if len(fold_r2) > 1 else 0.0,
        "mean_rmse": float(np.mean(fold_rmse)) if fold_rmse else float("nan"),
        "std_rmse": float(np.std(fold_rmse, ddof=1)) if len(fold_rmse) > 1 else 0.0,
    }
    return best_model, metrics


def _compute_shap_std(model: object, X: pd.DataFrame, out_dir: Path, prefix: str) -> Path:
    import shap

    rng = np.random.RandomState(RANDOM_STATE)
    if len(X) > SHAP_SAMPLE_MAX:
        idx = rng.choice(len(X), size=SHAP_SAMPLE_MAX, replace=False)
        Xs = X.iloc[idx].reset_index(drop=True)
    else:
        Xs = X.reset_index(drop=True)

    bg_n = min(SHAP_BACKGROUND, len(Xs))
    bg_idx = rng.choice(len(Xs), size=bg_n, replace=False)
    background = Xs.iloc[bg_idx]

    explainer = shap.TreeExplainer(model, background)
    exp = explainer(Xs)
    shap_vals = np.asarray(exp.values)

    vals = np.abs(shap_vals)
    n = vals.shape[0]
    ddof = 1 if n > 1 else 0
    mean_imp = vals.mean(axis=0)
    std_imp = vals.std(axis=0, ddof=ddof)
    sem_imp = std_imp / np.sqrt(n)

    std_df = pd.DataFrame(
        {
            "mean_abs_shap": mean_imp,
            "sem_abs_shap": sem_imp,
        },
        index=X.columns,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    std_csv = out_dir / f"{prefix}_shap_std_importance.csv"
    std_df.to_csv(std_csv)
    np.save(out_dir / f"{prefix}_shap_values.npy", shap_vals)
    return std_csv


def _plot_std_bar(
    std_csv: Path,
    out_svg: Path,
    *,
    title: str = "",
    color: str = fp.ML_STD_BAR_COLOR,
) -> None:
    fp.use_mpl_defaults()
    df = pd.read_csv(std_csv, index_col=0)
    if df.empty:
        raise ValueError(f"Empty SHAP std CSV: {std_csv}")

    vals = df.iloc[:, 0].copy()
    base = [c.replace("_std", "") for c in vals.index]
    vals.index = base

    ordered = vals.sort_values(ascending=False)

    # Keep top 6 features, group the rest into "other phi"
    keep_top = 5
    top = ordered.iloc[:keep_top]
    rest = ordered.iloc[keep_top:]
    other_sum = rest.sum()
    ordered = pd.concat([top, pd.Series({"other_phi": other_sum})])

    labels = [_phi_label(name) for name in ordered.index]
    x = np.arange(len(ordered))
    y_max = fp.ML_STD_BAR_XMAX

    # Per-bar colors: phi_1 -> #336699, phi_6 -> #cc6666, rest -> default
    COLOR_MAP = {"phi_1": "#336699", "phi_6": "#cc6666"}
    bar_colors = [COLOR_MAP.get(name, color) for name in ordered.index]

    fig, ax = plt.subplots(figsize=(2.205, 1.6))
    ax.bar(
        x,
        ordered.values,
        color=bar_colors,
        edgecolor="black",
        linewidth=fp.AXES_LINEWIDTH,
        width=fp.ML_STD_BAR_HEIGHT,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fp.LABEL_FONTSIZE, rotation=45, ha="right")

    # Color phi_1 and phi_6 tick labels to match their bars
    for tick_label, name in zip(ax.get_xticklabels(), ordered.index):
        if name in COLOR_MAP:
            tick_label.set_color(COLOR_MAP[name])

    ax.tick_params(
        axis="x",
        which="major",
        direction="in",
        length=fp.TICK_LENGTH,
        width=fp.TICK_WIDTH,
        labelsize=fp.LABEL_FONTSIZE,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="in",
        length=fp.TICK_LENGTH,
        width=fp.TICK_WIDTH,
        labelsize=fp.TICK_FONTSIZE,
    )
    for spine in ["left", "bottom", "right", "top"]:
        ax.spines[spine].set_linewidth(fp.AXES_LINEWIDTH)

    yticks = list(fp.ML_STD_BAR_XTICKS)
    ax.set_yticks(yticks)
    ax.set_ylim(0.0, y_max)
    ax.set_ylabel("mean |SHAP|", fontsize=fp.LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=fp.TITLE_FONTSIZE)

    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, dpi=fp.FIG_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Std-only Rg regressor (N=256) + SHAP bar")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--overall-out-dir", type=Path, default=DEFAULT_OVERALL_DIR)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--cluster-size", type=int, default=256)
    parser.add_argument("--bar-only", action="store_true", help="Skip training; plot bar from std CSV")
    parser.add_argument("--std-csv", type=Path, default=None, help="SHAP std CSV for bar-only mode")
    args = parser.parse_args()

    if args.bar_only:
        std_csv = args.std_csv or (args.overall_out_dir / f"{args.prefix}_shap_std_importance.csv")
    else:
        df, std_cols = _load_std_only(args.features, args.cluster_size, n_cpus=args.cores)
        std_df = df[std_cols + ["y"]].copy()
        model_path = args.out_dir / f"{args.prefix}_std_only.joblib"
        model, metrics = _train_regressor(std_df, std_cols, model_path, n_jobs=args.cores)
        metrics_path = args.out_dir / f"{args.prefix}_std_only_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        agg_out = args.out_dir / f"{args.prefix}_aggregated_features.csv"
        df.to_csv(agg_out, index=False)
        std_csv = _compute_shap_std(model, std_df[std_cols], args.out_dir, args.prefix)

    title = ""
    out_svg = args.out_dir / f"{args.prefix}_shap_bar.svg"
    _plot_std_bar(std_csv, out_svg, title=title)

    if args.overall_out_dir:
        overall_svg = args.overall_out_dir / f"{args.prefix}_shap_bar.svg"
        if overall_svg.resolve() != out_svg.resolve():
            _plot_std_bar(std_csv, overall_svg, title=title)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Temperature-dependent joy (ridgeline) plots for phi_1 and phi_6.

Two separate output SVGs, each containing P_monomer | M_monomer side-by-side
panels stacked over 10 temperatures (260-440 K every 20 K).
Semitransparent fills let temperature ridges show through one another.

METHODOLOGY
-----------
Feature extraction from MD trajectories:
  1. Load GROMACS .tpr/.xtc trajectory via MDAnalysis.
  2. Detect rotatable bonds using the SMARTS pattern
       [!#1]~[!$(*#*)&!D1:1]-,=;!@[!$(*#*)&!D1:2]~[!#1]
     applied to the first fragment (molecule) via RDKit.
  3. For each frame (every ``skip``-th), compute all dihedral angles
     (degrees, wrapped to (-180, 180]) and the radius of gyration (Rg)
     for every molecule in the box.
  4. Concatenate results across all temperatures and systems into a
     single features.csv.

The script checks for cached features.csv first and only runs the
extraction when the cache is missing.
"""

# ---------------------------------------------------------------------------
# DATA / SIMULATION PATHS
# ---------------------------------------------------------------------------
from __future__ import annotations

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

# -- figure params ----------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

# -- data source ------------------------------------------------------------
DATA_CSV = Path(DATA_ROOT) / (
    "old_ml_conformational_analysis/ML_analysis/previous_all/"
    "feature_extraction/features.csv"
)

# -- sampling parameters ----------------------------------------------------
TEMPS = list(range(260, 441, 40))   # 260, 300, 340, 380, 420  (5 ridges)
SYSTEMS = ["P_monomer", "M_monomer"]
TORSIONS = [1, 6]

# -- plot geometry ----------------------------------------------------------
# Data remapped to [0, 360] so that +/-180 sits at the centre (x=180).
# Transformation: x = angle % 360  ->  -180->180, -90->270, 0->0/360, 90->90
HIST_BINS = np.arange(0, 361, 10)     # 10 deg bins over [0, 360]
XLIM = (0.0, 360.0)
ANGLE_TICKS      = [0,  90,  180,  270,  360]
ANGLE_TICKLABELS = ["0", "90", "\u00b1180", "\u221290", "0"]

RIDGE_GAP = 1.35        # vertical spacing between ridge baselines
RIDGE_SCALE = 0.9       # peak height / RIDGE_GAP  (< 1 -> no overlap at peak)
FILL_ALPHA = fp.FILL_ALPHA   # 0.25 -- semitransparent

PANEL_W = 1.28          # inches per panel (~20% narrower again)
H_PER_RIDGE = 0.32      # inches per temperature ridge
H_BASE = 1.2            # inches of base margin (x-axis label, etc.)
WSPACE = 0.12           # fraction of panel width between panels

SYS_COLORS = fp.SYSTEM_COLORS
SYS_LABELS = fp.SYSTEM_LABELS

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
# PLOTTING (unchanged from original)
# ---------------------------------------------------------------------------

def _draw_panel(ax: plt.Axes, df_sys: pd.DataFrame, phi_col: str, color: str,
                show_yticks: bool) -> None:
    """Render stacked histogram ridges for one system onto ax."""
    n = len(TEMPS)

    # Pre-compute histograms and find global max for a consistent height scale
    hists = []
    global_max = 0.0
    for T in TEMPS:
        vals = df_sys.loc[df_sys["T"] == T, phi_col].dropna().values % 360
        counts, edges = np.histogram(vals, bins=HIST_BINS, density=True)
        hists.append((counts, edges))
        if counts.max() > global_max:
            global_max = counts.max()

    if global_max <= 0:
        return
    scale = RIDGE_SCALE * RIDGE_GAP / global_max

    for i, (T, (counts, edges)) in enumerate(zip(TEMPS, hists)):
        # i=0 -> coldest (260 K) -> top of plot
        offset = (n - 1 - i) * RIDGE_GAP

        # Build step-function x/y arrays for fill_between
        x_step = np.repeat(edges, 2)[1:-1]
        y_step = np.repeat(counts, 2)

        ax.fill_between(x_step, offset, offset + scale * y_step,
                        color=color, alpha=FILL_ALPHA, linewidth=0)
        ax.plot(x_step, offset + scale * y_step,
                color=color, lw=fp.AXES_LINEWIDTH * 0.55, alpha=0.85)

    # y-axis temperature labels (left panel only)
    if show_yticks:
        ytick_pos = [(n - 1 - i) * RIDGE_GAP for i in range(n)]
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels([str(T) for T in TEMPS], fontsize=fp.TICK_FONTSIZE)
    else:
        ax.set_yticks([])

    ylim_top = (n - 1) * RIDGE_GAP + RIDGE_GAP * 0.9
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.3 * RIDGE_GAP, ylim_top)
    ax.set_xticks(ANGLE_TICKS)
    ax.set_xticklabels(ANGLE_TICKLABELS)

    # Hide top/right spines (they add clutter on joy plots)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_yticks:
        ax.spines["left"].set_visible(False)

    fp.style_axis(ax, show_left=show_yticks, show_bottom=True)
    # Override tick direction for top/right from style_axis
    ax.tick_params(top=False, right=False)


def make_figure(df: pd.DataFrame, phi_num: int) -> None:
    """Build and save the joy-plot figure for one torsion."""
    phi_col = f"phi_{phi_num}"
    phi_label = "Torsion Angle (\u00b0)"
    out_path = SCRIPT_DIR / f"phi_{phi_num}_temp_dist.png"

    fp.use_mpl_defaults()

    n = len(TEMPS)
    fig_h = H_PER_RIDGE * n + H_BASE
    fig_w = PANEL_W * 2 + 0.7   # 0.7 extra for y-axis labels on left

    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw=dict(wspace=WSPACE),
    )

    for panel_idx, (ax, system) in enumerate(zip(axes, SYSTEMS)):
        df_sys = df[df["system"] == system]
        show_yticks = (panel_idx == 0)

        _draw_panel(ax, df_sys, phi_col, SYS_COLORS[system], show_yticks)

        ax.set_xlabel(phi_label, fontsize=fp.LABEL_FONTSIZE)
        ax.set_title(SYS_LABELS[system], fontsize=fp.TITLE_FONTSIZE, pad=4)

    # Shared y-axis label on the left panel
    axes[0].set_ylabel("Temperature (K)", fontsize=fp.LABEL_FONTSIZE)

    fig.savefig(out_path, dpi=fp.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    print("Loading data...")
    df = load_or_compute_features(DATA_CSV)
    use_cols = [c for c in ["phi_1", "phi_6", "T", "system"] if c in df.columns]
    df = df[use_cols]
    print(f"  {len(df):,} rows loaded")

    for phi_num in TORSIONS:
        print(f"Plotting phi_{phi_num}...")
        make_figure(df, phi_num)

    print("Done.")


if __name__ == "__main__":
    main()

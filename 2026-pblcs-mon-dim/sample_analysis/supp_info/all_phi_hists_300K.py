#!/usr/bin/env python3
"""
Single-page grid of dihedral-angle histograms at 300 K for phi_0 and phi_2-phi_16
(all torsions except phi_1 and phi_6, which have their own temperature-resolved figures).

Both P_monomer and M_monomer are overlaid with semitransparent fills.
x-axis is remapped so +/-180 sits in the centre (0->360 range, angle % 360).
Layout: 5 rows x 3 cols to fit a portrait SI page.

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
from matplotlib.lines import Line2D

# -- figure params ----------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

# -- data -------------------------------------------------------------------
DATA_CSV = Path(DATA_ROOT) / (
    "old_ml_conformational_analysis/ML_analysis/previous_all/"
    "feature_extraction/features.csv"
)
TEMP = 300

# phi_1 and phi_6 have their own dedicated figures
TORSIONS = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# -- histogram / style ------------------------------------------------------
HIST_BINS      = np.arange(0, 361, 10)     # 10 deg bins, [0, 360]
XLIM           = (0.0, 360.0)
ANGLE_TICKS    = [0,   90,   180,   270,   360]
ANGLE_LABELS   = ["0", "90", "\u00b1180", "\u221290", "0"]
FILL_ALPHA     = fp.FILL_ALPHA             # 0.25

SYS_COLORS = fp.SYSTEM_COLORS             # P_monomer: teal, M_monomer: pink
SYS_LABELS = fp.SYSTEM_LABELS

# -- layout -----------------------------------------------------------------
N_COLS   = 3
N_ROWS   = 5                              # 3x5 = 15 panels
FIG_W    = 6.5                            # inches -- fills a letter/A4 page width
FIG_H    = 8.5                            # inches -- fits portrait page
HSPACE   = 0.55                           # vertical space between rows (fraction of panel h)
WSPACE   = 0.35                           # horizontal space between cols

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
# HISTOGRAM HELPERS (unchanged from original)
# ---------------------------------------------------------------------------

def _hist(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, edges) for the remapped [0, 360] histogram."""
    v = vals[np.isfinite(vals)] % 360
    counts, edges = np.histogram(v, bins=HIST_BINS, density=True)
    return counts, edges


def _step_xy(counts: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert histogram arrays into step-function x/y for fill_between / plot."""
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(counts, 2)
    return x, y


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    fp.use_mpl_defaults()

    print("Loading data...")
    df = load_or_compute_features(DATA_CSV)
    phi_cols = [f"phi_{k}" for k in TORSIONS]
    use_cols = [c for c in phi_cols + ["T", "system"] if c in df.columns]
    df = df[use_cols]
    df300 = df[df["T"] == TEMP]
    print(f"  {len(df300):,} rows at {TEMP} K")

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(FIG_W, FIG_H),
        gridspec_kw=dict(hspace=HSPACE, wspace=WSPACE),
    )
    ax_flat = axes.ravel()

    for panel_idx, phi_num in enumerate(TORSIONS):
        ax = ax_flat[panel_idx]
        phi_col = f"phi_{phi_num}"

        global_max = 0.0
        curves = {}
        for system in ("P_monomer", "M_monomer"):
            vals = df300.loc[df300["system"] == system, phi_col].dropna().values
            counts, edges = _hist(vals)
            curves[system] = (counts, edges)
            if counts.max() > global_max:
                global_max = counts.max()

        for system, (counts, edges) in curves.items():
            x, y = _step_xy(counts, edges)
            color = SYS_COLORS[system]
            ax.fill_between(x, 0, y, color=color, alpha=FILL_ALPHA, linewidth=0)
            ax.plot(x, y, color=color, lw=fp.AXES_LINEWIDTH * 0.55, alpha=0.85)

        ax.set_xlim(*XLIM)
        ax.set_ylim(0, global_max * 1.12)
        ax.set_xticks(ANGLE_TICKS)
        ax.set_xticklabels(ANGLE_LABELS, fontsize=fp.TICK_FONTSIZE - 1)
        ax.set_yticks([])
        ax.set_title(rf"$\phi_{{{phi_num}}}$", fontsize=fp.TITLE_FONTSIZE, pad=3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        fp.style_axis(ax, show_left=False, show_bottom=True)
        ax.tick_params(top=False, right=False, left=False)

    # Hide the unused 16th panel slot (3x5=15 panels, 15 torsions, nothing spare)
    for ax in ax_flat[len(TORSIONS):]:
        ax.set_visible(False)

    # Shared x-axis label centred on the bottom row
    fig.text(0.5, 0.01, "Torsion Angle (\u00b0)", ha="center", va="bottom",
             fontsize=fp.LABEL_FONTSIZE)

    # Legend
    handles = [
        Line2D([], [], color=SYS_COLORS[s], lw=2.0, alpha=0.85, label=SYS_LABELS[s])
        for s in ("P_monomer", "M_monomer")
    ]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.01),
               frameon=False, fontsize=fp.LEGEND_FONTSIZE, ncol=2)

    out_path = SCRIPT_DIR / f"all_phi_hists_{TEMP}K.png"
    fig.savefig(out_path, dpi=fp.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

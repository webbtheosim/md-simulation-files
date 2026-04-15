#!/usr/bin/env python
"""
Stacked S(T) and density(T) panel for dimer systems.

Top: nematic order parameter vs temperature.
Bottom: density vs temperature.

This script computes nematic order parameters from raw GROMACS .tpr/.xtc
trajectories and densities from .edr energy files, OR loads cached .npy
results if they already exist.  The computation pipeline uses SMARTS-based
phenyl-O-phenyl detection (via RDKit) on MDAnalysis universes to build
molecular directors, then diagonalises the Q-tensor to obtain S.  Density
is read directly from GROMACS .edr files via MDAnalysis EDRReader.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Trajectory computation imports (used only when cached data is missing)
# ---------------------------------------------------------------------------
try:
    import MDAnalysis as mda
    from MDAnalysis.auxiliary.EDR import EDRReader
    _HAS_MDA = True
except ImportError:
    _HAS_MDA = False

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import GetPeriodicTable  # noqa: F401
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis.auxiliary.EDR")

# ---------------------------------------------------------------------------
# DATA PATHS
# SIMULATION_ROOT : top-level directory holding raw GROMACS trajectories
# DATA_ROOT       : root of the paper_1_figures analysis tree (cached .npy)
# ---------------------------------------------------------------------------
SIMULATION_ROOT = os.environ.get("SIMULATION_ROOT", "/scratch/gpfs/WEBB/jv6139")
DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/paper_1_figures")

# ---------------------------------------------------------------------------
# shared styling
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure_params"))
if FIG_PARAM_DIR not in sys.path:
    sys.path.insert(0, FIG_PARAM_DIR)
import figure_params as fp  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# paths and constants
# ---------------------------------------------------------------------------
_DATA_SUBDIR = os.path.join(DATA_ROOT, "dimer_phase_density")
PHASE_DIR = os.path.join(_DATA_SUBDIR, "phase_diagram")
DENSITY_DIR = os.path.join(_DATA_SUBDIR, "density")
OUTPUT_PATH = os.path.join(BASE_DIR, "polished_dimer_phase_diagram.svg")

SYSTEMS = ("PM", "MP", "P2", "M2")
SYSTEM_COLORS: Dict[str, str] = fp.DIMER_COLORS
SYSTEM_LABELS: Dict[str, str] = fp.DIMER_LABELS
SYSTEM_MARKERS: Dict[str, str] = {
    "P2": "o",   # circle
    "MP": ">",   # right-facing triangle
    "PM": "<",   # left-facing triangle
    "M2": "D",   # diamond
}

FIG_SIZE = (3.15, 3.0)  # match monomer stacked layout
TEMPERATURES = np.round(np.arange(375, 475, 2.5), 1)
XLIM = (370, 475)
XTICKS = np.arange(XLIM[0], XLIM[1] + 1, 15)
S_YLIM = (0.0, 0.8)
DENSITY_YLIM = (1000, 1150)

# ---------------------------------------------------------------------------
# SMARTS pattern for the phenyl-O-phenyl mesogenic core
# ---------------------------------------------------------------------------
_SMARTS_STR = "c1ccccc1COc2ccccc2"
_SMARTS = Chem.MolFromSmarts(_SMARTS_STR) if _HAS_RDKIT else None

# ---------------------------------------------------------------------------
# Raw trajectory path templates (used when computing from scratch)
# ---------------------------------------------------------------------------
_TPR_TEMPLATE = os.path.join(
    SIMULATION_ROOT, "dimers", "{system}", "simulations", "tREM", "{T}K", "trem_gpu.tpr"
)
_XTC_TEMPLATE = os.path.join(
    SIMULATION_ROOT, "dimers", "{system}", "simulations", "tREM", "{T}K", "trem_gpu.xtc"
)
_EDR_TEMPLATE = os.path.join(
    SIMULATION_ROOT, "dimers", "{system}", "simulations", "tREM", "{T}K", "trem_gpu.edr"
)

# Frame-skip used when computing S from trajectories (every 10th frame)
_TRAJ_SKIP = 10


# ═══════════════════════════════════════════════════════════════════════════
#  ORDER-PARAMETER COMPUTATION (from raw GROMACS trajectories)
# ═══════════════════════════════════════════════════════════════════════════

def get_directors_smarts(
    top: str,
    traj: str,
    start: int = 0,
    stop: int | None = None,
    skip: int = 1,
) -> np.ndarray:
    """
    Build directors[time, fragment, core, xyz] for every phenyl-O-phenyl
    mesogenic core in each molecular fragment.

    For each fragment the SMARTS pattern ``c1ccccc1COc2ccccc2`` is matched
    against the RDKit molecule built from the MDAnalysis fragment.  The two
    six-membered phenyl rings inside each match provide ring-centroid pairs
    whose normalised difference vector is the local director.

    Parameters
    ----------
    top : str
        Path to a GROMACS .tpr (or .gro) topology file.
    traj : str
        Path to a GROMACS .xtc trajectory file.
    start, stop, skip : int
        Slice parameters for the trajectory.

    Returns
    -------
    directors : np.ndarray, shape (n_frames, n_fragments, max_cores, 3)
        Unit director vectors; entries are NaN where no core exists.
    """
    if not _HAS_MDA or not _HAS_RDKIT:
        raise RuntimeError("MDAnalysis and RDKit are required for trajectory computation")

    u = mda.Universe(top, traj)

    ring_query = Chem.MolFromSmarts("c1ccccc1")  # six-member aromatic ring

    # -- pre-compute per-fragment core ring pairs ----------------------------
    frag_core_pairs = []
    for frag in u.atoms.fragments:
        rdmol = frag.convert_to("RDKIT")
        ring_info = rdmol.GetRingInfo().AtomRings()
        matches = rdmol.GetSubstructMatches(_SMARTS, uniquify=True)

        core_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for m in matches:
            # find every 6-ring completely inside this match
            mring = [r for r in ring_info if len(r) == 6 and set(r).issubset(m)]
            if len(mring) != 2:
                # fallback: match phenyl SMARTS inside the same atom set
                phenyls = [
                    rp
                    for rp in rdmol.GetSubstructMatches(ring_query, uniquify=False)
                    if set(rp).issubset(m)
                ]
                mring = phenyls
            if len(mring) == 2:
                core_pairs.append((tuple(mring[0]), tuple(mring[1])))

        frag_core_pairs.append((frag, core_pairs))

    n_frag = len(frag_core_pairs)
    max_core = max((len(cp[1]) for cp in frag_core_pairs), default=1) or 1

    # -- iterate frames ------------------------------------------------------
    traj_slice = u.trajectory[start:stop:skip]
    directors = np.full((len(traj_slice), n_frag, max_core, 3), np.nan)

    for ti, ts in enumerate(traj_slice):
        for fi, (frag, core_pairs) in enumerate(frag_core_pairs):
            for ci, (r1, r2) in enumerate(core_pairs):
                c1 = frag.atoms[list(r1)].positions.mean(axis=0)
                c2 = frag.atoms[list(r2)].positions.mean(axis=0)
                v = c2 - c1

                box = ts.dimensions[:3]
                if np.all(box > 0):
                    v -= box * np.round(v / box)  # minimum-image convention

                norm = np.linalg.norm(v)
                if norm > 1e-8:
                    directors[ti, fi, ci] = v / norm

    return directors


def calc_S(directors: np.ndarray) -> np.ndarray:
    """
    Compute the nematic order parameter S for each frame.

    For every frame the traceless Q-tensor is built from all valid director
    vectors and its largest eigenvalue is returned as S.

    Parameters
    ----------
    directors : np.ndarray, shape (n_frames, n_fragments, n_cores, 3)

    Returns
    -------
    S : np.ndarray, shape (n_frames,)
    """
    n_t = directors.shape[0]
    dirs = directors.reshape(n_t, -1, 3)
    valid = ~np.isnan(dirs[..., 0])
    S = np.zeros(n_t)
    eye = np.eye(3)
    for t in range(n_t):
        vecs = dirs[t][valid[t]]
        if len(vecs) == 0:
            S[t] = np.nan
            continue
        Q = sum(1.5 * (np.outer(d, d) - eye / 3.0) for d in vecs) / len(vecs)
        S[t] = np.max(np.linalg.eigvalsh(Q))
    return S


def compute_nematic_order(
    system: str,
    temp: float,
) -> np.ndarray | None:
    """
    Compute S(t) for one (system, temperature) from raw trajectories.

    Returns the per-frame S array, or None on failure.
    """
    T_tag = f"{int(temp)}" if float(temp).is_integer() else f"{temp}"
    tpr = _TPR_TEMPLATE.format(system=system, T=T_tag)
    xtc = _XTC_TEMPLATE.format(system=system, T=T_tag)

    if not os.path.isfile(tpr) or not os.path.isfile(xtc):
        print(f"[S compute] {system} {T_tag}K: trajectory files not found, skipping")
        return None

    try:
        directors = get_directors_smarts(tpr, xtc, start=0, stop=None, skip=_TRAJ_SKIP)
        return calc_S(directors)
    except Exception as e:
        print(f"[S compute] {system} {T_tag}K failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  DENSITY COMPUTATION (from GROMACS .edr energy files)
# ═══════════════════════════════════════════════════════════════════════════

def read_density_series(edr_path: str) -> tuple[float, float] | None:
    """
    Read density (kg/m^3) from a GROMACS .edr energy file.

    Returns (mean_density, std_density), or None if the standard deviation
    exceeds 10 kg/m^3 (indicating an unconverged or problematic run).
    """
    if not _HAS_MDA:
        raise RuntimeError("MDAnalysis is required for .edr reading")

    with EDRReader(edr_path) as edr:
        data = edr.get_data(["Density"])
        density = np.array(data["Density"])
        mean_density = np.mean(density)
        stds = np.std(density)
        if stds > 10:
            print(f"Warning: {edr_path} has high std in density: {stds:.2f} kg/m^3")
            return None

    return float(mean_density), float(stds)


def compute_density_for_temp(
    system: str,
    temp: float,
) -> tuple[float, float] | None:
    """
    Compute mean density and std for one (system, temperature) from raw .edr.

    Returns (mean_rho, std_rho) or None on failure.
    """
    T_tag = f"{int(temp)}" if float(temp).is_integer() else f"{temp}"
    edr_path = _EDR_TEMPLATE.format(system=system, T=T_tag)

    if not os.path.isfile(edr_path):
        print(f"[density compute] {system} {T_tag}K: .edr not found, skipping")
        return None

    try:
        return read_density_series(edr_path)
    except Exception as e:
        print(f"[density compute] {system} {T_tag}K failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS: caching, loading, block averaging, TNI fitting
# ═══════════════════════════════════════════════════════════════════════════

def _temp_tag(temp: float) -> str:
    """Format temperature for filenames (strip trailing .0)."""
    return f"{temp:.1f}".rstrip("0").rstrip(".")


def _load_series(base_dir: str, system: str, temp: float, suffix: str = "") -> np.ndarray | None:
    """Load an npy series for a specific temperature; checks with and without K."""
    tag = _temp_tag(temp)
    candidates = [
        os.path.join(base_dir, f"{system}_{tag}K{suffix}.npy"),
        os.path.join(base_dir, f"{system}_{tag}{suffix}.npy"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return np.load(path)
    return None


def block_avg_and_error(x: np.ndarray) -> Tuple[float, float, float, float]:
    """Mean and stderr from 10 contiguous block means; reports g and Neff."""
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(vals.mean())

    n_blocks_target = 10
    n_blocks = min(n_blocks_target, vals.size)
    blocks = [b for b in np.array_split(vals, n_blocks) if b.size > 0]
    block_means = np.array([float(np.mean(b)) for b in blocks], dtype=float)

    neff = float(block_means.size)
    if block_means.size > 1:
        stderr = float(np.std(block_means, ddof=1) / np.sqrt(block_means.size))
    else:
        stderr = 0.0
    g = float(len(vals) / neff) if neff > 0 else np.nan
    return mean, stderr, g, neff


def _estimate_tni(temps: np.ndarray, means: np.ndarray) -> tuple[float, float] | None:
    """Fit Maier-Saupe style model to estimate TNI (matches source script approach)."""
    if temps.size < 6:
        return None
    p2_iso_guess = float(np.nanmean(means[-5:]))

    def r_squared(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_resid = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_resid / ss_total)

    best_r2 = -np.inf
    best_fit: tuple[float, float] | None = None
    p2_grid = np.linspace(max(0.01, p2_iso_guess * 0.5), min(0.5, p2_iso_guess * 1.5), 15)

    for p2_iso in p2_grid:
        def ms_model(T, beta, TNI):
            fac = np.clip(1 - T / TNI, 0.0, None)
            return (1 - p2_iso) * fac**beta + p2_iso

        try:
            popt, _ = curve_fit(
                ms_model,
                temps,
                means,
                p0=(0.25, temps[-1] * 1.05),
                bounds=([0.0, 0.0], [1.0, 700.0]),
                maxfev=5000,
            )
            predicted = ms_model(temps, *popt)
            r2 = r_squared(means, predicted)
            if r2 > best_r2:
                best_r2 = r2
                best_fit = (float(popt[0]), float(popt[1]))
        except Exception:
            continue

    if best_fit is None:
        return None
    beta_fit, tni_fit = best_fit
    if np.isfinite(tni_fit) and temps.min() <= tni_fit <= temps.max():
        return float(tni_fit), float(beta_fit)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADERS (cache-first, compute-on-miss)
# ═══════════════════════════════════════════════════════════════════════════

def load_order_parameter(system: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load or compute S time series and return mean/SEM via block averaging.

    First checks for cached .npy files under PHASE_DIR.  If not found,
    computes S from raw GROMACS .tpr/.xtc trajectories.
    """
    temps: list[float] = []
    means: list[float] = []
    errs: list[float] = []
    for temp in TEMPERATURES:
        arr = _load_series(PHASE_DIR, system, temp)

        # If cached file not found, compute from raw trajectories
        if arr is None:
            arr = compute_nematic_order(system, temp)
            if arr is not None:
                # Cache the result for future runs
                tag = _temp_tag(temp)
                os.makedirs(PHASE_DIR, exist_ok=True)
                np.save(os.path.join(PHASE_DIR, f"{system}_{tag}K.npy"), arr)

        if arr is None or arr.size == 0:
            continue
        mean, sem, g, neff = block_avg_and_error(arr)
        print(f"[S] {system} {temp}K: mean={mean:.4f}, stderr={sem:.4f}")
        temps.append(temp)
        means.append(mean)
        errs.append(sem)
    if not temps:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    order = np.argsort(temps)
    temps_arr = np.asarray(temps)[order]
    means_arr = np.asarray(means)[order]
    errs_arr = np.asarray(errs)[order]
    return temps_arr, means_arr, errs_arr, temps_arr, means_arr


def load_density(system: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load or compute density mean/std for each temperature.

    First checks for cached .npy files under DENSITY_DIR.  If not found,
    reads density from raw GROMACS .edr energy files.
    """
    temps: list[float] = []
    means: list[float] = []
    errs: list[float] = []
    for temp in TEMPERATURES:
        arr = _load_series(DENSITY_DIR, system, temp, suffix="_rho")

        # If cached file not found, compute from raw .edr
        if arr is None:
            result = compute_density_for_temp(system, temp)
            if result is not None:
                mean_rho, std_rho = result
                arr = np.array([mean_rho, std_rho])
                # Cache the result for future runs
                tag = _temp_tag(temp)
                os.makedirs(DENSITY_DIR, exist_ok=True)
                np.save(os.path.join(DENSITY_DIR, f"{system}_{tag}K_rho.npy"), arr)

        if arr is None or arr.size < 2:
            continue
        temps.append(temp)
        means.append(float(arr[0]))
        errs.append(float(arr[1]))
    if not temps:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(temps)
    temps_arr = np.asarray(temps)[order]
    means_arr = np.asarray(means)[order]
    errs_arr = np.asarray(errs)[order]
    return temps_arr, means_arr, errs_arr


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_panel() -> None:
    """Build stacked S(T) and rho(T) axes sharing the temperature scale."""
    fp.use_mpl_defaults()
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.05, 1.0],
        left=0.24,
        right=0.96,
        top=0.98,
        bottom=0.12,
        hspace=0.08,
    )

    ax_s = fig.add_subplot(gs[0])
    ax_rho = fig.add_subplot(gs[1], sharex=ax_s)

    density_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for system in SYSTEMS:
        density_cache[system] = load_density(system)

    # --- S vs T ---
    for system in SYSTEMS:
        temps_plot, means_plot, errs_plot, temps_full, means_full = load_order_parameter(system)
        if temps_plot.size == 0:
            continue
        ax_s.errorbar(
            temps_plot,
            means_plot,
            yerr=errs_plot,
            fmt=SYSTEM_MARKERS.get(system, "o"),
            markersize=fp.MARKER_SIZE,
            mfc=SYSTEM_COLORS.get(system, "gray"),
            mec="black",
            mew=fp.MARKER_EDGEWIDTH,
            ecolor=SYSTEM_COLORS.get(system, "gray"),
            elinewidth=fp.MARKER_EDGEWIDTH,
            capsize=0,
            alpha=0.9,
            label=SYSTEM_LABELS.get(system, system),
            zorder=3,
        )
        tni_est = _estimate_tni(temps_full, means_full)
        if tni_est is not None:
            tni_val, beta_val = tni_est
            print(f"[TNI] {system}: beta={beta_val:.3f}, TNI={tni_val:.2f} K")
            t_margin = 0.5
            if (temps_full.min() + t_margin) <= tni_val <= (temps_full.max() - t_margin) and XLIM[0] <= tni_val <= XLIM[1]:
                ax_s.axvline(
                    tni_val,
                    color=SYSTEM_COLORS.get(system, "gray"),
                    linestyle="--",
                    linewidth=fp.MARKER_EDGEWIDTH,
                    alpha=0.7,
                    zorder=2.0,
                )
    ax_s.set_ylabel("Order Param.\n$S$")
    ax_s.set_ylim(*S_YLIM)
    ax_s.set_yticks(np.linspace(S_YLIM[0], S_YLIM[1], 5))
    ax_s.set_xlim(*XLIM)
    ax_s.set_xticks(XTICKS)
    ax_s.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    fp.style_axis(ax_s, show_bottom=False)
    ax_s.tick_params(labelbottom=False, labelsize=fp.TICK_FONTSIZE)
    ax_s.legend(frameon=False, fontsize=fp.LEGEND_FONTSIZE, loc="upper right")

    # --- density vs T ---
    for system in SYSTEMS:
        temps, means, errs = density_cache.get(system, (np.array([]), np.array([]), np.array([])))
        if temps.size == 0:
            continue
        ax_rho.errorbar(
            temps,
            means,
            yerr=errs,
            fmt=SYSTEM_MARKERS.get(system, "o"),
            markersize=fp.MARKER_SIZE,
            mfc=SYSTEM_COLORS.get(system, "gray"),
            mec="black",
            mew=fp.MARKER_EDGEWIDTH,
            ecolor=SYSTEM_COLORS.get(system, "gray"),
            elinewidth=fp.MARKER_EDGEWIDTH,
            capsize=0,
            alpha=0.9,
            zorder=3,
        )
    ax_rho.set_xlabel(r"Temperature, $T$ (K)")
    ax_rho.set_ylabel("Density,\n" + r"$\rho$ (kg m$^{-3}$)")
    ax_rho.set_ylim(*DENSITY_YLIM)
    ax_rho.set_yticks(np.arange(DENSITY_YLIM[0], DENSITY_YLIM[1] + 1, 50))
    ax_rho.set_xlim(*XLIM)
    ax_rho.set_xticks(XTICKS)
    ax_rho.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    fp.style_axis(ax_rho, show_bottom=True)
    ax_rho.tick_params(labelsize=fp.TICK_FONTSIZE)

    fig.align_ylabels([ax_s, ax_rho])
    fig.savefig(OUTPUT_PATH, dpi=fp.FIG_DPI)
    plt.close(fig)
    print(f"[done] wrote figure to {OUTPUT_PATH}")


def main() -> None:
    plot_panel()


if __name__ == "__main__":
    main()

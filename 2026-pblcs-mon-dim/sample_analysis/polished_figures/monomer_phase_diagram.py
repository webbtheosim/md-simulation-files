#!/usr/bin/env python
"""
Stacked S(T) and density(T) panel for monomer systems.

Top: nematic order parameter vs temperature.
Bottom: density vs temperature.

This script contains the full computation pipeline:
  - Order parameter S(T): extracts mesogenic-core directors from MD
    trajectories via SMARTS matching, builds the Q-tensor, and takes the
    largest eigenvalue as the nematic order parameter per frame.
  - Density rho(T): reads the Density time series from GROMACS .edr files.

Cached .npy files are used when available; if absent the quantities are
recomputed from the raw simulation trajectories / energy files.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DATA PATHS -- adjust these to point to your local data locations
# DATA_ROOT: root of the paper_1_figures analysis tree (cached .npy files)
# SIMULATION_ROOT: root of the raw GROMACS simulation directories
# ---------------------------------------------------------------------------
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/paper_1_figures")
SIMULATION_ROOT = os.environ.get("SIMULATION_ROOT", "/scratch/gpfs/WEBB/jv6139")


import sys
import warnings
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit

# Heavy imports guarded so the module can still be imported for inspection
# even on machines that lack MDAnalysis / RDKit (plotting from cache only).
try:
    import MDAnalysis as mda
    from MDAnalysis.auxiliary.EDR import EDRReader

    HAS_MDA = True
except ImportError:
    HAS_MDA = False

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import GetPeriodicTable

    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis.auxiliary.EDR")

# ---------------------------------------------------------------------------
# shared styling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure_params"))
if FIG_PARAM_DIR not in sys.path:
    sys.path.insert(0, FIG_PARAM_DIR)
import figure_params as fp  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# paths and constants
_DATA_SUBDIR = os.path.join(DATA_ROOT, "monomer_phase_density")
PHASE_DIR = os.path.join(_DATA_SUBDIR, "phase_diagram")
DENSITY_DIR = os.path.join(_DATA_SUBDIR, "density")
OUTPUT_PATH = os.path.join(BASE_DIR, "polished_monomer_phase_diagram.svg")

# Templates for raw simulation files ----------------------------------------
# Order parameter: need .tpr (topology) and .xtc (trajectory)
ORDER_PARAM_TPR = {
    "P_monomer": os.path.join(SIMULATION_ROOT, "N_monomer", "simulations", "tREM", "{T}K", "trem_gpu.tpr"),
    "M_monomer": os.path.join(SIMULATION_ROOT, "M_monomer", "simulations", "tREM", "{T}K", "trem_gpu.tpr"),
}
ORDER_PARAM_XTC = {
    "P_monomer": os.path.join(SIMULATION_ROOT, "N_monomer", "simulations", "tREM", "{T}K", "trem_gpu.xtc"),
    "M_monomer": os.path.join(SIMULATION_ROOT, "M_monomer", "simulations", "tREM", "{T}K", "trem_gpu.xtc"),
}
# Density: need .edr (energy file)
DENSITY_EDR = {
    "P_monomer": os.path.join(SIMULATION_ROOT, "N_monomer", "simulations", "tREM", "{T}K", "trem_gpu.edr"),
    "M_monomer": os.path.join(SIMULATION_ROOT, "M_monomer", "simulations", "tREM", "{T}K", "trem_gpu.edr"),
}

SYSTEMS = ("P_monomer", "M_monomer")
TEMPERATURES = np.round(np.arange(260, 440, 2.5), 1)  # matches source scripts
FIG_SIZE = (3.15, 3.0)  # (width, height) in inches
XLIM = (250, 450)
XTICKS = np.arange(250, 451, 50)
S_YLIM = (0.0, 0.8)
DENSITY_YLIM = (1000, 1200)
S_START_FRACTION = 0.0  # discard this fraction, keep later part of S time series
SYSTEM_MARKERS = {"P_monomer": "o", "M_monomer": "^"}
SYSTEM_LEGEND_LABELS = {
    "P_monomer": r"$\mathrm{P}_1$",
    "M_monomer": r"$\mathrm{M}_1$",
}

# SMARTS pattern for the phenyl-O-phenyl mesogenic core
_SMARTS_STR = "c1ccccc1COc2ccccc2"


# ===========================================================================
#  ORDER PARAMETER COMPUTATION
# ===========================================================================

def get_directors_smarts(top: str, traj: str,
                         start: int = 0, stop: int | None = None,
                         skip: int = 1) -> np.ndarray:
    """
    Build directors[time, fragment, core, xyz] for every phenyl-O-phenyl
    mesogenic core in each molecular fragment.

    For each trajectory frame the two phenyl rings of every matched core are
    identified via SMARTS, their centroids computed, and the unit vector
    connecting them stored as the director.

    Parameters
    ----------
    top : str
        Path to a GROMACS .tpr topology file.
    traj : str
        Path to a GROMACS .xtc trajectory file.
    start, stop, skip : int
        Trajectory slicing parameters (passed to ``universe.trajectory[start:stop:skip]``).

    Returns
    -------
    directors : np.ndarray, shape (n_frames, n_fragments, max_cores, 3)
        Unit director vectors; entries are NaN where no core exists.
    """
    if not HAS_MDA or not HAS_RDKIT:
        raise RuntimeError(
            "MDAnalysis and RDKit are required to compute directors from trajectories."
        )

    smarts_query = Chem.MolFromSmarts(_SMARTS_STR)
    ring_query = Chem.MolFromSmarts("c1ccccc1")

    u = mda.Universe(top, traj)

    # -- pre-compute per-fragment core ring pairs ---------------------------
    frag_core_pairs = []
    for frag in u.atoms.fragments:
        rdmol = frag.convert_to("RDKIT")
        ring_info = rdmol.GetRingInfo().AtomRings()
        matches = rdmol.GetSubstructMatches(smarts_query, uniquify=True)

        core_pairs = []
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

    # -- iterate frames -----------------------------------------------------
    traj_slice = u.trajectory[start:stop:skip]
    directors = np.full((len(traj_slice), n_frag, max_core, 3), np.nan)

    for ti, ts in enumerate(traj_slice):
        for fi, (frag, core_pairs) in enumerate(frag_core_pairs):
            for ci, (r1, r2) in enumerate(core_pairs):
                c1 = frag.atoms[list(r1)].positions.mean(axis=0)
                c2 = frag.atoms[list(r2)].positions.mean(axis=0)
                v = c2 - c1
                # minimum-image correction
                box = ts.dimensions[:3]
                if np.all(box > 0):
                    v -= box * np.round(v / box)
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


def nematic_order(top: str, traj: str,
                  start: int = 0, stop: int | None = None,
                  skip: int = 1) -> np.ndarray:
    """
    End-to-end computation of S(frame) from a topology + trajectory pair.

    Combines ``get_directors_smarts`` and ``calc_S``.
    """
    directors = get_directors_smarts(top, traj, start=start, stop=stop, skip=skip)
    return calc_S(directors)


# ===========================================================================
#  DENSITY COMPUTATION
# ===========================================================================

def read_density_series(edr_path: str) -> np.ndarray:
    """
    Return the full density time series (kg/m^3) from a GROMACS .edr file.

    Parameters
    ----------
    edr_path : str
        Path to a .edr energy file.

    Returns
    -------
    density : np.ndarray, shape (n_frames,)
    """
    if not HAS_MDA:
        raise RuntimeError("MDAnalysis is required to read .edr files.")
    with EDRReader(edr_path) as edr:
        data = edr.get_data(["Density"])
        density = np.asarray(data["Density"], dtype=float)
    return density


# ===========================================================================
#  HELPER UTILITIES
# ===========================================================================

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


def _save_series(base_dir: str, system: str, temp: float, data: np.ndarray, suffix: str = "") -> None:
    """Save an npy series so subsequent runs can use the cache."""
    os.makedirs(base_dir, exist_ok=True)
    tag = _temp_tag(temp)
    out_path = os.path.join(base_dir, f"{system}_{tag}K{suffix}.npy")
    np.save(out_path, data)


def block_avg_and_error(x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute mean and standard error from 10 contiguous block means.
    Returns mean, stderr, g, Neff where g/Neff are reported for continuity.
    """
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


# ===========================================================================
#  DATA LOADERS  (cache-first, compute-on-miss)
# ===========================================================================

def load_order_parameter(system: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load (or compute) S time series per temperature and return summary stats.

    Strategy:
      1. Try loading a cached .npy from PHASE_DIR.
      2. If missing, compute S from the raw .tpr/.xtc trajectory, cache the
         result, then proceed.

    Returns
    -------
    temps_plot, means_plot, errs_plot : arrays for plotting (every-other-point)
    temps_full, means_full            : full temperature / mean arrays (for TNI fit)
    """
    temps: list[float] = []
    means: list[float] = []
    errs: list[float] = []

    tpr_template = ORDER_PARAM_TPR.get(system)
    xtc_template = ORDER_PARAM_XTC.get(system)

    for temp in TEMPERATURES:
        # --- try cache first ------------------------------------------------
        arr = _load_series(PHASE_DIR, system, temp)

        # --- compute from trajectory if cache miss --------------------------
        if arr is None and tpr_template is not None and xtc_template is not None:
            tag = _temp_tag(temp)
            tpr_path = tpr_template.format(T=tag)
            xtc_path = xtc_template.format(T=tag)
            if os.path.isfile(tpr_path) and os.path.isfile(xtc_path):
                print(f"[S] computing {system} {temp}K from trajectory ...")
                try:
                    arr = nematic_order(tpr_path, xtc_path, start=0, stop=None, skip=1)
                    _save_series(PHASE_DIR, system, temp, arr)
                except Exception as exc:
                    print(f"[S] {system} {temp}K failed: {exc}")
                    arr = None

        if arr is None or arr.size == 0:
            continue

        start_idx = max(0, int(S_START_FRACTION * arr.size))
        vals = arr[start_idx:]
        try:
            mean, sem, g, neff = block_avg_and_error(vals)
        except ValueError:
            continue
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
    # plot every other point to match original script
    plot_mask = np.arange(temps_arr.size) % 2 == 0
    return temps_arr[plot_mask], means_arr[plot_mask], errs_arr[plot_mask], temps_arr, means_arr


def _estimate_tni(temps: np.ndarray, means: np.ndarray, system: str) -> tuple[float, float] | None:
    """Fit Maier-Saupe style model to estimate TNI (matches source script grid search).
        Range of values around the transition, then it fits.
    
    """
    if temps.size < 6:
        return None
    if system == "P_monomer":
        fit_lo, fit_hi = 340, 415
    elif system == "M_monomer":
        fit_lo, fit_hi = 300, 340
    else:
        fit_lo, fit_hi = temps.min(), temps.max()
    fit_mask = (temps >= fit_lo) & (temps <= fit_hi)
    temps_subset = temps[fit_mask]
    means_subset = means[fit_mask]
    if temps_subset.size < 6:
        return None

    def r_squared(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_resid = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_resid / ss_total)

    best_r2 = -np.inf
    best_fit: tuple[float, float] | None = None
    p2_iso_guess = np.mean(means_subset[-5:])
    p2_grid = np.linspace(max(0.01, p2_iso_guess * 0.5), min(0.5, p2_iso_guess * 1.5), 15)

    for p2_iso in p2_grid:
        def ms_model(T, beta, TNI):
            fac = np.clip(1 - T / TNI, 0.0, None)
            return (1 - p2_iso) * fac**beta + p2_iso

        try:
            popt, _ = curve_fit(
                ms_model,
                temps_subset,
                means_subset,
                p0=(0.25, temps_subset[-1] * 1.05),
                bounds=([0.0, 0.0], [1.0, 600.0]),
                maxfev=5000,
            )
            predicted = ms_model(temps_subset, *popt)
            r2 = r_squared(means_subset, predicted)
            if r2 > best_r2:
                best_r2 = r2
                best_fit = (float(popt[0]), float(popt[1]))
        except Exception:
            continue

    if best_fit is None:
        return None
    beta_fit, tni_fit = best_fit
    if np.isfinite(tni_fit) and temps_subset.min() <= tni_fit <= temps_subset.max():
        return float(tni_fit), float(beta_fit)
    return None


def load_density(system: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load (or compute) density vs temperature.

    Strategy:
      1. Try reading the raw .edr energy file and extracting the Density series.
      2. If the .edr is unavailable, fall back to cached _rho.npy files.
      3. If computed from .edr, cache the result for future runs.

    Returns
    -------
    temps, means, errs : arrays (every-other-point for plotting)
    """
    temps: list[float] = []
    means: list[float] = []
    errs: list[float] = []
    edr_template = DENSITY_EDR.get(system)

    for temp in TEMPERATURES:
        temp_tag = _temp_tag(temp)
        mean = None
        sem = None

        # --- try raw .edr first --------------------------------------------
        if edr_template is not None:
            edr_path = edr_template.format(T=temp_tag)
            if os.path.isfile(edr_path):
                try:
                    dens_series = read_density_series(edr_path)
                    mean, sem, g, neff = block_avg_and_error(dens_series)
                    print(f"[rho] {system} {temp}K: mean={mean:.2f}, stderr={sem:.2f}")
                    # cache for next time
                    _save_series(DENSITY_DIR, system, temp, np.array([mean, sem]), suffix="_rho")
                except Exception:
                    mean = None
                    sem = None

        # --- fall back to cached .npy --------------------------------------
        if mean is None or sem is None:
            cached = _load_series(DENSITY_DIR, system, temp, suffix="_rho")
            if cached is not None and cached.size >= 2:
                mean = float(cached[0])
                sem = float(cached[1])

        if mean is None or sem is None:
            continue
        temps.append(temp)
        means.append(mean)
        errs.append(sem)

    if not temps:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(temps)
    temps_arr = np.asarray(temps)[order]
    means_arr = np.asarray(means)[order]
    errs_arr = np.asarray(errs)[order]
    plot_mask = np.arange(temps_arr.size) % 2 == 0
    return temps_arr[plot_mask], means_arr[plot_mask], errs_arr[plot_mask]


# ===========================================================================
#  PLOTTING
# ===========================================================================

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

    # Preload density data so S can be aligned to the same temperature grid
    density_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
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
            fmt=SYSTEM_MARKERS[system],
            markersize=fp.MARKER_SIZE,
            mfc=fp.SYSTEM_COLORS[system],
            mec="black",
            mew=fp.MARKER_EDGEWIDTH,
            ecolor=fp.SYSTEM_COLORS[system],
            elinewidth=fp.MARKER_EDGEWIDTH,
            capsize=0,
            alpha=0.9,
            label=SYSTEM_LEGEND_LABELS[system],
            zorder=3,
        )
        tni_est = _estimate_tni(temps_full, means_full, system)
        if tni_est is not None:
            tni_val, beta_val = tni_est
            print(f"[TNI] {system}: beta={beta_val:.3f}, TNI={tni_val:.2f} K")
            if temps_full.min() <= tni_val <= temps_full.max() and XLIM[0] <= tni_val <= XLIM[1]:
                ax_s.axvline(
                    tni_val,
                    color=fp.SYSTEM_COLORS[system],
                    linestyle="--",
                    linewidth=1.2,
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
        temps, means, stds = density_cache.get(system, (np.array([]), np.array([]), np.array([])))
        if temps.size == 0:
            continue
        ax_rho.errorbar(
            temps,
            means,
            yerr=stds,
            fmt=SYSTEM_MARKERS[system],
            markersize=fp.MARKER_SIZE,
            mfc=fp.SYSTEM_COLORS[system],
            mec="black",
            mew=fp.MARKER_EDGEWIDTH,
            ecolor=fp.SYSTEM_COLORS[system],
            elinewidth=fp.MARKER_EDGEWIDTH,
            capsize=0,
            alpha=0.9,
            zorder=3,
        )
    ax_rho.set_xlabel(r"Temperature, $T$ (K)")
    ax_rho.set_ylabel("Density,\n" + r"$\rho$ (kg m$^{-3}$)")
    ax_rho.set_ylim(*DENSITY_YLIM)
    ax_rho.set_yticks(np.arange(DENSITY_YLIM[0] + 50, DENSITY_YLIM[1] + 1, 50))
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

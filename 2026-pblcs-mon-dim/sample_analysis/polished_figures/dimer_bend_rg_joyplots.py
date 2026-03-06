#!/usr/bin/env python3
"""Ridgeline (joy) plots for bend angle and Rg from cached .npy arrays.

Computes bend angle and Rg from dimer trajectories if cached data is not found.

Inputs (cached):
  DATA_ROOT/bend_angle_distributions/bend_angles/
    +-- PM/380K/bend_angle_380K.npy, rg_380K.npy
    +-- MP/...
    +-- P2/...
    +-- M2/...

Outputs (SVG) in this directory:
  - fig7_joy_bend_angle.svg
  - fig7_joy_rg.svg
  - fig7_joyplot_key.svg (systems color key)

The joy plots stack temperature ridges vertically (one ridge per available T),
with all four systems overlaid within each ridge in the proper colors.
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
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from scipy.stats import gaussian_kde

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---- Figure params ----
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

# ---- paths ----
BASE_DIR = Path(DATA_ROOT) / "bend_angle_distributions" / "bend_angles"
BASE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Only plot these temperatures
ALLOWED_TEMPS = [f"{t}K" for t in [400, 410, 420, 430, 440, 450, 460]]

# ---- style ----
fp.use_mpl_defaults()

# Colors & labels per system (match your bend_angle.py)
DIMER_COLOR = fp.DIMER_COLORS
DIMER_LABEL = fp.DIMER_LABELS
SYSTEM_ORDER = ["PM", "MP", "P2", "M2"]  # plotting order within each ridge

# All dimer temperatures
DIMER_TEMPS = [int(t) if t % 1 == 0 else round(t, 1) for t in np.arange(375, 476, 2.5)]
FRAME_SKIP = 25  # frame stride for trajectory analysis

KDE_POINTS = 400        # x-samples per KDE curve
RIDGE_GAP  = 1.05       # vertical distance between ridge baselines
RIDGE_SCALE = 0.9       # scale of peak height relative to gap (avoid overlap)
FILL_ALPHA = fp.FILL_ALPHA
CURVE_LW = fp.AXES_LINEWIDTH


# ====================== COMPUTATION CODE ======================
# Adapted from bend_angle_distributions/bend_angles/bend_angle.py

def _compute_bend_angles_for_system(sys_name: str, T: int, base_dir: Path) -> np.ndarray:
    """
    Compute bend angles for a dimer system at temperature T.
    Returns array of shape (n_frames, n_fragments).
    """
    import MDAnalysis as mda
    from MDAnalysis.transformations import unwrap
    from rdkit import Chem

    top = os.path.join(SIMULATION_ROOT, "dimers", sys_name, "simulations", "tREM", f"{T}K", "trem_gpu.tpr")
    xtc = os.path.join(SIMULATION_ROOT, "dimers", sys_name, "simulations", "tREM", f"{T}K", "trem_gpu.xtc")

    if not (os.path.isfile(top) and os.path.isfile(xtc)):
        print(f"[skip] {sys_name} {T} K - missing trajectory files")
        return np.array([])

    u = mda.Universe(top, xtc)
    u.trajectory.add_transformations(unwrap(u.atoms))

    smarts = Chem.MolFromSmarts("c1ccccc1COc2ccccc2")

    frames = list(u.trajectory[0:-1:FRAME_SKIP])
    n_frames = len(frames)
    n_frags = len(u.atoms.fragments)
    results = np.zeros((n_frames, n_frags))

    for f_idx, _ in enumerate(frames):
        for frag_idx, frag in enumerate(u.atoms.fragments):
            rdmol = frag.convert_to("RDKIT")
            directors = []
            for match in rdmol.GetSubstructMatches(smarts):
                rings = [
                    ring
                    for ring in rdmol.GetRingInfo().AtomRings()
                    if set(ring).issubset(match)
                ]
                if len(rings) != 2:
                    raise ValueError(f"Expected 2 rings, got {len(rings)}")

                coms = [frag.atoms[list(r)].center_of_mass() for r in rings]
                vec = coms[1] - coms[0]
                vec /= np.linalg.norm(vec)
                directors.append(vec)

            if len(directors) != 2:
                raise ValueError(f"Expected 2 directors, got {len(directors)}")

            cosang = np.clip(np.dot(directors[0], directors[1]), -1.0, 1.0)
            results[f_idx, frag_idx] = np.degrees(np.arccos(cosang))

    return results


def _compute_rg_values_for_system(sys_name: str, T: int) -> np.ndarray:
    """
    Compute Rg values for a dimer system at temperature T.
    Returns array of Rg values.
    """
    import MDAnalysis as mda
    from MDAnalysis.transformations import unwrap

    top = os.path.join(SIMULATION_ROOT, "dimers", sys_name, "simulations", "tREM", f"{T}K", "trem_gpu.tpr")
    xtc = os.path.join(SIMULATION_ROOT, "dimers", sys_name, "simulations", "tREM", f"{T}K", "trem_gpu.xtc")

    if not (os.path.isfile(top) and os.path.isfile(xtc)):
        print(f"[skip] {sys_name} {T} K - missing trajectory files")
        return np.array([])

    u = mda.Universe(top, xtc)
    u.trajectory.add_transformations(unwrap(u.atoms))
    vals = []
    for ts in u.trajectory[::FRAME_SKIP]:
        for frag in u.atoms.fragments:
            com = frag.atoms.center_of_mass()
            r = frag.atoms.positions - com
            G = np.einsum('im,in->mn', r, r) / r.shape[0]
            rg = float(np.sqrt(np.trace(G)))
            vals.append(rg)
    return np.asarray(vals, dtype=float)


def _ensure_cached_data(base_dir: Path, systems: List[str], temps_to_check: List[str]) -> None:
    """Check for cached .npy files and compute missing ones from trajectories."""
    for sys_name in systems:
        for Tdir in temps_to_check:
            m = TEMP_DIR_RE.match(Tdir)
            if not m:
                continue
            T_val = float(m.group(1))
            T_int = int(T_val) if T_val == int(T_val) else T_val

            bend_path = base_dir / sys_name / Tdir / f"bend_angle_{Tdir}.npy"
            rg_path = base_dir / sys_name / Tdir / f"rg_{Tdir}.npy"

            if bend_path.is_file() and rg_path.is_file():
                continue  # already cached

            # Create directory
            (base_dir / sys_name / Tdir).mkdir(parents=True, exist_ok=True)

            if not bend_path.is_file():
                print(f"[compute] bend_angle {sys_name} {Tdir}")
                angles = _compute_bend_angles_for_system(sys_name, T_int, base_dir)
                if angles.size > 0:
                    np.save(bend_path, angles)
                    print(f"[cached] {bend_path}")

            if not rg_path.is_file():
                print(f"[compute] rg {sys_name} {Tdir}")
                rg_vals = _compute_rg_values_for_system(sys_name, T_int)
                if rg_vals.size > 0:
                    np.save(rg_path, rg_vals)
                    print(f"[cached] {rg_path}")


# ====================== ORIGINAL PLOTTING CODE ======================

# ---- helpers ----
TEMP_DIR_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)K$")

def _find_systems(base_dir: Path) -> List[str]:
    systems = []
    if base_dir.is_dir():
        for path in sorted(base_dir.iterdir()):
            if path.is_dir():
                systems.append(path.name)
    # Keep only known systems if present
    systems = [s for s in SYSTEM_ORDER if s in systems] or systems
    # If no directories found, return all known systems for compute
    if not systems:
        systems = list(SYSTEM_ORDER)
    return systems


def _find_temps(base_dir: Path, systems: List[str]) -> List[str]:
    temps = set()
    for s in systems:
        sdir = base_dir / s
        if not sdir.is_dir():
            continue
        for path in sdir.iterdir():
            if path.is_dir() and TEMP_DIR_RE.match(path.name):
                temps.add(path.name)
    # If no cached temps found, generate from DIMER_TEMPS for computing
    if not temps:
        temps = {f"{int(t)}K" if t == int(t) else f"{t}K" for t in DIMER_TEMPS}
    # Sort numerically by the value before 'K'
    def _key(ts: str) -> float:
        return float(TEMP_DIR_RE.match(ts).group(1)) if TEMP_DIR_RE.match(ts) else math.inf
    return sorted(temps, key=_key)


def _load_npy(base_dir: Path, system: str, Tdir: str, kind: str) -> np.ndarray | None:
    """Load bend_angle or rg npy for a given (system, Tdir). Returns 1D array or None."""
    fname = f"{kind}_{Tdir}.npy"  # e.g., bend_angle_380K.npy, rg_380K.npy
    path = base_dir / system / Tdir / fname
    if path.is_file():
        try:
            arr = np.load(path)
            return np.asarray(arr).ravel()
        except Exception:
            return None
    return None


def _kde(vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    if vals is None or vals.size < 2 or not np.isfinite(vals).any():
        return np.zeros_like(x)
    try:
        kde = gaussian_kde(vals[np.isfinite(vals)])
        return kde(x)
    except Exception:
        return np.zeros_like(x)


def _global_range(base_dir: Path, systems: List[str], temps: List[str], kind: str) -> Tuple[float, float]:
    xmin, xmax = math.inf, -math.inf
    for s in systems:
        for Tdir in temps:
            arr = _load_npy(base_dir, s, Tdir, kind)
            if arr is None or arr.size == 0:
                continue
            a = np.asarray(arr, dtype=float)
            if not np.isfinite(a).any():
                continue
            xmin = min(xmin, float(np.nanmin(a)))
            xmax = max(xmax, float(np.nanmax(a)))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        # Sensible defaults
        return (0.0, 1.0)
    pad = 0.03 * (xmax - xmin)
    return (xmin - pad, xmax + pad)


# ---- plotting ----

def joyplot(kind: str, xlabel: str, xlim_hint: Tuple[float, float] | None, outfile: str) -> None:
    """Build one joy plot (ridgeline) for the given kind in {"bend_angle", "rg"}."""
    systems = _find_systems(BASE_DIR)
    temps   = _find_temps(BASE_DIR, systems)

    # Ensure cached data exists for requested temperatures
    requested_temps = [t for t in temps if t in ALLOWED_TEMPS]
    _ensure_cached_data(BASE_DIR, systems, requested_temps)

    # Re-discover after potential computation
    systems = _find_systems(BASE_DIR)
    temps = _find_temps(BASE_DIR, systems)

    # Drop temperatures that have no data across all systems
    have_any = []
    for Tdir in temps:
        ok = False
        for s in systems:
            if _load_npy(BASE_DIR, s, Tdir, kind) is not None:
                ok = True
                break
        if ok:
            have_any.append(Tdir)
    temps = have_any

    # Keep only the requested temperatures
    temps = [t for t in temps if t in ALLOWED_TEMPS]
    if not temps:
        print(f"[warn] No allowed temperatures present for kind={kind}; skipping {outfile}")
        return

    # X-range
    if xlim_hint is None:
        xmin, xmax = _global_range(BASE_DIR, systems, temps, kind)
    else:
        xmin, xmax = xlim_hint
    xs = np.linspace(xmin, xmax, KDE_POINTS)

    n_ridges = len(temps)
    fig_h = 0.3 * n_ridges + 1.05  # slightly taller for better separation
    fig = plt.figure(figsize=(1.8, fig_h))  # 10% narrower
    ax = fig.add_subplot(111)

    # Draw each temperature ridge
    for i, Tdir in enumerate(temps):
        offset = (n_ridges - 1 - i) * RIDGE_GAP  # coldest (first) on top, hottest on bottom
        # Compute all system KDEs for this temperature
        curves = {}
        local_max = 0.0
        for s in systems:
            arr = _load_npy(BASE_DIR, s, Tdir, kind)
            if arr is None or arr.size == 0:
                continue
            ys = _kde(arr, xs)
            if ys.size:
                local_max = max(local_max, float(np.max(ys)))
                curves[s] = ys
        if local_max <= 0:
            continue
        scale = (RIDGE_SCALE / local_max)
        # Fill and stroke for each system in a consistent order
        for s in systems:
            ys = curves.get(s)
            if ys is None:
                continue
            y = offset + scale * ys
            ax.fill_between(xs, offset, y, color=DIMER_COLOR[s], alpha=FILL_ALPHA, linewidth=0)
            ax.plot(xs, y, color=DIMER_COLOR[s], lw=CURVE_LW)

    # Aesthetics
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.5 * RIDGE_GAP, (n_ridges - 1) * RIDGE_GAP + RIDGE_GAP)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=fp.LABEL_FONTSIZE)
    ax.tick_params(
        direction="in",
        width=fp.TICK_WIDTH,
        length=fp.TICK_LENGTH,
        labelsize=fp.TICK_FONTSIZE,
        top=False,
    )
    ax.xaxis.set_ticks_position("bottom")
    for spine in ax.spines.values():
        spine.set_linewidth(fp.AXES_LINEWIDTH)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Keep x-axis ticks reasonable: about 4 ticks
    if kind == "rg":
        # exact ticks from left to right limit at 2.5 spacing
        try:
            left, right = ax.get_xlim()
        except Exception:
            left, right = xmin, xmax
        # round to avoid fp drift, and ensure inclusion of endpoints
        ticks = list(np.arange(left, right + 1e-9, 2.5))
        ax.xaxis.set_major_locator(FixedLocator(ticks))
    elif kind == "bend_angle":
        # evenly spaced ticks starting at 0 and ending at 180
        ax.set_xlim(0.0, 180.0)
        ticks = list(np.arange(0.0, 180.0 + 1e-9, 60.0))  # 0, 60, 120, 180
        ax.xaxis.set_major_locator(FixedLocator(ticks))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    plt.tight_layout()
    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=fp.FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_key(outfile: str = "fig7_joyplot_key.svg") -> None:
    fig = plt.figure(figsize=(1.0, 1.0))
    ax  = fig.add_subplot(111)
    for s in SYSTEM_ORDER:
        ax.plot([], [], color=DIMER_COLOR[s], lw=CURVE_LW, label=DIMER_LABEL[s])
    ax.legend(frameon=False, fontsize=fp.LEGEND_FONTSIZE)
    ax.axis("off")
    fig.tight_layout()
    path = OUT_DIR / outfile
    fig.savefig(path, dpi=fp.FIG_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    # Bend angle: natural range 0..180 deg
    joyplot(
        kind="bend_angle",
        xlabel=r"bend angle ($^\circ$)",
        xlim_hint=(0.0, 180.0),
        outfile="fig7_joy_bend_angle.svg",
    )

    # Rg: compute range from data (typ. ~8-18 Ang for dimers)
    joyplot(
        kind="rg",
        xlabel=r"$R_{\mathrm{g}}$ ($\AA$)",
        xlim_hint=(7.0, 17.0),
        outfile="fig7_joy_rg.svg",
    )

    # Separate color key
    save_key("fig7_joyplot_key.svg")


if __name__ == "__main__":
    main()

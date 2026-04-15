#!/usr/bin/env python3
"""
Plot nematic order parameter time series for each system and temperature.

Can load pre-cached S(t) .npy arrays **or** recompute them from raw GROMACS
trajectories (tpr + xtc) using ``--compute``.  The computation extracts
mesogenic-core director vectors via a SMARTS pattern on each fragment and
calculates the frame-wise nematic order parameter S(t).

Time axis is inferred from the trajectory metadata so it is in ns.
"""

# ---------------------------------------------------------------------------
# DATA PATHS -- adjust these to point to your local data locations
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/paper_1_figures")
SIMULATION_ROOT = os.environ.get("SIMULATION_ROOT", "/scratch/gpfs/WEBB/jv6139")
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

MONOMER_PHASE_DIR = Path(DATA_ROOT) / "monomer_phase_density" / "phase_diagram"
DIMER_PHASE_DIR = Path(DATA_ROOT) / "dimer_phase_density" / "phase_diagram"


# ---------------------------------------------------------------------------
# Computation from raw trajectories  (used with --compute)
# ---------------------------------------------------------------------------
# The SMARTS pattern matches the phenyl-O-phenyl mesogenic core common to all
# IEG monomers/dimers studied here.
_CORE_SMARTS_STR = "c1ccccc1COc2ccccc2"


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

    Parameters
    ----------
    top : str
        Path to a GROMACS .tpr (or other topology) file.
    traj : str
        Path to the corresponding .xtc trajectory.
    start, stop, skip : int
        Slice parameters for the trajectory.

    Returns
    -------
    directors : np.ndarray, shape (n_frames, n_frag, max_cores, 3)
        Unit director vectors; entries are NaN where a core is absent.
    """
    import MDAnalysis as mda
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    _SMARTS = Chem.MolFromSmarts(_CORE_SMARTS_STR)
    ring_query = Chem.MolFromSmarts("c1ccccc1")

    u = mda.Universe(top, traj)

    # -- pre-compute per-fragment core ring pairs ----------------------------
    frag_core_pairs = []
    for frag in u.atoms.fragments:
        rdmol = frag.convert_to("RDKIT")
        ring_info = rdmol.GetRingInfo().AtomRings()
        matches = rdmol.GetSubstructMatches(_SMARTS, uniquify=True)

        core_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for m in matches:
            mring = [r for r in ring_info if len(r) == 6 and set(r).issubset(m)]
            if len(mring) != 2:
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
                    v -= box * np.round(v / box)

                norm = np.linalg.norm(v)
                if norm > 1e-8:
                    directors[ti, fi, ci] = v / norm

    return directors


def calc_S(directors: np.ndarray) -> np.ndarray:
    """
    Frame-wise nematic order parameter from director vectors.

    Parameters
    ----------
    directors : np.ndarray, shape (n_frames, n_frag, n_core, 3)

    Returns
    -------
    S : np.ndarray, shape (n_frames,)
        Largest eigenvalue of the per-frame Q tensor.
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


def compute_and_save_series(
    tpr_path: str,
    xtc_path: str,
    out_path: Path,
    *,
    skip: int = 10,
) -> np.ndarray | None:
    """
    Compute S(t) from a trajectory pair and save the result as a .npy file.

    Returns the S(t) array, or None on failure.
    """
    if not (os.path.isfile(tpr_path) and os.path.isfile(xtc_path)):
        print(f"[skip] missing trajectory files: {tpr_path} / {xtc_path}")
        return None
    try:
        directors = get_directors_smarts(tpr_path, xtc_path, start=0, stop=None, skip=skip)
        S = calc_S(directors)
    except Exception as exc:
        print(f"[error] computation failed for {tpr_path}: {exc}")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, S)
    print(f"[computed] {out_path}  ({len(S)} frames)")
    return S


SYSTEM_CONFIG = {
    "P1": {
        "series_dir": MONOMER_PHASE_DIR,
        "series_prefix": "P_monomer",
        "tpr": f"{SIMULATION_ROOT}/N_monomer/simulations/tREM/{{T}}K/trem_gpu.tpr",
        "xtc": f"{SIMULATION_ROOT}/N_monomer/simulations/tREM/{{T}}K/trem_gpu.xtc",
    },
    "M1": {
        "series_dir": MONOMER_PHASE_DIR,
        "series_prefix": "M_monomer",
        "tpr": f"{SIMULATION_ROOT}/M_monomer/simulations/tREM/{{T}}K/trem_gpu.tpr",
        "xtc": f"{SIMULATION_ROOT}/M_monomer/simulations/tREM/{{T}}K/trem_gpu.xtc",
    },
    "PM": {
        "series_dir": DIMER_PHASE_DIR,
        "series_prefix": "PM",
        "tpr": f"{SIMULATION_ROOT}/dimers/PM/simulations/tREM/{{T}}K/trem_gpu.tpr",
        "xtc": f"{SIMULATION_ROOT}/dimers/PM/simulations/tREM/{{T}}K/trem_gpu.xtc",
    },
    "MP": {
        "series_dir": DIMER_PHASE_DIR,
        "series_prefix": "MP",
        "tpr": f"{SIMULATION_ROOT}/dimers/MP/simulations/tREM/{{T}}K/trem_gpu.tpr",
        "xtc": f"{SIMULATION_ROOT}/dimers/MP/simulations/tREM/{{T}}K/trem_gpu.xtc",
    },
    "P2": {
        "series_dir": DIMER_PHASE_DIR,
        "series_prefix": "P2",
        "tpr": f"{SIMULATION_ROOT}/dimers/P2/simulations/tREM/{{T}}K/trem_gpu.tpr",
        "xtc": f"{SIMULATION_ROOT}/dimers/P2/simulations/tREM/{{T}}K/trem_gpu.xtc",
    },
    "M2": {
        "series_dir": DIMER_PHASE_DIR,
        "series_prefix": "M2",
        "tpr": f"{SIMULATION_ROOT}/dimers/M2/simulations/tREM/{{T}}K/trem_gpu.tpr",
        "xtc": f"{SIMULATION_ROOT}/dimers/M2/simulations/tREM/{{T}}K/trem_gpu.xtc",
    },
}


def _temp_tag(temp: float) -> str:
    return f"{temp:.1f}".rstrip("0").rstrip(".")


def _load_series(
    series_dir: Path,
    prefix: str,
    temp: float,
    *,
    compute: bool = False,
    tpr_path: str | None = None,
    xtc_path: str | None = None,
    skip: int = 10,
) -> np.ndarray | None:
    tag = _temp_tag(temp)
    candidates = [
        series_dir / f"{prefix}_{tag}K.npy",
        series_dir / f"{prefix}_{tag}.npy",
    ]
    for path in candidates:
        if path.is_file():
            return np.load(path)
    # If no cached file and --compute was requested, generate from trajectory
    if compute and tpr_path and xtc_path:
        out_path = series_dir / f"{prefix}_{tag}K.npy"
        return compute_and_save_series(tpr_path, xtc_path, out_path, skip=skip)
    return None


def _available_temps(series_dir: Path, prefix: str) -> List[float]:
    temps = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(.+?)K?$")
    for path in series_dir.glob(f"{prefix}_*.npy"):
        match = pattern.match(path.stem)
        if not match:
            continue
        raw = match.group(1)
        try:
            temps.append(float(raw))
        except ValueError:
            continue
    temps = sorted(set(temps))
    return temps


def _trajectory_time_info(tpr_path: str, xtc_path: str) -> tuple[int, float, float]:
    import MDAnalysis as mda

    if not (os.path.isfile(tpr_path) and os.path.isfile(xtc_path)):
        raise FileNotFoundError(f"Missing trajectory: {tpr_path} or {xtc_path}")
    u = mda.Universe(tpr_path, xtc_path)
    n_frames = len(u.trajectory)
    u.trajectory[0]
    t0 = float(u.trajectory.time)
    dt_ps = getattr(u.trajectory, "dt", None)
    if dt_ps is None or not np.isfinite(dt_ps) or dt_ps == 0.0:
        if n_frames > 1:
            u.trajectory[1]
            dt_ps = float(u.trajectory.time) - t0
        else:
            dt_ps = 1.0
    return n_frames, float(dt_ps) / 1000.0, t0 / 1000.0


def _time_axis(
    series: np.ndarray,
    tpr_path: str,
    xtc_path: str,
    *,
    time_start_ns: float | None = None,
    time_end_ns: float | None = None,
    time_offset_ns: float = 0.0,
    run_index: int | None = None,
    run_length_ns: float | None = None,
) -> tuple[np.ndarray, int]:
    n_frames, dt_ns, t0_ns = _trajectory_time_info(tpr_path, xtc_path)
    if len(series) == 0:
        return np.array([]), 1
    if run_length_ns is not None:
        total_duration = float(run_length_ns)
    else:
        total_duration = (n_frames - 1) * dt_ns if n_frames > 1 else 0.0
    skip = max(1, int(round(n_frames / len(series))))
    if time_end_ns is not None:
        t0_ns = time_end_ns - total_duration
    elif time_start_ns is not None:
        t0_ns = time_start_ns
    elif run_index is not None:
        t0_ns = (run_index - 1) * total_duration
    t0_ns += time_offset_ns
    times = t0_ns + np.arange(len(series), dtype=float) * dt_ns * skip
    return times, skip


def _infer_run_index(path: str) -> int | None:
    match = re.search(r"trem(\d+)", path)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _plot_system(
    system_key: str,
    config: dict,
    *,
    time_start_ns: float | None = None,
    time_end_ns: float | None = None,
    time_offset_ns: float = 0.0,
    run_length_ns: float | None = None,
    compute: bool = False,
    compute_skip: int = 10,
) -> None:
    series_dir = Path(config["series_dir"])
    prefix = config["series_prefix"]
    temps = _available_temps(series_dir, prefix)
    if not temps:
        return

    cmap = get_cmap("viridis")
    norm = Normalize(vmin=min(temps), vmax=max(temps))

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    fp.use_mpl_defaults()

    for temp in temps:
        tpr_path = config["tpr"].format(T=_temp_tag(temp))
        xtc_path = config["xtc"].format(T=_temp_tag(temp))
        series = _load_series(
            series_dir,
            prefix,
            temp,
            compute=compute,
            tpr_path=tpr_path,
            xtc_path=xtc_path,
            skip=compute_skip,
        )
        if series is None or series.size == 0:
            continue
        run_index = _infer_run_index(tpr_path) or _infer_run_index(xtc_path)
        try:
            time_ns, skip = _time_axis(
                series,
                tpr_path,
                xtc_path,
                time_start_ns=time_start_ns,
                time_end_ns=time_end_ns,
                time_offset_ns=time_offset_ns,
                run_index=run_index,
                run_length_ns=run_length_ns,
            )
        except FileNotFoundError:
            continue
        if time_ns.size != series.size:
            continue
        if skip != 1:
            print(f"[warn] {system_key} {temp}K inferred skip={skip} from trajectory vs series length.")
        color = cmap(norm(temp))
        t_plot = time_ns[::2]
        s_plot = series[::2]
        ax.plot(
            t_plot,
            s_plot,
            color=color,
            lw=fp.AXES_LINEWIDTH,
            linestyle="--",
        )
        ax.scatter(
            t_plot,
            s_plot,
            s=fp.MARKER_SIZE ** 2,
            color=color,
            edgecolors="none",
        )

    ax.set_xlabel("Time (ns)", fontsize=fp.LABEL_FONTSIZE)
    ax.set_ylabel("Nematic order parameter", fontsize=fp.LABEL_FONTSIZE)
    ax.tick_params(
        direction="in",
        length=fp.TICK_LENGTH,
        width=fp.TICK_WIDTH,
        labelsize=fp.TICK_FONTSIZE,
        top=True,
        right=True,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(fp.AXES_LINEWIDTH)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Temperature (K)", fontsize=fp.LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=fp.TICK_FONTSIZE, width=fp.TICK_WIDTH, length=fp.TICK_LENGTH)

    out_dir = SCRIPT_DIR / system_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{system_key}_nematic_timeseries.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=fp.FIG_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nematic order parameter time series plots")
    parser.add_argument("--time-start-ns", type=float, default=None)
    parser.add_argument("--time-end-ns", type=float, default=None)
    parser.add_argument("--time-offset-ns", type=float, default=0.0)
    parser.add_argument("--run-length-ns", type=float, default=None)
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Recompute S(t) from raw trajectories when cached .npy files are missing.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=10,
        help="Frame stride when computing S(t) from trajectories (default: 10).",
    )
    args = parser.parse_args()

    for system_key, config in SYSTEM_CONFIG.items():
        _plot_system(
            system_key,
            config,
            time_start_ns=args.time_start_ns,
            time_end_ns=args.time_end_ns,
            time_offset_ns=args.time_offset_ns,
            run_length_ns=args.run_length_ns,
            compute=args.compute,
            compute_skip=args.skip,
        )


if __name__ == "__main__":
    main()

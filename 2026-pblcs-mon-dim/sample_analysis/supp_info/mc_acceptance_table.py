#!/usr/bin/env python3
"""
Build a summary table PNG of replica-exchange acceptance probabilities
per system from production GROMACS log files.
"""

# ---------------------------------------------------------------------------
# DATA PATHS -- adjust these to point to your local data locations
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

SIMULATION_ROOT = os.environ.get("SIMULATION_ROOT", "/scratch/gpfs/WEBB/jv6139")

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

SYSTEM_LOG_PATTERNS = {
    "P1": f"{SIMULATION_ROOT}/N_monomer/simulations/tREM/*K/trem_gpu.log",
    "M1": f"{SIMULATION_ROOT}/M_monomer/simulations/tREM/*K/trem_gpu.log",
    "PM": f"{SIMULATION_ROOT}/dimers/PM/simulations/tREM/*K/trem_gpu.log",
    "MP": f"{SIMULATION_ROOT}/dimers/MP/simulations/tREM/*K/trem_gpu.log",
    "P2": f"{SIMULATION_ROOT}/dimers/P2/simulations/tREM/*K/trem_gpu.log",
    "M2": f"{SIMULATION_ROOT}/dimers/M2/simulations/tREM/*K/trem_gpu.log",
}
SYSTEM_ORDER = ["M1", "P1", "MP", "PM", "P2", "M2"]


def _parse_acceptance_series(log_path: Path) -> Tuple[List[int], List[float]] | None:
    if not log_path.is_file():
        return None
    lines = log_path.read_text(errors="ignore").splitlines()
    in_block = False
    indices: List[int] = []
    values: List[float] = []
    for line in lines:
        if "Repl  average probabilities" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if "Repl  number of exchanges" in line:
            break
        row = line.strip()
        if not row.startswith("Repl"):
            continue
        tokens = row.split()[1:]
        if tokens and all(tok.isdigit() for tok in tokens):
            indices = [int(tok) for tok in tokens]
            continue
        row_vals: List[float] = []
        for tok in tokens:
            if any(ch in tok for ch in (".", "e", "E")) or tok.startswith("."):
                try:
                    row_vals.append(float(tok))
                except ValueError:
                    continue
        if row_vals:
            values.extend(row_vals)
        if indices and len(values) >= len(indices):
            break
    if indices and values:
        return indices, values[: len(indices)]
    return None


def _extract_temp(log_path: Path) -> float | None:
    for part in log_path.parts:
        if part.endswith("K"):
            try:
                return float(part[:-1])
            except ValueError:
                continue
    return None


def _collect_acceptance_by_temp(pattern: str) -> Tuple[List[int], Dict[float, List[float]]]:
    from glob import glob

    indices_ref: List[int] = []
    buckets: Dict[float, List[List[float]]] = {}
    for match in sorted(glob(pattern)):
        path = Path(match)
        temp = _extract_temp(path)
        if temp is None:
            continue
        parsed = _parse_acceptance_series(path)
        if parsed is None:
            continue
        indices, values = parsed
        if not indices_ref:
            indices_ref = indices
        if indices_ref and len(indices) != len(indices_ref):
            continue
        buckets.setdefault(temp, []).append(values)
    out: Dict[float, List[float]] = {}
    for temp, series in buckets.items():
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            continue
        out[temp] = np.nanmean(arr, axis=0).tolist()
    return indices_ref, out


def _format_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _build_system_table(data: Dict[str, float | None]) -> None:
    fp.use_mpl_defaults()
    if not data:
        return
    ordered = [s for s in SYSTEM_ORDER if s in data]
    ordered.extend([s for s in data.keys() if s not in ordered])

    cell_text = [[_system_label(s), _format_pct(data[s])] for s in ordered]
    fig_w = 2.4
    fig_h = 0.4 + 0.28 * max(1, len(cell_text))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["System", "Avg acceptance"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fp.TICK_FONTSIZE)
    table.scale(1.0, 1.2)
    for key, cell in table.get_celld().items():
        cell.set_linewidth(fp.AXES_LINEWIDTH)
        cell.set_edgecolor("black")

    fig.tight_layout(pad=0.2)
    out_path = SCRIPT_DIR / "mc_acceptance_table.png"
    fig.savefig(out_path, dpi=fp.FIG_DPI)
    plt.close(fig)


def _system_color(system: str) -> str:
    if system == "P1":
        return fp.SYSTEM_COLORS.get("P_monomer", "#4d4d4d")
    if system == "M1":
        return fp.SYSTEM_COLORS.get("M_monomer", "#4d4d4d")
    return fp.DIMER_COLORS.get(system, "#4d4d4d")


def _system_label(system: str) -> str:
    if system == "P1":
        return fp.SYSTEM_LABELS.get("P_monomer", system)
    if system == "M1":
        return fp.SYSTEM_LABELS.get("M_monomer", system)
    return fp.DIMER_LABELS.get(system, system)


def _build_system_chart(data: Dict[str, float | None]) -> None:
    fp.use_mpl_defaults()
    systems = list(data.keys())
    values = [data[k] if data[k] is not None else np.nan for k in systems]
    labels = [_system_label(s) for s in systems]
    colors = [_system_color(s) for s in systems]

    fig_w = 3.2
    fig_h = 1.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(len(systems))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=fp.AXES_LINEWIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fp.TICK_FONTSIZE)
    ax.set_ylabel("Acceptance (%)", fontsize=fp.LABEL_FONTSIZE)
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
    fig.tight_layout()
    out_path = SCRIPT_DIR / "mc_acceptance_by_system.png"
    fig.savefig(out_path, dpi=fp.FIG_DPI)
    plt.close(fig)


def main() -> None:
    system_means: Dict[str, float | None] = {}
    for system, pattern in SYSTEM_LOG_PATTERNS.items():
        _, data = _collect_acceptance_by_temp(pattern)
        if data:
            flat_vals = [np.nanmean(row) for row in data.values()]
            system_means[system] = float(np.nanmean(flat_vals))
        else:
            system_means[system] = None
    _build_system_table(system_means)
    _build_system_chart(system_means)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
#!/usr/bin/env python3
"""Parity plots (before/after) for QM vs MM energies across torsions.

Supports 3 display modes:
- scatter      : individual points
- hex          : per-monomer hexbin density layers (separate colors)
- bivariate    : single hex layer colored by *mixture* (hue = P<->M fraction,
                 brightness = total density, log-scaled)

Monomers and directories:
- N_monomer -> plotted as P_monomer (label $P_1$, color #90c0c8)
- M_monomer -> plotted as M_monomer (label $M_1$, color #c890c0)

The script searches under:
  DATA_ROOT/{N_monomer,M_monomer}/.../stage_0/optimize.tmp/torsion-*/iter_XXXX/EnergyCompare.txt

"Before" = iter_0000, "After" = lexicographically last iter_XXXX.
Outputs:
- parity_panel.svg                 (before|after side-by-side, shared y; legend in "after")
- parity_density_bars.svg          (two colorbars, one per monomer; hex mode)
- parity_bivar_ratio_bar.svg       (P<->M fraction bar; bivariate mode)
- parity_bivar_density_bar.svg     (total density (log) bar; bivariate mode)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# DATA PATHS -- adjust these to point to your local data locations
# DATA_ROOT: root of the openff_QM_optimization tree
# ---------------------------------------------------------------------------
import os
DATA_ROOT = os.environ.get("QM_DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/openff_QM_optimization")

from pathlib import Path
from typing import Dict, List, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.cm import ScalarMappable

# Use Matplotlib mathtext (no LaTeX); keep default font
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
})

# ---- Parity R^2 using Pearson correlation (no regression, no fallbacks) ----
def _r2_corrcoef(qm_vals, mm_vals) -> float:
    x = np.asarray(qm_vals, dtype=float)
    y = np.asarray(mm_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float('nan')
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r)

# ----------------------- Paths ----------------------- #
BASE_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

_DATA_ROOT = Path(DATA_ROOT)
OUTPUT_DIR = BASE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------- Styling ----------------------- #
def _upright_label(label: str) -> str:
    if label == r"$P_1$":
        return r"$\mathrm{P}_1$"
    if label == r"$M_1$":
        return r"$\mathrm{M}_1$"
    return label

COLORS: Dict[str, str] = {
    "P_monomer": fp.SYSTEM_COLORS["P_monomer"],
    "M_monomer": fp.SYSTEM_COLORS["M_monomer"],
}
LABELS: Dict[str, str] = {
    "P_monomer": _upright_label(fp.SYSTEM_LABELS["P_monomer"]),
    "M_monomer": _upright_label(fp.SYSTEM_LABELS["M_monomer"]),
}

# Small square panel, thick spines/ticks to match your example
FIGSIZE_IN = (1.5, 1.5)  # inches
PANEL_FIGSIZE_IN = (3.15, 1.8)  # two mini-panels side-by-side
DPI = fp.FIG_DPI
AX_POS = [0.30, 0.25, 0.60, 0.65]  # left, bottom, width, height
AX_POS_PANEL = (
    [0.16, 0.21, 0.32, 0.58],  # left panel
    [0.60, 0.21, 0.32, 0.58],  # right panel
)
SUBPLOT_ADJUST = dict(left=0.30, right=0.92, bottom=0.30, top=0.95)
SPINE_W = fp.AXES_LINEWIDTH
TICK_W = fp.TICK_WIDTH
LBL_FZ = fp.LABEL_FONTSIZE
TITLE_FZ = fp.LABEL_FONTSIZE
LINE_W = 2.0

ALPHA = 1
# ---- Unit conversion ----
KJ_PER_KCAL = 4.184
KJ_MAX = 10 * KJ_PER_KCAL  # axis top in kJ/mol (from previous 0-10 kcal/mol)

# ------------------- Display mode -------------------- #

# {"scatter", "hex", "bivariate"}
DENSITY_MODE = "scatter"
# Bivariate intensity tuning (more prominent color)
BIVAR_INTENSITY_GAMMA = 0.7   # <1 brightens mid densities
BIVAR_INTENSITY_FLOOR = 0.25  # minimum color strength

# Hexbin tuning for a 1.5" panel -- bigger, more visible hexes
HEX_GRIDSIZE = 18      # -> larger hexagons
HEX_MINCNT   = 1       # show sparse hexes too
HEX_BINS     = "log"   # for separate hex mode

# Fallback scatter settings (if DENSITY_MODE == 'scatter')
SCATTER_S = 20
SCATTER_ALPHA = 0.3
SCATTER_EDGE_W = 0.0
SCATTER_MARKERS = {"P_monomer": "o", "M_monomer": "^"}
SCATTER_Z = 1

# --------------------- Helpers ----------------------- #

def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _read_energy_compare(path: Path) -> List[Tuple[float, float]]:
    """Read (QM, MM) pairs from EnergyCompare.txt.

    Header is expected to start with '#', e.g.:
    #    QMEnergy      MMEnergy  Delta(MM-QM)        Weight
    """
    pairs: List[Tuple[float, float]] = []
    if not path.is_file():
        return pairs
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        qm = _safe_float(parts[0])
        mm = _safe_float(parts[1])
        if math.isfinite(qm) and math.isfinite(mm):
            pairs.append((qm, mm))
    return pairs


def _iter_dirs_for_before_after(torsion_dir: Path):
    """Return (before_dir, after_dir) under a torsion-* directory."""
    if not torsion_dir.is_dir():
        return (None, None)
    iters = sorted([p for p in torsion_dir.iterdir() if p.is_dir() and p.name.startswith('iter_')])
    before = next((p for p in iters if p.name == 'iter_0000'), None)
    after = iters[-1] if iters else None
    if after is None:
        after = before
    return before, after


def _collect_pairs(which: str) -> Dict[str, List[Tuple[float, float]]]:
    """Collect (QM, MM) pairs for 'before' or 'after' across both monomers."""
    assert which in {"before", "after"}
    out: Dict[str, List[Tuple[float, float]]] = {"P_monomer": [], "M_monomer": []}

    for sub in ("N_monomer", "M_monomer"):
        monomer_dir = _DATA_ROOT / sub
        if not monomer_dir.is_dir():
            print(f"[warn] missing {monomer_dir}")
            continue
        monomer_key = "P_monomer" if sub == "N_monomer" else "M_monomer"

        # find stage_0/optimize.tmp
        stage_root = None
        for p in monomer_dir.rglob("stage_0"):
            opt_tmp = p / "optimize.tmp"
            if opt_tmp.is_dir():
                stage_root = opt_tmp
                break
        if stage_root is None:
            print(f"[warn] no stage_0/optimize.tmp under {monomer_dir}")
            continue

        torsions = sorted([p for p in stage_root.iterdir() if p.is_dir() and p.name.startswith('torsion-')])
        if not torsions:
            print(f"[warn] no torsion-* under {stage_root}")
            continue

        for tdir in torsions:
            bdir, adir = _iter_dirs_for_before_after(tdir)
            use = bdir if which == 'before' else adir
            if use is None:
                continue
            ec = use / "EnergyCompare.txt"
            out[monomer_key].extend(_read_energy_compare(ec))

    return out


def _apply_axes_style(ax, pos=None, adjust=True):
    """Make axes look like the example script."""
    if pos is None:
        pos = AX_POS
    ax.set_position(pos)
    if adjust:
        ax.figure.subplots_adjust(**SUBPLOT_ADJUST)
    ax.tick_params(direction='in', top=True, right=True, which='both', width=TICK_W, labelsize=fp.TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_W)
        spine.set_zorder(10)
    # Raise tick lines above scatter as well
    for tl in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tl.set_zorder(11)


# ---------- Colormaps for separate-hex mode (per monomer) ---------- #

def _hex_cmap(hex_color: str) -> LinearSegmentedColormap:
    """Rich transparent->saturated colormap for density (easier to see).

    Low density: faint, slightly transparent tint of the color.
    Mid density: more saturated.
    High density: fully saturated with full opacity.
    """
    r, g, b = to_rgb(hex_color)
    light = (0.7 + 0.3*r, 0.7 + 0.3*g, 0.7 + 0.3*b, 0.20)
    mid   = (0.35 + 0.65*r, 0.35 + 0.65*g, 0.35 + 0.65*b, 0.70)
    high  = (r, g, b, 1.00)
    return LinearSegmentedColormap.from_list(
        f"hex_{hex_color}", [light, mid, high], N=256
    )


# ----------------------- Plotting core ----------------------- #

def _parity_plot_on_ax(ax, pairs_by_monomer: Dict[str, List[Tuple[float, float]]], title: str,
                       show_ylabel: bool, pos=None, adjust: bool = False):
    # Flatten for limits
    all_qm = [qm for pairs in pairs_by_monomer.values() for (qm, _mm) in pairs]
    all_mm = [mm for pairs in pairs_by_monomer.values() for (_qm, mm) in pairs]
    if not all_qm or not all_mm:
        print(f"[warn] no data for {title}")
        return {"mode": DENSITY_MODE}

    _apply_axes_style(ax, pos=pos, adjust=adjust)
    ax.set_title(title, fontsize=TITLE_FZ, pad=8)

    # Parity line
    ax.plot([0, KJ_MAX], [0, KJ_MAX], linestyle='--', linewidth=LINE_W, color='0.5', alpha=ALPHA, zorder=0)

    meta: Dict[str, object] = {"mode": DENSITY_MODE}

    if DENSITY_MODE == "hex":
        extent = (0, KJ_MAX, 0, KJ_MAX)
        hb_map: Dict[str, object] = {}
        for key in ("P_monomer", "M_monomer"):
            pairs = pairs_by_monomer.get(key, [])
            if not pairs:
                continue
            xs = np.array([qm * KJ_PER_KCAL for (qm, _mm) in pairs])
            ys = np.array([mm * KJ_PER_KCAL for (_qm, mm) in pairs])
            hb = ax.hexbin(
                xs, ys,
                gridsize=HEX_GRIDSIZE,
                extent=extent,
                mincnt=HEX_MINCNT,
                bins=HEX_BINS,
                linewidths=0.0,
                cmap=_hex_cmap(COLORS[key]),
                alpha=1.0,
                zorder=2,
            )
            hb_map[key] = hb
        meta.update(hb_map)

    elif DENSITY_MODE == "bivariate":
        # Build per-monomer hexbins *invisibly* to get matched counts by hex center
        extent = (0, KJ_MAX, 0, KJ_MAX)
        pairsP = pairs_by_monomer.get("P_monomer", [])
        pairsM = pairs_by_monomer.get("M_monomer", [])
        xsP = np.array([qm * KJ_PER_KCAL for (qm, _mm) in pairsP])
        ysP = np.array([mm * KJ_PER_KCAL for (_qm, mm) in pairsP])
        xsM = np.array([qm * KJ_PER_KCAL for (qm, _mm) in pairsM])
        ysM = np.array([mm * KJ_PER_KCAL for (_qm, mm) in pairsM])

        hbP = ax.hexbin(xsP, ysP, gridsize=HEX_GRIDSIZE, extent=extent, mincnt=HEX_MINCNT,
                         bins=None, linewidths=0.0)
        hbP.set_visible(False)
        hbM = ax.hexbin(xsM, ysM, gridsize=HEX_GRIDSIZE, extent=extent, mincnt=HEX_MINCNT,
                         bins=None, linewidths=0.0)
        hbM.set_visible(False)

        def _counts_dict(hb):
            offs = hb.get_offsets()
            arr = hb.get_array()
            return {(float(o[0]), float(o[1])): float(c) for o, c in zip(offs, arr)}

        cP = _counts_dict(hbP)
        cM = _counts_dict(hbM)

        # Union of all occupied hex centers via concatenated data
        xs_all = np.concatenate([xsP, xsM])
        ys_all = np.concatenate([ysP, ysM])
        hbU = ax.hexbin(xs_all, ys_all, gridsize=HEX_GRIDSIZE, extent=extent, mincnt=HEX_MINCNT,
                        bins=None, linewidths=0.0)

        offsets = hbU.get_offsets()
        n = len(offsets)
        face = np.zeros((n, 4), dtype=float)

        colorP = np.array(to_rgb(COLORS['P_monomer']))
        colorM = np.array(to_rgb(COLORS['M_monomer']))
        white  = np.array([1.0, 1.0, 1.0])

        totals = []
        ratios = []
        for i, (cx, cy) in enumerate(offsets):
            p = cP.get((float(cx), float(cy)), 0.0)
            m = cM.get((float(cx), float(cy)), 0.0)
            t = p + m
            totals.append(t)
            f = (p / t) if t > 0 else 0.5  # P fraction; neutral if empty (shouldn't happen with mincnt=1)
            ratios.append(f)

        # Normalize total density (log) to [0,1]
        pos = [t for t in totals if t > 0]
        if len(pos) > 0:
            tmin, tmax = min(pos), max(pos)
        else:
            tmin = tmax = 1.0
        if tmax == tmin:
            lin = np.ones_like(totals, dtype=float)
        else:
            logmin = math.log(tmin)
            logmax = math.log(tmax)
            lin = np.array([(math.log(t) - logmin) / (logmax - logmin) if t > 0 else 0.0
                            for t in totals], dtype=float)

        # Boost color prominence: gamma + floor so even low density shows color
        lin = BIVAR_INTENSITY_FLOOR + (1.0 - BIVAR_INTENSITY_FLOOR) * (np.clip(lin, 0.0, 1.0) ** BIVAR_INTENSITY_GAMMA)

        # Compose final colors: mix hue by ratio, brighten by total density
        for i, f in enumerate(ratios):
            base = f * colorP + (1.0 - f) * colorM
            inten = float(lin[i])
            rgb = (1.0 - inten) * white + inten * base
            face[i, :3] = rgb
            face[i, 3] = 1.0

        hbU.set_facecolors(face)
        hbU.set_array(None)  # use facecolors we computed, not the counts

        meta.update({
            "bivar_tmin": tmin,
            "bivar_tmax": tmax,
        })

    else:  # scatter
        for key, pairs in pairs_by_monomer.items():
            if not pairs:
                continue
            xs = [qm * KJ_PER_KCAL for (qm, _mm) in pairs]
            ys = [mm * KJ_PER_KCAL for (_qm, mm) in pairs]
            ax.scatter(
                xs, ys,
                s=SCATTER_S, alpha=SCATTER_ALPHA, marker=SCATTER_MARKERS.get(key, "o"),
                label=LABELS[key],
                c=COLORS[key], edgecolors='none', linewidths=SCATTER_EDGE_W, zorder=SCATTER_Z,
            )

    # ----- R^2 annotation (parity via Pearson correlation) -----
    r2 = _r2_corrcoef(all_qm, all_mm)
    if math.isfinite(r2):
        ax.text(0.05, 0.93, rf"$R^2 = {r2:.3f}$", transform=ax.transAxes,
                ha='left', va='top', fontsize=LBL_FZ, color='0.15')

    # Axis limits/labels
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_xticks([0, 20, 40])
    ax.set_yticks([0, 20, 40])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$E_{\mathrm{DFT}}$ ($\mathrm{kJ/mol}$)', fontsize=LBL_FZ)
    if show_ylabel:
        ax.set_ylabel(r'$E_{\mathrm{Sage}}$ ($\mathrm{kJ/mol}$)', fontsize=LBL_FZ)
    else:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)

    return meta


def _parity_plot(pairs_by_monomer: Dict[str, List[Tuple[float, float]]], title: str, out_path: Path):
    """Single parity plot (legacy helper)."""
    fig = plt.figure(figsize=FIGSIZE_IN, dpi=DPI)
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    meta = _parity_plot_on_ax(ax, pairs_by_monomer, title, show_ylabel=True, pos=AX_POS, adjust=True)
    fig.savefig(out_path, format='svg')
    plt.close(fig)
    print(f"Saved {out_path}")
    return meta


def _parity_panel(before_pairs: Dict[str, List[Tuple[float, float]]],
                  after_pairs: Dict[str, List[Tuple[float, float]]],
                  out_path: Path):
    """Before/after panel with shared y-axis and independent x-axes."""
    fig = plt.figure(figsize=PANEL_FIGSIZE_IN, dpi=DPI)
    ax_before = fig.add_axes(AX_POS_PANEL[0])
    ax_after = fig.add_axes(AX_POS_PANEL[1], sharey=ax_before)

    meta_before = _parity_plot_on_ax(
        ax_before, before_pairs, "Sage", show_ylabel=True, pos=AX_POS_PANEL[0], adjust=False
    )
    meta_after = _parity_plot_on_ax(
        ax_after, after_pairs, "Opt. Sage", show_ylabel=False, pos=AX_POS_PANEL[1], adjust=False
    )
    ax_after.tick_params(labelleft=False)

    if DENSITY_MODE in {"scatter", "hex"}:
        ax_after.legend(
            handles=_legend_handles(),
            loc="lower right",
            frameon=False,
            fontsize=fp.LEGEND_FONTSIZE,
            handlelength=1.2,
            borderpad=0.2,
            labelspacing=0.3,
        )

    fig.savefig(out_path, format='svg')
    plt.close(fig)
    print(f"Saved {out_path}")
    return meta_before, meta_after


# --------------------- Legend & Bars --------------------- #

def _legend_handles() -> List[Line2D]:
    ms = math.sqrt(SCATTER_S)  # scale legend marker to scatter size
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=COLORS["P_monomer"],
            markeredgecolor="none",
            markeredgewidth=SCATTER_EDGE_W,
            markersize=ms,
            label=LABELS["P_monomer"],
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markerfacecolor=COLORS["M_monomer"],
            markeredgecolor="none",
            markeredgewidth=SCATTER_EDGE_W,
            markersize=ms,
            label=LABELS["M_monomer"],
        ),
    ]

def _save_standalone_legend(out_path: Path) -> None:
    """Legend-only SVG (hexagon markers). Suitable for scatter/hex modes."""
    fig = plt.figure(figsize=(1.8, 0.45), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.axis('off')

    ax.legend(handles=_legend_handles(), frameon=False, fontsize=fp.LEGEND_FONTSIZE,
              handlelength=1.5, borderpad=0.2, labelspacing=0.3, ncol=2)

    fig.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_density_bars(hb_map: Dict[str, object], out_path: Path) -> None:
    """Two vertical colorbars (one per monomer). Use in separate-hex mode."""
    hb_map = {k: v for k, v in hb_map.items() if v is not None}
    if not hb_map:
        print("[warn] No hexbins present to build colorbars")
        return

    fig = plt.figure(figsize=(1.6, 1.0), dpi=DPI)
    gs = GridSpec(1, 2, figure=fig, wspace=0.8)

    # P colorbar
    axL = fig.add_subplot(gs[0, 0]); axL.axis('off')
    caxL = axL.inset_axes([0.35, 0.05, 0.25, 0.90])
    if 'P_monomer' in hb_map:
        smL = ScalarMappable(norm=hb_map['P_monomer'].norm, cmap=hb_map['P_monomer'].cmap)
        smL.set_array([])
        cbL = fig.colorbar(smL, cax=caxL, orientation='vertical')
        cbL.outline.set_linewidth(SPINE_W)
        cbL.set_label(LABELS['P_monomer'] + ' (log density)', fontsize=LBL_FZ)
        cbL.ax.tick_params(labelsize=LBL_FZ - 1, width=TICK_W, length=3)

    # M colorbar
    axR = fig.add_subplot(gs[0, 1]); axR.axis('off')
    caxR = axR.inset_axes([0.35, 0.05, 0.25, 0.90])
    if 'M_monomer' in hb_map:
        smR = ScalarMappable(norm=hb_map['M_monomer'].norm, cmap=hb_map['M_monomer'].cmap)
        smR.set_array([])
        cbR = fig.colorbar(smR, cax=caxR, orientation='vertical')
        cbR.outline.set_linewidth(SPINE_W)
        cbR.set_label(LABELS['M_monomer'] + ' (log density)', fontsize=LBL_FZ)
        cbR.ax.tick_params(labelsize=LBL_FZ - 1, width=TICK_W, length=3)

    fig.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_bivariate_bars(meta: Dict[str, object], out_ratio: Path, out_density: Path) -> None:
    """Save two helper bars for bivariate mode.

    - Ratio bar (horizontal): hue varies from M->P (0..1). Density not encoded.
    - Density bar (vertical): brightness varies from white->black (log density).
    """
    colorP = np.array(to_rgb(COLORS['P_monomer']))
    colorM = np.array(to_rgb(COLORS['M_monomer']))

    # -------- Ratio bar (horizontal): M -> P --------
    w, h = 240, 24
    f = np.linspace(0, 1, w)
    grad = np.zeros((h, w, 3), dtype=float)
    for i, ff in enumerate(f):
        grad[:, i, :] = ff * colorP + (1.0 - ff) * colorM

    fig = plt.figure(figsize=(2.4, 0.6), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.imshow(grad, aspect='auto', origin='lower', extent=[0, 1, 0, 1])
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels([r"$\mathrm{M}_1$", "mix", r"$\mathrm{P}_1$"], fontsize=LBL_FZ)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_W)
    ax.tick_params(width=TICK_W, length=3)
    ax.set_xlabel("P<->M mix (fraction $P_1$)", fontsize=LBL_FZ)
    fig.savefig(out_ratio, format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_ratio}")

    # -------- Density bar (vertical): white -> black (log density) --------
    h2, w2 = 160, 32
    # vertical gradient (0 at top -> 1 at bottom): show 0..1 bottom-up
    v = np.linspace(1.0, 0.0, h2)[:, None, None]
    grad2 = np.ones((h2, w2, 3), dtype=float) * v

    fig2 = plt.figure(figsize=(0.6, 1.6), dpi=DPI)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(grad2, aspect='auto', origin='lower', extent=[0, 1, 0, 1])
    ax2.set_xticks([])

    yticks = [0.0, 0.5, 1.0]
    if isinstance(meta.get('bivar_tmin'), (int, float)) and isinstance(meta.get('bivar_tmax'), (int, float)) \
       and meta['bivar_tmin'] > 0 and meta['bivar_tmax'] >= meta['bivar_tmin']:
        tmin = float(meta['bivar_tmin']); tmax = float(meta['bivar_tmax'])
        tmid = math.sqrt(tmin * tmax)
        ylabels = [f"{int(round(tmin))}", f"{tmid:.1f}", f"{int(round(tmax))}"]
    else:
        ylabels = ["low", "", "high"]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels, fontsize=LBL_FZ)

    for spine in ax2.spines.values():
        spine.set_linewidth(SPINE_W)
    ax2.tick_params(width=TICK_W, length=3)
    ax2.set_ylabel("log total density", fontsize=LBL_FZ)
    fig2.savefig(out_density, format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig2)
    print(f"Saved {out_density}")


# --------------------------- Main --------------------------- #

def main() -> None:
    fp.use_mpl_defaults()
    before = _collect_pairs('before')
    after = _collect_pairs('after')

    meta_before, meta_after = _parity_panel(before, after, OUTPUT_DIR / 'parity_panel.svg')
    _save_standalone_legend(OUTPUT_DIR / 'parity_legend.svg')

    # Legends/bars depending on mode
    if DENSITY_MODE == 'hex':
        # separate per-monomer bars
        hb_map = {k: v for k, v in meta_after.items() if k in ('P_monomer','M_monomer')}
        if not any(hb_map.values()):
            hb_map = {k: v for k, v in meta_before.items() if k in ('P_monomer','M_monomer')}
        _save_density_bars(hb_map, OUTPUT_DIR / 'parity_density_bars.svg')
    elif DENSITY_MODE == 'bivariate':
        _save_bivariate_bars(meta_after, OUTPUT_DIR / 'parity_bivar_ratio_bar.svg', OUTPUT_DIR / 'parity_bivar_density_bar.svg')
    else:  # scatter
        pass


if __name__ == '__main__':
    main()

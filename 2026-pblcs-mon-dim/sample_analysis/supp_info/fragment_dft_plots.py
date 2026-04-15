#!/usr/bin/env python3
"""
Generate high-resolution PNGs of torsion energy scans with an inset structure
highlighting the dihedral (P1/N1, M1, and P2). P1 corresponds to N_monomer.
"""

# ---------------------------------------------------------------------------
# DATA PATHS -- adjust these to point to your local data locations
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import io
import json
import os
import sys

DATA_ROOT = os.environ.get("QM_DATA_ROOT", "/scratch/gpfs/WEBB/jv6139/openff_QM_optimization")
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

BASE_DIR = Path(__file__).resolve().parent
FIG_PARAM_DIR = Path(__file__).resolve().parent.parent / "figure_params"
if str(FIG_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_PARAM_DIR))
import figure_params as fp  # type: ignore  # noqa: E402

_DATA_ROOT = Path(DATA_ROOT)

SYSTEM_CONFIG = {
    "P1": {
        "system_dir": "N_monomer",
    },
    "M1": {
        "system_dir": "M_monomer",
    },
    "P2": {
        "system_dir": "N2_dimer_visualize",
    },
}

FIG_SIZE_IN = (3.0, 1.8)
DPI = fp.FIG_DPI
HIGHLIGHT_COLOR = (1.0, 0.5, 0.5)
BOND_LINE_WIDTH = 4
HIGHLIGHT_BOND_WIDTH_MULT = 6
STRUCTURE_FONT_SIZE_PT = 4
STRUCTURE_FONT_SIZE = int(round(STRUCTURE_FONT_SIZE_PT * DPI / 72))
STRUCTURE_PADDING = 0.0
ANGLE_TICKS = [-180, -90, 0, 90, 180]
COLORS = {"initial": "#1f77b4", "final": "#ff7f0e"}
KJ_PER_KCAL = 4.184
# Allocate more space to the plot so traces are not squished, while keeping the figure size fixed.
PLOT_BBOX = (0.14, 0.2, 0.39, 0.65)
STRUCTURE_BBOX = (0.55, 0.2, 0.42, 0.65)
STRUCTURE_PIXELS = (
    int(FIG_SIZE_IN[0] * DPI * STRUCTURE_BBOX[2]),
    int(FIG_SIZE_IN[1] * DPI * STRUCTURE_BBOX[3]),
)
STRUCTURE_TARGET_FILL = 0.99
STRUCTURE_ZOOM = 2.0
STRUCTURE_BBOX_THRESHOLD = 250
STRUCTURE_CROP_PAD = 0


def _pick_bespoke_run(system_dir: Path) -> Path:
    candidates = []
    bespoke_root = system_dir / "bespoke"
    if bespoke_root.is_dir():
        for child in sorted(bespoke_root.iterdir()):
            target_dir = child / "stage_0" / "targets"
            optimize_dir = child / "stage_0" / "optimize.tmp"
            if target_dir.is_dir() and optimize_dir.is_dir():
                candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No targets found under {system_dir}")

    def torsion_count(path: Path) -> int:
        return len(list((path / "stage_0" / "targets").glob("torsion-*")))

    candidates.sort(key=lambda p: (torsion_count(p), p.name))
    return candidates[-1]


def _read_dihedral_from_metadata(path: Path) -> Optional[Tuple[int, int, int, int]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    dihedrals = data.get("dihedrals", [])
    if not dihedrals:
        return None
    d = dihedrals[0]
    if isinstance(d, list) and len(d) == 4:
        return tuple(int(x) for x in d)
    return None


def _read_dihedral_from_aggregated(path: Path) -> Optional[Tuple[int, int, int, int]]:
    if not path.is_file():
        return None
    with path.open() as handle:
        for line in handle:
            if line.startswith("# dihedral_0based="):
                values = line.split("=", 1)[1].strip().split(",")
                if len(values) == 4:
                    try:
                        return tuple(int(v) for v in values)
                    except ValueError:
                        return None
            if not line.startswith("#"):
                break
    return None


def _dihedral_bonds(mol: Chem.Mol, quad: Tuple[int, int, int, int]) -> List[int]:
    a, i, j, b = quad
    bonds = []
    for x, y in ((a, i), (i, j), (j, b)):
        bond = mol.GetBondBetweenAtoms(int(x), int(y))
        if bond is not None:
            bonds.append(bond.GetIdx())
    return bonds


def _load_mol(sdf_path: Path) -> Chem.Mol:
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to load molecule from {sdf_path}")
    return mol


def _prepare_draw_mol(
    mol: Chem.Mol,
    dihedral: Tuple[int, int, int, int],
) -> Tuple[Chem.Mol, Tuple[int, int, int, int]]:
    heavy_map = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        heavy_map[atom.GetIdx()] = len(heavy_map)

    stripped = Chem.RemoveHs(mol, updateExplicitCount=False, sanitize=True)

    def _map_index(idx: int) -> int:
        if idx in heavy_map:
            return heavy_map[idx]
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 1:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() != 1:
                    return heavy_map[neighbor.GetIdx()]
        raise KeyError(idx)

    mapped_dihedral = tuple(_map_index(idx) for idx in dihedral)

    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(stripped)
    rdMolDraw2D.PrepareMolForDrawing(stripped, kekulize=True, addChiralHs=False)
    return stripped, mapped_dihedral


def _render_structure_image(mol: Chem.Mol, dihedral: Tuple[int, int, int, int]) -> Image.Image:
    draw_mol, draw_dihedral = _prepare_draw_mol(mol, dihedral)
    base_img = _draw_structure_image(draw_mol, draw_dihedral, highlight=False)
    base_bbox = _structure_bbox(base_img)
    highlight_img = _draw_structure_image(draw_mol, draw_dihedral, highlight=True)
    return _crop_and_scale_structure(highlight_img, base_bbox)


def _draw_structure_image(
    draw_mol: Chem.Mol,
    draw_dihedral: Tuple[int, int, int, int],
    highlight: bool,
) -> Image.Image:
    drawer = rdMolDraw2D.MolDraw2DCairo(STRUCTURE_PIXELS[0], STRUCTURE_PIXELS[1])
    draw_options = drawer.drawOptions()
    draw_options.useBWAtomPalette = True
    draw_options.explicitMethyl = False
    draw_options.continuousHighlight = True
    draw_options.bondLineWidth = BOND_LINE_WIDTH
    draw_options.highlightBondWidthMultiplier = HIGHLIGHT_BOND_WIDTH_MULT
    draw_options.scaleBondWidth = True
    draw_options.scaleHighlightBondWidth = False
    draw_options.fixedFontSize = STRUCTURE_FONT_SIZE
    draw_options.padding = STRUCTURE_PADDING

    if highlight:
        atom_colors = {int(a): HIGHLIGHT_COLOR for a in draw_dihedral}
        bond_ids = _dihedral_bonds(draw_mol, draw_dihedral)
        bond_colors = {int(bid): HIGHLIGHT_COLOR for bid in bond_ids}
        drawer.DrawMolecule(
            draw_mol,
            highlightAtoms=list(draw_dihedral),
            highlightBonds=bond_ids,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
    else:
        drawer.DrawMolecule(draw_mol)
    drawer.FinishDrawing()

    png_bytes = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_bytes))


def _structure_bbox(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    rgba = image.convert("RGBA")
    data = np.array(rgba)
    rgb = data[:, :, :3]
    alpha = data[:, :, 3]
    mask = (alpha > 0) & np.any(rgb < STRUCTURE_BBOX_THRESHOLD, axis=2)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    min_x = max(min_x - STRUCTURE_CROP_PAD, 0)
    min_y = max(min_y - STRUCTURE_CROP_PAD, 0)
    max_x = min(max_x + STRUCTURE_CROP_PAD, rgba.width - 1)
    max_y = min(max_y + STRUCTURE_CROP_PAD, rgba.height - 1)
    return (min_x, min_y, max_x + 1, max_y + 1)


def _crop_and_scale_structure(
    image: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    target_w, target_h = STRUCTURE_PIXELS
    rgba = image.convert("RGBA")
    crop_box = bbox or _structure_bbox(rgba)
    if crop_box is None:
        return rgba
    cropped = rgba.crop(crop_box)

    avail_w = int(round(target_w * STRUCTURE_TARGET_FILL))
    avail_h = int(round(target_h * STRUCTURE_TARGET_FILL))
    scale = min(avail_w / cropped.width, avail_h / cropped.height) * STRUCTURE_ZOOM
    new_w = max(1, int(round(cropped.width * scale)))
    new_h = max(1, int(round(cropped.height * scale)))
    resized = cropped.resize((new_w, new_h), resample=Image.LANCZOS)

    canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset, resized)
    return canvas


def _iter_monomer_torsions(system: str) -> Iterable[Tuple[int, Path, Path, List[Path]]]:
    config = SYSTEM_CONFIG[system]
    system_dir = _DATA_ROOT / config["system_dir"]
    run_root = _pick_bespoke_run(system_dir)
    targets_root = run_root / "stage_0" / "targets"
    optimize_root = run_root / "stage_0" / "optimize.tmp"

    for tdir in sorted(targets_root.glob("torsion-*")):
        if not tdir.is_dir():
            continue
        try:
            torsion = int(tdir.name.split("-")[-1])
        except ValueError:
            continue
        sdf_path = tdir / "input.sdf"
        meta_paths = [tdir / "metadata.json"]
        torsion_dir = optimize_root / tdir.name
        yield torsion, sdf_path, torsion_dir, meta_paths


def _iter_p2_torsions() -> Iterable[Tuple[int, Path, Path, List[Path]]]:
    system_dir = _DATA_ROOT / SYSTEM_CONFIG["P2"]["system_dir"]
    run_root = _pick_bespoke_run(_DATA_ROOT / "N2_dimer")
    targets_root = run_root / "stage_0" / "targets"
    for tdir in sorted(system_dir.glob("torsion-*")):
        if not tdir.is_dir():
            continue
        try:
            torsion = int(tdir.name.split("-")[-1])
        except ValueError:
            continue
        sdf_path = tdir / "input.sdf"
        meta_paths = [tdir / "aggregated.txt", targets_root / tdir.name / "metadata.json"]
        yield torsion, sdf_path, tdir, meta_paths


def _resolve_dihedral(meta_paths: Iterable[Path]) -> Optional[Tuple[int, int, int, int]]:
    for path in meta_paths:
        if path.name == "aggregated.txt":
            dihedral = _read_dihedral_from_aggregated(path)
        else:
            dihedral = _read_dihedral_from_metadata(path)
        if dihedral is not None:
            return dihedral
    return None


def _read_energycompare(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _load_monomer_energies(torsion_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iter_dirs = sorted([p for p in torsion_dir.iterdir() if p.is_dir() and p.name.startswith("iter_")])
    if not iter_dirs:
        raise FileNotFoundError(f"No iter_* in {torsion_dir}")
    initial_dir = next((p for p in iter_dirs if p.name == "iter_0000"), iter_dirs[0])
    final_dir = iter_dirs[-1]

    initial_data = _read_energycompare(initial_dir / "EnergyCompare.txt")
    final_data = _read_energycompare(final_dir / "EnergyCompare.txt")

    initial_mm = initial_data[:, 1] * KJ_PER_KCAL
    final_mm = final_data[:, 1] * KJ_PER_KCAL
    qm = final_data[:, 0] * KJ_PER_KCAL

    for arr in (initial_mm, final_mm, qm):
        arr -= np.min(arr)

    n_points = min(initial_mm.size, final_mm.size, qm.size)
    if n_points == 0:
        raise ValueError(f"No points for {torsion_dir}")
    initial_mm = initial_mm[:n_points]
    final_mm = final_mm[:n_points]
    qm = qm[:n_points]

    shift = n_points // 2
    initial_mm = np.roll(initial_mm, shift)
    final_mm = np.roll(final_mm, shift)
    qm = np.roll(qm, shift)

    angles = np.linspace(-180, 180, n_points)
    return angles, initial_mm, final_mm, qm


def _load_p2_energies(torsion_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    agg_path = torsion_dir / "aggregated.txt"
    if not agg_path.is_file():
        raise FileNotFoundError(agg_path)
    rows = []
    for line in agg_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("angle_deg"):
            continue
        rows.append(stripped)
    if not rows:
        raise ValueError(f"No data rows in {agg_path}")
    data = np.loadtxt(io.StringIO("\n".join(rows)))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    angles = data[:, 0]
    initial_mm = data[:, 1] * KJ_PER_KCAL
    qm = data[:, 2] * KJ_PER_KCAL
    final_mm = data[:, 3] * KJ_PER_KCAL
    for arr in (initial_mm, final_mm, qm):
        arr -= np.min(arr)
    return angles, initial_mm, final_mm, qm


def _plot_energy_with_inset(
    angles: np.ndarray,
    initial_mm: np.ndarray,
    final_mm: np.ndarray,
    qm: np.ndarray,
    structure_img: Image.Image,
    out_path: Path,
) -> None:
    fp.use_mpl_defaults()
    fig = plt.figure(figsize=FIG_SIZE_IN)
    ax = fig.add_axes(PLOT_BBOX)

    ax.plot(angles, initial_mm, color=COLORS["initial"], linewidth=2.0, alpha=0.85)
    ax.plot(
        angles,
        final_mm,
        color=COLORS["final"],
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
    )
    ax.scatter(
        angles,
        qm,
        color=COLORS["final"],
        alpha=0.85,
        s=25,
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
    )

    max_energy = float(max(np.max(initial_mm), np.max(final_mm), np.max(qm)))
    tick_step = max(1, int(np.ceil(max_energy / 3.0)))
    tick_max = tick_step * 3

    ax.set_xlim(-180, 180)
    ax.set_xticks(ANGLE_TICKS)
    ax.set_ylim(0, tick_max)
    ax.set_yticks(np.arange(0, tick_max + 1, tick_step))
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("E (kJ/mol)")
    fp.style_axis(ax, show_left=True, show_bottom=True)

    structure_ax = fig.add_axes(STRUCTURE_BBOX)
    structure_ax.imshow(structure_img)
    structure_ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def build_all(systems: Iterable[str]) -> None:
    for system in systems:
        if system not in SYSTEM_CONFIG:
            print(f"[skip] unknown system {system}")
            continue

        out_dir = BASE_DIR / system
        if system == "P2":
            torsion_iter = _iter_p2_torsions()
        else:
            torsion_iter = _iter_monomer_torsions(system)

        for torsion, sdf_path, torsion_dir, meta_paths in torsion_iter:
            if not sdf_path.is_file():
                print(f"[warn] missing SDF for {system} torsion-{torsion}: {sdf_path}")
                continue
            dihedral = _resolve_dihedral(meta_paths)
            if dihedral is None:
                print(f"[warn] missing dihedral for {system} torsion-{torsion}")
                continue
            try:
                mol = _load_mol(sdf_path)
            except ValueError as exc:
                print(f"[warn] {exc}")
                continue

            try:
                if system == "P2":
                    angles, initial_mm, final_mm, qm = _load_p2_energies(torsion_dir)
                else:
                    angles, initial_mm, final_mm, qm = _load_monomer_energies(torsion_dir)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[warn] {exc}")
                continue

            structure_img = _render_structure_image(mol, dihedral)
            out_path = out_dir / f"torsion-{torsion}.png"
            _plot_energy_with_inset(angles, initial_mm, final_mm, qm, structure_img, out_path)
            print(f"[ok] {system} torsion-{torsion} -> {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate highlighted torsion fragment PNGs.")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=list(SYSTEM_CONFIG.keys()),
        help="Subset of systems to render (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_all(args.systems)


if __name__ == "__main__":
    main()

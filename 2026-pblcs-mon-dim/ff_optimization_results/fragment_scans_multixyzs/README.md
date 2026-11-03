# DFT torsion-scan geometries and energies (B3LYP-D3BJ/DZVP)

The directory layout mirrors the repository, so these files can be dropped next to the
existing `aggregated.txt` / `input.sdf` of each scan:

    ff_optimization_results/data/{M1,P1,dimers}/torsion-N/scan.xyz

M1 = M1 monomer (11 scans), P1 = P1 monomer (11 scans), dimers = P1 dimer fragments (17 scans); 39 scans total.
Each `input.sdf` in the repository was checked to be byte-identical (past the title line) to the
BespokeFit target it came from, so `torsion-N` here refers to the same scan as in the repository.

## File format

Each `scan.xyz` is a multi-frame XYZ file with 24 frames, one per TorsionDrive grid point, in
grid order -165, -150, ..., 165, 180 degrees (15 degree spacing). Atom order is identical to
`input.sdf`. Coordinates are in angstrom. The comment line of every frame carries:

- `dihedral_atoms_0based` / `dihedral_atoms_1based`: the four atoms of the driven dihedral
- `grid_angle_deg`: the TorsionDrive grid value the dihedral was constrained to
- `measured_dihedral_deg`: the dihedral recomputed from the frame's coordinates (IUPAC sign convention)
- `E_B3LYP-D3BJ_DZVP_hartree`: absolute electronic energy of the constrained-optimized geometry, in hartree
- `E_rel_kcal_per_mol`: energy relative to the lowest frame of that scan (1 hartree = 627.509474 kcal/mol)

## Energy-to-angle mapping

The rows of `aggregated.txt` in the repository are in the same grid order as these frames
(-165 to 180 in 15 degree steps); row i (0-based) is at -165 + 15*i degrees. The `angle_deg` column of
`aggregated.txt` was corrected to this grid on 2026-10-27 (it had been written with
`np.linspace(-180, 180, 24)`, i.e. 15.65 degree spacing). The `B3LYP-D3BJ_DZVP` column of
`aggregated.txt` agrees with `E_rel_kcal_per_mol` here to within 1e-6 kcal/mol row by row, and the
dihedral measured from every frame equals its grid angle to within 0.001 degree. No circular shift is needed.

## Provenance

The scans were produced by OpenFF BespokeFit (`openff-bespoke executor run --workflow default`), run on
Princeton's Stellar cluster in August 2024. The default workflow fragments the molecule (openff-fragmenter),
runs a 1-D TorsionDrive on each rotatable bond of each fragment with geomeTRIC constrained optimizations in
Psi4 at B3LYP-D3BJ/DZVP, and fits the bespoke torsion parameters with ForceBalance. These files are the
TorsionDrive final geometries and energies that ForceBalance used as its fitting targets
(`stage_0/targets/torsion-N/scan.xyz` and `qdata.txt` of each BespokeFit task).

Software versions in the environment used: openff-bespokefit 0.3.0, openff-qcsubmit 0.52.0,
openff-toolkit 0.16.0, openff-fragmenter 0.2.2, torsiondrive 1.1.0, geometric 1.0.2,
qcengine 0.30.0, qcelemental 0.28.0, forcebalance 1.9.6, psi4-1.9.1.

Source on the cluster: `/scratch/gpfs/jv6139/openff_QM_optimization/{M_monomer,N_monomer,N2_dimer}/bespoke/`
(M1 = M_monomer, P1 = N_monomer, dimers = N2_dimer). `make_multixyz.py` in this directory is the
script that wrote the `scan.xyz` files and ran the checks above.

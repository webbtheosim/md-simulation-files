from openff.toolkit import ForceField, Molecule, Topology
from openff.interchange import Interchange;
from openff.units import unit;
from openff.interchange.components._packmol import pack_box;
import numpy as np;
from openff.interchange.components.mdconfig import MDConfig
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper


if __name__ == "__main__":
    L=15
    Nnum = 1
    density = 0.1*unit.gram/(unit.centimeter**3)
    forcefield_file = "/home/jv6139/forcefields/openff_forcefields/P2_dimer.offxml"


    smi="O=C(OC1=CC=C(OCCCCCCCCN2C=C(CCCCOC3=CC=C(C(OC4=CC=C(OCCCCCCCCCl)C=C4)=O)C=C3)N=N2)C=C1)C5=CC=C(OCCCCC#C)C=C5"
    offmol = Molecule.from_smiles(smi)
    offmol.assign_partial_charges("espaloma-am1bcc",toolkit_registry=EspalomaChargeToolkitWrapper())
    #generate conformer to get initial conformation
    offmol.generate_conformers(n_conformers=1)


    #define force field 
    sage = ForceField(forcefield_file)

    #create topology object
    topology = offmol.to_topology()
    topology.box_vectors = unit.Quantity([L, L, L], unit.nanometer)

    system = Interchange.from_smirnoff(force_field=sage, topology=topology)

    system.to_gromacs("mol")


"""
pdbreporter.py: Outputs simulation trajectories in PDB format

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import absolute_import
__author__ = "Peter Eastman"
__version__ = "1.0"

import openmm as mm
import platform
import sys
cluster_path = '/scratch/gpfs/WEBB/bu9134/' if 'tiger' in platform.node() else '/projects/WEBB/eser/'
sys.path.append(cluster_path+'simulation_files/OPENMM')
from openmm_imports import *
from app import PDBFile, PDBxFile, LAMMPSTRJFile

class PDBReporter(object):
    """PDBReporter outputs a series of frames from a Simulation to a PDB file.

    To use it, create a PDBReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file, reportInterval, enforcePeriodicBox=None, atomSubset=None):
        """Create a PDBReporter.

        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        atomSubset: list
            Atom indices (zero indexed) of the particles to output. if None (the default), all particles will be output.
        """
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self._out = open(file, 'w')
        self._topology = None
        self._nextModel = 0
        self._atomSubset = atomSubset
        self._subsetTopology = None


    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        dict
            A dictionary describing the required information for the next report
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return {'steps':steps, 'periodic':self._enforcePeriodicBox, 'include':['positions']}

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if self._atomSubset is not None:
            if self._subsetTopology is None:
                self._createSubsetTopology(simulation.topology)

            topology = self._subsetTopology

            #PDBFile will convert to angstroms so do it here first instead
            positions = state.getPositions(asNumpy=True).value_in_unit(angstroms)
            positions = [positions[i] for i in self._atomSubset]

        else:
            topology = simulation.topology
            positions = state.getPositions(asNumpy=True)

        if self._nextModel == 0:
            PDBFile.writeHeader(topology, self._out)
            self._topology = topology
            self._nextModel += 1
        PDBFile.writeModel(topology, positions, self._out, self._nextModel)
        self._nextModel += 1
        if hasattr(self._out, 'flush') and callable(self._out.flush):
            self._out.flush()

    def _createSubsetTopology(self, topology):
        """Create a subset of an existing topology.

        Parameters
        ----------
        topology : Topology
            The Topology to create a subset from
        """
        # check atomSubset is valid 
        if len(self._atomSubset) == 0:
            self._out.close()
            raise ValueError('atomSubset cannot be an empty list')
        if not all(a == int(a) for a in self._atomSubset):
            self._out.close()
            raise ValueError('all of the indices in atomSubset must be integers')
        if len(set(self._atomSubset)) != len(self._atomSubset):
            self._out.close()
            raise ValueError('atomSubset must contain unique indices')
        if sorted(self._atomSubset) != self._atomSubset:
            self._out.close()
            raise ValueError('atomSubset must be sorted in ascending order')
        if self._atomSubset[0] < 0:
            self._out.close()
            raise ValueError('The smallest allowed value in atomSubset is zero')
        if self._atomSubset[-1] >= topology.getNumAtoms():
            self._out.close()
            raise ValueError('The maximum allowed value in atomSubset must be less than the total number of particles')
        
        self._subsetTopology = Topology()
        
        # convert to set for fast look up
        atomSubsetSet = set(self._atomSubset)

        # store a map from posIndex to Atom object for when we add the bonds
        indexToAtom = {}

        for chain in topology.chains():
            c = self._subsetTopology.addChain(chain.id)
            for res in chain.residues():
                r = self._subsetTopology.addResidue(res.name, c, res.id, res.insertionCode)
                for atom in res.atoms():
                    if atom.index in atomSubsetSet:
                        indexToAtom[atom.index] = self._subsetTopology.addAtom(atom.name, atom.element, r, atom.id)

        self._subsetTopology.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())

        for bond in topology.bonds():
            if bond[0].index in atomSubsetSet and bond[1].index in atomSubsetSet:
                atom1 = indexToAtom[bond[0].index]
                atom2 = indexToAtom[bond[1].index]
                self._subsetTopology.addBond(atom1, atom2, bond.type, bond.order)
        

    def __del__(self):
        if self._topology is not None:
            PDBFile.writeFooter(self._topology, self._out)
        self._out.close()

class LAMMPSTRJReporter(object):
    """PDBReporter outputs a series of frames from a Simulation to a PDB file.

    To use it, create a PDBReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file, velocityprint, reportInterval, enforcePeriodicBox=None):
        """Create a PDBReporter.

        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        """
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self._out = open(file, 'w')
        self._topology = None
        self._nextModel = 0
        self._velocityprint = velocityprint
        if self._velocityprint != False:
            self._out2 = open(self._velocityprint,'w')

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, self._velocityprint, False, False, self._enforcePeriodicBox)

    def report(self, simulation, state):
        """Generate a report.
       
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        self._topology = simulation.topology
        LAMMPSTRJFile.writeHeader(simulation.topology.getNumAtoms(), simulation.currentStep, state, self._out)
        LAMMPSTRJFile.writeModel(simulation.topology, state.getPositions(), self._out, self._nextModel)
        if self._velocityprint != False:
            LAMMPSTRJFile.writeHeader(simulation.topology.getNumAtoms(), simulation.currentStep, state, self._out2)
            LAMMPSTRJFile.writeVelocities(simulation.topology, state.getVelocities(), self._out2, self._nextModel)
        if hasattr(self._out, 'flush') and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
       # if self._topology is not None:
       #     PDBFile.writeFooter(self._topology, self._out)
        self._out.close()



class RPMDReporter(object):

    def __init__(self, file, velocityprint, reportInterval, enforcePeriodicBox=None):
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        ###self._out = open(file, 'w')
        self._topology = None
        self._nextModel = 0
        self._velocityprint = velocityprint
        if self._velocityprint != False:
           ### self._out2 = open(self._velocityprint,'w')
            self._out4 = open(self._velocityprint,'w')
        self._out3 = open(file,'w')
    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, self._velocityprint, False, False, self._enforcePeriodicBox)

    def report(self, simulation, state):
        self._topology = simulation.topology
        state = simulation.integrator.getState(0, getPositions=True, getVelocities=True, enforcePeriodicBox=True)

        ###PDBFile.writeHeader(simulation.topology.getNumAtoms(), simulation.currentStep, state, self._out)
        LAMMPSTRJFile.writeHeader(simulation.topology.getNumAtoms()*simulation.integrator.getNumCopies(), simulation.currentStep, state, file = self._out3)
        posdict = dict()
        veldict = dict()
        for j in range(simulation.integrator.getNumCopies()):
           state = simulation.integrator.getState(j, getPositions=True, getVelocities=True, enforcePeriodicBox=True)
           posdict[j] = LAMMPSTRJFile.writeRPMDModel(simulation.topology, state.getPositions(), self._out3, self._nextModel)
           if self._velocityprint != False:
               veldict[j] = LAMMPSTRJFile.writeRPMDVelocities(simulation.topology, state.getVelocities(), self._out4, self._nextModel)
        printcounter = 1
        printcounter2 = 1
        for i in range(simulation.topology.getNumAtoms()):
           ### c0list = []
           ### c1list = []
           ### c2list = []
            for j in range(simulation.integrator.getNumCopies()): 
                print('{0: <9}'.format(printcounter)+posdict[j][i][9:],file = self._out3)
                printcounter+=1
           ###     c0list.append(float(posdict[j][i].split()[-3]))
           ###     c1list.append(float(posdict[j][i].split()[-2]))
           ###     c2list.append(float(posdict[j][i].split()[-1]))
           ###     printcounter += 1
           ### print('{0: <9}'.format(printcounter2)+posdict[j][i][9:25]+'{0:>9.5f}{1:11.5f}{2:11.5f}'.format(sum(c0list)/len(c0list),sum(c1list)/len(c1list),sum(c2list)/len(c2list)),file=self._out)
           ### printcounter2 += 1
        if self._velocityprint != False:
            ###PDBFile.writeHeader(simulation.topology.getNumAtoms(), simulation.currentStep, state, self._out2)
            PDBFile.writeHeader(simulation.topology.getNumAtoms()*simulation.integrator.getNumCopies(), simulation.currentStep, state, file = self._out4)
            printcounter = 1
            printcounter2 = 1
            for i in range(simulation.topology.getNumAtoms()):
                for j in range(simulation.integrator.getNumCopies()):
                    print('{0: <9}'.format(printcounter)+veldict[j][i][9:],file = self._out4)
                    printcounter+=1
        if hasattr(self._out3, 'flush') and callable(self._out3.flush):
            self._out3.flush()

    def __del__(self):
      #  if self._topology is not None:
      #      PDBFile.writeFooter(self._topology, self._out)
        self._out3.close()


class PDBxReporter(PDBReporter):
    """PDBxReporter outputs a series of frames from a Simulation to a PDBx/mmCIF file.

    To use it, create a PDBxReporter, then add it to the Simulation's list of reporters.
    """

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if self._nextModel == 0:
            PDBxFile.writeHeader(simulation.topology, self._out)
            self._nextModel += 1
        PDBxFile.writeModel(simulation.topology, state.getPositions(), self._out, self._nextModel)
        self._nextModel += 1
        if hasattr(self._out, 'flush') and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
        self._out.close()





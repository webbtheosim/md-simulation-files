from __future__ import print_function
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmm.app.internal.unitcell import computeLengthsAndAngles
import re
import numpy as np
from scipy.stats import sem
from openmmtools import integrators
import time
import sys
import os
from special_simulation import time_to_step,run_special_steps
sys.setrecursionlimit(10000)

'''
This script runs the equilibrated configuration (often at 300K, but can be changed) for an additional 100 ps, and collects the
trajectory and thermo data at a high frequency of every 20fs. It is used for obtaining dynamic properties.
In total, that equates to 100,000 steps, 5000 trajectory snapshots and thermodynamic data.
'''

##### Parameters
TIMESTEP = 0.001 #picoseconds
THERMO_FREQ = 0.02/TIMESTEP #every 20 fs
TRAJ_FREQ = 0.02/TIMESTEP #every 20 fs

DYNAMICS_T = 300 #run dynamics analysis at this temperature
PRINT_VELOCITIES = False
#####

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";

pdb = PDBFile('init_templated.pdb')
forcefield = ForceField('charmm_chitosan.xml', 'water.xml')

system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=10*angstrom, removeCMMotion=True)
forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
nonbonded_force = forces['NonbondedForce']
nonbonded_force.setUseSwitchingFunction(True)
nonbonded_force.setSwitchingDistance(9*angstrom)

integrator = NoseHooverIntegrator(DYNAMICS_T * kelvin, 1/picosecond,TIMESTEP*picoseconds)
simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)

Restart.load_simulation('{}_equil.save'.format(DYNAMICS_T),simulation,'classical')

'''Adapt from Langevin to Nose Hoover for 0.5 ns'''
simulation.step(steps=time_to_step(500,TIMESTEP))

simulation.reporters.append(StateDataReporter(False,"{}_dynamics.avg".format(DYNAMICS_T), THERMO_FREQ, step=True, time=True, density=True, totalEnergy=True, kineticEnergy=True, volume=True, potentialEnergy=True, temperature=True))

simulation.reporters.append(PDBReporter('{}_dynamics.lammpstrj'.format(DYNAMICS_T),PRINT_VELOCITIES,TRAJ_FREQ))

simulation.step(steps=time_to_step(100,TIMESTEP))

'''Extend simulation below with infrequent reporting'''
simulation.reporters = []

THERMO_FREQ = 50/TIMESTEP #every 0.1 ps
TRAJ_FREQ = 50/TIMESTEP #every 0.1 ps
simulation.reporters.append(StateDataReporter(False,"{}_dynamics_extended.avg".format(DYNAMICS_T), THERMO_FREQ, step=True, time=True, density=True, totalEnergy=True, kineticEnergy=True, volume=True, potentialEnergy=True, temperature=True))

simulation.reporters.append(PDBReporter('{}_dynamics_extended.lammpstrj'.format(DYNAMICS_T),PRINT_VELOCITIES,TRAJ_FREQ))

simulation.step(steps=time_to_step(100000,TIMESTEP))

Restart.save_simulation('{}_dynamics_extended.save'.format(DYNAMICS_T),simulation,'classical')

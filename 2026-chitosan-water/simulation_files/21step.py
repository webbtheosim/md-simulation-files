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

'''
This script first prepares a .pdb file given the evaporated state of the system, 
then heats it up from the evaporation temperature of the solvent to the target temperature to anneal from,
and then carries out the 21-step slow decompression procedure.
'''

##### Input parameters
COMPOSITION = int(os.getcwd().split('/')[-2]) #Composition automatically fetched from directory name

if COMPOSITION < 30:
    TARGET_T = 900
elif COMPOSITION < 100:
    TARGET_T = 700
elif COMPOSITION == 100:
    TARGET_T = 600
if ('SPCE' in str(os.getcwd()) or 'TIP3P' in str(os.getcwd())) and COMPOSITION in [10,15]:
    TARGET_T = 800
if ('SPCE' in str(os.getcwd()) or 'TIP3P' in str(os.getcwd())) and COMPOSITION in [20]:
    TARGET_T = 700

HEAT_T = 1000
EVAPORATION_T = 373

TIMESTEP = 0.001 #picoseconds
Pmax = 200 #bar
THERMO_FREQ = 1/TIMESTEP
NVT_FRQ = 100000000
BAROSTAT_FRQ = 0.025/TIMESTEP
#####

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";

pdb = PDBFile('init_templated.pdb') 
forcefield = ForceField('charmm_chitosan.xml', 'water.xml')

##### Prepare System
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=10*angstrom, removeCMMotion=True)
forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
nonbonded_force = forces['NonbondedForce']
nonbonded_force.setUseSwitchingFunction(True)
nonbonded_force.setSwitchingDistance(9*angstrom)

barostat = MonteCarloBarostat(1*bar,EVAPORATION_T*kelvin,BAROSTAT_FRQ)
system.addForce(barostat)
integrator = LangevinIntegrator(EVAPORATION_T*kelvin,1/picosecond,TIMESTEP*picoseconds)
simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)
simulation.context.setPositions(pdb.positions)

simulation.minimizeEnergy()

THERMO_FILE = open('21step.avg','w')
#####

#Step 1: NVT, High Temperature
barostat.setFrequency(NVT_FRQ)
integrator.setTemperature(HEAT_T*kelvin)
run_special_steps(500,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 2: NVT, Target Temperature
integrator.setTemperature(TARGET_T*kelvin)
run_special_steps(250,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 3: NPT, 0.02*Pmax, Target Temperature
simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.02*Pmax*bar)
simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
integrator.setTemperature(TARGET_T*kelvin)
barostat.setFrequency(BAROSTAT_FRQ)
run_special_steps(500,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 4: NVT High Temperature
barostat.setFrequency(NVT_FRQ)
integrator.setTemperature(HEAT_T*kelvin)
run_special_steps(250,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 5: NVT, Target Temperature
integrator.setTemperature(TARGET_T*kelvin)
run_special_steps(500,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 6: NPT, 0.6*Pmax, Target Temperature
simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.6*Pmax*bar)
simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
integrator.setTemperature(TARGET_T*kelvin)
barostat.setFrequency(BAROSTAT_FRQ)
run_special_steps(250,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 7: NVT High Temperature
barostat.setFrequency(NVT_FRQ)
integrator.setTemperature(HEAT_T*kelvin)
run_special_steps(500,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 8: NVT, Target Temperature
integrator.setTemperature(TARGET_T*kelvin)
run_special_steps(500,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 9: NPT, Pmax, Target Temperature
simulation.context.setParameter(MonteCarloBarostat.Pressure(), Pmax*bar)
simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
integrator.setTemperature(TARGET_T*kelvin)
barostat.setFrequency(BAROSTAT_FRQ)
run_special_steps(500,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 10: NVT High Temperature
barostat.setFrequency(NVT_FRQ)
integrator.setTemperature(HEAT_T*kelvin)
run_special_steps(250,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 11: NVT, Target Temperature
integrator.setTemperature(TARGET_T*kelvin)
run_special_steps(500,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 12: NPT, 0.5*Pmax, Target Temperature
simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.5*Pmax*bar)
simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
integrator.setTemperature(TARGET_T*kelvin)
barostat.setFrequency(BAROSTAT_FRQ)
run_special_steps(50,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 13: NVT High Temperature
barostat.setFrequency(NVT_FRQ)
integrator.setTemperature(HEAT_T*kelvin)
run_special_steps(25,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 14: NVT, Target Temperature
integrator.setTemperature(TARGET_T*kelvin)
run_special_steps(50,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 15: NPT, 0.1*Pmax, Target Temperature
simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.1*Pmax*bar)
simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
integrator.setTemperature(TARGET_T*kelvin)
barostat.setFrequency(BAROSTAT_FRQ)
run_special_steps(25,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 16: NVT High Temperature
barostat.setFrequency(NVT_FRQ)
integrator.setTemperature(HEAT_T*kelvin)
run_special_steps(25,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 17: NVT, Target Temperature
integrator.setTemperature(TARGET_T*kelvin)
run_special_steps(50,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 18: NPT, 0.01*Pmax, Target Temperature
simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.01*Pmax*bar)
simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
integrator.setTemperature(TARGET_T*kelvin)
barostat.setFrequency(BAROSTAT_FRQ)
run_special_steps(25,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 19: NVT High Temperature
barostat.setFrequency(NVT_FRQ)
integrator.setTemperature(HEAT_T*kelvin)
run_special_steps(25,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 20: NVT, Target Temperature
integrator.setTemperature(TARGET_T*kelvin)
run_special_steps(50,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

#Step 21: NPT, 1 bar, Target Temperature
simulation.context.setParameter(MonteCarloBarostat.Pressure(), 1*bar)
simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
integrator.setTemperature(TARGET_T*kelvin)
barostat.setFrequency(BAROSTAT_FRQ)

Restart.save_simulation('20step.save',simulation,'classical')

run_special_steps(10000,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)

Restart.save_simulation('21step.save',simulation,'classical')

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
import subprocess

'''
info here

'''
def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

##### Input parameters
COMPOSITION = int(os.getcwd().split('/')[-2]) #Composition automatically fetched from directory name

pdb_file = 'init_templated.pdb'

if COMPOSITION < 30:
    TARGET_T = 900
elif COMPOSITION < 100:
    TARGET_T = 700
elif COMPOSITION == 100:
    TARGET_T = 600
    pdb_file = 'init_templated.pdb'
if ('SPCE' in str(os.getcwd()) or 'TIP3P' in str(os.getcwd())) and COMPOSITION in [10,15]:
    TARGET_T = 800
if ('SPCE' in str(os.getcwd()) or 'TIP3P' in str(os.getcwd())) and COMPOSITION in [20]:
    TARGET_T = 700

BLOCK_NO = int(sys.argv[2])
JOB_ID = sys.argv[1]

#DIVIDE SIMULATIONS TO 4 BLOCKS
temp_range = np.arange(TARGET_T,199,-1)
N_BLOCKS = 4
split_temp_range = split(temp_range,N_BLOCKS)
temp_range = split_temp_range[BLOCK_NO]

INITIAL_T = TARGET_T
TIMESTEP = 0.001 #picoseconds
THERMO_FREQ = 1/TIMESTEP #every 1 ps
TRAJ_FREQ = 10/TIMESTEP #every 10 ps
BAROSTAT_FRQ = 0.025/TIMESTEP #every 25 fs
#####

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";

pdb = PDBFile(pdb_file) 
forcefield = ForceField('charmm_chitosan.xml', 'water.xml')

##### Prepare System
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=10*angstrom, removeCMMotion=True)
forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
nonbonded_force = forces['NonbondedForce']
nonbonded_force.setUseSwitchingFunction(True)
nonbonded_force.setSwitchingDistance(9*angstrom)

barostat = MonteCarloBarostat(1*bar,INITIAL_T*kelvin,BAROSTAT_FRQ)
system.addForce(barostat)
integrator = LangevinIntegrator(INITIAL_T*kelvin,1/picosecond,TIMESTEP*picoseconds)
simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)

if BLOCK_NO == 0:
    LOAD_FILE = '21step.save'
else:
    LOAD_TEMPERATURE = split_temp_range[BLOCK_NO-1][-1]
    LOAD_FILE = '{}_equil.save'.format(LOAD_TEMPERATURE)

Restart.load_simulation(LOAD_FILE,simulation,'classical')
if BLOCK_NO == 0:
    simulation.minimizeEnergy()

THERMO_FILE = open('BLOCK{}.avg'.format(BLOCK_NO),'w')
#####

##### Every 10K, simulate for 2 ns; else, simulate for 0.1 ns
for temperature in temp_range:
    # Set temperature
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), temperature*kelvin)
    integrator.setTemperature(temperature*kelvin)
    if temperature%10 == 0:
        #Equilibrate for 2 ns
        run_special_steps(1000,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)
        simulation.reporters.append(PDBReporter('{}.lammpstrj'.format(temperature),False,TRAJ_FREQ))
        run_special_steps(1000,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)
        simulation.reporters = []
        Restart.save_simulation('{}_equil.save'.format(temperature),simulation,'classical')
    else:
        #Equilibrate for 0.1 ns
        run_special_steps(100,THERMO_FILE,TIMESTEP,THERMO_FREQ,system,simulation)
Restart.save_simulation('{}_equil.save'.format(temp_range[-1]),simulation,'classical')
#####

if BLOCK_NO != N_BLOCKS-1:
    output = subprocess.getoutput('sacct -B -j {}'.format(JOB_ID))
    output = output.split('\n')[2:]
    output = output[:-2] + ['python /projects/WEBB/eser/simulation_files/chitosan_simulation_files/900_200.py $SLURM_JOB_ID {}'.format(BLOCK_NO+1)]
    output = '\n'.join(output)

    randint = np.random.randint(0,100000)
    next_job_script = open('{}.submit'.format(randint),'w')
    next_job_script.write(output)
    next_job_script.close()
    os.system('sbatch {}.submit'.format(randint))
    os.system('rm {}.submit'.format(randint))

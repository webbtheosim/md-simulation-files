from __future__ import print_function
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmm.app.internal.unitcell import computeLengthsAndAngles
import re
import numpy as np
from scipy.stats import sem
import time
import sys
import os
import copy
import mdtraj as md
from compute_pressure import compute_pressure, compute_masses

def compute_moduli(simulation,P_EVERY, EQUILIBRATION_LENGTH, RUN_LENGTH, dh=0.02):
    context = simulation.context
    system = context.getSystem()
    
    state = context.getState(getPositions=True,getEnergy=True,enforcePeriodicBox=True)
    positions = copy.deepcopy(state.getPositions(asNumpy=True).value_in_unit(nanometer))
    rvecs = copy.deepcopy(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nanometer))

    imitate_triclinic_rvecs = copy.deepcopy(rvecs)
    imitate_triclinic_rvecs[1,0] = 0.000001
    system.setDefaultPeriodicBoxVectors(*imitate_triclinic_rvecs)
    context.reinitialize()

    context.setPeriodicBoxVectors(*rvecs)
    context.setState(state)

    masses = compute_masses(simulation)
    TIMESTEP = simulation.integrator.getStepSize().value_in_unit(picosecond)


    if any('arostat' in i for i in [system.getForce(index).__class__.__name__ for index in range(system.getNumForces())]) == True:
        BAROSTAT_CHECK = True
        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i)
        
        for i, f in enumerate(system.getForces()):
            if 'arostat' in f.getName():
                BAROSTAT_P = f.getDefaultPressure()._value
                BAROSTAT_T = f.getDefaultTemperature()._value
                BAROSTAT_FRQ = f.getFrequency()
                system.removeForce(i)
        context.reinitialize()
        context.setState(state)
    else:
        BAROSTAT_CHECK = False

    P_BATCH_COUNT = int(RUN_LENGTH/TIMESTEP/P_EVERY)

    directions = ['xx','yy','zz','yz','xz','xy']
    direction_numbers = [1,2,3,4,5,6]
    P0 = {}
    for direction_iter in directions:
        P0[direction_iter] = []
    
    simulation.step(EQUILIBRATION_LENGTH/TIMESTEP)
    for sim_batch in range(P_BATCH_COUNT):
        simulation.step(P_EVERY)
        frame_P0 = compute_pressure(simulation,masses)

        for direction in directions:
            P0[direction].append(frame_P0[direction])
            
    for direction_iter in directions:
        P0[direction_iter] = np.mean(P0[direction_iter])
    
    fractional = np.dot(positions, np.linalg.inv(rvecs))

    axes = ['x','y','z']

    indices = [(0,0),(1,1),(2,2),(2,1),(2,0),(1,0)] #reordered to xx,yy,zz,yz,xz,xy

    C,Cpos,Cneg = {},{},{}
    for which_index,(i, j) in enumerate(indices):
        direction = axes[j] + axes[i]
        len0 = copy.deepcopy(rvecs[j,j])
        absolute_dh = len0 * dh

        rvecs_scaled = rvecs.copy()
        rvecs_scaled[i, j] += absolute_dh
        pos_scaled   = np.dot(fractional, rvecs_scaled)
        context.setPeriodicBoxVectors(*rvecs_scaled)
        context.setPositions(pos_scaled)

        P_plus,P_minus = {},{}
        for direction_iter in directions:
            P_plus[direction_iter],P_minus[direction_iter] = [],[]

        simulation.step(EQUILIBRATION_LENGTH/TIMESTEP)
        for sim_batch in range(P_BATCH_COUNT):
            simulation.step(P_EVERY)
            frame_P = compute_pressure(simulation,masses)
            for direction_iter in directions:
                P_plus[direction_iter].append(frame_P[direction_iter])

        for direction_no,direction_iter in enumerate(directions):
            P_plus[direction_iter] = np.mean(P_plus[direction_iter])
            Cpos[direction_no+1] = -(P_plus[direction_iter] - P0[direction_iter]) / (absolute_dh/len0) * 0.0000986923
        
        rvecs_scaled = rvecs.copy()
        rvecs_scaled[i, j] -= absolute_dh
        pos_scaled   = np.dot(fractional, rvecs_scaled)
        context.setPeriodicBoxVectors(*rvecs_scaled)
        context.setPositions(pos_scaled)

        simulation.step(EQUILIBRATION_LENGTH/TIMESTEP)
        for sim_batch in range(P_BATCH_COUNT):
            simulation.step(P_EVERY)
            frame_P = compute_pressure(simulation,masses)
            for direction_iter in directions:
                P_minus[direction_iter].append(frame_P[direction_iter])

        for direction_no,direction_iter in enumerate(directions):
            P_minus[direction_iter] = np.mean(P_minus[direction_iter])
            Cneg[direction_no+1] = -(P_minus[direction_iter] - P0[direction_iter]) / (-absolute_dh/len0) * 0.0000986923

        for direction_number in direction_numbers:
            C[direction_number,which_index+1] = (Cpos[direction_number]+Cneg[direction_number])/2

    C_all = {}
    for direction_1 in direction_numbers:
        for direction_2 in direction_numbers:
            if direction_1 == direction_2:
                C_all[direction_1,direction_2] = C[direction_1,direction_2]
            elif direction_1 > direction_2:
                C_all[direction_2,direction_1] = (C[direction_1,direction_2]+C[direction_2,direction_1])/2
            else:
                continue

    C_cubic = {}
    C_cubic[1,1] = (C_all[1,1] + C_all[2,2] + C_all[3,3])/3
    C_cubic[1,2] = (C_all[1,2] + C_all[1,3] + C_all[2,3])/3
    C_cubic[4,4] = (C_all[4,4] + C_all[5,5] + C_all[6,6])/3

    BULK_MODULUS = (C_cubic[1,1]+2*C_cubic[1,2])/3
    SHEAR_MODULUS = C_cubic[4,4]
    POISSON_RATIO = 1/(1+C_cubic[1,1]/C_cubic[1,2])
    YOUNGS_MODULUS = 2*SHEAR_MODULUS*(1+POISSON_RATIO)

    context.setPeriodicBoxVectors(*rvecs)
    context.setPositions(positions)
    if BAROSTAT_CHECK == True:
        system.addForce(MonteCarloBarostat(BAROSTAT_P*bar,BAROSTAT_T*kelvin,BAROSTAT_FRQ))

    return BULK_MODULUS,SHEAR_MODULUS,POISSON_RATIO,YOUNGS_MODULUS

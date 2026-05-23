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

def compute_masses(simulation):
    context = simulation.context
    system = context.getSystem()
    masses = []
    for i in range(system.getNumParticles()):
        masses.append(system.getParticleMass(i).value_in_unit(kilogram/mole))
    masses = np.array(masses)
    return masses

def compute_pressure(simulation, masses, dh=1e-4):
    tinit = time.time()
    AVOGADRO = 6.02214076e23

    context = simulation.context
    system = context.getSystem()

    state = context.getState(getPositions=True,getVelocities=True,getEnergy=True,enforcePeriodicBox=True)
    positions = copy.deepcopy(state.getPositions(asNumpy=True).value_in_unit(nanometer))
    rvecs = copy.deepcopy(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nanometer))
    volume = copy.deepcopy(state.getPeriodicBoxVolume().value_in_unit(meter**3))

    velocities = copy.deepcopy(state.getVelocities(asNumpy=True).value_in_unit(meter/second))
    vel_dict = {0:velocities[:,0],1:velocities[:,1],2:velocities[:,2]}
   
    fractional = np.dot(positions, np.linalg.inv(rvecs))

    axes = ['x','y','z']
    indices = [
            (0, 0),
            (1, 0), (1, 1),
            (2, 0), (2, 1), (2, 2),]

    P = {}
    for (i, j) in indices:

        P_kinetic = np.sum(vel_dict[i] * vel_dict[j] * masses)/(volume*AVOGADRO)

        absolute_dh = rvecs[j,j] * dh

        rvecs_scaled = rvecs.copy()

        rvecs_scaled[i, j] += absolute_dh
        pos_scaled   = np.dot(fractional, rvecs_scaled)
        context.setPeriodicBoxVectors(*rvecs_scaled)
        context.setPositions(pos_scaled)
        E_plus = context.getState(getEnergy=True,enforcePeriodicBox=True).getPotentialEnergy().value_in_unit(joule/mole)

        rvecs_scaled = rvecs.copy()
        rvecs_scaled[i, j] -= absolute_dh
        pos_scaled   = np.dot(fractional, rvecs_scaled)

        context.setPeriodicBoxVectors(*rvecs_scaled)
        context.setPositions(pos_scaled)
        E_minus = context.getState(getEnergy=True,enforcePeriodicBox=True).getPotentialEnergy().value_in_unit(joule/mole)

        P[axes[j] + axes[i]] = (-((E_plus - E_minus) / (2*dh) / (volume*AVOGADRO)) + P_kinetic) * (10**-5) * (0.986923) #Pa to bar to atm

    context.setPeriodicBoxVectors(*rvecs)
    context.setPositions(positions)
    return P

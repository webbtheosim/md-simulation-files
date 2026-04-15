from openmm import *
from openmm.unit import *
from openmm.app import *
from app import pdbreporter,pdbfile,statedatareporter
from app.pdbreporter import *
from app.pdbfile import *
from app.statedatareporter import *
import app.pdbreporter as pdbreporter

#from app.forcefield import *
#import app.forcefield as forcefield

import numpy as np
import pandas
import time
import sys
from sys import stdout
import re
import argparse
import copy
import pickle
import random
from scipy.spatial import KDTree
import itertools
import h5py
from numba import njit, prange

def time_to_step(time_in_ps,timestep):
    ''' 
    Given the simulation time (in ps), returns the number of simulation steps.
    '''
    step_count = int(time_in_ps/timestep)
    return step_count

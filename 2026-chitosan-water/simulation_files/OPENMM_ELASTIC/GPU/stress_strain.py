import platform
import sys
cluster_path = '/scratch/gpfs/WEBB/bu9134/' if 'tiger' in platform.node() else '/projects/WEBB/eser/'
sys.path.append(cluster_path+'simulation_files/OPENMM')
from openmm_imports import *
sys.path.append('/projects/WEBB/eser/simulation_files/chitosan_simulation_files/OPENMM_ELASTIC/GPU')
from compute_pressure import compute_pressure, compute_masses
from compute_moduli import compute_moduli

def stress_strain(simulation, P_EVERY, EQUILIBRATION_LENGTH, RUN_LENGTH, dh=0.1):
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
    

    dh_list = np.arange(1,1+dh+0.001,0.001)
    scales = [dh_list[index]/dh_list[index-1] for index in range(1,len(dh_list))]

    axes = ['x','y','z']

    indices = [(0,0),(1,1),(2,2),(2,1),(2,0),(1,0)] #reordered to xx,yy,zz,yz,xz,xy
    simulation.reporters.append(LAMMPSTRJReporter('stress_strain.lammpstrj',False,10000))

    P_plus_scale = {}
    for which_index,(i, j) in enumerate(indices):
        print(which_index)
        for direction_no,direction_iter in enumerate(directions):
            P_plus_scale[direction_iter,which_index+1] = [P0[direction_iter]]

        for scale_ix,scale in enumerate(scales):
            context_live = simulation.context
            state_live = context.getState(getPositions=True,getEnergy=True,enforcePeriodicBox=True)
            positions_live = copy.deepcopy(state_live.getPositions(asNumpy=True).value_in_unit(nanometer))
            rvecs_live = copy.deepcopy(state_live.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nanometer))
            fractional = np.dot(positions_live, np.linalg.inv(rvecs_live))

            direction = axes[j] + axes[i]
            len0 = copy.deepcopy(rvecs_live[j,j])
            absolute_dh = len0 * (scale-1)
            if i != j:
                absolute_dh = len0 * (dh_list[2] - dh_list[1])

            rvecs_scaled = rvecs_live.copy()
            rvecs_scaled[i, j] += absolute_dh
            pos_scaled = np.dot(fractional, rvecs_scaled)
            context_live.setPeriodicBoxVectors(*rvecs_scaled)
            context_live.setPositions(pos_scaled)

    #        print(rvecs_scaled, flush=True)

            P_plus = {}
            for direction_iter in directions:
                P_plus[direction_iter] = []

            simulation.step(int(EQUILIBRATION_LENGTH/TIMESTEP))
            for sim_batch in range(P_BATCH_COUNT):
                simulation.step(P_EVERY)
                frame_P = compute_pressure(simulation,masses)
                for direction_iter in directions:
                    P_plus[direction_iter].append(frame_P[direction_iter])

            for direction_no,direction_iter in enumerate(directions):
                P_plus[direction_iter] = np.mean(P_plus[direction_iter])
                P_plus_scale[direction_iter,which_index+1].append(P_plus[direction_iter])

        context.setPeriodicBoxVectors(*rvecs)
        context.setPositions(positions)
    
    f = open('stress_strain.pkl','wb')
    pickle.dump(P_plus_scale, f, pickle.HIGHEST_PROTOCOL)
    return P_plus_scale

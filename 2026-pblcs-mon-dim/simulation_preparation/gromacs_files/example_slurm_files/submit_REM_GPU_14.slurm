#!/bin/bash
#SBATCH --job-name=M_14_GPU_tREM_AA
#SBATCH --gres=gpu:8            # 2 GPUs in total
#SBATCH --ntasks=73              # 73 MPI ranks, i.e. 73 replicas
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=48:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jv6139@princeton.edu
#SBATCH --account=webb

module purge
module load intel-mkl/2024.2
module load intel-oneapi/2024.2 
module load intel/2024.2
module load cudatoolkit/12.6
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.6

export GMX_MAXBACKUP=-1
export GMX_ENABLE_DIRECT_GPU_COMM=1
export GMX_GPU_PME_DECOMPOSITION=1
export GMX_FORCE_GPU_AWARE_MPI=1

# List of temperatures for 73 replicas
TEMPS=(260 262.5 265 267.5 270 272.5 275 277.5 280 282.5 285 287.5 290 292.5 295 297.5 \
       300 302.5 305 307.5 310 312.5 315 317.5 320 322.5 325 327.5 330 332.5 335 337.5 \
       340 342.5 345 347.5 350 352.5 355 357.5 360 362.5 365 367.5 370 372.5 375 377.5 \
       380 382.5 385 387.5 390 392.5 395 397.5 400 402.5 405 407.5 410 412.5 415 417.5 \
       420 422.5 425 427.5 430 432.5 435 437.5 440)

# Directories and files
simdir=/scratch/gpfs/WEBB/jv6139/M_monomer/simulations/tREM_GPU
controldir=/scratch/gpfs/WEBB/jv6139/M_monomer/control-scripts/tREM_GPU
top=/scratch/gpfs/WEBB/jv6139/M_monomer/structures/256M_molecules/1M_monomer.top
simdir_old=/scratch/gpfs/WEBB/jv6139/M_monomer/simulations/tREM_GPU
###########################
# 1) Generate the .tpr files
###########################
for T in "${TEMPS[@]}"; do
    mkdir -p "$simdir/${T}K"
    cd "$simdir/${T}K"
    srun gmx_gpu grompp \
        -f "$controldir/trem_${T}.mdp" \
        -c "$simdir_old/${T}K/trem13_gpu.gro" \
        -p "$top" \
        -o trem14_gpu.tpr \
        -maxwarn 2&

    cd "$simdir"
done
wait
###########################
# 2) Prepare the multidir array
###########################
MULTIDIRS=()
for T in "${TEMPS[@]}"; do
    MULTIDIRS+=("$simdir/${T}K")
done

#build gputasks string using number of GPUs (e.g. for 4 tasks with )

###########################
# 3) Run the multi-replica T-REMD simulation
###########################
srun --gpu-bind=map_gpu:0,1,2,3,4,5,6,7 gmx_gpu mdrun \
    -multidir "${MULTIDIRS[@]}" \
    -deffnm trem14_gpu \
    -replex 500 \
    -update gpu \
    -nb gpu \
    -bonded gpu \
    -pme gpu \
    -pmefft gpu \

echo "Done!"
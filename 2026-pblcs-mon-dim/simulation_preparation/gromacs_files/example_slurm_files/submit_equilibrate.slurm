#!/bin/bash
#SBATCH --job-name=AA_array
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jv6139@princeton.edu
#SBATCH --account=webb
#SBATCH --array 0-72

module purge
module load openmpi/gcc/4.1.6
module load intel-mkl/2024.2


export GMX_MAXBACKUP=-1
#TEMPS=(260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440)
TEMPS=(260 262.5 265 267.5 270 272.5 275 277.5 280 282.5 285 287.5 290 292.5 295 297.5 300 302.5 305 307.5 310 312.5 315 317.5 320 322.5 325 327.5 330 332.5 335 337.5 340 342.5 345 347.5 350 352.5 355 357.5 360 362.5 365 367.5 370 372.5 375 377.5 380 382.5 385 387.5 390 392.5 395 397.5 400 402.5 405 407.5 410 412.5 415 417.5 420 422.5 425 427.5 430 432.5 435 437.5 440)
#TEMPS=(267.5 277.5 287.5 297.5 307.5 317.5 327.5 337.5 347.5 357.5 367.5 377.5 387.5 397.5 407.5 417.5 427.5)
gro=/scratch/gpfs/WEBB/jv6139/M_monomer/simulations/tREM/minimize/minimize.gro
top=/scratch/gpfs/WEBB/jv6139/M_monomer/structures/256M_molecules/1M_monomer.top
control_loc=/scratch/gpfs/WEBB/jv6139/M_monomer/control-scripts/tREM/equilibrate
sim_loc=/scratch/gpfs/WEBB/jv6139/M_monomer/simulations/tREM

T=${TEMPS[$SLURM_ARRAY_TASK_ID]}

#make directory for this temperature if it doesn't exist
mkdir -p $sim_loc/${T}K

cd $sim_loc/${T}K
gmx_cpu grompp -f $control_loc/equilibrate_${T} -c $gro -p $top -o eqbm_${T}K.tpr -maxwarn 2
gmx_cpu mdrun -deffnm eqbm_${T}K -x
echo "Equilibration complete for temperature $T"

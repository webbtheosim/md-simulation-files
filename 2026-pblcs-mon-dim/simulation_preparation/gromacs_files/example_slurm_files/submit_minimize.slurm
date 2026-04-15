#!/bin/bash
#SBATCH --job-name=M_minimize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jv6139@princeton.edu
#SBATCH --account=webb

module purge
module load openmpi/gcc/4.1.6
module load intel-mkl/2024.2


export GMX_MAXBACKUP=-1

gro=/scratch/gpfs/WEBB/jv6139/M_monomer/structures/256M_molecules/M_packed.gro
top=/scratch/gpfs/WEBB/jv6139/M_monomer/structures/256M_molecules/1M_monomer.top
control_loc=/scratch/gpfs/WEBB/jv6139/M_monomer/control-scripts/tREM/minimize
sim_loc=/scratch/gpfs/WEBB/jv6139/M_monomer/simulations/tREM/minimize

#make directory for this temperature if it doesn't exist
mkdir -p $sim_loc

cd $sim_loc
gmx_cpu grompp -f /scratch/gpfs/WEBB/jv6139/M_monomer/control-scripts/tREM/minimize/min.mdp -c $gro -p $top -o minimize.tpr -maxwarn 2
gmx_cpu mdrun -deffnm minimize -x




LOC=/scratch/gpfs/WEBB/jv6139/dimers/P2/structures
GROFILE=/scratch/gpfs/WEBB/jv6139/dimers/P2/structures/P2.gro #Monomer gro file
cd $LOC

gmx_cpu editconf -f /scratch/gpfs/WEBB/jv6139/dimers/P2/structures/P2.gro -o P2_box.gro -box 15.0 15.0 15.0 #center molecule 
gmx_cpu insert-molecules -f P2_box.gro -rot none -ci /scratch/gpfs/WEBB/jv6139/dimers/P2/structures/P2.gro --scale 0.3 -nmol 1023 -try 1000  -o P2_packed.gro 




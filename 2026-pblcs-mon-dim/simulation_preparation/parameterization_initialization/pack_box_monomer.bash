

LOC=/scratch/gpfs/WEBB/jv6139/M_monomer/structures/256M_molecules
GROFILE=/scratch/gpfs/WEBB/jv6139/M_monomer/structures/256M_molecules/1M_monomer.gro

cd $LOC
rm -rf $LOC/M_box.gro $LOC/M_packed.gro

gmx_cpu editconf -f $GROFILE -o M_box.gro -box 9 9 9 #center molecule 
gmx_cpu insert-molecules -f $LOC/M_box.gro -ci $GROFILE -nmol 255 -o M_packed.gro #insert the other 1023 molecules


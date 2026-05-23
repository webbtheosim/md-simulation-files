import os

for conc in [0,5,10,15,20,30,40]:
    for simno in [1,2,3,4]:
        os.system('cp /scratch/gpfs/WEBB/bu9134/Plasticization/chitosan/POLYOLS/TIP3P/{}/{}/init_templated.pdb {}/{}/'.format(conc,simno,conc,simno))

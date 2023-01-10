#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -J n1B_sep_equil
#SBATCH --mail-user=a.kalaja@rug.nl
#SBATCH --mail-type=ALL
#SBATCH -t 00:20:00
#SBATCH --mem=20GB

#OpenMP settings:
export OMP_NUM_THREADS=64
###export OMP_PLACES=threads
###export OMP_PROC_BIND=true


#run the application:
module load openmpi
# module load python
source activate albaenv
srun python /global/cscratch1/sd/akalaja/lensbispectrum_noisebiases/get_n1B.py
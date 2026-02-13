#!/bin/bash
#SBATCH --job-name=qe2pert
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --output=qe2pert.out
#SBATCH --error=qe2pert.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=TODO_EMAIL

source ~/.bashrc
ulimit -s unlimited

export OMP_NUM_THREADS=4
source TODO_ENV_SCRIPT

mpirun -np 4 qe2pert.x -npools 4 -i qe2pert.in > qe2pert.out

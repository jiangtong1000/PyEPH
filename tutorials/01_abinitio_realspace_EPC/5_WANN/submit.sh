#!/bin/bash
#SBATCH --job-name=wann
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --output=wann.out
#SBATCH --error=wann.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=TODO_EMAIL

source ~/.bashrc
ulimit -s unlimited

source TODO_ENV_SCRIPT

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS

mpirun -np 8 pw2wannier90.x -i pw2wan.in > pw2wan.out

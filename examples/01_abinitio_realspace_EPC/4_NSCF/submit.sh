#!/bin/bash
#SBATCH --job-name=nscf
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --output=nscf.out
#SBATCH --error=nscf.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=TODO_EMAIL

source ~/.bashrc
ulimit -s unlimited

source TODO_ENV_SCRIPT

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS

mpirun -np 24 pw.x -nk 8 < nscf.in > nscf.out

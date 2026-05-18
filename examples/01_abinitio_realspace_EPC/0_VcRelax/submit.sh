#!/bin/bash
#SBATCH --job-name=relax
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=100GB
#SBATCH --output=scf.out
#SBATCH --error=scf.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=TODO_EMAIL

source ~/.bashrc
ulimit -s unlimited

source TODO_ENV_SCRIPT    # TODO: path to QE/Perturbo environment

mpirun -np 24 pw.x -nk 8 < scf.in > scf.out # nk is parallelization factor for k-points

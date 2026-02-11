#!/bin/bash
#SBATCH --job-name=d3hess
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100GB
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=TODO_EMAIL

source ~/.bashrc
ulimit -s unlimited

source TODO_ENV_SCRIPT    # TODO: path to QE/Perturbo environment

d3hess.x < d3hess.in > d3hess.out
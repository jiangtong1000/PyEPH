#!/bin/bash
#SBATCH --job-name=ephmat
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=1:00:00
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=0                             # entire node memory
#SBATCH --output=pert.out
#SBATCH --error=pert.err

source ~/.bashrc
module load gcc/14.2.0-fasrc01              # TODO: adjust modules for your cluster
module load intel/24.2.1-fasrc01
module load openmpi/5.0.5-fasrc01

export MPICC=$(which mpicc)

conda activate shadow                       # TODO: conda environment with PyEPH deps

export PYTHONPATH=TODO_PYEPH_PATH:$PYTHONPATH  # TODO: path to PyEPH package
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u run_pyeph.py > pert.out

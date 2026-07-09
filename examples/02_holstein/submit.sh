#!/bin/bash
#SBATCH --partition=xxx
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=50000
#SBATCH --output=eph.out
#SBATCH --error=eph.err

source ~/.bashrc
module load xxxx
conda activate xxxx
export PYTHONPATH=/path-to-PyEPH:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun python run.py 0 > eph.out

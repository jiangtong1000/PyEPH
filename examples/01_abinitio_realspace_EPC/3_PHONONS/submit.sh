#!/bin/bash
#SBATCH --job-name=phonon
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=72:00:00
#SBATCH --ntasks=192                        # TODO: adjust (= ni * total cores used in SCF)
#SBATCH --cpus-per-task=1
#SBATCH --mem=500GB
#SBATCH --output=ph.out
#SBATCH --error=ph.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=TODO_EMAIL

source ~/.bashrc
ulimit -s unlimited

source TODO_ENV_SCRIPT

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# Heavy DFPT calculation
# -ni 8: parallelize over 8 q-points (images); adjust to match your q-grid
# -nk 8: k-point pools per image, has to be consistent with SCF

# Try to run with mpirun -np 24 ph.x -nk 8 < ph.in > ph.out
# Then check the output file to see how many irreducible q-points are used
# This gives us how to set the -ni parameter
mpirun -np 192 ph.x -ni 8 -nk 8 < ph.in > ph.out

# Fast merge: 24 ranks, no -ni, recover=.true.
mpirun -np 24 ph.x -nk 8 < ph2.in > ph2.out

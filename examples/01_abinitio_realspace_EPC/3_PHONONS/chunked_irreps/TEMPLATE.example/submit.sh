#!/bin/bash
#SBATCH --job-name=@JOB_NAME@
#SBATCH --account=TODO_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --time=TODO_TIME
#SBATCH --ntasks=TODO_NTASKS
#SBATCH --cpus-per-task=1
#SBATCH --mem=TODO_MEMORY
#SBATCH --output=ph.out
#SBATCH --error=ph.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

# @PREFIX@ q=@Q_INDEX@ irreps=@START_IRR@-@LAST_IRR@
: "${QE_ENV:?Set QE_ENV to a QE environment setup script before sbatch}"
: "${NPOOL:?Set NPOOL before sbatch}"
source "$QE_ENV"

echo "date=$(date --iso-8601=seconds)"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} tasks=${SLURM_NTASKS:-none}"
echo "ph.x=$(command -v ph.x)"

if (( SLURM_NTASKS % NPOOL != 0 )); then
  echo "SLURM_NTASKS must be divisible by NPOOL" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
srun ph.x -nk "$NPOOL" -in ph.in


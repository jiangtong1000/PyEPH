#!/bin/bash
#SBATCH --job-name=phonon
#SBATCH --account=TODO_ACCOUNT
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=72:00:00
#SBATCH --ntasks=TODO_NTASKS
#SBATCH --cpus-per-task=1
#SBATCH --mem=TODO_MEMORY
#SBATCH --output=phonon.slurm.out
#SBATCH --error=phonon.slurm.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

: "${QE_ENV:?Export QE_ENV as the QE environment setup script}"
: "${NIMAGE:?Export NIMAGE after inspecting the irreducible q list}"
: "${NPOOL:?Export NPOOL consistently with the SCF calculation}"
source "$QE_ENV"

echo "date=$(date --iso-8601=seconds)"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} tasks=${SLURM_NTASKS:-none}"
echo "ph.x=$(command -v ph.x)"

case "$NIMAGE:$NPOOL" in
  *[!0-9:]*|:*|*:) echo "NIMAGE and NPOOL must be positive integers" >&2; exit 2 ;;
esac
if [ "$NIMAGE" -lt 1 ] || [ "$NPOOL" -lt 1 ]; then
  echo "NIMAGE and NPOOL must be positive integers" >&2
  exit 2
fi
parallel_factor=$((NIMAGE * NPOOL))
if (( SLURM_NTASKS % parallel_factor != 0 )); then
  echo "SLURM_NTASKS must be divisible by NIMAGE * NPOOL" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Main q-point-distributed calculation. NIMAGE need not equal the q-point count.
srun ph.x -ni "$NIMAGE" -nk "$NPOOL" -in ph.in > ph.out

# Final single-image recollection. This runs only after the main command succeeds.
srun ph.x -nk "$NPOOL" -in ph2.in > ph2.out

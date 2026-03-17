#!/bin/bash
# ============================================================================
# Submit a Single NSBH Training Data Generation Job
# ============================================================================
# This script submits one large job (no multi-seed batching) so the
# distribution file is not traversed repeatedly by bayestar-inject.
#
# Usage:
#   ./slurm/nsbh_training/submit_nsbh_training_batch.sh
#   ./slurm/nsbh_training/submit_nsbh_training_batch.sh 100000 1000 8 42 "O5a" 200
#
# Args (positional):
#   NSAMPLES      - Number of injections (default: 100000)
#   MAX_DIST      - Maximum distance in Mpc (default: 1000)
#   NET_SNR_THR   - Network SNR threshold (default: 8)
#   SEED          - Random seed (default: 42)
#   RUNS          - Observing run(s) (default: O5a)
#   EVENTS_PER_TASK - Events per localization task (default: 200)
# ============================================================================

set -euo pipefail

NSAMPLES="${1:-100000}"
MAX_DIST="${2:-1000}"
NET_SNR_THR="${3:-8}"
SEED="${4:-42}"
RUNS="${5:-O5a}"
EVENTS_PER_TASK="${6:-200}"

echo "=============================================="
echo "NSBH Training Data Submission (Single Job)"
echo "=============================================="
echo "Samples:          $NSAMPLES"
echo "Max distance:      $MAX_DIST Mpc"
echo "Net SNR threshold: $NET_SNR_THR"
echo "Seed:             $SEED"
echo "Runs:             $RUNS"
echo "Events per task:  $EVENTS_PER_TASK"
echo "=============================================="

PROJECT_DIR=/fred/oz016/bgao_kn/observing-scenarios-simulations

cd "$PROJECT_DIR"

if [ ! -f "nsbh_training.h5" ]; then
  echo "Note: nsbh_training.h5 not found. The job will generate it."
fi

export NSAMPLES MAX_DIST NET_SNR_THR SEED RUNS EVENTS_PER_TASK
JOB_ID=$(sbatch --parsable --export=ALL slurm/nsbh_training/make_nsbh_training_data.sh)
PREP_JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID --export=ALL slurm/nsbh_training/prepare_nsbh_training_bayestar_array.sh)

echo ""
echo "=============================================="
echo "Job submitted!"
echo "Job ID: $JOB_ID"
echo "Localization prep job ID: $PREP_JOB_ID (afterok:$JOB_ID)"
echo "The prep job will split events and submit an array localization."
echo ""
echo "Monitor with: squeue -u $USER"
echo "=============================================="

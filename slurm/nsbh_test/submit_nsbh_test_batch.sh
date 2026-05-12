#!/bin/bash
# ============================================================================
# Submit a Single O5a NSBH Test Data Generation Job
# ============================================================================
#
# Usage:
#   bash slurm/nsbh_test/submit_nsbh_test_batch.sh
#   bash slurm/nsbh_test/submit_nsbh_test_batch.sh 50000 1000 8 42 200
#
# Args (positional):
#   NSAMPLES        - Number of injections (default: 50000)
#   MAX_DIST        - Maximum distance in Mpc (default: 1000)
#   NET_SNR_THR     - Network SNR threshold (default: 8)
#   SEED            - Random seed (default: 42)
#   EVENTS_PER_TASK - Events per localization task (default: 200)
# ============================================================================

set -euo pipefail

NSAMPLES="${1:-50000}"
MAX_DIST="${2:-1000}"
NET_SNR_THR="${3:-8}"
SEED="${4:-42}"
EVENTS_PER_TASK="${5:-200}"
RUNS="${RUNS:-O5a}"
FORCE_DIST="${FORCE_DIST:-0}"

if [ "$RUNS" != "O5a" ]; then
  echo "NSBH test workflow is O5a-only; got RUNS=\"$RUNS\"."
  exit 2
fi

PROJECT_DIR=/fred/oz016/bgao_kn/observing-scenarios-simulations

cd "$PROJECT_DIR"
mkdir -p logs logs/arrays

echo "=============================================="
echo "NSBH Test Data Submission (O5a Single Job)"
echo "=============================================="
echo "Samples:          $NSAMPLES"
echo "Max distance:      $MAX_DIST Mpc"
echo "Net SNR threshold: $NET_SNR_THR"
echo "Seed:             $SEED"
echo "Events per task:  $EVENTS_PER_TASK"
echo "Force distribution regeneration: $FORCE_DIST"
echo "=============================================="

if [ ! -f "nsbh_test.h5" ]; then
  echo "Note: nsbh_test.h5 not found. The job will generate it."
fi

export NSAMPLES MAX_DIST NET_SNR_THR SEED EVENTS_PER_TASK RUNS FORCE_DIST
JOB_ID=$(sbatch --parsable --export=ALL slurm/nsbh_test/make_nsbh_test_data.sh)
PREP_JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID --export=ALL slurm/nsbh_test/prepare_nsbh_test_bayestar_array.sh)

echo ""
echo "=============================================="
echo "Job submitted!"
echo "Generation job ID: $JOB_ID"
echo "Localization prep job ID: $PREP_JOB_ID (afterok:$JOB_ID)"
echo "The prep job will split events and submit an array localization."
echo ""
echo "Monitor with: squeue -u $USER"
echo "Expected output directory: runs/O5a/nsbh_test_seed${SEED}/"
echo "=============================================="

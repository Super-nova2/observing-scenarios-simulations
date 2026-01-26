#!/bin/bash
# ============================================================================
# Submit a Single BNS Training Data Generation Job
# ============================================================================
# This script submits one large job (no multi-seed batching) so the
# distribution file is not traversed repeatedly by bayestar-inject.
#
# Usage:
#   ./slurm/submit_bns_training_batch.sh
#   ./slurm/submit_bns_training_batch.sh 1000000 400 8 42 "O5a"
#
# Args (positional):
#   NSAMPLES      - Number of injections (default: 1000000)
#   MAX_DIST      - Maximum distance in Mpc (default: 400)
#   NET_SNR_THR   - Network SNR threshold (default: 8)
#   SEED          - Random seed (default: 42)
#   RUNS          - Observing run(s) (default: O5a)
# ============================================================================

set -euo pipefail

NSAMPLES="${1:-100000}"
MAX_DIST="${2:-500}"
NET_SNR_THR="${3:-8}"
SEED="${4:-42}"
RUNS="${5:-O5a}"

echo "=============================================="
echo "BNS Training Data Submission (Single Job)"
echo "=============================================="
echo "Samples:          $NSAMPLES"
echo "Max distance:      $MAX_DIST Mpc"
echo "Net SNR threshold: $NET_SNR_THR"
echo "Seed:             $SEED"
echo "Runs:             $RUNS"
echo "=============================================="

PROJECT_DIR=/fred/oz016/bgao_kn/observing-scenarios-simulations

cd "$PROJECT_DIR"

if [ ! -f "bns_training.h5" ]; then
  echo "Note: bns_training.h5 not found. The job will generate it."
fi

export NSAMPLES MAX_DIST NET_SNR_THR SEED RUNS
JOB_ID=$(sbatch --parsable --export=ALL slurm/make_bns_training_data.sh)
LOC_JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID --export=ALL slurm/run_bns_training_bayestar.sh)

echo ""
echo "=============================================="
echo "Job submitted!"
echo "Job ID: $JOB_ID"
echo "Localization job ID: $LOC_JOB_ID (afterok:$JOB_ID)"
echo ""
echo "Monitor with: squeue -u $USER"
echo "If you split runs manually, combine with: python scripts/combine_bns_training_data.py"
echo "=============================================="

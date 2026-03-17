#!/bin/bash
#SBATCH --job-name=nsbh-train-gen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --tmp=1G
#SBATCH --output=logs/%x-%j.out

# ============================================================================
# NSBH Training Data Generator
# ============================================================================
# This script generates NSBH GW events optimized for deep learning training.
# It uses a custom uniform distribution and relaxed detection thresholds
# to maximize parameter space coverage.
#
# NSBH systems have asymmetric parameters:
#   mass1 (BH): [2.5, 20.0] Msun, spin [-0.99, 0.99]
#   mass2 (NS): [1.0, 2.5] Msun,  spin [-0.5, 0.5]
#
# Usage:
#   sbatch slurm/nsbh_training/make_nsbh_training_data.sh
#   sbatch --export=ALL,SEED=42,MAX_DIST=800 slurm/nsbh_training/make_nsbh_training_data.sh
#
# Configuration via environment variables:
#   SEED         - Random seed (default: 42)
#   RUNS         - Observing run (default: O5a)
#   NSAMPLES     - Number of injections (default: 500000)
#   MAX_DIST     - Maximum distance in Mpc (default: 1000)
#   NET_SNR_THR  - Network SNR threshold (default: 5, lower = more events)
#   DIST_FILE    - Distribution file (default: nsbh_training.h5)
# ============================================================================

set -euo pipefail

# Clean environment
unset VIRTUAL_ENV
unset PYTHONPATH
export PYTHONNOUSERSITE=1

module unload python-scientific >/dev/null 2>&1 || true
module unload python/3.10.4 >/dev/null 2>&1 || true
module load gcccore/13.2.0
module load python/3.11.5
which python

PROJECT_DIR=/fred/oz016/bgao_kn/observing-scenarios-simulations
export PATH="$HOME/.local/bin:$PATH"
JOB_TMP_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-/tmp}}}"
export UV_CACHE_DIR="${JOB_TMP_DIR}/uv-cache"
mkdir -p "$UV_CACHE_DIR"
export BAYESTAR_JOBS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=1

if [ -d "$HOME/lalsuite-waveform-data" ]; then
  export LAL_DATA_PATH="$HOME/lalsuite-waveform-data"
fi

# Configuration with defaults
SEED="${SEED:-42}"
RUNS="${RUNS:-O5a}"
NSAMPLES="${NSAMPLES:-100000}"
MAX_DIST="${MAX_DIST:-1000}"
NET_SNR_THR="${NET_SNR_THR:-8}"
DIST_FILE="${DIST_FILE:-nsbh_training.h5}"

cd "$PROJECT_DIR"
mkdir -p logs

echo "=============================================="
echo "NSBH Training Data Generator"
echo "=============================================="
echo "Run:              $RUNS"
echo "Seed:             $SEED"
echo "N samples:        $NSAMPLES"
echo "Max distance:     $MAX_DIST Mpc"
echo "Net SNR threshold: $NET_SNR_THR"
echo "Distribution:     $DIST_FILE"
echo "=============================================="

# Step 0: Generate distribution file if needed
if [ ! -f "$DIST_FILE" ]; then
  echo "[Step 0/4] Generating NSBH training distribution file..."
  uv run python scripts/generate_nsbh_training_distribution.py \
    -o "$DIST_FILE" \
    -n "$NSAMPLES" \
    --bh-mass-min 2.0 \
    --bh-mass-max 20.0 \
    --ns-mass-min 1.0 \
    --ns-mass-max 2.0 \
    --bh-spin-max 0.99 \
    --ns-spin-max 0.5 \
    --seed "$SEED" \
    --stratified
else
  echo "[Step 0/4] Using existing distribution file: $DIST_FILE"
fi

# Check PSD files exist
for run in $RUNS; do
  if [ ! -s "runs/$run/psds.xml" ]; then
    echo "Missing runs/$run/psds.xml. Run 'uv run make RUNS=\"$RUNS\" psds' first."
    exit 2
  fi
done

# Determine detectors
get_detectors() {
  local run=$1
  case "$run" in
    O4HL)  echo "H1 L1" ;;
    O4HLV) echo "H1 L1 V1" ;;
    O5*)   echo "H1 L1 V1" ;;
    *)     echo "H1 L1" ;;
  esac
}

for run in $RUNS; do
  echo ""
  echo "Processing run: $run"

  OUTDIR="runs/$run/nsbh_training_seed${SEED}"
  mkdir -p "$OUTDIR"

  # Step 1: Generate NSBH injections with custom distribution
  echo "[Step 1/4] Generating NSBH injections..."
  uv run bayestar-inject -l error --seed "$SEED" \
    -o "$OUTDIR/injections.xml" \
    -j "$BAYESTAR_JOBS" \
    --snr-threshold 1 \
    --distribution-samples "$DIST_FILE" \
    --reference-psd "runs/$run/psds.xml" \
    --min-triggers 1 \
    --max-distance "$MAX_DIST" \
    --nsamples "$NSAMPLES"

  # Step 2: Generate detected events with relaxed thresholds
  echo "[Step 2/4] Generating detected events..."
  DETECTORS=$(get_detectors "$run")
  uv run bayestar-realize-coincs \
    --seed "$SEED" \
    -j "$BAYESTAR_JOBS" \
    -l error \
    -o "$OUTDIR/events.xml.gz" \
    "$OUTDIR/injections.xml" \
    --reference-psd "runs/$run/psds.xml" \
    --snr-threshold 1 \
    --net-snr-threshold "$NET_SNR_THR" \
    --min-triggers 1 \
    --duty-cycle 0.7 \
    --keep-subthreshold \
    --measurement-error gaussian-noise \
    --detector $DETECTORS

  # Step 3: Generate injections.dat
  echo "[Step 3/4] Generating injections.dat..."
  INJECTIONS_COLS="simulation_id longitude latitude inclination distance mass1 mass2 spin1z spin2z"
  echo "${INJECTIONS_COLS// /	}" > "$OUTDIR/injections.dat"
  uv run igwn_ligolw_print -t sim_inspiral \
    $(echo "$INJECTIONS_COLS" | tr ' ' '\n' | sed 's/^/-c /') \
    -d "	" "$OUTDIR/events.xml.gz" >> "$OUTDIR/injections.dat"

  # Step 4: Generate coincs.dat
  echo "[Step 4/4] Generating coincs.dat..."
  COINCS_COLS="coinc_event_id ifos snr"
  echo "${COINCS_COLS// /	}" > "$OUTDIR/coincs.dat"
  uv run igwn_ligolw_print -t coinc_inspiral \
    $(echo "$COINCS_COLS" | tr ' ' '\n' | sed 's/^/-c /') \
    -d "	" "$OUTDIR/events.xml.gz" >> "$OUTDIR/coincs.dat"

  # Summary
  N_INJECTIONS=$(($(wc -l < "$OUTDIR/injections.dat") - 1))
  N_DETECTED=$(($(wc -l < "$OUTDIR/coincs.dat") - 1))
  DETECTION_RATE=$(echo "scale=2; 100 * $N_DETECTED / $NSAMPLES" | bc)

  echo ""
  echo "=============================================="
  echo "Summary for $run (seed=$SEED)"
  echo "=============================================="
  echo "Output directory: $OUTDIR"
  echo "Total injections: $N_INJECTIONS"
  echo "Detected events:  $N_DETECTED"
  echo "Detection rate:   ${DETECTION_RATE}%"
  echo ""
  ls -lh "$OUTDIR"
  echo "=============================================="
done

echo ""
echo "All done! Next step: Run BAYESTAR skymap generation."
echo "Use: sbatch --export=ALL,SEED=$SEED slurm/nsbh_training/run_nsbh_training_bayestar.sh"

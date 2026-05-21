#!/bin/bash
#SBATCH --job-name=bns-test-gen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

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

RUNS="${RUNS:-O5aLVK}"
SEED="${SEED:-42}"
NSAMPLES="${NSAMPLES:-100000}"
MAX_DIST="${MAX_DIST:-500}"
NET_SNR_THR="${NET_SNR_THR:-8}"
DIST_FILE="${DIST_FILE:-bns_test.h5}"
FORCE_DIST="${FORCE_DIST:-0}"
MASS_METHOD="${MASS_METHOD:-bgp}"
BGP_INPUT="${BGP_INPUT:-AllCBC_FullPopBGP.h5}"

if [ "$RUNS" != "O5aLVK" ]; then
  echo "BNS test workflow is O5aLVK-only; got RUNS=\"$RUNS\"."
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p logs

echo "=============================================="
echo "BNS Test Data Generator"
echo "=============================================="
echo "Run:               $RUNS"
echo "Seed:              $SEED"
echo "N samples:         $NSAMPLES"
echo "Max distance:      $MAX_DIST Mpc"
echo "Net SNR threshold: $NET_SNR_THR"
echo "Distribution:      $DIST_FILE"
echo "Mass method:       $MASS_METHOD"
echo "BGP input:         $BGP_INPUT"
echo "Force distribution regeneration: $FORCE_DIST"
echo "=============================================="

if [ "$MASS_METHOD" = "bgp" ] && [ ! -f "$BGP_INPUT" ]; then
  echo "Missing $BGP_INPUT. Run 'uv run make AllCBC_FullPopBGP.h5' on a login node first."
  exit 2
fi

if [ "$FORCE_DIST" -eq 1 ] || [ ! -f "$DIST_FILE" ]; then
  echo "[Step 0/4] Generating BNS test distribution file..."
  uv run python scripts/generate_bns_test_distribution.py \
    -o "$DIST_FILE" \
    -n "$NSAMPLES" \
    --seed "$SEED" \
    --mass-method "$MASS_METHOD" \
    --bgp-input "$BGP_INPUT"
else
  echo "[Step 0/4] Using existing distribution file: $DIST_FILE"
fi

if [ ! -s "runs/$RUNS/psds.xml" ]; then
  echo "Missing runs/$RUNS/psds.xml. Run 'uv run make RUNS=\"$RUNS\" psds' first."
  exit 2
fi

OUTDIR="runs/$RUNS/bns_test_seed${SEED}"
mkdir -p "$OUTDIR"

echo "[Step 1/4] Generating BNS test injections..."
uv run bayestar-inject -l error --seed "$SEED" \
  -o "$OUTDIR/injections.xml" \
  -j "$BAYESTAR_JOBS" \
  --snr-threshold 1 \
  --distribution-samples "$DIST_FILE" \
  --reference-psd "runs/$RUNS/psds.xml" \
  --min-triggers 1 \
  --max-distance "$MAX_DIST" \
  --nsamples "$NSAMPLES"

echo "[Step 2/4] Generating detected events..."
uv run bayestar-realize-coincs \
  --seed "$SEED" \
  -j "$BAYESTAR_JOBS" \
  -l error \
  -o "$OUTDIR/events.xml.gz" \
  "$OUTDIR/injections.xml" \
  --reference-psd "runs/$RUNS/psds.xml" \
  --snr-threshold 1 \
  --net-snr-threshold "$NET_SNR_THR" \
  --min-triggers 1 \
  --duty-cycle 0.7 \
  --keep-subthreshold \
  --measurement-error gaussian-noise \
  --detector H1 L1 V1 K1

echo "[Step 3/4] Generating injections.dat..."
INJECTIONS_COLS="simulation_id longitude latitude inclination distance mass1 mass2 spin1z spin2z"
echo "${INJECTIONS_COLS// /	}" > "$OUTDIR/injections.dat"
uv run igwn_ligolw_print -t sim_inspiral \
  $(echo "$INJECTIONS_COLS" | tr ' ' '\n' | sed 's/^/-c /') \
  -d "	" "$OUTDIR/events.xml.gz" >> "$OUTDIR/injections.dat"

echo "[Step 4/4] Generating coincs.dat..."
COINCS_COLS="coinc_event_id ifos snr"
echo "${COINCS_COLS// /	}" > "$OUTDIR/coincs.dat"
uv run igwn_ligolw_print -t coinc_inspiral \
  $(echo "$COINCS_COLS" | tr ' ' '\n' | sed 's/^/-c /') \
  -d "	" "$OUTDIR/events.xml.gz" >> "$OUTDIR/coincs.dat"

N_INJECTIONS=$(($(wc -l < "$OUTDIR/injections.dat") - 1))
N_DETECTED=$(($(wc -l < "$OUTDIR/coincs.dat") - 1))
DETECTION_RATE=$(echo "scale=2; 100 * $N_DETECTED / $NSAMPLES" | bc)

echo ""
echo "=============================================="
echo "Summary for $RUNS BNS test (seed=$SEED)"
echo "=============================================="
echo "Output directory: $OUTDIR"
echo "Total injections: $N_INJECTIONS"
echo "Detected events:  $N_DETECTED"
echo "Detection rate:   ${DETECTION_RATE}%"
echo ""
ls -lh "$OUTDIR"
echo "=============================================="
echo ""
echo "All done! Next step: Run BAYESTAR skymap generation."
echo "Use: sbatch --export=ALL,SEED=$SEED slurm/bns_test/prepare_bns_test_bayestar_array.sh"

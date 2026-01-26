#!/bin/bash
#SBATCH --job-name=gw-bns-inject
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

unset VIRTUAL_ENV
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Use a clean Python module to avoid numpy/astropy ABI mismatches.
module unload python-scientific >/dev/null 2>&1 || true
module unload python/3.10.4 >/dev/null 2>&1 || true
module load gcccore/13.2.0
module load python/3.11.5
which python

PROJECT_DIR=/fred/oz016/bgao_kn/observing-scenarios-simulations
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$PROJECT_DIR/.uv-cache"
export BAYESTAR_JOBS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=1

if [ -d "$HOME/lalsuite-waveform-data" ]; then
  export LAL_DATA_PATH="$HOME/lalsuite-waveform-data"
fi

# Configuration - override at submit time if needed
# sbatch --export=ALL,RUNS="O5a",BNS_NSAMPLES=100000 slurm/make_bns_injections.sh
RUNS="${RUNS:-O5a}"
BNS_NSAMPLES="${BNS_NSAMPLES:-50000}"
BNS_DISTRIBUTION="${BNS_DISTRIBUTION:-bns_astro}"  # or bns_broad

cd "$PROJECT_DIR"
mkdir -p logs

# Check PSD files exist
missing=0
for run in $RUNS; do
  if [ ! -s "runs/$run/psds.xml" ]; then
    echo "Missing runs/$run/psds.xml. Run 'uv run make RUNS=\"$RUNS\" psds' on the login node first."
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  exit 2
fi

# Determine detectors from PSD configuration
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
  echo "=========================================="
  echo "Processing BNS injections for run: $run"
  echo "Distribution: $BNS_DISTRIBUTION"
  echo "Number of samples: $BNS_NSAMPLES"
  echo "=========================================="

  OUTDIR="runs/$run/bns"
  mkdir -p "$OUTDIR"

  # Step 1: Generate BNS injections
  echo "[Step 1/4] Generating BNS injections..."
  uv run bayestar-inject -l error --seed 1 \
    -o "$OUTDIR/injections.xml" \
    -j "$BAYESTAR_JOBS" \
    --snr-threshold 1 \
    --distribution "$BNS_DISTRIBUTION" \
    --reference-psd "runs/$run/psds.xml" \
    --min-triggers 1 \
    --nsamples "$BNS_NSAMPLES"

  # Step 2: Generate detected events (coincs)
  echo "[Step 2/4] Generating detected events..."
  DETECTORS=$(get_detectors "$run")
  uv run bayestar-realize-coincs \
    --seed 1 \
    -j "$BAYESTAR_JOBS" \
    -l error \
    -o "$OUTDIR/events.xml.gz" \
    "$OUTDIR/injections.xml" \
    --reference-psd "runs/$run/psds.xml" \
    --snr-threshold 1 \
    --net-snr-threshold 8 \
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
  echo ""
  echo "=========================================="
  echo "BNS injection complete for $run"
  echo "Output directory: $OUTDIR"
  echo "Files generated:"
  ls -lh "$OUTDIR"
  echo ""
  echo "Injection stats:"
  echo "  Total injections: $(wc -l < "$OUTDIR/injections.dat") (including header)"
  echo "  Detected coincs:  $(wc -l < "$OUTDIR/coincs.dat") (including header)"
  echo "=========================================="
done

echo ""
echo "All BNS injections complete!"
echo "Next step: Run skymap generation with run_bns_bayestar.sh"

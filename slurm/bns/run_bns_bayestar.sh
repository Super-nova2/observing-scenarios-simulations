#!/bin/bash
#SBATCH --job-name=gw-bns-bayestar
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --gres=tmp:1G
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

PROJECT_DIR=/fred/oz016/bgao_kn/observing-scenarios-simulations
export PATH="$HOME/.local/bin:$PATH"
JOB_TMP_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-/tmp}}}"
export UV_CACHE_DIR="${JOB_TMP_DIR}/uv-cache"
mkdir -p "$UV_CACHE_DIR"

if [ -d "$HOME/lalsuite-waveform-data" ]; then
  export LAL_DATA_PATH="$HOME/lalsuite-waveform-data"
fi

# Override at submit time: sbatch --export=ALL,RUNS="O5a O5b O5c"
RUNS="${RUNS:-O5a}"

cd "$PROJECT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

found=0
for run in $RUNS; do
  eventsfile="runs/$run/bns/events.xml.gz"
  if [ ! -e "$eventsfile" ]; then
    echo "Warning: $eventsfile not found, skipping $run"
    continue
  fi

  found=1
  outdir="runs/$run/bns/allsky"

  echo "=========================================="
  echo "Generating skymaps for $run BNS events"
  echo "Input: $eventsfile"
  echo "Output: $outdir"
  echo "=========================================="

  uv run bayestar-localize-coincs "$eventsfile" -o "$outdir" --f-low 11 --cosmology

  echo ""
  echo "Skymap generation complete for $run"
  echo "Number of skymaps: $(ls -1 "$outdir"/*.fits* 2>/dev/null | wc -l)"
  echo ""
done

if [ "$found" -eq 0 ]; then
  echo "No BNS events.xml.gz files found for RUNS=\"$RUNS\". Run make_bns_injections.sh first."
  exit 2
fi

echo "All BNS skymap generation complete!"

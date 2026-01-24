#!/bin/bash
#SBATCH --job-name=gw-bayestar
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=10G
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
export UV_CACHE_DIR="$PROJECT_DIR/.uv-cache"

if [ -d "$HOME/lalsuite-waveform-data" ]; then
  export LAL_DATA_PATH="$HOME/lalsuite-waveform-data"
fi

# Override at submit time: sbatch --export=ALL,RUNS="O5a O5b O5c"
RUNS="${RUNS:-O5a O5b O5c}"

cd "$PROJECT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

found=0
for run in $RUNS; do
  for eventsfile in runs/$run/*/events.xml.gz; do
    if [ ! -e "$eventsfile" ]; then
      continue
    fi
    found=1
    outdir="$(dirname "$eventsfile")/allsky"
    uv run bayestar-localize-coincs "$eventsfile" -o "$outdir" --f-low 11 --cosmology
  done
done

if [ "$found" -eq 0 ]; then
  echo "No events.xml.gz files found for RUNS=\"$RUNS\". Run injections first."
  exit 2
fi

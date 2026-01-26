#!/bin/bash
#SBATCH --job-name=gw-injections
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
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
export BAYESTAR_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1

if [ -d "$HOME/lalsuite-waveform-data" ]; then
  export LAL_DATA_PATH="$HOME/lalsuite-waveform-data"
fi

# Override at submit time: sbatch --export=ALL,RUNS="O5a O5b O5c"
#RUNS="${RUNS:-O5a O5b O5c}"
RUNS="${RUNS:-O5a}"

cd "$PROJECT_DIR"
mkdir -p logs

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

if ! tar -tf analyses_AllCBC.tar >/dev/null 2>&1; then
  echo "analyses_AllCBC.tar is missing or invalid. Download it on the login node before submitting."
  exit 2
fi

uv run make RUNS="$RUNS" injections

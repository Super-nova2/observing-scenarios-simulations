#!/bin/bash
#SBATCH --job-name=gw-bns-test-bayestar-array
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/arrays/%x-%A_%a.out

set -euo pipefail

unset VIRTUAL_ENV
unset PYTHONPATH
export PYTHONNOUSERSITE=1

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

EVENT_LIST="${EVENT_LIST:-}"
EVENTS_PER_TASK="${EVENTS_PER_TASK:-50}"
F_LOW="${F_LOW:-11}"

if [ -z "$EVENT_LIST" ]; then
  echo "EVENT_LIST is required."
  exit 2
fi

if [ ! -s "$EVENT_LIST" ]; then
  echo "EVENT_LIST not found or empty: $EVENT_LIST"
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p logs logs/arrays

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

task_id="${SLURM_ARRAY_TASK_ID:-1}"
start=$(( (task_id - 1) * EVENTS_PER_TASK + 1 ))
end=$(( task_id * EVENTS_PER_TASK ))

mapfile -t events < <(sed -n "${start},${end}p" "$EVENT_LIST")

if [ "${#events[@]}" -eq 0 ]; then
  echo "No events assigned to task $task_id (lines ${start}-${end})."
  exit 0
fi

echo "Processing ${#events[@]} BNS test event files (task $task_id)."

for eventfile in "${events[@]}"; do
  if [ ! -e "$eventfile" ]; then
    echo "Warning: missing event file: $eventfile"
    continue
  fi

  outdir="$(dirname "$eventfile")/../allsky"
  mkdir -p "$outdir"

  uv run bayestar-localize-coincs "$eventfile" -o "$outdir" --f-low "$F_LOW" --cosmology
done

echo "Task $task_id complete."

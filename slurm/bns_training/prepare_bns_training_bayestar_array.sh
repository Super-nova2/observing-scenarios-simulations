#!/bin/bash
#SBATCH --job-name=gw-bns-train-bayestar-prep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --tmp=1G
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

# Override at submit time:
#   sbatch --export=ALL,RUNS="O5a O5b",SEED=42,EVENTS_PER_TASK=200 slurm/bns_training/prepare_bns_training_bayestar_array.sh
RUNS="${RUNS:-O5a}"
SEED="${SEED:-42}"
EVENTS_PER_TASK="${EVENTS_PER_TASK:-200}"
F_LOW="${F_LOW:-11}"
FORCE_SPLIT="${FORCE_SPLIT:-0}"

cd "$PROJECT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=1

for run in $RUNS; do
  eventsfile="runs/$run/bns_training_seed${SEED}/events.xml.gz"
  split_dir="runs/$run/bns_training_seed${SEED}/events_split"
  list_file="runs/$run/bns_training_seed${SEED}/events_split.list"

  if [ ! -e "$eventsfile" ]; then
    echo "Warning: $eventsfile not found, skipping $run"
    continue
  fi

  if [ "$FORCE_SPLIT" -eq 1 ]; then
    rm -rf "$split_dir" "$list_file"
  fi

  if [ ! -d "$split_dir" ]; then
    echo "Splitting events for $run (seed=$SEED)..."
    uv run python scripts/split-events.py "$eventsfile" "$split_dir"
  fi

  if [ ! -s "$list_file" ]; then
    find "$split_dir" -maxdepth 1 -name "*.xml.gz" -print | sort > "$list_file"
  fi

  if [ ! -s "$list_file" ]; then
    echo "No split events found for $run (seed=$SEED)."
    continue
  fi

  n_events=$(wc -l < "$list_file")
  n_tasks=$(( (n_events + EVENTS_PER_TASK - 1) / EVENTS_PER_TASK ))

  echo "Submitting localization array for $run (seed=$SEED)"
  echo "  Events: $n_events"
  echo "  Events per task: $EVENTS_PER_TASK"
  echo "  Array tasks: $n_tasks"

  ARRAY_JOB_ID=$(sbatch --parsable \
    --array=1-"$n_tasks" \
    --export=ALL,EVENT_LIST="$list_file",EVENTS_PER_TASK="$EVENTS_PER_TASK",F_LOW="$F_LOW" \
    slurm/bns_training/run_bns_training_bayestar_array.sh)

  echo "  -> Array job ID: $ARRAY_JOB_ID"
done

#!/bin/bash
#SBATCH --job-name=gw-bns-test-bayestar-prep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

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

RUNS="${RUNS:-O5a}"
SEED="${SEED:-42}"
EVENTS_PER_TASK="${EVENTS_PER_TASK:-200}"
F_LOW="${F_LOW:-11}"
FORCE_SPLIT="${FORCE_SPLIT:-0}"

if [ "$RUNS" != "O5a" ]; then
  echo "BNS test workflow is O5a-only; got RUNS=\"$RUNS\"."
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p logs logs/arrays

export OMP_NUM_THREADS=1

eventsfile="runs/$RUNS/bns_test_seed${SEED}/events.xml.gz"
split_dir="runs/$RUNS/bns_test_seed${SEED}/events_split"
list_file="runs/$RUNS/bns_test_seed${SEED}/events_split.list"

if [ ! -e "$eventsfile" ]; then
  echo "$eventsfile not found. Run slurm/bns_test/make_bns_test_data.sh first."
  exit 2
fi

if [ "$FORCE_SPLIT" -eq 1 ]; then
  rm -rf "$split_dir" "$list_file"
fi

if [ ! -d "$split_dir" ]; then
  echo "Splitting events for $RUNS BNS test (seed=$SEED)..."
  uv run python scripts/split-events.py "$eventsfile" "$split_dir"
fi

if [ ! -s "$list_file" ]; then
  find "$split_dir" -maxdepth 1 -name "*.xml.gz" -print | sort > "$list_file"
fi

if [ ! -s "$list_file" ]; then
  echo "No split events found for $RUNS BNS test (seed=$SEED)."
  exit 2
fi

n_events=$(wc -l < "$list_file")
n_tasks=$(( (n_events + EVENTS_PER_TASK - 1) / EVENTS_PER_TASK ))

echo "Submitting BNS test localization array for $RUNS (seed=$SEED)"
echo "  Events: $n_events"
echo "  Events per task: $EVENTS_PER_TASK"
echo "  Array tasks: $n_tasks"

ARRAY_JOB_ID=$(sbatch --parsable \
  --array=1-"$n_tasks" \
  --export=ALL,EVENT_LIST="$list_file",EVENTS_PER_TASK="$EVENTS_PER_TASK",F_LOW="$F_LOW" \
  slurm/bns_test/run_bns_test_bayestar_array.sh)

echo "  -> Array job ID: $ARRAY_JOB_ID"

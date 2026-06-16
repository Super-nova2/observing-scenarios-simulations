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
BASE_DIR="${BASE_DIR:-/fred/oz016/bgao_kn}"
SKYMAP_DIR="${SKYMAP_DIR:-$BASE_DIR/data/skymap/bns_skymap_test}"

if [ "$RUNS" != "O5a" ]; then
  echo "BNS test workflow is O5a-only; got RUNS=\"$RUNS\"."
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p logs logs/arrays

export OMP_NUM_THREADS=1

run_dir="runs/$RUNS/bns_test_seed${SEED}"
eventsfile="$run_dir/events.xml.gz"
coincs_file="$run_dir/coincs.dat"

if [ ! -e "$eventsfile" ]; then
  echo "$eventsfile not found. Run slurm/bns_test/make_bns_test_data.sh first."
  exit 2
fi

if [ ! -s "$coincs_file" ]; then
  echo "$coincs_file not found or empty."
  exit 2
fi

# A source-specific directory prevents a regenerated events.xml.gz from
# inheriting split files or lists from an earlier simulation.
events_signature=$(stat -c '%s-%Y-%i' "$eventsfile")
split_dir="$run_dir/events_split_$events_signature"
list_file="$run_dir/events_split_$events_signature.list"
missing_list="$run_dir/events_missing_$events_signature.list"
complete_file="$split_dir/.complete"
expected_events=$(( $(wc -l < "$coincs_file") - 1 ))

if [ "$expected_events" -le 0 ]; then
  echo "No detected events found in $coincs_file."
  exit 2
fi

if [ "$FORCE_SPLIT" -eq 1 ]; then
  rm -rf "$split_dir" "$list_file" "$missing_list"
fi

if [ ! -f "$complete_file" ]; then
  rm -rf "$split_dir" "$list_file" "$missing_list"
  echo "Splitting events for $RUNS BNS test (seed=$SEED)..."
  uv run python scripts/split-events.py "$eventsfile" "$split_dir"

  list_tmp="${list_file}.tmp.${SLURM_JOB_ID:-$$}"
  find "$split_dir" -maxdepth 1 -name "*.xml.gz" -print | sort -V > "$list_tmp"
  split_events=$(wc -l < "$list_tmp")
  if [ "$split_events" -ne "$expected_events" ]; then
    echo "Split event count mismatch: expected $expected_events, found $split_events."
    rm -f "$list_tmp"
    exit 2
  fi
  mv "$list_tmp" "$list_file"
  touch "$complete_file"
fi

if [ ! -s "$list_file" ]; then
  echo "No split events found for $RUNS BNS test (seed=$SEED)."
  exit 2
fi

split_events=$(wc -l < "$list_file")
if [ "$split_events" -ne "$expected_events" ]; then
  echo "Split event count mismatch: expected $expected_events, found $split_events."
  echo "Rerun with FORCE_SPLIT=1 to rebuild this source version."
  exit 2
fi

missing_tmp="${missing_list}.tmp.${SLURM_JOB_ID:-$$}"
: > "$missing_tmp"
while IFS= read -r eventfile; do
  event_id=$(basename "$eventfile" .xml.gz)
  fits_file="$SKYMAP_DIR/$event_id.fits"
  if [ ! -s "$fits_file" ] || [ ! "$fits_file" -nt "$eventsfile" ]; then
    printf '%s\n' "$eventfile" >> "$missing_tmp"
  fi
done < "$list_file"
mv "$missing_tmp" "$missing_list"

n_missing=$(wc -l < "$missing_list")
n_complete=$(( expected_events - n_missing ))

echo "BNS test localization status for $RUNS (seed=$SEED)"
echo "  Detected events: $expected_events"
echo "  Verified current sky maps: $n_complete"
echo "  Missing or stale sky maps: $n_missing"
echo "  Split source signature: $events_signature"
echo "  Sky-map directory: $SKYMAP_DIR"

if [ "$n_missing" -eq 0 ]; then
  echo "All BNS test events already have current sky maps; no array submitted."
  exit 0
fi

n_tasks=$(( (n_missing + EVENTS_PER_TASK - 1) / EVENTS_PER_TASK ))

echo "Submitting BNS test localization array for $RUNS (seed=$SEED)"
echo "  Events: $n_missing"
echo "  Events per task: $EVENTS_PER_TASK"
echo "  Array tasks: $n_tasks"

ARRAY_JOB_ID=$(sbatch --parsable \
  --array=1-"$n_tasks" \
  --export=ALL,EVENT_LIST="$PROJECT_DIR/$missing_list",EVENTS_SOURCE="$PROJECT_DIR/$eventsfile",SKYMAP_DIR="$SKYMAP_DIR",EVENTS_PER_TASK="$EVENTS_PER_TASK",F_LOW="$F_LOW" \
  slurm/bns_test/run_bns_test_bayestar_array.sh)

echo "  -> Array job ID: $ARRAY_JOB_ID"

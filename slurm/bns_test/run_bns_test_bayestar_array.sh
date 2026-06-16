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
EVENTS_SOURCE="${EVENTS_SOURCE:-}"
SKYMAP_DIR="${SKYMAP_DIR:-}"
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

if [ ! -s "$EVENTS_SOURCE" ]; then
  echo "EVENTS_SOURCE not found or empty: $EVENTS_SOURCE"
  exit 2
fi

if [ -z "$SKYMAP_DIR" ]; then
  echo "SKYMAP_DIR is required."
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p logs logs/arrays
mkdir -p "$SKYMAP_DIR"

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

  event_id=$(basename "$eventfile" .xml.gz)
  fits_file="$SKYMAP_DIR/$event_id.fits"
  if [ -s "$fits_file" ] && [ "$fits_file" -nt "$EVENTS_SOURCE" ]; then
    echo "Skipping event $event_id: current sky map already exists."
    continue
  fi

  event_tmp_dir="$JOB_TMP_DIR/bayestar-$event_id"
  rm -rf "$event_tmp_dir"
  mkdir -p "$event_tmp_dir"
  uv run bayestar-localize-coincs "$eventfile" -o "$event_tmp_dir" --f-low "$F_LOW" --cosmology

  tmp_fits="$event_tmp_dir/$event_id.fits"
  if [ ! -s "$tmp_fits" ]; then
    echo "BAYESTAR did not create $tmp_fits."
    exit 1
  fi
  mv "$tmp_fits" "$fits_file"

  if [ ! "$fits_file" -nt "$EVENTS_SOURCE" ]; then
    echo "Generated sky map is not newer than $EVENTS_SOURCE: $fits_file"
    exit 1
  fi
done

echo "Task $task_id complete."

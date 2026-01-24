#!/bin/bash
set -euo pipefail

PROJECT_DIR=/fred/oz016/bgao_kn/observing-scenarios-simulations
RUNS="${RUNS:-O5a O5b O5c}"

cd "$PROJECT_DIR"

inj_job=$(sbatch --parsable --export=ALL,RUNS="$RUNS" slurm/make_injections.sh)
loc_job=$(sbatch --parsable --dependency=afterok:"$inj_job" --export=ALL,RUNS="$RUNS" slurm/run_bayestar.sh)
sta_job=$(sbatch --parsable --dependency=afterok:"$loc_job" --export=ALL,RUNS="$RUNS" slurm/tabulate_stats.sh)

echo "Submitted jobs:"
echo "  injections: $inj_job"
echo "  bayestar:   $loc_job"
echo "  stats:      $sta_job"

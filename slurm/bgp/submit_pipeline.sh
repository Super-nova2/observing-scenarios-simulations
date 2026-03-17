#!/bin/bash
set -euo pipefail

PROJECT_DIR=/fred/oz016/bgao_kn/observing-scenarios-simulations
RUNS="${RUNS:-O5a}"

cd "$PROJECT_DIR"

for run in $RUNS; do
  case "$run" in
    O5*)
      if [ -d "runs/$run/bgp" ]; then
        echo "Clearing runs/$run/bgp"
        find "runs/$run/bgp" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
      fi
      ;;
  esac
done

inj_job=$(sbatch --parsable --export=ALL,RUNS="$RUNS" slurm/bgp/make_injections.sh)
loc_job=$(sbatch --parsable --dependency=afterok:"$inj_job" --export=ALL,RUNS="$RUNS" slurm/bgp/run_bayestar.sh)
#sta_job=$(sbatch --parsable --dependency=afterok:"$loc_job" --export=ALL,RUNS="$RUNS" slurm/bgp/tabulate_stats.sh)

echo "Submitted jobs:"
echo "  injections: $inj_job"
echo "  bayestar:   $loc_job"
echo "  stats:      submit manually with slurm/bgp/tabulate_stats.sh if needed"

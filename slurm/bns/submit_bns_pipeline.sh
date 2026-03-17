#!/bin/bash
#
# Submit the full BNS simulation pipeline:
#   1. Generate BNS injections (bayestar-inject with bns_astro)
#   2. Generate detected events (bayestar-realize-coincs)
#   3. Create injections.dat and coincs.dat
#   4. Generate skymaps (bayestar-localize-coincs)
#
# Usage:
#   ./slurm/bns/submit_bns_pipeline.sh                       # Default: O5a, 50000 samples
#   RUNS="O5a O5b" ./slurm/bns/submit_bns_pipeline.sh        # Multiple runs
#   BNS_NSAMPLES=100000 ./slurm/bns/submit_bns_pipeline.sh   # More samples
#   BNS_DISTRIBUTION=bns_broad ./slurm/bns/submit_bns_pipeline.sh  # Broad distribution
#

set -euo pipefail

PROJECT_DIR=/fred/oz016/bgao_kn/observing-scenarios-simulations
RUNS="${RUNS:-O5a}"
BNS_NSAMPLES="${BNS_NSAMPLES:-50000}"
BNS_DISTRIBUTION="${BNS_DISTRIBUTION:-bns_astro}"
CLEAR_EXISTING="${CLEAR_EXISTING:-yes}"

cd "$PROJECT_DIR"

echo "=========================================="
echo "BNS Simulation Pipeline"
echo "=========================================="
echo "Runs:         $RUNS"
echo "Samples:      $BNS_NSAMPLES"
echo "Distribution: $BNS_DISTRIBUTION"
echo ""

# Optionally clear existing BNS output
if [ "$CLEAR_EXISTING" = "yes" ]; then
  for run in $RUNS; do
    if [ -d "runs/$run/bns" ]; then
      echo "Clearing runs/$run/bns"
      find "runs/$run/bns" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    fi
  done
fi

# Check PSD files exist
for run in $RUNS; do
  if [ ! -s "runs/$run/psds.xml" ]; then
    echo "ERROR: Missing runs/$run/psds.xml"
    echo "Run this on the login node first:"
    echo "  cd $PROJECT_DIR && uv run make RUNS=\"$RUNS\" psds"
    exit 2
  fi
done

# Submit jobs
inj_job=$(sbatch --parsable \
  --export=ALL,RUNS="$RUNS",BNS_NSAMPLES="$BNS_NSAMPLES",BNS_DISTRIBUTION="$BNS_DISTRIBUTION" \
  slurm/bns/make_bns_injections.sh)

loc_job=$(sbatch --parsable \
  --dependency=afterok:"$inj_job" \
  --export=ALL,RUNS="$RUNS" \
  slurm/bns/run_bns_bayestar.sh)

echo ""
echo "=========================================="
echo "Submitted jobs:"
echo "=========================================="
echo "  BNS injections: $inj_job"
echo "  BNS skymaps:    $loc_job (depends on $inj_job)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/gw-bns-inject-$inj_job.out"
echo ""
echo "Expected output directory: runs/\$RUNS/bns/"
echo "  - injections.xml    (raw injection table)"
echo "  - events.xml.gz     (detected events)"
echo "  - injections.dat    (ASCII table of injections)"
echo "  - coincs.dat        (ASCII table of coincidences)"
echo "  - allsky/           (skymap FITS files)"
echo ""

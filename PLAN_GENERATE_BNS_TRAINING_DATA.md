# Plan: Generate BNS GW Data for Deep Learning Training

## Problem Summary

Current dataset has only **491 positive GW events**, which is far below the recommended 5,000-10,000+ for deep learning. The detection rate using astrophysical distributions is very low (~0.1%) because:

1. **bns_astro**: Mass peaked at 1.33±0.09 Msun, spin limited to ±0.05
2. **bns_broad**: Better coverage (mass 1.0-2.0 Msun, spin ±0.4), but still cosmological distance distribution
3. Most injected events are at large distances → fail SNR threshold

## Key Insight

For ML training, we **don't need** astrophysically accurate distributions. We need:
- Full parameter space coverage
- Balanced representation across all parameter values
- Enough samples for model generalization

## Solution: Multi-Strategy Approach

### Strategy 1: Use `bns_broad` with Distance Limiting

```bash
# Key change: Add --max-distance to limit to detectable events
bayestar-inject \
  --distribution bns_broad \
  --max-distance 300 \  # Limit to 300 Mpc where BNS are detectable
  --nsamples 500000 \
  ...
```

**Expected improvement**: ~10-20x higher detection rate

### Strategy 2: Create Custom Uniform Distribution File

Create `bns_training.h5` with uniform coverage:

```python
# scripts/generate_bns_training_distribution.py
import numpy as np
from astropy.table import Table

# Uniform sampling for maximum parameter coverage
n_samples = 1_000_000

# Mass range: extended to cover full NS range
mass1 = np.random.uniform(1.0, 2.5, n_samples)
mass2 = np.random.uniform(1.0, 2.5, n_samples)

# Ensure mass1 >= mass2
swap = mass1 < mass2
mass1[swap], mass2[swap] = mass2[swap].copy(), mass1[swap].copy()

# Spin range: full NS spin range
spin1z = np.random.uniform(-0.5, 0.5, n_samples)
spin2z = np.random.uniform(-0.5, 0.5, n_samples)

Table({
    "mass1": mass1.astype(np.float32),
    "mass2": mass2.astype(np.float32),
    "spin1z": spin1z.astype(np.float32),
    "spin2z": spin2z.astype(np.float32)
}).write("bns_training.h5", overwrite=True)
```

### Strategy 3: Distance-Stratified Sampling

Generate events in distance bins to ensure coverage at all distances:

| Distance Bin (Mpc) | Expected Detection Rate | Samples to Generate |
|-------------------|------------------------|---------------------|
| 0-100             | ~90%                   | 100,000             |
| 100-200           | ~50%                   | 200,000             |
| 200-400           | ~10%                   | 500,000             |

### Strategy 4: Lower Detection Thresholds (for training only)

Modify `bayestar-realize-coincs` parameters:

```bash
--net-snr-threshold 4   # Lower from 8 to 4
--min-triggers 1        # Allow single-detector events
```

**Warning**: This creates events that wouldn't be "detected" in real observations, but useful for training a model that needs to learn weak signals.

## Recommended Implementation

### Step 1: Generate Custom Distribution File

```bash
cd /fred/oz016/bgao_kn/observing-scenarios-simulations
python scripts/generate_bns_training_distribution.py
```

### Step 2: Create New Slurm Script

File: `slurm/make_bns_training_data.sh`

Key parameters:
- Use `--distribution-samples bns_training.h5`
- Use `--max-distance 400` (adjustable)
- Use `--nsamples 1000000`
- Lower `--net-snr-threshold` to 5 or 6

### Step 3: Run a Single Large Job (No Multi-Seed Batch)

```bash
sbatch --export=ALL,NSAMPLES=1000000,MAX_DIST=400,NET_SNR_THR=5 slurm/make_bns_training_data.sh
```

Note: keep one job so bayestar-inject only traverses bns_training.h5 once.

### Step 4: Combine Outputs (Optional)

Only needed if you intentionally split runs or seeds.

```bash
python scripts/combine_bns_training_data.py --run O5a --seeds 1
```

## Expected Results

| Strategy | Input Samples | Expected Detected | Improvement |
|----------|--------------|-------------------|-------------|
| Current (bns_astro) | 100,000 | ~100 | Baseline |
| bns_broad + max-dist | 500,000 | ~5,000-10,000 | 50-100x |
| Custom uniform | 1,000,000 | ~20,000-50,000 | 200-500x |
| Lower thresholds | 1,000,000 | ~100,000+ | 1000x |

## Parameter Space Comparison

| Parameter | bns_astro | bns_broad | Custom Training |
|-----------|-----------|-----------|-----------------|
| mass1 (Msun) | N(1.33, 0.09) | U[1.0, 2.0] | U[1.0, 2.5] |
| mass2 (Msun) | N(1.33, 0.09) | U[1.0, 2.0] | U[1.0, 2.5] |
| spin1z | U[-0.05, 0.05] | U[-0.4, 0.4] | U[-0.5, 0.5] |
| spin2z | U[-0.05, 0.05] | U[-0.4, 0.4] | U[-0.5, 0.5] |
| Distance | Cosmological | Cosmological | Limited to 400 Mpc |

## Files to Create

1. `scripts/generate_bns_training_distribution.py` - Create custom distribution
2. `slurm/make_bns_training_data.sh` - New slurm job script
3. `scripts/combine_bns_training_data.py` - Merge batches if you split jobs

## Next Steps

1. Review and approve this plan
2. Create the scripts
3. Run initial test with small sample size
4. Scale up to full production run
5. Validate parameter space coverage

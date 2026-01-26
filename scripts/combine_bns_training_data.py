#!/usr/bin/env python3
"""
Combine multiple BNS training data batches into a single dataset.

This script merges the injections.dat and coincs.dat files from multiple
seeds/runs into a combined dataset suitable for ML training.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def load_dat_file(filepath):
    """Load a tab-separated dat file."""
    return pd.read_csv(filepath, sep='\t')


def discover_seeds(run_dir):
    """Auto-detect seed directories under a run directory."""
    seed_re = re.compile(r"^bns_training_seed(\d+)$")
    seeds = []
    if not run_dir.exists():
        return seeds
    for path in run_dir.iterdir():
        if not path.is_dir():
            continue
        match = seed_re.match(path.name)
        if match:
            seeds.append(int(match.group(1)))
    return sorted(set(seeds))


def main():
    parser = argparse.ArgumentParser(
        description="Combine BNS training data from multiple seeds"
    )
    parser.add_argument(
        "-r", "--run",
        default="O5a",
        help="Observing run (default: O5a)"
    )
    parser.add_argument(
        "-s", "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to combine (default: auto-detect under runs/{run}/bns_training_seed*)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: runs/{run}/bns_training_combined)"
    )
    parser.add_argument(
        "--base-dir",
        default="/fred/oz016/bgao_kn/observing-scenarios-simulations",
        help="Base directory for runs"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    run_dir = base_dir / "runs" / args.run
    if args.seeds is None:
        args.seeds = discover_seeds(run_dir)
        if not args.seeds:
            print(f"No bns_training_seed* directories found under {run_dir}")
            return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_dir / "bns_training_combined"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Combining BNS Training Data")
    print("=" * 60)
    print(f"Run: {args.run}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Collect all data
    all_injections = []
    all_coincs = []
    stats = []

    for seed in args.seeds:
        seed_dir = run_dir / f"bns_training_seed{seed}"
        inj_file = seed_dir / "injections.dat"
        coinc_file = seed_dir / "coincs.dat"

        if not inj_file.exists():
            print(f"  Warning: {inj_file} not found, skipping seed {seed}")
            continue

        inj_df = load_dat_file(inj_file)
        coinc_df = load_dat_file(coinc_file)

        # Add seed column for tracking
        inj_df['seed'] = seed
        coinc_df['seed'] = seed

        # Modify IDs to be unique across seeds
        inj_df['simulation_id'] = inj_df['simulation_id'].astype(str) + f"_s{seed}"
        coinc_df['coinc_event_id'] = coinc_df['coinc_event_id'].astype(str) + f"_s{seed}"

        all_injections.append(inj_df)
        all_coincs.append(coinc_df)

        stats.append({
            'seed': seed,
            'n_injections': len(inj_df),
            'n_detected': len(coinc_df)
        })

        print(f"  Seed {seed}: {len(coinc_df):,} detected / {len(inj_df):,} injections")

    if not all_injections:
        print("No data found!")
        return

    # Combine all data
    combined_inj = pd.concat(all_injections, ignore_index=True)
    combined_coinc = pd.concat(all_coincs, ignore_index=True)

    # Save combined files
    combined_inj.to_csv(output_dir / "injections_combined.dat", sep='\t', index=False)
    combined_coinc.to_csv(output_dir / "coincs_combined.dat", sep='\t', index=False)

    # Calculate and print statistics
    print("\n" + "=" * 60)
    print("Combined Dataset Statistics")
    print("=" * 60)
    print(f"Total injections: {len(combined_inj):,}")
    print(f"Total detected:   {len(combined_coinc):,}")
    print(f"Detection rate:   {100 * len(combined_coinc) / len(combined_inj):.2f}%")

    print("\nParameter ranges (detected events only):")
    for col in ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance', 'inclination']:
        if col in combined_inj.columns:
            vals = combined_inj[col]
            print(f"  {col:12s}: [{vals.min():.4f}, {vals.max():.4f}], mean={vals.mean():.4f}")

    # Save statistics
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / "stats.csv", index=False)

    # Create summary for quick reference
    summary = {
        'run': args.run,
        'seeds': args.seeds,
        'n_seeds': len(args.seeds),
        'total_injections': len(combined_inj),
        'total_detected': len(combined_coinc),
        'detection_rate': len(combined_coinc) / len(combined_inj),
        'mass1_range': [float(combined_inj['mass1'].min()), float(combined_inj['mass1'].max())],
        'mass2_range': [float(combined_inj['mass2'].min()), float(combined_inj['mass2'].max())],
        'spin1z_range': [float(combined_inj['spin1z'].min()), float(combined_inj['spin1z'].max())],
        'spin2z_range': [float(combined_inj['spin2z'].min()), float(combined_inj['spin2z'].max())],
        'distance_range': [float(combined_inj['distance'].min()), float(combined_inj['distance'].max())],
    }

    import json
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutput files saved to: {output_dir}")
    print("  - injections_combined.dat")
    print("  - coincs_combined.dat")
    print("  - stats.csv")
    print("  - summary.json")


if __name__ == "__main__":
    main()

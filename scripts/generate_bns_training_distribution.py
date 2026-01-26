#!/usr/bin/env python3
"""
Generate a custom BNS distribution for ML training.

This creates a uniform distribution over the full BNS parameter space,
optimized for training deep learning models rather than astrophysical accuracy.
"""

import argparse
import numpy as np
from astropy.table import Table


def main():
    parser = argparse.ArgumentParser(
        description="Generate BNS training distribution with full parameter coverage"
    )
    parser.add_argument(
        "-o", "--output",
        default="bns_training.h5",
        help="Output file (default: bns_training.h5)"
    )
    parser.add_argument(
        "-n", "--nsamples",
        type=int,
        default=1_000_000,
        help="Number of samples (default: 1,000,000)"
    )
    parser.add_argument(
        "--mass-min",
        type=float,
        default=1.0,
        help="Minimum NS mass in Msun (default: 1.0)"
    )
    parser.add_argument(
        "--mass-max",
        type=float,
        default=2.5,
        help="Maximum NS mass in Msun (default: 2.5)"
    )
    parser.add_argument(
        "--spin-max",
        type=float,
        default=0.5,
        help="Maximum spin magnitude (default: 0.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified sampling for better coverage of parameter space edges"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    n = args.nsamples

    print(f"Generating {n:,} BNS training samples...")
    print(f"  Mass range: [{args.mass_min}, {args.mass_max}] Msun")
    print(f"  Spin range: [-{args.spin_max}, +{args.spin_max}]")

    if args.stratified:
        # Stratified sampling: divide parameter space into grid and sample uniformly from each cell
        # This ensures better coverage of edges and corners
        n_bins = 10
        samples_per_bin = n // (n_bins ** 4) + 1

        mass_edges = np.linspace(args.mass_min, args.mass_max, n_bins + 1)
        spin_edges = np.linspace(-args.spin_max, args.spin_max, n_bins + 1)

        all_mass1, all_mass2, all_spin1z, all_spin2z = [], [], [], []

        for i in range(n_bins):
            for j in range(n_bins):
                for k in range(n_bins):
                    for l in range(n_bins):
                        m1 = np.random.uniform(mass_edges[i], mass_edges[i+1], samples_per_bin)
                        m2 = np.random.uniform(mass_edges[j], mass_edges[j+1], samples_per_bin)
                        s1 = np.random.uniform(spin_edges[k], spin_edges[k+1], samples_per_bin)
                        s2 = np.random.uniform(spin_edges[l], spin_edges[l+1], samples_per_bin)
                        all_mass1.extend(m1)
                        all_mass2.extend(m2)
                        all_spin1z.extend(s1)
                        all_spin2z.extend(s2)

        mass1 = np.array(all_mass1[:n])
        mass2 = np.array(all_mass2[:n])
        spin1z = np.array(all_spin1z[:n])
        spin2z = np.array(all_spin2z[:n])
    else:
        # Simple uniform sampling
        mass1 = np.random.uniform(args.mass_min, args.mass_max, n)
        mass2 = np.random.uniform(args.mass_min, args.mass_max, n)
        spin1z = np.random.uniform(-args.spin_max, args.spin_max, n)
        spin2z = np.random.uniform(-args.spin_max, args.spin_max, n)

    # Ensure mass1 >= mass2 (convention)
    swap = mass1 < mass2
    mass1[swap], mass2[swap] = mass2[swap].copy(), mass1[swap].copy()
    spin1z[swap], spin2z[swap] = spin2z[swap].copy(), spin1z[swap].copy()

    # Create table and save
    table = Table({
        "mass1": mass1.astype(np.float32),
        "mass2": mass2.astype(np.float32),
        "spin1z": spin1z.astype(np.float32),
        "spin2z": spin2z.astype(np.float32)
    })

    table.write(args.output, overwrite=True)

    print(f"\nSaved {len(table):,} samples to {args.output}")
    print(f"\nParameter statistics:")
    print(f"  mass1:  min={mass1.min():.3f}, max={mass1.max():.3f}, mean={mass1.mean():.3f}")
    print(f"  mass2:  min={mass2.min():.3f}, max={mass2.max():.3f}, mean={mass2.mean():.3f}")
    print(f"  spin1z: min={spin1z.min():.3f}, max={spin1z.max():.3f}, mean={spin1z.mean():.3f}")
    print(f"  spin2z: min={spin2z.min():.3f}, max={spin2z.max():.3f}, mean={spin2z.mean():.3f}")


if __name__ == "__main__":
    main()

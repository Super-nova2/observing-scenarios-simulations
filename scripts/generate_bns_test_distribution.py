#!/usr/bin/env python3
"""
Generate an astrophysical BNS test distribution for bayestar-inject.

Masses follow the custom BNS population summarized in
BNS_NSBH_mass_distribution_summary.md:
  - recycled NS: double Gaussian mixture
  - slow/second-born NS: double Gaussian mixture
"""

from __future__ import annotations

import argparse

import numpy as np
from astropy.table import Table

from bgp_test_sampling import load_bgp_mass_grid, sample_bns_masses_from_grid


DEFAULT_NS_MASS_MIN = 1.0
DEFAULT_NS_MASS_MAX = 2.05
DEFAULT_RECYCLED_MASS_MIN = DEFAULT_NS_MASS_MIN
DEFAULT_RECYCLED_MASS_MAX = DEFAULT_NS_MASS_MAX
DEFAULT_SLOW_MASS_MIN = DEFAULT_NS_MASS_MIN
DEFAULT_SLOW_MASS_MAX = DEFAULT_NS_MASS_MAX
DEFAULT_SPIN_MAX = 0.1
DEFAULT_MASS_METHOD = "custom"
DEFAULT_BGP_INPUT = "AllCBC_FullPopBGP.h5"


def _sample_truncated_gaussian_mixture(
    rng: np.random.Generator,
    nsamples: int,
    mass_min: float,
    mass_max: float,
    weights: tuple[float, ...],
    means: tuple[float, ...],
    sigmas: tuple[float, ...],
) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=np.float64)
    means_array = np.asarray(means, dtype=np.float64)
    sigmas_array = np.asarray(sigmas, dtype=np.float64)
    weights_array = weights_array / np.sum(weights_array)

    mass = np.empty(nsamples, dtype=np.float64)
    remaining = np.ones(nsamples, dtype=bool)

    while remaining.any():
        n_draw = int(np.sum(remaining))
        component = rng.choice(len(weights_array), size=n_draw, p=weights_array)
        proposal = rng.normal(means_array[component], sigmas_array[component])
        keep = (proposal >= mass_min) & (proposal <= mass_max)
        remaining_idx = np.flatnonzero(remaining)
        mass[remaining_idx[keep]] = proposal[keep]
        remaining[remaining_idx[keep]] = False

    return mass


def _sample_recycled_mass(
    rng: np.random.Generator,
    nsamples: int,
    mass_min: float,
    mass_max: float,
) -> np.ndarray:
    return _sample_truncated_gaussian_mixture(
        rng,
        nsamples,
        mass_min,
        mass_max,
        weights=(0.68, 0.32),
        means=(1.34, 1.47),
        sigmas=(0.02, 0.15),
    )


def _sample_slow_mass(
    rng: np.random.Generator,
    nsamples: int,
    mass_min: float,
    mass_max: float,
) -> np.ndarray:
    return _sample_truncated_gaussian_mixture(
        rng,
        nsamples,
        mass_min,
        mass_max,
        weights=(0.50, 0.50),
        means=(1.29, 1.80),
        sigmas=(0.09, 0.15),
    )


def generate_distribution(
    nsamples: int = 100_000,
    seed: int = 42,
    recycled_mass_min: float = DEFAULT_RECYCLED_MASS_MIN,
    recycled_mass_max: float = DEFAULT_RECYCLED_MASS_MAX,
    nonrecycled_mass_min: float = DEFAULT_SLOW_MASS_MIN,
    nonrecycled_mass_max: float = DEFAULT_SLOW_MASS_MAX,
    spin_max: float = DEFAULT_SPIN_MAX,
    mass_method: str = DEFAULT_MASS_METHOD,
    bgp_input: str = DEFAULT_BGP_INPUT,
    bgp_edges: np.ndarray | None = None,
    bgp_rates: np.ndarray | None = None,
) -> Table:
    if nsamples <= 0:
        raise ValueError("nsamples must be > 0")
    if recycled_mass_min >= recycled_mass_max:
        raise ValueError("recycled_mass_min must be < recycled_mass_max")
    if nonrecycled_mass_min >= nonrecycled_mass_max:
        raise ValueError("nonrecycled_mass_min must be < nonrecycled_mass_max")
    if spin_max < 0:
        raise ValueError("spin_max must be >= 0")
    if mass_method not in {"bgp", "custom"}:
        raise ValueError("mass_method must be 'bgp' or 'custom'")

    rng = np.random.default_rng(seed)

    if mass_method == "bgp":
        if bgp_edges is None or bgp_rates is None:
            bgp_edges, bgp_rates = load_bgp_mass_grid(bgp_input)
        mass1, mass2 = sample_bns_masses_from_grid(
            bgp_edges,
            bgp_rates,
            nsamples=nsamples,
            rng=rng,
            ns_mass_min=recycled_mass_min,
            ns_mass_max=recycled_mass_max,
        )
    else:
        mass_recycled = _sample_recycled_mass(
            rng,
            nsamples,
            recycled_mass_min,
            recycled_mass_max,
        )
        mass_slow = _sample_slow_mass(
            rng,
            nsamples,
            nonrecycled_mass_min,
            nonrecycled_mass_max,
        )

        mass1 = np.maximum(mass_recycled, mass_slow)
        mass2 = np.minimum(mass_recycled, mass_slow)
    spin1z = rng.uniform(-spin_max, spin_max, nsamples)
    spin2z = rng.uniform(-spin_max, spin_max, nsamples)

    return Table(
        {
            "mass1": mass1.astype(np.float32),
            "mass2": mass2.astype(np.float32),
            "spin1z": spin1z.astype(np.float32),
            "spin2z": spin2z.astype(np.float32),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an astrophysical BNS test distribution for bayestar-inject."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="bns_test.h5",
        help="Output file (default: bns_test.h5)",
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        type=int,
        default=100_000,
        help="Number of samples (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--recycled-mass-min",
        type=float,
        default=DEFAULT_RECYCLED_MASS_MIN,
        help="Minimum recycled NS mass in Msun (default: 1.0)",
    )
    parser.add_argument(
        "--recycled-mass-max",
        type=float,
        default=DEFAULT_RECYCLED_MASS_MAX,
        help="Maximum recycled NS mass in Msun (default: 2.05)",
    )
    parser.add_argument(
        "--spin-max",
        type=float,
        default=DEFAULT_SPIN_MAX,
        help="Maximum aligned spin magnitude (default: 0.1)",
    )
    parser.add_argument(
        "--mass-method",
        choices=("bgp", "custom"),
        default=DEFAULT_MASS_METHOD,
        help="Mass sampling method: custom distribution or bgp rate grid (default: custom)",
    )
    parser.add_argument(
        "--bgp-input",
        default=DEFAULT_BGP_INPUT,
        help=f"BGP PopSummary input file (default: {DEFAULT_BGP_INPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = generate_distribution(
        nsamples=args.nsamples,
        seed=args.seed,
        recycled_mass_min=args.recycled_mass_min,
        recycled_mass_max=args.recycled_mass_max,
        spin_max=args.spin_max,
        mass_method=args.mass_method,
        bgp_input=args.bgp_input,
    )
    table.write(args.output, overwrite=True)

    print(f"Saved {len(table):,} BNS test samples to {args.output}")
    print(f"  mass_method: {args.mass_method}")
    if args.mass_method == "bgp":
        print(f"  bgp_input: {args.bgp_input}")
    for colname in table.colnames:
        values = np.asarray(table[colname])
        print(
            f"  {colname}: min={values.min():.5f}, "
            f"max={values.max():.5f}, mean={values.mean():.5f}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a simple astrophysical NSBH test distribution for bayestar-inject."""

from __future__ import annotations

import argparse

import numpy as np
from astropy.table import Table

from bgp_test_sampling import load_bgp_mass_grid, sample_nsbh_masses_from_grid


DEFAULT_BH_MASS_MIN = 2.05
DEFAULT_BH_MASS_MAX = 10.0
DEFAULT_NS_MASS_MIN = 1.0
DEFAULT_NS_MASS_MAX = 2.05
DEFAULT_BH_SPIN_MAX = 0.99
DEFAULT_NS_SPIN_MAX = 0.1
DEFAULT_BGP_INPUT = "AllCBC_FullPopBGP.h5"


def _sample_truncated_normal(
    rng: np.random.Generator,
    nsamples: int,
    mean: float,
    sigma: float,
    low: float,
    high: float,
) -> np.ndarray:
    values = np.empty(nsamples, dtype=np.float64)
    remaining = np.ones(nsamples, dtype=bool)

    while remaining.any():
        n_draw = int(np.sum(remaining))
        proposal = rng.normal(mean, sigma, n_draw)
        keep = (proposal >= low) & (proposal <= high)
        remaining_idx = np.flatnonzero(remaining)
        values[remaining_idx[keep]] = proposal[keep]
        remaining[remaining_idx[keep]] = False

    return values


def generate_distribution(
    nsamples: int = 100_000,
    seed: int = 42,
    bh_mass_min: float = DEFAULT_BH_MASS_MIN,
    bh_mass_max: float = DEFAULT_BH_MASS_MAX,
    ns_mass_mean: float = 1.35,
    ns_mass_sigma: float = 0.15,
    ns_mass_min: float = DEFAULT_NS_MASS_MIN,
    ns_mass_max: float = DEFAULT_NS_MASS_MAX,
    bh_spin_max: float = DEFAULT_BH_SPIN_MAX,
    ns_spin_max: float = DEFAULT_NS_SPIN_MAX,
    mass_method: str = "bgp",
    bgp_input: str = DEFAULT_BGP_INPUT,
    bgp_edges: np.ndarray | None = None,
    bgp_rates: np.ndarray | None = None,
) -> Table:
    if nsamples <= 0:
        raise ValueError("nsamples must be > 0")
    if bh_mass_min >= bh_mass_max:
        raise ValueError("bh_mass_min must be < bh_mass_max")
    if ns_mass_min >= ns_mass_max:
        raise ValueError("ns_mass_min must be < ns_mass_max")
    if ns_mass_sigma <= 0:
        raise ValueError("ns_mass_sigma must be > 0")
    if bh_spin_max < 0:
        raise ValueError("bh_spin_max must be >= 0")
    if ns_spin_max < 0:
        raise ValueError("ns_spin_max must be >= 0")
    if mass_method not in {"bgp", "custom"}:
        raise ValueError("mass_method must be 'bgp' or 'custom'")

    rng = np.random.default_rng(seed)

    if mass_method == "bgp":
        if bgp_edges is None or bgp_rates is None:
            bgp_edges, bgp_rates = load_bgp_mass_grid(bgp_input)
        mass1, mass2 = sample_nsbh_masses_from_grid(
            bgp_edges,
            bgp_rates,
            nsamples=nsamples,
            rng=rng,
            ns_mass_min=ns_mass_min,
            ns_mass_max=ns_mass_max,
            bh_mass_min=bh_mass_min,
            bh_mass_max=bh_mass_max,
        )
    else:
        mass1 = rng.uniform(bh_mass_min, bh_mass_max, nsamples)
        mass2 = _sample_truncated_normal(
            rng,
            nsamples,
            ns_mass_mean,
            ns_mass_sigma,
            ns_mass_min,
            ns_mass_max,
        )
    spin1z = rng.uniform(-bh_spin_max, bh_spin_max, nsamples)
    spin2z = rng.uniform(-ns_spin_max, ns_spin_max, nsamples)

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
        description="Generate a simple astrophysical NSBH test distribution."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="nsbh_test.h5",
        help="Output file (default: nsbh_test.h5)",
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
        "--bh-mass-min",
        type=float,
        default=DEFAULT_BH_MASS_MIN,
        help="Minimum BH mass in Msun (default: 2.05)",
    )
    parser.add_argument(
        "--bh-mass-max",
        type=float,
        default=DEFAULT_BH_MASS_MAX,
        help="Maximum BH mass in Msun (default: 10.0)",
    )
    parser.add_argument(
        "--ns-mass-mean",
        type=float,
        default=1.35,
        help="Mean NS mass in Msun (default: 1.35)",
    )
    parser.add_argument(
        "--ns-mass-sigma",
        type=float,
        default=0.15,
        help="NS mass Gaussian sigma in Msun (default: 0.15)",
    )
    parser.add_argument(
        "--ns-spin-max",
        type=float,
        default=DEFAULT_NS_SPIN_MAX,
        help="Maximum NS aligned spin magnitude (default: 0.1)",
    )
    parser.add_argument(
        "--bh-spin-max",
        type=float,
        default=DEFAULT_BH_SPIN_MAX,
        help="Maximum BH aligned spin magnitude (default: 0.99)",
    )
    parser.add_argument(
        "--mass-method",
        choices=("bgp", "custom"),
        default="bgp",
        help="Mass sampling method: bgp rate grid or legacy custom distribution (default: bgp)",
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
        bh_mass_min=args.bh_mass_min,
        bh_mass_max=args.bh_mass_max,
        ns_mass_mean=args.ns_mass_mean,
        ns_mass_sigma=args.ns_mass_sigma,
        ns_spin_max=args.ns_spin_max,
        bh_spin_max=args.bh_spin_max,
        mass_method=args.mass_method,
        bgp_input=args.bgp_input,
    )
    table.write(args.output, overwrite=True)

    print(f"Saved {len(table):,} NSBH test samples to {args.output}")
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

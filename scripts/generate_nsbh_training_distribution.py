#!/usr/bin/env python3
"""
Generate NSBH training distribution with KN-aware filtering.

Workflow:
1. Sample NSBH parameters from broad uniform ranges.
2. Compute ejecta masses (same prescription as KN_params_nsbh.ipynb):
   mej_tot, mej_dyn, mej_wind.
3. Keep only KN-positive systems with mej_dyn > threshold and
   mej_wind > threshold.
4. Save the resulting population to HDF5 for bayestar-inject.

Default broad prior:
  - BH mass (mass1): [2.0, 20.0] Msun
  - NS mass (mass2): [1.0, 2.0] Msun
  - BH spin z (spin1z): [-0.99, 0.99]
  - NS spin z (spin2z): [-0.5, 0.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d


def _default_eos_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    project_root = here.parents[1]  # observing-scenarios-simulations/
    repo_root = project_root.parent
    return [
        project_root / "eos.mr",
        repo_root / "ML+GW+KN/dataset/O5_sim_nsbh_aug/eos.mr",
        repo_root / "ML+GW+KN/dataset/O5_sim_nsbh/eos.mr",
    ]


def resolve_eos_file(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"EOS file not found: {p}")
        return p

    for p in _default_eos_candidates():
        if p.exists():
            return p

    candidates = "\n".join(f"  - {p}" for p in _default_eos_candidates())
    raise FileNotFoundError(
        "Could not find an EOS file automatically.\n"
        "Please pass --eos-file explicitly.\n"
        f"Tried:\n{candidates}"
    )


def load_eos_interpolator(eos_file: Path):
    eos = np.loadtxt(eos_file, skiprows=1)
    if eos.ndim != 2 or eos.shape[1] < 2:
        raise ValueError(f"Unexpected EOS format in {eos_file}")

    radius = eos[:, 0]
    mass = eos[:, 1]

    order = np.argsort(mass)
    mass = mass[order]
    radius = radius[order]

    return interp1d(mass, radius, kind="cubic", fill_value="extrapolate")


def sample_uniform(
    rng: np.random.Generator, low: float, high: float, n: int
) -> np.ndarray:
    return rng.uniform(low, high, n)


def sample_stratified_1d(
    rng: np.random.Generator, low: float, high: float, n: int
) -> np.ndarray:
    # Latin-hypercube-like 1D stratification for better marginal coverage.
    u = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(u)
    return low + (high - low) * u


def draw_candidates(args, rng: np.random.Generator, n_draw: int):
    sample_fn = sample_stratified_1d if args.stratified else sample_uniform

    mass1 = sample_fn(rng, args.bh_mass_min, args.bh_mass_max, n_draw)
    mass2 = sample_fn(rng, args.ns_mass_min, args.ns_mass_max, n_draw)
    spin1z = sample_fn(rng, -args.bh_spin_max, args.bh_spin_max, n_draw)
    spin2z = sample_fn(rng, -args.ns_spin_max, args.ns_spin_max, n_draw)

    swap = mass1 < mass2
    if swap.any():
        mass1[swap], mass2[swap] = mass2[swap].copy(), mass1[swap].copy()
        spin1z[swap], spin2z[swap] = spin2z[swap].copy(), spin1z[swap].copy()

    return mass1, mass2, spin1z, spin2z


def compute_compactness(mass: np.ndarray, radius_km: np.ndarray) -> np.ndarray:
    g_const = 6.67430e-11  # m^3 kg^-1 s^-2
    c_light = 299792458.0  # m / s
    solar_mass = 1.98855e30  # kg
    return (g_const * mass * solar_mass) / (c_light**2 * radius_km * 1000.0)


def compute_r_isco(spinz_bh: np.ndarray) -> np.ndarray:
    z1 = 1 + (1 - spinz_bh**2) ** (1 / 3) * (
        (1 + spinz_bh) ** (1 / 3) + (1 - spinz_bh) ** (1 / 3)
    )
    z2 = np.sqrt(3 * spinz_bh**2 + z1**2)
    return 3 + z2 - np.sign(spinz_bh) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def total_ejecta_mass(
    mass_ns: np.ndarray,
    mass_bh: np.ndarray,
    compactness_ns: np.ndarray,
    spinz_bh: np.ndarray,
) -> np.ndarray:
    # Same constants as KN_params_nsbh.ipynb.
    a = 0.406
    b = 0.139
    gamma = 0.255
    delta = 1.761

    q = mass_ns / mass_bh
    eta = (1 + 1 / q) ** (-2) * q ** (-1)
    r_isco = compute_r_isco(spinz_bh)
    mass_b_ns = mass_ns * (1 + 0.6 * compactness_ns / (1 - 0.5 * compactness_ns))

    term = a * (1 - 2 * compactness_ns) / eta ** (1 / 3) - b * r_isco * compactness_ns / eta + gamma
    return mass_b_ns * np.where(term > 0, term, 0.0) ** delta


def compute_dynamical_ejecta_mass(
    mass_ns: np.ndarray,
    mass_bh: np.ndarray,
    spinz_bh: np.ndarray,
    compactness_ns: np.ndarray,
) -> np.ndarray:
    # Same constants as KN_params_nsbh.ipynb.
    a1 = 0.007116
    a2 = 0.001436
    a4 = -0.02762
    n1 = 0.8636
    n2 = 1.6840

    r_isco = compute_r_isco(spinz_bh)
    q = mass_ns / mass_bh
    mass_b_ns = mass_ns * (1 + 0.6 * compactness_ns / (1 - 0.5 * compactness_ns))

    mej_dyn = mass_b_ns * (
        a1 * q ** (-n1) * (1 - 2 * compactness_ns) / compactness_ns
        - a2 * q ** (-n2) * r_isco
        + a4
    )
    return mej_dyn


def compute_wind_ejecta_mass(
    mej_tot: np.ndarray,
    mej_dyn: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    # Follow KN_params_nsbh.ipynb logic using a single random scale factor.
    mass_disc = np.where(mej_tot - mej_dyn > 0, mej_tot - mej_dyn, 0.0)
    xi = 0.18 + 0.11 / (1 + np.e ** (1.5 * (1 / q - 3)))
    mej_th = xi * mass_disc
    mej_mag = rng.uniform(0.1, 1.0) * mej_th
    return mej_th + mej_mag


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate NSBH population from broad priors, then keep KN-positive systems "
            "with mej_dyn > threshold and mej_wind > threshold."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        default="nsbh_training.h5",
        help="Output file (default: nsbh_training.h5)",
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        type=int,
        default=1_000_000,
        help="Number of KN-positive samples to keep (default: 1,000,000)",
    )
    parser.add_argument(
        "--bh-mass-min",
        type=float,
        default=2.0,
        help="Minimum BH mass in Msun (default: 2.0)",
    )
    parser.add_argument(
        "--bh-mass-max",
        type=float,
        default=20.0,
        help="Maximum BH mass in Msun (default: 20.0)",
    )
    parser.add_argument(
        "--ns-mass-min",
        type=float,
        default=1.0,
        help="Minimum NS mass in Msun (default: 1.0)",
    )
    parser.add_argument(
        "--ns-mass-max",
        type=float,
        default=2.0,
        help="Maximum NS mass in Msun (default: 2.0)",
    )
    parser.add_argument(
        "--bh-spin-max",
        type=float,
        default=0.99,
        help="Maximum BH spin magnitude (default: 0.99)",
    )
    parser.add_argument(
        "--ns-spin-max",
        type=float,
        default=0.5,
        help="Maximum NS spin magnitude (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help=(
            "Use 1D stratified sampling (Latin-hypercube-like) "
            "instead of pure uniform random."
        ),
    )
    parser.add_argument(
        "--eos-file",
        type=str,
        default=None,
        help="Path to EOS file (radius[m?km], mass[Msun]) used for compactness.",
    )
    parser.add_argument(
        "--mej-threshold",
        type=float,
        default=None,
        help=(
            "Deprecated alias to set both --mej-dyn-threshold and "
            "--mej-wind-threshold to the same value."
        ),
    )
    parser.add_argument(
        "--mej-dyn-threshold",
        type=float,
        default=0.0,
        help="KN-positive threshold on dynamical ejecta mass (default: 0.0).",
    )
    parser.add_argument(
        "--mej-wind-threshold",
        type=float,
        default=0.0,
        help="KN-positive threshold on wind ejecta mass (default: 0.0).",
    )
    parser.add_argument(
        "--proposal-chunk-size",
        type=int,
        default=500_000,
        help=(
            "Candidate samples proposed per rejection-sampling round "
            "(default: 500000)."
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=200,
        help="Safety cap on rejection-sampling rounds (default: 200).",
    )
    parser.add_argument(
        "--include-mej-tot",
        action="store_true",
        help=(
            "Store ejecta columns (mej_tot, mej_dyn, mej_wind) in output file "
            "in addition to mass/spin columns."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if args.nsamples <= 0:
        raise ValueError("--nsamples must be > 0")
    if args.proposal_chunk_size <= 0:
        raise ValueError("--proposal-chunk-size must be > 0")
    if args.max_rounds <= 0:
        raise ValueError("--max-rounds must be > 0")
    if args.bh_mass_min >= args.bh_mass_max:
        raise ValueError("Require --bh-mass-min < --bh-mass-max")
    if args.ns_mass_min >= args.ns_mass_max:
        raise ValueError("Require --ns-mass-min < --ns-mass-max")
    if args.bh_spin_max <= 0:
        raise ValueError("--bh-spin-max must be > 0")
    if args.ns_spin_max <= 0:
        raise ValueError("--ns-spin-max must be > 0")
    if args.mej_threshold is not None and args.mej_threshold < 0:
        raise ValueError("--mej-threshold must be >= 0")
    if args.mej_dyn_threshold < 0:
        raise ValueError("--mej-dyn-threshold must be >= 0")
    if args.mej_wind_threshold < 0:
        raise ValueError("--mej-wind-threshold must be >= 0")


def main():
    args = parse_args()
    validate_args(args)

    if args.mej_threshold is not None:
        args.mej_dyn_threshold = args.mej_threshold
        args.mej_wind_threshold = args.mej_threshold

    rng = np.random.default_rng(args.seed)
    eos_file = resolve_eos_file(args.eos_file)
    radius_interp = load_eos_interpolator(eos_file)

    target_n = args.nsamples
    accepted_mass1 = []
    accepted_mass2 = []
    accepted_spin1z = []
    accepted_spin2z = []
    accepted_mej_tot = []
    accepted_mej_dyn = []
    accepted_mej_wind = []

    n_keep = 0
    n_proposed = 0
    round_idx = 0

    print(f"Generating KN-positive NSBH population: target={target_n:,}")
    print(f"  BH mass range: [{args.bh_mass_min}, {args.bh_mass_max}] Msun")
    print(f"  NS mass range: [{args.ns_mass_min}, {args.ns_mass_max}] Msun")
    print(f"  BH spin range: [-{args.bh_spin_max}, +{args.bh_spin_max}]")
    print(f"  NS spin range: [-{args.ns_spin_max}, +{args.ns_spin_max}]")
    print(f"  mej_dyn threshold: {args.mej_dyn_threshold}")
    print(f"  mej_wind threshold: {args.mej_wind_threshold}")
    print(f"  EOS file: {eos_file}")
    print(f"  Sampling mode: {'stratified' if args.stratified else 'uniform'}")

    while n_keep < target_n:
        round_idx += 1
        if round_idx > args.max_rounds:
            raise RuntimeError(
                "Reached --max-rounds before collecting enough KN-positive samples. "
                "Try larger --proposal-chunk-size, wider parameter ranges, or lower "
                "--mej-dyn-threshold/--mej-wind-threshold."
            )

        n_draw = args.proposal_chunk_size
        mass1, mass2, spin1z, spin2z = draw_candidates(args, rng, n_draw)

        radius2 = radius_interp(mass2)
        compactness2 = compute_compactness(mass2, radius2)
        mej_tot = total_ejecta_mass(mass2, mass1, compactness2, spin1z)
        mej_dyn_raw = compute_dynamical_ejecta_mass(mass2, mass1, spin1z, compactness2)
        mej_dyn = np.where(mej_dyn_raw > 0, mej_dyn_raw, 0.0)
        q = mass2 / mass1
        mej_wind = compute_wind_ejecta_mass(mej_tot, mej_dyn, q, rng)

        keep_mask = (
            (mej_dyn > args.mej_dyn_threshold)
            & (mej_wind > args.mej_wind_threshold)
        )
        n_pos = int(np.sum(keep_mask))

        n_proposed += n_draw
        if n_pos > 0:
            remaining = target_n - n_keep
            idx = np.flatnonzero(keep_mask)[:remaining]

            accepted_mass1.append(mass1[idx])
            accepted_mass2.append(mass2[idx])
            accepted_spin1z.append(spin1z[idx])
            accepted_spin2z.append(spin2z[idx])
            accepted_mej_tot.append(mej_tot[idx])
            accepted_mej_dyn.append(mej_dyn[idx])
            accepted_mej_wind.append(mej_wind[idx])
            n_keep += len(idx)

        round_acceptance = n_pos / n_draw
        overall_acceptance = n_keep / n_proposed
        print(
            f"  Round {round_idx:03d}: proposed={n_draw:,}, "
            f"positive(dyn+wind)={n_pos:,} ({round_acceptance:.3%}), "
            f"kept={n_keep:,}/{target_n:,} ({overall_acceptance:.3%} overall)"
        )

    mass1 = np.concatenate(accepted_mass1).astype(np.float32, copy=False)
    mass2 = np.concatenate(accepted_mass2).astype(np.float32, copy=False)
    spin1z = np.concatenate(accepted_spin1z).astype(np.float32, copy=False)
    spin2z = np.concatenate(accepted_spin2z).astype(np.float32, copy=False)
    mej_tot = np.concatenate(accepted_mej_tot).astype(np.float32, copy=False)
    mej_dyn = np.concatenate(accepted_mej_dyn).astype(np.float32, copy=False)
    mej_wind = np.concatenate(accepted_mej_wind).astype(np.float32, copy=False)

    payload = {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": spin1z,
        "spin2z": spin2z,
    }
    if args.include_mej_tot:
        payload["mej_tot"] = mej_tot
        payload["mej_dyn"] = mej_dyn
        payload["mej_wind"] = mej_wind

    table = Table(payload)
    table.write(args.output, overwrite=True)

    print(f"\nSaved {len(table):,} KN-positive samples to {args.output}")
    print("\nParameter statistics:")
    print(f"  mass1 (BH): min={mass1.min():.3f}, max={mass1.max():.3f}, mean={mass1.mean():.3f}")
    print(f"  mass2 (NS): min={mass2.min():.3f}, max={mass2.max():.3f}, mean={mass2.mean():.3f}")
    print(f"  spin1z (BH): min={spin1z.min():.3f}, max={spin1z.max():.3f}, mean={spin1z.mean():.3f}")
    print(f"  spin2z (NS): min={spin2z.min():.3f}, max={spin2z.max():.3f}, mean={spin2z.mean():.3f}")
    print(f"  mej_tot:     min={mej_tot.min():.3e}, max={mej_tot.max():.3e}, mean={mej_tot.mean():.3e}")
    print(f"  mej_dyn:     min={mej_dyn.min():.3e}, max={mej_dyn.max():.3e}, mean={mej_dyn.mean():.3e}")
    print(f"  mej_wind:    min={mej_wind.min():.3e}, max={mej_wind.max():.3e}, mean={mej_wind.mean():.3e}")


if __name__ == "__main__":
    main()

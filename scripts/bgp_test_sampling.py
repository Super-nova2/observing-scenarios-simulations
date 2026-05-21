"""BGP-like mass sampling helpers for BNS/NSBH test distributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ETA_MIN = 0.01


def load_bgp_mass_grid(input_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the BGP primary/secondary mass rate grid from a PopSummary file."""
    from popsummary.popresult import PopulationResult

    popresult = PopulationResult(str(input_path))
    (mass1_edges, mass2_edges), rates = popresult.get_rates_on_grids(
        "ppd_primary_and_secondary_mass"
    )
    np.testing.assert_array_equal(mass1_edges, mass2_edges)
    return np.asarray(mass1_edges, dtype=float), np.asarray(rates, dtype=float)


def sample_bns_masses_from_grid(
    edges: np.ndarray,
    rates: np.ndarray,
    *,
    nsamples: int,
    rng: np.random.Generator,
    ns_mass_min: float = 1.0,
    ns_mass_max: float = 2.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample BNS masses from the BGP rate grid conditioned on the NS range."""
    mass1, mass2 = _sample_masses_from_grid(
        edges,
        rates,
        components=(((ns_mass_min, ns_mass_max), (ns_mass_min, ns_mass_max)),),
        nsamples=nsamples,
        rng=rng,
        label="BNS",
    )
    swap = mass1 < mass2
    mass1, mass2 = np.where(swap, mass2, mass1), np.where(swap, mass1, mass2)
    _require_eta_supported(mass1, mass2, "BNS")
    return mass1, mass2


def sample_nsbh_masses_from_grid(
    edges: np.ndarray,
    rates: np.ndarray,
    *,
    nsamples: int,
    rng: np.random.Generator,
    ns_mass_min: float = 1.0,
    ns_mass_max: float = 2.05,
    bh_mass_min: float = 2.05,
    bh_mass_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample NSBH masses from the BGP rate grid with mass1 fixed as the BH."""
    mass_a, mass_b = _sample_masses_from_grid(
        edges,
        rates,
        components=(
            ((bh_mass_min, bh_mass_max), (ns_mass_min, ns_mass_max)),
            ((ns_mass_min, ns_mass_max), (bh_mass_min, bh_mass_max)),
        ),
        nsamples=nsamples,
        rng=rng,
        label="NSBH",
    )
    mass1 = np.where(mass_a >= bh_mass_min, mass_a, mass_b)
    mass2 = np.where(mass_a >= bh_mass_min, mass_b, mass_a)
    _require_eta_supported(mass1, mass2, "NSBH")
    return mass1, mass2


def _sample_masses_from_grid(
    edges: np.ndarray,
    rates: np.ndarray,
    *,
    components: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    nsamples: int,
    rng: np.random.Generator,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if nsamples <= 0:
        raise ValueError("nsamples must be > 0")

    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges must be a one-dimensional array with at least two values")
    if np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be positive and strictly increasing")

    n_cells = len(edges) - 1
    rate_cells = _rate_cells(rates, n_cells)
    cell_count = n_cells * n_cells

    component_bounds = []
    weights = []
    for first_range, second_range in components:
        lo1, hi1, frac1 = _clipped_log_bin_bounds(edges, *first_range)
        lo2, hi2, frac2 = _clipped_log_bin_bounds(edges, *second_range)
        component_bounds.append((lo1, hi1, lo2, hi2))
        weights.append((rate_cells * frac1[:, None] * frac2[None, :]).ravel())

    flat_weights = np.concatenate(weights)
    total_weight = flat_weights.sum()
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError(f"No BGP rate support for {label} in the requested mass range")

    picks = rng.choice(flat_weights.size, nsamples, replace=True, p=flat_weights / total_weight)
    component_index = picks // cell_count
    local_index = picks % cell_count
    i, j = np.unravel_index(local_index, (n_cells, n_cells))

    mass1 = np.empty(nsamples, dtype=float)
    mass2 = np.empty(nsamples, dtype=float)
    for idx, (lo1, hi1, lo2, hi2) in enumerate(component_bounds):
        selected = component_index == idx
        if not np.any(selected):
            continue
        ii = i[selected]
        jj = j[selected]
        mass1[selected] = np.exp(
            rng.uniform(np.log(lo1[ii]), np.log(hi1[ii]), np.sum(selected))
        )
        mass2[selected] = np.exp(
            rng.uniform(np.log(lo2[jj]), np.log(hi2[jj]), np.sum(selected))
        )

    return mass1, mass2


def _rate_cells(rates: np.ndarray, n_cells: int) -> np.ndarray:
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 2:
        raise ValueError("rates must be a two-dimensional array")
    if rates.shape[0] < n_cells or rates.shape[1] < n_cells:
        raise ValueError("rates grid is smaller than the mass-edge cell grid")
    if rates.shape[0] == rates.shape[1]:
        rates = np.where(rates == 0, rates.T, rates)

    cells = rates[:n_cells, :n_cells].copy()
    cells[~np.isfinite(cells)] = 0
    cells[cells < 0] = 0
    return cells


def _clipped_log_bin_bounds(
    edges: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if low <= 0 or low >= high:
        raise ValueError("mass range must be positive with low < high")

    bin_low = edges[:-1]
    bin_high = edges[1:]
    clipped_low = np.maximum(bin_low, low)
    clipped_high = np.minimum(bin_high, high)

    overlap = clipped_high > clipped_low
    fraction = np.zeros_like(bin_low, dtype=float)
    full_width = np.log(bin_high) - np.log(bin_low)
    fraction[overlap] = (
        np.log(clipped_high[overlap]) - np.log(clipped_low[overlap])
    ) / full_width[overlap]
    return clipped_low, clipped_high, fraction


def _require_eta_supported(mass1: np.ndarray, mass2: np.ndarray, label: str) -> None:
    eta = mass1 * mass2 / np.square(mass1 + mass2)
    if np.any(eta < ETA_MIN):
        raise ValueError(f"{label} samples include eta < {ETA_MIN}, unsupported by SEOBNRv4ROM")

#!/usr/bin/env python3
"""
Analyze the potential for optical data augmentation by generating multiple
light curves per GW event based on different sky positions.

Key insight: The same kilonova observed from different positions within the
GW localization region will have different:
1. First detection time (depends on when LSST observes that position)
2. Observing cadence (LSST survey strategy varies across the sky)
3. Filter coverage (different filters at different epochs)
4. Detection significance (background varies with sky position)

This creates a natural, physically-motivated augmentation strategy.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def estimate_augmentation_potential(df, n_positions_per_event=10):
    """
    Estimate how much augmentation is possible based on GW localization areas.

    Larger localization areas = more diverse sky positions = more augmentation potential.
    """

    # Estimate localization area from distance and inclination
    # Rough scaling: area ~ distance^2 * (1 + 0.5*|cos(inc)|)
    # Face-on events have better localization
    distance = df['distance'].values
    inclination = df['inclination'].values

    # Simplified model for 90% credible area (deg^2)
    # Based on O4/O5 BNS detections, typical areas range from 10-1000 deg^2
    snr_proxy = 1 / (distance / 100)  # Higher SNR = better localization
    inclination_factor = 1 + 0.3 * np.abs(np.cos(inclination))

    # Rough area estimate (very simplified)
    area_90 = 100 * (distance / 200) ** 2 / inclination_factor
    area_90 = np.clip(area_90, 5, 2000)  # Realistic bounds

    return area_90


def analyze_augmentation_strategy(df, kn_detection_rate=0.28):
    """Analyze different augmentation strategies."""

    n_total = len(df)
    n_positive = int(n_total * kn_detection_rate)

    print("\n" + "=" * 70)
    print("OPTICAL DATA AUGMENTATION ANALYSIS")
    print("=" * 70)

    print(f"\n1. BASELINE DATASET")
    print("-" * 50)
    print(f"   Total GW events:        {n_total:,}")
    print(f"   Positive (with KN):     {n_positive:,}")
    print(f"   Current positive ratio: {kn_detection_rate:.1%}")

    # Estimate localization areas
    area_90 = estimate_augmentation_potential(df)

    print(f"\n2. GW LOCALIZATION AREAS (estimated)")
    print("-" * 50)
    print(f"   Median 90% area:   {np.median(area_90):.0f} deg²")
    print(f"   Min/Max area:      {area_90.min():.0f} - {area_90.max():.0f} deg²")

    print(f"\n3. AUGMENTATION STRATEGIES")
    print("-" * 70)

    strategies = [
        ("No augmentation (baseline)", 1, "Current state"),
        ("3 positions per event", 3, "Conservative, well-separated positions"),
        ("5 positions per event", 5, "Moderate augmentation"),
        ("10 positions per event", 10, "Aggressive augmentation"),
        ("Variable (area-based)", None, "More positions for larger areas"),
    ]

    results = []

    print(f"\n   {'Strategy':<30} {'Multiplier':<12} {'Effective Pos':<15} {'Total Pairs':<15}")
    print("   " + "-" * 72)

    for name, multiplier, desc in strategies:
        if multiplier is None:
            # Variable strategy: 3-15 positions based on area
            multiplier_array = np.clip(area_90 / 50, 3, 15).astype(int)
            avg_multiplier = multiplier_array.mean()
            effective_pos = int(n_positive * avg_multiplier)
            multiplier_str = f"~{avg_multiplier:.1f}x"
        else:
            effective_pos = n_positive * multiplier
            multiplier_str = f"{multiplier}x"

        # Total pairs for contrastive learning
        total_pairs = effective_pos * n_total

        results.append({
            'strategy': name,
            'multiplier': multiplier_str,
            'effective_positive': effective_pos,
            'total_pairs': total_pairs
        })

        print(f"   {name:<30} {multiplier_str:<12} {effective_pos:<15,} {total_pairs:<15,}")

    return results, area_90


def analyze_observation_diversity(n_positions=10):
    """Analyze the diversity of observations from different sky positions."""

    print(f"\n4. OBSERVATION DIVERSITY FROM DIFFERENT SKY POSITIONS")
    print("-" * 70)

    # Simulated LSST observing characteristics
    print("""
   For each GW event, different sky positions within the localization region
   will have different LSST observation characteristics:

   ┌─────────────────────────────────────────────────────────────────────┐
   │  Sky Position Effect on Light Curve Observations                   │
   ├─────────────────────────────────────────────────────────────────────┤
   │  Parameter              │ Variation Range    │ Impact on LC        │
   ├─────────────────────────┼────────────────────┼─────────────────────┤
   │  First detection Δt     │ 0 - 3 days         │ Miss early peak?    │
   │  Cadence gap            │ 1 - 10 days        │ LC sampling quality │
   │  Filter sequence        │ ugrizy varies      │ Color information   │
   │  Limiting magnitude     │ 23.5 - 25.0        │ Detection depth     │
   │  Background level       │ Varies with coords │ Photometric errors  │
   │  Galactic extinction    │ E(B-V) 0.01-0.5    │ Observed colors     │
   └─────────────────────────┴────────────────────┴─────────────────────┘
    """)

    print("   This diversity is VALUABLE for training because:")
    print("   → Model learns to handle varying observation quality")
    print("   → Robustness to different follow-up scenarios")
    print("   → Better generalization to real observations")


def analyze_model_requirements_with_augmentation(n_positive_base, results):
    """Re-analyze model requirements with augmentation."""

    print(f"\n5. MODEL REQUIREMENTS WITH AUGMENTATION")
    print("-" * 70)

    requirements = {
        'simple_cross_attention': 2000,
        'moderate_transformer': 10000,
        'albef_style': 30000,
        'production': 100000
    }

    print(f"\n   {'Strategy':<30} ", end="")
    for model in requirements.keys():
        print(f"{model[:12]:<14}", end="")
    print()
    print("   " + "-" * 86)

    for result in results:
        eff_pos = result['effective_positive']
        print(f"   {result['strategy']:<30} ", end="")
        for model, min_req in requirements.items():
            status = "✓" if eff_pos >= min_req else "✗"
            ratio = eff_pos / min_req
            print(f"{status} ({ratio:.1f}x)      ", end="")
        print()


def generate_recommendations(n_positive_base, area_90):
    """Generate specific recommendations."""

    print(f"\n6. RECOMMENDATIONS")
    print("=" * 70)

    print("""
   RECOMMENDED APPROACH: Variable Position Sampling

   For each positive BNS event:

   1. LOAD GW SKYMAP
      → Read the HEALPix probability map from BAYESTAR output

   2. SAMPLE SKY POSITIONS
      → Sample N positions weighted by GW posterior probability
      → N = max(3, min(15, area_90 / 50))
      → Ensure positions are well-separated (> 1 deg apart)

   3. FOR EACH POSITION, SIMULATE LSST OBSERVATION
      → Query LSST OpSim database for observation schedule
      → Get: observation times, filters, limiting magnitudes
      → Apply: Galactic extinction, sky background

   4. GENERATE KILONOVA LIGHT CURVE
      → Use KN model (e.g., POSSIS, Kasen) with BNS parameters
      → Sample at LSST observation epochs
      → Add realistic photometric noise

   5. CREATE TRAINING PAIRS
      → (GW_skymap, LC_position_1) → positive pair
      → (GW_skymap, LC_position_2) → positive pair
      → ...
      → (GW_skymap, LC_random_other) → negative pair

   IMPLEMENTATION PSEUDOCODE:
   ─────────────────────────────────────────────────────────────────────
   for gw_event in positive_events:
       skymap = load_skymap(gw_event.skymap_path)
       n_positions = calculate_n_positions(skymap.area_90)

       for i in range(n_positions):
           ra, dec = sample_position_from_skymap(skymap)
           lsst_obs = query_lsst_observations(ra, dec, gw_event.time)

           if lsst_obs.n_detections >= min_detections:
               lc = simulate_kilonova_lc(gw_event.params, lsst_obs)
               training_pairs.append((skymap, lc, label=1))

   # Add negative pairs
   for gw_event in all_events:
       random_lc = sample_random_lc(not_matching=gw_event)
       training_pairs.append((gw_event.skymap, random_lc, label=0))
   ─────────────────────────────────────────────────────────────────────
    """)

    # Estimate final numbers
    avg_multiplier = np.clip(area_90 / 50, 3, 15).mean()
    effective_positive = int(n_positive_base * avg_multiplier)

    print(f"\n   EXPECTED OUTCOME:")
    print(f"   ─────────────────────────────────────────────────────")
    print(f"   Base positive samples:     {n_positive_base:,}")
    print(f"   Average augmentation:      {avg_multiplier:.1f}x")
    print(f"   Effective positive pairs:  {effective_positive:,}")
    print(f"   Suitable for:              ", end="")

    if effective_positive >= 30000:
        print("ALBEF-style model ✓")
    elif effective_positive >= 10000:
        print("Moderate transformer ✓")
    else:
        print("Simple cross-attention ✓")


def create_visualization(df, area_90, output_path):
    """Create visualization of augmentation potential."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Localization area distribution
    ax = axes[0, 0]
    ax.hist(area_90, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.median(area_90), color='red', linestyle='--',
               label=f'Median: {np.median(area_90):.0f} deg²')
    ax.set_xlabel('90% Credible Area (deg²)')
    ax.set_ylabel('Count')
    ax.set_title('GW Localization Area Distribution')
    ax.legend()

    # 2. Area vs Distance
    ax = axes[0, 1]
    ax.scatter(df['distance'], area_90, alpha=0.3, s=5)
    ax.set_xlabel('Distance (Mpc)')
    ax.set_ylabel('90% Credible Area (deg²)')
    ax.set_title('Localization Area vs Distance')
    ax.set_yscale('log')

    # 3. Augmentation multiplier distribution
    ax = axes[1, 0]
    multipliers = np.clip(area_90 / 50, 3, 15).astype(int)
    unique, counts = np.unique(multipliers, return_counts=True)
    ax.bar(unique, counts, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Positions per Event')
    ax.set_ylabel('Number of Events')
    ax.set_title('Variable Augmentation Multiplier Distribution')

    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')

    n_positive_base = int(len(df) * 0.28)
    avg_mult = multipliers.mean()
    effective_pos = int(n_positive_base * avg_mult)

    summary = f"""
    AUGMENTATION SUMMARY
    ═══════════════════════════════════

    Base Dataset:
    ─────────────────────────────────
    Total GW events:      {len(df):,}
    Positive (with KN):   {n_positive_base:,}

    Augmentation Strategy:
    ─────────────────────────────────
    Method:               Variable positions
    Positions per event:  3-15 (area-based)
    Average multiplier:   {avg_mult:.1f}x

    Effective Dataset:
    ─────────────────────────────────
    Effective positives:  {effective_pos:,}
    Augmentation gain:    {effective_pos/n_positive_base:.1f}x

    Model Suitability:
    ─────────────────────────────────
    Simple cross-attn:    {'✓' if effective_pos >= 2000 else '✗'}
    Moderate transformer: {'✓' if effective_pos >= 10000 else '✗'}
    ALBEF-style:          {'✓' if effective_pos >= 30000 else '✗'}
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze optical data augmentation potential"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="/fred/oz016/bgao_kn/observing-scenarios-simulations/runs/O5a/bns_training_seed42/injections.dat",
        help="Path to injections.dat file"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for visualization"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "augmentation_analysis.png"

    print("\n" + "=" * 70)
    print("OPTICAL DATA AUGMENTATION POTENTIAL ANALYSIS")
    print("=" * 70)
    print(f"\nInput file: {input_path}")

    # Load data
    df = pd.read_csv(input_path, sep='\t')
    print(f"Loaded {len(df):,} GW events")

    # Estimate KN detection rate (from previous analysis)
    kn_detection_rate = 0.28
    n_positive_base = int(len(df) * kn_detection_rate)

    # Analyze augmentation potential
    results, area_90 = analyze_augmentation_strategy(df, kn_detection_rate)
    analyze_observation_diversity()
    analyze_model_requirements_with_augmentation(n_positive_base, results)
    generate_recommendations(n_positive_base, area_90)

    # Create visualization
    try:
        create_visualization(df, area_90, output_path)
    except Exception as e:
        print(f"\n   Warning: Could not create visualization: {e}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

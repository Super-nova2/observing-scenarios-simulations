#!/usr/bin/env python3
"""
Analyze GW dataset adequacy for cross-modal deep learning (GW-Kilonova matching).

This script evaluates whether the current GW dataset is sufficient for training
cross-modal models like ALBEF or cross-attention classifiers that learn the
relationship between BNS gravitational wave events and optical kilonova transients.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Model complexity and typical data requirements
CROSSMODAL_REQUIREMENTS = {
    'simple_cross_attention': {
        'min_positive_pairs': 2000,
        'min_total': 5000,
        'description': 'Simple cross-attention classifier'
    },
    'moderate_transformer': {
        'min_positive_pairs': 10000,
        'min_total': 30000,
        'description': 'Moderate transformer with cross-modal attention'
    },
    'albef_style': {
        'min_positive_pairs': 30000,
        'min_total': 100000,
        'description': 'ALBEF-style contrastive + matching model'
    },
    'production_robust': {
        'min_positive_pairs': 100000,
        'min_total': 500000,
        'description': 'Production-grade robust model'
    }
}


def estimate_kn_detection_rate(df):
    """
    Estimate what fraction of GW events will have detectable kilonova.

    Factors affecting KN detectability:
    1. Distance - KN brightness falls as d^2, LSST limiting mag ~24.5
    2. Inclination - face-on events brighter than edge-on
    3. Mass ratio - affects ejecta mass and thus KN brightness
    4. Observing cadence - need to catch the fast transient
    """

    # Rough estimates based on physical models
    # KN typical peak absolute mag: -15 to -17
    # LSST limiting mag: ~24.5 in r-band
    # This gives detection horizon of ~200-400 Mpc for bright KN

    distance = df['distance'].values
    inclination = df['inclination'].values

    # Distance effect: assume KN detectable to ~300 Mpc (optimistic)
    # with probability decreasing with distance
    distance_factor = np.clip(1 - (distance / 400) ** 2, 0, 1)

    # Inclination effect: face-on (inc~0 or ~pi) are brighter
    # Edge-on (inc~pi/2) have fainter blue component
    cos_inc = np.abs(np.cos(inclination))
    inclination_factor = 0.3 + 0.7 * cos_inc  # 30% base + 70% from viewing angle

    # Combined detection probability (simplified model)
    detection_prob = distance_factor * inclination_factor

    # Add some randomness for realistic simulation
    np.random.seed(42)
    detected = np.random.random(len(df)) < detection_prob

    return detected, detection_prob


def analyze_crossmodal_requirements(df, kn_detected):
    """Analyze if dataset meets cross-modal learning requirements."""

    n_total = len(df)
    n_positive = np.sum(kn_detected)
    n_negative = n_total - n_positive
    pos_ratio = n_positive / n_total

    print("\n" + "=" * 70)
    print("CROSS-MODAL LEARNING REQUIREMENTS ANALYSIS")
    print("=" * 70)

    print(f"\n1. DATASET COMPOSITION")
    print("-" * 50)
    print(f"   Total GW events:           {n_total:,}")
    print(f"   With detectable KN (+):    {n_positive:,} ({100*pos_ratio:.1f}%)")
    print(f"   Without detectable KN (-): {n_negative:,} ({100*(1-pos_ratio):.1f}%)")
    print(f"   Class ratio (pos:neg):     1:{n_negative/n_positive:.1f}")

    print(f"\n2. MODEL REQUIREMENTS ASSESSMENT")
    print("-" * 70)
    print(f"   {'Model Type':<35} {'Min Pos':<12} {'Min Total':<12} {'Status'}")
    print("   " + "-" * 67)

    assessments = {}
    for model_type, reqs in CROSSMODAL_REQUIREMENTS.items():
        pos_ok = n_positive >= reqs['min_positive_pairs']
        total_ok = n_total >= reqs['min_total']

        if pos_ok and total_ok:
            status = "✓ ADEQUATE"
        elif pos_ok or total_ok:
            status = "⚠ MARGINAL"
        else:
            status = "✗ INSUFFICIENT"

        assessments[model_type] = {
            'pos_ok': pos_ok,
            'total_ok': total_ok,
            'pos_ratio': n_positive / reqs['min_positive_pairs'],
            'total_ratio': n_total / reqs['min_total'],
            'status': status
        }

        print(f"   {reqs['description']:<35} {reqs['min_positive_pairs']:<12,} {reqs['min_total']:<12,} {status}")

    return assessments, n_positive, n_negative


def analyze_contrastive_learning(df, kn_detected):
    """Analyze suitability for contrastive learning (key component of ALBEF)."""

    print(f"\n3. CONTRASTIVE LEARNING CONSIDERATIONS")
    print("-" * 50)

    n_positive = np.sum(kn_detected)
    n_total = len(df)

    # For contrastive learning, batch size matters
    # Typical batch sizes: 64, 128, 256, 512
    # Need enough negative samples in each batch

    print("   Contrastive learning requires diverse negative samples in each batch")
    print("   Analyzing batch composition possibilities:")
    print()

    for batch_size in [64, 128, 256, 512]:
        # Expected positive samples per batch
        pos_ratio = n_positive / n_total
        expected_pos = batch_size * pos_ratio
        expected_neg = batch_size - expected_pos

        # For good contrastive learning, want at least 10+ negatives per positive
        neg_per_pos = expected_neg / max(expected_pos, 1)

        quality = "Good" if neg_per_pos >= 10 else "Acceptable" if neg_per_pos >= 5 else "Poor"

        print(f"   Batch size {batch_size}: ~{expected_pos:.0f} pos, ~{expected_neg:.0f} neg "
              f"(ratio 1:{neg_per_pos:.1f}) - {quality}")

    # Number of unique pairs for training
    n_pairs = n_positive * n_total  # Each positive can pair with all samples
    print(f"\n   Total possible training pairs: {n_pairs:,}")
    print(f"   Positive-positive pairs: {n_positive * (n_positive-1) // 2:,}")
    print(f"   Positive-negative pairs: {n_positive * (n_total - n_positive):,}")


def analyze_parameter_diversity(df, kn_detected):
    """Check if positive samples cover diverse parameter space."""

    print(f"\n4. PARAMETER DIVERSITY IN POSITIVE SAMPLES")
    print("-" * 50)

    pos_df = df[kn_detected]

    params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance', 'inclination']

    print(f"   {'Parameter':<15} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10}")
    print("   " + "-" * 55)

    for param in params:
        if param not in pos_df.columns:
            continue
        data = pos_df[param]
        print(f"   {param:<15} {data.min():<10.3f} {data.max():<10.3f} "
              f"{data.mean():<10.3f} {data.std():<10.3f}")

    # Check 2D coverage
    print(f"\n   2D Parameter Coverage (positive samples):")
    n_bins = 10
    for p1, p2 in [('mass1', 'mass2'), ('distance', 'inclination')]:
        if p1 in pos_df.columns and p2 in pos_df.columns:
            H, _, _ = np.histogram2d(pos_df[p1], pos_df[p2], bins=n_bins)
            coverage = np.sum(H > 0) / (n_bins * n_bins) * 100
            print(f"   - {p1}-{p2}: {coverage:.1f}% bins covered")


def generate_recommendations(assessments, n_positive, n_negative, n_total):
    """Generate recommendations for cross-modal learning."""

    print(f"\n5. RECOMMENDATIONS FOR ALBEF/CROSS-ATTENTION MODEL")
    print("=" * 70)

    # Determine best suited model
    best_model = None
    for model_type, assessment in assessments.items():
        if assessment['status'] == "✓ ADEQUATE":
            best_model = model_type
            break

    if best_model:
        print(f"\n   ✓ Dataset is ADEQUATE for: {CROSSMODAL_REQUIREMENTS[best_model]['description']}")
    else:
        print(f"\n   ⚠ Dataset may be MARGINAL for complex cross-modal models")

    print("\n   Specific Recommendations:")
    print("   " + "-" * 50)

    recommendations = []

    # Sample size recommendations
    if n_positive < 10000:
        recommendations.append(
            f"   1. INCREASE POSITIVE SAMPLES: Current {n_positive:,} positive samples "
            f"may limit model capacity.\n"
            f"      → Generate more GW events with shorter distances (higher KN detection rate)\n"
            f"      → Target: 20,000+ positive pairs for moderate transformer"
        )

    # Class imbalance
    imbalance = n_negative / n_positive
    if imbalance > 5:
        recommendations.append(
            f"   2. ADDRESS CLASS IMBALANCE: {imbalance:.1f}:1 negative:positive ratio\n"
            f"      → Use focal loss or class weights during training\n"
            f"      → Consider oversampling positive pairs\n"
            f"      → ALBEF's ITM loss naturally handles this via hard negative mining"
        )

    # Architecture suggestions
    recommendations.append(
        f"   3. ARCHITECTURE SUGGESTIONS based on data size:\n"
        f"      → Start with simple cross-attention classifier (fewer parameters)\n"
        f"      → If {n_positive:,}+ positives work well, try ALBEF-style model\n"
        f"      → Use pre-trained encoders if available (transfer learning)"
    )

    # Data augmentation
    recommendations.append(
        f"   4. DATA AUGMENTATION STRATEGIES:\n"
        f"      → GW: Add noise, rotate skymaps, vary SNR\n"
        f"      → Optical: Vary cadence, add photometric noise, missing bands\n"
        f"      → This can effectively 2-5x your dataset"
    )

    # Training strategy
    recommendations.append(
        f"   5. TRAINING STRATEGY:\n"
        f"      → Use curriculum learning: start with easy (close, bright) events\n"
        f"      → Multi-task: classification + contrastive + matching objectives\n"
        f"      → Cross-validation essential with this dataset size"
    )

    for rec in recommendations:
        print(rec)
        print()

    return recommendations


def create_visualization(df, kn_detected, output_path):
    """Create visualization for cross-modal analysis."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    pos_df = df[kn_detected]
    neg_df = df[~kn_detected]

    # 1. Distance distribution by detection status
    ax = axes[0, 0]
    ax.hist(neg_df['distance'], bins=50, alpha=0.5, label=f'No KN ({len(neg_df):,})', color='blue', density=True)
    ax.hist(pos_df['distance'], bins=50, alpha=0.5, label=f'With KN ({len(pos_df):,})', color='red', density=True)
    ax.set_xlabel('Distance (Mpc)')
    ax.set_ylabel('Density')
    ax.set_title('Distance Distribution by KN Detection')
    ax.legend()

    # 2. Inclination distribution
    ax = axes[0, 1]
    ax.hist(np.cos(neg_df['inclination']), bins=50, alpha=0.5, label='No KN', color='blue', density=True)
    ax.hist(np.cos(pos_df['inclination']), bins=50, alpha=0.5, label='With KN', color='red', density=True)
    ax.set_xlabel('cos(inclination)')
    ax.set_ylabel('Density')
    ax.set_title('Inclination Distribution by KN Detection')
    ax.legend()

    # 3. Mass distribution (positive samples)
    ax = axes[0, 2]
    ax.scatter(pos_df['mass1'], pos_df['mass2'], alpha=0.3, s=5, c='red', label='With KN')
    ax.scatter(neg_df['mass1'], neg_df['mass2'], alpha=0.1, s=2, c='blue', label='No KN')
    ax.set_xlabel('mass1 (Msun)')
    ax.set_ylabel('mass2 (Msun)')
    ax.set_title('Mass Distribution')
    ax.legend()

    # 4. Detection probability vs distance
    ax = axes[1, 0]
    distance_bins = np.linspace(0, df['distance'].max(), 20)
    detection_rate = []
    bin_centers = []
    for i in range(len(distance_bins) - 1):
        mask = (df['distance'] >= distance_bins[i]) & (df['distance'] < distance_bins[i+1])
        if mask.sum() > 0:
            detection_rate.append(kn_detected[mask].mean())
            bin_centers.append((distance_bins[i] + distance_bins[i+1]) / 2)
    ax.plot(bin_centers, detection_rate, 'o-', color='green')
    ax.set_xlabel('Distance (Mpc)')
    ax.set_ylabel('KN Detection Rate')
    ax.set_title('KN Detection Rate vs Distance')
    ax.set_ylim(0, 1)

    # 5. Class distribution pie chart
    ax = axes[1, 1]
    sizes = [len(pos_df), len(neg_df)]
    labels = [f'With KN\n({len(pos_df):,})', f'No KN\n({len(neg_df):,})']
    colors = ['#ff6b6b', '#4ecdc4']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Dataset Composition')

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""
    CROSS-MODAL DATASET SUMMARY
    ═══════════════════════════════════

    Total GW Events:     {len(df):,}
    With Detectable KN:  {len(pos_df):,} ({100*len(pos_df)/len(df):.1f}%)
    Without KN:          {len(neg_df):,} ({100*len(neg_df)/len(df):.1f}%)

    Model Suitability:
    ─────────────────────────────────
    Simple Cross-Attn:   {'✓' if len(pos_df) >= 2000 else '✗'}
    Moderate Transformer: {'✓' if len(pos_df) >= 10000 else '✗'}
    ALBEF-style:         {'✓' if len(pos_df) >= 30000 else '✗'}

    Key Statistics:
    ─────────────────────────────────
    Mean detection dist: {pos_df['distance'].mean():.0f} Mpc
    Max detection dist:  {pos_df['distance'].max():.0f} Mpc
    Class imbalance:     1:{len(neg_df)/len(pos_df):.1f}
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GW dataset for cross-modal deep learning requirements"
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
        output_path = input_path.parent / "crossmodal_analysis.png"

    print("\n" + "=" * 70)
    print("CROSS-MODAL LEARNING DATASET ANALYSIS")
    print("(GW Events + Kilonova Light Curves)")
    print("=" * 70)
    print(f"\nInput file: {input_path}")

    # Load data
    df = pd.read_csv(input_path, sep='\t')
    n_total = len(df)
    print(f"Loaded {n_total:,} GW events")

    # Estimate KN detection (this simulates your optical pipeline)
    kn_detected, detection_prob = estimate_kn_detection_rate(df)

    # Run analyses
    assessments, n_pos, n_neg = analyze_crossmodal_requirements(df, kn_detected)
    analyze_contrastive_learning(df, kn_detected)
    analyze_parameter_diversity(df, kn_detected)
    recommendations = generate_recommendations(assessments, n_pos, n_neg, n_total)

    # Create visualization
    try:
        create_visualization(df, kn_detected, output_path)
    except Exception as e:
        print(f"\n   Warning: Could not create visualization: {e}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

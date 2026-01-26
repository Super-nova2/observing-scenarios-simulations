#!/usr/bin/env python3
"""
Analyze BNS training injection distribution and coverage.

This script evaluates whether the generated injections are suitable
for training a deep learning model by checking:
1. Sample size adequacy
2. Parameter space coverage
3. Distribution uniformity
4. Potential gaps or biases
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Minimum recommended samples for different model complexities
MIN_SAMPLES = {
    'simple_mlp': 5000,
    'moderate_cnn': 20000,
    'complex_model': 50000,
    'production': 100000
}

# Expected physical ranges for BNS parameters
EXPECTED_RANGES = {
    'mass1': (1.0, 2.5),      # Msun
    'mass2': (1.0, 2.5),      # Msun
    'spin1z': (-0.5, 0.5),
    'spin2z': (-0.5, 0.5),
    'inclination': (0, np.pi),
    'distance': (0, 600),     # Mpc (depends on max_dist setting)
}


def load_injections(filepath):
    """Load injections.dat file."""
    df = pd.read_csv(filepath, sep='\t')
    return df


def analyze_sample_size(n_samples):
    """Evaluate if sample size is adequate."""
    print("\n" + "=" * 70)
    print("1. SAMPLE SIZE ANALYSIS")
    print("=" * 70)
    print(f"\n   Total detected events: {n_samples:,}")
    print("\n   Adequacy Assessment:")

    assessments = []
    for model_type, min_n in MIN_SAMPLES.items():
        status = "✓ ADEQUATE" if n_samples >= min_n else "✗ INSUFFICIENT"
        ratio = n_samples / min_n
        print(f"   - {model_type:20s}: {status} ({ratio:.1f}x minimum)")
        assessments.append((model_type, n_samples >= min_n, ratio))

    return assessments


def analyze_parameter_coverage(df):
    """Analyze parameter space coverage."""
    print("\n" + "=" * 70)
    print("2. PARAMETER SPACE COVERAGE")
    print("=" * 70)

    params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'inclination', 'distance']
    coverage_results = {}

    print(f"\n   {'Parameter':<15} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12} {'Coverage'}")
    print("   " + "-" * 75)

    for param in params:
        if param not in df.columns:
            continue

        data = df[param].values
        data_min, data_max = data.min(), data.max()
        data_mean, data_std = data.mean(), data.std()

        # Calculate coverage relative to expected range
        if param in EXPECTED_RANGES:
            exp_min, exp_max = EXPECTED_RANGES[param]
            exp_range = exp_max - exp_min
            actual_range = data_max - data_min
            coverage = min(actual_range / exp_range, 1.0) * 100

            # Check if we cover the edges
            low_coverage = (data_min - exp_min) / exp_range * 100
            high_coverage = (exp_max - data_max) / exp_range * 100
        else:
            coverage = 100.0
            low_coverage = high_coverage = 0

        coverage_results[param] = {
            'min': data_min, 'max': data_max,
            'mean': data_mean, 'std': data_std,
            'coverage': coverage
        }

        print(f"   {param:<15} {data_min:<12.4f} {data_max:<12.4f} {data_mean:<12.4f} {data_std:<12.4f} {coverage:>5.1f}%")

    return coverage_results


def analyze_uniformity(df):
    """Check if parameters are uniformly distributed (important for training)."""
    print("\n" + "=" * 70)
    print("3. DISTRIBUTION UNIFORMITY ANALYSIS")
    print("=" * 70)
    print("\n   Testing if parameters are uniformly distributed (ideal for training)")
    print("   Using Kolmogorov-Smirnov test against uniform distribution")
    print(f"\n   {'Parameter':<15} {'KS Statistic':<15} {'p-value':<15} {'Assessment'}")
    print("   " + "-" * 60)

    uniform_params = ['spin1z', 'spin2z']  # These should be uniform
    results = {}

    for param in ['mass1', 'mass2', 'spin1z', 'spin2z', 'inclination', 'distance']:
        if param not in df.columns:
            continue

        data = df[param].values
        # Normalize to [0, 1] for uniform test
        normalized = (data - data.min()) / (data.max() - data.min())
        ks_stat, p_value = stats.kstest(normalized, 'uniform')

        # For training, we want relatively uniform coverage
        if p_value > 0.05:
            assessment = "Uniform ✓"
        elif p_value > 0.001:
            assessment = "Slightly biased"
        else:
            assessment = "Non-uniform (expected for physical params)"

        results[param] = {'ks_stat': ks_stat, 'p_value': p_value}
        print(f"   {param:<15} {ks_stat:<15.4f} {p_value:<15.4e} {assessment}")

    return results


def analyze_2d_coverage(df):
    """Analyze 2D parameter space coverage for key parameter pairs."""
    print("\n" + "=" * 70)
    print("4. 2D PARAMETER SPACE COVERAGE")
    print("=" * 70)
    print("\n   Checking for gaps in 2D parameter combinations")

    pairs = [
        ('mass1', 'mass2', 'Mass space'),
        ('spin1z', 'spin2z', 'Spin space'),
        ('distance', 'inclination', 'Distance-Inclination'),
        ('mass1', 'distance', 'Mass-Distance'),
    ]

    results = {}
    n_bins = 20

    for p1, p2, name in pairs:
        if p1 not in df.columns or p2 not in df.columns:
            continue

        H, _, _ = np.histogram2d(df[p1], df[p2], bins=n_bins)
        total_bins = n_bins * n_bins
        empty_bins = np.sum(H == 0)
        sparse_bins = np.sum(H < 5)
        min_count = H.min()
        mean_count = H.mean()

        coverage_pct = (total_bins - empty_bins) / total_bins * 100

        results[(p1, p2)] = {
            'empty_bins': empty_bins,
            'sparse_bins': sparse_bins,
            'coverage': coverage_pct,
            'min_count': min_count,
            'mean_count': mean_count
        }

        status = "✓ Good" if coverage_pct > 90 else "⚠ Some gaps" if coverage_pct > 70 else "✗ Poor"
        print(f"\n   {name}:")
        print(f"     - Coverage: {coverage_pct:.1f}% of bins populated {status}")
        print(f"     - Empty bins: {empty_bins}/{total_bins}")
        print(f"     - Sparse bins (<5 samples): {sparse_bins}/{total_bins}")
        print(f"     - Min/Mean samples per bin: {int(min_count)}/{mean_count:.1f}")

    return results


def generate_recommendations(n_samples, coverage_results, uniformity_results, pairs_results):
    """Generate recommendations based on analysis."""
    print("\n" + "=" * 70)
    print("5. RECOMMENDATIONS FOR DEEP LEARNING TRAINING")
    print("=" * 70)

    issues = []
    recommendations = []

    # Sample size assessment
    if n_samples < MIN_SAMPLES['simple_mlp']:
        issues.append(f"Sample size ({n_samples:,}) too small for any model")
        recommendations.append("Generate more data with larger NSAMPLES")
    elif n_samples < MIN_SAMPLES['moderate_cnn']:
        issues.append(f"Sample size adequate for simple models only")
        recommendations.append("Consider generating more data for CNN/complex models")

    # Coverage assessment
    for param, results in coverage_results.items():
        if results['coverage'] < 80:
            issues.append(f"{param} coverage only {results['coverage']:.1f}%")
            recommendations.append(f"Extend {param} range in distribution file")

    # 2D coverage assessment
    for (p1, p2), results in pairs_results.items():
        if results['coverage'] < 80:
            issues.append(f"{p1}-{p2} 2D coverage only {results['coverage']:.1f}%")

    if issues:
        print("\n   Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\n   Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("\n   ✓ No major issues found!")

    # Overall verdict
    print("\n   " + "-" * 50)
    if n_samples >= MIN_SAMPLES['moderate_cnn'] and all(r['coverage'] > 80 for r in coverage_results.values()):
        print("   VERDICT: Dataset is READY for deep learning training")
        print(f"   Recommended for: CNN, Transformer, or similar architectures")
    elif n_samples >= MIN_SAMPLES['simple_mlp']:
        print("   VERDICT: Dataset is ADEQUATE for initial experiments")
        print("   Consider augmentation or generating more data for production")
    else:
        print("   VERDICT: Dataset NEEDS MORE SAMPLES before training")

    return issues, recommendations


def create_visualization(df, output_path):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Mass distribution (2D)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist2d(df['mass1'], df['mass2'], bins=30, cmap='YlOrRd')
    ax1.set_xlabel('mass1 (Msun)')
    ax1.set_ylabel('mass2 (Msun)')
    ax1.set_title(f'Mass Distribution (n={len(df):,})')
    ax1.plot([1, 2.5], [1, 2.5], 'k--', alpha=0.5, label='m1=m2')

    # 2. Spin distribution (2D)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist2d(df['spin1z'], df['spin2z'], bins=30, cmap='YlOrRd')
    ax2.set_xlabel('spin1z')
    ax2.set_ylabel('spin2z')
    ax2.set_title('Spin Distribution')

    # 3. Distance distribution
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(df['distance'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Distance (Mpc)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distance Distribution')
    ax3.axvline(df['distance'].median(), color='red', linestyle='--', label=f'Median: {df["distance"].median():.0f} Mpc')
    ax3.legend()

    # 4. Inclination distribution
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(np.cos(df['inclination']), bins=50, color='green', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('cos(inclination)')
    ax4.set_ylabel('Count')
    ax4.set_title('Inclination Distribution')

    # 5. Mass vs Distance
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(df['distance'], df['mass1'], alpha=0.1, s=1)
    ax5.set_xlabel('Distance (Mpc)')
    ax5.set_ylabel('mass1 (Msun)')
    ax5.set_title('Mass vs Distance')

    # 6. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = f"""
    DATASET SUMMARY
    ═══════════════════════════════════

    Total Samples: {len(df):,}

    Parameter Ranges:
    ─────────────────────────────────
    mass1:       [{df['mass1'].min():.2f}, {df['mass1'].max():.2f}] Msun
    mass2:       [{df['mass2'].min():.2f}, {df['mass2'].max():.2f}] Msun
    spin1z:      [{df['spin1z'].min():.3f}, {df['spin1z'].max():.3f}]
    spin2z:      [{df['spin2z'].min():.3f}, {df['spin2z'].max():.3f}]
    distance:    [{df['distance'].min():.0f}, {df['distance'].max():.0f}] Mpc
    inclination: [{df['inclination'].min():.2f}, {df['inclination'].max():.2f}] rad

    Training Adequacy:
    ─────────────────────────────────
    Simple MLP:    {'✓' if len(df) >= 5000 else '✗'} ({len(df)/5000:.1f}x min)
    Moderate CNN:  {'✓' if len(df) >= 20000 else '✗'} ({len(df)/20000:.1f}x min)
    Complex Model: {'✓' if len(df) >= 50000 else '✗'} ({len(df)/50000:.1f}x min)
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BNS training injection distribution"
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
        help="Output path for visualization (default: same dir as input)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "training_data_analysis.png"

    print("\n" + "=" * 70)
    print("BNS TRAINING DATA ANALYSIS")
    print("=" * 70)
    print(f"\nInput file: {input_path}")

    # Load data
    df = load_injections(input_path)
    n_samples = len(df)

    # Run analyses
    size_assessment = analyze_sample_size(n_samples)
    coverage_results = analyze_parameter_coverage(df)
    uniformity_results = analyze_uniformity(df)
    pairs_results = analyze_2d_coverage(df)
    issues, recommendations = generate_recommendations(
        n_samples, coverage_results, uniformity_results, pairs_results
    )

    # Create visualization
    try:
        create_visualization(df, output_path)
    except Exception as e:
        print(f"\n   Warning: Could not create visualization: {e}")

    # Output JSON if requested
    if args.json:
        json_output = {
            'n_samples': n_samples,
            'coverage': coverage_results,
            'uniformity': {k: {'ks_stat': v['ks_stat'], 'p_value': v['p_value']}
                          for k, v in uniformity_results.items()},
            'issues': issues,
            'recommendations': recommendations
        }
        json_path = input_path.parent / "analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2, default=float)
        print(f"\n   JSON results saved to: {json_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compile Parallel Validation Results
====================================

This script compiles and analyzes results from parallel validation jobs.
It loads baseline, pre-tuning, and post-tuning results and generates
comparative metrics and visualizations.

Usage:
    python scripts/compile_validation_results.py --input <dir> --output <dir>
"""

import argparse
from datetime import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load_results(input_dir, mode):
    """Load results for a specific mode."""
    filename = f"{mode}_GEM.npy"
    filepath = os.path.join(input_dir, filename)

    if not os.path.exists(filepath):
        print(f"  ⚠️  {mode} results not found: {filepath}")
        return None

    data = np.load(filepath)
    print(f"  ✓ Loaded {mode}: {data.shape}")
    return data


def load_metadata(input_dir, mode):
    """Load metadata for a specific mode."""
    filepath = os.path.join(input_dir, f"{mode}_metadata.json")

    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        metadata = json.load(f)
    return metadata


def calculate_correlations(experimental, predicted):
    """Calculate correlation metrics between experimental and predicted fitness."""
    # Flatten arrays
    exp_flat = experimental.flatten()
    pred_flat = predicted.flatten()

    # Remove NaN and Inf values
    mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
             np.isinf(exp_flat) | np.isinf(pred_flat))
    exp_clean = exp_flat[mask]
    pred_clean = pred_flat[mask]

    # Calculate correlations
    pearson_r, pearson_p = pearsonr(exp_clean, pred_clean)
    spearman_r, spearman_p = spearmanr(exp_clean, pred_clean)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((exp_clean - pred_clean) ** 2))

    # Calculate MAE
    mae = np.mean(np.abs(exp_clean - pred_clean))

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'rmse': rmse,
        'mae': mae,
        'n_points': len(exp_clean)
    }


def create_comparison_plot(experimental, baseline, pretuning, posttuning, output_dir):
    """Create comparison scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    exp_flat = experimental.flatten()

    models = [
        ('Baseline GEM', baseline, axes[0]),
        ('Pre-tuning kinGEMs', pretuning, axes[1]),
        ('Post-tuning kinGEMs', posttuning, axes[2])
    ]

    for model_name, predicted, ax in models:
        if predicted is None:
            ax.text(0.5, 0.5, 'Data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name)
            continue

        pred_flat = predicted.flatten()

        # Remove NaN/Inf
        mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
                np.isinf(exp_flat) | np.isinf(pred_flat))
        exp_clean = exp_flat[mask]
        pred_clean = pred_flat[mask]

        # Calculate correlation
        pearson_r, _ = pearsonr(exp_clean, pred_clean)

        # Create scatter plot
        ax.scatter(exp_clean, pred_clean, alpha=0.3, s=10)
        ax.plot([exp_clean.min(), exp_clean.max()],
               [exp_clean.min(), exp_clean.max()],
               'r--', lw=2, label='Perfect prediction')

        ax.set_xlabel('Experimental Fitness', fontsize=12)
        ax.set_ylabel('Predicted Growth Rate', fontsize=12)
        ax.set_title(f'{model_name}\nPearson r = {pearson_r:.3f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'validation_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comparison plot: {plot_file}")
    plt.close()


def create_metrics_table(metrics_dict, output_dir):
    """Create a table of metrics."""
    df = pd.DataFrame(metrics_dict).T

    # Round values
    df = df.round({
        'pearson_r': 4,
        'pearson_p': 6,
        'spearman_r': 4,
        'spearman_p': 6,
        'rmse': 4,
        'mae': 4
    })

    # Save to CSV
    csv_file = os.path.join(output_dir, 'validation_metrics.csv')
    df.to_csv(csv_file)
    print(f"  ✓ Saved metrics table: {csv_file}")

    # Print to console
    print("\n" + "="*70)
    print("=== Validation Metrics ===")
    print("="*70)
    print(df.to_string())
    print("="*70)

    return df


def create_improvement_analysis(baseline_metrics, pretuning_metrics, posttuning_metrics, output_dir):
    """Analyze improvement from baseline to kinGEMs."""
    improvements = {}

    if baseline_metrics and pretuning_metrics:
        improvements['Baseline → Pre-tuning'] = {
            'Δ Pearson r': pretuning_metrics['pearson_r'] - baseline_metrics['pearson_r'],
            'Δ Spearman r': pretuning_metrics['spearman_r'] - baseline_metrics['spearman_r'],
            'Δ RMSE': pretuning_metrics['rmse'] - baseline_metrics['rmse'],
            '% RMSE change': ((pretuning_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse'] * 100)
        }

    if baseline_metrics and posttuning_metrics:
        improvements['Baseline → Post-tuning'] = {
            'Δ Pearson r': posttuning_metrics['pearson_r'] - baseline_metrics['pearson_r'],
            'Δ Spearman r': posttuning_metrics['spearman_r'] - baseline_metrics['spearman_r'],
            'Δ RMSE': posttuning_metrics['rmse'] - baseline_metrics['rmse'],
            '% RMSE change': ((posttuning_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse'] * 100)
        }

    if pretuning_metrics and posttuning_metrics:
        improvements['Pre-tuning → Post-tuning'] = {
            'Δ Pearson r': posttuning_metrics['pearson_r'] - pretuning_metrics['pearson_r'],
            'Δ Spearman r': posttuning_metrics['spearman_r'] - pretuning_metrics['spearman_r'],
            'Δ RMSE': posttuning_metrics['rmse'] - pretuning_metrics['rmse'],
            '% RMSE change': ((posttuning_metrics['rmse'] - pretuning_metrics['rmse']) / pretuning_metrics['rmse'] * 100)
        }

    if improvements:
        df = pd.DataFrame(improvements).T.round(4)

        csv_file = os.path.join(output_dir, 'validation_improvements.csv')
        df.to_csv(csv_file)
        print(f"  ✓ Saved improvements analysis: {csv_file}")

        print("\n" + "="*70)
        print("=== Improvement Analysis ===")
        print("="*70)
        print(df.to_string())
        print("="*70)
        print("Note: Negative Δ RMSE indicates improvement (lower error)")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default='results/validation_parallel',
                       help='Input directory containing parallel validation results')
    parser.add_argument('--output', default='results/validation_compiled',
                       help='Output directory for compiled results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "="*70)
    print("=== Compiling Validation Results ===")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("="*70)

    # === Step 1: Load all results ===
    print("\n=== Step 1: Loading results ===")

    experimental = np.load(os.path.join(args.input, 'experimental_fitness.npy'))
    print(f"  ✓ Loaded experimental: {experimental.shape}")

    baseline_GEM = load_results(args.input, 'baseline')
    pretuning_GEM = load_results(args.input, 'pre_tuning')
    posttuning_GEM = load_results(args.input, 'post_tuning')

    # Load metadata
    baseline_meta = load_metadata(args.input, 'baseline')
    pretuning_meta = load_metadata(args.input, 'pretuning')
    posttuning_meta = load_metadata(args.input, 'posttuning')

    # === Step 2: Calculate metrics ===
    print("\n=== Step 2: Calculating metrics ===")

    metrics = {}

    if baseline_GEM is not None:
        metrics['Baseline GEM'] = calculate_correlations(experimental, baseline_GEM)
        print("  ✓ Baseline metrics calculated")

    if pretuning_GEM is not None:
        metrics['Pre-tuning kinGEMs'] = calculate_correlations(experimental, pretuning_GEM)
        print("  ✓ Pre-tuning metrics calculated")

    if posttuning_GEM is not None:
        metrics['Post-tuning kinGEMs'] = calculate_correlations(experimental, posttuning_GEM)
        print("  ✓ Post-tuning metrics calculated")

    # === Step 3: Create visualizations ===
    print("\n=== Step 3: Creating visualizations ===")

    create_comparison_plot(experimental, baseline_GEM, pretuning_GEM, posttuning_GEM, args.output)

    # === Step 4: Create metrics table ===
    print("\n=== Step 4: Creating metrics table ===")

    create_metrics_table(metrics, args.output)

    # === Step 5: Improvement analysis ===
    print("\n=== Step 5: Improvement analysis ===")

    create_improvement_analysis(
        metrics.get('Baseline GEM'),
        metrics.get('Pre-tuning kinGEMs'),
        metrics.get('Post-tuning kinGEMs'),
        args.output
    )

    # === Step 6: Save combined metadata ===
    print("\n=== Step 6: Saving metadata ===")

    combined_metadata = {
        'compilation_timestamp': datetime.now().isoformat(),
        'input_directory': args.input,
        'output_directory': args.output,
        'baseline': baseline_meta,
        'pretuning': pretuning_meta,
        'posttuning': posttuning_meta,
        'metrics': metrics
    }

    meta_file = os.path.join(args.output, 'compiled_metadata.json')
    with open(meta_file, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    print(f"  ✓ Saved combined metadata: {meta_file}")

    print("\n" + "="*70)
    print("=== Compilation Complete ===")
    print("="*70)
    print(f"Results saved to: {args.output}")
    print("\nGenerated files:")
    print("  - validation_metrics.csv")
    print("  - validation_improvements.csv")
    print("  - validation_comparison.png")
    print("  - compiled_metadata.json")
    print("="*70)


if __name__ == '__main__':
    main()

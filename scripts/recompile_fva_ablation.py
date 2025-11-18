#!/usr/bin/env python3
"""
Recompile FVA Ablation Results
==============================

This script recompiles FVA ablation study results from existing CSV files,
regenerating plots and summary statistics with updated terminology and
improved visualizations.

Usage:
    python scripts/recompile_fva_ablation.py [--model MODEL] [--run-dir RUN_DIR]
    python scripts/recompile_fva_ablation.py --model iML1515_GEM
    python scripts/recompile_fva_ablation.py --run-dir results/fva_ablation/specific_run/
"""

import argparse
from datetime import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import universal plotting functions
from kinGEMs.plots import create_fva_ablation_dashboard


def find_latest_ablation_run(model_name, results_base_dir):
    """Find the most recent FVA ablation run for a given model."""
    pattern = os.path.join(results_base_dir, f"{model_name}_FVA_ablation_*")
    matching_dirs = glob.glob(pattern)

    if not matching_dirs:
        raise FileNotFoundError(f"No FVA ablation runs found for {model_name}")

    # Sort by directory name (which includes timestamp)
    latest_dir = sorted(matching_dirs)[-1]
    return latest_dir


def calculate_flux_metrics(fva_df):
    """Calculate both Flux Variability (FVi) and Flux Variability Range (FVR).

    FVi = (max - min) / (max + min + ε) - Relative variability (0-1 scale) for reaction i
    FVR = |max - min| - Absolute flux range

    Returns:
        tuple: (fvi, fvr) as pandas Series
    """
    max_flux = fva_df['Max Solutions']
    min_flux = fva_df['Min Solutions']

    # Calculate FVR (Flux Variability Range) - absolute difference
    fvr = (max_flux - min_flux).abs()

    # Calculate FVi (Flux Variability for reaction i) - normalized relative variability
    fvi = (max_flux - min_flux) / (max_flux + min_flux + 1e-10)
    fvi = fvi.replace([np.inf, -np.inf], 0).fillna(0)

    return fvi, fvr


def load_ablation_results(run_dir):
    """Load all FVA ablation results from CSV files."""
    results = {}
    biomass_values = {}

    # Define expected files and their labels
    files_mapping = {
        'level1_baseline.csv': 'Level 1: Baseline GEM',
        'level2_single_enzyme.csv': 'Level 2: Single Enzyme',
        'level3a_isoenzymes.csv': 'Level 3a: + Isoenzymes',
        'level3b_complexes.csv': 'Level 3b: + Complexes',
        'level3c_promiscuous.csv': 'Level 3c: + Promiscuous',
        'level4_all_constraints.csv': 'Level 4: All Constraints',
        'level5_post_tuned.csv': 'Level 5: Post-Tuned'
    }

    for filename, label in files_mapping.items():
        filepath = os.path.join(run_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            results[label] = df
            # Extract biomass value (assuming it's consistent across all reactions)
            biomass_values[label] = df['Solution Biomass'].iloc[0]
            print(f"  ✓ Loaded {label}: {len(df)} reactions, biomass={biomass_values[label]:.6f}")
        else:
            print(f"  ⚠️  Missing: {filename}")

    return results, biomass_values


def plot_fva_ablation_enhanced(fva_results_dict, biomass_dict, output_file, model_name):
    """Create enhanced cumulative FVi distribution plot for all levels."""
    print("\n=== Generating Enhanced FVA Ablation Plot ===")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[3, 1])

    # Define colors for each level
    colors = {
        'Level 1: Baseline GEM': '#1f77b4',
        'Level 2: Single Enzyme': '#ff7f0e',
        'Level 3a: + Isoenzymes': '#2ca02c',
        'Level 3b: + Complexes': '#d62728',
        'Level 3c: + Promiscuous': '#9467bd',
        'Level 4: All Constraints': '#8c564b',
        'Level 5: Post-Tuned': '#e377c2'
    }

    # Main plot: Cumulative distribution of FVi
    all_fvi_values = []
    fvi_stats = {}

    for label, fva_df in fva_results_dict.items():
        fvi, fvr = calculate_flux_metrics(fva_df)
        # Filter out zero values for log plotting
        fvi_nonzero = fvi[fvi > 1e-15]
        if len(fvi_nonzero) == 0:
            fvi_nonzero = np.array([1e-15])

        all_fvi_values.extend(fvi_nonzero)
        fvi_sorted = np.sort(fvi_nonzero)
        cumulative = np.arange(1, len(fvi_sorted) + 1) / len(fvi_sorted)

        ax1.plot(fvi_sorted, cumulative, label=label,
                color=colors.get(label, None), linewidth=2.5)

        # Store stats for bottom plot
        fvi_stats[label] = {
            'mean': fvi.mean(),
            'median': fvi.median(),
            'biomass': biomass_dict[label]
        }

    # Format main plot
    ax1.set_xscale('log')
    ax1.set_xlabel('Flux Variability (FVi)', fontsize=13)
    ax1.set_ylabel('Cumulative Probability', fontsize=13)
    ax1.set_title(f'FVA Ablation Study: {model_name}\nImpact of kinGEMs Constraints on Flux Variability',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.set_ylim(0, 1.0)

    # Set dynamic x-axis range
    if all_fvi_values:
        min_fvi = min(all_fvi_values)
        max_fvi = max(all_fvi_values)
        x_min = max(min_fvi / 10, 1e-10)  # Prevent extremely small values
        x_max = min(max_fvi * 10, 1e3)    # Prevent extremely large values
        ax1.set_xlim(x_min, x_max)
    else:
        ax1.set_xlim(1e-6, 1e3)

    # Bottom plot: Biomass and mean FVi comparison
    levels = list(fvi_stats.keys())
    biomasses = [fvi_stats[level]['biomass'] for level in levels]
    mean_fvis = [fvi_stats[level]['mean'] for level in levels]
    level_colors = [colors[level] for level in levels]

    # Create twin axis for biomass
    ax2_twin = ax2.twinx()

    # Plot mean FVi as bars
    ax2.bar(range(len(levels)), mean_fvis, alpha=0.7, color=level_colors,
                   label='Mean FVi')
    ax2.set_ylabel('Mean FVi', fontsize=12)
    ax2.set_yscale('log')

    # Plot biomass as line
    ax2_twin.plot(range(len(levels)), biomasses, 'ko-', linewidth=2,
                        markersize=6, label='Biomass')
    ax2_twin.set_ylabel('Biomass (1/hr)', fontsize=12)

    # Format bottom plot
    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels([level.replace('Level ', 'L').replace(': ', '\n') for level in levels],
                       fontsize=10, rotation=45, ha='right')
    ax2.set_title('Mean Flux Variability and Biomass by Constraint Level', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add legends
    ax2.legend(loc='upper left', fontsize=10)
    ax2_twin.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved enhanced plot to: {output_file}")
    plt.close()


def generate_summary_statistics_updated(fva_results_dict, biomass_dict, output_file):
    """Generate updated summary statistics table with corrected terminology."""
    print("\n=== Generating Updated Summary Statistics ===")

    summary_data = []

    for label, fva_df in fva_results_dict.items():
        fvi, fvr = calculate_flux_metrics(fva_df)
        biomass = biomass_dict[label]

        # Flag reactions with high flux variability range (potential issues)
        high_range_reactions = fvr >= 1000

        # Additional metrics
        zero_flux_reactions = (fvi == 0).sum()
        high_var_reactions = (fvi > 1).sum()

        summary_data.append({
            'Level': label,
            'Biomass (1/hr)': biomass,
            'N Reactions': len(fva_df),
            'Mean FVi': fvi.mean(),
            'Median FVi': fvi.median(),
            'Std FVi': fvi.std(),
            'Min FVi': fvi.min(),
            'Max FVi': fvi.max(),
            'Q25 FVi': fvi.quantile(0.25),
            'Q75 FVi': fvi.quantile(0.75),
            '% Zero Flux': zero_flux_reactions / len(fvi) * 100,
            '% High Variability (FVi > 1)': high_var_reactions / len(fvi) * 100,
            'Mean FVR': fvr.mean(),
            'Median FVR': fvr.median(),
            'Max FVR': fvr.max(),
            '% High Range (FVR ≥ 1000)': high_range_reactions.sum() / len(fvr) * 100,
            'N High Range Reactions': high_range_reactions.sum(),
            'N Zero Flux': zero_flux_reactions,
            'N High Variability': high_var_reactions
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved updated summary to: {output_file}")

    # Print formatted table
    print("\n" + "="*120)
    print("Updated Summary Statistics (with corrected FVi terminology)")
    print("="*120)

    # Format for better display
    display_df = summary_df.copy()
    numeric_cols = ['Biomass (1/hr)', 'Mean FVi', 'Median FVi', 'Std FVi', 'Min FVi', 'Max FVi']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(6)

    print(display_df.to_string(index=False))
    print("="*120)

    return summary_df


def create_detailed_analysis_plots(fva_results_dict, biomass_dict, output_dir, model_name):
    """Create additional detailed analysis plots."""
    print("\n=== Generating Detailed Analysis Plots ===")

    # 1. Box plot of FVi distributions
    fig, ax = plt.subplots(figsize=(12, 8))

    fvi_data = []
    labels = []
    colors_list = []

    colors = {
        'Level 1: Baseline GEM': '#1f77b4',
        'Level 2: Single Enzyme': '#ff7f0e',
        'Level 3a: + Isoenzymes': '#2ca02c',
        'Level 3b: + Complexes': '#d62728',
        'Level 3c: + Promiscuous': '#9467bd',
        'Level 4: All Constraints': '#8c564b',
        'Level 5: Post-Tuned': '#e377c2'
    }

    for label, fva_df in fva_results_dict.items():
        fvi, fvr = calculate_flux_metrics(fva_df)
        # Use log scale for better visualization, filter zeros
        fvi_nonzero = fvi[fvi > 1e-15]
        if len(fvi_nonzero) > 0:
            fvi_data.append(np.log10(fvi_nonzero))
            labels.append(label.replace('Level ', 'L').replace(': ', '\n'))
            colors_list.append(colors[label])

    bp = ax.boxplot(fvi_data, labels=labels, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('log₁₀(FVi)', fontsize=12)
    ax.set_title(f'{model_name}: Distribution of Flux Variability (FVi) by Constraint Level', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    boxplot_file = os.path.join(output_dir, 'fva_ablation_boxplot.png')
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved box plot: {boxplot_file}")
    plt.close()

    # 2. Biomass progression plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract level numbers for x-axis
    level_nums = []
    biomass_vals = []
    level_labels = []

    for label in fva_results_dict.keys():
        if 'Level 1' in label:
            level_nums.append(1)
        elif 'Level 2' in label:
            level_nums.append(2)
        elif 'Level 3a' in label:
            level_nums.append(3.1)
        elif 'Level 3b' in label:
            level_nums.append(3.2)
        elif 'Level 3c' in label:
            level_nums.append(3.3)
        elif 'Level 4' in label:
            level_nums.append(4)
        elif 'Level 5' in label:
            level_nums.append(5)

        biomass_vals.append(biomass_dict[label])
        level_labels.append(label)

    # Sort by level number
    sorted_data = sorted(zip(level_nums, biomass_vals, level_labels))
    level_nums, biomass_vals, level_labels = zip(*sorted_data)

    ax.plot(level_nums, biomass_vals, 'o-', linewidth=2, markersize=8, color='steelblue')

    # Annotate points
    for x, y, label in zip(level_nums, biomass_vals, level_labels):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=10)

    ax.set_xlabel('Constraint Level', fontsize=12)
    ax.set_ylabel('Biomass (1/hr)', fontsize=12)
    ax.set_title(f'{model_name}: Biomass Production vs Constraint Complexity', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Custom x-axis labels
    ax.set_xticks(level_nums)
    ax.set_xticklabels([label.replace('Level ', 'L').replace(': ', '\n') for label in level_labels],
                      fontsize=10, rotation=45, ha='right')

    plt.tight_layout()

    biomass_file = os.path.join(output_dir, 'biomass_progression.png')
    plt.savefig(biomass_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved biomass progression: {biomass_file}")
    plt.close()


def create_reaction_level_analysis(fva_results_dict, output_dir, model_name):
    """Create reaction-level analysis comparing different constraint levels."""
    print("\n=== Generating Reaction-Level Analysis ===")

    # Get common reactions across all levels
    all_reactions = None
    for label, fva_df in fva_results_dict.items():
        reactions = set(fva_df['Reactions'])
        if all_reactions is None:
            all_reactions = reactions
        else:
            all_reactions = all_reactions.intersection(reactions)

    print(f"  Common reactions across all levels: {len(all_reactions)}")

    # Create comparison DataFrame
    comparison_data = []

    for reaction in all_reactions:
        row_data = {'Reaction': reaction}

        for label, fva_df in fva_results_dict.items():
            reaction_data = fva_df[fva_df['Reactions'] == reaction]
            if not reaction_data.empty:
                fvi, fvr = calculate_flux_metrics(reaction_data)
                row_data[f'{label}_FVi'] = fvi.iloc[0] if not fvi.empty else 0
                row_data[f'{label}_FVR'] = fvr.iloc[0] if not fvr.empty else 0

        comparison_data.append(row_data)

    comparison_df = pd.DataFrame(comparison_data)

    # Save detailed comparison
    comparison_file = os.path.join(output_dir, 'reaction_level_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"  ✓ Saved reaction-level comparison: {comparison_file}")

    # Find reactions with biggest changes
    if 'Level 1: Baseline GEM_FVi' in comparison_df.columns and 'Level 4: All Constraints_FVi' in comparison_df.columns:
        comparison_df['FVi_Change'] = (comparison_df['Level 4: All Constraints_FVi'] -
                                      comparison_df['Level 1: Baseline GEM_FVi'])
        comparison_df['FVi_Ratio'] = (comparison_df['Level 4: All Constraints_FVi'] /
                                     (comparison_df['Level 1: Baseline GEM_FVi'] + 1e-10))

        # Top reactions with increased variability
        top_increases = comparison_df.nlargest(20, 'FVi_Change')
        increases_file = os.path.join(output_dir, 'top_fvi_increases.csv')
        top_increases[['Reaction', 'Level 1: Baseline GEM_FVi', 'Level 4: All Constraints_FVi',
                      'FVi_Change', 'FVi_Ratio']].to_csv(increases_file, index=False)
        print(f"  ✓ Saved top FVi increases: {increases_file}")

        # Top reactions with decreased variability
        top_decreases = comparison_df.nsmallest(20, 'FVi_Change')
        decreases_file = os.path.join(output_dir, 'top_fvi_decreases.csv')
        top_decreases[['Reaction', 'Level 1: Baseline GEM_FVi', 'Level 4: All Constraints_FVi',
                      'FVi_Change', 'FVi_Ratio']].to_csv(decreases_file, index=False)
        print(f"  ✓ Saved top FVi decreases: {decreases_file}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Model name to find latest ablation run for (e.g., iML1515_GEM)')
    parser.add_argument('--run-dir', type=str, default=None,
                       help='Specific ablation run directory to recompile')
    parser.add_argument('--output-suffix', type=str, default='recompiled',
                       help='Suffix for recompiled output files')

    args = parser.parse_args()

    # Determine run directory
    if args.run_dir:
        run_dir = args.run_dir
        model_name = os.path.basename(run_dir).split('_FVA_')[0]
    elif args.model:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        results_base_dir = os.path.join(project_root, 'results', 'fva_ablation')
        run_dir = find_latest_ablation_run(args.model, results_base_dir)
        model_name = args.model
    else:
        print("Error: Must provide either --model or --run-dir")
        sys.exit(1)

    if not os.path.exists(run_dir):
        print(f"Error: Run directory does not exist: {run_dir}")
        sys.exit(1)

    print("\n" + "="*80)
    print("=== Recompiling FVA Ablation Results ===")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Run directory: {run_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load existing results
    print("\n=== Loading Existing Results ===")
    fva_results, biomass_values = load_ablation_results(run_dir)

    if not fva_results:
        print("Error: No FVA results found to recompile")
        sys.exit(1)

    # Generate updated outputs using universal plotting functions
    print("\n=== Regenerating Outputs with Updated Terminology ===")

    # Use the universal dashboard function to create all plots and statistics
    create_fva_ablation_dashboard(
        fva_results, biomass_values, model_name, run_dir,
        prefix=f"fva_ablation_{args.output_suffix}", show=False
    )

    print("\n" + "="*80)
    print("=== Recompilation Complete ===")
    print("="*80)
    print(f"Generated files in: {run_dir}")
    print(f"  - fva_ablation_{args.output_suffix}_cumulative.png (enhanced plot with legend in upper left)")
    print(f"  - fva_ablation_{args.output_suffix}_boxplot.png (distribution analysis)")
    print(f"  - fva_ablation_{args.output_suffix}_biomass_progression.png (biomass vs constraints)")
    print(f"  - fva_ablation_{args.output_suffix}_summary.csv (updated terminology with corrected FVi)")
    print("="*80)


if __name__ == '__main__':
    main()

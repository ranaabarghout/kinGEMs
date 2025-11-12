#!/usr/bin/env python3
"""
FVA Ablation Study - Systematic Evaluation of kinGEMs Constraints
==================================================================

This script performs a comprehensive ablation study to evaluate the impact of
different constraint types on flux variability in kinGEMs models.

Levels:
    1. Baseline GEM (no enzyme constraints)
    2. Single enzyme constraint only
    3a. + Isoenzymes
    3b. + Enzyme complexes
    3c. + Promiscuous enzymes
    4. All constraints (isoenzymes + complexes + promiscuous)
    5. Post-tuned (simulated annealing)

Usage:
    python scripts/run_fva_ablation.py <config_file> [--parallel] [--workers N]
    python scripts/run_fva_ablation.py configs/iML1515_GEM.json --parallel --workers 8

Output:
    - CSV files for each FVA level
    - Combined cumulative probability plot
    - Summary statistics comparison
"""

import argparse
from datetime import datetime
import json
import os
import random
import sys
import warnings

import cobra
from cobra.flux_analysis import flux_variability_analysis as cobra_fva
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.dataset import (
    load_model,
    merge_substrate_sequences,
    prepare_model_data,
    process_kcat_predictions,
)
from kinGEMs.dataset_modelseed import prepare_modelseed_model_data
from kinGEMs.modeling.fva import (
    flux_variability_analysis,
    flux_variability_analysis_parallel,
)
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.modeling.tuning import simulated_annealing

# Suppress warnings
warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def is_modelseed_model(model_name):
    """Detect if model should use ModelSEED functions."""
    return '_genome_' in model_name.lower()


def find_predictions_file(model_name, CPIPred_data_dir):
    """Find CPI-Pred predictions file with flexible naming."""
    import glob

    patterns = [
        f"X06A_kinGEMs_{model_name}_predictions.csv",
        f"*{model_name}*predictions.csv",
        f"*{model_name.replace('_GEM', '')}*predictions.csv",
    ]

    if '_GEM' in model_name:
        base_name = model_name.replace('_GEM', '')
        patterns.append(f"*ecoli_{base_name}*predictions.csv")
        patterns.append(f"*{base_name}*predictions.csv")

    for pattern in patterns:
        search_path = os.path.join(CPIPred_data_dir, pattern)
        matches = glob.glob(search_path)
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No predictions file found for {model_name}")


def calculate_fvi(fva_df):
    """Calculate Flux Variability Index for each reaction."""
    fvi = (fva_df['Max Solutions'] - fva_df['Min Solutions']).abs()
    return fvi


def run_baseline_fva(model):
    """Level 1: Baseline GEM (COBRApy FVA, no enzyme constraints)."""
    print("\n=== Level 1: Baseline GEM (no enzyme constraints) ===")
    fva_results = cobra_fva(model, fraction_of_optimum=0.9)

    df = pd.DataFrame({
        'Reactions': fva_results.index,
        'Min Solutions': fva_results['minimum'],
        'Max Solutions': fva_results['maximum'],
        'Solution Biomass': [model.slim_optimize()] * len(fva_results)
    })

    biomass = model.slim_optimize()
    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(df)}")

    return df, biomass


def run_single_enzyme_fva(model, processed_df, biomass_reaction, enzyme_upper_bound,
                          use_parallel=False, n_workers=None):
    """Level 2: Single enzyme constraint only."""
    print("\n=== Level 2: Single enzyme constraint only ===")
    print("  (multi_enzyme_off=True, isoenzymes_off=True, promiscuous_off=True, complexes_off=True)")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, biomass, _ = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            multi_enzyme_off=True,
            isoenzymes_off=True,
            promiscuous_off=True,
            complexes_off=True
        )
    else:
        fva_df, biomass, _ = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=True,
            isoenzymes_off=True,
            promiscuous_off=True,
            complexes_off=True
        )

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_isoenzymes_fva(model, processed_df, biomass_reaction, enzyme_upper_bound,
                       use_parallel=False, n_workers=None):
    """Level 3a: Single enzyme + isoenzymes."""
    print("\n=== Level 3a: + Isoenzymes ===")
    print("  (isoenzymes_off=False, multi_enzyme_off=True, promiscuous_off=True, complexes_off=True)")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, biomass, _ = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            multi_enzyme_off=True,
            isoenzymes_off=False,
            promiscuous_off=True,
            complexes_off=True
        )
    else:
        fva_df, biomass, _ = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=True,
            isoenzymes_off=False,
            promiscuous_off=True,
            complexes_off=True
        )

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_complexes_fva(model, processed_df, biomass_reaction, enzyme_upper_bound,
                      use_parallel=False, n_workers=None):
    """Level 3b: Single enzyme + complexes."""
    print("\n=== Level 3b: + Enzyme Complexes ===")
    print("  (complexes_off=False, multi_enzyme_off=True, isoenzymes_off=True, promiscuous_off=True)")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, biomass, _ = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            multi_enzyme_off=True,
            isoenzymes_off=True,
            promiscuous_off=True,
            complexes_off=False
        )
    else:
        fva_df, biomass, _ = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=True,
            isoenzymes_off=True,
            promiscuous_off=True,
            complexes_off=False
        )

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_promiscuous_fva(model, processed_df, biomass_reaction, enzyme_upper_bound,
                       use_parallel=False, n_workers=None):
    """Level 3c: Single enzyme + promiscuous enzymes."""
    print("\n=== Level 3c: + Promiscuous Enzymes ===")
    print("  (promiscuous_off=False, multi_enzyme_off=True, isoenzymes_off=True, complexes_off=True)")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, biomass, _ = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            multi_enzyme_off=True,
            isoenzymes_off=True,
            promiscuous_off=False,
            complexes_off=True
        )
    else:
        fva_df, biomass, _ = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=True,
            isoenzymes_off=True,
            promiscuous_off=False,
            complexes_off=True
        )

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_all_constraints_fva(model, processed_df, biomass_reaction, enzyme_upper_bound,
                           use_parallel=False, n_workers=None):
    """Level 4: All constraints (default kinGEMs)."""
    print("\n=== Level 4: All Constraints ===")
    print("  (all constraint types enabled)")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, biomass, _ = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers
        )
    else:
        fva_df, biomass, _ = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound
        )

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_tuned_fva(model, tuned_df, biomass_reaction, enzyme_upper_bound,
                 use_parallel=False, n_workers=None):
    """Level 5: Post-tuned kinGEMs (after simulated annealing)."""
    print("\n=== Level 5: Post-Tuned (Simulated Annealing) ===")
    print("  (all constraints with tuned kcat values)")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, biomass, _ = flux_variability_analysis_parallel(
            model=model,
            processed_df=tuned_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers
        )
    else:
        fva_df, biomass, _ = flux_variability_analysis(
            model=model,
            processed_df=tuned_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound
        )

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def plot_fva_ablation(fva_results_dict, biomass_dict, output_file):
    """Create cumulative FVI distribution plot for all levels."""
    print("\n=== Generating FVA Ablation Plot ===")

    fig, ax = plt.subplots(figsize=(12, 8))

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

    # Plot each level
    for label, fva_df in fva_results_dict.items():
        fvi = calculate_fvi(fva_df)
        fvi_sorted = np.sort(fvi)
        cumulative = np.arange(1, len(fvi_sorted) + 1) / len(fvi_sorted)

        ax.plot(fvi_sorted, cumulative, label=label,
                color=colors.get(label, None), linewidth=2)

    # Add biomass reference lines
    biomass_y = 0.9  # Position for biomass reference
    for label, biomass in biomass_dict.items():
        color = colors.get(label, 'gray')
        ax.axhline(y=biomass_y, color=color, linestyle='--', alpha=0.3, linewidth=1)

    # Create second y-axis for biomass
    ax2 = ax.twinx()
    ax2.set_ylabel('Biomass (1/hr)', fontsize=12)
    ax2.set_ylim(0, 1.0)

    # Add biomass values as text
    y_pos = 0.95
    for label, biomass in biomass_dict.items():
        short_label = label.split(':')[0]  # Just "Level X"
        ax2.text(1.01, y_pos, f"{short_label}: {biomass:.3f}",
                transform=ax2.transAxes, fontsize=9,
                color=colors.get(label, 'gray'))
        y_pos -= 0.04

    # Format plot
    ax.set_xscale('log')
    ax.set_xlabel('Flux Variability Range (mmol/gDCW/hr)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('FVA Ablation Study: Impact of kinGEMs Constraints', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1e-6, 1e3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to: {output_file}")
    plt.close()


def generate_summary_statistics(fva_results_dict, biomass_dict, output_file):
    """Generate summary statistics table for all levels."""
    print("\n=== Generating Summary Statistics ===")

    summary_data = []

    for label, fva_df in fva_results_dict.items():
        fvi = calculate_fvi(fva_df)
        biomass = biomass_dict[label]

        summary_data.append({
            'Level': label,
            'Biomass (1/hr)': biomass,
            'N Reactions': len(fva_df),
            'Mean FVI': fvi.mean(),
            'Median FVI': fvi.median(),
            'Std FVI': fvi.std(),
            'Min FVI': fvi.min(),
            'Max FVI': fvi.max(),
            '% Zero Flux': (fvi == 0).sum() / len(fvi) * 100,
            '% High Variability (FVI > 1)': (fvi > 1).sum() / len(fvi) * 100
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"  Saved summary to: {output_file}")

    # Print to console
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('config', help='Path to configuration JSON file')
    parser.add_argument('--parallel', action='store_true', help='Use parallel FVA')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers for parallel FVA')
    parser.add_argument('--skip-tuning', action='store_true', help='Skip simulated annealing (use existing tuned file)')
    parser.add_argument('--tuned-file', type=str, default=None, help='Path to pre-tuned data file')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    model_name = config['model_name']
    organism = config.get('organism', 'Unknown')
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.15)

    # Generate run ID
    run_id = f"{model_name}_FVA_ablation_{datetime.today().strftime('%Y%m%d')}_{random.randint(1000, 9999)}"

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")
    raw_data_dir = os.path.join(data_dir, "raw")
    interim_data_dir = os.path.join(data_dir, "interim", model_name)
    processed_data_dir = os.path.join(data_dir, "processed", model_name)
    CPIPred_data_dir = os.path.join(data_dir, "interim", "CPI-Pred predictions")
    results_dir = os.path.join(project_root, "results", "fva_ablation", run_id)
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(raw_data_dir, f"{model_name}.xml")
    processed_data_output = os.path.join(processed_data_dir, f"{model_name}_processed_data.csv")

    print("\n" + "="*80)
    print(f"=== kinGEMs FVA Ablation Study: {model_name} ===")
    print("="*80)
    print(f"Run ID: {run_id}")
    print(f"Organism: {organism}")
    print(f"Enzyme upper bound: {enzyme_upper_bound}")
    print(f"Parallel FVA: {args.parallel}")
    if args.parallel:
        print(f"Workers: {args.workers or 'auto'}")
    print(f"Results directory: {results_dir}")
    print("="*80)

    # Load model and data
    print("\n=== Loading Model and Data ===")
    model = load_model(model_path)

    # Determine biomass reaction
    obj_rxns = {rxn.id: rxn.objective_coefficient
                for rxn in model.reactions
                if rxn.objective_coefficient != 0}
    biomass_reaction = config.get('biomass_reaction') or next(iter(obj_rxns.keys()))
    print(f"Biomass reaction: {biomass_reaction}")

    # Load processed data
    if not os.path.exists(processed_data_output):
        print(f"  ⚠️  Processed data not found: {processed_data_output}")
        print("  Please run the main pipeline first to generate processed data")
        sys.exit(1)

    processed_df = pd.read_csv(processed_data_output)
    print(f"Loaded processed data: {len(processed_df)} rows")

    # Ensure kcat column exists
    if 'kcat_mean' in processed_df.columns and 'kcat' not in processed_df.columns:
        processed_df['kcat'] = processed_df['kcat_mean']

    # Dictionary to store results
    fva_results = {}
    biomass_values = {}

    # ===================================================================
    # Level 1: Baseline GEM
    # ===================================================================
    fva_df, biomass = run_baseline_fva(model)
    fva_results['Level 1: Baseline GEM'] = fva_df
    biomass_values['Level 1: Baseline GEM'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level1_baseline.csv'), index=False)

    # ===================================================================
    # Level 2: Single Enzyme Constraint
    # ===================================================================
    fva_df, biomass = run_single_enzyme_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        args.parallel, args.workers
    )
    fva_results['Level 2: Single Enzyme'] = fva_df
    biomass_values['Level 2: Single Enzyme'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level2_single_enzyme.csv'), index=False)

    # ===================================================================
    # Level 3a: + Isoenzymes
    # ===================================================================
    fva_df, biomass = run_isoenzymes_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        args.parallel, args.workers
    )
    fva_results['Level 3a: + Isoenzymes'] = fva_df
    biomass_values['Level 3a: + Isoenzymes'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level3a_isoenzymes.csv'), index=False)

    # ===================================================================
    # Level 3b: + Complexes
    # ===================================================================
    fva_df, biomass = run_complexes_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        args.parallel, args.workers
    )
    fva_results['Level 3b: + Complexes'] = fva_df
    biomass_values['Level 3b: + Complexes'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level3b_complexes.csv'), index=False)

    # ===================================================================
    # Level 3c: + Promiscuous
    # ===================================================================
    fva_df, biomass = run_promiscuous_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        args.parallel, args.workers
    )
    fva_results['Level 3c: + Promiscuous'] = fva_df
    biomass_values['Level 3c: + Promiscuous'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level3c_promiscuous.csv'), index=False)

    # ===================================================================
    # Level 4: All Constraints
    # ===================================================================
    fva_df, biomass = run_all_constraints_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        args.parallel, args.workers
    )
    fva_results['Level 4: All Constraints'] = fva_df
    biomass_values['Level 4: All Constraints'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level4_all_constraints.csv'), index=False)

    # ===================================================================
    # Level 5: Post-Tuned (if available or run tuning)
    # ===================================================================
    if args.skip_tuning and args.tuned_file:
        print("\n=== Level 5: Loading Pre-Tuned Data ===")
        tuned_df = pd.read_csv(args.tuned_file)
        print(f"  Loaded tuned data: {len(tuned_df)} rows")
    elif args.skip_tuning:
        print("\n=== Level 5: Skipped (no tuned data provided) ===")
        tuned_df = None
    else:
        print("\n=== Level 5: Running Simulated Annealing ===")
        sa_config = config.get('simulated_annealing', {})

        # Run optimization to get gene_sequences_dict
        (solution_value, df_FBA, gene_sequences_dict, _) = run_optimization_with_dataframe(
            model=model,
            processed_df=processed_df,
            objective_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            enzyme_ratio=True,
            maximization=True,
            output_dir=None,
            save_results=False,
            print_reaction_conditions=False,
            verbose=False
        )

        # Run simulated annealing
        kcat_dict, top_targets, tuned_df, iterations, biomasses, df_FBA = simulated_annealing(
            model=model,
            processed_data=processed_df,
            biomass_reaction=biomass_reaction,
            objective_value=sa_config.get('biomass_goal', 0.5),
            gene_sequences_dict=gene_sequences_dict,
            output_dir=results_dir,
            enzyme_fraction=enzyme_upper_bound,
            temperature=sa_config.get('temperature', 1.0),
            cooling_rate=sa_config.get('cooling_rate', 0.95),
            min_temperature=sa_config.get('min_temperature', 0.01),
            max_iterations=sa_config.get('max_iterations', 100),
            max_unchanged_iterations=sa_config.get('max_unchanged_iterations', 5),
            change_threshold=sa_config.get('change_threshold', 0.009),
            verbose=False
        )

        print(f"  Tuning complete: {biomasses[0]:.4f} → {biomasses[-1]:.4f}")
        tuned_df.to_csv(os.path.join(results_dir, 'tuned_data.csv'), index=False)

    if tuned_df is not None:
        fva_df, biomass = run_tuned_fva(
            model, tuned_df, biomass_reaction, enzyme_upper_bound,
            args.parallel, args.workers
        )
        fva_results['Level 5: Post-Tuned'] = fva_df
        biomass_values['Level 5: Post-Tuned'] = biomass
        fva_df.to_csv(os.path.join(results_dir, 'level5_post_tuned.csv'), index=False)

    # ===================================================================
    # Generate Plots and Summary
    # ===================================================================
    plot_file = os.path.join(results_dir, 'fva_ablation_cumulative.png')
    plot_fva_ablation(fva_results, biomass_values, plot_file)

    summary_file = os.path.join(results_dir, 'ablation_summary.csv')
    generate_summary_statistics(fva_results, biomass_values, summary_file)

    # ===================================================================
    # Final Summary
    # ===================================================================
    print("\n" + "="*80)
    print("=== FVA Ablation Study Complete ===")
    print("="*80)
    print(f"Run ID: {run_id}")
    print(f"Results directory: {results_dir}")
    print("\nGenerated files:")
    print("  - level1_baseline.csv")
    print("  - level2_single_enzyme.csv")
    print("  - level3a_isoenzymes.csv")
    print("  - level3b_complexes.csv")
    print("  - level3c_promiscuous.csv")
    print("  - level4_all_constraints.csv")
    if tuned_df is not None:
        print("  - level5_post_tuned.csv")
    print("  - fva_ablation_cumulative.png")
    print("  - ablation_summary.csv")
    print("="*80)


if __name__ == '__main__':
    main()

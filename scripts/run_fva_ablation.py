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
import gc
import os
import random
import sys
import warnings

from cobra.flux_analysis import flux_variability_analysis as cobra_fva
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.dataset import (
    load_model,
    convert_to_irreversible
)
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.modeling.tuning import simulated_annealing
from kinGEMs.plots import create_fva_ablation_dashboard
from kinGEMs.utils import (
    load_config,
    load_existing_results,
    run_single_enzyme_fva,
    run_isoenzymes_fva,
    run_complexes_fva,
    run_promiscuous_fva,
    run_all_constraints_fva,
    run_tuned_fva,
)

# Suppress warnings
warnings.filterwarnings('ignore')


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
    parser.add_argument('--no-biomass-constraint', action='store_true',
                        help='Run FVA without constraining biomass to near-optimal (allows biomass to vary freely)')
    parser.add_argument('--replot', type=str, default=None,
                        help='Path to existing results directory to regenerate figures only (skip FVA)')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    model_name = config['model_name']
    organism = config.get('organism', 'Unknown')
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.15)

    # =======================================================================
    # REPLOT MODE: Load existing results and regenerate figures only
    # =======================================================================
    if args.replot:
        results_dir = args.replot
        if not os.path.isdir(results_dir):
            print(f"Error: Results directory not found: {results_dir}")
            sys.exit(1)

        print("\n" + "="*80)
        print(f"=== REPLOT MODE: Regenerating Figures ===")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"Results directory: {results_dir}")
        print("="*80)

        # Load existing results
        fva_results, biomass_values = load_existing_results(results_dir)

        # Generate plots
        print("\n=== Regenerating Figures ===")
        create_fva_ablation_dashboard(
            fva_results, biomass_values, model_name, results_dir,
            prefix="fva_ablation", show=False
        )

        print("\n" + "="*80)
        print("=== Replot Complete ===")
        print("="*80)
        print(f"Results directory: {results_dir}")
        print("\nRegenerated files:")
        print("  - fva_ablation_cumulative.png")
        print("  - fva_ablation_boxplot.png")
        print("  - fva_ablation_violinplot.png")
        print("  - fva_ablation_biomass_progression.png")
        print("  - fva_ablation_summary.csv")
        print("="*80)

        return  # Exit early - don't run FVA computations

    # =======================================================================
    # NORMAL MODE: Run full FVA ablation study
    # =======================================================================

    # Get FVA configuration from config file, with command-line overrides
    fva_config = config.get('fva', {})
    use_parallel = args.parallel or fva_config.get('parallel', False)
    n_workers = args.workers or fva_config.get('workers', None)
    fva_method = fva_config.get('method', 'dask')
    chunk_size = fva_config.get('chunk_size', None)
    constrain_biomass = not args.no_biomass_constraint  # Invert flag for parameter

    # Generate run ID
    run_id = f"{model_name}_FVA_ablation_{datetime.today().strftime('%Y%m%d')}_{random.randint(1000, 9999)}"

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")
    raw_data_dir = os.path.join(data_dir, "raw")
    processed_data_dir = os.path.join(data_dir, "processed", model_name)
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
    print(f"Constrain biomass: {constrain_biomass}")
    print(f"Parallel FVA: {use_parallel}")
    if use_parallel:
        print(f"Workers: {n_workers or 'auto'}")
        print(f"Method: {fva_method}")
        print(f"Chunk size: {chunk_size or 'auto'}")
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
        print(f"  !!!  Processed data not found: {processed_data_output}")
        print("  Please run the main pipeline first to generate processed data")
        sys.exit(1)

    processed_df = pd.read_csv(processed_data_output)
    print(f"Loaded processed data: {len(processed_df)} rows")

    # Ensure kcat column exists
    if 'kcat_mean' in processed_df.columns and 'kcat' not in processed_df.columns:
        processed_df['kcat'] = processed_df['kcat_mean']

    # Print constraint configuration table
    print("\n" + "="*80)
    print("=== ABLATION STUDY CONSTRAINT CONFIGURATION ===")
    print("="*80)
    print(f"{'Level':<30} {'Multi-Enzyme':<12} {'Isoenzymes':<11} {'Complexes':<10} {'Promiscuous':<12} {'Description'}")
    print("-" * 80)
    print(f"{'1: Baseline GEM':<30} {'N/A':<12} {'N/A':<11} {'N/A':<10} {'N/A':<12} {'No constraints (irreversible)'}")
    print(f"{'2: Single Enzyme':<30} {'✓ Enabled':<12} {'✗ Disabled':<11} {'✗ Disabled':<10} {'✗ Disabled':<12} {'Basic enzyme constraints only'}")
    print(f"{'3a: + Isoenzymes':<30} {'✓ Enabled':<12} {'✓ Enabled':<11} {'✗ Disabled':<10} {'✗ Disabled':<12} {'+ OR-GPR handling'}")
    print(f"{'3b: + Complexes':<30} {'✓ Enabled':<12} {'✗ Disabled':<11} {'✓ Enabled':<10} {'✗ Disabled':<12} {'+ AND-GPR handling'}")
    print(f"{'3c: + Promiscuous':<30} {'✓ Enabled':<12} {'✗ Disabled':<11} {'✗ Disabled':<10} {'✓ Enabled':<12} {'+ Cross-reaction sharing'}")
    print(f"{'4: All Constraints':<30} {'✓ Enabled':<12} {'✓ Enabled':<11} {'✓ Enabled':<10} {'✓ Enabled':<12} {'Complete kinGEMs'}")
    print(f"{'5: Post-Tuned':<30} {'✓ Enabled':<12} {'✓ Enabled':<11} {'✓ Enabled':<10} {'✓ Enabled':<12} {'+ Optimized kcat values'}")
    print("="*80)

    # Dictionary to store results
    fva_results = {}
    biomass_values = {}

    # ===================================================================
    # Convert model to irreversible for all computations
    # ===================================================================
    model = convert_to_irreversible(model)

    # ===================================================================
    # Level 1: Baseline GEM (Irreversible, no enzyme constraints)
    # ===================================================================
    print("\n=== Level 1: Baseline GEM (irreversible, no enzyme constraints) ===")
    fva_results_irrev = cobra_fva(model, fraction_of_optimum=0.9)

    fva_df = pd.DataFrame({
        'Reactions': fva_results_irrev.index,
        'Min Solutions': fva_results_irrev['minimum'],
        'Max Solutions': fva_results_irrev['maximum'],
        'Solution Biomass': [model.slim_optimize()] * len(fva_results_irrev)
    })

    biomass = model.slim_optimize()
    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    fva_results['Level 1: Baseline GEM'] = fva_df
    biomass_values['Level 1: Baseline GEM'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level1_baseline_irreversible.csv'), index=False)

    # ===================================================================
    # Level 2: Single Enzyme Constraint
    # ===================================================================
    fva_df, biomass = run_single_enzyme_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        use_parallel, n_workers, fva_method, chunk_size, constrain_biomass
    )
    fva_results['Level 2: Single Enzyme'] = fva_df
    biomass_values['Level 2: Single Enzyme'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level2_single_enzyme.csv'), index=False)

    # ===================================================================
    # Level 3a: + Isoenzymes
    # ===================================================================
    fva_df, biomass = run_isoenzymes_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        use_parallel, n_workers, fva_method, chunk_size, constrain_biomass
    )
    fva_results['Level 3a: + Isoenzymes'] = fva_df
    biomass_values['Level 3a: + Isoenzymes'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level3a_isoenzymes.csv'), index=False)

    # ===================================================================
    # Level 3b: + Complexes
    # ===================================================================
    fva_df, biomass = run_complexes_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        use_parallel, n_workers, fva_method, chunk_size, constrain_biomass
    )
    fva_results['Level 3b: + Complexes'] = fva_df
    biomass_values['Level 3b: + Complexes'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level3b_complexes.csv'), index=False)

    # ===================================================================
    # Level 3c: + Promiscuous
    # ===================================================================
    fva_df, biomass = run_promiscuous_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        use_parallel, n_workers, fva_method, chunk_size, constrain_biomass
    )
    fva_results['Level 3c: + Promiscuous'] = fva_df
    biomass_values['Level 3c: + Promiscuous'] = biomass
    fva_df.to_csv(os.path.join(results_dir, 'level3c_promiscuous.csv'), index=False)

    # ===================================================================
    # Level 4: All Constraints
    # ===================================================================
    fva_df, biomass = run_all_constraints_fva(
        model, processed_df, biomass_reaction, enzyme_upper_bound,
        use_parallel, n_workers, fva_method, chunk_size, constrain_biomass
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
            n_top_enzymes=sa_config.get('n_top_enzymes', 65),
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

        # Clean up simulated annealing memory
        del df_FBA  # Large result dataframe from simulated annealing
        del iterations, biomasses  # Iteration data no longer needed for FVA
        gc.collect()  # Force garbage collection

    if tuned_df is not None:
        fva_df, biomass = run_tuned_fva(
            model, tuned_df, biomass_reaction, enzyme_upper_bound,
            use_parallel, n_workers, fva_method, chunk_size, constrain_biomass
        )
        fva_results['Level 5: Post-Tuned'] = fva_df
        biomass_values['Level 5: Post-Tuned'] = biomass
        fva_df.to_csv(os.path.join(results_dir, 'level5_post_tuned.csv'), index=False)

        # Memory cleanup after Level 5
        gc.collect()

    # ===================================================================
    # Generate Plots and Summary using universal plotting functions
    # ===================================================================

    # Create comprehensive dashboard with all plots and statistics
    create_fva_ablation_dashboard(
        fva_results, biomass_values, model_name, results_dir,
        prefix="fva_ablation", show=False
    )

    # ===================================================================
    # Final Summary
    # ===================================================================
    print("\n" + "="*80)
    print("=== FVA Ablation Study Complete ===")
    print("="*80)
    print(f"Run ID: {run_id}")
    print(f"Results directory: {results_dir}")
    print("\nGenerated files:")
    print("  - level1_baseline_irreversible.csv")
    print("  - level2_single_enzyme.csv")
    print("  - level3a_isoenzymes.csv")
    print("  - level3b_complexes.csv")
    print("  - level3c_promiscuous.csv")
    print("  - level4_all_constraints.csv")
    if tuned_df is not None:
        print("  - level5_post_tuned.csv")
    print("  - fva_ablation_cumulative.png (enhanced with biomass subplot)")
    print("  - fva_ablation_boxplot.png (distribution analysis)")
    print("  - fva_ablation_biomass_progression.png (biomass vs constraints)")
    print("  - fva_ablation_summary.csv (updated with corrected FVi terminology)")
    print("="*80)


if __name__ == '__main__':
    main()

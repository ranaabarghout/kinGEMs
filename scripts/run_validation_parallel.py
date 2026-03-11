#!/usr/bin/env python3
"""
Parallel Validation Runner - Runs individual validation components
===================================================================

This script runs individual parts of the validation pipeline that can be
executed in parallel on separate cluster nodes.

Usage:
    python scripts/run_validation_parallel.py --mode wildtype --config <config> --output <dir>
    python scripts/run_validation_parallel.py --mode baseline --config <config> --output <dir>
    python scripts/run_validation_parallel.py --mode pretuning --config <config> --output <dir>
    python scripts/run_validation_parallel.py --mode posttuning --config <config> --output <dir>

Modes:
    wildtype   - Calculate wild-type growth (no gene knockouts) for each carbon source
    baseline   - Run baseline GEM validation (no enzyme constraints)
    pretuning  - Run pre-tuning kinGEMs validation (initial kcat values)
    posttuning - Run post-tuning kinGEMs validation (tuned kcat values)
"""

import argparse
from datetime import datetime
import json
import logging
import os
import sys
import warnings

import cobra
import numpy as np
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.config import ECOLI_VALIDATION_DIR
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.validation_utils import (
    check_environment,
    load_data,
    load_environment,
    match_model_data,
    model_adjustments,
    prepare_model,
    simulate_phenotype,
    simulate_phenotype_parallel,
    set_carbon_source_safely,
    set_medium_safely,
    reset_carbon_source_safely,
)
from kinGEMs.dataset import convert_to_irreversible

# Silence warnings
warnings.filterwarnings('ignore')
logging.getLogger('distributed').setLevel(logging.ERROR)


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def simulate_wild_type_growth(model_adj, name_carbon_model_matched_adj,
                              medium_ex_inds, carbon_ex_inds):
    """Simulate wild-type growth (no gene knockouts) for each carbon source.

    This provides the reference growth rates needed to convert mutant growth
    rates to fitness values (log2 ratios).

    Parameters
    ----------
    model_adj : cobra.Model
        The adjusted model (no gene knockouts)
    name_carbon_model_matched_adj : list
        List of carbon source reaction IDs
    medium_ex_inds : list
        Indices of medium exchange reactions
    carbon_ex_inds : list
        Indices of carbon source exchange reactions
    Returns
    -------
    numpy.ndarray
        Wild-type growth rates for each carbon source (n_carbons,)
    """
    n_carbons = len(name_carbon_model_matched_adj)
    wild_type_growth = np.zeros(n_carbons)

    # Set medium using safe bounds handling
    set_medium_safely(model_adj, medium_ex_inds, uptake_rate=-1000)

    # IMPORTANT: Close all carbon sources initially to ensure only one is active at a time
    for carbon_idx in carbon_ex_inds:
        reset_carbon_source_safely(model_adj, carbon_idx)

    print(f"  Calculating wild-type growth for {n_carbons} carbon sources...")

    for e, carbon in enumerate(name_carbon_model_matched_adj):
        print(f"  Wild-type growth progress: {e+1}/{n_carbons}", end='\r')

        # Check if this carbon source was already tested
        if carbon in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(carbon)
            wild_type_growth[e] = wild_type_growth[e_found]
            continue

        # Set carbon source using safe bounds handling
        set_carbon_source_safely(model_adj, carbon_ex_inds[e], uptake_rate=-10)

        # Optimize (NO gene knockouts)
        solution = model_adj.slim_optimize()
        if np.isnan(solution):
            solution = 0
        wild_type_growth[e] = solution

        # Reset carbon source using safe bounds handling
        reset_carbon_source_safely(model_adj, carbon_ex_inds[e])

        if (e + 1) % 5 == 0:
            print(f"  Wild-type growth progress: {e+1}/{n_carbons} carbon sources completed")

    print(f"\n  Wild-type growth calculation complete ({n_carbons} carbon sources).")
    print(f"  Mean growth rate: {np.mean(wild_type_growth):.6f}")
    print(f"  Range: {np.min(wild_type_growth):.6f} to {np.max(wild_type_growth):.6f}")

    return wild_type_growth


def simulate_wild_type_growth_kingems(model_adj, name_carbon_model_matched_adj,
                                      medium_ex_inds, carbon_ex_inds,
                                      processed_df, objective_reaction, enzyme_upper_bound):
    """Simulate wild-type growth for kinGEMs model (WITH enzyme constraints, but no gene knockouts).

    This calculates the reference growth rates for kinGEMs models that have enzyme
    constraints applied, which is needed to convert mutant growth rates to fitness values.

    Parameters
    ----------
    model_adj : cobra.Model
        The adjusted model (no gene knockouts)
    name_carbon_model_matched_adj : list
        List of carbon source reaction IDs
    medium_ex_inds : list
        Indices of medium exchange reactions
    carbon_ex_inds : list
        Indices of carbon source exchange reactions
    processed_df : pandas.DataFrame
        DataFrame with enzyme constraint data (kcat values)
    objective_reaction : str
        Objective reaction ID
    enzyme_upper_bound : float
        Upper bound for enzyme concentration
    Returns
    -------
    numpy.ndarray
        Wild-type growth rates for each carbon source (n_carbons,) with enzyme constraints
    """
    n_carbons = len(name_carbon_model_matched_adj)
    wild_type_growth = np.zeros(n_carbons)
    # Ensure kcat_mean is numeric
    if 'kcat_mean' in processed_df.columns:
        processed_df['kcat_mean'] = processed_df['kcat_mean'].apply(
            lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else x)

    # Set medium using safe bounds handling
    set_medium_safely(model_adj, medium_ex_inds, uptake_rate=-1000)

    # IMPORTANT: Close all carbon sources initially to ensure only one is active at a time
    for carbon_idx in carbon_ex_inds:
        reset_carbon_source_safely(model_adj, carbon_idx)

    print(f"  Calculating kinGEMs wild-type growth for {n_carbons} carbon sources...")
    print("  (WITH enzyme constraints, NO gene knockouts)")

    for e, carbon in enumerate(name_carbon_model_matched_adj):
        print(f"  kinGEMs wild-type progress: {e+1}/{n_carbons}", end='\r')

        # Check if this carbon source was already tested
        if carbon in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(carbon)
            wild_type_growth[e] = wild_type_growth[e_found]
            continue

        # IMPORTANT: Close ALL carbon sources before opening the one we want to test
        for carbon_idx in carbon_ex_inds:
            reset_carbon_source_safely(model_adj, carbon_idx)

        # Set carbon source using safe bounds handling
        set_carbon_source_safely(model_adj, carbon_ex_inds[e], uptake_rate=-10)
        
        # DIAGNOSTIC: Check carbon source bounds before optimization
        if e == 0:  # Only print for first carbon to avoid spam
            print(f"\n  [DEBUG] Carbon exchange bounds BEFORE optimization (carbon {e}):")
            for idx, c_idx in enumerate(carbon_ex_inds[:min(4, len(carbon_ex_inds))]):
                if c_idx != -1:
                    ex = model_adj.exchanges[c_idx]
                    print(f"    Carbon {idx} ({ex.id}): LB={ex.lower_bound:.2f}, UB={ex.upper_bound:.2f}")
                    # Check reverse if exists
                    try:
                        rev = model_adj.reactions.get_by_id(ex.id + '_reverse')
                        print(f"      → Reverse: LB={rev.lower_bound:.2f}, UB={rev.upper_bound:.2f}")
                    except:
                        pass

        # Optimize WITH enzyme constraints, but NO gene knockouts
        # No gene knockout needed - just optimize with enzyme constraints
        try:
            solution = run_optimization_with_dataframe(
                model=model_adj,
                processed_df=processed_df,
                objective_reaction=objective_reaction,
                enzyme_upper_bound=enzyme_upper_bound,
                enzyme_ratio=True,
                maximization=True,
                multi_enzyme_off=False,
                isoenzymes_off=False,
                promiscuous_off=False,
                complexes_off=False,
                output_dir=None,
                save_results=False,
                print_reaction_conditions=False,
                verbose=False,
                solver_name='glpk'
            )
            # Extract objective value from tuple return (solution_value, df_FBA, gene_sequences_dict, _)
            if isinstance(solution, tuple):
                solution = solution[0]
            if solution is None or np.isnan(solution):
                solution = 0
        except Exception as ex:
            print(f"\n  Warning: Optimization failed for carbon {carbon}: {ex}")
            solution = 0

        wild_type_growth[e] = solution

        # Note: We'll reset all carbons at the start of the next iteration
        # No need to reset here since we close all before each test

        if (e + 1) % 5 == 0:
            print(f"  kinGEMs wild-type progress: {e+1}/{n_carbons} carbon sources completed")

    print(f"\n  kinGEMs wild-type growth calculation complete ({n_carbons} carbon sources).")
    print(f"  Mean growth rate: {np.mean(wild_type_growth):.6f}")
    print(f"  Range: {np.min(wild_type_growth):.6f} to {np.max(wild_type_growth):.6f}")

    return wild_type_growth


def simulate_baseline_only(model_adj, name_genes_matched_adj, name_carbon_model_matched_adj,
                           medium_ex_inds, carbon_ex_inds):
    """Simulate baseline GEM (no enzyme constraints) only."""
    n_genes = len(name_genes_matched_adj)
    n_carbons = len(name_carbon_model_matched_adj)
    baseline_GEM = np.zeros((n_genes, n_carbons))

    # Set medium using safe bounds handling
    set_medium_safely(model_adj, medium_ex_inds, uptake_rate=-1000)

    for e, carbon in enumerate(name_carbon_model_matched_adj):
        print(f"  Baseline GEM progress: {e+1}/{n_carbons}", end='\r')

        # Check if this carbon source was already tested
        if carbon in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(carbon)
            baseline_GEM[:, e] = baseline_GEM[:, e_found]
            continue

        # Set carbon source using safe bounds handling
        set_carbon_source_safely(model_adj, carbon_ex_inds[e], uptake_rate=-10)

        # Test each gene knockout
        for g, gene in enumerate(name_genes_matched_adj):
            with model_adj:
                model_adj.genes.get_by_id(gene).knock_out()
                solution = model_adj.slim_optimize()
                if np.isnan(solution):
                    solution = 0
                baseline_GEM[g, e] = solution

        # Reset carbon source using safe bounds handling
        reset_carbon_source_safely(model_adj, carbon_ex_inds[e])

        if (e + 1) % 5 == 0:
            print(f"  Baseline GEM progress: {e+1}/{n_carbons} carbon sources completed")

    print(f"\n  Baseline GEM simulation complete ({n_genes} genes × {n_carbons} carbon sources).")
    return baseline_GEM


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', required=True,
                       choices=['wildtype', 'baseline', 'pretuning', 'posttuning'],
                       help='Validation mode to run')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    parser.add_argument('--output', required=True, help='Output directory for results')

    # Enzyme constraint options (mutually exclusive)
    enzyme_group = parser.add_mutually_exclusive_group()
    enzyme_group.add_argument('--keep-enzyme-constraints', action='store_true',
                       help='Keep enzyme constraints for knocked-out genes (immediate knockout effect). '
                            'By default, enzyme constraints are removed when a gene is knocked out, '
                            'which frees protein pool capacity. Use this flag to model total knockout '
                            'with no proteome reallocation (enzyme cost is locked in/wasted).')
    enzyme_group.add_argument('--remove-knockout-enzyme', action='store_true',
                       help='Explicitly remove enzyme constraints for knocked-out genes (proteome reallocation). '
                            'This is the DEFAULT behavior. Use this flag for clarity when enzyme removal is intended. '
                            'When a gene is knocked out, its enzyme constraint is removed and protein pool is freed '
                            'for other reactions (may result in slight growth improvements).')

    # Sampling option for faster testing
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample a subset of gene-carbon combinations for faster testing. '
                            'E.g., --sample-size 1000 will sample ~1000 datapoints. '
                            'Default: use all data (no sampling).')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Extract configuration
    model_name = config['model_name']
    model_path = config['model_path']
    objective_reaction = config.get('objective_reaction', None)
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.15)

    # Data paths
    pre_tuning_data_path = config.get('pre_tuning_data_path', None)
    post_tuning_data_path = config.get('post_tuning_data_path', None)

    # Parallel configuration
    parallel_config = config.get('parallel', {})
    use_parallel = parallel_config.get('enabled', True)
    n_workers = parallel_config.get('workers', None)
    parallel_method = parallel_config.get('method', 'dask')
    chunk_size = parallel_config.get('chunk_size', None)

    # Enzyme constraint handling
    # When True, enzyme constraints are REMOVED for knocked-out genes (default behavior)
    # When False, enzyme constraints are KEPT even for knocked-out genes (--keep-enzyme-constraints flag)
    # The --remove-knockout-enzyme flag explicitly enables removal (same as default)
    remove_knockout_enzyme = not args.keep_enzyme_constraints

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "="*70)
    print(f"=== kinGEMs Parallel Validation: {args.mode.upper()} ===")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print(f"Parallel: {use_parallel} ({parallel_method} with {n_workers or 'auto'} workers)")
    if args.keep_enzyme_constraints:
        print(f"Enzyme constraints: KEPT for knocked-out genes (--keep-enzyme-constraints)")
        print(f"                   → Immediate knockout effect (no proteome reallocation)")
    elif args.remove_knockout_enzyme:
        print(f"Enzyme constraints: REMOVED for knocked-out genes (--remove-knockout-enzyme)")
        print(f"                   → Proteome reallocation enabled (may show growth improvements)")
    else:
        print(f"Enzyme constraints: REMOVED for knocked-out genes (default)")
        print(f"                   → Proteome reallocation enabled (may show growth improvements)")
    print("="*70)

    # === Step 1: Load Model ===
    print("\n=== Step 1: Loading model ===")
    model = cobra.io.read_sbml_model(model_path)
    model = convert_to_irreversible(model)
    print(f"  Genes: {len(model.genes)}")
    print(f"  Reactions: {len(model.reactions)}")

    # Set solver
    solver_name = config.get('solver', 'glpk')  # Default to GLPK

    # Force GLPK if gurobi causes license issues
    print(f"  Setting solver to: {solver_name.upper()}")
    try:
        model.solver = solver_name
        print(f"  ✓ Using solver: {solver_name.upper()}")
    except Exception as e:
        print(f"  ⚠️  Could not set solver to {solver_name}: {e}")
        print(f"  Falling back to GLPK...")
        try:
            model.solver = 'glpk'
            solver_name = 'glpk'
            print(f"  ✓ Using solver: GLPK")
        except Exception as e2:
            print(f"  ⚠️  Could not set GLPK: {e2}")
            print(f"  Using default solver: {model.solver.interface.__name__}")

    current_solver = model.solver.interface.__name__
    print(f"  Confirmed solver: {current_solver}")

    # Auto-detect objective
    if objective_reaction is None:
        obj_rxns = {rxn.id: rxn.objective_coefficient
                    for rxn in model.reactions
                    if rxn.objective_coefficient != 0}
        if obj_rxns:
            objective_reaction = next(iter(obj_rxns.keys()))
            print(f"  Objective: {objective_reaction}")
    else:
        print(f"  Objective: {objective_reaction}")

    # === Step 2: Prepare Environment ===
    print("\n=== Step 2: Preparing validation environment ===")
    og_model = model.copy()
    print("Initial growth rate: ", model.slim_optimize())
    model = prepare_model(model)
    name_medium_model, name_carbon_model, name_carbon_experiment = load_environment(ECOLI_VALIDATION_DIR)
    data_experiments, data_genes, data_fitness = load_data(ECOLI_VALIDATION_DIR)

    name_genes_matched, name_carbon_experiment_matched, name_carbon_model_matched, data_fitness_matched = match_model_data(
        model=model,
        name_carbon_model=name_carbon_model,
        name_carbon_experiment=name_carbon_experiment,
        data_experiments=data_experiments,
        data_genes=data_genes,
        data_fitness=data_fitness
    )

    model_adj, name_genes_matched_adj, name_carbon_experiment_matched_adj, name_carbon_model_matched_adj, data_fitness_matched_adj = model_adjustments(
        adj_strain=True,
        adj_essential=True,
        adj_carbon=True,
        model=model,
        name_genes_matched=name_genes_matched,
        name_carbon_experiment_matched=name_carbon_experiment_matched,
        name_carbon_model_matched=name_carbon_model_matched,
        data_fitness_matched=data_fitness_matched
    )

    medium_ex_inds, carbon_ex_inds = check_environment(
        model_adj=model_adj,
        name_medium_model=name_medium_model,
        name_carbon_model_matched_adj=name_carbon_model_matched_adj
    )

    print(f"  Matched genes: {len(name_genes_matched_adj)}")
    print(f"  Carbon sources: {len(name_carbon_model_matched_adj)}")

    # === Optional: Sample subset of data for faster testing ===
    if args.sample_size:
        print(f"\n=== Sampling subset of data ===")
        n_genes_full = len(name_genes_matched_adj)
        n_carbons_full = len(name_carbon_model_matched_adj)
        total_combinations = n_genes_full * n_carbons_full

        print(f"  Full dataset: {n_genes_full} genes × {n_carbons_full} carbons = {total_combinations:,} combinations")
        print(f"  Target sample size: {args.sample_size:,} combinations")

        # Calculate how many genes and carbons to sample
        # Try to keep a balanced ratio
        import math
        ratio = n_genes_full / n_carbons_full
        n_carbons_sample = int(math.sqrt(args.sample_size / ratio))
        n_genes_sample = int(args.sample_size / n_carbons_sample)

        # Find unique carbon sources (some may be duplicated in the list)
        unique_carbons = []
        unique_carbon_indices = []
        seen_carbons = set()
        for i, carbon in enumerate(name_carbon_model_matched_adj):
            if carbon not in seen_carbons:
                unique_carbons.append(carbon)
                unique_carbon_indices.append(i)
                seen_carbons.add(carbon)

        n_unique_carbons = len(unique_carbons)
        print(f"  Unique carbon sources: {n_unique_carbons} (from {n_carbons_full} total entries)")

        # Ensure we don't exceed available unique data
        n_carbons_sample = min(n_carbons_sample, n_unique_carbons)
        n_genes_sample = min(n_genes_sample, n_genes_full)

        # Random sampling with fixed seed for reproducibility
        np.random.seed(42)
        gene_indices = np.random.choice(n_genes_full, size=n_genes_sample, replace=False)

        # Sample from UNIQUE carbon indices to avoid duplicates
        sampled_unique_indices = np.random.choice(len(unique_carbon_indices), size=n_carbons_sample, replace=False)
        carbon_indices = [unique_carbon_indices[i] for i in sampled_unique_indices]

        # Sample the data
        name_genes_matched_adj = [name_genes_matched_adj[i] for i in sorted(gene_indices)]
        name_carbon_model_matched_adj = [name_carbon_model_matched_adj[i] for i in sorted(carbon_indices)]
        medium_ex_inds = medium_ex_inds  # Keep all medium components
        carbon_ex_inds = [carbon_ex_inds[i] for i in sorted(carbon_indices)]
        data_fitness_matched_adj = data_fitness_matched_adj[np.ix_(gene_indices, carbon_indices)]

        actual_size = len(name_genes_matched_adj) * len(name_carbon_model_matched_adj)
        print(f"  Sampled: {len(name_genes_matched_adj)} genes × {len(name_carbon_model_matched_adj)} carbons = {actual_size:,} combinations")
        print(f"  Reduction: {100 * (1 - actual_size/total_combinations):.1f}% fewer datapoints")

    # === Step 3: Run Simulation Based on Mode ===
    print(f"\n=== Step 3: Running {args.mode.upper()} simulation ===")

    if args.mode == 'wildtype':
        print("Calculating Wild-Type Growth (no gene knockouts)...")
        wild_type_growth = simulate_wild_type_growth(
            model_adj=model_adj,
            name_carbon_model_matched_adj=name_carbon_model_matched_adj,
            medium_ex_inds=medium_ex_inds,
            carbon_ex_inds=carbon_ex_inds
        )

        # Save results
        output_file = os.path.join(args.output, 'wild_type_growth.npy')
        np.save(output_file, wild_type_growth)
        print(f"  Saved: {output_file}")
        # Also save as text for easy inspection
        output_txt = os.path.join(args.output, 'wild_type_growth.txt')
        with open(output_txt, 'w') as f:
            f.write("# Wild-Type Growth Rates (no gene knockouts)\n")
            f.write("# Carbon Source | Growth Rate\n")
            f.write("#" + "="*50 + "\n")
            for i, (carbon, growth) in enumerate(zip(name_carbon_model_matched_adj, wild_type_growth)):
                f.write(f"{carbon:<30} {growth:.6f}\n")
            f.write("\n# Summary Statistics\n")
            f.write(f"# Mean: {np.mean(wild_type_growth):.6f}\n")
            f.write(f"# Std:  {np.std(wild_type_growth):.6f}\n")
            f.write(f"# Min:  {np.min(wild_type_growth):.6f}\n")
            f.write(f"# Max:  {np.max(wild_type_growth):.6f}\n")
        print(f"  Saved: {output_txt}")

    elif args.mode == 'baseline':
        print("Running Baseline GEM (no enzyme constraints)...")
        # First, calculate wild-type growth for baseline model
        print("\n  Calculating baseline wild-type growth...")
        baseline_wildtype = simulate_wild_type_growth(
            model_adj=model_adj,
            name_carbon_model_matched_adj=name_carbon_model_matched_adj,
            medium_ex_inds=medium_ex_inds,
            carbon_ex_inds=carbon_ex_inds
        )
        # Save baseline wild-type growth
        wt_file = os.path.join(args.output, 'baseline_wildtype.npy')
        np.save(wt_file, baseline_wildtype)
        print(f"  Saved: {wt_file}")
        # Now run gene knockout simulations
        baseline_GEM = simulate_baseline_only(
            model_adj=model_adj,
            name_genes_matched_adj=name_genes_matched_adj,
            name_carbon_model_matched_adj=name_carbon_model_matched_adj,
            medium_ex_inds=medium_ex_inds,
            carbon_ex_inds=carbon_ex_inds
        )

        # Save results
        output_file = os.path.join(args.output, 'baseline_GEM.npy')
        np.save(output_file, baseline_GEM)
        print(f"  Saved: {output_file}")

    elif args.mode == 'pretuning':
        if not pre_tuning_data_path or not os.path.exists(pre_tuning_data_path):
            print(f"  ⚠️  Pre-tuning data not found: {pre_tuning_data_path}")
            sys.exit(1)

        print(f"Loading pre-tuning data from: {pre_tuning_data_path}")
        pre_tuning_df = pd.read_csv(pre_tuning_data_path)

        # First, calculate wild-type growth for pre-tuning kinGEMs model
        print("\n  Calculating pre-tuning kinGEMs wild-type growth...")
        pretuning_wildtype = simulate_wild_type_growth_kingems(
            model_adj=model_adj,
            name_carbon_model_matched_adj=name_carbon_model_matched_adj,
            medium_ex_inds=medium_ex_inds,
            carbon_ex_inds=carbon_ex_inds,
            processed_df=pre_tuning_df,
            objective_reaction=objective_reaction,
            enzyme_upper_bound=enzyme_upper_bound
        )
        # Save pre-tuning wild-type growth
        wt_file = os.path.join(args.output, 'pretuning_wildtype.npy')
        np.save(wt_file, pretuning_wildtype)
        print(f"  Saved: {wt_file}")

        # Now run gene knockout simulations
        if use_parallel:
            print("Running PARALLEL pre-tuning validation...")
            _, pre_tuning_GEM = simulate_phenotype_parallel(
                model_run=model_adj,
                name_genes_matched_adj=name_genes_matched_adj,
                name_carbon_model_matched_adj=name_carbon_model_matched_adj,
                medium_ex_inds=medium_ex_inds,
                carbon_ex_inds=carbon_ex_inds,
                processed_df=pre_tuning_df,
                objective_reaction=objective_reaction,
                enzyme_upper_bound=enzyme_upper_bound,
                n_workers=n_workers,
                chunk_size=chunk_size,
                method=parallel_method,
                skip_baseline=True,
                solver_name=solver_name if solver_name else 'glpk',
                remove_knockout_enzyme=remove_knockout_enzyme
            )
        else:
            print("Running SEQUENTIAL pre-tuning validation...")
            _, pre_tuning_GEM = simulate_phenotype(
                model_run=model_adj,
                name_genes_matched_adj=name_genes_matched_adj,
                name_carbon_model_matched_adj=name_carbon_model_matched_adj,
                medium_ex_inds=medium_ex_inds,
                carbon_ex_inds=carbon_ex_inds,
                processed_df=pre_tuning_df,
                objective_reaction=objective_reaction,
                enzyme_upper_bound=enzyme_upper_bound
            )

        # Save results
        output_file = os.path.join(args.output, 'pre_tuning_GEM.npy')
        np.save(output_file, pre_tuning_GEM)
        print(f"  Saved: {output_file}")

    elif args.mode == 'posttuning':
        if not post_tuning_data_path or not os.path.exists(post_tuning_data_path):
            print(f"  ⚠️  Post-tuning data not found: {post_tuning_data_path}")
            sys.exit(1)

        print(f"Loading post-tuning data from: {post_tuning_data_path}")
        post_tuning_df = pd.read_csv(post_tuning_data_path)

        # First, calculate wild-type growth for post-tuning kinGEMs model
        print("\n  Calculating post-tuning kinGEMs wild-type growth...")
        posttuning_wildtype = simulate_wild_type_growth_kingems(
            model_adj=model_adj,
            name_carbon_model_matched_adj=name_carbon_model_matched_adj,
            medium_ex_inds=medium_ex_inds,
            carbon_ex_inds=carbon_ex_inds,
            processed_df=post_tuning_df,
            objective_reaction=objective_reaction,
            enzyme_upper_bound=enzyme_upper_bound
        )
        # Save post-tuning wild-type growth
        wt_file = os.path.join(args.output, 'posttuning_wildtype.npy')
        np.save(wt_file, posttuning_wildtype)
        print(f"  Saved: {wt_file}")

        # Now run gene knockout simulations
        if use_parallel:
            print("Running PARALLEL post-tuning validation...")
            _, post_tuning_GEM = simulate_phenotype_parallel(
                model_run=model_adj,
                name_genes_matched_adj=name_genes_matched_adj,
                name_carbon_model_matched_adj=name_carbon_model_matched_adj,
                medium_ex_inds=medium_ex_inds,
                carbon_ex_inds=carbon_ex_inds,
                processed_df=post_tuning_df,
                objective_reaction=objective_reaction,
                enzyme_upper_bound=enzyme_upper_bound,
                n_workers=n_workers,
                chunk_size=chunk_size,
                method=parallel_method,
                skip_baseline=True,
                solver_name=solver_name if solver_name else 'glpk',
                remove_knockout_enzyme=remove_knockout_enzyme
            )
        else:
            print("Running SEQUENTIAL post-tuning validation...")
            _, post_tuning_GEM = simulate_phenotype(
                model_run=model_adj,
                name_genes_matched_adj=name_genes_matched_adj,
                name_carbon_model_matched_adj=name_carbon_model_matched_adj,
                medium_ex_inds=medium_ex_inds,
                carbon_ex_inds=carbon_ex_inds,
                processed_df=post_tuning_df,
                objective_reaction=objective_reaction,
                enzyme_upper_bound=enzyme_upper_bound
            )

        # Save results
        output_file = os.path.join(args.output, 'post_tuning_GEM.npy')
        np.save(output_file, post_tuning_GEM)
        print(f"  Saved: {output_file}")

    # Save experimental data and metadata
    exp_file = os.path.join(args.output, 'experimental_fitness.npy')
    meta_file = os.path.join(args.output, f'{args.mode}_metadata.json')

    if not os.path.exists(exp_file):
        np.save(exp_file, data_fitness_matched_adj)
        print(f"  Saved: {exp_file}")

    metadata = {
        'mode': args.mode,
        'model_name': model_name,
        'n_genes': len(name_genes_matched_adj),
        'n_carbons': len(name_carbon_model_matched_adj),
        'timestamp': datetime.now().isoformat(),
        'solver': current_solver,
        'parallel': use_parallel,
        'method': parallel_method if use_parallel else 'sequential',
        'workers': n_workers
    }

    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {meta_file}")

    print("\n" + "="*70)
    print(f"=== {args.mode.upper()} Validation Complete ===")
    print("="*70)
    print(f"Results saved to: {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()

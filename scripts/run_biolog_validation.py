#!/usr/bin/env python3
"""
Biolog Validation Analysis for ModelSEED Models
===============================================

This script performs a comprehensive Biolog validation analysis for any ModelSEED genome model:
1. Tunes the model based on a reference compound (default: glucose cpd00027) growth rate
2. Validates against all other compounds in the Biolog dataset
3. Performs regression analysis with multiple correlation metrics
4. Compares against experimental growth rate columns

Usage:
    python scripts/run_biolog_validation.py <config_file> [options]
    python scripts/run_biolog_validation.py configs/382_genome_cpd03198.json
    python scripts/run_biolog_validation.py configs/376_genome_cpd03198.json --reference-cpd cpd00027

Arguments:
    config_file: Path to JSON configuration file for the model

Options:
    --reference-cpd: Reference compound for tuning (default: from config or cpd00027)
    --glucose-target: Target growth rate for reference compound (if known)
    --solver: Solver to use (glpk, cplex, gurobi)
    --uptake-rate: Compound uptake rate for simulations
    --force, -f: Force regeneration of all intermediate files

Output:
    - Tuned model parameters
    - Growth rate predictions for all compounds
    - Correlation analysis (R², Pearson r, Spearman ρ, Kendall τ)
    - Regression plots and statistical summaries
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import cobra

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.dataset_modelseed import prepare_modelseed_model_data
from kinGEMs.dataset import (
    annotate_model_with_kcat_and_gpr,
    merge_substrate_sequences,
    process_kcat_predictions,
)
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.modeling.tuning import simulated_annealing
from kinGEMs.config import ensure_dir_exists, get_organism_info

# Suppress warnings
warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_biolog_data(experiments_file, sheet_name='Ecoli'):
    """Load Biolog experimental data."""
    print(f"Loading Biolog data from: {experiments_file}")
    print(f"  Sheet name: {sheet_name}")

    try:
        df = pd.read_excel(experiments_file, sheet_name=sheet_name, engine="openpyxl")
        print(f"  Loaded {len(df)} compounds")

        # Show available columns
        print("  Available columns:")
        for col in df.columns:
            print(f"    - {col}")

        return df

    except Exception as e:
        print(f"Error loading Biolog data: {e}")
        print("Available sheets:")
        try:
            xl_file = pd.ExcelFile(experiments_file)
            for sheet in xl_file.sheet_names:
                print(f"  - {sheet}")
        except Exception:
            pass
        raise


def simulate_growth_rate(model, processed_df, biomass_reaction, enzyme_upper_bound,
                        cpd_id, blocked_cpds, uptake_rate=10.0, solver_name='glpk'):
    """Simulate enzyme-constrained growth rate for a specific compound."""
    try:
        # Prepare medium conditions for optimization
        medium = {}

        # Block all compounds in blocked list
        for cpd in blocked_cpds:
            ex_rxn_id = f"EX_{cpd}_e0"
            if ex_rxn_id in [r.id for r in model.reactions]:
                medium[ex_rxn_id] = 0.0  # Block by setting uptake to 0

        # Set target compound uptake
        target_ex = f"EX_{cpd_id}_e0"
        if target_ex not in [r.id for r in model.reactions]:
            print(f"    Warning: Exchange reaction {target_ex} not found")
            return 0.0

        medium[target_ex] = -abs(uptake_rate)  # Set compound uptake rate

        # Run optimization with medium conditions
        sol_val, _, _, _ = run_optimization_with_dataframe(
            model=model,
            processed_df=processed_df,
            objective_reaction=biomass_reaction,
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
            solver_name=solver_name,
            medium=medium,
            medium_upper_bound=False
        )

        return sol_val if sol_val is not None else 0.0

    except Exception as e:
        print(f"    Error simulating {cpd_id}: {e}")
        return 0.0


def simulate_baseline_growth_rate(model, biomass_reaction, cpd_id, blocked_cpds, uptake_rate=10.0):
    """Simulate standard COBRApy FBA growth rate (no enzyme constraints) for a specific compound."""
    from copy import deepcopy

    try:
        # Create a copy of the model to avoid modifying the original
        test_model = deepcopy(model)

        # Block all compounds in blocked list
        for cpd in blocked_cpds:
            ex_rxn_id = f"EX_{cpd}_e0"
            if ex_rxn_id in test_model.reactions:
                test_model.reactions.get_by_id(ex_rxn_id).lower_bound = 0

        # Set target compound uptake
        target_ex = f"EX_{cpd_id}_e0"
        if target_ex not in test_model.reactions:
            print(f"    Warning: Exchange reaction {target_ex} not found")
            return 0.0

        test_model.reactions.get_by_id(target_ex).lower_bound = -abs(uptake_rate)

        # Set objective to biomass reaction
        test_model.objective = biomass_reaction

        # Run standard COBRApy FBA
        solution = test_model.optimize()

        return solution.objective_value if solution.status == 'optimal' else 0.0

    except Exception as e:
        print(f"    Error simulating baseline {cpd_id}: {e}")
        return 0.0


def calculate_correlation_metrics(x, y):
    """Calculate comprehensive correlation metrics."""
    # Convert to numpy arrays and handle mixed types
    x = np.array(x, dtype=object)
    y = np.array(y, dtype=object)

    # Create mask for valid numeric values
    x_numeric = []
    y_numeric = []

    for i in range(len(x)):
        try:
            # Try to convert both values to float
            x_val = float(x[i]) if x[i] is not None and str(x[i]).strip() != '' and str(x[i]).lower() != 'nan' else np.nan
            y_val = float(y[i]) if y[i] is not None and str(y[i]).strip() != '' and str(y[i]).lower() != 'nan' else np.nan

            # Only include if both are valid numbers
            if not (np.isnan(x_val) or np.isnan(y_val)):
                x_numeric.append(x_val)
                y_numeric.append(y_val)
        except (ValueError, TypeError):
            # Skip rows with non-numeric data
            continue

    x_clean = np.array(x_numeric)
    y_clean = np.array(y_numeric)

    if len(x_clean) < 2:
        return {
            'n_points': len(x_clean),
            'r2': np.nan,
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'kendall_tau': np.nan,
            'kendall_p': np.nan
        }

    # R-squared
    r2 = r2_score(x_clean, y_clean) if len(set(y_clean)) > 1 else np.nan

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)

    # Spearman correlation
    spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)

    # Kendall's tau
    kendall_tau, kendall_p = stats.kendalltau(x_clean, y_clean)

    return {
        'n_points': len(x_clean),
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p
    }


def create_correlation_plot(x, y, xlabel, ylabel, title, output_path, metrics):
    """Create a correlation plot with metrics."""
    plt.figure(figsize=(10, 8))

    # Convert to numpy arrays and handle mixed types
    x = np.array(x, dtype=object)
    y = np.array(y, dtype=object)

    # Create arrays for valid numeric values
    x_plot = []
    y_plot = []

    for i in range(len(x)):
        try:
            # Try to convert both values to float
            x_val = float(x[i]) if x[i] is not None and str(x[i]).strip() != '' and str(x[i]).lower() != 'nan' else np.nan
            y_val = float(y[i]) if y[i] is not None and str(y[i]).strip() != '' and str(y[i]).lower() != 'nan' else np.nan

            # Only include if both are valid numbers
            if not (np.isnan(x_val) or np.isnan(y_val)):
                x_plot.append(x_val)
                y_plot.append(y_val)
        except (ValueError, TypeError):
            # Skip rows with non-numeric data
            continue

    x_plot = np.array(x_plot)
    y_plot = np.array(y_plot)

    # Scatter plot
    plt.scatter(x_plot, y_plot, alpha=0.6, s=60)

    # Add trend line if possible
    if len(x_plot) >= 2 and len(set(y_plot)) > 1:
        z = np.polyfit(x_plot, y_plot, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(x_plot), max(x_plot), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

    # Labels and title
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Add metrics text box
    metrics_text = f"""n = {metrics['n_points']}
    R² = {metrics['r2']:.4f}
    Pearson r = {metrics['pearson_r']:.4f} (p = {metrics['pearson_p']:.4f})
    Spearman ρ = {metrics['spearman_rho']:.4f} (p = {metrics['spearman_p']:.4f})
    Kendall τ = {metrics['kendall_tau']:.4f} (p = {metrics['kendall_p']:.4f})"""

    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    ensure_dir_exists(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config_file', help='Path to JSON configuration file for the model')
    parser.add_argument('--reference-cpd', type=str, default=None,
                       help='Reference compound for tuning (default: from config or cpd00027)')
    parser.add_argument('--glucose-target', type=float, default=None,
                       help='Target growth rate for reference compound (if known)')
    parser.add_argument('--solver', default=None, help='Solver to use (default: from config)')
    parser.add_argument('--uptake-rate', type=float, default=None,
                       help='Compound uptake rate for simulations (default: from config)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force regeneration of all intermediate files')

    args = parser.parse_args()
    FORCE_REGENERATE = args.force

    # Load configuration
    print(f"Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)

    # Get organism information from the mapping
    model_name = config['model_name']
    organism_info = get_organism_info(model_name)

    # Override config with organism mapping if available
    if organism_info['organism'] != 'Unknown':
        config['organism'] = organism_info['organism']
        config['strain'] = organism_info['strain']

    # Override config with command line arguments
    if args.solver:
        config['solver'] = args.solver
    if args.uptake_rate:
        if 'biolog_validation' not in config:
            config['biolog_validation'] = {}
        config['biolog_validation']['uptake_rate'] = args.uptake_rate

    # Set defaults
    if 'solver' not in config:
        config['solver'] = 'glpk'
    if 'biolog_validation' not in config:
        config['biolog_validation'] = {}
    if 'uptake_rate' not in config['biolog_validation']:
        config['biolog_validation']['uptake_rate'] = 10.0
    if 'experiments_file' not in config['biolog_validation']:
        config['biolog_validation']['experiments_file'] = "data/Biolog experiments/FBA_results.xlsx"

    # Use organism mapping for sheet name if available
    if 'sheet_name' not in config['biolog_validation']:
        if organism_info['biolog_sheet'] != 'Unknown':
            config['biolog_validation']['sheet_name'] = organism_info['biolog_sheet']
        else:
            config['biolog_validation']['sheet_name'] = config.get('organism', 'Ecoli')

    # Determine reference compound
    reference_cpd = args.reference_cpd
    if not reference_cpd:
        reference_cpd = config['biolog_validation'].get('reference_compound', 'cpd00027')

    print(f"Model: {config['model_name']}")
    print(f"Organism: {config.get('organism', 'Unknown')}")
    print(f"Strain: {config.get('strain', 'Unknown')}")
    print(f"Biolog sheet: {config['biolog_validation']['sheet_name']}")
    print(f"Reference compound: {reference_cpd}")
    print(f"Solver: {config['solver']}")

    # Generate run ID
    run_id = f"{config['model_name']}_biolog_validation_{datetime.today().strftime('%Y%m%d')}_{random.randint(1000, 9999)}"

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_name = config["model_name"]
    data_dir = os.path.join(project_root, "data")
    raw_data_dir = os.path.join(data_dir, "raw")
    interim_data_dir = os.path.join(data_dir, "interim", model_name)
    processed_data_dir = os.path.join(data_dir, "processed", model_name)
    results_dir = os.path.join(project_root, "results", "biolog_validation", run_id)
    ensure_dir_exists(results_dir)
    ensure_dir_exists(interim_data_dir)
    ensure_dir_exists(processed_data_dir)

    # File paths
    model_path = os.path.join(raw_data_dir, f"{model_name}.xml")
    if not os.path.exists(model_path):
        # Try models directory
        model_path = os.path.join(project_root, "models", f"{model_name}.xml")

    # Intermediate data paths
    substrates_output = os.path.join(interim_data_dir, f"{model_name}_substrates.csv")
    sequences_output = os.path.join(interim_data_dir, f"{model_name}_sequences.csv")
    merged_data_output = os.path.join(interim_data_dir, f"{model_name}_merged_data.csv")
    processed_data_output = os.path.join(processed_data_dir, f"{model_name}_processed_data.csv")

    predictions_path = os.path.join(data_dir, "interim", "CPI-Pred predictions",
                                   f"X06A_kinGEMs_{model_name}_predictions.csv")

    print("\n" + "="*80)
    print("=== Biolog Validation Analysis ===")
    print("="*80)
    print(f"Run ID: {run_id}")
    print(f"Model: {model_name}")
    print(f"Reference compound: {reference_cpd}")
    print(f"Solver: {config['solver']}")
    print(f"Results directory: {results_dir}")
    if FORCE_REGENERATE:
        print("⚠️  Force regenerate mode: will regenerate all intermediate files")
    print("="*80)

    # === Step 1: Load and prepare model ===
    print("\n=== Step 1: Loading and preparing model ===")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Searched locations:")
        print(f"  - {os.path.join(raw_data_dir, f'{model_name}.xml')}")
        print(f"  - {os.path.join(project_root, 'models', f'{model_name}.xml')}")
        sys.exit(1)

    model = cobra.io.read_sbml_model(model_path)
    print(f"  Loaded model: {len(model.genes)} genes, {len(model.reactions)} reactions")

    # Determine biomass reaction
    biomass_reaction = config.get('biomass_reaction')
    if not biomass_reaction:
        obj_rxns = {rxn.id: rxn.objective_coefficient for rxn in model.reactions if rxn.objective_coefficient != 0}
        biomass_reaction = next(iter(obj_rxns.keys())) if obj_rxns else 'bio1'
    print(f"  Biomass reaction: {biomass_reaction}")

    # === Step 2: Prepare model data ===
    print("\n=== Step 2: Preparing model data ===")

    metadata_dir = config.get("metadata_dir", "data/Biolog experiments")
    metadata_path = os.path.join(project_root, metadata_dir)

    if not FORCE_REGENERATE and os.path.exists(substrates_output) and os.path.exists(sequences_output):
        print("  ✓ Found existing files, loading cached data:")
        print(f"    - {substrates_output}")
        print(f"    - {sequences_output}")
        substrates_df = pd.read_csv(substrates_output)
        sequences_df = pd.read_csv(sequences_output)
        # Model was already loaded in Step 1
    else:
        if FORCE_REGENERATE:
            print("  ⟳ Regenerating model data (--force flag)")
        else:
            print("  No cached files found, preparing model data...")

        model, substrates_df, sequences_df = prepare_modelseed_model_data(
            model_path=model_path,
            substrates_output=substrates_output,
            sequences_output=sequences_output,
            organism=config.get("organism", "Unknown"),
            metadata_dir=metadata_path
        )
        print("  ✓ Generated and saved substrates and sequences")

    print(f"  Substrates: {len(substrates_df)} rows")
    print(f"  Sequences: {len(sequences_df)} rows")

    # Merge data
    print("\n=== Step 3: Merging substrate and sequence data ===")
    if not FORCE_REGENERATE and os.path.exists(merged_data_output):
        print("  ✓ Found existing file, loading cached data:")
        print(f"    - {merged_data_output}")
        merged_data = pd.read_csv(merged_data_output)
    else:
        if FORCE_REGENERATE:
            print("  ⟳ Regenerating merged data (--force flag)")
        else:
            print("  No cached file found, merging data...")
        merged_data = merge_substrate_sequences(substrates_df, sequences_df)
        # Save merged data
        merged_data.to_csv(merged_data_output, index=False)
        print("  ✓ Generated and saved merged data")

    print(f"  Merged data: {len(merged_data)} rows")

    # === Step 3: Process kcat predictions ===
    print("\n=== Step 4: Processing kcat predictions ===")

    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found: {predictions_path}")
        print(f"Expected location: {predictions_path}")
        sys.exit(1)

    if not FORCE_REGENERATE and os.path.exists(processed_data_output):
        print("  ✓ Found existing file, loading cached data:")
        print(f"    - {processed_data_output}")
        processed_data = pd.read_csv(processed_data_output)
    else:
        if FORCE_REGENERATE:
            print("  ⟳ Regenerating processed data (--force flag)")
        else:
            print("  No cached file found, processing kcat predictions...")
        processed_data = process_kcat_predictions(merged_data, predictions_path)
        # Save processed data
        processed_data.to_csv(processed_data_output, index=False)
        print("  ✓ Generated and saved processed data")

    print(f"  Processed data: {len(processed_data)} rows")

    # Ensure kcat column exists
    if 'kcat_mean' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_mean']

    # Annotate model
    model = annotate_model_with_kcat_and_gpr(model=model, df=processed_data)
    rxn_with_kcat = sum(1 for rxn in model.reactions
                       if hasattr(rxn, 'annotation') and 'kcat' in rxn.annotation
                       and rxn.annotation['kcat'] not in [None, '', 0, '0'])
    print(f"  Reactions with kcat: {rxn_with_kcat}/{len(model.reactions)}")

    # === Step 4: Load Biolog experimental data ===
    print("\n=== Step 5: Loading Biolog experimental data ===")

    biolog_file = config["biolog_validation"]["experiments_file"]
    biolog_path = os.path.join(project_root, biolog_file)
    sheet_name = config["biolog_validation"]["sheet_name"]

    if not os.path.exists(biolog_path):
        print(f"Error: Biolog file not found: {biolog_path}")
        sys.exit(1)

    exp_df = load_biolog_data(biolog_path, sheet_name)

    # Identify experimental growth rate columns
    growth_rate_columns = []
    for col in exp_df.columns:
        if 'growth rate' in col.lower() or 'od600' in col.lower() or 'od750' in col.lower():
            growth_rate_columns.append(col)

    print(f"  Found experimental growth rate columns: {growth_rate_columns}")

    # === Step 5: Get reference compound experimental target ===
    print(f"\n=== Step 6: Determining {reference_cpd} target growth rate ===")

    # Try different possible column names for compound ID
    cpd_column = None
    possible_cpd_columns = ['cpd', 'modelseed ID', 'compound_id', 'metabolite_id', 'ID']

    for col in possible_cpd_columns:
        if col in exp_df.columns:
            cpd_column = col
            break

    if cpd_column is None:
        print("Error: No compound ID column found in experimental data")
        print("Available columns:")
        for col in exp_df.columns:
            print(f"  - {col}")
        sys.exit(1)

    print(f"  Using compound ID column: {cpd_column}")
    reference_row = exp_df[exp_df[cpd_column] == reference_cpd]

    if len(reference_row) == 0:
        print(f"Error: Reference compound {reference_cpd} not found in experimental data")
        print("Available compounds:")
        print(exp_df[cpd_column].head(10).tolist())
        sys.exit(1)

    reference_targets = {}
    for col in growth_rate_columns:
        if col in reference_row.columns:
            reference_targets[col] = reference_row[col].iloc[0]
            print(f"  Reference target for {col}: {reference_targets[col]}")

    # Use the target for tuning
    if args.glucose_target:
        tuning_target = args.glucose_target
        print(f"  Using user-specified reference target: {tuning_target}")
        experimental_target = tuning_target  # Store for reporting
    else:
        experimental_target = list(reference_targets.values())[0] if reference_targets else 1.0
        # Use a realistic metabolic model target (1.0) instead of experimental values
        tuning_target = 1.0
        print(f"  Experimental reference value: {experimental_target}")
        print(f"  Using realistic model biomass target: {tuning_target}")

    # === Step 6: Run simulated annealing ===
    print(f"\n=== Step 7: Running simulated annealing ({reference_cpd} tuning) ===")

    sa_config = config.get("simulated_annealing", {
        "temperature": 1.0,
        "cooling_rate": 0.95,
        "min_temperature": 0.01,
        "max_iterations": 100,
        "max_unchanged_iterations": 4,
        "change_threshold": 0.009,
        "n_top_enzymes": 65,
        "verbose": False
    })
    sa_config["biomass_goal"] = tuning_target

    print(f"  Tuning target ({reference_cpd} growth rate): {tuning_target}")
    print(f"  Temperature: {sa_config['temperature']}")
    print(f"  Max iterations: {sa_config['max_iterations']}")
    print(f"  Top enzymes to modify: {sa_config['n_top_enzymes']}")

    # Set up reference compound environment for tuning
    tuning_model = model.copy()
    blocked_cpds = config["biolog_validation"].get("blocked_compounds", [])

    # Block all compounds except reference compound
    for cpd in blocked_cpds:
        if cpd != reference_cpd:  # Don't block reference compound
            ex_rxn_id = f"EX_{cpd}_e0"
            if ex_rxn_id in tuning_model.reactions:
                tuning_model.reactions.get_by_id(ex_rxn_id).lower_bound = 0

    # Set reference compound uptake
    reference_ex = f"EX_{reference_cpd}_e0"
    if reference_ex in tuning_model.reactions:
        tuning_model.reactions.get_by_id(reference_ex).lower_bound = -abs(config["biolog_validation"]["uptake_rate"])

    # Check initial growth rate before tuning
    print(f"  Checking initial {reference_cpd} growth rate before tuning...")
    initial_growth_rate = simulate_growth_rate(
        model=model,
        processed_df=processed_data,
        biomass_reaction=biomass_reaction,
        enzyme_upper_bound=config.get("enzyme_upper_bound", 0.15),
        cpd_id=reference_cpd,
        blocked_cpds=blocked_cpds,
        uptake_rate=config["biolog_validation"]["uptake_rate"],
        solver_name=config['solver']
    )

    # Check baseline COBRApy FBA growth rate (no enzyme constraints)
    baseline_growth_rate = simulate_baseline_growth_rate(
        model=model,
        biomass_reaction=biomass_reaction,
        cpd_id=reference_cpd,
        blocked_cpds=blocked_cpds,
        uptake_rate=config["biolog_validation"]["uptake_rate"]
    )

    print(f"  Initial kinGEMs {reference_cpd} growth rate: {initial_growth_rate:.4f}")
    print(f"  Baseline COBRApy {reference_cpd} growth rate: {baseline_growth_rate:.4f}")
    print(f"  Target {reference_cpd} growth rate: {tuning_target:.4f}")
    print(f"  Initial accuracy gap: {abs(initial_growth_rate - tuning_target):.4f}")
    print(f"  Baseline accuracy gap: {abs(baseline_growth_rate - tuning_target):.4f}")

    # Run simulated annealing
    kcat_dict, top_targets, tuned_df, iterations, biomasses, df_FBA = simulated_annealing(
        model=tuning_model,
        processed_data=processed_data,
        biomass_reaction=biomass_reaction,
        objective_value=tuning_target,
        gene_sequences_dict={},  # Will be computed internally
        output_dir=results_dir,
        enzyme_fraction=config.get("enzyme_upper_bound", 0.15),
        temperature=sa_config['temperature'],
        cooling_rate=sa_config['cooling_rate'],
        min_temperature=sa_config['min_temperature'],
        max_iterations=sa_config['max_iterations'],
        max_unchanged_iterations=sa_config['max_unchanged_iterations'],
        change_threshold=sa_config['change_threshold'],
        n_top_enzymes=sa_config['n_top_enzymes'],
        verbose=sa_config['verbose'],
    )

    print("  Simulated annealing completed")
    print(f"  Final {reference_cpd} growth rate: {biomasses[-1]:.4f}")
    print(f"  Target {reference_cpd} growth rate: {tuning_target:.4f}")
    print(f"  Tuning accuracy: {abs(biomasses[-1] - tuning_target):.4f}")

    # === Step 7: Validate on all compounds ===
    print("\n=== Step 8: Running validation on all compounds ===")

    validation_results = []
    n_compounds = len(exp_df)

    for i, row in exp_df.iterrows():
        cpd_id = row[cpd_column]
        print(f"  Progress: {i+1}/{n_compounds} - Testing {cpd_id}", end='\r')

        # Simulate growth rate with tuned parameters
        predicted_rate = simulate_growth_rate(
            model=model,
            processed_df=tuned_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=config.get("enzyme_upper_bound", 0.15),
            cpd_id=cpd_id,
            blocked_cpds=blocked_cpds,
            uptake_rate=config["biolog_validation"]["uptake_rate"],
            solver_name=config['solver']
        )

        # Simulate baseline COBRApy FBA growth rate (no enzyme constraints)
        baseline_rate = simulate_baseline_growth_rate(
            model=model,
            biomass_reaction=biomass_reaction,
            cpd_id=cpd_id,
            blocked_cpds=blocked_cpds,
            uptake_rate=config["biolog_validation"]["uptake_rate"]
        )

        # Collect experimental values
        result = {
            'cpd': cpd_id,
            'predicted_growth_rate': predicted_rate,
            'baseline_growth_rate': baseline_rate
        }

        for col in growth_rate_columns:
            if col in row:
                result[f'exp_{col}'] = row[col]

        validation_results.append(result)

    print(f"\n  Validation completed for {len(validation_results)} compounds")

    # Convert to DataFrame
    validation_df = pd.DataFrame(validation_results)

    # === Step 8: Correlation analysis ===
    print("\n=== Step 9: Correlation analysis ===")

    correlation_results = {}
    baseline_correlation_results = {}

    for col in growth_rate_columns:
        exp_col = f'exp_{col}'
        if exp_col in validation_df.columns:
            print(f"\n  Analyzing correlation with {col}:")

            x = validation_df[exp_col].values
            y = validation_df['predicted_growth_rate'].values
            y_baseline = validation_df['baseline_growth_rate'].values

            # Calculate metrics for kinGEMs
            metrics = calculate_correlation_metrics(x, y)
            correlation_results[col] = metrics

            # Calculate metrics for baseline COBRApy
            baseline_metrics = calculate_correlation_metrics(x, y_baseline)
            baseline_correlation_results[col] = baseline_metrics

            # Print results
            print("    kinGEMs Results:")
            print(f"      n = {metrics['n_points']}")
            print(f"      R² = {metrics['r2']:.4f}")
            print(f"      Pearson r = {metrics['pearson_r']:.4f} (p = {metrics['pearson_p']:.4f})")
            print(f"      Spearman ρ = {metrics['spearman_rho']:.4f} (p = {metrics['spearman_p']:.4f})")
            print(f"      Kendall τ = {metrics['kendall_tau']:.4f} (p = {metrics['kendall_p']:.4f})")

            print("    Baseline COBRApy Results:")
            print(f"      n = {baseline_metrics['n_points']}")
            print(f"      R² = {baseline_metrics['r2']:.4f}")
            print(f"      Pearson r = {baseline_metrics['pearson_r']:.4f} (p = {baseline_metrics['pearson_p']:.4f})")
            print(f"      Spearman ρ = {baseline_metrics['spearman_rho']:.4f} (p = {baseline_metrics['spearman_p']:.4f})")
            print(f"      Kendall τ = {baseline_metrics['kendall_tau']:.4f} (p = {baseline_metrics['kendall_p']:.4f})")

            # Create correlation plot for kinGEMs
            plot_path = os.path.join(results_dir, f"correlation_kinGEMs_{col.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}.png")
            create_correlation_plot(
                x=x, y=y,
                xlabel=f"Experimental {col}",
                ylabel="Predicted Growth Rate (1/hr)",
                title=f"{model_name}: kinGEMs {reference_cpd}-Tuned vs {col}",
                output_path=plot_path,
                metrics=metrics
            )
            print(f"    kinGEMs plot saved: {plot_path}")

            # Create correlation plot for baseline
            baseline_plot_path = os.path.join(results_dir, f"correlation_baseline_{col.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}.png")
            create_correlation_plot(
                x=x, y=y_baseline,
                xlabel=f"Experimental {col}",
                ylabel="Baseline Growth Rate (1/hr)",
                title=f"{model_name}: Baseline COBRApy vs {col}",
                output_path=baseline_plot_path,
                metrics=baseline_metrics
            )
            print(f"    Baseline plot saved: {baseline_plot_path}")

    # === Step 9: Summary and best correlation ===
    print("\n=== Step 10: Summary of results ===")

    # Find best correlation by R² for kinGEMs
    best_col = None
    best_r2 = -np.inf

    for col, metrics in correlation_results.items():
        if not np.isnan(metrics['r2']) and metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_col = col

    # Find best correlation by R² for baseline
    baseline_best_col = None
    baseline_best_r2 = -np.inf

    for col, metrics in baseline_correlation_results.items():
        if not np.isnan(metrics['r2']) and metrics['r2'] > baseline_best_r2:
            baseline_best_r2 = metrics['r2']
            baseline_best_col = col

    print(f"\n  kinGEMs best correlation found with: {best_col}")
    print(f"  kinGEMs best R² value: {best_r2:.4f}")
    print(f"  Baseline best correlation found with: {baseline_best_col}")
    print(f"  Baseline best R² value: {baseline_best_r2:.4f}")

    if best_r2 > baseline_best_r2:
        print(f"  🏆 kinGEMs outperforms baseline by {best_r2 - baseline_best_r2:.4f} R²")
    elif baseline_best_r2 > best_r2:
        print(f"  📊 Baseline outperforms kinGEMs by {baseline_best_r2 - best_r2:.4f} R²")
    else:
        print("  🤝 kinGEMs and baseline show similar performance")

    # Save detailed results
    validation_df.to_csv(os.path.join(results_dir, "validation_results.csv"), index=False)

    # Save correlation summaries
    correlation_summary = pd.DataFrame(correlation_results).T
    correlation_summary.to_csv(os.path.join(results_dir, "correlation_summary_kinGEMs.csv"))

    baseline_correlation_summary = pd.DataFrame(baseline_correlation_results).T
    baseline_correlation_summary.to_csv(os.path.join(results_dir, "correlation_summary_baseline.csv"))

    # Save configuration and parameters
    with open(os.path.join(results_dir, "analysis_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Save tuning results summary
    tuning_summary = {
        'model_name': model_name,
        'reference_compound': reference_cpd,
        'experimental_target': experimental_target,
        'model_target_growth_rate': tuning_target,
        'initial_growth_rate': initial_growth_rate,
        'baseline_growth_rate': baseline_growth_rate,
        'final_growth_rate': biomasses[-1],
        'tuning_accuracy': abs(biomasses[-1] - tuning_target),
        'iterations': len(biomasses),
        'kinGEMs_best_correlation_column': best_col,
        'kinGEMs_best_r2': best_r2,
        'baseline_best_correlation_column': baseline_best_col,
        'baseline_best_r2': baseline_best_r2
    }

    with open(os.path.join(results_dir, "tuning_summary.json"), 'w') as f:
        json.dump(tuning_summary, f, indent=2)

    print(f"\n  Results saved to: {results_dir}")
    print("  - validation_results.csv: All compound predictions (kinGEMs + baseline)")
    print("  - correlation_summary_kinGEMs.csv: Statistical metrics for kinGEMs predictions")
    print("  - correlation_summary_baseline.csv: Statistical metrics for baseline predictions")
    print("  - correlation_kinGEMs_*.png: kinGEMs scatter plots with trend lines")
    print("  - correlation_baseline_*.png: Baseline scatter plots with trend lines")
    print("  - analysis_config.json: Analysis configuration")
    print("  - tuning_summary.json: Tuning results summary with comparison")

    print("\n" + "="*80)
    print("=== Biolog Validation Analysis Complete ===")
    print("="*80)


if __name__ == '__main__':
    main()

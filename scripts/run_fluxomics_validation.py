#!/usr/bin/env python3
"""
kinGEMs Fluxomics Validation Script
====================================

This script validates metabolic flux predictions from ecGEMs constructed with kinGEMs
against experimental flux data (e.g., from 13C-MFA studies).

Usage:
    python scripts/ru    # === Run 2: Modified kinGEMs (enzyme-constrained, experimental medium) ===
    print("\n=== Step 4: Running modified kinGEMs optimization ===")
    print(f"  (Enzyme-constrained FBA with experimental medium: {medium_id})")
    
    # Get mediu        fig, result        # 2. Modified model
        print("\n  [2/4] Modified kinGEMs model...")
        modified_exp_validation_df = compare_fluxomics(
            fba_results_path=out_path_modified,
            exp_fluxes_path=experimental_fluxes_path
        )
        
        fig, results_modified = plot_flux_correlation(
            modified_exp_validation_df, 'exp_flux', 'FBA_flux',
            output_path=os.path.join(exp_output_dir, f"02_correlation_kinGEMs_exp_medium_{medium_id}_vs_experimental.png"),
            show=False,
            biomass2=solution_value_modified
        )
        plt.close()plot_flux_co        fig, results_cobrapy = plot_flux_correlation(
            cobrapy_exp_validation_df, 'exp_flux', 'FBA_flux',
            output_path=os.path.join(exp_output_dir, f"03_correlation_COBRApy_FBA_{medium_id}_vs_experimental.png"),
            show=False
        )
        plt.close()
        
        fig, diff_data_cobrapy = plot_flux_differences(
            cobrapy_exp_validation_df, 'exp_flux', 'FBA_flux', top_n=15,
            difference_type='absolute',
            output_path=os.path.join(exp_output_dir, f"03_differences_COBRApy_FBA_{medium_id}_vs_experimental.png"),
            show=False
        )
        plt.close()          modified_exp_validation_df, 'exp_flux', 'FBA_flux',
            output_path=os.path.join(exp_output_dir, f"02_correlation_kinGEMs_exp_medium_{modified_medium_id}_vs_experimental.png"),
            show=False
        )
        plt.close()
        
        fig, diff_data_modified = plot_flux_differences(
            modified_exp_validation_df, 'exp_flux', 'FBA_flux', top_n=15,
            difference_type='absolute',
            output_path=os.path.join(exp_output_dir, f"02_differences_kinGEMs_exp_medium_{modified_medium_id}_vs_experimental.png"),
            show=False
        )
        plt.close()
    medium = get_medium_dict(medium_id=medium_id)
    print(f"  Medium composition: {medium}")
    
    out_path_modified = os.path.join(exp_output_dir, f"fluxes_kinGEMs_exp_medium_{medium_id}.csv")cs_validation.py <config_file> [-    out_path_cobrapy = os.path.join(exp_output_dir, "fluxes_cobrapy.csv")
    
    # Initialize variable
    cobrapy_biomass = None
    
    if not FORCE_REGENERATE and os.path.exists(out_path_cobrapy):
        print(f"  Loading existing results from: {out_path_cobrapy}")
        cobrapy_fluxes_df = pd.read_csv(out_path_cobrapy)
        if biomass_reaction in cobrapy_fluxes_df['Index'].values:
            cobrapy_biomass = cobrapy_fluxes_df[cobrapy_fluxes_df['Index'] == biomass_reaction]['Value'].values[0]
        else:
            # If biomass not found, regenerate
            print("  Biomass reaction not found in existing file, regenerating...")
            cobrapy_biomass = None
    
    if cobrapy_biomass is None:python scripts/run_fluxomics_validation.py configs/fluxomics_iML1515_GEM.json
    python scripts/run_fluxomics_validation.py configs/fluxomics_iML1515_GEM.json --force

Arguments:
    config_file: Path to JSON configuration file
    --force, -f: Force regeneration of all intermediate files

Config File Format:
    See configs/fluxomics_iML1515_GEM.json for an example

The script performs:
1. Standard FBA (COBRApy, no enzyme constraints)
2. Enzyme-constrained FBA with kinGEMs (regular medium)
3. Enzyme-constrained FBA with kinGEMs (modified medium matching experimental)
4. Simulated annealing tuning
5. Comparison of all models against experimental fluxes
6. Correlation plots and difference analysis
"""

import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime

import cobra as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cobra.io import write_sbml_model

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.dataset import annotate_model_with_kcat_and_gpr, process_kcat_predictions
from kinGEMs.fluxomics_validation import (
    compare_fluxomics,
)
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.modeling.tuning import simulated_annealing
from kinGEMs.plots import (
    plot_flux_correlation,
    plot_flux_differences,
    kingems_cobrapy_dataframe,
)

# Suppress warnings and configure logging
warnings.filterwarnings('ignore')
logging.getLogger('distributed').setLevel(logging.ERROR)
try:
    import gurobipy
    gurobipy.setParam('OutputFlag', 0)
except ImportError:
    pass


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def determine_biomass_reaction(model):
    """Automatically determine the biomass reaction from model objective."""
    obj_rxns = {rxn.id: rxn.objective_coefficient
                for rxn in model.reactions
                if rxn.objective_coefficient != 0}
    if not obj_rxns:
        raise ValueError("No objective reaction found in model")
    return next(iter(obj_rxns.keys()))


def get_medium_from_config(config, use_sa_medium=False):
    """
    Get medium composition from config file.
    
    Parameters:
        config: dict
            The loaded configuration dictionary
        use_sa_medium: bool
            If True, use simulated annealing medium, otherwise use experimental validation medium
    
    Returns:
        dict: Medium composition mapping exchange reaction IDs to bounds
    """
    if use_sa_medium:
        return config.get('simulated_annealing', {}).get('medium', {})
    else:
        exp_medium = config.get('experimental_validation', {}).get('medium', {})
        if not exp_medium:
            # Fallback to simulated annealing medium
            exp_medium = config.get('simulated_annealing', {}).get('medium', {})
        return exp_medium


def apply_medium_to_model(model, medium_dict, stress='none'):
    """
    Apply medium composition to a cobra model.
    
    Parameters:
        model: cobra.Model
            The model to modify
        medium_dict: dict
            Dictionary mapping exchange reaction IDs to their bounds
        stress: str
            Stress condition (currently supports 'none', 'NADH-limitation', 'ATP-limitation')
    
    Returns:
        model: cobra.Model
            Modified model with applied medium
    """
    print("Applying medium composition from config")
    
    # Apply medium bounds
    for rxn_id, uptake in medium_dict.items():
        try:
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = uptake
            rxn.upper_bound = uptake
            print(f"Set {rxn_id} bounds to {uptake}")
        except KeyError:
            print(f"WARNING: Reaction {rxn_id} not found in model.")
    
    # Apply stress conditions if needed
    if stress == 'NADH-limitation':
        try:
            nadh_dehyd = model.reactions.get_by_id("NADH16pp")
            nadh_dehyd.upper_bound = 3
            print("Applied NADH dehydrogenase limitation")
        except KeyError:
            print("WARNING: NADH16pp reaction not found for NADH limitation")
    elif stress == 'ATP-limitation':
        try:
            atp_synth = model.reactions.get_by_id("ATPS4rpp")
            atp_synth.upper_bound = 3
            print("Applied ATP synthase limitation")
        except KeyError:
            print("WARNING: ATPS4rpp reaction not found for ATP limitation")
    
    return model


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config_path = sys.argv[1]
    FORCE_REGENERATE = '--force' in sys.argv or '-f' in sys.argv

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Extract configuration
    model_name = config['model_name']
    organism = config.get('organism', 'Unknown')
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.3)
    solver_name = config.get('solver', 'glpk')

    # Generate run ID
    run_id = f"{model_name}_{datetime.today().strftime('%Y%m%d')}_{random.randint(1000, 9999)}"

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")
    processed_data_dir = os.path.join(data_dir, "processed")
    experimental_data_dir = os.path.join(data_dir, "experimental")
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    fluxomics_validation_dir = os.path.join(results_dir, "fluxomics_validation")
    exp_validation_dir = os.path.join(fluxomics_validation_dir, "experimental_validation")
    tuning_results_dir = os.path.join(fluxomics_validation_dir, "tuning_results", run_id)
    
    # Create output directories
    os.makedirs(tuning_results_dir, exist_ok=True)

    # File paths - use model naming convention
    # For iML1515_GEM, processed data is in ecoli_iML1515 folder
    if 'iML1515' in model_name:
        processed_subdir = 'ecoli_iML1515'
    elif 'e_coli_core' in model_name:
        processed_subdir = 'ecoli_core'
    else:
        processed_subdir = model_name.replace('_GEM', '')
    
    model_xml = os.path.join(models_dir, f"{model_name}.xml")
    processed_data_csv = os.path.join(processed_data_dir, processed_subdir, f"{processed_subdir}_processed_data.csv")

    print("\n" + "="*70)
    print(f"=== kinGEMs Fluxomics Validation for {model_name} ===")
    print("="*70)
    print(f"Run ID: {run_id}")
    print(f"Organism: {organism}")
    print(f"Results directory: {tuning_results_dir}")
    print(f"Solver: {solver_name}")
    if FORCE_REGENERATE:
        print("Force regeneration: ENABLED")
    print("="*70)

    # Print configuration summary
    print("\n=== Configuration Summary ===")
    print(f"Config file: {config_path}")
    print(f"Model name: {model_name}")
    print(f"Model path: {model_xml}")
    print(f"Processed data: {processed_data_csv}")
    print(f"Enzyme upper bound: {enzyme_upper_bound}")

    # Verify files exist
    if not os.path.exists(model_xml):
        raise FileNotFoundError(f"Model file not found: {model_xml}")
    if not os.path.exists(processed_data_csv):
        raise FileNotFoundError(f"Processed data file not found: {processed_data_csv}")

    # Load experimental validation settings
    exp_config = config.get('experimental_validation', {})
    experiment_name = exp_config.get('experiment', 'W3110_MD121_M9+Glu_none')
    medium_id = exp_config.get('medium_id', 'MD121')
    stress = exp_config.get('stress', 'none')
    
    # Create experiment-specific output directory
    exp_output_dir = os.path.join(exp_validation_dir, experiment_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    print("\nExperimental Validation Settings:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Medium ID: {medium_id}")
    print(f"  Stress condition: {stress}")
    print(f"  Output directory: {exp_output_dir}")

    # Experimental flux file
    experimental_fluxes_path = os.path.join(experimental_data_dir, f"fluxes_{experiment_name}.csv")
    if not os.path.exists(experimental_fluxes_path):
        print(f"\nWARNING: Experimental flux file not found: {experimental_fluxes_path}")
        print("Validation will proceed without experimental comparison.")
        has_experimental_data = False
    else:
        print(f"  Experimental fluxes: {experimental_fluxes_path}")
        has_experimental_data = True

    # Print simulated annealing config
    sa_config = config.get('simulated_annealing', {})
    print("\nSimulated Annealing Settings:")
    print(f"  Temperature: {sa_config.get('temperature', 1.0)}")
    print(f"  Cooling rate: {sa_config.get('cooling_rate', 0.975)}")
    print(f"  Min temperature: {sa_config.get('min_temperature', 0.001)}")
    print(f"  Max iterations: {sa_config.get('max_iterations', 100)}")
    print(f"  Max unchanged iterations: {sa_config.get('max_unchanged_iterations', 5)}")
    print(f"  Change threshold: {sa_config.get('change_threshold', 0.009)}")
    print(f"  Biomass goal: {sa_config.get('biomass_goal', 0.87)}")

    print("="*70)

    # === Load Model and Data ===
    print("\n=== Step 1: Loading model and processed data ===")
    processed_data = pd.read_csv(processed_data_csv)
    model = cb.io.read_sbml_model(model_xml)
    
    # Determine biomass reaction
    biomass_reaction = config.get('biomass_reaction') or determine_biomass_reaction(model)
    print(f"  Model: {len(model.genes)} genes, {len(model.reactions)} reactions")
    print(f"  Biomass reaction: {biomass_reaction}")
    print(f"  Processed data: {len(processed_data)} rows")

    # Diagnostic: Check kcat coverage
    rxn_with_kcat = []
    rxn_without_kcat = []
    for rxn in model.reactions:
        ann = rxn.annotation if hasattr(rxn, 'annotation') else {}
        if 'kcat' in ann and ann['kcat'] not in [None, '', 0, '0']:
            rxn_with_kcat.append(rxn.id)
        else:
            rxn_without_kcat.append(rxn.id)
    print(f"  Initial kcat coverage: {len(rxn_with_kcat)}/{len(model.reactions)} reactions")

    # === Annotate Model ===
    print("\n=== Step 2: Annotating model with kcat and GPR ===")
    model = annotate_model_with_kcat_and_gpr(model=model, df=processed_data)
    
    # Check kcat coverage after annotation
    rxn_with_kcat = sum(1 for rxn in model.reactions
                        if hasattr(rxn, 'annotation') and 'kcat' in rxn.annotation
                        and rxn.annotation['kcat'] not in [None, '', 0, '0'])
    print(f"  Reactions with kcat after annotation: {rxn_with_kcat}/{len(model.reactions)}")

    # === Run 1: Regular kinGEMs (enzyme-constrained, default medium) ===
    print("\n=== Step 3: Running regular kinGEMs optimization ===")
    print("  (Enzyme-constrained FBA with default medium)")
    
    out_path_regular = os.path.join(exp_output_dir, "fluxes_kinGEMs_default_medium.csv")
    
    if not FORCE_REGENERATE and os.path.exists(out_path_regular):
        print(f"  Loading existing results from: {out_path_regular}")
        df_FBA_regular = pd.read_csv(out_path_regular)
        if 'Biomass' in df_FBA_regular.columns:
            solution_value_regular = df_FBA_regular['Biomass'].iloc[0]
        elif 'Solution Biomass' in df_FBA_regular.columns:
            solution_value_regular = df_FBA_regular['Solution Biomass'].iloc[0]
        else:
            # If biomass column not found, regenerate
            print("  Biomass column not found in existing file, regenerating...")
            FORCE_REGENERATE = True
    
    if FORCE_REGENERATE or not os.path.exists(out_path_regular):
        (solution_value_regular, df_FBA_regular, gene_sequences_dict_regular, _) = run_optimization_with_dataframe(
            model=model,
            processed_df=processed_data,
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
            solver_name=solver_name
        )
        df_FBA_regular.to_csv(out_path_regular, index=False)
        print(f"  Saved results to: {out_path_regular}")
    
    print(f"  Biomass value: {solution_value_regular:.4f}")

    # === Run 2: Modified kinGEMs (enzyme-constrained, experimental medium) ===
    print("\n=== Step 4: Running modified kinGEMs optimization ===")
    print(f"  (Enzyme-constrained FBA with experimental medium: {medium_id})")
    
    # Get medium composition from config
    medium = get_medium_from_config(config, use_sa_medium=False)
    print(f"  Medium composition: {medium}")
    
    out_path_modified = os.path.join(exp_output_dir, "df_FBA_modified_model.csv")
    
    # Initialize variable
    solution_value_modified = None
    
    if not FORCE_REGENERATE and os.path.exists(out_path_modified):
        print(f"  Loading existing results from: {out_path_modified}")
        df_FBA_modified = pd.read_csv(out_path_modified)
        if 'Biomass' in df_FBA_modified.columns:
            solution_value_modified = df_FBA_modified['Biomass'].iloc[0]
        elif 'Solution Biomass' in df_FBA_modified.columns:
            solution_value_modified = df_FBA_modified['Solution Biomass'].iloc[0]
        else:
            # If biomass column not found, regenerate
            print("  Biomass column not found in existing file, regenerating...")
            solution_value_modified = None
    
    if solution_value_modified is None:
        # Create a fresh model copy for modified medium
        mod_model = cb.io.read_sbml_model(model_xml)
        mod_annotated_model = annotate_model_with_kcat_and_gpr(model=mod_model, df=processed_data)
        
        (solution_value_modified, df_FBA_modified, gene_sequences_dict_modified, _) = run_optimization_with_dataframe(
            model=mod_annotated_model,
            processed_df=processed_data,
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
            medium=medium
        )
        df_FBA_modified.to_csv(out_path_modified, index=False)
        print(f"  Saved results to: {out_path_modified}")
    
    print(f"  Biomass value: {solution_value_modified:.4f}")

    # === Run 3: COBRApy FBA (no enzyme constraints, experimental medium) ===
    print("\n=== Step 5: Running COBRApy FBA for comparison ===")
    print(f"  (Standard FBA with experimental medium: {medium_id})")
    
    out_path_cobrapy = os.path.join(exp_output_dir, f"fluxes_COBRApy_FBA_{medium_id}.csv")
    
    # Initialize variable
    cobrapy_biomass = None
    
    if not FORCE_REGENERATE and os.path.exists(out_path_cobrapy):
        print(f"  Loading existing results from: {out_path_cobrapy}")
        cobrapy_fluxes_df = pd.read_csv(out_path_cobrapy)
        if biomass_reaction in cobrapy_fluxes_df['Index'].values:
            cobrapy_biomass = cobrapy_fluxes_df[cobrapy_fluxes_df['Index'] == biomass_reaction]['Value'].values[0]
        else:
            # If biomass not found, regenerate
            print("  Biomass reaction not found in existing file, regenerating...")
            cobrapy_biomass = None
    
    if cobrapy_biomass is None:
        # Apply medium changes from config
        cobrapy_model = cb.io.read_sbml_model(model_xml)
        cobrapy_medium = get_medium_from_config(config, use_sa_medium=False)
        cobrapy_model = apply_medium_to_model(cobrapy_model, cobrapy_medium, stress)
        
        # Run COBRApy FBA
        cobrapy_solution = cobrapy_model.optimize()
        cobrapy_biomass = cobrapy_solution.objective_value
        
        # Save fluxes
        cobrapy_fluxes_df = pd.DataFrame({
            "Variable": "flux",
            "Index": cobrapy_solution.fluxes.index,
            "Value": cobrapy_solution.fluxes.values
        })
        cobrapy_fluxes_df.to_csv(out_path_cobrapy, index=False)
        print(f"  Saved results to: {out_path_cobrapy}")
    
    print(f"  Biomass value (FBA): {cobrapy_biomass:.4f}")

    # === Run 4: Simulated Annealing ===
    print("\n=== Step 6: Running simulated annealing ===")
    
    out_path_modified_tuned = os.path.join(exp_output_dir, f"fluxes_kinGEMs_tuned_{medium_id}.csv")
    
    # Initialize variable
    solution_value_tuned = None
    
    if not FORCE_REGENERATE and os.path.exists(out_path_modified_tuned):
        print(f"  Loading existing tuned results from: {out_path_modified_tuned}")
        df_FBA_modified_tuned = pd.read_csv(out_path_modified_tuned)
        if 'Biomass' in df_FBA_modified_tuned.columns:
            solution_value_tuned = df_FBA_modified_tuned['Biomass'].iloc[0]
        elif 'Solution Biomass' in df_FBA_modified_tuned.columns:
            solution_value_tuned = df_FBA_modified_tuned['Solution Biomass'].iloc[0]
        else:
            solution_value_tuned = None
        if solution_value_tuned is not None:
            print(f"  Tuned biomass value: {solution_value_tuned:.4f}")
    
    if solution_value_tuned is None:
        # Create fresh model for tuning
        tuning_model = cb.io.read_sbml_model(model_xml)
        tuning_model = annotate_model_with_kcat_and_gpr(model=tuning_model, df=processed_data)
        
        # Get initial solution with modified medium
        (solution_value_init, _, gene_sequences_dict_init, _) = run_optimization_with_dataframe(
            model=tuning_model,
            processed_df=processed_data,
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
            medium=medium
        )
        
        # Simulated annealing configuration
        temperature = sa_config.get('temperature', 1.0)
        cooling_rate = sa_config.get('cooling_rate', 0.975)
        min_temperature = sa_config.get('min_temperature', 0.001)
        max_iterations = sa_config.get('max_iterations', 100)
        max_unchanged_iterations = sa_config.get('max_unchanged_iterations', 5)
        change_threshold = sa_config.get('change_threshold', 0.009)
        biomass_goal = sa_config.get('biomass_goal', 0.87)
        
        print("  Configuration:")
        print(f"    - Temperature: {temperature}")
        print(f"    - Cooling rate: {cooling_rate}")
        print(f"    - Max iterations: {max_iterations}")
        print(f"    - Biomass goal: {biomass_goal}")
        
        kcat_dict, top_targets, df_new, iterations, biomasses, df_FBA_modified_tuned = simulated_annealing(
            model=tuning_model,
            processed_data=processed_data,
            biomass_reaction=biomass_reaction,
            objective_value=biomass_goal,
            gene_sequences_dict=gene_sequences_dict_init,
            output_dir=tuning_results_dir,
            enzyme_fraction=enzyme_upper_bound,
            temperature=temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            max_iterations=max_iterations,
            max_unchanged_iterations=max_unchanged_iterations,
            change_threshold=change_threshold,
            medium=medium
        )
        
        # Extract the final biomass value
        solution_value_tuned = biomasses[-1]
        
        print(f"  Final biomass: {solution_value_tuned:.4f}")
        print(f"  Improvement: {(biomasses[-1] - biomasses[0]) / biomasses[0] * 100:.1f}%")
        print("  Top 10 enzymes by mass contribution:")
        print(top_targets[['Reactions', 'Single_gene', 'enzyme_mass']].head(10))
        
        # Save tuned results
        df_FBA_modified_tuned.to_csv(out_path_modified_tuned, index=False)
        print(f"  Saved tuned results to: {out_path_modified_tuned}")

    # === Summary of All Models ===
    print("\n" + "="*70)
    print("=== Model Performance Summary ===")
    print("="*70)
    print(f"COBRApy FBA (no constraints):         {cobrapy_biomass:.4f}")
    print(f"kinGEMs regular (default medium):     {solution_value_regular:.4f}")
    print(f"kinGEMs modified (exp medium):        {solution_value_modified:.4f}")
    if 'solution_value_tuned' in locals():
        print(f"kinGEMs tuned (exp medium):           {solution_value_tuned:.4f}")
    print("="*70)

    # === Experimental Validation ===
    if has_experimental_data:
        print("\n=== Step 7: Comparing against experimental fluxes ===")
        
        results_summary = []
        
        # Helper function to calculate RMSE
        def calculate_rmse(df, exp_col, pred_col):
            """Calculate RMSE between experimental and predicted fluxes."""
            df_clean = df.dropna(subset=[exp_col, pred_col])
            if len(df_clean) == 0:
                return np.nan
            return np.sqrt(np.mean((df_clean[exp_col] - df_clean[pred_col]) ** 2))
        
        # 1. Regular model
        print("\n  [1/4] Regular kinGEMs model...")
        regular_exp_validation_df = compare_fluxomics(
            fba_results_path=out_path_regular,
            exp_fluxes_path=experimental_fluxes_path
        )
        
        fig, results_regular = plot_flux_correlation(
            regular_exp_validation_df, 'exp_flux', 'FBA_flux',
            output_path=os.path.join(exp_output_dir, "01_correlation_kinGEMs_default_vs_experimental.png"),
            show=False,
            biomass2=solution_value_regular
        )
        plt.close()
        
        fig, diff_data_regular = plot_flux_differences(
            regular_exp_validation_df, 'exp_flux', 'FBA_flux', top_n=15,
            difference_type='absolute',
            output_path=os.path.join(exp_output_dir, "01_differences_kinGEMs_default_vs_experimental.png"),
            show=False
        )
        plt.close()
        
        # Calculate RMSE
        results_regular['rmse'] = calculate_rmse(regular_exp_validation_df, 'exp_flux', 'FBA_flux')
        
        results_summary.append({
            'Model': 'Regular kinGEMs',
            **results_regular
        })
        print(f"    R²: {results_regular['r_squared']:.3f}, RMSE: {results_regular['rmse']:.3f}")
        
        # 2. Modified model
        print("\n  [2/4] Modified kinGEMs model...")
        modified_exp_validation_df = compare_fluxomics(
            fba_results_path=out_path_modified,
            exp_fluxes_path=experimental_fluxes_path
        )
        
        fig, results_modified = plot_flux_correlation(
            modified_exp_validation_df, 'exp_flux', 'FBA_flux',
            output_path=os.path.join(exp_output_dir, "correlation_modified.png"),
            show=False
        )
        plt.close()
        
        fig, diff_data_modified = plot_flux_differences(
            modified_exp_validation_df, 'exp_flux', 'FBA_flux', top_n=15,
            difference_type='absolute',
            output_path=os.path.join(exp_output_dir, "differences_modified.png"),
            show=False
        )
        plt.close()
        
        # Calculate RMSE
        results_modified['rmse'] = calculate_rmse(modified_exp_validation_df, 'exp_flux', 'FBA_flux')
        
        results_summary.append({
            'Model': 'Modified kinGEMs',
            **results_modified
        })
        print(f"    R²: {results_modified['r_squared']:.3f}, RMSE: {results_modified['rmse']:.3f}")
        
        # 3. COBRApy model
        print("\n  [3/4] COBRApy FBA model...")
        cobrapy_exp_validation_df = compare_fluxomics(
            fba_results_path=out_path_cobrapy,
            exp_fluxes_path=experimental_fluxes_path
        )
        
        fig, results_cobrapy = plot_flux_correlation(
            cobrapy_exp_validation_df, 'exp_flux', 'FBA_flux',
            output_path=os.path.join(exp_output_dir, f"03_correlation_COBRApy_FBA_{medium_id}_vs_experimental.png"),
            show=False,
            biomass2=cobrapy_biomass
        )
        plt.close()
        
        fig, diff_data_cobrapy = plot_flux_differences(
            cobrapy_exp_validation_df, 'exp_flux', 'FBA_flux', top_n=15,
            difference_type='absolute',
            output_path=os.path.join(exp_output_dir, f"03_differences_COBRApy_FBA_{medium_id}_vs_experimental.png"),
            show=False
        )
        plt.close()
        
        # Calculate RMSE
        results_cobrapy['rmse'] = calculate_rmse(cobrapy_exp_validation_df, 'exp_flux', 'FBA_flux')
        
        results_summary.append({
            'Model': 'COBRApy FBA',
            **results_cobrapy
        })
        print(f"    R²: {results_cobrapy['r_squared']:.3f}, RMSE: {results_cobrapy['rmse']:.3f}")
        
        # 4. Tuned model (if available)
        if os.path.exists(out_path_modified_tuned):
            print("\n  [4/4] Tuned kinGEMs model...")
            modified_tuned_exp_validation_df = compare_fluxomics(
                fba_results_path=out_path_modified_tuned,
                exp_fluxes_path=experimental_fluxes_path
            )
            
            fig, results_tuned = plot_flux_correlation(
                modified_tuned_exp_validation_df, 'exp_flux', 'FBA_flux',
                output_path=os.path.join(exp_output_dir, f"04_correlation_kinGEMs_tuned_{medium_id}_vs_experimental.png"),
                show=False,
                biomass2=solution_value_tuned
            )
            plt.close()
            
            fig, diff_data_tuned = plot_flux_differences(
                modified_tuned_exp_validation_df, 'exp_flux', 'FBA_flux', top_n=15,
                difference_type='absolute',
                output_path=os.path.join(exp_output_dir, f"04_differences_kinGEMs_tuned_{medium_id}_vs_experimental.png"),
                show=False
            )
            plt.close()
            
            # Calculate RMSE
            results_tuned['rmse'] = calculate_rmse(modified_tuned_exp_validation_df, 'exp_flux', 'FBA_flux')
            
            results_summary.append({
                'Model': 'Tuned kinGEMs',
                **results_tuned
            })
            print(f"    R²: {results_tuned['r_squared']:.3f}, RMSE: {results_tuned['rmse']:.3f}")
        
        # Save summary
        summary_df = pd.DataFrame(results_summary)
        summary_path = os.path.join(exp_output_dir, "validation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Saved validation summary to: {summary_path}")
        
        # Print summary table
        print("\n" + "="*70)
        print("=== Experimental Validation Summary ===")
        print("="*70)
        print(summary_df.to_string(index=False))
        print("="*70)

    # === kinGEMs vs COBRApy Comparison ===
    print("\n=== Step 8: Comparing kinGEMs vs COBRApy ===")
    
    # 1. Modified kinGEMs vs COBRApy
    print("\n  [1/2] Modified kinGEMs vs COBRApy...")
    cobrapy_kingems_df = kingems_cobrapy_dataframe(
        kingems_path=out_path_modified,
        fba_path=out_path_cobrapy
    )
    
    fig, results_vs_cobrapy = plot_flux_correlation(
        cobrapy_kingems_df, 'cobrapy_flux', 'kinGEMs_flux',
        output_path=os.path.join(exp_output_dir, f"05_correlation_kinGEMs_exp_medium_vs_COBRApy_{medium_id}.png"),
        show=False,
        biomass1=cobrapy_biomass,
        biomass2=solution_value_modified
    )
    plt.close()
    
    fig, diff_data_vs_cobrapy = plot_flux_differences(
        cobrapy_kingems_df, 'cobrapy_flux', 'kinGEMs_flux', top_n=15,
        difference_type='absolute',
        output_path=os.path.join(exp_output_dir, f"05_differences_kinGEMs_exp_medium_vs_COBRApy_{medium_id}.png"),
        show=False
    )
    plt.close()
    
    # Calculate RMSE
    results_vs_cobrapy['rmse'] = calculate_rmse(cobrapy_kingems_df, 'cobrapy_flux', 'kinGEMs_flux')
    
    print(f"    R²: {results_vs_cobrapy['r_squared']:.3f}, RMSE: {results_vs_cobrapy['rmse']:.3f}")
    
    # 2. Tuned kinGEMs vs COBRApy (if available)
    if os.path.exists(out_path_modified_tuned):
        print("\n  [2/2] Tuned kinGEMs vs COBRApy...")
        cobrapy_kingems_tuned_df = kingems_cobrapy_dataframe(
            kingems_path=out_path_modified_tuned,
            fba_path=out_path_cobrapy
        )
        
        fig, results_tuned_vs_cobrapy = plot_flux_correlation(
            cobrapy_kingems_tuned_df, 'cobrapy_flux', 'kinGEMs_flux',
            output_path=os.path.join(exp_output_dir, f"06_correlation_kinGEMs_tuned_vs_COBRApy_{medium_id}.png"),
            show=False,
            biomass1=cobrapy_biomass,
            biomass2=solution_value_tuned
        )
        plt.close()
        
        fig, diff_data_tuned_vs_cobrapy = plot_flux_differences(
            cobrapy_kingems_tuned_df, 'cobrapy_flux', 'kinGEMs_flux', top_n=15,
            difference_type='absolute',
            output_path=os.path.join(exp_output_dir, f"06_differences_kinGEMs_tuned_vs_COBRApy_{medium_id}.png"),
            show=False
        )
        plt.close()
        
        # Calculate RMSE
        results_tuned_vs_cobrapy['rmse'] = calculate_rmse(cobrapy_kingems_tuned_df, 'cobrapy_flux', 'kinGEMs_flux')
        
        print(f"    R²: {results_tuned_vs_cobrapy['r_squared']:.3f}, RMSE: {results_tuned_vs_cobrapy['rmse']:.3f}")

    # === Final Summary ===
    print("\n" + "="*70)
    print("=== Fluxomics Validation Complete ===")
    print("="*70)
    print(f"Results saved to: {exp_output_dir}")
    print("\nGenerated files:")
    print("  Flux Data:")
    print("    - fluxes_kinGEMs_default_medium.csv")
    print(f"    - fluxes_kinGEMs_exp_medium_{medium_id}.csv")
    print(f"    - fluxes_COBRApy_FBA_{medium_id}.csv")
    if os.path.exists(out_path_modified_tuned):
        print(f"    - fluxes_kinGEMs_tuned_{medium_id}.csv")
    
    if has_experimental_data:
        print("\n  Experimental Validation:")
        print("    - validation_summary.csv")
        print("    - 01_correlation_kinGEMs_default_vs_experimental.png")
        print("    - 01_differences_kinGEMs_default_vs_experimental.png")
        print(f"    - 02_correlation_kinGEMs_exp_medium_{medium_id}_vs_experimental.png")
        print(f"    - 02_differences_kinGEMs_exp_medium_{medium_id}_vs_experimental.png")
        print(f"    - 03_correlation_COBRApy_FBA_{medium_id}_vs_experimental.png")
        print(f"    - 03_differences_COBRApy_FBA_{medium_id}_vs_experimental.png")
        if os.path.exists(out_path_modified_tuned):
            print(f"    - 04_correlation_kinGEMs_tuned_{medium_id}_vs_experimental.png")
            print(f"    - 04_differences_kinGEMs_tuned_{medium_id}_vs_experimental.png")
    
    print("\n  Model Comparisons:")
    print(f"    - 05_correlation_kinGEMs_exp_medium_vs_COBRApy_{medium_id}.png")
    print(f"    - 05_differences_kinGEMs_exp_medium_vs_COBRApy_{medium_id}.png")
    if os.path.exists(out_path_modified_tuned):
        print(f"    - 06_correlation_kinGEMs_tuned_vs_COBRApy_{medium_id}.png")
        print(f"    - 06_differences_kinGEMs_tuned_vs_COBRApy_{medium_id}.png")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

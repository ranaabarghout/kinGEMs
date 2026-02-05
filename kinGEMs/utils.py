"""
Utility functions for kinGEMs scripts.

This module contains helper functions for:
- Configuration loading
- Model type detection
- File finding
- Flux metrics calculation
- FVA execution at different constraint levels
- Loading existing results
"""

import glob
import json
import os

from cobra.flux_analysis import flux_variability_analysis as cobra_fva
import pandas as pd

from kinGEMs.modeling.fva import (
    flux_variability_analysis,
    flux_variability_analysis_parallel,
)


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


def calculate_flux_metrics(fva_df):
    """Calculate both Flux Variability (FVi) and Flux Variability Range (FVR).

    FVi = (max - min) / (max + min + ε) - Relative variability (0-1 scale) for reaction i
    FVR = |max - min| - Absolute flux range

    Returns:
        tuple: (fvi, fvr) as pandas Series
    """
    # Import from plots module to avoid duplication
    from kinGEMs.plots import calculate_flux_metrics as calc_flux_metrics
    return calc_flux_metrics(fva_df)


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
                          use_parallel=False, n_workers=None, method='dask', chunk_size=None, constrain_biomass=True):
    """Level 2: Single enzyme constraint only (basic enzyme constraints without complex features)."""
    print("\n=== Level 2: Single enzyme constraint only ===")
    print("  (multi_enzyme_off=False, isoenzymes_off=True, promiscuous_off=True, complexes_off=True)")
    print("  Only basic enzyme-reaction constraints (v_j ≤ kcat_j * e_i)")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, processed_df_result, df_FBA = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            method=method,
            chunk_size=chunk_size,
            multi_enzyme_off=False,  # Enable basic multi-enzyme (single/complex)
            isoenzymes_off=True,     # Disable isoenzymes (OR-GPR)
            promiscuous_off=True,    # Disable promiscuous enzyme sharing
            complexes_off=True,      # Disable complex enzyme logic
            constrain_biomass=constrain_biomass
        )
        # Extract biomass value from the FVA results
        biomass = fva_df['Solution Biomass'].iloc[0]
    else:
        fva_df, processed_df_result, df_FBA = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=False,  # Enable basic multi-enzyme (single/complex)
            isoenzymes_off=True,     # Disable isoenzymes (OR-GPR)
            promiscuous_off=True,    # Disable promiscuous enzyme sharing
            complexes_off=True,      # Disable complex enzyme logic
            constrain_biomass=constrain_biomass
        )
        # Extract biomass value from the FVA results
        biomass = fva_df['Solution Biomass'].iloc[0]

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_isoenzymes_fva(model, processed_df, biomass_reaction, enzyme_upper_bound,
                       use_parallel=False, n_workers=None, method='dask', chunk_size=None, constrain_biomass=True):
    """Level 3a: Basic enzyme constraints + isoenzymes (OR-GPR handling)."""
    print("\n=== Level 3a: + Isoenzymes ===")
    print("  (multi_enzyme_off=False, isoenzymes_off=False, promiscuous_off=True, complexes_off=True)")
    print("  Basic constraints + OR-GPR handling for alternative enzymes")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, processed_df_result, df_FBA = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            method=method,
            chunk_size=chunk_size,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=False,    # Enable isoenzymes (OR-GPR)
            promiscuous_off=True,    # Disable promiscuous enzyme sharing
            complexes_off=True,      # Disable complex enzyme logic
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]
    else:
        fva_df, processed_df_result, df_FBA = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=False,    # Enable isoenzymes (OR-GPR)
            promiscuous_off=True,    # Disable promiscuous enzyme sharing
            complexes_off=True,      # Disable complex enzyme logic
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_complexes_fva(model, processed_df, biomass_reaction, enzyme_upper_bound,
                      use_parallel=False, n_workers=None, method='dask', chunk_size=None, constrain_biomass=True):
    """Level 3b: Basic enzyme constraints + enzyme complexes (AND-GPR handling)."""
    print("\n=== Level 3b: + Enzyme Complexes ===")
    print("  (multi_enzyme_off=False, isoenzymes_off=True, promiscuous_off=True, complexes_off=False)")
    print("  Basic constraints + AND-GPR handling for enzyme complexes")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, processed_df_result, df_FBA = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            method=method,
            chunk_size=chunk_size,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=True,     # Disable isoenzymes (OR-GPR)
            promiscuous_off=True,    # Disable promiscuous enzyme sharing
            complexes_off=False,     # Enable complex enzyme logic (AND-GPR)
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]
    else:
        fva_df, processed_df_result, df_FBA = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=True,     # Disable isoenzymes (OR-GPR)
            promiscuous_off=True,    # Disable promiscuous enzyme sharing
            complexes_off=False,     # Enable complex enzyme logic (AND-GPR)
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_promiscuous_fva(model, processed_df, biomass_reaction, enzyme_upper_bound,
                       use_parallel=False, n_workers=None, method='dask', chunk_size=None, constrain_biomass=True):
    """Level 3c: Basic enzyme constraints + promiscuous enzymes (cross-reaction sharing)."""
    print("\n=== Level 3c: + Promiscuous Enzymes ===")
    print("  (multi_enzyme_off=False, isoenzymes_off=True, promiscuous_off=False, complexes_off=True)")
    print("  Basic constraints + enzyme sharing across multiple reactions")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, processed_df_result, df_FBA = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            method=method,
            chunk_size=chunk_size,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=True,     # Disable isoenzymes (OR-GPR)
            promiscuous_off=False,   # Enable promiscuous enzyme sharing
            complexes_off=True,      # Disable complex enzyme logic
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]
    else:
        fva_df, processed_df_result, df_FBA = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=True,     # Disable isoenzymes (OR-GPR)
            promiscuous_off=False,   # Enable promiscuous enzyme sharing
            complexes_off=True,      # Disable complex enzyme logic
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_all_constraints_fva(model, processed_df, biomass_reaction, enzyme_upper_bound,
                           use_parallel=False, n_workers=None, method='dask', chunk_size=None, constrain_biomass=True):
    """Level 4: All constraints (complete kinGEMs implementation)."""
    print("\n=== Level 4: All Constraints ===")
    print("  (multi_enzyme_off=False, isoenzymes_off=False, promiscuous_off=False, complexes_off=False)")
    print("  Complete kinGEMs: basic + isoenzymes + complexes + promiscuous enzymes")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, processed_df_result, df_FBA = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            method=method,
            chunk_size=chunk_size,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=False,    # Enable isoenzymes (OR-GPR)
            promiscuous_off=False,   # Enable promiscuous enzyme sharing
            complexes_off=False,     # Enable complex enzyme logic (AND-GPR)
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]
    else:
        fva_df, processed_df_result, df_FBA = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=False,    # Enable isoenzymes (OR-GPR)
            promiscuous_off=False,   # Enable promiscuous enzyme sharing
            complexes_off=False,     # Enable complex enzyme logic (AND-GPR)
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def run_tuned_fva(model, tuned_df, biomass_reaction, enzyme_upper_bound,
                 use_parallel=False, n_workers=None, method='dask', chunk_size=None, constrain_biomass=True):
    """Level 5: Post-tuned kinGEMs (complete constraints with optimized kcat values)."""
    print("\n=== Level 5: Post-Tuned (Simulated Annealing) ===")
    print("  (multi_enzyme_off=False, isoenzymes_off=False, promiscuous_off=False, complexes_off=False)")
    print("  Complete kinGEMs with tuned kcat values from simulated annealing")

    if use_parallel:
        print(f"  Starting parallel FVA with {len(model.reactions)} reactions...")
        fva_df, processed_df_result, df_FBA = flux_variability_analysis_parallel(
            model=model,
            processed_df=tuned_df,
            biomass_reaction=biomass_reaction,
            output_file=None,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers,
            method=method,
            chunk_size=chunk_size,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=False,    # Enable isoenzymes (OR-GPR)
            promiscuous_off=False,   # Enable promiscuous enzyme sharing
            complexes_off=False,     # Enable complex enzyme logic (AND-GPR)
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]
    else:
        fva_df, processed_df_result, df_FBA = flux_variability_analysis(
            model=model,
            processed_df=tuned_df,
            biomass_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            multi_enzyme_off=False,  # Enable basic multi-enzyme
            isoenzymes_off=False,    # Enable isoenzymes (OR-GPR)
            promiscuous_off=False,   # Enable promiscuous enzyme sharing
            complexes_off=False,     # Enable complex enzyme logic (AND-GPR)
            constrain_biomass=constrain_biomass
        )
        biomass = fva_df['Solution Biomass'].iloc[0]

    print(f"  Biomass: {biomass:.4f}")
    print(f"  Reactions analyzed: {len(fva_df)}")

    return fva_df, biomass


def load_existing_results(results_dir):
    """Load existing FVA results from CSV files in results directory.

    Parameters
    ----------
    results_dir : str
        Path to directory containing FVA result CSV files

    Returns
    -------
    tuple
        (fva_results dict, biomass_values dict)
    """
    fva_results = {}
    biomass_values = {}

    # Mapping of filenames to level names
    level_files = {
        'level1_baseline_irreversible.csv': 'Level 1: Baseline GEM',
        'level2_single_enzyme.csv': 'Level 2: Single Enzyme',
        'level3a_isoenzymes.csv': 'Level 3a: + Isoenzymes',
        'level3b_complexes.csv': 'Level 3b: + Complexes',
        'level3c_promiscuous.csv': 'Level 3c: + Promiscuous',
        'level4_all_constraints.csv': 'Level 4: All Constraints',
        'level5_post_tuned.csv': 'Level 5: Post-Tuned',
    }

    print("\n=== Loading Existing FVA Results ===")

    for filename, level_name in level_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            fva_results[level_name] = df

            # Extract biomass from the dataframe
            if 'Solution Biomass' in df.columns:
                biomass = df['Solution Biomass'].iloc[0]
            else:
                biomass = 0.0
            biomass_values[level_name] = biomass

            print(f"  ✓ Loaded {level_name}: {len(df)} reactions, biomass={biomass:.4f}")
        else:
            print(f"  ⚠ Not found: {filename}")

    if not fva_results:
        raise FileNotFoundError(f"No FVA result files found in {results_dir}")

    print(f"\n  Loaded {len(fva_results)} levels")

    return fva_results, biomass_values

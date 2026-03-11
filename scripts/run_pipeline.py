#!/usr/bin/env python3
"""
kinGEMs General Pipeline Script
================================

This script provides a unified pipeline for processing any genome-scale metabolic model (GEM).
It automatically detects whether to use ModelSEED or standard dataset functions based on the
model filename pattern.

Usage:
    python scripts/run_pipeline.py <config_file> [--force]
    python scripts/run_pipeline.py configs/iML1515_GEM.json
    python scripts/run_pipeline.py configs/382_genome_cpd03198.json --force

Arguments:
    config_file: Path to JSON configuration file
    --force, -f: Force regeneration of all intermediate files

Model Type Detection:
    - Models with '_genome_' in filename → Use dataset_modelseed functions
    - Other models → Use standard dataset functions

Config File Format:
    See example configs in the configs/ directory
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import random
import sys
import time
import warnings

import cobra
from cobra.flux_analysis import flux_variability_analysis as cobra_fva
from cobra.io import write_sbml_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.dataset import (
    annotate_model_with_kcat_and_gpr,
    assign_kcats_to_model,
    load_model,
    merge_substrate_sequences,
    prepare_model_data,
    process_kcat_predictions,
    convert_to_irreversible
)
from kinGEMs.dataset_modelseed import prepare_modelseed_model_data
from kinGEMs.modeling.fva import (
    flux_variability_analysis,
    flux_variability_analysis_parallel,
    plot_flux_variability,
)
from kinGEMs.plots import (
    plot_cumulative_fvi_distribution,
    plot_kcat_annealing_comparison,
    plot_kcat_annealing_comparison_by_subsystem,
)
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.modeling.tuning import simulated_annealing, sweep_maintenance_parameters

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


def configure_solver():
    """
    Configure the optimization solver.
    Try Gurobi first, fall back to GLPK if Gurobi is not available.
    Returns the solver name to use.
    """
    # Try Gurobi
    try:
        import gurobipy
        # Test if we can actually use it
        test_env = gurobipy.Env()
        test_env.dispose()
        print("  Using Gurobi solver")
        return 'gurobi'
    except Exception as e:
        print(f"  Gurobi not available: {e}")
        print("  Falling back to GLPK solver")
        return 'glpk'


def find_predictions_file(model_name, CPIPred_data_dir):
    """
    Find the CPI-Pred predictions file for a given model.
    Tries multiple naming patterns to handle inconsistencies.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'iML1515_GEM', '382_genome_cpd03198')
    CPIPred_data_dir : str
        Directory containing CPI-Pred predictions

    Returns
    -------
    str
        Path to the predictions file

    Raises
    ------
    FileNotFoundError
        If no predictions file is found
    """
    import glob

    # Try multiple naming patterns
    patterns = [
        f"X06A_kinGEMs_{model_name}_predictions.csv",  # Direct match
        f"*{model_name}*predictions.csv",  # Fuzzy match
        f"*{model_name.replace('_GEM', '')}*predictions.csv",  # Without _GEM suffix
        f"*ecoli*{model_name.split('_')[0]}*predictions.csv",  # E.coli specific patterns
    ]

    # Also try common substitutions
    # iML1515_GEM -> ecoli_iML1515
    if '_GEM' in model_name:
        base_name = model_name.replace('_GEM', '')
        patterns.append(f"*ecoli_{base_name}*predictions.csv")
        patterns.append(f"*{base_name}*predictions.csv")

    # e_coli_core -> ecoli_core
    if 'e_coli' in model_name:
        ecoli_variant = model_name.replace('e_coli', 'ecoli')
        patterns.append(f"*{ecoli_variant}*predictions.csv")

    for pattern in patterns:
        search_path = os.path.join(CPIPred_data_dir, pattern)
        matches = glob.glob(search_path)
        if matches:
            # Return the first match
            print(f"  Found predictions file: {os.path.basename(matches[0])}")
            return matches[0]

    # If no file found, list available files to help user
    available_files = glob.glob(os.path.join(CPIPred_data_dir, "*.csv"))
    if available_files:
        print(f"\n  ⚠️  No predictions file found for '{model_name}'")
        print("  Available prediction files:")
        for f in available_files:
            print(f"    - {os.path.basename(f)}")
        print("\n  Please ensure CPI-Pred predictions exist for this model.")

    raise FileNotFoundError(
        f"No CPI-Pred predictions file found for model '{model_name}' in {CPIPred_data_dir}"
    )


def is_modelseed_model(model_name):
    """Detect if model should use ModelSEED functions based on naming pattern."""
    return '_genome_' in model_name.lower()


def determine_biomass_reaction(model):
    """Automatically determine the biomass reaction from model objective."""
    obj_rxns = {rxn.id: rxn.objective_coefficient
                for rxn in model.reactions
                if rxn.objective_coefficient != 0}
    if not obj_rxns:
        raise ValueError("No objective reaction found in model")
    return next(iter(obj_rxns.keys()))


def clean_annotations(model):
    """Convert float values in annotations to strings for SBML compatibility."""
    for rxn in model.reactions:
        ann = rxn.annotation
        if not isinstance(ann, dict):
            rxn.annotation = {}
        else:
            new_ann = {}
            for k, v in ann.items():
                if isinstance(v, float):
                    new_ann[k] = str(v)
                elif isinstance(v, (list, tuple)):
                    new_ann[k] = [str(item) if isinstance(item, float) else item for item in v]
                elif isinstance(v, (str, dict)):
                    new_ann[k] = v
            rxn.annotation = new_ann
    return model


def simulate_enzyme_rate(base_model, processed_df, biomass_reaction, blocked_cpds,
                        cpd_id, enzyme_upper_bound, uptake_rate=10.0, solver_name='glpk'):
    """Simulate enzyme-constrained growth rate for a specific substrate."""
    from copy import deepcopy
    mdl = deepcopy(base_model)

    # Block all other compounds
    for cpd in blocked_cpds:
        if cpd.lower() == cpd_id.lower():
            continue
        ex_name = f"EX_{cpd}_e0"
        if ex_name in mdl.reactions:
            mdl.reactions.get_by_id(ex_name).lower_bound = 0.0

    # Set target compound uptake
    target_ex = f"EX_{cpd_id}_e0"
    if target_ex not in mdl.reactions:
        raise KeyError(f"Exchange {target_ex} not found")
    mdl.reactions.get_by_id(target_ex).lower_bound = -abs(uptake_rate)

    sol_val, _, _, _ = run_optimization_with_dataframe(
        model=mdl,
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
        solver_name=solver_name
    )
    return sol_val


@dataclass
class PipelineResults:
    """Container for pipeline outputs that can be used by downstream analyses."""
    model: object
    processed_df: pd.DataFrame
    df_new: pd.DataFrame
    biomass_reaction: str
    enzyme_upper_bound: float
    fva_config: dict
    cobra_fva_path: str | None
    kingems_fva_pre_tuning_path: str | None
    kingems_fva_post_tuning_path: str | None
    run_id: str
    output_dir: str
    optimal_ngam: float | None
    optimal_gam: float | None
    # Timing (seconds per pipeline step)
    time_data_prep_s: float = 0.0
    time_optimization_s: float = 0.0
    time_sa_s: float = 0.0
    time_total_pipeline_s: float = 0.0
    # Model structure
    n_reactions: int = 0
    n_genes: int = 0
    n_metabolites: int = 0
    # kcat / CPI-Pred coverage
    n_rxn_gene_pairs: int = 0
    n_unique_proteins: int = 0
    n_unique_substrates: int = 0
    cobra_biomass: float = 0.0
    initial_ec_biomass: float = 0.0
    final_ec_biomass: float = 0.0
    sa_iterations: int = 0


def run_pipeline_core(
    config: dict,
    output_dir: str,
    run_id: str,
    force_regenerate: bool = False,
    logger: logging.Logger | None = None
) -> PipelineResults:
    """
    Run the core kinGEMs pipeline (steps 1-5) and return results for downstream use.

    This function encapsulates data preparation, model loading, optimization,
    simulated annealing, and FVA. It can be called from other scripts.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : str
        Directory to save pipeline outputs
    run_id : str
        Unique identifier for this run
    force_regenerate : bool
        If True, regenerate all intermediate files even if cached
    logger : logging.Logger, optional
        Logger instance. If None, uses print statements.

    Returns
    -------
    PipelineResults
        Dataclass containing model, dataframes, paths, and config needed for
        downstream analyses.
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # ---- Timing ----
    _t_pipeline_start = time.time()
    _t_step_start = time.time()

    # Extract configuration
    model_name = config['model_name']
    organism = config.get('organism', 'Unknown')
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.15)
    enable_fva = config.get('enable_fva', False)
    solver_name = config.get('solver', 'glpk')
    medium = config.get('medium', None)
    medium_upper_bound = config.get('medium_upper_bound', True)
    fva_config = config.get('fva', {})
    sa_config = config.get('simulated_annealing', {})
    results_subdir = config.get('results_subdir', None)  # Optional subdirectory for results

    # Detect model type
    is_modelseed = is_modelseed_model(model_name)
    model_type = "ModelSEED" if is_modelseed else "Standard"

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")
    raw_data_dir = os.path.join(data_dir, "raw")
    interim_data_dir = os.path.join(data_dir, "interim", model_name)
    processed_data_dir = os.path.join(data_dir, "processed", model_name)
    CPIPred_data_dir = os.path.join(data_dir, "interim", "CPI-Pred predictions")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(interim_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # File paths
    # Check if model is in BiGG_models subdirectory
    if results_subdir == "BiGG_models":
        model_path = os.path.join(raw_data_dir, "BiGG_models", f"{model_name}.xml")
    else:
        model_path = os.path.join(raw_data_dir, f"{model_name}.xml")
    substrates_output = os.path.join(interim_data_dir, f"{model_name}_substrates.csv")
    sequences_output = os.path.join(interim_data_dir, f"{model_name}_sequences.csv")
    merged_data_output = os.path.join(interim_data_dir, f"{model_name}_merged_data.csv")
    processed_data_output = os.path.join(processed_data_dir, f"{model_name}_processed_data.csv")

    log(f"=== kinGEMs Pipeline for {model_name} ===")
    log(f"Run ID: {run_id}")
    log(f"Model type: {model_type}")
    log(f"Organism: {organism}")
    log(f"Results directory: {output_dir}")
    log(f"Solver: {solver_name}")

    # Determine biomass reaction
    temp_model = cobra.io.read_sbml_model(model_path)
    # Set solver before any optimization
    temp_model.solver = solver_name
    biomass_reaction = config.get('biomass_reaction') or determine_biomass_reaction(temp_model)
    log(f"Biomass reaction: {biomass_reaction}")
    log(f"Baseline growth: {temp_model.slim_optimize():.4f}")

    # === Step 1: Prepare model data ===
    log("=== Step 1: Preparing model data ===")
    if not force_regenerate and os.path.exists(substrates_output) and os.path.exists(sequences_output):
        log("  Found existing files, loading cached data")
        substrate_df = pd.read_csv(substrates_output)
        sequences_df = pd.read_csv(sequences_output)
        model = load_model(model_path)
        model.solver = solver_name  # Set solver before any operations
        model = convert_to_irreversible(model)
    else:
        if is_modelseed:
            metadata_dir = config.get('metadata_dir', os.path.join(data_dir, "Biolog experiments"))
            model, substrate_df, sequences_df = prepare_modelseed_model_data(
                model_path=model_path,
                substrates_output=substrates_output,
                sequences_output=sequences_output,
                organism=organism,
                metadata_dir=metadata_dir
            )
        else:
            model, substrate_df, sequences_df = prepare_model_data(
                model_path=model_path,
                substrates_output=substrates_output,
                sequences_output=sequences_output,
                organism=organism,
                convert_irreversible=True
            )

    # Set solver after model preparation
    model.solver = solver_name
    log(f"  Model: {len(model.genes)} genes, {len(model.reactions)} reactions")

    # === Step 2: Merge substrate and sequence data ===
    log("=== Step 2: Merging substrate and sequence data ===")
    if not force_regenerate and os.path.exists(merged_data_output):
        merged_data = pd.read_csv(merged_data_output)
    else:
        merged_data = merge_substrate_sequences(
            substrate_df=substrate_df,
            sequences_df=sequences_df,
            model=model,
            output_path=merged_data_output
        )
    log(f"  Merged data: {len(merged_data)} rows")

    # === Step 3: Process kcat predictions ===
    log("=== Step 3: Processing CPI-Pred kcat values ===")
    if not force_regenerate and os.path.exists(processed_data_output):
        processed_data = pd.read_csv(processed_data_output)
    else:
        predictions_csv_path = find_predictions_file(model_name, CPIPred_data_dir)
        processed_data = process_kcat_predictions(
            merged_df=merged_data,
            predictions_csv_path=predictions_csv_path,
            output_path=processed_data_output
        )
    log(f"  Processed data: {len(processed_data)} rows")

    # Ensure kcat column exists
    if 'kcat_mean' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_mean']
    elif 'kcat_y' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_y']

    # Annotate model
    model = annotate_model_with_kcat_and_gpr(model=model, df=processed_data)

    _time_data_prep_s = time.time() - _t_step_start
    log(f"  Data preparation (steps 1-3) took {_time_data_prep_s:.1f}s")

    # === Step 4: Optimization ===
    log("=== Step 4: Running optimization ===")

    # Apply medium constraints
    if medium is not None:
        mode = "fixed fluxes" if medium_upper_bound else "max uptake rates"
        log(f"  Applying medium conditions ({mode})")
        for rxn_id, flux_value in medium.items():
            try:
                rxn = model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = flux_value
                if medium_upper_bound:
                    rxn.upper_bound = flux_value
            except KeyError:
                log(f"    Warning: Reaction '{rxn_id}' not found")

    cobra_solution = model.optimize()
    cobra_biomass = cobra_solution.objective_value
    log(f"  COBRApy biomass: {cobra_biomass:.4f}")

    # Enzyme-constrained optimization
    solution_value, df_FBA, gene_sequences_dict, _ = run_optimization_with_dataframe(
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
        solver_name=solver_name,
        medium=medium,
        medium_upper_bound=medium_upper_bound,
    )
    log(f"  kinGEMs biomass: {solution_value:.4f}")

    _time_optimization_s = time.time() - _t_step_start - _time_data_prep_s
    log(f"  Optimization (step 4) took {_time_optimization_s:.1f}s")
    _t_sa_start = time.time()

    # === Step 5: Simulated Annealing ===
    log("=== Step 5: Running simulated annealing ===")
    kcat_dict, top_targets, df_new, iterations, biomasses, df_FBA = simulated_annealing(
        model=model,
        processed_data=processed_data,
        biomass_reaction=biomass_reaction,
        objective_value=sa_config.get('biomass_goal', 0.5),
        gene_sequences_dict=gene_sequences_dict,
        output_dir=output_dir,
        enzyme_fraction=enzyme_upper_bound,
        n_top_enzymes=sa_config.get('n_top_enzymes', 65),
        temperature=sa_config.get('temperature', 1.0),
        cooling_rate=sa_config.get('cooling_rate', 0.95),
        min_temperature=sa_config.get('min_temperature', 0.01),
        max_iterations=sa_config.get('max_iterations', 100),
        max_unchanged_iterations=sa_config.get('max_unchanged_iterations', 5),
        change_threshold=sa_config.get('change_threshold', 0.009),
        verbose=sa_config.get('verbose', False),
        medium=medium,
        medium_upper_bound=medium_upper_bound
    )

    improvement = (biomasses[-1] - biomasses[0]) / biomasses[0] * 100 if biomasses[0] > 0 else 0
    log(f"  Annealing complete! Initial: {biomasses[0]:.4f}, Final: {biomasses[-1]:.4f}, Improvement: {improvement:.1f}%")

    _time_sa_s = time.time() - _t_sa_start
    log(f"  Simulated annealing (step 5) took {_time_sa_s:.1f}s")

    # Save df_new
    df_new_path = os.path.join(output_dir, "df_new.csv")
    df_new.to_csv(df_new_path, index=False)
    log(f"  Saved tuned dataframe: {df_new_path}")

    # Merge kcat_dict into df_new
    kcat_dict_path = os.path.join(output_dir, "kcat_dict.csv")
    if os.path.exists(kcat_dict_path):
        kcat_dict_df = pd.read_csv(kcat_dict_path)
        if 'reaction_gene' not in kcat_dict_df.columns:
            kcat_dict_df.columns = ['reaction_gene', 'kcat_value']
        df_new['reaction_gene'] = df_new['Reactions'].astype(str) + '_' + df_new['Single_gene'].astype(str)
        df_new = df_new.merge(kcat_dict_df, on='reaction_gene', how='left')
        df_new.rename(columns={'kcat_value': 'kcat_tuned'}, inplace=True)
        final_info_path = os.path.join(output_dir, "final_model_info.csv")
        df_new.to_csv(final_info_path, index=False)

    # Generate kcat comparison plot
    log("  Generating kcat comparison plot...")
    kcat_comparison_plot_path = os.path.join(output_dir, "kcat_comparison_initial_vs_tuned.png")
    try:
        plot_kcat_annealing_comparison(
            initial_df=processed_data,
            tuned_df=df_new,
            output_path=kcat_comparison_plot_path,
            model_name=model_name,
            show=False
        )
        log(f"  Saved kcat comparison plot: {kcat_comparison_plot_path}")
    except Exception as e:
        log(f"  Warning: Could not generate kcat comparison plot: {e}")

    # Generate per-subsystem kcat comparison grid
    log("  Generating per-subsystem kcat comparison plot...")
    kcat_subsystem_plot_path = os.path.join(output_dir, "kcat_comparison_by_subsystem.png")
    try:
        sub_map = {
            r.id: (r.subsystem if r.subsystem else 'Unknown')
            for r in model.reactions
        }
        df_new_sub = df_new.copy()
        df_new_sub['subsystem'] = df_new_sub['Reactions'].map(sub_map).fillna('Unknown')
        initial_df_sub = processed_data.copy()
        initial_df_sub['subsystem'] = initial_df_sub['Reactions'].map(sub_map).fillna('Unknown')
        plot_kcat_annealing_comparison_by_subsystem(
            initial_df=initial_df_sub,
            tuned_df=df_new_sub,
            subsystem_col='subsystem',
            output_path=kcat_subsystem_plot_path,
            model_name=model_name,
            max_subsystems=12,
            ncols=4,
            show=False,
        )
        log(f"  Saved per-subsystem kcat comparison plot: {kcat_subsystem_plot_path}")
    except Exception as e:
        log(f"  Warning: Could not generate per-subsystem kcat comparison plot: {e}")

    # === Step 5b: Maintenance Parameter Sweep (if enabled) ===
    optimal_ngam = None
    optimal_gam = None
    fva_model = model  # Default to base model

    enable_maintenance_sweep = config.get('enable_maintenance_sweep', False)
    if enable_maintenance_sweep:
        from copy import deepcopy
        log("=== Step 5b: Maintenance Parameters Sweep ===")
        maintenance_config = config.get('maintenance_sweep', {})
        ngam_rxn_id = config.get('ngam_rxn_id', 'ATPM')
        ngam_range = maintenance_config.get('ngam_range', None)
        gam_range = maintenance_config.get('gam_range', None)
        biomass_goal = sa_config.get('biomass_goal', cobra_biomass)

        log(f"  Target: Match target biomass ({biomass_goal:.4f})")
        log(f"  NGAM range: {ngam_range if ngam_range else 'default'}")
        log(f"  GAM range: {gam_range if gam_range else 'constant'}")

        maintenance_results = sweep_maintenance_parameters(
            model=model,
            processed_data=df_new,  # Use tuned parameters
            biomass_reaction=biomass_reaction,
            ngam_rxn_id=ngam_rxn_id,
            ngam_range=ngam_range,
            gam_range=gam_range,
            enzyme_upper_bound=enzyme_upper_bound,
            output_dir=output_dir,
            medium=medium,
            medium_upper_bound=medium_upper_bound,
            biomass_goal=biomass_goal,
            verbose=maintenance_config.get('verbose', False)
        )

        maintenance_results.to_csv(os.path.join(output_dir, 'maintenance_sweep_results.csv'), index=False)
        log(f"  Maintenance sweep completed: {len(maintenance_results)} parameter combinations tested")

        # Find optimal parameters closest to target
        if len(maintenance_results) > 0 and maintenance_results['biomass'].max() > 0:
            maintenance_results['distance_to_goal'] = abs(maintenance_results['biomass'] - biomass_goal)
            best_idx = maintenance_results['distance_to_goal'].idxmin()

            optimal_ngam = float(maintenance_results.loc[best_idx, 'ngam'])
            optimal_gam = float(maintenance_results.loc[best_idx, 'gam'])
            optimal_biomass = float(maintenance_results.loc[best_idx, 'biomass'])

            log(f"  Optimal maintenance parameters found:")
            log(f"    - NGAM: {optimal_ngam:.2f} mmol/gDW/h")
            log(f"    - GAM: {optimal_gam:.2f} mmol ATP/gDW")
            log(f"    - Biomass: {optimal_biomass:.4f}")
            log(f"    - Target: {biomass_goal:.4f}")
            log(f"    - Deviation: {abs(optimal_biomass - biomass_goal):.4f}")

            # Apply optimal parameters to model for FVA
            fva_model = deepcopy(model)
            fva_model.solver = solver_name  # Ensure solver is set after deepcopy

            # Apply NGAM
            try:
                ngam_rxn = fva_model.reactions.get_by_id(ngam_rxn_id)
                ngam_rxn.lower_bound = float(optimal_ngam)
                log(f"    ✓ Applied NGAM: {optimal_ngam:.2f}")
            except KeyError:
                log(f"    Warning: NGAM reaction '{ngam_rxn_id}' not found")

            # Apply GAM (if > 0)
            if optimal_gam > 0:
                try:
                    biomass_rxn = fva_model.reactions.get_by_id(biomass_reaction)
                    atp_met_ids = ['atp_c', 'ATP_c', 'cpd00002_c0']

                    current_gam = None
                    atp_met = None
                    for met_id in atp_met_ids:
                        try:
                            met = fva_model.metabolites.get_by_id(met_id)
                            if met in biomass_rxn.metabolites:
                                current_gam = abs(biomass_rxn.metabolites[met])
                                atp_met = met
                                break
                        except KeyError:
                            continue

                    if current_gam and atp_met and current_gam > 0:
                        scale = optimal_gam / current_gam
                        current_mets = biomass_rxn.metabolites.copy()

                        met_mappings = {
                            'h2o': ['h2o_c', 'H2O_c', 'cpd00001_c0'],
                            'adp': ['adp_c', 'ADP_c', 'cpd00008_c0'],
                            'pi': ['pi_c', 'Pi_c', 'cpd00009_c0'],
                            'h': ['h_c', 'H_c', 'cpd00067_c0']
                        }

                        mets_to_scale = [atp_met]
                        for met_type, possible_ids in met_mappings.items():
                            for met_id in possible_ids:
                                try:
                                    met = fva_model.metabolites.get_by_id(met_id)
                                    if met in current_mets:
                                        mets_to_scale.append(met)
                                        break
                                except KeyError:
                                    continue

                        for met in mets_to_scale:
                            old_coef = current_mets[met]
                            biomass_rxn.add_metabolites({met: old_coef * (scale - 1.0)}, combine=True)

                        log(f"    ✓ Applied GAM: scaled from {current_gam:.2f} to {optimal_gam:.2f}")

                except KeyError:
                    log(f"    Warning: Biomass reaction '{biomass_reaction}' not found")
        else:
            log("    Warning: No feasible solutions found in maintenance sweep")

    # === Step 6: Run FVA ===
    cobra_fva_path = None
    kingems_fva_pre_path = None
    kingems_fva_post_path = None

    if enable_fva:
        log("=== Step 6: Running Flux Variability Analysis ===")
        opt_ratio = fva_config.get('opt_ratio', 0.9)

        # COBRApy FVA (use fva_model with optimal maintenance if available)
        cobra_fva_path = os.path.join(output_dir, f"{model_name}_cobra_fva_results.csv")
        log(f"  Running COBRApy FVA (opt_ratio={opt_ratio})...")
        cobra_fva_results = cobra_fva(fva_model, fraction_of_optimum=opt_ratio)
        cobra_fva_df = pd.DataFrame({
            "Reactions": cobra_fva_results.index,
            "Min Solutions": cobra_fva_results['minimum'],
            "Max Solutions": cobra_fva_results['maximum'],
            "Solution Biomass": [fva_model.slim_optimize()] * len(cobra_fva_results)
        })
        cobra_fva_df.to_csv(cobra_fva_path, index=False)
        log(f"  Saved COBRApy FVA: {cobra_fva_path}")

        # kinGEMs FVA (pre-tuning)
        kingems_fva_pre_path = os.path.join(output_dir, f"{model_name}_pre_tuning_fva_results.csv")
        run_fva_analysis(model, processed_data, biomass_reaction, enzyme_upper_bound,
                        output_dir, f"{model_name}_pre_tuning", fva_config)

        # kinGEMs FVA (post-tuning with optimal maintenance)
        kingems_fva_post_path = os.path.join(output_dir, f"{model_name}_fva_results.csv")
        run_fva_analysis(fva_model, df_new, biomass_reaction, enzyme_upper_bound,
                        output_dir, model_name, fva_config)

    log("=== Pipeline core complete ===")

    # ---- Compute kcat coverage metrics from processed_data ----
    _n_unique_proteins   = int(processed_data['Single_gene'].nunique()) if 'Single_gene' in processed_data.columns else 0
    _substrate_col       = next((c for c in ('substrate', 'Substrate', 'substrate_name') if c in processed_data.columns), None)
    _n_unique_substrates = int(processed_data[_substrate_col].nunique()) if _substrate_col else 0
    _time_total_s        = time.time() - _t_pipeline_start

    return PipelineResults(
        model=model,
        processed_df=processed_data,
        df_new=df_new,
        biomass_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        fva_config=fva_config,
        cobra_fva_path=cobra_fva_path,
        kingems_fva_pre_tuning_path=kingems_fva_pre_path,
        kingems_fva_post_tuning_path=kingems_fva_post_path,
        run_id=run_id,
        output_dir=output_dir,
        optimal_ngam=optimal_ngam,
        optimal_gam=optimal_gam,
        # Timing
        time_data_prep_s=round(_time_data_prep_s, 1),
        time_optimization_s=round(_time_optimization_s, 1),
        time_sa_s=round(_time_sa_s, 1),
        time_total_pipeline_s=round(_time_total_s, 1),
        # Model structure
        n_reactions=len(model.reactions),
        n_genes=len(model.genes),
        n_metabolites=len(model.metabolites),
        # kcat / CPI-Pred coverage
        n_rxn_gene_pairs=len(processed_data),
        n_unique_proteins=_n_unique_proteins,
        n_unique_substrates=_n_unique_substrates,
        cobra_biomass=round(float(cobra_biomass), 6),
        initial_ec_biomass=round(float(solution_value), 6),
        final_ec_biomass=round(float(biomasses[-1]), 6) if biomasses else 0.0,
        sa_iterations=len(biomasses),
    )


def run_fva_analysis(model, processed_df, biomass_reaction, enzyme_upper_bound,
                     tuning_results_dir, organism_strain_GEMname, fva_config=None, og_model=None):
    """Run flux variability analysis and generate plots."""
    print("\n=== Step 6: Running Flux Variability Analysis ===")

    # Get FVA configuration
    fva_config = fva_config or {}
    use_parallel = fva_config.get('parallel', False)
    n_workers = fva_config.get('workers', None)
    opt_ratio = fva_config.get('opt_ratio', 0.9)

    fva_results_path = os.path.join(tuning_results_dir, f"{organism_strain_GEMname}_fva_results.csv")
    fva_plot_path = os.path.join(tuning_results_dir, f"{organism_strain_GEMname}_fva_flux_range_plot.png")
    fva_cumulative_path = os.path.join(tuning_results_dir, f"{organism_strain_GEMname}_fva_cumulative_plot.png")
    fva_baseline_results_path = os.path.join(tuning_results_dir, f"{organism_strain_GEMname}_cobra_fva_results.csv")

    # Run kinGEMs FVA
    if use_parallel:
        print(f"  Using parallel FVA with {n_workers or 'auto'} workers...")
        fva_results, _, _ = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=fva_results_path,
            enzyme_upper_bound=enzyme_upper_bound,
            opt_ratio=opt_ratio,
            n_workers=n_workers
        )
    else:
        print("  Using sequential FVA...")
        fva_results, _, _ = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=fva_results_path,
            enzyme_upper_bound=enzyme_upper_bound,
            opt_ratio=opt_ratio
        )
    print(f"  kinGEMs FVA completed: {len(fva_results)} reactions analyzed")

    # Run COBRApy FVA for comparison
    print("  Running COBRApy FVA for comparison...")
    if og_model is not None:
        print("  Using original model for COBRApy FVA")
        cobra_fva_results = cobra_fva(og_model, fraction_of_optimum=0.9)
        cobra_fva_df = pd.DataFrame({
            "Reactions": cobra_fva_results.index,
            "Min Solutions": cobra_fva_results['minimum'],
            "Max Solutions": cobra_fva_results['maximum'],
            "Solution Biomass": [(og_model).slim_optimize()] * len(cobra_fva_results)
        })
        cobra_fva_df.to_csv(fva_baseline_results_path, index=False)
        print(f"  COBRApy FVA completed: {len(cobra_fva_df)} reactions analyzed and results saved to {fva_baseline_results_path}")
    else:
        print(" Using irreversible model for cobrapy FVA")
        cobra_fva_results = cobra_fva(model, fraction_of_optimum=0.9)
        cobra_fva_df = pd.DataFrame({
            "Reactions": cobra_fva_results.index,
            "Min Solutions": cobra_fva_results['minimum'],
            "Max Solutions": cobra_fva_results['maximum'],
            "Solution Biomass": [model.slim_optimize()] * len(cobra_fva_results)
        })
        cobra_fva_df.to_csv(fva_baseline_results_path, index=False)
        print(f"  COBRApy FVA completed: {len(cobra_fva_df)} reactions analyzed and results saved to {fva_baseline_results_path}")


    # Generate plots
    print("  Generating FVA plots...")
    plot_flux_variability(fva_results, output_file=fva_plot_path)
    print(f"  Saved FVA flux range plot to: {fva_plot_path}")

    plot_cumulative_fvi_distribution(
        fva_dataframes=[fva_results, cobra_fva_df],
        labels=["kinGEMs FVA", "COBRApy FVA"],
        output_path=fva_cumulative_path
    )
    print(f"  Saved FVA cumulative plot to: {fva_cumulative_path}")


def run_biolog_validation(model, processed_df, biomass_reaction, enzyme_upper_bound,
                         biolog_config, tuning_results_dir, solver_name='glpk'):
    """Run Biolog experimental validation."""
    print("\n=== Step 6: Biolog Experimental Validation ===")

    biolog_path = biolog_config['experiments_file']
    sheet_name = biolog_config.get('sheet_name', 'Ecoli')
    blocked_cpds = biolog_config.get('blocked_compounds', [])
    reference_cpd = biolog_config.get('reference_compound', 'cpd00027')
    uptake_rate = biolog_config.get('uptake_rate', 100.0)

    exp_df = pd.read_excel(biolog_path, sheet_name=sheet_name, engine="openpyxl")

    # Calculate reference (glucose) rate
    print(f"  Calculating reference growth rate for {reference_cpd}...")
    ref_rate = simulate_enzyme_rate(
        base_model=model,
        processed_df=processed_df,
        biomass_reaction=biomass_reaction,
        blocked_cpds=blocked_cpds,
        cpd_id=reference_cpd,
        enzyme_upper_bound=enzyme_upper_bound,
        uptake_rate=uptake_rate,
        solver_name=solver_name
    )
    print(f"  Reference enzyme-constrained growth: {ref_rate:.4f}")

    # Test all experimental substrates
    results = []
    for row in exp_df.itertuples():
        cpd = row.cpd
        print(f"  Testing substrate: {cpd}...")
        try:
            rate = simulate_enzyme_rate(
                base_model=model,
                processed_df=processed_df,
                biomass_reaction=biomass_reaction,
                blocked_cpds=blocked_cpds,
                cpd_id=cpd,
                enzyme_upper_bound=enzyme_upper_bound,
                uptake_rate=uptake_rate,
                solver_name=solver_name
            )
        except Exception as e:
            print(f"    ⚠️ Warning for {cpd}: {e}")
            rate = None

        norm = rate / ref_rate if rate is not None and ref_rate > 0 else None
        results.append({
            'cpd': cpd,
            'ec_rate': rate,
            'norm_rate': norm,
            'exp_value': row.exp_value
        })

    # Merge and save results
    result_df = pd.DataFrame(results)
    comp_df = exp_df.merge(result_df, on='cpd')

    comparison_path = os.path.join(tuning_results_dir, "biolog_comparison.csv")
    comp_df.to_csv(comparison_path, index=False)
    print(f"  Saved comparison to: {comparison_path}")

    # Generate plot
    plt.figure(figsize=(6, 4))
    plt.scatter(comp_df['exp_value'], comp_df['norm_rate'], s=50)
    plt.xlabel('Experimental value (normalized)')
    plt.ylabel('Model normalized rate')
    plt.title('Enzyme-constrained FBA vs. experimental')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(tuning_results_dir, "biolog_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to: {plot_path}")


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config_path = sys.argv[1]
    FORCE_REGENERATE = '--force' in sys.argv or '-f' in sys.argv

    # --replot <results_dir>: regenerate figures from an existing results folder
    if '--replot' in sys.argv:
        replot_idx = sys.argv.index('--replot')
        if replot_idx + 1 >= len(sys.argv):
            print("Error: --replot requires a path to an existing results directory")
            sys.exit(1)

        results_dir = os.path.abspath(sys.argv[replot_idx + 1])
        if not os.path.isdir(results_dir):
            print(f"Error: results directory not found: {results_dir}")
            sys.exit(1)

        # Load config to get model name / subsystem map
        config = load_config(config_path)
        model_name = config['model_name']
        results_subdir = config.get('results_subdir', None)

        print("\n" + "="*70)
        print(f"=== REPLOT MODE: {model_name} ===")
        print("="*70)
        print(f"Results directory: {results_dir}")

        # Load df_new
        df_new_path = os.path.join(results_dir, "df_new.csv")
        final_info_path = os.path.join(results_dir, "final_model_info.csv")
        df_fba_path = os.path.join(results_dir, "df_FBA.csv")

        if os.path.exists(final_info_path):
            df_new = pd.read_csv(final_info_path)
            print(f"Loaded final_model_info.csv: {len(df_new)} rows")
        elif os.path.exists(df_new_path):
            df_new = pd.read_csv(df_new_path)
            print(f"Loaded df_new.csv: {len(df_new)} rows")
        else:
            print("Error: neither df_new.csv nor final_model_info.csv found in results dir")
            sys.exit(1)

        # Load initial (processed) kcat data
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        processed_data_path = os.path.join(
            project_root, "data", "processed", model_name, f"{model_name}_processed_data.csv"
        )
        if os.path.exists(processed_data_path):
            processed_data = pd.read_csv(processed_data_path)
            if 'kcat_mean' in processed_data.columns and 'kcat' not in processed_data.columns:
                processed_data['kcat'] = processed_data['kcat_mean']
            print(f"Loaded processed data: {len(processed_data)} rows")
        else:
            processed_data = df_new.copy()
            print("Warning: processed data not found, using df_new as initial data")

        # Load model for subsystem map
        raw_data_dir = os.path.join(project_root, "data", "raw")
        if results_subdir == "BiGG_models":
            model_path = os.path.join(raw_data_dir, "BiGG_models", f"{model_name}.xml")
        else:
            model_path = os.path.join(raw_data_dir, f"{model_name}.xml")

        # Plot 1: kcat initial vs tuned
        print("\nGenerating: kcat_comparison_initial_vs_tuned.png")
        kcat_plot_path = os.path.join(results_dir, "kcat_comparison_initial_vs_tuned.png")
        try:
            plot_kcat_annealing_comparison(
                initial_df=processed_data,
                tuned_df=df_new,
                output_path=kcat_plot_path,
                model_name=model_name,
                show=False
            )
            print(f"  Saved: {kcat_plot_path}")
        except Exception as e:
            print(f"  Warning: {e}")

        # Plot 2: per-subsystem kcat comparison (needs model XML)
        print("Generating: kcat_comparison_by_subsystem.png")
        sub_plot_path = os.path.join(results_dir, "kcat_comparison_by_subsystem.png")
        if os.path.exists(model_path):
            try:
                replot_model = cobra.io.read_sbml_model(model_path)
                sub_map = {r.id: (r.subsystem if r.subsystem else 'Unknown')
                           for r in replot_model.reactions}
                df_new_sub = df_new.copy()
                df_new_sub['subsystem'] = df_new_sub['Reactions'].map(sub_map).fillna('Unknown')
                processed_sub = processed_data.copy()
                processed_sub['subsystem'] = processed_sub['Reactions'].map(sub_map).fillna('Unknown')
                plot_kcat_annealing_comparison_by_subsystem(
                    initial_df=processed_sub,
                    tuned_df=df_new_sub,
                    subsystem_col='subsystem',
                    output_path=sub_plot_path,
                    model_name=model_name,
                    max_subsystems=12,
                    ncols=4,
                    show=False,
                )
                print(f"  Saved: {sub_plot_path}")
            except Exception as e:
                print(f"  Warning: {e}")
        else:
            print(f"  Skipped (model XML not found at {model_path})")

        # Plot 3: flux distribution
        print("Generating: flux_distribution.png")
        if os.path.exists(df_fba_path):
            try:
                from kinGEMs.plots import plot_flux_distribution
                df_FBA = pd.read_csv(df_fba_path)
                fba_plot_path = os.path.join(results_dir, "flux_distribution.png")
                plot_flux_distribution(df_FBA=df_FBA, output_path=fba_plot_path, show=False)
                print(f"  Saved: {fba_plot_path}")
            except Exception as e:
                print(f"  Warning: {e}")
        else:
            print("  Skipped (df_FBA.csv not found)")

        # Plot 4: maintenance sweep
        print("Generating: maintenance_sweep_plot.png")
        maint_path = os.path.join(results_dir, "maintenance_sweep_results.csv")
        if os.path.exists(maint_path):
            try:
                import matplotlib.pyplot as plt
                from kinGEMs.plots import set_plotting_style, FONT_SIZES, DEFAULT_DPI
                set_plotting_style()

                maint_df = pd.read_csv(maint_path)
                gam_values = sorted(maint_df['gam'].unique())
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Left: Biomass vs NGAM per GAM
                for gam_val in gam_values:
                    subset = maint_df[maint_df['gam'] == gam_val]
                    axes[0].plot(subset['ngam'], subset['biomass'],
                                 marker='o', label=f'GAM={gam_val:.1f}')
                axes[0].set_xlabel('NGAM (mmol/gDW/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                axes[0].set_ylabel('Biomass (1/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                axes[0].set_title('Biomass vs NGAM', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
                axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                               ncol=3, frameon=True, fontsize=FONT_SIZES['legend'])
                axes[0].grid(True, alpha=0.3)

                # Right: heatmap if multiple GAM values
                if len(gam_values) > 1:
                    pivot_data = maint_df.pivot(index='gam', columns='ngam', values='biomass')
                    im = axes[1].imshow(pivot_data, aspect='auto', cmap='viridis', origin='lower')
                    axes[1].set_xlabel('NGAM (mmol/gDW/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                    axes[1].set_ylabel('GAM (mmol ATP/gDW)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                    axes[1].set_title('Biomass Heatmap', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
                    axes[1].set_xticks(range(len(pivot_data.columns)))
                    axes[1].set_xticklabels([f'{x:.1f}' for x in pivot_data.columns],
                                            fontsize=FONT_SIZES['tick_label'])
                    axes[1].set_yticks(range(len(pivot_data.index)))
                    axes[1].set_yticklabels([f'{y:.1f}' for y in pivot_data.index],
                                            fontsize=FONT_SIZES['tick_label'])
                    cbar = plt.colorbar(im, ax=axes[1])
                    cbar.set_label('Biomass (1/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                else:
                    axes[1].bar(maint_df['ngam'], maint_df['biomass'], edgecolor='black', linewidth=2)
                    axes[1].set_xlabel('NGAM (mmol/gDW/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                    axes[1].set_ylabel('Biomass (1/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                    axes[1].set_title('Biomass Distribution', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
                    axes[1].grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                maint_plot_path = os.path.join(results_dir, "maintenance_sweep_plot.png")
                plt.savefig(maint_plot_path, dpi=DEFAULT_DPI, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {maint_plot_path}")
            except Exception as e:
                print(f"  Warning: {e}")
        else:
            print("  Skipped (maintenance_sweep_results.csv not found)")

        print("\n" + "="*70)
        print("=== Replot Complete ===")
        print("="*70)
        return

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Extract configuration
    model_name = config['model_name']
    organism = config.get('organism', 'Unknown')
    enable_biolog = config.get('enable_biolog_validation', False)
    solver_name = config.get('solver', 'glpk')
    ngam_rxn_id = config.get('ngam_rxn_id', 'ATPM')
    results_subdir = config.get('results_subdir', None)  # Optional subdirectory for results

    # Detect model type
    is_modelseed = is_modelseed_model(model_name)
    model_type = "ModelSEED" if is_modelseed else "Standard"

    # Generate run ID
    run_id = f"{model_name}_{datetime.today().strftime('%Y%m%d')}_{random.randint(1000, 9999)}"

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, "results")

    # Use custom subdirectory if specified in config
    if results_subdir:
        tuning_results_dir = os.path.join(results_dir, "tuning_results", results_subdir, run_id)
    else:
        tuning_results_dir = os.path.join(results_dir, "tuning_results", run_id)

    os.makedirs(tuning_results_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"=== kinGEMs Pipeline for {model_name} ===")
    print("="*70)
    print(f"Run ID: {run_id}")
    print(f"Model type: {model_type}")
    print(f"Organism: {organism}")
    print(f"Results directory: {tuning_results_dir}")
    if FORCE_REGENERATE:
        print("⚠️  Force regenerate mode: will regenerate all intermediate files")
    print("="*70)

    # === Run core pipeline (Steps 1-6) ===
    pipeline_results = run_pipeline_core(
        config=config,
        output_dir=tuning_results_dir,
        run_id=run_id,
        force_regenerate=FORCE_REGENERATE,
        logger=None  # Use print statements
    )

    # Extract results
    model = pipeline_results.model
    df_new = pipeline_results.df_new
    biomass_reaction = pipeline_results.biomass_reaction
    enzyme_upper_bound = pipeline_results.enzyme_upper_bound
    optimal_ngam = pipeline_results.optimal_ngam
    optimal_gam = pipeline_results.optimal_gam

    # Load original model for Biolog validation if needed
    if enable_biolog:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_dir = os.path.join(project_root, "data")
        raw_data_dir = os.path.join(data_dir, "raw")
        model_path = os.path.join(raw_data_dir, f"{model_name}.xml")
        og_model = load_model(model_path)
        og_model.solver = solver_name  # Set solver before any operations
        processed_data = pipeline_results.processed_df

        biolog_config = config.get('biolog_validation', {})
        run_biolog_validation(model, processed_data, biomass_reaction, enzyme_upper_bound,
                             biolog_config, tuning_results_dir, solver_name)

    # === Step 7: Save Final Model ===
    print("\n=== Step 7: Saving final model ===")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Use custom subdirectory if specified in config
    if results_subdir:
        model_output_dir = os.path.join(project_root, "models", results_subdir)
    else:
        model_output_dir = os.path.join(project_root, "models")

    os.makedirs(model_output_dir, exist_ok=True)
    model_output_path = os.path.join(model_output_dir, f"{run_id}.xml")

    # Assign kcats first
    model_with_kcats = assign_kcats_to_model(model, df_new)

    # Apply optimal maintenance parameters to the model copy if found from sweep
    if optimal_ngam is not None and optimal_gam is not None:
        print(f"  Applying optimal maintenance parameters to final model:")
        print(f"    - Setting NGAM ({ngam_rxn_id}) lower bound to {optimal_ngam:.2f}")
        try:
            ngam_rxn = model_with_kcats.reactions.get_by_id(ngam_rxn_id)
            ngam_rxn.lower_bound = float(optimal_ngam)  # Ensure float
        except KeyError:
            print(f"    Warning: NGAM reaction '{ngam_rxn_id}' not found")

        # Apply GAM scaling (only if GAM > 0 to avoid creating invalid reactions)
        if optimal_gam > 0:
            print(f"    - Scaling GAM to {optimal_gam:.2f} mmol ATP/gDW")
            try:
                biomass_rxn = model_with_kcats.reactions.get_by_id(biomass_reaction)
                atp_met_ids = ['atp_c', 'ATP_c', 'cpd00002_c0']

                # Find ATP and get current GAM
                current_gam = None
                atp_met = None
                for met_id in atp_met_ids:
                    try:
                        met = model_with_kcats.metabolites.get_by_id(met_id)
                        if met in biomass_rxn.metabolites:
                            current_gam = abs(biomass_rxn.metabolites[met])
                            atp_met = met
                            break
                    except KeyError:
                        continue

                if current_gam and atp_met:
                    scale = optimal_gam / current_gam
                    current_mets = biomass_rxn.metabolites.copy()

                    # Scale ATP maintenance block (ATP, H2O, ADP, Pi, H+)
                    met_mappings = {
                        'h2o': ['h2o_c', 'H2O_c', 'cpd00001_c0'],
                        'adp': ['adp_c', 'ADP_c', 'cpd00008_c0'],
                        'pi': ['pi_c', 'Pi_c', 'cpd00009_c0'],
                        'h': ['h_c', 'H_c', 'cpd00067_c0']
                    }

                    mets_to_scale = [atp_met]
                    for met_type, possible_ids in met_mappings.items():
                        for met_id in possible_ids:
                            try:
                                met = model_with_kcats.metabolites.get_by_id(met_id)
                                if met in current_mets:
                                    mets_to_scale.append(met)
                                    break
                            except KeyError:
                                continue

                    for met in mets_to_scale:
                        old_coef = current_mets[met]
                        biomass_rxn.add_metabolites({met: old_coef * (scale - 1.0)}, combine=True)

                    print(f"    ✓ GAM scaled from {current_gam:.2f} to {optimal_gam:.2f}")
                else:
                    print(f"    Warning: Could not find ATP in biomass reaction for GAM scaling")

            except KeyError:
                print(f"    Warning: Biomass reaction '{biomass_reaction}' not found")
        else:
            print(f"    ⚠️  Skipping GAM scaling: GAM=0 would create an invalid biomass reaction")
            print(f"       Keeping original GAM value in the saved model")

    model_with_kcats = clean_annotations(model_with_kcats)

    # Ensure all reaction bounds are floats for SBML export (not integers)
    for rxn in model_with_kcats.reactions:
        if isinstance(rxn.lower_bound, int):
            rxn.lower_bound = float(rxn.lower_bound)
        if isinstance(rxn.upper_bound, int):
            rxn.upper_bound = float(rxn.upper_bound)

    write_sbml_model(model_with_kcats, model_output_path)
    print(f"  Final GEM saved to: {model_output_path}")

    # Save model configuration summary
    # Read biomass values from saved simulated annealing results
    _ib = pipeline_results.initial_ec_biomass
    _fb = pipeline_results.final_ec_biomass
    _imp = (_fb - _ib) / _ib * 100 if _ib > 0 else 0.0

    config_summary = {
        'run_id': run_id,
        'model_name': model_name,
        'organism': organism,
        'enzyme_upper_bound': float(enzyme_upper_bound),
        # SA performance
        'initial_biomass': _ib,
        'final_biomass': _fb,
        'improvement_percent': round(_imp, 4),
        'iterations': pipeline_results.sa_iterations,
        # Maintenance sweep
        'optimal_ngam': float(optimal_ngam) if optimal_ngam is not None else None,
        'optimal_gam': float(optimal_gam) if optimal_gam is not None else None,
        'ngam_rxn_id': ngam_rxn_id,
        'biomass_reaction': biomass_reaction,
        # Model structure
        'n_reactions': pipeline_results.n_reactions,
        'n_genes': pipeline_results.n_genes,
        'n_metabolites': pipeline_results.n_metabolites,
        # kcat / CPI-Pred coverage
        'n_rxn_gene_pairs': pipeline_results.n_rxn_gene_pairs,
        'n_unique_proteins': pipeline_results.n_unique_proteins,
        'n_unique_substrates': pipeline_results.n_unique_substrates,
        'cpipred_coverage_pct': round(
            pipeline_results.n_rxn_gene_pairs / max(pipeline_results.n_reactions, 1) * 100, 2
        ),
        # Biomass progression
        'cobra_biomass': pipeline_results.cobra_biomass,
        'initial_ec_biomass': pipeline_results.initial_ec_biomass,
        # kcat statistics (computed from df_new if kcat_updated exists)
        'median_kcat_initial_hr': None,
        'median_kcat_tuned_hr': None,
        'median_fold_change': None,
        'n_kcat_increased': None,
        'n_kcat_decreased': None,
        # Per-step wall time
        'time_data_prep_s': pipeline_results.time_data_prep_s,
        'time_optimization_s': pipeline_results.time_optimization_s,
        'time_sa_s': pipeline_results.time_sa_s,
        'time_total_pipeline_s': pipeline_results.time_total_pipeline_s,
        # SLURM metrics (patched in by the SLURM script after job completion)
        'slurm_job_id': None,
        'slurm_node': None,
        'slurm_cpus': None,
        'slurm_mem_requested_gb': None,
        'slurm_mem_used_mb': None,
        'slurm_walltime_s': None,
    }
    # Compute kcat stats from df_new
    df_new_local = pipeline_results.df_new
    if 'kcat_mean' in df_new_local.columns and 'kcat_updated' in df_new_local.columns:
        kc = df_new_local[['kcat_mean', 'kcat_updated']].dropna()
        kc = kc[(kc['kcat_mean'] > 0) & (kc['kcat_updated'] > 0)]
        if not kc.empty:
            fc = kc['kcat_updated'] / kc['kcat_mean']
            config_summary['median_kcat_initial_hr'] = round(float(kc['kcat_mean'].median() * 3600), 2)
            config_summary['median_kcat_tuned_hr']   = round(float(kc['kcat_updated'].median() * 3600), 2)
            config_summary['median_fold_change']      = round(float(fc.median()), 4)
            config_summary['n_kcat_increased']        = int((fc > 1).sum())
            config_summary['n_kcat_decreased']        = int((fc < 1).sum())
    config_summary_path = os.path.join(tuning_results_dir, "model_config_summary.json")
    with open(config_summary_path, 'w') as f:
        json.dump(config_summary, f, indent=2)
    print(f"  Configuration summary saved to: {config_summary_path}")

    # === Summary ===
    print("\n" + "="*70)
    print("=== Pipeline Complete ===")
    print("="*70)
    print(f"Run ID: {run_id}")
    print(f"Model: {model_name} ({model_type})")

    # Print biomass progression from pipeline_results
    _pr = pipeline_results
    if _pr.initial_ec_biomass > 0:
        _ann_imp = (_pr.final_ec_biomass - _pr.initial_ec_biomass) / _pr.initial_ec_biomass * 100
        print(f"Initial biomass (enzyme-constrained): {_pr.initial_ec_biomass:.4f}")
        print(f"Post-annealing biomass: {_pr.final_ec_biomass:.4f}")
        print(f"Annealing improvement: {_ann_imp:.1f}%")
        print(f"Annealing iterations: {_pr.sa_iterations}")

    if optimal_ngam is not None:
        print(f"\nOptimal maintenance parameters (applied to final model):")
        print(f"  NGAM: {optimal_ngam:.2f} mmol/gDW/h")
        print(f"  GAM: {optimal_gam:.2f} mmol ATP/gDW")

        # Read optimal biomass from maintenance sweep results
        maint_results_path = os.path.join(tuning_results_dir, "maintenance_sweep_results.csv")
        if os.path.exists(maint_results_path):
            maint_df = pd.read_csv(maint_results_path)
            maint_df['distance_to_goal'] = abs(maint_df['biomass'] - maint_df['biomass'].max())
            best_row = maint_df.loc[maint_df['distance_to_goal'].idxmin()]
            optimal_biomass = float(best_row['biomass'])
            print(f"  Final biomass with optimal maintenance: {optimal_biomass:.4f}")
            if os.path.exists(sa_results_path):
                total_improvement = (optimal_biomass - initial_biomass) / initial_biomass * 100 if initial_biomass > 0 else 0
                print(f"  Total improvement (annealing + maintenance): {total_improvement:.1f}%")

    print(f"\nResults directory: {tuning_results_dir}")
    print(f"Final model: {model_output_path}")
    print("="*70)


if __name__ == '__main__':
    main()

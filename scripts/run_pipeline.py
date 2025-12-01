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

from datetime import datetime
import gc  # For garbage collection
import json
import logging
import os
import random
import sys
import warnings

import cobra
from cobra.flux_analysis import flux_variability_analysis as cobra_fva
from cobra.io import write_sbml_model
import matplotlib.pyplot as plt
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
)
from kinGEMs.dataset_modelseed import prepare_modelseed_model_data
from kinGEMs.modeling.fva import (
    flux_variability_analysis,
    flux_variability_analysis_parallel,
    plot_flux_variability,
)
from kinGEMs.plots import (
    plot_cumulative_fvi_distribution,
)
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.modeling.tuning import simulated_annealing

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


def run_fva_analysis(model, processed_df, biomass_reaction, enzyme_upper_bound,
                     tuning_results_dir, organism_strain_GEMname, fva_config=None):
    """Run flux variability analysis and generate plots."""
    print("\n=== Step 6: Running Flux Variability Analysis ===")

    # Get FVA configuration
    fva_config = fva_config or {}
    use_parallel = fva_config.get('parallel', False)
    n_workers = fva_config.get('workers', None)

    fva_results_path = os.path.join(tuning_results_dir, f"{organism_strain_GEMname}_fva_results.csv")
    fva_plot_path = os.path.join(tuning_results_dir, f"{organism_strain_GEMname}_fva_flux_range_plot.png")
    fva_cumulative_path = os.path.join(tuning_results_dir, f"{organism_strain_GEMname}_fva_cumulative_plot.png")

    # Run kinGEMs FVA
    if use_parallel:
        print(f"  Using parallel FVA with {n_workers or 'auto'} workers...")
        fva_results, _, _ = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=fva_results_path,
            enzyme_upper_bound=enzyme_upper_bound,
            n_workers=n_workers
        )
    else:
        print("  Using sequential FVA...")
        fva_results, _, _ = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=fva_results_path,
            enzyme_upper_bound=enzyme_upper_bound
        )
    print(f"  kinGEMs FVA completed: {len(fva_results)} reactions analyzed")

    # Run COBRApy FVA for comparison
    print("  Running COBRApy FVA for comparison...")
    cobra_fva_results = cobra_fva(model, fraction_of_optimum=0.9)
    cobra_fva_df = pd.DataFrame({
        "Reactions": cobra_fva_results.index,
        "Min Solutions": cobra_fva_results['minimum'],
        "Max Solutions": cobra_fva_results['maximum'],
        "Solution Biomass": [model.slim_optimize()] * len(cobra_fva_results)
    })
    print(f"  COBRApy FVA completed: {len(cobra_fva_df)} reactions analyzed")

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

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Extract configuration
    model_name = config['model_name']
    organism = config.get('organism', 'Unknown')
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.15)
    enable_fva = config.get('enable_fva', False)
    enable_biolog = config.get('enable_biolog_validation', False)
    solver_name = config.get('solver', 'glpk')  # Default to GLPK (free solver)

    # Detect model type
    is_modelseed = is_modelseed_model(model_name)
    model_type = "ModelSEED" if is_modelseed else "Standard"

    # Generate run ID
    run_id = f"{model_name}_{datetime.today().strftime('%Y%m%d')}_{random.randint(1000, 9999)}"

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")
    raw_data_dir = os.path.join(data_dir, "raw")
    interim_data_dir = os.path.join(data_dir, "interim", model_name)
    processed_data_dir = os.path.join(data_dir, "processed", model_name)
    CPIPred_data_dir = os.path.join(data_dir, "interim", "CPI-Pred predictions")
    results_dir = os.path.join(project_root, "results")
    tuning_results_dir = os.path.join(results_dir, "tuning_results", run_id)
    os.makedirs(tuning_results_dir, exist_ok=True)

    # File paths
    model_path = os.path.join(raw_data_dir, f"{model_name}.xml")



    substrates_output = os.path.join(interim_data_dir, f"{model_name}_substrates.csv")
    sequences_output = os.path.join(interim_data_dir, f"{model_name}_sequences.csv")
    merged_data_output = os.path.join(interim_data_dir, f"{model_name}_merged_data.csv")
    processed_data_output = os.path.join(processed_data_dir, f"{model_name}_processed_data.csv")

    print("\n" + "="*70)
    print(f"=== kinGEMs Pipeline for {model_name} ===")
    print("="*70)
    print(f"Run ID: {run_id}")
    print(f"Model type: {model_type}")
    print(f"Organism: {organism}")
    print(f"Results directory: {tuning_results_dir}")
    print(f"Solver: {solver_name}")
    if FORCE_REGENERATE:
        print("⚠️  Force regenerate mode: will regenerate all intermediate files")
    print("="*70)

    # Print configuration summary
    print("\n=== Configuration Summary ===")
    print(f"Config file: {config_path}")
    print(f"Model name: {model_name}")
    print(f"Enzyme upper bound: {enzyme_upper_bound}")
    print(f"Enable FVA: {enable_fva}")
    print(f"Enable Biolog validation: {enable_biolog}")

    # Print FVA config if enabled
    if enable_fva:
        fva_config = config.get('fva', {})
        print("\nFVA Settings:")
        print(f"  Parallel execution: {fva_config.get('parallel', False)}")
        if fva_config.get('parallel'):
            print(f"  Workers: {fva_config.get('workers', 'auto')}")
            print(f"  Method: {fva_config.get('method', 'dask')}")

    # Print simulated annealing config
    sa_config = config.get('simulated_annealing', {})
    print("\nSimulated Annealing Settings:")
    print(f"  Temperature: {sa_config.get('temperature', 1.0)}")
    print(f"  Cooling rate: {sa_config.get('cooling_rate', 0.95)}")
    print(f"  Min temperature: {sa_config.get('min_temperature', 0.01)}")
    print(f"  Max iterations: {sa_config.get('max_iterations', 100)}")
    print(f"  Max unchanged iterations: {sa_config.get('max_unchanged_iterations', 5)}")
    print(f"  Change threshold: {sa_config.get('change_threshold', 0.009)}")
    print(f"  Biomass goal: {sa_config.get('biomass_goal', 0.5)}")

    # Print Biolog config if enabled
    if enable_biolog:
        biolog_cfg = config.get('biolog_validation', {})
        print("\nBiolog Validation Settings:")
        print(f"  Experiments file: {biolog_cfg.get('experiments_file', 'N/A')}")
        print(f"  Sheet name: {biolog_cfg.get('sheet_name', 'Ecoli')}")
        print(f"  Reference compound: {biolog_cfg.get('reference_compound', 'cpd00027')}")
        print(f"  Uptake rate: {biolog_cfg.get('uptake_rate', 100.0)}")

    print("="*70)

    # Determine biomass reaction
    temp_model = cobra.io.read_sbml_model(model_path)
    biomass_reaction = config.get('biomass_reaction') or determine_biomass_reaction(temp_model)
    print(f"\nBiomass reaction: {biomass_reaction}")

    # === Step 1: Prepare model data ===
    print("\n=== Step 1: Preparing model data ===")
    if not FORCE_REGENERATE and os.path.exists(substrates_output) and os.path.exists(sequences_output):
        print("  ✓ Found existing files, loading cached data:")
        print(f"    - {substrates_output}")
        print(f"    - {sequences_output}")
        substrate_df = pd.read_csv(substrates_output)
        sequences_df = pd.read_csv(sequences_output)
        model = load_model(model_path)
    else:
        if FORCE_REGENERATE:
            print("  ⟳ Regenerating model data (--force flag)")
        else:
            print("  No cached files found, preparing model data...")

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
                organism=organism
            )
        print("  ✓ Generated and saved substrates and sequences")

    print(f"  Model: {len(model.genes)} genes, {len(model.reactions)} reactions")

    # === Step 2: Merge substrate and sequence data ===
    print("\n=== Step 2: Merging substrate and sequence data ===")
    if not FORCE_REGENERATE and os.path.exists(merged_data_output):
        print("  ✓ Found existing file, loading cached data:")
        print(f"    - {merged_data_output}")
        merged_data = pd.read_csv(merged_data_output)
    else:
        if FORCE_REGENERATE:
            print("  ⟳ Regenerating merged data (--force flag)")
        else:
            print("  No cached file found, merging data...")
        merged_data = merge_substrate_sequences(
            substrate_df=substrate_df,
            sequences_df=sequences_df,
            model=model,
            output_path=merged_data_output
        )
        print("  ✓ Generated and saved merged data")

    print(f"  Merged data: {len(merged_data)} rows")
    
    # Debug: Check gene distribution in merged data
    if 'Gene' in merged_data.columns:
        unique_genes_merged = merged_data['Gene'].nunique()
        print(f"  [DEBUG] Unique genes in merged data: {unique_genes_merged}")
        print(f"  [DEBUG] Top 5 genes in merged data: {merged_data['Gene'].value_counts().head()}")
    elif 'gene_id' in merged_data.columns:
        unique_genes_merged = merged_data['gene_id'].nunique()
        print(f"  [DEBUG] Unique genes in merged data (gene_id): {unique_genes_merged}")
        print(f"  [DEBUG] Top 5 genes in merged data: {merged_data['gene_id'].value_counts().head()}")

    # === Step 3: Process kcat predictions ===
    # Find predictions file with flexible naming
    predictions_csv_path = find_predictions_file(model_name, CPIPred_data_dir)
    print("\n=== Step 3: Processing CPI-Pred kcat values & annotating model ===")
    if not FORCE_REGENERATE and os.path.exists(processed_data_output):
        print("  ✓ Found existing file, loading cached data:")
        print(f"    - {processed_data_output}")
        processed_data = pd.read_csv(processed_data_output)
    else:
        if FORCE_REGENERATE:
            print("  ⟳ Regenerating processed data (--force flag)")
        else:
            print("  No cached file found, processing kcat predictions...")
        processed_data = process_kcat_predictions(
            merged_df=merged_data,
            predictions_csv_path=predictions_csv_path,
            output_path=processed_data_output
        )
        print("  ✓ Generated and saved processed data")

    print(f"  Processed data: {len(processed_data)} rows")
    
    # Debug: Check gene distribution in processed data
    gene_columns = [col for col in processed_data.columns if 'gene' in col.lower()]
    print(f"  [DEBUG] Gene-related columns in processed data: {gene_columns}")
    
    for gene_col in gene_columns:
        unique_genes = processed_data[gene_col].nunique()
        print(f"  [DEBUG] Unique genes in {gene_col}: {unique_genes}")
        print(f"  [DEBUG] Top 5 genes in {gene_col}: {processed_data[gene_col].value_counts().head()}")
        
    # Check for b0929 specifically
    if any('gene' in col.lower() for col in processed_data.columns):
        for gene_col in gene_columns:
            b0929_count = (processed_data[gene_col] == 'b0929').sum()
            total_count = len(processed_data)
            print(f"  [DEBUG] {gene_col}: {b0929_count}/{total_count} entries are b0929 ({b0929_count/total_count*100:.1f}%)")

    # Ensure kcat column exists
    if 'kcat_mean' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_mean']
    elif 'kcat_y' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_y']

    # Debug: Check model genes before annotation
    print(f"  [DEBUG] Model has {len(model.genes)} genes total")
    print(f"  [DEBUG] First 10 model gene IDs: {[g.id for g in list(model.genes)[:10]]}")
    
    # Annotate model
    print("  Annotating model with kcat and GPR data...")
    model = annotate_model_with_kcat_and_gpr(model=model, df=processed_data)

    rxn_with_kcat = sum(1 for rxn in model.reactions
                        if hasattr(rxn, 'annotation') and 'kcat' in rxn.annotation
                        and rxn.annotation['kcat'] not in [None, '', 0, '0'])
    print(f"  Reactions with kcat: {rxn_with_kcat}/{len(model.reactions)}")
    
    # Debug: Check a few annotated reactions
    print("  [DEBUG] Sample annotated reactions:")
    annotated_count = 0
    for rxn in model.reactions:
        if hasattr(rxn, 'annotation') and 'kcat' in rxn.annotation and rxn.annotation['kcat'] not in [None, '', 0, '0']:
            if 'gpr' in rxn.annotation:
                print(f"    {rxn.id}: kcat={rxn.annotation['kcat']}, gpr={rxn.annotation.get('gpr', 'None')}")
                annotated_count += 1
                if annotated_count >= 5:  # Show first 5 annotated reactions
                    break

    # === Step 4: Optimization ===
    print("\n=== Step 4: Running optimization ===")

    # First run standard COBRApy optimization for comparison
    print("  Running standard COBRApy FBA (no enzyme constraints)...")
    cobra_solution = model.optimize()
    cobra_biomass = cobra_solution.objective_value
    print(f"    COBRApy biomass: {cobra_biomass:.4f}")

    # Now run enzyme-constrained optimization
    print("  Running kinGEMs enzyme-constrained optimization...")

    # Optimize performance: disable bidirectional constraints if not needed
    # This can significantly reduce memory usage and solve time
    use_bidirectional = config.get('bidirectional_constraints', True)
    if not use_bidirectional:
        print("    Using standard constraints (bidirectional disabled for speed)")

    (solution_value, df_FBA, gene_sequences_dict, _) = run_optimization_with_dataframe(
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
        print_reaction_conditions=True,
        verbose=True,  # Enable detailed timing output
        solver_name=solver_name,
        bidirectional_constraints=use_bidirectional  # Add performance option
    )
    print(f"    kinGEMs biomass: {solution_value:.4f}")
    
    # Debug: Check gene_sequences_dict
    print(f"  [DEBUG] gene_sequences_dict has {len(gene_sequences_dict)} entries")
    if gene_sequences_dict:
        sample_genes = list(gene_sequences_dict.keys())[:5]
        print(f"  [DEBUG] Sample genes in gene_sequences_dict: {sample_genes}")
        for gene in sample_genes:
            print(f"    {gene}: {len(gene_sequences_dict[gene])} characters")
    else:
        print("  [DEBUG] WARNING: gene_sequences_dict is empty!")

    # Show comparison
    reduction = (1 - solution_value / cobra_biomass) * 100 if cobra_biomass > 0 else 0
    print("\n  Comparison:")
    print(f"    Standard FBA:           {cobra_biomass:.4f}")
    print(f"    Enzyme-constrained FBA: {solution_value:.4f}")
    print(f"    Reduction due to enzyme budget: {reduction:.1f}%")

    # Memory cleanup after optimization
    print("  Cleaning up optimization memory...")
    del df_FBA  # Large result dataframe no longer needed
    gc.collect()  # Force garbage collection

    # For ModelSEED models, we need to let the optimization system handle constraints
    # Don't manually add enzyme constraints - let simulated annealing use the optimization framework
    if is_modelseed:
        print("  Using ModelSEED model - constraints handled by optimization framework")
        # Use the processed_data as-is, constraints will be calculated during optimization
        constraint_data = processed_data
    else:
        print("  Using standard model - may need additional constraint processing")
        constraint_data = processed_data

    # === Step 5: Simulated Annealing ===
    print("\n=== Step 5: Running simulated annealing ===")
    sa_config = config.get('simulated_annealing', {})
    temperature = sa_config.get('temperature', 1.0)
    cooling_rate = sa_config.get('cooling_rate', 0.95)
    min_temperature = sa_config.get('min_temperature', 0.01)
    max_iterations = sa_config.get('max_iterations', 100)
    max_unchanged_iterations = sa_config.get('max_unchanged_iterations', 5)
    change_threshold = sa_config.get('change_threshold', 0.009)
    biomass_goal = sa_config.get('biomass_goal', 0.5)
    verbose = sa_config.get('verbose', False)

    print("  Configuration:")
    print(f"    - Temperature: {temperature}")
    print(f"    - Cooling rate: {cooling_rate}")
    print(f"    - Max iterations: {max_iterations}")
    print(f"    - Max unchanged iterations: {max_unchanged_iterations}")
    print(f"    - Change threshold: {change_threshold}")
    print(f"    - Biomass goal: {biomass_goal}")
    print("  Starting optimization...\n")

    kcat_dict, top_targets, df_new, iterations, biomasses, df_FBA = simulated_annealing(
        model=model,
        processed_data=constraint_data,
        biomass_reaction=biomass_reaction,
        objective_value=biomass_goal,
        gene_sequences_dict=gene_sequences_dict,
        output_dir=tuning_results_dir,
        enzyme_fraction=enzyme_upper_bound,
        temperature=temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        max_iterations=max_iterations,
        max_unchanged_iterations=max_unchanged_iterations,
        change_threshold=change_threshold,
        verbose=verbose
    )

    improvement = (biomasses[-1] - biomasses[0]) / biomasses[0] * 100 if biomasses[0] > 0 else 0
    print("\n  Annealing complete!")
    print(f"  Initial biomass: {biomasses[0]:.4f}")
    print(f"  Final biomass: {biomasses[-1]:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Total iterations: {len(iterations)}")

    # Show biomass progression
    if len(biomasses) > 1:
        print("\n  Biomass progression:")
        step = max(1, len(biomasses) // 10)  # Show up to 10 checkpoints
        for i in range(0, len(biomasses), step):
            if i == 0:
                print(f"    Iter {iterations[i]:3d}: {biomasses[i]:.6f}")
            else:
                change_pct = (biomasses[i] - biomasses[i-step]) / biomasses[i-step] * 100 if i >= step else 0
                print(f"    Iter {iterations[i]:3d}: {biomasses[i]:.6f} ({change_pct:+.2f}%)")
        if len(biomasses) - 1 not in range(0, len(biomasses), step):
            idx = len(biomasses) - 1
            change_pct = (biomasses[idx] - biomasses[idx-1]) / biomasses[idx-1] * 100
            print(f"    Iter {iterations[idx]:3d}: {biomasses[idx]:.6f} ({change_pct:+.2f}%)")
    print("\n  Top 10 enzymes by mass contribution:")
    print(top_targets[['Reactions', 'Single_gene', 'enzyme_mass']].head(10))

    # Debug: Investigate gene mapping issue
    print("\n  [DEBUG] Gene distribution analysis:")
    print(f"  [DEBUG] df_new shape: {df_new.shape}")
    print(f"  [DEBUG] df_new columns: {list(df_new.columns)}")
    
    if 'Single_gene' in df_new.columns:
        unique_genes_final = df_new['Single_gene'].nunique()
        total_entries = len(df_new)
        b0929_count_final = (df_new['Single_gene'] == 'b0929').sum()
        print(f"  [DEBUG] Final unique genes: {unique_genes_final}")
        print(f"  [DEBUG] Final b0929 entries: {b0929_count_final}/{total_entries} ({b0929_count_final/total_entries*100:.1f}%)")
        print("  [DEBUG] Top 10 genes in final results:")
        print(df_new['Single_gene'].value_counts().head(10))
        
    # Compare with original processed data
    print("\n  [DEBUG] Comparing processed_data vs df_new:")
    if 'Single_gene' in constraint_data.columns:
        orig_unique = constraint_data['Single_gene'].nunique()
        orig_b0929 = (constraint_data['Single_gene'] == 'b0929').sum()
        print(f"  [DEBUG] Original processed_data unique genes: {orig_unique}")
        print(f"  [DEBUG] Original processed_data b0929 entries: {orig_b0929}/{len(constraint_data)} ({orig_b0929/len(constraint_data)*100:.1f}%)")

    # Merge kcat_dict into df_new
    df_new_path = os.path.join(tuning_results_dir, "df_new.csv")
    df_new.to_csv(df_new_path, index=False)
    kcat_dict_path = os.path.join(tuning_results_dir, "kcat_dict.csv")
    kcat_dict_df = pd.read_csv(kcat_dict_path)
    if 'reaction_gene' not in kcat_dict_df.columns:
        kcat_dict_df.columns = ['reaction_gene', 'kcat_value']
    df_new['reaction_gene'] = df_new['Reactions'].astype(str) + '_' + df_new['Single_gene'].astype(str)
    df_new = df_new.merge(kcat_dict_df, on='reaction_gene', how='left')
    df_new.rename(columns={'kcat_value': 'kcat_tuned'}, inplace=True)
    final_info_path = os.path.join(tuning_results_dir, "final_model_info.csv")
    df_new.to_csv(final_info_path, index=False)
    print(f"\n  Saved merged DataFrame to: {final_info_path}")

    # === Step 6: Optional analyses ===
    if enable_fva:
        fva_config = config.get('fva', {})
        run_fva_analysis(model, df_new, biomass_reaction, enzyme_upper_bound,
                        tuning_results_dir, model_name, fva_config)

    if enable_biolog:
        biolog_config = config.get('biolog_validation', {})
        run_biolog_validation(model, processed_data, biomass_reaction, enzyme_upper_bound,
                             biolog_config, tuning_results_dir, solver_name)

    # === Step 7: Save Final Model ===
    print("\n=== Step 7: Saving final model ===")
    model_output_dir = os.path.join(project_root, "models")
    os.makedirs(model_output_dir, exist_ok=True)
    model_output_path = os.path.join(model_output_dir, f"{run_id}.xml")

    model_with_kcats = assign_kcats_to_model(model, df_new)
    model_with_kcats = clean_annotations(model_with_kcats)
    write_sbml_model(model_with_kcats, model_output_path)
    print(f"  Final GEM saved to: {model_output_path}")

    # === Summary ===
    print("\n" + "="*70)
    print("=== Pipeline Complete ===")
    print("="*70)
    print(f"Run ID: {run_id}")
    print(f"Model: {model_name} ({model_type})")
    print(f"Initial biomass: {biomasses[0]:.4f}")
    print(f"Final biomass: {biomasses[-1]:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Iterations: {len(iterations)}")
    print(f"\nResults directory: {tuning_results_dir}")
    print(f"Final model: {model_output_path}")
    print("="*70)


if __name__ == '__main__':
    main()

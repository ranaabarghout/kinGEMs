#!/usr/bin/env python3
"""
Tuning Reproducibility Study
==============================

Runs simulated annealing N times from the same initial state to assess
whether the kcat optimization produces consistent results across runs.

Each run uses a different random seed but identical starting data.

Usage:
    python scripts/run_tuning_reproducibility.py configs/iML1515_GEM.json
    python scripts/run_tuning_reproducibility.py configs/iML1515_GEM.json --n-runs 10
    python scripts/run_tuning_reproducibility.py configs/iML1515_GEM.json \\
        --n-runs 5 --output-dir results/reproducibility/iML1515_GEM --base-seed 42

Arguments:
    config_path     Path to JSON configuration file
    --n-runs        Number of SA runs to perform (default: 10)
    --output-dir    Directory to save results (default: results/reproducibility/<model_name>_<timestamp>)
    --base-seed     Base random seed; run i uses seed base_seed+i (default: 0)
    --force         Force regeneration of intermediate files (processed data, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.dataset import (
    annotate_model_with_kcat_and_gpr,
    load_model,
    merge_substrate_sequences,
    prepare_model_data,
    process_kcat_predictions,
    convert_to_irreversible,
)
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.modeling.tuning import simulated_annealing
from kinGEMs.plots import (
    plot_reproducibility_kcat_kde_overlay,
    plot_reproducibility_biomass_per_run,
    plot_reproducibility_convergence,
    plot_reproducibility_fold_change_heatmap,
    plot_reproducibility_kcat_scatter_overlay,
    plot_kcat_annealing_comparison_by_subsystem,
)


# ---------------------------------------------------------------------------
# Helpers mirrored from run_pipeline.py
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def determine_biomass_reaction(model) -> str:
    obj_rxns = {r.id: r.objective_coefficient for r in model.reactions if r.objective_coefficient != 0}
    if not obj_rxns:
        raise ValueError("No objective reaction found in model")
    return next(iter(obj_rxns.keys()))


def find_predictions_file(model_name: str, cpipred_dir: str) -> str:
    import glob
    patterns = [
        f"X06A_kinGEMs_{model_name}_predictions.csv",
        f"*{model_name}*predictions.csv",
    ]
    if '_GEM' in model_name:
        base = model_name.replace('_GEM', '')
        patterns += [f"*ecoli_{base}*predictions.csv", f"*{base}*predictions.csv"]
    for pat in patterns:
        matches = glob.glob(os.path.join(cpipred_dir, pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No CPI-Pred predictions for '{model_name}' in {cpipred_dir}")


# ---------------------------------------------------------------------------
# Setup: load model + processed data + gene_sequences_dict (run once)
# ---------------------------------------------------------------------------

def setup_pipeline(config: dict, project_root: str, force: bool = False):
    """
    Run pipeline steps 1-4 once and return everything needed to replay SA.

    Returns
    -------
    model : cobra.Model
        Irreversible, annotated enzyme-constrained model
    processed_data : pd.DataFrame
        Initial kcat data (kcat_mean preserved throughout)
    gene_sequences_dict : dict
        Gene → protein sequence mapping (needed by SA)
    initial_biomass : float
        Biomass from first EC-FBA call (before any SA)
    config_info : dict
        Key scalars extracted from config for easy access
    """
    model_name       = config['model_name']
    organism         = config.get('organism', 'Unknown')
    enzyme_ub        = config.get('enzyme_upper_bound', 0.15)
    solver_name      = config.get('solver', 'glpk')
    medium           = config.get('medium', None)
    medium_upper_bound = config.get('medium_upper_bound', True)
    biomass_rxn_cfg  = config.get('biomass_reaction', None)
    results_subdir   = config.get('results_subdir', None)

    data_dir          = os.path.join(project_root, 'data')
    raw_data_dir      = os.path.join(data_dir, 'raw')
    interim_dir       = os.path.join(data_dir, 'interim', model_name)
    processed_dir     = os.path.join(data_dir, 'processed', model_name)
    cpipred_dir       = os.path.join(data_dir, 'interim', 'CPI-Pred predictions')

    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if results_subdir == 'BiGG_models':
        model_path = os.path.join(raw_data_dir, 'BiGG_models', f'{model_name}.xml')
    else:
        model_path = os.path.join(raw_data_dir, f'{model_name}.xml')

    substrates_out = os.path.join(interim_dir, f'{model_name}_substrates.csv')
    sequences_out  = os.path.join(interim_dir, f'{model_name}_sequences.csv')
    merged_out     = os.path.join(interim_dir, f'{model_name}_merged_data.csv')
    processed_out  = os.path.join(processed_dir, f'{model_name}_processed_data.csv')

    # ---- Step 1: Prepare model data ----
    print('[Setup] Step 1 – Prepare model data')
    if not force and os.path.exists(substrates_out) and os.path.exists(sequences_out):
        print('  Using cached substrates / sequences')
        substrate_df = pd.read_csv(substrates_out)
        sequences_df = pd.read_csv(sequences_out)
        model = load_model(model_path)
        model.solver = solver_name
        model = convert_to_irreversible(model)
    else:
        model, substrate_df, sequences_df = prepare_model_data(
            model_path=model_path,
            substrates_output=substrates_out,
            sequences_output=sequences_out,
            organism=organism,
            convert_irreversible=True,
        )

    model.solver = solver_name
    print(f'  Model: {len(model.genes)} genes, {len(model.reactions)} reactions')

    # ---- Step 2: Merge substrate + sequence data ----
    print('[Setup] Step 2 – Merge data')
    if not force and os.path.exists(merged_out):
        merged_data = pd.read_csv(merged_out)
    else:
        merged_data = merge_substrate_sequences(
            substrate_df=substrate_df,
            sequences_df=sequences_df,
            model=model,
            output_path=merged_out,
        )

    # ---- Step 3: Process kcat predictions ----
    print('[Setup] Step 3 – Process kcat predictions')
    if not force and os.path.exists(processed_out):
        processed_data = pd.read_csv(processed_out)
    else:
        predictions_path = find_predictions_file(model_name, cpipred_dir)
        processed_data = process_kcat_predictions(
            merged_df=merged_data,
            predictions_csv_path=predictions_path,
            output_path=processed_out,
        )

    # Normalise kcat column name
    if 'kcat_mean' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_mean']
    elif 'kcat_y' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_y']

    print(f'  Processed data: {len(processed_data)} rows')

    # Annotate model
    model = annotate_model_with_kcat_and_gpr(model=model, df=processed_data)

    # Determine biomass reaction
    biomass_rxn = biomass_rxn_cfg or determine_biomass_reaction(model)
    print(f'  Biomass reaction: {biomass_rxn}')

    # Apply medium constraints if defined
    if medium is not None:
        for rxn_id, flux_val in medium.items():
            try:
                rxn = model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = flux_val
                if medium_upper_bound:
                    rxn.upper_bound = flux_val
            except KeyError:
                print(f'    Warning: reaction {rxn_id} not found')

    # ---- Step 4: Enzyme-constrained FBA to get gene_sequences_dict ----
    print('[Setup] Step 4 – EC-FBA (initial state)')
    solution_value, df_FBA, gene_sequences_dict, _ = run_optimization_with_dataframe(
        model=model,
        processed_df=processed_data,
        objective_reaction=biomass_rxn,
        enzyme_upper_bound=enzyme_ub,
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
    print(f'  Initial kinGEMs biomass: {solution_value:.4f}')

    config_info = dict(
        model_name=model_name,
        enzyme_ub=enzyme_ub,
        solver_name=solver_name,
        biomass_rxn=biomass_rxn,
        medium=medium,
        medium_upper_bound=medium_upper_bound,
    )

    return model, processed_data, gene_sequences_dict, solution_value, config_info


# ---------------------------------------------------------------------------
# Run N reproducibility trials
# ---------------------------------------------------------------------------

def run_reproducibility_trials(
    model,
    processed_data: pd.DataFrame,
    gene_sequences_dict: dict,
    config: dict,
    config_info: dict,
    n_runs: int,
    output_dir: str,
    base_seed: int,
):
    """
    Run simulated annealing *n_runs* times with different random seeds.

    Returns
    -------
    run_results : list[dict]
        One dict per run with keys: run_id, seed, final_biomass, biomasses, df_new
    """
    sa_cfg       = config.get('simulated_annealing', {})
    enzyme_ub    = config_info['enzyme_ub']
    biomass_rxn  = config_info['biomass_rxn']
    medium       = config_info['medium']
    medium_ub    = config_info['medium_upper_bound']

    run_results = []

    for i in range(n_runs):
        seed = base_seed + i
        run_label = f'run_{i + 1:02d}'
        run_dir   = os.path.join(output_dir, run_label)
        os.makedirs(run_dir, exist_ok=True)

        print(f'\n[Run {i + 1}/{n_runs}] seed={seed}  →  {run_dir}')

        # Set all random seeds so each run is independently reproducible
        random.seed(seed)
        np.random.seed(seed)

        kcat_dict, top_targets, df_new, iterations, biomasses, df_FBA = simulated_annealing(
            model=model,
            processed_data=processed_data.copy(),   # pass a fresh copy each time
            biomass_reaction=biomass_rxn,
            objective_value=sa_cfg.get('biomass_goal', 0.5),
            gene_sequences_dict=gene_sequences_dict,
            output_dir=run_dir,
            enzyme_fraction=enzyme_ub,
            n_top_enzymes=sa_cfg.get('n_top_enzymes', 65),
            temperature=sa_cfg.get('temperature', 1.0),
            cooling_rate=sa_cfg.get('cooling_rate', 0.95),
            min_temperature=sa_cfg.get('min_temperature', 0.01),
            max_iterations=sa_cfg.get('max_iterations', 100),
            max_unchanged_iterations=sa_cfg.get('max_unchanged_iterations', 5),
            change_threshold=sa_cfg.get('change_threshold', 0.009),
            verbose=sa_cfg.get('verbose', False),
            medium=medium,
            medium_upper_bound=medium_ub,
        )

        final_biomass = biomasses[-1] if biomasses else float('nan')
        initial_biomass = biomasses[0] if biomasses else float('nan')
        improvement_pct = (
            (final_biomass - initial_biomass) / initial_biomass * 100
            if initial_biomass and initial_biomass > 0 else float('nan')
        )
        print(f'  Initial: {initial_biomass:.4f}  Final: {final_biomass:.4f}  '
              f'Improvement: {improvement_pct:.1f}%  Iterations: {len(iterations)}')

        # Save per-run CSVs
        df_new.to_csv(os.path.join(run_dir, 'df_new.csv'), index=False)
        pd.DataFrame({'iteration': range(len(biomasses)), 'biomass': biomasses}).to_csv(
            os.path.join(run_dir, 'biomasses.csv'), index=False
        )
        pd.DataFrame({'iteration': iterations}).to_csv(
            os.path.join(run_dir, 'iterations.csv'), index=False
        )

        run_results.append(dict(
            run_id=run_label,
            seed=seed,
            initial_biomass=initial_biomass,
            final_biomass=final_biomass,
            improvement_pct=improvement_pct,
            n_iterations=len(iterations),
            biomasses=biomasses,
            df_new=df_new,
        ))

    return run_results


# ---------------------------------------------------------------------------
# Plotting  (functions live in kinGEMs/plots.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_and_save_summary(run_results: list[dict], output_path: str):
    """Save a CSV with per-run summary statistics."""
    rows = []
    for res in run_results:
        df = res['df_new']
        fc_vals = []
        if 'kcat_mean' in df.columns and 'kcat_updated' in df.columns:
            merged = df[['kcat_mean', 'kcat_updated']].dropna()
            merged = merged[(merged['kcat_mean'] > 0) & (merged['kcat_updated'] > 0)]
            if not merged.empty:
                fc = merged['kcat_updated'] / merged['kcat_mean']
                fc_vals = fc.tolist()

        rows.append({
            'run_id':           res['run_id'],
            'seed':             res['seed'],
            'initial_biomass':  res['initial_biomass'],
            'final_biomass':    res['final_biomass'],
            'improvement_pct':  res['improvement_pct'],
            'n_iterations':     res['n_iterations'],
            'median_fold_change': float(np.median(fc_vals)) if fc_vals else float('nan'),
            'mean_fold_change':   float(np.mean(fc_vals))   if fc_vals else float('nan'),
            'std_fold_change':    float(np.std(fc_vals))    if fc_vals else float('nan'),
            'n_increased': int(np.sum(np.array(fc_vals) > 1.0)) if fc_vals else 0,
            'n_decreased': int(np.sum(np.array(fc_vals) < 1.0)) if fc_vals else 0,
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False)
    print(f'  Saved summary: {output_path}')

    # Print concise table to console
    print('\n' + '='*70)
    print(f'{"Run":<10} {"Seed":>5} {"Init Bio":>10} {"Final Bio":>10} '
          f'{"Improv %":>10} {"Median FC":>10} {"Iters":>7}')
    print('-'*70)
    for _, row in summary_df.iterrows():
        print(f'{row["run_id"]:<10} {row["seed"]:>5} {row["initial_biomass"]:>10.4f} '
              f'{row["final_biomass"]:>10.4f} {row["improvement_pct"]:>10.1f} '
              f'{row["median_fold_change"]:>10.2f} {row["n_iterations"]:>7}')
    print('='*70)

    finals = summary_df['final_biomass'].dropna()
    print(f'\nFinal biomass: mean={finals.mean():.4f}, '
          f'std={finals.std():.4f}, '
          f'CV={finals.std() / finals.mean() * 100:.2f}%')
    print(f'              min={finals.min():.4f}, max={finals.max():.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run simulated annealing N times and compare kcat reproducibility.'
    )
    parser.add_argument('config_path', help='Path to JSON configuration file')
    parser.add_argument('--n-runs', type=int, default=10,
                        help='Number of SA runs to perform (default: 10)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: results/reproducibility/<model>_<timestamp>)')
    parser.add_argument('--base-seed', type=int, default=0,
                        help='Base random seed; run i uses seed base_seed+i (default: 0)')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration of all intermediate files')
    args = parser.parse_args()

    # ---- Paths ----
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(args.config_path)
    model_name = config['model_name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            project_root, 'results', 'reproducibility',
            f'{model_name}_{timestamp}'
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  Tuning Reproducibility Study')
    print(f'  Model:      {model_name}')
    print(f'  Runs:       {args.n_runs}')
    print(f'  Base seed:  {args.base_seed}')
    print(f'  Output dir: {output_dir}')
    print(f'{"="*60}\n')

    # ---- Setup (run once) ----
    model, processed_data, gene_sequences_dict, initial_biomass, config_info = (
        setup_pipeline(config, project_root, force=args.force)
    )

    # ---- Attach subsystem column from model reactions ----
    rxn_subsystem = {r.id: (r.subsystem if r.subsystem else 'Other') for r in model.reactions}
    if 'Reactions' in processed_data.columns:
        processed_data = processed_data.copy()
        processed_data['subsystem'] = processed_data['Reactions'].map(rxn_subsystem).fillna('Other')

    # ---- Reproducibility trials ----
    print(f'\n{"="*60}')
    print(f'  Running {args.n_runs} SA trials …')
    print(f'{"="*60}')

    run_results = run_reproducibility_trials(
        model=model,
        processed_data=processed_data,
        gene_sequences_dict=gene_sequences_dict,
        config=config,
        config_info=config_info,
        n_runs=args.n_runs,
        output_dir=output_dir,
        base_seed=args.base_seed,
    )

    # ---- Summary CSV ----
    print(f'\n{"="*60}')
    print('  Saving summary statistics …')
    print(f'{"="*60}')
    compute_and_save_summary(
        run_results,
        os.path.join(output_dir, 'reproducibility_summary.csv'),
    )

    # ---- Plots ----
    print(f'\n{"="*60}')
    print('  Generating comparison plots …')
    print(f'{"="*60}')

    plot_reproducibility_kcat_kde_overlay(
        processed_data=processed_data,
        run_results=run_results,
        output_path=os.path.join(output_dir, 'kcat_kde_overlay.png'),
        model_name=model_name,
    )

    plot_reproducibility_biomass_per_run(
        run_results=run_results,
        output_path=os.path.join(output_dir, 'biomass_per_run.png'),
        model_name=model_name,
    )

    plot_reproducibility_convergence(
        run_results=run_results,
        output_path=os.path.join(output_dir, 'convergence_comparison.png'),
        model_name=model_name,
    )

    plot_reproducibility_kcat_scatter_overlay(
        processed_data=processed_data,
        run_results=run_results,
        output_path=os.path.join(output_dir, 'kcat_scatter_overlay.png'),
        model_name=model_name,
    )

    plot_reproducibility_fold_change_heatmap(
        processed_data=processed_data,
        run_results=run_results,
        output_path=os.path.join(output_dir, 'kcat_fold_change_heatmap.png'),
        model_name=model_name,
        top_n=40,
    )

    # ---- Subsystem comparison plots (per run + compiled) ----
    if 'subsystem' in processed_data.columns:
        print('  Generating subsystem comparison plots …')
        subsystem_dir = os.path.join(output_dir, 'subsystem_plots')
        os.makedirs(subsystem_dir, exist_ok=True)

        # Per-run plots
        for run_idx, run in enumerate(run_results, start=1):
            plot_kcat_annealing_comparison_by_subsystem(
                initial_df=processed_data,
                tuned_df=run['df_new'],
                subsystem_col='subsystem',
                output_path=os.path.join(subsystem_dir, f'subsystem_run{run_idx:02d}.png'),
                model_name=f'{model_name} – Run {run_idx}',
            )

        # Compiled plot: median kcat_updated across all runs per reaction-gene pair
        merge_cols = [c for c in ('Reactions', 'Single_gene') if c in run_results[0]['df_new'].columns]
        all_tuned = pd.concat([r['df_new'] for r in run_results], ignore_index=True)
        compiled_tuned = (
            all_tuned.groupby(merge_cols, as_index=False)['kcat_updated'].median()
            if merge_cols
            else all_tuned[['kcat_updated']].median().to_frame().T
        )
        plot_kcat_annealing_comparison_by_subsystem(
            initial_df=processed_data,
            tuned_df=compiled_tuned,
            subsystem_col='subsystem',
            output_path=os.path.join(subsystem_dir, 'subsystem_compiled_median.png'),
            model_name=f'{model_name} – Compiled (median across {len(run_results)} runs)',
        )
        print(f'  Subsystem plots saved to: {subsystem_dir}')

    print(f'\n✅ Done.  All outputs saved to:\n   {output_dir}\n')


if __name__ == '__main__':
    main()

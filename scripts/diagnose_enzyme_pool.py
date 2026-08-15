#!/usr/bin/env python3
"""Cheap enzyme-pool feasibility probes (no simulated annealing).

Tests A–D from the enzyme-pool feasibility plan, using
configs/r_toruloides_xy_lim.json by default.

Usage:
    python scripts/diagnose_enzyme_pool.py [config_file]
"""
from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy

import cobra
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))

from kinGEMs.dataset import convert_to_irreversible, load_model
from kinGEMs.modeling.optimize import apply_medium_to_model, run_optimization_with_dataframe
from kinGEMs.modeling.tuning import (
    apply_biomass_composition,
    apply_model_edits,
    find_min_feasible_pool,
)
from run_pipeline import apply_fixed_maintenance, select_kcat_source_column


def _probe(model, processed_df, biomass_reaction, pool, kcat_col, medium, medium_upper_bound, solver_name):
    t0 = time.time()
    result = run_optimization_with_dataframe(
        model=model,
        processed_df=processed_df,
        objective_reaction=biomass_reaction,
        enzyme_upper_bound=pool,
        output_dir=None,
        save_results=False,
        verbose=False,
        medium=medium,
        medium_upper_bound=medium_upper_bound,
        kcat_col=kcat_col,
        solver_name=solver_name,
    )
    elapsed = time.time() - t0
    if not isinstance(result, tuple) or result[0] is None or result[0] <= 0:
        return None, elapsed
    return float(result[0]), elapsed


def prepare_model_and_data(config, project_root):
    model_name = config['model_name']
    solver_name = config.get('solver', 'glpk')
    biomass_reaction = config.get('biomass_reaction', 'r_2111')
    medium = config.get('medium')
    medium_upper_bound = config.get('medium_upper_bound', False)

    processed_path = os.path.join(
        project_root, 'data', 'processed', model_name, f'{model_name}_processed_data.csv'
    )
    model_path = os.path.join(project_root, 'data', 'raw', f'{model_name}.xml')
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f'Missing cached processed data: {processed_path}')

    processed_data = pd.read_csv(processed_path)
    kcat_source = select_kcat_source_column(processed_data, config, log=print)

    model = load_model(model_path)
    model.solver = solver_name
    model = convert_to_irreversible(model)

    model_edits = config.get('model_edits')
    if model_edits:
        apply_model_edits(model, model_edits, verbose=False)
        model = convert_to_irreversible(model)
    if medium is not None:
        apply_medium_to_model(model, medium, medium_upper_bound=medium_upper_bound, verbose=False)
    biomass_composition = config.get('biomass_composition')
    if biomass_composition:
        gam_rxn_id = config.get('gam_reaction') or biomass_reaction
        multipliers = biomass_composition.get('multipliers', {})
        if multipliers:
            apply_biomass_composition(
                model, biomass_pseudoreaction=gam_rxn_id, multipliers=multipliers, verbose=False
            )
    apply_fixed_maintenance(model, config, biomass_reaction, log=print)
    return model, processed_data, kcat_source, biomass_reaction, medium, medium_upper_bound, solver_name


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        project_root, 'configs', 'r_toruloides_xy_lim.json'
    )
    with open(config_path) as f:
        config = json.load(f)

    pool = float(config.get('enzyme_upper_bound', 0.03))
    print(f'Loading {config_path}')
    print(f'Target enzyme pool: {pool:.5f} g/gDW')
    model, processed, kcat_source, biomass_rxn, medium, mub, solver = prepare_model_and_data(
        config, project_root
    )
    cobra_mu = model.slim_optimize()
    print(f'COBRApy biomass: {cobra_mu:.4f}')
    ngam_id = config.get('ngam_rxn_id', 'r_4046')
    try:
        default_ngam = model.reactions.get_by_id(ngam_id).lower_bound
    except KeyError:
        default_ngam = None
    print(f'NGAM ({ngam_id}) lb: {default_ngam}')

    results = []

    def record(name, kcat_col, pool_val, ngam, biomass, elapsed):
        status = 'feasible' if biomass is not None else 'infeasible'
        mu_str = f'{biomass:.4f}' if biomass is not None else 'None'
        print(f'  [{name}] kcat={kcat_col} pool={pool_val:.5f} NGAM={ngam} -> {status} mu={mu_str} ({elapsed:.1f}s)')
        results.append({
            'test': name, 'kcat_col': kcat_col, 'pool': pool_val,
            'ngam': ngam, 'biomass': biomass, 'status': status, 'seconds': round(elapsed, 1),
        })

    # A. kcat_mean, experimental pool, current NGAM
    print('\n=== Test A: kcat_mean, pool=target, current NGAM ===')
    df_mean = processed.copy()
    if 'kcat' in df_mean.columns:
        df_mean = df_mean.drop(columns=['kcat'])
    mu, dt = _probe(model, df_mean, biomass_rxn, pool, 'kcat_mean', medium, mub, solver)
    record('A', 'kcat_mean', pool, default_ngam, mu, dt)

    # B. kcat_max, experimental pool, current NGAM
    print('\n=== Test B: kcat_max, pool=target, current NGAM ===')
    mu, dt = _probe(model, processed, biomass_rxn, pool, 'kcat_max', medium, mub, solver)
    record('B', 'kcat_max', pool, default_ngam, mu, dt)

    # C. kcat_max, experimental pool, NGAM=0
    print('\n=== Test C: kcat_max, pool=target, NGAM=0 ===')
    model_c = deepcopy(model)
    if default_ngam is not None:
        model_c.reactions.get_by_id(ngam_id).lower_bound = 0.0
    mu, dt = _probe(model_c, processed, biomass_rxn, pool, 'kcat_max', medium, mub, solver)
    record('C', 'kcat_max', pool, 0.0, mu, dt)

    # D. binary-search min feasible pool
    print('\n=== Test D: min feasible pool (kcat_mean and kcat_max) ===')
    for col in ('kcat_mean', 'kcat_max'):
        print(f'  Searching with {col}...')
        t0 = time.time()
        seed, seed_mu, _, _ = find_min_feasible_pool(
            model=model,
            processed_df=processed if col == 'kcat_max' else df_mean,
            biomass_reaction=biomass_rxn,
            target_pool=pool,
            hi_pool=0.25,
            kcat_col=col,
            medium=medium,
            medium_upper_bound=mub,
            solver_name=solver,
            n_iter=6,
            skip_target=(col == 'kcat_max' and results[1]['biomass'] is None)
                        or (col == 'kcat_mean' and results[0]['biomass'] is None),
            log=print,
        )
        dt = time.time() - t0
        record(f'D_{col}', col, seed if seed is not None else pool, default_ngam, seed_mu, dt)
        print(f'  min feasible pool ({col}): {seed}  biomass={seed_mu}')

    out_dir = os.path.join(project_root, 'results', 'tuning_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'enzyme_pool_diagnostics.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()

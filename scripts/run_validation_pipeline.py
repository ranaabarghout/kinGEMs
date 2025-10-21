#!/usr/bin/env python3
"""
kinGEMs Validation Pipeline Script
===================================

This script provides a unified validation pipeline for comparing different model versions:
1. Baseline GEM (no enzyme constraints)
2. Pre-tuning kinGEMs (initial kcat values, before simulated annealing)
3. Post-tuning kinGEMs (tuned kcat values, after simulated annealing)

Usage:
    python scripts/run_validation_pipeline.py <config_file>
    python scripts/run_validation_pipeline.py configs/validation_iML1515.json

Config File Format:
    See example in configs/validation_iML1515.json
"""

from datetime import datetime
import json
import logging
import os
import sys
import warnings

import cobra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve as pre_rec

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.config import ECOLI_VALIDATION_DIR
from kinGEMs.validation_utils import (
    check_environment,
    load_data,
    load_environment,
    match_model_data,
    model_adjustments,
    prepare_model,
    simulate_phenotype,
    simulate_phenotype_parallel,
    test_growth,
)

# Silence warnings
warnings.filterwarnings('ignore')
logging.getLogger('distributed').setLevel(logging.ERROR)


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def calc_metrics(data_sim, data_fit, fit_thresh, sim_thresh):
    """Calculate advanced metrics including AUC, BAC, and ROC AUC."""
    data_sim_b = (data_sim > sim_thresh).astype(int)
    data_fit_b = (data_fit > fit_thresh).astype(int)
    num_genes = np.array([data_fit_b.shape[0]])
    num_genes_add = num_genes - num_genes[0]
    AC_i = np.zeros(len(num_genes_add))
    AUC_i = np.zeros(len(num_genes_add))
    BAC_i = np.zeros(len(num_genes_add))
    ROC_AUC_i = np.zeros(len(num_genes_add))
    for i in range(len(num_genes_add)):
        data_fit_b_V = data_fit_b.flatten()
        data_sim_b_V = data_sim_b.flatten()
        data_fit_V = data_fit.flatten()
        data_fit_b_V = np.append(data_fit_b_V, np.ones(num_genes_add[i]*data_fit_b.shape[1]))
        data_sim_b_V = np.append(data_sim_b_V, np.ones(num_genes_add[i]*data_fit_b.shape[1]))
        data_fit_V = np.append(data_fit_V, np.zeros(num_genes_add[i]*data_fit_b.shape[1]))
        tn, fp, fn, tp = confusion_matrix(data_fit_b_V, data_sim_b_V, labels=[1,0]).ravel()
        AC_i[i] = (tp+tn)/(tn+fn+tp+fp)
        pre, rec, thresholds = pre_rec(data_sim_b_V, data_fit_V*-1, pos_label=0)
        AUC_i[i] = sk_auc(rec, pre)
        TPR = tp/(tp+fn) if (tp+fn) > 0 else 0
        TNR = tn/(tn+fp) if (tn+fp) > 0 else 0
        BAC_i[i] = (TPR+TNR)/2
        ROC_AUC_i[i] = roc_auc_score(data_sim_b_V, data_fit_V)
    return AC_i, AUC_i, BAC_i, ROC_AUC_i


def print_basic_metrics(exp_binary, sim_binary, label):
    """Calculate and print basic performance metrics."""
    true_pos = np.sum((exp_binary == 1) & (sim_binary == 1))
    true_neg = np.sum((exp_binary == 0) & (sim_binary == 0))
    false_pos = np.sum((exp_binary == 0) & (sim_binary == 1))
    false_neg = np.sum((exp_binary == 1) & (sim_binary == 0))
    accuracy = (true_pos + true_neg) / np.prod(exp_binary.shape)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f'{label}:')
    print(f'  Accuracy: {accuracy:.3f}')
    print(f'  Precision: {precision:.3f}')
    print(f'  Recall: {recall:.3f}')
    print(f'  F1 Score: {f1_score:.3f}')
    return accuracy, precision, recall, f1_score


def simulate_baseline_only(model_adj, name_genes_matched_adj, name_carbon_model_matched_adj,
                           medium_ex_inds, carbon_ex_inds):
    """Simulate baseline GEM (no enzyme constraints) only."""
    from copy import deepcopy

    n_genes = len(name_genes_matched_adj)
    n_carbons = len(name_carbon_model_matched_adj)
    baseline_GEM = np.zeros((n_genes, n_carbons))

    # Set medium
    for e in medium_ex_inds:
        if e != -1:
            model_adj.exchanges[e].lower_bound = -1000

    for e, carbon in enumerate(name_carbon_model_matched_adj):
        print(f"  Baseline GEM progress: {e+1}/{n_carbons}", end='\r')

        # Check if this carbon source was already tested
        if carbon in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(carbon)
            baseline_GEM[:, e] = baseline_GEM[:, e_found]
            continue

        # Set carbon source
        if carbon_ex_inds[e] != -1:
            model_adj.exchanges[carbon_ex_inds[e]].lower_bound = -10

        # Test each gene knockout
        for g, gene in enumerate(name_genes_matched_adj):
            with model_adj:
                model_adj.genes.get_by_id(gene).knock_out()
                solution = model_adj.slim_optimize()
                if np.isnan(solution):
                    solution = 0
                baseline_GEM[g, e] = solution

        # Reset carbon source
        if carbon_ex_inds[e] != -1:
            model_adj.exchanges[carbon_ex_inds[e]].lower_bound = 0

        if (e + 1) % 10 == 0:
            print(f"  Baseline GEM progress: {e+1}/{n_carbons} carbon sources completed")

    print(f"\n  Baseline GEM simulation complete ({n_genes} genes × {n_carbons} carbon sources).")
    return baseline_GEM


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config_path = sys.argv[1]

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Extract configuration
    model_name = config['model_name']
    model_path = config['model_path']
    objective_reaction = config.get('objective_reaction', None)
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.15)

    # Data paths (optional, can be auto-detected)
    pre_tuning_data_path = config.get('pre_tuning_data_path', None)
    post_tuning_data_path = config.get('post_tuning_data_path', None)

    # Validation parameters
    sim_thresh = config.get('sim_thresh', 0.001)
    fit_thresh = config.get('fit_thresh', -2)

    # Models to run
    run_baseline = config.get('run_baseline', True)
    run_pre_tuning = config.get('run_pre_tuning', False)
    run_post_tuning = config.get('run_post_tuning', True)

    # Output configuration
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, 'results')

    # Create timestamped validation directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    validation_dir = os.path.join(results_dir, 'validation', f"{model_name}_{timestamp}")
    os.makedirs(validation_dir, exist_ok=True)

    # Get parallel configuration for display
    parallel_config = config.get('parallel', {})
    use_parallel_display = parallel_config.get('enabled', True)
    n_workers_display = parallel_config.get('workers', 'auto')
    parallel_method_display = parallel_config.get('method', 'dask')

    print("\n" + "="*70)
    print(f"=== kinGEMs Validation Pipeline for {model_name} ===")
    print("="*70)
    print(f"Model path: {model_path}")
    print(f"Results directory: {validation_dir}")
    print(f"Run baseline: {run_baseline}")
    print(f"Run pre-tuning: {run_pre_tuning}")
    print(f"Run post-tuning: {run_post_tuning}")
    print(f"\nParallel execution: {use_parallel_display}")
    if use_parallel_display:
        print(f"  Workers: {n_workers_display}")
        print(f"  Method: {parallel_method_display}")
    print("="*70)

    # === Step 1: Load Model and Experimental Data ===
    print("\n=== Step 1: Loading model and experimental data ===")
    model = cobra.io.read_sbml_model(model_path)
    print(f"  Genes in model: {len(model.genes)}")
    print(f"  Reactions in model: {len(model.reactions)}")

    # Auto-detect objective reaction if not specified
    if objective_reaction is None:
        obj_rxns = {rxn.id: rxn.objective_coefficient
                    for rxn in model.reactions
                    if rxn.objective_coefficient != 0}
        if obj_rxns:
            objective_reaction = next(iter(obj_rxns.keys()))
            print(f"  Auto-detected objective reaction: {objective_reaction}")
        else:
            raise ValueError("No objective reaction found in model. Please specify in config.")
    else:
        print(f"  Using objective reaction: {objective_reaction}")

    # Diagnostic: Print kcat coverage
    rxn_with_kcat = sum(1 for rxn in model.reactions
                        if hasattr(rxn, 'annotation') and 'kcat' in rxn.annotation
                        and rxn.annotation['kcat'] not in [None, '', 0, '0'])
    print(f"  Reactions with kcat: {rxn_with_kcat}/{len(model.reactions)}")

    # === Step 2: Apply Validation Functions ===
    print("\n=== Step 2: Preparing validation environment ===")

    # 1. Prepare model (set exchange bounds)
    model = prepare_model(model)

    # 2. Load environment (medium and carbon sources)
    name_medium_model, name_carbon_model, name_carbon_experiment = load_environment(ECOLI_VALIDATION_DIR)

    # 3. Load experimental data
    data_experiments, data_genes, data_fitness = load_data(ECOLI_VALIDATION_DIR)

    # 4. Match model and data
    name_genes_matched, name_carbon_experiment_matched, name_carbon_model_matched, data_fitness_matched = match_model_data(
        model=model,
        name_carbon_model=name_carbon_model,
        name_carbon_experiment=name_carbon_experiment,
        data_experiments=data_experiments,
        data_genes=data_genes,
        data_fitness=data_fitness
    )

    # 5. Adjust model and matched data
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

    # 6. Check environment setup
    medium_ex_inds, carbon_ex_inds = check_environment(
        model_adj=model_adj,
        name_medium_model=name_medium_model,
        name_carbon_model_matched_adj=name_carbon_model_matched_adj
    )

    print(f"  Matched genes: {len(name_genes_matched_adj)}")
    print(f"  Carbon sources: {len(name_carbon_model_matched_adj)}")
    print(f"  Experimental conditions: {data_fitness_matched_adj.shape}")

    # === Step 3: Run Simulations ===
    print("\n=== Step 3: Running growth simulations ===")

    simulation_results = {}

    # 3.1: Baseline GEM
    if run_baseline:
        print("\n--- Running Baseline GEM (no enzyme constraints) ---")
        baseline_GEM = simulate_baseline_only(
            model_adj=model_adj,
            name_genes_matched_adj=name_genes_matched_adj,
            name_carbon_model_matched_adj=name_carbon_model_matched_adj,
            medium_ex_inds=medium_ex_inds,
            carbon_ex_inds=carbon_ex_inds
        )
        simulation_results['baseline'] = baseline_GEM

        # Save results
        np.save(os.path.join(validation_dir, 'baseline_GEM.npy'), baseline_GEM)

    # Get parallel configuration
    parallel_config = config.get('parallel', {})
    use_parallel = parallel_config.get('enabled', True)
    n_workers = parallel_config.get('workers', None)
    parallel_method = parallel_config.get('method', 'dask')
    chunk_size = parallel_config.get('chunk_size', None)

    # 3.2: Pre-tuning kinGEMs
    if run_pre_tuning:
        if pre_tuning_data_path and os.path.exists(pre_tuning_data_path):
            print("\n--- Running Pre-tuning kinGEMs (initial kcat values) ---")
            print(f"  Loading data from: {pre_tuning_data_path}")
            pre_tuning_df = pd.read_csv(pre_tuning_data_path)

            # Run simulation
            if use_parallel:
                print("  Using PARALLEL simulation")
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
                    method=parallel_method
                )
            else:
                print("  Using SEQUENTIAL simulation")
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
            simulation_results['pre_tuning'] = pre_tuning_GEM

            # Save results
            np.save(os.path.join(validation_dir, 'pre_tuning_GEM.npy'), pre_tuning_GEM)
        else:
            print("\n⚠️  Pre-tuning data path not found, skipping pre-tuning simulation")
            run_pre_tuning = False

    # 3.3: Post-tuning kinGEMs
    if run_post_tuning:
        if post_tuning_data_path and os.path.exists(post_tuning_data_path):
            print("\n--- Running Post-tuning kinGEMs (tuned kcat values) ---")
            print(f"  Loading data from: {post_tuning_data_path}")
            post_tuning_df = pd.read_csv(post_tuning_data_path)

            # Run simulation
            if use_parallel:
                print("  Using PARALLEL simulation")
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
                    method=parallel_method
                )
            else:
                print("  Using SEQUENTIAL simulation")
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
            simulation_results['post_tuning'] = post_tuning_GEM

            # Save results
            np.save(os.path.join(validation_dir, 'post_tuning_GEM.npy'), post_tuning_GEM)
        else:
            print("\n⚠️  Post-tuning data path not found, skipping post-tuning simulation")
            run_post_tuning = False

    # Save experimental data
    np.save(os.path.join(validation_dir, 'experimental_fitness.npy'), data_fitness_matched_adj)

    # === Step 4: Calculate Metrics ===
    print("\n=== Step 4: Calculating performance metrics ===")

    exp_binary = (data_fitness_matched_adj > 0).astype(float)

    # Store all metrics
    all_metrics = {}

    print("\n--- Basic Metrics ---")
    for model_type, sim_data in simulation_results.items():
        sim_binary = (sim_data > sim_thresh).astype(float)
        acc, prec, rec, f1 = print_basic_metrics(exp_binary, sim_binary, model_type.replace('_', ' ').title())
        all_metrics[model_type] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}

    print("\n--- Advanced Metrics ---")
    for model_type, sim_data in simulation_results.items():
        AC, AUC, BAC, ROC_AUC = calc_metrics(sim_data, data_fitness_matched_adj, fit_thresh, sim_thresh)
        print(f'{model_type.replace("_", " ").title()}:')
        print(f'  Accuracy: {AC[0]:.3f}')
        print(f'  AUC: {AUC[0]:.3f}')
        print(f'  Balanced Accuracy: {BAC[0]:.3f}')
        print(f'  ROC AUC: {ROC_AUC[0]:.3f}')
        all_metrics[model_type].update({
            'accuracy_adv': AC[0],
            'auc': AUC[0],
            'balanced_accuracy': BAC[0],
            'roc_auc': ROC_AUC[0]
        })

    # === Step 5: Essential Genes Analysis ===
    print("\n=== Step 5: Analyzing essential genes ===")

    essential_gene_mask = np.min(data_fitness_matched_adj, axis=1) <= 0.001
    fit_ess = data_fitness_matched_adj[essential_gene_mask, :]

    for model_type, sim_data in simulation_results.items():
        sim_ess = sim_data[essential_gene_mask, :]
        AC_ess, AUC_ess, BAC_ess, ROC_AUC_ess = calc_metrics(sim_ess, fit_ess, fit_thresh, sim_thresh)
        print(f'{model_type.replace("_", " ").title()} (Essential Genes):')
        print(f'  Accuracy: {AC_ess[0]:.3f}')
        print(f'  AUC: {AUC_ess[0]:.3f}')
        print(f'  Balanced Accuracy: {BAC_ess[0]:.3f}')
        print(f'  ROC AUC: {ROC_AUC_ess[0]:.3f}')
        all_metrics[model_type].update({
            'accuracy_essential': AC_ess[0],
            'auc_essential': AUC_ess[0],
            'balanced_accuracy_essential': BAC_ess[0],
            'roc_auc_essential': ROC_AUC_ess[0]
        })

    # === Step 6: Genes with kcat Analysis ===
    if run_pre_tuning or run_post_tuning:
        print("\n=== Step 6: Analyzing genes with kcat data ===")

        genes_with_kcat = set()
        for rxn in model.reactions:
            ann = rxn.annotation if hasattr(rxn, 'annotation') else {}
            if 'kcat' in ann and ann['kcat'] not in [None, '', 0, '0']:
                for gene in rxn.genes:
                    genes_with_kcat.add(gene.id)

        print(f"  Genes with kcat: {len(genes_with_kcat)}/{len(model.genes)} ({len(genes_with_kcat)/len(model.genes)*100:.1f}%)")

        kcat_gene_mask = np.array([g in genes_with_kcat for g in name_genes_matched_adj])
        fit_kcat = data_fitness_matched_adj[kcat_gene_mask, :]

        for model_type, sim_data in simulation_results.items():
            sim_kcat = sim_data[kcat_gene_mask, :]
            AC_kcat, AUC_kcat, BAC_kcat, ROC_AUC_kcat = calc_metrics(sim_kcat, fit_kcat, fit_thresh, sim_thresh)
            print(f'{model_type.replace("_", " ").title()} (Genes with kcat):')
            print(f'  Accuracy: {AC_kcat[0]:.3f}')
            print(f'  AUC: {AUC_kcat[0]:.3f}')
            print(f'  Balanced Accuracy: {BAC_kcat[0]:.3f}')
            print(f'  ROC AUC: {ROC_AUC_kcat[0]:.3f}')
            all_metrics[model_type].update({
                'accuracy_kcat': AC_kcat[0],
                'auc_kcat': AUC_kcat[0],
                'balanced_accuracy_kcat': BAC_kcat[0],
                'roc_auc_kcat': ROC_AUC_kcat[0]
            })

    # === Step 7: Generate Visualizations ===
    print("\n=== Step 7: Generating visualizations ===")

    # 7.1: Growth heatmaps
    n_models = len(simulation_results)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(6 * (n_models + 1), 5))

    if n_models == 1:
        axes = [axes]

    # Experimental
    im0 = axes[0].imshow(exp_binary, cmap='YlOrRd', aspect='auto')
    axes[0].set_title('Experimental Growth')
    axes[0].set_xlabel('Carbon Sources')
    axes[0].set_ylabel('Genes')
    plt.colorbar(im0, ax=axes[0], label='Growth')

    # Simulated models
    for idx, (model_type, sim_data) in enumerate(simulation_results.items(), start=1):
        sim_binary = (sim_data > sim_thresh).astype(float)
        im = axes[idx].imshow(sim_binary, cmap='YlOrRd', aspect='auto')
        axes[idx].set_title(f'{model_type.replace("_", " ").title()}')
        axes[idx].set_xlabel('Carbon Sources')
        if idx > 1:
            axes[idx].set_yticks([])
        plt.colorbar(im, ax=axes[idx], label='Growth')

    plt.tight_layout()
    plt.savefig(os.path.join(validation_dir, 'growth_heatmaps.png'), dpi=300, bbox_inches='tight')
    print("  Saved: growth_heatmaps.png")
    plt.close()

    # 7.2: Prediction distributions
    plt.figure(figsize=(12, 6))
    colors = ['green', 'blue', 'orange', 'red']

    sns.histplot(data_fitness_matched_adj.flatten(), bins=50, color=colors[0],
                 label='Experimental', kde=True, stat='density', alpha=0.4)

    for idx, (model_type, sim_data) in enumerate(simulation_results.items(), start=1):
        sns.histplot(sim_data.flatten(), bins=50, color=colors[idx % len(colors)],
                     label=model_type.replace('_', ' ').title(), kde=True, stat='density', alpha=0.5)

    plt.xlabel('Growth Value')
    plt.ylabel('Density')
    plt.title('Distribution of Growth Values: Experimental vs Simulated')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(validation_dir, 'growth_distributions.png'), dpi=300, bbox_inches='tight')
    print("  Saved: growth_distributions.png")
    plt.close()

    # 7.3: Metrics comparison bar chart
    if len(simulation_results) > 1:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(simulation_results.keys())

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(model_names)

        for idx, model_type in enumerate(model_names):
            values = [all_metrics[model_type][m] for m in metrics_to_plot]
            offset = (idx - len(model_names)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model_type.replace('_', ' ').title())

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(validation_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        print("  Saved: metrics_comparison.png")
        plt.close()

    # === Step 8: Save Results Summary ===
    print("\n=== Step 8: Saving results summary ===")

    # Create summary DataFrame
    summary_rows = []
    for model_type, metrics in all_metrics.items():
        row = {'Model': model_type.replace('_', ' ').title()}
        row.update(metrics)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(validation_dir, 'validation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print("  Saved: validation_summary.csv")

    # Save configuration used
    config_save_path = os.path.join(validation_dir, 'config_used.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("  Saved: config_used.json")

    # === Summary ===
    print("\n" + "="*70)
    print("=== Validation Complete ===")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Models compared: {', '.join([m.replace('_', ' ').title() for m in simulation_results.keys()])}")
    print(f"\nResults directory: {validation_dir}")
    print("\nGenerated files:")
    print("  - validation_summary.csv (metrics table)")
    print("  - growth_heatmaps.png (visual comparison)")
    print("  - growth_distributions.png (distribution plots)")
    if len(simulation_results) > 1:
        print("  - metrics_comparison.png (bar chart comparison)")
    print("  - *.npy files (raw simulation data)")
    print("="*70)


if __name__ == '__main__':
    main()

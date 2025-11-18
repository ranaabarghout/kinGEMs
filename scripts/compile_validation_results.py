#!/usr/bin/env python3
"""
Compile Parallel Validation Results
====================================

This script compiles and analyzes results from parallel validation jobs.
It loads baseline, pre-tuning, and post-tuning results and generates
comparative metrics and visualizations.

Usage:
    python scripts/compile_validation_results.py --input <dir> --output <dir>
"""

import argparse
from datetime import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import energy_distance, pearsonr, spearmanr, wasserstein_distance
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def load_results(input_dir, mode):
    """Load results for a specific mode."""
    filename = f"{mode}_GEM.npy"
    filepath = os.path.join(input_dir, filename)

    if not os.path.exists(filepath):
        print(f"  ⚠️  {mode} results not found: {filepath}")
        return None

    data = np.load(filepath)
    print(f"  ✓ Loaded {mode}: {data.shape}")
    return data


def load_metadata(input_dir, mode):
    """Load metadata for a specific mode."""
    filepath = os.path.join(input_dir, f"{mode}_metadata.json")

    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        metadata = json.load(f)
    return metadata


def load_wild_type_growth(input_dir, mode='baseline'):
    """Load wild-type growth rates for a specific model mode.

    Each model (baseline, pre-tuning, post-tuning) should have its own wild-type
    growth rates calculated WITH the appropriate constraints:
    - baseline: no enzyme constraints
    - pretuning: enzyme constraints with initial kcat values
    - posttuning: enzyme constraints with tuned kcat values

    Parameters
    ----------
    input_dir : str
        Directory containing validation results
    mode : str
        Model mode: 'baseline', 'pretuning', or 'posttuning'

    Returns
    -------
    numpy.ndarray or None
        Wild-type growth rates for each carbon source, or None if not found
    """
    # Try mode-specific wild-type file first
    wt_file = os.path.join(input_dir, f'{mode}_wildtype.npy')

    if os.path.exists(wt_file):
        wt_growth = np.load(wt_file)
        print(f"  ✓ Loaded {mode} wild-type growth: {wt_growth.shape}")
        return wt_growth
    else:
        print(f"  ⚠️  {mode} wild-type growth file not found: {wt_file}")
        print(f"     Fitness-based correlations for {mode} will be skipped.")
        print(f"     To enable: re-run validation with mode '{mode}' to calculate model-specific wild-type growth")
        return None


def calculate_correlations(experimental, predicted, fit_thresh=-2):
    """Calculate correlation metrics between experimental and predicted fitness.

    Parameters
    ----------
    experimental : numpy.ndarray
        Experimental fitness values
    predicted : numpy.ndarray
        Predicted growth rates
    fit_thresh : float
        Only include data points where experimental fitness > fit_thresh
    """
    from scipy.stats import kendalltau
    from sklearn.metrics import r2_score

    # Flatten arrays
    exp_flat = experimental.flatten()
    pred_flat = predicted.flatten()

    # Remove NaN and Inf values, and filter by fitness threshold
    mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
             np.isinf(exp_flat) | np.isinf(pred_flat)) & (exp_flat > fit_thresh)
    exp_clean = exp_flat[mask]
    pred_clean = pred_flat[mask]

    # Calculate correlations
    pearson_r, pearson_p = pearsonr(exp_clean, pred_clean)
    spearman_r, spearman_p = spearmanr(exp_clean, pred_clean)

    # Calculate Kendall's tau
    try:
        kendall_tau, kendall_p = kendalltau(exp_clean, pred_clean)
    except Exception:
        kendall_tau, kendall_p = np.nan, np.nan

    # Calculate R² (coefficient of determination)
    try:
        r2 = r2_score(exp_clean, pred_clean)
    except Exception:
        r2 = np.nan

    # Calculate RMSE
    rmse = np.sqrt(np.mean((exp_clean - pred_clean) ** 2))

    # Calculate MAE
    mae = np.mean(np.abs(exp_clean - pred_clean))

    # Calculate Energy Distance and Wasserstein Distance
    try:
        energy_dist = energy_distance(exp_clean, pred_clean)
    except Exception:
        energy_dist = np.nan

    try:
        wasserstein_dist = wasserstein_distance(exp_clean, pred_clean)
    except Exception:
        wasserstein_dist = np.nan

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'energy_distance': energy_dist,
        'wasserstein_distance': wasserstein_dist,
        'n_points': len(exp_clean),
        'n_total_before_filter': len(experimental.flatten()),
        'n_filtered_out': len(experimental.flatten()) - len(exp_clean),
        'fitness_threshold_used': fit_thresh
    }


def calculate_fitness_correlations(experimental, predicted, wild_type_growth=None, fit_thresh=-2):
    """Calculate correlation metrics after converting growth rates to fitness values.

    Fitness is calculated as log2(mutant_growth / wild_type_growth).
    This makes predicted values comparable to experimental fitness values which are
    log2 ratios describing change in abundance during experiments.

    Parameters
    ----------
    experimental : numpy.ndarray
        Experimental fitness values (genes × carbons), already log2 ratios
    predicted : numpy.ndarray
        Predicted growth rates (genes × carbons), to be converted to fitness
    wild_type_growth : numpy.ndarray or float, optional
        Wild-type growth rate for each carbon source. Should be shape (n_carbons,)
        or a scalar if same for all conditions. If None, must be provided externally
        or function will raise an error.
    fit_thresh : float
        Only include data points where experimental fitness > fit_thresh

    Returns
    -------
    dict
        Dictionary of correlation metrics plus fitness-converted values

    Notes
    -----
    The wild-type growth rate should be the model's predicted growth WITHOUT any
    gene deletions for each carbon source condition. This represents the reference
    growth rate that all mutant fitness values are compared against.
    """
    # Check if wild_type_growth is provided
    if wild_type_growth is None:
        raise ValueError(
            "wild_type_growth must be provided. It should be the model's predicted "
            "growth rate without any gene deletions for each carbon source. "
            "Calculate this by simulating the wild-type model (no knockouts) on each "
            "carbon source and passing the result as wild_type_growth parameter."
        )

    # Flatten arrays
    exp_flat = experimental.flatten()
    pred_flat = predicted.flatten()

    # Remove NaN and Inf values, and filter by fitness threshold
    mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
             np.isinf(exp_flat) | np.isinf(pred_flat)) & (exp_flat > fit_thresh)
    exp_clean = exp_flat[mask]
    pred_clean = pred_flat[mask]

    # Handle wild_type_growth: if it's per-condition, expand it to match flattened data
    if isinstance(wild_type_growth, np.ndarray):
        # If it's 1D (per carbon source), tile it for each gene
        if wild_type_growth.ndim == 1:
            n_genes = predicted.shape[0]
            wt_expanded = np.tile(wild_type_growth, n_genes)
            wt_flat = wt_expanded.flatten()
            wt_clean = wt_flat[mask]
        else:
            # Already 2D (genes × carbons)
            wt_flat = wild_type_growth.flatten()
            wt_clean = wt_flat[mask]
    else:
        # Scalar wild-type growth (same for all conditions)
        wt_clean = wild_type_growth

    # Convert predicted growth rates to fitness values
    # Fitness = log2(mutant_growth / wild_type_growth)
    # Handle zero or very small growth rates
    epsilon = 1e-10  # Small value to avoid log2(0)
    pred_clean_safe = np.maximum(pred_clean, epsilon)
    wt_clean_safe = np.maximum(wt_clean, epsilon) if isinstance(wt_clean, np.ndarray) else max(wt_clean, epsilon)
    pred_fitness = np.log2(pred_clean_safe / wt_clean_safe)

    # Calculate correlations with fitness-converted predictions
    from scipy.stats import kendalltau
    from sklearn.metrics import r2_score

    pearson_r, pearson_p = pearsonr(exp_clean, pred_fitness)
    spearman_r, spearman_p = spearmanr(exp_clean, pred_fitness)

    # Calculate Kendall's tau on fitness scale
    try:
        kendall_tau, kendall_p = kendalltau(exp_clean, pred_fitness)
    except Exception:
        kendall_tau, kendall_p = np.nan, np.nan

    # Calculate R² on fitness scale
    try:
        r2 = r2_score(exp_clean, pred_fitness)
    except Exception:
        r2 = np.nan

    # Calculate RMSE on fitness scale
    rmse = np.sqrt(np.mean((exp_clean - pred_fitness) ** 2))

    # Calculate MAE on fitness scale
    mae = np.mean(np.abs(exp_clean - pred_fitness))

    # Calculate Energy Distance and Wasserstein Distance on fitness scale
    try:
        energy_dist = energy_distance(exp_clean, pred_fitness)
    except Exception:
        energy_dist = np.nan

    try:
        wasserstein_dist = wasserstein_distance(exp_clean, pred_fitness)
    except Exception:
        wasserstein_dist = np.nan

    # Get representative wild-type growth value for reporting
    if isinstance(wt_clean, np.ndarray):
        wt_mean = np.mean(wt_clean)
        wt_std = np.std(wt_clean)
        wt_repr = f"mean={wt_mean:.6f}, std={wt_std:.6f}"
    else:
        wt_repr = float(wt_clean)

    return {
        'pearson_r_fitness': pearson_r,
        'pearson_p_fitness': pearson_p,
        'spearman_r_fitness': spearman_r,
        'spearman_p_fitness': spearman_p,
        'kendall_tau_fitness': kendall_tau,
        'kendall_p_fitness': kendall_p,
        'r2_fitness': r2,
        'rmse_fitness': rmse,
        'mae_fitness': mae,
        'energy_distance_fitness': energy_dist,
        'wasserstein_distance_fitness': wasserstein_dist,
        'wild_type_growth_used': wt_repr,
        'n_points': len(exp_clean),
        'n_total_before_filter': len(experimental.flatten()),
        'n_filtered_out': len(experimental.flatten()) - len(exp_clean),
        'fitness_threshold_used': fit_thresh,
        'pred_fitness_mean': np.mean(pred_fitness),
        'pred_fitness_std': np.std(pred_fitness),
        'pred_fitness_min': np.min(pred_fitness),
        'pred_fitness_max': np.max(pred_fitness)
    }


def calculate_classification_metrics(experimental, predicted, fit_thresh=-2, sim_thresh=0.001):
    """Calculate classification metrics for growth/no-growth prediction.

    Parameters
    ----------
    experimental : numpy.ndarray
        Experimental fitness values (genes × carbons)
    predicted : numpy.ndarray
        Predicted growth rates (genes × carbons)
    fit_thresh : float
        Threshold for experimental fitness (growth if > fit_thresh)
    sim_thresh : float
        Threshold for predicted growth (growth if > sim_thresh)

    Returns
    -------
    dict
        Dictionary of classification metrics
    """
    # Flatten arrays
    exp_flat = experimental.flatten()
    pred_flat = predicted.flatten()

    # Remove NaN and Inf values
    mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
             np.isinf(exp_flat) | np.isinf(pred_flat))
    exp_clean = exp_flat[mask]
    pred_clean = pred_flat[mask]

    # Convert to binary labels
    y_true = (exp_clean > fit_thresh).astype(int)
    y_score = pred_clean  # Continuous scores for ROC/PR curves
    y_pred = (pred_clean > sim_thresh).astype(int)  # Binary predictions

    # Basic classification metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Matthews Correlation Coefficient
    try:
        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        mcc = np.nan

    # ROC AUC and Average Precision (only if both classes present)
    try:
        roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        roc_auc = np.nan

    try:
        avg_prec = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        avg_prec = np.nan

    # Notebook-style PR-AUC (treats NO GROWTH as positive class)
    # This matches the validation notebook's "AUC_i" calculation
    # Uses binary predictions as y_true and negated continuous fitness as y_score
    try:
        from sklearn.metrics import auc as sk_auc
        # Notebook uses: precision_recall_curve(data_sim_b_V, data_fit_V*-1, pos_label=0)
        # where data_sim_b_V is binary predictions and data_fit_V is continuous experimental fitness
        prec_nb, rec_nb, _ = precision_recall_curve(y_pred, -exp_clean, pos_label=0)
        pr_auc_notebook = sk_auc(rec_nb, prec_nb) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        pr_auc_notebook = np.nan

    # Find optimal threshold (Youden's J statistic)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden_idx = np.argmax(tpr - fpr)
        optimal_thresh = thresholds[youden_idx]

        # Calculate metrics at optimal threshold
        y_pred_opt = (y_score >= optimal_thresh).astype(int)
        acc_opt = accuracy_score(y_true, y_pred_opt)
        bal_acc_opt = balanced_accuracy_score(y_true, y_pred_opt)
    except Exception:
        optimal_thresh = np.nan
        acc_opt = np.nan
        bal_acc_opt = np.nan

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'average_precision': avg_prec,
        'pr_auc_notebook': pr_auc_notebook,  # Notebook-style PR-AUC (no-growth as positive)
        'optimal_threshold': optimal_thresh,
        'accuracy_optimal': acc_opt,
        'balanced_accuracy_optimal': bal_acc_opt,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'n_samples': len(y_true)
    }


def create_comparison_plot(experimental, baseline, pretuning, posttuning, output_dir):
    """Create comparison scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    exp_flat = experimental.flatten()

    models = [
        ('Baseline GEM', baseline, axes[0]),
        ('Pre-tuning kinGEMs', pretuning, axes[1]),
        ('Post-tuning kinGEMs', posttuning, axes[2])
    ]

    for model_name, predicted, ax in models:
        if predicted is None:
            ax.text(0.5, 0.5, 'Data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name)
            continue

        pred_flat = predicted.flatten()

        # Remove NaN/Inf
        mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
                np.isinf(exp_flat) | np.isinf(pred_flat))
        exp_clean = exp_flat[mask]
        pred_clean = pred_flat[mask]

        # Calculate correlation
        pearson_r, _ = pearsonr(exp_clean, pred_clean)

        # Create scatter plot
        ax.scatter(exp_clean, pred_clean, alpha=0.3, s=10)
        ax.plot([exp_clean.min(), exp_clean.max()],
               [exp_clean.min(), exp_clean.max()],
               'r--', lw=2, label='Perfect prediction')

        ax.set_xlabel('Experimental Fitness', fontsize=12)
        ax.set_ylabel('Predicted Growth Rate', fontsize=12)
        ax.set_title(f'{model_name}\nPearson r = {pearson_r:.3f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'validation_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comparison plot: {plot_file}")
    plt.close()


def create_fitness_comparison_plots(experimental, baseline, pretuning, posttuning,
                                   baseline_wt, pretuning_wt, posttuning_wt, output_dir, fit_thresh=-2):
    """Create scatter plots comparing predicted fitness vs experimental fitness.

    This converts predicted growth rates to fitness values using:
    fitness = log2(mutant_growth / wildtype_growth)

    Then creates scatter plots showing predicted fitness vs experimental fitness
    for each model. This is the proper comparison since experimental data are
    already fitness values (log2 ratios).

    Parameters
    ----------
    experimental : numpy.ndarray
        Experimental fitness values (genes × carbons), already log2 ratios
    baseline : numpy.ndarray or None
        Baseline GEM predictions (growth rates)
    pretuning : numpy.ndarray or None
        Pre-tuning kinGEMs predictions (growth rates)
    posttuning : numpy.ndarray or None
        Post-tuning kinGEMs predictions (growth rates)
    baseline_wt : numpy.ndarray or None
        Baseline wild-type growth rates
    pretuning_wt : numpy.ndarray or None
        Pre-tuning wild-type growth rates
    posttuning_wt : numpy.ndarray or None
        Post-tuning wild-type growth rates
    output_dir : str
        Directory to save plots
    fit_thresh : float
        Only include data points where experimental fitness > fit_thresh
    """
    # Collect available models
    models_data = []
    model_names = []
    model_wt = []

    if baseline is not None and baseline_wt is not None:
        models_data.append(baseline)
        model_names.append('Baseline GEM')
        model_wt.append(baseline_wt)

    if pretuning is not None and pretuning_wt is not None:
        models_data.append(pretuning)
        model_names.append('Pre-tuning kinGEMs')
        model_wt.append(pretuning_wt)

    if posttuning is not None and posttuning_wt is not None:
        models_data.append(posttuning)
        model_names.append('Post-tuning kinGEMs')
        model_wt.append(posttuning_wt)

    if not models_data:
        print("  ⚠️  No model data with wild-type growth available for fitness comparison plots")
        return

    # Create figure with subplots
    n_models = len(models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    exp_flat = experimental.flatten()

    for idx, (predicted, wt_growth, name, ax) in enumerate(zip(models_data, model_wt, model_names, axes)):
        pred_flat = predicted.flatten()

        # Expand wild-type growth to match flattened data
        if wt_growth.ndim == 1:
            n_genes = predicted.shape[0]
            wt_expanded = np.tile(wt_growth, n_genes)
        else:
            wt_expanded = wt_growth.flatten()

        # Clean data
        mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
                np.isinf(exp_flat) | np.isinf(pred_flat) |
                np.isnan(wt_expanded) | np.isinf(wt_expanded)) & (exp_flat > fit_thresh)
        exp_clean = exp_flat[mask]
        pred_clean = pred_flat[mask]
        wt_clean = wt_expanded[mask]

        # Convert predicted growth to fitness
        epsilon = 1e-10
        pred_clean_safe = np.maximum(pred_clean, epsilon)
        wt_clean_safe = np.maximum(wt_clean, epsilon)
        pred_fitness = np.log2(pred_clean_safe / wt_clean_safe)

        # Calculate correlation on fitness scale
        from scipy.stats import kendalltau
        from sklearn.metrics import r2_score

        pearson_r, pearson_p = pearsonr(exp_clean, pred_fitness)
        spearman_r, _ = spearmanr(exp_clean, pred_fitness)

        # Calculate Kendall's tau
        try:
            kendall_tau, _ = kendalltau(exp_clean, pred_fitness)
        except Exception:
            kendall_tau = np.nan

        # Calculate R²
        try:
            r2 = r2_score(exp_clean, pred_fitness)
        except Exception:
            r2 = np.nan

        # Calculate RMSE
        rmse = np.sqrt(np.mean((exp_clean - pred_fitness) ** 2))

        # Create scatter plot
        ax.scatter(exp_clean, pred_fitness, alpha=0.3, s=10, color='steelblue', edgecolors='none')

        # Add diagonal line (perfect prediction)
        min_val = min(exp_clean.min(), pred_fitness.min())
        max_val = max(exp_clean.max(), pred_fitness.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', lw=2, label='Perfect prediction', zorder=10)

        # Add horizontal and vertical lines at zero
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Labels and title
        ax.set_xlabel('Experimental Fitness (log₂ ratio)', fontsize=12)
        ax.set_ylabel('Predicted Fitness (log₂ ratio)', fontsize=12)
        ax.set_title(f'{name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}\n(fitness > {fit_thresh})', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics box
        stats_text = (f'Pearson r = {pearson_r:.4f}\n'
                     f'Spearman ρ = {spearman_r:.4f}\n'
                     f'Kendall τ = {kendall_tau:.4f}\n'
                     f'R² = {r2:.4f}\n'
                     f'RMSE = {rmse:.4f}\n'
                     f'p < {pearson_p:.2e}\n'
                     f'n = {len(exp_clean):,}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Equal aspect ratio for fair comparison
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'fitness_comparison_scatter.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved fitness comparison scatter plots: {plot_file}")
    plt.close()

    # Also create a single combined plot if multiple models
    if n_models > 1:
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

        for idx, (predicted, wt_growth, name, color) in enumerate(zip(models_data, model_wt, model_names, colors)):
            pred_flat = predicted.flatten()

            # Expand wild-type growth
            if wt_growth.ndim == 1:
                n_genes = predicted.shape[0]
                wt_expanded = np.tile(wt_growth, n_genes)
            else:
                wt_expanded = wt_growth.flatten()

            # Clean data
            mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
                    np.isinf(exp_flat) | np.isinf(pred_flat) |
                    np.isnan(wt_expanded) | np.isinf(wt_expanded)) & (exp_flat > fit_thresh)
            exp_clean = exp_flat[mask]
            pred_clean = pred_flat[mask]
            wt_clean = wt_expanded[mask]

            # Convert to fitness
            epsilon = 1e-10
            pred_clean_safe = np.maximum(pred_clean, epsilon)
            wt_clean_safe = np.maximum(wt_clean, epsilon)
            pred_fitness = np.log2(pred_clean_safe / wt_clean_safe)

            # Calculate correlation
            pearson_r, _ = pearsonr(exp_clean, pred_fitness)

            # Plot with transparency
            ax.scatter(exp_clean, pred_fitness, alpha=0.2, s=8,
                      color=color, label=f'{name} (r={pearson_r:.3f})', edgecolors='none')

        # Add diagonal line
        min_val = min(exp_flat[~(np.isnan(exp_flat) | np.isinf(exp_flat))].min(), -10)
        max_val = max(exp_flat[~(np.isnan(exp_flat) | np.isinf(exp_flat))].max(), 2)
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', lw=2, label='Perfect prediction', zorder=10)

        # Add zero lines
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_xlabel('Experimental Fitness (log₂ ratio)', fontsize=13)
        ax.set_ylabel('Predicted Fitness (log₂ ratio)', fontsize=13)
        ax.set_title(f'Fitness Comparison Across All Models\n(fitness > {fit_thresh})', fontsize=14)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'fitness_comparison_combined.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved combined fitness comparison plot: {plot_file}")
        plt.close()


def create_distribution_plots(experimental, baseline, pretuning, posttuning, output_dir):
    """Create distribution plots comparing predicted growth rates across models.

    Generates:
    1. Histograms of predicted growth rates for each model
    2. Overlaid density plots comparing all three models
    3. Box plots showing distribution statistics

    Parameters
    ----------
    experimental : numpy.ndarray
        Experimental fitness values (for reference statistics)
    baseline : numpy.ndarray or None
        Baseline GEM predictions
    pretuning : numpy.ndarray or None
        Pre-tuning kinGEMs predictions
    posttuning : numpy.ndarray or None
        Post-tuning kinGEMs predictions
    output_dir : str
        Directory to save plots
    """
    # Collect available models
    models_data = []
    model_names = []
    model_colors = []

    if baseline is not None:
        models_data.append(baseline.flatten())
        model_names.append('Baseline GEM')
        model_colors.append('#1f77b4')  # Blue

    if pretuning is not None:
        models_data.append(pretuning.flatten())
        model_names.append('Pre-tuning kinGEMs')
        model_colors.append('#ff7f0e')  # Orange

    if posttuning is not None:
        models_data.append(posttuning.flatten())
        model_names.append('Post-tuning kinGEMs')
        model_colors.append('#2ca02c')  # Green

    if not models_data:
        print("  ⚠️  No model data available for distribution plots")
        return

    # Clean data (remove NaN/Inf)
    models_clean = []
    for data in models_data:
        mask = ~(np.isnan(data) | np.isinf(data))
        models_clean.append(data[mask])

    # === 1. Individual Histograms ===
    n_models = len(models_clean)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (data, name, color, ax) in enumerate(zip(models_clean, model_names, model_colors, axes)):
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(data):.4f}')
        ax.axvline(np.median(data), color='darkred', linestyle=':', linewidth=2, label=f'Median = {np.median(data):.4f}')
        ax.set_xlabel('Predicted Growth Rate', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{name}\nDistribution of Predictions', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text box
        stats_text = (f'n = {len(data):,}\n'
                     f'μ = {np.mean(data):.4f}\n'
                     f'σ = {np.std(data):.4f}\n'
                     f'min = {np.min(data):.4f}\n'
                     f'max = {np.max(data):.4f}')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'prediction_distributions_histograms.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved distribution histograms: {plot_file}")
    plt.close()

    # === 2. Overlaid Density Plots ===
    fig, ax = plt.subplots(figsize=(10, 6))

    for data, name, color in zip(models_clean, model_names, model_colors):
        # Kernel Density Estimation
        from scipy.stats import gaussian_kde
        density = gaussian_kde(data)
        xs = np.linspace(min(data), max(data), 200)
        ax.plot(xs, density(xs), label=name, color=color, linewidth=2)
        ax.fill_between(xs, density(xs), alpha=0.2, color=color)

    ax.set_xlabel('Predicted Growth Rate', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Distribution of Predicted Growth Rates\nComparison Across Models', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'prediction_distributions_density.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved density plot: {plot_file}")
    plt.close()

    # === 3. Box Plots ===
    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(models_clean, labels=model_names, patch_artist=True,
                    showmeans=True, meanline=True,
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

    # Color the boxes
    for patch, color in zip(bp['boxes'], model_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Predicted Growth Rate', fontsize=13)
    ax.set_title('Distribution Statistics\nPredicted Growth Rates Across Models', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend for mean and median
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='best')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'prediction_distributions_boxplot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved box plot: {plot_file}")
    plt.close()

    # === 4. Cumulative Distribution Functions ===
    fig, ax = plt.subplots(figsize=(10, 6))

    for data, name, color in zip(models_clean, model_names, model_colors):
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cumulative, label=name, color=color, linewidth=2)

    ax.set_xlabel('Predicted Growth Rate', fontsize=13)
    ax.set_ylabel('Cumulative Probability', fontsize=13)
    ax.set_title('Cumulative Distribution Function\nPredicted Growth Rates', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'prediction_distributions_cdf.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved CDF plot: {plot_file}")
    plt.close()

    # === 5. Violin Plots (alternative to box plots) ===
    fig, ax = plt.subplots(figsize=(10, 6))

    parts = ax.violinplot(models_clean, showmeans=True, showmedians=True)

    # Color the violin plots
    for idx, (pc, color) in enumerate(zip(parts['bodies'], model_colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks(np.arange(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel('Predicted Growth Rate', fontsize=13)
    ax.set_title('Distribution Shape\nPredicted Growth Rates (Violin Plot)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'prediction_distributions_violin.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved violin plot: {plot_file}")
    plt.close()

    # === 6. Summary Statistics Table ===
    stats_dict = {}
    for data, name in zip(models_clean, model_names):
        stats_dict[name] = {
            'count': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            '25%': np.percentile(data, 25),
            'median': np.median(data),
            '75%': np.percentile(data, 75),
            'max': np.max(data),
            'zeros': np.sum(data == 0),
            'zeros_pct': (np.sum(data == 0) / len(data)) * 100
        }

    stats_df = pd.DataFrame(stats_dict).T
    stats_df = stats_df.round({
        'mean': 6, 'std': 6, 'min': 6,
        '25%': 6, 'median': 6, '75%': 6, 'max': 6,
        'zeros_pct': 2
    })

    stats_file = os.path.join(output_dir, 'prediction_distribution_statistics.csv')
    stats_df.to_csv(stats_file)
    print(f"  ✓ Saved distribution statistics: {stats_file}")

    print("\n" + "="*70)
    print("=== Prediction Distribution Statistics ===")
    print("="*70)
    print(stats_df.to_string())
    print("="*70)


def create_fitness_distribution_plots(experimental, baseline, pretuning, posttuning,
                                    baseline_wt, pretuning_wt, posttuning_wt, output_dir, fit_thresh=-2):
    """Create distribution plots comparing fitness values across models and experimental data.

    This function converts predicted growth rates to fitness values (log2 ratios) and compares
    their distributions with experimental fitness values. This provides insight into how well
    the models capture the range and distribution of experimental fitness effects.

    Parameters
    ----------
    experimental : numpy.ndarray
        Experimental fitness values (genes × carbons), already log2 ratios
    baseline : numpy.ndarray or None
        Baseline GEM predictions (growth rates)
    pretuning : numpy.ndarray or None
        Pre-tuning kinGEMs predictions (growth rates)
    posttuning : numpy.ndarray or None
        Post-tuning kinGEMs predictions (growth rates)
    baseline_wt : numpy.ndarray or None
        Baseline wild-type growth rates
    pretuning_wt : numpy.ndarray or None
        Pre-tuning wild-type growth rates
    posttuning_wt : numpy.ndarray or None
        Post-tuning wild-type growth rates
    output_dir : str
        Directory to save plots
    fit_thresh : float
        Only include data points where experimental fitness > fit_thresh
    """
    # Collect available models and convert to fitness
    fitness_data = []
    fitness_names = []
    fitness_colors = []

    # Always add experimental fitness first
    exp_flat = experimental.flatten()
    exp_mask = ~(np.isnan(exp_flat) | np.isinf(exp_flat)) & (exp_flat > fit_thresh)
    exp_clean = exp_flat[exp_mask]

    fitness_data.append(exp_clean)
    fitness_names.append('Experimental')
    fitness_colors.append('#333333')  # Dark gray

    # Helper function to convert growth rates to fitness
    def convert_to_fitness(predicted, wt_growth, name, color):
        if predicted is None or wt_growth is None:
            return None

        pred_flat = predicted.flatten()

        # Expand wild-type growth to match flattened data
        if wt_growth.ndim == 1:
            n_genes = predicted.shape[0]
            wt_expanded = np.tile(wt_growth, n_genes)
        else:
            wt_expanded = wt_growth.flatten()

        # Clean data with fitness threshold
        mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
                np.isinf(exp_flat) | np.isinf(pred_flat) |
                np.isnan(wt_expanded) | np.isinf(wt_expanded)) & (exp_flat > fit_thresh)
        pred_clean = pred_flat[mask]
        wt_clean = wt_expanded[mask]

        # Convert to fitness
        epsilon = 1e-10
        pred_clean_safe = np.maximum(pred_clean, epsilon)
        wt_clean_safe = np.maximum(wt_clean, epsilon)
        pred_fitness = np.log2(pred_clean_safe / wt_clean_safe)

        fitness_data.append(pred_fitness)
        fitness_names.append(name)
        fitness_colors.append(color)

        return pred_fitness

    # Convert model predictions to fitness
    convert_to_fitness(baseline, baseline_wt, 'Baseline GEM', '#1f77b4')  # Blue
    convert_to_fitness(pretuning, pretuning_wt, 'Pre-tuning kinGEMs', '#ff7f0e')  # Orange
    convert_to_fitness(posttuning, posttuning_wt, 'Post-tuning kinGEMs', '#2ca02c')  # Green

    if len(fitness_data) <= 1:
        print("  ⚠️  No model fitness data available for fitness distribution plots")
        return

    # === 1. Individual Histograms ===
    n_datasets = len(fitness_data)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for idx, (data, name, color, ax) in enumerate(zip(fitness_data, fitness_names, fitness_colors, axes)):
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(data):.3f}')
        ax.axvline(np.median(data), color='darkred', linestyle=':', linewidth=2, label=f'Median = {np.median(data):.3f}')
        ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.8, label='No effect')
        ax.set_xlabel('Fitness (log₂ ratio)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{name}\nFitness Distribution (fitness > {fit_thresh})', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text box
        stats_text = (f'n = {len(data):,}\n'
                     f'μ = {np.mean(data):.3f}\n'
                     f'σ = {np.std(data):.3f}\n'
                     f'min = {np.min(data):.3f}\n'
                     f'max = {np.max(data):.3f}\n'
                     f'% < 0 = {(np.sum(data < 0) / len(data) * 100):.1f}%\n'
                     f'% > 0 = {(np.sum(data > 0) / len(data) * 100):.1f}%')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'fitness_distributions_histograms.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved fitness distribution histograms: {plot_file}")
    plt.close()

    # === 2. Overlaid Density Plots ===
    fig, ax = plt.subplots(figsize=(12, 8))

    for data, name, color in zip(fitness_data, fitness_names, fitness_colors):
        # Kernel Density Estimation
        from scipy.stats import gaussian_kde
        density = gaussian_kde(data)
        xs = np.linspace(min(data.min() for data in fitness_data),
                        max(data.max() for data in fitness_data), 400)
        ax.plot(xs, density(xs), label=f'{name} (μ={np.mean(data):.3f})', color=color, linewidth=2)
        ax.fill_between(xs, density(xs), alpha=0.2, color=color)

    # Add vertical lines for reference
    ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='No effect (fitness=0)')
    ax.axvline(-1, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='50% reduction')
    ax.axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='2× improvement')

    ax.set_xlabel('Fitness (log₂ ratio)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'Fitness Value Distributions\nExperimental vs Predicted (fitness > {fit_thresh})', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'fitness_distributions_density.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved fitness density plot: {plot_file}")
    plt.close()

    # === 3. Box Plots ===
    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(fitness_data, labels=fitness_names, patch_artist=True,
                    showmeans=True, meanline=True,
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=2, alpha=0.3))

    # Color the boxes
    for patch, color in zip(bp['boxes'], fitness_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add horizontal reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    ax.axhline(-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylabel('Fitness (log₂ ratio)', fontsize=13)
    ax.set_title(f'Fitness Distribution Statistics\nExperimental vs Predicted (fitness > {fit_thresh})', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend for mean and median
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean'),
        Line2D([0], [0], color='black', linewidth=1, label='No effect'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='±50% effect')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='best')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'fitness_distributions_boxplot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved fitness box plot: {plot_file}")
    plt.close()

    # === 4. Violin Plots ===
    fig, ax = plt.subplots(figsize=(12, 6))

    parts = ax.violinplot(fitness_data, showmeans=True, showmedians=True)

    # Color the violin plots
    for idx, (pc, color) in enumerate(zip(parts['bodies'], fitness_colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks(np.arange(1, len(fitness_names) + 1))
    ax.set_xticklabels(fitness_names, fontsize=11)
    ax.set_ylabel('Fitness (log₂ ratio)', fontsize=13)
    ax.set_title(f'Fitness Distribution Shape\nExperimental vs Predicted (fitness > {fit_thresh})', fontsize=14)

    # Add horizontal reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    ax.axhline(-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'fitness_distributions_violin.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved fitness violin plot: {plot_file}")
    plt.close()

    # === 5. Cumulative Distribution Functions ===
    fig, ax = plt.subplots(figsize=(12, 8))

    for data, name, color in zip(fitness_data, fitness_names, fitness_colors):
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cumulative, label=f'{name}', color=color, linewidth=2)

    # Add vertical reference lines
    ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='No effect')
    ax.axvline(-1, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel('Fitness (log₂ ratio)', fontsize=13)
    ax.set_ylabel('Cumulative Probability', fontsize=13)
    ax.set_title(f'Cumulative Distribution Function\nFitness Values (fitness > {fit_thresh})', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'fitness_distributions_cdf.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved fitness CDF plot: {plot_file}")
    plt.close()

    # === 6. Summary Statistics Table ===
    stats_dict = {}
    for data, name in zip(fitness_data, fitness_names):
        stats_dict[name] = {
            'count': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            '25%': np.percentile(data, 25),
            'median': np.median(data),
            '75%': np.percentile(data, 75),
            'max': np.max(data),
            'negative_pct': (np.sum(data < 0) / len(data)) * 100,
            'positive_pct': (np.sum(data > 0) / len(data)) * 100,
            'neutral_pct': (np.sum(np.abs(data) < 0.1) / len(data)) * 100,  # Within ±10% (≈±0.1 log2 units)
            'strong_negative_pct': (np.sum(data < -1) / len(data)) * 100,  # >50% reduction
            'strong_positive_pct': (np.sum(data > 1) / len(data)) * 100   # >2× improvement
        }

    stats_df = pd.DataFrame(stats_dict).T
    stats_df = stats_df.round({
        'mean': 3, 'std': 3, 'min': 3,
        '25%': 3, 'median': 3, '75%': 3, 'max': 3,
        'negative_pct': 1, 'positive_pct': 1, 'neutral_pct': 1,
        'strong_negative_pct': 1, 'strong_positive_pct': 1
    })

    stats_file = os.path.join(output_dir, 'fitness_distribution_statistics.csv')
    stats_df.to_csv(stats_file)
    print(f"  ✓ Saved fitness distribution statistics: {stats_file}")

    print("\n" + "="*70)
    print("=== Fitness Distribution Statistics ===")
    print("="*70)
    print(stats_df.to_string())
    print("="*70)
    print("Notes:")
    print("  - negative_pct: % of mutations with fitness < 0 (deleterious)")
    print("  - positive_pct: % of mutations with fitness > 0 (beneficial)")
    print("  - neutral_pct: % of mutations with |fitness| < 0.1 (near-neutral)")
    print("  - strong_negative_pct: % with fitness < -1 (>50% reduction)")
    print("  - strong_positive_pct: % with fitness > 1 (>2× improvement)")
    print("="*70)


def create_classification_plots(experimental, predicted, model_name, output_dir,
                                fit_thresh=-2, sim_thresh=0.001):
    """Create classification plots (ROC, PR, confusion matrix) for a single model.

    Parameters
    ----------
    experimental : numpy.ndarray
        Experimental fitness values
    predicted : numpy.ndarray
        Predicted growth rates
    model_name : str
        Name of the model (for plot titles)
    output_dir : str
        Directory to save plots
    fit_thresh : float
        Threshold for experimental fitness
    sim_thresh : float
        Threshold for predicted growth
    """
    # Flatten and clean
    exp_flat = experimental.flatten()
    pred_flat = predicted.flatten()
    mask = ~(np.isnan(exp_flat) | np.isnan(pred_flat) |
             np.isinf(exp_flat) | np.isinf(pred_flat))
    exp_clean = exp_flat[mask]
    pred_clean = pred_flat[mask]

    # Binary labels
    y_true = (exp_clean > fit_thresh).astype(int)
    y_score = pred_clean
    y_pred = (pred_clean > sim_thresh).astype(int)

    # Skip if not enough data or only one class
    if len(np.unique(y_true)) < 2:
        print(f"  ⚠️  Skipping classification plots for {model_name} (only one class present)")
        return

    # Clean model name for filenames
    safe_name = model_name.replace(' ', '_').replace('-', '_').lower()

    # 1. ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{model_name}\nROC Curve', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = os.path.join(output_dir, f'{safe_name}_roc_curve.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved ROC curve: {plot_file}")
    except Exception as e:
        print(f"  ⚠️  Failed to create ROC curve for {model_name}: {e}")

    # 2. Precision-Recall Curve
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
        avg_prec = average_precision_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall_vals, precision_vals, label=f'AP = {avg_prec:.3f}', linewidth=2)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{model_name}\nPrecision-Recall Curve', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = os.path.join(output_dir, f'{safe_name}_pr_curve.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved PR curve: {plot_file}")
    except Exception as e:
        print(f"  ⚠️  Failed to create PR curve for {model_name}: {e}")

    # 3. Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Growth', 'Growth'])
        ax.set_yticklabels(['No Growth', 'Growth'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{model_name}\nConfusion Matrix (threshold={sim_thresh})', fontsize=13)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j],
                       ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black',
                       fontsize=14, weight='bold')

        plt.colorbar(im, ax=ax, label='Count')
        plt.tight_layout()

        plot_file = os.path.join(output_dir, f'{safe_name}_confusion_matrix.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved confusion matrix: {plot_file}")
    except Exception as e:
        print(f"  ⚠️  Failed to create confusion matrix for {model_name}: {e}")


def create_metrics_table(metrics_dict, output_dir):
    """Create a table of metrics."""
    df = pd.DataFrame(metrics_dict).T

    # Round values
    df = df.round({
        'pearson_r': 4,
        'pearson_p': 6,
        'spearman_r': 4,
        'spearman_p': 6,
        'kendall_tau': 4,
        'kendall_p': 6,
        'r2': 4,
        'rmse': 4,
        'mae': 4,
        'energy_distance': 4,
        'wasserstein_distance': 4
    })

    # Save to CSV
    csv_file = os.path.join(output_dir, 'validation_metrics.csv')
    df.to_csv(csv_file)
    print(f"  ✓ Saved metrics table: {csv_file}")

    # Print to console
    print("\n" + "="*70)
    print("=== Validation Metrics ===")
    print("="*70)
    print(df.to_string())
    print("="*70)

    return df


def create_improvement_analysis(baseline_metrics, pretuning_metrics, posttuning_metrics, output_dir):
    """Analyze improvement from baseline to kinGEMs."""
    improvements = {}

    if baseline_metrics and pretuning_metrics:
        improvements['Baseline → Pre-tuning'] = {
            'Δ Pearson r': pretuning_metrics['pearson_r'] - baseline_metrics['pearson_r'],
            'Δ Spearman r': pretuning_metrics['spearman_r'] - baseline_metrics['spearman_r'],
            'Δ RMSE': pretuning_metrics['rmse'] - baseline_metrics['rmse'],
            '% RMSE change': ((pretuning_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse'] * 100)
        }

    if baseline_metrics and posttuning_metrics:
        improvements['Baseline → Post-tuning'] = {
            'Δ Pearson r': posttuning_metrics['pearson_r'] - baseline_metrics['pearson_r'],
            'Δ Spearman r': posttuning_metrics['spearman_r'] - baseline_metrics['spearman_r'],
            'Δ RMSE': posttuning_metrics['rmse'] - baseline_metrics['rmse'],
            '% RMSE change': ((posttuning_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse'] * 100)
        }

    if pretuning_metrics and posttuning_metrics:
        improvements['Pre-tuning → Post-tuning'] = {
            'Δ Pearson r': posttuning_metrics['pearson_r'] - pretuning_metrics['pearson_r'],
            'Δ Spearman r': posttuning_metrics['spearman_r'] - pretuning_metrics['spearman_r'],
            'Δ RMSE': posttuning_metrics['rmse'] - pretuning_metrics['rmse'],
            '% RMSE change': ((posttuning_metrics['rmse'] - pretuning_metrics['rmse']) / pretuning_metrics['rmse'] * 100)
        }

    if improvements:
        df = pd.DataFrame(improvements).T.round(4)

        csv_file = os.path.join(output_dir, 'validation_improvements.csv')
        df.to_csv(csv_file)
        print(f"  ✓ Saved improvements analysis: {csv_file}")

        print("\n" + "="*70)
        print("=== Improvement Analysis ===")
        print("="*70)
        print(df.to_string())
        print("="*70)
        print("Note: Negative Δ RMSE indicates improvement (lower error)")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default='results/validation_parallel',
                       help='Input directory containing parallel validation results')
    parser.add_argument('--output', default='results/validation_compiled',
                       help='Output directory for compiled results')
    parser.add_argument('--fit-thresh', type=float, default=-2,
                       help='Threshold for experimental fitness (growth if > fit_thresh). Default: -2')
    parser.add_argument('--sim-thresh', type=float, default=0.001,
                       help='Threshold for simulated growth (growth if > sim_thresh). Default: 0.001')
    parser.add_argument('--config', type=str, default=None,
                       help='Optional: JSON config file to load thresholds from')

    args = parser.parse_args()

    # Load thresholds from config if provided
    if args.config and os.path.exists(args.config):
        print(f"Loading thresholds from config: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
            args.fit_thresh = config.get('fit_thresh', args.fit_thresh)
            args.sim_thresh = config.get('sim_thresh', args.sim_thresh)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "="*70)
    print("=== Compiling Validation Results ===")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Fitness threshold: {args.fit_thresh}")
    print(f"Simulation threshold: {args.sim_thresh}")
    print("="*70)

    # === Step 1: Load all results ===
    print("\n=== Step 1: Loading results ===")

    experimental = np.load(os.path.join(args.input, 'experimental_fitness.npy'))
    print(f"  ✓ Loaded experimental: {experimental.shape}")

    baseline_GEM = load_results(args.input, 'baseline')
    pretuning_GEM = load_results(args.input, 'pre_tuning')
    posttuning_GEM = load_results(args.input, 'post_tuning')

    # Load model-specific wild-type growth (needed for fitness-based correlations)
    # Each model has its own wild-type growth WITH appropriate constraints
    baseline_wildtype = load_wild_type_growth(args.input, 'baseline')
    pretuning_wildtype = load_wild_type_growth(args.input, 'pretuning')
    posttuning_wildtype = load_wild_type_growth(args.input, 'posttuning')

    # Load metadata
    baseline_meta = load_metadata(args.input, 'baseline')
    pretuning_meta = load_metadata(args.input, 'pretuning')
    posttuning_meta = load_metadata(args.input, 'posttuning')

    # === Step 2: Calculate correlation metrics ===
    print("\n=== Step 2: Calculating correlation metrics ===")

    correlation_metrics = {}
    fitness_correlation_metrics = {}

    if baseline_GEM is not None:
        correlation_metrics['Baseline GEM'] = calculate_correlations(experimental, baseline_GEM, args.fit_thresh)
        if baseline_wildtype is not None:
            fitness_correlation_metrics['Baseline GEM'] = calculate_fitness_correlations(
                experimental, baseline_GEM, baseline_wildtype, args.fit_thresh)
        print("  ✓ Baseline correlation metrics calculated")

    if pretuning_GEM is not None:
        correlation_metrics['Pre-tuning kinGEMs'] = calculate_correlations(experimental, pretuning_GEM, args.fit_thresh)
        if pretuning_wildtype is not None:
            fitness_correlation_metrics['Pre-tuning kinGEMs'] = calculate_fitness_correlations(
                experimental, pretuning_GEM, pretuning_wildtype, args.fit_thresh)
        print("  ✓ Pre-tuning correlation metrics calculated")

    if posttuning_GEM is not None:
        correlation_metrics['Post-tuning kinGEMs'] = calculate_correlations(experimental, posttuning_GEM, args.fit_thresh)
        if posttuning_wildtype is not None:
            fitness_correlation_metrics['Post-tuning kinGEMs'] = calculate_fitness_correlations(
                experimental, posttuning_GEM, posttuning_wildtype, args.fit_thresh)
        print("  ✓ Post-tuning correlation metrics calculated")

    # === Step 3: Calculate classification metrics ===
    print("\n=== Step 3: Calculating classification metrics ===")

    classification_metrics = {}

    if baseline_GEM is not None:
        classification_metrics['Baseline GEM'] = calculate_classification_metrics(
            experimental, baseline_GEM, args.fit_thresh, args.sim_thresh)
        print("  ✓ Baseline classification metrics calculated")

    if pretuning_GEM is not None:
        classification_metrics['Pre-tuning kinGEMs'] = calculate_classification_metrics(
            experimental, pretuning_GEM, args.fit_thresh, args.sim_thresh)
        print("  ✓ Pre-tuning classification metrics calculated")

    if posttuning_GEM is not None:
        classification_metrics['Post-tuning kinGEMs'] = calculate_classification_metrics(
            experimental, posttuning_GEM, args.fit_thresh, args.sim_thresh)
        print("  ✓ Post-tuning classification metrics calculated")

    # === Step 4: Create correlation visualizations ===
    print("\n=== Step 4: Creating correlation visualizations ===")

    create_comparison_plot(experimental, baseline_GEM, pretuning_GEM, posttuning_GEM, args.output)

    # === Step 4a: Create fitness-scale scatter plots ===
    print("\n=== Step 4a: Creating fitness-scale scatter plots ===")

    create_fitness_comparison_plots(experimental, baseline_GEM, pretuning_GEM, posttuning_GEM,
                                   baseline_wildtype, pretuning_wildtype, posttuning_wildtype,
                                   args.output, args.fit_thresh)

    # === Step 4b: Create distribution plots ===
    print("\n=== Step 4b: Creating distribution plots ===")

    create_distribution_plots(experimental, baseline_GEM, pretuning_GEM, posttuning_GEM, args.output)

    # === Step 4c: Create fitness distribution plots ===
    print("\n=== Step 4c: Creating fitness distribution plots ===")

    create_fitness_distribution_plots(experimental, baseline_GEM, pretuning_GEM, posttuning_GEM,
                                    baseline_wildtype, pretuning_wildtype, posttuning_wildtype,
                                    args.output, args.fit_thresh)

    # === Step 5: Create classification visualizations ===
    print("\n=== Step 5: Creating classification visualizations ===")

    if baseline_GEM is not None:
        create_classification_plots(experimental, baseline_GEM, 'Baseline GEM', args.output,
                                   args.fit_thresh, args.sim_thresh)

    if pretuning_GEM is not None:
        create_classification_plots(experimental, pretuning_GEM, 'Pre-tuning kinGEMs', args.output,
                                   args.fit_thresh, args.sim_thresh)

    if posttuning_GEM is not None:
        create_classification_plots(experimental, posttuning_GEM, 'Post-tuning kinGEMs', args.output,
                                   args.fit_thresh, args.sim_thresh)

    # === Step 6: Create metrics tables ===
    print("\n=== Step 6: Creating metrics tables ===")

    # Correlation metrics table (growth rate space)
    create_metrics_table(correlation_metrics, args.output)

    # Fitness-based correlation metrics table
    if fitness_correlation_metrics:
        fitness_df = pd.DataFrame(fitness_correlation_metrics).T

        # Round values
        fitness_df = fitness_df.round({
            'pearson_r_fitness': 4,
            'pearson_p_fitness': 6,
            'spearman_r_fitness': 4,
            'spearman_p_fitness': 6,
            'kendall_tau_fitness': 4,
            'kendall_p_fitness': 6,
            'r2_fitness': 4,
            'rmse_fitness': 4,
            'mae_fitness': 4,
            'energy_distance_fitness': 4,
            'wasserstein_distance_fitness': 4,
            'wild_type_growth_used': 6,
            'pred_fitness_mean': 4,
            'pred_fitness_std': 4,
            'pred_fitness_min': 4,
            'pred_fitness_max': 4
        })

        # Save to CSV
        fitness_csv = os.path.join(args.output, 'validation_metrics_fitness.csv')
        fitness_df.to_csv(fitness_csv)
        print(f"  ✓ Saved fitness-based correlation metrics: {fitness_csv}")

        # Print to console
        print("\n" + "="*70)
        print("=== Fitness-Based Correlation Metrics ===")
        print("="*70)
        print(fitness_df.to_string())
        print("="*70)
        print("Note: Predictions converted to log2(mutant/WT) fitness scale")
        print("="*70)

    # Classification metrics table
    if classification_metrics:
        class_df = pd.DataFrame(classification_metrics).T

        # Round values
        class_df = class_df.round({
            'accuracy': 4,
            'balanced_accuracy': 4,
            'precision': 4,
            'recall': 4,
            'f1_score': 4,
            'mcc': 4,
            'roc_auc': 4,
            'average_precision': 4,
            'optimal_threshold': 4,
            'accuracy_optimal': 4,
            'balanced_accuracy_optimal': 4
        })

        # Save to CSV
        class_csv = os.path.join(args.output, 'classification_metrics.csv')
        class_df.to_csv(class_csv)
        print(f"  ✓ Saved classification metrics: {class_csv}")

        # Print to console
        print("\n" + "="*70)
        print("=== Classification Metrics ===")
        print("="*70)
        print(class_df.to_string())
        print("="*70)

    # === Step 7: Improvement analysis ===
    print("\n=== Step 7: Improvement analysis ===")

    create_improvement_analysis(
        correlation_metrics.get('Baseline GEM'),
        correlation_metrics.get('Pre-tuning kinGEMs'),
        correlation_metrics.get('Post-tuning kinGEMs'),
        args.output
    )

    # === Step 8: Save combined metadata ===
    print("\n=== Step 8: Saving metadata ===")

    combined_metadata = {
        'compilation_timestamp': datetime.now().isoformat(),
        'input_directory': args.input,
        'output_directory': args.output,
        'fit_threshold': args.fit_thresh,
        'sim_threshold': args.sim_thresh,
        'baseline': baseline_meta,
        'pretuning': pretuning_meta,
        'posttuning': posttuning_meta,
        'correlation_metrics': correlation_metrics,
        'fitness_correlation_metrics': fitness_correlation_metrics,
        'classification_metrics': classification_metrics
    }

    meta_file = os.path.join(args.output, 'compiled_metadata.json')
    with open(meta_file, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    print(f"  ✓ Saved combined metadata: {meta_file}")

    print("\n" + "="*70)
    print("=== Compilation Complete ===")
    print("="*70)
    print(f"Results saved to: {args.output}")
    print("\nGenerated files:")
    print("  - validation_metrics.csv (correlation metrics in growth rate space)")
    print("  - validation_metrics_fitness.csv (correlation metrics in fitness space)")
    print("  - classification_metrics.csv (classification metrics)")
    print("  - validation_improvements.csv")
    print("  - validation_comparison.png (scatter: experimental fitness vs predicted growth)")
    print("  - fitness_comparison_scatter.png (scatter: experimental fitness vs predicted fitness)")
    print("  - fitness_comparison_combined.png (combined fitness scatter for all models)")
    print("  - prediction_distributions_histograms.png (individual histograms)")
    print("  - prediction_distributions_density.png (overlaid density plots)")
    print("  - prediction_distributions_boxplot.png (box plot comparison)")
    print("  - prediction_distributions_cdf.png (cumulative distribution)")
    print("  - prediction_distributions_violin.png (violin plot)")
    print("  - prediction_distribution_statistics.csv (distribution stats)")
    print("  - fitness_distributions_histograms.png (fitness histograms)")
    print("  - fitness_distributions_density.png (fitness density plots)")
    print("  - fitness_distributions_boxplot.png (fitness box plot)")
    print("  - fitness_distributions_violin.png (fitness violin plot)")
    print("  - fitness_distributions_cdf.png (fitness CDF)")
    print("  - fitness_distribution_statistics.csv (fitness distribution stats)")
    print("  - *_roc_curve.png (ROC curves for each model)")
    print("  - *_pr_curve.png (Precision-Recall curves)")
    print("  - *_confusion_matrix.png (confusion matrices)")
    print("  - compiled_metadata.json")
    print("="*70)


if __name__ == '__main__':
    main()

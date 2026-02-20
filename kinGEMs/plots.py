"""
Visualization module for kinGEMs.

This module provides functions for visualizing results from kinGEMs analyses,
including flux distributions, enzyme usage, and parameter optimization progress.
"""

import os

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea, HPacker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

from .config import ensure_dir_exists

matplotlib.use("Agg")

# ============================================================================
# GLOBAL PLOT CONFIGURATION
# ============================================================================

# FVA Ablation Level Colors - Consistent across all plots
FVA_LEVEL_COLORS = {
    'Level 1: Baseline GEM': '#1f77b4',
    'Level 2: Single Enzyme': '#ff7f0e',
    'Level 3a: + Isoenzymes': '#2ca02c',
    'Level 3b: + Complexes': '#d62728',
    'Level 3c: + Promiscuous': '#9467bd',
    'Level 4: All Constraints': '#8c564b',
    'Level 5: Post-Tuned': '#e377c2'
}

# Font sizes - Consistent across all plots
FONT_SIZES = {
    'title': 18,
    'subtitle': 16,
    'axis_label': 15,
    'tick_label': 14,
    'legend': 14,
    'annotation': 12
}

# Default figure settings
DEFAULT_DPI = 300
DEFAULT_FIGSIZE_SINGLE = (12, 8)
DEFAULT_FIGSIZE_WIDE = (14, 8)
DEFAULT_FIGSIZE_COMPACT = (10, 6)

# ============================================================================


def set_plotting_style(style="whitegrid"):
    """
    Set a consistent style for all plots using global configuration.

    Parameters
    ----------
    style : str, optional
        The seaborn style to use

    Returns
    -------
    None
    """
    sns.set_style(style)
    plt.rcParams['font.size'] = FONT_SIZES['tick_label']
    plt.rcParams['axes.labelsize'] = FONT_SIZES['axis_label']
    plt.rcParams['axes.titlesize'] = FONT_SIZES['title']
    plt.rcParams['xtick.labelsize'] = FONT_SIZES['tick_label']
    plt.rcParams['ytick.labelsize'] = FONT_SIZES['tick_label']
    plt.rcParams['legend.fontsize'] = FONT_SIZES['legend']

def plot_flux_distribution(df_FBA, n_reactions=20, output_path=None, figsize=(12, 8),
                          show=False, absolute=True, exclude_exchanges=True):
    """
    Plot the flux distribution for the top reactions.

    Parameters
    ----------
    df_FBA : pandas.DataFrame
        DataFrame with FBA results
    n_reactions : int, optional
        Number of top reactions to show
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    absolute : bool, optional
        Whether to sort by absolute flux values
    exclude_exchanges : bool, optional
        Whether to exclude exchange reactions

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    # Set the plotting style
    set_plotting_style()

    # Filter for reaction fluxes
    df_flux = df_FBA[df_FBA['Variable'] == 'reaction'].copy()

    # Exclude exchange reactions if requested
    if exclude_exchanges:
        df_flux = df_flux[~df_flux['Index'].str.startswith('EX_')]

    # Sort by absolute flux value if requested
    if absolute:
        df_flux['AbsValue'] = df_flux['Value'].abs()
        df_flux = df_flux.sort_values('AbsValue', ascending=False)
    else:
        df_flux = df_flux.sort_values('Value', ascending=False)

    # Take top N reactions
    df_flux = df_flux.head(n_reactions)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar plot
    colors = ['#1f77b4' if val >= 0 else '#d62728' for val in df_flux['Value']]
    ax.barh(df_flux['Index'], df_flux['Value'], color=colors)

    # Add labels and title
    ax.set_xlabel('Flux (mmol/gDW/h)')
    ax.set_title('Top Fluxes in Model')

    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Add grid
    ax.grid(True, axis='x', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig

def plot_enzyme_usage(df_enzyme, n_enzymes=20, output_path=None, figsize=(12, 8),
                     show=False, by_weight=True):
    """
    Plot enzyme usage distribution for the top enzymes.

    Parameters
    ----------
    df_enzyme : pandas.DataFrame
        DataFrame with enzyme data
    n_enzymes : int, optional
        Number of top enzymes to show
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    by_weight : bool, optional
        Whether to sort by enzyme weight (True) or concentration (False)

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    # Set the plotting style
    set_plotting_style()

    # Copy the dataframe to avoid modifying the original
    df = df_enzyme.copy()

    # Determine the sort column based on by_weight flag
    sort_column = 'enzyme weight (g/gDCW)' if by_weight else 'enzyme_conc'

    # Ensure the required columns exist
    if sort_column not in df.columns:
        raise ValueError(f"Required column '{sort_column}' not found in the dataframe")

    # Sort the dataframe and take the top N enzymes
    df = df.sort_values(sort_column, ascending=False).head(n_enzymes)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar plot
    ax.barh(df['gene_ID'], df[sort_column], color='#1f77b4')

    # Add labels and title
    if by_weight:
        ax.set_xlabel('Enzyme Weight (g/gDCW)')
        ax.set_title('Top Enzymes by Weight')
    else:
        ax.set_xlabel('Enzyme Concentration (mmol/gDCW)')
        ax.set_title('Top Enzymes by Concentration')

    # Add grid
    ax.grid(True, axis='x', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig

def plot_kcat_distribution(kcat_df, output_path=None, figsize=(10, 6), show=False,
                         log_scale=True):
    """
    Plot the distribution of kcat values.

    Parameters
    ----------
    kcat_df : pandas.DataFrame
        DataFrame with kcat values
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    log_scale : bool, optional
        Whether to use a log scale for the x-axis

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    # Set the plotting style
    set_plotting_style()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram
    sns.histplot(kcat_df['kcat (1/hr)'].dropna(), bins=30, kde=True, ax=ax)

    # Set log scale if requested
    if log_scale:
        ax.set_xscale('log')

    # Add labels and title
    ax.set_xlabel('kcat (1/hr)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of kcat Values')

    # Adjust layout
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig

def plot_kcat_comparison(original_kcat_df, optimized_kcat_df, output_path=None,
                        figsize=(12, 10), show=False):
    """
    Compare original and optimized kcat values.

    Parameters
    ----------
    original_kcat_df : pandas.DataFrame
        DataFrame with original kcat values
    optimized_kcat_df : pandas.DataFrame
        DataFrame with optimized kcat values
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    # Set the plotting style
    set_plotting_style()

    # Prepare the data
    original = original_kcat_df['kcat (1/hr)'].dropna()
    optimized = optimized_kcat_df['kcat (1/hr)'].dropna()

    # Create combined dataframe for seaborn
    combined_data = pd.DataFrame({
        'kcat (1/hr)': pd.concat([original, optimized]),
        'Type': ['Original kcats'] * len(original) + ['Optimized kcats'] * len(optimized),
    })

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # 1. Boxplot comparison
    sns.boxplot(x='kcat (1/hr)', y='Type', data=combined_data, ax=axes[0])
    axes[0].set_xscale('log')
    axes[0].set_title('Comparison of kcat Distributions')
    axes[0].set_xlabel('kcat (1/hr), log scale')
    axes[0].set_ylabel('')

    # 2. Histograms
    sns.histplot(original, bins=30, kde=True, ax=axes[1], color='orange', label='Original')
    axes[1].set_xscale('log')
    axes[1].set_title('Original kcat Distribution')
    axes[1].set_xlabel('kcat (1/hr), log scale')
    axes[1].set_ylabel('Count')

    sns.histplot(optimized, bins=30, kde=True, ax=axes[2], color='blue', label='Optimized')
    axes[2].set_xscale('log')
    axes[2].set_title('Optimized kcat Distribution')
    axes[2].set_xlabel('kcat (1/hr), log scale')
    axes[2].set_ylabel('Count')

    # Adjust layout
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig

def plot_fva_results(fva_results, n_reactions=20, output_path=None, figsize=(12, 8),
                    show=False, sort_by='range'):
    """
    Plot Flux Variability Analysis results.

    Parameters
    ----------
    fva_results : pandas.DataFrame
        DataFrame with FVA results
    n_reactions : int, optional
        Number of reactions to display
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    sort_by : str, optional
        How to sort the reactions ('range', 'min', 'max', 'abs_max')

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    # Set the plotting style
    set_plotting_style()

    # Copy and prepare the data
    df = fva_results.copy()
    df['Flux Range'] = df['Max Solutions'] - df['Min Solutions']
    df['Abs Max'] = df['Max Solutions'].abs()

    # Sort based on the specified criterion
    if sort_by == 'range':
        df = df.sort_values('Flux Range', ascending=False)
    elif sort_by == 'min':
        df = df.sort_values('Min Solutions')
    elif sort_by == 'max':
        df = df.sort_values('Max Solutions', ascending=False)
    elif sort_by == 'abs_max':
        df = df.sort_values('Abs Max', ascending=False)

    # Take the top N reactions
    df = df.head(n_reactions)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot flux ranges as horizontal lines with points at min and max
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([row['Min Solutions'], row['Max Solutions']], [i, i], 'o-',
               color='#1f77b4', linewidth=2, markersize=6)

    # Set y-axis ticks and labels
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Reactions'])

    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Flux (mmol/gDW/h)')
    ax.set_title('Flux Variability Analysis Results')

    # Add grid
    ax.grid(True, axis='x', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig

def plot_reaction_control(control_df, n_reactions=20, output_path=None, figsize=(12, 8),
                         show=False):
    """
    Plot reaction control coefficients showing which enzymes most control reaction fluxes.

    Parameters
    ----------
    control_df : pandas.DataFrame
        DataFrame with control coefficients
    n_reactions : int, optional
        Number of reaction-enzyme pairs to display
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    # Set the plotting style
    set_plotting_style()

    # Copy and prepare the data
    df = control_df.copy()

    # Sort by absolute control coefficient
    df['Abs Control'] = df['Control Coefficient'].abs()
    df = df.sort_values('Abs Control', ascending=False).head(n_reactions)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar plot
    bars = ax.barh(df['Pair'], df['Control Coefficient'])

    # Color bars based on sign
    for i, bar in enumerate(bars):
        if df.iloc[i]['Control Coefficient'] < 0:
            bar.set_color('#d62728')  # Red for negative
        else:
            bar.set_color('#1f77b4')  # Blue for positive

    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Control Coefficient')
    ax.set_title('Enzyme Control over Reaction Fluxes')

    # Add grid
    ax.grid(True, axis='x', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig

def plot_correlation_heatmap(df, columns=None, output_path=None, figsize=(12, 10),
                            show=False, method='pearson'):
    """
    Plot a correlation heatmap for model parameters or results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with data to correlate
    columns : list, optional
        List of columns to include in the correlation
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    method : str, optional
        Correlation method ('pearson', 'spearman', or 'kendall')

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    # Set the plotting style
    set_plotting_style()

    # Use specified columns or all numeric columns
    if columns is None:
        # Filter for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_corr = df[numeric_cols]
    else:
        df_corr = df[columns]

    # Calculate correlation matrix
    corr_matrix = df_corr.corr(method=method)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    # Add title
    ax.set_title(f'{method.capitalize()} Correlation Heatmap')

    # Adjust layout
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig

def create_summary_dashboard(results_dict, output_path=None, figsize=(18, 12), show=False):
    """
    Create a summary dashboard with multiple plots for model analysis.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing various results dataframes
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    # Set the plotting style
    set_plotting_style()

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    # Define grid layout
    gs = fig.add_gridspec(2, 3)

    # 1. Flux distribution
    if 'fba_results' in results_dict:
        ax1 = fig.add_subplot(gs[0, 0])
        df_flux = results_dict['fba_results']
        df_flux = df_flux[df_flux['Variable'] == 'reaction'].copy()
        df_flux = df_flux.sort_values('Value', key=abs, ascending=False).head(10)
        colors = ['#1f77b4' if val >= 0 else '#d62728' for val in df_flux['Value']]
        ax1.barh(df_flux['Index'], df_flux['Value'], color=colors)
        ax1.set_title('Top Fluxes')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # 2. Enzyme usage
    if 'enzyme_data' in results_dict:
        ax2 = fig.add_subplot(gs[0, 1])
        df_enzyme = results_dict['enzyme_data']
        sort_col = 'enzyme weight (g/gDCW)' if 'enzyme weight (g/gDCW)' in df_enzyme.columns else 'enzyme_conc'
        df_enzyme = df_enzyme.sort_values(sort_col, ascending=False).head(10)
        ax2.barh(df_enzyme['gene_ID'], df_enzyme[sort_col], color='#1f77b4')
        ax2.set_title('Top Enzymes')

    # 3. FVA results
    if 'fva_results' in results_dict:
        ax3 = fig.add_subplot(gs[0, 2])
        df_fva = results_dict['fva_results']
        df_fva['Flux Range'] = df_fva['Max Solutions'] - df_fva['Min Solutions']
        df_fva = df_fva.sort_values('Flux Range', ascending=False).head(10)

        for i, (_, row) in enumerate(df_fva.iterrows()):
            ax3.plot([row['Min Solutions'], row['Max Solutions']], [i, i], 'o-',
                   color='#1f77b4', linewidth=2, markersize=6)

        ax3.set_yticks(range(len(df_fva)))
        ax3.set_yticklabels(df_fva['Reactions'])
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Flux Variability')

    # 4. kcat distribution
    if 'kcat_data' in results_dict:
        ax4 = fig.add_subplot(gs[1, 0])
        df_kcat = results_dict['kcat_data']
        sns.histplot(df_kcat['kcat (1/hr)'].dropna(), bins=20, kde=True, ax=ax4)
        ax4.set_xscale('log')
        ax4.set_title('kcat Distribution')

    # 5. Simulated annealing progress
    if 'annealing_data' in results_dict:
        ax5 = fig.add_subplot(gs[1, 1:])
        annealing_data = results_dict['annealing_data']
        iterations = annealing_data['iterations']
        biomasses = annealing_data['biomasses']
        ax5.plot(iterations, biomasses, marker='o', linestyle='-', color='#1f77b4',
               markerfacecolor='white', markersize=6)
        ax5.set_title('Optimization Progress')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Biomass (1/hr)')

    # Adjust layout
    fig.suptitle('Model Analysis Summary', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_annealing_progress(iterations, biomasses, output_path=None, show=False):
    """
    Plot the progress of simulated annealing optimization.

    Parameters
    ----------
    iterations : list
        List of iteration numbers
    biomasses : list
        List of biomass values at each iteration
    output_path : str, optional
        Path to save the plot figure
    show : bool, optional
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure object
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, biomasses, marker='o', linestyle='-', color='b', label='Biomass')

    # Add labels and title
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Biomass (1/hr)', fontsize=16)
    plt.title('Biomass vs Iterations')
    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return plt.gcf()


def plot_kcat_annealing_comparison(initial_df, tuned_df, output_path=None,
                                  figsize=(16, 10), show=False, model_name=None):
    """
    Compare initial and post-annealing kcat values with multiple visualizations.

    Creates a comprehensive comparison showing:
    1. Scatter plot of initial vs tuned kcat values
    2. Histogram comparison of distributions
    3. Summary statistics

    Parameters
    ----------
    initial_df : pandas.DataFrame
        DataFrame with initial kcat values (must have 'kcat_mean' column)
    tuned_df : pandas.DataFrame
        DataFrame with tuned kcat values (must have 'kcat_updated' column)
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    model_name : str, optional
        Name of the model for plot title

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    set_plotting_style()

    # Check for required columns
    if 'kcat_mean' not in initial_df.columns:
        raise ValueError("No 'kcat_mean' column found in initial_df")

    if 'kcat_updated' not in tuned_df.columns:
        raise ValueError("No 'kcat_updated' column found in tuned_df")

    # Merge dataframes on reaction and gene identifiers
    merge_cols = []
    if 'Reactions' in initial_df.columns and 'Reactions' in tuned_df.columns:
        merge_cols.append('Reactions')
    if 'Single_gene' in initial_df.columns and 'Single_gene' in tuned_df.columns:
        merge_cols.append('Single_gene')

    if not merge_cols:
        raise ValueError("Cannot merge dataframes: no common identifier columns found")

    # Merge and clean data
    merged = pd.merge(
        initial_df[merge_cols + ['kcat_mean']],
        tuned_df[merge_cols + ['kcat_updated']],
        on=merge_cols,
        how='inner'
    ).dropna(subset=['kcat_mean', 'kcat_updated'])

    if len(merged) == 0:
        print("Warning: No matching kcat values found between initial and tuned datasets")
        return None

    # Calculate statistics
    initial_median = merged['kcat_mean'].median()
    tuned_median = merged['kcat_updated'].median()
    fold_changes = merged['kcat_updated'] / merged['kcat_mean']
    median_fold_change = fold_changes.median()
    n_increased = (merged['kcat_updated'] > merged['kcat_mean']).sum()
    n_decreased = (merged['kcat_updated'] < merged['kcat_mean']).sum()

    # Create figure with subplots - add space for stats on right
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3,
                          width_ratios=[1, 1, 0.85])

    # 1. Scatter plot: Initial vs Tuned
    ax1 = fig.add_subplot(gs[0, :2])

    ax1.scatter(merged['kcat_mean'], merged['kcat_updated'],
                alpha=0.6, s=50, color='#1f77b4')

    # Add diagonal line (y=x)
    lims = [
        min(merged['kcat_mean'].min(), merged['kcat_updated'].min()),
        max(merged['kcat_mean'].max(), merged['kcat_updated'].max())
    ]
    ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='No change (y=x)')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Initial kcat (1/hr)', fontsize=FONT_SIZES['axis_label'])
    ax1.set_ylabel('Post-Annealing kcat (1/hr)', fontsize=FONT_SIZES['axis_label'])
    title = 'Initial vs Post-Annealing kcat Values'
    if model_name:
        title = f'{model_name}: {title}'
    ax1.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold')
    ax1.legend(fontsize=FONT_SIZES['legend'])
    ax1.grid(True, alpha=0.3)

    # 2. KDE: Initial kcat distribution
    ax2 = fig.add_subplot(gs[1, 0])
    # Use log10 transformed data for KDE
    log_initial = np.log10(merged['kcat_mean'])
    log_initial_clean = log_initial[np.isfinite(log_initial)]

    # Compute shared x-axis limits across both KDE panels (in log10 space)
    log_tuned = np.log10(merged['kcat_updated'])
    log_tuned_clean = log_tuned[np.isfinite(log_tuned)]
    all_log = pd.concat([log_initial_clean, log_tuned_clean])
    shared_xmin = 10 ** (all_log.min() - 0.5)
    shared_xmax = 10 ** (all_log.max() + 0.5)

    if len(log_initial_clean) > 1:
        from scipy.stats import gaussian_kde
        kde_initial = gaussian_kde(log_initial_clean)
        x_range = np.linspace(log_initial_clean.min(), log_initial_clean.max(), 200)
        ax2.fill_between(10**x_range, kde_initial(x_range), alpha=0.7, color='#8c564b', label='Initial kcat')
        ax2.plot(10**x_range, kde_initial(x_range), color='#6a3d34', linewidth=2)

    ax2.axvline(initial_median, color='#8c564b', linestyle='--', linewidth=2.5, label=f'Median: {initial_median:.1f}')
    ax2.set_xscale('log')
    ax2.set_xlim(shared_xmin, shared_xmax)
    ax2.set_xlabel('kcat (1/hr)', fontsize=FONT_SIZES['axis_label'])
    ax2.set_ylabel('Density', fontsize=FONT_SIZES['axis_label'])
    ax2.set_title('Initial kcat Distribution', fontsize=FONT_SIZES['subtitle'])
    ax2.legend(fontsize=FONT_SIZES['legend'], loc='upper center', bbox_to_anchor=(0.35, -0.15), ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. KDE: Post-annealing kcat distribution
    ax3 = fig.add_subplot(gs[1, 1])

    if len(log_tuned_clean) > 1:
        kde_tuned = gaussian_kde(log_tuned_clean)
        x_range = np.linspace(log_tuned_clean.min(), log_tuned_clean.max(), 200)
        ax3.fill_between(10**x_range, kde_tuned(x_range), alpha=0.7, color='#e377c2', label='Post-Annealing kcat')
        ax3.plot(10**x_range, kde_tuned(x_range), color='#c45ba0', linewidth=2)

    ax3.axvline(tuned_median, color='#e377c2', linestyle='--', linewidth=2.5, label=f'Median: {tuned_median:.1f}')
    ax3.set_xscale('log')
    ax3.set_xlim(shared_xmin, shared_xmax)
    ax3.set_xlabel('kcat (1/hr)', fontsize=FONT_SIZES['axis_label'])
    ax3.set_ylabel('Density', fontsize=FONT_SIZES['axis_label'])
    ax3.set_title('Post-Annealing kcat Distribution', fontsize=FONT_SIZES['subtitle'])
    ax3.legend(fontsize=FONT_SIZES['legend'], loc='upper center', bbox_to_anchor=(0.65, -0.15), ncol=2)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Statistics panel on the right
    ax_stats = fig.add_subplot(gs[:, 2])
    ax_stats.axis('off')

    stats_text = (
        f'Summary Statistics\n'
        f'{"="*20}\n\n'
        f'N = {len(merged)}\n\n'
        f'Median initial:\n{initial_median:.1f} 1/hr\n\n'
        f'Median tuned:\n{tuned_median:.1f} 1/hr\n\n'
        f'Median fold change:\n{median_fold_change:.2f}×\n\n'
        f'Increased:\n{n_increased}\n({100*n_increased/len(merged):.1f}%)\n\n'
        f'Decreased:\n{n_decreased}\n({100*n_decreased/len(merged):.1f}%)'
    )

    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                 fontsize=FONT_SIZES['legend'], verticalalignment='center',
                 horizontalalignment='center', fontweight='bold', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                          edgecolor='black', linewidth=2, pad=1))

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"  Saved kcat comparison plot to: {output_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_kcat_annealing_comparison_by_subsystem(
        initial_df, tuned_df, subsystem_col,
        output_path=None, figsize=(20, 16), show=False,
        model_name=None, max_subsystems=12, ncols=4):
    """
    Compare initial and post-annealing kcat values in a grid of scatter subplots,
    one panel per metabolic subsystem.

    Parameters
    ----------
    initial_df : pandas.DataFrame
        DataFrame with initial kcat values (must have 'kcat_mean' column).
        Must also contain ``subsystem_col``.
    tuned_df : pandas.DataFrame
        DataFrame with tuned kcat values (must have 'kcat_updated' column).
    subsystem_col : str
        Column name in ``initial_df`` containing subsystem labels.
    output_path : str, optional
        Path to save the figure.
    figsize : tuple, optional
        Figure size (width, height).
    show : bool, optional
        Whether to display the plot.
    model_name : str, optional
        Name of the model for the overall figure title.
    max_subsystems : int, optional
        Maximum number of individually plotted subsystems (default 12).
        Remaining subsystems are grouped into an "Other" panel.
    ncols : int, optional
        Number of subplot columns (default 4).

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_plotting_style()

    if 'kcat_mean' not in initial_df.columns:
        raise ValueError("No 'kcat_mean' column found in initial_df")
    if 'kcat_updated' not in tuned_df.columns:
        raise ValueError("No 'kcat_updated' column found in tuned_df")
    if subsystem_col not in initial_df.columns:
        raise ValueError(f"Column '{subsystem_col}' not found in initial_df")

    # Merge on shared identifier columns
    merge_cols = []
    if 'Reactions' in initial_df.columns and 'Reactions' in tuned_df.columns:
        merge_cols.append('Reactions')
    if 'Single_gene' in initial_df.columns and 'Single_gene' in tuned_df.columns:
        merge_cols.append('Single_gene')
    if not merge_cols:
        raise ValueError("Cannot merge dataframes: no common identifier columns found")

    merged = pd.merge(
        initial_df[merge_cols + ['kcat_mean', subsystem_col]],
        tuned_df[merge_cols + ['kcat_updated']],
        on=merge_cols,
        how='inner'
    ).dropna(subset=['kcat_mean', 'kcat_updated'])

    if len(merged) == 0:
        print("Warning: No matching kcat values found between initial and tuned datasets")
        return None

    # Group rare subsystems into "Other"
    top_subs = (
        merged[subsystem_col]
        .value_counts()
        .head(max_subsystems)
        .index.tolist()
    )
    merged['_sub_label'] = merged[subsystem_col].where(
        merged[subsystem_col].isin(top_subs), other='Other'
    )

    # Plot order: top subsystems by count, "Other" at end
    sub_counts = merged['_sub_label'].value_counts()
    groups = [s for s in top_subs if s in sub_counts.index]
    if 'Other' in sub_counts.index:
        groups.append('Other')

    n_groups = len(groups)
    nrows = int(np.ceil(n_groups / ncols))

    # Colour palette: tab20 for named subsystems, grey for Other
    tab20 = plt.cm.get_cmap('tab20', len(top_subs) + 1)
    palette = {sub: tab20(i) for i, sub in enumerate(top_subs)}
    palette['Other'] = (0.6, 0.6, 0.6, 1.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Shared axis limits across all panels
    all_vals = pd.concat([merged['kcat_mean'], merged['kcat_updated']]).dropna()
    pos_vals = all_vals[all_vals > 0]
    global_min = pos_vals.min() * 0.5
    global_max = pos_vals.max() * 2.0
    lims = [global_min, global_max]

    for idx, sub in enumerate(groups):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        grp = merged[merged['_sub_label'] == sub]
        color = palette.get(sub, palette['Other'])

        ax.scatter(
            grp['kcat_mean'], grp['kcat_updated'],
            alpha=0.6, s=20, color=color, rasterized=True
        )

        # y = x diagonal
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f'{sub}\n(n={len(grp):,})',
            fontsize=FONT_SIZES['annotation'],
            fontweight='bold',
            pad=4
        )
        ax.tick_params(labelsize=FONT_SIZES['tick_label'] - 2)

    # Hide unused axes
    for idx in range(n_groups, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared axis labels
    fig.supxlabel('Initial kcat (1/hr)', fontsize=FONT_SIZES['axis_label'], y=0.01)
    fig.supylabel('Post-Annealing kcat (1/hr)', fontsize=FONT_SIZES['axis_label'], x=0.01)

    suptitle = 'Initial vs Post-Annealing kcat by Subsystem'
    if model_name:
        suptitle = f'{model_name}: {suptitle}'
    fig.suptitle(suptitle, fontsize=FONT_SIZES['title'], fontweight='bold', y=1.01)

    plt.tight_layout()

    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"  Saved kcat subsystem comparison plot to: {output_path}")

    if show:
        plt.show()

    return fig


def plot_davidi_kcat_kmax_analysis(df_kcat_kmax, df_kapp=None, output_path=None,
                                   figsize=(18, 10), show=False):
    """
    Analyze and visualize kcat vs. kmax data from Davidi et al. 2016 PNAS.

    Creates a comprehensive figure with:
    1. Scatter plot of kcat[s-1] vs kmax[s-1] (log-log scale, colored by kcat/kmax ratio)
    2. KDE of kcat[s-1] distribution
    3. KDE of kmax[s-1] distribution
    4. KDE of all kapp[s-1] values across conditions (if df_kapp provided)
    5. Summary statistics panel

    Parameters
    ----------
    df_kcat_kmax : pandas.DataFrame
        DataFrame from the 'kcat vs. kmax' sheet (read with header=2).
        Must contain columns 'kcat [s-1]' and 'kmax [s-1]'.
    df_kapp : pandas.DataFrame, optional
        DataFrame from the 'kapp 1s' sheet (read with header=2).
        Condition columns (after 'bnumber') are flattened and plotted.
    output_path : str, optional
        Path to save the figure.
    figsize : tuple, optional
        Figure size (width, height).
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure.
    """
    from scipy.stats import gaussian_kde, spearmanr

    set_plotting_style()

    # --- Parse kcat vs. kmax data ---
    kcat_col = 'kcat [s-1]'
    kmax_col = 'kmax [s-1]'
    ratio_col = 'kcat / kmax'

    df = df_kcat_kmax[[kcat_col, kmax_col, ratio_col]].dropna()
    kcat = df[kcat_col].values.astype(float)
    kmax = df[kmax_col].values.astype(float)
    ratio = df[ratio_col].values.astype(float)

    # --- Parse kapp data (melt all condition columns to 1-D) ---
    kapp_values = None
    if df_kapp is not None:
        id_cols = [c for c in df_kapp.columns if c in ('reaction (model name)', 'bnumber')]
        cond_cols = [c for c in df_kapp.columns if c not in id_cols]
        raw = df_kapp[cond_cols].values.flatten().astype(float)
        kapp_values = raw[np.isfinite(raw) & (raw > 0)]

    # --- Layout ---
    n_kde_panels = 3 if kapp_values is not None else 2
    ncols = n_kde_panels + 1          # KDE panels + stats
    width_ratios = [1] * n_kde_panels + [0.85]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, ncols, hspace=0.35, wspace=0.35,
                          width_ratios=width_ratios)

    # =========================================================
    # 1.  Scatter: kcat vs kmax  (spans top row, all KDE cols)
    # =========================================================
    ax1 = fig.add_subplot(gs[0, :n_kde_panels])

    log_ratio = np.log10(ratio)
    vmax = np.percentile(np.abs(log_ratio), 95)
    sc = ax1.scatter(kcat, kmax, c=log_ratio, cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax,
                     alpha=0.75, s=60, edgecolors='k', linewidths=0.4)
    cbar = plt.colorbar(sc, ax=ax1, pad=0.01)
    cbar.set_label('log₁₀(kcat / kmax)', fontsize=FONT_SIZES['annotation'])
    cbar.ax.tick_params(labelsize=FONT_SIZES['annotation'])

    # y = x line
    lims = [min(kcat.min(), kmax.min()) * 0.5,
            max(kcat.max(), kmax.max()) * 2]
    ax1.plot(lims, lims, 'k--', alpha=0.55, linewidth=2, label='kcat = kmax (y = x)')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('kcat [s⁻¹]  (in-vitro)', fontsize=FONT_SIZES['axis_label'])
    ax1.set_ylabel('kmax [s⁻¹]  (max in-vivo kapp)', fontsize=FONT_SIZES['axis_label'])
    ax1.set_title('kcat vs. kmax — Davidi et al. 2016 (N = {:d})'.format(len(df)),
                  fontsize=FONT_SIZES['title'], fontweight='bold')
    ax1.legend(fontsize=FONT_SIZES['legend'], loc='upper left')
    ax1.grid(True, alpha=0.3)

    # =========================================================
    # 2.  KDE — kcat
    # =========================================================
    ax2 = fig.add_subplot(gs[1, 0])
    kcat_color = '#8c564b'
    log_kcat = np.log10(kcat[kcat > 0])
    if len(log_kcat) > 1:
        kde = gaussian_kde(log_kcat)
        xr = np.linspace(log_kcat.min(), log_kcat.max(), 300)
        ax2.fill_between(10**xr, kde(xr), alpha=0.7, color=kcat_color)
        ax2.plot(10**xr, kde(xr), color='#5a3328', linewidth=2)
    med_kcat = np.median(kcat)
    ax2.axvline(med_kcat, color=kcat_color, linestyle='--', linewidth=2.5,
                label=f'Median: {med_kcat:.1f}')
    ax2.set_xscale('log')
    ax2.set_xlabel('kcat [s⁻¹]', fontsize=FONT_SIZES['axis_label'])
    ax2.set_ylabel('Density', fontsize=FONT_SIZES['axis_label'])
    ax2.set_title('kcat Distribution', fontsize=FONT_SIZES['subtitle'])
    ax2.legend(fontsize=FONT_SIZES['legend'], loc='upper center',
               bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')

    # =========================================================
    # 3.  KDE — kmax
    # =========================================================
    ax3 = fig.add_subplot(gs[1, 1])
    kmax_color = '#e377c2'
    log_kmax = np.log10(kmax[kmax > 0])
    if len(log_kmax) > 1:
        kde = gaussian_kde(log_kmax)
        xr = np.linspace(log_kmax.min(), log_kmax.max(), 300)
        ax3.fill_between(10**xr, kde(xr), alpha=0.7, color=kmax_color)
        ax3.plot(10**xr, kde(xr), color='#8f3d7a', linewidth=2)
    med_kmax = np.median(kmax)
    ax3.axvline(med_kmax, color=kmax_color, linestyle='--', linewidth=2.5,
                label=f'Median: {med_kmax:.1f}')
    ax3.set_xscale('log')
    ax3.set_xlabel('kmax [s⁻¹]', fontsize=FONT_SIZES['axis_label'])
    ax3.set_ylabel('Density', fontsize=FONT_SIZES['axis_label'])
    ax3.set_title('kmax Distribution', fontsize=FONT_SIZES['subtitle'])
    ax3.legend(fontsize=FONT_SIZES['legend'], loc='upper center',
               bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax3.grid(True, alpha=0.3, axis='y')

    # =========================================================
    # 4.  KDE — kapp (all conditions, optional)
    # =========================================================
    if kapp_values is not None and n_kde_panels == 3:
        ax4 = fig.add_subplot(gs[1, 2])
        kapp_color = '#1f77b4'
        log_kapp = np.log10(kapp_values[kapp_values > 0])
        if len(log_kapp) > 1:
            kde = gaussian_kde(log_kapp)
            xr = np.linspace(log_kapp.min(), log_kapp.max(), 300)
            ax4.fill_between(10**xr, kde(xr), alpha=0.7, color=kapp_color)
            ax4.plot(10**xr, kde(xr), color='#174f7a', linewidth=2)
        med_kapp = np.median(kapp_values)
        ax4.axvline(med_kapp, color=kapp_color, linestyle='--', linewidth=2.5,
                    label=f'Median: {med_kapp:.2f}')
        ax4.set_xscale('log')
        ax4.set_xlabel('kapp [s⁻¹]', fontsize=FONT_SIZES['axis_label'])
        ax4.set_ylabel('Density', fontsize=FONT_SIZES['axis_label'])
        ax4.set_title('kapp Distribution\n(all conditions)', fontsize=FONT_SIZES['subtitle'])
        ax4.legend(fontsize=FONT_SIZES['legend'], loc='upper center',
                   bbox_to_anchor=(0.5, -0.18), ncol=2)
        ax4.grid(True, alpha=0.3, axis='y')

    # =========================================================
    # 5.  Summary statistics panel
    # =========================================================
    ax_stats = fig.add_subplot(gs[:, -1])
    ax_stats.axis('off')

    rho, p_val = spearmanr(np.log10(kcat), np.log10(kmax))
    n_above = (ratio > 1).sum()   # kcat > kmax  (enzyme not at capacity in vivo)
    n_below = (ratio < 1).sum()   # kmax > kcat  (apparent saturation > in-vitro)

    stats_lines = [
        'Summary Statistics',
        '=' * 20,
        '',
        f'N pairs = {len(df)}',
        '',
        f'Median kcat:',
        f'  {med_kcat:.2f} s⁻¹',
        '',
        f'Median kmax:',
        f'  {med_kmax:.2f} s⁻¹',
        '',
        f'Spearman ρ:',
        f'  {rho:.3f}  (p={p_val:.2e})',
        '',
        f'kcat > kmax:',
        f'  {n_above} ({100*n_above/len(df):.1f}%)',
        '',
        f'kmax > kcat:',
        f'  {n_below} ({100*n_below/len(df):.1f}%)',
        '',
        f'Median kcat/kmax:',
        f'  {np.median(ratio):.1f}×',
    ]
    if kapp_values is not None:
        stats_lines += [
            '',
            f'kapp (all conds):',
            f'  N = {len(kapp_values)}',
            f'  Median = {med_kapp:.2f} s⁻¹',
        ]

    stats_text = '\n'.join(stats_lines)
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=FONT_SIZES['annotation'],
                  verticalalignment='center', horizontalalignment='center',
                  fontweight='bold', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                            edgecolor='black', linewidth=2, pad=1))

    plt.tight_layout()

    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"  Saved Davidi 2016 analysis plot to: {output_path}")

    if show:
        plt.show()

    return fig


def analyze_kcat_changes(original_kcat_file, optimized_kcat_df, output_dir=None, prefix=""):
    """
    Analyze and visualize changes in kcat values after optimization.

    Parameters
    ----------
    original_kcat_file : str
        Path to original kcat values file
    optimized_kcat_df : pandas.DataFrame
        DataFrame with optimized kcat values
    output_dir : str, optional
        Directory to save output files
    prefix : str, optional
        Prefix for output filenames

    Returns
    -------
    pandas.DataFrame
        DataFrame comparing original and optimized kcat values
    """
    # Load original kcat values
    old_kcats = pd.read_csv(original_kcat_file)
    new_kcats = optimized_kcat_df

    # Clean and prepare data
    old_kcats = old_kcats.dropna(subset=['kcat (1/hr)'])
    old_kcats["kcat (1/hr)"] = old_kcats["kcat (1/hr)"].astype(float)
    old_kcats = old_kcats.sort_values(by="kcat (1/hr)", ascending=False)
    old_kcats = old_kcats.drop_duplicates(subset=["Reactions", "Single_gene"], keep="first")
    old_kcats = old_kcats.reset_index(drop=True)
    old_kcats = old_kcats.sort_values(by='Reactions')

    # Clean new kcats
    new_kcats = new_kcats.dropna(subset=['kcat (1/hr)'])
    new_kcats["kcat (1/hr)"] = new_kcats["kcat (1/hr)"].astype(float)
    new_kcats = new_kcats.sort_values(by="kcat (1/hr)", ascending=False)
    new_kcats = new_kcats.drop_duplicates(subset=["Reactions", "Single_gene"], keep="first")
    new_kcats = new_kcats.reset_index(drop=True)
    new_kcats = new_kcats.sort_values(by='Reactions')

    # Merge the two dataframes on 'Reactions'
    merged_kcats = pd.merge(old_kcats, new_kcats,
                            on=['Reactions', 'SMILES', 'Single_gene'],
                            suffixes=('_old', '_new'))

    # Save if output_dir provided
    if output_dir:
        merged_kcats.to_csv(os.path.join(output_dir, f"{prefix}merged_kcats.csv"), index=False)

        # Create visualization plots
        plot_kcat_distribution_comparison(old_kcats, new_kcats, merged_kcats,
                                         output_dir=output_dir, prefix=prefix)

    return merged_kcats

def plot_kcat_distribution_comparison(old_kcats, new_kcats, merged_kcats, output_dir=None, prefix=""):
    """
    Create visualizations comparing original and optimized kcat distributions.

    Parameters
    ----------
    old_kcats : pandas.DataFrame
        DataFrame with original kcat values
    new_kcats : pandas.DataFrame
        DataFrame with optimized kcat values
    merged_kcats : pandas.DataFrame
        DataFrame comparing original and optimized kcat values
    output_dir : str, optional
        Directory to save output figures
    prefix : str, optional
        Prefix for output filenames
    """
    import seaborn as sns

    # Ensure directory exists if provided
    if output_dir:
        ensure_dir_exists(output_dir)

    # Set up visualization style
    sns.set_style("whitegrid")

    # Create histogram comparison
    plt.figure(figsize=(10, 6))
    sns.histplot(old_kcats['kcat (1/hr)'], color='orange', label='Original kcats', kde=False, bins=10)
    sns.histplot(new_kcats['kcat (1/hr)'], color='blue', label='Optimized kcats', kde=False, bins=10, alpha=0.6)
    plt.xlabel('kcat (1/hr), log scale', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Distribution of kcat Values', fontsize=16)
    plt.legend()
    plt.xscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{prefix}kcat_histogram.png"), dpi=300, bbox_inches='tight')

    # Create boxplot comparison
    plt.figure(figsize=(10, 6))
    combined_data = pd.DataFrame({
        'kcat (1/hr)': pd.concat([old_kcats['kcat (1/hr)'], new_kcats['kcat (1/hr)']]),
        'Type': ['Original kcats'] * len(old_kcats) + ['Optimized kcats'] * len(new_kcats),
    })

    sns.boxplot(x='kcat (1/hr)', y='Type', data=combined_data)
    plt.xscale('log')
    plt.title('Comparison of kcat Distributions', fontsize=16)
    plt.xlabel('kcat (1/hr), log scale', fontsize=16)
    plt.ylabel('', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{prefix}kcat_boxplot.png"), dpi=300, bbox_inches='tight')

    # Create scatter plot comparing old vs new kcats
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='kcat (1/hr)_old', y='kcat (1/hr)_new', data=merged_kcats)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Original vs. Optimized kcat Values', fontsize=16)
    plt.xlabel('Original kcat (1/hr)', fontsize=16)
    plt.ylabel('Optimized kcat (1/hr)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add a diagonal line for reference
    lims = [
        min(plt.xlim()[0], plt.ylim()[0]),
        max(plt.xlim()[1], plt.ylim()[1]),
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{prefix}kcat_scatter.png"), dpi=300, bbox_inches='tight')

def calculate_flux_metrics(fva_df):
    """Calculate Flux Variability (FVi) as absolute flux range.

    For irreversible models where reactions are split into forward and reverse
    directions (marked with '_reverse' suffix), this function combines them back
    into the original reversible reaction space before calculating FVi.

    FVi = |max - min| - Absolute flux range for reaction i

    Parameters
    ----------
    fva_df : pandas.DataFrame
        DataFrame with FVA results containing 'Min Solutions' and 'Max Solutions' columns.
        Index should contain reaction IDs.

    Returns
    -------
    pandas.Series
        FVi values (absolute flux ranges) for combined reactions
    """
    # Create a copy to avoid modifying the original
    df = fva_df.copy()

    # Ensure the index contains reaction IDs (strings)
    if 'Reactions' in df.columns:
        df = df.set_index('Reactions')

    # Identify forward and reverse reaction pairs
    # Convert index to strings to handle any non-string types
    reverse_reactions = [rxn for rxn in df.index if isinstance(rxn, str) and rxn.endswith('_reverse')]
    forward_reactions = [rxn.replace('_reverse', '') for rxn in reverse_reactions]

    # Check which forward reactions actually exist
    existing_pairs = [(fwd, rev) for fwd, rev in zip(forward_reactions, reverse_reactions)
                      if fwd in df.index]

    combined_data = {}
    processed_reactions = set()

    # Process paired reactions (forward and reverse)
    for fwd_id, rev_id in existing_pairs:
        if fwd_id in processed_reactions or rev_id in processed_reactions:
            continue

        # Get flux ranges for forward and reverse
        fwd_min = df.loc[fwd_id, 'Min Solutions']
        fwd_max = df.loc[fwd_id, 'Max Solutions']
        rev_min = df.loc[rev_id, 'Min Solutions']
        rev_max = df.loc[rev_id, 'Max Solutions']

        # Convert back to reversible reaction space:
        # Forward flux is positive, reverse flux is negative
        # Combined min = fwd_min - rev_max
        # Combined max = fwd_max - rev_min
        combined_min = fwd_min - rev_max
        combined_max = fwd_max - rev_min

        # Calculate FVi for the combined reaction
        combined_fvi = abs(combined_max - combined_min)
        combined_data[fwd_id] = combined_fvi

        processed_reactions.add(fwd_id)
        processed_reactions.add(rev_id)

    # Process unpaired reactions (already irreversible or not split)
    for rxn_id in df.index:
        if rxn_id not in processed_reactions:
            # Standard calculation for non-split reactions
            fvi = abs(df.loc[rxn_id, 'Max Solutions'] - df.loc[rxn_id, 'Min Solutions'])
            combined_data[rxn_id] = fvi

    # Return as Series with combined reaction IDs
    return pd.Series(combined_data)


def calculate_cumulative_fvi_auc(fvi_series, log_range=(1e-5, 1e3)):
    """
    Calculate the Area Under the Curve (AUC) for cumulative FVi distribution.

    The AUC is computed in log-space (log10 of FVi) using the trapezoidal rule.
    Lower AUC indicates more reactions with low variability (better constraint).

    Parameters
    ----------
    fvi_series : pandas.Series
        Series of FVi values for reactions
    log_range : tuple, optional
        Range for log-scale integration (min_fvi, max_fvi)

    Returns
    -------
    float
        AUC value in log-space
    """
    # Filter out zero and very small values
    fvi_nonzero = fvi_series[fvi_series > 1e-15].values

    if len(fvi_nonzero) == 0:
        return np.nan

    # Sort FVi values
    fvi_sorted = np.sort(fvi_nonzero)
    cumulative = np.arange(1, len(fvi_sorted) + 1) / len(fvi_sorted)

    # Take log10 of FVi values for integration
    log_fvi = np.log10(fvi_sorted)

    # Clip to specified range for consistent comparison
    log_min, log_max = np.log10(log_range[0]), np.log10(log_range[1])

    # Interpolate cumulative distribution to get values at consistent x-points
    # Create uniform grid in log space
    n_points = 1000
    log_grid = np.linspace(log_min, log_max, n_points)

    # Interpolate cumulative probability at grid points
    cumulative_interp = np.interp(log_grid, log_fvi, cumulative, left=0, right=1)

    # Calculate AUC using trapezoidal rule
    auc = np.trapz(cumulative_interp, log_grid)

    # Normalize by the total possible area
    total_area = log_max - log_min
    normalized_auc = auc / total_area

    return normalized_auc


def plot_fva_ablation_cumulative(fva_results_dict, biomass_dict, model_name,
                                output_path=None, figsize=(12, 8), show=False,
                                legend_position='upper left', enhanced=True, show_auc=True):
    """
    Create cumulative FVi distribution plot for FVA ablation study.

    For irreversible models, split reactions (forward and reverse) are automatically
    combined before plotting, ensuring fair comparison across baseline and enzyme-
    constrained models.

    Parameters
    ----------
    fva_results_dict : dict
        Dictionary with level labels as keys and FVA DataFrames as values
    biomass_dict : dict
        Dictionary with level labels as keys and biomass values as values
    model_name : str
        Name of the model for plot title
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    legend_position : str, optional
        Position for the legend ('upper left', 'lower right', etc.)
    enhanced : bool, optional
        Whether to create enhanced version with bottom subplot
    show_auc : bool, optional
        Whether to display AUC values in the legend

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    set_plotting_style()

    if enhanced:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    # Main plot: Cumulative distribution of FVi
    all_fvi_values = []
    fvi_stats = {}
    auc_values = {}

    for label, fva_df in fva_results_dict.items():
        fvi = calculate_flux_metrics(fva_df)
        # Filter out zero values for log plotting
        fvi_nonzero = fvi[fvi > 1e-15]
        if len(fvi_nonzero) == 0:
            fvi_nonzero = np.array([1e-15])

        all_fvi_values.extend(fvi_nonzero)
        fvi_sorted = np.sort(fvi_nonzero)
        cumulative = np.arange(1, len(fvi_sorted) + 1) / len(fvi_sorted)

        # Calculate AUC
        auc = calculate_cumulative_fvi_auc(fvi)
        auc_values[label] = auc

        # Create label with AUC if requested
        if show_auc and not np.isnan(auc):
            plot_label = f"{label} (AUC={auc:.2f})"
        else:
            plot_label = label

        ax1.plot(fvi_sorted, cumulative, label=plot_label,
                color=FVA_LEVEL_COLORS.get(label, None), linewidth=2.5)

        # Store stats for bottom plot
        if enhanced:
            fvi_stats[label] = {
                'mean': fvi.mean(),
                'median': fvi.median(),
                'biomass': biomass_dict[label],
                'auc': auc
            }

    # Add horizontal reference line at cumulative probability 0.5
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Format main plot
    ax1.set_xscale('log')
    # Set x-axis limits based on actual data range
    if len(all_fvi_values) > 0:
        x_min = max(min(all_fvi_values), 1e-15)  # Avoid log(0)
        x_max = max(all_fvi_values)
        ax1.set_xlim(max(x_min, 1e-5), x_max)  # Add some padding
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Flux Variability (FVi)', fontsize=13)
    ax1.set_ylabel('Cumulative Probability', fontsize=13)
    ax1.set_title(f'{model_name}: Cumulative Flux Variability Distribution', fontsize=16, fontweight='bold')
    ax1.legend(loc=legend_position, fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: Bar chart with Median FVi and Biomass
    if enhanced:
        # Prepare data for bar chart
        labels = list(fvi_stats.keys())
        median_fvis = [fvi_stats[label]['median'] for label in labels]
        biomass_values = [fvi_stats[label]['biomass'] for label in labels]

        # Create bar chart for Median FVi
        x_pos = np.arange(len(labels))
        bars = ax2.bar(x_pos, median_fvis, color=[FVA_LEVEL_COLORS.get(label, '#1f77b4') for label in labels], alpha=0.7)

        # Set up left y-axis for Median FVi
        ax2.set_ylabel('Median FVi', fontsize=FONT_SIZES['axis_label'])
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=1e-2)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([label.replace('Level ', 'L').replace(': ', '\n') for label in labels],
                           rotation=45, ha='right', fontsize=FONT_SIZES['annotation'])

        # Create second y-axis for biomass
        ax2_biomass = ax2.twinx()
        line = ax2_biomass.plot(x_pos, biomass_values, 'ko-', linewidth=2, markersize=6, label='Biomass')
        ax2_biomass.set_ylabel('Biomass (1/hr)', fontsize=FONT_SIZES['axis_label'])
        ax2_biomass.set_ylim(0, max(biomass_values) * 1.1)

        # Add subplot title
        ax2.set_title('Mean Flux Variability and Biomass by Constraint Level', fontsize=FONT_SIZES['subtitle'], fontweight='bold')

        # Adjust layout with extra bottom space for rotated labels
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_fva_ablation_boxplot(fva_results_dict, model_name, output_path=None,
                             figsize=(12, 8), show=False):
    """
    Create box plot of FVi distributions for FVA ablation study.

    For irreversible models, split reactions (forward and reverse) are automatically
    combined before plotting, ensuring fair comparison across baseline and enzyme-
    constrained models.

    Parameters
    ----------
    fva_results_dict : dict
        Dictionary with level labels as keys and FVA DataFrames as values
    model_name : str
        Name of the model for plot title
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    set_plotting_style()

    fig, ax = plt.subplots(figsize=figsize)

    fvi_data = []
    labels = []
    colors_list = []

    for label, fva_df in fva_results_dict.items():
        fvi = calculate_flux_metrics(fva_df)
        # Use log scale for better visualization, filter zeros
        fvi_nonzero = fvi[fvi > 1e-15]
        if len(fvi_nonzero) > 0:
            fvi_data.append(np.log10(fvi_nonzero))
            labels.append(label.replace('Level ', 'L').replace(': ', '\n'))
            colors_list.append(FVA_LEVEL_COLORS[label])

    bp = ax.boxplot(fvi_data, labels=labels, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('log₁₀(FVi)', fontsize=FONT_SIZES['axis_label'])
    ax.set_title(f'{model_name}: Distribution of Flux Variability (FVi) by Constraint Level', fontsize=FONT_SIZES['subtitle'])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_fva_ablation_violin(fva_results_dict, model_name, output_path=None,
                            figsize=(14, 8), show=False):
    """
    Create violin plot of FVi distributions for FVA ablation study.

    Violin plots show the full distribution shape (like a rotated kernel density plot)
    combined with a box plot, providing more detail than box plots alone.

    For irreversible models, split reactions (forward and reverse) are automatically
    combined before plotting, ensuring fair comparison across baseline and enzyme-
    constrained models.

    Parameters
    ----------
    fva_results_dict : dict
        Dictionary with level labels as keys and FVA DataFrames as values
    model_name : str
        Name of the model for plot title
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    set_plotting_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for violin plot
    data_for_plot = []
    labels = []

    for label, fva_df in fva_results_dict.items():
        fvi = calculate_flux_metrics(fva_df)
        # Use log10 for better visualization, filter zeros
        fvi_nonzero = fvi[fvi > 1e-15]
        if len(fvi_nonzero) > 0:
            log_fvi = np.log10(fvi_nonzero)
            data_for_plot.append(log_fvi)
            labels.append(label.replace('Level ', 'L').replace(': ', '\n'))

    # Create violin plot
    parts = ax.violinplot(data_for_plot, positions=range(len(labels)),
                         showmeans=True, showmedians=True, widths=0.7)

    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        label_key = list(fva_results_dict.keys())[i]
        pc.set_facecolor(FVA_LEVEL_COLORS.get(label_key, '#1f77b4'))
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    # Style the other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1.5)

    # Customize plot
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=FONT_SIZES['tick_label'])
    ax.set_ylabel('log₁₀(FVi)', fontsize=FONT_SIZES['axis_label'] + 2)
    ax.set_title(f'{model_name}: Distribution of Flux Variability (FVi) by Constraint Level',
                fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='y', labelsize=FONT_SIZES['tick_label'])

    # Add legend explaining violin plot elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='Mean'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=FONT_SIZES['legend'])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_biomass_progression(fva_results_dict, biomass_dict, model_name,
                            output_path=None, figsize=(10, 6), show=False):
    """
    Create biomass progression plot for FVA ablation study.

    Parameters
    ----------
    fva_results_dict : dict
        Dictionary with level labels as keys and FVA DataFrames as values
    biomass_dict : dict
        Dictionary with level labels as keys and biomass values as values
    model_name : str
        Name of the model for plot title
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    set_plotting_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Extract level numbers for x-axis
    level_nums = []
    biomass_vals = []
    level_labels = []

    for label in fva_results_dict.keys():
        if 'Level 1' in label:
            level_nums.append(1)
        elif 'Level 2' in label:
            level_nums.append(2)
        elif 'Level 3a' in label:
            level_nums.append(3.1)
        elif 'Level 3b' in label:
            level_nums.append(3.2)
        elif 'Level 3c' in label:
            level_nums.append(3.3)
        elif 'Level 4' in label:
            level_nums.append(4)
        elif 'Level 5' in label:
            level_nums.append(5)

        biomass_vals.append(biomass_dict[label])
        level_labels.append(label)

    # Sort by level number
    sorted_data = sorted(zip(level_nums, biomass_vals, level_labels))
    level_nums, biomass_vals, level_labels = zip(*sorted_data)

    ax.plot(level_nums, biomass_vals, 'o-', linewidth=2, markersize=8, color='steelblue')

    # Annotate points
    for x, y, label in zip(level_nums, biomass_vals, level_labels):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=10)

    ax.set_xlabel('Constraint Level', fontsize=12)
    ax.set_ylabel('Biomass (1/hr)', fontsize=12)
    ax.set_title(f'{model_name}: Biomass Production vs Constraint Complexity', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Custom x-axis labels
    ax.set_xticks(level_nums)
    ax.set_xticklabels([label.replace('Level ', 'L').replace(': ', '\n') for label in level_labels],
                      fontsize=10, rotation=45, ha='right')

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig


def generate_fva_ablation_summary_statistics(fva_results_dict, biomass_dict, output_path=None):
    """
    Generate summary statistics table for FVA ablation study.

    For irreversible models, split reactions (forward and reverse) are automatically
    combined before calculating statistics, ensuring consistent reaction counts across
    baseline and enzyme-constrained models.

    Parameters
    ----------
    fva_results_dict : dict
        Dictionary with level labels as keys and FVA DataFrames as values
    biomass_dict : dict
        Dictionary with level labels as keys and biomass values as values
    output_path : str, optional
        Path to save the summary CSV file

    Returns
    -------
    pandas.DataFrame
        DataFrame with summary statistics, including both original and combined
        reaction counts for comparison
    """
    summary_data = []

    for label, fva_df in fva_results_dict.items():
        fvi = calculate_flux_metrics(fva_df)
        fvr = fvi  # FVi is the same as FVR (absolute difference)
        biomass = biomass_dict[label]

        # Flag reactions with high flux variability range (potential issues)
        high_range_reactions = fvr >= 1900

        # Additional metrics
        zero_flux_reactions = (fvi == 0).sum()
        high_var_reactions = (fvi > 1).sum()

        # Calculate AUC for cumulative FVi distribution
        auc = calculate_cumulative_fvi_auc(fvi)

        summary_data.append({
            'Level': label,
            'Biomass (1/hr)': biomass,
            'N Reactions (original)': len(fva_df),
            'N Reactions (combined)': len(fvi),
            'AUC (Cumulative FVi)': auc,
            'Mean FVi': fvi.mean(),
            'Median FVi': fvi.median(),
            'Std FVi': fvi.std(),
            'Min FVi': fvi.min(),
            'Max FVi': fvi.max(),
            'Q25 FVi': fvi.quantile(0.25),
            'Q75 FVi': fvi.quantile(0.75),
            '% Zero Flux': zero_flux_reactions / len(fvi) * 100,
            '% High Variability (FVi > 1)': high_var_reactions / len(fvi) * 100,
            'Mean FVR': fvr.mean(),
            'Median FVR': fvr.median(),
            'Max FVR': fvr.max(),
            '% High Range (FVR ≥ 1900)': high_range_reactions.sum() / len(fvr) * 100,
            'N High Range Reactions': high_range_reactions.sum(),
            'N Zero Flux': zero_flux_reactions,
            'N High Variability': high_var_reactions
        })

    summary_df = pd.DataFrame(summary_data)

    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        summary_df.to_csv(output_path, index=False)

    return summary_df


def create_fva_ablation_dashboard(fva_results_dict, biomass_dict, model_name,
                                 output_dir, prefix="fva_ablation", show=False):
    """
    Create a complete FVA ablation analysis dashboard with multiple plots.

    Parameters
    ----------
    fva_results_dict : dict
        Dictionary with level labels as keys and FVA DataFrames as values
    biomass_dict : dict
        Dictionary with level labels as keys and biomass values as values
    model_name : str
        Name of the model for plot titles
    output_dir : str
        Directory to save output files
    prefix : str, optional
        Prefix for output filenames
    show : bool, optional
        Whether to display the plots

    Returns
    -------
    dict
        Dictionary of generated plot figures and summary statistics
    """
    ensure_dir_exists(output_dir)

    results = {}

    # 1. Enhanced cumulative plot
    cumulative_fig = plot_fva_ablation_cumulative(
        fva_results_dict, biomass_dict, model_name,
        output_path=os.path.join(output_dir, f'{prefix}_cumulative.png'),
        enhanced=True, show=show
    )
    results['cumulative_plot'] = cumulative_fig

    # 2. Box plot of FVi distributions
    boxplot_fig = plot_fva_ablation_boxplot(
        fva_results_dict, model_name,
        output_path=os.path.join(output_dir, f'{prefix}_boxplot.png'),
        show=show
    )
    results['boxplot'] = boxplot_fig

    # 3. Violin plot of FVi distributions
    violin_fig = plot_fva_ablation_violin(
        fva_results_dict, model_name,
        output_path=os.path.join(output_dir, f'{prefix}_violinplot.png'),
        show=show
    )
    results['violinplot'] = violin_fig

    # 4. Biomass progression plot
    biomass_fig = plot_biomass_progression(
        fva_results_dict, biomass_dict, model_name,
        output_path=os.path.join(output_dir, f'{prefix}_biomass_progression.png'),
        show=show
    )
    results['biomass_plot'] = biomass_fig

    # 5. Summary statistics
    summary_df = generate_fva_ablation_summary_statistics(
        fva_results_dict, biomass_dict,
        output_path=os.path.join(output_dir, f'{prefix}_summary.csv')
    )
    results['summary_statistics'] = summary_df

    # Print summary
    print("\n=== FVA Ablation Dashboard Generated ===")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print(f"  - {prefix}_cumulative.png (enhanced plot)")
    print(f"  - {prefix}_boxplot.png (distribution analysis)")
    print(f"  - {prefix}_violinplot.png (distribution shape visualization)")
    print(f"  - {prefix}_biomass_progression.png (biomass vs constraints)")
    print(f"  - {prefix}_summary.csv (summary statistics)")

    return results

def plot_cumulative_fvi_distribution(fva_dataframes, labels, output_path=None,
                                    figsize=(12, 8), show=False, title=None,
                                    legend_position='upper left'):
    """
    Plot cumulative distributions of Flux Variability (FVi) for multiple FVA result sets.

    This creates the standard FVi cumulative distribution plot used throughout kinGEMs,
    with proper styling, horizontal reference line at 0.5, and biomass information.

    For irreversible models, split reactions (forward and reverse) are automatically
    combined before plotting, ensuring fair comparison across baseline and enzyme-
    constrained models.

    Parameters
    ----------
    fva_dataframes : list of pandas.DataFrame
        List of FVA result dataframes, each containing 'Min Solutions', 'Max Solutions',
        and optionally 'Solution Biomass' columns
    labels : list of str
        Labels corresponding to each dataframe
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    title : str, optional
        Custom title for the plot
    legend_position : str, optional
        Position for the legend

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    set_plotting_style()

    fig, ax1 = plt.subplots(figsize=figsize, dpi=300)
    ax2 = ax1.twinx()

    # Define colors (cycling through if more datasets than colors)
    colors = plt.cm.tab10.colors

    fvi_at_0_5 = []

    for i, (df, label) in enumerate(zip(fva_dataframes, labels)):
        # Calculate FVi using standard method
        fvi = calculate_flux_metrics(df)

        # Filter out invalid values
        fvi_values = fvi.values
        fvi_values = fvi_values[~np.isnan(fvi_values)]
        fvi_values = fvi_values[fvi_values >= 1e-15]  # Avoid log(0)

        if len(fvi_values) == 0:
            print(f"Warning: No valid FVi values for {label}")
            continue

        # Sort and create cumulative distribution
        sorted_fvi = np.sort(fvi_values)
        cumulative = np.arange(1, len(sorted_fvi) + 1) / len(sorted_fvi)

        # Plot cumulative distribution
        color = colors[i % len(colors)]
        ax1.plot(sorted_fvi, cumulative, label=label, linewidth=2.5, color=color)

        # Get biomass value and plot reference line
        if 'Solution Biomass' in df.columns:
            biomass_value = df['Solution Biomass'].iloc[0]
            ax2.plot([fvi_values.min(), fvi_values.max()],
                    [biomass_value, biomass_value],
                    linestyle='--', color=color, linewidth=2, alpha=0.6)

        # Calculate FVi at 50th percentile
        fvi_50 = np.interp(0.5, cumulative, sorted_fvi) if len(sorted_fvi) > 0 else np.nan
        fvi_at_0_5.append((label, fvi_50))

    # Add horizontal reference line at cumulative probability 0.5
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Format main plot
    ax1.set_xscale('log')
    ax1.set_xlim(1e-6, 1e3)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Flux Variability (FVi)', fontsize=14)
    ax1.set_ylabel('Cumulative Probability', fontsize=14)
    ax2.set_ylabel('Biomass (1/hr)', fontsize=14)

    # Set title
    if title is None:
        title = 'Cumulative Flux Variability Distribution'
    ax1.set_title(title, fontsize=16, fontweight='bold')

    # Configure ticks and grid
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax1.grid(True, alpha=0.3)

    # Add legend
    ax1.legend(loc=legend_position, fontsize=12, framealpha=0.9)

    plt.tight_layout()

    # Print summary statistics
    print("\nFVi at cumulative probability = 0.5:")
    for label, val in fvi_at_0_5:
        print(f"  {label}: {val:.4f}")

    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig


# ============================================================================
# FLUXOMICS VALIDATION PLOTS
# ============================================================================

def _safe_ensure_dir_for_file(path: str | None) -> None:
    """Ensure parent directory exists for a file path (no-op for None or cwd)."""
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        ensure_dir_exists(out_dir)


def _order_models(models_data: dict[str, pd.DataFrame]):
    """
    Map raw model names to display labels and return an ordered list of (label, df).
    Keeps a preferred order when present, then appends any extras.
    """
    model_label_map = {
        'COBRA FVA': 'Baseline GEM',
        'kinGEMs FVA (pre-tuning)': 'kinGEMs (Pre-Tuning)',
        'kinGEMs FVA': 'kinGEMs (Post-Tuning)',
    }
    model_color_map = {
        'Baseline GEM': '#1f77b4',
        'kinGEMs (Pre-Tuning)': '#8c564b',
        'kinGEMs (Post-Tuning)': '#e377c2',
    }
    preferred = ['Baseline GEM', 'kinGEMs (Pre-Tuning)', 'kinGEMs (Post-Tuning)']

    # Map keys -> labels
    mapped_items = []
    for raw_name, df in models_data.items():
        mapped_items.append((model_label_map.get(raw_name, raw_name), raw_name, df))

    # Build order: preferred first (if present), then the rest (stable)
    by_label = {label: (label, raw, df) for (label, raw, df) in mapped_items}
    ordered = []
    for p in preferred:
        if p in by_label:
            ordered.append((by_label[p][0], by_label[p][2]))

    extras = [(label, df) for (label, raw, df) in mapped_items if label not in preferred]
    ordered.extend(extras)

    return ordered, model_color_map


def plot_fva_mfa_comparison(
    models_data: dict[str, pd.DataFrame],
    split_charts: bool = True,
    reactions_per_plot: int = 25,
    output_path: str | None = None,
    show: bool = True
) -> None:
    """
    Compare MFA experimental ranges against FVA prediction ranges for multiple models.
    Expects each df to contain: rxn_id, mfa_lb, mfa_ub, fva_lb, fva_ub (mfa_* can be NaN for some models).
    """
    set_plotting_style()

    if not models_data:
        raise ValueError("models_data is empty")

    # master list from first df
    primary_name = 'COBRA FVA' if 'COBRA FVA' in models_data else next(iter(models_data))
    master_df = models_data[primary_name].copy()


    required_master = {'rxn_id', 'mfa_lb', 'mfa_ub'}
    missing = required_master - set(master_df.columns)
    if missing:
        raise ValueError(f"Master dataframe missing columns: {missing}")

    master_df = master_df.dropna(subset=['mfa_lb', 'mfa_ub'])
    all_reactions = master_df['rxn_id'].unique()

    if len(all_reactions) == 0:
        raise ValueError(
            "No reactions left after dropping NaNs for MFA bounds. "
            "Check that your comparison df has non-null mfa_lb/mfa_ub for the selected rxns."
        )


    ordered_models, model_color_map = _order_models(models_data)
    num_models = len(ordered_models)

    total_bars = num_models + 1  # +1 for MFA
    step_size = 0.7 / total_bars
    mfa_offset = -0.35 + step_size / 2
    model_offsets = [mfa_offset + (i + 1) * step_size for i in range(num_models)]

    # Chunking
    chunks = [all_reactions]
    if split_charts and len(all_reactions) > reactions_per_plot:
        chunks = [all_reactions[i:i + reactions_per_plot]
                  for i in range(0, len(all_reactions), reactions_per_plot)]
        print(f"Splitting visualization into {len(chunks)} plots for readability.")

    for chunk_idx, rxn_chunk in enumerate(chunks):
        fig_height = len(rxn_chunk) * 0.5 + 2
        fig, ax = plt.subplots(figsize=(12, fig_height))

        for i, rxn in enumerate(rxn_chunk):
            y_pos = i

            # MFA range
            row = master_df[master_df['rxn_id'] == rxn]
            if row.empty:
                continue
            row0 = row.iloc[0]
            mfa_lb = row0['mfa_lb']
            mfa_ub = row0['mfa_ub']

            ax.hlines(
                y=y_pos + mfa_offset,
                xmin=mfa_lb,
                xmax=mfa_ub,
                color='black',
                linewidth=4,
                alpha=0.85,
                zorder=10,
                label='MFA' if i == 0 else ""
            )

            # FVA ranges for each model
            for model_idx, (label, df) in enumerate(ordered_models):
                if 'rxn_id' not in df.columns or 'fva_lb' not in df.columns or 'fva_ub' not in df.columns:
                    continue
                model_row = df[df['rxn_id'] == rxn]
                if model_row.empty:
                    continue

                fva_lb = model_row.iloc[0]['fva_lb']
                fva_ub = model_row.iloc[0]['fva_ub']
                y_offset = y_pos + model_offsets[model_idx]

                ax.hlines(
                    y=y_offset,
                    xmin=fva_lb,
                    xmax=fva_ub,
                    color=model_color_map.get(label, '#777777'),
                    linewidth=4,
                    alpha=0.85,
                    label=label if i == 0 else ""
                )

        # Formatting
        ax.set_yticks(range(len(rxn_chunk)))
        ax.set_yticklabels(rxn_chunk, fontsize=FONT_SIZES['tick_label'], fontfamily='monospace')
        ax.invert_yaxis()
        ax.set_xlabel('Flux (mmol/gDW/h)', fontsize=FONT_SIZES['axis_label'])
        ax.set_title('MFA vs FVA Range Comparison', fontsize=FONT_SIZES['title'])
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        # Custom legend (no duplicates)
        handles = [mlines.Line2D([], [], color='black', linewidth=4, label='MFA')]
        for label, _ in ordered_models:
            handles.append(
                mlines.Line2D([], [], color=model_color_map.get(label, '#777777'), linewidth=4, label=label)
            )
        ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=FONT_SIZES['legend'])

        plt.tight_layout()

        # Save
        if output_path:
            base, ext = os.path.splitext(output_path)
            ext = ext or ".png"
            chunk_path = f"{base}_part{chunk_idx+1:02d}{ext}" if len(chunks) > 1 else output_path
            _safe_ensure_dir_for_file(chunk_path)
            fig.savefig(chunk_path, dpi=DEFAULT_DPI, bbox_inches="tight")  # <-- use fig.savefig

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_fva_mfa_comparison_normalized(
    models_data: dict[str, pd.DataFrame],
    split_charts: bool = True,
    reactions_per_plot: int = 25,
    zoom_limit: float = 5.0,
    output_path: str | None = None,
    show: bool = True
) -> None:
    """
    Normalized view centered on MFA midpoint.
    x=0 is MFA midpoint, MFA always spans [-0.5, 0.5], FVA scaled by MFA width.
    """
    set_plotting_style()

    if not models_data:
        raise ValueError("models_data is empty")

    primary_name = 'COBRA FVA' if 'COBRA FVA' in models_data else next(iter(models_data))
    master_df = models_data[primary_name].copy()


    required_master = {'rxn_id', 'mfa_lb', 'mfa_ub'}
    missing = required_master - set(master_df.columns)
    if missing:
        raise ValueError(f"Master dataframe missing columns: {missing}")

    master_df = master_df.dropna(subset=['mfa_lb', 'mfa_ub'])
    all_reactions = master_df['rxn_id'].unique()

    # Normalization factors per rxn
    normalization_factors = {}
    for rxn in all_reactions:
        row = master_df[master_df['rxn_id'] == rxn]
        if row.empty:
            continue
        row0 = row.iloc[0]
        mfa_lb = float(row0['mfa_lb'])
        mfa_ub = float(row0['mfa_ub'])
        mfa_midpoint = (mfa_lb + mfa_ub) / 2.0
        mfa_range = (mfa_ub - mfa_lb)

        # Avoid explosion if MFA range ~ 0
        norm_factor = mfa_range if abs(mfa_range) > 1e-6 else 1.0
        normalization_factors[rxn] = {'factor': norm_factor, 'offset': mfa_midpoint}

    ordered_models, model_color_map = _order_models(models_data)
    num_models = len(ordered_models)

    total_bars = num_models + 1
    step_size = 0.8 / total_bars
    mfa_offset = -0.4 + step_size / 2
    model_offsets = [mfa_offset + (i + 1) * step_size for i in range(num_models)]

    # Chunking
    chunks = [all_reactions]
    if split_charts and len(all_reactions) > reactions_per_plot:
        chunks = [all_reactions[i:i + reactions_per_plot]
                  for i in range(0, len(all_reactions), reactions_per_plot)]
        print(f"Splitting visualization into {len(chunks)} plots.")

    for chunk_idx, rxn_chunk in enumerate(chunks):
        fig_height = len(rxn_chunk) * 0.5 + 2
        fig, ax = plt.subplots(figsize=(12, fig_height))

        for i, rxn in enumerate(rxn_chunk):
            if rxn not in normalization_factors:
                continue

            y_pos = i
            norm_factor = normalization_factors[rxn]['factor']
            norm_offset = normalization_factors[rxn]['offset']

            # MFA reference always [-0.5, 0.5]
            ax.hlines(
                y=y_pos + mfa_offset,
                xmin=-0.5,
                xmax=0.5,
                color='black',
                linewidth=5,
                alpha=0.9,
                zorder=20,
                label='MFA Range' if i == 0 else ""
            )
            ax.plot(0, y_pos + mfa_offset, '|', color='white', markersize=10, zorder=21)

            # Models (FVA normalized)
            for model_idx, (label, df) in enumerate(ordered_models):
                if 'rxn_id' not in df.columns or 'fva_lb' not in df.columns or 'fva_ub' not in df.columns:
                    continue
                model_row = df[df['rxn_id'] == rxn]
                if model_row.empty:
                    continue

                fva_lb = float(model_row.iloc[0]['fva_lb'])
                fva_ub = float(model_row.iloc[0]['fva_ub'])

                fva_lb_norm = (fva_lb - norm_offset) / norm_factor
                fva_ub_norm = (fva_ub - norm_offset) / norm_factor

                # Clamp drawn segment to avoid huge off-screen lines
                draw_min = max(fva_lb_norm, -zoom_limit * 1.5)
                draw_max = min(fva_ub_norm,  zoom_limit * 1.5)

                y_offset = y_pos + model_offsets[model_idx]
                ax.hlines(
                    y=y_offset,
                    xmin=draw_min,
                    xmax=draw_max,
                    color=model_color_map.get(label, '#777777'),
                    linewidth=4,
                    alpha=0.75,
                    label=label if i == 0 else ""
                )

        # Formatting
        ax.set_xlim(-zoom_limit, zoom_limit)
        ax.set_yticks(range(len(rxn_chunk)))
        ax.set_yticklabels(rxn_chunk, fontsize=FONT_SIZES['tick_label'], fontfamily='monospace')
        ax.invert_yaxis()
        ax.set_xlabel('Normalized Deviation (Units of MFA Range Width)', fontsize=FONT_SIZES['axis_label'])
        ax.set_title('MFA vs FVA: Normalized Deviations', fontsize=FONT_SIZES['title'])
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        handles = [mlines.Line2D([], [], color='black', linewidth=5, label='MFA Range')]
        for label, _ in ordered_models:
            handles.append(mlines.Line2D([], [], color=model_color_map.get(label, '#777777'), linewidth=4, label=label))
        ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=FONT_SIZES['legend'])

        plt.tight_layout()

        # Save
        if output_path:
            base, ext = os.path.splitext(output_path)
            ext = ext or ".png"
            chunk_path = f"{base}_part{chunk_idx+1:02d}{ext}" if len(chunks) > 1 else output_path
            _safe_ensure_dir_for_file(chunk_path)
            plt.savefig(chunk_path, dpi=DEFAULT_DPI, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

def plot_fva_mfa_comparison_zoom_with_jaccard_table(
    models_data: dict[str, pd.DataFrame],
    rxn_ids: list[str] | None = None,
    output_path: str | None = None,
    show: bool = True
) -> None:
    """
    Zoomed MFA vs FVA range comparison for a small set of reactions.

    Adds I (intersection) and U (union) numeric labels to the right of each
    Baseline/kinGEMs bar (not MFA), where I/U are computed between MFA interval
    and the model's FVA interval.
    """
    set_plotting_style()

    if not models_data:
        raise ValueError("models_data is empty")

    if rxn_ids is None:
        rxn_ids = ["EX_ac_e", "PPCK", "ME1"]

    primary_name = 'COBRA FVA' if 'COBRA FVA' in models_data else next(iter(models_data))
    master_df = models_data[primary_name].copy()

    required_master = {'rxn_id', 'mfa_lb', 'mfa_ub'}
    missing = required_master - set(master_df.columns)
    if missing:
        raise ValueError(f"Master dataframe missing columns: {missing}")

    master_df = master_df.dropna(subset=['mfa_lb', 'mfa_ub'])
    present = set(master_df['rxn_id'].astype(str).unique())
    rxn_ids = [r for r in rxn_ids if r in present]
    if len(rxn_ids) == 0:
        raise ValueError("None of the requested rxn_ids were found with valid MFA bounds in master_df.")

    ordered_models, model_color_map = _order_models(models_data)
    band_height = 1.35  # >1 = more space between reactions


    # ---- Layout: each reaction gets its own "band" from y=i to y=i+1 ----
    # Offsets relative to reaction center (top → bottom)
    band_offsets = [0.30, 0.10, -0.10, -0.30]


    fig_height = max(4.5, len(rxn_ids) * 1.1 + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    lw_mfa = 9
    lw_fva = 9

    # ---- I/U annotation styling ----
    # Two colors for the values (pick any you like; these are readable on white)
    inter_color = "#2ca02c"  # green
    union_color = "#d62728"  # red

    # Put the two labels slightly separated in x so they don't collide
    # These are in "data" units, so small values may need smaller offsets.
    # We'll compute offsets relative to the overall x-range later.
    # Font size: reuse your annotation size but a bit smaller usually looks better
    iu_fontsize = max(8, FONT_SIZES.get("annotation", 10) - 1)

    def _interval_intersection_union(a_lb, a_ub, b_lb, b_ub):
        """Return (intersection_length, union_length) for 1D closed intervals."""
        inter = max(0.0, min(a_ub, b_ub) - max(a_lb, b_lb))
        union = max(0.0, max(a_ub, b_ub) - min(a_lb, b_lb))
        return inter, union

    xmins, xmaxs = [], []

    # We'll store pending text placements until we know xlim (for consistent offsets)
    pending_labels = []  # list of dicts with keys: x_end, y, inter, union

    for i, rxn in enumerate(rxn_ids):
        row = master_df[master_df['rxn_id'] == rxn]
        if row.empty:
            continue
        row0 = row.iloc[0]
        mfa_lb = float(row0['mfa_lb'])
        mfa_ub = float(row0['mfa_ub'])
        xmins.append(mfa_lb); xmaxs.append(mfa_ub)

        y_center = i * band_height + 0.5 * band_height

        # MFA
        ax.hlines(
            y=y_center + band_offsets[0],
            xmin=mfa_lb,
            xmax=mfa_ub,
            color='black',
            linewidth=lw_mfa,
            capstyle='round',
            alpha=0.9,
            zorder=10,
            label='MFA' if i == 0 else ""
        )

        # FVA for models
        for model_idx, (label, df) in enumerate(ordered_models):
            if not {'rxn_id', 'fva_lb', 'fva_ub'}.issubset(df.columns):
                continue
            model_row = df[df['rxn_id'] == rxn]
            if model_row.empty:
                continue

            fva_lb = float(model_row.iloc[0]['fva_lb'])
            fva_ub = float(model_row.iloc[0]['fva_ub'])
            xmins.append(fva_lb); xmaxs.append(fva_ub)

            pos_idx = min(model_idx + 1, len(band_offsets) - 1)
            y_line = y_center + band_offsets[pos_idx]


            ax.hlines(
                y=y_line,
                xmin=fva_lb,
                xmax=fva_ub,
                color=model_color_map.get(label, '#777777'),
                linewidth=lw_fva,
                capstyle='round',
                alpha=0.9,
                label=label if i == 0 else ""
            )

            # ---- Compute I/U between MFA and this model's FVA interval ----
            inter, union = _interval_intersection_union(mfa_lb, mfa_ub, fva_lb, fva_ub)

            # Only label Baseline/Pre/Post bars (skip MFA; also skip any unexpected extra models)
            # If you WANT to label every model, just delete this if-block.
            wanted = {"Baseline GEM", "kinGEMs (Pre-Tuning)", "kinGEMs (Post-Tuning)"}
            if label in wanted:
                pending_labels.append({
                    "x_end": fva_ub,
                    "y": y_line,
                    "inter": inter,
                    "union": union
                })

    # ---- Y formatting ----
    ax.set_yticks([i * band_height + 0.5 * band_height for i in range(len(rxn_ids))])
    ax.set_yticklabels(rxn_ids, fontsize=FONT_SIZES['tick_label'], fontfamily='monospace')
    ax.set_ylim(0, len(rxn_ids) * band_height)
    ax.invert_yaxis()

    for i in range(len(rxn_ids) + 1):
        ax.axhline(i * band_height, color='black', linewidth=1, alpha=0.18, zorder=0)

    # ---- X formatting ----
    ax.set_xlabel('Flux (mmol/gDW/h)', fontsize=FONT_SIZES['axis_label'])
    ax.set_title('MFA vs FVA Range Comparison', fontsize=FONT_SIZES['title'])
    ax.grid(False)

    # xlim start at 0, pad on right
    if xmaxs:
        xmax = max(xmaxs)
        pad_data = 0.08 * xmax if xmax > 0 else 1.0
        pad_text = 0.12 * xmax
        ax.set_xlim(0, xmax + pad_data + pad_text)

    # ---- Now place the I/U labels with consistent *pixel* spacing ----
    # This avoids weird spacing when x-range is huge/small.
    for item in pending_labels:
        x_end = item["x_end"]
        y = item["y"]
        inter = item["inter"]
        union = item["union"]

        # Build a single packed label: "I=...  U=..."
        t1 = TextArea(f"I={inter:.3g}", textprops=dict(color=inter_color, fontsize=iu_fontsize))
        t2 = TextArea(f"  U={union:.3g}", textprops=dict(color=union_color, fontsize=iu_fontsize))
        packed = HPacker(children=[t1, t2], align="center", pad=0, sep=2)

        # Place label just to the right of bar end with a small fixed pixel offset
        ab = AnnotationBbox(
            packed,
            (x_end, y),               # anchor at the bar end
            xybox=(8, 0),             # 8 px to the right, centered vertically
            xycoords="data",
            boxcoords="offset points",
            frameon=False,
            box_alignment=(0, 0.5),   # left-middle of the packed text box
            zorder=30
        )
        ax.add_artist(ab)


    # ---- Main legend (ranges) ----
    handles = [mlines.Line2D([], [], color='black', linewidth=lw_mfa, label='MFA')]
    for label, _ in ordered_models:
        handles.append(
            mlines.Line2D([], [], color=model_color_map.get(label, '#777777'),
                          linewidth=lw_fva, label=label)
        )
    leg1 = ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=FONT_SIZES['legend'])

    # ---- Bottom-right legend for I/U label colors ----
    iu_handles = [
        mlines.Line2D([], [], color=inter_color, linewidth=6, label="I = intersection"),
        mlines.Line2D([], [], color=union_color, linewidth=6, label="U = union"),
    ]
    leg2 = ax.legend(handles=iu_handles, loc="lower right", frameon=True, fontsize=FONT_SIZES['legend'])
    ax.add_artist(leg1)  # keep both legends

    plt.tight_layout()

    if output_path:
        _safe_ensure_dir_for_file(output_path)
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_fva_mfa_comparison_zoom(
    models_data: dict[str, pd.DataFrame],
    rxn_ids: list[str] | None = None,
    output_path: str | None = None,
    show: bool = True,
    fig_w: float = 7.0,        # narrower for side-by-side
    fig_h: float = 8.0,        # match your stacked plot height
    show_legend: bool = True,  # set False to remove legend completely
) -> None:
    """
    Zoomed MFA vs FVA range comparison for a small set of reactions.

    Paper-friendly version:
      - No intersection/union annotations
      - No I/U legend
      - Optional: remove model legend entirely
      - More compact figure width for side-by-side layouts
    """
    set_plotting_style()

    if not models_data:
        raise ValueError("models_data is empty")

    if rxn_ids is None:
        rxn_ids = ["EX_ac_e", "PPCK", "ME1"]

    primary_name = 'COBRA FVA' if 'COBRA FVA' in models_data else next(iter(models_data))
    master_df = models_data[primary_name].copy()

    required_master = {'rxn_id', 'mfa_lb', 'mfa_ub'}
    missing = required_master - set(master_df.columns)
    if missing:
        raise ValueError(f"Master dataframe missing columns: {missing}")

    master_df = master_df.dropna(subset=['mfa_lb', 'mfa_ub'])
    present = set(master_df['rxn_id'].astype(str).unique())
    rxn_ids = [r for r in rxn_ids if r in present]
    if len(rxn_ids) == 0:
        raise ValueError("None of the requested rxn_ids were found with valid MFA bounds in master_df.")

    ordered_models, model_color_map = _order_models(models_data)

    # spacing between reactions
    band_height = 1.35  # >1 = more space between reactions
    # Offsets relative to reaction center (top → bottom)
    band_offsets = [0.30, 0.10, -0.10, -0.30]

    # Fixed paper-friendly size
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    lw_mfa = 9
    lw_fva = 9

    xmins, xmaxs = [], []

    for i, rxn in enumerate(rxn_ids):
        row = master_df[master_df['rxn_id'] == rxn]
        if row.empty:
            continue
        row0 = row.iloc[0]
        mfa_lb = float(row0['mfa_lb'])
        mfa_ub = float(row0['mfa_ub'])
        xmins.append(mfa_lb); xmaxs.append(mfa_ub)

        y_center = i * band_height + 0.5 * band_height

        # MFA
        ax.hlines(
            y=y_center + band_offsets[0],
            xmin=mfa_lb,
            xmax=mfa_ub,
            color='black',
            linewidth=lw_mfa,
            capstyle='round',
            alpha=0.9,
            zorder=10,
            label='MFA' if (i == 0 and show_legend) else ""
        )

        # FVA for models
        for model_idx, (label, df) in enumerate(ordered_models):
            if not {'rxn_id', 'fva_lb', 'fva_ub'}.issubset(df.columns):
                continue
            model_row = df[df['rxn_id'] == rxn]
            if model_row.empty:
                continue

            fva_lb = float(model_row.iloc[0]['fva_lb'])
            fva_ub = float(model_row.iloc[0]['fva_ub'])
            xmins.append(fva_lb); xmaxs.append(fva_ub)

            pos_idx = min(model_idx + 1, len(band_offsets) - 1)
            y_line = y_center + band_offsets[pos_idx]

            ax.hlines(
                y=y_line,
                xmin=fva_lb,
                xmax=fva_ub,
                color=model_color_map.get(label, '#777777'),
                linewidth=lw_fva,
                capstyle='round',
                alpha=0.9,
                label=label if (i == 0 and show_legend) else ""
            )

    # ---- Y formatting ----
    ax.set_yticks([i * band_height + 0.5 * band_height for i in range(len(rxn_ids))])
    ax.set_yticklabels(rxn_ids, fontsize=FONT_SIZES['tick_label'], fontfamily='monospace')
    ax.set_ylim(0, len(rxn_ids) * band_height)
    ax.invert_yaxis()

    # separators
    for i in range(len(rxn_ids) + 1):
        ax.axhline(i * band_height, color='black', linewidth=1, alpha=0.18, zorder=0)

    # ---- X formatting ----
    ax.set_xlabel('Flux (mmol/gDW/h)', fontsize=FONT_SIZES['axis_label'])
    ax.set_title('MFA vs FVA Range Comparison', fontsize=FONT_SIZES['title'])
    ax.grid(False)

    # xlim start at 0, modest right pad (no extra pad for text now)
    if xmaxs:
        xmax = max(xmaxs)
        pad = 0.06 * xmax if xmax > 0 else 1.0
        ax.set_xlim(0, xmax + pad)

    # ---- Optional legend (ranges only) ----
    if show_legend:
        handles = [mlines.Line2D([], [], color='black', linewidth=lw_mfa, label='MFA')]
        for label, _ in ordered_models:
            handles.append(
                mlines.Line2D(
                    [], [], color=model_color_map.get(label, '#777777'),
                    linewidth=lw_fva, label=label
                )
            )
        ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=FONT_SIZES['legend'])

    plt.tight_layout()

    if output_path:
        _safe_ensure_dir_for_file(output_path)
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)




def plot_jaccard_index_comparison(
    jaccard_indices: list[float],
    zero_overlaps: list[int],
    model_names: list[str] | None = None,
    output_path: str | None = None,
    show: bool = True,
    n_total: list[int] | None = None
) -> None:
    set_plotting_style()

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(jaccard_indices))]

    if not (len(jaccard_indices) == len(zero_overlaps) == len(model_names)):
        raise ValueError("jaccard_indices, zero_overlaps, and model_names must have same length")

    if n_total is not None and len(n_total) != len(model_names):
        raise ValueError("n_total must be None or have the same length as model_names")

    model_label_map = {
        "COBRA FVA": "Baseline GEM",
        "kinGEMs FVA (pre-tuning)": "kinGEMs (Pre-Tuning)",
        "kinGEMs FVA": "kinGEMs (Post-Tuning)",
        "COBRA FVA (pre-tuning)": "Baseline GEM (Pre-Tuning Cobra)",
    }
    model_color_map = {
        "Baseline GEM": "#1f77b4",
        "Baseline GEM (Pre-Tuning Cobra)": "#1f77b4",
        "kinGEMs (Pre-Tuning)": "#8c564b",
        "kinGEMs (Post-Tuning)": "#e377c2",
    }

    mapped_names = [model_label_map.get(n, n) for n in model_names]

    # ---- Deduplicate labels after mapping (keep first occurrence) ----
    seen = set()
    keep_idx = []
    for i, name in enumerate(mapped_names):
        if name in seen:
            continue
        seen.add(name)
        keep_idx.append(i)

    if len(keep_idx) != len(mapped_names):
        dropped = [mapped_names[i] for i in range(len(mapped_names)) if i not in keep_idx]
        print(
            "[plot_jaccard_index_comparison] Warning: duplicate model labels after mapping. "
            f"Dropping duplicates: {dropped}"
        )

    mapped_names = [mapped_names[i] for i in keep_idx]
    jaccard_indices = [float(jaccard_indices[i]) for i in keep_idx]
    zero_overlaps = [int(zero_overlaps[i]) for i in keep_idx]
    if n_total is not None:
        n_total = [int(n_total[i]) for i in keep_idx]

    # ---- Enforce canonical left-to-right order ----
    desired_order = ["Baseline GEM", "kinGEMs (Pre-Tuning)", "kinGEMs (Post-Tuning)"]
    order_index = {name: i for i, name in enumerate(desired_order)}
    sort_idx = sorted(range(len(mapped_names)), key=lambda i: order_index.get(mapped_names[i], 999))

    mapped_names = [mapped_names[i] for i in sort_idx]
    jaccard_indices = [jaccard_indices[i] for i in sort_idx]
    zero_overlaps = [zero_overlaps[i] for i in sort_idx]
    if n_total is not None:
        n_total = [n_total[i] for i in sort_idx]

    # Colors
    default_colors = ["#1f77b4", "#8c564b", "#e377c2", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
    bar_colors = [
        model_color_map.get(name, default_colors[i % len(default_colors)])
        for i, name in enumerate(mapped_names)
    ]

    x = np.arange(len(mapped_names))

    # ---- Layout: equal-sized subplots ----
    fig_w = max(9, 1.9 * len(mapped_names) + 6)
    fig_h = 6.2
    fig, (ax_j, ax_z) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        sharex=True  # ensures identical x scaling & bar spacing
    )

    bar_w = 0.55

    # Left: mean Jaccard
    bars_j = ax_j.bar(
        x, jaccard_indices, width=bar_w, color=bar_colors, alpha=0.85,
        edgecolor="black", linewidth=1
    )
    ax_j.set_ylabel("Mean Jaccard Index (↑ better)", fontsize=FONT_SIZES["axis_label"])
    ax_j.set_xticks([])
    ax_j.grid(axis="y", linestyle="--", alpha=0.3)

    max_j = max(jaccard_indices) if jaccard_indices else 1.0
    y_max = max(0.05, max_j * 1.25)
    ax_j.set_ylim(0, y_max)

    for bar in bars_j:
        h = bar.get_height()
        ax_j.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["annotation"]
        )

    # Right: zero overlaps
    bars_z = ax_z.bar(
        x, zero_overlaps, width=bar_w, color=bar_colors, alpha=0.85,
        edgecolor="black", linewidth=1
    )
    ax_z.set_ylabel("# Reaction with No Overlap (↓ better)", fontsize=FONT_SIZES["axis_label"])
    ax_z.set_xticks([])
    ax_z.grid(axis="y", linestyle="--", alpha=0.3)

    max_z = max(zero_overlaps) if zero_overlaps else 1
    ax_z.set_ylim(0, max(1, int(max_z * 1.15) + 1))

    for bar in bars_z:
        h = bar.get_height()
        ax_z.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["annotation"]
        )

    # ---- Figure-level title (over both subplots) ----
    title_n = None
    if n_total is not None and len(n_total) > 0 and all(v == n_total[0] for v in n_total):
        title_n = n_total[0]
    fig.suptitle(
        f"Over {title_n} Reactions" if title_n is not None else "Over Reactions",
        fontsize=FONT_SIZES["subtitle"],
        y=0.85  # lower and visually centered
    )


    # Tight layout with room for suptitle; no legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    if output_path:
        _safe_ensure_dir_for_file(output_path)
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_jaccard_index_comparison_stacked(
    jaccard_distributions: list[list[float]] | dict[str, list[float]],
    zero_overlaps: list[int] | dict[str, int],
    model_names: list[str] | None = None,
    output_path: str | None = None,
    show: bool = True,
    n_total: list[int] | None = None,
) -> None:
    set_plotting_style()

    # ----------------------------
    # Normalize inputs to lists aligned with model_names
    # ----------------------------
    if isinstance(jaccard_distributions, dict):
        if model_names is None:
            model_names = list(jaccard_distributions.keys())
        jaccard_dists_list = [list(map(float, jaccard_distributions[m])) for m in model_names]
    else:
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(jaccard_distributions))]
        jaccard_dists_list = [list(map(float, xs)) for xs in jaccard_distributions]

    if isinstance(zero_overlaps, dict):
        zero_overlaps_list = [int(zero_overlaps[m]) for m in model_names]
    else:
        zero_overlaps_list = [int(v) for v in zero_overlaps]

    if not (len(jaccard_dists_list) == len(zero_overlaps_list) == len(model_names)):
        raise ValueError("jaccard_distributions, zero_overlaps, and model_names must have same length")

    if n_total is not None and len(n_total) != len(model_names):
        raise ValueError("n_total must be None or have the same length as model_names")

    # ----------------------------
    # Mapping + dedup + ordering
    # ----------------------------
    model_label_map = {
        "COBRA FVA": "Baseline GEM",
        "kinGEMs FVA (pre-tuning)": "kinGEMs (Pre-Tuning)",
        "kinGEMs FVA": "kinGEMs (Post-Tuning)",
        "COBRA FVA (pre-tuning)": "Baseline GEM (Pre-Tuning Cobra)",
    }
    model_color_map = {
        "Baseline GEM": "#1f77b4",
        "Baseline GEM (Pre-Tuning Cobra)": "#1f77b4",
        "kinGEMs (Pre-Tuning)": "#8c564b",
        "kinGEMs (Post-Tuning)": "#e377c2",
    }

    mapped_names = [model_label_map.get(n, n) for n in model_names]

    # Deduplicate after mapping (keep first)
    seen = set()
    keep_idx = []
    for i, name in enumerate(mapped_names):
        if name in seen:
            continue
        seen.add(name)
        keep_idx.append(i)

    if len(keep_idx) != len(mapped_names):
        dropped = [mapped_names[i] for i in range(len(mapped_names)) if i not in keep_idx]
        print(
            "[plot_jaccard_index_comparison_stacked] Warning: duplicate model labels after mapping. "
            f"Dropping duplicates: {dropped}"
        )

    mapped_names = [mapped_names[i] for i in keep_idx]
    jaccard_dists_list = [jaccard_dists_list[i] for i in keep_idx]
    zero_overlaps_list = [zero_overlaps_list[i] for i in keep_idx]
    if n_total is not None:
        n_total = [int(n_total[i]) for i in keep_idx]

    # Canonical order
    desired_order = ["Baseline GEM", "kinGEMs (Pre-Tuning)", "kinGEMs (Post-Tuning)"]
    order_index = {name: i for i, name in enumerate(desired_order)}
    sort_idx = sorted(range(len(mapped_names)), key=lambda i: order_index.get(mapped_names[i], 999))

    mapped_names = [mapped_names[i] for i in sort_idx]
    jaccard_dists_list = [jaccard_dists_list[i] for i in sort_idx]
    zero_overlaps_list = [zero_overlaps_list[i] for i in sort_idx]
    if n_total is not None:
        n_total = [n_total[i] for i in sort_idx]

    # Colors
    default_colors = ["#1f77b4", "#8c564b", "#e377c2", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
    colors = [
        model_color_map.get(name, default_colors[i % len(default_colors)])
        for i, name in enumerate(mapped_names)
    ]

    # ----------------------------
    # X spacing (bring categories closer)
    # ----------------------------
    x_step = 1.1
    x = np.arange(len(mapped_names)) * x_step
    bar_w = 0.62
    box_w = 0.62

    # ----------------------------
    # Layout: stacked (condensed + more vertical space between panels)
    # ----------------------------
    fig_w = 8.4  # narrower than before
    fig_h = 8.0
    fig, (ax_j, ax_z) = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.30}  # more space between subplots
    )

    # ----------------------------
    # Top: boxplot + jitter markers (points match model colors)
    # ----------------------------
    bp = ax_j.boxplot(
        jaccard_dists_list,
        positions=x,
        widths=box_w,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 1},
        capprops={"color": "black", "linewidth": 1},
        boxprops={"edgecolor": "black", "linewidth": 1},
    )

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.75)

    ax_j.set_ylabel("Jaccard Index (↑ better)", fontsize=FONT_SIZES["axis_label"])
    ax_j.set_yscale("symlog", linthresh=0.01)
    ax_j.grid(axis="y", linestyle="--", alpha=0.25)

    all_vals = [v for sub in jaccard_dists_list for v in sub]
    ymax = max(0.05, (max(all_vals) * 1.15) if all_vals else 0.05)
    ax_j.set_ylim(0, ymax)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(jaccard_dists_list):
        if not vals:
            continue

        jitter = rng.normal(loc=0.0, scale=0.06, size=len(vals))
        xs = np.full(len(vals), x[i]) + jitter

        # points colored per model
        ax_j.scatter(
            xs, vals,
            s=22,
            alpha=0.55,
            color=colors[i],
            edgecolors="none",
            zorder=3
        )

        # place mean label ABOVE the box/whiskers
        mean_v = float(np.mean(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        upper_whisker = min(max(vals), q3 + 1.5 * iqr)
        y_text = min(ymax * 0.98, upper_whisker + 0.02 * ymax)

        ax_j.text(
            x[i], y_text,
            f"{mean_v:.3f}",
            ha="center", va="bottom",
            fontsize=FONT_SIZES["annotation"],
            zorder=6
        )

    # ----------------------------
    # Bottom: bars
    # ----------------------------
    bars_z = ax_z.bar(
        x, zero_overlaps_list,
        width=bar_w,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=1
    )

    ax_z.set_ylabel("# Disjoint Reactions (↓ better)", fontsize=FONT_SIZES["axis_label"])
    ax_z.grid(axis="y", linestyle="--", alpha=0.3)

    max_z = max(zero_overlaps_list) if zero_overlaps_list else 1
    ax_z.set_ylim(0, max(1, int(max_z * 1.15) + 1))

    for bar in bars_z:
        h = bar.get_height()
        ax_z.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["annotation"]
        )

    ax_z.set_xticks(x)
    ax_z.set_xticklabels(mapped_names, fontsize=FONT_SIZES["tick_label"], rotation=0)

    # ----------------------------
    # Figure-level title (lower than before)
    # ----------------------------
    title_n = None
    if n_total is not None and len(n_total) > 0 and all(v == n_total[0] for v in n_total):
        title_n = n_total[0]

    fig.suptitle(
        f"Over {title_n} Reactions" if title_n is not None else "Over Reactions",
        fontsize=FONT_SIZES["subtitle"],
        y=0.93
    )

    # Layout first, then align y-label x-positions exactly across both axes
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fig.canvas.draw()
    ax_j.yaxis.set_label_coords(-0.085, 0.5)
    ax_z.yaxis.set_label_coords(-0.085, 0.5)

    if output_path:
        _safe_ensure_dir_for_file(output_path)
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_mean_to_mean_distance_stacked(
    distance_distributions: list[list[float]] | dict[str, list[float]],
    zero_overlaps: list[int] | dict[str, int],
    model_names: list[str] | None = None,
    output_path: str | None = None,
    show: bool = True,
    n_total: list[int] | None = None,
) -> None:
    """
    Stacked plot: per-reaction mean-to-mean distance (boxplot, top) and
    # Disjoint Reactions (bar chart, bottom).  Mirrors
    plot_jaccard_index_comparison_stacked with the metric swapped.

    Parameters
    ----------
    distance_distributions : list or dict
        Per-reaction |fva_mean - mfa_mean| values for each model.
    zero_overlaps : list or dict
        Number of disjoint (zero-Jaccard) reactions per model.
    model_names : list[str], optional
        Ordered model labels.
    output_path : str, optional
        Save path for the figure.
    show : bool
        Whether to display interactively.
    n_total : list[int], optional
        Total reaction counts used in the figure title.
    """
    set_plotting_style()

    # --- Normalize to lists ---
    if isinstance(distance_distributions, dict):
        if model_names is None:
            model_names = list(distance_distributions.keys())
        dists_list = [list(map(float, distance_distributions[m])) for m in model_names]
    else:
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(distance_distributions))]
        dists_list = [list(map(float, xs)) for xs in distance_distributions]

    if isinstance(zero_overlaps, dict):
        zero_overlaps_list = [int(zero_overlaps[m]) for m in model_names]
    else:
        zero_overlaps_list = [int(v) for v in zero_overlaps]

    if not (len(dists_list) == len(zero_overlaps_list) == len(model_names)):
        raise ValueError("distance_distributions, zero_overlaps, and model_names must have same length")

    if n_total is not None and len(n_total) != len(model_names):
        raise ValueError("n_total must be None or have the same length as model_names")

    # --- Label / color mapping (same convention as Jaccard stacked plot) ---
    model_label_map = {
        "COBRA FVA": "Baseline GEM",
        "kinGEMs FVA (pre-tuning)": "kinGEMs (Pre-Tuning)",
        "kinGEMs FVA": "kinGEMs (Post-Tuning)",
        "COBRA FVA (pre-tuning)": "Baseline GEM (Pre-Tuning Cobra)",
    }
    model_color_map = {
        "Baseline GEM": "#1f77b4",
        "Baseline GEM (Pre-Tuning Cobra)": "#1f77b4",
        "kinGEMs (Pre-Tuning)": "#8c564b",
        "kinGEMs (Post-Tuning)": "#e377c2",
    }

    mapped_names = [model_label_map.get(n, n) for n in model_names]

    # Deduplicate after mapping (keep first occurrence)
    seen = set()
    keep_idx = []
    for i, name in enumerate(mapped_names):
        if name in seen:
            continue
        seen.add(name)
        keep_idx.append(i)

    if len(keep_idx) != len(mapped_names):
        dropped = [mapped_names[i] for i in range(len(mapped_names)) if i not in keep_idx]
        print(
            "[plot_mean_to_mean_distance_stacked] Warning: duplicate model labels after mapping. "
            f"Dropping duplicates: {dropped}"
        )

    mapped_names = [mapped_names[i] for i in keep_idx]
    dists_list = [dists_list[i] for i in keep_idx]
    zero_overlaps_list = [zero_overlaps_list[i] for i in keep_idx]
    if n_total is not None:
        n_total = [int(n_total[i]) for i in keep_idx]

    # Canonical order
    desired_order = ["Baseline GEM", "kinGEMs (Pre-Tuning)", "kinGEMs (Post-Tuning)"]
    order_index = {name: i for i, name in enumerate(desired_order)}
    sort_idx = sorted(range(len(mapped_names)), key=lambda i: order_index.get(mapped_names[i], 999))

    mapped_names = [mapped_names[i] for i in sort_idx]
    dists_list = [dists_list[i] for i in sort_idx]
    zero_overlaps_list = [zero_overlaps_list[i] for i in sort_idx]
    if n_total is not None:
        n_total = [n_total[i] for i in sort_idx]

    default_colors = ["#1f77b4", "#8c564b", "#e377c2", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
    colors = [
        model_color_map.get(name, default_colors[i % len(default_colors)])
        for i, name in enumerate(mapped_names)
    ]

    # --- Layout ---
    x_step = 1.1
    x = np.arange(len(mapped_names)) * x_step
    bar_w = 0.62
    box_w = 0.62

    fig, (ax_d, ax_z) = plt.subplots(
        2, 1,
        figsize=(8.4, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.30}
    )

    # --- Top: boxplot + jitter ---
    bp = ax_d.boxplot(
        dists_list,
        positions=x,
        widths=box_w,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 1},
        capprops={"color": "black", "linewidth": 1},
        boxprops={"edgecolor": "black", "linewidth": 1},
    )

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.75)

    all_vals = [v for sub in dists_list for v in sub]
    ymax = max(0.1, (max(all_vals) * 1.15) if all_vals else 0.1)
    ax_d.set_ylim(0, ymax)
    ax_d.set_ylabel("Mean-to-Mean Distance (↓ better)", fontsize=FONT_SIZES["axis_label"])
    ax_d.grid(axis="y", linestyle="--", alpha=0.25)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(dists_list):
        if not vals:
            continue
        jitter = rng.normal(loc=0.0, scale=0.06, size=len(vals))
        xs = np.full(len(vals), x[i]) + jitter
        ax_d.scatter(xs, vals, s=22, alpha=0.55, color=colors[i], edgecolors="none", zorder=3)

        mean_v = float(np.mean(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        upper_whisker = min(max(vals), q3 + 1.5 * iqr)
        y_text = min(ymax * 0.98, upper_whisker + 0.02 * ymax)
        ax_d.text(
            x[i], y_text, f"{mean_v:.3f}",
            ha="center", va="bottom",
            fontsize=FONT_SIZES["annotation"], zorder=6
        )

    # --- Bottom: # Disjoint Reactions bar chart ---
    bars_z = ax_z.bar(
        x, zero_overlaps_list,
        width=bar_w,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=1
    )

    ax_z.set_ylabel("# Disjoint Reactions (↓ better)", fontsize=FONT_SIZES["axis_label"])
    ax_z.grid(axis="y", linestyle="--", alpha=0.3)

    max_z = max(zero_overlaps_list) if zero_overlaps_list else 1
    ax_z.set_ylim(0, max(1, int(max_z * 1.15) + 1))

    for bar in bars_z:
        h = bar.get_height()
        ax_z.text(
            bar.get_x() + bar.get_width() / 2, h, f"{int(h)}",
            ha="center", va="bottom", fontsize=FONT_SIZES["annotation"]
        )

    ax_z.set_xticks(x)
    ax_z.set_xticklabels(mapped_names, fontsize=FONT_SIZES["tick_label"], rotation=0)

    # --- Title ---
    title_n = None
    if n_total is not None and len(n_total) > 0 and all(v == n_total[0] for v in n_total):
        title_n = n_total[0]

    fig.suptitle(
        f"Over {title_n} Reactions" if title_n is not None else "Over Reactions",
        fontsize=FONT_SIZES["subtitle"],
        y=0.93
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fig.canvas.draw()
    ax_d.yaxis.set_label_coords(-0.085, 0.5)
    ax_z.yaxis.set_label_coords(-0.085, 0.5)

    if output_path:
        _safe_ensure_dir_for_file(output_path)
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_jaccard_index_comparison_overlapping(
    jaccard_dfs: list[pd.DataFrame],
    model_names: list[str] | None = None,
    output_path: str | None = None,
    show: bool = True,
    normalize_by_coverage: bool = False
) -> None:
    """
    Plot bar chart of mean Jaccard index for only overlapping reactions (J > 0).
    Title includes number of overlapping reactions (per model).
    Deduplicates labels AFTER mapping (keeps first).
    Auto-scales y-axis to plotted values.

    Parameters
    ----------
    jaccard_dfs : list[pd.DataFrame]
        List of DataFrames with 'jaccard' column
    model_names : list[str], optional
        Names of models
    output_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    normalize_by_coverage : bool
        If True, normalize Jaccard by coverage (n/n_max) to account for
        fewer reactions having inflated scores
    """
    set_plotting_style()

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(jaccard_dfs))]

    if len(model_names) != len(jaccard_dfs):
        raise ValueError("model_names and jaccard_dfs must have the same length")

    # Compute overlapping-only stats
    mean_jaccards: list[float] = []
    n_overlapping: list[int] = []
    for df in jaccard_dfs:
        if "jaccard" not in df.columns:
            raise ValueError("Each dataframe in jaccard_dfs must contain a 'jaccard' column")
        overlapping = df[df["jaccard"] > 0]
        mean_jaccards.append(float(overlapping["jaccard"].mean()) if not overlapping.empty else 0.0)
        n_overlapping.append(int(len(overlapping)))

    # Normalize by coverage if requested
    if normalize_by_coverage:
        max_n = max(n_overlapping) if n_overlapping else 1
        mean_jaccards = [j * (n / max_n) for j, n in zip(mean_jaccards, n_overlapping)]

    model_label_map = {
        "COBRA FVA": "Baseline GEM",
        "kinGEMs FVA (pre-tuning)": "kinGEMs (Pre-Tuning)",
        "kinGEMs FVA": "kinGEMs (Post-Tuning)",
        # Optional if you add this upstream:
        "COBRA FVA (pre-tuning)": "Baseline GEM (Pre-Tuning Cobra)",
    }
    model_color_map = {
        "Baseline GEM": "#1f77b4",
        "Baseline GEM (Pre-Tuning Cobra)": "#1f77b4",
        "kinGEMs (Pre-Tuning)": "#8c564b",
        "kinGEMs (Post-Tuning)": "#e377c2",
    }

    mapped_names = [model_label_map.get(n, n) for n in model_names]

    # ---- Deduplicate labels after mapping (keep first occurrence) ----
    seen = set()
    keep_idx = []
    for i, name in enumerate(mapped_names):
        if name in seen:
            continue
        seen.add(name)
        keep_idx.append(i)

    if len(keep_idx) != len(mapped_names):
        dropped = [mapped_names[i] for i in range(len(mapped_names)) if i not in keep_idx]
        print(
            "[plot_jaccard_index_comparison_overlapping] Warning: duplicate model labels after mapping. "
            f"Dropping duplicates: {dropped}"
        )

    mapped_names = [mapped_names[i] for i in keep_idx]
    mean_jaccards = [mean_jaccards[i] for i in keep_idx]
    n_overlapping = [n_overlapping[i] for i in keep_idx]

    # Colors
    default_colors = ["#1f77b4", "#8c564b", "#e377c2", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
    bar_colors = [
        model_color_map.get(name, default_colors[i % len(default_colors)])
        for i, name in enumerate(mapped_names)
    ]

    x = np.arange(len(mapped_names))
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_SINGLE)
    bars = ax.bar(x, mean_jaccards, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1)

    ylabel = "Mean Jaccard Index (overlapping only)"
    if normalize_by_coverage:
        ylabel += " × Coverage"
    ylabel += " (↑ better)"
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis_label"])
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(mapped_names, fontsize=FONT_SIZES["tick_label"])

    # ---- Auto-scale y axis to values ----
    max_j = max(mean_jaccards) if mean_jaccards else 1.0
    y_max = max(0.05, max_j * 1.25)  # headroom, avoids a flat axis
    ax.set_ylim(0, y_max)

    # Title with n (overlapping counts)
    n_str = (
        f" (n={n_overlapping[0]})"
        if all(v == n_overlapping[0] for v in n_overlapping)
        else " (n=" + ", ".join(str(v) for v in n_overlapping) + ")"
    )
    title = "Mean Jaccard Index"
    if normalize_by_coverage:
        title += " (Coverage-Normalized)"
    else:
        title += " (Overlapping Only)"
    ax.set_title(f"{title}{n_str}", fontsize=FONT_SIZES["subtitle"])

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["annotation"],
        )

    plt.tight_layout()

    if output_path:
        _safe_ensure_dir_for_file(output_path)
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


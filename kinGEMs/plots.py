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


def plot_fva_ablation_cumulative(fva_results_dict, biomass_dict, model_name,
                                output_path=None, figsize=(12, 8), show=False,
                                legend_position='upper left', enhanced=True):
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

    for label, fva_df in fva_results_dict.items():
        fvi = calculate_flux_metrics(fva_df)
        # Filter out zero values for log plotting
        fvi_nonzero = fvi[fvi > 1e-15]
        if len(fvi_nonzero) == 0:
            fvi_nonzero = np.array([1e-15])

        all_fvi_values.extend(fvi_nonzero)
        fvi_sorted = np.sort(fvi_nonzero)
        cumulative = np.arange(1, len(fvi_sorted) + 1) / len(fvi_sorted)

        ax1.plot(fvi_sorted, cumulative, label=label,
                color=FVA_LEVEL_COLORS.get(label, None), linewidth=2.5)

        # Store stats for bottom plot
        if enhanced:
            fvi_stats[label] = {
                'mean': fvi.mean(),
                'median': fvi.median(),
                'biomass': biomass_dict[label]
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
        ax2.set_ylim(bottom=1e-6)
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

        summary_data.append({
            'Level': label,
            'Biomass (1/hr)': biomass,
            'N Reactions (original)': len(fva_df),
            'N Reactions (combined)': len(fvi),
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




def plot_jaccard_index_comparison_overlapping(
    jaccard_dfs: list[pd.DataFrame],
    model_names: list[str] | None = None,
    output_path: str | None = None,
    show: bool = True
) -> None:
    """
    Plot bar chart of mean Jaccard index for only overlapping reactions (J > 0).
    Title includes number of overlapping reactions (per model).
    Deduplicates labels AFTER mapping (keeps first).
    Auto-scales y-axis to plotted values.
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

    ax.set_ylabel("Mean Jaccard Index (overlapping only) (↑ better)", fontsize=FONT_SIZES["axis_label"])
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
    ax.set_title(f"Mean Jaccard Index (Overlapping Only){n_str}", fontsize=FONT_SIZES["subtitle"])

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

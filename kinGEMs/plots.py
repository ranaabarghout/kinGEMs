"""
Visualization module for kinGEMs.

This module provides functions for visualizing results from kinGEMs analyses,
including flux distributions, enzyme usage, and parameter optimization progress.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

from .config import ensure_dir_exists


def set_plotting_style(style="whitegrid"):
    """
    Set a consistent style for all plots.

    Parameters
    ----------
    style : str, optional
        The seaborn style to use

    Returns
    -------
    None
    """
    sns.set_style(style)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

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

    # Define colors for each level
    colors = {
        'Level 1: Baseline GEM': '#1f77b4',
        'Level 2: Single Enzyme': '#ff7f0e',
        'Level 3a: + Isoenzymes': '#2ca02c',
        'Level 3b: + Complexes': '#d62728',
        'Level 3c: + Promiscuous': '#9467bd',
        'Level 4: All Constraints': '#8c564b',
        'Level 5: Post-Tuned': '#e377c2'
    }

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
                color=colors.get(label, None), linewidth=2.5)

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
    ax1.set_xlim(1e-6, 1e3)
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
        bars = ax2.bar(x_pos, median_fvis, color=[colors.get(label, '#1f77b4') for label in labels], alpha=0.7)

        # Set up left y-axis for Median FVi
        ax2.set_ylabel('Median FVi', fontsize=13)
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=1e-6)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([label.replace('Level ', 'L').replace(': ', '\n') for label in labels],
                           rotation=45, ha='right', fontsize=10)

        # Create second y-axis for biomass
        ax2_biomass = ax2.twinx()
        line = ax2_biomass.plot(x_pos, biomass_values, 'ko-', linewidth=2, markersize=6, label='Biomass')
        ax2_biomass.set_ylabel('Biomass (1/hr)', fontsize=13)
        ax2_biomass.set_ylim(0, max(biomass_values) * 1.1)

        # Add subplot title
        ax2.set_title('Mean Flux Variability and Biomass by Constraint Level', fontsize=12, fontweight='bold')

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

    colors = {
        'Level 1: Baseline GEM': '#1f77b4',
        'Level 2: Single Enzyme': '#ff7f0e',
        'Level 3a: + Isoenzymes': '#2ca02c',
        'Level 3b: + Complexes': '#d62728',
        'Level 3c: + Promiscuous': '#9467bd',
        'Level 4: All Constraints': '#8c564b',
        'Level 5: Post-Tuned': '#e377c2'
    }

    for label, fva_df in fva_results_dict.items():
        fvi = calculate_flux_metrics(fva_df)
        # Use log scale for better visualization, filter zeros
        fvi_nonzero = fvi[fvi > 1e-15]
        if len(fvi_nonzero) > 0:
            fvi_data.append(np.log10(fvi_nonzero))
            labels.append(label.replace('Level ', 'L').replace(': ', '\n'))
            colors_list.append(colors[label])

    bp = ax.boxplot(fvi_data, labels=labels, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('log₁₀(FVi)', fontsize=12)
    ax.set_title(f'{model_name}: Distribution of Flux Variability (FVi) by Constraint Level', fontsize=14)
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
        high_range_reactions = fvr >= 1000

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
            '% High Range (FVR ≥ 1000)': high_range_reactions.sum() / len(fvr) * 100,
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

    # 3. Biomass progression plot
    biomass_fig = plot_biomass_progression(
        fva_results_dict, biomass_dict, model_name,
        output_path=os.path.join(output_dir, f'{prefix}_biomass_progression.png'),
        show=show
    )
    results['biomass_plot'] = biomass_fig

    # 4. Summary statistics
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

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
        

def kingems_cobrapy_dataframe(kingems_path: str, fba_path: str) -> pd.DataFrame:
    """
    Compare fluxomics simulation results
    
    Parameters:
        method1_path : str
            Path to the CSV file with kinGEMs fluxes
            Expected columns: 'Variable', 'Index', 'Value'
        method2_path : str
            Path to the CSV file with COBRApy FBA fluxes
            Expected columns: 'rxn_id', 'flux'
    
    Returns:
        pd.DataFrame: DataFrame with columns: 'rxn_id', 'flux', 'kinGEMs_flux'
    """
    
    # Load FBA results and filter for flux variables
    kingems_df = pd.read_csv(kingems_path)
    
    # Filter for flux variables only
    flux_df = kingems_df[kingems_df['Variable'] == 'flux'].copy()
    
    # Rename columns to match expected output format
    flux_df = flux_df.rename(columns={'Index': 'rxn_id', 'Value': 'kinGEMs_flux'})
    
    # Keep only the columns we need
    flux_df = flux_df[['rxn_id', 'kinGEMs_flux']]
    
    # Load experimental fluxes
    fba_df = pd.read_csv(fba_path)
    
    # Rename columns to match expected output format
    fba_df = fba_df.rename(columns={'Index': 'rxn_id', 'Value': 'cobrapy_flux'})
    
    # Merge the dataframes on rxn_id
    result_df = fba_df.merge(flux_df, on='rxn_id', how='right')
    
    # Reorder columns to match expected output
    result_df = result_df[['rxn_id', 'cobrapy_flux', 'kinGEMs_flux']]
    
    print(f"Loaded {len(flux_df)} kinGEMs fluxes")
    print(f"Loaded {len(fba_df)} COBRApy FBA fluxes")
    print(f"Merged dataframe has {len(result_df)} rows")
    print(f"Matched reactions: {result_df['cobrapy_flux'].notna().sum()}")
    print(f"Unmatched reactions: {result_df['cobrapy_flux'].isna().sum()}")
    
    return result_df


def plot_flux_correlation(df, method1_col, method2_col, rxn_id_col=None, 
                         output_path=None, figsize=(14, 6), show=False):
    """
    Plot correlation analysis between two flux methods with scatter and residual plots.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing flux data from two methods
    method1_col : str
        Name of the first flux column
    method2_col : str
        Name of the second flux column
    rxn_id_col : str, optional
        Name of the reaction ID column (uses first column if None)
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
        
    Returns
    -------
    tuple
        (matplotlib.figure.Figure, dict) - The plot figure and correlation results
    """
    # Set the plotting style
    set_plotting_style()
    
    # Auto-detect reaction ID column if not specified
    if rxn_id_col is None:
        rxn_id_col = df.columns[0]
    
    # Clean data - convert to numeric and remove NaN values
    df_clean = df.copy()
    df_clean[method1_col] = pd.to_numeric(df_clean[method1_col], errors='coerce')
    df_clean[method2_col] = pd.to_numeric(df_clean[method2_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[method1_col, method2_col])
    
    if len(df_clean) == 0:
        raise ValueError("No valid data points for correlation analysis")
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(df_clean[method1_col], df_clean[method2_col])
    spearman_r, spearman_p = spearmanr(df_clean[method1_col], df_clean[method2_col])
    r2 = pearson_r ** 2
    
    # Linear regression
    slope, intercept, _, _, _ = stats.linregress(df_clean[method1_col], df_clean[method2_col])
    
    # Store results
    results = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r_squared': r2,
        'slope': slope,
        'intercept': intercept,
        'n_points': len(df_clean)
    }
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot with regression line
    ax1.scatter(df_clean[method1_col], df_clean[method2_col], 
               alpha=0.7, s=60, edgecolors='black', linewidth=0.5, color='#1f77b4')
    
    # Add regression line
    x_range = np.linspace(df_clean[method1_col].min(), 
                         df_clean[method1_col].max(), 100)
    y_pred = slope * x_range + intercept
    ax1.plot(x_range, y_pred, 'r--', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')
    
    # Add diagonal line (perfect correlation)
    min_val = min(df_clean[method1_col].min(), df_clean[method2_col].min())
    max_val = max(df_clean[method1_col].max(), df_clean[method2_col].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
    
    ax1.set_xlabel(f'{method1_col}')
    ax1.set_ylabel(f'{method2_col}')
    ax1.set_title(f'Correlation Plot\nR² = {r2:.3f}, r = {pearson_r:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    residuals = df_clean[method2_col] - (slope * df_clean[method1_col] + intercept)
    ax2.scatter(df_clean[method1_col], residuals, alpha=0.7, s=60, 
               edgecolors='black', linewidth=0.5, color='#1f77b4')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel(f'{method1_col}')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig, results


def plot_flux_differences(df, method1_col, method2_col, rxn_id_col=None, 
                         top_n=20, difference_type='absolute', output_path=None, 
                         figsize=(12, 8), show=False):
    """
    Plot reactions with the biggest and smallest flux differences between two methods.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing flux data from two methods
    method1_col : str
        Name of the first flux column
    method2_col : str
        Name of the second flux column
    rxn_id_col : str, optional
        Name of the reaction ID column (uses first column if None)
    top_n : int, optional
        Number of top reactions to show
    difference_type : str, optional
        Type of difference to calculate ('absolute' or 'relative')
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
        
    Returns
    -------
    tuple
        (matplotlib.figure.Figure, dict) - The plot figure and difference data
    """
    # Set the plotting style
    set_plotting_style()
    
    # Auto-detect reaction ID column if not specified
    if rxn_id_col is None:
        rxn_id_col = df.columns[0]
    
    # Clean data
    df_clean = df.copy()
    df_clean[method1_col] = pd.to_numeric(df_clean[method1_col], errors='coerce')
    df_clean[method2_col] = pd.to_numeric(df_clean[method2_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[method1_col, method2_col])
    
    if len(df_clean) == 0:
        raise ValueError("No valid data points for difference analysis")
    
    # Calculate differences
    df_clean['abs_difference'] = np.abs(df_clean[method1_col] - df_clean[method2_col])
    df_clean['relative_difference'] = df_clean['abs_difference'] / (np.abs(df_clean[method1_col]) + 1e-10)
    
    # Choose difference type
    diff_col = 'abs_difference' if difference_type == 'absolute' else 'relative_difference'
    
    # Sort by difference
    df_sorted = df_clean.sort_values(diff_col, ascending=False)
    biggest_diff = df_sorted.head(top_n)
    smallest_diff = df_sorted.tail(top_n)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot biggest differences
    bars1 = ax1.barh(biggest_diff[rxn_id_col], biggest_diff[diff_col], color='#d62728')
    ax1.set_title(f'Top {top_n} Reactions with Biggest {difference_type.title()} Differences')
    ax1.set_xlabel(f'{difference_type.title()} Difference')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Plot smallest differences
    bars2 = ax2.barh(smallest_diff[rxn_id_col], smallest_diff[diff_col], color='#2ca02c')
    ax2.set_title(f'Top {top_n} Reactions with Smallest {difference_type.title()} Differences')
    ax2.set_xlabel(f'{difference_type.title()} Difference')
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Prepare return data
    results = {
        'biggest_differences': biggest_diff[[rxn_id_col, method1_col, method2_col, diff_col]],
        'smallest_differences': smallest_diff[[rxn_id_col, method1_col, method2_col, diff_col]],
        'sorted_df': df_sorted
    }
    
    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig, results

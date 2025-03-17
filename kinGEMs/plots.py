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

def plot_annealing_progress(iterations, biomasses, output_path=None, figsize=(10, 6),
                          show=False):
    """
    Plot simulated annealing progress.
    
    Parameters
    ----------
    iterations : list
        List of iteration numbers
    biomasses : list
        List of biomass values at each iteration
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create line plot
    ax.plot(iterations, biomasses, marker='o', linestyle='-', color='#1f77b4', 
           markerfacecolor='white', markersize=8)
    
    # Add labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Biomass (1/hr)')
    ax.set_title('Simulated Annealing Progress')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
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
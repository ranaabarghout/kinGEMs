"""
Parameter tuning module for kinGEMs.

This module provides simulated annealing functionality to tune kcat parameters
and optimize the model's performance.
"""

import math
import os
import random

import matplotlib.pyplot as plt
import pandas as pd

from ..config import ensure_dir_exists


def simulated_annealing(model, kcat_file, gene_seq_file, biomass_reaction, 
                        output_dir=None, enzyme_fraction=0.125, 
                        temperature=1.0, cooling_rate=0.98, min_temperature=0.01, 
                        max_iterations=250, max_unchanged_iterations=3, 
                        change_threshold=0.001):
    """
    Use simulated annealing to tune kcat values for improved biomass production.
    
    Parameters
    ----------
    model : str or cobra.Model
        Path to model file or COBRA model object
    kcat_file : str
        Path to kcat values file
    gene_seq_file : str
        Path to gene sequence file
    biomass_reaction : str
        ID of biomass reaction to optimize
    output_dir : str, optional
        Directory to save output files
    enzyme_fraction : float, optional
        Maximum enzyme fraction constraint
    temperature : float, optional
        Initial temperature for annealing
    cooling_rate : float, optional
        Rate at which temperature decreases
    min_temperature : float, optional
        Minimum temperature before stopping
    max_iterations : int, optional
        Maximum number of iterations
    max_unchanged_iterations : int, optional
        Maximum iterations without significant improvement
    change_threshold : float, optional
        Threshold for considering a change significant
        
    Returns
    -------
    tuple
        (kcat_dict, df_enzyme_sorted, df_new, iterations, biomasses, df_FBA)
    """
    # Import needed here to avoid circular imports
    from ..dataset import prepare_model_data
    from .optimize import run_optimization
    
    print('____________________')
    print('Simulated Annealing for tuning kcat function')
    
    # Helper functions for simulated annealing
    def acceptance_probability(old_cost, new_cost, temperature):
        """Calculate probability of accepting a new solution."""
        if new_cost > old_cost:
            return 1.0
        return math.exp((new_cost - old_cost) / temperature)

    def get_neighbor(kcat_value, std):
        """Generate a new kcat value with random perturbation within STD range."""
        new_kcat = kcat_value * (1.5 + random.uniform(-1, 3.5))
        
        if std == 0:
            std = kcat_value * 0.1
        
        ub = kcat_value + std
        lb = kcat_value - std
        
        # Ensure new kcat value is within the specified range
        new_kcat = min(max(new_kcat, lb), ub)
        
        # Also limit the new kcat value to the maximum reported kcat
        max_reported_kcat = 4.6 * 10**9  # Max reported kcat (carbonic anhydrase)
        new_kcat = min(new_kcat, max_reported_kcat)
        
        return new_kcat
    
    def update_kcat(df, reaction_id, gene_id, new_kcat_value):
        """Update a specific kcat value in the dataframe."""
        updated_df = df.copy()
        condition = (updated_df['Reactions'] == reaction_id) & (updated_df['Single_gene'] == gene_id)
        updated_df.loc[condition, 'kcat (1/hr)'] = new_kcat_value
        return updated_df
    
    # Initial solution - prepare the model and run initial optimization
    new_model_path, kcat_dict, df_data = prepare_model_data(
        model, kcat_file, biomass_reaction, output_dir)
        
    largest_rxn_id, largest_gene_id, largest_kcat, largest_STD, biomass, df_new, _, df_enzyme_sorted, df_FBA = run_optimization(
        new_model_path, kcat_dict, biomass_reaction, gene_seq_file, enzyme_fraction)

    print('Initial working biomass: ', biomass)
    
    # Initialize tracking variables
    current_solution = largest_kcat
    current_biomass = biomass
    best_solution = current_solution
    best_biomass = current_biomass
    current_STD = largest_STD
    
    iteration = 1
    no_change_counter = 0
    
    iterations = []
    biomasses = []
    biomasses.append(biomass)
    iterations.append(0)
    
    # Main simulated annealing loop
    while temperature > min_temperature and iteration < max_iterations:
        print("Iteration number: ", iteration)
        
        # Update kcat values for the target reactions
        for i in range(len(largest_rxn_id)):
            new_kcat = get_neighbor(current_solution[i], current_STD[i])
            if i == 0 and iteration == 1:
                updated_df = update_kcat(df_new, largest_rxn_id[i], largest_gene_id[i], new_kcat)
            else:
                updated_df = update_kcat(updated_df, largest_rxn_id[i], largest_gene_id[i], new_kcat)
        
        # Run optimization with updated kcats
        largest_rxn_id, largest_gene_id, largest_kcat, largest_STD, new_biomass, df_new, t1, t2, df_FBA = run_optimization(
            new_model_path, kcat_dict, biomass_reaction, gene_seq_file, enzyme_fraction, df=updated_df, data_available=True)
        
        # Determine whether to accept the new solution
        if acceptance_probability(current_biomass, new_biomass, temperature) > random.random():
            change = abs(new_biomass - current_biomass) / current_biomass
            current_solution = largest_kcat
            current_biomass = new_biomass
            
            if new_biomass > best_biomass:
                best_solution = current_solution
                best_biomass = new_biomass
        
        iterations.append(iteration)
        biomasses.append(new_biomass)
        
        # Check for significant change in biomass
        print('Change in biomass from last iteration: ', change)
        if change < change_threshold:
            no_change_counter += 1
        else:
            no_change_counter = 0
        
        if no_change_counter >= max_unchanged_iterations:
            print(f"No significant change in biomass for {max_unchanged_iterations} iterations. Stopping early.")
            break
        
        temperature *= cooling_rate
        iteration += 1
        
        print(f'Biomass with tuned kcats: {new_biomass}')
        print('------------')
    
    # If output_dir is provided, save results
    if output_dir:
        save_annealing_results(output_dir, kcat_dict, df_enzyme_sorted, df_new, iterations, biomasses, df_FBA)
    
    return kcat_dict, df_enzyme_sorted, df_new, iterations, biomasses, df_FBA

def save_annealing_results(output_dir, kcat_dict, df_enzyme_sorted, df_new, iterations, biomasses, df_FBA, prefix=""):
    """
    Save the results of the simulated annealing process.
    
    Parameters
    ----------
    output_dir : str
        Directory to save output files
    kcat_dict : dict
        Dictionary of optimized kcat values
    df_enzyme_sorted : pandas.DataFrame
        DataFrame with sorted enzyme data
    df_new : pandas.DataFrame
        DataFrame with updated kcat values
    iterations : list
        List of iteration numbers
    biomasses : list
        List of biomass values at each iteration
    df_FBA : pandas.DataFrame
        DataFrame with FBA results
    prefix : str, optional
        Prefix for output filenames
    """
    # Ensure directory exists
    ensure_dir_exists(output_dir)
    
    # Save kcat dictionary
    kcat_dict_df = pd.DataFrame(list(kcat_dict.items()), columns=['Key', 'Value'])
    kcat_dict_df.to_csv(os.path.join(output_dir, f"{prefix}kcat_dict.csv"), index=False)
    
    # Save sorted enzyme data
    df_enzyme_sorted.to_csv(os.path.join(output_dir, f"{prefix}df_enzyme_sorted.csv"), index=False)
    
    # Save updated data
    df_new.to_csv(os.path.join(output_dir, f"{prefix}df_new.csv"), index=False)
    
    # Save FBA results
    df_FBA.to_csv(os.path.join(output_dir, f"{prefix}df_FBA.csv"), index=False)
    
    # Save iterations data
    df_iterations = pd.DataFrame({"Iteration": iterations, "Biomass": biomasses})
    df_iterations.to_csv(os.path.join(output_dir, f"{prefix}iterations.csv"), index=False)
    
    # Create and save plot
    plot_annealing_progress(iterations, biomasses, 
                           output_path=os.path.join(output_dir, f"{prefix}annealing_progress.png"))
    
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
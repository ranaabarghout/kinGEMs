"""
Parameter tuning module for kinGEMs.

This module provides simulated annealing functionality to tune kcat parameters
and optimize the model's performance.
"""

import copy
import math  # noqa: F401
import os
import random  # noqa: F401

from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
import pandas as pd

from ..config import ensure_dir_exists
from ..dataset import annotate_model_with_kcat_and_gpr
from .optimize import run_optimization_with_dataframe


def simulated_annealing(
    model,
    processed_data,
    biomass_reaction,
    objective_value,
    gene_sequences_dict,
    output_dir=None,
    enzyme_fraction=0.125,
    temperature=1.0,
    cooling_rate=0.98,
    min_temperature=0.01,
    max_iterations=250,
    max_unchanged_iterations=3,
    change_threshold=0.001
):
    """
    Use simulated annealing to tune kcat values for improved biomass production.
    """

    def acceptance_probability(old_cost, new_cost, temperature):
        if new_cost > old_cost:
            return 1.0
        return math.exp((new_cost - old_cost) / temperature)

    def get_neighbor(kcat_value, std):
        k_val_hr = kcat_value * 3600
        std_hr = std * 3600
        new_kcat = k_val_hr * (1.5 + random.uniform(-1, 3.5))
        if std_hr == 0:
            std_hr = k_val_hr * 0.1
        ub = k_val_hr + std_hr
        lb = k_val_hr - std_hr
        # Also limit to max reported kcat (per hour)
        max_hr = 4.6e9  # carbonic anhydrase per hour
        return min(max(new_kcat, lb), min(ub, max_hr))

    def update_kcat(df, reaction_id, gene_id, new_kcat_value):
        updated_df = df.copy()
        cond = (
            (updated_df['Reactions'] == reaction_id) &
            (updated_df['Single_gene'] == gene_id)
        )
        # convert back to per-second
        updated_df.loc[cond, 'kcat_mean'] = new_kcat_value / 3600
        return updated_df

    def calculate_molecular_weight(seq):
        return ProteinAnalysis(seq).molecular_weight()

    # Precompute MWs
    mw_dict = {
        gene: calculate_molecular_weight(seq)
        for gene, seq in gene_sequences_dict.items() if seq
    }

    # INITIAL FBA
    biomass, df_FBA, _, _ = run_optimization_with_dataframe(
        model=model,
        processed_df=processed_data,
        objective_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_fraction,
        output_dir=output_dir,
        save_results=False
    )

    # pick top 25
    enzyme_df = df_FBA[df_FBA['Variable']=='enzyme'].copy()
    enzyme_df['MW'] = enzyme_df['Index'].map(mw_dict).fillna(0)
    enzyme_df['enzyme_mass'] = enzyme_df['Value'] * enzyme_df['MW'] * 1e-3
    top25 = enzyme_df.nlargest(25, 'enzyme_mass')
    top_targets = (
        top25[['Index','enzyme_mass']]
        .rename(columns={'Index':'Single_gene'})
        .merge(processed_data, on='Single_gene')
        [['Reactions','Single_gene','enzyme_mass','kcat_mean','kcat_std']]
        .reset_index(drop=True)
    )

    largest_rxn_id  = top_targets['Reactions'].tolist()
    largest_gene_id = top_targets['Single_gene'].tolist()
    current_solution = top_targets['kcat_mean'].tolist()
    stds             = top_targets['kcat_std'].fillna(0.1).tolist()

    df_new = processed_data.copy()
    current_biomass = biomass
    best_solution   = current_solution[:]
    best_biomass    = current_biomass
    best_df         = df_new.copy()

    iteration = 1
    no_change_counter = 0
    iterations = [0]
    biomasses  = [biomass]

    # ANNEALING
    while (temperature > min_temperature
           and iteration < max_iterations
           and current_biomass < objective_value):

        print(f"\n--- Iteration {iteration} ---")
        print(f"Current biomass = {current_biomass:.6e}")

        # PROPOSE & print old vs new kcats
        updated_df = df_new.copy()
        for i, (rxn, gene) in enumerate(zip(largest_rxn_id, largest_gene_id)):
            old_k = current_solution[i]
            new_k_hr = get_neighbor(old_k, stds[i])
            new_k = new_k_hr / 3600.0
            print(f"  {rxn}_{gene}: old kcat = {old_k:.3e}  →  new kcat = {new_k:.3e}")
            updated_df = update_kcat(updated_df, rxn, gene, new_k_hr)

        # ANNOTATE & EVALUATE on a clone
        temp_model = copy.deepcopy(model)
        temp_model = annotate_model_with_kcat_and_gpr(model=temp_model, df=updated_df)

        new_biomass, temp_df_FBA, _, _ = run_optimization_with_dataframe(
            model=temp_model,
            processed_df=updated_df,
            objective_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_fraction,
            output_dir=None,
            save_results=False
        )
        print(f"Proposed biomass = {new_biomass:.6e}")

        # ACCEPT or REJECT
        prob = acceptance_probability(current_biomass, new_biomass, temperature)
        if prob > random.random():
            print(f"Iteration {iteration}: ACCEPTED (Δ = {new_biomass-current_biomass:.2e})")
            model = temp_model
            df_FBA = temp_df_FBA
            df_new = updated_df.copy()
            current_biomass = new_biomass
            current_solution = [
                df_new.loc[
                    (df_new['Reactions']==rxn)&(df_new['Single_gene']==gene),
                    'kcat_mean'
                ].iat[0]
                for rxn,gene in zip(largest_rxn_id, largest_gene_id)
            ]
            if new_biomass > best_biomass:
                best_biomass  = new_biomass
                best_solution = current_solution[:]
                best_df       = df_new.copy()
        else:
            print(f"Iteration {iteration}: REJECTED (Δ = {new_biomass-current_biomass:.2e})")

        iterations.append(iteration)
        biomasses.append(new_biomass)

        # check stagnation
        change = abs(new_biomass - current_biomass) / (current_biomass or 1)
        if change < change_threshold:
            no_change_counter += 1
            if no_change_counter >= max_unchanged_iterations:
                print(f"No significant change for {max_unchanged_iterations} iterations; stopping early.")
                break
        else:
            no_change_counter = 0

        temperature *= cooling_rate
        iteration += 1

    # FINALIZE: build kcat_dict from best_solution
    kcat_dict = {
        f"{rxn}_{gene}": k
        for (rxn, gene), k in zip(zip(largest_rxn_id, largest_gene_id), best_solution)
    }

    if output_dir:
        save_annealing_results(
            output_dir,
            kcat_dict,
            top_targets,
            best_df,
            iterations,
            biomasses,
            df_FBA
        )

    return kcat_dict, top_targets, best_df, iterations, biomasses, df_FBA


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
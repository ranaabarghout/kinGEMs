"""
Flux Variability Analysis module for kinGEMs.

This module provides functions for performing Flux Variability Analysis (FVA)
with enzyme constraints, allowing exploration of the solution space.
"""

import os
import pandas as pd
import cobra as cb
from ..config import ensure_dir_exists

def flux_variability_analysis(model, kcat_dict, biomass_reaction, gene_sequence_file, 
                            output_file=None, enzyme_upper_bound=0.125, enzyme_ratio=True,
                            multi_enzyme_off=False, isoenzymes_off=False, 
                            promiscuous_off=False, complexes_off=False):
    """
    Perform Flux Variability Analysis with enzyme constraints.
    
    FVA calculates the minimum and maximum possible flux values through each
    reaction in the model while maintaining the optimal objective value.
    
    Parameters
    ----------
    model : cobra.Model or str
        COBRA model object or path to the model file
    kcat_dict : dict or str
        Dictionary of kcat values or path to kcat CSV file
    biomass_reaction : str
        ID of the biomass reaction to be fixed
    gene_sequence_file : str
        Path to file containing gene sequences
    output_file : str, optional
        Path to save results
    enzyme_upper_bound : float, optional
        Upper bound for total enzyme concentration
    enzyme_ratio : bool, optional
        Whether to use enzyme ratio constraint
    multi_enzyme_off : bool, optional
        Whether to disable multi-enzyme reactions
    isoenzymes_off : bool, optional
        Whether to disable isoenzyme handling
    promiscuous_off : bool, optional
        Whether to disable promiscuous enzyme handling
    complexes_off : bool, optional
        Whether to disable enzyme complex handling
        
    Returns
    -------
    tuple
        (df_FVA_solution, kcat_dict, df_FBA)
    """
    # Import here to avoid circular imports
    from .optimize import run_optimization
    
    # Model handling
    if isinstance(model, str):
        directory = os.path.dirname(__file__)
        model_path = os.path.join(directory, model)
        model_obj = cb.io.read_sbml_model(model_path)
    else:
        model_obj = model
        
    # Run initial optimization to get optimal biomass value
    solution_biomass, df_FBA, gene_seq_dict = run_optimization(
        model_obj,
        kcat_dict,
        biomass_reaction,
        gene_sequence_file,
        enzyme_upper_bound=enzyme_upper_bound,
        enzyme_ratio=enzyme_ratio,
        multi_enzyme_off=multi_enzyme_off,
        isoenzymes_off=isoenzymes_off,
        promiscuous_off=promiscuous_off,
        complexes_off=complexes_off
    )
    
    print('Optimal solution biomass:', solution_biomass)
     
    # Get all reaction IDs for FVA
    reactions = list(set(reaction.id for reaction in model_obj.reactions))
    
    # Set biomass reaction to fixed value from the optimization
    biomass_rxn = model_obj.reactions.get_by_id(biomass_reaction)
    biomass_rxn.lower_bound = solution_biomass
    biomass_rxn.upper_bound = solution_biomass
    
    # Arrays to store results
    max_solutions = []
    min_solutions = []
    max_solutions_rxn = [] 
    min_solutions_rxn = [] 
    
    # Perform FVA for each reaction
    for i, rxn in enumerate(reactions):
        print(f'Starting FVA with reaction {rxn} ({i+1}/{len(reactions)})')
        print("-------------------------------------------------------")
        
        # Maximize flux through the reaction
        print("Now starting maximization LP....")
        solution_max, _, _ = run_optimization(
            model_obj,
            kcat_dict,
            rxn,
            gene_sequence_file,
            enzyme_upper_bound=enzyme_upper_bound,
            enzyme_ratio=enzyme_ratio,
            multi_enzyme_off=multi_enzyme_off,
            isoenzymes_off=isoenzymes_off,
            promiscuous_off=promiscuous_off,
            complexes_off=complexes_off,
            maximization=True
        )
        
        max_solutions.append(solution_max)    
        max_solutions_rxn.append(i)
        
        # Minimize flux through the reaction
        print("Now starting minimization LP....")
        solution_min, _, _ = run_optimization(
            model_obj,
            kcat_dict,
            rxn,
            gene_sequence_file,
            enzyme_upper_bound=enzyme_upper_bound,
            enzyme_ratio=enzyme_ratio,
            multi_enzyme_off=multi_enzyme_off,
            isoenzymes_off=isoenzymes_off,
            promiscuous_off=promiscuous_off,
            complexes_off=complexes_off,
            maximization=False
        )
        
        min_solutions.append(solution_min)   
        min_solutions_rxn.append(i)
        
        print(f"Reaction {rxn} FVA now complete.")
        print("_____________________________________")
    
    # Compile results into a dataframe
    data = {
        'Reactions': reactions,
        'Min Solutions': min_solutions,
        'Max Solutions': max_solutions,
        'Solution Biomass': [solution_biomass] * len(reactions)
    }
    df_FVA_solution = pd.DataFrame(data)
    
    # Save results if output file specified
    if output_file:
        directory = os.path.dirname(__file__)
        output_path = os.path.join(directory, output_file)
        ensure_dir_exists(os.path.dirname(output_path))
        df_FVA_solution.to_csv(output_path)
    
    return df_FVA_solution, kcat_dict, df_FBA

def calculate_flux_ranges(fva_results):
    """
    Calculate the flux ranges from FVA results.
    
    Parameters
    ----------
    fva_results : pandas.DataFrame
        FVA results dataframe
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added flux range column
    """
    results = fva_results.copy()
    results['Flux Range'] = results['Max Solutions'] - results['Min Solutions']
    return results

def identify_essential_reactions(fva_results, threshold=1e-6):
    """
    Identify essential reactions based on FVA results.
    
    Essential reactions are those that must carry a non-zero flux
    for the model to achieve optimal growth.
    
    Parameters
    ----------
    fva_results : pandas.DataFrame
        FVA results dataframe
    threshold : float, optional
        Minimum absolute flux value to consider a reaction as carrying flux
        
    Returns
    -------
    list
        List of essential reaction IDs
    """
    essential = []
    for _, row in fva_results.iterrows():
        # A reaction is essential if both its min and max are non-zero and have the same sign
        if ((abs(row['Min Solutions']) > threshold and abs(row['Max Solutions']) > threshold) and
            (row['Min Solutions'] * row['Max Solutions'] > 0)):
            essential.append(row['Reactions'])
    return essential

def identify_blocked_reactions(fva_results, threshold=1e-6):
    """
    Identify blocked reactions based on FVA results.
    
    Blocked reactions are those that cannot carry flux under any condition
    while maintaining optimal growth.
    
    Parameters
    ----------
    fva_results : pandas.DataFrame
        FVA results dataframe
    threshold : float, optional
        Maximum absolute flux value to consider a reaction as blocked
        
    Returns
    -------
    list
        List of blocked reaction IDs
    """
    blocked = []
    for _, row in fva_results.iterrows():
        # A reaction is blocked if both its min and max are essentially zero
        if abs(row['Min Solutions']) < threshold and abs(row['Max Solutions']) < threshold:
            blocked.append(row['Reactions'])
    return blocked

def compare_fva_results(fva_result1, fva_result2, name1='Model 1', name2='Model 2', output_file=None):
    """
    Compare two FVA result sets to identify differences in flux ranges.
    
    Parameters
    ----------
    fva_result1 : pandas.DataFrame
        First FVA results dataframe
    fva_result2 : pandas.DataFrame
        Second FVA results dataframe
    name1 : str, optional
        Name for the first model
    name2 : str, optional
        Name for the second model
    output_file : str, optional
        Path to save the comparison results
        
    Returns
    -------
    pandas.DataFrame
        DataFrame comparing flux ranges between the two models
    """
    # Calculate flux ranges
    df1 = calculate_flux_ranges(fva_result1)
    df2 = calculate_flux_ranges(fva_result2)
    
    # Merge on reaction IDs
    comparison = pd.merge(
        df1[['Reactions', 'Min Solutions', 'Max Solutions', 'Flux Range']], 
        df2[['Reactions', 'Min Solutions', 'Max Solutions', 'Flux Range']], 
        on='Reactions', 
        suffixes=(f'_{name1}', f'_{name2}')
    )
    
    # Calculate differences
    comparison[f'Min_Diff'] = comparison[f'Min Solutions_{name2}'] - comparison[f'Min Solutions_{name1}']
    comparison[f'Max_Diff'] = comparison[f'Max Solutions_{name2}'] - comparison[f'Max Solutions_{name1}']
    comparison[f'Range_Diff'] = comparison[f'Flux Range_{name2}'] - comparison[f'Flux Range_{name1}']
    
    # Save results if output file specified
    if output_file:
        directory = os.path.dirname(__file__)
        output_path = os.path.join(directory, output_file)
        ensure_dir_exists(os.path.dirname(output_path))
        comparison.to_csv(output_path)
    
    return comparison

def plot_flux_variability(fva_results, reactions=None, figsize=(12, 8), output_file=None):
    """
    Create a plot visualizing the flux variability for selected reactions.
    
    Parameters
    ----------
    fva_results : pandas.DataFrame
        FVA results dataframe
    reactions : list, optional
        List of reaction IDs to include in the plot. If None, plots reactions with 
        the largest flux ranges.
    figsize : tuple, optional
        Figure size (width, height)
    output_file : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The plot figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    df = calculate_flux_ranges(fva_results)
    
    # If no reactions specified, select top 20 with largest flux range
    if reactions is None:
        df = df.sort_values('Flux Range', ascending=False)
        reactions = df['Reactions'].head(20).tolist()
    else:
        df = df[df['Reactions'].isin(reactions)]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up colors
    colors = plt.cm.tab20.colors
    
    # Create x-axis positions
    positions = np.arange(len(reactions))
    
    # Plot the flux ranges as horizontal lines
    for i, rxn in enumerate(reactions):
        row = df[df['Reactions'] == rxn].iloc[0]
        min_val = row['Min Solutions']
        max_val = row['Max Solutions']
        ax.plot([min_val, max_val], [i, i], 'o-', color=colors[i % len(colors)], 
                linewidth=2, markersize=8, label=rxn)
    
    # Configure plot appearance
    ax.set_yticks(positions)
    ax.set_yticklabels(reactions)
    ax.set_xlabel('Flux Value (mmol/gDW/h)', fontsize=14)
    ax.set_title('Flux Variability Analysis', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add zero line for reference
    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        directory = os.path.dirname(__file__)
        output_path = os.path.join(directory, output_file)
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
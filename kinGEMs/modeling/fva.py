"""
Flux Variability Analysis module for kinGEMs.

This module provides functions for performing Flux Variability Analysis (FVA)
with enzyme constraints, allowing exploration of the solution space.
"""

import os

import pandas as pd

from ..config import ensure_dir_exists


def flux_variability_analysis(model, processed_df, biomass_reaction,
                               output_file=None, enzyme_upper_bound=0.15, opt_ratio=0.9, enzyme_ratio=True,
                               multi_enzyme_off=False, isoenzymes_off=False,
                               promiscuous_off=False, complexes_off=False):
    """
    Perform Flux Variability Analysis (FVA) using enzyme-constrained optimization
    with kcat values from a processed dataframe.
    
    Parameters
    ----------
    model : cobra.Model
        COBRA model object
    processed_df : pandas.DataFrame
        DataFrame with kcat_mean, SEQ, Reactions, Single_gene
    biomass_reaction : str
        ID of the biomass reaction
    output_file : str, optional
        File path to save FVA results
    enzyme_upper_bound : float, optional
        Enzyme pool constraint
    enzyme_ratio : bool, optional
        Whether to apply enzyme ratio constraint
    multi_enzyme_off, isoenzymes_off, promiscuous_off, complexes_off : bool
        Logic switches for model complexity
        
    Returns
    -------
    tuple
        (df_FVA_solution, processed_df, df_FBA)
    """
    from .optimize import run_optimization_with_dataframe

    print("=== Starting FVA with enzyme constraints ===")

    # Step 1: Run optimization to get baseline biomass
    solution_biomass, df_FBA, _, _ = run_optimization_with_dataframe(
        model=model,
        processed_df=processed_df,
        objective_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        enzyme_ratio=enzyme_ratio,
        multi_enzyme_off=multi_enzyme_off,
        isoenzymes_off=isoenzymes_off,
        promiscuous_off=promiscuous_off,
        complexes_off=complexes_off,
        maximization=True,
        save_results=False
    )

    print(f"Optimal biomass: {solution_biomass:.6f}")

    # Step 2: Fix biomass value
    biomass_rxn = model.reactions.get_by_id(biomass_reaction)
    biomass_rxn.lower_bound = solution_biomass * opt_ratio
    # biomass_rxn.upper_bound = solution_biomass

    # Step 3: Run FVA
    min_fluxes = []
    max_fluxes = []
    reaction_ids = []

    for i, rxn in enumerate(model.reactions):
        print(f"[{i + 1}/{len(model.reactions)}] FVA for: {rxn.id}")

         # Deepcopy model to avoid accumulating changes
        model_copy_max = model.copy()
        model_copy_min = model.copy()

        # Maximize this reaction
        try:
            flux_max, _, _, _ = run_optimization_with_dataframe(
                model=model_copy_max,
                processed_df=processed_df,
                objective_reaction=rxn.id,
                enzyme_upper_bound=enzyme_upper_bound,
                enzyme_ratio=enzyme_ratio,
                multi_enzyme_off=multi_enzyme_off,
                isoenzymes_off=isoenzymes_off,
                promiscuous_off=promiscuous_off,
                complexes_off=complexes_off,
                maximization=True,
                save_results=False
            )
        except Exception as e:
            print(f"⚠️ Max FVA failed for {rxn.id}: {e}")
            flux_max = None


        # Minimize this reaction
        try:
            flux_min, _, _, _ = run_optimization_with_dataframe(
                model=model_copy_min,
                processed_df=processed_df,
                objective_reaction=rxn.id,
                enzyme_upper_bound=enzyme_upper_bound,
                enzyme_ratio=enzyme_ratio,
                multi_enzyme_off=multi_enzyme_off,
                isoenzymes_off=isoenzymes_off,
                promiscuous_off=promiscuous_off,
                complexes_off=complexes_off,
                maximization=False,
                save_results=False
            )
        except Exception as e:
            print(f"⚠️ Min FVA failed for {rxn.id}: {e}")
            flux_min = None

        reaction_ids.append(rxn.id)
        max_fluxes.append(flux_max)
        min_fluxes.append(flux_min)

    # Step 4: Compile results
    df_FVA_solution = pd.DataFrame({
        "Reactions": reaction_ids,
        "Min Solutions": min_fluxes,
        "Max Solutions": max_fluxes,
        "Solution Biomass": [solution_biomass] * len(reaction_ids)
    })

    if output_file:
        ensure_dir_exists(os.path.dirname(output_file))
        df_FVA_solution.to_csv(output_file, index=False)
        print(f"FVA results saved to: {output_file}")

    return df_FVA_solution, processed_df, df_FBA


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
    comparison['Min_Diff'] = comparison[f'Min Solutions_{name2}'] - comparison[f'Min Solutions_{name1}']
    comparison['Max_Diff'] = comparison[f'Max Solutions_{name2}'] - comparison[f'Max Solutions_{name1}']
    comparison['Range_Diff'] = comparison[f'Flux Range_{name2}'] - comparison[f'Flux Range_{name1}']
    
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

def plot_cumulative_fvi_distribution(dfs, labels, output_file=None, roboto_font=None):
    """
    Plot cumulative distributions of Flux Variability Index (FVi) for multiple FVA result sets.
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of FVA result dataframes.
    labels : list of str
        Labels corresponding to each dataframe.
    output_file : str, optional
        Path to save the plot.
    roboto_font : font manager or None
        Optional custom font (e.g., matplotlib.font_manager.FontProperties)
    
    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)
    ax2 = ax1.twinx()

    percentages = []
    fvi_at_0_5 = []

    for df, label in zip(dfs, labels):
        # Separate forward and reverse reactions
        reverse_rows = df[df['Reactions'].str.endswith('_reverse')].copy()
        normal_rows = df[~df['Reactions'].str.endswith('_reverse')].copy()
        reverse_rows['Reactions'] = reverse_rows['Reactions'].str.replace('_reverse', '', regex=False)

        combined_df = pd.merge(
            normal_rows,
            reverse_rows,
            on='Reactions',
            how='left',
            suffixes=('', '_Reverse')
        )

        def calculate_fvi(row):
            if pd.isna(row['Max Solutions']):
                return np.nan
            elif pd.isna(row['Max Solutions_Reverse']) or pd.isna(row['Min Solutions_Reverse']):
                return abs(row['Max Solutions'] - row['Min Solutions'])
            else:
                return abs((row['Max Solutions'] - row['Min Solutions']) -
                           (row['Max Solutions_Reverse'] - row['Min Solutions_Reverse']))

        combined_df['FVi'] = combined_df.apply(calculate_fvi, axis=1)
        fvi_values = combined_df['FVi'].dropna()
        fvi_values = fvi_values[fvi_values >= 1e-6]

        sorted_fvi = np.sort(fvi_values)
        cumulative = np.arange(1, len(sorted_fvi) + 1) / len(sorted_fvi)

        ax1.plot(sorted_fvi, cumulative, label=label, linewidth=3)

        biomass_value = df['Solution Biomass'].iloc[0] if 'Solution Biomass' in df.columns else 0.0
        fvi_50 = np.interp(0.5, cumulative, sorted_fvi) if len(sorted_fvi) > 0 else np.nan
        fvi_at_0_5.append((label, fvi_50))

        ax2.plot([fvi_values.min(), fvi_values.max()],
                 [biomass_value, biomass_value],
                 linestyle='--', color=ax1.lines[-1].get_color(), linewidth=2, alpha=0.6)

        percent_above_990 = (fvi_values > 990).mean() * 100
        percentages.append((label, percent_above_990))

    print("Percent of reactions with FVi > 990:")
    for label, pct in percentages:
        print(f"  {label}: {pct:.2f}%")

    print("\nFVi at cumulative probability = 0.5:")
    for label, val in fvi_at_0_5:
        print(f"  {label}: {val:.4f}")

    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1.5)

    ax1.set_xscale('log')
    ax1.set_xlim(1e-6, 1e3)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Flux Variability Range (FVi)', fontsize=16, fontproperties=roboto_font)
    ax1.set_ylabel('Cumulative Probability', fontsize=16, fontproperties=roboto_font)
    ax2.set_ylabel('Biomass (1/hr)', fontsize=16, fontproperties=roboto_font)

    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=12, prop=roboto_font)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.show()

## PARALLEL FVA

import os

from dask import compute, delayed
from dask.distributed import Client
import pandas as pd


# 1) A small helper that does the two optimizations for one reaction:
def _fva_for_reaction(model, processed_df,
                      rxn_id,
                      biomass_rxn_bounds,
                      enzyme_upper_bound, enzyme_ratio,
                      multi_enzyme_off, isoenzymes_off,
                      promiscuous_off, complexes_off):
    from .optimize import run_optimization_with_dataframe
    
    # restore the fixed‐biomass bounds
    biomass_rxn = model.reactions.get_by_id(biomass_rxn_bounds[0])
    biomass_rxn.lower_bound, biomass_rxn.upper_bound = biomass_rxn_bounds[1:]
    
    # copy model twice so we don’t clobber bounds
    m_max, m_min = model.copy(), model.copy()
    
    # maximize
    try:
        flux_max, *_ = run_optimization_with_dataframe(
            model=m_max,
            processed_df=processed_df,
            objective_reaction=rxn_id,
            enzyme_upper_bound=enzyme_upper_bound,
            enzyme_ratio=enzyme_ratio,
            multi_enzyme_off=multi_enzyme_off,
            isoenzymes_off=isoenzymes_off,
            promiscuous_off=promiscuous_off,
            complexes_off=complexes_off,
            maximization=True,
            save_results=False,
        )
    except Exception:
        flux_max = None
    
    # minimize
    try:
        flux_min, *_ = run_optimization_with_dataframe(
            model=m_min,
            processed_df=processed_df,
            objective_reaction=rxn_id,
            enzyme_upper_bound=enzyme_upper_bound,
            enzyme_ratio=enzyme_ratio,
            multi_enzyme_off=multi_enzyme_off,
            isoenzymes_off=isoenzymes_off,
            promiscuous_off=promiscuous_off,
            complexes_off=complexes_off,
            maximization=False,
            save_results=False,
        )
    except Exception:
        flux_min = None

    return rxn_id, flux_min, flux_max

# 2) Your FVA, rewritten to launch in parallel:
def flux_variability_analysis_parallel(model, processed_df, biomass_reaction,
                               output_file=None,
                               enzyme_upper_bound=0.15, enzyme_ratio=True,
                               multi_enzyme_off=False, isoenzymes_off=False,
                               promiscuous_off=False, complexes_off=False,
                               n_workers=None):
    """
    Perform Flux Variability Analysis (FVA) in parallel using Dask.
    """
    # Spin up a local Dask client (optional; if you omit this, compute() will
    # use threads by default)
    client = Client(n_workers=(n_workers or os.cpu_count()),
                processes=True,      # optional: use separate processes
                threads_per_worker=1)  # you can tune this
    
    # 1) baseline FBA to fix biomass
    from .optimize import run_optimization_with_dataframe
    sol_biomass, df_FBA, _, _ = run_optimization_with_dataframe(
        model=model, processed_df=processed_df,
        objective_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        enzyme_ratio=enzyme_ratio,
        multi_enzyme_off=multi_enzyme_off,
        isoenzymes_off=isoenzymes_off,
        promiscuous_off=promiscuous_off,
        complexes_off=complexes_off,
        maximization=True, save_results=False,
    )
    # freeze biomass
    biomass_rxn = model.reactions.get_by_id(biomass_reaction)
    biomass_bounds = (biomass_reaction, sol_biomass, sol_biomass)

    # 2) create one delayed task per reaction
    tasks = []
    for rxn in model.reactions:
        tasks.append(
            delayed(_fva_for_reaction)(
                model.copy(),           # each worker gets its own model copy
                processed_df,
                rxn.id,
                biomass_bounds,
                enzyme_upper_bound, enzyme_ratio,
                multi_enzyme_off, isoenzymes_off,
                promiscuous_off, complexes_off
            )
        )

    # 3) compute in parallel
    results = compute(*tasks)  # by default uses the Client’s cluster

    # 4) stitch results back into a DataFrame
    reaction_ids, min_vals, max_vals = zip(*results)
    df_FVA = pd.DataFrame({
        "Reactions": reaction_ids,
        "Min Solutions": min_vals,
        "Max Solutions": max_vals,
        "Solution Biomass": [sol_biomass] * len(results)
    })

    # 5) optionally save
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_FVA.to_csv(output_file, index=False)

    client.close()
    return df_FVA, processed_df, df_FBA

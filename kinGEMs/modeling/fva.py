"""
Flux Variability Analysis module for kinGEMs.

This module provides functions for performing Flux Variability Analysis (FVA)
with enzyme constraints, allowing exploration of the solution space.
"""
import logging
import os
import time
import warnings

from dask import compute, delayed
from dask.distributed import Client
import pandas as pd

from ..config import ensure_dir_exists

warnings.filterwarnings('ignore')


logging.getLogger('distributed').setLevel(logging.ERROR)
try:
    import gurobipy
    gurobipy.setParam('OutputFlag', 0)
except ImportError:
    pass




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
    biomass_rxn.upper_bound = solution_biomass

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
        print(f"Maximizing flux for reaction: {rxn.id}")
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
                save_results=False,
                verbose=True,
            )
        except Exception as e:
            print(f"⚠️ Max FVA failed for {rxn.id}: {e}")
            flux_max = None

        # Minimize this reaction
        print(f"Minimizing flux for reaction: {rxn.id}")
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
                save_results=False,
                verbose=True,
            )
        except Exception as e:
            print(f"⚠️ Min FVA failed for {rxn.id}: {e}")
            flux_min = None

        print(f"Reaction: {rxn.id} | Min Solution: {flux_min} | Max Solution: {flux_max}")

        print(f"Reaction: {rxn.id} | Min Solution: {flux_min} | Max Solution: {flux_max}")
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
        # For each reaction, treat min/max as forward/backward directions
        fvi_list = []
        for idx, row in df.iterrows():
            min_flux = row['Min Solutions']
            max_flux = row['Max Solutions']
            # Forward direction: positive flux range
            fwd_flux = max(max_flux, 0) - max(min_flux, 0)
            # Backward direction: negative flux range
            bwd_flux = abs(min(min_flux, 0) - min(max_flux, 0))
            # FVi is sum of both directions
            fvi = fwd_flux + bwd_flux
            fvi_list.append(fvi)

        fvi_values = np.array(fvi_list)
        fvi_values = fvi_values[~np.isnan(fvi_values)]
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
    print(f"[FVA] Starting optimization for reaction: {rxn_id}")

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
            verbose=False
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
            verbose=False
        )
    except Exception:
        flux_min = None

    return rxn_id, flux_min, flux_max

# 2) Your FVA, rewritten to launch in parallel:
# def flux_variability_analysis_parallel(model, processed_df, biomass_reaction,
#                                output_file=None,
#                                enzyme_upper_bound=0.15, enzyme_ratio=True,
#                                multi_enzyme_off=False, isoenzymes_off=False,
#                                promiscuous_off=False, complexes_off=False,
#                                n_workers=None):
#     """
#     Perform Flux Variability Analysis (FVA) in parallel using Dask.
#     """
#     # Spin up a local Dask client (optional; if you omit this, compute() will
#     # use threads by default)
#     client = Client(n_workers=(n_workers or os.cpu_count()),
#                 processes=True,      # optional: use separate processes
#                 threads_per_worker=1)  # you can tune this

#     # 1) baseline FBA to fix biomass
#     from .optimize import run_optimization_with_dataframe
#     sol_biomass, df_FBA, _, _ = run_optimization_with_dataframe(
#         model=model, processed_df=processed_df,
#         objective_reaction=biomass_reaction,
#         enzyme_upper_bound=enzyme_upper_bound,
#         enzyme_ratio=enzyme_ratio,
#         multi_enzyme_off=multi_enzyme_off,
#         isoenzymes_off=isoenzymes_off,
#         promiscuous_off=promiscuous_off,
#         complexes_off=complexes_off,
#         maximization=True, save_results=False,
#     )
#     # freeze biomass
#     # biomass_rxn = model.reactions.get_by_id(biomass_reaction)
#     biomass_bounds = (biomass_reaction, sol_biomass, sol_biomass)

#     # 2) create one delayed task per reaction
#     tasks = []
#     for rxn in model.reactions:
#         tasks.append(
#             delayed(_fva_for_reaction)(
#                 model.copy(),           # each worker gets its own model copy
#                 processed_df,
#                 rxn.id,
#                 biomass_bounds,
#                 enzyme_upper_bound, enzyme_ratio,
#                 multi_enzyme_off, isoenzymes_off,
#                 promiscuous_off, complexes_off
#             )
#         )

#     # 3) compute in parallel
#     results = compute(*tasks)  # by default uses the Client’s cluster

#     # Print condensed summary of min and max solution values for each reaction
#     print("\nFVA Results (Reaction | Min | Max):")
#     print("{:<20} {:>12} {:>12}".format("Reaction", "Min", "Max"))
#     print("-" * 46)
#     for rxn_id, min_val, max_val in results:
#         print("{:<20} {:>12.4g} {:>12.4g}".format(rxn_id, min_val if min_val is not None else float('nan'), max_val if max_val is not None else float('nan')))

#     # 4) stitch results back into a DataFrame
#     reaction_ids, min_vals, max_vals = zip(*results)
#     df_FVA = pd.DataFrame({
#         "Reactions": reaction_ids,
#         "Min Solutions": min_vals,
#         "Max Solutions": max_vals,
#         "Solution Biomass": [sol_biomass] * len(results)
#     })

#     # 5) optionally save
#     if output_file:
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         df_FVA.to_csv(output_file, index=False)

#     client.close()
#     return df_FVA, processed_df, df_FBA


# ============================================================================
# IMPROVED PARALLEL FVA IMPLEMENTATIONS
# ============================================================================

def _fva_for_reaction_chunk(model, processed_df, rxn_ids, biomass_bounds,
                            enzyme_upper_bound, enzyme_ratio,
                            multi_enzyme_off, isoenzymes_off,
                            promiscuous_off, complexes_off, chunk_id=None, total_chunks=None):
    """
    Process multiple reactions in one task (for chunked parallelization).
    Reduces task overhead by batching reactions together.
    """
    if chunk_id is not None and total_chunks is not None:
        print(f"  [Chunk {chunk_id:2d}/{total_chunks}] Processing {len(rxn_ids)} reactions...")

    results = []
    for i, rxn_id in enumerate(rxn_ids):
        result = _fva_for_reaction(
            model, processed_df, rxn_id, biomass_bounds,
            enzyme_upper_bound, enzyme_ratio,
            multi_enzyme_off, isoenzymes_off,
            promiscuous_off, complexes_off
        )
        results.append(result)

        # Progress within chunk (only for large chunks)
        if len(rxn_ids) >= 10 and (i + 1) % max(1, len(rxn_ids) // 4) == 0:
            progress = ((i + 1) / len(rxn_ids)) * 100
            if chunk_id is not None:
                print(f"    [Chunk {chunk_id:2d}] {progress:5.1f}% complete ({i+1}/{len(rxn_ids)} reactions)")

    if chunk_id is not None and total_chunks is not None:
        print(f"  [Chunk {chunk_id:2d}/{total_chunks}] ✓ Complete ({len(rxn_ids)} reactions)")

    return results


def flux_variability_analysis_parallel_chunked(model, processed_df, biomass_reaction,
                               output_file=None,
                               enzyme_upper_bound=0.15, opt_ratio=0.9, enzyme_ratio=True,
                               multi_enzyme_off=False, isoenzymes_off=False,
                               promiscuous_off=False, complexes_off=False,
                               n_workers=None, chunk_size=None, method='dask'):
    """
    Perform Flux Variability Analysis (FVA) in parallel with chunking support.

    Supports two parallelization backends:
    - 'dask': Uses Dask for distributed computing (default)
    - 'multiprocessing': Uses Python's multiprocessing.Pool (simpler, single-machine)

    Parameters
    ----------
    model : cobra.Model
        COBRA model object
    processed_df : pandas.DataFrame
        DataFrame with enzyme constraints data
    biomass_reaction : str
        ID of the biomass reaction
    output_file : str, optional
        File path to save FVA results
    enzyme_upper_bound : float, optional
        Enzyme pool constraint (default: 0.15)
    opt_ratio : float, optional
        Fraction of optimal biomass to constrain during FVA (default: 0.9)
    enzyme_ratio : bool, optional
        Whether to apply enzyme ratio constraint (default: True)
    multi_enzyme_off, isoenzymes_off, promiscuous_off, complexes_off : bool
        Logic switches for model complexity
    n_workers : int, optional
        Number of parallel workers (default: number of CPU cores)
    chunk_size : int, optional
        Number of reactions per chunk (default: auto-calculated)
    method : str, optional
        Parallelization method: 'dask' or 'multiprocessing' (default: 'dask')

    Returns
    -------
    tuple
        (df_FVA_solution, processed_df, df_FBA)
    """
    import sys

    from .optimize import run_optimization_with_dataframe

    # Determine number of workers
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    # Calculate optimal chunk size if not provided
    n_reactions = len(model.reactions)
    if chunk_size is None:
        # Aim for ~10-20 chunks per worker for good load balancing
        chunk_size = max(1, n_reactions // (n_workers * 15))

    print("  Parallel FVA configuration:")
    print(f"    Method: {method}")
    print(f"    Workers: {n_workers}")
    print(f"    Reactions: {n_reactions}")
    print(f"    Chunk size: {chunk_size}")

    n_chunks = (n_reactions + chunk_size - 1) // chunk_size
    print(f"    Number of chunks: {n_chunks}")

    # Estimate memory usage
    # Rough estimate: COBRApy models are ~100-200 MB per copy in memory
    # For enzyme-constrained models, add overhead from processed_df
    try:
        # Estimate based on model size (reactions, metabolites, genes)
        base_model_mb = (len(model.reactions) * 0.05 +  # ~50 KB per reaction
                        len(model.metabolites) * 0.02 +  # ~20 KB per metabolite
                        len(model.genes) * 0.01)         # ~10 KB per gene

        # Add overhead for processed_df (roughly 1 KB per row)
        if processed_df is not None:
            df_size_mb = len(processed_df) * 0.001
        else:
            df_size_mb = 0

        model_size_mb = base_model_mb + df_size_mb
        estimated_memory_gb = (model_size_mb * n_workers) / 1000
        print(f"    Estimated memory: ~{estimated_memory_gb:.1f} GB")
        if estimated_memory_gb > 8:
            print("    ⚠️  Warning: High memory usage expected")
    except Exception:
        pass  # Skip memory estimation if it fails

    # 1) Run baseline optimization to fix biomass
    print("\n  Running baseline optimization...")
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
        verbose=True
    )
    print(f"  Optimal biomass: {sol_biomass:.6f}")
    print(f"  Biomass constraint: {sol_biomass * opt_ratio:.6f} ≤ biomass ≤ {sol_biomass:.6f} (opt_ratio={opt_ratio})")

    biomass_bounds = (biomass_reaction, sol_biomass * opt_ratio, sol_biomass)

    # 2) Create reaction chunks
    reaction_ids = [rxn.id for rxn in model.reactions]
    chunks = [reaction_ids[i:i+chunk_size]
              for i in range(0, len(reaction_ids), chunk_size)]

    print(f"\n  Starting parallel FVA with {n_chunks} chunks...")
    print("  Progress will be shown by chunk completion...")

    # 3) Execute in parallel based on method
    start_time = time.time()

    if method.lower() == 'multiprocessing':
        results = _run_fva_multiprocessing(
            model, processed_df, chunks, biomass_bounds,
            enzyme_upper_bound, enzyme_ratio,
            multi_enzyme_off, isoenzymes_off,
            promiscuous_off, complexes_off,
            n_workers
        )
    else:  # dask (default)
        results = _run_fva_dask(
            model, processed_df, chunks, biomass_bounds,
            enzyme_upper_bound, enzyme_ratio,
            multi_enzyme_off, isoenzymes_off,
            promiscuous_off, complexes_off,
            n_workers
        )

    # Record execution time
    execution_time = time.time() - start_time

    # 4) Flatten results (each chunk returns a list of tuples)
    flat_results = []
    for chunk_results in results:
        flat_results.extend(chunk_results)

    # Print completion summary
    total_reactions = len(flat_results)
    successful_reactions = sum(1 for _, min_val, max_val in flat_results
                              if min_val is not None and max_val is not None)
    failed_reactions = total_reactions - successful_reactions

    print(f"\n  ✓ FVA Complete: {total_reactions} reactions processed")
    print(f"    - Successful: {successful_reactions}")
    if failed_reactions > 0:
        print(f"    - Failed: {failed_reactions}")
    print(f"    - Success rate: {(successful_reactions/total_reactions)*100:.1f}%")
    print(f"    - Execution time: {execution_time/60:.1f} minutes ({execution_time:.1f} seconds)")
    if successful_reactions > 0:
        print(f"    - Average time per reaction: {execution_time/successful_reactions:.2f} seconds")

    # 5) Create DataFrame
    reaction_ids, min_vals, max_vals = zip(*flat_results)
    df_FVA = pd.DataFrame({
        "Reactions": reaction_ids,
        "Min Solutions": min_vals,
        "Max Solutions": max_vals,
        "Solution Biomass": [sol_biomass] * len(flat_results)
    })

    # 6) Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_FVA.to_csv(output_file, index=False)
        print(f"\n  FVA results saved to: {output_file}")

    return df_FVA, processed_df, df_FBA


def _run_fva_dask(model, processed_df, chunks, biomass_bounds,
                 enzyme_upper_bound, enzyme_ratio,
                 multi_enzyme_off, isoenzymes_off,
                 promiscuous_off, complexes_off,
                 n_workers):
    """Execute FVA using Dask with chunking."""
    try:
        from dask import compute, delayed
        from dask.distributed import Client
    except ImportError:
        raise ImportError(
            "Dask is required for parallel FVA. Install with: pip install dask[distributed]"
        )

    # Create Dask client
    try:
        client = Client(
            n_workers=n_workers,
            processes=True,
            threads_per_worker=1,
            silence_logs=logging.ERROR
        )
        # Try to print dashboard link, but don't fail if bokeh is missing
        try:
            print(f"  Dask dashboard: {client.dashboard_link}")
        except Exception:
            print("  Dask dashboard: (install bokeh>=3.1.0 to enable dashboard)")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not start Dask client: {e}")
        print("  Falling back to sequential execution...")
        client = None

    # Create delayed tasks
    tasks = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        tasks.append(
            delayed(_fva_for_reaction_chunk)(
                model.copy(),
                processed_df,
                chunk,
                biomass_bounds,
                enzyme_upper_bound, enzyme_ratio,
                multi_enzyme_off, isoenzymes_off,
                promiscuous_off, complexes_off,
                i, total_chunks  # chunk_id and total_chunks for progress tracking
            )
        )

    # Execute
    try:
        results = compute(*tasks)
    finally:
        if client:
            client.close()

    return results


def _run_fva_multiprocessing(model, processed_df, chunks, biomass_bounds,
                             enzyme_upper_bound, enzyme_ratio,
                             multi_enzyme_off, isoenzymes_off,
                             promiscuous_off, complexes_off,
                             n_workers):
    """Execute FVA using multiprocessing.Pool."""
    from functools import partial
    from multiprocessing import Pool

    def worker_func(chunk_info):
        """Worker function that includes chunk progress info."""
        chunk, chunk_id, total_chunks = chunk_info
        return _fva_for_reaction_chunk(
            model.copy(),  # Each worker gets model copy
            processed_df,
            chunk,
            biomass_bounds,
            enzyme_upper_bound, enzyme_ratio,
            multi_enzyme_off, isoenzymes_off,
            promiscuous_off, complexes_off,
            chunk_id, total_chunks
        )

    # Create chunk info tuples (chunk, chunk_id, total_chunks)
    total_chunks = len(chunks)
    chunk_infos = [(chunk, i+1, total_chunks) for i, chunk in enumerate(chunks)]

    # Execute in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_func, chunk_infos)

    return results


# Wrapper function that chooses implementation based on config
def flux_variability_analysis_parallel(model, processed_df, biomass_reaction,
                                      output_file=None,
                                      enzyme_upper_bound=0.15, opt_ratio=0.9, enzyme_ratio=True,
                                      multi_enzyme_off=False, isoenzymes_off=False,
                                      promiscuous_off=False, complexes_off=False,
                                      n_workers=None, chunk_size=None, method='dask'):
    """
    Parallel FVA wrapper that auto-selects chunked implementation.

    This is the main function to call for parallel FVA. It automatically uses
    chunking for optimal performance.

    For backward compatibility, this function has the same signature as the
    original flux_variability_analysis_parallel but now uses chunking by default.
    """
    return flux_variability_analysis_parallel_chunked(
        model=model,
        processed_df=processed_df,
        biomass_reaction=biomass_reaction,
        output_file=output_file,
        enzyme_upper_bound=enzyme_upper_bound,
        opt_ratio=opt_ratio,
        enzyme_ratio=enzyme_ratio,
        multi_enzyme_off=multi_enzyme_off,
        isoenzymes_off=isoenzymes_off,
        promiscuous_off=promiscuous_off,
        complexes_off=complexes_off,
        n_workers=n_workers,
        chunk_size=chunk_size,
        method=method
    )

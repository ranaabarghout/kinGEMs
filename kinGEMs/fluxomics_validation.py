"""
Validate kinGEMs metabolic reaction flux predictions against experimental data.

This module provides functions to:
- Modify the iML1515 GEM  so its medium and stress conditions match the experimental conditions.
- Create a dataframe that combines experimental fluxomics data with kinGEMs results.
"""

import pandas as pd
import cobra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def compare_fluxomics(fba_results_path: str, exp_fluxes_path: str) -> pd.DataFrame:
    """
    Compare FBA simulation results with experimental fluxomics data.
    
    Parameters:
        fba_results_path : str
            Path to the CSV file with FBA results (df_FBA.csv format)
            Expected columns: 'Variable', 'Index', 'Value'
        exp_fluxes_path : str
            Path to the CSV file with experimental fluxes
            Expected columns: 'rxn_id', 'exp_reaction', 'exp_flux'
    
    Returns:
        pd.DataFrame: DataFrame with columns: 'rxn_id', 'exp_reaction', 'exp_flux', 'FBA_flux'
    """
    
    # Load FBA results and filter for flux variables
    fba_df = pd.read_csv(fba_results_path)
    
    # Filter for flux variables only
    flux_df = fba_df[fba_df['Variable'] == 'flux'].copy()
    
    # Rename columns to match expected output format
    flux_df = flux_df.rename(columns={'Index': 'rxn_id', 'Value': 'FBA_flux'})
    
    # Keep only the columns we need
    flux_df = flux_df[['rxn_id', 'FBA_flux']]
    
    # Load experimental fluxes
    exp_df = pd.read_csv(exp_fluxes_path)
    
    # Merge the dataframes on rxn_id
    result_df = exp_df.merge(flux_df, on='rxn_id', how='right')
    
    # Reorder columns to match expected output
    result_df = result_df[['rxn_id', 'exp_reaction', 'exp_flux', 'FBA_flux']]
    
    print(f"Loaded {len(flux_df)} FBA flux results")
    print(f"Loaded {len(exp_df)} experimental flux measurements")
    print(f"Merged dataframe has {len(result_df)} rows")
    print(f"Matched reactions: {result_df['exp_reaction'].notna().sum()}")
    print(f"Unmatched reactions: {result_df['exp_reaction'].isna().sum()}")
    
    return result_df


def create_fva_comparison_dataframe(
    fva_results_path: str,
    mfa_results_path: str,
    fva_columns: list[str],
    mfa_columns: list[str]
) -> pd.DataFrame:
    """
    Compare FVA simulation results with MFA experimental data.
    
    Parameters:
        fva_results_path : str
            Path to the CSV file with FVA results
        mfa_results_path : str
            Path to the CSV file with MFA experimental fluxes
        fva_columns : list[str]
            List of 3 column names for FVA file: [rxn_id, lb_flux, ub_flux]
        mfa_columns : list[str]
            List of 3 column names for MFA file: [rxn_id, lb_flux, ub_flux]
    
    Returns:
        pd.DataFrame: DataFrame with columns: 'rxn_id', 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub'
    """
    
    # Load FVA results
    fva_df = pd.read_csv(fva_results_path)
    
    # Extract and rename FVA columns
    fva_rxn_id_col, fva_lb_col, fva_ub_col = fva_columns
    fva_df = fva_df[[fva_rxn_id_col, fva_lb_col, fva_ub_col]].copy()
    fva_df = fva_df.rename(columns={
        fva_rxn_id_col: 'rxn_id',
        fva_lb_col: 'fva_lb',
        fva_ub_col: 'fva_ub'
    })
    
    # Load MFA results
    mfa_df = pd.read_csv(mfa_results_path)
    
    # Extract and rename MFA columns
    mfa_rxn_id_col, mfa_lb_col, mfa_ub_col = mfa_columns
    mfa_df = mfa_df[[mfa_rxn_id_col, mfa_lb_col, mfa_ub_col]].copy()
    mfa_df = mfa_df.rename(columns={
        mfa_rxn_id_col: 'rxn_id',
        mfa_lb_col: 'mfa_lb',
        mfa_ub_col: 'mfa_ub'
    })
    
    # Merge the dataframes on rxn_id, keeping all FVA rows
    result_df = fva_df.merge(mfa_df, on='rxn_id', how='left')
    
    # Reorder columns
    result_df = result_df[['rxn_id', 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub']]
    
    print(f"Loaded {len(fva_df)} FVA flux results")
    print(f"Loaded {len(mfa_df)} MFA flux measurements")
    print(f"Merged dataframe has {len(result_df)} rows")
    print(f"Matched reactions: {result_df['mfa_lb'].notna().sum()}")
    print(f"Unmatched reactions: {result_df['mfa_lb'].isna().sum()}\n")
    
    return result_df


def calculate_consistency_score(
    comparison_df: pd.DataFrame
) -> float:
    """
    Calculate the Consistency Score (fraction of overlapping ranges).
    
    Parameters:
        comparison_df : pd.DataFrame
            DataFrame with columns: 'rxn_id', 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub'
    
    Returns:
        float: The fraction of reactions (0.0 to 1.0) where FVA and MFA ranges overlap.
    """
    
    # Drop rows where MFA data is missing (cannot compare)
    df = comparison_df.dropna(subset=['mfa_lb', 'mfa_ub']).copy()
    
    # Determine overlap
    # Overlap exists if the lower bound of one is <= the upper bound of the other
    # Logic: max(lb1, lb2) <= min(ub1, ub2)
    overlap_lower = df[['fva_lb', 'mfa_lb']].max(axis=1)
    overlap_upper = df[['fva_ub', 'mfa_ub']].min(axis=1)
    
    consistent_mask = overlap_lower <= overlap_upper
    score = consistent_mask.mean()
    
    print(f"\n--- Consistency Score Analysis ---")
    print(f"Evaluated {len(df)} reactions")
    print(f"Consistent reactions: {consistent_mask.sum()}")
    print(f"Consistency Score: {score:.4f}")
    
    return score


def calculate_range_precision_ratio(
    comparison_df: pd.DataFrame
) -> pd.Series:
    """
    Calculate the Ratio of FVA width to MFA width.
    A value < 1.0 indicates the model is more precise than the experiment.
    
    Parameters:
        comparison_df : pd.DataFrame
            DataFrame with columns: 'rxn_id', 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub'
            
    Returns:
        pd.Series: Series containing the width ratio for each reaction.
    """
    
    df = comparison_df.dropna(subset=['mfa_lb', 'mfa_ub']).copy()
    
    # Calculate widths
    fva_width = df['fva_ub'] - df['fva_lb']
    mfa_width = df['mfa_ub'] - df['mfa_lb']
    
    # Avoid division by zero by adding a small epsilon where width is 0
    epsilon = 1e-9
    mfa_width = mfa_width.replace(0, epsilon)
    
    ratio = fva_width / mfa_width
    
    print(f"\n--- Range Precision Analysis ---")
    print(f"Evaluated {len(df)} reactions")
    print(f"Median FVA width: {fva_width.median():.4f}")
    print(f"Median MFA width: {mfa_width.median():.4f}")
    print(f"Median Precision Ratio: {ratio.median():.4f}")
    
    return ratio


def calculate_normalized_euclidean_dist(
    comparison_df: pd.DataFrame
) -> float:
    """
    Calculate the Normalized Euclidean Distance (Sum of Squared Distances).
    Measures the distance from the MFA mean to the nearest FVA bound.
    
    Parameters:
        comparison_df : pd.DataFrame
            DataFrame with columns: 'rxn_id', 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub'
            
    Returns:
        float: The Sum of Squared Distances (SSD) across all reactions.
    """
    
    df = comparison_df.dropna(subset=['mfa_lb', 'mfa_ub']).copy()
    
    # Calculate MFA mean
    mfa_mean = (df['mfa_lb'] + df['mfa_ub']) / 2
    
    # Calculate distance to range
    # If mean is inside [fva_lb, fva_ub], distance is 0
    # If mean < fva_lb, distance is fva_lb - mean
    # If mean > fva_ub, distance is mean - fva_ub
    # This can be vectorized as: max(0, fva_lb - mean, mean - fva_ub)
    
    dist_lower = df['fva_lb'] - mfa_mean
    dist_upper = mfa_mean - df['fva_ub']
    zero_baseline = pd.Series(0, index=df.index)
    
    # Find the max of (0, dist_lower, dist_upper)
    distance = pd.concat([zero_baseline, dist_lower, dist_upper], axis=1).max(axis=1)
    
    # Calculate Sum of Squared Errors
    ssd = (distance ** 2).sum()
    
    print(f"\n--- Euclidean Distance Analysis ---")
    print(f"Evaluated {len(df)} reactions")
    print(f"Reactions with mean outside FVA range: {(distance > 0).sum()}")
    print(f"Sum of Squared Distances (SSD): {ssd:.4f}")
    
    return ssd


def calculate_jaccard_index(
    comparison_df: pd.DataFrame
) -> tuple[float, pd.DataFrame]:
    """
    Calculate the mean Jaccard Index (Intersection over Union).
    1.0 = Perfect overlap, 0.0 = No overlap.
    
    Parameters:
        comparison_df : pd.DataFrame
            DataFrame with columns: 'rxn_id', 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub'
            
    Returns:
        tuple[float, pd.DataFrame]: 
            - float: The average Jaccard Index across all evaluated reactions.
            - pd.DataFrame: DataFrame with columns 'rxn_id' and 'jaccard' for each reaction.
    """
    
    df = comparison_df.dropna(subset=['mfa_lb', 'mfa_ub']).copy()
    
    # Calculate Intersection
    # Max of lower bounds, Min of upper bounds
    inter_lb = df[['fva_lb', 'mfa_lb']].max(axis=1)
    inter_ub = df[['fva_ub', 'mfa_ub']].min(axis=1)
    
    # Intersection width (clip negative values to 0 for non-overlapping cases)
    intersection = (inter_ub - inter_lb).clip(lower=0)
    
    # Calculate Union
    # Min of lower bounds, Max of upper bounds
    union_lb = df[['fva_lb', 'mfa_lb']].min(axis=1)
    union_ub = df[['fva_ub', 'mfa_ub']].max(axis=1)
    union = union_ub - union_lb
    
    # Avoid division by zero
    epsilon = 1e-9
    union = union.replace(0, epsilon)
    
    jaccard = intersection / union
    mean_jaccard = jaccard.mean()
    
    # Create DataFrame with rxn_id and jaccard values
    jaccard_df = pd.DataFrame({
        'rxn_id': df['rxn_id'],
        'jaccard': jaccard.values
    }).reset_index(drop=True)
    
    print(f"\n--- Jaccard Index Analysis ---")
    print(f"Evaluated {len(df)} reactions")
    print(f"Perfect overlaps (J=1.0): {(jaccard >= 0.99).sum()}")
    print(f"Zero overlaps (J=0.0): {(jaccard <= 0.01).sum()}")
    print(f"Mean Jaccard Index: {mean_jaccard:.4f}")
    
    return mean_jaccard, jaccard_df

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_fva_mfa_comparison(
    models_data: dict[str, pd.DataFrame],
    split_charts: bool = True,
    reactions_per_plot: int = 25
) -> None:
    """
    Generates a bar plot comparing MFA experimental ranges against 
    FVA prediction ranges for multiple models.
    
    Parameters:
        models_data : dict[str, pd.DataFrame]
            A dictionary where keys are model names and values are the DataFrames obtained with create_fva_comparison_dataframe()
        split_charts : bool
            If True, splits the plot into multiple figures if reaction count exceeds reactions_per_plot.
        reactions_per_plot : int
            Number of reactions to show per figure chunk.
    """
    
    # 1. Consolidate Data
    # Take the first dataframe as the master for the reaction list and MFA values.
    primary_name = list(models_data.keys())[0]
    master_df = models_data[primary_name].copy()
    
    # Filter for reactions that have MFA data
    master_df = master_df.dropna(subset=['mfa_lb', 'mfa_ub'])
    all_reactions = master_df['rxn_id'].unique()
    
    # 2. Setup Visualization Constants
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    bar_height = 0.6
    
    # Calculate offsets to 'dodge' the bars if multiple models exist
    num_models = len(models_data)
    # If 1 model, offset is 0. If 3, offsets might be [-0.2, 0, 0.2]
    offsets = [i - (num_models - 1) / 2 for i in range(num_models)]
    # Scale offsets to fit within a row height of 1.0
    step_size = 0.6 / max(num_models, 1)
    offsets = [o * step_size for o in offsets]
    
    # 3. Chunking Logic (to handle large lists of reactions)
    chunks = [all_reactions]
    if split_charts and len(all_reactions) > reactions_per_plot:
        chunks = [all_reactions[i:i + reactions_per_plot] 
                  for i in range(0, len(all_reactions), reactions_per_plot)]
        print(f"Splitting visualization into {len(chunks)} plots for readability.")

    # 4. Plotting Loop
    for chunk_idx, rxn_chunk in enumerate(chunks):
        
        # Dynamic height: 0.5 inch per reaction + buffer
        fig_height = len(rxn_chunk) * 0.5 + 2
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        # Iterate through reactions in this chunk
        for i, rxn in enumerate(rxn_chunk):
            y_pos = i  # Base Y position for this reaction
            
            # A. Plot MFA Reference (Black Error Bar)
            # We get MFA data from the master_df
            row = master_df[master_df['rxn_id'] == rxn].iloc[0]
            mfa_center = (row['mfa_lb'] + row['mfa_ub']) / 2
            mfa_err = (row['mfa_ub'] - row['mfa_lb']) / 2
            
            # Plot the "Target" zone
            ax.errorbar(
                x=mfa_center, y=y_pos, xerr=mfa_err, 
                color='black', capsize=5, elinewidth=2, markeredgewidth=2,
                zorder=10, label='MFA Experiment' if i == 0 else ""
            )
            
            # B. Plot Each Model's FVA Range
            for model_idx, (name, df) in enumerate(models_data.items()):
                # Find the row for this reaction in this specific model's DF
                model_row = df[df['rxn_id'] == rxn]
                
                if not model_row.empty:
                    fva_lb = model_row.iloc[0]['fva_lb']
                    fva_ub = model_row.iloc[0]['fva_ub']
                    
                    # Calculate dodged Y position
                    y_offset = y_pos + offsets[model_idx]
                    
                    # Draw the Range Bar
                    ax.hlines(
                        y=y_offset, xmin=fva_lb, xmax=fva_ub,
                        color=colors[model_idx % len(colors)],
                        linewidth=4, alpha=0.8,
                        label=name if i == 0 else ""
                    )
                    
                    # Draw faint guideline across the plot for readability
                    ax.axhline(y=y_pos, color='gray', linestyle=':', alpha=0.1, linewidth=0.5)

        # 5. Formatting
        ax.set_yticks(range(len(rxn_chunk)))
        ax.set_yticklabels(rxn_chunk, fontsize=10, fontfamily='monospace')
        ax.invert_yaxis()  # Top-down list
        ax.set_xlabel('Flux (mmol/gDW/h)')
        ax.set_title(f'MFA vs FVA Range Comparison')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Create a custom legend to avoid duplicates
        handles = [
            mlines.Line2D([], [], color='black', marker='|', markersize=10, 
                          linestyle='-', linewidth=2, label='MFA Range')
        ]
        for idx, name in enumerate(models_data.keys()):
            handles.append(mlines.Line2D([], [], color=colors[idx], linewidth=4, label=name))
            
        ax.legend(handles=handles, loc='upper right', frameon=True)
        
        plt.tight_layout()
        plt.show()

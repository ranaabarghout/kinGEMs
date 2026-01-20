"""
Uitilitary functions to validate kinGEMs metabolic reaction flux predictions against experimental data.
"""

import pandas as pd
import cobra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def create_net_FVA_dataframe(irreversible_fva_path: str) -> pd.DataFrame:
    """
    Create a dataframe with the net FVA results by collapsing split reversible reactions.
    
    Parameters:
        irreversible_fva_path : str
            Path to the CSV file with irreversible model FVA results
    
    Returns:
        pd.DataFrame: Dataframe containing 'rxn_id', 'fva_lb', and 'fva_ub' for net fluxes.
    """
    
    # Load irreversible model FVA results
    irreversible_fva_df = pd.read_csv(irreversible_fva_path)
    
    # Extract and rename FVA columns
    df = irreversible_fva_df[['Reactions', 'Min Solutions', 'Max Solutions']].copy()
    df = df.rename(columns={
        'Reactions': 'rxn_id',
        'Min Solutions': 'fva_lb_irrev',
        'Max Solutions': 'fva_ub_irrev'
    })
    
    print(f"Irreversible model FVA dataframe has {len(df)} rows")
    
    # Identify reverse reactions and base IDs
    df['is_reverse'] = df['rxn_id'].str.endswith('_reverse')
    
    # Create a 'base_id' column to join on
    df['base_id'] = df['rxn_id'].str.replace(r'_reverse$', '', regex=True)
    
    # Split into forward and reverse dataframes
    df_fwd = df[~df['is_reverse']].copy()
    df_rev = df[df['is_reverse']].copy()
    
    # Merge forward and reverse dataframes on 'base_id'
    merged_df = pd.merge(
        df_fwd, 
        df_rev[['base_id', 'fva_lb_irrev', 'fva_ub_irrev']], 
        on='base_id', 
        how='left', 
        suffixes=('', '_rev')
    )
    
    # Handle missing reverse values
    # (fill with 0.0 because an irreversible reaction has 0 reverse flux)
    merged_df['fva_lb_irrev_rev'] = merged_df['fva_lb_irrev_rev'].fillna(0.0)
    merged_df['fva_ub_irrev_rev'] = merged_df['fva_ub_irrev_rev'].fillna(0.0)
    
    # Calculate net FVA results
    # net_LB = Min(Forward) - Max(Reverse)
    # net_UB = Max(Forward) - Min(Reverse)
    merged_df['fva_lb'] = merged_df['fva_lb_irrev'] - merged_df['fva_ub_irrev_rev']
    merged_df['fva_ub'] = merged_df['fva_ub_irrev'] - merged_df['fva_lb_irrev_rev']
    
    # Keep relevant columns
    final_df = merged_df[['base_id', 'fva_lb', 'fva_ub']].rename(columns={'base_id': 'rxn_id'})
    
    print(f"Net FVA dataframe has {len(final_df)} rows")
    
    return final_df
    

def create_fva_comparison_dataframe(
    fva_results_path: str,
    mfa_results_path: str,
    fva_columns: list[str] | None = None,
    mfa_columns: list[str] = None
) -> pd.DataFrame:
    """
    Compare FVA simulation results with MFA experimental data.
    
    Parameters:
        fva_results_path : str or pd.DataFrame
            Path to the CSV file with FVA results
        mfa_results_path : str
            Path to the CSV file with MFA experimental fluxes
        fva_columns : list[str], optional
            List of 3 column names for FVA file: [rxn_id, lb_flux, ub_flux]
        mfa_columns : list[str]
            List of 3 column names for MFA file: [rxn_id, lb_flux, ub_flux]
    
    Returns:
        pd.DataFrame: DataFrame with columns: 'rxn_id', 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub'
    """
    
    # Load FVA results (if not a dataframe, read it from a CSV file)
    if not isinstance(fva_results_path, pd.DataFrame):
        fva_df = pd.read_csv(fva_results_path)

        if fva_columns is None:
            raise ValueError("fva_columns must be provided when fva_results_path is a CSV path")

        fva_rxn_id_col, fva_lb_col, fva_ub_col = fva_columns
        fva_df = fva_df[[fva_rxn_id_col, fva_lb_col, fva_ub_col]].copy()
        fva_df = fva_df.rename(columns={
            fva_rxn_id_col: 'rxn_id',
            fva_lb_col: 'fva_lb',
            fva_ub_col: 'fva_ub'
        })
    else:
        fva_df = fva_results_path.copy()
    
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
    
    print(f"\nConsistency Score Analysis")
    print(f"   Evaluated {len(df)} reactions")
    print(f"   Consistent reactions: {consistent_mask.sum()}")
    print(f"   Consistency Score: {score:.4f}")
    
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
    
    print(f"\nRange Precision Analysis")
    print(f"   Evaluated {len(df)} reactions")
    print(f"   Median FVA width: {fva_width.median():.4f}")
    print(f"   Median MFA width: {mfa_width.median():.4f}")
    print(f"   Median Precision Ratio: {ratio.median():.4f}")
    
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
    
    print(f"\nEuclidean Distance Analysis")
    print(f"   Evaluated {len(df)} reactions")
    print(f"   Reactions with mean outside FVA range: {(distance > 0).sum()}")
    print(f"   Sum of Squared Distances (SSD): {ssd:.4f}")
    
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
        tuple[float, pd.DataFrame, int]: 
            - float: The average Jaccard Index across all evaluated reactions.
            - pd.DataFrame: DataFrame with columns 'rxn_id' and 'jaccard' for each reaction.
            - int: Number of reactions with zero overlap.
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
    
    zero_overlaps = jaccard[jaccard == 0].count()
    
    print(f"\nJaccard Index Analysis")
    print(f"   Evaluated {len(df)} reactions")
    print(f"   Perfect overlaps (J=1.0): {(jaccard >= 0.99).sum()}")
    print(f"   Zero overlaps (J=0.0): {zero_overlaps}")
    print(f"   Mean Jaccard Index: {mean_jaccard:.4f}")
    
    return mean_jaccard, jaccard_df, zero_overlaps


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
    
    # Calculate offsets to 'dodge' the bars
    # MFA gets the top position, then models are stacked below
    num_models = len(models_data)
    total_bars = num_models + 1  # +1 for MFA
    
    # Create offsets: MFA at top, then models below
    step_size = 0.7 / total_bars  # Spread across 0.7 units
    mfa_offset = -0.35 + step_size / 2  # Top position
    model_offsets = [mfa_offset + (i + 1) * step_size for i in range(num_models)]
    
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
            
            # A. Plot MFA Reference (Black Range Bar)
            # We get MFA data from the master_df
            row = master_df[master_df['rxn_id'] == rxn].iloc[0]
            mfa_lb = row['mfa_lb']
            mfa_ub = row['mfa_ub']
            
            # Plot the "Target" zone as a horizontal range bar (independent position)
            ax.hlines(
                y=y_pos + mfa_offset, xmin=mfa_lb, xmax=mfa_ub,
                color='black', linewidth=4, alpha=0.8,
                zorder=10, label='MFA Experiment' if i == 0 else ""
            )
            
            # B. Plot Each Model's FVA Range
            for model_idx, (name, df) in enumerate(models_data.items()):
                # Find the row for this reaction in this specific model's DF
                model_row = df[df['rxn_id'] == rxn]
                
                if not model_row.empty:
                    fva_lb = model_row.iloc[0]['fva_lb']
                    fva_ub = model_row.iloc[0]['fva_ub']
                    
                    # Calculate dodged Y position (independent from MFA)
                    y_offset = y_pos + model_offsets[model_idx]
                    
                    # Draw the Range Bar
                    ax.hlines(
                        y=y_offset, xmin=fva_lb, xmax=fva_ub,
                        color=colors[model_idx % len(colors)],
                        linewidth=4, alpha=0.8,
                        label=name if i == 0 else ""
                    )
            
            # Guideline across the plot for readability
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
            mlines.Line2D([], [], color='black', linewidth=4, label='MFA')
        ]
        for idx, name in enumerate(models_data.keys()):
            handles.append(mlines.Line2D([], [], color=colors[idx], linewidth=4, label=name))
            
        ax.legend(handles=handles, loc='upper right', frameon=True)
        
        plt.tight_layout()
        plt.show()
    

def plot_fva_mfa_comparison_normalized(
    models_data: dict[str, pd.DataFrame],
    split_charts: bool = True,
    reactions_per_plot: int = 25,
    zoom_limit: float = 5.0  
) -> None:
    """
    Generates a normalized bar plot centered on the MFA midpoint.
    
    The X-axis represents 'MFA Range Units'. 
    - 0 is the MFA midpoint.
    - The MFA range always spans from -0.5 to +0.5.
    - FVA ranges are scaled relative to the MFA width.
    
    Parameters:
        models_data : dict[str, pd.DataFrame]
            A dictionary where keys are model names and values are the DataFrames 
            obtained with create_fva_comparison_dataframe()
        split_charts : bool
            If True, splits the plot into multiple figures if reaction count 
            exceeds reactions_per_plot.
        reactions_per_plot : int
            Number of reactions to show per figure chunk.
        zoom_limit : float
            Limits the x-axis to +/- zoom_limit * the MFA range width.
    """    
    # Consolidate data
    primary_name = list(models_data.keys())[0]
    master_df = models_data[primary_name].copy()
    
    master_df = master_df.dropna(subset=['mfa_lb', 'mfa_ub'])
    all_reactions = master_df['rxn_id'].unique()
    
    # Calculate normalization factors
    normalization_factors = {}
    for rxn in all_reactions:
        row = master_df[master_df['rxn_id'] == rxn].iloc[0]
        mfa_lb = row['mfa_lb']
        mfa_ub = row['mfa_ub']
        
        mfa_midpoint = (mfa_lb + mfa_ub) / 2
        mfa_range = mfa_ub - mfa_lb
        
        # Handle near-zero ranges to avoid explosion
        norm_factor = mfa_range if abs(mfa_range) > 1e-6 else 1.0
        
        normalization_factors[rxn] = {
            'factor': norm_factor,
            'offset': mfa_midpoint
        }
    
    # Setup visualization
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    
    # Calculate offsets
    num_models = len(models_data)
    total_bars = num_models + 1
    step_size = 0.8 / total_bars # Slightly increased spacing
    mfa_offset = -0.4 + step_size / 2
    model_offsets = [mfa_offset + (i + 1) * step_size for i in range(num_models)]
    
    # Chunking
    chunks = [all_reactions]
    if split_charts and len(all_reactions) > reactions_per_plot:
        chunks = [all_reactions[i:i + reactions_per_plot] 
                  for i in range(0, len(all_reactions), reactions_per_plot)]
        print(f"Splitting visualization into {len(chunks)} plots.")

    # Plotting loop
    for chunk_idx, rxn_chunk in enumerate(chunks):
        
        fig_height = len(rxn_chunk) * 0.6 + 2
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        for i, rxn in enumerate(rxn_chunk):
            y_pos = i
            norm_factor = normalization_factors[rxn]['factor']
            norm_offset = normalization_factors[rxn]['offset']
            
            # --- A. Plot MFA Reference (Always -0.5 to 0.5) ---
            # Hardcode -0.5 to 0.5 because we are normalizing by its own width
            ax.hlines(
                y=y_pos + mfa_offset, 
                xmin=-0.5, 
                xmax=0.5,
                color='black', 
                linewidth=5, 
                zorder=20, # Ensure it sits on top
                label='MFA Range' if i == 0 else ""
            )
            
            # Add a white tick at the midpoint (0) to help visibility
            ax.plot(0, y_pos + mfa_offset, '|', color='white', markersize=10, zorder=21)

            # --- B. Plot Each Model's FVA Range ---
            for model_idx, (name, df) in enumerate(models_data.items()):
                model_row = df[df['rxn_id'] == rxn]
                
                if not model_row.empty:
                    fva_lb = model_row.iloc[0]['fva_lb']
                    fva_ub = model_row.iloc[0]['fva_ub']
                    
                    # Normalize
                    fva_lb_norm = (fva_lb - norm_offset) / norm_factor
                    fva_ub_norm = (fva_ub - norm_offset) / norm_factor
                    
                    # Capping for visual cleanliness (optional, prevents infinity issues)
                    draw_min = max(fva_lb_norm, -zoom_limit * 1.5)
                    draw_max = min(fva_ub_norm, zoom_limit * 1.5)

                    y_offset = y_pos + model_offsets[model_idx]
                    
                    ax.hlines(
                        y=y_offset, 
                        xmin=draw_min, 
                        xmax=draw_max,
                        color=colors[model_idx % len(colors)],
                        linewidth=4, 
                        alpha=0.7,
                        label=name if i == 0 else ""
                    )
            
            # Grid lines
            ax.axhline(y=y_pos, color='gray', linestyle=':', alpha=0.15, linewidth=1)

        # Formatting. Clamp the X-Axis.
        ax.set_xlim(-zoom_limit, zoom_limit)
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=-0.5, color='black', linestyle=':', alpha=0.2) # MFA Lower Bound ref
        ax.axvline(x=0.5, color='black', linestyle=':', alpha=0.2)  # MFA Upper Bound ref

        ax.set_yticks(range(len(rxn_chunk)))
        ax.set_yticklabels(rxn_chunk, fontsize=10, fontfamily='monospace')
        ax.invert_yaxis()
        
        ax.set_xlabel('Normalized Deviation (Units of MFA Range Width)')
        ax.set_title(f'MFA vs FVA: Normalized Deviations (Zoomed)')
        
        # Custom Legend
        handles = [mlines.Line2D([], [], color='black', linewidth=4, label='MFA Range')]
        for idx, name in enumerate(models_data.items()):
            handles.append(mlines.Line2D([], [], color=colors[idx % len(colors)], linewidth=4, label=name[0]))
            
        ax.legend(handles=handles, loc='upper right', frameon=True)
        
        plt.tight_layout()
        plt.show()


def plot_jaccard_index_comparison(
    jaccard_indices: list[float],
    zero_overlaps: list[int],
    model_names: list[str] = None
) -> None:
    """
    Plot bar chart of Jaccard indices and zero overlaps for multiple models.
    
    Parameters:
        jaccard_indices : list[float]
            List of mean Jaccard indices for each model
        zero_overlaps : list[int]
            List of number of zero-overlap reactions for each model
        model_names : list[str], optional
            List of model names for x-axis labels. If None, uses "Model 1", "Model 2", etc.
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(jaccard_indices))]
    
    if len(jaccard_indices) != len(zero_overlaps) or len(jaccard_indices) != len(model_names):
        raise ValueError("jaccard_indices, zero_overlaps, and model_names must have the same length")
    
    # Create figure with 2 subplots arranged horizontally
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Left subplot: Jaccard indices
    bars1 = ax1.bar(model_names, jaccard_indices, color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Mean Jaccard Index', fontsize=12)
    ax1.set_title('Mean Jaccard Index', fontsize=13)
    ax1.set_ylim([0, 0.1])
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_xticklabels(model_names)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # Right subplot: Zero overlaps
    bars2 = ax2.bar(model_names, zero_overlaps, color='#d62728', alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Number of Reactions', fontsize=12)
    ax2.set_title('Non-overlapping Reactions (J = 0)', fontsize=13)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_xticklabels(model_names)
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

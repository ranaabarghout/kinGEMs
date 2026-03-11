"""
Utility functions to validate kinGEMs metabolic reaction flux predictions against experimental data.

Plotting functions have been moved to kinGEMs/plots.py:
    - plot_fva_mfa_comparison
    - plot_fva_mfa_comparison_normalized
    - plot_jaccard_index_comparison
"""

import pandas as pd
import numpy as np


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


def calculate_mean_to_mean_distance(
    comparison_df: pd.DataFrame
) -> float:
    """
    Calculate the mean distance between FVA range means and MFA range means.

    This metric measures how close the midpoint of the model's predicted flux range
    is to the midpoint of the experimentally measured flux range.

    Parameters:
        comparison_df : pd.DataFrame
            DataFrame with columns: 'rxn_id', 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub'

    Returns:
        float: The mean absolute distance between FVA and MFA range means.
    """

    df = comparison_df.dropna(subset=['mfa_lb', 'mfa_ub', 'fva_lb', 'fva_ub']).copy()

    # Calculate means for both FVA and MFA ranges
    fva_mean = (df['fva_lb'] + df['fva_ub']) / 2
    mfa_mean = (df['mfa_lb'] + df['mfa_ub']) / 2

    # Calculate absolute distance between means
    mean_distance = np.abs(fva_mean - mfa_mean)

    # Calculate mean of all distances
    mean_of_distances = mean_distance.mean()

    print(f"\nMean-to-Mean Distance Analysis")
    print(f"   Evaluated {len(df)} reactions")
    print(f"   Mean absolute distance: {mean_of_distances:.4f}")
    print(f"   Median absolute distance: {mean_distance.median():.4f}")
    print(f"   Max distance: {mean_distance.max():.4f}")

    return mean_of_distances


def calculate_per_reaction_distances(
    comparison_df: pd.DataFrame
) -> list[float]:
    """
    Return per-reaction absolute distance between FVA and MFA range midpoints.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns: 'rxn_id', 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub'

    Returns
    -------
    list[float]
        One distance value per reaction (NaN rows dropped).
    """
    df = comparison_df.dropna(subset=['mfa_lb', 'mfa_ub', 'fva_lb', 'fva_ub']).copy()
    fva_mean = (df['fva_lb'] + df['fva_ub']) / 2
    mfa_mean = (df['mfa_lb'] + df['mfa_ub']) / 2
    return np.abs(fva_mean - mfa_mean).tolist()


def calculate_average_interval_widths(comparison_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Calculate average interval widths and standard deviations for FVA and MFA results.

    For each reaction i, the width of the interval is:
        w_FVA,i = FVA_ub,i - FVA_lb,i
        w_MFA,i = MFA_ub,i - MFA_lb,i

    The average interval widths are:
        W_FVA = (1/N) * sum(w_FVA,i)
        W_MFA = (1/N) * sum(w_MFA,i)

    Smaller FVA interval widths indicate higher precision (narrower feasible flux range).
    Comparison to W_MFA indicates how closely the model's uncertainty matches experimental uncertainty.

    Parameters:
        comparison_df : pd.DataFrame
            DataFrame with columns: 'fva_lb', 'fva_ub', 'mfa_lb', 'mfa_ub'

    Returns:
        tuple[float, float, float, float]: (W_FVA, std_FVA, W_MFA, std_MFA)
            Average interval widths and their standard deviations for FVA and MFA
    """
    # Drop rows with missing values
    df = comparison_df.dropna(subset=['mfa_lb', 'mfa_ub', 'fva_lb', 'fva_ub']).copy()

    if len(df) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Calculate interval widths for each reaction
    df['w_fva'] = df['fva_ub'] - df['fva_lb']
    df['w_mfa'] = df['mfa_ub'] - df['mfa_lb']

    # Calculate average widths and standard deviations
    w_fva_avg = df['w_fva'].mean()
    w_fva_std = df['w_fva'].std()
    w_mfa_avg = df['w_mfa'].mean()
    w_mfa_std = df['w_mfa'].std()

    return float(w_fva_avg), float(w_fva_std), float(w_mfa_avg), float(w_mfa_std)


# Plotting functions have been moved to kinGEMs/plots.py
# Import them from there:
#   from kinGEMs.plots import (
#       plot_fva_mfa_comparison,
#       plot_fva_mfa_comparison_normalized,
#       plot_jaccard_index_comparison,
#   )
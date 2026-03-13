"""
Utility functions to validate kinGEMs metabolic reaction flux predictions against experimental data.

Plotting functions have been moved to kinGEMs/plots.py:
    - plot_fva_mfa_comparison
    - plot_fva_mfa_comparison_normalized
    - plot_jaccard_index_comparison
"""

import pandas as pd
import numpy as np
from scipy import stats


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


def calculate_statistical_significance(
    baseline_df: pd.DataFrame,
    model_df: pd.DataFrame,
    baseline_label: str = "Baseline",
    model_label: str = "kinGEMs",
) -> pd.DataFrame:
    """
    Run paired statistical significance tests comparing two models on per-reaction metrics.

    Uses the Wilcoxon signed-rank test (paired, non-parametric) for continuous
    per-reaction metrics (Jaccard index, mean-to-mean distance), and McNemar's test
    for the binary consistency score.

    Effect sizes are reported as:
    - Rank-biserial correlation (r) for Wilcoxon tests: |r| < 0.1 negligible,
      0.1–0.3 small, 0.3–0.5 medium, > 0.5 large.
    - Odds ratio for McNemar's test.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Comparison DataFrame for the baseline model (cols: rxn_id, fva_lb, fva_ub, mfa_lb, mfa_ub).
    model_df : pd.DataFrame
        Comparison DataFrame for the model being tested.
    baseline_label : str
        Human-readable label for the baseline.
    model_label : str
        Human-readable label for the model being tested.

    Returns
    -------
    pd.DataFrame
        One row per test with columns: test, baseline_label, model_label,
        baseline_median, model_median, statistic, p_value, effect_size,
        effect_size_label, n_pairs, interpretation.
    """
    results = []

    # ------------------------------------------------------------------ #
    # Helper: compute per-reaction metrics for a comparison df
    # ------------------------------------------------------------------ #
    def _per_reaction(df: pd.DataFrame) -> pd.DataFrame:
        d = df.dropna(subset=["mfa_lb", "mfa_ub", "fva_lb", "fva_ub"]).copy()
        # Jaccard
        inter_lb = d[["fva_lb", "mfa_lb"]].max(axis=1)
        inter_ub = d[["fva_ub", "mfa_ub"]].min(axis=1)
        intersection = (inter_ub - inter_lb).clip(lower=0)
        union_lb = d[["fva_lb", "mfa_lb"]].min(axis=1)
        union_ub = d[["fva_ub", "mfa_ub"]].max(axis=1)
        union = (union_ub - union_lb).replace(0, 1e-9)
        d["jaccard"] = intersection / union
        # Mean-to-mean distance
        fva_mean = (d["fva_lb"] + d["fva_ub"]) / 2
        mfa_mean = (d["mfa_lb"] + d["mfa_ub"]) / 2
        d["m2m_dist"] = np.abs(fva_mean - mfa_mean)
        # Consistent (binary)
        overlap_lb = d[["fva_lb", "mfa_lb"]].max(axis=1)
        overlap_ub = d[["fva_ub", "mfa_ub"]].min(axis=1)
        d["consistent"] = (overlap_lb <= overlap_ub).astype(int)
        return d[["rxn_id", "jaccard", "m2m_dist", "consistent"]]

    base_pr = _per_reaction(baseline_df)
    model_pr = _per_reaction(model_df)

    # Align on shared reactions
    merged = base_pr.merge(model_pr, on="rxn_id", suffixes=("_base", "_model"))
    n = len(merged)
    if n < 5:
        print("Warning: fewer than 5 shared reactions — skipping significance tests.")
        return pd.DataFrame()

    def _rank_biserial(stat, n_pairs):
        """Rank-biserial correlation from Wilcoxon statistic W and n pairs."""
        # r = 1 - (2W) / (n*(n+1))  — standard formula
        return 1 - (2 * stat) / (n_pairs * (n_pairs + 1))

    def _interpret_r(r):
        ar = abs(r)
        if ar < 0.1:
            return "negligible"
        if ar < 0.3:
            return "small"
        if ar < 0.5:
            return "medium"
        return "large"

    # ------------------------------------------------------------------ #
    # Test 1: Jaccard index (Wilcoxon signed-rank, paired)
    # ------------------------------------------------------------------ #
    x_base = merged["jaccard_base"].values
    x_model = merged["jaccard_model"].values
    stat_j, p_j = stats.wilcoxon(x_base, x_model, alternative="two-sided", zero_method="wilcox")
    r_j = _rank_biserial(stat_j, n)
    results.append({
        "test": "Jaccard Index (Wilcoxon signed-rank)",
        "baseline_label": baseline_label,
        "model_label": model_label,
        "baseline_median": float(np.median(x_base)),
        "model_median": float(np.median(x_model)),
        "statistic": float(stat_j),
        "p_value": float(p_j),
        "effect_size": float(r_j),
        "effect_size_label": f"rank-biserial r = {r_j:.3f} ({_interpret_r(r_j)})",
        "n_pairs": n,
        "interpretation": (
            f"{model_label} has {'higher' if np.median(x_model) > np.median(x_base) else 'lower'} "
            f"median Jaccard ({'p<0.05, significant' if p_j < 0.05 else 'not significant'})"
        ),
    })

    # ------------------------------------------------------------------ #
    # Test 2: Mean-to-mean distance (Wilcoxon signed-rank, paired)
    # ------------------------------------------------------------------ #
    d_base = merged["m2m_dist_base"].values
    d_model = merged["m2m_dist_model"].values
    stat_d, p_d = stats.wilcoxon(d_base, d_model, alternative="two-sided", zero_method="wilcox")
    r_d = _rank_biserial(stat_d, n)
    results.append({
        "test": "Mean-to-Mean Distance (Wilcoxon signed-rank)",
        "baseline_label": baseline_label,
        "model_label": model_label,
        "baseline_median": float(np.median(d_base)),
        "model_median": float(np.median(d_model)),
        "statistic": float(stat_d),
        "p_value": float(p_d),
        "effect_size": float(r_d),
        "effect_size_label": f"rank-biserial r = {r_d:.3f} ({_interpret_r(r_d)})",
        "n_pairs": n,
        "interpretation": (
            f"{model_label} has {'lower' if np.median(d_model) < np.median(d_base) else 'higher'} "
            f"median distance ({'p<0.05, significant' if p_d < 0.05 else 'not significant'})"
        ),
    })

    # ------------------------------------------------------------------ #
    # Test 3: Consistency score (McNemar's test on paired binary outcomes)
    # ------------------------------------------------------------------ #
    c_base = merged["consistent_base"].values
    c_model = merged["consistent_model"].values
    # Discordant pairs
    n_10 = int(((c_base == 1) & (c_model == 0)).sum())  # base consistent, model not
    n_01 = int(((c_base == 0) & (c_model == 1)).sum())  # model consistent, base not
    n_discordant = n_10 + n_01
    if n_discordant > 0:
        # Exact binomial (works for small n_discordant; equivalent to McNemar with continuity)
        binom_result = stats.binomtest(n_01, n_discordant, p=0.5, alternative="two-sided")
        p_mc = float(binom_result.pvalue)
        odds_ratio = (n_01 + 0.5) / (n_10 + 0.5)  # add 0.5 to avoid division by zero
    else:
        p_mc = 1.0
        odds_ratio = 1.0
    results.append({
        "test": "Consistency Score (McNemar's test)",
        "baseline_label": baseline_label,
        "model_label": model_label,
        "baseline_median": float(c_base.mean()),
        "model_median": float(c_model.mean()),
        "statistic": float(n_discordant),
        "p_value": p_mc,
        "effect_size": float(odds_ratio),
        "effect_size_label": f"odds ratio = {odds_ratio:.3f}",
        "n_pairs": n,
        "interpretation": (
            f"{model_label} consistency = {c_model.mean():.3f} vs baseline {c_base.mean():.3f} "
            f"({'p<0.05, significant' if p_mc < 0.05 else 'not significant'})"
        ),
    })

    # ------------------------------------------------------------------ #
    # Test 4: FVA interval width (Wilcoxon signed-rank on log-widths)
    #
    # Width = fva_ub - fva_lb per reaction. We test on log-scale because a
    # halving of a large interval (1000→500) is equivalent in biological
    # meaning to a halving of a small interval (10→5), but raw differences
    # would weight the former 100× more.  We drop pairs where both widths
    # are zero (fully pinned reactions that are uninformative).
    # ------------------------------------------------------------------ #
    def _fva_width(df: pd.DataFrame) -> pd.Series:
        # Restrict to MFA-matched reactions only (same 46 as other tests)
        d = df.dropna(subset=["fva_lb", "fva_ub", "mfa_lb", "mfa_ub"]).copy()
        w = (d["fva_ub"] - d["fva_lb"]).clip(lower=0)
        return pd.Series(w.values, index=d["rxn_id"].values, name="width")

    w_base  = _fva_width(baseline_df)
    w_model = _fva_width(model_df)
    # Align on shared reactions
    w_both = pd.DataFrame({"base": w_base, "model": w_model}).dropna()
    # Drop pairs where both are zero (uninformative)
    nonzero_mask = ~((w_both["base"] == 0) & (w_both["model"] == 0))
    w_both = w_both[nonzero_mask]
    # Replace exact zeros with a small epsilon before log so single-zero pairs
    # contribute rather than being silently dropped
    eps = 1e-9
    log_base  = np.log(w_both["base"].replace(0, eps)).values
    log_model = np.log(w_both["model"].replace(0, eps)).values
    n_w = len(w_both)
    if n_w >= 5:
        stat_w, p_w = stats.wilcoxon(log_base, log_model,
                                     alternative="two-sided", zero_method="wilcox")
        r_w = _rank_biserial(stat_w, n_w)
        results.append({
            "test": "FVA Interval Width — log-scale (Wilcoxon signed-rank)",
            "baseline_label": baseline_label,
            "model_label": model_label,
            "baseline_median": float(np.exp(np.median(log_base))),
            "model_median": float(np.exp(np.median(log_model))),
            "statistic": float(stat_w),
            "p_value": float(p_w),
            "effect_size": float(r_w),
            "effect_size_label": f"rank-biserial r = {r_w:.3f} ({_interpret_r(r_w)})",
            "n_pairs": n_w,
            "interpretation": (
                f"{model_label} has {'narrower' if np.median(log_model) < np.median(log_base) else 'wider'} "
                f"FVA intervals on log-scale "
                f"({'p<0.05, significant' if p_w < 0.05 else 'not significant'})"
            ),
        })
    else:
        print(f"Warning: only {n_w} non-zero width pairs — skipping FVA width test.")

    result_df = pd.DataFrame(results)

    # Pretty-print summary
    print(f"\n{'='*70}")
    print(f"Statistical Significance Tests: {baseline_label} vs {model_label}")
    print(f"  Paired reactions (n={n})")
    print(f"{'='*70}")
    for _, row in result_df.iterrows():
        print(f"\n  {row['test']}")
        print(f"    Baseline median : {row['baseline_median']:.4f}")
        print(f"    Model median    : {row['model_median']:.4f}")
        print(f"    p-value         : {row['p_value']:.4e}{'  ***' if row['p_value'] < 0.001 else ('  **' if row['p_value'] < 0.01 else ('  *' if row['p_value'] < 0.05 else '  ns'))}")
        print(f"    Effect size     : {row['effect_size_label']}")
        print(f"    → {row['interpretation']}")
    print(f"{'='*70}\n")

    return result_df


# Plotting functions have been moved to kinGEMs/plots.py
# Import them from there:
#   from kinGEMs.plots import (
#       plot_fva_mfa_comparison,
#       plot_fva_mfa_comparison_normalized,
#       plot_jaccard_index_comparison,
#   )
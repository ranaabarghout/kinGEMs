"""
Parameter tuning module for kinGEMs.

This module provides simulated annealing functionality to tune kcat parameters
and optimize the model's performance.
"""
import copy  # noqa: F401
import math  # noqa: F401
import os
import random  # noqa: F401
import warnings

from Bio.Data.IUPACData import protein_letters
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
import pandas as pd

from ..config import ensure_dir_exists
from ..dataset import annotate_model_with_kcat_and_gpr
from ..plots import plot_annealing_progress
from .optimize import run_optimization_with_dataframe

warnings.filterwarnings('ignore')
import logging

logging.getLogger('distributed').setLevel(logging.ERROR)
try:
    import gurobipy
    gurobipy.setParam('OutputFlag', 0)
except ImportError:
    pass




def simulated_annealing(
    model,
    processed_data,
    biomass_reaction,
    objective_value,
    gene_sequences_dict,
    output_dir=None,
    enzyme_fraction=0.15,
    temperature=1.0,
    cooling_rate=0.98,
    min_temperature=0.01,
    max_iterations=250,
    max_unchanged_iterations=3,
    change_threshold=0.001,
    n_top_enzymes=65,
    verbose=False,
    medium=None,
    medium_upper_bound=False
):
    """
    Use simulated annealing to tune kcat values for improved biomass production.

    This function preserves original kcat_mean values for proper neighbor calculation
    and creates a kcat_updated column to track tuned values. The optimization function
    automatically uses kcat_updated when available.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model to optimize
    processed_data : pandas.DataFrame
        DataFrame with enzyme kinetic data
    biomass_reaction : str
        ID of the biomass reaction to optimize
    objective_value : float
        Target biomass value
    gene_sequences_dict : dict
        Dictionary mapping gene IDs to protein sequences
    output_dir : str, optional
        Directory to save results
    enzyme_fraction : float, optional
        Maximum enzyme mass fraction (default: 0.15)
    temperature : float, optional
        Initial temperature for simulated annealing (default: 1.0)
    cooling_rate : float, optional
        Rate at which temperature decreases (default: 0.98)
    min_temperature : float, optional
        Minimum temperature before stopping (default: 0.01)
    max_iterations : int, optional
        Maximum number of iterations (default: 250)
    max_unchanged_iterations : int, optional
        Stop after this many iterations without improvement (default: 3)
    change_threshold : float, optional
        Minimum relative change to count as improvement (default: 0.001)
    n_top_enzymes : int, optional
        Number of top enzymes by mass to tune (default: 65)
    verbose : bool, optional
        Print detailed progress information (default: False)
    medium : dict, optional
        Growth medium composition
    medium_upper_bound : bool or float, optional
        Upper bound for medium exchanges (default: False)

    Returns
    -------
    tuple
        (kcat_dict, top_targets, best_df, iterations, biomasses, df_FBA)
    """

    def acceptance_probability(old_biomass, new_biomass, temperature):
        # For MAXIMIZATION: always accept if new > old, probabilistically accept if new < old
        if new_biomass > old_biomass:
            return 1.0
        return math.exp((new_biomass - old_biomass) / temperature)

    def get_neighbor(kcat_value, std):
        # Handle NaN kcat values - skip perturbation
        if pd.isna(kcat_value) or kcat_value <= 0:
            return kcat_value  # Return unchanged for invalid values

        k_val_hr = kcat_value * 3600
        std_hr = std * 3600 if not pd.isna(std) else 0

        # If no standard deviation, use a more aggressive default for exploration
        if std_hr == 0 or pd.isna(std_hr):
            std_hr = k_val_hr * 0.2  # 20% of original value (more aggressive)

        # Generate larger perturbations for meaningful improvements
        # Use 85% positive bias: 85% chance of increase, 15% chance of decrease
        if random.random() < 0.85:  # 85% chance of positive perturbation
            # Much larger increases - up to 50% of original kcat
            perturbation = abs(random.gauss(0, 10*std_hr))  # 1 order of magnitude larger std
            new_kcat = k_val_hr + perturbation  # Increase
        else:  # 10% chance of small decrease
            perturbation = abs(random.gauss(0, std_hr))  # std, negative
            new_kcat = k_val_hr - perturbation  # Decrease

        # Set bounds for perturbations - allow larger changes
        # Allow increases up to 100% above original
        ub = 10*std_hr # 1 order of magnitude higher! #k_val_hr * 2.0  # Double the original kcat
        ub = min(ub, 4.6e9)  # Biological maximum

        # Set lower bound to prevent going too low (10% of original minimum)
        lb = k_val_hr - std_hr

        # Clamp to bounds
        return max(min(new_kcat, ub), lb)

    def update_kcat(df, reaction_id, gene_id, new_kcat_value):
        updated_df = df.copy()
        cond = (
            (updated_df['Reactions'] == reaction_id) &
            (updated_df['Single_gene'] == gene_id)
        )
        # convert back to per-second
        new_value_s = new_kcat_value / 3600

        # Preserve original kcat_mean and kcat_std - only update kcat_updated column
        # Create kcat_updated column if it doesn't exist
        if 'kcat_updated' not in updated_df.columns:
            updated_df['kcat_updated'] = updated_df['kcat_mean'].copy()

        # Get old value for debug output
        old_value = updated_df.loc[cond, 'kcat_updated'].iloc[0] if cond.sum() > 0 else None

        # Update the kcat_updated column (optimization will prefer this over kcat_mean)
        updated_df.loc[cond, 'kcat_updated'] = new_value_s

        # Debug: verify update happened
        if cond.sum() > 0 and verbose:
            actual_new = updated_df.loc[cond, 'kcat_updated'].iloc[0]
            if verbose:
                print(f"    [UPDATE] {reaction_id}_{gene_id}: {old_value:.6e} → {actual_new:.6e} s⁻¹")

        return updated_df

    def calculate_molecular_weight(seq):
        return ProteinAnalysis(seq).molecular_weight()

    # Precompute MWs
    # mw_dict = {
    #     gene: calculate_molecular_weight(seq)
    #     for gene, seq in gene_sequences_dict.items() if seq
    # }

    def safe_mw(seq: str) -> float:
        # keep only standard amino acids
        cleaned = "".join([aa for aa in seq if aa in protein_letters])
        if not cleaned:
            cleaned = "A"  # fallback to alanine
        try:
            return molecular_weight(cleaned, seq_type="protein")
        except Exception:
            return 1e5  # large default if something still weird

    mw_dict = {
        gene: safe_mw(seq)
        for gene, seq in gene_sequences_dict.items() if seq
    }

    # INITIAL FBA
    biomass, df_FBA, _, _ = run_optimization_with_dataframe(
        model=model,
        processed_df=processed_data,
        objective_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_fraction,
        output_dir=output_dir,
        save_results=False,
        verbose=False,
        medium=medium,
        medium_upper_bound=medium_upper_bound
    )

    # Check if initial optimization failed
    if biomass is None or biomass <= 0:
        print(f"⚠️  ERROR: Initial enzyme-constrained optimization failed (biomass={biomass})")
        print("   This suggests the enzyme constraints are too restrictive.")
        print("   Consider increasing enzyme_upper_bound or checking your kinetic data.")
        # Return empty results rather than continuing with invalid state
        return {}, pd.DataFrame(), processed_data, [0], [0.0], pd.DataFrame()

    # Pick top N enzymes by mass
    enzyme_df = df_FBA[df_FBA['Variable']=='enzyme'].copy()
    enzyme_df['MW'] = enzyme_df['Index'].map(mw_dict).fillna(0)
    enzyme_df['enzyme_mass'] = enzyme_df['Value'] * enzyme_df['MW'] * 1e-3
    top_n = enzyme_df.nlargest(n_top_enzymes, 'enzyme_mass')
    top_targets = (
        top_n[['Index','enzyme_mass']]
        .rename(columns={'Index':'Single_gene'})
        .merge(processed_data, on='Single_gene')
        [['Reactions','Single_gene','enzyme_mass','kcat_mean','kcat_std']]
        .reset_index(drop=True)
    )

    # Check for duplicates BEFORE deduplication
    duplicates = top_targets.duplicated(subset=['Reactions', 'Single_gene'], keep=False)
    if duplicates.any():
        # print(f"[INFO] Found {duplicates.sum()} duplicate reaction-gene pairs in {len(top_targets)} total rows")
        # print(f"[INFO] Deduplicating to keep only first occurrence of each (Reaction, Gene) pair...")
        top_targets = top_targets.drop_duplicates(subset=['Reactions', 'Single_gene'], keep='first').reset_index(drop=True)
        # print(f"[INFO] After deduplication: {len(top_targets)} unique reaction-gene pairs")

    # print(f"\n[ANNEALING DEBUG] Top 5 target enzymes:")
    # print(top_targets.head()[['Reactions', 'Single_gene', 'enzyme_mass', 'kcat_mean']])
    # print(f"[ANNEALING DEBUG] Total targets: {len(top_targets)}")

    # Verify these reactions/genes exist in processed_data
    for idx, row in top_targets.head(3).iterrows():
        rxn, gene = row['Reactions'], row['Single_gene']
        matches = processed_data[(processed_data['Reactions']==rxn) & (processed_data['Single_gene']==gene)]
        # print(f"[DEBUG] {rxn}_{gene}: found {len(matches)} matches in processed_data, kcat_mean={matches['kcat_mean'].iloc[0] if len(matches)>0 else 'NOT FOUND'}")

    largest_rxn_id  = top_targets['Reactions'].tolist()
    largest_gene_id = top_targets['Single_gene'].tolist()
    # Keep original kcat_mean values for neighbor calculation (never changes)
    original_kcats = top_targets['kcat_mean'].tolist()
    # Track current tuned values (starts same as original, gets updated)
    current_solution = top_targets['kcat_mean'].tolist()
    stds             = top_targets['kcat_std'].fillna(0.1).tolist()

    df_new = processed_data.copy()
    # Initialize kcat_updated column with original kcat_mean values
    if 'kcat_updated' not in df_new.columns:
        df_new['kcat_updated'] = df_new['kcat_mean'].copy()

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
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Current biomass = {current_biomass:.6e}")

        # PROPOSE & print old vs new kcats - ONLY PERTURB A SUBSET
        updated_df = df_new.copy()
        actually_changed = 0

        # Perturb more enzymes per iteration for bigger impact
        # Use 30-60% of enzymes per iteration (was 10-20%)
        n_to_perturb = max(1, int(len(largest_rxn_id) * random.uniform(0.3, 0.6)))
        indices_to_perturb = random.sample(range(len(largest_rxn_id)), n_to_perturb)

        # if iteration <= 3:  # Debug info
        #     print(f"  [DEBUG] Perturbing {n_to_perturb} out of {len(largest_rxn_id)} enzymes")

        for i, (rxn, gene) in enumerate(zip(largest_rxn_id, largest_gene_id)):
            # Always use ORIGINAL kcat_mean for neighbor calculation
            original_k = original_kcats[i]  # in s⁻¹
            current_k = current_solution[i]  # current tuned value

            # Skip if not selected for perturbation this iteration
            if i not in indices_to_perturb:
                new_k_hr = current_k * 3600  # Keep current value
                new_k_s = current_k
                # if iteration <= 3:  # Debug first 3 iterations
                #     print(f"  [DEBUG] {rxn}_{gene}: {original_k:.3e} → {new_k_s:.3e} s⁻¹ (change: +0.0%) [UNCHANGED]")
            # Skip NaN values - don't perturb them
            elif pd.isna(original_k) or original_k <= 0:
                new_k_hr = original_k * 3600 if not pd.isna(original_k) else original_k
                new_k_s = original_k
                # if iteration <= 3:  # Debug first 3 iterations
                #     print(f"  [DEBUG] {rxn}_{gene}: {original_k:.3e} → {new_k_s:.3e} s⁻¹ (change: +nan%) [SKIPPED - NaN/invalid]")
            else:
                # Use original kcat for neighbor calculation (preserves proper exploration)
                new_k_hr = get_neighbor(original_k, stds[i])  # returns hr⁻¹
                new_k_s = new_k_hr / 3600

                # Check if actually different
                if abs(new_k_s - current_k) / max(current_k, 1e-12) > 0.01:  # >1% change
                    actually_changed += 1

                # if iteration <= 3:  # Debug first 3 iterations
                #     print(f"  [DEBUG] {rxn}_{gene}: {original_k:.3e} → {new_k_s:.3e} s⁻¹ (change: {(new_k_s-original_k)/original_k*100:+.1f}%)")

            # update_kcat expects hr⁻¹ and will convert to s⁻¹ internally
            updated_df = update_kcat(updated_df, rxn, gene, new_k_hr)

        if verbose:
            print(f"  Actually changed {actually_changed}/{len(largest_rxn_id)} kcats by >1%")

        # Debug: Verify kcats actually changed in updated_df
        if not verbose and iteration <= 3:
            first_rxn, first_gene = largest_rxn_id[0], largest_gene_id[0]
            old_val = df_new.loc[(df_new['Reactions']==first_rxn) & (df_new['Single_gene']==first_gene), 'kcat_mean'].iloc[0]
            new_val = updated_df.loc[(updated_df['Reactions']==first_rxn) & (updated_df['Single_gene']==first_gene), 'kcat_mean'].iloc[0]
            # print(f"DEBUG Iter {iteration}: First kcat change: {first_rxn}:{first_gene} {old_val:.3e} -> {new_val:.3e} ({((new_val/old_val-1)*100):.1f}%)")
            #     print(f"\n  [DEBUG] First target {first_rxn}_{first_gene}: old={old_val:.6f} s⁻¹, new={new_val:.6f} s⁻¹, changed={old_val != new_val}")

        # EVALUATE with updated kcats
        new_biomass, temp_df_FBA, _, _ = run_optimization_with_dataframe(
            model=model,
            processed_df=updated_df,
            objective_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_fraction,
            output_dir=None,
            save_results=False,
            verbose=True if iteration <= 3 else False,  # Enable verbose for first 3 iterations to debug
            medium=medium,
            medium_upper_bound=medium_upper_bound
        )

        # Handle optimization failures
        if new_biomass is None or new_biomass <= 0:
            if verbose or iteration <= 5:
                print(f"  [DEBUG] Iter {iteration}: Optimization failed (biomass={new_biomass})")
                print(f"    - Perturbed {n_to_perturb} enzymes with increases ranging 0-2%")
                print("    - This suggests even small increases overwhelm constraints")
            # Skip this iteration - don't accept failed optimizations
            # Keep using the current biomass instead of setting to 0.0
            new_biomass = current_biomass  # No change - keep current solution
            temp_df_FBA = df_FBA.copy()  # Use previous valid results
        else:
            if iteration <= 5:  # Debug successful optimizations too
                print(f"  [DEBUG] Iter {iteration}: Optimization succeeded! biomass={new_biomass:.6e}")
                if new_biomass > current_biomass:
                    print(f"    - IMPROVEMENT: +{((new_biomass/current_biomass-1)*100):.3f}%")

        # Debug: Check if enzyme allocations changed even if biomass didn't
        # (Currently disabled to avoid unused variables)
        # if not verbose and iteration <= 3:
        #     # Compare enzyme allocation for first target
        #     old_enzyme = df_FBA[df_FBA['Index']==largest_gene_id[0]]
        #     new_enzyme = temp_df_FBA[temp_df_FBA['Index']==largest_gene_id[0]]
        #     if len(old_enzyme) > 0 and len(new_enzyme) > 0:
        #         old_alloc = old_enzyme[old_enzyme['Variable']=='enzyme']['Value'].iloc[0] if len(old_enzyme[old_enzyme['Variable']=='enzyme']) > 0 else 0
        #         new_alloc = new_enzyme[new_enzyme['Variable']=='enzyme']['Value'].iloc[0] if len(new_enzyme[new_enzyme['Variable']=='enzyme']) > 0 else 0
        #         print(f"  [DEBUG] Enzyme allocation for {largest_gene_id[0]}: {old_alloc:.6e} → {new_alloc:.6e} mmol/gDW/h")

        if verbose:
            print(f"Proposed biomass = {new_biomass:.6e}")

        # ACCEPT or REJECT
        old_biomass = current_biomass  # Store for change calculation
        prob = acceptance_probability(current_biomass, new_biomass, temperature)
        random_val = random.random()
        accept = prob > random_val

        # Debug output for non-verbose mode
        # if not verbose and iteration <= 3:
            # print(f"\n  [DEBUG Iter {iteration}] current={current_biomass:.6f}, proposed={new_biomass:.6f}, prob={prob:.4f}, random={random_val:.4f}, accept={accept}")

        if accept:
            if verbose:
                print(f"Iteration {iteration}: ACCEPTED (Δ = {new_biomass-current_biomass:.2e})")
            df_FBA = temp_df_FBA
            df_new = updated_df.copy()
            current_biomass = new_biomass
            current_solution = [
                df_new.loc[
                    (df_new['Reactions']==rxn)&(df_new['Single_gene']==gene),
                    'kcat_updated'
                ].iat[0]
                for rxn,gene in zip(largest_rxn_id, largest_gene_id)
            ]
            if new_biomass > best_biomass:
                best_biomass  = new_biomass
                best_solution = current_solution[:]
                best_df       = df_new.copy()
        else:
            if verbose:
                print(f"Iteration {iteration}: REJECTED (Δ = {new_biomass-current_biomass:.2e})")

        # Compute ACTUAL change after acceptance/rejection
        change = abs(current_biomass - old_biomass) / max(old_biomass, 1e-6)

        iterations.append(iteration)
        biomasses.append(current_biomass)  # Record ACCEPTED biomass, not proposed

        # Print progress every iteration (non-verbose mode)
        if not verbose and iteration % 1 == 0:
            print(f"  Iter {iteration}/{max_iterations}: biomass={current_biomass:.6f}, temp={temperature:.4f}", end='\r')

        # STAGNATION check uses actual change
        if change < change_threshold:
            no_change_counter += 1
            if no_change_counter >= max_unchanged_iterations:
                print()  # New line after progress indicator
                if verbose:
                    print(f"No significant change for {max_unchanged_iterations} iterations; stopping early.")
                else:
                    print(f"  Early stop: No change for {no_change_counter} iterations")
                break
        else:
            no_change_counter = 0

        temperature *= cooling_rate
        iteration += 1

    # Clear progress line
    if not verbose:
        print()  # New line after progress indicator

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

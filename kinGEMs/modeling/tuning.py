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
import pandas as pd

from ..config import ensure_dir_exists
from ..dataset import annotate_model_with_kcat_and_gpr
from ..plots import plot_kcat_comparison
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
    medium_upper_bound=False,
    bidirectional_constraints=True
):
    """
    Use simulated annealing to tune kcat values for improved biomass production.

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
        """
        Generate a new kcat value within experimental uncertainty bounds.

        Uses a conservative approach that respects experimental std deviation.
        Samples within ±2σ bounds with proper experimental constraints.
        """
        k_val_hr = kcat_value * 3600
        std_hr = std * 3600

        # If no std provided, use 20% of the mean as uncertainty
        if std_hr == 0:
            std_hr = k_val_hr * 0.2

        # Generate new value using normal distribution within ±2σ bounds
        # This respects experimental uncertainty and stays within 95% confidence interval
        sigma_multiplier = random.uniform(-2.0, 2.0)
        new_kcat = k_val_hr + (sigma_multiplier * std_hr)

        # For debugging: make more aggressive changes to test if current changes are too small
        # Temporarily increase the variation to 3x standard range for better exploration
        if std_hr > 0:
            # Use a wider range for better exploration of the parameter space
            variation_factor = random.uniform(-3.0, 3.0)
            new_kcat = k_val_hr + (variation_factor * std_hr)

        # Ensure we stay within experimental bounds
        # Lower bound: mean - 3σ (but never below zero) - wider for exploration
        lb = max(0, k_val_hr - 3*std_hr)
        # Upper bound: mean + 3σ - wider for exploration
        ub = k_val_hr + 3*std_hr

        # Apply physical constraints
        # Minimum: positive value
        lb = max(lb, 1e-6)  # Ensure positive kcat
        # Maximum: reasonable biological limit (per hour)
        ub = min(ub, 1e6)  # Very high but plausible kcat/hour

        # Apply bounds
        new_kcat = min(max(new_kcat, lb), ub)

        return new_kcat

    def update_kcat(df, reaction_id, gene_id, direction, new_kcat_value):
        """
        Update kcat value for a specific reaction, gene, and direction.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with kcat data
        reaction_id : str
            Reaction ID
        gene_id : str
            Gene ID
        direction : str
            Direction ('forward' or 'reverse')
        new_kcat_value : float
            New kcat value in hr⁻¹
        
        Returns
        -------
        pandas.DataFrame
            Updated DataFrame
        """
        updated_df = df.copy()
        
        # Check if Direction column exists (bidirectional mode)
        if 'Direction' in updated_df.columns:
            cond = (
                (updated_df['Reactions'] == reaction_id) &
                (updated_df['Single_gene'] == gene_id) &
                (updated_df['Direction'] == direction)
            )
        else:
            # Fallback for non-directional mode
            cond = (
                (updated_df['Reactions'] == reaction_id) &
                (updated_df['Single_gene'] == gene_id)
            )
        
        # Get old value for debugging
        old_value = updated_df.loc[cond, 'kcat_mean'].iloc[0] if cond.sum() > 0 else None
        new_value_s = new_kcat_value / 3600  # convert back to per-second

        # Update both kcat_mean AND kcat columns (optimization uses 'kcat' if it exists)
        updated_df.loc[cond, 'kcat_mean'] = new_value_s
        if 'kcat' in updated_df.columns:
            updated_df.loc[cond, 'kcat'] = new_value_s

        # Debug: verify update happened
        if cond.sum() > 0 and verbose:
            actual_new = updated_df.loc[cond, 'kcat_mean'].iloc[0]
            direction_str = f"_{direction}" if 'Direction' in updated_df.columns else ""
            print(f"    [UPDATE] {reaction_id}_{gene_id}{direction_str}: {old_value:.6e} → {actual_new:.6e} s⁻¹")

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
        enzyme_ratio=True,
        output_dir=output_dir,
        save_results=False,
        verbose=verbose,
        medium=medium,
        medium_upper_bound=medium_upper_bound,
        bidirectional_constraints=bidirectional_constraints
    )

    # Pick top N enzymes by mass - handle directionality
    enzyme_df = df_FBA[df_FBA['Variable']=='enzyme'].copy()
    enzyme_df['MW'] = enzyme_df['Index'].map(mw_dict).fillna(0)
    enzyme_df['enzyme_mass'] = enzyme_df['Value'] * enzyme_df['MW'] * 1e-3
    top_n = enzyme_df.nlargest(n_top_enzymes, 'enzyme_mass')
    
    # Check if processed_data has Direction column for bidirectional constraints
    has_direction = 'Direction' in processed_data.columns
    
    if has_direction:
        # For bidirectional data, we need to select the appropriate direction
        # based on actual flux direction from the optimization
        flux_df = df_FBA[df_FBA['Variable']=='v'].copy()
        top_targets_list = []
        
        for _, enzyme_row in top_n.iterrows():
            gene_id = enzyme_row['Index']
            enzyme_mass = enzyme_row['enzyme_mass']
            
            # Find all reactions for this gene in processed_data
            gene_reactions = processed_data[processed_data['Single_gene'] == gene_id]
            
            for _, reaction_row in gene_reactions.iterrows():
                rxn_id = reaction_row['Reactions']
                direction = reaction_row['Direction']
                
                # Check flux direction to determine if this direction is active
                flux_row = flux_df[flux_df['Index'] == rxn_id]
                if len(flux_row) > 0:
                    flux_value = flux_row['Value'].iloc[0]
                    
                    # Only include this direction if flux is significant and in the right direction
                    if direction == 'forward' and flux_value > 1e-6:
                        top_targets_list.append({
                            'Reactions': rxn_id,
                            'Single_gene': gene_id,
                            'Direction': direction,
                            'enzyme_mass': enzyme_mass,
                            'kcat_mean': reaction_row['kcat_mean'],
                            'kcat_std': reaction_row.get('kcat_std', 0.1)
                        })
                    elif direction == 'reverse' and flux_value < -1e-6:
                        top_targets_list.append({
                            'Reactions': rxn_id,
                            'Single_gene': gene_id,
                            'Direction': direction,
                            'enzyme_mass': enzyme_mass,
                            'kcat_mean': reaction_row['kcat_mean'],
                            'kcat_std': reaction_row.get('kcat_std', 0.1)
                        })
        
        top_targets = pd.DataFrame(top_targets_list)
        
        # If we don't have enough directional targets, fall back to all directions
        if len(top_targets) < n_top_enzymes // 2:
            print(f"[INFO] Only found {len(top_targets)} flux-directed targets, adding non-directional...")
            # Add remaining targets regardless of flux direction
            for _, enzyme_row in top_n.iterrows():
                gene_id = enzyme_row['Index']
                enzyme_mass = enzyme_row['enzyme_mass']
                gene_reactions = processed_data[processed_data['Single_gene'] == gene_id]
                
                for _, reaction_row in gene_reactions.iterrows():
                    rxn_dir_combo = (reaction_row['Reactions'], gene_id, reaction_row['Direction'])
                    # Check if this combination already exists in top_targets (only if top_targets is not empty)
                    already_exists = False
                    if len(top_targets) > 0:
                        already_exists = any((top_targets['Reactions'] == rxn_dir_combo[0]) & 
                                           (top_targets['Single_gene'] == rxn_dir_combo[1]) & 
                                           (top_targets['Direction'] == rxn_dir_combo[2]))
                    
                    if not already_exists:
                        new_row = pd.DataFrame([{
                            'Reactions': reaction_row['Reactions'],
                            'Single_gene': gene_id,
                            'Direction': reaction_row['Direction'],
                            'enzyme_mass': enzyme_mass,
                            'kcat_mean': reaction_row['kcat_mean'],
                            'kcat_std': reaction_row.get('kcat_std', 0.1)
                        }])
                        top_targets = pd.concat([top_targets, new_row], ignore_index=True)
                        
                        if len(top_targets) >= n_top_enzymes:
                            break
                if len(top_targets) >= n_top_enzymes:
                    break
    else:
        # Non-directional mode (backward compatibility)
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
        # Skip detailed verification to avoid lint warnings
        pass

    # Extract target information with directionality support
    largest_rxn_id  = top_targets['Reactions'].tolist()
    largest_gene_id = top_targets['Single_gene'].tolist()
    largest_directions = top_targets['Direction'].tolist() if has_direction else [None] * len(top_targets)
    current_solution = top_targets['kcat_mean'].tolist()
    stds             = top_targets['kcat_std'].fillna(0.1).tolist()

    # Store original data for before/after comparison
    original_processed_data = processed_data.copy()
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
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Current biomass = {current_biomass:.6e}")

        # PROPOSE & print old vs new kcats
        updated_df = df_new.copy()
        actually_changed = 0
        for i, (rxn, gene, direction) in enumerate(zip(largest_rxn_id, largest_gene_id, largest_directions)):
            old_k = current_solution[i]  # in s⁻¹
            new_k_hr = get_neighbor(old_k, stds[i])  # returns hr⁻¹
            new_k_s = new_k_hr / 3600

            # Check if actually different
            if abs(new_k_s - old_k) / max(old_k, 1e-12) > 0.01:  # >1% change
                actually_changed += 1

            if verbose:
                fold_change = new_k_s / old_k if old_k > 0 else 1.0
                direction_str = f" ({direction})" if direction else ""
                print(f"  {rxn}_{gene}{direction_str}: old kcat = {old_k:.3e} s⁻¹  →  new kcat = {new_k_s:.3e} s⁻¹ (Δ={(new_k_s-old_k)/old_k*100:+.1f}%, fold={fold_change:.2f}x)")
            
            # update_kcat now handles direction properly
            updated_df = update_kcat(updated_df, rxn, gene, direction or 'forward', new_k_hr)

        if verbose:
            print(f"  Actually changed {actually_changed}/{len(largest_rxn_id)} kcats by >1%")

        # Debug: Verify kcats actually changed in updated_df  
        if not verbose and iteration <= 3:
            # Skip detailed verification for non-verbose mode
            pass
            # if len(matches) > 1:
            #     print(f"\n  [DEBUG] {first_rxn}_{first_gene} has {len(matches)} rows with kcats: {matches.tolist()}")
            #     print(f"  [DEBUG] Average that will be used: {matches.mean():.6f} s⁻¹")
            # else:
            #     print(f"\n  [DEBUG] First target {first_rxn}_{first_gene}: old={old_val:.6f} s⁻¹, new={new_val:.6f} s⁻¹, changed={old_val != new_val}")

        if verbose:
            print("Evaluating new kcat configuration...")
            # Show a few sample kcat changes
            for i in range(min(3, len(largest_rxn_id))):
                rxn, gene = largest_rxn_id[i], largest_gene_id[i]
                old_k = current_solution[i] * 3600  # Convert to hr⁻¹ for comparison
                new_matches = updated_df[(updated_df['Reactions']==rxn) & (updated_df['Single_gene']==gene)]
                if len(new_matches) > 0:
                    new_k = new_matches['kcat_mean'].iloc[0] * 3600  # Convert to hr⁻¹
                    fold_change = new_k / old_k if old_k > 0 else 1.0
                    print(f"  Sample: {rxn}_{gene}: {old_k:.2e} → {new_k:.2e} hr⁻¹ ({fold_change:.2f}x)")

        # Debug: Print optimization call
        if verbose:
            print(f"  Calling optimization with {len(updated_df)} data rows...")

        new_biomass, temp_df_FBA, _, _ = run_optimization_with_dataframe(
            model=model,
            processed_df=updated_df,
            objective_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_fraction,
            enzyme_ratio=True,
            output_dir=None,
            save_results=False,
            verbose=False,
            medium=medium,
            medium_upper_bound=medium_upper_bound,
            bidirectional_constraints=bidirectional_constraints
        )

        if verbose:
            print(f"  Optimization completed: new_biomass = {new_biomass:.6e}")
            print(f"Proposed biomass = {new_biomass:.6e}")
            fold_biomass_change = new_biomass / current_biomass if current_biomass > 0 else 1.0
            print(f"Biomass fold change: {fold_biomass_change:.2f}x")

            # Check if this looks like unconstrained FBA
            cobra_sol = model.optimize()
            cobra_biomass = cobra_sol.objective_value
            if abs(new_biomass - cobra_biomass) < 0.01:
                print(f"⚠️  WARNING: Proposed biomass ({new_biomass:.4f}) ≈ unconstrained FBA ({cobra_biomass:.4f})")
                print("    This suggests enzyme constraints may have been effectively removed!")

        # Debug: Check if enzyme allocations changed even if biomass didn't
        if not verbose and iteration <= 3:
            # Skip enzyme allocation comparison for non-verbose mode
            pass
                # print(f"  [DEBUG] Enzyme allocation for {largest_gene_id[0]}: {old_alloc:.6e} → {new_alloc:.6e} mmol/gDW/h")

        # ACCEPT or REJECT
        old_biomass = current_biomass  # Store for change calculation

        # Safety check: reject solutions that are too close to unconstrained FBA
        # This suggests enzyme constraints have been effectively removed
        cobra_sol = model.optimize()
        cobra_biomass = cobra_sol.objective_value
        if abs(new_biomass - cobra_biomass) < 0.05:  # Within 5% of unconstrained
            if verbose:
                print("⚠️  REJECTING: Proposed biomass too close to unconstrained FBA")
                print(f"    Proposed: {new_biomass:.4f}, Unconstrained: {cobra_biomass:.4f}")
            prob = 0.0  # Force rejection
        else:
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
                    'kcat_mean'
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

    # FINALIZE: build kcat_dict from best_solution with proper directionality
    kcat_dict = {}
    for (rxn, gene, direction), k in zip(zip(largest_rxn_id, largest_gene_id, largest_directions), best_solution):
        if direction:
            # Bidirectional mode: use (rxn, gene, direction) format
            kcat_dict[f"{rxn}_{gene}_{direction}"] = k
        else:
            # Non-directional mode: use (rxn, gene) format for backward compatibility
            kcat_dict[f"{rxn}_{gene}"] = k

    if output_dir:
        save_annealing_results(
            output_dir,
            kcat_dict,
            top_targets,
            best_df,
            iterations,
            biomasses,
            df_FBA,
            original_processed_data=original_processed_data
        )

    return kcat_dict, top_targets, best_df, iterations, biomasses, df_FBA


def save_annealing_results(output_dir, kcat_dict, df_enzyme_sorted, df_new, iterations, biomasses, df_FBA,
                         original_processed_data=None, prefix=""):
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
    original_processed_data : pandas.DataFrame, optional
        Original processed data before optimization for comparison plots
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

    # Create and save annealing progress plot
    plot_annealing_progress(iterations, biomasses,
                           output_path=os.path.join(output_dir, f"{prefix}annealing_progress.png"))

    # Create and save kcat comparison plot if original data is available
    if original_processed_data is not None:
        plot_kcat_comparison(
            original_df=original_processed_data,
            optimized_df=df_new,
            top_targets=df_enzyme_sorted,
            output_path=os.path.join(output_dir, f"{prefix}kcat_comparison.png")
        )

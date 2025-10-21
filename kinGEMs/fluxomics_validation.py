"""
Validate kinGEMs metabolic reaction flux predictions against experimental data.

This module provides functions to:
- Modify the iML1515 GEM  so its medium and stress conditions match the experimental conditions.
- Create a dataframe that combines experimental fluxomics data with kinGEMs results.
"""

import pandas as pd
import cobra


def apply_ecomics_condition(model: cobra.Model, medium_id: str, stress: str):
    """
    Modify iML1515 GEM based on specified experimental conditions.
    
    Parameters:
        model: cobra.Model
            Genome-scale model to modify
        medium_id : str
            Medium identifier. Options: 'MD066', 'MD120', 'MD004', 'MD121'
        stress : str
            Stress condition. Options: 'none', 'NADH-limitation', 'ATP-limitation'
    
    Returns:
        modified_model: cobra.Model
            Modified GEM model
    """
    
    # Validate input parameters
    valid_medium_ids = {'MD066', 'MD120', 'MD004', 'MD121'}
    valid_stress = {'none', 'NADH-limitation', 'ATP-limitation'}
    
    if medium_id not in valid_medium_ids:
        raise ValueError(f"Invalid medium_id '{medium_id}'. Must be one of {valid_medium_ids}")
    if stress not in valid_stress:
        raise ValueError(f"Invalid stress '{stress}'. Must be one of {valid_stress}")
    
    # Apply medium-specific modifications
    _apply_medium_conditions(model, medium_id)
    
    # Apply stress-specific modifications
    _apply_stress_conditions(model, stress)
    
    
    return model


def get_medium_dict(medium_id: str) -> dict:
    """
    Get medium composition as a dictionary for a given medium ID.
    
    Parameters:
        medium_id : str
            Medium identifier. Options: 'MD066', 'MD120', 'MD004', 'MD121'
    
    Returns:
        dict: Dictionary mapping exchange reaction IDs to their lower bounds (uptake rates)
    """
    
    valid_medium_ids = {'MD066', 'MD120', 'MD004', 'MD121'}
    if medium_id not in valid_medium_ids:
        raise ValueError(f"Invalid medium_id '{medium_id}'. Must be one of {valid_medium_ids}")
    
    if medium_id == 'MD066':
        # synthetic+Glu medium
        # TODO: add medium composition
        return {
            "EX_glc__D_e": -10
        }
        
    elif medium_id == 'MD004':
        # synthetic+Glu medium with higher glucose uptake
        # TODO: add medium composition
        return {
            "EX_glc__D_e": -10
        }
    
    elif medium_id == 'MD120':
        # MOPS+Glu(0.4%) medium
        # TODO: add medium composition
        return {
            "EX_glc__D_e": -10
        }
    
    elif medium_id == 'MD121':
        # M9+Glu medium
        return {
            "EX_glc__D_e": -10,
            "EX_so4_e": -1.699,
            "EX_o2_e": -14.49,
            #"EX_co2_e": 16.22,
            "EX_nh4_e": -5.229,
            "EX_h2o_e": -6.96
        }


def _apply_medium_conditions(model: cobra.Model, medium_id: str) -> None:
    """
    Apply medium-specific modifications to the model.
    
    Parameters:
        model : cobra.Model
            The genome-scale model to modify
        medium_id : str
            Medium identifier
    """
    
    # Get the medium dictionary
    medium = get_medium_dict(medium_id)
    
    # Print which medium is being applied
    medium_names = {
        'MD066': 'synthetic+Glu medium',
        'MD004': 'synthetic+Glu medium with higher glucose uptake',
        'MD120': 'MOPS+Glu(0.4%) medium',
        'MD121': 'M9+Glu medium'
    }
    print(f"Applying {medium_names.get(medium_id, medium_id)}")
    
    # Apply the medium by setting lower bounds
    for rxn_id, uptake in medium.items():
        try:
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = uptake
            print(f"Set {rxn_id} lower bound to {uptake}")
        except KeyError:
            print(f"Reaction {rxn_id} not found in model.")


def _apply_stress_conditions(model: cobra.Model, stress: str) -> None:
    """
    Apply stress-specific modifications to the model.
    
    Parameters:
        model : cobra.Model
            The genome-scale model to modify
        stress : str
            Stress condition
    """
    
    if stress == 'NADH-limitation':
        try:
            # Limit NADH dehydrogenase
            nadh_dehyd = model.reactions.get_by_id("NADH16pp")
            nadh_dehyd.upper_bound = 3
            print("Applied NADH dehydrogenase limitation")
            
            # Add NADH drain reaction
            nadh = model.metabolites.nadh_c
            dm_nadh = cobra.Reaction("DM_nadh_c")
            dm_nadh.name = "NADH drain reaction"
            dm_nadh.add_metabolites({nadh: -1})  # drains 1 NADH
            dm_nadh.lower_bound = dm_nadh.upper_bound = 0.1
            model.add_reactions([dm_nadh])
            print("Added NADH drain reaction")
            
        except KeyError as e:
            print(f"Error applying NADH limitation: {e}")
    
    elif stress == 'ATP-limitation':
        try:
            # Increase ATP maintenance requirement
            atp_maint = model.reactions.get_by_id("ATPM")
            atp_maint.lower_bound = 20 
            atp_maint.upper_bound = 20
            print("Increased ATP maintenance requirement")
            
            # Limit ATP synthase
            atp_synth = model.reactions.get_by_id("ATPS4rpp")
            atp_synth.upper_bound = 5
            print("Limited ATP synthase capacity")
            
        except KeyError as e:
            print(f"Error applying ATP limitation: {e}")
    
    elif stress == 'none':
        # No stress modifications
        pass


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


    
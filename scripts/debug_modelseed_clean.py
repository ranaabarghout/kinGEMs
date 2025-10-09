"""
Debug script for analyzing enzyme constraints in ModelSEED genome models.

This script helps identify why simulated annealing might not be working
for ModelSEED models by analyzing enzyme constraints and kcat mapping.
"""

import json
import os
import sys
from copy import deepcopy

import cobra
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_model_data(model_path, df_path, config_path=None):
    """Load model and dataframe for analysis."""
    print(f"Loading model from: {model_path}")
    model = cobra.io.read_sbml_model(model_path)
    
    print(f"Loading dataframe from: {df_path}")
    df = pd.read_csv(df_path)
    
    config = None
    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    print(f"Model: {model.id}")
    print(f"Model name: {model.name}")
    print(f"Number of reactions: {len(model.reactions)}")
    print(f"Number of metabolites: {len(model.metabolites)}")
    print(f"Number of genes: {len(model.genes)}")
    
    # If config is available, print config info
    if config:
        print(f"Config type: {config.get('type', 'Unknown')}")
        print(f"Config source: {config.get('source', 'Unknown')}")
    
    print(f"\nDataframe shape: {df.shape}")
    if not df.empty:
        print("Dataframe columns:", list(df.columns))
        print("Sample rows:")
        print(df.head(3))
    
    return model, df, config


def analyze_enzyme_constraints(model, df):
    """Analyze enzyme-related reactions and their constraints."""
    print("\n=== Enzyme Constraint Analysis ===")
    
    # Check for enzyme-related reactions
    enzyme_reactions = []
    for rxn in model.reactions:
        if 'enzyme' in rxn.id.lower() or any('enzyme' in met.id.lower() for met in rxn.metabolites):
            enzyme_reactions.append(rxn)
    
    print(f"Found {len(enzyme_reactions)} enzyme-related reactions")
    if enzyme_reactions[:5]:  # Show first 5
        print("Sample enzyme reactions:")
        for rxn in enzyme_reactions[:5]:
            print(f"  {rxn.id}: {rxn.reaction}")
    
    # Check reaction bounds
    constrained_reactions = [rxn for rxn in model.reactions if rxn.upper_bound < 1000]
    print(f"\nFound {len(constrained_reactions)} reactions with upper bounds < 1000")
    
    # Analyze dataframe for enzyme requirements
    if 'kcat_MW' in df.columns:
        print(f"\nDataframe has kcat_MW column with {df['kcat_MW'].notna().sum()} non-null values")
        kcat_stats = df['kcat_MW'].describe()
        print("kcat_MW statistics:")
        print(kcat_stats)
    
    if 'enzyme_constraint' in df.columns:
        print(f"\nDataframe has enzyme_constraint column with {df['enzyme_constraint'].notna().sum()} non-null values")
        constraint_stats = df['enzyme_constraint'].describe()
        print("Enzyme constraint statistics:")
        print(constraint_stats)
    
    return enzyme_reactions, constrained_reactions


def check_gene_reaction_mapping(model, df):
    """Check gene-reaction mapping between model and dataframe."""
    print("\n=== Gene-Reaction Mapping Analysis ===")
    
    # Get reactions with genes in model
    model_rxns_with_genes = {rxn.id for rxn in model.reactions if rxn.genes}
    print(f"Model reactions with genes: {len(model_rxns_with_genes)}")
    
    # Get reactions in dataframe
    if 'reaction_id' in df.columns:
        df_reactions = set(df['reaction_id'].dropna())
        print(f"Dataframe reactions: {len(df_reactions)}")
        
        # Find overlap
        overlap = model_rxns_with_genes.intersection(df_reactions)
        print(f"Overlapping reactions: {len(overlap)}")
        
        # Find missing
        missing_in_df = model_rxns_with_genes - df_reactions
        missing_in_model = df_reactions - model_rxns_with_genes
        
        print(f"Model reactions missing in dataframe: {len(missing_in_df)}")
        print(f"Dataframe reactions missing in model: {len(missing_in_model)}")
        
        if missing_in_df and len(missing_in_df) <= 10:
            print("Sample missing in dataframe:", list(missing_in_df)[:10])
        if missing_in_model and len(missing_in_model) <= 10:
            print("Sample missing in model:", list(missing_in_model)[:10])
        
        return overlap, missing_in_df, missing_in_model
    else:
        print("No 'reaction_id' column found in dataframe")
        return set(), set(), set()


def test_constraint_impact(model, df, enzyme_upper_bound=0.15):
    """Test if enzyme constraints actually impact the solution."""
    print("\n=== Constraint Impact Testing ===")
    
    try:
        # Store original bounds
        original_bounds = {}
        for rxn in model.reactions:
            original_bounds[rxn.id] = (rxn.lower_bound, rxn.upper_bound)
        
        # Run optimization without constraints
        print("Running FBA without enzyme constraints...")
        model_unconstrained = deepcopy(model)
        solution_unconstrained = model_unconstrained.optimize()
        
        if solution_unconstrained.status == 'optimal':
            obj_unconstrained = solution_unconstrained.objective_value
            print(f"Unconstrained objective value: {obj_unconstrained}")
        else:
            print(f"Unconstrained optimization failed: {solution_unconstrained.status}")
            return None
        
        # Apply enzyme constraints if dataframe has the right columns
        if 'enzyme_constraint' in df.columns and 'reaction_id' in df.columns:
            print("Applying enzyme constraints...")
            model_constrained = deepcopy(model)
            
            constraints_applied = 0
            for _, row in df.iterrows():
                rxn_id = row['reaction_id']
                if pd.notna(row['enzyme_constraint']) and rxn_id in model_constrained.reactions:
                    rxn = model_constrained.reactions.get_by_id(rxn_id)
                    new_bound = min(rxn.upper_bound, row['enzyme_constraint'])
                    if new_bound < rxn.upper_bound:
                        rxn.upper_bound = new_bound
                        constraints_applied += 1
            
            print(f"Applied {constraints_applied} enzyme constraints")
            
            # Run constrained optimization
            solution_constrained = model_constrained.optimize()
            
            if solution_constrained.status == 'optimal':
                obj_constrained = solution_constrained.objective_value
                print(f"Constrained objective value: {obj_constrained}")
                
                change = abs(obj_constrained - obj_unconstrained)
                print(f"Objective value change: {change}")
                print(f"Relative change: {change/abs(obj_unconstrained)*100:.2f}%")
                
                if change < 1e-6:
                    print("WARNING: Constraints had negligible impact on objective!")
                    return analyze_why_no_impact(model_constrained, df)
                else:
                    print("Constraints successfully impacted the solution")
            else:
                print(f"Constrained optimization failed: {solution_constrained.status}")
        
        else:
            print("Dataframe missing required columns for constraint testing")
            print("Available columns:", list(df.columns))
        
        # Restore original bounds
        for rxn_id, (lb, ub) in original_bounds.items():
            if rxn_id in model.reactions:
                rxn = model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = lb
                rxn.upper_bound = ub
        
    except Exception as e:
        print(f"Error in constraint testing: {e}")
        return None


def analyze_why_no_impact(model, df):
    """Analyze why constraints had no impact."""
    print("\n=== Analyzing Why Constraints Had No Impact ===")
    
    # Check if constraints are actually binding
    solution = model.optimize()
    if solution.status != 'optimal':
        print("Model is not optimal, cannot analyze flux values")
        return
    
    binding_constraints = 0
    loose_constraints = 0
    
    for _, row in df.iterrows():
        if pd.notna(row.get('enzyme_constraint')) and row.get('reaction_id') in model.reactions:
            rxn = model.reactions.get_by_id(row['reaction_id'])
            flux = solution.fluxes[rxn.id]
            constraint = row['enzyme_constraint']
            
            if abs(flux) >= constraint * 0.99:  # Close to constraint
                binding_constraints += 1
            else:
                loose_constraints += 1
    
    print(f"Binding constraints: {binding_constraints}")
    print(f"Loose constraints: {loose_constraints}")
    
    if binding_constraints == 0:
        print("No constraints are binding - they may be too loose!")
        
        # Suggest tighter constraints
        print("\nAnalyzing constraint tightness...")
        current_fluxes = solution.fluxes
        
        tight_needed = 0
        for _, row in df.iterrows():
            if (pd.notna(row.get('enzyme_constraint')) and 
                row.get('reaction_id') in model.reactions):
                rxn_id = row['reaction_id']
                flux = abs(current_fluxes[rxn_id])
                constraint = row['enzyme_constraint']
                
                if flux > 0 and constraint > flux * 2:  # Constraint is more than 2x current flux
                    tight_needed += 1
        
        print(f"Reactions needing tighter constraints: {tight_needed}")


def check_kcat_values(df):
    """Analyze kcat values in the dataframe."""
    print("\n=== kcat Value Analysis ===")
    
    kcat_cols = [col for col in df.columns if 'kcat' in col.lower()]
    print(f"Found kcat-related columns: {kcat_cols}")
    
    for col in kcat_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"\n{col}:")
            print(f"  Non-null values: {non_null}/{len(df)}")
            
            if non_null > 0:
                stats = df[col].describe()
                print(f"  Min: {stats['min']:.2e}")
                print(f"  Max: {stats['max']:.2e}")
                print(f"  Mean: {stats['mean']:.2e}")
                print(f"  Median: {stats['50%']:.2e}")
                
                # Check for unrealistic values
                very_low = (df[col] < 1e-3).sum()
                very_high = (df[col] > 1e6).sum()
                print(f"  Very low values (<1e-3): {very_low}")
                print(f"  Very high values (>1e6): {very_high}")


def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python debug_modelseed_clean.py <config.json>")
        print("\nConfig file should contain:")
        print("  - model_path: path to SBML model file")
        print("  - df_path: path to processed dataframe CSV")
        print("  - model_name: name for display")
        print("  - enzyme_upper_bound: enzyme constraint upper bound (optional)")
        return
    
    config_path = sys.argv[1]
    config = load_config(config_path)
    
    model_name = config['model_name']
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.15)
    
    print("="*70)
    print(f"=== Debugging ModelSEED Constraints for {model_name} ===")
    print("="*70)
    
    # Load data
    model, df, _ = load_model_data(
        config['model_path'], 
        config['df_path'],
        config_path
    )
    
    # Run analyses
    print("\n" + "="*50)
    print("STARTING COMPREHENSIVE ANALYSIS")
    print("="*50)
    
    # 1. Analyze enzyme constraints
    enzyme_rxns, constrained_rxns = analyze_enzyme_constraints(model, df)
    
    # 2. Check gene-reaction mapping
    overlap, missing_df, missing_model = check_gene_reaction_mapping(model, df)
    
    # 3. Analyze kcat values
    check_kcat_values(df)
    
    # 4. Test constraint impact
    test_constraint_impact(model, df, enzyme_upper_bound)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    
    # Summary recommendations
    print("\n=== RECOMMENDATIONS ===")
    
    if len(enzyme_rxns) == 0:
        print("1. No enzyme reactions found - check if model includes enzyme constraints")
    
    if len(overlap) == 0:
        print("2. No reaction overlap between model and dataframe - check ID matching")
    
    if 'enzyme_constraint' not in df.columns:
        print("3. Missing enzyme_constraint column - run constraint calculation")
    
    if len(constrained_rxns) == 0:
        print("4. No constrained reactions found - constraints may not be applied")


if __name__ == "__main__":
    main()

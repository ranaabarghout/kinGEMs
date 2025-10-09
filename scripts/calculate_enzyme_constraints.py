"""
Script to calculate enzyme constraints for ModelSEED models.
This will generate the missing enzyme_constraint column needed for simulated annealing.
"""

import json
import os
import sys
from copy import deepcopy

import cobra
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.dataset import annotate_model_with_kcat_and_gpr, load_model


def calculate_enzyme_constraints(model_path, df_path, output_path, enzyme_upper_bound=0.15):
    """Calculate enzyme constraints and save updated dataframe."""
    
    print(f"Loading model from: {model_path}")
    model = cobra.io.read_sbml_model(model_path)
    
    print(f"Loading dataframe from: {df_path}")
    df = pd.read_csv(df_path)
    
    print(f"Original dataframe shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Ensure we have a 'kcat' column
    if 'kcat_mean' in df.columns:
        df['kcat'] = df['kcat_mean']
        print("Set df['kcat'] = df['kcat_mean']")
    elif 'kcat_y' in df.columns:
        df['kcat'] = df['kcat_y']
        print("Set df['kcat'] = df['kcat_y']")
    else:
        raise ValueError("No kcat_mean or kcat_y column found!")
    
    # Load the model in irreversible form
    irrev_model = load_model(model_path)
    
    # Annotate model with kcat values
    print("Annotating model with kcat values...")
    irrev_model = annotate_model_with_kcat_and_gpr(
        model=irrev_model,
        df=df
    )
    
    # Find biomass reaction
    biomass_reactions = [rxn for rxn in irrev_model.reactions if 'biomass' in rxn.id.lower()]
    if not biomass_reactions:
        # Look for growth or other objective reactions
        biomass_reactions = [rxn for rxn in irrev_model.reactions if any(term in rxn.id.lower() 
                            for term in ['growth', 'objective', 'bio'])]
    
    if biomass_reactions:
        biomass_reaction = biomass_reactions[0].id
        print(f"Using biomass reaction: {biomass_reaction}")
    else:
        print("No biomass reaction found, using model objective")
        biomass_reaction = irrev_model.objective.direction
    
    # Run optimization to calculate enzyme constraints
    print("Running optimization to calculate enzyme constraints...")
    solution_value, df_FBA, gene_sequences_dict, _ = run_optimization_with_dataframe(
        model=irrev_model,
        processed_df=df,
        objective_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        enzyme_ratio=True,
        save_results=False,
        verbose=False
    )
    
    print(f"Optimization successful! Objective value: {solution_value}")
    print(f"Generated FBA dataframe shape: {df_FBA.shape}")
    print(f"New columns in FBA dataframe: {list(df_FBA.columns)}")
    
    # Save the updated dataframe with enzyme constraints
    print(f"Saving updated dataframe to: {output_path}")
    df_FBA.to_csv(output_path, index=False)
    
    # Also save gene sequences dict if needed
    gene_seq_path = output_path.replace('.csv', '_gene_sequences.json')
    with open(gene_seq_path, 'w') as f:
        json.dump(gene_sequences_dict, f, indent=2)
    
    print(f"✓ Enzyme constraints calculated and saved!")
    print(f"✓ Gene sequences saved to: {gene_seq_path}")
    
    return df_FBA


def main():
    if len(sys.argv) < 4:
        print("Usage: python calculate_enzyme_constraints.py <model.xml> <processed_data.csv> <output.csv> [enzyme_upper_bound]")
        return
    
    model_path = sys.argv[1]
    df_path = sys.argv[2] 
    output_path = sys.argv[3]
    enzyme_upper_bound = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
    
    print("="*70)
    print("=== Calculating Enzyme Constraints for ModelSEED Model ===")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Input dataframe: {df_path}")
    print(f"Output dataframe: {output_path}")
    print(f"Enzyme upper bound: {enzyme_upper_bound}")
    print()
    
    try:
        df_with_constraints = calculate_enzyme_constraints(
            model_path, df_path, output_path, enzyme_upper_bound
        )
        
        print("\n" + "="*50)
        print("SUCCESS: Enzyme constraints calculated!")
        print("="*50)
        print(f"Updated dataframe saved to: {output_path}")
        print("You can now use this file for simulated annealing.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Script to add enzyme constraints to ModelSEED processed dataframes.
This creates the missing enzyme_constraint and kcat_MW columns needed for simulated annealing.
"""

import os
import sys

import cobra
import numpy as np
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def calculate_molecular_weight(sequence):
    """Calculate molecular weight of a protein sequence (simplified)."""
    # Average amino acid molecular weight is approximately 110 Da
    # Remove invalid characters and calculate
    valid_sequence = ''.join(c for c in sequence.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')
    return len(valid_sequence) * 110.0  # Da


def add_enzyme_constraints_to_dataframe(model_path, df_path, output_path, enzyme_upper_bound=0.15):
    """Add enzyme constraint calculations to the processed dataframe."""

    print(f"Loading model from: {model_path}")
    model = cobra.io.read_sbml_model(model_path)

    print(f"Loading dataframe from: {df_path}")
    df = pd.read_csv(df_path).copy()

    print(f"Original dataframe shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Ensure we have a 'kcat' column
    if 'kcat_mean' in df.columns:
        df['kcat'] = df['kcat_mean']
        print("Using kcat_mean as kcat values")
    elif 'kcat_y' in df.columns:
        df['kcat'] = df['kcat_y']
        print("Using kcat_y as kcat values")
    else:
        raise ValueError("No kcat_mean or kcat_y column found!")

    # Calculate molecular weights for sequences
    print("Calculating molecular weights...")
    df['MW'] = df['SEQ'].apply(lambda seq: calculate_molecular_weight(seq) if pd.notna(seq) else np.nan)

    # Calculate kcat/MW (enzyme efficiency)
    print("Calculating kcat/MW ratios...")
    df['kcat_MW'] = df['kcat'] / (df['MW'] / 1000)  # Convert MW from Da to kDa

    # Calculate enzyme requirements (1/kcat_MW)
    print("Calculating enzyme requirements...")
    df['enzyme_requirement'] = 1.0 / df['kcat_MW']

    # Group by reaction and calculate aggregate enzyme constraints
    print("Calculating reaction-level enzyme constraints...")

    # For each reaction, sum the enzyme requirements of all enzyme-substrate pairs
    reaction_constraints = df.groupby('Reactions').agg({
        'enzyme_requirement': 'sum',
        'kcat_MW': 'mean',  # Average efficiency
        'kcat': 'mean'      # Average kcat
    }).reset_index()

    # Merge back to original dataframe
    df = df.merge(reaction_constraints, on='Reactions', suffixes=('', '_reaction'))

    # The enzyme constraint is the total enzyme requirement per reaction
    df['enzyme_constraint'] = df['enzyme_requirement_reaction']

    # Add reaction_id column for compatibility
    df['reaction_id'] = df['Reactions']

    # Clean up intermediate columns
    df = df.drop(['enzyme_requirement_reaction', 'kcat_MW_reaction', 'kcat_reaction'], axis=1)

    print(f"Final dataframe shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")

    # Show some statistics
    print(f"\nConstraint Statistics:")
    print(f"Reactions with constraints: {df['enzyme_constraint'].notna().sum()}")
    print(f"Unique reactions: {df['reaction_id'].nunique()}")

    if df['enzyme_constraint'].notna().sum() > 0:
        constraint_stats = df['enzyme_constraint'].describe()
        print(f"Enzyme constraint range: {constraint_stats['min']:.2e} to {constraint_stats['max']:.2e}")
        print(f"Enzyme constraint median: {constraint_stats['50%']:.2e}")

    # Save the updated dataframe
    print(f"Saving updated dataframe to: {output_path}")
    df.to_csv(output_path, index=False)

    print(f"✓ Enzyme constraints added to dataframe!")
    return df


def main():
    if len(sys.argv) < 4:
        print("Usage: python add_enzyme_constraints.py <model.xml> <processed_data.csv> <output.csv> [enzyme_upper_bound]")
        return

    model_path = sys.argv[1]
    df_path = sys.argv[2]
    output_path = sys.argv[3]
    enzyme_upper_bound = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15

    print("="*70)
    print("=== Adding Enzyme Constraints to ModelSEED Dataframe ===")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Input dataframe: {df_path}")
    print(f"Output dataframe: {output_path}")
    print(f"Enzyme upper bound: {enzyme_upper_bound}")
    print()

    try:
        df_with_constraints = add_enzyme_constraints_to_dataframe(
            model_path, df_path, output_path, enzyme_upper_bound
        )

        print("\n" + "="*50)
        print("SUCCESS: Enzyme constraints added!")
        print("="*50)
        print(f"Updated dataframe saved to: {output_path}")
        print("You can now use this file for simulated annealing.")
        print("\nKey new columns added:")
        print("- enzyme_constraint: Total enzyme requirement per reaction")
        print("- kcat_MW: Enzyme efficiency (kcat/MW)")
        print("- MW: Molecular weight of enzyme")
        print("- reaction_id: Reaction identifier for constraint mapping")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

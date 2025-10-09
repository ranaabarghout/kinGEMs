#!/usr/bin/env python3
"""
Debug script for ModelSEED enzyme constraints
============================================

This script helps debug why simulated annealing might not be working
for ModelSEED models (_genome_ models) by checking:

1. kcat coverage and values
2. Gene-reaction mappings
3. Enzyme constraint effectiveness
4. Model structure differences

Usage:
    python scripts/debug_modelseed_constraints.py <config_file>
"""

"""
Debug script for analyzing enzyme constraints in ModelSEED genome models.

This script helps identify why simulated annealing might not be working
for ModelSEED models by analyzing enzyme constraints and kcat mapping.
"""

import json
import os
import sys
import warnings
from copy import deepcopy

import cobra
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.modeling.optimize import run_optimization_with_dataframe


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

    return model, df, config

import json
import os
import sys
import warnings
from copy import deepcopy

import cobra
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.modeling.optimize import run_optimization_with_dataframe

warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def debug_gene_reaction_mapping(model, processed_df):
    """Debug gene-reaction mappings between model and processed data."""
    print("\n=== Gene-Reaction Mapping Debug ===")

    # Get genes from model
    model_genes = set(gene.id for gene in model.genes)
    print(f"Model genes: {len(model_genes)}")

    # Get genes from processed data
    data_genes = set(processed_df['Single_gene'].dropna().unique())
    print(f"Processed data genes: {len(data_genes)}")

    # Find overlaps
    common_genes = model_genes.intersection(data_genes)
    model_only = model_genes - data_genes
    data_only = data_genes - model_genes

    print(f"Common genes: {len(common_genes)}")
    print(f"Model-only genes: {len(model_only)}")
    print(f"Data-only genes: {len(data_only)}")

    if len(common_genes) == 0:
        print("⚠️  WARNING: No common genes found between model and data!")
        print("Sample model genes:", list(model_genes)[:10])
        print("Sample data genes:", list(data_genes)[:10])
        return False

    # Check reactions from processed data
    model_reactions = set(rxn.id for rxn in model.reactions)
    data_reactions = set(processed_df['Reactions'].dropna().unique())

    print(f"\nModel reactions: {len(model_reactions)}")
    print(f"Data reactions: {len(data_reactions)}")

    common_reactions = model_reactions.intersection(data_reactions)
    print(f"Common reactions: {len(common_reactions)}")

    if len(common_reactions) == 0:
        print("⚠️  WARNING: No common reactions found between model and data!")
        print("Sample model reactions:", list(model_reactions)[:10])
        print("Sample data reactions:", list(data_reactions)[:10])
        return False

    return True


def debug_kcat_values(processed_df):
    """Debug kcat values in processed data."""
    print("\n=== kcat Values Debug ===")

    # Check for kcat columns
    kcat_columns = [col for col in processed_df.columns if 'kcat' in col.lower()]
    print(f"kcat columns found: {kcat_columns}")

    if not kcat_columns:
        print("⚠️  ERROR: No kcat columns found in processed data!")
        return False

    # Check main kcat column
    if 'kcat' in processed_df.columns:
        kcat_col = 'kcat'
    elif 'kcat_mean' in processed_df.columns:
        kcat_col = 'kcat_mean'
        processed_df['kcat'] = processed_df['kcat_mean']
    elif 'kcat_y' in processed_df.columns:
        kcat_col = 'kcat_y'
        processed_df['kcat'] = processed_df['kcat_y']
    else:
        print("⚠️  ERROR: No usable kcat column found!")
        return False

    kcat_values = processed_df[kcat_col].dropna()
    print(f"kcat column used: {kcat_col}")
    print(f"Non-null kcat values: {len(kcat_values)}/{len(processed_df)}")
    print(f"kcat range: {kcat_values.min():.2e} to {kcat_values.max():.2e}")
    print(f"kcat median: {kcat_values.median():.2e}")

    # Check for very small or very large values
    very_small = (kcat_values < 1e-6).sum()
    very_large = (kcat_values > 1e6).sum()
    print(f"Very small kcat (<1e-6): {very_small}")
    print(f"Very large kcat (>1e6): {very_large}")

    if very_small > len(kcat_values) * 0.5:
        print("⚠️  WARNING: Many kcat values are very small!")

    return True


def test_enzyme_constraints(model, processed_df, biomass_reaction, enzyme_upper_bound):
    """Test if enzyme constraints are actually affecting the solution."""
    print("\n=== Enzyme Constraint Effectiveness Test ===")

    # Run optimization without enzyme constraints (regular FBA)
    model_copy = deepcopy(model)
    try:
        fba_solution = model_copy.optimize()
        fba_biomass = fba_solution.objective_value
        print(f"Regular FBA biomass: {fba_biomass:.6f}")
    except Exception as e:
        print(f"⚠️  Regular FBA failed: {e}")
        return False

    # Run optimization with enzyme constraints
    try:
        ec_biomass, df_FBA, _, _ = run_optimization_with_dataframe(
            model=model,
            processed_df=processed_df,
            objective_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            enzyme_ratio=True,
            maximization=True,
            save_results=False,
            verbose=True
        )
        print(f"Enzyme-constrained biomass: {ec_biomass:.6f}")
    except Exception as e:
        print(f"⚠️  Enzyme-constrained optimization failed: {e}")
        return False

    # Compare results
    if abs(fba_biomass - ec_biomass) < 1e-8:
        print("⚠️  WARNING: No difference between FBA and enzyme-constrained FBA!")
        print("This suggests enzyme constraints are not active.")

        # Try with stricter enzyme bound
        print("\nTesting with stricter enzyme bound...")
        try:
            strict_biomass, _, _, _ = run_optimization_with_dataframe(
                model=model,
                processed_df=processed_df,
                objective_reaction=biomass_reaction,
                enzyme_upper_bound=enzyme_upper_bound * 0.1,  # 10x stricter
                enzyme_ratio=True,
                maximization=True,
                save_results=False,
                verbose=False
            )
            print(f"Strict enzyme-constrained biomass: {strict_biomass:.6f}")

            if abs(fba_biomass - strict_biomass) > 1e-8:
                print("✓ Stricter constraints do affect the solution.")
                print("Issue: Original enzyme bound may be too loose.")
            else:
                print("⚠️  Even strict constraints don't affect the solution.")
                print("Issue: kcat values may not be properly applied as constraints.")
        except Exception as e:
            print(f"⚠️  Strict constraint test failed: {e}")

        return False
    else:
        print("✓ Enzyme constraints are affecting the solution.")
        print(f"Difference: {abs(fba_biomass - ec_biomass):.6f}")
        return True


def analyze_enzyme_usage(df_FBA):
    """Analyze enzyme usage from FBA results."""
    print("\n=== Enzyme Usage Analysis ===")

    if df_FBA is None:
        print("⚠️  No FBA results to analyze")
        return

    # Filter enzyme variables
    enzyme_df = df_FBA[df_FBA['Variable'] == 'enzyme'].copy()

    if len(enzyme_df) == 0:
        print("⚠️  No enzyme variables found in FBA results")
        return

    print(f"Total enzyme variables: {len(enzyme_df)}")

    # Analyze usage
    active_enzymes = enzyme_df[enzyme_df['Value'] > 1e-8]
    print(f"Active enzymes (>1e-8): {len(active_enzymes)}")

    if len(active_enzymes) > 0:
        print(f"Max enzyme usage: {active_enzymes['Value'].max():.6f}")
        print(f"Total enzyme usage: {active_enzymes['Value'].sum():.6f}")

        # Show top enzymes
        print("\nTop 10 enzyme usages:")
        top_enzymes = active_enzymes.nlargest(10, 'Value')[['Index', 'Value']]
        for _, row in top_enzymes.iterrows():
            print(f"  {row['Index']}: {row['Value']:.6f}")
    else:
        print("⚠️  No active enzymes found!")


def check_model_structure(model):
    """Check model structure for potential issues."""
    print("\n=== Model Structure Check ===")

    print(f"Model: {model.id}")
    print(f"Genes: {len(model.genes)}")
    print(f"Reactions: {len(model.reactions)}")
    print(f"Metabolites: {len(model.metabolites)}")

    # Check for reactions with genes
    reactions_with_genes = [rxn for rxn in model.reactions if rxn.genes]
    print(f"Reactions with genes: {len(reactions_with_genes)}")

    # Check for reactions without genes
    reactions_without_genes = [rxn for rxn in model.reactions if not rxn.genes]
    print(f"Reactions without genes: {len(reactions_without_genes)}")

    # Check for exchange reactions
    exchange_reactions = model.exchanges
    print(f"Exchange reactions: {len(exchange_reactions)}")

    # Check objective reaction
    objective_rxns = [rxn for rxn in model.reactions if rxn.objective_coefficient != 0]
    print(f"Objective reactions: {[rxn.id for rxn in objective_rxns]}")

    # Sample gene IDs
    if model.genes:
        sample_genes = [gene.id for gene in list(model.genes)[:10]]
        print(f"Sample gene IDs: {sample_genes}")

    # Sample reaction IDs
    sample_reactions = [rxn.id for rxn in list(model.reactions)[:10]]
    print(f"Sample reaction IDs: {sample_reactions}")


def debug_processed_data_structure(processed_df):
    """Debug the structure of processed data."""
    print("\n=== Processed Data Structure Debug ===")

    print(f"Shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")

    # Check required columns
    required_cols = ['Reactions', 'Single_gene', 'SEQ']
    missing_cols = [col for col in required_cols if col not in processed_df.columns]
    if missing_cols:
        print(f"⚠️  Missing required columns: {missing_cols}")

    # Check for null values
    print("\nNull value counts:")
    for col in processed_df.columns:
        null_count = processed_df[col].isnull().sum()
        if null_count > 0:
            print(f"  {col}: {null_count}/{len(processed_df)}")

    # Sample data
    print("\nSample rows:")
    print(processed_df.head())


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable config files:")
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
        if os.path.exists(config_dir):
            for f in os.listdir(config_dir):
                if f.endswith('.json'):
                    print(f"  configs/{f}")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    model_name = config['model_name']
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.15)

    print("="*70)
    print(f"=== Debugging ModelSEED Constraints for {model_name} ===")
    print("="*70)

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")
    raw_data_dir = os.path.join(data_dir, "raw")

    model_path = os.path.join(raw_data_dir, f"{model_name}.xml")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        sys.exit(1)

    # Load model
    print(f"\nLoading model: {model_path}")
    model = cobra.io.read_sbml_model(model_path)

    # Determine biomass reaction
    objective_rxns = [rxn.id for rxn in model.reactions if rxn.objective_coefficient != 0]
    biomass_reaction = config.get('biomass_reaction') or objective_rxns[0]
    print(f"Biomass reaction: {biomass_reaction}")

    # Check model structure
    check_model_structure(model)

    # Try to load processed data
    processed_data_dir = os.path.join(data_dir, "processed", model_name)
    processed_data_path = os.path.join(processed_data_dir, f"{model_name}_processed_data.csv")

    if os.path.exists(processed_data_path):
        print(f"\nLoading processed data: {processed_data_path}")
        processed_df = pd.read_csv(processed_data_path)

        # Ensure kcat column exists
        if 'kcat_mean' in processed_df.columns and 'kcat' not in processed_df.columns:
            processed_df['kcat'] = processed_df['kcat_mean']
        elif 'kcat_y' in processed_df.columns and 'kcat' not in processed_df.columns:
            processed_df['kcat'] = processed_df['kcat_y']

        # Debug processed data
        debug_processed_data_structure(processed_df)

        # Debug gene-reaction mappings
        mapping_ok = debug_gene_reaction_mapping(model, processed_df)

        # Debug kcat values
        kcat_ok = debug_kcat_values(processed_df)

        if mapping_ok and kcat_ok:
            # Test enzyme constraints
            constraints_ok = test_enzyme_constraints(model, processed_df, biomass_reaction, enzyme_upper_bound)

            if constraints_ok:
                print("\n✓ Basic enzyme constraints are working.")
                print("The issue may be in the simulated annealing logic.")
            else:
                print("\n⚠️  Enzyme constraints are not effective.")
                print("This explains why simulated annealing shows no change.")

    else:
        print(f"⚠️  Processed data file not found: {processed_data_path}")
        print("Run the full pipeline first to generate processed data.")

    print("\n" + "="*70)
    print("=== Debug Complete ===")
    print("="*70)


if __name__ == '__main__':
    main()

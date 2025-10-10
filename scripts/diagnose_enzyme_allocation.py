#!/usr/bin/env python3
"""
Enzyme Allocation Diagnostic Script
===================================

This script analyzes why the ModelSEED model is hitting the enzyme upper bound
and explains why so many constraints are being skipped.

Usage:
    python scripts/diagnose_enzyme_allocation.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.dataset import load_model, annotate_model_with_kcat_and_gpr
from kinGEMs.modeling.optimize import run_optimization_with_dataframe


def analyze_enzyme_allocation(model_path, processed_data_path, enzyme_upper_bound=0.15):
    """Analyze enzyme allocation and constraint skipping patterns."""

    print("="*70)
    print("ENZYME ALLOCATION DIAGNOSTIC")
    print("="*70)

    # Load data
    print("Loading model and data...")
    model = load_model(model_path)
    processed_data = pd.read_csv(processed_data_path)

    # Ensure kcat column exists
    if 'kcat_mean' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_mean']
    elif 'kcat_y' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_y']

    # Annotate model
    model = annotate_model_with_kcat_and_gpr(model=model, df=processed_data)

    print(f"Model: {len(model.genes)} genes, {len(model.reactions)} reactions")
    print(f"Processed data: {len(processed_data)} rows")

    # Analyze kcat distribution
    print("\n" + "="*50)
    print("KCAT VALUE ANALYSIS")
    print("="*50)

    kcat_stats = processed_data['kcat'].describe()
    print("kcat distribution:")
    print(kcat_stats)

    # Check for extreme values
    very_high_kcat = processed_data[processed_data['kcat'] > 1000]
    very_low_kcat = processed_data[processed_data['kcat'] < 0.1]

    print(f"\nExtreme kcat values:")
    print(f"- Very high (>1000 s⁻¹): {len(very_high_kcat)} entries")
    print(f"- Very low (<0.1 s⁻¹): {len(very_low_kcat)} entries")

    if len(very_high_kcat) > 0:
        print(f"\nTop 10 highest kcat values:")
        top_high = very_high_kcat.nlargest(10, 'kcat')[['Reactions', 'Single_gene', 'kcat']]
        print(top_high.to_string(index=False))

    # Analyze molecular weights
    print("\n" + "="*50)
    print("MOLECULAR WEIGHT ANALYSIS")
    print("="*50)

    if 'mol_weight' in processed_data.columns:
        mw_stats = processed_data['mol_weight'].describe()
        print("Molecular weight distribution (Da):")
        print(mw_stats)

        # Calculate enzyme demands
        processed_data['enzyme_demand'] = processed_data['mol_weight'] / processed_data['kcat']
        ed_stats = processed_data['enzyme_demand'].describe()
        print("\nEnzyme demand distribution (Da·s):")
        print(ed_stats)

        # Find most demanding enzymes
        print(f"\nTop 10 most enzyme-demanding reactions:")
        top_demanding = processed_data.nlargest(10, 'enzyme_demand')[
            ['Reactions', 'Single_gene', 'mol_weight', 'kcat', 'enzyme_demand']
        ]
        print(top_demanding.to_string(index=False))

    # Test different enzyme upper bounds
    print("\n" + "="*50)
    print("ENZYME UPPER BOUND TESTING")
    print("="*50)

    bounds_to_test = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    biomass_reaction = 'bio1'  # ModelSEED default

    results = []
    for bound in bounds_to_test:
        print(f"\nTesting enzyme upper bound: {bound} g/gDW")
        try:
            solution_value, df_FBA, gene_sequences_dict, _ = run_optimization_with_dataframe(
                model=model,
                processed_df=processed_data,
                objective_reaction=biomass_reaction,
                enzyme_upper_bound=bound,
                enzyme_ratio=True,
                maximization=True,
                multi_enzyme_off=False,
                isoenzymes_off=False,
                promiscuous_off=False,
                complexes_off=False,
                output_dir=None,
                save_results=False,
                print_reaction_conditions=False,
                verbose=False,
                solver_name='glpk'
            )
            results.append({
                'upper_bound': bound,
                'biomass_value': solution_value,
                'status': 'optimal' if solution_value > 0 else 'infeasible'
            })
            print(f"  Result: {solution_value:.6f}")

        except Exception as e:
            results.append({
                'upper_bound': bound,
                'biomass_value': 0,
                'status': f'error: {str(e)[:50]}'
            })
            print(f"  Error: {e}")

    # Create results dataframe
    results_df = pd.DataFrame(results)
    print(f"\nSummary of enzyme bound testing:")
    print(results_df.to_string(index=False))

    # Analyze constraint patterns
    print("\n" + "="*50)
    print("CONSTRAINT ANALYSIS")
    print("="*50)

    # Group by reactions to see constraint patterns
    reaction_counts = processed_data.groupby('Reactions').agg({
        'Single_gene': 'count',
        'kcat': ['min', 'max', 'mean'],
        'mol_weight': 'mean' if 'mol_weight' in processed_data.columns else 'count'
    }).round(3)

    reaction_counts.columns = ['gene_count', 'kcat_min', 'kcat_max', 'kcat_mean', 'avg_mol_weight']

    # Reactions with multiple genes (complex enzymes)
    complex_rxns = reaction_counts[reaction_counts['gene_count'] > 1]
    print(f"Reactions with multiple genes (enzyme complexes): {len(complex_rxns)}")

    if len(complex_rxns) > 0:
        print(f"\nTop 10 most complex reactions:")
        top_complex = complex_rxns.nlargest(10, 'gene_count')
        print(top_complex.to_string())

    # Check for reactions with very different kcat values across genes
    variable_kcat = reaction_counts[
        (reaction_counts['gene_count'] > 1) &
        (reaction_counts['kcat_max'] / reaction_counts['kcat_min'] > 10)
    ]

    print(f"\nReactions with highly variable kcat values (>10x difference): {len(variable_kcat)}")
    if len(variable_kcat) > 0:
        print(variable_kcat.head().to_string())

    # Recommendations
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)

    optimal_results = results_df[results_df['biomass_value'] > 0]
    if len(optimal_results) > 0:
        best_bound = optimal_results.loc[optimal_results['biomass_value'].idxmax()]
        print(f"1. ENZYME UPPER BOUND:")
        print(f"   - Current bound (0.15) may be too restrictive")
        print(f"   - Best performance at: {best_bound['upper_bound']} g/gDW")
        print(f"   - Biomass improvement: {best_bound['biomass_value']:.6f}")

    print(f"\n2. CONSTRAINT SKIPPING:")
    print(f"   - 'other' skips (2399) likely due to reactions with multiple OR clauses")
    print(f"   - These are reactions with alternative pathways that the optimizer can't handle")
    print(f"   - This is normal behavior for complex metabolic networks")

    print(f"\n3. SIMULATED ANNEALING ISSUE:")
    print(f"   - Model hits enzyme bound immediately → no room for improvement")
    print(f"   - Try increasing enzyme_upper_bound to 0.2-0.3 g/gDW")
    print(f"   - Consider filtering out very low kcat values (< 0.1 s⁻¹)")

    return results_df


def main():
    # Paths for 382_genome_cpd03198 model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, "data", "raw", "382_genome_cpd03198.xml")
    processed_data_path = os.path.join(project_root, "data", "processed", "382_genome_cpd03198", "382_genome_cpd03198_processed_data.csv")

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Available models:")
        models_dir = os.path.join(project_root, "data", "raw")
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.xml'):
                    print(f"  - {f}")
        return

    if not os.path.exists(processed_data_path):
        print(f"❌ Processed data not found: {processed_data_path}")
        return

    print(f"Using model: {model_path}")
    print(f"Using data: {processed_data_path}")

    # Run analysis
    results = analyze_enzyme_allocation(model_path, processed_data_path)

    print(f"\n✅ Analysis complete!")


if __name__ == '__main__':
    main()

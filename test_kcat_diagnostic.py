#!/usr/bin/env python3
"""
Diagnostic test to verify kcat changes affect optimization.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from kinGEMs.dataset import load_model
from kinGEMs.modeling.optimize import run_optimization_with_dataframe

def test_kcat_effect():
    # Load the processed data
    processed_data_path = "/project/def-mahadeva/ranaab/kinGEMs_v2/data/processed/iML1515_GEM/iML1515_GEM_processed_data.csv"
    processed_df = pd.read_csv(processed_data_path)
    
    # Load model  
    model_path = "/project/def-mahadeva/ranaab/kinGEMs_v2/data/raw/iML1515_GEM.xml"
    model = load_model(model_path)
    
    print("=== KCAT DIAGNOSTIC TEST ===")
    print(f"Loaded model: {len(model.reactions)} reactions")
    print(f"Loaded data: {len(processed_df)} rows")
    
    # Ensure kcat column exists
    if 'kcat_mean' in processed_df.columns and 'kcat' not in processed_df.columns:
        processed_df['kcat'] = processed_df['kcat_mean']
    
    # Test 1: Baseline optimization
    print("\n--- Test 1: Baseline optimization ---")
    baseline_biomass, _, _, _ = run_optimization_with_dataframe(
        model=model,
        processed_df=processed_df,
        objective_reaction="BIOMASS_Ec_iML1515_core_75p37M",
        enzyme_upper_bound=0.30,
        enzyme_ratio=True,
        output_dir=None,
        save_results=False,
        verbose=False
    )
    print(f"Baseline biomass: {baseline_biomass:.6f}")
    
    # Test 2: Increase kcat for a major enzyme (b0929)
    print("\n--- Test 2: Increase kcat for major enzyme (b0929) ---")
    modified_df = processed_df.copy()
    
    # Find b0929 entries and increase their kcat by 10x
    b0929_mask = modified_df['Single_gene'] == 'b0929'
    original_kcats = modified_df.loc[b0929_mask, 'kcat'].values[:5]  # Save first 5 for comparison
    
    print(f"Found {b0929_mask.sum()} entries for gene b0929")
    print(f"Original kcats (first 5): {original_kcats}")
    
    # Increase by 10x
    modified_df.loc[b0929_mask, 'kcat'] *= 10.0
    modified_df.loc[b0929_mask, 'kcat_mean'] *= 10.0
    
    new_kcats = modified_df.loc[b0929_mask, 'kcat'].values[:5]
    print(f"Modified kcats (first 5): {new_kcats}")
    
    # Run optimization with modified kcats
    modified_biomass, _, _, _ = run_optimization_with_dataframe(
        model=model,
        processed_df=modified_df,
        objective_reaction="BIOMASS_Ec_iML1515_core_75p37M",
        enzyme_upper_bound=0.30,
        enzyme_ratio=True,
        output_dir=None,
        save_results=False,
        verbose=False
    )
    print(f"Modified biomass: {modified_biomass:.6f}")
    
    # Results
    print("\n--- RESULTS ---")
    print(f"Baseline biomass:  {baseline_biomass:.6f}")
    print(f"Modified biomass:  {modified_biomass:.6f}")
    print(f"Change:           {modified_biomass - baseline_biomass:.6f} ({(modified_biomass/baseline_biomass - 1)*100:+.1f}%)")
    
    if abs(modified_biomass - baseline_biomass) > 0.001:
        print("✅ SUCCESS: kcat changes affect optimization!")
    else:
        print("❌ PROBLEM: kcat changes have no effect!")
    
    # Test 3: Decrease kcat significantly
    print("\n--- Test 3: Decrease kcat for major enzyme (b0929) ---")
    decreased_df = processed_df.copy()
    decreased_df.loc[b0929_mask, 'kcat'] *= 0.1  # Decrease by 10x
    decreased_df.loc[b0929_mask, 'kcat_mean'] *= 0.1
    
    decreased_biomass, _, _, _ = run_optimization_with_dataframe(
        model=model,
        processed_df=decreased_df,
        objective_reaction="BIOMASS_Ec_iML1515_core_75p37M",
        enzyme_upper_bound=0.30,
        enzyme_ratio=True,
        output_dir=None,
        save_results=False,
        verbose=False
    )
    print(f"Decreased biomass: {decreased_biomass:.6f}")
    print(f"Change from baseline: {decreased_biomass - baseline_biomass:.6f} ({(decreased_biomass/baseline_biomass - 1)*100:+.1f}%)")

if __name__ == "__main__":
    test_kcat_effect()

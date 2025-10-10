#!/usr/bin/env python3
"""Quick test to verify constraint summary is suppressed"""
import sys
sys.path.insert(0, '/project/def-mahadeva/ranaab/kinGEMs_v2')

from kinGEMs.modeling.optimize import run_optimization_with_dataframe
import cobra
import pandas as pd

print("Loading model and data...")
model = cobra.io.read_sbml_model("data/raw/iML1515_GEM.xml")
processed_data = pd.read_csv("data/processed/iML1515_GEM/iML1515_GEM_processed_data.csv")

print("\n=== Test 1: verbose=False (should NOT show constraint summary) ===")
biomass, df_FBA, gene_seq, _ = run_optimization_with_dataframe(
    model=model,
    processed_df=processed_data,
    objective_reaction="BIOMASS_Ec_iML1515_core_75p37M",
    enzyme_upper_bound=0.35,
    verbose=False,
    save_results=False
)
print(f"Biomass: {biomass:.6f}")

print("\n=== Test 2: verbose=True (SHOULD show constraint summary) ===")
biomass2, df_FBA2, gene_seq2, _ = run_optimization_with_dataframe(
    model=model,
    processed_df=processed_data,
    objective_reaction="BIOMASS_Ec_iML1515_core_75p37M",
    enzyme_upper_bound=0.35,
    verbose=True,
    save_results=False
)
print(f"Biomass: {biomass2:.6f}")

print("\n✓ Test complete!")

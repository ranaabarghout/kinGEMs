#!/usr/bin/env python3
"""Test that biomasses list now tracks accepted values correctly"""
import sys
sys.path.insert(0, '/project/def-mahadeva/ranaab/kinGEMs_v2')

from kinGEMs.modeling.tuning import simulated_annealing
import cobra
import pandas as pd

print("Loading model and data...")
model = cobra.io.read_sbml_model("data/raw/iML1515_GEM.xml")
processed_data = pd.read_csv("data/processed/iML1515_GEM/iML1515_GEM_processed_data.csv")

gene_sequences_dict = {}
for _, row in processed_data.iterrows():
    if pd.notna(row['SEQ']) and row['Single_gene'] not in gene_sequences_dict:
        gene_sequences_dict[row['Single_gene']] = row['SEQ']

print("\nRunning 3 iterations with verbose output...\n")
kcat_dict, top_targets, df_new, iterations, biomasses, df_FBA = simulated_annealing(
    model=model,
    processed_data=processed_data,
    biomass_reaction="BIOMASS_Ec_iML1515_core_75p37M",
    objective_value=0.5,
    gene_sequences_dict=gene_sequences_dict,
    output_dir=None,
    enzyme_fraction=0.35,
    temperature=1.0,
    cooling_rate=0.95,
    max_iterations=3,
    max_unchanged_iterations=10,
    change_threshold=0.001,
    verbose=True
)

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"{'='*60}")
print(f"Iterations completed: {len(iterations)}")
print(f"\nBiomass progression (ACCEPTED values):")
for i, (iter_num, biomass) in enumerate(zip(iterations, biomasses)):
    change = "" if i == 0 else f" (Δ = {biomasses[i] - biomasses[i-1]:+.6f})"
    print(f"  Iteration {iter_num}: {biomass:.6f}{change}")
print(f"\nTotal improvement: {biomasses[-1] - biomasses[0]:.6f} ({(biomasses[-1]/biomasses[0]-1)*100:+.2f}%)")

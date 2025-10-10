#!/usr/bin/env python3
"""
Quick test to debug simulated annealing with verbose output
"""
import sys
import os
sys.path.insert(0, '/project/def-mahadeva/ranaab/kinGEMs_v2')

from kinGEMs.modeling.tuning import simulated_annealing
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
import cobra
import pandas as pd

# Load model and data
model_path = "/project/def-mahadeva/ranaab/kinGEMs_v2/data/raw/iML1515_GEM.xml"
processed_data_path = "/project/def-mahadeva/ranaab/kinGEMs_v2/data/processed/iML1515_GEM/iML1515_GEM_processed_data.csv"

print("Loading model...")
model = cobra.io.read_sbml_model(model_path)
print(f"Loaded model: {len(model.genes)} genes, {len(model.reactions)} reactions")

print("Loading processed data...")
processed_data = pd.read_csv(processed_data_path)
print(f"Loaded data: {len(processed_data)} rows")

# Get gene sequences
gene_sequences_dict = {}
for _, row in processed_data.iterrows():
    if pd.notna(row['SEQ']) and row['Single_gene'] not in gene_sequences_dict:
        gene_sequences_dict[row['Single_gene']] = row['SEQ']
print(f"Gene sequences: {len(gene_sequences_dict)} genes")

# Run ONE iteration of annealing with verbose=True
print("\n=== Running simulated annealing (verbose, 2 iterations) ===")
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
    min_temperature=0.01,
    max_iterations=2,  # Just 2 iterations
    max_unchanged_iterations=10,
    change_threshold=0.001,
    verbose=True  # VERBOSE!
)

print(f"\n=== Results ===")
print(f"Iterations: {iterations}")
print(f"Biomasses: {biomasses}")
print(f"Change: {biomasses[-1] - biomasses[0]}")

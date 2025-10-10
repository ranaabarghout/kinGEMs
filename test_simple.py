#!/usr/bin/env python3
"""
Simpler test - just check if iterations complete
"""
import sys
sys.path.insert(0, '/project/def-mahadeva/ranaab/kinGEMs_v2')

from kinGEMs.modeling.tuning import simulated_annealing
import cobra
import pandas as pd

# Load model and data
print("Loading...")
model = cobra.io.read_sbml_model("data/raw/iML1515_GEM.xml")
processed_data = pd.read_csv("data/processed/iML1515_GEM/iML1515_GEM_processed_data.csv")

gene_sequences_dict = {}
for _, row in processed_data.iterrows():
    if pd.notna(row['SEQ']) and row['Single_gene'] not in gene_sequences_dict:
        gene_sequences_dict[row['Single_gene']] = row['SEQ']

print("Running annealing...")
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
    max_iterations=5,
    max_unchanged_iterations=3,
    change_threshold=0.009,
    verbose=False
)

print(f"\nCompleted {len(iterations)} iterations")
print(f"Biomasses: {biomasses}")
print(f"Initial: {biomasses[0]:.6f}")
print(f"Final: {biomasses[-1]:.6f}")
print(f"Change: {biomasses[-1] - biomasses[0]:.6f}")
print(f"% Change: {(biomasses[-1] - biomasses[0])/biomasses[0]*100:.2f}%")

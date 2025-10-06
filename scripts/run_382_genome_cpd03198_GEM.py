
# kinGEMs Pipeline Script for 382_genome_cpd03198
from datetime import datetime
import logging
import os
import random
import sys
import warnings

import cobra
from cobra.io import write_sbml_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to Python path before kinGEMs imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.dataset import (
    annotate_model_with_kcat_and_gpr,
    assign_kcats_to_model,
    format_kcats_like_gpr,
    merge_substrate_sequences,
    process_kcat_predictions,
)
from kinGEMs.dataset_modelseed import prepare_modelseed_model_data
from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.modeling.tuning import simulated_annealing

# === Configuration ===
organism_strain_GEMname = "382_genome_cpd03198"
organism = "E coli"
run_id = f"{organism_strain_GEMname}_{datetime.today().strftime('%Y%m%d')}_{random.randint(1000, 9999)}"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, "data")
biolog_dir = os.path.join(data_dir, "Biolog experiments")
raw_data_dir = os.path.join(data_dir, "raw")
interim_data_dir = os.path.join(data_dir, "interim")
interim_data_dir_spec = os.path.join(interim_data_dir, f"{organism_strain_GEMname}")
processed_data_dir = os.path.join(data_dir, "processed")
processed_data_dir_spec = os.path.join(processed_data_dir, f"{organism_strain_GEMname}")
CPIPred_data_dir = os.path.join(interim_data_dir, "CPI-Pred predictions")
results_dir = os.path.join(project_root, "results")
tuning_results_dir = os.path.join(results_dir, "tuning_results", run_id)
os.makedirs(tuning_results_dir, exist_ok=True)

model_path = os.path.join(raw_data_dir, f"{organism_strain_GEMname}.xml")
predictions_csv_path = os.path.join(CPIPred_data_dir, f"X06A_kinGEMs_{organism_strain_GEMname}_predictions.csv")
metadata_path = os.path.join(biolog_dir, "rxnXgenes_382_genome.csv")
biolog_experiments_path = os.path.join(biolog_dir, "FBA_results.xlsx")

substrates_output = os.path.join(interim_data_dir_spec, f"{organism_strain_GEMname}_substrates.csv")
sequences_output = os.path.join(interim_data_dir_spec, f"{organism_strain_GEMname}_sequences.csv")
merged_data_output = os.path.join(interim_data_dir_spec, f"{organism_strain_GEMname}_merged_data.csv")
processed_data_output = os.path.join(processed_data_dir_spec, f"{organism_strain_GEMname}_processed_data.csv")

logging.getLogger('distributed').setLevel(logging.ERROR)
try:
    import gurobipy
    gurobipy.setParam('OutputFlag', 0)
except ImportError:
    pass

# Load your model
model = cobra.io.read_sbml_model(model_path)
print("Objective expression:", model.objective.expression)
print("Direction:", model.objective.direction)
obj_rxns = {rxn.id: rxn.objective_coefficient for rxn in model.reactions if rxn.objective_coefficient != 0}
print("Objective reaction(s) and coefficient(s):")
for rxn_id, coeff in obj_rxns.items():
    print(f"  {rxn_id} → {coeff}")
obj_rxn_id = next(rxn.id for rxn in model.reactions if rxn.objective_coefficient != 0)
print("Primary objective reaction:", obj_rxn_id)
biomass_reaction = obj_rxn_id
enzyme_upper_bound = 0.15

# === Step 1: Preparing model data ===
print("=== Step 1: Preparing model data ===")
irrev_model, substrate_df, sequences_df = prepare_modelseed_model_data(
    model_path=model_path,
    substrates_output=substrates_output,
    sequences_output=sequences_output,
    organism=organism,
    metadata_dir=biolog_dir
)

# === Step 2: Merging substrate and sequence data ===
print("=== Step 2: Merging substrate and sequence data ===")
merged_data = merge_substrate_sequences(
    substrate_df=substrate_df,
    sequences_df=sequences_df,
    model=irrev_model,
    output_path=merged_data_output
)

# === Debug: Inspect processed_data and df_FBA structure ===
print('merged_data columns:', list(merged_data.columns))
print('merged_data head:')
print(merged_data.head())

# === Step 3: Processing CPI-Pred kcat values ===
print("=== Step 3: Processing CPI-Pred kcat values & annotating model ===")
processed_data = process_kcat_predictions(
    merged_df=merged_data,
    predictions_csv_path=predictions_csv_path,
    output_path=processed_data_output
)

# === Fix: Ensure processed_data has a 'kcat' column for constraints ===
if 'kcat_mean' in processed_data.columns:
    processed_data['kcat'] = processed_data['kcat_mean']
    print("[INFO] Set processed_data['kcat'] = processed_data['kcat_mean']")
elif 'kcat_y' in processed_data.columns:
    processed_data['kcat'] = processed_data['kcat_y']
    print("[INFO] Set processed_data['kcat'] = processed_data['kcat_y']")
else:
    print("[WARNING] No kcat_mean or kcat_y column found in processed_data! 'kcat' will not be set.")
print('processed_data columns after fix:', list(processed_data.columns))
print('First 5 kcat values:', processed_data['kcat'].head())

irrev_model = annotate_model_with_kcat_and_gpr(
    model=irrev_model,
    df=processed_data
)

# === Step 4: Optimization (FBA Sanity Check + kinGEMs) ===
print("=== Step 4: Running optimization with kcat constraints ===")
(solution_value, df_FBA, gene_sequences_dict, _) = run_optimization_with_dataframe(
    model=irrev_model,
    processed_df=processed_data,
    objective_reaction=biomass_reaction,
    enzyme_upper_bound=enzyme_upper_bound,
    enzyme_ratio=True,
    maximization=True,
    multi_enzyme_off=False,
    isoenzymes_off=False,
    promiscuous_off=False,
    complexes_off=False,
    output_dir=None,
    save_results=False,
    print_reaction_conditions=True
)
print("Biomass value:", solution_value)

# === Debug: Inspect enzyme constraint mapping and usage ===
if 'kcat' in processed_data.columns:
    kcats = processed_data['kcat'].dropna().values
    print(f"[DEBUG] kcat stats: min={np.min(kcats):.2e}, max={np.max(kcats):.2e}, median={np.median(kcats):.2e}, mean={np.mean(kcats):.2e}, n={len(kcats)}")
else:
    print("[DEBUG] No 'kcat' column in processed_data!")
if 'Reactions' in processed_data.columns and 'Single_gene' in processed_data.columns and 'kcat' in processed_data.columns:
    print("[DEBUG] Example enzyme constraint mappings (reaction, gene, kcat):")
    print(processed_data[['Reactions','Single_gene','kcat']].head(5))
else:
    print("[DEBUG] processed_data missing expected columns for mapping debug.")
if df_FBA is not None and 'enzyme_usage' in df_FBA.columns:
    print("[DEBUG] Example enzyme usage (first 5):")
    print(df_FBA[['Reactions','Single_gene','enzyme_usage']].head(5))
else:
    print("[DEBUG] df_FBA missing or missing 'enzyme_usage' column.")

# === Step 5: Simulated Annealing ===
print("=== Step 5: Running simulated annealing ===")
temperature = 1.0
cooling_rate = 0.95
min_temperature = 0.01
max_iterations = 100
max_unchanged_iterations = 4
change_threshold = 0.009
biomass_goal = 1
kcat_dict, top_targets, df_new, iterations, biomasses, df_FBA = simulated_annealing(
    model=irrev_model,
    processed_data=processed_data,
    biomass_reaction=biomass_reaction,
    objective_value=biomass_goal,
    gene_sequences_dict=gene_sequences_dict,
    output_dir=tuning_results_dir,
    enzyme_fraction=enzyme_upper_bound,
    temperature=temperature,
    cooling_rate=cooling_rate,
    min_temperature=min_temperature,
    max_iterations=max_iterations,
    max_unchanged_iterations=max_unchanged_iterations,
    change_threshold=change_threshold
)
print(f"Final biomass: {biomasses[-1]:.4f}")
print(f"Improvement: {(biomasses[-1] - biomasses[0]) / biomasses[0] * 100:.1f}%")
print("Top 10 enzymes by mass contribution:")
print(top_targets[['Reactions','Single_gene','enzyme_mass']])

# Save df_new to the tuning_results_dir if not already saved
df_new_path = os.path.join(tuning_results_dir, "df_new.csv")
df_new.to_csv(df_new_path, index=False)
kcat_dict_path = os.path.join(tuning_results_dir, "kcat_dict.csv")
final_info_path = os.path.join(tuning_results_dir, "final_model_info.csv")
# Load the dataframes
df_new = pd.read_csv(df_new_path)
kcat_dict_df = pd.read_csv(kcat_dict_path)
if 'reaction_gene' not in kcat_dict_df.columns:
    kcat_dict_df.columns = ['reaction_gene', 'kcat_value']
df_new['reaction_gene'] = df_new['Reactions'].astype(str) + '_' + df_new['Single_gene'].astype(str)
df_new = df_new.merge(kcat_dict_df, on='reaction_gene', how='left')
df_new.rename(columns={'kcat_value': 'kcat_tuned'}, inplace=True)
df_new.to_csv(final_info_path, index=False)
print(f'Saved merged DataFrame with kcat_tuned column to {final_info_path}')

# === Step 6: Experimental Comparative Analysis ===
ec_df = pd.read_excel(biolog_experiments_path, sheet_name="Ecoli", engine="openpyxl")
exp_df = ec_df
blocked_cpds = [
    "cpd00224","cpd00122","cpd00609","cpd00108","cpd00794","cpd00138",
    "cpd00588","cpd00751","cpd00164","cpd00222","cpd00154","cpd00314",
    "cpd00105","cpd00396","cpd00082","cpd00027","cpd00179","cpd03198",
    "cpd00184","cpd00208","cpd00249","cpd01262","cpd00182","cpd00246",
    "cpd00054","cpd00020","cpd00280","cpd00832","cpd00232","cpd00276"
]
def simulate_enzyme_rate(base_model, processed_df, biomass_reaction, gene_sequences_dict, cpd_id, uptake_rate=10.0):
    from copy import deepcopy
    mdl = deepcopy(base_model)
    for cpd in blocked_cpds:
        if cpd.lower() == cpd_id.lower():
            continue
        ex_name = f"EX_{cpd}_e0"
        if ex_name in mdl.reactions:
            mdl.reactions.get_by_id(ex_name).lower_bound = 0.0
    target_ex = f"EX_{cpd_id}_e0"
    if target_ex not in mdl.reactions:
        raise KeyError(f"Exchange {target_ex} not found")
    mdl.reactions.get_by_id(target_ex).lower_bound = -abs(uptake_rate)
    sol_val, df_FBA, _, _ = run_optimization_with_dataframe(
        model=mdl,
        processed_df=processed_df,
        objective_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        enzyme_ratio=True,
        maximization=True,
        multi_enzyme_off=False,
        isoenzymes_off=False,
        promiscuous_off=False,
        complexes_off=False,
        output_dir=None,
        save_results=False,
        print_reaction_conditions=False
    )
    return sol_val
glc = "cpd00027"
glc_rate = simulate_enzyme_rate(
    base_model=irrev_model,
    processed_df=processed_data,
    biomass_reaction=biomass_reaction,
    gene_sequences_dict=gene_sequences_dict,
    cpd_id=glc,
    uptake_rate=100.0
)
print(f"Enzyme-constrained glucose growth: {glc_rate:.4f}")
results = []
for row in exp_df.itertuples():
    cpd = row.cpd
    print(f"=== Testing substrate: {cpd} ===")
    try:
        rate = simulate_enzyme_rate(
            base_model=irrev_model,
            processed_df=processed_data,
            biomass_reaction=biomass_reaction,
            gene_sequences_dict=gene_sequences_dict,
            cpd_id=cpd,
            uptake_rate=100.0
        )
        print(f"Predicted enzyme-constrained rate: {rate:.4f}")
    except Exception as e:
        rate = None
        print(f"⚠️ Warning for {cpd}: {e}")
    norm = rate/glc_rate if rate is not None and glc_rate > 0 else None
    if norm is not None:
        print(f"Normalized rate (relative to glucose): {norm:.4f}")
    else:
        print("Normalized rate: N/A")
    print(f"Experimental value: {row.exp_value:.4f}")
    results.append({
        'cpd': cpd,
        'ec_rate': rate,
        'norm_rate': norm,
        'exp_value': row.exp_value
    })
enz_df = pd.DataFrame(results)
comp_df = exp_df.merge(enz_df, on='cpd')
print("=== Summary Comparison ===")
print(comp_df)
plt.figure(figsize=(6,4))
plt.scatter(comp_df['exp_value'], comp_df['norm_rate'], s=50)
plt.xlabel('Experimental value (normalized by glucose)')
plt.ylabel('Model normalized enzyme-constrained rate')
plt.title('Enzyme-constrained FBA vs. experimental')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 7: Save Final Model ===
model_output_dir = os.path.join(project_root, "models")
os.makedirs(model_output_dir, exist_ok=True)
model_output_path = os.path.join(model_output_dir, f"{run_id}.xml")
model_with_kcats = assign_kcats_to_model(irrev_model, df_new)
format_kcats_like_gpr(model_with_kcats.reactions.get_by_id("PGI"))
write_sbml_model(model_with_kcats, model_output_path)
print(f"Final GEM saved to: {model_output_path}")

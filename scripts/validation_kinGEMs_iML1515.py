# Validation of kinGEMs iML1515 Models
# This script uses the validation utility functions to analyze kinGEMs-produced 
# iML1515 models and compare them with experimental data.

import logging
import os
import sys
import warnings

import cobra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve as pre_rec

# Add parent directory to Python path before kinGEMs imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.config import ECOLI_VALIDATION_DIR, MODELS_DIR
from kinGEMs.validation_utils import (
    check_environment,
    load_data,
    load_environment,
    match_model_data,
    model_adjustments,
    prepare_model,
    simulate_phenotype,
    simulate_phenotype_flux,
    test_growth,
)

# Silence warnings
warnings.filterwarnings('ignore')
logging.getLogger('distributed').setLevel(logging.ERROR)

# === Configuration ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results')
validation_dir = os.path.join(results_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

# Model and data paths
model_path = os.path.join(MODELS_DIR, 'ecoli_iML1515_20250826_4941.xml')
objective_reaction = 'BIOMASS_Ec_iML1515_core_75p37M'
enzyme_upper_bound = 0.15

# Try to find the most recent tuning results directory for iML1515
# Check multiple possible locations
possible_tuning_bases = [
    os.path.join(results_dir, 'tuning_results'),  # Main results folder
    os.path.join(project_root, 'notebooks', 'results', 'tuning_results'),  # Notebooks results folder
]

processed_df_path = None
for tuning_results_base in possible_tuning_bases:
    if os.path.exists(tuning_results_base):
        print(f"Checking for tuning results in: {tuning_results_base}")
        # Look for any iML1515 tuning results directories
        try:
            all_dirs = os.listdir(tuning_results_base)
        except PermissionError:
            print(f"  Permission denied accessing {tuning_results_base}")
            continue
            
        iml1515_dirs = [d for d in all_dirs 
                        if os.path.isdir(os.path.join(tuning_results_base, d)) 
                        and ('iML1515' in d or 'ecoli' in d.lower())]
        if iml1515_dirs:
            # Use the most recent one (alphabetically last)
            latest_dir = sorted(iml1515_dirs)[-1]
            dir_path = os.path.join(tuning_results_base, latest_dir)
            
            # Check for df_new.csv
            potential_df_path = os.path.join(dir_path, 'df_new.csv')
            if os.path.exists(potential_df_path) and os.path.getsize(potential_df_path) > 0:
                processed_df_path = potential_df_path
                print(f"✓ Found tuning results at: {processed_df_path}")
                break
            else:
                # List what's actually in the directory
                try:
                    dir_contents = os.listdir(dir_path)
                    if dir_contents:
                        print(f"  Found directory {latest_dir} with files: {dir_contents[:5]}")
                        # Check if there's a processed_data.csv or similar
                        for fname in dir_contents:
                            if 'processed' in fname.lower() and fname.endswith('.csv'):
                                potential_df_path = os.path.join(dir_path, fname)
                                processed_df_path = potential_df_path
                                print(f"  Using alternative file: {fname}")
                                break
                    else:
                        print(f"  Found directory {latest_dir} but it's empty")
                except Exception:
                    print(f"  Found directory {latest_dir} but couldn't read contents")
            
            if processed_df_path:
                break
        else:
            print(f"  No iML1515 tuning results found in {tuning_results_base}")
    else:
        print(f"  Directory does not exist: {tuning_results_base}")

if processed_df_path is None:
    print("\nWARNING: No enzyme-constrained tuning data found.")
    print("The script will run baseline validation only (without ecGEM comparison).")
    print("To generate tuning data, run the full kinGEMs pipeline first.")
    run_ecgem = False
else:
    print(f"\nUsing tuning data from: {processed_df_path}")
    run_ecgem = True

# Threshold parameters
sim_thresh = 0.001
fit_thresh = -2

# === Step 1: Load Model and Experimental Data ===
print("=== Step 1: Loading model and experimental data ===")
model = cobra.io.read_sbml_model(model_path)
print(f"Genes in saved model: {len(model.genes)}")

# Load experimental validation data (Keio fitness)
data_dir = ECOLI_VALIDATION_DIR
keio_fitness_path = os.path.join(data_dir, 'fit_organism_Keio.tsv')
keio_fitness_data = pd.read_table(keio_fitness_path)

# Diagnostic: Print gene and reaction counts
print("\n[Diagnostic] After loading model:")
print(f"  Genes in model: {len(model.genes)}")
print(f"  Reactions in model: {len(model.reactions)}")
model_gene_ids = set([g.id for g in model.genes])
model_rxn_ids = set([r.id for r in model.reactions])
print(f"  First 10 gene IDs: {list(model_gene_ids)[:10]}")
print(f"  First 10 reaction IDs: {list(model_rxn_ids)[:10]}")

# Diagnostic: Print kcat coverage
rxn_with_kcat = []
rxn_without_kcat = []
for rxn in model.reactions:
    ann = rxn.annotation if hasattr(rxn, 'annotation') else {}
    if 'kcat' in ann and ann['kcat'] not in [None, '', 0, '0']:
        rxn_with_kcat.append(rxn.id)
    else:
        rxn_without_kcat.append(rxn.id)
print(f"  Reactions with kcat: {len(rxn_with_kcat)}")
print(f"  Reactions without kcat: {len(rxn_without_kcat)}")

# === Step 2: Apply Validation Functions ===
print("\n=== Step 2: Applying validation functions ===")

# 1. Prepare model (set exchange bounds)
model = prepare_model(model)

# 2. Load environment (medium and carbon sources)
name_medium_model, name_carbon_model, name_carbon_experiment = load_environment(ECOLI_VALIDATION_DIR)

# 3. Load experimental data
data_experiments, data_genes, data_fitness = load_data(ECOLI_VALIDATION_DIR)

# 4. Match model and data
name_genes_matched, name_carbon_experiment_matched, name_carbon_model_matched, data_fitness_matched = match_model_data(
    model=model,
    name_carbon_model=name_carbon_model,
    name_carbon_experiment=name_carbon_experiment,
    data_experiments=data_experiments,
    data_genes=data_genes,
    data_fitness=data_fitness
)

# 5. Adjust model and matched data
model_adj, name_genes_matched_adj, name_carbon_experiment_matched_adj, name_carbon_model_matched_adj, data_fitness_matched_adj = model_adjustments(
    adj_strain=True,
    adj_essential=True,
    adj_carbon=True,
    model=model,
    name_genes_matched=name_genes_matched,
    name_carbon_experiment_matched=name_carbon_experiment_matched,
    name_carbon_model_matched=name_carbon_model_matched,
    data_fitness_matched=data_fitness_matched
)

# 6. Check environment setup
medium_ex_inds, carbon_ex_inds = check_environment(
    model_adj=model_adj,
    name_medium_model=name_medium_model,
    name_carbon_model_matched_adj=name_carbon_model_matched_adj
)

# === Step 3: Test Growth Predictions ===
print("\n=== Step 3: Testing growth predictions ===")

# Test growth
growth_results = test_growth(
    model_adj=model_adj,
    name_carbon_model_matched_adj=name_carbon_model_matched_adj,
    medium_ex_inds=medium_ex_inds,
    carbon_ex_inds=carbon_ex_inds
)

# Simulate phenotypes
if run_ecgem:
    # Load processed dataframe for enzyme constraints
    processed_df = pd.read_csv(processed_df_path)
    
    # Simulate phenotypes for both baseline and enzyme-constrained models
    data_fitness_simulated_baseline, data_fitness_simulated_ecGEM = simulate_phenotype(
        model_run=model_adj,
        name_genes_matched_adj=name_genes_matched_adj,
        name_carbon_model_matched_adj=name_carbon_model_matched_adj,
        medium_ex_inds=medium_ex_inds,
        carbon_ex_inds=carbon_ex_inds,
        processed_df=processed_df,
        objective_reaction=objective_reaction,
        enzyme_upper_bound=enzyme_upper_bound
    )
else:
    # Only run baseline simulation without enzyme constraints
    print("\nRunning baseline-only simulation (no enzyme constraints)...")
    from copy import deepcopy
    
    # Initialize the results matrix
    n_genes = len(name_genes_matched_adj)
    n_carbons = len(name_carbon_model_matched_adj)
    data_fitness_simulated_baseline = np.zeros((n_genes, n_carbons))
    
    # Simulate growth for each gene knockout and carbon source
    for i, gene in enumerate(name_genes_matched_adj):
        for j, carbon in enumerate(name_carbon_model_matched_adj):
            try:
                # Create a copy of the model
                test_model = deepcopy(model_adj)
                
                # Set medium (close all carbon sources first)
                for idx in carbon_ex_inds:
                    test_model.reactions[idx].lower_bound = 0
                # Open the specific carbon source
                test_model.reactions[carbon_ex_inds[j]].lower_bound = -10
                
                # Knockout the gene
                gene_obj = test_model.genes.get_by_id(gene)
                gene_obj.knock_out()
                
                # Run FBA
                solution = test_model.optimize()
                if solution.status == 'optimal':
                    data_fitness_simulated_baseline[i, j] = solution.objective_value
                else:
                    data_fitness_simulated_baseline[i, j] = 0.0
            except Exception as e:
                data_fitness_simulated_baseline[i, j] = 0.0
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_genes} genes...")
    
    data_fitness_simulated_ecGEM = None
    print(f"Baseline simulation complete ({n_genes} genes × {n_carbons} carbon sources).")
    print("Enzyme-constrained comparison will be skipped.")

# === Step 4: Analyze Model Performance ===
print("\n=== Step 4: Analyzing model performance ===")

# Create binary growth matrices
exp_binary = (data_fitness_matched_adj > 0).astype(float)
sim_binary_baseline = (data_fitness_simulated_baseline > sim_thresh).astype(float)

def print_metrics(exp_binary, sim_binary, label):
    """Calculate and print performance metrics."""
    true_pos = np.sum((exp_binary == 1) & (sim_binary == 1))
    true_neg = np.sum((exp_binary == 0) & (sim_binary == 0))
    false_pos = np.sum((exp_binary == 0) & (sim_binary == 1))
    false_neg = np.sum((exp_binary == 1) & (sim_binary == 0))
    accuracy = (true_pos + true_neg) / np.prod(exp_binary.shape)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f'{label} Model Performance Metrics:')
    print(f'  Accuracy: {accuracy:.3f}')
    print(f'  Precision: {precision:.3f}')
    print(f'  Recall: {recall:.3f}')
    print(f'  F1 Score: {f1_score:.3f}')
    return accuracy, precision, recall, f1_score

# Print metrics for baseline (and ecGEM if available)
print("\n--- Basic Metrics ---")
acc_base, prec_base, rec_base, f1_base = print_metrics(exp_binary, sim_binary_baseline, 'Baseline GEM')

if run_ecgem:
    sim_binary_ecGEM = (data_fitness_simulated_ecGEM > sim_thresh).astype(float)
    acc_ec, prec_ec, rec_ec, f1_ec = print_metrics(exp_binary, sim_binary_ecGEM, 'ecGEM')
    
    # Plot comparison heatmaps for both models
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    im0 = axes[0].imshow(exp_binary, cmap='YlOrRd')
    axes[0].set_title('Experimental Growth')
    axes[0].set_xlabel('Carbon Sources')
    axes[0].set_ylabel('Genes')
    im1 = axes[1].imshow(sim_binary_baseline, cmap='YlOrRd')
    axes[1].set_title('Simulated Growth (Baseline GEM)')
    axes[1].set_xlabel('Carbon Sources')
    im2 = axes[2].imshow(sim_binary_ecGEM, cmap='YlOrRd')
    axes[2].set_title('Simulated Growth (ecGEM)')
    axes[2].set_xlabel('Carbon Sources')
    plt.colorbar(im0, ax=axes[0], label='Growth')
    plt.colorbar(im1, ax=axes[1], label='Growth')
    plt.colorbar(im2, ax=axes[2], label='Growth')
else:
    # Plot comparison heatmaps for baseline only
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    im0 = axes[0].imshow(exp_binary, cmap='YlOrRd')
    axes[0].set_title('Experimental Growth')
    axes[0].set_xlabel('Carbon Sources')
    axes[0].set_ylabel('Genes')
    im1 = axes[1].imshow(sim_binary_baseline, cmap='YlOrRd')
    axes[1].set_title('Simulated Growth (Baseline GEM)')
    axes[1].set_xlabel('Carbon Sources')
    plt.colorbar(im0, ax=axes[0], label='Growth')
    plt.colorbar(im1, ax=axes[1], label='Growth')

plt.tight_layout()
plt.savefig(os.path.join(validation_dir, 'growth_heatmaps.png'), dpi=300)
print(f"Saved growth heatmaps to {os.path.join(validation_dir, 'growth_heatmaps.png')}")
plt.close()

# === Step 5: Advanced Metrics Calculation ===
print("\n=== Step 5: Calculating advanced metrics ===")

def calc_metrics(data_sim, data_fit, fit_thresh, sim_thresh):
    """Calculate advanced metrics including AUC, BAC, and ROC AUC."""
    data_sim_b = (data_sim > sim_thresh).astype(int)
    data_fit_b = (data_fit > fit_thresh).astype(int)
    num_genes = np.array([data_fit_b.shape[0]])
    num_genes_add = num_genes - num_genes[0]
    AC_i = np.zeros(len(num_genes_add))
    AUC_i = np.zeros(len(num_genes_add))
    BAC_i = np.zeros(len(num_genes_add))
    ROC_AUC_i = np.zeros(len(num_genes_add))
    for i in range(len(num_genes_add)):
        data_fit_b_V = data_fit_b.flatten()
        data_sim_b_V = data_sim_b.flatten()
        data_fit_V = data_fit.flatten()
        data_fit_b_V = np.append(data_fit_b_V, np.ones(num_genes_add[i]*data_fit_b.shape[1]))
        data_sim_b_V = np.append(data_sim_b_V, np.ones(num_genes_add[i]*data_fit_b.shape[1]))
        data_fit_V = np.append(data_fit_V, np.zeros(num_genes_add[i]*data_fit_b.shape[1]))
        tn, fp, fn, tp = confusion_matrix(data_fit_b_V, data_sim_b_V, labels=[1,0]).ravel()
        AC_i[i] = (tp+tn)/(tn+fn+tp+fp)
        pre, rec, thresholds = pre_rec(data_sim_b_V, data_fit_V*-1, pos_label=0)
        AUC_i[i] = sk_auc(rec, pre)
        TPR = tp/(tp+fn) if (tp+fn) > 0 else 0
        TNR = tn/(tn+fp) if (tn+fp) > 0 else 0
        BAC_i[i] = (TPR+TNR)/2
        ROC_AUC_i[i] = roc_auc_score(data_sim_b_V, data_fit_V)
    return AC_i, AUC_i, BAC_i, ROC_AUC_i

# Calculate metrics for baseline
print("\n--- Advanced Metrics ---")
AC_baseline, AUC_baseline, BAC_baseline, ROC_AUC_baseline = calc_metrics(
    data_fitness_simulated_baseline, data_fitness_matched_adj, fit_thresh, sim_thresh
)
print('Baseline GEM:')
print(f'  Accuracy: {AC_baseline[0]:.3f}')
print(f'  AUC: {AUC_baseline[0]:.3f}')
print(f'  Balanced Accuracy: {BAC_baseline[0]:.3f}')
print(f'  ROC AUC: {ROC_AUC_baseline[0]:.3f}')

if run_ecgem:
    AC_ecGEM, AUC_ecGEM, BAC_ecGEM, ROC_AUC_ecGEM = calc_metrics(
        data_fitness_simulated_ecGEM, data_fitness_matched_adj, fit_thresh, sim_thresh
    )
    print('ecGEM:')
    print(f'  Accuracy: {AC_ecGEM[0]:.3f}')
    print(f'  AUC: {AUC_ecGEM[0]:.3f}')
    print(f'  Balanced Accuracy: {BAC_ecGEM[0]:.3f}')
    print(f'  ROC AUC: {ROC_AUC_ecGEM[0]:.3f}')
else:
    AC_ecGEM = AUC_ecGEM = BAC_ecGEM = ROC_AUC_ecGEM = None
    print('ecGEM: Skipped (no tuning data available)')

# === Step 6: Essential Genes Analysis ===
print("\n=== Step 6: Analyzing essential genes ===")

essential_gene_mask = np.min(data_fitness_matched_adj, axis=1) <= 0.001
fit_ess = data_fitness_matched_adj[essential_gene_mask, :]
sim_baseline_ess = data_fitness_simulated_baseline[essential_gene_mask, :]
sim_ecGEM_ess = data_fitness_simulated_ecGEM[essential_gene_mask, :]

AC_baseline_ess, AUC_baseline_ess, BAC_baseline_ess, ROC_AUC_baseline_ess = calc_metrics(
    sim_baseline_ess, fit_ess, fit_thresh, sim_thresh
)
print('Baseline GEM (Essential Genes):')
print(f'  Accuracy: {AC_baseline_ess[0]:.3f}')
print(f'  AUC: {AUC_baseline_ess[0]:.3f}')
print(f'  Balanced Accuracy: {BAC_baseline_ess[0]:.3f}')
print(f'  ROC AUC: {ROC_AUC_baseline_ess[0]:.3f}')

AC_ecGEM_ess, AUC_ecGEM_ess, BAC_ecGEM_ess, ROC_AUC_ecGEM_ess = calc_metrics(
    sim_ecGEM_ess, fit_ess, fit_thresh, sim_thresh
)
print('ecGEM (Essential Genes):')
print(f'  Accuracy: {AC_ecGEM_ess[0]:.3f}')
print(f'  AUC: {AUC_ecGEM_ess[0]:.3f}')
print(f'  Balanced Accuracy: {BAC_ecGEM_ess[0]:.3f}')
print(f'  ROC AUC: {ROC_AUC_ecGEM_ess[0]:.3f}')

# === Step 7: Genes with kcat Analysis ===
print("\n=== Step 7: Analyzing genes with kcat data ===")

genes_with_kcat = set()
for rxn in model.reactions:
    ann = rxn.annotation if hasattr(rxn, 'annotation') else {}
    if 'kcat' in ann and ann['kcat'] not in [None, '', 0, '0']:
        for gene in rxn.genes:
            genes_with_kcat.add(gene.id)
print(f"Number of genes with kcat data: {len(genes_with_kcat)}")
print(f"Proportion of genes with kcat data: {len(genes_with_kcat)/len(model.genes):.3f}")

kcat_gene_mask = np.array([g in genes_with_kcat for g in name_genes_matched_adj])
fit_kcat = data_fitness_matched_adj[kcat_gene_mask, :]
sim_baseline_kcat = data_fitness_simulated_baseline[kcat_gene_mask, :]
sim_ecGEM_kcat = data_fitness_simulated_ecGEM[kcat_gene_mask, :]

AC_baseline_kcat, AUC_baseline_kcat, BAC_baseline_kcat, ROC_AUC_baseline_kcat = calc_metrics(
    sim_baseline_kcat, fit_kcat, fit_thresh, sim_thresh
)
print('Baseline GEM (Genes with kcat):')
print(f'  Accuracy: {AC_baseline_kcat[0]:.3f}')
print(f'  AUC: {AUC_baseline_kcat[0]:.3f}')
print(f'  Balanced Accuracy: {BAC_baseline_kcat[0]:.3f}')
print(f'  ROC AUC: {ROC_AUC_baseline_kcat[0]:.3f}')

AC_ecGEM_kcat, AUC_ecGEM_kcat, BAC_ecGEM_kcat, ROC_AUC_ecGEM_kcat = calc_metrics(
    sim_ecGEM_kcat, fit_kcat, fit_thresh, sim_thresh
)
print('ecGEM (Genes with kcat):')
print(f'  Accuracy: {AC_ecGEM_kcat[0]:.3f}')
print(f'  AUC: {AUC_ecGEM_kcat[0]:.3f}')
print(f'  Balanced Accuracy: {BAC_ecGEM_kcat[0]:.3f}')
print(f'  ROC AUC: {ROC_AUC_ecGEM_kcat[0]:.3f}')

# === Step 8: Distribution Analysis ===
print("\n=== Step 8: Analyzing prediction distributions ===")

# Plot prediction distributions
plt.figure(figsize=(12, 6))
sns.histplot(data_fitness_simulated_baseline.flatten(), bins=50, color='blue', 
             label='Baseline GEM', kde=True, stat='density', alpha=0.6)
sns.histplot(data_fitness_simulated_ecGEM.flatten(), bins=50, color='orange', 
             label='ecGEM', kde=True, stat='density', alpha=0.6)
plt.xlabel('Predicted Growth Value')
plt.ylabel('Density')
plt.title('Distribution of Predicted Growth Values: Baseline GEM vs ecGEM')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(validation_dir, 'prediction_distributions.png'), dpi=300)
print(f"Saved prediction distributions to {os.path.join(validation_dir, 'prediction_distributions.png')}")
plt.close()

print('\nSummary statistics for Baseline GEM predictions:')
print(pd.Series(data_fitness_simulated_baseline.flatten()).describe())
print('\nSummary statistics for ecGEM predictions:')
print(pd.Series(data_fitness_simulated_ecGEM.flatten()).describe())

# Plot experimental vs predicted distributions
plt.figure(figsize=(12, 6))
sns.histplot(data_fitness_matched_adj.flatten(), bins=50, color='green', 
             label='Experimental Fitness', kde=True, stat='density', alpha=0.6)
sns.histplot(data_fitness_simulated_baseline.flatten(), bins=50, color='blue', 
             label='Baseline GEM', kde=True, stat='density', alpha=0.4)
sns.histplot(data_fitness_simulated_ecGEM.flatten(), bins=50, color='orange', 
             label='ecGEM', kde=True, stat='density', alpha=0.4)
plt.xlabel('Growth Value')
plt.ylabel('Density')
plt.title('Distribution of Experimental and Predicted Growth Values')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(validation_dir, 'exp_vs_pred_distributions.png'), dpi=300)
print(f"Saved experimental vs predicted distributions to {os.path.join(validation_dir, 'exp_vs_pred_distributions.png')}")
plt.close()

print('\nSummary statistics for Experimental Fitness values:')
print(pd.Series(data_fitness_matched_adj.flatten()).describe())

# === Step 9: Save Results Summary ===
print("\n=== Step 9: Saving results summary ===")

# Create summary DataFrame
summary_data = {
    'Model': ['Baseline GEM', 'ecGEM'],
    'Accuracy': [AC_baseline[0], AC_ecGEM[0]],
    'AUC': [AUC_baseline[0], AUC_ecGEM[0]],
    'Balanced_Accuracy': [BAC_baseline[0], BAC_ecGEM[0]],
    'ROC_AUC': [ROC_AUC_baseline[0], ROC_AUC_ecGEM[0]],
    'Accuracy_Essential': [AC_baseline_ess[0], AC_ecGEM_ess[0]],
    'AUC_Essential': [AUC_baseline_ess[0], AUC_ecGEM_ess[0]],
    'BAC_Essential': [BAC_baseline_ess[0], BAC_ecGEM_ess[0]],
    'ROC_AUC_Essential': [ROC_AUC_baseline_ess[0], ROC_AUC_ecGEM_ess[0]],
    'Accuracy_kcat': [AC_baseline_kcat[0], AC_ecGEM_kcat[0]],
    'AUC_kcat': [AUC_baseline_kcat[0], AUC_ecGEM_kcat[0]],
    'BAC_kcat': [BAC_baseline_kcat[0], BAC_ecGEM_kcat[0]],
    'ROC_AUC_kcat': [ROC_AUC_baseline_kcat[0], ROC_AUC_ecGEM_kcat[0]]
}
summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(validation_dir, 'validation_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"Saved validation summary to {summary_path}")

# Save simulation results
np.save(os.path.join(validation_dir, 'data_fitness_simulated_baseline.npy'), data_fitness_simulated_baseline)
np.save(os.path.join(validation_dir, 'data_fitness_simulated_ecGEM.npy'), data_fitness_simulated_ecGEM)
np.save(os.path.join(validation_dir, 'data_fitness_matched_adj.npy'), data_fitness_matched_adj)
print(f"Saved simulation arrays to {validation_dir}")

print("\n=== Validation complete! ===")
print(f"All results saved to: {validation_dir}")

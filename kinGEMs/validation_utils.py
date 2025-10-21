# import libraries
import copy
import os

import cobra
import fastcluster
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, ttest_ind
from seaborn import clustermap
import shap
from sklearn.decomposition import PCA
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.metrics import confusion_matrix as confusion_matrix
from sklearn.metrics import precision_recall_curve as pre_rec
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

from kinGEMs.config import ECOLI_VALIDATION_DIR, MODELS_DIR
from kinGEMs.modeling.optimize import run_optimization_with_dataframe


def prepare_model(model):
# INPUT
# - model: cobrapy model object
# OUTPUT
# - model: cobrapy model object with exchange bounds set
    # turn off exchange reactions by setting lower and upper bounds to 0 and 1000 respectively
    for ex in model.exchanges:
        ex.lower_bound = 0
        ex.upper_bound = 1000
    # We leave the maintenance on here
    # turn off maintenance reactions by setting the lower bound to 0 (turning off maintenance can make results easier to interpret)
    # model.reactions.get_by_id("ATPM").lower_bound = 0
    return model


def load_environment(base_directory):
# INPUT
# - base_directory: the directory containing the data files/directories
# OUTPUT
# - name_medium_model: the model's names of the media components
# - name_carbon_model: the model's name of the carbon sources
# - name_carbon_experiment: the name of the experiment corresponding to each carbon source

    # Loading the carbon sources
    fields = ['expName','media','BiGG Component']
    index_list = list(range(0,29))
    index_list.extend(list(range(31,51)))
    index_list.extend(list(range(53,59)))
    env_path = os.path.join(ECOLI_VALIDATION_DIR, 'exp_organism_Keio_Mapped.txt')
    df_env = pd.read_table(env_path, encoding='latin-1', usecols=fields, skiprows=lambda x: x not in index_list)
    name_carbon_model = df_env['BiGG Component'].tolist()
    name_carbon_experiment = df_env['expName'].tolist()

    # Loading the base medium
    media_path = os.path.join(ECOLI_VALIDATION_DIR, 'exp_organism_Keio_Mapped_Media.txt')
    df_med = pd.read_table(media_path, encoding='latin-1')
    med = df_med.loc[1, 'M9 minimal media_noCarbon']
    name_medium_model = med.split('; ')
    return (name_medium_model, name_carbon_model, name_carbon_experiment)


def load_data(base_directory):
# INPUT
# - base_directory: the directory containing the data files/directories
# OUTPUT
# - data_experiments: list of experiment IDs from fitness data
# - data_genes: list of gene IDs from fitness data
# - data_fitness: 2D array of fitness data [genes x experiments]

    # Loading the fitness data
    fit_path = os.path.join(ECOLI_VALIDATION_DIR, 'fit_organism_Keio.tsv')
    df_fit = pd.read_table(fit_path)

    data_genes = df_fit.loc[:,'sysName'].to_list()

    data_envs = df_fit.columns
    data_envs = data_envs[5:]
    data_experiments = []
    for e in range(len(data_envs)):
        tmp = data_envs[e].split(' ')
        data_experiments.append(tmp[0])

    data_fitness = df_fit.iloc[:,5:].to_numpy()

    return(data_experiments,data_genes,data_fitness)

def match_model_data(model, name_carbon_model, name_carbon_experiment, data_experiments, data_genes, data_fitness):
    """
    Match model genes and experiments with data IDs, average fitness data for replicate carbon sources.
    Returns matched gene names, experiment IDs, carbon source names, and fitness data.
    """
    name_genes_matched = []
    data_genes_matched_inds = []
    print("Matching model genes to data genes...")
    for f in range(len(data_genes)):
        for g in range(len(model.genes)):
            if data_genes[f] == model.genes[g].id:
                name_genes_matched.append(model.genes[g].id)
                data_genes_matched_inds.append(f)
    print(f"Matched {len(name_genes_matched)} genes.")

    name_carbon_experiment_matched = []
    name_carbon_model_matched = []
    data_carbon_matched_inds = []
    print("Matching experimental carbon sources to model carbon sources...")
    for e in range(len(data_experiments)):
        for e2 in range(len(name_carbon_experiment)):
            if data_experiments[e] == name_carbon_experiment[e2]:
                name_carbon_experiment_matched.append(name_carbon_experiment[e2])
                name_carbon_model_matched.append(name_carbon_model[e2])
                data_carbon_matched_inds.append(e)
    print(f"Matched {len(name_carbon_model_matched)} carbon sources.")

    data_fitness_matched = data_fitness[data_genes_matched_inds,:][:,data_carbon_matched_inds]
    print(f"Shape of matched fitness data: {data_fitness_matched.shape}")

    # Average Data for Same Carbon Source
    print("Averaging fitness data for replicate carbon sources...")
    name_carbon_model_matched_unique, map1, map2 = np.unique(name_carbon_model_matched, return_index=True, return_inverse=True)
    name_carbon_experiment_matched_unique = np.array(name_carbon_experiment_matched)[map1]
    data_fitness_matched_unique = np.zeros([data_fitness_matched.shape[0], len(name_carbon_model_matched_unique)])
    for i in range(len(name_carbon_model_matched_unique)):
        data_fitness_matched_unique[:, i] = np.transpose(np.mean(data_fitness_matched[:, np.argwhere(map2 == i)], 1))
    print(f"Final shape of averaged fitness data: {data_fitness_matched_unique.shape}")
    print(f"Unique carbon sources after averaging: {len(name_carbon_model_matched_unique)}")

    return name_genes_matched, name_carbon_experiment_matched_unique, name_carbon_model_matched_unique, data_fitness_matched_unique

def model_adjustments(adj_strain, adj_essential, adj_carbon, model, name_genes_matched, name_carbon_experiment_matched, name_carbon_model_matched, data_fitness_matched):
    """
    Adjust model and matched data for strain variation, essential genes, and carbon sources.
    Returns adjusted model and matched lists/arrays.
    """
    model_adj = copy.deepcopy(model)
    name_genes_matched_adj = name_genes_matched
    data_fitness_matched_adj = data_fitness_matched
    name_carbon_experiment_matched_adj = name_carbon_experiment_matched
    name_carbon_model_matched_adj = name_carbon_model_matched

    if adj_strain:
        strain_gene_remove_id = ['b0062','b0063','b3903','b3904','b0061','b0344','b3902']
        for gid in strain_gene_remove_id:
            if gid in model_adj.genes:
                model_adj.genes.get_by_id(gid).knock_out()

        print("Removing strain genes:")
        tmp_inds = [i for i, g in enumerate(name_genes_matched_adj) if g in strain_gene_remove_id]

        name_genes_matched_adj = np.delete(name_genes_matched_adj, tmp_inds, 0)
        print(name_genes_matched_adj[tmp_inds])
        data_fitness_matched_adj = np.delete(data_fitness_matched_adj, tmp_inds, 0)

    if adj_essential:
        thresh = 0.001
        for ex in model_adj.exchanges:
            ex.lower_bound = -1000
            ex.upper_bound = 1000
        tmp_inds = []
        print("Removing essential genes: ")
        for g in range(len(name_genes_matched_adj)):
            with model_adj:
                model_adj.genes.get_by_id(name_genes_matched_adj[g]).knock_out()
                solution = model_adj.slim_optimize()
                if solution < thresh:
                    tmp_inds.append(g)
                    print(name_genes_matched_adj[g])
        name_genes_matched_adj = np.delete(name_genes_matched_adj, tmp_inds, 0)
        data_fitness_matched_adj = np.delete(data_fitness_matched_adj, tmp_inds, 0)
        for ex in model_adj.exchanges:
            ex.lower_bound = 0
            ex.upper_bound = 1000

    if adj_carbon:
        print("Removing carbon sources: ")
        carbon_remove = ['man','sucr']
        tmp_inds = [c for c, name in enumerate(name_carbon_model_matched_adj) if name in carbon_remove]
        print(np.array(name_carbon_model_matched_adj)[tmp_inds])
        name_carbon_model_matched_adj = np.delete(name_carbon_model_matched_adj, tmp_inds, 0).tolist()
        name_carbon_experiment_matched_adj = np.delete(name_carbon_experiment_matched_adj, tmp_inds, 0).tolist()
        data_fitness_matched_adj = np.delete(data_fitness_matched_adj, tmp_inds, 1)

    return model_adj, name_genes_matched_adj, name_carbon_experiment_matched_adj, name_carbon_model_matched_adj, data_fitness_matched_adj

def check_environment(model_adj, name_medium_model, name_carbon_model_matched_adj):
    """
    Check if medium and carbon source components are present in the model. Returns exchange indices.
    """
    medium_ex_inds = np.ones(len(name_medium_model), dtype=int) * -1
    not_found_medium = []
    for m, comp in enumerate(name_medium_model):
        probe = 'EX_' + comp + '_e'
        found = False
        for e, ex in enumerate(model_adj.exchanges):
            if probe == ex.id:
                medium_ex_inds[m] = e
                found = True
                break
        if not found:
            not_found_medium.append(comp)
    if not_found_medium:
        print("Medium components not found in model:", not_found_medium)
    carbon_ex_inds = np.ones(len(name_carbon_model_matched_adj), dtype=int) * -1
    not_found_carbon = []
    for m, comp in enumerate(name_carbon_model_matched_adj):
        probe = 'EX_' + comp + '_e'
        found = False
        for e, ex in enumerate(model_adj.exchanges):
            if probe == ex.id:
                carbon_ex_inds[m] = e
                found = True
                break
        if not found:
            not_found_carbon.append(comp)
    if not_found_carbon:
        print("Carbon components not found in model:", not_found_carbon)
    return medium_ex_inds, carbon_ex_inds

def test_growth(model_adj, name_carbon_model_matched_adj, medium_ex_inds, carbon_ex_inds, thresh=0.001):
    """
    Test growth of model on base medium and each carbon source. Returns growth/no-growth results.
    """
    results = {}
    with model_adj:
        for e in medium_ex_inds:
            if e != -1:
                model_adj.exchanges[e].lower_bound = -1000
        solution = model_adj.optimize()
        base_growth = 1 if solution.objective_value and solution.objective_value > thresh else 0
        print(f"Growth with no carbon source (base medium only): {'Growth' if base_growth == 1 else 'No growth'} (Objective value: {solution.objective_value})")
        results['base_growth'] = base_growth
        base_growth_C = np.ones(len(carbon_ex_inds), dtype=float) * -1
        for e, idx in enumerate(carbon_ex_inds):
            if idx != -1:
                model_adj.exchanges[idx].lower_bound = -10
            sol = model_adj.slim_optimize()
            base_growth_C[e] = 1 if sol > thresh else 0
            carbon_name = name_carbon_model_matched_adj[e] if e < len(name_carbon_model_matched_adj) else f"Carbon {e}"
            print(f"Growth with carbon source '{carbon_name}': {'Growth' if base_growth_C[e] == 1 else 'No growth'} (Solution: {sol})")
            if idx != -1:
                model_adj.exchanges[idx].lower_bound = 0
        results['carbon_growth'] = base_growth_C
    return results

def simulate_phenotype(
    model_run,
    name_genes_matched_adj,
    name_carbon_model_matched_adj,
    medium_ex_inds,
    carbon_ex_inds,
    processed_df,
    objective_reaction,
    enzyme_upper_bound,
    thresh=0.001
):
    """
    Simulate growth/no-growth phenotypes for each gene and carbon source combination using:
    1. Baseline GEM (slim_optimize)
    2. Enzyme-constrained GEM (run_optimization_with_dataframe)
    Returns a tuple: (baseline_GEM, enzyme_constrained_GEM)
    """
    # Baseline GEM simulation
    baseline_GEM = np.zeros([len(name_genes_matched_adj), len(name_carbon_model_matched_adj)], dtype=float)
    for e in medium_ex_inds:
        if e != -1:
            model_run.exchanges[e].lower_bound = -1000
    for e in range(len(name_carbon_model_matched_adj)):
        print(f"Baseline GEM progress: {e+1}/{len(name_carbon_model_matched_adj)}", end='\r')
        if name_carbon_model_matched_adj[e] in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(name_carbon_model_matched_adj[e])
            baseline_GEM[:, e] = baseline_GEM[:, e_found]
        else:
            if carbon_ex_inds[e] != -1:
                model_run.exchanges[carbon_ex_inds[e]].lower_bound = -10
            for g in range(len(name_genes_matched_adj)):
                with model_run:
                    model_run.genes.get_by_id(name_genes_matched_adj[g]).knock_out()
                    solution = model_run.slim_optimize()
                    if np.isnan(solution):
                        solution = 0
                    baseline_GEM[g, e] = solution
            if carbon_ex_inds[e] != -1:
                model_run.exchanges[carbon_ex_inds[e]].lower_bound = 0

    # Enzyme-constrained GEM simulation
    if 'kcat_mean' in processed_df.columns:
        processed_df['kcat_mean'] = processed_df['kcat_mean'].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else x)

    enzyme_constrained_GEM = np.zeros([len(name_genes_matched_adj), len(name_carbon_model_matched_adj)], dtype=float)
    for e in range(len(name_carbon_model_matched_adj)):
        print(f"Enzyme-constrained GEM progress: {e+1}/{len(name_carbon_model_matched_adj)}", end='\r')
        if name_carbon_model_matched_adj[e] in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(name_carbon_model_matched_adj[e])
            enzyme_constrained_GEM[:, e] = enzyme_constrained_GEM[:, e_found]
        else:
            if carbon_ex_inds[e] != -1:
                model_run.exchanges[carbon_ex_inds[e]].lower_bound = -10
            for g in range(len(name_genes_matched_adj)):
                with model_run:
                    model_run.genes.get_by_id(name_genes_matched_adj[g]).knock_out()
                    try:
                        solution_value, df_FBA, gene_sequences_dict, _ = run_optimization_with_dataframe(
                            model=model_run,
                            processed_df=processed_df,
                            objective_reaction=objective_reaction,
                            enzyme_upper_bound=enzyme_upper_bound,
                            enzyme_ratio=True,
                            maximization=True,
                            multi_enzyme_off=False,
                            isoenzymes_off=False,
                            promiscuous_off=False,
                            complexes_off=False,
                            output_dir=None,
                            save_results=False,
                            print_reaction_conditions=False,
                            verbose=False
                        )
                        # Use solution_value as the simulated growth value
                        if solution_value is None or np.isnan(solution_value):
                            solution_value = 0
                        enzyme_constrained_GEM[g, e] = solution_value
                    except Exception as ex:
                        print(f"Enzyme-constrained optimization failed for gene {name_genes_matched_adj[g]}, carbon {name_carbon_model_matched_adj[e]}: {ex}")
                        enzyme_constrained_GEM[g, e] = 0
            if carbon_ex_inds[e] != -1:
                model_run.exchanges[carbon_ex_inds[e]].lower_bound = 0

    print("\nBaseline GEM and enzyme-constrained GEM simulation complete.")
    return baseline_GEM, enzyme_constrained_GEM

def simulate_phenotype_flux(
    model_run,
    name_genes_matched_adj,
    name_carbon_model_matched_adj,
    medium_ex_inds,
    carbon_ex_inds,
    processed_df,
    objective_reaction,
    enzyme_upper_bound,
    thresh=0.001
):
    """
    Simulate pFBA and record fluxes for all reactions for each gene and carbon source combination.
    Runs both baseline GEM and enzyme-constrained GEM scenarios.
    Returns a tuple: (baseline_fluxes, enzyme_constrained_fluxes)
    """
    n_genes = len(name_genes_matched_adj)
    n_carbons = len(name_carbon_model_matched_adj)
    n_rxns = len(model_run.reactions)

    # Baseline GEM flux simulation
    baseline_fluxes = np.zeros([n_genes, n_carbons, n_rxns], dtype=float)
    for e in medium_ex_inds:
        if e != -1:
            model_run.exchanges[e].lower_bound = -1000
    for e in range(n_carbons):
        print(f"Baseline GEM flux progress: {e+1}/{n_carbons}", end='\r')
        if name_carbon_model_matched_adj[e] in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(name_carbon_model_matched_adj[e])
            baseline_fluxes[:, e, :] = baseline_fluxes[:, e_found, :]
        else:
            if carbon_ex_inds[e] != -1:
                model_run.exchanges[carbon_ex_inds[e]].lower_bound = -10
            for g in range(n_genes):
                with model_run:
                    model_run.genes.get_by_id(name_genes_matched_adj[g]).knock_out()
                    solution = model_run.optimize()
                    if solution.status == 'optimal':
                        fluxes = solution.fluxes.values
                    else:
                        fluxes = np.zeros(n_rxns)
                    baseline_fluxes[g, e, :] = fluxes
            if carbon_ex_inds[e] != -1:
                model_run.exchanges[carbon_ex_inds[e]].lower_bound = 0

    # Enzyme-constrained GEM flux simulation
    if 'kcat_mean' in processed_df.columns:
        processed_df['kcat_mean'] = processed_df['kcat_mean'].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else x)

    enzyme_constrained_fluxes = np.zeros([n_genes, n_carbons, n_rxns], dtype=float)
    for e in range(n_carbons):
        print(f"Enzyme-constrained GEM flux progress: {e+1}/{n_carbons}", end='\r')
        if name_carbon_model_matched_adj[e] in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(name_carbon_model_matched_adj[e])
            enzyme_constrained_fluxes[:, e, :] = enzyme_constrained_fluxes[:, e_found, :]
        else:
            if carbon_ex_inds[e] != -1:
                model_run.exchanges[carbon_ex_inds[e]].lower_bound = -10
            for g in range(n_genes):
                with model_run:
                    model_run.genes.get_by_id(name_genes_matched_adj[g]).knock_out()
                    try:
                        solution_value, df_FBA, gene_sequences_dict, _ = run_optimization_with_dataframe(
                            model=model_run,
                            processed_df=processed_df,
                            objective_reaction=objective_reaction,
                            enzyme_upper_bound=enzyme_upper_bound,
                            enzyme_ratio=True,
                            maximization=True,
                            multi_enzyme_off=False,
                            isoenzymes_off=False,
                            promiscuous_off=False,
                            complexes_off=False,
                            output_dir=None,
                            save_results=False,
                            print_reaction_conditions=False,
                            verbose=False
                        )
                        # Try to extract fluxes from df_FBA
                        if df_FBA is not None and hasattr(df_FBA, 'loc') and 'fluxes' in df_FBA.columns:
                            # If df_FBA has a 'fluxes' column, use it
                            fluxes = df_FBA['fluxes'].values
                            if len(fluxes) == n_rxns:
                                enzyme_constrained_fluxes[g, e, :] = fluxes
                            else:
                                enzyme_constrained_fluxes[g, e, :] = np.zeros(n_rxns)
                        elif df_FBA is not None and hasattr(df_FBA, 'values') and df_FBA.shape[1] == n_rxns:
                            # If df_FBA is a DataFrame with fluxes as columns
                            enzyme_constrained_fluxes[g, e, :] = df_FBA.values[0]
                        else:
                            enzyme_constrained_fluxes[g, e, :] = np.zeros(n_rxns)
                    except Exception as ex:
                        print(f"Enzyme-constrained optimization failed for gene {name_genes_matched_adj[g]}, carbon {name_carbon_model_matched_adj[e]}: {ex}")
                        enzyme_constrained_fluxes[g, e, :] = np.zeros(n_rxns)
            if carbon_ex_inds[e] != -1:
                model_run.exchanges[carbon_ex_inds[e]].lower_bound = 0

    print("\nBaseline GEM and enzyme-constrained GEM flux simulation complete.")
    return baseline_fluxes, enzyme_constrained_fluxes


def calculate_phenotypes_with_dataframe(
    model,
    processed_df,
    objective_reaction,
    enzyme_upper_bound=0.125,
    enzyme_ratio=True,
    maximization=True,
    multi_enzyme_off=False,
    isoenzymes_off=False,
    promiscuous_off=False,
    complexes_off=False,
    output_dir=None,
    save_results=True,
    print_reaction_conditions=True,
    verbose=True
):
    """
    Wrapper for run_optimization_with_dataframe to calculate phenotypes for a kinGEMs model and processed DataFrame.
    Returns solution_value, df_FBA, gene_sequences_dict, output_filepath.
    """
    return run_optimization_with_dataframe(
        model=model,
        processed_df=processed_df,
        objective_reaction=objective_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        enzyme_ratio=enzyme_ratio,
        maximization=maximization,
        multi_enzyme_off=multi_enzyme_off,
        isoenzymes_off=isoenzymes_off,
        promiscuous_off=promiscuous_off,
        complexes_off=complexes_off,
        output_dir=output_dir,
        save_results=save_results,
        print_reaction_conditions=print_reaction_conditions,
        verbose=verbose
    )

# ============================================================================
# PARALLEL VALIDATION SIMULATION FUNCTIONS
# ============================================================================

def _simulate_gene_carbon_combo(model, processed_df, gene, carbon_idx, carbon_name,
                                medium_ex_inds, carbon_ex_inds, objective_reaction,
                                enzyme_upper_bound, mode='baseline'):
    """
    Simulate a single gene knockout × carbon source combination.

    Parameters
    ----------
    model : cobra.Model
        Model copy for this task
    processed_df : pandas.DataFrame
        Enzyme constraint data (only used for enzyme mode)
    gene : str
        Gene ID to knockout
    carbon_idx : int
        Index of carbon source
    carbon_name : str
        Name of carbon source
    medium_ex_inds : list
        Medium exchange reaction indices
    carbon_ex_inds : list
        Carbon source exchange reaction indices
    objective_reaction : str
        Biomass reaction ID
    enzyme_upper_bound : float
        Enzyme constraint
    mode : str
        'baseline' for slim_optimize or 'enzyme' for enzyme-constrained

    Returns
    -------
    tuple
        (gene, carbon_idx, growth_value)
    """
    from kinGEMs.modeling.optimize import run_optimization_with_dataframe

    # Set up medium
    for e in medium_ex_inds:
        if e != -1:
            model.exchanges[e].lower_bound = -1000

    # Set carbon source
    if carbon_ex_inds[carbon_idx] != -1:
        model.exchanges[carbon_ex_inds[carbon_idx]].lower_bound = -10

    # Knockout gene
    try:
        gene_obj = model.genes.get_by_id(gene)
        gene_obj.knock_out()
    except Exception:
        return (gene, carbon_idx, 0.0)

    # Simulate growth
    if mode == 'baseline':
        try:
            solution = model.slim_optimize()
            if np.isnan(solution):
                solution = 0.0
            return (gene, carbon_idx, solution)
        except Exception:
            return (gene, carbon_idx, 0.0)

    else:  # enzyme-constrained
        try:
            solution_value, _, _, _ = run_optimization_with_dataframe(
                model=model,
                processed_df=processed_df,
                objective_reaction=objective_reaction,
                enzyme_upper_bound=enzyme_upper_bound,
                enzyme_ratio=True,
                maximization=True,
                multi_enzyme_off=False,
                isoenzymes_off=False,
                promiscuous_off=False,
                complexes_off=False,
                output_dir=None,
                save_results=False,
                print_reaction_conditions=False,
                verbose=False
            )
            if solution_value is None or np.isnan(solution_value):
                solution_value = 0.0
            return (gene, carbon_idx, solution_value)
        except Exception:
            return (gene, carbon_idx, 0.0)


def _simulate_gene_carbon_chunk(model, processed_df, tasks, medium_ex_inds,
                                carbon_ex_inds, objective_reaction,
                                enzyme_upper_bound, mode='baseline'):
    """
    Process a chunk of gene × carbon combinations.

    Parameters
    ----------
    model : cobra.Model
        Model copy for this worker
    processed_df : pandas.DataFrame
        Enzyme constraint data
    tasks : list of tuples
        List of (gene, carbon_idx, carbon_name) tuples
    medium_ex_inds : list
        Medium exchange indices
    carbon_ex_inds : list
        Carbon exchange indices
    objective_reaction : str
        Biomass reaction ID
    enzyme_upper_bound : float
        Enzyme constraint
    mode : str
        'baseline' or 'enzyme'

    Returns
    -------
    list of tuples
        Results as (gene, carbon_idx, growth_value)
    """
    results = []
    for gene, carbon_idx, carbon_name in tasks:
        result = _simulate_gene_carbon_combo(
            model=model.copy(),
            processed_df=processed_df,
            gene=gene,
            carbon_idx=carbon_idx,
            carbon_name=carbon_name,
            medium_ex_inds=medium_ex_inds,
            carbon_ex_inds=carbon_ex_inds,
            objective_reaction=objective_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            mode=mode
        )
        results.append(result)
    return results


def simulate_phenotype_parallel(
    model_run,
    name_genes_matched_adj,
    name_carbon_model_matched_adj,
    medium_ex_inds,
    carbon_ex_inds,
    processed_df,
    objective_reaction,
    enzyme_upper_bound,
    thresh=0.001,
    n_workers=None,
    chunk_size=None,
    method='dask'
):
    """
    Parallel version of simulate_phenotype using Dask or multiprocessing.

    Simulates growth/no-growth phenotypes for each gene and carbon source combination:
    1. Baseline GEM (slim_optimize) - run in parallel
    2. Enzyme-constrained GEM (run_optimization_with_dataframe) - run in parallel

    Parameters
    ----------
    model_run : cobra.Model
        COBRA model
    name_genes_matched_adj : list
        List of gene IDs
    name_carbon_model_matched_adj : list
        List of carbon source names
    medium_ex_inds : list
        Medium exchange reaction indices
    carbon_ex_inds : list
        Carbon source exchange reaction indices
    processed_df : pandas.DataFrame
        Enzyme constraint data
    objective_reaction : str
        Biomass reaction ID
    enzyme_upper_bound : float
        Enzyme constraint (default: 0.15)
    thresh : float
        Growth threshold (default: 0.001)
    n_workers : int, optional
        Number of parallel workers (default: auto-detect)
    chunk_size : int, optional
        Tasks per chunk (default: auto-calculate)
    method : str
        'dask' or 'multiprocessing' (default: 'dask')

    Returns
    -------
    tuple
        (baseline_GEM, enzyme_constrained_GEM) as numpy arrays
    """
    import os
    import logging

    # Determine number of workers
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    n_genes = len(name_genes_matched_adj)
    n_carbons = len(name_carbon_model_matched_adj)
    total_tasks = n_genes * n_carbons

    # Calculate optimal chunk size
    if chunk_size is None:
        # Aim for ~15-20 chunks per worker
        chunk_size = max(1, total_tasks // (n_workers * 15))

    print(f"\n  Parallel validation configuration:")
    print(f"    Method: {method}")
    print(f"    Workers: {n_workers}")
    print(f"    Total simulations: {total_tasks} ({n_genes} genes × {n_carbons} carbons)")
    print(f"    Chunk size: {chunk_size}")
    print(f"    Number of chunks: {(total_tasks + chunk_size - 1) // chunk_size}")

    # Estimate memory
    try:
        model_size_mb = (len(model_run.reactions) * 0.05 +
                        len(model_run.metabolites) * 0.02 +
                        len(model_run.genes) * 0.01)
        df_size_mb = len(processed_df) * 0.001 if processed_df is not None else 0
        total_size_mb = model_size_mb + df_size_mb
        estimated_memory_gb = (total_size_mb * n_workers) / 1000
        print(f"    Estimated memory: ~{estimated_memory_gb:.1f} GB")
        if estimated_memory_gb > 8:
            print("    ⚠️  Warning: High memory usage expected")
    except Exception:
        pass

    # Handle duplicate carbon sources (use cached results)
    unique_carbons = []
    carbon_cache_map = {}
    for idx, carbon in enumerate(name_carbon_model_matched_adj):
        if carbon in name_carbon_model_matched_adj[:idx]:
            # This carbon was already processed
            cached_idx = name_carbon_model_matched_adj[:idx].index(carbon)
            carbon_cache_map[idx] = cached_idx
        else:
            unique_carbons.append((idx, carbon))

    print(f"    Unique carbon sources: {len(unique_carbons)} (cached: {len(carbon_cache_map)})")

    # Create tasks for unique carbons only
    tasks = []
    for carbon_idx, carbon_name in unique_carbons:
        for gene in name_genes_matched_adj:
            tasks.append((gene, carbon_idx, carbon_name))

    # Create chunks
    chunks = [tasks[i:i+chunk_size] for i in range(0, len(tasks), chunk_size)]

    # ===== Baseline GEM Simulation (Parallel) =====
    print(f"\n  Starting parallel baseline GEM simulation...")

    if method.lower() == 'multiprocessing':
        baseline_results = _run_validation_multiprocessing(
            model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
            objective_reaction, enzyme_upper_bound, n_workers, mode='baseline'
        )
    else:  # dask
        baseline_results = _run_validation_dask(
            model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
            objective_reaction, enzyme_upper_bound, n_workers, mode='baseline'
        )

    # Flatten results
    flat_baseline = []
    for chunk_results in baseline_results:
        flat_baseline.extend(chunk_results)

    # Build baseline matrix
    baseline_GEM = np.zeros((n_genes, n_carbons), dtype=float)
    for gene, carbon_idx, growth_value in flat_baseline:
        gene_idx = name_genes_matched_adj.index(gene)
        baseline_GEM[gene_idx, carbon_idx] = growth_value

    # Fill in cached carbon sources
    for cached_idx, original_idx in carbon_cache_map.items():
        baseline_GEM[:, cached_idx] = baseline_GEM[:, original_idx]

    print(f"  Baseline GEM simulation complete.")

    # ===== Enzyme-constrained GEM Simulation (Parallel) =====
    print(f"\n  Starting parallel enzyme-constrained GEM simulation...")

    # Clean processed_df
    if 'kcat_mean' in processed_df.columns:
        processed_df['kcat_mean'] = processed_df['kcat_mean'].apply(
            lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else x
        )

    if method.lower() == 'multiprocessing':
        enzyme_results = _run_validation_multiprocessing(
            model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
            objective_reaction, enzyme_upper_bound, n_workers, mode='enzyme'
        )
    else:  # dask
        enzyme_results = _run_validation_dask(
            model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
            objective_reaction, enzyme_upper_bound, n_workers, mode='enzyme'
        )

    # Flatten results
    flat_enzyme = []
    for chunk_results in enzyme_results:
        flat_enzyme.extend(chunk_results)

    # Build enzyme-constrained matrix
    enzyme_constrained_GEM = np.zeros((n_genes, n_carbons), dtype=float)
    for gene, carbon_idx, growth_value in flat_enzyme:
        gene_idx = name_genes_matched_adj.index(gene)
        enzyme_constrained_GEM[gene_idx, carbon_idx] = growth_value

    # Fill in cached carbon sources
    for cached_idx, original_idx in carbon_cache_map.items():
        enzyme_constrained_GEM[:, cached_idx] = enzyme_constrained_GEM[:, original_idx]

    print(f"  Enzyme-constrained GEM simulation complete.")
    print(f"\n  Parallel validation simulations finished!")

    return baseline_GEM, enzyme_constrained_GEM


def _run_validation_dask(model, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
                        objective_reaction, enzyme_upper_bound, n_workers, mode='baseline'):
    """Execute validation using Dask."""
    import logging

    try:
        from dask import compute, delayed
        from dask.distributed import Client
    except ImportError:
        raise ImportError(
            "Dask is required for parallel validation. Install with: pip install dask[distributed]"
        )

    # Create Dask client
    client = None
    try:
        client = Client(
            n_workers=n_workers,
            processes=True,
            threads_per_worker=1,
            silence_logs=logging.ERROR
        )
        try:
            print(f"    Dask dashboard: {client.dashboard_link}")
        except Exception:
            print("    Dask dashboard: (install bokeh>=3.1.0 to enable dashboard)")
    except Exception as e:
        print(f"    ⚠️  Warning: Could not start Dask client: {e}")
        print("    Falling back to sequential execution...")

    # Create delayed tasks
    tasks = []
    for chunk in chunks:
        tasks.append(
            delayed(_simulate_gene_carbon_chunk)(
                model.copy(),
                processed_df,
                chunk,
                medium_ex_inds,
                carbon_ex_inds,
                objective_reaction,
                enzyme_upper_bound,
                mode
            )
        )

    # Execute
    try:
        results = compute(*tasks)
    finally:
        if client:
            client.close()

    return results


def _run_validation_multiprocessing(model, processed_df, chunks, medium_ex_inds,
                                    carbon_ex_inds, objective_reaction,
                                    enzyme_upper_bound, n_workers, mode='baseline'):
    """Execute validation using multiprocessing.Pool."""
    from functools import partial
    from multiprocessing import Pool

    # Create partial function
    worker_func = partial(
        _simulate_gene_carbon_chunk,
        model.copy(),
        processed_df,
        medium_ex_inds=medium_ex_inds,
        carbon_ex_inds=carbon_ex_inds,
        objective_reaction=objective_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        mode=mode
    )

    # Execute in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_func, chunks)

    return results
# import libraries
import copy
import os

import cobra
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
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
from tqdm import tqdm

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
        
        print(f"\nTesting growth on {len(carbon_ex_inds)} carbon sources...")
        for e, idx in tqdm(enumerate(carbon_ex_inds), total=len(carbon_ex_inds), desc="Testing carbon sources", unit="carbon"):
            if idx != -1:
                model_adj.exchanges[idx].lower_bound = -10
            sol = model_adj.slim_optimize()
            base_growth_C[e] = 1 if sol > thresh else 0
            carbon_name = name_carbon_model_matched_adj[e] if e < len(name_carbon_model_matched_adj) else f"Carbon {e}"
            growth_status = 'Growth' if base_growth_C[e] == 1 else 'No growth'
            tqdm.write(f"  Growth with carbon source '{carbon_name}': {growth_status} (Solution: {sol})")
            if idx != -1:
                model_adj.exchanges[idx].lower_bound = 0
        results['carbon_growth'] = base_growth_C
    return results

def _simulate_baseline_gene_carbon(model_json, gene_id, carbon_idx, carbon_ex_idx, medium_ex_inds, timeout):
    """
    Helper function to simulate a single gene knockout on a single carbon source for baseline GEM.
    Designed to run in parallel with Dask.
    """
    import cobra
    import numpy as np
    
    # Reconstruct model from JSON
    model = cobra.io.json.from_json(model_json)
    
    # Set solver timeout
    try:
        model.solver.configuration.timeout = timeout
    except Exception:
        pass
    
    # Set medium bounds
    for e in medium_ex_inds:
        if e != -1:
            model.exchanges[e].lower_bound = -1000
    
    # Set carbon source bound
    if carbon_ex_idx != -1:
        model.exchanges[carbon_ex_idx].lower_bound = -10
    
    # Knock out gene and optimize
    with model:
        model.genes.get_by_id(gene_id).knock_out()
        try:
            solution = model.slim_optimize()
            if solution is None or np.isnan(solution) or np.isinf(solution):
                solution = 0
        except Exception:
            solution = 0
    
    return (carbon_idx, solution)


def _simulate_ecgem_gene_carbon(model_json, gene_id, carbon_idx, carbon_ex_idx, medium_ex_inds, 
                                 processed_df, objective_reaction, enzyme_upper_bound, timeout):
    """
    Helper function to simulate a single gene knockout on a single carbon source for enzyme-constrained GEM.
    Designed to run in parallel with Dask.
    """
    import cobra
    import numpy as np
    from kinGEMs.modeling.optimize import run_optimization_with_dataframe
    
    # Reconstruct model from JSON
    model = cobra.io.json.from_json(model_json)
    
    # Set solver timeout
    try:
        model.solver.configuration.timeout = timeout
    except Exception:
        pass
    
    # Set medium bounds
    for e in medium_ex_inds:
        if e != -1:
            model.exchanges[e].lower_bound = -1000
    
    # Set carbon source bound
    if carbon_ex_idx != -1:
        model.exchanges[carbon_ex_idx].lower_bound = -10
    
    # Knock out gene and optimize with enzyme constraints
    with model:
        model.genes.get_by_id(gene_id).knock_out()
        try:
            solution_value, df_FBA, gene_sequences_dict, _ = run_optimization_with_dataframe(
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
                solution_value = 0
        except Exception:
            solution_value = 0
    
    return (carbon_idx, solution_value)


def simulate_phenotype(
    model_run,
    name_genes_matched_adj,
    name_carbon_model_matched_adj,
    medium_ex_inds,
    carbon_ex_inds,
    processed_df,
    objective_reaction,
    enzyme_upper_bound,
    thresh=0.001,
    timeout=10
):
    """
    Simulate growth/no-growth phenotypes for each gene and carbon source combination using:
    1. Baseline GEM (slim_optimize)
    2. Enzyme-constrained GEM (run_optimization_with_dataframe)
    Returns a tuple: (baseline_GEM, enzyme_constrained_GEM)
    """
    n_genes = len(name_genes_matched_adj)
    n_carbons = len(name_carbon_model_matched_adj)
    
    # Set solver timeout
    try:
        model_run.solver.configuration.timeout = timeout
    except Exception:
        pass

    # Baseline GEM simulation
    print("\nBaseline GEM Simulation")
    print(f"Total simulations: {n_genes} genes × {n_carbons} carbon sources = {n_genes * n_carbons} optimizations")
    try:
        solver_name = model_run.solver.interface.__name__
    except Exception:
        solver_name = "unknown"
    print(f"Solver: {solver_name} (timeout: {timeout}s per optimization)")
    
    baseline_GEM = np.zeros([n_genes, n_carbons], dtype=float)
    for e in medium_ex_inds:
        if e != -1:
            model_run.exchanges[e].lower_bound = -1000
    
    # Calculate total iterations for progress bar
    total_baseline_iterations = sum(
        n_genes if name_carbon_model_matched_adj[e] not in name_carbon_model_matched_adj[:e] else 0
        for e in range(n_carbons)
    )
    
    timeout_count = 0
    error_count = 0
    invalid_count = 0
    
    with tqdm(total=total_baseline_iterations, desc="Baseline GEM", unit="simulation") as pbar:
        for e in range(n_carbons):
            carbon_name = name_carbon_model_matched_adj[e]
            if carbon_name in name_carbon_model_matched_adj[:e]:
                e_found = name_carbon_model_matched_adj[:e].index(carbon_name)
                baseline_GEM[:, e] = baseline_GEM[:, e_found]
                pbar.set_postfix({"carbon": carbon_name, "status": "cached"})
            else:
                if carbon_ex_inds[e] != -1:
                    model_run.exchanges[carbon_ex_inds[e]].lower_bound = -10
                for g in range(n_genes):
                    with model_run:
                        model_run.genes.get_by_id(name_genes_matched_adj[g]).knock_out()
                        try:
                            solution = model_run.slim_optimize()
                            # Check if optimization actually timed out
                            if solution is None:
                                timeout_count += 1
                                gene_id = name_genes_matched_adj[g]
                                #tqdm.write(f"Timeout #{timeout_count}: Gene {gene_id} on carbon {carbon_name} (returned None)")
                                solution = 0
                            elif np.isnan(solution) or np.isinf(solution):
                                invalid_count += 1
                                gene_id = name_genes_matched_adj[g]
                                #tqdm.write(f"Invalid result #{error_count}: Gene {gene_id} on carbon {carbon_name} (returned {solution})")
                                solution = 0
                            baseline_GEM[g, e] = solution
                        except Exception as ex:
                            # Handle timeout or other solver errors
                            msg = str(ex).lower()
                            gene_id = name_genes_matched_adj[g]
                            if ('timeout' in msg) or ('time limit' in msg) or ('tmlim' in msg):
                                timeout_count += 1
                                #tqdm.write(f"Timeout #{timeout_count}: Gene {gene_id} on carbon {carbon_name}")
                            else:
                                error_count += 1
                                #tqdm.write(f"Error #{error_count}: Gene {gene_id} on carbon {carbon_name}: {ex}")
                            baseline_GEM[g, e] = 0
                    pbar.update(1)
                    if g % 50 == 0:
                        pbar.set_postfix({
                            "carbon": carbon_name, 
                            "gene": f"{g+1}/{n_genes}",
                            "timeouts": timeout_count,
                            "errors": error_count,
                            "invalid": invalid_count
                        })
                if carbon_ex_inds[e] != -1:
                    model_run.exchanges[carbon_ex_inds[e]].lower_bound = 0
    
    # Print summary of issues
    print("\nBaseline GEM Issues Summary:")
    if timeout_count > 0:
        print(f"  - Timeouts: {timeout_count}/{total_baseline_iterations} optimizations")
    if error_count > 0:
        print(f"  - Errors: {error_count}/{total_baseline_iterations} optimizations")
    if invalid_count > 0:
        print(f"  - Invalid: {invalid_count}/{total_baseline_iterations} optimizations")
    else:
        print(f"  - No errors or invalid results")

    # Enzyme-constrained GEM simulation
    print("\nEnzyme-Constrained GEM Simulation")
    print(f"Total simulations: {n_genes} genes x {n_carbons} carbon sources = {n_genes * n_carbons} optimizations")
    
    if 'kcat_mean' in processed_df.columns:
        processed_df['kcat_mean'] = processed_df['kcat_mean'].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else x)

    enzyme_constrained_GEM = np.zeros([len(name_genes_matched_adj), len(name_carbon_model_matched_adj)], dtype=float)
    
    # Calculate total iterations for progress bar
    total_ec_iterations = sum(
        n_genes if name_carbon_model_matched_adj[e] not in name_carbon_model_matched_adj[:e] else 0
        for e in range(n_carbons)
    )
    
    kingems_error_count = 0
    kingems_invalid_count = 0
    
    with tqdm(total=total_ec_iterations, desc="Enzyme-Constrained GEM", unit="simulation") as pbar:
        for e in range(len(name_carbon_model_matched_adj)):
            carbon_name = name_carbon_model_matched_adj[e]
            if carbon_name in name_carbon_model_matched_adj[:e]:
                e_found = name_carbon_model_matched_adj[:e].index(carbon_name)
                enzyme_constrained_GEM[:, e] = enzyme_constrained_GEM[:, e_found]
                pbar.set_postfix({"carbon": carbon_name, "status": "cached"})
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
                                kingems_invalid_count += 1
                            enzyme_constrained_GEM[g, e] = solution_value
                        except Exception as ex:
                            tqdm.write(f"Enzyme-constrained optimization failed for gene {name_genes_matched_adj[g]}, carbon {carbon_name}: {ex}")
                            kingems_error_count += 1
                            enzyme_constrained_GEM[g, e] = 0
                    pbar.update(1)
                    if g % 50 == 0:
                        pbar.set_postfix({"carbon": carbon_name, "gene": f"{g+1}/{n_genes}", 
                                          "invalid": invalid_count, "errors": error_count})
                if carbon_ex_inds[e] != -1:
                    model_run.exchanges[carbon_ex_inds[e]].lower_bound = 0

    # Print summary of issues
    print("\nEnzyme-constrained GEM Issues Summary:")
    if kingems_invalid_count > 0:
        print(f"  - Invalid: {kingems_invalid_count}/{total_ec_iterations} optimizations")
    if kingems_error_count > 0:
        print(f"  - Errors: {kingems_error_count}/{total_ec_iterations} optimizations")
    else:
        print("  - No errors or invalid results")

    print("\nBaseline GEM and enzyme-constrained GEM simulation complete.")
    return baseline_GEM, enzyme_constrained_GEM


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
    timeout=10,
    n_workers=None,
    use_distributed=False
):
    """
    Parallel version of simulate_phenotype using Dask for multiprocessing.
    Simulates growth/no-growth phenotypes for each gene and carbon source combination using:
    1. Baseline GEM (slim_optimize)
    2. Enzyme-constrained GEM (run_optimization_with_dataframe)
    """
    import multiprocessing as mp
    
    n_genes = len(name_genes_matched_adj)
    n_carbons = len(name_carbon_model_matched_adj)
    
    # Default to number of CPU cores
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    print(f"Parallel phenotype simulation using Dask")
    print(f"Workers: {n_workers}")
    print(f"Scheduler: {'Distributed' if use_distributed else 'Threaded'}")
    print(f"Total simulations: {n_genes} genes × {n_carbons} carbon sources = {n_genes * n_carbons}")
    
    # Serialize model to JSON for passing to worker processes
    print("\nSerializing model for parallel processing...")
    model_json = cobra.io.json.to_json(model_run)
    
    # Set solver timeout
    try:
        model_run.solver.configuration.timeout = timeout
    except Exception:
        pass
    
    # Initialize Dask client if using distributed scheduler
    client = None
    if use_distributed:
        print(f"Starting Dask distributed cluster with {n_workers} workers...")
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True)
        client = Client(cluster)
        print(f"Dask dashboard available at: {client.dashboard_link}")
    
    # Baseline GEM simulation
    print("\nBaseline GEM simulation (Parallel)")
    
    baseline_GEM = np.zeros([n_genes, n_carbons], dtype=float)
    
    # Build list of tasks, handling carbon source caching
    baseline_tasks = []
    task_map = {}  # Maps (gene_idx, carbon_idx) to task
    
    for e in range(n_carbons):
        carbon_name = name_carbon_model_matched_adj[e]
        
        # Check if this carbon source was already processed
        if carbon_name in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(carbon_name)
            baseline_GEM[:, e] = baseline_GEM[:, e_found]
            print(f"Carbon '{carbon_name}' (idx {e}): Using cached results from idx {e_found}")
        else:
            # Create tasks for this carbon source
            for g in range(n_genes):
                task = delayed(_simulate_baseline_gene_carbon)(
                    model_json,
                    name_genes_matched_adj[g],
                    e,
                    carbon_ex_inds[e],
                    medium_ex_inds,
                    timeout
                )
                baseline_tasks.append(task)
                task_map[(g, e)] = len(baseline_tasks) - 1
    
    # Execute tasks in parallel
    if baseline_tasks:
        print(f"\nExecuting {len(baseline_tasks)} baseline simulations in parallel...")
        
        if use_distributed and client:
            # using distributed scheduler - submit all, gather all
            futures = client.compute(baseline_tasks)
            results = client.gather(futures) 
            print(f"Completed {len(results)} simulations")
        else:
            # using threaded scheduler - compute all at once
            results = dask.compute(*baseline_tasks, scheduler='threads', num_workers=n_workers)
            results = list(results)
            print(f"Completed {len(results)} simulations")
        
        # Collect results back into the matrix
        result_idx = 0
        for e in range(n_carbons):
            carbon_name = name_carbon_model_matched_adj[e]
            if carbon_name not in name_carbon_model_matched_adj[:e]:
                for g in range(n_genes):
                    carbon_idx, solution = results[result_idx]
                    baseline_GEM[g, carbon_idx] = solution
                    result_idx += 1
    
    print("\nBaseline GEM simulation complete!")
    print(f"Non-zero results: {np.count_nonzero(baseline_GEM)}/{baseline_GEM.size}")
    
    # kinGEMs simulation
    print("\nEnzyme-constrained GEM simulation (Parallel)")
    
    if 'kcat_mean' in processed_df.columns:
        processed_df['kcat_mean'] = processed_df['kcat_mean'].apply(
            lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else x
        )
    
    enzyme_constrained_GEM = np.zeros([n_genes, n_carbons], dtype=float)
    
    # Build list of tasks
    ecgem_tasks = []
    ecgem_task_map = {}
    
    for e in range(n_carbons):
        carbon_name = name_carbon_model_matched_adj[e]
        
        # Check if this carbon source was already processed
        if carbon_name in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(carbon_name)
            enzyme_constrained_GEM[:, e] = enzyme_constrained_GEM[:, e_found]
            print(f"Carbon '{carbon_name}' (idx {e}): Using cached results from idx {e_found}")
        else:
            # Create tasks for this carbon source
            for g in range(n_genes):
                task = delayed(_simulate_ecgem_gene_carbon)(
                    model_json,
                    name_genes_matched_adj[g],
                    e,
                    carbon_ex_inds[e],
                    medium_ex_inds,
                    processed_df,
                    objective_reaction,
                    enzyme_upper_bound,
                    timeout
                )
                ecgem_tasks.append(task)
                ecgem_task_map[(g, e)] = len(ecgem_tasks) - 1
    
    # Execute tasks in parallel
    if ecgem_tasks:
        print(f"\nExecuting {len(ecgem_tasks)} enzyme-constrained simulations in parallel...")
        with tqdm(total=len(ecgem_tasks), desc="Enzyme-Constrained GEM (Parallel)", unit="simulation") as pbar:
            if use_distributed and client:
                # using distributed scheduler
                futures = client.compute(ecgem_tasks)
                ecgem_results = []
                for future in futures:
                    result = future.result()
                    ecgem_results.append(result)
                    pbar.update(1)
            else:
                # using threaded scheduler
                ecgem_results = []
                for task in ecgem_tasks:
                    result = task.compute()
                    ecgem_results.append(result)
                    pbar.update(1)
        
        # Collect results back into the matrix
        result_idx = 0
        for e in range(n_carbons):
            carbon_name = name_carbon_model_matched_adj[e]
            if carbon_name not in name_carbon_model_matched_adj[:e]:
                for g in range(n_genes):
                    carbon_idx, solution = ecgem_results[result_idx]
                    enzyme_constrained_GEM[g, carbon_idx] = solution
                    result_idx += 1
    
    print("\nEnzyme-constrained GEM simulation complete!")
    print(f"Non-zero results: {np.count_nonzero(enzyme_constrained_GEM)}/{enzyme_constrained_GEM.size}")
    
    # Clean up Dask client if using distributed scheduler
    if client:
        client.close()
        cluster.close()
        print("\nDask cluster closed.")
    
    print("Parallel simulation complete!")
    
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
    thresh=0.001,
    timeout=10
):
    """
    Simulate pFBA and record fluxes for all reactions for each gene and carbon source combination.
    Runs both baseline GEM and enzyme-constrained GEM scenarios.
    Returns a tuple: (baseline_fluxes, enzyme_constrained_fluxes)
    """
    n_genes = len(name_genes_matched_adj)
    n_carbons = len(name_carbon_model_matched_adj)
    n_rxns = len(model_run.reactions)
    
    # Solver timeout
    model_run.solver.configuration.timeout = timeout

    # Baseline GEM flux simulation
    print("\nBaseline GEM Flux Simulation")
    print(f"Total simulations: {n_genes} genes x {n_carbons} carbon sources x {n_rxns} reactions")
    print(f"Solver: {model_run.solver.interface.__name__} (timeout: {timeout}s per optimization)")
    
    baseline_fluxes = np.zeros([n_genes, n_carbons, n_rxns], dtype=float)
    for e in medium_ex_inds:
        if e != -1:
            model_run.exchanges[e].lower_bound = -1000
    
    # Total iterations for progress bar
    total_baseline_iterations = sum(
        n_genes if name_carbon_model_matched_adj[e] not in name_carbon_model_matched_adj[:e] else 0
        for e in range(n_carbons)
    )
    
    timeout_count = 0
    error_count = 0
    invalid_count = 0
    
    with tqdm(total=total_baseline_iterations, desc="Baseline Flux", unit="simulation") as pbar:
        for e in range(n_carbons):
            carbon_name = name_carbon_model_matched_adj[e]
            if carbon_name in name_carbon_model_matched_adj[:e]:
                e_found = name_carbon_model_matched_adj[:e].index(carbon_name)
                baseline_fluxes[:, e, :] = baseline_fluxes[:, e_found, :]
                pbar.set_postfix({"carbon": carbon_name, "status": "cached"})
            else:
                if carbon_ex_inds[e] != -1:
                    model_run.exchanges[carbon_ex_inds[e]].lower_bound = -10
                for g in range(n_genes):
                    with model_run:
                        model_run.genes.get_by_id(name_genes_matched_adj[g]).knock_out()
                        try:
                            solution = model_run.optimize()
                            if solution.status == 'optimal':
                                fluxes = solution.fluxes.values
                                baseline_fluxes[g, e, :] = fluxes
                            else:
                                invalid_count += 1
                                baseline_fluxes[g, e, :] = np.zeros(n_rxns)
                        except Exception as ex:
                            # Handle timeout or other solver errors
                            error_msg = str(ex).lower()
                            if 'timeout' in error_msg or 'time limit' in error_msg or 'tmlim' in error_msg:
                                timeout_count += 1
                                #tqdm.write(f"Timeout #{timeout_count}: Gene {name_genes_matched_adj[g]} on carbon {carbon_name}")
                            else:
                                #tqdm.write(f"Error: Gene {name_genes_matched_adj[g]} on carbon {carbon_name}: {ex}")
                                error_count += 1
                            baseline_fluxes[g, e, :] = np.zeros(n_rxns)
                    
                    pbar.update(1)
                    if g % 50 == 0:
                        pbar.set_postfix({"carbon": carbon_name, "gene": f"{g+1}/{n_genes}", "timeouts": timeout_count})
                
                if carbon_ex_inds[e] != -1:
                    model_run.exchanges[carbon_ex_inds[e]].lower_bound = 0
    
    if timeout_count > 0:
        print(f"\nTotal optimizations that timed out: {timeout_count}/{total_baseline_iterations}")

    # Enzyme-constrained GEM flux simulation
    print("\nEnzyme-Constrained GEM Flux Simulation")
    print(f"Total simulations: {n_genes} genes x {n_carbons} carbon sources x {n_rxns} reactions")
    
    if 'kcat_mean' in processed_df.columns:
        processed_df['kcat_mean'] = processed_df['kcat_mean'].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else x)

    enzyme_constrained_fluxes = np.zeros([n_genes, n_carbons, n_rxns], dtype=float)
    
    # Total iterations for progress bar
    total_ec_iterations = sum(
        n_genes if name_carbon_model_matched_adj[e] not in name_carbon_model_matched_adj[:e] else 0
        for e in range(n_carbons)
    )
    
    with tqdm(total=total_ec_iterations, desc="Enzyme-Constrained Flux", unit="simulation") as pbar:
        for e in range(len(name_carbon_model_matched_adj)):
            carbon_name = name_carbon_model_matched_adj[e]
            if carbon_name in name_carbon_model_matched_adj[:e]:
                e_found = name_carbon_model_matched_adj[:e].index(carbon_name)
                enzyme_constrained_fluxes[:, e, :] = enzyme_constrained_fluxes[:, e_found, :]
                pbar.set_postfix({"carbon": carbon_name, "status": "cached"})
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
                            tqdm.write(f"Enzyme-constrained optimization failed for gene {name_genes_matched_adj[g]}, carbon {carbon_name}: {ex}")
                            enzyme_constrained_fluxes[g, e, :] = np.zeros(n_rxns)
                    pbar.update(1)
                    if g % 50 == 0:
                        pbar.set_postfix({"carbon": carbon_name, "gene": f"{g+1}/{n_genes}"})
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
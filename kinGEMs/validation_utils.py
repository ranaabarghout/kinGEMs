# import libraries
import copy
import os
import signal
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as confusion_matrix

from kinGEMs.config import ECOLI_VALIDATION_DIR
from kinGEMs.modeling.optimize import run_optimization_with_dataframe


class TimeoutException(Exception):
    """Exception raised when an optimization times out."""
    pass


@contextmanager
def time_limit(seconds, description="operation"):
    """
    Context manager that raises TimeoutException if code block exceeds time limit.

    Parameters
    ----------
    seconds : int
        Maximum time allowed in seconds
    description : str
        Description of the operation for error messages

    Raises
    ------
    TimeoutException
        If the code block exceeds the time limit
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"{description} exceeded {seconds}s timeout")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Cancel the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _is_model_with_reverse_exchanges(model):
    """
    Check if the model has already been converted to use _reverse exchange reactions.

    Parameters
    ----------
    model : cobra.Model
        The model to check

    Returns
    -------
    bool
        True if model has _reverse exchanges, False otherwise
    """
    exchange_ids = {ex.id for ex in model.exchanges}
    # Check if any exchange has a corresponding _reverse version
    for ex_id in exchange_ids:
        if ex_id + '_reverse' in exchange_ids:
            return True
    return False


def _get_exchange_directionality(model, ex_id):
    """
    Determine exchange directionality by checking forward and _reverse versions.

    Parameters
    ----------
    model : cobra.Model
        The model
    ex_id : str
        Exchange reaction ID (without _reverse suffix)

    Returns
    -------
    str
        One of: 'uptake_only', 'export_only', 'bidirectional'
    """
    # Get or create the IDs
    reverse_id = ex_id + '_reverse'

    try:
        forward_ex = model.reactions.get_by_id(ex_id)
        forward_ub = forward_ex.upper_bound
    except KeyError:
        forward_ub = 0

    try:
        reverse_ex = model.reactions.get_by_id(reverse_id)
        reverse_ub = reverse_ex.upper_bound
    except KeyError:
        reverse_ub = 0

    # Determine directionality based on which direction has capacity
    if forward_ub > 0 and reverse_ub <= 0:
        return 'export_only'
    elif forward_ub <= 0 and reverse_ub > 0:
        return 'uptake_only'
    elif forward_ub > 0 and reverse_ub > 0:
        return 'bidirectional'
    else:
        return 'bidirectional'  # Default to bidirectional if unclear


def prepare_model(model, handle_irreversible=True):
    """
    Prepare model for validation by setting appropriate exchange bounds.

    Parameters
    ----------
    model : cobra.Model
        The model to prepare
    handle_irreversible : bool
        If True, preserve irreversible structure for irreversible exchanges
        If False, use original logic (all exchanges become reversible)

    Returns
    -------
    cobra.Model
        Model with exchange bounds set for validation
    """
    if not handle_irreversible:
        print("Using irreversible model handling: all exchanges set to reversible")
        for ex in model.exchanges:
            ex.lower_bound = 0
            ex.upper_bound = 1000
    else:
        print("Using irreversible model handling: converting to non-negative exchange format")

        # Check if model has _reverse exchanges (already split)
        has_reverse_exchanges = _is_model_with_reverse_exchanges(model)

        if has_reverse_exchanges:
            print("  Model has _reverse exchanges - using forward/reverse pairs for directionality")
            # Only process forward exchanges (skip the _reverse ones)
            processed = set()
            for ex in model.exchanges:
                if ex.id in processed or ex.id.endswith('_reverse'):
                    continue
                processed.add(ex.id)

                # Determine directionality from forward and reverse versions
                directionality = _get_exchange_directionality(model, ex.id)

                if directionality == 'uptake_only':
                    # Uptake: forward (0, 0), reverse (0, 1000)
                    ex.lower_bound = 0
                    ex.upper_bound = 0
                    try:
                        reverse_ex = model.reactions.get_by_id(ex.id + '_reverse')
                        reverse_ex.lower_bound = 0
                        reverse_ex.upper_bound = 1000
                    except KeyError:
                        pass
                elif directionality == 'export_only':
                    # Export: forward (0, 1000), reverse (0, 0)
                    ex.lower_bound = 0
                    ex.upper_bound = 1000
                    try:
                        reverse_ex = model.reactions.get_by_id(ex.id + '_reverse')
                        reverse_ex.lower_bound = 0
                        reverse_ex.upper_bound = 0
                    except KeyError:
                        pass
                else:  # bidirectional
                    # Bidirectional: forward (0, 1000), reverse (0, 1000)
                    ex.lower_bound = 0
                    ex.upper_bound = 1000
                    try:
                        reverse_ex = model.reactions.get_by_id(ex.id + '_reverse')
                        reverse_ex.lower_bound = 0
                        reverse_ex.upper_bound = 1000
                    except KeyError:
                        pass
        else:
            # Original logic for models without _reverse exchanges
            print("  Model does not have _reverse exchanges - using original bounds")
            for ex in model.exchanges:
                lb, ub = ex.lower_bound, ex.upper_bound

                if lb < 0 and ub <= 0:
                    # Uptake-only exchange: convert to forward (0, 0), reverse (0, |lb|)
                    ex.lower_bound = 0
                    ex.upper_bound = max(0, -lb)  # Use reverse capacity
                elif lb >= 0 and ub > 0:
                    # Export-only exchange: convert to forward (0, ub), reverse (0, 0)
                    ex.lower_bound = 0
                    ex.upper_bound = ub  # Use forward capacity
                else:
                    # Bidirectional exchange: convert to forward (0, ub), can also use reverse
                    ex.lower_bound = 0
                    ex.upper_bound = max(ub, -lb)  # Take the larger capacity

    # We leave the maintenance on here
    # turn off maintenance reactions by setting the lower bound to 0 (turning off maintenance can make results easier to interpret)
    # model.reactions.get_by_id("ATPM").lower_bound = 0
    return model


def set_carbon_source_safely(model, carbon_ex_idx, uptake_rate=-10):
    """
    Safely set carbon source uptake rate while respecting irreversible constraints.
    Handles both models with and without _reverse exchanges.

    Parameters
    ----------
    model : cobra.Model
        The model to modify
    carbon_ex_idx : int
        Index of the carbon exchange reaction in model.exchanges
    uptake_rate : float
        Desired uptake rate (should be negative for uptake)

    Returns
    -------
    bool
        True if carbon source was set successfully, False otherwise
    """
    if carbon_ex_idx == -1:
        return False

    ex_rxn = model.exchanges[carbon_ex_idx]
    has_reverse = _is_model_with_reverse_exchanges(model)

    if uptake_rate < 0:  # Uptake (negative flux)
        if has_reverse:
            # For models with _reverse exchanges: set the reverse reaction to allow uptake
            try:
                reverse_rxn = model.reactions.get_by_id(ex_rxn.id + '_reverse')
                # Set reverse reaction to allow the uptake (convert negative to positive)
                reverse_rxn.upper_bound = abs(uptake_rate)
                return True
            except KeyError:
                # No reverse reaction found - exchange is export-only, cannot support uptake
                print(f"Warning: Exchange {ex_rxn.id} cannot handle uptake rate {uptake_rate} (bounds: [{ex_rxn.lower_bound}, {ex_rxn.upper_bound}])")
                return False
        else:
            # Original logic for bidirectional bounds
            if ex_rxn.lower_bound <= uptake_rate:
                ex_rxn.lower_bound = uptake_rate
                return True
            else:
                print(f"Warning: Exchange {ex_rxn.id} cannot handle uptake rate {uptake_rate} (bounds: [{ex_rxn.lower_bound}, {ex_rxn.upper_bound}])")
                return False
    else:  # Export (positive flux)
        if has_reverse:
            # For models with _reverse exchanges: set the forward reaction to allow export
            ex_rxn.upper_bound = uptake_rate
            return True
        else:
            # Original logic for bidirectional bounds
            if ex_rxn.upper_bound >= uptake_rate:
                ex_rxn.upper_bound = uptake_rate
                return True
            else:
                print(f"Warning: Exchange {ex_rxn.id} cannot handle export rate {uptake_rate} (bounds: [{ex_rxn.lower_bound}, {ex_rxn.upper_bound}])")
                return False

def set_medium_safely(model, medium_ex_inds, uptake_rate=-1000):
    """
    Safely set medium exchange reactions for irreversible models.

    For irreversible models, this handles both uptake-only exchanges and
    split reversible exchanges properly.

    Parameters
    ----------
    model : cobra.Model
        The model containing exchange reactions
    medium_ex_inds : list
        Indices of medium exchange reactions
    uptake_rate : float
        Uptake rate to set (negative value)
    """
    has_reverse = _is_model_with_reverse_exchanges(model)

    for e in medium_ex_inds:
        if e != -1:
            exchange_rxn = model.exchanges[e]

            if has_reverse:
                # For models with _reverse exchanges: check directionality and set appropriately
                directionality = _get_exchange_directionality(model, exchange_rxn.id)

                if directionality == 'uptake_only':
                    # Set reverse reaction to allow uptake
                    try:
                        reverse_rxn = model.reactions.get_by_id(exchange_rxn.id + '_reverse')
                        reverse_rxn.upper_bound = abs(uptake_rate)
                    except KeyError:
                        pass
                elif directionality == 'export_only':
                    # Set forward reaction to export capacity
                    exchange_rxn.upper_bound = abs(uptake_rate)
                else:  # bidirectional
                    # Allow both directions with wide bounds
                    exchange_rxn.upper_bound = abs(uptake_rate)
                    try:
                        reverse_rxn = model.reactions.get_by_id(exchange_rxn.id + '_reverse')
                        reverse_rxn.upper_bound = abs(uptake_rate)
                    except KeyError:
                        pass
            else:
                # Original logic for models without _reverse exchanges
                exchange_rxn.lower_bound = uptake_rate

def reset_carbon_source_safely(model, carbon_ex_idx):
    """
    Reset carbon source to default bounds while preserving irreversible structure.
    Handles both models with and without _reverse exchanges.

    Parameters
    ----------
    model : cobra.Model
        The model to modify
    carbon_ex_idx : int
        Index of the carbon exchange reaction in model.exchanges
    """
    if carbon_ex_idx == -1:
        return

    ex_rxn = model.exchanges[carbon_ex_idx]
    has_reverse = _is_model_with_reverse_exchanges(model)

    if has_reverse:
        # For models with _reverse exchanges, reset based on directionality
        directionality = _get_exchange_directionality(model, ex_rxn.id)

        if directionality == 'uptake_only':
            # Uptake: forward (0, 0), reverse (0, 1000)
            ex_rxn.lower_bound = 0
            ex_rxn.upper_bound = 0
            try:
                reverse_ex = model.reactions.get_by_id(ex_rxn.id + '_reverse')
                reverse_ex.lower_bound = 0
                reverse_ex.upper_bound = 1000
            except KeyError:
                pass
        elif directionality == 'export_only':
            # Export: forward (0, 1000), reverse (0, 0)
            ex_rxn.lower_bound = 0
            ex_rxn.upper_bound = 1000
            try:
                reverse_ex = model.reactions.get_by_id(ex_rxn.id + '_reverse')
                reverse_ex.lower_bound = 0
                reverse_ex.upper_bound = 0
            except KeyError:
                pass
        else:  # bidirectional
            # Bidirectional: forward (0, 1000), reverse (0, 1000)
            ex_rxn.lower_bound = 0
            ex_rxn.upper_bound = 1000
            try:
                reverse_ex = model.reactions.get_by_id(ex_rxn.id + '_reverse')
                reverse_ex.lower_bound = 0
                reverse_ex.upper_bound = 1000
            except KeyError:
                pass
    else:
        # Original logic for bidirectional bounds
        if ex_rxn.lower_bound < 0 and ex_rxn.upper_bound <= 0:
            # This was uptake-only, keep it that way
            ex_rxn.lower_bound = -1000
            ex_rxn.upper_bound = 0
        elif ex_rxn.lower_bound >= 0 and ex_rxn.upper_bound > 0:
            # This was export-only, keep it that way
            ex_rxn.lower_bound = 0
            ex_rxn.upper_bound = 1000
        else:
            # This was bidirectional, keep it that way
            ex_rxn.lower_bound = -1000
            ex_rxn.upper_bound = 1000


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
        # Temporarily set all exchanges to wide bounds for essential gene testing
        # Using irreversible format: forward (0, UB_forward), reverse (0, -LB_original)
        original_bounds = {}
        for ex in model_adj.exchanges:
            original_bounds[ex.id] = (ex.lower_bound, ex.upper_bound)
            lb, ub = ex.lower_bound, ex.upper_bound

            # Convert to irreversible format:
            # forward reaction bounds: (0, max(0, UB))
            # reverse reaction bounds: (0, max(0, -LB))
            # For exchanges, we set the reaction bounds to allow wide uptake/export

            if lb < 0 and ub <= 0:
                # Uptake-only exchange: original is (-1000, 0)
                # Forward stays minimal: (0, 0)
                # Reverse allows large uptake: (0, 1000)
                ex.lower_bound = 0
                ex.upper_bound = max(0, -lb)  # Use reverse capacity
            elif lb >= 0 and ub > 0:
                # Export-only exchange: original is (0, 1000)
                # Forward allows large export: (0, 1000)
                # Reverse stays minimal: (0, 0)
                ex.lower_bound = 0
                ex.upper_bound = ub  # Use forward capacity
            else:
                # Bidirectional exchange: original is (-1000, 1000)
                # Forward allows large export: (0, 1000)
                # Reverse allows large uptake: (0, 1000)
                # We set to allow forward direction with max(0, ub)
                ex.lower_bound = 0
                ex.upper_bound = max(ub, -lb)  # Take the larger of forward/reverse capacity

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

        # Restore original bounds
        for ex in model_adj.exchanges:
            if ex.id in original_bounds:
                ex.lower_bound, ex.upper_bound = original_bounds[ex.id]

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
    Uses safe bounds handling for irreversible models.
    """
    results = {}
    with model_adj:
        # Set medium with safe bounds handling using irreversible format
        for e in medium_ex_inds:
            if e != -1:
                ex = model_adj.exchanges[e]
                # Allow uptake while preserving irreversible structure
                # Convert to forward (0, UB_forward), reverse (0, -LB_original)
                lb, ub = ex.lower_bound, ex.upper_bound
                if ub <= 0:  # Uptake-only
                    ex.lower_bound = 0
                    ex.upper_bound = max(0, -lb)
                else:  # Bidirectional or export
                    ex.lower_bound = 0
                    ex.upper_bound = max(ub, -lb if lb < 0 else 0)

        solution = model_adj.optimize()
        base_growth = 1 if solution.objective_value and solution.objective_value > thresh else 0
        print(f"Growth with no carbon source (base medium only): {'Growth' if base_growth == 1 else 'No growth'} (Objective value: {solution.objective_value})")
        results['base_growth'] = base_growth

        base_growth_C = np.ones(len(carbon_ex_inds), dtype=float) * -1
        for e, idx in enumerate(carbon_ex_inds):
            if idx != -1:
                # Use safe carbon source setting
                carbon_set = set_carbon_source_safely(model_adj, idx, -10)
                if not carbon_set:
                    print(f"Warning: Could not set carbon source {e}, skipping")
                    base_growth_C[e] = 0
                    continue

            sol = model_adj.slim_optimize()
            base_growth_C[e] = 1 if sol > thresh else 0
            carbon_name = name_carbon_model_matched_adj[e] if e < len(name_carbon_model_matched_adj) else f"Carbon {e}"
            print(f"Growth with carbon source '{carbon_name}': {'Growth' if base_growth_C[e] == 1 else 'No growth'} (Solution: {sol})")

            if idx != -1:
                # Reset carbon source safely
                reset_carbon_source_safely(model_adj, idx)

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
    # Set medium with safe bounds handling using irreversible format
    for e in medium_ex_inds:
        if e != -1:
            ex = model_run.exchanges[e]
            # Allow uptake while preserving irreversible structure
            # Convert to forward (0, UB_forward), reverse (0, -LB_original)
            lb, ub = ex.lower_bound, ex.upper_bound
            if ub <= 0:  # Uptake-only
                ex.lower_bound = 0
                ex.upper_bound = max(0, -lb)
            else:  # Bidirectional or export
                ex.lower_bound = 0
                ex.upper_bound = max(ub, -lb if lb < 0 else 0)

    for e in range(len(name_carbon_model_matched_adj)):
        print(f"Baseline GEM progress: {e+1}/{len(name_carbon_model_matched_adj)}", end='\r')
        if name_carbon_model_matched_adj[e] in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(name_carbon_model_matched_adj[e])
            baseline_GEM[:, e] = baseline_GEM[:, e_found]
        else:
            for g in range(len(name_genes_matched_adj)):
                with model_run:
                    # Set carbon source INSIDE context so bounds persist through optimization
                    carbon_set = False
                    if carbon_ex_inds[e] != -1:
                        carbon_set = set_carbon_source_safely(model_run, carbon_ex_inds[e], -10)

                    # Knock out gene and optimize
                    model_run.genes.get_by_id(name_genes_matched_adj[g]).knock_out()
                    solution = model_run.slim_optimize()
                    if np.isnan(solution):
                        solution = 0
                    baseline_GEM[g, e] = solution

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
            for g in range(len(name_genes_matched_adj)):
                with model_run:
                    # Set carbon source INSIDE context so bounds persist through optimization
                    carbon_set = False
                    if carbon_ex_inds[e] != -1:
                        carbon_set = set_carbon_source_safely(model_run, carbon_ex_inds[e], -10)

                    # Knock out gene and optimize
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
    # Set medium with safe bounds handling using irreversible format
    for e in medium_ex_inds:
        if e != -1:
            ex = model_run.exchanges[e]
            # Allow uptake while preserving irreversible structure
            # Convert to forward (0, UB_forward), reverse (0, -LB_original)
            lb, ub = ex.lower_bound, ex.upper_bound
            if ub <= 0:  # Uptake-only
                ex.lower_bound = 0
                ex.upper_bound = max(0, -lb)
            else:  # Bidirectional or export
                ex.lower_bound = 0
                ex.upper_bound = max(ub, -lb if lb < 0 else 0)

    for e in range(n_carbons):
        print(f"Baseline GEM flux progress: {e+1}/{n_carbons}", end='\r')
        if name_carbon_model_matched_adj[e] in name_carbon_model_matched_adj[:e]:
            e_found = name_carbon_model_matched_adj[:e].index(name_carbon_model_matched_adj[e])
            baseline_fluxes[:, e, :] = baseline_fluxes[:, e_found, :]
        else:
            # Use safe carbon source setting
            carbon_set = False
            if carbon_ex_inds[e] != -1:
                carbon_set = set_carbon_source_safely(model_run, carbon_ex_inds[e], -10)

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
            # Use safe carbon source setting
            carbon_set = False
            if carbon_ex_inds[e] != -1:
                carbon_set = set_carbon_source_safely(model_run, carbon_ex_inds[e], -10)

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

            # Reset carbon source safely
            if carbon_ex_inds[e] != -1 and carbon_set:
                reset_carbon_source_safely(model_run, carbon_ex_inds[e])

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
    verbose=False
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

def _simulate_gene_carbon(model, processed_df, gene, carbon_idx, carbon_name,
                                medium_ex_inds, carbon_ex_inds, objective_reaction,
                                enzyme_upper_bound, mode='baseline', solver_name='glpk',
                                remove_knockout_enzyme=True):
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
    solver_name : str, optional
        Solver to use for optimization (default: 'glpk')
    remove_knockout_enzyme : bool, optional
        If True (default), remove enzyme constraints for the knocked-out gene,
        freeing protein pool capacity. If False, keep enzyme constraints for
        the knocked-out gene, which may better model immediate knockout effects
        without proteome reallocation.

    Returns
    -------
    tuple
        (gene, carbon_idx, growth_value, improvement_info)
        where improvement_info is a dict with growth improvement details or None
    """
    from kinGEMs.modeling.optimize import run_optimization_with_dataframe

    improvement_info = None  # Will store growth improvement details if detected

    print(f"  🚀 Starting {mode} simulation: Gene {gene} × Carbon {carbon_name} (idx {carbon_idx})")

    # Make a copy of the model so we can modify bounds without affecting the original
    # that's used for other parallel simulations
    model = model.copy()

    # Apply medium bounds directly to the model (instead of passing to run_optimization)
    # For irreversible models, _reverse reactions need upper_bound set (not lower_bound)

    # Set base medium components
    for e in medium_ex_inds:
        if e != -1:
            ex = model.exchanges[e]
            reverse_id = ex.id + '_reverse'

            # Check if this exchange has a _reverse version (uptake-capable)
            if reverse_id in [r.id for r in model.reactions]:
                # Set _reverse reaction upper_bound for uptake
                reverse_rxn = model.reactions.get_by_id(reverse_id)
                reverse_rxn.upper_bound = 1000
            elif ex.id.endswith('_reverse'):
                # This IS the _reverse reaction
                ex.upper_bound = 1000
            else:
                # Forward exchange only - set upper_bound for export
                ex.upper_bound = 1000

    # Set carbon source (uptake via _reverse reaction)
    if carbon_ex_inds[carbon_idx] != -1:
        carbon_ex = model.exchanges[carbon_ex_inds[carbon_idx]]
        carbon_reverse_id = carbon_ex.id + '_reverse' if not carbon_ex.id.endswith('_reverse') else carbon_ex.id
        try:
            carbon_reverse_rxn = model.reactions.get_by_id(carbon_reverse_id)
            carbon_reverse_rxn.upper_bound = 10  # Specific carbon uptake rate
            print(f"    - Medium configured with carbon source '{carbon_name}' (EX: {carbon_reverse_id}, UB=10)")
        except KeyError:
            print(f"    - Warning: Could not find carbon source exchange {carbon_reverse_id} for '{carbon_name}'")
    else:
        print(f"    - Warning: Could not find carbon source exchange for '{carbon_name}'")

    # Knockout gene
    try:
        gene_obj = model.genes.get_by_id(gene)
        print(f"    - Knocking out gene: {gene}")

        # Check wild-type growth before knockout
        if mode == 'baseline':
            try:
                with time_limit(300, "Wild-type baseline optimization"):
                    wild_type_growth = model.slim_optimize()
                if np.isnan(wild_type_growth):
                    wild_type_growth = 0.0
                print(f"    - Wild-type growth (before knockout): {wild_type_growth:.6f}")
            except TimeoutException as te:
                wild_type_growth = 0.0
                print(f"    - Wild-type growth (before knockout): {wild_type_growth:.6f} (timeout: {te})")
            except Exception:
                wild_type_growth = 0.0
                print(f"    - Wild-type growth (before knockout): {wild_type_growth:.6f} (failed)")
        else:  # enzyme-constrained
            try:
                with time_limit(600, "Wild-type enzyme-constrained optimization"):
                    wt_solution_value, _, _, _ = run_optimization_with_dataframe(
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
                        verbose=False,
                        solver_name=solver_name
                        # NOTE: Don't pass medium - bounds already set on model directly
                    )
                if wt_solution_value is None or np.isnan(wt_solution_value):
                    wt_solution_value = 0.0
                print(f"    - Wild-type growth (before knockout): {wt_solution_value:.6f}")
            except TimeoutException as te:
                wt_solution_value = 0.0
                print(f"    - Wild-type growth (before knockout): {wt_solution_value:.6f} (timeout: {te})")
            except Exception:
                wt_solution_value = 0.0
                print(f"    - Wild-type growth (before knockout): {wt_solution_value:.6f} (failed)")

        # Check gene status before knockout
        reactions_before = [r for r in gene_obj.reactions if r.id != objective_reaction]
        active_reactions_before = [r for r in reactions_before if r.bounds != (0, 0)]
        print(f"    - Gene {gene} is associated with {len(reactions_before)} reactions (excluding objective)")
        print(f"    - Before knockout: {len(active_reactions_before)} reactions are active")

        gene_obj.knock_out()

        # Check gene status after knockout
        active_reactions_after = [r for r in reactions_before if r.bounds != (0, 0)]
        knocked_out_reactions = [r for r in reactions_before if r.bounds == (0, 0)]

        print(f"    - After knockout: {len(active_reactions_after)} reactions still active")
        print(f"    ✅ Knockout verified: {len(knocked_out_reactions)} reactions set to bounds (0,0)")

        # List knocked-out reactions with their bounds
        if len(knocked_out_reactions) > 0:
            print(f"    📋 Knocked-out reactions ({len(knocked_out_reactions)}):")
            for rxn in knocked_out_reactions[:5]:  # Show first 5 to avoid excessive output
                print(f"       • {rxn.id}: bounds = {rxn.bounds}")
            if len(knocked_out_reactions) > 5:
                print(f"       ... and {len(knocked_out_reactions) - 5} more")

        # List reactions that remain active (potential GPR issues)
        if len(active_reactions_after) > 0:
            print(f"    ⚠️  Reactions still active after knockout ({len(active_reactions_after)}):")
            for rxn in active_reactions_after[:5]:  # Show first 5
                print(f"       • {rxn.id}: bounds = {rxn.bounds}, GPR = {rxn.gene_reaction_rule}")
            if len(active_reactions_after) > 5:
                print(f"       ... and {len(active_reactions_after) - 5} more")

        # Additional verification: check if knockout had any effect
        if len(knocked_out_reactions) == 0 and len(reactions_before) > 0:
            print("    ⚠️  Warning: Gene knockout may not have affected any reactions!")
        elif len(knocked_out_reactions) > 0:
            print(f"    ✓ Gene knockout successfully affected {len(knocked_out_reactions)} reaction(s)")

    except Exception as ex:
        print(f"    ❌ Gene knockout failed for {gene}: {ex}")
        return (gene, carbon_idx, 0.0)

    # Simulate growth
    if mode == 'baseline':
        # Medium and carbon source bounds already set on model above (lines 960-989)
        # Just proceed with optimization
        print(f"    - Running baseline GEM simulation for gene {gene}, carbon {carbon_idx}")
        try:
            with time_limit(300, f"Baseline optimization for {gene}"):
                solution = model.slim_optimize()
            if np.isnan(solution):
                solution = 0.0
            print(f"    - Baseline result: {solution:.6f}")

            # Calculate growth reduction/improvement if we have wild-type growth
            try:
                if 'wild_type_growth' in locals() and wild_type_growth > 0:
                    change = ((solution - wild_type_growth) / wild_type_growth) * 100

                    # Check for growth IMPROVEMENT (>1% increase)
                    if change > 1.0:
                        print(f"    🌟 GROWTH IMPROVEMENT DETECTED: {change:.2f}% increase!")
                        print(f"       Wild-type: {wild_type_growth:.6f} → Knockout: {solution:.6f}")

                        # Get associated reactions for this gene
                        gene_obj = model.genes.get_by_id(gene)
                        associated_reactions = [r.id for r in gene_obj.reactions if r.id != objective_reaction]

                        improvement_info = {
                            'gene': gene,
                            'carbon_source': carbon_name,
                            'carbon_idx': carbon_idx,
                            'wild_type_growth': wild_type_growth,
                            'knockout_growth': solution,
                            'percent_increase': change,
                            'associated_reactions': associated_reactions,
                            'mode': mode
                        }
                    elif change < 0:
                        print(f"    - Growth reduction: {-change:.1f}% (from {wild_type_growth:.6f} to {solution:.6f})")
                    else:
                        print(f"    - Growth change: {change:.1f}% (from {wild_type_growth:.6f} to {solution:.6f})")
            except Exception as e:
                print(f"    - Error calculating growth change: {e}")

            print(f"  ✅ Completed {mode} simulation: Gene {gene} × Carbon {carbon_name} → {solution:.6f}")
            return (gene, carbon_idx, solution, improvement_info)
        except TimeoutException as te:
            print(f"    ⏱️ Baseline simulation timed out: {te}")
            return (gene, carbon_idx, 0.0, None)
        except Exception as e:
            print(f"    ❌ Baseline simulation failed: {str(e)[:100]}")
            return (gene, carbon_idx, 0.0, None)

    else:  # enzyme-constrained
        print(f"    🔬 Running enzyme-constrained GEM simulation for gene {gene}, carbon {carbon_idx}")
        try:
            # Control whether enzyme constraints are removed for the knocked-out gene
            # If remove_knockout_enzyme=True (default): Remove enzyme constraints for the KO gene,
            #   which frees protein pool capacity and can result in slight growth improvements
            # If remove_knockout_enzyme=False: Keep enzyme constraints even for KO gene,
            #   which models immediate knockout effects without proteome reallocation
            if remove_knockout_enzyme:
                processed_df_knockout = processed_df[processed_df['Single_gene'] != gene].copy()
                # Log the filtering for verification
                n_original = len(processed_df)
                n_filtered = len(processed_df_knockout)
                if n_original != n_filtered:
                    print(f"    - Filtered enzyme data: removed gene {gene} ({n_original} → {n_filtered} entries)")
            else:
                processed_df_knockout = processed_df  # Keep all enzyme constraints
                print(f"    - Keeping enzyme constraints for knocked-out gene {gene}")

            # Try with the provided solver first
            with time_limit(600, f"Enzyme-constrained optimization for {gene}"):
                solution_value, _, _, _ = run_optimization_with_dataframe(
                    model=model,
                    processed_df=processed_df_knockout,  # Use (possibly filtered) dataframe
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
                    verbose=False,
                    solver_name=solver_name,
                    # Medium bounds already applied directly to model above
                )
            if solution_value is None or np.isnan(solution_value):
                solution_value = 0.0
            print(f"    - Enzyme-constrained result: {solution_value:.6f}")

            # Calculate growth reduction/improvement if we have wild-type growth
            try:
                if 'wt_solution_value' in locals() and wt_solution_value > 0:
                    change = ((solution_value - wt_solution_value) / wt_solution_value) * 100

                    # Check for growth IMPROVEMENT (>1% increase)
                    if change > 1.0:
                        print(f"    🌟 GROWTH IMPROVEMENT DETECTED: {change:.2f}% increase!")
                        print(f"       Wild-type: {wt_solution_value:.6f} → Knockout: {solution_value:.6f}")

                        # Get associated reactions for this gene
                        gene_obj = model.genes.get_by_id(gene)
                        associated_reactions = [r.id for r in gene_obj.reactions if r.id != objective_reaction]

                        improvement_info = {
                            'gene': gene,
                            'carbon_source': carbon_name,
                            'carbon_idx': carbon_idx,
                            'wild_type_growth': wt_solution_value,
                            'knockout_growth': solution_value,
                            'percent_increase': change,
                            'associated_reactions': associated_reactions,
                            'mode': mode
                        }
                    elif change < 0:
                        print(f"    - Growth reduction: {-change:.1f}% (from {wt_solution_value:.6f} to {solution_value:.6f})")
                    else:
                        print(f"    - Growth change: {change:.1f}% (from {wt_solution_value:.6f} to {solution_value:.6f})")
            except Exception as e:
                print(f"    - Error calculating growth change: {e}")

            print(f"  ✅ Completed {mode} simulation: Gene {gene} × Carbon {carbon_name} → {solution_value:.6f}")
            return (gene, carbon_idx, solution_value, improvement_info)
        except TimeoutException as te:
            print(f"    ⏱️ Enzyme-constrained simulation timed out with {solver_name}: {te}")

            # Try with a different solver if GLPK times out
            if solver_name.lower() == 'glpk':
                print("    🔄 Retrying with CPLEX solver (shorter timeout)...")
                try:
                    with time_limit(300, f"CPLEX retry for {gene}"):
                        solution_value, _, _, _ = run_optimization_with_dataframe(
                            model=model,
                            processed_df=processed_df_knockout,  # Use filtered dataframe
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
                            verbose=False,
                            solver_name='cplex',
                            # Medium bounds already applied directly to model above
                        )
                    if solution_value is None or np.isnan(solution_value):
                        solution_value = 0.0
                    print(f"    - Enzyme-constrained result (CPLEX): {solution_value:.6f}")

                    # Calculate growth improvement for CPLEX success
                    try:
                        if 'wt_solution_value' in locals() and wt_solution_value > 0:
                            change = ((solution_value - wt_solution_value) / wt_solution_value) * 100
                            if change > 1.0:
                                print(f"    🌟 GROWTH IMPROVEMENT DETECTED: {change:.2f}% increase!")
                                print(f"       Wild-type: {wt_solution_value:.6f} → Knockout: {solution_value:.6f}")
                                gene_obj = model.genes.get_by_id(gene)
                                associated_reactions = [r.id for r in gene_obj.reactions if r.id != objective_reaction]
                                improvement_info = {
                                    'gene': gene,
                                    'carbon_source': carbon_name,
                                    'carbon_idx': carbon_idx,
                                    'wild_type_growth': wt_solution_value,
                                    'knockout_growth': solution_value,
                                    'percent_increase': change,
                                    'associated_reactions': associated_reactions,
                                    'mode': mode
                                }
                    except Exception as e:
                        print(f"    - Error calculating growth change: {e}")

                    print(f"  ✅ Completed {mode} simulation: Gene {gene} × Carbon {carbon_name} → {solution_value:.6f}")
                    return (gene, carbon_idx, solution_value, improvement_info)
                except TimeoutException as te2:
                    print(f"    ⏱️ CPLEX also timed out: {te2}")
                except Exception as e2:
                    print(f"    ❌ CPLEX also failed: {str(e2)[:100]}...")

            print("    ⚠️ Returning 0.0 for timed-out simulation")
            return (gene, carbon_idx, 0.0, None)
        except Exception as e:
            print(f"    ❌ Enzyme-constrained simulation failed with {solver_name}: {str(e)[:100]}...")

            # Try with a different solver if GLPK fails
            if solver_name.lower() == 'glpk':
                print("    🔄 Retrying with CPLEX solver...")
                try:
                    with time_limit(600, f"CPLEX retry for {gene}"):
                        solution_value, _, _, _ = run_optimization_with_dataframe(
                        model=model,
                        processed_df=processed_df_knockout,  # Use filtered dataframe
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
                        verbose=False,
                        solver_name='cplex',
                        medium=medium,  # Pass medium dictionary
                        medium_upper_bound=True  # Use upper bound convention for _reverse reactions
                    )
                    if solution_value is None or np.isnan(solution_value):
                        solution_value = 0.0
                    print(f"    - Enzyme-constrained result (CPLEX): {solution_value:.6f}")

                    # Calculate growth reduction/improvement if we have wild-type growth
                    try:
                        if 'wt_solution_value' in locals() and wt_solution_value > 0:
                            change = ((solution_value - wt_solution_value) / wt_solution_value) * 100

                            # Check for growth IMPROVEMENT (>1% increase)
                            if change > 1.0:
                                print(f"    🌟 GROWTH IMPROVEMENT DETECTED: {change:.2f}% increase!")
                                print(f"       Wild-type: {wt_solution_value:.6f} → Knockout: {solution_value:.6f}")

                                # Get associated reactions for this gene
                                gene_obj = model.genes.get_by_id(gene)
                                associated_reactions = [r.id for r in gene_obj.reactions if r.id != objective_reaction]

                                improvement_info = {
                                    'gene': gene,
                                    'carbon_source': carbon_name,
                                    'carbon_idx': carbon_idx,
                                    'wild_type_growth': wt_solution_value,
                                    'knockout_growth': solution_value,
                                    'percent_increase': change,
                                    'associated_reactions': associated_reactions,
                                    'mode': mode
                                }
                            elif change < 0:
                                print(f"    - Growth reduction: {-change:.1f}% (from {wt_solution_value:.6f} to {solution_value:.6f})")
                            else:
                                print(f"    - Growth change: {change:.1f}% (from {wt_solution_value:.6f} to {solution_value:.6f})")
                    except Exception as e:
                        print(f"    - Error calculating growth change: {e}")

                    print(f"  ✅ Completed {mode} simulation: Gene {gene} × Carbon {carbon_name} → {solution_value:.6f}")
                    return (gene, carbon_idx, solution_value, improvement_info)
                except TimeoutException as te2:
                    print(f"    ⏱️ CPLEX also timed out: {te2}")
                except Exception as e2:
                    print(f"    ❌ CPLEX also failed: {str(e2)[:100]}...")

            print("    ⚠️ Returning 0.0 for failed simulation")
            return (gene, carbon_idx, 0.0, None)


def _simulate_gene_carbon_chunk(model, processed_df, tasks, medium_ex_inds,
                                carbon_ex_inds, objective_reaction,
                                enzyme_upper_bound, mode='baseline', solver_name='glpk',
                                remove_knockout_enzyme=True):
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
    solver_name : str, optional
        Solver to use (default: 'glpk')
    remove_knockout_enzyme : bool, optional
        If True (default), remove enzyme constraints for knocked-out genes.
        If False, keep enzyme constraints even for knocked-out genes.

    Returns
    -------
    tuple
        (results, improvements) where:
        - results: list of tuples (gene, carbon_idx, growth_value)
        - improvements: list of improvement_info dicts (or empty list if none)
    """
    results = []
    improvements = []

    for gene, carbon_idx, carbon_name in tasks:
        gene_result, carbon_idx_result, growth_value, improvement_info = _simulate_gene_carbon(
            model=model.copy(),
            processed_df=processed_df,
            gene=gene,
            carbon_idx=carbon_idx,
            carbon_name=carbon_name,
            medium_ex_inds=medium_ex_inds,
            carbon_ex_inds=carbon_ex_inds,
            objective_reaction=objective_reaction,
            enzyme_upper_bound=enzyme_upper_bound,
            mode=mode,
            solver_name=solver_name,
            remove_knockout_enzyme=remove_knockout_enzyme
        )
        results.append((gene_result, carbon_idx_result, growth_value))

        if improvement_info is not None:
            improvements.append(improvement_info)

    return (results, improvements)


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
    method='dask',
    skip_baseline=False,
    solver_name='glpk',
    remove_knockout_enzyme=True
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
    skip_baseline : bool, optional
        If True, skip baseline simulation and only run enzyme-constrained (default: False)
    remove_knockout_enzyme : bool, optional
        If True (default), remove enzyme constraints for knocked-out genes,
        freeing protein pool capacity. If False, keep enzyme constraints even
        for knocked-out genes (use --keep-enzyme-constraints CLI flag).

    Returns
    -------
    tuple
        (baseline_GEM, enzyme_constrained_GEM) as numpy arrays
    """
    import os

    # Determine number of workers
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    n_genes = len(name_genes_matched_adj)
    n_carbons = len(name_carbon_model_matched_adj)
    total_tasks = n_genes * n_carbons

    # Calculate optimal chunk size
    if chunk_size is None:
        # For validation, use moderate chunks for better performance
        # Aim for ~20-50 chunks per worker
        chunk_size = max(1, total_tasks // (n_workers * 30))

    print("\n  Parallel validation configuration:")
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
    baseline_GEM = None

    if skip_baseline:
        print("\n  ⏭️  Skipping baseline GEM simulation (already completed)")
        baseline_GEM = np.zeros((n_genes, n_carbons), dtype=float)
        baseline_improvements = []  # No improvements to track when skipping
    else:
        print("\n  Starting parallel baseline GEM simulation...")

        if method.lower() == 'multiprocessing':
            baseline_results, baseline_improvements = _run_validation_multiprocessing(
                model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
                objective_reaction, enzyme_upper_bound, n_workers, mode='baseline', solver_name=solver_name,
                remove_knockout_enzyme=remove_knockout_enzyme
            )
        else:  # dask
            try:
                baseline_results = _run_validation_dask(
                    model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
                    objective_reaction, enzyme_upper_bound, n_workers, mode='baseline', solver_name=solver_name,
                    remove_knockout_enzyme=remove_knockout_enzyme
                )
                baseline_improvements = []  # Dask doesn't return improvements yet
            except Exception:
                print("    ⚠️  Dask failed, switching to multiprocessing")
                baseline_results, baseline_improvements = _run_validation_multiprocessing(
                    model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
                    objective_reaction, enzyme_upper_bound, n_workers, mode='baseline', solver_name=solver_name,
                    remove_knockout_enzyme=remove_knockout_enzyme
                )

        # Flatten results
        flat_baseline = []
        for chunk_results in baseline_results:
            flat_baseline.extend(chunk_results)

        # Build baseline matrix
        baseline_GEM = np.zeros((n_genes, n_carbons), dtype=float)
        # Convert to list if numpy array for indexing
        genes_list = list(name_genes_matched_adj) if isinstance(name_genes_matched_adj, np.ndarray) else name_genes_matched_adj
        for gene, carbon_idx, growth_value in flat_baseline:
            gene_idx = genes_list.index(gene)
            baseline_GEM[gene_idx, carbon_idx] = growth_value

        # Fill in cached carbon sources
        for cached_idx, original_idx in carbon_cache_map.items():
            baseline_GEM[:, cached_idx] = baseline_GEM[:, original_idx]

        print("  Baseline GEM simulation complete.")

    # ===== Enzyme-constrained GEM Simulation (Parallel) =====
    print("\n  Starting parallel enzyme-constrained GEM simulation...")

    # Clean processed_df
    if 'kcat_mean' in processed_df.columns:
        processed_df['kcat_mean'] = processed_df['kcat_mean'].apply(
            lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else x
        )

    # Get genes_list for indexing (need it for enzyme section too)
    genes_list = list(name_genes_matched_adj) if isinstance(name_genes_matched_adj, np.ndarray) else name_genes_matched_adj

    if method.lower() == 'multiprocessing':
        enzyme_results, enzyme_improvements = _run_validation_multiprocessing(
            model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
            objective_reaction, enzyme_upper_bound, n_workers, mode='enzyme', solver_name=solver_name,
            remove_knockout_enzyme=remove_knockout_enzyme
        )
    else:  # dask
        try:
            enzyme_results = _run_validation_dask(
                model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
                objective_reaction, enzyme_upper_bound, n_workers, mode='enzyme', solver_name=solver_name,
                remove_knockout_enzyme=remove_knockout_enzyme
            )
            enzyme_improvements = []  # Dask doesn't return improvements yet
        except Exception:
            print("    ⚠️  Dask failed, switching to multiprocessing")
            enzyme_results, enzyme_improvements = _run_validation_multiprocessing(
                model_run, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
                objective_reaction, enzyme_upper_bound, n_workers, mode='enzyme', solver_name=solver_name,
                remove_knockout_enzyme=remove_knockout_enzyme
            )

    # Flatten results
    flat_enzyme = []
    for chunk_results in enzyme_results:
        flat_enzyme.extend(chunk_results)

    # Build enzyme-constrained matrix
    enzyme_constrained_GEM = np.zeros((n_genes, n_carbons), dtype=float)
    # Use the same genes_list from baseline section
    for gene, carbon_idx, growth_value in flat_enzyme:
        gene_idx = genes_list.index(gene)
        enzyme_constrained_GEM[gene_idx, carbon_idx] = growth_value

    # Fill in cached carbon sources
    for cached_idx, original_idx in carbon_cache_map.items():
        enzyme_constrained_GEM[:, cached_idx] = enzyme_constrained_GEM[:, original_idx]

    print("  Enzyme-constrained GEM simulation complete.")

    # ===== Save and Report Growth Improvements =====
    all_improvements = []
    if not skip_baseline:
        all_improvements.extend(baseline_improvements)
    all_improvements.extend(enzyme_improvements)

    if all_improvements:
        print(f"\n  {'='*80}")
        print(f"  🌟 GROWTH IMPROVEMENTS DETECTED: {len(all_improvements)} cases")
        print(f"  {'='*80}")

        # Save improvements to CSV
        import pandas as pd
        improvements_df = pd.DataFrame(all_improvements)

        # Sort by percent increase (descending)
        improvements_df = improvements_df.sort_values('percent_increase', ascending=False)

        # Create output filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"growth_improvements_{timestamp}.csv"
        improvements_df.to_csv(output_file, index=False)

        print(f"  ✅ Saved growth improvements to: {output_file}")
        print(f"\n  Top 10 growth improvements:")
        print(f"  {'-'*80}")

        # Display top 10
        for idx, row in improvements_df.head(10).iterrows():
            print(f"  {idx+1}. Gene: {row['gene']}, Carbon: {row['carbon_source']}")
            print(f"     WT: {row['wild_type_growth']:.6f} → KO: {row['knockout_growth']:.6f} (+{row['percent_increase']:.2f}%)")
            print(f"     Reactions: {', '.join(row['associated_reactions'][:5])}{' ...' if len(row['associated_reactions']) > 5 else ''}")
            print(f"     Mode: {row['mode']}")
            print()

        print(f"  {'-'*80}")
        print(f"  See full list in: {output_file}")
        print(f"  {'='*80}\n")
    else:
        print("\n  ℹ️  No growth improvements detected (>1% increase)")

    print("\n  Parallel validation simulations finished!")

    return baseline_GEM, enzyme_constrained_GEM


def _run_validation_dask(model, processed_df, chunks, medium_ex_inds, carbon_ex_inds,
                        objective_reaction, enzyme_upper_bound, n_workers, mode='baseline', solver_name='glpk',
                        remove_knockout_enzyme=True):
    """Execute validation using Dask with HPC-friendly configuration."""
    import logging
    import tempfile

    try:
        from dask import delayed
        from dask.distributed import Client, LocalCluster
    except ImportError:
        raise ImportError(
            "Dask is required for parallel validation. Install with: pip install dask[distributed]"
        )

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
                mode,
                solver_name,
                remove_knockout_enzyme
            )
        )

    # Configure Dask for slow tasks and HPC environment
    import dask
    dask.config.set({
        'distributed.comm.timeouts.connect': '120s',
        'distributed.comm.timeouts.tcp': '120s',
        'distributed.scheduler.idle-timeout': '7200s',
        'distributed.worker.profile.interval': '10000ms',
        'distributed.worker.profile.cycle': '100000ms',
        'distributed.scheduler.allowed-failures': 10,  # Allow some task retries
        'distributed.worker.daemon': False,  # Easier cleanup on HPC
    })

    cluster = None
    client = None

    try:
        # Use SLURM_TMPDIR if available (Compute Canada standard), otherwise temp dir
        import os
        local_dir = os.environ.get('SLURM_TMPDIR', tempfile.gettempdir())

        # Create cluster explicitly with HPC-friendly settings
        cluster = LocalCluster(
            n_workers=n_workers,
            processes=True,
            threads_per_worker=1,
            silence_logs=logging.ERROR,
            local_directory=local_dir,
            death_timeout='30s',  # Kill workers faster if they hang
        )

                # Connect client to cluster
        client = Client(cluster, timeout='60s')

        print(f"    Dask cluster created with {n_workers} workers")
        try:
            print(f"    Dask dashboard: {client.dashboard_link}")
        except Exception:
            print("    Dask dashboard: (not available - install bokeh>=3.1.0)")

        # Execute tasks with progress tracking
        print(f"    Submitting {len(tasks)} chunks to Dask workers...")
        import sys
        sys.stdout.flush()  # Force print before compute
        from dask.distributed import progress
        futures = client.compute(tasks)

        # Wait for results with progress
        try:
            progress(futures)
        except Exception:
            # Progress bar not available, just wait
            pass

        results = client.gather(futures)

        print("    ✓ Results gathered successfully")

        return results

    except Exception as e:
        print(f"    ⚠️  Dask execution failed: {e}")
        print("    Falling back to sequential execution...")
        raise

    finally:
        # Aggressive cleanup - close client and cluster separately
        if client is not None:
            try:
                client.close(timeout=2)
            except Exception:
                pass

        if cluster is not None:
            try:
                cluster.close(timeout=5)
            except Exception:
                pass


def _run_validation_multiprocessing(model, processed_df, chunks, medium_ex_inds,
                                    carbon_ex_inds, objective_reaction,
                                    enzyme_upper_bound, n_workers, mode='baseline', solver_name='glpk',
                                    remove_knockout_enzyme=True):
    """Execute validation using multiprocessing.Pool with progress tracking."""
    from functools import partial
    from multiprocessing import Pool
    import time

    # Create partial function
    worker_func = partial(
        _simulate_gene_carbon_chunk,
        model.copy(),
        processed_df,
        medium_ex_inds=medium_ex_inds,
        carbon_ex_inds=carbon_ex_inds,
        objective_reaction=objective_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        mode=mode,
        solver_name=solver_name,
        remove_knockout_enzyme=remove_knockout_enzyme
    )

    # Execute in parallel with progress tracking
    total_chunks = len(chunks)
    total_tasks = sum(len(chunk) for chunk in chunks)
    results = []
    all_improvements = []

    start_time = time.time()
    chunk_times = []

    with Pool(processes=n_workers) as pool:
        # Use imap_unordered for progress tracking
        for i, chunk_result in enumerate(pool.imap_unordered(worker_func, chunks), 1):
            # Unpack chunk results and improvements
            chunk_data, chunk_improvements = chunk_result
            results.append(chunk_data)

            # Collect any growth improvements from this chunk
            if chunk_improvements:
                all_improvements.extend(chunk_improvements)
                print(f"    🌟 Found {len(chunk_improvements)} growth improvement(s) in this chunk!")

            # Track timing
            current_time = time.time()
            elapsed_time = current_time - start_time
            chunk_times.append(elapsed_time / i)  # Average time per chunk

            # Calculate completed tasks (approximate)
            completed_tasks = sum(len(chunks[j]) for j in range(min(i, len(chunks))))

            # Progress message
            percent = (i / total_chunks) * 100
            avg_chunk_time = sum(chunk_times[-10:]) / min(len(chunk_times), 10)  # Moving average

            # Format time
            if elapsed_time < 60:
                time_str = f"{elapsed_time:.1f}s"
            else:
                mins = int(elapsed_time / 60)
                secs = elapsed_time % 60
                time_str = f"{mins}m {secs:.0f}s"

            print(f"    Progress: {i}/{total_chunks} chunks ({completed_tasks}/{total_tasks} simulations, {percent:.1f}%)")
            print(f"    Chunk time: {avg_chunk_time:.1f}s, Total time: {time_str}")

    return (results, all_improvements)

def reset_carbon_source_safely(model, carbon_ex_idx):
    """
    Reset carbon source to default bounds while preserving irreversible structure.

    Parameters
    ----------
    model : cobra.Model
        The model to modify
    carbon_ex_idx : int
        Index of the carbon exchange reaction in model.exchanges
    """
    if carbon_ex_idx == -1:
        return

    ex_rxn = model.exchanges[carbon_ex_idx]

    # Reset to wide bounds while preserving directionality
    if ex_rxn.lower_bound < 0 and ex_rxn.upper_bound <= 0:
        # This was uptake-only, keep it that way
        ex_rxn.lower_bound = -1000
        ex_rxn.upper_bound = 0
    elif ex_rxn.lower_bound >= 0 and ex_rxn.upper_bound > 0:
        # This was export-only, keep it that way
        ex_rxn.lower_bound = 0
        ex_rxn.upper_bound = 1000
    else:
        # This was bidirectional, keep it that way
        ex_rxn.lower_bound = -1000
        ex_rxn.upper_bound = 1000

def diagnose_model_for_validation(model, verbose=True):
    """
    Diagnose if a model is suitable for validation and identify potential issues.

    Parameters
    ----------
    model : cobra.Model
        The model to diagnose
    verbose : bool
        Whether to print detailed diagnostic information

    Returns
    -------
    dict
        Diagnostic results including compatibility status and recommendations
    """
    results = {
        'compatible': True,
        'warnings': [],
        'recommendations': [],
        'exchange_analysis': {},
        'model_type': 'unknown'
    }

    if verbose:
        print("\n" + "="*60)
        print("kinGEMs Validation Compatibility Diagnostic")
        print("="*60)

    # Analyze exchange reactions
    uptake_only = 0
    export_only = 0
    bidirectional = 0
    problematic = []

    for ex in model.exchanges:
        lb, ub = ex.lower_bound, ex.upper_bound

        if lb < 0 and ub <= 0:
            uptake_only += 1
        elif lb >= 0 and ub > 0:
            export_only += 1
        elif lb < 0 and ub > 0:
            bidirectional += 1
        else:
            problematic.append(ex.id)

    results['exchange_analysis'] = {
        'total': len(model.exchanges),
        'uptake_only': uptake_only,
        'export_only': export_only,
        'bidirectional': bidirectional,
        'problematic': len(problematic)
    }

    # Determine model type
    if uptake_only > 0 or export_only > 0:
        results['model_type'] = 'irreversible_exchanges'
        if verbose:
            print(f"Model Type: Contains irreversible exchanges")
            print(f"  - Uptake-only exchanges: {uptake_only}")
            print(f"  - Export-only exchanges: {export_only}")
            print(f"  - Bidirectional exchanges: {bidirectional}")
    else:
        results['model_type'] = 'reversible_exchanges'
        if verbose:
            print(f"Model Type: All exchanges are bidirectional ({bidirectional})")

    # Check for problematic exchanges
    if problematic:
        results['compatible'] = False
        results['warnings'].append(f"Found {len(problematic)} exchanges with invalid bounds: {problematic[:5]}")
        if verbose:
            print(f"⚠️  Warning: {len(problematic)} exchanges have invalid bounds")

    # Analyze reactions
    reversible_rxns = sum(1 for rxn in model.reactions if rxn.reversibility)
    irreversible_rxns = len(model.reactions) - reversible_rxns

    if verbose:
        print(f"\nReaction Analysis:")
        print(f"  - Total reactions: {len(model.reactions)}")
        print(f"  - Reversible: {reversible_rxns}")
        print(f"  - Irreversible: {irreversible_rxns}")

    # Check for enzyme constraint compatibility
    has_kcats = 0
    for rxn in model.reactions:
        if hasattr(rxn, 'annotation') and 'kcat' in rxn.annotation:
            if rxn.annotation['kcat'] not in [None, '', 0, '0']:
                has_kcats += 1

    if verbose:
        print(f"\nEnzyme Constraint Readiness:")
        print(f"  - Reactions with kcat data: {has_kcats}/{len(model.reactions)} ({100*has_kcats/len(model.reactions):.1f}%)")

    if has_kcats == 0:
        results['warnings'].append("No kcat data found in model annotations")
        results['recommendations'].append("Process kcat data with kinGEMs pipeline before validation")

    # Provide recommendations
    if results['model_type'] == 'irreversible_exchanges':
        results['recommendations'].append("Use prepare_model(model, handle_irreversible=True) for proper bounds handling")
        results['recommendations'].append("Consider using the updated validation functions that respect irreversible constraints")

    if verbose:
        print(f"\nCompatibility Status: {'✅ Compatible' if results['compatible'] else '❌ Issues Found'}")

        if results['warnings']:
            print(f"\nWarnings:")
            for warning in results['warnings']:
                print(f"  ⚠️  {warning}")

        if results['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  💡 {rec}")

        print("="*60)

    return results

def validate_kinGEMs_model(model, processed_df, objective_reaction,
                          enzyme_upper_bound=0.125, handle_irreversible=True,
                          **kwargs):
    """
    Convenience function to run kinGEMs validation with proper irreversible model support.

    This function automatically handles:
    - Model preparation with irreversible constraint preservation
    - Safe bounds handling for exchange reactions
    - Compatibility checking and warnings

    Parameters
    ----------
    model : cobra.Model
        The kinGEMs model to validate
    processed_df : pandas.DataFrame
        Processed enzyme constraint data
    objective_reaction : str
        Biomass reaction ID
    enzyme_upper_bound : float
        Enzyme upper bound constraint (default: 0.125)
    handle_irreversible : bool
        Whether to preserve irreversible exchange constraints (default: True)
    **kwargs
        Additional arguments passed to validation functions

    Returns
    -------
    dict
        Validation results including compatibility status and recommendations

    Examples
    --------
    >>> # Basic validation
    >>> results = validate_kinGEMs_model(model, processed_df, 'BIOMASS_Ecoli_core')

    >>> # With parallel processing
    >>> results = validate_kinGEMs_model(
    ...     model, processed_df, 'BIOMASS_Ecoli_core',
    ...     use_parallel=True, n_workers=4
    ... )
    """
    # First diagnose the model for compatibility
    diagnosis = diagnose_model_for_validation(model, verbose=True)

    if not diagnosis['compatible']:
        print("❌ Model has compatibility issues. See warnings above.")
        return diagnosis

    # Prepare model with appropriate settings
    model_prepared = prepare_model(model.copy(), handle_irreversible=handle_irreversible)

    print(f"✅ Model prepared successfully with irreversible support: {handle_irreversible}")
    print(f"📊 Ready to run kinGEMs validation with {len(model.reactions)} reactions")

    # Add model preparation info to diagnosis
    diagnosis['model_prepared'] = True
    diagnosis['settings'] = {
        'handle_irreversible': handle_irreversible,
        'enzyme_upper_bound': enzyme_upper_bound,
        'objective_reaction': objective_reaction
    }

    return diagnosis

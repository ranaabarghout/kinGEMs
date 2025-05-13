"""
Optimization module for kinGEMs.

This module provides functions for enzyme-constrained flux balance analysis,
including core optimization functionality from the original KG03b module.
"""

import math
import os
import re

from Bio.SeqUtils import molecular_weight
import cobra as cb

# Troubleshooting infeasible optimization
# Add this code before running your optimization to diagnose the issue
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *  # noqa: F403
from pyomo.opt import SolverFactory

from ..config import ensure_dir_exists  # noqa: F401


def diagnose_infeasibility(model, processed_data, biomass_reaction):
    """
    Diagnose potential infeasibility issues before running enzyme-constrained optimization
    """
    print("=== Diagnosing Potential Infeasibility Issues ===\n")
    
    # Check 1: Basic FBA without enzyme constraints
    print("1. Testing basic FBA (without enzyme constraints)...")
    try:
        basic_solution = model.optimize()
        print(f"   Basic FBA status: {basic_solution.status}")
        print(f"   Biomass flux without constraints: {basic_solution.fluxes[biomass_reaction]}")
    except Exception as e:
        print(f"   ERROR: Basic FBA failed - {e}")
        print("   This indicates a problem with the model itself, not the enzyme constraints")
        return
    
    # Check 2: Analyze enzyme upper bound
    print("\n2. Analyzing enzyme fraction constraints...")
    enzyme_upper_bound = 0.125  # gP/gDCW
    
    # Estimate minimum enzyme requirement for current biomass flux
    genes_with_data = set()
    
    for _, row in processed_data.iterrows():
        if pd.notna(row['SEQ']) and pd.notna(row['kcat_mean']):
            genes_with_data.add(row['Single_gene'])
    
    print(f"   Number of genes with complete data: {len(genes_with_data)}")
    print(f"   Enzyme upper bound: {enzyme_upper_bound} gP/gDCW")
    
    # Check 3: Look for unrealistic kcat values
    print("\n3. Checking for unrealistic kcat values...")
    kcat_values = processed_data['kcat_mean'].dropna()
    
    print(f"   Mean kcat: {kcat_values.mean():.2f} 1/s")
    print(f"   Min kcat: {kcat_values.min():.2f} 1/s")
    print(f"   Max kcat: {kcat_values.max():.2f} 1/s")
    
    # Very low kcat values can cause infeasibility
    low_kcat_threshold = 0.1
    low_kcat_count = sum(kcat_values < low_kcat_threshold)
    if low_kcat_count > 0:
        print(f"   WARNING: {low_kcat_count} reactions have very low kcat values (< {low_kcat_threshold} 1/s)")
        print("   This might cause infeasibility due to excessive enzyme requirements")
    
    # Check 4: Identify essential reactions without enzyme data
    print("\n4. Checking for essential reactions without enzyme data...")
    essential_reactions = []
    
    # Test each reaction's importance by knocking it out
    for reaction in model.reactions:
        original_bounds = (reaction.lower_bound, reaction.upper_bound)
        reaction.bounds = (0, 0)  # Knockout
        
        try:
            ko_solution = model.optimize()
            if ko_solution.status == 'optimal' and ko_solution.objective_value < 0.01:
                essential_reactions.append(reaction.id)
        except:  # noqa: E722
            pass
        
        reaction.bounds = original_bounds  # Restore
    
    print(f"   Found {len(essential_reactions)} essential reactions")
    
    # Check if any essential reactions lack enzyme data
    reactions_with_data = set(processed_data['Reactions'].dropna())
    essential_without_data = [r for r in essential_reactions if r not in reactions_with_data]
    
    if essential_without_data:
        print(f"   WARNING: {len(essential_without_data)} essential reactions lack enzyme data:")
        for r in essential_without_data[:5]:  # Show first 5
            print(f"      - {r}")
        if len(essential_without_data) > 5:
            print(f"      ... and {len(essential_without_data) - 5} more")
    
    # Check 5: Analyze biomass reaction components
    print("\n5. Analyzing biomass reaction...")
    biomass_rxn = model.reactions.get_by_id(biomass_reaction)
    biomass_metabolites = [m.id for m in biomass_rxn.metabolites]
    
    print(f"   Biomass reaction has {len(biomass_metabolites)} metabolites")
    
    # Check production pathways for biomass components
    blocked_metabolites = []
    for met_id in biomass_metabolites:
        if biomass_rxn.get_coefficient(met_id) < 0:  # Reactant (consumed)
            # Temporarily require this metabolite
            temp_rxn = model.add_boundary(model.metabolites.get_by_id(met_id), 
                                        type='demand', ub=0)
            temp_rxn.lower_bound = -0.1
            
            try:
                temp_solution = model.optimize()
                if temp_solution.status != 'optimal':
                    blocked_metabolites.append(met_id)
            except:  # noqa: E722
                blocked_metabolites.append(met_id)
            finally:
                model.remove_reactions([temp_rxn])
    
    if blocked_metabolites:
        print(f"   WARNING: {len(blocked_metabolites)} biomass components cannot be produced:")
        for met in blocked_metabolites[:3]:
            print(f"      - {met}")
    
    print("\n=== Recommendations ===")
    
    if basic_solution.status != 'optimal':
        print("1. Fix the basic model issues first")
    elif essential_without_data:
        print("1. Add enzyme data for essential reactions or exclude them from constraints")
    elif low_kcat_count > 0:
        print("1. Consider filtering out reactions with very low kcat values")
        print("2. Or increase the enzyme upper bound")
    elif blocked_metabolites:
        print("1. Check the model for blocked reactions in biomass precursor pathways")
    else:
        print("1. Try increasing the enzyme upper bound (e.g., 0.2 or 0.3 gP/gDCW)")
        print("2. Consider relaxing some enzyme constraints")
    
    print("\nTry these solutions in the following order:")
    print("1. Increase enzyme_upper_bound from 0.125 to 0.2 or higher")
    print("2. Filter out reactions with kcat < 0.1 s^-1")
    print("3. Run diagnostics on specific problematic reactions")
    
    return basic_solution

# Usage example:
# diagnose_infeasibility(irrev_model, processed_data, biomass_reaction)

def relaxed_optimization(model, processed_df, objective_reaction, 
                        initial_enzyme_bound=0.125, max_enzyme_bound=1.0, 
                        bound_increment=0.05):
    """
    Attempt optimization with progressively relaxed enzyme constraints
    """
    print("=== Attempting Relaxed Optimization ===\n")
    
    enzyme_bound = initial_enzyme_bound
    
    while enzyme_bound <= max_enzyme_bound:
        print(f"Trying enzyme upper bound: {enzyme_bound}")
        
        try:
            solution, flux_distribution, _, _ = run_optimization_with_dataframe(
                model=model,
                processed_df=processed_df,
                objective_reaction=objective_reaction,
                enzyme_upper_bound=enzyme_bound,
                enzyme_ratio=True,
                output_dir=None
            )
            
            if solution is not None:
                print(f"SUCCESS! Optimal solution found with enzyme bound: {enzyme_bound}")
                print(f"Biomass flux: {solution}")
                return solution, flux_distribution, enzyme_bound
                
        except Exception as e:
            print(f"Failed with error: {e}")
        
        enzyme_bound += bound_increment
    
    print(f"No feasible solution found up to enzyme bound: {max_enzyme_bound}")
    return None, None, None

# Usage example:
# solution, flux_distribution, optimal_bound = relaxed_optimization(
#     irrev_model, processed_data, biomass_reaction)

def simplified_optimization(model, processed_df, objective_reaction):
    """
    Run optimization with minimal enzyme constraints for debugging
    """
    print("=== Running Simplified Optimization ===\n")
    
    # Filter to only include reactions with complete data
    complete_data = processed_df.dropna(subset=['SEQ', 'kcat_mean'])
    print(f"Using {len(complete_data)} reactions with complete enzyme data")
    
    # First, try with a very high enzyme bound (effectively no constraint)
    try:
        solution, flux_distribution, _, _ = run_optimization_with_dataframe(
            model=model,
            processed_df=complete_data,
            objective_reaction=objective_reaction,
            enzyme_upper_bound=10.0,  # Very high bound
            enzyme_ratio=True,
            # Disable all additional constraints for debugging
            multi_enzyme_off=True,
            isoenzymes_off=True,
            promiscuous_off=True,
            complexes_off=True,
            output_dir=None
        )
        
        if solution is not None:
            print("Simplified optimization successful!")
            print(f"Biomass flux: {solution}")
            return solution, flux_distribution
        else:
            print("Even simplified optimization failed")
            return None, None
            
    except Exception as e:
        print(f"Simplified optimization failed: {e}")
        return None, None


def run_optimization(model, kcat_dict, objective_reaction, gene_sequences_dict=None, 
                    enzyme_upper_bound=0.125, enzyme_ratio=True, maximization=True, 
                    multi_enzyme_off=False, isoenzymes_off=False, 
                    promiscuous_off=False, complexes_off=False):
    """
    Run enzyme-constrained flux balance analysis.
    
    Modified to handle missing data gracefully by treating reactions with missing 
    enzyme data as regular reactions without kcat constraints.
    
    FIXED: Convert kcat from 1/s to 1/hr to match flux units
    - Flux is in mmol/gDCW/hr
    - kcat is provided in 1/s and converted to 1/hr by multiplying by 3600
    - Constraint: flux <= kcat * enzyme
    
    Parameters
    ----------
    model : cobra.Model or str
        COBRA model object or path to model file
    kcat_dict : dict or str
        Dictionary of kcat values or path to kcat CSV file
    objective_reaction : str
        Reaction ID to maximize/minimize
    gene_sequences_dict : dict, optional
        Dictionary mapping gene IDs to sequences
    enzyme_upper_bound : float, optional
        Upper bound for total enzyme concentration
    enzyme_ratio : bool, optional
        Whether to use enzyme ratio constraint
    maximization : bool, optional
        Whether to maximize (True) or minimize (False) the objective
    multi_enzyme_off : bool, optional
        Whether to disable multi-enzyme reactions
    isoenzymes_off : bool, optional
        Whether to disable isoenzyme handling
    promiscuous_off : bool, optional
        Whether to disable promiscuous enzyme handling
    complexes_off : bool, optional
        Whether to disable enzyme complex handling
        
    Returns
    -------
    tuple
        (solution_value, df_FBA, gene_sequences_dict)
    """
    # Model handling - load if string path provided
    if isinstance(model, str):
        try:
            directory = os.path.dirname(__file__)
            GEM_file = os.path.join(directory, model)
            mod = cb.io.read_sbml_model(GEM_file)
        except:  # noqa: E722
            raise ValueError(f"Could not load model from path: {model}")
    else:
        mod = model
    
    # Gene sequences handling
    if gene_sequences_dict is None and isinstance(gene_sequences_dict, str):
        try:
            directory = os.path.dirname(__file__)
            gene_seq_file = os.path.join(directory, gene_sequences_dict)
            gene_seq_df = pd.read_csv(gene_seq_file)
            gene_sequences_dict = pd.Series(gene_seq_df.Sequence.values, 
                                         index=gene_seq_df.Single_gene).to_dict()
        except:  # noqa: E722
            raise ValueError(f"Could not load gene sequences from: {gene_sequences_dict}")
    
    # kcat_dict handling
    if isinstance(kcat_dict, str):
        try:
            df_kcat = pd.read_csv(kcat_dict)
            kcat_dict = df_kcat.set_index('Key').to_dict()['Value']
        except:  # noqa: E722
            raise ValueError(f"Could not load kcat dictionary from: {kcat_dict}")
    
    # ============================================================ #
    #                         MODEL SET UP 
    # ============================================================ #
    # Indexes
    metabolites = list(set(metabolite.id for metabolite in mod.metabolites))
    reactions = list(set(reaction.id for reaction in mod.reactions))
    genes = list(set(gene.id for reaction in mod.reactions for gene in reaction.genes))
    
    # DICTIONARIES for flux bounds, kcats, replaced GPRs and S matrix 
    S_mat = {}
    lower_bounds = {}
    upper_bounds = {}
    kcat = {}
    gpr = {}
    reaction_gene_tuple = set()

    for reaction in mod.reactions:
        # Flux bounds dict
        lower_bounds[reaction.id] = reaction.lower_bound
        upper_bounds[reaction.id] = reaction.upper_bound
        
        # kcat dict - FIXED: Convert kcat from 1/s to 1/hr
        kcat_value = (reaction.annotation).get('kcat')
        if kcat_value is not None:
            if isinstance(kcat_value, list):  # For multiple kcat
                # Convert each kcat from 1/s to 1/hr by multiplying by 3600
                kcat_list = [float(value) * 3600 for value in kcat_value]
                if kcat_list:
                    kcat[reaction.id] = kcat_list
            else:  # For single kcats
                try: 
                    # Convert kcat from 1/s to 1/hr by multiplying by 3600
                    single_kcat = float(kcat_value) * 3600
                    kcat[reaction.id] = [single_kcat]
                except ValueError:
                    pass
        
        # GPR with replaced kcats dict
        gpr_value = (reaction.annotation).get('gpr_replaced')
        if gpr_value is not None:
            gpr[reaction.id] = gpr_value
            
        # Reaction ID - Gene ID tuple    
        for gene in reaction.genes:
            reaction_gene_tuple.add((reaction.id, gene.id))
        
        # S matrix dict
        for met in mod.metabolites:
            try:
                reaction.get_coefficient(met.id)
            except:  # noqa: E722
                pass
            else:
                S_mat[met.id, reaction.id] = reaction.get_coefficient(met.id)

    # Convert kcat_dict values from 1/s to 1/hr
    kcat_dict_hr = {}
    for key, value in kcat_dict.items():
        if isinstance(value, list):
            # Convert each kcat from 1/s to 1/hr by multiplying by 3600
            kcat_dict_hr[key] = [v * 3600 for v in value]
        else:
            # Single kcat value
            kcat_dict_hr[key] = value * 3600
    kcat_dict = kcat_dict_hr  # Replace original with converted values

    # LISTS used in rule_kcat
    single_enzyme = []
    multiple_enzyme = []
    no_enzyme = []
    
    # CHECKPOINTS for rule_kcat
    single_enzyme_pass = [] 
    multiple_enzyme_pass = []
    no_enzyme_pass = []

    for reaction in mod.reactions:
        gpr_tag = (reaction.annotation).get('gpr')
        enzymes_for_reaction = [i for j, i in reaction_gene_tuple if j == reaction.id]
        
        if multi_enzyme_off:
            if gpr_tag == '1' or gpr_tag == 'AND/OR':
                single_enzyme.extend([(reaction.id, i) for i in enzymes_for_reaction])
            else: 
                no_enzyme.append(reaction.id)
        else:
            if gpr_tag == '1':
                single_enzyme.extend([(reaction.id, i) for i in enzymes_for_reaction])
            elif gpr_tag == 'AND/OR':
                multiple_enzyme.extend([(reaction.id, i) for i in enzymes_for_reaction])
            else:
                no_enzyme.append(reaction.id)

    # ============================================================ #
    #                       PYOMO MODEL 
    # ============================================================ #
    # VARIABLES
    Concretemodel = ConcreteModel()  # noqa: F405
    Concretemodel.reaction = Var(reactions, within=NonNegativeReals, bounds=(0, 999)) # Flux - mmol/gDCW/hr  # noqa: F405
    Concretemodel.enzyme = Var(genes, within=NonNegativeReals) # mmol/gDCW  # noqa: F405
    Concretemodel.enzyme_set = Set(initialize=genes)  # noqa: F405
    Concretemodel.enzyme_min = Var(reaction_gene_tuple, within=NonNegativeReals, initialize=0) # mmol/gDCW  # noqa: F405

    # OBJECTIVE FUNCTION: maximizing or minimizing reaction
    if maximization:
        def rule_obj(m, objective_var):
            return m.reaction[objective_var]
        Concretemodel.objective = Objective(rule=rule_obj(Concretemodel, objective_reaction), sense=maximize)  # noqa: F405
    else:
        def rule_obj(m, objective_var):
            return m.reaction[objective_var]
        Concretemodel.objective = Objective(rule=rule_obj(Concretemodel, objective_reaction), sense=minimize)  # noqa: F405

    # CONSTRAINT: steady state
    def rule_S_mat(m, t):
        return sum(S_mat[t, j] * m.reaction[j] for j in reactions if (t, j) in S_mat.keys()) == 0
    Concretemodel.set_S_mat = Constraint(metabolites, rule=rule_S_mat)  # noqa: F405

    # CONSTRAINT: flux bounds
    def rule_bounds(m, j):
        return inequality(lower_bounds[j], m.reaction[j], upper_bounds[j])  # noqa: F405
    Concretemodel.rxn_bounds = Constraint(reactions, rule=rule_bounds)  # noqa: F405

    # CONSTRAINT: minimum enzyme concentration
    def enzyme_min_constraint(m, j, i):
        if j in gpr:
            gpr_string = gpr[j]
            if 'and' in gpr_string:
                return m.enzyme_min[j, i] <= m.enzyme[i]
            else:
                return Constraint.Feasible  # noqa: F405
        else:
            return Constraint.Feasible  # noqa: F405
    
    # FUNCTION for rule_kcat to handle parentheses in GPRs
    def evaluate_parentheses(m, j, i, gpr_string):
        # Count the number of parentheses
        num_parentheses = gpr_string.count('(')
        
        while num_parentheses > 0:
            # Find most inner parentheses with rfind (begin from right)
            start = gpr_string.rfind('(')
            end = gpr_string.find(')', start) 

            # Extract and evaluate the gpr inside
            inner_gpr = gpr_string[start + 1:end]
            inner_result = evaluate_gpr(m, j, i, inner_gpr)

            # Replace gpr_string with new result
            gpr_string = gpr_string[:start] + str(inner_result) + gpr_string[end + 1:]
            
            # Update the count of parentheses
            num_parentheses = gpr_string.count('(')
        
        return evaluate_gpr(m, j, i, gpr_string)
    
    # FUNCTION for rule_kcat to optimize GPRs
    sum_enzymes_check = []
    gpr_string_check = []
    def evaluate_gpr(m, j, i, gpr_string):
        enzyme_kcats = re.findall(r'[0-9.]+', gpr_string)
        current_set = []
        
        for k in enzyme_kcats:
            try:
                # Convert each kcat from 1/s to 1/hr by multiplying by 3600
                current_set.append(float(k) * 3600)
            except:  # noqa: E722
                pass
            
        # isozymes
        if not isoenzymes_off: 
            if 'or' in gpr_string:
                # Now both flux and kcat are in 1/hr units
                return m.reaction[j] <= sum(k * m.enzyme[i] for k in current_set)
        else:
            if 'or' in gpr_string:
                return m.reaction[j] <= 1000
                
        # complexes
        if not complexes_off:
            if 'and' in gpr_string:
                sum_enzymes_check.append([j, i])
                gpr_string_check.append([j, gpr_string])
                mean_kcat = max(current_set)  # Change to max or mean (min might be too small)
                # Now both flux and kcat are in 1/hr units
                return m.reaction[j] <= mean_kcat * m.enzyme_min[j, i]
        else:
            if 'and' in gpr_string:
                return m.reaction[j] <= 1000    

    # MODIFIED: CONSTRAINT: enzyme kinetics - Now handles missing data gracefully
    def rule_kcat(m, j, i):
        # First, check if we have data for this reaction-gene pair
        # This is the key modification - we check for missing data first
        reaction_has_kcat = False
        gene_has_sequence = False
        
        # Check if reaction has kcat data (either from model annotation or kcat_dict)
        if j in kcat:
            if kcat[j] and not (math.isnan(kcat[j][0]) or kcat[j][0] is None):
                reaction_has_kcat = True
        # Also check in the provided kcat_dict
        elif (j, i) in kcat_dict:
            if kcat_dict[(j, i)] and not (math.isnan(kcat_dict[(j, i)][0]) or kcat_dict[(j, i)][0] is None):
                reaction_has_kcat = True
        
        # Check if gene has sequence data
        if i in gene_sequences_dict and gene_sequences_dict[i]:
            gene_has_sequence = True
        
        # If we're missing either kcat or sequence, treat as regular reaction
        if not reaction_has_kcat or not gene_has_sequence:
            no_enzyme_pass.append(j)
            return Constraint.Feasible  # noqa: F405
        
        # If we have both data, proceed with original logic
        if (j, i) in single_enzyme and j in kcat:
            single_enzyme_pass.append(j)
            # Now both flux and kcat are in 1/hr units
            return m.reaction[j] <= kcat[j][0] * m.enzyme[i]

        if (j, i) in multiple_enzyme and j in gpr:
            multiple_enzyme_pass.append(j)
            
            gpr_string = gpr[j]

            # GPRs with parentheses
            if '(' in gpr_string:
                return evaluate_parentheses(m, j, i, gpr_string)
            
            # GPRs without parentheses
            else:
                return evaluate_gpr(m, j, i, gpr_string)
        
        else:
            no_enzyme_pass.append(j)
            return Constraint.Feasible  # noqa: F405

    Concretemodel.set_kcat = Constraint(reaction_gene_tuple, rule=rule_kcat)  # noqa: F405

    # MODIFIED: CONSTRAINT: promiscuous enzymes - Now handles missing data
    def rule_promiscuous_E(m, i):
        # Check if the gene has a sequence
        if i not in gene_sequences_dict or not gene_sequences_dict.get(i):
            return Constraint.Feasible  # noqa: F405
        
        # Get valid reactions for this gene (ones with kcat data)
        valid_reactions = []
        for j in reactions:
            if (j, i) in kcat_dict and kcat_dict[(j, i)] and kcat_dict[(j, i)][0] is not None:
                valid_reactions.append(j)
        
        # If no valid reactions, no constraint
        if not valid_reactions:
            return Constraint.Feasible  # noqa: F405
        
        try:
            # Now both flux and kcat are in 1/hr units
            return max(m.reaction[j] / kcat_dict[j, i][0] for j in valid_reactions) <= m.enzyme[i]
        except:  # noqa: E722
            return Constraint.Feasible  # noqa: F405
            
    if not promiscuous_off:  
        Concretemodel.set_promiscuous_E = Constraint(genes, rule=rule_promiscuous_E)  # noqa: F405
    
    # FUNCTION for retrieving molecular weights from protein sequences
    def calculate_molecular_weight(sequence):
        if not sequence:  # Handle empty sequences
            return 0
        try:
            return molecular_weight(sequence, seq_type='protein')
        except:  # noqa: E722
            return 0  # Return 0 for invalid sequences
    
    # MODIFIED: CONSTRAINT total enzyme - Now handles missing data
    if enzyme_ratio:  
        # VARIABLE
        Concretemodel.E_ratio = Var(within=NonNegativeReals, bounds=(0, enzyme_upper_bound)) # gP/gDCW  # noqa: F405
        
        # Handle missing sequences by using default molecular weight
        def get_molecular_weight(gene):
            if gene in gene_sequences_dict and gene_sequences_dict[gene]:
                mw = calculate_molecular_weight(gene_sequences_dict[gene])
                return mw if mw > 0 else 100000  # Default MW if calculation fails
            else:
                return 100000  # Default molecular weight for missing sequences (in g/mol)
        
        Concretemodel.enzyme_molecular_weights = Param(  # noqa: F405
            Concretemodel.enzyme_set, 
            initialize={gene: get_molecular_weight(gene) for gene in genes}
        )
        
        # CONSTRAINT
        def rule_E_total(m):
            total_enzyme_weight_expr = sum(m.enzyme[i] * m.enzyme_molecular_weights[i] 
                                          for i in m.enzyme) * 0.001
            return total_enzyme_weight_expr <= m.E_ratio
        Concretemodel.set_E_total = Constraint(rule=rule_E_total)  # noqa: F405
        
    else:
        # VARIABLE
        Concretemodel.E_total = Var(within=NonNegativeReals, bounds=(0, enzyme_upper_bound)) # mmol/gDCW  # noqa: F405
        
        # CONSTRAINT
        def rule_E_total(m):
            return sum(m.enzyme[i] for (j, i) in reaction_gene_tuple) <= m.E_total
        Concretemodel.set_E_total = Constraint(rule=rule_E_total)  # noqa: F405
    
    # Solving
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 1000  # Increased iteration limit
    solver.options['acceptable_tol'] = 1e-6  # More lenient tolerance
    
    try:
        # Attempt to solve the model
        results = solver.solve(Concretemodel, tee=False)
        
        # Check the solver status
        if (results.solver.status == pyo.SolverStatus.ok and 
            results.solver.termination_condition == pyo.TerminationCondition.optimal):
            # Successful optimization
            solution_value = pyo.value(Concretemodel.objective)
    
            variable = []
            index = []
            value = []
    
            for v in Concretemodel.component_objects(pyo.Var, active=True):
                for i in v:
                    variable.append(v.name)
                    index.append(i)
                    value.append(pyo.value(v[i]))
    
            df_FBA = pd.DataFrame({"Variable": variable, "Index": index, "Value": value})
            
            # Report statistics about missing data handling
            total_reactions = len(reaction_gene_tuple)
            constrained_reactions = len(single_enzyme_pass) + len(multiple_enzyme_pass)
            unconstrained_reactions = len(no_enzyme_pass)
            
            print("Optimization completed successfully!")
            print(f"Total reaction-gene pairs: {total_reactions}")
            print(f"Enzyme-constrained pairs: {constrained_reactions}")
            print(f"Unconstrained pairs (missing data): {unconstrained_reactions}")
            
        else:
            # Handle unsuccessful optimization
            raise ValueError(f"Solver did not find an optimal solution. Status: {results.solver.status}, "
                            f"Termination condition: {results.solver.termination_condition}")
    
    except Exception as e:
        # Handle exceptions raised by the solver
        print(f"An error occurred during optimization: {e}")
        df_FBA = pd.DataFrame()  # Return an empty DataFrame
        solution_value = None
            
    return solution_value, df_FBA, gene_sequences_dict

def create_descriptive_filename(objective_reaction, enzyme_upper_bound, maximization, 
                         multi_enzyme_off, isoenzymes_off, promiscuous_off, complexes_off,
                         output_dir=None, extension='.csv'):
    """
    Create a descriptive filename based on optimization parameters.
    
    Parameters
    ----------
    objective_reaction : str
        The reaction ID used as objective
    enzyme_upper_bound : float
        Upper bound for total enzyme concentration
    maximization : bool
        Whether maximization was used
    multi_enzyme_off : bool
        Whether multi-enzyme reactions were disabled
    isoenzymes_off : bool
        Whether isoenzyme handling was disabled
    promiscuous_off : bool
        Whether promiscuous enzyme handling was disabled
    complexes_off : bool
        Whether enzyme complex handling was disabled
    output_dir : str, optional
        Directory to save the file in
    extension : str, optional
        File extension to use
        
    Returns
    -------
    str
        The complete filepath
    """
    import os
    
    # Shorten the objective reaction name if it's too long
    if len(objective_reaction) > 20:
        obj_short = objective_reaction[:20]
    else:
        obj_short = objective_reaction
    
    # Create descriptive part for optimization direction
    opt_dir = "max" if maximization else "min"
    
    # Create descriptive parts for enzyme constraints
    multi = "noMulti" if multi_enzyme_off else "Multi"
    iso = "noIso" if isoenzymes_off else "Iso"
    promis = "noPromis" if promiscuous_off else "Promis"
    complex_str = "noComplex" if complexes_off else "Complex"
    
    # Create the filename
    filename = f"FBA_{obj_short}_{opt_dir}_E{enzyme_upper_bound}_{multi}_{iso}_{promis}_{complex_str}{extension}"
    
    # Clean up any characters that might cause issues in filenames
    filename = filename.replace(' ', '_').replace('/', '_').replace('\\', '_')
    
    # Join with output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    else:
        return filename


def run_optimization_with_dataframe(model, processed_df, objective_reaction, 
                    enzyme_upper_bound=0.125, enzyme_ratio=True, maximization=True, 
                    multi_enzyme_off=False, isoenzymes_off=False, 
                    promiscuous_off=False, complexes_off=False,
                    output_dir=None, save_results=True):
    """
    Run enzyme-constrained flux balance analysis using a processed dataframe.
    
    Parameters
    ----------
    model : cobra.Model or str
        COBRA model object or path to model file
    processed_df : pandas.DataFrame
        DataFrame containing Reactions, Single_gene, SEQ, SMILES, and kcat_mean columns
    objective_reaction : str
        Reaction ID to maximize/minimize
    enzyme_upper_bound : float, optional
        Upper bound for total enzyme concentration
    enzyme_ratio : bool, optional
        Whether to use enzyme ratio constraint
    maximization : bool, optional
        Whether to maximize (True) or minimize (False) the objective
    multi_enzyme_off : bool, optional
        Whether to disable multi-enzyme reactions
    isoenzymes_off : bool, optional
        Whether to disable isoenzyme handling
    promiscuous_off : bool, optional
        Whether to disable promiscuous enzyme handling
    complexes_off : bool, optional
        Whether to disable enzyme complex handling
    output_dir : str, optional
        Directory to save results in
    save_results : bool, optional
        Whether to automatically save results to a file
        
    Returns
    -------
    tuple
        (solution_value, df_FBA, gene_sequences_dict, output_filepath)
    """

    
    # Check if required columns exist in processed_df
    required_cols = ['Reactions', 'Single_gene', 'SEQ', 'kcat_mean']
    missing_cols = [col for col in required_cols if col not in processed_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in processed_df: {missing_cols}")

    # Extract kcat dictionary and gene sequences from processed_df
    kcat_dict = {}
    gene_sequences_dict = {}
    
    # Create a dictionary mapping from (reaction_id, gene_id) to kcat value
    for _, row in processed_df.iterrows():
        if pd.notna(row['kcat_mean']) and pd.notna(row['SEQ']):
            reaction_id = row['Reactions']
            gene_id = row['Single_gene']
            
            # Store the kcat value as a list (to match the original function's format)
            kcat_dict[(reaction_id, gene_id)] = [row['kcat_mean']]
            
            # Store gene sequence for molecular weight calculation
            if gene_id not in gene_sequences_dict and pd.notna(row['SEQ']):
                gene_sequences_dict[gene_id] = row['SEQ']
    
    # Call the original run_optimization function with the extracted kcat_dict
    solution_value, df_FBA, gene_sequences_dict = run_optimization(
        model=model, 
        kcat_dict=kcat_dict, 
        objective_reaction=objective_reaction,
        gene_sequences_dict=gene_sequences_dict,
        enzyme_upper_bound=enzyme_upper_bound, 
        enzyme_ratio=enzyme_ratio, 
        maximization=maximization,
        multi_enzyme_off=multi_enzyme_off, 
        isoenzymes_off=isoenzymes_off,
        promiscuous_off=promiscuous_off, 
        complexes_off=complexes_off
    )
    
    # Create descriptive filename and save results if requested
    output_filepath = create_descriptive_filename(
        objective_reaction, 
        enzyme_upper_bound, 
        maximization,
        multi_enzyme_off, 
        isoenzymes_off, 
        promiscuous_off, 
        complexes_off,
        output_dir=output_dir
    )
    
    if save_results and solution_value is not None:
        # Create directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save the results
        df_FBA.to_csv(output_filepath, index=False)
        print(f"Results saved to: {output_filepath}")
    
    return solution_value, df_FBA, gene_sequences_dict, output_filepath

# Create a version of run_optimization with fixed unit conversions



def debug_enzyme_constraints_detailed(model, processed_data, objective_reaction):
    """
    Debug the specific enzyme constraint implementation
    """
    print("=== ENZYME CONSTRAINT IMPLEMENTATION DEBUG ===\n")
    
    # Create kcat dictionary and gene sequences
    kcat_dict = {}
    gene_sequences_dict = {}
    
    # Process data - exactly as in run_optimization_with_dataframe
    print("1. Creating kcat dictionary and gene sequences...")
    for _, row in processed_data.iterrows():
        if pd.notna(row['kcat_mean']) and pd.notna(row['SEQ']):
            reaction_id = row['Reactions']
            gene_id = row['Single_gene']
            
            # Store the kcat value as a list
            kcat_dict[(reaction_id, gene_id)] = [row['kcat_mean']]
            
            # Store gene sequence
            if gene_id not in gene_sequences_dict and pd.notna(row['SEQ']):
                gene_sequences_dict[gene_id] = row['SEQ']
    
    print(f"Created kcat_dict with {len(kcat_dict)} entries")
    print(f"Created gene_sequences_dict with {len(gene_sequences_dict)} entries")
    
    # Sample some entries
    print("\nSample kcat_dict entries:")
    for i, (key, value) in enumerate(list(kcat_dict.items())[:3]):
        print(f"  {key}: {value}")
    
    # Check for problematic values
    print("\n2. Checking for problematic values...")
    
    problematic_kcats = []
    problematic_sequences = []
    
    for key, kcat_list in kcat_dict.items():
        if len(kcat_list) == 0 or kcat_list[0] <= 0 or math.isnan(kcat_list[0]) or math.isinf(kcat_list[0]):
            problematic_kcats.append((key, kcat_list))
    
    for gene, seq in gene_sequences_dict.items():
        if not seq or seq == '' or seq == 'None':
            problematic_sequences.append((gene, seq))
    
    print(f"Problematic kcat values: {len(problematic_kcats)}")
    print(f"Problematic sequences: {len(problematic_sequences)}")
    
    # Show examples
    if problematic_kcats:
        print("Sample problematic kcats:")
        for item in problematic_kcats[:3]:
            print(f"  {item}")
    
    if problematic_sequences:
        print("Sample problematic sequences:")
        for item in problematic_sequences[:3]:
            print(f"  {item}")
    
    # Test molecular weight calculation
    print("\n3. Testing molecular weight calculation...")
    
    test_sequences = list(gene_sequences_dict.values())[:5]
    for seq in test_sequences:
        try:
            mw = molecular_weight(seq, seq_type='protein')
            print(f"Sequence length {len(seq)}: MW = {mw}")
        except Exception as e:
            print(f"Error calculating MW for sequence: {e}")
    
    # Check enzyme concentration calculation
    print("\n4. Checking enzyme concentration calculation...")
    
    # Use just one reaction-gene pair for testing
    test_key = list(kcat_dict.keys())[0]
    test_kcat = kcat_dict[test_key][0]
    test_gene = test_key[1]
    test_seq = gene_sequences_dict.get(test_gene, '')
    
    print(f"Test case: {test_key}")
    print(f"kcat: {test_kcat}")
    print(f"Gene: {test_gene}")
    print(f"Sequence length: {len(test_seq)}")
    
    # Calculate enzyme requirement for a typical flux
    typical_flux = 1.0  # mmol/gDCW/hr
    enzyme_mmol = typical_flux / test_kcat
    
    if test_seq:
        mw = molecular_weight(test_seq, seq_type='protein')
        enzyme_g = enzyme_mmol * mw / 1000 / 1000  # Convert to g/gDCW
        
        print(f"For flux {typical_flux}, requires {enzyme_mmol} mmol enzyme")
        print(f"MW: {mw}")
        print(f"Enzyme requirement: {enzyme_g} g/gDCW")
        
        if enzyme_g > 1.0:
            print("WARNING: This reaction requires more than cell mass!")
    
    # Test the enzyme constraint creation
    print("\n5. Testing enzyme constraint creation...")
    # Create minimal Pyomo model to test constraints
    from pyomo.environ import (
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        Var,
        maximize,
        value,
    )
    from pyomo.opt import SolverFactory
    
    test_model = ConcreteModel()
    # Add variables
    test_model.reaction = Var(within=NonNegativeReals, bounds=(0, 10))
    test_model.enzyme = Var(within=NonNegativeReals, bounds=(0, 1))
    
    # Add enzyme constraint
    def enzyme_constraint(m):
        return m.reaction <= test_kcat * m.enzyme
    
    test_model.enzyme_con = Constraint(rule=enzyme_constraint)
    
    # Add enzyme bound
    def enzyme_bound(m):
        return m.enzyme <= 0.1  # 10% of cell mass
    
    test_model.enzyme_bound_con = Constraint(rule=enzyme_bound)
    
    # Add objective
    test_model.obj = Objective(expr=test_model.reaction, sense=maximize)
    
    # Solve
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 100
    
    try:
        results = solver.solve(test_model, tee=False)
        print(f"Test model status: {results.solver.status}")
        print(f"Test reaction flux: {value(test_model.reaction)}")
        print(f"Test enzyme amount: {value(test_model.enzyme)}")
        
        # Calculate theoretical maximum
        theoretical_max = test_kcat * 0.1
        print(f"Theoretical maximum flux: {theoretical_max}")
        
    except Exception as e:
        print(f"Error solving test model: {e}")
    
    # Test with enzyme ratio constraint
    print("\n6. Testing enzyme ratio constraint...")
    
    test_model_ratio = ConcreteModel()
    
    # Add variables
    test_model_ratio.reaction = Var(within=NonNegativeReals, bounds=(0, 10))
    test_model_ratio.enzyme = Var(within=NonNegativeReals)
    test_model_ratio.E_ratio = Var(within=NonNegativeReals, bounds=(0, 0.125))
    
    # Add enzyme constraint
    def enzyme_constraint_ratio(m):
        return m.reaction <= test_kcat * m.enzyme
    
    test_model_ratio.enzyme_con = Constraint(rule=enzyme_constraint_ratio)
    
    # Add enzyme ratio constraint
    if test_seq:
        mw = molecular_weight(test_seq, seq_type='protein')
    else:
        mw = 50000  # Default MW
    
    def enzyme_ratio_constraint(m):
        return m.enzyme * mw * 0.001 <= m.E_ratio
    
    test_model_ratio.ratio_con = Constraint(rule=enzyme_ratio_constraint)
    
    # Add objective
    test_model_ratio.obj = Objective(expr=test_model_ratio.reaction, sense=maximize)
    
    # Solve
    try:
        results_ratio = solver.solve(test_model_ratio, tee=False)
        print(f"Test ratio model status: {results_ratio.solver.status}")
        print(f"Test reaction flux: {value(test_model_ratio.reaction)}")
        print(f"Test enzyme amount: {value(test_model_ratio.enzyme)}")
        print(f"Test E_ratio: {value(test_model_ratio.E_ratio)}")
        
        # Calculate theoretical limits
        max_enzyme_mmol = 0.125 / (mw / 1000 / 1000)
        max_flux = test_kcat * max_enzyme_mmol
        print(f"Theoretical max enzyme: {max_enzyme_mmol} mmol")
        print(f"Theoretical max flux: {max_flux}")
        
    except Exception as e:
        print(f"Error solving test ratio model: {e}")
    
    # Check for unit conversion issues
    print("\n7. Checking unit conversions...")
    
    # Example calculation
    flux_mmol_per_hr = 1.0  # mmol/gDCW/hr
    kcat_per_s = test_kcat  # 1/s
    
    # Convert flux to 1/s
    flux_per_s = flux_mmol_per_hr / 3600  # Convert hr to s
    
    # Calculate required enzyme
    enzyme_required_mmol = flux_per_s / kcat_per_s
    
    print(f"Flux: {flux_mmol_per_hr} mmol/gDCW/hr = {flux_per_s} mmol/gDCW/s")
    print(f"kcat: {kcat_per_s} 1/s")
    print(f"Required enzyme: {enzyme_required_mmol} mmol/gDCW")
    
    if test_seq:
        mw = molecular_weight(test_seq, seq_type='protein')
        enzyme_required_g = enzyme_required_mmol * mw / 1000 / 1000
        print(f"Required enzyme: {enzyme_required_g} g/gDCW")
    
    return kcat_dict, gene_sequences_dict


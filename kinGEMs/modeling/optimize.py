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
from cobra.util.array import create_stoichiometric_matrix
import numpy as np

# Troubleshooting infeasible optimization
# Add this code before running your optimization to diagnose the issue
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *  # noqa: F403
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from ..config import ensure_dir_exists  # noqa: F401
from collections import Counter

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
                    promiscuous_off=False, complexes_off=False, print_reaction_conditions=False):
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

    from pyomo.environ import Suffix

    # Model handling - load if string path provided
    if isinstance(model, str):
        try:
            directory = os.path.dirname(__file__)
            GEM_file = os.path.join(directory, model)
            mod = cb.io.read_sbml_model(GEM_file)
            print("Loaded model from GEM file")
        except:  # noqa: E722
            raise ValueError(f"Could not load model from path: {model}")
    else:
        mod = model
        print("Loaded model from input irreversible model!")
    
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
        
    print("PRINT INITIAL KCAT DICT: ", kcat_dict)
    
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

        # print(f"Reaction {reaction} has lower bound {reaction.lower_bound} and upper bound {reaction.upper_bound}")
        
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

    # CHECKPOINTS for evaluate_gpr & rule_promiscuous
    isoenzymes_pass = []
    enzyme_complexes_pass = [] 
    promiscuous_pass = []


    for reaction in mod.reactions:
        gpr_tag = (reaction.annotation).get('gpr')
        # Get the reaction object from the model
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
    Concretemodel.reaction = Var(reactions, within=Reals, bounds=(-999, 999)) # Flux - mmol/gDCW/hr  # noqa: F405
    Concretemodel.enzyme = Var(genes, within=NonNegativeReals) # mmol/gDCW  # noqa: F405
    Concretemodel.enzyme_set = Set(initialize=genes)  # noqa: F405
    Concretemodel.enzyme_min = Var(reaction_gene_tuple, within=NonNegativeReals, initialize=0) # mmol/gDCW  # noqa: F405

    # OBJECTIVE FUNCTION: maximizing or minimizing reaction
    if maximization:
        def rule_obj(m, objective_var):
            print("OBJECTIVE FUNCTION IS: ", m.reaction[objective_var])
            return m.reaction[objective_var]
        Concretemodel.objective = Objective(rule=rule_obj(Concretemodel, objective_reaction), sense=maximize)  # noqa: F405
    else:
        def rule_obj(m, objective_var):
            return m.reaction[objective_var]
        Concretemodel.objective = Objective(rule=rule_obj(Concretemodel, objective_reaction), sense=minimize)  # noqa: F405

    rxn = mod.reactions.get_by_id(objective_reaction)
    print("BIOMASS bounds:", rxn.lower_bound, rxn.upper_bound)
    
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
                current_set.append(float(k))
            except:  # noqa: E722
                pass
            
        # isozymes
        if not isoenzymes_off: 
            if 'or' in gpr_string:
                isoenzymes_pass.append(j)
                # Now both flux and kcat are in 1/hr units
                # print(f"ISOENZYMES SCENARIO: kcat value {current_set} 1/hr")
                return m.reaction[j] <= sum(k * m.enzyme[i] for k in current_set)
        else:
            if 'or' in gpr_string:
                return m.reaction[j] <= 1000
                
        # complexes
        if not complexes_off:
            if 'and' in gpr_string:
                enzyme_complexes_pass.append(j)
                sum_enzymes_check.append([j, i])
                gpr_string_check.append([j, gpr_string])
                mean_kcat = max(current_set)  # Change to max or mean (min might be too small)
                # Now both flux and kcat are in 1/hr units
                # print(f"COMPLEX SCENARIO: kcat value {mean_kcat} 1/hr")
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
                # print("REACTION SHOULD HAVE A KCAT HERE", reaction_has_kcat)
                # print(f"reaction: {j} and kcat: {kcat[j][0]}")
        # Also check in the provided kcat_dict
        elif (j, i) in kcat_dict:
            if kcat_dict[(j, i)] and not (math.isnan(kcat_dict[(j, i)][0]) or kcat_dict[(j, i)][0] is None):
                reaction_has_kcat = True
                # print("REACTION SHOULD HAVE A KCAT HERE", reaction_has_kcat)
                # print(f"reaction: {j} and gene{i} and kcat: {kcat[j, i][0]}")


        # print(f"CHECKING FOR KCAT DATA IN REACTION {j}: {reaction_has_kcat}")
        # Check if gene has sequence data
        if i in gene_sequences_dict and gene_sequences_dict[i]:
            gene_has_sequence = True
        # print(f"CHECKING FOR GENE DATA IN REACTION {j}: {gene_has_sequence}")
        
        # If we're missing either kcat or sequence, treat as regular reaction
        if not reaction_has_kcat or not gene_has_sequence:
            no_enzyme_pass.append(j)
            return Constraint.Feasible  # noqa: F405
        
        if reaction_has_kcat is False or gene_has_sequence is False:
            no_enzyme_pass.append(j)
            return Constraint.Feasible  # noqa: F405
        
        # If we have both data, proceed with original logic
        if (j, i) in single_enzyme and j in kcat:
            single_enzyme_pass.append(j)
            # Now both flux and kcat are in 1/hr units
            # print(f"SINGLE ENZYME CASE, reaction: {j} and kcat {kcat[j][0]}")
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
            promiscuous_pass.append(j)
            return max(m.reaction[j] / kcat_dict[j, i][0] for j in valid_reactions) <= m.enzyme[i]
            # (f"PROMISCUOUS ENZYME CASE: kcat: {kcat[j, i]}, reaction: {j}")
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
    
    def get_molecular_weight(gene):
            if gene in gene_sequences_dict and gene_sequences_dict[gene]:
                mw = calculate_molecular_weight(gene_sequences_dict[gene])
                return mw if mw > 0 else 100000  # Default MW if calculation fails
            else:
                return 100000  # Default molecular weight for missing sequences (in g/mol)
    
    # MODIFIED: CONSTRAINT total enzyme - Now handles missing data
    if enzyme_ratio: 
        # VARIABLE
        Concretemodel.E_ratio = Var(within=NonNegativeReals, bounds=(0, enzyme_upper_bound)) # gP/gDCW  # noqa: F405
        
        # Handle missing sequences by using default molecular weight
        
        
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

    
    ## CONCRETE MODEL BUILDING DONE! Now we start optimization portion 
    Concretemodel.reaction[objective_reaction].value = 0.1
    init_e = enzyme_upper_bound / len(genes)
    for g in genes:
        Concretemodel.enzyme[g].value = init_e 

    # 3) Use IPOPT for (non)linear optimization
    from pyomo.environ import SolverFactory
    solver = SolverFactory('ipopt')
    # IPOPT options — you can tweak these
    solver.options['warm_start_init_point']       = 'yes'   # use your .value starts :contentReference[oaicite:0]{index=0}

    # (optional) let IPOPT start closer to the corner by reducing the barrier shift
    solver.options['mu_strategy'] = 'monotone'
    solver.options['mu_init']     = 1e-6

    # 4) Now solve (your Concretemodel.dual suffix was already attached above)
    try:
        results = solver.solve(
            Concretemodel,
            tee=False
        )

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            # Do something when the solution in optimal and feasible
            print("Solver status is okay! ")
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            # Do something when model in infeasible
            print("Solver status is infeasible")
        else:
            # Something else is wrong
            print("Solver Status: ",  result.solver.status)
            
            # … (often‐used code to extract objective, variables, df_FBA, etc.) …
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

        # (print your reaction‐condition statistics here, if desired)
        total_reactions = len(reaction_gene_tuple)
        constrained_reactions = len(single_enzyme_pass) + len(multiple_enzyme_pass)
        unconstrained_reactions = len(no_enzyme_pass)
        promiscuous_enzymes = len(promiscuous_pass)
        isoenzyme_reactions = len(isoenzymes_pass)
        enzyme_complexes_reactions = len(enzyme_complexes_pass)

        if print_reaction_conditions:
            print("Optimization completed successfully!")
            print(f"Total reaction-gene pairs: {total_reactions}")
            print(f"Enzyme-constrained pairs: {constrained_reactions}")
            print(f"Unconstrained pairs (missing data): {unconstrained_reactions}")
            print(f"Promiscuous enzymes in system: {promiscuous_enzymes}")
            print(f"Isoenzymatic passes: {isoenzyme_reactions}")
            print(f"Enzyme complex reactions: {enzyme_complexes_reactions}")

        else:
            # Handle unsuccessful optimization
            raise ValueError(
                f"Solver did not find an optimal solution. "
                f"Status: {results.solver.status}, "
                f"Termination: {results.solver.termination_condition}"
            )

    except Exception as e:
        # Handle exceptions raised by the solver
        print(f"An error occurred during optimization: {e}")
        df_FBA = pd.DataFrame()  # Return an empty DataFrame
        solution_value = None

    # Print a sanity‐check of which solver was actually used:   
    print("Termination condition:", results.solver.termination_condition)
    print("Solver status:", results.solver.status)

    print("Biomass flux:", pyo.value(Concretemodel.objective))

    return solution_value, df_FBA, gene_sequences_dict, Concretemodel


def run_optimization2(model,
                     kcat_dict,
                     objective_reaction,
                     gene_sequences_dict=None,
                     enzyme_upper_bound=0.125,
                     enzyme_ratio=True,
                     maximization=True,
                     multi_enzyme_off=False,
                     isoenzymes_off=False,
                     promiscuous_off=False,
                     complexes_off=False,
                     print_reaction_conditions=True):
    """
    Enzyme‐constrained FBA via Pyomo, reproducing cobra.optimize() when all flags are off.
    """

    # 1) Load COBRA model if a path was given
    if isinstance(model, str):
        mod = read_sbml_model(model)
    else:
        mod = model.copy()

    # 2) Load or convert kcat_dict to hr⁻¹
    if isinstance(kcat_dict, str):
        df_k = pd.read_csv(kcat_dict)
        kcat_dict = dict(zip(df_k.Key, df_k.Value))
    for key, v in list(kcat_dict.items()):
        if isinstance(v, list):
            kcat_dict[key] = [val * 3600 for val in v]
        else:
            kcat_dict[key] = v * 3600

    # 3) Gather S, bounds, GPR maps
    S = create_stoichiometric_matrix(mod)
    mets      = [m.id for m in mod.metabolites]
    rxns      = [r.id for r in mod.reactions]
    lb = {r.id: r.lower_bound for r in mod.reactions}
    ub = {r.id: r.upper_bound for r in mod.reactions}

    # GPR parsing → reaction–gene pairs and strings
    reaction_gene = []
    gpr_strings   = {}
    for r in mod.reactions:
        gr = r.annotation.get('gpr','').lower()
        gpr_strings[r.id] = gr
        for g in r.genes:
            reaction_gene.append((r.id, g.id))

    # 4) Categorize single vs complex based on flags
    single_enzyme, multiple_enzyme = [], []
    for rid, gid in reaction_gene:
        gpr = gpr_strings.get(rid, '')
        if ('and' in gpr) and not multi_enzyme_off:
            multiple_enzyme.append((rid, gid))
        else:
            single_enzyme.append((rid, gid))

    # 4.1) Count reaction categories
    single_count = len(single_enzyme)
    complex_count = len(multiple_enzyme)
    iso_count = sum(1 for rid, gid in reaction_gene if 'or' in gpr_strings.get(rid, ''))
    # promiscuous: genes associated with >1 reaction
    gene_counts = Counter(gid for rid, gid in reaction_gene if (rid, gid) in kcat_dict)
    prom_count = sum(1 for count in gene_counts.values() if count > 1)

    if print_reaction_conditions:
        print(f"Single‐enzyme pairs: {single_count}")
        print(f"Complex (AND) pairs:   {complex_count}")
        print(f"Isoenzymatic (OR) pairs: {iso_count}")
        print(f"Promiscuous enzymes (genes w/>1 reaction): {prom_count}")

    # 5) Genes list & default molecular weights
    genes = list({g for _, g in reaction_gene})
    if gene_sequences_dict is None:
        gene_sequences_dict = {}
    def get_mw(g):
        seq = gene_sequences_dict.get(g, '')
        return molecular_weight(seq, 'protein') if seq else 1e5
    mw_map = {g: get_mw(g) for g in genes}

    # 6) Build Pyomo model
    m = ConcreteModel()
    m.mets  = Set(initialize=mets)
    m.rxns  = Set(initialize=rxns)
    m.genes = Set(initialize=genes)
    m.rg    = Set(initialize=reaction_gene, dimen=2)

    m.v = Var(m.rxns, domain=Reals, bounds=lambda mo,j: (lb[j], ub[j]))
    m.E = Var(m.genes, domain=NonNegativeReals)
    m.Emin = Var(m.rg, domain=NonNegativeReals)

    if enzyme_ratio:
        m.E_ratio = Var(domain=NonNegativeReals, bounds=(0, enzyme_upper_bound))
        m.mw = mw_map
    else:
        m.E_total = Var(domain=NonNegativeReals, bounds=(0, enzyme_upper_bound))

    m.dual = Suffix(direction=Suffix.IMPORT)
    sense = maximize if maximization else minimize
    m.obj = Objective(expr=m.v[objective_reaction], sense=sense)

    # Mass-balance
    def mass_balance(mo, met_id):
        i = mets.index(met_id)
        return sum(S[i, rxns.index(r)] * mo.v[r] for r in mo.rxns) == 0
    m.mass_balance = Constraint(m.mets, rule=mass_balance)

    # kcat and complex constraints
    def rule_kcat(mo, rid, gid):
        if ((rid, gid) not in kcat_dict) or (gid not in gene_sequences_dict):
            return Constraint.Feasible
        klist = kcat_dict[(rid, gid)]
        if (rid, gid) in single_enzyme:
            return mo.v[rid] <= klist[0] * mo.E[gid]
        if ((rid, gid) in multiple_enzyme) and (not complexes_off):
            return mo.v[rid] <= max(klist) * mo.Emin[rid, gid]
        return Constraint.Feasible
    m.kcat_con = Constraint(m.rg, rule=rule_kcat)

    def rule_complex_min(mo, rid, gid):
        if (rid, gid) in multiple_enzyme:
            return mo.Emin[rid, gid] <= mo.E[gid]
        return Constraint.Feasible
    m.complex_min = Constraint(m.rg, rule=rule_complex_min)

    if not promiscuous_off:
        def rule_prom(mo, gid):
            valid = [(rid, gid) for (rid, g) in reaction_gene if g == gid and (rid, gid) in kcat_dict]
            if not valid:
                return Constraint.Feasible
            return sum(mo.v[rid]/kcat_dict[(rid, gid)][0] for rid, _ in valid) <= mo.E[gid]
        m.promiscuous = Constraint(m.genes, rule=rule_prom)

    if enzyme_ratio:
        def rule_totalE(mo):
            return sum(mo.E[g]*mw_map[g] for g in mo.genes)*1e-3 <= mo.E_ratio
        m.total_enzyme = Constraint(rule=rule_totalE)
    else:
        def rule_totalE2(mo):
            return sum(mo.E[g] for g in mo.genes) <= mo.E_total
        m.total_enzyme = Constraint(rule=rule_totalE2)

    # 7) Solve
    solver = SolverFactory('ipopt')
    solver.options.update({
        'warm_start_init_point':'yes',
        'mu_strategy':'monotone',
        'mu_init':1e-6
    })
    res = solver.solve(m, tee=False)

    # 8) Collect results
    sol_val = m.obj()
    fluxes  = {j: m.v[j]() for j in m.rxns}
    enzymes = {g: m.E[g]() for g in m.genes}
    records = [('flux', r, m.v[r].value) for r in m.rxns]
    records += [('enzyme', g, m.E[g].value) for g in m.genes]
    df_FBA = pd.DataFrame(records, columns=['Variable','ID','Value'])

    return sol_val, fluxes, enzymes, df_FBA

def run_optimization3(
    model,
    kcat_dict,
    objective_reaction,
    gene_sequences_dict=None,
    enzyme_upper_bound=0.125,
    enzyme_ratio=True,
    maximization=True,
    solver_name='glpk'
):
    """
    Enzyme-constrained FBA using Pyomo, matching cobra.optimize() plus kcat constraints.

    Returns solution value, DataFrame of fluxes & enzymes, updated gene_sequences_dict, and the Pyomo model.
    """
    # Load COBRA model
    if isinstance(model, str):
        mod = (
            cobra.io.read_sbml_model(model)
            if model.endswith(('.xml', '.sbml'))
            else cobra.io.load_json_model(model)
        )
    else:
        mod = model

    # Load kcat_dict from CSV file if provided
    if isinstance(kcat_dict, str):
        df = pd.read_csv(kcat_dict)
        kcat_dict = {(r, g): k for r, g, k in zip(df.reaction, df.gene, df.kcat)}

    # Normalize kcat_dict entries: take max element of list (if present), convert to 1/hr
    for key, k in list(kcat_dict.items()):
        raw = max(k) if isinstance(k, list) else k
        k_hr = raw * 3600 if raw < 1000 else raw
        kcat_dict[key] = [k_hr]

    # Build stoichiometry, bounds, and objective
    S = create_stoichiometric_matrix(mod)
    mets = [m.id for m in mod.metabolites]
    rxns = [r.id for r in mod.reactions]
    genes = [g.id for g in mod.genes]
    lb = {r.id: r.lower_bound for r in mod.reactions}
    ub = {r.id: r.upper_bound for r in mod.reactions}
    obj_coef = {r.id: r.objective_coefficient for r in mod.reactions}
    met_index = {mid: i for i, mid in enumerate(mets)}
    rxn_index = {rid: j for j, rid in enumerate(rxns)}
    pairs = [(r.id, g.id) for r in mod.reactions for g in r.genes]

    # Initialize Pyomo model
    m = ConcreteModel()
    m.M = Set(initialize=mets)
    m.R = Set(initialize=rxns)
    m.G = Set(initialize=genes)
    m.K = Set(initialize=pairs, dimen=2)
    m.v = Var(m.R, domain=Reals, bounds=lambda mo,j: (lb[j], ub[j]))
    m.E = Var(m.G, domain=NonNegativeReals)

    # Mass-balance constraints
    def mass_balance(mo, met_id):
        i = met_index[met_id]
        return sum(S[i, rxn_index[r]] * mo.v[r] for r in mo.R) == 0
    m.mass_balance = Constraint(m.M, rule=mass_balance)

    # Objective
    sense = maximize if maximization else minimize
    m.obj = Objective(expr=sum(obj_coef[r] * m.v[r] for r in m.R), sense=sense)

    # kcat constraints
    def kcat_rule(mo, rxn_id, gene_id):
        key = (rxn_id, gene_id)
        if key in kcat_dict:
            k_val = kcat_dict[key][0]
            return mo.v[rxn_id] <= k_val * mo.E[gene_id]
        return Constraint.Skip
    m.kcat_constr = Constraint(m.K, rule=kcat_rule)

    # Total enzyme pool constraint
    if enzyme_ratio:
        if gene_sequences_dict is None:
            gene_sequences_dict = {}
        mw = {g: (molecular_weight(gene_sequences_dict.get(g, ''), seq_type='protein') or 100000)
              for g in genes}
        m.E_ratio = Var(domain=NonNegativeReals, bounds=(0, enzyme_upper_bound))
        m.total_enzyme = Constraint(
            expr=sum(m.E[g] * mw[g] for g in m.G) * 1e-3 <= m.E_ratio
        )
    else:
        m.E_total = Var(domain=NonNegativeReals, bounds=(0, enzyme_upper_bound))
        m.total_enzyme = Constraint(expr=sum(m.E[g] for g in m.G) <= m.E_total)

    # Solve model and load solutions
    solver = SolverFactory(solver_name)
    results = solver.solve(m, tee=False, load_solutions=True)

    # Post-process variable values: ensure defined and within bounds
    for r in m.R:
        val = m.v[r].value
        if val is None:
            m.v[r].value = lb[r]
        else:
            if val < lb[r]:
                m.v[r].value = lb[r]
            elif val > ub[r]:
                m.v[r].value = ub[r]
    for g in m.G:
        if m.E[g].value is None:
            m.E[g].value = 0.0

    # Collect results
    sol_val = value(m.obj)
    records = [('flux', r, m.v[r].value) for r in m.R]
    records += [('enzyme', g, m.E[g].value) for g in m.G]
    df_FBA = pd.DataFrame(records, columns=['Variable','ID','Value'])

    return sol_val, df_FBA, gene_sequences_dict, m



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
                    output_dir=None, save_results=True, print_reaction_conditions=True):
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
    
    # # Call the original run_optimization function with the extracted kcat_dict
    # solution_value, df_FBA, gene_sequences_dict, pm_model = run_optimization(
    #     model=model, 
    #     kcat_dict=kcat_dict, 
    #     objective_reaction=objective_reaction,
    #     gene_sequences_dict=gene_sequences_dict,
    #     enzyme_upper_bound=enzyme_upper_bound, 
    #     enzyme_ratio=enzyme_ratio, 
    #     maximization=maximization,
    #     multi_enzyme_off=multi_enzyme_off, 
    #     isoenzymes_off=isoenzymes_off,
    #     promiscuous_off=promiscuous_off, 
    #     complexes_off=complexes_off,
    #     print_reaction_conditions=print_reaction_conditions
    # )

    # Call the original run_optimization function with the extracted kcat_dict
    # solution_value, fluxes, enzymes, df_FBA = run_optimization2(
    #     model=model, 
    #     kcat_dict=kcat_dict, 
    #     objective_reaction=objective_reaction,
    #     gene_sequences_dict=gene_sequences_dict,
    #     enzyme_upper_bound=enzyme_upper_bound, 
    #     enzyme_ratio=enzyme_ratio, 
    # )

    solution_value, df_FBA, gene_sequences_dict, m = run_optimization3(
        model = model,
        kcat_dict=kcat_dict,
        objective_reaction=objective_reaction,
        gene_sequences_dict=gene_sequences_dict,
        enzyme_upper_bound=enzyme_upper_bound,
        enzyme_ratio=True
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
        Suffix,
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
    solver = SolverFactory('glpk')
    # solver.options['max_iter'] = 100
    
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

def validate_enzyme_constraints(df_FBA,
                                kcat_dict_hr,
                                gene_sequences_dict,
                                reaction_gene_list,
                                gpr_dict,
                                enzyme_ratio,            # True or False
                                enzyme_upper_bound,      # the same upper bound you passed
                                enzyme_mw_dict,          # {gene: molecular_weight_in_dalton}
                                S_mat):
    """
    Given the DataFrame df_FBA (with columns ['Variable','Index','Value'])
    and the same dictionaries used inside run_optimization, check:
      1) For every (reaction, gene) that should be constrained, v_j ≤ kcat_j * e_i.
      2) Total‐enzyme constraint: sum(e_i * MW_i)*0.001 ≤ enzyme_upper_bound (if enzyme_ratio=True),
         or sum(e_i) ≤ enzyme_upper_bound (if enzyme_ratio=False).
      3) Steady‐state (S · v = 0) up to a small numerical tolerance.
    """

    import numpy as np

    # 1) Build solution dicts for v_j and e_i
    #    df_FBA rows look like: Variable='reaction', Index='R_EX_glc__D_e', Value=4.5
    flux_sol = {}   # reaction_id → flux_value
    enz_sol  = {}   # gene_id     → enzyme_amount

    for _, row in df_FBA.iterrows():
        varname = row['Variable']
        idx     = row['Index']
        val     = float(row['Value'])
        if varname == 'reaction':
            flux_sol[idx] = val
        elif varname == 'enzyme':
            enz_sol[idx] = val
    # 1a) Check that we really extracted them
    if not flux_sol:
        raise RuntimeError("Could not find any 'reaction' entries in df_FBA.")
    if not enz_sol:
        raise RuntimeError("Could not find any 'enzyme' entries in df_FBA.")

    # 2) Check each (reaction, gene) in reaction_gene_list
    violations = []
    for (j, i) in reaction_gene_list:
        # Was (j,i) supposed to be constrained?  It will be constrained if:
        #   • j in kcat_dict_hr (or in reaction.annotation['kcat']) AND
        #   • i in gene_sequences_dict
        #   • AND inside your rule_kcat pipeline, it did NOT fall into the “missing data → Feasible” case.
        #
        # For simplicity, assume: if j in kcat_dict_hr and i in gene_sequences_dict, then it should be constrained.
        if j not in kcat_dict_hr or i not in gene_sequences_dict or not gene_sequences_dict[i]:
            continue

        vj = flux_sol.get(j, 0.0)
        # there might be multiple possible kcats for this reaction—your code picks the first (or does a max).
        # Here we assume kcat_dict_hr[j] is a float or a one-element list.
        kc = kcat_dict_hr[j]
        if isinstance(kc, list):
            kc_val = kc[0]
        else:
            kc_val = kc

        ei = enz_sol.get(i, 0.0)
        if vj > kc_val * ei + 1e-6:    # small tolerance for numerical solver noise
            violations.append((j, i, vj, kc_val*ei))

    if violations:
        print("FOUND violations of (v_j ≤ kcat·e_i):")
        for j, i, vj, bound in violations[:5]:
            print(f"  • Reaction {j}, gene {i}: flux {vj:.4g} > kcat·e = {bound:.4g}")
        print(f"...plus {len(violations)-5} more." if len(violations)>5 else "")
    else:
        print("All (reaction, gene) v_j ≤ kcat·e_i constraints are satisfied (within 1e-6 tolerance).")

    # 3) Check total‐enzyme constraint
    if enzyme_ratio:
        # Reconstruct: total_weight_grams_per_gDCW = ∑ (e_i [mmol/gDCW] * MW_i [g/mol]) * 1e−3 (to get g/gDCW).
        total_weight = 0.0
        for gene, ei in enz_sol.items():
            mw = enzyme_mw_dict.get(gene, 1e5)  # fallback if missing
            total_weight += ei * mw * 1e-3
        if total_weight > enzyme_upper_bound + 1e-6:
            print(f"Total‐enzyme weight constraint violated: {total_weight:.6f} > {enzyme_upper_bound:.6f}")
        else:
            print(f"Total‐enzyme weight = {total_weight:.6f} ≤ {enzyme_upper_bound:.6f} (OK)")

    else:
        total_e = sum(enz_sol.values())
        if total_e > enzyme_upper_bound + 1e-6:
            print(f"Total‐enzyme mmol constraint violated: {total_e:.6f} > {enzyme_upper_bound:.6f}")
        else:
            print(f"Sum(e_i) = {total_e:.6f} ≤ {enzyme_upper_bound:.6f} (OK)")

    # 4) Check steady‐state: S·v ≈ 0 for every metabolite t
    ss_violations = []
    for (met_id, rxn_id), coeff in S_mat.items():
        # S_mat keys are tuples (met_id, rxn_id) → stoich.  We want ∑_j S[t,j]·v_j = 0
        # so accumulate into a per‐met residual.
        pass
    # Instead, build a metabolite → sum(S[t,j]*v_j)
    met_residual = {t: 0.0 for (t, _) in S_mat.keys()}
    for (t, j), coeff in S_mat.items():
        vj = flux_sol.get(j, 0.0)
        met_residual[t] += coeff * vj

    # Now check if any |residual| > tol
    for t, resid in met_residual.items():
        if abs(resid) > 1e-6:
            ss_violations.append((t, resid))

    if ss_violations:
        print("Steady‐state (S·v=0) violations (|residual| > 1e−6):")
        for t, resid in ss_violations[:5]:
            print(f"  • Metabolite {t}: ∑ S[{t},j]·v_j = {resid:.4g}")
        print(f"...plus {len(ss_violations)-5} more." if len(ss_violations)>5 else "")
    else:
        print("All metabolites satisfy steady‐state (S·v≈0).")

def convert_to_irreversible(model):
        """
        Convert all non-exchange reversible reactions to irreversible and ensure all exchange reactions
        have a reversible counterpart (create reverse reactions if needed).
        """
        # List to hold reactions to add
        reactions_to_add = []
        coefficients = {}
    
        # Convert only non-exchange reversible reactions to irreversible
        non_exchange_reactions = [rxn for rxn in model.reactions if rxn not in model.exchanges]
        exchange_reactions = [rxn for rxn in model.exchanges]
        print('Number of reactions that are non-exchange: ', len(non_exchange_reactions))
        print('Number of reactions that are exchange: ', len(exchange_reactions))
    
        for reaction in non_exchange_reactions:
            if reaction.reversibility:
                reverse_reaction_id = reaction.id + "_reverse"
                
                # Check if the reverse reaction already exists in the model
                if reverse_reaction_id not in [rxn.id for rxn in model.reactions]:
                    reverse_reaction = Reaction(reverse_reaction_id)
                    reverse_reaction.lower_bound = max(0, -reaction.upper_bound)
                    reverse_reaction.upper_bound = abs(reaction.lower_bound)
                    coefficients[reverse_reaction] = reaction.objective_coefficient * -1
    
                    # Modify the original reaction to be irreversible
                    reaction.lower_bound = max(0, reaction.lower_bound)
                    reaction.upper_bound = max(0, reaction.upper_bound)
    
                    # Create the reverse reaction metabolites with reversed stoichiometry
                    reaction_dict = {k: v * -1 for k, v in reaction._metabolites.items()}
                    reverse_reaction.add_metabolites(reaction_dict)
    
                    # Copy genes and GPR rule from the original reaction
                    reverse_reaction._model = reaction._model
                    reverse_reaction._genes = reaction._genes
                    reverse_reaction._gpr = reaction._gpr
    
                    for gene in reaction._genes:
                        gene._reaction.add(reverse_reaction)
    
                    # Add reverse reaction to the list
                    reactions_to_add.append(reverse_reaction)
        
        print('Number of reactions being added from non-exchange:', len(reactions_to_add))
        # Ensure all exchange reactions are reversible by creating reverse reactions
        for exchange_reaction in model.exchanges:
            reverse_exchange_id = exchange_reaction.id + "_reverse"
            
            # Check if the reverse reaction already exists in the model
            if reverse_exchange_id not in [rxn.id for rxn in model.reactions]:
                # Create a reverse reaction for exchange reactions without reversible behavior
                reverse_exchange = Reaction(reverse_exchange_id)
                reverse_exchange.lower_bound = 0
                reverse_exchange.upper_bound = -exchange_reaction.lower_bound
    
                # Reverse the metabolites in the exchange reaction (flip stoichiometry)
                reverse_metabolites = {met: -coeff for met, coeff in exchange_reaction.metabolites.items()}
                reverse_exchange.add_metabolites(reverse_metabolites)
    
                # Copy the GPR rule (if any) from the original exchange reaction
                reverse_exchange.gene_reaction_rule = exchange_reaction.gene_reaction_rule
    
                # Add reverse exchange reaction to the list
                reactions_to_add.append(reverse_exchange)
        
        print('Number of reactions being added from exchange:', len(reactions_to_add))
        # Add the newly created reverse reactions to the model
        model.add_reactions(reactions_to_add)
    
        # Set new objective with the added reverse reactions
        set_objective(model, coefficients, additive=True)
    
        return model


def build_pyomo_fba(model, objective_reaction, sense='max', convert_irreversible=False):
    """
    Build and return a simple FBA Pyomo model from a COBRApy model.

    Parameters
    ----------
    model : cobra.Model
        A COBRA model (reversible or irreversible).
    objective_reaction : str
        ID of the reaction to optimize.
    sense : {'max','min'}, optional
        Whether to maximize or minimize the objective. Default 'max'.
    convert_irreversible : bool, optional
        If True, convert a reversible model to irreversible using COBRApy's built-in function.

    Returns
    -------
    pm : pyomo.environ.ConcreteModel
        A Pyomo model with:
          - pm.v[j] ∈ ℝ for each reaction j
          - pm.set_bound[j]: lb ≤ v[j] ≤ ub constraints
          - pm.mass_balance[m]: ∑ S[m,j]·v[j] = 0 for internal metabolites m
          - pm.obj: objective on v[objective_reaction]
    """
    # Optionally convert
    if convert_irreversible:
        model = cobra_convert(model)

    # Reaction and metabolite lists
    reactions = [r.id for r in model.reactions]
    metabolites = [m.id for m in model.metabolites]

    # Validate objective
    if objective_reaction not in reactions:
        raise KeyError(f"Objective reaction '{objective_reaction}' not found in model.")

    # Stoichiometry dict S[(met,rx)] = coeff
    S = {(met.id, rx.id): coeff
         for rx in model.reactions
         for met, coeff in rx.metabolites.items() if coeff != 0}

    # Bounds dicts
    lb = {rx.id: rx.lower_bound for rx in model.reactions}
    ub = {rx.id: rx.upper_bound for rx in model.reactions}

    # Identify biomass metabolites to exclude from mass-balance
    obj_rxn = model.reactions.get_by_id(objective_reaction)
    biomass_mets = {m.id for m, c in obj_rxn.metabolites.items() if c != 0}

    # Identify boundary metabolites (in any exchange reaction)
    external_mets = set()
    for ex in model.exchanges:
        external_mets.update(m.id for m in ex.metabolites.keys())

    # Internal metabolites for steady-state
    internal_mets = [m for m in metabolites
                     if m not in biomass_mets and m not in external_mets]

    # Build Pyomo model
    pm = ConcreteModel()
    pm.v = Var(reactions, within=Reals)

    # Flux bounds constraints
    def bound_rule(m, j):
        return inequality(lb[j], m.v[j], ub[j])
    pm.set_bound = Constraint(reactions, rule=bound_rule)

    # Mass-balance constraints
    def mb_rule(m, met):
        return sum(S.get((met, rx), 0) * m.v[rx] for rx in reactions) == 0
    pm.mass_balance = Constraint(internal_mets, rule=mb_rule)

        # Objective: replicate COBRApy's built-in objective
    # Use each reaction's objective_coefficient
    obj_coefs = {rx.id: rx.objective_coefficient for rx in model.reactions}
    def obj_rule(m):
        return sum(obj_coefs[j] * m.v[j] for j in reactions)
    if sense == 'max':
        pm.obj = Objective(rule=obj_rule, sense=maximize)
    else:
        pm.obj = Objective(rule=obj_rule, sense=minimize)

    return pm


def solve_pyomo_fba(pm, solver_name='glpk'):
    """
    Solve the Pyomo FBA model and return objective value and fluxes.

    Returns
    -------
    biomass : float
    fluxes : dict of {reaction_id: flux_value}
    """
    solver = SolverFactory(solver_name)
    result = solver.solve(pm, tee=False)
    from pyomo.environ import SolverStatus, TerminationCondition
    if not (result.solver.status == SolverStatus.ok and
            result.solver.termination_condition == TerminationCondition.optimal):
        raise RuntimeError(
            f"Solver failed: {result.solver.status}, "
            f"{result.solver.termination_condition}"
        )

    biomass = value(pm.obj)
    fluxes = {r: value(pm.v[r]) for r in pm.v}
    return biomass, fluxes

def run_linprog_fba(model, objective_reaction, reversible=False):
    """
    Perform FBA via scipy.optimize.linprog on a COBRA model.

    Parameters
    ----------
    model : cobra.Model
        A COBRA model (reversible or irreversible).
    objective_reaction : str
        Reaction ID to maximize.
    reversible : bool, optional
        If True and model is reversible, keep negative bounds; otherwise assume irreversible.

    Returns
    -------
    biomass : float
        Optimal value of the objective reaction flux.
    fluxes : dict
        Mapping reaction ID to flux value.
    """
    # Build S matrix (met x rxn)
    mets = [m.id for m in model.metabolites]
    rxns = [r.id for r in model.reactions]
    M = len(mets); N = len(rxns)

    S = np.zeros((M, N))
    for i, m in enumerate(model.metabolites):
        for j, r in enumerate(model.reactions):
            coeff = r.get_coefficient(m.id)
            if coeff is None:
                continue
            S[i, j] = coeff

    # Determine internal metabolites (exclude biomass and exchange)
    # biomass metabolites
    obj = model.reactions.get_by_id(objective_reaction)
    biomass_mets = {m.id for m, c in obj.metabolites.items() if c != 0}
    # external metabolites
    external_mets = set()
    for ex in model.exchanges:
        external_mets.update(m.id for m in ex.metabolites.keys())
    internal_idx = [i for i, m in enumerate(mets)
                    if m not in biomass_mets and m not in external_mets]

    # Build A_eq and b_eq: S_internal * v = 0
    A_eq = S[internal_idx, :]
    b_eq = np.zeros(len(internal_idx))

    # Objective c: maximize flux of objective_reaction => minimize -flux
    c = np.zeros(N)
    obj_idx = rxns.index(objective_reaction)
    c[obj_idx] = -1.0

    # Bounds for each v_j
    bounds = []
    for r in model.reactions:
        lb = r.lower_bound if reversible else max(0, r.lower_bound)
        ub = r.upper_bound
        bounds.append((lb, ub))

    # Solve LP via HiGHS
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f"Linprog failed: {res.message}")

    flux_vals = res.x
    biomass = flux_vals[obj_idx]
    fluxes = {rxns[j]: flux_vals[j] for j in range(N)}
    return biomass, fluxes
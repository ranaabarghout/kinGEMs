"""
Optimization module for kinGEMs.

This module provides functions for enzyme-constrained flux balance analysis,
including core optimization functionality from the original KG03b module.
"""

import os
import pandas as pd
import cobra as cb
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import math
import numpy as np
import re
from Bio.SeqUtils import molecular_weight

from ..config import (
    ensure_dir_exists
)

def run_optimization(model, kcat_dict, objective_reaction, gene_sequences_dict=None, 
                    enzyme_upper_bound=0.125, enzyme_ratio=True, maximization=True, 
                    multi_enzyme_off=False, isoenzymes_off=False, 
                    promiscuous_off=False, complexes_off=False):
    """
    Run enzyme-constrained flux balance analysis.
    
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
        except:
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
        except:
            raise ValueError(f"Could not load gene sequences from: {gene_sequences_dict}")
    
    # kcat_dict handling
    if isinstance(kcat_dict, str):
        try:
            df_kcat = pd.read_csv(kcat_dict)
            kcat_dict = df_kcat.set_index('Key').to_dict()['Value']
        except:
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
        
        # kcat dict
        kcat_value = (reaction.annotation).get('kcat')
        if kcat_value is not None:
            if isinstance(kcat_value, list):  # For multiple kcat
                kcat_list = [float(value) for value in kcat_value]
                if kcat_list:
                    kcat[reaction.id] = kcat_list
            else:  # For single kcats
                try: 
                    single_kcat = float(kcat_value)
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
            except:
                pass
            else:
                S_mat[met.id, reaction.id] = reaction.get_coefficient(met.id)

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
    Concretemodel = ConcreteModel()
    Concretemodel.reaction = Var(reactions, within=NonNegativeReals, bounds=(0, 999)) # Flux - mmol/gDCW/hr
    Concretemodel.enzyme = Var(genes, within=NonNegativeReals) # mmol/gDCW
    Concretemodel.enzyme_set = Set(initialize=genes)
    Concretemodel.enzyme_min = Var(reaction_gene_tuple, within=NonNegativeReals, initialize=0) # mmol/gDCW

    # OBJECTIVE FUNCTION: maximizing or minimizing reaction
    if maximization:
        def rule_obj(m, objective_var):
            return m.reaction[objective_var]
        Concretemodel.objective = Objective(rule=rule_obj(Concretemodel, objective_reaction), sense=maximize)
    else:
        def rule_obj(m, objective_var):
            return m.reaction[objective_var]
        Concretemodel.objective = Objective(rule=rule_obj(Concretemodel, objective_reaction), sense=minimize)

    # CONSTRAINT: steady state
    def rule_S_mat(m, t):
        return sum(S_mat[t, j] * m.reaction[j] for j in reactions if (t, j) in S_mat.keys()) == 0
    Concretemodel.set_S_mat = Constraint(metabolites, rule=rule_S_mat)

    # CONSTRAINT: flux bounds
    def rule_bounds(m, j):
        return inequality(lower_bounds[j], m.reaction[j], upper_bounds[j])
    Concretemodel.rxn_bounds = Constraint(reactions, rule=rule_bounds)

    # CONSTRAINT: minimum enzyme concentration
    def enzyme_min_constraint(m, j, i):
        if j in gpr:
            gpr_string = gpr[j]
            if 'and' in gpr_string:
                return m.enzyme_min[j, i] <= m.enzyme[i]
            else:
                return Constraint.Feasible
        else:
            return Constraint.Feasible
    
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
            except:
                pass
            
        # isozymes
        if not isoenzymes_off: 
            if 'or' in gpr_string:
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
                return m.reaction[j] <= mean_kcat * m.enzyme_min[j, i]
        else:
            if 'and' in gpr_string:
                return m.reaction[j] <= 1000    

    # CONSTRAINT: enzyme kinetics
    def rule_kcat(m, j, i):
        if (j, i) in single_enzyme and j in kcat:
            if not (math.isnan(kcat[j][0]) or kcat[j][0] is None):
                single_enzyme_pass.append(j)
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
            return Constraint.Feasible

    Concretemodel.set_kcat = Constraint(reaction_gene_tuple, rule=rule_kcat)

    # CONSTRAINT: promiscuous enzymes
    def rule_promiscuous_E(m, i):
        try:
            return max(m.reaction[j]/kcat_dict[j, i][0] for j in reactions) <= m.enzyme[i]
        except:
            return Constraint.Feasible
            
    if not promiscuous_off:  
        Concretemodel.set_promiscuous_E = Constraint(genes, rule=rule_promiscuous_E)
    
    # FUNCTION for retrieving molecular weights from protein sequences
    def calculate_molecular_weight(sequence):
        return molecular_weight(sequence, seq_type='protein')
    
    # CONSTRAINT total enzyme (2 variants)
    if enzyme_ratio:  
        # VARIABLE
        Concretemodel.E_ratio = Var(within=NonNegativeReals, bounds=(0, enzyme_upper_bound)) # gP/gDCW
        Concretemodel.enzyme_molecular_weights = Param(
            Concretemodel.enzyme_set, 
            initialize={gene: calculate_molecular_weight(gene_sequences_dict.get(gene, '')) 
                       if gene in gene_sequences_dict else None for gene in genes}
        )
        
        # CONSTRAINT
        def rule_E_total(m):
            total_enzyme_weight_expr = sum(m.enzyme[i] * (m.enzyme_molecular_weights[i] 
                                          if m.enzyme_molecular_weights[i] is not None else 0) 
                                          for i in m.enzyme) * 0.001
            return total_enzyme_weight_expr <= m.E_ratio
        Concretemodel.set_E_total = Constraint(rule=rule_E_total)
        
    else:
        # VARIABLE
        Concretemodel.E_total = Var(within=NonNegativeReals, bounds=(0, enzyme_upper_bound)) # mmol/gDCW
        
        # CONSTRAINT
        def rule_E_total(m):
            return sum(m.enzyme[i] for (j, i) in reaction_gene_tuple) <= m.E_total
        Concretemodel.set_E_total = Constraint(rule=rule_E_total) 
    
    # Solving
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 500
    
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
        else:
            # Handle unsuccessful optimization
            raise ValueError(f"Solver did not find an optimal solution. Status: {results.solver.status}, "
                            f"Termination condition: {results.solver.termination_condition}")
    
    except ValueError as e:
        # Handle exceptions raised by the solver
        print(f"An error occurred during optimization: {e}")
        df_FBA = pd.DataFrame()  # Return an empty DataFrame
        solution_value = None
            
    return solution_value, df_FBA, gene_sequences_dict
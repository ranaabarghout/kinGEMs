"""
Optimization module for kinGEMs.

This module provides functions for enzyme-constrained flux balance analysis,
including core optimization functionality from the original KG03b module.
"""

from collections import Counter  # noqa: F401
from itertools import product
import math
import os
import re

from Bio.Data.IUPACData import protein_letters
from Bio.SeqUtils import molecular_weight
import cobra as cb
from cobra.util.array import create_stoichiometric_matrix
import numpy as np  # noqa: F401

# Troubleshooting infeasible optimization
# Add this code before running your optimization to diagnose the issue
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *  # noqa: F403
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from ..config import ensure_dir_exists  # noqa: F401


def _tokenize_gpr(rule):
    """Split a GPR rule into tokens: parentheses, 'and', 'or', gene IDs."""
    token_spec = r'\(|\)|and|or|[^()\s]+'
    return re.findall(token_spec, rule.lower())


def _parse_gpr_to_dnf(tokens):
    """
    Parse tokens into Disjunctive Normal Form (DNF):
    returns a list of clauses, each clause is a list of gene IDs.
    e.g. "g1 and (g2 or g3)" -> [['g1','g2'], ['g1','g3']]
    """
    def parse_expression(idx=0):
        clauses, idx = parse_term(idx)
        while idx < len(tokens) and tokens[idx] == 'or':
            right, idx = parse_term(idx+1)
            clauses += right
        return clauses, idx

    def parse_term(idx):
        clauses, idx = parse_factor(idx)
        while idx < len(tokens) and tokens[idx] == 'and':
            right, idx = parse_factor(idx+1)
            clauses = [c1 + c2 for c1, c2 in product(clauses, right)]
        return clauses, idx

    def parse_factor(idx):
        tok = tokens[idx]
        if tok == '(':
            clauses, idx = parse_expression(idx+1)
            if tokens[idx] != ')':
                raise ValueError(f"Mismatched parentheses in GPR: {tokens}")
            return clauses, idx+1
        else:
            return [[tok]], idx+1

    dnf, _ = parse_expression(0)
    # dedupe and sort
    unique = []
    for clause in dnf:
        cl = sorted(set(clause))
        if cl not in unique:
            unique.append(cl)
    return unique


# def run_optimization4(
#     model,
#     kcat_dict,
#     objective_reaction,
#     gene_sequences_dict=None,
#     enzyme_upper_bound=0.125,
#     enzyme_ratio=True,
#     maximization=True,
#     solver_name='gurobi',
#     tee=False,
# ):
#     """
#     Enzyme-constrained FBA via Pyomo, handling:
#       - single enzymes (max kcat)
#       - enzyme complexes (avg kcat)
#       - isoenzymes (OR-GPR)
#       - promiscuous enzymes
#     Returns: sol_val, df_FBA, gene_sequences_dict, model
#     """
#     import logging
#     import time  # Add this import

#     import gurobipy as gp

#     # 1) Silence the gurobipy Python logger
#     logging.getLogger('gurobipy').setLevel(logging.ERROR)

#     # 2) Globally mute Pyomo solver chatter
#     logging.getLogger('pyomo').setLevel(logging.WARNING)
    
#     total_start = time.time()
    
#     # 1) Load COBRA model
#     # print("Step 1: Loading COBRA model...")
#     step_start = time.time()
#     if isinstance(model, str):
#         mod = (
#             cb.io.read_sbml_model(model)
#             if model.endswith(('.xml', '.sbml'))
#             else cb.io.load_json_model(model)
#         )
#     else:
#         mod = model
#     # print(f"Model loaded in {time.time() - step_start:.2f}s\n")

#     # 2) Initial flux guess
#     # print("Step 2: Getting initial flux guess...")
#     step_start = time.time()
#     mod.objective = objective_reaction
#     print("Model objective:", mod.objective)
#     cobra_sol = mod.optimize()
#     flux0 = cobra_sol.fluxes.to_dict()
#     # print(f"Initial optimization completed in {time.time() - step_start:.2f}s\n")

#     # 3) Load & normalize kcat_dict
#     # print("Step 3: Processing kcat dictionary...")
#     step_start = time.time()
#     if isinstance(kcat_dict, str):
#         df = pd.read_csv(kcat_dict)
#         tmp = {}
#         for r, g, k in zip(df.reaction, df.gene, df.kcat):
#             tmp.setdefault((r, g), []).append(k)
#         kcat_dict = tmp
#     # convert all values to hr^-1 lists
#     for key, vals in list(kcat_dict.items()):
#         kcat_dict[key] = [(v * 3600 if v < 1000 else v) for v in vals]

#     kcat_dict = {
#     (rxn_id.lower(), gene_id.lower()): ks
#     for (rxn_id, gene_id), ks in kcat_dict.items()
#     }
#     # print(f"kcat dictionary processed ({len(kcat_dict)} entries) in {time.time() - step_start:.2f}s\n")

#     # DEBUG: show how many (reaction,gene) pairs we have, and print a few examples
#     print(f"\nLoaded {len(kcat_dict)} (reaction,gene) kcat entries. Sample:")
#     for i,(rg, ks) in enumerate(kcat_dict.items()):
#         print(f"  {rg} → {ks}")
#         if i >= 9:
#             break
#     print()

#     # 4) Build stoichiometry, bounds, objective
#     # print("Step 4: Building model data structures...")
#     step_start = time.time()
#     S = create_stoichiometric_matrix(mod)
#     mets = [m.id for m in mod.metabolites]
#     rxns = [r.id for r in mod.reactions]
#     genes = [g.id for g in mod.genes]
#     lb = {r.id: r.lower_bound for r in mod.reactions}
#     ub = {r.id: r.upper_bound for r in mod.reactions}
#     obj_coef = {r.id: (1.0 if r.id == objective_reaction else 0.0) for r in mod.reactions} #obj_coef = {r.id: r.objective_coefficient for r in mod.reactions}
#     met_index = {m: i for i, m in enumerate(mets)}
#     rxn_index = {r: j for j, r in enumerate(rxns)}
#     # print(f"Data structures built ({len(mets)} mets, {len(rxns)} rxns, {len(genes)} genes) in {time.time() - step_start:.2f}s\n")

#     # 5) Generate DNF clauses
#     # print("Step 5: Parsing gene-protein-reaction rules...")
#     step_start = time.time()
#     dnf_clauses = {}
#     for r in mod.reactions:
#         rule = (r.gene_reaction_rule or '').strip()
#         if not rule:
#             continue
#         tokens = _tokenize_gpr(rule)
#         dnf_clauses[r.id] = _parse_gpr_to_dnf(tokens)

#     print(dnf_clauses)
#     # print(f"GPR rules parsed ({len(dnf_clauses)} reactions with GPRs) in {time.time() - step_start:.2f}s\n")

#     # --- DEBUG INTERSECTION OF GPRs & kcats ----------------------------------

#     # all reactions that have at least one GPR clause
#     gpr_rxns = set(dnf_clauses.keys())

#     # all (reaction,gene) keys in your kcat_dict
#     kcat_keys = set(kcat_dict.keys())

#     # find the overlap
#     valid_pairs = [pair for pair in kcat_keys if pair[0] in gpr_rxns]
#     print(f"\nFound {len(valid_pairs)} (reaction,gene) pairs where you have BOTH a GPR and a kcat.")
#     print("Sample valid pairs:")
#     for rg in valid_pairs[:10]:
#         print("  ", rg, "→", kcat_dict[rg])
#     print()  
#     # ------------------------------------------------------------------------



#     # 6) Build Pyomo model
#     # print("Step 6: Building Pyomo optimization model...")
#     step_start = time.time()
#     m = ConcreteModel()  # noqa: F405
#     m.M = Set(initialize=mets)  # noqa: F405
#     m.R = Set(initialize=rxns)  # noqa: F405
#     m.G = Set(initialize=genes)  # noqa: F405
#     m.K = Set(initialize=[(r, g) for r in rxns for g in genes], dimen=2)  # noqa: F405
#     # print(f"Sets created in {time.time() - step_start:.2f}s\n")

#     # Variables
#     substep_start = time.time()
#     m.v = Var(  # noqa: F405
#         m.R,
#         domain=Reals,  # noqa: F405
#         bounds=lambda mo, j: (lb[j], ub[j]),
#         initialize=lambda mo, j: flux0.get(j, 0.0)
#     )
#     m.E = Var(m.G, domain=NonNegativeReals, initialize=0.01)  # noqa: F405
#     # print(f"Variables created in {time.time() - substep_start:.2f}s\n")

#     # Mass balance
#     # substep_start = time.time()
#     # def mass_balance(mo, met):
#     #     i = met_index[met]
#     #     return sum(S[i, rxn_index[r]] * mo.v[r] for r in mo.R) == 0
#     # m.mass_balance = Constraint(m.M, rule=mass_balance)  # noqa: F405
#     # print(f"Mass balance constraints created in {time.time() - substep_start:.2f}s\n")
    
#     # OPTIMIZED mass balance
#     substep_start = time.time()
    
#     # Pre-compute which reactions involve each metabolite
#     met_reactions = {}
#     for met in mets:
#         met_reactions[met] = []
#         for rxn in rxns:
#             if S[met_index[met], rxn_index[rxn]] != 0:
#                 met_reactions[met].append(rxn)

#     # print(f"Sparse matrix analysis: avg {sum(len(v) for v in met_reactions.values()) / len(mets):.1f} reactions per metabolite")

#     def mass_balance_sparse(mo, met):
#         # Only iterate over reactions that involve this metabolite
#         relevant_rxns = met_reactions[met]
#         if not relevant_rxns:
#             return Constraint.Feasible  # noqa: F405
        
#         i = met_index[met]
#         return sum(S[i, rxn_index[r]] * mo.v[r] for r in relevant_rxns) == 0

#     m.mass_balance = Constraint(m.M, rule=mass_balance_sparse)  # noqa: F405
#     # print(f"Mass balance constraints created in {time.time() - substep_start:.2f}s\n")
    
#     # Objective
#     sense = maximize if maximization else minimize  # noqa: F405
#     m.obj = Objective(expr=sum(obj_coef[r] * m.v[r] for r in m.R), sense=sense)  # noqa: F405

#     # 6a) AND‐GPR: single or complex
#     substep_start = time.time()
#     def and_rule(mo, rxn_id, gene_id):
#         clauses = dnf_clauses.get(rxn_id, [])
#         if not clauses:
#             return Constraint.Skip  # noqa: F405
#         # single enzyme: max kcat
#         if len(clauses) == 1 and len(clauses[0]) == 1:
#             g = clauses[0][0]
#             if gene_id != g:
#                 return Constraint.Skip  # noqa: F405
#             k_list = kcat_dict.get((rxn_id.lower(), g.lower()), [])
#             if not k_list:
#                 return Constraint.Skip  # noqa: F405
#             k_val = max(k_list)
#             print(f"[kcat‐AND] {rxn_id}:{g} → k_val = {k_val}")
#             return mo.v[rxn_id] <= k_val * mo.E[g]
#         # enzyme complex: avg kcat
#         if len(clauses) == 1 and len(clauses[0]) > 1:
#             clause = clauses[0]
#             if gene_id not in clause:
#                 return Constraint.Skip  # noqa: F405
#             all_ks = []
#             for g in clause:
#                 all_ks.extend(kcat_dict.get((rxn_id, g), []))
#             if not all_ks:
#                 return Constraint.Skip  # noqa: F405
#             k_val = sum(all_ks) / len(all_ks)
#             return mo.v[rxn_id] <= k_val * mo.E[gene_id]
#         return Constraint.Skip  # noqa: F405
#     m.kcat_and = Constraint(m.K, rule=and_rule)  # noqa: F405
#     print(f"  → AND‐GPR constraints added: {len(m.kcat_and)}")
#     if len(m.kcat_and) > 0:
#         # grab the first few keys
#         sample_keys = list(m.kcat_and.keys())[:5]
#         print("    Sample AND constraint expressions:")
#         for key in sample_keys:
#             c = m.kcat_and[key]
#             print(f"      {key} : {c.expr}")   # prints the actual inequality
#     # print(f"AND-GPR constraints created in {time.time() - substep_start:.2f}s\n")

#     # 6b) OR‐GPR: isoenzymes
#     substep_start = time.time()
#     def iso_rule(mo, rxn_id):
#         clauses = dnf_clauses.get(rxn_id, [])
#         if len(clauses) <= 1:
#             return Constraint.Skip  # noqa: F405
#         terms = []
#         for clause in clauses:
#             ks = []
#             for g in clause:
#                 kl = kcat_dict.get((rxn_id.lower(), g.lower()), [])
#                 if kl:
#                     ks.append(min(kl))
#             if not ks:
#                 continue
#             kmin = min(ks)
#             for g in clause:
#                 terms.append(kmin * mo.E[g])
#         if not terms:
#             return Constraint.Skip  # noqa: F405
#         return mo.v[rxn_id] <= sum(terms)
#     m.kcat_iso = Constraint(m.R, rule=iso_rule)  # noqa: F405
#     print(f"  → OR‐GPR (isoenzyme) constraints added: {len(m.kcat_iso)}")
#     if len(m.kcat_iso) > 0:
#         sample_rxns = list(m.kcat_iso.keys())[:5]
#         print("    Sample OR constraint expressions:")
#         for rxn in sample_rxns:
#             c = m.kcat_iso[rxn]
#             print(f"      {rxn} : {c.expr}")

#     # print(f"OR-GPR constraints created in {time.time() - substep_start:.2f}s\n")

#     # 6c) Promiscuous enzymes
#     substep_start = time.time()
#     def promis_rule(mo, g_id):
#         usage = []
#         for r_id, clauses in dnf_clauses.items():
#             for clause in clauses:
#                 if g_id not in clause:
#                     continue
#                 for k in kcat_dict.get((r_id.lower(), g_id.lower()), []):
#                     usage.append(mo.v[r_id] / k)
#         if not usage:
#             return Constraint.Skip  # noqa: F405
#         return sum(usage) <= mo.E[g_id]
#     m.promiscuous = Constraint(m.G, rule=promis_rule)  # noqa: F405
#     print(f"  → Promiscuous enzyme constraints added: {len(m.promiscuous)}")
#     if len(m.promiscuous) > 0:
#         sample_genes = list(m.promiscuous.keys())[:5]
#         print("    Sample PROM constraint expressions:")
#         for g in sample_genes:
#             c = m.promiscuous[g]
#             print(f"      {g} : {c.expr}")
#     # print(f"Promiscuous enzyme constraints created in {time.time() - substep_start:.2f}s\n")

#     # 7) Total enzyme pool / ratio
#     # print("Step 7: Adding enzyme pool constraints...")
#     step_start = time.time()
#     if enzyme_ratio:
#         if gene_sequences_dict is None:
#             gene_sequences_dict = {}
#         # mw = {g: (molecular_weight(gene_sequences_dict.get(g, ''), seq_type='protein') or 1e5)
#         #       for g in genes}
#         def safe_mw(seq):
#             # remove any non-standard letters (e.g. X, B, Z, etc.)
#             clean = "".join([aa for aa in seq if aa in protein_letters])
#             try:
#                 return molecular_weight(clean or "A", seq_type='protein')
#             except ValueError:
#                 # fallback if something still weird
#                 return 1e5
#         mw = {g: safe_mw(gene_sequences_dict.get(g, "")) for g in genes}
#         m.E_ratio = Var(domain=NonNegativeReals, bounds=(0, enzyme_upper_bound))  # noqa: F405
#         m.total_enzyme = Constraint(  # noqa: F405
#             expr=sum(m.E[g] * mw[g] for g in m.G) * 1e-3 <= m.E_ratio
#         )
#         print(f"  → Total enzyme pool constraint added: 1")
#     else:
#         m.E_total = Var(domain=NonNegativeReals, bounds=(0, enzyme_upper_bound))  # noqa: F405
#         m.total_enzyme = Constraint(expr=sum(m.E[g] for g in m.G) <= m.E_total)  # noqa: F405
#         print(f"  → Total enzyme pool constraint added: 1")
#     # print(f"Enzyme pool constraints added in {time.time() - step_start:.2f}s\n")

#     # print(f"\nMODEL BUILDING COMPLETED in {time.time() - total_start:.2f}s")

#     # 8) Solve
#     # print("Step 8: Setting up and running solver...")
#     solver = SolverFactory(solver_name)
#     solver.options['LogToConsole'] = 0
#     solver.options['OutputFlag']   = 0
#     solver.options['LogFile']      = ''
#     #solver.options['threads'] = 4
#     # print("Solver:", solver)
#     # print(f"Solver options: {dict(solver.options)}")
#     solver.solve(m, tee=tee, load_solutions=True)

#     # 9) Post-process
#     for r in m.R:
#         val = m.v[r].value if m.v[r].value is not None else lb[r]
#         m.v[r].value = max(min(val, ub[r]), lb[r])
#     for g in m.G:
#         if m.E[g].value is None:
#             m.E[g].value = 0.0

#     # 10) Collect results
#     sol_val = value(m.obj)  # noqa: F405
#     records = [('flux', r, m.v[r].value) for r in m.R]
#     records += [('enzyme', g, m.E[g].value) for g in m.G]
#     df_FBA = pd.DataFrame(records, columns=['Variable','Index','Value'])

#     return sol_val, df_FBA, gene_sequences_dict, m

def run_optimization4(
    model,
    kcat_dict,
    objective_reaction,
    gene_sequences_dict=None,
    enzyme_upper_bound=0.125,
    enzyme_ratio=True,
    maximization=True,
    solver_name='gurobi',
    tee=False,
):
    """
    Enzyme-constrained FBA via Pyomo, handling:
      - single enzymes (max kcat)
      - enzyme complexes (avg kcat)
      - isoenzymes (OR-GPR)
      - promiscuous enzymes
    Returns: sol_val, df_FBA, gene_sequences_dict, model
    """
    import logging
    import time

    from Bio.Data.IUPACData import protein_letters
    from Bio.SeqUtils import molecular_weight
    from cobra import io as cbio
    import pandas as pd
    from pyomo.environ import (
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        Reals,
        Set,
        SolverFactory,
        Var,
        maximize,
        minimize,
        value,
    )

    def _tokenize_gpr(gpr):
        """Tokenize a GPR rule string into symbols and operators."""
        # ensure spacing around parentheses
        spaced = gpr.replace('(', ' ( ').replace(')', ' ) ')
        tokens = [tok.lower() for tok in spaced.split() if tok]
        return tokens


    def _parse_gpr_to_dnf(tokens):
        """Parse tokenized GPR into Disjunctive Normal Form (list of lists)."""
        # recursive descent parser
        def parse_expr(tok_list):
            terms = [parse_term(tok_list)]
            while tok_list and tok_list[0] == 'or':
                tok_list.pop(0)
                terms.append(parse_term(tok_list))
            if len(terms) == 1:
                return terms[0]
            return ('or', terms)

        def parse_term(tok_list):
            factors = [parse_factor(tok_list)]
            while tok_list and tok_list[0] == 'and':
                tok_list.pop(0)
                factors.append(parse_factor(tok_list))
            if len(factors) == 1:
                return factors[0]
            return ('and', factors)

        def parse_factor(tok_list):
            token = tok_list.pop(0)
            if token == '(':
                expr = parse_expr(tok_list)
                if tok_list and tok_list[0] == ')':
                    tok_list.pop(0)
                return expr
            return token

        def to_dnf(expr):
            if isinstance(expr, str):
                return [[expr]]
            op, args = expr
            if op == 'and':
                dnfs = [to_dnf(arg) for arg in args]
                result = dnfs[0]
                for sub in dnfs[1:]:
                    result = [r + s for r in result for s in sub]
                return result
            if op == 'or':
                result = []
                for sub in args:
                    result.extend(to_dnf(sub))
                return result
            return []

        # work on a copy so original tokens remain intact
        tree = parse_expr(tokens[:])
        return to_dnf(tree)


    # 1) Load model
    if isinstance(model, str):
        mod = (
            cbio.read_sbml_model(model)
            if model.endswith(('.xml', '.sbml'))
            else cbio.load_json_model(model)
        )
    else:
        mod = model

    # 2) Initial flux guess
    mod.objective = objective_reaction
    print("Model objective:", mod.objective)
    flux0 = mod.optimize().fluxes.to_dict()

    # 3) Load & normalize kcat_dict
    if isinstance(kcat_dict, str):
        df_k = pd.read_csv(kcat_dict)
        tmp = {}
        for r, g, k in zip(df_k.reaction, df_k.gene, df_k.kcat):
            tmp.setdefault((r, g), []).append(k)
        kcat_dict = tmp
    # convert to s^-1
    for key, vals in list(kcat_dict.items()):
        kcat_dict[key] = [(v * 3600 if v < 1000 else v) for v in vals]
    # lowercase keys
    kcat_dict = { (r.lower(), g.lower()): ks for (r, g), ks in kcat_dict.items() }
    # print(f"\nLoaded {len(kcat_dict)} kcat entries. Sample:")
    # for i, ((r, g), ks) in enumerate(kcat_dict.items()):
    #     print(f"  {(r, g)} -> {ks}")
    #     if i >= 4: break
    # print()

    # 4) Stoichiometry, bounds, objective
    S = create_stoichiometric_matrix(mod)
    mets = [m.id for m in mod.metabolites]
    rxns = [r.id for r in mod.reactions]
    genes = [g.id for g in mod.genes]
    lb = {r.id: r.lower_bound for r in mod.reactions}
    ub = {r.id: r.upper_bound for r in mod.reactions}
    obj_coef = {r.id: (1.0 if r.id == objective_reaction else 0.0) for r in mod.reactions}
    met_index = {m: i for i, m in enumerate(mets)}
    rxn_index = {r: j for j, r in enumerate(rxns)}

    # allow mapping from lower‐case gene → original‐case gene
    gene_map = { g.lower(): g for g in genes }


    # 5) Parse & lowercase GPRs
    dnf_clauses = {}
    for r in mod.reactions:
        rule = (r.gene_reaction_rule or '').strip()
        if not rule:
            continue
        tokens = _tokenize_gpr(rule)
        clauses = _parse_gpr_to_dnf(tokens)
        dnf_clauses[r.id.lower()] = [[g.lower() for g in clause] for clause in clauses]
    # print(f"Parsed {len(dnf_clauses)} GPR rules.")

    # DEBUG: intersection of kcat & GPR
    valid_pairs = [(r,g) for (r,g) in kcat_dict if r in dnf_clauses]
    # print(f"Found {len(valid_pairs)} overlapping (rxn,gene) pairs. Sample: {valid_pairs[:5]}\n")

    # 6) Build Pyomo model
    m = ConcreteModel()
    m.M = Set(initialize=mets)
    m.R = Set(initialize=rxns)
    m.G = Set(initialize=genes)
    m.K = Set(initialize=[(r,g) for r in rxns for g in genes], dimen=2)

    m.v = Var(
        m.R, domain=Reals,
        bounds=lambda mo,j: (lb[j], ub[j]),
        initialize=lambda mo,j: flux0.get(j, 0.0)
    )
    m.E = Var(m.G, domain=NonNegativeReals, initialize=0.01)

    # mass balances
    met_reactions = { met: [rxn for rxn in rxns if S[met_index[met], rxn_index[rxn]] != 0] for met in mets }
    def mb_rule(mo, met):
        rxn_list = met_reactions[met]
        if not rxn_list:
            return Constraint.Feasible
        i = met_index[met]
        return sum(S[i, rxn_index[r]] * mo.v[r] for r in rxn_list) == 0
    m.mass_balance = Constraint(m.M, rule=mb_rule)

    # objective
    sense = maximize if maximization else minimize
    m.obj = Objective(expr=sum(obj_coef[r] * m.v[r] for r in m.R), sense=sense)

    # AND-GPR
    def and_rule(mo, rxn_id, gene_id):
        clauses = dnf_clauses.get(rxn_id.lower(), [])
        if len(clauses)==1 and len(clauses[0])==1:
            g = clauses[0][0]
            if gene_id.lower()!=g:
                return Constraint.Skip
            ks = kcat_dict.get((rxn_id.lower(), g), [])
            if not ks:
                return Constraint.Skip
            return mo.v[rxn_id] <= max(ks) * mo.E[gene_id]
        return Constraint.Skip
    m.kcat_and = Constraint(m.K, rule=and_rule)
    # print(f" → AND-GPR constraints: {len(m.kcat_and)}")

    # OR-GPR
    def or_rule(mo, rxn_id):
        clauses = dnf_clauses.get(rxn_id.lower(), [])
        if len(clauses)<=1:
            return Constraint.Skip
        terms = []
        for clause in clauses:
            ks_list = [min(kcat_dict.get((rxn_id.lower(), g), [])) for g in clause if kcat_dict.get((rxn_id.lower(), g), [])]
            if not ks_list:
                continue
            kmin = min(ks_list)
            for g_l in clause:
                orig = gene_map[g_l]
                terms.append(kmin * mo.E[orig])
        if not terms:
            return Constraint.Skip
        return mo.v[rxn_id] <= sum(terms)
    m.kcat_iso = Constraint(m.R, rule=or_rule)
    # print(f" → OR-GPR constraints: {len(m.kcat_iso)}")

    # Promiscuous
    def prom_rule(mo, g_id):
        usage = []
        for rxn_id, clauses in dnf_clauses.items():
            for clause in clauses:
                if g_id.lower() not in clause:
                    continue
                for k in kcat_dict.get((rxn_id, g_id.lower()), []):
                    usage.append(mo.v[rxn_id] / k)
        if not usage:
            return Constraint.Skip
        return sum(usage) <= mo.E[g_id]
    m.promiscuous = Constraint(m.G, rule=prom_rule)
    # print(f" → Promiscuous constraints: {len(m.promiscuous)}")

    # total enzyme pool
    def safe_mw(seq):
        clean = ''.join(aa for aa in seq if aa in protein_letters)
        if not clean:
            clean = 'A'
        try:
            return molecular_weight(clean, seq_type='protein')
        except:
            return 1e5
    mw = {g: safe_mw(gene_sequences_dict.get(g, '') or '') for g in genes}

    if enzyme_ratio:
        m.E_ratio = Var(domain=NonNegativeReals, bounds=(0, enzyme_upper_bound))
        m.total_enzyme = Constraint(expr=sum(m.E[g] * mw[g] for g in m.G) * 1e-3 <= m.E_ratio)
    else:
        m.E_total = Var(domain=NonNegativeReals, bounds=(0, enzyme_upper_bound))
        m.total_enzyme = Constraint(expr=sum(m.E[g] for g in m.G) <= m.E_total)

    # solve
    solver = SolverFactory(solver_name)
    solver.options.update({'LogToConsole': 0, 'OutputFlag': 0, 'LogFile': ''})
    solver.solve(m, tee=tee, load_solutions=True)

    # post-process
    for r in m.R:
        v = m.v[r].value
        m.v[r].value = max(min(v if v is not None else lb[r], ub[r]), lb[r])
    for g in m.G:
        if m.E[g].value is None:
            m.E[g].value = 0.0

    sol_val = value(m.obj)
    records = [('flux', r, m.v[r].value) for r in m.R] + [('enzyme', g, m.E[g].value) for g in m.G]
    df_FBA = pd.DataFrame(records, columns=['Variable','Index','Value'])

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

    solution_value, df_FBA, gene_sequences_dict, m = run_optimization4(
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
        Suffix,  # noqa: F401
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

import cobra
import numpy as np
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients

def apply_experimental_fluxes(model: cobra.Model, 
                              experimental_flux_dict: dict,
                              stdev_factor: float = None,
                              verbose: bool = True,
                              experimental_bounds: dict = None,
                              ):
    """
    Apply experimental fluxes as bounds to a model.
    
    Args:
        model: Model to apply experimental fluxes as bounds
        experimental_flux_dict: Dictionary of experimental fluxes
        stdev_factor: Factor to multiply the experimental flux by to get the bounds. Optional
        experimental_bounds: Dictionary of experimental upper and lower bounds. Optional
        
    Returns:
        model: Model with experimental fluxes applied as bounds
    """
    for rxn_id, v in experimental_flux_dict.items():
        try:
            # Get the specific reaction object from the model
            reaction = model.reactions.get_by_id(rxn_id)
            
            original_lower = reaction.lower_bound
            original_upper = reaction.upper_bound
            
            # New bounds based on experimental data
            if stdev_factor is not None:
                new_lower_bound = v - (abs(v)*stdev_factor)
                new_upper_bound = v + (abs(v)*stdev_factor)
            elif experimental_bounds is not None:
                new_lower_bound = experimental_bounds[rxn_id][0]
                new_upper_bound = experimental_bounds[rxn_id][1]
            else:
                new_lower_bound = v
                new_upper_bound = v
            
            # New bounds 
            reaction.lower_bound = max(reaction.lower_bound, new_lower_bound)        
            reaction.upper_bound = min(reaction.upper_bound, new_upper_bound)
            
            if verbose:
                if reaction.lower_bound == original_lower:
                    print(f"{rxn_id}: Kept original lower bound {original_lower} (exp: {new_lower_bound})")
                else:
                    print(f"{rxn_id}: Kept experimental lower bound {new_lower_bound} (orig: {original_lower})")
                    
                if reaction.upper_bound == original_upper:
                    print(f"{rxn_id}: Kept original upper bound {original_upper} (exp: {new_upper_bound})\n")
                else:
                    print(f"{rxn_id}: Kept experimental upper bound {new_upper_bound} (orig: {original_upper})\n")
                
                # Sanity check if bounds are still valid
                if reaction.lower_bound > reaction.upper_bound:
                    print(f"Warning: Infeasible bounds for {rxn_id}. "
                        f"Original: ({reaction.lower_bound}, {reaction.upper_bound}), "
                        f"Experimental: ({new_lower_bound}, {new_upper_bound})")
                
        except KeyError:
            print(f"\nError: Reaction ID {rxn_id} from data not found in model.")
            
    return model


def optimize_model(model: cobra.Model,
                   fluxes_to_print: dict = None) -> np.array:
    """
    Optimize the model and return the solution vector.
    
    Args:
        model: Model to optimize
        fluxes_to_print: Dictionary of fluxes to print, optional
    Returns:
        solution_vector: Solution flux vector
    """
    
    solution = model.optimize()

    # Check solution status
    if solution.status == 'optimal':
        print("SUCCESS: The experimental data is CONSISTENT with the model.")
        solution_vector = solution.fluxes
        if fluxes_to_print is not None:
            print(pd.DataFrame(solution_vector[list(fluxes_to_print.keys())], index=fluxes_to_print.keys()))
        return solution_vector
        
    elif solution.status == 'infeasible':
        print("FAILURE: The experimental data is INCONSISTENT with the model.")
        return None
    
    else:
        print(f"Solver returned an unusual status: {solution.status}")
        return None
    
    
def find_infeasible_constraints(model: cobra.Model) -> list:
    """
    Identifies infeasible constraints in a COBRA model.

    Args:
        model: SBML model to check for infeasibility
    Returns:
        infeasible_constraints: List of infeasible constraints
    """
    infeasible_constraints = []
    
    with model as m:
        # Create a new objective to minimize the sum of slacks
        slacks = {}
        all_slacks = []
        for c in m.constraints:
            # Get existing coefficients to preserve them
            existing_coeffs = c.get_linear_coefficients(c.variables)

            # Add a slack variable to each constraint
            s_pos = m.problem.Variable(f"s_pos_{c.name}", lb=0)
            s_neg = m.problem.Variable(f"s_neg_{c.name}", lb=0)
            m.add_cons_vars([s_pos, s_neg])
            
            new_coeffs = existing_coeffs.copy()
            new_coeffs[s_pos] = -1.0
            new_coeffs[s_neg] = 1.0
            c.set_linear_coefficients(new_coeffs)

            slacks[c] = (s_pos, s_neg)
            all_slacks.extend([s_pos, s_neg])

        # Set the objective to minimize the sum of slack variables
        m.objective = m.problem.Objective(
            sum(all_slacks),
            direction='min'
        )
        
        # Optimize the model to find the minimum sum of slacks
        solution = m.optimize()
        
        if solution.status == 'optimal' and solution.objective_value > 1e-6:
            for c, (s_pos, s_neg) in slacks.items():
                if s_pos.primal > 1e-6 or s_neg.primal > 1e-6:
                    infeasible_constraints.append(c)
        
    return infeasible_constraints


def identify_reactions_causing_imbalance(model: cobra.Model, 
                                         experimental_flux_dict: dict, 
                                         infeasible_metabolites: list,
                                         verbose: bool = True) -> dict:
    """
    For each infeasible metabolite, identify which constrained reactions involve it.
    This is much faster and more direct.
    
    Args:
        model: Model to identify reactions causing imbalance
        experimental_flux_dict: Dictionary of experimental fluxes
        infeasible_metabolites: List of infeasible metabolites
    Returns:
        problematic_reactions: Dictionary of problematic reactions
    """
    problematic_reactions = {}
    
    for constraint in infeasible_metabolites:
        try:
            met_id = constraint.name
            met = model.metabolites.get_by_id(met_id)
        except KeyError:
            print(f"Metabolite {met_id} not found in model")
            continue
            
        # Find all reactions involving this metabolite 
        # and have experimental measurements
        constrained_rxns = []
        rxns_not_in_exp = [] #wip
        for rxn in met.reactions:
            if rxn.id in experimental_flux_dict:
                coeff = rxn.get_coefficient(met_id)
                flux = experimental_flux_dict[rxn.id]
                net_contribution = coeff * flux
                constrained_rxns.append({
                    'reaction': rxn.id,
                    'coefficient': coeff,
                    'exp_flux': flux,
                    'net_effect': net_contribution,
                    'formula': rxn.reaction
                })
            else:
                coeff = rxn.get_coefficient(met_id)
                rxns_not_in_exp.append({
                        'reaction': rxn.id,
                        'coefficient': coeff,
                        'formula': rxn.reaction
                    })
                
                
        
        if constrained_rxns:
            problematic_reactions[met_id] = constrained_rxns

            if verbose:
                print(f"\n{'='*80}")
                print(f"Metabolite: {met_id}")
                print(f"Constrained reactions involving this metabolite:")
                for info in constrained_rxns:
                    print(f"  {info['reaction']}: coeff={info['coefficient']:+.2f}, "
                        f"flux={info['exp_flux']:.4f}, net={info['net_effect']:+.4f}")
                    print(f"    {info['formula']}")
        if rxns_not_in_exp:
            if verbose:
                print(f"\nReactions not in experimental data with this metabolite:")
                for info in rxns_not_in_exp:
                    print(f"  {info['reaction']}: coeff={info['coefficient']:+.2f}")
                    print(f"    {info['formula']}")
    
    return problematic_reactions
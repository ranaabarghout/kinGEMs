"""
Parameter tuning module for kinGEMs.

This module provides simulated annealing functionality to tune kcat parameters
and optimize the model's performance.
"""
import copy  # noqa: F401
import math  # noqa: F401
import os
import random  # noqa: F401
import warnings

from Bio.Data.IUPACData import protein_letters
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from cobra import Metabolite, Reaction
import matplotlib.pyplot as plt
import pandas as pd

from ..config import ensure_dir_exists
from ..dataset import annotate_model_with_kcat_and_gpr
from ..plots import plot_annealing_progress
from .optimize import run_optimization_with_dataframe

warnings.filterwarnings('ignore')
import logging

logging.getLogger('distributed').setLevel(logging.ERROR)
try:
    import gurobipy
    gurobipy.setParam('OutputFlag', 0)
except ImportError:
    pass


# Metabolite ID candidates for the growth-associated maintenance (GAM) ATP
# hydrolysis block across common namespaces:
#   BiGG (atp_c ...), ModelSEED (cpd00002_c0 ...), yeast/rhto-GEM (s_0434 ...)
ATP_MET_IDS = ['atp_c', 'ATP_c', 'cpd00002_c0', 's_0434']
MAINTENANCE_MET_IDS = {
    'h2o': ['h2o_c', 'H2O_c', 'cpd00001_c0', 's_0803'],
    'adp': ['adp_c', 'ADP_c', 'cpd00008_c0', 's_0394'],
    'pi':  ['pi_c', 'Pi_c', 'cpd00009_c0', 's_1322'],
    'h':   ['h_c', 'H_c', 'cpd00067_c0', 's_0794'],
}


def find_gam_components(model, biomass_reaction, gam_reaction_id=None, verbose=False):
    """
    Locate the reaction that carries the growth-associated maintenance (GAM) ATP
    cost, plus the ATP metabolite and the associated maintenance metabolites
    (H2O, ADP, Pi, H+).

    Some GEMs (e.g. yeast-GEM / rhto-GEM) split growth into a 'growth' reaction
    (the objective, e.g. r_2111) that contains no ATP, and a separate 'biomass
    pseudoreaction' (e.g. r_4041) that actually carries the GAM ATP hydrolysis.
    This helper finds the reaction that truly contains the ATP maintenance block
    so GAM scaling targets the right one.

    Parameters
    ----------
    model : cobra.Model
    biomass_reaction : str
        The objective/growth reaction id (checked first if no explicit GAM id).
    gam_reaction_id : str, optional
        Explicit id of the reaction carrying GAM (e.g. 'r_4041'). If provided it
        takes precedence.
    verbose : bool, optional

    Returns
    -------
    (gam_rxn, atp_met, maintenance_mets, original_gam)
        gam_rxn : cobra.Reaction or None
        atp_met : cobra.Metabolite or None
        maintenance_mets : dict {type: cobra.Metabolite}
        original_gam : float or None  (absolute ATP coefficient)
    """
    def _atp_in(rxn):
        for met in rxn.metabolites:
            if met.id in ATP_MET_IDS:
                return met
        return None

    candidate_rxn_ids = []
    if gam_reaction_id:
        candidate_rxn_ids.append(gam_reaction_id)
    if biomass_reaction:
        candidate_rxn_ids.append(biomass_reaction)

    gam_rxn = None
    atp_met = None
    for rid in candidate_rxn_ids:
        try:
            rxn = model.reactions.get_by_id(rid)
        except KeyError:
            continue
        met = _atp_in(rxn)
        if met is not None:
            gam_rxn, atp_met = rxn, met
            break

    # Fallback: scan for a biomass-like reaction containing the ATP hydrolysis block
    if gam_rxn is None:
        for rxn in model.reactions:
            met = _atp_in(rxn)
            if met is None:
                continue
            met_ids = {m.id for m in rxn.metabolites}
            has_adp = any(i in met_ids for i in MAINTENANCE_MET_IDS['adp'])
            has_pi = any(i in met_ids for i in MAINTENANCE_MET_IDS['pi'])
            looks_biomass = ('biomass' in rxn.id.lower()) or ('biomass' in (rxn.name or '').lower())
            if has_adp and has_pi and looks_biomass:
                gam_rxn, atp_met = rxn, met
                break

    if gam_rxn is None or atp_met is None:
        if verbose:
            print("  [GAM] Could not locate a reaction containing the ATP maintenance block")
        return None, None, {}, None

    original_gam = abs(gam_rxn.metabolites[atp_met])

    maintenance_mets = {}
    for met_type, ids in MAINTENANCE_MET_IDS.items():
        for met in gam_rxn.metabolites:
            if met.id in ids:
                maintenance_mets[met_type] = met
                break

    if verbose:
        print(f"  [GAM] Reaction: {gam_rxn.id} | ATP met: {atp_met.id} | GAM: {original_gam:.4f}")

    return gam_rxn, atp_met, maintenance_mets, original_gam


def apply_gam_scaling(model, target_gam, biomass_reaction, gam_reaction_id=None, verbose=False):
    """
    Scale the GAM ATP hydrolysis block of `model` (in place) to `target_gam`
    (mmol ATP/gDW). The whole block (ATP, H2O, ADP, Pi, H+) is scaled together so
    the reaction stays mass/charge balanced.

    Returns
    -------
    (success, original_gam)
        success : bool
        original_gam : float or None
    """
    gam_rxn, atp_met, maintenance_mets, original_gam = find_gam_components(
        model, biomass_reaction, gam_reaction_id, verbose=verbose
    )
    if gam_rxn is None or not original_gam:
        return False, original_gam

    scale = target_gam / original_gam
    current = gam_rxn.metabolites.copy()
    for met in [atp_met] + list(maintenance_mets.values()):
        old_coef = current.get(met)
        if old_coef is None:
            continue
        gam_rxn.add_metabolites({met: old_coef * (scale - 1.0)}, combine=True)

    if verbose:
        print(f"  [GAM] Scaled {gam_rxn.id} ATP from {original_gam:.2f} to {target_gam:.2f}")
    return True, original_gam


# Rekena et al. 2023 (PLoS Comput Biol) / edit_rhtoGEM.m alternative xylose
# assimilation pathway. 
_XYLOSE_ARABINITOL_METABOLITES = [
    {
        "id": "s_D-arabinitol_c",
        "name": "D-arabinitol",
        "compartment": "c",
        "formula": "C5H12O5",
        "charge": 0,
    },
    {
        "id": "s_D-arabinitol_e",
        "name": "D-arabinitol",
        "compartment": "e",
        "formula": "C5H12O5",
        "charge": 0,
    },
    {
        "id": "s_D-ribulose",
        "name": "D-ribulose",
        "compartment": "c",
        "formula": "C5H10O5",
        "charge": 0,
    },
]

_XYLOSE_ARABINITOL_REACTIONS = [
    {
        "id": "t_0883",
        "name": "D-arabinitol 4-dehydrogenase",
        "metabolites": {
            "s_0580": -1,
            "s_1203": -1,
            "s_0794": -1,
            "s_D-arabinitol_c": 1,
            "s_1198": 1,
        },
        "lower_bound": -1000,
        "upper_bound": 1000,
        "gene_reaction_rule": "RHTO_07844",
    },
    {
        "id": "r_4339",
        "name": "D-arabinitol transport",
        "metabolites": {"s_D-arabinitol_c": -1, "s_D-arabinitol_e": 1},
        "lower_bound": -1000,
        "upper_bound": 1000,
        "gene_reaction_rule": "",
    },
    {
        "id": "r_4340",
        "name": "D-arabinitol exchange",
        "metabolites": {"s_D-arabinitol_e": -1},
        "lower_bound": 0,
        "upper_bound": 1000,
        "gene_reaction_rule": "",
    },
    {
        "id": "t_0884",
        "name": "D-arabinitol 2-dehydrogenase/D-ribulose reductase",
        "metabolites": {
            "s_D-ribulose": -1,
            "s_1212": -1,
            "s_0794": -1,
            "s_D-arabinitol_c": 1,
            "s_1207": 1,
        },
        "lower_bound": -1000,
        "upper_bound": 1000,
        "gene_reaction_rule": "RHTO_00373",
    },
    {
        "id": "t_0885",
        "name": "D-ribulokinase",
        "metabolites": {
            "s_0434": -1,
            "s_D-ribulose": -1,
            "s_0394": 1,
            "s_0577": 1,
            "s_0794": 1,
        },
        "lower_bound": 0,
        "upper_bound": 1000,
        "gene_reaction_rule": "RHTO_00950",
    },
]


def _iter_model_edit_metabolites(edits):
    mets = []
    if edits.get("add_xylose_arabinitol_pathway"):
        mets.extend(_XYLOSE_ARABINITOL_METABOLITES)
    mets.extend(edits.get("metabolites") or [])
    return mets


def _iter_model_edit_reactions(edits):
    rxns = []
    if edits.get("add_xylose_arabinitol_pathway"):
        rxns.extend(_XYLOSE_ARABINITOL_REACTIONS)
    rxns.extend(edits.get("reactions") or [])
    return rxns


def apply_model_edits(model, edits, verbose=False):
    """
    Add metabolites/reactions and knock out reactions specified in a config.

    ``edits`` is the ``model_edits`` object from a pipeline JSON config:

    - ``add_xylose_arabinitol_pathway`` (bool): insert Rekena's D-arabinitol /
      D-ribulose xylose path (``t_0883``, ``t_0884``, ``t_0885``, transport
      ``r_4339``, exchange ``r_4340``).
    - ``metabolites`` / ``reactions``: extra entries with the same schema as
      the preset (``id``, ``name``, ``compartment`` / ``metabolites`` dict,
      optional ``lower_bound``, ``upper_bound``, ``gene_reaction_rule``).
    - ``knock_out_reactions``: reaction ids to block (also blocks
      ``{id}_reverse`` if present after irreversible conversion).

    Existing metabolite/reaction ids are left unchanged (idempotent). Call this
    *before* ``apply_medium_to_model`` so medium bounds on ``r_4340`` apply,
    and re-run ``convert_to_irreversible`` afterwards so new reversible
    reactions are split.

    Returns
    -------
    dict
        ``{"added_metabolites": [...], "added_reactions": [...],
        "skipped": [...], "knocked_out": [...]}``
    """
    if not edits:
        return {
            "added_metabolites": [],
            "added_reactions": [],
            "skipped": [],
            "knocked_out": [],
        }

    added_mets = []
    added_rxns = []
    skipped = []

    new_mets = []
    for spec in _iter_model_edit_metabolites(edits):
        met_id = spec["id"]
        if met_id in model.metabolites:
            skipped.append(met_id)
            continue
        met = Metabolite(
            met_id,
            name=spec.get("name", met_id),
            compartment=spec.get("compartment", "c"),
            formula=spec.get("formula"),
            charge=spec.get("charge", 0),
        )
        new_mets.append(met)
        added_mets.append(met_id)
    if new_mets:
        model.add_metabolites(new_mets)

    new_rxns = []
    for spec in _iter_model_edit_reactions(edits):
        rxn_id = spec["id"]
        if rxn_id in model.reactions:
            skipped.append(rxn_id)
            continue
        rxn = Reaction(rxn_id)
        rxn.name = spec.get("name", rxn_id)
        rxn.lower_bound = float(spec.get("lower_bound", 0))
        rxn.upper_bound = float(spec.get("upper_bound", 1000))
        gpr = spec.get("gene_reaction_rule") or ""
        if gpr:
            rxn.gene_reaction_rule = gpr
        stoich = {}
        for met_id, coef in (spec.get("metabolites") or {}).items():
            if met_id not in model.metabolites:
                raise KeyError(
                    f"model_edits reaction '{rxn_id}' refers to unknown "
                    f"metabolite '{met_id}'"
                )
            stoich[model.metabolites.get_by_id(met_id)] = float(coef)
        rxn.add_metabolites(stoich)
        new_rxns.append(rxn)
        added_rxns.append(rxn_id)
    if new_rxns:
        model.add_reactions(new_rxns)

    knocked_out = []
    for rxn_id in edits.get("knock_out_reactions") or []:
        for kid in (rxn_id, f"{rxn_id}_reverse"):
            if kid not in model.reactions:
                continue
            model.reactions.get_by_id(kid).knock_out()
            knocked_out.append(kid)

    if verbose:
        if added_mets:
            print(f"  [model_edits] added metabolites: {', '.join(added_mets)}")
        if added_rxns:
            print(f"  [model_edits] added reactions: {', '.join(added_rxns)}")
        if knocked_out:
            print(f"  [model_edits] knocked out: {', '.join(knocked_out)}")
        if skipped:
            print(f"  [model_edits] already present: {', '.join(skipped)}")

    return {
        "added_metabolites": added_mets,
        "added_reactions": added_rxns,
        "skipped": skipped,
        "knocked_out": knocked_out,
    }


# Alias map from macromolecule names to SBML metabolite ids used
# by rhto-GEM biomass pseudoreaction.
BIOMASS_PSEUDO_METS = {
    "protein":      ["s_3717"],
    "lipid":        ["s_1096"],
    "carbohydrate": ["s_3718"],
    "rna":          ["s_3719"],
    "dna":          ["s_3720"],
}


def apply_biomass_composition(model, biomass_pseudoreaction, multipliers, verbose=False):
    """
    Rescale pseudo-metabolite stoichiometry of the biomass pseudoreaction in place.

    ``multipliers`` maps a macromolecule name (case-insensitive; see
    ``BIOMASS_PSEUDO_METS``) or a raw metabolite id already present in the
    reaction to a positive float. The current stoichiometric coefficient of
    each targeted pseudo-metabolite is multiplied by that factor via
    ``rxn.add_metabolites({met: new_coef}, combine=False)``.

    Parameters
    ----------
    model : cobra.Model
    biomass_pseudoreaction : str
        Reaction id (e.g. ``r_4041`` in rhto-GEM) that consumes the protein /
        lipid / carbohydrate / RNA / DNA pseudo-metabolites.
    multipliers : dict[str, float]
        Keys are macromolecule names (``"protein"``, ``"lipid"``, ...) or raw
        metabolite ids. Values are positive scaling factors.
    verbose : bool, optional
        If True, print each change.

    Returns
    -------
    dict[str, tuple[float, float]]
        ``{met_id: (old_coefficient, new_coefficient)}`` for every metabolite
        that was rescaled.
    """
    if not multipliers:
        return {}

    if biomass_pseudoreaction not in model.reactions:
        raise KeyError(
            f"Biomass pseudoreaction '{biomass_pseudoreaction}' not found in model"
        )
    rxn = model.reactions.get_by_id(biomass_pseudoreaction)
    rxn_met_ids = {m.id for m in rxn.metabolites}

    changes: dict[str, tuple[float, float]] = {}
    for key, mult in multipliers.items():
        try:
            mult_f = float(mult)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"biomass_composition multiplier for '{key}' is not a number: {mult!r}"
            ) from exc
        if mult_f <= 0:
            raise ValueError(
                f"biomass_composition multiplier for '{key}' must be > 0, got {mult_f}"
            )

        key_lower = key.lower()
        if key_lower in BIOMASS_PSEUDO_METS:
            candidate_ids = BIOMASS_PSEUDO_METS[key_lower]
        else:
            candidate_ids = [key]

        met_id = next((mid for mid in candidate_ids if mid in rxn_met_ids), None)
        if met_id is None:
            raise KeyError(
                f"None of the candidate metabolite ids {candidate_ids} for "
                f"'{key}' are present in reaction {biomass_pseudoreaction}"
            )

        met = model.metabolites.get_by_id(met_id)
        old_coef = rxn.metabolites[met]
        new_coef = old_coef * mult_f
        rxn.add_metabolites({met: new_coef}, combine=False)
        changes[met_id] = (float(old_coef), float(new_coef))
        if verbose:
            print(
                f"  [biomass] {biomass_pseudoreaction}: {key} ({met_id}) "
                f"{old_coef:.6g} -> {new_coef:.6g}  (x{mult_f:.4g})"
            )

    return changes


def simulated_annealing(
    model,
    processed_data,
    biomass_reaction,
    objective_value,
    gene_sequences_dict,
    output_dir=None,
    enzyme_fraction=0.15,
    temperature=1.0,
    cooling_rate=0.98,
    min_temperature=0.01,
    max_iterations=250,
    max_unchanged_iterations=3,
    change_threshold=0.001,
    n_top_enzymes=65,
    verbose=False,
    medium=None,
    medium_upper_bound=False,
    edit_ngam=False,
    ngam_rxn_id='ATPM'
):
    """
    Use simulated annealing to tune kcat values for improved biomass production.

    This function preserves original kcat_mean values for proper neighbor calculation
    and creates a kcat_updated column to track tuned values. The optimization function
    automatically uses kcat_updated when available.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model to optimize
    processed_data : pandas.DataFrame
        DataFrame with enzyme kinetic data
    biomass_reaction : str
        ID of the biomass reaction to optimize
    objective_value : float
        Target biomass value
    gene_sequences_dict : dict
        Dictionary mapping gene IDs to protein sequences
    output_dir : str, optional
        Directory to save results
    enzyme_fraction : float, optional
        Maximum enzyme mass fraction (default: 0.15)
    temperature : float, optional
        Initial temperature for simulated annealing (default: 1.0)
    cooling_rate : float, optional
        Rate at which temperature decreases (default: 0.98)
    min_temperature : float, optional
        Minimum temperature before stopping (default: 0.01)
    max_iterations : int, optional
        Maximum number of iterations (default: 250)
    max_unchanged_iterations : int, optional
        Stop after this many iterations without improvement (default: 3)
    change_threshold : float, optional
        Minimum relative change to count as improvement (default: 0.001)
    n_top_enzymes : int, optional
        Number of top enzymes by mass to tune (default: 65)
    verbose : bool, optional
        Print detailed progress information (default: False)
    medium : dict, optional
        Growth medium composition
    medium_upper_bound : bool or float, optional
        Upper bound for medium exchanges (default: False)

    Returns
    -------
    tuple
        (kcat_dict, top_targets, best_df, iterations, biomasses, df_FBA)
    """

    def acceptance_probability(old_biomass, new_biomass, temperature):
        # For MAXIMIZATION: always accept if new > old, probabilistically accept if new < old
        if new_biomass > old_biomass:
            return 1.0
        return math.exp((new_biomass - old_biomass) / temperature)

    def get_neighbor(kcat_value, original_kcat_value, std):
        # Handle NaN kcat values - skip perturbation
        if pd.isna(kcat_value) or kcat_value <= 0:
            return kcat_value  # Return unchanged for invalid values

        k_val_hr = kcat_value * 3600  # Current kcat in hr^-1
        k_orig_hr = original_kcat_value * 3600  # Original kcat in hr^-1 (for bounds)
        std_hr = std * 3600 if not pd.isna(std) else 0

        # If no standard deviation, use a more aggressive default for exploration
        if std_hr == 0 or pd.isna(std_hr):
            std_hr = k_orig_hr * 0.2  # 20% of original value (more aggressive)

        # Generate perturbations for gradual optimization
        # Use 70% positive bias: 70% chance of increase, 30% chance of decrease
        if random.random() < 0.70:  # 70% chance of positive perturbation
            # Moderate increases for gradual improvement
            perturbation = abs(random.gauss(0, std_hr * 20))  # 3x std for more controlled changes
            new_kcat = k_val_hr + perturbation  # Increase
        else:  # 30% chance of decrease
            perturbation = abs(random.gauss(0, std_hr))  # 1x std
            new_kcat = k_val_hr - perturbation  # Decrease

        # Set bounds for perturbations relative to ORIGINAL kcat
        # Allow up to 10x original for exploration
        ub = min(k_orig_hr * 100.0, 4.6e9)  # Biological maximum

        # Set lower bound to 1% of original (prevent going too low)
        lb = max(k_orig_hr * 0.01, 1e-6)

        # Clamp to bounds
        return max(min(new_kcat, ub), lb)

    def update_kcat(df, reaction_id, gene_id, new_kcat_value):
        updated_df = df.copy()
        cond = (
            (updated_df['Reactions'] == reaction_id) &
            (updated_df['Single_gene'] == gene_id)
        )
        # convert back to per-second
        new_value_s = new_kcat_value / 3600

        # Preserve original kcat_mean and kcat_std - only update kcat_updated column
        # Create kcat_updated column if it doesn't exist
        if 'kcat_updated' not in updated_df.columns:
            updated_df['kcat_updated'] = updated_df['kcat_mean'].copy()

        # Get old value for debug output
        old_value = updated_df.loc[cond, 'kcat_updated'].iloc[0] if cond.sum() > 0 else None

        # Update the kcat_updated column (optimization will prefer this over kcat_mean)
        updated_df.loc[cond, 'kcat_updated'] = new_value_s

        # Debug: verify update happened
        if cond.sum() > 0 and verbose:
            actual_new = updated_df.loc[cond, 'kcat_updated'].iloc[0]
            if verbose:
                print(f"    [UPDATE] {reaction_id}_{gene_id}: {old_value:.6e} → {actual_new:.6e} s⁻¹")

        return updated_df

    def calculate_molecular_weight(seq):
        return ProteinAnalysis(seq).molecular_weight()

    # Precompute MWs
    # mw_dict = {
    #     gene: calculate_molecular_weight(seq)
    #     for gene, seq in gene_sequences_dict.items() if seq
    # }

    def safe_mw(seq: str) -> float:
        # keep only standard amino acids
        cleaned = "".join([aa for aa in seq if aa in protein_letters])
        if not cleaned:
            cleaned = "A"  # fallback to alanine
        try:
            return molecular_weight(cleaned, seq_type="protein")
        except Exception:
            return 1e5  # large default if something still weird

    mw_dict = {
        gene: safe_mw(seq)
        for gene, seq in gene_sequences_dict.items() if seq
    }

    # INITIAL FBA
    biomass, df_FBA, _, _ = run_optimization_with_dataframe(
        model=model,
        processed_df=processed_data,
        objective_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_fraction,
        output_dir=output_dir,
        save_results=False,
        verbose=False,
        medium=medium,
        medium_upper_bound=medium_upper_bound,
        edit_ngam=edit_ngam,
        ngam_rxn_id=ngam_rxn_id
    )

    # Check if initial optimization failed
    if biomass is None or biomass <= 0:
        print(f"⚠️  ERROR: Initial enzyme-constrained optimization failed (biomass={biomass})")
        print("   This suggests the enzyme constraints are too restrictive.")
        print("   Consider increasing enzyme_upper_bound or checking your kinetic data.")
        # Return empty results rather than continuing with invalid state
        return {}, pd.DataFrame(), processed_data, [0], [0.0], pd.DataFrame()

    # Select top enzymes by mass (Jan 26 version)
    enzyme_df = df_FBA[df_FBA['Variable']=='enzyme'].copy()
    enzyme_df['MW'] = enzyme_df['Index'].map(mw_dict).fillna(0)
    enzyme_df['enzyme_mass'] = enzyme_df['Value'] * enzyme_df['MW'] * 1e-3

    top_n = enzyme_df.nlargest(n_top_enzymes, 'enzyme_mass')
    top_targets = (
        top_n[['Index','enzyme_mass']]
        .rename(columns={'Index':'Single_gene'})
        .merge(processed_data, on='Single_gene')
        [['Reactions','Single_gene','enzyme_mass','kcat_mean','kcat_std']]
        .reset_index(drop=True)
    )

    # Check for duplicates BEFORE deduplication
    duplicates = top_targets.duplicated(subset=['Reactions', 'Single_gene'], keep=False)
    if duplicates.any():
        top_targets = top_targets.drop_duplicates(subset=['Reactions', 'Single_gene'], keep='first').reset_index(drop=True)

    if verbose:
        print(f"\n[ENZYME SELECTION] Selected {len(top_targets)} enzymes by mass:")
        print(f"  Top 5 targets:")
        for idx, row in top_targets.head(5).iterrows():
            print(f"    {row['Reactions']:15s} {row['Single_gene']:10s} mass={row['enzyme_mass']:.4f}")

    # print(f"\n[ANNEALING DEBUG] Top 5 target enzymes:")
    # print(top_targets.head()[['Reactions', 'Single_gene', 'enzyme_mass', 'kcat_mean']])
    # print(f"[ANNEALING DEBUG] Total targets: {len(top_targets)}")

    # Verify these reactions/genes exist in processed_data
    for idx, row in top_targets.head(3).iterrows():
        rxn, gene = row['Reactions'], row['Single_gene']
        matches = processed_data[(processed_data['Reactions']==rxn) & (processed_data['Single_gene']==gene)]
        # print(f"[DEBUG] {rxn}_{gene}: found {len(matches)} matches in processed_data, kcat_mean={matches['kcat_mean'].iloc[0] if len(matches)>0 else 'NOT FOUND'}")

    largest_rxn_id  = top_targets['Reactions'].tolist()
    largest_gene_id = top_targets['Single_gene'].tolist()
    # Keep original kcat_mean values for neighbor calculation (never changes)
    original_kcats = top_targets['kcat_mean'].tolist()
    # Track current tuned values (starts same as original, gets updated)
    current_solution = top_targets['kcat_mean'].tolist()
    stds             = top_targets['kcat_std'].fillna(0.1).tolist()

    df_new = processed_data.copy()
    # Initialize kcat_updated column with original kcat_mean values
    if 'kcat_updated' not in df_new.columns:
        df_new['kcat_updated'] = df_new['kcat_mean'].copy()

    current_biomass = biomass
    best_solution   = current_solution[:]
    best_biomass    = current_biomass
    best_df         = df_new.copy()

    iteration = 1
    no_change_counter = 0
    iterations = [0]
    biomasses  = [biomass]

    # ANNEALING
    while (temperature > min_temperature
           and iteration < max_iterations
           and current_biomass < objective_value):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Current biomass = {current_biomass:.6e}")

        # PROPOSE & print old vs new kcats - ONLY PERTURB A SUBSET
        updated_df = df_new.copy()
        actually_changed = 0

        # Perturb a smaller subset of enzymes for gradual optimization
        # Use 10-25% of enzymes per iteration for controlled changes
        n_to_perturb = max(1, int(len(largest_rxn_id) * random.uniform(0.10, 0.25)))
        indices_to_perturb = random.sample(range(len(largest_rxn_id)), n_to_perturb)

        # if iteration <= 3:  # Debug info
        #     print(f"  [DEBUG] Perturbing {n_to_perturb} out of {len(largest_rxn_id)} enzymes")

        for i, (rxn, gene) in enumerate(zip(largest_rxn_id, largest_gene_id)):
            # Always use ORIGINAL kcat_mean for neighbor calculation
            original_k = original_kcats[i]  # in s⁻¹
            current_k = current_solution[i]  # current tuned value

            # Skip if not selected for perturbation this iteration
            if i not in indices_to_perturb:
                new_k_hr = current_k * 3600  # Keep current value
                new_k_s = current_k
                # if iteration <= 3:  # Debug first 3 iterations
                #     print(f"  [DEBUG] {rxn}_{gene}: {original_k:.3e} → {new_k_s:.3e} s⁻¹ (change: +0.0%) [UNCHANGED]")
            # Skip NaN values - don't perturb them
            elif pd.isna(original_k) or original_k <= 0:
                new_k_hr = original_k * 3600 if not pd.isna(original_k) else original_k
                new_k_s = original_k
                # if iteration <= 3:  # Debug first 3 iterations
                #     print(f"  [DEBUG] {rxn}_{gene}: {original_k:.3e} → {new_k_s:.3e} s⁻¹ (change: +nan%) [SKIPPED - NaN/invalid]")
            else:
                # Perturb from CURRENT kcat (incremental optimization)
                # But keep bounds relative to ORIGINAL kcat (valid biological range)
                new_k_hr = get_neighbor(current_k, original_k, stds[i])  # returns hr⁻¹
                new_k_s = new_k_hr / 3600

                # Check if actually different
                if abs(new_k_s - current_k) / max(current_k, 1e-12) > 0.01:  # >1% change
                    actually_changed += 1

                # if iteration <= 3:  # Debug first 3 iterations
                #     print(f"  [DEBUG] {rxn}_{gene}: {current_k:.3e} → {new_k_s:.3e} s⁻¹ (change: {(new_k_s-current_k)/current_k*100:+.1f}%)")

            # update_kcat expects hr⁻¹ and will convert to s⁻¹ internally
            updated_df = update_kcat(updated_df, rxn, gene, new_k_hr)

        if verbose:
            print(f"  Actually changed {actually_changed}/{len(largest_rxn_id)} kcats by >1%")

        # Debug: Verify kcats actually changed in updated_df
        if not verbose and iteration <= 3:
            first_rxn, first_gene = largest_rxn_id[0], largest_gene_id[0]
            old_val = df_new.loc[(df_new['Reactions']==first_rxn) & (df_new['Single_gene']==first_gene), 'kcat_mean'].iloc[0]
            new_val = updated_df.loc[(updated_df['Reactions']==first_rxn) & (updated_df['Single_gene']==first_gene), 'kcat_mean'].iloc[0]
            # print(f"DEBUG Iter {iteration}: First kcat change: {first_rxn}:{first_gene} {old_val:.3e} -> {new_val:.3e} ({((new_val/old_val-1)*100):.1f}%)")
            #     print(f"\n  [DEBUG] First target {first_rxn}_{first_gene}: old={old_val:.6f} s⁻¹, new={new_val:.6f} s⁻¹, changed={old_val != new_val}")

        # EVALUATE with updated kcats
        new_biomass, temp_df_FBA, _, _ = run_optimization_with_dataframe(
            model=model,
            processed_df=updated_df,
            objective_reaction=biomass_reaction,
            enzyme_upper_bound=enzyme_fraction,
            output_dir=None,
            save_results=False,
            verbose=False,
            medium=medium,
            medium_upper_bound=medium_upper_bound,
            edit_ngam=edit_ngam,
            ngam_rxn_id=ngam_rxn_id
        )

        # Handle optimization failures
        if new_biomass is None or new_biomass <= 0:
            if verbose or iteration <= 5:
                print(f"  [DEBUG] Iter {iteration}: Optimization failed (biomass={new_biomass})")
                print(f"    - Perturbed {n_to_perturb} enzymes but constraints may be too tight or kcats too exploratory")
                print("    - This suggests even small increases overwhelm constraints")
            # Skip this iteration - don't accept failed optimizations
            # Keep using the current biomass instead of setting to 0.0
            new_biomass = current_biomass  # No change - keep current solution
            temp_df_FBA = df_FBA.copy()  # Use previous valid results
        else:
            if iteration <= 5:  # Debug successful optimizations too
                print(f"  [DEBUG] Iter {iteration}: Optimization succeeded! biomass={new_biomass:.6e}")
                if new_biomass > current_biomass:
                    print(f"    - IMPROVEMENT: +{((new_biomass/current_biomass-1)*100):.3f}%")

        # Debug: Check if enzyme allocations changed even if biomass didn't
        # (Currently disabled to avoid unused variables)
        # if not verbose and iteration <= 3:
        #     # Compare enzyme allocation for first target
        #     old_enzyme = df_FBA[df_FBA['Index']==largest_gene_id[0]]
        #     new_enzyme = temp_df_FBA[temp_df_FBA['Index']==largest_gene_id[0]]
        #     if len(old_enzyme) > 0 and len(new_enzyme) > 0:
        #         old_alloc = old_enzyme[old_enzyme['Variable']=='enzyme']['Value'].iloc[0] if len(old_enzyme[old_enzyme['Variable']=='enzyme']) > 0 else 0
        #         new_alloc = new_enzyme[new_enzyme['Variable']=='enzyme']['Value'].iloc[0] if len(new_enzyme[new_enzyme['Variable']=='enzyme']) > 0 else 0
        #         print(f"  [DEBUG] Enzyme allocation for {largest_gene_id[0]}: {old_alloc:.6e} → {new_alloc:.6e} mmol/gDW/h")

        if verbose:
            print(f"Proposed biomass = {new_biomass:.6e}")

        # ACCEPT or REJECT
        old_biomass = current_biomass  # Store for change calculation
        prob = acceptance_probability(current_biomass, new_biomass, temperature)
        random_val = random.random()
        accept = prob > random_val

        # Debug output for non-verbose mode
        # if not verbose and iteration <= 3:
            # print(f"\n  [DEBUG Iter {iteration}] current={current_biomass:.6f}, proposed={new_biomass:.6f}, prob={prob:.4f}, random={random_val:.4f}, accept={accept}")

        if accept:
            if verbose:
                print(f"Iteration {iteration}: ACCEPTED (Δ = {new_biomass-current_biomass:.2e})")
            df_FBA = temp_df_FBA
            df_new = updated_df.copy()
            current_biomass = new_biomass
            current_solution = [
                df_new.loc[
                    (df_new['Reactions']==rxn)&(df_new['Single_gene']==gene),
                    'kcat_updated'
                ].iat[0]
                for rxn,gene in zip(largest_rxn_id, largest_gene_id)
            ]
            if new_biomass > best_biomass:
                best_biomass  = new_biomass
                best_solution = current_solution[:]
                best_df       = df_new.copy()
        else:
            if verbose:
                print(f"Iteration {iteration}: REJECTED (Δ = {new_biomass-current_biomass:.2e})")

        # Compute ACTUAL change after acceptance/rejection
        change = abs(current_biomass - old_biomass) / max(old_biomass, 1e-6)

        iterations.append(iteration)
        biomasses.append(current_biomass)  # Record ACCEPTED biomass, not proposed

        # Print progress every iteration (non-verbose mode)
        if not verbose and iteration % 1 == 0:
            print(f"  Iter {iteration}/{max_iterations}: biomass={current_biomass:.6f}, temp={temperature:.4f}", end='\r')

        # STAGNATION check uses actual change
        if change < change_threshold:
            no_change_counter += 1
            if no_change_counter >= max_unchanged_iterations:
                print()  # New line after progress indicator
                if verbose:
                    print(f"No significant change for {max_unchanged_iterations} iterations; stopping early.")
                else:
                    print(f"  Early stop: No change for {no_change_counter} iterations")
                break
        else:
            no_change_counter = 0

        temperature *= cooling_rate
        iteration += 1

    # Clear progress line
    if not verbose:
        print()  # New line after progress indicator

    # FINALIZE: build kcat_dict from best_solution
    kcat_dict = {
        f"{rxn}_{gene}": k
        for (rxn, gene), k in zip(zip(largest_rxn_id, largest_gene_id), best_solution)
    }

    if output_dir:
        save_annealing_results(
            output_dir,
            kcat_dict,
            top_targets,
            best_df,
            iterations,
            biomasses,
            df_FBA
        )

    return kcat_dict, top_targets, best_df, iterations, biomasses, df_FBA


def sweep_maintenance_parameters(
    model,
    processed_data,
    biomass_reaction,
    ngam_rxn_id='ATPM',
    ngam_range=None,
    gam_range=None,
    enzyme_upper_bound=0.15,
    output_dir=None,
    medium=None,
    medium_upper_bound=False,
    biomass_goal=None,
    gam_reaction_id=None,
    verbose=False
):
    """
    Sweep NGAM (non-growth associated maintenance) and GAM (growth-associated maintenance)
    parameters to analyze their impact on biomass production.

    NGAM is typically the lower bound of the ATP maintenance reaction (ATPM).
    GAM is the ATP coefficient in the biomass reaction.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model to analyze
    processed_data : pandas.DataFrame
        DataFrame with enzyme kinetic data
    biomass_reaction : str
        ID of the biomass reaction
    ngam_rxn_id : str, optional
        ID of the NGAM reaction (default: 'ATPM')
    ngam_range : list or np.ndarray, optional
        Range of NGAM values to test (mmol/gDW/h)
        If None, uses [0, 1, 2, 3.15, 5, 7, 10, 15, 20]
    gam_range : list or np.ndarray, optional
        Range of GAM values to test (mmol ATP/gDW)
        If None, keeps GAM constant at original value
    enzyme_upper_bound : float, optional
        Maximum enzyme mass fraction (default: 0.15)
    output_dir : str, optional
        Directory to save results
    medium : dict, optional
        Growth medium composition
    medium_upper_bound : bool or float, optional
        Upper bound for medium exchanges (default: False)
    biomass_goal : float, optional
        Target biomass value. If specified, sweep stops early when this goal is reached
        (default: None, tests all combinations)
    verbose : bool, optional
        Print detailed progress (default: False)

    Returns
    -------
    pandas.DataFrame
        Results with columns: ngam, gam, biomass, status
    """
    import numpy as np
    from copy import deepcopy

    # Default ranges
    if ngam_range is None:
        ngam_range = [0, 1, 2, 3.15, 5, 7, 10, 15, 20]  # 3.15 is typical E. coli value

    # Locate the reaction that actually carries the GAM ATP cost. In yeast-GEM /
    # rhto-GEM this is the biomass pseudoreaction (e.g. r_4041), NOT the growth
    # objective (e.g. r_2111) which contains no ATP.
    gam_rxn, atp_met, maintenance_mets, original_gam = find_gam_components(
        model, biomass_reaction, gam_reaction_id, verbose=verbose
    )
    gam_rxn_id_resolved = gam_rxn.id if gam_rxn is not None else gam_reaction_id
    if original_gam is None and verbose:
        print("  Warning: Could not find ATP maintenance block - GAM adjustment disabled")

    # If no GAM range specified, use original value
    if gam_range is None:
        gam_range = [original_gam] if original_gam is not None else [None]

    # Store original NGAM value
    try:
        ngam_rxn = model.reactions.get_by_id(ngam_rxn_id)
        original_ngam = ngam_rxn.lower_bound
        if verbose:
            print(f"  Original NGAM ({ngam_rxn_id}): {original_ngam:.2f} mmol/gDW/h")
    except KeyError:
        print(f"  Warning: NGAM reaction '{ngam_rxn_id}' not found in model")
        return pd.DataFrame()

    results = []
    total_combinations = len(ngam_range) * len(gam_range)
    current = 0

    print(f"\n  Testing {len(ngam_range)} NGAM values × {len(gam_range)} GAM values = {total_combinations} combinations")

    for ngam_val in ngam_range:
        for gam_val in gam_range:
            current += 1

            # Create a copy of the model
            test_model = deepcopy(model)

            # Set NGAM
            test_ngam_rxn = test_model.reactions.get_by_id(ngam_rxn_id)
            test_ngam_rxn.lower_bound = ngam_val

            # Set GAM if specified (scales the GAM-bearing reaction, e.g. r_4041)
            if gam_val is not None and original_gam:
                apply_gam_scaling(
                    test_model,
                    target_gam=gam_val,
                    biomass_reaction=biomass_reaction,
                    gam_reaction_id=gam_rxn_id_resolved,
                    verbose=False
                )

            # Run optimization
            try:
                biomass, _, _, _ = run_optimization_with_dataframe(
                    model=test_model,
                    processed_df=processed_data,
                    objective_reaction=biomass_reaction,
                    enzyme_upper_bound=enzyme_upper_bound,
                    output_dir=None,
                    save_results=False,
                    verbose=False,
                    medium=medium,
                    medium_upper_bound=medium_upper_bound
                )
                status = 'optimal' if biomass and biomass > 0 else 'infeasible'
                biomass = biomass if biomass else 0.0
            except Exception as e:
                biomass = 0.0
                status = f'error: {str(e)[:50]}'

            results.append({
                'ngam': ngam_val,
                'gam': gam_val if gam_val is not None else original_gam,
                'biomass': biomass,
                'status': status
            })

            if verbose or current % 5 == 0:
                print(f"    [{current}/{total_combinations}] NGAM={ngam_val:.2f}, GAM={gam_val if gam_val else 'orig'}→{biomass:.4f}", end='\r')

        # Continue to next iteration without breaking

    print()  # New line after progress

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results if output directory specified
    if output_dir:
        ensure_dir_exists(output_dir)
        results_path = os.path.join(output_dir, "maintenance_sweep_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"  Saved results to: {results_path}")

        # Create visualization
        try:
            import matplotlib.pyplot as plt
            from kinGEMs.plots import set_plotting_style, FONT_SIZES, DEFAULT_DPI
            set_plotting_style()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot 1: Biomass vs NGAM (for each GAM value)
            for gam_val in results_df['gam'].unique():
                subset = results_df[results_df['gam'] == gam_val]
                axes[0].plot(subset['ngam'], subset['biomass'],
                           marker='o', label=f'GAM={gam_val:.1f}')
            axes[0].set_xlabel('NGAM (mmol/gDW/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
            axes[0].set_ylabel('Biomass (1/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
            axes[0].set_title('Biomass vs NGAM', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
            axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True,
                           fontsize=FONT_SIZES['legend'])
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Heatmap if multiple GAM values tested
            if len(gam_range) > 1:
                pivot_data = results_df.pivot(index='gam', columns='ngam', values='biomass')
                im = axes[1].imshow(pivot_data, aspect='auto', cmap='viridis', origin='lower')
                axes[1].set_xlabel('NGAM (mmol/gDW/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                axes[1].set_ylabel('GAM (mmol ATP/gDW)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                axes[1].set_title('Biomass Heatmap', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
                axes[1].set_xticks(range(len(pivot_data.columns)))
                axes[1].set_xticklabels([f'{x:.1f}' for x in pivot_data.columns],
                                        fontsize=FONT_SIZES['tick_label'])
                axes[1].set_yticks(range(len(pivot_data.index)))
                axes[1].set_yticklabels([f'{y:.1f}' for y in pivot_data.index],
                                        fontsize=FONT_SIZES['tick_label'])
                cbar = plt.colorbar(im, ax=axes[1], label='Biomass (1/h)')
                cbar.set_label('Biomass (1/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
            else:
                axes[1].bar(results_df['ngam'], results_df['biomass'], edgecolor='black', linewidth=2)
                axes[1].set_xlabel('NGAM (mmol/gDW/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                axes[1].set_ylabel('Biomass (1/h)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
                axes[1].set_title('Biomass Distribution', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plot_path = os.path.join(output_dir, "maintenance_sweep_plot.png")
            plt.savefig(plot_path, dpi=DEFAULT_DPI, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot to: {plot_path}")
        except Exception as e:
            print(f"  Warning: Could not create plot: {e}")

    # Print summary
    print("\n  Summary:")
    print(f"    Total combinations tested: {len(results_df)}")
    print(f"    Feasible solutions: {(results_df['biomass'] > 0).sum()}")
    print(f"    Best biomass: {results_df['biomass'].max():.4f} at NGAM={results_df.loc[results_df['biomass'].idxmax(), 'ngam']:.2f}, GAM={results_df.loc[results_df['biomass'].idxmax(), 'gam']:.2f}")

    return results_df


def save_annealing_results(output_dir, kcat_dict, df_enzyme_sorted, df_new, iterations, biomasses, df_FBA, prefix=""):
    """
    Save the results of the simulated annealing process.

    Parameters
    ----------
    output_dir : str
        Directory to save output files
    kcat_dict : dict
        Dictionary of optimized kcat values
    df_enzyme_sorted : pandas.DataFrame
        DataFrame with sorted enzyme data
    df_new : pandas.DataFrame
        DataFrame with updated kcat values
    iterations : list
        List of iteration numbers
    biomasses : list
        List of biomass values at each iteration
    df_FBA : pandas.DataFrame
        DataFrame with FBA results
    prefix : str, optional
        Prefix for output filenames
    """
    # Ensure directory exists
    ensure_dir_exists(output_dir)

    # Save kcat dictionary
    kcat_dict_df = pd.DataFrame(list(kcat_dict.items()), columns=['Key', 'Value'])
    kcat_dict_df.to_csv(os.path.join(output_dir, f"{prefix}kcat_dict.csv"), index=False)

    # Save sorted enzyme data
    df_enzyme_sorted.to_csv(os.path.join(output_dir, f"{prefix}df_enzyme_sorted.csv"), index=False)

    # Save updated data
    df_new.to_csv(os.path.join(output_dir, f"{prefix}df_new.csv"), index=False)

    # Save FBA results
    df_FBA.to_csv(os.path.join(output_dir, f"{prefix}df_FBA.csv"), index=False)

    # Save iterations data
    df_iterations = pd.DataFrame({"Iteration": iterations, "Biomass": biomasses})
    df_iterations.to_csv(os.path.join(output_dir, f"{prefix}iterations.csv"), index=False)

    # Create and save plot
    plot_annealing_progress(iterations, biomasses,
                           output_path=os.path.join(output_dir, f"{prefix}annealing_progress.png"))

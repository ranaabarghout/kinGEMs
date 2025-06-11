"""
Dataset module for kinGEMs.

This module provides functions for loading models, mapping metabolites, retrieving
protein sequences, and preparing model data for kinetic analysis.
"""

import logging
import os
import random  # noqa: F401
import re  # noqa: F401
import time  # noqa: F401
import urllib
import urllib.error
from urllib.parse import quote  # noqa: F401
from urllib.request import urlopen  # noqa: F401

from bioservices import UniProt
import cobra
from cobra.core import Reaction
from cobra.util.solver import set_objective
import numpy as np
import pandas as pd
import pubchempy as pcp  # noqa: F401

from .config import (
    CHEBI_COMPOUNDS,
    CHEBI_INCHI,
    METANETX_COMPOUNDS,
    METANETX_XREF,
    SEED_COMPOUNDS,
    TAXONOMY_IDS,
    BiGG_MAPPING,
    ensure_dir_exists,
)


def load_model(model_path):
    """
    Load a COBRA model from file.
    
    Parameters
    ----------
    model_path : str
        Path to the model file (SBML format)
        
    Returns
    -------
    cobra.Model
        The loaded model
    """
    try:
        model, errors = cobra.io.validate_sbml_model(model_path)
        if errors:
            print(f"Warning: Model has {len(errors)} validation errors")
        return cobra.io.read_sbml_model(model_path)
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")

# def convert_to_irreversible(model):
#     """
#     Convert a reversible COBRA model to irreversible form.
#     Splits reversible and exchange reactions into forward and reverse.
#     """
#     model_irrev = model.copy()
#     reactions_to_add = []

#     # Process non-exchange reactions
#     non_ex = [rxn for rxn in model_irrev.reactions if rxn not in model_irrev.exchanges]
#     for rxn in non_ex:
#         if rxn.reversibility:
#             rev_id = rxn.id + '_reverse'
#             if rev_id not in model_irrev.reactions:
#                 orig_lb, orig_ub = rxn.lower_bound, rxn.upper_bound
#                 # Make original irreversible
#                 rxn.lower_bound = max(0, orig_lb)
#                 rxn.upper_bound = max(0, orig_ub)
#                 # Create reverse reaction
#                 rev = Reaction(rev_id)
#                 rev.lower_bound = 0
#                 rev.upper_bound = max(0, -orig_lb)
#                 rev.add_metabolites({m: -c for m,c in rxn.metabolites.items()})
#                 rev.gene_reaction_rule = rxn.gene_reaction_rule
#                 reactions_to_add.append(rev)

#     # Process exchange reactions
#     for ex in model_irrev.exchanges:
#         rev_ex_id = ex.id + '_reverse'
#         if rev_ex_id not in model_irrev.reactions:
#             orig_lb = ex.lower_bound
#             ex.lower_bound = max(0, orig_lb)
#             rev_ex = Reaction(rev_ex_id)
#             rev_ex.lower_bound = 0
#             rev_ex.upper_bound = max(0, -orig_lb)
#             rev_ex.add_metabolites({m: -c for m,c in ex.metabolites.items()})
#             rev_ex.gene_reaction_rule = ex.gene_reaction_rule
#             reactions_to_add.append(rev_ex)

#     model_irrev.add_reactions(reactions_to_add)
#     return model_irrev

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


def get_substrate_metabolites(reaction):
    """
    Get the substrates (reactants) for a reaction.
    
    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction to analyze
        
    Returns
    -------
    list
        List of substrate metabolite IDs
    """
    substrates = [met.id for met in reaction.reactants]
    return substrates

def map_metabolites(substrate_df, external_db_dir=None, max_retries=3, retry_delay=2):
    """
    Map metabolites to SMILES structures using external databases.
    
    Parameters
    ----------
    substrate_df : pandas.DataFrame
        DataFrame with substrate information
    external_db_dir : str, optional
        Directory containing external database files. If None, uses default.
    max_retries : int, optional
        Maximum number of retries for web service requests. Default is 3.
    retry_delay : int, optional
        Delay between retries in seconds. Default is 2.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added SMILES information
    """
    import logging
    import os
    import re  # noqa: F811
    import time  # noqa: F401, F811
    from urllib.error import HTTPError  # noqa: F401

    import numpy as np
    import pandas as pd
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load database files
    if external_db_dir is None:
        external_db_dir = os.path.dirname(BiGG_MAPPING)
    
    # Load database files
    BiGG_comps = pd.read_csv(BiGG_MAPPING)
    CHEBI_comps = pd.read_csv(CHEBI_COMPOUNDS, sep='\t')  # noqa: F841
    CHEBIInChI_comps = pd.read_csv(CHEBI_INCHI, sep='\t')
    MetaNetX_comps = pd.read_csv(METANETX_COMPOUNDS, sep='\t')
    MetaNetX_refcomps = pd.read_csv(METANETX_XREF, sep='\t')  # noqa: F841
    SEED_comps = pd.read_csv(SEED_COMPOUNDS, sep='\t')
    
    # Initialize new columns
    df = substrate_df.copy()
    df['SMILES'] = np.nan
    df['BiGG Name'] = np.nan
    df['DB Name'] = np.nan

    # Clean substrate names before processing
    df['Cleaned Substrate'] = df['Substrate partner'].apply(clean_metabolite_names)

    # Get unique substrates with their details
    unique_substrates_df = df[['Substrate partner', 'Cleaned Substrate']].drop_duplicates()
    logger.info(f"There are {len(unique_substrates_df)} substrates in the GEM.")

    # Prepare for tracking SMILES and names
    smiles_mapping = {}
    bigg_name_mapping = {}
    db_name_mapping = {}

    # Expand old_bigg_ids to create a list for comparison
    BiGG_comps['old_bigg_ids_list'] = BiGG_comps['old_bigg_ids'].str.split(';')
    BiGG_comps_expl = BiGG_comps.explode('old_bigg_ids_list')
    BiGG_comps_unique = BiGG_comps.reset_index(drop=True)
    BiGG_comps_expl_unique = BiGG_comps_expl.reset_index(drop=True)

    # Iterate through each unique substrate
    for _, row in unique_substrates_df.iterrows():
        substrate = row['Substrate partner']
        cleaned_substrate = row['Cleaned Substrate']
        logger.info('-----------------------------')
        logger.info(f'Mapping substrate: {substrate}')

        # Search in BiGG database
        bigg_hit = BiGG_comps_unique.loc[
            BiGG_comps_unique['bigg_id'].str.contains(fr'\b{re.escape(substrate)}\b', regex=True, case=False) |
            BiGG_comps_unique['universal_bigg_id'].str.contains(fr'\b{re.escape(substrate)}\b', regex=True, case=False) |
            BiGG_comps_expl_unique['old_bigg_ids_list'].str.contains(fr'\b{re.escape(substrate)}\b', regex=True, case=False)
        ].head(1)

        if not bigg_hit.empty:
            name = bigg_hit['name'].values[0]
            bigg_name_mapping[substrate] = name
            logger.info(f"BiGG Name: {name}")

            # Check MetaNetX for SMILES
            if not bigg_hit['MetaNetX'].isna().all():
                metanetx_hit = MetaNetX_comps[MetaNetX_comps['ID'] == bigg_hit['MetaNetX'].values[0]]
                if not metanetx_hit.empty:
                    smiles = metanetx_hit['SMILES'].values[0]
                    if pd.notna(smiles):
                        smiles_mapping[substrate] = smiles
                        db_name_mapping[substrate] = metanetx_hit['name'].values[0]
                        logger.info(f"SMILES found in MetaNetX: {smiles}")
                        continue
            
            # Check SEED database
            if not bigg_hit['SEED'].isna().all():
                seed_hit = SEED_comps[SEED_comps['id'] == bigg_hit['SEED'].values[0]]
                if not seed_hit.empty:
                    smiles = seed_hit['smiles'].values[0] if 'smiles' in seed_hit else np.nan
                    name = seed_hit['name'].values[0]
                    if pd.notna(smiles):
                        smiles_mapping[substrate] = smiles
                        db_name_mapping[substrate] = name
                        logger.info(f"SMILES found in SEED: {smiles}")
                        continue
            
            # Check CHEBI database
            if not bigg_hit['CHEBI'].isna().all():
                try:
                    chebi_hit = CHEBIInChI_comps[CHEBIInChI_comps['CHEBI_ID'].str.contains(
                        fr'\b{re.escape(bigg_hit["CHEBI"].values[0])}\b', regex=True, case=False)]
                except AttributeError:
                    # If string methods fail, convert to string first
                    CHEBIInChI_comps['CHEBI_ID'] = CHEBIInChI_comps['CHEBI_ID'].astype(str)
                    chebi_hit = CHEBIInChI_comps[CHEBIInChI_comps['CHEBI_ID'].str.contains(
                        fr'\b{re.escape(bigg_hit["CHEBI"].values[0])}\b', regex=True, case=False)]
                
                if not chebi_hit.empty:
                    inchi = chebi_hit['InChI'].values[0]
                    inchi_hit = MetaNetX_comps[MetaNetX_comps['InChI'] == inchi]
                    if not inchi_hit.empty:
                        smiles = inchi_hit['SMILES'].values[0]
                        name = inchi_hit['name'].values[0]
                        if pd.notna(smiles):
                            smiles_mapping[substrate] = smiles
                            db_name_mapping[substrate] = name
                            logger.info(f"SMILES found in ChEBI: {smiles}")
                            continue
        
        # If no BiGG hit, check other databases directly
        else:
            logger.info("No BiGG match found, checking other databases...")
            
            # Check MetaNetX directly
            metanetx_hit = MetaNetX_comps[MetaNetX_comps['ID'].str.contains(
                fr'\b{re.escape(substrate)}\b', regex=True, case=False)]
            if not metanetx_hit.empty:
                name = metanetx_hit['name'].values[0]
                smiles = metanetx_hit['SMILES'].values[0] if 'SMILES' in metanetx_hit.columns else np.nan
                if pd.notna(smiles):
                    smiles_mapping[substrate] = smiles
                    db_name_mapping[substrate] = name
                    logger.info(f"MetaNetX match: {name}")
                    continue
            
            # Check SEED
            seed_hit = SEED_comps[SEED_comps['id'].str.contains(
                fr'\b{re.escape(substrate)}\b', regex=True, case=False)]
            if not seed_hit.empty:
                name = seed_hit['name'].values[0]
                smiles = seed_hit['smiles'].values[0] if 'smiles' in seed_hit.columns else np.nan
                if pd.notna(smiles):
                    smiles_mapping[substrate] = smiles
                    db_name_mapping[substrate] = name
                    logger.info(f"SEED match: {name}")
                    continue

    # If SMILES is still missing, try to get from web services
    missing_substrates = [sub for sub in df['Substrate partner'].unique() if sub not in smiles_mapping]
    
    for substrate in missing_substrates:
        # First, check alternative databases directly
        metanetx_hit = MetaNetX_comps[MetaNetX_comps['ID'].str.contains(
            fr'\b{re.escape(substrate)}\b', regex=True, case=False)]
        if not metanetx_hit.empty:
            smiles = metanetx_hit['SMILES'].values[0] if 'SMILES' in metanetx_hit.columns else np.nan
            if pd.notna(smiles):
                smiles_mapping[substrate] = smiles
                db_name_mapping[substrate] = metanetx_hit['name'].values[0]
                logger.info(f"Found SMILES in MetaNetX for {substrate}: {smiles}")
                continue

        seed_hit = SEED_comps[SEED_comps['id'].str.contains(
            fr'\b{re.escape(substrate)}\b', regex=True, case=False)]
        if not seed_hit.empty:
            smiles = seed_hit['smiles'].values[0] if 'smiles' in seed_hit.columns else np.nan
            if pd.notna(smiles):
                smiles_mapping[substrate] = smiles
                db_name_mapping[substrate] = seed_hit['name'].values[0]
                logger.info(f"Found SMILES in SEED for {substrate}: {smiles}")
                continue

        # Prepare for web service search
        cleaned_substrate = df.loc[df['Substrate partner'] == substrate, 'Cleaned Substrate'].iloc[0]
        logger.info('-----------------------------')
        logger.info(f'Mapping substrate: {substrate}')
        
        # Try searches with both original and cleaned names
        names_to_try = [cleaned_substrate, substrate]
        
        for name in names_to_try:
            if pd.notna(name):
                # Try CIR with retries
                smiles = get_SMILES_with_retries(name, service='cactus', max_retries=max_retries, retry_delay=retry_delay)
                if smiles:
                    smiles_mapping[substrate] = smiles
                    logger.info(f"Found SMILES via Cactus for {name}: {smiles}")
                    break
                else:
                    # Try PubChem with retries
                    smiles_list = get_SMILES_with_retries(name, service='pubchem', max_retries=max_retries, retry_delay=retry_delay)
                    if smiles_list and len(smiles_list) > 0:
                        smiles_mapping[substrate] = smiles_list[0]
                        logger.info(f"Found SMILES via PubChem for {name}: {smiles_list[0]}")
                        break

    # Apply mappings back to the dataframe
    for substrate, smiles in smiles_mapping.items():
        df.loc[df['Substrate partner'] == substrate, 'SMILES'] = smiles
    
    for substrate, bigg_name in bigg_name_mapping.items():
        df.loc[df['Substrate partner'] == substrate, 'BiGG Name'] = bigg_name
    
    for substrate, db_name in db_name_mapping.items():
        df.loc[df['Substrate partner'] == substrate, 'DB Name'] = db_name
    
    return df


def get_SMILES_with_retries(name, service='cactus', max_retries=3, retry_delay=2):
    """
    Get SMILES from various services with retry logic.
    
    Parameters
    ----------
    name : str
        Compound name to search
    service : str
        Service to use ('cactus' or 'pubchem')
    max_retries : int
        Maximum number of retries
    retry_delay : int
        Delay between retries in seconds
        
    Returns
    -------
    str or list
        SMILES string or list of SMILES strings
    """
    import logging
    import time  # noqa: F811
    
    logger = logging.getLogger(__name__)
    
    for attempt in range(1, max_retries + 1):
        try:
            if service == 'cactus':
                smiles = get_SMILES_from_cactus(name)
                return smiles
            elif service == 'pubchem':
                smiles_list = get_PubChem_SMILES(name)
                return smiles_list
        except Exception as e:
            logger.warning(f"Error retrieving SMILES for {name} from {service} (Attempt {attempt}/{max_retries}): {str(e)}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to retrieve SMILES for {name} after {max_retries} attempts")
                return None
    
    return None


def get_SMILES_from_cactus(name):
    """
    Get SMILES from Chemical Identifier Resolver (CIR) with improved error handling.
    
    Parameters
    ----------
    name : str
        Compound name to search
        
    Returns
    -------
    str
        SMILES string
    """
    import logging
    import urllib.error
    import urllib.parse
    import urllib.request
    
    logger = logging.getLogger(__name__)
    
    # Properly encode the name for URL
    encoded_name = urllib.parse.quote(name)
    url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_name}/smiles"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            smiles = response.read().decode('utf8')
            return smiles.strip()
    except urllib.error.HTTPError as e:
        logger.warning(f"HTTP error retrieving SMILES for {name}: {e}")
        return None
    except urllib.error.URLError as e:
        logger.warning(f"URL error retrieving SMILES for {name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"General error retrieving SMILES for {name}: {e}")
        return None


def get_PubChem_SMILES(name):
    """
    Get SMILES from PubChem with improved error handling.
    
    Parameters
    ----------
    name : str
        Compound name to search
        
    Returns
    -------
    list
        List of SMILES strings
    """
    import json
    import logging
    import time  # noqa: F811

    import requests
    
    logger = logging.getLogger(__name__)
    
    # First search for the compound to get CIDs
    search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(name)}/cids/JSON"
    smiles_list = []
    
    try:
        search_response = requests.get(search_url, timeout=15)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        if 'IdentifierList' in search_data and 'CID' in search_data['IdentifierList']:
            cids = search_data['IdentifierList']['CID']
            
            # Limit to first 3 results to avoid too many requests
            for cid in cids[:3]:
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
                # Get SMILES for each CID
                property_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                try:
                    prop_response = requests.get(property_url, timeout=15)
                    prop_response.raise_for_status()
                    prop_data = prop_response.json()
                    
                    if 'PropertyTable' in prop_data and 'Properties' in prop_data['PropertyTable']:
                        for compound in prop_data['PropertyTable']['Properties']:
                            if 'CanonicalSMILES' in compound:
                                smiles_list.append(compound['CanonicalSMILES'])
                except Exception as e:
                    logger.warning(f"Error retrieving SMILES properties for CID {cid}: {e}")
                    continue
        
        return smiles_list
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error retrieving PubChem data for {name}: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error for PubChem data for {name}: {e}")
        return []
    except Exception as e:
        logger.warning(f"General error retrieving PubChem data for {name}: {e}")
        return []


def clean_metabolite_names(name):
    """
    Clean metabolite names for better matching in databases.
    
    Parameters
    ----------
    name : str
        Raw metabolite name
        
    Returns
    -------
    str
        Cleaned metabolite name
    """
    import re  # noqa: F811
    
    if pd.isna(name):
        return name
    
    # Convert to string if not already
    name = str(name)
    
    # Remove compartment suffix (e.g., _c, _e, _p)
    name = re.sub(r'_[a-z]$', '', name)
    
    # Remove common prefixes/suffixes
    name = re.sub(r'^(cpd|M_|m_)', '', name)
    
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    
    # Remove concentration indicators like (e) or [e]
    name = re.sub(r'[\(\[].[^\)\]]*[\)\]]', '', name)
    
    # Remove charge indicators
    name = re.sub(r'[+-]\d*', '', name)
    
    # Strip whitespace
    name = name.strip()
    
    return name

def retrieve_sequences(model, organism, output_path=None):
    """
    Retrieve protein sequences for genes in a model using UniProt web service.
    
    Parameters
    ----------
    model : cobra.Model or str
        COBRA model object or path to model file
    organism : str
        Organism name (e.g., 'E coli', 'Yeast')
    output_path : str, optional
        Path to save sequence data
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with gene IDs and sequences
    """
    # Load model if string
    if isinstance(model, str):
        model = load_model(model)
    
    # Get taxonomy ID
    if organism in TAXONOMY_IDS:
        taxon_ID = TAXONOMY_IDS[organism]
    else:
        raise ValueError(f"Unknown organism '{organism}'. Available options: {list(TAXONOMY_IDS.keys())}")
    
    # Helper functions
    def extract_genes(gpr_rules):
        """Extract individual genes from a GPR rule."""
        genes = []
        for rule in gpr_rules.split(' and '):
            sub_genes = rule.split(' or ')
            for gene in sub_genes:
                gene = gene.replace('(', '').replace(')', '')
                genes.append(gene)
        return genes
    
    def get_UniProt_sequence(gene):
        """
        Query UniProt for gene sequence.
        
        Parameters
        ----------
        gene : str
            Gene name to query
        
        Returns
        -------
        str or None
            Protein sequence if found, None otherwise
        """
        if not gene:
            return None
        
        try:
            query = f"gene_exact:({gene}) AND taxonomy_id:({taxon_ID})"
            result = service.search(query, frmt="fasta")
            
            # Validate result
            if not result:
                logging.warning(f"No sequence found for gene {gene}")
                return None
            
            # Extract sequence from FASTA format
            sequence = result.split("\n", 1)[-1]
            sequence = sequence.replace("\n", "").strip()
            
            # Additional validation
            if not sequence:
                logging.warning(f"Empty sequence retrieved for gene {gene}")
                return None
            
            return sequence
        
        except urllib.error.URLError as url_err:
            # Network-related errors
            logging.warning(f"Network error retrieving sequence for {gene}: {url_err}")
            return None
        
        except AttributeError as attr_err:
            # Potential issues with service or method
            logging.error(f"Attribute error for gene {gene}: {attr_err}", exc_info=True)
            return None
        
        except Exception as e:
            # Catch any other unexpected errors
            logging.error(f"Unexpected error retrieving sequence for {gene}: {e}", exc_info=True)
            return None
    
    # Create dataframe with all genes
    gene_data = []
    for gene in model.genes:
        gene_data.append({
            'Single_gene': gene.id,
            'Name': gene.name
        })
    
    df_genes = pd.DataFrame(gene_data)
    
    # Initialize UniProt service
    service = UniProt()
    
    # Query sequences for each gene
    df_genes['Sequence'] = df_genes['Single_gene'].apply(get_UniProt_sequence)
    
    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        df_genes.to_csv(output_path, index=False)
    
    return df_genes

def prepare_model_data(model_path, substrates_output=None, sequences_output=None, 
                      organism='E coli', metadata_dir=None):
    """
    Prepare model data for kinetic analysis.
    
    This function performs several preprocessing steps:
    1. Load the model
    2. Convert to irreversible reactions
    3. Extract substrate information
    4. Map metabolites to SMILES
    5. Retrieve protein sequences
    
    Parameters
    ----------
    model_path : str
        Path to the model file
    substrates_output : str, optional
        Path to save substrate data
    sequences_output : str, optional
        Path to save sequence data
    organism : str, optional
        Organism name for sequence retrieval
    metadata_dir : str, optional
        Directory containing metadata files
        
    Returns
    -------
    tuple
        (irreversible_model, substrate_df, sequences_df)
    """
    # Load model
    model = load_model(model_path)
    print(f"Loaded model with {len(model.reactions)} reactions and {len(model.metabolites)} metabolites")
    
    # Convert to irreversible
    irrev_model = model #convert_to_irreversible(model)
    # print(f"Converted to irreversible model with {len(irrev_model.reactions)} reactions")
    
    # Extract substrates
    rxn_data = []
    for reaction in irrev_model.reactions:
        substrates = get_substrate_metabolites(reaction)
        for substrate in substrates:
            rxn_data.append({
                "Reaction": reaction.id,
                "Substrate partner": substrate
            })
    
    substrate_df = pd.DataFrame(rxn_data)
    print(f"Extracted {len(substrate_df)} substrate-reaction pairs")
    
    # Map metabolites to SMILES
    substrate_df_with_smiles = map_metabolites(substrate_df, metadata_dir)
    print(f"Mapped metabolites to SMILES ({substrate_df_with_smiles['SMILES'].notna().sum()} found)")
    
    # Retrieve protein sequences
    sequences_df = retrieve_sequences(irrev_model, organism)
    print(f"Retrieved {sequences_df['Sequence'].notna().sum()} protein sequences")
    
    # Save outputs if paths provided
    if substrates_output:
        ensure_dir_exists(os.path.dirname(substrates_output))
        substrate_df_with_smiles.to_csv(substrates_output, index=False)
    
    if sequences_output:
        ensure_dir_exists(os.path.dirname(sequences_output))
        sequences_df.to_csv(sequences_output, index=False)
    
    return irrev_model, substrate_df_with_smiles, sequences_df

def merge_substrate_sequences(substrate_df, sequences_df, model, output_path=None):
    """
    Merge substrate and sequence data for kinetic analysis.
    
    Parameters
    ----------
    substrate_df : pandas.DataFrame
        DataFrame with substrate information and SMILES
    sequences_df : pandas.DataFrame
        DataFrame with gene sequences
    model : cobra.Model
        The model containing reactions and their GPR rules
    output_path : str, optional
        Path to save the merged data
        
    Returns
    -------
    pandas.DataFrame
        Merged dataframe with substrates and sequences
    """
    # Extract gene information from model reactions
    reaction_genes = []
    
    # Helper function to extract genes from GPR
    def extract_genes(gpr_rule):
        genes = []
        if not gpr_rule or pd.isna(gpr_rule):
            return genes
            
        for rule in gpr_rule.split(' and '):
            sub_genes = rule.split(' or ')
            for gene in sub_genes:
                gene = gene.replace('(', '').replace(')', '')
                genes.append(gene)
        return genes
    
    def get_gpr_rule(reaction_id, model):
        """
        Extract the Gene-Protein-Reaction (GPR) rule for a given reaction.
        
        Parameters
        ----------
        reaction_id : str
            The ID of the reaction
        model : cobra.Model
            The model containing the reaction
            
        Returns
        -------
        str or None
            The GPR rule as a string, or None if the reaction doesn't exist
            or doesn't have a rule
        """
        try:
            # Get the reaction object from the model
            reaction = model.reactions.get_by_id(reaction_id)
            
            # Return the gene_reaction_rule attribute
            return reaction.gene_reaction_rule
        except KeyError:
            # If reaction doesn't exist in the model
            print(f"Warning: Reaction {reaction_id} not found in the model")
            return None
        except AttributeError:
            # If the reaction doesn't have a gene_reaction_rule attribute
            print(f"Warning: No GPR rule found for reaction {reaction_id}")
            return None

    # Create extended dataframe with gene-reaction mappings
    for reaction_id in substrate_df['Reaction'].unique():
        gpr_rule = get_gpr_rule(reaction_id, model)
        genes = extract_genes(gpr_rule)
        for gene in genes:
            reaction_genes.append({
                'Reaction': reaction_id,
                'Single_gene': gene,
                'GPR_rules': gpr_rule
            })
    
    df_reaction_genes = pd.DataFrame(reaction_genes)
    
    # Merge substrate data with gene data
    merged_df = substrate_df.merge(df_reaction_genes, on='Reaction', how='left')
    
    # Merge with sequences
    merged_df = merged_df.merge(sequences_df[['Single_gene', 'Sequence']], 
                               on='Single_gene', how='left')
    
    # Rename columns for consistency with original code
    merged_df = merged_df.rename(columns={
        'Reaction': 'Reactions', 
        'Sequence': 'SEQ',
        'Substrate partner': 'Reaction_partners',
        'SMILES': 'CMPD_SMILES'
    })

    merged_df['kcat'] = np.random.random(len(merged_df))
    
    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        merged_df.to_csv(output_path, index=False)
    
    return merged_df

def process_kcat_predictions(merged_df, predictions_csv_path, output_path=None):
    """
    Process k-fold predictions for kcat values and merge with substrate-sequence data.
    
    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame with substrates and sequences from merge_substrate_sequences
    predictions_csv_path : str
        Path to the CSV file containing kcat predictions for all folds in separate columns
        (pred_value_0, pred_value_1, pred_value_2, pred_value_3, pred_value_4)
    output_path : str, optional
        Path to save the processed data
        
    Returns
    -------
    pandas.DataFrame
        Processed dataframe with merged predictions and statistics
    """
    import os

    import numpy as np  # noqa: F401
    import pandas as pd
    
    # Validate inputs
    if not os.path.exists(predictions_csv_path):
        raise FileNotFoundError(f"Predictions CSV file not found: {predictions_csv_path}")
    
    # Make sure required columns exist in merged_df
    required_cols = ['CMPD_SMILES', 'SEQ']
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in merged_df: {missing_cols}")
    
    # Create a deep copy to avoid modifying the original dataframe
    result_df = merged_df.copy()
    
    # Load the predictions CSV
    predictions_df = pd.read_csv(predictions_csv_path)
    
    # Ensure the predictions CSV has the required columns
    required_pred_cols = ['CMPD_SMILES', 'SEQ'] + [f'pred_value_{i}' for i in range(5)]
    missing_pred_cols = [col for col in required_pred_cols if col not in predictions_df.columns]
    if missing_pred_cols:
        raise ValueError(f"Missing required columns in predictions CSV: {missing_pred_cols}")
    
    # Merge with the predictions
    result_df = pd.merge(result_df, predictions_df, on=['CMPD_SMILES', 'SEQ'], how='left')
    
    # Calculate statistics across the folds
    kcat_cols = [f'pred_value_{i}' for i in range(5)]
    
    # Calculate mean kcat (average of the 5 folds)
    result_df['kcat_mean'] = result_df[kcat_cols].mean(axis=1)
    
    # Calculate standard deviation of kcat values
    result_df['kcat_std'] = result_df[kcat_cols].std(axis=1)
    
    # Calculate coefficient of variation (relative standard deviation)
    result_df['kcat_cv'] = result_df['kcat_std'] / result_df['kcat_mean'] * 100
    
    # Calculate min and max kcat values
    result_df['kcat_min'] = result_df[kcat_cols].min(axis=1)
    result_df['kcat_max'] = result_df[kcat_cols].max(axis=1)

    result_df = result_df.drop('kcat_x', axis=1)
    
    # Save if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
    
    return result_df

def process_merged_data_with_folds(merged_df, fold_csv_paths, output_path=None):
    """
    Directly process a merged dataframe with fold predictions.
    
    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame already containing substrate and sequence information 
        (output from merge_substrate_sequences function)
    fold_csv_paths : list
        List of paths to the 5 CSV files containing kcat predictions for each fold
    output_path : str, optional
        Path to save the final processed data
        
    Returns
    -------
    pandas.DataFrame
        Processed dataframe with merged predictions and statistics
    """
    # Process the kcat predictions
    result_df = process_kcat_predictions(merged_df, fold_csv_paths, output_path)
    
    return result_df

def assign_kcats_to_model(model, df_new):
    """
    Assign kcat values from df_new to the corresponding reactions in the model.
    Each reaction will receive a `.kcats` attribute that maps genes to kcat values.

    Parameters
    ----------
    model : cobra.Model
        The COBRA model to annotate
    df_new : pandas.DataFrame
        DataFrame containing columns: ['Reactions', 'Single_gene', 'kcat_mean']

    Returns
    -------
    cobra.Model
        The same model object with .kcats added to reactions
    """
    from collections import defaultdict

    kcats_by_reaction = defaultdict(dict)

    for _, row in df_new.iterrows():
        rxn_id = row['Reactions']
        gene_id = row['Single_gene']
        kcat = row['kcat_mean']
        if pd.notna(kcat):
            kcats_by_reaction[rxn_id][gene_id] = kcat

    for rxn_id, gene_kcats in kcats_by_reaction.items():
        try:
            reaction = model.reactions.get_by_id(rxn_id)
            reaction.kcats = gene_kcats  # Attach as a custom attribute
        except KeyError:
            print(f"Warning: Reaction {rxn_id} not found in model.")

    return model

def format_kcats_like_gpr(reaction):
    """
    Return a GPR-style string representation of the kcats assigned to a reaction.

    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction object with a `.kcats` dictionary attribute

    Returns
    -------
    str
        A string that mirrors the GPR rule but shows kcat values instead of gene names
    """
    if not hasattr(reaction, 'kcats') or not reaction.kcats:
        return "No kcats assigned"

    gpr = reaction.gene_reaction_rule
    if not gpr:
        return "No GPR rule"

    formatted = gpr

    for gene, kcat in reaction.kcats.items():
        formatted = formatted.replace(gene, f"{kcat:.2e}")

    return formatted

def annotate_model_with_kcat_and_gpr(model: cobra.Model,
                                     df: pd.DataFrame,
                                     reaction_col: str = "Reactions",
                                     gene_col: str     = "Single_gene",
                                     gpr_col: str      = "GPR_rules",
                                     kcat_col: str     = "kcat_mean") -> cobra.Model:
    """
    For each reaction in `model` that appears in df[reaction_col]:
      • Collect all df[kcat_col] values for each gene (df[gene_col]), convert to 1/hr.
      • reaction.annotation['kcat'] = flat list of all those converted kcats.
      • reaction.annotation['gpr_replaced'] = original GPR (df[gpr_col]) 
        with each gene name replaced by its kcat (or "(k1 or k2…)" if multiple).
    
    Parameters
    ----------
    model : cobra.Model
        Your COBRApy model.
    df : pandas.DataFrame
        Must contain columns:
          - reaction IDs in `reaction_col`
          - gene IDs in `gene_col`
          - numeric kcat values (1/s) in `kcat_col`
          - logical GPR strings in `gpr_col`
    reaction_col : str
        Name of the DataFrame column with reaction IDs.
    gene_col : str
        Name of the column with gene IDs.
    gpr_col : str
        Name of the column with the original GPR expression.
    kcat_col : str
        Name of the column with kcat values in 1/s.
    
    Returns
    -------
    cobra.Model
        The same model, with each annotated reaction carrying:
          - annotation['kcat']
          - annotation['gpr_replaced']
    """
    # group your data by reaction
    for rxn_id, subdf in df.groupby(reaction_col, dropna=False):
        if rxn_id not in model.reactions:
            continue
        rxn = model.reactions.get_by_id(rxn_id)

        # ---- tag raw GPR structure
        rule = (rxn.gene_reaction_rule or "").lower()
        if "and" in rule or "or" in rule:
            rxn.annotation["gpr"] = "AND/OR"
        else:
            rxn.annotation["gpr"] = "1"

        # build gene → [kcat1, kcat2, …] (in 1/hr)
        gene2kcats = {}
        for _, row in subdf.iterrows():
            gene = row[gene_col]
            raw_k = row[kcat_col]
            if pd.isna(raw_k):
                continue
            k_hr = float(raw_k) * 3600
            gene2kcats.setdefault(gene, []).append(k_hr)

        # flatten all kcats for reaction-level annotation
        all_kcats = [k for ks in gene2kcats.values() for k in ks]
        if all_kcats:
            rxn.annotation["kcat"] = all_kcats

        # pick one representative GPR string (assumes identical in each sub-row)
        gpr = subdf[gpr_col].dropna().unique()
        if len(gpr) == 0:
            continue
        gpr = gpr[0]

        # build a regex that matches any of your gene IDs as whole words
        pattern = r'\b(' + '|'.join(map(re.escape, gene2kcats.keys())) + r')\b'

        def _replace_gene(match):
            gene = match.group(1)
            klist = gene2kcats.get(gene, [])
            if not klist:
                return gene
            if len(klist) == 1:
                return f"{klist[0]:.6g}"
            # multiple measurements → parenthesize them
            inner = " or ".join(f"{k:.6g}" for k in klist)
            return f"({inner})"

        # do the substitution
        gpr_replaced = re.sub(pattern, _replace_gene, gpr)
        rxn.annotation["gpr_replaced"] = gpr_replaced

    return model
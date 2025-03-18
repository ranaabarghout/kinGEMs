"""
Dataset module for kinGEMs.

This module provides functions for loading models, mapping metabolites, retrieving
protein sequences, and preparing model data for kinetic analysis.
"""

import logging
import os
import random
import re
import time
import urllib
import urllib.error
from urllib.parse import quote
from urllib.request import urlopen

from bioservices import UniProt
import cobra
from cobra.core import Reaction
from cobra.util.solver import set_objective
import numpy as np
import pandas as pd
import pubchempy as pcp

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

def convert_to_irreversible(model):
    """
    Convert all reversible reactions to irreversible reactions.
    
    Non-exchange reversible reactions are split into forward and reverse reactions.
    Exchange reactions are made reversible by creating reverse reactions.
    
    Parameters
    ----------
    model : cobra.Model
        The model to convert
        
    Returns
    -------
    cobra.Model
        The converted model with irreversible reactions
    """
    # Create a copy of the model to avoid modifying the original
    model_irrev = model.copy()
    
    # List to hold reactions to add
    reactions_to_add = []
    coefficients = {}

    # Convert only non-exchange reversible reactions to irreversible
    non_exchange_reactions = [rxn for rxn in model_irrev.reactions if rxn not in model_irrev.exchanges]
    exchange_reactions = [rxn for rxn in model_irrev.exchanges]
    print('Number of reactions that are non-exchange: ', len(non_exchange_reactions))
    print('Number of reactions that are exchange: ', len(exchange_reactions))

    # Process non-exchange reactions
    for reaction in non_exchange_reactions:
        if reaction.reversibility:
            reverse_reaction_id = reaction.id + "_reverse"
            
            # Check if the reverse reaction already exists in the model
            if reverse_reaction_id not in [rxn.id for rxn in model_irrev.reactions]:
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
    
    # Ensure all exchange reactions have reverse reactions
    for exchange_reaction in model_irrev.exchanges:
        reverse_exchange_id = exchange_reaction.id + "_reverse"
        
        # Check if the reverse reaction already exists in the model
        if reverse_exchange_id not in [rxn.id for rxn in model_irrev.reactions]:
            # Create a reverse reaction for exchange reactions 
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
    model_irrev.add_reactions(reactions_to_add)

    # Set new objective with the added reverse reactions
    set_objective(model_irrev, coefficients, additive=True)

    return model_irrev

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

def map_metabolites(substrate_df, external_db_dir=None):
    """
    Map metabolites to SMILES structures using external databases.
    
    Parameters
    ----------
    substrate_df : pandas.DataFrame
        DataFrame with substrate information
    external_db_dir : str, optional
        Directory containing external database files. If None, uses default.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added SMILES information
    """
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
    print(f"There are {len(unique_substrates_df)} substrates in the GEM.")

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
        print('-----------------------------')
        print('Mapping substrate:', substrate)

        # Search in BiGG database
        bigg_hit = BiGG_comps_unique.loc[
            BiGG_comps_unique['bigg_id'].str.contains(fr'\b{re.escape(substrate)}\b', regex=True, case=False) |
            BiGG_comps_unique['universal_bigg_id'].str.contains(fr'\b{re.escape(substrate)}\b', regex=True, case=False) |
            BiGG_comps_expl_unique['old_bigg_ids_list'].str.contains(fr'\b{re.escape(substrate)}\b', regex=True, case=False)
        ].head(1)

        if not bigg_hit.empty:
            name = bigg_hit['name'].values[0]
            bigg_name_mapping[substrate] = name
            print("BiGG Name:", name)

            # Check MetaNetX for SMILES
            if not bigg_hit['MetaNetX'].isna().all():
                metanetx_hit = MetaNetX_comps[MetaNetX_comps['ID'] == bigg_hit['MetaNetX'].values[0]]
                if not metanetx_hit.empty:
                    smiles = metanetx_hit['SMILES'].values[0]
                    if pd.notna(smiles):
                        smiles_mapping[substrate] = smiles
                        db_name_mapping[substrate] = metanetx_hit['name'].values[0]
                        print("SMILES found in MetaNetX:", smiles)
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
                        print("SMILES found in SEED:", smiles)
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
                            print("SMILES found in ChEBI:", smiles)
                            continue
        
        # If no BiGG hit, check other databases directly
        else:
            print("No BiGG match found, checking other databases...")
            
            # Check MetaNetX directly
            metanetx_hit = MetaNetX_comps[MetaNetX_comps['ID'].str.contains(
                fr'\b{re.escape(substrate)}\b', regex=True, case=False)]
            if not metanetx_hit.empty:
                name = metanetx_hit['name'].values[0]
                smiles = metanetx_hit['SMILES'].values[0] if 'SMILES' in metanetx_hit.columns else np.nan
                if pd.notna(smiles):
                    smiles_mapping[substrate] = smiles
                    db_name_mapping[substrate] = name
                    print("MetaNetX match:", name)
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
                    print("SEED match:", name)
                    continue

    # If SMILES is still missing, try to get from web services
    missing_substrates = df[df['SMILES'].isna()]['Substrate partner'].unique()
    
    for substrate in missing_substrates:
        # First, check alternative databases directly
        metanetx_hit = MetaNetX_comps[MetaNetX_comps['ID'].str.contains(
            fr'\b{re.escape(substrate)}\b', regex=True, case=False)]
        if not metanetx_hit.empty:
            smiles = metanetx_hit['SMILES'].values[0] if 'SMILES' in metanetx_hit.columns else np.nan
            if pd.notna(smiles):
                smiles_mapping[substrate] = smiles
                db_name_mapping[substrate] = metanetx_hit['name'].values[0]
                print(f"Found SMILES in MetaNetX for {substrate}: {smiles}")
                continue

        seed_hit = SEED_comps[SEED_comps['id'].str.contains(
            fr'\b{re.escape(substrate)}\b', regex=True, case=False)]
        if not seed_hit.empty:
            smiles = seed_hit['smiles'].values[0] if 'smiles' in seed_hit.columns else np.nan
            if pd.notna(smiles):
                smiles_mapping[substrate] = smiles
                db_name_mapping[substrate] = seed_hit['name'].values[0]
                print(f"Found SMILES in SEED for {substrate}: {smiles}")
                continue

        # Prepare for web service search
        cleaned_substrate = df.loc[df['Substrate partner'] == substrate, 'Cleaned Substrate'].iloc[0]
        print('-----------------------------')
        print('Mapping substrate:', substrate)
        
        # Try searches with both original and cleaned names
        names_to_try = [cleaned_substrate, substrate]
        
        for name in names_to_try:
            if pd.notna(name):
                # Try CIR
                smiles = get_SMILES_from_cactus(name)
                if smiles:
                    smiles_mapping[substrate] = smiles
                    print(f"Found SMILES via Cactus for {name}: {smiles}")
                    break
                else:
                    # Try PubChem
                    smiles_list = get_PubChem_SMILES(name)
                    if smiles_list and len(smiles_list) > 0:
                        smiles_mapping[substrate] = smiles_list[0]
                        print(f"Found SMILES via PubChem for {name}: {smiles_list[0]}")
                        break

    # Apply mappings back to the dataframe
    for substrate, smiles in smiles_mapping.items():
        df.loc[df['Substrate partner'] == substrate, 'SMILES'] = smiles
    
    for substrate, bigg_name in bigg_name_mapping.items():
        df.loc[df['Substrate partner'] == substrate, 'BiGG Name'] = bigg_name
    
    for substrate, db_name in db_name_mapping.items():
        df.loc[df['Substrate partner'] == substrate, 'DB Name'] = db_name
    
    return df

def clean_metabolite_names(string):
    """
    Clean metabolite names for improved mapping.
    
    Parameters
    ----------
    string : str
        The metabolite name to clean
        
    Returns
    -------
    str
        The cleaned metabolite name
    """
    if not isinstance(string, str):
        string = str(string)  # Convert to string if not already
    
    # Handle specific cases
    if 'L Major' in string:
        string = string.split(' L major')[0]
        string = re.sub(r'(\d) (\d) (\w)', r'\1,\2-\3', string)
        string = re.sub(r'(\d) (\w)', r'\1-\2', string)
        string = re.sub(r'[CHNPO]\d+', '', string)
    
    elif 'CoA' in string:
        string = string.split('CoA', 1)[0]
        string = re.sub(r'(\d) (\d) (\w)', r'\1,\2-\3', string)
        string = re.sub(r'(\d) (\w)', r'\1-\2', string)
        string = re.sub(r'[CHNPO]\d+', '', string)
    
    elif 'A DASH D DASH' in string:
        string = string.replace('A DASH ', 'alpha-')
        string = string.replace('D DASH ', 'D-')
        string = string.replace('B DASH ', 'beta-')
        string = string.replace(' c', '')
        string = re.sub(r'(\d) (\d) (\w)', r'\1,\2-\3', string)
        string = re.sub(r'(\d) (\w)', r'\1-\2', string)
        string = re.sub(r'[CHNPO]\d+', '', string)
    
    elif 'S  3 Methylbutanoyl  dihydrolipoamide' in string:
        string = string.replace("S  3 Methylbutanoyl  dihydrolipoamide C13H25NO2S2", 
                              "S-(3-methylbutanoyl)-dihydrolipoamide")
    
    else:
        # General cleaning
        string = string.replace("", "")
        string = string.replace(" ", "")
        string = string.replace(";", "")
        string = string.replace("H2O H2O", "H2O")
        string = re.sub(r'(\d) (\d) (\w)', r'\1,\2-\3', string)
        string = re.sub(r'(\d) (\w)', r'\1-\2', string)
        string = re.sub(r'[CHNPO]\d+', '', string)
        string = string.replace("1,2-Diacylglycerol  L major  ", "1,2-diacylglycerol")
        string = string.replace("AMP P", "AMP")
        string = string.replace("coa_c", "Coenzyme A")
        string = string.replace("coa_e", "Coenzyme A")
        string = string.replace("h2o_c", "h2o")
        string = string.replace("h2o_e", "h2o")
        string = string.replace('A DASH ', 'alpha-')
        string = string.replace('D DASH ', 'D-')
        string = string.replace('B DASH ', 'beta-')
        string = string.replace(' c', '')
        string = re.sub(r'\(.*?\)', '', string)
        string = string.replace('C04051', '5-Amino-4-imidazolecarboxyamid')
    
    return string

def get_SMILES_from_cactus(name, max_retries=1):
    """
    Get SMILES from NIH Chemical Identifier Resolver web service.
    
    Parameters
    ----------
    name : str
        Compound name to query
    max_retries : int, optional
        Number of times to retry the request
        
    Returns
    -------
    str or None
        SMILES string if found, None otherwise
    """
    if not name or pd.isna(name):
        return None
    
    for attempt in range(max_retries):
        try: 
            url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(name) + '/smiles' 
            
            # Add random delay between retries
            if attempt > 0:
                time.sleep(random.uniform(1, 3))
            
            # Set a timeout to prevent hanging
            cmpd_smiles = urlopen(url, timeout=10).read().decode('utf8').strip()
            
            # Validate SMILES (basic check)
            if cmpd_smiles and len(cmpd_smiles) > 3:
                return cmpd_smiles 
            
            logging.warning(f"Invalid SMILES for {name}: {cmpd_smiles}")
            return None
        
        except urllib.error.HTTPError as http_err:
            # Log specific HTTP errors
            logging.warning(f"HTTP error retrieving SMILES for {name} (Attempt {attempt+1}/{max_retries}): {http_err}")
            
            # Specific handling for 504 Gateway Timeout
            if http_err.code == 504:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to retrieve SMILES for {name} after {max_retries} attempts")
                continue
            
            return None
        
        except urllib.error.URLError as url_err:
            # Network-related errors
            logging.warning(f"URL error retrieving SMILES for {name} (Attempt {attempt+1}/{max_retries}): {url_err}")
            
            if attempt == max_retries - 1:
                logging.error(f"Failed to retrieve SMILES for {name} after {max_retries} attempts")
            
            continue
        
        except Exception as e: 
            # Catch any other unexpected errors
            logging.error(f"Unexpected error retrieving SMILES for {name}: {e}", exc_info=True)
            return None
    
    return None

def get_PubChem_SMILES(name, max_retries=3):
    """
    Get SMILES from PubChem web service.
    
    Parameters
    ----------
    name : str
        Compound name to query
    max_retries : int, optional
        Number of times to retry the request
        
    Returns
    -------
    list or None
        List of SMILES strings if found, None otherwise
    """
    if not name or pd.isna(name):
        return None
    
    for attempt in range(max_retries):
        try:
            # Add random delay between retries
            if attempt > 0:
                time.sleep(random.uniform(1, 3))
            
            compounds = pcp.get_compounds(name, 'name')
            
            # Validate SMILES
            smiles = [
                compound.isomeric_smiles 
                for compound in compounds 
                if hasattr(compound, 'isomeric_smiles') and compound.isomeric_smiles
            ]
            
            return smiles if smiles else None
        
        except pcp.PubChemHTTPError as http_err:
            # Handle specific PubChem HTTP errors
            logging.warning(f"PubChem HTTP error for {name} (Attempt {attempt+1}/{max_retries}): {http_err}")
            
            # Check if it's the last retry
            if attempt == max_retries - 1:
                logging.error(f"Failed to retrieve SMILES for {name} after {max_retries} attempts")
            
            continue
        
        except pcp.PubChemTypeError as type_err:
            # Errors related to incorrect input type
            logging.warning(f"PubChem type error for {name}: {type_err}")
            return None
        
        except Exception as e:
            # Catch any other unexpected errors
            logging.error(f"Unexpected error retrieving SMILES for {name}: {e}", exc_info=True)
            return None
    
    return None

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
    irrev_model = convert_to_irreversible(model)
    print(f"Converted to irreversible model with {len(irrev_model.reactions)} reactions")
    
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
        'Substrate partner': 'Reaction_partners'
    })
    
    # Save if output path provided
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        merged_df.to_csv(output_path, index=False)
    
    return merged_df
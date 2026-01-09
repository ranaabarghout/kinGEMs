import os
import time
import urllib.parse
import urllib.request

import pandas as pd

from .dataset import ensure_dir_exists, load_model, retrieve_sequences


def retrieve_sequences_modelseed(model, metadata_dir=None, output_path=None):
    def model_gene_to_csv_gene(gene_id):
        # Replace the last underscore with a colon
        parts = gene_id.rsplit('_', 1)
        if len(parts) == 2:
            return f"{parts[0]}:{parts[1]}"
        return gene_id
    """
    Retrieve protein sequences for genes in a model using rxnXgenes_382_genome.csv in metadata_dir.
    For each gene, finds qseqid in the CSV and fetches the sequence from UniProt/UniRef if possible.
    Returns a DataFrame with columns: Single_gene, qseqid, Sequence
    """
    # Load model if string
    if isinstance(model, str):
        from .dataset import load_model
        model = load_model(model)

    # Find the metadata CSV
    if metadata_dir is None:
        raise ValueError("metadata_dir must be provided for ModelSEED sequence retrieval.")
    csv_path = os.path.join(metadata_dir, "rxnXgenes_382_genome.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    meta_df = pd.read_csv(csv_path)

    # Helper: fetch sequence from UniProt/UniRef
    def fetch_sequence_from_uniprot(qseqid):
        # Accepts formats like 'sp|P08142|ILVB_ECOLI', 'UniRef90_Q8FAG0', 'gnl|TC-DB|P08142', etc.
        # Try to extract the accession
        if pd.isna(qseqid):
            print("  [No qseqid provided]")
            return None
        acc = None
        db_type = None
        if qseqid.startswith('sp|') or qseqid.startswith('tr|'):
            # SwissProt/TrEMBL: sp|P08142|ILVB_ECOLI
            db_type = 'UniProtKB'
            parts = qseqid.split('|')
            if len(parts) >= 2:
                acc = parts[1]
        elif qseqid.startswith('UniRef'):
            # UniRef90_Q8FAG0
            db_type = 'UniRef'
            acc = qseqid.split('_')[-1]
        elif qseqid.startswith('gnl|TC-DB|'):
            # gnl|TC-DB|P08142 → treat as UniProt accession
            db_type = 'UniProtKB'
            parts = qseqid.split('|')
            if len(parts) == 3:
                acc = parts[2]
        else:
            db_type = 'Other'
            acc = qseqid
        print(f"    [qseqid: {qseqid}] [db: {db_type}] [accession: {acc}]", end='')
        if not acc:
            print("  [No accession extracted]")
            return None
        # Try UniProt FASTA endpoint
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                fasta = response.read().decode('utf8')
                seq = ''.join(fasta.split('\n')[1:]).strip()
                if seq:
                    print("  [UniProtKB: HIT]")
                    return seq
        except Exception:
            print("  [UniProtKB: miss]", end='')
            pass
        # Try UniRef FASTA endpoint
        if qseqid.startswith('UniRef'):
            url2 = f"https://rest.uniprot.org/uniref/{qseqid}.fasta"
            try:
                with urllib.request.urlopen(url2, timeout=10) as response:
                    fasta = response.read().decode('utf8')
                    seq = ''.join(fasta.split('\n')[1:]).strip()
                    if seq:
                        print("  [UniRef: HIT]")
                        return seq
            except Exception:
                print("  [UniRef: miss]", end='')
                pass
        print("  [No sequence found]")
        return None

    # Build gene list
    gene_data = []
    for gene in model.genes:
        print(f"Gene: {gene.id}")
        csv_gene_id = model_gene_to_csv_gene(gene.id)
        # Find all rows in meta_df for this gene
        matches = meta_df[meta_df['gene'] == csv_gene_id]
        if not matches.empty:
            # Use the first qseqid (could be extended to all)
            qseqid = matches.iloc[0]['qseqid']
            seq = fetch_sequence_from_uniprot(qseqid)
            # If not found, try next qseqid in matches
            if not seq and len(matches) > 1:
                for _, row in matches.iloc[1:].iterrows():
                    qseqid2 = row['qseqid']
                    seq = fetch_sequence_from_uniprot(qseqid2)
                    if seq:
                        qseqid = qseqid2
                        break
            gene_data.append({'Single_gene': gene.id, 'qseqid': qseqid, 'Sequence': seq})
        else:
            print("    [No qseqid mapping found in metadata]")
            gene_data.append({'Single_gene': gene.id, 'qseqid': None, 'Sequence': None})

    df_genes = pd.DataFrame(gene_data)
    # Save if output path provided
    if output_path:
        from .dataset import ensure_dir_exists
        ensure_dir_exists(os.path.dirname(output_path))
        df_genes.to_csv(output_path, index=False)
    return df_genes



SEED_COMPOUNDS_PATH = os.path.join(os.path.dirname(__file__), '../data/external databases/SEED_compounds.tsv')


def strip_compartment(met_id):
    """Remove compartment suffixes like _e0, _c0, etc."""
    return met_id.split('_')[0]



def prepare_modelseed_model_data(model_path, substrates_output=None, sequences_output=None, organism='E coli', metadata_dir=None, convert_to_irreversible=True):
    """
    Prepare model data using ModelSEED compound IDs to retrieve SMILES.
    - Removes compartment suffixes from metabolite IDs (e.g., cpd00010_e0 -> cpd00010)
    - Looks up SMILES in SEED_compounds.tsv
    - Returns (model, substrate_df_with_smiles, sequences_df)
    metadata_dir: if provided, use as directory for SEED_compounds.tsv
    """
    # Load model
    model = load_model(model_path)
    if convert_to_irreversible:
        model = convert_to_irreversible(model)
        print(f"Converted to irreversible model with {len(model.reactions)} reactions")


    print(f"Loaded model with {len(model.reactions)} reactions and {len(model.metabolites)} metabolites")

    # Extract substrate info
    rxn_data = []
    for reaction in model.reactions:
        substrates = [strip_compartment(met.id) for met in reaction.reactants]
        for substrate in substrates:
            rxn_data.append({
                "Reaction": reaction.id,
                "Substrate partner": substrate,
                "Direction": "forward"
            })
        if getattr(reaction, "reversibility", False):
            products = [strip_compartment(met.id) for met in reaction.products]
            for product in products:
                rxn_data.append({
                    "Reaction": reaction.id,
                    "Substrate partner": product,
                    "Direction": "reverse"
                })
    substrate_df = pd.DataFrame(rxn_data)
    print(f"Extracted {len(substrate_df)} substrate-reaction-direction pairs")

    # Determine SEED compounds path
    # if metadata_dir is not None:
    #     seed_compounds_path = os.path.join('SEED_compounds.tsv')
    # else:
    seed_compounds_path = SEED_COMPOUNDS_PATH

    # Load SEED compounds
    seed_df = pd.read_csv(seed_compounds_path, sep='\t')
    smiles_map = dict(zip(seed_df['id'], seed_df['smiles']))
    name_map = dict(zip(seed_df['id'], seed_df['name']))

    # Map SMILES and names
    substrate_df['SMILES'] = substrate_df['Substrate partner'].map(smiles_map)
    substrate_df['DB Name'] = substrate_df['Substrate partner'].map(name_map)

    num_mapped_metabolites = substrate_df['SMILES'].notna().sum()
    total_metabolites = len(model.metabolites)
    print(f"Mapped metabolites to SMILES ({num_mapped_metabolites} found)")

    # Retrieve protein sequences
    sequences_df = retrieve_sequences_modelseed(model, metadata_dir=metadata_dir)
    num_sequences = sequences_df['Sequence'].notna().sum()
    total_genes = len(model.genes)
    print(f"Retrieved {num_sequences} protein sequences")
    print(f"Summary: {num_mapped_metabolites} / {total_metabolites} metabolites mapped to SMILES.")
    print(f"Summary: {num_sequences} / {total_genes} genes with protein sequences retrieved.")

    # Save outputs if paths provided
    if substrates_output:
        ensure_dir_exists(os.path.dirname(substrates_output))
        substrate_df.to_csv(substrates_output, index=False)
    if sequences_output:
        ensure_dir_exists(os.path.dirname(sequences_output))
        sequences_df.to_csv(sequences_output, index=False)

    return model, substrate_df, sequences_df

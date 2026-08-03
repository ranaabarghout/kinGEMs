import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external databases"

MODELS_DIR = PROJ_ROOT / "models"

ECOLI_VALIDATION_DIR = DATA_DIR / "E_coli fitness validation"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Define paths to database files
BiGG_MAPPING = EXTERNAL_DATA_DIR / "BiGG_mapping.csv"
BIGG_METABOLITES = EXTERNAL_DATA_DIR / "bigg_models_metabolites.txt"
CHEBI_COMPOUNDS = EXTERNAL_DATA_DIR / "CHEBI_compounds.tsv"
CHEBI_INCHI = EXTERNAL_DATA_DIR / "CHEBI_InChI.tsv"
METANETX_COMPOUNDS = EXTERNAL_DATA_DIR / "MetaNetX_compounds.tsv"
METANETX_DEPR = EXTERNAL_DATA_DIR / "MetaNetX_compoundsdepr.tsv"
METANETX_XREF = EXTERNAL_DATA_DIR / "MetaNetX_compoundsxref.tsv"
SEED_COMPOUNDS = EXTERNAL_DATA_DIR / "SEED_compounds.tsv"
SEED_ALIASES = EXTERNAL_DATA_DIR / "Unique_ModelSEED_Compound_Aliases.txt"

# Define taxonomy IDs for sequence retrieval
TAXONOMY_IDS = {
    # Bacteria - Proteobacteria - Enterobacteriaceae
    'E coli': 83333,  # Escherichia coli K-12 MG1655
    'Salmonella': 99287,  # Salmonella enterica subsp. enterica serovar Typhimurium str. LT2
    'Klebsiella': 272620,  # Klebsiella pneumoniae subsp. pneumoniae MGH 78578
    'Yersinia': 214092,  # Yersinia pestis CO92
    
    # Bacteria - Proteobacteria - Other
    'Pseudomonas': 160488,  # Pseudomonas putida KT2440
    'Acinetobacter': 400667,  # Acinetobacter baumannii ATCC 17978
    'Geobacter': 243231,  # Geobacter metallireducens GS-15
    'Burkholderia': 269482,  # Burkholderia cenocepacia J2315
    
    # Bacteria - Firmicutes
    'B subtilis': 224308,  # Bacillus subtilis subsp. subtilis str. 168
    'S aureus': 93061,  # Staphylococcus aureus subsp. aureus NCTC 8325
    'L lactis': 272623,  # Lactococcus lactis subsp. lactis Il1403
    'Clostridium': 272563,  # Clostridioides difficile 630
    'Clostridioides': 272563,  # Clostridioides difficile 630
    'Thermoanaerobacter': 187420,  # Thermoanaerobacter tengcongensis MB4
    
    # Bacteria - Actinobacteria
    'M tuberculosis': 83332,  # Mycobacterium tuberculosis H37Rv
    'Streptomyces': 100226,  # Streptomyces coelicolor A3(2)
    
    # Bacteria - Cyanobacteria
    'Synechococcus': 1140,  # Synechococcus elongatus PCC 7942
    'Synechocystis': 1148,  # Synechocystis sp. PCC 6803
    
    # Bacteria - Other
    'H pylori': 85962,  # Helicobacter pylori 26695
    'Thermotoga': 243274,  # Thermotoga maritima MSB8
    'Bacteroides': 28116,  # Bacteroides ovatus ATCC 8483
    
    # Archaea
    'Methanosarcina': 188937,  # Methanosarcina barkeri str. Fusaro
    'Methanobacterium': 145262,  # Methanobacterium sp. AL-21
    
    # Eukaryotes - Fungi
    'Yeast': 4932,  # Saccharomyces cerevisiae S288C
    'R toruloides': 5286,  # R toruloides
    
    # Eukaryotes - Mammals
    'Human': 9606,  # Homo sapiens
    'Mouse': 10090,  # Mus musculus
    'Chinese Hamster Ovary': 10029,  # Cricetulus griseus
    
    # Eukaryotes - Parasites
    'T cruzi': 5693,  # Trypanosoma cruzi
    'Leishmania': 5664,  # Leishmania major
    'P falciparum': 36329,  # Plasmodium falciparum 3D7
    'P berghei': 5823,  # Plasmodium berghei
    'P chabaudi': 5825,  # Plasmodium chabaudi
    'P knowlesi': 5850,  # Plasmodium knowlesi
    'P vivax': 5855,  # Plasmodium vivax
    'P cynomolgi': 5827,  # Plasmodium cynomolgi
    
    # Eukaryotes - Algae/Diatoms
    'Phaeodactylum': 2850,  # Phaeodactylum tricornutum
    'Chlamydomonas': 3055,  # Chlamydomonas reinhardtii
}

# REST datasets/endpoints to try, in order
REST_DATASETS = [
    'uniprotkb',      # Swiss-Prot + TrEMBL
    'uniref50',       # UniRef50 clusters
    'uniref90',       # UniRef90 clusters
    'uniref100',      # UniRef100 clusters
    'tcdb'            # TC-DB FASTA
]

# Helper function to ensure a directory exists
def ensure_dir_exists(directory):
    os.makedirs(directory, exist_ok=True)

# If tqdm is installed, configure loguru with tqdm.write
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Export variables for import
__all__ = [
    'PROJ_ROOT', 'DATA_DIR', 'RAW_DATA_DIR', 'INTERIM_DATA_DIR',
    'PROCESSED_DATA_DIR', 'EXTERNAL_DATA_DIR', 'MODELS_DIR', 'ECOLI_VALIDATION_DIR',
    'REPORTS_DIR', 'FIGURES_DIR', 'BiGG_MAPPING', 'BIGG_METABOLITES',
    'CHEBI_COMPOUNDS', 'CHEBI_INCHI', 'METANETX_COMPOUNDS', 'METANETX_DEPR',
    'METANETX_XREF', 'SEED_COMPOUNDS', 'SEED_ALIASES', 'TAXONOMY_IDS',
    'ensure_dir_exists'
]
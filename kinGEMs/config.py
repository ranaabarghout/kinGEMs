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
    'E coli': 83333,  # Taxonomy ID for E. coli K-12
    'Yeast': 4932,    # Taxonomy ID for S. cerevisiae
    'S elongatus': 1140,  # Taxonomy ID for S. elongatus PCC 7942: https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=1140
    'P putida': 160488, # Taxonomy ID for P. putida KT2440: https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=160488
    'Helicobacter': 1287064, # Taxonomy ID for Helicobacter pylori UM034: https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=1287064&lvl=3&lin=f&keep=1&srchmode=1&unlock
    'B ovatus': 28116 , # Taxonomy ID for B. ovatus: https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=28116
      # Add more organisms as needed
}

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
    'PROCESSED_DATA_DIR', 'EXTERNAL_DATA_DIR', 'MODELS_DIR',
    'REPORTS_DIR', 'FIGURES_DIR', 'BiGG_MAPPING', 'BIGG_METABOLITES',
    'CHEBI_COMPOUNDS', 'CHEBI_INCHI', 'METANETX_COMPOUNDS', 'METANETX_DEPR',
    'METANETX_XREF', 'SEED_COMPOUNDS', 'SEED_ALIASES', 'TAXONOMY_IDS',
    'ensure_dir_exists'
]
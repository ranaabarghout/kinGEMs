"""
kinGEMs: Kinetic Genome-scale Metabolic Models
==============================================

A package for integrating kinetic information with genome-scale metabolic models
to improve flux predictions by accounting for enzyme kinetics.

Main modules:
------------
dataset      : Data loading, metabolite ID mapping, and model preparation
modeling     : Optimization, FVA, and simulated annealing for parameter tuning
plots        : Visualization tools for model results
"""

__version__ = '0.1.0'

# Import core functionality for easier access
from .dataset import (
    convert_to_irreversible,
    extract_metabolite_annotations,
    load_model,
    map_metabolites,
    prepare_model_data,
    retrieve_sequences,
)
from .modeling import flux_variability_analysis, run_optimization, simulated_annealing
from .plots import plot_annealing_progress, plot_enzyme_usage, plot_flux_distribution

# Define what gets imported with `from kinGEMs import *`
__all__ = [
    # Dataset functions
    'map_metabolites',
    'extract_metabolite_annotations',
    'retrieve_sequences',
    'load_model',
    'convert_to_irreversible',
    'prepare_model_data',
    
    # Modeling functions
    'run_optimization',
    'flux_variability_analysis',
    'simulated_annealing',
    
    # Plotting functions
    'plot_flux_distribution',
    'plot_enzyme_usage',
    'plot_annealing_progress'
]
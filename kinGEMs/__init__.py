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
    map_metabolites,
    retrieve_sequences,
    load_model,
    convert_to_irreversible,
    prepare_model_data
)

from .modeling import (
    run_optimization,
    flux_variability_analysis,
    simulated_annealing
)

from .plots import (
    plot_flux_distribution,
    plot_enzyme_usage,
    plot_annealing_progress
)

# Define what gets imported with `from kinGEMs import *`
__all__ = [
    # Dataset functions
    'map_metabolites',
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
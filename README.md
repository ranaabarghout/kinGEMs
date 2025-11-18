# kinGEMs

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A pipeline for automatic reconstruction of enzyme-constrained genome-scale models using CPI-Pred for kinetic parameter annotation.

## TO DO:
 - move plotting functions in validation (parallel) to src plots.py
 - clean up docs 
 - organize all results in results folder
 
## Quick Start

```bash
# Run the pipeline with a configuration file
python scripts/run_pipeline.py configs/iML1515_GEM.json

# Force regenerate all intermediate files
python scripts/run_pipeline.py configs/iML1515_GEM.json --force
```

For detailed usage instructions, see [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md).

## Documentation

📚 **[Comprehensive documentation is available in the `docs/` directory](docs/README.md)**

Key guides:
- **[Configuration Guide](docs/CONFIG_GUIDE.md)** - How to use the pipeline
- **[Pipeline Summary](docs/PIPELINE_SUMMARY.md)** - Workflow overview
- **[Constraint Explanations](docs/MIXED_CONSTRAINTS_EXPLAINED.md)** - Understanding enzyme constraints
- **[Virtual Environment Setup](docs/VENV_SETUP.md)** - Environment configuration

## Project Organization

```
├── LICENSE                      <- Open-source license
├── Makefile                     <- Convenience commands
├── README.md                    <- Top-level README (this file)
├── environment.yml              <- Conda environment specification
├── requirements.txt             <- Python package requirements
├── pyproject.toml              <- Project configuration
│
├── configs/                     <- JSON configuration files for different models
│   ├── iML1515_GEM.json        <- E. coli iML1515 configuration
│   ├── e_coli_core.json        <- E. coli core model
│   ├── 382_genome_cpd03198.json <- ModelSEED configurations
│   └── ...                     <- Other organism configurations
│
├── data/
│   ├── raw/                    <- Original model files (.xml)
│   ├── interim/                <- Intermediate data (substrates, sequences, CPI-Pred predictions)
│   └── processed/              <- Final processed data ready for modeling
│
├── docs/                        <- 📚 Comprehensive documentation
│   ├── README.md               <- Documentation index
│   ├── CONFIG_GUIDE.md         <- Configuration guide
│   ├── PIPELINE_SUMMARY.md     <- Pipeline workflow
│   ├── MIXED_CONSTRAINTS_EXPLAINED.md <- Enzyme constraint types
│   ├── CONSTRAINT_SKIPPING_EXPLAINED.md <- Troubleshooting
│   └── ...                     <- Additional technical docs
│
├── kinGEMs/                     <- Source code for kinGEMs package
│   ├── __init__.py
│   ├── config.py               <- Configuration variables
│   ├── dataset.py              <- Standard model data processing
│   ├── dataset_modelseed.py    <- ModelSEED-specific processing
│   ├── plots.py                <- Visualization functions
│   ├── validation_utils.py     <- Validation utilities
│   └── modeling/               <- Optimization and tuning modules
│       ├── __init__.py
│       ├── fva.py              <- Flux variability analysis
│       ├── optimize.py         <- Enzyme-constrained optimization
│       └── tuning.py           <- Simulated annealing for kcat tuning
│
├── models/                      <- Final enzyme-constrained GEMs (.xml)
│
├── notebooks/                   <- Jupyter notebooks for analysis
│   ├── Analysis_Notebook.ipynb
│   ├── run_iML1515_GEM.ipynb   <- Example runs for different models
│   └── ...
│
├── results/                     <- Pipeline outputs
│   ├── tuning_results/         <- Simulated annealing results by run_id
│   │   └── {model}_{date}_{id}/
│   │       ├── df_new.csv              <- Enzyme mass contributions
│   │       ├── kcat_dict.csv           <- Tuned kcat values
│   │       ├── final_model_info.csv    <- Complete results
│   │       ├── annealing_progress.png  <- Optimization progress
│   │       └── *_fva_*.csv/png         <- FVA results (if enabled)
│   └── validation/             <- Validation results
│
└── scripts/                     <- Executable scripts
    ├── run_pipeline.py         <- 🚀 Main pipeline script (use this!)
    ├── run_pipeline_with_logging.sh <- Wrapper for automatic logging
    └── ...                     <- Model-specific legacy scripts
```

## Key Features

- **Unified Pipeline**: Single script handles any genome-scale metabolic model
- **Automatic Model Type Detection**: Detects ModelSEED vs standard models automatically
- **Enzyme Constraint Types**: Handles AND (complexes), OR (isoenzymes), and promiscuous constraints
- **Simulated Annealing**: Optimizes kcat values to improve model predictions
- **File Caching**: Speeds up subsequent runs by reusing intermediate files
- **Flexible Configuration**: JSON-based configs for easy model switching
- **Optional Analyses**: FVA and Biolog validation on demand

## Installation

### Prerequisites

- Python 3.9
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git (for cloning the repository)

### Quick Start

We recommend using [mamba](https://github.com/mamba-org/mamba) since the linear programming libraries
require non-python dependencies (i.g. [glpk](https://www.gnu.org/software/glpk/) & glpsol.exe)

```bash
   git clone https://github.com/FILL_ME/kinGEMs.git
   cd kinGEMs
   mamba env create -f enviroment.yml
```

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/kinGEMs.git
   cd kinGEMs
   ```

2. **Create and activate a conda environment:**

   ```bash
   conda create -n kingems python=3.9 -y
   conda activate kingems
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install System-Level Dependencies:**

   **GLPK Solver**:
   The GLPK (GNU Linear Programming Kit) solver is required for optimization operations. There are two options for installation:

   **_Option A: Using Conda (Recommended)_**

   1. Run:

   ```bash
   conda install -c conda-forge glpk
   ```

   2. (On Windows) Ensure the following directory is added to your system PATH environment variable:

   ```bash
   C:\Users\<your-username>\anaconda3\envs\kingems\Library\bin
   ```

   This step is required so that Pyomo can locate the `glpsol.exe` executable.

   **_Option B: Manual Installation (Windows)_**

   1. Get the precompiled binary for Windows from [GLPK for Windows](https://sourceforge.net/projects/winglpk/).
   2. Extract the downloaded files to a directory on your system (e.g., `C:\glpk`).
   3. Add the directory containing `glpsol.exe` to your system's PATH environment variable.

   **_Verify GLPK Installation_**

   After installing GLPK, verify that it is accessible by running:

   ```bash
   glpsol --version
   ```

## Set up with uv

The following steps will set up a reproducible Python environment using `uv`, a fast and lightweight package manager.

### 1. Install uv 
This will install a single static binary in your `$HOME/bin`:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### 2. Navigate to the repository
```
cd ~/kinGEMs_v2
```

### 3. Create a virtual environment
Creates an isolated environment inside the repository:
```
uv venv .venv
```

### 4. Activate venv
```
source .venv/bin/activate
```

### 5. Install dependencies
Synchronize and install all dependencies listed in pyproject.toml:
```
uv sync
```
   This automatically resolves and locks dependencies for reproducibility (`uv.lock` file).

## Updating the environment
Whenever you modify dependencies (add or update a package), run:
```
uv sync --upgrade
```

or to install a single new package:
```
uv add <package-name>
```
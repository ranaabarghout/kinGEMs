# kinGEMs v2 - Complete System Summary

This document provides a complete overview of the kinGEMs v2 pipeline system.

## Quick Start

### Run a Single Model

```bash
# Activate environment
source venv/bin/activate

# Run pipeline with config file
python scripts/run_pipeline.py configs/iML1515_GEM.json

# Force regenerate (ignore cached files)
python scripts/run_pipeline.py configs/iML1515_GEM.json --force
```

### Batch Process Multiple Models

```bash
# Process all E. coli models
bash scripts/batch_run.sh ecoli

# Process all ModelSEED models
bash scripts/batch_run.sh genome

# Process all standard (BiGG) models
bash scripts/batch_run.sh standard

# Process all models
bash scripts/batch_run.sh all
```

## System Architecture

### Pipeline Flow

```
1. Data Preparation
   ↓
2. CPI-Pred Merge
   ↓
3. kcat Processing
   ↓
4. FBA Optimization
   ↓
5. Simulated Annealing (Tuning)
   ↓
6. Experimental Validation (Optional)
```

### Key Components

```
kinGEMs_v2/
├── configs/                    # Model configurations (12 JSON files)
├── scripts/
│   ├── run_pipeline.py         # Main pipeline script
│   └── batch_run.sh            # Batch processing
├── kinGEMs/
│   ├── dataset.py              # Standard model functions
│   ├── dataset_modelseed.py    # ModelSEED model functions
│   └── modeling/
│       └── optimize.py         # Optimization with solver support
├── data/
│   ├── raw/                    # Input .xml model files
│   └── interim/
│       └── CPI-Pred predictions/  # Enzyme predictions
└── results/
    ├── tuning_results/         # Pipeline outputs (NOT in git)
    └── validation/             # Validation results (NOT in git)
```

## Available Models

### E. coli Models
- **iML1515_GEM** - Full E. coli BiGG model (2,712 reactions)
- **e_coli_core** - Core metabolism model (95 reactions)

### ModelSEED Models
- **382_genome_cpd03198** - With Biolog validation
- **376_genome_cpd03198**
- **378_genome_cpd03198**
- **380_genome_cpd01262**

### Other Organisms
- **yeast-GEM9** - Saccharomyces cerevisiae
- **Kmarxianus_GEM** - Kluyveromyces marxianus
- **Lmajor_GEM** - Leishmania major
- **Pputida_iJN1463** - Pseudomonas putida (large)
- **Pputida_iJN746** - Pseudomonas putida (small)
- **Selongatus_iJB785** - Synechococcus elongatus

## Configuration Files

Each model has a JSON config in `configs/`:

```json
{
  "model_name": "iML1515_GEM",
  "organism": "E coli",
  "biomass_reaction": "BIOMASS_Ec_iML1515_core_75p37M",
  "enzyme_upper_bound": 0.15,
  "solver": "glpk",
  "enable_fva": true,
  "enable_biolog_validation": false
}
```

**Key Parameters:**
- `model_name` - Must match .xml filename in data/raw/
- `biomass_reaction` - Objective function to optimize
- `enzyme_upper_bound` - Maximum total enzyme mass constraint
- `solver` - `glpk` (default), `gurobi`, `cplex`, or `ipopt`
- `enable_fva` - Run flux variability analysis
- `enable_biolog_validation` - Run experimental validation (ModelSEED models only)

## Solvers

### GLPK (Default)
- **Free and open source**
- No license required
- Works everywhere
- Good for most models

### Gurobi
- Commercial solver
- Requires license (not available on Compute Canada for all users)
- Best performance for large models
- Edit config: `"solver": "gurobi"`

### CPLEX / IPOPT
- Alternative commercial/open source solvers
- Edit config: `"solver": "cplex"` or `"solver": "ipopt"`

**See `SOLVER_GUIDE.md` for detailed configuration.**

## Model Type Detection

Pipeline automatically detects model type:

```python
# ModelSEED models (use dataset_modelseed.py)
if '_genome_' in model_name.lower():
    use dataset_modelseed functions

# Standard models (use dataset.py)  
else:
    use dataset functions
```

**ModelSEED models:** 376_genome_*, 378_genome_*, 380_genome_*, 382_genome_*  
**Standard models:** All others

## File Caching

Pipeline caches intermediate files to save time:

**Cached Files:**
- Step 1: `{model}_merged_data.csv` (data preparation)
- Step 2: `{model}_df_new_kinGEMs.csv` (CPI-Pred merge)
- Step 3: `{model}_processed_df.csv` (kcat processing)

**Force Regenerate:**
```bash
python scripts/run_pipeline.py configs/iML1515_GEM.json --force
```

**See `SCRIPT_CACHING_GUIDE.md` for details.**

## CPI-Pred Predictions

Pipeline automatically finds prediction files using multiple patterns:

```python
# Tries these patterns in order:
1. "X06A_kinGEMs_{model_name}_predictions.csv"
2. "*{model_name}*predictions.csv"
3. "*{model_name without _GEM}*predictions.csv"
4. "*ecoli*{base_name}*predictions.csv"  # E. coli special case
```

**Supported naming:**
- `X06A_kinGEMs_iML1515_GEM_predictions.csv` ✓
- `X06A_kinGEMs_ecoli_iML1515_predictions.csv` ✓
- `iML1515_predictions.csv` ✓

**See `PREDICTIONS_FILE_GUIDE.md` for details.**

## Output Files

Each run creates a timestamped directory:

```
results/tuning_results/{model}_{date}_{id}/
├── df_new.csv                  # Enzyme masses (100-400 MB)
├── final_model_info.csv        # Merged results (100-400 MB)
├── kcat_dict.csv               # Tuned kcat values (<1 MB)
├── df_FBA.csv                  # FBA results (1-10 MB)
├── iterations.csv              # Annealing history (<1 MB)
└── annealing_progress.png      # Convergence plot (<1 MB)
```

**Note:** These files are **excluded from git** due to large size.

**See `results/README.md` for file descriptions and management.**

## Git Repository

### What's Tracked
✓ Source code (kinGEMs/)
✓ Scripts (scripts/)
✓ Configs (configs/)
✓ Documentation (*.md)
✓ Requirements (requirements.txt, environment.yml)

### What's Excluded
✗ Data files (data/)
✗ Results (results/tuning_results/, results/validation/)
✗ Virtual environment (venv/)
✗ Cache files (__pycache__/, *.pyc)
✗ Jupyter checkpoints (.ipynb_checkpoints/)

### Repository Status
- **Clean history:** Large files removed
- **Synced with GitHub:** Latest push successful
- **Size optimized:** Garbage collected

## Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and setup |
| `QUICK_REFERENCE.md` | Command cheat sheet |
| `PIPELINE_SUMMARY.md` | Pipeline architecture |
| `CONFIG_GUIDE.md` | Configuration reference (400+ lines) |
| `SOLVER_GUIDE.md` | Solver setup and troubleshooting |
| `SCRIPT_CACHING_GUIDE.md` | Caching feature guide |
| `PREDICTIONS_FILE_GUIDE.md` | CPI-Pred file naming |
| `results/README.md` | Results management |
| `SYSTEM_SUMMARY.md` | This file |

## Common Tasks

### Add a New Model

1. **Get the model file:**
   ```bash
   cp new_model.xml data/raw/new_model_GEM.xml
   ```

2. **Create config:**
   ```bash
   cp configs/iML1515_GEM.json configs/new_model_GEM.json
   # Edit with your model's parameters
   ```

3. **Ensure predictions file exists:**
   ```bash
   ls data/interim/CPI-Pred\ predictions/*new_model*
   ```

4. **Run pipeline:**
   ```bash
   python scripts/run_pipeline.py configs/new_model_GEM.json
   ```

### Change Solver

Edit the config file:
```json
{
  "solver": "glpk"     // Change to "gurobi", "cplex", or "ipopt"
}
```

### Enable FVA

Edit the config file:
```json
{
  "enable_fva": true   // Change from false to true
}
```

### Run Biolog Validation

Only for ModelSEED models:
```json
{
  "enable_biolog_validation": true
}
```

Requires Biolog data in `data/Biolog experiments/`.

### Clean Old Results

```bash
# Check disk usage
du -sh results/tuning_results/

# Remove specific run
rm -rf results/tuning_results/iML1515_GEM_20250826_4941/

# Remove all (careful!)
rm -rf results/tuning_results/*/
```

### Debug Pipeline

1. **Check cached files:**
   ```bash
   ls data/interim/382_genome_cpd03198/
   ```

2. **Force regenerate:**
   ```bash
   python scripts/run_pipeline.py configs/382_genome_cpd03198.json --force
   ```

3. **Check predictions file:**
   ```bash
   ls data/interim/CPI-Pred\ predictions/*382_genome*
   ```

4. **Check solver:**
   ```bash
   python -c "from cobra.util import solver; print(solver.solvers)"
   ```

## Environment Setup

### Virtual Environment

```bash
# Create
python -m venv venv

# Activate
source venv/bin/activate

# Install
pip install -r requirements.txt
```

**See `VENV_SETUP.md` for detailed setup.**

### Python Environment

- **Python:** 3.11.4
- **Location:** /project/def-mahadeva/ranaab/kinGEMs_v2/venv
- **Platform:** Compute Canada HPC

### Key Dependencies

- cobra
- pyomo
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn

## Performance Notes

### Typical Runtime

| Model | Reactions | Step 4 | Step 5 | Total |
|-------|-----------|--------|--------|-------|
| e_coli_core | 95 | 1-2 min | 5-10 min | ~15 min |
| iML1515_GEM | 2,712 | 5-10 min | 20-40 min | ~1 hour |
| 382_genome_* | ~1,500 | 3-5 min | 15-30 min | ~45 min |

**Factors:**
- Model size (reactions/genes)
- Solver choice (GLPK slower than Gurobi)
- FVA enabled (adds 50-100% time)
- System load

### Disk Usage

- **Per run:** 300-500 MB
- **With FVA:** +100-200 MB
- **All 12 models:** ~5-8 GB

### Memory Requirements

- **Small models (<500 rxns):** ~1-2 GB
- **Medium models (500-1500 rxns):** ~2-4 GB
- **Large models (>1500 rxns):** ~4-8 GB

## Troubleshooting

### "Solver not found"

**Solution:** Install solver or change config to `"solver": "glpk"`

### "Predictions file not found"

**Solution:** Check file naming in `data/interim/CPI-Pred predictions/`
```bash
ls data/interim/CPI-Pred\ predictions/
```

### "Biomass reaction not found"

**Solution:** Check reaction ID in model
```python
from cobra.io import read_sbml_model
model = read_sbml_model('data/raw/your_model.xml')
print([r.id for r in model.reactions if 'biomass' in r.id.lower()])
```

### "Disk quota exceeded"

**Solution:** Clean old results or move to project space
```bash
mv results/tuning_results/* ~/projects/def-mahadeva/kinGEMs_results/
```

### "Git push rejected - file too large"

**Solution:** Files should be in .gitignore. If you see this:
```bash
git rm --cached -r results/
git commit -m "Remove large files"
```

## Getting Help

1. **Check documentation:**
   - Start with `QUICK_REFERENCE.md`
   - See specific guides for detailed info

2. **Check configuration:**
   - Verify config file syntax
   - Check solver availability
   - Confirm file paths

3. **Run with --force:**
   - Regenerate cached files
   - May resolve stale data issues

4. **Check logs:**
   - Pipeline prints progress to console
   - Check for error messages in output

## Next Steps

### Basic Usage
1. Activate venv: `source venv/bin/activate`
2. Run a model: `python scripts/run_pipeline.py configs/e_coli_core.json`
3. Check results: `ls results/tuning_results/`

### Advanced Usage
1. Batch process: `bash scripts/batch_run.sh ecoli`
2. Enable FVA: Edit config `"enable_fva": true`
3. Try different solver: Edit config `"solver": "gurobi"`

### Analysis
1. Load results: `import pandas as pd; df = pd.read_csv('results/.../final_model_info.csv')`
2. Check convergence: View `annealing_progress.png`
3. Analyze fluxes: Load `df_FBA.csv`

## Project Status

✅ **Complete and tested:**
- Universal pipeline system
- 12 model configurations
- Batch processing
- Solver flexibility (GLPK, Gurobi, CPLEX, IPOPT)
- Automatic model type detection
- File caching with force regenerate
- CPI-Pred file discovery
- Comprehensive documentation
- Clean git repository

🎯 **Ready for:**
- Production runs
- Batch processing all models
- New model additions
- Result analysis

---

**Last Updated:** January 2025  
**Version:** kinGEMs v2  
**Repository:** https://github.com/ranaabarghout/kinGEMs_v2

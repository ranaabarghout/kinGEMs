# kinGEMs

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Automated enzyme-constrained genome-scale model reconstruction with kinetic parameter prediction.**

kinGEMs combines CPI-Pred kinetic parameter prediction with genome-scale metabolic models to create enzyme-constrained models with optimized kcat values through simulated annealing.

---

## Quick Start

### Basic Usage

```bash
# Run full pipeline (data processing → optimization → tuning)
python scripts/run_pipeline.py configs/iML1515_GEM.json

# Force regenerate all intermediate files
python scripts/run_pipeline.py configs/iML1515_GEM.json --force

# Include flux variability analysis
python scripts/run_pipeline.py configs/iML1515_GEM_parallel_fva.json
```

### Available Analyses

**FVA Ablation Study** - Compare constraint levels systematically:
```bash
# With biomass constraint (90-100% optimal)
python scripts/run_fva_ablation.py configs/iML1515_GEM_fva_ablation.json --parallel

# Without biomass constraint (explore full flux space)
python scripts/run_fva_ablation.py configs/iML1515_GEM_fva_ablation_no_biomass_constraint.json --parallel --no-biomass-constraint
```

**SLURM Users:**
```bash
sbatch slurm_jobs/run_pipeline.sh configs/iML1515_GEM.json
sbatch slurm_jobs/run_fva_ablation.sh
sbatch slurm_jobs/run_fva_ablation_no_biomass_constraint.sh
```

### Common Scenarios

**Test a new model:**
```bash
# 1. Place your model.xml in data/raw/
# 2. Create config file (copy from configs/iML1515_GEM.json)
# 3. Run pipeline
python scripts/run_pipeline.py configs/my_model.json
```

**Tune existing model:**
```bash
# Enable simulated annealing in config, then:
python scripts/run_pipeline.py configs/my_model.json
# Results in: results/tuning_results/{model}_{date}_{id}/
```

**Run maintenance parameter sweep:**
```bash
# Set "enable_maintenance_sweep": true in config, then:
python scripts/run_pipeline.py configs/iML1515_GEM.json
# Finds optimal NGAM/GAM values automatically
```

---

## Configuration Files

All configuration is done through JSON files in `configs/`. Key parameters:

```json
{
  "model_name": "iML1515_GEM",
  "organism": "E coli",
  "enzyme_upper_bound": 0.25,           // Enzyme pool constraint
  "enable_fva": true,                   // Run flux variability analysis
  "enable_maintenance_sweep": true,     // Optimize NGAM/GAM parameters

  "simulated_annealing": {
    "biomass_goal": 0.87,               // Target biomass
    "n_top_enzymes": 500,               // Number of enzymes to tune
    "max_iterations": 1000              // Optimization iterations
  },

  "fva": {
    "parallel": true,                   // Use parallel FVA
    "workers": 8,                       // Number of CPU cores
    "chunk_size": 50                    // Reactions per task
  }
}
```

**Example configs:**
- `iML1515_GEM.json` - Basic E. coli pipeline
- `iML1515_GEM_parallel_fva.json` - With parallel FVA
- `iML1515_GEM_fva_ablation.json` - FVA ablation study
- `iML1515_GEM_fva_ablation_no_biomass_constraint.json` - Unconstrained FVA

See [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md) for full details.

## Output Files

After running the pipeline, find results in `results/tuning_results/{model}_{date}_{id}/`:

| File | Description |
|------|-------------|
| `final_model_info.csv` | Complete enzyme data with tuned kcat values |
| `kcat_dict.csv` | Mapping of reaction-gene pairs to tuned kcats |
| `annealing_progress.png` | Biomass optimization over iterations |
| `model_config_summary.json` | Run configuration and optimal parameters |
| `*_fva_results.csv` | Flux variability analysis (if enabled) |
| `maintenance_sweep_results.csv` | NGAM/GAM sweep (if enabled) |

Final enzyme-constrained model saved to: `models/{model}_{date}_{id}.xml`

---

## Key Features

- **One-Command Pipeline**: `run_pipeline.py` handles everything from data processing to optimization
- **Smart Caching**: Reuses intermediate files (use `--force` to regenerate)
- **Automated Tuning**: Simulated annealing optimizes kcat values for target biomass
- **Flexible Constraints**: Handles enzyme complexes (AND), isoenzymes (OR), and promiscuous reactions
- **Parallel FVA**: Dask-based parallelization with chunking for large models
- **Maintenance Optimization**: Automatic NGAM/GAM parameter sweep
- **Model Agnostic**: Works with standard models and ModelSEED (auto-detected)

---

## Documentation

**Essential Guides:**
- [Configuration Guide](docs/CONFIG_GUIDE.md) - Complete config reference
- [Pipeline Summary](docs/PIPELINE_SUMMARY.md) - Step-by-step workflow
- [Constraint Types](docs/MIXED_CONSTRAINTS_EXPLAINED.md) - Understanding enzyme constraints

**Troubleshooting:**
- [CPLEX Setup](docs/CPLEX_SETUP.md) - Commercial solver configuration
- [Parallel FVA](docs/PARALLEL_VALIDATION_GUIDE.md) - Performance tuning
- [Solver Guide](docs/SOLVER_GUIDE.md) - GLPK vs CPLEX comparison

[Full documentation index](docs/README.md)

---

## Installation

### Quick Install (Recommended)

```bash
# Clone repository
git clone https://github.com/your-username/kinGEMs.git
cd kinGEMs

# Create environment with mamba (faster) or conda
mamba env create -f environment.yml
# OR: conda env create -f environment.yml

# Activate environment
conda activate kingems

# Verify installation
python scripts/run_pipeline.py --help
```

### Manual Install

```bash
# Create environment
conda create -n kingems python=3.11 -y
conda activate kingems

# Install dependencies
pip install -r requirements.txt

# Install GLPK solver
conda install -c conda-forge glpk

# Verify
glpsol --version
python -c "import cobra; print(cobra.__version__)"
```

**Solver Options:**
- **GLPK** (free, default): Good for most models
- **CPLEX** (commercial): Faster for large models, parallel FVA
  - See [docs/CPLEX_SETUP.md](docs/CPLEX_SETUP.md) for installation

---

## 🔬 Typical Workflow

```bash
# 1. Place your model in data/raw/
cp my_model.xml data/raw/

# 2. Create config (copy and edit existing one)
cp configs/iML1515_GEM.json configs/my_model.json
# Edit: model_name, organism, enzyme_upper_bound, biomass_goal

# 3. Run pipeline
python scripts/run_pipeline.py configs/my_model.json

# 4. Check results
ls results/tuning_results/my_model_*/
# View: final_model_info.csv, annealing_progress.png

# 5. Load optimized model
# Your enzyme-constrained model is in: models/my_model_*.xml
```

---

## Project Structure

```
kinGEMs/
├── configs/              # JSON configuration files
├── data/
│   ├── raw/              # Input models (.xml)
│   ├── interim/          # CPI-Pred predictions, substrates, sequences
│   └── processed/        # Processed enzyme data
├── scripts/
│   ├── run_pipeline.py           # 🚀 Main pipeline (START HERE)
│   └── run_fva_ablation.py       # FVA ablation study
├── slurm_jobs/           # SLURM batch scripts
├── results/
│   ├── tuning_results/   # Optimization outputs
│   └── fva_ablation/     # FVA analysis results
├── models/               # Final enzyme-constrained GEMs
└── kinGEMs/              # Source code
    └── modeling/         # Optimization, FVA, tuning modules
```

---

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

# Contact

For questions or issues, please [open an issue](https://github.com/ranaabarghout/kinGEMs/issues) on GitHub.

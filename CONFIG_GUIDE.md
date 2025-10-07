# kinGEMs Pipeline Configuration Guide

This guide explains how to use the general pipeline script with configuration files for different genome-scale metabolic models (GEMs).

## Overview

The `run_pipeline.py` script provides a unified interface for processing any GEM. It automatically detects the model type and applies the appropriate functions from `kinGEMs.dataset` or `kinGEMs.dataset_modelseed`.

## Quick Start

```bash
# Run pipeline for E. coli iML1515
python scripts/run_pipeline.py configs/iML1515_GEM.json

# Run pipeline with forced regeneration of intermediate files
python scripts/run_pipeline.py configs/382_genome_cpd03198.json --force

# Run pipeline for yeast model
python scripts/run_pipeline.py configs/yeast-GEM9.json
```

## Model Type Detection

The script automatically determines which dataset functions to use:

- **Models with `_genome_` in filename** → Uses `kinGEMs.dataset_modelseed` functions
  - Examples: `382_genome_cpd03198`, `376_genome_cpd03198`, `380_genome_cpd01262`
  - These are ModelSEED-based models requiring metadata files

- **Other models** → Uses standard `kinGEMs.dataset` functions
  - Examples: `iML1515_GEM`, `e_coli_core`, `yeast-GEM9`, `Kmarxianus_GEM`
  - These are BiGG or other standard format models

## Configuration File Structure

### Required Fields

```json
{
  "model_name": "iML1515_GEM",
  "organism": "E coli",
  "enzyme_upper_bound": 0.15
}
```

- **model_name**: Name of the XML file in `data/raw/` (without .xml extension)
- **organism**: Organism name for sequence retrieval
- **enzyme_upper_bound**: Maximum enzyme mass fraction (default: 0.15)

### Optional Fields

#### Biomass Reaction
```json
{
  "biomass_reaction": "BIOMASS_Ec_iML1515_core_75p37M"
}
```
If `null` or omitted, the script will auto-detect from the model's objective function.

#### Analysis Options
```json
{
  "enable_fva": true,
  "enable_biolog_validation": false
}
```

- **enable_fva**: Run flux variability analysis and generate comparison plots
- **enable_biolog_validation**: Run experimental validation against Biolog data

#### Simulated Annealing Parameters
```json
{
  "simulated_annealing": {
    "temperature": 1.0,
    "cooling_rate": 0.975,
    "min_temperature": 0.001,
    "max_iterations": 100,
    "max_unchanged_iterations": 5,
    "change_threshold": 0.009,
    "biomass_goal": 0.5
  }
}
```

All parameters are optional and will use defaults if not specified.

#### ModelSEED-Specific Configuration
```json
{
  "metadata_dir": "data/Biolog experiments"
}
```

Only needed for `_genome_` models. Points to directory containing metadata files.

#### Biolog Validation Configuration
```json
{
  "biolog_validation": {
    "experiments_file": "data/Biolog experiments/FBA_results.xlsx",
    "sheet_name": "Ecoli",
    "reference_compound": "cpd00027",
    "uptake_rate": 100.0,
    "blocked_compounds": [
      "cpd00224", "cpd00122", "cpd00609", ...
    ]
  }
}
```

Only needed when `enable_biolog_validation` is `true`.

## Available Model Configurations

The following pre-configured files are available in the `configs/` directory:

### E. coli Models
- `iML1515_GEM.json` - E. coli iML1515 (BiGG model, FVA enabled)
- `e_coli_core.json` - E. coli core model (BiGG model, FVA enabled)
- `382_genome_cpd03198.json` - ModelSEED model (Biolog validation enabled)
- `376_genome_cpd03198.json` - ModelSEED model
- `378_genome_cpd03198.json` - ModelSEED model
- `380_genome_cpd01262.json` - ModelSEED model

### Other Organisms
- `yeast-GEM9.json` - S. cerevisiae GEM (FVA enabled)
- `Kmarxianus_GEM.json` - K. marxianus GEM (FVA enabled)
- `Lmajor_GEM.json` - L. major GEM (FVA enabled)
- `Pputida_iJN1463.json` - P. putida iJN1463 (FVA enabled)
- `Pputida_iJN746.json` - P. putida iJN746 (FVA enabled)
- `Selongatus_iJB785.json` - S. elongatus iJB785 (FVA enabled)

## File Caching

The pipeline automatically caches intermediate files to save computation time:

### Cached Files
- **Step 1**: `{model_name}_substrates.csv`, `{model_name}_sequences.csv`
- **Step 2**: `{model_name}_merged_data.csv`
- **Step 3**: `{model_name}_processed_data.csv`

### Force Regeneration
Use the `--force` or `-f` flag to regenerate all intermediate files:

```bash
python scripts/run_pipeline.py configs/iML1515_GEM.json --force
```

This is useful when:
- Input model file has been updated
- CPI-Pred predictions have changed
- You want to test different parameters in Steps 1-3

## Pipeline Steps

1. **Prepare Model Data**
   - Extract substrates and gene sequences
   - Uses `prepare_model_data()` or `prepare_modelseed_model_data()` based on model type

2. **Merge Data**
   - Combine substrate and sequence information
   - Create enzyme-substrate pairs

3. **Process kcat Predictions**
   - Load CPI-Pred predictions
   - Annotate model with kcat values

4. **Enzyme-Constrained Optimization**
   - Run initial FBA with enzyme constraints
   - Calculate baseline biomass production

5. **Simulated Annealing**
   - Tune kcat values to reach biomass goal
   - Identify key enzymes by mass contribution

6. **Optional Analyses**
   - Flux Variability Analysis (if `enable_fva: true`)
   - Biolog Experimental Validation (if `enable_biolog_validation: true`)

7. **Save Final Model**
   - Export enzyme-constrained GEM with tuned kcat values
   - Save to `models/{run_id}.xml`

## Output Structure

```
results/tuning_results/{run_id}/
├── df_new.csv                    # Enzyme mass contributions
├── kcat_dict.csv                 # Tuned kcat values
├── final_model_info.csv          # Merged results with kcat_tuned
├── {model_name}_fva_results.csv  # FVA results (if enabled)
├── {model_name}_fva_flux_range_plot.png
├── {model_name}_fva_cumulative_plot.png
├── biolog_comparison.csv         # Biolog validation (if enabled)
└── biolog_comparison.png

models/
└── {model_name}_{date}_{random}.xml  # Final enzyme-constrained GEM
```

## Creating New Configurations

To create a configuration for a new model:

1. Add the model XML file to `data/raw/`
2. Create a new JSON config file in `configs/`
3. Specify at minimum: `model_name`, `organism`, `enzyme_upper_bound`
4. Set `enable_fva` and/or `enable_biolog_validation` as needed
5. Run the pipeline: `python scripts/run_pipeline.py configs/your_model.json`

### Example: New Model Configuration

```json
{
  "model_name": "my_new_model",
  "organism": "My organism",
  "biomass_reaction": null,
  "enzyme_upper_bound": 0.15,
  "enable_fva": true,
  "enable_biolog_validation": false,
  "simulated_annealing": {
    "temperature": 1.0,
    "cooling_rate": 0.95,
    "min_temperature": 0.01,
    "max_iterations": 100,
    "max_unchanged_iterations": 5,
    "change_threshold": 0.009,
    "biomass_goal": 0.5
  }
}
```

## Troubleshooting

### Model Not Found
**Error**: `FileNotFoundError: data/raw/my_model.xml`

**Solution**: Ensure the XML file exists and `model_name` matches the filename (without .xml)

### No CPI-Pred Predictions
**Error**: `FileNotFoundError: No CPI-Pred predictions file found`

**Solution**: The pipeline automatically searches for predictions files with flexible naming patterns:
- `X06A_kinGEMs_{model_name}_predictions.csv`
- `X06A_kinGEMs_ecoli_{model_base}_predictions.csv`
- Other variations

If predictions are missing, you'll see a list of available files. Ensure you have:
1. Run CPI-Pred to generate predictions for your model
2. Placed the predictions CSV in `data/interim/CPI-Pred predictions/`
3. Named the file following the `X06A_kinGEMs_*_predictions.csv` pattern

**Example naming patterns that work:**
- `iML1515_GEM` → finds `X06A_kinGEMs_ecoli_iML1515_predictions.csv`
- `382_genome_cpd03198` → finds `X06A_kinGEMs_382_genome_cpd03198_predictions.csv`

### Biomass Reaction Not Found
**Error**: `No objective reaction found in model`

**Solution**: Check your model's objective function. Set `biomass_reaction` explicitly in config if auto-detection fails.

### ModelSEED Metadata Missing
**Error**: `FileNotFoundError: .../rxnXgenes_my_model.csv`

**Solution**: For `_genome_` models, ensure metadata files are in the specified `metadata_dir`

## Performance Tips

1. **Use Caching**: Don't use `--force` unless necessary. Cached files provide ~98% speedup.
2. **Disable Unnecessary Analyses**: Set `enable_fva: false` and `enable_biolog_validation: false` if not needed.
3. **Adjust SA Parameters**: Reduce `max_iterations` or increase `max_unchanged_iterations` for faster tuning.
4. **Parallel Runs**: Process multiple models in parallel by running separate pipeline instances.

## See Also

- `SCRIPT_CACHING_GUIDE.md` - Detailed caching information
- `VENV_SETUP.md` - Virtual environment setup
- `scripts/run_iML1515_GEM.py` - Model-specific script example
- `scripts/run_382_genome_cpd03198_GEM.py` - ModelSEED script example

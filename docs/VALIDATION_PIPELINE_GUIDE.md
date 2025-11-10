# kinGEMs Validation Pipeline Guide

## Overview

The `run_validation_pipeline.py` script provides a unified, config-driven approach to validate kinGEMs models against experimental data. It can compare up to three model versions:

1. **Baseline GEM** - Standard genome-scale model without enzyme constraints
2. **Pre-tuning kinGEMs** - Enzyme-constrained model with initial kcat values (before simulated annealing)
3. **Post-tuning kinGEMs** - Enzyme-constrained model with tuned kcat values (after simulated annealing)

## Quick Start

```bash
# Run validation with all three models
python scripts/run_validation_pipeline.py configs/validation_iML1515.json

# Create a custom config for your model
cp configs/validation_iML1515.json configs/validation_my_model.json
# Edit the config file...
python scripts/run_validation_pipeline.py configs/validation_my_model.json
```

## Configuration File Structure

### Required Fields

```json
{
  "model_name": "iML1515_GEM",
  "model_path": "models/ecoli_iML1515_20250826_4941.xml",
  "objective_reaction": "BIOMASS_Ec_iML1515_core_75p37M",
  "enzyme_upper_bound": 0.15
}
```

- **model_name**: Identifier for this validation run
- **model_path**: Path to the SBML model file
- **objective_reaction**: Biomass reaction ID (can be `null` for auto-detection)
- **enzyme_upper_bound**: Maximum enzyme mass fraction (default: 0.15)

### Model Comparison Configuration

```json
{
  "run_baseline": true,
  "run_pre_tuning": true,
  "run_post_tuning": true
}
```

Control which models to include in the comparison. You can:
- Run all three for comprehensive comparison
- Run only baseline + post-tuning for before/after analysis
- Run only baseline for quick validation

### Data Paths

```json
{
  "pre_tuning_data_path": "data/processed/iML1515_GEM/iML1515_GEM_processed_data.csv",
  "post_tuning_data_path": "results/tuning_results/ecoli_iML1515_20250826_4941/df_new.csv"
}
```

**Pre-tuning data**: Output from Step 3 of the pipeline (processed kcat predictions)
- File: `{model_name}_processed_data.csv` from `data/processed/{model_name}/`
- Contains: Initial kcat values from CPI-Pred without tuning

**Post-tuning data**: Output from Step 5 of the pipeline (simulated annealing results)
- File: `df_new.csv` from `results/tuning_results/{run_id}/`
- Contains: Tuned kcat values after simulated annealing optimization

### Threshold Parameters

```json
{
  "sim_thresh": 0.001,
  "fit_thresh": -2
}
```

- **sim_thresh**: Minimum growth rate to classify as "growth" in simulations (default: 0.001)
- **fit_thresh**: Threshold for experimental fitness data classification (default: -2)

## What the Pipeline Does

### Step 1: Load Model and Experimental Data
- Loads the SBML model
- Auto-detects objective reaction if not specified
- Checks kcat coverage in the model
- Loads Keio collection experimental fitness data

### Step 2: Prepare Validation Environment
- Sets appropriate exchange reaction bounds
- Loads medium and carbon source definitions
- Matches model genes with experimental data
- Adjusts for strain-specific and essential gene information

### Step 3: Run Growth Simulations
For each enabled model type:
- Simulates growth for all gene knockout × carbon source combinations
- Baseline: Uses COBRApy's `slim_optimize()` (fast)
- kinGEMs: Uses enzyme-constrained FBA with kcat values (slower)
- Progress tracking for long runs

### Step 4: Calculate Performance Metrics
**Basic Metrics:**
- Accuracy: Correct predictions / Total predictions
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1 Score: Harmonic mean of precision and recall

**Advanced Metrics:**
- AUC: Area Under the Precision-Recall Curve
- Balanced Accuracy: Average of sensitivity and specificity
- ROC AUC: Area Under the ROC Curve

### Step 5: Essential Genes Analysis
Calculates all metrics specifically for essential genes (genes required for growth on all carbon sources)

### Step 6: Genes with kcat Analysis
For kinGEMs models, calculates metrics specifically for genes that have kcat data

### Step 7: Generate Visualizations
- **growth_heatmaps.png**: Side-by-side comparison of growth patterns
- **growth_distributions.png**: Density plots of predicted growth values
- **metrics_comparison.png**: Bar chart comparing model performance (if multiple models)

### Step 8: Save Results
- **validation_summary.csv**: Table of all metrics for all models
- **config_used.json**: Copy of configuration for reproducibility
- ***.npy files**: Raw simulation data (numpy arrays)

## Output Structure

```
results/validation/{model_name}_{timestamp}/
├── validation_summary.csv              # Metrics table
├── config_used.json                    # Configuration used
├── growth_heatmaps.png                 # Visual comparison
├── growth_distributions.png            # Distribution plots
├── metrics_comparison.png              # Bar chart (if multiple models)
├── baseline_GEM.npy                    # Raw baseline data
├── pre_tuning_GEM.npy                  # Raw pre-tuning data (if enabled)
├── post_tuning_GEM.npy                 # Raw post-tuning data (if enabled)
└── experimental_fitness.npy            # Experimental data
```

## Example Use Cases

### Use Case 1: Full Comparison (Baseline, Pre-tuning, Post-tuning)

```json
{
  "model_name": "iML1515_full_comparison",
  "model_path": "models/ecoli_iML1515_20250826_4941.xml",
  "objective_reaction": "BIOMASS_Ec_iML1515_core_75p37M",
  "enzyme_upper_bound": 0.15,
  "pre_tuning_data_path": "data/processed/iML1515_GEM/iML1515_GEM_processed_data.csv",
  "post_tuning_data_path": "results/tuning_results/ecoli_iML1515_20250826_4941/df_new.csv",
  "run_baseline": true,
  "run_pre_tuning": true,
  "run_post_tuning": true
}
```

**Result**: Shows improvement from:
1. No enzyme constraints (baseline)
2. Initial kcat values (pre-tuning)
3. Optimized kcat values (post-tuning)

### Use Case 2: Before/After Tuning Only

```json
{
  "model_name": "iML1515_tuning_effect",
  "model_path": "models/ecoli_iML1515_20250826_4941.xml",
  "objective_reaction": "BIOMASS_Ec_iML1515_core_75p37M",
  "enzyme_upper_bound": 0.15,
  "post_tuning_data_path": "results/tuning_results/ecoli_iML1515_20250826_4941/df_new.csv",
  "run_baseline": true,
  "run_pre_tuning": false,
  "run_post_tuning": true
}
```

**Result**: Direct comparison of baseline vs. tuned kinGEMs model

### Use Case 3: Quick Baseline Validation

```json
{
  "model_name": "iML1515_baseline_only",
  "model_path": "models/ecoli_iML1515_20250826_4941.xml",
  "objective_reaction": "BIOMASS_Ec_iML1515_core_75p37M",
  "run_baseline": true,
  "run_pre_tuning": false,
  "run_post_tuning": false
}
```

**Result**: Fast validation of baseline model only (useful for debugging)

### Use Case 4: Test Different Enzyme Upper Bounds

Create multiple configs with different `enzyme_upper_bound` values:

```bash
# Test enzyme_upper_bound = 0.10
python scripts/run_validation_pipeline.py configs/validation_iML1515_eub_010.json

# Test enzyme_upper_bound = 0.15 (default)
python scripts/run_validation_pipeline.py configs/validation_iML1515_eub_015.json

# Test enzyme_upper_bound = 0.20
python scripts/run_validation_pipeline.py configs/validation_iML1515_eub_020.json
```

## Performance Notes

### Computational Time

For E. coli iML1515 with ~1,500 genes × ~40 carbon sources:

- **Baseline GEM**: ~30 minutes (fast, no enzyme constraints)
- **Pre-tuning kinGEMs**: ~8-12 hours (enzyme-constrained optimization)
- **Post-tuning kinGEMs**: ~8-12 hours (enzyme-constrained optimization)

**Tips for faster runs:**
- Start with `run_baseline: true` only to verify setup
- Use parallel simulations (future enhancement)
- Test on smaller carbon source subsets first

### Memory Requirements

- **Baseline**: ~2-4 GB
- **kinGEMs models**: ~4-8 GB (depends on model size and enzyme data)

## Interpreting Results

### What Good Performance Looks Like

**Accuracy**: 0.70-0.85 is typical for GEMs validated against Keio collection
**Precision**: Higher is better (fewer false positives)
**Recall**: Higher is better (fewer false negatives)
**F1 Score**: Balanced measure; 0.65-0.80 is good

### Expected Improvements

**Baseline → Pre-tuning**: May decrease slightly (enzyme constraints are more restrictive)
**Pre-tuning → Post-tuning**: Should increase (tuned kcats better match biology)

### Red Flags

- **Accuracy < 0.60**: Model may have fundamental issues
- **Post-tuning worse than pre-tuning**: Check tuning convergence
- **All models perform similarly**: Check if enzyme constraints are actually being applied

## Troubleshooting

### Issue: "Pre-tuning data path not found"

**Solution**: Make sure you've run the full pipeline through Step 3:
```bash
python scripts/run_pipeline.py configs/iML1515_GEM.json
```

The pre-tuning data is in: `data/processed/{model_name}/{model_name}_processed_data.csv`

### Issue: "Post-tuning data path not found"

**Solution**: Run the full pipeline through Step 5 (simulated annealing):
```bash
python scripts/run_pipeline.py configs/iML1515_GEM.json
```

The post-tuning data is in: `results/tuning_results/{run_id}/df_new.csv`

### Issue: Validation takes too long

**Solutions**:
1. Test with baseline only first
2. Use a smaller subset of genes (modify validation_utils.py)
3. Run on HPC cluster with more resources
4. Consider parallelizing the simulation loop (future enhancement)

### Issue: "No objective reaction found"

**Solution**: Specify the objective reaction explicitly in your config:
```json
{
  "objective_reaction": "BIOMASS_Ec_iML1515_core_75p37M"
}
```

## Comparison with Original Script

### Original (`validation_kinGEMs_iML1515.py`)
- Hardcoded file paths
- Only compares baseline vs. post-tuning
- Single model validation
- Requires manual editing for different models

### New (`run_validation_pipeline.py`)
- ✅ Config-driven (JSON files)
- ✅ Compares baseline, pre-tuning, AND post-tuning
- ✅ Flexible model selection
- ✅ Timestamped output directories
- ✅ Better error handling
- ✅ More comprehensive visualizations
- ✅ Reproducible (saves config used)

## Next Steps

After validation:

1. **Review metrics**: Check if tuning improved performance
2. **Examine heatmaps**: Visual inspection of growth predictions
3. **Analyze discrepancies**: Which genes/carbons have largest errors?
4. **Iterate tuning**: Adjust simulated annealing parameters if needed
5. **Compare enzyme upper bounds**: Test different constraint levels

## See Also

- `PIPELINE_SUMMARY.md` - Overview of the full kinGEMs pipeline
- `CONFIG_GUIDE.md` - Configuration options for model building
- `SYSTEM_SUMMARY.md` - Technical details about the kinGEMs system
- Original validation script: `scripts/validation_kinGEMs_iML1515.py`

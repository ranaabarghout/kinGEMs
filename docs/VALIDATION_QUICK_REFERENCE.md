# kinGEMs Validation - Quick Reference

## One-Command Validation

```bash
python scripts/run_validation_pipeline.py configs/validation_iML1515.json
```

## Minimal Config

```json
{
  "model_name": "my_model",
  "model_path": "models/my_model.xml",
  "post_tuning_data_path": "results/tuning_results/my_run_id/df_new.csv",
  "run_baseline": true,
  "run_post_tuning": true
}
```

## Config Options Cheat Sheet

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `model_name` | ✅ Yes | - | Model identifier |
| `model_path` | ✅ Yes | - | Path to SBML file |
| `objective_reaction` | ❌ No | auto | Biomass reaction ID |
| `enzyme_upper_bound` | ❌ No | 0.15 | Enzyme constraint |
| `pre_tuning_data_path` | ❌ No | - | Pre-tuning CSV |
| `post_tuning_data_path` | ❌ No | - | Post-tuning CSV |
| `run_baseline` | ❌ No | true | Run baseline GEM |
| `run_pre_tuning` | ❌ No | false | Run pre-tuning |
| `run_post_tuning` | ❌ No | true | Run post-tuning |
| `sim_thresh` | ❌ No | 0.001 | Growth threshold |
| `fit_thresh` | ❌ No | -2 | Fitness threshold |

## Data Paths Reference

### Pre-tuning Data (Initial kcat)
```
data/processed/{model_name}/{model_name}_processed_data.csv
```
**Source**: Step 3 of pipeline (process_kcat_predictions)
**Contains**: Initial kcat values from CPI-Pred

### Post-tuning Data (Optimized kcat)
```
results/tuning_results/{run_id}/df_new.csv
```
**Source**: Step 5 of pipeline (simulated_annealing)
**Contains**: Tuned kcat values after optimization

## Output Files

```
results/validation/{model_name}_{timestamp}/
├── validation_summary.csv              ← Metrics table
├── growth_heatmaps.png                 ← Visual comparison
├── growth_distributions.png            ← Density plots
├── metrics_comparison.png              ← Bar chart
├── config_used.json                    ← Reproducibility
├── baseline_GEM.npy                    ← Raw data
├── pre_tuning_GEM.npy                  ← Raw data
├── post_tuning_GEM.npy                 ← Raw data
└── experimental_fitness.npy            ← Raw data
```

## Common Workflows

### Workflow 1: Full Analysis
```bash
# 1. Run full pipeline
python scripts/run_pipeline.py configs/iML1515_GEM.json

# 2. Create validation config
cat > configs/validation_full.json << EOF
{
  "model_name": "iML1515_full",
  "model_path": "models/ecoli_iML1515_20250826_4941.xml",
  "pre_tuning_data_path": "data/processed/iML1515_GEM/iML1515_GEM_processed_data.csv",
  "post_tuning_data_path": "results/tuning_results/ecoli_iML1515_20250826_4941/df_new.csv",
  "run_baseline": true,
  "run_pre_tuning": true,
  "run_post_tuning": true
}
EOF

# 3. Run validation (16-24 hours)
python scripts/run_validation_pipeline.py configs/validation_full.json
```

### Workflow 2: Quick Test
```bash
# Test baseline only (30 minutes)
cat > configs/validation_quick.json << EOF
{
  "model_name": "iML1515_quick",
  "model_path": "models/ecoli_iML1515_20250826_4941.xml",
  "run_baseline": true,
  "run_pre_tuning": false,
  "run_post_tuning": false
}
EOF

python scripts/run_validation_pipeline.py configs/validation_quick.json
```

### Workflow 3: Before/After Tuning
```bash
# Compare baseline vs tuned only
cat > configs/validation_compare.json << EOF
{
  "model_name": "iML1515_compare",
  "model_path": "models/ecoli_iML1515_20250826_4941.xml",
  "post_tuning_data_path": "results/tuning_results/ecoli_iML1515_20250826_4941/df_new.csv",
  "run_baseline": true,
  "run_post_tuning": true
}
EOF

python scripts/run_validation_pipeline.py configs/validation_compare.json
```

## Metrics Quick Reference

| Metric | Formula | Range | Good Value |
|--------|---------|-------|------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | 0-1 | >0.70 |
| Precision | TP/(TP+FP) | 0-1 | >0.75 |
| Recall | TP/(TP+FN) | 0-1 | >0.70 |
| F1 Score | 2×(Prec×Rec)/(Prec+Rec) | 0-1 | >0.72 |
| AUC | Area under PR curve | 0-1 | >0.75 |
| Balanced Acc | (Sensitivity+Specificity)/2 | 0-1 | >0.70 |
| ROC AUC | Area under ROC curve | 0-1 | >0.75 |

**Legend**: TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative

## Time Estimates (E. coli iML1515)

| Configuration | Time | Use Case |
|---------------|------|----------|
| Baseline only | ~30 min | Quick test |
| Baseline + Post | ~8-12 hours | Standard validation |
| All three models | ~16-24 hours | Full comparison |

## Troubleshooting Fast Reference

| Error | Solution |
|-------|----------|
| "Pre-tuning data not found" | Run pipeline through Step 3 |
| "Post-tuning data not found" | Run pipeline through Step 5 |
| "No objective reaction" | Add `objective_reaction` to config |
| Takes too long | Start with `run_baseline: true` only |
| Out of memory | Reduce `enzyme_upper_bound` |

## Expected Performance Trend

```
Baseline GEM:      Accuracy ~0.75  ─────────●
                                            /
Pre-tuning kinGEMs: Accuracy ~0.70 ────●   /
                                          \/
Post-tuning kinGEMs: Accuracy ~0.80 ─────────●  ← Target improvement
```

**Goal**: Post-tuning should outperform both baseline and pre-tuning

## File Size Reference

| File | Typical Size |
|------|--------------|
| validation_summary.csv | <10 KB |
| growth_heatmaps.png | ~500 KB |
| growth_distributions.png | ~300 KB |
| metrics_comparison.png | ~200 KB |
| baseline_GEM.npy | ~500 KB |
| pre_tuning_GEM.npy | ~500 KB |
| post_tuning_GEM.npy | ~500 KB |
| experimental_fitness.npy | ~500 KB |

## Quick Commands

```bash
# List all validation runs
ls -lht results/validation/

# View latest validation summary
ls -t results/validation/*/validation_summary.csv | head -1 | xargs cat

# Compare metrics from multiple runs
for dir in results/validation/*/; do
    echo "=== $(basename $dir) ==="
    grep -A1 "^Post" "$dir/validation_summary.csv"
done

# Open latest heatmap
ls -t results/validation/*/growth_heatmaps.png | head -1 | xargs open
```

## Documentation Links

- **Full Guide**: [docs/VALIDATION_PIPELINE_GUIDE.md](VALIDATION_PIPELINE_GUIDE.md)
- **Implementation**: [docs/VALIDATION_SYSTEM_SUMMARY.md](VALIDATION_SYSTEM_SUMMARY.md)
- **Pipeline Config**: [docs/CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- **System Overview**: [docs/SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md)

## Support

For issues or questions:
1. Check [VALIDATION_PIPELINE_GUIDE.md](VALIDATION_PIPELINE_GUIDE.md) troubleshooting section
2. Review example configs in `configs/`
3. Compare with original script: `scripts/validation_kinGEMs_iML1515.py`

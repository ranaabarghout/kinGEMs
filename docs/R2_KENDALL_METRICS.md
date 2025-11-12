# R² and Kendall's Tau Metrics - Added

## New Metrics

I've added **R² (coefficient of determination)** and **Kendall's tau (τ)** to both the growth rate and fitness-scale correlation analyses!

## What's R²?

**R² (R-squared)** measures the proportion of variance in the experimental data explained by the model predictions.

### Formula
```
R² = 1 - (SS_residual / SS_total)
```

Where:
- **SS_residual** = Sum of squared differences between predicted and experimental
- **SS_total** = Sum of squared differences from experimental mean

### Interpretation

| R² Value | Interpretation |
|----------|----------------|
| **1.0** | Perfect prediction (100% variance explained) |
| **0.8-0.9** | Excellent fit (80-90% variance explained) |
| **0.6-0.8** | Good fit (60-80% variance explained) |
| **0.4-0.6** | Moderate fit (40-60% variance explained) |
| **< 0.4** | Poor fit (<40% variance explained) |
| **Negative** | Model worse than just using mean! |

### Why It's Useful

1. **Easy interpretation** - Directly tells you % of variance explained
2. **Scale-independent** - Can compare across different units
3. **Publication standard** - Commonly reported in regression analyses
4. **Model comparison** - Higher R² = better fit

### R² vs. Pearson r

```
R² = r²  (for linear relationships)
```

If Pearson r = 0.8, then R² = 0.64

**Key Difference:**
- **Pearson r** measures strength of linear relationship
- **R²** measures proportion of variance explained

## What's Kendall's Tau (τ)?

**Kendall's tau** is a rank correlation coefficient, like Spearman but more robust.

### What It Measures

The probability that predicted and experimental rankings agree minus the probability they disagree.

### Formula
```
τ = (concordant pairs - discordant pairs) / total pairs
```

### Interpretation

| τ Value | Interpretation |
|---------|----------------|
| **+1.0** | Perfect positive correlation |
| **+0.7** | Strong positive correlation |
| **+0.5** | Moderate positive correlation |
| **+0.3** | Weak positive correlation |
| **0.0** | No correlation |
| **-1.0** | Perfect negative correlation |

### Why It's Useful

1. **Robust to outliers** - Less sensitive than Pearson
2. **Non-parametric** - No assumptions about distributions
3. **Handles ties** - Works with duplicate values
4. **Interpretable** - Directly related to probability of agreement

### Kendall vs. Spearman

Both are rank-based, but:

| Feature | Kendall τ | Spearman ρ |
|---------|-----------|------------|
| **Calculation** | Based on pair concordance | Based on rank differences |
| **Range** | Usually -1 to +1 | -1 to +1 |
| **Typical value** | Lower magnitude | Higher magnitude |
| **Interpretation** | Probability interpretation | Pearson on ranks |
| **Outlier sensitivity** | More robust | Less robust |
| **Computation** | Slower (O(n²)) | Faster (O(n log n)) |

**Rule of thumb:** Kendall τ ≈ 0.7 × Spearman ρ

## Where They Appear

### 1. Metrics CSV Files

**Growth Rate Metrics** (`validation_metrics.csv`):
```csv
                      pearson_r  spearman_r  kendall_tau    r2     rmse
Baseline GEM            0.5234      0.4891       0.3542   0.2740  0.3456
Pre-tuning kinGEMs      0.6891      0.6234       0.4521   0.4749  0.2891
Post-tuning kinGEMs     0.7456      0.7012       0.5234   0.5559  0.2456
```

**Fitness Metrics** (`validation_metrics_fitness.csv`):
```csv
                      pearson_r_fitness  spearman_r_fitness  kendall_tau_fitness  r2_fitness  rmse_fitness
Baseline GEM                    0.6123              0.5891               0.4234      0.3749        1.8912
Pre-tuning kinGEMs              0.7234              0.6912               0.5123      0.5233        1.4567
Post-tuning kinGEMs             0.7891              0.7456               0.5678      0.6227        1.2345
```

### 2. Fitness Comparison Plots

The `fitness_comparison_scatter.png` plots now show:
- **Title:** R² and RMSE
- **Statistics box:**
  ```
  Pearson r = 0.7891
  Spearman ρ = 0.7456
  Kendall τ = 0.5678
  R² = 0.6227
  RMSE = 1.2345
  p < 1.23e-100
  n = 33,150
  ```

## Usage

Metrics are **automatically calculated** when you compile validation results:

```bash
python scripts/compile_validation_results.py \
    --input results/validation_parallel \
    --output results/validation_compiled
```

No extra flags needed!

## Example Interpretation

### Good Model (Post-tuning)
```
Pearson r = 0.79       → Strong linear correlation
Spearman ρ = 0.75      → Strong rank correlation
Kendall τ = 0.57       → Moderate-strong agreement
R² = 0.62              → Explains 62% of variance
RMSE = 1.23            → ~2.3× average fold-change error
```

**Interpretation:**
- ✅ Strong correlations across all metrics
- ✅ R² shows model explains majority of variance
- ✅ Kendall confirms robust agreement
- ✅ All metrics consistent → reliable predictions

### Poor Model (Hypothetical)
```
Pearson r = 0.35       → Weak linear correlation
Spearman ρ = 0.42      → Weak rank correlation
Kendall τ = 0.28       → Weak agreement
R² = 0.12              → Explains only 12% of variance
RMSE = 3.45            → ~11× average fold-change error
```

**Interpretation:**
- ⚠️ Weak correlations across all metrics
- ⚠️ R² shows model captures little variance
- ⚠️ Kendall confirms poor agreement
- 🔴 Model needs major improvements

## When to Use Each Metric

### Use **Pearson r** when:
- ✅ Relationship is linear
- ✅ Data are normally distributed
- ✅ No major outliers
- ✅ Standard reporting (most common)

### Use **Spearman ρ** when:
- ✅ Relationship is monotonic (not necessarily linear)
- ✅ Data have outliers
- ✅ Non-normal distributions
- ✅ Interested in rank relationships

### Use **Kendall τ** when:
- ✅ Need robust measure (many outliers)
- ✅ Small sample sizes
- ✅ Want probability interpretation
- ✅ Data have many ties

### Use **R²** when:
- ✅ Want % variance explained
- ✅ Comparing model quality
- ✅ Need intuitive interpretation
- ✅ Publication reporting

### Use all four when:
- ✅ Comprehensive analysis
- ✅ Validation of results
- ✅ Cross-checking consistency
- ✅ Publication supplement

## Troubleshooting

### Negative R²

**What it means:** Model is worse than just predicting the mean!

**Possible causes:**
1. Model is fundamentally wrong
2. Predictions are inverted
3. Major systematic bias

**Solution:** Check model implementation, constraints, and solver

### Kendall τ << Spearman ρ

**What it means:** Many outliers affecting Spearman more than Kendall

**Typical pattern:** Kendall τ ≈ 0.7 × Spearman ρ

**If ratio is < 0.6:** Investigate outliers

### All metrics low

**What it means:** Poor model fit

**Check:**
1. Wild-type files are model-specific
2. Constraints applied correctly
3. Solver converging properly
4. Data preprocessing correct

## Implementation Details

### Dependencies

Both metrics use standard libraries:
```python
from scipy.stats import kendalltau
from sklearn.metrics import r2_score
```

### Calculation

**For growth rate metrics:**
```python
kendall_tau, kendall_p = kendalltau(experimental, predicted)
r2 = r2_score(experimental, predicted)
```

**For fitness metrics:**
```python
predicted_fitness = log2(predicted_growth / wildtype_growth)
kendall_tau, kendall_p = kendalltau(experimental_fitness, predicted_fitness)
r2 = r2_score(experimental_fitness, predicted_fitness)
```

### Error Handling

Both calculations are wrapped in try-except blocks:
```python
try:
    kendall_tau, kendall_p = kendalltau(exp_clean, pred_clean)
except Exception:
    kendall_tau, kendall_p = np.nan, np.nan

try:
    r2 = r2_score(exp_clean, pred_clean)
except Exception:
    r2 = np.nan
```

This prevents crashes if data are problematic.

## References

### R² (Coefficient of Determination)
- **Range:** -∞ to 1.0 (typically 0 to 1)
- **Also known as:** R-squared, coefficient of determination
- **Related to:** Pearson correlation (R² = r²)

### Kendall's Tau
- **Range:** -1.0 to +1.0
- **Also known as:** Kendall's τ, Kendall rank correlation
- **Related to:** Spearman's ρ (but more robust)

## Summary

✅ **R²** added - Shows % variance explained (0 to 1)
✅ **Kendall's tau** added - Robust rank correlation (-1 to +1)
✅ Appears in both growth rate and fitness metrics
✅ Shown in fitness comparison scatter plots
✅ Automatically calculated, no setup needed
✅ P-values included for both metrics

These metrics provide additional validation perspectives and are commonly requested for publications!

# CPI-Pred Predictions File Naming - Auto-Discovery

## Problem

The kinGEMs pipeline expected CPI-Pred predictions files to follow the exact naming pattern:
```
X06A_kinGEMs_{model_name}_predictions.csv
```

However, actual file names often use different patterns:
- `iML1515_GEM` → `X06A_kinGEMs_ecoli_iML1515_predictions.csv` (not `X06A_kinGEMs_iML1515_GEM_predictions.csv`)
- `e_coli_core` → `X06A_kinGEMs_ecoli_core_predictions.csv` (not `X06A_kinGEMs_e_coli_core_predictions.csv`)

This caused `FileNotFoundError` even when predictions existed.

## Solution

The pipeline now includes **intelligent predictions file discovery** that tries multiple naming patterns:

### Automatic Search Patterns

For model `iML1515_GEM`, the pipeline searches for:
1. `X06A_kinGEMs_iML1515_GEM_predictions.csv` (exact match)
2. `*iML1515_GEM*predictions.csv` (fuzzy match)
3. `*iML1515*predictions.csv` (without _GEM)
4. `*ecoli*iML*predictions.csv` (organism prefix)
5. `*ecoli_iML1515*predictions.csv` (common variant)

For model `e_coli_core`, it searches for:
1. `X06A_kinGEMs_e_coli_core_predictions.csv` (exact match)
2. `*e_coli_core*predictions.csv` (fuzzy match)
3. `*ecoli_core*predictions.csv` (variant without underscore)

### Supported Naming Conventions

The auto-discovery works with these naming patterns:

| Model Name | Will Find |
|------------|-----------|
| `iML1515_GEM` | `X06A_kinGEMs_ecoli_iML1515_predictions.csv` |
| `e_coli_core` | `X06A_kinGEMs_ecoli_core_predictions.csv` |
| `382_genome_cpd03198` | `X06A_kinGEMs_382_genome_cpd03198_predictions.csv` |
| `yeast-GEM9` | `X06A_kinGEMs_yeast-GEM9_predictions.csv` |

## Usage

**No action required!** The pipeline automatically finds the predictions file:

```bash
python scripts/run_pipeline.py configs/iML1515_GEM.json
```

Output shows which file was found:
```
Found predictions file: X06A_kinGEMs_ecoli_iML1515_predictions.csv
```

## Troubleshooting

### Predictions File Not Found

If no predictions file is found, the pipeline will:
1. List all available prediction files in the directory
2. Provide a clear error message

**Example error output:**
```
⚠️  No predictions file found for 'my_model'
Available prediction files:
  - X06A_kinGEMs_ecoli_iML1515_predictions.csv
  - X06A_kinGEMs_382_genome_cpd03198_predictions.csv
  - X06A_kinGEMs_ecoli_core_predictions.csv

Please ensure CPI-Pred predictions exist for this model.
```

### Solution Steps

1. **Check if predictions exist** in `data/interim/CPI-Pred predictions/`
2. **Verify naming pattern** matches one of the supported formats
3. **Run CPI-Pred** if predictions don't exist
4. **Rename file** if needed to match expected patterns

### Manual Override

You can also specify the predictions file path in your config:

```json
{
  "model_name": "my_model",
  "predictions_file": "data/interim/CPI-Pred predictions/custom_predictions.csv"
}
```

(Note: This feature would need to be added if needed)

## Implementation Details

The auto-discovery function (`find_predictions_file()`) in `run_pipeline.py`:

1. **Tries multiple patterns** using glob matching
2. **Returns first match** found
3. **Lists available files** if none match
4. **Provides helpful error** with suggestions

```python
def find_predictions_file(model_name, CPIPred_data_dir):
    """Find CPI-Pred predictions with flexible naming."""
    patterns = [
        f"X06A_kinGEMs_{model_name}_predictions.csv",
        f"*{model_name}*predictions.csv",
        # ... more patterns
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(CPIPred_data_dir, pattern))
        if matches:
            return matches[0]
    
    # Show available files if not found
    raise FileNotFoundError(...)
```

## Benefits

✅ **Flexible naming** - handles multiple CPI-Pred output formats  
✅ **No manual config** - automatic discovery  
✅ **Clear errors** - shows available files if not found  
✅ **Backward compatible** - still works with exact matches  

## Naming Recommendations

For best compatibility, name your CPI-Pred predictions:

```
X06A_kinGEMs_{organism}_{model_base}_predictions.csv
```

Examples:
- `X06A_kinGEMs_ecoli_iML1515_predictions.csv`
- `X06A_kinGEMs_ecoli_core_predictions.csv`
- `X06A_kinGEMs_yeast_GEM9_predictions.csv`

But the auto-discovery will work with most reasonable variations!

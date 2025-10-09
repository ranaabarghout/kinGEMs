# Enzyme-Constrained Optimization Fixes

## Date: October 9, 2025

## Problem Summary

The ModelSEED genome-scale models (e.g., `382_genome_cpd03198`) were not applying enzyme constraints during optimization, resulting in:
- 0 enzyme constraints added
- 0 enzyme usage (should be ~0.15 g/gDW)
- Incorrect biomass predictions
- Very slow processing times (minutes for 398K rows)

## Root Causes Identified

### 1. **Slow Data Processing (Performance Issue)**
**Problem:** Using pandas `.iterrows()` on 398,949 rows was extremely slow.
```python
# OLD (SLOW - minutes for 398K rows)
for _, row in processed_df.iterrows():
    if pd.notna(row['kcat_mean']) and pd.notna(row['SEQ']):
        kcat_dict[(reaction_id, gene_id)] = [row['kcat_mean']]
```

**Solution:** Use vectorized pandas operations
```python
# NEW (FAST - instant even for 398K rows)
valid_rows = processed_df[processed_df[kcat_col].notna() & processed_df['SEQ'].notna()].copy()
grouped = valid_rows.groupby(['Reactions', 'Single_gene'])[kcat_col].mean()
kcat_dict = {key: [value] for key, value in grouped.items()}
```

**Impact:** 
- ModelSEED (398K rows): Minutes → **Seconds**
- iML1515 (33K rows): Already fast → **Even faster**

### 2. **Inefficient Constraint Domain (Performance Issue)**
**Problem:** Creating Cartesian product of all reactions × all genes
```python
# OLD (SLOW - creates 3.8 million pairs for ModelSEED)
m.K = Set(initialize=[(r, g) for r in rxns for g in genes], dimen=2)
# For 382_genome: 3,011 reactions × 1,270 genes = 3,823,970 pairs!
```

**Solution:** Only include (reaction, gene) pairs that have kcat data
```python
# NEW (FAST - only 4,567 pairs for ModelSEED)
m.K = Set(initialize=list(kcat_dict.keys()), dimen=2)
```

**Impact:**
- ModelSEED: 3,823,970 → **4,567 constraint checks** (836x fewer!)
- iML1515: Similar reduction in unnecessary checks

### 3. **Case Sensitivity Bug (Critical Bug)**
**Problem:** GPR tokenizer was lowercasing all gene IDs, breaking lookups
```python
# OLD (BROKEN)
def _tokenize_gpr(rule):
    token_spec = r'\(|\)|and|or|[^()\s]+'
    return re.findall(token_spec, rule.lower())  # <-- Lowercases everything!

# Gene IDs in kcat_dict: 'NODE_1_3110591_3110064'
# Gene IDs in dnf_clauses: 'node_1_3110591_3110064'  (lowercased)
# Result: No matches → 0 constraints added
```

**Solution:** Preserve gene ID case, only lowercase 'and'/'or' keywords
```python
# NEW (CORRECT)
def _tokenize_gpr(rule):
    token_spec = r'\(|\)|(?i:and)|(?i:or)|[^()\s]+'
    tokens = re.findall(token_spec, rule)
    # Normalize 'and'/'or' to lowercase for parser, keep gene IDs as-is
    return [t.lower() if t.lower() in ('and', 'or') else t for t in tokens]
```

**Impact:** 
- ModelSEED: 0 constraints → **2,474 constraints** (1,628 AND + 846 OR)
- iML1515: Already working → **Still works correctly**

## Results Comparison

### ModelSEED Model (382_genome_cpd03198)

| Metric | Before Fix | After Fix | Change |
|--------|------------|-----------|---------|
| **Processing Time** | Minutes | Seconds | ✅ ~100x faster |
| **Constraint Checks** | 3,823,970 | 4,567 | ✅ 836x fewer |
| **AND Constraints Added** | 0 | 1,628 | ✅ Fixed |
| **OR/ISO Constraints Added** | 0 | 846 | ✅ Fixed |
| **Genes with Enzyme Allocation** | 0/1270 | 138/1270 | ✅ Fixed |
| **Total Enzyme Usage** | 0 g/gDW | 0.15 g/gDW | ✅ Fixed |
| **Initial Biomass** | 0.6176 | 0.3196 | ✅ Correctly constrained |

### Standard Model (iML1515_GEM)

| Metric | Before Fix | After Fix | Change |
|--------|------------|-----------|---------|
| **Processing Time** | Fast | Faster | ✅ Improved |
| **Constraint Checks** | Many | 4,348 | ✅ Optimized |
| **AND Constraints Added** | 1,949 | 1,949 | ✅ Still works |
| **OR/ISO Constraints Added** | 704 | 704 | ✅ Still works |
| **Genes with Enzyme Allocation** | 72/1516 | 72/1516 | ✅ Still works |
| **Total Enzyme Usage** | 0.15 g/gDW | 0.15 g/gDW | ✅ Still works |
| **Initial Biomass** | 0.0348 | 0.0348 | ✅ Still works |

## Files Modified

### `/project/def-mahadeva/ranaab/kinGEMs_v2/kinGEMs/modeling/optimize.py`

**Changes:**
1. Line 40-46: Fixed `_tokenize_gpr()` to preserve gene ID case
2. Line 468-489: Optimized kcat_dict construction with vectorized operations
3. Line 167-169: Optimized m.K constraint domain to use only kcat_dict keys
4. Added debug output for diagnostics

## Verification

Both model types now work correctly:

✅ **ModelSEED models** (382_genome_cpd03198):
- Fast processing (seconds instead of minutes)
- Enzyme constraints properly applied
- Realistic biomass predictions

✅ **Standard models** (iML1515_GEM):
- Still works correctly
- Performance improved
- Same constraint counts and biomass values

## Technical Details

### Why ModelSEED Has More Data Rows

ModelSEED models generate multiple substrate-enzyme combinations per reaction:
- **ModelSEED (382_genome):** 398,949 rows for 3,011 reactions = ~132 rows/reaction
- **Standard (iML1515):** 33,657 rows for 2,712 reactions = ~12 rows/reaction

This is because ModelSEED predictions consider multiple substrates and conditions for each reaction-gene pair.

### Why Vectorized Operations Are Faster

```python
# iterrows() is slow because it:
# 1. Creates a pandas Series for each row (overhead)
# 2. Converts data types repeatedly
# 3. Can't use optimized C code

# Vectorized operations are fast because they:
# 1. Work on entire arrays at once
# 2. Use optimized NumPy/pandas C code
# 3. Avoid Python loops
```

## Impact on Workflow

Users can now:
1. Run ModelSEED models with realistic computation times
2. Get accurate enzyme constraint predictions
3. Trust that both ModelSEED and standard models work correctly
4. Process large datasets (398K+ rows) efficiently

## Recommendations

1. **For large DataFrames (>10K rows):** Always use vectorized pandas operations instead of `.iterrows()`
2. **For constraint generation:** Only create constraint domains for relevant (reaction, gene) pairs
3. **For GPR parsing:** Preserve original case for gene IDs; they are case-sensitive in SBML models
4. **For debugging:** Use the diagnostic output to verify constraints are being added:
   ```
   [DEBUG] AND constraints: X added, Y skipped
   [DEBUG] OR/ISO constraints: X added, Y skipped
   [DIAG] Number of genes with enzyme allocation: X/Y
   ```

## Future Work

Consider removing debug print statements once the fixes are validated in production, or convert them to proper logging statements using the `loguru` logger already in the project.

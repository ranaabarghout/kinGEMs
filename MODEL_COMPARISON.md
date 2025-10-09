# Model Comparison: iML1515 vs 382_genome_cpd03198

## After Optimization Fixes (October 9, 2025)

### Quick Comparison

| Feature | iML1515 (Standard) | 382_genome (ModelSEED) |
|---------|-------------------|------------------------|
| **Model Size** | | |
| Genes | 1,516 | 1,270 |
| Reactions | 2,712 | 3,011 |
| Reactions with GPR | 2,266 | 2,614 |
| **Processed Data** | | |
| Total rows | 33,657 | 398,949 |
| Valid rows (with kcat) | 31,364 | 393,682 |
| Unique (reaction, gene) pairs | 4,348 | 4,567 |
| Rows per reaction (avg) | ~12 | ~132 |
| **Constraints Added** | | |
| AND constraints | 1,949 | 1,628 |
| OR/ISO constraints | 704 | 846 |
| Total enzyme constraints | 2,653 | 2,474 |
| **Enzyme Allocation** | | |
| Genes with allocation | 72 / 1,516 (4.7%) | 138 / 1,270 (10.9%) |
| Total enzyme usage | 0.15 g/gDW | 0.15 g/gDW |
| Upper bound | 0.15 g/gDW | 0.15 g/gDW |
| At capacity? | Yes | Yes |
| **Biomass** | | |
| Initial biomass (w/ constraints) | 0.0348 | 0.3196 |
| Organism | E. coli | E. coli |
| **Processing Performance** | | |
| Data processing speed | Fast (< 1 sec) | Fast (< 1 sec) |
| Constraint generation | Fast | Fast |
| Model type | BiGG/Standard | ModelSEED |

## Key Differences

### 1. Data Structure
- **iML1515:** Clean BiGG database format, ~12 substrate combinations per reaction
- **382_genome:** ModelSEED format with extensive substrate predictions, ~132 combinations per reaction

### 2. Gene Naming
- **iML1515:** Standard gene names (e.g., `b0241`, `b0356`)
- **382_genome:** NODE-based naming (e.g., `NODE_1_3110591_3110064`)

### 3. Constraint Distribution
- **iML1515:** More AND constraints (single enzymes/complexes)
- **382_genome:** More OR/ISO constraints (alternative enzyme forms)

### 4. Enzyme Allocation
- **iML1515:** 4.7% of genes used (more selective)
- **382_genome:** 10.9% of genes used (broader enzyme utilization)

### 5. Growth Rate
- **iML1515:** Lower growth rate (0.0348) - more constrained metabolism
- **382_genome:** Higher growth rate (0.3196) - less constrained despite enzyme limits

## Why Both Reach Maximum Enzyme Capacity

Both models show `Total enzyme usage: 0.15 g/gDW (upper bound: 0.15)`, meaning they're at maximum capacity. This indicates:

1. **The enzyme constraint is binding** - both models would produce more biomass if allowed more enzyme
2. **Different efficiency** - 382_genome achieves higher biomass (0.3196 vs 0.0348) with same enzyme budget
3. **Model calibration differences** - ModelSEED and BiGG models have different reaction stoichiometries

## Validation

Both models are working correctly:
- ✅ Enzyme constraints properly applied
- ✅ Realistic enzyme usage (at upper bound)
- ✅ No errors in constraint generation
- ✅ Fast processing times
- ✅ Proper gene-reaction associations

## Conclusion

The optimization fixes successfully resolved the ModelSEED constraint application issues while maintaining compatibility with standard BiGG models. Both model types now:
- Process data efficiently (vectorized operations)
- Apply enzyme constraints correctly (case-sensitive gene matching)
- Generate appropriate constraint counts (optimized domain)

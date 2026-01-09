# kinGEMs Constraints and Scenarios Documentation

## Overview

This document provides a comprehensive overview of all constraint types and scenarios implemented in the kinGEMs optimization framework. The kinGEMs system implements enzyme-constrained flux balance analysis (FBA) with support for various enzyme types and reaction patterns.

## Table of Contents

1. [Variable Architecture](#variable-architecture)
2. [Core Constraint Types](#core-constraint-types)
3. [DNF (Disjunctive Normal Form) Processing](#dnf-processing)
4. [Constraint Implementation Details](#constraint-implementation-details)
5. [Bidirectional Constraints](#bidirectional-constraints)
6. [Control Flags](#control-flags)
7. [Multiple Substrate Handling](#multiple-substrate-handling)
8. [Mathematical Formulation](#mathematical-formulation)

## Variable Architecture

### Standard Mode
In standard mode, the optimization uses a single flux variable for each reaction:
- `v[rxn]`: Flux through reaction `rxn` (can be positive or negative)
- `E[gene]`: Enzyme allocation for gene `gene` (always non-negative)

### Bidirectional Mode
In bidirectional mode, reversible reactions are split into separate forward and reverse components:
- `v_fwd[rxn]`: Forward flux (≥ 0) for reversible reaction `rxn`
- `v_rev[rxn]`: Reverse flux (≥ 0) for reversible reaction `rxn`
- `v_irr[rxn]`: Standard flux for irreversible reaction `rxn`
- `E[gene]`: Enzyme allocation for gene `gene` (always non-negative)

Net flux for reversible reactions: `v_net[rxn] = v_fwd[rxn] - v_rev[rxn]`

## Core Constraint Types

### 1. Mass Balance Constraints
Ensures steady-state mass conservation for all metabolites.

**Standard Mode:**
```
∑ S[i,j] × v[j] = 0  ∀ metabolites i
```

**Bidirectional Mode:**
```
∑ S[i,j] × (v_fwd[j] - v_rev[j]) + ∑ S[i,k] × v_irr[k] = 0  ∀ metabolites i
```

### 2. Enzyme Pool Constraints
Limits the total enzyme allocation based on cellular enzyme budget.

**Enzyme Ratio Mode:**
```
∑ E[g] × MW[g] × 0.001 ≤ enzyme_upper_bound
```

**Simple Enzyme Mode:**
```
∑ E[g] ≤ enzyme_upper_bound
```

### 3. Enzyme Kinetic Constraints
Links reaction fluxes to enzyme allocations through kcat values.

## DNF (Disjunctive Normal Form) Processing

The system converts Gene-Protein-Reaction (GPR) rules into Disjunctive Normal Form to handle complex enzyme relationships.

### DNF Conversion Examples

| Original GPR | DNF Representation | Interpretation |
|--------------|-------------------|----------------|
| `g1` | `[['g1']]` | Single enzyme |
| `g1 and g2` | `[['g1', 'g2']]` | Enzyme complex |
| `g1 or g2` | `[['g1'], ['g2']]` | Isoenzymes |
| `g1 and (g2 or g3)` | `[['g1', 'g2'], ['g1', 'g3']]` | Mixed scenario |
| `(g1 and g2) or (g3 and g4)` | `[['g1', 'g2'], ['g3', 'g4']]` | Alternative complexes |

### DNF Parsing Algorithm

```python
def _parse_gpr_to_dnf(tokens):
    """
    Converts tokenized GPR into DNF clauses.
    Each clause represents one possible enzyme combination.
    """
    # Recursive descent parser handling precedence:
    # 1. Parentheses (highest)
    # 2. AND operations
    # 3. OR operations (lowest)
```

## Constraint Implementation Details

### 1. AND Constraints (Single Enzymes and Complexes)

Handles reactions with simple GPR rules containing only AND operations.

#### Single Enzyme Case
**Condition:** One clause with one gene
**Constraint:** `v[rxn] ≤ max(kcat_list) × E[gene]`
**Strategy:** Use maximum kcat (optimistic)

#### Enzyme Complex Case
**Condition:** One clause with multiple genes
**Constraint:** `v[rxn] ≤ avg(kcat_list) × E[gene]` (for each gene in complex)
**Strategy:** Use average kcat, constraint applied to each subunit

#### Bidirectional AND Constraints
For reversible reactions with bidirectional data:
- Forward: `v_fwd[rxn] ≤ kcat_forward × E[gene]`
- Reverse: `v_rev[rxn] ≤ kcat_reverse × E[gene]`

### 2. ISO Constraints (Pure Isoenzymes)

Handles reactions where multiple single genes can catalyze the same reaction.

#### Standard ISO Case
**Condition:** Multiple clauses, all with single genes
**Constraint:** `v[rxn] ≤ ∑(min(kcat_g) × E[g])` for all isoenzymes g
**Strategy:** Sum capacities, use minimum kcat per gene (conservative)

#### Bidirectional ISO Case
For reversible reactions:
- Forward: `v_fwd[rxn] ≤ ∑(min(kcat_g_fwd) × E[g])`
- Reverse: `v_rev[rxn] ≤ ∑(min(kcat_g_rev) × E[g])`

### 3. MIXED Constraints (Complex OR+AND scenarios)

Handles reactions with both AND and OR operations in their GPR.

**Condition:** Multiple clauses, at least one with multiple genes
**Constraint:** `v[rxn] ≤ ∑(capacity_clause)` for all clauses

Where `capacity_clause` is:
- For single gene clause: `max(kcat_list) × E[gene]`
- For complex clause: `avg(kcat_list) × ∑(E[g])/n_genes` (approximation)

### 4. Promiscuous Enzyme Constraints

Ensures that each enzyme's total usage across all reactions doesn't exceed its allocation.

**Constraint:** `∑(v[rxn]/kcat[rxn,gene]) ≤ E[gene]` for all reactions catalyzed by gene

#### Bidirectional Promiscuous Constraints
For bidirectional mode:
```
∑((v_fwd[rxn] - v_rev[rxn])/kcat[rxn,gene]) + ∑(v_irr[rxn]/kcat[rxn,gene]) ≤ E[gene]
```

## Bidirectional Constraints

### Motivation
Traditional enzyme constraints assume the same kcat for both directions of reversible reactions. However, enzyme kinetics are substrate-specific, and different substrates (forward vs. reverse direction) may have different kcat values.

### Implementation Strategy

#### 1. Variable Separation
- Reversible reactions: Separate `v_fwd` and `v_rev` variables
- Irreversible reactions: Standard `v_irr` variable
- Net flux: `v_net = v_fwd - v_rev`

#### 2. Bounds Splitting
For reversible reaction with bounds `[lb, ub]`:
- Forward flux: `v_fwd ∈ [0, max(0, ub)]`
- Reverse flux: `v_rev ∈ [0, max(0, -lb)]`

#### 3. Direction-Specific kcats
- Forward direction: Uses kcat for forward substrate
- Reverse direction: Uses kcat for reverse substrate
- Key format: `(reaction, gene, direction)` where direction ∈ {'forward', 'reverse'}

### Data Structure
```python
# Standard mode
kcat_dict = {('RXN1', 'gene1'): [kcat_values]}

# Bidirectional mode
kcat_dict = {
    ('RXN1', 'gene1', 'forward'): [kcat_fwd],
    ('RXN1', 'gene1', 'reverse'): [kcat_rev]
}
```

### Constraint Examples

#### Single Enzyme Bidirectional
```
v_fwd[RXN1] ≤ kcat_forward × E[gene1]
v_rev[RXN1] ≤ kcat_reverse × E[gene1]
```

#### Isoenzyme Bidirectional
```
v_fwd[RXN1] ≤ ∑(kcat_g_forward × E[g]) for all isoenzymes g
v_rev[RXN1] ≤ ∑(kcat_g_reverse × E[g]) for all isoenzymes g
```

## Control Flags

The optimization behavior can be controlled through several boolean flags:

| Flag | Default | Effect When True | Effect When False |
|------|---------|------------------|-------------------|
| `multi_enzyme_off` | False | Disable AND constraints | Enable single enzyme and complex constraints |
| `isoenzymes_off` | False | Disable ISO constraints | Enable isoenzyme constraints |
| `promiscuous_off` | False | Disable promiscuous constraints | Enable promiscuous enzyme constraints |
| `complexes_off` | False | Disable complex handling in AND constraints | Enable enzyme complex constraints |
| `bidirectional_constraints` | True | Use substrate-specific bidirectional constraints | Use standard symmetric constraints |

## Multiple Substrate Handling

When multiple substrates exist for the same reaction-gene-direction combination, the system uses different aggregation strategies:

### Current Implementation
- **Data Aggregation**: `grouped.max()` - Uses maximum kcat value
- **Rationale**: Optimistic bound assuming best-case substrate kinetics

### Alternative Strategies
1. **Conservative**: `grouped.min()` - Use minimum kcat (worst-case)
2. **Average**: `grouped.mean()` - Use mean kcat (balanced)
3. **Weighted**: Weight by substrate concentrations (if available)

### Information Loss
The current aggregation approach loses substrate-specific information:
- No tracking of which specific substrate is being processed
- No substrate-specific flux variables
- Assumes "average" substrate with aggregated kinetics

## Mathematical Formulation

### Objective Function
**Standard Mode:**
```
maximize/minimize ∑(c[j] × v[j])
```

**Bidirectional Mode:**
```
maximize/minimize ∑(c[j] × (v_fwd[j] - v_rev[j])) + ∑(c[k] × v_irr[k])
```

### Complete Constraint Set

#### 1. Mass Balance
```
∑ S[i,j] × flux[j] = 0  ∀ i ∈ metabolites
```

#### 2. Flux Bounds
**Standard:** `lb[j] ≤ v[j] ≤ ub[j]`
**Bidirectional:**
- `0 ≤ v_fwd[j] ≤ max(0, ub[j])`
- `0 ≤ v_rev[j] ≤ max(0, -lb[j])`

#### 3. Enzyme Constraints
Based on GPR structure and constraint type (AND/ISO/MIXED/PROMISCUOUS)

#### 4. Enzyme Pool
```
∑ E[g] × weight[g] ≤ enzyme_budget
```

### Constraint Priority and Interaction

1. **Mass Balance**: Always active (fundamental)
2. **Enzyme Pool**: Always active (resource limitation)
3. **Enzyme Kinetics**: Controlled by flags
   - AND constraints: Basic enzyme-reaction relationships
   - ISO constraints: Handle alternative enzymes
   - MIXED constraints: Handle complex GPR patterns
   - PROMISCUOUS constraints: Prevent enzyme overuse

## Computational Complexity

### Constraint Generation Scaling
- **Variables**: O(reactions + genes) in standard mode, O(reactions + genes) in bidirectional mode
- **Mass Balance**: O(metabolites × average_reactions_per_metabolite)
- **AND Constraints**: O(reaction-gene pairs with kcat data)
- **ISO Constraints**: O(reactions with isoenzymes)
- **MIXED Constraints**: O(reactions with complex GPR)
- **PROMISCUOUS Constraints**: O(genes × reactions_per_gene)

### Optimization Complexity
The resulting model is a Linear Program (LP) when all constraints are linear, ensuring polynomial-time solvability with interior point methods.

## Implementation Notes

### Performance Optimizations
1. **Sparse Matrix Operations**: Only iterate over non-zero stoichiometric coefficients
2. **Selective Constraint Generation**: Skip constraint generation for reactions without kcat data
3. **Vectorized Data Processing**: Use pandas groupby operations instead of row iteration
4. **Efficient Set Operations**: Pre-compute reaction-metabolite relationships

### Numerical Considerations
1. **Unit Consistency**: Convert all kcats from s⁻¹ to hr⁻¹ (multiply by 3600)
2. **Tolerance Handling**: Apply numerical tolerance (1e-10) to avoid floating-point precision issues
3. **Bounds Clamping**: Ensure all variables stay within their feasible bounds

### Error Handling
1. **Missing Data**: Skip constraints gracefully when kcat or sequence data is unavailable
2. **Invalid GPR**: Handle malformed gene-protein-reaction rules
3. **Solver Failures**: Fallback to alternative solvers (GLPK if Gurobi unavailable)

## Example Use Cases

### Case 1: Standard E. coli Model
```python
run_optimization(
    model="ecoli_model.xml",
    kcat_dict=standard_kcat_data,
    bidirectional_constraints=False
)
```

### Case 2: Substrate-Specific Kinetics
```python
run_optimization(
    model="ecoli_model.xml",
    kcat_dict=bidirectional_kcat_data,  # Contains direction information
    bidirectional_constraints=True
)
```

### Case 3: Simplified Analysis (No Complex Enzymes)
```python
run_optimization(
    model="ecoli_model.xml",
    kcat_dict=kcat_data,
    complexes_off=True,
    isoenzymes_off=True
)
```

## Future Extensions

### Potential Enhancements
1. **Substrate-Specific Variables**: Create separate flux variables for each substrate
2. **Dynamic kcat Selection**: Choose kcats based on metabolite concentrations
3. **Regulatory Constraints**: Include allosteric regulation and enzyme regulation
4. **Spatial Constraints**: Add compartment-specific enzyme limitations
5. **Temporal Constraints**: Dynamic enzyme allocation over time

### Research Directions
1. **Experimental Validation**: Compare predictions with proteomics data
2. **Parameter Sensitivity**: Analyze robustness to kcat uncertainty
3. **Multi-Objective**: Balance growth rate, protein cost, and metabolic efficiency
4. **Machine Learning Integration**: Predict missing kcat values using ML models

---

*This documentation covers kinGEMs v2 as of November 2025. For the latest updates, see the project repository.*

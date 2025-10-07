# Solver Configuration Guide for kinGEMs

## Problem: Gurobi License Error

On Compute Canada (and other HPC systems), you may encounter this error:

```
gurobipy._exception.GurobiError: Request denied: user 'username' not in authorized user list
```

This means your account is not authorized to use Gurobi's commercial license.

## Solution: Use GLPK (Free Open-Source Solver)

### Option 1: Use GLPK (Recommended - Free)

GLPK (GNU Linear Programming Kit) is a free, open-source solver that works well for most metabolic models.

**To use GLPK, add this line to your config file:**

```json
{
  "model_name": "iML1515_GEM",
  "organism": "E coli",
  "solver": "glpk",
  ...
}
```

All the config files in `configs/` have been updated to use GLPK by default.

### Option 2: Use Gurobi (If You Have Access)

If you have authorized access to Gurobi, you can use it for potentially faster optimization:

```json
{
  "solver": "gurobi"
}
```

**Note:** You'll need to load the Gurobi module first:
```bash
module load gurobi
```

### Option 3: Use Other Solvers

The pipeline supports any Pyomo-compatible solver:

- **glpk** - Free, good for most models
- **gurobi** - Commercial, fast (requires license)
- **cplex** - Commercial, fast (requires license)
- **ipopt** - Free, for nonlinear problems

## Checking Available Solvers

To see which solvers are available in your environment:

```python
from pyomo.opt import SolverFactory

# Test GLPK
glpk_available = SolverFactory('glpk').available()
print(f"GLPK available: {glpk_available}")

# Test Gurobi
gurobi_available = SolverFactory('gurobi').available()
print(f"Gurobi available: {gurobi_available}")
```

## Installing GLPK

If GLPK is not available, install it:

**On Compute Canada:**
```bash
module spider glpk  # Check if available as module
module load glpk    # Load if available
```

**Or install via conda/pip:**
```bash
# Using conda
conda install -c conda-forge glpk

# Using pip (requires system GLPK)
pip install glpk
```

## Performance Comparison

| Solver | Speed | License | Availability |
|--------|-------|---------|-------------|
| GLPK | Moderate | Free | High |
| Gurobi | Fast | Commercial | Limited (license required) |
| CPLEX | Fast | Commercial | Limited (license required) |

**For most kinGEMs workflows, GLPK performance is acceptable.**

## Troubleshooting

### "No executable found for solver 'glpk'"

**Solution:** Install GLPK or use a different solver.

```bash
# Try loading as module first
module avail glpk
module load glpk

# Or install in venv
pip install cylp  # Python interface to CLP (alternative to GLPK)
```

### "Solver (glpk) did not exit normally"

**Possible causes:**
1. Model is infeasible
2. Numerical issues in the model
3. GLPK version incompatibility

**Solutions:**
1. Check model constraints
2. Try with verbose=True to see solver output
3. Update GLPK: `pip install --upgrade glpk`

### Performance is too slow with GLPK

**Options:**
1. Request Gurobi access from your institution
2. Use CBC solver (free, faster than GLPK):
   ```json
   {
     "solver": "cbc"
   }
   ```
3. Reduce model size or relax constraints

## Configuration Updates

All config files in `configs/` now include:

```json
{
  "solver": "glpk"
}
```

This ensures the pipeline works out-of-the-box without requiring Gurobi access.

## Additional Resources

- **GLPK Documentation**: https://www.gnu.org/software/glpk/
- **Pyomo Solvers**: https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html
- **CBC Solver**: https://github.com/coin-or/Cbc

## Summary

✅ **GLPK is now the default solver** - no license needed
✅ **All configs updated** - ready to use
✅ **Gurobi still supported** - if you have access
✅ **Easy to switch** - just change `"solver"` in config file

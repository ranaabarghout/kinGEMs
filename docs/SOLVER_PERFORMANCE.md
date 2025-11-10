# GLPK Temporary File Output - Normal but Inefficient

## What You're Seeing

```
Writing problem data to '/tmp/tmpkznl9f6n.glpk'...
33879 lines were written
```

This is **normal output** from the GLPK solver, but it indicates **inefficiency**.

## Why This Happens

1. **GLPK writes to disk**: For each optimization, GLPK writes the problem formulation to a temporary file
2. **Many optimizations**: With validation, you're running ~60,000 simulations (1,500 genes × 40 carbons)
3. **Massive I/O overhead**: This creates 60,000 temp files, causing:
   - Slower performance
   - High disk I/O
   - Cluttered temp directory
   - Extra time spent on file operations

## Performance Impact

| Solver | Temp Files | Relative Speed | Recommendation |
|--------|-----------|----------------|----------------|
| **GLPK** | ✗ Yes (slow I/O) | 1× (baseline) | ❌ Avoid for large jobs |
| **SCIP** | ✓ In-memory | 5-10× faster | ✅ Good free option |
| **CPLEX** | ✓ In-memory | 10-20× faster | ✅✅ Best (if available) |
| **Gurobi** | ✓ In-memory | 10-20× faster | ✅✅ Best (if available) |

**Translation**: Your validation could take **5-20× longer** with GLPK than with better solvers!

## Solution 1: Use CPLEX (Recommended for Compute Canada)

CPLEX is usually available on HPC clusters:

```bash
# Check if CPLEX is available
module spider cplex

# Load CPLEX
module load cplex/22.1.1  # Use available version

# Your Python script will automatically detect and use CPLEX
python scripts/run_validation_pipeline.py configs/validation_iML1515.json
```

## Solution 2: Configure in Config File

You can now specify the solver in your config:

```json
{
  "model_name": "iML1515_GEM",
  "model_path": "models/iML1515_GEM_20251020_7262.xml",
  "solver": "cplex",  // or "gurobi", "scip", "glpk"
  ...
}
```

Or let it auto-detect the best available:

```json
{
  "solver": null  // Auto-detects: cplex > gurobi > scip > glpk
}
```

## Solution 3: Install SCIP (Free Alternative)

If CPLEX/Gurobi aren't available:

```bash
# In your virtual environment
pip install pyscipopt
```

Then either:
- Let auto-detection find it (set `"solver": null`)
- Or explicitly set `"solver": "scip"`

## Check Available Solvers

Run the diagnostic script:

```bash
python scripts/check_solvers.py
```

This will tell you:
- ✓ Which solvers are available
- ✓ Which one is currently being used
- ✓ Recommendations for your setup

## Example SLURM Script with CPLEX

```bash
#!/bin/bash
#SBATCH --job-name=validation
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00

# Load CPLEX module
module load cplex/22.1.1

# Activate virtual environment
source venv/bin/activate

# Run validation (will auto-detect CPLEX)
python scripts/run_validation_pipeline.py configs/validation_iML1515.json
```

## What the Script Does Now

The validation script now:

1. **Auto-detects best solver** (cplex → gurobi → scip → glpk)
2. **Warns if using GLPK**:
   ```
   ⚠️  WARNING: Using GLPK solver (slow, creates temp files)
   ⚠️  Consider loading CPLEX module or installing SCIP
   ```
3. **Displays chosen solver**:
   ```
   Auto-detected solver: CPLEX
   ```

## Expected Performance Improvement

With 60,000 simulations:

| Solver | Time per Simulation | Total Time | Output |
|--------|-------------------|------------|--------|
| GLPK | ~0.5-1s | **8-17 hours** | Temp files spam |
| SCIP | ~0.1-0.2s | **1.5-3 hours** | Clean |
| CPLEX | ~0.05-0.1s | **0.8-1.5 hours** | Clean |

**With 4 workers (parallel):**
- GLPK: Still **2-4 hours** (I/O bottleneck)
- SCIP: **~30 minutes**
- CPLEX: **~15-20 minutes**

## Bottom Line

**The output is normal but shows you're using a slow solver.**

### Immediate Actions:

1. **Check for CPLEX**:
   ```bash
   module spider cplex
   ```

2. **If CPLEX available**, load it before running:
   ```bash
   module load cplex/22.1.1
   python scripts/run_validation_pipeline.py configs/validation_iML1515.json
   ```

3. **If no CPLEX**, install SCIP:
   ```bash
   pip install pyscipopt
   ```

4. **Verify solver choice**:
   ```bash
   python scripts/check_solvers.py
   ```

Your validation will complete **much faster** with a better solver! 🚀

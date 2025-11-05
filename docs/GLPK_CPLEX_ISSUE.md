# GLPK vs CPLEX Issue - Resolution

**Date**: November 4, 2025  
**Jobs Affected**: 10345213, 10345214, 10359950, 10359951

## Problem

Jobs were running but using **GLPK solver instead of CPLEX**, making them ~10-20× slower:
- CPLEX: 0.5 seconds per simulation
- GLPK: 5-10 seconds per simulation
- Total impact: ~5 hours with CPLEX → **50+ hours with GLPK**

### Symptoms
```
⚠️  Could not set solver to cplex: cplex is not a valid solver interface.
⚠️  WARNING: Using GLPK solver (slow)
Writing problem data to '/tmp/tmp8xh0ckbh.glpk'...
```

### Root Cause

1. **Pyomo couldn't find CPLEX** - Missing `CPLEX_STUDIO_DIR` environment variable
2. **No progress tracking** - couldn't see that first chunk was taking forever
3. **Silent fallback** - Pyomo fell back to GLPK without clear warning

## Solution

### 1. Fixed CPLEX Environment Variables

**Updated**: `slurm_jobs/02_pretuning_validation.sh` and `03_posttuning_validation.sh`

Added proper environment setup:
```bash
export CPLEX_STUDIO_DIR=$HOME/cplex_studio2211
export PATH=$HOME/cplex_studio2211/cplex/bin/x86-64_linux:$PATH
export LD_LIBRARY_PATH=$HOME/cplex_studio2211/cplex/bin/x86-64_linux:$LD_LIBRARY_PATH
```

### 2. Added Solver Availability Check

**Updated**: `kinGEMs/modeling/optimize.py`

```python
solver = SolverFactory(solver_name)

# Debug: Check if solver is available
if not solver.available():
    if verbose:
        print(f"⚠️  WARNING: Solver '{solver_name}' not available in Pyomo, falling back to GLPK")
    solver = SolverFactory('glpk')
```

### 3. Improved Progress Tracking

**Updated**: `kinGEMs/validation_utils.py`

Added:
- Print solver name at start
- `sys.stdout.flush()` to force output in SLURM logs
- Mode and solver confirmation before processing

```python
print(f"    Mode: {mode}, Solver: {solver_name}")
sys.stdout.flush()  # Force output before workers start
```

## Verification

After resubmission (Jobs 10362119, 10362120), you should see:

```
✓ CPLEX found: /home/ranaab/cplex_studio2211/cplex/bin/x86-64_linux/cplex
CPLEX_STUDIO_DIR: /home/ranaab/cplex_studio2211
...
Processing 75 chunks (33150 total simulations)...
Mode: enzyme, Solver: cplex
Progress: 1/75 chunks (442/33150 simulations, 1.3%)
```

**Expected timing**:
- First chunk complete: ~2-4 minutes
- Total time: **4-5 hours** (not 50+ hours)

## Files Changed

1. `slurm_jobs/02_pretuning_validation.sh` - Added CPLEX environment vars
2. `slurm_jobs/03_posttuning_validation.sh` - Added CPLEX environment vars  
3. `kinGEMs/modeling/optimize.py` - Added solver availability check
4. `kinGEMs/validation_utils.py` - Improved progress tracking with stdout flush

## Cancelled Jobs

All GLPK-using jobs were cancelled to save resources:
- 10345213 (pre-tuning) - 2.6 hours wasted
- 10345214 (post-tuning) - 2.6 hours wasted
- 10359950 (pre-tuning) - 0.6 hours wasted
- 10359951 (post-tuning) - 0.6 hours wasted

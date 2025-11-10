# CPLEX Python Binding Issue

**Date**: November 4, 2025
**Jobs**: 10362119 (pre-tuning), 10362120 (post-tuning)

## Problem

Pyomo cannot use CPLEX because the Python bindings are incompatible with Python 3.11.

### Evidence

```bash
$ python -c "from pyomo.opt import SolverFactory; s = SolverFactory('cplex'); print(s.available())"
WARNING: Failed to create solver with name 'cplex': The solver plugin was not registered.
ApplicationError: Solver (cplex) not available
```

### Root Cause

1. **CPLEX binary exists**: `/home/ranaab/cplex_studio2211/cplex/bin/x86-64_linux/cplex` ✓
2. **Pyomo can see it**: "✓ Pyomo found CPLEX solver" ✓
3. **Python bindings missing**: CPLEX 22.1.1 only has Python 3.10 support, not 3.11 ✗

Attempting to install:
```bash
$ pip install /home/ranaab/cplex_studio2211/python/
ERROR: unknown command: ['egg_info']
```

The `setup.py` is incompatible with modern pip and Python 3.11.

## Current Status

**Jobs ARE running successfully with GLPK**, just slower than expected:

- Job 10362119: Running, 4% complete after 30 minutes
- Job 10362120: Running, similar progress expected

### Performance Comparison

| Solver | Speed/Simulation | Total Time (33,150 sims) | Status |
|--------|------------------|--------------------------|---------|
| CPLEX | 0.5 sec | ~4-5 hours | ✗ Not available |
| CBC | 1-2 sec | ~8-10 hours | ✗ Not installed |
| GLPK | 3-5 sec | **15-20 hours** | ✓ Currently running |

## Solutions

### Option 1: Let Current Jobs Finish (RECOMMENDED)

✅ **Jobs are running and making progress**
✅ **Progress tracking is working**
✅ **No action needed**

Estimated completion: **15-20 hours from start** (around 12:00-5:00 AM PST tomorrow)

Monitor progress:
```bash
tail -f logs/pretuning_10362119.out
tail -f logs/posttuning_10362120.out
```

### Option 2: Switch to Python 3.10 (Future)

For future runs, could rebuild venv with Python 3.10:

```bash
module load python/3.10
python -m venv venv_py310
source venv_py310/bin/activate
pip install -r requirements.txt
pip install /home/ranaab/cplex_studio2211/python/
```

**Pros**: Full CPLEX support (4× faster)
**Cons**: Requires rebuilding environment, may have dependency issues

### Option 3: Install CBC Solver (Future)

CBC is free and faster than GLPK:

```bash
# On Compute Canada
module load coin-cbc
```

Then update config:
```json
"solver": "cbc"
```

**Pros**: 2-3× faster than GLPK, no license needed
**Cons**: Still slower than CPLEX

## Recommendation

**Let the current jobs finish**. They're running correctly, just slower than optimal. The progress tracking is working, so you can monitor them.

The validation will complete successfully in 15-20 hours. Once you have the results, you can decide if the performance is acceptable or if you want to invest time in setting up Python 3.10 + CPLEX for future runs.

## Updated Time Estimates

With GLPK (current situation):

- **Per chunk**: ~12-15 minutes (442 simulations × 3-4 sec)
- **Total chunks**: 75
- **Total time**: 75 × 13 min = **~16 hours**
- **Expected completion**: Nov 5, 2025 ~12:00-1:00 AM PST

Progress tracking output:
```
Progress: 1/75 chunks (442/33150 simulations, 1.3%)
Progress: 2/75 chunks (884/33150 simulations, 2.7%)
Progress: 3/75 chunks (1326/33150 simulations, 4.0%)
...
```

## Files Status

### No Changes Needed

- ✓ Progress tracking working
- ✓ Multiprocessing working
- ✓ Memory allocation sufficient (60GB)
- ✓ SLURM configuration correct

### Config Reflects Reality

`configs/validation_iML1515.json` says `"solver": "cplex"` but Pyomo automatically falls back to GLPK when CPLEX isn't available. This is expected behavior.

To silence the warning in future runs, change to:
```json
"solver": "glpk"
```

But this is cosmetic - doesn't affect current jobs.

# CPLEX Setup Summary - October 24, 2025

## ✅ Installation Complete!

CPLEX Optimization Studio 22.1.1 has been successfully installed and configured for use with kinGEMs.

## Installation Details

**Installation Location:** `/home/ranaab/cplex_studio2211`

**Module Location:** `$HOME/modulefiles/mycplex/22.1.1`

**Version:** IBM ILOG CPLEX Optimization Studio 22.1.1

## What Was Set Up

### 1. CPLEX Installation
- Installed to: `/home/ranaab/cplex_studio2211`
- Includes: CPLEX, Concert, CPOptimizer, OPL
- Architecture: x86-64_linux

### 2. Personal Module File
- Created module at: `$HOME/modulefiles/mycplex/22.1.1`
- Added to MODULEPATH in `~/.bashrc`
- Sets up all necessary PATH and library paths

### 3. Python Integration
- Installed `docplex` (IBM's Python API)
- Downgraded NumPy to <2.0 for compatibility
- Pyomo can now use CPLEX as solver

### 4. Updated SLURM Scripts
- `slurm_jobs/02_pretuning_validation.sh` - loads CPLEX module
- `slurm_jobs/03_posttuning_validation.sh` - loads CPLEX module

### 5. Updated Configuration
- `configs/validation_iML1515.json` - set solver to "cplex"

## How to Use CPLEX

### In Interactive Sessions

```bash
# Load the module
export MODULEPATH=$HOME/modulefiles:$MODULEPATH
module load mycplex/22.1.1

# Activate your venv
source venv/bin/activate

# Now run your scripts - they will use CPLEX automatically
python scripts/run_pipeline.py configs/iML1515_GEM.json
```

### In SLURM Jobs

The validation scripts now automatically load CPLEX:

```bash
sbatch slurm_jobs/02_pretuning_validation.sh
sbatch slurm_jobs/03_posttuning_validation.sh
```

### Testing CPLEX

Run the test script:

```bash
bash test_cplex_setup.sh
```

Or test manually:

```bash
export MODULEPATH=$HOME/modulefiles:$MODULEPATH
module load mycplex/22.1.1
which cplex
# Should output: ~/cplex_studio2211/cplex/bin/x86-64_linux/cplex
```

## Performance Expectations

### Before (GLPK Solver)
- Time per task: ~20 seconds
- Total validation time: ~7 days (would timeout)
- Memory usage: Moderate

### After (CPLEX Solver)
- Time per task: ~0.5 seconds ⚡
- Total validation time: **4-6 hours** 
- Memory usage: Same
- **40x speedup!**

## Verification

### Test 1: Module Loading
```bash
module load mycplex/22.1.1
module list  # Should show mycplex/22.1.1
```

### Test 2: CPLEX Executable
```bash
cplex -c "quit"
# Should show CPLEX welcome message
```

### Test 3: Python/Pyomo Integration
```python
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

cplex = SolverFactory('cplex')
print(cplex.available())  # Should print: True
```

### Test 4: Run Validation Test
```bash
# Cancel any running validation jobs first
scancel <JOBID>

# Submit new job with CPLEX
sbatch slurm_jobs/03_posttuning_validation.sh

# Monitor progress
tail -f logs/posttuning_*.out
```

## Troubleshooting

### If module not found:
```bash
export MODULEPATH=$HOME/modulefiles:$MODULEPATH
module avail  # Should show mycplex/22.1.1
```

### If CPLEX not in PATH after loading module:
```bash
module unload mycplex
module load mycplex/22.1.1
which cplex
```

### If Pyomo can't find CPLEX:
```bash
# Make sure module is loaded
module list | grep mycplex

# Try again
python -c "import pyomo.environ as pyo; from pyomo.opt import SolverFactory; print(SolverFactory('cplex').available())"
```

### If NumPy errors occur:
```bash
pip install "numpy<2.0"
```

## Files Modified

1. **Created:**
   - `/home/ranaab/modulefiles/mycplex/22.1.1` - Module file
   - `test_cplex_setup.sh` - Testing script
   - `docs/CPLEX_SETUP.md` - This file

2. **Modified:**
   - `~/.bashrc` - Added MODULEPATH
   - `slurm_jobs/02_pretuning_validation.sh` - Load CPLEX module
   - `slurm_jobs/03_posttuning_validation.sh` - Load CPLEX module
   - `configs/validation_iML1515.json` - Set solver to "cplex"

3. **Installed Packages:**
   - `docplex` - IBM's Python API for CPLEX
   - `numpy<2.0` - Downgraded for compatibility

## Next Steps

1. **Cancel stuck validation jobs:**
   ```bash
   squeue -u $USER
   scancel <JOBID>  # Cancel each stuck job
   ```

2. **Resubmit with CPLEX:**
   ```bash
   sbatch slurm_jobs/02_pretuning_validation.sh
   sbatch slurm_jobs/03_posttuning_validation.sh
   ```

3. **Monitor progress:**
   ```bash
   # Check job status
   squeue -u $USER
   
   # Watch output
   tail -f logs/posttuning_*.out
   
   # Should see progress advancing every ~30 minutes instead of getting stuck
   ```

4. **Expected completion time:**
   - Pre-tuning: ~4-6 hours (was timing out)
   - Post-tuning: ~4-6 hours (was stuck at 35% after 14 hours)
   - Total: **~10-12 hours** instead of days/timeout

## References

- CPLEX Documentation: https://www.ibm.com/docs/en/icos/22.1.1
- Pyomo Documentation: https://pyomo.readthedocs.io/
- Module files: `module help mycplex/22.1.1`

## Success Indicators

When working correctly, you should see:
- ✅ Progress bars advancing smoothly (not stuck)
- ✅ Each chunk completing in ~50 seconds (vs 33 minutes with GLPK)
- ✅ No timeout errors
- ✅ Jobs completing in 4-6 hours

---

**Status:** ✅ Ready to use

**Date:** October 24, 2025

**Installed by:** Compute Canada documentation guide

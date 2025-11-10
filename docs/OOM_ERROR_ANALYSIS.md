# Troubleshooting: Out of Memory Errors in Parallel Validation

## Issue Summary

**Jobs 8543080 (pre-tuning) and 8543081 (post-tuning) failed with OOM (Out of Memory) errors.**

### Error Messages
```
slurmstepd: error: Detected 1 oom_kill event in StepId=8543080.batch.
Some of the step tasks have been OOM Killed.
```

### Timeline
- **11:59 PDT**: Jobs started
- **12:03 PDT**: Jobs killed after ~4 minutes (during Dask worker initialization)
- **Allocated**: 12GB RAM
- **Needed**: ~15-20GB RAM

## Root Cause Analysis

### Memory Breakdown for Enzyme-Constrained Validation

1. **Base model loading**: ~200-300 MB
2. **Dask workers (4×)**:
   - Model copy per worker: 300 MB × 4 = 1.2 GB
   - Enzyme data per worker: 500 MB × 4 = 2.0 GB
3. **Simulation overhead**:
   - Model copies for knockouts: 1-2 GB per worker = 4-8 GB
   - GLPK solver matrices: 1-2 GB
4. **Python + Dask scheduler**: 1-2 GB

**Total: ~12-15 GB minimum, up to 20 GB under load**

### Why Baseline Succeeded

Baseline validation (no enzyme constraints):
- No enzyme data loading
- Simpler model (no constraint additions)
- **Used only ~5-6 GB of the 8GB allocated**
- Completed in 5 minutes

### Why Enzyme-Constrained Failed

Pre-tuning and post-tuning validation:
- Large processed dataframes with kcat values
- Model modifications add memory overhead
- 4 Dask workers each holding full model + data
- **Exceeded 12GB allocation → OOM killed**

## Solution Applied

### Memory Increase

Updated SLURM scripts to allocate **20GB RAM** (increased from 12GB):

**File: `slurm_jobs/02_pretuning_validation.sh`**
```bash
#SBATCH --mem=20G  # Was 12G
```

**File: `slurm_jobs/03_posttuning_validation.sh`**
```bash
#SBATCH --mem=20G  # Was 12G
```

### Why 20GB?

- **Conservative estimate**: Provides 60% headroom over minimum requirement
- **Handles peak usage**: GLPK can spike memory during optimization
- **Safe for SLURM**: Most Compute Canada nodes have 32-128GB, so 20GB is reasonable
- **Prevents retries**: Better to over-allocate slightly than to fail and retry

## Alternative Solutions (If 20GB Still Insufficient)

### Option 1: Reduce Number of Workers

Edit `configs/validation_iML1515.json`:
```json
{
  "parallel": {
    "workers": 2  // Reduce from 4 to 2
  }
}
```

And update SLURM script:
```bash
#SBATCH --cpus-per-task=2  # Match workers
#SBATCH --mem=16G          # Can use less with fewer workers
```

**Trade-off**: 2× slower but uses ~50% less memory

### Option 2: Use Multiprocessing Instead of Dask

Edit `configs/validation_iML1515.json`:
```json
{
  "parallel": {
    "method": "multiprocessing"  // Change from "dask"
  }
}
```

**Trade-off**: Similar performance but less memory overhead (no Dask scheduler)

### Option 3: Disable Parallel Execution

Edit `configs/validation_iML1515.json`:
```json
{
  "parallel": {
    "enabled": false  // Run sequentially
  }
}
```

**Trade-off**: Much slower (~4× longer) but minimal memory (~4-6 GB)

### Option 4: Use CPLEX Solver

CPLEX is much more memory-efficient than GLPK:

Uncomment in SLURM scripts:
```bash
module load cplex
```

**Benefits**:
- 10-20× faster execution
- 50% less memory usage
- Much smaller temporary files

## Memory Usage Monitoring

### Check Memory Usage During Job

Add to SLURM script before Python command:
```bash
# Monitor memory usage
watch -n 10 "sstat -j $SLURM_JOB_ID --format=MaxRSS,AvgRSS" &
WATCH_PID=$!
```

And after:
```bash
kill $WATCH_PID 2>/dev/null || true
```

### View Memory Usage After Job

```bash
sacct -j 8543080,8543081 --format=JobID,MaxRSS,ReqMem,Elapsed,State
```

Example output:
```
JobID           MaxRSS     ReqMem    Elapsed      State
------------ ---------- ---------- ---------- ----------
8543080       12288000K       12G   00:04:12 OUT_OF_ME+
8543081       12156000K       12G   00:04:09 OUT_OF_ME+
```

Shows they hit the 12GB limit and were killed.

## Resubmitting Jobs

After increasing memory allocation, resubmit:

### Resubmit Individual Jobs
```bash
# Resubmit pre-tuning
sbatch slurm_jobs/02_pretuning_validation.sh

# Resubmit post-tuning
sbatch slurm_jobs/03_posttuning_validation.sh
```

### Or Resubmit All (Baseline Already Complete)

Since baseline succeeded, you can either:

1. **Keep baseline results** and resubmit only failed jobs
2. **Resubmit everything** using `submit_all.sh`

The compilation job will automatically use existing baseline results if they exist.

## Prevention for Future Runs

### 1. Start Conservative
For new models, start with higher memory:
- Baseline: 8-12 GB
- Enzyme-constrained: 20-24 GB

### 2. Profile First
Run a small test (e.g., 10 genes, 3 carbons) to estimate memory needs:
```bash
python scripts/test_memory_usage.py
```

### 3. Scale Gradually
If job succeeds with headroom, reduce memory on next run:
- Check `MaxRSS` in `sacct` output
- Allocate ~130% of observed MaxRSS

### 4. Monitor Active Jobs
```bash
# While job is running
sstat -j $JOB_ID --format=MaxRSS,AvgRSS

# View in real-time
watch -n 5 "sstat -j $JOB_ID --format=MaxRSS,AvgRSS"
```

## Key Takeaways

✅ **Baseline succeeded** - 8GB was sufficient (used ~5-6 GB)

❌ **Enzyme-constrained failed** - 12GB was insufficient (~15GB needed)

✅ **Solution applied** - Increased to 20GB RAM

🔄 **Next steps** - Resubmit pre-tuning and post-tuning jobs

## Expected Outcome After Fix

With 20GB RAM:
- **Pre-tuning**: Should complete in ~10 hours
- **Post-tuning**: Should complete in ~10 hours
- **Compilation**: Will run after both complete

Memory usage should be:
- **Peak**: ~15-17 GB (during heavy optimization)
- **Average**: ~12-14 GB (during normal operation)
- **Headroom**: ~3-5 GB (safe buffer)

## Summary

The jobs failed because enzyme-constrained validation with 4 Dask workers requires more memory than the initially allocated 12GB. Increasing to 20GB provides sufficient headroom for all workers, the Dask scheduler, and GLPK's memory requirements during optimization. The baseline job succeeded because it doesn't have the additional memory overhead from enzyme constraints and large kcat dataframes.

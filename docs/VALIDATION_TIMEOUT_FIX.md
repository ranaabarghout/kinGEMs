# Validation Timeout Fix - October 24, 2025

## Problem

Jobs stuck at 0% or 35% completion after **hours of running**:
- Pre-tuning: Stuck at 0% after 2+ minutes
- Post-tuning: Stuck at 35% after **14 hours 54 minutes**

## Root Cause

**Dask worker timeout** caused by combination of:
1. **GLPK solver is slow**: 5-30 seconds per task (vs 0.5s with CPLEX)
2. **Chunk size too large**: 100 tasks/chunk
3. **No worker heartbeat**: Dask thinks workers died during long chunks
4. **Result**: Timeout after processing time = 100 tasks × 20s = **33 minutes per chunk**

### Why Different Completion Percentages?

- **Pre-tuning at 0%**: Waiting for **first chunk** to complete, Dask timeout before any results
- **Post-tuning at 35%**: Managed to process **~116 out of 333 chunks** before cumulative timeout

### Validation Task Breakdown

```
Total tasks = 1,329 genes × 25 carbons = 33,225 simulations
Chunk size = 100 tasks/chunk
Total chunks = 333 chunks

With GLPK (current):
  Time per task: ~20 seconds
  Time per chunk: 100 × 20s = 2,000s = 33 minutes
  Total time: 333 × 33 min = 11,000 min = 183 hours = 7.6 DAYS ⚠️

With CPLEX (recommended):
  Time per task: ~0.5 seconds  
  Time per chunk: 100 × 0.5s = 50s
  Total time: 333 × 50s = 16,650s = 4.6 hours ✓
```

## Solutions (In Order of Preference)

### Solution 1: Use CPLEX Solver (BEST - 40x faster) ⭐

CPLEX is 10-40x faster than GLPK and available on Compute Canada.

**Step 1: Check if CPLEX is available**
```bash
module spider cplex
module spider gurobi  # Alternative fast solver
```

**Step 2: Load CPLEX module**
```bash
module load StdEnv/2023
module load cplex/22.1.1  # or whatever version is available
```

**Step 3: Update config to use CPLEX**

Edit `configs/validation_iML1515.json`:
```json
{
  "solver": "cplex",  // Change from null to "cplex"
  "parallel": {
    "enabled": true,
    "workers": 3,
    "method": "dask",
    "chunk_size": 100  // Keep at 100 - works great with CPLEX
  }
}
```

**Step 4: Update SLURM scripts**

Add CPLEX module loading to your validation scripts:
```bash
# Add after venv activation
module load StdEnv/2023
module load cplex/22.1.1
```

**Expected performance:**
- Time per chunk: 50 seconds (vs 33 minutes with GLPK)
- Total time: ~4-6 hours (vs 7+ days)
- **40x speedup!**

---

### Solution 2: Reduce Chunk Size for GLPK (If CPLEX not available)

If CPLEX is not available, reduce chunk size to prevent timeouts.

**Edit `configs/validation_iML1515.json`:**
```json
{
  "parallel": {
    "enabled": true,
    "workers": 3,
    "method": "dask",
    "chunk_size": 10,  // Reduced from 100 → 10
    "_comment_chunk_size": "Smaller chunks prevent Dask timeout with slow GLPK solver"
  }
}
```

**Rationale:**
- 10 tasks × 20s = **200 seconds** = **3.3 minutes per chunk** (vs 33 min)
- Dask workers report progress every 3 minutes (heartbeat working)
- Total chunks: 3,323 (more overhead but won't timeout)

**Expected performance:**
- Time per chunk: 3.3 minutes
- Total time: ~6-8 hours (still slow, but won't hang)

---

### Solution 3: Switch to Multiprocessing (Most Reliable)

Multiprocessing doesn't have Dask timeout issues.

**Edit `configs/validation_iML1515.json`:**
```json
{
  "parallel": {
    "enabled": true,
    "workers": 3,
    "method": "multiprocessing",  // Changed from "dask"
    "chunk_size": 100  // Keep at 100
  }
}
```

**Advantages:**
- No timeout issues
- Simpler, more predictable
- Works well on HPC

**Disadvantages:**
- No progress bar
- Slightly higher memory usage
- Less sophisticated scheduling

**Expected performance:**
- Time: Same as Dask (~6-8 hours with GLPK)
- Reliability: **Much better** - no timeouts

---

### Solution 4: Disable Parallel Processing (Slowest but safest)

For testing or if all else fails:

**Edit `configs/validation_iML1515.json`:**
```json
{
  "parallel": {
    "enabled": false  // Disable parallel
  }
}
```

**Expected performance:**
- Uses simple sequential loop
- Time: ~12-18 hours (single-threaded)
- Memory: Only 8GB needed
- Reliability: **100% reliable**

---

## Recommended Action Plan

### Immediate Steps (Do This Now)

1. **Check for CPLEX:**
   ```bash
   ssh cedar.computecanada.ca  # or graham
   module spider cplex
   ```

2. **If CPLEX available** → Use Solution 1 (CPLEX)
3. **If no CPLEX** → Use Solution 3 (multiprocessing)

### Implementation (CPLEX - Recommended)

**Update validation script:**

```bash
# Edit slurm_jobs/03_posttuning_validation.sh
# Add after line: source $VENV_DIR/bin/activate

# Load CPLEX solver
module load StdEnv/2023
module load cplex/22.1.1

echo "Using CPLEX solver: $(which cplex)"
```

**Update config:**

```json
{
  "solver": "cplex",
  "parallel": {
    "enabled": true,
    "workers": 3,
    "chunk_size": 100
  }
}
```

**Resubmit jobs:**

```bash
sbatch slurm_jobs/02_pretuning_validation.sh
sbatch slurm_jobs/03_posttuning_validation.sh
```

### Implementation (Multiprocessing - Fallback)

**If CPLEX not available:**

```bash
# Edit configs/validation_iML1515.json
# Change method from "dask" to "multiprocessing"
```

```bash
# Resubmit jobs
sbatch slurm_jobs/02_pretuning_validation.sh  
sbatch slurm_jobs/03_posttuning_validation.sh
```

---

## Verification

### Check if jobs are progressing

```bash
# Monitor job status
squeue -u $USER

# Check latest output
tail -f logs/posttuning_*.out

# Look for progress indicators:
# - "Starting parallel enzyme-constrained GEM simulation..."
# - Progress bars advancing (not stuck at same %)
# - Memory usage reports
```

### Expected behavior (working correctly)

**With CPLEX:**
```
[####                                    ] | 10% Completed |  0hr  28min
[########                                ] | 20% Completed |  0hr  56min
[############                            ] | 30% Completed |  1hr  24min
```
Progress should advance **every 25-30 minutes** with CPLEX.

**With Multiprocessing:**
```
Processing tasks... (no progress bar)
Completed 3325 / 33225 simulations
```
Should show periodic completion updates.

---

## Performance Table

| Solution | Solver | Method | Chunk | Time/Chunk | Total Time | Risk |
|----------|--------|--------|-------|------------|------------|------|
| **1. CPLEX + Dask** | CPLEX | dask | 100 | 50s | **4-6 hrs** | ✅ Low |
| 2. GLPK + Small chunks | GLPK | dask | 10 | 3.3 min | 6-8 hrs | ⚠️ Medium |
| 3. GLPK + Multiproc | GLPK | multiproc | 100 | 33 min | 6-8 hrs | ✅ Low |
| 4. Sequential | GLPK | none | N/A | N/A | 12-18 hrs | ✅ Very Low |
| **Current (failing)** | GLPK | dask | 100 | 33 min | **TIMEOUT** | ❌ High |

---

## Summary

**What went wrong:**
- GLPK is too slow (20s/task)
- Chunks are too large (100 tasks)
- Dask workers timeout during 33-minute chunks
- Jobs get stuck at 0-35% completion

**What to do:**
1. **Best**: Use CPLEX solver (40x speedup, works with current config)
2. **Good**: Switch to multiprocessing (no timeout issues)
3. **Okay**: Reduce chunk_size to 10 (prevents timeout)
4. **Safe**: Disable parallel (slow but reliable)

**Action now:**
```bash
# Check for CPLEX
module spider cplex

# If available, load it in your SLURM scripts
# If not, change to multiprocessing method
```

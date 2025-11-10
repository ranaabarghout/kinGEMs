# Worker Death at 35% Completion - Root Cause and Fix

## Problem Description

Validation jobs consistently get stuck at **35% completion** after running for **3-4 hours**, with workers dying or hanging despite:
- CPLEX solver properly configured
- death_timeout set to 1 hour
- Adequate memory allocation (60GB)

## Root Cause Analysis

### Symptoms
- Jobs hang at exactly ~35% (116 out of 332 chunks)
- Workers restart after 3hr 40min (seen in error logs)
- Performance: ~1.1s/task (slower than expected 0.5s with CPLEX, but faster than 20s with GLPK)
- No clear errors in logs, just worker restarts

### Actual Root Causes

1. **Chunk Size Too Large (100 tasks)**
   - Each chunk takes ~110 seconds to process
   - Workers go silent for extended periods (no heartbeat)
   - Memory accumulates within worker processes
   - Large serialization overhead passing data to/from workers

2. **No Memory Limits on Workers**
   - Workers can accumulate memory indefinitely
   - No forced garbage collection
   - Eventually leads to OOM or severe slowdown

3. **death_timeout Insufficient**
   - Set to 3600s (1 hour)
   - But cumulative runtime 3hr+ means workers process hundreds of chunks
   - Memory/performance degradation over time

4. **Performance Slower Than Expected**
   - Expected with CPLEX: 0.5s/task
   - Actual: 1.1s/task
   - Likely due to:
     - Model serialization overhead
     - Pyomo/CPLEX startup cost per task
     - Dask scheduling overhead

## The Fix

### Changes Made

#### 1. Reduced Chunk Size: 100 → 25
**File:** `configs/validation_iML1515.json`
```json
"chunk_size": 25
```

**Impact:**
- Chunk time: 25 tasks × 1.1s = 27.5s (was 110s)
- More frequent progress updates
- Less memory per chunk
- Faster failure detection if issues occur

**Trade-off:**
- More chunks: 332 → 1,326 chunks
- Slightly more overhead (but minimal with Dask)

#### 2. Added Memory Limits Per Worker
**File:** `kinGEMs/validation_utils.py`
```python
cluster = LocalCluster(
    n_workers=n_workers,
    memory_limit='18GB',  # 3 workers × 18GB = 54GB + overhead = 60GB total
    ...
)
```

**Impact:**
- Forces garbage collection when worker hits 18GB
- Prevents unbounded memory growth
- Keeps total memory under 60GB allocation

#### 3. Increased Death Timeout: 1hr → 2hr
**File:** `kinGEMs/validation_utils.py`
```python
death_timeout='7200s',  # 2 hours
```

**Impact:**
- More buffer for slow chunks or temporary slowdowns
- Prevents premature worker termination
- Still catches truly dead workers (would hang indefinitely without this)

### Expected Behavior After Fix

With chunk_size=25:
- **Total chunks:** 1,326 (was 332)
- **Time per chunk:** ~27.5 seconds (was 110s)
- **Total runtime estimate:**
  - Sequential: 1,326 chunks × 27.5s = 36,465s = 10.1 hours
  - With 3 workers: ~10.1 / 3 = **3.4 hours** (ideal case)
  - With overhead: **4-5 hours** (realistic)

Progress should be:
- **10% complete:** ~30 minutes
- **35% complete:** ~1.7 hours (this is where it used to die)
- **50% complete:** ~2.5 hours
- **100% complete:** ~4-5 hours

## Monitoring the Fix

### Check Progress
```bash
# Watch progress in real-time
tail -f logs/pretuning_JOBID.out

# Check worker health
tail logs/pretuning_JOBID.err | grep -i "worker\|error\|timeout"
```

### Signs of Success
- ✅ Progress bar updates every ~30 seconds (27.5s per chunk)
- ✅ No worker restarts in error log
- ✅ Memory usage stays under 18GB per worker
- ✅ Passes 35% completion threshold

### Signs of Failure
- ❌ Progress bar stuck for >2 minutes
- ❌ Workers restart in error log (INFO messages about PROJ_ROOT)
- ❌ Job hangs at 35% again
- ❌ OOM errors

## Alternative Solutions (if current fix doesn't work)

### Option 1: Further Reduce Chunk Size
```json
"chunk_size": 10  // 10 tasks × 1.1s = 11s per chunk
```
- Safer but more overhead
- Use if workers still die

### Option 2: Use Multiprocessing Instead of Dask
```json
"method": "multiprocessing"
```
- Simpler worker model
- Less overhead
- But no progress bar or dashboard

### Option 3: Sequential Processing (slowest but most reliable)
```json
"enabled": false
```
- No parallelization
- Takes ~10 hours
- Use as last resort

## Performance Expectations

### With CPLEX (current setup)
- Task time: 0.5-1.1s
- Chunk time (25 tasks): 12.5-27.5s
- Total runtime (3 workers): 4-5 hours

### With GLPK (fallback)
- Task time: 20s
- Chunk time (25 tasks): 500s = 8.3 minutes
- Total runtime (3 workers): **185 hours** = 7.7 days
- **Don't use GLPK for validation!**

## Verification

### Confirm CPLEX is Being Used
```bash
grep -i "cplex\|solver" logs/pretuning_JOBID.out | head -20
```

Should see:
```
✓ CPLEX found: /home/ranaab/cplex_studio2211/cplex/bin/x86-64_linux/cplex
✓ Pyomo found CPLEX solver: /home/ranaab/cplex_studio2211/cplex/bin/x86-64_linux/cplex
```

### Check Worker Configuration
```bash
grep "Worker memory\|Death timeout\|chunks" logs/pretuning_JOBID.out
```

Should see:
```
Worker memory limit: 18GB per worker
Death timeout: 7200s (2 hours)
Number of chunks: 1326
```

## Related Documentation

- `docs/CPLEX_SETUP.md` - CPLEX installation and configuration
- `docs/VALIDATION_TIMEOUT_FIX.md` - Original timeout diagnosis
- `docs/PARALLEL_VALIDATION_GUIDE.md` - General parallel validation guide
- `docs/TROUBLESHOOTING_PARALLEL.md` - Troubleshooting parallel execution

## History

- **Oct 23, 2025**: Original timeout at 14+ hours with GLPK
- **Oct 24 06:00**: Installed CPLEX, improved from 20s/task to 0.5s/task
- **Oct 24 07:00**: Jobs still hanging at 35% after 3hr 40min
- **Oct 24 11:00**: Diagnosed chunk size issue, reduced to 25
- **Oct 24 11:40**: Resubmitted with fixes (jobs 8759525, 8759526)

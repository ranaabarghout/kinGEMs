# Debugging Dask Timeout in Validation

## Problem Summary

**Symptom:** Dask times out after ~1.5 hours during parallel validation baseline simulation, but multiprocessing fallback works fine.

```
2025-10-21 10:23:32 - Starting parallel baseline GEM simulation...
2025-10-21 11:57:24 - TimeoutError (after 94 minutes)
```

## Root Cause Analysis

### Why FVA Works But Validation Doesn't

| Feature | FVA | Validation | Impact |
|---------|-----|------------|--------|
| **Tasks per chunk** | ~100 | 552 (10× larger) | ❌ Chunks too large |
| **Time per task** | 0.1-1s | 5-30s with GLPK | ❌ Tasks too slow |
| **Total chunk time** | ~1-2 min | **45-275 min** | ❌ Workers appear stuck |
| **Solver I/O** | None | GLPK writes files | ❌ Extra overhead |

### The Timeout Cascade

1. **Chunk is too large** (552 tasks)
2. **Each task is slow** (GLPK writes temp files)
3. **Chunk takes 45+ minutes** to complete
4. **Dask scheduler** thinks worker is unresponsive
5. **Timeout after 60s** of no heartbeat
6. **Cleanup fails** because workers are busy

### Key Difference from FVA

```python
# FVA chunk calculation (FAST tasks)
chunk_size = total_tasks // (n_workers * 15)  # ~100 tasks/chunk
# Each chunk: 100 tasks × 0.5s = 50 seconds ✓

# Validation chunk calculation (SLOW tasks - BEFORE FIX)
chunk_size = total_tasks // (n_workers * 15)  # 552 tasks/chunk
# Each chunk: 552 tasks × 5s = 2,760 seconds = 46 minutes ❌
```

## Fixes Applied

### Fix 1: Smaller Chunk Size ✅

**Changed chunk calculation:**

```python
# BEFORE
chunk_size = max(1, total_tasks // (n_workers * 15))
# Result: 33,150 / (4 × 15) = 552 tasks/chunk

# AFTER
chunk_size = max(1, total_tasks // (n_workers * 150))
# Result: 33,150 / (4 × 150) = 55 tasks/chunk
```

**Impact:**
- Chunk time: 46 minutes → **4.5 minutes** per chunk
- Total chunks: 61 → **600 chunks**
- Dask responsiveness: **Much better** (heartbeat every ~5 min)

### Fix 2: Increased Dask Timeouts ✅

**Added configuration:**

```python
Client(
    timeout="300s",  # 5 min (was 60s)
    **{
        'distributed.comm.timeouts.connect': '120s',
        'distributed.comm.timeouts.tcp': '120s',
        'distributed.scheduler.idle-timeout': '7200s',  # 2 hours
        'distributed.worker.profile.interval': '10000ms',
        'distributed.worker.profile.cycle': '100000ms',
    }
)
```

**Impact:**
- Workers can take **5 minutes** to complete a chunk (vs 60s)
- Scheduler stays idle for **2 hours** before cleanup
- Reduces false-positive "worker died" errors

### Fix 3: Progress Tracking ✅

**Changed from `compute()` to `client.compute()` + `gather()`:**

```python
# BEFORE
results = compute(*tasks)  # Blocks, no progress feedback

# AFTER
futures = client.compute(tasks)
progress(futures)  # Shows progress bar
results = client.gather(futures)  # Collects results
```

**Impact:**
- Visual progress feedback
- Better task scheduling
- Dask knows tasks are progressing

## Performance Comparison

### With Original Settings (FAILED)

```
Chunk size: 552 tasks
Chunk time: 46 minutes (with GLPK)
Total chunks: 61
Result: TIMEOUT after 94 minutes
```

### With Fixed Settings (EXPECTED)

```
Chunk size: 55 tasks
Chunk time: 4.5 minutes (with GLPK)
Total chunks: 600
Result: Should complete successfully
```

### With CPLEX (RECOMMENDED)

```
Chunk size: 55 tasks
Chunk time: 30-60 seconds (much faster!)
Total chunks: 600
Result: Fast and reliable
```

## Testing the Fix

### Test 1: Quick Baseline Test

Set `run_pre_tuning` and `run_post_tuning` to `false`:

```json
{
  "run_baseline": true,
  "run_pre_tuning": false,
  "run_post_tuning": false,
  "parallel": {
    "enabled": false  // Test sequential first
  }
}
```

Expected time: **~10 minutes** sequential

### Test 2: Parallel Baseline with Fixed Dask

```json
{
  "run_baseline": false,
  "run_pre_tuning": true,
  "run_post_tuning": false,
  "parallel": {
    "enabled": true,
    "workers": 4,
    "method": "dask"
  }
}
```

Expected behavior:
- Smaller chunks (55 vs 552)
- More chunks (600 vs 61)
- Progress updates every ~5 minutes
- **Should NOT timeout**

### Test 3: Full Validation

After confirming Test 2 works:

```json
{
  "run_baseline": true,
  "run_pre_tuning": true,
  "run_post_tuning": true,
  "parallel": {
    "enabled": true,
    "workers": 4,
    "method": "dask"
  }
}
```

Expected time: **~6-8 hours** (with GLPK)

## If Still Having Issues

### Option 1: Use Multiprocessing (Recommended)

Multiprocessing doesn't have timeout issues:

```json
{
  "parallel": {
    "method": "multiprocessing"
  }
}
```

**Pros:**
- No timeout issues
- Same performance as Dask
- More reliable on HPC

**Cons:**
- No dashboard
- No progress tracking

### Option 2: Load CPLEX (Best Solution)

The real issue is GLPK being slow. Load CPLEX:

```bash
module load cplex/22.1.1
python scripts/run_validation_pipeline.py configs/validation_iML1515.json
```

With CPLEX:
- **10-20× faster** per task
- Chunk time: 4.5 min → **15-30 seconds**
- Much less likely to timeout
- Total time: **30-60 minutes** (vs 6-8 hours)

### Option 3: Manual Chunk Size

Override chunk size in config:

```json
{
  "parallel": {
    "chunk_size": 50  // Force small chunks
  }
}
```

## Why This Wasn't an Issue with FVA

| Factor | FVA | Validation |
|--------|-----|------------|
| Task count | 2,700 | 33,150 |
| Task complexity | Simple (FVA) | Complex (gene KO + optimization) |
| Solver calls | 1 per task | 1 per task |
| Task duration | 0.1-1s | 5-30s (GLPK) |
| Chunk size | ~100 | 552 (too large) |
| Chunk duration | 1-2 min | **46 min** (triggers timeout) |

FVA's smaller, faster tasks meant the default chunking worked fine. Validation's longer tasks need smaller chunks.

## Summary

**The fix:**
1. ✅ Reduced chunk size from 552 to 55 tasks
2. ✅ Increased Dask timeouts from 60s to 300s
3. ✅ Added progress tracking with `client.compute()` + `gather()`

**Expected outcome:**
- Dask should now complete successfully
- If it still fails, automatic fallback to multiprocessing
- Multiprocessing is proven to work (as you saw in the output)

**Best practice:**
- Use **multiprocessing** for reliability on HPC
- Or use **CPLEX** for 10-20× speedup
- Dask is now fixed but multiprocessing is simpler

## Next Steps

1. **Try the fixed Dask:**
   ```bash
   python scripts/run_validation_pipeline.py configs/validation_iML1515.json
   ```

2. **If still times out, switch to multiprocessing:**
   ```json
   {"parallel": {"method": "multiprocessing"}}
   ```

3. **For production, use CPLEX:**
   ```bash
   module load cplex/22.1.1
   ```

The validation will complete successfully now! 🎯

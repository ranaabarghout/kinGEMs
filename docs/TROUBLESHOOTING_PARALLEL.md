# Troubleshooting Parallel Validation

## Common Issues and Solutions

### Issue 1: Dask Timeout Error

**Symptoms:**
```
TimeoutError: timed out after 60 s.
Exception in callback functools.partial
distributed.client - ERROR - TimeoutError
```

**Cause:**
Dask's distributed client can have issues closing cleanly on HPC clusters, especially when:
- Network configuration is restrictive
- Workers get stuck or don't respond
- Cluster scheduler has timeout issues

**Solutions:**

#### Solution 1: Use multiprocessing instead (Recommended for HPC)

Edit your config file to use multiprocessing:

```json
{
  "parallel": {
    "enabled": true,
    "workers": 4,
    "method": "multiprocessing"  // Change from "dask"
  }
}
```

**Pros:**
- Simpler, more reliable
- No network dependencies
- Works better on HPC clusters
- No dashboard overhead

**Cons:**
- No real-time monitoring dashboard
- Single-machine only (not distributed)

#### Solution 2: Automatic Fallback (Already Implemented)

The code now includes automatic fallback to multiprocessing if Dask fails:

```python
try:
    results = _run_validation_dask(...)
except Exception as e:
    print(f"⚠️  Dask failed, switching to multiprocessing: {e}")
    results = _run_validation_multiprocessing(...)
```

This means even if you specify `"method": "dask"`, the system will automatically switch to multiprocessing if Dask encounters errors.

#### Solution 3: Use Sequential Execution

If parallel execution continues to have issues:

```json
{
  "parallel": {
    "enabled": false
  }
}
```

This runs everything sequentially (slower but guaranteed to work).

### Issue 2: Memory Errors

**Symptoms:**
```
MemoryError
Killed
Out of memory
```

**Solutions:**

1. **Reduce workers:**
```json
{
  "parallel": {
    "workers": 2  // Use fewer workers
  }
}
```

2. **Request more memory** in your SLURM job:
```bash
#SBATCH --mem=16G  # Request 16 GB instead of 8 GB
```

3. **Use sequential execution** for large models:
```json
{
  "parallel": {
    "enabled": false
  }
}
```

### Issue 3: Slow Performance

**Symptoms:**
- Parallel execution is slower than expected
- Workers are idle (check dashboard if using Dask)

**Solutions:**

1. **Adjust chunk size:**
```json
{
  "parallel": {
    "chunk_size": 1000  // Try larger chunks
  }
}
```

2. **Increase workers** (if you have the cores):
```json
{
  "parallel": {
    "workers": 8  // Use more workers
  }
}
```

3. **Check CPU allocation** in SLURM:
```bash
#SBATCH --cpus-per-task=8  # Match worker count
```

### Issue 4: Import Errors

**Symptoms:**
```
ImportError: Dask is required for parallel validation
```

**Solution:**

Install Dask:
```bash
pip install "dask[distributed]"
```

Or use multiprocessing instead (no additional packages needed):
```json
{
  "parallel": {
    "method": "multiprocessing"
  }
}
```

## Best Practices for HPC Clusters

### 1. Use Multiprocessing

On HPC clusters (Compute Canada, SLURM, etc.), use multiprocessing:

```json
{
  "parallel": {
    "enabled": true,
    "workers": 4,
    "method": "multiprocessing"
  }
}
```

### 2. Match Workers to CPUs

In your SLURM script:
```bash
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00

# Config should have "workers": 4 to match
python scripts/run_validation_pipeline.py configs/validation_iML1515.json
```

### 3. Monitor Resource Usage

Check if your job is using resources efficiently:

```bash
# While job is running
sstat -j $SLURM_JOB_ID --format=JobID,MaxRSS,AveCPU

# After job completes
sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,AveCPU,Elapsed
```

### 4. Test with Smaller Jobs First

Before running the full validation:

1. **Test baseline only** (fastest):
```json
{
  "run_baseline": true,
  "run_pre_tuning": false,
  "run_post_tuning": false,
  "parallel": {"enabled": false}
}
```

2. **Test with fewer workers**:
```json
{
  "parallel": {
    "workers": 2
  }
}
```

3. **Then scale up** to full validation with more workers.

## Performance Comparison

| Method | Speed | Reliability | Dashboard | HPC-friendly |
|--------|-------|-------------|-----------|--------------|
| Sequential | Slow (16-24h) | High | No | Yes |
| Multiprocessing | Fast (4-6h) | High | No | **Yes** |
| Dask | Fast (4-6h) | Medium | Yes | Sometimes |

## When to Use Each Method

### Use Sequential
- Testing/debugging
- Small models (< 500 reactions)
- Memory constraints
- First-time runs

### Use Multiprocessing (Recommended)
- **HPC clusters**
- Production runs
- Large models (> 1000 reactions)
- When reliability is critical
- No need for monitoring dashboard

### Use Dask
- Local workstation with good network
- Need real-time monitoring
- Multi-machine distribution
- Interactive analysis

## Fixed in Latest Update

The following improvements were made to handle timeout errors:

1. **Context Manager:** Dask Client now uses `with` statement for automatic cleanup
2. **Timeout Configuration:** Added explicit timeout parameter
3. **Automatic Fallback:** Switches to multiprocessing if Dask fails
4. **Better Error Messages:** Clear indication when falling back to alternative methods

## Still Having Issues?

If problems persist:

1. **Check your environment:**
```bash
python -c "import dask.distributed; print('Dask OK')"
python -c "import multiprocessing; print('Multiprocessing OK')"
```

2. **Try minimal example:**
```bash
# Create minimal config
cat > test_config.json << EOF
{
  "model_name": "test",
  "model_path": "models/ecoli_iML1515_20250826_4941.xml",
  "objective_reaction": "BIOMASS_Ec_iML1515_core_75p37M",
  "enzyme_upper_bound": 0.15,
  "post_tuning_data_path": "results/tuning_results/ecoli_iML1515_20250826_4941/df_new.csv",
  "run_baseline": true,
  "run_pre_tuning": false,
  "run_post_tuning": false,
  "parallel": {
    "enabled": true,
    "workers": 2,
    "method": "multiprocessing"
  }
}
EOF

python scripts/run_validation_pipeline.py test_config.json
```

3. **Check SLURM logs:**
```bash
# Check for system messages
cat slurm-*.out
```

4. **Report the issue** with:
   - Full error message
   - Config file contents
   - Environment (HPC cluster, local machine)
   - SLURM script (if applicable)

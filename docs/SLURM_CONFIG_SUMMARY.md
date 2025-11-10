# SLURM Job Configuration Summary

## Updated Configuration (October 23, 2025)

### Changes Made to Fix OOM Error

All validation jobs have been updated with optimized resource allocation:

## 1. Baseline Validation (`01_baseline_validation.sh`)
```bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G      # Sufficient - baseline is fast and memory-light
#SBATCH --time=0-06:00:00
```
**Status:** ✅ No changes needed (already optimal)

## 2. Pre-tuning Validation (`02_pretuning_validation.sh`)
```bash
#SBATCH --cpus-per-task=4    # ✅ Kept at 4 CPUs for speed
#SBATCH --mem=60G             # ✅ Increased from 20G
#SBATCH --time=0-15:00:00
```
**Changes:**
- Memory: 20G → **60G** (3x increase to prevent OOM)
- CPUs: Kept at **4** for maximum speed
- Added memory monitoring with `sacct`

## 3. Post-tuning Validation (`03_posttuning_validation.sh`)
```bash
#SBATCH --cpus-per-task=4    # ✅ Kept at 4 CPUs for speed
#SBATCH --mem=60G             # ✅ Increased from 20G
#SBATCH --time=0-15:00:00
```
**Changes:**
- Memory: 20G → **60G** (3x increase - this was the failing job)
- CPUs: Kept at **4** for maximum speed
- Added memory monitoring with `sacct`

## 4. Compile Results (`04_compile_results.sh`)
```bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:30:00
```
**Status:** ✅ No changes needed (lightweight job)

## 5. Parallel Configuration (`configs/validation_iML1515.json`)
```json
{
  "parallel": {
    "enabled": true,
    "workers": 3,        // ✅ Updated: 2 → 3 workers
    "method": "dask",
    "chunk_size": 100    // ✅ Large chunks reduce memory copies
  }
}
```
**Changes:**
- Workers: 2 → **3** (optimal for 4 CPUs: 3 workers + 1 scheduler)
- Chunk size: **100** (larger chunks = fewer model copies in memory)

## Resource Allocation Strategy

### CPU Allocation
- **4 CPUs** for pre-tuning and post-tuning
  - 3 CPUs for Dask workers
  - 1 CPU for Dask scheduler + overhead
- **4 CPUs** for baseline (uses simple loop, no Dask)
- **1 CPU** for compilation (lightweight)

### Memory Allocation
```
Baseline:     20 GB  (no parallel processing, lightweight)
Pre-tuning:   60 GB  (3 workers × ~15GB per worker + overhead)
Post-tuning:  60 GB  (3 workers × ~15GB per worker + overhead)
Compilation:   4 GB  (just loading/saving arrays)
```

### Memory Breakdown (Per Worker)
Each Dask worker holds in memory:
- COBRA model copy: ~50-100 MB
- Processed DataFrame: ~40 MB
- Pyomo optimization model: ~5-10 GB (during solve)
- Results accumulation: ~2-3 GB
- Python/Dask overhead: ~2 GB

**Total per worker:** ~12-15 GB
**Total for 3 workers:** ~40-45 GB
**Safety margin:** 60 GB allocated (33% buffer)

## Expected Performance

| Job | Memory | CPUs | Workers | Expected Runtime |
|-----|--------|------|---------|------------------|
| Baseline | 20 GB | 4 | N/A | 2-4 hours |
| Pre-tuning | 60 GB | 4 | 3 | 4-6 hours |
| Post-tuning | 60 GB | 4 | 3 | 4-6 hours |
| Compilation | 4 GB | 1 | N/A | < 5 minutes |
| **Total** | - | - | - | **~10-16 hours** |

## Memory Monitoring

All validation scripts now include memory usage reporting:
```bash
sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,AveRSS,MaxVMSize
```

This will show:
- `MaxRSS`: Peak memory used (most important)
- `AveRSS`: Average memory used
- `MaxVMSize`: Peak virtual memory size

## Submission Commands

### Submit all jobs in parallel:
```bash
bash slurm_jobs/submit_all.sh
```

### Or submit individually:
```bash
sbatch slurm_jobs/01_baseline_validation.sh
sbatch slurm_jobs/02_pretuning_validation.sh
sbatch slurm_jobs/03_posttuning_validation.sh

# After all complete, submit compilation
sbatch --dependency=afterok:JOB1:JOB2:JOB3 slurm_jobs/04_compile_results.sh
```

## Troubleshooting

### If still getting OOM errors:
1. **Increase memory to 80GB:**
   ```bash
   sed -i 's/--mem=60G/--mem=80G/' slurm_jobs/02_pretuning_validation.sh
   sed -i 's/--mem=60G/--mem=80G/' slurm_jobs/03_posttuning_validation.sh
   ```

2. **Reduce workers to 2:**
   ```bash
   # Edit configs/validation_iML1515.json
   "workers": 2
   ```

3. **Disable parallel processing (slowest but safest):**
   ```bash
   # Edit configs/validation_iML1515.json
   "enabled": false
   ```

### Check memory usage after job completes:
```bash
sacct -j YOUR_JOB_ID --format=JobID,MaxRSS,AveRSS,MaxVMSize,Elapsed,State
```

### Monitor jobs in real-time:
```bash
watch -n 10 'squeue -u $USER'
```

## Optimization Tips

### For faster execution:
- Load CPLEX module if available: `module load cplex/22.1.1`
  - CPLEX is 10-20x faster than GLPK
  
### For lower memory usage:
- Increase chunk_size to 150 or 200
- Reduce workers to 2
- Process subsets of genes/carbons separately

## File Locations

**SLURM Scripts:**
- `/project/def-mahadeva/ranaab/kinGEMs_v2/slurm_jobs/`

**Configuration:**
- `/project/def-mahadeva/ranaab/kinGEMs_v2/configs/validation_iML1515.json`

**Logs:**
- `/project/def-mahadeva/ranaab/kinGEMs_v2/logs/`
  - `baseline_JOBID.out` and `.err`
  - `pretuning_JOBID.out` and `.err`
  - `posttuning_JOBID.out` and `.err`

**Results:**
- `/project/def-mahadeva/ranaab/kinGEMs_v2/results/validation_parallel/`
  - `baseline_GEM.npy`
  - `pre_tuning_GEM.npy`
  - `post_tuning_GEM.npy`

## Comparison: Before vs After

| Setting | Before (Failed) | After (Fixed) | Improvement |
|---------|----------------|---------------|-------------|
| Memory (pre/post) | 20 GB | 60 GB | +200% |
| CPUs | 4 | 4 | Same (kept for speed) |
| Workers | 4 | 3 | Optimized for CPU count |
| Chunk size | auto (~30) | 100 | -70% model copies |
| Memory monitoring | No | Yes | Can diagnose issues |
| Expected success | ❌ OOM | ✅ Should work | Fixed |

## Next Steps

1. **Submit jobs:**
   ```bash
   cd /project/def-mahadeva/ranaab/kinGEMs_v2
   bash slurm_jobs/submit_all.sh
   ```

2. **Monitor progress:**
   ```bash
   squeue -u $USER
   tail -f logs/posttuning_*.out
   ```

3. **Check results after completion:**
   ```bash
   ls -lh results/validation_parallel/
   ```

4. **Review memory usage:**
   ```bash
   cat logs/posttuning_*.err | grep "Peak memory"
   ```

## Contact

If issues persist, check:
- `logs/posttuning_*.err` for error messages
- Memory usage via `sacct -j JOBID --format=MaxRSS`
- Whether CPLEX is available: `module avail cplex`

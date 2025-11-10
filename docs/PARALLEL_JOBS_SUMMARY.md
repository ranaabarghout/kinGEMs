# Parallel Validation Job System - Quick Reference

## What Was Created

This document summarizes the parallel validation job infrastructure created for running kinGEMs validation across multiple SLURM cluster nodes simultaneously.

## Created Files

### 1. SLURM Job Scripts (`slurm_jobs/`)

#### `01_baseline_validation.sh`
- Runs baseline GEM validation (no enzyme constraints)
- Resources: 4 CPUs, 8GB RAM, 3 hours
- Calls: `run_validation_parallel.py --mode baseline`

#### `02_pretuning_validation.sh`
- Runs pre-tuning kinGEMs validation (initial kcat)
- Resources: 4 CPUs, 12GB RAM, 10 hours
- Calls: `run_validation_parallel.py --mode pretuning`

#### `03_posttuning_validation.sh`
- Runs post-tuning kinGEMs validation (tuned kcat)
- Resources: 4 CPUs, 12GB RAM, 10 hours
- Calls: `run_validation_parallel.py --mode posttuning`

#### `04_compile_results.sh`
- Compiles and analyzes all validation results
- Resources: 1 CPU, 4GB RAM, 30 minutes
- Calls: `compile_validation_results.py`
- Dependency: Waits for jobs 1, 2, 3 to complete

#### `submit_all.sh`
- Master submission script
- Submits all 4 jobs with proper dependencies
- Provides monitoring commands and job tracking

### 2. Python Scripts (`scripts/`)

#### `run_validation_parallel.py`
- Mode-based validation runner (baseline/pretuning/posttuning)
- Wraps `validation_utils.py` functions
- Saves mode-specific results to separate files
- **Key features**:
  - Loads config from JSON
  - Auto-detects solver (cplex → gurobi → scip → glpk)
  - Supports parallel and sequential execution
  - Saves metadata for each run
  - Handles skip-baseline logic

#### `compile_validation_results.py`
- Results aggregation and analysis script
- **Calculates**:
  - Pearson correlation
  - Spearman correlation
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
- **Generates**:
  - `validation_metrics.csv` - Metrics table
  - `validation_improvements.csv` - Improvement analysis
  - `validation_comparison.png` - Scatter plots
  - `compiled_metadata.json` - Combined metadata

### 3. Documentation (`docs/`)

#### `PARALLEL_VALIDATION_GUIDE.md`
- Comprehensive guide for using parallel validation
- **Sections**:
  - Overview and architecture
  - Job components
  - Quick start instructions
  - Expected timeline
  - Output file structure
  - Manual job submission
  - Email notifications
  - Troubleshooting
  - Performance optimization
  - Advanced usage
  - Comparison: parallel vs sequential

## Usage

### Quick Start

1. **Configure email** (edit all SLURM scripts):
   ```bash
   #SBATCH --mail-user=your.email@example.com
   ```

2. **Submit all jobs**:
   ```bash
   bash slurm_jobs/submit_all.sh
   ```

3. **Monitor progress**:
   ```bash
   squeue -u $USER
   tail -f logs/baseline_<JOB_ID>.out
   ```

### Manual Submission

```bash
# Submit validation jobs
JOB1=$(sbatch slurm_jobs/01_baseline_validation.sh | awk '{print $4}')
JOB2=$(sbatch slurm_jobs/02_pretuning_validation.sh | awk '{print $4}')
JOB3=$(sbatch slurm_jobs/03_posttuning_validation.sh | awk '{print $4}')

# Submit compilation job (runs after all complete)
sbatch --dependency=afterok:$JOB1:$JOB2:$JOB3 slurm_jobs/04_compile_results.sh
```

## Output Structure

```
results/
├── validation_parallel/          # Individual job results
│   ├── baseline_GEM.npy
│   ├── pre_tuning_GEM.npy
│   ├── post_tuning_GEM.npy
│   ├── experimental_fitness.npy
│   ├── *_metadata.json
│   └── job_ids.txt
└── validation_compiled/          # Compiled analysis
    ├── validation_metrics.csv
    ├── validation_improvements.csv
    ├── validation_comparison.png
    └── compiled_metadata.json

logs/
├── baseline_<JOB_ID>.out
├── baseline_<JOB_ID>.err
├── pretuning_<JOB_ID>.out
├── pretuning_<JOB_ID>.err
├── posttuning_<JOB_ID>.out
├── posttuning_<JOB_ID>.err
├── compile_<JOB_ID>.out
└── compile_<JOB_ID>.err
```

## Key Features

### Parallel Execution
- 3 validation jobs run **simultaneously** on different nodes
- Total time: **~10 hours** (vs ~23 hours sequential)
- **2.3× speedup**

### Job Dependencies
- Compilation job waits for ALL validation jobs to complete
- Uses SLURM `--dependency=afterok:JOB1:JOB2:JOB3`
- Automatic failure handling

### Email Notifications
- Sent when jobs start, complete, or fail
- Configure with `#SBATCH --mail-user=your.email@example.com`

### Resource Allocation
- Baseline: 4 CPUs, 8GB RAM (fast, ~3h)
- Pre-tuning: 4 CPUs, 12GB RAM (slow, ~10h)
- Post-tuning: 4 CPUs, 12GB RAM (slow, ~10h)
- Compilation: 1 CPU, 4GB RAM (fast, ~30min)

### Solver Optimization
- Auto-detects best available solver
- CPLEX provides **10-20× speedup** over GLPK
- Optional CPLEX module load in scripts (commented)

## Workflow Diagram

```
         submit_all.sh
              │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
┌────────┐        ┌────────────┐
│ Job 1  │        │  Job 2     │
│Baseline│        │Pre-tuning  │
│ 3 hrs  │        │ 10 hrs     │
└───┬────┘        └─────┬──────┘
    │                   │
    │             ┌────────────┐
    │             │  Job 3     │
    │             │Post-tuning │
    │             │ 10 hrs     │
    │             └─────┬──────┘
    │                   │
    └──────┬────────────┘
           │
           ▼
    ┌──────────────┐
    │   Job 4      │
    │ Compilation  │
    │ 30 mins      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │Final Results │
    └──────────────┘
```

## Timeline

| Time | Event |
|------|-------|
| T+0h | All 3 validation jobs start in parallel |
| T+3h | Baseline job completes (fastest) |
| T+10h | Pre-tuning and post-tuning complete |
| T+10h | Compilation job starts automatically |
| T+10.5h | Final results ready in `results/validation_compiled/` |

## Next Steps

1. ✅ **Configure email** in all SLURM scripts
2. ✅ **Test submission** with `bash slurm_jobs/submit_all.sh`
3. ✅ **Monitor jobs** with `squeue -u $USER`
4. ✅ **Check results** in `results/validation_compiled/`

## Performance Tips

1. **Use CPLEX**: Uncomment `module load cplex` in scripts for 10-20× speedup
2. **Match workers to CPUs**: Ensure `parallel.workers` in config matches `--cpus-per-task`
3. **Monitor memory**: Increase `--mem` if jobs fail with out-of-memory errors
4. **Check cluster load**: Run `sinfo` to check node availability

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Job pending | Check `sinfo` for cluster availability |
| Job fails immediately | Check `.err` log file for Python errors |
| Out of memory | Increase `--mem` in SLURM script |
| Timeout | Increase `--time` parameter |
| Compilation fails | Run `compile_validation_results.py` manually |

## Documentation

- **Full guide**: `docs/PARALLEL_VALIDATION_GUIDE.md`
- **Dask debugging**: `docs/DASK_TIMEOUT_DEBUG.md`
- **Solver performance**: `docs/SOLVER_PERFORMANCE.md`
- **General troubleshooting**: `docs/TROUBLESHOOTING_PARALLEL.md`
- **Skip baseline feature**: `RUN_ENZYME_ONLY.md`

## Summary

This parallel validation system provides:
- ✅ **2.3× faster** validation (10h vs 23h)
- ✅ **Automatic** job dependency management
- ✅ **Email** notifications for all job events
- ✅ **Comprehensive** analysis and visualization
- ✅ **Robust** error handling and logging
- ✅ **Scalable** to different cluster configurations

Ready to use! 🚀

# Parallel Validation Job Submission Guide

This guide explains how to run kinGEMs validation in parallel across multiple SLURM jobs on Compute Canada clusters.

## Overview

Instead of running validation sequentially (baseline → pre-tuning → post-tuning = ~23 hours total), this approach runs all three components **in parallel on separate compute nodes**, reducing total time to **~10 hours** (limited by the slowest job).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Master Submission Script                  │
│                (submit_all.sh)                           │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
        ▼                            ▼
┌──────────────┐            ┌──────────────┐
│  Baseline    │            │  Pre-tuning  │
│  Job #1      │            │  Job #2      │
│  (~3 hours)  │            │  (~10 hours) │
└──────┬───────┘            └──────┬───────┘
       │                           │
       │                           │
       │                    ┌──────────────┐
       │                    │ Post-tuning  │
       │                    │  Job #3      │
       │                    │  (~10 hours) │
       │                    └──────┬───────┘
       │                           │
       └───────────┬───────────────┘
                   │
                   ▼
          ┌────────────────┐
          │  Compilation   │
          │  Job #4        │
          │  (~30 minutes) │
          └────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Final Results  │
          └────────────────┘
```

## Job Components

### 1. Baseline Validation (`01_baseline_validation.sh`)
- **Purpose**: Validate baseline GEM without enzyme constraints
- **Resources**: 4 CPUs, 8GB RAM, 3 hours
- **Output**: `results/validation_parallel/baseline_GEM.npy`

### 2. Pre-tuning Validation (`02_pretuning_validation.sh`)
- **Purpose**: Validate kinGEMs with initial kcat values (before tuning)
- **Resources**: 4 CPUs, 12GB RAM, 10 hours
- **Output**: `results/validation_parallel/pre_tuning_GEM.npy`

### 3. Post-tuning Validation (`03_posttuning_validation.sh`)
- **Purpose**: Validate kinGEMs with tuned kcat values (after optimization)
- **Resources**: 4 CPUs, 12GB RAM, 10 hours
- **Output**: `results/validation_parallel/post_tuning_GEM.npy`

### 4. Results Compilation (`04_compile_results.sh`)
- **Purpose**: Merge results and generate comparative analysis
- **Resources**: 1 CPU, 4GB RAM, 30 minutes
- **Dependency**: Runs **only after** all three validation jobs complete successfully
- **Outputs**:
  - `results/validation_compiled/validation_metrics.csv`
  - `results/validation_compiled/validation_improvements.csv`
  - `results/validation_compiled/validation_comparison.png`
  - `results/validation_compiled/compiled_metadata.json`

## Quick Start

### Step 1: Configure Email Notifications

Edit each SLURM script (`slurm_jobs/01_*.sh`, `02_*.sh`, `03_*.sh`, `04_*.sh`) and replace:

```bash
#SBATCH --mail-user=your.email@example.com
```

with your actual email address.

### Step 2: Submit All Jobs

From the project root directory:

```bash
bash slurm_jobs/submit_all.sh
```

This will:
1. Submit all three validation jobs in parallel
2. Submit the compilation job with dependency on validation jobs
3. Display job IDs and monitoring commands
4. Save job information to `results/validation_parallel/job_ids.txt`

### Step 3: Monitor Progress

Check job status:
```bash
squeue -u $USER
```

View live output:
```bash
tail -f logs/baseline_<JOB_ID>.out
tail -f logs/pretuning_<JOB_ID>.out
tail -f logs/posttuning_<JOB_ID>.out
```

Check specific jobs:
```bash
squeue -j JOB1,JOB2,JOB3,JOB4
```

## Expected Timeline

- **T+0h**: All validation jobs start
- **T+3h**: Baseline job completes (fastest)
- **T+10h**: Pre-tuning and post-tuning jobs complete
- **T+10h**: Compilation job starts automatically
- **T+10.5h**: Final results ready

**Total time: ~10.5 hours** (vs ~23 hours sequential)

## Output Files

### Validation Outputs (`results/validation_parallel/`)
```
results/validation_parallel/
├── baseline_GEM.npy                # Baseline predictions
├── pre_tuning_GEM.npy             # Pre-tuning predictions
├── post_tuning_GEM.npy            # Post-tuning predictions
├── experimental_fitness.npy        # Experimental data
├── baseline_metadata.json          # Baseline job metadata
├── pretuning_metadata.json         # Pre-tuning job metadata
├── posttuning_metadata.json        # Post-tuning job metadata
└── job_ids.txt                     # Job IDs and monitoring info
```

### Compiled Results (`results/validation_compiled/`)
```
results/validation_compiled/
├── validation_metrics.csv          # Correlation metrics (Pearson, Spearman, RMSE, MAE)
├── validation_improvements.csv     # Improvement analysis
├── validation_comparison.png       # Scatter plots comparing all three
└── compiled_metadata.json          # Combined metadata
```

### Log Files (`logs/`)
```
logs/
├── baseline_<JOB_ID>.out          # Baseline stdout
├── baseline_<JOB_ID>.err          # Baseline stderr
├── pretuning_<JOB_ID>.out         # Pre-tuning stdout
├── pretuning_<JOB_ID>.err         # Pre-tuning stderr
├── posttuning_<JOB_ID>.out        # Post-tuning stdout
├── posttuning_<JOB_ID>.err        # Post-tuning stderr
├── compile_<JOB_ID>.out           # Compilation stdout
└── compile_<JOB_ID>.err           # Compilation stderr
```

## Manual Job Submission

If you prefer to submit jobs individually:

```bash
# Submit validation jobs
JOB1=$(sbatch slurm_jobs/01_baseline_validation.sh | awk '{print $4}')
JOB2=$(sbatch slurm_jobs/02_pretuning_validation.sh | awk '{print $4}')
JOB3=$(sbatch slurm_jobs/03_posttuning_validation.sh | awk '{print $4}')

# Submit compilation job (runs after all validation jobs complete)
sbatch --dependency=afterok:$JOB1:$JOB2:$JOB3 slurm_jobs/04_compile_results.sh
```

## Canceling Jobs

Cancel all jobs:
```bash
scancel JOB1 JOB2 JOB3 JOB4
```

Cancel specific job:
```bash
scancel JOB1
```

## Email Notifications

You will receive emails when:
- ✉️ Each job **starts**
- ✅ Each job **completes successfully**
- ❌ Any job **fails**

## Troubleshooting

### Job Fails Immediately
Check the log file for errors:
```bash
cat logs/baseline_<JOB_ID>.err
```

Common issues:
- Virtual environment not activated
- Missing Python packages
- Configuration file not found
- Model files not accessible

### Out of Memory
If a job runs out of memory, increase the `--mem` parameter in the SLURM script:
```bash
#SBATCH --mem=16G  # Increase from 8G or 12G
```

### Timeout
If a job times out, increase the `--time` parameter:
```bash
#SBATCH --time=0-15:00:00  # Increase from 10 hours
```

### Job Pending for Long Time
Check cluster status:
```bash
sinfo
```

Your job may be waiting for resources. You can check why:
```bash
scontrol show job JOB_ID
```

### Compilation Job Fails
If validation jobs succeed but compilation fails, you can run compilation manually:
```bash
python scripts/compile_validation_results.py \
    --input results/validation_parallel \
    --output results/validation_compiled
```

## Performance Optimization

### Using CPLEX (Recommended)

If CPLEX is available on your cluster, uncomment the module load in the SLURM scripts:

```bash
# Load CPLEX for faster optimization (10-20x speedup)
module load cplex
```

This reduces runtime from ~10 hours to **~30 minutes** for enzyme-constrained validation!

### Adjusting Workers

Edit `configs/validation_iML1515.json`:

```json
{
  "parallel": {
    "workers": 4  // Match your --cpus-per-task
  }
}
```

Then update SLURM scripts:
```bash
#SBATCH --cpus-per-task=8  // Increase from 4
```

## Advanced Usage

### Running Only Specific Jobs

Run only baseline:
```bash
sbatch slurm_jobs/01_baseline_validation.sh
```

Run only pre-tuning:
```bash
sbatch slurm_jobs/02_pretuning_validation.sh
```

Run only post-tuning:
```bash
sbatch slurm_jobs/03_posttuning_validation.sh
```

### Using Different Configuration

Pass custom config to individual jobs by editing the SLURM script:
```bash
python scripts/run_validation_parallel.py \
    --mode baseline \
    --config configs/custom_config.json \
    --output results/custom_output
```

### Analyzing Results Separately

Load results in Python:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load results
baseline = np.load('results/validation_parallel/baseline_GEM.npy')
pretuning = np.load('results/validation_parallel/pre_tuning_GEM.npy')
posttuning = np.load('results/validation_parallel/post_tuning_GEM.npy')
experimental = np.load('results/validation_parallel/experimental_fitness.npy')

# Calculate custom metrics
pearson_baseline, _ = pearsonr(experimental.flatten(), baseline.flatten())
pearson_pretuning, _ = pearsonr(experimental.flatten(), pretuning.flatten())
pearson_posttuning, _ = pearsonr(experimental.flatten(), posttuning.flatten())

print(f"Baseline:     r = {pearson_baseline:.4f}")
print(f"Pre-tuning:   r = {pearson_pretuning:.4f}")
print(f"Post-tuning:  r = {pearson_posttuning:.4f}")
```

## Comparison: Parallel vs Sequential

| Approach | Time | Resource Efficiency | Complexity |
|----------|------|---------------------|------------|
| **Sequential** | ~23 hours | Uses 1 node efficiently | Simple |
| **Parallel** | ~10 hours | Uses 3 nodes simultaneously | Moderate |

**Recommendation**: Use parallel approach when:
- You have access to multiple compute nodes
- Results are needed quickly
- Cluster is not heavily loaded

Use sequential approach when:
- Cluster resources are limited
- No rush for results
- Simplicity is preferred

## Support

For issues or questions:
1. Check log files in `logs/`
2. Review error messages in `.err` files
3. Consult the main `TROUBLESHOOTING_PARALLEL.md` documentation
4. Check SLURM job status with `scontrol show job JOB_ID`

## Summary

This parallel validation system allows you to:
- ✅ Run validation **2.3× faster** (10h vs 23h)
- ✅ Utilize cluster resources efficiently
- ✅ Get **automatic email notifications**
- ✅ Generate **comprehensive comparative analysis**
- ✅ Monitor progress in **real-time**

Happy validating! 🚀

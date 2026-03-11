#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for Tuning Reproducibility Study
# Runs simulated annealing N times from the same initial state and
# generates comparison plots to assess kcat optimization consistency.
# ---------------------------------------------------------------------
#SBATCH --job-name=kGEMs_repro
#SBATCH --account=def-mahadeva
#SBATCH --cpus-per-task=4
#SBATCH --time=0-24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/reproducibility_%j.out
#SBATCH --error=logs/reproducibility_%j.err
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# ---------------------------------------------------------------------
echo "========================================="
echo "Job: Tuning Reproducibility Study"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo "========================================="
# ---------------------------------------------------------------------

# Load required modules
module load python/3.11

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# ---- Parameters (override via sbatch --export or positional args) ----
# Usage:
#   sbatch slurm_jobs/run_tuning_reproducibility.sh
#   sbatch slurm_jobs/run_tuning_reproducibility.sh configs/iML1515_GEM.json 10 0
CONFIG_FILE=${1:-configs/iML1515_GEM.json}
N_RUNS=${2:-10}
BASE_SEED=${3:-0}

echo ""
echo "Configuration : $CONFIG_FILE"
echo "Number of runs: $N_RUNS"
echo "Base seed     : $BASE_SEED"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo ""

# Check that config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Run the reproducibility study
echo "Running tuning reproducibility study..."
python scripts/run_tuning_reproducibility.py \
    "$CONFIG_FILE" \
    --n-runs "$N_RUNS" \
    --base-seed "$BASE_SEED"

exitcode=$?

# Memory usage statistics
echo ""
echo "Memory Usage Statistics:"
if command -v sacct &> /dev/null; then
    sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,AveRSS,MaxVMSize --noheader
    PEAK_MEM=$(sacct -j $SLURM_JOB_ID --format=MaxRSS --noheader | sort -n | tail -1)
    echo "Peak memory usage: $PEAK_MEM"
else
    echo "  sacct not available - memory stats unavailable"
fi

# Show output directory size
echo ""
echo "Results disk usage:"
if [ -d "results/reproducibility" ]; then
    du -sh results/reproducibility/* 2>/dev/null | tail -5
else
    echo "  No reproducibility results directory found"
fi

# ---------------------------------------------------------------------
echo "========================================="
echo "Finished at: `date`"
echo "Exit code: $exitcode"
echo "========================================="

exit $exitcode

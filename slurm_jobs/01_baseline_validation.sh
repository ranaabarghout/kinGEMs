#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for Baseline GEM validation
# ---------------------------------------------------------------------
#SBATCH --job-name=baseline_GEM
#SBATCH --account=def-mahadeva
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# ---------------------------------------------------------------------
echo "========================================="
echo "Job: Baseline GEM Validation"
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
mkdir -p logs results/validation_parallel

# Run baseline validation only
echo "Running Baseline GEM validation..."
python scripts/run_validation_parallel.py \
    --mode baseline \
    --config configs/validation_iML1515.json \
    --output results/validation_parallel

exitcode=$?

# Monitor memory usage
echo ""
echo "Memory Usage Statistics:"
if command -v sacct &> /dev/null; then
    sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,AveRSS,MaxVMSize --noheader
    PEAK_MEM=$(sacct -j $SLURM_JOB_ID --format=MaxRSS --noheader | sort -n | tail -1)
    echo "Peak memory usage: $PEAK_MEM"
else
    echo "  sacct not available - memory stats unavailable"
fi

# ---------------------------------------------------------------------
echo "========================================="
echo "Finished at: `date`"
echo "Exit code: $exitcode"
echo "========================================="

exit $exitcode

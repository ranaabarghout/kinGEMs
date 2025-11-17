#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for kinGEMs General Pipeline
# ---------------------------------------------------------------------
#SBATCH --job-name=kinGEMs_pipeline
#SBATCH --account=def-mahadeva
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# ---------------------------------------------------------------------
echo "========================================="
echo "Job: kinGEMs General Pipeline"
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

# Configuration file - default to the parallel FVA config
CONFIG_FILE=${1:-configs/iML1515_GEM_parallel_fva.json}

echo ""
echo "Configuration: $CONFIG_FILE"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Run kinGEMs pipeline with logging wrapper
echo "Running kinGEMs pipeline with logging..."
./scripts/run_pipeline_with_logging.sh "$CONFIG_FILE"

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

# Show disk usage of results
echo ""
echo "Results disk usage:"
if [ -d "results" ]; then
    du -sh results/tuning_results/* 2>/dev/null | tail -5
else
    echo "  No results directory found"
fi

# ---------------------------------------------------------------------
echo "========================================="
echo "Finished at: `date`"
echo "Exit code: $exitcode"
echo "========================================="

exit $exitcode

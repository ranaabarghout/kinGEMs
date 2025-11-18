#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for kinGEMs General Pipeline (Direct)
# ---------------------------------------------------------------------
#SBATCH --job-name=kinGEMs_direct
#SBATCH --account=def-mahadeva
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/pipeline_direct_%j.out
#SBATCH --error=logs/pipeline_direct_%j.err
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# ---------------------------------------------------------------------
echo "========================================="
echo "Job: kinGEMs General Pipeline (Direct)"
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

# Update config to use the number of SLURM CPUs for FVA workers
# This creates a temporary config with the SLURM CPU count
TEMP_CONFIG=$(mktemp --suffix=.json)
python3 -c "
import json
import sys
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
if 'fva' in config:
    config['fva']['workers'] = $SLURM_CPUS_PER_TASK
with open('$TEMP_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
"

echo "Updated FVA workers to match SLURM CPUs: $SLURM_CPUS_PER_TASK"
echo "Using temporary config: $TEMP_CONFIG"
echo ""

# Run kinGEMs pipeline directly
echo "Running kinGEMs pipeline..."
python scripts/run_pipeline.py "$TEMP_CONFIG"

exitcode=$?

# Clean up temporary config
rm -f "$TEMP_CONFIG"

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

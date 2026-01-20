#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for FVA Ablation Study
# ---------------------------------------------------------------------
#SBATCH --job-name=fva_ablation
#SBATCH --account=def-mahadeva
#SBATCH --cpus-per-task=8
#SBATCH --time=0-30:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/fva_ablation_%j.out
#SBATCH --error=logs/fva_ablation_%j.err
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# ---------------------------------------------------------------------
echo "========================================="
echo "Job: FVA Ablation Study"
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

# Configuration file
CONFIG_FILE=${1:-configs/iML1515_GEM_fva_ablation.json}

echo ""
echo "Configuration: $CONFIG_FILE"
echo "Workers: $SLURM_CPUS_PER_TASK"
echo ""

# Run FVA ablation study
echo "Running FVA ablation study..."
python scripts/run_fva_ablation.py \
    "$CONFIG_FILE" \
    --parallel \
    --workers $SLURM_CPUS_PER_TASK

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

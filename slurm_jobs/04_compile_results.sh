#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for compiling validation results from parallel jobs
# This job should run AFTER all validation jobs complete
# ---------------------------------------------------------------------
#SBATCH --job-name=compile_validation
#SBATCH --account=def-mahadeva
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:30:00
#SBATCH --mem=4G
#SBATCH --output=logs/compile_%j.out
#SBATCH --error=logs/compile_%j.err
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# Dependency: Run after the other jobs complete
# Use with: sbatch --dependency=afterok:JOB1:JOB2:JOB3 04_compile_results.sh
# ---------------------------------------------------------------------
echo "========================================="
echo "Job: Compile Validation Results"
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

# Compile results from all validation runs
echo "Compiling validation results..."
python scripts/compile_validation_results.py \
    --input results/validation_parallel \
    --output results/validation_compiled

exitcode=$?

# ---------------------------------------------------------------------
echo "========================================="
echo "Finished at: `date`"
echo "Exit code: $exitcode"
echo "========================================="

exit $exitcode

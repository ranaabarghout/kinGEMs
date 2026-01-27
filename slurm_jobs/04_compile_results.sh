#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for compiling validation results from parallel jobs
# This job should run AFTER all validation jobs complete
#
# Supports both standard and extended enzyme allocation validation runs
# Automatically detects and processes:
# - Baseline results
# - Pre-tuning results (standard and enzyme variants)
# - Post-tuning results (standard and enzyme variants)
#
# Dependencies: All validation jobs must complete first
# Use with: sbatch --dependency=afterok:JOB1:JOB2:JOB3:... 04_compile_results.sh
# ---------------------------------------------------------------------
#SBATCH --job-name=compile_validation
#SBATCH --account=def-mahadeva
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/compile_%j.out
#SBATCH --error=logs/compile_%j.err
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# Dependency: Run after the other jobs complete
# Use with: sbatch --dependency=afterok:JOB1:JOB2:JOB3:... 04_compile_results.sh
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
echo ""
echo "Compiling validation results..."
echo "  Input: results/validation_parallel"
echo "  Output: results/validation_compiled"
echo ""
echo "This script will automatically detect and process:"
echo "  ✓ Baseline results"
echo "  ✓ Pre-tuning results (if available: standard, enzyme_removed, enzyme_kept)"
echo "  ✓ Post-tuning results (if available: standard, enzyme_removed, enzyme_kept)"
echo ""

python scripts/compile_validation_results.py \
    --input results/validation_parallel \
    --output results/validation_compiled

exitcode=$?

echo ""
echo "========================================="

if [ $exitcode -eq 0 ]; then
    echo "✓ Compilation completed successfully"
else
    echo "!!! Compilation failed with exit code: $exitcode"
fi

echo "========================================="
echo "Finished at: `date`"
echo "Exit code: $exitcode"
echo "========================================="

# List generated files
if [ $exitcode -eq 0 ]; then
    echo ""
    echo "Generated output files:"
    echo "  - Correlation metrics:"
    ls -lh results/validation_compiled/validation_metrics*.csv 2>/dev/null | awk '{print "     " $NF}'
    echo ""
    echo "  - Visualization plots:"
    ls -lh results/validation_compiled/*.png 2>/dev/null | awk '{print "     " $NF}'
    echo ""
    echo "  - Summary tables:"
    ls -lh results/validation_compiled/*summary*.csv 2>/dev/null | awk '{print "     " $NF}'
    echo ""
    echo "  - Results directory: results/validation_compiled/"
fi

exit $exitcode


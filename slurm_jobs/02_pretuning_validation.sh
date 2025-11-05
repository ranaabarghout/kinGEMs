#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for Pre-tuning kinGEMs validation (enzyme-constrained)
# ---------------------------------------------------------------------
#SBATCH --job-name=pretuning_kinGEMs
#SBATCH --account=def-mahadeva
#SBATCH --cpus-per-task=4
#SBATCH --time=0-15:00:00
#SBATCH --mem=60G
#SBATCH --output=/project/def-mahadeva/ranaab/kinGEMs_v2/logs/pretuning_%j.out
#SBATCH --error=/project/def-mahadeva/ranaab/kinGEMs_v2/logs/pretuning_%j.err
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# ---------------------------------------------------------------------

# Change to project directory
cd /project/def-mahadeva/ranaab/kinGEMs_v2 || exit 1

echo "========================================="
echo "Job: Pre-tuning kinGEMs Validation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo "========================================="
# ---------------------------------------------------------------------

# Load required modules
module load python/3.11

# Load personal CPLEX module for 40x faster optimization
export MODULEPATH=$HOME/modulefiles:$MODULEPATH
module load mycplex/22.1.1

# Verify CPLEX is accessible and set environment variables for Pyomo
echo "Checking CPLEX availability..."
export CPLEX_STUDIO_DIR=$HOME/cplex_studio2211
export PATH=$HOME/cplex_studio2211/cplex/bin/x86-64_linux:$PATH
export LD_LIBRARY_PATH=$HOME/cplex_studio2211/cplex/bin/x86-64_linux:$LD_LIBRARY_PATH

which cplex && echo "✓ CPLEX found: $(which cplex)" || echo "⚠️  CPLEX not found in PATH"
echo "CPLEX_STUDIO_DIR: $CPLEX_STUDIO_DIR"

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs results/validation_parallel

# Run pre-tuning validation only
echo "Running Pre-tuning kinGEMs validation..."
python scripts/run_validation_parallel.py \
    --mode pretuning \
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

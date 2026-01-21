#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for kinGEMs Fluxomics Validation Pipeline
# ---------------------------------------------------------------------
#SBATCH --job-name=kinGEMs_fluxomics_validation
#SBATCH --account=def-mahadeva
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/fluxomics_pipeline_%j.out
#SBATCH --error=logs/fluxomics_pipeline_%j.err
#SBATCH --mail-user=lya.chinas@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL
# ---------------------------------------------------------------------
echo "========================================="
echo "Job: kinGEMs Fluxomics Validation Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo "========================================="
# ---------------------------------------------------------------------

# Load required modules
module load python/3.11

# Activate virtual environment
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Input arguments:
#   Case 1 (config): run_fluxomics_pipeline.sh <config.json> [experimental.csv]
#   Case 2 (CSV-only): run_fluxomics_pipeline.sh <experimental.csv> <fva_results.csv> [more_fva_results.csv ...]
INPUT_1="$1"
INPUT_2="$2"

if [ -z "$INPUT_1" ]; then
    echo "Error: No arguments provided."
    echo "Usage:"
    echo "  $0 <config.json> [experimental.csv]"
    echo "  $0 <experimental.csv> <fva_results.csv> [more_fva_results.csv ...]"
    exit 1
fi

echo ""
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo ""

if [[ "$INPUT_1" == *.json ]]; then
    CONFIG_FILE="$INPUT_1"
    EXP_FILE="$INPUT_2"

    echo "Configuration: $CONFIG_FILE"
    if [ -n "$EXP_FILE" ]; then
        echo "Experimental data: $EXP_FILE"
    fi

    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Configuration file '$CONFIG_FILE' not found!"
        exit 1
    fi

    # Update config to use the number of SLURM CPUs for FVA workers
    # This creates a temporary config with the SLURM CPU count
    TEMP_CONFIG=$(mktemp --suffix=.json)
    
    python3 - <<EOF
import json
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = json.load(f)
    
    if 'fva' in config:
        config['fva']['workers'] = $SLURM_CPUS_PER_TASK
    else:
        # Ensure the key exists if you want to force worker count
        config['fva'] = {'workers': $SLURM_CPUS_PER_TASK}

    with open('$TEMP_CONFIG', 'w') as f:
        json.dump(config, f, indent=2)
except Exception as e:
    print(f"Error updating temp config: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    # Run fluxomics validation (config mode)
    echo "Running fluxomics validation (config mode)..."
    if [ -n "$EXP_FILE" ]; then
        python scripts/run_fluxomics_validation.py "$TEMP_CONFIG" "$EXP_FILE"
    else
        python scripts/run_fluxomics_validation.py "$TEMP_CONFIG"
    fi

    exitcode=$?
    rm -f "$TEMP_CONFIG"
else
    EXP_FILE="$INPUT_1"
    shift
    FVA_FILES=("$@")

    if [ ${#FVA_FILES[@]} -eq 0 ]; then
        echo "Error: CSV-only mode requires at least one FVA results file."
        exit 1
    fi

    echo "Experimental data: $EXP_FILE"
    echo "FVA results files: ${FVA_FILES[*]}"
    echo ""

    # Run fluxomics validation (CSV-only mode)
    echo "Running fluxomics validation (CSV-only mode)..."
    python scripts/run_fluxomics_validation.py "$EXP_FILE" "${FVA_FILES[@]}"
    exitcode=$?
fi

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

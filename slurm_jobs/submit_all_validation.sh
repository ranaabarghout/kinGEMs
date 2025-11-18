#!/bin/bash
#
# Master SLURM Job Submission Script for Parallel Validation
# ===========================================================
#
# This script submits all three validation jobs (baseline, pre-tuning, post-tuning)
# in parallel and schedules a compilation job to run after all complete.
#
# Usage:
#   bash slurm_jobs/submit_all_validation.sh
#
# The script will:
#   1. Submit baseline validation job (fastest, ~45 min - 1 hour)
#      - Calculates baseline wild-type growth (NO enzyme constraints)
#      - Runs gene knockout simulations for baseline GEM
#   2. Submit pre-tuning validation job (~2-3 hours)
#      - Calculates pre-tuning wild-type growth (WITH initial kcat constraints)
#      - Runs gene knockout simulations for pre-tuning kinGEMs
#   3. Submit post-tuning validation job (~2-3 hours)
#      - Calculates post-tuning wild-type growth (WITH tuned kcat constraints)
#      - Runs gene knockout simulations for post-tuning kinGEMs
#   4. Submit compilation job with dependency on all three
#      - Uses model-specific wild-type files for fitness-based correlations
#
# NOTE: Each model now calculates its own wild-type growth rates with appropriate
#       constraints, ensuring accurate fitness value calculations (log2 ratios).
#
# Email notifications will be sent for each job's start, completion, and failures.
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../configs/validation_iML1515.json"
OUTPUT_DIR="${SCRIPT_DIR}/../results/validation_parallel"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "========================================================================"
echo "=== kinGEMs Parallel Validation - Job Submission ==="
echo "========================================================================"
echo ""
echo -e "${YELLOW}IMPORTANT: Model-Specific Wild-Type Calculation${NC}"
echo "----------------------------------------------------------------------"
echo "Each model now calculates its own wild-type growth rates:"
echo "  • Baseline:    No enzyme constraints → higher wild-type growth"
echo "  • Pre-tuning:  Initial kcat constraints → constrained wild-type"
echo "  • Post-tuning: Tuned kcat constraints → optimized wild-type"
echo ""
echo "This ensures accurate fitness calculations: log2(mutant/wild-type)"
echo "----------------------------------------------------------------------"
echo ""

# Check if job scripts exist
if [ ! -f "${SCRIPT_DIR}/01_baseline_validation.sh" ]; then
    echo "❌ Error: Job script not found: 01_baseline_validation.sh"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/02_pretuning_validation.sh" ]; then
    echo "❌ Error: Job script not found: 02_pretuning_validation.sh"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/03_posttuning_validation.sh" ]; then
    echo "❌ Error: Job script not found: 03_posttuning_validation.sh"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/04_compile_results.sh" ]; then
    echo "❌ Error: Job script not found: 04_compile_results.sh"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRIPT_DIR}/../logs"

echo -e "${BLUE}Configuration:${NC}"
echo "  Config file: ${CONFIG_FILE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Job scripts: ${SCRIPT_DIR}"
echo ""

# Submit baseline validation job
echo -e "${GREEN}[1/4] Submitting baseline validation job...${NC}"
JOB1=$(sbatch "${SCRIPT_DIR}/01_baseline_validation.sh" | awk '{print $4}')
if [ -z "$JOB1" ]; then
    echo "❌ Error: Failed to submit baseline job"
    exit 1
fi
echo "  ✓ Job ID: ${JOB1}"
echo ""

# Submit pre-tuning validation job
echo -e "${GREEN}[2/4] Submitting pre-tuning validation job...${NC}"
JOB2=$(sbatch "${SCRIPT_DIR}/02_pretuning_validation.sh" | awk '{print $4}')
if [ -z "$JOB2" ]; then
    echo "❌ Error: Failed to submit pre-tuning job"
    exit 1
fi
echo "  ✓ Job ID: ${JOB2}"
echo ""

# Submit post-tuning validation job
echo -e "${GREEN}[3/4] Submitting post-tuning validation job...${NC}"
JOB3=$(sbatch "${SCRIPT_DIR}/03_posttuning_validation.sh" | awk '{print $4}')
if [ -z "$JOB3" ]; then
    echo "❌ Error: Failed to submit post-tuning job"
    exit 1
fi
echo "  ✓ Job ID: ${JOB3}"
echo ""

# Submit compilation job with dependency
echo -e "${GREEN}[4/4] Submitting compilation job (dependency on all validation jobs)...${NC}"
JOB4=$(sbatch --dependency=afterok:${JOB1}:${JOB2}:${JOB3} "${SCRIPT_DIR}/04_compile_results.sh" | awk '{print $4}')
if [ -z "$JOB4" ]; then
    echo "❌ Error: Failed to submit compilation job"
    exit 1
fi
echo "  ✓ Job ID: ${JOB4}"
echo ""

echo "========================================================================"
echo -e "${BLUE}=== Submission Summary ===${NC}"
echo "========================================================================"
echo ""
echo "Validation jobs (running in parallel):"
echo "  Baseline:     Job ${JOB1} (estimated: 1 hour)"
echo "                → Calculates baseline wild-type + gene knockouts"
echo "  Pre-tuning:   Job ${JOB2} (estimated: 2-3 hours)"
echo "                → Calculates pre-tuning wild-type + gene knockouts WITH enzyme constraints"
echo "  Post-tuning:  Job ${JOB3} (estimated: 2-3 hours)"
echo "                → Calculates post-tuning wild-type + gene knockouts WITH tuned constraints"
echo ""
echo "Compilation job (will run after all validation jobs complete):"
echo "  Compile:      Job ${JOB4}"
echo "                → Generates fitness-based correlations using model-specific wild-type files"
echo ""
echo "========================================================================"
echo -e "${YELLOW}Monitoring Commands:${NC}"
echo "========================================================================"
echo ""
echo "Check job status:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job:"
echo "  squeue -j ${JOB1},${JOB2},${JOB3},${JOB4}"
echo ""
echo "View job details:"
echo "  scontrol show job ${JOB1}"
echo ""
echo "View live output:"
echo "  tail -f logs/baseline_${JOB1}.out"
echo "  tail -f logs/pretuning_${JOB2}.out"
echo "  tail -f logs/posttuning_${JOB3}.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel ${JOB1} ${JOB2} ${JOB3} ${JOB4}"
echo ""
echo "========================================================================"
echo -e "${BLUE}Expected Timeline:${NC}"
echo "========================================================================"
echo ""
echo "  T+0h:    All validation jobs start"
echo "  T+1h:    Baseline job completes (fastest, no enzyme constraints)"
echo "  T+2-3h:  Pre-tuning and post-tuning jobs complete (enzyme-constrained)"
echo "  T+3h:    Compilation job starts automatically"
echo "  T+3.1h:  Compilation job completes, final results ready"
echo ""
echo "Output files per model:"
echo "  - baseline_GEM.npy          (gene knockout results)"
echo "  - baseline_wildtype.npy     (wild-type growth, no constraints)"
echo "  - pre_tuning_GEM.npy        (gene knockout results)"
echo "  - pretuning_wildtype.npy    (wild-type growth, initial kcats)"
echo "  - post_tuning_GEM.npy       (gene knockout results)"
echo "  - posttuning_wildtype.npy   (wild-type growth, tuned kcats)"
echo ""
echo "========================================================================"
echo -e "${GREEN}Jobs submitted successfully!${NC}"
echo "========================================================================"
echo ""
echo "Email notifications will be sent to the configured address when:"
echo "  - Each job starts"
echo "  - Each job completes"
echo "  - Any job fails"
echo ""
echo "Results will be saved to: ${OUTPUT_DIR}"
echo ""

# Save job IDs for reference
JOB_INFO_FILE="${OUTPUT_DIR}/job_ids.txt"
cat > "${JOB_INFO_FILE}" <<EOF
Parallel Validation Jobs
========================
Submitted: $(date)

Job IDs:
  Baseline:     ${JOB1}
  Pre-tuning:   ${JOB2}
  Post-tuning:  ${JOB3}
  Compilation:  ${JOB4}

Log Files:
  Baseline:     logs/baseline_${JOB1}.out
  Pre-tuning:   logs/pretuning_${JOB2}.out
  Post-tuning:  logs/posttuning_${JOB3}.out
  Compilation:  logs/compile_${JOB4}.out

Monitor Command:
  squeue -j ${JOB1},${JOB2},${JOB3},${JOB4}
EOF

echo "Job IDs saved to: ${JOB_INFO_FILE}"
echo ""

#!/bin/bash
#
# Master SLURM Job Submission Script for Parallel Validation (Extended Options)
# ==============================================================================
#
# This script submits all validation jobs with enzyme allocation options:
# 1. Baseline validation (1 job)
# 2. Pre-tuning validation with 2 enzyme allocation options (2 jobs)
# 3. Post-tuning validation with 2 enzyme allocation options (2 jobs)
# 4. Compilation job (runs after all validation completes)
#
# Enzyme Allocation Options:
# ========================
# A) ENZYME REMOVED (--remove-knockout-enzyme):
#    - When a gene is knocked out, its enzyme is removed from constraints
#    - Simulates PROTEOME REALLOCATION
#    - Freed protein pool can be used by other reactions
#    - May show growth IMPROVEMENTS due to enzyme cost reduction
#    - Useful for: Understanding optimal enzyme allocation
#
# B) ENZYME KEPT (default):
#    - When a gene is knocked out, enzyme constraints remain
#    - Simulates IMMEDIATE KNOCKOUT EFFECT
#    - Enzyme cost is locked in (wasted)
#    - Shows TRUE essential genes (growth drops to zero)
#    - Useful for: Understanding essential genes
#
# Job Submission Order:
# ====================
# Baseline (no enzyme constraints - fastest)
# Pre-tuning Enzyme Removed (2-3 hours)
# Pre-tuning Enzyme Kept (2-3 hours)
# Post-tuning Enzyme Removed (2-3 hours)
# Post-tuning Enzyme Kept (2-3 hours)
# Compile (after all validation jobs complete)
#
# Usage:
#   bash slurm_jobs/submit_all_validation.sh
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
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "============================================================================"
echo "=== kinGEMs Parallel Validation - Extended Submission (Enzyme Options) ==="
echo "============================================================================"
echo ""
echo -e "${YELLOW}Enzyme Allocation Options Explained:${NC}"
echo "-----------------------------------------------------------------------"
echo ""
echo "OPTION A: Enzyme REMOVED (Proteome Reallocation)"
echo "  • When gene knocked out, enzyme constraint is removed from model"
echo "  • Freed protein pool can be allocated to other reactions"
echo "  • May show growth IMPROVEMENTS (enzyme cost reduction)"
echo "  • Biological scenario: Cell adapts to loss by reallocating resources"
echo ""
echo "OPTION B: Enzyme KEPT (Immediate Knockout)"
echo "  • When gene knocked out, enzyme constraint remains (cost is locked in)"
echo "  • No protein reallocation possible"
echo "  • Shows TRUE essential genes (growth drops to zero)"
echo "  • Biological scenario: Immediate knockout with no adaptation"
echo ""
echo "-----------------------------------------------------------------------"
echo ""

# Check if job scripts exist
required_scripts=(
    "01_baseline_validation.sh"
    "02a_pretuning_validation_enzyme_removed.sh"
    "02b_pretuning_validation_enzyme_kept.sh"
    "03a_posttuning_validation_enzyme_removed.sh"
    "03b_posttuning_validation_enzyme_kept.sh"
    "04_compile_results.sh"
)

for script in "${required_scripts[@]}"; do
    if [ ! -f "${SCRIPT_DIR}/${script}" ]; then
        echo -e "${RED}❌ Error: Job script not found: ${script}${NC}"
        exit 1
    fi
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRIPT_DIR}/../logs"

echo -e "${BLUE}Configuration:${NC}"
echo "  Config file: ${CONFIG_FILE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Job scripts: ${SCRIPT_DIR}"
echo "  Time per job: 7 days (1 week)"
echo ""

# Job tracking variables
declare -a JOB_IDS
declare -a JOB_NAMES
declare -a JOB_DESCRIPTIONS

job_count=0

# Function to submit job and track
submit_job() {
    local script=$1
    local name=$2
    local description=$3

    echo -e "${GREEN}[$(($job_count + 1))/6] Submitting: ${name}${NC}"
    local job_id=$(sbatch "${SCRIPT_DIR}/${script}" | awk '{print $4}')

    if [ -z "$job_id" ]; then
        echo -e "${RED}❌ Error: Failed to submit ${name}${NC}"
        exit 1
    fi

    JOB_IDS+=("$job_id")
    JOB_NAMES+=("$name")
    JOB_DESCRIPTIONS+=("$description")

    echo "  ✓ Job ID: ${job_id}"
    echo "  → ${description}"
    echo ""

    job_count=$((job_count + 1))
}

# Submit all jobs
submit_job "01_baseline_validation.sh" \
    "Baseline GEM" \
    "Baseline gene knockout validation (no enzyme constraints)"

submit_job "02a_pretuning_validation_enzyme_removed.sh" \
    "Pre-tuning (Enzyme Removed)" \
    "Gene knockout with proteome reallocation (enzyme freed)"

submit_job "02b_pretuning_validation_enzyme_kept.sh" \
    "Pre-tuning (Enzyme Kept)" \
    "Gene knockout with immediate effect (enzyme locked in)"

submit_job "03a_posttuning_validation_enzyme_removed.sh" \
    "Post-tuning (Enzyme Removed)" \
    "Gene knockout with proteome reallocation (enzyme freed)"

submit_job "03b_posttuning_validation_enzyme_kept.sh" \
    "Post-tuning (Enzyme Kept)" \
    "Gene knockout with immediate effect (enzyme locked in)"

# Create dependency string for compilation job
dependency_str="${JOB_IDS[0]}"
for ((i=1; i<${#JOB_IDS[@]}; i++)); do
    dependency_str="${dependency_str}:${JOB_IDS[$i]}"
done

# Submit compilation job with dependency on all validation jobs
echo -e "${GREEN}[6/6] Submitting compilation job (dependency on all validation jobs)${NC}"
local compile_job=$(sbatch --dependency=afterok:${dependency_str} "${SCRIPT_DIR}/04_compile_results.sh" | awk '{print $4}')
if [ -z "$compile_job" ]; then
    echo -e "${RED}❌ Error: Failed to submit compilation job${NC}"
    exit 1
fi
JOB_IDS+=("$compile_job")
JOB_NAMES+=("Compile Results")
JOB_DESCRIPTIONS+=("Generate fitness-based correlations (runs after all validation jobs)")

echo "  ✓ Job ID: ${compile_job}"
echo "  → Will start after all validation jobs complete"
echo ""

# Summary output
echo "============================================================================"
echo -e "${BLUE}=== Submission Summary ===${NC}"
echo "============================================================================"
echo ""
echo "Total Jobs Submitted: ${#JOB_IDS[@]}"
echo ""

echo -e "${YELLOW}Validation Jobs (Running in Parallel):${NC}"
echo ""
echo "  1. Baseline (No Enzyme Constraints)"
echo "     Job: ${JOB_IDS[0]}"
echo "     Time: ~1 hour (fastest - no enzyme constraints)"
echo ""
echo "  2. Pre-tuning - Option A: Enzyme REMOVED (Proteome Reallocation)"
echo "     Job: ${JOB_IDS[1]}"
echo "     Time: ~2-3 hours"
echo "     Shows: Potential growth improvements from enzyme cost reduction"
echo ""
echo "  3. Pre-tuning - Option B: Enzyme KEPT (Immediate Knockout)"
echo "     Job: ${JOB_IDS[2]}"
echo "     Time: ~2-3 hours"
echo "     Shows: True knockout effects, essential genes visible"
echo ""
echo "  4. Post-tuning - Option A: Enzyme REMOVED (Proteome Reallocation)"
echo "     Job: ${JOB_IDS[3]}"
echo "     Time: ~2-3 hours"
echo "     Shows: Potential growth improvements from enzyme cost reduction"
echo ""
echo "  5. Post-tuning - Option B: Enzyme KEPT (Immediate Knockout)"
echo "     Job: ${JOB_IDS[4]}"
echo "     Time: ~2-3 hours"
echo "     Shows: True knockout effects, essential genes visible"
echo ""
echo -e "${YELLOW}Compilation Job (Runs After All Validation):${NC}"
echo ""
echo "  6. Compile Results"
echo "     Job: ${JOB_IDS[5]}"
echo "     Dependency: All validation jobs complete"
echo "     Generates: Fitness-based correlations for each validation run"
echo ""

echo "============================================================================"
echo -e "${BLUE}Monitoring Commands:${NC}"
echo "============================================================================"
echo ""
echo "Check all job status:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific jobs:"
echo "  squeue -j ${JOB_IDS[0]},${JOB_IDS[1]},${JOB_IDS[2]},${JOB_IDS[3]},${JOB_IDS[4]},${JOB_IDS[5]}"
echo ""
echo "View job details:"
echo "  scontrol show job ${JOB_IDS[0]}"
echo ""
echo "View live output (follow):"
echo "  tail -f logs/baseline_${JOB_IDS[0]}.out"
echo "  tail -f logs/pretuning_removed_${JOB_IDS[1]}.out"
echo "  tail -f logs/pretuning_kept_${JOB_IDS[2]}.out"
echo "  tail -f logs/posttuning_removed_${JOB_IDS[3]}.out"
echo "  tail -f logs/posttuning_kept_${JOB_IDS[4]}.out"
echo "  tail -f logs/compile_${JOB_IDS[5]}.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel ${JOB_IDS[0]} ${JOB_IDS[1]} ${JOB_IDS[2]} ${JOB_IDS[3]} ${JOB_IDS[4]} ${JOB_IDS[5]}"
echo ""

echo "============================================================================"
echo -e "${BLUE}Expected Timeline:${NC}"
echo "============================================================================"
echo ""
echo "  T+0h:    All validation jobs start"
echo "  T+1h:    Baseline job completes"
echo "  T+2-3h:  Pre-tuning and Post-tuning jobs complete (5 jobs running in parallel)"
echo "  T+3h:    Compilation job starts automatically"
echo "  T+3.1h:  All results ready in: ${OUTPUT_DIR}/"
echo ""

echo "============================================================================"
echo -e "${BLUE}Output Files per Validation Run:${NC}"
echo "============================================================================"
echo ""
echo "Baseline:"
echo "  - baseline_GEM.npy          (gene knockout results)"
echo "  - baseline_wildtype.npy     (wild-type growth, no constraints)"
echo ""
echo "Pre-tuning (Enzyme Removed):"
echo "  - pretuning_GEM_enzyme_removed.npy      (gene knockout results)"
echo "  - pretuning_wildtype_enzyme_removed.npy (wild-type growth)"
echo ""
echo "Pre-tuning (Enzyme Kept):"
echo "  - pretuning_GEM_enzyme_kept.npy         (gene knockout results)"
echo "  - pretuning_wildtype_enzyme_kept.npy    (wild-type growth)"
echo ""
echo "Post-tuning (Enzyme Removed):"
echo "  - posttuning_GEM_enzyme_removed.npy     (gene knockout results)"
echo "  - posttuning_wildtype_enzyme_removed.npy (wild-type growth)"
echo ""
echo "Post-tuning (Enzyme Kept):"
echo "  - posttuning_GEM_enzyme_kept.npy        (gene knockout results)"
echo "  - posttuning_wildtype_enzyme_kept.npy   (wild-type growth)"
echo ""

echo "============================================================================"
echo -e "${GREEN}✓ All jobs submitted successfully!${NC}"
echo "============================================================================"
echo ""
echo "Email notifications will be sent to ranamoneim@gmail.com when:"
echo "  - Each job starts"
echo "  - Each job completes"
echo "  - Any job fails"
echo ""
echo "Results directory: ${OUTPUT_DIR}/"
echo ""

# Save job IDs for reference
JOB_INFO_FILE="${OUTPUT_DIR}/job_ids_extended.txt"
cat > "${JOB_INFO_FILE}" <<EOF
Parallel Validation Jobs (Extended - Enzyme Options)
=====================================================
Submitted: $(date)

Job IDs:
  Baseline:                  ${JOB_IDS[0]}
  Pre-tuning Enzyme Removed: ${JOB_IDS[1]}
  Pre-tuning Enzyme Kept:    ${JOB_IDS[2]}
  Post-tuning Enzyme Removed: ${JOB_IDS[3]}
  Post-tuning Enzyme Kept:   ${JOB_IDS[4]}
  Compilation:               ${JOB_IDS[5]}

Log Files:
  Baseline:                  logs/baseline_${JOB_IDS[0]}.out
  Pre-tuning Enzyme Removed: logs/pretuning_removed_${JOB_IDS[1]}.out
  Pre-tuning Enzyme Kept:    logs/pretuning_kept_${JOB_IDS[2]}.out
  Post-tuning Enzyme Removed: logs/posttuning_removed_${JOB_IDS[3]}.out
  Post-tuning Enzyme Kept:   logs/posttuning_kept_${JOB_IDS[4]}.out
  Compilation:               logs/compile_${JOB_IDS[5]}.out

Monitor Command:
  squeue -j ${JOB_IDS[0]},${JOB_IDS[1]},${JOB_IDS[2]},${JOB_IDS[3]},${JOB_IDS[4]},${JOB_IDS[5]}

Enzyme Allocation Descriptions:
  REMOVED: Enzyme constraints removed for knockout genes (proteome reallocation)
           → May show growth improvements due to enzyme cost reduction
  KEPT:    Enzyme constraints retained for knockout genes (immediate effect)
           → Shows true knockout effects and essential genes

Expected Completion: ~3-4 hours (T+3h compilation starts, T+3.1h all done)
EOF

echo "Job information saved to: ${JOB_INFO_FILE}"
echo ""

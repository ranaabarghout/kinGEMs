#!/bin/bash
#
# Pre-flight Check for Parallel Validation Jobs
# ==============================================
#
# This script verifies that all required files and configurations
# are in place before submitting parallel validation jobs.
#

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================================================"
echo "=== Pre-flight Check: Parallel Validation Jobs ==="
echo "========================================================================"
echo ""

ERRORS=0
WARNINGS=0

# Function to check file existence
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# Function to check directory existence
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1/"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Missing directory (will be created): $1/"
        WARNINGS=$((WARNINGS + 1))
        return 1
    fi
}

# Function to check if file is executable
check_executable() {
    if [ -x "$1" ]; then
        echo -e "${GREEN}✓${NC} Executable: $1"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Not executable: $1"
        WARNINGS=$((WARNINGS + 1))
        return 1
    fi
}

# === Check SLURM Job Scripts ===
echo -e "${BLUE}[1/7] Checking SLURM job scripts...${NC}"
check_file "slurm_jobs/01_baseline_validation.sh"
check_file "slurm_jobs/02_pretuning_validation.sh"
check_file "slurm_jobs/03_posttuning_validation.sh"
check_file "slurm_jobs/04_compile_results.sh"
check_file "slurm_jobs/submit_all.sh"
echo ""

# === Check Python Scripts ===
echo -e "${BLUE}[2/7] Checking Python scripts...${NC}"
check_file "scripts/run_validation_parallel.py"
check_file "scripts/compile_validation_results.py"
echo ""

# === Check Executability ===
echo -e "${BLUE}[3/7] Checking script executability...${NC}"
check_executable "slurm_jobs/submit_all.sh"
check_executable "scripts/run_validation_parallel.py"
check_executable "scripts/compile_validation_results.py"
echo ""

# === Check Configuration Files ===
echo -e "${BLUE}[4/7] Checking configuration files...${NC}"
check_file "configs/validation_iML1515.json"
echo ""

# === Check Model Files ===
echo -e "${BLUE}[5/7] Checking model files...${NC}"
if [ -f "configs/validation_iML1515.json" ]; then
    MODEL_PATH=$(python3 -c "import json; print(json.load(open('configs/validation_iML1515.json'))['model_path'])" 2>/dev/null || echo "")
    if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
        echo -e "${GREEN}✓${NC} Found model: $MODEL_PATH"
    else
        echo -e "${RED}✗${NC} Model file not found or could not read config"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# === Check Data Files ===
echo -e "${BLUE}[6/7] Checking tuning data files...${NC}"
if [ -f "configs/validation_iML1515.json" ]; then
    PRE_TUNING=$(python3 -c "import json; print(json.load(open('configs/validation_iML1515.json')).get('pre_tuning_data_path', ''))" 2>/dev/null || echo "")
    POST_TUNING=$(python3 -c "import json; print(json.load(open('configs/validation_iML1515.json')).get('post_tuning_data_path', ''))" 2>/dev/null || echo "")

    if [ -n "$PRE_TUNING" ] && [ -f "$PRE_TUNING" ]; then
        echo -e "${GREEN}✓${NC} Found pre-tuning data: $PRE_TUNING"
    else
        echo -e "${RED}✗${NC} Pre-tuning data not found: $PRE_TUNING"
        ERRORS=$((ERRORS + 1))
    fi

    if [ -n "$POST_TUNING" ] && [ -f "$POST_TUNING" ]; then
        echo -e "${GREEN}✓${NC} Found post-tuning data: $POST_TUNING"
    else
        echo -e "${RED}✗${NC} Post-tuning data not found: $POST_TUNING"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# === Check Directories ===
echo -e "${BLUE}[7/7] Checking output directories...${NC}"
check_dir "logs"
check_dir "results"
check_dir "results/validation_parallel"
check_dir "results/validation_compiled"
echo ""

# === Check Email Configuration ===
echo -e "${BLUE}Checking email configuration...${NC}"
EMAIL_COUNT=$(grep -l "ranamoneim@gmail.com" slurm_jobs/*.sh 2>/dev/null | wc -l)
if [ "$EMAIL_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠${NC} $EMAIL_COUNT job script(s) still have placeholder email address"
    echo -e "  ${YELLOW}→${NC} Update '--mail-user' in SLURM scripts before submitting"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✓${NC} Email addresses configured"
fi
echo ""

# === Check Python Environment ===
echo -e "${BLUE}Checking Python environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment exists: venv/"

    # Check if venv can be activated
    if [ -f "venv/bin/activate" ]; then
        echo -e "${GREEN}✓${NC} Virtual environment is activatable"
    else
        echo -e "${RED}✗${NC} Virtual environment cannot be activated"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}⚠${NC} Virtual environment not found: venv/"
    echo -e "  ${YELLOW}→${NC} Create with: python -m venv venv"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# === Check Required Python Packages ===
echo -e "${BLUE}Checking Python packages (if venv active)...${NC}"
if [ -n "$VIRTUAL_ENV" ]; then
    PACKAGES=("cobra" "numpy" "pandas" "matplotlib" "scipy" "dask")
    for pkg in "${PACKAGES[@]}"; do
        if python3 -c "import $pkg" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Package installed: $pkg"
        else
            echo -e "${RED}✗${NC} Package missing: $pkg"
            ERRORS=$((ERRORS + 1))
        fi
    done
else
    echo -e "${YELLOW}⚠${NC} Virtual environment not active - skipping package check"
    echo -e "  ${YELLOW}→${NC} Activate with: source venv/bin/activate"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# === Summary ===
echo "========================================================================"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}=== ✓ All checks passed! Ready to submit jobs. ===${NC}"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review job scripts in slurm_jobs/"
    echo "  2. Update email addresses if needed"
    echo "  3. Submit jobs: bash slurm_jobs/submit_all.sh"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}=== ⚠ Checks passed with $WARNINGS warning(s) ===${NC}"
    echo "========================================================================"
    echo ""
    echo "Warnings should be addressed before submitting jobs."
    echo "Most warnings are non-critical and jobs may still run."
    echo ""
    exit 0
else
    echo -e "${RED}=== ✗ Checks failed with $ERRORS error(s) and $WARNINGS warning(s) ===${NC}"
    echo "========================================================================"
    echo ""
    echo "Please fix the errors above before submitting jobs."
    echo ""
    exit 1
fi

#!/bin/bash
#SBATCH --account=def-mahadeva
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=iECIAI39_1322
#SBATCH --output=logs/BiGG_models/iECIAI39_1322_%j.out
#SBATCH --error=logs/BiGG_models/iECIAI39_1322_%j.err

# Load modules
module load StdEnv/2023
module load python/3.11

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs/BiGG_models

# Run pipeline
echo "Starting kinGEMs pipeline for iECIAI39_1322"
echo "Config: configs/BiGG_models/iECIAI39_1322.json"
CONFIG="configs/BiGG_models/iECIAI39_1322.json"
echo "Starting run at: $(date)"
echo ""

python scripts/run_pipeline.py configs/BiGG_models/iECIAI39_1322.json
EXIT_CODE=$?

echo ""
echo "Completed at: $(date)"

# ---- Patch model_config_summary.json with SLURM metrics ----
MODEL_NAME=$(basename "$CONFIG" .json)
RESULTS_DIR=$(ls -dt results/tuning_results/BiGG_models/${MODEL_NAME}_* 2>/dev/null | head -1)
if [ -n "$RESULTS_DIR" ] && [ -f "$RESULTS_DIR/model_config_summary.json" ]; then
    sleep 10  # allow sacct to register job stats
    MEM_KB=$(sacct -j "$SLURM_JOB_ID" --format=MaxRSS --noheader 2>/dev/null \
        | tr -d ' K' | grep -E '^[0-9]+$' | sort -n | tail -1)
    WALLTIME=$(sacct -j "$SLURM_JOB_ID" --format=Elapsed --noheader 2>/dev/null \
        | head -1 | tr -d ' ')
    source venv/bin/activate
    python3 -c "
import json, re
path = '$RESULTS_DIR/model_config_summary.json'
with open(path) as f:
    d = json.load(f)
d['slurm_job_id']          = '$SLURM_JOB_ID'
d['slurm_node']            = '$SLURM_NODELIST'
d['slurm_cpus']            = int('$SLURM_CPUS_PER_TASK')
d['slurm_mem_requested_gb'] = 32
mem_kb = '$MEM_KB'
d['slurm_mem_used_mb']     = int(mem_kb) // 1024 if mem_kb.isdigit() else None
# Convert HH:MM:SS walltime to seconds
wt = '$WALLTIME'
m = re.match(r'(\d+):(\d+):(\d+)', wt)
d['slurm_walltime_s']      = int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3)) if m else None
with open(path, 'w') as f:
    json.dump(d, f, indent=2)
print('  Patched SLURM metrics into', path)
"
fi

exit $EXIT_CODE

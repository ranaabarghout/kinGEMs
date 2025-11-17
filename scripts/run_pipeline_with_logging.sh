#!/usr/bin/env bash
#
# Wrapper script to run the kinGEMs pipeline with automatic logging
# Usage: ./run_pipeline_with_logging.sh <config_file> [--force]
#

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [--force]"
    echo "Example: $0 configs/iML1515_GEM.json"
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create logs directory
LOGS_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGS_DIR"

# Generate a run ID (will be overwritten by actual ID from Python script, but needed for temp file)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEMP_LOG="$LOGS_DIR/temp_$TIMESTAMP.out"

# Run the pipeline and capture output
echo "Running pipeline with logging enabled..."
echo "Temporary log: $TEMP_LOG"
echo ""

# Run Python script with both stdout and stderr captured
python "$SCRIPT_DIR/run_pipeline.py" "$@" 2>&1 | tee "$TEMP_LOG"

# Extract the actual run ID from the output
RUN_ID=$(grep "^Run ID:" "$TEMP_LOG" | head -1 | awk '{print $3}')

if [ -n "$RUN_ID" ]; then
    # Rename log file to match run ID
    FINAL_LOG="$LOGS_DIR/${RUN_ID}.out"
    mv "$TEMP_LOG" "$FINAL_LOG"
    echo ""
    echo "Log saved to: $FINAL_LOG"
else
    echo ""
    echo "Warning: Could not extract Run ID. Log saved to: $TEMP_LOG"
fi

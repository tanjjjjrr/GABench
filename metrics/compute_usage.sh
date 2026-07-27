#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

BASE_DIR="xxx"
USER_NAME="xxx"
MODEL_NAME="xxx"

echo "================================================="
echo "Starting token usage computation"
echo "================================================="

CATEGORIES=("theory_output" "retrieval_output" "ml_output" "insight_output" "prediction_output")

# Allow a single category override from the command line.
if [ "$#" -ge 1 ]; then
    CATEGORIES=("$1")
fi

for CATEGORY in "${CATEGORIES[@]}"; do
    INPUT_DIR="${BASE_DIR}/${CATEGORY}/${USER_NAME}/${MODEL_NAME}"
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Warning: Target Directory $INPUT_DIR not found, skipping..."
        continue
    fi

    echo "Processing: $INPUT_DIR"
    
    MODEL_OUTPUT_DIR="${SCRIPT_DIR}/${MODEL_NAME}"
    mkdir -p "$MODEL_OUTPUT_DIR"
    CAT_OUTPUT_TXT="${MODEL_OUTPUT_DIR}/compute_usage_summary_${CATEGORY}.txt"
    
    python "${SCRIPT_DIR}/compute_usage.py" \
      --output_dir "$INPUT_DIR" \
      --output_file "$CAT_OUTPUT_TXT"
      
    echo "-------------------------------------------------"
done

echo "================================================="
echo "Computation completed!"
echo "================================================="

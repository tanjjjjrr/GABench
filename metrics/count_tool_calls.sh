#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

BASE_DIR="xxx"
USER_NAME="xxx"
MODEL_NAME="xxx"

echo "================================================="
echo "Starting tool calls counting"
echo "================================================="

CATEGORIES=("theory_output" "retrieval_output" "ml_output" "insight_output" "prediction_output")

if [ "$#" -ge 1 ]; then
    CATEGORIES=("$1")
fi

for CATEGORY in "${CATEGORIES[@]}"; do
    if [ -d "${BASE_DIR}/metrics_1/${USER_NAME}/${MODEL_NAME}/${CATEGORY}" ]; then
        INPUT_DIR="${BASE_DIR}/metrics_1/${USER_NAME}/${MODEL_NAME}/${CATEGORY}"
    else
        INPUT_DIR="${BASE_DIR}/metrics_1/${USER_NAME}/${MODEL_NAME}"
    fi
    
    EVAL_DIR="${BASE_DIR}/${CATEGORY}/${USER_NAME}/${MODEL_NAME}"
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Warning: Target Directory $INPUT_DIR not found, skipping..."
        continue
    fi
    
    if [ ! -d "$EVAL_DIR" ]; then
        echo "Warning: Eval Directory $EVAL_DIR not found, skipping..."
        continue
    fi

    echo "Processing: Tools in $INPUT_DIR | Eval in $EVAL_DIR"
    
    CAT_OUTPUT_TXT="${SCRIPT_DIR}/${MODEL_NAME}/count_tool_calls_summary_${CATEGORY}.txt"
    
    python "${SCRIPT_DIR}/count_tool_calls.py" \
      --input_dir "$INPUT_DIR" \
      --eval_dir "$EVAL_DIR" \
      --output_file "$CAT_OUTPUT_TXT"
      
    echo "-------------------------------------------------"
done

echo "================================================="
echo "Counting completed!"
echo "================================================="

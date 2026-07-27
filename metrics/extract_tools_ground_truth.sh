#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

USER_NAME="xxx"
MODEL_NAME="xxx"
OUTPUT_DIR="xxx"

CATEGORIES=("insight" "prediction" "ml" "theory" "retrieval")

if [ "$#" -ge 1 ]; then
    CATEGORIES=("$1")
fi

echo "================================================="
echo "Starting tool extraction"
echo "User: $USER_NAME"
echo "Model: $MODEL_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "================================================="

for CATEGORY in "${CATEGORIES[@]}"; do
    TASK_NAME="${CATEGORY}_output"
    
    INPUT_DIR="xxx"
    
    echo "-------------------------------------------------"
    echo "Processing Category: $CATEGORY"
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Input Directory $INPUT_DIR not found, skipping..."
        continue
    fi
    
    python "ground_truth_tools.py" \
      --input_dir "$INPUT_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --model_name "$MODEL_NAME" \
      --task_name "$TASK_NAME"
      
done

echo "================================================="
echo "Done! Results have been saved to $OUTPUT_DIR"
echo "================================================="

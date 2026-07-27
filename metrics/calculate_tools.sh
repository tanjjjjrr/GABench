#!/bin/bash

# Configuration
PRED_DIR="xxx"
GT_DIR="xxx"
MODEL_NAME="xxx"
MAPPING_FILE="xxx"

CATEGORIES=("insight" "prediction" "ml" "theory" "retrieval")

# You can override with command line arguments if desired
if [ "$#" -eq 1 ]; then
    CATEGORIES=("$1")
elif [ "$#" -ge 4 ]; then
    PRED_DIR=$1
    GT_DIR=$2
    MODEL_NAME=$3
    CATEGORIES=("$4")
fi

if [ "$#" -ge 5 ]; then
    MAPPING_FILE=$5
fi

echo "================================================="
echo "Calculating LCS Score"
echo "Model: $MODEL_NAME"
echo "Prediction Dir: $PRED_DIR"
echo "Ground Truth Dir: $GT_DIR"
echo "================================================="

for CATEGORY in "${CATEGORIES[@]}"; do
    echo "-------------------------------------------------"
    echo "Category: $CATEGORY"
    python compute_tool_metrics.py \
        --pred_dir "$PRED_DIR" \
        --gt_dir "$GT_DIR" \
        --model_name "$MODEL_NAME" \
        --category "$CATEGORY" \
        --mapping_file "$MAPPING_FILE"
done

echo "================================================="
echo "Calculation finished!"
echo "================================================="
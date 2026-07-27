#!/bin/bash

# Supported categories
CATEGORIES=("insight" "prediction" "ml" "theory" "retrieval")

# You can override with command line arguments if desired
if [ "$#" -eq 1 ]; then
    CATEGORIES=("$1")
fi

echo "================================================="
echo "Starting extraction..."
echo "================================================="

for CATEGORY in "${CATEGORIES[@]}"; do
    # Map category to local folder name
    if [ "$CATEGORY" = "theory" ]; then
        INPUT_PATH="xxx"
    elif [ "$CATEGORY" = "retrieval" ]; then
        INPUT_PATH="xxx"
    else
        INPUT_PATH="xxx"
    fi
    
    OUTPUT_PATH="xxx"
    
    echo "Processing $CATEGORY from $INPUT_PATH to $OUTPUT_PATH"
    
    mkdir -p "$OUTPUT_PATH"
    
    python extract_tools.py \
        --input_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --category "$CATEGORY"
done

echo "================================================="
echo "Extraction finished!"
echo "================================================="

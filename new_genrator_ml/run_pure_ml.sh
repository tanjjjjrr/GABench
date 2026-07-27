#!/bin/bash
# Pure machine-learning task generation script

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# ==========================================
# Basic configuration
# ==========================================
TARGET_INPUT=${1:-"xxx"}

if [ -f "$TARGET_INPUT" ]; then
    FILES=("$TARGET_INPUT")
elif [ -d "$TARGET_INPUT" ]; then
    FILES=("$TARGET_INPUT"/*.pt)
else
    echo "Error: $TARGET_INPUT is neither a file nor a directory"
    exit 1
fi

# ==========================================
# Task-count configuration
# ==========================================
# Number of node-classification/link-prediction tasks in TAG mode
NUM_TASKS=10

# Number of graph-classification tasks in TPG mode (number of randomly sampled molecular graphs)
NUM_GRAPHS=5

for DATASET_PATH in "${FILES[@]}"; do
    if [ ! -f "$DATASET_PATH" ]; then
        echo "No .pt files found for $TARGET_INPUT"
        continue
    fi

    BASENAME=$(basename "$DATASET_PATH" .pt)
    
    if [ "$BASENAME" = "hiv" ] || [ "$BASENAME" = "pcba" ]; then
        GRAPH_FORMAT="TPG"
    else
        GRAPH_FORMAT="TAG"
    fi

    OUT_DIR="$DIR/$BASENAME"
    mkdir -p "$OUT_DIR"
    cd "$OUT_DIR"

    OUTPUT_FILE="pure_ml_bench.json"

    echo "================================================================="
    echo "Starting Pure ML Pipeline with dataset: $DATASET_PATH"
    echo "Format: $GRAPH_FORMAT"
    echo "Output Directory: $OUT_DIR"
    echo "================================================================="

    python "$DIR/generate_pure_ml.py" \
        --dataset "$DATASET_PATH" \
        --output "$OUTPUT_FILE" \
        --graph-format "$GRAPH_FORMAT" \
        --num-tasks "$NUM_TASKS" \
        --num-graphs "$NUM_GRAPHS"

    echo "================================================================="
    echo "Pure ML Benchmark generated successfully for $BASENAME!"
    echo "Saved to: $OUT_DIR/$OUTPUT_FILE"
    echo "================================================================="
done

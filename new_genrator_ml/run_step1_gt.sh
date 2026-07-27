#!/bin/bash
# Run Stage 1: compute only Ground Truth and save it to JSON

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# ==========================================
# Parameter configuration
# ==========================================
# 1. Dataset configuration
DATASET_PATH="xxx"
NUM_TASKS=5

# 2. Enable graph machine-learning (ML) tasks (optional: true or false)
ENABLE_ML="true"

# 3. Output-file configuration
OUTPUT_FILE="intermediate_ground_truth.json"

# 4. Control the generation mode (keep it consistent with step0)
MODE="ego"

# 5. Graph structure format
GRAPH_FORMAT="TAG"
# 6. [TPG mode] Number of tasks generated per molecular graph
TASKS_PER_GRAPH=3

echo "=== Stage 1: Computing Ground Truth ==="
if [ "$ENABLE_ML" = "true" ]; then
    ML_FLAG="--enable-ml"
else
    ML_FLAG=""
fi

python step1_compute_gt.py \
    --dataset "$DATASET_PATH" \
    --num-tasks "$NUM_TASKS" \
    --output "$OUTPUT_FILE" \
    --mode "$MODE" \
    --graph-format "${GRAPH_FORMAT:-TAG}" \
    --tasks-per-graph "$TASKS_PER_GRAPH" \
    $ML_FLAG

echo "Stage 1 finished. Results saved to $OUTPUT_FILE"

#!/bin/bash
# Script that runs all generation steps
# Execution follows the logical order: Step 0 -> Step 0.5 -> Step 1 -> Step 2

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# ==========================================
# Unified parameter configuration
# ==========================================

# 1. Dataset directory path (can be passed as the first script argument; default below)
DATASET_DIR=${1:-"xxx"}

# 2. Global xxx configuration
export xxx_PROVIDER_TYPE="xxx"
export xxx_API_KEY="xxx"
export xxx_BASE_URL="xxx"

# 3. Common configuration
MODE="ego"
GRAPH_TYPE="" # optional: citation, social, recommendation
GRAPH_FORMAT="PG" # optional: TAG, TPG, PG
ENABLE_ML="true"
ENABLE_MEM="false"

# mutually exclusive check
if [ "$ENABLE_ML" = "true" ] && [ "$ENABLE_MEM" = "true" ]; then
    echo "Error: ENABLE_ML and ENABLE_MEM cannot both be true. They are mutually exclusive."
    exit 1
fi

# Detailed node-classification task description (passed to step2 for xxx question generation)
# Use {node1_placeholder} instead of a concrete node; the program replaces it with the selected node ID or text
NODE_CLS_DESC="Use graph machine learning algorithms to classify whether a defendant is likely to commit a crime if released on bail."

# Detailed graph-classification task description
GRAPH_CLS_DESC="Predict the classification label of the entire graph (use {node1_placeholder} to identify the graph)"

if [ -z "$GRAPH_TYPE" ]; then
    # (Optional) For graph data without text attributes, provide a global graph description as a fallback
    GLOBAL_DESCRIPTION="overall description: This graph belongs to the criminal justice and legal domain. It describes a scenario of predicting recidivism, specifically determining whether a defendant is likely to commit a crime if released on bail.

node description: Nodes represent individual defendants who were released on bail.

edge description: Edges represent the similarity between defendants, typically constructed based on their past criminal records, case details, and demographic attributes."
fi

# [Common task configuration]
# Total number of generated tasks (for step1, non-TPG mode)
NUM_TASKS=200

# [TPG-specific configuration]
# Number of molecular graphs to sample from the graph list (for step0)
NUM_GRAPHS=50
# Number of tasks to generate for each molecular graph (for step1)
TASKS_PER_GRAPH=4

for DATASET_PATH in "$DATASET_DIR"/*.pt; do
    if [ ! -f "$DATASET_PATH" ]; then
        echo "No .pt files found in $DATASET_DIR"
        continue
    fi

    BASENAME=$(basename "$DATASET_PATH" .pt)
    
    OUT_DIR="$DIR/$BASENAME"
    mkdir -p "$OUT_DIR"
    cd "$OUT_DIR"

    # 4. Output file configuration
    METRICS_CACHE="global_metrics_cache.json"
    ANCHORS_CACHE="anchor_nodes_cache.json"
    GT_OUTPUT="intermediate_ground_truth.json"
    if [ "$ENABLE_ML" = "true" ]; then
        FINAL_BENCHMARK="graph-prediction.json"
    else
        FINAL_BENCHMARK="graph-insight.json"
    fi

    echo "================================================================="
    echo "Starting full pipeline with dataset: $DATASET_PATH"
    echo "Output directory: $OUT_DIR"
    echo "================================================================="

    if [ "$ENABLE_ML" = "true" ]; then
        ML_FLAG="--enable-ml"
    else
        ML_FLAG=""
    fi

    if [ "$ENABLE_MEM" = "true" ]; then
        MEM_FLAG="--enable-mem"
    else
        MEM_FLAG=""
    fi

    # ==========================================
    # Stage 0: Pre-calculating Global Graph Metrics
    # ==========================================
    echo -e "\n>>> Stage 0: Pre-calculating Global Graph Metrics & Dynamic Tool Descriptions <<<"
    # Use the pro model for step0
    export xxx_MODEL_NAME="xxx"

    python "$DIR/step0_precalc.py" \
        --dataset "$DATASET_PATH" \
        --output "$METRICS_CACHE" \
        --mode "$MODE" \
        ${GRAPH_TYPE:+--graph-type "$GRAPH_TYPE"} \
        --graph-format "$GRAPH_FORMAT" \
        --dynamic-desc-seed "42" \
        --walk-depth 5 \
        --num-graphs "$NUM_GRAPHS" \
        ${GLOBAL_DESCRIPTION:+--global-description "$GLOBAL_DESCRIPTION"} \
        $ML_FLAG

    # ==========================================
    # Stage 0.5: Generating specific anchor node lists
    # ==========================================
    echo -e "\n>>> Stage 0.5: Generating specific anchor node lists <<<"
    python "$DIR/step0_5_generate_anchors.py" \
        --dataset "$DATASET_PATH" \
        --output "$ANCHORS_CACHE" \
        --max-size 500 \
        --graph-format "$GRAPH_FORMAT"

    # ==========================================
    # Stage 1: Computing Ground Truth
    # ==========================================
    echo -e "\n>>> Stage 1: Computing Ground Truth <<<"


    python "$DIR/step1_compute_gt.py" \
        --dataset "$DATASET_PATH" \
        --num-tasks "$NUM_TASKS" \
        --output "$GT_OUTPUT" \
        --mode "$MODE" \
        --graph-format "$GRAPH_FORMAT" \
        ${GRAPH_TYPE:+--graph-type "$GRAPH_TYPE"} \
        --tasks-per-graph "$TASKS_PER_GRAPH" \
        $ML_FLAG \
        $MEM_FLAG

    # ==========================================
    # Stage 2: Generating Questions with xxx
    # ==========================================
    echo -e "\n>>> Stage 2: Generating Questions with xxx <<<"
    # Switch to the flash model for step2
    export xxx_MODEL_NAME="xxx"

    python "$DIR/step2_generate_questions.py" \
        --input "$GT_OUTPUT" \
        --output "$FINAL_BENCHMARK" \
        --metrics-cache "$METRICS_CACHE" \
        --graph-format "$GRAPH_FORMAT" \
        ${NODE_CLS_DESC:+--node-cls-desc "$NODE_CLS_DESC"} \
        ${GRAPH_CLS_DESC:+--graph-cls-desc "$GRAPH_CLS_DESC"} \
        $ML_FLAG

    echo -e "\n================================================================="
    echo "All stages finished successfully for $BASENAME!"
    echo "Final benchmark saved to: $OUT_DIR/$FINAL_BENCHMARK"
    echo "================================================================="
done

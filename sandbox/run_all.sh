#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# ==========================================
# ⚙️ Main orchestration script configuration
# ==========================================

# 1. Datasets and tasks to iterate over
DATASETS=("arxiv" "cora" "history" "reddit" "instagram" "pcba" "hiv" "bail" "creddit" "NBA" "pokec") # Configure the dataset list here.
TASKS=("insight" "ml" "prediction") # Configure the task list here.

# 2. Evaluation parameters
MODEL="xxx"
LIMIT=50

# 3. Log directory
LOG_DIR="logs/run_all"

# ==========================================
# Execution logic starts below.
# ==========================================

mkdir -p "$LOG_DIR"
echo "=========================================================="
echo "🚀 Preparing to start the WildClawBench orchestration script"
echo "Datasets: ${DATASETS[*]}"
echo "Model: $MODEL"
echo "LIMIT: $LIMIT"
echo "=========================================================="

# Preallocate base ports to avoid concurrency conflicts.
declare -A TASK_PORTS=(
    ["retrieval"]=20100
    ["theory"]=20400
    ["insight"]=20700
    ["ml"]=20600
    ["prediction"]=20900
)

# Wrap a single task execution in a function.
run_single_task() {
    local TASK=$1
    local DATASET=$2
    local MODE=$3 # "parallel" or "sequential"

    # Build the output directory dynamically.
    local OUTPUT_DIR="xxx/${TASK}_output/xxx/${MODEL}/${DATASET}"
    
    # Skip the task if its output directory already exists and is not empty.
    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR")" ]; then
        echo "⏭️ Skipping task [$TASK - $DATASET]: $OUTPUT_DIR already exists and is not empty."
        return 0
    fi

    # Build file paths dynamically based on the task type.
    local JSON_FILE=""
    local INTERMEDIATE_PT_FILE=""
    
    
    if [ "$TASK" = "insight" ]; then
        JSON_FILE="xxx/insight/${DATASET}/merged_output.json"
    elif [ "$TASK" = "prediction" ]; then
        JSON_FILE="xxx/prediction/${DATASET}/merged_output.json"
        INTERMEDIATE_PT_FILE="xxx/prediction/${DATASET}/intermediate_ground_truth_masked_data.pt"
    elif [ "$TASK" = "ml" ]; then
        JSON_FILE="xxx/ml/${DATASET}/pure_ml_bench.json"
        INTERMEDIATE_PT_FILE="xxx/ml/${DATASET}/pure_ml_bench_masked_data.pt"
    elif [ "$TASK" = "theory" ]; then
        JSON_FILE="xxx/theory_miniset/${DATASET}.json"
    elif [ "$TASK" = "retrieval" ]; then
        JSON_FILE="xxx/retrieval_miniset/${DATASET}.json"
    else
        echo "❌ Unknown task: $TASK"
        return 1
    fi
    local PT_FILE="xxx/dataset/${DATASET}.pt"
    local SCRIPT="${TASK}.sh"

    # Check required JSON/PT dataset files and skip safely if any are missing.
    if [ -n "$JSON_FILE" ] && [ ! -f "$JSON_FILE" ]; then
        echo "⚠️ Required data file not found: $JSON_FILE; skipping [$TASK - $DATASET]."
        return 0
    fi

    if [ -n "$INTERMEDIATE_PT_FILE" ] && [ ! -f "$INTERMEDIATE_PT_FILE" ]; then
        echo "⚠️ Required intermediate data file not found: $INTERMEDIATE_PT_FILE; skipping [$TASK - $DATASET]."
        return 0
    fi

    if [ ! -f "$PT_FILE" ]; then
        echo "⚠️ Graph dataset file not found: $PT_FILE; skipping [$TASK - $DATASET]."
        return 0
    fi

    if [ ! -f "$SCRIPT" ]; then
        echo "❌ Child script not found: $SCRIPT; skipping [$TASK - $DATASET]."
        return 1
    fi

    # Expose configuration to child scripts through environment variables.
    export RUN_ALL_MODEL="$MODEL"
    export RUN_ALL_LIMIT="$LIMIT"
    export RUN_ALL_OUTPUT_DIR="$OUTPUT_DIR"
    export RUN_ALL_JSON_FILE="$JSON_FILE"
    export RUN_ALL_INTERMEDIATE_PT_FILE="$INTERMEDIATE_PT_FILE"
    export RUN_ALL_PT_FILE="$PT_FILE"
    export RUN_ALL_PORT="${TASK_PORTS[$TASK]}"

    local LOG_FILE="${LOG_DIR}/${TASK}_${DATASET}.log"
    echo "▶️ Starting task: [$TASK - $DATASET] ..."

    if [ "$MODE" = "parallel" ]; then
        bash "$SCRIPT" > "$LOG_FILE" 2>&1 &
        echo "   [background parallel] PID: $!, log: $LOG_FILE"
    else
        echo "   [foreground sequential] log: $LOG_FILE"
        bash "$SCRIPT" > "$LOG_FILE" 2>&1
    fi
}

# Check whether the task is enabled in the configured TASKS list.
is_task_enabled() {
    local target=$1
    for t in "${TASKS[@]}"; do
        if [ "$t" = "$target" ]; then
            return 0
        fi
    done
    return 1
}

# Iterate over every dataset.
for DATASET in "${DATASETS[@]}"; do
    echo "=========================================================="
    echo "🔥 Processing dataset: $DATASET"
    echo "=========================================================="
    
    # ---------------------------------------------------------
    # Stage 1: run retrieval, theory, and insight in parallel when enabled.
    # ---------------------------------------------------------
    echo "⚡ [Stage 1] Starting retrieval, theory, and insight in parallel when enabled..."
    has_parallel=0
    for TASK in "retrieval" "theory" "insight"; do
        if is_task_enabled "$TASK"; then
            run_single_task "$TASK" "$DATASET" "parallel"
            has_parallel=1
        fi
    done
    
    if [ "$has_parallel" -eq 1 ]; then
        echo "⏳ Waiting for all Stage 1 parallel tasks for $DATASET to finish..."
        wait
        echo "✅ All Stage 1 tasks for $DATASET are complete."
    fi
    echo "----------------------------------------------------------"

    # ---------------------------------------------------------
    # Stage 2: run ML and prediction sequentially when enabled.
    # ---------------------------------------------------------
    echo "⚡ [Stage 2] Starting ML followed by prediction sequentially when enabled..."
    for TASK in "ml" "prediction"; do
        if is_task_enabled "$TASK"; then
            run_single_task "$TASK" "$DATASET" "sequential"
        fi
    done
    
    echo "✅ All tasks for $DATASET are complete."
done

echo "=========================================================="
echo "🎉 All dataset evaluations are complete."
echo "=========================================================="

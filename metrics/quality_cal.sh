#!/bin/bash

# ==============================================================================
# Configuration Variables (Modify these as needed)
# ==============================================================================
CATEGORY="prediction"  # Options: prediction, insight, ml, theory, retrieval
MODEL_NAME="xxx"  # Model used for the generation (to locate folders)

# Paths for predictions and ground truth
# Update PRED_BASE_DIR to point to where the model's outputs are stored.
PRED_BASE_DIR="xxx"

GT_BASE_DIR="xxx" 

# LLM Evaluator API Parameters
EVAL_MODEL_NAME="xxx"
API_URL="xxx" # Adjust if using a custom endpoint (e.g. xxx's official API url)
API_KEY="xxx"

# Path to the eval python script
EVAL_SCRIPT="$(dirname "$0")/llm_eval.py"

echo "========================================"
echo "Starting LLM Batch Evaluation for Category: $CATEGORY"
echo "Prediction Base: xxx"
echo "========================================"

CATEGORY_DIR="xxx"

if [ ! -d "$CATEGORY_DIR" ]; then
    echo "Error: Directory $CATEGORY_DIR does not exist."
    echo "Please update PRED_BASE_DIR to the correct path."
    exit 1
fi

# Iterate over datasets (e.g., cora, pubmed)
for dataset_path in "$CATEGORY_DIR"/*; do
    if [ -d "$dataset_path" ]; then
        dataset=$(basename "$dataset_path")
        
        # Determine the Ground Truth file for this dataset based on category.
        # You may need to adjust these paths if your ground truth folders differ slightly.
        if [ "$CATEGORY" = "theory" ]; then
            GT_FILE="$GT_BASE_DIR/theory_sub/${dataset}.json"
        elif [ "$CATEGORY" = "retrieval" ]; then
            GT_FILE="$GT_BASE_DIR/retrieval_sub/${dataset}.json"
        elif [ "$CATEGORY" = "insight" ] || [ "$CATEGORY" = "prediction" ]; then
            GT_FILE="$GT_BASE_DIR/$CATEGORY/$dataset/merged_output.json"
        elif [ "$CATEGORY" = "ml" ]; then
            # From the prompt context, ml benchmark seems to be pure_ml_bench.json
            GT_FILE="$GT_BASE_DIR/$CATEGORY/$dataset/pure_ml_bench.json"
        else
            GT_FILE="$GT_BASE_DIR/$CATEGORY/$dataset/${dataset}.json"
        fi
        
        echo "Processing Dataset: $dataset (GT: $GT_FILE)"
        
        # Iterate over task instances (0, 1, 2...)
        for task_dir in "$dataset_path"/*; do
            if [ -d "$task_dir" ]; then
                task_id=$(basename "$task_dir")
                
                if [ -f "$task_dir/results/chat.jsonl" ]; then
                    chat_file="$task_dir/results/chat.jsonl"
                else
                    chat_file="$task_dir/chat.jsonl"
                fi
                
                out_file="$task_dir/llm_eval_result.json"
                
                if [ -f "$chat_file" ]; then
                    echo "  -> Evaluating Task $task_id ..."
                    python "$EVAL_SCRIPT" \
                        --task "$CATEGORY" \
                        --input_path "$chat_file" \
                        --output_path "$out_file" \
                        --gt_file "$GT_FILE" \
                        --model_name "$EVAL_MODEL_NAME" \
                        --api_url "$API_URL" \
                        --api_key "$API_KEY"
                else
                    echo "  -> Warning: No chat.jsonl found in $task_dir"
                fi
            fi
        done
    fi
done

echo "========================================"
echo "Batch Evaluation Completed!"
echo "========================================"

echo ""
echo "Calculating Detailed Pass Rates..."
python "compute_pass_rate.py" "$CATEGORY_DIR" | tee "$CATEGORY_DIR/pass_rate_summary.txt"
echo "Summary saved to: $CATEGORY_DIR/pass_rate_summary.txt"




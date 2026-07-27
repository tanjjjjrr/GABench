#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# ==========================================
# ⚙️ Configuration (edit the settings here directly)
# ==========================================

# 1. Required: input file paths (replace them with your actual paths)
PT_FILE="${RUN_ALL_PT_FILE:-xxx}"
JSON_FILE="${RUN_ALL_JSON_FILE:-xxx}"
INTERMEDIATE_PT_FILE="${RUN_ALL_INTERMEDIATE_PT_FILE:-xxx}" # Optional: path to intermediate_ground_truth_masked_data.pt for ML evaluations
# 2. Optional: xxx configuration
AGENT="xxx" # Options: xxx, xxx, xxx, xxx
if [ "$AGENT" = "xxx" ]; then
    # xxx does not require cc-switch; enter the actual configuration.
    MODEL="${RUN_ALL_MODEL:-xxx}"
    API_URL="xxx"
    API_KEY="xxx"
elif [ "$AGENT" = "xxx" ]; then
    # xxx generates a dedicated configuration from a template; enter the model name to test here.
    MODEL="xxx"
    API_URL="xxx"
    API_KEY="xxx"
elif [ "$AGENT" = "xxx" ]; then
    # xxx Code uses an xxx-compatible endpoint.
    MODEL="xxx"
    API_URL="xxx"
    API_KEY="xxx"
else
    # xxx uses cc-switch as a proxy for protocol conversion.
    MODEL="xxx"
    API_URL="xxx"
    API_KEY="xxx"
fi

# 3. Optional: local directory containing custom tools/skills (leave empty to disable loading)
SKILLS_DIR="xxx"

# 4. Optional: number of parallel evaluations
PARALLEL=4

# 5. Optional: evaluation question limit (empty or 0 means unlimited)
LIMIT="${RUN_ALL_LIMIT:-100}"

# 6. Optional: base port (avoids collisions during parallel evaluations)
PORT="${RUN_ALL_PORT:-18789}"

# 7. Output directory
OUTPUT_DIR="${RUN_ALL_OUTPUT_DIR:-xxx}"

# ==========================================
# No changes are needed below this line.
# ==========================================

# Validate required inputs.
if [ -z "$PT_FILE" ] || [ -z "$JSON_FILE" ]; then
    echo "Error: PT_FILE and JSON_FILE cannot be empty. Update the configuration at the top of the script."
    exit 1
fi
echo "=========================================================="
echo "🚀 Preparing to start the WildClawBench simple evaluation (simple_eval.py)"
echo "----------------------------------------------------------"
echo "📦 Dataset (PT):  $PT_FILE"
echo "📝 Questions (JSON):  $JSON_FILE"
echo "🤖 Agent framework:    $AGENT"
echo "🤖 Model:    $MODEL"
echo "⚡ Parallel jobs:    $PARALLEL"
if [ -n "$LIMIT" ] && [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    echo "🎯 Evaluation limit:    $LIMIT"
fi
if [ -n "$API_URL" ]; then
    echo "🔗 API URL:     $API_URL"
fi
if [ -n "$API_KEY" ]; then
    echo "🔑 API Key:     (set)"
fi
if [ -n "$SKILLS_DIR" ]; then
    echo "🛠️ Custom tools:  $SKILLS_DIR"
fi
echo "📁 Output directory:    $OUTPUT_DIR"
echo "=========================================================="

# Build the base command.
CMD=(python simple_eval.py \
    --pt_file "$PT_FILE" \
    --json_file "$JSON_FILE" \
    --agent "$AGENT" \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --parallel "$PARALLEL")

# Add optional API URL and API key arguments.
if [ -n "$API_URL" ]; then
    CMD+=(--api_url "$API_URL")
fi

if [ -n "$API_KEY" ]; then
    CMD+=(--api_key "$API_KEY")
fi

if [ -n "$SKILLS_DIR" ]; then
    CMD+=(--skills_dir "$SKILLS_DIR")
fi

# Add the optional intermediate_pt_file argument.
if [ -n "$INTERMEDIATE_PT_FILE" ]; then
    CMD+=(--intermediate_pt_file "$INTERMEDIATE_PT_FILE")
fi

# Add the optional limit argument.
if [ -n "$LIMIT" ] && [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    CMD+=(--limit "$LIMIT")
fi

# Add the optional port argument.
if [ -n "$PORT" ]; then
    CMD+=(--port "$PORT")
fi

# Run the Python script.
"${CMD[@]}"

echo "=========================================================="
echo "✅ Evaluation complete. All outputs and trajectory files have been saved under $OUTPUT_DIR."
echo "=========================================================="

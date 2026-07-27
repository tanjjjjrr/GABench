#!/bin/bash
# Run Stage 0: precompute global graph metrics and persist them locally
# New (new_instruct.md): also call DynamicToolDescriptionGenerator during step0,
# write dynamic tool descriptions to the same JSON cache for step2

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# ==========================================
# Parameter configuration
# ==========================================
# 1. Dataset configuration
DATASET_PATH="xxx"

# 2. xxx configuration (for dynamic tool descriptions and real-world scenarios)
export xxx_PROVIDER_TYPE="xxx"
export xxx_MODEL_NAME="xxx"
export xxx_API_KEY="xxx"
export xxx_BASE_URL="xxx"

# 3. Output-file configuration
OUTPUT_FILE="global_metrics_cache.json"

# 4. Other advanced configuration
# (Optional) For graph data without text attributes, provide a global graph description as the fallback (Task 4)
GLOBAL_DESCRIPTION=""
# (Optional) Random seed for dynamic-description sampling; leave empty for non-deterministic sampling
DYNAMIC_DESC_SEED="42"
# Control the DFS random-walk step size during subgraph extraction
WALK_DEPTH=5
# Add --skip-dynamic-desc to skip dynamic-description generation and only precompute graph metrics
EXTRA_FLAGS=""
# Choose the operator set: global (basic tools by default) or ego (local tools)
MODE="ego"
# Specify the graph type to control hard-coded edge descriptions in the xxx prompt: citation, social, recommendation
GRAPH_TYPE="citation"
# Graph structure format: TAG (text-attributed graph), TPG (Test-Paired graph classification), PG (pure structural graph)
GRAPH_FORMAT="TAG"
# [TPG mode] Number of molecular graphs to sample from the graph list
NUM_GRAPHS=5

echo "=== Stage 0: Pre-calculating Global Graph Metrics & Dynamic Tool Descriptions ==="
python step0_precalc.py \
    --dataset "$DATASET_PATH" \
    --output "$OUTPUT_FILE" \
    --mode "$MODE" \
    --graph-type "$GRAPH_TYPE" \
    --graph-format "${GRAPH_FORMAT:-TAG}" \
    --dynamic-desc-seed "$DYNAMIC_DESC_SEED" \
    --walk-depth "$WALK_DEPTH" \
    --num-graphs "$NUM_GRAPHS" \
    ${GLOBAL_DESCRIPTION:+--global-description "$GLOBAL_DESCRIPTION"} \
    $EXTRA_FLAGS

echo "Stage 0 finished."
echo "  - Graph metrics cache : $OUTPUT_FILE"
echo "  - Dynamic tool descs  : embedded in $OUTPUT_FILE (key: dynamic_tool_descriptions)"

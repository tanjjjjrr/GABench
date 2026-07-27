#!/bin/bash
# Run the anchor precomputation script
# Run this script after run_step0_precalc.sh and before run_step1_gt.sh

DATASET=${1:-"../../data/processed_data.pt"}
OUTPUT=${2:-"anchor_nodes_cache.json"}
MAX_SIZE=${3:-500}
GRAPH_FORMAT=${4:-"TAG"}

echo "Generating specific anchor node lists (max $MAX_SIZE per list) for high-level stages..."
python step0_5_generate_anchors.py \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --max-size $MAX_SIZE \
    --graph-format "$GRAPH_FORMAT"

echo "Done."

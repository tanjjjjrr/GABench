#!/usr/bin/env python3
import json
import logging
import argparse
from pathlib import Path
import torch
import sys

_this_dir = Path(__file__).parent
sys.path.insert(0, str(_this_dir))
try:
    from base_graph_loader import BaseGraphLoader
except ImportError:
    print("Error: Could not import base_graph_loader. Please run this script from the new_genrator directory.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fix_masked_data(dataset_path, json_path, pt_path):
    logger.info(f"Processing JSON: {json_path}")
    logger.info(f"Processing Masked PT: {pt_path}")
    logger.info(f"Using Original Dataset: {dataset_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    graph_format = data.get("metadata", {}).get("graph_format", "TAG")
    
    loader = BaseGraphLoader(dataset_path, graph_format=graph_format)
    
    orig_labels = getattr(loader.data, 'y', None)
    if orig_labels is None:
        orig_labels = getattr(loader.data, 'label', None)
        
    if orig_labels is None:
        logger.warning(f"No labels found in {dataset_path} (checked 'y' and 'label'). Skipping.")
        return
        
    masked_data = torch.load(pt_path)
    tasks = data.get("intermediate_tasks", [])
    
    # Create mapping from task_index to task data
    task_map = {t["task_index"]: t for t in tasks}
    
    fixed_count = 0
    for task_idx, masked_item in masked_data.items():
        if masked_item.get("labels") is None:
            task = task_map.get(task_idx)
            if not task:
                continue
                
            labels = orig_labels.clone()
            selected_tools = task.get("selected_tools", [])
            
            if "structural_node_embedding_classification" in selected_tools:
                node_id = task.get("anchor_nodes", {}).get("node1")
                if node_id is not None:
                    try:
                        labels[node_id] = -100
                    except Exception:
                        pass
            elif "structural_graph_embedding_classification" in selected_tools:
                # Need to find graph_idx for the node
                node_id = task.get("anchor_nodes", {}).get("node1")
                if node_id is not None and hasattr(loader.data, 'batch'):
                    graph_idx = loader.data.batch[node_id].item()
                    try:
                        labels[graph_idx] = -100
                    except Exception:
                        pass
                elif "graph_idx" in task:
                    graph_idx = task["graph_idx"]
                    try:
                        labels[graph_idx] = -100
                    except Exception:
                        pass
            
            # For link prediction, labels remain unchanged
            
            masked_item["labels"] = labels
            fixed_count += 1
            
    if fixed_count > 0:
        torch.save(masked_data, pt_path)
        logger.info(f"Fixed {fixed_count} tasks in {pt_path}")
    else:
        logger.info(f"No missing labels needed fixing in {pt_path}")

def main():
    parser = argparse.ArgumentParser(description="Fix masked data pt file missing labels.")
    parser.add_argument("--dataset_pt", type=str, required=True, help="Path to the original dataset .pt file")
    parser.add_argument("--json_path", type=str, required=True, help="Path to intermediate_ground_truth.json")
    parser.add_argument("--masked_pt", type=str, required=True, help="Path to intermediate_ground_truth_masked_data.pt")
    
    args = parser.parse_args()
    
    if not Path(args.dataset_pt).exists():
        logger.error(f"Dataset not found: {args.dataset_pt}")
        return
    if not Path(args.json_path).exists():
        logger.error(f"JSON not found: {args.json_path}")
        return
    if not Path(args.masked_pt).exists():
        logger.error(f"Masked PT not found: {args.masked_pt}")
        return
        
    fix_masked_data(args.dataset_pt, args.json_path, args.masked_pt)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pure ML Benchmark generation script.

Generates graph machine-learning benchmark tasks without an xxx. TAG graphs produce node-classification or link-prediction tasks, while TPG graphs produce graph-classification tasks.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
import copy
import torch

_this_dir = Path(__file__).parent
_root_dir = _this_dir.parent
sys.path.insert(0, str(_root_dir))
sys.path.insert(0, str(_this_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from base_graph_loader import BaseGraphLoader, TPGDataset

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to processed_data.pt")
    parser.add_argument("--output", type=str, default="pure_ml_bench.json", help="Output JSON benchmark file")
    parser.add_argument("--graph-format", type=str, choices=["TAG", "TPG"], required=True, help="Graph format: TAG or TPG")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks for TAG mode")
    parser.add_argument("--num-graphs", type=int, default=5, help="Number of graphs for TPG mode")
    return parser

def main():
    args = build_parser().parse_args()
    logger.info(f"Generating Pure ML benchmark. Format: {args.graph_format}")
    
    tasks = []
    ml_masked_data = {}
    
    if args.graph_format == "TPG":
        tpg_dataset = TPGDataset(args.dataset)
        num_avail = getattr(tpg_dataset, "num_graphs", len(tpg_dataset.graphs) if hasattr(tpg_dataset, 'graphs') else 0)
        
        if num_avail == 0:
            logger.error("No graphs found in TPG dataset.")
            return
            
        selected_indices = random.sample(range(num_avail), min(args.num_graphs, num_avail))
        
        for task_idx, graph_idx in enumerate(selected_indices):
            g = tpg_dataset.get_graph(graph_idx)
            label_val = tpg_dataset.get_graph_label(graph_idx)
            
            ml_masked_data[task_idx] = {
                "labels": torch.tensor([-100], dtype=torch.long),
                "edge_index": g.edge_index if hasattr(g, 'edge_index') and g.edge_index is not None else None
            }
            
            tasks.append({
                "task_index": task_idx,
                "task_type": "GC",
                "graph_idx": graph_idx,
                "question": f"Predict the category of Graph {graph_idx}.",
                "ground_truth": label_val
            })
            
    elif args.graph_format == "TAG":
        loader = BaseGraphLoader(args.dataset, graph_format="TAG")
        data = loader.data
        
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
        edge_index = data.edge_index
        num_edges = edge_index.size(1) if edge_index is not None else 0
        
        for task_idx in range(args.num_tasks):
            # Randomly choose NC or LP (force NC when no edges exist)
            if num_edges > 0:
                task_type = random.choice(["NC", "LP"])
            else:
                task_type = "NC"
                
            if task_type == "NC":
                node = random.randint(0, num_nodes - 1)
                
                orig_labels = getattr(data, 'y', None)
                if orig_labels is None:
                    orig_labels = getattr(data, 'label', None)
                
                label_val = orig_labels[node].item() if orig_labels is not None else -1
                
                # Mask node label
                labels = orig_labels.clone() if orig_labels is not None else None
                if labels is not None:
                    try:
                        labels[node] = -100
                    except Exception:
                        pass
                
                ml_masked_data[task_idx] = {
                    "labels": labels,
                    "edge_index": data.edge_index if hasattr(data, 'edge_index') and data.edge_index is not None else None
                }
                
                tasks.append({
                    "task_index": task_idx,
                    "task_type": "NC",
                    "node_id": node,
                    "question": f"Predict the category of Node {node}.",
                    "ground_truth": label_val
                })
                
            elif task_type == "LP":
                # Randomly choose from the set of existing edges
                edge_idx = random.randint(0, num_edges - 1)
                node1 = data.edge_index[0, edge_idx].item()
                node2 = data.edge_index[1, edge_idx].item()
                
                # Mask all edges between this pair of nodes (including both directions)
                mask1 = (data.edge_index[0] == node1) & (data.edge_index[1] == node2)
                mask2 = (data.edge_index[0] == node2) & (data.edge_index[1] == node1)
                keep_mask = ~(mask1 | mask2)
                new_edge_index = data.edge_index[:, keep_mask]
                
                orig_labels = getattr(data, 'y', None)
                if orig_labels is None:
                    orig_labels = getattr(data, 'label', None)
                
                ml_masked_data[task_idx] = {
                    "labels": orig_labels,
                    "edge_index": new_edge_index
                }
                
                tasks.append({
                    "task_index": task_idx,
                    "task_type": "LP",
                    "node1": node1,
                    "node2": node2,
                    "question": f"Predict whether there is a link between Node {node1} and Node {node2}.",
                    "ground_truth": True
                })
                
    out_data = {
        "metadata": {
            "dataset": args.dataset,
            "graph_format": args.graph_format
        },
        "benchmark": tasks
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
        
    masked_data_path = Path(args.output).parent / f"{Path(args.output).stem}_masked_data.pt"
    torch.save(ml_masked_data, masked_data_path)
        
    logger.info(f"Successfully generated {len(tasks)} pure ML tasks. Masked data saved to {masked_data_path}")

if __name__ == "__main__":
    main()

import os
import json
import argparse
from pathlib import Path

def process_insight_prediction(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_tools = []
    task_tools = []
    if "tasks" in data:
        for idx, task in enumerate(data["tasks"]):
            t = task.get("required_tools", [])
            task_tools.append({"task_id": idx, "tools": t})
            all_tools.extend(t)
    return list(set(all_tools)), task_tools

def process_ml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_tools = set()
    task_tools = []
    if "benchmark" in data:
        for idx, task in enumerate(data["benchmark"]):
            task_type = task.get("task_type", "")
            t = []
            if task_type == "NC":
                t = ["structural_node_embedding_classification_train", "structural_node_embedding_classification_infer"]
            elif task_type == "LP":
                t = ["structural_node_embedding_link_prediction_train", "structural_node_embedding_link_prediction_infer"]
            elif task_type == "GC":
                t = ["structural_graph_embedding_classification_train", "structural_graph_embedding_classification_infer"]
            
            task_tools.append({"task_id": idx, "task_type": task_type, "tools": t})
            for tool in t:
                all_tools.add(tool)
    return list(all_tools), task_tools

def process_theory_retrieval(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_tools = []
    task_tools = []
    
    def extract_from_list(tasks, start_idx=0, task_type=None):
        for idx, task in enumerate(tasks, start=start_idx):
            t = task.get("tool_select", [])
            task_item = {"task_id": idx, "tools": t}
            if task_type:
                task_item["task_type"] = task_type
            task_tools.append(task_item)
            all_tools.extend(t)
        return start_idx + len(tasks)
            
    if isinstance(data, dict):
        curr_idx = 0
        for key in ["node", "edge", "graph"]:
            if key in data:
                curr_idx = extract_from_list(data[key], curr_idx, key)
    elif isinstance(data, list):
        extract_from_list(data, 0, None)
        
    return list(set(all_tools)), task_tools

def main():
    parser = argparse.ArgumentParser(description="Extract tools from datasets.")
    parser.add_argument('--input_path', type=str, required=True, help='Input base path')
    parser.add_argument('--output_path', type=str, help='Output base path, defaults to input_path')
    parser.add_argument('--category', type=str, required=True, choices=['insight', 'prediction', 'ml', 'theory', 'retrieval'], help='Category of the dataset')
    args = parser.parse_args()

    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path) if args.output_path else input_dir

    if not input_dir.exists():
        print(f"Input path {input_dir} does not exist.")
        return

    def save_output(dataset_name, unique_tools, task_tools):
        out_dir = output_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "extracted_tools.json"
        
        output_data = {
            "dataset": dataset_name,
            "category": args.category,
            "unique_tools": unique_tools,
            "task_tools": task_tools
        }
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"Processed {dataset_name} for {args.category}: extracted {len(unique_tools)} unique tools across {len(task_tools)} tasks -> saved to {out_file}")

    if args.category in ['insight', 'prediction']:
        for dataset_dir in input_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                json_path = dataset_dir / "merged_output.json"
                if json_path.exists():
                    unique_tools, task_tools = process_insight_prediction(json_path)
                    save_output(dataset_name, unique_tools, task_tools)

    elif args.category == 'ml':
        for dataset_dir in input_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                json_path = dataset_dir / "pure_ml_bench.json"
                if json_path.exists():
                    unique_tools, task_tools = process_ml(json_path)
                    save_output(dataset_name, unique_tools, task_tools)

    elif args.category in ['theory', 'retrieval']:
        for json_path in input_dir.glob("*.json"):
            dataset_name = json_path.stem
            unique_tools, task_tools = process_theory_retrieval(json_path)
            save_output(dataset_name, unique_tools, task_tools)

if __name__ == "__main__":
    main()

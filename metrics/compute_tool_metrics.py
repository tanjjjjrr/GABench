import os
import json
import argparse
from pathlib import Path

def expand_tools(tools, mapping):
    expanded = []
    for tool in tools:
        mapped_tool = mapping.get(tool, tool)
        if isinstance(mapped_tool, list):
            expanded.extend(mapped_tool)
        else:
            expanded.append(mapped_tool)
    return expanded

def compute_lcs(s_agent, s_true_expanded):
    n = len(s_agent)
    m = len(s_true_expanded)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s_agent[i-1] == s_true_expanded[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def main():
    parser = argparse.ArgumentParser(description="Calculate LCS Metric for Predicted Tools vs Ground Truth")
    parser.add_argument('--pred_dir', type=str, required=True, help='Base directory for predictions')
    parser.add_argument('--gt_dir', type=str, required=True, help='Base directory for ground truth')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--category', type=str, required=True, help='Category (e.g., insight, prediction, ml)')
    parser.add_argument('--mapping_file', type=str, default='one-to-one.json', help='Path to one-to-one.json mapping')
    args = parser.parse_args()

    pred_base = Path(args.pred_dir) / args.model_name / f"{args.category}_output"
    
    if not pred_base.exists():
        print(f"Prediction directory {pred_base} does not exist. Cannot proceed.")
        return

    mapping = {}
    if Path(args.mapping_file).exists():
        with open(args.mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    else:
        print(f"Warning: Mapping file {args.mapping_file} not found. Proceeding without mapping.")

    # Iterate over all datasets for the given category
    summary_results = {}
    
    for dataset_dir in pred_base.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset = dataset_dir.name
        gt_file = Path(args.gt_dir) / args.category / "tools" / dataset / "extracted_tools.json"
        
        if not gt_file.exists():
            print(f"Ground truth file {gt_file} not found. Skipping dataset: {dataset}.")
            continue
            
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            
        gt_tasks = gt_data.get("task_tools", [])
        # Ensure ground truth tasks are sorted by their id 0, 1, 2...
        gt_tasks.sort(key=lambda x: x["task_id"])
        
        # Get task directories in prediction (e.g., 0, 1, 2... or 1, 2, 3...)
        task_dirs = []
        for d in dataset_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                task_dirs.append(d)
        
        # Sort task directories numerically to align with ground truth
        task_dirs.sort(key=lambda x: int(x.name))
        
        total_lcs = 0
        total_gt_len = 0
        
        type_lcs = {}
        type_gt_len = {}
        
        task_results = []
        
        for idx, task_dir in enumerate(task_dirs):
            if idx >= len(gt_tasks):
                print(f"Warning: More prediction folders than ground truth tasks for dataset {dataset}. Truncating the rest.")
                break
                
            gt_task = gt_tasks[idx]
            s_true = gt_task.get("tools", [])
            task_type = gt_task.get("task_type", "default")
            
            select_tools_file = task_dir / "select_tools.json"
            s_agent = []
            if select_tools_file.exists():
                with open(select_tools_file, 'r', encoding='utf-8') as f:
                    try:
                        s_agent = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse {select_tools_file}. Assuming empty tools.")
                        s_agent = []
            else:
                print(f"Warning: Prediction file {select_tools_file} not found. Assuming empty tools.")
                
            if not s_agent:
                continue
                
            s_true_expanded = expand_tools(s_true, mapping)
            lcs_len = compute_lcs(s_agent, s_true_expanded)
            gt_len = len(s_true_expanded)
            
            total_lcs += lcs_len
            total_gt_len += gt_len
            
            if task_type not in type_lcs:
                type_lcs[task_type] = 0
                type_gt_len[task_type] = 0
            
            type_lcs[task_type] += lcs_len
            type_gt_len[task_type] += gt_len
            
            task_results.append({
                "pred_task_id": task_dir.name,
                "gt_task_id": gt_task["task_id"],
                "task_type": task_type,
                "lcs_len": lcs_len,
                "gt_len": gt_len,
                "s_agent": s_agent,
                "s_true": s_true_expanded
            })
            
        score = total_lcs / (total_gt_len + 1e-9)
        
        type_scores = {}
        for t in type_lcs:
            type_scores[t] = type_lcs[t] / (type_gt_len[t] + 1e-9)
            
        # Target output directory
        out_dir = Path(args.gt_dir) / args.category / "tools" / dataset / args.model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_file = out_dir / "lcs_score.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset": dataset,
                "model_name": args.model_name,
                "category": args.category,
                "score": score,
                "type_scores": type_scores,
                "total_lcs": total_lcs,
                "total_gt_len": total_gt_len,
                "tasks": task_results
            }, f, indent=4)
            
        print(f"[{args.category} | {dataset} | {args.model_name}] Overall LCS Score = {score:.4f}")
        for t, s in type_scores.items():
            if t != "default":
                print(f"  - {t}: {s:.4f}")
        print(f"  (Saved to {out_file})")
        
        summary_results[dataset] = {
            "overall": score,
            "by_type": type_scores
        }

    if summary_results:
        summary_file = Path(args.gt_dir) / args.category / "tools" / f"{args.model_name}_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "category": args.category,
                "model_name": args.model_name,
                "results": summary_results
            }, f, indent=4)
        print(f"\\nAll summary results saved to {summary_file}")

if __name__ == "__main__":
    main()

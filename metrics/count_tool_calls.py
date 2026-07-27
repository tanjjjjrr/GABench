import os
import json
import argparse
from pathlib import Path
import re

def count_tool_calls(content):
    calls = 0
    if not content:
        return 0

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            if "task_tools" in data:
                for task in data["task_tools"]:
                    calls += len(task.get("tools", []))
                return calls
            else:
                return len(data)
        elif isinstance(data, list):
            return len(data)
    except json.JSONDecodeError:
        pass 

    # Fallback to line counting
    for line in content.split('\n'):
        if line.strip():
            calls += 1
    return calls

def extract_core_id(name):
    match = re.search(r'(?:task_|custom_task_)?0*(\d+)', name)
    if match:
        return match.group(1)
    return None

def main():
    parser = argparse.ArgumentParser(description="Count tool calls from extracted tools json files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the extracted select_tools.json (e.g. metrics_2/...)")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory containing original llm_eval_result.json (e.g. theory_output/...)")
    parser.add_argument("--output_file", type=str, default="count_tool_calls_summary.txt", help="File to save the summary results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return
        
    eval_dir = Path(args.eval_dir)
    output_file_path = Path(args.output_file)
    
    valid_files = []
    skipped_count = 0
    missing_eval_count = 0

    tool_files = list(input_dir.rglob("select_tools.json")) + list(input_dir.rglob("extracted_tools.json"))
    
    if not tool_files:
        print(f"No select_tools.json or extracted_tools.json found in {input_dir}")
        return

    for tool_file in tool_files:
        try:
            with open(tool_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            calls = count_tool_calls(content)
            
            rel_path = tool_file.relative_to(input_dir)
            
            if len(rel_path.parts) >= 2:
                if rel_path.parts[0].endswith("_output") and len(rel_path.parts) >= 3:
                    dataset_name = rel_path.parts[1]
                    task_name = rel_path.parts[2]
                else:
                    dataset_name = rel_path.parts[0]
                    task_name = rel_path.parts[1]
                    
                eval_file = eval_dir / dataset_name / task_name / "llm_eval_result.json"
                
                if not eval_file.exists():
                    dataset_eval_dir = eval_dir / dataset_name
                    if dataset_eval_dir.exists():
                        for child in dataset_eval_dir.iterdir():
                            if not child.is_dir():
                                continue
                            
                            if child.name == task_name or child.name.startswith(task_name + "_") or task_name.startswith(child.name + "_"):
                                possible_eval = child / "llm_eval_result.json"
                                if possible_eval.exists():
                                    eval_file = possible_eval
                                    break
                            
                            if task_name.isdigit():
                                numeric_id = int(task_name)
                                prefixes = [
                                    f"task_{numeric_id:04d}_",
                                    f"task_{numeric_id}_",
                                    f"custom_task_{numeric_id}_"
                                ]
                                if any(child.name.startswith(prefix) for prefix in prefixes):
                                    possible_eval = child / "llm_eval_result.json"
                                    if possible_eval.exists():
                                        eval_file = possible_eval
                                        break
                                        
                            task_core_id = extract_core_id(task_name)
                            child_core_id = extract_core_id(child.name)
                            if task_core_id and child_core_id and task_core_id == child_core_id:
                                possible_eval = child / "llm_eval_result.json"
                                if possible_eval.exists():
                                    eval_file = possible_eval
                                    break

            else:
                dataset_name = input_dir.name
                eval_file = eval_dir / "llm_eval_result.json"
                
            sub_category = "overall"
            
            if eval_file.exists():
                try:
                    with open(eval_file, 'r', encoding='utf-8') as ef:
                        eval_data = json.load(ef)
                        sc = eval_data.get("sub_category", "")
                        if sc and str(sc).strip().lower() not in ["", "default", "overall", "none"]:
                            sub_category = str(sc).strip()
                except Exception:
                    pass
            else:
                missing_eval_count += 1
                
            valid_files.append({
                "dataset": dataset_name,
                "sub_category": sub_category,
                "calls": calls
            })
            
        except Exception as e:
            print(f"Warning: Failed to read {tool_file}: {e}")

    # Aggregate by dataset and sub_category
    dataset_stats = {}
    
    for item in valid_files:
        ds = item["dataset"]
        sc = item["sub_category"]
        
        if ds not in dataset_stats:
            dataset_stats[ds] = {}
        if sc not in dataset_stats[ds]:
            dataset_stats[ds][sc] = {
                "file_count": 0,
                "calls": 0
            }
            
        dataset_stats[ds][sc]["file_count"] += 1
        dataset_stats[ds][sc]["calls"] += item["calls"]

    total_tool_calls = sum(item["calls"] for item in valid_files)
    file_count = len(valid_files)

    # Generate report
    report_lines = []
    report_lines.append("=================================================")
    report_lines.append("           Tool Calls Summary Report             ")
    report_lines.append("=================================================")
    report_lines.append(f"Tools directory     : {input_dir}")
    report_lines.append(f"Eval directory      : {eval_dir}")
    report_lines.append(f"Total files counted : {file_count}")
    report_lines.append(f"Total tool calls    : {total_tool_calls}")
    if missing_eval_count > 0:
        report_lines.append(f"Warning             : {missing_eval_count} tool files lacked a paired llm_eval_result.json")
    report_lines.append("-------------------------------------------------")
    report_lines.append("[Breakdown by Dataset & Sub-Category]")
    
    for ds in sorted(dataset_stats.keys()):
        ds_file_count = 0
        ds_calls = 0
        for sc, stats in dataset_stats[ds].items():
            ds_file_count += stats["file_count"]
            ds_calls += stats["calls"]
            
        report_lines.append(f"\nDataset: {ds} (Valid files: {ds_file_count})")
        if ds_file_count > 0:
            report_lines.append(f" - Overall : Total calls: {ds_calls}")
            
        sub_cats = dataset_stats[ds]
        if not (len(sub_cats) == 1 and "overall" in sub_cats):
            for sc, stats in sorted(sub_cats.items()):
                report_lines.append(f"   -> {sc:10}: (n={stats['file_count']:3}) Total calls: {stats['calls']}")

    report_lines.append("\n=================================================")

    report_content = "\n".join(report_lines)
    
    print(report_content)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(report_content + "\n")
    print(f"Summary successfully saved to: {output_file_path.absolute()}")

if __name__ == "__main__":
    main()

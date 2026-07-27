import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Calculate average usage metrics from usage.json files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory (e.g., xxx)")
    parser.add_argument("--output_file", type=str, default="compute_usage_summary.txt", help="File to save the summary results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        print(f"Error: Directory '{output_dir}' does not exist or is not a directory.")
        return

    output_file_path = Path(args.output_file)

    valid_usages = []
    skipped_count = 0
    missing_eval_count = 0
    
    # Recursively find all usage.json files
    for usage_file in output_dir.rglob("usage.json"):
        try:
            with open(usage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            input_tokens = data.get("input_tokens", 0)
            output_tokens = data.get("output_tokens", 0)
            cache_read_tokens = data.get("cache_read_tokens", 0)
            elapsed_time = data.get("elapsed_time", 0.0)
            
            # Get the path relative to output_dir.
            rel_path = usage_file.relative_to(output_dir)
            if len(rel_path.parts) >= 2:
                dataset_name = rel_path.parts[0]
                task_name = rel_path.parts[1]
                eval_file = output_dir / dataset_name / task_name / "llm_eval_result.json"
            else:
                dataset_name = output_dir.name
                eval_file = usage_file.parent / "llm_eval_result.json"
                
            sub_category = "overall"
            
            if eval_file.exists():
                try:
                    with open(eval_file, 'r', encoding='utf-8') as ef:
                        eval_data = json.load(ef)
                        sc = eval_data.get("sub_category", "")
                        # Keep only non-default subcategories; otherwise use overall.
                        if sc and str(sc).strip().lower() not in ["", "default", "overall", "none"]:
                            sub_category = str(sc).strip()
                except Exception:
                    pass
            else:
                missing_eval_count += 1
            
            # Skip records with fewer than 2000 input tokens.
            if input_tokens >= 2000:
                valid_usages.append({
                    "dataset": dataset_name,
                    "sub_category": sub_category,
                    "file": str(usage_file),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "elapsed_time": elapsed_time
                })
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"Warning: Failed to read or parse {usage_file}: {e}")

    if not valid_usages:
        print(f"No valid usage.json files found. (Skipped {skipped_count} files with input_tokens < 2000).")
        return

    # Aggregate by dataset and sub_category
    dataset_stats = {}
    
    for u in valid_usages:
        ds = u["dataset"]
        sc = u["sub_category"]
        
        if ds not in dataset_stats:
            dataset_stats[ds] = {}
        if sc not in dataset_stats[ds]:
            dataset_stats[ds][sc] = {
                "count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "elapsed_time": 0.0
            }
            
        dataset_stats[ds][sc]["count"] += 1
        dataset_stats[ds][sc]["input_tokens"] += u["input_tokens"]
        dataset_stats[ds][sc]["output_tokens"] += u["output_tokens"]
        dataset_stats[ds][sc]["cache_read_tokens"] += u["cache_read_tokens"]
        dataset_stats[ds][sc]["elapsed_time"] += u["elapsed_time"]

    # Overall stats for all valid usages
    total_input = sum(u["input_tokens"] for u in valid_usages)
    total_output = sum(u["output_tokens"] for u in valid_usages)
    total_cache_read = sum(u["cache_read_tokens"] for u in valid_usages)
    total_time = sum(u["elapsed_time"] for u in valid_usages)
    count = len(valid_usages)
    
    avg_input = total_input / count if count else 0
    avg_output = total_output / count if count else 0
    avg_cache_read = total_cache_read / count if count else 0
    avg_time = total_time / count if count else 0

    # Generate report
    report_lines = []
    report_lines.append("=================================================")
    report_lines.append("           Token Usage Summary Report            ")
    report_lines.append("=================================================")
    report_lines.append(f"Processed directory : {output_dir}")
    report_lines.append(f"Valid files counted : {count}")
    report_lines.append(f"Files skipped       : {skipped_count} (input_tokens < 2000)")
    if missing_eval_count > 0:
        report_lines.append(f"Warning             : {missing_eval_count} usage.json files lacked a paired llm_eval_result.json")
    report_lines.append("-------------------------------------------------")
    report_lines.append("[Overall Average Metrics]")
    report_lines.append(f"  Average input_tokens     : {avg_input:.2f}")
    report_lines.append(f"  Average output_tokens    : {avg_output:.2f}")
    report_lines.append(f"  Average cache_read_tokens: {avg_cache_read:.2f}")
    report_lines.append(f"  Average elapsed_time     : {avg_time:.2f}s")
    report_lines.append("-------------------------------------------------")
    report_lines.append("[Breakdown by Dataset & Sub-Category]")
    
    for ds in sorted(dataset_stats.keys()):
        # Calculate dataset overall
        ds_total_count = 0
        ds_input = 0
        ds_output = 0
        ds_cache = 0
        ds_time = 0
        for sc, stats in dataset_stats[ds].items():
            ds_total_count += stats["count"]
            ds_input += stats["input_tokens"]
            ds_output += stats["output_tokens"]
            ds_cache += stats["cache_read_tokens"]
            ds_time += stats["elapsed_time"]
            
        report_lines.append(f"\nDataset: {ds} (Valid files: {ds_total_count})")
        if ds_total_count > 0:
            report_lines.append(f" - Overall : Avg input: {ds_input/ds_total_count:.2f} | Avg output: {ds_output/ds_total_count:.2f} | Avg cache: {ds_cache/ds_total_count:.2f} | Avg time: {ds_time/ds_total_count:.2f}s")
            
        sub_cats = dataset_stats[ds]
        # Only print sub_categories if there's actually a breakdown, or if the only sub_category isn't 'overall'
        if not (len(sub_cats) == 1 and "overall" in sub_cats):
            for sc, stats in sorted(sub_cats.items()):
                c = stats["count"]
                if c > 0:
                    report_lines.append(f"   -> {sc:10}: (n={c:3}) Avg input: {stats['input_tokens']/c:.2f} | Avg output: {stats['output_tokens']/c:.2f} | Avg cache: {stats['cache_read_tokens']/c:.2f} | Avg time: {stats['elapsed_time']/c:.2f}s")

    report_lines.append("\n=================================================")

    report_content = "\n".join(report_lines)
    
    # Print to terminal
    print(report_content)
    
    # Save to txt file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(report_content + "\n")
    print(f"Summary successfully saved to: {output_file_path.absolute()}")

if __name__ == "__main__":
    main()

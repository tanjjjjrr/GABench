import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Calculate pass rate from llm_eval_result.json files, grouping by sub-category and ignoring scores of -1.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory (e.g., insight_output)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        print(f"Error: Directory '{output_dir}' does not exist or is not a directory.")
        return

    # Dictionary structure: { dataset: { sub_category: {"valid": 0, "passed": 0, "ignored": 0} } }
    stats = {}
    
    # Recursively find all llm_eval_result.json files
    # The structure is assumed to be <output_dir>/<dataset>/<task_id>/llm_eval_result.json
    for result_file in output_dir.rglob("llm_eval_result.json"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            score = data.get("score")
            sub_category = data.get("sub_category", "default")
            dataset = result_file.parent.parent.name
            
            if dataset not in stats:
                stats[dataset] = {}
            if sub_category not in stats[dataset]:
                stats[dataset][sub_category] = {"valid": 0, "passed": 0, "ignored": 0}
                
            if score == -1:
                stats[dataset][sub_category]["ignored"] += 1
            elif score is not None:
                stats[dataset][sub_category]["valid"] += 1
                if score == 1:
                    stats[dataset][sub_category]["passed"] += 1
                
        except Exception as e:
            print(f"Warning: Failed to read or parse {result_file}: {e}")

    print(f"--- Evaluation Statistics ---")
    print(f"Processed directory : {output_dir}")
    
    overall_valid = 0
    overall_passed = 0
    overall_ignored = 0
    
    for dataset, sub_cats in stats.items():
        for sub_cat, counts in sub_cats.items():
            overall_valid += counts["valid"]
            overall_passed += counts["passed"]
            overall_ignored += counts["ignored"]
        
    print(f"\n=============================")
    print(f"[Overall Results (All Datasets)]")
    print(f"Total valid tasks   : {overall_valid}")
    print(f"Total passed tasks  : {overall_passed}")
    print(f"Ignored (-1 score)  : {overall_ignored}")
    if overall_valid > 0:
        print(f"Overall Pass rate   : {(overall_passed / overall_valid) * 100:.2f}%")
    else:
        print("Overall Pass rate   : N/A")
        
    if len(stats) > 0:
        print(f"\n=============================")
        print(f"[Results by Dataset & Sub-Category]")
        for dataset, sub_cats in sorted(stats.items()):
            print(f"\nDataset: {dataset}")
            dataset_valid = sum(c["valid"] for c in sub_cats.values())
            dataset_passed = sum(c["passed"] for c in sub_cats.values())
            dataset_ignored = sum(c["ignored"] for c in sub_cats.values())
            
            if dataset_valid > 0:
                print(f" - Overall : {(dataset_passed / dataset_valid) * 100:6.2f}% (Passed: {dataset_passed}/{dataset_valid}, Ignored: {dataset_ignored})")
            else:
                print(f" - Overall :    N/A (Passed: 0/0, Ignored: {dataset_ignored})")
                
            for sub_cat, counts in sorted(sub_cats.items()):
                valid = counts["valid"]
                passed = counts["passed"]
                ignored = counts["ignored"]
                if valid > 0:
                    rate = (passed / valid) * 100
                    print(f"   -> {sub_cat:10}: {rate:6.2f}% (Passed: {passed}/{valid}, Ignored: {ignored})")
                else:
                    print(f"   -> {sub_cat:10}:    N/A (Passed: 0/0, Ignored: {ignored})")

if __name__ == "__main__":
    main()

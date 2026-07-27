import json
import argparse
import sys

def process_jsons(file1_path, file2_path, output_path):
    # 1. Read 1.json
    print(f"Reading {file1_path}...")
    with open(file1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
        
    # 2. Read 2.json
    print(f"Reading {file2_path}...")
    with open(file2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
        
    # 3. Filter tasks in 1.json
    original_tasks = data1.get('tasks', [])
    original_count = len(original_tasks)
    
    filtered_tasks = []
    for task in original_tasks:
        req_tools = task.get('required_tools', [])
        # Skip (remove) the task when required_tools contains "simulate_edge_perturbation"
        if "simulate_edge_perturbation" in req_tools:
            continue
        filtered_tasks.append(task)
        
    filtered_count = original_count - len(filtered_tasks)
    print(f"Filtered out {filtered_count} tasks with 'simulate_edge_perturbation' from {file1_path}.")
    
    # 4. Append the tasks from 2.json to the end of 1.json
    tasks2_to_add = data2.get('tasks', [])
    filtered_tasks.extend(tasks2_to_add)
    
    # 5. Update data1
    data1['tasks'] = filtered_tasks
    
    # Also update the num_tasks_generated field when present
    if 'generation_info' in data1 and 'num_tasks_generated' in data1['generation_info']:
        data1['generation_info']['num_tasks_generated'] = len(filtered_tasks)
        
    # 6. Save to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data1, f, indent=4, ensure_ascii=False)
        
    print(f"Appended {len(tasks2_to_add)} tasks from {file2_path}.")
    print(f"Total tasks in {output_path}: {len(filtered_tasks)}")
    print(f"Successfully saved to {output_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and merge JSON tasks")
    parser.add_argument("file1", help="Path to the first JSON file (tasks to be filtered)")
    parser.add_argument("file2", help="Path to the second JSON file (tasks to append)")
    parser.add_argument("-o", "--output", default="merged_output.json", help="Path to the output JSON file")
    
    args = parser.parse_args()
    
    process_jsons(args.file1, args.file2, args.output)

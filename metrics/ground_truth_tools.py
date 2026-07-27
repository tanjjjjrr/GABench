import json
import os
import re
import argparse
from pathlib import Path

def extract_invoked_tools(chat_file):
    """
    Extracts all the tools called by the agent from the chat.jsonl file.
    It looks for 'toolCall' in the assistant's message content.
    For 'exec' or 'shell' tool calls, it also extracts the specific python script/command.
    """
    invoked_tools = []
    try:
        with open(chat_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                # In python, json.loads instead of json.load for strings
                data = json.loads(line.strip())
                if data.get("type") == "message" and data.get("message", {}).get("role") == "assistant":
                    content = data["message"].get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "toolCall":
                                tool_name = item.get("name")
                                args = item.get("arguments", {})
                                
                                # Add the base tool name (e.g., 'exec', 'write', 'read')
                                invoked_tools.append(tool_name)
                                
                                # For shell/exec, extract the actual command being run
                                if tool_name in ["exec", "shell", "execute_bash_command"]:
                                    cmd = args.get("command", "")
                                    if "graph_theory_tools.py" in cmd:
                                        invoked_tools.append("graph_theory_tools")
                                        # extract sub-command if possible
                                        parts = cmd.split("graph_theory_tools.py")
                                        if len(parts) > 1 and parts[1].strip():
                                            invoked_tools.append(parts[1].strip().split()[0])
                                    elif "graph_ml_tools.py" in cmd:
                                        invoked_tools.append("graph_ml_tools")
                                        parts = cmd.split("graph_ml_tools.py")
                                        if len(parts) > 1 and parts[1].strip():
                                            invoked_tools.append(parts[1].strip().split()[0])
                                    elif "graph_insight_tools.py" in cmd:
                                        invoked_tools.append("graph_insight_tools")
                                        parts = cmd.split("graph_insight_tools.py")
                                        if len(parts) > 1 and parts[1].strip():
                                            invoked_tools.append(parts[1].strip().split()[0])
    except Exception as e:
        print(f"Error reading {chat_file}: {e}")
    return invoked_tools

def extract_final_answer(chat_file):
    """
    Extract the final answer from the agent.
    Often, the agent writes the final answer to a file (like final_result.json) or outputs it in the last text block.
    This is a heuristic extraction.
    """
    final_text = ""
    try:
        with open(chat_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line.strip())
                if data.get("type") == "message" and data.get("message", {}).get("role") == "assistant":
                    content = data["message"].get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                final_text = item.get("text")
                            elif item.get("type") == "toolCall" and item.get("name") == "write":
                                # Sometimes the agent writes the final result using the write tool
                                args = item.get("arguments", {})
                                if "final_result.json" in args.get("file_path", ""):
                                    final_text = args.get("content", "")
    except Exception as e:
        print(f"Error reading {chat_file}: {e}")
    return final_text

def evaluate(chat_file, ground_truth_file):
    # 1. Load ground truth
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    # Assuming there's only one task in the ground truth for this chat.jsonl
    # Adjust logic if chat.jsonl contains multiple tasks or if you need to match by ID
    nodes = gt_data.get("node", [])
    if not nodes:
        print("No tasks found in ground truth.")
        return
    
    task_gt = nodes[0] # taking the first task as an example
    required_tools = set(task_gt.get("required_tools", []))
    expected_answer = task_gt.get("ground_truth", {})
    
    print(f"--- Task ID: {task_gt.get('id')} ---")
    
    # 2. Extract tools and calculate Tool-Select Appropriation
    invoked_tools_list = extract_invoked_tools(chat_file)
    
    # Save the extracted tools to select_tools.json in order
    select_tools_path = "select_tools.json"
    with open(select_tools_path, "w", encoding="utf-8") as f:
        json.dump(invoked_tools_list, f, indent=4, ensure_ascii=False)
    print(f"\\n[Storage] Extracted tools have been saved to {select_tools_path} in order.")
    
    # For evaluation, we convert it back to a set
    invoked_tools = set(invoked_tools_list)
    
    print(f"\\n[Tool Extraction]")
    print(f"Required Tools: {required_tools}")
    print(f"Invoked Tools: {invoked_tools}")
    
    # Calculate Tool-Select Appropriation (Recall: how many required tools were actually invoked)
    matched_tools = 0
    for req_tool in required_tools:
        if any(req_tool.lower() in inv.lower() or inv.lower() in req_tool.lower() for inv in invoked_tools):
            matched_tools += 1
            
    tsa_score = (matched_tools / len(required_tools)) * 100 if required_tools else 100
    print(f"=> Tool-Select Appropriation (TSA): {tsa_score:.2f}% ({matched_tools}/{len(required_tools)} matched)")

    # 3. Extract final answer and calculate Success Rate
    final_output = extract_final_answer(chat_file)
    
    print(f"\\n[Success Rate Evaluation]")
    print(f"Expected Answer: {expected_answer}")
    
    is_success = False
    
    if isinstance(expected_answer, dict) and "answer" in expected_answer:
        ans = str(expected_answer["answer"])
        if ans in final_output:
            is_success = True
    elif isinstance(expected_answer, bool):
        ans_str = str(expected_answer).lower()
        if ans_str in final_output.lower():
             is_success = True
    else:
        if str(expected_answer) in final_output:
             is_success = True
             
    sr_score = 100 if is_success else 0
    print(f"=> Success Rate (SR): {sr_score:.2f}% (Found expected answer in final output: {is_success})")


def process_directory(input_dir, output_dir, model_name=None, task_name=None):
    input_path = Path(input_dir)
    output_base = Path(output_dir)
    
    if not model_name or not task_name:
        if len(input_path.parts) >= 3:
            inferred_model = f"{input_path.parts[-2]}/{input_path.parts[-1]}"
            inferred_task = input_path.parts[-3]
            model_name = model_name or inferred_model
            task_name = task_name or inferred_task
        else:
            model_name = model_name or "unknown_model"
            task_name = task_name or "unknown_task"

    print(f"Using Model Name: {model_name}")
    print(f"Using Task Name: {task_name}")

    count = 0
    # traverse datasets
    for dataset_dir in input_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        # traverse task directories
        for task_dir in dataset_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            task_id_raw = task_dir.name
            match = re.search(r'_(\d+)(?:_|$)', task_id_raw)
            if match:
                task_id = str(int(match.group(1)))
            else:
                match = re.search(r'\d+', task_id_raw)
                task_id = str(int(match.group(0))) if match else task_id_raw
            
            chat_file = task_dir / "results" / "chat.jsonl"
            if not chat_file.exists():
                chat_file = task_dir / "chat.jsonl"
            
            if not chat_file.exists():
                continue
                
            invoked_tools_list = extract_invoked_tools(chat_file)
            
            # target: output_dir / model_name / task_name / dataset_name / task_id / select_tools.json
            dataset_name = dataset_dir.name
            out_file = output_base / model_name / task_name / dataset_name / task_id / "select_tools.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(invoked_tools_list, f, indent=4, ensure_ascii=False)
            
            count += 1
            print(f"Saved {out_file}")
            
    print(f"Processed {count} tasks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # New arguments for directory processing
    parser.add_argument("--input_dir", type=str, help="Input directory (e.g., xxx)")
    parser.add_argument("--output_dir", type=str, help="Target storage directory")
    parser.add_argument("--model_name", type=str, default=None, help="Optional: override model name")
    parser.add_argument("--task_name", type=str, default=None, help="Optional: override task name")
    
    # Keep old arguments optional for backward compatibility
    parser.add_argument("--chat_file", type=str, help="Path to chat.jsonl")
    parser.add_argument("--gt_file", type=str, help="Path to ground_truth.json")
    args = parser.parse_args()
    
    if args.input_dir and args.output_dir:
        process_directory(args.input_dir, args.output_dir, args.model_name, args.task_name)
    elif args.chat_file and args.gt_file:
        evaluate(args.chat_file, args.gt_file)
    else:
        parser.print_help()
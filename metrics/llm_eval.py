import os
import json
import argparse
import requests
import re
from pathlib import Path

def extract_agent_output(chat_file):
    """
    Extracts the user prompt, final JSON output (if any), and final text output from chat.jsonl
    """
    agent_final_json = None
    agent_final_text = None
    user_prompt = None

    if not os.path.exists(chat_file):
        return None, None, None

    with open(chat_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1. Extract user prompt (first user message)
    for line in lines:
        try:
            data = json.loads(line)
            if data.get("type") == "message" and data.get("message", {}).get("role") == "user":
                content = data["message"].get("content", [])
                for c in content:
                    if c.get("type") == "text":
                        user_prompt = c["text"]
                        break
                if user_prompt:
                    break
        except Exception:
            continue

    # 2. Extract agent final text directly from agent.log
    agent_log_file = Path(chat_file).parent / "agent.log"
    if agent_log_file.exists():
        try:
            with open(agent_log_file, 'r', encoding='utf-8') as f:
                agent_final_text = f.read()
        except Exception:
            agent_final_text = None
    else:
        agent_final_text = None
            
    return user_prompt, agent_final_text


def mask_ground_truth(data):
    """
    Masks large list contents in ground truth if the key contains specific keywords.
    This prevents the LLM prompt from exceeding length limits while retaining behavior checks.
    """
    mask_keywords = ['subgraph', 'filter', 'boundary', 'nodes', 'edges', 'perimeter']
    
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if any(word in k.lower() for word in mask_keywords):
                if isinstance(v, list):
                    new_dict[k] = f"[MASKED_LIST of length {len(v)}: CHECK_BEHAVIOR_ONLY]"
                elif isinstance(v, dict):
                    new_dict[k] = mask_ground_truth(v)
                else:
                    new_dict[k] = v
            else:
                new_dict[k] = mask_ground_truth(v)
        return new_dict
    elif isinstance(data, list):
        return [mask_ground_truth(item) for item in data]
    else:
        return data


def call_llm(prompt, model_name, api_url, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a fair and strict evaluator for an AI agent's performance. You output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        print(f"LLM API Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response text: {response.text}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent output with LLM")
    parser.add_argument('--task', type=str, required=True, help="Category: prediction, insight, ml, theory, retrieval")
    parser.add_argument('--input_path', type=str, required=True, help="Path to chat.jsonl")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save evaluation result JSON")
    parser.add_argument('--gt_file', type=str, required=True, help="Path to ground truth JSON file")
    parser.add_argument('--model_name', type=str, required=True, help="LLM model name (e.g., xxx)")
    parser.add_argument('--api_url', type=str, required=True, help="LLM API Base URL (e.g., xxx)")
    parser.add_argument('--api_key', type=str, required=True, help="LLM API Key")
    args = parser.parse_args()

    user_prompt, agent_text = extract_agent_output(args.input_path)
    
    # --- 1. Parse task_id from input_path ---
    input_path_obj = Path(args.input_path)
    task_dir_name = input_path_obj.parent.name
    if task_dir_name == "results":
        task_dir_name = input_path_obj.parent.parent.name
        
    match = re.search(r'_(\d+)(?:_|$)', task_dir_name)
    if match:
        parsed_task_id = int(match.group(1))
    else:
        match = re.search(r'\d+', task_dir_name)
        parsed_task_id = int(match.group(0)) if match else None
        
    # --- 2. Match Ground Truth and Determine Sub-category ---
    matched_gt = None
    task_sub_category = "default"
    
    if os.path.exists(args.gt_file):
        with open(args.gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            
        if parsed_task_id is not None:
            formatted_id = f"task_{parsed_task_id:04d}"
            
            def is_match(t):
                # ML case
                if "task_index" in t and t["task_index"] == parsed_task_id:
                    return True
                # Insight/Theory case
                if "id" in t and (t["id"] == formatted_id or str(parsed_task_id) in str(t["id"])):
                    return True
                # Fallback: some GTs might use 'task_id'
                if "task_id" in t and (t["task_id"] == formatted_id or str(parsed_task_id) in str(t["task_id"])):
                    return True
                return False

            # Case 1: "tasks" array (insight/prediction)
            if "tasks" in gt_data and isinstance(gt_data["tasks"], list):
                for t in gt_data["tasks"]:
                    if is_match(t):
                        matched_gt = t.get("ground_truth", t)
                        break
            
            # Case 2: "benchmark" array (ml)
            if not matched_gt and "benchmark" in gt_data and isinstance(gt_data["benchmark"], list):
                for t in gt_data["benchmark"]:
                    if is_match(t):
                        matched_gt = t.get("ground_truth", t)
                        task_sub_category = t.get("task_type", "default")
                        break
                        
            # Case 3: "node", "edge", "graph" dict (theory/retrieval)
            if not matched_gt:
                for sub_cat in ["node", "edge", "graph"]:
                    if sub_cat in gt_data and isinstance(gt_data[sub_cat], list):
                        for t in gt_data[sub_cat]:
                            if is_match(t):
                                matched_gt = t.get("ground_truth", t)
                                task_sub_category = sub_cat
                                break
                    if matched_gt:
                        break
        
        # If no match or structure is different, use the entire file content as GT
        if not matched_gt:
            matched_gt = gt_data
    else:
        print(f"Warning: Ground truth file not found at {args.gt_file}")
        matched_gt = {}

    # --- 2. Check extraction success based on task rules ---
    # Check if agent.log was missing
    if agent_text is None:
        print(f"agent.log not found for {args.task} task. Assigning Score = -1.")
        result = {"score": -1, "reason": "agent.log not found in the task directory.", "sub_category": task_sub_category}
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return

    # Check for request error in the final text
    if any(err in agent_text.lower() for err in ["llm request error", "request error", "api error", "connection error", "timeout", "bad request"]):
        print(f"Agent encountered a request error. Assigning Score = -1.")
        result = {"score": -1, "reason": "Agent output contains request error or similar API failure.", "sub_category": task_sub_category}
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return

    # 1. Check extraction success
    if not agent_text.strip():
        print(f"agent.log is empty for {args.task} task. Assigning Score = 0.")
        result = {"score": 0, "reason": "agent.log is empty.", "sub_category": task_sub_category}
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return
    else:
        agent_output_to_eval = agent_text

    masked_gt = mask_ground_truth(matched_gt)
    
    # 3. Construct LLM Prompt
    prompt = f"""
You are evaluating the final output of an AI agent solving a graph/dataset task.

Task Category: {args.task}
User Request Prefix: {user_prompt[:500] if user_prompt else 'N/A'}...

Ground Truth:
Note: Large lists (like 'subgraph_nodes' or 'filtered_nodes') in the ground truth are masked with '[MASKED_LIST...]' to save space. 
For masked fields, just verify if the agent's output shows evidence of correctly completing that step/behavior (e.g., if the agent output has the key and a list). Do not penalize for exact element mismatch if the elements are masked.
{json.dumps(masked_gt, indent=2, ensure_ascii=False)}

Agent's Final Output:
{agent_output_to_eval}

Evaluate the agent's output against the Ground Truth.
Assign a score of either 1 (Pass) or 0 (Fail). If the agent's output correctly matches our ground truth (Pass), assign 1. If it does not match (Fail), assign 0. If the agent's output indicates an API request error, connection error, or similar failure, assign -1.

Respond ONLY with a valid JSON object in this format:
{{
  "score": <int, 1, 0, or -1>,
  "reason": "<string explaining the evaluation detail>"
}}
"""

    print("Calling LLM for evaluation...")
    eval_result = call_llm(prompt, args.model_name, args.api_url, args.api_key)
    
    if eval_result:
        eval_result["sub_category"] = task_sub_category
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)
        print(f"✅ Evaluation completed!")
        print(f"Score: {eval_result.get('score')} (Sub-category: {task_sub_category})")
        print(f"Reason: {eval_result.get('reason')}")
        print(f"Result saved to {args.output_path}")
    else:
        print("❌ Evaluation failed due to LLM API error.")

if __name__ == '__main__':
    main()

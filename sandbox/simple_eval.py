import argparse
import json
import os
import shutil
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv

# Import the required project modules.
from src.agents.xxx.runner import xxxAgent
from src.agents.xxx.runner import xxxAgent
from src.agents.xxx.runner import xxxAgent
from src.agents.xxx.runner import xxxAgent
from src.agents.base import AgentTaskSpec
from src.utils.docker_utils import remove_container, collect_output_from_container

def main():
    parser = argparse.ArgumentParser(description="Simplified WildClawBench runner for custom inputs")
    parser.add_argument("--pt_file", type=str, required=True, help="Path to the .pt dataset file")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the .json questions file")
    parser.add_argument("--intermediate_pt_file", type=str, default="", help="Optional path to intermediate ground truth masked data")
    parser.add_argument("--model", type=str, default="xxx", help="Model to evaluate")
    parser.add_argument("--api_url", type=str, default="", help="Custom xxx API URL")
    parser.add_argument("--api_key", type=str, default="", help="Custom xxx API Key")
    parser.add_argument("--skills_dir", type=str, default="", help="Optional local directory containing custom xxx skills")
    parser.add_argument("--agent", type=str, default="xxx", choices=["xxx", "xxx", "xxx", "xxx"], help="Which agent harness to use")
    parser.add_argument("--output_dir", type=str, default="output/simple_eval", help="Output directory")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of questions to evaluate (0 means no limit)")
    parser.add_argument("--port", type=int, default=18789, help="Base port for container bindings")
    args = parser.parse_args()

    # Initialize environment variables so values such as xxx_API_KEY are available.
    load_dotenv()
    
    import threading
    data_prep_lock = threading.Lock()

    pt_file = Path(args.pt_file).resolve()
    json_file = Path(args.json_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not pt_file.exists():
        raise FileNotFoundError(f"PT file not found: {pt_file}")
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
        
    # 1. Extract questions from the JSON data.
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Recursively collect objects containing a "question" field.
    def extract_questions(obj):
        questions = []
        if isinstance(obj, dict):
            if "question" in obj:
                questions.append(obj)
            else:
                for v in obj.values():
                    questions.extend(extract_questions(v))
        elif isinstance(obj, list):
            for item in obj:
                questions.extend(extract_questions(item))
        return questions

    items = extract_questions(data)
    
    # If none are found, fall back to treating the original object as a task.
    if not items:
        if isinstance(data, list):
            items = data
        else:
            items = [data]

    if args.limit > 0:
        items = items[:args.limit]

    # Select the API key and base URL, preferring CLI arguments over environment variables.
    api_key = args.api_key if args.api_key else os.environ.get("xxx_API_KEY", "")
    api_url = args.api_url if args.api_url else os.environ.get("xxx_BASE_URL", "xxx")

    intermediate_data = None
    if args.intermediate_pt_file:
        intermediate_pt_file = Path(args.intermediate_pt_file).resolve()
        if intermediate_pt_file.exists():
            import torch
            intermediate_data = torch.load(intermediate_pt_file, weights_only=False)
            # --- Diagnostic: print the intermediate_data structure to troubleshoot indexing issues. ---
            print(f"[DiagInfo] intermediate_data type: {type(intermediate_data).__name__}")
            if isinstance(intermediate_data, (list, tuple)):
                print(f"[DiagInfo] intermediate_data length: {len(intermediate_data)}")
                if len(intermediate_data) > 0:
                    print(f"[DiagInfo] intermediate_data[0] type: {type(intermediate_data[0]).__name__}")
            elif isinstance(intermediate_data, dict):
                print(f"[DiagInfo] intermediate_data keys: {list(intermediate_data.keys())[:10]}")
            else:
                print(f"[DiagInfo] intermediate_data is a single object (not list/dict), cannot be indexed by task.")
        else:
            print(f"Warning: intermediate_pt_file does not exist: {intermediate_pt_file}")

    # Force the underlying image name to sandbox_{agent_name}.
    agent_name = args.agent.lower()
    custom_image_name = f"sandbox_{agent_name}:latest"

    # Select and initialize the requested agent from the arguments.
    if agent_name == "xxx":
        agent = xxxAgent(
            image=custom_image_name,
            gateway_port=18789, 
            xxx_api_key=api_key,
            xxx_base_url=api_url
        )
    elif agent_name == "xxx":
        agent = xxxAgent(
            image=custom_image_name,
            xxx_api_key=api_key,
            xxx_base_url=api_url
        )
    elif agent_name == "xxx":
        agent = xxxAgent(
            image=custom_image_name,
            xxx_api_key=api_key,
            xxx_base_url=api_url
        )
    elif agent_name == "xxx":
        agent = xxxAgent(
            image=custom_image_name,
            xxx_api_key=api_key,
            xxx_base_url=api_url
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    
    def process_task(i, item):
        question = item.get("question", "") if isinstance(item, dict) else str(item)
        if not question:
            return
            
        base_id = item.get("id", f"custom_task_{i}") if isinstance(item, dict) else f"custom_task_{i}"
        task_id = f"{base_id}_{uuid.uuid4().hex[:6]}"
        print(f"\n{'='*50}")
        print(f"--- Starting task: {task_id} ---")
        print(f"Question: {question}")
        print(f"{'='*50}")
        
        # --- Infer the task type early. ---
        task_type = None
        if args.intermediate_pt_file and pt_file.name in ["hiv.pt", "pcba.pt"]:
            task_type = "GC"
        elif "task_type" in item:
            task_type = item.get("task_type")
        elif "ground_truth" in item and isinstance(item["ground_truth"], dict):
            gt_dict = item["ground_truth"]
            if "structural_node_embedding_classification" in gt_dict:
                task_type = "NC"
            elif "structural_node_embedding_link_prediction" in gt_dict:
                task_type = "LP"
            elif "structural_graph_embedding_classification" in gt_dict:
                task_type = "GC"

        # 2. Prepare a dedicated workspace for the Docker mount.
        task_workspace = output_dir / task_id / "workspace"
        task_workspace_exec = task_workspace / "exec"
        task_workspace_exec.mkdir(parents=True, exist_ok=True)
        
        # --- Resolve the task's actual index in intermediate_data. ---
        def resolve_task_index(item, fallback_i):
            if isinstance(item, dict):
                if "task_index" in item:
                    return int(item["task_index"])
                item_id = item.get("id", "")
                if item_id:
                    import re
                    m = re.search(r'(\d+)$', item_id)
                    if m:
                        return int(m.group(1)) - 1
            return fallback_i
        
        # Copy the user's PT dataset file (or its modified version) into workspace/exec for the Docker mount.
        with data_prep_lock:
            if intermediate_data is not None:
                try:
                    import torch
                    import traceback
                    task_idx = resolve_task_index(item, i)
                    torch.set_num_threads(1)  # Prevent OpenMP deadlocks during multithreaded execution.
                    print(f"[DiagInfo] Task {i}: resolved task_idx={task_idx}", flush=True)

                    # --- Support both list and dict intermediate_data structures. ---
                    def lookup_inter_item(data, idx):
                        if isinstance(data, (list, tuple)):
                            if idx < 0 or idx >= len(data):
                                raise IndexError(
                                    f"task_idx={idx} is out of bounds for intermediate_data "
                                    f"of length {len(data)}."
                                )
                            return data[idx]
                        elif isinstance(data, dict):
                            # Try the integer key first, then the string key.
                            if idx in data:
                                return data[idx]
                            if str(idx) in data:
                                return data[str(idx)]
                            sample_keys = list(data.keys())[:5]
                            raise KeyError(
                                f"task_idx={idx} (type={type(idx).__name__}) not found in intermediate_data dict. "
                                f"Key type={type(sample_keys[0]).__name__}, sample keys={sample_keys}"
                            )
                        else:
                            raise TypeError(
                                f"intermediate_data type {type(data).__name__} not supported, expected list or dict"
                            )

                    print(f"[DiagInfo] Task {i}: Looking up inter_item...", flush=True)
                    inter_item = lookup_inter_item(intermediate_data, task_idx)
                    print(f"[DiagInfo] Task {i}: Loading pt_file from disk...", flush=True)
                    pt_obj = torch.load(pt_file, weights_only=False)
                    print(f"[DiagInfo] Task {i}: Loaded pt_file successfully.", flush=True)
                    
                    labels = inter_item.get("labels", inter_item.get("y")) if isinstance(inter_item, dict) else getattr(inter_item, 'labels', getattr(inter_item, 'y', None))
                    edges = inter_item.get("edge_index") if isinstance(inter_item, dict) else getattr(inter_item, 'edge_index', None)
                    
                    # Replace y when inter_item contains labels/y or the task is NC/GC.
                    if labels is not None or task_type in ["NC", "GC"]:
                        if labels is not None:
                            if pt_file.name in ["hiv.pt", "pcba.pt"] and "graph_idx" in item:
                                graph_idx = int(item["graph_idx"])
                                if isinstance(pt_obj, dict) and "graphs" in pt_obj:
                                    pt_obj["graphs"][graph_idx].y = labels
                                else:
                                    print(f"Warning: pt_obj is not a dict or missing 'graphs' key for {pt_file.name}", flush=True)
                            else:
                                if isinstance(pt_obj, dict):
                                    pt_obj["y"] = labels
                                else:
                                    pt_obj.y = labels
                                
                    # Replace edge_index when present or for LP tasks; ignore it for graph classification.
                    if task_type != "GC":
                        if edges is not None or task_type == "LP":
                            if edges is not None:
                                if isinstance(pt_obj, dict):
                                    pt_obj["edge_index"] = edges
                                else:
                                    pt_obj.edge_index = edges
                            
                    print(f"[DiagInfo] Task {i}: Saving modified pt_obj to workspace...", flush=True)
                    torch.save(pt_obj, task_workspace_exec / pt_file.name)
                    print(f"Info: Injected intermediate data for task {i} (idx={task_idx}, labels: {labels is not None}, edges: {edges is not None})", flush=True)
                except Exception as e:
                    print(f"Warning: Failed to inject intermediate data for task {i} (idx={task_idx if 'task_idx' in locals() else '?'}): {type(e).__name__}: {e}")
                    print(f"--- Full Traceback ---")
                    traceback.print_exc()
                    print(f"--- End Traceback ---")
                    shutil.copy(pt_file, task_workspace_exec / pt_file.name)
            else:
                shutil.copy(pt_file, task_workspace_exec / pt_file.name)
        
        task_output_dir = output_dir / task_id / "results"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Assemble the full prompt and explicitly tell the agent that the dataset file exists.
        full_prompt = f"{question}\n\nNote: There is a dataset file named '{pt_file.name}' in the current working directory. Please use it when processing the task."

        task_gateway_port = args.port + i

        spec = AgentTaskSpec(
            task_id=task_id,
            task={
                "env": "",
                # workspace/exec is mounted at /app as read-only.
                # setup_workspace copies all of /app/. to /root (TMP_WORKSPACE).
                # The PT file therefore appears at /root/{pt_file.name} without an extra warmup symlink.
                "warmup": "",
                "port_bindings": {task_gateway_port: 18789}
            },
            workspace_path=str(task_workspace),
            prompt=full_prompt,
            timeout_seconds=600, # Default per-task timeout: 10 minutes.
            output_dir=task_output_dir,
            model=args.model
        )
        
        execution = None
        try:
            # Mount or inject the environment before starting agent.run_task.
            # run_task calls start_container internally, so injection can happen inside or outside run_task.
            # We need to modify run_task or use its internal mechanism because this minimal script calls agent.run_task(spec) directly.
            # We can place tools in the workspace temporarily or update spec.task so the framework loads skills normally.
            
            # Inject ground truth dynamically into an isolated skills directory for thread safety.
            if args.skills_dir and Path(args.skills_dir).exists():
                original_skills_dir = Path(args.skills_dir)
                task_skills_parent = output_dir / task_id / "skills"
                task_skills_dir = task_skills_parent / original_skills_dir.name
                
                # Create a dedicated skills copy for this task.
                if not task_skills_dir.exists():
                    shutil.copytree(original_skills_dir, task_skills_dir)
                
                # Extract task-specific ground truth.
                nc_label, lp_label, gc_label = None, None, None
                
                # Support pure_ml_bench.json items containing task_type.
                if "task_type" in item and "ground_truth" in item:
                    task_type = item.get("task_type")
                    gt = item.get("ground_truth")
                    if task_type == "NC":
                        nc_label = gt
                    elif task_type == "LP":
                        lp_label = gt
                    elif task_type == "GC":
                        gc_label = gt
                        
                # Support graph_prediction.json items containing a ground_truth object.
                elif "ground_truth" in item and isinstance(item["ground_truth"], dict):
                    gt_dict = item["ground_truth"]
                    if "structural_node_embedding_classification" in gt_dict:
                        nc_label = gt_dict["structural_node_embedding_classification"].get("label")
                    if "structural_node_embedding_link_prediction" in gt_dict:
                        lp_label = gt_dict["structural_node_embedding_link_prediction"].get("label")
                    if "structural_graph_embedding_classification" in gt_dict:
                        gc_label = gt_dict["structural_graph_embedding_classification"].get("label")

                # Apply dynamic injection to graph_ml_tools.py.
                graph_ml_tools_path = task_skills_dir / "graph_ml_tools.py"
                if graph_ml_tools_path.exists():
                    with open(graph_ml_tools_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    if nc_label is not None:
                        val_str = f"'{nc_label}'" if isinstance(nc_label, str) else str(nc_label)
                        code = code.replace("INJECTED_NC_LABEL = None", f"INJECTED_NC_LABEL = {val_str}")
                    if lp_label is not None:
                        val_str = f"'{lp_label}'" if isinstance(lp_label, str) else str(lp_label)
                        code = code.replace("INJECTED_LP_LABEL = None", f"INJECTED_LP_LABEL = {val_str}")
                    if gc_label is not None:
                        val_str = f"'{gc_label}'" if isinstance(gc_label, str) else str(gc_label)
                        code = code.replace("INJECTED_GC_LABEL = None", f"INJECTED_GC_LABEL = {val_str}")
                    with open(graph_ml_tools_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                
                # Tell the evaluation framework to load the isolated, dynamically modified skills copy.
                spec.task["skills_path"] = str(task_skills_parent)
                spec.task["skills"] = task_skills_dir.name
            
            # 3. Send the task to the agent and execute it in a Docker container.
            execution = agent.run_task(spec)
            
            # 4. Extract usage data and trajectory JSONL after the task finishes.
            # (This generates chat.jsonl and usage.json under task_output_dir.)
            usage = agent.collect_usage(task_id, task_output_dir, execution.elapsed_time)
            with open(task_output_dir / "usage.json", "w", encoding='utf-8') as f:
                json.dump(usage, f, indent=2)
                
            # 5. Extract files produced by the agent, including modified code and generated files.
            # collect_output_from_container(task_id, task_output_dir, include_workspace_changes=True)
            
            print(f"Task {i} finished. Elapsed: {execution.elapsed_time:.2f}s")
            print(f"Trajectory & Results saved to: {task_output_dir}")
            
        except Exception as e:
            print(f"Task {i} failed with error: {e}")
        finally:
            # 6. Always clean up temporary resources.
            if execution and execution.gateway_proc:
                execution.gateway_proc.terminate()
            remove_container(task_id)
            
            # Remove workspace and skills copies created for mounting to save space.
            if task_workspace.exists():
                shutil.rmtree(task_workspace, ignore_errors=True)
            if 'task_skills_parent' in locals() and task_skills_parent.exists():
                shutil.rmtree(task_skills_parent, ignore_errors=True)

    if args.parallel <= 1:
        for i, item in enumerate(items):
            process_task(i, item)
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_task, i, item): i for i, item in enumerate(items)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    task_index = futures[future]
                    print(f"Task {task_index} thread failed with exception: {e}")

    print("\nAll custom tasks finished.")

if __name__ == "__main__":
    main()

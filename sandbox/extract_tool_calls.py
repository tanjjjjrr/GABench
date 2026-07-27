import sys
import json
from pathlib import Path

# Add src to sys.path so transcript_loader can be imported.
sys.path.append(str(Path(__file__).parent))
from src.utils.transcript_loader import load_transcript

def refine_tool_name(name, args_str):
    if name in ("shell", "execute_bash_command"):
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            cmd = args.get("command", "")
            if isinstance(cmd, list):
                cmd_str = " ".join(cmd)
            else:
                cmd_str = str(cmd)
            
            if "pt_utils.py" in cmd_str:
                parts = cmd_str.split("pt_utils.py")
                if len(parts) > 1 and parts[1].strip():
                    sub = parts[1].strip().split()[0]
                    return f"pt_utils ({sub})" if sub else "pt_utils"
                return "pt_utils"
            
            if "graph_tools.py" in cmd_str:
                parts = cmd_str.split("graph_tools.py")
                if len(parts) > 1 and parts[1].strip():
                    # extract_ego_boundary, simulate_edge_perturbation, and similar tools
                    return parts[1].strip().split()[0]
                return "graph_tools"
                
            # Extract simple non-Python commands such as ls and cat.
            if "bash" in cmd_str and "-lc" in cmd_str:
                parts = cmd_str.split("-lc")
                if len(parts) > 1:
                    actual_cmd = parts[1].strip().strip('\'"')
                    first_word = actual_cmd.split()[0] if actual_cmd else "bash"
                    if first_word not in ["python", "python3"]:
                        return first_word
        except Exception:
            pass
    return name

def analyze_tool_calls(chat_file):
    messages = load_transcript(chat_file)
    if not messages:
        print(f"No messages found or unable to parse {chat_file}")
        return

    tool_chain = []
    tool_counts = {}
    
    # Track pending tool-call requests (tool_call_id -> request_info).
    pending_calls = {}

    for msg in messages:
        # Support the standard xxx format.
        role = msg.get("role")
        
        # Support the xxx/xxx format.
        msg_type = msg.get("type")
        payload = msg.get("payload", {})
        payload_type = payload.get("type") if isinstance(payload, dict) else None
        
        # 1. Detect when the agent decides to call a tool (xxx format).
        if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id")
                func = tc.get("function", {})
                raw_name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                
                name = refine_tool_name(raw_name, args)
                pending_calls[tc_id] = {
                    "name": name,
                    "arguments": args
                }
                tool_counts[name] = tool_counts.get(name, 0) + 1

        # 1. Detect when the agent decides to call a tool (xxx format).
        elif msg_type == "response_item" and payload_type == "function_call":
            tc_id = payload.get("call_id")
            raw_name = payload.get("name", "unknown")
            args = payload.get("arguments", "{}")
            
            name = refine_tool_name(raw_name, args)
            pending_calls[tc_id] = {
                "name": name,
                "arguments": args
            }
            tool_counts[name] = tool_counts.get(name, 0) + 1

        # 2. Detect tool execution results returned by the environment (xxx format).
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            content = msg.get("content", "")
            
            if tc_id in pending_calls:
                req = pending_calls.pop(tc_id)
                name = req["name"]
                args = req["arguments"]
                
                # Heuristically determine success using common error keywords.
                content_lower = str(content).lower()
                is_error = False
                if any(err_kw in content_lower for err_kw in [
                    "error:", "exception:", "traceback (most recent call last)", 
                    "command not found", "no such file or directory"
                ]):
                    is_error = True
                
                tool_chain.append({
                    "name": name,
                    "arguments": args,
                    "success": not is_error,
                    "result_snippet": str(content)[:200].replace('\n', ' ') + ("..." if len(str(content)) > 200 else "")
                })
                
        # 2. Detect tool execution results returned by the environment (xxx format).
        elif msg_type == "response_item" and payload_type == "function_call_output":
            tc_id = payload.get("call_id")
            content = payload.get("output", "")
            
            if tc_id in pending_calls:
                req = pending_calls.pop(tc_id)
                name = req["name"]
                args = req["arguments"]
                
                is_error = False
                output_text = str(content)
                try:
                    # Try to parse xxx JSON output containing exit_code and output.
                    content_json = json.loads(content)
                    if isinstance(content_json, dict):
                        if content_json.get("metadata", {}).get("exit_code", 0) != 0:
                            is_error = True
                        output_text = content_json.get("output", output_text)
                except Exception:
                    pass
                
                if not is_error:
                    content_lower = output_text.lower()
                    if any(err_kw in content_lower for err_kw in [
                        "error:", "exception:", "traceback (most recent call last)", 
                        "command not found", "no such file or directory"
                    ]):
                        is_error = True
                
                tool_chain.append({
                    "name": name,
                    "arguments": args,
                    "success": not is_error,
                    "result_snippet": output_text[:200].replace('\n', ' ') + ("..." if len(output_text) > 200 else "")
                })

    # Print statistics and the tool-call chain log.
    print("=" * 60)
    print("🛠️  Tool Call Statistics")
    print("=" * 60)
    if not tool_counts:
        print("No tool calls detected.")
    for name, count in tool_counts.items():
        print(f"- {name}: {count} calls")
    
    print("\n" + "=" * 60)
    print("🔗 Tool Call Chain (chronological order)")
    print("=" * 60)
    for i, call in enumerate(tool_chain, 1):
        status = "❌ Possible execution error (ERROR)" if not call["success"] else "✅ Execution succeeded (SUCCESS)"
        print(f"[{i}] Tool name: {call['name']}")
        print(f"    Status: {status}")
        print(f"    Arguments: {call['arguments']}")
        print(f"    Result (truncated): {call['result_snippet'].strip()}")
        print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_tool_calls.py <path-to-chat.jsonl>")
    else:
        analyze_tool_calls(sys.argv[1])

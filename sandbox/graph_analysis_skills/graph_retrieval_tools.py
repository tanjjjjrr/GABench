import os
import json
import torch

def _format_tensor_info(obj):
    """
    Helper function: format output object info (shape, type, sample data).
    """
    if isinstance(obj, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "sample": obj.flatten()[:5].tolist() if obj.numel() > 0 else []
        }
    elif isinstance(obj, (int, float, str, bool)):
        return {
            "type": type(obj).__name__,
            "value": obj
        }
    elif isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "length": len(obj),
            "sample": obj[:5]
        }
    elif hasattr(obj, 'shape'):
        return {
            "type": type(obj).__name__,
            "shape": list(obj.shape) if hasattr(obj.shape, '__iter__') else obj.shape,
            "sample": "..."
        }
    else:
        return {
            "type": type(obj).__name__,
            "sample": str(obj)[:50]
        }

def _to_json_serializable(obj):
    """
    Helper function: convert Tensor and other types to JSON serializable format.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    elif hasattr(obj, 'numpy'):
        return obj.numpy().tolist()
    else:
        return str(obj)

# -------------------------------------------------------------------------
# Tool 1: Function to read pt file with PyG library
# -------------------------------------------------------------------------
def inspect_pt_with_pyg(filepath):
    """
    1: Function to read pt file with PyG, input is pt file path or name,
    output is the keys in pt file and corresponding shape/sample/format.
    """
    print("[Output Description]: JSON containing the structural contents, keys, and feature dimensions of the dataset.")
    try:
        import torch_geometric
    except ImportError:
        return {"error": "torch_geometric (PyG) is not installed."}
    
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        info = {}
        
        if hasattr(data, 'keys') and callable(data.keys): # PyG Data object
            for key in data.keys():
                info[key] = _format_tensor_info(data[key])
        elif isinstance(data, dict):
            for key, val in data.items():
                info[key] = _format_tensor_info(val)
        else:
            info["data"] = _format_tensor_info(data)
            
        return {"filepath": filepath, "method": "PyG", "info": info}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------------
# Tool 2: Function to read pt file without PyG library
# -------------------------------------------------------------------------
def inspect_pt_without_pyg(filepath):
    """
    2: Function to read pt file without importing PyG.
    output is the keys in pt file and corresponding shape/sample/format.
    """
    print("[Output Description]: JSON containing the structural contents, keys, and feature dimensions of the dataset.")
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        info = {}
        
        if isinstance(data, dict):
            for key, val in data.items():
                info[key] = _format_tensor_info(val)
        elif hasattr(data, '__dict__'):
            for key, val in data.__dict__.items():
                if not key.startswith('_'):
                    info[key] = _format_tensor_info(val)
        else:
            info["data"] = _format_tensor_info(data)
            
        return {"filepath": filepath, "method": "Native Torch", "info": info}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------------
# Tool 3: Extract specific key and save as JSON with PyG
# -------------------------------------------------------------------------
def extract_key_with_pyg(filepath, key_name):
    """
    3: PyG version, input is pt file path and key name,
    output is shape/sample of the key, and save the data to a json file,
    json file name is the key name, and print output to xxx.json in terminal.
    """
    print("[Output Description]: A `.json` file containing the serialized graph structures or requested features.")
    try:
        import torch_geometric
    except ImportError:
        print("Error: torch_geometric (PyG) is not installed.")
        return
        
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        val = None
        if hasattr(data, 'keys') and callable(data.keys) and key_name in data.keys():
            val = data[key_name]
        elif isinstance(data, dict) and key_name in data:
            val = data[key_name]
        elif hasattr(data, key_name):
            val = getattr(data, key_name)
        else:
            print(f"Error: Key '{key_name}' not found in the pt file (PyG Mode).")
            return
            
        info = _format_tensor_info(val)
        print("[Extracted Data Info]:", json.dumps(info, indent=2, ensure_ascii=False))
        
        out_filename = f"{key_name}.json"
        serializable_data = _to_json_serializable(val)
        
        with open(out_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False)
            
        print(f"Output to {out_filename}")
        
        return info
    except Exception as e:
        print(f"Error: {str(e)}")

# -------------------------------------------------------------------------
# Tool 4: Extract specific key and save as JSON without PyG
# -------------------------------------------------------------------------
def extract_key_without_pyg(filepath, key_name):
    """
    4: No-PyG version, input is pt file path and key name,
    output is shape/sample of the key, and save the data to a json file,
    json file name is the key name, and print output to xxx.json in terminal.
    """
    print("[Output Description]: A `.json` file containing the serialized graph structures or requested features.")
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        val = None
        if isinstance(data, dict) and key_name in data:
            val = data[key_name]
        elif hasattr(data, '__dict__') and key_name in data.__dict__:
            val = data.__dict__[key_name]
        elif hasattr(data, key_name):
            val = getattr(data, key_name)
        else:
            print(f"Error: Key '{key_name}' not found in the pt file (Native Torch Mode).")
            return
            
        info = _format_tensor_info(val)
        print("[Extracted Data Info]:", json.dumps(info, indent=2, ensure_ascii=False))
        
        out_filename = f"{key_name}.json"
        serializable_data = _to_json_serializable(val)
        
        with open(out_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False)
            
        print(f"Output to {out_filename}")
        
        return info
    except Exception as e:
        print(f"Error: {str(e)}")

# ==============================================================================
# CLI (Command Line Interface) wrapper
# ==============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python graph_retrieval_tools.py <action> <filepath> [key_name]")
        print("Example: python graph_retrieval_tools.py inspect_pt_without_pyg my_graph.pt")
        print("Example: python graph_retrieval_tools.py extract_key_without_pyg my_graph.pt x")
        sys.exit(1)
        
    action = sys.argv[1]
    filepath = sys.argv[2]
    
    if action == "inspect_pt_with_pyg":
        result = inspect_pt_with_pyg(filepath)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif action == "inspect_pt_without_pyg":
        result = inspect_pt_without_pyg(filepath)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif action == "extract_key_with_pyg":
        if len(sys.argv) < 4:
            print("Error: extract_key_with_pyg requires key_name")
            sys.exit(1)
        extract_key_with_pyg(filepath, sys.argv[3])
    elif action == "extract_key_without_pyg":
        if len(sys.argv) < 4:
            print("Error: extract_key_without_pyg requires key_name")
            sys.exit(1)
        extract_key_without_pyg(filepath, sys.argv[3])
    else:
        print(f"Unknown action: {action}")

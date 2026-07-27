import os
import sys
import json
import torch
import numpy as np

# Try to reuse formatting helper function from graph_retrieval_tools
try:
    from graph_retrieval_tools import _format_tensor_info
except ImportError:
    def _format_tensor_info(obj):
        if isinstance(obj, torch.Tensor):
            return {
                "type": "torch.Tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "sample": obj.flatten()[:5].tolist() if obj.numel() > 0 else []
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

def _to_numpy(obj):
    """
    Helper function: safely convert extracted data to numpy array
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    else:
        # Try forced conversion with np.array (for list, tuple, etc.)
        return np.array(obj)

# -------------------------------------------------------------------------
# Tool 1: Extract specific key and save as numpy array (.npy) with PyG
# -------------------------------------------------------------------------
def extract_key_to_numpy_with_pyg(filepath, key_name):
    """
    Extract specific key data with PyG, output shape/sample, and save as {key_name}.npy.
    """
    print("[Output Description]: A `.npy` file containing the extracted tensor data as a Numpy array.")
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
        
        out_filename = f"{key_name}.npy"
        np_data = _to_numpy(val)
        
        np.save(out_filename, np_data)
            
        print(f"Output to {out_filename}")
        
        return info
    except Exception as e:
        print(f"Error: {str(e)}")

# -------------------------------------------------------------------------
# Tool 2: Extract specific key and save as numpy array (.npy) without PyG
# -------------------------------------------------------------------------
def extract_key_to_numpy_without_pyg(filepath, key_name):
    """
    Extract specific key data without PyG, output shape/sample, and save as {key_name}.npy.
    """
    print("[Output Description]: A `.npy` file containing the extracted tensor data as a Numpy array.")
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
        
        out_filename = f"{key_name}.npy"
        np_data = _to_numpy(val)
        
        np.save(out_filename, np_data)
            
        print(f"Output to {out_filename}")
        
        return info
    except Exception as e:
        print(f"Error: {str(e)}")

# -------------------------------------------------------------------------
# Tool 3: Extract graph with specific id from graph list and save as pt with PyG
# -------------------------------------------------------------------------
def extract_graph_by_id_with_pyg(filepath, key_name, graph_id):
    """
    Extract graph with specific id from pt file's key (which stores a list of graphs) with PyG,
    output sample of the graph, and save it as an independent .pt file.
    """
    print("[Output Description]: A `.pt` file containing the isolated single graph instance.")
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
            
        if not isinstance(val, (list, tuple)):
            print(f"Error: The data under key '{key_name}' is not a list. Found: {type(val)}")
            return
            
        graph_id = int(graph_id)
        if graph_id < 0 or graph_id >= len(val):
            print(f"Error: graph_id {graph_id} is out of bounds for list of size {len(val)}")
            return
            
        graph_obj = val[graph_id]
        
        print("[Extracted Graph Sample]:")
        print(graph_obj)
        
        if hasattr(graph_obj, 'x') and hasattr(graph_obj.x, 'shape'):
            print(f" - Nodes: {graph_obj.x.shape[0]}")
        if hasattr(graph_obj, 'edge_index') and hasattr(graph_obj.edge_index, 'shape'):
            print(f" - Edges: {graph_obj.edge_index.shape[1]}")
            
        out_filename = f"{key_name}_{graph_id}.pt"
        torch.save(graph_obj, out_filename)
        
        print(f"Output pt file storing the graph to {out_filename}")
        
        return graph_obj
    except Exception as e:
        print(f"Error: {str(e)}")

# ==============================================================================
# CLI (Command Line Interface) wrapper
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python graph_io_tools.py <action> <filepath> [key_name]")
        print("Example: python graph_io_tools.py extract_key_to_numpy_without_pyg my_graph.pt x")
        sys.exit(1)
        
    action = sys.argv[1]
    filepath = sys.argv[2]
    
    if action == "extract_key_to_numpy_with_pyg":
        if len(sys.argv) < 4:
            print("Error: extract_key_to_numpy_with_pyg requires key_name")
            sys.exit(1)
        extract_key_to_numpy_with_pyg(filepath, sys.argv[3])
    elif action == "extract_key_to_numpy_without_pyg":
        if len(sys.argv) < 4:
            print("Error: extract_key_to_numpy_without_pyg requires key_name")
            sys.exit(1)
        extract_key_to_numpy_without_pyg(filepath, sys.argv[3])
    elif action == "extract_graph_by_id_with_pyg":
        if len(sys.argv) < 5:
            print("Error: extract_graph_by_id_with_pyg requires key_name and graph_id")
            sys.exit(1)
        extract_graph_by_id_with_pyg(filepath, sys.argv[3], sys.argv[4])
    else:
        print(f"Unknown action: {action}")

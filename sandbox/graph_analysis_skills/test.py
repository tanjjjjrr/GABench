import argparse
import torch

def get_attr(obj, attr_names):
    """Attempt to get an attribute from a dict or object using a list of possible names."""
    if isinstance(obj, dict):
        for name in attr_names:
            if name in obj:
                return obj[name]
    else:
        for name in attr_names:
            if hasattr(obj, name):
                return getattr(obj, name)
    return None

def compare_tensors_detailed(t1, t2, name):
    if t1 is None and t2 is None:
        print(f"[{name}] Both are None.")
        return
    elif t1 is None or t2 is None:
        print(f"[{name}] One is None and the other is not. t1={type(t1)}, t2={type(t2)}")
        return
        
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        print(f"[{name}] Types: t1={type(t1)}, t2={type(t2)}")
        if t1 == t2:
            print(f"[{name}] Values are identical (non-tensor comparison).")
        else:
            print(f"[{name}] Values are DIFFERENT (non-tensor comparison).")
        return

    if t1.shape != t2.shape:
        print(f"[{name}] Shape changed: {t1.shape} -> {t2.shape}")
    elif torch.equal(t1, t2):
        print(f"[{name}] Values are EXACTLY THE SAME! Shape: {t1.shape}")
        return
    else:
        print(f"[{name}] Values changed! Shape remains {t1.shape}.")

    print(f"--- {name} Detailed Differences ---")
    if name == "edge_index" and t1.dim() == 2 and t2.dim() == 2:
        # Assuming shape [2, num_edges]
        if t1.shape[0] == 2:
            set1 = set(map(tuple, t1.t().tolist()))
            set2 = set(map(tuple, t2.t().tolist()))
        else:
            set1 = set(map(tuple, t1.tolist()))
            set2 = set(map(tuple, t2.tolist()))
            
        added = set2 - set1
        removed = set1 - set2
        print(f"  -> Edges removed ({len(removed)}): {list(removed)[:10]}" + ("..." if len(removed)>10 else ""))
        print(f"  -> Edges added ({len(added)}): {list(added)[:10]}" + ("..." if len(added)>10 else ""))
        
    elif name in ["y", "labels", "label"] and t1.dim() == 1 and t2.dim() == 1:
        if t1.shape == t2.shape:
            diff_indices = (t1 != t2).nonzero(as_tuple=True)[0]
            print(f"  -> Labels changed at {len(diff_indices)} indices.")
            print(f"  -> Changed indices: {diff_indices.tolist()[:10]}" + ("..." if len(diff_indices)>10 else ""))
            print(f"  -> File 1 values: {t1[diff_indices].tolist()[:10]}" + ("..." if len(diff_indices)>10 else ""))
            print(f"  -> File 2 values: {t2[diff_indices].tolist()[:10]}" + ("..." if len(diff_indices)>10 else ""))
        else:
            print("  -> Cannot do element-wise comparison because shapes differ.")
    else:
        print("  -> Detailed comparison not implemented for this shape/dimension.")

def main():
    parser = argparse.ArgumentParser(description="Compare two .pt files on label (y) and edge_index dimensions")
    parser.add_argument("--pt_file1", type=str, required=True, help="Path to the first .pt file")
    parser.add_argument("--pt_file2", type=str, required=True, help="Path to the second .pt file")
    
    args = parser.parse_args()
    
    print(f"Loading file 1: {args.pt_file1}")
    try:
        pt1 = torch.load(args.pt_file1, weights_only=False)
    except Exception as e:
        print(f"Failed to load {args.pt_file1}: {e}")
        return
        
    print(f"Loading file 2: {args.pt_file2}")
    try:
        pt2 = torch.load(args.pt_file2, weights_only=False)
    except Exception as e:
        print(f"Failed to load {args.pt_file2}: {e}")
        return
        
    print("\n" + "="*50)
    print("Comparing 'label / y' dimension...")
    print("="*50)
    y1 = get_attr(pt1, ["y", "labels", "label"])
    y2 = get_attr(pt2, ["y", "labels", "label"])
    compare_tensors_detailed(y1, y2, "y")
    
    print("\n" + "="*50)
    print("Comparing 'edge_index' dimension...")
    print("="*50)
    edge_index1 = get_attr(pt1, ["edge_index"])
    edge_index2 = get_attr(pt2, ["edge_index"])
    compare_tensors_detailed(edge_index1, edge_index2, "edge_index")

if __name__ == "__main__":
    main()

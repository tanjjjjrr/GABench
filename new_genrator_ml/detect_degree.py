import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="View top 100 indegree nodes and average degree of a .pt graph file.")
    parser.add_argument("pt_file", type=str, help="Path to the .pt file")
    args = parser.parse_args()

    print(f"Loading {args.pt_file}...")
    try:
        data = torch.load(args.pt_file, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading {args.pt_file}: {e}")
        return

    if not hasattr(data, "edge_index") or data.edge_index is None:
        print("Error: The loaded data does not have an 'edge_index' attribute.")
        return

    edge_index = data.edge_index
    num_edges = edge_index.shape[1]
    
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    else:
        num_nodes = int(edge_index.max().item() + 1)
        print(f"Warning: 'num_nodes' not found. Inferred from edge_index as {num_nodes}.")

    print(f"Graph loaded. Nodes: {num_nodes}, Edges: {num_edges}")

    # Calculate indegree (occurrences in edge_index[1])
    dst_indices = edge_index[1]
    
    # Compute degree distribution using torch.bincount
    indegrees = torch.bincount(dst_indices, minlength=num_nodes)
    
    # Average degree
    avg_indegree = num_edges / num_nodes
    avg_total_degree = (num_edges * 2) / num_nodes

    print(f"\nAverage In-Degree (or Out-Degree): {avg_indegree:.4f}")
    print(f"Average Total-Degree: {avg_total_degree:.4f}")

    # Top 100 nodes with highest indegree
    topk = min(100, num_nodes)
    top_indegrees, top_indices = torch.topk(indegrees, k=topk)

    print(f"\nTop {topk} nodes with highest indegree:")
    print(f"{'Rank':<6} {'Node ID':<10} {'In-Degree':<12}")
    print("-" * 30)
    for i in range(topk):
        node_id = top_indices[i].item()
        degree = top_indegrees[i].item()
        print(f"{i+1:<6} {node_id:<10} {degree:<12}")

if __name__ == "__main__":
    main()

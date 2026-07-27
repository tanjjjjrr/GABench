import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Count nodes and edges of a .pt graph file.")
    parser.add_argument("pt_file", type=str, help="Path to the .pt file")
    args = parser.parse_args()

    print(f"Loading {args.pt_file}...")
    try:
        data = torch.load(args.pt_file, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading {args.pt_file}: {e}")
        return

    # Calculate number of edges
    if hasattr(data, "edge_index") and data.edge_index is not None:
        num_edges = data.edge_index.shape[1]
    else:
        print("Warning: The loaded data does not have an 'edge_index' attribute.")
        num_edges = 0

    # Calculate number of nodes
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    elif hasattr(data, "edge_index") and data.edge_index is not None:
        num_nodes = int(data.edge_index.max().item() + 1)
        print(f"Warning: 'num_nodes' not found. Inferred from edge_index max value as {num_nodes}.")
    else:
        num_nodes = 0
        print("Warning: Cannot determine number of nodes.")

    print("=" * 30)
    print(f"Number of Nodes: {num_nodes:,}")
    print(f"Number of Edges: {num_edges:,}")
    print("=" * 30)

if __name__ == "__main__":
    main()

import argparse
import sys
import torch
import networkx as nx

# Import reusable graph helpers
from graph_theory_tools import extract_n_hop_ego_subgraph, get_node_degree

def load_graph(filepath):
    if filepath.endswith('.pt'):
        obj = torch.load(filepath, weights_only=False)
        if isinstance(obj, nx.Graph):
            return obj
        try:
            from torch_geometric.utils import to_networkx
            # Convert common graph data objects to a NetworkX graph.
            # Directed inputs stay directed; undirected inputs can set to_undirected=True when needed.
            # Adjust this conversion if the graph source has different direction semantics.
            return to_networkx(obj)
        except Exception as e:
            raise ValueError(f"Failed to convert PyTorch geometric object to NetworkX: {e}")
    else:
        raise ValueError("Only .pt files are supported in this test script.")

def main():
    parser = argparse.ArgumentParser(description="Extract n-hop subgraph and calculate node degree.")
    parser.add_argument("--pt_file", type=str, required=True, help="Path to the .pt dataset file")
    parser.add_argument("--center_node", type=int, required=True, help="Node ID of the center of the completely induced subgraph")
    parser.add_argument("--hops", type=int, required=True, help="Number of hops for the ego subgraph")
    parser.add_argument("--target_node", type=int, required=True, help="Target node to calculate degree for in the subgraph")
    
    args = parser.parse_args()
    
    print(f"Loading graph from {args.pt_file} ...")
    try:
        G = load_graph(args.pt_file)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)
        
    print(f"Extracting {args.hops}-hop ego subgraph centered at node {args.center_node} ...")
    # Relax min_nodes and max_nodes so the requested center is less likely to fall back.
    H, actual_center, actual_hops = extract_n_hop_ego_subgraph(
        G, 
        center_node=args.center_node, 
        hops=args.hops,
        min_nodes=0,
        max_nodes=float('inf')
    )
    
    print(f"Successfully extracted subgraph. Nodes: {H.number_of_nodes()}, Edges: {H.number_of_edges()}")
    
    if args.target_node not in H:
        print(f"Error: Target node {args.target_node} is NOT in the extracted subgraph.")
        sys.exit(1)
        
    degree = get_node_degree(H, args.target_node)
    
    print("-" * 50)
    print(f"Result:")
    print(f"Degree of node {args.target_node} in the {args.hops}-hop subgraph centered at node {args.center_node} is: {degree}")
    print("-" * 50)

if __name__ == "__main__":
    main()

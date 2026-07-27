import torch
import networkx as nx
import sys

def detect_cycles(file_path):
    try:
        data = torch.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False

    # Try to extract edge_index
    edges = []
    
    # Check if it's a tuple (data, slices) from PyG in-memory dataset
    if isinstance(data, tuple) and len(data) == 2 and hasattr(data[0], 'edge_index'):
        edge_index = data[0].edge_index
    elif hasattr(data, 'edge_index'):
        edge_index = data.edge_index
    elif isinstance(data, dict) and 'edge_index' in data:
        edge_index = data['edge_index']
    elif isinstance(data, list):
        # Process the first item if it's a list
        if len(data) > 0 and hasattr(data[0], 'edge_index'):
            edge_index = data[0].edge_index
        elif len(data) > 0 and isinstance(data[0], dict) and 'edge_index' in data[0]:
            edge_index = data[0]['edge_index']
        else:
            edge_index = None
    else:
        edge_index = None

    if edge_index is None:
        print("Could not find edge_index in the data.")
        return False

    # Format edge_index into a list of tuples
    if isinstance(edge_index, torch.Tensor):
        if edge_index.size(0) == 2:
            edges = edge_index.t().tolist()
        else:
            edges = edge_index.tolist()
    else:
        edges = list(edge_index)

    # Build Directed Graph
    G = nx.DiGraph()
    G.add_edges_from(edges)

    cycle_count = 0
    try:
        # nx.simple_cycles returns a generator of cycles
        for cycle in nx.simple_cycles(G):
            # Skip cycles of length 2 or less (which often correspond to bidirectional edges in undirected graphs)
            if len(cycle) <= 2:
                continue
                
            cycle_count += 1
            print(f"Found cycle {cycle_count}: {cycle}")
            if cycle_count >= 10:
                print("Count reached 10, stopping and exiting.")
                sys.exit(0)
    except Exception as e:
        print(f"Error during cycle detection: {e}")
        return False

    if cycle_count > 0:
        return True
    else:
        return False

if __name__ == "__main__":
    file_path = "processed_data.pt"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    has_cycle = detect_cycles(file_path)
    if not has_cycle:
        print("False")
        sys.exit(0)

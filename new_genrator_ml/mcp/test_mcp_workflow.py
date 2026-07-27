import networkx as nx
from tools_subgraph import extract_ego_graph
from tools_metrics import compute_density, count_connected_components, check_reachability
from tools_modification import remove_edges
from graph_state import GraphSessionManager

def test_perturb_workflow():
    # 1. Initialize a global graph for testing
    print("=== Step 1: Initialize Global Graph ===")
    G_global = nx.fast_gnp_random_graph(20, 0.2, seed=42) # Generate a random graph
    global_id = GraphSessionManager.register_graph(G_global, "global_graph")
    print(f"Global graph registered with ID: {global_id}")
    print(f"Global nodes: {G_global.number_of_nodes()}, edges: {G_global.number_of_edges()}")
    
    # 2. Simulate the xxx extracting a local subgraph (3-hop ego graph)
    print("\n=== Step 2: Extract Subgraph ===")
    target_node = 0
    subgraph_id = extract_ego_graph(global_id, center_node=target_node, radius=3)
    subgraph = GraphSessionManager.get_graph(subgraph_id)
    print(f"Extracted ego graph ID: {subgraph_id}")
    print(f"Subgraph nodes: {subgraph.number_of_nodes()}, edges: {subgraph.number_of_edges()}")
    
    # 3. Compute baseline metrics before perturbation
    print("\n=== Step 3: Compute Baseline Metrics ===")
    orig_density = compute_density(subgraph_id)
    orig_cc = count_connected_components(subgraph_id)
    reachable_10 = check_reachability(subgraph_id, source=0, target=10, max_hops=3)
    print(f"Original Density: {orig_density:.4f}")
    print(f"Original Connected Components: {orig_cc}")
    print(f"Is Node 10 reachable from 0 in 3 hops? {reachable_10}")
    
    # 4. Simulate the xxx deciding which edges to remove
    print("\n=== Step 4: Perturb Subgraph (Remove Edges) ===")
    edges_to_remove = list(subgraph.edges())[:3] # Remove the first three edges
    print(f"Removing edges: {edges_to_remove}")
    success = remove_edges(subgraph_id, edges_to_remove)
    print(f"Edge removal success: {success}")
    print(f"Subgraph edges after removal: {subgraph.number_of_edges()}")
    
    # 5. Recompute metrics after perturbation
    print("\n=== Step 5: Compute New Metrics & Compare ===")
    new_density = compute_density(subgraph_id)
    new_cc = count_connected_components(subgraph_id)
    new_reachable_10 = check_reachability(subgraph_id, source=0, target=10, max_hops=3)
    print(f"New Density: {new_density:.4f} (Change: {new_density - orig_density:.4f})")
    print(f"New Connected Components: {new_cc} (Change: {new_cc - orig_cc})")
    print(f"Is Node 10 reachable from 0 in 3 hops now? {new_reachable_10}")
    
    # 6. Clean up memory
    GraphSessionManager.clear_all()
    print("\n=== Test completed and memory cleared ===")

if __name__ == "__main__":
    test_perturb_workflow()

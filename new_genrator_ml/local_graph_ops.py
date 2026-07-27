import networkx as nx
from typing import List, Dict, Any, Optional, Set

# ==============================================================================
# 1. Local-neighborhood evolution and pruning operators
# ==============================================================================

def get_local_neighbors_by_hop(G: nx.Graph, node_id: Any, hop_num: int = 1, 
                               node_filters: Optional[Dict[str, Any]] = None, 
                               edge_filters: Optional[Dict[str, Any]] = None) -> List[Any]:
    """Explore outward from node_id and return neighbors reached within hop_num hops, typically one hop."""
    if node_id not in G:
        return []
        
    current_nodes = {node_id}
    visited = {node_id}
    
    for _ in range(hop_num):
        next_nodes = set()
        for u in current_nodes:
            # Support neighbor traversal for directed and undirected graphs
            neighbors = G.successors(u) if G.is_directed() else G.neighbors(u)
            for v in neighbors:
                if v not in visited:
                    # Filter edge attributes
                    if edge_filters:
                        edge_data = G.get_edge_data(u, v) or {}
                        # Handle multigraphs, where G.get_edge_data may return data for multiple edges
                        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                            # It is sufficient for one edge to satisfy the condition
                            edge_match = any(
                                all(e_data.get(k) == v_val for k, v_val in edge_filters.items())
                                for e_data in edge_data.values()
                            )
                        else:
                            edge_match = all(edge_data.get(k) == v_val for k, v_val in edge_filters.items())
                            
                        if not edge_match:
                            continue
                            
                    # Filter node attributes
                    if node_filters:
                        node_data = G.nodes[v]
                        if not all(node_data.get(k) == v_val for k, v_val in node_filters.items()):
                            continue
                            
                    next_nodes.add(v)
                    visited.add(v)
        current_nodes = next_nodes
        
    return list(current_nodes)

def extract_ego_boundary(G: nx.Graph, node_id: Any, radius: int) -> List[Any]:
    """Extract the induced ego subgraph centered on node_id and return only nodes on its outer boundary."""
    if node_id not in G:
        return []
    
    # Extract the ego_graph with radius radius
    ego_full = nx.ego_graph(G, node_id, radius=radius)
    # Extract the radius-1 ego_graph as the inner nodes
    if radius > 0:
        ego_inner = nx.ego_graph(G, node_id, radius=radius - 1)
        # Boundary nodes = full - inner
        boundary = set(ego_full.nodes()) - set(ego_inner.nodes())
        return list(boundary)
    else:
        return [node_id]

def filter_nodes_by_all_degree(G: nx.Graph, node_list: List[Any], 
                               min_degree: Optional[int] = None, 
                               max_degree: Optional[int] = None) -> List[Any]:
    """Filter node_list by total degree in graph G."""
    filtered_nodes = []
    for node in node_list:
        if node not in G:
            continue
        degree = G.degree(node)
        if min_degree is not None and degree < min_degree:
            continue
        if max_degree is not None and degree > max_degree:
            continue
        filtered_nodes.append(node)
    return filtered_nodes

def filter_nodes_by_in_degree(G: nx.Graph, node_list: List[Any], 
                              min_degree: Optional[int] = None, 
                              max_degree: Optional[int] = None) -> List[Any]:
    """Filter node_list by in-degree in graph G. For undirected graphs, use ordinary degree."""
    filtered_nodes = []
    for node in node_list:
        if node not in G:
            continue
        if G.is_directed():
            degree = G.in_degree(node)
        else:
            degree = G.degree(node)
            
        if min_degree is not None and degree < min_degree:
            continue
        if max_degree is not None and degree > max_degree:
            continue
        filtered_nodes.append(node)
    return filtered_nodes

def filter_nodes_by_out_degree(G: nx.Graph, node_list: List[Any], 
                               min_degree: Optional[int] = None, 
                               max_degree: Optional[int] = None) -> List[Any]:
    """Filter node_list by out-degree in graph G. For undirected graphs, use ordinary degree."""
    filtered_nodes = []
    for node in node_list:
        if node not in G:
            continue
        if G.is_directed():
            degree = G.out_degree(node)
        else:
            degree = G.degree(node)
            
        if min_degree is not None and degree < min_degree:
            continue
        if max_degree is not None and degree > max_degree:
            continue
        filtered_nodes.append(node)
    return filtered_nodes


# ==============================================================================
# 2. Local-cycle and topological-loop operators
# ==============================================================================

def detect_local_cycles(G: nx.Graph, node_id: Any, max_depth: int = 4, exclude_nodes: set = None) -> List[List[Any]]:
    """
    Detect local cycles with a bounded search depth.
    
    The cutoff constrains exploration to the neighborhood of node_id and prevents full-graph expansion. Worst-case work grows approximately with the maximum degree raised to max_depth.
    """
    if node_id not in G:
        return []
        
    if exclude_nodes:
        # Use subgraph_view to create a fast graph view without modifying or copying the original graph
        G = nx.subgraph_view(G, filter_node=lambda n: n not in exclude_nodes)
        
    if node_id not in G:
        return []
        
    cycles = []
    
    if G.is_directed():
        successors = list(G.successors(node_id))
        for succ in successors:
            if succ == node_id:  # Self-loop
                cycles.append([node_id, node_id])
                continue
            # Find all simple paths that return from a neighbor to the start within the length limit
            paths = nx.all_simple_paths(G, source=succ, target=node_id, cutoff=max_depth-1)
            for path in paths:
                cycles.append([node_id] + path)
    else:
        neighbors = list(G.neighbors(node_id))
        for neighbor in neighbors:
            if neighbor == node_id:  # Self-loop
                cycles.append([node_id, node_id])
                continue
            paths = nx.all_simple_paths(G, source=neighbor, target=node_id, cutoff=max_depth-1)
            for path in paths:
                # For undirected graphs, exclude the trivial node_id -> neighbor -> node_id backtrack
                if len(path) > 2:
                    cycles.append([node_id] + path)
                    
    return cycles

def compute_local_clustering(G: nx.Graph, node_id: Any) -> float:
    """Compute the local clustering coefficient of the specified node."""
    if node_id not in G:
        return 0.0
    return float(nx.clustering(G, node_id))


# ==============================================================================
# 3. Local path-guidance and multi-source convergence operators
# ==============================================================================

def find_stepped_shortest_path(G: nx.Graph, source_node: Any, target_node: Any, max_depth: int = 5) -> Optional[List[Any]]:
    """Find a shortest path subject to max_depth, returning None when the limit is exceeded."""
    if source_node not in G or target_node not in G:
        return None
        
    try:
        # Use cutoff to limit BFS search depth
        paths = nx.single_source_shortest_path(G, source_node, cutoff=max_depth)
        if target_node in paths:
            return paths[target_node]
        return None
    except nx.NetworkXNoPath:
        return None

def detect_path_convergence(G: nx.Graph, source_node: Any, target_node: Any, mid_max_hop: int = 2) -> Dict[str, Any]:
    """Detect local bidirectional path convergence by expanding forward from source_node and backward from target_node for mid_max_hop steps."""
    result = {"convergence_nodes": [], "source_paths": {}, "target_paths": {}}
    if source_node not in G or target_node not in G:
        return result
        
    if G.is_directed():
        # Search forward from source_node
        source_paths = nx.single_source_shortest_path(G, source_node, cutoff=mid_max_hop)
        # Search backward from target_node; reverse directed graphs when needed
        G_rev = G.reverse(copy=False)
        target_paths_rev = nx.single_source_shortest_path(G_rev, target_node, cutoff=mid_max_hop)
        # Reverse the path to match the direction
        target_paths = {node: path[::-1] for node, path in target_paths_rev.items()}
    else:
        source_paths = nx.single_source_shortest_path(G, source_node, cutoff=mid_max_hop)
        target_paths_rev = nx.single_source_shortest_path(G, target_node, cutoff=mid_max_hop)
        target_paths = {node: path[::-1] for node, path in target_paths_rev.items()}
        
    source_reach = set(source_paths.keys())
    target_reach = set(target_paths.keys())
    
    intersection = list(source_reach.intersection(target_reach))
    
    result["convergence_nodes"] = intersection
    result["source_paths"] = {n: source_paths[n] for n in intersection}
    result["target_paths"] = {n: target_paths[n] for n in intersection}
    
    return result

def find_local_bridge_nodes(G: nx.Graph, node_id: Any, radius: int = 3) -> List[Any]:
    """Identify local articulation points by extracting an ego graph and computing articulation points within that induced subgraph."""
    if node_id not in G:
        return []
        
    ego_graph = nx.ego_graph(G, node_id, radius=radius)
    
    if ego_graph.is_directed():
        ego_graph = ego_graph.to_undirected()
        
    # Find articulation points in the undirected local subgraph
    cut_vertices = list(nx.articulation_points(ego_graph))
    return cut_vertices


# ==============================================================================
# 4. Local-community and iterative core-peeling operators
# ==============================================================================

def extract_local_k_core(G: nx.Graph, node_id: Any, target_k: int = 2, radius: int = 4) -> List[Any]:
    """Extract a local k-core structure from the network centered on node_id."""
    if node_id not in G:
        return []
        
    ego_graph = nx.ego_graph(G, node_id, radius=radius)
    
    # Remove self-loop interference when needed and extract the target_k k-core directly
    try:
        ego_graph.remove_edges_from(nx.selfloop_edges(ego_graph))
        k_core_subgraph = nx.k_core(ego_graph, k=target_k)
        return list(k_core_subgraph.nodes())
    except nx.NetworkXError:
        return []

def enumerate_local_cliques(G: nx.Graph, node_id: Any, exclude_nodes: set = None) -> List[List[Any]]:
    """Enumerate maximal cliques containing node_id within its local neighborhood."""
    if node_id not in G:
        return []
        
    # Convert the graph to undirected form to find cliques
    calc_G = G.to_undirected() if G.is_directed() else G
    
    # Extract an ego network to bound local computation; global cliques_containing_node can explode on huge graphs
    # Radius 1 is sufficient because every member of a maximum clique containing node_id is within one hop
    ego_graph = nx.ego_graph(calc_G, node_id, radius=1)
    
    if exclude_nodes:
        nodes_to_remove = exclude_nodes.intersection(set(ego_graph.nodes()))
        ego_graph.remove_nodes_from(nodes_to_remove)
        
    if node_id not in ego_graph:
        return []
        
    # Replace cliques_containing_node for compatibility across NetworkX versions
    cliques = [c for c in nx.find_cliques(ego_graph) if node_id in c]
    return cliques

def compute_ego_density(G: nx.Graph, node_id: Any, radius: int = 2) -> float:
    """Compute graph density for the ego subgraph centered on node_id with the specified radius."""
    if node_id not in G:
        return 0.0
        
    ego_graph = nx.ego_graph(G, node_id, radius=radius)
    return nx.density(ego_graph)


# ==============================================================================
# 5. Local graph queries and multimodal/text-similarity helpers
# ==============================================================================

def count_common_neighbors(G: nx.Graph, node_id_1: Any, node_id_2: Any) -> Dict[str, Any]:
    """Count one-hop common neighbors for two nodes and return their intersection and Jaccard similarity."""
    if node_id_1 not in G or node_id_2 not in G:
        return {"common_neighbors": [], "count": 0, "jaccard_similarity": 0.0}
        
    if G.is_directed():
        # For directed graphs, common neighbors may be nodes they both point to or nodes that point to both,
        # For broader coverage, take the union of successors and predecessors
        neighbors_1 = set(G.successors(node_id_1)) | set(G.predecessors(node_id_1))
        neighbors_2 = set(G.successors(node_id_2)) | set(G.predecessors(node_id_2))
    else:
        neighbors_1 = set(G.neighbors(node_id_1))
        neighbors_2 = set(G.neighbors(node_id_2))
        
    common = list(neighbors_1.intersection(neighbors_2))
    union = neighbors_1.union(neighbors_2)
    
    jaccard = len(common) / len(union) if union else 0.0
    
    return {
        "common_neighbors": common,
        "count": len(common),
        "jaccard_similarity": jaccard
    }



def local_label_propagation(G: nx.Graph, node_id: Any, radius: int = 3, label_key: str = 'category') -> Any:
    """Propagate text labels from neighboring nodes using majority voting."""
    if node_id not in G:
        return None
        
    ego_graph = nx.ego_graph(G, node_id, radius=radius)
    
    label_counts = {}
    for n in ego_graph.nodes():
        if n == node_id:
            continue
        label = ego_graph.nodes[n].get(label_key)
        if label is not None:
            # Support unhashable label types such as lists by converting them to strings; assume scalar values here
            label_counts[label] = label_counts.get(label, 0) + 1
            
    if not label_counts:
        return None
        
    # Choose the label with the highest vote count
    max_label = max(label_counts.items(), key=lambda x: x[1])[0]
    return max_label

def simulate_edge_perturbation(G: nx.Graph, node_id: Any, remove_edges_list: List[tuple], use_ego_graph: bool = True) -> Dict[str, Any]:
    """Simulate local edge perturbations and measure their effect on topological reachability."""
    if node_id not in G:
        return {"error": "Node not in graph"}
        
    # Use the parameters to decide whether to compute on a 3-hop ego graph or the full graph
    if use_ego_graph:
        ego_graph = nx.ego_graph(G, node_id, radius=3).copy()
    else:
        ego_graph = G.copy()
    
    # Compute metrics before edge removal
    if node_id in ego_graph:
        if ego_graph.is_directed():
            original_reachable_nodes = set(nx.descendants(ego_graph, node_id)) | {node_id}
            original_components = nx.number_weakly_connected_components(ego_graph)
        else:
            original_reachable_nodes = nx.node_connected_component(ego_graph, node_id)
            original_components = nx.number_connected_components(ego_graph)
        original_reachability = len(original_reachable_nodes)
        original_density = nx.density(ego_graph)
        try:
            original_clustering = nx.average_clustering(ego_graph)
        except ZeroDivisionError:
            original_clustering = 0.0
    else:
        original_reachability = 0
        original_components = 0
        original_density = 0.0
        original_clustering = 0.0

    removed = 0
    for u, v in remove_edges_list:
        if ego_graph.has_edge(u, v):
            ego_graph.remove_edge(u, v)
            removed += 1
            
    # Compute metrics after edge removal
    if node_id in ego_graph:
        if ego_graph.is_directed():
            new_reachable_nodes = set(nx.descendants(ego_graph, node_id)) | {node_id}
            new_components = nx.number_weakly_connected_components(ego_graph)
        else:
            new_reachable_nodes = nx.node_connected_component(ego_graph, node_id)
            new_components = nx.number_connected_components(ego_graph)
        new_reachability = len(new_reachable_nodes)
        new_density = nx.density(ego_graph)
        try:
            new_clustering = nx.average_clustering(ego_graph)
        except ZeroDivisionError:
            new_clustering = 0.0
    else:
        new_reachability = 0
        new_components = 0
        new_density = 0.0
        new_clustering = 0.0
        
    return {
        "edges_removed_successfully": removed,
        "original_reachable_nodes_in_ego": original_reachability,
        "original_density": original_density,
        "original_clustering": original_clustering,
        "original_components": original_components,
        "new_reachable_nodes_in_ego": new_reachability,
        "new_density": new_density,
        "new_clustering": new_clustering,
        "new_components": new_components,
        "impact_ratio_reachability": (original_reachability - new_reachability) / original_reachability if original_reachability > 0 else 0.0,
        "impact_ratio_density": (original_density - new_density) / original_density if original_density > 0 else 0.0
    }

def check_local_reachability(G: nx.Graph, source_node: Any, target_node: Any, max_hops: int = 3) -> bool:
    """Check whether source_node can reach target_node within max_hops in the local network."""
    if source_node not in G or target_node not in G:
        return False
        
    try:
        length = nx.shortest_path_length(G, source=source_node, target=target_node)
        return length <= max_hops
    except nx.NetworkXNoPath:
        return False

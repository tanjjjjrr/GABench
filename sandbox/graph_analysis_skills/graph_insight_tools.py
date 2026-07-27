import networkx as nx
from typing import List, Dict, Any, Optional, Set

# ==============================================================================
# 1. Local neighborhood evolution and pruning graph operators
# ==============================================================================

def get_local_neighbors_by_hop(G: nx.Graph, node_id: Any, hop_num: int = 1, 
                               node_filters: Optional[Dict[str, Any]] = None, 
                               edge_filters: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Starting from the specified node_id, explore only the 1-hop neighbors outwards up to the specified hop_num (usually 1).
    """
    if node_id not in G:
        return []
        
    current_nodes = {node_id}
    visited = {node_id}
    
    for _ in range(hop_num):
        next_nodes = set()
        for u in current_nodes:
            # Compatible with neighbor traversal for directed and undirected graphs
            neighbors = G.successors(u) if G.is_directed() else G.neighbors(u)
            for v in neighbors:
                if v not in visited:
                    # Edge attribute filtering
                    if edge_filters:
                        edge_data = G.get_edge_data(u, v) or {}
                        # Handle multigraph cases, G.get_edge_data might return a data dict for multiple edges
                        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                            # As long as one edge satisfies the condition
                            edge_match = any(
                                all(e_data.get(k) == v_val for k, v_val in edge_filters.items())
                                for e_data in edge_data.values()
                            )
                        else:
                            edge_match = all(edge_data.get(k) == v_val for k, v_val in edge_filters.items())
                            
                        if not edge_match:
                            continue
                            
                    # Node attribute filtering
                    if node_filters:
                        node_data = G.nodes[v]
                        if not all(node_data.get(k) == v_val for k, v_val in node_filters.items()):
                            continue
                            
                    next_nodes.add(v)
                    visited.add(v)
        current_nodes = next_nodes
        
    return list(current_nodes)

def extract_ego_boundary(G: nx.Graph, node_id: Any, radius: int) -> List[Any]:
    """
    Cut out an induced subgraph centered at node_id with a given radius, and return only the nodes on the outermost boundary of this local subgraph.
    """
    if node_id not in G:
        return []
    
    # extract ego_graph with given radius
    ego_full = nx.ego_graph(G, node_id, radius=radius)
    # extract ego_graph with radius-1 as inner nodes
    if radius > 0:
        ego_inner = nx.ego_graph(G, node_id, radius=radius - 1)
        # boundary nodes = full - inner
        boundary = set(ego_full.nodes()) - set(ego_inner.nodes())
        return list(boundary)
    else:
        return [node_id]

def filter_nodes_by_all_degree(G: nx.Graph, node_list: List[Any], 
                               min_degree: Optional[int] = None, 
                               max_degree: Optional[int] = None) -> List[Any]:
    """
    In the current graph G, filter the passed node_list node set by total degree (All Degree).
    """
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
    """
    In the current graph G, filter the passed node_list node set by in-degree. For undirected graphs, it degrades to normal degree.
    """
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
    """
    In the current graph G, filter the passed node_list node set by out-degree. For undirected graphs, it degrades to normal degree.
    """
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
# 2. Local cycle and topological closed-loop operators
# ==============================================================================

def detect_local_cycles(G: nx.Graph, node_id: Any, max_depth: int = 4, exclude_nodes: set = None) -> List[List[Any]]:
    """
    Bounded Local Cycle Detection with depth limit.
    Algorithm complexity description:
    By using nx.all_simple_paths with cutoff=max_depth-1, time complexity is strictly bounded to O(V + E) (local spanning tree starting from node_id).
    Worst-case complexity is O(d^max_depth), where d is max degree. This keeps runtime at a controllable constant or neighborhood level, preventing full-graph explosion.
    """
    if node_id not in G:
        return []
        
    if exclude_nodes:
        # Use subgraph_view to quickly create a view of the graph, avoiding modifying original graph or deep copying
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
            # Find all length-bounded simple paths from neighbors back to the start point
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
                # For undirected graphs, exclude trivial path backtracking like node_id -> neighbor -> node_id
                if len(path) > 2:
                    cycles.append([node_id] + path)
                    
    return cycles

def compute_local_clustering(G: nx.Graph, node_id: Any) -> float:
    """
    Calculate Local Clustering Coefficient for a specified node.
    """
    if node_id not in G:
        return 0.0
    return float(nx.clustering(G, node_id))


# ==============================================================================
# 3. Local path guidance and multi-source convergence detection operators
# ==============================================================================

def find_stepped_shortest_path(G: nx.Graph, source_node: Any, target_node: Any, max_depth: int = 5) -> Optional[List[Any]]:
    """
    Stepped conditional shortest path search. Limits max depth to max_depth, returns None if exceeded.
    """
    if source_node not in G or target_node not in G:
        return None
        
    try:
        # Depth-truncated BFS search using cutoff
        paths = nx.single_source_shortest_path(G, source_node, cutoff=max_depth)
        if target_node in paths:
            return paths[target_node]
        return None
    except nx.NetworkXNoPath:
        return None

def detect_path_convergence(G: nx.Graph, source_node: Any, target_node: Any, mid_max_hop: int = 2) -> Dict[str, Any]:
    """
    Local bidirectional path convergence detector. Expands forward from source_node and backward from target_node by mid_max_hop steps, calculating the intersection.
    """
    result = {"convergence_nodes": [], "source_paths": {}, "target_paths": {}}
    if source_node not in G or target_node not in G:
        return result
        
    if G.is_directed():
        # source_node forward search
        source_paths = nx.single_source_shortest_path(G, source_node, cutoff=mid_max_hop)
        # target_node backward search (traceback), requires reversing directed graph
        G_rev = G.reverse(copy=False)
        target_paths_rev = nx.single_source_shortest_path(G_rev, target_node, cutoff=mid_max_hop)
        # Path reversal to align with direction
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
    """
    Local articulation point / topological bridge recognizer. Computes articulation points in the induced subgraph after extracting local ego_graph.
    """
    if node_id not in G:
        return []
        
    ego_graph = nx.ego_graph(G, node_id, radius=radius)
    
    if ego_graph.is_directed():
        ego_graph = ego_graph.to_undirected()
        
    # Find articulation points in undirected local subgraph
    cut_vertices = list(nx.articulation_points(ego_graph))
    return cut_vertices


# ==============================================================================
# 4. Local community and layer-by-layer core peeling operators
# ==============================================================================

def extract_local_k_core(G: nx.Graph, node_id: Any, target_k: int = 2, radius: int = 4) -> List[Any]:
    """
    Local k-Core dynamic attenuator. Extracts k-Core structure on local network centered at node_id.
    """
    if node_id not in G:
        return []
        
    ego_graph = nx.ego_graph(G, node_id, radius=radius)
    
    # Eliminate self-loop interference (if any) and directly extract target_k k-Core
    try:
        ego_graph.remove_edges_from(nx.selfloop_edges(ego_graph))
        k_core_subgraph = nx.k_core(ego_graph, k=target_k)
        return list(k_core_subgraph.nodes())
    except nx.NetworkXError:
        return []

def enumerate_local_cliques(G: nx.Graph, node_id: Any, exclude_nodes: set = None) -> List[List[Any]]:
    """
    Maximal clique local enumerator. Finds all maximal complete subgraphs containing node_id within its neighborhood.
    """
    if node_id not in G:
        return []
        
    # Convert graph to undirected to find cliques
    calc_G = G.to_undirected() if G.is_directed() else G
    
    # extract ego network to limit local computation, otherwise global cliques_containing_node will explode on billion-scale graphs
    # Radius can be 1, because all members of a maximal clique containing node_id must be within 1 hop of node_id
    ego_graph = nx.ego_graph(calc_G, node_id, radius=1)
    
    if exclude_nodes:
        nodes_to_remove = exclude_nodes.intersection(set(ego_graph.nodes()))
        ego_graph.remove_nodes_from(nodes_to_remove)
        
    if node_id not in ego_graph:
        return []
        
    # Replace cliques_containing_node, compatible with various NetworkX versions
    cliques = [c for c in nx.find_cliques(ego_graph) if node_id in c]
    return cliques

def compute_ego_density(G: nx.Graph, node_id: Any, radius: int = 2) -> float:
    """
    Ego-centric network densitometer. Calculates graph density of a local subgraph centered at node_id within a specified radius.
    """
    if node_id not in G:
        return 0.0
        
    ego_graph = nx.ego_graph(G, node_id, radius=radius)
    return nx.density(ego_graph)


# ==============================================================================
# 5. Local graph query and multi-modal / text similarity auxiliary operators
# ==============================================================================

def count_common_neighbors(G: nx.Graph, node_id_1: Any, node_id_2: Any) -> Dict[str, Any]:
    """
    Calculates the number of 1-hop common neighbors of two nodes on a graph, and returns their intersection node list along with Jaccard similarity coefficient.
    """
    if node_id_1 not in G or node_id_2 not in G:
        return {"common_neighbors": [], "count": 0, "jaccard_similarity": 0.0}
        
    if G.is_directed():
        # For directed graphs, common neighbors can be seen as the intersection of nodes they point to or that point to them,
        # For more general coverage, take the union of successors and predecessors
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
    """
    Neighborhood text label propagator. Conducts majority voting based on existing text labels of surrounding neighbors.
    """
    if node_id not in G:
        return None
        
    ego_graph = nx.ego_graph(G, node_id, radius=radius)
    
    label_counts = {}
    for n in ego_graph.nodes():
        if n == node_id:
            continue
        label = ego_graph.nodes[n].get(label_key)
        if label is not None:
            # Compatible with unhashable label types, e.g. list needs to be converted to string, assuming scalar here
            label_counts[label] = label_counts.get(label, 0) + 1
            
    if not label_counts:
        return None
        
    # Select the label with the highest vote (most frequent)
    max_label = max(label_counts.items(), key=lambda x: x[1])[0]
    return max_label


def check_local_reachability(G: nx.Graph, source_node: Any, target_node: Any, max_hops: int = 3) -> bool:
    """
    Ego-centric path reachability checker. Verify if source_node can reach target_node within max_hops.
    """
    if source_node not in G or target_node not in G:
        return False
        
    try:
        length = nx.shortest_path_length(G, source=source_node, target=target_node)
        return length <= max_hops
    except nx.NetworkXNoPath:
        return False

# ==============================================================================
# CLI (Command Line Interface) wrapper
# ==============================================================================
if __name__ == "__main__":
    import sys
    import json
    import pickle
    try:
        import torch
    except ImportError:
        torch = None

    def _load_graph(filepath):
        if filepath.endswith('.pt') and torch is not None:
            obj = torch.load(filepath, weights_only=False)
            if isinstance(obj, nx.Graph):
                return obj
            try:
                from torch_geometric.utils import to_networkx
                return to_networkx(obj, to_undirected=True)
            except ImportError:
                pass
        elif filepath.endswith(('.pkl', '.gpickle')):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.graphml'):
            return nx.read_graphml(filepath)
        elif filepath.endswith('.gml'):
            return nx.read_gml(filepath)
        raise ValueError(f"Cannot recognize or load graph file: {filepath}")

    if len(sys.argv) < 3:
        print("Usage: python graph_tools.py <action> <graph_file> [json_args]")
        print("Example: python graph_tools.py count_common_neighbors my_graph.pt '{\"node_id_1\": 1, \"node_id_2\": 2}'")
        sys.exit(1)
        
    action = sys.argv[1]
    graph_file = sys.argv[2]
    
    try:
        G = _load_graph(graph_file)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)
        
    args_dict = {}
    if len(sys.argv) > 3:
        try:
            args_dict = json.loads(sys.argv[3])
        except Exception as e:
            print(f"Error parsing JSON arguments: {e}")
            sys.exit(1)
            
    # Dynamic function dispatch
    if action in globals() and callable(globals()[action]):
        func = globals()[action]
        try:
            result = func(G, **args_dict)
            # Convert to JSON string for friendly bash output
            desc = {'connectivity_checking': 'JSON containing boolean for connectivity and integer for number of connected components.', 'get_undirected_graph_no_selfloops': 'JSON indicating success and path to the generated graph if saved.', 'extract_n_hop_ego_subgraph': 'JSON containing the nodes and edges of the extracted subgraph.', 'sample_n_hop_subgraph': 'JSON containing the sampled nodes and edges of the subgraph.', 'check_edge_existence': 'JSON containing a boolean indicating if the edge exists.', 'get_node_degree': 'JSON containing an integer representing the degree of the node.', 'get_node_neighbors': 'JSON containing a list of node IDs that are neighbors.', 'get_common_neighbors': 'JSON containing a list of node IDs that are common neighbors.', 'get_node_number': 'JSON containing an integer representing the total number of nodes.', 'get_edge_number': 'JSON containing an integer representing the total number of edges.', 'calculate_shortest_path': 'JSON containing a list of node IDs representing the shortest path.', 'get_bfs_edges': 'JSON containing a list of edges (node pairs) traversed during BFS.', 'get_dfs_edges': 'JSON containing a list of edges (node pairs) traversed during DFS.', 'has_path': 'JSON containing a boolean indicating if a path exists.', 'get_triangles': 'JSON containing an integer representing the number of triangles.', 'get_density': 'JSON containing a float representing the graph density.', 'get_closeness_centrality': 'JSON containing a float representing the closeness centrality.', 'get_betweenness_centrality': 'JSON containing a float representing the betweenness centrality.', 'get_harmonic_centrality': 'JSON containing a float representing the harmonic centrality.', 'get_degree_centrality': 'JSON containing a float representing the degree centrality.', 'get_avg_neighbor_degree': 'JSON containing a float representing the average neighbor degree.', 'get_wiener_index': 'JSON containing a float representing the Wiener index.', 'get_barycenter': 'JSON containing a list of node IDs that form the barycenter.', 'get_center': 'JSON containing a list of node IDs that form the center.', 'get_diameter': 'JSON containing an integer or float representing the graph diameter.', 'get_periphery': 'JSON containing a list of node IDs that form the periphery.', 'get_radius': 'JSON containing an integer or float representing the graph radius.', 'get_global_efficiency': 'JSON containing a float representing the global efficiency.', 'get_transitivity': 'JSON containing a float representing the transitivity.', 'get_clustering_coefficient': 'JSON containing a float representing the clustering coefficient.', 'get_constraint': 'JSON containing a float representing the constraint.', 'get_effective_size': 'JSON containing a float representing the effective size.', 'get_degree_assortativity': 'JSON containing a float representing the degree assortativity.', 'get_degree_connectivity': 'JSON containing a float representing the degree connectivity.', 'get_isolate_number': 'JSON containing an integer representing the number of isolated nodes.', 'get_connected_components_number': 'JSON containing an integer representing the number of connected components.', 'get_strongly_connected_components_number': 'JSON containing an integer representing the number of strongly connected components.', 'get_weakly_connected_components_number': 'JSON containing an integer representing the number of weakly connected components.', 'is_tournament': 'JSON containing a boolean indicating if the graph is a tournament.', 'is_bipartite': 'JSON containing a boolean indicating if the graph is bipartite.', 'is_eulerian': 'JSON containing a boolean indicating if the graph is Eulerian.', 'is_regular': 'JSON containing a boolean indicating if the graph is regular.', 'is_connected': 'JSON containing a boolean indicating if the graph is connected.', 'is_weakly_connected': 'JSON containing a boolean indicating if the graph is weakly connected.', 'is_strongly_connected': 'JSON containing a boolean indicating if the graph is strongly connected.', 'get_topological_sort': 'JSON containing a list of node IDs representing the topological sort order.', 'find_cycle': 'JSON containing a list of node IDs representing a cycle, if one exists.', 'get_dominating_set': 'JSON containing a list of node IDs that form a dominating set.', 'get_maximal_independent_set': 'JSON containing a list of node IDs that form a maximal independent set.', 'get_min_edge_cover': 'JSON containing a list of edges that form a minimum edge cover.', 'get_max_weight_matching': 'JSON containing a list of edges representing the maximum weight matching.', 'get_min_weight_matching': 'JSON containing a list of edges representing the minimum weight matching.', 'get_bridges': 'JSON containing a list of edges that are bridges.', 'get_minimum_spanning_edges': 'JSON containing a list of edges representing the minimum spanning tree.', 'get_maximum_flow': 'JSON containing a number representing the maximum flow value.', 'get_k_core': 'JSON containing the nodes and edges of the k-core subgraph.', 'get_maximal_cliques': 'JSON containing a list of lists of node IDs, each representing a maximal clique.', 'get_min_vertex_cover': 'JSON containing a list of node IDs that form a minimum vertex cover.', 'get_resource_allocation_index': 'JSON containing a float representing the resource allocation index.', 'get_jaccard_coefficient': 'JSON containing a float representing the Jaccard coefficient.', 'get_adamic_adar_index': 'JSON containing a float representing the Adamic-Adar index.', 'get_node_text': "JSON containing a string of the node's text or metadata.", 'find_node_by_text': 'JSON containing a list of node IDs that match the given text.', 'get_local_neighbors_by_hop': 'JSON containing lists of node IDs categorized by their hop distance.', 'extract_ego_boundary': 'JSON containing a list of boundary node IDs of the ego network.', 'filter_nodes_by_all_degree': 'JSON containing a list of node IDs meeting the total degree criteria.', 'filter_nodes_by_in_degree': 'JSON containing a list of node IDs meeting the in-degree criteria.', 'filter_nodes_by_out_degree': 'JSON containing a list of node IDs meeting the out-degree criteria.', 'detect_local_cycles': 'JSON containing a list of local cycles (each a list of node IDs).', 'compute_local_clustering': 'JSON containing a float representing the local clustering coefficient.', 'find_stepped_shortest_path': 'JSON containing a list of node IDs representing the stepped shortest path.', 'detect_path_convergence': 'JSON containing a list of bottleneck/convergence node IDs.', 'check_local_reachability': 'JSON containing a boolean indicating if local reachability exists.', 'find_local_bridge_nodes': 'JSON containing a list of local bridge node IDs.', 'extract_local_k_core': 'JSON containing the nodes and edges of the local k-core subgraph.', 'enumerate_local_cliques': 'JSON containing a list of local cliques (each a list of node IDs).', 'compute_ego_density': 'JSON containing a float representing the ego network density.', 'count_common_neighbors': 'JSON containing an integer representing the number of common neighbors.', 'local_label_propagation': 'JSON containing predicted labels or communities for the local nodes.', 'extract_key_to_numpy_with_pyg': 'A `.npy` file containing the extracted tensor data as a Numpy array.', 'extract_key_to_numpy_without_pyg': 'A `.npy` file containing the extracted tensor data as a Numpy array.', 'extract_graph_by_id_with_pyg': 'A `.pt` file containing the isolated single graph instance.', 'structural_node_embedding_classification_train': 'A trained model state saved as a JSON or PT file, along with training logs.', 'structural_node_embedding_classification_infer': 'A `.npy` file containing the predicted category labels for nodes.', 'structural_node_embedding_link_prediction_train': 'A trained model state saved as a JSON or PT file, along with training logs.', 'structural_node_embedding_link_prediction_infer': 'A `.npy` file containing the predicted existence probabilities of queried edges.', 'structural_graph_embedding_classification_train': 'A trained model state saved as a JSON or PT file, along with training logs.', 'structural_graph_embedding_classification_infer': 'A `.npy` file containing the predicted category labels for graphs.', 'inspect_pt_with_pyg': 'JSON containing the structural contents, keys, and feature dimensions of the dataset.', 'inspect_pt_without_pyg': 'JSON containing the structural contents, keys, and feature dimensions of the dataset.', 'extract_key_with_pyg': 'A `.json` file containing the serialized graph structures or requested features.', 'extract_key_without_pyg': 'A `.json` file containing the serialized graph structures or requested features.'}.get(action, '')
            if desc:
                print(f"[Output Description]: {desc}")
            print(json.dumps(result, default=str, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error executing {action}: {e}")
    else:
        print(f"Unknown action: {action}")

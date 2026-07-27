import random
from collections import deque
import networkx as nx

def connectivity_checking(graph, directed):
    """
    Check graph connectivity (normal connectivity for undirected graph, strong connectivity for directed graph).
    """
    print("[Output Description]: JSON containing boolean for connectivity and integer for number of connected components.")
    if not directed:
        return nx.is_connected(graph)
    return nx.is_strongly_connected(graph)

def get_undirected_graph_no_selfloops(graph):
    """
    Convert to undirected graph and remove self-loops (to support certain graph algorithms).
    """
    print("[Output Description]: JSON indicating success and path to the generated graph if saved.")
    G_und = graph.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))
    return G_und

def extract_n_hop_ego_subgraph(graph, center_node=None, hops=3, min_nodes=5, max_nodes=500):
    """
    Extract an n-hop completely induced subgraph centered on a node.
    
    Args:
        graph: Original graph (NetworkX Graph/DiGraph)
        center_node: Center node, if None then randomly selected
        hops: Number of hops to extract (n-hop), can be specified via parameter to avoid hardcoding
        min_nodes: Minimum number of nodes required for subgraph
        max_nodes: Maximum number of nodes required for subgraph
        
    Returns:
        H: Extracted n-hop induced subgraph
        actual_center: Actual center node used
        actual_hops: Actual number of hops used
    """
    print("[Output Description]: JSON containing the nodes and edges of the extracted subgraph.")
    max_attempts = 100
    for _ in range(max_attempts):
        if center_node is None:
            current_center = random.choice(list(graph.nodes()))
        else:
            current_center = center_node

        sampled = {current_center}
        queue = deque([(current_center, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in sampled:
                    sampled.add(neighbor)
                    queue.append((neighbor, depth + 1))

        H = graph.subgraph(sampled).copy()
        num_nodes = H.number_of_nodes()
        
        # Check if the number of nodes is within a reasonable range
        if min_nodes <= num_nodes <= max_nodes:
            return H, current_center, hops

        # If center node is forced and requirements are not met, return directly (or throw an error)
        # Return itself for fault tolerance here
        if center_node is not None:
            return H, center_node, hops

    # Multiple random attempts failed, return the result of the last attempt
    return H, current_center, hops

def sample_n_hop_subgraph(G, target_size='Tiny', hops=3):
    """
    Randomly sample in the whole graph and truncate to target size n-hop subgraph.
    """
    print("[Output Description]: JSON containing the sampled nodes and edges of the subgraph.")
    if target_size == 'Tiny':
        n_target = random.randint(5, 36)
    elif target_size == 'Small':
        n_target = random.randint(36, 101)
    elif target_size == 'Medium':
        n_target = random.randint(100, 501)
    else:
        n_target = random.randint(5, 36)

    max_nodes = 500
    nodes = list(G.nodes())
    max_attempts = 500

    for _ in range(max_attempts):
        center = random.choice(nodes)
        sampled = {center}
        queue = deque([(center, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbor in G.neighbors(node):
                if neighbor not in sampled:
                    sampled.add(neighbor)
                    queue.append((neighbor, depth + 1))

        if len(sampled) < 10:
            continue

        # If n-hop subgraph is too large, try gradually reducing hops (n-1, n-2... down to 1 hop)
        if len(sampled) > max_nodes:
            current_hops = hops - 1
            while len(sampled) > max_nodes and current_hops > 0:
                sampled = {center}
                queue = deque([(center, 0)])
                while queue:
                    node, depth = queue.popleft()
                    if depth >= current_hops:
                        continue
                    for neighbor in G.neighbors(node):
                        if neighbor not in sampled:
                            sampled.add(neighbor)
                            queue.append((neighbor, depth + 1))
                current_hops -= 1

        if len(sampled) >= 5:
            if len(sampled) > n_target:
                sampled = set(random.sample(list(sampled), n_target))
            H = G.subgraph(sampled).copy()
            if H.number_of_edges() >= 1 and nx.is_connected(H.to_undirected()):
                return H

    return G.subgraph({random.choice(nodes)}).copy()


# ==========================================
# NetworkX Graph Algorithm Wrappers
# ==========================================

# --- Basic Node/Edge Queries ---
def check_edge_existence(graph, u, v):
    print("[Output Description]: JSON containing a boolean indicating if the edge exists.")
    return graph.has_edge(u, v)

def get_node_degree(graph, node):
    print("[Output Description]: JSON containing an integer representing the degree of the node.")
    return graph.degree[node] if hasattr(graph.degree, '__getitem__') else graph.degree(node)

def get_node_neighbors(graph, node):
    print("[Output Description]: JSON containing a list of node IDs that are neighbors.")
    return list(graph.neighbors(node))

def get_common_neighbors(graph, u, v):
    print("[Output Description]: JSON containing a list of node IDs that are common neighbors.")
    return list(nx.common_neighbors(graph, u, v))

def get_node_number(graph):
    print("[Output Description]: JSON containing an integer representing the total number of nodes.")
    return graph.number_of_nodes()

def get_edge_number(graph):
    print("[Output Description]: JSON containing an integer representing the total number of edges.")
    return graph.number_of_edges()

# --- Paths & Traversals ---
def calculate_shortest_path(graph, source, target, weight=None):
    print("[Output Description]: JSON containing a list of node IDs representing the shortest path.")
    return nx.shortest_path(graph, source=source, target=target, weight=weight)

def get_bfs_edges(graph, source):
    print("[Output Description]: JSON containing a list of edges (node pairs) traversed during BFS.")
    return list(nx.bfs_edges(graph, source=source))

def get_dfs_edges(graph, source):
    print("[Output Description]: JSON containing a list of edges (node pairs) traversed during DFS.")
    return list(nx.dfs_edges(graph, source=source))

def has_path(graph, source, target):
    print("[Output Description]: JSON containing a boolean indicating if a path exists.")
    return nx.has_path(graph, source, target)

# --- Graph Metrics & Centrality ---
def get_triangles(graph, node=None):
    print("[Output Description]: JSON containing an integer representing the number of triangles.")
    return nx.triangles(graph, node)

def get_density(graph):
    print("[Output Description]: JSON containing a float representing the graph density.")
    return nx.density(graph)

def get_closeness_centrality(graph, node=None):
    print("[Output Description]: JSON containing a float representing the closeness centrality.")
    c = nx.closeness_centrality(graph)
    return c[node] if node is not None else c

def get_betweenness_centrality(graph, node=None):
    print("[Output Description]: JSON containing a float representing the betweenness centrality.")
    c = nx.betweenness_centrality(graph)
    return c[node] if node is not None else c

def get_harmonic_centrality(graph, node=None):
    print("[Output Description]: JSON containing a float representing the harmonic centrality.")
    c = nx.harmonic_centrality(graph)
    return c[node] if node is not None else c

def get_degree_centrality(graph, node=None):
    print("[Output Description]: JSON containing a float representing the degree centrality.")
    c = nx.degree_centrality(graph)
    return c[node] if node is not None else c

def get_avg_neighbor_degree(graph, node=None):
    print("[Output Description]: JSON containing a float representing the average neighbor degree.")
    c = nx.average_neighbor_degree(graph)
    return c[node] if node is not None else c

def get_wiener_index(graph):
    print("[Output Description]: JSON containing a float representing the Wiener index.")
    return nx.wiener_index(graph)

def get_barycenter(graph):
    print("[Output Description]: JSON containing a list of node IDs that form the barycenter.")
    return nx.barycenter(graph)

def get_center(graph):
    print("[Output Description]: JSON containing a list of node IDs that form the center.")
    return nx.center(graph)

def get_diameter(graph):
    print("[Output Description]: JSON containing an integer or float representing the graph diameter.")
    return nx.diameter(graph)

def get_periphery(graph):
    print("[Output Description]: JSON containing a list of node IDs that form the periphery.")
    return nx.periphery(graph)

def get_radius(graph):
    print("[Output Description]: JSON containing an integer or float representing the graph radius.")
    return nx.radius(graph)

def get_global_efficiency(graph):
    print("[Output Description]: JSON containing a float representing the global efficiency.")
    return nx.global_efficiency(graph)

def get_transitivity(graph):
    print("[Output Description]: JSON containing a float representing the transitivity.")
    return nx.transitivity(graph)

def get_clustering_coefficient(graph, node=None):
    print("[Output Description]: JSON containing a float representing the clustering coefficient.")
    return nx.clustering(graph, node)

def get_constraint(graph, nodes=None):
    print("[Output Description]: JSON containing a float representing the constraint.")
    return nx.constraint(graph, nodes)

def get_effective_size(graph, nodes=None):
    print("[Output Description]: JSON containing a float representing the effective size.")
    return nx.effective_size(graph, nodes)

def get_degree_assortativity(graph):
    print("[Output Description]: JSON containing a float representing the degree assortativity.")
    return nx.degree_assortativity_coefficient(graph)

def get_degree_connectivity(graph):
    print("[Output Description]: JSON containing a float representing the degree connectivity.")
    return nx.degree_connectivity(graph)

# --- Components & Connectivity ---
def get_isolate_number(graph):
    print("[Output Description]: JSON containing an integer representing the number of isolated nodes.")
    return nx.number_of_isolates(graph)

def get_connected_components_number(graph):
    print("[Output Description]: JSON containing an integer representing the number of connected components.")
    return nx.number_connected_components(graph)

def get_strongly_connected_components_number(graph):
    print("[Output Description]: JSON containing an integer representing the number of strongly connected components.")
    return nx.number_strongly_connected_components(graph)

def get_weakly_connected_components_number(graph):
    print("[Output Description]: JSON containing an integer representing the number of weakly connected components.")
    return nx.number_weakly_connected_components(graph)

def is_tournament(graph):
    print("[Output Description]: JSON containing a boolean indicating if the graph is a tournament.")
    return nx.is_tournament(graph)

def is_bipartite(graph):
    print("[Output Description]: JSON containing a boolean indicating if the graph is bipartite.")
    return nx.is_bipartite(graph)

def is_eulerian(graph):
    print("[Output Description]: JSON containing a boolean indicating if the graph is Eulerian.")
    return nx.is_eulerian(graph)

def is_regular(graph):
    print("[Output Description]: JSON containing a boolean indicating if the graph is regular.")
    return nx.is_regular(graph)

def is_connected(graph):
    print("[Output Description]: JSON containing a boolean indicating if the graph is connected.")
    return nx.is_connected(graph)

def is_weakly_connected(graph):
    print("[Output Description]: JSON containing a boolean indicating if the graph is weakly connected.")
    return nx.is_weakly_connected(graph)

def is_strongly_connected(graph):
    print("[Output Description]: JSON containing a boolean indicating if the graph is strongly connected.")
    return nx.is_strongly_connected(graph)

# --- Advanced Algorithms ---
def get_topological_sort(graph):
    print("[Output Description]: JSON containing a list of node IDs representing the topological sort order.")
    return list(nx.topological_sort(graph))

def find_cycle(graph):
    print("[Output Description]: JSON containing a list of node IDs representing a cycle, if one exists.")
    return list(nx.find_cycle(graph))

def get_dominating_set(graph):
    print("[Output Description]: JSON containing a list of node IDs that form a dominating set.")
    return nx.dominating_set(graph)

def get_maximal_independent_set(graph):
    print("[Output Description]: JSON containing a list of node IDs that form a maximal independent set.")
    return nx.maximal_independent_set(graph)

def get_min_edge_cover(graph):
    print("[Output Description]: JSON containing a list of edges that form a minimum edge cover.")
    return nx.min_edge_cover(graph)

def get_max_weight_matching(graph):
    print("[Output Description]: JSON containing a list of edges representing the maximum weight matching.")
    return nx.max_weight_matching(graph)

def get_min_weight_matching(graph):
    print("[Output Description]: JSON containing a list of edges representing the minimum weight matching.")
    return nx.min_weight_matching(graph)

def get_bridges(graph):
    print("[Output Description]: JSON containing a list of edges that are bridges.")
    return list(nx.bridges(graph))

def get_minimum_spanning_edges(graph, weight="weight"):
    print("[Output Description]: JSON containing a list of edges representing the minimum spanning tree.")
    return list(nx.minimum_spanning_edges(graph, weight=weight))

def get_maximum_flow(graph, s, t):
    print("[Output Description]: JSON containing a number representing the maximum flow value.")
    return nx.maximum_flow(graph, s, t)

def get_k_core(graph, k=None):
    print("[Output Description]: JSON containing the nodes and edges of the k-core subgraph.")
    return nx.k_core(graph, k)

def get_maximal_cliques(graph):
    print("[Output Description]: JSON containing a list of lists of node IDs, each representing a maximal clique.")
    return list(nx.find_cliques(graph))

def is_vertex_cover(G, vertex_set):
    vertex_set = set(vertex_set)
    for u, v in G.edges():
        if u not in vertex_set and v not in vertex_set:
            return False
    return True

def min_vertex_cover_branch_and_bound(G):
    nodes = list(G.nodes())
    best_cover = set(nodes)
    
    def branch_and_bound(G, current_cover, remaining_nodes):
        nonlocal best_cover
        
        if len(current_cover) >= len(best_cover):
            return
        
        if not remaining_nodes:
            if is_vertex_cover(G, current_cover):
                best_cover = current_cover.copy()
            return
        
        node = remaining_nodes[0]
        new_remaining = remaining_nodes[1:]
        
        branch_and_bound(G, current_cover | {node}, new_remaining)
        
        if G.is_directed():
            edges_to_cover = list(G.out_edges(node)) + list(G.in_edges(node))
        else:
            edges_to_cover = list(G.edges(node))
        
        nodes_to_add = set()
        for u, v in edges_to_cover:
            if u != node and v != node:
                continue
            other_node = v if u == node else u
            if other_node not in current_cover:
                nodes_to_add.add(other_node)
        
        new_cover = current_cover | nodes_to_add
        remaining_filtered = [n for n in new_remaining if n not in nodes_to_add]
        branch_and_bound(G, new_cover, remaining_filtered)
    
    branch_and_bound(G, set(), nodes)
    return best_cover

def min_vertex_cover_ilp(G):
    from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
    model = LpProblem(name="min_vertex_cover", sense=LpMinimize)
    x = {node: LpVariable(name=f"x_{node}", cat='Binary') for node in G.nodes()}
    
    model += lpSum(x.values())
    
    for u, v in G.edges():
        model += x[u] + x[v] >= 1
    
    model.solve(PULP_CBC_CMD(msg=False))
    min_cover = {node for node in G.nodes() if x[node].value() > 0.5}
    return min_cover

def get_min_vertex_cover(graph):
    print("[Output Description]: JSON containing a list of node IDs that form a minimum vertex cover.")
    if len(graph.nodes()) <= 20:
        return list(min_vertex_cover_branch_and_bound(graph))
    else:
        return list(min_vertex_cover_ilp(graph))

# --- Link Prediction ---
def get_resource_allocation_index(graph, ebunch=None):
    print("[Output Description]: JSON containing a float representing the resource allocation index.")
    return list(nx.resource_allocation_index(graph, ebunch))

def get_jaccard_coefficient(graph, ebunch=None):
    print("[Output Description]: JSON containing a float representing the Jaccard coefficient.")
    return list(nx.jaccard_coefficient(graph, ebunch))

def get_adamic_adar_index(graph, ebunch=None):
    print("[Output Description]: JSON containing a float representing the Adamic-Adar index.")
    return list(nx.adamic_adar_index(graph, ebunch))


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
        print("Usage: python graph_theory_tools.py <action> <graph_file> [json_args]")
        print("Example: python graph_theory_tools.py check_edge_existence my_graph.pt '{\"u\": 1, \"v\": 2}'")
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
            
            # Helper function to save graph objects
            def save_graph_if_needed(obj):
                if isinstance(obj, nx.Graph):
                    import uuid
                    import os
                    base_name = f"subgraph_{uuid.uuid4().hex[:6]}"
                    if torch is not None:
                        saved_path = f"{base_name}.pt"
                        torch.save(obj, saved_path)
                    else:
                        saved_path = f"{base_name}.pkl"
                        with open(saved_path, 'wb') as f:
                            pickle.dump(obj, f)
                    return {"message": "Subgraph extracted and saved", "saved_path": os.path.abspath(saved_path)}
                return obj

            if isinstance(result, tuple):
                result = tuple(save_graph_if_needed(item) for item in result)
            else:
                result = save_graph_if_needed(result)

            # Convert to JSON string for friendly bash output
            desc = {'connectivity_checking': 'JSON containing boolean for connectivity and integer for number of connected components.', 'get_undirected_graph_no_selfloops': 'JSON indicating success and path to the generated graph if saved.', 'extract_n_hop_ego_subgraph': 'JSON containing the nodes and edges of the extracted subgraph.', 'sample_n_hop_subgraph': 'JSON containing the sampled nodes and edges of the subgraph.', 'check_edge_existence': 'JSON containing a boolean indicating if the edge exists.', 'get_node_degree': 'JSON containing an integer representing the degree of the node.', 'get_node_neighbors': 'JSON containing a list of node IDs that are neighbors.', 'get_common_neighbors': 'JSON containing a list of node IDs that are common neighbors.', 'get_node_number': 'JSON containing an integer representing the total number of nodes.', 'get_edge_number': 'JSON containing an integer representing the total number of edges.', 'calculate_shortest_path': 'JSON containing a list of node IDs representing the shortest path.', 'get_bfs_edges': 'JSON containing a list of edges (node pairs) traversed during BFS.', 'get_dfs_edges': 'JSON containing a list of edges (node pairs) traversed during DFS.', 'has_path': 'JSON containing a boolean indicating if a path exists.', 'get_triangles': 'JSON containing an integer representing the number of triangles.', 'get_density': 'JSON containing a float representing the graph density.', 'get_closeness_centrality': 'JSON containing a float representing the closeness centrality.', 'get_betweenness_centrality': 'JSON containing a float representing the betweenness centrality.', 'get_harmonic_centrality': 'JSON containing a float representing the harmonic centrality.', 'get_degree_centrality': 'JSON containing a float representing the degree centrality.', 'get_avg_neighbor_degree': 'JSON containing a float representing the average neighbor degree.', 'get_wiener_index': 'JSON containing a float representing the Wiener index.', 'get_barycenter': 'JSON containing a list of node IDs that form the barycenter.', 'get_center': 'JSON containing a list of node IDs that form the center.', 'get_diameter': 'JSON containing an integer or float representing the graph diameter.', 'get_periphery': 'JSON containing a list of node IDs that form the periphery.', 'get_radius': 'JSON containing an integer or float representing the graph radius.', 'get_global_efficiency': 'JSON containing a float representing the global efficiency.', 'get_transitivity': 'JSON containing a float representing the transitivity.', 'get_clustering_coefficient': 'JSON containing a float representing the clustering coefficient.', 'get_constraint': 'JSON containing a float representing the constraint.', 'get_effective_size': 'JSON containing a float representing the effective size.', 'get_degree_assortativity': 'JSON containing a float representing the degree assortativity.', 'get_degree_connectivity': 'JSON containing a float representing the degree connectivity.', 'get_isolate_number': 'JSON containing an integer representing the number of isolated nodes.', 'get_connected_components_number': 'JSON containing an integer representing the number of connected components.', 'get_strongly_connected_components_number': 'JSON containing an integer representing the number of strongly connected components.', 'get_weakly_connected_components_number': 'JSON containing an integer representing the number of weakly connected components.', 'is_tournament': 'JSON containing a boolean indicating if the graph is a tournament.', 'is_bipartite': 'JSON containing a boolean indicating if the graph is bipartite.', 'is_eulerian': 'JSON containing a boolean indicating if the graph is Eulerian.', 'is_regular': 'JSON containing a boolean indicating if the graph is regular.', 'is_connected': 'JSON containing a boolean indicating if the graph is connected.', 'is_weakly_connected': 'JSON containing a boolean indicating if the graph is weakly connected.', 'is_strongly_connected': 'JSON containing a boolean indicating if the graph is strongly connected.', 'get_topological_sort': 'JSON containing a list of node IDs representing the topological sort order.', 'find_cycle': 'JSON containing a list of node IDs representing a cycle, if one exists.', 'get_dominating_set': 'JSON containing a list of node IDs that form a dominating set.', 'get_maximal_independent_set': 'JSON containing a list of node IDs that form a maximal independent set.', 'get_min_edge_cover': 'JSON containing a list of edges that form a minimum edge cover.', 'get_max_weight_matching': 'JSON containing a list of edges representing the maximum weight matching.', 'get_min_weight_matching': 'JSON containing a list of edges representing the minimum weight matching.', 'get_bridges': 'JSON containing a list of edges that are bridges.', 'get_minimum_spanning_edges': 'JSON containing a list of edges representing the minimum spanning tree.', 'get_maximum_flow': 'JSON containing a number representing the maximum flow value.', 'get_k_core': 'JSON containing the nodes and edges of the k-core subgraph.', 'get_maximal_cliques': 'JSON containing a list of lists of node IDs, each representing a maximal clique.', 'get_min_vertex_cover': 'JSON containing a list of node IDs that form a minimum vertex cover.', 'get_resource_allocation_index': 'JSON containing a float representing the resource allocation index.', 'get_jaccard_coefficient': 'JSON containing a float representing the Jaccard coefficient.', 'get_adamic_adar_index': 'JSON containing a float representing the Adamic-Adar index.', 'get_node_text': "JSON containing a string of the node's text or metadata.", 'find_node_by_text': 'JSON containing a list of node IDs that match the given text.', 'get_local_neighbors_by_hop': 'JSON containing lists of node IDs categorized by their hop distance.', 'extract_ego_boundary': 'JSON containing a list of boundary node IDs of the ego network.', 'filter_nodes_by_all_degree': 'JSON containing a list of node IDs meeting the total degree criteria.', 'filter_nodes_by_in_degree': 'JSON containing a list of node IDs meeting the in-degree criteria.', 'filter_nodes_by_out_degree': 'JSON containing a list of node IDs meeting the out-degree criteria.', 'detect_local_cycles': 'JSON containing a list of local cycles (each a list of node IDs).', 'compute_local_clustering': 'JSON containing a float representing the local clustering coefficient.', 'find_stepped_shortest_path': 'JSON containing a list of node IDs representing the stepped shortest path.', 'detect_path_convergence': 'JSON containing a list of bottleneck/convergence node IDs.', 'check_local_reachability': 'JSON containing a boolean indicating if local reachability exists.', 'find_local_bridge_nodes': 'JSON containing a list of local bridge node IDs.', 'extract_local_k_core': 'JSON containing the nodes and edges of the local k-core subgraph.', 'enumerate_local_cliques': 'JSON containing a list of local cliques (each a list of node IDs).', 'compute_ego_density': 'JSON containing a float representing the ego network density.', 'count_common_neighbors': 'JSON containing an integer representing the number of common neighbors.', 'local_label_propagation': 'JSON containing predicted labels or communities for the local nodes.', 'extract_key_to_numpy_with_pyg': 'A `.npy` file containing the extracted tensor data as a Numpy array.', 'extract_key_to_numpy_without_pyg': 'A `.npy` file containing the extracted tensor data as a Numpy array.', 'extract_graph_by_id_with_pyg': 'A `.pt` file containing the isolated single graph instance.', 'structural_node_embedding_classification_train': 'A trained model state saved as a JSON or PT file, along with training logs.', 'structural_node_embedding_classification_infer': 'A `.npy` file containing the predicted category labels for nodes.', 'structural_node_embedding_link_prediction_train': 'A trained model state saved as a JSON or PT file, along with training logs.', 'structural_node_embedding_link_prediction_infer': 'A `.npy` file containing the predicted existence probabilities of queried edges.', 'structural_graph_embedding_classification_train': 'A trained model state saved as a JSON or PT file, along with training logs.', 'structural_graph_embedding_classification_infer': 'A `.npy` file containing the predicted category labels for graphs.', 'inspect_pt_with_pyg': 'JSON containing the structural contents, keys, and feature dimensions of the dataset.', 'inspect_pt_without_pyg': 'JSON containing the structural contents, keys, and feature dimensions of the dataset.', 'extract_key_with_pyg': 'A `.json` file containing the serialized graph structures or requested features.', 'extract_key_without_pyg': 'A `.json` file containing the serialized graph structures or requested features.'}.get(action, '')
            if desc:
                print(f"[Output Description]: {desc}")
            print(json.dumps(result, default=str, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error executing {action}: {e}")
    else:
        print(f"Unknown action: {action}")

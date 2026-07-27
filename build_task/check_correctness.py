import networkx as nx
from collections import deque

from algorithms import min_vertex_cover


def is_valid_bfs_sequence(graph, edge_sequence, start_node=None):
    if not edge_sequence:
        return len(list(graph.neighbors(start_node))) == 0

    G = graph.copy()
    is_directed = isinstance(G, nx.DiGraph)
    if start_node not in G.nodes():
        return False

    # Keep track of discovered nodes and edges
    discovered_nodes = {start_node}
    discovered_edges = set()
    
    queue = deque([start_node])
    edge_index = 0
    
    while queue and edge_index < len(edge_sequence):
        current_node = queue.popleft()
        
        # Get all unvisited neighbors of the current node
        unvisited_neighbors = []
        for neighbor in G.neighbors(current_node):
            edge = (current_node, neighbor)
            reverse_edge = (neighbor, current_node)
            
            # Check if this edge has been discovered
            edge_discovered = edge in discovered_edges
            reverse_edge_discovered = reverse_edge in discovered_edges if not is_directed else False
                
            # Check if this neighbor has been discovered
            if neighbor not in discovered_nodes and not (edge_discovered or reverse_edge_discovered):
                unvisited_neighbors.append(neighbor)
        
        possible_next_edges = [(current_node, neighbor) for neighbor in unvisited_neighbors]
        while possible_next_edges and edge_index < len(edge_sequence):
            current_edge = edge_sequence[edge_index]
            edge_index += 1
            
            # For undirected graphs, normalize the edge orientation
            if not is_directed and current_edge[::-1] in G.edges():
                if current_edge not in G.edges():
                    current_edge = current_edge[::-1]
            
            # The edge must exist in the graph
            if current_edge not in G.edges():
                return False
                
            u, v = current_edge
            
            # One of u or v must be the current node from the BFS queue
            if u != current_node and v != current_node:
                return False
                
            # Ensure the source node is the one we're currently processing
            if v == current_node:
                u, v = v, u 
                if is_directed and (v, u) not in G.edges():
                    return False
            
            # The destination node must be unvisited
            if v in discovered_nodes:
                return False
                
            # Mark the edge and destination node as discovered
            discovered_edges.add(current_edge)
            if not is_directed:
                discovered_edges.add((v, u))
            discovered_nodes.add(v)
            
            queue.append(v)
            if (u, v) in possible_next_edges:
                possible_next_edges.remove((u, v))
            elif not is_directed and (v, u) in possible_next_edges:
                possible_next_edges.remove((v, u))
            else:
                return False
    
    return edge_index == len(edge_sequence)


def is_valid_dfs_sequence(graph, edge_sequence, start_node=None):
    if not edge_sequence:
        return len(list(graph.neighbors(start_node))) == 0

    G = graph.copy()
    is_directed = isinstance(G, nx.DiGraph)
    
    if start_node not in G.nodes():
        return False
    
    # Keep track of discovered nodes and edges
    discovered_nodes = {start_node}
    discovered_edges = set()
    
    stack = [start_node]
    edge_index = 0
    backtracks = []
    
    while stack and edge_index < len(edge_sequence):
        current_node = stack[-1]  # Peek at the top of the stack
        
        # Get all unvisited neighbors of the current node
        unvisited_neighbors = []
        for neighbor in G.neighbors(current_node):
            edge = (current_node, neighbor)
            reverse_edge = (neighbor, current_node)
            
            # Check if this edge has been discovered
            edge_discovered = edge in discovered_edges
            reverse_edge_discovered = reverse_edge in discovered_edges if not is_directed else False
                
            # Check if this neighbor has been discovered
            if neighbor not in discovered_nodes and not (edge_discovered or reverse_edge_discovered):
                unvisited_neighbors.append(neighbor)
        
        # If no unvisited neighbors, we need to backtrack
        if not unvisited_neighbors:
            stack.pop()  # Backtrack
            backtracks.append(current_node)
            continue
        
        # Get current edge from sequence
        current_edge = edge_sequence[edge_index]
        edge_index += 1
        
        # For undirected graphs, normalize the edge orientation
        if not is_directed and current_edge[::-1] in G.edges():
            if current_edge not in G.edges():
                current_edge = current_edge[::-1]
        
        # The edge must exist in the graph
        if current_edge not in G.edges():
            return False
            
        u, v = current_edge
        
        # One of u or v must be the current node from the DFS stack
        if u != current_node and v != current_node:
            return False
            
        # Ensure the source node is the one we're currently processing
        if v == current_node:
            u, v = v, u  # swap to make u the current node
            if is_directed and (v, u) not in G.edges():
                return False  # Can't traverse against edge direction in directed graph
        
        # The destination node must be unvisited
        if v in discovered_nodes:
            return False
            
        # Mark the edge and destination node as discovered
        discovered_edges.add(current_edge)
        if not is_directed:
            discovered_edges.add((v, u))  # Add reverse edge for undirected graphs
        discovered_nodes.add(v)
        
        # Push the newly discovered node to the stack
        stack.append(v)
    
    # If we processed all edges in the sequence, it's valid
    return edge_index == len(edge_sequence)


def is_valid_dominating_set(graph, node_set):
    return nx.is_dominating_set(graph, node_set)


def is_maximal_independent_set(G, node_set, include_node):
    node_set = set(node_set)
    if include_node not in node_set:
        return False

    # For each pair of nodes in the set, they should not be adjacent
    for u in node_set:
        for v in node_set:
            if u != v and G.has_edge(u, v):
                return False
    
    # For each node outside the set, it must be adjacent to at least one node in the set
    remaining_nodes = set(G.nodes()) - node_set
    for node in remaining_nodes:
        has_neighbor_in_set = False
        for u in node_set:
            if G.has_edge(node, u):
                has_neighbor_in_set = True
                break
        if not has_neighbor_in_set:
            return False
    return True


def is_minimum_edge_cover(G, edge_set, min_cover_size=None):
    if list(nx.isolates(G)):
        # If there are isolated nodes, no edge cover exists
        return False
    
    # Convert edge_set to a set if it's a list
    edge_set = set(tuple(sorted((u, v))) for u, v in edge_set)
    
    # Step 1: Check if edge_set is an edge cover
    # Every vertex must be incident to at least one edge in edge_set
    covered_vertices = set()
    for u, v in edge_set:
        covered_vertices.add(u)
        covered_vertices.add(v)
    
    if covered_vertices != set(G.nodes()):
        return False
    
    # Step 2: Check if edge_set is minimum by comparing with the provided size
    if min_cover_size is None:
        min_cover = nx.min_edge_cover(G)
        min_cover_size = len(min_cover)
    
    return len(edge_set) == min_cover_size


def is_minimum_vertex_cover(G, vertex_set, min_cover_size=None):
    vertex_set = set(vertex_set)
    
    # Step 1: Check if vertex_set is a valid vertex cover
    for u, v in G.edges():
        if u not in vertex_set and v not in vertex_set:
            return False
    
    # Step 2: Check if the size of vertex_set equals the minimum cover size
    if min_cover_size is None:
        min_cover_size = min_vertex_cover(G)
        
    return len(vertex_set) == len(min_cover_size)


def is_minimum_spanning_tree(graph, edge_list, minial_weight=None):
    num_nodes = graph.number_of_nodes()
    
    tree = nx.Graph()
    tree.add_nodes_from(graph.nodes())  
    tree.add_edges_from(edge_list)
    
    # Check if the tree is connected
    if not nx.is_connected(tree):
        print('not connected')
        return False
    
    # Check if the number of edges is exactly (num_nodes - 1)
    if len(edge_list) != num_nodes - 1:
        print('not nodes-1')
        return False
    
    # Check if there are no cycles (a tree has no cycles)
    if not nx.is_tree(tree):
        print('not a tree')
        return False
    
    # For unweighted graphs, any spanning tree with (n-1) edges is a minimum spanning tree
    # For weighted graphs, we would need to compare with the actual MST
    return True


def is_valid_shortest_path(G, path, source, target, shortest_length=None):
    if not path or path[0] != source or path[-1] != target:
        return False
    
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return False
    
    path_length = sum(G[path[i]][path[i + 1]].get('weight', 1) for i in range(len(path) - 1))
    
    if shortest_length is None:
        shortest_length = nx.shortest_path_length(G, source=source, target=target, weight='weight')
    return path_length == shortest_length


def is_maximal_weight_matching(G, edge_sequence, maximal_weight=None):
    # Check if the edge sequence forms a valid matching
    vertices_used = set()
    matching_weight = 0
    
    matching = nx.Graph()
    for edge in edge_sequence:
        u, v = edge
        
        # Check if the edge exists in the original graph
        if not G.has_edge(u, v):
            return False
        
        # Check if either vertex is already used in the matching
        if u in vertices_used or v in vertices_used:
            return False
        
        vertices_used.add(u)
        vertices_used.add(v)
    
        matching.add_edge(u, v, weight=G[u][v].get('weight', 1))
        matching_weight += G[u][v].get('weight', 1)
    
    # Check if the matching is maximal (has maximum possible weight)
    if maximal_weight is None:
        optimal_matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)
        maximal_weight = sum(G[u][v].get('weight', 1) for u, v in optimal_matching)
    
    return matching_weight == maximal_weight


def is_valid_topological_sort(G, node_sequence):
    # Check if all nodes in the sequence are in the graph
    if set(node_sequence) != set(G.nodes()):
        return False
    
    # Create a dictionary to store the position of each node in the sequence
    position = {node: idx for idx, node in enumerate(node_sequence)}
    
    # Check if for every edge (u, v), u comes before v in the sequence
    for u, v in G.edges():
        if position[u] >= position[v]:
            return False
    
    return True


def is_valid_hamiltonian_path(G, path):
    # Check if the path contains all nodes in the graph
    if set(path) != set(G.nodes()):
        return False
    
    # Check if the path contains any duplicate nodes
    if len(path) != len(set(path)):
        counts = {}
        duplicates = []
        
        for node in path:
            counts[node] = counts.get(node, 0) + 1
            if counts[node] > 1:
                duplicates.append(node)
                
        return False
    
    # Check if each consecutive pair of nodes forms an edge in the graph
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i+1]):
            return False
    
    return True


def is_valid_isomorphism(G1, G2, mapping):
    if set(G1.nodes()) != set(mapping.keys()):
        return False
    
    if not all(node in G2.nodes() for node in mapping.values()):
        return False
    
    if len(set(mapping.values())) != len(mapping):
        return False
    
    # Check that adjacency relationships are preserved
    for u in G1.nodes():
        for v in G1.nodes():
            edge_in_g1 = G1.has_edge(u, v)
            edge_in_g2 = G2.has_edge(mapping[u], mapping[v])
            if edge_in_g1 != edge_in_g2:
                return False
    return True


def is_valid_shortest_path_weighted(graph, path, source, target) -> bool:
    if len(path) < 2:
        return False
    try:
        shortest_length = nx.dijkstra_path_length(graph, source, target, weight='weight')
        actual_length = sum(
            graph[path[i]][path[i + 1]].get('weight', 1)
            for i in range(len(path) - 1)
        )
        return abs(actual_length - shortest_length) < 1e-8  # Avoid floating-point error
    except (nx.NetworkXNoPath, KeyError):
        return False


def is_minimum_spanning_tree_weighted(graph, edge_list: list,) -> bool:
    T = nx.Graph()
    T.add_nodes_from(graph.nodes)
    T.add_edges_from(edge_list)

    if not nx.is_connected(T) or T.number_of_edges() != graph.number_of_nodes() - 1:
        return False

    mst = nx.minimum_spanning_tree(graph, weight='weight')
    mst_weight = mst.size(weight='weight')
    edge_weight = sum(
        graph.get_edge_data(u, v).get('weight', 1) for u, v in T.edges
    )
    return abs(edge_weight - mst_weight) < 1e-8


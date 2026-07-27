import networkx as nx
from typing import List, Dict, Any, Optional
from graph_state import GraphSessionManager

def find_stepped_shortest_path(graph_id: str, source: Any, target: Any, max_depth: int = 5) -> Optional[List[Any]]:
    """
    Finds the shortest path between source and target up to max_depth.
    
    Args:
        graph_id: The ID of the graph.
        source: The source node.
        target: The target node.
        max_depth: Maximum search depth.
        
    Returns:
        The shortest path as a list of nodes, or None if not reachable within max_depth.
    """
    G = GraphSessionManager.get_graph(graph_id)
    if source not in G or target not in G:
        return None
        
    try:
        paths = nx.single_source_shortest_path(G, source, cutoff=max_depth)
        if target in paths:
            return paths[target]
        return None
    except nx.NetworkXNoPath:
        return None


def detect_local_cycles(graph_id: str, node: Any, max_depth: int = 4) -> List[List[Any]]:
    """
    Detects local cycles originating from a given node.
    
    Args:
        graph_id: The ID of the graph.
        node: The node ID.
        max_depth: The maximum cycle length to search for.
        
    Returns:
        A list of cycles (each cycle is a list of nodes).
    """
    G = GraphSessionManager.get_graph(graph_id)
    if node not in G:
        return []
        
    cycles = []
    
    if G.is_directed():
        successors = list(G.successors(node))
        for succ in successors:
            if succ == node:
                cycles.append([node, node])
                continue
            paths = nx.all_simple_paths(G, source=succ, target=node, cutoff=max_depth-1)
            for path in paths:
                cycles.append([node] + path)
    else:
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            if neighbor == node:
                cycles.append([node, node])
                continue
            paths = nx.all_simple_paths(G, source=neighbor, target=node, cutoff=max_depth-1)
            for path in paths:
                if len(path) > 2:
                    cycles.append([node] + path)
                    
    return cycles


def find_local_bridge_nodes(graph_id: str, node: Any, radius: int = 3) -> List[Any]:
    """
    Finds bridge nodes (articulation points) in the local ego network.
    
    Args:
        graph_id: The ID of the graph.
        node: The central node ID.
        radius: The radius of the ego network to extract before finding bridges.
        
    Returns:
        A list of articulation point nodes.
    """
    G = GraphSessionManager.get_graph(graph_id)
    if node not in G:
        return []
        
    ego_graph = nx.ego_graph(G, node, radius=radius)
    if ego_graph.is_directed():
        ego_graph = ego_graph.to_undirected()
        
    cut_vertices = list(nx.articulation_points(ego_graph))
    return cut_vertices


def extract_local_k_core(graph_id: str, node: Any, target_k: int = 2, radius: int = 4) -> List[Any]:
    """
    Extracts the k-core nodes of the local ego network.
    
    Args:
        graph_id: The ID of the graph.
        node: The central node ID.
        target_k: The k value for k-core.
        radius: The radius of the ego network.
        
    Returns:
        A list of nodes belonging to the k-core.
    """
    G = GraphSessionManager.get_graph(graph_id)
    if node not in G:
        return []
        
    ego_graph = nx.ego_graph(G, node, radius=radius)
    
    try:
        ego_graph.remove_edges_from(nx.selfloop_edges(ego_graph))
        k_core_subgraph = nx.k_core(ego_graph, k=target_k)
        return list(k_core_subgraph.nodes())
    except nx.NetworkXError:
        return []


def enumerate_local_cliques(graph_id: str, node: Any) -> List[List[Any]]:
    """
    Finds maximal cliques containing the target node in its local neighborhood.
    
    Args:
        graph_id: The ID of the graph.
        node: The target node ID.
        
    Returns:
        A list of cliques (each clique is a list of node IDs).
    """
    G = GraphSessionManager.get_graph(graph_id)
    if node not in G:
        return []
        
    calc_G = G.to_undirected() if G.is_directed() else G
    ego_graph = nx.ego_graph(calc_G, node, radius=1)
    
    cliques = [c for c in nx.find_cliques(ego_graph) if node in c]
    return cliques


def detect_path_convergence(graph_id: str, source: Any, target: Any, hops: int = 2) -> Dict[str, Any]:
    """
    Detects path convergence (nodes where paths from source and target meet within given hops).
    
    Args:
        graph_id: The ID of the graph.
        source: The source node ID.
        target: The target node ID.
        hops: Number of hops to expand from each node.
        
    Returns:
        A dictionary containing convergence nodes and their respective paths.
    """
    G = GraphSessionManager.get_graph(graph_id)
    result = {"convergence_nodes": [], "source_paths": {}, "target_paths": {}}
    
    if source not in G or target not in G:
        return result
        
    if G.is_directed():
        source_paths = nx.single_source_shortest_path(G, source, cutoff=hops)
        G_rev = G.reverse(copy=False)
        target_paths_rev = nx.single_source_shortest_path(G_rev, target, cutoff=hops)
        target_paths = {n: path[::-1] for n, path in target_paths_rev.items()}
    else:
        source_paths = nx.single_source_shortest_path(G, source, cutoff=hops)
        target_paths_rev = nx.single_source_shortest_path(G, target, cutoff=hops)
        target_paths = {n: path[::-1] for n, path in target_paths_rev.items()}
        
    source_reach = set(source_paths.keys())
    target_reach = set(target_paths.keys())
    
    intersection = list(source_reach.intersection(target_reach))
    
    result["convergence_nodes"] = intersection
    result["source_paths"] = {n: source_paths[n] for n in intersection}
    result["target_paths"] = {n: target_paths[n] for n in intersection}
    
    return result

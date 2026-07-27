import networkx as nx
from typing import Dict, Any, Optional
from graph_state import GraphSessionManager

def compute_density(graph_id: str) -> float:
    """
    Computes the density of the specified graph.
    
    Args:
        graph_id: The ID of the graph.
        
    Returns:
        The density value as a float.
    """
    G = GraphSessionManager.get_graph(graph_id)
    return nx.density(G)


def compute_clustering(graph_id: str, node: Optional[Any] = None) -> float:
    """
    Computes the clustering coefficient. 
    If node is provided, computes local clustering. 
    Otherwise, computes the global average clustering of the graph.
    
    Args:
        graph_id: The ID of the graph.
        node: Optional node ID.
        
    Returns:
        The clustering coefficient as a float.
    """
    G = GraphSessionManager.get_graph(graph_id)
    try:
        if node is not None:
            if node not in G:
                return 0.0
            return float(nx.clustering(G, node))
        else:
            return float(nx.average_clustering(G))
    except ZeroDivisionError:
        return 0.0


def check_reachability(graph_id: str, source: Any, target: Any, max_hops: int) -> bool:
    """
    Checks if a target node is reachable from a source node within max_hops.
    
    Args:
        graph_id: The ID of the graph.
        source: The source node ID.
        target: The target node ID.
        max_hops: The maximum allowed path length.
        
    Returns:
        True if reachable within max_hops, False otherwise.
    """
    G = GraphSessionManager.get_graph(graph_id)
    if source not in G or target not in G:
        return False
        
    try:
        length = nx.shortest_path_length(G, source=source, target=target)
        return length <= max_hops
    except nx.NetworkXNoPath:
        return False


def count_connected_components(graph_id: str) -> int:
    """
    Counts the number of connected components in the graph.
    Handles both directed (weakly connected) and undirected graphs.
    
    Args:
        graph_id: The ID of the graph.
        
    Returns:
        The number of connected components.
    """
    G = GraphSessionManager.get_graph(graph_id)
    if G.is_directed():
        return nx.number_weakly_connected_components(G)
    else:
        return nx.number_connected_components(G)


def count_common_neighbors(graph_id: str, node1: Any, node2: Any) -> Dict[str, Any]:
    """
    Counts common neighbors and calculates Jaccard similarity.
    
    Args:
        graph_id: The ID of the graph.
        node1: The first node ID.
        node2: The second node ID.
        
    Returns:
        A dictionary containing the count, Jaccard similarity, and the list of common neighbors.
    """
    G = GraphSessionManager.get_graph(graph_id)
    if node1 not in G or node2 not in G:
        return {"common_neighbors": [], "count": 0, "jaccard_similarity": 0.0}
        
    if G.is_directed():
        neighbors_1 = set(G.successors(node1)) | set(G.predecessors(node1))
        neighbors_2 = set(G.successors(node2)) | set(G.predecessors(node2))
    else:
        neighbors_1 = set(G.neighbors(node1))
        neighbors_2 = set(G.neighbors(node2))
        
    common = list(neighbors_1.intersection(neighbors_2))
    union = neighbors_1.union(neighbors_2)
    
    jaccard = len(common) / len(union) if union else 0.0
    
    return {
        "common_neighbors": common,
        "count": len(common),
        "jaccard_similarity": jaccard
    }

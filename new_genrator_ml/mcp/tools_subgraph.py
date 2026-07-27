import networkx as nx
from typing import List, Any
from graph_state import GraphSessionManager

def extract_ego_graph(graph_id: str, center_node: Any, radius: int) -> str:
    """
    Extracts an ego network (subgraph) centered around a given node within a specific radius.
    Registers the new subgraph in the session manager.
    
    Args:
        graph_id: The ID of the original graph.
        center_node: The central node ID.
        radius: The radius (number of hops) of the ego network.
        
    Returns:
        The graph_id of the newly extracted ego graph.
    """
    G = GraphSessionManager.get_graph(graph_id)
    if center_node not in G:
        raise ValueError(f"Node {center_node} not found in graph {graph_id}.")
        
    ego_subgraph = nx.ego_graph(G, center_node, radius=radius).copy()
    new_graph_id = f"{graph_id}_ego_{center_node}_r{radius}"
    return GraphSessionManager.register_graph(ego_subgraph, new_graph_id)


def get_neighbors_by_hop(graph_id: str, node: Any, hop: int) -> List[Any]:
    """
    Get neighbors at exactly `hop` distance from the given node.
    
    Args:
        graph_id: The ID of the graph.
        node: The node ID.
        hop: The exact hop distance.
        
    Returns:
        A list of node IDs at the specified hop distance.
    """
    G = GraphSessionManager.get_graph(graph_id)
    if node not in G:
        return []
        
    current_nodes = {node}
    visited = {node}
    
    for _ in range(hop):
        next_nodes = set()
        for u in current_nodes:
            neighbors = G.successors(u) if G.is_directed() else G.neighbors(u)
            for v in neighbors:
                if v not in visited:
                    next_nodes.add(v)
                    visited.add(v)
        current_nodes = next_nodes
        
    return list(current_nodes)


def extract_ego_boundary(graph_id: str, node: Any, radius: int) -> List[Any]:
    """
    Extracts the boundary nodes of an ego graph (nodes exactly at the given radius).
    
    Args:
        graph_id: The ID of the graph.
        node: The central node ID.
        radius: The radius of the boundary.
        
    Returns:
        A list of boundary node IDs.
    """
    G = GraphSessionManager.get_graph(graph_id)
    if node not in G:
        return []
        
    ego_full = nx.ego_graph(G, node, radius=radius)
    if radius > 0:
        ego_inner = nx.ego_graph(G, node, radius=radius - 1)
        boundary = set(ego_full.nodes()) - set(ego_inner.nodes())
        return list(boundary)
    else:
        return [node]

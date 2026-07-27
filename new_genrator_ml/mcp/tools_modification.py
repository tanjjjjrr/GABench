from typing import List, Tuple, Any
from graph_state import GraphSessionManager

def remove_edges(graph_id: str, edge_list: List[Tuple[Any, Any]]) -> bool:
    """
    Removes a list of edges from the graph.
    Modifies the graph in-place.
    
    Args:
        graph_id: The ID of the graph to modify.
        edge_list: A list of edge tuples (u, v) to remove.
        
    Returns:
        True if at least one edge was successfully removed, False otherwise.
    """
    G = GraphSessionManager.get_graph(graph_id)
    removed = False
    
    for u, v in edge_list:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            removed = True
            
    return removed


def remove_nodes(graph_id: str, node_list: List[Any]) -> bool:
    """
    Removes a list of nodes from the graph.
    Modifies the graph in-place.
    
    Args:
        graph_id: The ID of the graph to modify.
        node_list: A list of nodes to remove.
        
    Returns:
        True if at least one node was successfully removed, False otherwise.
    """
    G = GraphSessionManager.get_graph(graph_id)
    removed = False
    
    for node in node_list:
        if node in G:
            G.remove_node(node)
            removed = True
            
    return removed


def add_edges(graph_id: str, edge_list: List[Tuple[Any, Any]]) -> bool:
    """
    Adds a list of edges to the graph.
    Modifies the graph in-place.
    
    Args:
        graph_id: The ID of the graph to modify.
        edge_list: A list of edge tuples (u, v) to add.
        
    Returns:
        Always True after execution.
    """
    G = GraphSessionManager.get_graph(graph_id)
    G.add_edges_from(edge_list)
    return True

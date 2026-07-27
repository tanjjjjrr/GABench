import uuid
from typing import Dict, Any, Optional
import networkx as nx

class GraphSessionManager:
    """
    Manages graph instances for MCP tools.
    Allows passing graphs between tools using string-based graph_ids.
    """
    _instance = None
    _graphs: Dict[str, nx.Graph] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GraphSessionManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_graph(cls, graph: nx.Graph, graph_id: Optional[str] = None) -> str:
        """
        Register a graph in the session manager.
        Args:
            graph: The networkx graph object.
            graph_id: Optional explicit graph ID. If None, a UUID will be generated.
        Returns:
            The registered graph_id.
        """
        if graph_id is None:
            graph_id = str(uuid.uuid4())
        cls._graphs[graph_id] = graph
        return graph_id

    @classmethod
    def get_graph(cls, graph_id: str) -> nx.Graph:
        """
        Retrieve a graph by its ID.
        Args:
            graph_id: The ID of the graph.
        Returns:
            The networkx graph object.
        Raises:
            KeyError: If the graph_id is not found.
        """
        if graph_id not in cls._graphs:
            raise KeyError(f"Graph with ID '{graph_id}' not found in session.")
        return cls._graphs[graph_id]

    @classmethod
    def delete_graph(cls, graph_id: str) -> bool:
        """
        Delete a graph from the session to free up memory.
        Args:
            graph_id: The ID of the graph to delete.
        Returns:
            True if deleted, False if not found.
        """
        if graph_id in cls._graphs:
            del cls._graphs[graph_id]
            return True
        return False

    @classmethod
    def clear_all(cls):
        """Clear all graphs from the session."""
        cls._graphs.clear()

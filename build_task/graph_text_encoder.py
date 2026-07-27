import networkx as nx
from typing import Union, List, Optional


class GraphTextEncoder:
    def __init__(self, graph: Union[nx.Graph, nx.DiGraph], weighted=False, raw_texts: Optional[List[str]] = None):
        self.graph = graph
        self.nnodes = len(graph.nodes())
        self.is_directed = isinstance(graph, nx.DiGraph)
        self.weighted = weighted
        self.raw_texts = raw_texts
        # name_dict maps original node IDs to themselves
        # This allows for consistent node naming across the pipeline
        self.name_dict = {node: node for node in graph.nodes()}

    def node_encoder(self) -> str:
        node_list = sorted(self.graph.nodes())
        output = f"containing nodes from {node_list[0]} to {node_list[-1]}."
        return output

    def edge_encoder(self) -> str:
        if not self.graph.edges():
            return " The graph contains no edges.\n"
        output = " The edges are: "
        if self.weighted:
            edges = list(self.graph.edges(data=True))
            weight_key = list(edges[0][2].keys())[0]
            edge_list = [f"({u}, {v}, {w[weight_key]})" for u, v, w in edges]
            output += ", ".join(edge_list) + ". "
            output += f"(u ,v, w) denotes the edge from node *u* to node *v* has a {weight_key} of *w*.\n\n"
        else:
            edge_list = [f"({u}, {v})" for u, v in self.graph.edges()]
            output += ", ".join(edge_list) + ".\n\n"
        return output

    def node_text_encoder(self) -> str:
        """Generate text descriptions for nodes if raw_texts is available."""
        if self.raw_texts is None:
            return ""

        output = " The text descriptions of nodes are: "
        text_descriptions = []
        for node in sorted(self.graph.nodes()):
            if node < len(self.raw_texts):
                text_descriptions.append(f"Node {node}: {self.raw_texts[node]}")
        if not text_descriptions:
            return ""
        output += ", ".join(text_descriptions) + ".\n\n"
        return output

    def generate_graph_description(self, include_texts: bool = False) -> str:
        description = f"Here is {'a directed' if self.is_directed else 'an undirected'} graph "
        description += self.node_encoder()
        if include_texts and self.raw_texts is not None:
            description += self.node_text_encoder()
        description += self.edge_encoder()
        return description


if __name__ == '__main__':
    from utils import load_npz_data, npz_to_nx, get_dataset_paths
    dataset_paths = get_dataset_paths(graph_size='Tiny')
    index = 0
    for dataset_path in dataset_paths:
        edges, directed, weighted = load_npz_data(file_path=dataset_path)
        graph = npz_to_nx(edges, directed, weighted)
        if len(edges) == 0 or len(graph.nodes()) == 0:
            continue
        if not directed:
            continue
        index += 1
        print('==='*20 + '\n')
        print(edges)
        encoder = GraphTextEncoder(graph)
        print(encoder.generate_graph_description())
        print('==='*20 + '\n')

        if index > 5:
            break

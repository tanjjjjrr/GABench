import numpy as np
import networkx as nx


def load_dgraphfin(npz_path, directed=True, to_undirected=False):
    data = np.load(npz_path, allow_pickle=True)
    edge_index = data['edge_index']

    num_nodes = int(data['x'].shape[0])
    src = edge_index[:, 0].astype(np.int64)
    dst = edge_index[:, 1].astype(np.int64)
    edges = list(zip(src.tolist(), dst.tolist()))

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    if to_undirected and directed:
        G = G.to_undirected()

    return G

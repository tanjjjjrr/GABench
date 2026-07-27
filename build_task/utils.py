import os
import pandas as pd
import numpy as np
import networkx as nx


def load_npz_data(file_path):
    npzfile = np.load(file_path)
    edges = npzfile['edges']
    directed = npzfile['directed']
    weighted = npzfile['weighted']
    # npz files don't have raw_texts
    raw_texts = None
    return edges, directed, weighted, raw_texts


def load_pt_data(file_path):
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required to load .pt files. Install with: pip install torch")

    data = torch.load(file_path, map_location='cpu', weights_only=False)
    raw_texts = None

    # Handle torch_geometric Data objects
    try:
        from torch_geometric.data.data import Data as PyGData
        if isinstance(data, PyGData):
            edge_index = data.edge_index
            if isinstance(edge_index, torch.Tensor):
                edge_index = edge_index.numpy()
            # Handle (2, N) format
            if edge_index.shape[0] == 2:
                src = edge_index[0].astype(np.int64)
                dst = edge_index[1].astype(np.int64)
            else:
                src = edge_index[:, 0].astype(np.int64)
                dst = edge_index[:, 1].astype(np.int64)
            edges = list(zip(src.tolist(), dst.tolist()))
            num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else None
            # Handle is_directed - it can be a method (PyG Data), a boolean, or None
            is_directed_attr = getattr(data, 'is_directed', None)
            if callable(is_directed_attr):
                directed = is_directed_attr()
            else:
                directed = is_directed_attr if is_directed_attr is not None else True
            weighted = hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.shape[-1] > 1
            # Extract raw_texts from PyG Data stores
            if hasattr(data, 'stores'):
                for store in data.stores:
                    if 'raw_texts' in store:
                        raw_texts = store['raw_texts']
                        break
            elif hasattr(data, 'raw_texts'):
                raw_texts = data.raw_texts
            return edges, directed, weighted, raw_texts
    except ImportError:
        pass  # torch_geometric not installed, continue with other handlers

    # Handle different PT data formats
    if isinstance(data, dict):
        # Common formats: {'edge_index': ..., 'num_nodes': ...}
        if 'edge_index' in data:
            edge_index = data['edge_index']
            if isinstance(edge_index, torch.Tensor):
                edge_index = edge_index.numpy()
            # Handle (2, N) format vs (N, 2) format
            if edge_index.shape[0] == 2:
                src = edge_index[0].astype(np.int64)
                dst = edge_index[1].astype(np.int64)
            else:
                src = edge_index[:, 0].astype(np.int64)
                dst = edge_index[:, 1].astype(np.int64)
            edges = list(zip(src.tolist(), dst.tolist()))
        elif 'edges' in data:
            edges_data = data['edges']
            if isinstance(edges_data, torch.Tensor):
                edges_data = edges_data.numpy()
            edges = edges_data.tolist() if edges_data.ndim == 2 else edges_data
        else:
            raise ValueError(f"Unsupported PT dict keys: {list(data.keys())}")

        num_nodes = data.get('num_nodes', None)
        directed = data.get('directed', data.get('is_directed', None))
        weighted = data.get('weighted', False)
        # Extract raw_texts from dict if available
        raw_texts = data.get('raw_texts', None)

    elif isinstance(data, tuple) or isinstance(data, list):
        # Format: (edge_index, num_nodes) or [edge_index, num_nodes]
        edge_index = data[0]
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        if edge_index.shape[0] == 2:
            src = edge_index[0].astype(np.int64)
            dst = edge_index[1].astype(np.int64)
        else:
            src = edge_index[:, 0].astype(np.int64)
            dst = edge_index[:, 1].astype(np.int64)
        edges = list(zip(src.tolist(), dst.tolist()))
        num_nodes = data[1] if len(data) > 1 else None
        directed = None
        weighted = False
        raw_texts = None

    else:
        raise ValueError(f"Unsupported PT data type: {type(data)}")

    # Infer directed/weighted if not present
    if directed is None:
        directed = True  # Default to directed

    return edges, directed, weighted, raw_texts


def load_graph_data(file_path):
    """Load graph data from file, supporting both .npz and .pt formats.

    Args:
        file_path: Path to the graph file (.npz or .pt)

    Returns:
        edges: List of (src, dst) or (src, dst, weight) tuples
        directed: Whether the graph is directed
        weighted: Whether the graph is weighted
        raw_texts: List of raw text descriptions for each node (only for .pt files)
    """
    if file_path.endswith('.pt'):
        return load_pt_data(file_path)
    elif file_path.endswith('.npz'):
        return load_npz_data(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Supported: .npz, .pt")


def load_graph_list(file_path):
    """Load a list of graphs from a .pt file.

    Args:
        file_path: Path to the .pt file containing a list of graphs

    Returns:
        List of tuples: [(edges, directed, weighted, raw_texts), ...]
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required to load .pt files. Install with: pip install torch")

    data = torch.load(file_path, map_location='cpu', weights_only=False)

    if isinstance(data, dict) and 'graphs' in data:
        data = data['graphs']

    if not isinstance(data, (list, tuple)):
        raise ValueError(f"Expected a list of graphs, got {type(data)}")

    graph_list = []
    for i, item in enumerate(data):
        edges, directed, weighted, raw_texts = None, None, False, None

        try:
            from torch_geometric.data.data import Data as PyGData
            if isinstance(item, PyGData):
                edge_index = item.edge_index
                if isinstance(edge_index, torch.Tensor):
                    edge_index = edge_index.numpy()
                if edge_index.shape[0] == 2:
                    src = edge_index[0].astype(np.int64)
                    dst = edge_index[1].astype(np.int64)
                else:
                    src = edge_index[:, 0].astype(np.int64)
                    dst = edge_index[:, 1].astype(np.int64)
                edges = list(zip(src.tolist(), dst.tolist()))
                is_directed_attr = getattr(item, 'is_directed', None)
                if callable(is_directed_attr):
                    directed = is_directed_attr()
                else:
                    directed = is_directed_attr if is_directed_attr is not None else True
                weighted = hasattr(item, 'edge_attr') and item.edge_attr is not None and item.edge_attr.shape[-1] > 1
                if hasattr(item, 'stores'):
                    for store in item.stores:
                        if 'raw_texts' in store:
                            raw_texts = store['raw_texts']
                            break
                elif hasattr(item, 'raw_texts'):
                    raw_texts = item.raw_texts
                graph_list.append((edges, directed, weighted, raw_texts))
                continue
        except ImportError:
            pass

        if isinstance(item, dict):
            if 'edge_index' in item:
                edge_index = item['edge_index']
                if isinstance(edge_index, torch.Tensor):
                    edge_index = edge_index.numpy()
                if edge_index.shape[0] == 2:
                    src = edge_index[0].astype(np.int64)
                    dst = edge_index[1].astype(np.int64)
                else:
                    src = edge_index[:, 0].astype(np.int64)
                    dst = edge_index[:, 1].astype(np.int64)
                edges = list(zip(src.tolist(), dst.tolist()))
            elif 'edges' in item:
                edges_data = item['edges']
                if isinstance(edges_data, torch.Tensor):
                    edges_data = edges_data.numpy()
                edges = edges_data.tolist() if edges_data.ndim == 2 else edges_data
            directed = item.get('directed', item.get('is_directed', True))
            weighted = item.get('weighted', False)
            raw_texts = item.get('raw_texts', None)
            graph_list.append((edges, directed, weighted, raw_texts))
            continue

        raise ValueError(f"Unsupported graph item type at index {i}: {type(item)}")

    return graph_list


def npz_to_nx(edges, directed, weighted):
    graph = nx.DiGraph() if directed else nx.Graph()
    if weighted:
        graph.add_weighted_edges_from(edges)
    else:
        if len(edges)!= 0 and len(edges[0]) == 3:
            edges = [(edge[0], edge[1]) for edge in edges]
        graph.add_edges_from(edges)
    return graph

 
def get_dataset_paths(graph_size):
    assert graph_size in ['Tiny', 'Small', 'Medium', 'Large', 'Huge'], "graph_size should be one of ['Tiny', 'Small', 'Medium', 'Large', 'Huge']"
    parent_dir = "/data2/graph/networkrepository/sampled"
    dataset_dir_list = os.listdir(parent_dir)
    dataset_paths = []
    for dataset_dir in dataset_dir_list:
        dataset_dir_path = f"{parent_dir}/{dataset_dir}"
        if not os.path.isdir(dataset_dir_path):
            continue
        dataset_list = os.listdir(dataset_dir_path)
        for dataset in dataset_list:
            if not dataset.startswith(graph_size):
                continue
            dataset_path = f"{dataset_dir_path}/{dataset}"
            if not (dataset.endswith(".npz") or dataset.endswith(".pt")):
                continue
            dataset_paths.append(dataset_path)
    return dataset_paths


def save_datasets(full_data, save_path):
    df = pd.DataFrame(full_data)
    df.to_csv(save_path, index=False)
    # train_table = pa.Table.from_pandas(df)
    # pq.write_table(train_table, parent_path+f'/graph_{sample_num}.parquet')
    print(f"CSV file saved to {save_path} successfully.")


if __name__ == "__main__":
    pass

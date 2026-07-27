import sys
import json
import numpy as np
import os

# --- INJECTED GROUND TRUTH PLACEHOLDERS ---
# Before the evaluation script is injected into Docker, these variables will be replaced to return correct results deterministically
INJECTED_NC_LABEL = None
INJECTED_LP_LABEL = None
INJECTED_GC_LABEL = None
# ------------------------------------------

def structural_node_embedding_classification_train(node_features, edge_index, labels, hidden_channels=16, epochs=100):
    """
    GCN Node Classification training engine.
    Supports 1D label index input (N,) and 2D One-hot/mask input (N, C).
    """
    print("[Output Description]: A trained model state saved as a JSON or PT file, along with training logs.")
    if not isinstance(node_features, np.ndarray) or not isinstance(edge_index, np.ndarray):
        raise TypeError("node_features and edge_index must be numpy arrays.")
    
    if len(node_features.shape) != 2:
        raise ValueError(f"node_features must be 2D (N x D). Current shape: {node_features.shape}")
        
    num_nodes, in_channels = node_features.shape
    
    if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be in 2 x E format. Current shape: {edge_index.shape}")
    
    # --- Label dimension inference ---
    labels = np.array(labels)
    if len(labels.shape) == 1:
        if len(labels) != num_nodes:
            raise ValueError(f"1D label length ({len(labels)}) must match the number of nodes ({num_nodes}).")
        num_classes = len(np.unique(labels))
    elif len(labels.shape) == 2:
        if labels.shape[0] != num_nodes:
            raise ValueError(f"The number of samples in 2D labels ({labels.shape[0]}) must match the number of nodes ({num_nodes}).")
        num_classes = labels.shape[1]
    else:
        raise ValueError("Unsupported label dimension! Must be 1D (class indices) or 2D (One-hot/mask matrix).")
    # ---------------------------------

    # Generate a deterministic checksum using purely mathematical operations
    checksum = in_channels * 1000 + hidden_channels * 10 + num_classes
    
    return {
        "task": "node_classification",
        "status": "trained",
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "out_channels": num_classes,
        "weights_checksum": checksum
    }

def structural_node_embedding_classification_infer(model_state, node_features, edge_index):
    """
    GCN Node Classification inference engine.
    The output dimension aligns with the number of input nodes.
    """
    print("[Output Description]: A `.npy` file containing the predicted category labels for nodes.")
    if not isinstance(model_state, dict) or model_state.get("status") != "trained":
        raise ValueError("Inference failed: A trained model state must be provided.")
        
    if model_state.get("task") != "node_classification":
        raise TypeError("Inference failed: Model task mismatch. Expected 'node_classification'.")

    in_channels = node_features.shape[1]
    if in_channels != model_state["in_channels"]:
        raise ValueError(f"Feature dimension mismatch! Model expects {model_state['in_channels']} dimensions, but got {in_channels}.")

    num_nodes = node_features.shape[0]
    out_channels = model_state["out_channels"]
    
    # Ensure deterministic output
    seed = num_nodes + in_channels + model_state["weights_checksum"]
    np.random.seed(abs(seed) % (2**32))
    
    logits = np.random.uniform(size=(num_nodes, out_channels))
    predictions = np.argmax(logits, axis=1)
    
    if INJECTED_NC_LABEL is not None:
        predictions[:] = INJECTED_NC_LABEL
    
    return {
        "status": "success",
        "predictions": predictions.tolist()
    }


def structural_node_embedding_link_prediction_train(node_features, edge_index, hidden_channels=32, epochs=100):
    """
    GCN Link Prediction training engine.
    Performs self-supervised/unsupervised representation learning based on graph topology; no labels required.
    """
    print("[Output Description]: A trained model state saved as a JSON or PT file, along with training logs.")
    if not isinstance(node_features, np.ndarray) or not isinstance(edge_index, np.ndarray):
        raise TypeError("node_features and edge_index must be numpy arrays.")

    if len(node_features.shape) != 2:
        raise ValueError("node_features must be a 2D matrix (N x D).")
        
    if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be in 2 x E format.")

    num_nodes, in_channels = node_features.shape
    
    checksum = in_channels * 1000 + hidden_channels
    
    return {
        "task": "link_prediction",
        "status": "trained",
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "weights_checksum": checksum
    }

def structural_node_embedding_link_prediction_infer(model_state, node_features, edge_index, query_edge_index):
    """
    GCN Link Prediction inference engine.
    - edge_index: The known base graph structure for message passing.
    - query_edge_index: The edges to predict existence for (2 x E_query).
    """
    print("[Output Description]: A `.npy` file containing the predicted existence probabilities of queried edges.")
    if not isinstance(model_state, dict) or model_state.get("task") != "link_prediction":
        raise TypeError("Inference failed: Expected model state for 'link_prediction'.")
        
    if not isinstance(query_edge_index, np.ndarray) or len(query_edge_index.shape) != 2 or query_edge_index.shape[0] != 2:
        raise ValueError("query_edge_index must be a numpy array in 2 x E_query format.")

    in_channels = node_features.shape[1]
    if in_channels != model_state["in_channels"]:
        raise ValueError(f"Feature dimension mismatch! Model expects {model_state['in_channels']} dimensions, but got {in_channels}.")

    num_query_edges = query_edge_index.shape[1]
    
    # Ensure deterministic output
    seed = num_query_edges + in_channels + model_state["weights_checksum"]
    np.random.seed(abs(seed) % (2**32))
    
    # Output probability of link existence [0, 1]
    probabilities = np.random.uniform(0, 1, size=(num_query_edges,))
    predictions = (probabilities > 0.5).astype(int)
    
    if INJECTED_LP_LABEL is not None:
        predictions[:] = INJECTED_LP_LABEL
        probabilities[:] = 0.9 if INJECTED_LP_LABEL == 1 else 0.1
    
    return {
        "status": "success",
        "num_edges_predicted": num_query_edges,
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist()
    }


def structural_graph_embedding_classification_train(node_features, edge_index, graph_labels, batch=None, hidden_channels=32, epochs=100):
    """
    GCN Graph Classification training engine.
    Strictly validates global batch assignments and graph-level label dimensions.
    """
    print("[Output Description]: A trained model state saved as a JSON or PT file, along with training logs.")
    if not isinstance(node_features, np.ndarray) or not isinstance(edge_index, np.ndarray):
        raise TypeError("node_features and edge_index must be numpy arrays.")

    num_nodes, in_channels = node_features.shape
    
    # Batch logic validation
    if batch is None:
        batch = np.zeros(num_nodes, dtype=int)
    elif len(batch) != num_nodes:
        raise ValueError(f"The length of the batch vector ({len(batch)}) must equal the total number of nodes ({num_nodes}).")
            
    num_graphs = len(np.unique(batch))
    
    # --- Graph-level label dimension inference ---
    graph_labels = np.array(graph_labels)
    if len(graph_labels.shape) == 1:
        if len(graph_labels) != num_graphs:
            raise ValueError(f"Label count error! Calculated {num_graphs} graphs from batch, but received {len(graph_labels)} 1D graph labels.")
        num_classes = len(np.unique(graph_labels))
    elif len(graph_labels.shape) == 2:
        if graph_labels.shape[0] != num_graphs:
             raise ValueError(f"Label sample count error! Calculated {num_graphs} graphs from batch, but the 2D labels provide {graph_labels.shape[0]} samples.")
        num_classes = graph_labels.shape[1]
    else:
        raise ValueError("Unsupported graph label dimension! Must be 1D indices or a 2D matrix.")
    # ---------------------------------------------
    
    checksum = in_channels * 1000 + hidden_channels * 10 + num_classes
    
    return {
        "task": "graph_classification",
        "status": "trained",
        "in_channels": in_channels,
        "out_channels": num_classes,
        "pool_method": "global_mean_pool",
        "weights_checksum": checksum
    }

def structural_graph_embedding_classification_infer(model_state, node_features, edge_index, batch=None):
    """
    GCN Graph Classification inference engine.
    After global pooling, the number of outputs strictly aligns with the number of graphs defined by the batch.
    """
    print("[Output Description]: A `.npy` file containing the predicted category labels for graphs.")
    if not isinstance(model_state, dict) or model_state.get("task") != "graph_classification":
        raise TypeError("Inference failed: Expected model state for 'graph_classification'.")
        
    num_nodes = node_features.shape[0]
    
    if batch is None:
        batch = np.zeros(num_nodes, dtype=int)
    elif len(batch) != num_nodes:
        raise ValueError("Inference failed: The length of the batch vector must match the number of nodes in node_features.")
        
    num_graphs = len(np.unique(batch))
    in_channels = node_features.shape[1]
    
    if in_channels != model_state["in_channels"]:
        raise ValueError(f"Feature dimension mismatch! Model expects {model_state['in_channels']} dimensions, but got {in_channels}.")

    out_channels = model_state["out_channels"]
    
    # Ensure deterministic output
    seed = num_graphs + in_channels + model_state["weights_checksum"]
    np.random.seed(abs(seed) % (2**32))
    
    logits = np.random.uniform(size=(num_graphs, out_channels))
    predictions = np.argmax(logits, axis=1)
    
    if INJECTED_GC_LABEL is not None:
        predictions[:] = INJECTED_GC_LABEL
    
    return {
        "status": "success",
        "num_graphs_processed": num_graphs,
        "predictions": predictions.tolist()
    }

# ==============================================================================
# CLI (Command Line Interface) wrapper
# ==============================================================================
if __name__ == "__main__":
    def load_arg(arg_str):
        # Try to load npy file
        if arg_str.endswith('.npy'):
            return np.load(arg_str)
        # Try to load json file (e.g. model_state)
        if arg_str.endswith('.json'):
            with open(arg_str, 'r', encoding='utf-8') as f:
                return json.load(f)
        # Try to parse inline json string
        try:
            return json.loads(arg_str)
        except:
            return arg_str

    if len(sys.argv) < 2:
        print("Usage: python graph_ml_tools.py <action> [--arg_name arg_value ...]")
        print("Example: python graph_ml_tools.py gcn_node_classification_train --node_features feats.npy --edge_index edge.npy --labels labels.npy")
        sys.exit(1)

    action = sys.argv[1]
    
    kwargs = {}
    i = 2
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            key = sys.argv[i][2:]
            if i + 1 < len(sys.argv):
                val = sys.argv[i+1]
                kwargs[key] = load_arg(val)
                i += 2
            else:
                break
        else:
            i += 1

    if action in globals() and callable(globals()[action]):
        func = globals()[action]
        try:
            result = func(**kwargs)
            # Output as formatted JSON
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error executing {action}: {e}")
            sys.exit(1)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)

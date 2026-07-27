import argparse
import ast
import copy
import json
import math
import re
import sys
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from algorithms import min_vertex_cover
from utils import load_graph_data, npz_to_nx


DEFAULT_MAX_NODES = 500


NODE_ID_FROM_TEXT_PATTERNS = [
    re.compile(
        r"Given the text description (?P<quote>['\"])(?P<text>.*?)(?P=quote), find the node ID",
        re.DOTALL,
    ),
    re.compile(
        r"Given the text description (?P<quote>['\"])(?P<text>.*?)(?P=quote), find the ID of the node",
        re.DOTALL,
    ),
]

QUESTION_PATTERNS = [
    ("edge_existence", re.compile(r"Is there an edge between node (\d+) and node (\d+)")),
    ("degree", re.compile(r"What is the degree of node (\d+)")),
    ("common_neighbor", re.compile(r"What are the common neighbors between node (\d+) and node (\d+)")),
    ("neighbor", re.compile(r"What are the neighbors of node (\d+)")),
    ("node_number", re.compile(r"How many nodes are in this subgraph")),
    ("edge_number", re.compile(r"How many edges are in this subgraph")),
    ("density", re.compile(r"What is the density of this subgraph")),
    ("is_regular", re.compile(r"Is this subgraph regular")),
    ("has_cycle", re.compile(r"Does this subgraph have a cycle")),
    ("is_bipartite", re.compile(r"Is this subgraph bipartite")),
    ("is_connected", re.compile(r"Is this subgraph connected")),
    ("is_weakly_connected", re.compile(r"Is this subgraph weakly connected")),
    ("is_strongly_connected", re.compile(r"Is this subgraph strongly connected")),
    ("transitivity", re.compile(r"What is the transitivity of this subgraph")),
    ("clustering_coefficient", re.compile(r"What is the clustering coefficient of node (\d+)")),
    ("clustering", re.compile(r"What is the clustering of node (\d+)")),
    ("topological_sort", re.compile(r"What is the topological sort of this subgraph")),
    ("diameter", re.compile(r"What is the diameter of this subgraph")),
    ("radius", re.compile(r"What is the radius of this subgraph")),
    ("periphery", re.compile(r"What is the periphery of this subgraph")),
    ("center", re.compile(r"What is the center of this subgraph")),
    ("barycenter", re.compile(r"What is the barycenter of this subgraph")),
    ("wiener_index", re.compile(r"What is the Wiener index of this subgraph")),
    ("global_efficiency", re.compile(r"What is the global efficiency of this subgraph")),
    ("isolate_number", re.compile(r"How many isolated nodes are there in this subgraph")),
    ("degree_assortativity", re.compile(r"What is the degree assortativity of this subgraph")),
    ("degree_connectivity", re.compile(r"What is the degree connectivity of this subgraph(?: for the degree (\d+))?")),
    ("k_core", re.compile(r"What is the k-core of this subgraph")),
    ("strongly_connected_number", re.compile(r"How many strongly connected components are there")),
    ("weakly_connected_number", re.compile(r"How many weakly connected components are there")),
    ("connected_component_number", re.compile(r"How many connected components are there")),
    ("is_tournament", re.compile(r"Is this subgraph a tournament")),
    ("is_eulerian", re.compile(r"Is this subgraph Eulerian")),
    ("minimum_spanning_tree", re.compile(r"What is the minimum spanning tree of this subgraph")),
    ("weighted_minimum_spanning_tree", re.compile(r"What is the minimum spanning tree of this weighted subgraph")),
    ("maximal_cliques", re.compile(r"What are the maximal cliques of this subgraph")),
    ("min_weight_matching", re.compile(r"What is the minimum weight matching of this subgraph")),
    ("max_weight_matching", re.compile(r"What is the maximum weight matching of this subgraph")),
    ("min_edge_covering", re.compile(r"What is the minimum edge covering of this subgraph")),
    ("triangles", re.compile(r"How many triangles include node (\d+)")),
    ("constraint", re.compile(r"What is the constraint of node (\d+)")),
    ("effective_size", re.compile(r"What is the effective size of node (\d+)")),
    ("degree_centrality", re.compile(r"What is the degree centrality of node (\d+)")),
    ("avg_neighbor_degree", re.compile(r"What is the average neighbor degree of node (\d+)")),
    ("closeness_centrality", re.compile(r"What is the closeness centrality of node (\d+)")),
    ("betweenness_centrality", re.compile(r"What is the betweenness centrality of node (\d+)")),
    ("harmonic_centrality", re.compile(r"What is the harmonic centrality of node (\d+)")),
    ("min_vertex_cover", re.compile(r"What is the minimum vertex cover of this subgraph")),
    ("dominating_set", re.compile(r"What is the dominating set of this subgraph")),
    ("maximal_independent_set", re.compile(r"What is the maximal independent set that includes node (\d+)")),
    ("shortest_path", re.compile(r"What is the shortest path between node (\d+) and node (\d+) in this subgraph")),
    ("weighted_shortest_path", re.compile(r"What is the shortest path between node (\d+) and node (\d+) in this weighted subgraph")),
    ("dfs", re.compile(r"What is the depth-first search \(DFS\) traversal order starting from node (\d+)")),
    ("bfs", re.compile(r"What is the breadth-first search \(BFS\) traversal order starting from node (\d+)")),
    ("maximum_flow", re.compile(r"What is the value of the maximum flow from node (\d+) to node (\d+)")),
    ("local_connectivity", re.compile(r"What is the local connectivity between node (\d+) and node (\d+)")),
    ("resource_allocation_index", re.compile(r"What is the resource allocation index between node (\d+) and node (\d+)")),
    ("jaccard_coefficient", re.compile(r"What is the Jaccard coefficient between node (\d+) and node (\d+)")),
    ("adamic_adar_index", re.compile(r"What is the Adamic-Adar index between node (\d+) and node (\d+)")),
    ("bridges", re.compile(r"What are the bridges of this subgraph")),
    ("ego_subgraph_3hop", re.compile(r"Extract the 3-hop ego subgraph centered at node \d+ in the graph")),
    ("node_text_description", re.compile(r"What is the text description of node (\d+)")),
    ("node_id_from_text", re.compile(r"Given the text description ['\"](.+?)['\"], find the node ID", re.DOTALL)),
]


def extract_ego_subgraph(graph, center_node, hops=3, min_nodes=1, max_nodes=DEFAULT_MAX_NODES, required_nodes=None):
    required_nodes = {int(node) for node in (required_nodes or [])}
    required_nodes.add(int(center_node))

    def build_subgraph(current_hops):
        sampled = {center_node}
        queue = deque([(center_node, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= current_hops:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in sampled:
                    sampled.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return graph.subgraph(sampled).copy()

    candidates = []
    for current_hops in range(hops, 0, -1):
        subgraph = build_subgraph(current_hops)
        candidates.append((current_hops, subgraph))
        if (
            min_nodes <= subgraph.number_of_nodes() <= max_nodes
            and required_nodes.issubset(set(subgraph.nodes()))
        ):
            return subgraph

    containing_required = [
        subgraph
        for _, subgraph in candidates
        if min_nodes <= subgraph.number_of_nodes() and required_nodes.issubset(set(subgraph.nodes()))
    ]
    if containing_required:
        return min(containing_required, key=lambda item: item.number_of_nodes())

    sized_candidates = [
        subgraph for _, subgraph in candidates if min_nodes <= subgraph.number_of_nodes() <= max_nodes
    ]
    if sized_candidates:
        return sized_candidates[0]

    return candidates[-1][1]


def graph_to_ground_truth_subgraph(graph):
    return {
        "nodes": sorted(int(node) for node in graph.nodes()),
        "edges": [[int(u), int(v)] for u, v in graph.edges()],
    }


def graph_from_subgraph_data(subgraph_data, directed):
    graph = nx.DiGraph() if directed else nx.Graph()
    if not subgraph_data:
        return graph
    graph.add_nodes_from(int(node) for node in subgraph_data.get("nodes", []))
    graph.add_edges_from((int(u), int(v)) for u, v in subgraph_data.get("edges", []))
    return graph


def data_to_nx(data):
    try:
        import torch
    except ImportError:
        torch = None

    edge_index = getattr(data, "edge_index", None)
    if edge_index is None and isinstance(data, dict):
        edge_index = data.get("edge_index")
    if edge_index is None:
        raise ValueError(f"graph item has no edge_index: {type(data)}")

    if torch is not None and isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    else:
        edge_index = np.asarray(edge_index)

    if edge_index.size == 0:
        edges = []
    elif edge_index.shape[0] == 2:
        edges = list(zip(edge_index[0].astype(np.int64).tolist(), edge_index[1].astype(np.int64).tolist()))
    else:
        edges = list(zip(edge_index[:, 0].astype(np.int64).tolist(), edge_index[:, 1].astype(np.int64).tolist()))

    is_directed_attr = getattr(data, "is_directed", None)
    if callable(is_directed_attr):
        directed = bool(is_directed_attr())
    elif isinstance(data, dict):
        directed = bool(data.get("directed", data.get("is_directed", True)))
    else:
        directed = bool(is_directed_attr) if is_directed_attr is not None else True

    graph = nx.DiGraph() if directed else nx.Graph()
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and isinstance(data, dict):
        num_nodes = data.get("num_nodes")
    if num_nodes is not None:
        graph.add_nodes_from(range(int(num_nodes)))
    graph.add_edges_from((int(u), int(v)) for u, v in edges)
    return graph


def load_graph_context(graph_path):
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and str(graph_path).endswith(".pt"):
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "graphs" in data:
            raw_texts = data.get("texts")
            return {
                "mode": "multi",
                "graphs": data["graphs"],
                "raw_texts": raw_texts,
                "cache": {},
            }

    edges, directed, weighted, raw_texts = load_graph_data(str(graph_path))
    graph = npz_to_nx(edges, directed, weighted)
    return {
        "mode": "single",
        "graph": graph,
        "raw_texts": raw_texts,
    }


def parse_graph_id_from_text(text):
    match = re.search(r"graph_id=(\d+)", str(text))
    if match:
        return int(match.group(1))
    return None


def graph_for_question(context, question=None, row=None):
    if context["mode"] == "single":
        return context["graph"], context.get("raw_texts")

    graph_id = None
    if row is not None and "graph_id" in row.index and not pd.isna(row["graph_id"]):
        graph_id = int(row["graph_id"])
    if graph_id is None and question is not None:
        graph_id = parse_graph_id_from_text(question)
    if graph_id is None:
        raise ValueError("cannot parse graph_id for multi-graph dataset")

    cache = context["cache"]
    if graph_id not in cache:
        graphs = context["graphs"]
        if graph_id < 0 or graph_id >= len(graphs):
            raise ValueError(f"graph_id out of range: {graph_id}")
        cache[graph_id] = data_to_nx(graphs[graph_id])

    raw_texts = context.get("raw_texts")
    if isinstance(raw_texts, (list, tuple)) and graph_id < len(raw_texts):
        raw_texts = raw_texts[graph_id]
    return cache[graph_id], raw_texts


def graph_from_csv_row(row, directed):
    nodes = None
    edges = None
    for node_col, edge_col in (("subgraph_nodes", "subgraph_edges"), ("nodes", "edges")):
        if node_col in row.index and edge_col in row.index:
            nodes = normalize_answer(row[node_col])
            edges = normalize_answer(row[edge_col])
            break
    if nodes is None or edges is None:
        return None
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(int(node) for node in nodes)
    graph.add_edges_from((int(u), int(v)) for u, v in edges)
    return graph


EXPENSIVE_TASKS = {
    "avg_neighbor_degree",
    "betweenness_centrality",
    "closeness_centrality",
    "constraint",
    "effective_size",
    "global_efficiency",
    "harmonic_centrality",
    "min_vertex_cover",
    "dominating_set",
    "maximal_independent_set",
    "min_edge_covering",
    "max_weight_matching",
    "min_weight_matching",
    "maximal_cliques",
    "barycenter",
    "center",
    "diameter",
    "periphery",
    "radius",
    "wiener_index",
}


def parse_question(question):
    center_match = re.search(r"centered at node (\d+)", question)
    if not center_match:
        raise ValueError("cannot parse center node")
    center_node = int(center_match.group(1))

    for pattern in NODE_ID_FROM_TEXT_PATTERNS:
        match = pattern.search(question)
        if match:
            return "node_id_from_text", center_node, [match.group("text")]

    for task, pattern in QUESTION_PATTERNS:
        match = pattern.search(question)
        if match:
            args = []
            for value in match.groups():
                if value is None:
                    args.append(None)
                elif value.isdigit():
                    args.append(int(value))
                else:
                    args.append(value)
            return task, center_node, args
    raise ValueError("cannot infer task from question")


def required_nodes_for_task(center_node, task_args):
    nodes = {int(center_node)}
    for value in task_args:
        if isinstance(value, int):
            nodes.add(value)
    return nodes


def get_undirected_graph_no_selfloops(graph):
    undirected = graph.to_undirected()
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    return undirected


def sorted_edge_list(edges):
    return sorted(tuple(int(value) for value in edge) for edge in edges)


def normalize_answer(answer):
    if isinstance(answer, str):
        stripped = answer.strip()
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return stripped
    return answer


def compute_answer(task, args, subgraph, raw_texts):
    if task == "edge_existence":
        u, v = args
        return "Yes" if subgraph.has_edge(u, v) else "No"
    if task == "degree":
        return int(subgraph.degree(args[0]))
    if task == "common_neighbor":
        u, v = args
        return sorted(int(node) for node in nx.common_neighbors(subgraph.to_undirected(), u, v))
    if task == "neighbor":
        return sorted(int(node) for node in subgraph.neighbors(args[0]))
    if task == "node_number":
        return int(subgraph.number_of_nodes())
    if task == "edge_number":
        return int(subgraph.number_of_edges())
    if task == "density":
        return nx.density(subgraph)
    if task == "has_cycle":
        try:
            nx.find_cycle(subgraph)
            return "Yes"
        except (nx.NetworkXNoCycle, nx.NetworkXUnfeasible):
            return "No"
    if task == "is_regular":
        return "Yes" if nx.is_regular(subgraph) else "No"
    if task == "is_bipartite":
        return "Yes" if nx.is_bipartite(subgraph) else "No"
    if task == "is_connected":
        return "Yes" if nx.is_connected(subgraph.to_undirected()) else "No"
    if task == "is_weakly_connected":
        if subgraph.is_directed():
            return "Yes" if nx.is_weakly_connected(subgraph) else "No"
        return "Yes" if nx.is_connected(subgraph) else "No"
    if task == "is_strongly_connected":
        if subgraph.is_directed():
            return "Yes" if nx.is_strongly_connected(subgraph) else "No"
        return "Yes" if nx.is_connected(subgraph) else "No"
    if task == "is_tournament":
        return "Yes" if subgraph.is_directed() and nx.is_tournament(subgraph) else "No"
    if task == "is_eulerian":
        return "Yes" if nx.is_eulerian(subgraph) else "No"
    if task == "topological_sort":
        return [int(node) for node in nx.topological_sort(subgraph)]
    if task == "transitivity":
        return nx.transitivity(subgraph)
    if task == "diameter":
        return nx.diameter(subgraph.to_undirected())
    if task == "radius":
        return nx.radius(subgraph.to_undirected())
    if task == "periphery":
        return sorted(int(node) for node in nx.periphery(subgraph.to_undirected()))
    if task == "center":
        return sorted(int(node) for node in nx.center(subgraph.to_undirected()))
    if task == "barycenter":
        return sorted(int(node) for node in nx.barycenter(subgraph.to_undirected()))
    if task == "wiener_index":
        return nx.wiener_index(subgraph.to_undirected())
    if task == "global_efficiency":
        return nx.global_efficiency(subgraph.to_undirected())
    if task == "isolate_number":
        return int(nx.number_of_isolates(subgraph))
    if task == "connected_component_number":
        return int(nx.number_connected_components(subgraph.to_undirected()))
    if task == "strongly_connected_number":
        if subgraph.is_directed():
            return int(nx.number_strongly_connected_components(subgraph))
        return int(nx.number_connected_components(subgraph))
    if task == "weakly_connected_number":
        if subgraph.is_directed():
            return int(nx.number_weakly_connected_components(subgraph))
        return int(nx.number_connected_components(subgraph))
    if task == "degree_assortativity":
        return nx.degree_assortativity_coefficient(subgraph)
    if task == "degree_connectivity":
        degree = args[0]
        values = nx.degree_connectivity(subgraph)
        if degree is not None:
            return {int(degree): values.get(degree, {})}
        return {int(k): v for k, v in values.items()}
    if task == "min_edge_covering":
        return sorted_edge_list(nx.min_edge_cover(get_undirected_graph_no_selfloops(subgraph)))
    if task == "bridges":
        return sorted_edge_list(nx.bridges(subgraph.to_undirected()))
    if task == "minimum_spanning_tree":
        return sorted_edge_list(nx.minimum_spanning_edges(subgraph.to_undirected(), data=False))
    if task == "weighted_minimum_spanning_tree":
        return sorted_edge_list(nx.minimum_spanning_edges(subgraph.to_undirected(), data=False))
    if task == "dominating_set":
        return sorted(int(node) for node in nx.dominating_set(subgraph))
    if task == "min_vertex_cover":
        return sorted(int(node) for node in min_vertex_cover(subgraph.to_undirected()))
    if task == "max_weight_matching":
        return sorted_edge_list(nx.max_weight_matching(get_undirected_graph_no_selfloops(subgraph)))
    if task == "min_weight_matching":
        return sorted_edge_list(nx.max_weight_matching(get_undirected_graph_no_selfloops(subgraph)))
    if task == "maximal_cliques":
        return sorted(sorted(int(node) for node in clique) for clique in nx.find_cliques(subgraph.to_undirected()))
    if task == "k_core":
        return sorted(int(node) for node in nx.k_core(get_undirected_graph_no_selfloops(subgraph)).nodes())
    if task == "triangles":
        return int(nx.triangles(get_undirected_graph_no_selfloops(subgraph), args[0]))
    if task == "clustering_coefficient" or task == "clustering":
        return nx.clustering(subgraph.to_undirected(), args[0])
    if task == "constraint":
        return nx.constraint(subgraph, [args[0]])[args[0]]
    if task == "effective_size":
        return nx.effective_size(subgraph, [args[0]])[args[0]]
    if task == "degree_centrality":
        return nx.degree_centrality(subgraph)[args[0]]
    if task == "closeness_centrality":
        return nx.closeness_centrality(subgraph)[args[0]]
    if task == "betweenness_centrality":
        return nx.betweenness_centrality(subgraph)[args[0]]
    if task == "harmonic_centrality":
        return nx.harmonic_centrality(subgraph)[args[0]]
    if task == "avg_neighbor_degree":
        return nx.average_neighbor_degree(subgraph)[args[0]]
    if task == "maximal_independent_set":
        return sorted(int(node) for node in nx.maximal_independent_set(get_undirected_graph_no_selfloops(subgraph)))
    if task == "local_connectivity":
        u, v = args
        return "Yes" if nx.has_path(subgraph, u, v) else "No"
    if task == "shortest_path" or task == "weighted_shortest_path":
        u, v = args
        try:
            return [int(node) for node in nx.shortest_path(subgraph, u, v)]
        except nx.NetworkXNoPath:
            return []
    if task == "bfs":
        return sorted_edge_list(nx.bfs_edges(subgraph, source=args[0], sort_neighbors=sorted))
    if task == "dfs":
        return sorted_edge_list(nx.dfs_edges(subgraph, source=args[0]))
    if task == "maximum_flow":
        source, target = args
        flow_graph = subgraph.copy()
        for u, v in flow_graph.edges():
            flow_graph[u][v].setdefault("capacity", 1)
        flow_value, _ = nx.maximum_flow(flow_graph, source, target)
        return flow_value
    if task == "resource_allocation_index":
        u, v = args
        return list(nx.resource_allocation_index(subgraph.to_undirected(), [(u, v)]))[0][2]
    if task == "jaccard_coefficient":
        u, v = args
        return list(nx.jaccard_coefficient(subgraph.to_undirected(), [(u, v)]))[0][2]
    if task == "adamic_adar_index":
        u, v = args
        return list(nx.adamic_adar_index(subgraph.to_undirected(), [(u, v)]))[0][2]
    if task == "ego_subgraph_3hop":
        return sorted_edge_list(subgraph.edges())
    if task == "node_text_description":
        node = args[0]
        if raw_texts is None or node >= len(raw_texts):
            raise ValueError("raw_texts missing or node out of range")
        return str(raw_texts[node])
    if task == "node_id_from_text":
        target_text = args[0]
        if raw_texts is None:
            raise ValueError("raw_texts missing")
        for idx, text in enumerate(raw_texts):
            if str(text) == target_text:
                return int(idx)
        raise ValueError("text not found")
    raise ValueError(f"unsupported task: {task}")


def answer_equal(left, right):
    left = normalize_answer(left)
    right = normalize_answer(right)
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return left == right


def rerun_entry(entry, context):
    task, center_node, task_args = parse_question(entry["question"])
    graph, raw_texts = graph_for_question(context, entry["question"])
    subgraph = extract_ego_subgraph(
        graph,
        center_node,
        max_nodes=DEFAULT_MAX_NODES,
        required_nodes=required_nodes_for_task(center_node, task_args),
    )
    fallback_subgraph = graph_from_subgraph_data(
        entry.get("ground_truth", {}).get("subgraph"),
        graph.is_directed(),
    )
    used_fallback = False
    try:
        new_answer = compute_answer(task, task_args, subgraph, raw_texts)
    except Exception:
        if fallback_subgraph.number_of_nodes() == 0:
            raise
        subgraph = fallback_subgraph
        new_answer = compute_answer(task, task_args, subgraph, raw_texts)
        used_fallback = True

    updated = copy.deepcopy(entry)
    updated.setdefault("ground_truth", {})
    updated["ground_truth"]["subgraph"] = graph_to_ground_truth_subgraph(subgraph)
    updated["ground_truth"]["answer"] = new_answer
    return updated, task, used_fallback


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def rerun_json_file(path, output_path, context):
    entries = load_json(path)
    updated_entries = []
    changed_answers = 0
    changed_subgraphs = 0
    failed = []
    fallback_count = 0
    tasks = {}

    for idx, entry in enumerate(tqdm(entries, desc=str(path), leave=False)):
        try:
            updated, task, used_fallback = rerun_entry(entry, context)
            tasks[task] = tasks.get(task, 0) + 1
            if used_fallback:
                fallback_count += 1
        except Exception as exc:
            failed.append((idx, str(exc)))
            updated_entries.append(entry)
            continue

        old_gt = entry.get("ground_truth", {})
        if not answer_equal(old_gt.get("answer"), updated["ground_truth"].get("answer")):
            changed_answers += 1
        if old_gt.get("subgraph") != updated["ground_truth"].get("subgraph"):
            changed_subgraphs += 1
        updated_entries.append(updated)

    save_json(output_path, updated_entries)
    return {
        "file": str(path),
        "output": str(output_path),
        "entries": len(entries),
        "changed_answers": changed_answers,
        "changed_subgraphs": changed_subgraphs,
        "fallbacks": fallback_count,
        "failed": failed,
        "tasks": tasks,
    }


def categories_from_dataset_json(data):
    if not isinstance(data, dict):
        return None
    categories = []
    for category, subcats in data.items():
        if isinstance(subcats, dict):
            for subcat, entries in subcats.items():
                if isinstance(entries, list):
                    categories.append((category, subcat, entries))
    return categories


def rerun_dataset_json(path, output_path, context):
    data = load_json(path)
    categories = categories_from_dataset_json(data)
    if not categories:
        raise ValueError(f"{path} does not look like a dataset aggregate JSON")

    updated_data = copy.deepcopy(data)
    changed_answers = 0
    changed_subgraphs = 0
    failed = []
    fallback_count = 0
    tasks = {}
    total = 0

    for category, subcat, entries in categories:
        new_entries = []
        for idx, entry in enumerate(tqdm(entries, desc=f"{category}/{subcat}", leave=False)):
            total += 1
            try:
                updated, task, used_fallback = rerun_entry(entry, context)
                tasks[task] = tasks.get(task, 0) + 1
                if used_fallback:
                    fallback_count += 1
            except Exception as exc:
                failed.append((category, subcat, idx, str(exc)))
                new_entries.append(entry)
                continue

            old_gt = entry.get("ground_truth", {})
            if not answer_equal(old_gt.get("answer"), updated["ground_truth"].get("answer")):
                changed_answers += 1
            if old_gt.get("subgraph") != updated["ground_truth"].get("subgraph"):
                changed_subgraphs += 1
            new_entries.append(updated)
        updated_data[category][subcat] = new_entries

    save_json(output_path, updated_data)
    return {
        "file": str(path),
        "output": str(output_path),
        "entries": total,
        "changed_answers": changed_answers,
        "changed_subgraphs": changed_subgraphs,
        "fallbacks": fallback_count,
        "failed": failed,
        "tasks": tasks,
    }


def serialize_csv_value(value):
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str)):
        return value
    return str(value)


def rerun_csv_file(path, output_path, context):
    df = pd.read_csv(path)
    if df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return {
            "file": str(path),
            "output": str(output_path),
            "entries": 0,
            "changed_answers": 0,
            "changed_subgraphs": 0,
            "fallbacks": 0,
            "failed": [],
            "tasks": {},
        }

    if "prompt" not in df.columns:
        raise ValueError(f"{path} has no prompt column")

    changed_answers = 0
    changed_subgraphs = 0
    failed = []
    fallback_count = 0
    tasks = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=str(path), leave=False):
        try:
            task, center_node, task_args = parse_question(str(row["prompt"]))
            graph, raw_texts = graph_for_question(context, row["prompt"], row)
            subgraph = extract_ego_subgraph(
                graph,
                center_node,
                max_nodes=DEFAULT_MAX_NODES,
                required_nodes=required_nodes_for_task(center_node, task_args),
            )
            fallback_subgraph = graph_from_csv_row(row, graph.is_directed())
            used_fallback = False
            try:
                new_answer = compute_answer(task, task_args, subgraph, raw_texts)
            except Exception:
                if fallback_subgraph is None or fallback_subgraph.number_of_nodes() == 0:
                    raise
                subgraph = fallback_subgraph
                new_answer = compute_answer(task, task_args, subgraph, raw_texts)
                used_fallback = True
            if used_fallback:
                fallback_count += 1
            subgraph_data = graph_to_ground_truth_subgraph(subgraph)
            tasks[task] = tasks.get(task, 0) + 1
        except Exception as exc:
            failed.append((int(idx), str(exc)))
            continue

        if "answer" in df.columns and not answer_equal(row["answer"], new_answer):
            changed_answers += 1
        if "subgraph_nodes" in df.columns and "subgraph_edges" in df.columns:
            old_nodes = normalize_answer(row["subgraph_nodes"])
            old_edges = normalize_answer(row["subgraph_edges"])
            if old_nodes != subgraph_data["nodes"] or old_edges != subgraph_data["edges"]:
                changed_subgraphs += 1
        elif "nodes" in df.columns and "edges" in df.columns:
            old_nodes = normalize_answer(row["nodes"])
            old_edges = normalize_answer(row["edges"])
            if old_nodes != subgraph_data["nodes"] or old_edges != subgraph_data["edges"]:
                changed_subgraphs += 1

        if "task" in df.columns:
            df.at[idx, "task"] = task
        if "answer" in df.columns:
            df.at[idx, "answer"] = serialize_csv_value(new_answer)
        if "direction" in df.columns:
            df.at[idx, "direction"] = "directed" if subgraph.is_directed() else "undirected"
        if "nodes" in df.columns:
            df.at[idx, "nodes"] = str(subgraph_data["nodes"])
        if "edges" in df.columns:
            df.at[idx, "edges"] = str([tuple(edge) for edge in subgraph_data["edges"]])
        if "subgraph_nodes" in df.columns:
            df.at[idx, "subgraph_nodes"] = str(subgraph_data["nodes"])
        if "subgraph_edges" in df.columns:
            df.at[idx, "subgraph_edges"] = str([tuple(edge) for edge in subgraph_data["edges"]])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return {
        "file": str(path),
        "output": str(output_path),
        "entries": len(df),
        "changed_answers": changed_answers,
        "changed_subgraphs": changed_subgraphs,
        "fallbacks": fallback_count,
        "failed": failed,
        "tasks": tasks,
    }


def json_paths_for_dataset(dataset_dir, dataset_name):
    paths = []
    for category in ("graph_info_retrieval", "graph_theory"):
        category_dir = dataset_dir / category
        for subcat in ("node", "edge", "graph"):
            path = category_dir / f"{subcat}.json"
            if path.exists():
                paths.append(path)
    aggregate = dataset_dir / f"{dataset_name}.json"
    if aggregate.exists():
        paths.append(aggregate)
    return paths


def csv_paths_for_dataset(dataset_dir):
    paths = []
    for category in ("graph_info_retrieval", "graph_theory"):
        category_dir = dataset_dir / category
        for subcat in ("node", "edge", "graph"):
            subcat_dir = category_dir / subcat
            if subcat_dir.exists():
                paths.extend(sorted(subcat_dir.glob("*.csv")))
    return paths


def write_report(path, summaries):
    lines = []
    for summary in summaries:
        lines.append(
            f"{summary['file']} -> {summary['output']}: "
            f"{summary['entries']} entries, "
            f"{summary['changed_answers']} changed answers, "
            f"{summary['changed_subgraphs']} changed subgraphs, "
            f"{summary.get('fallbacks', 0)} fallbacks, "
            f"{len(summary['failed'])} failed"
        )
        for failure in summary["failed"][:20]:
            lines.append(f"  FAILED {failure}")
        if len(summary["failed"]) > 20:
            lines.append(f"  ... {len(summary['failed']) - 20} more failures")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Re-extract ego subgraphs and recompute answers for existing JSON questions."
    )
    parser.add_argument("--dataset", default="cora", help="Dataset directory/name, e.g. cora")
    parser.add_argument("--graph", default=None, help="Original graph file. Defaults to <dataset>.pt")
    parser.add_argument("--input-dir", default=None, help="Directory containing graph_info_retrieval/graph_theory")
    parser.add_argument("--output-dir", default=None, help="Where to write updated JSON. Defaults to <dataset>_rerun")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the existing JSON files")
    parser.add_argument("--json-only", action="store_true", help="Only recompute JSON files")
    parser.add_argument("--csv-only", action="store_true", help="Only recompute per-task CSV files")
    parser.add_argument("--report", default=None, help="Report path. Defaults under the output directory")
    args = parser.parse_args()

    dataset_dir = Path(args.input_dir or args.dataset)
    dataset_name = dataset_dir.name
    graph_path = Path(args.graph or f"{dataset_name}.pt")
    output_dir = dataset_dir if args.in_place else Path(args.output_dir or f"{dataset_name}_rerun")

    context = load_graph_context(graph_path)
    if context["mode"] == "single":
        graph = context["graph"]
        print(f"Loaded {graph_path}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    else:
        print(f"Loaded {graph_path}: {len(context['graphs'])} graphs")

    summaries = []
    if not args.csv_only:
        for path in json_paths_for_dataset(dataset_dir, dataset_name):
            rel_path = path.relative_to(dataset_dir)
            output_path = output_dir / rel_path
            if path.name == f"{dataset_name}.json":
                summary = rerun_dataset_json(path, output_path, context)
            else:
                summary = rerun_json_file(path, output_path, context)
            summaries.append(summary)
            print(
                f"{path}: {summary['entries']} entries, "
                f"{summary['changed_answers']} changed answers, "
                f"{summary['changed_subgraphs']} changed subgraphs, "
                f"{summary.get('fallbacks', 0)} fallbacks, "
                f"{len(summary['failed'])} failed"
            )

    if not args.json_only:
        for path in csv_paths_for_dataset(dataset_dir):
            rel_path = path.relative_to(dataset_dir)
            output_path = output_dir / rel_path
            summary = rerun_csv_file(path, output_path, context)
            summaries.append(summary)
            print(
                f"{path}: {summary['entries']} entries, "
                f"{summary['changed_answers']} changed answers, "
                f"{summary['changed_subgraphs']} changed subgraphs, "
                f"{summary.get('fallbacks', 0)} fallbacks, "
                f"{len(summary['failed'])} failed"
            )

    report_path = Path(args.report) if args.report else output_dir / "rerun_report.txt"
    write_report(report_path, summaries)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()

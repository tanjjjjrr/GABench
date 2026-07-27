import os
from collections import deque
import networkx as nx
import random
import argparse
import math
import numpy as np
from tqdm import tqdm
 
from graph_text_encoder import GraphTextEncoder
from utils import load_graph_data, load_graph_list, npz_to_nx, get_dataset_paths, save_datasets
from algorithms import min_vertex_cover
from check_correctness import *

from instructions import INSTRUCTIONS
from questions import QUESTIONS
from formats import FORMATS
from answer_types import ANSWER_TYPES



# =============================================================================
# Task list definition
# =============================================================================
# Note:
#   - edge_existence, degree, common_neighbor, and neighbor do not require subgraph extraction.
#   - All other tasks require extracting a 3-hop completely induced subgraph centered on node{}.
# =============================================================================

FULL_TASKS = [
    # Tasks that do not require subgraph extraction
    'edge_existence', 'degree', 'common_neighbor', 'neighbor',

    # Tasks that require subgraph extraction
    'node_number', 'edge_number', 'shortest_path', 'bfs', 'dfs',
    'triangles', 'density', 'topological_sort', 'has_cycle',
    'closeness_centrality', 'betweenness_centrality', 'harmonic_centrality',
    'wiener_index', 'barycenter', 'center', 'diameter', 'periphery', 'radius',
    'global_efficiency', 'transitivity', 'clustering_coefficient',
    'constraint', 'effective_size', 'isolate_number', 'is_regular',
    'degree_centrality', 'degree_assortativity', 'degree_connectivity',
    'avg_neighbor_degree', 'local_connectivity',
    'min_edge_covering', 'dominating_set', 'maximal_independent_set',
    'min_vertex_cover', 'max_weight_matching', 'min_weight_matching',
    'connected_component_number', 'strongly_connected_number',
    'weakly_connected_number', 'is_bipartite', 'is_eulerian',
    'is_connected', 'is_weakly_connected', 'is_strongly_connected',
    'is_tournament', 'bridges', 'minimum_spanning_tree',
    'weighted_minimum_spanning_tree',
    'resource_allocation_index', 'jaccard_coefficient', 'adamic_adar_index',
    'maximum_flow', 'k_core', 'maximal_cliques',
    'weighted_shortest_path', 'ego_subgraph_3hop',

    # Graph information retrieval: node text tasks
    'node_text_description', 'node_id_from_text',
]

# =============================================================================
# Categorize by level: node / edge / graph
# =============================================================================
TASK_LEVELS = {
    'node': {
        'degree', 'triangles', 'closeness_centrality', 'betweenness_centrality',
        'harmonic_centrality', 'degree_centrality', 'avg_neighbor_degree',
        'is_regular', 'local_connectivity', 'constraint', 'effective_size',
        'isolate_number', 'k_core', 'maximal_cliques',
        'neighbor', 'common_neighbor',
        # Graph information retrieval: node-related tasks
        'node_text_description',
    },
    'edge': {
        'edge_existence', 'shortest_path', 'bfs', 'dfs',
        'weighted_shortest_path', 'maximum_flow',
        'resource_allocation_index', 'jaccard_coefficient', 'adamic_adar_index',
        'bridges',
    },
    'graph': {
        'node_number', 'edge_number', 'node_id_from_text',
        'ego_subgraph_3hop',
        'density', 'topological_sort', 'has_cycle',
        'is_connected', 'is_weakly_connected', 'is_strongly_connected',
        'is_bipartite', 'is_eulerian', 'is_tournament',
        'wiener_index', 'barycenter', 'center', 'diameter', 'periphery', 'radius',
        'global_efficiency', 'transitivity', 'clustering_coefficient',
        'degree_assortativity', 'degree_connectivity',
        'connected_component_number', 'strongly_connected_number',
        'weakly_connected_number', 'min_edge_covering', 'dominating_set',
        'maximal_independent_set', 'min_vertex_cover',
        'max_weight_matching', 'min_weight_matching',
        'minimum_spanning_tree', 'weighted_minimum_spanning_tree',
    },
}

def get_task_level(task_name):
    """Return the task level (node/edge/graph)."""
    for level, tasks in TASK_LEVELS.items():
        if task_name in tasks:
            return level
    return 'graph'  # Default to graph-level.

# =============================================================================
# Task toolset definitions
# =============================================================================
# All tasks require extracting a 3-hop completely induced subgraph centered on node{}.
# =============================================================================

TOOLSETS = {
    # -------------------------------------------------------------------------
    # Tasks that do not require subgraph extraction
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Tasks that require extracting a 3-hop ego subgraph centered on node{}
    # -------------------------------------------------------------------------
    'edge_existence': ['extract_subgraph', 'graph.has_edge'],
    'degree': ['extract_subgraph', 'graph.degree'],
    'neighbor': ['extract_subgraph', 'graph.neighbors'],
    'common_neighbor': ['extract_subgraph', 'nx.common_neighbors'],
    'node_number': ['extract_subgraph', 'len(graph.nodes())'],
    'edge_number': ['extract_subgraph', 'len(graph.edges())'],
    'shortest_path': ['extract_subgraph', 'nx.shortest_path'],
    'bfs': ['extract_subgraph', 'nx.bfs_edges'],
    'dfs': ['extract_subgraph', 'nx.dfs_edges'],
    'triangles': ['extract_subgraph', 'nx.triangles'],
    'density': ['extract_subgraph', 'nx.density'],
    'topological_sort': ['extract_subgraph', 'nx.topological_sort'],
    'has_cycle': ['extract_subgraph', 'nx.find_cycle'],
    'closeness_centrality': ['extract_subgraph', 'nx.closeness_centrality'],
    'betweenness_centrality': ['extract_subgraph', 'nx.betweenness_centrality'],
    'harmonic_centrality': ['extract_subgraph', 'nx.harmonic_centrality'],
    'degree_centrality': ['extract_subgraph', 'nx.degree_centrality'],
    'avg_neighbor_degree': ['extract_subgraph', 'nx.average_neighbor_degree'],
    'local_connectivity': ['extract_subgraph', 'nx.has_path'],
    'isolate_number': ['extract_subgraph', 'nx.number_of_isolates'],
    'wiener_index': ['extract_subgraph', 'nx.wiener_index'],
    'barycenter': ['extract_subgraph', 'nx.barycenter'],
    'center': ['extract_subgraph', 'nx.center'],
    'diameter': ['extract_subgraph', 'nx.diameter'],
    'periphery': ['extract_subgraph', 'nx.periphery'],
    'radius': ['extract_subgraph', 'nx.radius'],
    'global_efficiency': ['extract_subgraph', 'nx.global_efficiency'],
    'transitivity': ['extract_subgraph', 'nx.transitivity'],
    'clustering_coefficient': ['extract_subgraph', 'nx.clustering'],
    'dominating_set': ['extract_subgraph', 'nx.dominating_set'],
    'maximal_independent_set': ['extract_subgraph', 'nx.maximal_independent_set'],
    'min_edge_covering': ['extract_subgraph', 'nx.min_edge_cover'],
    'min_vertex_cover': ['extract_subgraph', 'min_vertex_cover'],
    'max_weight_matching': ['extract_subgraph', 'nx.max_weight_matching'],
    'connected_component_number': ['extract_subgraph', 'nx.number_connected_components'],
    'strongly_connected_number': ['extract_subgraph', 'nx.number_strongly_connected_components'],
    'weakly_connected_number': ['extract_subgraph', 'nx.number_weakly_connected_components'],
    'is_tournament': ['extract_subgraph', 'nx.is_tournament'],
    'is_bipartite': ['extract_subgraph', 'nx.is_bipartite'],
    'is_eulerian': ['extract_subgraph', 'nx.is_eulerian'],
    'is_regular': ['extract_subgraph', 'nx.is_regular'],
    'is_connected': ['extract_subgraph', 'nx.is_connected'],
    'is_weakly_connected': ['extract_subgraph', 'nx.is_weakly_connected'],
    'is_strongly_connected': ['extract_subgraph', 'nx.is_strongly_connected'],
    'bridges': ['extract_subgraph', 'nx.bridges'],
    'minimum_spanning_tree': ['extract_subgraph', 'nx.minimum_spanning_edges'],
    'maximum_flow': ['extract_subgraph', 'nx.maximum_flow'],
    'resource_allocation_index': ['extract_subgraph', 'nx.resource_allocation_index'],
    'jaccard_coefficient': ['extract_subgraph', 'nx.jaccard_coefficient'],
    'adamic_adar_index': ['extract_subgraph', 'nx.adamic_adar_index'],
    'constraint': ['extract_subgraph', 'nx.constraint'],
    'effective_size': ['extract_subgraph', 'nx.effective_size'],
    'degree_assortativity': ['extract_subgraph', 'nx.degree_assortativity_coefficient'],
    'degree_connectivity': ['extract_subgraph', 'nx.degree_connectivity'],
    'k_core': ['extract_subgraph', 'nx.k_core'],
    'maximal_cliques': ['extract_subgraph', 'nx.find_cliques'],
    'weighted_shortest_path': ['extract_subgraph', 'nx.shortest_path'],
    'min_weight_matching': ['extract_subgraph', 'nx.min_weight_matching'],
    'weighted_minimum_spanning_tree': ['extract_subgraph', 'nx.minimum_spanning_edges'],
    'ego_subgraph_3hop': ['extract_subgraph'],

    # Graph information retrieval: node text tasks
    'node_text_description': [],
    'node_id_from_text': [],
}

# Tasks that should run on undirected graphs
UNDIRECTED_TASKS = [
    'common_neighbor', 'triangles', 'global_efficiency', 'resource_allocation_index',
    'jaccard_coefficient', 'adamic_adar_index', 'min_edge_covering',
    'maximal_independent_set', 'max_weight_matching', 'min_weight_matching',
    'connected_component_number', 'weakly_connected_number',
    'bridges', 'minimum_spanning_tree', 'weighted_minimum_spanning_tree',
    'is_bipartite', 'is_regular', 'clustering_coefficient',
    'constraint', 'effective_size',
]

# Tasks that should run on weighted graphs
TRANSFER_TASKS = [
    'constraint', 'effective_size', 'degree_assortativity'
]

# Binary classification tasks
BINARY_TASKS = [
    'edge_existence', 'has_cycle', 'is_regular', 'is_bipartite',
    'is_eulerian', 'is_tournament', 'local_connectivity', 'is_connected',
    'is_weakly_connected', 'is_strongly_connected',
]

# Tasks with non-unique answers
NOT_UNIQUE_TASKS = [
    'bfs', 'dfs', 'dominating_set', 'maximal_independent_set', 'min_edge_covering',
    'min_vertex_cover', 'minimum_spanning_tree', 'topological_sort', 'shortest_path',
    'max_weight_matching',
    'maximal_cliques',
]

# Node-centered questions
NODE_TASKS = [
    'degree', 'neighbor', 'common_neighbor', 'triangles',
    'clustering_coefficient', 'constraint', 'effective_size', 'isolate_number',
    'betweenness_centrality', 'closeness_centrality', 'harmonic_centrality',
    'degree_centrality', 'avg_neighbor_degree', 'maximal_independent_set',
    'local_connectivity',
    'node_text_description',
]

# Edge-centered questions
EDGE_TASKS = [
    'edge_existence', 'shortest_path', 'weighted_shortest_path',
    'bfs', 'dfs', 'maximum_flow',
    'min_edge_covering', 'max_weight_matching', 'min_weight_matching',
    'minimum_spanning_tree', 'bridges',
    'resource_allocation_index', 'jaccard_coefficient', 'adamic_adar_index',
]

# Graph-level questions
GRAPH_TASKS = [
    'node_number', 'edge_number', 'density',
    'diameter', 'radius', 'wiener_index',
    'global_efficiency', 'transitivity', 'avg_neighbor_degree',
    'has_cycle', 'is_bipartite', 'is_regular', 'is_connected',
    'is_weakly_connected', 'is_strongly_connected', 'is_tournament',
    'is_eulerian', 'topological_sort',
    'connected_component_number', 'strongly_connected_number', 'weakly_connected_number',
    'barycenter', 'center', 'periphery', 'dominating_set',
    'min_vertex_cover', 'bridges', 'minimum_spanning_tree', 'weighted_minimum_spanning_tree',
    'maximal_cliques',
    'degree_assortativity', 'degree_connectivity',
    'k_core', 'constraint', 'effective_size',
    'ego_subgraph_3hop',
    'node_id_from_text',
]


def connectivity_checking(graph, directed):
    if not directed:
        return nx.is_connected(graph)
    return nx.is_strongly_connected(graph)


def get_undirected_graph_no_selfloops(graph):
    """Convert to undirected graph and remove self-loops for algorithms that don't support them."""
    G_und = graph.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))
    return G_und


def extract_ego_subgraph(graph, center_node=None, min_nodes=5, max_nodes=500):
    """
    Extract a 3-hop completely induced subgraph centered on a node.

    Args:
        graph: Source graph
        center_node: Center node; choose a random node when None
        min_nodes: Minimum number of nodes in the subgraph
        max_nodes: Maximum number of nodes before resampling the center node

    Returns:
        H: Subgraph
        center: Center node
        actual_hops: Number of hops used, fixed at 3
    """
    max_attempts = 100
    for _ in range(max_attempts):
        if center_node is None:
            center_node = random.choice(list(graph.nodes()))

        sampled = {center_node}
        queue = deque([(center_node, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= 3:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in sampled:
                    sampled.add(neighbor)
                    queue.append((neighbor, depth + 1))

        H = graph.subgraph(sampled).copy()

        num_nodes = H.number_of_nodes()
        if min_nodes <= num_nodes <= max_nodes:
            return H, center_node, 3

        center_node = random.choice(list(graph.nodes()))

    return H, center_node, 3


def sample_subgraph(G, target_size='Tiny'):
    if target_size == 'Tiny':
        n_target = random.randint(5, 36)
    elif target_size == 'Small':
        n_target = random.randint(36, 101)
    elif target_size == 'Medium':
        n_target = random.randint(100, 501)
    else:
        n_target = random.randint(5, 36)

    # Resample the center node when the 3-hop subgraph is too large.
    max_nodes = 500

    nodes = list(G.nodes())
    max_attempts = 500

    for _ in range(max_attempts):
        center = random.choice(nodes)
        sampled = {center}
        queue = deque([(center, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= 3:
                continue
            for neighbor in G.neighbors(node):
                if neighbor not in sampled:
                    sampled.add(neighbor)
                    queue.append((neighbor, depth + 1))

        # Resample the center node when the 3-hop subgraph is too small.
        if len(sampled) < 10:
            continue

        # Resample the center node when the 3-hop subgraph is too large.
        if len(sampled) > max_nodes:
            continue

        if len(sampled) > n_target:
            sampled = set(random.sample(list(sampled), n_target))

        H = G.subgraph(sampled).copy()
        if H.number_of_edges() >= 1 and nx.is_connected(H.to_undirected()):
            return H

    return G.subgraph({random.choice(nodes)}).copy()


def generate_prompt_answer(task_name, graph, raw_texts=None, graph_id=None):
    assert task_name in FULL_TASKS, f"Invalid task name: {task_name}"

    # Select a center node and extract a 3-hop subgraph; retry if the size is out of range.
    center_node = random.choice(list(graph.nodes()))
    subgraph, actual_center, actual_hops = extract_ego_subgraph(graph, center_node)
    if subgraph is None:
        return None, None, None, None

    # Add graph ID information when available.
    graph_id_hint = f"\n[Note: This graph is from graph_id={graph_id}]\n" if graph_id is not None else ""

    # Pass raw_texts to the encoder for node text tasks.
    needs_text = task_name in ['node_text_description', 'node_id_from_text']
    subgraph_encoder = GraphTextEncoder(subgraph, raw_texts=raw_texts if needs_text else None)
    graph_description = subgraph_encoder.generate_graph_description(include_texts=needs_text) + graph_id_hint
    directed = subgraph_encoder.is_directed
    # Run calculations on the extracted subgraph.
    working_graph = subgraph

    # =========================================================================
    # edge_existence
    # =========================================================================
    if task_name == 'edge_existence':
        if len(working_graph.nodes()) < 2:
            return None, None, None, subgraph
        sample_nodes = random.sample(list(working_graph.nodes()), 2)
        edge_existence = working_graph.has_edge(sample_nodes[0], sample_nodes[1])
        answer = f"{'Yes' if edge_existence else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_nodes[0], sample_nodes[1])
            + FORMATS[task_name]
        )
        sample = f"{center_node}, ({sample_nodes[0]}, {sample_nodes[1]})"

    # =========================================================================
    # degree
    # =========================================================================
    elif task_name == 'degree':
        sample_node = random.choice(list(working_graph.nodes()))
        answer = f"{working_graph.degree(sample_node)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}, {sample_node}"

    # =========================================================================
    # common_neighbor
    # =========================================================================
    elif task_name == 'common_neighbor':
        G_und = working_graph.to_undirected()
        if len(G_und.nodes()) < 2:
            return None, None, None, subgraph
        sample_nodes = random.sample(list(G_und.nodes()), 2)
        common_neighbor = list(nx.common_neighbors(G_und, sample_nodes[0], sample_nodes[1]))
        if len(common_neighbor) == 0:
            return None, None, None, subgraph
        common_neighbor = sorted([int(u) for u in common_neighbor])
        answer = f"{common_neighbor}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_nodes[0], sample_nodes[1])
            + FORMATS[task_name]
        )
        sample = f"{center_node}, ({sample_nodes[0]}, {sample_nodes[1]})"

    # =========================================================================
    # neighbor
    # =========================================================================
    elif task_name == 'neighbor':
        sample_node = random.choice(list(working_graph.nodes()))
        neighbor = list(working_graph.neighbors(sample_node))
        if len(neighbor) == 0:
            return None, None, None, subgraph
        neighbor = sorted([int(u) for u in neighbor])
        answer = f"{neighbor}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}, {sample_node}"

    # =========================================================================
    # Tasks requiring subgraph extraction: graph-level questions
    # =========================================================================
    elif task_name == 'node_number':
        answer = f"{len(working_graph.nodes())}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'edge_number':
        answer = f"{len(working_graph.edges())}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'density':
        answer = f"{nx.density(working_graph)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'has_cycle':
        try:
            nx.find_cycle(working_graph)
            has_cycle = True
        except nx.NetworkXNoCycle:
            has_cycle = False

        if binary_cate[has_cycle] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[has_cycle] += 1
        answer = f"{'Yes' if has_cycle else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'is_regular':
        is_regular = nx.is_regular(working_graph)
        if binary_cate[is_regular] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[is_regular] += 1
        answer = f"{'Yes' if is_regular else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'is_bipartite':
        is_bipartite = nx.is_bipartite(working_graph)
        if binary_cate[is_bipartite] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[is_bipartite] += 1
        answer = f"{'Yes' if is_bipartite else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'is_connected':
        G_und = working_graph.to_undirected()
        is_connected = nx.is_connected(G_und)
        if binary_cate[is_connected] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[is_connected] += 1
        answer = f"{'Yes' if is_connected else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'is_weakly_connected' and directed:
        is_weakly_connected = nx.is_weakly_connected(working_graph)
        if binary_cate[is_weakly_connected] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[is_weakly_connected] += 1
        answer = f"{'Yes' if is_weakly_connected else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'is_strongly_connected' and directed:
        is_strongly_connected = nx.is_strongly_connected(working_graph)
        if binary_cate[is_strongly_connected] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[is_strongly_connected] += 1
        answer = f"{'Yes' if is_strongly_connected else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'is_tournament' and directed:
        is_tournament = nx.is_tournament(working_graph)
        if binary_cate[is_tournament] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[is_tournament] += 1
        answer = f"{'Yes' if is_tournament else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'is_eularian':
        is_eulerian = nx.is_eulerian(working_graph)
        if binary_cate[is_eulerian] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[is_eulerian] += 1
        answer = f"{'Yes' if is_eulerian else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'is_eulerian':
        is_eulerian = nx.is_eulerian(working_graph)
        if binary_cate[is_eulerian] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[is_eulerian] += 1
        answer = f"{'Yes' if is_eulerian else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'topological_sort' and directed:
        try:
            nx.find_cycle(working_graph)
            return None, None, None, subgraph
        except nx.NetworkXUnfeasible:
            topological_sort = list(nx.topological_sort(working_graph))
            assert is_valid_topological_sort(working_graph, topological_sort)
            topological_sort = [int(u) for u in topological_sort]
            answer = f"{topological_sort}"
            prompt = (
                INSTRUCTIONS[task_name]
                + graph_description
                + QUESTIONS[task_name].format(center_node)
                + FORMATS[task_name]
            )
            sample = f"{center_node}"

    elif task_name == 'transitivity':
        answer = f"{nx.transitivity(working_graph)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'diameter':
        G_und = working_graph.to_undirected()
        answer = f"{nx.diameter(G_und)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'radius':
        G_und = working_graph.to_undirected()
        answer = f"{nx.radius(G_und)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'periphery':
        G_und = working_graph.to_undirected()
        periphery = nx.periphery(G_und)
        periphery = sorted([int(u) for u in periphery])
        answer = f"{periphery}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'center':
        G_und = working_graph.to_undirected()
        center = nx.center(G_und)
        center = sorted([int(u) for u in center])
        answer = f"{center}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'barycenter':
        G_und = working_graph.to_undirected()
        barycenter = nx.barycenter(G_und)
        barycenter = sorted([int(u) for u in barycenter])
        answer = f"{barycenter}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'wiener_index':
        G_und = working_graph.to_undirected()
        wiener_index = nx.wiener_index(G_und)
        if math.isinf(wiener_index):
            return None, None, None, subgraph
        answer = f"{wiener_index}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'global_efficiency':
        G_und = working_graph.to_undirected()
        answer = f"{nx.global_efficiency(G_und)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'isolate_number':
        answer = f"{nx.number_of_isolates(working_graph)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'connected_component_number':
        G_und = working_graph.to_undirected()
        answer = f"{nx.number_connected_components(G_und)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'strongly_connected_number' and directed:
        answer = f"{nx.number_strongly_connected_components(working_graph)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'weakly_connected_number':
        if directed:
            answer = f"{nx.number_weakly_connected_components(working_graph)}"
        else:
            # For undirected graphs, use number_connected_components
            answer = f"{nx.number_connected_components(working_graph)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'degree_assortativity':
        degree_assortativity = nx.degree_assortativity_coefficient(working_graph)
        if math.isnan(degree_assortativity):
            return None, None, None, subgraph
        answer = f"{degree_assortativity}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'min_edge_covering':
        G_und = get_undirected_graph_no_selfloops(working_graph)
        # min_edge_cover requires all nodes to have at least one incident edge
        if nx.number_of_isolates(G_und) > 0:
            return None, None, None, subgraph
        min_edge_covering = nx.min_edge_cover(G_und)
        assert is_minimum_edge_cover(G_und, list(min_edge_covering))
        min_edge_covering = sorted([tuple(int(v) for v in tup) for tup in min_edge_covering])
        answer = f"{min_edge_covering}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'bridges':
        G_und = working_graph.to_undirected()
        bridges = list(nx.bridges(G_und))
        bridges = sorted([tuple(int(v) for v in tup) for tup in bridges])
        answer = f"{bridges}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'minimum_spanning_tree':
        G_und = working_graph.to_undirected()
        mst = list(nx.minimum_spanning_edges(G_und, data=False))
        assert is_minimum_spanning_tree(G_und, mst)
        mst = sorted([(int(tup[0]), int(tup[1])) for tup in mst])
        answer = f"{mst}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'weighted_minimum_spanning_tree':
        working_graph_und = working_graph.to_undirected()
        mst = list(nx.minimum_spanning_edges(working_graph_und, data=False))
        mst = sorted([(int(tup[0]), int(tup[1])) for tup in mst])
        answer = f"{mst}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'dominating_set':
        dominating_set = list(nx.dominating_set(working_graph))
        assert is_valid_dominating_set(working_graph, dominating_set)
        dominating_set = sorted([int(u) for u in dominating_set])
        answer = f"{dominating_set}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'min_vertex_cover':
        G_und = working_graph.to_undirected()
        minimal_vertex_cover = list(min_vertex_cover(G_und))
        assert is_minimum_vertex_cover(G_und, minimal_vertex_cover)
        minimal_vertex_cover = sorted([int(u) for u in minimal_vertex_cover])
        answer = f"{minimal_vertex_cover}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'max_weight_matching':
        G_und = get_undirected_graph_no_selfloops(working_graph)
        max_weight_matching = nx.max_weight_matching(G_und)
        assert is_maximal_weight_matching(G_und, max_weight_matching)
        max_weight_matching = sorted([tuple(int(v) for v in tup) for tup in max_weight_matching])
        answer = f"{max_weight_matching}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'min_weight_matching':
        G_und = get_undirected_graph_no_selfloops(working_graph)
        max_weight_matching = nx.max_weight_matching(G_und)
        max_weight_matching = sorted([tuple(int(v) for v in tup) for tup in max_weight_matching])
        answer = f"{max_weight_matching}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'maximal_cliques':
        G_und = working_graph.to_undirected()
        maximal_cliques = list(nx.find_cliques(G_und))
        maximal_cliques = sorted([sorted([int(u) for u in clique]) for clique in maximal_cliques])
        answer = f"{maximal_cliques}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'maximal_cliques':
        try:
            maximal_cliques = list(nx.find_cliques(working_graph))
            maximal_cliques = [sorted([int(u) for u in clique]) for clique in maximal_cliques]
            answer = f"{maximal_cliques}"
        except nx.NetworkXError:
            return None, None, None, subgraph
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    elif task_name == 'k_core':
        G_und = get_undirected_graph_no_selfloops(working_graph)
        k_core = nx.k_core(G_und)
        k_core_nodes = sorted([int(u) for u in k_core.nodes()])
        answer = f"{k_core_nodes}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    # =========================================================================
    # Tasks requiring subgraph extraction: node-level questions
    # =========================================================================
    elif task_name == 'triangles':
        G_und = get_undirected_graph_no_selfloops(working_graph)
        triangles = nx.triangles(G_und)
        triangles = sorted(triangles.items(), key=lambda x: x[1], reverse=True)
        triangles_node = triangles[0][0]
        triangles_count = triangles[0][1]
        answer = f"{triangles_count}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, triangles_node)
            + FORMATS[task_name]
        )
        sample = f"{triangles_node}"

    elif task_name == 'clustering':
        G_und = working_graph.to_undirected()
        sample_node = random.choice(list(G_und.nodes()))
        answer = f"{nx.clustering(G_und, sample_node)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'clustering_coefficient':
        G_und = working_graph.to_undirected()
        sample_node = random.choice(list(G_und.nodes()))
        answer = f"{nx.clustering(G_und, sample_node)}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'constraint':
        constraint = nx.constraint(working_graph)
        constraint = dict(filter(lambda x: not math.isnan(x[1]), constraint.items()))
        if len(constraint) == 0:
            return None, None, None, subgraph
        sample_node = random.choice(list(constraint.keys()))
        answer = f"{constraint[sample_node]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'effective_size':
        try:
            effective_size = nx.effective_size(working_graph)
        except ZeroDivisionError:
            return None, None, None, subgraph
        effective_size = dict(filter(lambda x: not math.isnan(x[1]), effective_size.items()))
        if len(effective_size) == 0:
            return None, None, None, subgraph
        sample_node = random.choice(list(effective_size.keys()))
        answer = f"{effective_size[sample_node]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'degree_centrality':
        sample_node = random.choice(list(working_graph.nodes()))
        answer = f"{nx.degree_centrality(working_graph)[sample_node]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'closeness_centrality':
        sample_node = random.choice(list(working_graph.nodes()))
        closeness_centrality = nx.closeness_centrality(working_graph)
        answer = f"{closeness_centrality[sample_node]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'betweenness_centrality':
        sample_node = random.choice(list(working_graph.nodes()))
        betweenness_centrality = nx.betweenness_centrality(working_graph)
        answer = f"{betweenness_centrality[sample_node]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'harmonic_centrality':
        harmonic_centrality = nx.harmonic_centrality(working_graph)
        sample_node = random.choice(list(working_graph.nodes()))
        answer = f"{harmonic_centrality[sample_node]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'avg_neighbor_degree':
        sample_node = random.choice(list(working_graph.nodes()))
        avg_neighbor_degree = nx.average_neighbor_degree(working_graph)
        answer = f"{avg_neighbor_degree[sample_node]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'maximal_independent_set':
        G_und = get_undirected_graph_no_selfloops(working_graph)
        # First compute the maximal independent set, then pick a node from it
        maximal_independent_set = list(nx.maximal_independent_set(G_und))
        if not maximal_independent_set:
            return None, None, None, subgraph
        sample_node = random.choice(maximal_independent_set)
        maximal_independent_set = sorted([int(u) for u in maximal_independent_set])
        answer = f"{maximal_independent_set}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'local_connectivity':
        if len(list(working_graph.nodes())) < 2:
            return None, None, None, subgraph
        sample_nodes = random.sample(list(working_graph.nodes()), 2)
        local_connectivity = nx.has_path(working_graph, sample_nodes[0], sample_nodes[1])
        if binary_cate[local_connectivity] > sample_num // 2:
            return None, None, None, subgraph
        binary_cate[local_connectivity] += 1
        answer = f"{'Yes' if local_connectivity else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_nodes[0], sample_nodes[1])
            + FORMATS[task_name]
        )
        sample = f"{sample_nodes[0]}"

    # =========================================================================
    # Tasks requiring subgraph extraction: edge-level questions
    # =========================================================================
    elif task_name == 'shortest_path':
        shortest_path_all = dict(nx.shortest_path(working_graph))
        shortest_path_all = {(u, v): path for u, inner_dict in shortest_path_all.items() for v, path in inner_dict.items()}
        shortest_path_all = dict(sorted(shortest_path_all.items(), key=lambda x: len(x[1]), reverse=True))
        sample_nodes = list(shortest_path_all.keys())[0]
        shortest_path = shortest_path_all[sample_nodes]
        assert is_valid_shortest_path(working_graph, shortest_path, sample_nodes[0], sample_nodes[1])
        shortest_path = [int(u) for u in shortest_path]
        answer = f"{shortest_path}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_nodes[0], sample_nodes[1])
            + FORMATS[task_name]
        )
        sample = f"{sample_nodes[0]}"

    elif task_name == 'weighted_shortest_path':
        shortest_path_all = dict(nx.shortest_path(working_graph))
        shortest_path_all = {(u, v): path for u, inner_dict in shortest_path_all.items() for v, path in inner_dict.items()}
        shortest_path_all = dict(sorted(shortest_path_all.items(), key=lambda x: len(x[1]), reverse=True))
        sample_nodes = list(shortest_path_all.keys())[0]
        shortest_path = shortest_path_all[sample_nodes]
        shortest_path = [int(u) for u in shortest_path]
        answer = f"{shortest_path}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_nodes[0], sample_nodes[1])
            + FORMATS[task_name]
        )
        sample = f"{sample_nodes[0]}"

    elif task_name == 'bfs':
        sample_node = random.choice(list(working_graph.nodes()))
        bfs = list(nx.bfs_edges(working_graph, source=sample_node, sort_neighbors=sorted))
        assert is_valid_bfs_sequence(working_graph, bfs, sample_node)
        bfs = [(int(e[0]), int(e[1])) for e in bfs]
        answer = f"{bfs}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'dfs':
        sample_node = random.choice(list(working_graph.nodes()))
        dfs = list(nx.dfs_edges(working_graph, source=sample_node))
        assert is_valid_dfs_sequence(working_graph, dfs, sample_node)
        dfs = [(int(e[0]), int(e[1])) for e in dfs]
        answer = f"{dfs}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'maximum_flow':
        if working_graph.number_of_nodes() < 2:
            return None, None, None, subgraph
        sample_nodes = random.sample(list(working_graph.nodes()), 2)
        G_flow = working_graph.copy()
        if not all('capacity' in data for _, _, data in G_flow.edges(data=True)):
            if G_flow.is_multigraph():
                for u, v, k in G_flow.edges(keys=True):
                    if 'capacity' not in G_flow[u][v][k]:
                        G_flow[u][v][k]['capacity'] = 1
            else:
                for u, v in G_flow.edges():
                    if 'capacity' not in G_flow[u][v]:
                        G_flow[u][v]['capacity'] = 1
        try:
            flow_value, _ = nx.maximum_flow(G_flow, sample_nodes[0], sample_nodes[1])
            answer = f"{flow_value}"
        except (nx.NetworkXError, nx.NetworkXUnbounded):
            return None, None, None, subgraph
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_nodes[0], sample_nodes[1])
            + FORMATS[task_name]
        )
        sample = f"{sample_nodes[0]}"

    elif task_name == 'resource_allocation_index':
        G_und = working_graph.to_undirected()
        resource_allocation_index = list(nx.resource_allocation_index(G_und))
        if len(resource_allocation_index) == 0:
            return None, None, None, subgraph
        sample_nodes = random.choice(resource_allocation_index)
        answer = f"{sample_nodes[2]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_nodes[0], sample_nodes[1])
            + FORMATS[task_name]
        )
        sample = f"{sample_nodes[0]}"

    elif task_name == 'jaccard_coefficient':
        G_und = working_graph.to_undirected()
        jaccard_coefficient = list(nx.jaccard_coefficient(G_und))
        if len(jaccard_coefficient) == 0:
            return None, None, None, subgraph
        sample_nodes = random.choice(jaccard_coefficient)
        answer = f"{sample_nodes[2]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_nodes[0], sample_nodes[1])
            + FORMATS[task_name]
        )
        sample = f"{sample_nodes[0]}"

    elif task_name == 'adamic_adar_index':
        G_und = working_graph.to_undirected()
        adamic_adar_index = list(nx.adamic_adar_index(G_und))
        if len(adamic_adar_index) == 0:
            return None, None, None, subgraph
        sample_nodes = random.choice(adamic_adar_index)
        answer = f"{sample_nodes[2]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_nodes[0], sample_nodes[1])
            + FORMATS[task_name]
        )
        sample = f"{sample_nodes[0]}"

    # =========================================================================
    # Special tasks
    # =========================================================================
    elif task_name == 'ego_subgraph_3hop':
        ego_edges = sorted([(int(u), int(v)) for u, v in working_graph.edges()])
        answer = f"{ego_edges}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    # =========================================================================
    # Graph information retrieval: node text tasks
    # =========================================================================
    elif task_name == 'node_text_description':
        if raw_texts is None:
            return None, None, None, subgraph
        sample_node = random.choice(list(working_graph.nodes()))
        # Get the text description for the sample node
        if sample_node < len(raw_texts):
            text_description = raw_texts[sample_node]
        else:
            return None, None, None, subgraph
        answer = f"{text_description}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}, {sample_node}"

    elif task_name == 'node_id_from_text':
        if raw_texts is None:
            return None, None, None, subgraph
        # Randomly select a node that has a valid text description
        valid_nodes = [n for n in working_graph.nodes() if n < len(raw_texts)]
        if len(valid_nodes) == 0:
            return None, None, None, subgraph
        sample_node = random.choice(valid_nodes)
        target_text = raw_texts[sample_node]
        answer = f"{sample_node}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, target_text)
            + FORMATS[task_name]
        )
        sample = f"{center_node}, '{target_text[:50]}...'"

    else:
        answer, prompt, sample = None, None, None

    return prompt, answer, sample, subgraph



def generate_graph_tasks(task_name, sample_num, graph_size, verbose=False, single_graph_path=None, graph_list_size=40, samples_per_graph=5):
    task_data = {
        'task': [],
        'prompt': [],
        'answer': [],
        'source_file': [],
        'direction': [],
        'nodes': [],
        'edges': [],
        'samples': [],
        'answer_type': [],
        'tool_select': [],
        'subgraph_nodes': [],
        'subgraph_edges': [],
        'graph_id': [],
    }
    binary_cate[True] = 0
    binary_cate[False] = 0

    if single_graph_path is not None:
        # Check if it's a graph list file
        if single_graph_path.endswith('.pt'):
            try:
                import torch
                data = torch.load(single_graph_path, map_location='cpu', weights_only=False)
                if isinstance(data, (list, tuple)) and len(data) > 0:
                    # It's a list of graphs
                    graph_list = load_graph_list(single_graph_path)
                    total_graphs = len(graph_list)
                    print(f"Loaded {total_graphs} graphs from {single_graph_path}")

                    # Randomly select graph_list_size graphs
                    if len(graph_list) <= graph_list_size:
                        selected_indices = list(range(len(graph_list)))
                    else:
                        selected_indices = random.sample(range(len(graph_list)), graph_list_size)

                    print(f"Selected {len(selected_indices)} graphs for task generation")

                    for graph_idx in selected_indices:
                        edges, directed, weighted, raw_texts = graph_list[graph_idx]
                        G = npz_to_nx(edges, directed, weighted)

                        if len(G.nodes()) <= 2 or len(G.edges()) <= 2:
                            continue

                        # Generate samples_per_graph samples for this graph
                        for _ in range(samples_per_graph * 3):  # try 3x to ensure we get enough
                            graph = sample_subgraph(G, graph_size)
                            prompt, answer, sample, subgraph = generate_prompt_answer(task_name, graph, raw_texts, graph_id=graph_idx)
                            if prompt is None or answer is None:
                                continue

                            subgraph_encoder = GraphTextEncoder(subgraph)
                            subgraph_nodes = sorted(subgraph_encoder.name_dict.values())
                            subgraph_edges = [(u, v) for u, v in subgraph.edges()]
                            is_directed = subgraph_encoder.is_directed

                            task_data['task'].append(task_name)
                            task_data['prompt'].append(prompt)
                            task_data['answer'].append(answer)
                            task_data['source_file'].append(single_graph_path)
                            task_data['direction'].append('directed' if is_directed else 'undirected')
                            task_data['nodes'].append(subgraph_nodes)
                            task_data['edges'].append(subgraph_edges)
                            task_data['samples'].append(sample)
                            task_data['answer_type'].append(ANSWER_TYPES[task_name])
                            task_data['tool_select'].append(
                                ['extract_subgraph'] + TOOLSETS[task_name]
                                if not TOOLSETS[task_name] or TOOLSETS[task_name][0] != 'extract_subgraph'
                                else TOOLSETS[task_name]
                            )
                            task_data['subgraph_nodes'].append(subgraph_nodes)
                            task_data['subgraph_edges'].append(subgraph_edges)
                            task_data['graph_id'].append(graph_idx)

                            if len(task_data['answer']) >= sample_num:
                                break
                        if len(task_data['answer']) >= sample_num:
                            break

                    print(f"Generated {len(task_data['answer'])} samples from {len(selected_indices)} graphs")
                    return task_data
            except Exception as e:
                print(f"Error loading graph list, falling back to single graph mode: {e}")

        # Single graph mode (original behavior)
        edges, directed, weighted, raw_texts = load_graph_data(single_graph_path)
        G = npz_to_nx(edges, directed, weighted)
        for _ in range(100):
            graph = sample_subgraph(G, graph_size)
            prompt, answer, sample, subgraph = generate_prompt_answer(task_name, graph, raw_texts, graph_id=-1)
            if prompt is None or answer is None:
                continue

            subgraph_encoder = GraphTextEncoder(subgraph)
            subgraph_nodes = sorted(subgraph_encoder.name_dict.values())
            subgraph_edges = [(u, v) for u, v in subgraph.edges()]
            is_directed = subgraph_encoder.is_directed

            task_data['task'].append(task_name)
            task_data['prompt'].append(prompt)
            task_data['answer'].append(answer)
            task_data['source_file'].append(single_graph_path)
            task_data['direction'].append('directed' if is_directed else 'undirected')
            task_data['nodes'].append(subgraph_nodes)
            task_data['edges'].append(subgraph_edges)
            task_data['samples'].append(sample)
            task_data['answer_type'].append(ANSWER_TYPES[task_name])
            task_data['tool_select'].append(
                ['extract_subgraph'] + TOOLSETS[task_name]
                if not TOOLSETS[task_name] or TOOLSETS[task_name][0] != 'extract_subgraph'
                else TOOLSETS[task_name]
            )
            task_data['subgraph_nodes'].append(subgraph_nodes)
            task_data['subgraph_edges'].append(subgraph_edges)
            task_data['graph_id'].append(-1)  # -1 indicates single graph mode

            if len(task_data['answer']) >= sample_num:
                break
    else:
        dataset_paths = get_dataset_paths(graph_size)
        print(f"Total number of datasets: {len(dataset_paths)}")

        index = 0
        for dataset_path in tqdm(dataset_paths):
            edges, directed, weighted, raw_texts = load_graph_data(file_path=dataset_path)
            graph = npz_to_nx(edges, directed, weighted)
            graph_text_encoder = GraphTextEncoder(graph)

            if len(graph.nodes()) <=2 or len(graph.edges()) <= 2:
                continue
            if weighted:
                continue
            prompt, answer, sample, subgraph = generate_prompt_answer(task_name, graph, raw_texts, graph_id=index)
            if prompt is None or answer is None:
                continue
            
            subgraph_encoder = GraphTextEncoder(subgraph)
            subgraph_nodes = sorted(subgraph_encoder.name_dict.values())
            subgraph_edges = [(u, v) for u, v in subgraph.edges()]

            task_data['task'].append(task_name)
            task_data['prompt'].append(prompt)
            task_data['answer'].append(answer)
            task_data['source_file'].append(dataset_path)
            task_data['direction'].append('directed' if directed else 'undirected')
            # Store the full node list.
            task_data['nodes'].append(subgraph_nodes)
            task_data['edges'].append(subgraph_edges)
            task_data['samples'].append(sample)
            task_data['answer_type'].append(ANSWER_TYPES[task_name])
            task_data['tool_select'].append(
                ['extract_subgraph'] + TOOLSETS[task_name] 
                if not TOOLSETS[task_name] or TOOLSETS[task_name][0] != 'extract_subgraph' 
                else TOOLSETS[task_name]
            )
            task_data['subgraph_nodes'].append(subgraph_nodes)
            task_data['subgraph_edges'].append(subgraph_edges)

            if len(task_data['answer']) >= sample_num:
                print(f"Processed {len(task_data['answer'])} tasks, stopping.")
                if task_name in BINARY_TASKS:
                    print(f"Binary cate: {binary_cate}")
                break

    return task_data



parser = argparse.ArgumentParser(description="Generate RL datasets")
parser.add_argument('--sample_num', type=int, default=20000, help='Number of samples to generate')
parser.add_argument('--graph_size', type=str, default='Tiny', help='Size of the graph: Tiny, Small, Medium, Large')
parser.add_argument('--single_graph', type=str, default=None, help='Path to a single graph npz file, e.g. xxx')
parser.add_argument('--graph_list_size', type=int, default=40, help='Number of graphs to select from graph list (for .pt list files)')
parser.add_argument('--samples_per_graph', type=int, default=5, help='Number of samples to generate per graph')
args = parser.parse_args()

from questions import QUESTION_CATEGORIES

parent_path='pcba'
os.makedirs(parent_path, exist_ok=True)
sample_num = args.sample_num
graph_size = args.graph_size
single_graph_path = args.single_graph
suffix = f"single_{os.path.splitext(os.path.basename(single_graph_path))[0]}" if single_graph_path else graph_size

# Create the category directory structure.
graph_info_dir = os.path.join(parent_path, 'graph_info_retrieval')
graph_theory_dir = os.path.join(parent_path, 'graph_theory')
for category_dir in [graph_info_dir, graph_theory_dir]:
    os.makedirs(os.path.join(category_dir, 'node'), exist_ok=True)
    os.makedirs(os.path.join(category_dir, 'edge'), exist_ok=True)
    os.makedirs(os.path.join(category_dir, 'graph'), exist_ok=True)

binary_cate = {True: 0, False: 0}
print(f"Total number of tasks: {len(FULL_TASKS)}")
print(f"Sample number per task: {sample_num}")

# Build the task-to-directory mapping before it is used.
def get_task_dir(task_name):
    for category, subcats in QUESTION_CATEGORIES.items():
        for subcat, tasks in subcats.items():
            if task_name in tasks:
                if category == 'graph_info_retrieval':
                    base_dir = graph_info_dir
                else:
                    base_dir = graph_theory_dir
                return os.path.join(base_dir, subcat)
    return None

# Count tasks in each category.
graph_info_tasks = []
graph_theory_tasks = []
for task in FULL_TASKS:
    task_dir = get_task_dir(task)
    if task_dir and 'graph_info_retrieval' in task_dir:
        graph_info_tasks.append(task)
    else:
        graph_theory_tasks.append(task)

num_info_tasks = len(graph_info_tasks)
num_theory_tasks = len(graph_theory_tasks)
print(f"graph_info_retrieval tasks: {num_info_tasks}")
print(f"graph_theory tasks: {num_theory_tasks}")

# Allocate balanced sample counts across categories.
# Using the same per-task sample count would make category totals differ.
# Adjust by category task counts while keeping enough samples per task.

# Compute per-task sample counts so category totals are close.
# Total samples / task count = per-task sample count.
# graph_info: sample_per_info = total / num_info_tasks
# graph_theory: sample_per_theory = total / num_theory_tasks
# Keep sample_num as the base and adjust totals where needed.

def task_output_path(task_name, task_dir, suffix, count):
    filename = f"{task_name}_{suffix}_{count}.csv"
    return os.path.join(task_dir, filename) if task_dir else os.path.join(parent_path, filename)


def generate_mixed_tasks(sample_num, graph_size, task_list, graph_list_size=40, samples_per_graph=5, graph_list=None, single_graph_path=None):
    """Generate mixed tasks from a list of task types.

    Args:
        sample_num: Total number of samples to generate
        graph_size: Size of sampled subgraphs
        task_list: List of task names to generate
        graph_list_size: Number of graphs to select
        samples_per_graph: Number of samples per graph
        graph_list: Pre-loaded list of graphs (optional)
        single_graph_path: Path to graph list file (optional, will load from file if graph_list not provided)

    Returns:
        dict: Task data with all fields
    """
    import torch

    task_data = {
        'task': [],
        'prompt': [],
        'answer': [],
        'source_file': [],
        'direction': [],
        'nodes': [],
        'edges': [],
        'samples': [],
        'answer_type': [],
        'tool_select': [],
        'subgraph_nodes': [],
        'subgraph_edges': [],
        'graph_id': [],
        'level': [],  # node / edge / graph
    }

    # Load graph list if not provided
    if graph_list is None:
        if single_graph_path is None or not single_graph_path.endswith('.pt'):
            raise ValueError("Either graph_list or single_graph_path must be provided")
        graph_list = load_graph_list(single_graph_path)
    else:
        # Normalize raw graph items to (edges, directed, weighted, raw_texts) tuples
        normalized = []
        for i, item in enumerate(graph_list):
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
                        raw_texts = None
                        for store in item.stores:
                            if 'raw_texts' in store:
                                raw_texts = store['raw_texts']
                                break
                    elif hasattr(item, 'raw_texts'):
                        raw_texts = item.raw_texts
                    else:
                        raw_texts = None
                    normalized.append((edges, directed, weighted, raw_texts))
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
                    edges = edges_data.tolist() if hasattr(edges_data, 'ndim') and edges_data.ndim == 2 else edges_data
                else:
                    raise ValueError(f"Unsupported graph item at index {i}: missing 'edge_index' or 'edges'")
                directed = item.get('directed', item.get('is_directed', True))
                weighted = item.get('weighted', False)
                raw_texts = item.get('raw_texts', None)
                normalized.append((edges, directed, weighted, raw_texts))
                continue

            raise ValueError(f"Unsupported graph item type at index {i}: {type(item)}")
        graph_list = normalized
    
    total_graphs = len(graph_list)
    print(f"Loaded {total_graphs} graphs")

    # Select graphs
    if len(graph_list) <= graph_list_size:
        selected_indices = list(range(len(graph_list)))
    else:
        selected_indices = random.sample(range(len(graph_list)), graph_list_size)

    print(f"Selected {len(selected_indices)} graphs")

    # Shuffle task list for variety
    shuffled_tasks = task_list.copy()
    random.shuffle(shuffled_tasks)
    task_idx = 0

    # Calculate how many samples per task
    num_tasks = len(task_list)
    samples_per_task = sample_num // num_tasks
    extra_samples = sample_num % num_tasks  # distribute remaining samples

    # Track samples per task
    task_sample_count = {t: 0 for t in task_list}

    # Generate samples: cycle through tasks for each graph
    for graph_idx in selected_indices:
        edges, directed, weighted, raw_texts = graph_list[graph_idx]
        G = npz_to_nx(edges, directed, weighted)

        if len(G.nodes()) <= 2 or len(G.edges()) <= 2:
            continue

        # Get next samples_per_graph task types for this graph
        for _ in range(samples_per_graph):
            if len(task_data['answer']) >= sample_num:
                break

            task_name = shuffled_tasks[task_idx % num_tasks]
            task_idx += 1

            # Skip if we've generated enough for this task
            if task_sample_count[task_name] >= samples_per_task + (1 if task_idx <= extra_samples else 0):
                continue

            # Try multiple times to generate a valid sample
            for _ in range(50):
                graph = sample_subgraph(G, graph_size)
                prompt, answer, sample, subgraph = generate_prompt_answer(task_name, graph, raw_texts, graph_id=graph_idx)
                if prompt is None or answer is None:
                    continue

                subgraph_encoder = GraphTextEncoder(subgraph)
                subgraph_nodes = sorted(subgraph_encoder.name_dict.values())
                subgraph_edges = [(u, v) for u, v in subgraph.edges()]
                is_directed = subgraph_encoder.is_directed

                task_data['task'].append(task_name)
                task_data['prompt'].append(prompt)
                task_data['answer'].append(answer)
                task_data['source_file'].append(single_graph_path)
                task_data['direction'].append('directed' if is_directed else 'undirected')
                task_data['nodes'].append(subgraph_nodes)
                task_data['edges'].append(subgraph_edges)
                task_data['samples'].append(sample)
                task_data['answer_type'].append(ANSWER_TYPES[task_name])
                task_data['tool_select'].append(
                    ['extract_subgraph'] + TOOLSETS[task_name]
                    if not TOOLSETS[task_name] or TOOLSETS[task_name][0] != 'extract_subgraph'
                    else TOOLSETS[task_name]
                )
                task_data['subgraph_nodes'].append(subgraph_nodes)
                task_data['subgraph_edges'].append(subgraph_edges)
                task_data['graph_id'].append(graph_idx)
                task_data['level'].append(get_task_level(task_name))  # node / edge / graph
                task_sample_count[task_name] += 1
                break

        if len(task_data['answer']) >= sample_num:
            break

    print(f"Generated {len(task_data['answer'])} samples covering {len(set(task_data['task']))} task types")
    return task_data


# Main execution
if __name__ == "__main__":
    print(f"[DEBUG] single_graph_path: {single_graph_path}")
    print(f"[DEBUG] graph_list_size: {args.graph_list_size}")
    print(f"[DEBUG] samples_per_graph: {args.samples_per_graph}")

    if single_graph_path and single_graph_path.endswith('.pt'):
        # Check if it's a graph list
        try:
            import torch
            print(f"[DEBUG] Loading {single_graph_path}...")
            data = torch.load(single_graph_path, map_location='cpu', weights_only=False)
            print(f"[DEBUG] Loaded data type: {type(data)}, len: {len(data) if hasattr(data, '__len__') else 'N/A'}")

            # Handle dict format (e.g., hiv.pt with {'graphs': [...], 'texts': [...], 'split': [...]})
            if isinstance(data, dict):
                if 'graphs' in data:
                    print(f"[DEBUG] Detected dict format with 'graphs' key")
                    graph_list = data['graphs']
                    # Process as graph list
                    total_graphs = len(graph_list)
                    print(f"[DEBUG] Found {total_graphs} graphs in dict")
                else:
                    raise ValueError(f"Dict format not recognized. Keys: {data.keys()}")
                is_graph_list = True
            elif isinstance(data, (list, tuple)) and len(data) > 0:
                graph_list = data
                is_graph_list = True
            else:
                is_graph_list = False
                graph_list = None
            print(f"[DEBUG] is_graph_list: {is_graph_list}")

            if is_graph_list and args.graph_list_size > 0:
                # Graph list mode: generate 200 samples for each category
                print(f"\n=== Graph List Mode ===")

                for category, tasks in [('graph_info_retrieval', graph_info_tasks), ('graph_theory', graph_theory_tasks)]:
                    print(f"\n--- {category} ---")
                    print(f"Tasks: {len(tasks)}")
                    print(f"Target samples: {sample_num}")

                    task_data = generate_mixed_tasks(
                        sample_num=sample_num,
                        graph_size=graph_size,
                        task_list=tasks,
                        graph_list_size=args.graph_list_size,
                        samples_per_graph=args.samples_per_graph,
                        graph_list=graph_list
                    )

                    print(f"[DEBUG] task_data answer count: {len(task_data['answer'])}")

                    # Save to category directory, split by level
                    category_dir = os.path.join(parent_path, category)
                    os.makedirs(category_dir, exist_ok=True)
                    total_count = len(task_data['answer'])
                    for level_value in ['node', 'edge', 'graph']:
                        level_indices = [i for i, lv in enumerate(task_data['level']) if lv == level_value]
                        level_count = len(level_indices)
                        if level_count == 0:
                            continue
                        level_data = {key: [values[i] for i in level_indices] for key, values in task_data.items()}
                        level_path = os.path.join(category_dir, f"mixed_tasks_{suffix}_{level_count}_{level_value}.csv")
                        save_datasets(level_data, save_path=level_path)
                        print(f"[DEBUG] Saved level={level_value} ({level_count} samples) to {level_path}")

                    # Print task distribution
                    from collections import Counter
                    task_counts = Counter(task_data['task'])
                    print(f"\n=== {category} Task Distribution ===")
                    for task, count in sorted(task_counts.items()):
                        print(f"  {task}: {count}")
                    print(f"\nTotal: {len(task_data['answer'])} samples from {len(set(task_data['task']))} task types")

                print(f"\n=== All Done ===")
                exit(0)
            else:
                print(f"[DEBUG] Graph list mode skipped: is_graph_list={is_graph_list}, graph_list_size={args.graph_list_size}")
        except Exception as e:
            print(f"Error in graph list mode: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to standard task-by-task generation...")

# Print summary statistics.
print(f"\n=== Summary ===")
print(f"graph_info_retrieval: {num_info_tasks} tasks × {sample_num * num_theory_tasks // num_info_tasks} = {num_info_tasks * sample_num * num_theory_tasks // num_info_tasks} samples")
print(f"graph_theory: {num_theory_tasks} tasks × {sample_num} = {num_theory_tasks * sample_num} samples")

import os
from collections import deque
import networkx as nx
import random
import argparse
import math
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
# Task List Definition
# =============================================================================
# Note:
#   - edge_existence, degree, common_neighbor, neighbor do not require subgraph extraction
#   - All other tasks require extracting a 3-hop completely induced subgraph centered on node{}
# =============================================================================

FULL_TASKS = [
    # Tasks that do not require subgraph extraction
    'edge_existence', 'degree', 'common_neighbor', 'neighbor',

    # Tasks that require subgraph extraction
    'node_number', 'edge_number', 'shortest_path', 'bfs', 'dfs',
    'triangles', 'density', #'topological_sort',
    'has_cycle',
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
    'is_weakly_connected', 'is_strongly_connected',
    'is_tournament', 'bridges', 'minimum_spanning_tree',
    'weighted_minimum_spanning_tree',
    'resource_allocation_index', 'jaccard_coefficient', 'adamic_adar_index',
    'maximum_flow', 'k_core', 'maximal_cliques',
    'weighted_shortest_path', 'ego_subgraph_3hop',

    # Graph info retrieval - node text-related tasks
    'node_text_description', 'node_id_from_text',
]

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
    # 'topological_sort': ['extract_subgraph', 'nx.topological_sort'],
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
    'is_eulerian', 'is_tournament', 'local_connectivity',
    'is_weakly_connected', 'is_strongly_connected',
]

TEXT_TASKS = {'node_text_description', 'node_id_from_text'}

SLOW_EXACT_TASKS = {
    'diameter', 'radius', 'periphery', 'center', 'barycenter', 'wiener_index',
    'global_efficiency', 'degree_assortativity', 'degree_connectivity',
    'minimum_spanning_tree', 'weighted_minimum_spanning_tree',
    'maximal_cliques', 'maximum_flow', 'min_vertex_cover',
    'max_weight_matching', 'min_weight_matching', 'min_edge_covering',
    'dominating_set', 'maximal_independent_set',
    'betweenness_centrality', 'closeness_centrality', 'harmonic_centrality',
    'constraint', 'effective_size', 'shortest_path', 'weighted_shortest_path',
    'local_connectivity',
}

# Tasks with non-unique answers
NOT_UNIQUE_TASKS = [
    'bfs', 'dfs', 'dominating_set', 'maximal_independent_set', 'min_edge_covering',
    'min_vertex_cover', 'minimum_spanning_tree', 'shortest_path',
    # 'topological_sort',
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
    'has_cycle', 'is_bipartite', 'is_regular',
    'is_weakly_connected', 'is_strongly_connected', 'is_tournament',
    'is_eulerian',
    # 'topological_sort',
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


def extract_n_hop_ego_subgraph(graph, center_node=None, hops=3, min_nodes=5, max_nodes=500):
    """Extract an n-hop completely induced subgraph centered on a node."""
    max_attempts = 100
    for _ in range(max_attempts):
        if center_node is None:
            current_center = random.choice(list(graph.nodes()))
        else:
            current_center = center_node

        sampled = {current_center}
        queue = deque([(current_center, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in sampled:
                    sampled.add(neighbor)
                    queue.append((neighbor, depth + 1))

        H = graph.subgraph(sampled).copy()
        num_nodes = H.number_of_nodes()

        if min_nodes <= num_nodes <= max_nodes:
            return H, current_center, hops

        if center_node is not None:
            return H, center_node, hops

    return H, current_center, hops


def extract_ego_subgraph(graph, center_node=None, hops=3, min_nodes=5, max_nodes=500):
    return extract_n_hop_ego_subgraph(graph, center_node, hops, min_nodes, max_nodes)


def sample_subgraph(G, target_size='Tiny', hops=3, nodes=None):
    """
    Extract an n-hop completely induced subgraph centered on a node from G.

    Args:
        G: Source graph
        target_size: Target size ('Tiny', 'Small', 'Medium')
        hops: Number of hops

    Returns:
        H: Extracted subgraph
        center: Center node used for extraction
    """
    center_node = random.choice(nodes) if nodes else None
    H, center, _ = extract_n_hop_ego_subgraph(
        G,
        center_node=center_node,
        hops=hops,
        min_nodes=0,
        max_nodes=float('inf'),
    )
    return H, center


def sample_subgraph_limited(G, target_size='Tiny', hops=3, nodes=None, max_nodes=None):
    center_node = random.choice(nodes) if nodes else random.choice(list(G.nodes()))
    sampled = {center_node}
    queue = deque([(center_node, 0)])

    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        for neighbor in G.neighbors(node):
            if neighbor in sampled:
                continue
            sampled.add(neighbor)
            if max_nodes is not None and len(sampled) > max_nodes:
                return None, center_node
            queue.append((neighbor, depth + 1))

    return G.subgraph(sampled).copy(), center_node


def generate_prompt_answer(task_name, graph, raw_texts=None, center_node=None, binary_cate=None, sample_num=10):
    assert task_name in FULL_TASKS, f"Invalid task name: {task_name}"

    # Balance yes/no sample counts for binary classification tasks.
    if binary_cate is None:
        binary_cate = {True: 0, False: 0}

    # Use the provided subgraph directly.
    working_graph = graph
    working_nodes = list(working_graph.nodes())

    # Pass raw_texts to the encoder for node text tasks.
    needs_text = task_name in ['node_text_description', 'node_id_from_text']
    subgraph_encoder = GraphTextEncoder(working_graph, raw_texts=raw_texts if needs_text else None)
    graph_description = subgraph_encoder.generate_graph_description(include_texts=needs_text)
    directed = subgraph_encoder.is_directed

    # Use the provided center node; fall back to random selection for old call sites.
    if center_node is None:
        center_node = random.choice(working_nodes)

    # =========================================================================
    # edge_existence
    # =========================================================================
    if task_name == 'edge_existence':
        if len(working_nodes) < 2:
            return None, None, None, None
        sample_nodes = random.sample(working_nodes, 2)
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
        sample_node = random.choice(working_nodes)
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
        undirected_nodes = list(G_und.nodes())
        if len(undirected_nodes) < 2:
            return None, None, None, None
        sample_nodes = random.sample(undirected_nodes, 2)
        common_neighbor = list(nx.common_neighbors(G_und, sample_nodes[0], sample_nodes[1]))
        if len(common_neighbor) == 0:
            return None, None, None, None
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
        sample_node = random.choice(working_nodes)
        neighbor = list(working_graph.neighbors(sample_node))
        if len(neighbor) == 0:
            return None, None, None, None
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
        answer = f"{len(working_nodes)}"
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
            return None, None, None, None
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
            return None, None, None, None
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
            return None, None, None, None
        binary_cate[is_bipartite] += 1
        answer = f"{'Yes' if is_bipartite else 'No'}"
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
            return None, None, None, None
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
            return None, None, None, None
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
            return None, None, None, None
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
            return None, None, None, None
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
            return None, None, None, None
        binary_cate[is_eulerian] += 1
        answer = f"{'Yes' if is_eulerian else 'No'}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node)
            + FORMATS[task_name]
        )
        sample = f"{center_node}"

    # elif task_name == 'topological_sort' and directed:
    #     try:
    #         nx.find_cycle(working_graph)
    #         return None, None, None, None
    #     except nx.NetworkXUnfeasible:
    #         topological_sort = list(nx.topological_sort(working_graph))
    #         assert is_valid_topological_sort(working_graph, topological_sort)
    #         topological_sort = [int(u) for u in topological_sort]
    #         answer = f"{topological_sort}"
    #         prompt = (
    #             INSTRUCTIONS[task_name]
    #             + graph_description
    #             + QUESTIONS[task_name].format(center_node)
    #             + FORMATS[task_name]
    #         )
    #         sample = f"{center_node}"

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
            return None, None, None, None
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
            return None, None, None, None
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
            return None, None, None, None
        min_edge_covering = nx.min_edge_cover(G_und)
        if not is_minimum_edge_cover(G_und, list(min_edge_covering)):
            return None, None, None, None
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
        if not is_minimum_spanning_tree(G_und, mst):
            return None, None, None, None
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
        if not is_valid_dominating_set(working_graph, dominating_set):
            return None, None, None, None
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
        if not is_minimum_vertex_cover(G_und, minimal_vertex_cover):
            return None, None, None, None
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
        if not is_maximal_weight_matching(G_und, max_weight_matching):
            return None, None, None, None
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
            return None, None, None, None
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
        sample_node = random.choice(working_nodes)
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
        sample_node = random.choice(working_nodes)
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
            return None, None, None, None
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
            return None, None, None, None
        effective_size = dict(filter(lambda x: not math.isnan(x[1]), effective_size.items()))
        if len(effective_size) == 0:
            return None, None, None, None
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
        sample_node = random.choice(working_nodes)
        answer = f"{nx.degree_centrality(working_graph)[sample_node]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'closeness_centrality':
        sample_node = random.choice(working_nodes)
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
        sample_node = random.choice(working_nodes)
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
        sample_node = random.choice(working_nodes)
        answer = f"{harmonic_centrality[sample_node]}"
        prompt = (
            INSTRUCTIONS[task_name]
            + graph_description
            + QUESTIONS[task_name].format(center_node, sample_node)
            + FORMATS[task_name]
        )
        sample = f"{sample_node}"

    elif task_name == 'avg_neighbor_degree':
        sample_node = random.choice(working_nodes)
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
            return None, None, None, None
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
        if len(working_nodes) < 2:
            return None, None, None, None
        sample_nodes = random.sample(working_nodes, 2)
        local_connectivity = nx.has_path(working_graph, sample_nodes[0], sample_nodes[1])
        if binary_cate[local_connectivity] > sample_num // 2:
            return None, None, None, None
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
        if not is_valid_shortest_path(working_graph, shortest_path, sample_nodes[0], sample_nodes[1]):
            return None, None, None, None
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
        sample_node = random.choice(working_nodes)
        bfs = list(nx.bfs_edges(working_graph, source=sample_node, sort_neighbors=sorted))
        if not is_valid_bfs_sequence(working_graph, bfs, sample_node):
            return None, None, None, None
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
        sample_node = random.choice(working_nodes)
        dfs = list(nx.dfs_edges(working_graph, source=sample_node))
        if not is_valid_dfs_sequence(working_graph, dfs, sample_node):
            return None, None, None, None
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
            return None, None, None, None
        sample_nodes = random.sample(working_nodes, 2)
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
            return None, None, None, None
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
            return None, None, None, None
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
            return None, None, None, None
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
            return None, None, None, None
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
            return None, None, None, None
        sample_node = random.choice(working_nodes)
        # Get the text description for the sample node
        if sample_node < len(raw_texts):
            text_description = raw_texts[sample_node]
        else:
            return None, None, None, None
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
            return None, None, None, None
        # Randomly select a node that has a valid text description
        valid_nodes = [n for n in working_graph.nodes() if n < len(raw_texts)]
        if len(valid_nodes) == 0:
            return None, None, None, None
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

    return prompt, answer, sample, working_graph, binary_cate



def generate_graph_tasks(
    task_name,
    sample_num,
    graph_size,
    verbose=False,
    single_graph_path=None,
    original_graph=None,
    loaded_graph_data=None,
    loaded_graph_list=None,
    samples_per_graph=5,
):
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
    binary_cate = {True: 0, False: 0}
    binary_cate[True] = 0
    binary_cate[False] = 0

    # Load original_graph when provided; otherwise load single_graph.
    if loaded_graph_list is not None:
        for graph_item in loaded_graph_list:
            graph_idx = graph_item['graph_id']
            G = graph_item['graph']
            raw_texts = graph_item['raw_texts']
            graph_nodes = graph_item['nodes']
            graph_answers_before = len(task_data['answer'])
            attempts = 0
            max_attempts = max(20, samples_per_graph * 20)

            while len(task_data['answer']) - graph_answers_before < samples_per_graph and attempts < max_attempts:
                attempts += 1
                subgraph, center_node = sample_subgraph(G, graph_size, nodes=graph_nodes)
                result = generate_prompt_answer(task_name, subgraph, raw_texts, center_node, binary_cate, sample_num)
                if len(result) == 4:
                    prompt, answer, sample, _ = result
                else:
                    prompt, answer, sample, _, binary_cate = result
                if prompt is None or answer is None:
                    continue

                subgraph_nodes = sorted(subgraph.nodes())
                subgraph_edges = list(subgraph.edges())
                is_directed = subgraph.is_directed()

                task_data['task'].append(task_name)
                task_data['prompt'].append(f"Graph ID: {graph_idx}\n" + prompt)
                task_data['answer'].append(answer)
                task_data['source_file'].append({'graph_list': single_graph_path, 'graph_id': graph_idx})
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

        if len(task_data['answer']) == 0:
            print(f"No valid samples found for task '{task_name}' from graph list '{single_graph_path}'. Skipping.")

    elif loaded_graph_data is not None or (original_graph if original_graph else single_graph_path) is not None:
        graph_to_load = original_graph if original_graph else single_graph_path
        if loaded_graph_data is None:
            edges, directed, weighted, raw_texts = load_graph_data(graph_to_load)
            G = npz_to_nx(edges, directed, weighted)
        else:
            G = loaded_graph_data['graph']
            directed = loaded_graph_data['directed']
            weighted = loaded_graph_data['weighted']
            raw_texts = loaded_graph_data['raw_texts']
            graph_to_load = loaded_graph_data.get('source_path', graph_to_load)
        graph_nodes = list(G.nodes())
        max_attempts = max(100, sample_num * 200)
        stagnant_attempts = 0
        max_stagnant_attempts = max(100, sample_num * 20)
        for _ in range(max_attempts):
            subgraph, center_node = sample_subgraph(G, graph_size, nodes=graph_nodes)
            result = generate_prompt_answer(task_name, subgraph, raw_texts, center_node, binary_cate, sample_num)
            if len(result) == 4:
                prompt, answer, sample, _ = result
            else:
                prompt, answer, sample, _, binary_cate = result
            if prompt is None or answer is None:
                stagnant_attempts += 1
                if stagnant_attempts >= max_stagnant_attempts:
                    break
                continue
            stagnant_attempts = 0

            subgraph_nodes = sorted(subgraph.nodes())
            subgraph_edges = list(subgraph.edges())
            is_directed = subgraph.is_directed()

            task_data['task'].append(task_name)
            task_data['prompt'].append(prompt)
            task_data['answer'].append(answer)
            task_data['source_file'].append({'original': original_graph, 'single': single_graph_path})
            task_data['direction'].append('directed' if is_directed else 'undirected')
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
            task_data['graph_id'].append(-1)

            if len(task_data['answer']) >= sample_num:
                break

        if len(task_data['answer']) == 0:
            print(f"No valid subgraphs found for task '{task_name}' on graph '{graph_to_load}' after {max_attempts} attempts. Skipping.")
    else:
        dataset_paths = get_dataset_paths(graph_size)
        print(f"Total number of datasets: {len(dataset_paths)}")

        index = 0
        for dataset_path in tqdm(dataset_paths):
            edges, directed, weighted, raw_texts = load_graph_data(file_path=dataset_path)
            G = npz_to_nx(edges, directed, weighted)
            if len(G.nodes()) <= 2 or len(G.edges()) <= 2:
                continue
            if weighted:
                continue
            subgraph, center_node = sample_subgraph(G, graph_size)
            result = generate_prompt_answer(task_name, subgraph, raw_texts, center_node, binary_cate, sample_num)
            if len(result) == 4:
                prompt, answer, sample, _ = result
            else:
                prompt, answer, sample, _, binary_cate = result
            if prompt is None or answer is None:
                continue
            
            subgraph_nodes = sorted(subgraph.nodes())
            subgraph_edges = list(subgraph.edges())

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
            task_data['graph_id'].append(index)

            if len(task_data['answer']) >= sample_num:
                print(f"Processed {len(task_data['answer'])} tasks, stopping.")
                if task_name in BINARY_TASKS:
                    print(f"Binary cate: {binary_cate}")
                break

    return task_data


def generate_mixed_graph_list_tasks(task_list, graph_size, loaded_graph_list, single_graph_path=None, samples_per_graph=5):
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

    for graph_item in loaded_graph_list:
        graph_idx = graph_item['graph_id']
        G = graph_item['graph']
        raw_texts = graph_item['raw_texts']
        graph_nodes = graph_item['nodes']
        binary_cate = {True: 0, False: 0}

        shuffled_tasks = task_list.copy()
        random.shuffle(shuffled_tasks)
        graph_answers_before = len(task_data['answer'])
        task_cursor = 0

        while len(task_data['answer']) - graph_answers_before < samples_per_graph and task_cursor < len(shuffled_tasks):
            task_name = shuffled_tasks[task_cursor]
            task_cursor += 1
            max_attempts = 50
            for _ in range(max_attempts):
                subgraph, center_node = sample_subgraph(G, graph_size, nodes=graph_nodes)
                result = generate_prompt_answer(task_name, subgraph, raw_texts, center_node, binary_cate, samples_per_graph)
                if len(result) == 4:
                    prompt, answer, sample, _ = result
                else:
                    prompt, answer, sample, _, binary_cate = result
                if prompt is None or answer is None:
                    continue

                subgraph_nodes = sorted(subgraph.nodes())
                subgraph_edges = list(subgraph.edges())
                is_directed = subgraph.is_directed()

                task_data['task'].append(task_name)
                task_data['prompt'].append(prompt)
                task_data['answer'].append(answer)
                task_data['source_file'].append({'graph_list': single_graph_path, 'graph_id': graph_idx})
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
                break

    return task_data


def empty_task_data():
    return {
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


def append_task_record(task_data, task_name, prompt, answer, sample, subgraph, single_graph_path, graph_idx):
    subgraph_nodes = sorted(subgraph.nodes())
    subgraph_edges = list(subgraph.edges())
    is_directed = subgraph.is_directed()
    is_graph_list_item = graph_idx is not None and graph_idx >= 0

    task_data['task'].append(task_name)
    task_data['prompt'].append((f"Graph ID: {graph_idx}\n" if is_graph_list_item else "") + prompt)
    task_data['answer'].append(answer)
    task_data['source_file'].append(
        {'graph_list': single_graph_path, 'graph_id': graph_idx}
        if is_graph_list_item else {'single': single_graph_path, 'graph_id': graph_idx}
    )
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


def get_available_category_tasks(question_categories, has_text=True, skip_slow_tasks=False):
    available_tasks = {}
    required_maps = (TOOLSETS, ANSWER_TYPES, INSTRUCTIONS, QUESTIONS, FORMATS)
    for category, subcats in question_categories.items():
        available_tasks[category] = []
        seen = set()
        for subcat, tasks in subcats.items():
            for task_name in tasks:
                if task_name in seen:
                    continue
                if task_name not in FULL_TASKS:
                    continue
                if any(task_name not in required_map for required_map in required_maps):
                    continue
                if not has_text and task_name in TEXT_TASKS:
                    continue
                if skip_slow_tasks and task_name in SLOW_EXACT_TASKS:
                    continue
                available_tasks[category].append((subcat, task_name))
                seen.add(task_name)
    return available_tasks


def distribute_task_quota(task_items, total_count):
    if not task_items or total_count <= 0:
        return {}
    base_count, remainder = divmod(total_count, len(task_items))
    return {
        item: base_count + (1 if idx < remainder else 0)
        for idx, item in enumerate(task_items)
    }


def subgraph_node_count(task_data):
    if not task_data['subgraph_nodes']:
        return 0
    return len(task_data['subgraph_nodes'][-1])


def pick_available_task(task_items, task_counts, blocked_tasks):
    candidates = [item for item in task_items if item not in blocked_tasks]
    if not candidates:
        return None
    min_count = min(task_counts.get(item, 0) for item in candidates)
    least_used = [item for item in candidates if task_counts.get(item, 0) == min_count]
    return random.choice(least_used)


def try_append_categorized_task(
    task_data,
    task_name,
    graph_size,
    graph_item,
    single_graph_path,
    binary_cate,
    task_sample_num,
    max_attempts=10,
    max_prompt_nodes=None,
):
    graph_idx = graph_item['graph_id']
    G = graph_item['graph']
    raw_texts = graph_item['raw_texts']
    graph_nodes = graph_item['nodes']
    oversized_count = 0

    for _ in range(max_attempts):
        subgraph, center_node = sample_subgraph_limited(
            G,
            graph_size,
            nodes=graph_nodes,
            max_nodes=max_prompt_nodes,
        )
        if subgraph is None:
            oversized_count += 1
            continue
        result = generate_prompt_answer(task_name, subgraph, raw_texts, center_node, binary_cate, task_sample_num)
        if len(result) == 4:
            prompt, answer, sample, _ = result
        else:
            prompt, answer, sample, _, binary_cate = result
        if prompt is None or answer is None:
            continue

        append_task_record(task_data, task_name, prompt, answer, sample, subgraph, single_graph_path, graph_idx)
        return True, binary_cate

    if oversized_count == max_attempts and max_prompt_nodes is not None:
        print(f"Skip {task_name}: all {max_attempts} sampled centers exceeded {max_prompt_nodes} nodes")
    return False, binary_cate


def generate_categorized_quota_tasks(
    question_categories,
    graph_size,
    loaded_graph_items,
    single_graph_path=None,
    target_per_category=200,
    samples_per_graph=None,
    max_attempts_per_task=10,
    max_prompt_nodes=None,
    skip_slow_tasks=False,
    on_category_done=None,
):
    categorized_data = {
        category: {subcat: empty_task_data() for subcat in subcats}
        for category, subcats in question_categories.items()
    }
    category_counts = {category: 0 for category in question_categories}
    category_task_counts = {category: {} for category in question_categories}
    category_task_targets = {category: {} for category in question_categories}
    has_text = any(item.get('raw_texts') is not None for item in loaded_graph_items)
    available_category_tasks = get_available_category_tasks(
        question_categories,
        has_text=has_text,
        skip_slow_tasks=skip_slow_tasks,
    )
    if not loaded_graph_items:
        return categorized_data, category_counts, category_task_targets, category_task_counts

    for category, subcats in question_categories.items():
        flat_tasks = available_category_tasks.get(category, [])
        if not flat_tasks:
            continue

        target_count = target_per_category.get(category, 0) if isinstance(target_per_category, dict) else target_per_category
        task_targets = distribute_task_quota(flat_tasks, target_count)
        task_item_counts = {item: 0 for item in flat_tasks}
        task_failures = {item: 0 for item in flat_tasks}
        blocked_tasks = set()
        max_task_failures = max(3, len(loaded_graph_items))
        binary_cates = {task_name: {True: 0, False: 0} for _, task_name in flat_tasks}
        category_task_targets[category] = {task_name: quota for (_, task_name), quota in task_targets.items()}
        category_task_counts[category] = {task_name: 0 for _, task_name in flat_tasks}

        for graph_item in loaded_graph_items:
            graph_target = samples_per_graph if samples_per_graph is not None else target_count
            graph_count = 0

            while (
                graph_count < graph_target
                and category_counts[category] < target_count
                and len(blocked_tasks) < len(flat_tasks)
            ):
                task_item = pick_available_task(flat_tasks, task_item_counts, blocked_tasks)
                if task_item is None:
                    break

                subcat, task_name = task_item
                task_data = categorized_data[category][subcat]
                task_sample_num = max(1, task_targets[task_item])
                success, binary_cates[task_name] = try_append_categorized_task(
                    task_data,
                    task_name,
                    graph_size,
                    graph_item,
                    single_graph_path,
                    binary_cates[task_name],
                    task_sample_num,
                    max_attempts=max_attempts_per_task,
                    max_prompt_nodes=max_prompt_nodes,
                )
                if success:
                    task_item_counts[task_item] += 1
                    category_counts[category] += 1
                    category_task_counts[category][task_name] += 1
                    graph_count += 1
                    if category_counts[category] == 1 or category_counts[category] % 10 == 0:
                        print(f"{category}: {category_counts[category]}/{target_count} latest={task_name} nodes={subgraph_node_count(task_data)}")
                    continue

                task_failures[task_item] += 1
                if task_failures[task_item] >= max_task_failures:
                    blocked_tasks.add(task_item)

        fill_attempts = 0
        max_fill_attempts = max(100, target_count * 50)
        while (
            category_counts[category] < target_count
            and len(blocked_tasks) < len(flat_tasks)
            and fill_attempts < max_fill_attempts
        ):
            fill_attempts += 1
            graph_item = random.choice(loaded_graph_items)
            task_item = pick_available_task(flat_tasks, task_item_counts, blocked_tasks)
            if task_item is None:
                break

            subcat, task_name = task_item
            task_data = categorized_data[category][subcat]
            task_sample_num = max(1, task_targets[task_item])
            success, binary_cates[task_name] = try_append_categorized_task(
                task_data,
                task_name,
                graph_size,
                graph_item,
                single_graph_path,
                binary_cates[task_name],
                task_sample_num,
                max_attempts=max_attempts_per_task,
                max_prompt_nodes=max_prompt_nodes,
            )
            if success:
                task_item_counts[task_item] += 1
                category_counts[category] += 1
                category_task_counts[category][task_name] += 1
                if category_counts[category] == 1 or category_counts[category] % 10 == 0:
                    print(f"{category}: {category_counts[category]}/{target_count} latest={task_name} nodes={subgraph_node_count(task_data)}")
                continue

            task_failures[task_item] += 1
            if task_failures[task_item] >= max_task_failures:
                blocked_tasks.add(task_item)

        if category_counts[category] < target_count:
            print(f"Warning: {category} generated {category_counts[category]}/{target_count}; blocked tasks: {[task for _, task in blocked_tasks]}")

        if on_category_done is not None:
            on_category_done(category, categorized_data[category])

    return categorized_data, category_counts, category_task_targets, category_task_counts


def generate_categorized_graph_list_tasks(question_categories, graph_size, loaded_graph_list, single_graph_path=None, samples_per_graph=5):
    target_per_category = len(loaded_graph_list) * samples_per_graph
    categorized_data, category_counts, _, _ = generate_categorized_quota_tasks(
        question_categories,
        graph_size,
        loaded_graph_list,
        single_graph_path=single_graph_path,
        target_per_category=target_per_category,
        samples_per_graph=samples_per_graph,
    )
    return categorized_data, category_counts



parser = argparse.ArgumentParser(description="Generate RL datasets")
parser.add_argument('--sample_num', type=int, default=200, help='Number of samples to generate per task')
parser.add_argument('--graph_size', type=str, default='Tiny', help='Size of the graph: Tiny, Small, Medium, Large')
parser.add_argument('--single_graph', type=str, default=None, help='Path to a single graph npz file, e.g. xxx')
parser.add_argument('--original_graph', type=str, default=None, help='Path to the original graph npz file for reference')
parser.add_argument('--parent_path', type=str, default=None, help='Output parent directory (default: derived from single_graph basename)')
parser.add_argument('--graphs', type=str, nargs='+', default=None, help='Process multiple graphs sequentially, e.g. --graphs NBA.pt pokec_n.pt pokec_z.pt pubmed.pt reddit.pt')
parser.add_argument('--graph_list_size', type=int, default=40, help='Number of graphs to select when --single_graph points to a .pt graph list')
parser.add_argument('--samples_per_graph', type=int, default=5, help='Number of samples to generate per selected graph in graph-list mode')
parser.add_argument('--max_attempts_per_task', type=int, default=10, help='Maximum retries for one task sample before trying another task')
parser.add_argument('--max_subgraph_nodes', type=int, default=1000, help='Resample center node if the extracted 3-hop subgraph has more nodes than this')
parser.add_argument('--include_slow_tasks', action='store_true', help='Include exact graph-theory tasks that can be very slow on large graphs')
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])

from questions import QUESTION_CATEGORIES


def process_single_graph(single_graph_arg, args):
    """Process one graph and derive parent_path and suffix automatically."""
    sample_num = args.sample_num
    graph_size = args.graph_size
    single_graph_path = single_graph_arg
    original_graph_path = args.original_graph

    # Derive parent_path from the single_graph file name.
    if args.parent_path:
        parent_path = args.parent_path
    else:
        basename = os.path.splitext(os.path.basename(single_graph_path))[0]
        parent_path = basename

    os.makedirs(parent_path, exist_ok=True)
    suffix = f"single_{os.path.splitext(os.path.basename(single_graph_path))[0]}" if single_graph_path else graph_size

    # Create the category directory structure.
    graph_info_dir = os.path.join(parent_path, 'graph_info_retrieval')
    graph_theory_dir = os.path.join(parent_path, 'graph_theory')
    for category_dir in [graph_info_dir, graph_theory_dir]:
        os.makedirs(os.path.join(category_dir, 'node'), exist_ok=True)
        os.makedirs(os.path.join(category_dir, 'edge'), exist_ok=True)
        os.makedirs(os.path.join(category_dir, 'graph'), exist_ok=True)

    binary_cate = {True: 0, False: 0}
    print(f"\n{'='*60}")
    print(f"Processing: {single_graph_path}")
    print(f"Output dir: {parent_path}")
    print(f"{'='*60}")
    print(f"Total number of tasks: {len(FULL_TASKS)}")
    print(f"Sample number per task: {sample_num}")

    graph_to_load = original_graph_path if original_graph_path else single_graph_path
    loaded_graph_data = None
    loaded_graph_list = None
    if graph_to_load is not None:
        if original_graph_path is None and graph_to_load.endswith('.pt'):
            try:
                graph_list = load_graph_list(graph_to_load)
                if graph_list:
                    loaded_graph_list = []
                    candidate_indices = list(range(len(graph_list)))
                    random.shuffle(candidate_indices)
                    for graph_idx in candidate_indices:
                        edges, directed, weighted, raw_texts = graph_list[graph_idx]
                        G = npz_to_nx(edges, directed, weighted)
                        if G.number_of_nodes() <= 2 or G.number_of_edges() <= 2:
                            continue
                        if weighted:
                            continue
                        loaded_graph_list.append({
                            'graph_id': graph_idx,
                            'graph': G,
                            'directed': directed,
                            'weighted': weighted,
                            'raw_texts': raw_texts,
                            'nodes': list(G.nodes()),
                        })
                        if len(loaded_graph_list) >= args.graph_list_size:
                            break
                    print(f"Loaded graph list once: {graph_to_load}")
                    print(f"Selected {len(loaded_graph_list)} valid graphs from {len(graph_list)} total graphs")
                    print(f"Graph-list target: 2 categories x ({len(loaded_graph_list)} graphs x {args.samples_per_graph} random tasks) = {2 * len(loaded_graph_list) * args.samples_per_graph} samples")
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                print(f"Graph-list loading skipped ({exc}); falling back to single-graph mode.")

        if loaded_graph_list is None:
            print(f"Loading graph once: {graph_to_load}")
            edges, directed, weighted, raw_texts = load_graph_data(graph_to_load)
            loaded_graph_data = {
                'graph': npz_to_nx(edges, directed, weighted),
                'directed': directed,
                'weighted': weighted,
                'raw_texts': raw_texts,
                'source_path': graph_to_load,
            }

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

    def task_output_path(task_name, task_dir, suffix, count):
        filename = f"{task_name}_{suffix}_{count}.csv"
        return os.path.join(task_dir, filename) if task_dir else os.path.join(parent_path, filename)

    if loaded_graph_list is not None:
        target_per_category = len(loaded_graph_list) * args.samples_per_graph
        print(f"Generating categorized graph-list tasks: 2 categories x {target_per_category} samples")
        categorized_data, category_counts = generate_categorized_graph_list_tasks(
            QUESTION_CATEGORIES,
            graph_size,
            loaded_graph_list,
            single_graph_path=single_graph_path,
            samples_per_graph=args.samples_per_graph,
        )
        for category, subcats in categorized_data.items():
            for subcat, task_data in subcats.items():
                if not task_data['answer']:
                    continue
                base_dir = graph_info_dir if category == 'graph_info_retrieval' else graph_theory_dir
                save_path = task_output_path(f'{category}_{subcat}', os.path.join(base_dir, subcat), suffix, len(task_data['answer']))
                save_datasets(task_data, save_path=save_path)

        print(f"\n=== Summary for {single_graph_path} ===")
        for category, count in category_counts.items():
            print(f"{category}: {count}/{target_per_category} samples")
        return

    for task in FULL_TASKS:
        print(f"Generating tasks for: {task}")
        task_dir = get_task_dir(task)

        if loaded_graph_list is not None:
            adjusted_sample_num = len(loaded_graph_list) * args.samples_per_graph
        elif task_dir and 'graph_info_retrieval' in task_dir:
            adjusted_sample_num = sample_num * num_theory_tasks // num_info_tasks
        else:
            adjusted_sample_num = sample_num

        preview_path = task_output_path(task, task_dir, suffix, adjusted_sample_num)
        if os.path.exists(preview_path):
            print(f"Skip existing task file: {preview_path}")
            continue

        task_data = generate_graph_tasks(
            task,
            sample_num=adjusted_sample_num,
            graph_size=graph_size,
            verbose=False,
            single_graph_path=single_graph_path,
            original_graph=original_graph_path,
            loaded_graph_data=loaded_graph_data,
            loaded_graph_list=loaded_graph_list,
            samples_per_graph=args.samples_per_graph,
        )
        save_path = task_output_path(task, task_dir, suffix, len(task_data['answer']))
        save_datasets(task_data, save_path=save_path)

    if loaded_graph_list is not None:
        graph_list_sample_num = len(loaded_graph_list) * args.samples_per_graph
        print(f"\n=== Summary for {single_graph_path} ===")
        print(f"graph-list mode: {len(loaded_graph_list)} graphs x {args.samples_per_graph} samples = {graph_list_sample_num} samples per task")
        print(f"total target: {len(FULL_TASKS)} tasks x {graph_list_sample_num} = {len(FULL_TASKS) * graph_list_sample_num} samples")
        return

    print(f"\n=== Summary for {single_graph_path} ===")
    print(f"graph_info_retrieval: {num_info_tasks} tasks × {sample_num * num_theory_tasks // num_info_tasks} = {num_info_tasks * sample_num * num_theory_tasks // num_info_tasks} samples")
    print(f"graph_theory: {num_theory_tasks} tasks × {sample_num} = {num_theory_tasks * sample_num} samples")


# Decide which graphs to process.
if __name__ == "__main__" and args.graphs:
    raise ValueError("This version only supports single-graph processing via --single_graph. Please run multiple times for multiple graphs.")

def process_graph_fast(single_graph_arg, args):
    sample_num = args.sample_num
    graph_size = args.graph_size
    single_graph_path = single_graph_arg
    original_graph_path = args.original_graph

    if args.parent_path:
        parent_path = args.parent_path
    else:
        parent_path = os.path.splitext(os.path.basename(single_graph_path))[0]

    os.makedirs(parent_path, exist_ok=True)
    suffix = f"single_{os.path.splitext(os.path.basename(single_graph_path))[0]}" if single_graph_path else graph_size

    graph_info_dir = os.path.join(parent_path, 'graph_info_retrieval')
    graph_theory_dir = os.path.join(parent_path, 'graph_theory')
    for category_dir in (graph_info_dir, graph_theory_dir):
        os.makedirs(os.path.join(category_dir, 'node'), exist_ok=True)
        os.makedirs(os.path.join(category_dir, 'edge'), exist_ok=True)
        os.makedirs(os.path.join(category_dir, 'graph'), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {single_graph_path}")
    print(f"Output dir: {parent_path}")
    print(f"{'='*60}")

    graph_to_load = original_graph_path if original_graph_path else single_graph_path
    graph_items = None
    is_graph_list_mode = False

    if graph_to_load and original_graph_path is None and graph_to_load.endswith('.pt'):
        try:
            graph_list = load_graph_list(graph_to_load)
            if graph_list:
                graph_items = []
                candidate_indices = list(range(len(graph_list)))
                random.shuffle(candidate_indices)
                for graph_idx in candidate_indices:
                    edges, directed, weighted, raw_texts = graph_list[graph_idx]
                    G = npz_to_nx(edges, directed, weighted)
                    if G.number_of_nodes() <= 2 or G.number_of_edges() <= 2:
                        continue
                    if weighted:
                        continue
                    graph_items.append({
                        'graph_id': graph_idx,
                        'graph': G,
                        'directed': directed,
                        'weighted': weighted,
                        'raw_texts': raw_texts,
                        'nodes': list(G.nodes()),
                    })
                    if len(graph_items) >= args.graph_list_size:
                        break
                is_graph_list_mode = True
                print(f"Loaded graph list once: {graph_to_load}")
                print(f"Selected {len(graph_items)} valid graphs from {len(graph_list)} total graphs")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            print(f"Graph-list loading skipped ({exc}); falling back to single-graph mode.")

    if graph_items is None:
        print(f"Loading graph once: {graph_to_load}")
        edges, directed, weighted, raw_texts = load_graph_data(graph_to_load)
        G = npz_to_nx(edges, directed, weighted)
        graph_items = [{
            'graph_id': -1,
            'graph': G,
            'directed': directed,
            'weighted': weighted,
            'raw_texts': raw_texts,
            'nodes': list(G.nodes()),
        }]

    if not graph_items:
        raise ValueError(f"No valid graph loaded from {graph_to_load}")

    skip_slow_tasks = not args.include_slow_tasks
    available_category_tasks = get_available_category_tasks(
        QUESTION_CATEGORIES,
        has_text=any(item.get('raw_texts') is not None for item in graph_items),
        skip_slow_tasks=skip_slow_tasks,
    )
    for category, tasks in available_category_tasks.items():
        print(f"{category} available tasks: {len(tasks)}")

    target_per_category = len(graph_items) * args.samples_per_graph if is_graph_list_mode else sample_num
    print(f"Generating categorized tasks: 2 categories x {target_per_category} samples")
    print(f"Resample center node when 3-hop subgraph nodes > {args.max_subgraph_nodes}")
    if skip_slow_tasks:
        print("Skipping slow exact graph-theory tasks by default. Use --include_slow_tasks to enable them.")

    saved_categories = set()

    def save_category_outputs(category, subcats):
        if category in saved_categories:
            return
        for subcat, task_data in subcats.items():
            if not task_data['answer']:
                continue
            base_dir = graph_info_dir if category == 'graph_info_retrieval' else graph_theory_dir
            task_dir = os.path.join(base_dir, subcat)
            filename = f"{category}_{subcat}_{suffix}_{len(task_data['answer'])}.csv"
            save_datasets(task_data, save_path=os.path.join(task_dir, filename))
        saved_categories.add(category)

    categorized_data, category_counts, category_task_targets, category_task_counts = generate_categorized_quota_tasks(
        QUESTION_CATEGORIES,
        graph_size,
        graph_items,
        single_graph_path=single_graph_path,
        target_per_category=target_per_category,
        samples_per_graph=args.samples_per_graph if is_graph_list_mode else None,
        max_attempts_per_task=args.max_attempts_per_task,
        max_prompt_nodes=args.max_subgraph_nodes,
        skip_slow_tasks=skip_slow_tasks,
        on_category_done=save_category_outputs,
    )

    for category, subcats in categorized_data.items():
        if category in saved_categories:
            continue
        for subcat, task_data in subcats.items():
            if not task_data['answer']:
                continue
            base_dir = graph_info_dir if category == 'graph_info_retrieval' else graph_theory_dir
            task_dir = os.path.join(base_dir, subcat)
            filename = f"{category}_{subcat}_{suffix}_{len(task_data['answer'])}.csv"
            save_datasets(task_data, save_path=os.path.join(task_dir, filename))

    print(f"\n=== Summary for {single_graph_path} ===")
    for category, count in category_counts.items():
        print(f"{category}: {count}/{target_per_category} samples")
        task_counts = category_task_counts.get(category, {})
        task_targets = category_task_targets.get(category, {})
        for task_name in sorted(task_targets):
            print(f"  {task_name}: {task_counts.get(task_name, 0)}/{task_targets[task_name]}")


if __name__ == "__main__":
    process_graph_fast(args.single_graph, args)

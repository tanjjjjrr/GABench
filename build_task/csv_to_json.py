import ast
import csv
csv.field_size_limit(10 * 1024 * 1024)  # 10MB field limit
import re
import json
from pathlib import Path


NOTE_RE = re.compile(r'\[Note:\s*(.*?)\]', re.DOTALL)
QUESTION_RE = re.compile(
    r'Question:\s*(Extract the 3-hop ego subgraph.*?)(?=\nYour answer|\Z)',
    re.DOTALL,
)
GENERIC_QUESTION_RE = re.compile(r'Question:\s*(.*?)(?=\nYour answer|\Z)', re.DOTALL)
GRAPH_ID_RE = re.compile(r'Graph ID:\s*([^\n]+)')
NODE_RANGE_RE = re.compile(r'\((\d+),\s*(\d+)\)')
EDGE_RE = re.compile(r'\((\d+),\s*(\d+)\)')
JSON_SEPARATORS = (',', ':')


def dumps_json_fields(value, indent=0):
    """Pretty-print dict fields on separate lines while keeping lists compact."""
    if isinstance(value, dict):
        if not value:
            return '{}'
        lines = ['{']
        items = list(value.items())
        for idx, (key, item_value) in enumerate(items):
            comma = ',' if idx < len(items) - 1 else ''
            formatted_value = dumps_json_fields(item_value, indent + 2)
            lines.append(
                f"{' ' * (indent + 2)}"
                f"{json.dumps(str(key), ensure_ascii=False)}: "
                f"{formatted_value}{comma}"
            )
        lines.append(f"{' ' * indent}" + '}')
        return '\n'.join(lines)
    return json.dumps(value, ensure_ascii=False, separators=JSON_SEPARATORS)


def extract_subgraph_info(row: dict) -> dict:
    """Extract subgraph info from CSV row - directly use nodes/edges columns"""
    result = {}
    
    # Use the nodes column from the CSV directly; it already contains the subgraph node list.
    nodes_data = row.get('nodes', '')
    if nodes_data:
        # nodes may be a list or a string representation.
        try:
            if isinstance(nodes_data, list):
                result['nodes'] = nodes_data
            else:
                # Try to parse the string representation.
                nodes_data = str(nodes_data)
                if nodes_data.startswith('['):
                    result['nodes'] = ast.literal_eval(nodes_data)
                else:
                    # Formats like "(56092, 3247090)" represent a node range.
                    match = re.match(r'\((\d+),\s*(\d+)\)', nodes_data)
                    if match:
                        start, end = int(match.group(1)), int(match.group(2))
                        result['nodes'] = list(range(start, end + 1))
        except:
            pass
    
    # Use the edges column from the CSV directly; it already contains the subgraph edge list.
    edges_data = row.get('edges', '')
    if edges_data:
        try:
            if isinstance(edges_data, list):
                result['edges'] = edges_data
            else:
                edges_str = str(edges_data)
                if edges_str.startswith('['):
                    result['edges'] = ast.literal_eval(edges_str)
                else:
                    edges = re.findall(r'\((\d+),\s*(\d+)\)', edges_str)
                    result['edges'] = [(int(u), int(v)) for u, v in edges]
        except:
            pass
    
    return result


def parse_literal_list(value):
    if isinstance(value, list):
        return value
    if not value:
        return None
    value_str = str(value).strip()
    if not value_str.startswith('['):
        return None
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return None


def extract_subgraph_info_fast(row: dict) -> dict:
    result = {}

    nodes_data = row.get('nodes', '')
    nodes = parse_literal_list(nodes_data)
    if nodes is not None:
        result['nodes'] = nodes
    elif nodes_data:
        match = NODE_RANGE_RE.match(str(nodes_data).strip())
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            result['nodes'] = list(range(start, end + 1))

    edges_data = row.get('edges', '')
    edges = parse_literal_list(edges_data)
    if edges is not None:
        result['edges'] = edges
    elif edges_data:
        result['edges'] = [(int(u), int(v)) for u, v in EDGE_RE.findall(str(edges_data))]

    return result


# Tasks that require subgraph extraction and therefore include subgraph in ground_truth.
# Almost every task using extract_subgraph should include subgraph metadata.
SUBGRAPH_TASKS = {
    'node_number', 'edge_number', 'edge_existence', 'degree', 'common_neighbor',
    'neighbor', 'shortest_path', 'bfs', 'dfs', 'triangles',
    'density', 'topological_sort', 'has_cycle',
    'closeness_centrality', 'betweenness_centrality', 'harmonic_centrality',
    'degree_centrality', 'avg_neighbor_degree', 'local_connectivity', 'isolate_number',
    'wiener_index', 'barycenter', 'center', 'diameter', 'periphery', 'radius',
    'global_efficiency', 'transitivity', 'clustering_coefficient',
    'dominating_set', 'maximal_independent_set',
    'min_edge_covering', 'min_vertex_cover', 'max_weight_matching', 'min_weight_matching',
    'connected_component_number', 'strongly_connected_number', 'weakly_connected_number',
    'is_tournament', 'is_bipartite', 'is_eulerian', 'is_regular',
    'is_connected', 'is_weakly_connected', 'is_strongly_connected',
    'bridges', 'minimum_spanning_tree', 'weighted_minimum_spanning_tree',
    'maximum_flow',
    'resource_allocation_index', 'jaccard_coefficient', 'adamic_adar_index',
    'constraint', 'effective_size', 'degree_assortativity', 'degree_connectivity',
    'k_core', 'maximal_cliques',
    'weighted_shortest_path', 'ego_subgraph_3hop',
}


def extract_question(prompt: str) -> str:
    graph_id_match = GRAPH_ID_RE.search(prompt)
    note_match = NOTE_RE.search(prompt)
    question_matches = list(QUESTION_RE.finditer(prompt))
    if not question_matches:
        question_matches = list(GENERIC_QUESTION_RE.finditer(prompt))
    
    parts = []
    if graph_id_match:
        parts.append(f"Graph ID: {graph_id_match.group(1).strip()}")
    if note_match:
        parts.append(f"[Note: {note_match.group(1).strip()}]")
    if question_matches:
        question_match = question_matches[-1]
        parts.append(question_match.group(1).rstrip())
    
    return ' '.join(parts) if parts else prompt


def task_to_tools(task_name: str):
    mapping = {
        'node_number': ['extract_subgraph', 'len(graph.nodes())'],
        'edge_number': ['extract_subgraph', 'len(graph.edges())'],
        'edge_existence': ['extract_subgraph', 'graph.has_edge'],
        'degree': ['extract_subgraph', 'graph.degree'],
        'neighbor': ['extract_subgraph', 'graph.neighbors'],
        'common_neighbor': ['extract_subgraph', 'nx.common_neighbors'],
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
        'weighted_minimum_spanning_tree': ['extract_subgraph', 'nx.minimum_spanning_edges'],
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
        'local_connectivity': ['extract_subgraph', 'nx.has_path'],
        'ego_subgraph_3hop': ['extract_subgraph'],
    }
    return mapping.get(task_name, [task_name])


def csvs_to_json_legacy(task_output_dir: str, output_dir: str):
    return csvs_to_json(task_output_dir, output_dir)


def csvs_to_json(task_output_dir: str, output_dir: str):
    task_output = Path(task_output_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    category_map = {
        'graph_info_retrieval': ['node', 'edge', 'graph'],
        'graph_theory': ['node', 'edge', 'graph'],
    }

    for category, subcats in category_map.items():
        category_path = task_output / category
        if not category_path.exists():
            print(f"[skip] {category}/ not found")
            continue

        for subcat in subcats:
            subdir_path = category_path / subcat
            if not subdir_path.exists():
                print(f"[skip] {category}/{subcat}/ not found")
                continue

            csv_files = sorted(subdir_path.glob('*.csv'))
            if not csv_files:
                continue

            out_file = output_path / category / f"{subcat}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            record_count = 0
            with out_file.open('w', encoding='utf-8', newline='') as out_handle:
                out_handle.write('[')
                for csv_file in csv_files:
                    with csv_file.open('r', encoding='utf-8-sig', newline='') as csv_handle:
                        reader = csv.DictReader(csv_handle)
                        for row in reader:
                            task_name = str(row.get('task', ''))
                            tool_select = parse_literal_list(row.get('tool_select', '')) or task_to_tools(task_name)
                            ground_truth = {'answer': row.get('answer', '')}
                            if task_name in SUBGRAPH_TASKS:
                                ground_truth['subgraph'] = extract_subgraph_info_fast(row)

                            record = {
                                'id': f"task_{record_count + 1:04d}",
                                'question': extract_question(row.get('prompt', '')),
                                'tool_select': tool_select,
                                'ground_truth': ground_truth,
                            }
                            if record_count:
                                out_handle.write(',')
                            out_handle.write('\n  ')
                            out_handle.write(dumps_json_fields(record, indent=2))
                            record_count += 1
                if record_count:
                    out_handle.write('\n')
                out_handle.write(']')

            print(f"{category}/{subcat}/ -> {out_file}  ({record_count} records)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert task CSVs to grouped JSON files')
    parser.add_argument('--input', type=str, default='build_benchmark/build_task/task_output',
                        help='Directory containing node_tasks/, edge_tasks/, graph_tasks/')
    parser.add_argument('--output', type=str, default='build_benchmark/build_task/task_output',
                        help='Output directory for JSON files')
    args = parser.parse_args()

    csvs_to_json(args.input, args.output)

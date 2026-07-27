import networkx as nx
import random
import argparse
from tqdm import tqdm

from graph_text_encoder import GraphTextEncoder
from check_correctness import is_valid_hamiltonian_path, is_valid_shortest_path_weighted, is_minimum_spanning_tree_weighted
from utils import save_datasets

from instructions import INSTRUCTIONS
from questions import QUESTIONS
from formats import FORMATS
from answer_types import ANSWER_TYPES


def cycle_checking(graph):
    try:
        nx.find_cycle(graph)
        cycle = True
    except nx.NetworkXNoCycle:
        cycle = False
    return cycle


def generate_random_graph(task_name, graph_size='Tiny'):
    if graph_size == 'Tiny':
        n_lower = 5
        n_upper = 36
    elif graph_size == 'Small':
        n_lower = 36
        n_upper = 101
    elif graph_size == 'Medium':
        n_lower = 100
        n_upper = 501
    

    if task_name == 'hamiltonian_path':
        n_nodes = random.randint(n_lower, n_upper)
        graph = nx.tournament.random_tournament(n=n_nodes)
    

    elif task_name == 'maximum_flow':
        n_nodes = random.randint(n_lower, n_upper)
        edge_probability = random.uniform(0.2, 0.5)
        min_capacity = 1
        max_capacity = 11

        graph = nx.DiGraph()
        graph.add_nodes_from(range(n_nodes))
    
        # an Erdős–Rényi random graph model
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and random.random() < edge_probability:
                    capacity = random.randint(min_capacity, max_capacity)
                    graph.add_edge(i, j, capacity=capacity)
    

    elif task_name == 'is_regular':
        n_nodes = 1
        d = 1
        while n_nodes * d % 2 != 0:
            n_nodes = random.randint(n_lower, n_upper)
            d = random.randint(1, n_nodes // 3) if random.random() < 0.8 else random.randint(1, n_nodes-1)
        regular = True if random.random() > 0.5 else False
        graph = nx.random_regular_graph(d, n_nodes)
        if not regular:
            max_n = len(list(graph.edges())) // 5
            while nx.is_regular(graph):
                del_n = random.randint(1, max_n+1)
                chosen_edges = random.sample(list(graph.edges()), del_n)
                for e in chosen_edges:
                    graph.remove_edge(e[0], e[1])
                
                add_n = random.randint(1, max_n+1)
                for _ in range(add_n):
                    attempts = 0
                    while attempts < n_upper:
                        a, b = random.sample(list(graph.nodes()), 2)
                        if not graph.has_edge(a, b):
                            graph.add_edge(a, b)
                            break
                        attempts += 1
        assert nx.is_regular(graph) if regular else not nx.is_regular(graph)


    elif task_name == 'is_bipartite':
        n_left = random.randint(n_lower//2, n_upper//2)
        n_right = random.randint(n_lower//2, n_upper//2)
        edge_prob = random.uniform(0.2, 0.8)
        graph = nx.bipartite.random_graph(n_left, n_right, edge_prob)
        while len(list(graph.edges())) == 0:
            graph = nx.bipartite.random_graph(n_left, n_right, edge_prob)

        bipartite = True if random.random() > 0.5 else False
        if not bipartite:
            max_n = len(list(graph.edges())) // 5
            del_n = min(len(list(graph.edges()))-1, random.randint(1, max_n+1))
            chosen_edges = random.sample(list(graph.edges()), del_n)
            for e in chosen_edges:
                graph.remove_edge(e[0], e[1])

            odds = list(range(1, len(list(graph.nodes())), 2))
            cycle_length = random.choice(odds)
            chosen_nodes = random.sample(list(graph.nodes()), cycle_length)
            for i in range(cycle_length):
                graph.add_edge(chosen_nodes[i], chosen_nodes[(i + 1) % cycle_length])

        assert nx.is_bipartite(graph) if bipartite else not nx.is_bipartite(graph)


    elif task_name == 'has_cycle':
        n = random.randint(n_lower, n_upper)
        k_components = random.choice([1, 2, 3])
        # num_edges = random.randint(n - k_components, n * (n - 1) // 2)
    
        sizes = [1] * k_components
        for _ in range(n - k_components):
            sizes[random.randint(0, k_components - 1)] += 1

        graph = nx.Graph()
        offset = 0
        for size in sizes:
            tree = nx.random_tree(size)
            tree = nx.relabel_nodes(tree, lambda x: x + offset)
            graph = nx.disjoint_union(graph, tree)
            offset += size
        
        has_cycle = True if random.random() > 0.5 else False
        if has_cycle:
            current_edges = set(graph.edges())
            nodes = list(graph.nodes())
            add_n = random.choice([1,2,3])
            while not cycle_checking(graph):
                for _ in range(add_n):
                    u, v = random.sample(nodes, 2)
                    if u == v:
                        continue
                    edge = (min(u, v), max(u, v))
                    if edge not in current_edges:
                        graph.add_edge(u, v)
                        current_edges.add(edge)

        assert cycle_checking(graph) if has_cycle else not cycle_checking(graph)


    elif task_name == 'weighted_shortest_path':
        n_nodes = random.randint(n_lower, n_upper)
        edge_probability = random.uniform(0.2, 0.5)
        min_weight = 1
        max_weight = 11

        graph = nx.DiGraph() if random.random() < 0.5 else nx.Graph()
        graph.add_nodes_from(range(n_nodes))
    
        # an Erdős–Rényi random graph model
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and random.random() < edge_probability:
                    weight = random.randint(min_weight, max_weight)
                    graph.add_edge(i, j, weight=weight)


    elif task_name == "weighted_minimum_spanning_tree":
        n_nodes = random.randint(n_lower, n_upper)
        n_edges = random.randint(n_nodes, min(n_nodes**2 // 4, n_nodes+1)) if random.random() < 0.8 else random.randint(n_nodes**2 // 4, n_nodes**2 // 2)
        min_weight = 1
        max_weight = 11
        while True:
            graph = nx.gnm_random_graph(n_nodes, n_edges)
            if nx.is_connected(graph) and len(list(nx.isolates(graph))) == 0:
                break
        for u, v in graph.edges():
            graph[u][v]['weight'] = random.randint(min_weight, max_weight)

    return graph
    

def generate_prompt_answer(task_name, graph, graph_text_encoder):
    graph_description = graph_text_encoder.generate_graph_description()
    name_dict = graph_text_encoder.name_dict

    if task_name == 'hamiltonian_path':
        hamilton_path = nx.tournament.hamiltonian_path(graph)
        assert is_valid_hamiltonian_path(graph, hamilton_path)
        hamilton_path = [int(u) for u in hamilton_path]

        answer = f"{hamilton_path}"
        prompt = INSTRUCTIONS[task_name] + graph_description + QUESTIONS[task_name] + FORMATS[task_name]
        sample = "none"
    
    elif task_name == 'maximum_flow':
        while True:
            sample_nodes = random.sample(list(graph.nodes()), 2)
            try:
                flow_value, flow_dict = nx.maximum_flow(graph, sample_nodes[0], sample_nodes[1])
                break
            except nx.NetworkXError: # In case there's no path from source to sink
                pass
        answer = f"{float(flow_value)}"
        prompt = (
            INSTRUCTIONS[task_name] 
            + graph_description 
            + QUESTIONS[task_name].format(sample_nodes[0], sample_nodes[1]) 
            + FORMATS[task_name]
        )
        sample = f"{(sample_nodes[0], sample_nodes[1])}"
    
    elif task_name == 'is_regular':
        is_regular = nx.is_regular(graph)
        answer = "Yes" if is_regular else "No"
        prompt = INSTRUCTIONS[task_name] + graph_description + QUESTIONS[task_name] + FORMATS[task_name]
        sample = "none"

    elif task_name == 'is_bipartite':
        is_bipartite = nx.is_bipartite(graph)
        answer = "Yes" if is_bipartite else "No"
        prompt = INSTRUCTIONS[task_name] + graph_description + QUESTIONS[task_name] + FORMATS[task_name]
        sample = "none"

    elif task_name == 'has_cycle':
        has_cycle = cycle_checking(graph)
        answer = "Yes" if has_cycle else "No"
        prompt = INSTRUCTIONS[task_name] + graph_description + QUESTIONS[task_name] + FORMATS[task_name]
        sample = "none"
    
    elif task_name == 'weighted_shortest_path':
        if len(list(graph.edges())) == 0:
            return None, None, None
        shortest_path_all = dict(nx.shortest_path(graph, weight='weight'))
        shortest_path_all = {(u, v): path for u, inner_dict in shortest_path_all.items() for v, path in inner_dict.items()}
        shortest_path_all = dict(sorted(shortest_path_all.items(), key=lambda x: len(x[1]), reverse=True))
        sample_nodes = list(shortest_path_all.keys())[0]
        shortest_path = shortest_path_all[sample_nodes]
        assert is_valid_shortest_path_weighted(graph, shortest_path, sample_nodes[0], sample_nodes[1])

        shortest_path = [int(u) for u in shortest_path]
        answer = f"{shortest_path}"
        prompt = (
            INSTRUCTIONS[task_name] 
            + graph_description 
            + QUESTIONS[task_name].format(sample_nodes[0], sample_nodes[1]) 
            + FORMATS[task_name]
        )
        sample = f"{(sample_nodes[0], sample_nodes[1])}"

    elif task_name == 'weighted_minimum_spanning_tree':
        mst = list(nx.minimum_spanning_tree(graph, weight='weight').edges())
        assert is_minimum_spanning_tree_weighted(graph, mst)
        # total_weight = sum(G[u][v]['weight'] for u, v in mst_edges)
        mst = sorted([(int(tup[0]), int(tup[1])) for tup in mst])
        answer = f"{mst}"
        prompt = INSTRUCTIONS[task_name] + graph_description + QUESTIONS[task_name] + FORMATS[task_name]
        sample = "none"
               
    return prompt, answer, sample



def generate_task_data(task_name, verbose=False):
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
    }
    # answer_dict={"Yes":[], "No": []}
    # answer_dict = []
    for i in tqdm(range(args.sample_num * 2)):
        graph = generate_random_graph(task_name)

        graph_text_encoder = GraphTextEncoder(graph, weighted=True if task_name in WEIGHTED_TASKS else False)
        name_dict = graph_text_encoder.name_dict
        directed = graph_text_encoder.is_directed

        prompt, answer, sample = generate_prompt_answer(task_name, graph, graph_text_encoder)
        if prompt is None:
            continue

        # answer_dict[answer].append(len(list(graph.edges)))
        # answer_dict.append(len(list(graph.edges())))

        renamed_nodes = sorted(name_dict.values())
        if task_name in WEIGHTED_TASKS:
            renamed_edges = [(u, v, int(list(w.values())[0])) for u, v, w in graph.edges(data=True)]
        else:
            renamed_edges = [(u, v) for u, v in graph.edges()]

        task_data['task'].append(task_name)
        task_data['prompt'].append(prompt)
        task_data['answer'].append(answer)
        task_data['source_file'].append("none")
        task_data['direction'].append('directed' if directed else 'undirected')
        task_data['nodes'].append(f'({renamed_nodes[0]}, {renamed_nodes[-1]})')
        task_data['edges'].append(f'{renamed_edges}')
        task_data['samples'].append(sample)
        task_data['answer_type'].append(ANSWER_TYPES[task_name])

        if verbose:
            print(task_data['prompt'])
            print(task_data['answer'])

        if len(task_data['answer']) >= args.sample_num:
            break
    # print(len(answer_dict['Yes']), len(answer_dict['No']))
    # for key, value in answer_dict.items():
    #     answer_dict[key] = sum(value) / len(value)
    # print(answer_dict)
    #print(sum(answer_dict) / len(answer_dict))
    return task_data


EXTEND_TASKS = [
    'hamiltonian_path', 'maximum_flow', 
    "has_cycle", "is_regular", "is_bipartite",
    "weighted_shortest_path", "weighted_minimum_spanning_tree",
]
WEIGHTED_TASKS = ['maximum_flow', 'weighted_shortest_path', "weighted_minimum_spanning_tree"]


parser = argparse.ArgumentParser(description="Generate RL datasets")
parser.add_argument('--graph_size', type=str, default='Tiny', help='Size of the graph: Tiny, Small, Medium, Large')
parser.add_argument('--sample_num', type=int, default=20000, help='Number of samples to generate')
args = parser.parse_args()

parent_path='/data2/graph/graph_task_small_raw'
for task_name in EXTEND_TASKS:
    print(f"Generating tasks for: {task_name}")
    task_data = generate_task_data(task_name, verbose = False)
    save_datasets(task_data, save_path=parent_path+f"/{task_name}_{args.graph_size}_{len(task_data['answer'])}.csv")

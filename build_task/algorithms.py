import networkx as nx
from itertools import permutations
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD


def is_vertex_cover(G, vertex_set):
    # Convert vertex_set to a set if it's a list
    vertex_set = set(vertex_set)
    
    # Check if every edge has at least one endpoint in vertex_set
    for u, v in G.edges():
        if u not in vertex_set and v not in vertex_set:
            return False
    
    return True


def min_vertex_cover(G):
    if len(G.nodes()) <= 20:
        # For small graphs, use branch and bound
        return min_vertex_cover_branch_and_bound(G)
    else:
        # For larger graphs, use ILP
        return min_vertex_cover_ilp(G)


def min_vertex_cover_branch_and_bound(G):
    nodes = list(G.nodes())
    best_cover = set(nodes)  # Initialize with all nodes (worst case)
    
    def branch_and_bound(G, current_cover, remaining_nodes):
        nonlocal best_cover
        
        # If current cover is already worse than best, prune
        if len(current_cover) >= len(best_cover):
            return
        
        # If no nodes left to consider
        if not remaining_nodes:
            if is_vertex_cover(G, current_cover):
                best_cover = current_cover.copy()
            return
        
        # Take next node to branch on
        node = remaining_nodes[0]
        new_remaining = remaining_nodes[1:]
        
        # Branch 1: Include the node in the cover
        branch_and_bound(G, current_cover | {node}, new_remaining)
        
        # Branch 2: Exclude the node from the cover
        # For this branch, we need to ensure all edges where this node is an endpoint are covered
        
        # Get all edges where node is an endpoint
        if G.is_directed():
            edges_to_cover = list(G.out_edges(node)) + list(G.in_edges(node))
        else:
            edges_to_cover = list(G.edges(node))
        
        # Find nodes that must be added to cover these edges
        nodes_to_add = set()
        for u, v in edges_to_cover:
            if u != node and v != node:
                continue
            
            other_node = v if u == node else u
            if other_node not in current_cover:
                nodes_to_add.add(other_node)
        
        # If we can exclude node, continue with branch 2
        new_cover = current_cover | nodes_to_add
        remaining_filtered = [n for n in new_remaining if n not in nodes_to_add]
        branch_and_bound(G, new_cover, remaining_filtered)
    
    branch_and_bound(G, set(), nodes)
    return best_cover


def min_vertex_cover_ilp(G):
    # Create the ILP problem
    model = LpProblem(name="min_vertex_cover", sense=LpMinimize)
    
    # Create binary variables for each node
    # x[i] = 1 if node i is in the vertex cover, 0 otherwise
    x = {node: LpVariable(name=f"x_{node}", cat='Binary') for node in G.nodes()}
    
    # Objective: minimize the number of nodes in the cover
    model += lpSum(x.values())
    
    # Constraints: for each edge (u,v), at least one of u or v must be in the cover
    for u, v in G.edges():
        model += x[u] + x[v] >= 1
    
    model.solve(PULP_CBC_CMD(msg=False))
    min_cover = {node for node in G.nodes() if x[node].value() > 0.5}
    return min_cover



def solve_tsp_exact(G):
    n = G.number_of_nodes()

    # Generate all possible permutations of nodes
    nodes = list(G.nodes())
    all_permutations = permutations(nodes)
    
    min_cost = float('inf')
    best_path = None
    
    for path in all_permutations:
        cost = 0
        # Calculate path cost
        for i in range(n - 1):
            cost += G[path[i]][path[i+1]]['weight']
        # Add cost of returning to the starting node
        cost += G[path[-1]][path[0]]['weight']
        
        if cost < min_cost:
            min_cost = cost
            best_path = list(path) + [path[0]]  # Complete the cycle
    
    return best_path, min_cost


def find_isomorphism_mapping(G1, G2):
    if not nx.is_isomorphic(G1, G2):
        return None
    
    nodes_G1 = list(G1.nodes())
    nodes_G2 = list(G2.nodes())
    if len(nodes_G1) != len(nodes_G2):
        return None
    
    GM = nx.isomorphism.GraphMatcher(G1, G2)
    if GM.is_isomorphic():
        return GM.mapping
    return None

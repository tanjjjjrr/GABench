# Question categories
# Note: all tasks require extracting a 3-hop completely induced subgraph centered on a node.

QUESTION_CATEGORIES = {
    # Graph information retrieval: simple basic information queries
    'graph_info_retrieval': {
        'node': [
            'degree', 'neighbor', 'node_text_description',  # No subgraph extraction required
        ],
        'edge': [
            'edge_existence',  # No subgraph extraction required
        ],
        'graph': [
            'node_number', 'edge_number', 'node_id_from_text',
            'ego_subgraph_3hop',
        ],
    },
    # Graph theory: graph structure analysis, metric calculations, and complex algorithms
    'graph_theory': {
        'node': [
            'harmonic_centrality', 'closeness_centrality', 'betweenness_centrality',
            'clustering', 'clustering_coefficient', 'constraint', 'effective_size',
            'degree_centrality', 'avg_neighbor_degree', 'common_neighbor',
            'min_vertex_cover', 'dominating_set', 'maximal_independent_set', 'triangles', 'isolate_number',
        ],
        'edge': [
            'maximum_flow', 'local_connectivity', 'resource_allocation_index',
            'jaccard_coefficient', 'adamic_adar_index', 'weighted_shortest_path',
            'min_weight_matching', 'max_weight_matching', 'min_edge_covering',
            'bridges', 'shortest_path', 'bfs', 'dfs',
        ],
        'graph': [
            'diameter', 'radius', 'periphery', 'center', 'barycenter', 'wiener_index',
            'global_efficiency', 'transitivity', 'is_regular', 'degree_assortativity',
            'degree_connectivity', 'k_core', 'strongly_connected_number',
            'weakly_connected_number', 'is_tournament', 'is_eularian', 'is_eulerian',
            'is_strongly_connected', 'is_weakly_connected',
            'minimum_spanning_tree', 'weighted_minimum_spanning_tree',
            'maximal_cliques', 'density', 'has_cycle', 'is_bipartite',
            'is_connected', 'is_weakly_connected', 'is_strongly_connected',
            'topological_sort', 'connected_component_number',
        ],
    },
}

# =============================================================================
# Question template definitions
# =============================================================================
# All tasks require extracting a 3-hop completely induced subgraph centered on a node.

QUESTIONS = {
    # -------------------------------------------------------------------------
    # Tasks that require extracting a 3-hop ego subgraph centered on node{}.
    # -------------------------------------------------------------------------

    'edge_existence':
        "Question: Extract the 3-hop ego subgraph centered at node {}. Is there an edge between node {} and node {} in this subgraph?\n\n",

    'degree':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the degree of node {} in this subgraph?\n\n",

    'common_neighbor':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What are the common neighbors between node {} and node {} in this subgraph?\n\n",

    'neighbor':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What are the neighbors of node {} in this subgraph?\n\n",

    # -------------------------------------------------------------------------
    # Tasks that require extracting a 3-hop ego subgraph centered on node{}.
    # -------------------------------------------------------------------------

    # Graph-level questions answered from the subgraph
    'node_number': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. How many nodes are in this subgraph?\n\n",
    
    'edge_number': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. How many edges are in this subgraph?\n\n",
    
    'density': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the density of this subgraph?\n\n",
    
    'is_regular': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. Is this subgraph regular?\n\n",
    
    'has_cycle': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. Does this subgraph have a cycle?\n\n",
    
    'is_bipartite': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. Is this subgraph bipartite?\n\n",
    
    'is_connected': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. Is this subgraph connected?\n\n",
    
    'is_weakly_connected': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. Is this subgraph weakly connected?\n\n",
    
    'is_strongly_connected': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. Is this subgraph strongly connected?\n\n",
    
    'transitivity': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the transitivity of this subgraph?\n\n",
    
    'clustering_coefficient': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the clustering coefficient of node {} in this subgraph?\n\n",
    
    'topological_sort': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the topological sort of this subgraph?\n\n",
    
    'diameter': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the diameter of this subgraph?\n\n",
    
    'radius': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the radius of this subgraph?\n\n",
    
    'periphery': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the periphery of this subgraph?\n\n",
    
    'center': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the center of this subgraph?\n\n",
    
    'barycenter': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the barycenter of this subgraph?\n\n",
    
    'wiener_index': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the Wiener index of this subgraph?\n\n",
    
    'global_efficiency': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the global efficiency of this subgraph?\n\n",
    
    'isolate_number': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. How many isolated nodes are there in this subgraph?\n\n",
    
    'degree_assortativity': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the degree assortativity of this subgraph?\n\n",
    
    'degree_connectivity': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the degree connectivity of this subgraph for the degree {}?\n\n",
    
    'k_core': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the k-core of this subgraph?\n\n",
    
    'strongly_connected_number': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. How many strongly connected components are there in this subgraph?\n\n",
    
    'weakly_connected_number': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. How many weakly connected components are there in this subgraph?\n\n",
    
    'connected_component_number': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. How many connected components are there in this subgraph?\n\n",

    'is_tournament': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. Is this subgraph a tournament?\n\n",
    
    'is_eularian': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. Is this subgraph Eulerian?\n\n",
    
    'is_eulerian': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. Is this subgraph Eulerian?\n\n",
    
    'minimum_spanning_tree': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the minimum spanning tree of this subgraph?\n\n",
    
    'weighted_minimum_spanning_tree': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the minimum spanning tree of this weighted subgraph?\n\n",
    
    'maximal_cliques': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What are the maximal cliques of this subgraph?\n\n",
    
    'min_weight_matching': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the minimum weight matching of this subgraph?\n\n",
    
    'max_weight_matching': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the maximum weight matching of this subgraph?\n\n",
    
    'min_edge_covering': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the minimum edge covering of this subgraph?\n\n",

    # Node-level questions answered from the subgraph
    'triangles':
        "Question: Extract the 3-hop ego subgraph centered at node {}. How many triangles include node {} in this subgraph?\n\n",
    
    'clustering':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the clustering of node {} in this subgraph?\n\n",
    
    'constraint':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the constraint of node {} in this subgraph?\n\n",
    
    'effective_size':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the effective size of node {} in this subgraph?\n\n",
    
    'degree_centrality':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the degree centrality of node {} in this subgraph?\n\n",
    
    'avg_neighbor_degree':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the average neighbor degree of node {} in this subgraph?\n\n",
    
    'closeness_centrality': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the closeness centrality of node {} in this subgraph?\n\n",
    
    'betweenness_centrality':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the betweenness centrality of node {} in this subgraph?\n\n",
    
    'harmonic_centrality':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the harmonic centrality of node {} in this subgraph?\n\n",
    
    'min_vertex_cover':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the minimum vertex cover of this subgraph?\n\n",
    
    'dominating_set':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the dominating set of this subgraph?\n\n",
    
    'maximal_independent_set':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the maximal independent set that includes node {} of this subgraph?\n\n",

    # Edge-level questions answered from the subgraph
    'shortest_path': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the shortest path between node {} and node {} in this subgraph?\n\n",
    
    'weighted_shortest_path': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the shortest path between node {} and node {} in this weighted subgraph?\n\n",
    
    'dfs': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the depth-first search (DFS) traversal order starting from node {} in this subgraph?\n\n",
    
    'bfs': 
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the breadth-first search (BFS) traversal order starting from node {} in this subgraph?\n\n",
    
    'maximum_flow':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the value of the maximum flow from node {} to node {} in this subgraph?\n\n",
    
    'local_connectivity':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the local connectivity between node {} and node {} in this subgraph?\n\n",
    
    'resource_allocation_index':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the resource allocation index between node {} and node {} in this subgraph?\n\n",
    
    'jaccard_coefficient':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the Jaccard coefficient between node {} and node {} in this subgraph?\n\n",
    
    'adamic_adar_index':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the Adamic-Adar index between node {} and node {} in this subgraph?\n\n",
    
    'bridges':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What are the bridges of this subgraph?\n\n",

    # -------------------------------------------------------------------------
    # Special tasks
    # -------------------------------------------------------------------------
    'ego_subgraph_3hop':
        "Question: Extract the 3-hop ego subgraph centered at node {} in the graph.\n\n",

    # -------------------------------------------------------------------------
    # Graph information retrieval: node text tasks
    # -------------------------------------------------------------------------
    'node_text_description':
        "Question: Extract the 3-hop ego subgraph centered at node {}. What is the text description of node {} in this subgraph?\n\n",

    'node_id_from_text':
        "Question: Extract the 3-hop ego subgraph centered at node {}. Given the text description '{}', find the node ID that matches this description in the subgraph.\n\n",
}

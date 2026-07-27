INSTRUCTIONS = {
'node_number':
   "The task is to determine the number of nodes in the graph.\n\n",
'edge_number': 
   "The task is to determine the number of edges in the graph.\n\n" 
   "For the undirected graph, you should count the edge between two nodes only once.\n\n",
'degree': 
   "The task is to determine the degree of a node in the graph.\n\n"
   "For the undirected graph, you should count the edge between two nodes only once.\n\n",    
'common_neighbor': 
   "The task is to determine common neighbors between two nodes in the graph.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'neighbor': 
   "The task is to determine the neighbors of a node in the graph.\n\n"
   "For directed graph, you should return the successors of the node.\n\n",
'triangles':
   "The task is to find the number of triangles that include a specific node as one vertex.\n\n"
   "A triangle is a set of three nodes that are all connected to each other.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'density':
   "The task is to determine the density of the graph.\n\n"
   "Density is defined as the ratio of the number of edges in the graph to the number of possible edges.\n\n",
'closeness_centrality': 
   "The task is to determine the closeness centrality of a node in the graph.\n\n"
   "For a node *u*, closeness centrality is the reciprocal of the average shortest path distance to *u* over all *n-1* reachable nodes. For directed graphs, it computes the incoming distance to *u*.\n\n",
'betweenness_centrality':
   "The task is to determine the betweenness centrality of a node in the graph.\n\n"
   "Betweenness centrality of a node *u* is the sum of the fraction of all-pairs shortest paths that pass through *u*.\n\n",
'harmonic_centrality':
   "The task is to determine the harmonic centrality of a node in the graph.\n\n"
   "Harmonic centrality of a node *u* is the sum of the reciprocal of the shortest path distances from all other nodes to u.\n\n",
'wiener_index':
   "The task is to determine the Wiener index of a connected graph.\n\n"
   "The Wiener index of a graph is the sum of the shortest-path distances between each pair of reachable nodes. For pairs of nodes in undirected graphs, only one orientation of the pair is counted.\n\n"
   "In the input graph, all node pairs are guaranteed to be reachable.\n\n",
'barycenter':
   "The task is to determine the barycenter of a graph.\n\n"
   "The barycenter of a graph is also called the median. It includes the node that minimizes the sum of shortest path lengths to all other nodes.\n\n"
   "The input graph is guaranteed to be connected.\n\n",
'center':
   "The task is to determine the center of a graph.\n\n"
   "The center of a graph includes the node that minimizes the maximum distance to any other nodes in the graph.\n\n"
   "The input graph is guaranteed to be connected.\n\n",
'diameter':
   "The task is to determine the diameter of a graph.\n\n"
   "The diameter of a graph is the longest shortest path between any two nodes in the graph.\n\n"
   "The input graph is guaranteed to be connected.\n\n",
'periphery':
   "The task is to determine the periphery of a graph.\n\n"
   "The periphery of a graph is the set of nodes with the maximum eccentricity. The eccentricity of a node is the maximum distance from this node to all other nodes in the graph.\n\n"
   "The input graph is guaranteed to be connected.\n\n",
'radius':
   "The task is to determine the radius of a graph.\n\n"
   "The radius of a graph is the minimum eccentricity of any node in the graph. The eccentricity of a node is the maximum distance from this node to all other nodes in the graph.\n\n"
   "The input graph is guaranteed to be connected.\n\n",
'global_efficiency':
   "The task is to determine the global efficiency of a graph.\n\n"
   "Global efficiency is the average efficiency of all pairs of nodes. The efficiency of a pair of nodes is the multiplicative inverse of the shortest path distance between the nodes.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'transitivity':
   "The task is to determine the transitivity of a graph.\n\n"
   "The transitivity is the fraction of all possible triangles present in the graph. Possible triangles are identified by the number of triads (two edges with a shared vertex). The transitivity is 3*(#triangles)/(#triads).\n\n",
'constraint':
   "The task is to determine the structural constraint of a node in the graph.\n\n"
   "The constraint is a measure of how constrained a node is by its neighbors in terms of redundancy of connections.\n\n",
'resource_allocation_index':
   "The task is to determine the resource allocation index of two nodes in a graph.\n\n"
   "The resource allocation index of two nodes is the sum of the inverse of the degrees of the common neighbors of the two nodes.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'jaccard_coefficient':
   "The task is to determine the Jaccard coefficient of two nodes in a graph.\n\n"
   "The Jaccard coefficient is the size of the intersection divided by the size of the union of the neighbors of the two nodes.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'adamic_adar_index':
   "The task is to determine the Adamic-Adar index of two nodes in a graph.\n\n"
   "The Adamic-Adar index is the sum of the inverse logarithm of the degrees of the common neighbors of the two nodes.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'isolate_number':
   "The task is to determine the number of isolated nodes in the graph.\n\n"
   "An isolated node is a node that has no edges connecting it to any other nodes.\n\n",
'degree_centrality':
   "The task is to determine the degree centrality of a node in the graph.\n\n"
   "Degree centrality for a node is the fraction of nodes it is connected to.\n\n",
'avg_neighbor_degree':
   "The task is to determine the average degree of the neighbors of a node in the graph.\n\n",
'degree_assortativity':
   "The task is to determine the degree assortativity of the graph.\n\n"
   "Degree assortativity is the correlation between the degrees of connected nodes in the graph.\n\n",
'strongly_connected_number':
   "The task is to determine the number of strongly connected components in a directed graph.\n\n"
   "A strongly connected component is a maximal subgraph where every node is reachable from every other node.\n\n",
'connected_component_number':
   "The task is to determine the number of connected components in an undirected graph.\n\n"
   "A connected component is a subgraph where any two nodes are connected to each other by paths.\n\n",
'bridges':
   "The task is to find all bridges of a graph.\n\n"
   "A bridge is an edge in a graph whose removal increases the number of connected components.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'edge_existence': 
   "The task is to determine if there is an edge connecting two nodes.\n\n"
   "For an undirected graph, determine if there is an edge between nodes *u* and *v*. For a directed graph, determine if there is an edge from *u* to *v*.\n\n",
'local_connectivity':
   "The task is to determine the local connectivity of two nodes in the graph.\n\n"
   "Local connectivity is whether there exists at least one path between the two nodes.\n\n",
'has_cycle':
   "The task is to determine if the graph has a cycle.\n\n",
'is_regular':
   "The task is to determine if the graph is regular.\n\n"
   "A regular graph is a graph where every node has the same degree.\n\n",
'is_tournament':
   "The task is to determine if the graph is a tournament.\n\n"
   "A tournament is a directed graph where every pair of nodes is connected by a single directed edge.\n\n"
   "The input graph is guaranteed to be directed.\n\n",
'is_bipartite':
   "The task is to determine if the graph is bipartite.\n\n"
   "A bipartite graph is a graph whose nodes can be divided into two disjoint sets such that no two graph vertices within the same set are adjacent.\n\n",
'topological_sort':
   "The task is to determine the topological sort of a directed acyclic graph (DAG).\n\n",
'shortest_path': 
   "The task is to determine the shortest path between two nodes.\n\n"
   "The input nodes are guaranteed to be connected.\n\n",
'dfs':
   "The task is to determine the depth-first search (DFS) traversal order given a starting node.\n\n"
   "Stop when the DFS cannot be continued.\n\n",
'bfs': 
   "The task is to determine the breadth-first search (BFS) traversal order given a starting node.\n\n"
   "Stop when the BFS cannot be continued.\n\n",
'dominating_set':
   "The task is to determine the dominating set of a graph.\n\n"
   "A dominating set is a subset of nodes such that every node in the graph is either in the set or adjacent to a node in the set.\n\n"
   "For directed graphs, any node not in the dominating set must be a successor of a node within the set.\n\n",
'maximal_independent_set':
   "The task is to determine the maximal independent set guaranteed to contain a given node in the graph.\n\n"
   "An independent set is a set of nodes such that the subgraph induced by these nodes contains no edges. A maximal independent set is an independent set such that it is not possible to add a new node and still get an independent set.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'min_edge_covering':
   "The task is to determine the minimum edge covering of a graph.\n\n"
   "An edge cover is a set of edges such that every vertex in the graph is incident to at least one edge in the set. The minimum edge cover is the edge cover with the smallest number of edges.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'min_vertex_cover':
   "The task is to determine the minimum vertex cover of a graph.\n\n"
   "A vertex cover is a set of nodes such that every edge in the graph is incident to at least one node in the set.\n\n",
'max_weight_matching':
   "The task is to determine the maximum weight matching of a graph.\n\n"
   "A matching is a set of edges without common vertices. A maximal matching cannot add more edges and still be a matching. The weight of a matching is the sum of the weights of its edges. If not specified, all edges have equal edge weights.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'minimum_spanning_tree':
   "The task is to determine the minimum spanning tree of a graph.\n\n"
   "A minimum spanning tree is a subset of the edges that connects all vertices in the graph with the minimum possible total edge weights. If not specified, all edges have equal edge weights.\n\n"
   "The input graph is guaranteed to be undirected and connected.\n\n",
'maximum_flow':
   "The task is to determine the value of the maximum flow for the given source node and sink node.\n\n"
   "The maximum flow is the greatest amount of flow that can be sent from the source to the sink without violating capacity constraints.\n\n",
'clustering_coefficient':
   "The task is to determine the clustering coefficient of a node in the graph.\n\n"
   "The clustering coefficient of a node is the fraction of possible triangles that exist through that node.\n\n",
'is_connected':
   "The task is to determine if the graph is connected.\n\n"
   "A graph is connected if there is a path between every pair of nodes.\n\n",
'is_weakly_connected':
   "The task is to determine if the directed graph is weakly connected.\n\n"
   "A directed graph is weakly connected if replacing all of its directed edges with undirected edges produces a connected graph.\n\n"
   "The input graph is guaranteed to be directed.\n\n",
'is_strongly_connected':
   "The task is to determine if the directed graph is strongly connected.\n\n"
   "A directed graph is strongly connected if there is a directed path from every node to every other node.\n\n"
   "The input graph is guaranteed to be directed.\n\n",
'weakly_connected_number':
   "The task is to determine the number of weakly connected components in a directed graph.\n\n"
   "A weakly connected component is a maximal subgraph where the underlying undirected graph is connected.\n\n",
'k_core':
   "The task is to determine the k-core of a graph.\n\n"
   "A k-core is the maximal subgraph in which every vertex has degree at least k.\n\n",
'maximal_cliques':
   "The task is to find all maximal cliques in a graph.\n\n"
   "A clique is a set of vertices where every two vertices are adjacent. A maximal clique is a clique that cannot be extended by adding any other vertex.\n\n",
'degree_connectivity':
   "The task is to determine the degree connectivity for a given degree.\n\n"
   "Degree connectivity is the average degree of the neighbors of all nodes with the specified degree.\n\n",
'effective_size':
   "The task is to determine the effective size of a node in the graph.\n\n"
   "Effective size is the number of neighbors minus the average number of neighbors that are also neighbors of each other.\n\n",
'is_eulerian':
   "The task is to determine if the graph is Eulerian.\n\n"
   "An Eulerian graph is a graph that contains an Eulerian circuit, which is a cycle that visits every edge exactly once.\n\n",
'node_existence':
   "The task is to determine if the specified node exists in the graph.\n\n"
   "Answer Yes if the node exists, otherwise answer No.\n\n",
'avg_neighbor_degree':
   "The task is to determine the average degree of the neighbors of a node in the graph.\n\n"
   "For an undirected graph, only one orientation of each edge is counted in neighbor degrees.\n\n",
'local_connectivity':
   "The task is to determine whether there exists at least one path between two nodes in the graph.\n\n"
   "Answer Yes if the two nodes are connected, otherwise answer No.\n\n",
'connected_component_number':
   "The task is to determine the number of connected components in an undirected graph.\n\n"
   "A connected component is a maximal subgraph where any two nodes are connected by a path.\n\n",
'strongly_connected_number':
   "The task is to determine the number of strongly connected components in a directed graph.\n\n"
   "A strongly connected component is a maximal subgraph where every node is reachable from every other node.\n\n",
'weakly_connected_number':
   "The task is to determine the number of weakly connected components in a directed graph.\n\n"
   "A weakly connected component is a maximal subgraph where the underlying undirected graph is connected.\n\n",
'is_bipartite':
   "The task is to determine if the graph is bipartite.\n\n"
   "A bipartite graph is a graph whose nodes can be divided into two disjoint sets such that no two vertices within the same set are adjacent.\n\n",
'is_tournament':
   "The task is to determine if the graph is a tournament.\n\n"
   "A tournament is a directed graph where every pair of nodes is connected by a single directed edge.\n\n"
   "The input graph is guaranteed to be directed.\n\n",
'bridges':
   "The task is to find all bridges in the graph.\n\n"
   "A bridge is an edge whose removal increases the number of connected components.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'minimum_spanning_tree':
   "The task is to determine the minimum spanning tree of the graph.\n\n"
   "A minimum spanning tree is a subset of edges that connects all vertices with minimum total edge weight.\n\n"
   "If not specified, all edges have equal edge weights.\n\n"
   "The input graph is guaranteed to be undirected and connected.\n\n",
'resource_allocation_index':
   "The task is to determine the resource allocation index of two nodes.\n\n"
   "The resource allocation index of two nodes is the sum of the inverse of the degrees of their common neighbors.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'jaccard_coefficient':
   "The task is to determine the Jaccard coefficient of two nodes.\n\n"
   "The Jaccard coefficient is the size of the intersection divided by the size of the union of the neighbors of the two nodes.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'adamic_adar_index':
   "The task is to determine the Adamic-Adar index of two nodes.\n\n"
   "The Adamic-Adar index is the sum of the inverse logarithm of the degrees of the common neighbors of the two nodes.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'maximum_flow':
   "The task is to determine the value of the maximum flow from a source node to a sink node.\n\n"
   "The maximum flow is the greatest amount of flow that can be sent from the source to the sink without violating capacity constraints.\n\n",
'k_core':
   "The task is to determine the k-core of the graph.\n\n"
   "A k-core is the maximal subgraph in which every vertex has degree at least k.\n\n",
'maximal_cliques':
   "The task is to find all maximal cliques in the graph.\n\n"
   "A clique is a set of vertices where every two vertices are adjacent. A maximal clique cannot be extended by adding any other vertex.\n\n",
'weighted_shortest_path':
   "The task is to determine the weighted shortest path between two nodes.\n\n"
   "The shortest path is the path between two nodes with the minimum total edge weight.\n\n",
'min_weight_matching':
   "The task is to determine the minimum weight matching of a graph.\n\n"
   "A matching is a set of edges without common vertices. The weight of a matching is the sum of the weights of its edges. The minimum weight matching finds the matching with the smallest total edge weight.\n\n"
   "The input graph is guaranteed to be undirected.\n\n",
'weighted_minimum_spanning_tree':
   "The task is to determine the weighted minimum spanning tree of a graph.\n\n"
   "A minimum spanning tree is a subset of edges that connects all vertices with minimum total edge weight.\n\n"
   "The input graph is guaranteed to be undirected and connected.\n\n",
'ego_subgraph_3hop':
   "The task is to extract the 3-hop ego subgraph centered at a given node.\n\n"
   "The 3-hop ego subgraph contains the center node and all nodes that are at most 3 hops away from it, along with all edges between these nodes.\n\n",

# Graph information retrieval: node text tasks
'node_text_description':
   "The task is to find the text description of a specific node in the graph.\n\n"
   "Each node in the graph has an associated raw text description.\n\n",

'node_id_from_text':
   "The task is to find the node ID that matches a given text description.\n\n"
   "Each node in the graph has an associated raw text description. Given a text description, you need to find which node has that exact description.\n\n",
}

if __name__ == "__main__":
   print(f"Total number of tasks: {len(INSTRUCTIONS)}")

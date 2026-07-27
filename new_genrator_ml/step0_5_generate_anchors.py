#!/usr/bin/env python3
"""
Step 0.5 extracts anchors with special topological structures for higher-order operators in Stages 4, 5, and 6.

To control runtime on large graphs, it uses local sampling, early stopping, and a default capacity limit of 500 anchors.
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx

_this_dir = Path(__file__).parent
_root_dir = _this_dir.parent
sys.path.insert(0, str(_root_dir))
sys.path.insert(0, str(_this_dir))

from base_graph_loader import BaseGraphLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset processed_data.pt file")
    parser.add_argument("--output", type=str, default="anchor_nodes_cache.json", help="Output anchor-cache file")
    parser.add_argument("--max-size", type=int, default=500, help="Maximum capacity of each anchor list")
    parser.add_argument("--graph-format", type=str, choices=["TAG", "TPG", "PG"], default="TAG", help="Graph format")
    return parser

def find_cycle_anchors(G: nx.Graph, max_size: int = 500):
    """Stage 4: locally sample cycles of length 3 to 5 so selected anchors support later Ground Truth calculations for triangles and larger cycles."""
    nodes = list(G.nodes())
    cycle_nodes = set()
    attempts = 0
    max_attempts = 100000
    
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    while len(cycle_nodes) < max_size and attempts < max_attempts:
        attempts += 1
        u = random.choice(nodes)
        if u in cycle_nodes:
            continue
            
        neighbors = list(G_undirected.neighbors(u))
        if not neighbors:
            continue
            
        v = random.choice(neighbors)
        
        # After removing edge (u, v), find a path; if it has 3 to 5 nodes, adding (u, v) forms a cycle of length 3, 4, or 5
        if G_undirected.has_edge(u, v):
            G_undirected.remove_edge(u, v)
            try:
                # Limit the search depth to 4 (at most a 5-hop cycle) to avoid a full-graph explosion
                paths = nx.single_source_shortest_path(G_undirected, u, cutoff=4)
                if v in paths:
                    path = paths[v]
                    if 3 <= len(path) <= 5:
                        cycle_nodes.update(path)
            except Exception:
                pass
            finally:
                G_undirected.add_edge(u, v)
                
    return list(cycle_nodes)[:max_size]

def find_clique_anchors(G: nx.Graph, max_size: int = 500):
    """Stage 4: locally sample cliques of size 3, which correspond to triangles."""
    nodes = list(G.nodes())
    clique_nodes = set()
    attempts = 0
    max_attempts = 100000
    
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    while len(clique_nodes) < max_size and attempts < max_attempts:
        attempts += 1
        u = random.choice(nodes)
        if u in clique_nodes:
            continue
            
        neighbors = list(G_undirected.neighbors(u))
        if len(neighbors) < 2:
            continue
            
        found = False
        check_pairs = 0
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                check_pairs += 1
                if check_pairs > 100:  # Prevent hub nodes from hanging the process
                    break
                v = neighbors[i]
                w = neighbors[j]
                if G_undirected.has_edge(v, w):
                    # Found triangle u-v-w
                    clique_nodes.update([u, v, w])
                    found = True
                    break
            if found or check_pairs > 100:
                break
                
    return list(clique_nodes)[:max_size]

def find_bridge_anchors(G: nx.Graph, max_size: int = 500):
    """Stage 5: find articulation points and their one-hop neighbors, using a generator to avoid exhaustive full-graph computation."""
    G_undirected = G.to_undirected() if G.is_directed() else G
    bridge_nodes = set()
    
    try:
        # articulation_points is a generator
        art_points_gen = nx.articulation_points(G_undirected)
        for ap in art_points_gen:
            if len(bridge_nodes) >= max_size:
                break
            bridge_nodes.add(ap)
            # For completeness, also include one-hop neighbors (perturbations often target edges among them)
            for nbr in G_undirected.neighbors(ap):
                bridge_nodes.add(nbr)
                if len(bridge_nodes) >= max_size:
                    break
    except Exception as e:
        logger.warning(f"Error finding articulation points: {e}.The graph may be disconnected or may not contain articulation points.")
        
    return list(bridge_nodes)[:max_size]

def find_path_anchor_pairs(G: nx.Graph, max_size: int = 500):
    """Stage 6: find node pairs that are connected by a path but not directly adjacent."""
    nodes = list(G.nodes())
    valid_pairs = []
    attempts = 0
    max_attempts = 100000
    
    while len(valid_pairs) < max_size and attempts < max_attempts:
        attempts += 1
        u = random.choice(nodes)
        
        if G.is_directed():
            neighbors = list(G.successors(u))
        else:
            neighbors = list(G.neighbors(u))
            
        if not neighbors:
            continue
            
        # Randomly walk 2 to 4 hops
        hops = random.randint(2, 4)
        current = u
        for _ in range(hops):
            if G.is_directed():
                next_options = list(G.successors(current))
            else:
                next_options = list(G.neighbors(current))
            if not next_options:
                break
            current = random.choice(next_options)
            
        v = current
        if u != v:
            # Check whether the nodes are directly connected
            if not G.has_edge(u, v):
                # Deduplicate to avoid a large number of repeated pairs
                if [u, v] not in valid_pairs and [v, u] not in valid_pairs:
                    valid_pairs.append([u, v])
                
    return valid_pairs[:max_size]

def main():
    args = build_parser().parse_args()
    logger.info(f"Loading dataset from {args.dataset}")
    
    loader = BaseGraphLoader(args.dataset, graph_format=args.graph_format)
    G = nx.DiGraph()
    edge_index = getattr(loader.data, "edge_index", None)
    if edge_index is not None:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        G.add_edges_from(zip(srcs, dsts))
        
    num_nodes = getattr(loader.data, "num_nodes", None)
    if num_nodes is not None:
        G.add_nodes_from(range(num_nodes))
        
    logger.info(f"Graph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    logger.info("Extracting Cycle Anchors (Stage 4)...")
    cycle_anchors = find_cycle_anchors(G, args.max_size)
    logger.info(f"  Found {len(cycle_anchors)} cycle nodes (length >= 4).")
    
    logger.info("Extracting Clique Anchors (Stage 4)...")
    clique_anchors = find_clique_anchors(G, args.max_size)
    logger.info(f"  Found {len(clique_anchors)} clique nodes (size 3).")
    
    logger.info("Extracting Bridge/Cut-point Anchors (Stage 5)...")
    bridge_anchors = find_bridge_anchors(G, args.max_size)
    logger.info(f"  Found {len(bridge_anchors)} bridge-related nodes.")
    
    logger.info("Extracting Path/Convergence Anchor Pairs (Stage 6)...")
    path_anchor_pairs = find_path_anchor_pairs(G, args.max_size)
    logger.info(f"  Found {len(path_anchor_pairs)} connected but non-adjacent pairs.")
    
    output_data = {
        "metadata": {
            "dataset": args.dataset,
            "timestamp": datetime.now().isoformat(),
            "max_size_limit": args.max_size
        },
        "cycle_anchors": cycle_anchors,
        "clique_anchors": clique_anchors,
        "bridge_anchors": bridge_anchors,
        "path_anchor_pairs": path_anchor_pairs
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Anchor lists saved to {args.output}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 1 generates Ground Truth from precomputed global metrics and native graph operators.

The refactored workflow avoids meaningless physical-damage simulations, caches global rankings such as PageRank, samples tool combinations for each task, and records actual operator results as Ground Truth.
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import torch
import signal

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("Timed out!")

if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, _timeout_handler)

_this_dir = Path(__file__).parent
_root_dir = _this_dir.parent
sys.path.insert(0, str(_root_dir))
sys.path.insert(0, str(_this_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from base_graph_loader import BaseGraphLoader
from domain_config import DOMAIN_CONFIGS

import local_graph_ops

STAGE_TOOLS = {
    "global": [
        "shortest_path", "graph_connectivity", "subgraph_extraction", 
        "minimum_spanning_tree"
    ],
    1: ["check_local_reachability", "count_common_neighbors", "compute_local_clustering"],
    2: ["get_local_neighbors_by_hop", "extract_ego_boundary"],
    3: ["filter_nodes_by_all_degree", "filter_nodes_by_in_degree", "filter_nodes_by_out_degree"],
    4: ["detect_local_cycles", "extract_local_k_core", "enumerate_local_cliques", "compute_ego_density"],
    5: ["find_local_bridge_nodes", "simulate_edge_perturbation"],
    6: ["find_stepped_shortest_path", "detect_path_convergence"],
    7: []   # Used only for ML operators (appended dynamically with --enable-ml); local_label_propagation has been removed
}

# Graph machine-learning tools (appended only with --enable-ml)
ML_TOOL_NAMES = [
    "structural_node_embedding_classification",
    "structural_node_embedding_link_prediction",
]

# =====================================================================
# Clean graph-theoretic algorithm set
# =====================================================================


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--dataset-name", type=str, default=None, help="Optional override for dataset name")
    parser.add_argument("--output", type=str, default="intermediate_ground_truth.json")
    parser.add_argument("--enable-ml", action="store_true", help="Enable graph machine learning tools")
    parser.add_argument("--enable-mem", action="store_true", help="Enable memory/perturb questions")
    parser.add_argument("--mode", type=str, choices=["global", "ego"], default="global", help="Control tool-selection logic")
    parser.add_argument("--graph-format", type=str, choices=["TAG", "TPG", "PG"], default="TAG", help="Graph format")
    parser.add_argument("--graph-type", type=str, choices=["citation", "social", "recommendation"], default="citation", help="Domain of the graph")
    parser.add_argument("--tasks-per-graph", type=int, default=3, help="[TPG mode] Number of tasks generated per molecular graph")
    return parser

def _run_tpg_gt(args, cache_data: dict) -> list:
    """
    Compute Ground Truth in TPG mode by executing graph operators within each selected molecular graph.
    
    Returns a flat intermediate_tasks list.
    """
    from base_graph_loader import TPGDataset
    tpg_dataset = TPGDataset(args.dataset)

    selected_indices = cache_data.get("metadata", {}).get("selected_graph_indices", [])
    if not selected_indices:
        logger.error("[TPG] selected_graph_indices is missing from the cache; run step0 first.")
        return []

    # Build an index keyed by graph_idx
    tpg_graphs_cache = {g["graph_idx"]: g for g in cache_data.get("tpg_graphs", [])}
    dataset_name = args.dataset_name if args.dataset_name else "the molecular dataset"
    tasks_per_graph = getattr(args, "tasks_per_graph", 3)

    # Build the TPG tool pool (retain only operators with polymer/chemical meaning)
    _TPG_ALLOWED_TOOLS = {
        "shortest_path", "subgraph_extraction", 
        "get_local_neighbors_by_hop", "extract_ego_boundary", 
        "detect_local_cycles", "find_stepped_shortest_path", 
        "detect_path_convergence", "find_local_bridge_nodes", 
        "count_common_neighbors", "simulate_edge_perturbation", 
        "check_local_reachability"
    }
    tpg_stage_tools = {}
    for stage, tools in STAGE_TOOLS.items():
        filtered_tools = [t for t in tools if t in _TPG_ALLOWED_TOOLS]
        if filtered_tools:
            tpg_stage_tools[stage] = filtered_tools
            
    if getattr(args, "enable_ml", False):
        tpg_stage_tools.setdefault(7, []).append("structural_graph_embedding_classification")
        
    if not getattr(args, "enable_mem", False):
        if "simulate_edge_perturbation" in tpg_stage_tools.get(5, []):
            tpg_stage_tools[5].remove("simulate_edge_perturbation")
            logger.info("[TPG] --enable-mem is False; removing simulate_edge_perturbation from Stage 5")

    all_intermediate_tasks = []
    task_global_idx = 0
    ml_masked_data = {}

    for graph_idx in selected_indices:
        g = tpg_dataset.get_graph(graph_idx)

        # Build the NetworkX graph for this molecule
        G_mol = nx.DiGraph()
        edge_index = getattr(g, "edge_index", None)
        if edge_index is not None:
            G_mol.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        num_nodes = getattr(g, "num_nodes", 0)
        if num_nodes:
            G_mol.add_nodes_from(range(num_nodes))

        all_nodes = list(G_mol.nodes())
        if len(all_nodes) < 2:
            logger.warning(f"[TPG] Graph #{graph_idx} has fewer than 2 nodes; skipping.")
            continue
        logger.info(f"[TPG] Graph #{graph_idx}: {G_mol.number_of_nodes()} nodes, {G_mol.number_of_edges()} edges")

        # Load this graph's descriptions from the cache
        graph_cache = tpg_graphs_cache.get(graph_idx, {})
        dynamic_descriptions = graph_cache.get("dynamic_tool_descriptions", {})
        label_text = graph_cache.get("label_text", "")
        label_value = graph_cache.get("label_value", -1)

        # Compute degree rankings for this subgraph instead of using the global cache
        deg = dict(G_mol.degree())
        top5_count = max(1, int(len(all_nodes) * 0.05))
        sorted_by_deg = [n for n, _ in sorted(deg.items(), key=lambda x: -x[1])]
        top5_deg = sorted_by_deg[:top5_count]
        if G_mol.is_directed():
            top5_in_deg = [n for n, _ in sorted(dict(G_mol.in_degree()).items(), key=lambda x: -x[1])][:top5_count]
            top5_out_deg = [n for n, _ in sorted(dict(G_mol.out_degree()).items(), key=lambda x: -x[1])][:top5_count]
        else:
            top5_in_deg = top5_out_deg = top5_deg

        completed_tasks_in_graph = 0
        attempts = 0
        while completed_tasks_in_graph < tasks_per_graph and attempts < tasks_per_graph * 10:
            attempts += 1
            # Randomly select anchors
            node1, node2 = random.sample(all_nodes, 2)
            node1_desc = f"the atom with Node ID {node1} in molecular compound Graph #{graph_idx}"
            node2_desc = f"the atom with Node ID {node2} in molecular compound Graph #{graph_idx}"

            # Choose the Stage combination (TPG requires Stages 1 and 2, skips Stage 3 filtering, and excludes empty stages)
            selected_stages = [1, 2]
            avail_late = [s for s in [4, 5, 6, "global"] if tpg_stage_tools.get(s)]
            num_late = random.randint(1, 2)
            late_stages = random.sample(avail_late, min(num_late, len(avail_late)))
            selected_stages.extend(late_stages)
            if tpg_stage_tools.get(7):
                selected_stages.append(7)
            # Preserve ordering: treat global as 0 (or 8, but below 7) so Stage 7 (ML operators) is always last
            selected_stages.sort(key=lambda x: 999 if x == 7 else (0 if x == "global" else int(x)))
            selected_stages = [s for s in selected_stages if tpg_stage_tools.get(s)]
            selected_tools = [random.choice(tpg_stage_tools[s]) for s in selected_stages]

            # [NEW] Mutual-exclusion rule: if simulate_edge_perturbation is selected, clear all other operators
            if "simulate_edge_perturbation" in selected_tools:
                selected_tools = ["simulate_edge_perturbation"]
                selected_stages = [5]

            G_current = G_mol.copy()
            ground_truth = {}
            task_semantics = []
            task_dataset_path = args.dataset

            timeout_occurred = False
            for tool in selected_tools:
                semantic_text = dynamic_descriptions.get(tool, tool)
                task_semantics.append({"tool": tool, "semantic": semantic_text})
                try:
                    if hasattr(signal, "SIGALRM"): signal.alarm(900)
                    # ===== Stage global =====
                    if tool == "shortest_path":
                        try:
                            res = {"success": True, "path": nx.shortest_path(G_current, node1, node2)}
                        except nx.NetworkXNoPath:
                            res = {"success": True, "path": None}
                    elif tool == "graph_connectivity":
                        conn = nx.is_strongly_connected(G_current) if G_current.is_directed() else nx.is_connected(G_current)
                        res = {"success": True, "is_connected": conn}
                    elif tool == "subgraph_extraction":
                        G_undir = G_current.to_undirected() if G_current.is_directed() else G_current
                        # Performance optimization: extract a 1-hop ego_graph to avoid timeouts on large graphs
                        ego_graph = nx.ego_graph(G_undir, node1, radius=1)
                        # Replace cliques_containing_node for compatibility across NetworkX versions
                        cliques = [c for c in nx.find_cliques(ego_graph) if node1 in c]
                        # Return the largest clique (usually a two-atom bond or a three-atom ring in a molecular graph)
                        largest_clique = max(cliques, key=len) if cliques else [node1]
                        res = {"success": True, "subgraph_nodes": largest_clique}
                    elif tool == "minimum_spanning_tree":
                        G_undir = G_current.to_undirected()
                        mst = nx.minimum_spanning_tree(G_undir)
                        res = {"success": True, "mst_edges_count": mst.number_of_edges()}
                    # ===== Stage 1 =====
                    elif tool == "check_local_reachability":
                        res = {"success": True, "reachable": local_graph_ops.check_local_reachability(G_current, node1, node2)}
                    elif tool == "count_common_neighbors":
                        res = local_graph_ops.count_common_neighbors(G_current, node1, node2)
                        res["success"] = True
                    elif tool == "compute_local_clustering":
                        res = {"success": True, "clustering": local_graph_ops.compute_local_clustering(G_current, node1)}
                    # ===== Stage 2 =====
                    elif tool == "get_local_neighbors_by_hop":
                        res = {"success": True, "neighbors": local_graph_ops.get_local_neighbors_by_hop(G_current, node1, hop_num=1)}
                    elif tool == "extract_ego_boundary":
                        res = {"success": True, "boundary": local_graph_ops.extract_ego_boundary(G_current, node1, radius=2)}
                    # ===== Stage 3 =====
                    elif tool == "filter_nodes_by_all_degree":
                        res = {"success": True, "filtered_nodes": top5_deg}
                        G_current.remove_nodes_from([n for n in top5_deg if n in G_current and n not in (node1, node2)])
                    elif tool == "filter_nodes_by_in_degree":
                        res = {"success": True, "filtered_nodes": top5_in_deg}
                        G_current.remove_nodes_from([n for n in top5_in_deg if n in G_current and n not in (node1, node2)])
                    elif tool == "filter_nodes_by_out_degree":
                        res = {"success": True, "filtered_nodes": top5_out_deg}
                        G_current.remove_nodes_from([n for n in top5_out_deg if n in G_current and n not in (node1, node2)])
                    # ===== Stage 4 =====
                    elif tool == "detect_local_cycles":
                        res = {"success": True, "cycles": local_graph_ops.detect_local_cycles(G_current, node1, max_depth=G_current.number_of_nodes())}
                    elif tool == "extract_local_k_core":
                        res = {"success": True, "k_core_nodes": local_graph_ops.extract_local_k_core(G_current, node1)}
                    elif tool == "enumerate_local_cliques":
                        res = {"success": True, "cliques": local_graph_ops.enumerate_local_cliques(G_current, node1)}
                    elif tool == "compute_ego_density":
                        res = {"success": True, "density": local_graph_ops.compute_ego_density(G_current, node1)}
                    # ===== Stage 5 =====
                    elif tool == "find_local_bridge_nodes":
                        res = {"success": True, "bridge_nodes": local_graph_ops.find_local_bridge_nodes(G_current, node1)}
                    elif tool == "simulate_edge_perturbation":
                        edges = list(nx.ego_graph(G_current, node1, radius=1).edges())
                        remove_edges = random.sample(edges, max(1, int(len(edges) * 0.8))) if edges else []
                        res = local_graph_ops.simulate_edge_perturbation(G_current, node1, remove_edges, use_ego_graph=False)
                        res["success"] = True
                        res["removed_edges"] = remove_edges
                    # ===== Stage 6 =====
                    elif tool == "find_stepped_shortest_path":
                        res = {"success": True, "path": local_graph_ops.find_stepped_shortest_path(G_current, node1, node2)}
                    elif tool == "detect_path_convergence":
                        res = local_graph_ops.detect_path_convergence(G_current, node1, node2)
                        res["success"] = True
                    # ===== Stage 7 =====
                    elif tool == "local_label_propagation":
                        res = {"success": True, "predicted_label": local_graph_ops.local_label_propagation(G_current, node1)}
                    # ===== TPG graph-classification ML operators =====
                    elif tool == "structural_graph_embedding_classification":
                        label_val = tpg_dataset.get_graph_label(graph_idx)
                        res = {"success": True, "graph_idx": graph_idx, "label": label_val, "label_text": label_text}
                        
                        labels = torch.tensor([-100], dtype=torch.long)
                        ml_masked_data[task_global_idx] = {
                            "labels": labels,
                            "edge_index": g.edge_index if hasattr(g, 'edge_index') and g.edge_index is not None else None
                        }
                    else:
                        res = {"success": True, "value": "executed"}
                except TimeoutException:
                    logger.warning(f"[TPG] Operator {tool} timed out after 15 minutes; discarding the case and regenerating...")
                    timeout_occurred = True
                    break
                except Exception as e:
                    res = {"success": False, "error": str(e)}
                finally:
                    if hasattr(signal, "SIGALRM"): signal.alarm(0)
                ground_truth[tool] = res

            if timeout_occurred:
                continue

            task_type = "GraphQuery"
            for t in selected_tools:
                if t == "structural_graph_embedding_classification":
                    task_type = "GC"
                elif t == "structural_node_embedding_classification":
                    task_type = "NC"
                elif t == "structural_node_embedding_link_prediction":
                    task_type = "LP"

            all_intermediate_tasks.append({
                "task_index": task_global_idx,
                "task_type": task_type,
                "graph_idx": graph_idx,
                "label_text": label_text,
                "label_value": label_value,
                "dataset_name": dataset_name,
                "domain": getattr(args, "graph_type", "citation"),
                "selected_tools": selected_tools,
                "task_semantics": task_semantics,
                "anchor_nodes": {
                    "node1": node1,
                    "node2": node2,
                    "node1_desc": node1_desc,
                    "node2_desc": node2_desc,
                    "strategy": "random",
                },
                "ground_truth": ground_truth,
            })
            completed_tasks_in_graph += 1
            task_global_idx += 1
            print(f"[Progress] Generated Ground Truth for TPG question {task_global_idx}")

    return all_intermediate_tasks, ml_masked_data


def main():
    args = build_parser().parse_args()
    logger.info(f"Loading dataset from {args.dataset}")

    dataset_lower = args.dataset.lower()
    if "product" in dataset_lower or "dgraphfin" in dataset_lower:
        if "simulate_edge_perturbation" in STAGE_TOOLS.get(5, []):
            STAGE_TOOLS[5].remove("simulate_edge_perturbation")
            logger.info("The dataset name contains product or dgraphfin; simulate_edge_perturbation has been disabled.")

    # =====================================================================
    # Independent TPG-mode branch
    # =====================================================================
    if args.graph_format == "TPG":
        cache_file = "global_metrics_cache.json"
        if not Path(cache_file).exists():
            logger.error(f"Cache file {cache_file} not found! Please run step0 first.")
            return
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        intermediate_tasks, ml_masked_data = _run_tpg_gt(args, cache_data)

        out_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset": args.dataset,
                "graph_format": "TPG",
                "selected_graph_indices": cache_data.get("metadata", {}).get("selected_graph_indices", []),
            },
            "intermediate_tasks": intermediate_tasks,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
            
        if ml_masked_data:
            masked_data_path = Path(args.output).parent / f"{Path(args.output).stem}_masked_data.pt"
            torch.save(ml_masked_data, masked_data_path)
            logger.info(f"[TPG] Masked data saved to {masked_data_path}")
            
        logger.info(f"[TPG] Step1 completed with {len(intermediate_tasks)} tasks.")
        return

    # =====================================================================
    # TAG / PG modes (existing logic)
    # =====================================================================
    loader = BaseGraphLoader(args.dataset, graph_format=args.graph_format)

    # Fallback default dataset name from a generic config if domain is removed, or just use "the dataset"
    dataset_name = args.dataset_name if args.dataset_name else "the dataset"

    # semantics_lookup is used only as a backward-compatible fallback (when step2 has no dynamic descriptions)
    graph_tool_semantics = DOMAIN_CONFIGS["citation"]["semantics"] # fallback
    semantics_lookup = {entry["tool"]: entry["semantics"] for entry in graph_tool_semantics}

    logger.info("Building full global NetworkX graph...")
    G = nx.DiGraph()
    edge_index = getattr(loader.data, "edge_index", None)
    if edge_index is not None:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        G.add_edges_from(zip(srcs, dsts))

    num_nodes = getattr(loader.data, "num_nodes", None)
    if num_nodes is not None:
        G.add_nodes_from(range(num_nodes))

    logger.info(f"Global graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # ==========================================
    # Load precomputed global metrics (table lookup)
    # ==========================================
    cache_file = "global_metrics_cache.json"
    if not Path(cache_file).exists():
        logger.error(f"Cache file {cache_file} not found! Please run 'bash run_step0_precalc.sh' first.")
        return

    logger.info(f"Loading pre-calculated metrics from {cache_file}...")
    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    metrics = cache_data.get("metrics_ranked_nodes", {})
    pr_nodes = metrics.get("pagerank", [])
    deg_nodes = metrics.get("degree", [])
    in_deg_nodes = metrics.get("in_degree", [])
    out_deg_nodes = metrics.get("out_degree", [])
    k_core_nodes = metrics.get("k_core", [])

    # Load dynamic descriptions from cache
    dynamic_descriptions = cache_data.get("dynamic_tool_descriptions", {})
    dynamic_node_strategies = cache_data.get("dynamic_node_strategies", {})

    # Load anchor cache
    anchor_cache_file = "anchor_nodes_cache.json"
    if Path(anchor_cache_file).exists():
        with open(anchor_cache_file, "r", encoding="utf-8") as f:
            anchor_data = json.load(f)
        cycle_anchors = anchor_data.get("cycle_anchors", [])
        clique_anchors = anchor_data.get("clique_anchors", [])
        bridge_anchors = anchor_data.get("bridge_anchors", [])
        path_anchor_pairs = anchor_data.get("path_anchor_pairs", [])
    else:
        logger.warning(f"Cache file {anchor_cache_file} not found. Stage 4-6 anchors will be empty.")
        cycle_anchors, clique_anchors, bridge_anchors, path_anchor_pairs = [], [], [], []

    all_nodes = list(G.nodes())

    # The tool pool is defined in STAGE_TOOLS; append ML tools to Stage 7 based on the flag
    if getattr(args, "enable_ml", False):
        if args.graph_format == "TPG":
            ml_tools = ["structural_graph_embedding_classification"]
        else:
            ml_tools = ML_TOOL_NAMES
        STAGE_TOOLS[7].extend(ml_tools)
        logger.info(f"--enable-ml is enabled; appending to Stage 7: {ml_tools}")
        
    if not getattr(args, "enable_mem", False):
        if "simulate_edge_perturbation" in STAGE_TOOLS.get(5, []):
            STAGE_TOOLS[5].remove("simulate_edge_perturbation")
            logger.info("Because --enable-mem is False, simulate_edge_perturbation was removed from Stage 5")
    
    intermediate_tasks = []
    ml_masked_data = {}
    
    # Precompute the top 5% node list for the full graph as the filter tool's Ground Truth
    top5_count = max(1, int(len(all_nodes) * 0.05))
    top5_deg = deg_nodes[:top5_count]
    top5_in_deg = in_deg_nodes[:top5_count]
    top5_out_deg = out_deg_nodes[:top5_count]
    
    completed_global_tasks = 0
    attempts = 0
    while completed_global_tasks < args.num_tasks and attempts < args.num_tasks * 10:
        attempts += 1
        
        # 1. Prefer Stage combinations
        if getattr(args, "mode", "global") == "ego":
            # Ego mode: select one from Stages 1, 2, and 3, and 1–2 from Stages 4, 5, 6, and 7
            # Automatically exclude empty stages (for example, empty Stage 7 when ML is disabled)
            selected_stages = [1, 2, 3]
            num_late_stages = random.randint(1, 2)
            avail_late_tag = [s for s in [4, 5, 6, "global"] if STAGE_TOOLS.get(s)]
            late_stages = random.sample(avail_late_tag, min(num_late_stages, len(avail_late_tag)))
            selected_stages.extend(late_stages)
            if STAGE_TOOLS.get(7):
                selected_stages.append(7)
            # Preserve ordering: treat global as 0 (or 8, but below 7) so Stage 7 (ML operators) is always last
            selected_stages.sort(key=lambda x: 999 if x == 7 else (0 if x == "global" else int(x)))
            selected_stages = [s for s in selected_stages if STAGE_TOOLS.get(s)]
            selected_tools = [random.choice(STAGE_TOOLS[stage]) for stage in selected_stages]
        else:
            # Global-mode fallback logic (empty stages are excluded)
            num_tools_to_select = random.randint(3, 5)
            num_stages_to_select = num_tools_to_select
            avail_global_stages = [s for s in STAGE_TOOLS if STAGE_TOOLS.get(s) and s != 7]
            if STAGE_TOOLS.get(7):
                num_stages_to_select = max(1, num_stages_to_select - 1)
            selected_stages_raw = random.sample(avail_global_stages, min(num_stages_to_select, len(avail_global_stages)))
            if STAGE_TOOLS.get(7):
                selected_stages_raw.append(7)
            selected_stages = sorted(
                selected_stages_raw,
                key=lambda x: 999 if x == 7 else (0 if x == "global" else int(x))
            )
            selected_tools = [random.choice(STAGE_TOOLS[stage]) for stage in selected_stages]
            
        # [NEW] Mutual-exclusion rule: if simulate_edge_perturbation is selected, clear all other operators so it is the only task
        if "simulate_edge_perturbation" in selected_tools:
            selected_tools = ["simulate_edge_perturbation"]
            selected_stages = [5]
            
        # After applying mutual exclusion, perturbation tasks use random anchors and skip bridge/clique strategies
        is_perturb_only = (selected_tools == ["simulate_edge_perturbation"])
        
        # Force the anchor strategy based on the highest-level structural operator selected
        override_strategy = None
        if is_perturb_only:
            override_strategy = None  # The perturbation operator uses random nodes as anchors
        elif 6 in selected_stages and path_anchor_pairs:
            override_strategy = "path_pairs"
        elif 5 in selected_stages and bridge_anchors:
            override_strategy = "bridge"
        elif 4 in selected_stages:
            has_clique_tool = any("clique" in t for t in selected_tools)
            if has_clique_tool and clique_anchors:
                override_strategy = "clique"
            elif cycle_anchors:
                override_strategy = "cycle"

        def get_desc(m, n, nodes_list, desc_template):
            if len(nodes_list) < 2:
                n1 = n2 = nodes_list[0] if nodes_list else 0
                return n1, n2, desc_template.format(rank=1), desc_template.format(rank=2)
            idx1, idx2 = random.sample(range(min(30, len(nodes_list))), 2)
            n1, n2 = nodes_list[idx1], nodes_list[idx2]
            return n1, n2, desc_template.format(rank=idx1+1), desc_template.format(rank=idx2+1)
            
        if override_strategy == "path_pairs":
            strategy = "path_pairs"
            pair = random.choice(path_anchor_pairs)
            node1, node2 = pair[0], pair[1]
            template = dynamic_node_strategies.get("random", "the specific node with Node ID {node_id}")
            node1_desc = template.format(node_id=node1)
            node2_desc = template.format(node_id=node2)
        elif override_strategy == "bridge":
            strategy = "bridge"
            node1 = random.choice(bridge_anchors)
            node2 = random.choice(all_nodes) if all_nodes else 0
            template = dynamic_node_strategies.get("random", "the specific node with Node ID {node_id}")
            node1_desc = template.format(node_id=node1)
            node2_desc = template.format(node_id=node2)
        elif override_strategy == "cycle":
            strategy = "cycle"
            node1 = random.choice(cycle_anchors)
            node2 = random.choice(all_nodes) if all_nodes else 0
            template = dynamic_node_strategies.get("random", "the specific node with Node ID {node_id}")
            node1_desc = template.format(node_id=node1)
            node2_desc = template.format(node_id=node2)
        elif override_strategy == "clique":
            strategy = "clique"
            node1 = random.choice(clique_anchors)
            node2 = random.choice(all_nodes) if all_nodes else 0
            template = dynamic_node_strategies.get("random", "the specific node with Node ID {node_id}")
            node1_desc = template.format(node_id=node1)
            node2_desc = template.format(node_id=node2)
        else:
            strategy = random.choice(["pagerank", "degree", "in_degree", "out_degree", "k_core", "random", "raw_text"])
            if strategy == "pagerank" and pr_nodes:
                template = dynamic_node_strategies.get("pagerank", "the node ranked {rank} globally in PageRank")
                node1, node2, node1_desc, node2_desc = get_desc(0, 0, pr_nodes, template)
            elif strategy == "degree" and deg_nodes:
                template = dynamic_node_strategies.get("degree", "the node with the {rank}th highest total degree")
                node1, node2, node1_desc, node2_desc = get_desc(0, 0, deg_nodes, template)
            elif strategy == "in_degree" and in_deg_nodes:
                template = dynamic_node_strategies.get("in_degree", "the node with the {rank}th highest in-degree")
                node1, node2, node1_desc, node2_desc = get_desc(0, 0, in_deg_nodes, template)
            elif strategy == "out_degree" and out_deg_nodes:
                template = dynamic_node_strategies.get("out_degree", "the node with the {rank}th highest out-degree")
                node1, node2, node1_desc, node2_desc = get_desc(0, 0, out_deg_nodes, template)
            elif strategy == "k_core" and k_core_nodes:
                template = dynamic_node_strategies.get("k_core", "the node ranked {rank} in terms of k-core centrality")
                node1, node2, node1_desc, node2_desc = get_desc(0, 0, k_core_nodes, template)
            elif strategy == "raw_text":
                pool = deg_nodes if deg_nodes else all_nodes
                node1, node2, _, _ = get_desc(0, 0, pool, "")
                raw_text_attr = None
                if hasattr(loader.data, "raw_text") and loader.data.raw_text is not None:
                    raw_text_attr = loader.data.raw_text
                elif hasattr(loader.data, "raw_texts") and loader.data.raw_texts is not None:
                    raw_text_attr = loader.data.raw_texts
                
                if raw_text_attr is not None:
                    n1_raw = str(raw_text_attr[node1])
                    n2_raw = str(raw_text_attr[node2])
                else:
                    n1_raw = f"Node {node1} raw text"
                    n2_raw = f"Node {node2} raw text"
    
                template = dynamic_node_strategies.get("raw_text", "node: {raw_text}")
                node1_desc = template.format(raw_text=n1_raw)
                node2_desc = template.format(raw_text=n2_raw)
            else:
                # Use pure random selection as a fallback
                strategy = "random"
                node1, node2 = random.sample(all_nodes, 2)
                template = dynamic_node_strategies.get("random", "the specific node with Node ID {node_id}")
                node1_desc = template.format(node_id=node1)
                node2_desc = template.format(node_id=node2)
            
        if strategy != "raw_text":
            def enrich_node_desc(node_id, current_desc):
                info = loader.get_node_info(node_id)
                extras = []
                if info.get("title"):
                    extras.append(f"Title: {info['title']}")
                if info.get("abstract"):
                    abs_text = str(info['abstract'])
                    if len(abs_text) > 300:
                        abs_text = abs_text[:300] + "..."
                    extras.append(f"Abstract: {abs_text}")
                if not extras and info.get("raw_text"):
                    text = str(info['raw_text'])
                    if len(text) > 300:
                        text = text[:300] + "..."
                    extras.append(f"Text: {text}")
                
                if extras:
                    return f"{current_desc} - {', '.join(extras)}"
                return current_desc

            node1_desc = enrich_node_desc(node1, node1_desc)
            node2_desc = enrich_node_desc(node2, node2_desc)

        ground_truth = {}
        task_semantics = []
        task_dataset_path = args.dataset
        
        # At the start of each task, initialize a working-graph copy that filters can modify
        G_current = G.copy()
        
        timeout_occurred = False
        for tool in selected_tools:
            # Collect semantics (prefer step0 dynamic descriptions; otherwise use the fallback)
            if tool in dynamic_descriptions and dynamic_descriptions[tool]:
                semantic_text = dynamic_descriptions[tool]
            else:
                sems = semantics_lookup.get(tool, [tool])
                semantic_text = random.choice(sems)

            task_semantics.append({
                "tool": tool,
                "semantic": semantic_text
            })
            
            # Compute Ground Truth
            try:
                if hasattr(signal, "SIGALRM"): signal.alarm(900)
                # ================= Stage global =================
                if tool == "shortest_path":
                    try:
                        ego_graph_4hop = nx.ego_graph(G_current, node1, radius=4)
                        res = {"success": True, "path": nx.shortest_path(ego_graph_4hop, node1, node2)}
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        res = {"success": True, "path": None}
                elif tool == "graph_connectivity":
                    ego_graph_3hop = nx.ego_graph(G_current, node1, radius=3)
                    conn = nx.is_strongly_connected(ego_graph_3hop) if ego_graph_3hop.is_directed() else nx.is_connected(ego_graph_3hop)
                    res = {"success": True, "is_connected": conn}
                elif tool == "subgraph_extraction":
                    G_undir = G_current.to_undirected() if G_current.is_directed() else G_current
                    # Performance optimization: extract a 1-hop ego_graph to avoid timeouts on large graphs
                    ego_graph = nx.ego_graph(G_undir, node1, radius=1)
                    # Replace cliques_containing_node for compatibility across NetworkX versions
                    cliques = [c for c in nx.find_cliques(ego_graph) if node1 in c]
                    largest_clique = max(cliques, key=len) if cliques else [node1]
                    res = {"success": True, "subgraph_nodes": largest_clique}
                elif tool == "minimum_spanning_tree":
                    G_undir = G_current.to_undirected()
                    # Avoid full-graph MST blowups; solve on the ego graph when the graph is too large
                    calc_G = nx.ego_graph(G_undir, node1, radius=3) if G_undir.number_of_nodes() > 5000 else G_undir
                    mst = nx.minimum_spanning_tree(calc_G)
                    res = {"success": True, "mst_edges_count": mst.number_of_edges()}
                # ================= Stage 1 =================
                elif tool == "check_local_reachability":
                    ego_graph_3hop = nx.ego_graph(G_current, node1, radius=3)
                    res = {"success": True, "reachable": local_graph_ops.check_local_reachability(ego_graph_3hop, node1, node2)}
                elif tool == "count_common_neighbors":
                    res = local_graph_ops.count_common_neighbors(G_current, node1, node2)
                    res["success"] = True
                elif tool == "compute_local_clustering":
                    res = {"success": True, "clustering": local_graph_ops.compute_local_clustering(G_current, node1)}
                # ================= Stage 2 =================
                elif tool == "get_local_neighbors_by_hop":
                    res = {"success": True, "neighbors": local_graph_ops.get_local_neighbors_by_hop(G_current, node1, hop_num=1)}
                elif tool == "extract_ego_boundary":
                    res = {"success": True, "boundary": local_graph_ops.extract_ego_boundary(G_current, node1, radius=2)}
                # ================= Stage 3 =================
                elif tool == "filter_nodes_by_all_degree":
                    res = {"success": True, "filtered_nodes": top5_deg}
                    G_current.remove_nodes_from([n for n in top5_deg if n in G_current and n not in (node1, node2)])
                elif tool == "filter_nodes_by_in_degree":
                    res = {"success": True, "filtered_nodes": top5_in_deg}
                    G_current.remove_nodes_from([n for n in top5_in_deg if n in G_current and n not in (node1, node2)])
                elif tool == "filter_nodes_by_out_degree":
                    res = {"success": True, "filtered_nodes": top5_out_deg}
                    G_current.remove_nodes_from([n for n in top5_out_deg if n in G_current and n not in (node1, node2)])
                # ================= Stage 4 =================
                elif tool == "detect_local_cycles":
                    res = {"success": True, "cycles": local_graph_ops.detect_local_cycles(G_current, node1, max_depth=5)}
                elif tool == "extract_local_k_core":
                    res = {"success": True, "k_core_nodes": local_graph_ops.extract_local_k_core(G_current, node1)}
                elif tool == "enumerate_local_cliques":
                    res = {"success": True, "cliques": local_graph_ops.enumerate_local_cliques(G_current, node1)}
                elif tool == "compute_ego_density":
                    res = {"success": True, "density": local_graph_ops.compute_ego_density(G_current, node1)}
                # ================= Stage 5 =================
                elif tool == "find_local_bridge_nodes":
                    res = {"success": True, "bridge_nodes": local_graph_ops.find_local_bridge_nodes(G_current, node1)}
                elif tool == "simulate_edge_perturbation":
                    edges = list(nx.ego_graph(G_current, node1, radius=1).edges())
                    remove_edges = random.sample(edges, max(1, int(len(edges) * 0.8))) if edges else []
                    res = local_graph_ops.simulate_edge_perturbation(G_current, node1, remove_edges, use_ego_graph=True)
                    res["success"] = True
                    res["removed_edges"] = remove_edges
                # ================= Stage 6 =================
                elif tool == "find_stepped_shortest_path":
                    res = {"success": True, "path": local_graph_ops.find_stepped_shortest_path(G_current, node1, node2)}
                elif tool == "detect_path_convergence":
                    res = local_graph_ops.detect_path_convergence(G_current, node1, node2)
                    res["success"] = True
                # ================= Stage 7 =================
                elif tool == "local_label_propagation":
                    res = {"success": True, "predicted_label": local_graph_ops.local_label_propagation(G_current, node1)}
                # ================= ML Tools =================
                elif tool == "structural_graph_embedding_classification":
                    graph_idx = loader.data.batch[node1].item() if hasattr(loader.data, 'batch') and loader.data.batch is not None else 0
                    label_val = loader.data.y[graph_idx].item() if hasattr(loader.data, 'y') and loader.data.y is not None else -1
                    res = {"success": True, "node_id": node1, "graph_idx": graph_idx, "label": label_val}
                    
                    labels = loader.data.y.clone() if hasattr(loader.data, 'y') and loader.data.y is not None else None
                    if labels is not None:
                        try:
                            labels[graph_idx] = -100
                        except Exception:
                            pass
                    
                    ml_masked_data[completed_global_tasks] = {
                        "labels": labels,
                        "edge_index": loader.data.edge_index if hasattr(loader.data, 'edge_index') and loader.data.edge_index is not None else None
                    }
                elif tool == "structural_node_embedding_classification":
                    label_val = loader.data.y[node1].item() if hasattr(loader.data, 'y') and loader.data.y is not None else -1
                    res = {"success": True, "node_id": node1, "label": label_val}
                    
                    labels = loader.data.y.clone() if hasattr(loader.data, 'y') and loader.data.y is not None else None
                    if labels is not None:
                        try:
                            labels[node1] = -100
                        except Exception:
                            pass
                            
                    ml_masked_data[completed_global_tasks] = {
                        "labels": labels,
                        "edge_index": loader.data.edge_index if hasattr(loader.data, 'edge_index') and loader.data.edge_index is not None else None
                    }
                elif tool == "structural_node_embedding_link_prediction":
                    edge_index = loader.data.edge_index
                    mask1 = (edge_index[0] == node1) & (edge_index[1] == node2)
                    mask2 = (edge_index[0] == node2) & (edge_index[1] == node1)
                    has_edge = bool((mask1 | mask2).any().item())
                    res = {"success": True, "node1": node1, "node2": node2, "has_edge": has_edge}
                    
                    keep_mask = ~(mask1 | mask2)
                    new_edge_index = edge_index[:, keep_mask]
                    
                    ml_masked_data[completed_global_tasks] = {
                        "labels": loader.data.y if hasattr(loader.data, 'y') and loader.data.y is not None else None,
                        "edge_index": new_edge_index
                    }
                else:
                    res = {"success": True, "value": "executed"}
            except TimeoutException:
                logger.warning(f"[Global] Operator {tool} timed out after 15 minutes; discarding the case and regenerating...")
                timeout_occurred = True
                break
            except Exception as e:
                res = {"success": False, "error": str(e)}
            finally:
                if hasattr(signal, "SIGALRM"): signal.alarm(0)
            
            ground_truth[tool] = res
        
        if timeout_occurred:
            continue

        task_type = "GraphQuery"
        for t in selected_tools:
            if t == "structural_graph_embedding_classification":
                task_type = "GC"
            elif t == "structural_node_embedding_classification":
                task_type = "NC"
            elif t == "structural_node_embedding_link_prediction":
                task_type = "LP"

        intermediate_tasks.append({
            "task_index": completed_global_tasks,
            "task_type": task_type,
            "dataset_name": dataset_name,
            "domain": getattr(args, "graph_type", "citation"),
            "selected_tools": selected_tools,
            "task_semantics": task_semantics,
            "anchor_nodes": {
                "node1": node1,
                "node2": node2,
                "node1_desc": node1_desc,
                "node2_desc": node2_desc,
                "strategy": strategy
            },
            "ground_truth": ground_truth
        })
        
        completed_global_tasks += 1
        print(f"[Progress] Generated Ground Truth for question {completed_global_tasks}")
        
    out_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": args.dataset,
            "global_metrics_cached": ["pagerank"]
        },
        "intermediate_tasks": intermediate_tasks
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
        
    if ml_masked_data:
        masked_data_path = Path(args.output).parent / f"{Path(args.output).stem}_masked_data.pt"
        torch.save(ml_masked_data, masked_data_path)
        logger.info(f"Masked data saved to {masked_data_path}")
        
    logger.info(f"Successfully computed {len(intermediate_tasks)} multi-tool ground truths.")

if __name__ == "__main__":
    main()

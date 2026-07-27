#!/usr/bin/env python3
"""
Step 0 precomputes global graph metrics and stores them in a local cache.

It also runs DynamicToolDescriptionGenerator to create fuzzy, domain-adaptive tool descriptions and stores the results in the same JSON cache. Per new_instruct.md, all xxx-based description generation runs during step0 preprocessing so it does not block the core agent loop.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx

_this_dir = Path(__file__).parent
_root_dir = _this_dir.parent
sys.path.insert(0, str(_root_dir))
sys.path.insert(0, str(_this_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Unified base tool list (shared across domains); topological_sort is disabled because graphs may contain cycles
BASE_TOOL_NAMES = [
    "shortest_path", 
    "graph_connectivity", "subgraph_extraction", "link_prediction_score",
    "minimum_spanning_tree",
]

# Local ego-centric tool list (new)
EGO_TOOL_NAMES = [
    "get_local_neighbors_by_hop", "extract_ego_boundary", "filter_nodes_by_all_degree",
    "filter_nodes_by_in_degree", "filter_nodes_by_out_degree",
    "detect_local_cycles", "compute_local_clustering", "find_stepped_shortest_path",
    "detect_path_convergence", "find_local_bridge_nodes", "extract_local_k_core",
    "enumerate_local_cliques", "compute_ego_density", "count_common_neighbors",
    "simulate_edge_perturbation",
    "check_local_reachability"
]

# Graph machine-learning tools (appended only with --enable-ml)
ML_TOOL_NAMES = [
    "structural_node_embedding_classification",
    "structural_node_embedding_link_prediction",
]


def _init_xxx():
    """
    Initialize the asynchronous xxx provider from environment variables.
    
    Return None when the API key is unavailable so dynamic-description generation can be skipped.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from xxx.provider import xxx
    from xxx import Asyncxxxxxx, Asyncxxx

    provider_type = os.getenv("xxx_PROVIDER_TYPE", "xxx").lower()
    model_name = os.getenv("xxx_MODEL_NAME", "xxx")

    try:
        if provider_type == "xxx":
            api_key = os.getenv("xxx_API_KEY")
            if not api_key:
                logger.warning("xxx_API_KEY is not set; skipping dynamic tool-description generation")
                return None
            base_url = os.getenv("xxx_BASE_URL", "xxx")
            client = Asyncxxx(base_url=base_url, api_key=api_key, timeout=600.0)
            return xxx(client, model_name, "xxx")
        else:
            endpoint = os.getenv("xxx_xxx_ENDPOINT")
            api_key = os.getenv("xxx_xxx_API_KEY")
            api_version = os.getenv("xxx_API_VERSION", "2024-12-01-preview")
            if not endpoint or not api_key:
                logger.warning("xxx_xxx_ENDPOINT / xxx_xxx_API_KEY is not set; skipping dynamic tool-description generation")
                return None
            client = Asyncxxxxxx(
                xxx_endpoint=endpoint, api_key=api_key, api_version=api_version
            )
            return xxx(client, model_name, "xxx")
    except Exception as e:
        logger.warning(f"xxx initialization failed: {e}; skipping dynamic tool-description generation")
        return None


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset processed_data.pt file")
    parser.add_argument("--output", type=str, default="global_metrics_cache.json", help="Output cache file")
    parser.add_argument(
        "--mode", type=str, choices=["global", "ego"], default="global",
        help="Choose whether to generate global tools (global) or local ego-centered tools (ego)"
    )
    parser.add_argument(
        "--graph-type", type=str, default="",
        help="Specify the graph-data type used for hard-coded edge descriptions. When empty, global-description is used as the fallback."
    )
    parser.add_argument(
        "--enable-ml", action="store_true",
        help="Append graph machine-learning tools (structural_node_embedding_classification / link_prediction)"
    )
    parser.add_argument(
        "--global-description", type=str, default="",
        help="Optional global graph description for the non-text graph fallback (Task 4)"
    )
    parser.add_argument(
        "--skip-dynamic-desc", action="store_true",
        help="Skip dynamic tool-description generation and only precompute graph metrics"
    )
    parser.add_argument(
        "--dynamic-desc-seed", type=int, default=None,
        help="Random seed for dynamic-description sampling"
    )
    parser.add_argument(
        "--walk-depth", type=int, default=5,
        help="Step count for the DFS random walk"
    )
    parser.add_argument(
        "--graph-format", type=str, choices=["TAG", "TPG", "PG"], default="TAG",
        help="Graph format: TAG(Text-Attributed), TPG(Test-Paired), PG(Pure)"
    )
    parser.add_argument(
        "--num-graphs", type=int, default=5,
        help="[TPG mode] Number of molecular graphs to sample from the graph list"
    )
    return parser


async def _generate_dynamic_descriptions(
    loader,
    mode: str = "global",
    enable_ml: bool = False,
    global_description: str = "",
    walk_depth: int = 5,
    seed: int = None,
    graph_type: str = "citation",
) -> dict:
    """
    Generate dynamic tool descriptions asynchronously for Tasks 1–4 in new_instruct.md.
    
    The mode selects the tool list. ML_TOOL_NAMES are appended when mode is global and enable_ml is true.
    
    Returns
    -------
    dict
        Mapping from tool names to fuzzy descriptions.
    """
    from dynamic_tool_description import DynamicToolDescriptionGenerator

    xxx = _init_xxx()
    if xxx is None:
        logger.warning("The xxx is unavailable; dynamic tool descriptions will not be cached (step2 will use hard-coded descriptions)")
        return {}, {}

    if mode == "ego":
        tool_names = list(EGO_TOOL_NAMES)
        logger.info(f"[Step0] --mode ego is enabled; using 16 local tools: {tool_names}")
    else:
        tool_names = list(BASE_TOOL_NAMES)
        
    if enable_ml:
        graph_format = getattr(loader, "graph_format", "TAG") if loader else "TAG"
        if graph_format == "TPG":
            ml_tools = [] # Graph classification is provided by the user interface; no fuzzy description is needed from the xxx
        else:
            ml_tools = ["structural_node_embedding_link_prediction"] # Node classification is provided by the user interface; retain only link prediction
        tool_names = tool_names + ml_tools
        logger.info(f"[Step0] --enable-ml is enabled (format: {graph_format}); appending ML tools: {ml_tools}")

    generator = DynamicToolDescriptionGenerator(
        xxx_provider=xxx,
        tool_names=tool_names,
        loader=loader,
        fallback_descriptions={},
        global_description=global_description,
        sample_seed=seed,
        max_retries=3,
        walk_depth=walk_depth,
        graph_type=graph_type,
    )

    logger.info("[Step0] Starting dynamic tool-description generation...")
    descriptions, node_strategies = await generator.run()
    logger.info(f"[Step0] Dynamic tool-description generation completed: {len(descriptions)} descriptions and {len(node_strategies)} node strategies")
    return descriptions, node_strategies


# ── Real-world scenario generation prompt ─────────────────────────────────
_SCENARIO_SYSTEM_PROMPT = """\
You are an expert in designing high-level, abstract real-world contexts for graph datasets.
Given the general domain of a graph dataset, your task is to generate broad, domain-wide background scenarios.
These scenarios should describe an overall situation or context in the given domain.
They MUST NOT be phrased as specific analytical tasks, investigative questions, or concrete queries.
"""

_SCENARIO_USER_PROMPT_TEMPLATE = """\
Graph domain: {domain_label}
General graph context:
{graph_context}

Please generate between 5 and 10 abstract, high-level real-world situations or background contexts
that represent the general environment for this graph domain. A {role} might operate within this context.

CRITICAL CONSTRAINTS:
1. Abstract & High-Level: The scenarios must be broad, relating ONLY to the overall domain (e.g., general academic research evolution, macro e-commerce market trends).
2. NO Specific Details: Do NOT include any specific node names, edge details, or highly specific topics (e.g., do not mention specific medical terms, paper titles, product names, or user names).
3. NO Tasks or Queries: Do NOT phrase the scenarios as tasks, questions, or analytical operations. Do NOT use terms like "investigate", "find", "analyze", "identify", or "trace". Instead, describe a general state of affairs or situational background (e.g., "A national science foundation is observing the macroeconomic shift in foundational science over the last decade.").
4. Output ONLY a JSON array of strings, one string per scenario, like:
  ["Scenario 1 text...", "Scenario 2 text...", ...]
"""

# The xxx infers domain/role from the subgraph summary; provide a generic fallback here
_FALLBACK_ROLE = "domain analyst or researcher"
_FALLBACK_LABEL = "Graph Dataset"


async def _generate_real_world_scenarios(
    loader,
    fuzzy_descriptions: dict,
    global_description: str = "",
    walk_depth: int = 5,
    seed: int = None,
    max_retries: int = 3,
) -> list:
    """
    Ask the xxx to infer the domain from the sampled subgraph summary and generate 5–10 real-world scenarios without tool names.
    
    Returns
    -------
    list[str]
        Real-world scenario descriptions.
    """
    xxx = _init_xxx()
    if xxx is None:
        logger.warning("[Step0] The xxx is unavailable; skipping real-world scenario generation")
        return []

    # Build graph-context text (infer from the subgraph summary when possible; no domain parameter required)
    if global_description:
        graph_context = global_description
    else:
        try:
            from dynamic_tool_description import GraphContextSampler, xxx
            sampler = GraphContextSampler(loader=loader, max_sample_nodes=50, max_depth=walk_depth, seed=seed)
            ctx = sampler.sample()
            if ctx:
                summary_pipe = xxx(xxx)
                graph_context = await summary_pipe.generate(ctx)
            else:
                graph_context = ""
        except Exception as exc:
            logger.warning(f"Failed to generate semantic graph summary: {exc}")
            graph_context = ""

    if not graph_context:
        graph_context = f"A {_FALLBACK_LABEL} dataset."

    # The xxx infers domain_label and role from graph_context
    user_prompt = _SCENARIO_USER_PROMPT_TEMPLATE.format(
        domain_label="(infer from the Graph Context below)",
        graph_context=graph_context,
        role=_FALLBACK_ROLE,
    )

    import json as _json, re as _re

    for attempt in range(1, max_retries + 1):
        try:
            raw = await xxx.get_completion(
                system_prompt=_SCENARIO_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=2000,
            )
            raw = raw.strip()
            # Try to parse a JSON array
            # 1. Parse directly
            try:
                scenarios = _json.loads(raw)
                if isinstance(scenarios, list) and all(isinstance(s, str) for s in scenarios):
                    logger.info(f"[Step0] Successfully generated {len(scenarios)} real-world scenarios")
                    return scenarios
            except _json.JSONDecodeError:
                pass
            # 2. Extract the array from a ```json ... ``` block
            m = _re.search(r"```json\s*(\[.*?\])\s*```", raw, _re.DOTALL)
            if not m:
                m = _re.search(r"```\s*(\[.*?\])\s*```", raw, _re.DOTALL)
            if m:
                try:
                    scenarios = _json.loads(m.group(1))
                    if isinstance(scenarios, list):
                        logger.info(f"[Step0] Successfully generated {len(scenarios)} real-world scenarios")
                        return scenarios
                except _json.JSONDecodeError:
                    pass
            # 3. Find the first [...]
            start = raw.find("[")
            if start >= 0:
                depth = 0
                for ci in range(start, len(raw)):
                    if raw[ci] == "[":
                        depth += 1
                    elif raw[ci] == "]":
                        depth -= 1
                        if depth == 0:
                            try:
                                scenarios = _json.loads(raw[start: ci + 1])
                                if isinstance(scenarios, list):
                                    return scenarios
                            except _json.JSONDecodeError:
                                break
            logger.warning(f"[Step0] Attempt {attempt} scenario response parsing failed; retrying...")
        except Exception as exc:
            logger.warning(f"[Step0] Scenario-generation call attempt {attempt} failed: {exc}")

    logger.error("[Step0] All scenario-generation attempts failed; returning an empty list")
    return []


# ── TPG-specific asynchronous helper functions ────────────────────────────

async def _generate_tpg_scenarios(tpg_data, xxx, max_retries: int = 3) -> list:
    """Derive biochemical activity context from positive labels_features descriptions and ask the xxx to generate scenarios once for all graphs."""
    pos_descs = tpg_data.get_positive_label_descriptions()
    if not pos_descs:
        logger.warning("[TPG] No positive-sample descriptions were found; skipping scenario generation")
        return []

    assay_lines = "\n".join(f"- {d}" for d in pos_descs[:10])
    global_description = (
        "This is a molecular graph classification dataset from biochemistry/cheminformatics. "
        "The task is to predict whether a given molecular compound is biologically active "
        "for specific biochemical assays. "
        "The dataset covers the following biological assay targets" + assay_lines
    )
    logger.info("[TPG] Calling the xxx to generate biochemistry scenarios...")
    return await _generate_real_world_scenarios(
        loader=None,
        fuzzy_descriptions={},
        global_description=global_description,
        max_retries=max_retries,
    )


async def _generate_tpg_per_graph_descriptions(
    mol_ctx: dict,
    tool_names: list,
    xxx,
    max_retries: int = 3,
) -> tuple:
    """
    Generate fuzzy tool descriptions for one molecular graph with xxx.
    
    Returns
    -------
    tuple[dict, dict]
        Tool descriptions and node-strategy descriptions.
    """
    from dynamic_tool_description import xxx, xxx
    pipeline = xxx(xxx, tool_names, max_retries, is_tpg=True)
    descs = await pipeline.generate(mol_ctx)
    node_pipeline = xxx(xxx, max_retries)
    node_strategies = await node_pipeline.generate(mol_ctx)
    return descs or {}, node_strategies or {}


def main():
    args = build_parser().parse_args()
    logger.info(f"Loading dataset from {args.dataset} (enable_ml={args.enable_ml})")

    # =====================================================================
    # TPG mode: handle this branch independently and return when complete
    # =====================================================================
    if args.graph_format == "TPG":
        from base_graph_loader import TPGDataset
        tpg_data = TPGDataset(args.dataset)
        selected_indices = tpg_data.sample_graph_indices(args.num_graphs, seed=args.dynamic_desc_seed)
        logger.info(f"[TPG] Sampled {len(selected_indices)} molecular graphs: {selected_indices}")

        # In TPG mode, retain only operators with strong chemical or molecular meaning
        _TPG_ALLOWED_TOOLS = {
            "shortest_path", "subgraph_extraction", 
            "get_local_neighbors_by_hop", "extract_ego_boundary", 
            "detect_local_cycles", "find_stepped_shortest_path", 
            "detect_path_convergence", "find_local_bridge_nodes", 
            "count_common_neighbors", "simulate_edge_perturbation", 
            "check_local_reachability"
        }
        all_tools = list(BASE_TOOL_NAMES) + list(EGO_TOOL_NAMES)
        tpg_tool_names = [t for t in all_tools if t in _TPG_ALLOWED_TOOLS]
        
        # Note: graph classification (structural_graph_embedding_classification) is provided by the user interface; the xxx no longer generates a dynamic description

        cache_data = {
            "metadata": {
                "dataset": args.dataset,
                "graph_format": "TPG",
                "enable_ml": args.enable_ml,
                "timestamp": datetime.now().isoformat(),
                "selected_graph_indices": selected_indices,
                "num_graphs_total": tpg_data.num_graphs,
            },
            "metrics_ranked_nodes": {},   # TPG mode does not compute global metrics
            "dynamic_tool_descriptions": {},
            "real_world_scenarios": [],
            "dynamic_node_strategies": {},
            "tpg_graphs": [],
        }

        if not args.skip_dynamic_desc:
            xxx = _init_xxx()
            if xxx:
                async def _run_tpg_xxx():
                    # 1. Generate scenarios only once
                    scenarios = await _generate_tpg_scenarios(tpg_data, xxx)
                    logger.info(f"[TPG] Scenario generation completed: {len(scenarios)} entries")

                    # 2. Iterate over each selected graph and generate a per-graph fuzzy description
                    tpg_graphs_result = []
                    for idx in selected_indices:
                        logger.info(f"[TPG] Processing fuzzy descriptions for Graph #{idx}...")
                        mol_ctx = tpg_data.make_molecule_context(idx)
                        descs, node_strategies = await _generate_tpg_per_graph_descriptions(
                            mol_ctx, tpg_tool_names, xxx
                        )
                        tpg_graphs_result.append({
                            "graph_idx": idx,
                            "label_text": mol_ctx["label_text"],
                            "label_value": mol_ctx["label_value"],
                            "assay_desc": mol_ctx["assay_desc"],
                            "dynamic_tool_descriptions": descs,
                            "dynamic_node_strategies": node_strategies,
                        })
                        logger.info(f"[TPG] Graph #{idx}: {len(descs)} tool descriptions completed")
                    return scenarios, tpg_graphs_result

                try:
                    scenarios, tpg_graphs_result = asyncio.run(_run_tpg_xxx())
                    cache_data["real_world_scenarios"] = scenarios
                    cache_data["tpg_graphs"] = tpg_graphs_result
                except Exception as exc:
                    logger.warning(f"[TPG] xxx generation failed: {exc}; the cache will remain empty")
            else:
                logger.warning("[TPG] The xxx is unavailable; skipping dynamic-description generation")
        else:
            logger.info("[TPG] --skip-dynamic-desc is set; skipping")

        logger.info(f"[TPG] Writing the cache to {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        logger.info("[TPG] Step0 completed.")
        return

    # =====================================================================
    # TAG / PG modes (existing logic)
    # =====================================================================
    from base_graph_loader import BaseGraphLoader
    loader = BaseGraphLoader(args.dataset, graph_format=args.graph_format)

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

    logger.info(f"Graph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    cache_data = {
        "metadata": {
            "dataset": args.dataset,
            "enable_ml": args.enable_ml,
            "timestamp": datetime.now().isoformat(),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges()
        },
        "metrics_ranked_nodes": {},
        # Dynamic tool-description cache (new_instruct.md)
        "dynamic_tool_descriptions": {},
        # Real-world scenario cache (new_instruct.md): scenarios only, without tool names
        "real_world_scenarios": []
    }

    # 1. PageRank (reflects a node's influence on knowledge flow through the network)
    logger.info("Computing PageRank...")
    pr = nx.pagerank(G)
    cache_data["metrics_ranked_nodes"]["pagerank"] = [n for n, v in sorted(pr.items(), key=lambda x: -x[1])]

    # 2. Total Degree (overall activity)
    logger.info("Computing Total Degree...")
    deg = dict(G.degree())
    cache_data["metrics_ranked_nodes"]["degree"] = [n for n, v in sorted(deg.items(), key=lambda x: -x[1])]

    # 3. In-Degree (directed graphs: citation count, representing authority)
    if G.is_directed():
        logger.info("Computing In-Degree...")
        in_deg = dict(G.in_degree())
        cache_data["metrics_ranked_nodes"]["in_degree"] = [n for n, v in sorted(in_deg.items(), key=lambda x: -x[1])]

        # 4. Out-Degree (directed graphs: number of outgoing citations, representing survey breadth)
        logger.info("Computing Out-Degree...")
        out_deg = dict(G.out_degree())
        cache_data["metrics_ranked_nodes"]["out_degree"] = [n for n, v in sorted(out_deg.items(), key=lambda x: -x[1])]

    # 5. Core Number (K-core decomposition finds the densest core groups and is very fast)
    logger.info("Computing K-Core Number...")
    G_undirected = G.to_undirected()
    G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
    core_num = nx.core_number(G_undirected)
    cache_data["metrics_ranked_nodes"]["k_core"] = [n for n, v in sorted(core_num.items(), key=lambda x: -x[1])]

    # ── New: dynamic tool descriptions and real-world scenario generation (new_instruct.md) ──
    if not args.skip_dynamic_desc:
        try:
            async def _run_step0_xxx():
                # First: generate fuzzy tool descriptions and node strategies
                descs, node_strategies = await _generate_dynamic_descriptions(
                    loader=loader,
                    mode=args.mode,
                    enable_ml=args.enable_ml,
                    global_description=args.global_description,
                    walk_depth=args.walk_depth,
                    seed=args.dynamic_desc_seed,
                    graph_type=args.graph_type,
                )
                # Second: let the xxx infer the domain from the subgraph summary and generate real-world scenarios
                scenarios = await _generate_real_world_scenarios(
                    loader=loader,
                    fuzzy_descriptions=descs,
                    global_description=args.global_description,
                    walk_depth=args.walk_depth,
                    seed=args.dynamic_desc_seed,
                )
                return descs, scenarios, node_strategies

            dynamic_descs, real_world_scenarios, node_strategies = asyncio.run(_run_step0_xxx())
            cache_data["dynamic_tool_descriptions"] = dynamic_descs
            cache_data["real_world_scenarios"] = real_world_scenarios
            cache_data["dynamic_node_strategies"] = node_strategies
            logger.info(
                f"[Step0] Cache written: {len(dynamic_descs)} tool descriptions, "
                f"{len(real_world_scenarios)} real-world scenarios, "
                f"{len(node_strategies)} node-description strategies"
            )
        except Exception as exc:
            logger.warning(
                f"An unexpected error occurred during dynamic-description/scenario generation: {exc}, "
                "This cache entry will remain empty, and step2 will automatically use hard-coded descriptions"
            )
            cache_data["dynamic_tool_descriptions"] = {}
            cache_data["real_world_scenarios"] = []
            cache_data["dynamic_node_strategies"] = {}
    else:
        logger.info("--skip-dynamic-desc is set; skipping dynamic tool-description and scenario generation")

    # Write the cache file
    logger.info(f"Saving pre-calculated cache to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    logger.info("Pre-calculation finished successfully.")


if __name__ == "__main__":
    main()

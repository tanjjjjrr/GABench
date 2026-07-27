#!/usr/bin/env python3
"""
Base Graph Loader
=================

Domain-independent base class for loading graph datasets. It extracts reusable logic from CitationGraphLoader.

Domain-specific graph loaders, including citation, recommendation, and social-network loaders, should inherit this class and implement only a small set of abstract methods. The base class provides:

- PyG .pt file loading with torch.load
- Lazy adjacency-table construction through _ensure_adj
- BFS subgraph extraction through extract_subgraph
- Edge-list formatting through build_edge_text
- Human-readable subgraph summaries through summarize_subgraph
- Degree-related utility methods

Subclasses must implement:

- get_node_info(node_id) -> dict
- sample_nodes(n, seed) -> list[dict]
- _make_node_name(info) -> str
- _graph_type_label() -> str
- _format_node_summary_line(nid, info, id_to_name, is_seed) -> str
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class DummyData:
    pass


class BaseGraphLoader:
    """
    Generic graph-data loader that automatically extracts node attributes from PyG Data objects.
    
    Parameters
    ----------
    dataset_path : str
        Path to a serialized PyG Data object (.pt file).
    """

    def __init__(self, dataset_path: str, graph_format: str = "TAG") -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed, so .pt files cannot be loaded. Run: pip install torch"
            )

        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

        logger.info(f"Loading graph data: {dataset_path}")
        loaded_data = torch.load(
            str(self.dataset_path),
            map_location="cpu",
            weights_only=False,
        )
        
        self.graph_format = graph_format
        
        if graph_format == "TPG":
            # TPG: Dict containing 'graphs', 'texts', 'split'
            graphs = loaded_data.get("graphs", [])
            texts = loaded_data.get("texts", [])
            u_node_texts_lst = texts[0] if len(texts) > 0 else []
            
            all_edge_index = []
            all_raw_texts = []
            all_y = []
            all_batch = []
            offset = 0
            graph_idx = 0
            
            for g in graphs:
                num_g_nodes = getattr(g, "num_nodes", 0)
                if hasattr(g, "x") and g.x is not None:
                    node_indices = g.x.view(-1).tolist()
                    node_texts = [u_node_texts_lst[idx] if idx < len(u_node_texts_lst) else "" for idx in node_indices]
                else:
                    node_texts = ["" for _ in range(num_g_nodes)]
                all_raw_texts.extend(node_texts)
                
                if hasattr(g, "edge_index") and g.edge_index is not None:
                    ei = g.edge_index + offset
                    all_edge_index.append(ei)
                    
                orig_y = getattr(g, "y", None)
                if orig_y is None:
                    orig_y = getattr(g, "label", None)
                
                if orig_y is not None:
                    all_y.append(orig_y)
                    
                all_batch.extend([graph_idx] * num_g_nodes)
                
                offset += num_g_nodes
                graph_idx += 1
                
            self.data = DummyData()
            self.data.batch = torch.tensor(all_batch, dtype=torch.long)
            if all_edge_index:
                self.data.edge_index = torch.cat(all_edge_index, dim=1)
            else:
                self.data.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.data.num_nodes = offset
            self.data.raw_texts = all_raw_texts
            self.data.raw_text = all_raw_texts
            if all_y:
                try:
                    self.data.y = torch.cat(all_y, dim=0)
                except Exception:
                    self.data.y = all_y
        elif graph_format == "PG":
            # Pure Graph placeholder
            self.data = loaded_data
        else:
            self.data = loaded_data

        if isinstance(self.data, dict):
            dummy = DummyData()
            for k, v in self.data.items():
                setattr(dummy, k, v)
            self.data = dummy
            
        if not hasattr(self.data, "num_nodes"):
            if hasattr(self.data, "x") and self.data.x is not None:
                self.data.num_nodes = self.data.x.shape[0]
            elif hasattr(self.data, "edge_index") and self.data.edge_index is not None:
                self.data.num_nodes = int(self.data.edge_index.max()) + 1
            else:
                self.data.num_nodes = 0

        self.num_nodes: int = int(self.data.num_nodes)
        logger.info(f"Dataset loaded: {self.num_nodes:,} nodes (format: {graph_format})")

        # Lazily load the adjacency table
        self._adj_out: Optional[Dict[int, List[int]]] = None  # src -> [dst]
        self._adj_in:  Optional[Dict[int, List[int]]] = None  # dst -> [src]

    # ── Adjacency table construction ────────────────────────────────────────────

    def _ensure_adj(self) -> None:
        """Build adjacency tables lazily from the edge_index tensor."""
        if self._adj_out is not None:
            return

        logger.info("Building adjacency tables from edge_index...")

        if not hasattr(self.data, "edge_index") or self.data.edge_index is None:
            raise RuntimeError(
                "The edge_index field was not found in the dataset. "
                "Ensure that the .pt file contains edge_index (Tensor[2, num_edges])."
            )

        edge_index = self.data.edge_index
        src_list: List[int] = edge_index[0].tolist()
        dst_list: List[int] = edge_index[1].tolist()

        adj_out: Dict[int, List[int]] = {}
        adj_in:  Dict[int, List[int]] = {}

        for src, dst in zip(src_list, dst_list):
            adj_out.setdefault(src, []).append(dst)
            adj_in.setdefault(dst, []).append(src)

        self._adj_out = adj_out
        self._adj_in  = adj_in
        total_edges = sum(len(v) for v in adj_out.values())
        logger.info(
            f"Adjacency tables built: {len(adj_out):,} nodes with outgoing edges, "
            f"{total_edges:,} directed edges"
        )

    # ── Generic dynamic extraction interface ───────────────────────────────────

    def get_node_info(self, node_id: int) -> Dict[str, Any]:
        """Return a metadata dictionary for one node by scanning list and tensor attributes on self.data."""
        info = {"node_id": node_id}
        
        # Try to extract every attribute whose length matches num_nodes
        for key in dir(self.data):
            if key.startswith("_") or key in ["edge_index", "x", "y", "train_mask", "val_mask", "test_mask", "num_nodes"]:
                continue
            val = getattr(self.data, key)
            
            if isinstance(val, (list, tuple)) and len(val) > node_id:
                info[key] = val[node_id]
            elif isinstance(val, dict) and node_id in val:
                info[key] = val[node_id]
            elif hasattr(val, "shape") and len(val.shape) > 0 and val.shape[0] > node_id:
                try:
                    node_val = val[node_id]
                    if hasattr(node_val, "numel"):
                        info[key] = node_val.item() if node_val.numel() == 1 else node_val.tolist()
                    elif hasattr(node_val, "item"):
                        info[key] = node_val.item()
                    else:
                        info[key] = node_val
                except Exception:
                    pass
        
        # Explicitly try raw_texts or raw_text in case they are hidden from dir() or have an unusual format
        for text_attr in ["raw_texts", "raw_text"]:
            if hasattr(self.data, text_attr) and getattr(self.data, text_attr) is not None:
                try:
                    info[text_attr] = getattr(self.data, text_attr)[node_id]
                except Exception:
                    pass
        
        # Handle labels specially
        if hasattr(self.data, "y") and hasattr(self.data, "label_texts"):
            try:
                label_idx = int(self.data.y[node_id].item())
                info["label"] = self.data.label_texts[label_idx]
            except Exception:
                pass
                
        # Normalize common name fields to name
        info["name"] = f"Node_{node_id}"
        if info.get("title"):
            info["name"] = self._sanitize_name(str(info["title"]), info["name"])
            
        return info

    def sample_nodes(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Randomly sample n nodes from the dataset and return their metadata."""
        if seed is not None:
            random.seed(seed)

        n = min(n, self.num_nodes)
        indices = random.sample(range(self.num_nodes), n)
        nodes = [self.get_node_info(idx) for idx in indices]
        logger.info(f"Sampled {len(nodes)} core nodes")
        return nodes

    def _make_node_name(self, info: Dict[str, Any]) -> str:
        return info.get("name", f"Node_{info['node_id']}")

    def _graph_type_label(self) -> str:
        return "Dynamic Extracted Graph"

    def _format_node_summary_lines(
        self,
        nid: int,
        info: Dict[str, Any],
        id_to_name: Dict[int, str],
        is_seed: bool,
    ) -> List[str]:
        """Format extracted attributes as plain text."""
        star = "* " if is_seed else "  "
        lines = [f"{star}Node {nid} ({info['name']})"]
        
        for k, v in info.items():
            if k in ["node_id", "name"]:
                continue
            # Truncate overly long text and show only a readable excerpt
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            lines.append(f"       {k.capitalize():<8}: {val_str}")
        return lines

    # ── Subgraph extraction ────────────────────────────────────────────────────

    def extract_subgraph(
        self,
        seed_node_ids: List[int],
        max_nodes: int = 60,
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """Extract a local subgraph around the seed nodes with DFS random walks of at most max_depth steps."""
        self._ensure_adj()

        visited: Set[int] = set(seed_node_ids)

        for seed in seed_node_ids:
            current_node = seed
            for _ in range(max_depth):
                if len(visited) >= max_nodes:
                    break
                neighbors_out = self._adj_out.get(current_node, [])
                neighbors_in = self._adj_in.get(current_node, [])
                
                all_neighbors = neighbors_out + neighbors_in
                if not all_neighbors:
                    break
                    
                next_node = random.choice(all_neighbors)
                visited.add(next_node)
                current_node = next_node

        node_list = sorted(visited)
        visited_set = set(node_list)

        edges: List[Tuple[int, int]] = []
        for src in node_list:
            for dst in self._adj_out.get(src, []):
                if dst in visited_set:
                    edges.append((src, dst))

        nodes_info: Dict[int, Dict[str, Any]] = {
            nid: self.get_node_info(nid) for nid in node_list
        }

        logger.info(
            f"Subgraph extracted: {len(node_list)} nodes, {len(edges)} directed edges"
        )
        return {
            "nodes":      nodes_info,
            "edges":      edges,
            "seed_nodes": seed_node_ids,
            "num_nodes":  len(node_list),
            "num_edges":  len(edges),
        }

    # ── Edge-list formatting ──────────────────────────────────────────────────

    def build_edge_text(
        self,
        subgraph: Dict[str, Any],
    ) -> Tuple[str, Dict[int, str]]:
        nodes_info = subgraph["nodes"]
        edges      = subgraph["edges"]

        id_to_name: Dict[int, str] = {
            nid: self._make_node_name(info)
            for nid, info in nodes_info.items()
        }

        lines: List[str] = []
        seen: Set[str] = set()
        for src, dst in edges:
            if src in id_to_name and dst in id_to_name:
                line = f"{id_to_name[src]}\t{id_to_name[dst]}"
                if line not in seen:
                    seen.add(line)
                    lines.append(line)

        return "\n".join(lines), id_to_name

    # ── Human-readable summary ────────────────────────────────────────────────

    def summarize_subgraph(
        self,
        subgraph: Dict[str, Any],
        id_to_name: Optional[Dict[int, str]] = None,
        max_edge_preview: int = 30,
    ) -> str:
        """Serialize a subgraph as text for later xxx compression into a domain-level semantic summary."""
        nodes_info = subgraph["nodes"]
        seed_ids: Set[int] = set(subgraph.get("seed_nodes", []))

        if id_to_name is None:
            _, id_to_name = self.build_edge_text(subgraph)

        lines = [
            f"=== {self._graph_type_label()} Subgraph Data ===",
            f"Nodes: {subgraph['num_nodes']}  |  Directed Edges: {subgraph['num_edges']}",
            "",
            "Nodes Raw Metadata (* = seed node):",
        ]

        for nid, info in sorted(nodes_info.items()):
            is_seed = nid in seed_ids
            node_lines = self._format_node_summary_lines(nid, info, id_to_name, is_seed)
            lines.extend(node_lines)
            lines.append("")

        lines.append("Edges List (source -> target):")
        edge_text, _ = self.build_edge_text(subgraph)
        edge_lines = [l for l in edge_text.split("\n") if l.strip()]
        if len(edge_lines) > max_edge_preview:
            for el in edge_lines[:max_edge_preview]:
                lines.append(f"  {el}")
            lines.append(f"  ... ({len(edge_lines) - max_edge_preview} more edges)")
        else:
            for el in edge_lines:
                lines.append(f"  {el}")

        return "\n".join(lines)

    # ── Degree utilities ───────────────────────────────────────────────────────

    def get_in_degree(self, node_id: int) -> int:
        self._ensure_adj()
        return len(self._adj_in.get(node_id, []))

    def get_out_degree(self, node_id: int) -> int:
        self._ensure_adj()
        return len(self._adj_out.get(node_id, []))

    def compute_subgraph_in_degrees(
        self,
        subgraph: Dict[str, Any],
    ) -> Dict[int, int]:
        in_degree: Dict[int, int] = {nid: 0 for nid in subgraph["nodes"]}
        for _src, dst in subgraph["edges"]:
            if dst in in_degree:
                in_degree[dst] += 1
        return in_degree

    def compute_subgraph_out_degrees(
        self,
        subgraph: Dict[str, Any],
    ) -> Dict[int, int]:
        out_degree: Dict[int, int] = {nid: 0 for nid in subgraph["nodes"]}
        for src, _dst in subgraph["edges"]:
            if src in out_degree:
                out_degree[src] += 1
        return out_degree

    # ── General helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_name(raw: str, fallback: str) -> str:
        if not raw or not raw.strip():
            return fallback
        safe = "".join(
            ch if ch >= " " and ch != "\t" else " "
            for ch in raw
        ).strip()
        return safe if safe else fallback


# =============================================================================
# TPGDataset — Test-Paired Graph dataset loader (for graph classification)
# =============================================================================

class TPGDataset:
    """
    Load a OneForAll-format TPG .pt file.
    
    The top-level object is a dictionary containing graphs, texts, and split data. The graphs list contains molecular PyG Data objects. The texts entries provide atom descriptions, bond descriptions, positive and negative label descriptions, edge prompts, and graph prompts.
    """

    def __init__(self, dataset_path: str) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed, so .pt files cannot be loaded.")
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
        logger.info(f"[TPGDataset] Loading TPG dataset: {dataset_path}")
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        self.graphs = loaded.get("graphs", [])
        
        # PCBA dataset handling logic
        if "pcba" in dataset_path.lower():
            needs_conversion = False
            for g in self.graphs:
                if hasattr(g, "y") and g.y is not None and g.y.numel() == 128:
                    needs_conversion = True
                    break
            
            if needs_conversion:
                logger.info(f"[TPGDataset] Detected a PCBA dataset with y length 128; converting one-hot labels to scalar values...")
                for g in self.graphs:
                    if hasattr(g, "y") and g.y is not None and g.y.numel() == 128:
                        # Get indices whose one-hot value is 1 and convert them to a one-dimensional scalar tensor
                        g.y = torch.argmax(g.y).view(1)
                
                # Store the converted dataset again
                processed_path = str(path).replace(".pt", "_processed.pt")
                if "processed" not in str(path):
                    torch.save(loaded, processed_path)
                    logger.info(f"[TPGDataset] Saved the converted PCBA dataset to {processed_path}")
                    # Update the in-memory data so later TPG steps use the converted graphs

        texts = loaded.get("texts", [])
        self.u_node_texts: list = texts[0] if len(texts) > 0 else []
        self.u_edge_texts: list = texts[1] if len(texts) > 1 else []
        self.labels_features: list = texts[2] if len(texts) > 2 else []
        self.num_graphs: int = len(self.graphs)
        logger.info(f"[TPGDataset] Contains {self.num_graphs} molecular graphs and "
                    f"{len(self.labels_features)} label-description strings")

    # ── Public query interface ────────────────────────────────────────────────

    def get_graph(self, idx: int):
        """Return the PyG Data object for molecular graph idx."""
        return self.graphs[idx]

    def get_graph_label(self, idx: int):
        """Return the label for graph idx, or -1 when unavailable. NaN labels are permitted."""
        g = self.graphs[idx]
        
        orig_y = getattr(g, "y", None)
        if orig_y is None:
            orig_y = getattr(g, "label", None)
            
        if orig_y is not None:
            import math
            try:
                val = orig_y.item()
                if isinstance(val, float) and math.isnan(val):
                    return float('nan')
                return int(val)
            except Exception:
                if orig_y.numel() > 0:
                    val = orig_y.tolist()[0]
                    if isinstance(val, float) and math.isnan(val):
                        return float('nan')
                    return int(val)
                return -1
        return -1

    def get_positive_label_descriptions(self) -> list:
        """
        Extract biochemical activity descriptions from the positive-label half of labels_features.
        
        The returned strings contain only the assay-description portion of each positive sample.
        """
        prefix = "The molecule is effective to the following assay."
        descs = []
        half = len(self.labels_features) // 2
        for text in self.labels_features[:half]:
            if prefix in text:
                desc = text.split(prefix, 1)[-1].strip()
                if desc:
                    descs.append(desc)
        return descs

    def get_graph_label_text(self, idx: int) -> str:
        """Return the full positive-label text for graph idx, using the first positive description."""
        if not self.labels_features:
            return ""
        # Return the positive-sample description as assay context regardless of the true label
        half = len(self.labels_features) // 2
        return self.labels_features[0] if half > 0 else self.labels_features[0]

    def sample_graph_indices(self, n: int, seed: int = None) -> list:
        """Randomly sample n unique, sorted indices from the graph list."""
        import random as _random
        if seed is not None:
            _random.seed(seed)
        n = min(n, self.num_graphs)
        return sorted(_random.sample(range(self.num_graphs), n))

    # ── Molecular context construction ───────────────────────────────────────

    def make_molecule_context(self, idx: int) -> dict:
        """Build a context dictionary for molecular graph idx that is compatible with GraphContextSampler.sample() and can be consumed directly by xxx."""
        g = self.graphs[idx]
        num_nodes = getattr(g, "num_nodes", 0)

        # Extract atom text (up to 20 atoms)
        node_texts = []
        if hasattr(g, "x") and g.x is not None:
            indices = g.x.view(-1).tolist()
            for ni in indices[:20]:
                ni = int(ni)
                if ni < len(self.u_node_texts):
                    node_texts.append(self.u_node_texts[ni])

        # Extract chemical-bond text (deduplicated, up to 10 bond types)
        edge_texts = []
        if hasattr(g, "xe") and g.xe is not None:
            seen_ei = set()
            for ei in g.xe.view(-1).tolist():
                ei = int(ei)
                if ei not in seen_ei and ei < len(self.u_edge_texts):
                    seen_ei.add(ei)
                    edge_texts.append(self.u_edge_texts[ei])
                if len(edge_texts) >= 10:
                    break
        if not edge_texts and self.u_edge_texts:
            edge_texts = self.u_edge_texts[:3]

        # Get labels and assay descriptions
        label_value = self.get_graph_label(idx)
        label_full_text = self.get_graph_label_text(idx)
        pos_descs = self.get_positive_label_descriptions()
        assay_desc = pos_descs[0] if pos_descs else "molecular biological activity"

        label_str = (
            "positive (effective)" if label_value == 1
            else "negative (not effective)" if label_value == 0
            else str(label_value)
        )
        subgraph_summary = (
            f"This is molecular compound Graph #{idx} from a biochemistry graph classification dataset. "
            f"The classification task: predict whether this molecule is active for the biological assay: "
            f"{assay_desc}. "
            f"This molecule (Graph #{idx}) has ground-truth label: {label_str}. "
            f"The molecule contains {num_nodes} atoms (nodes) and the following atom types are present: "
            f"{'; '.join(node_texts[:5])}."
        )

        return {
            "node_texts": node_texts,
            "edge_texts": edge_texts,
            "subgraph_summary": subgraph_summary,
            "has_text": bool(node_texts),
            "graph_idx": idx,
            "label_text": label_full_text,
            "label_value": label_value,
            "assay_desc": assay_desc,
        }

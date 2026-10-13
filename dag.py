"""dag.py - category DAG loading and ancestor suppression (torch-free).

PDF section 3: suggest the most specific categories, but keep an ancestor when none of its
descendants is above the selection threshold.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import networkx as nx

logger = logging.getLogger("CommonsDAG")


def save_category_graph(path: Union[str, Path], graph: nx.DiGraph) -> None:
    Path(path).write_text(json.dumps({"edges": [[p, c] for p, c in graph.edges]}, ensure_ascii=False), encoding="utf-8")


def load_category_graph(path: Union[str, Path]) -> nx.DiGraph:
    """Load ``{"edges": [[parent, child], ...]}``. A missing/invalid file yields an EMPTY graph (logged),
    never a toy graph: wrong hierarchy is worse than none."""
    path = Path(path)
    g = nx.DiGraph()
    if not path.is_file():
        logger.warning("category graph %s not found; DAG post-processing is disabled", path)
        return g
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for parent, child in raw["edges"]:
            g.add_edge(str(parent), str(child))
    except Exception as exc:
        logger.warning("category graph %s unreadable (%s); DAG post-processing is disabled", path, exc)
        return nx.DiGraph()
    return g


def post_process_dag(candidates: List[Dict[str, Any]], graph: nx.DiGraph, threshold: float = 0.5,
                     key: str = "confidence") -> List[Dict[str, Any]]:
    """Drop a candidate iff some DESCENDANT of it is *selected* (score >= threshold).

    Candidates below the threshold never suppress anything, so a confident ancestor survives when
    its children are only weak candidates. Returns candidates sorted by score, best first.
    """
    ranked = sorted(candidates, key=lambda c: c[key], reverse=True)
    selected = {c["category"] for c in ranked if c[key] >= threshold}
    out = []
    for c in ranked:
        name = c["category"]
        if name in graph:
            # A descendant that can reach `name` back is in a cycle with it, so it is not "more
            # specific"; skip those or cycle members would suppress each other and vanish.
            if any(d in selected and d != name and name not in nx.descendants(graph, d)
                   for d in nx.descendants(graph, name)):
                continue
        out.append(c)
    return out

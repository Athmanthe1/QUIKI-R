"""Bug 2: ancestors were dropped whenever a descendant was merely a candidate; and the toy graph survived artifact load."""
import json

import networkx as nx

from dag import load_category_graph, post_process_dag, save_category_graph


def cand(name, p):
    return {"category": name, "confidence": p}


def graph():
    g = nx.DiGraph()
    g.add_edges_from([("Nature", "Mountains"), ("Mountains", "Alps"), ("Nature", "Rivers"), ("Places", "Alps")])
    return g


def old_post_process(candidates, g):
    """Verbatim logic of the buggy engine._post_process_dag (selected_names = ALL candidates)."""
    selected = {c["category"] for c in candidates}
    return [c for c in sorted(candidates, key=lambda x: -x["confidence"])
            if not (c["category"] in g and any(d in selected for d in nx.descendants(g, c["category"])))]


def names(cs):
    return [c["category"] for c in cs]


def test_old_behaviour_dropped_ancestor_even_when_descendant_was_a_weak_candidate():
    cs = [cand("Nature", 0.9), cand("Mountains", 0.01)]
    assert names(old_post_process(cs, graph())) == ["Mountains"]            # the bug: confident ancestor lost


def test_ancestor_kept_when_no_descendant_above_threshold():
    out = post_process_dag([cand("Nature", 0.9), cand("Mountains", 0.01)], graph(), threshold=0.5)
    assert "Nature" in names(out)


def test_ancestor_dropped_when_a_descendant_is_above_threshold():
    out = post_process_dag([cand("Nature", 0.9), cand("Mountains", 0.8), cand("Rivers", 0.1)], graph(), threshold=0.5)
    assert "Nature" not in names(out) and "Mountains" in names(out)


def test_transitive_descendants_and_multiple_parents():
    cs = [cand("Nature", 0.9), cand("Mountains", 0.6), cand("Places", 0.7), cand("Alps", 0.8)]
    out = post_process_dag(cs, graph(), threshold=0.5)
    assert names(out) == ["Alps"]          # Nature, Mountains and Places all have 'Alps' below them


def test_unknown_nodes_are_untouched_and_output_sorted():
    out = post_process_dag([cand("Cats", 0.3), cand("Dogs", 0.9)], graph(), threshold=0.5)
    assert names(out) == ["Dogs", "Cats"]


def test_cycles_do_not_crash():
    g = nx.DiGraph([("A", "B"), ("B", "A")])
    out = post_process_dag([cand("A", 0.9), cand("B", 0.8)], g, threshold=0.5)
    assert len(out) >= 1


def test_graph_roundtrip_and_missing_file_is_empty_not_toy(tmp_path):
    save_category_graph(tmp_path / "category_graph.json", graph())
    g = load_category_graph(tmp_path / "category_graph.json")
    assert set(g.edges) == set(graph().edges)
    empty = load_category_graph(tmp_path / "nope.json")
    assert empty.number_of_nodes() == 0
    (tmp_path / "bad.json").write_text("{not json")
    assert load_category_graph(tmp_path / "bad.json").number_of_nodes() == 0

"""evaluate.py - temporal hold-out evaluation (problem statement, section 5).

Protocol: the artifact index contains TRAIN images only (extract_embeddings.py); the TEST split is later in time,
uploader-disjoint and near-duplicate-free (cleaning_and_split.py / dedup_pass). Queries use the production code path
(scoring.rank_categories). Ground truth is the stored label set restricted to the label space; the share of test
labels outside it is reported. Stored labels are incomplete, so the strict score is a LOWER bound: use
--review_csv_out / --review_csv_in for the human-reviewed score.

    python evaluate.py --artifacts_dir artifacts --split test --out_dir artifacts/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

import metrics as M
from scoring import Knowledge, load_knowledge, rank_categories

logger = logging.getLogger("quiki.evaluate")
KS = (1, 3, 5, 10)
K_MAX = max(KS)


def _slice(preds, truths, idx, ks=KS):
    return M.sample_wise([preds[i] for i in idx], [truths[i] for i in idx], ks)


def evaluate_rows(rows: Sequence[Dict[str, Any]], kn: Knowledge, train_counts: Dict[str, int], *,
                  category_types: Optional[Dict[str, str]] = None, ks: Sequence[int] = KS, k_neighbors: int = 10,
                  dag_threshold: float = 0.5, review_sample: int = 0, seed: int = 0) -> Dict[str, Any]:
    """``rows``: dicts with embedding, meta_embedding (or None), categories, width, height, path."""
    ks = tuple(ks)
    kmax = max(ks)
    classes = list(kn.classes)
    cls_set = set(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    preds: List[List[str]] = []
    truths: List[Set[str]] = []
    lat: List[float] = []
    n_oov = n_labels = 0
    S = np.zeros((len(rows), len(classes)), dtype=np.float32)
    Y = np.zeros_like(S)
    health_bad: Counter = Counter()
    for r_i, r in enumerate(rows):
        cats = set(r["categories"])
        truth = cats & cls_set
        n_labels += len(cats)
        n_oov += len(cats - cls_set)
        z = np.asarray(r["embedding"], dtype=np.float32)
        meta = r.get("meta_embedding")
        t0 = time.perf_counter()
        out = rank_categories(z, None if meta is None else np.asarray(meta, np.float32), kn, k=k_neighbors,
                              top_k=max(kmax, 10), dag_threshold=dag_threshold)
        lat.append(time.perf_counter() - t0)
        preds.append([s["category"] for s in out["suggestions"]])
        truths.append(truth)
        for s in out["fused"]:
            S[r_i, cidx[s["category"]]] = s["confidence"]
        for c in truth:
            Y[r_i, cidx[c]] = 1
        for name, h in out["health"].items():
            if h.get("status") == "error":
                health_bad[name] += 1

    ev = [i for i, t in enumerate(truths) if t]                 # images with at least one in-vocabulary label
    rep: Dict[str, Any] = {
        "n_images": len(rows), "n_images_evaluated": len(ev), "n_classes": len(classes),
        "index_vectors": int(getattr(kn.index, "ntotal", 0)), "index_type": kn.index_type,
        "oov_label_fraction": (n_oov / n_labels) if n_labels else None,
        "scorers_with_errors": dict(health_bad),
        "stacker": "fitted" if kn.stacker else "none (uncalibrated mean)",
        "head": "present" if kn.head else "absent",
        "protocol": "TRAIN-only index; truth restricted to the label space; images with no in-vocabulary label are excluded",
    }
    rep["sample_wise"] = _slice(preds, truths, ev, ks)
    macro = {k: M.macro_at_k(preds, truths, classes, k) for k in ks}
    rep["macro"] = macro
    mAP, per_ap = M.mean_average_precision(S[ev], Y[ev]) if ev else (None, [])
    rep["mAP"] = mAP
    rep["hierarchy_aware_P"] = {k: float(np.mean([M.hier_precision_at_k(preds[i], truths[i], kn.graph, k) for i in ev])) if ev else None
                                for k in ks}
    # head / torso / tail: an image counts for a bucket if it has a true label in it; truth restricted to that bucket
    buckets = M.frequency_buckets({c: train_counts.get(c, 0) for c in classes})
    rep["frequency_buckets"] = {}
    for b, members in buckets.items():
        idx = [i for i in ev if truths[i] & members]
        bt = [truths[i] & members for i in range(len(rows))]
        bp = [[c for c in preds[i]] for i in range(len(rows))]
        rep["frequency_buckets"][b] = {
            "n_classes": len(members), "n_images": len(idx),
            "sample_wise": M.sample_wise([bp[i] for i in idx], [bt[i] for i in idx], ks) if idx else None,
            "macro": {k: M.macro_at_k([bp[i] for i in idx], [bt[i] for i in idx], sorted(members), k) for k in ks} if idx else None,
            "recall_any_in_top10": float(np.mean([bool(set(preds[i][:10]) & bt[i]) for i in idx])) if idx else None}
    if category_types:
        rep["category_type"] = {}
        for t in sorted(set(category_types.values())):
            members = {c for c in classes if category_types.get(c) == t}
            idx = [i for i in ev if truths[i] & members]
            rep["category_type"][t] = {"n_classes": len(members), "n_images": len(idx),
                                       "sample_wise": M.sample_wise([preds[i] for i in idx], [truths[i] & members for i in idx], ks) if idx else None}
        rep["category_type"]["_uncategorised_classes"] = sum(1 for c in classes if c not in category_types)
    else:
        rep["category_type"] = "not computed: pass --category_types CSV (category,type); no type labels exist in the data"
    bands: Dict[str, List[int]] = defaultdict(list)
    for i in ev:
        bands[M.resolution_band(rows[i].get("width", 0), rows[i].get("height", 0))].append(i)
    rep["resolution_band"] = {b: {"n_images": len(ix), "sample_wise": _slice(preds, truths, ix, ks)} for b, ix in sorted(bands.items())}
    rep["language"] = "not computed: no language labels in the data"
    rep["latency_scoring_only"] = M.latency_summary(lat)
    rep["latency_note"] = "scoring from precomputed embeddings (kNN + text + head + fusion + DAG); excludes image decode and the encoder forward pass"
    rep["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    # review sample: wrong top-k suggestions (strict), for human labelling
    rng = random.Random(seed)
    wrong = [(rows[i].get("path", str(i)), c, rank + 1) for i in ev for rank, c in enumerate(preds[i][:5]) if c not in truths[i]]
    rep["n_wrong_top5_suggestions"] = len(wrong)
    rep["_review_candidates"] = rng.sample(wrong, min(review_sample, len(wrong))) if review_sample else []
    return rep


def write_review_template(path: Path, candidates: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "suggested_category", "rank", "verdict"])
        for p, c, r in candidates:
            w.writerow([p, c, r, ""])


def apply_review(report: Dict[str, Any], csv_path: Path, k: int = 5) -> Dict[str, Any]:
    """Fill in ``verdict`` (correct / incorrect / unsure) for a random sample of strict-wrong suggestions."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        verdicts = [row["verdict"].strip().lower() for row in csv.DictReader(f)]
    strict_p = report["sample_wise"][k]["P"]
    return M.reviewed_precision(strict_p, report["n_wrong_top5_suggestions"], verdicts) | {"k": k, "strict_P": strict_p}


def to_markdown(rep: Dict[str, Any]) -> str:
    L = ["# QUIKI-R evaluation report", "",
         f"Images evaluated: {rep['n_images_evaluated']} of {rep['n_images']} | classes: {rep['n_classes']} | "
         f"index: {rep['index_vectors']} vectors ({rep['index_type']}) | stacker: {rep['stacker']} | head: {rep['head']}",
         f"Labels outside the label space: {rep['oov_label_fraction']}", "", rep["protocol"], "",
         "## Sample-wise (per image)", "", "| k | P | R | F1 |", "|---|---|---|---|"]
    for k, v in rep["sample_wise"].items():
        L.append(f"| {k} | {v['P']:.3f} | {v['R']:.3f} | {v['F1']:.3f} |")
    L += ["", f"mAP over categories: {rep['mAP']}", "", "## Macro (per category)", "", "| k | P | R | F1 | categories |", "|---|---|---|---|---|"]
    for k, v in rep["macro"].items():
        L.append(f"| {k} | {v['P']:.3f} | {v['R']:.3f} | {v['F1']:.3f} | {v['n_categories']} |")
    L += ["", "## Head / torso / tail (sample-wise @5)", "", "| bucket | classes | images | P@5 | R@5 | any-hit@10 |", "|---|---|---|---|---|---|"]
    for b, v in rep["frequency_buckets"].items():
        sw = v["sample_wise"][5] if v["sample_wise"] else None
        L.append(f"| {b} | {v['n_classes']} | {v['n_images']} | {sw['P']:.3f} | {sw['R']:.3f} | {v['recall_any_in_top10']:.3f} |" if sw
                 else f"| {b} | {v['n_classes']} | 0 | - | - | - |")
    L += ["", "## Resolution bands (sample-wise @5)", "", "| band | images | P@5 | R@5 |", "|---|---|---|---|"]
    for b, v in rep["resolution_band"].items():
        s = v["sample_wise"][5]
        L.append(f"| {b} | {v['n_images']} | {s['P']:.3f} | {s['R']:.3f} |")
    lt = rep["latency_scoring_only"]
    L += ["", "## Efficiency", "", f"- scoring latency p50 {lt['p50_ms']:.2f} ms, p95 {lt['p95_ms']:.2f} ms ({rep['latency_note']})",
          f"- peak RSS {rep['peak_rss_mb']:.0f} MB; index on disk {rep.get('index_bytes', 'n/a')} bytes"]
    if "encoder_latency" in rep:
        L.append(f"- end-to-end (decode+encode+score) {rep['encoder_latency']}")
    L += ["", f"- category type: {rep['category_type'] if isinstance(rep['category_type'], str) else 'see JSON'}",
          f"- language: {rep['language']}"]
    if "review" in rep:
        L += ["", "## Human-reviewed", "", json.dumps(rep["review"], indent=2)]
    return "\n".join(L) + "\n"


def load_rows(artifacts_dir: Path, split: str) -> List[Dict[str, Any]]:
    from extract_embeddings import ParquetStore
    from manifest import DEFAULT_FILES
    store = ParquetStore(artifacts_dir / DEFAULT_FILES["embeddings"])
    return [r for b in store.iter_batches(2048) for r in b if r["split"] == split]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--category_types", default=None, help="CSV with columns category,type")
    ap.add_argument("--review_csv_out", default=None)
    ap.add_argument("--review_csv_in", default=None)
    ap.add_argument("--review_sample", type=int, default=200)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    art = Path(a.artifacts_dir)
    out_dir = Path(a.out_dir or art / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    kn = load_knowledge(art, encoder=None)
    from manifest import read_manifest
    man = read_manifest(art)
    rows = load_rows(art, a.split)
    types = None
    if a.category_types:
        with open(a.category_types, newline="", encoding="utf-8") as f:
            types = {r["category"]: r["type"] for r in csv.DictReader(f)}
    rep = evaluate_rows(rows, kn, dict(zip(man.classes, man.class_counts)), category_types=types,
                        review_sample=a.review_sample if a.review_csv_out else 0)
    rep["index_bytes"] = sum(p.stat().st_size for p in art.glob("faiss_index*") if p.is_file())
    if a.review_csv_out:
        write_review_template(Path(a.review_csv_out), rep["_review_candidates"])
    rep.pop("_review_candidates", None)
    if a.review_csv_in:
        rep["review"] = apply_review(rep, Path(a.review_csv_in))
    (out_dir / "eval_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (out_dir / "eval_report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))


if __name__ == "__main__":
    main()

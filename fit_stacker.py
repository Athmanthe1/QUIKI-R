"""fit_stacker.py - fit the calibrated logistic-regression fusion on the VALIDATION split (problem statement, section 3).

Runs the real scorers (kNN over the train-only index, text prototypes, offline head, metadata) on val images and fits
sigma(w.[s_knn, s_cls, s_txt, s_meta, missing flags] + b) on the union of per-scorer top-N candidates.
Writes stacker.json next to manifest.json.

    python fit_stacker.py --artifacts_dir artifacts
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from fusion import Stacker, fit_stacker
from scoring import Knowledge, compute_scores, load_knowledge

logger = logging.getLogger("quiki.fit_stacker")


def fit_from_rows(rows: Sequence[Dict[str, Any]], kn: Knowledge, k: int = 10, top_n: int = 200,
                  missing_augment=None, seed: int = 0) -> Stacker:
    classes = set(kn.classes)
    examples = []
    for r in rows:
        truth = set(r["categories"]) & classes
        if not truth:
            continue
        meta = r.get("meta_embedding")
        scores, _nb, _h = compute_scores(np.asarray(r["embedding"], np.float32),
                                         None if meta is None else np.asarray(meta, np.float32), kn, k=k)
        examples.append((scores, truth))
    if not examples:
        raise ValueError("no validation image has a label inside the label space; cannot fit the stacker")
    logger.info("fitting stacker on %d validation images", len(examples))
    return fit_stacker(examples, top_n=top_n, missing_augment=missing_augment or {"meta": 0.3}, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    from evaluate import load_rows
    art = Path(a.artifacts_dir)
    kn = load_knowledge(art, encoder=None)
    kn.stacker = None                                           # fit against raw scorer outputs
    rows = load_rows(art, a.split)
    st = fit_from_rows(rows, kn)
    st.save(art / "stacker.json")
    logger.info("wrote %s", art / "stacker.json")


if __name__ == "__main__":
    main()

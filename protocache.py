"""protocache.py - on-disk cache for text prototypes, so startup does not re-run the text tower."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np

logger = logging.getLogger("CommonsProtoCache")


def load_or_compute_prototypes(cache_dir: Union[str, Path], names: Sequence[str],
                               compute: Callable[[Sequence[str]], np.ndarray], model_name: str,
                               incremental: bool = False) -> np.ndarray:
    """Return an array [len(names), D]. Cache key = model name + exact class list (order matters).

    With ``incremental=True`` rows of already-cached names are reused and only new names are computed
    (new categories become available without re-embedding the rest).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr_path, meta_path = cache_dir / "text_prototypes.npy", cache_dir / "text_prototypes.meta.json"
    names = list(names)
    cached_names, cached = None, None
    if arr_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("model_name") == model_name:
                cached_names, cached = meta["names"], np.load(arr_path)
                if len(cached_names) != len(cached):
                    cached_names = cached = None
        except Exception as exc:
            logger.warning("prototype cache unreadable (%s); recomputing", exc)
    if cached is not None and cached_names == names:
        logger.info("Text prototypes loaded from cache (%d classes)", len(names))
        return cached
    if incremental and cached is not None:
        have = {n: i for i, n in enumerate(cached_names)}
        missing = [n for n in names if n not in have]
        new_rows = compute(missing) if missing else np.zeros((0, cached.shape[1]), cached.dtype)
        new_map = {n: new_rows[i] for i, n in enumerate(missing)}
        out = np.stack([cached[have[n]] if n in have else new_map[n] for n in names]) if names else cached[:0]
    else:
        out = np.asarray(compute(names))
    np.save(arr_path, out)
    meta_path.write_text(json.dumps({"model_name": model_name, "names": names}, ensure_ascii=False), encoding="utf-8")
    return out

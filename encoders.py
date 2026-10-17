"""encoders.py - the encoder boundary. Everything else talks to an ``Encoder``; only SigLIPEncoder needs torch.

* SigLIPEncoder : the frozen google/siglip-base-patch16-224 (768-d). torch/transformers are imported lazily.
* FakeEncoder   : deterministic, dependency-free stand-in for unit tests and the CI smoke test. It is NOT a
                  vision-language model: text and image embeddings are unrelated, so zero-shot text scores
                  from it are meaningless. It only lets the pipeline stages be exercised end to end.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence

import numpy as np

from manifest import SIGLIP_EMBED_DIM, SIGLIP_MODEL_NAME

TEMPLATES = ("a photo of {}.", "an image of {}.", "{}.")


def bare_name(category: str) -> str:
    """Category name as natural text: no 'Category:' prefix, underscores -> spaces."""
    c = category.strip()
    if c.lower().startswith("category:"):
        c = c[len("category:"):]
    return re.sub(r"\s+", " ", c.replace("_", " ")).strip()


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.where(n == 0, 1.0, n)).astype(np.float32)


def make_text_prototypes(encoder, names: Sequence[str]) -> np.ndarray:
    """One L2-normalised prototype per name: mean of the embeddings of 3 prompt templates."""
    if not len(names):
        return np.zeros((0, encoder.embed_dim), np.float32)
    texts = [t.format(bare_name(n)) for n in names for t in TEMPLATES]
    emb = np.asarray(encoder.encode_texts(texts), dtype=np.float32).reshape(len(names), len(TEMPLATES), -1)
    return _l2(emb.mean(axis=1))


class SigLIPEncoder:
    def __init__(self, model_name: str = SIGLIP_MODEL_NAME, device: Optional[str] = None):
        import torch
        from transformers import AutoModel, AutoProcessor
        from common import extract_tensor
        self._torch, self._extract = torch, extract_tensor
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.embed_dim = int(self.model.config.vision_config.hidden_size)
        self.logit_scale = float(self.model.logit_scale.detach().cpu().item())      # log-scale parameter
        self.logit_bias = float(self.model.logit_bias.detach().cpu().item())

    def encode_images(self, images: Sequence) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._extract(self.model.get_image_features(**inputs))
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        # truncation=True: long descriptions must be cut to the text tower's max length, not crash.
        inputs = self.processor(text=list(texts), return_tensors="pt", padding="max_length",
                                truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            feats = self._extract(self.model.get_text_features(**inputs))
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype(np.float32)


class FakeEncoder:
    """Deterministic test double (see module docstring)."""
    model_name = "fake-encoder"

    def __init__(self, dim: int = 64, logit_scale: float = float(np.log(10.0)), logit_bias: float = -5.0):
        self.embed_dim, self.logit_scale, self.logit_bias = dim, logit_scale, logit_bias
        self._proj = np.random.default_rng(1234).normal(size=(8 * 8 * 3, dim)).astype(np.float32)

    def encode_images(self, images: Sequence) -> np.ndarray:
        feats = []
        for im in images:
            a = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32).ravel() / 255.0
            feats.append((a - 0.5) @ self._proj)
        return _l2(np.stack(feats))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.embed_dim), np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                seed = int.from_bytes(hashlib.sha1(tok.encode()).digest()[:4], "little")
                out[i] += np.random.default_rng(seed).normal(size=self.embed_dim)
        return _l2(out)


def get_encoder(name: Optional[str] = None):
    """``fake`` (or env QUIKI_ENCODER=fake) -> FakeEncoder; anything else -> the real SigLIPEncoder."""
    import os
    name = name or os.environ.get("QUIKI_ENCODER", "siglip")
    return FakeEncoder() if name == "fake" else SigLIPEncoder()

"""common.py - code shared by extract_embeddings.py, trainer.py and engine.py.

Single definitions of the model constants, tensor extraction and the supervised head, so the
stages cannot drift apart. (The manifest contract lives in the torch-free manifest.py and is
re-exported here for convenience.)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from manifest import (DEFAULT_FILES, MANIFEST_NAME, SIGLIP_EMBED_DIM, SIGLIP_MODEL_NAME,  # noqa: F401
                      Manifest, ManifestError, load_head_npz, read_manifest, save_head_npz, write_manifest)

MODEL_NAME = SIGLIP_MODEL_NAME
EMBED_DIM = SIGLIP_EMBED_DIM


def extract_tensor(output: Any) -> torch.Tensor:
    """Return a feature tensor from whatever get_*_features returned (tensor or ModelOutput)."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    try:
        return output[0]
    except Exception:
        raise TypeError(f"Unexpected model output type: {type(output)}")


POS, UNK, NEG = 1.0, 0.0, -1.0     # three-state target encoding


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss, PDF section 3, with explicit three-state targets.

    ``targets`` has the same shape as ``logits`` with values
        +1  positive            -> -(1-p)^gamma_pos * log p
         0  unknown (unlabelled)-> -p_m^gamma_neg * log(1 - p_m),   p_m = max(p - margin, 0)
        -1  explicit reject     -> -log(1 - p)        (full strength: no margin, no focal weight)
    Plain multi-hot {0, 1} targets are valid input and mean positive / unknown (a missing label is
    NOT a hard negative). ``unknown_weight=0`` ignores unknowns entirely. Reduction: sum over
    classes, mean over the batch.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, margin: float = 0.05,
                 unknown_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin = margin
        self.unknown_weight = unknown_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        pos_loss = -((1.0 - p) ** self.gamma_pos) * torch.nn.functional.logsigmoid(logits)
        p_m = torch.clamp(p - self.margin, min=0.0)
        unk_loss = -(p_m ** self.gamma_neg) * torch.log(torch.clamp(1.0 - p_m, min=self.eps))
        neg_loss = -torch.nn.functional.logsigmoid(-logits)          # == -log(1 - p), numerically stable
        loss = torch.where(targets > 0.5, pos_loss,
                           torch.where(targets < -0.5, neg_loss, self.unknown_weight * unk_loss))
        return loss.sum(dim=-1).mean()


class WikimediaClassifier(nn.Module):
    """Linear head, optionally initialised with text prototypes. ``forward`` returns probabilities."""

    def __init__(self, embed_dim: int, num_classes: int, text_prototypes: Optional[np.ndarray] = None):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes, bias=True)
        if text_prototypes is not None:
            with torch.no_grad():
                self.fc.weight.copy_(torch.as_tensor(text_prototypes, dtype=torch.float32))
                self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))

    def to_arrays(self):
        """(W [C, D], b [C]) as float32 numpy arrays, for the torch-free ``.npz`` head format."""
        return (self.fc.weight.detach().cpu().numpy().astype("float32"),
                self.fc.bias.detach().cpu().numpy().astype("float32"))

    @classmethod
    def from_arrays(cls, W: np.ndarray, b: np.ndarray) -> "WikimediaClassifier":
        head = cls(W.shape[1], W.shape[0])
        with torch.no_grad():
            head.fc.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
            head.fc.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
        return head

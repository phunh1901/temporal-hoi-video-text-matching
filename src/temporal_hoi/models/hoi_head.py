"""Residual frozen-feature HOI heads with positive-only supervision.

Unannotated predicate entries do not become negative labels. The text encoder,
visual encoder and anchor features are detached at the training boundary.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

MODES = ("object", "union", "fusion")


def _features(values: torch.Tensor, name: str) -> torch.Tensor:
    if (values.ndim != 2 or not values.shape[0] or not values.shape[1]
            or not torch.isfinite(values).all() or not (values.norm(dim=-1) > 0).all()):
        raise ValueError(f"{name} requires finite nonzero [N,D] features")
    return values.detach()


def view_features(object_features: torch.Tensor, union_features: torch.Tensor,
                  mode: str) -> torch.Tensor:
    """Fixed object/union views or the normalized equal-weight fusion."""
    obj = _features(object_features, "object")
    union = _features(union_features, "union")
    if obj.shape != union.shape:
        raise ValueError("Object and union feature shapes differ")
    if mode == "object":
        return obj
    if mode == "union":
        return union
    if mode != "fusion":
        raise ValueError(f"mode must be one of {MODES}")
    fused = (F.normalize(obj, dim=-1) + F.normalize(union, dim=-1)) / 2
    if not (fused.norm(dim=-1) > 0).all():
        raise ValueError("Object and union features cancel in fixed fusion")
    return F.normalize(fused, dim=-1)


def identity_projection(features: torch.Tensor) -> torch.Tensor:
    """Use exactly the same normalization operations as the zero-residual head."""
    return F.normalize(F.normalize(_features(features, "base"), dim=-1), dim=-1)


class HOIResidualHead(nn.Module):
    """Equal-capacity D -> 128 -> D GELU residual; zero final layer starts at identity."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim < 1 or hidden_dim != 128:
            raise ValueError("Require positive embedding_dim and fixed hidden_dim=128")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.residual = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = _features(features, "base")
        if values.shape[1] != self.embedding_dim:
            raise ValueError("Head input embedding dimension differs")
        base = F.normalize(values, dim=-1)
        projected = base + self.residual(base)
        if not torch.isfinite(projected).all() or not (projected.norm(dim=-1) > 0).all():
            raise ValueError("Residual head produced nonfinite or zero embeddings")
        return F.normalize(projected, dim=-1)


def positive_anchor_loss(
    projected: torch.Tensor, base: torch.Tensor, object_class: torch.Tensor,
    positive_mask: torch.Tensor, text_features: torch.Tensor,
    candidate_mask: torch.Tensor, *, anchor_weight: float = 1.0,
) -> dict:
    """Average positive cosine loss per labeled pair and anchor loss over all pairs.

    Other class columns never enter a softmax denominator or a negative target.
    A batch without annotated positives still has an anchor objective.
    """
    if (not isinstance(anchor_weight, (int, float)) or isinstance(anchor_weight, bool)
            or not math.isfinite(anchor_weight) or anchor_weight < 0):
        raise ValueError("anchor_weight must be finite and nonnegative")
    if projected.ndim != 2 or not projected.shape[0] or not torch.isfinite(projected).all():
        raise ValueError("projected requires finite nonempty [N,D] features")
    frozen_base = _features(base, "anchor")
    if frozen_base.shape != projected.shape:
        raise ValueError("Anchor and projected shapes differ")
    if (text_features.ndim != 3 or text_features.shape[-1] != projected.shape[-1]
            or not torch.isfinite(text_features).all()):
        raise ValueError("text_features require finite [objects,predicates,D]")
    if candidate_mask.dtype != torch.bool or candidate_mask.shape != text_features.shape[:2]:
        raise ValueError("candidate_mask must be bool [objects,predicates]")
    if (positive_mask.dtype != torch.bool
            or positive_mask.shape != (len(projected), text_features.shape[1])):
        raise ValueError("positive_mask must be bool [N,predicates]")
    if (object_class.ndim != 1 or len(object_class) != len(projected)
            or object_class.dtype not in (torch.int32, torch.int64)
            or (object_class < 0).any() or (object_class >= text_features.shape[0]).any()):
        raise ValueError("object_class must contain valid integer object IDs")
    objects = object_class.detach().long()
    allowed = candidate_mask.detach()[objects]
    if (positive_mask & ~allowed).any():
        raise ValueError("Observed positive is excluded by candidate_mask")
    rows, predicates = positive_mask.nonzero(as_tuple=True)
    counts = positive_mask.sum(dim=1)
    labeled = counts > 0
    if len(rows):
        text = F.normalize(text_features.detach()[objects[rows], predicates], dim=-1)
        if not (text.norm(dim=-1) > 0).all():
            raise ValueError("Positive text prototype is zero")
        losses = 1 - (projected[rows] * text).sum(dim=-1).clamp(-1, 1)
        per_pair = projected.new_zeros(len(projected)).scatter_add(0, rows, losses)
        positive = (per_pair[labeled] / counts[labeled]).mean()
    else:
        positive = projected.sum() * 0
    anchor = (1 - (projected * F.normalize(frozen_base, dim=-1)).sum(dim=-1).clamp(-1, 1)).mean()
    return {"loss": positive + anchor_weight * anchor, "positive_loss": positive,
            "anchor_loss": anchor, "labeled_pairs": int(labeled.sum().item()),
            "positive_labels": len(rows), "all_pairs": len(projected)}

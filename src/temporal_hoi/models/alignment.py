"""Small shared-space projection; backbone features remain frozen."""

import torch
from torch import nn
from torch.nn import functional as F


class LinearAlignmentHead(nn.Module):
    """Separate bias-free projections followed by L2 normalization.

    Square projections start at identity, retaining the encoder's initial
    cosine geometry. Input dimensions come from the cache rather than a fixed
    assumption about a particular encoder.
    """

    def __init__(self, video_dim: int, text_dim: int, output_dim: int | None = None):
        super().__init__()
        output_dim = video_dim if output_dim is None else output_dim
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
               for n in (video_dim, text_dim, output_dim)):
            raise ValueError("Projection dimensions must be positive integers")
        self.video_dim, self.text_dim, self.output_dim = video_dim, text_dim, output_dim
        self.video_projection = nn.Linear(video_dim, output_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, output_dim, bias=False)
        for layer in (self.video_projection, self.text_projection):
            if layer.in_features == layer.out_features:
                nn.init.eye_(layer.weight)

    @staticmethod
    def _project(features, layer):
        if (features.ndim != 2 or features.shape[1] != layer.in_features
                or not torch.isfinite(features).all()
                or not (features.norm(dim=-1) > 0).all()):
            raise ValueError("Features require finite nonzero vectors of the declared dimension")
        # Detaching makes the frozen-feature boundary explicit even for callers
        # supplying a tensor with requires_grad=True.
        projected = layer(F.normalize(features.detach(), dim=-1))
        if not torch.isfinite(projected).all() or not (projected.norm(dim=-1) > 0).all():
            raise ValueError("Head produced a nonfinite or zero embedding")
        return F.normalize(projected, dim=-1)

    def encode_video(self, features):
        return self._project(features, self.video_projection)

    def encode_text(self, features):
        return self._project(features, self.text_projection)

    def forward(self, video_features, text_features):
        return self.encode_video(video_features) @ self.encode_text(text_features).T


# Public name used by the project SOP; retain the shorter API spelling too.
VideoTextLinearHead = LinearAlignmentHead


class MLPAlignmentHead(LinearAlignmentHead):
    """Two independent D -> hidden -> output projections with GELU, no dropout.

    This capacity control consumes the same pooled features as the linear
    head. It adds neither geometry nor temporal information. All layers use
    PyTorch's seeded default initialization; inputs remain detached and L2
    normalized, and output vectors are L2 normalized as in the linear head.
    """

    def __init__(self, video_dim, text_dim, hidden_dim, output_dim=None):
        super().__init__(video_dim, text_dim, output_dim)
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    @staticmethod
    def _project(features, layer):
        if (features.ndim != 2 or features.shape[1] != layer[0].in_features
                or not torch.isfinite(features).all()
                or not (features.norm(dim=-1) > 0).all()):
            raise ValueError("Features require finite nonzero vectors of the declared dimension")
        projected = layer(F.normalize(features.detach(), dim=-1))
        if not torch.isfinite(projected).all() or not (projected.norm(dim=-1) > 0).all():
            raise ValueError("Head produced a nonfinite or zero embedding")
        return F.normalize(projected, dim=-1)


def build_alignment_head(config, video_dim, text_dim):
    """Construct a head without changing the caller's seed or backbone."""
    if config["head"] == "linear":
        return LinearAlignmentHead(video_dim, text_dim)
    if config["head"] == "mlp":
        return MLPAlignmentHead(video_dim, text_dim, config["hidden_dim"])
    raise ValueError("head must be linear or mlp")

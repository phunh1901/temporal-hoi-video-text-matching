"""Frozen-feature retrieval, reproducible evaluation, and a small optional head.

Cache contract: {text: FloatTensor[T,D], frames: FloatTensor[V,F,D],
mask: BoolTensor[V,F], text_video_ids: list[str], video_ids: list[str],
text_ids: list[str], metadata: dict}. Features may be raw and are normalized here.

score_matrix returns CPU float32 numpy [T,V]. B0 = per-frame-normalized
mean pooling; B1 = query-weighted pooling; B2 = (1-alpha)*B0 + alpha*B1.
With a TinyAttentionHead, method B2 is the separately reported trained B3.
Metrics use explicit identities, a fixed full gallery, and stable candidate-order ties.
Only train/validation caches enter train_attention_head; test scores never do.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _as_float_tensor(value: Any, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32).detach()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def validate_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Validate identities, shapes, valid frames, and nonzero embeddings."""
    required = ("text", "frames", "mask", "text_video_ids", "video_ids", "text_ids")
    absent = [key for key in required if key not in cache]
    if absent:
        raise ValueError(f"Cache is missing keys: {absent}")
    text = _as_float_tensor(cache["text"], "text")
    frames = _as_float_tensor(cache["frames"], "frames")
    mask = torch.as_tensor(cache["mask"], dtype=torch.bool)
    if text.ndim != 2 or frames.ndim != 3:
        raise ValueError("Expected text[T,D] and frames[V,F,D]")
    if min(text.shape) == 0 or min(frames.shape) == 0:
        raise ValueError("Empty features are not evaluable")
    if text.shape[-1] != frames.shape[-1] or tuple(mask.shape) != tuple(frames.shape[:2]):
        raise ValueError("Feature dimensions or frame mask shape do not match")
    text_video_ids = [str(x) for x in cache["text_video_ids"]]
    video_ids = [str(x) for x in cache["video_ids"]]
    text_ids = [str(x) for x in cache["text_ids"]]
    if len(text_video_ids) != len(text) or len(text_ids) != len(text):
        raise ValueError("Text identity lengths do not match text features")
    if len(video_ids) != len(frames):
        raise ValueError("video_ids length does not match frame features")
    if len(set(video_ids)) != len(video_ids) or len(set(text_ids)) != len(text_ids):
        raise ValueError("video_ids and text_ids must each be unique")
    if any(not x for x in video_ids + text_ids + text_video_ids):
        raise ValueError("Empty identities are not allowed")
    if not set(text_video_ids).issubset(video_ids):
        raise ValueError("Some captions have no ground-truth candidate video")
    if set(video_ids) != set(text_video_ids):
        raise ValueError("Every candidate video needs at least one ground-truth caption")
    if not mask.any(dim=1).all():
        raise ValueError("A video has no valid decoded frame; repair the cache first")
    if (text.norm(dim=-1) <= 1e-12).any():
        raise ValueError("A text embedding has zero norm")
    if (frames.norm(dim=-1)[mask] <= 1e-12).any():
        raise ValueError("A valid frame embedding has zero norm")
    return dict(
        cache,
        text=text,
        frames=frames,
        mask=mask,
        text_video_ids=text_video_ids,
        video_ids=video_ids,
        text_ids=text_ids,
    )


def direct_meanp_reference(text: Any, frames: Any, mask: Any) -> np.ndarray:
    """Independent, unchunked CPU B0 reference; no learned CLIP logit scale."""
    q = _as_float_tensor(text, "text").cpu()
    f = _as_float_tensor(frames, "frames").cpu()
    m = torch.as_tensor(mask, dtype=torch.float32).cpu()
    if q.ndim != 2 or f.ndim != 3 or tuple(m.shape) != tuple(f.shape[:2]):
        raise ValueError("Reference expects text[T,D], frames[V,F,D], mask[V,F]")
    if not (m.sum(1) > 0).all():
        raise ValueError("Reference cannot pool a video with zero valid frames")
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = (f * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (q @ v.T).numpy()


class TinyAttentionHead(nn.Module):
    """Learn weights only: normalized residual query transform + temperature.

    Zero initialization exactly reproduces the fixed attention formula initially.
    The final dot product always uses the original normalized text feature.
    """

    def __init__(self, dim: int, tau: float = 0.05):
        super().__init__()
        if dim < 1 or not 0.005 <= tau <= 1.0:
            raise ValueError("dim must be positive and head tau must be in [.005, 1]")
        self.dim = int(dim)
        self.query_delta = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.zeros_(self.query_delta.weight)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau), dtype=torch.float32))

    def weighting_query(self, q: torch.Tensor) -> torch.Tensor:
        return F.normalize(q + self.query_delta(q), dim=-1)

    def temperature(self) -> torch.Tensor:
        return self.log_tau.clamp(math.log(0.005), math.log(1.0)).exp()


def _score_block(
    q: torch.Tensor,
    f: torch.Tensor,
    m: torch.Tensor,
    *,
    method: str,
    tau: float,
    alpha: float,
    head: TinyAttentionHead | None = None,
) -> torch.Tensor:
    """Differentiable score block; inputs have already been normalized."""
    mean = F.normalize((f * m[..., None]).sum(1) / m.sum(1, keepdim=True), dim=-1)
    base = q @ mean.T
    if method == "B0":
        return base
    weighting_query = q if head is None else head.weighting_query(q)
    temperature = tau if head is None else head.temperature()
    logits = torch.einsum("qd,vfd->qvf", weighting_query, f) / temperature
    weights = logits.masked_fill(~m[None], -torch.inf).softmax(dim=-1)
    pooled = F.normalize(torch.einsum("qvf,vfd->qvd", weights, f), dim=-1)
    attention = torch.einsum("qd,qvd->qv", q, pooled)
    return attention if method == "B1" else (1.0 - alpha) * base + alpha * attention


def score_matrix(
    text: Any,
    frames: Any,
    mask: Any,
    *,
    method: str = "B0",
    tau: float = 0.05,
    alpha: float = 0.25,
    query_chunk: int = 32,
    video_chunk: int = 32,
    device: str | torch.device | None = None,
    head: TinyAttentionHead | None = None,
) -> np.ndarray:
    """Score every query against every candidate without truncating the gallery.

    Chunked FP32 with autocast disabled bounds intermediate [Q,V,D] memory.
    Encoder features stay on CPU between chunks; CUDA OOM is retried by caller
    with smaller chunks, rather than silently deleting candidates.
    """
    if method not in {"B0", "B1", "B2"}:
        raise ValueError("method must be B0, B1, or B2 (B3 uses B2 with head=...)")
    if not math.isfinite(tau) or tau <= 0 or not 0 <= alpha <= 1:
        raise ValueError("tau must be finite and positive; alpha must be in [0,1]")
    if query_chunk < 1 or video_chunk < 1:
        raise ValueError("Chunk sizes must be positive")
    q_cpu = _as_float_tensor(text, "text").cpu()
    f_cpu = _as_float_tensor(frames, "frames").cpu()
    m_cpu = torch.as_tensor(mask, dtype=torch.bool).cpu()
    if q_cpu.ndim != 2 or f_cpu.ndim != 3 or q_cpu.shape[-1] != f_cpu.shape[-1]:
        raise ValueError("Expected compatible text[T,D] and frames[V,F,D]")
    if min(q_cpu.shape) == 0 or min(f_cpu.shape) == 0:
        raise ValueError("Empty features are not evaluable")
    if tuple(m_cpu.shape) != tuple(f_cpu.shape[:2]) or not m_cpu.any(dim=1).all():
        raise ValueError("Frame mask mismatch or video without valid frames")
    if (q_cpu.norm(dim=-1) <= 1e-12).any() or (f_cpu.norm(dim=-1)[m_cpu] <= 1e-12).any():
        raise ValueError("Zero-norm embeddings are invalid")
    # Keep the accepted baseline's arithmetic: per-video CPU pooling, per-text CPU
    # normalization, then a single NumPy FP32 matrix multiplication. A GPU/chunked
    # replacement can perturb exact ties even when individual scores are close.
    native_q = torch.stack([row / row.norm() for row in q_cpu]).numpy()
    native_v = []
    for features, valid in zip(f_cpu, m_cpu, strict=True):
        normalized = features[valid] / features[valid].norm(dim=-1, keepdim=True)
        pooled = normalized.mean(dim=0)
        if pooled.norm() <= 1e-12:
            raise ValueError("Mean pooled video has zero norm")
        native_v.append((pooled / pooled.norm()).numpy())
    native_base = native_q @ np.asarray(native_v, dtype=np.float32).T
    if method == "B0":
        return native_base
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if head is not None and head.dim != q_cpu.shape[-1]:
        raise ValueError("Attention head dimension does not match cached embeddings")
    previous_device = next(head.parameters()).device if head is not None else None
    previous_training = head.training if head is not None else None
    result = np.empty((len(q_cpu), len(f_cpu)), dtype=np.float32)
    try:
        if head is not None:
            head.to(selected).eval()
        with torch.inference_mode(), torch.autocast(device_type=selected.type, enabled=False):
            for qi in range(0, len(q_cpu), query_chunk):
                q = F.normalize(q_cpu[qi : qi + query_chunk].to(selected), dim=-1)
                for vi in range(0, len(f_cpu), video_chunk):
                    f = F.normalize(f_cpu[vi : vi + video_chunk].to(selected), dim=-1)
                    m = m_cpu[vi : vi + video_chunk].to(selected)
                    block = _score_block(q, f, m, method="B1", tau=tau, alpha=alpha, head=head)
                    result[qi : qi + len(q), vi : vi + len(f)] = block.cpu().numpy()
    finally:
        if head is not None:
            head.to(previous_device).train(previous_training)
    if method == "B2":
        result = (1.0 - alpha) * native_base + alpha * result
    if not np.isfinite(result).all():
        raise ValueError("Scoring produced non-finite values")
    return result


def _metric_summary(ranks: np.ndarray) -> dict[str, float | int]:
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 1 or len(ranks) == 0 or (ranks < 1).any():
        raise ValueError("Ranks must be a nonempty vector of positive integers")
    return {
        "R@1": float(100 * np.mean(ranks <= 1)),
        "R@5": float(100 * np.mean(ranks <= 5)),
        "R@10": float(100 * np.mean(ranks <= 10)),
        "MedR": float(np.median(ranks)),
        "MeanR": float(np.mean(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
        "NumQueries": int(len(ranks)),
    }


def _ranks(
    scores: np.ndarray,
    query_gt_ids: Sequence[str],
    candidate_gt_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    if len(candidate_ids) != scores.shape[1]:
        raise ValueError("Candidate IDs differ from score columns")
    gt_candidates = np.asarray([str(x) for x in candidate_gt_ids], dtype=str)
    positive_positions: dict[str, np.ndarray] = {}
    for gt in set(query_gt_ids):
        positive_positions[str(gt)] = np.flatnonzero(gt_candidates == str(gt))
        if not len(positive_positions[str(gt)]):
            raise ValueError(f"No positive candidate for {gt}")
    ranks = np.empty(len(query_gt_ids), dtype=np.int64)
    tie_at_best_positive = 0
    ties_at_top = 0
    for i, gt in enumerate(query_gt_ids):
        # Preserve the locked candidate order on exact score ties.
        order = np.argsort(-scores[i], kind="stable")
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        pos = positive_positions[str(gt)]
        ranks[i] = int(inverse[pos].min()) + 1
        best_positive_score = scores[i, pos].max()
        tie_at_best_positive += int(np.count_nonzero(scores[i] == best_positive_score) > 1)
        ties_at_top += int(np.count_nonzero(scores[i] == scores[i].max()) > 1)
    return ranks, {
        "queries_with_tie_at_best_positive": tie_at_best_positive,
        "queries_with_top_score_tie": ties_at_top,
    }


def evaluate_bidirectional(
    scores: Any, text_video_ids: Sequence[str], video_ids: Sequence[str], text_ids: Sequence[str]
) -> dict[str, Any]:
    """Multi-positive best-positive ranks, candidate-order ties, both directions."""
    s = np.asarray(scores, dtype=np.float32)
    tvid = [str(x) for x in text_video_ids]
    vids = [str(x) for x in video_ids]
    tids = [str(x) for x in text_ids]
    if s.shape != (len(tvid), len(vids)) or len(tids) != len(tvid) or not np.isfinite(s).all():
        raise ValueError("Score shape, identities, or finite-value check failed")
    if not len(tvid) or not len(vids) or len(set(vids)) != len(vids) or len(set(tids)) != len(tids):
        raise ValueError("Candidate identities must be unique and nonempty")
    if set(vids) != set(tvid):
        raise ValueError("Ground-truth videos and candidate videos do not match")
    tr, tt = _ranks(s, tvid, vids, vids)
    vr, vt = _ranks(s.T, vids, tvid, tids)
    return {
        "text_to_video": {
            **_metric_summary(tr),
            "NumCandidates": len(vids),
            "NumPositivePairs": len(tvid),
        },
        "video_to_text": {
            **_metric_summary(vr),
            "NumCandidates": len(tvid),
            "NumPositivePairs": len(tvid),
        },
        "ranks_text_to_video": tr,
        "ranks_video_to_text": vr,
        "ties_text_to_video": tt,
        "ties_video_to_text": vt,
        "tie_rule": "descending FP32 score; stable input candidate order on exact ties",
    }


def original_clip4clip_metrics(scores_diagonal: Any) -> dict[str, Any]:
    """Reproduce original CLIP4Clip compute_metrics for a one-caption square split.

    Caller must first align rows/columns by GT identity. The original evaluator
    counts every tie at the diagonal score, so legacy denominator can exceed Q.
    Report this separately from ID-stable evaluation, never silently reconcile it.
    """
    s = np.asarray(scores_diagonal)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not len(s) or not np.isfinite(s).all():
        raise ValueError("Original evaluator requires a finite, nonempty square matrix")
    sorted_negative = np.sort(-s, axis=1)
    diagonal = np.diag(-s)[:, None]
    matches = np.where(sorted_negative - diagonal == 0)
    zero_based = matches[1]
    out = _metric_summary(zero_based + 1)
    per_query_ties = np.bincount(matches[0], minlength=len(s))
    out.update(
        original_num_rank_entries=int(len(zero_based)),
        original_num_queries=int(len(s)),
        queries_with_ground_truth_score_ties=int(np.count_nonzero(per_query_ties > 1)),
        warning="Original evaluator expands tied diagonal scores; denominator is rank entries.",
    )
    return out


def aligned_original_metrics(
    scores: Any, text_video_ids: Sequence[str], video_ids: Sequence[str]
) -> dict[str, Any]:
    """Align one-caption-per-video by ID, then run original evaluator twice."""
    tv = [str(x) for x in text_video_ids]
    vv = [str(x) for x in video_ids]
    if len(tv) != len(vv) or len(set(tv)) != len(tv) or set(tv) != set(vv):
        return {
            "available": False,
            "reason": "Original diagonal evaluator requires exactly one caption per video.",
        }
    s = np.asarray(scores)
    if s.shape != (len(tv), len(vv)):
        raise ValueError("Score shape does not match identities")
    lookup = {vid: i for i, vid in enumerate(vv)}
    aligned = s[:, [lookup[vid] for vid in tv]]
    return {
        "available": True,
        "text_to_video": original_clip4clip_metrics(aligned),
        "video_to_text": original_clip4clip_metrics(aligned.T),
    }


def paired_bootstrap(
    base_ranks: Any,
    new_ranks: Any,
    cluster_ids: Sequence[str],
    *,
    n_bootstrap: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired video-cluster bootstrap of precomputed full-gallery rank outcomes.

    Does not resample or shrink the candidate gallery. Delta recalls are pp;
    delta MRR is in [0,1] units. Unequal caption counts retain cluster weights.
    """
    base = np.asarray(base_ranks, dtype=np.int64)
    new = np.asarray(new_ranks, dtype=np.int64)
    ids = [str(x) for x in cluster_ids]
    if base.ndim != 1 or base.shape != new.shape or len(base) != len(ids) or not len(base):
        raise ValueError("Paired ranks and cluster identities must have matching nonempty lengths")
    if (base < 1).any() or (new < 1).any() or n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid ranks, bootstrap count, or confidence")
    unique = sorted(set(ids))
    index = {key: i for i, key in enumerate(unique)}
    counts = np.zeros(len(unique), dtype=np.int64)
    deltas = np.zeros((len(unique), 4), dtype=np.float64)
    for i, key in enumerate(ids):
        j = index[key]
        counts[j] += 1
        deltas[j] += [100 * (float(new[i] <= k) - float(base[i] <= k)) for k in (1, 5, 10)] + [
            1 / new[i] - 1 / base[i]
        ]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, 4), dtype=np.float64)
    for b in range(n_bootstrap):
        selected = rng.integers(0, len(unique), size=len(unique))
        draws[b] = deltas[selected].sum(0) / counts[selected].sum()
    low, high = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    point = deltas.sum(0) / counts.sum()
    return {
        "clusters": len(unique),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "confidence": confidence,
        "gallery": "fixed; ranks computed before bootstrap",
        "delta": {
            name: {
                "estimate": float(point[i]),
                "ci_low": float(np.quantile(draws[:, i], low)),
                "ci_high": float(np.quantile(draws[:, i], high)),
            }
            for i, name in enumerate(("R@1_pp", "R@5_pp", "R@10_pp", "MRR"))
        },
    }


def compare_runs(
    base: Mapping[str, Any],
    new: Mapping[str, Any],
    text_video_ids: Sequence[str],
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    *,
    n_bootstrap: int = 2000,
    seed: int = 2026,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Count corrected/broken R@1 queries; optionally export every changed rank."""
    records = []
    result: dict[str, Any] = {}
    for direction, identities, clusters in (
        ("text_to_video", text_ids, text_video_ids),
        ("video_to_text", video_ids, video_ids),
    ):
        before = np.asarray(base[f"ranks_{direction}"], dtype=np.int64)
        after = np.asarray(new[f"ranks_{direction}"], dtype=np.int64)
        if before.shape != after.shape or len(before) != len(identities):
            raise ValueError("Run ranks and identities do not match")
        corrected = (before > 1) & (after == 1)
        broken = (before == 1) & (after > 1)
        result[direction] = {
            "corrected_at_1": int(corrected.sum()),
            "broken_at_1": int(broken.sum()),
            "unchanged_at_1": int(len(before) - corrected.sum() - broken.sum()),
            "bootstrap": paired_bootstrap(
                before, after, clusters, n_bootstrap=n_bootstrap, seed=seed
            ),
        }
        for i in np.flatnonzero(before != after):
            records.append(
                {
                    "direction": direction,
                    "query_id": str(identities[i]),
                    "gt_video_id": str(clusters[i]),
                    "rank_before": int(before[i]),
                    "rank_after": int(after[i]),
                    "r1_change": "corrected"
                    if corrected[i]
                    else "broken"
                    if broken[i]
                    else "unchanged",
                }
            )
    if output_csv is not None:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "direction",
                    "query_id",
                    "gt_video_id",
                    "rank_before",
                    "rank_after",
                    "r1_change",
                ),
            )
            writer.writeheader()
            writer.writerows(records)
    result["changed_queries"] = records
    return result


def multi_positive_contrastive_loss(
    scores: torch.Tensor, positives: torch.Tensor, logit_scale: float = 20.0
) -> torch.Tensor:
    """Symmetric log probability of any positive, valid for rectangular batches."""
    if (
        scores.ndim != 2
        or positives.shape != scores.shape
        or not math.isfinite(logit_scale)
        or logit_scale <= 0
    ):
        raise ValueError("Loss expects [queries,videos] scores and a matching positive mask")
    positives = positives.to(device=scores.device, dtype=torch.bool)
    if not positives.any(0).all() or not positives.any(1).all():
        raise ValueError("Every training query/candidate must have a positive")
    logits = scores.float() * logit_scale
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    text_loss = logits.logsumexp(1) - positive_logits.logsumexp(1)
    video_loss = logits.logsumexp(0) - positive_logits.logsumexp(0)
    return (text_loss.mean() + video_loss.mean()) / 2


def _cache_fingerprint(cache: Mapping[str, Any]) -> str:
    """Bind features/identities and immutable semantics, excluding cache location.

    Extraction metadata also carries elapsed seconds, cache namespace, and input
    paths. Those observations can change when identical shards are reused on a
    new Kaggle session. Only semantic provenance below belongs in resume identity.
    Both settings and the upstream fingerprint are retained so changing either
    explicitly refuses resume, even when tensor bytes happen to be unchanged.
    """
    h = hashlib.sha256()
    for key in ("text", "frames", "mask"):
        value = cache[key].detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    h.update(
        json.dumps(
            {key: cache[key] for key in ("text_video_ids", "video_ids", "text_ids")},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    )
    metadata = cache.get("metadata", {})
    immutable_keys = (
        "fingerprint",
        "cache_fingerprint",
        "cache_hash",
        "settings",
        "split",
        "encoder_training_overlap",
        "development_status",
        "records_sha256",
        "video_inputs_sha256",
        "input_sha256",
    )
    semantic_metadata = {key: metadata[key] for key in immutable_keys if key in metadata}
    h.update(
        json.dumps(semantic_metadata, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    )
    return h.hexdigest()


def _atomic_torch_save(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_attention_head(
    checkpoint_path: str | Path, *, device: str | torch.device = "cpu"
) -> TinyAttentionHead:
    """Load this module's small head checkpoint without unpickling arbitrary code."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if payload.get("format") != "msrvtt_tiny_attention_v1":
        raise ValueError("Not a supported tiny attention checkpoint")
    head = TinyAttentionHead(payload["config"]["dim"], payload["config"]["tau"])
    head.load_state_dict(payload["model"])
    return head.to(device).eval()


def train_attention_head(
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any] | None,
    out_dir: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    tau: float = 0.05,
    alpha: float = 0.25,
    logit_scale: float = 20.0,
    seed: int = 42,
    device: str | torch.device | None = None,
    resume: bool = False,
    query_chunk: int = 32,
    video_chunk: int = 32,
    selection_policy: str = "fixed_epochs",
    validation_provenance: str = "",
    max_epochs_this_call: int | None = None,
) -> dict[str, Any]:
    """Fit only the attention head, with an immutable epoch schedule.

    Default: no validation access, no early stopping; use the final epoch.
    Optional independent_validation requires an explicit provenance assertion and
    val metadata encoder_training_overlap=False. This cannot prove historical
    independence; the operator must verify the assertion from the actual sources.
    Checkpoints bind input bytes, schedule, code, optimizer and RNG. A partial
    invocation can resume the same schedule, but may never extend that schedule.
    """
    train = validate_cache(train_cache)
    if "test" in str(train.get("metadata", {}).get("split", "")).lower():
        raise ValueError("A test cache cannot be used for training")
    if selection_policy not in {"fixed_epochs", "independent_validation"}:
        raise ValueError("selection_policy must be fixed_epochs or independent_validation")
    val = None
    if selection_policy == "fixed_epochs":
        if val_cache is not None:
            raise ValueError("fixed_epochs must not receive validation features")
        validation_status = "reserved_encoder_seen_dev_not_accessed_for_selection"
    else:
        if val_cache is None or not str(validation_provenance).strip():
            raise ValueError("Independent validation requires cache and explicit provenance")
        val = validate_cache(val_cache)
        if "test" in str(val.get("metadata", {}).get("split", "")).lower():
            raise ValueError("A test cache cannot be used for training selection")
        if set(train["video_ids"]) & set(val["video_ids"]):
            raise ValueError("Train/validation video identities overlap")
        if val.get("metadata", {}).get("encoder_training_overlap") is not False:
            raise ValueError(
                "Independent validation must explicitly be absent from encoder training"
            )
        if train["text"].shape[-1] != val["text"].shape[-1]:
            raise ValueError("Train/validation embedding dimensions differ")
        validation_status = "independent_validation_as_asserted_by_operator"
    if epochs < 1 or batch_size < 2 or len(train["video_ids"]) < 2:
        raise ValueError("Need >=1 epoch, batch_size>=2 and >=2 train videos")
    numeric = (lr, weight_decay, tau, alpha, logit_scale)
    if not all(math.isfinite(float(x)) for x in numeric):
        raise ValueError("Training hyperparameters must be finite")
    if (
        lr <= 0
        or weight_decay < 0
        or not 0 < alpha <= 1
        or not 0.005 <= tau <= 1
        or logit_scale <= 0
    ):
        raise ValueError("Invalid learning rate, decay, mixture alpha, tau or logit scale")
    if max_epochs_this_call is not None and max_epochs_this_call < 1:
        raise ValueError("max_epochs_this_call must be positive")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    last_path = target / "last.pt"
    selected_path = target / ("final.pt" if selection_policy == "fixed_epochs" else "best.pt")
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = {
        "dim": int(train["text"].shape[-1]),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "tau": float(tau),
        "alpha": float(alpha),
        "logit_scale": float(logit_scale),
        "seed": int(seed),
        "epochs": int(epochs),
        "selection_policy": selection_policy,
        "validation_provenance": str(validation_provenance),
        "core_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    fingerprints = {
        "train": _cache_fingerprint(train),
        "validation": _cache_fingerprint(val) if val is not None else None,
    }
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    head = TinyAttentionHead(config["dim"], tau).to(selected)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    start_epoch, selected_epoch, best_r1 = 0, 0, -float("inf")
    selected_model = None
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"No completed-epoch checkpoint at {last_path}")
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state.get("format") != "msrvtt_tiny_attention_v1" or state["config"] != config:
            raise ValueError("Resume configuration or fixed epoch schedule differs")
        if state["cache_fingerprints"] != fingerprints:
            raise ValueError("Resume refused: cached features, identities, or metadata changed")
        head.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["completed_epochs"])
        selected_epoch = int(state["selected_epoch"])
        best_r1 = float(state["best_val_r1"]) if state["best_val_r1"] is not None else -float("inf")
        history = list(state["history"])
        selected_model = state["selected_model"]
        rng.bit_generator.state = state["numpy_generator_state"]
        torch.set_rng_state(state["torch_rng_state"])
        if selected.type == "cuda" and state.get("cuda_rng_states"):
            states = state["cuda_rng_states"]
            if len(states) != torch.cuda.device_count():
                raise ValueError("CUDA device count changed; exact resume is unavailable")
            torch.cuda.set_rng_state_all(states)
        print(f"Resumed after epoch {start_epoch}/{epochs}.", flush=True)
    elif last_path.exists() or selected_path.exists():
        raise FileExistsError("Checkpoint directory contains a run; resume or use a new directory")
    captions_by_video = {vid: [] for vid in train["video_ids"]}
    for i, vid in enumerate(train["text_video_ids"]):
        captions_by_video[vid].append(i)
    q_cpu = F.normalize(train["text"].cpu(), dim=-1)
    f_cpu = F.normalize(train["frames"].cpu(), dim=-1)
    m_cpu = train["mask"].cpu()
    end_epoch = (
        epochs if max_epochs_this_call is None else min(epochs, start_epoch + max_epochs_this_call)
    )
    for epoch in range(start_epoch, end_epoch):
        head.train()
        permutation = rng.permutation(len(train["video_ids"]))
        batches = [permutation[i : i + batch_size] for i in range(0, len(permutation), batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = np.concatenate([batches[-2], batches[-1]])
            batches.pop()
        total_loss, examples = 0.0, 0
        for video_batch in batches:
            batch_vids = [train["video_ids"][int(i)] for i in video_batch]
            caption_batch = [int(rng.choice(captions_by_video[vid])) for vid in batch_vids]
            q = q_cpu[torch.tensor(caption_batch)].to(selected)
            f = f_cpu[torch.tensor(video_batch)].to(selected)
            m = m_cpu[torch.tensor(video_batch)].to(selected)
            positives = torch.tensor(
                [
                    [train["text_video_ids"][ci] == vid for vid in batch_vids]
                    for ci in caption_batch
                ],
                device=selected,
                dtype=torch.bool,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=selected.type, enabled=False):
                scores = _score_block(q, f, m, method="B2", tau=tau, alpha=alpha, head=head)
                loss = multi_positive_contrastive_loss(scores, positives, logit_scale)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite head loss; last completed epoch remains saved")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(
                    "Non-finite head gradient; last completed epoch remains saved"
                )
            optimizer.step()
            total_loss += float(loss.detach()) * len(q)
            examples += len(q)
        model_state = {
            key: value.detach().cpu().clone() for key, value in head.state_dict().items()
        }
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / examples,
            "learned_tau": float(head.temperature().detach().cpu()),
        }
        if val is None:
            selected_epoch, selected_model = epoch + 1, model_state
        else:
            val_scores = score_matrix(
                val["text"],
                val["frames"],
                val["mask"],
                method="B2",
                tau=tau,
                alpha=alpha,
                query_chunk=query_chunk,
                video_chunk=video_chunk,
                device=selected,
                head=head,
            )
            val_result = evaluate_bidirectional(
                val_scores, val["text_video_ids"], val["video_ids"], val["text_ids"]
            )
            row.update(
                val_text_to_video=val_result["text_to_video"],
                val_video_to_text=val_result["video_to_text"],
            )
            r1 = float(val_result["text_to_video"]["R@1"])
            if r1 > best_r1:
                best_r1, selected_epoch, selected_model = r1, epoch + 1, model_state
        history.append(row)
        payload = {
            "format": "msrvtt_tiny_attention_v1",
            "config": config,
            "validation_status": validation_status,
            "cache_fingerprints": fingerprints,
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "completed_epochs": epoch + 1,
            "selected_epoch": selected_epoch,
            "best_val_r1": None if val is None else best_r1,
            "selected_model": selected_model,
            "history": history,
            "numpy_generator_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if selected.type == "cuda" else [],
        }
        _atomic_torch_save(payload, last_path)
        print(
            f"Epoch {epoch + 1}/{epochs}: loss={row['train_loss']:.5f}; "
            f"tau={row['learned_tau']:.5f}; selection={selection_policy}",
            flush=True,
        )
    if selected_model is None:
        raise ValueError("No completed training epoch is available")
    completed = len(history) == epochs
    head.load_state_dict(selected_model)
    head.cpu().eval()
    if completed:
        # Inference snapshot has no mismatched last-epoch optimizer state.
        _atomic_torch_save(
            {
                "format": "msrvtt_tiny_attention_v1",
                "config": config,
                "model": selected_model,
                "selected_epoch": selected_epoch,
                "validation_status": validation_status,
                "cache_fingerprints": fingerprints,
            },
            selected_path,
        )
    temporary_history = target / "history.json.tmp"
    temporary_history.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_history, target / "history.json")
    return {
        "head": head,
        "history": history,
        "selected_epoch": selected_epoch,
        "best_epoch": selected_epoch,
        "best_val_r1": None if val is None else best_r1,
        "checkpoint_path": str(selected_path) if completed else None,
        "last_checkpoint_path": str(last_path),
        "cache_fingerprints": fingerprints,
        "validation_status": validation_status,
        "selection_policy": selection_policy,
        "training_complete": completed,
    }

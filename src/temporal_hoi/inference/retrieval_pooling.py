"""Frozen-feature B0/B1/B2 retrieval with explicit identities and fixed galleries.

Inputs are text[T,D], frames[V,F,D], and a boolean valid-frame mask[V,F].
Scores are unscaled FP32 cosine similarities[T,V], not probabilities. B0 uses
per-frame L2 normalization, valid-frame mean, then video L2 normalization. B1
uses query-dependent softmax frame weights; B2 blends B0 and B1. No weights are
trained here. Apply an encoder logit scale explicitly in the caller if needed.

Evaluation reports recalls in percent, MRR in [0,1], and 1-based ranks. Exact
score ties preserve the supplied candidate order. A caption's positive video
is identified by text_video_ids, allowing several captions per candidate video.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _as_float_tensor(value: Any, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _identities(values: Sequence[str], name: str, *, unique: bool) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of nonempty string identities")
    result = list(values)
    if not result or any(not isinstance(x, str) or not x.strip() for x in result):
        raise ValueError(f"{name} requires nonempty string identities")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique identities")
    return result


def _feature_tensors(text: Any, frames: Any, mask: Any):
    q = _as_float_tensor(text, "text")
    f = _as_float_tensor(frames, "frames")
    m = torch.as_tensor(mask).detach().cpu()
    if m.dtype != torch.bool:
        raise ValueError("Frame mask must have boolean dtype")
    if q.ndim != 2 or f.ndim != 3 or q.shape[-1] != f.shape[-1]:
        raise ValueError("Expected compatible text[T,D] and frames[V,F,D]")
    if min(q.shape) == 0 or min(f.shape) == 0:
        raise ValueError("Empty features are not evaluable")
    if tuple(m.shape) != tuple(f.shape[:2]) or not m.any(dim=1).all():
        raise ValueError("Frame mask mismatch or video without valid frames")
    norms = (q.norm(dim=-1), f.norm(dim=-1)[m])
    if any(not torch.isfinite(n).all() or (n <= 1e-12).any() for n in norms):
        raise ValueError("Nonfinite or zero-norm embeddings are invalid")
    return q, f, m


def validate_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Validate cache shapes, IDs, finite features, and decoded-frame coverage.

    Required keys: text, frames, mask, text_video_ids, video_ids, text_ids.
    Unrecognized metadata is preserved; it is data and is never executed.
    Every caption and candidate must have at least one declared positive.
    """
    if not isinstance(cache, Mapping):
        raise ValueError("Cache must be a mapping")
    required = ("text", "frames", "mask", "text_video_ids", "video_ids", "text_ids")
    absent = [key for key in required if key not in cache]
    if absent:
        raise ValueError(f"Cache is missing keys: {absent}")
    text, frames, mask = _feature_tensors(cache["text"], cache["frames"], cache["mask"])
    text_video_ids = _identities(cache["text_video_ids"], "text_video_ids", unique=False)
    video_ids = _identities(cache["video_ids"], "video_ids", unique=True)
    text_ids = _identities(cache["text_ids"], "text_ids", unique=True)
    if len(text_video_ids) != len(text) or len(text_ids) != len(text):
        raise ValueError("Text identity lengths do not match text features")
    if len(video_ids) != len(frames):
        raise ValueError("video_ids length does not match frame features")
    if set(video_ids) != set(text_video_ids):
        raise ValueError("Ground-truth videos and candidate videos do not match")
    return dict(cache, text=text, frames=frames, mask=mask,
                text_video_ids=text_video_ids, video_ids=video_ids, text_ids=text_ids)


def direct_meanp_reference(text: Any, frames: Any, mask: Any) -> np.ndarray:
    """Independent unchunked CPU B0 check; allclose, not bitwise-tie parity.

    Uses masked tensor sums rather than the native per-video pooling path.
    The score scale is one. Exact-rank reproduction needs an external baseline
    with its original arithmetic and candidate ordering, especially on ties.
    """
    q, f, m = _feature_tensors(text, frames, mask)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = (f * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    if (v.norm(dim=-1) <= 1e-12).any():
        raise ValueError("Mean pooled video has zero norm")
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (q @ v.T).numpy()


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
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Score every text against every video, retaining the full candidate pool.

    B0 deliberately pools each video on CPU and multiplies one NumPy FP32
    matrix, preserving baseline arithmetic and exact-tie behavior. B1/B2 use
    bounded FP32 blocks with autocast disabled; CPU is the explicit default.
    Chunking limits working memory, not the number of queries or candidates.
    """
    if method not in {"B0", "B1", "B2"}:
        raise ValueError("method must be B0, B1, or B2")
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("tau must be finite and positive; alpha must be in [0,1]")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
           for n in (query_chunk, video_chunk)):
        raise ValueError("Chunk sizes must be positive integers")
    q_cpu, f_cpu, m_cpu = _feature_tensors(text, frames, mask)
    native_q = torch.stack([row / row.norm() for row in q_cpu]).numpy()
    native_v = []
    for features, valid in zip(f_cpu, m_cpu, strict=True):
        normalized = features[valid] / features[valid].norm(dim=-1, keepdim=True)
        pooled = normalized.mean(dim=0)
        if pooled.norm() <= 1e-12:
            raise ValueError("Mean pooled video has zero norm")
        native_v.append((pooled / pooled.norm()).numpy())
    native_base = native_q @ np.asarray(native_v, dtype=np.float32).T
    if not np.isfinite(native_base).all():
        raise ValueError("Baseline scoring produced non-finite values")
    if method == "B0":
        return native_base
    selected = torch.device(device)
    result = np.empty((len(q_cpu), len(f_cpu)), dtype=np.float32)
    with torch.inference_mode(), torch.autocast(device_type=selected.type, enabled=False):
        for qi in range(0, len(q_cpu), query_chunk):
            q = F.normalize(q_cpu[qi:qi + query_chunk].to(selected), dim=-1)
            for vi in range(0, len(f_cpu), video_chunk):
                f = F.normalize(f_cpu[vi:vi + video_chunk].to(selected), dim=-1)
                m = m_cpu[vi:vi + video_chunk].to(selected)
                logits = torch.einsum("qd,vfd->qvf", q, f) / tau
                weights = logits.masked_fill(~m[None], -torch.inf).softmax(dim=-1)
                pooled = F.normalize(torch.einsum("qvf,vfd->qvd", weights, f), dim=-1)
                block = torch.einsum("qd,qvd->qv", q, pooled)
                result[qi:qi + len(q), vi:vi + len(f)] = block.cpu().numpy()
    if method == "B2":
        result = (1.0 - alpha) * native_base + alpha * result
    if not np.isfinite(result).all():
        raise ValueError("Scoring produced non-finite values")
    return result


def _metric_summary(ranks: np.ndarray) -> dict[str, float | int]:
    return {
        "R@1": float(100 * np.mean(ranks <= 1)),
        "R@5": float(100 * np.mean(ranks <= 5)),
        "R@10": float(100 * np.mean(ranks <= 10)),
        "MedR": float(np.median(ranks)),
        "MeanR": float(np.mean(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
        "NumQueries": int(len(ranks)),
    }


def _ranks(scores, query_gt_ids, candidate_gt_ids):
    gt_candidates = np.asarray(candidate_gt_ids, dtype=str)
    positive_positions = {
        gt: np.flatnonzero(gt_candidates == gt) for gt in set(query_gt_ids)
    }
    ranks = np.empty(len(query_gt_ids), dtype=np.int64)
    tie_at_best_positive, ties_at_top = 0, 0
    for i, gt in enumerate(query_gt_ids):
        order = np.argsort(-scores[i], kind="stable")
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        pos = positive_positions[gt]
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
    """Evaluate explicit positives both ways; recalls are percent, ranks 1-based."""
    s = np.asarray(scores, dtype=np.float32)
    tvid = _identities(text_video_ids, "text_video_ids", unique=False)
    vids = _identities(video_ids, "video_ids", unique=True)
    tids = _identities(text_ids, "text_ids", unique=True)
    if s.shape != (len(tvid), len(vids)) or len(tids) != len(tvid) or not np.isfinite(s).all():
        raise ValueError("Score shape, identities, or finite-value check failed")
    if set(vids) != set(tvid):
        raise ValueError("Ground-truth videos and candidate videos do not match")
    tr, tt = _ranks(s, tvid, vids)
    vr, vt = _ranks(s.T, vids, tvid)
    return {
        "text_to_video": {**_metric_summary(tr), "NumCandidates": len(vids),
                          "NumPositivePairs": len(tvid)},
        "video_to_text": {**_metric_summary(vr), "NumCandidates": len(tvid),
                          "NumPositivePairs": len(tvid)},
        "ranks_text_to_video": tr,
        "ranks_video_to_text": vr,
        "ties_text_to_video": tt,
        "ties_video_to_text": vt,
        "tie_rule": "descending FP32 score; stable input candidate order on exact ties",
    }


def original_clip4clip_metrics(scores_diagonal: Any) -> dict[str, Any]:
    """Legacy diagonal evaluator; tied positives can create extra rank entries.

    Compatibility with ArrowLuo/CLIP4Clip metrics.py at commit 508ffa3.
    The upstream MIT notice is retained in the adjacent CLIP4Clip_LICENSE.txt.
    This is a separately labeled compatibility result, never the ID-stable
    multi-positive metric. Rows/columns must first be aligned by positive ID.
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
    """Align one caption per video by ID and report the legacy evaluator twice."""
    tv = _identities(text_video_ids, "text_video_ids", unique=False)
    vv = _identities(video_ids, "video_ids", unique=True)
    if len(tv) != len(vv) or len(set(tv)) != len(tv) or set(tv) != set(vv):
        return {"available": False,
                "reason": "Original diagonal evaluator requires exactly one caption per video."}
    s = np.asarray(scores)
    if s.shape != (len(tv), len(vv)):
        raise ValueError("Score shape does not match identities")
    lookup = {vid: i for i, vid in enumerate(vv)}
    aligned = s[:, [lookup[vid] for vid in tv]]
    return {"available": True, "text_to_video": original_clip4clip_metrics(aligned),
            "video_to_text": original_clip4clip_metrics(aligned.T)}


def _positive_ranks(value: Any) -> np.ndarray:
    ranks = np.asarray(value)
    if (ranks.ndim != 1 or not len(ranks) or not np.issubdtype(ranks.dtype, np.integer)
            or (ranks < 1).any()):
        raise ValueError("Ranks must be a nonempty vector of positive integers")
    return ranks.astype(np.int64)


def paired_bootstrap(
    base_ranks: Any,
    new_ranks: Any,
    cluster_ids: Sequence[str],
    *,
    n_bootstrap: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired video-ID cluster bootstrap with a fixed full candidate gallery.

    Recall differences use percentage points; MRR differences retain [0,1]
    units. This does not model training-seed or source-overlap uncertainty.
    """
    base, new = _positive_ranks(base_ranks), _positive_ranks(new_ranks)
    ids = _identities(cluster_ids, "cluster_ids", unique=False)
    if base.shape != new.shape or len(base) != len(ids):
        raise ValueError("Paired ranks and cluster identities must have matching lengths")
    if (isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int)
            or n_bootstrap < 1 or not 0 < confidence < 1):
        raise ValueError("Invalid bootstrap count or confidence")
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
        "clusters": len(unique), "n_bootstrap": int(n_bootstrap), "seed": int(seed),
        "confidence": confidence, "gallery": "fixed; ranks computed before bootstrap",
        "delta": {
            name: {"estimate": float(point[i]), "ci_low": float(np.quantile(draws[:, i], low)),
                   "ci_high": float(np.quantile(draws[:, i], high))}
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
    """Compare paired ranks, keeping both improved and worsened query outcomes."""
    records, result = [], {}
    for direction, identities, clusters in (
        ("text_to_video", text_ids, text_video_ids),
        ("video_to_text", video_ids, video_ids),
    ):
        before = _positive_ranks(base[f"ranks_{direction}"])
        after = _positive_ranks(new[f"ranks_{direction}"])
        if before.shape != after.shape or len(before) != len(identities):
            raise ValueError("Run ranks and identities do not match")
        corrected, broken = (before > 1) & (after == 1), (before == 1) & (after > 1)
        result[direction] = {
            "corrected_at_1": int(corrected.sum()), "broken_at_1": int(broken.sum()),
            "unchanged_at_1": int(len(before) - corrected.sum() - broken.sum()),
            "bootstrap": paired_bootstrap(before, after, clusters,
                                          n_bootstrap=n_bootstrap, seed=seed),
        }
        for i in np.flatnonzero(before != after):
            records.append({
                "direction": direction, "query_id": str(identities[i]),
                "gt_video_id": str(clusters[i]), "rank_before": int(before[i]),
                "rank_after": int(after[i]),
                "r1_change": "corrected" if corrected[i] else "broken" if broken[i] else "unchanged",
            })
    if output_csv is not None:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "direction", "query_id", "gt_video_id", "rank_before", "rank_after", "r1_change"))
            writer.writeheader()
            writer.writerows(records)
    result["changed_queries"] = records
    return result

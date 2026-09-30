"""W02 metrics. Undefined ratios are None (JSON null), never an invented zero."""

from typing import Any

import numpy as np


def _interval(interval):
    if len(interval) != 2:
        raise ValueError("An interval needs start and end")
    start, end = map(float, interval)
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValueError("Intervals must be finite and satisfy 0 <= start < end")
    return start, end


def compute_temporal_iou(pred_interval, gt_interval) -> float:
    ps, pe = _interval(pred_interval)
    gs, ge = _interval(gt_interval)
    intersection = max(0.0, min(pe, ge) - max(ps, gs))
    return intersection / ((pe - ps) + (ge - gs) - intersection)


def match_events_greedy(predictions, ground_truths, iou_threshold=0.5) -> dict[str, Any]:
    """Stable descending-score matching, same video AND rule; input indices preserved."""
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    for event in [*predictions, *ground_truths]:
        if not event.get("video_id") or not event.get("rule_id"):
            raise ValueError("Every event requires video_id and rule_id")
        _interval((event["start_s"], event["end_s"]))
        if not np.isfinite(event.get("score", 0.0)):
            raise ValueError("Event score must be finite")
    matched, matches = set(), []
    for p_idx, pred in sorted(enumerate(predictions), key=lambda pair: -pair[1].get("score", 0.0)):
        best_iou, best_idx = -1.0, None
        for g_idx, gt in enumerate(ground_truths):
            if g_idx in matched or (pred["video_id"], pred["rule_id"]) != (
                gt["video_id"],
                gt["rule_id"],
            ):
                continue
            iou = compute_temporal_iou(
                (pred["start_s"], pred["end_s"]), (gt["start_s"], gt["end_s"])
            )
            if iou >= iou_threshold and iou > best_iou:
                best_iou, best_idx = iou, g_idx
        if best_idx is not None:
            matched.add(best_idx)
            matches.append(
                dict(
                    pred_index=p_idx,
                    gt_index=best_idx,
                    iou=best_iou,
                    rule_id=pred["rule_id"],
                    video_id=pred["video_id"],
                )
            )
    return dict(
        true_positives=len(matches),
        false_positives=len(predictions) - len(matches),
        false_negatives=len(ground_truths) - len(matches),
        matches=matches,
    )


def compute_event_metrics(
    predictions,
    ground_truths,
    normal_duration_hours=0.0,
    iou_threshold=0.5,
    *,
    normal_false_alarms: int | None = None,
) -> dict[str, Any]:
    """FA/hour requires an explicit count from separate normal-only footage."""
    if not np.isfinite(normal_duration_hours) or normal_duration_hours < 0:
        raise ValueError("Normal duration must be finite and nonnegative")
    if normal_false_alarms is not None and (
        not isinstance(normal_false_alarms, int) or normal_false_alarms < 0
    ):
        raise ValueError("Normal false alarms must be a nonnegative integer")
    result = match_events_greedy(predictions, ground_truths, iou_threshold)
    tp, fp, fn = (result[k] for k in ("true_positives", "false_positives", "false_negatives"))

    def ratio(a, b):
        return a / b if b else None

    return {
        **result,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1_score": ratio(2 * tp, 2 * tp + fp + fn),
        "iou_threshold": iou_threshold,
        "normal_duration_hours": normal_duration_hours,
        "normal_false_alarms": normal_false_alarms,
        "false_alarms_per_hour": (
            ratio(normal_false_alarms, normal_duration_hours)
            if normal_false_alarms is not None
            else None
        ),
    }


def _indices(indices, size):
    values = set(indices)
    if any(
        isinstance(i, (bool, np.bool_))
        or not isinstance(i, (int, np.integer))
        or i < 0
        or i >= size
        for i in values
    ):
        raise ValueError("Candidate index out of range")
    return sorted(values)


def compute_retrieval_metrics(similarity_matrix, ground_truth_indices, k_list=None):
    """Rank the first positive (1-based); report R@K, MRR, MedR and coverage.

    Rows are queries, columns are candidates. Each scored query must have its
    complete positive set for this candidate pool. None or an empty set excludes
    the query, rather than labeling it negative. Partially annotated queries
    should be passed as None until their positive set is known. Scores must be
    finite; ties use the original candidate order, which callers must preserve.
    K larger than the candidate pool uses the full pool. Undefined metrics are
    None. Sorting one row at a time avoids a second full ranking matrix.
    """
    ks = [1, 5, 10] if k_list is None else list(k_list)
    sim = np.asarray(similarity_matrix, dtype=float)
    if sim.ndim != 2 or not np.isfinite(sim).all():
        raise ValueError("Scores must be a finite 2D matrix")
    if len(ground_truth_indices) != len(sim):
        raise ValueError("One relevance annotation is required per query")
    if any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0
        for k in ks
    ):
        raise ValueError("K must be a positive integer")
    positives = [
        [] if gt is None else _indices(gt, sim.shape[1]) for gt in ground_truth_indices
    ]
    first_ranks = []
    for scores, gt in zip(sim, positives):
        if not gt:
            first_ranks.append(None)
            continue
        order = np.argsort(-scores, kind="stable")
        first_ranks.append(int(np.flatnonzero(np.isin(order, gt))[0]) + 1)
    valid_ranks = np.asarray([rank for rank in first_ranks if rank is not None], dtype=float)
    count = len(valid_ranks)
    return {
        **{f"R@{k}": float(np.mean(valid_ranks <= k)) if count else None for k in ks},
        "MRR": float(np.mean(1.0 / valid_ranks)) if count else None,
        "MedR": float(np.median(valid_ranks)) if count else None,
        "num_queries": sim.shape[0],
        "num_candidates": sim.shape[1],
        "num_evaluated_queries": count,
        "num_excluded_queries": sim.shape[0] - count,
        "first_positive_ranks": first_ranks,
        "tie_breaking": "candidate_order",
    }


def compute_recall_at_k(similarity_matrix, ground_truth_indices, k_list=None):
    """Compatibility API retaining the pilot defaults and recall-only output."""
    ks = [1, 3, 5] if k_list is None else list(k_list)
    result = compute_retrieval_metrics(similarity_matrix, ground_truth_indices, ks)
    return {f"R@{k}": result[f"R@{k}"] for k in ks}


def compute_bidirectional_retrieval(
    video_text_similarity,
    video_to_text_positives,
    text_to_video_positives,
    k_list=None,
):
    """Evaluate a [videos, texts] matrix in both directions.

    Supply positive mappings explicitly for each direction: do not infer negative
    or fully annotated reverse queries from an incomplete forward annotation.
    A reverse mapping has one entry per text; its indices refer to video rows.
    """
    sim = np.asarray(video_text_similarity, dtype=float)
    if sim.ndim != 2:
        raise ValueError("Scores must be a finite 2D matrix")
    ks = None if k_list is None else list(k_list)
    return {
        "video_to_text": compute_retrieval_metrics(sim, video_to_text_positives, ks),
        "text_to_video": compute_retrieval_metrics(sim.T, text_to_video_positives, ks),
    }


def compute_pairwise_metrics(scores, positive_indices, negative_indices):
    """Strict positive > negative; unknown candidates are not negative."""
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("Scores must be a finite vector")
    pos, neg = _indices(positive_indices, len(arr)), _indices(negative_indices, len(arr))
    if set(pos) & set(neg):
        raise ValueError("Positive and negative labels overlap")
    differences = (arr[pos, None] - arr[neg]).ravel()
    return {
        "num_pairs": len(differences),
        "pairwise_accuracy": float(np.mean(differences > 0)) if len(differences) else None,
        "mean_margin": float(np.mean(differences)) if len(differences) else None,
    }


def compute_pairwise_accuracy(scores, positive_indices, negative_indices):
    return compute_pairwise_metrics(scores, positive_indices, negative_indices)["pairwise_accuracy"]

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
    if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= size for i in values):
        raise ValueError("Candidate index out of range")
    return sorted(values)


def compute_recall_at_k(similarity_matrix, ground_truth_indices, k_list=None):
    """Multi-positive recall; unlabeled rows excluded, score ties use column order."""
    ks = [1, 3, 5] if k_list is None else k_list
    sim = np.asarray(similarity_matrix, dtype=float)
    if sim.ndim != 2 or not np.isfinite(sim).all():
        raise ValueError("Scores must be a finite 2D matrix")
    if len(ground_truth_indices) != len(sim):
        raise ValueError("One relevance annotation is required per query")
    if any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("K must be a positive integer")
    positives = [_indices(gt, sim.shape[1]) for gt in ground_truth_indices]
    valid = [i for i, gt in enumerate(positives) if gt]
    ranks = np.argsort(-sim, axis=1, kind="stable")
    return {
        f"R@{k}": (
            sum(bool(set(ranks[i, :k]) & set(positives[i])) for i in valid) / len(valid)
            if valid
            else None
        )
        for k in ks
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

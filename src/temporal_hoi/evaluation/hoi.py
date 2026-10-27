"""Frame-level ST-HOI mAP, including the upstream shared-pair matching rule."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

from temporal_hoi.data.vidhoi import VidHOIFrame, identity_hash


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    intersection = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * max(
        0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    )
    if intersection == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def voc_11_point_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    return float(sum(
        float(np.max(precision[recall >= threshold])) / 11
        if np.any(recall >= threshold) else 0.0
        for threshold in np.arange(0.0, 1.1, 0.1)
    ))


def _log_score(value: float) -> float:
    return math.log(value if value > 0 else 1e-300)


def evaluate_frame_hoi(
    frames: Sequence[VidHOIFrame], *, class_subset: str = "full", top_k: int = 100,
    iou_threshold: float = 0.5,
) -> dict:
    """Match the pinned notebook; all returned AP values use the [0,1] scale.

    A GT pair is consumed across predicates within a frame, as in upstream.
    This behavior intentionally differs from per-class independent matching.
    Empty GT-supported subsets return null mAP rather than NaN or a fake zero.
    """
    if not frames:
        raise ValueError("Evaluation needs at least one selected frame")
    if class_subset not in {"full", "rare", "nonrare"}:
        raise ValueError("class_subset must be full, rare, or nonrare")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    keys = [frame.key for frame in frames]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate result frame IDs")
    for field in ("split", "box_source", "coordinate_system"):
        if len({getattr(frame, field) for frame in frames}) != 1:
            raise ValueError(f"Cannot mix {field} in one evaluator run")
    counts: Counter = Counter()
    for frame in frames:
        for pair in frame.ground_truth:
            for predicate, value in enumerate(pair.predicate_values):
                if value == 1:
                    counts[(0, predicate, pair.object_class)] += 1
    counts = Counter({key: count for key, count in counts.items() if (
        class_subset == "full" or (count < 25 if class_subset == "rare" else count >= 25)
    )})
    observations = {key: [] for key in counts}
    for frame in frames:
        ranked = []
        for pair_index, pair in enumerate(frame.predictions):
            for predicate, value in enumerate(pair.predicate_values):
                ranked.append((_log_score(value) + _log_score(pair.subject_score)
                               + _log_score(pair.object_score), pair_index, predicate))
        ranked.sort(reverse=True)  # Notebook tuple tie order: score, pair, predicate.
        matched_pairs: set[int] = set()
        for score, pair_index, predicate in ranked[:top_k]:
            predicted = frame.predictions[pair_index]
            key = (predicted.subject_class, predicate, predicted.object_class)
            if key not in counts:
                continue  # Class filtering happens AFTER the per-frame top-k.
            best_overlap = 0.0
            best_index = None
            for gt_index, gt in enumerate(frame.ground_truth):
                if (gt.subject_class != predicted.subject_class
                        or gt.object_class != predicted.object_class
                        or gt.predicate_values[predicate] != 1):
                    continue
                overlap = min(bbox_iou(gt.subject_box, predicted.subject_box),
                              bbox_iou(gt.object_box, predicted.object_box))
                if overlap >= iou_threshold and overlap > best_overlap:
                    best_overlap, best_index = overlap, gt_index
            matched = best_index is not None and best_index not in matched_pairs
            if matched:
                matched_pairs.add(best_index)
            observations[key].append((score, int(matched)))
    class_results = []
    for key, count in counts.items():
        rows = observations[key]
        ap = max_recall = 0.0
        true_positives = []
        ordered_scores = []
        if rows:
            scores = np.asarray([row[0] for row in rows])
            order = np.argsort(-scores)  # Match upstream default; record NumPy version.
            tp = np.asarray([row[1] for row in rows])[order]
            cum_tp = np.cumsum(tp)
            recall = cum_tp / count
            precision = cum_tp / np.arange(1, len(rows) + 1)
            ap = voc_11_point_ap(recall, precision)
            max_recall = float(np.max(recall))
            true_positives = tp.tolist()
            ordered_scores = scores[order].tolist()
        class_results.append({
            "triplet": list(key), "gt_count": count, "prediction_count": len(rows),
            "ap": ap, "max_recall": max_recall, "sorted_tp": true_positives,
            "sorted_scores": ordered_scores,
        })
    return {
        "evaluator": "ST-HOI_notebook_compat_v1", "numpy_version": np.__version__,
        "metric_scale": "fraction", "ap_definition": "VOC_11_point",
        "class_subset": class_subset, "rare_threshold": 25,
        "rare_count_source": "selected_frame_GT_instances", "top_k": top_k,
        "top_k_before_class_filter": True, "iou_threshold": iou_threshold,
        "gt_matching_scope": "shared_pair_across_predicates_per_frame",
        "global_ties": "numpy_argsort_default", "num_frames": len(frames),
        "frame_ids_sha256": identity_hash(keys), "num_classes": len(class_results),
        "box_source": frames[0].box_source, "split": frames[0].split,
        "coordinate_system": frames[0].coordinate_system,
        "status": "evaluated" if class_results else "no_GT_supported_classes",
        "map": float(np.mean([row["ap"] for row in class_results])) if class_results else None,
        "mean_max_recall": float(np.mean([row["max_recall"] for row in class_results]))
        if class_results else None,
        "classes": class_results,
    }

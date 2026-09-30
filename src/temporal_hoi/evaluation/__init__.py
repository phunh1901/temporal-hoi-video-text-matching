"""Evaluation metrics for Video-Text Matching (NC2) and Event Violation Detection (NC1)."""

from temporal_hoi.evaluation.metrics import (
    compute_event_metrics,
    compute_pairwise_accuracy,
    compute_recall_at_k,
    compute_temporal_iou,
    match_events_greedy,
)

__all__ = [
    "compute_temporal_iou",
    "match_events_greedy",
    "compute_event_metrics",
    "compute_recall_at_k",
    "compute_pairwise_accuracy",
]

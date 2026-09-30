"""Unit tests for evaluation metrics (W02-04).

Contains 5 explicit hand-calculated test cases verifying mathematical correctness:
1. Exact match (tIoU = 1.0 -> TP=1, FP=0, FN=0, F1=1.0)
2. Boundary test around tIoU threshold 0.5 (overlap = 8s/12s = 0.67 >= 0.5 -> TP=1; overlap = 2s/10s = 0.20 < 0.5 -> FP=1, FN=1)
3. Duplicate predictions on a single GT event (1-to-1 matching: 1st matches, duplicate is penalized as FP)
4. Edge cases & safe division (no ground truths, empty predictions, zero denominator handling)
5. Multi-positive ranking for Recall@K (doesn't penalize when second true positive appears in top-K)
"""

import numpy as np
import pytest

from temporal_hoi.evaluation.metrics import (
    compute_event_metrics,
    compute_pairwise_accuracy,
    compute_recall_at_k,
    compute_temporal_iou,
    match_events_greedy,
)


def test_example_1_exact_match():
    """Case 1: Hand-calculated exact interval match."""
    iou = compute_temporal_iou([10.0, 20.0], [10.0, 20.0])
    assert iou == 1.0

    preds = [
        {
            "video_id": "video_01",
            "start_s": 10.0,
            "end_s": 20.0,
            "rule_id": "RULE_01",
            "score": 0.95,
        }
    ]
    gts = [{"video_id": "video_01", "start_s": 10.0, "end_s": 20.0, "rule_id": "RULE_01"}]

    metrics = compute_event_metrics(
        preds, gts, normal_duration_hours=1.0, normal_false_alarms=0, iou_threshold=0.5
    )
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["false_alarms_per_hour"] == 0.0


def test_example_2_boundary_tiou_threshold():
    """Case 2: Boundary test around tIoU threshold 0.5.

    GT: [10, 20] (duration 10s)
    Pred A: [12, 22] -> Intersection [12, 20] = 8s, Union [10, 22] = 12s -> tIoU = 8/12 = 0.6667 >= 0.5 -> MATCH
    Pred B: [18, 26] -> Intersection [18, 20] = 2s, Union [10, 26] = 16s -> tIoU = 2/16 = 0.125 < 0.5 -> NO MATCH
    """
    iou_a = compute_temporal_iou([12.0, 22.0], [10.0, 20.0])
    assert abs(iou_a - 8.0 / 12.0) < 1e-4

    iou_b = compute_temporal_iou([18.0, 26.0], [10.0, 20.0])
    assert abs(iou_b - 2.0 / 16.0) < 1e-4

    gts = [{"video_id": "video_01", "start_s": 10.0, "end_s": 20.0, "rule_id": "RULE_01"}]

    # Pred A should match GT
    metrics_a = compute_event_metrics(
        [
            {
                "video_id": "video_01",
                "start_s": 12.0,
                "end_s": 22.0,
                "rule_id": "RULE_01",
                "score": 0.8,
            }
        ],
        gts,
        iou_threshold=0.5,
    )
    assert metrics_a["true_positives"] == 1
    assert metrics_a["false_positives"] == 0
    assert metrics_a["false_negatives"] == 0

    # Pred B should fail to match GT (below threshold 0.5)
    metrics_b = compute_event_metrics(
        [
            {
                "video_id": "video_01",
                "start_s": 18.0,
                "end_s": 26.0,
                "rule_id": "RULE_01",
                "score": 0.8,
            }
        ],
        gts,
        iou_threshold=0.5,
    )
    assert metrics_b["true_positives"] == 0
    assert metrics_b["false_positives"] == 1
    assert metrics_b["false_negatives"] == 1
    assert metrics_b["f1_score"] == 0.0


def test_example_3_duplicate_predictions_penalty():
    """Case 3: Single GT event with TWO overlapping predictions.

    1-to-1 matching rule: The first prediction matches GT (TP=1).
    The duplicate prediction CANNOT match the same GT again, and is penalized as FP (FP=1).
    """
    gts = [{"video_id": "video_01", "start_s": 10.0, "end_s": 20.0, "rule_id": "RULE_01"}]
    preds = [
        {
            "video_id": "video_01",
            "start_s": 10.0,
            "end_s": 20.0,
            "rule_id": "RULE_01",
            "score": 0.90,
        },  # Match 1 (TP)
        {
            "video_id": "video_01",
            "start_s": 11.0,
            "end_s": 21.0,
            "rule_id": "RULE_01",
            "score": 0.85,
        },  # Duplicate (FP)
    ]

    metrics = compute_event_metrics(preds, gts, iou_threshold=0.5)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert abs(metrics["f1_score"] - (2 * 0.5 * 1.0) / (0.5 + 1.0)) < 1e-4  # F1 = 0.6667


def test_example_4_edge_cases_and_zero_division():
    """Case 4: Empty predictions, empty ground truths, different rules, and division by zero safety."""
    # Sub-case 4A: Normal video (0 GT, 0 Preds) -> No crash, 0 metrics
    metrics_empty = compute_event_metrics([], [], normal_duration_hours=0.5, normal_false_alarms=0)
    assert metrics_empty["true_positives"] == 0
    assert metrics_empty["false_positives"] == 0
    assert metrics_empty["false_negatives"] == 0
    assert metrics_empty["precision"] is None
    assert metrics_empty["recall"] is None
    assert metrics_empty["f1_score"] is None
    assert metrics_empty["false_alarms_per_hour"] == 0.0

    # Sub-case 4B: False alarm in normal video (0 GT, 1 Pred)
    metrics_fa = compute_event_metrics(
        [
            {
                "video_id": "video_01",
                "start_s": 5.0,
                "end_s": 15.0,
                "rule_id": "RULE_01",
                "score": 0.7,
            }
        ],
        [],
        normal_duration_hours=0.5,
        normal_false_alarms=1,
    )
    assert metrics_fa["true_positives"] == 0
    assert metrics_fa["false_positives"] == 1
    assert metrics_fa["false_negatives"] == 0
    assert metrics_fa["precision"] == 0.0
    assert metrics_fa["recall"] is None
    assert metrics_fa["false_alarms_per_hour"] == 2.0  # 1 alarm / 0.5 hour = 2.0 FA/hr

    # Sub-case 4C: Overlapping intervals but DIFFERENT rules -> No match
    metrics_diff_rule = compute_event_metrics(
        [
            {
                "video_id": "video_01",
                "start_s": 10.0,
                "end_s": 20.0,
                "rule_id": "RULE_01",
                "score": 0.9,
            }
        ],
        [{"video_id": "video_01", "start_s": 10.0, "end_s": 20.0, "rule_id": "RULE_02"}],
    )
    assert metrics_diff_rule["true_positives"] == 0
    assert metrics_diff_rule["false_positives"] == 1
    assert metrics_diff_rule["false_negatives"] == 1


def test_example_5_multi_positive_recall_at_k():
    """Case 5: Multi-positive ranking for Video-Text Matching (NC2).

    Video 1 has TWO correct candidate texts: {T0, T1}.
    Candidate scores:
    T2 (negative): 0.88 (Rank 1)
    T0 (positive): 0.82 (Rank 2)
    T1 (positive): 0.79 (Rank 3)
    T3 (negative): 0.40 (Rank 4)

    Expected:
    R@1: Top-1 is T2 (negative) -> 0.0
    R@3: Top-3 contains {T2, T0, T1}. Since T0 and T1 are valid positives -> 1.0
    Crucial: T1 at Rank 3 must NOT be penalized as error because T0 is also positive!
    """
    sim_matrix = np.array(
        [
            [
                0.82,
                0.79,
                0.88,
                0.40,
            ],  # Video 0: T0=0.82 (pos), T1=0.79 (pos), T2=0.88 (neg), T3=0.40 (neg)
        ]
    )
    gt_indices = [{0, 1}]  # Both text 0 and text 1 are valid positives

    recalls = compute_recall_at_k(sim_matrix, gt_indices, k_list=[1, 3])
    assert recalls["R@1"] == 0.0
    assert recalls["R@3"] == 1.0

    # Test pairwise accuracy: Positives {0, 1} vs Negatives {2, 3}
    # Pairs:
    # (T0: 0.82, T2: 0.88) -> Loss
    # (T0: 0.82, T3: 0.40) -> Win
    # (T1: 0.79, T2: 0.88) -> Loss
    # (T1: 0.79, T3: 0.40) -> Win
    # Pairwise accuracy = 2 wins / 4 pairs = 0.50
    pw_acc = compute_pairwise_accuracy(
        sim_matrix[0], positive_indices=[0, 1], negative_indices=[2, 3]
    )
    assert pw_acc == 0.50


def test_different_videos_never_match_and_original_indices_preserved():
    gt = [{"video_id": "a", "rule_id": "r", "start_s": 0, "end_s": 2}]
    wrong = {"video_id": "b", "rule_id": "r", "start_s": 0, "end_s": 2, "score": 0.1}
    right = {**gt[0], "score": 0.9}
    result = match_events_greedy([wrong, right], gt)
    assert result["matches"][0]["pred_index"] == 1
    assert result["false_positives"] == 1


def test_exact_half_iou_and_no_predictions():
    assert compute_temporal_iou([0, 2], [0, 4]) == 0.5
    gt = [{"video_id": "a", "rule_id": "r", "start_s": 0, "end_s": 4}]
    assert compute_event_metrics([{**gt[0], "end_s": 2}], gt)["true_positives"] == 1
    result = compute_event_metrics([], gt)
    assert result["precision"] is None
    assert result["recall"] == result["f1_score"] == 0


def test_undefined_matching_and_strict_ties():
    assert compute_pairwise_accuracy([0.5, 0.5], [0], [1]) == 0
    assert compute_pairwise_accuracy([0.5], [0], []) is None
    assert compute_recall_at_k(np.empty((0, 2)), [], [1])["R@1"] is None
    assert compute_recall_at_k([[0.9, 0.1], [0.9, 0.1]], [[], [0]], [1])["R@1"] == 1


def test_normal_only_false_alarms_not_all_false_positives():
    pred = [{"video_id": "a", "rule_id": "r", "start_s": 0, "end_s": 2}]
    assert compute_event_metrics(pred, [], 0.5)["false_alarms_per_hour"] is None
    assert compute_event_metrics(pred, [], 0.5, normal_false_alarms=0)["false_alarms_per_hour"] == 0


def test_invalid_metric_inputs():
    with pytest.raises(ValueError):
        compute_temporal_iou([2, 1], [0, 1])
    with pytest.raises(ValueError):
        match_events_greedy([{"start_s": 0, "end_s": 1}], [])
    with pytest.raises(ValueError):
        compute_recall_at_k([[float("nan")]], [[0]])
    with pytest.raises(ValueError):
        compute_recall_at_k([[0.5]], [[1]])
    with pytest.raises(ValueError):
        compute_pairwise_accuracy([0.5], [0], [0])


def test_pairwise_margin():
    from temporal_hoi.evaluation.metrics import compute_pairwise_metrics

    result = compute_pairwise_metrics([0.8, 0.6, 0.7, 0.3], [0, 1], [2, 3])
    assert result["num_pairs"] == 4
    assert result["pairwise_accuracy"] == 0.75
    assert result["mean_margin"] == pytest.approx(0.2)

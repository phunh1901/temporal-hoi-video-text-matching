"""Hand-calculated retrieval ranks, coverage, and directionality."""

import json

import numpy as np
import pytest

from temporal_hoi.evaluation import (
    compute_bidirectional_retrieval,
    compute_recall_at_k,
    compute_retrieval_metrics,
)


def test_rectangular_bidirectional_hand_calculation():
    # V0: T1, T0, T2 -> positive rank 2. V1: T2, T0, T1 -> rank 3.
    # T0: V0, V1 -> rank 1. T1: V0, V1 -> rank 2. T2 is unlabeled.
    result = compute_bidirectional_retrieval(
        [[0.8, 0.9, 0.1], [0.7, 0.6, 0.95]], [[0], [1]], [[0], [1], None], [1, 2, 5, 10]
    )
    forward, reverse = result["video_to_text"], result["text_to_video"]
    assert forward["first_positive_ranks"] == [2, 3]
    assert forward["MRR"] == pytest.approx(5 / 12)
    assert forward["MedR"] == 2.5
    assert forward["R@1"] == 0 and forward["R@2"] == 0.5
    assert forward["R@5"] == forward["R@10"] == 1
    assert reverse["first_positive_ranks"] == [1, 2, None]
    assert reverse["MRR"] == 0.75 and reverse["MedR"] == 1.5
    assert reverse["R@1"] == 0.5 and reverse["R@2"] == 1
    assert reverse["num_queries"] == 3 and reverse["num_candidates"] == 2
    assert reverse["num_evaluated_queries"] == 2 and reverse["num_excluded_queries"] == 1
    json.dumps(result, allow_nan=False)


def test_multi_positive_uses_first_relevant_rank_and_stable_ties():
    result = compute_retrieval_metrics([[0.9, 0.9, 0.8, 0.1]], [[1, 2, 2]])
    assert result["first_positive_ranks"] == [2]
    assert result["MRR"] == 0.5 and result["MedR"] == 2
    assert result["R@1"] == 0 and result["R@5"] == result["R@10"] == 1
    assert result["tie_breaking"] == "candidate_order"


def test_unknown_queries_do_not_become_negative_or_change_denominator():
    result = compute_retrieval_metrics([[0.9, 0.1]] * 3, [None, [], [1]])
    assert result["first_positive_ranks"] == [None, None, 2]
    assert result["num_evaluated_queries"] == 1 and result["num_excluded_queries"] == 2
    assert result["MRR"] == 0.5 and result["MedR"] == 2


@pytest.mark.parametrize("shape,labels", [((0, 3), []), ((2, 0), [[], None]), ((2, 2), [[], []])])
def test_empty_or_unlabeled_returns_null_metrics(shape, labels):
    result = compute_retrieval_metrics(np.zeros(shape), labels)
    assert all(result[key] is None for key in ["MRR", "MedR", "R@1", "R@5", "R@10"])
    assert result["num_evaluated_queries"] == 0
    assert result["num_excluded_queries"] == shape[0]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "scores,labels,ks",
    [
        ([0.1, 0.2], [[0]], [1]),
        ([[np.nan]], [[0]], [1]),
        ([[np.inf]], [[0]], [1]),
        ([[0.5]], [], [1]),
        ([[0.5]], [[1]], [1]),
        ([[0.5]], [[-1]], [1]),
        ([[0.5]], [[0.5]], [1]),
        ([[0.5]], [[True]], [1]),
        ([[0.5]], [[0]], [0]),
        ([[0.5]], [[0]], [1.5]),
        ([[0.5]], [[0]], [True]),
    ],
)
def test_invalid_retrieval_inputs_are_rejected(scores, labels, ks):
    with pytest.raises(ValueError):
        compute_retrieval_metrics(scores, labels, ks)


def test_reverse_mapping_must_match_text_axis():
    with pytest.raises(ValueError, match="One relevance"):
        compute_bidirectional_retrieval(np.zeros((2, 3)), [[0], [1]], [[0], [1]])


def test_recall_compatibility_and_explicit_benchmark_cutoffs():
    assert compute_recall_at_k([[0.1, 0.9]], [[0]]) == {"R@1": 0.0, "R@3": 1.0, "R@5": 1.0}
    assert compute_recall_at_k([[0.1, 0.9]], [[0]], [1, 5, 10]) == {
        "R@1": 0.0, "R@5": 1.0, "R@10": 1.0,
    }

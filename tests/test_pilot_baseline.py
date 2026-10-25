"""Pilot aggregation and complete binary relevance contracts, without model weights."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "run_pilot_baseline", Path(__file__).parents[1] / "scripts/run_pilot_baseline.py"
)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def test_global_mean_normalizes_frames_before_averaging():
    # Unequal raw norms must not change the relative weight of the frames.
    actual = baseline.global_mean_features([[10, 0], [0, 1]])
    np.testing.assert_allclose(actual, [1 / np.sqrt(2), 1 / np.sqrt(2)], rtol=1e-6)


@pytest.mark.parametrize("features", [[], [[0, 0]], [[1, 0], [-1, 0]], [[np.nan, 1]]])
def test_zero_or_nonfinite_video_features_fail(features):
    with pytest.raises(ValueError):
        baseline.global_mean_features(features)


def _fixture():
    clips = [{"clip_id": "v0"}, {"clip_id": "v1"}, {"clip_id": "v2"}]
    texts = [{"text_id": "t0"}, {"text_id": "t1"}]
    relevance = [
        {"clip_id": f"v{i}", "text_id": f"t{j}", "is_match": int(i == 0 or j == 1)}
        for i in range(3)
        for j in range(2)
    ]
    return clips, texts, relevance


def test_reverse_mapping_keeps_multiple_relevant_videos():
    forward, reverse, negatives = baseline.relevance_mappings(*_fixture())
    assert forward == [[0, 1], [1], [1]]
    assert reverse == [[0], [0, 1, 2]]
    assert negatives == [[], [0], [0]]


def test_partial_pool_cannot_be_silently_treated_as_negative():
    clips, texts, relevance = _fixture()
    with pytest.raises(ValueError, match="entire candidate pool"):
        baseline.relevance_mappings(clips, texts, relevance[:-1])


def test_duplicate_pairs_cannot_change_positive_counts():
    clips, texts, relevance = _fixture()
    with pytest.raises(ValueError, match="Duplicate relevance"):
        baseline.relevance_mappings(clips, texts, relevance + [relevance[0]])


def test_split_reverse_queries_without_positives_are_excluded():
    clips, texts, relevance = _fixture()
    result = baseline.evaluate_scopes(
        np.asarray([[0.9, 0.8], [0.2, 0.7], [0.1, 0.6]]),
        clips,
        texts,
        relevance,
        {"v0": "dev", "v1": "extension", "v2": "extension"},
    )
    reverse = result["extension"]["text_to_video"]
    assert reverse["first_positive_ranks"] == [None, 1]
    assert reverse["num_evaluated_queries"] == 1
    assert reverse["num_excluded_queries"] == 1
    assert result["extension"]["video_to_text"]["num_candidates"] == 2

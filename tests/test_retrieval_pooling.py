"""Hand-computed pooling/identity checks, independent of private run artifacts."""

import numpy as np
import pytest
import torch

from temporal_hoi.inference.retrieval_pooling import (
    aligned_original_metrics,
    compare_runs,
    direct_meanp_reference,
    evaluate_bidirectional,
    paired_bootstrap,
    score_matrix,
    validate_cache,
)


def tiny_features():
    text = torch.tensor([[2.0, 0.0]])
    frames = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [999.0, 0.0]],
                           [[0.0, 2.0], [0.0, 4.0], [999.0, 0.0]]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    return text, frames, mask


def test_mean_and_query_weighted_pooling_against_hand_calculation():
    text, frames, mask = tiny_features()
    b0 = score_matrix(text, frames, mask)
    b1 = score_matrix(text, frames, mask, method="B1", tau=1.0)
    b2 = score_matrix(text, frames, mask, method="B2", tau=1.0)
    p = np.exp(1) / (np.exp(1) + 1)
    attention = p / np.sqrt(p ** 2 + (1 - p) ** 2)
    np.testing.assert_allclose(b0, [[1 / np.sqrt(2), 0]], atol=1e-7)
    np.testing.assert_allclose(b1, [[attention, 0]], atol=1e-7)
    np.testing.assert_allclose(b2, [[0.75 / np.sqrt(2) + 0.25 * attention, 0]], atol=1e-7)
    assert b0.dtype == b1.dtype == b2.dtype == np.float32
    assert b1[0, 0] > b0[0, 0]


def test_masked_frames_never_contribute_and_features_are_not_mutated():
    text, frames, mask = tiny_features()
    original = frames.clone()
    for method in ("B0", "B1", "B2"):
        np.testing.assert_allclose(
            score_matrix(text, frames, mask, method=method),
            score_matrix(text, frames[:, :2], mask[:, :2], method=method), atol=1e-7)
    assert torch.equal(frames, original)


def test_b2_endpoints_equal_frozen_baselines():
    text, frames, mask = tiny_features()
    for alpha, method in ((0.0, "B0"), (1.0, "B1")):
        np.testing.assert_array_equal(
            score_matrix(text, frames, mask, method="B2", alpha=alpha),
            score_matrix(text, frames, mask, method=method))


@pytest.mark.parametrize("method", ["B0", "B1", "B2"])
def test_chunking_preserves_full_gallery_and_scores(method):
    generator = torch.Generator().manual_seed(812)
    text = torch.randn(7, 6, generator=generator)
    frames = torch.randn(5, 4, 6, generator=generator)
    mask = torch.tensor([[True, True, False, False]] * 5)
    whole = score_matrix(text, frames, mask, method=method, query_chunk=20, video_chunk=20)
    chunked = score_matrix(text, frames, mask, method=method, query_chunk=2, video_chunk=3)
    assert chunked.shape == (7, 5)
    np.testing.assert_allclose(chunked, whole, rtol=1e-6, atol=1e-7)
    if method == "B0":
        np.testing.assert_array_equal(chunked, whole)
        np.testing.assert_allclose(whole, direct_meanp_reference(text, frames, mask), atol=2e-7)


@pytest.mark.parametrize("options", [
    {"method": "B3"}, {"tau": 0}, {"tau": float("nan")},
    {"alpha": float("nan")}, {"alpha": 1.1}, {"query_chunk": 0},
    {"query_chunk": 1.5}, {"video_chunk": True},
])
def test_invalid_scoring_parameters_are_rejected(options):
    with pytest.raises(ValueError):
        score_matrix(*tiny_features(), **options)


@pytest.mark.parametrize("kind", ["numeric_mask", "all_masked", "zero_text", "zero_frame",
                                  "nan_frame", "empty", "dimension", "zero_mean"])
def test_invalid_feature_inputs_are_rejected(kind):
    text, frames, mask = tiny_features()
    if kind == "numeric_mask":
        mask = mask.float()
    elif kind == "all_masked":
        mask[0] = False
    elif kind == "zero_text":
        text.zero_()
    elif kind == "zero_frame":
        frames[0, 0] = 0
    elif kind == "nan_frame":
        frames[0, 2, 0] = float("nan")
    elif kind == "empty":
        text = text[:0]
    elif kind == "dimension":
        text = torch.ones(1, 3)
    elif kind == "zero_mean":
        frames[0, 1] = -frames[0, 0]
    with pytest.raises(ValueError):
        score_matrix(text, frames, mask)


def valid_cache():
    text, frames, mask = tiny_features()
    return {"text": text.repeat(3, 1), "frames": frames, "mask": mask,
            "text_video_ids": ["b", "a", "a"], "video_ids": ["a", "b"],
            "text_ids": ["t0", "t1", "t2"], "metadata": {"split": "validation"}}


def test_cache_preserves_metadata_and_all_declared_positives():
    cache = validate_cache(valid_cache())
    assert cache["text_video_ids"] == ["b", "a", "a"]
    assert cache["metadata"] == {"split": "validation"}
    assert cache["text"].dtype == torch.float32
    assert cache["mask"].dtype == torch.bool


@pytest.mark.parametrize("key,value", [
    ("video_ids", ["a", "a"]), ("text_ids", ["t0", "t0", "t2"]),
    ("text_ids", ["t0", "", "t2"]), ("text_ids", ["t0", None, "t2"]),
    ("text_video_ids", ["a", "a", "a"]), ("text_video_ids", ["a", "a", "missing"]),
    ("text_video_ids", ["a"]), ("video_ids", "ab"),
])
def test_cache_rejects_missing_ambiguous_or_wrong_identities(key, value):
    cache = valid_cache()
    cache[key] = value
    with pytest.raises(ValueError):
        validate_cache(cache)


def test_bidirectional_multi_positive_and_tie_ranks_are_hand_computed():
    scores = np.array([[0.9, 0.9], [0.8, 0.1], [0.7, 0.9]], dtype=np.float32)
    result = evaluate_bidirectional(scores, ["b", "a", "a"], ["a", "b"], ["t0", "t1", "t2"])
    assert result["ranks_text_to_video"].tolist() == [2, 1, 2]
    assert result["ranks_video_to_text"].tolist() == [2, 1]
    assert result["text_to_video"]["R@1"] == pytest.approx(100 / 3)
    assert result["text_to_video"]["MRR"] == pytest.approx(2 / 3)
    assert result["video_to_text"]["R@1"] == 50
    assert result["video_to_text"]["MRR"] == 0.75
    assert result["text_to_video"]["NumPositivePairs"] == 3
    assert result["video_to_text"]["NumPositivePairs"] == 3
    assert result["ties_text_to_video"]["queries_with_top_score_tie"] == 1
    assert result["ties_video_to_text"]["queries_with_top_score_tie"] == 1


def test_legacy_tie_expansion_is_separate_from_stable_identity_metrics():
    scores = np.array([[1, 1], [0, 1]], dtype=np.float32)
    stable = evaluate_bidirectional(scores, ["a", "b"], ["a", "b"], ["ta", "tb"])
    legacy = aligned_original_metrics(scores, ["a", "b"], ["a", "b"])
    assert stable["text_to_video"]["R@1"] == 100
    assert legacy["text_to_video"]["original_num_rank_entries"] == 3
    assert legacy["text_to_video"]["R@1"] == pytest.approx(200 / 3)
    assert not aligned_original_metrics(np.zeros((3, 2)), ["a", "a", "b"], ["a", "b"])["available"]


def test_bootstrap_is_paired_by_video_and_keeps_full_gallery():
    result = paired_bootstrap([2, 1, 1], [1, 2, 1], ["a", "b", "b"], n_bootstrap=20, seed=42)
    assert result == paired_bootstrap([2, 1, 1], [1, 2, 1], ["a", "b", "b"], n_bootstrap=20, seed=42)
    assert result["clusters"] == 2
    assert result["delta"]["R@1_pp"]["estimate"] == 0
    assert result["delta"]["R@1_pp"]["ci_low"] < 0
    assert result["delta"]["R@1_pp"]["ci_high"] > 0
    assert result["gallery"] == "fixed; ranks computed before bootstrap"
    with pytest.raises(ValueError):
        paired_bootstrap([1.5], [1], ["a"])


def test_compare_exports_both_corrected_and_broken_queries(tmp_path):
    ids, tids = ["a", "b"], ["ta", "tb"]
    before = evaluate_bidirectional([[0, 1], [0, 1]], ids, ids, tids)
    after = evaluate_bidirectional([[1, 0], [1, 0]], ids, ids, tids)
    output = tmp_path / "changed.csv"
    result = compare_runs(before, after, ids, ids, tids, n_bootstrap=10, seed=42, output_csv=output)
    assert result["text_to_video"]["corrected_at_1"] == 1
    assert result["text_to_video"]["broken_at_1"] == 1
    assert len(result["changed_queries"]) > 0
    assert "rank_before,rank_after,r1_change" in output.read_text(encoding="utf-8")

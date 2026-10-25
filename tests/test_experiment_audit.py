"""Hand-calculated seed dispersion and deliberately leaky development manifests."""

import copy
import math

import pytest
import torch

from temporal_hoi.evaluation.experiment_audit import audit_zero_shot_usage, summarize_seeds
from temporal_hoi.models.alignment import MLPAlignmentHead, build_alignment_head
from temporal_hoi.training.losses import bidirectional_multi_positive_loss


def test_mlp_gradients_normalization_and_frozen_boundary():
    torch.manual_seed(42)
    head = MLPAlignmentHead(3, 3, hidden_dim=5)
    video = torch.eye(3, requires_grad=True)
    text = torch.tensor([[1., .2, .3], [.2, 1., .3], [.2, .3, 1.]], requires_grad=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
    before = [p.detach().clone() for p in head.parameters()]
    loss, _ = bidirectional_multi_positive_loss(head(video, text), torch.eye(3))
    loss.backward()
    assert video.grad is None and text.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
               for p in head.parameters())
    optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, head.parameters(), strict=True))
    torch.testing.assert_close(head.encode_video(video).norm(dim=-1), torch.ones(3))
    assert sum(p.numel() for p in head.parameters()) == 2 * (3 * 5 + 5 + 5 * 3 + 3)
    for bad in (torch.zeros(3, 3), torch.full((3, 3), math.nan), torch.ones(3, 4)):
        with pytest.raises(ValueError, match="finite nonzero"):
            head.encode_video(bad)
    with pytest.raises(ValueError, match="hidden_dim"):
        MLPAlignmentHead(3, 3, True)
    with pytest.raises(ValueError, match="linear or mlp"):
        build_alignment_head({"head": "unimplemented"}, 3, 3)


def records():
    return [{"seed": 42 + i, "value": value, "metric": "text_to_video.R@1",
             "units": "fraction", "comparison_sha256": "same_config",
             "input_sha256": "same_input", "execution_level": "synthetic_technical_probe"}
            for i, value in enumerate((.2, .4, .6))]


def test_sample_standard_deviation_and_one_seed_not_zero_uncertainty():
    result = summarize_seeds(records())
    assert result["mean"] == pytest.approx(.4)
    assert result["sample_std"] == pytest.approx(.2)
    assert result["execution_level"] == "synthetic_technical_probe"
    single = summarize_seeds(records()[:1])
    assert single["sample_std"] is None and single["interpretation"] == "one-seed exploratory"


@pytest.mark.parametrize("key,value", [
    ("seed", 42), ("value", float("nan")), ("value", True),
    ("input_sha256", "different"), ("comparison_sha256", "different"),
    ("execution_level", "benchmark"), ("units", "percent"), ("metric", "video_to_text.R@1"),
])
def test_noncomparable_or_duplicate_runs_are_rejected(key, value):
    rows = records()
    rows[1][key] = value
    with pytest.raises(ValueError):
        summarize_seeds(rows)


def manifest():
    return {
        "schema_version": "zero_shot_usage_v1", "seen_classes": ["hold_cup"],
        "pseudo_unseen_classes": ["move_cup"], "final_unseen_classes": ["move_bottle"],
        "samples": [
            {"sample_id": "a", "video_id": "train_video", "split": "train",
             "positive_classes": ["hold_cup"], "negative_classes": [],
             "unknown_classes": ["move_cup", "move_bottle"], "context_positive_classes": []},
            {"sample_id": "b", "video_id": "selection_video", "split": "selection",
             "positive_classes": ["move_cup"], "negative_classes": [],
             "unknown_classes": [], "context_positive_classes": []},
        ],
        "uses": [{"sample_id": "a", "purpose": "cache_training"},
                 {"sample_id": "b", "purpose": "checkpoint_selection"}],
    }


def test_clean_manifest_unknowns_not_negative_and_does_not_certify_benchmark():
    result = audit_zero_shot_usage(manifest())
    assert result["status"] == "manifest_consistent"
    assert not result["benchmark_evaluated"] and result["scope"] == "declared_manifest_only"


@pytest.mark.parametrize("purpose", ["validation_selection", "prompt_selection", "checkpoint_selection"])
def test_final_scores_cannot_choose_prompts_or_checkpoints(purpose):
    data = manifest()
    data["samples"][1]["split"] = "final"
    data["uses"][1]["purpose"] = purpose
    assert audit_zero_shot_usage(data)["status"] == "failed"


@pytest.mark.parametrize("field", ["positive_classes", "negative_classes", "context_positive_classes"])
@pytest.mark.parametrize("label", ["move_cup", "move_bottle"])
def test_heldout_labels_in_window_cache_or_negative_targets_are_detected(field, label):
    data = manifest()
    data["samples"][0]["unknown_classes"] = []
    data["samples"][0][field] = [label]
    assert audit_zero_shot_usage(data)["status"] == "failed"


def test_unknown_negative_conflict_video_leakage_and_unresolved_use():
    data = manifest()
    data["samples"][0]["negative_classes"] = ["move_bottle"]
    assert audit_zero_shot_usage(data)["status"] == "failed"
    data = manifest()
    data["samples"][1]["video_id"] = data["samples"][0]["video_id"]
    assert audit_zero_shot_usage(data)["status"] == "failed"
    data = manifest()
    data["uses"][0]["sample_id"] = "missing"
    with pytest.raises(ValueError, match="resolve"):
        audit_zero_shot_usage(data)
    data = manifest()
    data["samples"].append(copy.deepcopy(data["samples"][0]))
    with pytest.raises(ValueError, match="unique"):
        audit_zero_shot_usage(data)

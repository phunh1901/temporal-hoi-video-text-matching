"""Behavioral controls for frozen-feature learning; fixtures are not VidHOI results."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from temporal_hoi.data.vidhoi import ClassMapping, HOIPair
from temporal_hoi.inference.hoi_static import OracleEntity, OracleFrame
from temporal_hoi.models.hoi_head import (
    HOIResidualHead,
    identity_projection,
    positive_anchor_loss,
    view_features,
)
from temporal_hoi.training.hoi_head_study import (
    evaluate_head_state,
    run_head_study,
    selection_key,
    study_config,
    validate_cache,
)


def _cache(role, *, positive=0):
    mapping = ClassMapping({0: "person", 1: "cup"}, {0: "hold", 1: "open", 2: "touch"})
    person = OracleEntity("p1", 0, (0, 0, 4, 8))
    cup = OracleEntity("c1", 1, (6, 0, 8, 4))
    other = OracleEntity("c2", 1, (8, 0, 10, 4))
    labels = tuple(float(predicate == positive) for predicate in range(3))
    truth = HOIPair(0, 1, person.bbox_xyxy, cup.bbox_xyxy, labels,
                    subject_track_id="p1", object_track_id="c1")
    frame = OracleFrame(
        f"fixture/{role}", 1, 0, 0.0, Path("not_read_by_feature_training.png"), "0" * 64,
        12, 8, (person, cup, other), (truth,))
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, positive] = True
    metadata = {
        "role": role, "level": "synthetic_encoder_fixture", "test_fixture": True,
        "source_split": "train", "label_policy": "positive_only_unknown",
        "final_unseen_accessed": False, "official_final_split_accessed": False,
        "pseudo_unseen_in_train": False,
        **{name: "a" * 64 for name in (
            "protocol_sha256", "mapping_sha256", "checkpoint_sha256", "config_sha256")},
    }
    return {
        "object_features": np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.float32),
        "union_features": np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.float32),
        "object_class": np.asarray([1, 1], dtype=np.int64),
        "positive_mask": mask, "frame_index": np.asarray([0, 0], dtype=np.int64),
        "subject_track_id": np.asarray(["p1", "p1"]),
        "object_track_id": np.asarray(["c1", "c2"]),
        "mapping": mapping, "frames": [frame], "metadata": metadata,
    }


def _texts():
    return np.tile(np.eye(3, dtype=np.float32)[None], (2, 1, 1)), np.ones((2, 3), dtype=bool)


def test_zero_residual_equals_frozen_and_detaches_backbone():
    torch.manual_seed(42)
    obj = torch.tensor([[1.0, 2.0, 0.0]], requires_grad=True)
    union = torch.tensor([[0.0, 2.0, 1.0]], requires_grad=True)
    for mode in ("object", "union", "fusion"):
        head = HOIResidualHead(3)
        base = view_features(obj, union, mode)
        assert torch.equal(head(base), identity_projection(base))
        head(base)[0, 0].backward()
        assert obj.grad is None and union.grad is None
        assert head.residual[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="cancel"):
        view_features(torch.ones(1, 3), -torch.ones(1, 3), "fusion")


def test_unknown_text_never_enters_positive_loss_or_gradient():
    projected = torch.tensor([[0.6, 0.8, 0.0]], requires_grad=True)
    base = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    texts = torch.eye(3).repeat(2, 1, 1).requires_grad_()
    positive = torch.tensor([[True, False, False]])
    allowed = torch.ones((2, 3), dtype=torch.bool)
    losses = positive_anchor_loss(projected, base, torch.tensor([1]), positive, texts, allowed)
    assert losses["positive_loss"].item() == pytest.approx(0.4)
    assert losses["anchor_loss"].item() == pytest.approx(0.4)
    assert losses["loss"].item() == pytest.approx(0.8)
    gradient = torch.autograd.grad(losses["loss"], projected)[0]
    changed = texts.detach().clone()
    changed[1, 1:] = torch.tensor([[100.0, -22.0, 7.0], [-5.0, 12.0, 33.0]])
    repeated = positive_anchor_loss(projected, base, torch.tensor([1]), positive, changed, allowed)
    assert torch.equal(losses["loss"], repeated["loss"])
    assert torch.equal(gradient, torch.autograd.grad(repeated["loss"], projected)[0])
    losses = positive_anchor_loss(projected, base, torch.tensor([1]), positive, texts, allowed)
    losses["loss"].backward()
    assert base.grad is None and texts.grad is None
    allowed[1, 0] = False
    with pytest.raises(ValueError, match="excluded"):
        positive_anchor_loss(projected, base, torch.tensor([1]), positive, texts, allowed)


def test_multilabel_loss_weights_labeled_pairs_and_all_pair_anchor_separately():
    projected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    base = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masks = torch.tensor([[True, True, False], [False, False, False]])
    losses = positive_anchor_loss(
        projected, base, torch.tensor([1, 1]), masks, torch.eye(3).repeat(2, 1, 1),
        torch.ones((2, 3), dtype=torch.bool))
    assert losses["positive_loss"].item() == pytest.approx(0.5)
    assert losses["anchor_loss"].item() == pytest.approx(0.5)
    assert (losses["labeled_pairs"], losses["positive_labels"], losses["all_pairs"]) == (1, 2, 2)


@pytest.mark.parametrize("mutation", ["row_order", "labels", "extra", "missing", "gt_missing"])
def test_cache_cannot_change_proposal_or_annotation_inventory(mutation):
    cache = _cache("train")
    if mutation == "row_order":
        cache["object_track_id"] = cache["object_track_id"][::-1]
    elif mutation == "labels":
        cache["positive_mask"][1, 2] = True
    elif mutation in {"extra", "missing"}:
        for name, value in list(cache.items()):
            if isinstance(value, np.ndarray):
                cache[name] = np.concatenate([value, value[:1]]) if mutation == "extra" else value[:1]
    else:
        frame = cache["frames"][0]
        truth = replace(frame.ground_truth[0], object_track_id="absent")
        cache["frames"] = [replace(frame, ground_truth=(truth,))]
    with pytest.raises(ValueError):
        validate_cache(cache)


def test_excluded_text_and_scores_stay_zero_without_dropping_unknown_pairs():
    cache = _cache("validation")
    texts, allowed = _texts()
    allowed[1, 2] = False
    texts[1, 2] = 0
    result = evaluate_head_state(None, cache, texts, allowed, "union")
    assert result["score_values"].shape == (2, 3)
    assert np.all(result["score_values"][:, 2] == 0)
    assert len(result["frames"][0].predictions) == 2
    assert result["known_positive_ranks"]["records"][0]["num_candidate_predicates"] == 2
    texts[1, 2] = 1
    with pytest.raises(ValueError, match="exactly zero"):
        evaluate_head_state(None, cache, texts, allowed, "union")


@pytest.mark.parametrize("mutation", ["overlap", "hash", "official", "fixture_level"])
def test_study_rejects_leakage_and_wrong_provenance_before_training(tmp_path, mutation):
    train, dev = _cache("train", positive=1), _cache("validation")
    texts, allowed = _texts()
    if mutation == "overlap":
        dev["frames"] = [replace(dev["frames"][0], video_id=train["frames"][0].video_id)]
    elif mutation == "hash":
        dev["metadata"]["protocol_sha256"] = "b" * 64
    elif mutation == "official":
        dev["metadata"]["official_final_split_accessed"] = True
    else:
        dev["metadata"]["test_fixture"] = False
    with pytest.raises(ValueError):
        run_head_study(train, dev, texts, allowed, {"test_fixture": True, "max_epochs": 1},
                       tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_checkpoint_selection_prefers_map_then_rank_then_earliest():
    assert selection_key({"map": 0.6}, {"mean_rank": 4}, 2) > selection_key(
        {"map": 0.5}, {"mean_rank": 1}, 0)
    assert selection_key({"map": 0.6}, {"mean_rank": 2}, 4) > selection_key(
        {"map": 0.6}, {"mean_rank": 3}, 0)
    assert selection_key({"map": 0.6}, {"mean_rank": 2}, 0) > selection_key(
        {"map": 0.6}, {"mean_rank": 2}, 4)
    with pytest.raises(ValueError, match="explicit test_fixture"):
        study_config({"max_epochs": 1})
    assert study_config({"study": {"device": "cuda"}, "device": "cpu"})["device"] == "cpu"


def test_nine_actual_optimization_runs_preserve_epoch_zero_when_dev_does_not_improve(tmp_path):
    train, dev = _cache("train", positive=1), _cache("validation", positive=0)
    original = copy.deepcopy(train)
    texts, allowed = _texts()
    output = tmp_path / "study"
    result = run_head_study(
        train, dev, texts, allowed, {"test_fixture": True, "max_epochs": 1}, output)
    assert result["level"] == "synthetic_feature_head_study"
    assert len(result["runs"]) == 9 and len(result["frozen_baselines"]) == 3
    assert result["selected_best_seed"] is None
    for run in result["runs"]:
        assert run["selected_epoch"] == 0 and run["last_epoch"] == 1
        assert run["optimizer_steps"] == 1 and run["nonzero_gradient_observed"]
        assert run["last_parameter_delta_l2"] > 0 and run["last_embedding_delta_l2"] > 0
        assert run["selected_parameter_delta_l2"] == 0
        directory = output / run["directory"]
        initial = load_file(str(directory / "initial_head.safetensors"))
        selected = load_file(str(directory / "selected_head.safetensors"))
        last = load_file(str(directory / "last_head.safetensors"))
        assert all(torch.equal(initial[key], selected[key]) for key in initial)
        assert any(not torch.equal(initial[key], last[key]) for key in initial)
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        assert [row["epoch"] for row in history] == [0, 1]
        head = HOIResidualHead(3)
        head.load_state_dict(selected)
        replay = evaluate_head_state(head, dev, texts, allowed, run["mode"])
        assert replay["metrics"] == run["selected_metrics"]
    assert len({run["parameter_count"] for run in result["runs"]}) == 1
    for mode in ("object", "union", "fusion"):
        assert result["aggregate"][mode]["selected_epochs"] == [0, 0, 0]
        assert result["aggregate"][mode]["selected_best_seed"] is None
    for name, values in original.items():
        if isinstance(values, np.ndarray):
            assert np.array_equal(values, train[name])


def test_zero_gradient_cannot_be_declared_completed_learning(tmp_path):
    train, dev = _cache("train"), _cache("validation")
    texts, allowed = _texts()
    with pytest.raises(ValueError, match="No verified"):
        run_head_study(train, dev, texts, allowed, {"test_fixture": True, "max_epochs": 1},
                       tmp_path / "no_update")

"""Regime summaries keep the original evaluator budget/matching and actual supervision."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from temporal_hoi.data.vidhoi import HOIPair, VidHOIFrame
from temporal_hoi.evaluation.hoi import evaluate_frame_hoi
from temporal_hoi.evaluation.hoi_head_audit import summarize_development_regimes
from temporal_hoi.inference.hoi_static import OracleEntity, OracleFrame


def _frame():
    person = OracleEntity("p", 0, (0, 0, 4, 8))
    first = OracleEntity("one", 1, (6, 0, 8, 4))
    second = OracleEntity("two", 2, (8, 0, 10, 4))
    truth = (
        HOIPair(0, 1, person.bbox_xyxy, first.bbox_xyxy, (1, 0, 0, 0),
                subject_track_id="p", object_track_id="one"),
        HOIPair(0, 2, person.bbox_xyxy, second.bbox_xyxy, (0, 1, 0, 0),
                subject_track_id="p", object_track_id="two"),
    )
    return OracleFrame("train/v", 1, 0, 0, Path("not_read.png"), "0" * 64, 12, 8,
                       (person, first, second), truth)


@pytest.fixture
def study():
    seen = [[0, 0, 1], [0, 1, 2], [0, 1, 1]]
    classes = [
        {"triplet": [0, 0, 1], "ap": 0.2, "gt_count": 2},
        {"triplet": [0, 1, 1], "ap": 0.4, "gt_count": 3},
        {"triplet": [0, 0, 2], "ap": 0.6, "gt_count": 4},
        {"triplet": [0, 2, 3], "ap": 0.8, "gt_count": 5},
        {"triplet": [0, 1, 4], "ap": 0.0, "gt_count": 1},
    ]
    protocol = {
        "schema_version": "vidhoi_head_protocol_v1", "label_policy": "positive_only_unknown",
        "train_video_ids": ["train/v"], "dev_video_ids": ["dev/v"],
        "seen_triplets": seen, "pseudo_unseen_triplets": [[0, 0, 2]],
        "final_unseen_triplets": [[0, 3, 2]],
        "video_positive_triplets": {"train/v": copy.deepcopy(seen),
                                     "dev/v": [row["triplet"][:] for row in classes]},
    }
    metrics = {
        "evaluator": "ST-HOI_notebook_compat_v1", "metric_scale": "fraction",
        "class_subset": "full", "split": "research_validation", "box_source": "oracle",
        "top_k_before_class_filter": True, "gt_matching_scope": "shared_pair_across_predicates_per_frame",
        "top_k": 100, "iou_threshold": 0.5, "num_classes": 5, "map": 0.4,
        "ap_definition": "VOC_11_point", "classes": classes,
    }
    return metrics, protocol, [_frame()]


def test_partition_uses_actual_sampled_supervision_and_macro_class_ap(study):
    before = copy.deepcopy(study)
    result = summarize_development_regimes(*study)
    groups = result["groups"]
    for name, ap, instances in (
        ("sampled_train_seen", 0.2, 2), ("whole_train_seen_not_sampled", 0.4, 3),
        ("designed_pseudo_unseen", 0.6, 4), ("incidental_dev_only", 0.4, 6),
    ):
        assert groups[name]["map"] == pytest.approx(ap)
        assert groups[name]["num_gt_instances"] == instances
    assert groups["incidental_dev_only"]["num_gt_supported_classes"] == 2
    assert groups["incidental_dev_only"]["predicate_novelty_in_sampled_train_classes"] == 1
    assert groups["incidental_dev_only"]["object_novelty_in_sampled_train_classes"] == 2
    by_triplet = {tuple(row["triplet"]): row for row in result["composition_support"]}
    assert by_triplet[0, 1, 1]["sampled_train_gt_instances"] == 0
    assert by_triplet[0, 1, 1]["present_in_whole_training_video_metadata"]
    assert by_triplet[0, 2, 3]["predicate_unseen_in_sampled_train"]
    assert by_triplet[0, 2, 3]["object_unseen_in_sampled_train"]
    assert result["designed_pseudo_component_audit"] == [{
        "triplet": [0, 0, 2], "predicate_sampled_train_gt_instances": 1,
        "object_sampled_train_gt_instances": 1, "composition_sampled_train_gt_instances": 0}]
    assert result["num_gt_instances"] == 15 and result["num_gt_supported_classes"] == 5
    assert result["selection_uses_all_development_map"]
    assert not result["groups_used_to_reselect_checkpoint"]
    assert not result["zero_shot_benchmark_claim"]
    assert json.loads(json.dumps(result)) == result
    assert study == before


def test_actual_shared_pair_matching_is_kept_across_groups():
    train = _frame()
    truth = replace(train.ground_truth[0], predicate_values=(1, 1, 0, 0))
    prediction = replace(truth, predicate_values=(2.0, 1.0, 0.5, 0.1))
    dev = VidHOIFrame("dev/v", 0, "research_validation", "oracle", "xyxy_pixels_original_frame",
                      (truth,), (prediction,), 0)
    metrics = evaluate_frame_hoi([dev])
    protocol = {
        "schema_version": "vidhoi_head_protocol_v1", "label_policy": "positive_only_unknown",
        "train_video_ids": ["train/v"], "dev_video_ids": ["dev/v"],
        "seen_triplets": [[0, 0, 1], [0, 1, 2]],
        "pseudo_unseen_triplets": [[0, 1, 1]], "final_unseen_triplets": [],
        "video_positive_triplets": {"train/v": [[0, 0, 1], [0, 1, 2]],
                                     "dev/v": [[0, 0, 1], [0, 1, 1]]},
    }
    result = summarize_development_regimes(metrics, protocol, [train])
    # The higher-scored sampled-seen predicate already consumed the shared GT pair.
    # Re-evaluating the pseudo group independently would incorrectly give it AP 1.
    assert result["groups"]["sampled_train_seen"]["map"] == pytest.approx(1)
    assert result["groups"]["designed_pseudo_unseen"]["map"] == 0
    assert result["groups"]["whole_train_seen_not_sampled"]["map"] is None
    assert result["groups"]["incidental_dev_only"]["map"] is None
    assert result["all_development_map"] == pytest.approx(0.5)
    assert result["global_matching_unchanged"] and result["prediction_budget_unchanged"]


def test_incidental_components_distinguish_full_video_and_sampled_support(study):
    metrics, protocol, frames = study
    protocol["seen_triplets"].append([0, 2, 1])
    protocol["video_positive_triplets"]["train/v"].append([0, 2, 1])
    result = summarize_development_regimes(metrics, protocol, frames)
    row = next(row for row in result["composition_support"] if row["triplet"] == [0, 2, 3])
    assert row["regime"] == "incidental_dev_only"
    assert row["predicate_unseen_in_sampled_train"]
    assert not row["predicate_unseen_in_whole_training_videos"]
    assert row["object_unseen_in_whole_training_videos"]


@pytest.mark.parametrize("component", ["predicate", "object"])
def test_pseudo_components_must_be_observed_in_sampled_training_not_only_video_metadata(study, component):
    metrics, protocol, frames = study
    if component == "predicate":
        pseudo = [0, 2, 1]
        protocol["seen_triplets"].append([0, 2, 2])
        protocol["video_positive_triplets"]["train/v"].append([0, 2, 2])
    else:
        pseudo = [0, 0, 3]
        protocol["seen_triplets"].append([0, 1, 3])
        protocol["video_positive_triplets"]["train/v"].append([0, 1, 3])
    protocol["pseudo_unseen_triplets"] = [pseudo]
    with pytest.raises(ValueError, match="actual sampled training"):
        summarize_development_regimes(metrics, protocol, frames)


@pytest.mark.parametrize("mutation", ["seen", "support", "class_duplicate", "nan", "map",
                                       "video_overlap", "training_identity", "final"])
def test_regime_audit_rejects_inconsistent_evidence(study, mutation):
    metrics, protocol, frames = study
    if mutation == "seen":
        protocol["seen_triplets"].append([0, 0, 4])
    elif mutation == "support":
        metrics["classes"][0]["gt_count"] = 0
    elif mutation == "class_duplicate":
        metrics["classes"][1]["triplet"] = [0, 0, 1]
    elif mutation == "nan":
        metrics["classes"][0]["ap"] = float("nan")
    elif mutation == "map":
        metrics["map"] = 0.9
    elif mutation == "video_overlap":
        protocol["dev_video_ids"] = ["train/v"]
    elif mutation == "training_identity":
        frames[0] = replace(frames[0], video_id="other/v")
    else:
        protocol["video_positive_triplets"]["dev/v"].append([0, 3, 2])
    with pytest.raises(ValueError):
        summarize_development_regimes(metrics, protocol, frames)


def test_empty_evaluator_subset_has_null_groups(study):
    metrics, protocol, frames = study
    metrics.update(classes=[], num_classes=0, map=None)
    result = summarize_development_regimes(metrics, protocol, frames)
    assert result["all_development_map"] is None
    assert result["num_gt_instances"] == result["num_gt_supported_classes"] == 0
    assert all(group["map"] is None for group in result["groups"].values())

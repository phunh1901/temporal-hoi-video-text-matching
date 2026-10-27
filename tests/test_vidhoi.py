"""Identity, annotation track joins, mask boundaries and coordinate validation."""

import copy
import math

import pytest

from temporal_hoi.data.vidhoi import (
    BoxTransform,
    ClassMapping,
    adapt_sthoi_result,
    adapt_vidhoi_annotations,
    select_frame_mask,
    validate_video_partitions,
)


@pytest.fixture
def mapping():
    return ClassMapping({0: "person", 1: "cup"}, {0: "hold", 1: "watch"})


@pytest.fixture
def row():
    return {
        "gt_boxes": [[0, 0, 0, 10, 10], [0, 20, 0, 30, 10]],
        "gt_obj_classes": [[0, 0], [0, 1]], "gt_bbox_pair_ids": [[0, 1]],
        "gt_action_labels": [[1, 0]], "proposal_boxes": [[0, 0, 0, 10, 10],
                                                           [0, 20, 0, 30, 10]],
        "proposal_classes": [[0, 0], [0, 1]], "proposal_scores": [[0, 1], [0, 1]],
        "preds_bbox_pair_ids": [[0, 1]], "preds_score": [[0.9, 0.1]],
    }


def adapt(row, mapping, **kwargs):
    return adapt_sthoi_result(row, mapping, video_id="folder/video_with_underscore", frame_id=12,
                              split="official_val", box_source="oracle", **kwargs)


def test_identity_is_explicit_and_unknown_time_tracks_are_not_invented(row, mapping):
    frame = adapt(row, mapping)
    assert frame.key == "folder/video_with_underscore_000012"
    assert frame.timestamp_s is None and frame.ground_truth[0].subject_track_id is None
    assert frame.coordinate_system == "xyxy_pixels_model_frame"
    assert frame.ground_truth[0].subject_box == (0, 0, 10, 10)


def test_coordinate_transform_applies_to_GT_and_predictions(row, mapping):
    frame = adapt(row, mapping, box_transform=BoxTransform(2, 4, 2, -4),
                  gt_track_ids={0: "person-7", 1: "cup-3"})
    assert frame.ground_truth[0].subject_box == (-1, 1, 4, 3.5)
    assert frame.predictions[0].subject_box == frame.ground_truth[0].subject_box
    assert frame.coordinate_system == "xyxy_pixels_original_frame"
    assert frame.ground_truth[0].subject_track_id == "person-7"


@pytest.mark.parametrize("field,value", [
    ("gt_boxes", [[0, 10, 0, 0, 10], [0, 20, 0, 30, 10]]),
    ("gt_bbox_pair_ids", [[0, 8]]),
    ("gt_action_labels", [[0.5, 0]]),
    ("preds_score", [[math.nan, 0.1]]),
    ("proposal_classes", [[0, 0], [0, 77]]),
    ("proposal_scores", [[0, 1]]),
    ("proposal_boxes", [[1, 0, 0, 10, 10], [0, 20, 0, 30, 10]]),
])
def test_invalid_native_rows_fail_closed(row, mapping, field, value):
    row[field] = value
    with pytest.raises(ValueError):
        adapt(row, mapping)


def test_missing_field_and_mapping_person_ID_are_rejected(row, mapping):
    del row["gt_boxes"]
    with pytest.raises(ValueError, match="Missing ST-HOI"):
        adapt(row, mapping)
    with pytest.raises(ValueError, match="person"):
        ClassMapping({0: "cup", 1: "person"}, {0: "hold"})


def test_masks_are_ID_based_preserve_result_order_and_reject_missing(row, mapping):
    first = adapt(row, mapping)
    second = adapt_sthoi_result(row, mapping, video_id="other", frame_id=5, split="official_val",
                               box_source="oracle")
    assert select_frame_mask([first, second], [second.key, first.key]) == [first, second]
    for mask in ([], [first.key, first.key], ["unknown_000001"]):
        with pytest.raises(ValueError):
            select_frame_mask([first, second], mask)
    with pytest.raises(ValueError, match="Duplicate result"):
        select_frame_mask([first, first], [first.key])


def test_video_partitions_prevent_split_leakage():
    validate_video_partitions({"train": ["a"], "research_validation": ["b"]})
    with pytest.raises(ValueError, match="leaking"):
        validate_video_partitions({"train": ["a"], "research_validation": ["a"]})


def test_annotation_adapter_preserves_tracks_and_merges_multilabel_pair(mapping):
    row = {"video_folder": "a", "video_id": "v", "frame_id": 4, "person_id": 11,
           "object_id": 20, "object_class": 1, "action_class": 0,
           "person_box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
           "object_box": {"xmin": 20, "ymin": 0, "xmax": 30, "ymax": 10}}
    other = copy.deepcopy(row)
    other["action_class"] = 1
    manifest = [{"video_id": "a/v", "source_frame_id": 4, "frame_id": 106, "timestamp_s": 3.5}]
    frame = adapt_vidhoi_annotations([row, other], mapping, manifest, split="train")[0]
    assert frame.frame_id == 106 and frame.timestamp_s == 3.5
    assert frame.ground_truth[0].predicate_values == (1, 1)
    assert (frame.ground_truth[0].subject_track_id, frame.ground_truth[0].object_track_id) == (
        "11", "20"
    )
    other["person_box"]["xmax"] = 15
    with pytest.raises(ValueError, match="Inconsistent"):
        adapt_vidhoi_annotations([row, other], mapping, manifest, split="train")
    with pytest.raises(ValueError, match="every annotation"):
        adapt_vidhoi_annotations([row], mapping, [], split="train")

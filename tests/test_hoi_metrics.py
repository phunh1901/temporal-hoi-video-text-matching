"""Hand-calculated frame-HOI expectations, independent of model predictions."""

from dataclasses import replace

import numpy as np
import pytest

from temporal_hoi.data.vidhoi import HOIPair, VidHOIFrame
from temporal_hoi.evaluation.hoi import bbox_iou, evaluate_frame_hoi, voc_11_point_ap


def make_frame(predicates=(1.0, 0.0), scores=(0.9, 0.1), frame_id=0):
    pair = HOIPair(0, 1, (0, 0, 10, 10), (20, 0, 30, 10), predicates)
    return VidHOIFrame("fixture", frame_id, "fixture", "oracle", "xyxy_pixels_original_frame",
                       (pair,), (replace(pair, predicate_values=scores),))


def test_IoU_and_VOC11_are_hand_calculated():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 20, 10)) == 0.5
    assert bbox_iou((0, 0, 10, 10), (20, 0, 30, 10)) == 0
    # Thresholds 0..0.5: six precisions of 1; 0.6..1: five precisions of 0.5.
    assert voc_11_point_ap(np.array([0.5, 1.0]), np.array([1.0, 0.5])) == pytest.approx(17 / 22)


def test_inclusive_IoU_and_duplicate_pair_matching():
    frame = make_frame()
    lower = replace(frame.predictions[0], predicate_values=(0.8, 0.01))
    frame = replace(frame, predictions=(frame.predictions[0], lower))
    result = evaluate_frame_hoi([frame])
    assert result["map"] == pytest.approx(1)
    assert result["classes"][0]["sorted_tp"] == [1, 0]
    boundary = replace(frame.predictions[0], subject_box=(0, 0, 20, 10))
    assert evaluate_frame_hoi([replace(frame, predictions=(boundary,))])["map"] == pytest.approx(1)


def test_shared_GT_pair_preserves_upstream_multilabel_behavior_and_tie_order():
    result = evaluate_frame_hoi([make_frame((1, 1), (0.9, 0.9))])
    assert result["map"] == pytest.approx(0.5)
    assert [row["ap"] for row in result["classes"]] == pytest.approx([0, 1])
    assert result["gt_matching_scope"] == "shared_pair_across_predicates_per_frame"


def test_prediction_absence_and_noGT_subset_are_distinct():
    frame = replace(make_frame(), predictions=())
    assert evaluate_frame_hoi([frame])["map"] == 0
    empty_subset = evaluate_frame_hoi([frame], class_subset="nonrare")
    assert empty_subset["map"] is None and empty_subset["num_classes"] == 0
    assert empty_subset["status"] == "no_GT_supported_classes"


def test_rare_25_boundary_and_macro_average_are_not_half_rare_half_nonrare():
    frames = [make_frame((1, 0), frame_id=index) for index in range(24)]
    frames += [make_frame((0, 1), (0.1, 0.9), frame_id=index + 24) for index in range(25)]
    rare = evaluate_frame_hoi(frames, class_subset="rare")
    nonrare = evaluate_frame_hoi(frames, class_subset="nonrare")
    assert rare["classes"][0]["gt_count"] == 24
    assert nonrare["classes"][0]["gt_count"] == 25
    assert rare["classes"][0]["triplet"] == [0, 0, 1]
    assert nonrare["classes"][0]["triplet"] == [0, 1, 1]


def test_topk_is_applied_before_GT_supported_class_filter():
    frame = make_frame()
    noise = replace(frame.predictions[0], object_class=2, predicate_values=(1.0, 1.0))
    frame = replace(frame, predictions=(frame.predictions[0], noise))
    assert evaluate_frame_hoi([frame], top_k=2)["map"] == 0
    assert evaluate_frame_hoi([frame], top_k=3)["map"] == pytest.approx(1)


def test_nonpositive_scores_use_finite_floor():
    frame = make_frame(scores=(0, -1))
    frame = replace(frame, predictions=(replace(frame.predictions[0], subject_score=-1),))
    result = evaluate_frame_hoi([frame])
    assert result["map"] == pytest.approx(1)
    assert np.isfinite(result["classes"][0]["sorted_scores"]).all()


def test_mixed_box_source_split_coordinate_system_and_duplicate_ids_fail():
    first = make_frame()
    for field, value in (("box_source", "predicted"), ("split", "train"),
                         ("coordinate_system", "xyxy_pixels_model_frame")):
        with pytest.raises(ValueError, match=f"mix {field}"):
            evaluate_frame_hoi([first, replace(first, frame_id=1, **{field: value})])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_frame_hoi([first, first])
    with pytest.raises(ValueError, match="at least one"):
        evaluate_frame_hoi([])

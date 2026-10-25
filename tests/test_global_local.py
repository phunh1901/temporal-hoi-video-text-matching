"""Single-variable global/local retrieval comparison contracts."""

import copy

import pytest

from temporal_hoi.evaluation.global_local import validate_paired_runs


def _runs():
    reference = {"status": "PASS_TECHNICAL_PILOT", "baseline": "B0_frozen_openclip_global_mean",
                 "checkpoint": {"sha256": "encoder"},
                 "inputs": dict.fromkeys(["manifest_lock_sha256", "manifest_sha256", "config_sha256",
                                          "source_videos", "clip_ids", "text_ids", "positive_pairs",
                                          "annotated_pairs"], "locked"),
                 "environment": dict.fromkeys(["seed", "batch_size", "num_workers", "precision",
                                               "device", "torch_num_threads", "torch_num_interop_threads",
                                               "versions"], "same"),
                 "preprocess": {"pooling": "mean", "visual_view": "global"},
                 "head": {"name": "frozen_identity"}, "coverage": {"clips": 1},
                 "clip_records": [{"clip_id": "v", "split": "dev", "start_s": 0, "end_s": 2,
                                   "frames": 1, "timestamps_s": [0], "video_metadata": {"fps": 24}}]}
    local = copy.deepcopy(reference)
    local["baseline"] = "B1_frozen_openclip_union_mean"
    local["preprocess"]["visual_view"] = "person_object_union"
    local["box_source"] = {"source_type": "detector", "full_frame_fallback": False}
    return reference, local


def test_only_view_changes_are_allowed_in_a_paired_run():
    validate_paired_runs(*_runs())


@pytest.mark.parametrize("part,key,new_value", [
    ("head", "name", "trained_head"), ("checkpoint", "sha256", "other_encoder"),
    ("inputs", "text_ids", ["changed_pool"]), ("environment", "seed", 43),
    ("preprocess", "pooling", "attention"), ("coverage", "clips", 2),
    ("box_source", "full_frame_fallback", True),
])
def test_confounded_comparisons_are_rejected(part, key, new_value):
    reference, local = _runs()
    local[part][key] = new_value
    with pytest.raises(ValueError):
        validate_paired_runs(reference, local)


def test_different_sampled_frame_cannot_be_hidden_under_same_clip():
    reference, local = _runs()
    local["clip_records"][0]["timestamps_s"] = [1]
    with pytest.raises(ValueError, match="frame cohort"):
        validate_paired_runs(reference, local)

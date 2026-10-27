"""Input locks must bind actual annotations to explicit IDs and disjoint splits."""

import copy
import json

import pytest

from temporal_hoi.data import vidhoi_lock


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    mapping = {
        "objects": {str(index): "person" if index == 0 else f"object_{index}"
                    for index in range(78)},
        "predicates": {str(index): f"predicate_{index}" for index in range(50)},
    }
    def annotation(video):
        return {"video_folder": "folder", "video_id": video, "frame_id": 4,
                "person_id": 11, "object_id": 20, "object_class": 1, "action_class": 0,
                "person_box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
                "object_box": {"xmin": 20, "ymin": 0, "xmax": 30, "ymax": 10}}
    def frame(video):
        return {"video_id": f"folder/{video}", "source_frame_id": 4, "frame_id": 106,
                "timestamp_s": 3.5, "result_frame_key": f"folder/{video}_000004",
                "box_transform": {"scale_x": 2, "scale_y": 2, "offset_x": 0, "offset_y": 0},
                "box_transform_provenance": "synthetic fixture known resize, not benchmark media"}
    content = {
        "mapping": mapping,
        "train_annotations": [annotation("train"), annotation("development")],
        "val_annotations": [annotation("final")],
        "train_frame_manifest": [frame("train"), frame("development")],
        "val_frame_manifest": [frame("final")],
        "video_partitions": {"research_train": ["folder/train"],
                             "research_validation": ["folder/development"],
                             "official_val": ["folder/final"]},
        "frame_mask": {"protocol_id": "fixture_v2", "frame_ids": ["folder/final_000106"]},
    }
    spec = {"protocol_id": "fixture_v2", "expected_mask_frame_count": 1, "inputs": {}}
    for name, data in content.items():
        path = tmp_path / f"{name}.json"
        write_json(path, data)
        spec["inputs"][name] = {"path": path.name}
    monkeypatch.setattr(vidhoi_lock, "CANONICAL_MAPPING_SHA256",
                        vidhoi_lock.file_sha256(tmp_path / "mapping.json"))
    return tmp_path, spec, content


def test_missing_files_produce_machine_readable_block_without_fake_metrics(tmp_path):
    output = vidhoi_lock.lock_vidhoi_inputs({"protocol_id": "missing", "inputs": {}},
                                          base_dir=tmp_path)
    assert output["status"] == "blocked_missing_local_inputs"
    assert set(output["missing_inputs"]) == set(vidhoi_lock.REQUIRED_INPUTS)
    assert not output["input_lock_ready"] and output["benchmark_map"] is None
    assert not output["official_final_split_annotations_accessed"]


def test_complete_local_lock_has_source_ids_support_and_no_reproduction_claim(local_inputs):
    directory, spec, _ = local_inputs
    output = vidhoi_lock.lock_vidhoi_inputs(spec, base_dir=directory)
    assert output["input_lock_ready"] and output["missing_inputs"] == []
    train = output["annotation_splits"]["train"]
    assert train["frames"] == 2 and train["gt_pairs"] == 2
    assert train["triplet_support"][0]["gt_positive_pair_frames"] == 2
    assert train["triplet_support"][0]["videos"] == 2
    assert train["ordered_media_frame_ids"] == ["folder/train_000106",
                                                "folder/development_000106"]
    assert not output["benchmark_comparison_eligible"]
    assert not output["model_inference_run"] and not output["zero_shot_split_materialized"]


def test_multilabel_duplicate_rows_count_unique_pairs_once(local_inputs):
    directory, spec, content = local_inputs
    rows = content["train_annotations"]
    second = copy.deepcopy(rows[0])
    second["action_class"] = 1
    rows[1:1] = [copy.deepcopy(rows[0]), second]
    write_json(directory / "train_annotations.json", rows)
    output = vidhoi_lock.lock_vidhoi_inputs(spec, base_dir=directory)
    support = output["annotation_splits"]["train"]["triplet_support"]
    assert [row["gt_positive_pair_frames"] for row in support] == [2, 1]


@pytest.mark.parametrize("change,match", [
    ("leaking_partition", "leaking"),
    ("missing_partition_video", "cover"),
    ("wrong_mask_ID_same_count", "without official_val"),
    ("mask_protocol", "protocol_id"),
    ("mask_count", "expected unique"),
    ("duplicate_manifest", "Duplicate source"),
    ("missing_result_key", "result_frame_key"),
    ("missing_frame", "missing from"),
    ("invalid_transform", "positive"),
    ("missing_transform", "explicit box_transform"),
])
def test_input_invariants_reject_wrong_ids_and_sources(local_inputs, change, match):
    directory, spec, content = local_inputs
    if change == "leaking_partition":
        content["video_partitions"]["research_train"].append("folder/final")
    elif change == "missing_partition_video":
        content["video_partitions"]["research_train"] = ["folder/unannotated"]
    elif change == "wrong_mask_ID_same_count":
        content["frame_mask"]["frame_ids"] = ["folder/final_000004"]
    elif change == "mask_protocol":
        content["frame_mask"]["protocol_id"] = "other"
    elif change == "mask_count":
        spec["expected_mask_frame_count"] = 2
    elif change == "duplicate_manifest":
        content["val_frame_manifest"] *= 2
    elif change == "missing_result_key":
        del content["val_frame_manifest"][0]["result_frame_key"]
    elif change == "missing_frame":
        content["val_frame_manifest"][0]["source_frame_id"] = 5
    elif change == "invalid_transform":
        content["val_frame_manifest"][0]["box_transform"]["scale_x"] = 0
    elif change == "missing_transform":
        del content["val_frame_manifest"][0]["box_transform"]
    for name, data in content.items():
        write_json(directory / f"{name}.json", data)
    with pytest.raises(ValueError, match=match):
        vidhoi_lock.lock_vidhoi_inputs(spec, base_dir=directory)


def test_declared_file_hash_prevents_replacement(local_inputs):
    directory, spec, _ = local_inputs
    spec["inputs"]["val_annotations"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="val_annotations hash mismatch"):
        vidhoi_lock.lock_vidhoi_inputs(spec, base_dir=directory)


@pytest.mark.parametrize("invalid", ["[{\"a\":1}", "[{\"a\":1},]", "[1]", "[]x", "{}"])
def test_incremental_parser_rejects_invalid_and_truncated_arrays(tmp_path, invalid):
    path = tmp_path / "annots.json"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError):
        list(vidhoi_lock.iter_json_array(path, chunk_size=2))


def test_incremental_parser_preserves_unicode_and_escaped_separators(tmp_path):
    rows = [{"caption": "Tiếng Việt, [a] \"b\"", "nested": {"a": [1, 2]}}] * 3
    path = tmp_path / "annots.json"
    write_json(path, rows)
    assert list(vidhoi_lock.iter_json_array(path, chunk_size=3)) == rows

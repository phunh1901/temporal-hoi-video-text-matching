"""Feature caches preserve observed labels, identities, role provenance and holdouts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from temporal_hoi.data.hoi_head_cache import (
    cache_head_features,
    config_fingerprint,
    load_head_cache,
    preflight_head_data,
)
from temporal_hoi.inference.hoi_static import file_sha256


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class FixtureEncoder:
    metadata = {"test_fixture": True, "name": "explicit_pixel_fixture"}

    def __init__(self):
        self.prompts = []
        self.images = []

    def encode_texts(self, texts):
        self.prompts.extend(texts)
        return np.asarray([[1, len(text), 2] for text in texts], dtype=np.float32)

    def encode_images(self, images):
        self.images.extend(np.asarray(image).copy() for image in images)
        return np.asarray([np.asarray(image).mean(axis=(0, 1)) + 1 for image in images],
                          dtype=np.float32)


@pytest.fixture
def inputs(tmp_path):
    mapping = {"objects": {"0": "person", "1": "cup"},
               "predicates": {"0": "hold", "1": "open", "2": "watch"}}
    write(tmp_path / "mapping.json", mapping)
    (tmp_path / "checkpoint.safetensors").write_bytes(b"explicit fixture checkpoint not loaded")
    pixels = np.zeros((8, 12, 3), dtype=np.uint8)
    pixels[:, :3] = [250, 0, 0]
    pixels[:, 3:7] = [0, 200, 0]
    pixels[:, 8:] = [0, 0, 150]
    Image.fromarray(pixels).save(tmp_path / "frame.png")
    dev_pixels = pixels.copy()
    dev_pixels[0, 0, 0] -= 1
    Image.fromarray(dev_pixels).save(tmp_path / "dev_frame.png")
    protocol = {"schema_version": "vidhoi_head_protocol_v1", "label_policy": "positive_only_unknown",
                "train_video_ids": ["train/v"], "dev_video_ids": ["dev/v"],
                "seen_triplets": [[0, 0, 1], [0, 2, 0]],
                "pseudo_unseen_triplets": [[0, 2, 1]], "final_unseen_triplets": [[0, 1, 1]],
                "excluded_final_video_ids": ["final/v"],
                "video_positive_triplets": {"train/v": [[0, 0, 1], [0, 2, 0]],
                                             "dev/v": [[0, 2, 1]]},
                "prior_inspected_video_ids_always_train": ["train/v"]}
    write(tmp_path / "protocol.json", protocol)
    frame = {"video_id": "train/v", "source_frame_id": 1, "frame_id": 0, "timestamp_s": 0,
             "image_path": "frame.png", "image_sha256": file_sha256(tmp_path / "frame.png"),
             "width": 12, "height": 8,
             "entities": [{"track_id": "p1", "class_id": 0, "bbox_xyxy": [0, 0, 3, 8]},
                          {"track_id": "p2", "class_id": 0, "bbox_xyxy": [3, 0, 7, 8]},
                          {"track_id": "cup", "class_id": 1, "bbox_xyxy": [8, 0, 12, 8]}],
             "ground_truth": [{"subject_track_id": "p1", "object_track_id": "cup",
                                "predicate_ids": [0]}]}
    config = {"schema_version": "vidhoi_head_study_config_v1", "batch_size": 2,
              "checkpoint_provenance": {"source": "explicit test fixture"}}
    for role, name, video, predicate in (("train", "train_manifest", "train/v", 0),
                                         ("validation", "dev_manifest", "dev/v", 2)):
        row = copy.deepcopy(frame)
        row["video_id"] = video
        if role == "validation":
            row["image_path"] = "dev_frame.png"
            row["image_sha256"] = file_sha256(tmp_path / "dev_frame.png")
        row["ground_truth"][0]["predicate_ids"] = [predicate]
        if role == "train":
            row["ground_truth"].append({"subject_track_id": "p1", "object_track_id": "p2",
                                        "predicate_ids": [2]})
        empty = {**row, "frame_id": 1, "source_frame_id": 2, "entities": [], "ground_truth": []}
        manifest = {"schema_version": "vidhoi_head_frames_v1", "upstream_split": "train",
                    "role": role, "box_source": "oracle", "object_label_source": "oracle",
                    "protocol_sha256": file_sha256(tmp_path / "protocol.json"),
                    "mapping_sha256": file_sha256(tmp_path / "mapping.json"), "frames": [row, empty]}
        write(tmp_path / f"{name}.json", manifest)
    for name, filename in (("mapping", "mapping.json"), ("checkpoint", "checkpoint.safetensors"),
                            ("protocol", "protocol.json"), ("train_manifest", "train_manifest.json"),
                            ("dev_manifest", "dev_manifest.json")):
        config[name] = str(tmp_path / filename)
        config[f"{name}_sha256"] = file_sha256(tmp_path / filename)
    return config


def mutate_protocol(config, change):
    protocol = json.loads(Path(config["protocol"]).read_text(encoding="utf-8"))
    change(protocol)
    write(config["protocol"], protocol)
    config["protocol_sha256"] = file_sha256(config["protocol"])
    for key in ("train_manifest", "dev_manifest"):
        manifest = json.loads(Path(config[key]).read_text(encoding="utf-8"))
        manifest["protocol_sha256"] = config["protocol_sha256"]
        write(config[key], manifest)
        config[f"{key}_sha256"] = file_sha256(config[key])


def test_cache_keeps_all_proposals_unknown_labels_and_zero_pair_frames(inputs, tmp_path):
    encoder = FixtureEncoder()
    receipt = cache_head_features(inputs, tmp_path / "cache", encoder=encoder)
    assert receipt["model_inference_run"] is False
    assert receipt["roles"] == {role: {"num_frames": 2, "num_pairs": 4}
                                for role in ("train", "validation")}
    # Final composition never reaches the encoder; other object/predicate prompts remain.
    assert not any("opening a cup" in prompt for prompt in encoder.prompts)
    assert len(encoder.prompts) == 5
    with np.load(tmp_path / "cache/text_features.npz", allow_pickle=False) as archive:
        assert archive["text_features"].shape == (2, 3, 3)
        assert not archive["candidate_mask"][1, 1]
        assert np.array_equal(archive["text_features"][1, 1], [0, 0, 0])
    train = load_head_cache(tmp_path / "cache/train")
    assert len(train["frames"]) == 2
    assert train["positive_mask"].sum() == 2
    assert train["positive_mask"].shape == (4, 3)
    assert "negative_mask" not in train
    assert train["subject_track_id"].tolist() == ["p1", "p1", "p2", "p2"]
    assert train["object_track_id"].tolist() == ["cup", "p2", "cup", "p1"]
    assert train["object_features"][0].tolist() == train["object_features"][2].tolist()
    assert np.isfinite(train["union_features"]).all()
    assert train["metadata"]["pseudo_unseen_in_train"] is False
    # Cached training can run after its source images become unavailable.
    Path(train["frames"][0].image_path).unlink()
    assert load_head_cache(tmp_path / "cache/train")["positive_mask"].sum() == 2


@pytest.mark.parametrize("triplet", [[0, 2, 1], [0, 1, 1]])
def test_full_video_holdout_leakage_rejected_even_outside_sampled_frames(inputs, triplet):
    mutate_protocol(inputs, lambda value: value["video_positive_triplets"]["train/v"].append(triplet))
    with pytest.raises(ValueError, match="Held-out composition"):
        preflight_head_data(inputs, test_fixture=True)


def test_video_overlap_and_prior_inspection_rejected(inputs):
    mutate_protocol(inputs, lambda value: value.update(dev_video_ids=["train/v"]))
    with pytest.raises(ValueError, match="overlap"):
        preflight_head_data(inputs, test_fixture=True)


def test_inspected_video_cannot_move_to_dev(inputs):
    mutate_protocol(inputs, lambda value: value.update(prior_inspected_video_ids_always_train=["dev/v"]))
    with pytest.raises(ValueError, match="Previously inspected"):
        preflight_head_data(inputs, test_fixture=True)


def test_fixture_not_accepted_as_real_mapping(inputs):
    with pytest.raises(ValueError, match="pinned"):
        preflight_head_data(inputs)


def test_duplicate_image_bytes_across_roles_rejected(inputs):
    path = Path(inputs["dev_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for frame in manifest["frames"]:
        frame["image_path"] = "frame.png"
        frame["image_sha256"] = file_sha256(path.parent / "frame.png")
    write(path, manifest)
    inputs["dev_manifest_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="Identical image bytes"):
        preflight_head_data(inputs, test_fixture=True)


def test_pseudo_components_must_appear_in_actual_sampled_train_labels(inputs):
    path = Path(inputs["train_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["frames"][0]["ground_truth"] = manifest["frames"][0]["ground_truth"][:1]
    write(path, manifest)
    inputs["train_manifest_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="unseen sampled-train component"):
        preflight_head_data(inputs, test_fixture=True)


def test_sampled_coverage_is_explicit_and_final_composition_components_optional(inputs):
    audit = preflight_head_data(inputs, test_fixture=True)["audit"]
    assert audit["sampled_train_predicate_ids"] == [0, 2]
    assert audit["sampled_dev_designed_pseudo_triplets"] == [[0, 2, 1]]
    assert audit["sampled_dev_incidental_unseen_triplets"] == []
    mutate_protocol(inputs, lambda value: value.update(final_heldout_type="composition"))
    with pytest.raises(ValueError, match="Designed final-unseen"):
        preflight_head_data(inputs, test_fixture=True)


def test_cache_modified_bytes_rejected(inputs, tmp_path):
    cache_head_features(inputs, tmp_path / "cache", encoder=FixtureEncoder())
    path = tmp_path / "cache/train/arrays.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_head_cache(tmp_path / "cache/train")


@pytest.mark.parametrize("field", ["positive_mask", "object_track_id"])
def test_semantic_pair_or_label_corruption_rejected_even_with_updated_file_hash(inputs, tmp_path, field):
    cache_head_features(inputs, tmp_path / "cache", encoder=FixtureEncoder())
    directory = tmp_path / "cache/train"
    path = directory / "arrays.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays[field][0] = False if field == "positive_mask" else "p2"
    np.savez_compressed(path, **arrays)
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for row in receipt["files"]:
        if row["path"] == "arrays.npz":
            row.update(sha256=file_sha256(path), bytes=path.stat().st_size)
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="ordered proposals"):
        load_head_cache(directory)


def test_config_fingerprint_is_order_independent_and_changes_with_runtime_settings():
    first = {"device": "cpu", "study": {"seed": 42, "epochs": 20}}
    reordered = {"study": {"epochs": 20, "seed": 42}, "device": "cpu"}
    assert config_fingerprint(first) == config_fingerprint(reordered)
    assert config_fingerprint(first) != config_fingerprint({**first, "device": "cuda"})


def test_sampled_gt_absent_from_whole_video_metadata_rejected(inputs):
    mutate_protocol(inputs, lambda value: value["video_positive_triplets"].update({"dev/v": []}))
    with pytest.raises(ValueError, match="Sampled GT"):
        preflight_head_data(inputs, test_fixture=True)


def test_frame_join_and_missing_images_fail_without_silent_substitution(inputs):
    path = Path(inputs["train_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["frames"][0]["source_frame_id"] = 99
    write(path, manifest)
    inputs["train_manifest_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="Frame join"):
        preflight_head_data(inputs, test_fixture=True)

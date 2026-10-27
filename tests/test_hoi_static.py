"""All-pair oracle inference must not prune candidates using GT interactions."""

import copy
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from PIL import Image

from temporal_hoi.data.vidhoi import ClassMapping
from temporal_hoi.evaluation.hoi import evaluate_frame_hoi
from temporal_hoi.inference.hoi_static import (
    CONFIG_SCHEMA,
    MANIFEST_SCHEMA,
    PREDICATE_PHRASES,
    build_prompts,
    file_sha256,
    infer_frame,
    load_manifest,
    preflight,
    proposal_pairs,
    resolve_config,
    run_static_probe,
    safe_checkpoint_state,
)


class ExplicitTestEncoder:
    metadata = {"test_fixture": True, "name": "controlled_two_dimensional_encoder"}

    def encode_images(self, images):
        # A red frame and a blue frame carry different, independently known features.
        return np.asarray([[np.asarray(image)[..., 0].mean(),
                            np.asarray(image)[..., 2].mean()] for image in images], dtype=np.float32)

    def encode_texts(self, texts):
        return np.asarray([[1.0, 0.0] if "holding" in text else [0.0, 1.0]
                           for text in texts], dtype=np.float32)


@pytest.fixture
def inputs(tmp_path):
    mapping = {"objects": {"0": "person", "1": "cup"},
               "predicates": {"0": "hold", "1": "open"}}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (12, 8), (255, 0, 0)).save(image_path)
    row = {"video_id": "folder/video", "source_frame_id": 1, "frame_id": 24,
           "timestamp_s": 1.0, "image_path": "frame.png", "image_sha256": file_sha256(image_path),
           "width": 12, "height": 8,
           "entities": [{"track_id": "p1", "class_id": 0, "bbox_xyxy": [0, 0, 5, 8]},
                        {"track_id": "c1", "class_id": 1, "bbox_xyxy": [6, 0, 12, 8]}],
           "ground_truth": [{"subject_track_id": "p1", "object_track_id": "c1",
                             "predicate_ids": [0]}]}
    manifest = {"schema_version": MANIFEST_SCHEMA, "split": "train", "purpose": "train_probe",
                "box_source": "oracle", "object_label_source": "oracle",
                "mapping_sha256": file_sha256(mapping_path), "frames": [row]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = tmp_path / "explicit_fixture.safetensors"
    checkpoint.write_bytes(b"explicit dependency-injected fixture; never loaded as a model")
    config = {"schema_version": CONFIG_SCHEMA, "manifest": str(manifest_path),
              "manifest_sha256": file_sha256(manifest_path), "mapping": str(mapping_path),
              "mapping_sha256": file_sha256(mapping_path), "checkpoint": str(checkpoint),
              "checkpoint_sha256": file_sha256(checkpoint), "batch_size": 2,
              "max_frames": 4, "max_pairs_per_frame": 10, "preview_frames": 1,
              "checkpoint_provenance": {"source": "explicit test fixture"}}
    return tmp_path, config, manifest


def _save_manifest(inputs):
    _, config, manifest = inputs
    path = Path(config["manifest"])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = file_sha256(path)


def _load(inputs):
    _, config, _ = inputs
    return preflight(config, test_fixture=True)


def test_hand_calculated_scoring_and_shared_evaluator_parity(inputs):
    root, config, _ = inputs
    mapping, frames, _ = _load(inputs)
    encoder = ExplicitTestEncoder()
    text = {1: encoder.encode_texts(build_prompts(mapping, 1))}
    inferred = infer_frame(frames[0], encoder, text, batch_size=1, logit_scale=1.0)
    assert inferred.predictions[0].predicate_values == pytest.approx((math.e, 1.0))
    assert evaluate_frame_hoi([inferred])["map"] == pytest.approx(1.0)
    receipt = run_static_probe(config, root / "result", encoder=encoder)
    stored = json.loads((root / "result" / "predictions.json").read_text(encoding="utf-8"))
    assert stored[0]["predictions"][0]["predicate_values"] == pytest.approx((math.e, 1.0))
    assert receipt["metrics"] == evaluate_frame_hoi([inferred])
    assert receipt["level"] == "synthetic_encoder_fixture"
    assert not receipt["model_inference_run"] and not receipt["model_training_run"]
    assert not receipt["score_is_probability"] and not receipt["benchmark_comparison_eligible"]
    assert (root / "result" / "preview_0000.jpg").is_file()


def test_all_person_to_distinct_entities_include_pairs_absent_from_positive_gt(inputs):
    _, _, manifest = inputs
    manifest["frames"][0]["entities"].append(
        {"track_id": "p2", "class_id": 0, "bbox_xyxy": [5, 0, 10, 8]})
    _save_manifest(inputs)
    mapping, frames, _ = _load(inputs)
    pairs = proposal_pairs(frames[0])
    assert {(sub.track_id, obj.track_id) for sub, obj in pairs} == {
        ("p1", "p2"), ("p1", "c1"), ("p2", "p1"), ("p2", "c1")}
    encoder = ExplicitTestEncoder()
    text = {class_id: encoder.encode_texts(build_prompts(mapping, class_id))
            for class_id in (0, 1)}
    inferred = infer_frame(frames[0], encoder, text, batch_size=2, logit_scale=1)
    assert len(inferred.predictions) == 4
    assert all(len(pair.predicate_values) == 2 for pair in inferred.predictions)


def test_changing_gt_action_does_not_change_proposals_prompts_or_predictions(inputs):
    root, config, manifest = inputs
    first = run_static_probe(config, root / "first", encoder=ExplicitTestEncoder())
    manifest["frames"][0]["ground_truth"][0]["predicate_ids"] = [1]
    _save_manifest(inputs)
    second = run_static_probe(config, root / "second", encoder=ExplicitTestEncoder())
    first_rows = json.loads((root / "first" / "predictions.json").read_text())
    second_rows = json.loads((root / "second" / "predictions.json").read_text())
    assert first_rows[0]["predictions"] == second_rows[0]["predictions"]
    assert first["prompt_sha256"] == second["prompt_sha256"]
    assert first["num_proposal_pairs"] == second["num_proposal_pairs"]


def test_full_fifty_predicate_vocabulary_is_always_scored_in_mapping_order():
    names = list(PREDICATE_PHRASES)
    assert len(names) == 50
    mapping = ClassMapping({0: "person", 1: "cup"}, dict(enumerate(names)))
    prompts = build_prompts(mapping, 1)
    assert len(prompts) == 50
    assert prompts[names.index("hold")] == "A photo of a person holding a cup."
    assert prompts[names.index("above")] == "A photo of a person above a cup."


@pytest.mark.parametrize("field", ["manifest", "mapping", "checkpoint"])
def test_wrong_input_hash_fails_before_output_is_created(inputs, field):
    root, config, _ = inputs
    config[f"{field}_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_static_probe(config, root / "no_output", encoder=ExplicitTestEncoder())
    assert not (root / "no_output").exists()


def test_missing_image_and_dimension_mismatch_are_not_silently_dropped(inputs):
    root, _, manifest = inputs
    frame_path = root / "frame.png"
    frame_path.rename(root / "preserved.png")
    with pytest.raises(ValueError, match="Missing frame image"):
        _load(inputs)
    (root / "preserved.png").rename(frame_path)
    manifest["frames"][0]["width"] = 13
    _save_manifest(inputs)
    with pytest.raises(ValueError, match="dimensions"):
        _load(inputs)


@pytest.mark.parametrize("change, message", [
    ({"class_id": 2}, "Unknown entity class"),
    ({"class_id": True}, "integer"),
    ({"bbox_xyxy": [0, 0, 0, 3]}, "positive area"),
    ({"bbox_xyxy": [0, 0, 13, 8]}, "inside original image"),
    ({"bbox_xyxy": [0, 0, float("nan"), 8]}, "finite"),
])
def test_bad_oracle_entities_are_rejected(inputs, change, message):
    inputs[2]["frames"][0]["entities"][1].update(change)
    _save_manifest(inputs)
    with pytest.raises(ValueError, match=message):
        _load(inputs)


def test_unknown_action_or_absent_entity_is_rejected(inputs):
    relation = inputs[2]["frames"][0]["ground_truth"][0]
    relation["predicate_ids"] = [2]
    _save_manifest(inputs)
    with pytest.raises(ValueError, match="Unknown predicate"):
        _load(inputs)
    relation["predicate_ids"] = [0]
    relation["object_track_id"] = "missing"
    _save_manifest(inputs)
    with pytest.raises(ValueError, match="absent"):
        _load(inputs)


@pytest.mark.parametrize("field,value", [("split", "official_val"),
                                         ("box_source", "predicted"),
                                         ("object_label_source", "detector")])
def test_train_oracle_scope_cannot_be_relabeled(inputs, field, value):
    inputs[2][field] = value
    _save_manifest(inputs)
    with pytest.raises(ValueError, match=field):
        _load(inputs)


def test_frame_limit_and_pair_limit_fail_instead_of_truncating(inputs):
    _, config, manifest = inputs
    second = copy.deepcopy(manifest["frames"][0])
    second.update(frame_id=25, source_frame_id=2)
    manifest["frames"].append(second)
    _save_manifest(inputs)
    config["max_frames"] = 1
    with pytest.raises(ValueError, match="no frames were silently dropped"):
        _load(inputs)
    config["max_frames"] = 4
    manifest["frames"][0]["entities"].append(
        {"track_id": "p2", "class_id": 0, "bbox_xyxy": [5, 0, 10, 8]})
    config["max_pairs_per_frame"] = 1
    _save_manifest(inputs)
    with pytest.raises(ValueError, match="no pairs were silently dropped"):
        _load(inputs)


def test_zero_proposal_frame_is_kept_and_declared(inputs):
    root, config, manifest = inputs
    second = copy.deepcopy(manifest["frames"][0])
    second.update(frame_id=25, source_frame_id=2, entities=[], ground_truth=[])
    manifest["frames"].append(second)
    _save_manifest(inputs)
    receipt = run_static_probe(config, root / "result", encoder=ExplicitTestEncoder())
    assert receipt["num_frames"] == receipt["metrics"]["num_frames"] == 2
    assert receipt["frames_without_proposals"] == 1
    assert receipt["dropped_frames"] == receipt["dropped_proposal_pairs"] == 0


def test_image_path_cannot_escape_manifest_directory(inputs):
    inputs[2]["frames"][0]["image_path"] = "../outside.png"
    _save_manifest(inputs)
    with pytest.raises(ValueError, match="escapes"):
        _load(inputs)


def test_real_preflight_requires_canonical_mapping_and_no_fixture_default(inputs):
    _, config, _ = inputs
    with pytest.raises(ValueError, match="pinned canonical"):
        preflight(config)
    encoder = ExplicitTestEncoder()
    encoder.metadata = {"test_fixture": False}
    with pytest.raises(ValueError, match="explicitly marked"):
        run_static_probe(config, inputs[0] / "result", encoder=encoder)


def test_zero_encoder_features_fail_instead_of_producing_metrics(inputs):
    encoder = ExplicitTestEncoder()
    encoder.encode_images = lambda images: np.zeros((len(images), 2))
    with pytest.raises(ValueError, match="zero features"):
        run_static_probe(inputs[1], inputs[0] / "result", encoder=encoder)
    assert not (inputs[0] / "result" / "metrics.json").exists()


def test_torchscript_archive_rejected_before_torch_loader(tmp_path, monkeypatch):
    archive = tmp_path / "script.pt"
    with ZipFile(archive, "w") as stream:
        stream.writestr("archive/constants.pkl", b"not executed")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("loader must not run"))
    with pytest.raises(ValueError, match="TorchScript"):
        safe_checkpoint_state(archive)


def test_plain_state_loader_uses_weights_only_and_rejects_non_tensor_payload(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fixture for mocked safe loader")
    calls = []

    def safe_load(path, **kwargs):
        calls.append(kwargs)
        return {"not_a_weight": "untrusted text"}

    monkeypatch.setattr(torch, "load", safe_load)
    with pytest.raises(ValueError, match="only named tensors"):
        safe_checkpoint_state(checkpoint)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_resolve_config_uses_config_directory_without_changing_cwd(inputs):
    root, config, _ = inputs
    relative = dict(config)
    for name in ("manifest", "mapping", "checkpoint"):
        relative[name] = Path(relative[name]).name
    resolved = resolve_config(relative, root / "config.json")
    assert all(Path(resolved[name]).is_file() for name in ("manifest", "mapping", "checkpoint"))


def test_manifest_direct_loader_preserves_source_and_media_frame_ids(inputs):
    _, config, _ = inputs
    mapping = ClassMapping.from_json(config["mapping"])
    frames, _ = load_manifest(Path(config["manifest"]), mapping,
                              max_frames=4, max_pairs_per_frame=10)
    assert frames[0].source_frame_id == 1
    assert frames[0].frame_id == 24
    assert frames[0].timestamp_s == 1.0

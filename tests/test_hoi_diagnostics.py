"""Pixel controls and known-positive ranks must preserve the oracle cohort."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from temporal_hoi.evaluation.hoi import evaluate_frame_hoi
from temporal_hoi.inference.hoi_diagnostics import (
    VARIANTS,
    compare_diagnostic_variants,
    infer_representation_frame,
    positive_predicate_ranks,
    role_symmetry_diagnostics,
    run_representation_diagnostics,
)
from temporal_hoi.inference.hoi_static import (
    CONFIG_SCHEMA,
    MANIFEST_SCHEMA,
    PREDICATE_PHRASES,
    _normalized,
    build_prompts,
    file_sha256,
    infer_frame,
    preflight,
)


class PixelTestEncoder:
    metadata = {"test_fixture": True, "name": "explicit_RGB_mean_fixture"}

    def __init__(self):
        self.images = []
        self.text_calls = []

    def encode_images(self, images):
        pixels = [np.asarray(image).copy() for image in images]
        self.images.extend(pixels)
        return np.asarray([image.mean(axis=(0, 1)) for image in pixels], dtype=np.float32)

    def encode_texts(self, texts):
        self.text_calls.append(list(texts))
        return np.asarray([[1, 0, 0] if "holding a " in text else [0, 0, 1]
                           if "opening a " in text else [0, 1, 0] for text in texts],
                          dtype=np.float32)


@pytest.fixture
def diagnostic_inputs(tmp_path):
    names = list(PREDICATE_PHRASES)
    mapping = {"objects": {"0": "person", "1": "cup"},
               "predicates": dict(enumerate(names))}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    pixels = np.zeros((8, 12, 3), dtype=np.uint8)
    pixels[..., 1] = 100
    pixels[:, :3] = [255, 0, 0]
    pixels[:, 8:10] = [0, 0, 255]
    image_path = tmp_path / "frame.png"
    Image.fromarray(pixels).save(image_path)
    row = {"video_id": "folder/video", "source_frame_id": 1, "frame_id": 24,
           "timestamp_s": 1.0, "image_path": "frame.png", "image_sha256": file_sha256(image_path),
           "width": 12, "height": 8,
           "entities": [{"track_id": "p1", "class_id": 0, "bbox_xyxy": [0, 0, 3, 8]},
                        {"track_id": "c1", "class_id": 1, "bbox_xyxy": [8, 0, 10, 8]}],
           "ground_truth": [{"subject_track_id": "p1", "object_track_id": "c1",
                             "predicate_ids": [names.index("hold"), names.index("open")]}]}
    manifest = {"schema_version": MANIFEST_SCHEMA, "split": "train", "purpose": "train_probe",
                "box_source": "oracle", "object_label_source": "oracle",
                "mapping_sha256": file_sha256(mapping_path), "frames": [row]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = tmp_path / "explicit_fixture.safetensors"
    checkpoint.write_bytes(b"explicit injected fixture; never loaded as a real model")
    config = {"schema_version": CONFIG_SCHEMA, "manifest": str(manifest_path),
              "manifest_sha256": file_sha256(manifest_path), "mapping": str(mapping_path),
              "mapping_sha256": file_sha256(mapping_path), "checkpoint": str(checkpoint),
              "checkpoint_sha256": file_sha256(checkpoint), "batch_size": 2,
              "max_frames": 4, "max_pairs_per_frame": 10, "preview_frames": 0,
              "checkpoint_provenance": {"source": "explicit test fixture"}}
    return tmp_path, config, manifest


def _save(inputs):
    _, config, manifest = inputs
    path = Path(config["manifest"])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = file_sha256(path)


def _frame_and_text(inputs):
    mapping, frames, _ = preflight(inputs[1], test_fixture=True)
    encoder = PixelTestEncoder()
    text = {class_id: _normalized(encoder.encode_texts(build_prompts(mapping, class_id)),
                                  len(mapping.predicates), "text")
            for class_id in mapping.objects}
    return mapping, frames[0], text


def test_three_modes_use_distinct_actual_pixels_and_preserve_union_baseline(diagnostic_inputs):
    _, frame, text = _frame_and_text(diagnostic_inputs)
    encoder = PixelTestEncoder()
    predictions = {}
    for mode in VARIANTS:
        predictions[mode], stats = infer_representation_frame(
            frame, encoder, text, variant=mode, batch_size=2, logit_scale=1)
        assert stats["image_encoder_input_count"] == 1
        assert len(predictions[mode].predictions[0].predicate_values) == 50
    assert [image.shape for image in encoder.images] == [(8, 12, 3), (8, 2, 3), (8, 10, 3)]
    assert np.all(encoder.images[1] == [0, 0, 255])
    assert not np.array_equal(encoder.images[0].mean((0, 1)), encoder.images[2].mean((0, 1)))
    assert predictions["object_only"].predictions != predictions["union"].predictions
    baseline = infer_frame(frame, PixelTestEncoder(), text, batch_size=2, logit_scale=1)
    assert asdict(predictions["union"]) == asdict(baseline)
    assert evaluate_frame_hoi([predictions["union"]]) == evaluate_frame_hoi([baseline])


def test_fullframe_and_object_cache_identity_and_reverse_role_limits(diagnostic_inputs):
    diagnostic_inputs[2]["frames"][0]["entities"].append(
        {"track_id": "p2", "class_id": 0, "bbox_xyxy": [4, 0, 7, 8]})
    _save(diagnostic_inputs)
    _, frame, text = _frame_and_text(diagnostic_inputs)
    observed = {}
    for mode, image_count in (("full_frame", 1), ("object_only", 3), ("union", 4)):
        inferred, stats = infer_representation_frame(
            frame, PixelTestEncoder(), text, variant=mode, batch_size=2, logit_scale=1)
        assert len(inferred.predictions) == 4
        assert stats["image_encoder_input_count"] == image_count
        observed[mode] = role_symmetry_diagnostics([inferred], [stats], variant=mode)
    for mode in ("full_frame", "union"):
        reverse = observed[mode]["reverse_pairs"]
        assert len(reverse) == 1
        assert reverse[0]["pixels_exactly_equal"]
        assert reverse[0]["score_vectors_exactly_equal"]
        assert reverse[0]["predicate_order_equal"]
    assert not observed["object_only"]["reverse_pairs"][0]["pixels_exactly_equal"]
    for mode in ("full_frame", "object_only"):
        group = observed[mode]["same_object_different_subject_groups"][0]
        assert group["object_track_id"] == "c1"
        assert group["comparisons_to_first"][0]["score_vectors_exactly_equal"]
    assert all(item["not_actor_correctness"] for item in observed.values())
    assert all("actor_accuracy" not in item for item in observed.values())


def test_known_positive_multilabel_ties_are_not_accuracy(diagnostic_inputs):
    mapping, frame, text = _frame_and_text(diagnostic_inputs)
    inferred, _ = infer_representation_frame(
        frame, PixelTestEncoder(), text, variant="union", batch_size=2, logit_scale=1)
    equal = replace(inferred, predictions=tuple(
        replace(pair, predicate_values=(1.0,) * 50) for pair in inferred.predictions))
    ranks = positive_predicate_ranks([equal], mapping)
    assert ranks["num_known_positive_labels"] == 2
    assert ranks["num_annotated_pairs"] == 1
    assert ranks["not_accuracy"] and ranks["unknown_candidates_are_not_verified_negative"]
    assert all(row["rank_best"] == 1 and row["rank_worst"] == 50 for row in ranks["records"])
    assert [row["rank"] for row in ranks["records"]] == [10, 30]
    assert ranks["annotated_pairs_with_any_positive_rank_le"] == {"1": 0, "5": 0, "10": 1}


def test_one_text_cache_output_hashes_and_no_model_claim_for_fixture(diagnostic_inputs):
    root, config, _ = diagnostic_inputs
    encoder = PixelTestEncoder()
    receipt = run_representation_diagnostics(config, root / "result", encoder=encoder)
    assert len(encoder.text_calls) == 25  # 50 predicates once, not three times.
    assert receipt["shared_text_encoder_input_count"] == 50
    assert receipt["shared_model_load_count"] == 0
    assert receipt["variants"] == list(VARIANTS)
    assert receipt["level"] == "representation_diagnostic_encoder_fixture"
    assert not receipt["model_inference_run"] and not receipt["model_training_run"]
    assert not receipt["full_benchmark_reproduction_claim"]
    assert not receipt["equal_image_encode_compute"]
    assert receipt["num_frames"] == receipt["num_proposal_pairs"] == 1
    actual = {path.relative_to(root / "result").as_posix()
              for path in (root / "result").rglob("*") if path.is_file()}
    declared = {record["path"] for record in receipt["files"]}
    assert actual == declared | {"receipt.json"}
    for record in receipt["files"]:
        path = root / "result" / record["path"]
        assert file_sha256(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]
    effective = json.loads((root / "result" / "run_config.json").read_text(encoding="utf-8"))
    assert effective["diagnostic_variants"] == list(VARIANTS)
    for mode in VARIANTS:
        variant = json.loads((root / "result" / mode / "receipt.json").read_text(encoding="utf-8"))
        assert variant["variant"] == mode and variant["num_proposal_pairs"] == 1
        assert not variant["model_inference_run"]


def test_changed_gt_does_not_change_prompts_pixels_or_predictions(diagnostic_inputs):
    root, config, manifest = diagnostic_inputs
    first = run_representation_diagnostics(config, root / "first", encoder=PixelTestEncoder())
    manifest["frames"][0]["ground_truth"][0]["predicate_ids"] = [0]
    _save(diagnostic_inputs)
    second = run_representation_diagnostics(config, root / "second", encoder=PixelTestEncoder())
    assert first["prompt_sha256"] == second["prompt_sha256"]
    for mode in VARIANTS:
        paths = [root / name / mode / "predictions.json" for name in ("first", "second")]
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        assert rows[0][0]["predictions"] == rows[1][0]["predictions"]
        for field in ("pair_input_indices", "inputs"):
            assert (first["variant_summaries"][mode]["frame_statistics"][0][field]
                    == second["variant_summaries"][mode]["frame_statistics"][0][field])


def test_zero_pair_frame_is_preserved_in_every_mode(diagnostic_inputs):
    root, config, manifest = diagnostic_inputs
    second = copy.deepcopy(manifest["frames"][0])
    second.update(frame_id=25, source_frame_id=2, entities=[], ground_truth=[])
    manifest["frames"].append(second)
    _save(diagnostic_inputs)
    receipt = run_representation_diagnostics(config, root / "result", encoder=PixelTestEncoder())
    assert receipt["num_frames"] == 2 and receipt["num_proposal_pairs"] == 1
    for summary in receipt["variant_summaries"].values():
        assert summary["num_frames"] == 2 and summary["frames_without_proposals"] == 1
        assert summary["dropped_frames"] == summary["dropped_proposal_pairs"] == 0


def test_comparison_refuses_changed_proposal_order(diagnostic_inputs):
    mapping, frame, text = _frame_and_text(diagnostic_inputs)
    inferred, _ = infer_representation_frame(
        frame, PixelTestEncoder(), text, variant="union", batch_size=2, logit_scale=1)
    result = {"frames": [inferred], "metrics": evaluate_frame_hoi([inferred]),
              "positive_ranks": positive_predicate_ranks([inferred], mapping)}
    results = {mode: copy.deepcopy(result) for mode in VARIANTS}
    results["object_only"]["frames"][0] = replace(inferred, predictions=())
    with pytest.raises(ValueError, match="proposal pairs"):
        compare_diagnostic_variants(results)


def test_no_partial_variant_selection_or_implicit_test_encoder(diagnostic_inputs):
    root, config, _ = diagnostic_inputs
    config["diagnostic_variants"] = ["union"]
    with pytest.raises(ValueError, match="all three fixed"):
        run_representation_diagnostics(config, root / "result", encoder=PixelTestEncoder())
    config.pop("diagnostic_variants")
    encoder = PixelTestEncoder()
    encoder.metadata = {"test_fixture": False}
    with pytest.raises(ValueError, match="explicitly marked"):
        run_representation_diagnostics(config, root / "result", encoder=encoder)


def test_changed_image_hash_prevents_all_output(diagnostic_inputs):
    root, config, _ = diagnostic_inputs
    (root / "frame.png").write_bytes(b"changed image")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_representation_diagnostics(config, root / "result", encoder=PixelTestEncoder())
    assert not (root / "result").exists()

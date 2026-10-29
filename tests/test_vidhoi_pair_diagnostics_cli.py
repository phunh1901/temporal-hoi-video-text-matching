"""Review overrides stay bound to their case set; unknowns never become negatives."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from temporal_hoi.data.hoi_pair_review import new_review_annotations
from temporal_hoi.inference.hoi_static import file_sha256


def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/run_vidhoi_pair_diagnostics.py"
    spec = importlib.util.spec_from_file_location("_pair_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    manifest = {"schema_version": "vidhoi_pair_review_v1", "source_split": "train", "role": "train",
        "protocol_sha256": "a" * 64, "mapping_sha256": "b" * 64,
        "selection": {"role": "train", "model_scores_used": False,
            "source_train_video_ids": ["train/v"], "excluded_dev_video_ids": ["dev/v"],
            "excluded_final_video_ids": ["final/v"], "selected_case_ids": ["c"]},
        "files": {"frame.png": {"bytes": 1, "sha256": "c" * 64}},
        "cases": [{"case_id": "c", "video_id": "train/v", "frame_id": 0,
            "source_frame_id": 1, "timestamp_s": 0.0, "width": 16, "height": 12,
            "object_track_id": "o", "object_class": 1, "object_bbox_xyxy": [12, 0, 16, 12],
            "predicate_id": 0, "predicate_name": "hold", "actors": [
                {"track_id": track, "bbox_xyxy": [i * 4, 0, i * 4 + 4, 12],
                 "source_annotation_positive": i == 0, "label": "unknown"}
                for i, track in enumerate(["p1", "p2", "p3"])],
            "image_path": "frame.png", "image_sha256": "c" * 64,
            "context_frames": [{"path": "frame.png", "sha256": "c" * 64,
                "frame_id": 0, "timestamp_s": 0.0, "width": 16, "height": 12}], "video": None}]}
    scores = {"states": [{"mode": "union", "seed": 42, "state": "initial", "cases": [
        {"case_id": "c", "actors": [{"track_id": "p1", "score": 1.2},
            {"track_id": "p2", "score": 1.0}, {"track_id": "p3", "score": 2.0}]}]}]}
    return manifest, new_review_annotations(manifest), scores


def asserted_labels(labels):
    value = copy.deepcopy(labels)
    value["reviewer_id"] = "explicit_synthetic_fixture"
    value["reviewed_at_utc"] = "2026-10-30T01:00:00Z"
    value["assertions"] = {"human_reviewed": True, "not_ai_generated_ground_truth": True,
                           "model_scores_hidden": True}
    value["cases"][0]["actors"][0]["label"] = "positive"
    actor = value["cases"][0]["actors"][1]
    actor.update(label="confirmed_negative", evidence_context_indices=[0],
                 rationale="Explicit fixture assertion, not a real reviewed person")
    return value


def test_unknown_only_review_returns_pending_and_null_metrics(inputs):
    manifest, labels, scores = inputs
    result = cli().evaluate_actor_contrasts(labels, manifest, scores)
    assert result["status"] == "pending_verified_contrasts"
    assert result["records"] == [] and not result["unknown_labels_used_as_negative"]
    assert all(v["strict_win_fraction"] is None for v in result["aggregate"].values())


def test_review_metrics_ignore_high_scoring_unknown_actor(inputs):
    manifest, labels, scores = inputs
    result = cli().evaluate_actor_contrasts(asserted_labels(labels), manifest, scores)
    assert len(result["records"]) == 1
    assert result["records"][0]["negative_track_id"] == "p2"
    assert result["records"][0]["score_margin"] == pytest.approx(.2)
    assert result["aggregate"]["union/initial"]["strict_win_fraction"] == 1


def test_unasserted_negative_cannot_produce_actor_metrics(inputs):
    manifest, labels, scores = inputs
    labels["cases"][0]["actors"][1]["label"] = "confirmed_negative"
    with pytest.raises(ValueError):
        cli().evaluate_actor_contrasts(labels, manifest, scores)


def test_missing_annotation_file_keeps_packaged_unknowns(inputs, tmp_path):
    manifest, labels, _ = inputs
    result, source = cli().choose_annotation_input(labels, manifest, input_root=tmp_path)
    assert result == labels and source["external"] is False


def test_discovery_rejects_multiple_label_versions(inputs, tmp_path):
    manifest, labels, _ = inputs
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / cli().LABEL_FILENAME).write_text(json.dumps(labels), encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple"):
        cli().choose_annotation_input(labels, manifest, input_root=tmp_path)


def test_explicit_labels_need_hash_and_matching_case_fingerprint(inputs, tmp_path):
    manifest, labels, _ = inputs
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(asserted_labels(labels)), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        cli().choose_annotation_input(labels, manifest, explicit=path, expected_sha="0" * 64)
    actual, receipt = cli().choose_annotation_input(labels, manifest, explicit=path,
                                                   expected_sha=file_sha256(path))
    assert receipt["external"] and actual["reviewer_id"] == "explicit_synthetic_fixture"
    wrong = copy.deepcopy(actual)
    wrong["manifest_fingerprint"] = "d" * 64
    path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="different review manifest"):
        cli().choose_annotation_input(labels, manifest, explicit=path, expected_sha=file_sha256(path))


def test_inventory_rejects_extra_or_tampered_review_outputs(tmp_path):
    module = cli()
    path = tmp_path / "labels.json"
    path.write_text("{}", encoding="utf-8")
    receipt = {"files": [module.record(path, tmp_path)]}
    (tmp_path / "receipt.json").write_text("{}", encoding="utf-8")
    module.verify_inventory(tmp_path, receipt)
    extra = tmp_path / "unexpected.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        module.verify_inventory(tmp_path, receipt)
    extra.unlink()
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        module.verify_inventory(tmp_path, receipt)

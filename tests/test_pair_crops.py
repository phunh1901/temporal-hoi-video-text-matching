"""Original-coordinate union crops, provenance and detector export contracts."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from temporal_hoi.data.pair_crops import VerifiedUnionBoxes, select_pair, union_bounds


def _pair():
    return {"width": 1000, "height": 500, "person_class": "person", "object_class": "bicycle",
            "person_box_xyxy": [100.1, 50.1, 300, 400],
            "object_box_xyxy": [250, 200, 800.2, 450.2],
            "pair_score": .8, "person_track_id": "p1", "object_track_id": "o1"}


def test_union_inverse_resize_preserves_both_boxes_and_rounds_outward():
    # Original image 1000x500 was resized to 320x160 before encoder transforms.
    assert union_bounds(_pair(), (1000, 500), (320, 160)) == (32, 16, 257, 145)


def test_out_of_frame_detector_coordinates_are_clipped_without_inventing_area():
    pair = _pair()
    pair["person_box_xyxy"] = [-25, -10, 300, 400]
    assert union_bounds(pair, (1000, 500), (320, 160)) == (0, 0, 257, 145)
    pair["person_box_xyxy"] = [-30, 0, -1, 100]
    with pytest.raises(ValueError, match="inside"):
        union_bounds(pair, (1000, 500), (320, 160))


@pytest.mark.parametrize("box", [[100, 100, 100, 110], [0, 0, float("nan"), 10],
                                  [True, 0, 10, 10], [0, 0, 10]])
def test_invalid_boxes_cannot_be_repaired_by_fallback(box):
    pair = _pair()
    pair["person_box_xyxy"] = box
    with pytest.raises(ValueError):
        union_bounds(pair, (1000, 500), (320, 160))


def test_pair_selection_is_order_invariant_and_uses_no_caption():
    low, high, tie = _pair(), _pair(), _pair()
    low["pair_score"] = .5
    high["person_track_id"] = "p2"
    tie["person_track_id"] = "p0"
    assert select_pair([low, high, tie], 1000, 500)["person_track_id"] == "p0"
    assert select_pair([tie, high, low], 1000, 500)["person_track_id"] == "p0"
    invalid = {**high, "person_box_xyxy": [0, 0, 0, 0]}
    assert select_pair([invalid], 1000, 500) is None


def _boxes(tmp_path):
    artifact = tmp_path / "upstream_annotations.json"
    artifact.write_text("verified source annotations", encoding="utf-8")
    evidence = tmp_path / "review.json"
    evidence.write_text("review fixture", encoding="utf-8")
    record = {**_pair(), "clip_id": "v", "frame_index": 24, "timestamp_s": 1.0,
              "source_video_sha256": "video_hash"}
    payload = {"schema_version": "person_object_union/v1",
               "coordinates": "original_frame_pixels_xyxy",
               "provenance": {"source_type": "oracle", "artifact_path": str(artifact),
                              "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                              "verification": {"status": "verified", "verifier": "test fixture",
                                               "evidence_path": str(evidence),
                                               "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}},
               "records": [record]}
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload, artifact


def test_missing_boxes_and_wrong_video_hash_never_fallback(tmp_path):
    path, _, _ = _boxes(tmp_path)
    boxes = VerifiedUnionBoxes(path)
    image = Image.new("RGB", (320, 160))
    metadata = {"fps": 24, "width": 1000, "height": 500}
    with pytest.raises(ValueError, match="Missing"):
        boxes.crop(image, "v", 2, metadata, "video_hash")
    with pytest.raises(ValueError, match="hash"):
        boxes.crop(image, "v", 1, metadata, "different_video")
    with pytest.raises(ValueError, match="all and only"):
        boxes.assert_complete()
    cropped, bounds = boxes.crop(image, "v", 1, metadata, "video_hash")
    assert cropped.size == (225, 129)
    assert bounds == (32, 16, 257, 145)
    boxes.assert_complete()


def test_changed_provenance_artifact_is_rejected(tmp_path):
    path, _, artifact = _boxes(tmp_path)
    artifact.write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        VerifiedUnionBoxes(path)


def test_real_verified_full_frame_union_is_logged_without_fallback(tmp_path):
    path, payload, _ = _boxes(tmp_path)
    payload["records"][0]["person_box_xyxy"] = [0, 0, 300, 400]
    payload["records"][0]["object_box_xyxy"] = [250, 200, 1000, 500]
    path.write_text(json.dumps(payload), encoding="utf-8")
    boxes = VerifiedUnionBoxes(path)
    cropped, _ = boxes.crop(Image.new("RGB", (320, 160)), "v", 1,
                           {"fps": 24, "width": 1000, "height": 500}, "video_hash")
    assert cropped.size == (320, 160)
    assert boxes.full_frame_union_count == 1


def test_extractor_uses_boxes_directory_and_does_not_write_into_training_output(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("pair_extractor_test", scripts / "extract_pair_boxes.py")
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    _, payload, _ = _boxes(tmp_path)
    checkpoint = tmp_path / "detector.fixture"
    checkpoint.write_text("Synthetic fixture checkpoint; never a real detector")
    detector_config = tmp_path / "detector_config.fixture"
    detector_config.write_text("Synthetic fixture config")
    provenance = payload["provenance"]
    provenance.update({"source_type": "detector", "detector_name": "synthetic test detector",
                       "detector_checkpoint_path": str(checkpoint),
                       "detector_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "detector_config_path": str(detector_config),
                       "detector_config_sha256": hashlib.sha256(detector_config.read_bytes()).hexdigest()})
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"coordinates": "original_frame_pixels_xyxy",
                                      "provenance": provenance,
                                      "frames": [{"video_id": "v", "frame_index": 24,
                                                  "timestamp_s": 1.0, "width": 1000, "height": 500,
                                                  "source_video_sha256": "video_hash",
                                                  "pairs": [_pair()]}]}))
    training_dir, boxes_dir = tmp_path / "training", tmp_path / "boxes"
    config = {"detector": {"checkpoint_sha256": provenance["detector_checkpoint_sha256"],
                            "tracker_version": "synthetic_fixture"},
              "required_detector_metadata": ["checkpoint_sha256", "tracker_version"],
              "verified_predictions": str(predictions), "output_dir": str(training_dir),
              "boxes_output_dir": str(boxes_dir)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = extractor.extract(config, config_path)
    assert result["status"] == "verified_prediction_adapter_exported"
    assert Path(result["boxes_file"]["path"]) == boxes_dir / "boxes.jsonl"
    assert (boxes_dir / "verified_union_boxes.json").is_file()
    assert not training_dir.exists()

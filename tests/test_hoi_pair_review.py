"""Score-blind actor review preserves unknowns and requires explicit human evidence."""

import copy
import json
import re
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from temporal_hoi.data.hoi_pair_review import (
    choose_review_cases,
    new_review_annotations,
    render_review_html,
    review_manifest_fingerprint,
    validate_review_annotations,
    validate_review_manifest,
)
from temporal_hoi.data.vidhoi import ClassMapping, HOIPair
from temporal_hoi.inference.hoi_static import (
    OracleEntity,
    OracleFrame,
    file_sha256,
    proposal_pairs,
)


def make_cache(tmp_path):
    mapping = ClassMapping({0: "person", 1: "cup"},
                           {0: "hold", 1: "watch", 2: "play(instrument)", 3: "open"})
    image = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (20, 100, 200)).save(image)
    entities = (OracleEntity("p1", 0, (0., 0., 3., 8.)),
                OracleEntity("p2", 0, (3., 0., 6., 8.)),
                OracleEntity("cup", 1, (8., 0., 12., 8.)))
    truth = (HOIPair(0, 1, entities[0].bbox_xyxy, entities[2].bbox_xyxy, (1., 0., 0., 0.),
                     subject_track_id="p1", object_track_id="cup"),)
    frames = [OracleFrame(video, fid + 1, fid, float(fid), image, file_sha256(image), 12, 8,
                          entities, truth)
              for video, count in (("train/a", 3), ("train/b", 2), ("train/c", 1))
              for fid in range(count)]
    protocol = {"schema_version": "vidhoi_head_protocol_v1", "label_policy": "positive_only_unknown",
                "train_video_ids": ["train/a", "train/b", "train/c"], "dev_video_ids": ["dev/d"],
                "excluded_final_video_ids": ["final/f"], "pseudo_unseen_triplets": [[0, 3, 0]],
                "final_unseen_triplets": [[0, 3, 1]],
                "video_positive_triplets": {video: [[0, 0, 1]]
                                             for video in ("train/a", "train/b", "train/c")}}
    rows = []
    for fi, frame in enumerate(frames):
        for subject, obj in proposal_pairs(frame):
            rows.append((fi, subject.track_id, obj.track_id, obj.class_id,
                         [subject.track_id == "p1" and obj.track_id == "cup", False, False, False]))
    vectors = {"cup": [1., 0., 0.], "p1": [0., 1., 0.], "p2": [0., 0., 1.]}
    cache = {"mapping": mapping, "frames": frames,
             "object_features": np.asarray([vectors[r[2]] for r in rows], dtype=np.float32),
             "frame_index": np.asarray([r[0] for r in rows], dtype=np.int64),
             "subject_track_id": np.asarray([r[1] for r in rows]),
             "object_track_id": np.asarray([r[2] for r in rows]),
             "object_class": np.asarray([r[3] for r in rows], dtype=np.int64),
             "positive_mask": np.asarray([r[4] for r in rows], dtype=bool),
             "metadata": {"role": "train", "source_split": "train", "test_fixture": True,
                          "label_policy": "positive_only_unknown", "official_final_split_accessed": False,
                          "final_unseen_accessed": False, "pseudo_unseen_in_train": False}}
    return cache, protocol, mapping


@pytest.fixture
def review(tmp_path):
    cache, protocol, mapping = make_cache(tmp_path)
    chosen = choose_review_cases(cache, protocol, mapping, count=2)
    manifest = {"schema_version": "vidhoi_pair_review_v1", "role": "train", "source_split": "train",
                "protocol_sha256": "a" * 64, "mapping_sha256": "b" * 64,
                "selection": chosen["selection"], "cases": chosen["cases"], "files": {}}
    for case in manifest["cases"]:
        center = tmp_path / case["image_path"]
        center.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache["frames"][0].image_path, center)
        manifest["files"][case["image_path"]] = {"bytes": center.stat().st_size,
                                                 "sha256": file_sha256(center)}
        for index in range(2):
            relative = f"context/{case['case_id']}_{index}.jpg"
            context = tmp_path / relative
            context.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 8), (index * 20, 40, 90)).save(context)
            digest = file_sha256(context)
            manifest["files"][relative] = {"bytes": context.stat().st_size, "sha256": digest}
            case["context_frames"].append({"path": relative, "sha256": digest,
                "frame_id": index, "timestamp_s": float(index), "width": 12, "height": 8,
                "boxes": [{"track_id": a["track_id"], "class_id": 0, "bbox_xyxy": a["bbox_xyxy"]}
                          for a in case["actors"]]})
    return manifest, tmp_path


def assert_human(labels):
    labels["reviewer_id"] = "explicit_test_fixture_reviewer"
    labels["reviewed_at_utc"] = "2026-10-30T01:00:00Z"
    labels["assertions"] = {"human_reviewed": True, "not_ai_generated_ground_truth": True,
                            "model_scores_hidden": True}


def test_selection_is_deterministic_video_diverse_and_not_feature_score_driven(tmp_path):
    cache, protocol, mapping = make_cache(tmp_path)
    first = choose_review_cases(cache, protocol, mapping, count=4, seed=42)
    second = choose_review_cases(cache, protocol, mapping, count=4, seed=42)
    assert first == second
    assert len({c["video_id"] for c in first["cases"][:3]}) == 3
    assert first["selection"]["eligible_cases"] == 6
    assert first["selection"]["selected_videos"] == 3
    assert first["selection"]["model_scores_used"] is False
    # Rotate all target embeddings consistently; their values cannot select cases.
    cache["object_features"] = np.roll(cache["object_features"], 1, axis=1)
    cache["model_scores"] = np.arange(1000)  # Data ignored by candidate selection.
    rotated = choose_review_cases(cache, protocol, mapping, count=4, seed=42)
    assert rotated["selection"]["selected_case_ids"] == first["selection"]["selected_case_ids"]


def test_source_positive_badges_do_not_initialize_manual_labels(tmp_path):
    cache, protocol, mapping = make_cache(tmp_path)
    chosen = choose_review_cases(cache, protocol, mapping, count=3)
    for case in chosen["cases"]:
        assert all(a["label"] == "unknown" for a in case["actors"])
        assert sum(a["source_annotation_positive"] for a in case["actors"]) == 1
        for actor in case["actors"]:
            row = actor["human_cache_row"]
            assert cache["object_track_id"][row] == actor["track_id"]
            assert cache["frame_index"][row] == case["frame_index"]
    assert chosen["selection"]["human_feature_reuse"]["recoverable_human_embeddings"] == 12
    assert chosen["selection"]["human_feature_reuse"]["encoder_run"] is False


@pytest.mark.parametrize("reason", ["dev_role", "video_overlap", "full_video_final", "full_video_pseudo"])
def test_development_and_heldout_context_cannot_enter_selection(tmp_path, reason):
    cache, protocol, mapping = make_cache(tmp_path)
    if reason == "dev_role":
        cache["metadata"]["role"] = "validation"
    elif reason == "video_overlap":
        protocol["dev_video_ids"].append("train/a")
    elif reason == "full_video_final":
        protocol["video_positive_triplets"]["train/a"].append([0, 3, 1])
    else:
        protocol["video_positive_triplets"]["train/a"].append([0, 3, 0])
    with pytest.raises(ValueError):
        choose_review_cases(cache, protocol, mapping, count=2)


def test_cache_label_change_and_short_candidate_pool_fail_explicitly(tmp_path):
    cache, protocol, mapping = make_cache(tmp_path)
    with pytest.raises(ValueError, match="no silent fallback"):
        choose_review_cases(cache, protocol, mapping, count=7)
    cache["positive_mask"][0, 0] = False
    with pytest.raises(ValueError, match="labels differ"):
        choose_review_cases(cache, protocol, mapping, count=2)


def test_blank_annotation_is_complete_unknown_pending_and_manifest_bound(review):
    manifest, _ = review
    labels = new_review_annotations(manifest)
    receipt = validate_review_annotations(labels, manifest)
    assert receipt["review_labels_status"] == "pending_labels"
    assert receipt["positive_count"] == receipt["confirmed_negative_count"] == 0
    assert receipt["unknown_count"] == receipt["actor_count"] == 4
    assert receipt["eligible_contrast_case_count"] == 0
    assert all(a["label"] == "unknown" for c in manifest["cases"] for a in c["actors"])
    changed = {**manifest, "notes": "different reviewed artifact"}
    with pytest.raises(ValueError, match="different review manifest"):
        validate_review_annotations(labels, changed)


def test_confirmed_negative_requires_context_rationale_and_human_assertions(review):
    manifest, _ = review
    labels = new_review_annotations(manifest)
    negative = labels["cases"][0]["actors"][1]
    negative["label"] = "confirmed_negative"
    with pytest.raises(ValueError, match="context evidence"):
        validate_review_annotations(labels, manifest)
    negative.update(evidence_context_indices=[0], rationale="Explicit synthetic fixture evidence.")
    with pytest.raises(ValueError, match="reviewer identity"):
        validate_review_annotations(labels, manifest)
    assert_human(labels)
    labels["cases"][0]["actors"][0]["label"] = "positive"
    receipt = validate_review_annotations(labels, manifest)
    assert receipt["review_labels_status"] == "human_asserted_review"
    assert receipt["positive_count"] == receipt["confirmed_negative_count"] == 1
    assert receipt["unknown_count"] == 2
    assert receipt["eligible_contrast_case_count"] == 1


def test_multiple_actors_can_be_positive_without_invented_negative(review):
    manifest, _ = review
    labels = new_review_annotations(manifest)
    assert_human(labels)
    for actor in labels["cases"][0]["actors"]:
        actor["label"] = "positive"
    receipt = validate_review_annotations(labels, manifest)
    assert receipt["positive_count"] == 2
    assert receipt["confirmed_negative_count"] == receipt["eligible_contrast_case_count"] == 0


@pytest.mark.parametrize("change", ["missing_case", "duplicate_case", "duplicate_actor", "bad_label",
                                    "extra_field", "wrong_evidence", "ai_assertion", "naive_time"])
def test_malformed_or_incomplete_annotations_fail_without_silent_omission(review, change):
    manifest, _ = review
    labels = new_review_annotations(manifest)
    if change == "missing_case":
        labels["cases"].pop()
    elif change == "duplicate_case":
        labels["cases"][1] = copy.deepcopy(labels["cases"][0])
    elif change == "duplicate_actor":
        labels["cases"][0]["actors"][1] = copy.deepcopy(labels["cases"][0]["actors"][0])
    elif change == "bad_label":
        labels["cases"][0]["actors"][0]["label"] = "AI_negative"
    elif change == "extra_field":
        labels["model_scores"] = [1, 2]
    elif change == "wrong_evidence":
        labels["cases"][0]["actors"][0]["evidence_context_indices"] = [999]
    else:
        assert_human(labels)
        labels["cases"][0]["actors"][0]["label"] = "positive"
        if change == "ai_assertion":
            labels["assertions"]["not_ai_generated_ground_truth"] = False
        else:
            labels["reviewed_at_utc"] = "2026-10-30T01:00:00"
    with pytest.raises(ValueError):
        validate_review_annotations(labels, manifest)


def test_manifest_rejects_dev_cases_unsafe_paths_and_preconfirmed_labels(review):
    manifest, _ = review
    bad = copy.deepcopy(manifest)
    bad["cases"][0]["video_id"] = "dev/d"
    with pytest.raises(ValueError, match="Development/final"):
        validate_review_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["files"]["../outside.png"] = next(iter(bad["files"].values()))
    with pytest.raises(ValueError, match="Unsafe"):
        validate_review_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["cases"][0]["actors"][0]["label"] = "positive"
    with pytest.raises(ValueError, match="all remain unknown"):
        validate_review_manifest(bad)


def test_standalone_html_embeds_verified_assets_escapes_payload_and_hides_scores(review):
    manifest, root = review
    manifest["cases"][0]["object_name"] = "cup </script><script>unsafe()</script>"
    manifest["cases"][0]["model_scores"] = [111111111]
    manifest["cases"][0]["actors"][0]["model_score"] = 222222222
    manifest["cases"][0]["context_frames"][0]["scores"] = [333333333]
    page = render_review_html(manifest, root)
    payload = re.search(r'id="reviewData">(.*?)</script>', page, re.S).group(1)
    parsed = json.loads(payload)
    assert "\\u003c/script\\u003e" in payload
    assert "model_scores" not in parsed["cases"][0]
    assert "model_score" not in parsed["cases"][0]["actors"][0]
    assert "scores" not in parsed["cases"][0]["context_frames"][0]
    assert "pair_cache_row" not in parsed["cases"][0]["actors"][0]
    assert all(value.startswith("data:image/") for value in parsed["images"].values())
    assert parsed["labels"]["manifest_fingerprint"] == review_manifest_fingerprint(manifest)
    assert "vidhoi_pair_review_annotations.json" in page
    for forbidden in ("fetch(", "localStorage", "sessionStorage", '<script src=', '<link '):
        assert forbidden not in page
    assert "f.boxes||[]" in page  # Moving context uses its own scaled coordinates.
    image = root / manifest["cases"][0]["image_path"]
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes/hash"):
        render_review_html(manifest, root)


def test_review_javascript_parses_if_node_is_available(review, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node runtime unavailable; Python/HTML data invariants still tested")
    manifest, root = review
    page = render_review_html(manifest, root)
    code = re.search(r"<script>\s*(.*?)</script>", page, re.S).group(1)
    path = tmp_path / "review_script.js"
    path.write_text(code, encoding="utf-8")
    checked = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, check=False)
    assert checked.returncode == 0, checked.stderr

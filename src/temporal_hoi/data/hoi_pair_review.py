"""Score-blind train-only actor review and explicit human annotation validation.

Review labels start unknown, including actors carrying a source-positive badge.
Unannotated actors are candidates for review, never automatic negative examples.
All feature reuse refers to the existing frozen cache; selection uses no scores.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Mapping

import numpy as np
from PIL import Image

from temporal_hoi.data.vidhoi import ClassMapping
from temporal_hoi.inference.hoi_static import OracleFrame, file_sha256, proposal_pairs

MANIFEST_SCHEMA = "vidhoi_pair_review_v1"
ANNOTATIONS_SCHEMA = "vidhoi_pair_annotations_v1"
REVIEW_LABELS = ("positive", "confirmed_negative", "unknown")
# Fixed before candidate scoring. These define this contact/manipulation pilot,
# not an exhaustive list of interactions or a general actor-negative ontology.
CONTACT_PREDICATES = (
    "hold", "play(instrument)", "ride", "use", "feed", "touch", "carry", "hug",
    "hit", "kick", "pull", "push", "grab", "lift", "throw", "cut", "pat", "caress",
    "shake_hand_with", "hold_hand_of",
)


def review_manifest_fingerprint(manifest: dict) -> str:
    """SHA256 of sorted compact UTF-8 JSON, independent of file indentation."""
    data = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _check(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative(value: str) -> str:
    _check(isinstance(value, str) and bool(value), "Asset path must be a nonempty relative path")
    path = PurePosixPath(value)
    _check(not path.is_absolute() and ".." not in path.parts and "\\" not in value
           and ":" not in value and str(path) == value, "Unsafe review asset path")
    return value


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _hash_order(seed: int, kind: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{kind}|{value}".encode()).hexdigest()


def _cache_rows(cache: Mapping) -> tuple[dict, dict, dict]:
    """Bind all proposal identities/positives and audit reuse without scoring features."""
    frames, mapping = cache["frames"], cache["mapping"]
    features = np.asarray(cache["object_features"])
    _check(features.ndim == 2 and features.dtype == np.float32
           and np.isfinite(features).all(), "Need finite FP32 cached object features")
    _check(np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5),
           "Cached object features must already be normalized")
    count = len(features)
    for name in ("object_class", "frame_index", "subject_track_id", "object_track_id"):
        _check(np.asarray(cache[name]).shape == (count,), "Cache identity vector shape mismatch")
    for name in ("object_class", "frame_index"):
        _check(np.asarray(cache[name]).dtype.kind in "iu", "Cache indices/classes must be integers")
    for name in ("subject_track_id", "object_track_id"):
        _check(np.asarray(cache[name]).dtype.kind == "U", "Cache track IDs must be Unicode vectors")
    labels = np.asarray(cache["positive_mask"])
    _check(labels.dtype == bool and labels.shape == (count, len(mapping.predicates)),
           "Cache positive_mask shape/dtype mismatch")
    pair_rows, target_rows = {}, {}
    cursor = 0
    multi_frames = multi_people = recovered_people = 0
    for fi, frame in enumerate(frames):
        _check(isinstance(frame, OracleFrame), "Review needs canonical OracleFrame records")
        known = {(p.subject_track_id, p.object_track_id): np.asarray(p.predicate_values) == 1
                 for p in frame.ground_truth}
        for subject, obj in proposal_pairs(frame):
            _check(cursor < count, "Cache dropped proposal rows")
            _check(int(cache["frame_index"][cursor]) == fi
                   and int(cache["object_class"][cursor]) == obj.class_id
                   and str(cache["subject_track_id"][cursor]) == subject.track_id
                   and str(cache["object_track_id"][cursor]) == obj.track_id,
                   "Cache proposal order differs from source frames")
            expected = known.get((subject.track_id, obj.track_id), np.zeros(labels.shape[1], bool))
            _check(np.array_equal(labels[cursor], expected), "Cache labels differ from source positives")
            pair_rows[(fi, subject.track_id, obj.track_id)] = cursor
            target_key = (fi, obj.track_id)
            if target_key in target_rows:
                _check(np.array_equal(features[target_rows[target_key]], features[cursor]),
                       "Repeated target does not reuse identical object features")
            else:
                target_rows[target_key] = cursor
            cursor += 1
        people = [e.track_id for e in frame.entities if e.class_id == 0]
        if len(people) >= 2:
            multi_frames += 1
            multi_people += len(people)
            recovered_people += sum((fi, track) in target_rows for track in people)
    _check(cursor == count and multi_people == recovered_people,
           "Human crops are missing from multi-person person-as-target rows")
    return pair_rows, target_rows, {
        "multi_person_frames": multi_frames, "multi_person_human_tracks": multi_people,
        "recoverable_human_embeddings": recovered_people, "encoder_run": False,
        "method": "object_features at the same frame/person track when that person is another actor's target",
        "single_person_self_crop_available": False,
    }


def choose_review_cases(source_train_cache: Mapping, protocol: dict, mapping: ClassMapping,
                        count: int = 30, seed: int = 42) -> dict:
    """Choose one seeded case per eligible video, then fill by a fixed seeded hash.

    An eligible group has the same target/predicate, at least two candidate people,
    an upstream positive, and an actor with no upstream label for that relation.
    Metadata controls eligibility; frozen feature values never determine selection.
    Asset paths are planned relative paths; the caller materializes and hashes
    context frames before validating/rendering the final manifest.
    """
    _check(type(count) is int and count > 0 and type(seed) is int, "Need positive count/integer seed")
    _check(isinstance(mapping, ClassMapping), "Require the locked ClassMapping")
    _check(source_train_cache["mapping"] == mapping, "Cache and requested mapping differ")
    meta = source_train_cache.get("metadata", {})
    _check(meta.get("role") == "train" and meta.get("source_split") == "train"
           and meta.get("label_policy") == "positive_only_unknown", "Only upstream-train cache is eligible")
    _check(all(meta.get(key) is False for key in
               ("official_final_split_accessed", "final_unseen_accessed", "pseudo_unseen_in_train")),
           "Cache must declare final/pseudo context isolation")
    _check(protocol.get("schema_version") == "vidhoi_head_protocol_v1"
           and protocol.get("label_policy") == "positive_only_unknown", "Wrong source protocol")
    train = set(protocol["train_video_ids"])
    dev = set(protocol["dev_video_ids"])
    excluded = set(protocol["excluded_final_video_ids"])
    _check(not train & (dev | excluded), "Train/development/final video overlap")
    frames = source_train_cache["frames"]
    _check({f.video_id for f in frames} == train, "Train frame video IDs differ from locked protocol")
    forbidden = {tuple(x) for key in ("pseudo_unseen_triplets", "final_unseen_triplets")
                 for x in protocol[key]}
    for video in train:
        _check(video in protocol["video_positive_triplets"], "Missing full-video annotation context")
        _check(not ({tuple(x) for x in protocol["video_positive_triplets"][video]} & forbidden),
               "Pseudo/final composition occurs in a train video context")
    pair_rows, human_rows, reuse = _cache_rows(source_train_cache)
    eligible = []
    by_predicate = Counter()
    for fi, frame in enumerate(frames):
        entities = {e.track_id: e for e in frame.entities}
        people = sorted((e for e in frame.entities if e.class_id == 0), key=lambda e: e.track_id)
        positives = defaultdict(set)
        for pair in frame.ground_truth:
            for pid, value in enumerate(pair.predicate_values):
                if value == 1:
                    _check((0, pid, pair.object_class) not in forbidden,
                           "Forbidden sampled positive in training cache")
                    positives[(pair.object_track_id, pid)].add(pair.subject_track_id)
        for (target_id, pid), source_positive in sorted(positives.items()):
            predicate = mapping.predicates[pid]
            if predicate not in CONTACT_PREDICATES:
                continue
            actors = [p for p in people if p.track_id != target_id]
            if len(actors) < 2 or not ({p.track_id for p in actors} - source_positive):
                continue
            obj = entities[target_id]
            key = f"{frame.video_id}|{frame.frame_id}|{target_id}|{pid}"
            case_id = "pair_" + hashlib.sha256(key.encode()).hexdigest()[:20]
            eligible.append({
                "case_id": case_id, "frame_index": fi, "video_id": frame.video_id,
                "frame_id": frame.frame_id, "source_frame_id": frame.source_frame_id,
                "timestamp_s": frame.timestamp_s, "width": frame.width, "height": frame.height,
                "object_track_id": target_id, "object_bbox_xyxy": list(obj.bbox_xyxy),
                "object_class": obj.class_id, "object_name": mapping.objects[obj.class_id],
                "predicate_id": pid, "predicate_name": predicate,
                "actors": [{"track_id": actor.track_id, "bbox_xyxy": list(actor.bbox_xyxy),
                            "source_annotation_positive": actor.track_id in source_positive,
                            "label": "unknown", "pair_cache_row": pair_rows[(fi, actor.track_id, target_id)],
                            "human_cache_row": human_rows[(fi, actor.track_id)]} for actor in actors],
                "image_path": f"frames/{case_id}.png", "image_sha256": frame.image_sha256,
                "source_image_sha256": frame.image_sha256, "context_frames": [], "video": None,
            })
            by_predicate[predicate] += 1
    _check(len(eligible) >= count,
           f"Only {len(eligible)} eligible contact groups for requested {count}; no silent fallback")
    _check(len({c["case_id"] for c in eligible}) == len(eligible), "Review case ID collision")
    by_video = defaultdict(list)
    for case in eligible:
        by_video[case["video_id"]].append(case)
    video_order = sorted(by_video, key=lambda v: _hash_order(seed, "video", v))
    selected = [min(by_video[v], key=lambda c: _hash_order(seed, "case", c["case_id"]))
                for v in video_order[:count]]
    ids = {c["case_id"] for c in selected}
    remainder = sorted((c for c in eligible if c["case_id"] not in ids),
                       key=lambda c: _hash_order(seed, "case", c["case_id"]))
    selected.extend(remainder[:count - len(selected)])
    selection = {
        "schema_version": "vidhoi_pair_review_selection_v1", "status": "completed",
        "role": "train", "source_split": "train", "seed": seed,
        "requested_cases": count, "selected_cases": len(selected),
        "eligible_cases": len(eligible), "eligible_videos": len(by_video),
        "selected_videos": len({c["video_id"] for c in selected}),
        "contact_predicates_fixed_before_selection": list(CONTACT_PREDICATES),
        "eligibility": "same target/predicate; at least two people; source-positive and source-unannotated actors",
        "selection_rule": "one SHA256(seed|case|case_id) minimum per video ordered by SHA256(seed|video|video_id), then fill remaining cases by case hash",
        "selection_before_new_model_scores": True, "model_scores_used": False,
        "source_GT_used_for_candidate_definition_only": True,
        "initial_human_labels": "all_unknown_including_source_positive_actors",
        "eligible_predicate_counts": dict(sorted(by_predicate.items())),
        "selected_case_ids": [c["case_id"] for c in selected],
        "source_train_video_ids": sorted(train), "excluded_dev_video_ids": sorted(dev),
        "excluded_final_video_ids": sorted(excluded),
        "input_hashes": meta.get("input_hashes", {}),
        "human_feature_reuse": reuse, "full_benchmark_reproduction_claim": False,
        "official_final_split_accessed": False, "final_unseen_accessed": False,
        "test_fixture": meta.get("test_fixture") is True,
    }
    return {"cases": selected, "selection": selection}


def _box(value, width: int, height: int) -> None:
    _check(isinstance(value, list) and len(value) == 4
           and all(type(x) in (int, float) and math.isfinite(x) for x in value),
           "Box requires four finite numeric pixel coordinates")
    _check(0 <= value[0] < value[2] <= width and 0 <= value[1] < value[3] <= height,
           "Review box outside its image or degenerate")


def validate_review_manifest(manifest: dict) -> dict:
    """Validate a materialized source-only review document, not inferred labels."""
    _check(isinstance(manifest, dict) and manifest.get("schema_version") == MANIFEST_SCHEMA,
           "Require vidhoi_pair_review_v1")
    _check(manifest.get("role") == "train" and manifest.get("source_split") == "train",
           "Review must remain upstream train only")
    for key in ("protocol_sha256", "mapping_sha256"):
        _check(_digest(manifest.get(key)), "Review needs pinned protocol/mapping hashes")
    for key in ("official_final_split_accessed", "final_unseen_accessed"):
        _check(manifest.get(key, False) is False, "Final data is excluded from this review")
    cases = manifest.get("cases")
    _check(isinstance(cases, list) and bool(cases), "Review cases must be nonempty")
    selection = manifest.get("selection")
    _check(isinstance(selection, dict) and selection.get("role") == "train"
           and selection.get("model_scores_used") is False, "Score-blind train selection receipt required")
    train = set(selection.get("source_train_video_ids", []))
    forbidden = set(selection.get("excluded_dev_video_ids", [])) | set(selection.get("excluded_final_video_ids", []))
    _check(bool(train) and not train & forbidden, "Review role whitelist overlaps development/final")
    _check(selection.get("selected_case_ids") == [c.get("case_id") for c in cases],
           "Review case IDs/order differ from score-blind selection receipt")
    files = manifest.get("files")
    _check(isinstance(files, dict) and bool(files), "Materialized review needs an asset inventory")
    for path, entry in files.items():
        _relative(path)
        _check(isinstance(entry, dict) and _digest(entry.get("sha256"))
               and type(entry.get("bytes")) is int and entry["bytes"] > 0, "Invalid review file inventory")
    seen_ids, assets = set(), set()
    actor_count = context_count = 0
    for case in cases:
        cid = case.get("case_id")
        _check(isinstance(cid, str) and bool(cid) and cid not in seen_ids, "Duplicate/missing review case ID")
        seen_ids.add(cid)
        _check(isinstance(case.get("video_id"), str) and bool(case["video_id"]), "Missing video ID")
        _check(case["video_id"] in train and case["video_id"] not in forbidden,
               "Development/final case cannot enter this train review")
        _check(type(case.get("frame_id")) is int and case["frame_id"] >= 0
               and case.get("source_frame_id") == case["frame_id"] + 1, "Review frame-index join mismatch")
        width, height = case.get("width"), case.get("height")
        _check(type(width) is int and width > 0 and type(height) is int and height > 0,
               "Positive original image dimensions required")
        _check(type(case.get("timestamp_s")) in (int, float) and math.isfinite(case["timestamp_s"])
               and case["timestamp_s"] >= 0, "Review requires an explicit finite timestamp")
        _check(type(case.get("predicate_id")) is int and case["predicate_id"] >= 0
               and case.get("predicate_name") in CONTACT_PREDICATES, "Review predicate outside fixed pilot")
        _check(type(case.get("object_class")) is int and case["object_class"] >= 0
               and isinstance(case.get("object_track_id"), str), "Invalid target identity")
        _box(case.get("object_bbox_xyxy"), width, height)
        actors = case.get("actors")
        _check(isinstance(actors, list) and len(actors) >= 2, "At least two candidate actors required")
        tracks = set()
        positives = []
        for actor in actors:
            track = actor.get("track_id")
            _check(isinstance(track, str) and bool(track) and track not in tracks
                   and track != case["object_track_id"], "Duplicate/invalid/self target actor")
            tracks.add(track)
            _check(actor.get("label") == "unknown", "Source review manifest labels must all remain unknown")
            _check(type(actor.get("source_annotation_positive")) is bool,
                   "Source annotation badge must be an explicit boolean")
            positives.append(actor["source_annotation_positive"])
            _box(actor.get("bbox_xyxy"), width, height)
        _check(any(positives) and not all(positives), "Review needs positive and source-unannotated actors")
        actor_count += len(actors)
        path = _relative(case.get("image_path"))
        _check(path in files and files[path]["sha256"] == case.get("image_sha256"),
               "Center image hash absent from inventory")
        assets.add(path)
        contexts = case.get("context_frames")
        _check(isinstance(contexts, list) and bool(contexts), "Materialized temporal context is required")
        previous = -1.0
        for context in contexts:
            path = _relative(context.get("path"))
            _check(path in files and files[path]["sha256"] == context.get("sha256"),
                   "Context image hash absent from inventory")
            _check(type(context.get("frame_id")) is int and context["frame_id"] >= 0,
                   "Context frame index must be explicit")
            timestamp = context.get("timestamp_s")
            _check(type(timestamp) in (int, float) and math.isfinite(timestamp)
                   and timestamp >= previous and timestamp >= 0, "Context timestamps must be ordered")
            previous = timestamp
            cw, ch = context.get("width"), context.get("height")
            _check(type(cw) is int and cw > 0 and type(ch) is int and ch > 0,
                   "Context view dimensions required")
            for box in context.get("boxes", []):
                _check(isinstance(box.get("track_id"), str), "Context track IDs must be strings")
                _box(box.get("bbox_xyxy"), cw, ch)
            assets.add(path)
            context_count += 1
        if case.get("video") is not None:
            video = case["video"]
            path = _relative(video.get("path"))
            _check(path in files and files[path]["sha256"] == video.get("sha256"),
                   "Context video hash absent from inventory")
            assets.add(path)
    _check(set(files) == assets, "Review asset inventory must contain only referenced images/context video")
    return {"case_count": len(cases), "actor_count": actor_count,
            "context_frame_count": context_count, "referenced_assets": sorted(assets),
            "manifest_fingerprint": review_manifest_fingerprint(manifest)}


def new_review_annotations(manifest: dict) -> dict:
    """A complete blank label set is valid pending work, not human verification."""
    validate_review_manifest(manifest)
    return {"schema_version": ANNOTATIONS_SCHEMA,
            "manifest_fingerprint": review_manifest_fingerprint(manifest),
            "reviewer_id": "", "reviewed_at_utc": None,
            "assertions": {"human_reviewed": False, "not_ai_generated_ground_truth": False,
                           "model_scores_hidden": True},
            "cases": [{"case_id": case["case_id"], "actors": [
                {"track_id": actor["track_id"], "label": "unknown",
                 "evidence_context_indices": [], "rationale": ""} for actor in case["actors"]]}
                for case in manifest["cases"]]}


def validate_review_annotations(labels: dict, manifest: dict) -> dict:
    """Require complete IDs and explicit evidence for each confirmed negative.

    Reviewer assertions are provenance supplied by the reviewer, not an identity
    verification service. Unknown labels can remain unknown after real review.
    """
    validated = validate_review_manifest(manifest)
    _check(isinstance(labels, dict) and labels.get("schema_version") == ANNOTATIONS_SCHEMA,
           "Require vidhoi_pair_annotations_v1")
    _check(labels.get("manifest_fingerprint") == validated["manifest_fingerprint"],
           "Labels belong to a different review manifest")
    _check(set(labels) == {"schema_version", "manifest_fingerprint", "reviewer_id", "reviewed_at_utc",
                           "assertions", "cases"}, "Unexpected/missing annotation provenance fields")
    assertions = labels.get("assertions")
    _check(isinstance(assertions, dict)
           and set(assertions) == {"human_reviewed", "not_ai_generated_ground_truth", "model_scores_hidden"}
           and all(type(v) is bool for v in assertions.values()), "Explicit reviewer assertions required")
    _check(isinstance(labels.get("reviewer_id"), str), "reviewer_id must be an explicit string")
    expected = {c["case_id"]: c for c in manifest["cases"]}
    records = labels.get("cases")
    _check(isinstance(records, list) and len(records) == len(expected), "Complete case label set required")
    counts = Counter({name: 0 for name in REVIEW_LABELS})
    seen = set()
    contrasts = fully_labeled = 0
    for case in records:
        _check(isinstance(case, dict) and set(case) == {"case_id", "actors"}, "Unexpected case label fields")
        cid = case.get("case_id")
        _check(isinstance(cid, str) and cid in expected and cid not in seen,
               "Unknown/duplicate case ID in labels")
        seen.add(cid)
        wanted = {a["track_id"] for a in expected[cid]["actors"]}
        actors = case.get("actors")
        _check(isinstance(actors, list) and len(actors) == len(wanted), "Complete actor label set required")
        tracks, states = set(), set()
        for actor in actors:
            _check(isinstance(actor, dict) and set(actor) ==
                   {"track_id", "label", "evidence_context_indices", "rationale"},
                   "Unexpected/missing actor annotation fields")
            track = actor.get("track_id")
            _check(isinstance(track, str) and track in wanted and track not in tracks,
                   "Unknown/duplicate actor label ID")
            tracks.add(track)
            label = actor.get("label")
            _check(label in REVIEW_LABELS, "Unsupported review label")
            rationale = actor.get("rationale")
            _check(isinstance(rationale, str), "Annotation rationale must be explicit text")
            evidence = actor.get("evidence_context_indices")
            _check(isinstance(evidence, list)
                   and all(type(i) is int and 0 <= i < len(expected[cid]["context_frames"]) for i in evidence)
                   and len(set(evidence)) == len(evidence),
                   "Evidence must reference unique context frame indices from this case")
            if label == "confirmed_negative":
                _check(bool(evidence) and bool(rationale.strip()),
                       "Confirmed negative requires viewed context evidence and a rationale")
            counts[label] += 1
            states.add(label)
        contrasts += int("positive" in states and "confirmed_negative" in states)
        fully_labeled += int("unknown" not in states)
    changed = counts["positive"] + counts["confirmed_negative"] > 0
    reviewed = assertions["human_reviewed"]
    if changed or reviewed:
        _check(bool(labels["reviewer_id"].strip()) and all(assertions.values()),
               "Human labels require reviewer identity and all explicit human/score-blind assertions")
        timestamp = labels.get("reviewed_at_utc")
        _check(isinstance(timestamp, str), "Human-reviewed labels need a UTC review timestamp")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("Invalid human review timestamp") from error
        _check(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
               "Human review timestamp must specify UTC")
    else:
        _check(labels.get("reviewed_at_utc") is None, "Pending labels must not claim a review timestamp")
    return {"schema_version": "vidhoi_pair_annotations_validation_v1", "status": "completed",
            "review_labels_status": "human_asserted_review" if reviewed else "pending_labels",
            "manifest_fingerprint": validated["manifest_fingerprint"],
            "annotations_fingerprint": review_manifest_fingerprint(labels),
            "case_count": len(expected), "actor_count": sum(counts.values()),
            "positive_count": counts["positive"], "confirmed_negative_count": counts["confirmed_negative"],
            "unknown_count": counts["unknown"], "eligible_contrast_case_count": contrasts,
            "fully_labeled_case_count": fully_labeled,
            "reviewer_provenance_level": "reviewer assertion; identity/context viewing not independently verified",
            "source_GT_is_separate_from_human_labels": True,
            "source_annotation_badges_visible": True, "source_label_blind_claim": False,
            "official_final_split_accessed": False}


def render_review_html(manifest: dict, assets_root: str | Path) -> str:
    """Return a standalone offline review page with source pixels, never scores.

    Images are verified and embedded as data URIs. All state stays in browser
    memory until the user downloads a complete JSON label set; no network or
    local/browser storage is used. Context overlays use context view coordinates.
    """
    info = validate_review_manifest(manifest)
    root = Path(assets_root).resolve()
    images, dimensions = {}, {}
    for relative in info["referenced_assets"]:
        target = root / relative
        _check(target.resolve().is_relative_to(root) and not target.is_symlink(), "Review asset escapes root")
        entry = manifest["files"][relative]
        _check(target.is_file() and target.stat().st_size == entry["bytes"]
               and file_sha256(target) == entry["sha256"], "Review asset bytes/hash mismatch")
        if target.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(target) as image:
                dimensions[relative] = image.size
                image.verify()
            mime = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
            images[relative] = f"data:{mime};base64," + base64.b64encode(target.read_bytes()).decode("ascii")
    for case in manifest["cases"]:
        _check(case["image_path"] in images and all(c["path"] in images for c in case["context_frames"]),
               "Review center/context assets must be PNG or JPEG images")
        _check(dimensions[case["image_path"]] == (case["width"], case["height"])
               and all(dimensions[c["path"]] == (c["width"], c["height"]) for c in case["context_frames"]),
               "Review raster dimensions disagree with manifest/view coordinates")
    # Whitelisted source fields only: model scores/checkpoints cannot leak into UI.
    cases = [{key: case[key] for key in
              ("case_id", "video_id", "frame_id", "timestamp_s", "width", "height", "object_track_id",
               "object_bbox_xyxy", "object_name", "predicate_name", "actors", "image_path", "context_frames")}
             for case in manifest["cases"]]
    for case in cases:
        case["actors"] = [{key: actor[key] for key in
                           ("track_id", "bbox_xyxy", "source_annotation_positive", "label")}
                          for actor in case["actors"]]
        case["context_frames"] = [{**{key: context[key] for key in
                                      ("path", "timestamp_s", "frame_id", "width", "height")},
                                   "boxes": [{key: box[key] for key in ("track_id", "bbox_xyxy")}
                                             for box in context.get("boxes", [])]}
                                  for context in case["context_frames"]]
    payload = {"cases": cases, "images": images, "labels": new_review_annotations(manifest)}
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for char, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                          ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        serialized = serialized.replace(char, escaped)
    return _REVIEW_HTML.replace("__REVIEW_PAYLOAD__", serialized)


_REVIEW_HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Xác nhận người đang tương tác</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#162536;font:16px/1.5 system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:22px}h1{font-size:28px;margin:0 0 8px}h2{font-size:18px;margin:0 0 12px}
p{margin:8px 0}.muted{color:#526478}.card{background:white;border:1px solid #d9e1e8;border-radius:14px;padding:18px;margin:16px 0}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.row.between{justify-content:space-between}.views{display:grid;grid-template-columns:1fr 1fr;gap:18px}
canvas{display:block;max-width:100%;max-height:540px;object-fit:contain;margin:auto;background:#152332;border-radius:8px}
button,select,input,textarea{font:inherit;border:1px solid #a8b6c4;border-radius:7px;padding:8px 12px;background:#fff;color:inherit}
button{cursor:pointer}button.primary{background:#174d74;color:#fff;border-color:#174d74}button:disabled{opacity:.45;cursor:default}
input[type=checkbox]{width:20px;height:20px}input[type=text]{min-width:230px}select{max-width:100%}textarea{width:100%;min-height:72px;resize:vertical}
.actor{border-left:6px solid #48a6cf;padding:12px 16px;border-radius:8px;background:#f6f9fc;margin:14px 0}.badge{font-size:13px;background:#e9edf3;border-radius:5px;padding:2px 7px;color:#405068}
.help{font-size:14px;color:#526478}.warn{color:#854315}.error{color:#ab2525;white-space:pre-wrap}.success{color:#146239}
label{display:block;margin:6px 0}label.inline{display:flex;gap:10px;align-items:flex-start}.nav select{flex:1;min-width:160px}
@media(max-width:780px){main{padding:12px}.views{grid-template-columns:1fr}h1{font-size:23px}}
</style></head><body><main>
<h1>Xác nhận người đang tương tác</h1>
<p>Xem khung đích và chuỗi ngữ cảnh, rồi đánh dấu từng người cho <strong>cùng vật và cùng hành động</strong>.</p>
<p class="muted">Tất cả nhãn của bạn bắt đầu ở “Chưa rõ”. Nhiều người có thể cùng đúng. Không có nhãn trong nguồn không có nghĩa là sai; “chơi đàn” cũng không loại trừ “cầm đàn”.</p>
<section class="card"><div class="row nav"><button id="previous">← Trước</button><select id="caseSelect" aria-label="Chọn tình huống"></select><button id="next">Tiếp →</button></div>
<p id="caseMeta" class="help"></p><h2 id="question"></h2><div class="views"><div><h2>Khung đích — các hộp ở tọa độ ảnh gốc</h2><canvas id="center"></canvas></div>
<div><div class="row between"><h2>Ngữ cảnh — màu và mã người được giữ nhất quán</h2><button id="play">Tạm dừng</button></div><canvas id="context"></canvas>
<div class="row"><input type="range" id="contextSeek" min="0" value="0" aria-label="Chọn khung ngữ cảnh"><span id="contextTime" class="help"></span></div>
<p class="help">Hộp vàng là vật đang xét. Hộp có màu là người trong câu hỏi. Hộp có thể vắng khi người/vật không hiện trong annotation ở khung đó.</p></div></div>
<div id="actors"></div><p id="progress" class="help"></p></section>
<section class="card"><h2>Lưu nhãn đã xem</h2><label>Mã người kiểm tra (có thể dùng bí danh)<br><input id="reviewer" type="text" autocomplete="off"></label>
<label class="inline"><input id="humanAssertion" type="checkbox"><span>Tôi đã tự xem ngữ cảnh để gán các nhãn xác nhận; tôi không dùng điểm mô hình hoặc AI làm nhãn chuẩn.</span></label>
<p class="help">Mỗi “Xác nhận không tương tác” cần khung ngữ cảnh làm bằng chứng và lý do. Nếu không chắc, giữ “Chưa rõ”.</p>
<div class="row"><button id="download" class="primary">Tải toàn bộ nhãn JSON</button><label>Nạp tiếp nhãn đã lưu <input id="importLabels" type="file" accept="application/json,.json"></label></div>
<p class="warn">Trang hoạt động ngoại tuyến. Dữ liệu chỉ ở bộ nhớ trang; hãy tải JSON trước khi đóng hoặc tải lại.</p><p id="message" role="status"></p></section>
</main><script type="application/json" id="reviewData">__REVIEW_PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('reviewData').textContent);
let labels=JSON.parse(JSON.stringify(data.labels)),caseIndex=0,contextIndex=0,playing=true;
const colors=['#00c9ff','#ff6a88','#6df59a','#d1a0ff','#ffad48','#72d6d0','#e8df59','#ffffff'];
const $=id=>document.getElementById(id);const imageCache=new Map();
function imageFor(path){if(!imageCache.has(path)){const image=new Image();image.src=data.images[path];imageCache.set(path,image);}return imageCache.get(path);}
function current(){return data.cases[caseIndex];}function currentLabels(){return labels.cases.find(c=>c.case_id===current().case_id);}
function actorColor(track){return colors[current().actors.findIndex(a=>a.track_id===track)%colors.length];}
function draw(canvasId,path,width,height,boxes){const canvas=$(canvasId),owner=current().case_id,request=owner+'|'+path,im=imageFor(path);canvas.dataset.request=request;const paint=()=>{if(canvas.dataset.request!==request||current().case_id!==owner)return;canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d');ctx.drawImage(im,0,0,width,height);ctx.lineWidth=Math.max(2,width/260);ctx.font=Math.max(13,width/40)+'px system-ui';
for(const b of boxes){const isObject=b.track_id===current().object_track_id;const isActor=current().actors.some(a=>a.track_id===b.track_id);if(!isObject&&!isActor)continue;const color=isObject?'#ffdf48':actorColor(b.track_id),v=b.bbox_xyxy;ctx.strokeStyle=color;ctx.strokeRect(v[0],v[1],v[2]-v[0],v[3]-v[1]);const text=(isObject?'Vật #':'Người #')+b.track_id;const tw=ctx.measureText(text).width+10;const top=Math.max(0,v[1]-24);ctx.fillStyle='#142230';ctx.fillRect(v[0],top,tw,25);ctx.fillStyle=color;ctx.fillText(text,v[0]+5,top+18);}};
if(im.complete&&im.naturalWidth)paint();else im.addEventListener('load',paint,{once:true});}
function drawContext(){const c=current(),f=c.context_frames[contextIndex];$('contextSeek').value=contextIndex;$('contextTime').textContent=f.timestamp_s.toFixed(3)+' s · frame '+f.frame_id;draw('context',f.path,f.width,f.height,f.boxes||[]);}
function progress(){let positive=0,negative=0,unknown=0;for(const c of labels.cases)for(const a of c.actors){if(a.label==='positive')positive++;else if(a.label==='confirmed_negative')negative++;else unknown++;}$('progress').textContent='Đã đánh dấu tương tác: '+positive+' · Xác nhận không tương tác: '+negative+' · Chưa rõ: '+unknown;}
function render(){const c=current();contextIndex=0;$('caseSelect').value=caseIndex;$('previous').disabled=caseIndex===0;$('next').disabled=caseIndex===data.cases.length-1;$('caseMeta').textContent='Tình huống '+(caseIndex+1)+'/'+data.cases.length+' · Video '+c.video_id+' · Khung '+c.frame_id+' · '+c.timestamp_s.toFixed(3)+' s';$('question').textContent='Người nào thực hiện “'+c.predicate_name+'” với '+c.object_name+' #'+c.object_track_id+'?';
const boxes=c.actors.map(a=>({track_id:a.track_id,bbox_xyxy:a.bbox_xyxy}));boxes.push({track_id:c.object_track_id,bbox_xyxy:c.object_bbox_xyxy});draw('center',c.image_path,c.width,c.height,boxes);$('contextSeek').max=c.context_frames.length-1;drawContext();$('actors').replaceChildren();
for(const [i,actor] of c.actors.entries()){const a=currentLabels().actors.find(x=>x.track_id===actor.track_id),section=document.createElement('section');section.className='actor';section.style.borderColor=colors[i%colors.length];const title=document.createElement('h2');title.textContent='Người #'+actor.track_id;section.append(title);const badge=document.createElement('span');badge.className='badge';badge.textContent=actor.source_annotation_positive?'Nguồn VidHOI đã gán positive — bạn vẫn cần tự xem':'Nguồn chưa gán nhãn quan hệ này — chưa kết luận sai';section.append(badge);
const choice=document.createElement('select');choice.setAttribute('aria-label','Nhãn người '+actor.track_id);for(const [value,text] of [['unknown','Chưa rõ'],['positive','Xác nhận có tương tác'],['confirmed_negative','Xác nhận không tương tác']]){const option=document.createElement('option');option.value=value;option.textContent=text;choice.append(option);}choice.value=a.label;choice.onchange=()=>{a.label=choice.value;progress();};const label=document.createElement('label');label.textContent='Kết luận của bạn ';label.append(choice);section.append(label);
const evidence=document.createElement('select');const empty=document.createElement('option');empty.value='';empty.textContent='Chọn khung ngữ cảnh bạn đã kiểm tra';evidence.append(empty);for(const [j,f] of c.context_frames.entries()){const op=document.createElement('option');op.value=j;op.textContent=f.timestamp_s.toFixed(3)+' s · frame '+f.frame_id;evidence.append(op);}evidence.value=a.evidence_context_indices.length?String(a.evidence_context_indices[0]):'';evidence.onchange=()=>{a.evidence_context_indices=evidence.value===''?[]:[Number(evidence.value)];if(evidence.value!==''){contextIndex=Number(evidence.value);playing=false;$('play').textContent='Phát tiếp';drawContext();}};const evidenceLabel=document.createElement('label');evidenceLabel.textContent='Bằng chứng ngữ cảnh ';evidenceLabel.append(evidence);section.append(evidenceLabel);
const reason=document.createElement('textarea');reason.placeholder='Mô tả điều bạn nhìn thấy; nếu không đủ bằng chứng hãy giữ Chưa rõ.';reason.value=a.rationale;reason.oninput=()=>a.rationale=reason.value;const reasonLabel=document.createElement('label');reasonLabel.textContent='Lý do ';reasonLabel.append(reason);section.append(reasonLabel);$('actors').append(section);}progress();}
function message(text,error=false){$('message').textContent=text;$('message').className=error?'error':'success';}
function validateForDownload(){let changed=false;for(const c of labels.cases)for(const a of c.actors){if(a.label!=='unknown')changed=true;if(a.label==='confirmed_negative'&&(!a.evidence_context_indices.length||!a.rationale.trim()))throw Error('Người #'+a.track_id+' trong '+c.case_id+' cần bằng chứng ngữ cảnh và lý do, hoặc giữ Chưa rõ.');}if(changed&&(!$('humanAssertion').checked||!$('reviewer').value.trim()))throw Error('Điền mã người kiểm tra và xác nhận tự xem trước khi lưu nhãn có kết luận.');labels.reviewer_id=$('reviewer').value.trim();const reviewed=$('humanAssertion').checked;if(reviewed&&!labels.reviewer_id)throw Error('Điền mã người kiểm tra.');labels.assertions={human_reviewed:reviewed,not_ai_generated_ground_truth:reviewed,model_scores_hidden:true};labels.reviewed_at_utc=reviewed?new Date().toISOString():null;}
$('download').onclick=()=>{try{validateForDownload();const blob=new Blob([JSON.stringify(labels,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='vidhoi_pair_review_annotations.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message('Đã tạo tệp nhãn. Chưa rõ vẫn được giữ nguyên.');}catch(e){message(e.message,true);}};
$('importLabels').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const incoming=JSON.parse(await file.text());if(incoming.schema_version!==labels.schema_version||incoming.manifest_fingerprint!==labels.manifest_fingerprint)throw Error('Tệp nhãn thuộc bộ dữ liệu khác.');if(!Array.isArray(incoming.cases)||incoming.cases.length!==data.cases.length)throw Error('Tệp thiếu tình huống.');const ids=new Set();for(const c of incoming.cases){const expected=data.cases.find(x=>x.case_id===c.case_id);if(!expected||ids.has(c.case_id)||!Array.isArray(c.actors)||c.actors.length!==expected.actors.length)throw Error('Tệp có mã tình huống sai hoặc trùng.');ids.add(c.case_id);const tracks=new Set();for(const a of c.actors){if(tracks.has(a.track_id)||!expected.actors.some(x=>x.track_id===a.track_id)||!['positive','confirmed_negative','unknown'].includes(a.label)||typeof a.rationale!=='string'||!Array.isArray(a.evidence_context_indices)||a.evidence_context_indices.some(j=>!Number.isInteger(j)||j<0||j>=expected.context_frames.length))throw Error('Nhãn người hoặc bằng chứng không hợp lệ.');tracks.add(a.track_id);}}labels=incoming;$('reviewer').value=incoming.reviewer_id||'';$('humanAssertion').checked=incoming.assertions?.human_reviewed===true;render();message('Đã nạp nhãn đã lưu. Bộ chạy Kaggle sẽ kiểm tra đầy đủ provenance và bằng chứng.');}catch(e){message(e.message,true);}};
for(const [i,c] of data.cases.entries()){const option=document.createElement('option');option.value=i;option.textContent=(i+1)+' · '+c.predicate_name+' / '+c.object_name;$('caseSelect').append(option);}
$('caseSelect').onchange=e=>{caseIndex=Number(e.target.value);render();};$('previous').onclick=()=>{if(caseIndex>0){caseIndex--;render();}};$('next').onclick=()=>{if(caseIndex+1<data.cases.length){caseIndex++;render();}};$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Tạm dừng':'Phát tiếp';};$('contextSeek').oninput=e=>{contextIndex=Number(e.target.value);playing=false;$('play').textContent='Phát tiếp';drawContext();};setInterval(()=>{if(playing){contextIndex=(contextIndex+1)%current().context_frames.length;drawContext();}},250);render();
</script></body></html>'''

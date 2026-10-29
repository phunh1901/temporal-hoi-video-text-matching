"""Audited, video-disjoint frozen CLIP features for the internal M1 head study.

An absent predicate annotation is unknown. ``positive_mask`` records only observed
positives; its False entries never certify negatives. Images use original pixel
coordinates, integer outward crop bounds, and the unchanged original CLIP encoder.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from temporal_hoi.data.vidhoi import ClassMapping, HOIPair
from temporal_hoi.inference.hoi_static import (
    MAPPING_SHA256,
    PROMPT_VERSION,
    OracleEntity,
    OracleFrame,
    OriginalCLIPEncoder,
    _finite,
    _integer,
    _normalized,
    _peak_rss_bytes,
    _track,
    _union,
    build_prompts,
    checked_file,
    file_sha256,
    proposal_pairs,
)

INPUTS = ("train_manifest", "dev_manifest", "protocol", "mapping", "checkpoint")
ARRAY_KEYS = ("object_features", "union_features", "object_class", "positive_mask",
              "frame_index", "subject_track_id", "object_track_id")
LABEL_POLICY = "positive_only_unknown"


def config_fingerprint(value: dict) -> str:
    """Canonical SHA256 of the complete resolved runtime configuration."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _record(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
            "bytes": path.stat().st_size}


def _inventory(directory: Path) -> list[dict]:
    return [_record(path, directory) for path in sorted(directory.rglob("*")) if path.is_file()]


def _triplets(values: Any, mapping: ClassMapping, label: str) -> set[tuple[int, int, int]]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be an explicit list")
    output = set()
    for value in values:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{label} requires [person,predicate,object] triplets")
        subject, predicate, obj = value
        for number in value:
            _integer(number, label)
        if subject != 0 or predicate not in mapping.predicates or obj not in mapping.objects:
            raise ValueError(f"Unknown triplet in {label}")
        if tuple(value) in output:
            raise ValueError(f"Duplicate triplet in {label}")
        output.add(tuple(value))
    return output


def _videos(values: Any, label: str) -> set[str]:
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} must be a nonempty unique video ID list")
    return set(values)


def _load_role(path: Path, role: str, config: dict, mapping: ClassMapping) -> list[OracleFrame]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version": "vidhoi_head_frames_v1", "upstream_split": "train",
                "role": role, "box_source": "oracle", "object_label_source": "oracle",
                "mapping_sha256": config["mapping_sha256"],
                "protocol_sha256": config["protocol_sha256"]}
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Invalid {role} manifest schema/scope/hash")
    rows = data.get("frames")
    if not isinstance(rows, list) or not rows or len(rows) > config.get("max_frames", 512):
        raise ValueError(f"Nonempty bounded frames required for {role}; no silent truncation")
    frames, keys, sources = [], set(), set()
    for row in rows:
        video = row.get("video_id")
        if not isinstance(video, str) or not video.strip():
            raise ValueError("video_id must be a nonempty string")
        source_id = _integer(row.get("source_frame_id"), "source_frame_id", 1)
        frame_id = _integer(row.get("frame_id"), "frame_id")
        if frame_id != source_id - 1:
            raise ValueError("Frame join must preserve zero-based media index = source ID - 1")
        width, height = (_integer(row.get(name), name, 1) for name in ("width", "height"))
        if "timestamp_s" not in row:
            raise ValueError("timestamp_s must be explicit; null means unknown")
        timestamp = row["timestamp_s"]
        if timestamp is not None and _finite(timestamp, "timestamp_s") < 0:
            raise ValueError("Negative timestamp_s")
        raw_path = row.get("image_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("image_path must be a relative path")
        image_path = (path.parent / raw_path).resolve()
        if (Path(raw_path).is_absolute() or image_path.is_symlink()
                or not image_path.is_relative_to(path.parent.resolve())):
            raise ValueError("Frame image path escapes its manifest folder")
        checked_file(image_path, row.get("image_sha256"), "frame image")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError("Image dimensions differ from manifest")
            image.verify()
        if not isinstance(row.get("entities"), list):
            raise ValueError("entities must be an explicit list")
        entities = {}
        for item in row["entities"]:
            track = _track(item.get("track_id"))
            class_id = _integer(item.get("class_id"), "class_id")
            if track in entities or class_id not in mapping.objects:
                raise ValueError("Duplicate track or unknown object class")
            box = item.get("bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("bbox_xyxy needs four original-pixel coordinates")
            box = tuple(_finite(value, "bbox_xyxy") for value in box)
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError("Invalid original-pixel bounding box")
            entities[track] = OracleEntity(track, class_id, box)
        if not isinstance(row.get("ground_truth"), list):
            raise ValueError("ground_truth must be an explicit list")
        labels = {}
        for item in row["ground_truth"]:
            sub, obj = (_track(item.get(name)) for name in
                        ("subject_track_id", "object_track_id"))
            if sub == obj or sub not in entities or obj not in entities or entities[sub].class_id != 0:
                raise ValueError("GT must link distinct existing person/object tracks")
            predicates = item.get("predicate_ids")
            if not isinstance(predicates, list) or not predicates:
                raise ValueError("predicate_ids must contain observed positives")
            for predicate in predicates:
                _integer(predicate, "predicate_id")
                if predicate not in mapping.predicates:
                    raise ValueError("Unknown predicate_id")
            labels.setdefault((sub, obj), set()).update(predicates)
        truth = tuple(HOIPair(0, entities[obj].class_id, entities[sub].bbox_xyxy,
                              entities[obj].bbox_xyxy,
                              tuple(float(index in predicates)
                                    for index in range(len(mapping.predicates))),
                              subject_track_id=sub, object_track_id=obj)
                      for (sub, obj), predicates in sorted(labels.items()))
        frame = OracleFrame(video, source_id, frame_id, timestamp, image_path,
                            row["image_sha256"], width, height, tuple(entities.values()), truth)
        if frame.key in keys or (video, source_id) in sources:
            raise ValueError("Duplicate frame identity")
        keys.add(frame.key)
        sources.add((video, source_id))
        if len(proposal_pairs(frame)) > config.get("max_pairs_per_frame", 128):
            raise ValueError("Frame exceeds pair budget; no pair was dropped")
        frames.append(frame)
    return frames


def preflight_head_data(config: dict, *, test_fixture: bool = False) -> dict:
    """Validate byte identity, images, labels, full-video holdouts and role isolation."""
    if config.get("schema_version") != "vidhoi_head_study_config_v1":
        raise ValueError("Require vidhoi_head_study_config_v1")
    if config.get("test_fixture", False) and not test_fixture:
        raise ValueError("Fixture config requires explicit fixture execution")
    for name, default in (("batch_size", 8), ("threads", 2), ("max_frames", 512),
                          ("max_pairs_per_frame", 128)):
        _integer(config.get(name, default), name, 1)
    if config.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.get("device", "cpu") == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no fallback")
    paths = {name: checked_file(config[name], config.get(f"{name}_sha256"), name)
             for name in INPUTS}
    if paths["checkpoint"].stat().st_size > config.get("max_checkpoint_bytes", 1024**3):
        raise ValueError("Checkpoint exceeds declared byte budget")
    provenance = config.get("checkpoint_provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("checkpoint_provenance must be explicit")
    if provenance.get("checkpoint_sha256", config["checkpoint_sha256"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint provenance digest mismatch")
    mapping = ClassMapping.from_json(paths["mapping"])
    if not test_fixture and (config["mapping_sha256"] != MAPPING_SHA256
                             or len(mapping.objects) != 78 or len(mapping.predicates) != 50):
        raise ValueError("Real VidHOI cache requires pinned 78-object/50-predicate mapping")
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    if (protocol.get("schema_version") != "vidhoi_head_protocol_v1"
            or protocol.get("label_policy") != LABEL_POLICY):
        raise ValueError("Invalid protocol schema or unknown-label policy")
    for name in ("official_final_split_accessed", "final_unseen_accessed"):
        if protocol.get(name, False) is not False:
            raise ValueError("Final evaluation data cannot enter development")
    train_ids = _videos(protocol.get("train_video_ids"), "train_video_ids")
    dev_ids = _videos(protocol.get("dev_video_ids"), "dev_video_ids")
    if train_ids & dev_ids:
        raise ValueError("Train and development videos overlap")
    excluded = protocol.get("excluded_final_video_ids")
    if not isinstance(excluded, list) or len(set(excluded)) != len(excluded):
        raise ValueError("excluded_final_video_ids must be an explicit unique list")
    if (train_ids | dev_ids) & set(excluded):
        raise ValueError("Final held-out video entered development")
    seen = _triplets(protocol.get("seen_triplets"), mapping, "seen_triplets")
    pseudo = _triplets(protocol.get("pseudo_unseen_triplets"), mapping, "pseudo_unseen_triplets")
    final = _triplets(protocol.get("final_unseen_triplets"), mapping, "final_unseen_triplets")
    if (seen & pseudo) or (seen & final) or (pseudo & final):
        raise ValueError("Seen/pseudo/final composition sets overlap")
    full = protocol.get("video_positive_triplets")
    if not isinstance(full, dict) or not (train_ids | dev_ids).issubset(full):
        raise ValueError("Full-video positive metadata required, not sampled-frame labels only")
    full = {video: _triplets(full[video], mapping, f"video_positive_triplets/{video}")
            for video in train_ids | dev_ids}
    for video, triplets in full.items():
        if triplets & final or (video in train_ids and triplets & pseudo):
            raise ValueError("Held-out composition appears in a selected full-video context")
        if video in train_ids and not triplets.issubset(seen):
            raise ValueError("Training full-video composition is absent from seen set")
    prior = protocol.get("prior_inspected_video_ids_always_train", [])
    if not isinstance(prior, list) or not set(prior).issubset(train_ids):
        raise ValueError("Previously inspected videos must stay in train")
    train = _load_role(paths["train_manifest"], "train", config, mapping)
    dev = _load_role(paths["dev_manifest"], "validation", config, mapping)
    if {frame.image_sha256 for frame in train} & {frame.image_sha256 for frame in dev}:
        raise ValueError("Identical image bytes appear in both train and development")
    sampled = {}
    for role, frames, wanted in (("train", train, train_ids), ("validation", dev, dev_ids)):
        if {frame.video_id for frame in frames} != wanted:
            raise ValueError(f"{role} manifest videos differ from locked protocol")
        observed = set()
        for frame in frames:
            for pair in frame.ground_truth:
                for predicate, present in enumerate(pair.predicate_values):
                    if present and (0, predicate, pair.object_class) not in full[frame.video_id]:
                        raise ValueError("Sampled GT is absent from full-video metadata")
                    if present:
                        observed.add((0, predicate, pair.object_class))
        if not observed:
            raise ValueError(f"{role} requires observed positives; no fabricated metrics")
        sampled[role] = observed
    predicates = {value[1] for value in sampled["train"]}
    objects = {value[2] for value in sampled["train"]}
    for label, triplets in (("pseudo-unseen", pseudo), ("final-unseen", final
                            if protocol.get("final_heldout_type") == "composition" else set())):
        if any(value[1] not in predicates or value[2] not in objects for value in triplets):
            raise ValueError(f"Designed {label} composition has an unseen sampled-train component")
    audit = {"sampled_train_triplets": [list(value) for value in sorted(sampled["train"])],
             "sampled_train_predicate_ids": sorted(predicates), "sampled_train_object_ids": sorted(objects),
             "sampled_dev_triplets": [list(value) for value in sorted(sampled["validation"])],
             "sampled_dev_seen_triplets": [list(value) for value in
                                          sorted(sampled["validation"] & sampled["train"])],
             "sampled_dev_designed_pseudo_triplets": [list(value) for value in
                                                      sorted(sampled["validation"] & pseudo)],
             "sampled_dev_incidental_unseen_triplets": [list(value) for value in
                sorted(sampled["validation"] - sampled["train"] - pseudo)],
             "designed_pseudo_components_seen_in_sampled_train": True,
             "final_component_audit_requested": protocol.get("final_heldout_type") == "composition",
             "duplicate_image_bytes_across_roles": 0}
    return {"mapping": mapping, "train_frames": train, "dev_frames": dev,
            "protocol": protocol, "input_hashes": {name: config[f"{name}_sha256"] for name in INPUTS},
            "config_sha256": config_fingerprint(config), "audit": audit}


def _frame_record(frame: OracleFrame) -> dict:
    record = asdict(frame)
    record["image_path"] = str(frame.image_path)
    return record


def _restore_frame(value: dict) -> OracleFrame:
    record = dict(value)
    record["image_path"] = Path(record["image_path"])
    record["entities"] = tuple(OracleEntity(item["track_id"], item["class_id"],
                                             tuple(item["bbox_xyxy"])) for item in record["entities"])
    record["ground_truth"] = tuple(HOIPair(**{**item,
        "subject_box": tuple(item["subject_box"]), "object_box": tuple(item["object_box"]),
        "predicate_values": tuple(item["predicate_values"])}) for item in record["ground_truth"])
    return OracleFrame(**record)


def _role_features(frames: list[OracleFrame], encoder: Any, batch: int, dimension: int,
                   predicates: int) -> dict:
    rows = {name: [] for name in ARRAY_KEYS}
    for index, frame in enumerate(frames):
        pairs = proposal_pairs(frame)
        object_by_track = {}
        with Image.open(frame.image_path) as original:
            image = original.convert("RGB")
            targets = sorted({obj.track_id: obj for _, obj in pairs}.values(),
                             key=lambda entity: entity.track_id)
            for offset in range(0, len(targets), batch):
                current = targets[offset:offset + batch]
                crops = [image.crop(tuple(math.floor(value) if axis < 2 else math.ceil(value)
                                          for axis, value in enumerate(obj.bbox_xyxy))) for obj in current]
                features = _normalized(encoder.encode_images(crops), len(current), "object image")
                for obj, feature in zip(current, features, strict=True):
                    object_by_track[obj.track_id] = feature
            known = {(pair.subject_track_id, pair.object_track_id): pair.predicate_values
                     for pair in frame.ground_truth}
            for offset in range(0, len(pairs), batch):
                current = pairs[offset:offset + batch]
                crops = [image.crop(_union(subject, obj)) for subject, obj in current]
                union = _normalized(encoder.encode_images(crops), len(current), "union image")
                if union.shape[1] != dimension:
                    raise ValueError("Image and text dimensions disagree")
                for (subject, obj), feature in zip(current, union, strict=True):
                    rows["object_features"].append(object_by_track[obj.track_id])
                    rows["union_features"].append(feature)
                    rows["object_class"].append(obj.class_id)
                    rows["positive_mask"].append(known.get((subject.track_id, obj.track_id),
                                                          (0,) * predicates))
                    rows["frame_index"].append(index)
                    rows["subject_track_id"].append(subject.track_id)
                    rows["object_track_id"].append(obj.track_id)
        if index % 16 == 0:
            print(f"FEATURE_CACHE_FRAME: {index + 1}/{len(frames)} {frame.key}", flush=True)
    return rows


def cache_head_features(config: dict, output_dir: str | Path, *, encoder: Any = None) -> dict:
    """Encode real inputs once; injected encoders must explicitly declare fixture status."""
    fixture = encoder is not None
    if fixture and encoder.metadata.get("test_fixture") is not True:
        raise ValueError("Injected encoder must declare test_fixture=True")
    checked = preflight_head_data(config, test_fixture=fixture)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Cache output already exists; preserve evidence and choose a new run")
    started = time.perf_counter()
    torch.set_num_threads(config.get("threads", 2))
    if encoder is None:
        encoder = OriginalCLIPEncoder(Path(config["checkpoint"]), device=config.get("device", "cpu"))
    output.mkdir(parents=True)
    mapping, batch = checked["mapping"], config.get("batch_size", 8)
    final = {tuple(value) for value in checked["protocol"]["final_unseen_triplets"]}
    candidate = np.ones((len(mapping.objects), len(mapping.predicates)), dtype=bool)
    for _, predicate, obj in final:
        candidate[obj, predicate] = False
    dimension, encoded, prompts = None, {}, {}
    for obj in mapping.objects:
        names = build_prompts(mapping, obj)
        indices = np.flatnonzero(candidate[obj])
        if not len(indices):
            raise ValueError("At least one allowed text candidate per object is required")
        selected = [names[index] for index in indices]
        chunks = [_normalized(encoder.encode_texts(selected[offset:offset + batch]),
                              len(selected[offset:offset + batch]), "text")
                  for offset in range(0, len(selected), batch)]
        vectors = np.concatenate(chunks)
        if dimension is not None and vectors.shape[1] != dimension:
            raise ValueError("Text encoder dimension changed")
        dimension = vectors.shape[1]
        encoded[obj] = (indices, vectors)
        prompts[str(obj)] = {str(index): names[index] for index in indices.tolist()}
    texts = np.zeros((*candidate.shape, dimension), dtype=np.float32)
    for obj, (indices, vectors) in encoded.items():
        texts[obj, indices] = vectors
    np.savez_compressed(output / "text_features.npz", text_features=texts, candidate_mask=candidate)
    _write(output / "prompts.json", prompts)
    roles = {}
    level = "synthetic_encoder_fixture" if fixture else "vidhoi_head_frozen_clip_feature_cache"
    for role, frames in (("train", checked["train_frames"]), ("validation", checked["dev_frames"])):
        directory = output / role
        directory.mkdir()
        rows = _role_features(frames, encoder, batch, dimension, len(mapping.predicates))
        count = len(rows["object_class"])
        if not count:
            raise ValueError("Each role requires at least one proposal; zero-pair frames remain recorded")
        # A frame with no observed relation still has all-unknown proposal labels.
        positive = np.asarray(rows["positive_mask"], dtype=bool)
        arrays = {"object_features": np.asarray(rows["object_features"], dtype=np.float32),
                  "union_features": np.asarray(rows["union_features"], dtype=np.float32),
                  "positive_mask": positive,
                  **{name: np.asarray(rows[name], dtype=np.int64) for name in ("object_class", "frame_index")},
                  **{name: np.asarray(rows[name], dtype=str) for name in
                     ("subject_track_id", "object_track_id")}}
        if arrays["object_features"].shape != (count, dimension):
            raise ValueError("Object image and text dimensions disagree")
        np.savez_compressed(directory / "arrays.npz", **arrays)
        metadata = {"role": role, "source_split": "train", "level": level, "test_fixture": fixture,
                    "config_sha256": checked["config_sha256"], "input_hashes": checked["input_hashes"],
                    **{f"{name}_sha256": config[f"{name}_sha256"]
                       for name in ("protocol", "mapping", "checkpoint")},
                    "label_policy": LABEL_POLICY, "final_unseen_accessed": False,
                    "official_final_split_accessed": False, "pseudo_unseen_in_train": False}
        _write(directory / "frames.json", {"mapping": {"objects": dict(mapping.objects),
                                                        "predicates": dict(mapping.predicates)},
                                            "frames": [_frame_record(frame) for frame in frames],
                                            "metadata": metadata})
        roles[role] = {"num_frames": len(frames), "num_pairs": count}
        _write(directory / "receipt.json", {"schema_version": "vidhoi_head_role_cache_v1",
               "status": "completed", **metadata, **roles[role], "files": _inventory(directory)})
    receipt = {"schema_version": "vidhoi_head_cache_receipt_v1", "status": "completed",
               "level": level, "source_split": "train", "model_inference_run": not fixture,
               "model_training_run": False, "official_final_split_accessed": False,
               "final_unseen_accessed": False, "input_hashes": checked["input_hashes"],
               "config_sha256": checked["config_sha256"], "roles": roles,
               "data_audit": checked["audit"],
               "encoder": encoder.metadata, "prompt_version": PROMPT_VERSION,
               "prompt_sha256": file_sha256(output / "prompts.json"),
               "text_cache": {**_record(output / "text_features.npz", output), "shape": list(texts.shape),
                              "final_unseen_rows_zero": True, "final_unseen_encoded": False},
               "resources": {"total_seconds": time.perf_counter() - started,
                             "process_peak_rss_bytes": _peak_rss_bytes()},
               "files": _inventory(output)}
    _write(output / "receipt.json", receipt)
    return receipt


def load_head_cache(role_dir: str | Path) -> dict:
    """Read a verified numeric cache without reopening or requiring source images."""
    directory = Path(role_dir)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("schema_version") != "vidhoi_head_role_cache_v1"
            or receipt.get("status") != "completed" or receipt.get("role") not in {"train", "validation"}):
        raise ValueError("Incomplete or wrong-role feature cache receipt")
    records = receipt.get("files")
    if not isinstance(records, list) or {row.get("path") for row in records} != {"arrays.npz", "frames.json"}:
        raise ValueError("Role cache inventory must contain arrays.npz and frames.json")
    for row in records:
        target = directory / row["path"]
        checked_file(target, row.get("sha256"), "cache inventory")
        if target.stat().st_size != row.get("bytes"):
            raise ValueError("Cache file byte size mismatch")
    data = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
    metadata = data["metadata"]
    for key in ("role", "config_sha256", "protocol_sha256", "checkpoint_sha256", "mapping_sha256",
                "input_hashes", "level", "test_fixture", "label_policy"):
        if metadata.get(key) != receipt.get(key):
            raise ValueError(f"Cache metadata differs from receipt: {key}")
    if (metadata.get("label_policy") != LABEL_POLICY or metadata.get("source_split") != "train"
            or any(metadata.get(name) is not False for name in
                   ("official_final_split_accessed", "final_unseen_accessed", "pseudo_unseen_in_train"))
            or metadata.get("level") != ("synthetic_encoder_fixture" if metadata.get("test_fixture")
                                         else "vidhoi_head_frozen_clip_feature_cache")):
        raise ValueError("Cache scope/label policy/fixture evidence mismatch")
    mapping = ClassMapping({int(key): value for key, value in data["mapping"]["objects"].items()},
                           {int(key): value for key, value in data["mapping"]["predicates"].items()})
    frames = [_restore_frame(value) for value in data["frames"]]
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_KEYS):
            raise ValueError("Wrong cache array schema")
        arrays = {name: archive[name].copy() for name in ARRAY_KEYS}
    count = receipt["num_pairs"]
    if (len(frames) != receipt["num_frames"] or count != sum(len(proposal_pairs(frame)) for frame in frames)
            or arrays["positive_mask"].shape != (count, len(mapping.predicates))
            or arrays["positive_mask"].dtype != bool):
        raise ValueError("Cache canonical frame/pair/label coverage mismatch")
    for name in ("object_features", "union_features"):
        value = arrays[name]
        if (value.ndim != 2 or value.shape[0] != count or value.dtype != np.float32
                or not np.isfinite(value).all()
                or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=1e-5)):
            raise ValueError("Cache features must be finite normalized FP32 [N,D]")
    if arrays["object_features"].shape != arrays["union_features"].shape:
        raise ValueError("Object and union cache shapes differ")
    for name in ("subject_track_id", "object_track_id"):
        if arrays[name].shape != (count,) or arrays[name].dtype.kind != "U":
            raise ValueError("Track IDs must be plain unicode vectors")
    for name in ("object_class", "frame_index"):
        if arrays[name].shape != (count,) or arrays[name].dtype.kind not in "iu":
            raise ValueError("Class/frame index must be plain integer vectors")
    cursor = 0
    for frame_index, frame in enumerate(frames):
        known = {(pair.subject_track_id, pair.object_track_id): pair.predicate_values
                 for pair in frame.ground_truth}
        for subject, obj in proposal_pairs(frame):
            pair_key = (subject.track_id, obj.track_id)
            expected = np.asarray(known.get(pair_key, (0,) * len(mapping.predicates)), dtype=bool)
            if (arrays["frame_index"][cursor] != frame_index
                    or arrays["object_class"][cursor] != obj.class_id
                    or arrays["subject_track_id"][cursor] != subject.track_id
                    or arrays["object_track_id"][cursor] != obj.track_id
                    or not np.array_equal(arrays["positive_mask"][cursor], expected)):
                raise ValueError("Cache labels or ordered proposals differ from canonical frames")
            cursor += 1
    return {**arrays, "frames": frames, "mapping": mapping, "metadata": metadata}

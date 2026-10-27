"""Materialize VidHOI annotation, identity, partition and frame-mask locks.

This only reads local inputs. Missing inputs produce an explicit blocked
artifact; a lock never proves model inference or a zero-shot benchmark run.
Annotation arrays are read incrementally to avoid loading the 493 MB train
JSON and its expanded Python representation into memory simultaneously.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Mapping

from temporal_hoi.data.vidhoi import (
    BoxTransform,
    ClassMapping,
    adapt_vidhoi_annotations,
    identity_hash,
    validate_video_partitions,
)

CANONICAL_MAPPING_SHA256 = "a74c9c6c0e1efa116eb368935aded8b5a124c3d758376391924db8e2a58368d7"
REQUIRED_INPUTS = (
    "mapping", "train_annotations", "val_annotations", "train_frame_manifest",
    "val_frame_manifest", "video_partitions", "frame_mask",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_json_array(path: Path, chunk_size: int = 65536) -> Iterator[dict]:
    """Read a JSON object array incrementally; reject truncation/trailing data."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer, position, eof = "", 0, False

        def refill() -> None:
            nonlocal buffer, position, eof
            chunk = stream.read(chunk_size)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk

        def next_character() -> str:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    return ""
                refill()

        if next_character() != "[":
            raise ValueError(f"Annotation must be a JSON array: {path}")
        position += 1
        first = True
        while True:
            character = next_character()
            if character == "]":
                position += 1
                if next_character():
                    raise ValueError("Unexpected data after annotation array")
                return
            if not first:
                if character != ",":
                    raise ValueError("Missing separator or truncated annotation array")
                position += 1
                if next_character() == "]":
                    raise ValueError("Trailing comma in annotation array")
            first = False
            while True:
                try:
                    row, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError("Invalid or truncated annotation row") from error
                    refill()
            if not isinstance(row, dict):
                raise ValueError("Every annotation row must be an object")
            yield row


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_annotation_split(
    path: Path, manifest: list[dict], mapping: ClassMapping, *, split: str,
) -> dict:
    """Validate every frame/track join and count triplets on unique GT pairs."""
    manifest_by_source = {}
    media_keys = []
    result_keys = []
    for info in manifest:
        key = (info["video_id"], info["source_frame_id"])
        if key in manifest_by_source:
            raise ValueError("Duplicate source frame in annotation manifest")
        if not isinstance(info.get("result_frame_key"), str) or not info["result_frame_key"]:
            raise ValueError("Manifest requires a verified explicit result_frame_key")
        if not isinstance(info.get("box_transform"), dict):
            raise ValueError("Manifest requires an explicit box_transform from upstream preprocessing")
        if set(info["box_transform"]) != {"scale_x", "scale_y", "offset_x", "offset_y"}:
            raise ValueError("Explicit box_transform requires both scales and both offsets")
        provenance = info.get("box_transform_provenance")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("Manifest requires box_transform_provenance; identity is not assumed")
        BoxTransform(**info["box_transform"])
        manifest_by_source[key] = info
        media_keys.append(f"{info['video_id']}_{info['frame_id']:06d}")
        result_keys.append(info["result_frame_key"])
    if len(set(media_keys)) != len(media_keys) or len(set(result_keys)) != len(result_keys):
        raise ValueError("Manifest media/result frame identities must be unique")
    seen = set()
    counts = Counter()
    video_support: dict[tuple, set] = {}
    rows_count = pairs_count = 0
    group = []
    active_key = None

    def consume(key, rows) -> None:
        nonlocal pairs_count
        if key in seen:
            raise ValueError("Annotations must group each frame contiguously")
        if key not in manifest_by_source:
            raise ValueError("Annotation frame is missing from the explicit manifest")
        seen.add(key)
        frame = adapt_vidhoi_annotations(
            rows, mapping, [manifest_by_source[key]], split=split,
        )[0]
        pairs_count += len(frame.ground_truth)
        for pair in frame.ground_truth:
            for predicate, value in enumerate(pair.predicate_values):
                if value == 1:
                    triplet = (0, predicate, pair.object_class)
                    counts[triplet] += 1
                    video_support.setdefault(triplet, set()).add(frame.video_id)

    for row in iter_json_array(path):
        key = (f"{row['video_folder']}/{row['video_id']}", row["frame_id"])
        if active_key is not None and key != active_key:
            consume(active_key, group)
            group = []
        active_key = key
        group.append(row)
        rows_count += 1
    if active_key is not None:
        consume(active_key, group)
    if not seen or seen != set(manifest_by_source):
        raise ValueError("Manifest must identify every annotation frame exactly once")
    videos = sorted({key[0] for key in seen})
    support = [
        {"triplet": list(key), "predicate_name": mapping.predicates[key[1]],
         "object_name": mapping.objects[key[2]], "gt_positive_pair_frames": count,
         "videos": len(video_support[key]),
         "rare_on_this_split": count < 25}
        for key, count in sorted(counts.items())
    ]
    return {
        "annotation_rows": rows_count, "frames": len(seen), "gt_pairs": pairs_count,
        "video_ids": videos, "video_ids_sha256": identity_hash(videos),
        "ordered_media_frame_ids": media_keys,
        "ordered_media_frame_ids_sha256": identity_hash(media_keys),
        "ordered_result_frame_keys_sha256": identity_hash(result_keys),
        "triplet_support": support,
        "triplet_support_sha256": hashlib.sha256(json.dumps(
            support, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
    }


def lock_vidhoi_inputs(spec: Mapping, *, base_dir: Path) -> dict:
    """Build a reproducible input lock, or list the exact missing local inputs."""
    protocol_id = spec.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("protocol_id is required")
    declared = spec.get("inputs", {})
    files, missing, paths = {}, [], {}
    for name in REQUIRED_INPUTS:
        entry = declared.get(name, {})
        raw_path = entry.get("path")
        path = base_dir / raw_path if raw_path else None
        if path is None or not path.is_file():
            missing.append(name)
            files[name] = {"path": raw_path, "present": False, "sha256": None}
            continue
        digest = file_sha256(path)
        if entry.get("sha256") and entry["sha256"] != digest:
            raise ValueError(f"Declared {name} hash mismatch")
        paths[name] = path
        files[name] = {"path": raw_path, "present": True, "bytes": path.stat().st_size,
                       "sha256": digest, "source": entry.get("source")}
    output = {
        "schema_version": 1, "protocol_id": protocol_id,
        "status": "blocked_missing_local_inputs" if missing else "validating_local_inputs",
        "files": files, "missing_inputs": missing, "input_lock_ready": False,
        "benchmark_evaluated": False, "benchmark_map": None,
        "model_inference_run": False, "zero_shot_split_materialized": False,
        "official_final_split_annotations_accessed": "val_annotations" in paths and not missing,
        "official_final_split_scores_accessed": False,
        "upstream_commit": spec.get("upstream_commit"),
        "scope": "local_input_lock_only_no_download_no_model_or_prediction_execution",
    }
    if missing:
        return output
    mapping = ClassMapping.from_json(paths["mapping"])
    if (files["mapping"]["sha256"] != CANONICAL_MAPPING_SHA256
            or len(mapping.objects) != 78 or len(mapping.predicates) != 50):
        raise ValueError("VidHOI requires the canonical 78/50 numeric mapping")
    partitions = _read_json(paths["video_partitions"])
    required_partitions = {"research_train", "research_validation", "official_val"}
    if set(partitions) != required_partitions or any(not part for part in partitions.values()):
        raise ValueError("Nonempty research_train/research_validation/official_val partitions required")
    validate_video_partitions(partitions)
    summaries = {
        name: summarize_annotation_split(
            paths[f"{prefix}_annotations"], _read_json(paths[f"{prefix}_frame_manifest"]),
            mapping, split=name,
        ) for prefix, name in (("train", "train"), ("val", "official_val"))
    }
    if set(summaries["train"]["video_ids"]) != (
        set(partitions["research_train"]) | set(partitions["research_validation"])
    ) or set(summaries["official_val"]["video_ids"]) != set(partitions["official_val"]):
        raise ValueError("Video partitions must cover upstream annotation videos exactly")
    mask = _read_json(paths["frame_mask"])
    ids = mask.get("frame_ids", [])
    expected = spec.get("expected_mask_frame_count")
    if (not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0
            or len(ids) != expected or len(set(ids)) != len(ids)):
        raise ValueError("Frame mask must exactly match the declared expected unique frame count")
    if mask.get("protocol_id") != protocol_id:
        raise ValueError("Frame mask protocol_id differs from the input lock")
    if not set(ids) <= set(summaries["official_val"]["ordered_media_frame_ids"]):
        raise ValueError("Frame mask contains IDs without official_val annotation")
    output.update({
        "status": "local_inputs_locked_external_provenance_audit_required",
        "input_lock_ready": True, "annotation_splits": summaries,
        "partitions": {key: {"videos": len(value), "video_ids_sha256": identity_hash(value)}
                       for key, value in partitions.items()},
        "frame_mask": {"frames": len(ids), "ordered_ids_sha256": identity_hash(ids),
                       "source": mask.get("source"),
                       "source_provenance": "provided_export_requires_independent_source_audit"},
        "benchmark_comparison_eligible": False,
        "limits": ["Local hash/identity validation cannot certify upstream completeness or media rights",
                   "Box transforms and mask source must be independently audited before scoring",
                   "Triplet support is downstream label accounting, not a seen/unseen split"],
    })
    return output

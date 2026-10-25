"""Validated person/object union crops in original-frame pixel coordinates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(record, prefix):
    path = Path(record[f"{prefix}_path"])
    expected = record[f"{prefix}_sha256"]
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"Box provenance {prefix} file/hash cannot be verified")


def validate_provenance(provenance):
    source_type = provenance.get("source_type")
    if source_type not in {"oracle", "detector"}:
        raise ValueError("Box source must explicitly be oracle or detector")
    _verified_file(provenance, "artifact")
    verification = provenance.get("verification", {})
    if verification.get("status") != "verified" or not str(verification.get("verifier", "")).strip():
        raise ValueError("Boxes require an identified verifier and verified evidence")
    _verified_file(verification, "evidence")
    if source_type == "detector":
        if not str(provenance.get("detector_name", "")).strip():
            raise ValueError("Detector identity is required")
        _verified_file(provenance, "detector_checkpoint")
        _verified_file(provenance, "detector_config")
    return source_type


def _box(values, width, height):
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("A box must contain four xyxy pixel coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in values):
        raise ValueError("Box coordinates must be numeric")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Box coordinates must be finite")
    x1, y1, x2, y2 = values
    if not (x1 < x2 and y1 < y2):
        raise ValueError("Box must have positive source area")
    clipped = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
    if not (clipped[0] < clipped[2] and clipped[1] < clipped[3]):
        raise ValueError("Box has no positive area inside the original frame")
    return clipped


def union_bounds(record, original_size, decoded_size):
    """Scale xyxy from original image to decoded image, rounding outwards."""
    width, height = original_size
    dw, dh = decoded_size
    if min(width, height, dw, dh) <= 0:
        raise ValueError("Frame dimensions must be positive")
    if (record["width"], record["height"]) != (width, height):
        raise ValueError("Box dimensions disagree with original video metadata")
    if record.get("person_class") != "person" or not str(record.get("object_class", "")).strip():
        raise ValueError("Crop requires explicit person and object class identities")
    person = _box(record["person_box_xyxy"], width, height)
    obj = _box(record["object_box_xyxy"], width, height)
    union = [min(person[0], obj[0]), min(person[1], obj[1]),
             max(person[2], obj[2]), max(person[3], obj[3])]
    bounds = (math.floor(union[0] * dw / width), math.floor(union[1] * dh / height),
              math.ceil(union[2] * dw / width), math.ceil(union[3] * dh / height))
    return bounds


class VerifiedUnionBoxes:
    def __init__(self, path):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "person_object_union/v1":
            raise ValueError("Unsupported union-box schema")
        if payload.get("coordinates") != "original_frame_pixels_xyxy":
            raise ValueError("Coordinates must be original-frame xyxy pixels")
        self.source_type = validate_provenance(payload["provenance"])
        self.provenance = payload["provenance"]
        self.records, self.used = {}, set()
        self.full_frame_union_count = 0
        for record in payload["records"]:
            index = record["frame_index"]
            if type(index) is not int or index < 0:
                raise ValueError("Frame index must be a nonnegative integer")
            key = (record["clip_id"], index)
            if key in self.records:
                raise ValueError("Duplicate clip/frame boxes")
            self.records[key] = record

    def crop(self, frame, clip_id, timestamp, metadata, source_hash):
        index = round(timestamp * metadata["fps"])
        key = (clip_id, index)
        if key not in self.records:
            raise ValueError(f"Missing verified union boxes for {key}; no full-frame fallback")
        record = self.records[key]
        if record["source_video_sha256"] != source_hash:
            raise ValueError("Box source video hash differs from decoded video")
        recorded_time = record["timestamp_s"]
        if not math.isfinite(recorded_time) or abs(recorded_time - timestamp) > 1e-6:
            raise ValueError("Box timestamp does not match the sampled frame")
        bounds = union_bounds(record, (metadata["width"], metadata["height"]), frame.size)
        self.used.add(key)
        if bounds == (0, 0, frame.width, frame.height):
            self.full_frame_union_count += 1
        return frame.crop(bounds), bounds

    def assert_complete(self):
        if self.used != set(self.records):
            raise ValueError("Box manifest must cover all and only the sampled comparison frames")


def select_pair(pairs, width, height):
    """Select from valid detector pairs using only score and stable IDs/boxes.

    Detector adapters must already invert resize transforms. A missing valid
    pair is returned as None so the caller can lock an explicit shared cohort.
    """
    valid = []
    for pair in pairs:
        try:
            score = pair["pair_score"]
            if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
                continue
            if not str(pair["person_track_id"]).strip() or not str(pair["object_track_id"]).strip():
                continue
            record = {**pair, "width": width, "height": height, "person_class": "person"}
            union_bounds(record, (width, height), (width, height))
            valid.append(record)
        except (KeyError, ValueError, TypeError):
            continue
    if not valid:
        return None
    return min(valid, key=lambda p: (
        -p["pair_score"], str(p["person_track_id"]), str(p["object_track_id"]),
        p["object_class"], tuple(p["person_box_xyxy"]), tuple(p["object_box_xyxy"]),
    ))

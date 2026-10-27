"""Validated ST-HOI result adapter with explicit frame identity and box provenance.

Upstream boxes are ``[batch_index, x1, y1, x2, y2]`` in the resized model
frame, not necessarily original pixels. IDs and transformations must come
from a manifest when the result JSON omits them; row position is never an ID.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def identity_hash(values: Sequence[str]) -> str:
    """Hash the ordered identity list, preserving the evaluator's tie order."""
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(value) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


@dataclass(frozen=True)
class BoxTransform:
    """Forward image transform: model_x = original_x * scale_x + offset_x."""

    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self) -> None:
        for name in ("scale_x", "scale_y", "offset_x", "offset_y"):
            _number(getattr(self, name), name)
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("Box scales must be positive")

    def invert(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        return (
            (box[0] - self.offset_x) / self.scale_x,
            (box[1] - self.offset_y) / self.scale_y,
            (box[2] - self.offset_x) / self.scale_x,
            (box[3] - self.offset_y) / self.scale_y,
        )


@dataclass(frozen=True)
class HOIPair:
    subject_class: int
    object_class: int
    subject_box: tuple[float, float, float, float]
    object_box: tuple[float, float, float, float]
    predicate_values: tuple[float, ...]
    subject_score: float = 1.0
    object_score: float = 1.0
    subject_track_id: str | None = None
    object_track_id: str | None = None


@dataclass(frozen=True)
class VidHOIFrame:
    video_id: str
    frame_id: int
    split: str
    box_source: str
    coordinate_system: str
    ground_truth: tuple[HOIPair, ...]
    predictions: tuple[HOIPair, ...]
    timestamp_s: float | None = None

    @property
    def key(self) -> str:
        return f"{self.video_id}_{self.frame_id:06d}"


@dataclass(frozen=True)
class ClassMapping:
    objects: Mapping[int, str]
    predicates: Mapping[int, str]

    def __post_init__(self) -> None:
        for name, mapping in (("objects", self.objects), ("predicates", self.predicates)):
            if not mapping or len(set(mapping.values())) != len(mapping):
                raise ValueError(f"{name} names must be nonempty and unique")
            if set(mapping) != set(range(len(mapping))):
                raise ValueError(f"{name} IDs must be contiguous and zero-based")
            if any(not isinstance(value, str) or not value for value in mapping.values()):
                raise ValueError(f"{name} contains an invalid name")
        if self.objects.get(0) != "person":
            raise ValueError("ST-HOI requires person class ID 0")

    @classmethod
    def from_json(cls, path: str | Path) -> ClassMapping:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            {int(key): value for key, value in data["objects"].items()},
            {int(key): value for key, value in data["predicates"].items()},
        )


def _boxes(rows: Sequence, transform: BoxTransform | None) -> list[tuple]:
    result = []
    for row in rows:
        if len(row) != 5 or _integer(row[0], "box batch ID") != 0:
            raise ValueError("Expected [0,x1,y1,x2,y2] boxes from single-frame results")
        box = tuple(_number(value, "box coordinate") for value in row[1:])
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Boxes must have positive width and height")
        result.append(transform.invert(box) if transform else box)
    return result


def _classes(rows: Sequence, count: int, mapping: ClassMapping) -> list[int]:
    if len(rows) != count:
        raise ValueError("Box and class counts disagree")
    result = []
    for row in rows:
        if len(row) != 2 or _integer(row[0], "class batch ID") != 0:
            raise ValueError("Expected [0,class_id] class rows")
        value = _integer(row[1], "object class")
        if value not in mapping.objects:
            raise ValueError("Unknown object class ID")
        result.append(value)
    return result


def _pairs(indices, labels, boxes, classes, mapping, scores=None, tracks=None) -> tuple[HOIPair, ...]:
    if len(indices) != len(labels):
        raise ValueError("Pair and predicate row counts disagree")
    pairs = []
    for pair, values in zip(indices, labels, strict=True):
        if len(pair) != 2 or len(values) != len(mapping.predicates):
            raise ValueError("Pair must have two indices and every predicate value")
        sub, obj = (_integer(value, "pair index") for value in pair)
        if sub >= len(boxes) or obj >= len(boxes) or sub == obj:
            raise ValueError("Pair indices are invalid")
        if scores is None and classes[sub] != 0:
            raise ValueError("Ground-truth HOI subject must be person")
        numeric = tuple(_number(value, "predicate value") for value in values)
        if scores is None and any(value not in (0, 1) for value in numeric):
            raise ValueError("Ground-truth predicate labels must be binary")
        pairs.append(HOIPair(
            classes[sub], classes[obj], boxes[sub], boxes[obj], numeric,
            scores[sub] if scores is not None else 1.0,
            scores[obj] if scores is not None else 1.0,
            str(tracks[sub]) if tracks and sub in tracks else None,
            str(tracks[obj]) if tracks and obj in tracks else None,
        ))
    return tuple(pairs)


def adapt_sthoi_result(
    result: Mapping[str, Any],
    mapping: ClassMapping,
    *,
    video_id: str,
    frame_id: int,
    split: str,
    box_source: str,
    box_transform: BoxTransform | None = None,
    timestamp_s: float | None = None,
    gt_track_ids: Mapping[int, str] | None = None,
    proposal_track_ids: Mapping[int, str] | None = None,
) -> VidHOIFrame:
    """Convert one upstream JSON row; never infer missing frame/track identity."""
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("A nonempty upstream video_id is required")
    frame_id = _integer(frame_id, "frame_id")
    if split not in {"train", "research_train", "research_validation", "official_val", "fixture"}:
        raise ValueError("Unknown split; official val is final evaluation, not tuning validation")
    if box_source not in {"oracle", "predicted"}:
        raise ValueError("box_source must be oracle or predicted")
    if timestamp_s is not None and _number(timestamp_s, "timestamp_s") < 0:
        raise ValueError("Timestamp cannot be negative")
    required = {
        "gt_boxes", "gt_obj_classes", "gt_bbox_pair_ids", "gt_action_labels",
        "proposal_boxes", "proposal_classes", "proposal_scores", "preds_bbox_pair_ids",
        "preds_score",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Missing ST-HOI fields: {sorted(missing)}")
    gt_boxes = _boxes(result["gt_boxes"], box_transform)
    pred_boxes = _boxes(result["proposal_boxes"], box_transform)
    gt_classes = _classes(result["gt_obj_classes"], len(gt_boxes), mapping)
    pred_classes = _classes(result["proposal_classes"], len(pred_boxes), mapping)
    if len(result["proposal_scores"]) != len(pred_boxes):
        raise ValueError("Box and proposal score counts disagree")
    scores = []
    for row in result["proposal_scores"]:
        if len(row) != 2 or _integer(row[0], "score batch ID") != 0:
            raise ValueError("Expected [0,proposal_score] rows")
        scores.append(_number(row[1], "proposal score"))
    return VidHOIFrame(
        video_id, frame_id, split, box_source,
        "xyxy_pixels_original_frame" if box_transform else "xyxy_pixels_model_frame",
        _pairs(result["gt_bbox_pair_ids"], result["gt_action_labels"], gt_boxes,
               gt_classes, mapping, tracks=gt_track_ids),
        _pairs(result["preds_bbox_pair_ids"], result["preds_score"], pred_boxes,
               pred_classes, mapping, scores=scores, tracks=proposal_track_ids),
        timestamp_s,
    )


def select_frame_mask(frames: Sequence[VidHOIFrame], frame_ids: Sequence[str]) -> list[VidHOIFrame]:
    """Require an explicit nonempty mask and retain original result ordering."""
    keys = [frame.key for frame in frames]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate result frame IDs")
    if not frame_ids or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Frame mask must be nonempty and contain unique IDs")
    missing = set(frame_ids) - set(keys)
    if missing:
        raise ValueError(f"Frame mask contains missing frames: {sorted(missing)[:5]}")
    selected = set(frame_ids)
    return [frame for frame in frames if frame.key in selected]


def validate_video_partitions(partitions: Mapping[str, Sequence[str]]) -> None:
    seen: set[str] = set()
    for name, videos in partitions.items():
        if len(set(videos)) != len(videos) or seen.intersection(videos):
            raise ValueError(f"Duplicate or leaking video IDs in {name}")
        seen.update(videos)


def adapt_vidhoi_annotations(
    rows: Sequence[Mapping[str, Any]], mapping: ClassMapping,
    frame_manifest: Sequence[Mapping[str, Any]], *, split: str,
) -> list[VidHOIFrame]:
    """Adapt upstream frame annotations without confusing seconds and frame IDs.

    ``source_frame_id`` in the explicit manifest joins the upstream ``frame_id``;
    ``frame_id`` is the verified media frame index. The adapter does not guess
    an offset from ``middle_frame_timestamp`` or video FPS. Boxes are annotation
    pixels, and person_id/object_id supply actual track identity.
    """
    grouped: dict[tuple[str, int], list[Mapping]] = {}
    for row in rows:
        video = f"{row['video_folder']}/{row['video_id']}"
        key = (video, _integer(row["frame_id"], "source_frame_id"))
        grouped.setdefault(key, []).append(row)
    output = []
    seen = set()
    for info in frame_manifest:
        key = (info["video_id"], _integer(info["source_frame_id"], "source_frame_id"))
        if key in seen or key not in grouped:
            raise ValueError("Manifest has duplicate or missing annotation frame keys")
        seen.add(key)
        boxes: dict[str, tuple] = {}
        classes: dict[str, int] = {}
        labels: dict[tuple[str, str], list[float]] = {}
        for row in grouped[key]:
            subject = str(_integer(row["person_id"], "person track ID"))
            obj = str(_integer(row["object_id"], "object track ID"))
            object_class = _integer(row["object_class"], "object class")
            predicate = _integer(row["action_class"], "predicate class")
            if object_class not in mapping.objects or predicate not in mapping.predicates:
                raise ValueError("Unknown annotation class ID")
            for track, box_name, class_id in ((subject, "person_box", 0),
                                              (obj, "object_box", object_class)):
                box = tuple(_number(row[box_name][axis], "box coordinate")
                            for axis in ("xmin", "ymin", "xmax", "ymax"))
                if box[0] >= box[2] or box[1] >= box[3]:
                    raise ValueError("Annotation boxes must have positive size")
                if track in boxes and (boxes[track] != box or classes[track] != class_id):
                    raise ValueError("Inconsistent annotation box/class for the same track")
                boxes[track], classes[track] = box, class_id
            if subject == obj:
                raise ValueError("Subject and object track IDs must differ")
            labels.setdefault((subject, obj), [0.0] * len(mapping.predicates))[predicate] = 1.0
        track_order = sorted(boxes)
        track_index = {track: index for index, track in enumerate(track_order)}
        native = {
            "gt_boxes": [[0, *boxes[track]] for track in track_order],
            "gt_obj_classes": [[0, classes[track]] for track in track_order],
            "gt_bbox_pair_ids": [[track_index[sub], track_index[obj]] for sub, obj in labels],
            "gt_action_labels": list(labels.values()), "proposal_boxes": [],
            "proposal_classes": [], "proposal_scores": [], "preds_bbox_pair_ids": [],
            "preds_score": [],
        }
        output.append(adapt_sthoi_result(
            native, mapping, video_id=info["video_id"], frame_id=info["frame_id"],
            split=split, box_source="oracle", box_transform=BoxTransform(),
            timestamp_s=info.get("timestamp_s"),
            gt_track_ids={index: track for track, index in track_index.items()},
        ))
    if set(grouped) != seen:
        raise ValueError("Manifest must identify every annotation frame; subset explicitly beforehand")
    return output

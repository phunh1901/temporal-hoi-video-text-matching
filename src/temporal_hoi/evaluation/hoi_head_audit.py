"""Describe development regimes using AP values from one unchanged evaluator run.

Class AP is partitioned after the global proposal budget and shared-pair GT
matching. Predictions are never filtered or evaluated again to form these groups.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from temporal_hoi.inference.hoi_static import OracleFrame

REGIMES = (
    "sampled_train_seen", "whole_train_seen_not_sampled",
    "designed_pseudo_unseen", "incidental_dev_only",
)


def _triplet(value: Any, name: str) -> tuple[int, int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3 or value[0] != 0
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError(f"{name} requires [0, predicate_id, object_id] integer composition IDs")
    return tuple(value)


def _triplets(value: Any, name: str) -> set[tuple[int, int, int]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} requires an explicit composition list")
    result = {_triplet(row, name) for row in value}
    if len(result) != len(value):
        raise ValueError(f"Duplicate composition in {name}")
    return result


def _video_ids(value: Any, name: str) -> set[str]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} requires unique nonempty video IDs")
    return set(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_development_regimes(
    metrics: dict, protocol: dict, train_frames: list[OracleFrame],
) -> dict:
    """Partition GT-supported development class AP without changing its score context.

    ``seen`` in the first group means actual sampled positive supervision. Full
    training-video metadata is reported separately because unsampled labels were
    not used to optimize this head. Designed composition holdouts require both
    predicate and object to have observed sampled-train positive supervision.
    """
    if (protocol.get("schema_version") != "vidhoi_head_protocol_v1"
            or protocol.get("label_policy") != "positive_only_unknown"):
        raise ValueError("Expected the locked positive-only head development protocol")
    train_ids = _video_ids(protocol.get("train_video_ids"), "train_video_ids")
    dev_ids = _video_ids(protocol.get("dev_video_ids"), "dev_video_ids")
    if train_ids & dev_ids:
        raise ValueError("Training/development video overlap")
    seen = _triplets(protocol.get("seen_triplets"), "seen_triplets")
    pseudo = _triplets(protocol.get("pseudo_unseen_triplets"), "pseudo_unseen_triplets")
    final = _triplets(protocol.get("final_unseen_triplets"), "final_unseen_triplets")
    if seen & pseudo or seen & final or pseudo & final:
        raise ValueError("Seen/pseudo/final composition overlap")
    full_video = protocol.get("video_positive_triplets")
    if not isinstance(full_video, dict) or not (train_ids | dev_ids).issubset(full_video):
        raise ValueError("Full-video positive metadata is required for both cohort roles")
    full_train: set[tuple[int, int, int]] = set()
    for video in train_ids:
        full_train.update(_triplets(full_video[video], f"video_positive_triplets/{video}"))
    full_dev: set[tuple[int, int, int]] = set()
    for video in dev_ids:
        full_dev.update(_triplets(full_video[video], f"video_positive_triplets/{video}"))
    if full_train != seen:
        raise ValueError("Declared seen set must equal the selected full training-video compositions")
    if full_train & (pseudo | final) or full_dev & final:
        raise ValueError("A held-out composition entered a forbidden full-video role")
    if (not isinstance(train_frames, list) or not train_frames
            or any(not isinstance(frame, OracleFrame) for frame in train_frames)
            or {frame.video_id for frame in train_frames} != train_ids
            or len({frame.key for frame in train_frames}) != len(train_frames)):
        raise ValueError("Actual sampled training frames must match the locked video inventory")
    support: Counter = Counter()
    predicates: Counter = Counter()
    objects: Counter = Counter()
    for frame in train_frames:
        video_compositions = _triplets(full_video[frame.video_id], "training-video positives")
        for pair in frame.ground_truth:
            if pair.subject_class != 0 or any(value not in (0, 1) for value in pair.predicate_values):
                raise ValueError("Sampled GT must contain person-subject observed positive labels")
            for predicate, present in enumerate(pair.predicate_values):
                if present != 1:
                    continue
                composition = _triplet((0, predicate, pair.object_class), "sampled GT")
                if composition not in video_compositions:
                    raise ValueError("Sampled positive is absent from its full-video metadata")
                support[composition] += 1
                predicates[predicate] += 1
                objects[pair.object_class] += 1
    sampled = set(support)
    component_audit = []
    for composition in sorted(pseudo):
        _, predicate, obj = composition
        component_audit.append({
            "triplet": list(composition),
            "predicate_sampled_train_gt_instances": predicates[predicate],
            "object_sampled_train_gt_instances": objects[obj],
            "composition_sampled_train_gt_instances": support[composition],
        })
        if not predicates[predicate] or not objects[obj]:
            raise ValueError(
                "Designed pseudo-unseen composition lacks predicate/object positive support "
                f"in actual sampled training frames: {composition}")
    if (metrics.get("evaluator") != "ST-HOI_notebook_compat_v1"
            or metrics.get("metric_scale") != "fraction" or metrics.get("class_subset") != "full"
            or metrics.get("split") != "research_validation" or metrics.get("box_source") != "oracle"
            or metrics.get("top_k_before_class_filter") is not True
            or metrics.get("gt_matching_scope") != "shared_pair_across_predicates_per_frame"):
        raise ValueError("Expected one complete oracle development evaluator result")
    classes = metrics.get("classes")
    if not isinstance(classes, list) or metrics.get("num_classes") != len(classes):
        raise ValueError("Metric GT-supported class inventory differs from num_classes")
    grouped: dict[str, list] = {regime: [] for regime in REGIMES}
    records = []
    visited = set()
    whole_predicates = {predicate for _, predicate, _ in full_train}
    whole_objects = {obj for _, _, obj in full_train}
    for row in classes:
        composition = _triplet(row.get("triplet"), "evaluator class")
        if composition in visited or composition not in full_dev or composition in final:
            raise ValueError("Duplicate, unsupported or final composition in development metrics")
        visited.add(composition)
        count, ap = row.get("gt_count"), row.get("ap")
        if (type(count) is not int or count < 1 or isinstance(ap, bool)
                or not isinstance(ap, (int, float)) or not math.isfinite(ap) or not 0 <= ap <= 1):
            raise ValueError("Class AP/support must be finite [0,1] and a positive integer")
        regime = ("sampled_train_seen" if composition in sampled
                  else "whole_train_seen_not_sampled" if composition in full_train
                  else "designed_pseudo_unseen" if composition in pseudo
                  else "incidental_dev_only")
        _, predicate, obj = composition
        record = {
            "triplet": list(composition), "regime": regime, "ap": float(ap),
            "development_gt_instances": count,
            "sampled_train_gt_instances": support[composition],
            "present_in_whole_training_video_metadata": composition in full_train,
            "predicate_sampled_train_gt_instances": predicates[predicate],
            "object_sampled_train_gt_instances": objects[obj],
            "predicate_unseen_in_sampled_train": predicate not in predicates,
            "object_unseen_in_sampled_train": obj not in objects,
            "predicate_unseen_in_whole_training_videos": predicate not in whole_predicates,
            "object_unseen_in_whole_training_videos": obj not in whole_objects,
        }
        grouped[regime].append(record)
        records.append(record)
    overall = _mean([row["ap"] for row in records])
    if ((overall is None and metrics.get("map") is not None)
            or (overall is not None and (isinstance(metrics.get("map"), bool)
                or not isinstance(metrics.get("map"), (int, float))
                or not math.isclose(overall, metrics["map"], rel_tol=1e-12, abs_tol=1e-12)))):
        raise ValueError("Overall mAP differs from its GT-supported class AP values")
    groups = {}
    for regime, rows in grouped.items():
        groups[regime] = {
            "num_gt_supported_classes": len(rows),
            "num_gt_instances": sum(row["development_gt_instances"] for row in rows),
            "map": _mean([row["ap"] for row in rows]),
            "triplets": [row["triplet"] for row in rows],
            "predicate_novelty_in_sampled_train_classes": sum(
                row["predicate_unseen_in_sampled_train"] for row in rows),
            "object_novelty_in_sampled_train_classes": sum(
                row["object_unseen_in_sampled_train"] for row in rows),
        }
    return {
        "schema_version": "vidhoi_development_regime_summary_v1",
        "scope": "internal_oracle_development_after_checkpoint_selection",
        "metric_scale": "fraction", "ap_definition": metrics.get("ap_definition"),
        "class_ids": "[subject_class=0, predicate_id, object_class] in the pinned class mapping",
        "groups": groups, "composition_support": records,
        "composition_to_regime": {",".join(map(str, row["triplet"])): row["regime"]
                                  for row in records},
        "component_support": {
            "sampled_train_predicate_positive_instances": {
                str(key): value for key, value in sorted(predicates.items())},
            "sampled_train_object_positive_instances": {
                str(key): value for key, value in sorted(objects.items())},
            "whole_training_video_predicate_ids": sorted(whole_predicates),
            "whole_training_video_object_ids": sorted(whole_objects),
        },
        "designed_pseudo_component_audit": component_audit,
        "designed_pseudo_components_seen_in_actual_sampled_train": True,
        "all_development_map": overall,
        "num_gt_supported_classes": len(records),
        "num_gt_instances": sum(row["development_gt_instances"] for row in records),
        "selection_uses_all_development_map": True,
        "groups_used_to_reselect_checkpoint": False,
        "zero_shot_benchmark_claim": False,
        "official_final_split_accessed": False,
        "prediction_budget_unchanged": True, "global_matching_unchanged": True,
        "evaluator_context": {name: metrics.get(name) for name in (
            "evaluator", "top_k", "top_k_before_class_filter", "iou_threshold",
            "gt_matching_scope", "global_ties", "num_frames", "frame_ids_sha256")},
        "interpretation": (
            "Group means reuse class AP from one global evaluator run. Full-video seen labels "
            "outside sampled frames are not positive head supervision. Incidental holdouts "
            "can include unseen predicates or objects. Development groups describe the same "
            "selection data and do not estimate final zero-shot performance."),
    }

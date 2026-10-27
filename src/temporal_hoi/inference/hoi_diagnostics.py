"""Fixed image-representation diagnostics on the same oracle VidHOI train probe.

The three views change only pixels: full frame, target-object crop, or union
crop. All directed oracle proposals, predicate prompts and evaluator settings
remain fixed. Annotated-positive predicate ranks describe conditional rankings,
not accuracy or verified negatives among the other candidate predicates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from temporal_hoi.data.vidhoi import ClassMapping, HOIPair, VidHOIFrame, identity_hash
from temporal_hoi.evaluation.hoi import evaluate_frame_hoi
from temporal_hoi.inference.hoi_static import (
    PROMPT_VERSION,
    Encoder,
    OracleFrame,
    OriginalCLIPEncoder,
    _normalized,
    _peak_rss_bytes,
    _union,
    _write_json,
    build_prompts,
    file_sha256,
    preflight,
    proposal_pairs,
)

VARIANTS = ("full_frame", "object_only", "union")
RANK_TIES = "predicate_id_ascending_for_ordinal_rank; best/worst_tie_interval_also_reported"
UNKNOWN_WARNING = (
    "Only annotated positives have predicate-rank diagnostics. Other predicate labels and "
    "unannotated actor links are not independently verified negatives. These ranks and pixel/"
    "score symmetry observations are not detection accuracy or actor correctness."
)


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _pixel_fingerprint(image: Image.Image) -> str:
    digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _view_descriptors(frame: OracleFrame, variant: str) -> tuple[list[dict], list[int]]:
    if variant not in VARIANTS:
        raise ValueError(f"Representation must be one of {VARIANTS}")
    descriptors: list[dict] = []
    lookup: dict[Any, int] = {}
    pair_inputs = []
    for pair_index, (subject, obj) in enumerate(proposal_pairs(frame)):
        if variant == "full_frame":
            key, bounds = "frame", None
        elif variant == "object_only":
            key = obj.track_id
            bounds = tuple(math.floor(value) if index < 2 else math.ceil(value)
                           for index, value in enumerate(obj.bbox_xyxy))
        else:
            key, bounds = pair_index, _union(subject, obj)
        if key not in lookup:
            lookup[key] = len(descriptors)
            descriptors.append({"input_key": str(key), "crop_bounds_xyxy": bounds,
                                "pair_indices": []})
        input_index = lookup[key]
        descriptors[input_index]["pair_indices"].append(pair_index)
        pair_inputs.append(input_index)
    return descriptors, pair_inputs


def infer_representation_frame(
    frame: OracleFrame, encoder: Encoder, text_features: dict[int, np.ndarray], *,
    variant: str, batch_size: int, logit_scale: float, device: str = "cpu",
) -> tuple[VidHOIFrame, dict]:
    """Reuse equivalent pixels within a frame; union preserves static arithmetic/batches."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(logit_scale) or not 0 < logit_scale <= 100:
        raise ValueError("logit_scale must be in (0,100]")
    started = time.perf_counter()
    pairs = proposal_pairs(frame)
    descriptors, pair_inputs = _view_descriptors(frame, variant)
    features: list[np.ndarray] = []
    image_calls, encode_seconds = 0, 0.0
    preparation_started = time.perf_counter()
    with Image.open(frame.image_path) as source:
        image = source.convert("RGB")
        for offset in range(0, len(descriptors), batch_size):
            current = descriptors[offset:offset + batch_size]
            crops = [image if item["crop_bounds_xyxy"] is None
                     else image.crop(item["crop_bounds_xyxy"]) for item in current]
            for crop, item in zip(crops, current, strict=True):
                item["rgb_size"] = list(crop.size)
                item["pixel_sha256"] = _pixel_fingerprint(crop)
            _synchronize(device)
            encode_started = time.perf_counter()
            encoded = _normalized(encoder.encode_images(crops), len(crops), "image")
            _synchronize(device)
            encode_seconds += time.perf_counter() - encode_started
            image_calls += 1
            features.extend(encoded[index] for index in range(len(crops)))
    preparation_and_encode = time.perf_counter() - preparation_started
    scoring_started = time.perf_counter()
    predictions = []
    for pair_index, (subject, obj) in enumerate(pairs):
        feature = features[pair_inputs[pair_index]]
        text = text_features[obj.class_id]
        if text.shape[1] != feature.shape[0]:
            raise ValueError("Image and text encoder dimensions disagree")
        cosine = np.clip(text @ feature, -1.0, 1.0).astype(np.float64)
        values = np.exp(logit_scale * cosine)
        predictions.append(HOIPair(
            0, obj.class_id, subject.bbox_xyxy, obj.bbox_xyxy,
            tuple(float(value) for value in values), subject_track_id=subject.track_id,
            object_track_id=obj.track_id))
    inferred = VidHOIFrame(frame.video_id, frame.frame_id, "train", "oracle",
                            "xyxy_pixels_original_frame", frame.ground_truth,
                            tuple(predictions), frame.timestamp_s)
    return inferred, {
        "frame_key": frame.key, "source_frame_id": frame.source_frame_id,
        "num_proposal_pairs": len(pairs), "image_encoder_input_count": len(descriptors),
        "image_encoder_calls": image_calls, "image_encode_seconds": encode_seconds,
        "decode_crop_fingerprint_seconds": preparation_and_encode - encode_seconds,
        "scoring_seconds": time.perf_counter() - scoring_started,
        "total_frame_seconds": time.perf_counter() - started,
        "pair_input_indices": pair_inputs, "inputs": descriptors,
    }


def _pair_identity(pair: HOIPair) -> tuple[str, str]:
    if pair.subject_track_id is None or pair.object_track_id is None:
        raise ValueError("Oracle diagnostic requires explicit entity track IDs")
    return pair.subject_track_id, pair.object_track_id


def positive_predicate_ranks(frames: list[VidHOIFrame], mapping: ClassMapping) -> dict:
    """Rank each known positive against all predicates; unannotated labels stay unknown."""
    rows = []
    for frame in frames:
        proposals = {_pair_identity(pair): pair for pair in frame.predictions}
        if len(proposals) != len(frame.predictions):
            raise ValueError("Duplicate proposal identity")
        for truth in frame.ground_truth:
            key = _pair_identity(truth)
            if key not in proposals:
                raise ValueError("Known positive has no proposal; no silent rank-row dropping")
            prediction = proposals[key]
            values = np.asarray(prediction.predicate_values, dtype=np.float64)
            for predicate, value in enumerate(truth.predicate_values):
                if value != 1:
                    continue
                score = float(values[predicate])
                greater = int(np.count_nonzero(values > score))
                equal = np.flatnonzero(values == score)
                rows.append({
                    "frame_key": frame.key, "subject_track_id": key[0],
                    "object_track_id": key[1], "predicate_id": predicate,
                    "predicate_name": mapping.predicates[predicate], "score": score,
                    "rank": greater + 1 + int(np.count_nonzero(equal < predicate)),
                    "rank_best": greater + 1, "rank_worst": greater + len(equal),
                    "tie_count": len(equal), "num_candidate_predicates": len(values),
                })
    ranks = [row["rank"] for row in rows]
    pair_best_ranks = {}
    for row in rows:
        key = (row["frame_key"], row["subject_track_id"], row["object_track_id"])
        pair_best_ranks[key] = min(pair_best_ranks.get(key, row["rank"]), row["rank"])
    return {
        "schema_version": "vidhoi_known_positive_predicate_ranks_v1",
        "scope": "annotated_positive_labels_conditional_on_oracle_pair",
        "not_accuracy": True, "unknown_candidates_are_not_verified_negative": True,
        "rank_tie_policy": RANK_TIES, "warning": UNKNOWN_WARNING,
        "num_known_positive_labels": len(rows),
        "num_annotated_pairs": len(pair_best_ranks),
        "annotated_pairs_with_any_positive_rank_le": {
            str(k): sum(rank <= k for rank in pair_best_ranks.values()) for k in (1, 5, 10)},
        "mean_rank": float(np.mean(ranks)) if rows else None,
        "median_rank": float(np.median(ranks)) if rows else None,
        "known_positive_labels_rank_le": {
            str(k): sum(rank <= k for rank in ranks) for k in (1, 5, 10)},
        "records": rows,
    }


def _vector_identity(left: HOIPair, right: HOIPair) -> dict:
    a = np.asarray(left.predicate_values, dtype=np.float64)
    b = np.asarray(right.predicate_values, dtype=np.float64)
    order_a = np.lexsort((np.arange(len(a)), -a))
    order_b = np.lexsort((np.arange(len(b)), -b))
    return {"score_vectors_exactly_equal": bool(np.array_equal(a, b)),
            "max_absolute_score_difference": float(np.max(np.abs(a - b))),
            "predicate_order_equal": bool(np.array_equal(order_a, order_b)),
            "predicate_tie_groups_equal": bool(np.array_equal(a[:, None] == a[None, :],
                                                              b[:, None] == b[None, :]))}


def role_symmetry_diagnostics(
    frames: list[VidHOIFrame], frame_statistics: list[dict], *, variant: str,
) -> dict:
    """Describe representational symmetries without inventing actor ground truth."""
    duplicate_groups, reverse_pairs = [], []
    for frame, stats in zip(frames, frame_statistics, strict=True):
        indices_by_object: dict[str, list[int]] = {}
        index_by_pair = {}
        for index, pair in enumerate(frame.predictions):
            subject, obj = _pair_identity(pair)
            indices_by_object.setdefault(obj, []).append(index)
            index_by_pair[(subject, obj)] = index

        def pixels_equal(left: int, right: int) -> bool:
            inputs = stats["inputs"]
            indices = stats["pair_input_indices"]
            return inputs[indices[left]]["pixel_sha256"] == inputs[indices[right]]["pixel_sha256"]

        for obj, indices in sorted(indices_by_object.items()):
            if len(indices) < 2:
                continue
            reference = indices[0]
            duplicate_groups.append({
                "frame_key": frame.key, "object_track_id": obj,
                "subject_track_ids": [frame.predictions[index].subject_track_id for index in indices],
                "comparisons_to_first": [
                    {"pair_index": index, "pixels_exactly_equal": pixels_equal(reference, index),
                     **_vector_identity(frame.predictions[reference], frame.predictions[index])}
                    for index in indices[1:]],
            })
        for (subject, obj), index in sorted(index_by_pair.items()):
            reverse = index_by_pair.get((obj, subject))
            if reverse is None or subject >= obj:
                continue
            reverse_pairs.append({
                "frame_key": frame.key, "track_pair": [subject, obj],
                "pixels_exactly_equal": pixels_equal(index, reverse),
                "same_object_class": frame.predictions[index].object_class
                == frame.predictions[reverse].object_class,
                **_vector_identity(frame.predictions[index], frame.predictions[reverse]),
            })
    return {
        "variant": variant, "not_actor_correctness": True, "warning": UNKNOWN_WARNING,
        "interpretation": (
            "Full-frame and union views are symmetric for reversed human-human pairs. "
            "Object-only may change when the target crop changes; that difference alone "
            "does not establish understanding of actor roles."),
        "same_object_different_subject_groups": duplicate_groups,
        "num_same_object_different_subject_groups": len(duplicate_groups),
        "reverse_pairs": reverse_pairs, "num_reverse_pairs": len(reverse_pairs),
    }


def _positive_key(row: dict) -> tuple:
    return (row["frame_key"], row["subject_track_id"], row["object_track_id"],
            row["predicate_id"])


def compare_diagnostic_variants(variant_results: dict[str, dict]) -> dict:
    if tuple(variant_results) != VARIANTS:
        raise ValueError("All three fixed variants in their declared order are required")
    reference = variant_results["union"]
    frame_keys = [frame.key for frame in reference["frames"]]
    pair_keys = [(frame.key, *_pair_identity(pair)) for frame in reference["frames"]
                 for pair in frame.predictions]
    truth = [[asdict(pair) for pair in frame.ground_truth] for frame in reference["frames"]]
    for variant in VARIANTS:
        result = variant_results[variant]
        if [frame.key for frame in result["frames"]] != frame_keys:
            raise ValueError("Variants changed frame cohort or order")
        if [(frame.key, *_pair_identity(pair)) for frame in result["frames"]
                for pair in frame.predictions] != pair_keys:
            raise ValueError("Variants changed proposal pairs or candidate order")
        if [[asdict(pair) for pair in frame.ground_truth] for frame in result["frames"]] != truth:
            raise ValueError("Variants changed ground truth")
    rank_lookup = {
        variant: {_positive_key(row): row for row in result["positive_ranks"]["records"]}
        for variant, result in variant_results.items()}
    keys = list(rank_lookup["union"])
    if any(list(rank_lookup[variant]) != keys for variant in VARIANTS):
        raise ValueError("Known-positive rank rows are not paired identically")
    paired = []
    for key in keys:
        paired.append({
            "frame_key": key[0], "subject_track_id": key[1], "object_track_id": key[2],
            "predicate_id": key[3],
            "ranks": {variant: {field: rank_lookup[variant][key][field]
                                for field in ("rank", "rank_best", "rank_worst", "tie_count")}
                      for variant in VARIANTS},
            "rank_deltas_vs_union": {
                variant: rank_lookup[variant][key]["rank"] - rank_lookup["union"][key]["rank"]
                for variant in VARIANTS},
        })
    union_map = reference["metrics"]["map"]
    return {
        "schema_version": "vidhoi_representation_comparison_v1",
        "variants": list(VARIANTS), "same_frame_cohort_and_order": True,
        "same_all_directed_proposals_and_order": True, "same_ground_truth": True,
        "pair_identity_sha256": identity_hash([json.dumps(key) for key in pair_keys]),
        "selected_best_variant": None, "configuration_selection_run": False,
        "metric_scale": "fraction", "warning": UNKNOWN_WARNING,
        "variant_metrics": {
            variant: {"map": result["metrics"]["map"],
                      "num_classes": result["metrics"]["num_classes"],
                      "known_positive_labels_rank_le":
                      result["positive_ranks"]["known_positive_labels_rank_le"],
                      "num_annotated_pairs": result["positive_ranks"]["num_annotated_pairs"],
                      "annotated_pairs_with_any_positive_rank_le":
                      result["positive_ranks"]["annotated_pairs_with_any_positive_rank_le"],
                      "mean_known_positive_rank": result["positive_ranks"]["mean_rank"]}
            for variant, result in variant_results.items()},
        "map_deltas_vs_union": {
            variant: None if union_map is None or result["metrics"]["map"] is None
            else result["metrics"]["map"] - union_map
            for variant, result in variant_results.items()},
        "paired_known_positive_ranks": paired,
    }


def _file_records(root: Path) -> list[dict]:
    return [{"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
             "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file()]


def run_representation_diagnostics(
    config: dict, output_dir: str | Path, *, encoder: Encoder | None = None,
) -> dict:
    started = time.perf_counter()
    fixture = encoder is not None
    if fixture and encoder.metadata.get("test_fixture") is not True:
        raise ValueError("Injected encoder must be explicitly marked test_fixture=True")
    if "diagnostic_variants" in config and config["diagnostic_variants"] != list(VARIANTS):
        raise ValueError("diagnostic_variants must contain all three fixed views")
    mapping, source_frames, _ = preflight(config, test_fixture=fixture)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cache_policy = {"full_frame": "one_image_per_frame_with_proposals",
                    "object_only": "one_crop_per_target_track_per_frame",
                    "union": "one_crop_per_directed_pair_no_deduplication"}
    effective_config = {**config, "diagnostic_variants": list(VARIANTS),
                        "representation_cache_policy": cache_policy}
    _write_json(output / "run_config.json", effective_config)
    torch.set_num_threads(config.get("threads", 2))
    device = config.get("device", "cpu")
    print(f"DIAGNOSTIC_PREFLIGHT_PASSED: {len(source_frames)} train frames, same oracle pairs",
          flush=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    if encoder is None:
        encoder = OriginalCLIPEncoder(Path(config["checkpoint"]), device=device)
    _synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    batch_size = config.get("batch_size", 8)
    text_started = time.perf_counter()
    text_features, prompts_by_class = {}, {}
    text_calls = 0
    classes = sorted({obj.class_id for frame in source_frames for _, obj in proposal_pairs(frame)})
    for class_id in classes:
        prompts = build_prompts(mapping, class_id)
        prompts_by_class[class_id] = prompts
        chunks = []
        for offset in range(0, len(prompts), batch_size):
            current = prompts[offset:offset + batch_size]
            chunks.append(_normalized(encoder.encode_texts(current), len(current), "text"))
            text_calls += 1
        text_features[class_id] = np.concatenate(chunks, axis=0)
    _synchronize(device)
    text_seconds = time.perf_counter() - text_started
    prompt_hash = hashlib.sha256(json.dumps(prompts_by_class, sort_keys=True,
                                            ensure_ascii=False).encode()).hexdigest()
    _write_json(output / "prompts.json", prompts_by_class)
    results, summaries, symmetries = {}, {}, {}
    for variant in VARIANTS:
        print(f"DIAGNOSTIC_VARIANT_START: {variant}", flush=True)
        variant_started = time.perf_counter()
        variant_dir = output / variant
        variant_dir.mkdir()
        frames, statistics = [], []
        for source in source_frames:
            inferred, stats = infer_representation_frame(
                source, encoder, text_features, variant=variant, batch_size=batch_size,
                logit_scale=config.get("logit_scale", 1.0), device=device)
            frames.append(inferred)
            statistics.append(stats)
        evaluation_started = time.perf_counter()
        metrics = evaluate_frame_hoi(frames)
        positive_ranks = positive_predicate_ranks(frames, mapping)
        evaluation_seconds = time.perf_counter() - evaluation_started
        symmetries[variant] = role_symmetry_diagnostics(frames, statistics, variant=variant)
        results[variant] = {"frames": frames, "metrics": metrics, "positive_ranks": positive_ranks}
        _write_json(variant_dir / "predictions.json", [asdict(frame) for frame in frames])
        _write_json(variant_dir / "metrics.json", metrics)
        _write_json(variant_dir / "known_positive_ranks.json", positive_ranks)
        summary = {
            "variant": variant, "map": metrics["map"], "num_frames": len(frames),
            "num_proposal_pairs": sum(len(frame.predictions) for frame in frames),
            "frames_without_proposals": sum(not frame.predictions for frame in frames),
            "dropped_frames": 0, "dropped_proposal_pairs": 0,
            "image_encoder_input_count": sum(row["image_encoder_input_count"] for row in statistics),
            "image_encoder_calls": sum(row["image_encoder_calls"] for row in statistics),
            "image_encode_seconds": sum(row["image_encode_seconds"] for row in statistics),
            "evaluation_seconds": evaluation_seconds,
            "total_variant_seconds": time.perf_counter() - variant_started,
            "frame_statistics": statistics,
        }
        summaries[variant] = summary
        _write_json(variant_dir / "receipt.json", {
            "schema_version": "vidhoi_representation_variant_receipt_v1", "status": "completed",
            "level": "representation_diagnostic_encoder_fixture" if fixture
            else "representation_diagnostic_train_oracle",
            "model_inference_run": not fixture, "model_training_run": False,
            "full_benchmark_reproduction_claim": False, "benchmark_comparison_eligible": False,
            "official_final_split_accessed": False, "split": "train", "box_source": "oracle",
            "object_label_source": "oracle", "prompt_sha256": prompt_hash,
            "shared_model_and_text_cache": True, "not_actor_correctness": True,
            "files": _file_records(variant_dir), **summary,
        })
        print(f"DIAGNOSTIC_VARIANT_COMPLETE: {variant}, frames={len(frames)}, "
              f"image_inputs={summary['image_encoder_input_count']}", flush=True)
    comparison = compare_diagnostic_variants(results)
    _write_json(output / "comparison.json", comparison)
    _write_json(output / "role_symmetry.json", {
        "schema_version": "vidhoi_role_symmetry_v1", "warning": UNKNOWN_WARNING,
        "not_actor_correctness": True, "variants": symmetries})
    versions = {}
    for package in ("torch", "torchvision", "open-clip-torch", "numpy", "pillow", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    receipt = {
        "schema_version": "vidhoi_representation_diagnostic_receipt_v1", "status": "completed",
        "level": "representation_diagnostic_encoder_fixture" if fixture
        else "representation_diagnostic_train_oracle",
        "variants": list(VARIANTS), "variant_summaries": summaries,
        "num_frames": len(source_frames),
        "num_proposal_pairs": sum(len(proposal_pairs(frame)) for frame in source_frames),
        "dropped_frames": 0, "dropped_proposal_pairs": 0, "model_inference_run": not fixture,
        "model_training_run": False, "full_benchmark_reproduction_claim": False,
        "zero_shot_benchmark_claim": False, "benchmark_comparison_eligible": False,
        "official_final_split_accessed": False, "split": "train", "box_source": "oracle",
        "object_label_source": "oracle", "shared_model_load_count": 1 if not fixture else 0,
        "shared_text_encode_calls": text_calls,
        "shared_text_encoder_input_count": sum(len(prompts) for prompts in prompts_by_class.values()),
        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_hash,
        "score_transform": "exp(fixed_logit_scale * normalized_image_text_cosine)",
        "score_is_probability": False, "logit_scale": config.get("logit_scale", 1.0),
        "frame_ids_sha256": identity_hash([frame.key for frame in source_frames]),
        "input_hashes": {key: config[f"{key}_sha256"]
                         for key in ("manifest", "mapping", "checkpoint")},
        "checkpoint_provenance": config["checkpoint_provenance"], "encoder": encoder.metadata,
        "resources": {
            "shared_model_load_seconds": model_load_seconds, "shared_text_encode_seconds": text_seconds,
            "total_seconds": time.perf_counter() - started, "process_peak_rss_bytes": _peak_rss_bytes(),
            "process_peak_scope": "whole_process_lifetime_not_isolated_run_delta",
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if device == "cuda" else None},
        "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
        "same_frame_pair_predicate_cohort": True, "rank_tie_policy": RANK_TIES,
        "representation_cache_policy": cache_policy, "equal_image_encode_compute": False,
        "warning": UNKNOWN_WARNING, "files": _file_records(output),
    }
    _write_json(output / "receipt.json", receipt)
    return receipt

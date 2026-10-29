"""Audit fixed initial/last HOI heads from a verified result ZIP, without training.

The benchmark evaluator remains unchanged. Budget/matching traces describe its
behavior; unannotated candidate entries are never relabeled as verified negatives.
Recorded source-environment metrics and current-environment rescoring are separate.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import stat
import time
import zipfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load as load_tensors
from torch.nn import functional as F

from temporal_hoi.data.hoi_head_cache import _restore_frame
from temporal_hoi.data.vidhoi import ClassMapping, HOIPair, VidHOIFrame
from temporal_hoi.evaluation.hoi import _log_score, bbox_iou, evaluate_frame_hoi
from temporal_hoi.inference.hoi_kaggle import safe_member
from temporal_hoi.inference.hoi_static import MAPPING_SHA256, _peak_rss_bytes, file_sha256
from temporal_hoi.models.hoi_head import MODES, HOIResidualHead, view_features
from temporal_hoi.training.hoi_head_study import (
    ARRAY_KEYS,
    SEEDS,
    _common,
    _provenance,
    evaluate_head_state,
    validate_cache,
)

MAX_SOURCE_BYTES = 128 * 1024**2
MAX_SOURCE_EXPANDED_BYTES = 512 * 1024**2
MAX_SOURCE_MEMBER_BYTES = 96 * 1024**2
MAX_SOURCE_ENTRIES = 2048
INPUT_KEYS = ("train_manifest", "dev_manifest", "protocol", "mapping", "checkpoint")
STATES = ("initial", "last")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _json(data: bytes) -> Any:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON key in source result")
            result[key] = value
        return result

    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Nonfinite JSON constant {value}")))


def _numeric_npz(data: bytes, expected: set[str]) -> dict:
    # Inspect nested member sizes before NumPy allocates arrays from their headers.
    with zipfile.ZipFile(io.BytesIO(data)) as nested:
        entries = nested.infolist()
        if (len(entries) != len(expected) or len({row.filename for row in entries}) != len(entries)
                or sum(row.file_size for row in entries) > MAX_SOURCE_EXPANDED_BYTES
                or any(row.file_size > MAX_SOURCE_MEMBER_BYTES for row in entries)):
            raise ValueError("Invalid or oversized numeric NPZ inventory")
        for row in entries:
            safe_member(row.filename)
            with nested.open(row) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported numeric NPY header version")
                if (dtype.hasobject or len(shape) > 3
                        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                        or math.prod(shape) * dtype.itemsize > MAX_SOURCE_MEMBER_BYTES
                        or stream.tell() + math.prod(shape) * dtype.itemsize != row.file_size):
                    raise ValueError("Unsafe object/oversized/inconsistent numeric NPY header")
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        if set(arrays.files) != expected:
            raise ValueError("Numeric NPZ array schema mismatch")
        result = {name: arrays[name].copy() for name in arrays.files}
    if any(value.dtype.hasobject for value in result.values()):
        raise ValueError("Object arrays are not accepted")
    return result


def _inventory(members: dict[str, bytes], prefix: str, receipt: dict) -> None:
    records = receipt.get("files")
    if not isinstance(records, list):
        raise ValueError("Missing source file inventory")
    actual = {name[len(prefix):] for name in members
              if name.startswith(prefix) and name != prefix + "receipt.json"}
    declared = set()
    for row in records:
        name = str(safe_member(row.get("path")))
        if name in declared:
            raise ValueError("Duplicate source inventory record")
        declared.add(name)
        data = members.get(prefix + name)
        if (data is None or type(row.get("bytes")) is not int or len(data) != row["bytes"]
                or _sha(data) != _hash(row.get("sha256"), "inventory sha256")):
            raise ValueError("Source inventory hash/byte mismatch")
    if actual != declared:
        raise ValueError("Source inventory omits or adds files")


def _scope(receipt: dict, schema: str) -> None:
    if receipt.get("schema_version") != schema or receipt.get("status") != "completed":
        raise ValueError("Incomplete or wrong-schema source receipt")
    for name in ("official_final_split_accessed", "final_unseen_accessed"):
        if receipt.get(name) is not False:
            raise ValueError("Final data cannot enter this development diagnostic")


def _role(members: dict[str, bytes], role: str, fixture: bool) -> dict:
    prefix = f"study/cache/{role}/"
    receipt = _json(members[prefix + "receipt.json"])
    _scope(receipt, "vidhoi_head_role_cache_v1")
    _inventory(members, prefix, receipt)
    if receipt.get("role") != role:
        raise ValueError("Source cache role mismatch")
    payload = _json(members[prefix + "frames.json"])
    mapping = ClassMapping({int(key): value for key, value in payload["mapping"]["objects"].items()},
                           {int(key): value for key, value in payload["mapping"]["predicates"].items()})
    arrays = _numeric_npz(members[prefix + "arrays.npz"], set(ARRAY_KEYS))
    cache = {**arrays, "mapping": mapping, "frames": [_restore_frame(row) for row in payload["frames"]],
             "metadata": payload["metadata"]}
    for key, value in payload["metadata"].items():
        if receipt.get(key) != value:
            raise ValueError("Source role metadata differs from its receipt")
    checked = validate_cache(cache, role=role)
    _provenance(checked, fixture=fixture)
    if receipt.get("num_frames") != len(checked["frames"]) or receipt.get("num_pairs") != checked["count"]:
        raise ValueError("Source cache count differs from its canonical inventory")
    for name in ("object_features", "union_features"):
        if not np.allclose(np.linalg.norm(checked[name], axis=1), 1, atol=1e-5):
            raise ValueError("Source features must retain normalized FP32 values")
    return checked


def _prediction_frames(data: bytes) -> list[VidHOIFrame]:
    def pair(value):
        return HOIPair(**{**value, "subject_box": tuple(value["subject_box"]),
                          "object_box": tuple(value["object_box"]),
                          "predicate_values": tuple(value["predicate_values"])})

    return [VidHOIFrame(**{**row,
        "ground_truth": tuple(pair(value) for value in row["ground_truth"]),
        "predictions": tuple(pair(value) for value in row["predictions"])}) for row in _json(data)]


def _recorded_frames_match_cache(frames: list[VidHOIFrame], cache: dict) -> None:
    if len(frames) != len(cache["frames"]):
        raise ValueError("Recorded predictions dropped source frames")
    cursor = 0
    for frame, expected in zip(frames, cache["frames"], strict=True):
        wanted = int(np.count_nonzero(cache["frame_index"] == cursor))
        if (frame.video_id != expected.video_id or frame.frame_id != expected.frame_id
                or frame.timestamp_s != expected.timestamp_s or frame.ground_truth != expected.ground_truth
                or frame.split != "research_validation" or frame.box_source != "oracle"
                or frame.coordinate_system != "xyxy_pixels_original_frame"
                or len(frame.predictions) != wanted):
            raise ValueError("Recorded prediction geometry/GT/frame coverage differs from cache")
        cursor += 1
    for pair, expected in zip((pair for frame in frames for pair in frame.predictions),
                              cache["templates"], strict=True):
        if (len(pair.predicate_values) != len(cache["mapping"].predicates)
                or any(not math.isfinite(value) or value < 0 for value in pair.predicate_values)
                or replace(pair, predicate_values=expected.predicate_values) != expected):
            raise ValueError("Recorded proposal identity/geometry/score schema differs from cache")


def load_pair_diagnostic_source(
    path: str | Path, expected_sha: str, *, test_fixture: bool = False,
) -> dict:
    """Read and verify a result archive; never extract or execute its contents."""
    path = Path(path)
    _hash(expected_sha, "expected source SHA256")
    if (not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_SOURCE_BYTES
            or file_sha256(path) != expected_sha):
        raise ValueError("Source result file/hash/size mismatch")
    members = {}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (not entries or len(entries) > MAX_SOURCE_ENTRIES
                or sum(row.file_size for row in entries) > MAX_SOURCE_EXPANDED_BYTES):
            raise ValueError("Source ZIP exceeds the bounded entry/byte budget")
        for row in entries:
            name = str(safe_member(row.filename))
            mode = (row.external_attr >> 16) & 0xFFFF
            if (name in members or row.is_dir() or stat.S_ISLNK(mode) or row.flag_bits & 1
                    or row.file_size > MAX_SOURCE_MEMBER_BYTES
                    or row.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}):
                raise ValueError("Unsafe/duplicate/oversized source ZIP member")
            members[name] = archive.read(row)  # ZipFile validates each CRC during this read.
    outer = _json(members["study/receipt.json"])
    training = _json(members["study/training/receipt.json"])
    cache_receipt = _json(members["study/cache/receipt.json"])
    evaluation = _json(members["study/evaluation/receipt.json"])
    for prefix, receipt, schema in (
        ("study/", outer, "vidhoi_head_study_run_v1"),
        ("study/cache/", cache_receipt, "vidhoi_head_cache_receipt_v1"),
        ("study/training/", training, "vidhoi_head_study_result_v1"),
        ("study/evaluation/", evaluation, "vidhoi_head_evaluation_audit_v1"),
    ):
        _scope(receipt, schema)
        _inventory(members, prefix, receipt)
    if (training.get("level") != ("synthetic_feature_head_study" if test_fixture
                                  else "oracle_frozen_feature_development_head_study")
            or training.get("modes") != list(MODES) or training.get("seeds") != list(SEEDS)
            or training.get("model_training_run") is not True
            or training.get("selected_best_seed") is not None):
        raise ValueError("Source is not the completed fixed three-view/three-seed head study")
    if (cache_receipt.get("level") != ("synthetic_encoder_fixture" if test_fixture
                                      else "vidhoi_head_frozen_clip_feature_cache")
            or cache_receipt.get("encoder", {}).get("test_fixture") is not test_fixture
            or cache_receipt.get("model_inference_run") is not (not test_fixture)
            or cache_receipt.get("model_training_run") is not False):
        raise ValueError("Source cache inference/fixture level mismatch")
    hashes = outer.get("input_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(INPUT_KEYS):
        raise ValueError("Source requires all five locked input hashes")
    for name, value in hashes.items():
        _hash(value, name)
    if hashes != cache_receipt.get("input_hashes") or hashes != evaluation.get("input_hashes"):
        raise ValueError("Source stages refer to different inputs")
    train, dev = _role(members, "train", test_fixture), _role(members, "validation", test_fixture)
    if train["mapping"] != dev["mapping"] or train["dimension"] != dev["dimension"]:
        raise ValueError("Source roles differ in mapping/dimension")
    if {frame.video_id for frame in train["frames"]} & {frame.video_id for frame in dev["frames"]}:
        raise ValueError("Source training/development videos overlap")
    if not test_fixture and (hashes["mapping"] != MAPPING_SHA256 or train["dimension"] != 512
                             or len(train["mapping"].objects) != 78
                             or len(train["mapping"].predicates) != 50):
        raise ValueError("Real diagnostics require the pinned original CLIP/VidHOI cache")
    for role, loaded in (("train", train), ("dev", dev)):
        if (training.get(f"{role}_cache_fingerprint") != loaded["fingerprint"]
                or loaded["metadata"].get("input_hashes") != hashes
                or training.get(f"{role}_cache_metadata") != loaded["metadata"]
                or any(loaded["metadata"].get(name + "_sha256") != hashes[name]
                       for name in ("mapping", "protocol", "checkpoint"))):
            raise ValueError("Source training/cache identity or input hashes differ")
    text_cache = _numeric_npz(members["study/cache/text_features.npz"],
                              {"text_features", "candidate_mask"})
    texts, allowed = _common(text_cache["text_features"], text_cache["candidate_mask"], dev)
    _common(texts, allowed, train)
    if (training.get("candidate_mask_sha256") != _sha(allowed.tobytes())
            or training.get("text_features_sha256") != _sha(texts.tobytes())):
        raise ValueError("Source text vocabulary/cache hashes differ")
    runs = training.get("runs")
    if (not isinstance(runs, list) or len(runs) != 9
            or {(row.get("mode"), row.get("seed")) for row in runs}
            != {(mode, seed) for mode in MODES for seed in SEEDS}):
        raise ValueError("Source must contain exactly nine fixed head runs")
    states, recorded, frozen = {}, {}, {}
    for mode in MODES:
        prefix = f"study/training/frozen/{mode}/"
        frozen[mode] = {"metrics": _json(members[prefix + "metrics.json"]),
                        "ranks": _json(members[prefix + "known_positive_ranks.json"]),
                        "frames": _prediction_frames(members[prefix + "predictions.json"])}
        _recorded_frames_match_cache(frozen[mode]["frames"], dev)
        for seed in SEEDS:
            prefix = f"study/training/runs/{mode}/seed_{seed}/"
            metadata = _json(members[prefix + "metadata.json"])
            receipt = _json(members[prefix + "receipt.json"])
            _inventory(members, prefix, receipt)
            if (metadata.get("mode") != mode or metadata.get("seed") != seed
                    or metadata.get("optimizer_steps", 0) <= 0
                    or metadata.get("last_parameter_delta_l2", 0) <= 0
                    or metadata.get("last_embedding_delta_l2", 0) <= 0):
                raise ValueError("Missing source last-state learning evidence")
            loaded = {}
            for state in STATES:
                raw = members[prefix + f"{state}_head.safetensors"]
                description = metadata["checkpoints"][state]
                if description.get("sha256") != _sha(raw) or description.get("bytes") != len(raw):
                    raise ValueError("Source head checkpoint hash mismatch")
                values = load_tensors(raw)
                if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                       for value in values.values()):
                    raise ValueError("Source head tensors must be finite FP32")
                head = HOIResidualHead(dev["dimension"])
                head.load_state_dict(values, strict=True)
                if state == "initial" and (torch.count_nonzero(values["residual.2.weight"])
                                            or torch.count_nonzero(values["residual.2.bias"])):
                    raise ValueError("Initial head is not the fixed identity projection")
                loaded[state] = values
            states[mode, seed] = loaded
            recorded[mode, seed] = {"metadata": metadata,
                "metrics": _json(members[prefix + "metrics.json"]),
                "ranks": _json(members[prefix + "known_positive_ranks.json"]),
                "frames": _prediction_frames(members[prefix + "predictions.json"])}
            _recorded_frames_match_cache(recorded[mode, seed]["frames"], dev)
    source_config = _json(members["run_config.json"])
    if any(source_config.get(name + "_sha256") != value for name, value in hashes.items()):
        raise ValueError("Source run config differs from locked input hashes")
    source_receipt = {
        "schema_version": "vidhoi_pair_source_validation_v1", "status": "completed",
        "level": "synthetic_cached_pair_source_fixture" if test_fixture
        else "verified_real_cached_head_study_source", "test_fixture": test_fixture,
        "source_result_sha256": expected_sha, "source_result_bytes": path.stat().st_size,
        "zip_entries": len(members), "zip_expanded_bytes": sum(map(len, members.values())),
        "all_member_crc_and_stage_inventory_hashes_verified": True,
        "input_hashes": hashes, "config_sha256": cache_receipt["config_sha256"],
        "source_cache_receipt_sha256": _sha(members["study/cache/receipt.json"]),
        "source_training_receipt_sha256": _sha(members["study/training/receipt.json"]),
        "source_cache_level": cache_receipt.get("level"),
        "source_training_level": training.get("level"),
        "source_cache_test_fixture": cache_receipt.get("encoder", {}).get("test_fixture"),
        "num_head_states": 18, "files_extracted_or_executed": False,
        "official_final_split_accessed": False, "final_unseen_accessed": False,
    }
    return {"train_cache": train, "dev_cache": dev, "text_features": texts,
            "candidate_mask": allowed, "training_receipt": training,
            "source_env": _json(members["environment.json"]), "source_config": source_config,
            "source_receipt": source_receipt, "states": states, "recorded": recorded,
            "recorded_frozen": frozen}


def pair_budget_diagnostics(frames: list[VidHOIFrame], *, top_k: int = 100) -> dict:
    """Trace the unchanged global budget/matching rule for every annotated GT entry."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be positive")
    metrics = evaluate_frame_hoi(frames, top_k=top_k)
    supported = {tuple(row["triplet"]) for row in metrics["classes"]}
    records, frame_rows = [], []
    outcomes: Counter = Counter()
    total_matched = 0
    for frame in frames:
        by_identity = {}
        ranked = []
        for index, pair in enumerate(frame.predictions):
            key = (pair.subject_track_id, pair.object_track_id)
            if None in key or key in by_identity:
                raise ValueError("Diagnostic requires unique explicit actor/object proposal identities")
            by_identity[key] = index
            for predicate, value in enumerate(pair.predicate_values):
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Diagnostic scores must be finite/nonnegative")
                score = (_log_score(value) + _log_score(pair.subject_score)
                         + _log_score(pair.object_score))
                ranked.append((score, index, predicate))
        ranked.sort(reverse=True)
        ranks = {(index, predicate): rank for rank, (_, index, predicate) in enumerate(ranked, 1)}
        consumed, events, accepted = {}, {}, set()
        retained_unknown_slots = 0
        known_slots = set()
        truth_ids = set()
        for gt_index, truth in enumerate(frame.ground_truth):
            identity = (truth.subject_track_id, truth.object_track_id)
            if identity in truth_ids or identity not in by_identity:
                raise ValueError("Known GT pair has duplicate or absent canonical proposal identity")
            truth_ids.add(identity)
            prediction = frame.predictions[by_identity[identity]]
            if (truth.subject_class != prediction.subject_class
                    or truth.object_class != prediction.object_class
                    or tuple(truth.subject_box) != tuple(prediction.subject_box)
                    or tuple(truth.object_box) != tuple(prediction.object_box)):
                raise ValueError("Canonical actor/object identity geometry differs from GT")
            for predicate, present in enumerate(truth.predicate_values):
                if present == 1:
                    known_slots.add((by_identity[identity], predicate))
        for rank, (_, pair_index, predicate) in enumerate(ranked[:top_k], 1):
            prediction = frame.predictions[pair_index]
            key = (prediction.subject_class, predicate, prediction.object_class)
            retained_unknown_slots += (pair_index, predicate) not in known_slots
            best_overlap, best_index = 0.0, None
            if key not in supported:
                status = "class_without_GT_support_ignored"
            else:
                for gt_index, truth in enumerate(frame.ground_truth):
                    if (truth.subject_class != prediction.subject_class
                            or truth.object_class != prediction.object_class
                            or truth.predicate_values[predicate] != 1):
                        continue
                    overlap = min(bbox_iou(truth.subject_box, prediction.subject_box),
                                  bbox_iou(truth.object_box, prediction.object_box))
                    if overlap >= 0.5 and overlap > best_overlap:
                        best_overlap, best_index = overlap, gt_index
                if best_index is None:
                    status = "evaluator_no_supported_GT_match"
                elif best_index in consumed:
                    status = ("shared_pair_blocked_by_other_predicate"
                              if consumed[best_index]["predicate_id"] != predicate
                              else "duplicate_same_predicate_pair_consumed")
                else:
                    status = "evaluator_true_positive"
                    consumed[best_index] = {"predicate_id": predicate, "proposal_index": pair_index,
                                            "global_rank": rank}
                    accepted.add((best_index, predicate))
                    total_matched += 1
            outcomes[status] += 1
            events[pair_index, predicate] = {"outcome": status, "matched_gt_index": best_index}
        survived, gt_total = 0, 0
        for gt_index, truth in enumerate(frame.ground_truth):
            pair_index = by_identity[truth.subject_track_id, truth.object_track_id]
            for predicate, present in enumerate(truth.predicate_values):
                if present != 1:
                    continue
                rank = ranks[pair_index, predicate]
                survived += rank <= top_k
                gt_total += 1
                event = events.get((pair_index, predicate), {"outcome": "outside_top_k",
                                                            "matched_gt_index": None})
                records.append({"frame_key": frame.key, "gt_pair_index": gt_index,
                    "subject_track_id": truth.subject_track_id,
                    "object_track_id": truth.object_track_id,
                    "triplet": [truth.subject_class, predicate, truth.object_class],
                    "predicate_id": predicate, "canonical_proposal_index": pair_index,
                    "global_slot_rank": rank, "canonical_slot_inside_budget": rank <= top_k,
                    "canonical_slot_evaluator_outcome": event["outcome"],
                    "canonical_slot_matched_gt_index": event["matched_gt_index"],
                    "GT_entry_accepted_by_any_retained_prediction": (gt_index, predicate) in accepted,
                    "GT_pair_consumed_by": consumed.get(gt_index)})
        frame_rows.append({"frame_key": frame.key, "num_pairs": len(frame.predictions),
            "num_candidate_slots": len(ranked), "num_retained_slots": min(top_k, len(ranked)),
            "num_zero_score_slots": sum(value == 0 for pair in frame.predictions
                                         for value in pair.predicate_values),
            "num_observed_positive_entries": gt_total, "canonical_positive_slots_survived": survived,
            "canonical_positive_slots_outside_budget": gt_total - survived,
            "retained_unannotated_slots_not_verified_negative": retained_unknown_slots,
            "GT_pair_consumption_count": len(consumed)})
    evaluator_tp = sum(sum(row["sorted_tp"]) for row in metrics["classes"])
    if evaluator_tp != total_matched:
        raise ValueError("Budget/matching trace disagrees with the unchanged evaluator")
    return {"schema_version": "vidhoi_pair_budget_trace_v1", "scope": "annotated_oracle_GT_entries",
        "top_k": top_k, "top_k_before_class_filter": True, "iou_threshold": 0.5,
        "global_rank_ties": "score_desc_pair_index_desc_predicate_id_desc",
        "gt_matching_scope": "shared_pair_across_predicates_per_frame",
        "num_frames": len(frames), "num_gt_positive_entries": len(records),
        "num_canonical_positive_slots_survived": sum(row["canonical_slot_inside_budget"] for row in records),
        "num_canonical_positive_slots_outside_budget": sum(not row["canonical_slot_inside_budget"] for row in records),
        "num_GT_entries_accepted_by_any_retained_prediction": total_matched,
        "retained_prediction_outcomes": dict(sorted(outcomes.items())),
        "matching_trace_TP_count_equals_evaluator": True,
        "unannotated_is_not_verified_negative": True,
        "not_actor_correctness_accuracy": True, "frames": frame_rows, "records": records}


def _device(config: dict) -> str:
    if config.get("threads", 2) != 2:
        raise ValueError("Diagnostic keeps the fixed two-thread runtime")
    torch.set_num_threads(2)
    device = config.get("device", "cpu")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested diagnostic device is unavailable; no fallback")
    return device


def _head(source: dict, mode: str, seed: int, state: str, device: str) -> HOIResidualHead:
    if mode not in MODES or seed not in SEEDS or state not in STATES:
        raise ValueError("Only the fixed mode/seed/initial-last states are diagnostic inputs")
    head = HOIResidualHead(source["dev_cache"]["dimension"])
    head.load_state_dict(source["states"][mode, seed][state], strict=True)
    return head.eval().requires_grad_(False).to(device)


def score_pair_rows(
    source: dict, mode: str, seed: int, state: str, row_indices: list[int], device: str = "cpu",
    *, threads: int = 2, batch_size: int = 64,
) -> np.ndarray:
    """Score only requested *train-cache* proposal rows, including all-unknown rows.

    Output is FP64 ``[len(row_indices), predicates]`` with excluded slots zero.
    Different batching can cause small FP32 score differences from a full replay;
    this helper makes no claim of source-environment numerical reproduction.
    """
    device = _device({"device": device, "threads": threads})
    cache = source["train_cache"]
    if (not isinstance(row_indices, list) or not row_indices or type(batch_size) is not int
            or batch_size < 1 or any(type(value) is not int or not 0 <= value < cache["count"]
                                     for value in row_indices)):
        raise ValueError("Train row indices must be a nonempty bounded integer list")
    head = _head(source, mode, seed, state, device)
    texts, allowed = source["text_features"], source["candidate_mask"]
    values = np.zeros((len(row_indices), texts.shape[1]), dtype=np.float64)
    with torch.inference_mode():
        for offset in range(0, len(row_indices), batch_size):
            selected = np.asarray(row_indices[offset:offset + batch_size], dtype=np.int64)
            obj = torch.from_numpy(cache["object_features"][selected].copy()).to(device)
            union = torch.from_numpy(cache["union_features"][selected].copy()).to(device)
            projected = head(view_features(obj, union, mode))
            classes = cache["object_class"][selected]
            for object_class in np.unique(classes):
                rows = np.flatnonzero(classes == object_class)
                predicates = np.flatnonzero(allowed[object_class])
                prototypes = F.normalize(torch.from_numpy(texts[object_class, predicates].copy())
                                         .to(device), dim=-1)
                cosine = (projected[torch.as_tensor(rows, device=device)] @ prototypes.T)
                score = cosine.clamp(-1, 1).float().cpu().numpy().astype(np.float64)
                values[np.ix_(offset + rows, predicates)] = np.exp(score)
    return values


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _record(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": file_sha256(path)}


def _transitions(before: dict, after: dict) -> dict:
    def identity(row):
        return row["frame_key"], row["gt_pair_index"], row["predicate_id"]

    left, right = {identity(row): row for row in before["records"]}, {identity(row): row for row in after["records"]}
    if left.keys() != right.keys():
        raise ValueError("Initial/last diagnostics changed the canonical positive inventory")
    return {"positive_entries_lost_from_top_k": sum(
                left[key]["canonical_slot_inside_budget"] and not right[key]["canonical_slot_inside_budget"]
                for key in left),
            "positive_entries_gained_into_top_k": sum(
                not left[key]["canonical_slot_inside_budget"] and right[key]["canonical_slot_inside_budget"]
                for key in left),
            "GT_entries_lost_evaluator_TP": sum(
                left[key]["GT_entry_accepted_by_any_retained_prediction"]
                and not right[key]["GT_entry_accepted_by_any_retained_prediction"] for key in left),
            "GT_entries_gained_evaluator_TP": sum(
                not left[key]["GT_entry_accepted_by_any_retained_prediction"]
                and right[key]["GT_entry_accepted_by_any_retained_prediction"] for key in left)}


def run_pair_score_diagnostics(source: dict, config: dict, outdir: str | Path) -> dict:
    """Rescore eighteen fixed head states; no optimizer, encoder, tuning or final data."""
    fixture = config.get("test_fixture") is True
    validated = source["source_receipt"]
    if validated.get("test_fixture") is not fixture or validated.get("status") != "completed":
        raise ValueError("Source fixture/production scope differs from diagnostic config")
    if config.get("source_result_sha256") != validated["source_result_sha256"]:
        raise ValueError("Diagnostic source hash differs from the locked result")
    hashes = validated["input_hashes"]
    if config.get("source_input_hashes", hashes) != hashes:
        raise ValueError("Diagnostic source input hashes differ")
    for key in ("protocol", "mapping"):
        if config.get(key + "_sha256", hashes[key]) != hashes[key]:
            raise ValueError("Diagnostic protocol/mapping hash differs from source")
    if any(config.get(name, False) is not False for name in
           ("official_final_split_accessed", "final_unseen_accessed", "unknown_labels_treated_as_negative")):
        raise ValueError("Diagnostic cannot use final data or infer negative labels")
    device = _device(config)
    dev = validate_cache(source["dev_cache"], role="validation")
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _write(output / "source_receipt.json", validated)
    comparisons = []
    for mode in MODES:
        for seed in SEEDS:
            results = {}
            for state in STATES:
                head = _head(source, mode, seed, state, device)
                result = evaluate_head_state(head, dev, source["text_features"],
                                             source["candidate_mask"], mode, device=device)
                trace = pair_budget_diagnostics(result["frames"])
                directory = output / "states" / mode / f"seed_{seed}" / state
                _write(directory / "predictions.json", [asdict(frame) for frame in result["frames"]])
                _write(directory / "metrics.json", result["metrics"])
                _write(directory / "known_positive_ranks.json", result["known_positive_ranks"])
                _write(directory / "pair_budget_diagnostic.json", trace)
                results[state] = {"metrics": result["metrics"], "ranks": result["known_positive_ranks"],
                                  "trace": trace, "score_values": result["score_values"]}
                print(f"PAIR_HEAD_DIAGNOSED: {mode} seed={seed} state={state} "
                      f"map={result['metrics']['map']}", flush=True)
            initial, last = results["initial"], results["last"]
            recorded = source["recorded_frozen"][mode]
            comparisons.append({"mode": mode, "seed": seed,
                "recorded_source_frozen_map": recorded["metrics"]["map"],
                "current_initial_map": initial["metrics"]["map"],
                "current_last_map": last["metrics"]["map"],
                "within_current_environment_map_delta_pp": 100 * (last["metrics"]["map"] - initial["metrics"]["map"]),
                "recorded_source_vs_current_initial_map_delta_pp": 100 * (initial["metrics"]["map"] - recorded["metrics"]["map"]),
                "recorded_source_vs_current_initial_max_score_absolute_difference": float(np.max(
                    np.abs(initial["score_values"] - np.asarray([
                        pair.predicate_values for frame in recorded["frames"]
                        for pair in frame.predictions], dtype=np.float64)), initial=0)),
                "initial_mean_positive_rank": initial["ranks"]["mean_rank"],
                "last_mean_positive_rank": last["ranks"]["mean_rank"],
                "initial_budget": {key: value for key, value in initial["trace"].items() if key not in {"records", "frames"}},
                "last_budget": {key: value for key, value in last["trace"].items() if key not in {"records", "frames"}},
                "transitions": _transitions(initial["trace"], last["trace"])})
    aggregate = {}
    for mode in MODES:
        current = [row for row in comparisons if row["mode"] == mode]
        aggregate[mode] = {"seeds": list(SEEDS), "n": 3,
            **{key + "_mean": float(np.mean([row[key] for row in current])) for key in (
                "current_initial_map", "current_last_map", "within_current_environment_map_delta_pp",
                "initial_mean_positive_rank", "last_mean_positive_rank")},
            "current_last_map_std_sample": float(np.std([row["current_last_map"] for row in current], ddof=1)),
            "selected_best_seed": None}
    summary = {"schema_version": "vidhoi_pair_score_summary_v1", "comparisons": comparisons,
        "aggregate": aggregate, "recorded_source_environment": source["source_env"],
        "current_environment": {"device": device, "threads": 2,
                                "torch": torch.__version__, "numpy": np.__version__},
        "cross_environment_policy": "Recorded source metrics remain authoritative for their run. "
            "Initial and last are compared within this current execution environment. Cross-device "
            "score/tie/AP differences are reported, never relabeled exact reproduction.",
        "new_training_or_configuration_selection": False,
        "unannotated_is_not_verified_negative": True,
        "benchmark_replacement_claim": False}
    _write(output / "summary.json", summary)
    receipt = {"schema_version": "vidhoi_pair_score_diagnostics_v1", "status": "completed",
        "level": "synthetic_pair_score_diagnostic" if fixture
        else "cached_oracle_initial_last_pair_diagnostic", "test_fixture": fixture,
        "diagnosed_states": 18, "modes": list(MODES), "seeds": list(SEEDS), "states": list(STATES),
        "source_result_sha256": validated["source_result_sha256"],
        "source_input_hashes": hashes, "input_hashes": config.get("input_hashes", hashes),
        "source_config_sha256": validated["config_sha256"], "device": device, "threads": 2,
        "model_training_run": False, "backbone_inference_run": False, "head_inference_run": True,
        "official_final_split_accessed": False, "final_unseen_accessed": False,
        "unknown_labels_treated_as_negative": False, "full_benchmark_reproduction_claim": False,
        "elapsed_seconds": time.perf_counter() - started, "process_peak_rss_bytes": _peak_rss_bytes(),
        "files": [_record(path, output) for path in sorted(output.rglob("*")) if path.is_file()]}
    _write(output / "receipt.json", receipt)
    return receipt

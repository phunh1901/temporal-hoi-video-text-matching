"""Reject confounded global/local retrieval comparisons before reporting deltas."""

from __future__ import annotations


def validate_paired_runs(global_run, local_run):
    for run in (global_run, local_run):
        if run.get("status") != "PASS_TECHNICAL_PILOT":
            raise ValueError("Both paired runs must complete checkpoint inference")
    if global_run["baseline"] != "B0_frozen_openclip_global_mean":
        raise ValueError("Control must be the frozen global B0")
    if local_run["baseline"] != "B1_frozen_openclip_union_mean":
        raise ValueError("Treatment must be the frozen union B1")
    if local_run.get("box_source", {}).get("source_type") not in {"oracle", "detector"}:
        raise ValueError("B1 needs a verified explicit box source")
    if local_run["box_source"].get("full_frame_fallback") is not False:
        raise ValueError("Full-frame fallback is forbidden")
    if global_run["checkpoint"] != local_run["checkpoint"]:
        raise ValueError("Paired runs changed encoder/checkpoint")
    for key in ("manifest_lock_sha256", "manifest_sha256", "config_sha256", "source_videos",
                "clip_ids", "text_ids", "positive_pairs", "annotated_pairs"):
        if global_run["inputs"][key] != local_run["inputs"][key]:
            raise ValueError(f"Paired runs changed data/cohort/query pool: {key}")
    for key in ("seed", "batch_size", "num_workers", "precision", "device",
                "torch_num_threads", "torch_num_interop_threads", "versions"):
        if global_run["environment"][key] != local_run["environment"][key]:
            raise ValueError(f"Paired runs changed execution contract: {key}")
    if global_run["head"] != local_run["head"]:
        raise ValueError("Paired runs changed head")
    global_preprocess = {k: v for k, v in global_run["preprocess"].items() if k != "visual_view"}
    local_preprocess = {k: v for k, v in local_run["preprocess"].items() if k != "visual_view"}
    if global_preprocess != local_preprocess:
        raise ValueError("Paired runs changed sampling/pooling/normalization/preprocess")
    if global_run["coverage"] != local_run["coverage"]:
        raise ValueError("Paired runs have different coverage")
    for a, b in zip(global_run["clip_records"], local_run["clip_records"], strict=True):
        for key in ("clip_id", "split", "start_s", "end_s", "frames", "timestamps_s",
                    "video_metadata"):
            if a[key] != b[key]:
                raise ValueError(f"Paired runs changed frame cohort: {key}")


def paired_metric_deltas(global_run, local_run):
    validate_paired_runs(global_run, local_run)
    deltas = {}
    for scope in ("all_pilot", "dev", "extension"):
        deltas[scope] = {}
        for direction in ("video_to_text", "text_to_video"):
            a = global_run["metrics"][scope][direction]
            b = local_run["metrics"][scope][direction]
            deltas[scope][direction] = {
                metric: b[metric] - a[metric]
                if a[metric] is not None and b[metric] is not None else None
                for metric in ("R@1", "R@5", "R@10", "MRR", "MedR")
            }
    return deltas

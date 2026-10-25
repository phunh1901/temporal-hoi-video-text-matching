"""Run the frozen global-mean OpenCLIP pilot against the locked small manifests.

This is a technical pipeline pilot, not a VidHOI/MSR-VTT benchmark or a
violation detector. Only an explicitly supplied local checkpoint is accepted.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from temporal_hoi.evaluation.metrics import (
    compute_bidirectional_retrieval,
    compute_pairwise_metrics,
)
from temporal_hoi.evaluation.profiling import StageTimer, retrieval_error_groups

KNOWN_CHECKPOINT_SHA256 = "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows(features) -> np.ndarray:
    """Reject degenerate embeddings before computing cosine similarity."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Features must be a finite two-dimensional matrix")
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("Zero-norm embeddings cannot be normalized")
    return features / norms


def global_mean_features(frame_features) -> np.ndarray:
    """Normalize each valid frame, average them, then normalize the video."""
    frame_features = normalize_rows(frame_features)
    if not len(frame_features):
        raise ValueError("Global mean needs at least one valid frame")
    return normalize_rows(frame_features.mean(axis=0, keepdims=True))[0]


def relevance_mappings(clips, texts, relevance):
    """Use all and only explicit complete binary annotations in both directions."""
    clip_ids = [row["clip_id"] for row in clips]
    text_ids = [row["text_id"] for row in texts]
    if len(set(clip_ids)) != len(clip_ids) or len(set(text_ids)) != len(text_ids):
        raise ValueError("Duplicate clip or text IDs")
    ci, ti = {v: i for i, v in enumerate(clip_ids)}, {v: i for i, v in enumerate(text_ids)}
    labels = {}
    for row in relevance:
        key = (row["clip_id"], row["text_id"])
        if key in labels:
            raise ValueError("Duplicate relevance pair")
        if key[0] not in ci or key[1] not in ti or str(row["is_match"]) not in {"0", "1"}:
            raise ValueError("Unknown IDs or invalid binary label")
        labels[key] = int(row["is_match"])
    expected = {(clip_id, text_id) for clip_id in clip_ids for text_id in text_ids}
    if set(labels) != expected:
        raise ValueError("Pilot retrieval requires the entire candidate pool annotated")
    forward = [[ti[t] for t in text_ids if labels[c, t]] for c in clip_ids]
    reverse = [[ci[c] for c in clip_ids if labels[c, t]] for t in text_ids]
    negatives = [[ti[t] for t in text_ids if not labels[c, t]] for c in clip_ids]
    return forward, reverse, negatives


def evaluate_scopes(similarity, clips, texts, relevance, splits):
    forward, reverse, negatives = relevance_mappings(clips, texts, relevance)
    result = {"all_pilot": compute_bidirectional_retrieval(similarity, forward, reverse)}
    for split in sorted(set(splits.values())):
        rows = [i for i, clip in enumerate(clips) if splits[clip["clip_id"]] == split]
        reverse_subset = [
            [new_index for new_index, old_index in enumerate(rows) if old_index in positive]
            for positive in reverse
        ]
        result[split] = compute_bidirectional_retrieval(
            similarity[rows], [forward[i] for i in rows], reverse_subset
        )
        result[split]["clip_ids"] = [clips[i]["clip_id"] for i in rows]
        result[split]["text_candidate_pool"] = "all_locked_10_texts"
    pairs = [
        compute_pairwise_metrics(similarity[i], forward[i], negatives[i]) for i in range(len(clips))
    ]
    count = sum(row["num_pairs"] for row in pairs)
    result["pairwise"] = {
        "num_pairs": count,
        "pairwise_accuracy": sum(
            row["pairwise_accuracy"] * row["num_pairs"] for row in pairs if row["num_pairs"]
        )
        / count
        if count
        else None,
        "mean_margin": sum(
            row["mean_margin"] * row["num_pairs"] for row in pairs if row["num_pairs"]
        )
        / count
        if count
        else None,
        "per_clip": [{"clip_id": clip["clip_id"], **row} for clip, row in zip(clips, pairs)],
    }
    return result


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("load_percent", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("commit_limit", ctypes.c_ulonglong),
        ("available_commit", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set", ctypes.c_size_t),
        ("working_set", ctypes.c_size_t),
        ("quota_peak_paged_pool", ctypes.c_size_t),
        ("quota_paged_pool", ctypes.c_size_t),
        ("quota_peak_non_paged_pool", ctypes.c_size_t),
        ("quota_non_paged_pool", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    ]


def memory_snapshot():
    if os.name != "nt":
        return {"supported": False, "reason": "Windows memory counters only"}
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot query global memory status")
    counters = _ProcessMemory()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemory),
        ctypes.c_ulong,
    ]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError("Cannot query process memory counters")
    return {
        "supported": True,
        "total_physical_bytes": status.total_physical,
        "available_physical_bytes": status.available_physical,
        "available_commit_bytes": status.available_commit,
        "commit_limit_bytes": status.commit_limit,
        "load_percent": status.load_percent,
        "process_working_set_bytes": counters.working_set,
        "peak_process_working_set_bytes": counters.peak_working_set,
        "process_private_bytes": counters.private_usage,
        "peak_process_commit_bytes": counters.peak_pagefile_usage,
    }


class MemoryMonitor:
    """Observe system pressure without reading or changing other applications."""

    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            self.samples.append(memory_snapshot())
            self.stop_event.wait(0.5)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.samples.append(memory_snapshot())
        supported = [row for row in self.samples if row.get("supported")]
        if not supported:
            return {"supported": False}
        return {
            "samples": len(supported),
            "sampling_interval_seconds": 0.5,
            "minimum_available_physical_bytes": min(
                row["available_physical_bytes"] for row in supported
            ),
            "minimum_available_commit_bytes": min(
                row["available_commit_bytes"] for row in supported
            ),
            "peak_process_working_set_bytes": max(
                row["peak_process_working_set_bytes"] for row in supported
            ),
            "peak_process_commit_bytes": max(
                row["peak_process_commit_bytes"] for row in supported
            ),
            "note": "Windows OS peak counters; sampled system minima can miss sub-0.5s dips",
        }


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_locked_manifests(manifest_dir, lock_path, config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["profile"] != config["profile"] or lock["purpose"] != "technical_pilot_only":
        raise ValueError("Manifest lock and pilot config disagree")
    required = ["clips.csv", "texts.csv", "splits.csv", "relevance.csv", "sources.csv"]
    hashes = {}
    for name in required:
        hashes[name] = sha256_file(manifest_dir / name)
        if hashes[name] != lock["files"].get(name):
            raise ValueError(f"Manifest changed since the locked pilot: {name}")
    clips, texts, splits, relevance, sources = [_csv(manifest_dir / name) for name in required]
    if len(clips) != config["expected_clips"] or len(texts) != 10:
        raise ValueError("The small locked pilot must contain 8 clips and 10 texts")
    split_map = {row["clip_id"]: row["split"] for row in splits}
    if len(split_map) != len(splits) or set(split_map) != {row["clip_id"] for row in clips}:
        raise ValueError("Each clip must have exactly one split")
    if any(row["matching_pilot_eligible"] != "true" for row in clips):
        raise ValueError("A clip is not approved for this technical matching pilot")
    relevance_mappings(clips, texts, relevance)
    source_records = []
    for source in sources:
        path = Path(source["video_path"])
        digest = sha256_file(path)
        if digest != source["sha256"] or path.stat().st_size != int(source["bytes"]):
            raise ValueError(f"Raw source differs from the reviewed pilot: {path}")
        source_records.append(
            {"path": path.as_posix(), "sha256": digest, "bytes": path.stat().st_size}
        )
    raw_bytes = sum(row["bytes"] for row in source_records)
    if raw_bytes > config["max_raw_bytes"]:
        raise ValueError("Raw pilot exceeds the locked 50 MiB ceiling")
    return clips, texts, split_map, relevance, config, hashes, source_records


def run(args):
    started = time.perf_counter()
    timer = StageTimer()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "baseline_run.json"
    report = {
        "task_id": getattr(args, "task_id", "W04-04"),
        "baseline": "B1_frozen_openclip_union_mean"
        if getattr(args, "union_boxes", None) else "B0_frozen_openclip_global_mean",
        "scope": "technical_pilot_only",
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "command_argv": [sys.executable, *sys.argv],
        "independent_test_available": False,
        "benchmark_reproduction": False,
        "checkpoint_inference": False,
        "model_training": False,
        "official_eval_accesses": 0,
        "cloud_spend_usd": 0,
        "limitations": [
            "8 convenience clips from 5 videos; scene/action labels reviewed by AI, not a human",
            "No human-object boxes, interaction labels, event boundaries or violation ground truth",
            "Dev and extension are reported separately; neither is an independent benchmark test",
            "OpenCLIP LAION weights differ from the CLIP4Clip fine-tuned MSR-VTT reference",
            "Full-clip global means do not model temporal order or duration",
            "R@10 with 10 text candidates and R@10 with 8 video candidates saturate by construction",
        ],
    }
    run_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    monitor = MemoryMonitor()
    monitor.start()
    try:
        clips, texts, split_map, relevance, config, hashes, sources = load_locked_manifests(
            args.manifest_dir, args.manifest_lock, args.config
        )
        boxes = None
        if getattr(args, "union_boxes", None):
            from temporal_hoi.data.pair_crops import VerifiedUnionBoxes

            boxes = VerifiedUnionBoxes(args.union_boxes)
            report["box_source"] = {
                "path": args.union_boxes.as_posix(),
                "sha256": sha256_file(args.union_boxes),
                "source_type": boxes.source_type,
                "provenance": boxes.provenance,
                "full_frame_fallback": False,
            }
        source_hashes = {Path(row["path"]): row["sha256"] for row in sources}
        checkpoint = args.checkpoint.resolve(strict=True)
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != args.checkpoint_sha256:
            raise ValueError("Local checkpoint SHA256 does not match the explicitly expected hash")
        report.update(
            {
                "inputs": {
                    "manifest_dir": args.manifest_dir.as_posix(),
                    "manifest_lock": args.manifest_lock.as_posix(),
                    "manifest_lock_sha256": sha256_file(args.manifest_lock),
                    "manifest_sha256": hashes,
                    "config_sha256": sha256_file(args.config),
                    "config_path": args.config.as_posix(),
                    "source_videos": sources,
                    "raw_source_bytes": sum(row["bytes"] for row in sources),
                    "clip_ids": [row["clip_id"] for row in clips],
                    "text_ids": [row["text_id"] for row in texts],
                    "positive_pairs": sum(int(row["is_match"]) for row in relevance),
                    "annotated_pairs": len(relevance),
                },
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_hash,
                    "bytes": checkpoint.stat().st_size,
                    "model_name": "ViT-B-32",
                    "pretrained": "laion2b_s34b_b79k"
                    if checkpoint_hash == KNOWN_CHECKPOINT_SHA256
                    else "explicit_custom_checkpoint",
                    "strict_state_dict_loading": True,
                    "downloaded": False,
                    "frozen": True,
                },
            }
        )
        before = memory_snapshot()
        # A local commit headroom bound, not a published model hardware requirement.
        minimum_commit = 2 * checkpoint.stat().st_size + 512 * 1024 * 1024
        report["memory_before_model"] = before
        report["memory_policy"] = {
            "minimum_available_commit_before_load_bytes": minimum_commit,
            "basis": "2 x local checkpoint file bytes + 512 MiB transient headroom",
            "minimum_available_commit_between_batches_bytes": 256 * 1024 * 1024,
            "physical_ram_minimum_enforced": False,
            "note": "CPU batch1 can page; Windows free physical RAM alone is not a feasibility proof",
        }
        if before.get("supported") and before["available_commit_bytes"] < minimum_commit:
            report["status"] = "BLOCKED_MEMORY_HEADROOM"
            raise RuntimeError(
                "Insufficient available commit headroom for safe local model loading"
            )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import open_clip
        import torch

        from temporal_hoi.data.video_reader import VideoReader

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        report["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "batch_size": 1,
            "num_workers": 0,
            "precision": "float32",
            "seed": 42,
            "versions": {
                package: importlib.metadata.version(package)
                for package in [
                    "torch",
                    "torchvision",
                    "open-clip-torch",
                    "numpy",
                    "opencv-python-headless",
                    "pillow",
                    "safetensors",
                ]
            },
        }
        load_start = time.perf_counter()
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=str(checkpoint),
            device="cpu",
            precision="fp32",
            image_mean=CLIP_MEAN,
            image_std=CLIP_STD,
            image_interpolation="bicubic",
            image_resize_mode="shortest",
        )
        model.eval()
        model.requires_grad_(False)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        report["preprocess"] = {
            "reader": "VideoReader CFR strict distinct frames, interval [start_s,end_s)",
            "num_frames_per_clip": config["num_frames"],
            "interval": "entire locked clip; uniform 8 distinct frame indices",
            "decode_max_frame_side": config["max_frame_side"],
            "decode_downsampling": "OpenCV INTER_AREA when max side exceeds 320",
            "image_transform": str(preprocess),
            "resize": "bicubic shortest side 224 then center crop 224",
            "rgb_mean": list(CLIP_MEAN),
            "rgb_std": list(CLIP_STD),
            "tokenizer": "OpenCLIP ViT-B-32 SimpleTokenizer, context_length77",
            "pooling": "L2 each frame -> arithmetic mean all 8 valid frames -> L2 video",
            "text_normalization": "L2 final text features",
            "score": "cosine (no learned logit_scale; positive scaling does not change ranks)",
            "candidate_order": "locked CSV row order; stable tie break",
        }
        report["model_load_seconds"] = time.perf_counter() - load_start
        timer.seconds["model_load"] = report["model_load_seconds"]
        timer.calls["model_load"] = 1
        report["preprocess"]["visual_view"] = "person_object_union" if boxes else "global"
        report["head"] = {"name": "frozen_identity", "trainable_parameters": 0,
                          "learned_head_seconds": 0.0,
                          "note": "No learned head; cosine scoring timed separately"}
        report["vram"] = {"status": "not_applicable", "device": "cpu",
                          "peak_cuda_allocated_bytes": None,
                          "reason": "This run executes on CPU; CUDA memory is not allocated"}
        print(f"Loaded frozen local OpenCLIP in {report['model_load_seconds']:.2f}s", flush=True)
        inference_start = time.perf_counter()
        text_features = []
        video_features, all_frame_features, timestamps, clip_records = [], [], [], []
        with torch.inference_mode():
            for text in texts:
                with timer.measure("text_preprocess"):
                    tokens = tokenizer([text["prompt_text"]])
                with timer.measure("text_encoder"):
                    feature = model.encode_text(tokens).cpu().numpy()
                with timer.measure("pooling_normalization"):
                    text_features.append(normalize_rows(feature)[0])
            for clip in clips:
                current_memory = memory_snapshot()
                if (
                    current_memory.get("supported")
                    and current_memory["available_commit_bytes"] < 256 * 1024 * 1024
                ):
                    raise RuntimeError(
                        "Available commit fell below 256 MiB; stopping before next clip"
                    )
                clip_start = time.perf_counter()
                with timer.measure("decode"):
                    with VideoReader(clip["video_path"]) as reader:
                        frames, times = reader.read_interval_frames(
                            float(clip["start_s"]),
                            float(clip["end_s"]),
                            num_frames=config["num_frames"],
                            max_frame_side=config["max_frame_side"],
                        )
                        video_meta = reader.get_metadata()
                frame_features = []
                crop_bounds = []
                for frame, timestamp in zip(frames, times):
                    if boxes:
                        with timer.measure("union_crop"):
                            frame, bounds = boxes.crop(
                                frame, clip["clip_id"], timestamp, video_meta,
                                source_hashes[Path(clip["video_path"])],
                            )
                            crop_bounds.append(list(bounds))
                    with timer.measure("image_preprocess"):
                        tensor = preprocess(frame).unsqueeze(0)
                    with timer.measure("image_encoder"):
                        features = model.encode_image(tensor).cpu().numpy()
                    with timer.measure("pooling_normalization"):
                        frame_features.append(normalize_rows(features)[0])
                with timer.measure("pooling_normalization"):
                    frame_array = np.stack(frame_features)
                    video_features.append(global_mean_features(frame_array))
                all_frame_features.append(frame_array)
                timestamps.append(times)
                clip_records.append(
                    {
                        "clip_id": clip["clip_id"],
                        "split": split_map[clip["clip_id"]],
                        "start_s": float(clip["start_s"]),
                        "end_s": float(clip["end_s"]),
                        "frames": len(frames),
                        "timestamps_s": times,
                        "video_metadata": video_meta,
                        "crop_bounds_decoded_xyxy": crop_bounds if boxes else None,
                        "elapsed_seconds": time.perf_counter() - clip_start,
                    }
                )
                print(
                    f"Encoded {clip['clip_id']}: {len(frames)} frames in {clip_records[-1]['elapsed_seconds']:.2f}s",
                    flush=True,
                )
        if boxes:
            boxes.assert_complete()
            report["box_source"]["full_frame_union_count"] = boxes.full_frame_union_count
            report["box_source"]["full_frame_union_note"] = (
                "Verified pairs reaching all edges are accepted and have no local contrast; "
                "never generated as fallback boxes"
            )
        video_array, text_array = np.stack(video_features), np.stack(text_features)
        with timer.measure("similarity"):
            similarity = video_array @ text_array.T
        if not np.isfinite(similarity).all():
            raise ValueError("Nonfinite similarity scores")
        archive_path = output_dir / "embeddings.npz"
        np.savez_compressed(
            archive_path,
            video_features=video_array,
            text_features=text_array,
            frame_features=np.stack(all_frame_features),
            timestamps_s=np.asarray(timestamps),
            similarity=similarity,
            clip_ids=np.asarray([row["clip_id"] for row in clips]),
            text_ids=np.asarray([row["text_id"] for row in texts]),
        )
        with timer.measure("evaluation"):
            metrics = evaluate_scopes(similarity, clips, texts, relevance, split_map)
        predictions = []
        forward, reverse, _ = relevance_mappings(clips, texts, relevance)
        for i, clip in enumerate(clips):
            order = np.argsort(-similarity[i], kind="stable")
            predictions.append(
                {
                    "direction": "video_to_text",
                    "query_id": clip["clip_id"],
                    "split": split_map[clip["clip_id"]],
                    "ranked_candidates": [
                        {
                            "candidate_id": texts[j]["text_id"],
                            "rank": rank + 1,
                            "score": float(similarity[i, j]),
                            "is_positive": int(j) in forward[i],
                        }
                        for rank, j in enumerate(order)
                    ],
                }
            )
        for j, text in enumerate(texts):
            order = np.argsort(-similarity[:, j], kind="stable")
            predictions.append(
                {
                    "direction": "text_to_video",
                    "query_id": text["text_id"],
                    "ranked_candidates": [
                        {
                            "candidate_id": clips[i]["clip_id"],
                            "rank": rank + 1,
                            "score": float(similarity[i, j]),
                            "is_positive": int(i) in reverse[j],
                        }
                        for rank, i in enumerate(order)
                    ],
                }
            )
        prediction_path = output_dir / "predictions.json"
        prediction_path.write_text(
            json.dumps(predictions, indent=2, allow_nan=False), encoding="utf-8"
        )
        report.update(
            {
                "status": "PASS_TECHNICAL_PILOT",
                "checkpoint_inference": True,
                "coverage": {
                    "clips_requested": len(clips),
                    "clips_encoded": len(video_array),
                    "texts_requested": len(texts),
                    "texts_encoded": len(text_array),
                    "frames_encoded": sum(row["frames"] for row in clip_records),
                    "similarity_shape": list(similarity.shape),
                    "positive_labels_complete_for_candidate_pool": True,
                    "all_scores_finite": True,
                },
                "metrics": metrics,
                "scores_video_text": similarity.tolist(),
                "clip_records": clip_records,
                "retrieval_errors": retrieval_error_groups(
                    predictions, {row["clip_id"]: row["candidate_group"] for row in clips}
                ),
                "inference_seconds": time.perf_counter() - inference_start,
                "outputs": {
                    "embeddings": {
                        "path": archive_path.as_posix(),
                        "sha256": sha256_file(archive_path),
                    },
                    "predictions": {
                        "path": prediction_path.as_posix(),
                        "sha256": sha256_file(prediction_path),
                    },
                },
            }
        )
    except Exception as exc:
        if report["status"] == "RUNNING":
            report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["stage_timing"] = {
            "clock": "time.perf_counter; synchronous CPU",
            "stages": timer.summary(),
            "measured_stage_seconds": sum(timer.seconds.values()),
            "decode_includes": "OpenCV open/seek/decode, max-side resize and RGB PIL conversion",
            "preprocess_includes": "CLIP transforms; tokenizer timed separately",
            "learned_head_seconds": 0.0,
            "outside_stages": "imports, hashes, memory checks, output writes and other orchestration",
        }
        report["memory_monitor"] = monitor.finish()
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["total_seconds"] = time.perf_counter() - started
        run_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {report['status']} record: {run_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Local safetensors file; no download"
    )
    parser.add_argument("--checkpoint-sha256", default=KNOWN_CHECKPOINT_SHA256)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument(
        "--manifest-lock", type=Path, default=Path("outputs/experiments/pilot/data_review/pilot_manifest_lock.json")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_video_loading.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/pilot/baseline/b0"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--task-id", default="W04-04")
    parser.add_argument("--union-boxes", type=Path,
                        help="Verified original-frame person/object box manifest; no fallback")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("Use 1 to 4 CPU threads for this small low-memory pilot")
    run(args)


if __name__ == "__main__":
    main()

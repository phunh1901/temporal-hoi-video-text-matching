"""Small extraction probe; estimates only, never a benchmark score."""

from __future__ import annotations

import platform
import shutil
import time
from pathlib import Path

import torch
from data_pipeline import atomic_json, build_cache


def resource_probe(state):
    cfg, info = state["config"], state["model_info"]
    count = min(int(cfg.get("RESOURCE_PROBE_VIDEOS", 8)), len(state["test_records"]))
    if count < 1:
        raise ValueError("RESOURCE_PROBE_VIDEOS must be positive")
    device = state["device"]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    sample = build_cache(
        state["test_records"][:count],
        state["index"],
        state["model"],
        state["tokenizer"],
        info,
        Path(cfg["WORK_DIR"]) / "cache",
        max_frames=cfg["MAX_FRAMES"],
        max_words=cfg["MAX_WORDS"],
        device=device,
        encoder_batch=cfg["ENCODER_BATCH"],
        text_batch=cfg["TEXT_BATCH"],
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    records = state["test_records"] + (state["train_records"] or [])
    total_videos = len({r["video_id"] for r in records})
    dim = sample["text"].shape[-1]
    feature_bytes = 4 * dim * (len(records) + total_videos * cfg["MAX_FRAMES"])
    report = {
        "level": "small_extraction_probe_not_benchmark",
        "sample_videos": len(sample["video_ids"]),
        "sample_seconds": seconds,
        "target_videos": total_videos,
        "target_captions": len(records),
        "rough_video_extraction_hours": seconds / count * total_videos / 3600,
        "raw_feature_tensor_gib": feature_bytes / 2**30,
        "ram_caveat": "Tensor payload only; encoder, copies, Python objects and runtime add overhead",
        "time_caveat": "Small sample; cache hits/caption encoding/training/eval/setup vary",
        "free_disk_gib": shutil.disk_usage(cfg["WORK_DIR"]).free / 2**30,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
        if device.startswith("cuda")
        else None,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
        if device.startswith("cuda")
        else None,
    }
    atomic_json(Path(cfg["WORK_DIR"]) / "resource_probe.json", report)
    return report

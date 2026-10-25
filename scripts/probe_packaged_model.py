"""Measure native model resources on two fixed development clips, without scores."""

import argparse
import json
import time
from pathlib import Path


def probe(config):
    import torch
    from run_msrvtt_baseline import preflight

    from temporal_hoi.data.msrvtt import file_sha256
    from temporal_hoi.inference.clip4clip import (
        decode_video,
        load_component,
        load_model,
        text_tokens,
    )
    from temporal_hoi.training.resources import ensure_allocation

    config = dict(config, split="validation", max_run_hours=0.1)
    pool, readiness = preflight(config, probe_videos=2)
    if readiness["blockers"]:
        raise ValueError("Packaged model probe blocked: " + "; ".join(readiness["blockers"]))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU for the native model probe")
    free, total = torch.cuda.mem_get_info()
    if free < 3 * 1024**3:
        raise ValueError("Need 3 GiB free VRAM for the bounded model probe")
    torch.set_num_threads(config["threads"])
    started = time.monotonic()

    def guard(extra=0):
        return ensure_allocation({"max_rss_bytes": config["max_rss_bytes"],
                                  "max_train_hours": config["max_run_hours"]},
                                 started, extra, "packaged model resource probe")

    guard(Path(config["checkpoint"]).stat().st_size * 4)
    torch.cuda.reset_peak_memory_stats()
    model = load_model(config["source_root"], config["checkpoint"], "cuda")
    tokenizer = load_component(config["source_root"], "modules/tokenization_clip.py").SimpleTokenizer()
    transform = load_component(config["source_root"], "dataloaders/rawvideo_util.py").RawVideoExtractor(
        framerate=1, size=224).transform
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    records = []
    text_started = time.monotonic()
    with torch.inference_mode():
        for video_id in pool.video_ids:
            caption = next(row["caption"] for row in pool.queries if row["video_id"] == video_id)
            tokens = torch.from_numpy(text_tokens(tokenizer, caption)).cuda()
            feature = model.encode_text(tokens)
            if feature.shape != (1, 512) or not torch.isfinite(feature).all():
                raise ValueError("Invalid native text encoder output")
            guard()
        torch.cuda.synchronize()
        text_seconds = time.monotonic() - text_started
        video_started = time.monotonic()
        for video_id in pool.video_ids:
            path = Path(config["media_root"]) / f"{video_id}.mp4"
            frames, timestamps, indices = decode_video(path, transform, guard)
            for offset in range(0, len(frames), config["frame_batch_size"]):
                feature = model.encode_image(frames[offset:offset + config["frame_batch_size"]].cuda())
                if feature.shape[-1] != 512 or not torch.isfinite(feature).all():
                    raise ValueError("Invalid native image encoder output")
                guard()
            records.append({"video_id": video_id, "sha256": file_sha256(path),
                            "frame_indices": indices, "timestamps_s": timestamps})
            del frames
        torch.cuda.synchronize()
        video_seconds = time.monotonic() - video_started
    estimated = load_seconds + 2 * ((text_seconds + video_seconds) / len(pool.video_ids) * 1000)
    return {"gpu_name": torch.cuda.get_device_name(), "vram_total_bytes": total,
            "vram_free_before_bytes": free, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "rss_or_peak_bytes": guard(), "model_load_seconds": load_seconds,
            "text_seconds": text_seconds, "video_seconds": video_seconds,
            "wall_seconds": time.monotonic() - started, "decode_records": records,
            "estimated_1k_wall_seconds_with_factor2": estimated,
            "estimate_note": "Two development clips only; extrapolation is not a measured benchmark duration",
            "validation_quality_independent": False,
            "validation_note": "Claimed train9k includes internal validation; probe resources/function only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/msrvtt_clip4clip_packaged.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Require an existing output directory and a fresh result")
    report = {"schema_version": "native_model_resource_probe_v1", "status": "failed",
              "benchmark_inference_run": False, "benchmark_metrics": None}
    try:
        config = json.loads(args.config.read_text())
        report.update(probe(config), status="native_model_resource_probe_verified")
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))
    return 0 if report["status"] == "native_model_resource_probe_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())

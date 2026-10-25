"""Preflight or execute locked native CLIP4Clip meanP checkpoint evaluation.

Default mode performs metadata/file checks without loading a model. Execution
requires explicit flags and genuine weights/provenance; no automatic downloads.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from temporal_hoi.data.msrvtt import MSRVTTAdapter, evaluate_pool_scores, file_sha256
from temporal_hoi.inference.clip4clip import (
    UPSTREAM_COMMIT,
    decode_video,
    load_component,
    load_model,
    normalized_mean,
    text_tokens,
    verify_source,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def event(stage, **values):
    print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
                      **values}), flush=True)


def cached_feature(cache, identity, compute):
    """Atomic numeric-only feature cache bound to exact input/model/runtime identity."""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache) / f"{key}.npz" if cache else None
    if path and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            feature = saved["feature"]
            record = json.loads(str(saved["record"]))
            if str(saved["identity"]) != json.dumps(identity, sort_keys=True):
                raise ValueError("Feature cache identity mismatch")
        if feature.shape != (512,) or not np.isfinite(feature).all():
            raise ValueError("Invalid cached feature")
        return feature, record, True
    feature, record = compute()
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape != (512,) or not np.isfinite(feature).all():
        raise ValueError("Invalid feature before caching")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, feature=feature, record=json.dumps(record),
                                identity=json.dumps(identity, sort_keys=True))
        temporary.replace(path)
    return feature, record, False


def preflight(config, probe_videos=0):
    if (config.get("schema_version") != "clip4clip_meanp_inference_v1"
            or config.get("upstream_commit") != UPSTREAM_COMMIT
            or config.get("max_words") != 32 or config.get("max_frames") != 12
            or config.get("split") not in {"test", "validation"}):
        raise ValueError("Require reference commit/max_words/max_frames and test or validation")
    if (config.get("device") not in {"cpu", "cuda"}
            or not isinstance(config.get("frame_batch_size"), int)
            or config["frame_batch_size"] < 1
            or config.get("max_rss_bytes", 0) <= 0 or config.get("max_run_hours", 0) <= 0):
        raise ValueError("Invalid device, batching or resource budget")
    adapter = MSRVTTAdapter.from_files(config["metadata_archive"], config["metadata_lock"])
    pool = adapter.pool(config["split"])
    if probe_videos:
        if config["split"] != "validation" or not 1 <= probe_videos <= len(pool.video_ids):
            raise ValueError("Probe must select a valid number of validation videos")
        pool = adapter.pool("validation", selected_video_ids=pool.video_ids[:probe_videos])
    blockers = []
    try:
        verify_source(config["source_root"])
    except (OSError, ValueError) as exc:
        blockers.append(str(exc))
    media = Path(config["media_root"])
    missing = [vid for vid in pool.video_ids
               if not (media / f"{vid}.mp4").is_file()
               or (media / f"{vid}.mp4").stat().st_size == 0]
    if missing:
        blockers.append(f"Missing {len(missing)}/{len(pool.video_ids)} nonempty .mp4 files")
    checkpoint = config.get("checkpoint")
    expected = config.get("checkpoint_sha256")
    checkpoint_ok = False
    if not checkpoint or not Path(checkpoint).is_file():
        blockers.append("Missing verified native fine-tuned checkpoint")
    elif (not isinstance(expected, str) or len(expected) != 64
          or any(c not in "0123456789abcdef" for c in expected)):
        blockers.append("Require checkpoint SHA256, not an unknown/placeholder hash")
    elif file_sha256(Path(checkpoint)) != expected:
        blockers.append("Checkpoint SHA256 mismatch")
    else:
        checkpoint_ok = True
    provenance = config.get("checkpoint_provenance")
    if not provenance or not Path(provenance).is_file():
        blockers.append("Missing checkpoint source/training-regime provenance JSON")
    else:
        provenance = json.loads(Path(provenance).read_text(encoding="utf-8-sig"))
        required = {"source_url", "training_regime", "architecture", "checkpoint_sha256"}
        if (not isinstance(provenance, dict) or not required <= provenance.keys()
                or not all(provenance.get(k) for k in required)
                or provenance["architecture"] != "CLIP4Clip_ViT-B32_meanP_loose_2d"
                or provenance["checkpoint_sha256"] != expected):
            blockers.append("Checkpoint provenance missing/incompatible with native meanP")
    return pool, {"status": "blocked" if blockers else "files_present_not_execution_verified",
                  "blockers": blockers, "missing_video_ids": missing,
                  "checkpoint_hash_verified": checkpoint_ok, "pool_hashes": pool.hashes(),
                  "decode_checked": False, "model_loaded": False,
                  "benchmark_inference_run": False, "benchmark_metrics": None}


def execute(config, pool, output):
    import platform

    import torch

    from temporal_hoi.training.resources import ensure_allocation

    started = time.monotonic()
    budget = {"max_rss_bytes": config["max_rss_bytes"],
              "max_train_hours": config["max_run_hours"]}

    def guard(extra=0):
        return ensure_allocation(budget, started, extra, "baseline inference")

    guard(Path(config["checkpoint"]).stat().st_size * 4)
    torch.set_num_threads(config["threads"])
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    event("model_loading", device=device)
    model = load_model(config["source_root"], config["checkpoint"], device)
    event("model_loaded", texts=len(pool.text_ids), videos=len(pool.video_ids))
    tokenizer = load_component(config["source_root"], "modules/tokenization_clip.py").SimpleTokenizer()
    transform = load_component(config["source_root"], "dataloaders/rawvideo_util.py").RawVideoExtractor(
        framerate=1, size=224).transform
    guard((len(pool.text_ids) + len(pool.video_ids)) * 512 * 4
          + len(pool.text_ids) * len(pool.video_ids) * 24)
    text_features, video_features, decode_records = [], [], []
    cache = config.get("feature_cache_root")
    identity = {"checkpoint": config["checkpoint_sha256"], "pool": pool.hashes(),
                "source": verify_source(config["source_root"]), "dtype": "float32",
                "torch": str(torch.__version__), "device": device,
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "runner_sha256": file_sha256(Path(__file__)),
                "inference_sha256": file_sha256(Path(__file__).resolve().parents[1]
                                                / "src/temporal_hoi/inference/clip4clip.py"),
                "frame_batch_size": config["frame_batch_size"]}
    with torch.inference_mode():
        for number, row in enumerate(pool.queries, 1):
            guard()

            def encode_text():
                ids = torch.from_numpy(text_tokens(tokenizer, row["caption"])).to(device)
                feature = model.encode_text(ids).float()
                norm = feature.norm(dim=-1, keepdim=True)
                if not torch.isfinite(feature).all() or (norm <= 0).any():
                    raise ValueError("Nonfinite/zero text feature")
                return (feature / norm).cpu().numpy()[0], {}

            feature, _, reused = cached_feature(cache, dict(identity, kind="text",
                                                            caption=row["caption"]), encode_text)
            text_features.append(feature)
            if number % 25 == 0 or number == len(pool.queries):
                event("text_encoding", done=number, total=len(pool.queries),
                      last_cache_hit=reused, elapsed_s=round(time.monotonic() - started, 1))
        for number, vid in enumerate(pool.video_ids, 1):
            guard(12 * 3 * 224 * 224 * 4 * 4)
            path = Path(config["media_root"]) / f"{vid}.mp4"
            media_hash = file_sha256(path)
            event("video_start", done=number - 1, total=len(pool.video_ids), video_id=vid)

            def encode_video():
                frames, timestamps, indices = decode_video(path, transform, guard)
                features = []
                for offset in range(0, len(frames), config["frame_batch_size"]):
                    guard()
                    batch = frames[offset:offset + config["frame_batch_size"]].to(device)
                    features.append(model.encode_image(batch).float().cpu())
                return normalized_mean(torch.cat(features)).numpy(), {
                    "video_id": vid, "sha256": media_hash,
                    "frame_indices": indices, "timestamps_s": timestamps}

            feature, record, reused = cached_feature(
                cache, dict(identity, kind="video", video_id=vid, media_sha256=media_hash),
                encode_video)
            video_features.append(feature)
            decode_records.append(record)
            event("video_encoded", done=number, total=len(pool.video_ids), video_id=vid,
                  cache_hit=reused, elapsed_s=round(time.monotonic() - started, 1),
                  rss_bytes=guard(), cuda_peak_bytes=torch.cuda.max_memory_allocated()
                  if device == "cuda" else None)
        text = np.asarray(text_features, dtype=np.float32)
        video = np.asarray(video_features, dtype=np.float32)
        scale = float(model.logit_scale.exp().cpu())
    guard(len(pool.text_ids) * len(pool.video_ids) * 24)
    event("scoring", texts=len(pool.text_ids), videos=len(pool.video_ids))
    scores = scale * (text @ video.T)
    if not np.isfinite(scores).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Nonfinite inference scores or logit scale")
    score_path = output / "scores.npz"
    np.savez_compressed(score_path, scores=scores, text_ids=np.asarray(pool.text_ids),
                        video_ids=np.asarray(pool.video_ids))
    provenance = {"model_name": "native_CLIP4Clip_ViT-B32_meanP",
                  "checkpoint_sha256": config["checkpoint_sha256"],
                  "checkpoint_source": json.loads(Path(config["checkpoint_provenance"]).read_text(
                      encoding="utf-8-sig")),
                  "upstream_commit": UPSTREAM_COMMIT, "source_hashes": verify_source(config["source_root"]),
                  "preprocess": "native int(fps) 1Hz; uniform12; RGB bicubic short224 center224; CLIP norm",
                  "tokenizer": "native BPE, start/text/end padded to32; raw locked caption",
                  "pooling": "frame_L2_valid_mean_video_L2; normalized text; exp(logit_scale)",
                  "score_type": "scaled_cosine", "training_regime": "provided_finetuned_checkpoint",
                  "artifact_kind": "new_model_inference", "dtype": "float32",
                  "precision_note": "FP32 execution; report numeric differences versus upstream device/dtype",
                  "scores_sha256": file_sha256(score_path), "pool_hashes": pool.hashes(),
                  "decode_records": decode_records, "config": config,
                  "python": platform.python_version(), "torch": torch.__version__,
                  "wall_seconds": time.monotonic() - started, "rss_or_peak_bytes": guard(),
                  "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
    write_json(output / "provenance.json", provenance)
    metrics = evaluate_pool_scores(pool, scores, text_ids=pool.text_ids, video_ids=pool.video_ids)
    write_json(output / "metrics.json", metrics)
    event("metrics_saved", output=str(output))
    return {"status": "completed_checkpoint_inference", "benchmark_inference_run": True,
            "reference_reproduction_verified": False, "comparison_note": "Requires reference comparison",
            "scope": metrics["scope"], "scores_sha256": file_sha256(score_path),
            "wall_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/msrvtt_clip4clip_inference.json"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-final-evaluation", action="store_true")
    parser.add_argument("--probe-videos", type=int, default=0,
                        help="First N validation videos only; never a test tuning subset")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.source_root:
        config["source_root"] = str(args.source_root)
    if args.probe_videos:
        if args.probe_videos < 1:
            parser.error("--probe-videos must be positive")
        config["split"] = "validation"
    output = args.output_dir or Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["readiness.json", "run.json", "scores.npz",
                                                "provenance.json", "metrics.json"]):
        parser.error("Output contains a previous run; choose a new directory")
    result = None
    try:
        pool, readiness = preflight(config, args.probe_videos)
        write_json(output / "readiness.json", readiness)
        if not args.execute:
            print(json.dumps({"status": readiness["status"], "blockers": readiness["blockers"]}))
            return 2 if readiness["blockers"] else 0
        if readiness["blockers"]:
            raise ValueError("Preflight blocked: " + "; ".join(readiness["blockers"]))
        if config["split"] == "test" and (not args.allow_final_evaluation
                                              or config.get("selection_fixed_before_scores") is not True):
            raise ValueError("Test requires fixed selection assertion and --allow-final-evaluation")
        result = execute(config, pool, output)
        if args.probe_videos:
            result["benchmark_inference_run"] = False
            result["status"] = "completed_validation_subset_probe"
        write_json(output / "run.json", result)
        print(json.dumps(result))
        return 0
    except Exception as exc:
        write_json(output / "run.json", {"status": "failed", "benchmark_inference_run": False,
                                         "benchmark_metrics": None, "error": str(exc)})
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline frozen encoder caches, exact caption identities and bounded frame shards."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from temporal_hoi.data.msrvtt import MSRVTTAdapter, file_sha256, semantic_hash
from temporal_hoi.data.video_reader import VideoReader
from temporal_hoi.training.alignment import validate_config, write_json
from temporal_hoi.training.features import encoder_descriptor
from temporal_hoi.training.resources import enforce_resources


def normalize(features):
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise ValueError("Encoder features must be finite nonzero vectors")
    return values / norms


def cache_split(config, split, *, boxes_path=None, cohort_lock=None, allow_subset=False):
    validate_config(config)
    if split not in {"train", "validation"}:
        raise ValueError("Cache only locked development train/validation, never train9k/v1/test")
    adapter = MSRVTTAdapter.from_files(config["metadata_archive"], config["metadata_lock"])
    def blocked(reason, missing_ids, expected_videos):
        report = {"schema_version": "cache_readiness_v1", "status": "blocked", "split": split,
                  "missing_ids": missing_ids, "expected_videos": expected_videos,
                  "benchmark_training_run": False, "cache_created": False, "reason": reason}
        write_json(Path(config["output_dir"]).parent / f"cache_{split}_readiness.json", report)
        return report
    coverage = adapter.media_coverage(split, config["media_root"])
    if not coverage["present_nonempty_files"]:
        return blocked("Real MSR-VTT media is absent; metadata is not video features",
                       coverage["missing_video_ids"], coverage["expected_videos"])
    selection, cohort_sha = None, None
    if cohort_lock is not None:
        if not Path(cohort_lock).is_file():
            return blocked("The configured comparison cohort lock is absent",
                           coverage["missing_video_ids"], coverage["expected_videos"])
        cohort = json.loads(Path(cohort_lock).read_text(encoding="utf-8-sig"))
        selection = cohort["splits"][split]["video_ids"]
        cohort_sha = file_sha256(cohort_lock)
        if not allow_subset:
            raise ValueError("A comparison cohort requires explicit --allow-subset")
    pool = adapter.pool(split, selected_video_ids=selection)
    if len(pool.video_ids) != pool.full_video_count and not allow_subset:
        raise ValueError("Subset cache requires explicit --allow-subset")
    media = Path(config["media_root"])
    missing = [v for v in pool.video_ids if not (media / f"{v}.mp4").is_file()
               or not (media / f"{v}.mp4").stat().st_size]
    if missing:
        return blocked("Real MSR-VTT media is incomplete; metadata is not video features",
                       missing, len(pool.video_ids))
    if config["crop_mode"] == "union" and (boxes_path is None or not Path(boxes_path).is_file()):
        return blocked("Verified person/object union boxes are absent; no full-frame fallback",
                       [], len(pool.video_ids))
    encoder_run = json.loads(Path(config["encoder_run_json"]).read_text(encoding="utf-8-sig"))
    encoder = encoder_descriptor(config, encoder_run)
    checkpoint = Path(encoder_run["checkpoint"]["path"])
    if file_sha256(checkpoint) != encoder["checkpoint_sha256"]:
        raise ValueError("Local checkpoint differs from frozen encoder run; no substitute/download allowed")
    boxes = None
    if config["crop_mode"] == "union":
        if cohort_lock is None:
            raise ValueError("Union comparison requires the identical global/local --cohort-lock")
        from temporal_hoi.data.pair_crops import VerifiedUnionBoxes
        boxes = VerifiedUnionBoxes(boxes_path)
    cache = Path(config["cache_dir"])
    manifest_path = cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "schema_version": "msrvtt_frozen_cache_v1", "status": "building", "dataset": "MSR-VTT",
        "protocol_lock_sha256": file_sha256(config["metadata_lock"]), "encoder": encoder,
        "cohort_lock_sha256": cohort_sha, "splits": {},
    }
    if (manifest["encoder"] != encoder or manifest["cohort_lock_sha256"] != cohort_sha
            or manifest["protocol_lock_sha256"] != file_sha256(config["metadata_lock"])):
        raise ValueError("Existing cache belongs to another encoder/preprocess/split/cohort")
    if split in manifest["splits"]:
        raise ValueError("Split cache already exists; use another cache directory to avoid overwriting")
    import open_clip
    started = time.monotonic()
    torch.set_num_threads(config["threads"])
    enforce_resources(config, started)
    model, _, preprocess = open_clip.create_model_and_transforms(
        encoder["model_name"], pretrained=str(checkpoint), device="cpu", precision="fp32")
    if str(preprocess) != encoder_run["preprocess"]["image_transform"]:
        raise ValueError("Installed encoder image transform differs from the frozen baseline run")
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(encoder["model_name"])
    peak_rss = enforce_resources(config, started)
    cache.mkdir(parents=True, exist_ok=True)
    shards, dimension = [], None
    for start in range(0, len(pool.video_ids), config["shard_videos"]):
        ids = pool.video_ids[start:start + config["shard_videos"]]
        selected = set(ids)
        queries = [q for q in pool.queries if q["video_id"] in selected]
        frame_rows, timestamp_rows, media_hashes = [], [], {}
        for video_id in ids:
            path = media / f"{video_id}.mp4"
            source_sha = file_sha256(path)
            media_hashes[video_id] = source_sha
            with VideoReader(path) as reader:
                frames, times = reader.read_interval_frames(
                    0, reader.duration_s, config["frames"], config["decode_max_frame_side"])
                metadata = reader.get_metadata()
                encoded = []
                for frame, timestamp in zip(frames, times, strict=True):
                    if boxes is not None:
                        frame, _ = boxes.crop(frame, video_id, timestamp, metadata, source_sha)
                    with torch.inference_mode():
                        encoded.append(model.encode_image(preprocess(frame).unsqueeze(0)).numpy()[0])
                frame_rows.append(normalize(np.asarray(encoded)))
                timestamp_rows.append(times)
            peak_rss = max(peak_rss, enforce_resources(config, started))
        text_rows = []
        with torch.inference_mode():
            for text_start in range(0, len(queries), config["text_batch_size"]):
                rows = queries[text_start:text_start + config["text_batch_size"]]
                text_rows.append(normalize(model.encode_text(tokenizer([q["caption"] for q in rows])).numpy()))
                peak_rss = max(peak_rss, enforce_resources(config, started))
        frames = np.asarray(frame_rows, dtype=np.float32)
        text = np.concatenate(text_rows)
        dimension = frames.shape[-1]
        if text.shape[1] != dimension:
            raise ValueError("Encoder text/image output dimensions differ")
        relative = f"{split}_{start // config['shard_videos']:04d}.npz"
        np.savez_compressed(cache / relative, video_ids=np.asarray(ids), frame_embeddings=frames,
                            valid_frame_mask=np.ones(frames.shape[:2], dtype=bool),
                            timestamps_s=np.asarray(timestamp_rows, dtype=np.float64),
                            text_ids=np.asarray([q["text_id"] for q in queries]),
                            text_video_ids=np.asarray([q["video_id"] for q in queries]),
                            text_embeddings=text)
        shards.append({"path": relative, "sha256": file_sha256(cache / relative),
                       "videos": len(ids), "texts": len(queries), "media_sha256": media_hashes})
    if boxes is not None:
        boxes.assert_complete()
    manifest["dimension"] = dimension
    manifest["splits"][split] = {
        "video_ids": list(pool.video_ids), "pool_hashes": pool.hashes(),
        "raw_caption_sha256": semantic_hash(list(pool.queries)), "shards": shards,
        "source_group_sha256": semantic_hash([adapter.source_urls[v] for v in pool.video_ids]),
        "boxes_sha256": file_sha256(boxes_path) if boxes is not None else None,
        "box_coverage_verified": boxes is not None, "box_source": boxes.source_type if boxes else None,
        "scope": "full_locked_pool" if len(pool.video_ids) == pool.full_video_count else "subset",
        "observed_rss_bytes": peak_rss, "elapsed_seconds": time.monotonic() - started,
    }
    manifest["status"] = "ready" if set(manifest["splits"]) == {"train", "validation"} else "building"
    write_json(manifest_path, manifest)
    return {"status": manifest["status"], "split": split, "videos": len(pool.video_ids),
            "texts": len(pool.text_ids), "dimension": dimension, "manifest": str(manifest_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/frozen_feature_alignment.json"))
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--crop-mode", choices=["global", "union"])
    parser.add_argument("--boxes", type=Path)
    parser.add_argument("--cohort-lock", type=Path)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.crop_mode is not None:
        config["crop_mode"] = args.crop_mode
    if args.crop_mode == "union" and config["cache_dir"] == "data/features/msrvtt_global_v1":
        config["cache_dir"] = "data/features/msrvtt_union_v1"
    configured_boxes = config.get("boxes")
    if isinstance(configured_boxes, dict):
        configured_boxes = configured_boxes.get(args.split)
    result = cache_split(config, args.split, boxes_path=args.boxes or configured_boxes,
                         cohort_lock=args.cohort_lock or config.get("cohort_lock"),
                         allow_subset=args.allow_subset or config["allow_subset"])
    summary = ({"status": "blocked", "split": args.split, "reason": result["reason"],
                "missing_video_count": len(result["missing_ids"]), "cache_created": False,
                "readiness_path": str(Path(config["output_dir"]).parent / f"cache_{args.split}_readiness.json")}
               if result["status"] == "blocked" else result)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())

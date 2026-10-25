"""Frozen CLIP4Clip frame-cache creation and identified B0/B1/B2 evaluation.

Nothing is downloaded or trained. Cache arrays are text[T,D], frames[V,F,D]
and mask[V,F] (True means a decoded frame). Scores are float32[T,V] scaled
by the explicitly recorded encoder exp(logit_scale), never probabilities.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any
from zipfile import ZipFile, is_zipfile

import numpy as np
import torch

from temporal_hoi.data.msrvtt import MSRVTTAdapter, file_sha256
from temporal_hoi.inference import clip4clip
from temporal_hoi.inference.retrieval_pooling import (
    aligned_original_metrics,
    compare_runs,
    direct_meanp_reference,
    evaluate_bidirectional,
    score_matrix,
    validate_cache,
)
from temporal_hoi.training.resources import ensure_allocation

SCHEMA = "msrvtt_frozen_pooling_v1"
CACHE_KEYS = ("text", "frames", "mask", "text_video_ids", "video_ids", "text_ids")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    """Write a task-owned result; paths are never selected from input payloads."""
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _checked_file(path: str | Path | None, expected: str | None, label: str) -> Path:
    if not path or not Path(path).is_file():
        raise ValueError(f"Missing {label} file")
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(f"Require a lowercase SHA256 for {label}")
    result = Path(path)
    if file_sha256(result) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return result


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA or config.get("split") not in {"test", "validation"}:
        raise ValueError("Require msrvtt_frozen_pooling_v1 and split test or validation")
    for key in ("query_chunk", "video_chunk", "threads", "frame_batch_size", "max_cache_bytes"):
        if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("tau", "logit_scale", "max_rss_bytes", "max_run_hours"):
        value = config.get(key)
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if not isinstance(config.get("alpha"), (float, int)) or not 0 <= config["alpha"] <= 1:
        raise ValueError("alpha must be in [0,1]")
    for key in ("rtol", "atol"):
        if not isinstance(config.get(key), (float, int)) or not 0 <= config[key] < 0.01:
            raise ValueError(f"{key} must be a finite tolerance below 0.01")
    if config.get("device") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not isinstance(config.get("bootstrap_samples"), int) or config["bootstrap_samples"] < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not isinstance(config.get("seed"), int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")


def _guard(config: dict[str, Any], started: float, additional: int = 0) -> int:
    return ensure_allocation({"max_rss_bytes": config["max_rss_bytes"],
                              "max_train_hours": config["max_run_hours"]},
                             started, additional, "frozen pooling")


def _check_cache_ids(cache: dict[str, Any], pool: Any) -> None:
    if (cache["video_ids"] != list(pool.video_ids)
            or cache["text_ids"] != list(pool.text_ids)
            or cache["text_video_ids"] != [row["video_id"] for row in pool.queries]):
        raise ValueError("Feature-cache IDs/order differ from locked metadata pool")


def load_feature_cache(path: Path, expected_sha256: str, *, max_bytes: int) -> dict[str, Any]:
    """Load explicit SHA-checked .npz or .pt; object pickle execution is disabled.

    NPZ stores six cache arrays and optional scalar UTF-8 JSON `metadata`.
    PT supports the existing tensor/primitive-only Kaggle cache contract using
    weights_only=True. It never retries with unrestricted pickle loading.
    """
    path = _checked_file(path, expected_sha256, "feature cache")
    if path.stat().st_size > max_bytes:
        raise ValueError("Feature cache exceeds configured byte budget")
    if is_zipfile(path):
        with ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > max_bytes:
                raise ValueError("Expanded feature cache exceeds configured byte budget")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            cache = {key: arrays[key] for key in CACHE_KEYS}
            for key in ("text_video_ids", "video_ids", "text_ids"):
                if cache[key].dtype.kind not in "US" or cache[key].ndim != 1:
                    raise ValueError(f"{key} must be a plain string vector")
                cache[key] = cache[key].tolist()
            cache["metadata"] = json.loads(str(arrays["metadata"])) if "metadata" in arrays else {}
    elif path.suffix.lower() == ".pt":
        cache = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("Feature cache must be .npz or .pt")
    if not isinstance(cache, dict) or not isinstance(cache.get("metadata", {}), dict):
        raise ValueError("Feature cache and metadata must be dictionaries")
    return validate_cache(cache)


def save_feature_cache(path: Path, cache: dict[str, Any]) -> None:
    """Save numeric/string-only NPZ; refuse to replace an earlier cache."""
    if path.exists():
        raise FileExistsError(f"Refusing to replace feature cache: {path}")
    checked = validate_cache(cache)
    arrays = {key: checked[key].detach().cpu().numpy() for key in ("text", "frames", "mask")}
    arrays.update({key: np.asarray(checked[key], dtype=str)
                   for key in ("text_video_ids", "video_ids", "text_ids")})
    arrays["metadata"] = np.asarray(json.dumps(_jsonable(checked.get("metadata", {})),
                                                sort_keys=True, allow_nan=False))
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def preflight(config: dict[str, Any], *, build_cache: bool = False) -> tuple[Any, dict[str, Any]]:
    """Metadata and byte/hash checks only; no decoder, encoder or scoring call."""
    validate_config(config)
    pool = MSRVTTAdapter.from_files(config["metadata_archive"], config["metadata_lock"]).pool(
        config["split"])
    blockers = []
    if config.get("require_external_reference", True) or config.get("baseline_reference"):
        try:
            _checked_file(config.get("baseline_reference"), config.get("reference_sha256"),
                          "baseline reference")
        except (OSError, ValueError) as exc:
            blockers.append(str(exc))
    if build_cache:
        try:
            clip4clip.verify_source(config["source_root"])
            _checked_file(config.get("checkpoint"), config.get("checkpoint_sha256"), "checkpoint")
            provenance = json.loads(Path(config["checkpoint_provenance"]).read_text(
                encoding="utf-8-sig"))
            if (provenance.get("architecture") != "CLIP4Clip_ViT-B32_meanP_loose_2d"
                    or provenance.get("checkpoint_sha256") != config["checkpoint_sha256"]
                    or not provenance.get("source_url") or not provenance.get("training_regime")):
                raise ValueError("Checkpoint provenance is incompatible or incomplete")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            blockers.append(str(exc))
        missing = [vid for vid in pool.video_ids
                   if not (Path(config["media_root"]) / f"{vid}.mp4").is_file()
                   or not (Path(config["media_root"]) / f"{vid}.mp4").stat().st_size]
        if missing:
            blockers.append(f"Missing {len(missing)} media files")
    else:
        try:
            _checked_file(config.get("feature_cache"), config.get("feature_cache_sha256"),
                          "feature cache")
        except (OSError, ValueError) as exc:
            blockers.append(str(exc))
        missing = []
    return pool, {"status": "blocked" if blockers else "files_verified_not_executed",
                  "blockers": blockers, "pool_hashes": pool.hashes(),
                  "num_queries": len(pool.text_ids), "num_candidates": len(pool.video_ids),
                  "missing_video_ids": missing, "model_loaded": False,
                  "benchmark_inference_run": False, "benchmark_metrics": None}


def build_frame_cache(config: dict[str, Any], pool: Any, *, started: float) -> dict[str, Any]:
    """Encode all locked captions and at most 12 valid frames/video in FP32.

    Uses the existing hash-pinned native tokenizer/decoder/model loader. No
    malformed video is omitted and no zero/black replacement frame is created.
    """
    _guard(config, started, Path(config["checkpoint"]).stat().st_size * 4)
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no device substitution")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    source_hashes = clip4clip.verify_source(config["source_root"])
    _checked_file(config["checkpoint"], config["checkpoint_sha256"], "checkpoint")
    model = clip4clip.load_model(config["source_root"], config["checkpoint"], device)
    model.requires_grad_(False)
    tokenizer = clip4clip.load_component(config["source_root"],
                                         "modules/tokenization_clip.py").SimpleTokenizer()
    transform = clip4clip.load_component(config["source_root"],
                                         "dataloaders/rawvideo_util.py").RawVideoExtractor(
                                             framerate=1, size=224).transform
    total_bytes = (len(pool.text_ids) + len(pool.video_ids) * 12) * 512 * 4
    _guard(config, started, total_bytes)
    if total_bytes > config["max_cache_bytes"]:
        raise MemoryError("Projected frame cache exceeds max_cache_bytes")
    text = torch.empty((len(pool.text_ids), 512), dtype=torch.float32)
    frames = torch.zeros((len(pool.video_ids), 12, 512), dtype=torch.float32)
    mask = torch.zeros((len(pool.video_ids), 12), dtype=torch.bool)
    records = []
    with torch.inference_mode():
        for index, row in enumerate(pool.queries):
            _guard(config, started)
            tokens = torch.from_numpy(clip4clip.text_tokens(tokenizer, row["caption"])).to(device)
            text[index] = model.encode_text(tokens).float().cpu()[0]
        for index, video_id in enumerate(pool.video_ids):
            _guard(config, started, 12 * 3 * 224 * 224 * 4 * 3)
            path = Path(config["media_root"]) / f"{video_id}.mp4"
            video_hash = file_sha256(path)
            pixels, timestamps, indices = clip4clip.decode_video(
                path, transform, lambda: _guard(config, started))
            count = len(pixels)
            for offset in range(0, count, config["frame_batch_size"]):
                _guard(config, started)
                end = min(count, offset + config["frame_batch_size"])
                frames[index, offset:end] = model.encode_image(
                    pixels[offset:end].to(device)).float().cpu()
            mask[index, :count] = True
            records.append({"video_id": video_id, "sha256": video_hash,
                            "frame_indices": indices, "timestamps_s": timestamps})
            print(json.dumps({"stage": "frame_cache", "done": index + 1,
                              "total": len(pool.video_ids)}), flush=True)
        scale = float(model.logit_scale.exp().cpu())
    if not np.isclose(scale, config["logit_scale"], rtol=1e-6, atol=1e-6):
        raise ValueError("Native checkpoint logit_scale differs from explicit config")
    cache = validate_cache({"text": text, "frames": frames, "mask": mask,
                            "video_ids": list(pool.video_ids), "text_ids": list(pool.text_ids),
                            "text_video_ids": [q["video_id"] for q in pool.queries],
                            "metadata": {"schema": "msrvtt_frame_cache_v1", "split": pool.split,
                                         "pool_hashes": pool.hashes(), "logit_scale": scale,
                                         "checkpoint_sha256": config["checkpoint_sha256"],
                                         "checkpoint_provenance": json.loads(Path(
                                             config["checkpoint_provenance"]).read_text(
                                                 encoding="utf-8-sig")),
                                         "upstream_commit": clip4clip.UPSTREAM_COMMIT,
                                         "source_hashes": source_hashes,
                                         "sampling": "native_int_fps_1Hz_uniform12",
                                         "max_words": 32, "max_frames": 12,
                                         "precision": "float32", "video_records": records,
                                         "torch": str(torch.__version__),
                                         "numpy": str(np.__version__),
                                         "python": platform.python_version()}})
    del model
    return cache


def _public_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("ranks_")}


def _same_ranks(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(np.array_equal(left[key], right[key])
               for key in ("ranks_text_to_video", "ranks_video_to_text"))


def execute(config: dict[str, Any], pool: Any, output: Path, *, build_cache: bool = False):
    """Run B0 first, then B1/B2 only after required score/rank parity passes."""
    started = time.monotonic()
    torch.set_num_threads(config["threads"])
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no device substitution")
    if config["device"] == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if build_cache:
        cache = build_frame_cache(config, pool, started=started)
        cache_path = output / "frame_features.npz"
        save_feature_cache(cache_path, cache)
        cache_hash = file_sha256(cache_path)
    else:
        _guard(config, started)
        cache_path = _checked_file(config["feature_cache"], config["feature_cache_sha256"],
                                   "feature cache")
        expanded_bytes = cache_path.stat().st_size
        if is_zipfile(cache_path):
            with ZipFile(cache_path) as archive:
                expanded_bytes = sum(item.file_size for item in archive.infolist())
        _guard(config, started, expanded_bytes * 3)
        cache = load_feature_cache(cache_path, config["feature_cache_sha256"],
                                   max_bytes=config["max_cache_bytes"])
        cache_hash = config["feature_cache_sha256"]
    _check_cache_ids(cache, pool)
    metadata = cache.get("metadata", {})
    if metadata.get("split") not in {None, pool.split}:
        raise ValueError("Cached split differs from configured locked split")
    cached_scale = metadata.get("logit_scale", metadata.get("settings", {}).get("logit_scale"))
    if cached_scale is not None and not np.isclose(cached_scale, config["logit_scale"], rtol=1e-6):
        raise ValueError("Cached logit_scale differs from explicit config")
    _guard(config, started, len(pool.text_ids) * len(pool.video_ids) * 4 * 8)
    args = (cache["text_video_ids"], cache["video_ids"], cache["text_ids"])
    cli_path = Path(__file__).resolve().parents[3] / "scripts/run_msrvtt_pooling.py"
    report = {"schema_version": SCHEMA, "config": config, "variants": {}, "comparisons": {},
              "artifact_kind": "new_frozen_feature_checkpoint_inference" if build_cache
              else "existing_feature_cache_scoring_replay", "encoder_train_run": False,
              "feature_cache_sha256": cache_hash, "cache_metadata": metadata,
              "pool_hashes": pool.hashes(), "device": config["device"],
              "dtype": "float32", "score_type": "scaled_cosine_not_probability",
              "selection_performed": False, "python": platform.python_version(),
              "torch": str(torch.__version__), "numpy": str(np.__version__),
              "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
              "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
              "implementation_sha256": file_sha256(Path(__file__)),
              "core_sha256": file_sha256(Path(__file__).with_name("retrieval_pooling.py")),
              "cli_sha256": file_sha256(cli_path) if cli_path.is_file() else None}
    results, rows = {}, []
    scale = np.float32(config["logit_scale"])
    for name in ("B0", "B1", "B2"):
        _guard(config, started)
        tick = time.perf_counter()
        scores = score_matrix(cache["text"], cache["frames"], cache["mask"], method=name,
                              tau=config["tau"], alpha=config["alpha"],
                              query_chunk=config["query_chunk"], video_chunk=config["video_chunk"],
                              device=config["device"]) * scale
        elapsed = time.perf_counter() - tick
        result = evaluate_bidirectional(scores, *args)
        results[name] = result
        np.savez_compressed(output / f"{name}_scores.npz", scores=scores,
                            text_ids=np.asarray(cache["text_ids"], dtype=str),
                            text_video_ids=np.asarray(cache["text_video_ids"], dtype=str),
                            video_ids=np.asarray(cache["video_ids"], dtype=str),
                            logit_scale=scale, score_type=np.asarray("scaled_cosine"))
        report["variants"][name] = {"metrics": _public_metrics(result),
                                    "legacy_metrics": aligned_original_metrics(scores, args[0], args[1]),
                                    "score_seconds": elapsed}
        for direction in ("text_to_video", "video_to_text"):
            rows.append({"variant": name, "direction": direction, **result[direction],
                         "score_seconds": elapsed})
        if name == "B0":
            reference = direct_meanp_reference(cache["text"], cache["frames"], cache["mask"]) * scale
            local_ok = bool(np.allclose(scores, reference, rtol=config["rtol"], atol=config["atol"]))
            local_ranks = _same_ranks(result, evaluate_bidirectional(reference, *args))
            external_ok = external_ranks = None
            if config.get("baseline_reference"):
                path = _checked_file(config["baseline_reference"], config["reference_sha256"],
                                      "baseline reference")
                with np.load(path, allow_pickle=False) as saved:
                    if (saved["text_ids"].tolist() != cache["text_ids"]
                            or saved["video_ids"].tolist() != cache["video_ids"]):
                        raise ValueError("Baseline reference IDs/order differ")
                    expected = np.asarray(saved["scores"], dtype=np.float32)
                if not config.get("reference_is_logits", True):
                    expected = expected * scale
                external_ok = expected.shape == scores.shape and bool(np.isfinite(expected).all())
                external_ok = external_ok and bool(np.allclose(
                    scores, expected, rtol=config["rtol"], atol=config["atol"]))
                external_ranks = external_ok and _same_ranks(
                    result, evaluate_bidirectional(expected, *args))
            gate = {"implementation_score_parity": local_ok, "implementation_rank_parity": local_ranks,
                    "external_score_parity": external_ok, "external_rank_parity": external_ranks,
                    "external_baseline_reproduced": external_ok is True and external_ranks is True}
            gate["passed"] = local_ok and local_ranks and (
                gate["external_baseline_reproduced"] if config.get("require_external_reference", True)
                else external_ok is not False and external_ranks is not False)
            report["baseline_gate"] = gate
            write_json(output / "baseline_gate.json", gate)
            write_json(output / "benchmark.json", report)
            if not gate["passed"]:
                raise ValueError("B0 baseline parity failed; B1/B2 were not evaluated")
        else:
            report["comparisons"][f"{name}-B0"] = compare_runs(
                results["B0"], result, *args, n_bootstrap=config["bootstrap_samples"],
                seed=config["seed"], output_csv=output / f"{name}_query_changes.csv")
        write_json(output / "benchmark.json", report)
    with (output / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["wall_seconds"] = time.monotonic() - started
    report["status"] = "completed_frozen_pooling_evaluation"
    write_json(output / "benchmark.json", report)
    return {"status": report["status"], "artifact_kind": report["artifact_kind"],
            "benchmark_inference_run": build_cache, "baseline_gate": report["baseline_gate"],
            "selection_performed": False, "wall_seconds": report["wall_seconds"]}

"""Inference-only evaluation of the hash-locked B3 epoch5 head on MSR-VTT test1k.

The immutable portable workflow supplies B0-B2 and evaluators on the exact
previously verified test cache. No encoder, decoding, training or selection
entry point is called. CPU debug overrides are
explicitly synthetic, force CPU, and cannot be reported as the official run.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import random
import shutil
import time
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import data_pipeline as pipeline
import numpy as np
import retrieval_core as core
import torch
import workflow

SOURCE_PINS = {
    "data_pipeline": "0d0f87116af9f69205a06eb652d5606f8bed08e5ec6b4cb51006861aa5c7e5bb",
    "retrieval_core": "94fd6527fa59cb7eae5c884444a02ff25eda753d399c821cddeddbfde6c2dd64",
    "workflow": "9c57681138093fabe39157c57b8bfc9dc7fff2d71cea6aa4b0d8131d733a47f9",
}
METRICS_SHA256 = "103e93090de14f55d1db61e40ecf1fbc814670975bd9b9ab1fe341d0c95f0a6f"
DEFAULT_CONFIG = {
    "INPUT_ROOT": "/kaggle/input",
    "WORK_DIR": "/kaggle/working/msrvtt_b3_evaluation/jaewooo_epoch4_fp32_b3_epoch5_eval_v1",
    "TEST_CSV": "",
    "TEST_CSV_SHA256": "e61144b312c8a2fc2126f96504c981f8c23c05b3ba9655a56e2af50ac8a4b16a",
    "CHECKPOINT_SHA256": "3d7e2ba508986c7347380767233b017d4ac4c9fabc135f3457de8625dc833dca",
    "TEST_CACHE_PATH": "",
    "TEST_CACHE_SHA256": "c721c99d5d29e47661f2cbdd74ff59f1454b265d035a6a11d7faf88444c30095",
    "TEST_CACHE_BYTES": 27131378,
    "TEST_FEATURE_MANIFEST_PATH": "",
    "TEST_FEATURE_MANIFEST_SHA256": "928925476b4bc4b077ff36519d4b6eb5c2fe8f5ba4447a8091c0d10639cbf0ac",
    "TEST_CACHE_PROVENANCE_PATH": "",
    "TEST_CACHE_PROVENANCE_SHA256": "67973ec246b6265b3d927480c5bb90b80559a451851c17979784b804c5e207ea",
    "TEST_CACHE_FINGERPRINT": "192caf0144845f10b52980fd53546b0c7fc4265ac1b249f3b472ad91a6a9c47c",
    "TEST_NUMERIC_FINGERPRINT": "7ade86bebc0c965e7f102db0d74663da344614404c0a4aba07df9b52e52911f2",
    "TEST_CACHE_VALID_FRAMES": 11727,
    "CACHE_SOURCE_ARCHIVE_SHA256": "44ff9576c45ed5b0e8e92b04271da25764ba4b0b2f999496067d64bfd4bbd890",
    "HEAD_PATH": "",
    "HEAD_SHA256": "374a56f283203051a8c27285c4d5b37ebc440b73c41eea450c593dbce11523a5",
    "HEAD_BYTES": 1051117,
    "HEAD_DIM": 512,
    "HEAD_LOCK_PATH": "",
    "HEAD_LOCK_SHA256": "60040d04c9b86a94db2b758c50d56b1c24c84971caf56bde3f14de37d2c7e752",
    "TRAIN_REPORT_PATH": "",
    "TRAIN_REPORT_SHA256": "d8eab730c28499b3e54f04aa20a4530695a74565496bf5efa0fb57b056a623eb",
    "TRAIN_EXPERIMENT_ID": "0b1aea8b67054ec5719e93428394bac44a363ff14f55e674224b3bf19021f2a2",
    "TRAIN_RESULT_ZIP_SHA256": "52b9167686590a8c8b31fe6db90ebbe1d9e12dc92b444298e34a7a866a22be99",
    "BASELINE_REFERENCE_NPZ": "",
    "BASELINE_REFERENCE_SHA256": "429403c137f57f98ff93d7f3fcf121e32a5c3b88f0b60688c784a2028eaa15ba",
    "REFERENCE_IS_LOGITS": True,
    "REQUIRE_EXTERNAL_REFERENCE": True,
    "ALLOW_UNVERIFIED_EXPLORATORY": False,
    "EXPECTED_TEST_QUERIES": 1000,
    "EXPECTED_TEST_VIDEOS": 1000,
    "EXPECTED_BASELINE_R1": 41.0,
    "EXPECTED_R1_TOL_PP": 1e-8,
    "RUN_TRAIN": False,
    "SELECTION_POLICY": "fixed_epochs",
    "PRECISION": "fp32",
    "MAX_FRAMES": 12,
    "MAX_WORDS": 32,
    "ENCODER_BATCH": 4,
    "TEXT_BATCH": 1,
    "QUERY_CHUNK": 32,
    "VIDEO_CHUNK": 32,
    "TAU": 0.05,
    "ALPHA": 0.25,
    "SEED": 42,
    "BOOTSTRAP_SAMPLES": 2000,
    "ATOL": 1e-5,
    "RTOL": 1e-5,
    "ALLOW_CPU_DEBUG": False,
    "MIN_FREE_DISK_GIB": 2.0,
}


def checked_config(config):
    cfg = {**DEFAULT_CONFIG, **config}
    immutable = (
        "RUN_TRAIN",
        "SELECTION_POLICY",
        "PRECISION",
        "MAX_FRAMES",
        "MAX_WORDS",
        "ENCODER_BATCH",
        "TEXT_BATCH",
        "QUERY_CHUNK",
        "VIDEO_CHUNK",
        "TAU",
        "ALPHA",
        "SEED",
        "REFERENCE_IS_LOGITS",
        "REQUIRE_EXTERNAL_REFERENCE",
        "ALLOW_UNVERIFIED_EXPLORATORY",
        "ATOL",
        "RTOL",
        "EXPECTED_R1_TOL_PP",
    )
    for key in immutable:
        if type(cfg[key]) is not type(DEFAULT_CONFIG[key]) or cfg[key] != DEFAULT_CONFIG[key]:
            raise ValueError(f"Locked evaluation setting {key} must equal {DEFAULT_CONFIG[key]!r}")
    if type(cfg["ALLOW_CPU_DEBUG"]) is not bool:
        raise ValueError("ALLOW_CPU_DEBUG must be boolean")
    if cfg.get("VAL_CSV") or cfg.get("VAL_PROVENANCE") or cfg.get("PREVIOUS_RUN_DIR"):
        raise ValueError("Evaluation uses no validation/training/resume input")
    fixture_keys = (
        "TEST_CSV_SHA256",
        "CHECKPOINT_SHA256",
        "HEAD_SHA256",
        "HEAD_BYTES",
        "HEAD_DIM",
        "HEAD_LOCK_SHA256",
        "TRAIN_REPORT_SHA256",
        "TRAIN_EXPERIMENT_ID",
        "TRAIN_RESULT_ZIP_SHA256",
        "BASELINE_REFERENCE_SHA256",
        "EXPECTED_TEST_QUERIES",
        "EXPECTED_TEST_VIDEOS",
        "EXPECTED_BASELINE_R1",
        "BOOTSTRAP_SAMPLES",
        "TEST_CACHE_SHA256",
        "TEST_CACHE_BYTES",
        "TEST_FEATURE_MANIFEST_SHA256",
        "TEST_CACHE_PROVENANCE_SHA256",
        "TEST_CACHE_FINGERPRINT",
        "TEST_NUMERIC_FINGERPRINT",
        "TEST_CACHE_VALID_FRAMES",
        "CACHE_SOURCE_ARCHIVE_SHA256",
    )
    if not cfg["ALLOW_CPU_DEBUG"]:
        for key in fixture_keys:
            if type(cfg[key]) is not type(DEFAULT_CONFIG[key]) or cfg[key] != DEFAULT_CONFIG[key]:
                raise ValueError(f"Official evaluation requires the original {key}")
    for key in (
        "HEAD_BYTES",
        "HEAD_DIM",
        "EXPECTED_TEST_QUERIES",
        "EXPECTED_TEST_VIDEOS",
        "BOOTSTRAP_SAMPLES",
        "TEST_CACHE_BYTES",
        "TEST_CACHE_VALID_FRAMES",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "TEST_CSV_SHA256",
        "CHECKPOINT_SHA256",
        "HEAD_SHA256",
        "HEAD_LOCK_SHA256",
        "TRAIN_REPORT_SHA256",
        "TRAIN_EXPERIMENT_ID",
        "TRAIN_RESULT_ZIP_SHA256",
        "BASELINE_REFERENCE_SHA256",
        "TEST_CACHE_SHA256",
        "TEST_FEATURE_MANIFEST_SHA256",
        "TEST_CACHE_PROVENANCE_SHA256",
        "TEST_CACHE_FINGERPRINT",
        "TEST_NUMERIC_FINGERPRINT",
        "CACHE_SOURCE_ARCHIVE_SHA256",
    ):
        value = cfg[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"{key} must be a lowercase SHA256")
    if not math.isfinite(float(cfg["MIN_FREE_DISK_GIB"])) or cfg["MIN_FREE_DISK_GIB"] < 0:
        raise ValueError("MIN_FREE_DISK_GIB must be finite and nonnegative")
    return cfg


def _verified_file(cfg, path_key, filename, sha_key, *, limit=None):
    path = pipeline.resolve_file(cfg[path_key], filename, cfg["INPUT_ROOT"])
    if limit is not None and path.stat().st_size > limit:
        raise ValueError(f"Unexpected file size for {filename}")
    if pipeline.sha256_file(path) != cfg[sha_key]:
        raise ValueError(f"SHA256 mismatch for {filename}; attach the locked original file")
    return path


def verify_frozen_head(cfg):
    """Verify bytes, provenance and complete final-epoch payload before test scoring."""
    head_path = _verified_file(cfg, "HEAD_PATH", "b3_final.pt", "HEAD_SHA256", limit=4 * 1024**2)
    if head_path.stat().st_size != cfg["HEAD_BYTES"]:
        raise ValueError("B3 head byte count differs from its locked snapshot")
    lock_path = _verified_file(
        cfg, "HEAD_LOCK_PATH", "b3_evaluation_lock.json", "HEAD_LOCK_SHA256", limit=1024**2
    )
    report_path = _verified_file(
        cfg, "TRAIN_REPORT_PATH", "b3_training_report.json", "TRAIN_REPORT_SHA256", limit=1024**2
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    trained = json.loads(report_path.read_text(encoding="utf-8"))
    expected_lock = {
        "schema": "msrvtt_b3_frozen_head_before_evaluation_v1",
        "source_archive_sha256": cfg["TRAIN_RESULT_ZIP_SHA256"],
        "head_sha256": cfg["HEAD_SHA256"],
        "head_bytes": cfg["HEAD_BYTES"],
        "experiment_identity": cfg["TRAIN_EXPERIMENT_ID"],
        "encoder_checkpoint_sha256": cfg["CHECKPOINT_SHA256"],
        "selected_epoch": 5,
        "selection_policy": "fixed_epochs",
        "test_scores_computed": False,
    }
    if any(lock.get(key) != value for key, value in expected_lock.items()):
        raise ValueError("B3 evaluation lock differs from the preregistered epoch5 head")
    if (
        trained.get("status") != "completed_train_only"
        or trained.get("training_complete") is not True
        or trained.get("test_scores_computed") is not False
        or trained.get("validation_scores_computed") is not False
        or trained.get("benchmark_metrics") is not None
        or trained.get("experiment_identity") != cfg["TRAIN_EXPERIMENT_ID"]
        or trained.get("checkpoint_sha256") != cfg["CHECKPOINT_SHA256"]
        or trained.get("code_file_sha256", {}).get("retrieval_core")
        != SOURCE_PINS["retrieval_core"]
    ):
        raise ValueError("Training provenance is not the verified completed epoch5 run")
    training_cfg = {
        "EPOCHS": 5,
        "SELECTION_POLICY": "fixed_epochs",
        "SEED": 42,
        "PRECISION": "fp32",
        "MAX_FRAMES": 12,
        "MAX_WORDS": 32,
        "ENCODER_BATCH": 4,
        "TEXT_BATCH": 1,
        "TRAIN_BATCH": 32,
        "LR": 0.001,
        "WEIGHT_DECAY": 0.0001,
        "TRAIN_LOGIT_SCALE": 20.0,
        "TAU": 0.05,
        "ALPHA": 0.25,
    }
    if any(trained.get("config", {}).get(key) != value for key, value in training_cfg.items()):
        raise ValueError("Training hyperparameters differ from the locked B3 schedule")
    payload = torch.load(head_path, map_location="cpu", weights_only=True)
    expected_head = {
        "dim": cfg["HEAD_DIM"],
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "tau": 0.05,
        "alpha": 0.25,
        "logit_scale": 20.0,
        "seed": 42,
        "epochs": 5,
        "selection_policy": "fixed_epochs",
        "validation_provenance": "",
        "core_sha256": SOURCE_PINS["retrieval_core"],
    }
    if (
        payload.get("format") != "msrvtt_tiny_attention_v1"
        or payload.get("config") != expected_head
        or payload.get("selected_epoch") != 5
        or payload.get("validation_status")
        != "reserved_encoder_seen_dev_not_accessed_for_selection"
        or payload.get("cache_fingerprints", {}).get("validation") is not None
        or not payload.get("cache_fingerprints", {}).get("train")
        or trained.get("training", {}).get("selected_epoch") != 5
        or trained.get("training", {}).get("selection_policy") != "fixed_epochs"
        or trained.get("training", {}).get("cache_fingerprints")
        != payload.get("cache_fingerprints")
    ):
        raise ValueError("B3 checkpoint is not the locked final completed epoch5 payload")
    model = payload.get("model", {})
    shapes = {"query_delta.weight": (cfg["HEAD_DIM"], cfg["HEAD_DIM"]), "log_tau": ()}
    if set(model) != set(shapes) or any(
        not isinstance(model[key], torch.Tensor)
        or tuple(model[key].shape) != shape
        or model[key].dtype != torch.float32
        or not torch.isfinite(model[key]).all()
        for key, shape in shapes.items()
    ):
        raise ValueError("B3 model tensor shape/dtype/finite contract failed")
    return {
        "head_path": head_path,
        "head_lock_path": lock_path,
        "train_report_path": report_path,
        "head_lock": lock,
        "training_report": trained,
        "head_payload": payload,
    }


def verify_fixed_cache(cfg, records):
    """Load the exact prior test tensors safely; never encode or silently repair."""
    cache_path = _verified_file(
        cfg, "TEST_CACHE_PATH", "test_features.pt", "TEST_CACHE_SHA256", limit=64 * 1024**2
    )
    if cache_path.stat().st_size != cfg["TEST_CACHE_BYTES"]:
        raise ValueError("Fixed test cache byte count differs from its source receipt")
    manifest_path = _verified_file(
        cfg,
        "TEST_FEATURE_MANIFEST_PATH",
        "test_feature_manifest.json",
        "TEST_FEATURE_MANIFEST_SHA256",
        limit=4 * 1024**2,
    )
    provenance_path = _verified_file(
        cfg,
        "TEST_CACHE_PROVENANCE_PATH",
        "test_cache_provenance.json",
        "TEST_CACHE_PROVENANCE_SHA256",
        limit=1024**2,
    )
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict) or cache.get("metadata") != metadata:
        raise ValueError("Cached tensor metadata differs from the original feature manifest")
    original_types = {"text": torch.float32, "frames": torch.float32, "mask": torch.bool}
    if any(
        not isinstance(cache.get(key), torch.Tensor) or cache[key].dtype != dtype
        for key, dtype in original_types.items()
    ):
        raise ValueError("Fixed cache requires float32 text/frames and a bool frame mask")
    cache = core.validate_cache(cache)
    nq, nv, dim = cfg["EXPECTED_TEST_QUERIES"], cfg["EXPECTED_TEST_VIDEOS"], cfg["HEAD_DIM"]
    if (
        tuple(cache["text"].shape) != (nq, dim)
        or tuple(cache["frames"].shape) != (nv, cfg["MAX_FRAMES"], dim)
        or int(cache["mask"].sum()) != cfg["TEST_CACHE_VALID_FRAMES"]
        or torch.count_nonzero(cache["frames"][~cache["mask"]]).item() != 0
    ):
        raise ValueError("Fixed cache shape, valid-frame count or zero-padding differs")
    ids = list(dict.fromkeys(row["video_id"] for row in records))
    if (
        cache["video_ids"] != ids
        or cache["text_video_ids"] != [row["video_id"] for row in records]
        or cache["text_ids"] != [row["text_id"] for row in records]
        or metadata.get("texts") != [row["text"] for row in records]
        or metadata.get("records_sha256") != pipeline.digest(records)
    ):
        raise ValueError("Fixed cache raw caption/ID/positive mapping/order differs from test CSV")
    video_meta = metadata.get("video_metadata", [])
    if [item.get("video_id") for item in video_meta] != ids:
        raise ValueError("Fixed cache video metadata order differs from test candidates")
    media_digest = pipeline.digest(
        [(item["video_id"], item["input_sha256"]) for item in video_meta]
    )
    if metadata.get("video_inputs_sha256") != media_digest:
        raise ValueError("Fixed cache source video hashes are inconsistent")
    settings = metadata.get("settings", {})
    expected_settings = {
        "checkpoint_sha256": cfg["CHECKPOINT_SHA256"],
        "source_revision": "508ffa3de39ba0563a03199c440ab602a72e9b6f",
        "precision": "fp32",
        "effective_weight_dtype": "torch.float32",
        "embed_dim": dim,
        "resolution": 224,
        "schema": 3,
        "pipeline_sha256": SOURCE_PINS["data_pipeline"],
        "max_frames": 12,
        "max_words": 32,
        "sampling": "original_integer_fps_1fps_then_uniform12",
        "preprocess": "PIL_bicubic_shortedge_centercrop_RGB_CLIP",
    }
    if any(settings.get(key) != value for key, value in expected_settings.items()):
        raise ValueError("Fixed cache encoder/preprocessing origin differs from the B3 protocol")
    scale = settings.get("logit_scale", 0)
    if (
        not math.isfinite(float(scale))
        or scale <= 0
        or (not cfg["ALLOW_CPU_DEBUG"] and scale != 100.0)
    ):
        raise ValueError("Fixed cache has a different encoder evaluation logit scale")
    feature_digest = pipeline.digest(
        {key: metadata[key] for key in ("settings", "records_sha256", "video_inputs_sha256")}
    )
    if (
        metadata.get("fingerprint") != feature_digest
        or feature_digest != cfg["TEST_CACHE_FINGERPRINT"]
    ):
        raise ValueError("Fixed cache metadata fingerprint differs")
    numeric_digest = core._cache_fingerprint(cache)
    if numeric_digest != cfg["TEST_NUMERIC_FINGERPRINT"]:
        raise ValueError("Fixed test feature tensor fingerprint differs")
    source_kind = (
        "synthetic_cpu_fixture_generated_before_notebook"
        if cfg["ALLOW_CPU_DEBUG"]
        else "previous_verified_B0_B1_B2_run"
    )
    expected_provenance = {
        "schema": "msrvtt_b3_eval_fixed_test_cache_v1",
        "cache_source": source_kind,
        "source_baseline_archive_sha256": cfg["CACHE_SOURCE_ARCHIVE_SHA256"],
        "cache_filename": "test_features.pt",
        "cache_sha256": cfg["TEST_CACHE_SHA256"],
        "cache_bytes": cfg["TEST_CACHE_BYTES"],
        "feature_manifest_filename": "test_feature_manifest.json",
        "feature_manifest_sha256": cfg["TEST_FEATURE_MANIFEST_SHA256"],
        "feature_fingerprint": feature_digest,
        "core_cache_fingerprint": numeric_digest,
        "test_csv_sha256": cfg["TEST_CSV_SHA256"],
        "baseline_reference_sha256": cfg["BASELINE_REFERENCE_SHA256"],
        "encoder_checkpoint_sha256": cfg["CHECKPOINT_SHA256"],
        "source_pipeline_sha256": SOURCE_PINS["data_pipeline"],
        "test_queries": nq,
        "test_videos": nv,
        "valid_frames": cfg["TEST_CACHE_VALID_FRAMES"],
        "records_sha256": metadata["records_sha256"],
        "video_inputs_sha256": media_digest,
        "ordered_text_ids_sha256": pipeline.digest(cache["text_ids"]),
        "ordered_video_ids_sha256": pipeline.digest(cache["video_ids"]),
        "settings": settings,
        "new_encoder_inference_performed": False,
        "new_test_scores_computed": False,
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("Fixed cache provenance receipt is inconsistent")
    return {
        "cache": cache,
        "cache_path": cache_path,
        "feature_manifest_path": manifest_path,
        "cache_provenance_path": provenance_path,
        "cache_provenance": provenance,
    }


def prepare_evaluation(config):
    """Hash/cache preflight; no video decode, encoder, retrieval scoring or training."""
    cfg = checked_config(config)
    code = {
        "data_pipeline": pipeline.sha256_file(pipeline.__file__),
        "retrieval_core": pipeline.sha256_file(core.__file__),
        "workflow": pipeline.sha256_file(workflow.__file__),
    }
    if code != SOURCE_PINS:
        raise ValueError("Portable source changed; preserve the reviewed evaluation code")
    frozen = verify_frozen_head(cfg)
    test_csv = _verified_file(cfg, "TEST_CSV", "MSRVTT_JSFUSION_test.csv", "TEST_CSV_SHA256")
    reference = _verified_file(
        cfg, "BASELINE_REFERENCE_NPZ", "baseline_reference.npz", "BASELINE_REFERENCE_SHA256"
    )
    records = pipeline.evaluation_records(test_csv)
    ids = list(dict.fromkeys(row["video_id"] for row in records))
    if len(records) != cfg["EXPECTED_TEST_QUERIES"] or len(ids) != cfg["EXPECTED_TEST_VIDEOS"]:
        raise ValueError("Test query/candidate count differs from the locked test1k pool")
    if len(records) != len(ids):
        raise ValueError("This locked test requires exactly one caption for each candidate video")
    cfg["TEST_CSV"], cfg["BASELINE_REFERENCE_NPZ"] = str(test_csv), str(reference)
    fixed_cache = verify_fixed_cache(cfg, records)
    inventory = {
        "test_queries": len(records),
        "test_videos": len(ids),
        "valid_frames": cfg["TEST_CACHE_VALID_FRAMES"],
        "selected_epoch": 5,
        "head_sha256": cfg["HEAD_SHA256"],
        "cache_sha256": cfg["TEST_CACHE_SHA256"],
        "feature_fingerprint": cfg["TEST_CACHE_FINGERPRINT"],
        "new_encoder_inference_performed": False,
        "training_executed": False,
    }
    return {
        "config": cfg,
        **frozen,
        **fixed_cache,
        "test_records": records,
        "inventory": inventory,
        "source_sha256": {**code, "b3_eval_runtime": pipeline.sha256_file(__file__)},
    }


def initialize_device(allow_cpu_debug=False, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if allow_cpu_debug:
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("Bật Kaggle Settings → GPU; CPU_DEBUG chỉ dùng fixture tổng hợp")
    torch.cuda.init()
    selected = torch.device("cuda:0")
    torch.cuda.set_device(selected)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.synchronize(selected)
    torch.cuda.reset_peak_memory_stats(selected)
    return "cuda:0"


def _write_snapshot(source, target):
    raw = Path(source).read_bytes()
    if target.exists() and target.read_bytes() != raw:
        raise ValueError(f"Preserve an existing different evaluation snapshot: {target.name}")
    if not target.exists():
        target.write_bytes(raw)


def score_b3(state, cache, head):
    """Score immutable B3 with learned temperature and encoder evaluation scale."""
    cfg = state["config"]
    if int(cache["text"].shape[-1]) != cfg["HEAD_DIM"]:
        raise ValueError("Test feature dimension differs from the locked B3 head")
    head = head.to(state["device"]).eval()
    head.requires_grad_(False)
    with torch.inference_mode():
        scores = core.score_matrix(
            cache["text"],
            cache["frames"],
            cache["mask"],
            method="B2",
            head=head,
            tau=cfg["TAU"],
            alpha=cfg["ALPHA"],
            query_chunk=cfg["QUERY_CHUNK"],
            video_chunk=cfg["VIDEO_CHUNK"],
            device=state["device"],
        ) * workflow._logit_scale(state)
    result = core.evaluate_bidirectional(
        scores, cache["text_video_ids"], cache["video_ids"], cache["text_ids"]
    )
    original = workflow._official_source_metrics(scores, cache, state["vendor_dir"])
    copied = core.aligned_original_metrics(scores, cache["text_video_ids"], cache["video_ids"])
    if not original["available"]:
        raise ValueError("Locked B3 evaluation requires the pinned original singleton evaluator")
    for direction in ("text_to_video", "video_to_text"):
        for upstream, internal in (
            ("R1", "R@1"),
            ("R5", "R@5"),
            ("R10", "R@10"),
            ("MR", "MedR"),
            ("MeanR", "MeanR"),
        ):
            if upstream in original[direction] and not np.isclose(
                original[direction][upstream], copied[direction][internal], atol=1e-12
            ):
                raise AssertionError(f"Original B3 evaluator mismatch: {direction}/{upstream}")
    return scores, result, original


def _record_artifacts(report, work, names):
    """Bind only outputs produced by this attempt, never discover old score files."""
    for name in names:
        path = Path(work) / name
        if Path(name).name != name or "\\" in name or ":" in name:
            raise ValueError("Export artifact must be a flat filename")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing current-attempt artifact: {name}")
        report["export_artifacts"][name] = {
            "bytes": path.stat().st_size,
            "sha256": pipeline.sha256_file(path),
        }


def evaluation_export_files(work):
    """Return hash-verified current-attempt files; stale metrics stay on disk."""
    work = Path(work)
    report_path = work / "b3_evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != "msrvtt_b3_evaluation_v1":
        raise ValueError("Unsupported evaluation report")
    attempt = report["attempt"]
    if not isinstance(attempt, str) or any(c in attempt for c in "/\\:"):
        raise ValueError("Unsafe report attempt identifier")
    files = []
    for name, expected in report.get("export_artifacts", {}).items():
        path = work / name
        if Path(name).name != name or "\\" in name or ":" in name or path.is_symlink():
            raise ValueError("Unsafe evaluation export path")
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or pipeline.sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(f"Current-attempt export changed: {name}")
        files.append(path)
    required = [
        report_path,
        work / f"evaluation_attempt_{attempt}.json",
        work / "evaluation_log.json",
    ]
    for path in required:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Current evaluation report/log is missing or linked")
    if json.loads(required[1].read_text(encoding="utf-8")) != report:
        raise ValueError("Current report and attempt snapshot differ")
    files.extend(required)
    for name in ("source_manifest.json", "runtime_sources.zip"):
        path = work / name
        if path.is_file() and not path.is_symlink():
            files.append(path)
    return sorted(set(files)), report


def export_evaluation_results(work, destination):
    """Export only this attempt; preserve a previously created result ZIP."""
    files, report = evaluation_export_files(work)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        suffix = report["attempt"] + "_" + uuid.uuid4().hex[:6]
        destination = destination.with_name(f"{destination.stem}_{suffix}{destination.suffix}")
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return destination


def run_evaluation(config, vendor_dir):
    """Run B0/B1/B2 gate then hash-locked B3; always preserve report/error evidence."""
    cfg = {**DEFAULT_CONFIG, **config}
    work = Path(cfg["WORK_DIR"])
    work.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    report = {
        "format": "msrvtt_b3_evaluation_v1",
        "attempt": attempt,
        "status": "running",
        "stage": "preflight",
        "config": cfg,
        "started_utc": datetime.now(UTC).isoformat(),
        "events": [],
        "training_executed": False,
        "validation_scores_computed": False,
        "b3_test_scores_computed": False,
        "benchmark_metrics": None,
        "new_encoder_inference_performed": False,
        "raw_video_decode_performed": False,
        "encoder_checkpoint_loaded": False,
        "generated_variants": [],
        "export_artifacts": {},
        "synthetic_cpu_fixture": cfg["ALLOW_CPU_DEBUG"],
    }

    def save():
        pipeline.atomic_json(work / "b3_evaluation_report.json", report)
        pipeline.atomic_json(work / f"evaluation_attempt_{attempt}.json", report)
        pipeline.atomic_json(work / "evaluation_log.json", report["events"])

    def event(stage, message):
        report["stage"] = stage
        report["events"].append(
            {"time_utc": datetime.now(UTC).isoformat(), "stage": stage, "message": message}
        )
        print(message, flush=True)
        save()

    try:
        cfg = checked_config(cfg)
        prepared = prepare_evaluation(cfg)
        cfg = prepared["config"]
        metrics_path = Path(vendor_dir) / "metrics.reference.py"
        if not metrics_path.is_file() or pipeline.sha256_file(metrics_path) != METRICS_SHA256:
            raise ValueError("Pinned original evaluator SHA256 differs")
        if shutil.disk_usage(work).free < cfg["MIN_FREE_DISK_GIB"] * 1024**3:
            raise OSError("Insufficient working disk for test cache; preserve the existing output")
        torch.set_num_threads(2)
        device = initialize_device(cfg["ALLOW_CPU_DEBUG"], cfg["SEED"])
        _write_snapshot(prepared["head_path"], work / "b3_frozen_head.pt")
        _write_snapshot(prepared["head_lock_path"], work / "b3_evaluation_lock.json")
        _write_snapshot(prepared["train_report_path"], work / "b3_training_report.json")
        _write_snapshot(prepared["cache_provenance_path"], work / "test_cache_provenance.json")
        _write_snapshot(prepared["feature_manifest_path"], work / "test_feature_manifest.json")
        _write_snapshot(cfg["TEST_CSV"], work / "MSRVTT_JSFUSION_test.csv")
        if pipeline.sha256_file(work / "b3_frozen_head.pt") != cfg["HEAD_SHA256"]:
            raise ValueError("Head changed while creating its immutable evaluation snapshot")
        _record_artifacts(
            report,
            work,
            [
                "b3_frozen_head.pt",
                "b3_evaluation_lock.json",
                "b3_training_report.json",
                "test_cache_provenance.json",
                "test_feature_manifest.json",
                "MSRVTT_JSFUSION_test.csv",
            ],
        )
        identity_keys = {
            key: value
            for key, value in cfg.items()
            if not key.endswith("PATH")
            and key not in {"INPUT_ROOT", "WORK_DIR", "TEST_CSV", "BASELINE_REFERENCE_NPZ"}
        }
        registration = {
            "config": cfg,
            "source_sha256": prepared["source_sha256"],
            "head_sha256": cfg["HEAD_SHA256"],
            "selected_epoch": 5,
            "test_cache_sha256": cfg["TEST_CACHE_SHA256"],
            "cache_numeric_fingerprint": cfg["TEST_NUMERIC_FINGERPRINT"],
            "new_encoder_inference_performed": False,
            "test_records_sha256": pipeline.digest(prepared["test_records"]),
            "test_used_for_selection": False,
            "training_executed": False,
            "primary_comparisons": ["B3-B0 text_to_video R@1", "B3-B2 text_to_video R@1"],
            "tie_rule": "descending FP32 score; stable input candidate order on exact ties",
        }
        identity = pipeline.digest(
            {
                "config": identity_keys,
                "source_sha256": prepared["source_sha256"],
                "test_records_sha256": registration["test_records_sha256"],
            }
        )
        identity_path = work / "evaluation_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text())["identity"] != identity:
            raise ValueError("WORK_DIR has a different evaluation identity; choose a new RUN_NAME")
        pipeline.atomic_json(identity_path, {"identity": identity})
        pipeline.atomic_json(work / "evaluation_preregistration.json", registration)
        _record_artifacts(
            report, work, ["evaluation_identity.json", "evaluation_preregistration.json"]
        )
        report.update(
            config=cfg,
            experiment_identity=identity,
            source_sha256=prepared["source_sha256"],
            head_sha256=cfg["HEAD_SHA256"],
            selected_epoch=5,
            training_experiment_identity=cfg["TRAIN_EXPERIMENT_ID"],
            evidence_level="synthetic_cpu_fixture_evaluation"
            if cfg["ALLOW_CPU_DEBUG"]
            else "locked_head_inference_on_official_msrvtt_test1k",
            device="CPU_DEBUG" if device == "cpu" else torch.cuda.get_device_name(0),
            environment={
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
            },
        )
        head = core.load_attention_head(work / "b3_frozen_head.pt", device="cpu")
        head.requires_grad_(False)
        cache = prepared["cache"]
        state = {
            "config": cfg,
            "device": device,
            "model_info": cache["metadata"]["settings"],
            "vendor_dir": str(vendor_dir),
        }
        report["test_cache"] = {
            "videos": len(cache["video_ids"]),
            "captions": len(cache["text_ids"]),
            "valid_frames": int(np.asarray(cache["mask"]).sum()),
            "feature_fingerprint": cache["metadata"]["fingerprint"],
            "numeric_fingerprint": cfg["TEST_NUMERIC_FINGERPRINT"],
            "sha256": cfg["TEST_CACHE_SHA256"],
            "reused_verified_fixed_cache": True,
        }
        report["model_info"] = state["model_info"]
        report["inventory"] = prepared["inventory"]
        event(
            "fixed_cache_verified",
            "Verified fixed test features and epoch5 head; no encoder or video decode.",
        )
        event("baseline_scoring", "Score frozen B0/B1/B2 with mandatory external B0 parity gate.")
        base, base_results = workflow.benchmark(state, cache)
        report["generated_variants"] = list(base_results)
        base_files = ["baseline_gate.json", "benchmark.json", "benchmark.csv"]
        base_files.extend(f"{name}_scores.npz" for name in base_results)
        base_files.extend(f"{name}_query_changes.csv" for name in base_results if name != "B0")
        _record_artifacts(report, work, base_files)
        report["baseline_gate"] = base["baseline_gate"]
        report["baselines"] = base
        if not (
            base["baseline_gate"].get("passed") is True
            and base["baseline_gate"].get("external_baseline_reproduced") is True
        ):
            report.update(
                status="baseline_gate_failed",
                stage="stopped_before_b3",
                finished_utc=datetime.now(UTC).isoformat(),
            )
            save()
            print(
                "B3 stopped: external B0 scores/ranks did not reproduce; inspect baseline_gate.json."
            )
            return report
        if not {"B0", "B1", "B2"}.issubset(base_results):
            raise ValueError("Baseline gate passed but required frozen variants are missing")
        event("b3_scoring", "Evaluate the locked B3 epoch5 head once; no optimizer or selection.")
        started = time.perf_counter()
        scores, result, original = score_b3(state, cache, head)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        report["b3_test_scores_computed"] = True
        comparisons = {}
        for variant in ("B0", "B2"):
            comparisons[f"B3-{variant}"] = core.compare_runs(
                base_results[variant],
                result,
                cache["text_video_ids"],
                cache["video_ids"],
                cache["text_ids"],
                n_bootstrap=cfg["BOOTSTRAP_SAMPLES"],
                seed=cfg["SEED"],
                output_csv=work / f"B3_vs_{variant}_query_changes.csv",
            )
        summary = {
            "internal": workflow._public_result(result),
            "original": original,
            "seconds": elapsed,
            "selected_epoch": 5,
            "selection_policy": "fixed_epochs",
            "head_parameters": sum(p.numel() for p in head.parameters()),
            "learned_tau": float(head.temperature().detach().cpu()),
            "alpha": cfg["ALPHA"],
            "score_scale": "checkpoint_FP32_logits",
            "logit_scale": float(workflow._logit_scale(state)),
            "head_sha256": cfg["HEAD_SHA256"],
            "comparisons": comparisons,
        }
        np.savez_compressed(
            work / "B3_scores.npz",
            scores=scores,
            text_ids=np.asarray(cache["text_ids"]),
            text_video_ids=np.asarray(cache["text_video_ids"]),
            video_ids=np.asarray(cache["video_ids"]),
            score_scale=np.asarray("logits"),
            logit_scale=workflow._logit_scale(state),
        )
        report["generated_variants"].append("B3")
        _record_artifacts(
            report,
            work,
            ["B3_scores.npz", "B3_vs_B0_query_changes.csv", "B3_vs_B2_query_changes.csv"],
        )
        combined = {
            "variants": {**base["variants"], "B3": summary},
            "comparisons": {**base["comparisons"], **comparisons},
            "baseline_gate": base["baseline_gate"],
            "training_executed": False,
            "synthetic_cpu_fixture": cfg["ALLOW_CPU_DEBUG"],
        }
        pipeline.atomic_json(work / "evaluation_benchmark.json", combined)
        rows = [
            {
                "variant": name,
                "direction": direction,
                **data["internal"][direction],
                "score_seconds": data["seconds"],
            }
            for name, data in combined["variants"].items()
            for direction in ("text_to_video", "video_to_text")
        ]
        with (work / "evaluation_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _record_artifacts(report, work, ["evaluation_benchmark.json", "evaluation_benchmark.csv"])
        report.update(
            status="completed_evaluation_only",
            stage="completed",
            benchmark_metrics=combined,
            b3=summary,
            finished_utc=datetime.now(UTC).isoformat(),
        )
        save()
        print(
            f"B3 epoch5: T→V R@1={result['text_to_video']['R@1']:.2f}% | "
            f"V→T R@1={result['video_to_text']['R@1']:.2f}%",
            flush=True,
        )
        return report
    except Exception as exc:
        report.update(
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
            finished_utc=datetime.now(UTC).isoformat(),
        )
        save()
        raise

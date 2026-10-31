"""Train-only B3 entry point: locked train8100, frozen encoder, five final epochs.

Imports the reviewed portable modules beside this file. Test CSV is read only
for exclusion identities. No test media, reference scores, or dev features are
required or evaluated. Feature shards and completed-epoch checkpoints resume.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import random
import shutil
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import data_pipeline as pipeline
import numpy as np
import retrieval_core as core
import torch
import workflow

DEFAULT_CONFIG = {
    "INPUT_ROOT": "/kaggle/input",
    "WORK_DIR": "/kaggle/working/msrvtt_b3_fulltrain/jaewooo_epoch4_fp32_b3_train8100_v1",
    "VIDEO_ROOT": "",
    "TEST_CSV": "",
    "TRAIN_CSV": "",
    "CAPTIONS_JSON": "",
    "CHECKPOINT_PATH": "",
    "CHECKPOINT_SHA256": "3d7e2ba508986c7347380767233b017d4ac4c9fabc135f3457de8625dc833dca",
    "DOWNLOAD_CHECKPOINT": True,
    "HF_REVISION": "de15dc5686a8628d051dae0a85b80d34ed690c68",
    "HF_FILENAME": "ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.4",
    "EXPECTED_INPUT_SHA256": {
        "train_csv": "0a7ece2418e1f7ac5680555a04ddfef8950293ac3c4bdd6584fb0fa6cc0ff7d3",
        "test_csv": "e61144b312c8a2fc2126f96504c981f8c23c05b3ba9655a56e2af50ac8a4b16a",
        "captions_json": "3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014",
    },
    "EXPECTED_TRAIN_VIDEOS": 9000,
    "EXPECTED_TEST_VIDEOS": 1000,
    "EXPECTED_CAPTIONS_PER_VIDEO": 20,
    "EXPECTED_TRAIN_ORDER_SHA256": "722482c1fb6e560e594134ecf076a997701fbc0d10f5f0a7800ba17c0f5b1e27",
    "EXPECTED_RESERVED_ORDER_SHA256": "d7ace5316b94212bd0fc58cc913d4137f8b324368632ea518e34e62d5343a900",
    "MEDIA_MANIFEST_PATH": "",
    "MEDIA_MANIFEST_FILENAME": "msrvtt_train8100_v2.manifest.json",
    "EXPECTED_MEDIA_MANIFEST_SHA256": "",
    "REQUIRE_MEDIA_MANIFEST": True,
    "RUN_TRAIN": True,
    "SELECTION_POLICY": "fixed_epochs",
    "EPOCHS": 5,
    "DEV_VAL_SIZE": 900,
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
    "RESUME": True,
    "PREVIOUS_RUN_DIR": "",
    "ALLOW_CPU_DEBUG": False,
    "MIN_FREE_DISK_GIB": 2.0,
}
def checked_config(config):
    cfg = {**DEFAULT_CONFIG, **config}
    immutable = (
        "RUN_TRAIN", "SELECTION_POLICY", "EPOCHS", "SEED", "PRECISION", "MAX_FRAMES",
        "MAX_WORDS", "ENCODER_BATCH", "TEXT_BATCH", "TRAIN_BATCH", "LR", "WEIGHT_DECAY",
        "TRAIN_LOGIT_SCALE", "TAU", "ALPHA", "HF_REVISION", "HF_FILENAME",
    )
    for key in immutable:
        if type(cfg[key]) is not type(DEFAULT_CONFIG[key]) or cfg[key] != DEFAULT_CONFIG[key]:
            raise ValueError(f"Locked B3 setting {key} must equal {DEFAULT_CONFIG[key]!r}")
    if cfg.get("VAL_CSV") or cfg.get("VAL_PROVENANCE"):
        raise ValueError("This final-epoch schedule does not use validation input")
    for key in ("EXPECTED_TRAIN_VIDEOS", "EXPECTED_TEST_VIDEOS",
                "EXPECTED_CAPTIONS_PER_VIDEO", "DEV_VAL_SIZE"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < cfg["DEV_VAL_SIZE"] < cfg["EXPECTED_TRAIN_VIDEOS"]:
        raise ValueError("Reserved development count must be smaller than train9k count")
    if not np.isfinite(cfg["MIN_FREE_DISK_GIB"]) or cfg["MIN_FREE_DISK_GIB"] < 0:
        raise ValueError("MIN_FREE_DISK_GIB must be finite and non-negative")
    for key in ("RESUME", "ALLOW_CPU_DEBUG", "REQUIRE_MEDIA_MANIFEST"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    official = cfg["EXPECTED_INPUT_SHA256"] == DEFAULT_CONFIG["EXPECTED_INPUT_SHA256"]
    if official:
        pins = ("EXPECTED_TRAIN_VIDEOS", "EXPECTED_TEST_VIDEOS",
                "EXPECTED_CAPTIONS_PER_VIDEO", "DEV_VAL_SIZE", "CHECKPOINT_SHA256",
                "EXPECTED_TRAIN_ORDER_SHA256", "EXPECTED_RESERVED_ORDER_SHA256")
        if any(cfg[key] != DEFAULT_CONFIG[key] for key in pins):
            raise ValueError("Official metadata requires the original counts, split and checkpoint")
        if not cfg["REQUIRE_MEDIA_MANIFEST"]:
            raise ValueError("Official training requires the pinned source media manifest")
    elif not cfg["ALLOW_CPU_DEBUG"]:
        raise ValueError("Metadata/hash overrides are allowed only for explicit CPU_DEBUG fixtures")
    return cfg


def ordered_ids(records):
    order = {}
    for row in records:
        vid, position = row["video_id"], row["candidate_index"]
        if vid in order and order[vid] != position:
            raise ValueError("Conflicting candidate index for a video")
        order[vid] = position
    if len(set(order.values())) != len(order):
        raise ValueError("Distinct videos must have distinct candidate indices")
    return sorted(order, key=order.__getitem__)


def inventory_media(root, wanted):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Video input directory does not exist: {root}")
    result, duplicate = {}, set()
    for parent, _, names in os.walk(root):
        for name in names:
            path = Path(parent) / name
            if path.stem in wanted and path.suffix.lower() in {
                ".mp4", ".avi", ".mkv", ".mov", ".webm"
            }:
                if path.stem in result:
                    duplicate.add(path.stem)
                result[path.stem] = path
    if duplicate:
        raise ValueError(f"Duplicate media IDs; attach only one train Dataset: {sorted(duplicate)[:8]}")
    return result


def prepare_fulltrain(config):
    """Verify locked annotations, URL-group split, all media and optional source bytes."""
    cfg = checked_config(config)
    names = {"train_csv": ("TRAIN_CSV", "MSRVTT_train.9k.csv"),
             "test_csv": ("TEST_CSV", "MSRVTT_JSFUSION_test.csv"),
             "captions_json": ("CAPTIONS_JSON", "MSRVTT_data.json")}
    paths = {key: pipeline.resolve_file(cfg[setting], filename, cfg["INPUT_ROOT"])
             for key, (setting, filename) in names.items()}
    hashes = {key: pipeline.sha256_file(path) for key, path in paths.items()}
    if hashes != cfg["EXPECTED_INPUT_SHA256"]:
        raise ValueError("Raw annotation SHA256 differs from locked metadata")
    test_ids = [row["video_id"] for row in pipeline.read_rows(paths["test_csv"])]
    if len(set(test_ids)) != len(test_ids) or len(test_ids) != cfg["EXPECTED_TEST_VIDEOS"]:
        raise ValueError("Test exclusion identities differ from the locked pool")
    records = pipeline.training_records(paths["train_csv"], paths["captions_json"])
    if len({row["video_id"] for row in records}) != cfg["EXPECTED_TRAIN_VIDEOS"]:
        raise ValueError("Wrong official train9k video count")
    if set(Counter(row["video_id"] for row in records).values()) != {
        cfg["EXPECTED_CAPTIONS_PER_VIDEO"]
    }:
        raise ValueError("Every training video must retain all original caption rows")
    urls = pipeline.source_video_urls(paths["captions_json"])
    train, reserved, status = pipeline.make_development_split(
        records, test_ids, val_size=cfg["DEV_VAL_SIZE"], seed=cfg["SEED"],
        selection_policy="fixed_epochs", source_urls=urls,
    )
    train_ids, reserved_ids = ordered_ids(train), ordered_ids(reserved)
    if pipeline.digest(train_ids) != cfg["EXPECTED_TRAIN_ORDER_SHA256"]:
        raise ValueError("Train ordered IDs differ from source-group v2")
    if pipeline.digest(reserved_ids) != cfg["EXPECTED_RESERVED_ORDER_SHA256"]:
        raise ValueError("Reserved ordered IDs differ from source-group v2")
    train_sources = {urls[vid] for vid in train_ids}
    if set(train_ids) & (set(test_ids) | set(reserved_ids)) or train_sources & {
        urls[vid] for vid in reserved_ids
    }:
        raise ValueError("Training identities/source URLs overlap test or reserved development")
    if set(test_ids) - urls.keys():
        raise ValueError("Missing test source URLs needed for provenance audit")
    index = inventory_media(cfg["VIDEO_ROOT"] or cfg["INPUT_ROOT"], set(train_ids))
    missing = [vid for vid in train_ids if vid not in index]
    if missing:
        raise FileNotFoundError(f"Need all {len(train_ids)} train videos; missing {len(missing)}: "
                                f"{missing[:12]}; no encoder or training started")
    manifest_path, manifest_sha, source_rows = None, None, None
    if cfg["REQUIRE_MEDIA_MANIFEST"] or cfg["MEDIA_MANIFEST_PATH"]:
        manifest_path = pipeline.resolve_file(cfg["MEDIA_MANIFEST_PATH"],
                                              cfg["MEDIA_MANIFEST_FILENAME"], cfg["INPUT_ROOT"])
        manifest_sha = pipeline.sha256_file(manifest_path)
        if not cfg["EXPECTED_MEDIA_MANIFEST_SHA256"]:
            raise ValueError("Set EXPECTED_MEDIA_MANIFEST_SHA256 to the supplied source manifest pin")
        if manifest_sha != cfg["EXPECTED_MEDIA_MANIFEST_SHA256"]:
            raise ValueError("Source media manifest SHA256 differs from the supplied pin")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_rows = {row["video_id"]: row for row in manifest["videos"]}
        if len(source_rows) != len(manifest["videos"]) or set(source_rows) != set(train_ids):
            raise ValueError("Source manifest must contain exactly train8100 with no repeated IDs")
    media_rows = []
    for position, vid in enumerate(train_ids):
        path = index[vid]
        actual = {"video_id": vid, "bytes": path.stat().st_size,
                  "sha256": pipeline.sha256_file(path)}
        if source_rows is not None and any(actual[k] != source_rows[vid][k]
                                           for k in ("bytes", "sha256")):
            raise ValueError(f"Source media byte identity differs: {vid}")
        media_rows.append(actual)
        if position == 0 or (position + 1) % 1000 == 0:
            print(f"Verify train media {position + 1}/{len(train_ids)}", flush=True)
    inventory = {"train_total": len(train_ids), "train_available": len(index),
                 "full_train_media_coverage": True, "train_missing_ids": [],
                 "train_captions": len(train), "reserved_videos_not_used": len(reserved_ids),
                 "reserved_captions_not_used": len(reserved), "test_ids_exclusion_only": len(test_ids),
                 "train_reserved_shared_source_urls": 0,
                 "train_test_shared_source_urls": len(train_sources & {urls[x] for x in test_ids}),
                 "train_order_sha256": pipeline.digest(train_ids),
                 "reserved_order_sha256": pipeline.digest(reserved_ids),
                 "media_bytes": sum(row["bytes"] for row in media_rows),
                 "media_identity_sha256": pipeline.digest(media_rows),
                 "source_manifest_sha256": manifest_sha,
                 "source_media_verified": source_rows is not None}
    return {"config": cfg, "paths": paths, "input_hashes": hashes, "train_records": train,
            "reserved_records": reserved, "train_ids": train_ids, "reserved_ids": reserved_ids,
            "index": index, "inventory": inventory, "development_status": status,
            "media_rows": media_rows, "source_manifest_path": manifest_path}


def feature_bytes(videos, captions, dim=512, frames=12):
    """CPU float32 [V,F,D]/[T,D] plus bool [V,F], excluding Python/temporary storage."""
    return int(4 * dim * (captions + videos * frames) + videos * frames)


def initialize_device(allow_cpu=False, seed=42):
    """Initialize the allocator before reading/resetting CUDA statistics (PyTorch2.11)."""
    if not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("Enable a Kaggle GPU before full cache/training")
        return "cpu"
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return str(device)


def resource_observation(work):
    result = {"disk_free_bytes": shutil.disk_usage(work).free}
    try:
        import psutil

        result.update(rss_bytes=psutil.Process().memory_info().rss,
                      host_available_ram_bytes=psutil.virtual_memory().available)
    except ImportError:
        result.update(rss_bytes=None, host_available_ram_bytes=None)
    if torch.cuda.is_initialized():
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    return result


def restore_previous_run(config, identity):
    """Copy fixed resumable artifacts from an attached completed cache/run without overwrite."""
    if not config.get("PREVIOUS_RUN_DIR"):
        return []
    if not config["RESUME"]:
        raise ValueError("PREVIOUS_RUN_DIR requires RESUME=True")
    source, work = Path(config["PREVIOUS_RUN_DIR"]).resolve(), Path(config["WORK_DIR"]).resolve()
    if source == work or not source.is_dir():
        raise ValueError("PREVIOUS_RUN_DIR must be a distinct attached full-training run directory")
    registry = source / "experiment_identity.json"
    if not registry.is_file() or json.loads(registry.read_text(encoding="utf-8"))["identity"] != identity:
        raise ValueError("Previous run experiment identity differs; probe outputs cannot resume training")
    if not (source / "train_features.pt").is_file() or not (source / "train_features.json").is_file():
        raise FileNotFoundError("Previous run must include the completed train_features.pt/JSON bundle")
    names = ["train_features.pt", "train_features.json", "train_feature_manifest.json",
             "train_media_manifest.json", "preregistration.json", "heads/last.pt",
             "heads/final.pt", "heads/history.json"]
    copied = []
    for name in names:
        old, new = source / name, work / name
        if not old.is_file():
            continue
        if old.is_symlink() or not old.resolve().is_relative_to(source):
            raise ValueError(f"Refuse a previous-run artifact outside its directory: {name}")
        old_sha = pipeline.sha256_file(old)
        if new.exists():
            if not new.is_file() or pipeline.sha256_file(new) != old_sha:
                raise FileExistsError(f"Existing output differs; previous run cannot overwrite {name}")
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        temporary = new.with_name(new.name + ".restore.tmp")
        if temporary.exists():
            raise FileExistsError(f"Interrupted restore exists; inspect before retrying: {temporary}")
        shutil.copyfile(old, temporary)
        if pipeline.sha256_file(temporary) != old_sha:
            raise ValueError(f"Previous-run copy failed byte verification: {name}")
        os.replace(temporary, new)
        copied.append(name)
    return copied


def cache_settings(model_info, config, device):
    """Mirror the reviewed cache namespace contract to reject incompatible portable bundles."""
    import cv2
    import ftfy
    import PIL
    import regex

    return {**model_info, "schema": pipeline.CACHE_SCHEMA,
            "pipeline_sha256": pipeline.sha256_file(pipeline.__file__),
            "max_frames": config["MAX_FRAMES"], "max_words": config["MAX_WORDS"],
            "sampling": "original_integer_fps_1fps_then_uniform12",
            "preprocess": "PIL_bicubic_shortedge_centercrop_RGB_CLIP",
            "environment": {"torch": str(torch.__version__), "opencv": str(cv2.__version__),
                            "numpy": str(np.__version__), "python": platform.python_version(),
                            "pillow": PIL.__version__, "ftfy": str(ftfy.__version__),
                            "regex": str(regex.__version__), "cuda": str(torch.version.cuda),
                            "device": str(device),
                            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}}


def load_feature_bundle(work, identity, meta, settings):
    """Validate bytes, CPU tensors, ordered identities and semantic extraction provenance."""
    bundle, receipt = work / "train_features.pt", work / "train_features.json"
    if not bundle.exists() and not receipt.exists():
        return None
    if not bundle.is_file() or not receipt.is_file():
        raise FileNotFoundError("Incomplete train feature bundle; inspect PT/JSON pair before retry")
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    if saved.get("format") != "msrvtt_train_features_bundle_v1" or (
        saved.get("bytes") != bundle.stat().st_size
    ):
        raise ValueError("Portable train feature bundle receipt format/byte count differs")
    if saved["experiment_identity"] != identity or saved["sha256"] != pipeline.sha256_file(bundle):
        raise ValueError("Portable train feature bundle identity or SHA256 differs")
    cache = torch.load(bundle, map_location="cpu", weights_only=True)
    core.validate_cache(cache)
    if (cache["video_ids"] != meta["train_ids"]
            or cache["text_ids"] != [row["text_id"] for row in meta["train_records"]]
            or cache["text_video_ids"] != [row["video_id"] for row in meta["train_records"]]):
        raise ValueError("Portable train feature bundle ordered caption/video identities differ")
    cm = cache["metadata"]
    video_inputs = [(row["video_id"], row["sha256"]) for row in meta["media_rows"]]
    expected = {"settings": settings, "records_sha256": pipeline.digest(meta["train_records"]),
                "video_inputs_sha256": pipeline.digest(video_inputs)}
    if any(cm.get(key) != value for key, value in expected.items()):
        raise ValueError("Portable train feature settings/environment/input provenance changed")
    if cm.get("fingerprint") != pipeline.digest(expected) or cm.get("split") != "train":
        raise ValueError("Portable train feature fingerprint/split differs")
    if saved["core_cache_fingerprint"] != core._cache_fingerprint(cache):
        raise ValueError("Portable train feature tensor/metadata fingerprint differs")
    return cache


def save_feature_bundle(work, identity, cache):
    """Keep a single CPU-only resumable bundle; raw reusable shards remain untouched."""
    pipeline.atomic_torch(work / "train_features.pt", cache)
    pipeline.atomic_json(work / "train_features.json", {
        "format": "msrvtt_train_features_bundle_v1", "experiment_identity": identity,
        "sha256": pipeline.sha256_file(work / "train_features.pt"),
        "bytes": (work / "train_features.pt").stat().st_size,
        "core_cache_fingerprint": core._cache_fingerprint(cache),
        "feature_fingerprint": cache["metadata"]["fingerprint"],
    })


def run_cache_train(config, vendor_dir):
    """Build/reuse only training shards and fit unchanged B3 head for the locked schedule."""
    cfg = checked_config(config)
    work = Path(cfg["WORK_DIR"])
    work.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    report = {"format": "msrvtt_b3_fulltrain_v1", "attempt": attempt,
              "status": "started", "stage": "metadata_preflight", "config": cfg,
              "started_utc": datetime.now(UTC).isoformat(), "events": [],
              "test_scores_computed": False, "validation_scores_computed": False,
              "benchmark_metrics": None, "training_complete": False,
              "evidence_level": "metadata_only", "resource_observations": [],
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "numpy": str(np.__version__), "cuda": torch.version.cuda}}
    runtime, model = None, None

    def save():
        pipeline.atomic_json(work / f"fulltrain_attempt_{attempt}.json", report)
        pipeline.atomic_json(work / "fulltrain_report.json", report)
        pipeline.atomic_json(work / "fulltrain_log.json", report["events"])

    def event(stage, message):
        report["stage"] = stage
        report["events"].append({"time_utc": datetime.now(UTC).isoformat(), "stage": stage,
                                 "message": message})
        report["resource_observations"].append({"stage": stage, **resource_observation(work)})
        print(message, flush=True)
        save()

    try:
        meta = prepare_fulltrain(cfg)
        report.update(inventory=meta["inventory"], input_file_sha256=meta["input_hashes"],
                      development_status=meta["development_status"])
        code = {"runtime": pipeline.sha256_file(__file__),
                "data_pipeline": pipeline.sha256_file(pipeline.__file__),
                "retrieval_core": pipeline.sha256_file(core.__file__),
                "workflow": pipeline.sha256_file(workflow.__file__)}
        excluded = {"INPUT_ROOT", "WORK_DIR", "VIDEO_ROOT", "TRAIN_CSV", "TEST_CSV",
                    "CAPTIONS_JSON", "CHECKPOINT_PATH", "MEDIA_MANIFEST_PATH", "RESUME",
                    "PREVIOUS_RUN_DIR"}
        semantic = {key: value for key, value in cfg.items() if key not in excluded}
        identity = pipeline.digest({"config": semantic, "code": code,
                                    "annotation_hashes": meta["input_hashes"],
                                    "media_sha256": meta["inventory"]["media_identity_sha256"]})
        lock = work / "experiment_identity.json"
        if lock.is_file() and json.loads(lock.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Run identity changed; choose a new WORK_DIR to preserve the old run")
        if not lock.exists():
            pipeline.atomic_json(lock, {"identity": identity})
        report.update(experiment_identity=identity, code_file_sha256=code)
        report["restored_artifacts"] = restore_previous_run(cfg, identity)
        pipeline.atomic_json(work / "train_media_manifest.json", {"videos": meta["media_rows"]})
        estimated = feature_bytes(len(meta["train_ids"]), len(meta["train_records"]))
        observed = resource_observation(work)
        report["expected_feature_bytes_dim512"] = estimated
        report["resource_notes"] = [
            "CPU cache byte count excludes model, Python strings and temporary normalization tensors",
            "Probe64 measured ~55.39s cache and 0.65s/5 head steps on T4; full8100 remains unmeasured",
            "Linear sample estimate ~117min cache/~165s head is planning evidence, not a deadline",
        ]
        if observed["disk_free_bytes"] < max(cfg["MIN_FREE_DISK_GIB"] * 2**30, estimated):
            raise RuntimeError("Insufficient free disk for the configured floor/train features")
        if observed["host_available_ram_bytes"] is not None and (
            observed["host_available_ram_bytes"] < estimated
        ):
            raise RuntimeError("Available host RAM is smaller than the raw feature arrays alone")
        torch.set_num_threads(2)
        random.seed(cfg["SEED"])
        np.random.seed(cfg["SEED"])
        torch.manual_seed(cfg["SEED"])
        fixture = cfg["EXPECTED_INPUT_SHA256"] != DEFAULT_CONFIG["EXPECTED_INPUT_SHA256"]
        device = "cpu" if fixture else initialize_device(cfg["ALLOW_CPU_DEBUG"], cfg["SEED"])
        report["metadata_pin_context"] = "synthetic_cpu_fixture" if fixture else "official_msrvtt_v2"
        report["device"] = torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU_DEBUG"
        checkpoint = workflow.download_checkpoint(cfg)
        actual_sha = pipeline.sha256_file(checkpoint)
        if actual_sha != cfg["CHECKPOINT_SHA256"]:
            raise ValueError("Checkpoint SHA256 differs from the locked jaewooo epoch4 weights")
        report["checkpoint_sha256"] = actual_sha
        registration = {"config": cfg, "experiment_identity": identity,
                        "code_file_sha256": code, "input_file_sha256": meta["input_hashes"],
                        "checkpoint_sha256": actual_sha, "inventory": meta["inventory"],
                        "train_records_sha256": pipeline.digest(meta["train_records"]),
                        "reserved_records_sha256": pipeline.digest(meta["reserved_records"]),
                        "selection_policy": "fixed_epochs=5; final completed checkpoint",
                        "reserved_dev_used_for_selection": False, "test_used_for_selection": False,
                        "encoder_training_overlap": "Encoder publisher used train9k, including reserved900",
                        "environment": report["environment"]}
        if not (work / "preregistration.json").exists():
            pipeline.atomic_json(work / "preregistration.json", registration)
        pipeline.atomic_json(work / f"registration_attempt_{attempt}.json", registration)
        event("encoder_loading", "Load exact FP32 frozen encoder; training/test scoring remain disabled.")
        started = time.perf_counter()
        runtime = workflow.load_runtime({"config": cfg, "device": device,
                                         "checkpoint": checkpoint}, vendor_dir)
        model = runtime["model"]
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Encoder must remain frozen")
        report["encoder_load_and_warmup_seconds"] = time.perf_counter() - started
        report["model_info"] = runtime["model_info"]
        report["evidence_level"] = "encoder_loaded"
        event("train_cache", "Build/reuse train8100 shards and all 162000 original captions only.")
        started = time.perf_counter()
        settings = cache_settings(runtime["model_info"], cfg, device)
        cache = load_feature_bundle(work, identity, meta, settings)
        report["train_feature_bundle_reused"] = cache is not None
        if cache is None:
            cache = pipeline.build_cache(meta["train_records"], meta["index"], model,
                                         runtime["tokenizer"], runtime["model_info"], work / "cache",
                                         max_frames=cfg["MAX_FRAMES"], max_words=cfg["MAX_WORDS"],
                                         device=device, encoder_batch=cfg["ENCODER_BATCH"],
                                         text_batch=cfg["TEXT_BATCH"])
            cache["metadata"].update(split="train", development_status=meta["development_status"],
                                      encoder_training_overlap=True)
        core.validate_cache(cache)
        if cache["video_ids"] != meta["train_ids"] or len(cache["text"]) != len(meta["train_records"]):
            raise ValueError("Built cache differs from the complete locked training cohort")
        if not report["train_feature_bundle_reused"]:
            save_feature_bundle(work, identity, cache)
        report["train_cache_seconds_including_reuse_io"] = time.perf_counter() - started
        pipeline.atomic_json(work / "train_feature_manifest.json", cache["metadata"])
        report["train_cache"] = {"fingerprint": cache["metadata"]["fingerprint"],
                                 "namespace": cache["metadata"]["namespace"],
                                 "valid_frames": int(cache["mask"].sum()),
                                 "videos": len(cache["video_ids"]), "captions": len(cache["text"])}
        report["evidence_level"] = "complete_train_features_encoded_or_reused"
        event("release_encoder", "Complete train cache verified; release encoder before learning head.")
        model.cpu()
        runtime.clear()
        model, runtime = None, None
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
        head_dir = work / "heads"
        resume = bool(cfg["RESUME"] and (head_dir / "last.pt").is_file())
        event("head_training", "Learn only B3 head: five fixed epochs, final checkpoint; no dev/test access.")
        started = time.perf_counter()
        result = core.train_attention_head(
            cache, None, head_dir, epochs=cfg["EPOCHS"], batch_size=cfg["TRAIN_BATCH"],
            lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"], tau=cfg["TAU"], alpha=cfg["ALPHA"],
            logit_scale=cfg["TRAIN_LOGIT_SCALE"], seed=cfg["SEED"], device=device,
            resume=resume, selection_policy="fixed_epochs",
        )
        report["head_training_seconds_including_resume_and_checkpoint_io"] = time.perf_counter() - started
        if not result["training_complete"] or result["selected_epoch"] != cfg["EPOCHS"]:
            raise ValueError("Training did not finish the locked five-epoch schedule")
        report["training"] = {key: value for key, value in result.items() if key != "head"}
        report["checkpoint_files"] = {
            name: {"sha256": pipeline.sha256_file(head_dir / name),
                   "bytes": (head_dir / name).stat().st_size}
            for name in ("last.pt", "final.pt", "history.json")
        }
        report.update(status="completed_train_only", training_complete=True,
                      evidence_level=("synthetic_cpu_fixture_encoder_and_five_head_epochs" if fixture
                                      else "five_epoch_train_head_completed_without_benchmark_evaluation"),
                      finished_utc=datetime.now(UTC).isoformat())
        event("completed", "B3 final.pt is ready for a separate frozen-checkpoint evaluation; no R@1 computed.")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      failed_utc=datetime.now(UTC).isoformat())
        save()
        raise
    finally:
        if runtime:
            runtime.clear()
        model, runtime = None, None
        gc.collect()
        if torch.cuda.is_initialized():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                report["cleanup_warning"] = f"{type(exc).__name__}: {exc}"
                save()
    return report

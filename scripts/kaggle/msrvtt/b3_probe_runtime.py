"""Train-only, bounded B3 resource/functionality probe; never evaluates test scores.

Import beside the reviewed data_pipeline/retrieval_core/workflow modules. The
default cohort is the first 64 train8100 IDs in the official CSV candidate order,
with every original caption retained. No media are downloaded by this module.
"""

from __future__ import annotations

import gc
import json
import math
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
import torch.nn.functional as F
import workflow

DEFAULTS = {
    "INPUT_ROOT": "/kaggle/input",
    "WORK_DIR": "/kaggle/working/msrvtt_b3_probe/train64",
    "VIDEO_ROOT": "",
    "TEST_CSV": "",
    "TRAIN_CSV": "",
    "CAPTIONS_JSON": "",
    "CHECKPOINT_PATH": "",
    "CHECKPOINT_SHA256": "3d7e2ba508986c7347380767233b017d4ac4c9fabc135f3457de8625dc833dca",
    "DOWNLOAD_CHECKPOINT": True,
    "HF_REVISION": "de15dc5686a8628d051dae0a85b80d34ed690c68",
    "HF_FILENAME": "ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.4",
    "SELECTION_POLICY": "fixed_epochs",
    "EPOCHS": 5,
    "DEV_VAL_SIZE": 900,
    "EXPECTED_TRAIN_VIDEOS": 9000,
    "EXPECTED_TEST_VIDEOS": 1000,
    "EXPECTED_CAPTIONS_PER_VIDEO": 20,
    "EXPECTED_INPUT_SHA256": {
        "train_csv": "0a7ece2418e1f7ac5680555a04ddfef8950293ac3c4bdd6584fb0fa6cc0ff7d3",
        "test_csv": "e61144b312c8a2fc2126f96504c981f8c23c05b3ba9655a56e2af50ac8a4b16a",
        "captions_json": "3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014",
    },
    "EXPECTED_TRAIN_ORDER_SHA256": "722482c1fb6e560e594134ecf076a997701fbc0d10f5f0a7800ba17c0f5b1e27",
    "EXPECTED_RESERVED_ORDER_SHA256": "d7ace5316b94212bd0fc58cc913d4137f8b324368632ea518e34e62d5343a900",
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
    "PROBE_VIDEOS": 64,
    "PROBE_STEPS": 5,
    "PROBE_MAX_FEATURE_MIB": 128,
    "ALLOW_CPU_DEBUG": False,
    "MIN_FREE_DISK_GIB": 2.0,
    "RUN_TRAIN": False,
}
DEFAULT_CONFIG = DEFAULTS


def checked_config(config):
    cfg = {**DEFAULTS, **config}
    if cfg["RUN_TRAIN"] is not False:
        raise ValueError("RUN_TRAIN must be False: this notebook only performs a bounded probe")
    if cfg["SELECTION_POLICY"] != "fixed_epochs" or cfg["EPOCHS"] != 5:
        raise ValueError("B3 plan is fixed_epochs=5, final checkpoint; probe cannot select epochs")
    if cfg["PRECISION"] != "fp32" or cfg["SEED"] != 42:
        raise ValueError("Locked B3 probe uses FP32 and seed42")
    for key in ("PROBE_VIDEOS", "PROBE_STEPS", "TRAIN_BATCH", "MAX_FRAMES",
                "MAX_WORDS", "ENCODER_BATCH", "TEXT_BATCH"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 2 <= cfg["PROBE_VIDEOS"] <= 128 or not 1 <= cfg["PROBE_STEPS"] <= 5:
        raise ValueError("Probe must use 2..128 train videos and at most 5 optimizer steps")
    if cfg["TRAIN_BATCH"] < 2 or cfg["MAX_FRAMES"] != 12 or cfg["MAX_WORDS"] != 32:
        raise ValueError("Train batch must be >=2; frozen encoder contract is 12 frames/32 words")
    numeric = ("LR", "WEIGHT_DECAY", "TRAIN_LOGIT_SCALE", "TAU", "ALPHA",
               "PROBE_MAX_FEATURE_MIB", "MIN_FREE_DISK_GIB")
    if any(not math.isfinite(float(cfg[key])) for key in numeric):
        raise ValueError("Probe numeric settings must be finite")
    if (cfg["LR"] <= 0 or cfg["WEIGHT_DECAY"] < 0 or cfg["TRAIN_LOGIT_SCALE"] <= 0
            or not 0.005 <= cfg["TAU"] <= 1 or not 0 < cfg["ALPHA"] <= 1
            or not 0 < cfg["PROBE_MAX_FEATURE_MIB"] <= 128
            or cfg["MIN_FREE_DISK_GIB"] < 0):
        raise ValueError("Invalid head hyperparameters, feature cap, or disk requirement")
    return cfg


def cohort_ids(train_records, count=64):
    """Return CSV-ordered IDs, rather than caption annotation order."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("A probe needs at least two distinct training videos")
    if not train_records or any("candidate_index" not in r for r in train_records):
        raise ValueError("Probe requires original CSV candidate_index on every caption")
    order = {}
    for row in train_records:
        vid, index = row["video_id"], row["candidate_index"]
        if vid in order and order[vid] != index:
            raise ValueError("Conflicting candidate_index for a video")
        order[vid] = index
    if len(set(order.values())) != len(order):
        raise ValueError("Distinct videos cannot share a candidate_index")
    if len(order) < count:
        raise ValueError(f"Need {count} locked train videos; found {len(order)}")
    return sorted(order, key=order.__getitem__)[:count]


def select_probe_records(train_records, count=64):
    selected = set(cohort_ids(train_records, count))
    return [row for row in train_records if row["video_id"] in selected]


def inventory_media(root, wanted_ids):
    """Inventory once; missing train videos are reported, never dropped from the cohort."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"VIDEO_ROOT does not exist: {root}")
    wanted, index, duplicates = set(wanted_ids), {}, set()
    for parent, _, names in os.walk(root):
        for name in names:
            path = Path(parent) / name
            if path.suffix.lower() in {".mp4", ".avi", ".webm", ".mkv", ".mov"}:
                if path.stem in wanted:
                    if path.stem in index:
                        duplicates.add(path.stem)
                    else:
                        index[path.stem] = path
    if duplicates:
        raise ValueError(f"Ambiguous duplicate media IDs: {sorted(duplicates)[:10]}")
    return index


def prepare_probe_metadata(config):
    """Pure metadata/media preflight. Reads test IDs only to audit separation."""
    cfg = checked_config(config)
    paths = {
        "train_csv": pipeline.resolve_file(cfg["TRAIN_CSV"], "MSRVTT_train.9k.csv",
                                           cfg["INPUT_ROOT"]),
        "test_csv": pipeline.resolve_file(cfg["TEST_CSV"], "MSRVTT_JSFUSION_test.csv",
                                          cfg["INPUT_ROOT"]),
        "captions_json": pipeline.resolve_file(cfg["CAPTIONS_JSON"], "MSRVTT_data.json",
                                               cfg["INPUT_ROOT"]),
    }
    hashes = {key: pipeline.sha256_file(value) for key, value in paths.items()}
    if hashes != cfg["EXPECTED_INPUT_SHA256"]:
        raise ValueError("Raw annotation SHA256 differs from locked metadata; no media was opened")
    test_ids = [row["video_id"] for row in pipeline.read_rows(paths["test_csv"])]
    if len(test_ids) != len(set(test_ids)) or len(test_ids) != cfg["EXPECTED_TEST_VIDEOS"]:
        raise ValueError("Test identity pool differs from the locked metadata")
    records = pipeline.training_records(paths["train_csv"], paths["captions_json"])
    if len({r["video_id"] for r in records}) != cfg["EXPECTED_TRAIN_VIDEOS"]:
        raise ValueError("Train9k identity count differs from the locked metadata")
    counts = Counter(row["video_id"] for row in records)
    if set(counts.values()) != {cfg["EXPECTED_CAPTIONS_PER_VIDEO"]}:
        raise ValueError("Every train video must retain all original caption rows")
    urls = pipeline.source_video_urls(paths["captions_json"])
    train, reserved, status = pipeline.make_development_split(
        records, test_ids, val_size=cfg["DEV_VAL_SIZE"], seed=cfg["SEED"],
        selection_policy="fixed_epochs", source_urls=urls,
    )
    ids = cohort_ids(train, cfg["PROBE_VIDEOS"])
    sample = select_probe_records(train, cfg["PROBE_VIDEOS"])
    train_ids = cohort_ids(train, len({r["video_id"] for r in train}))
    reserved_ids = {r["video_id"] for r in reserved}
    reserved_order_lookup = {r["video_id"]: r["candidate_index"] for r in reserved}
    reserved_order = sorted(reserved_ids, key=reserved_order_lookup.__getitem__)
    if pipeline.digest(train_ids) != cfg["EXPECTED_TRAIN_ORDER_SHA256"]:
        raise ValueError("Source-group train ID order differs from locked v2")
    if pipeline.digest(reserved_order) != cfg["EXPECTED_RESERVED_ORDER_SHA256"]:
        raise ValueError("Source-group reserved ID order differs from locked v2")
    train_urls, reserved_urls = {urls[x] for x in train_ids}, {urls[x] for x in reserved_ids}
    if (set(ids) & (set(test_ids) | reserved_ids)) or train_urls & reserved_urls:
        raise ValueError("Train/probe overlaps test or reserved development identities/sources")
    missing_sources = set(test_ids) - urls.keys()
    if missing_sources:
        raise ValueError("Missing test source URLs needed for provenance audit")
    index = inventory_media(cfg["VIDEO_ROOT"] or cfg["INPUT_ROOT"],
                            set(train_ids) | reserved_ids | set(test_ids))
    inventory = {
        "train_total": len(train_ids),
        "train_available": sum(vid in index for vid in train_ids),
        "full_train_media_coverage": all(vid in index for vid in train_ids),
        "train_missing_ids": [vid for vid in train_ids if vid not in index],
        "reserved_total": len(reserved_ids),
        "reserved_available": len(reserved_ids & index.keys()),
        "test_media_present_but_never_opened": len(set(test_ids) & index.keys()),
        "probe_ids": ids,
        "probe_missing_ids": [vid for vid in ids if vid not in index],
        "available_train_raw_bytes": sum(index[x].stat().st_size for x in train_ids if x in index),
        "probe_raw_bytes": sum(index[x].stat().st_size for x in ids if x in index),
        "train_reserved_shared_ids": 0,
        "train_reserved_shared_source_urls": 0,
        "train_test_shared_source_urls": len(train_urls & {urls[x] for x in test_ids}),
        "source_overlap_note": "Official train/test may share source URLs; test IDs stay excluded",
        "train_order_sha256": pipeline.digest(train_ids),
        "reserved_order_sha256": pipeline.digest(reserved_order),
        "probe_order_sha256": pipeline.digest(ids),
    }
    return {"config": cfg, "paths": paths, "input_hashes": hashes,
            "train_records": train, "reserved_records": reserved,
            "cohort_records": sample, "cohort_ids": ids, "index": index,
            "inventory": inventory, "development_status": status}


def feature_bytes(video_count, caption_count, dim=512, frames=12):
    """CPU cache float32 text[T,D]/frames[V,F,D], bool mask[V,F], excluding metadata."""
    return int(4 * dim * (caption_count + video_count * frames) + video_count * frames)


def head_probe(cache, config, device="cpu"):
    """Same FP32 score/loss/optimizer path as B3; discard every learned weight."""
    cfg, cache = checked_config(config), core.validate_cache(cache)
    if "test" in str(cache.get("metadata", {}).get("split", "")).lower():
        raise ValueError("Test features cannot enter the resource probe")
    expected = feature_bytes(len(cache["video_ids"]), len(cache["text"]),
                             cache["text"].shape[1], cache["frames"].shape[1])
    if expected > cfg["PROBE_MAX_FEATURE_MIB"] * 2**20:
        raise ValueError("Sample feature cache exceeds explicit memory cap")
    selected = torch.device(device)
    torch.manual_seed(cfg["SEED"])
    rng = np.random.default_rng(cfg["SEED"])
    head = core.TinyAttentionHead(cache["text"].shape[1], cfg["TAU"]).to(selected)
    before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["LR"],
                                 weight_decay=cfg["WEIGHT_DECAY"])
    q_cpu, f_cpu = F.normalize(cache["text"], dim=-1), F.normalize(cache["frames"], dim=-1)
    captions = {vid: [] for vid in cache["video_ids"]}
    for row, vid in enumerate(cache["text_video_ids"]):
        captions[vid].append(row)
    if len(captions) < 2:
        raise ValueError("Head probe requires at least two training videos")
    rows = []
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
        torch.cuda.synchronize(selected)
    started = time.perf_counter()
    batches = []
    while len(rows) < cfg["PROBE_STEPS"]:
        if not batches:
            permutation = rng.permutation(len(cache["video_ids"]))
            batches = [permutation[i:i + cfg["TRAIN_BATCH"]]
                       for i in range(0, len(permutation), cfg["TRAIN_BATCH"])]
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2] = np.concatenate((batches[-2], batches[-1]))
                batches.pop()
        video_batch = batches.pop(0)
        vids = [cache["video_ids"][int(i)] for i in video_batch]
        caption_batch = [int(rng.choice(captions[vid])) for vid in vids]
        q = q_cpu[caption_batch].to(selected)
        frames = f_cpu[video_batch].to(selected)
        mask = cache["mask"][video_batch].to(selected)
        positives = torch.tensor([[cache["text_video_ids"][ci] == vid for vid in vids]
                                  for ci in caption_batch], dtype=torch.bool, device=selected)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=selected.type, enabled=False):
            scores = core._score_block(q, frames, mask, method="B2", tau=cfg["TAU"],
                                       alpha=cfg["ALPHA"], head=head)
            loss = core.multi_positive_contrastive_loss(scores, positives,
                                                        cfg["TRAIN_LOGIT_SCALE"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite probe loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite probe gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in head.parameters()):
            raise FloatingPointError("Non-finite head parameter after optimizer update")
        rows.append({"step": len(rows) + 1, "loss": float(loss.detach().cpu()),
                     "gradient_norm_before_clip": float(norm.detach().cpu()),
                     "videos": len(vids), "captions": len(caption_batch)})
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
    elapsed = time.perf_counter() - started
    changes = {key: float((value.detach() - before[key]).abs().max().cpu())
               for key, value in head.state_dict().items()}
    if not any(value > 0 for value in changes.values()):
        raise FloatingPointError("No head parameter changed during optimizer probe")
    return {"steps": rows, "seconds": elapsed, "parameter_max_abs_changes": changes,
            "parameters": sum(p.numel() for p in head.parameters()),
            "finite_loss_and_gradient": True, "updated_parameters": True,
            "weights_discarded": True, "checkpoint_written": False,
            "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated(selected)
                                            if selected.type == "cuda" else None)}


def process_rss_bytes():
    """Observed process RSS when the preinstalled psutil is available, otherwise unknown."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


class TimedEncoder:
    """Measure encoder compute only; cache IO/decoding/transfer remain separately labeled."""

    def __init__(self, model, device):
        self.model, self.device = model, torch.device(device)
        self.stats = {key: {"seconds": 0.0, "items": 0, "calls": 0}
                      for key in ("image", "text")}

    def _call(self, values, kind):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        output = (self.model.encode_image(values) if kind == "image"
                  else self.model.encode_text(values))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        row = self.stats[kind]
        row["seconds"] += time.perf_counter() - started
        row["items"] += len(values)
        row["calls"] += 1
        return output

    def encode_image(self, values):
        return self._call(values, "image")

    def encode_text(self, values):
        return self._call(values, "text")


def run_probe(config, vendor_dir):
    """Write bounded reports/cache and return a report; never starts full B3 training."""
    cfg = checked_config(config)
    work = Path(cfg["WORK_DIR"])
    work.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    report = {"format": "msrvtt_b3_resource_probe_v1", "attempt": attempt,
              "started_utc": datetime.now(UTC).isoformat(), "config": cfg,
              "status": "started", "stage": "metadata_preflight", "events": [],
              "full_training_started": False, "benchmark_metrics": None,
              "test_scores_computed": False, "validation_scores_computed": False,
              "selection_policy": "fixed_epochs=5; final checkpoint in future full run",
              "evidence_level": "metadata_only", "observed_rss_bytes": []}
    report["environment"] = {"python": platform.python_version(),
                             "platform": platform.platform(),
                             "torch": str(torch.__version__), "numpy": str(np.__version__)}

    def save():
        pipeline.atomic_json(work / f"probe_attempt_{attempt}.json", report)
        pipeline.atomic_json(work / "probe_report.json", report)
        pipeline.atomic_json(work / "probe_log.json", report["events"])

    def event(stage, message):
        report["stage"] = stage
        report["events"].append({"time_utc": datetime.now(UTC).isoformat(),
                                 "stage": stage, "message": message})
        print(message, flush=True)
        report["observed_rss_bytes"].append({"stage": stage, "rss": process_rss_bytes()})
        save()

    runtime, model, timed = None, None, None
    try:
        meta = prepare_probe_metadata(cfg)
        report.update(inventory=meta["inventory"], input_file_sha256=meta["input_hashes"],
                      development_status=meta["development_status"],
                      probe_captions=len(meta["cohort_records"]),
                      full_train_captions=len(meta["train_records"]))
        report["metadata_pin_context"] = (
            "official_msrvtt_w03_v2" if cfg["EXPECTED_INPUT_SHA256"] == DEFAULTS["EXPECTED_INPUT_SHA256"]
            else "explicit_expected_hash_override_for_fixture_or_separate_protocol"
        )
        report["code_file_sha256"] = {
            "runtime": pipeline.sha256_file(__file__),
            "data_pipeline": pipeline.sha256_file(pipeline.__file__),
            "retrieval_core": pipeline.sha256_file(core.__file__),
            "workflow": pipeline.sha256_file(workflow.__file__),
        }
        semantic = {key: value for key, value in cfg.items()
                    if key not in {"INPUT_ROOT", "WORK_DIR", "VIDEO_ROOT", "TEST_CSV",
                                   "TRAIN_CSV", "CAPTIONS_JSON", "CHECKPOINT_PATH"}}
        identity = pipeline.digest({"config": semantic, "inputs": meta["input_hashes"],
                                    "cohort_ids": meta["cohort_ids"],
                                    "code": report["code_file_sha256"]})
        lock = work / "probe_identity.json"
        if lock.is_file() and json.loads(lock.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Probe identity changed; use a new WORK_DIR to preserve prior evidence")
        pipeline.atomic_json(lock, {"identity": identity})
        report["probe_identity"] = identity
        report["disk_free_bytes"] = shutil.disk_usage(work).free
        report["sample_feature_bytes_assuming_dim512"] = feature_bytes(
            len(meta["cohort_ids"]), len(meta["cohort_records"]))
        report["full_train_feature_bytes_assuming_dim512"] = feature_bytes(
            meta["inventory"]["train_total"], len(meta["train_records"]))
        if meta["inventory"]["probe_missing_ids"]:
            report["status"] = "blocked_missing_probe_media"
            event("metadata_preflight", f"Need {len(meta['inventory']['probe_missing_ids'])} "
                  "locked probe videos; no encoder download/inference/training was started.")
            return report
        if report["disk_free_bytes"] < cfg["MIN_FREE_DISK_GIB"] * 2**30:
            raise RuntimeError("Insufficient free disk for the explicitly configured resource floor")
        if report["sample_feature_bytes_assuming_dim512"] > cfg["PROBE_MAX_FEATURE_MIB"] * 2**20:
            raise ValueError("Expected sample feature cache exceeds configured cap")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and not cfg["ALLOW_CPU_DEBUG"]:
            raise RuntimeError("GPU is required for resource measurements; CPU_DEBUG is explicit")
        torch.set_num_threads(2)
        random.seed(cfg["SEED"])
        np.random.seed(cfg["SEED"])
        torch.manual_seed(cfg["SEED"])
        if device.startswith("cuda"):
            # is_available/manual_seed do not initialize the CUDA allocator.
            # PyTorch 2.11 rejects allocator statistics until explicit init.
            selected_cuda = torch.device(device)
            torch.cuda.init()
            torch.cuda.set_device(selected_cuda)
            torch.cuda.manual_seed_all(cfg["SEED"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.cuda.synchronize(selected_cuda)
            torch.cuda.reset_peak_memory_stats(selected_cuda)
        checkpoint = workflow.download_checkpoint(cfg)
        actual_sha = pipeline.sha256_file(checkpoint)
        if actual_sha != cfg["CHECKPOINT_SHA256"]:
            raise ValueError("Checkpoint SHA256 differs from the locked epoch4 model")
        report.update(device=(torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU_DEBUG"),
                      checkpoint_sha256=actual_sha, torch=str(torch.__version__),
                      cuda=torch.version.cuda)
        event("encoder_loading", "Load the locked FP32 frozen encoder; warm up once.")
        started = time.perf_counter()
        runtime = workflow.load_runtime({"config": cfg, "device": device,
                                         "checkpoint": checkpoint}, vendor_dir)
        report["encoder_load_and_warmup_seconds"] = time.perf_counter() - started
        report["evidence_level"] = "encoder_loaded_and_warmed_up"
        model = runtime["model"]
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Encoder must remain frozen")
        dim = runtime["model_info"]["embed_dim"]
        actual_features = feature_bytes(len(meta["cohort_ids"]), len(meta["cohort_records"]), dim)
        if actual_features > cfg["PROBE_MAX_FEATURE_MIB"] * 2**20:
            raise ValueError("Actual embedding dimension exceeds sample feature cap")
        event("sample_cache", "Encode only the locked train cohort and its original captions.")
        timed = TimedEncoder(model, device)
        cache_started = time.perf_counter()
        cache = pipeline.build_cache(meta["cohort_records"], meta["index"], timed,
                                     runtime["tokenizer"], runtime["model_info"], work / "cache",
                                     max_frames=cfg["MAX_FRAMES"], max_words=cfg["MAX_WORDS"],
                                     device=device, encoder_batch=cfg["ENCODER_BATCH"],
                                     text_batch=cfg["TEXT_BATCH"])
        cache["metadata"]["split"] = "train_resource_probe"
        core.validate_cache(cache)
        cache_seconds = time.perf_counter() - cache_started
        report["sample_cache"] = {"seconds_including_io_decode_transfer": cache_seconds,
                                  "encoder_compute": timed.stats,
                                  "feature_bytes": actual_features,
                                  "valid_frame_count": int(cache["mask"].sum()),
                                  "fingerprint": cache["metadata"]["fingerprint"],
                                  "namespace": cache["metadata"]["namespace"]}
        if device.startswith("cuda"):
            report["encoder_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            report["encoder_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        report["evidence_level"] = "sample_features_encoded_or_reused"
        model.cpu()
        runtime.clear()
        model, timed = None, None
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        event("head_steps", "Release encoder GPU memory; run at most five disposable head steps.")
        report["head_probe"] = head_probe(cache, cfg, device)
        train_ratio = meta["inventory"]["train_total"] / len(meta["cohort_ids"])
        text_ratio = len(meta["train_records"]) / len(meta["cohort_records"])
        stats = report["sample_cache"]["encoder_compute"]
        fresh = (stats["image"]["items"] == int(cache["mask"].sum())
                 and stats["text"]["items"] == len(cache["text"]))
        steps_full = math.ceil(meta["inventory"]["train_total"] / cfg["TRAIN_BATCH"])
        report["estimates"] = {
            "basis": "Linear sample extrapolation, no full-train run; not a memory-fit guarantee",
            "sample_cache_fully_cold": fresh,
            "full_train_cache_seconds_linear": cache_seconds * max(train_ratio, text_ratio)
                if fresh else None,
            "full_train_encoder_compute_seconds_only": (
                stats["image"]["seconds"] * train_ratio + stats["text"]["seconds"] * text_ratio
            ) if fresh else None,
            "full_train_feature_bytes": feature_bytes(meta["inventory"]["train_total"],
                                                        len(meta["train_records"]), dim),
            "full_train_raw_bytes_from_sample_mean": meta["inventory"]["probe_raw_bytes"] * train_ratio,
            "five_epoch_head_seconds_linear": report["head_probe"]["seconds"]
                / cfg["PROBE_STEPS"] * steps_full * cfg["EPOCHS"],
            "caveats": ["Cold cache time estimate unavailable when shards were reused",
                        "Video duration, disk IO, GPU contention and host memory may vary",
                        "Feature bytes exclude Python strings, optimizer, encoder weights and workspace",
                        "Head estimate excludes full-cache normalization, IO and epoch checkpoint writes",
                        "A small sample does not prove full8100 coverage or fit; inspect inventory first"],
        }
        report.update(status="completed_probe_only", evidence_level="sample_encoder_and_head_steps",
                      finished_utc=datetime.now(UTC).isoformat())
        event("completed", "Probe complete. Weights discarded; no test/dev score or full training.")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      failed_utc=datetime.now(UTC).isoformat())
        save()
        raise
    finally:
        if runtime:
            runtime.clear()
        model, timed = None, None
        gc.collect()
        if torch.cuda.is_initialized():
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_error:
                report["cleanup_warning"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                save()
    return report

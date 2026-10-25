"""Train frozen-feature heads, audit readiness, or explicitly run a synthetic code probe."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from temporal_hoi.data.msrvtt import MSRVTTAdapter, RetrievalPool, file_sha256, semantic_hash
from temporal_hoi.models.alignment import build_alignment_head
from temporal_hoi.training.alignment import (
    distinct_video_batches,
    train_alignment,
    validate_config,
    write_json,
)
from temporal_hoi.training.features import (
    FrozenFeatureSplit,
    encoder_descriptor,
    load_cached_split,
    validate_training_splits,
)
from temporal_hoi.training.losses import bidirectional_multi_positive_loss
from temporal_hoi.training.resources import enforce_resources


def alignment_readiness(config):
    adapter = MSRVTTAdapter.from_files(config["metadata_archive"], config["metadata_lock"])
    coverage = {split: adapter.media_coverage(split, config["media_root"])
                for split in ("train", "validation")}
    cache = Path(config["cache_dir"]) / "cache_manifest.json"
    missing = {split: row["missing_video_ids"] for split, row in coverage.items()}
    if config["allow_subset"] and cache.is_file():
        manifest = json.loads(cache.read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            if split in manifest.get("splits", {}):
                pool = adapter.pool(split, selected_video_ids=manifest["splits"][split]["video_ids"])
                missing[split] = [v for v in pool.video_ids if v in missing[split]]
    blockers = [f"{split}: {len(ids)} missing MSR-VTT video files"
                for split, ids in missing.items() if ids]
    if not cache.is_file():
        blockers.append("Verified frozen-feature cache_manifest.json is absent")
    if config["crop_mode"] == "union":
        boxes = config.get("boxes")
        box_paths = list(boxes.values()) if isinstance(boxes, dict) else [boxes]
        if not box_paths or any(not path or not Path(path).is_file() for path in box_paths):
            blockers.append("Verified person/object union boxes are absent")
        if not config.get("cohort_lock") or not Path(config["cohort_lock"]).is_file():
            blockers.append("Locked shared global/local comparison cohort is absent")
    return adapter, {
        "schema_version": "alignment_readiness_v1", "status": "blocked" if blockers else "inputs_present",
        "benchmark_training_status": "not_run", "benchmark_training_run": False,
        "benchmark_metrics": None, "checkpoint_path": None,
        "missing_ids": missing, "media_coverage": coverage,
        "cache_manifest_path": str(cache), "cache_manifest_present": cache.is_file(),
        "blockers": blockers, "selection_split": "validation", "test_accessed": False,
        "development_split": "sourcegroup_v2", "protocol_lock_sha256": file_sha256(config["metadata_lock"]),
        "config_sha256": semantic_hash(config),
        "planned_output_dir": config["output_dir"],
        "code_probe_scope": "synthetic_technical_probe does not measure benchmark quality",
        "input_integrity_validated": False,
    }


def run_synthetic_probe(config, output_dir, steps=3):
    """An explicit seeded tensor fixture, saved as evidence; no pilot/benchmark labels."""
    if steps < 1:
        raise ValueError("Probe steps must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probe_config = copy.deepcopy(config)
    # One distinct-video batch per epoch makes the requested step count exact.
    # Extend patience only for the probe if more than the normal patience steps
    # were explicitly requested, keeping that deviation recorded.
    probe_config["max_epochs"] = max(config["max_epochs"], steps + 1)
    probe_config["patience"] = max(config["patience"], steps)
    # Hold inputs fixed when comparing optimization seeds and head capacity.
    fixture_seed = config.get("probe_fixture_seed", 42)
    random = np.random.default_rng(fixture_seed)
    dimension, count = 12, 6
    rotation, _ = np.linalg.qr(random.normal(size=(dimension, dimension)))
    arrays = {}
    pools = {}
    for index, split in enumerate(("train", "validation")):
        ids = tuple(f"synthetic_{split}_video_{i}" for i in range(count))
        rows = tuple({"text_id": f"synthetic_{split}_text_{i}_{j}", "video_id": ids[i],
                      "caption": f"synthetic tensor fixture {i} caption {j}"}
                     for i in range(count) for j in range(2))
        pools[split] = RetrievalPool(split, ids, rows, count, len(rows))
        video = np.eye(dimension)[index * count:(index + 1) * count]
        text = np.repeat(video @ rotation, 2, axis=0)
        text += random.normal(scale=0.015, size=text.shape)
        arrays[f"{split}_video_embeddings"] = video.astype(np.float32)
        arrays[f"{split}_text_embeddings"] = text.astype(np.float32)
        arrays[f"{split}_video_ids"] = np.asarray(ids)
        arrays[f"{split}_text_ids"] = np.asarray(pools[split].text_ids)
    feature_path = output / "synthetic_features.npz"
    np.savez_compressed(feature_path, **arrays)
    input_sha = file_sha256(feature_path)
    encoder = {"model_name": "seeded_synthetic_tensor_fixture", "checkpoint_sha256": input_sha,
               "checkpoint_kind": "synthetic_features_npz_not_encoder_weights", "backbone_frozen": True}
    splits = {}
    for split in ("train", "validation"):
        pool = pools[split]
        provenance = {
            "dataset": "synthetic_tensor_fixture", "artifact_kind": "synthetic_technical_probe",
            "encoder": encoder, "protocol_lock_sha256": semantic_hash({s: p.hashes()
                                                                        for s, p in pools.items()}),
            "pool_hashes": pool.hashes(), "scope": "synthetic_fixture_only",
            "raw_caption_sha256": semantic_hash(list(pool.queries)),
            "feature_sha256": input_sha, "relevance_policy": "synthetic_identity_complete",
        }
        splits[split] = FrozenFeatureSplit(pool, arrays[f"{split}_video_embeddings"],
                                           arrays[f"{split}_text_embeddings"],
                                           tuple(f"synthetic_source_{v}" for v in pool.video_ids),
                                           provenance)
    if config["feature_batch_videos"] < count:
        probe_config["feature_batch_videos"] = count
    paused = train_alignment(splits["train"], splits["validation"], probe_config, output,
                             stop_after_epoch=steps)
    before_resume = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    resumed = train_alignment(splits["train"], splits["validation"], probe_config, output,
                              resume=output / "last.pt", stop_after_epoch=steps + 1)
    after_resume = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    # Exercise a feature tensor requiring gradients and prove that the frozen
    # boundary detaches it while preserving head gradients.
    frozen_input = torch.from_numpy(arrays["train_video_embeddings"]).requires_grad_(True)
    head = build_alignment_head(probe_config, dimension, dimension)
    head.encode_video(frozen_input)[:, 0].sum().backward()
    result = {
        "schema_version": "alignment_probe_v1", "status": "PASS_SYNTHETIC_TECHNICAL_PROBE",
        "execution_level": "synthetic_technical_probe", "requested_probe_steps": steps,
        "completed_probe_steps": paused["step_count"], "resume_extra_steps": resumed["step_count"] - steps,
        "resume_optimizer_state_loaded": len(before_resume["optimizer_state_dict"]["state"]) > 0,
        "resume_optimizer_steps_advanced": all(
            float(after_resume["optimizer_state_dict"]["state"][key]["step"])
            > float(state["step"])
            for key, state in before_resume["optimizer_state_dict"]["state"].items()),
        "head_gradient_finite_nonzero": resumed["last_head_gradient_norm"] > 0,
        "head_gradient_norm": resumed["last_head_gradient_norm"],
        "head_update_norm": resumed["head_update_norm"], "all_losses_finite": resumed["all_losses_finite"],
        "frozen_input_grad": None if frozen_input.grad is None else "unexpected_gradient",
        "encoder_loaded": False, "encoder_grad": None,
        "encoder_gradient_evidence": "No backbone loaded; detached frozen-feature boundary directly verified",
        "validation_scope": "6 synthetic videos and 12 tensor captions; diagnostic only",
        "benchmark_training_status": "not_run", "benchmark_training_run": False,
        "benchmark_metrics": None, "test_accessed": False,
        "synthetic_feature_sha256": input_sha, "probe_config": probe_config,
        "fixture_seed": fixture_seed, "optimization_seed": config["seed"],
        "probe_checkpoint_sha256": resumed["last_checkpoint_sha256"],
        "trainable_parameters": resumed["trainable_parameters"],
        "elapsed_seconds": resumed["elapsed_seconds"],
        "observed_rss_bytes": resumed["observed_rss_bytes"],
        "limitations": ["No real video encoder, MSR-VTT media or retrieval benchmark was trained",
                        "Head changes and finite loss prove code execution only",
                        "Synthetic validation scores are not evidence of benchmark improvement"],
    }
    if (not result["head_gradient_finite_nonzero"] or result["head_update_norm"] <= 0
            or not result["resume_optimizer_steps_advanced"] or frozen_input.grad is not None):
        raise AssertionError("Synthetic probe gradient/update/resume check failed")
    write_json(output / "probe.json", result)
    return result


def run_feature_probe(train, validation, config, output_dir, steps):
    """Exactly N optimizer steps on real train cache; no epoch/checkpoint selection."""
    if steps < 1:
        raise ValueError("Probe steps must be positive")
    validate_training_splits(train, validation)
    output = Path(output_dir)
    if (output / "probe.pt").exists():
        raise ValueError("Feature probe output already exists; choose a fresh probe directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    head = build_alignment_head(config, train.video_embeddings.shape[1],
                                train.text_embeddings.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    initial = {key: tensor.detach().clone() for key, tensor in head.state_dict().items()}
    generator = torch.Generator().manual_seed(config["seed"])
    owners = np.asarray([q["video_id"] for q in train.pool.queries])
    started = time.monotonic()
    losses, gradients = [], []
    frozen_video, frozen_text, labels = None, None, None
    while len(losses) < steps:
        for indices in distinct_video_batches(len(train.pool.video_ids), config["feature_batch_videos"], generator):
            enforce_resources(config, started)
            video_ids = np.asarray(train.pool.video_ids)[indices]
            text_indices = np.flatnonzero(np.isin(owners, video_ids))
            frozen_video = torch.from_numpy(train.video_embeddings[indices]).requires_grad_(True)
            frozen_text = torch.from_numpy(train.text_embeddings[text_indices]).requires_grad_(True)
            labels = torch.from_numpy(train.relations(indices, text_indices))
            optimizer.zero_grad(set_to_none=True)
            loss, _ = bidirectional_multi_positive_loss(head(frozen_video, frozen_text), labels,
                                                        temperature=config["temperature"])
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise AssertionError("Feature probe head gradient is missing or nonfinite")
            gradient = float(sum(p.grad.square().sum() for p in head.parameters()).sqrt())
            if gradient <= 0 or frozen_video.grad is not None or frozen_text.grad is not None:
                raise AssertionError("Feature probe needs a nonzero head gradient and frozen inputs")
            optimizer.step()
            losses.append(float(loss.detach()))
            gradients.append(gradient)
            if len(losses) == steps:
                break
    checkpoint = {
        "schema_version": "alignment_feature_probe_v1", "model_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": config, "seed": config["seed"],
        "step_count": steps, "encoder_sha256": train.provenance["encoder"]["checkpoint_sha256"],
        "split_hashes": {"train": train.pool.hashes(), "validation": validation.pool.hashes()},
        "execution_level": "frozen_feature_code_probe", "input_provenance": train.provenance,
    }
    torch.save(checkpoint, output / "probe.pt")
    stored = torch.load(output / "probe.pt", weights_only=True)
    old_steps = {key: float(state["step"]) for key, state in stored["optimizer_state_dict"]["state"].items()}
    restored = build_alignment_head(config, train.video_embeddings.shape[1],
                                    train.text_embeddings.shape[1])
    restored.load_state_dict(stored["model_state_dict"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=config["lr"])
    restored_optimizer.load_state_dict(stored["optimizer_state_dict"])
    restored_optimizer.zero_grad(set_to_none=True)
    extra_loss, _ = bidirectional_multi_positive_loss(restored(frozen_video, frozen_text), labels,
                                                     temperature=config["temperature"])
    extra_loss.backward()
    restored_optimizer.step()
    advanced = all(float(restored_optimizer.state_dict()["state"][key]["step"]) > value
                   for key, value in old_steps.items())
    if not advanced:
        raise AssertionError("Restored feature probe optimizer did not advance")
    result = {
        "status": "PASS_FROZEN_FEATURE_CODE_PROBE", "execution_level": "frozen_feature_code_probe",
        "requested_probe_steps": steps, "completed_probe_steps": len(losses), "losses": losses,
        "step_count": steps,
        "head_gradient_finite_nonzero": True, "head_gradient_norms": gradients,
        "head_update_norm": float(sum((head.state_dict()[key] - value).square().sum()
                                      for key, value in initial.items()).sqrt()),
        "all_losses_finite": all(np.isfinite(losses)), "frozen_input_grad": None,
        "encoder_loaded": False, "encoder_grad": None,
        "encoder_gradient_evidence": "Backbone features are cached; input detachment directly verified",
        "resume_optimizer_state_loaded": True, "resume_optimizer_steps_advanced": advanced,
        "probe_checkpoint_sha256": file_sha256(output / "probe.pt"),
        "benchmark_training_status": "not_run", "benchmark_training_run": False,
        "benchmark_metrics": None, "test_accessed": False, "checkpoint_selection_performed": False,
    }
    write_json(output / "probe.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/frozen_feature_alignment.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--probe-steps", type=int)
    parser.add_argument("--synthetic-probe", action="store_true",
                        help="Explicit synthetic tensor code probe; never benchmark training")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    validate_config(config)
    if args.synthetic_probe:
        if args.resume or args.readiness_only:
            parser.error("Synthetic probe manages its own checkpoint-resume verification")
        output = args.output_dir or Path("outputs/experiments/pilot/alignment_and_profiling/alignment/probe")
        report = run_synthetic_probe(config, output, args.probe_steps or 3)
        print(json.dumps({"status": report["status"], "output": str(output / "probe.json"),
                          "head_update_norm": report["head_update_norm"],
                          "benchmark_training_run": False}, indent=2))
        return 0
    adapter, readiness = alignment_readiness(config)
    readiness_path = Path(config["output_dir"]).parent / "readiness.json"
    write_json(readiness_path, readiness)
    if args.readiness_only or readiness["status"] == "blocked":
        print(json.dumps({"status": readiness["status"], "output": str(readiness_path),
                          "missing_video_count": {s: len(ids) for s, ids in readiness["missing_ids"].items()},
                          "benchmark_training_run": False}, indent=2))
        return 0 if args.readiness_only else 2
    encoder_run = json.loads(Path(config["encoder_run_json"]).read_text(encoding="utf-8-sig"))
    descriptor = encoder_descriptor(config, encoder_run)
    checkpoint = Path(encoder_run["checkpoint"]["path"])
    if file_sha256(checkpoint) != descriptor["checkpoint_sha256"]:
        raise ValueError("Local encoder checkpoint hash differs from locked encoder run")
    manifest = json.loads((Path(config["cache_dir"]) / "cache_manifest.json").read_text(encoding="utf-8"))
    cache_started = time.monotonic()
    features = {split: load_cached_split(config, adapter, manifest, split, descriptor, started=cache_started)
                for split in ("train", "validation")}
    if args.probe_steps is not None:
        if args.resume:
            parser.error("Feature probe handles its own optimizer-load check; --resume is for epoch training")
        result = run_feature_probe(features["train"], features["validation"], config,
                                   args.output_dir or Path("outputs/experiments/pilot/alignment_and_profiling/alignment/probe"), args.probe_steps)
    else:
        result = train_alignment(features["train"], features["validation"], config,
                                  args.output_dir or config["output_dir"], resume=args.resume)
    print(json.dumps({"status": result["status"], "steps": result["step_count"],
                      "benchmark_training_run": result["benchmark_training_run"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

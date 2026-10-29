"""Run a locked upstream-train VidHOI frozen-feature head development study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from temporal_hoi.data.hoi_head_cache import (
    cache_head_features,
    config_fingerprint,
    load_head_cache,
    preflight_head_data,
)
from temporal_hoi.data.vidhoi import HOIPair, VidHOIFrame
from temporal_hoi.evaluation.hoi import evaluate_frame_hoi
from temporal_hoi.evaluation.hoi_head_audit import summarize_development_regimes
from temporal_hoi.inference.hoi_static import file_sha256, proposal_pairs
from temporal_hoi.models.hoi_head import MODES, HOIResidualHead
from temporal_hoi.training.hoi_head_study import (
    SEEDS,
    evaluate_head_state,
    run_head_study,
    study_config,
)

INPUT_KEYS = ("train_manifest", "dev_manifest", "protocol", "mapping", "checkpoint")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def record(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
            "bytes": path.stat().st_size}


def inventory(root: Path) -> list[dict]:
    return [record(path, root) for path in sorted(root.rglob("*")) if path.is_file()]


def verify_inventory(root: Path, receipt: dict) -> None:
    listed = receipt.get("files")
    if not isinstance(listed, list) or not listed:
        raise ValueError("Missing stage output inventory")
    names = set()
    for entry in listed:
        name = entry["path"]
        path = root / name
        if (Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve())
                or path.is_symlink() or name in names):
            raise ValueError("Unsafe or duplicate stage inventory path")
        names.add(name)
        if record(path, root) != entry:
            raise ValueError(f"Stage output identity mismatch: {name}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != names | {"receipt.json"}:
        raise ValueError("Unexpected/missing stage outputs")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config(path: Path) -> dict:
    config = load_json(path)
    for key in INPUT_KEYS:
        config[key] = str((path.parent / config[key]).resolve())
    return config


def load_caches(root: Path, expected: dict, config: dict, checked: dict) -> tuple:
    receipt = load_json(root / "receipt.json")
    if (receipt.get("schema_version") != "vidhoi_head_cache_receipt_v1"
            or receipt.get("status") != "completed"
            or receipt.get("level") != "vidhoi_head_frozen_clip_feature_cache"
            or receipt.get("encoder", {}).get("test_fixture") is not False
            or receipt.get("input_hashes") != expected
            or receipt.get("config_sha256") != config_fingerprint(config)):
        raise ValueError("Cache is incomplete, synthetic or from different inputs")
    verify_inventory(root, receipt)
    train, dev = load_head_cache(root / "train"), load_head_cache(root / "validation")
    for cache, frames in ((train, checked["train_frames"]), (dev, checked["dev_frames"])):
        if (cache["frames"] != frames or cache["mapping"] != checked["mapping"]
                or cache["metadata"].get("config_sha256") != config_fingerprint(config)
                or cache["metadata"].get("input_hashes") != expected):
            raise ValueError("Cached geometry/GT/identity/config differs from current input")
    with np.load(root / "text_features.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"text_features", "candidate_mask"}:
            raise ValueError("Unexpected text cache arrays")
        texts = arrays["text_features"].copy()
        allowed = arrays["candidate_mask"].copy()
    expected_mask = np.ones((len(checked["mapping"].objects),
                             len(checked["mapping"].predicates)), dtype=bool)
    for _, predicate, obj in checked["protocol"]["final_unseen_triplets"]:
        expected_mask[obj, predicate] = False
    if (allowed.dtype != np.bool_ or not np.array_equal(allowed, expected_mask)
            or texts.shape[:2] != expected_mask.shape or np.any(texts[~expected_mask] != 0)):
        raise ValueError("Cached text vocabulary differs from current held-out protocol")
    return train, dev, texts, allowed, receipt


def prediction_frames(rows: list[dict]) -> list[VidHOIFrame]:
    return [VidHOIFrame(**{**row,
        "ground_truth": tuple(HOIPair(**pair) for pair in row["ground_truth"]),
        "predictions": tuple(HOIPair(**pair) for pair in row["predictions"])}) for row in rows]


def replay_one(directory: Path, head, dev, texts, allowed, mode, device) -> dict:
    """Independently reload the saved selected tensors and replay the evaluator."""
    result = evaluate_head_state(head, dev, texts, allowed, mode, device=device)
    persisted = load_json(directory / "predictions.json")
    regenerated = json.loads(json.dumps([asdict(frame) for frame in result["frames"]]))
    if len(persisted) != len(regenerated):
        raise ValueError("Persisted prediction frame count mismatch")
    max_difference = 0.0
    for old, new in zip(persisted, regenerated, strict=True):
        old_pairs, new_pairs = old.pop("predictions"), new.pop("predictions")
        if old != new or len(old_pairs) != len(new_pairs):
            raise ValueError("Persisted frame/GT identity mismatch")
        for old_pair, new_pair in zip(old_pairs, new_pairs, strict=True):
            a = np.asarray(old_pair.pop("predicate_values"), dtype=np.float64)
            b = np.asarray(new_pair.pop("predicate_values"), dtype=np.float64)
            if old_pair != new_pair or a.shape != b.shape or not np.allclose(a, b, atol=1e-6, rtol=1e-6):
                raise ValueError("Selected-state prediction replay mismatch")
            max_difference = max(max_difference, float(np.max(np.abs(a - b), initial=0)))
    metrics = load_json(directory / "metrics.json")
    persisted_frames = prediction_frames(load_json(directory / "predictions.json"))
    if evaluate_frame_hoi(persisted_frames) != metrics or result["metrics"] != metrics:
        raise ValueError("Persisted or regenerated metric replay mismatch")
    if result["known_positive_ranks"] != load_json(directory / "known_positive_ranks.json"):
        raise ValueError("Selected-state known-positive rank replay mismatch")
    return {"mode": mode, "max_score_absolute_difference": max_difference,
            "metric_replay_exact": True, "known_positive_rank_replay_exact": True}


def evaluate_study(config: dict, root: Path, expected: dict) -> dict:
    # BLAS reductions can change ties/AP across thread counts, even with the same
    # FP32 tensors. Restore the training execution settings in this new process.
    settings = study_config(config)
    torch.set_num_threads(settings["threads"])
    checked = preflight_head_data(config)
    train, dev, texts, allowed, cache = load_caches(root / "cache", expected, config, checked)
    training_root = root / "training"
    training = load_json(training_root / "receipt.json")
    if load_json(training_root / "study_config.json").get("study") != settings:
        raise ValueError("Replay execution settings differ from recorded training settings")
    if (training.get("schema_version") != "vidhoi_head_study_result_v1"
            or training.get("status") != "completed"
            or training.get("level") != "oracle_frozen_feature_development_head_study"
            or training.get("model_training_run") is not True
            or training.get("selected_best_seed") is not None
            or training.get("modes") != list(MODES) or training.get("seeds") != list(SEEDS)):
        raise ValueError("Missing real completed three-mode, three-seed head study")
    verify_inventory(training_root, training)
    device = settings["device"]
    replayed, frozen = [], []
    regimes = {"frozen": {}, "selected": {}}
    for mode in MODES:
        frozen.append(replay_one(training_root / "frozen" / mode, None, dev, texts, allowed,
                                 mode, device))
        regimes["frozen"][mode] = summarize_development_regimes(
            load_json(training_root / "frozen" / mode / "metrics.json"),
            checked["protocol"], train["frames"])
        for seed in SEEDS:
            directory = training_root / "runs" / mode / f"seed_{seed}"
            metadata = load_json(directory / "metadata.json")
            if (metadata.get("optimizer_steps", 0) <= 0
                    or metadata.get("nonzero_gradient_observed") is not True
                    or metadata.get("last_parameter_delta_l2", 0) <= 0
                    or metadata.get("last_embedding_delta_l2", 0) <= 0):
                raise ValueError("Missing optimizer/gradient/last-state learning evidence")
            for checkpoint in metadata["checkpoints"].values():
                if record(directory / checkpoint["path"], directory) != checkpoint:
                    raise ValueError("Saved checkpoint hash mismatch")
            head = HOIResidualHead(int(texts.shape[-1]), config.get("study", {}).get("hidden_dim", 128))
            state = load_file(str(directory / "selected_head.safetensors"), device="cpu")
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Selected checkpoint contains invalid tensors")
            head.load_state_dict(state, strict=True)
            result = replay_one(directory, head, dev, texts, allowed, mode, device)
            replayed.append({**result, "seed": seed, "selected_epoch": metadata["selected_epoch"]})
            regimes["selected"][f"{mode}/seed_{seed}"] = summarize_development_regimes(
                load_json(directory / "metrics.json"), checked["protocol"], train["frames"])
    evaluation = root / "evaluation"
    evaluation.mkdir(exist_ok=False)
    write_json(evaluation / "replays.json", {"selected": replayed, "frozen": frozen})
    write_json(evaluation / "regime_summaries.json", regimes)
    receipt = {"schema_version": "vidhoi_head_evaluation_audit_v1", "status": "completed",
        "selected_state_replayed": len(replayed), "frozen_baselines_replayed": len(frozen),
        "input_hashes": expected, "cache_receipt_sha256": file_sha256(root / "cache/receipt.json"),
        "training_receipt_sha256": file_sha256(training_root / "receipt.json"),
        "backbone_inference_run": False, "model_training_run": False,
        "torch_num_threads": torch.get_num_threads(), "runtime_settings_match_training": True,
        "official_final_split_accessed": False, "final_unseen_accessed": False,
        "files": inventory(evaluation)}
    write_json(evaluation / "receipt.json", receipt)
    outer = {"schema_version": "vidhoi_head_study_run_v1", "status": "completed",
        "level": "internal_oracle_head_development", "training_run": True,
        "model_training_run": True, "official_final_split_accessed": False,
        "final_unseen_accessed": False, "full_benchmark_reproduction_claim": False,
        "experiment_heads": list(MODES), "seeds": list(SEEDS), "input_hashes": expected,
        "stage_receipts": {stage: record(root / stage / "receipt.json", root)
                           for stage in ("cache", "training", "evaluation")},
        "files": inventory(root)}
    write_json(root / "receipt.json", outer)
    return outer


def run_stage(config: dict, root: Path, stage: str) -> dict:
    checked = preflight_head_data(config)
    expected = {key: config[f"{key}_sha256"] for key in INPUT_KEYS}
    root.mkdir(parents=True, exist_ok=True)
    if stage == "preflight":
        receipt = {"schema_version": "vidhoi_head_preflight_v1", "status": "completed",
            "input_hashes": expected, "model_training_run": False, "model_inference_run": False,
            "official_final_split_accessed": False, "final_unseen_accessed": False,
            "sampled_data_audit": checked["audit"],
            "roles": {role: {"num_frames": len(frames),
                             "num_pairs": sum(len(proposal_pairs(frame)) for frame in frames)}
                      for role, frames in (("train", checked["train_frames"]),
                                           ("validation", checked["dev_frames"]))}}
        write_json(root / "preflight.json", receipt)
        return receipt
    if stage == "cache":
        return cache_head_features(config, root / "cache")
    if stage == "train":
        train, dev, texts, allowed, _ = load_caches(root / "cache", expected, config, checked)
        return run_head_study(train, dev, texts, allowed, config, root / "training")
    if stage == "evaluate":
        return evaluate_study(config, root, expected)
    raise ValueError("Unknown study stage")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=["preflight", "cache", "train", "evaluate", "all"],
                        default="all")
    args = parser.parse_args(argv)
    config = resolve_config(args.config)
    for stage in (["preflight", "cache", "train", "evaluate"] if args.stage == "all" else [args.stage]):
        print(f"HEAD_STAGE_STARTED: {stage}", flush=True)
        result = run_stage(config, args.output_dir, stage)
        print(json.dumps({"stage": stage, "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()

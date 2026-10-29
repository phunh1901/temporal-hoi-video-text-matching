"""Positive-only frozen-feature M1 development study with video-disjoint selection.

Nine heads (three views x three seeds) share the same data, text vocabulary,
capacity and optimizer budget. Epoch zero is a genuine selectable candidate.
No unannotated predicate is converted into a supervised negative.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from temporal_hoi.data.vidhoi import ClassMapping, HOIPair, VidHOIFrame
from temporal_hoi.evaluation.hoi import evaluate_frame_hoi
from temporal_hoi.inference.hoi_diagnostics import positive_predicate_ranks
from temporal_hoi.inference.hoi_static import (
    OracleFrame,
    _peak_rss_bytes,
    file_sha256,
    proposal_pairs,
)
from temporal_hoi.models.hoi_head import (
    MODES,
    HOIResidualHead,
    identity_projection,
    positive_anchor_loss,
    view_features,
)

SEEDS = (42, 43, 44)
SELECTION = "dev_map_then_mean_positive_rank_then_earliest"
ARRAY_KEYS = ("object_features", "union_features", "object_class", "positive_mask",
              "frame_index", "subject_track_id", "object_track_id")


def _plain(value: Any) -> Any:
    if isinstance(value, ClassMapping):
        return {"objects": dict(value.objects), "predicates": dict(value.predicates)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _record(path: Path, relative: Path) -> dict:
    return {"path": path.relative_to(relative).as_posix(), "sha256": file_sha256(path),
            "bytes": path.stat().st_size}


def _inventory(directory: Path) -> list[dict]:
    return [_record(path, directory) for path in sorted(directory.rglob("*")) if path.is_file()]


def study_config(config: dict) -> dict:
    source = config.get("study", config)
    defaults = {
        "seeds": list(SEEDS), "modes": list(MODES), "max_epochs": 20, "patience": 5,
        "batch_size": 64, "hidden_dim": 128, "learning_rate": 1e-3,
        "weight_decay": 1e-4, "anchor_lambda": 1.0, "selection": SELECTION,
        "device": "cpu", "threads": 2,
    }
    result = {**defaults, **source}
    if "study" in config:
        for name in ("device", "threads"):
            if name in config:
                result[name] = config[name]
    for name, expected in (("seeds", list(SEEDS)), ("modes", list(MODES)),
                           ("patience", 5), ("batch_size", 64), ("hidden_dim", 128),
                           ("learning_rate", 1e-3), ("weight_decay", 1e-4),
                           ("anchor_lambda", 1.0), ("selection", SELECTION)):
        if result[name] != expected:
            raise ValueError(f"Study requires fixed {name}={expected}")
    if type(result["max_epochs"]) is not int or not 1 <= result["max_epochs"] <= 20:
        raise ValueError("max_epochs must be an integer in [1,20]")
    if result["max_epochs"] != 20 and result.get("test_fixture") is not True:
        raise ValueError("Shortened study requires explicit test_fixture=True")
    if type(result["threads"]) is not int or result["threads"] < 1:
        raise ValueError("threads must be positive")
    if result["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if result["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no fallback")
    return result


def _templates(frame: Any, predicates: int) -> list[HOIPair]:
    if isinstance(frame, OracleFrame):
        return [HOIPair(0, obj.class_id, subject.bbox_xyxy, obj.bbox_xyxy,
                        (0.0,) * predicates, subject_track_id=subject.track_id,
                        object_track_id=obj.track_id)
                for subject, obj in proposal_pairs(frame)]
    if isinstance(frame, VidHOIFrame) and frame.box_source == "oracle":
        return list(frame.predictions)
    raise ValueError("Cache frames must retain OracleFrame or oracle VidHOIFrame templates")


def _cache_hash(cache: Mapping) -> str:
    digest = hashlib.sha256()
    for name in ARRAY_KEYS:
        array = np.asarray(cache[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_cache(cache: Mapping, *, role: str | None = None) -> dict:
    if not isinstance(cache, Mapping) or not set(ARRAY_KEYS).issubset(cache):
        raise ValueError("Feature cache lacks required arrays")
    mapping = cache.get("mapping")
    frames = cache.get("frames")
    if not isinstance(mapping, ClassMapping) or not isinstance(frames, list) or not frames:
        raise ValueError("Cache requires ClassMapping and nonempty canonical frames")
    metadata = cache.get("metadata", {})
    declared_role = metadata.get("role")
    if declared_role not in {"train", "validation", "dev", "research_train", "research_validation"}:
        raise ValueError("Cache must declare a development train/validation role")
    if role == "train" and declared_role not in {"train", "research_train"}:
        raise ValueError("Training cache role is not train")
    if role == "validation" and declared_role not in {"validation", "dev", "research_validation"}:
        raise ValueError("Development cache role is not validation")
    arrays = {name: np.asarray(cache[name]) for name in ARRAY_KEYS}
    objects, unions = arrays["object_features"], arrays["union_features"]
    if (objects.dtype != np.float32 or unions.dtype != np.float32
            or objects.ndim != 2 or objects.shape != unions.shape or not len(objects)
            or not objects.shape[1] or not np.isfinite(objects).all()
            or not np.isfinite(unions).all()
            or (np.linalg.norm(objects, axis=1) <= 0).any()
            or (np.linalg.norm(unions, axis=1) <= 0).any()):
        raise ValueError("Object/union features must be finite nonzero FP32 [N,D] arrays")
    count, dimension = objects.shape
    if (arrays["positive_mask"].dtype != np.bool_
            or arrays["positive_mask"].shape != (count, len(mapping.predicates))):
        raise ValueError("positive_mask must be bool [N,predicates]")
    for name in ("object_class", "frame_index"):
        if arrays[name].dtype.kind not in "iu" or arrays[name].shape != (count,):
            raise ValueError(f"{name} must be an integer [N] vector")
    for name in ("subject_track_id", "object_track_id"):
        if arrays[name].dtype.kind not in "US" or arrays[name].shape != (count,):
            raise ValueError(f"{name} must be a plain string [N] vector")
    if "negative_mask" in cache and np.asarray(cache["negative_mask"]).any():
        raise ValueError("This study does not accept supervised negative labels")
    if len({frame.key for frame in frames}) != len(frames):
        raise ValueError("Duplicate frame identity in cache")
    templates = []
    cursor = 0
    for frame_index, frame in enumerate(frames):
        if isinstance(frame, VidHOIFrame) and frame.split == "official_val":
            raise ValueError("Official final split cannot enter a development study")
        frame_templates = _templates(frame, len(mapping.predicates))
        known = {}
        for pair in frame.ground_truth:
            key = (pair.subject_track_id, pair.object_track_id)
            if key in known:
                raise ValueError("Duplicate canonical GT pair; merge positive predicates explicitly")
            values = np.asarray(pair.predicate_values)
            if len(values) != len(mapping.predicates) or not np.isin(values, [0, 1]).all():
                raise ValueError("Canonical GT must contain binary observed-positive labels")
            known[key] = values == 1
        proposal_ids = {(pair.subject_track_id, pair.object_track_id) for pair in frame_templates}
        if not set(known).issubset(proposal_ids):
            raise ValueError("Canonical positive pair is absent from the proposal inventory")
        for pair in frame_templates:
            if cursor >= count:
                raise ValueError("Cache dropped canonical proposal pairs")
            key = (pair.subject_track_id, pair.object_track_id)
            if (arrays["frame_index"][cursor] != frame_index
                    or str(arrays["subject_track_id"][cursor]) != key[0]
                    or str(arrays["object_track_id"][cursor]) != key[1]
                    or arrays["object_class"][cursor] != pair.object_class):
                raise ValueError("Cache proposal identity/class/order differs from canonical frames")
            expected = known.get(key, np.zeros(len(mapping.predicates), dtype=bool))
            if not np.array_equal(arrays["positive_mask"][cursor], expected):
                raise ValueError("Cache positive labels differ from canonical annotation")
            templates.append(pair)
            cursor += 1
    if cursor != count:
        raise ValueError("Cache has extra rows beyond canonical proposals")
    return {**dict(cache), **arrays, "dimension": dimension, "count": count,
            "templates": templates, "role": declared_role, "fingerprint": _cache_hash(cache)}


def _provenance(cache: Mapping, *, fixture: bool) -> None:
    metadata = cache.get("metadata", {})
    expected = {
        "test_fixture": fixture,
        "level": "synthetic_encoder_fixture" if fixture
        else "vidhoi_head_frozen_clip_feature_cache",
        "source_split": "train", "label_policy": "positive_only_unknown",
        "final_unseen_accessed": False, "official_final_split_accessed": False,
        "pseudo_unseen_in_train": False,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError("Feature cache provenance differs from the declared study scope")
    for name in ("protocol_sha256", "checkpoint_sha256", "mapping_sha256", "config_sha256"):
        value = metadata.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"Feature cache requires a valid {name}")


def _common(text_features: Any, candidate_mask: Any, cache: Mapping) -> tuple[np.ndarray, np.ndarray]:
    texts = np.asarray(text_features)
    allowed = np.asarray(candidate_mask)
    expected = (len(cache["mapping"].objects), len(cache["mapping"].predicates), cache["dimension"])
    if texts.dtype != np.float32 or texts.shape != expected or not np.isfinite(texts).all():
        raise ValueError("text_features must be finite FP32 [objects,predicates,D]")
    if allowed.dtype != np.bool_ or allowed.shape != texts.shape[:2]:
        raise ValueError("candidate_mask must be bool [objects,predicates]")
    norms = np.linalg.norm(texts, axis=-1)
    if (norms[allowed] <= 0).any() or (texts[~allowed] != 0).any():
        raise ValueError("Allowed text rows must be nonzero; held-out text rows must be exactly zero")
    classes = cache["object_class"]
    if (classes < 0).any() or (classes >= len(texts)).any():
        raise ValueError("Unknown object class in feature cache")
    if (cache["positive_mask"] & ~allowed[classes]).any():
        raise ValueError("Final/excluded composition appears as an observed positive")
    return texts, allowed


def _base(cache: Mapping, mode: str, device: str) -> torch.Tensor:
    obj = torch.from_numpy(np.array(cache["object_features"], copy=True)).to(device)
    union = torch.from_numpy(np.array(cache["union_features"], copy=True)).to(device)
    return view_features(obj, union, mode)


def _evaluate(
    head: HOIResidualHead | None, cache: Mapping, texts: np.ndarray, allowed: np.ndarray,
    mode: str, device: str, batch_size: int,
) -> dict:
    if head is not None:
        head.eval()
    base = _base(cache, mode, device)
    score_values = np.zeros((cache["count"], texts.shape[1]), dtype=np.float64)
    with torch.inference_mode():
        for offset in range(0, cache["count"], batch_size):
            current = base[offset:offset + batch_size]
            projected = identity_projection(current) if head is None else head(current)
            classes = cache["object_class"][offset:offset + len(current)]
            for object_class in np.unique(classes):
                rows = np.flatnonzero(classes == object_class)
                predicates = np.flatnonzero(allowed[object_class])
                if not len(predicates):
                    continue
                prototypes = torch.from_numpy(np.array(texts[object_class, predicates],
                                                        copy=True)).to(device)
                prototypes = F.normalize(prototypes, dim=-1)
                selected = projected[torch.as_tensor(rows, device=device)]
                cosine = (selected @ prototypes.T).clamp(-1, 1).float().cpu().numpy()
                score_values[np.ix_(offset + rows, predicates)] = np.exp(cosine.astype(np.float64))
    frames = []
    cursor = 0
    split = "research_train" if cache["role"] in {"train", "research_train"} else "research_validation"
    for frame in cache["frames"]:
        predictions = []
        for template in _templates(frame, texts.shape[1]):
            predictions.append(HOIPair(
                template.subject_class, template.object_class, template.subject_box,
                template.object_box, tuple(float(value) for value in score_values[cursor]),
                template.subject_score, template.object_score,
                template.subject_track_id, template.object_track_id))
            cursor += 1
        frames.append(VidHOIFrame(
            frame.video_id, frame.frame_id, split, "oracle", "xyxy_pixels_original_frame",
            frame.ground_truth, tuple(predictions), frame.timestamp_s))
    metrics = evaluate_frame_hoi(frames)
    ranks = positive_predicate_ranks(frames, cache["mapping"])
    objects_by_pair = {(frame.key, pair.subject_track_id, pair.object_track_id): pair.object_class
                       for frame in frames for pair in frame.predictions}
    for row in ranks["records"]:
        obj = objects_by_pair[(row["frame_key"], row["subject_track_id"], row["object_track_id"])]
        row["num_mapping_predicate_slots"] = texts.shape[1]
        row["num_candidate_predicates"] = int(allowed[obj].sum())
    ranks["excluded_compositions_have_zero_score"] = True
    return {"frames": frames, "metrics": metrics, "known_positive_ranks": ranks,
            "score_values": score_values}


def evaluate_head_state(
    head: HOIResidualHead | None, cache: Mapping, text_features: Any, candidate_mask: Any,
    mode: str, device: str = "cpu", batch_size: int = 64,
) -> dict:
    checked = validate_cache(cache)
    texts, allowed = _common(text_features, candidate_mask, checked)
    if type(batch_size) is not int or batch_size < 1 or mode not in MODES:
        raise ValueError("Require positive evaluation batch_size and a fixed study mode")
    if head is not None:
        head.to(device)
    return _evaluate(head, checked, texts, allowed, mode, device, batch_size)


def selection_key(metrics: Mapping, ranks: Mapping, epoch: int) -> tuple:
    ap = metrics["map"]
    rank = ranks["mean_rank"]
    return (float("-inf") if ap is None else float(ap),
            float("-inf") if rank is None else -float(rank), -epoch)


def _state(model: HOIResidualHead) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()}


def _delta(initial: Mapping, current: Mapping) -> float:
    return math.sqrt(sum(float((current[key].double() - value.double()).square().sum())
                         for key, value in initial.items()))


def _save_evaluation(directory: Path, result: dict) -> None:
    _write(directory / "predictions.json", [asdict(frame) for frame in result["frames"]])
    _write(directory / "metrics.json", result["metrics"])
    _write(directory / "known_positive_ranks.json", result["known_positive_ranks"])


def _run_one(
    train: Mapping, dev: Mapping, texts: np.ndarray, allowed: np.ndarray, config: Mapping,
    directory: Path, mode: str, seed: int,
) -> dict:
    directory.mkdir(parents=True)
    started = time.perf_counter()
    device = config["device"]
    torch.manual_seed(seed)
    head = HOIResidualHead(train["dimension"], config["hidden_dim"]).to(device)
    initial = _state(head)
    base = _base(train, mode, device)
    with torch.inference_mode():
        initial_embeddings = head(base).cpu().clone()
    positive = torch.from_numpy(np.array(train["positive_mask"], copy=True)).to(device)
    classes = torch.from_numpy(np.array(train["object_class"], copy=True)).long().to(device)
    text_tensor = torch.from_numpy(np.array(texts, copy=True)).to(device)
    mask_tensor = torch.from_numpy(np.array(allowed, copy=True)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    epoch0 = _evaluate(head, dev, texts, allowed, mode, device, config["batch_size"])
    best_key = selection_key(epoch0["metrics"], epoch0["known_positive_ranks"], 0)
    selected_state, selected_epoch = initial, 0
    history = [{"epoch": 0, "optimizer_steps": 0, "train_loss": None, "positive_loss": None,
                "anchor_loss": None, "dev_map": epoch0["metrics"]["map"],
                "dev_known_positive_mean_rank": epoch0["known_positive_ranks"]["mean_rank"]}]
    steps, bad_epochs, max_gradient = 0, 0, 0.0
    for epoch in range(1, config["max_epochs"] + 1):
        head.train()
        order = torch.randperm(train["count"], generator=generator).tolist()
        positive_sum, anchor_sum, labeled_count = 0.0, 0.0, 0
        for offset in range(0, len(order), config["batch_size"]):
            indices = torch.as_tensor(order[offset:offset + config["batch_size"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            projected = head(base[indices])
            losses = positive_anchor_loss(
                projected, base[indices], classes[indices], positive[indices],
                text_tensor, mask_tensor, anchor_weight=config["anchor_lambda"])
            if not torch.isfinite(losses["loss"]):
                raise ValueError("Training loss is nonfinite")
            losses["loss"].backward()
            gradients = [parameter.grad for parameter in head.parameters()]
            if any(value is None or not torch.isfinite(value).all() for value in gradients):
                raise ValueError("Head gradient is missing or nonfinite")
            gradient = math.sqrt(sum(float(value.detach().double().square().sum())
                                     for value in gradients))
            max_gradient = max(max_gradient, gradient)
            optimizer.step()
            steps += 1
            labeled_count += losses["labeled_pairs"]
            positive_sum += float(losses["positive_loss"].detach()) * losses["labeled_pairs"]
            anchor_sum += float(losses["anchor_loss"].detach()) * losses["all_pairs"]
        evaluated = _evaluate(head, dev, texts, allowed, mode, device, config["batch_size"])
        rank = evaluated["known_positive_ranks"]["mean_rank"]
        positive_mean = positive_sum / labeled_count if labeled_count else 0.0
        anchor_mean = anchor_sum / train["count"]
        history.append({
            "epoch": epoch, "optimizer_steps": steps,
            "train_loss": positive_mean + config["anchor_lambda"] * anchor_mean,
            "positive_loss": positive_mean, "anchor_loss": anchor_mean,
            "dev_map": evaluated["metrics"]["map"], "dev_known_positive_mean_rank": rank})
        key = selection_key(evaluated["metrics"], evaluated["known_positive_ranks"], epoch)
        if key > best_key:
            best_key, selected_state, selected_epoch = key, _state(head), epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"HOI_HEAD_EPOCH: mode={mode} seed={seed} epoch={epoch} steps={steps} "
              f"dev_map={evaluated['metrics']['map']} selected_epoch={selected_epoch}", flush=True)
        if bad_epochs >= config["patience"]:
            break
    last = _state(head)
    with torch.inference_mode():
        last_embeddings = head(base).cpu()
    embedding_delta = float((last_embeddings.double() - initial_embeddings.double()).norm())
    parameter_delta = _delta(initial, last)
    if steps < 1 or max_gradient <= 0 or parameter_delta <= 0 or embedding_delta <= 0:
        raise ValueError("No verified gradient/parameter/embedding update from head training")
    head.load_state_dict(selected_state, strict=True)
    selected = _evaluate(head, dev, texts, allowed, mode, device, config["batch_size"])
    _save_evaluation(directory, selected)
    checkpoints = {}
    for name, state in (("initial", initial), ("selected", selected_state), ("last", last)):
        path = directory / f"{name}_head.safetensors"
        save_file(state, str(path))
        checkpoints[name] = _record(path, directory)
    _write(directory / "history.json", history)
    with (directory / "train_log.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "mode": mode, "seed": seed, "selected_epoch": selected_epoch, "last_epoch": history[-1]["epoch"],
        "optimizer_steps": steps, "max_gradient_norm": max_gradient,
        "nonzero_gradient_observed": max_gradient > 0,
        "last_parameter_delta_l2": parameter_delta,
        "selected_parameter_delta_l2": _delta(initial, selected_state),
        "last_embedding_delta_l2": embedding_delta,
        "train_pairs": train["count"], "dev_pairs": dev["count"],
        "train_labeled_pairs": int(train["positive_mask"].any(axis=1).sum()),
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "selected_metrics": selected["metrics"],
        "selected_known_positive_mean_rank": selected["known_positive_ranks"]["mean_rank"],
        "selection": SELECTION, "epoch_zero_was_selection_candidate": True,
        "model_training_run": True, "backbone_training_run": False,
        "backbone_features_detached": True, "text_features_detached": True,
        "label_policy": "observed_positive_only_other_entries_unknown",
        "loss": "mean_positive_cosine_per_labeled_pair_plus_all_pair_feature_anchor",
        "anchor_lambda": config["anchor_lambda"], "checkpoints": checkpoints,
        "train_cache_fingerprint": train["fingerprint"], "dev_cache_fingerprint": dev["fingerprint"],
        "train_cache_metadata": train.get("metadata", {}), "dev_cache_metadata": dev.get("metadata", {}),
        "elapsed_seconds": time.perf_counter() - started,
        "full_benchmark_reproduction_claim": False, "official_final_split_accessed": False,
    }
    _write(directory / "metadata.json", metadata)
    receipt = {"schema_version": "vidhoi_head_run_result_v1", "status": "completed",
               **metadata, "files": _inventory(directory)}
    _write(directory / "receipt.json", receipt)
    return receipt


def _aggregate(runs: list[dict]) -> dict:
    result = {}
    for mode in MODES:
        chosen = [run for run in runs if run["mode"] == mode]
        if [run["seed"] for run in chosen] != list(SEEDS):
            raise ValueError("All three declared seeds are required; no best-seed selection")
        values = [run["selected_metrics"]["map"] for run in chosen]
        ranks = [run["selected_known_positive_mean_rank"] for run in chosen]
        result[mode] = {
            "seeds": list(SEEDS), "n": len(chosen), "metric_scale": "fraction",
            "dev_map_mean": float(np.mean(values)), "dev_map_std": float(np.std(values, ddof=1)),
            "mean_positive_rank_mean": float(np.mean(ranks)),
            "mean_positive_rank_std": float(np.std(ranks, ddof=1)),
            "selected_epochs": [run["selected_epoch"] for run in chosen],
            "optimizer_steps": [run["optimizer_steps"] for run in chosen],
            "selected_best_seed": None,
        }
    return result


def run_head_study(
    train_cache: Mapping, dev_cache: Mapping, text_features: Any, candidate_mask: Any,
    config: dict, output_dir: str | Path,
) -> dict:
    settings = study_config(config)
    train = validate_cache(train_cache, role="train")
    dev = validate_cache(dev_cache, role="validation")
    _provenance(train, fixture=settings.get("test_fixture") is True)
    _provenance(dev, fixture=settings.get("test_fixture") is True)
    if (train["mapping"] != dev["mapping"] or train["dimension"] != dev["dimension"]):
        raise ValueError("Train/development mapping or feature dimensions differ")
    train_videos = {frame.video_id for frame in train["frames"]}
    dev_videos = {frame.video_id for frame in dev["frames"]}
    if train_videos & dev_videos:
        raise ValueError("Train/development video leakage")
    for key in ("protocol_sha256", "checkpoint_sha256", "mapping_sha256", "config_sha256"):
        left, right = train.get("metadata", {}).get(key), dev.get("metadata", {}).get(key)
        if left != right:
            raise ValueError(f"Train/development {key} mismatch")
    if not train["positive_mask"].any() or not dev["positive_mask"].any():
        raise ValueError("Train and development each require observed-positive support")
    texts, allowed = _common(text_features, candidate_mask, train)
    _common(text_features, candidate_mask, dev)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(settings["threads"])
    _write(directory / "study_config.json", {"schema_version": "vidhoi_head_study_config_v1",
                                            "study": settings, "provided_config": config})
    frozen = {}
    for mode in MODES:
        path = directory / "frozen" / mode
        path.mkdir(parents=True)
        result = _evaluate(None, dev, texts, allowed, mode, settings["device"], settings["batch_size"])
        if result["metrics"]["map"] is None or result["known_positive_ranks"]["mean_rank"] is None:
            raise ValueError("Development has no GT-supported selection metric")
        _save_evaluation(path, result)
        frozen[mode] = {
            "mode": mode, "directory": f"frozen/{mode}", "metrics": result["metrics"],
            "known_positive_mean_rank": result["known_positive_ranks"]["mean_rank"],
            "num_frames": len(dev["frames"]), "num_proposal_pairs": dev["count"],
        }
        _write(path / "receipt.json", {
            "schema_version": "vidhoi_frozen_head_baseline_v1", "status": "completed",
            "model_training_run": False, "full_benchmark_reproduction_claim": False,
            "official_final_split_accessed": False, **frozen[mode], "files": _inventory(path)})
    runs = []
    for mode in MODES:
        for seed in SEEDS:
            relative = Path("runs") / mode / f"seed_{seed}"
            result = _run_one(train, dev, texts, allowed, settings, directory / relative, mode, seed)
            runs.append({key: value for key, value in result.items() if key != "files"}
                        | {"directory": relative.as_posix()})
    if _cache_hash(train_cache) != train["fingerprint"] or _cache_hash(dev_cache) != dev["fingerprint"]:
        raise ValueError("Training or selection mutated the input feature cache")
    aggregate = _aggregate(runs)
    _write(directory / "aggregate.json", aggregate)
    receipt = {
        "schema_version": "vidhoi_head_study_result_v1", "status": "completed",
        "level": "synthetic_feature_head_study" if settings.get("test_fixture")
        else "oracle_frozen_feature_development_head_study",
        "training_run": True, "model_training_run": True, "backbone_training_run": False,
        "backbone_inference_run": False, "full_benchmark_reproduction_claim": False,
        "official_final_split_accessed": False, "final_unseen_accessed": False,
        "zero_shot_benchmark_claim": False, "benchmark_comparison_eligible": False,
        "seeds": list(SEEDS), "modes": list(MODES), "runs": runs,
        "frozen_baselines": frozen, "aggregate": aggregate, "selected_best_seed": None,
        "train_videos": sorted(train_videos), "dev_videos": sorted(dev_videos),
        "train_pairs": train["count"], "dev_pairs": dev["count"],
        "train_cache_fingerprint": train["fingerprint"], "dev_cache_fingerprint": dev["fingerprint"],
        "train_cache_metadata": train.get("metadata", {}), "dev_cache_metadata": dev.get("metadata", {}),
        "candidate_mask_sha256": hashlib.sha256(allowed.tobytes()).hexdigest(),
        "text_features_sha256": hashlib.sha256(texts.tobytes()).hexdigest(),
        "heldout_text_rows_zero": True, "heldout_scores_zero": True,
        "cache_inputs_unchanged": True, "video_disjoint": True,
        "label_policy": "observed_positive_only_other_entries_unknown",
        "score_transform": "exp(normalized_head_text_cosine); excluded_compositions_score_zero",
        "score_is_probability": False, "parameter_capacity_equal_across_modes": True,
        "selection": SELECTION, "epoch_zero_was_selection_candidate": True,
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(), "torch_version": torch.__version__,
        "files": _inventory(directory),
    }
    _write(directory / "receipt.json", receipt)
    return receipt

"""Train frozen-feature heads and select checkpoints on locked validation features."""

import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from temporal_hoi.data.msrvtt import file_sha256, semantic_hash
from temporal_hoi.evaluation.metrics import compute_retrieval_metrics
from temporal_hoi.models.alignment import build_alignment_head
from temporal_hoi.training.features import validate_training_splits
from temporal_hoi.training.losses import bidirectional_multi_positive_loss
from temporal_hoi.training.resources import enforce_resources, ensure_allocation


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def validate_config(config):
    if (config.get("schema_version") != "frozen_alignment_config_v1"
            or config["train_split"] != "train" or config["val_split"] != "validation"
            or config["development_split"] != "sourcegroup_v2"
            or config["backbone_frozen"] is not True or config["head"] not in {"linear", "mlp"}
            or config["optimizer"] != "AdamW" or config["device"] != "cpu"
            or config["dtype"] != "float32"
            or config["embedding_dim"] != "infer_from_encoder"
            or config["selection_metric"] != "text_to_video.R@1"):
        raise ValueError("Require frozen CPU heads and sourcegroup_v2 validation R@1 selection")
    if config["head"] == "mlp" and (type(config.get("hidden_dim")) is not int
                                    or config["hidden_dim"] < 1):
        raise ValueError("MLP hidden_dim must be a positive integer")
    for key in ("frames", "decode_batch_size", "feature_batch_videos", "text_batch_size",
                "shard_videos", "validation_query_chunk", "max_epochs", "patience", "threads",
                "max_rss_bytes"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (config["feature_batch_videos"] < 2 or config["decode_batch_size"] != 1
            or config["shard_videos"] > 500):
        raise ValueError("At least 2 distinct feature videos, decode batch1, <=500-video shards")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer")
    for key in ("lr", "temperature", "max_train_hours"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if config["crop_mode"] not in {"global", "union"}:
        raise ValueError("crop_mode must be global or union")


def distinct_video_batches(count, batch_size, generator):
    """Merge a singleton tail into the previous batch; never fake negatives."""
    if count < 2 or batch_size < 2:
        raise ValueError("Contrastive batches require at least two distinct videos")
    indices = torch.randperm(count, generator=generator).tolist()
    batches = [indices[start:start + batch_size] for start in range(0, count, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def _metrics_chunked(scores, positives, chunk, config, started):
    ranks = []
    for start in range(0, len(scores), chunk):
        enforce_resources(config, started)
        result = compute_retrieval_metrics(scores[start:start + chunk],
                                           positives[start:start + chunk])
        ranks.extend(result["first_positive_ranks"])
    valid = np.asarray([r for r in ranks if r is not None], dtype=np.float64)
    return {
        **{f"R@{k}": float(np.mean(valid <= k)) if len(valid) else None for k in (1, 5, 10)},
        "MedR": float(np.median(valid)) if len(valid) else None,
        "MRR": float(np.mean(1 / valid)) if len(valid) else None,
        "num_queries": len(scores), "num_candidates": scores.shape[1],
        "num_evaluated_queries": len(valid), "num_excluded_queries": len(ranks) - len(valid),
        "first_positive_ranks": ranks, "tie_breaking": "candidate_order",
    }


@torch.no_grad()
def evaluate_alignment(model, features, config, *, started=None):
    """Keep only CPU scores; project and rank in bounded query chunks."""
    model.eval()
    started = time.monotonic() if started is None else started
    chunk = config["validation_query_chunk"]
    videos, texts = len(features.pool.video_ids), len(features.pool.text_ids)
    # Chunking bounds projection/ranking working buffers, not CPU score storage.
    # Include both projected matrices and the full CPU score matrix, then the
    # largest temporary phase: encode, chunk matmul, float64 ranking, or loss.
    resident = 4 * (model.output_dim * (videos + texts) + videos * texts)
    candidates = max(videos, texts)
    working = max(8 * 4 * chunk * model.output_dim,
                  4 * chunk * videos,
                  8 * chunk * candidates + 24 * candidates,
                  32 * candidates)
    ensure_allocation(config, started, resident + working, "validation")
    vectors = {}
    for key, raw, encode in [("video", features.video_embeddings, model.encode_video),
                             ("text", features.text_embeddings, model.encode_text)]:
        projected = np.empty((len(raw), model.output_dim), dtype=np.float32)
        for start in range(0, len(raw), chunk):
            enforce_resources(config, started)
            projected[start:start + chunk] = encode(torch.from_numpy(raw[start:start + chunk])).numpy()
        vectors[key] = projected
    scores = np.empty((len(features.pool.text_ids), len(features.pool.video_ids)), dtype=np.float32)
    for start in range(0, len(scores), chunk):
        enforce_resources(config, started)
        scores[start:start + chunk] = vectors["text"][start:start + chunk] @ vectors["video"].T
    text_pos, video_pos = features.pool.positive_indices()
    if features.relation_labels is not None:
        # Unknown relevance is excluded from candidate ranking. Each row with
        # known positives and any candidates is evaluated over its known pool.
        # Such diagnostic results must never be presented as the locked full
        # MSR-VTT benchmark; production caches use identity-complete relevance.
        raise ValueError("Training validation requires complete locked retrieval relevance")
    metrics = {
        "text_to_video": _metrics_chunked(scores, text_pos, chunk, config, started),
        "video_to_text": _metrics_chunked(scores.T, video_pos, chunk, config, started),
        "pool_hashes": features.pool.hashes(), "matrix_axes": "text,video",
        "scope": features.provenance["scope"], "recall_scale": "fraction",
        "selection_split": "validation",
    }
    losses = []
    for matrix, positives in ((scores, text_pos), (scores.T, video_pos)):
        direction = []
        for row, positive in zip(matrix, positives, strict=True):
            enforce_resources(config, started)
            logits = row.astype(np.float64) / config["temperature"]
            maximum, pos_maximum = logits.max(), logits[positive].max()
            denominator = maximum + np.log(np.exp(logits - maximum).sum())
            numerator = pos_maximum + np.log(np.exp(logits[positive] - pos_maximum).sum())
            direction.append(float(denominator - numerator))
        losses.append(float(np.mean(direction)))
    metrics["loss"] = sum(losses) / 2
    return metrics, scores


def _state_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_alignment(train, validation, config, output_dir, *, resume=None, stop_after_epoch=None):
    """AdamW and patience on validation R@1; resumable optimizer/config/RNG state."""
    validate_config(config)
    validate_training_splits(train, validation)
    output = Path(output_dir)
    if resume is None and (output / "last.pt").exists():
        raise ValueError("Existing training output requires --resume or a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    model = build_alignment_head(config, train.video_embeddings.shape[1],
                                 train.text_embeddings.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    identities = {
        "train": train.provenance, "validation": validation.provenance,
        "split_hashes": {"train": train.pool.hashes(), "validation": validation.pool.hashes()},
    }
    config_hash, input_hash = semantic_hash(config), semantic_hash(identities)
    initial = _state_cpu(model)
    history, best_score, stale, start_epoch, step_count = [], -1.0, 0, 1, 0
    elapsed_before = 0.0
    if resume is not None:
        stored = torch.load(resume, map_location="cpu", weights_only=True)
        if (stored.get("schema_version") != "alignment_checkpoint_v1"
                or stored["config_sha256"] != config_hash or stored["input_sha256"] != input_hash):
            raise ValueError("Resume checkpoint config/cache/split provenance changed")
        if Path(resume).resolve() != (output / "last.pt").resolve():
            raise ValueError("Resume must continue its original output directory and last.pt")
        model.load_state_dict(stored["model_state_dict"], strict=True)
        optimizer.load_state_dict(stored["optimizer_state_dict"])
        generator.set_state(stored["batch_rng_state"])
        torch.set_rng_state(stored["torch_rng_state"])
        initial, history = stored["initial_state_dict"], stored["history"]
        best_score, stale = stored["best_score"], stored["patience_counter"]
        start_epoch, step_count = stored["epoch"] + 1, stored["step_count"]
        elapsed_before = stored["elapsed_seconds"]
        if not (output / "best.pt").is_file():
            raise ValueError("Resumed run is missing its previously selected best checkpoint")
    write_json(output / "provenance.json", {
        "execution_level": train.provenance["artifact_kind"], "seed": config["seed"],
        "config": config, "config_sha256": config_hash, "inputs": identities,
        "input_sha256": input_hash, "backbone_frozen": True,
        "selection_split": "validation", "selection_metric": "text_to_video.R@1",
        "test_accessed": False, "python": platform.python_version(), "torch": str(torch.__version__),
    })
    caption_owners = np.asarray([q["video_id"] for q in train.pool.queries])
    started = time.monotonic() - elapsed_before
    peak_observed, stopped = enforce_resources(config, started), "max_epochs"
    last_gradient_norm = None
    for epoch in range(start_epoch, config["max_epochs"] + 1):
        if stale >= config["patience"]:
            stopped = "early_stopping"
            break
        model.train()
        epoch_started, video_total, text_total, nv, nt = time.monotonic(), 0.0, 0.0, 0, 0
        gradients, loss_steps = [], []
        for video_indices in distinct_video_batches(len(train.pool.video_ids),
                                                     config["feature_batch_videos"], generator):
            peak_observed = max(peak_observed, enforce_resources(config, started))
            chosen_ids = np.asarray(train.pool.video_ids)[video_indices]
            text_indices = np.flatnonzero(np.isin(caption_owners, chosen_ids))
            optimizer.zero_grad(set_to_none=True)
            cosine = model(torch.from_numpy(train.video_embeddings[video_indices]),
                           torch.from_numpy(train.text_embeddings[text_indices]))
            labels = torch.from_numpy(train.relations(video_indices, text_indices))
            loss, stats = bidirectional_multi_positive_loss(cosine, labels,
                                                           temperature=config["temperature"])
            loss.backward()
            grads = [p.grad for p in model.parameters()]
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise ValueError("Head gradients must be present and finite")
            last_gradient_norm = float(sum(g.square().sum() for g in grads).sqrt())
            gradients.append(last_gradient_norm)
            loss_steps.append(float(loss.detach()))
            optimizer.step()
            step_count += 1
            video_total += stats["video_to_text_loss"] * stats["video_anchors"]
            text_total += stats["text_to_video_loss"] * stats["text_anchors"]
            nv += stats["video_anchors"]
            nt += stats["text_anchors"]
        metrics, scores = evaluate_alignment(model, validation, config, started=started)
        peak_observed = max(peak_observed, enforce_resources(config, started))
        score = metrics["text_to_video"]["R@1"]
        row = {
            "epoch": epoch, "train_loss": (video_total / nv + text_total / nt) / 2,
            "val_loss": metrics["loss"], "val_t2v_r1": score,
            "val_v2t_r1": metrics["video_to_text"]["R@1"],
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": time.monotonic() - epoch_started, "selection_split": "validation",
            "head_gradient_norm": max(gradients), "min_step_loss": min(loss_steps),
            "max_step_loss": max(loss_steps), "step_count": step_count,
        }
        history.append(row)
        improved = score > best_score
        best_score, stale = (score, 0) if improved else (best_score, stale + 1)
        state = {
            "schema_version": "alignment_checkpoint_v1", "model_state_dict": _state_cpu(model),
            "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "seed": config["seed"],
            "config": config, "config_sha256": config_hash, "input_sha256": input_hash,
            "encoder_sha256": train.provenance["encoder"]["checkpoint_sha256"],
            "split_hashes": identities["split_hashes"], "input_provenance": identities,
            "batch_rng_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(),
            "initial_state_dict": initial, "history": history, "step_count": step_count,
            "best_score": best_score, "patience_counter": stale,
            "elapsed_seconds": time.monotonic() - started, "selection_split": "validation",
            "selection_metric": config["selection_metric"],
            "execution_level": train.provenance["artifact_kind"],
        }
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
            np.savez_compressed(output / "val_scores.npz", scores=scores,
                                video_ids=np.asarray(validation.pool.video_ids),
                                text_ids=np.asarray(validation.pool.text_ids))
            write_json(output / "val_metrics.json", {**metrics, "selected_epoch": epoch,
                       "execution_level": train.provenance["artifact_kind"]})
        with (output / "train_log.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            stopped = "explicit_probe_pause"
            break
    if not history:
        raise ValueError("No optimizer steps completed")
    final = _state_cpu(model)
    update_norm = float(sum((final[name] - initial[name]).square().sum() for name in final).sqrt())
    result = {
        "status": "completed", "execution_level": train.provenance["artifact_kind"],
        "stop_reason": stopped, "epochs_completed": len(history), "step_count": step_count,
        "head_update_norm": update_norm, "last_head_gradient_norm": last_gradient_norm,
        "all_losses_finite": all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"])
                                 for h in history),
        "observed_rss_bytes": peak_observed, "best_validation_text_to_video_R@1": best_score,
        "best_checkpoint_sha256": file_sha256(output / "best.pt"),
        "last_checkpoint_sha256": file_sha256(output / "last.pt"),
        "config_sha256": config_hash, "input_sha256": input_hash,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": time.monotonic() - started,
        "benchmark_training_run": train.provenance["dataset"] == "MSR-VTT",
        "test_accessed": False, "selection_split": "validation", "history": history,
    }
    write_json(output / "run_summary.json", result)
    return result

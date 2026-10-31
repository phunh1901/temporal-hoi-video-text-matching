"""Notebook entry points. No filesystem writes outside configured WORK_DIR."""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from data_pipeline import (
    atomic_json,
    atomic_torch,
    build_cache,
    digest,
    evaluation_records,
    import_file,
    load_encoder,
    make_development_split,
    resolve_file,
    sha256_file,
    source_video_urls,
    training_records,
    video_index,
)
from retrieval_core import (
    aligned_original_metrics,
    compare_runs,
    direct_meanp_reference,
    evaluate_bidirectional,
    score_matrix,
    train_attention_head,
    validate_cache,
)


def download_checkpoint(config):
    explicit = config["CHECKPOINT_PATH"]
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"CHECKPOINT_PATH không tồn tại: {p}")
        return p
    # Never pick one of several epoch files silently.
    attached = sorted(Path(config["INPUT_ROOT"]).rglob("pytorch_model.bin.*"))
    attached = [p for p in attached if p.is_file() and not p.name.endswith(".tmp")]
    if len(attached) == 1:
        return attached[0]
    if len(attached) > 1:
        raise ValueError(
            "Có nhiều checkpoint. Đặt CHECKPOINT_PATH chính xác:\n" + "\n".join(map(str, attached))
        )
    if not config["DOWNLOAD_CHECKPOINT"]:
        raise FileNotFoundError(
            "Thêm checkpoint vào Input và đặt CHECKPOINT_PATH, "
            "hoặc bật Internet và DOWNLOAD_CHECKPOINT."
        )
    path = Path(config["WORK_DIR"]) / "checkpoint" / "jaewooo_meanP.bin"
    provenance_path = path.with_suffix(".json")
    source = {
        "repo": "jaewooo/test",
        "revision": config["HF_REVISION"],
        "filename": config["HF_FILENAME"],
    }
    if path.is_file() and provenance_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous.get("source") == source and previous.get("sha256") == sha256_file(path):
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/jaewooo/test/resolve/{source['revision']}/{source['filename']}"
    temporary = path.with_suffix(".download")
    print("Tải một checkpoint đã khóa phiên bản; không tải toàn bộ repository...", flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "msrvtt-kaggle-benchmark/1"})
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
        if temporary.stat().st_size < 10_000_000:
            raise ValueError("Tệp tải về quá nhỏ để là checkpoint CLIP4Clip đầy đủ.")
        os.replace(temporary, path)
        atomic_json(provenance_path, {"source": source, "sha256": sha256_file(path)})
    except Exception as exc:
        raise RuntimeError(
            "Không tải được checkpoint. Bật Internet Kaggle, hoặc thêm checkpoint "
            "vào Input và đặt CHECKPOINT_PATH. Tệp tải dở chưa được dùng."
        ) from exc
    return path


def preflight(config):
    work = Path(config["WORK_DIR"])
    work.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["WORK_DIR"] = str(work)
    input_hashes = {}
    torch.set_num_threads(2)
    import cv2

    cv2.setNumThreads(1)
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["SEED"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not config["ALLOW_CPU_DEBUG"]:
        raise RuntimeError(
            "Chưa có GPU. Kaggle Settings → Accelerator → GPU; "
            "CPU chỉ dùng smoke test khi ALLOW_CPU_DEBUG=True."
        )
    test_csv = resolve_file(config["TEST_CSV"], "MSRVTT_JSFUSION_test.csv", config["INPUT_ROOT"])
    test_records = evaluation_records(test_csv)
    input_hashes["test_csv"] = sha256_file(test_csv)
    if (
        len(test_records) != config["EXPECTED_TEST_QUERIES"]
        or len({r["video_id"] for r in test_records}) != config["EXPECTED_TEST_VIDEOS"]
    ):
        raise ValueError(
            f"Sai candidate pool: {len(test_records)} caption / "
            f"{len({r['video_id'] for r in test_records})} video. "
            "Mặc định phải là 1000 caption và 1000 video của 1k-A."
        )
    train_records, val_records, development_status = None, None, "not_training"
    reserved_records = None
    policy = config.get("SELECTION_POLICY", "fixed_epochs")
    if policy not in {"fixed_epochs", "independent_validation"}:
        raise ValueError("SELECTION_POLICY must be fixed_epochs or independent_validation")
    if config["RUN_TRAIN"]:
        train_csv = resolve_file(config["TRAIN_CSV"], "MSRVTT_train.9k.csv", config["INPUT_ROOT"])
        captions_json = resolve_file(
            config["CAPTIONS_JSON"], "MSRVTT_data.json", config["INPUT_ROOT"]
        )
        records = training_records(train_csv, captions_json)
        input_hashes.update(
            train_csv=sha256_file(train_csv), captions_json=sha256_file(captions_json)
        )
        if config.get("VAL_CSV"):
            input_hashes["val_csv"] = sha256_file(config["VAL_CSV"])
        expected_train = config.get("EXPECTED_TRAIN_VIDEOS", 9000)
        if len({r["video_id"] for r in records}) != expected_train:
            raise ValueError(f"TRAIN_CSV must contain {expected_train} unique videos")
        if policy == "independent_validation":
            if not config.get("VAL_CSV") or config.get("VAL_SEEN_BY_ENCODER") is not False:
                raise ValueError(
                    "Independent validation needs VAL_CSV and VAL_SEEN_BY_ENCODER=False"
                )
            if not str(config.get("VAL_PROVENANCE", "")).strip():
                raise ValueError("Independent validation needs explicit VAL_PROVENANCE")
        elif config.get("VAL_CSV"):
            raise ValueError("fixed_epochs does not use VAL_CSV; leave it empty")
        train_records, reserved_records, development_status = make_development_split(
            records,
            [r["video_id"] for r in test_records],
            Path(config["VAL_CSV"]) if config.get("VAL_CSV") else None,
            config.get("DEV_VAL_SIZE", 900),
            config["SEED"],
            selection_policy=policy,
            source_urls=source_video_urls(captions_json),
        )
        if policy == "independent_validation":
            val_records = reserved_records
    all_records = test_records + (train_records or []) + (val_records or [])
    root = config["VIDEO_ROOT"] or config["INPUT_ROOT"]
    index = video_index(root, {r["video_id"] for r in all_records})
    reference_required = bool(config.get("REQUIRE_EXTERNAL_REFERENCE", True))
    reference_path = resolve_file(
        config.get("BASELINE_REFERENCE_NPZ", ""),
        "baseline_reference.npz",
        config["INPUT_ROOT"],
        required=reference_required,
    )
    if reference_path is not None:
        expected_reference = config.get("BASELINE_REFERENCE_SHA256", "")
        if expected_reference and sha256_file(reference_path) != expected_reference:
            raise ValueError("Baseline reference SHA256 differs; attach the original reference")
        config["BASELINE_REFERENCE_NPZ"] = str(reference_path)
    checkpoint = download_checkpoint(config)
    expected_sha = config.get("CHECKPOINT_SHA256", "")
    actual_sha = sha256_file(checkpoint)
    if expected_sha and actual_sha.lower() != expected_sha.lower():
        raise ValueError("SHA256 checkpoint không khớp. Không tự thay thế trọng số.")
    free_gib = shutil.disk_usage(work).free / 2**30
    if free_gib < config["MIN_FREE_DISK_GIB"]:
        raise RuntimeError(
            f"Chỉ còn {free_gib:.1f} GiB. Cần ít nhất {config['MIN_FREE_DISK_GIB']} GiB."
        )
    preregistration = {
        "config": config,
        "checkpoint_sha256": actual_sha,
        "input_file_sha256": input_hashes,
        "test_records_sha256": digest(test_records),
        "train_records_sha256": digest(train_records),
        "val_records_sha256": digest(val_records),
        "reserved_dev_records_sha256": digest(reserved_records),
        "selection_policy": policy,
        "reserved_dev_used_for_selection": policy == "independent_validation",
        "development_status": development_status,
        "primary_comparison": "B2-B0 text_to_video R@1",
        "tie_rule": "descending FP32 score; stable input candidate order on exact ties",
        "original_evaluator": "separate pinned CLIP4Clip compute_metrics",
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU_DEBUG",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    registry_path = work / "preregistration.json"
    # Environment may change across Kaggle resumes. Semantic experiment inputs may not.
    path_keys = {
        "INPUT_ROOT",
        "WORK_DIR",
        "VIDEO_ROOT",
        "TEST_CSV",
        "TRAIN_CSV",
        "CAPTIONS_JSON",
        "VAL_CSV",
        "CHECKPOINT_PATH",
        "BASELINE_REFERENCE_NPZ",
    }
    identity_config = {k: v for k, v in config.items() if k not in path_keys | {"RESUME"}}
    identity = digest(
        {
            "config": identity_config,
            **{
                k: preregistration[k]
                for k in (
                    "checkpoint_sha256",
                    "input_file_sha256",
                    "test_records_sha256",
                    "train_records_sha256",
                    "val_records_sha256",
                    "reserved_dev_records_sha256",
                )
            },
        }
    )
    lock = work / "experiment_identity.txt"
    if lock.is_file() and lock.read_text().strip() != identity:
        # The epoch schedule is immutable, including after benchmark access.
        raise ValueError("RUN_NAME đã có cấu hình khác. Chọn RUN_NAME mới để giữ nguyên đối chứng.")
    lock.write_text(identity, encoding="utf-8")
    atomic_json(registry_path, preregistration)
    print(f"Preflight OK: {len(test_records)} test captions, {len(index)} videos; {device}.")
    print(f"Checkpoint SHA256: {actual_sha}")
    print(f"Training: {config['RUN_TRAIN']}; validation status: {development_status}")
    return {
        "config": config,
        "device": device,
        "checkpoint": checkpoint,
        "test_records": test_records,
        "train_records": train_records,
        "val_records": val_records,
        "index": index,
        "development_status": development_status,
        "selection_policy": policy,
    }


def load_runtime(state, vendor_dir):
    cfg = state["config"]
    model, tokenizer, info = load_encoder(
        state["checkpoint"], vendor_dir, state["device"], cfg["PRECISION"]
    )
    # Warm up with a real correctly sized batch before decoding thousands of videos.
    with torch.inference_mode():
        dummy = torch.zeros((1, 3, info["resolution"], info["resolution"]), device=state["device"])
        image = model.encode_image(dummy).float()
        from data_pipeline import tokenize_records

        token = tokenize_records(["a person is walking"], tokenizer, cfg["MAX_WORDS"]).to(
            state["device"]
        )
        text = model.encode_text(token).float()
        if not torch.isfinite(image).all() or not torch.isfinite(text).all():
            raise ValueError("Encoder warm-up tạo NaN/Inf.")
    state.update(model=model, tokenizer=tokenizer, model_info=info, vendor_dir=str(vendor_dir))
    return state


def cache_split(state, split):
    cfg = state["config"]
    records = state[f"{split}_records"]
    if records is None:
        raise ValueError(f"Không có dữ liệu split {split}.")
    cache = build_cache(
        records,
        state["index"],
        state["model"],
        state["tokenizer"],
        state["model_info"],
        Path(cfg["WORK_DIR"]) / "cache",
        max_frames=cfg["MAX_FRAMES"],
        max_words=cfg["MAX_WORDS"],
        device=state["device"],
        encoder_batch=cfg["ENCODER_BATCH"],
        text_batch=cfg["TEXT_BATCH"],
    )
    cache["metadata"]["split"] = split
    if split == "val":
        cache["metadata"]["development_status"] = state["development_status"]
        cache["metadata"]["encoder_training_overlap"] = cfg.get("VAL_SEEN_BY_ENCODER", True)
    validate_cache(cache)
    # Test cache is small and portable. Train shards are already stored atomically.
    if split == "test":
        atomic_torch(Path(cfg["WORK_DIR"]) / "test_features.pt", cache)
        atomic_json(Path(cfg["WORK_DIR"]) / "test_feature_manifest.json", cache["metadata"])
    return cache


def _public_result(result):
    return {k: v for k, v in result.items() if not k.startswith("ranks_")}


def _official_source_metrics(scores, cache, vendor_dir):
    # Actually execute the pinned original evaluator, then compare with our copy.
    if len(cache["text_video_ids"]) != len(cache["video_ids"]) or len(
        set(cache["text_video_ids"])
    ) != len(cache["video_ids"]):
        return {"available": False, "reason": "Original evaluator needs 1 caption/video."}
    row_by_video = {vid: i for i, vid in enumerate(cache["text_video_ids"])}
    aligned = scores[[row_by_video[vid] for vid in cache["video_ids"]]]
    original = import_file("pinned_original_metrics", Path(vendor_dir) / "metrics.reference.py")
    return {
        "available": True,
        "text_to_video": original.compute_metrics(aligned),
        "video_to_text": original.compute_metrics(aligned.T),
    }


def _logit_scale(state):
    info = state.get("model_info", {})
    if "logit_scale" in info:
        scale = float(info["logit_scale"])
    elif "model" in state:
        scale = float(state["model"].logit_scale.exp().detach().cpu())
    else:
        scale = 1.0  # Only synthetic unit fixtures omit a model and model_info.
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Checkpoint logit scale must be finite and positive")
    return np.float32(scale)


def _two_way_rank_equal(a, b):
    return all(np.array_equal(a[k], b[k]) for k in ("ranks_text_to_video", "ranks_video_to_text"))


def benchmark(state, cache):
    cfg, output = state["config"], Path(state["config"]["WORK_DIR"])
    kwargs = dict(
        tau=cfg["TAU"],
        alpha=cfg["ALPHA"],
        query_chunk=cfg["QUERY_CHUNK"],
        video_chunk=cfg["VIDEO_CHUNK"],
        device=state["device"],
    )
    report, results = {"variants": {}, "comparisons": {}}, {}
    scale = _logit_scale(state)
    for name in ("B0", "B1", "B2"):
        if name != "B0" and not report["baseline_gate"]["passed"]:
            print(
                "Baseline parity failed; stop downstream variants and inspect baseline_gate.json."
            )
            break
        if state["device"].startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        cosine_scores = score_matrix(
            cache["text"], cache["frames"], cache["mask"], method=name, **kwargs
        )
        scores = cosine_scores * scale
        if state["device"].startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = evaluate_bidirectional(
            scores, cache["text_video_ids"], cache["video_ids"], cache["text_ids"]
        )
        original = _official_source_metrics(scores, cache, state["vendor_dir"])
        copied = aligned_original_metrics(scores, cache["text_video_ids"], cache["video_ids"])
        if original["available"]:
            for direction in ("text_to_video", "video_to_text"):
                for orig_key, our_key in (
                    ("R1", "R@1"),
                    ("R5", "R@5"),
                    ("R10", "R@10"),
                    ("MR", "MedR"),
                    ("MeanR", "MeanR"),
                ):
                    if orig_key in original[direction] and not np.isclose(
                        original[direction][orig_key], copied[direction][our_key], atol=1e-12
                    ):
                        raise AssertionError(
                            f"Copied and original evaluators differ: {direction}/{orig_key}"
                        )
        report["variants"][name] = {
            "internal": _public_result(result),
            "original": original,
            "seconds": elapsed,
            "tau": cfg["TAU"],
            "alpha": cfg["ALPHA"],
            "score_scale": "checkpoint_FP32_logits",
            "logit_scale": float(scale),
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20
            if state["device"].startswith("cuda")
            else None,
        }
        results[name] = result
        np.savez_compressed(
            output / f"{name}_scores.npz",
            scores=scores,
            text_ids=np.asarray(cache["text_ids"]),
            text_video_ids=np.asarray(cache["text_video_ids"]),
            video_ids=np.asarray(cache["video_ids"]),
            score_scale=np.asarray("logits"),
            logit_scale=scale,
        )
        if name == "B0":
            reference = (
                direct_meanp_reference(cache["text"], cache["frames"], cache["mask"]) * scale
            )
            numerical_ok = bool(np.allclose(scores, reference, rtol=cfg["RTOL"], atol=cfg["ATOL"]))
            reference_result = evaluate_bidirectional(
                reference, cache["text_video_ids"], cache["video_ids"], cache["text_ids"]
            )
            rank_ok = _two_way_rank_equal(result, reference_result)
            official_r1 = original["text_to_video"]["R1"] if original["available"] else None
            expected = cfg.get("EXPECTED_BASELINE_R1")
            metric_ok = expected is None or (
                official_r1 is not None
                and abs(official_r1 - expected) <= cfg.get("EXPECTED_R1_TOL_PP", 1e-8)
            )
            external = cfg.get("BASELINE_REFERENCE_NPZ", "")
            external_ok, external_rank_ok, external_max_error = None, None, None
            if external:
                expected_hash = cfg.get("BASELINE_REFERENCE_SHA256", "")
                if expected_hash and sha256_file(external) != expected_hash:
                    raise ValueError("Baseline reference SHA256 changed after preflight")
                with np.load(external, allow_pickle=False) as saved:
                    if not {"scores", "text_ids", "video_ids"}.issubset(saved.files):
                        raise ValueError("Reference NPZ needs scores, text_ids, video_ids")
                    if (
                        saved["text_ids"].tolist() != cache["text_ids"]
                        or saved["video_ids"].tolist() != cache["video_ids"]
                    ):
                        raise ValueError("Reference IDs/order differ from the locked cache")
                    original_scores = saved["scores"].astype(np.float32)
                    if not cfg.get("REFERENCE_IS_LOGITS", True):
                        original_scores = original_scores * scale
                    external_ok = original_scores.shape == scores.shape and bool(
                        np.allclose(scores, original_scores, rtol=cfg["RTOL"], atol=cfg["ATOL"])
                    )
                    if original_scores.shape == scores.shape:
                        external_max_error = float(np.max(np.abs(scores - original_scores)))
                        external_result = evaluate_bidirectional(
                            original_scores,
                            cache["text_video_ids"],
                            cache["video_ids"],
                            cache["text_ids"],
                        )
                        external_rank_ok = _two_way_rank_equal(result, external_result)
            implementation_ok = numerical_ok and rank_ok
            strict_required = bool(cfg.get("REQUIRE_EXTERNAL_REFERENCE", True))
            verified = external_ok is True and external_rank_ok is True
            passed = implementation_ok and (
                verified
                if strict_required
                else external_ok is not False and external_rank_ok is not False
            )
            gate = {
                "passed": bool(passed),
                "external_baseline_reproduced": verified,
                "implementation_score_parity": numerical_ok,
                "implementation_rank_parity": rank_ok,
                "max_abs_error": float(np.max(np.abs(scores - reference))),
                "expected_reported_r1": expected,
                "actual_original_r1": official_r1,
                "reported_metric_matches": metric_ok,
                "reported_R1_is_diagnostic_only": True,
                "external_score_parity": external_ok,
                "external_rank_parity": external_rank_ok,
                "external_max_abs_error": external_max_error,
                "external_reference_provided": bool(external),
                "external_reference_required": strict_required,
                "status": "external_scores_and_two_way_ranks_verified"
                if verified
                else "implementation_only_no_external_baseline_parity"
                if passed
                else "baseline_parity_failed_or_reference_missing",
            }
            report["baseline_gate"] = gate
            atomic_json(output / "baseline_gate.json", gate)
        else:
            report["comparisons"][f"{name}-B0"] = compare_runs(
                results["B0"],
                result,
                cache["text_video_ids"],
                cache["video_ids"],
                cache["text_ids"],
                n_bootstrap=cfg["BOOTSTRAP_SAMPLES"],
                seed=cfg["SEED"],
                output_csv=output / f"{name}_query_changes.csv",
            )
        atomic_json(output / "benchmark.json", report)
        print(
            f"{name}: t2v R@1={result['text_to_video']['R@1']:.2f}% | "
            f"v2t R@1={result['video_to_text']['R@1']:.2f}% | score {elapsed:.2f}s",
            flush=True,
        )
    rows = [
        {
            "variant": name,
            "direction": direction,
            **variant["internal"][direction],
            "score_seconds": variant["seconds"],
        }
        for name, variant in report["variants"].items()
        for direction in ("text_to_video", "video_to_text")
    ]
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report, results


def train_and_benchmark(state, test_cache, report, base_results):
    cfg, output = state["config"], Path(state["config"]["WORK_DIR"])
    if not cfg["RUN_TRAIN"]:
        print("RUN_TRAIN=False: finished frozen B0/B1/B2 benchmark.")
        return None
    if not report["baseline_gate"]["passed"]:
        print("B3 stopped because baseline parity failed or its required reference is missing.")
        return None
    if "model" not in state or "tokenizer" not in state:
        load_runtime(state, state["vendor_dir"])
    scale = _logit_scale(state)
    train = cache_split(state, "train")
    policy = cfg.get("SELECTION_POLICY", "fixed_epochs")
    val = cache_split(state, "val") if policy == "independent_validation" else None
    state["model"].cpu()
    del state["model"], state["tokenizer"]
    gc.collect()
    if state["device"].startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    trained = train_attention_head(
        train,
        val,
        output / "attention_head",
        epochs=cfg["EPOCHS"],
        batch_size=cfg["TRAIN_BATCH"],
        lr=cfg["LR"],
        weight_decay=cfg["WEIGHT_DECAY"],
        tau=cfg["TAU"],
        alpha=cfg["ALPHA"],
        logit_scale=cfg["TRAIN_LOGIT_SCALE"],
        seed=cfg["SEED"],
        device=state["device"],
        resume=bool(cfg["RESUME"] and (output / "attention_head" / "last.pt").is_file()),
        query_chunk=cfg["QUERY_CHUNK"],
        video_chunk=cfg["VIDEO_CHUNK"],
        selection_policy=policy,
        validation_provenance=cfg.get("VAL_PROVENANCE", ""),
    )
    if not trained["training_complete"]:
        raise RuntimeError("Training schedule incomplete; final benchmark is unavailable")
    if state["device"].startswith("cuda"):
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    scoring_started = time.perf_counter()
    scores = (
        score_matrix(
            test_cache["text"],
            test_cache["frames"],
            test_cache["mask"],
            method="B2",
            head=trained["head"],
            tau=cfg["TAU"],
            alpha=cfg["ALPHA"],
            query_chunk=cfg["QUERY_CHUNK"],
            video_chunk=cfg["VIDEO_CHUNK"],
            device=state["device"],
        )
        * scale
    )
    if state["device"].startswith("cuda"):
        torch.cuda.synchronize()
    score_seconds = time.perf_counter() - scoring_started
    result = evaluate_bidirectional(
        scores, test_cache["text_video_ids"], test_cache["video_ids"], test_cache["text_ids"]
    )
    summary = {
        "variant": "B3_trained_attention",
        "internal": _public_result(result),
        "original": _official_source_metrics(scores, test_cache, state["vendor_dir"]),
        "training_seconds": training_seconds,
        "score_seconds": score_seconds,
        "selected_epoch": trained["selected_epoch"],
        "best_epoch": trained["selected_epoch"],
        "best_val_r1": trained["best_val_r1"],
        "history": trained["history"],
        "selection_policy": policy,
        "validation_status": trained["validation_status"],
        "development_status": state["development_status"],
        "encoder_training_overlap": True
        if val is None
        else val["metadata"]["encoder_training_overlap"],
        "head_parameters": sum(p.numel() for p in trained["head"].parameters()),
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20
        if state["device"].startswith("cuda")
        else None,
        "score_scale": "checkpoint_FP32_logits",
        "logit_scale": float(scale),
        "comparisons": {},
    }
    for name in ("B0", "B2"):
        if name in base_results:
            summary["comparisons"][f"B3-{name}"] = compare_runs(
                base_results[name],
                result,
                test_cache["text_video_ids"],
                test_cache["video_ids"],
                test_cache["text_ids"],
                n_bootstrap=cfg["BOOTSTRAP_SAMPLES"],
                seed=cfg["SEED"],
                output_csv=output / f"B3_vs_{name}_query_changes.csv",
            )
    atomic_json(output / "trained_benchmark.json", summary)
    with (output / "trained_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        rows = [
            {
                "variant": "B3",
                "direction": direction,
                **result[direction],
                "training_seconds": training_seconds,
                "selected_epoch": trained["selected_epoch"],
                "selection_policy": policy,
            }
            for direction in ("text_to_video", "video_to_text")
        ]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output / "B3_scores.npz",
        scores=scores,
        text_ids=np.asarray(test_cache["text_ids"]),
        text_video_ids=np.asarray(test_cache["text_video_ids"]),
        video_ids=np.asarray(test_cache["video_ids"]),
        score_scale=np.asarray("logits"),
        logit_scale=scale,
    )
    print(
        f"B3: t2v R@1={result['text_to_video']['R@1']:.2f}%, "
        f"train={training_seconds:.1f}s, epoch={trained['selected_epoch']}, policy={policy}."
    )
    return summary

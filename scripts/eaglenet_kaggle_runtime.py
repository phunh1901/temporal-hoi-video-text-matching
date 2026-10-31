"""Offline, inference-only EagleNet runner; source and data must match the pinned bundle.

The upstream decoder, model constructor, stochastic scorer and legacy evaluator are
used unchanged. The adapter replaces permissive pretrained loading with numeric-only
strict loading and rejects partial decodes. A CPU probe is a technical check, never a
benchmark. No training, automatic download, sample skipping or checkpoint selection.
"""

import argparse
import csv
import hashlib
import importlib
import json
import logging
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

UPSTREAM_COMMIT = "00e6bcdcc13dbdefbc793ad9b811e3608cbddebe"
METADATA_HASHES = {
    "annotations": "3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014",
    "test_csv": "e61144b312c8a2fc2126f96504c981f8c23c05b3ba9655a56e2af50ac8a4b16a",
    "train_csv": "0a7ece2418e1f7ac5680555a04ddfef8950293ac3c4bdd6584fb0fa6cc0ff7d3",
}
# Filled from the independently downloaded archive at the commit above, not a runtime manifest.
SOURCE_HASHES = {
    "args.py": "21d1b7d164095e72b633775f3dd2ca6be7ed04d61ad53c46e17c43b2e1159d8d",
    "main_my.py": "6108b6fe9eb6e092b344d76a6105a9fa35fda9c5763c89152dff931e11452442",
    "metrics.py": "6e8b42ae69e889ee6d80efd135f6094d9bdb6f554166ce08e6c2cbcd7e486b1f",
    "train_and_eval.py": "a5fdf60f6a1fd9f40a0b5fd6aecd7a6c21373a32ae06ded465b13e50daeb9471",
    "util.py": "c7be5c2ccd823483062a1f7c7865b2d462428d0a5febeb8e02961869b937909e",
    "dataloaders/dataloader_didemo_retrieval.py": "51fa65886954a4eb1c90a77bb3b5a2c3539ccaca081d2bd509de9e2bf6d14f48",
    "dataloaders/dataloader_msrvtt_retrieval.py": "b4e194c81465c33a3a69dc524c989619123cdad3fd50c944a34718c1c5fdf0fa",
    "dataloaders/dataloader_msvd_retrieval.py": "521119c9819a3f7dd238e75a5a78cf2b4fca9c43a991e1c715f0b06f42179b90",
    "dataloaders/dataloader_vatex_retrieval.py": "eda3529855c3a70005b74371bfe21d5e6f9472e76151daec66db6675fd0f7a89",
    "dataloaders/data_dataloaders.py": "df199fc0d381f5686955fbe68821ae854a1f560065df27f1a3c2055f604b91af",
    "dataloaders/rawvideo_util.py": "cb62541f12da740d9c1511519c9e4ae0785517413bd712ac32e7fe6776ba255a",
    "modules/bpe_simple_vocab_16e6.txt.gz": "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
    "modules/decoder.py": "451d1b4955fe78ea287d7c0f4b4809d6e7ea511d4d354e8bba968e2a366a1e53",
    "modules/differential_topk.py": "42f9b0cb08bc3bbd431d5f89638aad03e93b1cc3f8b8b56eca10f1e2079055ca",
    "modules/ebm.py": "2b15b2e8b3c84fe8d6e2c39a6b088fe2b008faa6045841f72953b8b9bfa18337",
    "modules/file_utils.py": "9bbc073d4ac83c28e8c17d745daac1c6b70c5c3b2581c7541cd66c888bdb5679",
    "modules/gnn.py": "45c63ffb6639ae7e37b4026dc177a14a5a50ff8f819a8ddb721d1a70bf23e713",
    "modules/modeling.py": "fcae758692415210a2291932d3d4fdb85ea8204a366753b4e7e5cb6362022293",
    "modules/module_clip.py": "60edc99178e9521a9aa066ce05aee18c8c78d78944b7c386ccb3576528a79f92",
    "modules/module_cross.py": "5d109edc3f2fd150d4c3db91912d4c220fa594e2e4708bf313fcf55968195d5d",
    "modules/optimization.py": "7dab5de2bbd1398e07ddbb18667544caae385d8ef9c17c55e0358303ed28b4b8",
    "modules/stochastic_module.py": "cde9a76cdc36692403a3699f44ab0c3210eb4ff54305af4f59cd78660445ada0",
    "modules/tokenization_clip.py": "60af2eccd9d1208933213b57d24c9ac337ca4be935bc6550a7300b656a6f3bb5",
    "modules/transformer.py": "dc6e11f512a1f59c01575539942cb5290afa4efa6a36f160b9c32145ff046b60",
    "modules/transformer_eaglenet.py": "91b41eb272358501036ca9617a60b5c905259e65c4a226ce85eb580c2d2d1029",
    "modules/Tran_utils.py": "b1b5d638a10894554a2df292f02fdd4dc536bc5a3a09a0fb5450fb6b493ad766",
    "modules/until_config.py": "a410fb06ad08e3a7b5b9cb2f9b598148990f9d2005919f8ac964e0e80c231bf0",
    "modules/until_module.py": "28104d41a6a57fe517ea83a24b26c21037917cab1821344d42dde5f7a12ff94d",
    "modules/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "preprocess/compress_video.py": "b147745b9c9b11468ee504fbacbf001096b908adb238fb8faab4a14e70bee02e",
    "modules/cross-base/cross_config.json": "a19c3ce510f6cf2d765e5337cfb65a623526ab53cc6c73d0c4185124fc3f9477"
}
FIXED_CONFIG = {
    "seed": 0, "batch_size_val": 8, "max_frames": 12, "max_words": 32,
    "stochasic_trials": 20, "feature_framerate": 1, "eval_frame_order": 0,
    "slice_framepos": 2, "sim_header": "seqTransf", "linear_patch": "2d",
    "loose_type": 1, "cross_num_hidden_layers": 4, "gnn_type": "rgat",
    "gnn_num_layers": 2, "gnn_nheads": 4, "gnn_nrels": 3, "gnn_dropout": 0.5,
    "gnn_leaky_slope": 0.1, "gnn_v2": 0, "framepe": 1, "loss_fn": "sig",
    "stochastic_prior": "uniform01", "stochastic_prior_std": 1.0,
    "pooling_head": 1, "pooling_dropout": 0.3, "support_loss_weight": 0.8,
    "eam_loss_weight": 1.0, "eam_support_loss_weight": 1.0,
    "mcmc_coef_reg": 1.0, "mcmc_steps": 20, "mcmc_step_size": 1.0,
    "mcmc_noise": 0.005, "max_buffer_vol": 500, "energy_fn": "bilinear",
    "energy_pooling": "avg", "fix_sig_loss_param": 0, "temperature": 4.77,
    "bias": -12.93,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
                          + "\n", encoding="utf-8")


def event(name, **values):
    print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name,
                      **values}, ensure_ascii=False), flush=True)


def safe_path(root, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Bundle path must be a relative POSIX path")
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Unsafe bundle path")
    if not candidate.is_relative_to(root.resolve()) or candidate == root.resolve():
        raise ValueError("Bundle path escapes root")
    return candidate


def check_asset(root, item):
    if not isinstance(item, dict) or not {"path", "sha256"} <= item.keys():
        raise ValueError("Asset needs path and SHA256")
    path = safe_path(root, item["path"])
    expected = item["sha256"]
    if (not isinstance(expected, str) or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)):
        raise ValueError("Invalid asset SHA256")
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"Missing/nonempty asset: {item['path']}")
    if "bytes" in item and path.stat().st_size != item["bytes"]:
        raise ValueError(f"Asset size mismatch: {item['path']}")
    if sha256(path) != expected:
        raise ValueError(f"Asset hash mismatch: {item['path']}")
    return path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def preflight(root):
    manifest_path = root / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != "eaglenet_eval_bundle_v1":
        raise ValueError("Unsupported EagleNet asset manifest")
    if not isinstance(manifest.get("fixture"), bool):
        raise ValueError("Manifest must explicitly distinguish fixture from real assets")
    if manifest["source"]["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Source commit is not the locked EagleNet commit")
    source = safe_path(root, manifest["source"]["path"])
    expected_python = {name for name in SOURCE_HASHES if name.endswith(".py")}
    actual_python = {path.relative_to(source).as_posix() for path in source.rglob("*.py")}
    if actual_python != expected_python:
        raise ValueError("Unexpected or missing Python files in pinned source tree")
    for relative, expected in SOURCE_HASHES.items():
        path = safe_path(source, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned source hash mismatch: {relative}")
    for relative, item in manifest["source"]["files"].items():
        expected = item["sha256"] if isinstance(item, dict) else item
        if sha256(safe_path(source, relative)) != expected:
            raise ValueError(f"Source manifest hash mismatch: {relative}")
    if not SOURCE_HASHES:
        raise ValueError("Runtime source pin inventory is empty")
    for key in ("checkpoint", "clip_checkpoint"):
        if key in manifest:
            check_asset(root, manifest[key])
    if "checkpoint" not in manifest:
        raise ValueError("Missing fine-tuned EagleNet checkpoint")
    metadata = {key: check_asset(root, manifest["metadata"][key])
                for key in ("test_csv", "train_csv", "annotations")}
    for key, path in metadata.items():
        if sha256(path) != METADATA_HASHES[key]:
            raise ValueError(f"Canonical metadata hash mismatch: {key}")
    test = read_csv(metadata["test_csv"])
    train = read_csv(metadata["train_csv"])
    test_ids = [row["video_id"] for row in test]
    train_ids = [row["video_id"] for row in train]
    if len(test_ids) != 1000 or len(set(test_ids)) != 1000:
        raise ValueError("Require the entire unique MSRVTT test1k pool")
    if len(train_ids) != 9000 or len(set(train_ids)) != 9000:
        raise ValueError("Require original unique train9k IDs")
    if set(test_ids) & set(train_ids):
        raise ValueError("Train/test ID overlap")
    if test_ids != manifest["videos"]["test_ids"]:
        raise ValueError("Test ID/order mismatch")
    probe_ids = manifest["videos"]["probe_ids"]
    if (not probe_ids or len(set(probe_ids)) != len(probe_ids)
            or not set(probe_ids) <= set(train_ids)):
        raise ValueError("Probe IDs must be unique train-only IDs")
    captions = json.loads(metadata["annotations"].read_text(encoding="utf-8-sig"))
    known_captions = {}
    for row in captions["sentences"]:
        known_captions.setdefault(row["video_id"], []).append(row["caption"])
    if any(not row.get("sentence", "").strip() for row in test):
        raise ValueError("Test CSV query is empty")
    # The original JSFusion CSV has 35 sentences absent verbatim from its video's
    # JSON captions. They remain the canonical queries; record, never rewrite them.
    mismatch_ids = [row["video_id"] for row in test
                    if row["sentence"] not in known_captions.get(row["video_id"], [])]
    if any(not known_captions.get(video_id) for video_id in probe_ids):
        raise ValueError("Probe caption missing from original metadata")
    video_root = safe_path(root, manifest["videos"]["root"])
    for video_id in test_ids + probe_ids:
        if not video_id.startswith("video") or not video_id[5:].isdigit():
            raise ValueError("Invalid video ID")
        item = manifest["videos"]["files"].get(video_id)
        if item is None:
            raise ValueError(f"Missing video inventory: {video_id}")
        path = check_asset(root, item)
        if path != video_root / f"{video_id}.mp4":
            raise ValueError("Video layout must match the original decoder")
    if manifest["variant"] not in {"ViT-B/16", "ViT-B/32"}:
        raise ValueError("Unsupported architecture")
    for key, value in manifest.get("config", {}).items():
        if key not in FIXED_CONFIG or value != FIXED_CONFIG[key]:
            raise ValueError(f"Configuration differs from locked inference: {key}")
    report = {"schema": "eaglenet_preflight_v1", "status": "preflight_passed",
              "fixture": manifest["fixture"], "source_commit": UPSTREAM_COMMIT,
              "manifest_sha256": sha256(manifest_path), "source_files": len(SOURCE_HASHES),
              "test_queries": len(test), "test_candidates": len(test_ids),
              "csv_json_caption_mismatch_count": len(mismatch_ids),
              "csv_json_caption_mismatch_video_ids": mismatch_ids,
              "canonical_csv_queries_preserved": True,
              "probe_ids": probe_ids, "model_loaded": False, "decode_checked": False,
              "benchmark_inference_completed": False}
    return manifest, test, known_captions, report


def numeric_state(path):
    """No TorchScript/pickle fallback, no model objects or partial base-CLIP state."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and set(state) == {"state_dict"}:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint must be a nonempty tensor state dict")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ValueError("Checkpoint contains non-tensor data")
    prefixes = [key.startswith("module.") for key in state]
    if any(prefixes) and not all(prefixes):
        raise ValueError("Mixed DDP checkpoint prefixes")
    if all(prefixes):
        state = {key[7:]: value for key, value in state.items()}
    for key, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {key}")
    return state


def strict_load(model, state):
    expected = model.state_dict()
    missing, extra = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError(f"Incomplete checkpoint: missing={missing[:8]}, unexpected={extra[:8]}")
    for key, value in expected.items():
        if value.shape != state[key].shape:
            raise ValueError(f"Checkpoint shape mismatch: {key}")
    model.load_state_dict(state, strict=True)
    return {"strict_load": True, "missing_keys": [], "unexpected_keys": [],
            "tensor_count": len(state), "parameters": sum(p.numel() for p in model.parameters())}


def import_upstream(source):
    """Import only after hashes are checked; reject a contaminated shared interpreter."""
    for name, module in list(sys.modules.items()):
        if name in {"metrics", "train_and_eval"} or name.split(".")[0] in {
                "modules", "dataloaders"}:
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(source):
                raise RuntimeError(f"Conflicting upstream import: {name}; use a fresh process")
            for location in getattr(module, "__path__", []):
                if not Path(location).resolve().is_relative_to(source):
                    raise RuntimeError(f"Conflicting upstream namespace: {name}")
    sys.path.insert(0, str(source))
    return SimpleNamespace(
        model=importlib.import_module("modules.modeling"),
        data=importlib.import_module("dataloaders.dataloader_msrvtt_retrieval"),
        tokenizer=importlib.import_module("modules.tokenization_clip"),
        evaluator=importlib.import_module("train_and_eval"),
        metrics=importlib.import_module("metrics"),
    )


def make_model(root, manifest, upstream, device):
    state = numeric_state(safe_path(root, manifest["checkpoint"]["path"]))
    patch = 16 if manifest["variant"] == "ViT-B/16" else 32
    if tuple(state.get("clip.visual.conv1.weight", torch.empty(0)).shape) != (768, 3, patch, patch):
        raise ValueError("Checkpoint architecture does not match selected CLIP variant")
    config = SimpleNamespace(**FIXED_CONFIG, pretrained_clip_name=manifest["variant"],
                             local_rank=0, n_gpu=1, world_size=1, rank=0)
    clip_shapes = {key[5:].replace("attn.net.", "attn."): value
                   for key, value in state.items() if key.startswith("clip.")}
    cross_path = safe_path(root, manifest["source"]["path"]) / "modules/cross-base/cross_config.json"
    cross = upstream.model.CrossConfig.from_json_file(str(cross_path))
    cross.type_vocab_size = 2
    # Same constructor and patch-shift wrapper as upstream from_pretrained, with
    # no base-model filling, downloaded config, permissive key loader or JIT execution.
    model = upstream.model.EagleNet(cross, clip_shapes, task_config=config)
    upstream.model.make_patch_shift(model, video_frame=12, n_div=7)
    audit = strict_load(model, state)
    audit.update({"checkpoint_sha256": manifest["checkpoint"]["sha256"],
                  "base_weights_used_as_fallback": False, "loader": "weights_only=True",
                  "constructor": "upstream EagleNet + make_patch_shift + strict state_dict",
                  "cpu_float32_adapter": device.type == "cpu"})
    if device.type == "cpu":
        model.float()
    model.requires_grad_(False).to(device).eval()
    return model, config, audit


def stable_metrics(scores):
    """One positive per row; stable candidate order breaks ties, ranks are 1-based."""
    scores = np.asarray(scores)
    if (scores.ndim != 2 or scores.shape[0] != scores.shape[1]
            or not scores.size or not np.isfinite(scores).all()):
        raise ValueError("Require a finite nonempty square score matrix")
    result = {}
    for name, matrix in (("text_to_video", scores), ("video_to_text", scores.T)):
        order = np.argsort(-matrix, axis=1, kind="stable")
        ranks = np.argmax(order == np.arange(len(matrix))[:, None], axis=1) + 1
        result[name] = {**{f"R@{k}": float(np.mean(ranks <= k) * 100) for k in (1, 5, 10)},
                        "MedR": float(np.median(ranks)), "MeanR": float(np.mean(ranks)),
                        "MRR": float(np.mean(1 / ranks)), "positive_ranks": ranks.tolist(),
                        "query_count": len(matrix), "candidate_count": len(matrix),
                        "positive_pairs": len(matrix)}
    return result


class CheckedDataset(torch.utils.data.Dataset):
    """Exact upstream values; fail early on metadata/partial decode instead of zero-padding it."""

    def __init__(self, dataset, ids):
        self.dataset, self.ids, self.records = dataset, ids, []
        original = dataset.rawVideoExtractor.get_video_data

        def checked(path, start_time=None, end_time=None):
            import cv2

            cap = cv2.VideoCapture(path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            opened = cap.isOpened()
            cap.release()
            if not opened or frames <= 0 or fps <= 0:
                raise ValueError(f"Invalid media metadata: {Path(path).name}")
            started = time.perf_counter()
            output = original(path, start_time=start_time, end_time=end_time)
            tensor = output["video"]
            expected = len(range(0, frames, fps))
            if tensor.ndim != 4 or len(tensor) != expected or not torch.isfinite(tensor).all():
                raise ValueError(f"Partial/invalid native decode: {Path(path).name}")
            selection = np.linspace(0, expected - 1, 12, dtype=int) if expected > 12 else np.arange(expected)
            self.records.append({"video_id": Path(path).stem, "frame_count": frames,
                                 "integer_fps": fps, "native_sampled_frames": expected,
                                 "selected_frame_indices": (selection * fps).tolist(),
                                 "valid_frames": len(selection),
                                 "seconds": time.perf_counter() - started})
            return output

        dataset.rawVideoExtractor.get_video_data = checked

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        row = self.dataset[index]
        if not np.any(row[-1]) or not np.isfinite(row[-2]).all():
            raise ValueError(f"Invalid native dataset row: {self.ids[index]}")
        return row


def environment(device):
    versions = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(), "device": str(device)}
    for name in ("torchvision", "cv2", "PIL", "pandas", "scipy", "ftfy", "regex"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError as exc:
            versions[name] = f"unavailable: {exc}"
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        versions.update({"gpu_name": props.name, "gpu_total_memory_bytes": props.total_memory,
                         "gpu_capability": list(torch.cuda.get_device_capability(device))})
    return versions


def seed_runtime(seed, device, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate_native(upstream, config, model, dataloader, device, logger, output, probe):
    """Observe the native loop without changing its block order or stochastic RNG draws."""
    original = upstream.evaluator._run_on_single_gpu
    captured = {}

    def observe(model, text_masks, video_masks, text_features, video_features):
        captured.update({"args": (model, text_masks, video_masks, text_features, video_features),
                         "cpu_rng": torch.get_rng_state().clone(),
                         "cuda_rng": torch.cuda.get_rng_state(device).clone()
                         if device.type == "cuda" else None})
        return original(model, text_masks, video_masks, text_features, video_features)

    upstream.evaluator._run_on_single_gpu = observe
    try:
        _, scores, timings = upstream.evaluator.eval_epoch(config, model, dataloader,
                                                          device, 1, logger)
    finally:
        upstream.evaluator._run_on_single_gpu = original
    _, text_masks, video_masks, text_features, video_features = captured["args"]
    np.savez_compressed(
        output / "feature_cache.npz",
        text=torch.cat(text_features).detach().cpu().numpy(),
        video=torch.cat(video_features).detach().cpu().numpy(),
        text_mask=torch.cat([item[0] for item in text_masks]).detach().cpu().numpy(),
        text_segments=torch.cat([item[1] for item in text_masks]).detach().cpu().numpy(),
        video_mask=torch.cat([item[0] for item in video_masks]).detach().cpu().numpy(),
        cpu_rng_before_scoring=captured["cpu_rng"].numpy(),
        cuda_rng_before_scoring=captured["cuda_rng"].cpu().numpy()
        if captured["cuda_rng"] is not None else np.empty(0, dtype=np.uint8),
    )
    if probe:
        torch.set_rng_state(captured["cpu_rng"])
        if captured["cuda_rng"] is not None:
            torch.cuda.set_rng_state(captured["cuda_rng"], device)
        with torch.no_grad():
            replay = np.concatenate(original(*captured["args"]), axis=0)
        equal = np.array_equal(scores, replay)
        write_json(output / "probe_parity.json", {
            "schema": "eaglenet_fixed_cache_replay_v1", "score_bitwise_equal": equal,
            "max_abs_difference": float(np.max(np.abs(scores - replay))),
            "scope": "same-device same-cache same-blocks restored-RNG native scorer",
            "decoder_encoder_repeated": False,
        })
        if not equal:
            raise ValueError("Native probe scoring differs after exact RNG/cache replay")
    return scores, timings


def run(args):
    root, output = Path(args.bundle_root).resolve(), Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    status = {"schema": "eaglenet_runtime_status_v1", "stage": args.stage,
              "status": "running", "fixture": None, "benchmark_inference_completed": False,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    try:
        event("preflight_started", stage=args.stage)
        if args.threads < 1 or not 1 <= args.probe_videos <= 64:
            raise ValueError("Invalid CPU threads or probe size")
        manifest, test, captions, report = preflight(root)
        status["fixture"] = manifest["fixture"]
        status["asset_manifest_sha256"] = report["manifest_sha256"]
        write_json(output / "preflight.json", report)
        event("preflight_passed", fixture=manifest["fixture"], test_videos=len(test))
        write_json(output / "config.json", {"stage": args.stage, "device": args.device,
                   "threads": args.threads, "variant": manifest["variant"], **FIXED_CONFIG})
        if args.stage == "preflight":
            status["status"] = "preflight_passed"
            return status
        if manifest["fixture"]:
            raise ValueError("Fixture assets cannot execute a real probe or benchmark")
        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; no automatic CPU fallback")
        if args.device == "cuda" and torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA GPU to preserve the single-GPU protocol")
        if args.stage == "evaluate" and args.device != "cuda":
            raise ValueError("Full benchmark is CUDA-only; use CPU only for the train probe")
        write_json(output / "environment.json", environment(device))
        seed_runtime(0, device, args.threads)
        upstream = import_upstream(safe_path(root, manifest["source"]["path"]))
        logger = logging.getLogger("eaglenet_runtime")
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(output / "run.log", encoding="utf-8"))
        logger.addHandler(logging.StreamHandler(sys.stdout))
        event("model_loading_started", variant=manifest["variant"], device=str(device))
        model, config, audit = make_model(root, manifest, upstream, device)
        write_json(output / "checkpoint_audit.json", audit)
        event("model_strict_loaded", tensors=audit["tensor_count"])
        if args.stage == "probe":
            if args.probe_videos > len(manifest["videos"]["probe_ids"]):
                raise ValueError("Requested probe is larger than packaged train media")
            ids = manifest["videos"]["probe_ids"][:args.probe_videos]
            rows = [{"video_id": video_id, "sentence": captions[video_id][0]} for video_id in ids]
            csv_path = output / "probe_pool.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["video_id", "sentence"])
                writer.writeheader()
                writer.writerows(rows)
        else:
            rows, ids = test, [row["video_id"] for row in test]
            csv_path = safe_path(root, manifest["metadata"]["test_csv"]["path"])
        dataset = upstream.data.MSRVTT_DataLoader(
            str(csv_path), str(safe_path(root, manifest["videos"]["root"])),
            upstream.tokenizer.SimpleTokenizer(), max_words=32, feature_framerate=1,
            max_frames=12, image_resolution=224, frame_order=0, slice_framepos=2)
        checked = CheckedDataset(dataset, ids)
        dataloader = torch.utils.data.DataLoader(checked, batch_size=8, num_workers=0,
                                               shuffle=False, drop_last=False)
        write_json(output / "ids.json", {"video_ids": ids, "text_video_ids": ids,
                   "text_ids": [f"{vid}:query0" for vid in ids],
                   "captions": [row["sentence"] for row in rows],
                   "scope": "upstream_train_probe" if args.stage == "probe" else "JSFusion_test1k"})
        # Calling the entire original loop preserves DataLoader RNG consumption,
        # feature-cache batching, text-major block order and stochastic draw order.
        event("native_evaluation_started", stage=args.stage, videos=len(ids), batch_size=8)
        scores, timings = evaluate_native(upstream, config, model, dataloader, device, logger,
                                         output, args.stage == "probe")
        if len(checked.records) != len(ids) or [r["video_id"] for r in checked.records] != ids:
            raise ValueError("Native decoder did not cover the entire requested ordered pool")
        scores = np.asarray(scores)
        internal = stable_metrics(scores)
        if scores.shape != (len(ids), len(ids)):
            raise ValueError("Unexpected score shape")
        np.save(output / "scores.npy", scores, allow_pickle=False)
        write_json(output / "metrics_internal.json", {"tie_break": "stable candidate order",
                   "recall_unit": "percent", "rank_unit": "one_based", **internal})
        legacy = {"text_to_video": upstream.metrics.compute_metrics(scores),
                  "video_to_text": upstream.metrics.compute_metrics(scores.T)}
        write_json(output / "metrics_upstream.json", {"tie_break": "legacy all diagonal ties",
                   "recall_unit": "percent", **legacy})
        write_json(output / "decode_audit.json", {"drop_count": 0, "records": checked.records})
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cost = {"total_seconds": time.perf_counter() - started,
                "encoder_cache_seconds": timings[0], "pair_scoring_seconds": timings[1],
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                if device.type == "cuda" else None,
                "training_run": False, "encoder_run": True,
                "scope": "technical_probe" if args.stage == "probe" else "test1k_inference"}
        write_json(output / "cost.json", cost)
        status.update({"status": "probe_completed" if args.stage == "probe"
                       else "evaluation_completed", "model_strict_loaded": True,
                       "query_count": len(ids), "candidate_count": len(ids),
                       "score_shape": list(scores.shape), "all_scores_finite": True,
                       "benchmark_inference_completed": args.stage == "evaluate"})
        return status
    except Exception as exc:
        status.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                       "benchmark_inference_completed": False})
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "status.json", status)
        inventory = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                     for p in output.iterdir() if p.is_file() and p.name != "artifacts.json"}
        write_json(output / "artifacts.json", {"schema": "eaglenet_output_inventory_v1",
                   "files": inventory, "fixture": status["fixture"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("preflight", "probe", "evaluate"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--probe-videos", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    # This is a fresh-process requirement; setting it here would not reset Python's hash seed.
    if os.environ.get("PYTHONHASHSEED") not in {None, "0"}:
        parser.error("Set PYTHONHASHSEED=0 before launching the runtime")
    try:
        print(json.dumps(run(args), ensure_ascii=False), flush=True)
    except Exception as exc:
        print(f"EagleNet {args.stage} failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

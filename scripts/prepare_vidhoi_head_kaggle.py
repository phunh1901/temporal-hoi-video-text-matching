"""Package a bounded VidHOI train/development frozen-feature head study locally."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import textwrap
import zipfile
from pathlib import Path

from temporal_hoi.inference.hoi_kaggle import (
    MAX_BUNDLE_BYTES,
    MAX_FILES,
    safe_member,
    sha256_file,
    validate_manifest,
    validate_requirements,
    wheel_distribution,
)

BUNDLE_SCHEMA = "vidhoi_frozen_development_bundle_v1"
MODES = ("object", "union", "fusion")
SEEDS = (42, 43, 44)
SOURCES = (
    "src/temporal_hoi/__init__.py", "src/temporal_hoi/inference/__init__.py",
    "src/temporal_hoi/data/vidhoi.py", "src/temporal_hoi/evaluation/hoi.py",
    "src/temporal_hoi/evaluation/hoi_head_audit.py",
    "src/temporal_hoi/inference/hoi_static.py", "src/temporal_hoi/inference/hoi_kaggle.py",
    "src/temporal_hoi/inference/hoi_diagnostics.py",
    "src/temporal_hoi/data/hoi_head_cache.py", "src/temporal_hoi/models/hoi_head.py",
    "src/temporal_hoi/training/hoi_head_study.py", "scripts/run_vidhoi_head_study.py",
)
NAMESPACES = ("data", "evaluation", "models", "training")


def _base_module():
    spec = importlib.util.spec_from_file_location(
        "_vidhoi_head_base_packager", Path(__file__).with_name("prepare_vidhoi_kaggle.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_bytes(value):
    return _base_module()._json_bytes(value)


def _record(value):
    return _base_module()._record(value)


def _file(path, label):
    return _base_module()._plain_file(Path(path), label)


def _collect_role(manifest_path: Path, role: str, *, protocol_hash: str, mapping_hash: str):
    manifest_path = _file(manifest_path, f"{role} manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"schema_version": "vidhoi_head_frames_v1", "upstream_split": "train",
                "role": role, "box_source": "oracle", "object_label_source": "oracle",
                "protocol_sha256": protocol_hash, "mapping_sha256": mapping_hash}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Invalid {role} manifest scope or protocol/mapping hash")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"The {role} manifest needs actual nonempty frames")
    files, identities, videos, remap = {}, set(), set(), []
    root = manifest_path.parent.resolve()
    for index, frame in enumerate(frames):
        identity = (frame["video_id"], frame["frame_id"])
        if identity in identities:
            raise ValueError(f"Duplicate frame identity in {role}")
        identities.add(identity)
        videos.add(frame["video_id"])
        safe_member(frame["image_path"])
        image = root / frame["image_path"]
        if not image.resolve().is_relative_to(root):
            raise ValueError("Frame image escapes its role manifest folder")
        _file(image, "role frame image")
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Study frames must be raster images")
        if sha256_file(image) != frame.get("image_sha256"):
            raise ValueError(f"Frame image SHA256 mismatch in {role} at index {index}")
        destination = f"assets/frames/{role}/{index:05d}{image.suffix.lower()}"
        files[destination] = image
        frame["image_path"] = destination.removeprefix("assets/")
        remap.append({"role": role, "image_path": destination,
                      "image_sha256": frame["image_sha256"]})
    return manifest, files, videos, remap


def prepare(train_manifest, dev_manifest, protocol, mapping, clip_checkpoint, clip_sha256,
            output, *, checkpoint_provenance, wheelhouse, study_config,
            data_provenance=(), model_license=None, project_root=None):
    """Collect two audited roles and reuse the safe bootstrap; never run the study."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new package output; existing evidence is preserved")
    project = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    protocol, mapping = _file(protocol, "development protocol"), _file(mapping, "numeric mapping")
    checkpoint = _file(clip_checkpoint, "original CLIP checkpoint")
    if checkpoint.suffix.lower() != ".safetensors" or sha256_file(checkpoint) != clip_sha256:
        raise ValueError("Require the SHA256-verified original CLIP safetensors checkpoint")
    provenance = json.loads(_file(checkpoint_provenance, "checkpoint provenance").read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or not provenance or provenance.get("checkpoint_sha256") != clip_sha256:
        raise ValueError("Require checkpoint provenance bound to the actual SHA256")
    config = json.loads(_file(study_config, "reviewed head study config").read_text(encoding="utf-8"))
    if config.get("schema_version") != "vidhoi_head_study_config_v1":
        raise ValueError("Require a reviewed vidhoi_head_study_config_v1")
    protocol_hash, mapping_hash = sha256_file(protocol), sha256_file(mapping)
    train, train_files, train_videos, train_remap = _collect_role(
        Path(train_manifest), "train", protocol_hash=protocol_hash, mapping_hash=mapping_hash)
    dev, dev_files, dev_videos, dev_remap = _collect_role(
        Path(dev_manifest), "validation", protocol_hash=protocol_hash, mapping_hash=mapping_hash)
    if train_videos & dev_videos:
        raise ValueError("Train and development videos overlap")
    files = {name: _file(project / name, "head study source") for name in SOURCES}
    for namespace in NAMESPACES:
        files[f"src/temporal_hoi/{namespace}/__init__.py"] = (
            b'"""Minimal namespace for the portable frozen-feature head development study."""\n')
    files.update(train_files)
    files.update(dev_files)
    files.update({"assets/train_manifest.json": _json_bytes(train),
                  "assets/dev_manifest.json": _json_bytes(dev), "assets/protocol.json": protocol,
                  "assets/class_mapping.json": mapping,
                  "assets/clip_original.safetensors": checkpoint,
                  "provenance/checkpoint.json": _json_bytes(provenance),
                  "provenance/source_train_manifest.json": Path(train_manifest),
                  "provenance/source_dev_manifest.json": Path(dev_manifest)})
    for name, value in (("train_manifest", "train_manifest.json"),
                        ("dev_manifest", "dev_manifest.json"), ("protocol", "protocol.json"),
                        ("mapping", "class_mapping.json"), ("checkpoint", "clip_original.safetensors")):
        config[name] = value
        config[f"{name}_sha256"] = _record(files[f"assets/{value}"])["sha256"]
    config["checkpoint_provenance"] = provenance
    files["assets/run_config.json"] = _json_bytes(config)
    files["provenance/reviewed_study_config.json"] = Path(study_config)
    for index, source in enumerate(data_provenance):
        source = _file(source, "public data provenance")
        if source.suffix.lower() not in {".json", ".txt", ".md"} or source.stat().st_size > 8*1024**2:
            raise ValueError("Provenance must be a small public JSON/text/Markdown receipt")
        safe_member(source.name)
        files[f"provenance/data/{index:02d}_{source.name}"] = source
    if model_license is not None:
        license_path = _file(model_license, "model license")
        if license_path.stat().st_size > 1024**2:
            raise ValueError("Model license is too large")
        files["provenance/CLIP_LICENSE.txt"] = license_path
    wheelhouse = Path(wheelhouse)
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise ValueError("An ordinary local offline wheelhouse is required")
    requirements = _file(wheelhouse / "requirements-offline.txt", "offline dependency pins")
    validate_requirements(requirements.read_text(encoding="utf-8"))
    files["wheelhouse/requirements-offline.txt"] = requirements
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise ValueError("The offline wheelhouse is empty")
    for wheel in wheels:
        _file(wheel, "dependency wheel")
        wheel_distribution(wheel.name)
        files[f"wheelhouse/{wheel.name}"] = wheel
    scope = ("# VidHOI frozen-feature head development\n\n"
             "Upstream train only, separate audited train/development roles. Oracle boxes "
             "and object labels. Frozen original CLIP, small object/union/fusion heads. "
             "No official final split or final-unseen access. Development scores are not "
             "full benchmark or independent final results. No M2 improvement claim.\n"
             "Only selected frames, labels, reviewed public protocol/config, source and "
             "local weights/wheels are packaged. No private project instructions, reports, "
             "progress records or credentials. Hashes do not grant redistribution rights.\n")
    files["BUNDLE_SCOPE.md"] = scope.encode("utf-8")
    records = {name: _record(value) for name, value in sorted(files.items())}
    total = sum(value["bytes"] for value in records.values())
    if len(records) > MAX_FILES or total > MAX_BUNDLE_BYTES:
        raise ValueError("The actual development bundle exceeds the existing file/byte limit")
    manifest = {"schema_version": BUNDLE_SCHEMA, "dataset": "VidHOI", "split": "train",
                "purpose": "head_development", "box_source": "oracle", "object_label_source": "oracle",
                "benchmark_evaluated": False, "training_run": False,
                "official_final_split_accessed": False, "final_unseen_accessed": False,
                "entrypoint": "scripts/run_vidhoi_head_study.py", "files": records, "total_bytes": total,
                "role_counts": {"train": {"videos": len(train_videos), "frames": len(train["frames"])},
                                "validation": {"videos": len(dev_videos), "frames": len(dev["frames"])}},
                "protocol_sha256": protocol_hash, "mapping_sha256": mapping_hash,
                "source_train_manifest_sha256": sha256_file(train_manifest),
                "source_dev_manifest_sha256": sha256_file(dev_manifest),
                "frame_remap": train_remap + dev_remap, "bundled_wheels": [wheel.name for wheel in wheels]}
    manifest_bytes = _json_bytes(manifest)
    validate_manifest(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    runtime_source = (project / "src/temporal_hoi/inference/hoi_kaggle.py").read_text(encoding="utf-8")
    notebook_data = notebook(runtime_source, digest)
    output.mkdir(parents=True, exist_ok=False)
    bundle_path = output / "vidhoi_m1_head_input.zip"
    with zipfile.ZipFile(bundle_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bundle_manifest.json", manifest_bytes)
        for name, value in sorted(files.items()):
            compression = zipfile.ZIP_STORED if name.startswith(("assets/clip_original", "wheelhouse/")) else zipfile.ZIP_DEFLATED
            if isinstance(value, bytes):
                archive.writestr(name, value, compress_type=compression)
            else:
                archive.write(value, name, compress_type=compression)
    notebook_path = output / "vidhoi_m1_head_kaggle.ipynb"
    notebook_path.write_bytes(_json_bytes(notebook_data))
    receipt = {"schema_version": "vidhoi_head_kaggle_build_v1", "status": "local_bundle_prepared",
               "bundle": bundle_path.name, "bundle_sha256": sha256_file(bundle_path),
               "bundle_bytes": bundle_path.stat().st_size, "manifest_sha256": digest,
               "notebook": notebook_path.name, "notebook_sha256": sha256_file(notebook_path),
               "notebook_code_cells": len(notebook_data["cells"]), "role_counts": manifest["role_counts"],
               "protocol_sha256": protocol_hash, "files": records, "modes": list(MODES), "seeds": list(SEEDS),
               "training_run": False, "model_inference_run": False, "benchmark_evaluated": False,
               "official_final_split_accessed": False, "final_unseen_accessed": False, "uploaded": False}
    (output / "build_receipt.json").write_bytes(_json_bytes(receipt))
    (output / "HUONG_DAN.md").write_text(instructions(receipt), encoding="utf-8")
    return receipt


def validate_head_export(run: Path, input_root: Path, *, allow_fixture: bool = False) -> dict:
    """Validate all stages and actual nine-head state changes before completion."""
    import hashlib
    import math

    import numpy as np
    import torch
    from safetensors.torch import load_file

    study, input_root = Path(run) / "study", Path(input_root)

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def inventory(directory, records):
        if not isinstance(records, list) or not records:
            raise RuntimeError("Missing head study output inventory")
        names = set()
        for record in records:
            name = record.get("path")
            safe_member(name)
            path = directory / name
            if (name in names or name == "receipt.json" or path.is_symlink()
                    or not path.resolve().is_relative_to(directory.resolve()) or not path.is_file()
                    or path.stat().st_size != record.get("bytes")
                    or sha256_file(path) != record.get("sha256")):
                raise RuntimeError(f"Head study output inventory mismatch: {name}")
            names.add(name)
        actual = set()
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise RuntimeError("Head study output symlinks are forbidden")
            if path.is_file():
                actual.add(path.relative_to(directory).as_posix())
        if actual != names | {"receipt.json"}:
            raise RuntimeError("Head study files differ from the complete inventory")

    outer = read(study / "receipt.json")
    if (outer.get("schema_version") != "vidhoi_head_study_run_v1"
            or outer.get("status") != "completed" or outer.get("training_run") is not True
            or outer.get("model_training_run") is not True
            or outer.get("official_final_split_accessed") is not False
            or outer.get("final_unseen_accessed") is not False
            or outer.get("full_benchmark_reproduction_claim") is not False
            or outer.get("experiment_heads") != ["object", "union", "fusion"]
            or outer.get("seeds") != [42, 43, 44]):
        raise RuntimeError("Incomplete or mismatched head development run receipt")
    inventory(study, outer.get("files"))
    config = read(Path(run) / "run_config.json")
    packaged = read(input_root / "assets/run_config.json")
    expected_hashes = {}
    for name in ("train_manifest", "dev_manifest", "protocol", "mapping", "checkpoint"):
        safe_member(packaged[name])
        asset = input_root / "assets" / packaged[name]
        if asset.is_symlink() or not asset.resolve().is_relative_to(input_root.resolve()):
            raise RuntimeError("Unsafe head study input identity")
        digest = sha256_file(asset)
        expected_hashes[name] = digest
        if (packaged.get(f"{name}_sha256") != digest or config.get(f"{name}_sha256") != digest
                or outer.get("input_hashes", {}).get(name) != digest
                or Path(config[name]).resolve() != asset.resolve()):
            raise RuntimeError(f"Head study input identity mismatch: {name}")
    stages = outer.get("stage_receipts")
    if not isinstance(stages, dict) or set(stages) != {"cache", "training", "evaluation"}:
        raise RuntimeError("Missing cache/training/evaluation receipts")
    stage_data = {}
    for stage, record in stages.items():
        expected_path = f"{stage}/receipt.json"
        if record.get("path") != expected_path or sha256_file(study / expected_path) != record.get("sha256"):
            raise RuntimeError(f"Stage receipt identity mismatch: {stage}")
        stage_data[stage] = read(study / expected_path)
        if stage_data[stage].get("status") != "completed":
            raise RuntimeError(f"Incomplete stage receipt: {stage}")
    cache, trained, evaluated = (stage_data[name] for name in ("cache", "training", "evaluation"))
    if (evaluated.get("schema_version") != "vidhoi_head_evaluation_audit_v1"
            or evaluated.get("selected_state_replayed") != 9
            or evaluated.get("frozen_baselines_replayed") != 3
            or evaluated.get("input_hashes") != expected_hashes
            or evaluated.get("cache_receipt_sha256") != stages["cache"]["sha256"]
            or evaluated.get("training_receipt_sha256") != stages["training"]["sha256"]
            or evaluated.get("official_final_split_accessed") is not False
            or evaluated.get("final_unseen_accessed") is not False
            or evaluated.get("backbone_inference_run") is not False
            or evaluated.get("model_training_run") is not False):
        raise RuntimeError("Incomplete selected-state development evaluation audit")
    inventory(study / "evaluation", evaluated.get("files"))
    fixture = cache.get("level") == "synthetic_encoder_fixture"
    if fixture and (not allow_fixture or cache.get("encoder", {}).get("test_fixture") is not True):
        raise RuntimeError("Fixture cache cannot complete a real head development notebook")
    expected_level = "internal_oracle_head_development_fixture" if fixture else "internal_oracle_head_development"
    if (outer.get("level") != expected_level
            or cache.get("schema_version") != "vidhoi_head_cache_receipt_v1"
            or cache.get("level") != ("synthetic_encoder_fixture" if fixture else "vidhoi_head_frozen_clip_feature_cache")
            or cache.get("model_inference_run") is not (not fixture)
            or cache.get("model_training_run") is not False
            or cache.get("encoder", {}).get("test_fixture") is not fixture
            or cache.get("official_final_split_accessed") is not False
            or cache.get("final_unseen_accessed") is not False
            or cache.get("source_split") != "train" or cache.get("input_hashes") != expected_hashes
            or cache.get("text_cache", {}).get("final_unseen_rows_zero") is not True
            or cache.get("text_cache", {}).get("final_unseen_encoded") is not False):
        raise RuntimeError("Invalid frozen encoder cache evidence")
    inventory(study / "cache", cache.get("files"))
    with np.load(study / "cache/text_features.npz", allow_pickle=False) as text_archive:
        texts = text_archive["text_features"]
        allowed = text_archive["candidate_mask"]
    mapping = read(input_root / "assets" / packaged["mapping"])
    shape = (len(mapping["objects"]), len(mapping["predicates"]))
    protocol = read(input_root / "assets" / packaged["protocol"])
    expected_allowed = np.ones(shape, dtype=bool)
    for subject, predicate, obj in protocol["final_unseen_triplets"]:
        if subject != 0 or not 0 <= predicate < shape[1] or not 0 <= obj < shape[0]:
            raise RuntimeError("Invalid final composition in the locked protocol")
        expected_allowed[obj, predicate] = False
    if (texts.dtype != np.float32 or texts.ndim != 3 or texts.shape[:2] != shape
            or allowed.dtype != np.bool_ or not np.array_equal(allowed, expected_allowed)
            or not np.isfinite(texts).all() or np.any(texts[~allowed] != 0)
            or not np.allclose(np.linalg.norm(texts[allowed], axis=1), 1.0, atol=1e-5)):
        raise RuntimeError("Text cache or excluded-composition mask differs from the locked protocol")
    text_hash = hashlib.sha256(texts.tobytes()).hexdigest()
    mask_hash = hashlib.sha256(allowed.tobytes()).hexdigest()
    roles = {}
    fingerprints = {}
    keys = ("object_features", "union_features", "object_class", "positive_mask",
            "frame_index", "subject_track_id", "object_track_id")
    for role, manifest_name in (("train", "train_manifest"), ("validation", "dev_manifest")):
        source = read(input_root / "assets" / packaged[manifest_name])
        pairs = sum(sum(entity["class_id"] == 0 for entity in frame["entities"])
                    * max(0, len(frame["entities"]) - 1) for frame in source["frames"])
        role_count = {"num_frames": len(source["frames"]), "num_pairs": pairs}
        if cache.get("roles", {}).get(role) != role_count:
            raise RuntimeError(f"Cache frame/pair coverage mismatch: {role}")
        role_receipt = read(study / "cache" / role / "receipt.json")
        if (role_receipt.get("schema_version") != "vidhoi_head_role_cache_v1"
                or role_receipt.get("role") != role or role_receipt.get("status") != "completed"
                or any(role_receipt.get(name) != count for name, count in role_count.items())
                or role_receipt.get("protocol_sha256") != expected_hashes["protocol"]
                or role_receipt.get("config_sha256") != cache.get("config_sha256")):
            raise RuntimeError(f"Role cache receipt mismatch: {role}")
        digest = hashlib.sha256()
        with np.load(study / "cache" / role / "arrays.npz", allow_pickle=False) as arrays:
            for name in keys:
                array = arrays[name]
                digest.update(name.encode())
                digest.update(str(array.dtype).encode())
                digest.update(json.dumps(array.shape).encode())
                digest.update(array.tobytes(order="C"))
            if arrays["object_features"].shape[0] != pairs or arrays["union_features"].shape[0] != pairs:
                raise RuntimeError(f"Cache feature row count mismatch: {role}")
            classes, positives = arrays["object_class"], arrays["positive_mask"]
            if (classes.dtype.kind not in "iu" or classes.shape != (pairs,)
                    or np.any(classes < 0) or np.any(classes >= shape[0])
                    or positives.dtype != np.bool_ or positives.shape != (pairs, shape[1])
                    or np.any(positives & ~allowed[classes])):
                raise RuntimeError(f"Cache positive labels include excluded compositions: {role}")
            for name in ("object_features", "union_features"):
                values = arrays[name]
                if (values.dtype != np.float32 or values.shape != (pairs, texts.shape[-1])
                        or not np.isfinite(values).all()
                        or not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5)):
                    raise RuntimeError(f"Invalid normalized frozen features: {role}/{name}")
        roles[role] = role_count
        fingerprints[role] = digest.hexdigest()
    if (trained.get("schema_version") != "vidhoi_head_study_result_v1"
            or trained.get("level") != ("synthetic_feature_head_study" if fixture else
                                        "oracle_frozen_feature_development_head_study")
            or trained.get("training_run") is not True or trained.get("model_training_run") is not True
            or trained.get("backbone_training_run") is not False
            or trained.get("official_final_split_accessed") is not False
            or trained.get("final_unseen_accessed") is not False
            or trained.get("full_benchmark_reproduction_claim") is not False
            or trained.get("modes") != ["object", "union", "fusion"]
            or trained.get("seeds") != [42, 43, 44] or trained.get("selected_best_seed") is not None
            or trained.get("cache_inputs_unchanged") is not True
            or trained.get("text_features_sha256") != text_hash
            or trained.get("candidate_mask_sha256") != mask_hash
            or trained.get("train_cache_fingerprint") != fingerprints["train"]
            or trained.get("dev_cache_fingerprint") != fingerprints["validation"]):
        raise RuntimeError("Invalid controlled head training evidence")
    inventory(study / "training", trained.get("files"))
    if set(trained.get("frozen_baselines", {})) != {"object", "union", "fusion"}:
        raise RuntimeError("Missing three frozen baselines")
    for mode in ("object", "union", "fusion"):
        frozen = read(study / "training/frozen" / mode / "receipt.json")
        if (frozen.get("status") != "completed" or frozen.get("mode") != mode
                or frozen.get("model_training_run") is not False
                or frozen.get("num_frames") != roles["validation"]["num_frames"]
                or frozen.get("num_proposal_pairs") != roles["validation"]["num_pairs"]):
            raise RuntimeError(f"Frozen baseline evidence mismatch: {mode}")
    runs = trained.get("runs", [])
    expected_runs = {(mode, seed) for mode in ("object", "union", "fusion") for seed in (42, 43, 44)}
    if len(runs) != 9 or {(row.get("mode"), row.get("seed")) for row in runs} != expected_runs:
        raise RuntimeError("Need all three heads and all three seeds")
    capacities = set()
    for row in runs:
        mode, seed = row["mode"], row["seed"]
        relative = f"runs/{mode}/seed_{seed}"
        if row.get("directory") != relative:
            raise RuntimeError("Unexpected head run directory")
        directory = study / "training" / relative
        metadata = read(directory / "metadata.json")
        child = read(directory / "receipt.json")
        history = read(directory / "history.json")
        for name in ("mode", "seed", "selected_epoch", "last_epoch", "optimizer_steps",
                     "parameter_count", "checkpoints", "train_cache_fingerprint", "dev_cache_fingerprint"):
            if metadata.get(name) != row.get(name) or child.get(name) != row.get(name):
                raise RuntimeError(f"Head metadata identity mismatch: {relative}/{name}")
        if (child.get("status") != "completed" or row.get("optimizer_steps", 0) <= 0
                or row.get("nonzero_gradient_observed") is not True
                or not math.isfinite(row.get("max_gradient_norm", float("nan")))
                or row["max_gradient_norm"] <= 0 or row.get("last_parameter_delta_l2", 0) <= 0
                or row.get("last_embedding_delta_l2", 0) <= 0
                or row.get("train_cache_fingerprint") != fingerprints["train"]
                or row.get("dev_cache_fingerprint") != fingerprints["validation"]
                or row.get("train_pairs") != roles["train"]["num_pairs"]
                or row.get("dev_pairs") != roles["validation"]["num_pairs"]
                or not isinstance(history, list) or len(history) < 2
                or history[0].get("epoch") != 0 or history[-1].get("epoch") != row.get("last_epoch")
                or history[-1].get("optimizer_steps") != row.get("optimizer_steps")
                or row.get("selected_epoch") not in [entry.get("epoch") for entry in history]):
            raise RuntimeError(f"Missing real optimization/history evidence: {relative}")
        capacities.add(row["parameter_count"])
        states = {}
        for name, filename in (("initial", "initial_head.safetensors"),
                               ("selected", "selected_head.safetensors"), ("last", "last_head.safetensors")):
            record = row.get("checkpoints", {}).get(name, {})
            path = directory / filename
            if (record.get("path") != filename or path.stat().st_size != record.get("bytes")
                    or sha256_file(path) != record.get("sha256")):
                raise RuntimeError(f"Head checkpoint identity mismatch: {relative}/{name}")
            states[name] = load_file(str(path), device="cpu")
        initial = states["initial"]
        if any(set(state) != set(initial) for state in states.values()):
            raise RuntimeError("Head state keys differ")
        if any(not torch.isfinite(value).all() or value.shape != initial[key].shape
               for state in states.values() for key, value in state.items()):
            raise RuntimeError("Invalid head state tensors")
        delta = math.sqrt(sum(float((states["last"][key].double() - value.double()).square().sum())
                              for key, value in initial.items()))
        if delta <= 0 or not math.isclose(delta, row["last_parameter_delta_l2"], rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError("Head last-state update does not match optimization evidence")
        if row["selected_epoch"] == 0 and any(not torch.equal(states["selected"][key], value)
                                               for key, value in initial.items()):
            raise RuntimeError("Selected epoch zero is not the initial head state")
    if len(capacities) != 1 or next(iter(capacities)) <= 0:
        raise RuntimeError("Head parameter capacity is not equal across controls")
    return outer

def notebook(runtime_source: str, manifest_sha256: str) -> dict:
    """Seven noninteractive cells: setup, deps, preflight, cache, train, eval, export."""
    result = _base_module()._notebook(runtime_source, manifest_sha256)
    first = result["cells"][0]["source"]
    replacements = {
        'BASE = WORK / "vidhoi_static_probe"': 'BASE = WORK / "vidhoi_m1_head_development"',
        'RUN = BASE / ("oracle_train_" + run_id)': 'RUN = BASE / ("head_development_" + run_id)',
        "CELL_1_STARTED — VidHOI train-only oracle probe": "CELL_1_STARTED — VidHOI frozen-feature head development",
        "INPUT_VERIFIED. Small train probe; no training, final-val or M2 retrieval.":
            "INPUT_VERIFIED. Upstream-train development only; final evaluation stays closed.",
    }
    for old, new in replacements.items():
        if first.count(old) != 1:
            raise ValueError("Base bootstrap changed; review head-study notebook before packaging")
        first = first.replace(old, new, 1)
    result["cells"][0]["source"] = first
    second = result["cells"][1]["source"] + "\n" + result["cells"][2]["source"].replace(
        "CELL_3_STARTED — environment and fixed probe config", "ENVIRONMENT_CHECK — head study config")
    cells = result["cells"][:1]

    def add(source):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {}, "source": source,
                      "execution_count": None, "outputs": []})

    add(second)
    for index, stage in enumerate(("preflight", "cache", "train", "evaluate"), start=3):
        add(f'''
            print("CELL_{index}_STARTED — head study {stage}", flush=True)
            try:
                verify_installed_input(ROOT, PINNED_MANIFEST)
                REC.command("head_{stage}", [sys.executable, "-u",
                            str(ROOT / "scripts/run_vidhoi_head_study.py"),
                            "--config", str(CONFIG), "--output-dir", str(RUN / "study"),
                            "--stage", "{stage}"], timeout_seconds=7200)
            except BaseException as error:
                REC.event("head_{stage}_failed", error=str(error))
                REC.snapshot()
                raise
        ''')
    validator = inspect.getsource(validate_head_export)
    final = '''
print("CELL_7_STARTED — verify training/evaluation evidence and download", flush=True)
try:
    head_receipt = validate_head_export(RUN, ROOT)
except BaseException as error:
    REC.event("export_incomplete", error=str(error))
    REC.snapshot()
    raise
REC.event("completed", level=head_receipt["level"], training_run=True,
          official_final_split_accessed=False, final_unseen_accessed=False,
          full_benchmark_evaluated=False,
          limits="Internal train/development oracle head study. No independent final benchmark or M2 improvement claim.")
RESULT_ZIP = REC.snapshot()
from IPython.display import FileLink, display
print("Download from this link or the Kaggle Output files panel:", flush=True)
display(FileLink(str(RESULT_ZIP)))
'''
    add(validator + "\n\n" + final)
    result["cells"] = cells
    if len(cells) != 7:
        raise ValueError("The head development notebook must have seven cells")
    return result


def instructions(receipt):
    return f'''# Học head M1 trên Kaggle

1. Import `{receipt["notebook"]}`, đặt tên notebook **VidHOI M1 Head Development**.
2. Tạo Dataset riêng tư **VidHOI M1 Head Development Input** chứa
   `{receipt["bundle"]}`; đợi xử lý hoàn tất, rồi **Add Input** vào notebook.
3. Internet Off, GPU nếu có. **Save Version → Save & Run All**; chạy tự động đủ 7 cell.
4. Tải `head_development_<thời gian>_<mã>.zip` ở cell 7 hoặc mục Output.

Input train: {receipt["role_counts"]["train"]["videos"]} video,
{receipt["role_counts"]["train"]["frames"]} frame. Development:
{receipt["role_counts"]["validation"]["videos"]} video,
{receipt["role_counts"]["validation"]["frames"]} frame. Cả hai lấy từ upstream train,
khóa role/video/protocol/hash; không dùng official final-val hoặc final-unseen.

Notebook tự kiểm input → cache frozen CLIP → học object/union/fusion với seeds
42/43/44 → đánh giá development → xuất artifact. Encoder CLIP giữ nguyên;
chỉ head residual nhỏ được học. Nhãn không biết không tự thành negative;
loss và feature-anchor lấy đúng config/protocol đã khóa. Epoch 0 được giữ làm
mốc và có thể được chọn nếu học không cải thiện development.

Giữ mọi seed/phương pháp, frozen baselines, cache/checkpoint/history/metric/
receipt và kết quả âm. Development dùng chọn epoch/config không phải final
evaluation độc lập; không gọi mAP này là benchmark VidHOI đầy đủ, zero-shot
cuối hoặc cải thiện M2. Chưa có kết quả trước khi notebook thật chạy xong.

Gói có source, frame/nhãn, original CLIP weights và wheels CPython3.11/12/13
Linux x86_64. Cài offline vào working, giữ Torch/torchvision/NumPy/Pillow Kaggle.
Không tự tải data/weights, không tự publish. ZIP có thể đổi tên hoặc giải nén;
notebook nhận đúng phiên bản bằng manifest, không cần sửa tên/hash.

Nếu không thấy input, kiểm Add Input và quá trình xử lý Dataset. Nếu lỗi,
giữ ZIP logs đã xuất; không bỏ sample, đổi nhãn hoặc thay protocol để chạy tiếp.
Mỗi Run All tạo kết quả mới. Manifest SHA256: `{receipt["manifest_sha256"]}`.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-sha256", required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--data-provenance", type=Path, action="append", default=[])
    parser.add_argument("--model-license", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.train_manifest, args.dev_manifest, args.protocol, args.mapping,
        args.clip_checkpoint, args.clip_sha256, args.output,
        checkpoint_provenance=args.checkpoint_provenance, wheelhouse=args.wheelhouse,
        study_config=args.study_config, data_provenance=args.data_provenance,
        model_license=args.model_license)
    print(json.dumps({key: receipt[key] for key in ("status", "bundle", "bundle_bytes",
                     "role_counts", "modes", "seeds", "training_run", "uploaded")}))


if __name__ == "__main__":
    main()

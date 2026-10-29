"""Authenticated head-study packaging and exports; feature fixtures are explicit."""

import ast
import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from safetensors.torch import load_file

from temporal_hoi.data.vidhoi import ClassMapping, HOIPair
from temporal_hoi.inference import hoi_kaggle as runtime
from temporal_hoi.inference.hoi_static import OracleEntity, OracleFrame
from temporal_hoi.models.hoi_head import HOIResidualHead
from temporal_hoi.training.hoi_head_study import run_head_study

PROJECT = Path(__file__).resolve().parents[1]


def packager():
    spec = importlib.util.spec_from_file_location(
        "_head_packager_test", PROJECT / "scripts/prepare_vidhoi_head_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def inventory(directory):
    return [{"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
             "sha256": runtime.sha256_file(path)}
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "receipt.json"]


@pytest.fixture
def assets(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    mapping = data / "mapping.json"
    write(mapping, {"objects": {"0": "person", "1": "cup"},
                    "predicates": {"0": "hold", "1": "open"}})
    protocol = data / "protocol.json"
    write(protocol, {"schema_version": "vidhoi_head_protocol_v1",
                     "train_video_ids": ["fixture/train"], "dev_video_ids": ["fixture/dev"],
                     "seen_triplets": [[0, 0, 1]], "pseudo_unseen_triplets": [],
                     "final_unseen_triplets": [[0, 1, 0]], "excluded_final_video_ids": [],
                     "video_positive_triplets": {"fixture/train": [[0, 0, 1]],
                                                "fixture/dev": [[0, 0, 1]]},
                     "label_policy": "positive_only_unknown"})
    paths = {}
    for role, name in (("train", "train"), ("validation", "dev")):
        image = data / f"{role}.png"
        Image.new("RGB", (12, 8), (255, 0, 0)).save(image)
        frame = {"video_id": f"fixture/{name}", "source_frame_id": 1, "frame_id": 0,
                 "timestamp_s": 0.0, "image_path": image.name,
                 "image_sha256": runtime.sha256_file(image), "width": 12, "height": 8,
                 "entities": [{"track_id": "p", "class_id": 0, "bbox_xyxy": [0, 0, 5, 8]},
                              {"track_id": "c", "class_id": 1, "bbox_xyxy": [6, 0, 12, 8]}],
                 "ground_truth": [{"subject_track_id": "p", "object_track_id": "c",
                                   "predicate_ids": [0]}]}
        manifest = data / f"{name}_manifest.json"
        write(manifest, {"schema_version": "vidhoi_head_frames_v1", "upstream_split": "train",
                         "role": role, "box_source": "oracle", "object_label_source": "oracle",
                         "protocol_sha256": runtime.sha256_file(protocol),
                         "mapping_sha256": runtime.sha256_file(mapping), "frames": [frame]})
        paths[f"{name}_manifest"] = manifest
    checkpoint = data / "explicit_fixture.safetensors"
    checkpoint.write_bytes(b"Explicit packaging fixture only; never loaded as encoder weights")
    provenance = data / "checkpoint.json"
    write(provenance, {"test_fixture": True, "checkpoint_sha256": runtime.sha256_file(checkpoint)})
    config = data / "study.json"
    write(config, {"schema_version": "vidhoi_head_study_config_v1",
                   "study": {"test_fixture": True, "max_epochs": 1}})
    wheels = data / "wheelhouse"
    wheels.mkdir()
    (wheels / "requirements-offline.txt").write_text("open-clip-torch==3.3.0\n")
    (wheels / "open_clip_torch-3.3.0-py3-none-any.whl").write_bytes(
        b"Explicit packaging-only wheel fixture, never installed")
    source = tmp_path / "source"
    for name in packager().SOURCES:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (PROJECT / name).is_file():
            shutil.copy2(PROJECT / name, destination)
        else:
            # Packaging guards are independent of modules another agent is still implementing.
            destination.write_text('"""Explicit source packaging fixture, never executed."""\n')
    return {**paths, "protocol": protocol, "mapping": mapping,
            "clip_checkpoint": checkpoint, "clip_sha256": runtime.sha256_file(checkpoint),
            "checkpoint_provenance": provenance, "study_config": config,
            "wheelhouse": wheels, "project_root": source}


@pytest.fixture
def package(assets, tmp_path):
    output = tmp_path / "package"
    receipt = packager().prepare(**assets, output=output)
    return output, receipt


def test_package_authenticated_roles_sources_and_seven_cells(package, tmp_path):
    output, receipt = package
    assert receipt["bundle"] == "vidhoi_m1_head_input.zip"
    assert receipt["notebook"] == "vidhoi_m1_head_kaggle.ipynb"
    assert receipt["role_counts"] == {"train": {"videos": 1, "frames": 1},
                                      "validation": {"videos": 1, "frames": 1}}
    assert not receipt["uploaded"] and not receipt["training_run"]
    assert not receipt["model_inference_run"]
    with zipfile.ZipFile(output / receipt["bundle"]) as archive:
        manifest = runtime.validate_manifest(archive.read(runtime.MANIFEST_NAME))
        assert manifest["schema_version"] == "vidhoi_frozen_development_bundle_v1"
        assert manifest["entrypoint"] == "scripts/run_vidhoi_head_study.py"
        assert set(packager().SOURCES).issubset(manifest["files"])
        assert not any(name.startswith(("reports/", ".agents/", ".git/", "noiquy/"))
                       or name.lower().endswith(("agents.md", "rules.md", "rubric.md"))
                       for name in manifest["files"])
    target = tmp_path / "installed"
    runtime.install_input(output, target, receipt["manifest_sha256"])
    config_path = runtime.prepare_run_config(target, tmp_path / "run/config.json", device="cpu")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for name in ("train_manifest", "dev_manifest", "protocol", "mapping", "checkpoint"):
        assert Path(config[name]).is_absolute()
        assert runtime.sha256_file(config[name]) == config[f"{name}_sha256"]
    notebook = json.loads((output / receipt["notebook"]).read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 7
    for cell in notebook["cells"]:
        ast.parse(cell["source"])
    assert 'BASE = WORK / "vidhoi_m1_head_development"' in notebook["cells"][0]["source"]
    assert '"head_development_" + run_id' in notebook["cells"][0]["source"]
    assert "PINNED_MANIFEST[:12]" in notebook["cells"][1]["source"]
    for index, stage in enumerate(("preflight", "cache", "train", "evaluate"), 2):
        assert f'"--stage", "{stage}"' in notebook["cells"][index]["source"]
    assert "validate_head_export(RUN, ROOT)" in notebook["cells"][6]["source"]
    assert not notebook["metadata"]["kaggle"]["isInternetEnabled"]


@pytest.mark.parametrize("change", ["role", "mapping_hash", "protocol_hash", "overlap", "image"])
def test_wrong_roles_binding_overlap_or_changed_image_rejected(assets, tmp_path, change):
    path = assets["dev_manifest"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if change == "role":
        value["role"] = "official_val"
    elif change == "mapping_hash":
        value["mapping_sha256"] = "0" * 64
    elif change == "protocol_hash":
        value["protocol_sha256"] = "0" * 64
    elif change == "overlap":
        value["frames"][0]["video_id"] = "fixture/train"
    else:
        (path.parent / value["frames"][0]["image_path"]).write_bytes(b"changed")
    write(path, value)
    with pytest.raises(ValueError):
        packager().prepare(**assets, output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_packager_preserves_existing_output(assets, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user.txt").write_text("keep this")
    with pytest.raises(FileExistsError):
        packager().prepare(**assets, output=output)
    assert (output / "user.txt").read_text() == "keep this"


def test_changed_source_changes_manifest_and_pin(assets, tmp_path):
    module = packager()
    first = module.prepare(**assets, output=tmp_path / "first")
    source = assets["project_root"] / "src/temporal_hoi/models/hoi_head.py"
    source.write_bytes(source.read_bytes() + b"\n# Explicit source hash test.\n")
    second = module.prepare(**assets, output=tmp_path / "second")
    assert first["manifest_sha256"] != second["manifest_sha256"]
    with pytest.raises(FileNotFoundError, match="pinned"):
        runtime.install_input(tmp_path / "second", tmp_path / "bad", first["manifest_sha256"])


def make_completed_fixture(package, tmp_path):
    """Run nine tiny tensor heads; no image encoder or benchmark inference is simulated."""
    output, build = package
    root = tmp_path / "installed"
    runtime.install_input(output, root, build["manifest_sha256"])
    run = tmp_path / "run"
    config_path = runtime.prepare_run_config(root, run / "run_config.json", device="cpu")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = {key: config[f"{key}_sha256"]
              for key in ("train_manifest", "dev_manifest", "protocol", "mapping", "checkpoint")}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    mapping = ClassMapping.from_json(config["mapping"])
    study = run / "study"
    cache = study / "cache"
    cache.mkdir(parents=True)
    texts = np.array([[[1., 0.], [0., 0.]], [[1., 0.], [0., 1.]]], dtype=np.float32)
    allowed = np.array([[True, False], [True, True]], dtype=bool)
    np.savez(cache / "text_features.npz", text_features=texts, candidate_mask=allowed)
    roles = {}
    for role, manifest_name in (("train", "train_manifest"), ("validation", "dev_manifest")):
        directory = cache / role
        directory.mkdir()
        source = json.loads(Path(config[manifest_name]).read_text(encoding="utf-8"))
        row = source["frames"][0]
        gt = HOIPair(0, 1, (0., 0., 5., 8.), (6., 0., 12., 8.), (1., 0.),
                     subject_track_id="p", object_track_id="c")
        frame = OracleFrame(row["video_id"], 1, 0, 0., root / "assets" / row["image_path"],
                            row["image_sha256"], 12, 8,
                            (OracleEntity("p", 0, gt.subject_box),
                             OracleEntity("c", 1, gt.object_box)), (gt,))
        arrays = {"object_features": np.array([[0., 1.]], dtype=np.float32),
                  "union_features": np.array([[0.6, 0.8]], dtype=np.float32),
                  "object_class": np.array([1], dtype=np.int64),
                  "positive_mask": np.array([[True, False]], dtype=bool),
                  "frame_index": np.array([0], dtype=np.int64),
                  "subject_track_id": np.array(["p"]), "object_track_id": np.array(["c"])}
        metadata = {"role": role, "num_frames": 1, "num_pairs": 1,
                    "test_fixture": True, "level": "synthetic_encoder_fixture",
                    "source_split": "train", "label_policy": "positive_only_unknown",
                    "final_unseen_accessed": False, "official_final_split_accessed": False,
                    "pseudo_unseen_in_train": False,
                    "protocol_sha256": hashes["protocol"], "config_sha256": config_sha,
                    "checkpoint_sha256": hashes["checkpoint"], "mapping_sha256": hashes["mapping"]}
        roles[role] = {**arrays, "mapping": mapping, "frames": [frame], "metadata": metadata}
        np.savez(directory / "arrays.npz", **arrays)
        write(directory / "frames.json", source)
        write(directory / "receipt.json", {"schema_version": "vidhoi_head_role_cache_v1",
                                           "status": "completed", **metadata})
    cache_receipt = {"schema_version": "vidhoi_head_cache_receipt_v1", "status": "completed",
                     "level": "synthetic_encoder_fixture", "source_split": "train",
                     "model_inference_run": False, "model_training_run": False,
                     "official_final_split_accessed": False, "final_unseen_accessed": False,
                     "encoder": {"test_fixture": True}, "input_hashes": hashes,
                     "config_sha256": config_sha,
                     "roles": {role: {"num_frames": 1, "num_pairs": 1} for role in roles},
                     "text_cache": {"final_unseen_rows_zero": True, "final_unseen_encoded": False},
                     "files": inventory(cache)}
    write(cache / "receipt.json", cache_receipt)
    run_head_study(roles["train"], roles["validation"], texts, allowed,
                   {"study": {"test_fixture": True, "max_epochs": 1}}, study / "training")
    spec = importlib.util.spec_from_file_location(
        "_head_replay_cli_fixture", PROJECT / "scripts/run_vidhoi_head_study.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    frozen, selected = [], []
    for mode in ("object", "union", "fusion"):
        frozen.append(runner.replay_one(study / "training/frozen" / mode, None,
                                       roles["validation"], texts, allowed, mode, "cpu"))
        for seed in (42, 43, 44):
            directory = study / "training/runs" / mode / f"seed_{seed}"
            head = HOIResidualHead(texts.shape[-1])
            head.load_state_dict(load_file(str(directory / "selected_head.safetensors")))
            selected.append(runner.replay_one(directory, head, roles["validation"],
                                              texts, allowed, mode, "cpu"))
    evaluation = study / "evaluation"
    evaluation.mkdir()
    write(evaluation / "replay.json", {"test_fixture": True,
                                       "frozen": frozen, "selected": selected})
    write(evaluation / "receipt.json", {
        "schema_version": "vidhoi_head_evaluation_audit_v1", "status": "completed",
        "selected_state_replayed": 9, "frozen_baselines_replayed": 3,
        "input_hashes": hashes, "cache_receipt_sha256": runtime.sha256_file(cache / "receipt.json"),
        "training_receipt_sha256": runtime.sha256_file(study / "training/receipt.json"),
        "official_final_split_accessed": False, "final_unseen_accessed": False,
        "backbone_inference_run": False, "model_training_run": False, "test_fixture": True,
        "files": inventory(evaluation)})
    write(study / "receipt.json", {
        "schema_version": "vidhoi_head_study_run_v1", "status": "completed",
        "level": "internal_oracle_head_development_fixture", "training_run": True,
        "model_training_run": True, "official_final_split_accessed": False,
        "final_unseen_accessed": False, "full_benchmark_reproduction_claim": False,
        "experiment_heads": ["object", "union", "fusion"], "seeds": [42, 43, 44],
        "input_hashes": hashes,
        "stage_receipts": {name: {"path": f"{name}/receipt.json",
                                  "sha256": runtime.sha256_file(study / name / "receipt.json")}
                           for name in ("cache", "training", "evaluation")},
        "files": inventory(study)})
    return run, root


@pytest.fixture
def completed(package, tmp_path):
    return make_completed_fixture(package, tmp_path)


def refresh_outer(run):
    study = run / "study"
    path = study / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["files"] = inventory(study)
    for stage, record in value["stage_receipts"].items():
        record["sha256"] = runtime.sha256_file(study / stage / "receipt.json")
    write(path, value)


def test_fixture_cannot_finish_real_notebook(completed):
    run, root = completed
    with pytest.raises(RuntimeError, match="Fixture cache"):
        packager().validate_head_export(run, root)
    result = packager().validate_head_export(run, root, allow_fixture=True)
    assert result["level"] == "internal_oracle_head_development_fixture"


def test_export_requires_all_outputs_and_evaluation_audit(completed):
    run, root = completed
    path = run / "study/evaluation/receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["selected_state_replayed"] = 8
    write(path, value)
    refresh_outer(run)
    with pytest.raises(RuntimeError, match="evaluation audit"):
        packager().validate_head_export(run, root, allow_fixture=True)


def test_export_rejects_modified_checkpoint(completed):
    run, root = completed
    path = run / "study/training/runs/object/seed_42/selected_head.safetensors"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        packager().validate_head_export(run, root, allow_fixture=True)


def test_export_rejects_final_text_row_even_with_refreshed_file_hashes(completed):
    run, root = completed
    cache = run / "study/cache"
    path = cache / "text_features.npz"
    with np.load(path, allow_pickle=False) as arrays:
        texts, allowed = arrays["text_features"].copy(), arrays["candidate_mask"].copy()
    texts[0, 1] = [0., 1.]
    allowed[0, 1] = True
    np.savez(path, text_features=texts, candidate_mask=allowed)
    receipt_path = cache / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"] = inventory(cache)
    write(receipt_path, receipt)
    evaluation_path = run / "study/evaluation/receipt.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["cache_receipt_sha256"] = runtime.sha256_file(receipt_path)
    write(evaluation_path, evaluation)
    refresh_outer(run)
    with pytest.raises(RuntimeError, match="excluded-composition mask"):
        packager().validate_head_export(run, root, allow_fixture=True)


def test_export_requires_all_nine_runs_even_with_refreshed_file_hashes(completed):
    run, root = completed
    path = run / "study/training/receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["runs"].pop()
    write(path, value)
    evaluation_path = run / "study/evaluation/receipt.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["training_receipt_sha256"] = runtime.sha256_file(path)
    write(evaluation_path, evaluation)
    refresh_outer(run)
    with pytest.raises(RuntimeError, match="all three heads and all three seeds"):
        packager().validate_head_export(run, root, allow_fixture=True)


def test_export_cell_retains_failure_archive_without_completed_event(package, tmp_path, monkeypatch):
    output, receipt = package
    monkeypatch.setenv("VIDHOI_INPUT_ROOT", str(output))
    monkeypatch.setenv("VIDHOI_WORK_ROOT", str(tmp_path / "working"))
    notebook = json.loads((output / receipt["notebook"]).read_text(encoding="utf-8"))
    namespace = {"__name__": "__head_export_failure_fixture__"}
    exec(compile(notebook["cells"][0]["source"], "head_cell1", "exec"), namespace)
    with pytest.raises(FileNotFoundError):
        exec(compile(notebook["cells"][6]["source"], "head_cell7", "exec"), namespace)
    events = [json.loads(row) for row in
              (namespace["RUN"] / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["stage"] == "export_incomplete" for row in events)
    assert not any(row["stage"] == "completed" for row in events)
    assert list(namespace["BASE"].glob("head_development_*.zip"))

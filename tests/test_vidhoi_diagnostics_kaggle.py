"""Three-view packaging/export checks with an explicitly injected tiny encoder."""

import ast
import importlib.util
import json
import shutil
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from temporal_hoi.inference import hoi_kaggle as runtime
from temporal_hoi.inference.hoi_diagnostics import run_representation_diagnostics
from temporal_hoi.inference.hoi_static import preflight, resolve_config

PROJECT = Path(__file__).resolve().parents[1]


def packager():
    spec = importlib.util.spec_from_file_location(
        "_diagnostic_packager_test", PROJECT / "scripts/prepare_vidhoi_diagnostics_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExplicitEncoder:
    metadata = {"test_fixture": True, "name": "three_view_notebook_fixture"}

    def encode_images(self, images):
        return np.asarray([[np.asarray(image)[..., 0].mean(), np.asarray(image)[..., 2].mean()]
                           for image in images], dtype=np.float32)

    def encode_texts(self, texts):
        return np.asarray([[1., 0.] if "holding" in text else [0., 1.] for text in texts],
                          dtype=np.float32)


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "train_fixture"
    root.mkdir()
    mapping = root / "mapping.json"
    mapping.write_text(json.dumps({"objects": {"0": "person", "1": "cup"},
                                   "predicates": {"0": "hold", "1": "open"}}), encoding="utf-8")
    image = root / "frame.png"
    Image.new("RGB", (12, 8), (255, 0, 0)).save(image)
    frame = {"video_id": "fixture/train", "source_frame_id": 1, "frame_id": 24,
             "timestamp_s": 1.0, "image_path": image.name,
             "image_sha256": runtime.sha256_file(image), "width": 12, "height": 8,
             "entities": [{"track_id": "p", "class_id": 0, "bbox_xyxy": [0, 0, 5, 8]},
                          {"track_id": "c", "class_id": 1, "bbox_xyxy": [6, 0, 12, 8]}],
             "ground_truth": [{"subject_track_id": "p", "object_track_id": "c",
                               "predicate_ids": [0]}]}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "vidhoi_static_oracle_probe_v1",
        "split": "train", "purpose": "train_probe", "box_source": "oracle",
        "object_label_source": "oracle", "mapping_sha256": runtime.sha256_file(mapping),
        "frames": [frame]}), encoding="utf-8")
    checkpoint = root / "explicit_encoder_fixture.safetensors"
    checkpoint.write_bytes(b"Explicit encoder fixture only; never load this as model weights")
    provenance = root / "provenance.json"
    provenance.write_text(json.dumps({"test_fixture": True,
        "checkpoint_sha256": runtime.sha256_file(checkpoint)}), encoding="utf-8")
    return {"probe_manifest": manifest, "mapping": mapping, "clip_checkpoint": checkpoint,
            "clip_sha256": runtime.sha256_file(checkpoint), "checkpoint_provenance": provenance}


@pytest.fixture
def package(assets, tmp_path):
    output = tmp_path / "package"
    receipt = packager().prepare(**assets, output=output)
    return output, receipt


def install_package(package, tmp_path):
    output, receipt = package
    root = tmp_path / "installed"
    runtime.install_input(output, root, receipt["manifest_sha256"])
    return root


def test_packaged_source_entrypoint_and_six_automatic_cells(package):
    output, receipt = package
    assert receipt["bundle"] == "vidhoi_m1_diagnostics_input.zip"
    assert receipt["notebook"] == "vidhoi_m1_diagnostics_kaggle.ipynb"
    assert receipt["variants"] == ["full_frame", "object_only", "union"]
    assert not receipt["uploaded"] and not receipt["model_inference_run"]
    with zipfile.ZipFile(output / receipt["bundle"]) as archive:
        manifest = runtime.validate_manifest(archive.read(runtime.MANIFEST_NAME))
        assert manifest["entrypoint"] == "scripts/run_vidhoi_diagnostics.py"
        assert "src/temporal_hoi/inference/hoi_diagnostics.py" in manifest["files"]
        assert "src/temporal_hoi/inference/hoi_static.py" in manifest["files"]
        assert not any(name.startswith(("reports/", ".agents/", ".git/"))
                       or name.lower().endswith(("agents.md", "rules.md", "rubric.md"))
                       for name in manifest["files"])
    notebook = json.loads((output / receipt["notebook"]).read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 6
    for cell in notebook["cells"]:
        ast.parse(cell["source"])
    assert 'BASE = WORK / "vidhoi_m1_diagnostics"' in notebook["cells"][0]["source"]
    assert '"diagnostics_train_" + run_id' in notebook["cells"][0]["source"]
    assert "PINNED_MANIFEST[:12]" in notebook["cells"][1]["source"]
    assert "run_vidhoi_diagnostics.py" in notebook["cells"][4]["source"]
    assert not notebook["metadata"]["kaggle"]["isInternetEnabled"]


def test_static_packaging_defaults_remain_compatible(assets, tmp_path):
    base = packager()._base_module()
    static = base.prepare(**assets, output=tmp_path / "static")
    assert static["bundle"] == "vidhoi_oracle_train_input.zip"
    with zipfile.ZipFile(tmp_path / "static" / static["bundle"]) as archive:
        manifest = runtime.validate_manifest(archive.read(runtime.MANIFEST_NAME))
        assert manifest["entrypoint"] == "scripts/run_vidhoi_static.py"
        assert "src/temporal_hoi/inference/hoi_diagnostics.py" not in manifest["files"]


def test_changed_diagnostic_source_changes_manifest_and_is_authenticated(assets, tmp_path):
    module = packager()
    base = module._base_module()
    source = tmp_path / "source"
    for name in (*base.SOURCE_FILES, "src/temporal_hoi/inference/hoi_diagnostics.py",
                 "scripts/run_vidhoi_diagnostics.py"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, target)
    first = module.prepare(**assets, output=tmp_path / "first", project_root=source)
    changed = source / "src/temporal_hoi/inference/hoi_diagnostics.py"
    changed.write_bytes(changed.read_bytes() + b"\n# Explicit packaging source-hash fixture.\n")
    second = module.prepare(**assets, output=tmp_path / "second", project_root=source)
    assert first["manifest_sha256"] != second["manifest_sha256"]
    with zipfile.ZipFile(tmp_path / "second" / second["bundle"]) as archive:
        manifest = runtime.validate_manifest(archive.read(runtime.MANIFEST_NAME))
        assert manifest["files"][changed.relative_to(source).as_posix()]["sha256"] == runtime.sha256_file(changed)
    with pytest.raises(FileNotFoundError, match="pinned"):
        runtime.install_input(tmp_path / "second", tmp_path / "wrong_pin", first["manifest_sha256"])


@pytest.fixture
def completed(package, tmp_path):
    root = install_package(package, tmp_path)
    run = tmp_path / "run"
    config_path = runtime.prepare_run_config(root, run / "run_config.json", device="cpu")
    config = resolve_config(json.loads(config_path.read_text(encoding="utf-8")), config_path)
    run_representation_diagnostics(config, run / "diagnostics", encoder=ExplicitEncoder())
    return run, root


def test_export_requires_real_inference_unless_test_api_explicit(completed):
    run, root = completed
    with pytest.raises(RuntimeError, match="fixture receipt"):
        packager().validate_diagnostic_export(run, root)
    receipt = packager().validate_diagnostic_export(run, root, allow_fixture=True)
    assert receipt["level"] == "representation_diagnostic_encoder_fixture"
    assert not receipt["model_inference_run"] and not receipt["model_training_run"]
    assert receipt["variants"] == ["full_frame", "object_only", "union"]


def test_missing_one_variant_output_prevents_export(completed):
    run, root = completed
    (run / "diagnostics/object_only/metrics.json").unlink()
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        packager().validate_diagnostic_export(run, root, allow_fixture=True)


@pytest.mark.parametrize("field,value", [("num_frames", 2), ("num_proposal_pairs", 2),
    ("dropped_frames", 1), ("dropped_proposal_pairs", 1), ("official_final_split_accessed", True),
    ("zero_shot_benchmark_claim", True), ("variants", ["union"]), ("model_training_run", True)])
def test_mismatched_root_receipt_cannot_complete(completed, field, value):
    run, root = completed
    path = run / "diagnostics/receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt[field] = value
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incomplete or mismatched"):
        packager().validate_diagnostic_export(run, root, allow_fixture=True)


def test_wrong_variant_receipt_rejected_even_when_file_inventory_refreshed(completed):
    run, root = completed
    view = run / "diagnostics/union/receipt.json"
    details = json.loads(view.read_text(encoding="utf-8"))
    details["variant"] = "object_only"
    view.write_text(json.dumps(details), encoding="utf-8")
    root_path = run / "diagnostics/receipt.json"
    receipt = json.loads(root_path.read_text(encoding="utf-8"))
    record = next(row for row in receipt["files"] if row["path"] == "union/receipt.json")
    record.update(bytes=view.stat().st_size, sha256=runtime.sha256_file(view))
    root_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incomplete diagnostic variant"):
        packager().validate_diagnostic_export(run, root, allow_fixture=True)


def test_missing_receipt_export_cell_saves_failure_without_completion(package, tmp_path, monkeypatch):
    output, receipt = package
    monkeypatch.setenv("VIDHOI_INPUT_ROOT", str(output))
    monkeypatch.setenv("VIDHOI_WORK_ROOT", str(tmp_path / "working"))
    notebook = json.loads((output / receipt["notebook"]).read_text(encoding="utf-8"))
    namespace = {"__name__": "__diagnostic_failure_fixture__"}
    exec(compile(notebook["cells"][0]["source"], "cell1", "exec"), namespace)
    with pytest.raises(FileNotFoundError):
        exec(compile(notebook["cells"][5]["source"], "cell6", "exec"), namespace)
    events = [json.loads(line) for line in
              (namespace["RUN"] / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["stage"] == "export_incomplete" for row in events)
    assert not any(row["stage"] == "completed" for row in events)
    assert namespace["RUN"].with_suffix(".zip").is_file()


def test_six_cell_orchestration_with_explicit_encoder_fixture(package, tmp_path, monkeypatch):
    """Only this test injects the encoder and opts into fixture export validation."""
    output, receipt = package
    monkeypatch.setenv("VIDHOI_INPUT_ROOT", str(output))
    monkeypatch.setenv("VIDHOI_WORK_ROOT", str(tmp_path / "working"))
    display_module = types.ModuleType("IPython.display")
    displayed = []
    display_module.FileLink = str
    display_module.display = displayed.append
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    notebook = json.loads((output / receipt["notebook"]).read_text(encoding="utf-8"))
    namespace = {"__name__": "__diagnostic_notebook_cpu_fixture__"}
    prior_path, stages = sys.path.copy(), []
    try:
        exec(compile(notebook["cells"][0]["source"], "cell1", "exec"), namespace)
        namespace["dependency_plan"] = lambda *args, **kwargs: None
        namespace["environment_report"] = lambda: {"test_fixture": True, "cuda_available": False}
        recorder = namespace["REC"]

        def fixture_command(label, command, **kwargs):
            stages.append(label)
            config_path = Path(command[command.index("--config") + 1])
            config = resolve_config(json.loads(config_path.read_text(encoding="utf-8")), config_path)
            destination = Path(command[command.index("--output-dir") + 1])
            if "--preflight-only" in command:
                _, frames, _ = preflight(config, test_fixture=True)
                recorder.event("explicit_diagnostic_fixture_preflight", frames=len(frames))
            else:
                result = run_representation_diagnostics(config, destination, encoder=ExplicitEncoder())
                assert result["level"] == "representation_diagnostic_encoder_fixture"
                assert not result["model_inference_run"]
            recorder.snapshot()

        recorder.command = fixture_command
        for index, cell in enumerate(notebook["cells"][1:5], start=2):
            exec(compile(cell["source"], f"cell{index}", "exec"), namespace)
        # Production cell 6 never allows fixture completion. Execute its exact validator,
        # then opt in at this test's dispatch boundary without altering production source.
        tree = ast.parse(notebook["cells"][5]["source"])
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        rest = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "cell6_validator", "exec"), namespace)
        real_validator = namespace["validate_diagnostic_export"]
        namespace["validate_diagnostic_export"] = lambda run, root: real_validator(
            run, root, allow_fixture=True)
        exec(compile(ast.Module(body=rest, type_ignores=[]), "cell6_fixture_export", "exec"), namespace)
    finally:
        sys.path[:] = prior_path
    assert stages == ["diagnostic_preflight", "three_view_diagnostic"]
    assert namespace["DEVICE"] == "cpu"
    result = namespace["diagnostic_receipt"]
    assert result["variants"] == ["full_frame", "object_only", "union"]
    assert not result["model_inference_run"]
    assert Path(namespace["RESULT_ZIP"]).is_file() and displayed
    with zipfile.ZipFile(namespace["RESULT_ZIP"]) as archive:
        assert "diagnostics/object_only/known_positive_ranks.json" in archive.namelist()
        assert "diagnostics/role_symmetry.json" in archive.namelist()


@pytest.mark.parametrize("where", ["root_input_hash", "output_config", "frame_order_hash", "prompt_hash"])
def test_stale_identity_rejected_after_consistent_output_inventory(completed, where):
    run, root = completed
    receipt_path = run / "diagnostics/receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if where == "root_input_hash":
        receipt["input_hashes"]["mapping"] = "0" * 64
    elif where == "output_config":
        config_path = run / "diagnostics/run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["checkpoint_sha256"] = "0" * 64
        config_path.write_text(json.dumps(config), encoding="utf-8")
        record = next(row for row in receipt["files"] if row["path"] == "run_config.json")
        record.update(bytes=config_path.stat().st_size, sha256=runtime.sha256_file(config_path))
    elif where == "frame_order_hash":
        receipt["frame_ids_sha256"] = "0" * 64
    else:
        receipt["prompt_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity"):
        packager().validate_diagnostic_export(run, root, allow_fixture=True)

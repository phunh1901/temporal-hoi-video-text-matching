"""Authenticated Kaggle input and notebook orchestration; no benchmark assertions."""

import ast
import importlib.util
import json
import shutil
import stat
import sys
import types
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from temporal_hoi.inference import hoi_kaggle as runtime

PROJECT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location(
        "prepare_vidhoi_kaggle_test", PROJECT / "scripts/prepare_vidhoi_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def package(tmp_path):
    data = tmp_path / "probe"
    data.mkdir()
    mapping = data / "mapping.json"
    mapping.write_text(json.dumps({"objects": {"0": "person", "1": "cup"},
                                   "predicates": {"0": "hold", "1": "open"}}))
    image = data / "frame.png"
    Image.new("RGB", (12, 8), (255, 0, 0)).save(image)
    frame = {"video_id": "fixture/train", "source_frame_id": 1, "frame_id": 24,
             "timestamp_s": 1.0, "image_path": image.name,
             "image_sha256": runtime.sha256_file(image), "width": 12, "height": 8,
             "entities": [{"track_id": "p", "class_id": 0, "bbox_xyxy": [0, 0, 5, 8]},
                          {"track_id": "c", "class_id": 1, "bbox_xyxy": [6, 0, 12, 8]}],
             "ground_truth": [{"subject_track_id": "p", "object_track_id": "c",
                               "predicate_ids": [0]}]}
    manifest = data / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "vidhoi_static_oracle_probe_v1",
        "split": "train", "purpose": "train_probe", "box_source": "oracle",
        "object_label_source": "oracle", "mapping_sha256": runtime.sha256_file(mapping),
        "frames": [frame]}))
    checkpoint = data / "explicit_fixture.safetensors"
    checkpoint.write_bytes(b"Explicit injected test fixture; never loaded as a model")
    provenance = data / "provenance.json"
    provenance.write_text(json.dumps({"test_fixture": True,
                                      "checkpoint_sha256": runtime.sha256_file(checkpoint)}))
    output = tmp_path / "package"
    receipt = builder().prepare(manifest, mapping, checkpoint, runtime.sha256_file(checkpoint),
                                output, checkpoint_provenance=provenance)
    return output, receipt


def rewrite_zip(path, change=None, additions=()):
    with zipfile.ZipFile(path) as archive:
        rows = [(info, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w") as archive:
        for info, data in rows:
            archive.writestr(info, change(info.filename, data) if change else data)
        for name, data in additions:
            archive.writestr(name, data)


def test_renamed_zip_expanded_input_and_repeat(package, tmp_path):
    output, receipt = package
    inputs = tmp_path / "input"
    inputs.mkdir()
    renamed = inputs / "user_renamed.zip"
    shutil.copy2(output / receipt["bundle"], renamed)
    target = tmp_path / "dest"
    assert runtime.install_input(inputs, target, receipt["manifest_sha256"])["format"] == "zip"
    assert runtime.install_input(inputs, target, receipt["manifest_sha256"])["format"] == "zip"
    expanded = inputs / "expanded"
    with zipfile.ZipFile(renamed) as archive:
        for name in archive.namelist():
            runtime.safe_member(name)
            path = expanded / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
    assert runtime.install_input(inputs, target, receipt["manifest_sha256"])["format"] == "directory"
    runtime.verify_installed_input(target, receipt["manifest_sha256"])


def test_empty_input_has_actionable_diagnostic(tmp_path):
    inputs = tmp_path / "empty_input"
    inputs.mkdir()
    events = []
    destination = tmp_path / "dest"
    with pytest.raises(FileNotFoundError, match="No readable input files.*Add Input"):
        runtime.install_input(inputs, destination, "0" * 64,
                              progress=lambda stage, **values: events.append((stage, values)))
    assert not destination.exists()
    assert events[-1][0] == "input_scan_completed"
    assert events[-1][1]["visible_files"] == []
    assert events[-1][1]["scan_errors"] == []


def test_wrong_bundle_version_reports_expected_manifest(package, tmp_path):
    output, _ = package
    with pytest.raises(FileNotFoundError, match="Expected manifest SHA256: " + "0" * 64):
        runtime.install_input(output, tmp_path / "dest", "0" * 64)
    assert not (tmp_path / "dest").exists()


def test_input_walk_failure_is_not_reported_as_empty(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    inputs.mkdir()

    def unreadable_walk(root, *, followlinks, onerror):
        assert root == inputs and followlinks is False
        onerror(PermissionError("explicit test: input is unreadable"))
        return iter(())

    monkeypatch.setattr(runtime.os, "walk", unreadable_walk)
    with pytest.raises(OSError, match="Cannot read probe input.*unreadable"):
        runtime.install_input(inputs, tmp_path / "dest", "0" * 64)
    assert not (tmp_path / "dest").exists()


def test_tampered_payload_never_imported(package, tmp_path):
    output, receipt = package
    target = "src/temporal_hoi/inference/hoi_static.py"
    rewrite_zip(output / receipt["bundle"],
                lambda name, data: b"x" * len(data) if name == target else data)
    with pytest.raises(ValueError, match="hash/size mismatch"):
        runtime.install_input(output, tmp_path / "dest", receipt["manifest_sha256"])
    assert not (tmp_path / "dest" / target).exists()


@pytest.mark.parametrize("name", ["../bad", "/absolute", "C:/bad", "a\\b", "a/./b", "a//b", ".", ""])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        runtime.safe_member(name)


@pytest.mark.parametrize("extra", ["unexpected.py", "../escape.txt"])
def test_extra_or_traversal_zip_member_rejected(package, tmp_path, extra):
    output, receipt = package
    rewrite_zip(output / receipt["bundle"], additions=[(extra, b"untrusted")])
    with pytest.raises(ValueError, match="Unexpected or missing"):
        runtime.install_input(output, tmp_path / "dest", receipt["manifest_sha256"])
    assert not (tmp_path / "escape.txt").exists()


def test_symlink_zip_member_rejected(package, tmp_path):
    output, receipt = package
    bundle = output / receipt["bundle"]
    with zipfile.ZipFile(bundle) as archive:
        rows = [(info, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(bundle, "w") as archive:
        for info, data in rows:
            if info.filename == "assets/class_mapping.json":
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, data)
    with pytest.raises(ValueError, match="symlinks"):
        runtime.install_input(output, tmp_path / "dest", receipt["manifest_sha256"])


def test_notebook_six_cells_source_allowlist_and_no_benchmark_claim(package):
    output, receipt = package
    notebook = json.loads((output / receipt["notebook"]).read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 6
    assert sum(len(cell["source"].encode()) for cell in notebook["cells"]) < 1_000_000
    for cell in notebook["cells"]:
        ast.parse(cell["source"])
    assert notebook["cells"][0]["source"].startswith('print("CELL_1_STARTED')
    with zipfile.ZipFile(output / receipt["bundle"]) as archive:
        manifest = runtime.validate_manifest(archive.read(runtime.MANIFEST_NAME))
        assert set(archive.namelist()) == set(manifest["files"]) | {runtime.MANIFEST_NAME}
        assert not any(name.startswith(("reports/", ".agents/", ".git/"))
                       or name.lower().endswith(("agents.md", "rules.md", "rubric.md"))
                       for name in archive.namelist())
    assert not receipt["model_inference_run"] and not receipt["training_run"]
    assert not receipt["benchmark_evaluated"] and not receipt["uploaded"]


@pytest.mark.parametrize("line", ["torch==2.6.0", "torchvision==0.21.0", "numpy==2.0.0",
    "Pillow==11.0.0", "--extra-index-url https://bad.example", "timm @ https://bad.example/x.whl"])
def test_dependency_pins_forbid_core_replacement_and_options(line):
    with pytest.raises(ValueError):
        runtime.validate_requirements("open-clip-torch==3.3.0\n" + line)


def test_offline_plan_uses_target_and_resolver_not_all_wheels(tmp_path):
    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    (wheels / "requirements-offline.txt").write_text("open-clip-torch==3.3.0\nsafetensors==0.8.0\n")
    for name in ("open_clip_torch-3.3.0-py3-none-any.whl",
                 "safetensors-0.8.0-cp311-abi3-manylinux_2_17_x86_64.whl",
                 "safetensors-0.8.0-cp312-abi3-manylinux_2_17_x86_64.whl"):
        (wheels / name).write_bytes(b"planning-only wheel fixture")
    target = tmp_path / "local_deps"
    command = runtime.dependency_plan(tmp_path, target=target)
    assert "--no-index" in command and "--no-deps" in command
    assert command[command.index("--target") + 1] == str(target)
    assert command[command.index("--find-links") + 1] == str(wheels)
    assert "-r" in command and not any(part.endswith(".whl") for part in command)
    assert not target.exists()


def test_failed_command_keeps_downloadable_logs(tmp_path):
    recorder = runtime.RunRecorder(tmp_path / "result", tmp_path, heartbeat_seconds=.05)
    with pytest.raises(RuntimeError, match="exit 3"):
        recorder.command("explicit_failure", [sys.executable, "-u", "-c",
                         "print('visible before failure'); raise SystemExit(3)"])
    with zipfile.ZipFile(tmp_path / "result.zip") as archive:
        assert b"visible before failure" in archive.read("console.log")
        assert b"command_failed" in archive.read("events.jsonl")


def test_all_six_cells_with_explicit_cpu_encoder_fixture(package, tmp_path, monkeypatch):
    """Test-only injection exercises real probe engine, never real model or benchmark."""
    import numpy as np

    from temporal_hoi.inference.hoi_static import preflight, resolve_config, run_static_probe

    class TinyEncoder:
        metadata = {"test_fixture": True, "name": "explicit_notebook_cpu_fixture"}

        def encode_images(self, images):
            return np.asarray([[np.asarray(image)[..., 0].mean(),
                                np.asarray(image)[..., 2].mean()] for image in images], np.float32)

        def encode_texts(self, texts):
            return np.asarray([[1., 0.] if "holding" in text else [0., 1.] for text in texts],
                              np.float32)

    output, receipt = package
    monkeypatch.setenv("VIDHOI_INPUT_ROOT", str(output))
    monkeypatch.setenv("VIDHOI_WORK_ROOT", str(tmp_path / "working"))
    display_module = types.ModuleType("IPython.display")
    displayed = []
    display_module.FileLink = str
    display_module.display = displayed.append
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    notebook = json.loads((output / receipt["notebook"]).read_text(encoding="utf-8"))
    namespace = {"__name__": "__explicit_notebook_fixture__"}
    prior_path = sys.path.copy()
    stages = []
    try:
        exec(compile(notebook["cells"][0]["source"], "cell1", "exec"), namespace)
        namespace["dependency_plan"] = lambda *args, **kwargs: None
        namespace["environment_report"] = lambda: {
            "test_fixture": True, "cuda_available": False, "model_loaded": False}
        recorder = namespace["REC"]

        def fixture_command(label, command, **kwargs):
            stages.append(label)
            config_path = Path(command[command.index("--config") + 1])
            config = resolve_config(json.loads(config_path.read_text(encoding="utf-8")), config_path)
            destination = Path(command[command.index("--output-dir") + 1])
            if "--preflight-only" in command:
                _, frames, _ = preflight(config, test_fixture=True)
                recorder.event("explicit_fixture_preflight", frames=len(frames))
            else:
                result = run_static_probe(config, destination, encoder=TinyEncoder())
                assert result["level"] == "synthetic_encoder_fixture"
                assert not result["model_inference_run"]
            recorder.snapshot()

        recorder.command = fixture_command
        for index, cell in enumerate(notebook["cells"][1:], start=2):
            exec(compile(cell["source"], f"cell{index}", "exec"), namespace)
    finally:
        sys.path[:] = prior_path
    assert stages == ["oracle_preflight", "oracle_train_inference"]
    assert namespace["DEVICE"] == "cpu"
    result = json.loads((namespace["RUN"] / "oracle_probe/receipt.json").read_text(encoding="utf-8"))
    assert result["level"] == "synthetic_encoder_fixture"
    assert not result["model_inference_run"] and not result["model_training_run"]
    assert result["metrics"]["map"] == pytest.approx(1.)
    assert Path(namespace["RESULT_ZIP"]).is_file() and displayed


def test_export_cell_cannot_mark_completion_without_an_inference_receipt(
    package, tmp_path, monkeypatch,
):
    output, receipt = package
    monkeypatch.setenv("VIDHOI_INPUT_ROOT", str(output))
    monkeypatch.setenv("VIDHOI_WORK_ROOT", str(tmp_path / "working"))
    notebook = json.loads((output / receipt["notebook"]).read_text(encoding="utf-8"))
    namespace = {"__name__": "__explicit_export_failure_fixture__"}
    exec(compile(notebook["cells"][0]["source"], "cell1", "exec"), namespace)
    with pytest.raises(FileNotFoundError):
        exec(compile(notebook["cells"][5]["source"], "cell6", "exec"), namespace)
    events = [json.loads(line) for line in
              (namespace["RUN"] / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["stage"] == "export_incomplete" for event in events)
    assert not any(event["stage"] == "completed" for event in events)
    assert namespace["RUN"].with_suffix(".zip").is_file()

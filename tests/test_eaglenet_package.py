"""Transport checks for the private EagleNet notebook, without model inference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "eaglenet_packager", Path(__file__).resolve().parents[1] / "scripts/prepare_eaglenet_kaggle.py")
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def record(payload):
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def fixture_bundle(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    payloads = {"upstream/main_my.py": b"# reviewed source fixture\n",
                "weights/eaglenet.bin": b"fixture checkpoint; never inference",
                "weights/clip.bin": b"fixture CLIP",
                "metadata/test.csv": b"video_id,sentence\nvideo1,test\n",
                "metadata/train.csv": b"video_id\nvideo2\n",
                "metadata/annotations.json": b"{}\n",
                "videos/video1.mp4": b"test media fixture",
                "videos/video2.mp4": b"train media fixture"}
    inventory = {name: record(value) for name, value in payloads.items()}
    for name, value in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def asset(name):
        return {"path": name, **inventory[name]}

    manifest = {"schema": PACKAGE.SCHEMA, "fixture": True,
                "source": {"commit": PACKAGE.SOURCE_COMMIT, "path": "upstream",
                           "files": {"main_my.py": inventory["upstream/main_my.py"]}},
                "variant": "ViT-B/16", "checkpoint": asset("weights/eaglenet.bin"),
                "clip_checkpoint": asset("weights/clip.bin"),
                "metadata": {"test_csv": asset("metadata/test.csv"),
                             "train_csv": asset("metadata/train.csv"),
                             "annotations": asset("metadata/annotations.json")},
                "videos": {"root": "videos", "files": {
                    "video1": asset("videos/video1.mp4"),
                    "video2": asset("videos/video2.mp4")},
                           "test_ids": ["video1"], "probe_ids": ["video2"]},
                "config": {"max_frames": 12, "max_words": 32, "seed": 0},
                "inventory": inventory}
    (root / "asset_manifest.json").write_bytes(PACKAGE.json_bytes(manifest))
    runtime = tmp_path / "runtime.py"
    runtime.write_text('print("runtime fixture only")\n', encoding="utf-8")
    return root, manifest, runtime


def build_fixture(tmp_path):
    root, manifest, runtime = fixture_bundle(tmp_path)
    output = tmp_path / "package"
    receipt = PACKAGE.prepare(root, runtime, output, allow_fixture=True)
    return root, manifest, output, receipt


def test_fixture_never_claims_benchmark_or_real_handoff(tmp_path):
    root, _, runtime = fixture_bundle(tmp_path)
    with pytest.raises(ValueError, match="Fixture"):
        PACKAGE.prepare(root, runtime, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    receipt = PACKAGE.prepare(root, runtime, tmp_path / "fixture", allow_fixture=True)
    assert receipt["status"] == "fixture_only"
    assert receipt["benchmark_inference_completed"] is False
    notebook = json.loads((tmp_path / "fixture/eaglenet_eval_kaggle.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'if MANIFEST["fixture"]:' in source
    assert "benchmark execution is forbidden" in source


def test_roundtrip_and_pinned_notebook_guards(tmp_path):
    _, _, output, receipt = build_fixture(tmp_path)
    archive = output / "eaglenet_eval_input.bin"
    extracted = tmp_path / "expanded"
    manifest = PACKAGE.safe_extract(archive, extracted, receipt["asset_manifest_sha256"])
    assert "runtime/eaglenet_kaggle_runtime.py" in manifest["inventory"]
    assert PACKAGE.discover_input(output, receipt["asset_manifest_sha256"]) == ("zip", archive)
    assert PACKAGE.discover_input(extracted, receipt["asset_manifest_sha256"]) == (
        "expanded", extracted)
    notebook_bytes = (output / "eaglenet_eval_kaggle.ipynb").read_bytes()
    assert len(notebook_bytes) < 1_000_000
    notebook = json.loads(notebook_bytes)
    assert len(notebook["cells"]) == 9
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert code.index('os.environ["CUDA_VISIBLE_DEVICES"]') < code.index("def run_stage")
    assert 'run_stage("preflight", "cpu")' in code
    assert 'run_stage("probe", "cuda", 8)' in code
    assert 'run_stage("probe", "cuda", 64)' in code
    assert 'run_stage("evaluate", "cuda")' in code
    assert '--no-index' in code
    assert 'TORCH_FORCE_WEIGHTS_ONLY_LOAD' in code
    assert 'STATE["failure"] is not None' in code
    assert '"partial"' in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "a\\b",
                                    "a/./b", "a//b", "CON.txt", "a. "])
def test_unsafe_archive_names_rejected_before_extraction(tmp_path, name):
    _, _, output, receipt = build_fixture(tmp_path)
    archive = output / "eaglenet_eval_input.bin"
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr(name, "bad")
    target = tmp_path / "must_not_exist"
    with pytest.raises(ValueError):
        PACKAGE.safe_extract(archive, target, receipt["asset_manifest_sha256"])
    assert not target.exists()


def test_zip_symlink_rejected_before_extraction(tmp_path):
    _, _, output, receipt = build_fixture(tmp_path)
    archive = output / "eaglenet_eval_input.bin"
    with zipfile.ZipFile(archive, "a") as handle:
        entry = zipfile.ZipInfo("linked")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(entry, "outside")
    with pytest.raises(ValueError, match="ordinary"):
        PACKAGE.safe_extract(archive, tmp_path / "rejected", receipt["asset_manifest_sha256"])
    assert not (tmp_path / "rejected").exists()


def test_tampered_expanded_media_and_missing_checkpoint_rejected(tmp_path):
    root, manifest, _ = fixture_bundle(tmp_path)
    (root / "videos/video1.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size/hash"):
        PACKAGE.verify_inventory(root, manifest, allow_fixture=True)
    (root / "weights/eaglenet.bin").unlink()
    with pytest.raises(ValueError, match="coverage"):
        PACKAGE.verify_inventory(root, manifest, allow_fixture=True)


def test_pinned_manifest_and_duplicate_input_fail(tmp_path):
    _, _, output, receipt = build_fixture(tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        PACKAGE.safe_extract(output / "eaglenet_eval_input.bin", tmp_path / "badpin", "0" * 64)
    input_root = tmp_path / "inputs"
    PACKAGE.safe_extract(output / "eaglenet_eval_input.bin", input_root / "one", receipt["asset_manifest_sha256"])
    PACKAGE.safe_extract(output / "eaglenet_eval_input.bin", input_root / "two", receipt["asset_manifest_sha256"])
    with pytest.raises(ValueError, match="exactly one"):
        PACKAGE.discover_input(input_root, receipt["asset_manifest_sha256"])


def test_opaque_transport_ignores_checkpoint_bin_and_accepts_legacy_zip(tmp_path):
    _, _, output, receipt = build_fixture(tmp_path)
    unrelated = output / "msrvtt9k-vit_b_16.bin"
    unrelated.write_bytes(b"must not be interpreted as an input archive")
    kind, path = PACKAGE.discover_input(output, receipt["asset_manifest_sha256"])
    assert kind == "zip" and path.name == "eaglenet_eval_input.bin"
    legacy = output / "eaglenet_eval_input.zip"
    path.rename(legacy)
    assert PACKAGE.discover_input(output, receipt["asset_manifest_sha256"]) == ("zip", legacy)


def test_unknown_extra_file_and_existing_output_preserved(tmp_path):
    root, manifest, runtime = fixture_bundle(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "evidence"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        PACKAGE.prepare(root, runtime, output, allow_fixture=True)
    assert marker.read_text() == "keep"
    (root / "unlisted.txt").write_text("unreviewed")
    with pytest.raises(ValueError, match="coverage"):
        PACKAGE.verify_inventory(root, manifest, allow_fixture=True)


def test_small_fixture_cannot_be_relabeled_as_real(tmp_path):
    _, manifest, _ = fixture_bundle(tmp_path)
    manifest["fixture"] = False
    with pytest.raises(ValueError, match="1000"):
        PACKAGE.validate_manifest(manifest)


def test_full_native_checkpoint_makes_clip_base_optional(tmp_path):
    _, manifest, _ = fixture_bundle(tmp_path)
    del manifest["clip_checkpoint"]
    PACKAGE.validate_manifest(manifest, allow_fixture=True)


def test_real_checkpoint_inspection_rejects_non_tensor_placeholder(tmp_path):
    import torch

    checkpoint = tmp_path / "placeholder.bin"
    torch.save({"note": "no weights"}, checkpoint)
    with pytest.raises(ValueError, match="tensor"):
        PACKAGE.check_checkpoint(checkpoint, "ViT-B/16")


@pytest.mark.parametrize("requirement", ["torch==2.9", "numpy==2.1", "ftfy>=6",
                                            "ftfy @ https://bad/thing", "-r unsafe.txt",
                                            "ftfy==6; bad", "ftfy==6 --extra-index-url bad"])
def test_offline_install_rejects_core_replacement_or_remote_requirement(requirement):
    with pytest.raises(ValueError):
        PACKAGE.validate_offline_requirements(requirement)


def test_notebook_fixture_failure_still_exports_partial(tmp_path):
    _, _, output, receipt = build_fixture(tmp_path)
    input_root = tmp_path / "inputs"
    PACKAGE.safe_extract(output / "eaglenet_eval_input.bin", input_root / "bundle",
                         receipt["asset_manifest_sha256"])
    working = tmp_path / "working"
    working.mkdir()
    notebook = json.loads((output / "eaglenet_eval_kaggle.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]).replace("/kaggle/input", input_root.as_posix())
            source = source.replace("/kaggle/working", working.as_posix())
            exec(compile(source, "notebook_test", "exec"), namespace)
    assert namespace["STATE"]["status"] == "partial"
    assert namespace["STATE"]["benchmark_inference_completed"] is False
    assert namespace["STATE"]["stages"] == []
    results = list(working.glob("*_partial.zip"))
    assert len(results) == 1
    with zipfile.ZipFile(results[0]) as archive:
        state = json.loads(archive.read("notebook_status.json"))
        assert "fixture" in state["failure"]
        assert not state["benchmark_inference_completed"]


def test_notebook_rejects_completion_without_actual_artifact_evidence(tmp_path):
    notebook = json.loads(PACKAGE.make_notebook("a" * 64))
    source = "".join(notebook["cells"][2]["source"])
    source = source[source.index("def run_stage"):]
    state = {"failure": None, "asset_manifest_sha256": "a" * 64,
             "stages": [{"label": name} for name in
                                         ("preflight", "probe8", "probe64")],
             "benchmark_inference_completed": False}

    def fake_process(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        receipt = {"schema": "eaglenet_runtime_status_v1", "status": "evaluation_completed",
                   "fixture": False, "asset_manifest_sha256": "a" * 64,
                   "model_strict_loaded": True, "all_scores_finite": True,
                   "query_count": 1000, "candidate_count": 1000,
                   "score_shape": [1000, 1000], "benchmark_inference_completed": True}
        payload = PACKAGE.json_bytes(receipt)
        (output / "status.json").write_bytes(payload)
        (output / "artifacts.json").write_bytes(PACKAGE.json_bytes({
            "schema": "eaglenet_output_inventory_v1", "fixture": False,
            "files": {"status.json": record(payload)}}))
        return SimpleNamespace(stdout=[], wait=lambda: 0)

    def fail(message):
        state["failure"] = message

    namespace = {"STATE": state, "WORK": tmp_path, "BUNDLE": tmp_path,
                 "PINNED_MANIFEST_SHA": "a" * 64, "json": json, "sys": sys,
                 "os": os, "subprocess": SimpleNamespace(Popen=fake_process, PIPE=-1, STDOUT=-2),
                 "plain_file": PACKAGE.plain_file, "sha256_file": PACKAGE.sha256_file,
                 "fail": fail, "traceback": __import__("traceback")}
    exec(compile(source, "stage_guard", "exec"), namespace)
    namespace["run_stage"]("evaluate", "cuda")
    assert "Required runtime evidence" in state["failure"]
    assert not state["benchmark_inference_completed"]


def external_contract_fixture(tmp_path):
    """Synthetic contract data only; no runtime model or benchmark is executed."""
    root, manifest, runtime = fixture_bundle(tmp_path)
    manifest["fixture"] = False
    checkpoint = manifest.pop("checkpoint")
    manifest["inventory"].pop(checkpoint["path"])
    (root / checkpoint["path"]).unlink()
    manifest["checkpoint_request"] = {
        "filename": "msrvtt9k-vit_b_16.bin", "bytes": 3,
        "path": "weights/eaglenet_msrvtt_b16.bin", "variant": "ViT-B/16",
        "source_urls": ["https://github.com/draym28/EagleNet"],
        "provenance": "operator source assertion, not an author-published checksum"}
    videos = manifest["videos"]
    videos["test_ids"] = [f"test{index}" for index in range(1000)]
    videos["probe_ids"] = [f"train{index}" for index in range(64)]
    videos["files"] = {video_id: videos["files"]["video1"]
                       for video_id in videos["test_ids"] + videos["probe_ids"]}
    payload = PACKAGE.json_bytes(manifest)
    (root / "asset_manifest.json").write_bytes(payload)
    return root, manifest, runtime, hashlib.sha256(payload).hexdigest()


def test_pending_checkpoint_mode_is_explicit_and_not_ready(tmp_path):
    root, _, runtime, _ = external_contract_fixture(tmp_path)
    with pytest.raises(ValueError, match="external mode"):
        PACKAGE.prepare(root, runtime, tmp_path / "rejected")
    receipt = PACKAGE.prepare(root, runtime, tmp_path / "external", external_checkpoint=True)
    assert receipt["status"] == "needs_external_checkpoint"
    assert not receipt["benchmark_inference_completed"]
    notebook = json.loads((tmp_path / "external/eaglenet_eval_kaggle.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "needs_checkpoint" in source
    assert "author_published_checksum" in source


def test_pending_notebook_run_all_exports_blocked_without_starting_runtime(tmp_path, monkeypatch):
    import importlib.util

    root, _, runtime, _ = external_contract_fixture(tmp_path)
    output = tmp_path / "package"
    PACKAGE.prepare(root, runtime, output, external_checkpoint=True)
    inputs, working = tmp_path / "inputs", tmp_path / "working"
    inputs.mkdir()
    working.mkdir()
    (inputs / "eaglenet_eval_input.bin").write_bytes((output / "eaglenet_eval_input.bin").read_bytes())
    original_find = importlib.util.find_spec
    names = {"torch", "numpy", "cv2", "pandas", "ftfy", "regex", "tqdm", "boto3", "requests"}
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: object() if name in names else original_find(name))
    notebook = json.loads((output / "eaglenet_eval_kaggle.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]).replace("/kaggle/input", inputs.as_posix())
            source = source.replace("/kaggle/working", working.as_posix())
            exec(compile(source, "pending_notebook", "exec"), namespace)
    state = namespace["STATE"]
    assert state["status"] == "blocked" and state["blocked_reason"] == "needs_checkpoint"
    assert state["stages"] == [] and state["benchmark_inference_completed"] is False
    assert len(list(working.glob("*_blocked.zip"))) == 1


def test_external_checkpoint_missing_multiple_and_wrong_size_fail_before_copy(tmp_path):
    root, manifest, _, pin = external_contract_fixture(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(FileNotFoundError, match="needs_checkpoint"):
        PACKAGE.bind_external_checkpoint(root, inputs, tmp_path / "bound0", pin)
    checkpoint = inputs / manifest["checkpoint_request"]["filename"]
    checkpoint.write_bytes(b"wrong size")
    with pytest.raises(ValueError, match="size differs"):
        PACKAGE.bind_external_checkpoint(root, inputs, tmp_path / "bound1", pin)
    second = inputs / "duplicate" / checkpoint.name
    second.parent.mkdir()
    second.write_bytes(b"123")
    with pytest.raises(ValueError, match="exactly one"):
        PACKAGE.bind_external_checkpoint(root, inputs, tmp_path / "bound2", pin)
    assert not any(tmp_path.glob("bound*"))


def test_external_checkpoint_wrong_tensor_shape_or_changed_source_rejected(tmp_path, monkeypatch):
    root, manifest, _, pin = external_contract_fixture(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / manifest["checkpoint_request"]["filename"]).write_bytes(b"123")

    def reject_shape(*args):
        raise ValueError("checkpoint shape rejected")

    monkeypatch.setattr(PACKAGE, "check_checkpoint", reject_shape)
    with pytest.raises(ValueError, match="shape rejected"):
        PACKAGE.bind_external_checkpoint(root, inputs, tmp_path / "bound", pin)
    (root / "videos/video1.mp4").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size/hash"):
        PACKAGE.bind_external_checkpoint(root, inputs, tmp_path / "bound2", pin)


def test_external_binding_records_observed_hash_and_source_pin_without_author_hash(tmp_path, monkeypatch):
    root, manifest, _, pin = external_contract_fixture(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / manifest["checkpoint_request"]["filename"]).write_bytes(b"123")
    monkeypatch.setattr(PACKAGE, "check_checkpoint", lambda *args: {"synthetic_test": True})
    bound, binding = PACKAGE.bind_external_checkpoint(root, inputs, tmp_path / "bound", pin)
    assert binding["first_observed_sha256"] == hashlib.sha256(b"123").hexdigest()
    assert binding["author_published_checksum"] is None
    assert binding["source_manifest_sha256"] == pin
    assert binding["bound_manifest_sha256"] != pin
    PACKAGE.verify_inventory(tmp_path / "bound", bound)


def test_fixture_cannot_use_external_checkpoint_to_become_real(tmp_path):
    root, manifest, _, _ = external_contract_fixture(tmp_path)
    manifest["fixture"] = True
    (root / "asset_manifest.json").write_bytes(PACKAGE.json_bytes(manifest))
    with pytest.raises(ValueError, match="Fixture"):
        PACKAGE.bind_external_checkpoint(root, tmp_path, tmp_path / "bound",
                                         PACKAGE.sha256_file(root / "asset_manifest.json"))

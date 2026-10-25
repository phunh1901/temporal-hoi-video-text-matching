"""Kaggle input layouts, durable failures, and safe feature-cache reuse."""

import ast
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from temporal_hoi.inference.clip4clip import (
    decode_video,
    load_component,
    normalized_mean,
    text_tokens,
    verify_source,
)

PROJECT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(root, name="renamed-dataset.zip", payload=b"print('verified')"):
    manifest = {"scripts/run.py": hashlib.sha256(payload).hexdigest()}
    manifest_bytes = json.dumps(manifest).encode()
    path = root / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle_manifest.json", manifest_bytes)
        archive.writestr("scripts/run.py", payload)
    return path, [hashlib.sha256(manifest_bytes).hexdigest()]


def test_renamed_zip_and_expanded_input_and_repeat(tmp_path):
    module = script("kaggle_baseline_runtime")
    inputs = tmp_path / "input"
    inputs.mkdir()
    path, trusted = bundle(inputs)
    dest = tmp_path / "work"
    assert module.install_input(inputs, dest, trusted)["format"] == "zip"
    assert module.install_input(inputs, dest, trusted)["files"] == 1
    with zipfile.ZipFile(path) as archive:
        archive.extractall(inputs / "expanded")  # This fixture contains only fixed safe paths.
    assert module.install_input(inputs, dest, trusted)["format"] == "directory"


def test_no_input_reports_visible_files(tmp_path):
    module = script("kaggle_baseline_runtime")
    (tmp_path / "wrong.txt").write_text("wrong")
    with pytest.raises(FileNotFoundError, match="wrong.txt"):
        module.install_input(tmp_path, tmp_path / "dest", [])


def test_modified_payload_is_rejected_before_execution(tmp_path):
    module = script("kaggle_baseline_runtime")
    inputs = tmp_path / "input"
    inputs.mkdir()
    path, trusted = bundle(inputs)
    with zipfile.ZipFile(path) as archive:
        manifest = archive.read("bundle_manifest.json")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle_manifest.json", manifest)
        archive.writestr("scripts/run.py", "raise RuntimeError('untrusted')")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.install_input(inputs, tmp_path / "work", trusted)


@pytest.mark.parametrize("name", ["../escape.py", "/absolute", "C:/escape", "a\\b", "a/./b"])
def test_bundle_path_traversal_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        script("kaggle_baseline_runtime").safe_member(name)


def test_failed_command_preserves_live_output_events_and_zip(tmp_path):
    module = script("kaggle_baseline_runtime")
    results = tmp_path / "results"
    runner = module.LoggedCommands(results, tmp_path, dict(os.environ), heartbeat_seconds=0.05)
    with pytest.raises(RuntimeError, match="exit 3"):
        runner.run("failure", [sys.executable, "-u", "-c",
                               "import time; print('before failure'); time.sleep(.2); exit(3)"])
    assert "before failure" in (results / "console.log").read_text()
    stages = [json.loads(line)["stage"] for line in (results / "events.jsonl").read_text().splitlines()]
    assert "heartbeat" in stages and stages[-1] == "failed"
    with zipfile.ZipFile(tmp_path / "results.zip") as archive:
        assert "before failure" in archive.read("console.log").decode()


def test_command_timeout_preserves_failure(tmp_path):
    module = script("kaggle_baseline_runtime")
    runner = module.LoggedCommands(tmp_path / "results", tmp_path, dict(os.environ),
                                   heartbeat_seconds=0.05)
    with pytest.raises(TimeoutError):
        runner.run("timeout", [sys.executable, "-c", "import time; time.sleep(10)"], timeout=.2)
    assert (tmp_path / "results.zip").is_file()


def test_feature_cache_reuses_exact_identity_and_recomputes_changed_input(tmp_path):
    module = script("run_msrvtt_baseline")
    calls = []

    def compute():
        calls.append(1)
        return np.ones(512, dtype=np.float32), {"video_id": "video1"}

    identity = {"model": "hash", "media": "one"}
    first = module.cached_feature(tmp_path, identity, compute)
    second = module.cached_feature(tmp_path, identity, compute)
    third = module.cached_feature(tmp_path, dict(identity, media="changed"), compute)
    assert first[2] is False and second[2] is True and third[2] is False
    assert len(calls) == 2 and np.array_equal(first[0], second[0])
    assert not list(tmp_path.glob("*.partial"))


def test_corrupt_cache_never_silently_falls_back(tmp_path):
    module = script("run_msrvtt_baseline")
    module.cached_feature(tmp_path, {"id": "one"}, lambda: (np.ones(512), {}))
    path = next(tmp_path.glob("*.npz"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        module.cached_feature(tmp_path, {"id": "one"}, lambda: pytest.fail("silent fallback"))


def test_meanp_scores_equal_pinned_upstream_method_with_masks():
    root = PROJECT / "outputs/experiments/msrvtt/20261021_clip4clip_baseline/reference_source"
    if not root.is_dir():
        pytest.skip("Pinned source snapshot not present")
    verify_source(root)
    tree = ast.parse((root / "modules/modeling.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CLIP4Clip")
    names = {"_loose_similarity", "_mean_pooling_for_similarity_visual"}
    methods = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "verified_upstream", "exec"), scope)
    reference = types.SimpleNamespace(training=False, clip=types.SimpleNamespace(logit_scale=torch.tensor(1.)))
    reference._mean_pooling_for_similarity_visual = types.MethodType(
        scope["_mean_pooling_for_similarity_visual"], reference)
    generator = torch.Generator().manual_seed(42)
    frames = torch.randn(2, 12, 512, generator=generator)
    mask = torch.tensor([[1]*12, [1]*5+[0]*7])
    text = torch.randn(3, 1, 512, generator=generator)
    actual = scope["_loose_similarity"](reference, text, frames, None, mask, "meanP")
    videos = torch.stack([normalized_mean(frames[0]), normalized_mean(frames[1, :5])])
    text_norm = text[:, 0] / text[:, 0].norm(dim=-1, keepdim=True)
    expected = torch.exp(torch.tensor(1.)) * (text_norm @ videos.T)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_sampling_and_transform_equal_native_decoder_on_synthetic_mp4(tmp_path):
    import cv2

    root = PROJECT / "outputs/experiments/msrvtt/20261021_clip4clip_baseline/reference_source"
    if not root.is_dir():
        pytest.skip("Pinned source snapshot not present")
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(131):
        writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
    writer.release()
    native = load_component(root, "dataloaders/rawvideo_util.py").RawVideoExtractor(
        framerate=1, size=224)
    original = native.get_video_data(str(path))["video"]
    original = original[np.linspace(0, len(original) - 1, 12, dtype=int)]
    actual, _, indices = decode_video(path, native.transform)
    assert len(indices) == 12
    torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_token_ids_equal_pinned_upstream_text_loader():
    root = PROJECT / "outputs/experiments/msrvtt/20261021_clip4clip_baseline/reference_source"
    if not root.is_dir():
        pytest.skip("Pinned source snapshot not present")
    verify_source(root)
    tree = ast.parse((root / "dataloaders/dataloader_msrvtt_retrieval.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MSRVTT_DataLoader")
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_get_text")
    # Only replace removed NumPy dtype alias; retain the exact pinned method body.
    scope = {"np": types.SimpleNamespace(zeros=np.zeros, long=np.int64, array=np.array)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "verified_text_loader", "exec"), scope)
    tokenizer = load_component(root, "modules/tokenization_clip.py").SimpleTokenizer()
    reference = types.SimpleNamespace(tokenizer=tokenizer, max_words=32,
                                     SPECIAL_TOKEN={"CLS_TOKEN": "<|startoftext|>",
                                                    "SEP_TOKEN": "<|endoftext|>"})
    for sentence in ("a person moves", "a person moves " * 40):
        original = scope["_get_text"](reference, "video1", sentence)[0]
        np.testing.assert_array_equal(text_tokens(tokenizer, sentence), original)


@pytest.mark.parametrize("expanded", [False, True, "nested_archives_expanded"])
@pytest.mark.parametrize("revised", [False, True])
def test_actual_notebook_bootstrap_accepts_old_renamed_or_expanded_bundle(tmp_path, monkeypatch,
                                                                       expanded, revised):
    notebook_path = PROJECT / "outputs/experiments/msrvtt/20261021_clip4clip_baseline/msrvtt_baseline_kaggle.ipynb"
    old = PROJECT / "outputs/experiments/msrvtt/20261021_clip4clip_baseline/baseline_kaggle_input.zip"
    if not notebook_path.is_file() or not old.is_file():
        pytest.skip("Local notebook handoff absent")
    inputs, work = tmp_path / "input", tmp_path / "working"
    inputs.mkdir()
    work.mkdir()
    if expanded:
        with zipfile.ZipFile(old) as archive:
            archive.extractall(inputs)  # Trusted fixed local bundle, not user-supplied archive.
        if expanded == "nested_archives_expanded":
            import gzip

            bpe = inputs / "external/CLIP4Clip/modules/bpe_simple_vocab_16e6.txt.gz"
            bpe.with_suffix("").write_bytes(gzip.decompress(bpe.read_bytes()))
            bpe.unlink()  # Only this test's own fixture in tmp_path.
            # The preserved ZIP uses its historical internal path, not the host layout.
            metadata = inputs / "outputs/w03/sources/m2/msrvtt_data.zip"
            with zipfile.ZipFile(metadata) as archive:
                archive.extractall(metadata.parent / "msrvtt_data")
            metadata.unlink()
    else:
        shutil.copyfile(old, inputs / "temporal-hoi-baseline-input.zip")
    original_popen = subprocess.Popen
    import urllib.request

    downloaded = []

    def urlopen(request, timeout):
        url = request.full_url
        name = ("external/CLIP4Clip/modules/bpe_simple_vocab_16e6.txt.gz"
                if url.endswith(".gz") else "outputs/w03/sources/m2/msrvtt_data.zip")
        with zipfile.ZipFile(old) as archive:
            response = io.BytesIO(archive.read(name))
        response.status = 200
        downloaded.append(name)
        return response

    def popen(command, **kwargs):
        if command[0] == "nvidia-smi":
            command = [sys.executable, "-c", "print('simulated GPU inventory; no GPU test')"]
        return original_popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if revised:
        script("kaggle_notebook_cells").revise(notebook, PROJECT, old)
    code = next(c["source"] for c in notebook["cells"] if c["cell_type"] == "code")
    code = code.replace('Path("/kaggle/working")', f"Path({str(work)!r})")
    code = code.replace('Path("/kaggle/input")', f"Path({str(inputs)!r})")
    scope = {}
    exec(compile(code, "notebook_input_cell", "exec"), scope)
    exec(compile(code, "notebook_input_cell_repeat", "exec"), scope)
    executed = json.loads((work / "baseline_results/executed_source_hashes.json").read_text())
    assert executed["scripts/run_msrvtt_baseline.py"] == hashlib.sha256(
        (PROJECT / "scripts/run_msrvtt_baseline.py").read_bytes()).hexdigest()
    if revised:
        manifest = json.loads((work / "baseline/bundle_manifest.json").read_text())
        aliases = json.loads((work / "baseline_results/input_path_aliases.json").read_text())
        assert aliases == {
            "data/metadata/msrvtt_data.zip": manifest["outputs/w03/sources/m2/msrvtt_data.zip"],
            "data/metadata/msrvtt_protocol_lock.json": manifest["outputs/w03/msrvtt_protocol_lock.json"],
        }
        for name, expected_hash in aliases.items():
            assert hashlib.sha256((work / "baseline" / name).read_bytes()).hexdigest() == expected_hash
    if expanded == "nested_archives_expanded":
        receipt = json.loads((work / "baseline_results/input_receipt.json").read_text())
        assert set(receipt["recovered_exact_assets"]) == {
            "external/CLIP4Clip/modules/bpe_simple_vocab_16e6.txt.gz",
            "outputs/w03/sources/m2/msrvtt_data.zip",
        }
        assert set(downloaded) == set(receipt["recovered_exact_assets"])
        assert len(downloaded) == 2  # Repeated cell reuses already hash-checked local archives.
    else:
        assert downloaded == []
    assert len(code.encode()) < 100000
    assert code.startswith('print("CELL_1_STARTED:')


@pytest.mark.parametrize("present_corrupt", [False, True])
@pytest.mark.parametrize("name", [
    "external/CLIP4Clip/modules/bpe_simple_vocab_16e6.txt.gz",
    "data/metadata/msrvtt_data.zip",
    "outputs/w03/sources/m2/msrvtt_data.zip",
])
def test_archive_recovery_preserves_hash_and_never_hides_existing_corruption(
    tmp_path, present_corrupt, name,
):
    module = script("kaggle_baseline_runtime")
    source = tmp_path / "input"
    source.mkdir()
    original = b"exact compressed bytes"
    data = json.dumps({name: hashlib.sha256(original).hexdigest()}).encode()
    (source / "bundle_manifest.json").write_bytes(data)
    if present_corrupt:
        path = source / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.install_input(source, tmp_path / "work", [hashlib.sha256(data).hexdigest()],
                             recovery_assets={name: original if present_corrupt else b"wrong recovery"})


@pytest.mark.parametrize("data", [b"bad", b"oversized response"])
def test_recovery_download_rejects_incorrect_bytes(data, monkeypatch):
    module = script("kaggle_baseline_runtime")

    def response(*args, **kwargs):
        stream = io.BytesIO(data)
        stream.status = 200
        return stream

    monkeypatch.setattr(module.urllib.request, "urlopen", response)
    with pytest.raises(ValueError, match="exceeds|mismatch"):
        module.download_recovery("https://example.org/pinned", 3, hashlib.sha256(b"yes").hexdigest())


def test_recovery_network_failure_has_bounded_retries_and_visible_error(monkeypatch):
    module = script("kaggle_baseline_runtime")
    attempts, events = [], []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise TimeoutError("network timeout")

    monkeypatch.setattr(module.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="Kaggle Internet"):
        module.download_recovery("https://example.org/pinned", 3, "0"*64,
                                 progress=lambda stage, **kwargs: events.append(stage))
    assert len(attempts) == 3 and events.count("asset_download_retry") == 3

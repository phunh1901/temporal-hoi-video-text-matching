"""Runtime input/metric/failure checks using explicitly synthetic fixtures only."""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def runtime():
    path = Path(__file__).resolve().parents[1] / "scripts/eaglenet_kaggle_runtime.py"
    spec = importlib.util.spec_from_file_location("eaglenet_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path, runtime, monkeypatch):
    """File/schema fixture; the bytes are deliberately not real model/video evidence."""
    root = tmp_path / "bundle"
    root.mkdir()
    source = root / "upstream"
    source.mkdir()
    (source / "fixture.py").write_text("# synthetic source fixture\n", encoding="utf-8")
    pins = {"fixture.py": runtime.sha256(source / "fixture.py")}
    monkeypatch.setattr(runtime, "SOURCE_HASHES", pins)

    def asset(path):
        relative = path.relative_to(root).as_posix()
        return {"path": relative, "sha256": runtime.sha256(path), "bytes": path.stat().st_size}

    metadata = root / "metadata"
    metadata.mkdir()
    test_ids = [f"video{i}" for i in range(1000)]
    train_ids = [f"video{i}" for i in range(1000, 10000)]
    for name, ids, fieldnames in (("test.csv", test_ids, ["video_id", "sentence"]),
                                ("train.csv", train_ids, ["video_id"])):
        with (metadata / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for video_id in ids:
                writer.writerow({"video_id": video_id, **({"sentence": video_id}
                                                         if len(fieldnames) == 2 else {})})
    runtime.write_json(metadata / "annotations.json", {"sentences": [
        {"video_id": video_id, "caption": video_id} for video_id in test_ids + train_ids[:2]]})
    (root / "fixture_checkpoint.bin").write_bytes(b"not weights; schema test only")
    videos = root / "videos"
    videos.mkdir()
    video_files = {}
    for video_id in test_ids + train_ids[:2]:
        path = videos / f"{video_id}.mp4"
        path.write_bytes(b"not media; schema test only")
        video_files[video_id] = asset(path)
    manifest = {"schema": "eaglenet_eval_bundle_v1", "fixture": True,
                "source": {"path": "upstream", "commit": runtime.UPSTREAM_COMMIT,
                           "files": pins}, "checkpoint": asset(root / "fixture_checkpoint.bin"),
                "metadata": {"test_csv": asset(metadata / "test.csv"),
                             "train_csv": asset(metadata / "train.csv"),
                             "annotations": asset(metadata / "annotations.json")},
                "videos": {"root": "videos", "files": video_files,
                           "test_ids": test_ids, "probe_ids": train_ids[:2]},
                "variant": "ViT-B/16", "config": dict(runtime.FIXED_CONFIG)}
    monkeypatch.setattr(runtime, "METADATA_HASHES",
                        {key: item["sha256"] for key, item in manifest["metadata"].items()})
    runtime.write_json(root / "asset_manifest.json", manifest)
    return root, manifest


def save_manifest(runtime, root, manifest):
    runtime.write_json(root / "asset_manifest.json", manifest)


def test_preflight_checks_inventory_without_loading_fixture_model(runtime, bundle, monkeypatch):
    root, _ = bundle
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Preflight loaded weights"))
    _, rows, _, report = runtime.preflight(root)
    assert len(rows) == report["test_queries"] == 1000
    assert report["fixture"] and not report["model_loaded"]
    assert not report["benchmark_inference_completed"]


@pytest.mark.parametrize("path", ["../outside", "C:/outside", "/outside", "a\\b"])
def test_paths_cannot_escape_bundle(runtime, tmp_path, path):
    with pytest.raises(ValueError, match="path|POSIX"):
        runtime.safe_path(tmp_path, path)


def test_tampered_source_fails_before_import_or_model_load(runtime, bundle):
    root, _ = bundle
    (root / "upstream/fixture.py").write_text("raise SystemExit('not executed')", encoding="utf-8")
    with pytest.raises(ValueError, match="Pinned source hash"):
        runtime.preflight(root)


def test_original_csv_query_mismatch_is_preserved_not_replaced(runtime, bundle, monkeypatch):
    root, manifest = bundle
    csv_path = root / "metadata/test.csv"
    rows = runtime.read_csv(csv_path)
    rows[0]["sentence"] = "original CSV sentence absent from JSON"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "sentence"])
        writer.writeheader()
        writer.writerows(rows)
    manifest["metadata"]["test_csv"].update(
        sha256=runtime.sha256(csv_path), bytes=csv_path.stat().st_size)
    monkeypatch.setitem(runtime.METADATA_HASHES, "test_csv", runtime.sha256(csv_path))
    save_manifest(runtime, root, manifest)
    _, actual, _, report = runtime.preflight(root)
    assert actual[0]["sentence"] == "original CSV sentence absent from JSON"
    assert report["csv_json_caption_mismatch_count"] == 1
    assert report["canonical_csv_queries_preserved"]


def test_rehashed_metadata_tamper_still_fails_static_canonical_pin(runtime, bundle):
    root, manifest = bundle
    csv_path = root / "metadata/test.csv"
    original = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(original.replace("video0,video0", "video0,tampered"), encoding="utf-8")
    manifest["metadata"]["test_csv"].update(
        sha256=runtime.sha256(csv_path), bytes=csv_path.stat().st_size)
    save_manifest(runtime, root, manifest)
    with pytest.raises(ValueError, match="Canonical metadata hash mismatch"):
        runtime.preflight(root)


@pytest.mark.parametrize("change,pattern", [
    ("order", "order mismatch"), ("missing_video", "Missing video inventory"),
    ("source_commit", "commit"), ("probe_test", "train-only"),
    ("config", "Configuration differs"), ("weights_hash", "hash mismatch")])
def test_protocol_mutations_fail(runtime, bundle, change, pattern):
    root, manifest = bundle
    if change == "order":
        manifest["videos"]["test_ids"].reverse()
    elif change == "missing_video":
        del manifest["videos"]["files"]["video0"]
    elif change == "source_commit":
        manifest["source"]["commit"] = "0" * 40
    elif change == "probe_test":
        manifest["videos"]["probe_ids"] = ["video0"]
    elif change == "config":
        manifest["config"]["seed"] = 42
    elif change == "weights_hash":
        manifest["checkpoint"]["sha256"] = "0" * 64
    save_manifest(runtime, root, manifest)
    with pytest.raises(ValueError, match=pattern):
        runtime.preflight(root)


def test_numeric_loader_requires_tensors_finite_and_uniform_prefix(runtime, tmp_path):
    path = tmp_path / "weights.bin"
    torch.save({"module.weight": torch.ones(2)}, path)
    assert set(runtime.numeric_state(path)) == {"weight"}
    for state, error in (({"weight": "bad"}, "non-tensor"),
                         ({"module.weight": torch.ones(2), "other": torch.ones(1)}, "Mixed"),
                         ({"weight": torch.tensor([float("nan")])}, "Nonfinite")):
        torch.save(state, path)
        with pytest.raises(ValueError, match=error):
            runtime.numeric_state(path)


def test_numeric_loader_never_uses_unsafe_fallback(runtime, tmp_path, monkeypatch):
    calls = []

    def blocked(path, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("unsupported weights-only payload")

    monkeypatch.setattr(torch, "load", blocked)
    with pytest.raises(RuntimeError, match="weights-only"):
        runtime.numeric_state(tmp_path / "untrusted.bin")
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_partial_or_wrong_shape_checkpoint_is_not_filled(runtime):
    model = torch.nn.Linear(2, 3)
    original = model.weight.detach().clone()
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        runtime.strict_load(model, {"weight": torch.ones(3, 2)})
    assert torch.equal(model.weight, original)
    with pytest.raises(ValueError, match="shape mismatch"):
        runtime.strict_load(model, {"weight": torch.ones(3, 1), "bias": torch.ones(3)})
    report = runtime.strict_load(model, {"weight": torch.ones(3, 2), "bias": torch.zeros(3)})
    assert report["strict_load"] and torch.equal(model.weight, torch.ones(3, 2))


def test_stable_ties_one_rank_per_query_and_mrr(runtime):
    result = runtime.stable_metrics(np.array([[1., 1.], [0., 1.]], dtype=np.float32))
    assert result["text_to_video"]["positive_ranks"] == [1, 1]
    assert result["video_to_text"]["positive_ranks"] == [1, 2]
    assert result["video_to_text"]["R@1"] == 50
    assert result["video_to_text"]["MedR"] == 1.5
    assert result["video_to_text"]["MRR"] == 0.75
    assert result["video_to_text"]["query_count"] == 2


@pytest.mark.parametrize("scores", [np.empty((0, 0)), np.ones((2, 3)),
                                   np.array([[float("inf")]])])
def test_invalid_scores_cannot_produce_metrics(runtime, scores):
    with pytest.raises(ValueError, match="finite nonempty square"):
        runtime.stable_metrics(scores)


def test_fixture_cannot_become_real_completed_probe(runtime, bundle, tmp_path):
    root, _ = bundle
    output = tmp_path / "run"
    args = argparse.Namespace(bundle_root=root, output_dir=output, stage="probe", device="cpu",
                              probe_videos=2, threads=2)
    with pytest.raises(ValueError, match="Fixture"):
        runtime.run(args)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and status["fixture"]
    assert not status["benchmark_inference_completed"]
    assert (output / "error.txt").is_file() and (output / "artifacts.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        runtime.run(args)


def test_cpu_evaluate_has_no_automatic_fallback(runtime, bundle, tmp_path):
    root, manifest = bundle
    manifest["fixture"] = False
    save_manifest(runtime, root, manifest)
    args = argparse.Namespace(bundle_root=root, output_dir=tmp_path / "cpu_eval", stage="evaluate",
                              device="cpu", probe_videos=2, threads=2)
    with pytest.raises(ValueError, match="CUDA-only"):
        runtime.run(args)


def test_partial_native_decode_fails_without_substituting_zero_frames(runtime, monkeypatch):
    class Capture:
        def __init__(self, path):
            pass

        def get(self, key):
            return 60 if key == 1 else 30

        def isOpened(self):
            return True

        def release(self):
            pass

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=Capture, CAP_PROP_FRAME_COUNT=1, CAP_PROP_FPS=2))
    extractor = SimpleNamespace(get_video_data=lambda *a, **k: {"video": torch.ones(1, 3, 4, 4)})
    dataset = SimpleNamespace(rawVideoExtractor=extractor)
    checked = runtime.CheckedDataset(dataset, ["video1"])
    with pytest.raises(ValueError, match="Partial/invalid native decode"):
        checked.dataset.rawVideoExtractor.get_video_data("video1.mp4")
    assert checked.records == []


def test_native_decode_values_and_sample_indices_preserved(runtime, monkeypatch):
    class Capture:
        def __init__(self, path):
            pass

        def get(self, key):
            return 390 if key == 1 else 30

        def isOpened(self):
            return True

        def release(self):
            pass

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=Capture, CAP_PROP_FRAME_COUNT=1, CAP_PROP_FPS=2))
    original = {"video": torch.arange(13 * 3 * 4 * 4).reshape(13, 3, 4, 4).float()}
    extractor = SimpleNamespace(get_video_data=lambda *a, **k: original)
    checked = runtime.CheckedDataset(SimpleNamespace(rawVideoExtractor=extractor), ["video1"])
    result = checked.dataset.rawVideoExtractor.get_video_data("video1.mp4")
    assert result is original
    assert checked.records[0]["selected_frame_indices"] == (
        np.linspace(0, 12, 12, dtype=int) * 30).tolist()


def test_native_scorer_replay_restores_rng_and_exports_numeric_cache(runtime, tmp_path):
    calls = []

    def scorer(model, tm, vm, tf, vf):
        calls.append(torch.get_rng_state().clone())
        return [torch.rand(2, 2).numpy()]

    evaluator = SimpleNamespace(_run_on_single_gpu=scorer)

    def evaluate(config, model, dataloader, device, n_gpu, logger):
        torch.rand(7)  # Native feature preparation can consume CPU RNG.
        tm = [(torch.ones(2, 1, 32), torch.zeros(2, 1, 32))]
        vm = [(torch.ones(2, 1, 12),)]
        tf, vf = [torch.ones(2, 1, 512)], [torch.ones(2, 12, 512)]
        scores = np.concatenate(evaluator._run_on_single_gpu(model, tm, vm, tf, vf))
        return 0, scores, [0.1, 0.2]

    evaluator.eval_epoch = evaluate
    torch.manual_seed(42)
    output, timings = runtime.evaluate_native(SimpleNamespace(evaluator=evaluator), None, None,
                                              None, torch.device("cpu"), None, tmp_path, True)
    assert output.shape == (2, 2) and timings == [0.1, 0.2]
    assert len(calls) == 2 and torch.equal(calls[0], calls[1])
    parity = json.loads((tmp_path / "probe_parity.json").read_text(encoding="utf-8"))
    assert parity["score_bitwise_equal"]
    with np.load(tmp_path / "feature_cache.npz", allow_pickle=False) as cache:
        assert cache["video"].shape == (2, 12, 512)
        assert cache["cpu_rng_before_scoring"].dtype == np.uint8
    assert evaluator._run_on_single_gpu is scorer

"""Meaningful cache, parity and native-encoding checks on controlled fixtures."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from temporal_hoi.data.msrvtt import MSRVTTAdapter, file_sha256
from temporal_hoi.inference import pooling_workflow as workflow


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_msrvtt_pooling.py"
    spec = importlib.util.spec_from_file_location("pooling_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    from test_msrvtt_adapter import fixture_files

    archive, lock = fixture_files(tmp_path)
    pool = MSRVTTAdapter.from_files(archive, lock).pool("test")
    config = json.loads((Path(__file__).resolve().parents[1]
                         / "configs/msrvtt_pooling.json").read_text(encoding="utf-8"))
    config.update(metadata_archive=str(archive), metadata_lock=str(lock),
                  require_external_reference=False, selection_fixed_before_scores=True,
                  bootstrap_samples=20)
    cache = {
        "text": torch.tensor([[1., 0.], [0., 1.]]),
        "frames": torch.tensor([[[8., 0.], [0., 1.]], [[0., 3.], [0., 0.]]]),
        "mask": torch.tensor([[True, True], [True, False]]),
        "text_ids": list(pool.text_ids), "video_ids": list(pool.video_ids),
        "text_video_ids": [row["video_id"] for row in pool.queries],
        "metadata": {"split": "test", "logit_scale": 100.0},
    }
    cache_path = tmp_path / "features.npz"
    workflow.save_feature_cache(cache_path, cache)
    config.update(feature_cache=str(cache_path), feature_cache_sha256=file_sha256(cache_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config, config_path, pool, cache


def test_numeric_cache_roundtrip_and_no_overwrite(inputs):
    config, _, pool, expected = inputs
    cache = workflow.load_feature_cache(Path(config["feature_cache"]),
                                        config["feature_cache_sha256"], max_bytes=100_000)
    assert torch.equal(cache["frames"], expected["frames"])
    assert cache["video_ids"] == list(pool.video_ids)
    assert cache["metadata"]["logit_scale"] == 100
    with pytest.raises(FileExistsError, match="replace"):
        workflow.save_feature_cache(Path(config["feature_cache"]), expected)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        workflow.load_feature_cache(Path(config["feature_cache"]), "f" * 64, max_bytes=100_000)


def test_safe_pt_cache_and_expansion_budget(inputs, tmp_path):
    _, _, _, cache = inputs
    path = tmp_path / "features.pt"
    torch.save(cache, path)
    checked = workflow.load_feature_cache(path, file_sha256(path), max_bytes=100_000)
    assert checked["text_ids"] == cache["text_ids"]
    with pytest.raises(ValueError, match="byte budget"):
        workflow.load_feature_cache(path, file_sha256(path), max_bytes=32)


def test_object_array_pickle_is_not_loaded(inputs, tmp_path):
    _, _, _, _ = inputs
    path = tmp_path / "object.npz"
    np.savez(path, text=np.asarray([object()], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        workflow.load_feature_cache(path, file_sha256(path), max_bytes=100_000)


def test_cli_runs_three_methods_and_refuses_output_reuse(inputs, tmp_path):
    config, path, _, _ = inputs
    cli = runner()
    output = tmp_path / "evaluation"
    args = ["--config", str(path), "--output-dir", str(output), "--execute",
            "--allow-final-evaluation"]
    assert cli.main(args) == 0
    report = json.loads((output / "benchmark.json").read_text())
    assert report["status"] == "completed_frozen_pooling_evaluation"
    assert report["artifact_kind"] == "existing_feature_cache_scoring_replay"
    assert report["encoder_train_run"] is False and report["selection_performed"] is False
    assert set(report["variants"]) == {"B0", "B1", "B2"}
    assert report["baseline_gate"]["passed"] is True
    assert report["baseline_gate"]["external_baseline_reproduced"] is False
    assert len((output / "metrics.csv").read_text().splitlines()) == 7
    with np.load(output / "B0_scores.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["scores"], [[100 / np.sqrt(2), 0],
                                                    [100 / np.sqrt(2), 100]], atol=1e-5)
    original = (output / "benchmark.json").read_bytes()
    with pytest.raises(SystemExit):
        cli.main(args)
    assert (output / "benchmark.json").read_bytes() == original
    assert config["tau"] == 0.05 and config["alpha"] == 0.25


def test_test_scoring_requires_explicit_flag_before_execute(inputs, tmp_path, monkeypatch):
    _, path, _, _ = inputs
    cli = runner()
    monkeypatch.setattr(cli, "execute", lambda *a, **k: pytest.fail("Test scoring happened"))
    output = tmp_path / "no_flag"
    assert cli.main(["--config", str(path), "--output-dir", str(output), "--execute"]) == 2
    report = json.loads((output / "run.json").read_text())
    assert "fixed selection" in report["error"]
    assert report["benchmark_metrics"] is None


def test_preflight_checks_hash_without_loading_features(inputs, monkeypatch):
    config, _, _, _ = inputs
    monkeypatch.setattr(workflow, "load_feature_cache", lambda *a, **k: pytest.fail("Cache loaded"))
    monkeypatch.setattr(workflow, "build_frame_cache", lambda *a, **k: pytest.fail("Encoder loaded"))
    _, report = workflow.preflight(config)
    assert report["status"] == "files_verified_not_executed"
    assert report["model_loaded"] is False and report["benchmark_metrics"] is None
    _, blocked = workflow.preflight(dict(config, feature_cache_sha256="0" * 64))
    assert blocked["status"] == "blocked" and "SHA256 mismatch" in blocked["blockers"][0]


def test_cache_identity_mismatch_does_not_score(inputs, tmp_path, monkeypatch):
    config, _, pool, cache = inputs
    altered = dict(cache, text_ids=list(reversed(cache["text_ids"])))
    path = tmp_path / "wrong_ids.npz"
    workflow.save_feature_cache(path, altered)
    config.update(feature_cache=str(path), feature_cache_sha256=file_sha256(path))
    monkeypatch.setattr(workflow, "score_matrix", lambda *a, **k: pytest.fail("Scored bad IDs"))
    with pytest.raises(ValueError, match="IDs/order"):
        workflow.execute(config, pool, tmp_path)


def test_bad_external_reference_stops_before_improvements(inputs, tmp_path):
    config, _, pool, cache = inputs
    reference = tmp_path / "bad_baseline.npz"
    np.savez(reference, scores=np.zeros((2, 2), dtype=np.float32),
             video_ids=np.asarray(cache["video_ids"]), text_ids=np.asarray(cache["text_ids"]))
    config.update(baseline_reference=str(reference), reference_sha256=file_sha256(reference),
                  require_external_reference=True)
    output = tmp_path / "gate_failure"
    output.mkdir()
    with pytest.raises(ValueError, match="baseline parity failed"):
        workflow.execute(config, pool, output)
    assert (output / "B0_scores.npz").is_file()
    assert not (output / "B1_scores.npz").exists()
    assert not (output / "B2_scores.npz").exists()
    assert json.loads((output / "baseline_gate.json").read_text())["passed"] is False


def test_cache_replay_disables_tf32_and_records_code_provenance(inputs, tmp_path, monkeypatch):
    config, _, pool, _ = inputs
    config = dict(config, device="cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    original_scoring = workflow.score_matrix

    def cpu_fixture_scores(*args, **kwargs):
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        # CUDA readiness is mocked only to exercise configuration; no GPU workload.
        kwargs["device"] = "cpu"
        return original_scoring(*args, **kwargs)

    monkeypatch.setattr(workflow, "score_matrix", cpu_fixture_scores)
    output = tmp_path / "tf32_configuration"
    output.mkdir()
    workflow.execute(config, pool, output)
    report = json.loads((output / "benchmark.json").read_text())
    assert report["matmul_tf32"] is False and report["cudnn_tf32"] is False
    assert report["core_sha256"] == file_sha256(Path(workflow.__file__).with_name(
        "retrieval_pooling.py"))
    assert report["cli_sha256"] == file_sha256(Path(__file__).resolve().parents[1]
                                               / "scripts/run_msrvtt_pooling.py")


def test_native_cache_keeps_each_frame_and_frozen_encoder(inputs, tmp_path, monkeypatch):
    import time

    config, _, pool, _ = inputs
    checkpoint = tmp_path / "native.pt"
    checkpoint.write_bytes(b"mocked native weights; model loader is patched")
    media = tmp_path / "media"
    media.mkdir()
    for vid in pool.video_ids:
        (media / f"{vid}.mp4").write_bytes(b"fixture video")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"source_url": "https://example.invalid/model",
                                      "training_regime": "controlled test"}), encoding="utf-8")
    config.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                  checkpoint_provenance=str(provenance), media_root=str(media))

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = torch.nn.Parameter(torch.tensor(np.log(100), dtype=torch.float32))

        def encode_text(self, tokens):
            assert not torch.is_grad_enabled()
            return torch.ones((len(tokens), 512))

        def encode_image(self, pixels):
            assert not torch.is_grad_enabled()
            return torch.ones((len(pixels), 512)) * torch.arange(1, len(pixels) + 1)[:, None]

    model = Encoder()
    monkeypatch.setattr(workflow.clip4clip, "verify_source", lambda _: {"fixture": "hash"})
    monkeypatch.setattr(workflow.clip4clip, "load_model", lambda *a: model)
    monkeypatch.setattr(workflow.clip4clip, "text_tokens", lambda *a: np.ones((1, 32), dtype=np.int64))
    monkeypatch.setattr(workflow.clip4clip, "load_component", lambda root, path: (
        SimpleNamespace(SimpleTokenizer=lambda: object()) if "tokenization" in path
        else SimpleNamespace(RawVideoExtractor=lambda **kw: SimpleNamespace(transform=object()))))
    monkeypatch.setattr(workflow.clip4clip, "decode_video", lambda *a: (
        torch.ones((2, 3, 224, 224)), [0.0, 1.0], [0, 30]))
    cache = workflow.build_frame_cache(config, pool, started=time.monotonic())
    assert cache["frames"].shape == (2, 12, 512)
    assert cache["mask"].sum().item() == 4
    assert torch.all(cache["frames"][:, 1] == 2)
    assert torch.all(cache["frames"][:, 2:] == 0)
    assert model.logit_scale.requires_grad is False
    assert cache["metadata"]["checkpoint_sha256"] == file_sha256(checkpoint)


@pytest.mark.parametrize(("key", "value"), [("tau", float("nan")), ("alpha", 2),
                                             ("query_chunk", True), ("device", "automatic")])
def test_invalid_config_fails_before_runtime(inputs, key, value):
    config, _, _, _ = inputs
    with pytest.raises(ValueError):
        workflow.validate_config(dict(config, **{key: value}))

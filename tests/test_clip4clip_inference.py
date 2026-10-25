"""Protocol and failure checks without requiring benchmark weights or videos."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from temporal_hoi.inference import clip4clip


def test_token_truncation_keeps_end_and_zero_padding():
    class Tokenizer:
        def tokenize(self, sentence):
            return sentence.split()

        def convert_tokens_to_ids(self, words):
            return [101 if w == "<|startoftext|>" else 999 if w == "<|endoftext|>" else 1
                    for w in words]

    long = clip4clip.text_tokens(Tokenizer(), "word " * 50)
    short = clip4clip.text_tokens(Tokenizer(), "one")
    assert long.shape == short.shape == (1, 32)
    assert long[0, 0] == 101 and long[0, -1] == 999
    assert short[0, 2] == 999 and (short[0, 3:] == 0).all()
    with pytest.raises(ValueError, match="caption"):
        clip4clip.text_tokens(Tokenizer(), " ")


def test_sampling_matches_native_int_fps_and_uniform_slice():
    all_indices, selected = clip4clip.sampled_indices(901, 29.97)
    native_indices = np.arange(0, 901, 29)
    assert np.array_equal(all_indices, native_indices)
    assert np.array_equal(selected, native_indices[np.linspace(0, 31, 12, dtype=int)])
    assert clip4clip.sampled_indices(15, 30)[1].tolist() == [0]
    with pytest.raises(ValueError, match="metadata"):
        clip4clip.sampled_indices(100, 0)


def test_pool_normalizes_each_frame_before_mean():
    # Unequal feature norms expose averaging-before-normalization mistakes.
    features = torch.tensor([[10., 0.], [0., 1.]])
    expected = torch.tensor([2 ** -0.5, 2 ** -0.5])
    assert torch.allclose(clip4clip.normalized_mean(features), expected)
    with pytest.raises(ValueError, match="Zero frame"):
        clip4clip.normalized_mean(torch.tensor([[0., 0.]]))
    with pytest.raises(ValueError, match="mean-pooled"):
        clip4clip.normalized_mean(torch.tensor([[1., 0.], [-1., 0.]]))


def test_checkpoint_rejects_alternate_architecture_and_partial_weights():
    with pytest.raises(ValueError, match="non-clip"):
        clip4clip.checkpoint_clip_state({"clip.visual.conv1.weight": torch.ones(1),
                                        "frame_position_embeddings.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="ViT-B/32"):
        clip4clip.checkpoint_clip_state({"clip.visual.conv1.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="tensor"):
        clip4clip.checkpoint_clip_state({"clip.bad": "not a tensor"})


def test_checkpoint_handles_only_uniform_ddp_prefix():
    shapes = {"visual.conv1.weight": (768, 3, 32, 32), "text_projection": (512, 512),
              "positional_embedding": (77, 512), "token_embedding.weight": (49408, 512)}
    tensors = {f"module.clip.{key}": torch.empty(shape, device="meta")
               for key, shape in shapes.items()}
    state = clip4clip.checkpoint_clip_state({"state_dict": tensors})
    assert set(state) == set(shapes)
    tensors["clip.extra"] = torch.ones(1)
    with pytest.raises(ValueError, match="non-clip"):
        clip4clip.checkpoint_clip_state(tensors)


def test_source_tampering_fails_before_execution(tmp_path, monkeypatch):
    path = tmp_path / "module.py"
    path.write_text("raise RuntimeError('must not run')", encoding="utf-8")
    monkeypatch.setattr(clip4clip, "SOURCE_HASHES", {"module.py": hashlib.sha256(b"trusted").hexdigest()})
    with pytest.raises(ValueError, match="hash mismatch"):
        clip4clip.load_component(tmp_path, "module.py")


def runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_msrvtt_baseline.py"
    spec = importlib.util.spec_from_file_location("baseline_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_does_not_load_model_or_imply_execution(tmp_path, monkeypatch):
    module = runner_module()

    class Pool:
        video_ids = ("video1", "video2")

        def hashes(self):
            return {"test": "hash"}

    class Adapter:
        def pool(self, split, selected_video_ids=None):
            pool = Pool()
            if selected_video_ids:
                pool.video_ids = tuple(selected_video_ids)
            return pool

    monkeypatch.setattr(module.MSRVTTAdapter, "from_files", lambda *a: Adapter())
    monkeypatch.setattr(module, "verify_source", lambda root: {})
    monkeypatch.setattr(module, "load_model", lambda *a: pytest.fail("Model loaded during preflight"))
    config = {"schema_version": "clip4clip_meanp_inference_v1",
              "upstream_commit": clip4clip.UPSTREAM_COMMIT,
              "max_words": 32, "max_frames": 12, "split": "validation", "device": "cpu",
              "frame_batch_size": 1, "max_rss_bytes": 100, "max_run_hours": 1,
              "metadata_archive": "unused", "metadata_lock": "unused", "source_root": "unused",
              "media_root": str(tmp_path)}
    pool, report = module.preflight(config, probe_videos=1)
    assert pool.video_ids == ("video1",)
    assert report["missing_video_ids"] == ["video1"]
    assert report["benchmark_inference_run"] is False and report["benchmark_metrics"] is None
    assert report["status"] == "blocked"
    with pytest.raises(ValueError, match="Probe"):
        module.preflight(dict(config, split="test"), probe_videos=1)


def test_decode_missing_video_fails_without_black_frame(tmp_path):
    with pytest.raises(ValueError, match="Cannot open"):
        clip4clip.decode_video(tmp_path / "missing.mp4", lambda image: image)


def test_runner_preserves_existing_provenance_before_preflight(tmp_path, monkeypatch):
    module = runner_module()
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "previous"
    output.mkdir()
    provenance = output / "provenance.json"
    provenance.write_text('{"keep":true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["runner", "--config", str(config),
                                     "--output-dir", str(output)])
    monkeypatch.setattr(module, "preflight", lambda *a: pytest.fail("Preflight should not run"))
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert provenance.read_text(encoding="utf-8") == '{"keep":true}'

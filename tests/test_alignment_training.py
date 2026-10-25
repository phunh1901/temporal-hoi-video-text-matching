"""Training correctness on tensor fixtures, not MSR-VTT benchmark evidence."""

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from temporal_hoi.data.msrvtt import RetrievalPool, file_sha256, semantic_hash
from temporal_hoi.models.alignment import VideoTextLinearHead
from temporal_hoi.training.alignment import (
    distinct_video_batches,
    evaluate_alignment,
    train_alignment,
    validate_config,
)
from temporal_hoi.training.features import (
    FrozenFeatureSplit,
    load_cached_split,
    validate_training_splits,
)
from temporal_hoi.training.losses import bidirectional_multi_positive_loss

ROOT = Path(__file__).resolve().parents[1]


def config_fixture():
    config = json.loads((ROOT / "configs/frozen_feature_alignment.json").read_text())
    config.update(max_epochs=4, patience=10, feature_batch_videos=2, validation_query_chunk=2)
    return config


def feature_fixture(split, count=5, dimension=6):
    ids = tuple(f"{split}_v{i}" for i in range(count))
    queries = tuple({"text_id": f"{split}_t{i}_{j}", "video_id": video,
                     "caption": "duplicate raw caption"}
                    for i, video in enumerate(ids) for j in range(2))
    pool = RetrievalPool(split, ids, queries, count, len(queries))
    random = np.random.default_rng(4 if split == "train" else 5)
    video = random.normal(size=(count, dimension)).astype(np.float32)
    text = np.repeat(video, 2, axis=0) + random.normal(0, 0.5, (2 * count, dimension)).astype(np.float32)
    provenance = {"dataset": "tensor_fixture", "artifact_kind": "synthetic_technical_probe",
                  "encoder": {"checkpoint_sha256": semantic_hash("tensor_fixture")},
                  "protocol_lock_sha256": semantic_hash("same_locked_protocol"),
                  "pool_hashes": pool.hashes(), "scope": "synthetic_fixture_only"}
    return FrozenFeatureSplit(pool, video, text, tuple(f"source_{v}" for v in ids), provenance)


def test_multi_positive_uniform_loss_matches_hand_calculation_and_gradients():
    cosine = torch.zeros(2, 4, requires_grad=True)
    labels = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    loss, stats = bidirectional_multi_positive_loss(cosine, labels, temperature=1)
    assert float(loss.detach()) == pytest.approx(math.log(2))
    loss.backward()
    assert stats["positive_pairs"] == 4 and stats["negative_pairs"] == 4
    assert torch.isfinite(cosine.grad).all()
    assert (cosine.grad[labels == 1] < 0).all()
    assert (cosine.grad[labels == 0] > 0).all()


def test_unknown_pair_is_not_a_negative_and_has_zero_gradient():
    labels = torch.tensor([[1, 0, -1], [0, 1, 1]])
    first = torch.tensor([[0.5, 0.1, 10000.0], [0.2, 0.4, 0.3]], requires_grad=True)
    second = first.detach().clone()
    second[0, 2] = -10000
    loss, stats = bidirectional_multi_positive_loss(first, labels)
    other, _ = bidirectional_multi_positive_loss(second, labels)
    assert float(loss.detach()) == pytest.approx(float(other))
    loss.backward()
    assert first.grad[0, 2] == 0
    assert stats["unknown_pairs"] == 1 and stats["excluded_text_anchors"] == 1


def test_logsumexp_handles_large_logits_and_negative_free_batches_are_rejected():
    scores = torch.tensor([[0.9, -0.9], [-0.9, 0.9]], requires_grad=True)
    loss, _ = bidirectional_multi_positive_loss(scores, torch.eye(2), temperature=1e-5)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    with pytest.raises(ValueError, match="positive and negative"):
        bidirectional_multi_positive_loss(torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError, match="Relation labels"):
        bidirectional_multi_positive_loss(scores, torch.tensor([[1, 2], [0, 1]]))
    with pytest.raises(ValueError, match="finite"):
        bidirectional_multi_positive_loss(scores * torch.nan, torch.eye(2))


def test_head_detaches_frozen_inputs_and_updates_both_projections():
    video = torch.tensor([[1., 0., 0.], [0., 1., 0.]], requires_grad=True)
    text = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.8, 0.3]], requires_grad=True)
    head = VideoTextLinearHead(3, 3)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in head.parameters()]
    loss, _ = bidirectional_multi_positive_loss(head(video, text), torch.eye(2))
    loss.backward()
    assert video.grad is None and text.grad is None
    for param in head.parameters():
        assert torch.isfinite(param.grad).all() and param.grad.norm() > 0
    optimizer.step()
    assert all((p.detach() - old).norm() > 0 for p, old in zip(head.parameters(), before, strict=True))
    with pytest.raises(ValueError, match="nonzero"):
        head.encode_video(torch.zeros(2, 3))


@pytest.mark.parametrize("count,batch", [(2, 16), (3, 2), (5, 2), (17, 16), (33, 16)])
def test_distinct_batch_negatives_and_singleton_tail_are_preserved(count, batch):
    batches = distinct_video_batches(count, batch, torch.Generator().manual_seed(42))
    assert all(len(b) >= 2 and len(set(b)) == len(b) for b in batches)
    assert sorted(v for b in batches for v in b) == list(range(count))
    with pytest.raises(ValueError, match="two distinct"):
        distinct_video_batches(1, batch, torch.Generator())


def test_source_leakage_pool_hash_and_test_split_are_rejected():
    train, validation = feature_fixture("train"), feature_fixture("validation")
    validate_training_splits(train, validation)
    validation.source_groups = (train.source_groups[0],) + validation.source_groups[1:]
    with pytest.raises(ValueError, match="overlap"):
        validate_training_splits(train, validation)
    validation = feature_fixture("validation")
    validation.provenance["pool_hashes"]["positive_mapping_sha256"] = "changed"
    with pytest.raises(ValueError, match="pool hashes"):
        validate_training_splits(train, validation)
    with pytest.raises(ValueError, match="Only development"):
        feature_fixture("test")


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_optimizer_checkpoint_resume_matches_uninterrupted_training(tmp_path, head):
    config = config_fixture()
    config.update(head=head, hidden_dim=8)
    train, validation = feature_fixture("train"), feature_fixture("validation", count=4)
    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "resumed"
    result = train_alignment(train, validation, config, continuous)
    paused = train_alignment(train, validation, config, interrupted, stop_after_epoch=2)
    resumed = train_alignment(train, validation, config, interrupted, resume=interrupted / "last.pt")
    assert result["all_losses_finite"] and result["head_update_norm"] > 0
    assert result["trainable_parameters"] > 0 and result["elapsed_seconds"] > 0
    assert paused["step_count"] == 4 and resumed["step_count"] == 8
    full = torch.load(continuous / "last.pt", weights_only=True)
    restored = torch.load(interrupted / "last.pt", weights_only=True)
    for key, expected in full["model_state_dict"].items():
        assert torch.equal(restored["model_state_dict"][key], expected)
    for a, b in zip(result["history"], resumed["history"], strict=True):
        assert a["train_loss"] == b["train_loss"] and a["val_t2v_r1"] == b["val_t2v_r1"]
    best = torch.load(interrupted / "best.pt", weights_only=True)
    assert best["selection_split"] == "validation" and best["optimizer_state_dict"]["state"]
    assert best["config"] == config and best["encoder_sha256"] == train.provenance["encoder"]["checkpoint_sha256"]
    assert not resumed["benchmark_training_run"] and not resumed["test_accessed"]
    changed = copy.deepcopy(config)
    changed["temperature"] = 0.1
    with pytest.raises(ValueError, match="provenance changed"):
        train_alignment(train, validation, changed, interrupted, resume=interrupted / "last.pt")


def test_early_stopping_uses_validation_r1_and_keeps_first_tied_checkpoint(tmp_path):
    config = config_fixture()
    config.update(max_epochs=10, patience=2, lr=1e-8)
    train, validation = feature_fixture("train"), feature_fixture("validation")
    result = train_alignment(train, validation, config, tmp_path)
    assert result["epochs_completed"] == 3 and result["stop_reason"] == "early_stopping"
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert best["epoch"] == 1
    model = VideoTextLinearHead(6, 6)
    model.load_state_dict(best["model_state_dict"])
    metrics, scores = evaluate_alignment(model, validation, config)
    recorded = json.loads((tmp_path / "val_metrics.json").read_text())
    assert metrics["text_to_video"] == recorded["text_to_video"]
    with np.load(tmp_path / "val_scores.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["scores"], scores)
        assert saved["text_ids"].tolist() == list(validation.pool.text_ids)


def cache_fixture(tmp_path):
    # Adapter pool contract is isolated here; full metadata lock validation is
    # exercised by test_msrvtt_adapter.py. Interleaved caption order is deliberate.
    features = feature_fixture("train", count=2, dimension=3)
    old = features.pool
    queries = (old.queries[0], old.queries[2], old.queries[1], old.queries[3])
    expected_pool = RetrievalPool("train", old.video_ids, queries, 2, 4)
    class Adapter:
        source_urls = {v: f"source_{v}" for v in expected_pool.video_ids}

        def pool(self, split, selected_video_ids=None):
            assert split == "train" and set(selected_video_ids) == set(expected_pool.video_ids)
            return expected_pool
    pool = expected_pool
    config = config_fixture()
    config["cache_dir"] = str(tmp_path)
    lock = tmp_path / "lock.json"
    lock.write_text("locked metadata", encoding="utf-8")
    config["metadata_lock"] = str(lock)
    encoder = {"frames": 2, "crop_mode": "global", "checkpoint_sha256": semantic_hash("encoder")}
    frames = np.repeat(np.eye(3, dtype=np.float32)[:2, None, :], 2, axis=1)
    text = np.asarray([frames[0, 0], frames[1, 0], frames[0, 0], frames[1, 0]])
    path = tmp_path / "train_0000.npz"
    np.savez_compressed(path, video_ids=np.asarray(pool.video_ids), text_ids=np.asarray(pool.text_ids),
                        text_video_ids=np.asarray([q["video_id"] for q in pool.queries]),
                        frame_embeddings=frames, valid_frame_mask=np.ones((2, 2), dtype=bool),
                        timestamps_s=np.asarray([[0, 1], [0, 1]], dtype=np.float64), text_embeddings=text)
    manifest = {"schema_version": "msrvtt_frozen_cache_v1", "status": "ready", "encoder": encoder,
                "protocol_lock_sha256": file_sha256(lock), "dimension": 3, "cohort_lock_sha256": None,
                "splits": {"train": {"video_ids": list(pool.video_ids), "pool_hashes": pool.hashes(),
                           "raw_caption_sha256": semantic_hash(list(pool.queries)),
                           "source_group_sha256": semantic_hash([Adapter.source_urls[v] for v in pool.video_ids]),
                           "shards": [{"path": path.name, "sha256": file_sha256(path)}]}}}
    (tmp_path / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return config, Adapter(), manifest, encoder, path


def test_cache_validates_hashes_axes_all_captions_and_masked_mean(tmp_path):
    config, adapter, manifest, encoder, path = cache_fixture(tmp_path)
    cached = load_cached_split(config, adapter, manifest, "train", encoder)
    assert cached.pool.text_ids == ("train_t0_0", "train_t1_0", "train_t0_1", "train_t1_1")
    np.testing.assert_array_equal(cached.video_embeddings, np.eye(3, dtype=np.float32)[:2])
    manifest["splits"]["train"]["pool_hashes"]["positive_mapping_sha256"] = "changed"
    with pytest.raises(ValueError, match="order hash"):
        load_cached_split(config, adapter, manifest, "train", encoder)
    config, adapter, manifest, encoder, path = cache_fixture(tmp_path)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="shard file hash"):
        load_cached_split(config, adapter, manifest, "train", encoder)


def test_cache_rejects_missing_caption_even_when_file_hash_is_updated(tmp_path):
    config, adapter, manifest, encoder, path = cache_fixture(tmp_path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    for key in ("text_ids", "text_video_ids", "text_embeddings"):
        arrays[key] = arrays[key][:-1]
    np.savez_compressed(path, **arrays)
    manifest["splits"]["train"]["shards"][0]["sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="incomplete raw captions"):
        load_cached_split(config, adapter, manifest, "train", encoder)


def test_resource_and_config_limits_are_enforced(tmp_path):
    config = config_fixture()
    config["feature_batch_videos"] = 1
    with pytest.raises(ValueError, match="At least 2"):
        validate_config(config)
    config = config_fixture()
    config["max_rss_bytes"] = 1
    with pytest.raises(MemoryError, match="Observed RSS"):
        train_alignment(feature_fixture("train"), feature_fixture("validation"), config, tmp_path)
    assert not (tmp_path / "best.pt").exists()


def test_readiness_with_absent_media_never_creates_a_benchmark_checkpoint(tmp_path):
    # Use a validated metadata adapter fixture from the adapter tests.
    spec = importlib.util.spec_from_file_location("train_alignment_cli", ROOT / "scripts/train_alignment.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    from test_msrvtt_adapter import fixture_files
    archive, lock = fixture_files(tmp_path)
    config = config_fixture()
    config.update(metadata_archive=str(archive), metadata_lock=str(lock), media_root=str(tmp_path / "media"),
                  cache_dir=str(tmp_path / "cache"))
    _, result = cli.alignment_readiness(config)
    assert result["status"] == "blocked" and result["benchmark_metrics"] is None
    assert result["missing_ids"] == {"train": ["video0", "video1"], "validation": ["video2"]}
    assert result["checkpoint_path"] is None and not result["benchmark_training_run"]


def test_real_cache_code_probe_stops_at_exact_optimizer_step_count(tmp_path):
    spec = importlib.util.spec_from_file_location("feature_probe_cli", ROOT / "scripts/train_alignment.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config = config_fixture()
    train, validation = feature_fixture("train", count=5), feature_fixture("validation", count=4)
    # 5 videos at batch2 means two steps per epoch; 3 steps must not mean 3 epochs.
    result = cli.run_feature_probe(train, validation, config, tmp_path, steps=3)
    assert result["completed_probe_steps"] == 3 and result["step_count"] == 3
    assert result["resume_optimizer_steps_advanced"] and result["head_update_norm"] > 0
    assert result["checkpoint_selection_performed"] is False
    assert result["benchmark_metrics"] is None and not (tmp_path / "best.pt").exists()


def test_cache_producer_writes_bounded_shards_without_dropping_duplicate_captions(tmp_path, monkeypatch):
    from PIL import Image
    from test_msrvtt_adapter import fixture_files
    archive, lock = fixture_files(tmp_path)
    spec = importlib.util.spec_from_file_location("cache_features_cli", ROOT / "scripts/cache_msrvtt_features.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config = config_fixture()
    config.update(metadata_archive=str(archive), metadata_lock=str(lock), media_root=str(tmp_path),
                  cache_dir=str(tmp_path / "cache"), shard_videos=1)
    for i in range(3):
        (tmp_path / f"video{i}.mp4").write_bytes(b"unit fixture: decoder is stubbed")
    checkpoint = tmp_path / "encoder.tensor"
    checkpoint.write_bytes(b"unit fixture encoder marker; no pretrained model")
    encoder_run = {"checkpoint_inference": True,
                   "checkpoint": {"frozen": True, "path": str(checkpoint),
                                  "sha256": file_sha256(checkpoint), "model_name": "fixture_model"},
                   "preprocess": {"num_frames_per_clip": 8, "decode_max_frame_side": 320,
                                  "image_transform": "fixture_preprocess"}}
    encoder_json = tmp_path / "encoder_run.json"
    encoder_json.write_text(json.dumps(encoder_run), encoding="utf-8")
    config["encoder_run_json"] = str(encoder_json)
    class Preprocess:
        def __call__(self, image):
            return torch.tensor([1., 2., 3.])

        def __str__(self):
            return "fixture_preprocess"

    class Encoder(torch.nn.Module):
        def encode_image(self, tensor):
            return tensor

        def encode_text(self, tensor):
            return tensor

    model = Encoder()
    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=lambda *a, **k: (model, None, Preprocess()),
        get_tokenizer=lambda name: lambda captions: torch.ones(len(captions), 3))
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
    class Reader:
        duration_s = 8

        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read_interval_frames(self, *args):
            return [Image.new("RGB", (8, 8)) for _ in range(8)], list(range(8))

        def get_metadata(self):
            return {"fps": 1, "width": 8, "height": 8}
    monkeypatch.setattr(cli, "VideoReader", Reader)
    result = cli.cache_split(config, "train")
    assert result["status"] == "building" and result["videos"] == 2 and result["texts"] == 4
    cli.cache_split(config, "validation")
    manifest_path = Path(config["cache_dir"]) / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "ready" and len(manifest["splits"]["train"]["shards"]) == 2
    adapter = cli.MSRVTTAdapter.from_files(archive, lock)
    cached = load_cached_split(config, adapter, manifest, "train", cli.encoder_descriptor(config, encoder_run))
    assert cached.pool.text_ids == adapter.pool("train").text_ids
    assert len(cached.text_embeddings) == 4  # Two duplicate strings retain distinct row IDs.


def test_validation_checks_peak_before_allocation_and_deadline_between_chunks(monkeypatch):
    from temporal_hoi.training import resources
    features = feature_fixture("validation")
    model = VideoTextLinearHead(6, 6)
    config = config_fixture()
    monkeypatch.setattr(resources, "process_rss_bytes", lambda: 100)
    config["max_rss_bytes"] = 200  # Current RSS fits; projected resident/working arrays do not.
    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("Validation allocated before checking its projected peak")
    with monkeypatch.context() as allocation_patch:
        allocation_patch.setattr(np, "empty", unexpected_allocation)
        with pytest.raises(MemoryError, match="Planned validation"):
            evaluate_alignment(model, features, config)
    config.update(max_rss_bytes=1073741824, max_train_hours=0.001)
    clock_calls = []
    def clock():
        clock_calls.append(True)
        return 0 if len(clock_calls) <= 2 else 10
    monkeypatch.setattr(resources.time, "monotonic", clock)
    encoded_chunks = []
    original_encode = model.encode_video
    def encode(raw):
        encoded_chunks.append(len(raw))
        return original_encode(raw)
    monkeypatch.setattr(model, "encode_video", encode)
    with pytest.raises(TimeoutError, match="max_train_hours"):
        evaluate_alignment(model, features, config, started=0)
    assert encoded_chunks == [2]  # Deadline stops before the next projection chunk.


def test_cache_preflight_counts_decompressed_and_pooling_peak_before_allocation(tmp_path, monkeypatch):
    from temporal_hoi.training import resources
    config, adapter, manifest, encoder, _ = cache_fixture(tmp_path)
    output_array_bytes = 4 * 3 * (2 + 4)
    # Enough for persistent output matrices, insufficient for the NPZ and
    # float64 mean-pooling intermediates. The old output-only check passed it.
    config["max_rss_bytes"] = 100 + output_array_bytes + 1
    monkeypatch.setattr(resources, "process_rss_bytes", lambda: 100)
    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("Cache allocated before checking decompression/pooling peak")
    monkeypatch.setattr(np, "empty", unexpected_allocation)
    with pytest.raises(MemoryError, match="decompressed/pooling shard"):
        load_cached_split(config, adapter, manifest, "train", encoder)

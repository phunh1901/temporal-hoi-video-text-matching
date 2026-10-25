"""Hash-validated, bounded-shard frozen features for the locked MSR-VTT pool."""

import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from temporal_hoi.data.msrvtt import (
    RetrievalPool,
    file_sha256,
    mean_pool_valid_frames,
    semantic_hash,
)
from temporal_hoi.training.resources import enforce_resources, ensure_allocation


@dataclass
class FrozenFeatureSplit:
    pool: RetrievalPool
    video_embeddings: np.ndarray
    text_embeddings: np.ndarray
    source_groups: tuple[str, ...]
    provenance: dict
    # Explicit relevance is optional for reusable technical probes. Production
    # MSR-VTT follows the locked complete video-identity relevance definition.
    relation_labels: np.ndarray | None = None

    def __post_init__(self):
        pool = self.pool
        if pool.split not in {"train", "validation"}:
            raise ValueError("Only development train/validation features may enter training")
        if (len(pool.video_ids) < 2 or len(set(pool.video_ids)) != len(pool.video_ids)
                or len(set(pool.text_ids)) != len(pool.text_ids)
                or any(q["video_id"] not in pool.video_ids for q in pool.queries)):
            raise ValueError("Need at least two distinct videos and valid unique caption IDs")
        if {q["video_id"] for q in pool.queries} != set(pool.video_ids):
            raise ValueError("Every video must have a known positive caption")
        if len(self.source_groups) != len(pool.video_ids) or not all(self.source_groups):
            raise ValueError("Require source-group provenance for every video")
        for array, length in [(self.video_embeddings, len(pool.video_ids)),
                              (self.text_embeddings, len(pool.text_ids))]:
            if (array.dtype != np.float32 or array.ndim != 2 or array.shape[0] != length
                    or not array.shape[1] or not np.isfinite(array).all()
                    or (np.linalg.norm(array, axis=-1) <= 0).any()):
                raise ValueError("Feature matrices must be finite nonzero float32 with exact pool axes")
        if self.video_embeddings.shape[1] != self.text_embeddings.shape[1]:
            raise ValueError("Image and text features must share the encoder output dimension")
        if self.relation_labels is not None:
            relations = self.relation_labels
            if (relations.shape != (len(pool.video_ids), len(pool.text_ids))
                    or not np.isin(relations, [-1, 0, 1]).all()):
                raise ValueError("Explicit relevance must identify every pair including unknowns")
            identity = self.relations(np.arange(len(pool.video_ids)),
                                      np.arange(len(pool.text_ids)), identity_only=True)
            if not (relations[identity == 1] == 1).all():
                raise ValueError("Captions of the same video must remain known positives")

    def relations(self, videos, texts, *, identity_only=False):
        if self.relation_labels is not None and not identity_only:
            return self.relation_labels[np.ix_(videos, texts)]
        video_ids = np.asarray(self.pool.video_ids)[videos]
        text_video_ids = np.asarray([q["video_id"] for q in self.pool.queries])[texts]
        return (video_ids[:, None] == text_video_ids[None, :]).astype(np.int8)


def validate_training_splits(train, validation):
    if train.pool.split != "train" or validation.pool.split != "validation":
        raise ValueError("Model selection must use validation, and gradients must use train")
    if (set(train.pool.video_ids) & set(validation.pool.video_ids)
            or set(train.pool.text_ids) & set(validation.pool.text_ids)
            or set(train.source_groups) & set(validation.source_groups)):
        raise ValueError("Train/validation video, caption or source-group overlap")
    for key in ("dataset", "artifact_kind", "encoder", "protocol_lock_sha256"):
        if train.provenance.get(key) != validation.provenance.get(key):
            raise ValueError(f"Train/validation cache provenance mismatch: {key}")
    for split in (train, validation):
        if split.provenance.get("pool_hashes") != split.pool.hashes():
            raise ValueError("Feature pool hashes do not match exact ordered identities")


def encoder_descriptor(config, encoder_run):
    checkpoint = encoder_run["checkpoint"]
    if (not encoder_run.get("checkpoint_inference") or not checkpoint.get("frozen")
            or config["frames"] != encoder_run["preprocess"]["num_frames_per_clip"]
            or config["decode_max_frame_side"]
            != encoder_run["preprocess"]["decode_max_frame_side"]):
        raise ValueError("Encoder run must identify the frozen encoder and matching preprocessing")
    return {
        "model_name": checkpoint["model_name"],
        "checkpoint_sha256": checkpoint["sha256"],
        "encoder_run_sha256": file_sha256(config["encoder_run_json"]),
        "preprocess_sha256": semantic_hash(encoder_run["preprocess"]),
        "sampling_sha256": semantic_hash({"frames": config["frames"],
                                         "reader": "VideoReader CFR strict whole-video uniform",
                                         "decode_max_frame_side": config["decode_max_frame_side"]}),
        "pooling": "frame_L2_valid_masked_mean_video_L2",
        "backbone_frozen": True,
        "crop_mode": config["crop_mode"],
        "frames": config["frames"],
    }


def load_cached_split(config, adapter, manifest, split, expected_encoder, *, started=None):
    """Pool one <=500-video shard at a time; preallocate ordered feature rows."""
    if (manifest.get("schema_version") != "msrvtt_frozen_cache_v1"
            or manifest.get("status") != "ready"
            or manifest.get("protocol_lock_sha256") != file_sha256(config["metadata_lock"])
            or manifest.get("encoder") != expected_encoder):
        raise ValueError("Cache schema/status/protocol/encoder provenance mismatch")
    started = time.monotonic() if started is None else started
    record = manifest["splits"][split]
    pool = adapter.pool(split, selected_video_ids=record["video_ids"])
    if record["pool_hashes"] != pool.hashes():
        raise ValueError("Cache candidate/query/positive/caption order hash mismatch")
    if len(pool.video_ids) != pool.full_video_count and not config["allow_subset"]:
        raise ValueError("Subset cache requires explicit allow_subset, never silently shrink the pool")
    if record["raw_caption_sha256"] != semantic_hash(list(pool.queries)):
        raise ValueError("Raw caption content hash mismatch")
    source_groups = tuple(adapter.source_urls[v] for v in pool.video_ids)
    if record.get("source_group_sha256") != semantic_hash(list(source_groups)):
        raise ValueError("Cache source-group provenance hash mismatch")
    if manifest.get("cohort_lock_sha256"):
        cohort_path = config.get("cohort_lock")
        if not cohort_path or file_sha256(cohort_path) != manifest["cohort_lock_sha256"]:
            raise ValueError("Subset/global-local cohort lock is missing or changed")
    if expected_encoder["crop_mode"] == "union":
        boxes_path = config.get("boxes")
        if isinstance(boxes_path, dict):
            boxes_path = boxes_path.get(split)
        if (not boxes_path or file_sha256(boxes_path) != record.get("boxes_sha256")
                or record.get("box_coverage_verified") is not True):
            raise ValueError("Union features require verified box provenance and complete coverage")
    dimension = manifest["dimension"]
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Cache dimension must be a positive integer")
    allocated = 4 * dimension * (len(pool.video_ids) + len(pool.text_ids))
    checked_shards = []
    for shard in record["shards"]:
        relative = Path(shard["path"])
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("Shard path must remain inside cache_dir")
        path = Path(config["cache_dir"]) / relative
        if not path.resolve().is_relative_to(Path(config["cache_dir"]).resolve()):
            raise ValueError("Resolved shard path escaped cache_dir")
        if file_sha256(path) != shard["sha256"]:
            raise ValueError("Cache shard file hash mismatch")
        with zipfile.ZipFile(path) as archive:
            decompressed = sum(info.file_size for info in archive.infolist())
        shard_videos = shard.get("videos", min(500, len(pool.video_ids)))
        if type(shard_videos) is not int or not 1 <= shard_videos <= 500:
            raise ValueError("Declared shard videos must be within 1..500")
        frame_bytes = 4 * shard_videos * expected_encoder["frames"] * dimension
        # NPZ array/decompression buffers plus float64 conversion, normalized
        # frames, masked multiply and pooled-vector temporaries in mean pooling.
        working = 2 * decompressed + 6 * frame_bytes + 32 * shard_videos * dimension
        checked_shards.append((shard, path, working))
        enforce_resources(config, started)
    ensure_allocation(config, started,
                      allocated + max((working for _, _, working in checked_shards), default=0),
                      "feature cache and decompressed/pooling shard")
    videos = np.empty((len(pool.video_ids), dimension), dtype=np.float32)
    texts = np.empty((len(pool.text_ids), dimension), dtype=np.float32)
    vi = {key: i for i, key in enumerate(pool.video_ids)}
    ti = {key: i for i, key in enumerate(pool.text_ids)}
    text_video = {q["text_id"]: q["video_id"] for q in pool.queries}
    seen_video, seen_text = set(), set()
    for shard, path, working in checked_shards:
        ensure_allocation(config, started, working, "decompressed/pooling shard")
        with np.load(path, allow_pickle=False) as stored:
            video_ids, text_ids = stored["video_ids"], stored["text_ids"]
            if (video_ids.dtype.kind != "U" or text_ids.dtype.kind != "U"
                    or video_ids.ndim != 1 or text_ids.ndim != 1
                    or not 1 <= len(video_ids) <= 500
                    or len(set(video_ids.tolist())) != len(video_ids)
                    or len(set(text_ids.tolist())) != len(text_ids)):
                raise ValueError("Shard IDs require unique Unicode arrays and <=500 videos")
            vset, tset = set(video_ids.tolist()), set(text_ids.tolist())
            expected_texts = {q["text_id"] for q in pool.queries if q["video_id"] in vset}
            if (not vset <= set(vi) or tset != expected_texts
                    or vset & seen_video or tset & seen_text
                    or stored["text_video_ids"].tolist() != [text_video[t] for t in text_ids]):
                raise ValueError("Duplicate/foreign IDs or incomplete raw captions in shard")
            frames, mask = stored["frame_embeddings"], stored["valid_frame_mask"]
            times, text_features = stored["timestamps_s"], stored["text_embeddings"]
            if (frames.dtype != np.float32 or mask.dtype != np.bool_
                    or times.dtype != np.float64 or text_features.dtype != np.float32
                    or frames.shape != (len(video_ids), expected_encoder["frames"], dimension)
                    or text_features.shape != (len(text_ids), dimension)
                    or times.shape != frames.shape[:2] or mask.shape != times.shape
                    or not np.isfinite(times).all() or (times[mask] < 0).any()
                    or any(np.any(np.diff(row[valid]) <= 0)
                           for row, valid in zip(times, mask, strict=True))):
                raise ValueError("Shard feature dtype/shape/timestamps mismatch")
            pooled = mean_pool_valid_frames(frames, mask).astype(np.float32)
            if (not np.allclose(np.linalg.norm(frames[mask], axis=-1), 1, atol=1e-5)
                    or not np.allclose(np.linalg.norm(text_features, axis=-1), 1, atol=1e-5)):
                raise ValueError("Cache requires L2-normalized valid frame and text embeddings")
            videos[[vi[v] for v in video_ids]] = pooled
            texts[[ti[t] for t in text_ids]] = text_features
        seen_video.update(vset)
        seen_text.update(tset)
        del video_ids, text_ids, frames, mask, times, text_features, pooled
        enforce_resources(config, started)
    if seen_video != set(vi) or seen_text != set(ti):
        raise ValueError("Cache shards do not cover the exact complete identified pool")
    provenance = {
        "dataset": "MSR-VTT", "artifact_kind": "frozen_encoder_features",
        "encoder": expected_encoder, "protocol_lock_sha256": manifest["protocol_lock_sha256"],
        "pool_hashes": pool.hashes(), "raw_caption_sha256": record["raw_caption_sha256"],
        "cache_manifest_sha256": file_sha256(Path(config["cache_dir"]) / "cache_manifest.json"),
        "scope": "full_locked_pool" if len(pool.video_ids) == pool.full_video_count else "subset",
        "relevance_policy": "locked_complete_video_identity",
        "cohort_lock_sha256": manifest.get("cohort_lock_sha256"),
    }
    return FrozenFeatureSplit(pool, videos, texts,
                              source_groups, provenance)

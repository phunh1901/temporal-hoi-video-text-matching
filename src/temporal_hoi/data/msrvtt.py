"""Hash-checked MSR-VTT metadata adapter and explicitly identified retrieval pools.

This module does not download media, infer missing labels, or select models from
the official test split. A score matrix is meaningful only with its ordered IDs.
"""

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from temporal_hoi.evaluation.metrics import compute_retrieval_metrics

SPLIT_NAMES = {
    "train": "research_train8100_sourcegroup_v2",
    "validation": "research_validation900_sourcegroup_v2",
    "test": "official_test1000",
    "train9k": "official_train9k",
    "test_all_captions": "official_test1000",
}
CAPTION_GROUPS = {
    "train": "research_train8100_sourcegroup_v2",
    "validation": "research_validation900_sourcegroup_v2",
    "train9k": "official_train9k",
    "test_all_captions": "official_test1000_all_captions_extension",
}


def semantic_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_hash(value, expected, label):
    if semantic_hash(value) != expected:
        raise ValueError(f"Locked {label} hash mismatch")


def _read_checked_archive(path, lock):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Metadata archive exceeds 8 MiB limit")
    if file_sha256(path) != lock["upstream"]["archive_sha256"]:
        raise ValueError("Locked metadata archive hash mismatch")
    output = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if sum(member.file_size for member in members) > 64 * 1024 * 1024:
            raise ValueError("Metadata archive exceeds 64 MiB expanded limit")
        for name, provenance in lock["upstream"]["members"].items():
            matches = [m for m in members if m.filename == provenance["archive_member"]]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous or missing archive member: {name}")
            raw = archive.read(matches[0])
            if hashlib.sha256(raw).hexdigest() != provenance["sha256"]:
                raise ValueError(f"Locked archive member hash mismatch: {name}")
            decoded = raw.decode("utf-8-sig")
            output[name] = (json.loads(decoded) if name.endswith(".json")
                            else list(csv.DictReader(io.StringIO(decoded))))
    return output


@dataclass(frozen=True)
class RetrievalPool:
    split: str
    video_ids: tuple[str, ...]
    queries: tuple[dict, ...]
    full_video_count: int
    full_text_count: int

    @property
    def text_ids(self):
        return tuple(row["text_id"] for row in self.queries)

    @property
    def positive_pairs(self):
        return [[row["text_id"], row["video_id"]] for row in self.queries]

    def hashes(self):
        return {
            "video_candidate_order_sha256": semantic_hash(list(self.video_ids)),
            "text_query_order_sha256": semantic_hash(list(self.text_ids)),
            "query_rows_sha256": semantic_hash(list(self.queries)),
            "positive_mapping_sha256": semantic_hash(self.positive_pairs),
        }

    def positive_indices(self):
        video_indices = {video_id: index for index, video_id in enumerate(self.video_ids)}
        text_to_video = [[video_indices[row["video_id"]]] for row in self.queries]
        video_to_text = [[] for _ in self.video_ids]
        for index, row in enumerate(self.queries):
            video_to_text[video_indices[row["video_id"]]].append(index)
        return text_to_video, video_to_text


class MSRVTTAdapter:
    """Metadata, ordered captions and source-group validation from a W03 lock."""

    def __init__(self, lock, captions, source_urls):
        if lock.get("schema_version") != "msrvtt_w03_v2":
            raise ValueError("Require source-group v2 metadata lock")
        self.lock, self.captions, self.source_urls = lock, captions, source_urls
        _assert_hash(captions, lock["captions"]["mapping_sha256"], "caption mapping")
        if len({r["text_id"] for r in captions}) != len(captions):
            raise ValueError("Duplicate caption row IDs")
        for name in set(SPLIT_NAMES.values()):
            ids = lock["video_ids"][name]
            if not ids or len(set(ids)) != len(ids):
                raise ValueError("Split must contain nonempty unique video IDs")
            if any(not isinstance(v, str) or Path(v).name != v or "/" in v or "\\" in v
                   for v in ids):
                raise ValueError("Video IDs must be plain filenames without path components")
            _assert_hash(ids, lock["video_id_sha256"][name], f"{name} IDs")
        train = set(lock["video_ids"][SPLIT_NAMES["train"]])
        validation = set(lock["video_ids"][SPLIT_NAMES["validation"]])
        official_train = set(lock["video_ids"][SPLIT_NAMES["train9k"]])
        test = set(lock["video_ids"][SPLIT_NAMES["test"]])
        if train & validation or train | validation != official_train or official_train & test:
            raise ValueError("Train/validation/test video split overlap or incomplete partition")
        if set(source_urls) != official_train | test or any(not u for u in source_urls.values()):
            raise ValueError("Missing source URL provenance")
        if {source_urls[v] for v in train} & {source_urls[v] for v in validation}:
            raise ValueError("Source-group v2 train/validation URL overlap")
        known_videos = official_train | test
        if any(row["video_id"] not in known_videos for row in captions):
            raise ValueError("Caption refers to unknown video")
        for split, group in CAPTION_GROUPS.items():
            ids = set(lock["video_ids"][SPLIT_NAMES[split]])
            rows = [row for row in captions if row["video_id"] in ids]
            expected = lock["captions"]["groups"][group]
            if len(rows) != expected["count"] or {row["video_id"] for row in rows} != ids:
                raise ValueError(f"Incomplete caption coverage for {split}")
            _assert_hash(rows, expected["sha256"], f"{split} captions")
        official = lock["official_test"]
        _assert_hash(official["video_candidates"], official["candidate_order_sha256"], "test pool")
        _assert_hash(official["text_queries"], official["caption_queries_sha256"], "test queries")
        _assert_hash(official["positive_pairs"], official["positive_mapping_sha256"], "test positives")
        if official["video_candidates"] != lock["video_ids"][SPLIT_NAMES["test"]]:
            raise ValueError("Official candidate order differs from test split")
        if (len(official["text_queries"]) != len(test)
                or [q["video_id"] for q in official["text_queries"]]
                != official["video_candidates"]
                or len({q["text_id"] for q in official["text_queries"]}) != len(test)
                or [[q["text_id"], q["video_id"]] for q in official["text_queries"]]
                != official["positive_pairs"]):
            raise ValueError("Official singleton positive mapping is inconsistent")

    @classmethod
    def from_files(cls, archive_path, lock_path):
        lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        members = _read_checked_archive(archive_path, lock)
        annotations = members["MSRVTT_data.json"]
        captions = [{"text_id": f"msrvtt_sen_{row.get('sen_id', f'row_{index:06d}')}",
                     "video_id": row["video_id"], "caption": row["caption"],
                     "annotation_row": index}
                    for index, row in enumerate(annotations["sentences"])]
        source_urls = {row["video_id"]: row["url"] for row in annotations["videos"]}
        adapter = cls(lock, captions, source_urls)
        if [r["video_id"] for r in members["MSRVTT_train.9k.csv"]] != lock[
                "video_ids"]["official_train9k"]:
            raise ValueError("Train CSV does not match locked train IDs")
        queries = lock["official_test"]["text_queries"]
        test_rows = members["MSRVTT_JSFUSION_test.csv"]
        if (len(test_rows) != len(queries)
                or [(r["video_id"], r["sentence"]) for r in test_rows]
                != [(r["video_id"], r["caption"]) for r in queries]):
            raise ValueError("Official CSV query text or order differs from lock")
        return adapter

    def pool(self, split, *, selected_video_ids=None):
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}; choose {list(SPLIT_NAMES)}")
        ids = self.lock["video_ids"][SPLIT_NAMES[split]]
        selected = set(ids)
        if selected_video_ids is not None:
            selection = list(selected_video_ids)
            selected = set(selection)
            if not selected or len(selected) != len(selection) or not selected <= set(ids):
                raise ValueError("Subset must contain nonempty unique IDs from the requested split")
        candidates = tuple(v for v in ids if v in selected)
        all_ids = set(ids)
        rows = (self.lock["official_test"]["text_queries"] if split == "test"
                else [r for r in self.captions if r["video_id"] in all_ids])
        queries = tuple(dict(row) for row in rows if row["video_id"] in selected)
        return RetrievalPool(split, candidates, queries, len(ids), len(rows))

    def media_coverage(self, split, media_root, *, suffix=".mp4"):
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("Media suffix must be a plain extension")
        pool = self.pool(split)
        root = Path(media_root)
        present = [v for v in pool.video_ids
                   if (root / f"{v}{suffix}").is_file()
                   and (root / f"{v}{suffix}").stat().st_size > 0]
        present_set = set(present)
        return {"split": split, "media_root": str(root.resolve()), "suffix": suffix,
                "expected_videos": len(pool.video_ids), "present_nonempty_files": len(present),
                "file_presence_fraction": len(present) / len(pool.video_ids),
                "present_video_ids": present,
                "missing_video_ids": [v for v in pool.video_ids if v not in present_set],
                "decode_checked": False,
                "scope": "Exact ID filenames only; existence does not establish decodability"}


def mean_pool_valid_frames(frame_embeddings, valid_mask):
    """CLIP4Clip meanP contract: frame L2 -> masked mean -> video L2.

    Masked padding may be zero, but valid frames and complete videos may not.
    This pools supplied features; it does not claim to reproduce frame sampling.
    """
    frames = np.asarray(frame_embeddings, dtype=np.float64)
    mask = np.asarray(valid_mask)
    if (frames.ndim != 3 or mask.shape != frames.shape[:2]
            or not all(frames.shape) or not np.isfinite(frames).all()
            or not np.isin(mask, [0, 1]).all()):
        raise ValueError("Frame features require finite [videos,frames,dim] and binary valid mask")
    mask = mask.astype(bool)
    if not mask.any(axis=1).all():
        raise ValueError("Every video requires at least one valid frame")
    norms = np.linalg.norm(frames, axis=2)
    if (norms[mask] <= 0).any():
        raise ValueError("Valid frames may not have zero embedding")
    normalized = np.divide(frames, norms[..., None], out=np.zeros_like(frames),
                           where=(norms > 0)[..., None])
    mean = (normalized * mask[..., None]).sum(axis=1) / mask.sum(axis=1)[:, None]
    video_norms = np.linalg.norm(mean, axis=1)
    if (video_norms <= 0).any():
        raise ValueError("Mean-pooled video embedding may not be zero")
    return mean / video_norms[:, None]


def scores_from_embeddings(text_embeddings, video_embeddings, *, scale=1.0):
    """L2-normalized cosine, [text, video]; reject padding/zero vectors explicitly."""
    text = np.asarray(text_embeddings, dtype=np.float64)
    video = np.asarray(video_embeddings, dtype=np.float64)
    if (text.ndim != 2 or video.ndim != 2 or not text.shape[0] or not video.shape[0]
            or not text.shape[1] or text.shape[1] != video.shape[1]
            or not np.isfinite(text).all() or not np.isfinite(video).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("Embeddings need nonempty compatible finite 2D arrays and positive scale")
    text_norm, video_norm = np.linalg.norm(text, axis=1), np.linalg.norm(video, axis=1)
    if (text_norm <= 0).any() or (video_norm <= 0).any():
        raise ValueError("Zero embedding is not a valid text/video; filter missing media before scoring")
    return (text / text_norm[:, None]) @ (video / video_norm[:, None]).T * scale


def evaluate_pool_scores(pool, scores, *, text_ids, video_ids):
    """Validate both axes before scoring; do not silently reorder or drop IDs."""
    matrix = np.asarray(scores, dtype=float)
    if (tuple(text_ids) != pool.text_ids or tuple(video_ids) != pool.video_ids
            or matrix.shape != (len(pool.text_ids), len(pool.video_ids))):
        raise ValueError("Score axes/order/shape differ from the identified candidate pool")
    positives_text, positives_video = pool.positive_indices()
    return {
        "text_to_video": compute_retrieval_metrics(matrix, positives_text),
        "video_to_text": compute_retrieval_metrics(matrix.T, positives_video),
        "scope": ("full_locked_pool" if len(pool.video_ids) == pool.full_video_count
                  else "subset_not_comparable_to_full_benchmark"),
        "recall_scale": "fraction", "matrix_axes": "text,video", "pool_hashes": pool.hashes(),
    }


def internvideo_test_annotations(pool, *, suffix=".mp4"):
    """Raw JSON consumed by InternVideo2 load_anno(media_type='video').

    The upstream loader applies pre_text after import. This conversion preserves
    raw locked caption identity; its text processing must be logged by the model.
    """
    if pool.split != "test" or len(pool.queries) != len(pool.video_ids):
        raise ValueError("InternVideo official adapter requires the singleton test query pool")
    return [{"video": f"{q['video_id']}{suffix}", "caption": q["caption"]}
            for q in pool.queries]

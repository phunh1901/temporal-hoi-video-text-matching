"""Protocol safeguards on tiny labeled fixtures, never benchmark model scores."""

import copy
import hashlib
import importlib.util
import io
import json
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from temporal_hoi.data.msrvtt import (
    CAPTION_GROUPS,
    SPLIT_NAMES,
    MSRVTTAdapter,
    evaluate_pool_scores,
    internvideo_test_annotations,
    mean_pool_valid_frames,
    scores_from_embeddings,
    semantic_hash,
)


def fixture_records():
    splits = {"official_train9k": ["video0", "video1", "video2"],
              "research_train8100_sourcegroup_v2": ["video0", "video1"],
              "research_validation900_sourcegroup_v2": ["video2"],
              "official_test1000": ["video3", "video4"]}
    captions = [{"text_id": f"msrvtt_sen_{i * 2 + j}", "video_id": f"video{i}",
                 "caption": "same text" if i == 0 else f"caption {i}-{j}",
                 "annotation_row": i * 2 + j} for i in range(5) for j in range(2)]
    queries = [{"text_id": "msrvtt_jsfusion_0000", "video_id": "video3",
                "caption": "CSV spelling differs", "candidate_index": 0,
                "exact_match_in_full_annotations": False},
               {"text_id": "msrvtt_jsfusion_0001", "video_id": "video4",
                "caption": "caption 4-0", "candidate_index": 1,
                "exact_match_in_full_annotations": True}]
    pairs = [[q["text_id"], q["video_id"]] for q in queries]
    groups = {}
    for split, group in CAPTION_GROUPS.items():
        selected = set(splits[SPLIT_NAMES[split]])
        rows = [c for c in captions if c["video_id"] in selected]
        groups[group] = {"count": len(rows), "sha256": semantic_hash(rows)}
    lock = {"schema_version": "msrvtt_w03_v2", "video_ids": splits,
            "video_id_sha256": {k: semantic_hash(v) for k, v in splits.items()},
            "captions": {"mapping_sha256": semantic_hash(captions), "groups": groups},
            "official_test": {"video_candidates": splits["official_test1000"],
                              "text_queries": queries, "positive_pairs": pairs,
                              "candidate_order_sha256": semantic_hash(splits["official_test1000"]),
                              "caption_queries_sha256": semantic_hash(queries),
                              "positive_mapping_sha256": semantic_hash(pairs)}}
    urls = {f"video{i}": f"source-{i}" for i in range(5)}
    urls["video1"] = urls["video0"]
    urls["video3"] = urls["video0"]  # Inherited official train/test source overlap is preserved.
    return lock, captions, urls


def adapter_fixture():
    return MSRVTTAdapter(*fixture_records())


def fixture_files(tmp_path):
    lock, captions, urls = fixture_records()
    annotations = {"videos": [{"video_id": v, "url": url} for v, url in urls.items()],
                   "sentences": [{"video_id": c["video_id"], "caption": c["caption"],
                                  "sen_id": index} for index, c in enumerate(captions)]}
    raw = {
        "MSRVTT_train.9k.csv": b"video_id\nvideo0\nvideo1\nvideo2\n",
        "MSRVTT_JSFUSION_test.csv": (
            b"video_id,sentence\nvideo3,CSV spelling differs\nvideo4,caption 4-0\n"),
        "MSRVTT_data.json": json.dumps(annotations).encode(),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in raw.items():
            archive.writestr(name, content)
    lock["upstream"] = {
        "archive_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "members": {name: {"archive_member": name, "sha256": hashlib.sha256(content).hexdigest()}
                    for name, content in raw.items()},
    }
    archive_path, lock_path = tmp_path / "metadata.zip", tmp_path / "lock.json"
    archive_path.write_bytes(buffer.getvalue())
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return archive_path, lock_path


def test_checked_archive_loads_and_preserves_csv_mismatches(tmp_path):
    adapter = MSRVTTAdapter.from_files(*fixture_files(tmp_path))
    assert adapter.pool("test").queries[0]["caption"] == "CSV spelling differs"
    assert not adapter.pool("test").queries[0]["exact_match_in_full_annotations"]
    assert adapter.pool("test").video_ids == ("video3", "video4")


def test_changed_archive_or_member_provenance_is_rejected(tmp_path):
    archive_path, lock_path = fixture_files(tmp_path)
    lock = json.loads(lock_path.read_text())
    lock["upstream"]["members"]["MSRVTT_data.json"]["sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="member hash mismatch"):
        MSRVTTAdapter.from_files(archive_path, lock_path)
    archive_path.write_bytes(archive_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        MSRVTTAdapter.from_files(archive_path, lock_path)


def test_duplicate_text_rows_remain_distinct_with_multi_positive_reverse_mapping():
    pool = adapter_fixture().pool("train")
    assert len(pool.queries) == 4
    assert pool.queries[0]["caption"] == pool.queries[1]["caption"]
    assert pool.text_ids[0] != pool.text_ids[1]
    text_to_video, video_to_text = pool.positive_indices()
    assert text_to_video == [[0], [0], [1], [1]]
    assert video_to_text == [[0, 1], [2, 3]]


@pytest.mark.parametrize("change", ["caption", "video_order", "positive_mapping"])
def test_tampered_semantic_lock_is_rejected(change):
    lock, captions, urls = fixture_records()
    if change == "caption":
        captions[0]["caption"] = "changed"
    elif change == "video_order":
        lock["video_ids"]["official_train9k"].reverse()
    else:
        lock["official_test"]["positive_pairs"][0][1] = "video4"
    with pytest.raises(ValueError, match="hash mismatch"):
        MSRVTTAdapter(lock, captions, urls)


def test_source_group_overlap_is_rejected_even_if_video_ids_are_distinct():
    lock, captions, urls = fixture_records()
    urls["video2"] = urls["video0"]
    with pytest.raises(ValueError, match="URL overlap"):
        MSRVTTAdapter(lock, captions, urls)


def test_media_presence_is_not_decode_coverage_and_empty_files_are_missing(tmp_path):
    (tmp_path / "video3.mp4").write_bytes(b"not yet decoded")
    (tmp_path / "video4.mp4").write_bytes(b"")
    coverage = adapter_fixture().media_coverage("test", tmp_path)
    assert coverage["present_nonempty_files"] == 1
    assert coverage["file_presence_fraction"] == 0.5
    assert coverage["missing_video_ids"] == ["video4"]
    assert not coverage["decode_checked"]


def test_subset_filters_queries_by_positive_video_and_keeps_original_candidate_order():
    adapter = adapter_fixture()
    subset = adapter.pool("train", selected_video_ids=["video1"])
    assert subset.video_ids == ("video1",) and len(subset.queries) == 2
    assert subset.positive_indices() == ([[0], [0]], [[0, 1]])
    reordered_request = adapter.pool("test", selected_video_ids=["video4", "video3"])
    assert reordered_request.video_ids == ("video3", "video4")
    with pytest.raises(ValueError, match="nonempty unique IDs"):
        adapter.pool("test", selected_video_ids=[])
    with pytest.raises(ValueError, match="nonempty unique IDs"):
        adapter.pool("test", selected_video_ids=["video0"])


def test_explicit_score_axes_and_two_directions_match_hand_calculation():
    pool = adapter_fixture().pool("test")
    scores = [[0.9, 0.1], [0.8, 0.7]]
    result = evaluate_pool_scores(pool, scores, text_ids=pool.text_ids, video_ids=pool.video_ids)
    assert result["text_to_video"]["R@1"] == 0.5
    assert result["text_to_video"]["MedR"] == 1.5
    assert result["video_to_text"]["R@1"] == 1.0
    assert result["scope"] == "full_locked_pool"
    with pytest.raises(ValueError, match="axes/order/shape"):
        evaluate_pool_scores(pool, scores, text_ids=pool.text_ids[::-1], video_ids=pool.video_ids)
    with pytest.raises(ValueError, match="axes/order/shape"):
        evaluate_pool_scores(pool, [[1]], text_ids=pool.text_ids, video_ids=pool.video_ids)
    subset = adapter_fixture().pool("test", selected_video_ids=["video3"])
    result = evaluate_pool_scores(subset, [[1]], text_ids=subset.text_ids, video_ids=subset.video_ids)
    assert result["scope"] == "subset_not_comparable_to_full_benchmark"


def test_normalized_embeddings_reject_zero_padding_and_incompatible_dimensions():
    scores = scores_from_embeddings([[2, 0], [0, 3]], [[4, 0], [0, 5]], scale=2)
    np.testing.assert_array_equal(scores, [[2, 0], [0, 2]])
    with pytest.raises(ValueError, match="Zero embedding"):
        scores_from_embeddings([[0, 0]], [[1, 2]])
    with pytest.raises(ValueError, match="compatible finite"):
        scores_from_embeddings([[1, 2]], [[1, 2, 3]])


def test_meanp_normalizes_each_valid_frame_before_pooling_and_excludes_padding():
    features = [[[3, 0], [0, 8], [99, -99]], [[4, 0], [0, 0], [0, 0]]]
    pooled = mean_pool_valid_frames(features, [[1, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(pooled, [[2 ** -0.5, 2 ** -0.5], [1, 0]])
    with pytest.raises(ValueError, match="at least one valid frame"):
        mean_pool_valid_frames(features, [[1, 1, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="zero embedding"):
        mean_pool_valid_frames([[[0, 0]]], [[1]])
    with pytest.raises(ValueError, match="binary valid mask"):
        mean_pool_valid_frames([[[1, 0]]], [[0.5]])


def test_internvideo_conversion_preserves_locked_singleton_queries():
    pool = adapter_fixture().pool("test")
    rows = internvideo_test_annotations(pool)
    assert rows[0] == {"video": "video3.mp4", "caption": "CSV spelling differs"}
    assert [r["caption"] for r in rows] == [q["caption"] for q in pool.queries]
    with pytest.raises(ValueError, match="singleton test"):
        internvideo_test_annotations(adapter_fixture().pool("test_all_captions"))


def test_consistent_rehashed_but_wrong_singleton_mapping_is_rejected():
    lock, captions, urls = copy.deepcopy(fixture_records())
    lock["official_test"]["text_queries"][0]["video_id"] = "video4"
    lock["official_test"]["caption_queries_sha256"] = semantic_hash(
        lock["official_test"]["text_queries"])
    with pytest.raises(ValueError, match="singleton positive mapping"):
        MSRVTTAdapter(lock, captions, urls)


def _cli_module():
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_msrvtt.py"
    spec = importlib.util.spec_from_file_location("msrvtt_cli_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_args(tmp_path):
    archive, lock = fixture_files(tmp_path)
    return Namespace(archive=archive, lock=lock, media_root=tmp_path / "media",
                     media_suffix=".mp4", output=tmp_path / "audit.json", scores=None,
                     provenance=None, checkpoint=None, split="test", allow_subset=False,
                     score_scale=1.0, official_metrics=tmp_path / "not-used.py")


def test_cli_metadata_only_does_not_create_quality_metrics(tmp_path):
    report = _cli_module().run(_cli_args(tmp_path))
    assert report["benchmark_metrics"] is None
    assert not report["benchmark_inference_run"]
    assert not report["official_test_access"]["quality_scores_computed"]
    assert report["media_coverage"]["test"]["present_nonempty_files"] == 0
    assert (tmp_path / "m2_internvideo_jsfusion_test.json").is_file()


def test_cli_relocated_report_keeps_explicit_source_provenance(tmp_path):
    args = _cli_args(tmp_path)
    args.output = tmp_path / "new_review" / "audit.json"
    args.internvideo_source_dir = tmp_path / "pinned_sources"
    args.internvideo_source_dir.mkdir()
    for name in ["internvideo_ret_dataset.py", "internvideo_dataset_utils.py"]:
        (args.internvideo_source_dir / name).write_text("# pinned source fixture\n")
    report = _cli_module().run(args)
    snapshots = report["internvideo_annotation"]["source_snapshots"]
    assert all(row["exists"] and len(row["sha256"]) == 64 for row in snapshots)
    assert all(Path(row["path"]).parent == args.internvideo_source_dir for row in snapshots)
    assert report["benchmark_metrics"] is None


def test_cli_external_meanp_artifact_keeps_inference_unclaimed(tmp_path, monkeypatch):
    module, args = _cli_module(), _cli_args(tmp_path)
    pool = adapter_fixture().pool("test")
    args.scores = tmp_path / "scores.npz"
    args.provenance = tmp_path / "scores.json"
    args.provenance.write_text(json.dumps({"model_name": "fixture",
        "checkpoint_sha256": "1" * 64, "preprocess": "fixture features",
        "pooling": "meanP", "score_type": "cosine", "training_regime": "fixture",
        "artifact_kind": "test_fixture"}))
    np.savez(args.scores, text_ids=np.array(pool.text_ids), video_ids=np.array(pool.video_ids),
             text_embeddings=np.array([[1, 0], [0, 1]]),
             frame_embeddings=np.array([[[2, 0], [0, 0]], [[0, 3], [0, 0]]]),
             valid_frame_mask=np.array([[1, 0], [1, 0]]))
    monkeypatch.setattr(module, "_official_metrics", lambda path, scores: {"scope": "fixture"})
    report = module.run(args)
    assert report["benchmark_metrics"]["text_to_video"]["R@1"] == 1.0
    assert report["execution_level"] == "external_score_artifact_evaluation"
    assert not report["benchmark_inference_run"] and not report["benchmark_training_run"]
    assert not report["score_artifact"]["local_checkpoint_hash_verified"]
    args.checkpoint = tmp_path / "different.pt"
    args.checkpoint.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        module.run(args)

"""Leakage and exact-caption safeguards for the W03 metadata lock."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lock_msrvtt_protocol.py"
SPEC = importlib.util.spec_from_file_location("msrvtt_protocol", SCRIPT)
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


def metadata_fixture():
    return ([{"video_id": "v0"}, {"video_id": "v1"}],
            [{"video_id": "v2", "sentence": "same sentence"}],
            {"videos": [{"video_id": f"v{i}"} for i in range(3)], "sentences": [
                {"video_id": "v0", "sen_id": 0, "caption": "same sentence"},
                {"video_id": "v0", "sen_id": 1, "caption": "same sentence"},
                {"video_id": "v1", "sen_id": 2, "caption": "a second caption"},
                {"video_id": "v2", "sen_id": 3, "caption": "same sentence"}]})


def test_video_level_development_split_is_disjoint_and_repeatable():
    ids = [f"video{i}" for i in range(9000)]
    train, validation = PROTOCOL.split_development(ids)
    assert len(train) == 8100 and len(validation) == 900
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(ids)
    assert (train, validation) == PROTOCOL.split_development(ids)
    assert validation != PROTOCOL.split_development(ids, seed=43)[1]
    assert train == [v for v in ids if v not in set(validation)]


def test_source_group_split_keeps_siblings_together_and_reaches_exact_quota():
    ids = [f"v{i}" for i in range(8)]
    urls = ["same0", "same0", "same1", "same1", "same1", "same2", "same3", "same4"]
    rows = [{"video_id": video_id, "url": url} for video_id, url in zip(ids, urls, strict=True)]
    train, validation = PROTOCOL.split_development_source_groups(ids, rows, validation_count=3)
    assert len(train) == 5 and len(validation) == 3
    lookup = dict(zip(ids, urls, strict=True))
    assert not {lookup[v] for v in train} & {lookup[v] for v in validation}
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(ids)
    assert (train, validation) == PROTOCOL.split_development_source_groups(
        ids, rows, validation_count=3)
    assert validation == [v for v in ids if v in set(validation)]


def test_source_group_split_rejects_missing_provenance_and_unfillable_quota():
    with pytest.raises(ValueError, match="nonempty URL"):
        PROTOCOL.split_development_source_groups(["v0", "v1"], [{"video_id": "v0", "url": "u"}],
                                                 validation_count=1)
    ids = ["v0", "v1", "v2", "v3"]
    rows = [{"video_id": v, "url": "u0" if i < 2 else "u1"} for i, v in enumerate(ids)]
    with pytest.raises(ValueError, match="Whole source groups"):
        PROTOCOL.split_development_source_groups(ids, rows, validation_count=3)


def test_duplicate_caption_strings_preserve_ids_and_positive_mapping():
    train, test, captions, queries, counts = PROTOCOL.validate_metadata(
        *metadata_fixture(), train_count=2, test_count=1)
    assert train == ["v0", "v1"] and test == ["v2"]
    assert len(captions) == 4 and counts["v0"] == 2
    assert captions[0]["text_id"] != captions[1]["text_id"]
    assert queries[0]["video_id"] == "v2" and queries[0]["candidate_index"] == 0
    assert PROTOCOL.semantic_hash(captions) != PROTOCOL.semantic_hash(captions[::-1])


def test_official_query_text_is_kept_even_when_full_annotation_text_differs():
    train, test, annotations = metadata_fixture()
    test[0]["sentence"] = "the official csv may supply a different spelling"
    result = PROTOCOL.validate_metadata(train, test, annotations, train_count=2, test_count=1)
    assert result[3][0]["caption"] == test[0]["sentence"]
    assert not result[3][0]["exact_match_in_full_annotations"]


@pytest.mark.parametrize("error", ["overlap", "missing_caption", "unknown_video", "empty_sentence",
                                   "duplicate_id", "duplicate_video"])
def test_invalid_annotation_or_leaking_split_is_rejected(error):
    train, test, annotations = metadata_fixture()
    if error == "overlap":
        test[0]["video_id"] = "v0"
    elif error == "missing_caption":
        annotations["sentences"] = annotations["sentences"][:-1]
    elif error == "unknown_video":
        annotations["sentences"][0]["video_id"] = "other"
    elif error == "empty_sentence":
        test[0]["sentence"] = ""
    elif error == "duplicate_id":
        annotations["sentences"][1]["sen_id"] = 0
    elif error == "duplicate_video":
        train[1]["video_id"] = "v0"
    with pytest.raises(ValueError):
        PROTOCOL.validate_metadata(train, test, annotations, train_count=2, test_count=1)

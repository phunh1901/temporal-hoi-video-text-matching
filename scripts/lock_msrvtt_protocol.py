"""Lock CLIP4Clip 9k/1k metadata and a video-level development split.

No video, model, training, feature extraction, or benchmark scoring is performed.
Input is the small upstream msrvtt_data.zip release; members are read in memory,
never extracted. Re-running the script must produce identical semantic hashes.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

CLIP4CLIP_COMMIT = "508ffa3de39ba0563a03199c440ab602a72e9b6f"
ARCHIVE_URL = "https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip"
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


def semantic_hash(value):
    """SHA256 of UTF-8 canonical JSON (order matters for arrays)."""
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def split_development(train_ids, seed=42, validation_count=900):
    """Shuffle the locked upstream order once; output each subset in that order."""
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Duplicate training video IDs")
    if not 0 < validation_count < len(train_ids):
        raise ValueError("Validation count must leave nonempty train and validation")
    shuffled = list(train_ids)
    random.Random(seed).shuffle(shuffled)
    val_set = set(shuffled[:validation_count])
    return ([v for v in train_ids if v not in val_set], [v for v in train_ids if v in val_set])


def split_development_source_groups(train_ids, video_rows, seed=42, validation_count=900):
    """Select complete exact-URL groups; never split siblings to hit a quota.

    Groups retain first occurrence in upstream train ID order before seed42
    shuffle. Greedily accept only groups that fit the remaining video quota.
    If no exact solution is reached, fail rather than silently split a group.
    """
    if len(set(train_ids)) != len(train_ids) or not 0 < validation_count < len(train_ids):
        raise ValueError("Invalid source-group training IDs or validation quota")
    urls = {row["video_id"]: row.get("url") for row in video_rows}
    groups = {}
    for video_id in train_ids:
        url = urls.get(video_id)
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Source-group independence requires a nonempty URL for every video")
        groups.setdefault(url, []).append(video_id)
    source_order = list(groups)
    random.Random(seed).shuffle(source_order)
    validation_set = set()
    for url in source_order:
        group = groups[url]
        remaining = validation_count - len(validation_set)
        if len(group) <= remaining:
            validation_set.update(group)
        if len(validation_set) == validation_count:
            break
    if len(validation_set) != validation_count:
        raise ValueError(f"Whole source groups reached {len(validation_set)}, not {validation_count}")
    return ([v for v in train_ids if v not in validation_set],
            [v for v in train_ids if v in validation_set])


def validate_metadata(train_rows, test_rows, annotations, *, train_count=9000, test_count=1000):
    """Reject overlap/incomplete mappings; keep rows even if text strings duplicate."""
    train_ids = [row["video_id"] for row in train_rows]
    test_ids = [row["video_id"] for row in test_rows]
    for name, ids, expected in (("train", train_ids, train_count), ("test", test_ids, test_count)):
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"Expected exactly {expected} unique {name} video IDs")
    if set(train_ids) & set(test_ids):
        raise ValueError("Train/test video overlap")
    all_ids = set(train_ids) | set(test_ids)
    video_rows = annotations.get("videos", [])
    declared_ids = [row["video_id"] for row in video_rows]
    if len(set(declared_ids)) != len(declared_ids) or set(declared_ids) != all_ids:
        raise ValueError("Annotation video universe differs from locked split universe")
    caption_rows = []
    sent_ids = set()
    for index, row in enumerate(annotations["sentences"]):
        video_id, caption = row["video_id"], row["caption"]
        if video_id not in all_ids or not isinstance(caption, str) or not caption.strip():
            raise ValueError("Caption has unknown video ID or empty/non-string text")
        sent_id = str(row.get("sen_id", f"row_{index:06d}"))
        if sent_id in sent_ids:
            raise ValueError("Duplicate upstream sentence ID")
        sent_ids.add(sent_id)
        caption_rows.append({"text_id": f"msrvtt_sen_{sent_id}", "video_id": video_id,
                             "caption": caption, "annotation_row": index})
    count_by_video = Counter(row["video_id"] for row in caption_rows)
    if set(count_by_video) != all_ids:
        raise ValueError("Some locked videos have no captions")
    captions_by_video = {}
    for row in caption_rows:
        captions_by_video.setdefault(row["video_id"], set()).add(row["caption"])
    queries = []
    for index, row in enumerate(test_rows):
        caption = row["sentence"]
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Official test sentence is empty/non-string")
        queries.append({"text_id": f"msrvtt_jsfusion_{index:04d}", "video_id": row["video_id"],
                        "caption": caption, "candidate_index": index,
                        "exact_match_in_full_annotations":
                            caption in captions_by_video[row["video_id"]]})
    return train_ids, test_ids, caption_rows, queries, count_by_video


def read_archive(path):
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Metadata archive exceeds the agreed small-metadata limit")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Metadata archive expands beyond allowed size")
        result, hashes = {}, {}
        for basename in ("MSRVTT_train.9k.csv", "MSRVTT_JSFUSION_test.csv", "MSRVTT_data.json"):
            matches = [m for m in members if Path(m.filename).name == basename and not m.is_dir()]
            if len(matches) != 1:
                raise ValueError(f"Need one unambiguous {basename} archive member")
            raw = archive.read(matches[0])
            hashes[basename] = {"archive_member": matches[0].filename, "bytes": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest()}
            decoded = raw.decode("utf-8-sig")
            result[basename] = (json.loads(decoded) if basename.endswith(".json")
                                else list(csv.DictReader(io.StringIO(decoded))))
    return result, hashes


def build_lock(archive_path):
    members, member_hashes = read_archive(archive_path)
    train_ids, test_ids, captions, queries, counts = validate_metadata(
        members["MSRVTT_train.9k.csv"], members["MSRVTT_JSFUSION_test.csv"],
        members["MSRVTT_data.json"])
    research_train, research_val = split_development(train_ids)
    source_train, source_val = split_development_source_groups(
        train_ids, members["MSRVTT_data.json"]["videos"])
    train_set, val_set, test_set = set(research_train), set(research_val), set(test_ids)
    source_train_set, source_val_set = set(source_train), set(source_val)
    caption_groups = {
        "official_train9k": [r for r in captions if r["video_id"] not in test_set],
        "research_train8100": [r for r in captions if r["video_id"] in train_set],
        "research_validation900": [r for r in captions if r["video_id"] in val_set],
        "research_train8100_sourcegroup_v2": [r for r in captions if r["video_id"] in source_train_set],
        "research_validation900_sourcegroup_v2": [r for r in captions if r["video_id"] in source_val_set],
        "official_test1000_all_captions_extension": [r for r in captions if r["video_id"] in test_set],
    }
    duplicate_texts = sum(count - 1 for count in Counter(r["caption"] for r in captions).values())
    duplicate_video_text = sum(count - 1 for count in Counter(
        (r["video_id"], r["caption"]) for r in captions).values())
    positive_pairs = [[q["text_id"], q["video_id"]] for q in queries]
    lock = {
        "schema_version": "msrvtt_w03_v2", "scope": "metadata_and_protocol_only",
        "upstream": {"repository": "https://github.com/ArrowLuo/CLIP4Clip",
                     "commit": CLIP4CLIP_COMMIT, "archive_url": ARCHIVE_URL,
                     "archive_bytes": archive_path.stat().st_size,
                     "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                     "members": member_hashes},
        "development_split": {"seed": 42, "algorithm": "stdlib_random_shuffle_upstream_order_first900",
                              "selection_metric": "heldout_train_text_to_video_R@1",
                              "official_test_for_model_selection": False,
                              "role": "nominal_v1_retained_for_audit_not_selected_for_head_training"},
        "source_group_development_split": {
            "version": "v2", "seed": 42,
            "algorithm": "group exact upstream source URL in train order; seeded shuffle group keys; greedily accept whole groups that fit remaining quota900; output upstream ID order",
            "selection_metric": "heldout_train_text_to_video_R@1",
            "official_test_for_model_selection": False,
            "role": "selected_for_future_head_training",
            "note": "official train/test inherited source overlap is unchanged and separately reported"},
        "video_ids": {"official_train9k": train_ids, "research_train8100": research_train,
                      "research_validation900": research_val, "official_test1000": test_ids,
                      "research_train8100_sourcegroup_v2": source_train,
                      "research_validation900_sourcegroup_v2": source_val},
        "video_id_sha256": {},
        "captions": {"annotation_count": len(captions), "per_video_count_histogram": dict(
            sorted(Counter(counts.values()).items())), "mapping_sha256": semantic_hash(captions),
            "duplicates_kept": True, "duplicate_text_extra_rows": duplicate_texts,
            "duplicate_video_text_extra_rows": duplicate_video_text,
            "groups": {name: {"count": len(rows), "sha256": semantic_hash(rows)}
                       for name, rows in caption_groups.items()}},
        "official_test": {"video_candidates": test_ids, "text_queries": queries,
                          "positive_pairs": positive_pairs,
                          "matrix_shape_text_to_video": [len(queries), len(test_ids)],
                          "candidate_order_sha256": semantic_hash(test_ids),
                          "caption_queries_sha256": semantic_hash(queries),
                          "positive_mapping_sha256": semantic_hash(positive_pairs)},
        "annotation_metadata_acquired": True, "video_media_acquired": False,
        "feature_extraction_run": False, "benchmark_evaluation_run": False,
        "caption_text_normalization": "none; preserve upstream English text and row identities",
    }
    lock["official_test"]["csv_sentence_annotation_exact_mismatch_count"] = sum(
        not q["exact_match_in_full_annotations"] for q in queries)
    lock["official_test"]["csv_sentence_annotation_mismatch_video_ids"] = [
        q["video_id"] for q in queries if not q["exact_match_in_full_annotations"]]
    parent_by_video = {r["video_id"]: r.get("url", "") for r in members["MSRVTT_data.json"]["videos"]}
    parent_sets = {name: {parent_by_video[v] for v in ids if parent_by_video[v]}
                   for name, ids in lock["video_ids"].items()}
    lock["source_url_audit"] = {
        "unit": "exact upstream URL; no network/video download",
        "unique_nonempty_urls": len({url for url in parent_by_video.values() if url}),
        "official_train_test_shared_urls": len(
            parent_sets["official_train9k"] & parent_sets["official_test1000"]),
        "research_train_validation_shared_urls": len(
            parent_sets["research_train8100"] & parent_sets["research_validation900"]),
        "research_sourcegroup_v2_train_validation_shared_urls": len(
            parent_sets["research_train8100_sourcegroup_v2"] &
            parent_sets["research_validation900_sourcegroup_v2"]),
        "research_sourcegroup_v2_train_validation_video_overlap": len(source_train_set & source_val_set),
        "sourcegroup_v2_train_urls_sha256": semantic_hash(
            sorted(parent_sets["research_train8100_sourcegroup_v2"])),
        "sourcegroup_v2_validation_urls_sha256": semantic_hash(
            sorted(parent_sets["research_validation900_sourcegroup_v2"])),
        "scope": "source-group v2 resolves train/validation URL overlap; official train/test inherited overlap preserved",
    }
    lock["video_id_sha256"] = {name: semantic_hash(ids) for name, ids in lock["video_ids"].items()}
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("outputs/w03/sources/m2/msrvtt_data.zip"))
    parser.add_argument("--output", type=Path, default=Path("outputs/w03/msrvtt_protocol_lock.json"))
    args = parser.parse_args()
    lock = build_lock(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_METADATA_LOCK", "output": str(args.output),
                      "split_counts": {name: len(ids) for name, ids in lock["video_ids"].items()},
                      "caption_count": lock["captions"]["annotation_count"],
                      "archive_sha256": lock["upstream"]["archive_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

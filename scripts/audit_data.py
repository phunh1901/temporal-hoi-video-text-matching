"""Validate the user-approved lightweight pilot separately from research readiness."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from temporal_hoi.data.dataset import TemporalHOIDataset


def audit(root=Path(".")):
    cfg = json.loads((root / "configs/w02_lightweight.json").read_text(encoding="utf-8"))
    manifest = root / "data/manifests"
    dataset = TemporalHOIDataset(
        manifest / "clips.csv",
        manifest / "texts.csv",
        manifest / "splits.csv",
        relevance_csv=manifest / "relevance.csv",
    )
    clips, texts, rel = dataset.clips, dataset.texts, dataset.relevance
    reviews = pd.read_csv(manifest / "label_reviews.csv")
    sources = pd.read_csv(manifest / "sources.csv")
    errors = []
    if len(clips) != cfg["expected_clips"]:
        errors.append("Pilot clip count differs from locked configuration")
    for split, expected in [
        ("dev", cfg["expected_dev_clips"]),
        ("extension", cfg["expected_extension_clips"]),
    ]:
        if clips.split.eq(split).sum() != expected:
            errors.append(f"Unexpected {split} count")
    if clips.split.isin(["val", "test"]).any():
        errors.append("Previously inspected pilot footage cannot be a blind test set")
    if texts.prompt_text.duplicated().any():
        errors.append("Duplicate candidate sentences")
    for _, clip in clips.iterrows():
        annotations = rel[rel.clip_id == clip.clip_id]
        if set(annotations.text_id) != set(texts.text_id):
            errors.append(f"{clip.clip_id}: incomplete candidate relevance")
        if (
            annotations.is_match.eq(1).sum() < cfg["min_positive_texts"]
            or annotations.is_match.eq(0).sum() < cfg["min_negative_texts"]
        ):
            errors.append(f"{clip.clip_id}: missing reviewed positive/negative sentences")
        if clip.clip_id not in set(reviews.clip_id):
            errors.append(f"{clip.clip_id}: missing review record")
        if clip.video_id not in set(sources.video_id):
            errors.append(f"{clip.clip_id}: missing provenance")
        if pd.notna(clip.has_violation) and clip.evidence_status == "insufficient_for_violation":
            errors.append(f"{clip.clip_id}: unsupported event label")
        if str(clip.evaluation_eligible).lower() == "true":
            errors.append(f"{clip.clip_id}: pilot incorrectly marked for final evaluation")
        if (
            pd.notna(clip.has_violation)
            and clip.has_violation == 1
            and clip.rule_id == "RULE_01_PARKING"
            and clip.end_s - clip.start_s < cfg["parking_min_duration_s"]
        ):
            errors.append(f"{clip.clip_id}: duration cannot establish parking violation")
    total_bytes = 0
    for _, source in sources.iterrows():
        path = root / source.video_path
        if not path.is_file():
            errors.append(f"Missing source {path}")
            continue
        total_bytes += path.stat().st_size
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != source.sha256:
                errors.append(f"Changed source content: {source.video_id}")
        for value in [source.source_url, source.license_status, source.license_url]:
            if pd.isna(value):
                errors.append(f"Incomplete provenance: {source.video_id}")
    if total_bytes > cfg["max_raw_bytes"]:
        errors.append("Raw footage exceeds configured storage limit")
    lock = json.loads((root / "outputs/w02/pilot_manifest_lock.json").read_text())
    for name, expected in lock["files"].items():
        if hashlib.sha256((manifest / name).read_bytes()).hexdigest() != expected:
            errors.append(f"Manifest changed after lock: {name}")
    return dict(
        profile=cfg["profile"],
        status="FAIL" if errors else "PASS_TECHNICAL_PILOT",
        total_clips=len(clips),
        dev_clips=int(clips.split.eq("dev").sum()),
        extension_demo_clips=int(clips.split.eq("extension").sum()),
        unique_source_videos=len(sources),
        raw_bytes=total_bytes,
        raw_mib=round(total_bytes / 1024**2, 2),
        max_raw_mib=cfg["max_raw_bytes"] / 1024**2,
        candidates=len(texts),
        relevance_pairs=len(rel),
        review_method="AI visual review of source contact sheets; not human independent review",
        independent_test_available=False,
        findings=errors,
        research_status="NOT_READY_FOR_VIOLATION_EVALUATION",
        research_gaps=[
            "No verified person-littering positive clip",
            "Parking footage cannot establish the 30-second rule",
            "Zones are unvalidated; unknown event labels remain null",
            "No independent validation/test sessions or long normal-video exposure",
            "Some repository licenses do not establish provenance of third-party footage",
        ],
    )


def main():
    report = audit()
    Path("outputs/w02/data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(bool(report["findings"]))


if __name__ == "__main__":
    raise SystemExit(main())

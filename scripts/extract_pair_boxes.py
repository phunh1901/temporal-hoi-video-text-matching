"""Materialize deterministic union pairs from verified detector predictions.

No detector checkpoint is bundled or selected. Without a verified source the
CLI exports a readiness blocker, never fabricated boxes or full-frame fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_pilot_baseline import sha256_file

from temporal_hoi.data.pair_crops import select_pair, validate_provenance


def extract(config, config_path):
    audit = {"task_id": "W05-03", "config": config_path.as_posix(),
             "config_sha256": sha256_file(config_path), "status": "blocked",
             "detector_inference": False, "boxes_file": None, "cohort_lock": None,
             "missing_frames": [], "full_frame_fallback": False}
    if not config.get("detector") or not config.get("verified_predictions"):
        audit["reason"] = "No verified runnable detector/checkpoint and prediction artifact selected"
        return audit
    detector = config["detector"]
    missing = [key for key in config["required_detector_metadata"] if detector.get(key) is None]
    if missing:
        raise ValueError(f"Missing verified detector metadata: {missing}")
    input_path = Path(config["verified_predictions"])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("coordinates") != "original_frame_pixels_xyxy":
        raise ValueError("Detector adapter must invert resize to original pixels before pair selection")
    if validate_provenance(payload["provenance"]) != "detector":
        raise ValueError("Prediction adapter requires detector provenance")
    if payload["provenance"]["detector_checkpoint_sha256"] != detector["checkpoint_sha256"]:
        raise ValueError("Prediction provenance changed the configured detector checkpoint")
    records, seen = [], set()
    for frame in payload["frames"]:
        key = (frame["video_id"], frame["frame_index"])
        if key in seen:
            raise ValueError("Duplicate video/frame predictions")
        seen.add(key)
        pair = select_pair(frame["pairs"], frame["width"], frame["height"])
        if pair is None:
            audit["missing_frames"].append({"video_id": key[0], "frame_index": key[1],
                                             "reason": "missing_pair"})
            continue
        records.append({**pair, "clip_id": frame["video_id"], "video_id": frame["video_id"],
                        "frame_index": frame["frame_index"], "timestamp_s": frame["timestamp_s"],
                        "source_video_sha256": frame["source_video_sha256"],
                        "box_source": "predicted", "detector_sha256": detector["checkpoint_sha256"],
                        "tracker_version": detector["tracker_version"]})
    output_dir = Path(config["boxes_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "boxes.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    manifest_path = output_dir / "verified_union_boxes.json"
    manifest = {"schema_version": "person_object_union/v1",
                "coordinates": "original_frame_pixels_xyxy",
                "provenance": payload["provenance"], "records": records}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    audit.update({"status": "verified_prediction_adapter_exported",
                  "reason": "Cohort must still exclude every video with a missing sampled pair",
                  "boxes_file": {"path": records_path.as_posix(),
                                 "sha256": sha256_file(records_path)},
                  "union_manifest": {"path": manifest_path.as_posix(),
                                     "sha256": sha256_file(manifest_path)},
                  "input_predictions_sha256": sha256_file(input_path),
                  "exported_pairs": len(records)})
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hoi_union_crop.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = extract(config, args.config)
    output = Path(config["readiness_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 2 if audit["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())

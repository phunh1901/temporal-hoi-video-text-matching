"""Diagnose fixed M1 checkpoints and prepare train-only human actor review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from temporal_hoi.data.hoi_pair_review import (
    choose_review_cases,
    render_review_html,
    validate_review_annotations,
    validate_review_manifest,
)
from temporal_hoi.evaluation.hoi_pair_diagnostics import (
    load_pair_diagnostic_source,
    run_pair_score_diagnostics,
    score_pair_rows,
)
from temporal_hoi.inference.hoi_kaggle import safe_member
from temporal_hoi.inference.hoi_static import checked_file, file_sha256
from temporal_hoi.models.hoi_head import MODES
from temporal_hoi.training.hoi_head_study import SEEDS

CONFIG_SCHEMA = "vidhoi_pair_diagnostics_config_v1"
INPUTS = ("source_result", "protocol", "mapping", "review_manifest", "annotations")
LABEL_FILENAME = "vidhoi_pair_review_annotations.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def record(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": file_sha256(path)}


def inventory(root: Path) -> list[dict]:
    return [record(path, root) for path in sorted(root.rglob("*")) if path.is_file()]


def verify_inventory(root: Path, receipt: dict) -> None:
    names = set()
    for row in receipt.get("files", []):
        safe_member(row["path"])
        path = root / row["path"]
        if (row["path"] in names or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve()) or record(path, root) != row):
            raise ValueError("Diagnostic stage inventory differs")
        names.add(row["path"])
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if not names or actual != names | {"receipt.json"}:
        raise ValueError("Incomplete or unexpected diagnostic stage output")


def resolve_config(path: Path) -> dict:
    config = read_json(path)
    for key in INPUTS:
        config[key] = str((path.parent / config[key]).resolve())
    return config


def _triplets(values) -> set[tuple]:
    if not isinstance(values, list) or any(not isinstance(v, list) or len(v) != 3 for v in values):
        raise ValueError("Protocol triplets must be explicit triples")
    return {tuple(v) for v in values}


def validate_review_source(manifest: dict, source: dict, protocol: dict, config: dict) -> dict:
    """Bind every case, box, label badge and cache row to score-blind train selection."""
    audit = validate_review_manifest(manifest)
    if (manifest.get("source_result_sha256") != config["source_result_sha256"]
            or manifest["protocol_sha256"] != config["protocol_sha256"]
            or manifest["mapping_sha256"] != config["mapping_sha256"]):
        raise ValueError("Review/source/protocol/mapping identities differ")
    selection = manifest.get("selection")
    if (not isinstance(selection, dict) or selection.get("seed") != 42
            or selection.get("model_scores_used") is not False
            or selection.get("test_fixture") is True):
        raise ValueError("Review requires real score-blind fixed train selection")
    cache = source["train_cache"]
    selected = choose_review_cases(cache, protocol, cache["mapping"],
                                   count=config.get("review_case_count", 30), seed=42)
    if selection != selected["selection"] or len(manifest["cases"]) != len(selected["cases"]):
        raise ValueError("Review selection differs from the declared score-free rule")
    for materialized, expected in zip(manifest["cases"], selected["cases"], strict=True):
        for key, value in expected.items():
            if key not in {"context_frames", "video"} and materialized.get(key) != value:
                raise ValueError(f"Review canonical actor/frame/source differs: {key}")
    return audit


def preflight_pair(config: dict) -> dict:
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("test_fixture", False):
        raise ValueError("Require a real cached pair diagnostic configuration")
    if config.get("threads", 2) != 2 or config.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("Diagnostic requires declared cpu/cuda device and fixed two threads")
    paths = {key: checked_file(config[key], config.get(key + "_sha256"), key) for key in INPUTS}
    source = load_pair_diagnostic_source(paths["source_result"], config["source_result_sha256"])
    hashes = source["source_receipt"]["input_hashes"]
    if any(config[key + "_sha256"] != hashes[key] for key in ("protocol", "mapping")):
        raise ValueError("Diagnostic protocol/mapping differs from source run")
    protocol = read_json(paths["protocol"])
    if (protocol.get("schema_version") != "vidhoi_head_protocol_v1"
            or protocol.get("label_policy") != "positive_only_unknown"):
        raise ValueError("Protocol/unknown-label policy differs")
    train_ids, dev_ids = set(protocol["train_video_ids"]), set(protocol["dev_video_ids"])
    if train_ids & dev_ids or (train_ids | dev_ids) & set(protocol["excluded_final_video_ids"]):
        raise ValueError("Role or final-held-out video leakage")
    if any(protocol.get(k, False) is not False for k in
           ("official_final_split_accessed", "final_unseen_accessed")):
        raise ValueError("Final data cannot enter pair diagnosis")
    final, pseudo = (_triplets(protocol[key]) for key in ("final_unseen_triplets", "pseudo_unseen_triplets"))
    for role, videos in (("train", train_ids), ("validation", dev_ids)):
        cache = source["train_cache" if role == "train" else "dev_cache"]
        if {f.video_id for f in cache["frames"]} != videos:
            raise ValueError("Source cache videos differ from locked protocol")
        for video in videos:
            positive = _triplets(protocol["video_positive_triplets"][video])
            if positive & final or (role == "train" and positive & pseudo):
                raise ValueError("Reserved composition appears in full-video context")
    expected_mask = np.ones(source["candidate_mask"].shape, dtype=bool)
    for _, predicate, obj in final:
        expected_mask[obj, predicate] = False
    if (not np.array_equal(source["candidate_mask"], expected_mask)
            or np.any(source["text_features"][~expected_mask] != 0)):
        raise ValueError("Source text candidate mask exposes excluded final concepts")
    manifest = read_json(paths["review_manifest"])
    review = validate_review_source(manifest, source, protocol, config)
    if set(manifest["files"]) != set(review["referenced_assets"]):
        raise ValueError("Unreferenced review media cannot enter the input bundle")
    assets_root = paths["review_manifest"].parent
    for name, identity in manifest["files"].items():
        safe_member(name)
        path = assets_root / name
        if path.is_symlink() or not path.resolve().is_relative_to(assets_root.resolve()):
            raise ValueError("Review image escapes its asset root")
        checked_file(path, identity["sha256"], "review media")
        if path.stat().st_size != identity["bytes"]:
            raise ValueError("Review media byte inventory differs")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Only audited raster review context is accepted")
        with Image.open(path) as image:
            image.verify()
    annotations = read_json(paths["annotations"])
    validate_review_annotations(annotations, manifest)
    return {"source": source, "protocol": protocol, "manifest": manifest,
            "assets_root": assets_root, "annotations": annotations, "review_audit": review,
            "input_hashes": {key: config[key + "_sha256"] for key in INPUTS}}


def choose_annotation_input(default: dict, manifest: dict, *, explicit: Path | None = None,
                            expected_sha: str | None = None, input_root: Path | None = None):
    if explicit is not None and input_root is not None:
        raise ValueError("Choose explicit annotation path or discovery root, not both")
    selected = explicit
    if input_root is not None:
        candidates = sorted(input_root.rglob(LABEL_FILENAME))
        if len(candidates) > 1:
            raise ValueError("Multiple review label files attached; keep one version only")
        selected = candidates[0] if candidates else None
    if selected is None:
        return default, {"kind": "packaged_blank_or_reviewed_annotations", "external": False}
    if selected.is_symlink() or selected.stat().st_size > 2 * 1024**2:
        raise ValueError("Review label file must be a bounded plain JSON file")
    digest = file_sha256(selected)
    if explicit is not None and digest != expected_sha:
        raise ValueError("Explicit review annotations require their matching SHA256")
    labels = read_json(selected)
    validate_review_annotations(labels, manifest)
    return labels, {"kind": "user_supplied_review_assertions", "external": True,
                    "filename": selected.name, "sha256": digest, "bytes": selected.stat().st_size,
                    "_resolved_source_path": str(selected.resolve())}


def evaluate_actor_contrasts(labels: dict, manifest: dict, scores: dict,
                             *, tolerance: float = 1e-6) -> dict:
    """Only human-asserted positive/negative pairs contribute; unknowns are ignored."""
    validate_review_annotations(labels, manifest)
    by_case = {case["case_id"]: {a["track_id"]: a["label"] for a in case["actors"]}
               for case in labels["cases"]}
    records = []
    for state in scores["states"]:
        for case in state["cases"]:
            states = by_case[case["case_id"]]
            positive = [a for a in case["actors"] if states[a["track_id"]] == "positive"]
            negative = [a for a in case["actors"] if states[a["track_id"]] == "confirmed_negative"]
            for pos in positive:
                for neg in negative:
                    margin = float(pos["score"] - neg["score"])
                    records.append({"mode": state["mode"], "seed": state["seed"],
                        "state": state["state"], "case_id": case["case_id"],
                        "positive_track_id": pos["track_id"], "negative_track_id": neg["track_id"],
                        "score_margin": margin, "strict_win": margin > 0,
                        "exact_tie": margin == 0, "near_tie": abs(margin) <= tolerance})
    aggregate = {}
    for mode in MODES:
        for state in ("initial", "last"):
            current = [r for r in records if r["mode"] == mode and r["state"] == state]
            aggregate[f"{mode}/{state}"] = {
                "seed_evaluations": 3, "num_pair_comparisons_across_seeds": len(current),
                "strict_win_fraction": float(np.mean([r["strict_win"] for r in current])) if current else None,
                "near_tie_fraction": float(np.mean([r["near_tie"] for r in current])) if current else None,
                "mean_margin": float(np.mean([r["score_margin"] for r in current])) if current else None}
    return {"schema_version": "vidhoi_review_actor_contrasts_v1",
            "status": "evaluated_human_asserted_pairs" if records else "pending_verified_contrasts",
            "records": records, "aggregate": aggregate, "near_tie_tolerance": tolerance,
            "unknown_labels_used_as_negative": False, "benchmark_metric": False,
            "interpretation": "Diagnostic conditional comparisons from reviewer assertions; not HOI mAP, calibrated probabilities, or independent benchmark generalization."}


def run_review(checked: dict, config: dict, directory: Path, *, explicit=None,
               expected_sha=None, input_root=None) -> dict:
    manifest, source = checked["manifest"], checked["source"]
    labels, annotation_source = choose_annotation_input(
        checked["annotations"], manifest, explicit=explicit, expected_sha=expected_sha, input_root=input_root)
    validation = validate_review_annotations(labels, manifest)
    directory.mkdir(parents=True, exist_ok=False)
    if annotation_source.get("external"):
        original = Path(annotation_source.pop("_resolved_source_path"))
        checked_file(original, annotation_source["sha256"], "external review labels")
        (directory / "annotations_original.json").write_bytes(original.read_bytes())
    html = render_review_html(manifest, checked["assets_root"])
    (directory / "actor_review.html").write_text(html, encoding="utf-8")
    write_json(directory / "review_manifest.json", manifest)
    write_json(directory / LABEL_FILENAME, labels)
    write_json(directory / "annotation_validation.json", validation)
    requested = sorted({a["pair_cache_row"] for c in manifest["cases"] for a in c["actors"]})
    lookup = {row: index for index, row in enumerate(requested)}
    scores = {"schema_version": "vidhoi_actor_review_scores_v1", "states": [],
              "model_scores_hidden_in_review_HTML": True, "training_run": False}
    for mode in MODES:
        for seed in SEEDS:
            for state in ("initial", "last"):
                values = score_pair_rows(source, mode, seed, state, requested, config.get("device", "cpu"))
                scores["states"].append({"mode": mode, "seed": seed, "state": state,
                    "cases": [{"case_id": case["case_id"], "video_id": case["video_id"],
                        "predicate_id": case["predicate_id"], "object_track_id": case["object_track_id"],
                        "actors": [{"track_id": a["track_id"], "score": float(values[lookup[a["pair_cache_row"]], case["predicate_id"]])}
                                   for a in case["actors"]]} for case in manifest["cases"]]})
    write_json(directory / "actor_scores_after_review.json", scores)
    contrasts = evaluate_actor_contrasts(labels, manifest, scores)
    write_json(directory / "confirmed_actor_contrasts.json", contrasts)
    receipt = {"schema_version": "vidhoi_pair_review_result_v1", "status": "completed",
        "level": "train_only_actor_review_diagnostic", "source_result_sha256": config["source_result_sha256"],
        "input_hashes": checked["input_hashes"], "review_manifest_sha256": config["review_manifest_sha256"],
        **{key: validation[key] for key in ("manifest_fingerprint", "case_count", "actor_count",
            "review_labels_status", "positive_count", "confirmed_negative_count", "unknown_count",
            "eligible_contrast_case_count", "fully_labeled_case_count")},
        "annotation_source": annotation_source, "annotation_content_sha256": hashlib.sha256(
            json.dumps(labels, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "reviewer_provenance": validation["reviewer_provenance_level"],
        "model_training_run": False, "encoder_inference_run": False, "head_inference_run": True,
        "model_scores_hidden_in_review_HTML": True, "unknown_labels_treated_as_negative": False,
        "official_final_split_accessed": False, "final_unseen_accessed": False,
        "files": inventory(directory)}
    write_json(directory / "receipt.json", receipt)
    return receipt


def run_stage(config: dict, output: Path, stage: str, *, explicit=None, expected_sha=None, input_root=None):
    checked = preflight_pair(config)
    output.mkdir(parents=True, exist_ok=True)
    if stage == "preflight":
        receipt = {"schema_version": "vidhoi_pair_preflight_v1", "status": "completed",
            "input_hashes": checked["input_hashes"], "source": checked["source"]["source_receipt"],
            "review": checked["review_audit"], "model_training_run": False,
            "encoder_inference_run": False, "head_inference_run": False}
        write_json(output / "preflight.json", receipt)
        return receipt
    if stage == "diagnose":
        return run_pair_score_diagnostics(checked["source"],
            {**config, "input_hashes": checked["input_hashes"]}, output / "diagnostics")
    if stage != "review":
        raise ValueError("Unknown diagnostic stage")
    diagnostic = read_json(output / "diagnostics/receipt.json")
    verify_inventory(output / "diagnostics", diagnostic)
    if (diagnostic.get("status") != "completed" or diagnostic.get("diagnosed_states") != 18
            or diagnostic.get("source_result_sha256") != config["source_result_sha256"]
            or diagnostic.get("input_hashes") != checked["input_hashes"]):
        raise ValueError("Complete diagnostics with matching inputs are required before review export")
    review = run_review(checked, config, output / "review", explicit=explicit,
                        expected_sha=expected_sha, input_root=input_root)
    receipt = {"schema_version": "vidhoi_pair_diagnostic_run_v1", "status": "completed",
        "level": "cached_feature_pair_diagnostic", "input_hashes": checked["input_hashes"],
        "source_result_sha256": config["source_result_sha256"], "diagnosed_states": 18,
        "review_case_count": review["case_count"], "confirmed_negative_count": review["confirmed_negative_count"],
        "review_labels_status": review["review_labels_status"], "training_run": False,
        "encoder_inference_run": False, "head_inference_run": True,
        "official_final_split_accessed": False, "final_unseen_accessed": False,
        "unknown_labels_treated_as_negative": False, "full_benchmark_reproduction_claim": False,
        "stage_receipts": {name: record(output / name / "receipt.json", output)
                           for name in ("diagnostics", "review")}, "files": inventory(output)}
    write_json(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=["preflight", "diagnose", "review", "all"], default="all")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--annotations-sha256")
    parser.add_argument("--annotation-input-root", type=Path)
    args = parser.parse_args(argv)
    config = resolve_config(args.config)
    for stage in (["preflight", "diagnose", "review"] if args.stage == "all" else [args.stage]):
        print(f"PAIR_STAGE_STARTED: {stage}", flush=True)
        result = run_stage(config, args.output_dir, stage, explicit=args.annotations,
                           expected_sha=args.annotations_sha256, input_root=args.annotation_input_root)
        print(json.dumps({"stage": stage, "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()

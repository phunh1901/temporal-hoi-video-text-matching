"""Audit locked MSR-VTT adapters, or evaluate an explicitly identified score artifact.

Default execution audits metadata/media and exports InternVideo2-compatible JSON.
It never downloads weights or media and never invents scores for missing videos.
An optional .npz contains text_ids, video_ids and either scores [text,video], or
text_embeddings/video_embeddings (or frame_embeddings/valid_frame_mask for
meanP). Supply its .json provenance separately.
"""

import argparse
import importlib.util
import json
import platform
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

from temporal_hoi.data.msrvtt import (
    MSRVTTAdapter,
    evaluate_pool_scores,
    file_sha256,
    internvideo_test_annotations,
    mean_pool_valid_frames,
    scores_from_embeddings,
    semantic_hash,
)

OFFICIAL_METRICS_SHA256 = "103e93090de14f55d1db61e40ecf1fbc814670975bd9b9ab1fe341d0c95f0a6f"
INTERNVIDEO_COMMIT = "3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _official_metrics(path, scores):
    if file_sha256(path) != OFFICIAL_METRICS_SHA256:
        raise ValueError("Pinned CLIP4Clip evaluator hash mismatch")
    loader = SourceFileLoader("msrvtt_locked_official", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def normalize(result):
        return {"R@1": float(result["R1"]) / 100,
                "R@5": float(result["R5"]) / 100,
                "R@10": float(result["R10"]) / 100,
                "MedR": float(result["MR"]), "MeanR": float(result["MeanR"])}

    return {"text_to_video": normalize(module.compute_metrics(scores)),
            "video_to_text": normalize(module.compute_metrics(scores.T)),
            "recall_scale": "fraction", "source_sha256": OFFICIAL_METRICS_SHA256,
            "ties": "unchanged upstream equal-to-positive matching; may expand query count"}


def _read_provenance(path):
    provenance = json.loads(path.read_text(encoding="utf-8"))
    required = ["model_name", "checkpoint_sha256", "preprocess", "pooling", "score_type",
                "training_regime", "artifact_kind"]
    if any(not provenance.get(key) for key in required):
        raise ValueError(f"Score provenance requires nonempty fields {required}")
    checkpoint_hash = provenance["checkpoint_sha256"]
    if (not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64
            or any(char not in "0123456789abcdef" for char in checkpoint_hash)):
        raise ValueError("Provenance checkpoint_sha256 must be lowercase SHA256")
    return provenance


def run(args):
    adapter = MSRVTTAdapter.from_files(args.archive, args.lock)
    pools = {split: adapter.pool(split) for split in ["train", "validation", "test"]}
    coverage = {split: adapter.media_coverage(split, args.media_root, suffix=args.media_suffix)
                for split in pools}
    test = pools["test"]
    annotation = internvideo_test_annotations(test, suffix=args.media_suffix)
    internvideo_path = args.output.parent / "m2_internvideo_jsfusion_test.json"
    _write_json(internvideo_path, annotation)
    # Reconstruct identities from the exported, raw JSON, rather than assuming conversion worked.
    roundtrip = [[q["text_id"], row["video"][:-len(args.media_suffix)]]
                 for q, row in zip(test.queries, annotation, strict=True)]
    if (semantic_hash(roundtrip) != test.hashes()["positive_mapping_sha256"]
            or [row["caption"] for row in annotation] != [q["caption"] for q in test.queries]):
        raise ValueError("InternVideo conversion changed positive mapping or query text")
    source_dir = getattr(args, "internvideo_source_dir", Path("outputs/experiments/pilot/baseline/m2_sources"))
    source_provenance = []
    for filename in ["internvideo_ret_dataset.py", "internvideo_dataset_utils.py"]:
        source = source_dir / filename
        source_provenance.append({"path": str(source), "exists": source.is_file(),
                                  "sha256": file_sha256(source) if source.is_file() else None})
    report = {
        "schema_version": "msrvtt_adapter_w04_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_ADAPTER_CHECKPOINT_INFERENCE_BLOCKED",
        "execution_level": "metadata_adapter_validation",
        "archive": {"path": str(args.archive), "sha256": file_sha256(args.archive)},
        "protocol_lock": {"path": str(args.lock), "sha256": file_sha256(args.lock)},
        "adapter": {"path": "src/temporal_hoi/data/msrvtt.py",
                    "sha256": file_sha256(Path("src/temporal_hoi/data/msrvtt.py"))},
        "split_counts": {split: {"videos": len(pool.video_ids), "captions": len(pool.queries),
                                "pool_hashes": pool.hashes()} for split, pool in pools.items()},
        "source_group_train_validation_shared_urls": len(
            {adapter.source_urls[v] for v in pools["train"].video_ids}
            & {adapter.source_urls[v] for v in pools["validation"].video_ids}),
        "official_train_test_shared_urls": len(
            {adapter.source_urls[v] for v in adapter.pool("train9k").video_ids}
            & {adapter.source_urls[v] for v in test.video_ids}),
        "official_csv_annotation_text_mismatches_kept": sum(
            not q["exact_match_in_full_annotations"] for q in test.queries),
        "duplicate_caption_strings": "preserved by distinct upstream sentence row IDs",
        "media_coverage": coverage,
        "internvideo_annotation": {
            "path": str(internvideo_path), "bytes": internvideo_path.stat().st_size,
            "file_sha256": file_sha256(internvideo_path),
            "semantic_sha256": semantic_hash(annotation), "rows": len(annotation),
            "fields": ["video", "caption"], "raw_query_text_preserved": True,
            "same_positive_mapping_verified": True, "commit": INTERNVIDEO_COMMIT,
            "source_snapshots": source_provenance,
            "loader": "load_anno(media_type='video'); VidTxtRetEvalDataset multi_txt_gt",
            "text_preprocess": "upstream pre_text lowercases/removes punctuation; model log required",
            "vindlu_original_byte_identity_claimed": False,
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "device_used": "CPU metadata only"},
        "benchmark_inference_run": False,
        "benchmark_training_run": False,
        "benchmark_metrics": None,
        "local_checkpoint_candidates": [
            {"path": str(path), "bytes": path.stat().st_size,
             "compatibility_verified": False}
            for directory in [Path("weights"), Path("checkpoints")]
            if directory.is_dir()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".bin", ".safetensors"}],
        "blockers": ["No compatible local MSR-VTT fine-tuned CLIP4Clip checkpoint verified",
                     ("No full official video media coverage"
                      if coverage["test"]["present_nonempty_files"] < len(test.video_ids)
                      else "Media files present but decodability not verified"),
                     "InternVideo2 weights entitlement and CUDA environment not verified"],
        "official_test_access": {
            "purpose": "fixed-protocol metadata/order/JSON verification, no model selection",
            "quality_scores_computed": False,
            "development_model_selection_split": "validation900_sourcegroup_v2"},
    }
    if args.scores is not None:
        if args.provenance is None:
            raise ValueError("--scores requires --provenance JSON")
        provenance = _read_provenance(args.provenance)
        checkpoint_verified = False
        if args.checkpoint is not None:
            if file_sha256(args.checkpoint) != provenance["checkpoint_sha256"]:
                raise ValueError("Local checkpoint hash differs from score provenance")
            checkpoint_verified = True
        with np.load(args.scores, allow_pickle=False) as stored:
            video_ids, text_ids = stored["video_ids"].tolist(), stored["text_ids"].tolist()
            full_pool = adapter.pool(args.split)
            if tuple(video_ids) == full_pool.video_ids:
                pool = full_pool
            elif args.allow_subset:
                pool = adapter.pool(args.split, selected_video_ids=video_ids)
            else:
                raise ValueError("Missing/different candidate IDs; explicitly pass --allow-subset")
            if "scores" in stored:
                scores = stored["scores"].copy()
                scoring = "provided score matrix; provenance-reported score type"
            else:
                video_embeddings = (stored["video_embeddings"] if "video_embeddings" in stored
                                    else mean_pool_valid_frames(stored["frame_embeddings"],
                                                                stored["valid_frame_mask"]))
                scores = scores_from_embeddings(stored["text_embeddings"],
                                                video_embeddings, scale=args.score_scale)
                scoring = "L2-normalized cosine * positive score_scale"
                if "video_embeddings" not in stored:
                    scoring = "frame L2 -> valid masked mean -> video/text L2 -> scaled cosine"
            metrics = evaluate_pool_scores(pool, scores, text_ids=text_ids, video_ids=video_ids)
        report.update({"status": "PASS_IDENTIFIED_SCORE_EVALUATION",
                       "execution_level": "external_score_artifact_evaluation",
                       "benchmark_metrics": metrics, "score_artifact": {
                           "path": str(args.scores), "sha256": file_sha256(args.scores),
                           "provenance_sha256": file_sha256(args.provenance),
                           "provenance_reported": provenance,
                           "local_checkpoint_hash_verified": checkpoint_verified,
                           "scoring": scoring,
                           "score_media_decode_validation_performed": False},
                       "score_pool_coverage": {"videos": len(pool.video_ids),
                                               "full_videos": pool.full_video_count,
                                               "texts": len(pool.text_ids),
                                               "full_texts": pool.full_text_count}})
        if args.split == "test":
            report["official_clip4clip_metrics_same_pool"] = _official_metrics(
                args.official_metrics, scores)
            report["official_test_access"]["quality_scores_computed"] = True
            report["official_test_access"]["purpose"] = "fixed reference score artifact evaluation"
        else:
            report["official_clip4clip_metrics_same_pool"] = None
    _write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "split_counts": {s: [len(p.video_ids), len(p.text_ids)] for s, p in pools.items()},
                      "media_files_present": {s: c["present_nonempty_files"]
                                               for s, c in coverage.items()},
                      "benchmark_metrics_computed": report["benchmark_metrics"] is not None},
                     indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/metadata/msrvtt_data.zip"))
    parser.add_argument("--lock", type=Path, default=Path("data/metadata/msrvtt_protocol_lock.json"))
    parser.add_argument("--media-root", type=Path, default=Path("data/raw/msrvtt"))
    parser.add_argument("--media-suffix", default=".mp4")
    parser.add_argument("--internvideo-source-dir", type=Path,
                        default=Path("outputs/experiments/pilot/baseline/m2_sources"),
                        help="Pinned source snapshots, independent of the report directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/pilot/baseline/m2_adapter_audit.json"))
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test", "test_all_captions"],
                        default="test")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--official-metrics", type=Path,
                        default=Path("src/temporal_hoi/evaluation/reference/clip4clip_metrics.py.txt"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()

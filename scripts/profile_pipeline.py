"""Rerun a locked local B0 pipeline and export measured CPU stage costs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_pilot_baseline import _csv, sha256_file


def replay_command(reference, output_dir):
    if reference.get("status") != "PASS_TECHNICAL_PILOT":
        raise ValueError("Profile requires a successful locked technical pilot run")
    if reference.get("baseline") != "B0_frozen_openclip_global_mean":
        raise ValueError("This profiler currently supports the frozen B0 global pilot")
    if reference["environment"]["device"] != "cpu":
        raise ValueError("This profiler implements synchronous CPU measurement only")
    inputs = reference["inputs"]
    return [
        sys.executable, str(Path(__file__).with_name("run_pilot_baseline.py")),
        "--checkpoint", reference["checkpoint"]["path"],
        "--checkpoint-sha256", reference["checkpoint"]["sha256"],
        "--manifest-dir", inputs["manifest_dir"],
        "--manifest-lock", inputs["manifest_lock"],
        "--config", inputs.get("config_path", "configs/pilot_video_loading.json"),
        "--threads", str(reference["environment"]["torch_num_threads"]),
        "--task-id", "W05-01", "--output-dir", str(output_dir),
    ]


def validate_replay(reference, actual):
    if actual.get("status") != "PASS_TECHNICAL_PILOT":
        raise ValueError("Replay did not finish successfully")
    for key in ("manifest_lock_sha256", "manifest_sha256", "config_sha256",
                "source_videos", "clip_ids", "text_ids"):
        if reference["inputs"][key] != actual["inputs"][key]:
            raise ValueError(f"Profile replay changed {key}")
    if reference["checkpoint"]["sha256"] != actual["checkpoint"]["sha256"]:
        raise ValueError("Profile replay changed checkpoint")
    for key in ("num_frames_per_clip", "decode_max_frame_side", "image_transform", "pooling",
                "text_normalization", "score", "candidate_order"):
        if reference["preprocess"][key] != actual["preprocess"][key]:
            raise ValueError(f"Profile replay changed {key}")


def profile_row(run, run_path, repeat_index, subprocess_seconds):
    stages = run["stage_timing"]["stages"]

    def ms(name):
        return stages[name]["seconds"] * 1000

    memory = run["memory_monitor"]
    return {
        "repeat": repeat_index,
        "model_load_ms": ms("model_load"),
        "decode_ms": ms("decode"),
        "preprocess_ms": ms("image_preprocess") + ms("text_preprocess"),
        "image_encoder_ms": ms("image_encoder"),
        "text_encoder_ms": ms("text_encoder"),
        "pooling_normalization_ms": ms("pooling_normalization"),
        "head_ms": run["head"]["learned_head_seconds"] * 1000,
        "similarity_ms": ms("similarity"),
        "evaluation_ms": ms("evaluation"),
        "end_to_end_ms": run["total_seconds"] * 1000,
        "subprocess_wall_ms": subprocess_seconds * 1000,
        "peak_working_set_bytes": memory.get("peak_process_working_set_bytes"),
        "peak_commit_bytes": memory.get("peak_process_commit_bytes"),
        "peak_cuda_allocated_bytes": run["vram"]["peak_cuda_allocated_bytes"],
        "device": "cpu", "batch_size": 1,
        "num_videos": run["coverage"]["clips_encoded"],
        "num_frames": run["coverage"]["frames_encoded"],
        "cache_mode": "fresh_decode_and_encode; OS page cache may be warm",
        "status": run["status"],
        "run_json": run_path.as_posix(),
        "run_sha256": sha256_file(run_path),
        "head_note": "Frozen identity; no learned head. Cosine scoring in similarity_ms",
    }


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_errors(run, path):
    clips = {c["clip_id"]: c for c in _csv(Path(run["inputs"]["manifest_dir"]) / "clips.csv")}
    splits = {r["clip_id"]: r["split"]
              for r in _csv(Path(run["inputs"]["manifest_dir"]) / "splits.csv")}
    prediction_path = run["outputs"]["predictions"]["path"]
    rows = []
    for error in run["retrieval_errors"]["top1_errors"]:
        if error["direction"] == "video_to_text":
            clip = clips[error["query_id"]]
            text_id = error["first_positive_candidate"]
        else:
            clip = clips[error["first_positive_candidate"]]
            text_id = error["query_id"]
        rows.append({
            "module": "M2", "split": "pilot_" + splits[clip["clip_id"]],
            "video_id": clip["video_id"], "text_id": text_id,
            "error_kind": "top1_negative_" + error["direction"],
            "true_rank": error["first_positive_rank"],
            "predicted_id": error["top1_candidate"], "box_source": "none_global",
            "evidence_path": prediction_path,
            "query_id": error["query_id"], "candidate_group": error["group"],
            "score_gap": error["top1_minus_first_positive"],
        })
    write_csv(path, rows, ["module", "split", "video_id", "text_id", "error_kind", "true_rank",
                           "predicted_id", "box_source", "evidence_path", "query_id",
                           "candidate_group", "score_gap"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/pilot/alignment_and_profiling/profile.csv"))
    args = parser.parse_args()
    if args.batch_size != 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("The locked low-memory CPU pilot requires batch-size 1 and positive repeats")
    reference = json.loads(args.run_json.read_text(encoding="utf-8"))
    root = args.output.parent / "bottleneck"
    rows, runs = [], []
    for index in range(args.warmup + args.repeat):
        is_warmup = index < args.warmup
        run_index = index + 1 if is_warmup else index - args.warmup + 1
        run_dir = root / (("warmup_" if is_warmup else "repeat_") + f"{run_index:02d}")
        command = replay_command(reference, run_dir)
        start = time.perf_counter()
        completed = subprocess.run(command, check=False)
        elapsed = time.perf_counter() - start
        path = run_dir / "baseline_run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        if completed.returncode:
            blocker = {
                "task_id": "W05-01", "status": run.get("status", "FAILED"),
                "warmup_runs_requested": args.warmup, "measured_repeats_requested": args.repeat,
                "measured_repeats_completed": len(rows),
                "reason": run.get("error"), "failed_run": path.as_posix(),
                "failed_run_sha256": sha256_file(path), "means": None,
                "benchmark_metrics": None,
            }
            (args.output.parent / "bottleneck_summary.json").write_text(
                json.dumps(blocker, indent=2, allow_nan=False), encoding="utf-8"
            )
            if rows:
                write_csv(args.output, rows)
            else:
                failed_row = dict.fromkeys([
                    "repeat", "model_load_ms", "decode_ms", "preprocess_ms", "image_encoder_ms",
                    "text_encoder_ms", "pooling_normalization_ms", "head_ms", "similarity_ms",
                    "evaluation_ms", "end_to_end_ms", "subprocess_wall_ms", "peak_working_set_bytes",
                    "peak_commit_bytes", "peak_cuda_allocated_bytes", "device", "batch_size",
                    "num_videos", "num_frames", "cache_mode", "status", "run_json", "run_sha256",
                    "head_note",
                ])
                failed_row.update({"status": run.get("status", "FAILED"), "device": "cpu",
                                   "batch_size": 1, "run_json": path.as_posix(),
                                   "run_sha256": sha256_file(path)})
                write_csv(args.output, [failed_row])
            print(json.dumps(blocker, indent=2))
            return completed.returncode
        validate_replay(reference, run)
        if not is_warmup:
            rows.append(profile_row(run, path, run_index, elapsed))
            runs.append(run)
            write_csv(args.output, rows)
    latest = runs[-1]
    write_errors(latest, args.output.parent / "error_cases.csv")
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    summary = {
        "task_id": "W05-01", "status": "PASS_TECHNICAL_PILOT_PROFILE",
        "reference": {"path": args.run_json.as_posix(), "sha256": sha256_file(args.run_json)},
        "warmup_runs": args.warmup, "measured_repeats": args.repeat,
        "warmup_policy": "Separate warmup process primes OS caches; each measured run loads model anew",
        "scope": "8 locked convenience clips; technical CPU pipeline, not benchmark",
        "means": {key: statistics.mean(row[key] for row in rows) for key in numeric},
        "sample_std": {key: statistics.stdev(row[key] for row in rows) if len(rows) > 1 else None
                       for key in numeric},
        "csv_sha256": sha256_file(args.output),
        "identical_scores_across_repeats": all(
            run["scores_video_text"] == runs[0]["scores_video_text"] for run in runs
        ),
        "retrieval_errors": latest["retrieval_errors"],
        "m1_status": "not_run_missing_real_predictions_and_gt_frame_mask",
        "benchmark_metrics": None,
        "cuda_peak": None,
        "memory_note": "Working Set is resident RAM; commit is process committed memory; CPU CUDA=null",
    }
    summary["dominant_pipeline_stage"] = max(
        ["decode_ms", "preprocess_ms", "image_encoder_ms", "text_encoder_ms",
         "pooling_normalization_ms", "similarity_ms", "evaluation_ms"],
        key=lambda key: summary["means"][key],
    )
    (args.output.parent / "bottleneck_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"profile_csv": args.output.as_posix(),
                      "dominant_pipeline_stage": summary["dominant_pipeline_stage"],
                      "measured_repeats": args.repeat}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

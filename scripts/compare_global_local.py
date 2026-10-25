"""Compare verified paired pilot runs; export explicit not-run benchmark rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from profile_pipeline import write_csv
from run_pilot_baseline import sha256_file

from temporal_hoi.evaluation.global_local import paired_metric_deltas

FIELDS = ["variant", "crop_mode", "box_source", "head", "cohort_hash", "encoder_sha256",
          "seed", "train_videos", "val_videos", "val_queries", "t2v_r1", "t2v_r5", "t2v_r10",
          "t2v_medr", "v2t_r1", "v2t_medr", "latency_ms", "peak_working_set_bytes",
          "trainable_params", "status", "scope", "reason"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-run-json", type=Path, default=Path("outputs/experiments/pilot/baseline/b0/baseline_run.json"))
    parser.add_argument("--b1-run-json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/pilot/alignment_and_profiling/global_local.csv"))
    args = parser.parse_args()
    b0 = json.loads(args.b0_run_json.read_text(encoding="utf-8"))
    report = {
        "task_id": "W05-03", "scope": "technical_pilot_only",
        "control": {"path": args.b0_run_json.as_posix(),
                    "sha256": sha256_file(args.b0_run_json),
                    "level": "existing frozen checkpoint pilot",
                    "metrics": b0.get("metrics")},
        "status": "BLOCKED_VERIFIED_PAIR_BOXES", "treatment": None,
        "delta": None, "benchmark_reproduction": False,
        "reason": "No verified person/object boxes or runnable detector/checkpoint available",
        "cohort_lock": None,
        "next_action": "Verify detector artifacts, extract deterministic pairs, lock identical cohort",
    }
    if args.b1_run_json:
        b1 = json.loads(args.b1_run_json.read_text(encoding="utf-8"))
        delta = paired_metric_deltas(b0, b1)
        with np.load(b0["outputs"]["embeddings"]["path"], allow_pickle=False) as a:
            with np.load(b1["outputs"]["embeddings"]["path"], allow_pickle=False) as b:
                if not np.array_equal(a["text_features"], b["text_features"]):
                    raise ValueError("Paired runs changed frozen text embeddings")
        report.update({"status": "PASS_PAIRED_TECHNICAL_PILOT", "delta": delta,
                       "reason": None,
                       "treatment": {"path": args.b1_run_json.as_posix(),
                                     "sha256": sha256_file(args.b1_run_json),
                                     "box_source": b1["box_source"], "metrics": b1["metrics"]}})
    rows = []
    for variant, mode, head in [("B0_frozen", "global", "frozen_identity"),
                                ("B1_frozen", "union", "frozen_identity"),
                                ("B0_H", "global", "linear_alignment"),
                                ("B1_H", "union", "linear_alignment")]:
        row = dict.fromkeys(FIELDS)
        row.update({"variant": variant, "crop_mode": mode,
                    "box_source": "none_global" if mode == "global" else "unverified_missing",
                    "head": head, "encoder_sha256": b0["checkpoint"]["sha256"], "seed": 42,
                    "trainable_params": 0 if head == "frozen_identity" else None,
                    "status": "not_run", "scope": "MSR-VTT_development_benchmark",
                    "reason": "Missing MSR-VTT media/cache" +
                    ("; missing verified boxes/detector/cohort" if mode == "union" else "") +
                    ("; no trained benchmark head" if head != "frozen_identity" else "")})
        rows.append(row)
    write_csv(args.output, rows, FIELDS)
    output_dir = args.output.parent / "global_local"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output": args.output.as_posix()}))


if __name__ == "__main__":
    main()

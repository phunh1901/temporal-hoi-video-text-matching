"""Prepare head controls and readiness; optionally run small explicit CPU tensor probes.

Run from the repository root. This command never starts benchmark training,
downloads inputs, evaluates a final split, or invents temporal/geometry features.
"""

import argparse
import copy
import json
from pathlib import Path

from train_alignment import alignment_readiness, run_synthetic_probe

from temporal_hoi.data.msrvtt import file_sha256, semantic_hash
from temporal_hoi.evaluation.experiment_audit import audit_zero_shot_usage, summarize_seeds
from temporal_hoi.training.alignment import validate_config, write_json


def prepare_study(spec, output, *, synthetic_probes=False):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new output directory; existing study artifacts are preserved")
    if (spec.get("schema_version") != "development_head_study_v1"
            or spec.get("heads") != ["linear", "mlp"]
            or spec.get("selection_split") != "validation"
            or spec.get("selection_metric") != "text_to_video.R@1"
            or spec.get("final_test_access") is not False):
        raise ValueError("Require linear/MLP development-only study")
    seeds = spec.get("seeds", [])
    if (not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Require distinct nonnegative seeds")
    if (type(spec.get("probe_steps")) is not int or not 1 <= spec["probe_steps"] <= 10
            or type(spec.get("probe_fixture_seed")) is not int
            or spec["probe_fixture_seed"] < 0):
        raise ValueError("Require 1..10 probe steps and a fixed fixture seed")
    base = json.loads(Path(spec["base_config"]).read_text(encoding="utf-8-sig"))
    union = json.loads(Path(spec["union_config"]).read_text(encoding="utf-8-sig"))
    validate_config(base)
    validate_config(union)
    if base["crop_mode"] != "global" or union["crop_mode"] != "union":
        raise ValueError("Require explicit global and union base configurations")
    validate_config({**base, "head": "mlp", "hidden_dim": spec["hidden_dim"]})
    output.mkdir(parents=True)
    write_json(output / "study_config.json", spec)
    _, global_readiness = alignment_readiness(base)
    _, union_readiness = alignment_readiness(union)
    write_json(output / "global_readiness.json", global_readiness)
    write_json(output / "union_readiness.json", union_readiness)
    matrix = []
    for temporal, geometry in ((False, False), (True, False), (False, True), (True, True)):
        matrix.append({
            "condition": f"C{int(temporal)}{int(geometry)}", "temporal": temporal,
            "geometry": geometry, "crop_mode": "union", "head": "linear",
            "head_trained_in_control": True, "seeds": seeds,
            "budget": {k: union[k] for k in ("max_epochs", "patience", "lr", "weight_decay",
                                            "feature_batch_videos", "max_train_hours")},
            "status": "not_run", "metrics": None, "measured_cost": None,
            "input_blockers": union_readiness["blockers"],
            "implementation_blockers": [name for flag, name in (
                (temporal, "Temporal sequence trainer with timestamps/mask is not implemented"),
                (geometry, "Geometry/velocity encoder with verified boxes is not implemented")) if flag],
        })
    write_json(output / "ablation_matrix.json", {
        "scope": "planned_real_union_feature_experiments", "conditions": matrix,
        "runnable_temporal_geometry_trainer": False,
        "warning": "Global pooled head probes are not C00/C10/C01/C11 results",
    })
    runs, summaries = [], {}
    for head in spec["heads"]:
        records = []
        for seed in seeds:
            label = f"{head}_seed{seed}"
            config = copy.deepcopy(base)
            config.update(head=head, seed=seed, probe_fixture_seed=spec["probe_fixture_seed"],
                          output_dir=(output / "benchmark" / label).as_posix())
            if head == "mlp":
                config["hidden_dim"] = spec["hidden_dim"]
            validate_config(config)
            config_path = output / "configs" / f"{label}.json"
            write_json(config_path, config)
            run = {"head": head, "seed": seed, "config": config_path.as_posix(),
                   "config_sha256": file_sha256(config_path), "benchmark_status": "not_run",
                   "benchmark_metrics": None, "synthetic_probe": None}
            if synthetic_probes:
                probe_dir = output / "synthetic_probes" / label
                result = run_synthetic_probe(config, probe_dir, steps=spec["probe_steps"])
                run["synthetic_probe"] = {
                    "path": (probe_dir / "probe.json").as_posix(),
                    "sha256": file_sha256(probe_dir / "probe.json"), "status": result["status"],
                    "parameters": result["trainable_parameters"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "observed_rss_bytes": result["observed_rss_bytes"],
                }
                common = {k: v for k, v in config.items() if k not in {"seed", "output_dir"}}
                records.append({"seed": seed, "value": result["head_update_norm"],
                                "metric": "head_update_norm", "units": "L2_parameter_distance",
                                "comparison_sha256": semantic_hash(common),
                                "input_sha256": result["synthetic_feature_sha256"],
                                "execution_level": "synthetic_technical_probe"})
            runs.append(run)
        if records:
            summaries[head] = summarize_seeds(records)
    result = {
        "schema_version": "development_study_v1", "runs": runs,
        "probe_summaries": summaries, "benchmark_training_run": False,
        "final_split_evaluated": False, "model_selected": False,
        "paired_inputs": "Fixed tensor fixture across heads and optimization seeds",
        "limits": ["No benchmark quality, confidence interval or improvement claim from probes",
                   "Full matrix requires union features, geometry, sequence trainer and real runs",
                   "Per-run resource limits are not authorization for remote or paid compute"],
    }
    write_json(output / "study_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/development_head_study.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-probes", action="store_true")
    parser.add_argument("--zero-shot-manifest", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8-sig"))
    result = prepare_study(spec, args.output, synthetic_probes=args.synthetic_probes)
    if args.zero_shot_manifest:
        manifest = json.loads(args.zero_shot_manifest.read_text(encoding="utf-8-sig"))
        audit = audit_zero_shot_usage(manifest)
        audit["manifest_sha256"] = file_sha256(args.zero_shot_manifest)
        write_json(args.output / "zero_shot_usage_audit.json", audit)
        if audit["status"] == "failed":
            raise SystemExit(2)
    print(json.dumps({"output": str(args.output), "runs_prepared": len(result["runs"]),
                      "benchmark_training_run": False, "model_selected": False}))


if __name__ == "__main__":
    main()

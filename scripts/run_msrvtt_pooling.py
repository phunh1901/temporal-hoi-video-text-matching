"""Preflight or run frozen B0/B1/B2 retrieval from locked media or frame features."""

import argparse
import json
from pathlib import Path

from temporal_hoi.inference.pooling_workflow import execute, preflight, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/msrvtt_pooling.json"))
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--cache-sha256")
    parser.add_argument("--baseline-reference", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--build-cache", action="store_true",
                        help="Encode local media/checkpoint; otherwise replay an explicit cache")
    parser.add_argument("--allow-implementation-only", action="store_true",
                        help="Permit internal formula parity without claiming external baseline parity")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-final-evaluation", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for key, value in (("feature_cache", args.feature_cache),
                       ("feature_cache_sha256", args.cache_sha256),
                       ("baseline_reference", args.baseline_reference),
                       ("reference_sha256", args.reference_sha256)):
        if value is not None:
            config[key] = str(value)
    if args.allow_implementation_only:
        config["require_external_reference"] = False
    output = args.output_dir or Path(config["output_dir"])
    if output.exists():
        parser.error("Output already exists; choose a new directory to preserve previous runs")
    output.mkdir(parents=True)
    try:
        pool, readiness = preflight(config, build_cache=args.build_cache)
        write_json(output / "readiness.json", readiness)
        if not args.execute:
            print(json.dumps(readiness))
            return 2 if readiness["blockers"] else 0
        if readiness["blockers"]:
            raise ValueError("Preflight blocked: " + "; ".join(readiness["blockers"]))
        if config["split"] == "test" and (not args.allow_final_evaluation
                                              or config.get("selection_fixed_before_scores") is not True):
            raise ValueError("Test requires fixed selection and --allow-final-evaluation")
        result = execute(config, pool, output, build_cache=args.build_cache)
        write_json(output / "run.json", result)
        print(json.dumps(result))
        return 0
    except Exception as exc:
        failed = {"status": "failed", "benchmark_inference_run": False,
                  "benchmark_metrics": None, "error": str(exc)}
        write_json(output / "run.json", failed)
        print(json.dumps(failed))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

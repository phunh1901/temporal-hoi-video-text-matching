"""Compare full-frame, object-only and union pixels on one fixed oracle train probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_hoi.inference.hoi_diagnostics import VARIANTS, run_representation_diagnostics
from temporal_hoi.inference.hoi_static import preflight, proposal_pairs, resolve_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    print("DIAGNOSTIC_PREFLIGHT_START: verifying fixed train cohort and local input hashes",
          flush=True)
    config = resolve_config(json.loads(args.config.read_text(encoding="utf-8")), args.config)
    if args.preflight_only:
        _, frames, _ = preflight(config)
        print(json.dumps({"status": "preflight_passed", "variants": list(VARIANTS),
                          "num_frames": len(frames),
                          "num_proposal_pairs": sum(len(proposal_pairs(frame)) for frame in frames),
                          "model_inference_run": False, "model_training_run": False}))
        return
    receipt = run_representation_diagnostics(config, args.output_dir)
    print(json.dumps({"status": receipt["status"], "level": receipt["level"],
                      "variants": receipt["variants"], "num_frames": receipt["num_frames"],
                      "num_proposal_pairs": receipt["num_proposal_pairs"],
                      "receipt": str((args.output_dir / "receipt.json").resolve())}))


if __name__ == "__main__":
    main()

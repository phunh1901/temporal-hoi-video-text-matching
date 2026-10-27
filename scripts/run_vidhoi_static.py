"""Run a bounded frozen-CLIP oracle VidHOI train probe from local hashed files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_hoi.inference.hoi_static import preflight, resolve_config, run_static_probe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Verify all input hashes/images without loading an encoder")
    args = parser.parse_args(argv)
    print("VIDHOI_STATIC_STARTED: train oracle probe; no training or final evaluation", flush=True)
    config = resolve_config(json.loads(args.config.read_text(encoding="utf-8")), args.config)
    if args.preflight_only:
        _, frames, _ = preflight(config)
        print(json.dumps({"status": "preflight_passed", "num_frames": len(frames),
                          "model_inference_run": False, "model_training_run": False}))
        return
    receipt = run_static_probe(config, args.output_dir)
    print(json.dumps({"status": receipt["status"], "level": receipt["level"],
                      "num_frames": receipt["num_frames"],
                      "num_proposal_pairs": receipt["num_proposal_pairs"],
                      "train_probe_map": receipt["metrics"]["map"],
                      "receipt": str((args.output_dir / "receipt.json").resolve())}))


if __name__ == "__main__":
    main()

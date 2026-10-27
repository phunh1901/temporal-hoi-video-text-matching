"""Lock local VidHOI annotation/frame-mask/split inputs without downloading data."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from temporal_hoi.data.vidhoi_lock import file_sha256, lock_vidhoi_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Input specification JSON; paths resolve against --base-dir")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true",
                        help="Exit 2 after saving the blocked artifact when inputs are missing")
    args = parser.parse_args()
    spec = json.loads(args.inputs.read_text(encoding="utf-8"))
    output = lock_vidhoi_inputs(spec, base_dir=args.base_dir)
    output.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "input_spec_sha256": file_sha256(args.inputs),
                   "python_version": platform.python_version()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": output["status"], "missing_inputs": output["missing_inputs"]}))
    if args.require_ready and not output["input_lock_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

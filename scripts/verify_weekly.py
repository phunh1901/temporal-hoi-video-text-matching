"""Verify W01/W02 code and the small pilot without downloads or model training.

Run from the project root: python scripts/verify_weekly.py
Writes outputs/w02/completion_verification.json, preserving earlier audit logs.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from audit_data import audit

from temporal_hoi.data import TemporalHOIDataset
from temporal_hoi.evaluation import compute_bidirectional_retrieval


def main():
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root:
        raise SystemExit("Run from the project root so manifest video paths resolve correctly")
    checks = {}
    for name, arguments in (
        ("pytest", ["-m", "pytest", "tests", "-q"]),
        ("ruff", ["-m", "ruff", "check", "src", "tests", "scripts"]),
    ):
        run = subprocess.run([sys.executable, *arguments], capture_output=True, text=True)
        checks[name] = {
            "exit_code": run.returncode,
            "output": (run.stdout + run.stderr).strip(),
        }
        print(name, checks[name]["output"])

    checks["pilot_audit"] = audit()
    dataset = TemporalHOIDataset(
        relevance_csv="data/manifests/relevance.csv", num_frames=8, max_frame_side=320
    )
    clips, errors = [], []
    for index in range(len(dataset)):
        try:
            item = dataset[index]
            frames, times = item["pil_frames"], item["timestamps"]
            valid = (
                len(frames) == len(times) == 8
                and all(max(frame.size) <= 320 for frame in frames)
                and all(item["start_s"] <= t < item["end_s"] + 1e-6 for t in times)
                and all(b > a for a, b in zip(times, times[1:]))
                and len(item["positive_prompts"]) == 2
                and len(item["negative_prompts"]) == 8
                and item["has_violation"] is None
            )
            clips.append({"clip_id": item["clip_id"], "split": item["split"], "passed": valid})
            if not valid:
                errors.append(f"Invalid pilot sample: {item['clip_id']}")
        except (ValueError, OSError, KeyError) as exc:
            errors.append(f"Sample {index}: {exc}")
    checks["full_interval_decode"] = {"clips": clips, "errors": errors}
    checks["hand_calculated_retrieval"] = compute_bidirectional_retrieval(
        [[0.8, 0.9, 0.1], [0.7, 0.6, 0.95]], [[0], [1]], [[0], [1], None], [1, 2, 5, 10]
    )
    smoke = json.loads((root / "outputs/w01/smoke_test.json").read_text(encoding="utf-8"))
    checks["existing_smoke_log"] = {
        "path": "outputs/w01/smoke_test.json",
        "status": smoke["status"],
        "executed_again": False,
    }
    passed = (
        checks["pytest"]["exit_code"] == checks["ruff"]["exit_code"] == 0
        and checks["pilot_audit"]["status"] == "PASS_TECHNICAL_PILOT"
        and len(clips) == 8
        and not errors
        and smoke["status"] == "SUCCESS"
    )
    inputs = [
        "pyproject.toml", "uv.lock", "configs/w02_lightweight.json",
        "outputs/w01/smoke_test.json", "outputs/w01/environment.json",
        "outputs/w02/pilot_manifest_lock.json", "scripts/verify_weekly.py",
        "scripts/audit_data.py",
    ]
    inputs += [str(path) for folder in ("src", "tests") for path in Path(folder).rglob("*.py")]
    report = {
        "verified_at": datetime.now().astimezone().isoformat(),
        "status": "PASS_W01_W02_TECHNICAL_SCOPE" if passed else "FAIL",
        "scope": "Weekly preparation/pilot and added retrieval metrics, not benchmark quality",
        "checks": checks,
        "sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in inputs
        },
        "not_claimed": [
            "HOI benchmark reproduction", "learned alignment benchmark results",
            "human-independent label review", "report submission or supervisor approval",
        ],
    }
    output = root / "outputs/w02/completion_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(report["status"], "| full pilot clips:", len(clips), "|", output.relative_to(root))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

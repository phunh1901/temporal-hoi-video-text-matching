"""Restore reviewed Kaggle source snapshots without publishing private run inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe(root: Path, name: str) -> Path:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe source path: {name}")
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError(f"Source path escapes root: {name}")
    return target


def verify_sources(source_root: Path) -> dict:
    """Verify every published Python/reference byte against the captured source pins."""
    source_root = Path(source_root)
    provenance = json.loads((source_root / "provenance.json").read_text(encoding="utf-8"))
    for name, record in provenance["files"].items():
        _safe(source_root, name)
        data = _safe(source_root, record["path"]).read_bytes()
        if len(data) != record["bytes"] or _digest(data) != record["sha256"]:
            raise ValueError(f"Captured source SHA256 mismatch: {name}")
    return provenance


def _write_new_or_equal(target: Path, data: bytes) -> None:
    if target.exists():
        if not target.is_file() or target.read_bytes() != data:
            raise ValueError(f"Existing working file differs; refusing overwrite: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)


def restore_sources(source_root: Path, destination: Path) -> dict:
    """Copy source only; restore vendor .py.txt snapshots to their original .py names."""
    source_root, destination = Path(source_root), Path(destination)
    provenance = verify_sources(source_root)
    if destination.is_symlink():
        raise ValueError("Working source directory must not be a link")
    # Verify the complete source set before creating any working files.
    for name, record in provenance["files"].items():
        target = _safe(destination, name)
        data = _safe(source_root, record["path"]).read_bytes()
        _write_new_or_equal(target, data)
    return provenance


def capture_stage(
    source_root: Path, code_dir: Path, work_dir: Path, stage: str, input_root: Path,
) -> dict:
    """Recreate source capture; require withheld data manifests as verified external inputs.

    This function never downloads data or runs model code. The caller prepares
    the separately pinned tokenizer before pooling/training if it is required.
    """
    provenance = verify_sources(Path(source_root))
    pins = provenance["stages"][stage]["original_source_sha256"]
    code_dir, work_dir, input_root = Path(code_dir), Path(work_dir), Path(input_root)
    if work_dir.is_symlink():
        raise ValueError("Working result directory must not be a link")
    payloads = {}
    for name, expected in pins.items():
        target = _safe(code_dir, name)
        if target.is_file():
            data = target.read_bytes()
            if _digest(data) != expected:
                raise ValueError(f"Working source/input SHA256 mismatch: {name}")
        elif name in provenance["files"]:
            raise ValueError(f"Restore sources first; missing source: {name}")
        else:
            matches = [p for p in input_root.rglob(Path(name).name)
                       if p.is_file() and p.stat().st_size <= 32 * 1024**2]
            matches = [p for p in matches if not p.is_symlink()
                       and p.resolve().is_relative_to(input_root.resolve())]
            matches = [p for p in matches if _digest(p.read_bytes()) == expected]
            if not matches:
                raise FileNotFoundError(f"Attach the original input with SHA256 {expected}: {name}")
            data = matches[0].read_bytes()
        payloads[name] = data
    # Missing/corrupt inputs fail before writing the source capture or copying inputs.
    for name, data in payloads.items():
        _write_new_or_equal(_safe(code_dir, name), data)
    manifest = (json.dumps(pins, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), data)
    _write_new_or_equal(_safe(work_dir, "source_manifest.json"), manifest)
    _write_new_or_equal(_safe(work_dir, "runtime_sources.zip"), buffer.getvalue())
    return pins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    if args.verify_only:
        print(json.dumps({"verified_source_files": len(verify_sources(source)["files"])}))
    elif args.destination:
        restore_sources(source, args.destination)
        print(json.dumps({"restored": str(args.destination)}))
    else:
        parser.error("Choose --verify-only or provide --destination")


if __name__ == "__main__":
    main()

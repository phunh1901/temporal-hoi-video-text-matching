"""Portable, offline-first handoff for a small VidHOI train-only oracle probe.

The notebook embeds this standard-library bootstrap before any bundled source is
imported. A pinned manifest authenticates each copied file. This module never
downloads model weights or data, and it never treats a probe as a benchmark.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import queue
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

BUNDLE_SCHEMA = "vidhoi_oracle_train_bundle_v1"
DEVELOPMENT_BUNDLE_SCHEMA = "vidhoi_frozen_development_bundle_v1"
PAIR_DIAGNOSTIC_BUNDLE_SCHEMA = "vidhoi_pair_diagnostics_bundle_v1"
MANIFEST_NAME = "bundle_manifest.json"
MAX_MANIFEST_BYTES = 4 * 1024**2
MAX_BUNDLE_BYTES = 2 * 1024**3
MAX_FILES = 4096
COPY_CHUNK = 1024**2
WHEEL_DISTRIBUTIONS = {
    "open_clip_torch", "safetensors", "ftfy", "regex", "timm", "huggingface_hub",
    "tqdm", "packaging", "pyyaml", "fsspec", "filelock", "requests", "wcwidth",
    "typing_extensions", "hf_xet", "typer_slim", "typer", "click", "shellingham",
    "charset_normalizer", "idna", "urllib3", "certifi", "httpx", "httpcore", "h11",
    "anyio", "sniffio", "exceptiongroup", "annotated_doc",
}
DEPENDENCY_SPECS = {
    "open_clip": "open-clip-torch==3.3.0", "safetensors": "safetensors>=0.4,<1",
    "ftfy": "ftfy>=6,<7", "regex": "regex", "timm": "timm>=1,<2",
    "huggingface_hub": "huggingface-hub>=0.30,<2", "tqdm": "tqdm",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member(name: str) -> PurePosixPath:
    """Require a canonical relative POSIX member on both Windows and Linux."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError(f"Unsafe bundle path: {name!r}")
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or str(path) != name
            or path == PurePosixPath(".")):
        raise ValueError(f"Unsafe bundle path: {name!r}")
    return path


def wheel_distribution(name: str) -> str:
    safe_member(name)
    if "/" in name or not name.endswith(".whl") or len(name.split("-")) < 5:
        raise ValueError(f"Expected a portable wheel filename: {name}")
    distribution = name.split("-", 1)[0].lower().replace("-", "_")
    if distribution not in WHEEL_DISTRIBUTIONS:
        raise ValueError(f"Wheel distribution is not allowlisted: {distribution}")
    return distribution


def validate_requirements(text: str) -> list[str]:
    """Accept pinned allowlisted distributions only, never pip options or URLs."""
    requirements = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)", line)
        if match is None:
            raise ValueError("Offline requirements must contain only name==version pins")
        name = match[1].lower().replace("-", "_").replace(".", "_")
        if name not in WHEEL_DISTRIBUTIONS:
            raise ValueError(f"Dependency is not allowlisted: {name}")
        requirements.append(line)
    if not requirements or len(requirements) != len(set(requirements)):
        raise ValueError("Offline requirements must be nonempty and unique")
    if "open-clip-torch==3.3.0" not in requirements:
        raise ValueError("Offline requirements must pin open-clip-torch==3.3.0")
    return requirements


def validate_manifest(data: bytes) -> dict:
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Bundle manifest is too large")
    manifest = json.loads(data)
    if not isinstance(manifest, dict):
        raise ValueError("Require a VidHOI bundle object")
    development = manifest.get("schema_version") == DEVELOPMENT_BUNDLE_SCHEMA
    pair_diagnostic = manifest.get("schema_version") == PAIR_DIAGNOSTIC_BUNDLE_SCHEMA
    purpose = ("head_pair_diagnostics" if pair_diagnostic else
               "head_development" if development else "train_probe")
    if (manifest.get("schema_version") not in {
            BUNDLE_SCHEMA, DEVELOPMENT_BUNDLE_SCHEMA, PAIR_DIAGNOSTIC_BUNDLE_SCHEMA}
            or manifest.get("split") != "train" or manifest.get("purpose") != purpose
            or manifest.get("box_source") != "oracle"
            or manifest.get("object_label_source") != "oracle"
            or manifest.get("benchmark_evaluated") is not False
            or manifest.get("training_run") is not False):
        raise ValueError("Require a train-only oracle probe bundle, never a benchmark claim")
    if (development or pair_diagnostic) and (
            manifest.get("official_final_split_accessed") is not False
            or manifest.get("final_unseen_accessed") is not False):
        raise ValueError("Development bundle must keep final evaluation closed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > MAX_FILES:
        raise ValueError("Invalid bundle file inventory")
    total = 0
    for name, record in files.items():
        safe_member(name)
        if name == MANIFEST_NAME or not isinstance(record, dict):
            raise ValueError("Invalid bundle file record")
        size, digest = record.get("bytes"), record.get("sha256")
        if (type(size) is not int or size < 0 or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"Invalid size/SHA256: {name}")
        total += size
    if total != manifest.get("total_bytes") or total > MAX_BUNDLE_BYTES:
        raise ValueError("Bundle byte inventory exceeds its limit or differs from total_bytes")
    entrypoints = ({"scripts/run_vidhoi_pair_diagnostics.py"} if pair_diagnostic else
                   {"scripts/run_vidhoi_head_study.py"} if development else {
                       "scripts/run_vidhoi_static.py", "scripts/run_vidhoi_diagnostics.py"})
    if manifest.get("entrypoint") not in entrypoints:
        raise ValueError("Unexpected bundle entrypoint")
    required = {manifest["entrypoint"], "assets/run_config.json",
                "assets/class_mapping.json", "src/temporal_hoi/inference/hoi_kaggle.py"}
    if pair_diagnostic:
        source_member = manifest.get("source_result_member", "assets/source_result.zip")
        source_names = {"assets/source_result.zip", "assets/source_result.bin"}
        if (not isinstance(source_member, str) or source_member not in source_names
                or source_names.intersection(files) != {source_member}):
            raise ValueError("Pair bundle must declare exactly one pinned source result member")
        required |= {source_member, "assets/protocol.json",
                     "assets/review/review_manifest.json",
                     "src/temporal_hoi/evaluation/hoi_pair_diagnostics.py",
                     "src/temporal_hoi/data/hoi_pair_review.py"}
    elif development:
        required |= {"assets/train_manifest.json", "assets/dev_manifest.json",
                     "assets/protocol.json", "src/temporal_hoi/data/hoi_head_cache.py",
                     "src/temporal_hoi/models/hoi_head.py",
                     "src/temporal_hoi/training/hoi_head_study.py"}
    else:
        required.add("assets/probe_manifest.json")
    if manifest["entrypoint"] == "scripts/run_vidhoi_diagnostics.py":
        required |= {"src/temporal_hoi/inference/hoi_diagnostics.py",
                     "src/temporal_hoi/inference/hoi_static.py"}
    if not required <= files.keys():
        raise ValueError("Bundle is missing required source or configuration")
    for name in files:
        if name.startswith("wheelhouse/") and name != "wheelhouse/requirements-offline.txt":
            wheel_distribution(name.split("/", 1)[1])
    return manifest


def _inside_file(root: Path, name: str) -> Path:
    safe_member(name)
    root = root.resolve()
    target = root / name
    cursor = target
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"Symlinks are forbidden in bundle paths: {name}")
        cursor = cursor.parent
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Bundle path escapes its root: {name}")
    return target


def _copy_verified(stream, destination: Path, record: dict, *, name: str, emit) -> None:
    if destination.exists():
        if (not destination.is_file() or destination.stat().st_size != record["bytes"]
                or sha256_file(destination) != record["sha256"]):
            raise ValueError(f"Existing destination differs: {name}; use a fresh work directory")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    if temporary.exists():
        raise ValueError(f"Interrupted copy exists: {temporary.name}; use a fresh work directory")
    digest, received = hashlib.sha256(), 0
    try:
        with temporary.open("xb") as output:
            while chunk := stream.read(COPY_CHUNK):
                received += len(chunk)
                if received > record["bytes"]:
                    raise ValueError(f"Input exceeds declared bytes: {name}")
                digest.update(chunk)
                output.write(chunk)
                if received % (16 * COPY_CHUNK) < COPY_CHUNK:
                    emit("input_copy_progress", file=name, bytes=received)
        if received != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Input hash/size mismatch: {name}")
        temporary.replace(destination)
    except BaseException:
        # This is only this invocation's newly created partial file.
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise
    emit("input_file_verified", file=name, bytes=received)


def install_input(input_root: str | Path, destination: str | Path,
                  trusted_manifest_sha256: str, *, progress=None) -> dict:
    """Discover a renamed ZIP or expanded Kaggle Dataset by its pinned manifest.

    Sources are authenticated before importing any code. Identical ZIP/expanded
    copies are acceptable; no filename, Dataset title or untrusted manifest can
    authorize a different payload. No source symlink is traversed.
    """
    input_root, destination = Path(input_root), Path(destination)
    emit = progress or (lambda *args, **kwargs: None)
    candidates, archives, visible, scan_errors = [], [], [], []
    skipped_symlinks = 0
    if not input_root.is_dir() or input_root.is_symlink() or destination.is_symlink():
        raise ValueError("Input root and destination must be ordinary directories")
    for number, (directory, subdirs, names) in enumerate(
        os.walk(input_root, followlinks=False,
                onerror=lambda error: scan_errors.append(str(error))), start=1
    ):
        if number > 1000:
            raise ValueError("Input scan exceeded 1000 directories; attach only this probe Dataset")
        ordinary_subdirs = [name for name in subdirs
                            if not (Path(directory) / name).is_symlink()]
        skipped_symlinks += len(subdirs) - len(ordinary_subdirs)
        subdirs[:] = ordinary_subdirs
        for name in names:
            source = Path(directory) / name
            if source.is_symlink():
                skipped_symlinks += 1
                continue
            if len(visible) < 25:
                visible.append(str(source.relative_to(input_root)))
            if name == MANIFEST_NAME and source.stat().st_size <= MAX_MANIFEST_BYTES:
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() == trusted_manifest_sha256:
                    candidates.append(("directory", source.parent, data))
                    subdirs[:] = []
            elif name.lower().endswith(".zip"):
                archives.append(source)
    for source in sorted(archives):
        try:
            with zipfile.ZipFile(source) as archive:
                if MANIFEST_NAME not in archive.namelist():
                    continue
                entry = archive.getinfo(MANIFEST_NAME)
                if entry.file_size > MAX_MANIFEST_BYTES:
                    continue
                data = archive.read(entry)
                if hashlib.sha256(data).hexdigest() == trusted_manifest_sha256:
                    candidates.append(("zip", source, data))
        except zipfile.BadZipFile:
            continue
    emit("input_scan_completed", input_root=str(input_root), visible_files=visible,
         skipped_symlinks=skipped_symlinks, scan_errors=scan_errors)
    if not candidates:
        if scan_errors:
            raise OSError(f"Cannot read probe input at {input_root}: {scan_errors}")
        if not visible:
            raise FileNotFoundError(
                f"No readable input files at {input_root}. In the notebook editor, use "
                "Add Input to attach the private Dataset containing this package's ZIP "
                "or expanded folder. Wait for Dataset processing to finish, then rerun cell 1. "
                "Importing the notebook alone does not attach its data. "
                f"Skipped symlinks: {skipped_symlinks}; Visible files: []")
        raise FileNotFoundError("No input with the pinned VidHOI probe manifest. "
                                "Attach the ZIP paired with this notebook version. "
                                f"Expected manifest SHA256: {trusted_manifest_sha256}. "
                                f"Visible files: {visible}")
    kind, source, manifest_bytes = candidates[0]
    manifest = validate_manifest(manifest_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < manifest["total_bytes"] + 64 * COPY_CHUNK:
        raise OSError("Insufficient disk space to copy the verified probe input")
    emit("input_discovered", format=kind, files=len(manifest["files"]),
         bytes=manifest["total_bytes"])

    if kind == "zip":
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP member names")
            if set(names) != set(manifest["files"]) | {MANIFEST_NAME}:
                raise ValueError("Unexpected or missing ZIP members")
            for info in infos:
                safe_member(info.filename.rstrip("/") if info.is_dir() else info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("ZIP symlinks are forbidden")
                if info.flag_bits & 1:
                    raise ValueError("Encrypted ZIP members are unsupported")
                if info.is_dir() or info.filename == MANIFEST_NAME:
                    continue
                record = manifest["files"][info.filename]
                if info.file_size != record["bytes"]:
                    raise ValueError(f"ZIP size mismatch: {info.filename}")
            for name, record in manifest["files"].items():
                with archive.open(name) as stream:
                    _copy_verified(stream, _inside_file(destination, name), record,
                                   name=name, emit=emit)
    else:
        # Check the complete expanded inventory before copying hundreds of review images.
        for name, record in manifest["files"].items():
            path = _inside_file(source, name)
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                found = (f"{path.stat().st_size} bytes" if path.is_file() else
                         "directory" if path.is_dir() else "missing")
                hint = ""
                if name in {"assets/source_result.zip", "assets/source_result.bin"}:
                    hint = (" Dataset processing may have expanded the nested result ZIP. "
                            "Use the matched rebuilt notebook and input bundle containing "
                            "assets/source_result.bin. Do not reconstruct the original ZIP: "
                            "its exact source SHA256 is required.")
                raise ValueError(f"Missing/size-mismatched expanded input: {name} "
                                 f"(expected {record['bytes']} bytes; found {found}).{hint}")
        for name, record in manifest["files"].items():
            path = _inside_file(source, name)
            with path.open("rb") as stream:
                _copy_verified(stream, _inside_file(destination, name), record,
                               name=name, emit=emit)
    target_manifest = _inside_file(destination, MANIFEST_NAME)
    if target_manifest.exists() and target_manifest.read_bytes() != manifest_bytes:
        raise ValueError("Existing destination has a different bundle manifest")
    target_manifest.write_bytes(manifest_bytes)
    return {"schema_version": manifest["schema_version"], "format": kind,
            "manifest_sha256": trusted_manifest_sha256, "files": len(manifest["files"]),
            "bytes": manifest["total_bytes"], "model_inference_run": False,
            "benchmark_evaluated": False, "training_run": False}


def verify_installed_input(root: str | Path, expected_manifest_sha256: str) -> dict:
    """Recheck all installed bytes immediately before importing/running source."""
    root = Path(root)
    manifest_path = _inside_file(root, MANIFEST_NAME)
    data = manifest_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_manifest_sha256:
        raise ValueError("Installed bundle manifest hash mismatch")
    manifest = validate_manifest(data)
    for name, record in manifest["files"].items():
        path = _inside_file(root, name)
        if (not path.is_file() or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]):
            raise ValueError(f"Installed bundle hash/size mismatch: {name}")
    return manifest


def dependency_plan(root: str | Path, *, target: str | Path | None = None,
                    allow_internet: bool = False) -> list[str] | None:
    """Build a pip command which can never replace Torch/Torchvision implicitly."""
    root = Path(root)
    target = Path(target) if target is not None else root.parent / "python_deps"
    wheels = sorted((root / "wheelhouse").glob("*.whl"))
    if wheels:
        for wheel in wheels:
            wheel_distribution(wheel.name)
        requirements = root / "wheelhouse/requirements-offline.txt"
        validate_requirements(requirements.read_text(encoding="utf-8"))
        return [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                "--disable-pip-version-check", "--target", str(target), "--upgrade",
                "--find-links", str(root / "wheelhouse"), "-r", str(requirements)]
    missing = [name for name in DEPENDENCY_SPECS if importlib.util.find_spec(name) is None]
    try:
        correct_clip = importlib.metadata.version("open-clip-torch") == "3.3.0"
    except importlib.metadata.PackageNotFoundError:
        correct_clip = False
    if not correct_clip and "open_clip" not in missing:
        missing.append("open_clip")
    if not missing:
        return None
    if not allow_internet:
        raise RuntimeError("Missing probe dependencies and no bundled wheelhouse: "
                           + ", ".join(missing) + ". Use the complete input bundle, or explicitly "
                           "enable dependency Internet in cell 2. Data/model downloads stay disabled.")
    return [sys.executable, "-m", "pip", "install", "--no-deps",
            "--disable-pip-version-check", "--target", str(target), "--upgrade",
            *[DEPENDENCY_SPECS[name] for name in missing]]


def environment_report() -> dict:
    """Import required libraries after offline wheel installation; never load a model."""
    import numpy
    import open_clip
    import PIL
    import safetensors
    import torch
    import torchvision

    if importlib.metadata.version("open-clip-torch") != "3.3.0":
        raise RuntimeError("This probe requires open-clip-torch 3.3.0")
    return {"python": sys.version, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": numpy.__version__,
            "pillow": PIL.__version__, "safetensors": safetensors.__version__,
            "open_clip": getattr(open_clip, "__version__", "3.3.0"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_loaded": False, "training_run": False, "benchmark_evaluated": False}


def prepare_run_config(root: str | Path, destination: str | Path, *, device: str) -> Path:
    """Resolve packaged paths when copying a reviewed config to a fresh result run."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    root, destination = Path(root), Path(destination)
    packaged = root / "assets/run_config.json"
    config = json.loads(packaged.read_text(encoding="utf-8"))
    if config.get("schema_version") == "vidhoi_static_config_v1":
        path_fields = ("manifest", "mapping", "checkpoint")
    elif config.get("schema_version") == "vidhoi_head_study_config_v1":
        path_fields = ("train_manifest", "dev_manifest", "protocol", "mapping", "checkpoint")
    elif config.get("schema_version") == "vidhoi_pair_diagnostics_config_v1":
        path_fields = ("source_result", "protocol", "mapping", "review_manifest", "annotations")
    else:
        raise ValueError("Unexpected static probe configuration")
    for name in path_fields:
        value = config.get(name)
        safe_member(value)
        path = (packaged.parent / value).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaping packaged {name}")
        config[name] = str(path)
    config["device"] = device
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return destination


class RunRecorder:
    """Stream subprocess logs and keep a downloadable receipt on ordinary failures."""

    def __init__(self, results: str | Path, cwd: str | Path, env=None, heartbeat_seconds=15):
        self.results, self.cwd = Path(results), Path(cwd)
        self.env = dict(os.environ) if env is None else dict(env)
        self.heartbeat_seconds = heartbeat_seconds
        self.results.mkdir(parents=True, exist_ok=False)

    def event(self, stage: str, **values) -> None:
        record = {"utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **values}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with (self.results / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def write_json(self, name: str, value) -> Path:
        path = _inside_file(self.results, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        return path

    def snapshot(self) -> Path:
        target = self.results.with_suffix(".zip")
        temporary = target.with_suffix(".partial.zip")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.results.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Result symlinks are forbidden")
                if path.is_file():
                    archive.write(path, path.relative_to(self.results).as_posix())
        temporary.replace(target)
        print(f"RESULT_ARCHIVE: {target}", flush=True)
        return target

    def command(self, label: str, command: list[str], *, timeout_seconds=3600) -> None:
        self.event("command_start", label=label)
        started, process = time.monotonic(), None
        lines = queue.Queue()
        try:
            process = subprocess.Popen(list(map(str, command)), cwd=self.cwd, env=self.env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)

            def read_output():
                try:
                    for line in process.stdout:
                        lines.put(line)
                finally:
                    lines.put(None)

            threading.Thread(target=read_output, daemon=True).start()
            with (self.results / "console.log").open("a", encoding="utf-8") as log:
                while True:
                    elapsed = time.monotonic() - started
                    if elapsed >= timeout_seconds:
                        raise TimeoutError(f"{label} exceeded {timeout_seconds} seconds")
                    try:
                        line = lines.get(timeout=min(self.heartbeat_seconds,
                                                     timeout_seconds - elapsed))
                    except queue.Empty:
                        self.event("heartbeat", label=label, elapsed_s=round(elapsed, 1))
                        continue
                    if line is None:
                        break
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
            returncode = process.wait(timeout=max(0.1, timeout_seconds - (time.monotonic() - started)))
            if returncode:
                raise RuntimeError(f"{label} failed with exit {returncode}; keep the result ZIP")
            self.event("command_completed", label=label,
                       elapsed_s=round(time.monotonic() - started, 3))
            self.snapshot()
        except BaseException as error:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            self.event("command_failed", label=label, error=f"{type(error).__name__}: {error}")
            self.snapshot()
            raise

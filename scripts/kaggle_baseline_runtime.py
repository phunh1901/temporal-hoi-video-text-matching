"""Portable input validation and streamed, durable logs for the Kaggle notebook."""

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

RECOVERABLE_ASSETS = {
    "external/CLIP4Clip/modules/bpe_simple_vocab_16e6.txt.gz",
    "data/metadata/msrvtt_data.zip",
    # Authenticated historical bundles retain their original internal layout.
    "outputs/w03/sources/m2/msrvtt_data.zip",
}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def safe_member(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or str(path) != name):
        raise ValueError(f"Unsafe bundle path: {name}")
    return path


def download_recovery(url, size, expected_hash, progress=None):
    """Read a pinned public non-code asset with bounded size/time and visible progress."""
    emit = progress or (lambda *args, **kwargs: None)
    for attempt in range(1, 4):
        emit("asset_download_start", url=url, attempt=attempt, expected_bytes=size)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "baseline-kaggle/4"})
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError("Recovery requires a complete HTTP 200 response")
                chunks, received = [], 0
                while chunk := response.read(min(256 * 1024, size + 1 - received)):
                    received += len(chunk)
                    if received > size:
                        raise ValueError("Recovery response exceeds pinned byte count")
                    chunks.append(chunk)
                    emit("asset_download_progress", received_bytes=received, expected_bytes=size)
            data = b"".join(chunks)
            if len(data) != size or digest_bytes(data) != expected_hash:
                raise ValueError("Recovery asset size/SHA256 mismatch")
            emit("asset_download_verified", bytes=size, sha256=expected_hash)
            return data
        except OSError as exc:
            emit("asset_download_retry", attempt=attempt, error=str(exc))
            if attempt == 3:
                raise RuntimeError("Pinned recovery download unavailable; check Kaggle Internet") from exc


def install_input(input_root, destination, trusted_manifests, recovery_assets=None, progress=None):
    """Accept renamed ZIP or Kaggle-expanded input, authenticated by manifest bytes."""
    input_root, destination = Path(input_root), Path(destination)
    recovery_assets = {} if recovery_assets is None else recovery_assets
    recovered = []
    emit = progress or (lambda *args, **kwargs: None)
    candidates = []
    archives = []
    scanned = 0
    for directory, subdirs, files in os.walk(input_root, followlinks=False):
        scanned += 1
        if scanned > 500:
            raise ValueError("Input scan exceeded 500 directories; attach only the baseline Dataset")
        emit("input_scan", directory=str(directory), directories_scanned=scanned)
        if "bundle_manifest.json" in files:
            path = Path(directory) / "bundle_manifest.json"
            if path.stat().st_size <= 1024**2:
                data = path.read_bytes()
                if digest_bytes(data) in trusted_manifests:
                    candidates.append(("directory", path.parent, data))
                    subdirs[:] = []  # Do not traverse a verified Dataset's expanded inner archives.
                    continue
        archives.extend(Path(directory) / name for name in files if name.lower().endswith(".zip"))
    for path in sorted(archives):
        try:
            with zipfile.ZipFile(path) as archive:
                if "bundle_manifest.json" in archive.namelist():
                    info = archive.getinfo("bundle_manifest.json")
                    if info.file_size > 1024**2:
                        continue
                    data = archive.read(info)
                    if digest_bytes(data) in trusted_manifests:
                        candidates.append(("zip", path, data))
        except zipfile.BadZipFile:
            continue
    if not candidates:
        visible = [str(p.relative_to(input_root)) for p in input_root.rglob("*") if p.is_file()]
        raise FileNotFoundError("No verified baseline input. Expected ZIP or expanded "
                                "bundle_manifest.json; visible files: " + str(visible[:30]))
    # Identical expanded/ZIP copies are harmless. Different bundle versions are ambiguous.
    hashes = {digest_bytes(row[2]) for row in candidates}
    if len(hashes) != 1:
        raise ValueError("Multiple baseline versions attached; retain exactly one input version")
    kind, source, manifest_bytes = candidates[0]
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or len(manifest) > 200:
        raise ValueError("Invalid bundle manifest")
    for name in manifest:
        safe_member(name)
    destination.mkdir(parents=True, exist_ok=True)

    def save(name, data):
        if name in manifest and digest_bytes(data) != manifest[name]:
            raise ValueError(f"Input hash mismatch: {name}")
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Destination symlink escapes baseline directory")
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f"Existing source differs: {name}; use a fresh working folder")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        emit("input_file_verified", file=name, bytes=len(data))

    if kind == "zip":
        with zipfile.ZipFile(source) as archive:
            entries = [i for i in archive.infolist() if not i.is_dir()]
            names = [i.filename for i in entries]
            if (len(names) != len(set(names))
                    or set(names) != set(manifest) | {"bundle_manifest.json"}
                    or sum(i.file_size for i in entries) > 32 * 1024**2):
                raise ValueError("Unexpected bundle contents or excessive expansion")
            for info in entries:
                safe_member(info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Bundle symlinks are forbidden")
                save(info.filename, archive.read(info))
    else:
        for name in manifest:
            emit("input_file_check", file=name)
            path = source / name
            if not path.resolve().is_relative_to(source.resolve()) or path.is_symlink():
                raise ValueError(f"Unsafe expanded input path: {name}")
            if path.is_file():
                if path.stat().st_size > 32 * 1024**2:
                    raise ValueError(f"Invalid expanded input: {name}")
                save(name, path.read_bytes())
            elif name in RECOVERABLE_ASSETS and name in recovery_assets:
                emit("input_asset_recovery", file=name)
                cached = destination / name
                if (cached.is_file() and cached.resolve().is_relative_to(destination.resolve())
                        and cached.stat().st_size <= 32 * 1024**2):
                    data = cached.read_bytes()
                else:
                    asset = recovery_assets[name]
                    data = asset() if callable(asset) else asset
                if not isinstance(data, bytes) or len(data) > 32 * 1024**2:
                    raise ValueError(f"Invalid recovery asset: {name}")
                # Original archive bytes must still match the authenticated manifest.
                # Never recreate ZIP/GZIP from extracted content or bypass a hash.
                save(name, data)
                recovered.append(name)
            else:
                nearby = [p.name for p in path.parent.iterdir()][:12] if path.parent.is_dir() else []
                raise ValueError(f"Missing expanded input: {name}; nearby files: {nearby}")
        save("bundle_manifest.json", manifest_bytes)
    return {"format": kind, "files": len(manifest),
            "manifest_sha256": digest_bytes(manifest_bytes), "recovered_exact_assets": recovered}


class LoggedCommands:
    def __init__(self, results, cwd, env, heartbeat_seconds=30):
        self.results, self.cwd, self.env = Path(results), Path(cwd), env
        self.heartbeat_seconds = heartbeat_seconds
        self.results.mkdir(parents=True, exist_ok=True)

    def event(self, stage, **values):
        record = {"utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **values}
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        with (self.results / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def snapshot(self):
        target = self.results.parent / (self.results.name + ".zip")
        partial = target.with_suffix(".partial.zip")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.results.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(self.results))
        partial.replace(target)
        return target

    def run(self, label, command, env=None, timeout=4500):
        """Stream stdout/stderr, heartbeat during silence, preserve failure and partial output."""
        self.event("start", label=label)
        started = time.monotonic()
        lines = queue.Queue()
        process = None
        captured = []
        try:
            process = subprocess.Popen(list(map(str, command)), cwd=self.cwd,
                                       env=self.env if env is None else env,
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
                    if elapsed > timeout:
                        raise TimeoutError(f"Stage {label} exceeded {timeout}s")
                    try:
                        line = lines.get(timeout=min(self.heartbeat_seconds, timeout - elapsed))
                    except queue.Empty:
                        self.event("heartbeat", label=label, elapsed_s=round(elapsed, 1))
                        continue
                    if line is None:
                        break
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    captured.append(line)
                    if len(captured) > 20000:
                        captured = captured[-10000:]
            code = process.wait(timeout=max(1, timeout - (time.monotonic() - started)))
            if code:
                raise RuntimeError(f"Stage {label} failed (exit {code}); see console.log")
            self.event("completed", label=label, elapsed_s=round(time.monotonic() - started, 1))
            return "".join(captured)
        except BaseException as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            self.event("failed", label=label, error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            self.snapshot()

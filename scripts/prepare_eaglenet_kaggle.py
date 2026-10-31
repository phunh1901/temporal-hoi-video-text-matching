"""Build a private, pinned EagleNet inference handoff; never upload or train."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import stat
import textwrap
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA = "eaglenet_eval_bundle_v1"
SOURCE_COMMIT = "00e6bcdcc13dbdefbc793ad9b811e3608cbddebe"
MAX_FILES = 20_000
MAX_BYTES = 16 * 1024**3


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    """Reject absolute, ambiguous and Windows-special archive member names."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError(f"Unsafe payload path: {name!r}")
    parts = name.split("/")
    if name.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe payload path: {name!r}")
    if any(part.endswith((".", " ")) for part in parts):
        raise ValueError(f"Ambiguous payload path: {name!r}")
    if any(part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL"}
           or part.split(".")[0].upper() in {f"COM{i}" for i in range(1, 10)}
           or part.split(".")[0].upper() in {f"LPT{i}" for i in range(1, 10)} for part in parts):
        raise ValueError(f"Windows device path: {name!r}")
    return PurePosixPath(name)


def plain_file(root, name):
    safe_name(name)
    root = Path(root).resolve()
    path = root.joinpath(*PurePosixPath(name).parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Payload escapes bundle root: {name}")
    for item in (path, *path.parents):
        if item == root.parent:
            break
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Symlink/reparse payload is forbidden: {name}")
    if not path.is_file():
        raise ValueError(f"Missing ordinary payload file: {name}")
    return path


def validate_manifest(manifest, *, allow_fixture=False, allow_external=False):
    """Check the locked contract without importing any bundled source."""
    if manifest.get("schema") != SCHEMA or type(manifest.get("fixture")) is not bool:
        raise ValueError("Require the explicit EagleNet bundle schema and fixture flag")
    if manifest["fixture"] and not allow_fixture:
        raise ValueError("Fixture bundles cannot be packaged as a real benchmark handoff")
    if manifest.get("source", {}).get("commit") != SOURCE_COMMIT:
        raise ValueError("Require the reviewed, pinned upstream source commit")
    if manifest.get("variant") not in {"ViT-B/16", "ViT-B/32"}:
        raise ValueError("Unsupported or missing checkpoint variant")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, dict) or not inventory or len(inventory) > MAX_FILES:
        raise ValueError("Require a bounded, nonempty payload inventory")
    folded = set()
    total = 0
    for name, record in inventory.items():
        safe_name(name)
        if name.casefold() in folded or name == "asset_manifest.json":
            raise ValueError("Duplicate, case-colliding or self-referential inventory path")
        folded.add(name.casefold())
        if not isinstance(record, dict) or type(record.get("bytes")) is not int:
            raise ValueError(f"Missing byte count: {name}")
        digest = record.get("sha256", "")
        if record["bytes"] < 0 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid size or SHA256: {name}")
        total += record["bytes"]
    if total > MAX_BYTES:
        raise ValueError("Bundle exceeds the extraction size limit")
    records = []
    if manifest.get("checkpoint") is not None:
        records.append(manifest["checkpoint"])
    else:
        request = manifest.get("checkpoint_request", {})
        if not allow_external or manifest["fixture"]:
            raise ValueError("A real checkpoint is required; external mode must be explicit")
        if (request.get("filename") != "msrvtt9k-vit_b_16.bin"
                or request.get("variant") != "ViT-B/16" or manifest["variant"] != "ViT-B/16"
                or type(request.get("bytes")) is not int or request["bytes"] <= 0
                or not request.get("source_urls") or not request.get("provenance")):
            raise ValueError("Require the pinned author checkpoint request and its source assertion")
        safe_name(request.get("path", ""))
        if not request["path"].startswith("weights/") or request["path"] in inventory:
            raise ValueError("External checkpoint destination must be new and inside weights/")
    if manifest.get("clip_checkpoint") is not None:
        records.append(manifest["clip_checkpoint"])
    metadata = manifest.get("metadata", {})
    records.extend(metadata.get(key) for key in ("test_csv", "train_csv", "annotations"))
    for record in records:
        if not isinstance(record, dict) or record.get("path") not in inventory:
            raise ValueError("A required checkpoint/metadata asset is missing from inventory")
        if any(record.get(key) != inventory[record["path"]].get(key) for key in ("bytes", "sha256")):
            raise ValueError("Asset identity differs from the top-level inventory")
    videos = manifest.get("videos", {})
    test_ids, probe_ids = videos.get("test_ids"), videos.get("probe_ids")
    if not isinstance(test_ids, list) or not isinstance(probe_ids, list):
        raise ValueError("Require locked test and train-probe ID lists")
    if len(set(test_ids)) != len(test_ids) or len(set(probe_ids)) != len(probe_ids):
        raise ValueError("Duplicate video IDs are forbidden")
    if set(test_ids) & set(probe_ids):
        raise ValueError("Train probe and benchmark video IDs must be disjoint")
    if not manifest["fixture"] and (len(test_ids) != 1000 or len(probe_ids) < 64):
        raise ValueError("A real handoff needs all 1000 test videos and at least 64 fixed train probes")
    files = videos.get("files", {})
    for video_id in test_ids + probe_ids:
        record = files.get(video_id)
        if not isinstance(record, dict) or record.get("path") not in inventory:
            raise ValueError(f"Missing media for locked video ID: {video_id}")
        if any(record.get(key) != inventory[record["path"]].get(key) for key in ("bytes", "sha256")):
            raise ValueError(f"Video identity differs: {video_id}")
    source = manifest["source"]
    source_root = str(safe_name(source.get("path", "")))
    source_files = source.get("files", {})
    if not source_files:
        raise ValueError("Pinned upstream source inventory is required")
    for name, record in source_files.items():
        joined = f"{source_root}/{safe_name(name)}"
        if inventory.get(joined) != record:
            raise ValueError(f"Source inventory mismatch: {joined}")
    config = manifest.get("config", {})
    for key, expected in (("max_frames", 12), ("max_words", 32), ("seed", 0)):
        if config.get(key) != expected:
            raise ValueError(f"The locked inference configuration must retain {key}={expected}")
    return manifest


def verify_inventory(root, manifest, *, allow_fixture=False, allow_external=False):
    validate_manifest(manifest, allow_fixture=allow_fixture, allow_external=allow_external)
    root = Path(root)
    if root.is_symlink() or getattr(root.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("Bundle root must not be a symlink or reparse point")
    expected = set(manifest["inventory"]) | {"asset_manifest.json"}
    found = set()
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Symlink/reparse payload is forbidden: {name}")
        if path.is_file():
            found.add(name)
    if found != expected:
        raise ValueError(f"Bundle file coverage differs: missing={sorted(expected-found)[:3]}, "
                         f"extra={sorted(found-expected)[:3]}")
    for name, record in manifest["inventory"].items():
        path = plain_file(root, name)
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Payload size/hash mismatch: {name}")
    return manifest


def safe_extract(archive_path, target, expected_manifest_sha):
    """Validate all members and the pinned manifest before writing any payload."""
    target = Path(target)
    if target.exists():
        raise FileExistsError("Extraction requires a new directory")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES or sum(info.file_size for info in infos) > MAX_BYTES:
            raise ValueError("Archive exceeds bounded file/byte limits")
        names = set()
        for info in infos:
            safe_name(info.filename)
            if info.filename.casefold() in names or info.flag_bits & 1:
                raise ValueError("Duplicate/case-colliding/encrypted ZIP member")
            names.add(info.filename.casefold())
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or mode & 0o170000 == 0):
                raise ValueError("Require ordinary, non-directory ZIP payload members")
        payload = archive.read("asset_manifest.json")
        if hashlib.sha256(payload).hexdigest() != expected_manifest_sha:
            raise ValueError("Input manifest is not the one pinned by this notebook")
        manifest = validate_manifest(json.loads(payload), allow_fixture=True, allow_external=True)
        expected = set(manifest["inventory"]) | {"asset_manifest.json"}
        if {info.filename for info in infos} != expected:
            raise ValueError("ZIP coverage differs from the pinned inventory")
        for info in infos:
            if info.filename in manifest["inventory"]:
                if info.file_size != manifest["inventory"][info.filename]["bytes"]:
                    raise ValueError("ZIP member declared size differs from inventory")
        target.mkdir(parents=True, exist_ok=False)
        for info in infos:
            destination = target.joinpath(*PurePosixPath(info.filename).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(info) as source, destination.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
            if info.filename in manifest["inventory"]:
                if digest.hexdigest() != manifest["inventory"][info.filename]["sha256"]:
                    raise ValueError(f"Extracted member hash mismatch: {info.filename}")
    return verify_inventory(target, manifest, allow_fixture=True, allow_external=True)


def discover_input(input_root, expected_manifest_sha):
    """Require exactly one matching archive or Kaggle-expanded equivalent."""
    candidates = []
    input_root = Path(input_root)
    for path in input_root.rglob("asset_manifest.json"):
        plain_file(input_root, path.relative_to(input_root).as_posix())
        if sha256_file(path) == expected_manifest_sha:
            candidates.append(("expanded", path.parent))
    archives = list(input_root.rglob("eaglenet_eval_input.bin"))
    archives.extend(input_root.rglob("eaglenet_eval_input.zip"))
    for path in archives:
        plain_file(input_root, path.relative_to(input_root).as_posix())
        try:
            with zipfile.ZipFile(path) as archive:
                payload = archive.read("asset_manifest.json")
            if hashlib.sha256(payload).hexdigest() == expected_manifest_sha:
                candidates.append(("zip", path))
        except (zipfile.BadZipFile, KeyError):
            continue
    if len(candidates) != 1:
        raise ValueError(f"Require exactly one pinned private input; found {len(candidates)}")
    return candidates[0]


def validate_offline_requirements(payload):
    """Allow only exact local ancillary dependency pins, never the CUDA/core stack."""
    lines = [line.strip() for line in payload.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    blocked = {"torch", "torchvision", "torchaudio", "numpy", "pandas", "pillow",
               "opencv-python", "opencv-python-headless"}
    if not lines:
        raise ValueError("Offline requirements are empty")
    for line in lines:
        if line.count("==") != 1:
            raise ValueError("Offline requirements must use exact package==version pins")
        name, version = line.split("==")
        if not name or not version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in name) or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.+!-" for c in version):
            raise ValueError("Unsafe offline dependency requirement")
        if name.lower().replace("_", "-") in blocked:
            raise ValueError("Do not replace Kaggle Torch/CUDA/core dependencies automatically")
    return lines


def check_checkpoint(path, variant):
    """Inspect plain tensor state without executing pickled user objects or a model."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state.items()):
        raise ValueError("Require a nonempty plain tensor state_dict")
    required = {"clip.visual.conv1.weight", "clip.token_embedding.weight", "clip.text_projection"}
    if not required <= set(state) or not any(name.startswith("gnn.") for name in state):
        raise ValueError("The checkpoint lacks EagleNet CLIP/graph tensors; no placeholder accepted")
    patch = 16 if variant == "ViT-B/16" else 32
    if tuple(state["clip.visual.conv1.weight"].shape) != (768, 3, patch, patch):
        raise ValueError("Checkpoint patch shape differs from its declared variant")
    return {"plain_tensor_count": len(state), "weights_only": True, "fake_tensor_inspection": True}


def bind_external_checkpoint(bundle_root, input_root, target, original_manifest_sha):
    """Bind one operator-supplied author file; record a first-observed SHA, not an author checksum."""
    bundle_root, input_root, target = map(Path, (bundle_root, input_root, target))
    original = plain_file(bundle_root, "asset_manifest.json")
    if sha256_file(original) != original_manifest_sha:
        raise ValueError("Original data/source manifest changed before checkpoint binding")
    manifest = json.loads(original.read_text(encoding="utf-8"))
    verify_inventory(bundle_root, manifest, allow_external=True)
    if manifest["fixture"] or manifest.get("checkpoint") is not None:
        raise ValueError("External binding requires an explicitly pending, non-fixture bundle")
    request = manifest["checkpoint_request"]
    candidates = []
    for path in input_root.rglob(request["filename"]):
        plain_file(input_root, path.relative_to(input_root).as_posix())
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError("needs_checkpoint: download the documented author file "
                                + request["filename"] + " and add it as a separate private Kaggle input")
    if len(candidates) != 1:
        raise ValueError("Require exactly one external checkpoint with the documented filename")
    checkpoint = candidates[0]
    if checkpoint.stat().st_size != request["bytes"]:
        raise ValueError("External checkpoint size differs from the author directory listing")
    audit = check_checkpoint(checkpoint, manifest["variant"])
    identity = {"path": request["path"], "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint)}
    if target.exists():
        raise FileExistsError("Checkpoint binding requires a new mutable bundle directory")
    target.mkdir(parents=True, exist_ok=False)
    for name in manifest["inventory"]:
        source = plain_file(bundle_root, name)
        destination = target.joinpath(*PurePosixPath(name).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    destination = target.joinpath(*PurePosixPath(request["path"]).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, destination)
    manifest["checkpoint"] = identity
    manifest["inventory"][identity["path"]] = {key: identity[key] for key in ("bytes", "sha256")}
    binding = {"schema": "eaglenet_external_checkpoint_binding_v1",
               "source_manifest_sha256": original_manifest_sha,
               "supplied_filename": request["filename"], "checkpoint": identity,
               "first_observed_sha256": identity["sha256"],
               "author_published_checksum": None,
               "source_assertion": request["provenance"], "source_urls": request["source_urls"],
               "checkpoint_inspection": audit}
    binding_payload = json_bytes(binding)
    binding_path = "checkpoint_binding.json"
    (target / binding_path).write_bytes(binding_payload)
    manifest["inventory"][binding_path] = {"bytes": len(binding_payload),
                                          "sha256": hashlib.sha256(binding_payload).hexdigest()}
    payload = json_bytes(manifest)
    (target / "asset_manifest.json").write_bytes(payload)
    verify_inventory(target, manifest)
    return manifest, {**binding, "bound_manifest_sha256": hashlib.sha256(payload).hexdigest()}


def _cell(source, kind="code"):
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
            "id": hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def make_notebook(manifest_sha):
    helpers = "\n\n".join(inspect.getsource(function) for function in (
        json_bytes, sha256_file, safe_name, plain_file, validate_manifest, verify_inventory,
        safe_extract, discover_input, validate_offline_requirements, check_checkpoint,
        bind_external_checkpoint))
    setup = textwrap.dedent('''\
        import os, sys, json, hashlib, stat, zipfile, traceback, subprocess, time, importlib.util, shutil
        from pathlib import Path, PurePosixPath
        from datetime import datetime, timezone
        # Set this before any torch import, including imports in subprocesses.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        STATE = {"schema": "eaglenet_notebook_run_v1", "failure": None, "stages": [],
                 "benchmark_inference_completed": False}
        WORK = Path("/kaggle/working") / ("eaglenet_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid()))
        WORK.mkdir(parents=True, exist_ok=False)
        BUNDLE = None
        def fail(message):
            STATE["failure"] = str(message)
            print("STOP:", message, flush=True)
            (WORK / "notebook_status.json").write_text(json.dumps(STATE, indent=2), encoding="utf-8")
        print("Starting pinned EagleNet inference handoff:", WORK, flush=True)
    ''')
    setup += f"\nSCHEMA={SCHEMA!r}\nSOURCE_COMMIT={SOURCE_COMMIT!r}\nMAX_FILES={MAX_FILES}\nMAX_BYTES={MAX_BYTES}\nPINNED_MANIFEST_SHA={manifest_sha!r}\n"
    setup += helpers
    setup += textwrap.dedent('''

        try:
            kind, source = discover_input("/kaggle/input", PINNED_MANIFEST_SHA)
            print("Checking private input:", kind, source, flush=True)
            if kind == "zip":
                BUNDLE = WORK / "bundle"
                MANIFEST = safe_extract(source, BUNDLE, PINNED_MANIFEST_SHA)
            else:
                BUNDLE = source
                MANIFEST = json.loads((BUNDLE / "asset_manifest.json").read_text(encoding="utf-8"))
                verify_inventory(BUNDLE, MANIFEST, allow_fixture=True, allow_external=True)
            if MANIFEST["fixture"]:
                raise ValueError("This is a test fixture; benchmark execution is forbidden")
            STATE["asset_manifest_sha256"] = PINNED_MANIFEST_SHA
            STATE["source_manifest_sha256"] = PINNED_MANIFEST_SHA
            STATE["needs_external_checkpoint"] = MANIFEST.get("checkpoint") is None
            print("All input hashes and file coverage passed; variant:", MANIFEST["variant"], flush=True)
        except Exception:
            fail(traceback.format_exc())
    ''')
    deps = textwrap.dedent('''\
        if STATE["failure"] is None:
            try:
                core = ["torch", "numpy", "cv2", "pandas"]
                missing_core = [name for name in core if importlib.util.find_spec(name) is None]
                if missing_core:
                    raise RuntimeError("Kaggle core packages missing; do not replace CUDA/Torch automatically: " + repr(missing_core))
                optional = ["ftfy", "regex", "tqdm", "boto3", "requests"]
                missing = [name for name in optional if importlib.util.find_spec(name) is None]
                if missing:
                    wheelhouse = BUNDLE / "wheelhouse"
                    requirements = wheelhouse / "requirements-offline.txt"
                    if not requirements.is_file():
                        raise RuntimeError("Missing offline dependency wheels for: " + repr(missing))
                    validate_offline_requirements(requirements.read_text())
                    deps = WORK / "deps"
                    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--disable-pip-version-check", "--find-links", str(wheelhouse), "--target", str(deps), "-r", str(requirements)], check=True)
                    sys.path.insert(0, str(deps))
                    os.environ["PYTHONPATH"] = str(deps) + os.pathsep + os.environ.get("PYTHONPATH", "")
                    importlib.invalidate_caches()
                print("Dependency availability checked; no network or Torch reinstall.", flush=True)
            except Exception:
                fail(traceback.format_exc())

        def run_stage(stage, device, probe_videos=None):
            if STATE["failure"] is not None:
                print("Skipping", stage, "because an earlier stage failed; export will retain the evidence.", flush=True)
                return
            label = stage + (str(probe_videos) if probe_videos else "")
            completed = {item["label"] for item in STATE["stages"]}
            prerequisites = {"preflight": set(), "probe8": {"preflight"},
                             "probe64": {"preflight", "probe8"},
                             "evaluate": {"preflight", "probe8", "probe64"}}
            if label not in prerequisites or not prerequisites[label] <= completed:
                fail("Run the pinned preflight → probe8 → probe64 → evaluate sequence in order")
                return
            output = WORK / label
            command = [sys.executable, "-u", str(BUNDLE / "runtime/eaglenet_kaggle_runtime.py"),
                       "--bundle-root", str(BUNDLE), "--output-dir", str(output),
                       "--stage", stage, "--device", device, "--threads", "2"]
            if probe_videos is not None:
                command += ["--probe-videos", str(probe_videos)]
            print("Starting", label, "on", device, flush=True)
            try:
                with (WORK / (label + ".log")).open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, encoding="utf-8", errors="replace", env=os.environ.copy())
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                if code:
                    raise RuntimeError(label + " failed, exit code " + str(code))
                receipt = json.loads((output / "status.json").read_text(encoding="utf-8"))
                expected = {"preflight": "preflight_passed", "probe": "probe_completed", "evaluate": "evaluation_completed"}[stage]
                if receipt.get("schema") != "eaglenet_runtime_status_v1" or receipt.get("status") != expected or receipt.get("fixture") is not False:
                    raise RuntimeError("Runtime status is incomplete or a fixture: " + repr(receipt))
                if receipt.get("asset_manifest_sha256") != STATE["asset_manifest_sha256"]:
                    raise RuntimeError("Runtime receipt is bound to a different input manifest")
                inventory = json.loads((output / "artifacts.json").read_text(encoding="utf-8"))
                if inventory.get("schema") != "eaglenet_output_inventory_v1" or inventory.get("fixture") is not False:
                    raise RuntimeError("Runtime artifact inventory is invalid")
                expected_files = set(inventory["files"]) | {"artifacts.json"}
                if {path.name for path in output.iterdir()} != expected_files:
                    raise RuntimeError("Runtime output file coverage differs from inventory")
                required = {"status.json", "preflight.json", "config.json"}
                if stage != "preflight":
                    required |= {"scores.npy", "feature_cache.npz", "ids.json", "metrics_internal.json",
                                 "metrics_upstream.json", "decode_audit.json", "checkpoint_audit.json",
                                 "cost.json", "environment.json"}
                if stage == "probe":
                    required.add("probe_parity.json")
                if not required <= expected_files:
                    raise RuntimeError("Required runtime evidence is missing: " + repr(sorted(required - expected_files)))
                for name, identity in inventory["files"].items():
                    artifact = plain_file(output, name)
                    if artifact.stat().st_size != identity["bytes"] or sha256_file(artifact) != identity["sha256"]:
                        raise RuntimeError("Runtime artifact hash mismatch: " + name)
                if stage == "evaluate":
                    if (receipt.get("benchmark_inference_completed") is not True or receipt.get("model_strict_loaded") is not True
                            or receipt.get("all_scores_finite") is not True or receipt.get("query_count") != 1000
                            or receipt.get("candidate_count") != 1000 or receipt.get("score_shape") != [1000, 1000]):
                        raise RuntimeError("Full benchmark completion/counts/strict load were not verified")
                elif receipt.get("benchmark_inference_completed") is not False:
                    raise RuntimeError("A technical probe cannot claim benchmark completion")
                if stage == "probe":
                    parity = json.loads((output / "probe_parity.json").read_text(encoding="utf-8"))
                    if parity.get("score_bitwise_equal") is not True or receipt.get("query_count") != probe_videos or receipt.get("candidate_count") != probe_videos:
                        raise RuntimeError("Train-probe replay or pool coverage was not verified")
                STATE["stages"].append({"stage": stage, "label": label, "status": receipt["status"]})
                if stage == "evaluate":
                    STATE["benchmark_inference_completed"] = True
                print(label, "completed.", flush=True)
            except Exception:
                fail(traceback.format_exc())
    ''')
    bind = textwrap.dedent('''\
        if STATE["failure"] is None and STATE["needs_external_checkpoint"]:
            print("Binding a separate operator-supplied author checkpoint; no author checksum is assumed.", flush=True)
            try:
                BOUND = WORK / "bound_bundle"
                MANIFEST, BINDING = bind_external_checkpoint(BUNDLE, Path("/kaggle/input"), BOUND, PINNED_MANIFEST_SHA)
                BUNDLE = BOUND
                STATE["asset_manifest_sha256"] = BINDING["bound_manifest_sha256"]
                STATE["needs_external_checkpoint"] = False
                STATE["checkpoint_first_observed_sha256"] = BINDING["first_observed_sha256"]
                (WORK / "checkpoint_binding.json").write_bytes(json_bytes(BINDING))
                print("Checkpoint bound; strict full model load remains mandatory in the probe.", flush=True)
            except FileNotFoundError as exc:
                STATE["blocked_reason"] = "needs_checkpoint"
                fail(str(exc))
            except Exception:
                fail(traceback.format_exc())
        elif STATE["failure"] is None:
            print("Checkpoint already pinned in the private bundle.", flush=True)
    ''')
    export = textwrap.dedent('''\
        # This cell still runs after guarded failures. Never present partial output as completion.
        STATE["status"] = "completed" if STATE["failure"] is None and STATE["benchmark_inference_completed"] else ("blocked" if STATE.get("blocked_reason") else "partial")
        if STATE["status"] == "completed" and STATE.get("checkpoint_first_observed_sha256"):
            binding = json.loads((WORK / "checkpoint_binding.json").read_text(encoding="utf-8"))
            if binding["source_manifest_sha256"] != PINNED_MANIFEST_SHA or binding["bound_manifest_sha256"] != STATE["asset_manifest_sha256"] or sha256_file(BUNDLE / "asset_manifest.json") != STATE["asset_manifest_sha256"]:
                STATE.update(status="partial", failure="Checkpoint binding changed before export", benchmark_inference_completed=False)
        (WORK / "notebook_status.json").write_text(json.dumps(STATE, indent=2), encoding="utf-8")
        result = WORK.parent / (WORK.name + "_" + STATE["status"] + ".zip")
        exported = {}
        with zipfile.ZipFile(result, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(WORK.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(WORK)
                if relative.parts[0] in {"bundle", "bound_bundle", "deps"}:
                    continue
                archive.write(path, relative.as_posix())
                exported[relative.as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            archive.writestr("export_inventory.json", json.dumps(exported, indent=2))
        print("Result:", STATE["status"], "—", result, flush=True)
        print("Completed benchmark:", STATE["benchmark_inference_completed"], flush=True)
        try:
            from IPython.display import FileLink, display
            display(FileLink(str(result)))
        except ImportError:
            pass
    ''')
    intro = ("# EagleNet — inference checkpoint trên MSR-VTT test1k\n\n"
             "Add Input: private `eaglenet_eval_input.bin` đi kèm (nội dung ZIP nguyên byte, "
             "đuôi opaque bảo vệ tokenizer gzip khỏi tự giải nén); chọn **GPU T4 ×2**, "
             "chương trình chỉ dùng GPU 0. **Save Version → Run All** chạy kiểm đầu vào, "
             "probe 8/64 video train rồi đánh giá đủ test1k khi các bước trước đạt. "
             "Không training, không chọn checkpoint/cấu hình bằng test. Internet không cần bật.\n\n"
             "Notebook ghim hash gói; lỗi sẽ chặn bước phụ thuộc và vẫn xuất ZIP partial/log. "
             "Chỉ kết quả `completed` đã kiểm đủ 1.000 query/candidate mới là benchmark hoàn tất. "
             "Giữ frame/prompt/seed và toàn bộ candidate pool theo manifest.\n")
    intro += ("\nNếu gói còn thiếu weights: tải **msrvtt9k-vit_b_16.bin** từ liên kết tác giả "
              "trong hướng dẫn và Add Input riêng tư với đúng tên. Không có file thì notebook "
              "chỉ kiểm code/data và xuất trạng thái **blocked / needs_checkpoint**, không chạy mô hình. "
              "SHA ghi sau khi tiếp nhận là dấu vân tay quan sát lần đầu, không phải checksum tác giả công bố.\n")
    cells = [_cell(intro, "markdown"), _cell(setup), _cell(deps), _cell(bind),
             _cell('run_stage("preflight", "cpu")\n'),
             _cell('run_stage("probe", "cuda", 8)\n'),
             _cell('run_stage("probe", "cuda", 64)\n'),
             _cell('run_stage("evaluate", "cuda")\n'), _cell(export)]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    payload = json_bytes(notebook)
    if len(payload) >= 1_000_000:
        raise ValueError("Notebook exceeds the Kaggle 1 MB source limit")
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "eaglenet_notebook", "exec")
    return payload


def prepare(asset_dir, runtime_script, output_dir, *, allow_fixture=False, external_checkpoint=False):
    """Validate all real inputs, then create a new ZIP/notebook without overwriting evidence."""
    asset_dir, runtime_script, output_dir = map(Path, (asset_dir, runtime_script, output_dir))
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing evidence is preserved")
    manifest_path = plain_file(asset_dir, "asset_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_inventory(asset_dir, manifest, allow_fixture=allow_fixture,
                     allow_external=external_checkpoint)
    requirements = asset_dir / "wheelhouse/requirements-offline.txt"
    if requirements.is_file():
        validate_offline_requirements(requirements.read_text(encoding="utf-8"))
    if not runtime_script.is_file() or runtime_script.is_symlink():
        raise ValueError("The reviewed runtime script is required")
    compile(runtime_script.read_text(encoding="utf-8"), str(runtime_script), "exec")
    audit = {"fixture": manifest["fixture"], "checked": False}
    if not manifest["fixture"] and manifest.get("checkpoint") is not None:
        audit = check_checkpoint(plain_file(asset_dir, manifest["checkpoint"]["path"]),
                                 manifest["variant"])
    files = {name: plain_file(asset_dir, name) for name in manifest["inventory"]}
    runtime_name = "runtime/eaglenet_kaggle_runtime.py"
    runtime_payload = runtime_script.read_bytes()
    if runtime_name in files and files[runtime_name].read_bytes() != runtime_payload:
        raise ValueError("An existing runtime payload differs; update the asset manifest explicitly")
    files[runtime_name] = runtime_script
    manifest["inventory"][runtime_name] = {"bytes": len(runtime_payload),
                                          "sha256": hashlib.sha256(runtime_payload).hexdigest()}
    manifest_payload = json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    notebook = make_notebook(manifest_sha)
    output_dir.mkdir(parents=True, exist_ok=False)
    archive_path = output_dir / "eaglenet_eval_input.bin"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("asset_manifest.json", manifest_payload)
        for name, path in sorted(files.items()):
            archive.write(path, name)
    notebook_path = output_dir / "eaglenet_eval_kaggle.ipynb"
    notebook_path.write_bytes(notebook)
    pending = manifest.get("checkpoint") is None
    receipt = {"schema": "eaglenet_package_receipt_v1", "fixture": manifest["fixture"],
               "status": "fixture_only" if manifest["fixture"] else (
                   "needs_external_checkpoint" if pending else "ready_for_preflight"),
               "benchmark_inference_completed": False, "source_commit": SOURCE_COMMIT,
               "variant": manifest["variant"], "asset_manifest_sha256": manifest_sha,
               "inventory_files": len(manifest["inventory"]), "checkpoint_inspection": audit,
               "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                         for path in (archive_path, notebook_path)}}
    (output_dir / "package_manifest.json").write_bytes(json_bytes(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--runtime-script", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--external-checkpoint", action="store_true",
                        help="Explicit code/data handoff; checkpoint must be supplied separately")
    args = parser.parse_args()
    print(json.dumps(prepare(args.asset_dir, args.runtime_script, args.output_dir,
                             external_checkpoint=args.external_checkpoint), indent=2))


if __name__ == "__main__":
    main()

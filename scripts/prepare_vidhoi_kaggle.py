"""Build an allowlisted, self-contained Kaggle input and train-only oracle notebook.

All data, checkpoint and optional dependency wheels must already exist locally.
This command never downloads or uploads files and never executes model inference.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path

from temporal_hoi.inference.hoi_kaggle import (
    BUNDLE_SCHEMA,
    MANIFEST_NAME,
    MAX_BUNDLE_BYTES,
    safe_member,
    sha256_file,
    validate_manifest,
    validate_requirements,
    wheel_distribution,
)

SOURCE_FILES = (
    "src/temporal_hoi/__init__.py", "src/temporal_hoi/inference/__init__.py",
    "src/temporal_hoi/data/vidhoi.py", "src/temporal_hoi/evaluation/hoi.py",
    "src/temporal_hoi/inference/hoi_static.py", "src/temporal_hoi/inference/hoi_kaggle.py",
    "scripts/run_vidhoi_static.py",
)
GENERATED_NAMESPACES = ("src/temporal_hoi/data/__init__.py",
                        "src/temporal_hoi/evaluation/__init__.py")


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _record(value: bytes | Path) -> dict:
    return {"bytes": len(value) if isinstance(value, bytes) else value.stat().st_size,
            "sha256": hashlib.sha256(value).hexdigest() if isinstance(value, bytes)
            else sha256_file(value)}


def _plain_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked {label}: {path.name}")
    return path


def _notebook(runtime_source: str, manifest_sha256: str) -> dict:
    cells = []

    def code(source):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {}, "source": source,
                      "execution_count": None, "outputs": []})

    code(f'''
        print("CELL_1_STARTED — VidHOI train-only oracle probe", flush=True)
        import json, os, sys, uuid
        from datetime import datetime, timezone
        from pathlib import Path
        # Bootstrap source is embedded in this reviewed notebook, before Dataset imports.
        exec(compile({runtime_source!r}, "embedded_vidhoi_bootstrap.py", "exec"))
        PINNED_MANIFEST = "{manifest_sha256}"
        WORK = Path(os.environ.get("VIDHOI_WORK_ROOT", "/kaggle/working"))
        INPUTS = Path(os.environ.get("VIDHOI_INPUT_ROOT", "/kaggle/input"))
        BASE = WORK / "vidhoi_static_probe"
        BASE.mkdir(parents=True, exist_ok=True)
        ROOT = BASE / ("verified_input_" + PINNED_MANIFEST[:12])
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        RUN = BASE / ("oracle_train_" + run_id)
        REC = RunRecorder(RUN, BASE)
        REC.event("started", split="train", box_source="oracle", object_label_source="oracle",
                  training_run=False, full_benchmark_evaluated=False,
                  python=sys.version, input_root=str(INPUTS))
        try:
            receipt = install_input(INPUTS, ROOT, PINNED_MANIFEST, progress=REC.event)
            REC.write_json("input_receipt.json", receipt)
        except BaseException as error:
            REC.event("input_failed", error=str(error))
            REC.snapshot()
            raise
        REC.cwd = ROOT
        REC.env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        print("INPUT_VERIFIED. Small train probe; no training, final-val or M2 retrieval.", flush=True)
    ''')
    code('''
        print("CELL_2_STARTED — dependencies; preserve Kaggle Torch/CUDA", flush=True)
        # Complete input ZIP has offline wheels. Set True only for missing Python dependencies.
        ALLOW_DEPENDENCY_INTERNET = False
        try:
            verify_installed_input(ROOT, PINNED_MANIFEST)
            DEPS = BASE / ("python_deps_" + PINNED_MANIFEST[:12])
            install_command = dependency_plan(ROOT, target=DEPS,
                                              allow_internet=ALLOW_DEPENDENCY_INTERNET)
            if install_command:
                REC.command("dependencies", install_command, timeout_seconds=600)
            else:
                REC.event("dependencies_already_installed")
            sys.path.insert(0, str(DEPS))
            REC.env["PYTHONPATH"] = str(DEPS) + os.pathsep + str(ROOT / "src")
        except BaseException as error:
            REC.event("dependencies_failed", error=str(error))
            REC.snapshot()
            raise
    ''')
    code('''
        print("CELL_3_STARTED — environment and fixed probe config", flush=True)
        # Auto is explicit: GPU when enabled, otherwise this small probe runs on CPU.
        DEVICE_REQUEST = "auto"  # Set "cuda" to require GPU, or "cpu" explicitly.
        try:
            verify_installed_input(ROOT, PINNED_MANIFEST)
            environment = environment_report()
            if DEVICE_REQUEST not in {"auto", "cpu", "cuda"}:
                raise ValueError("DEVICE_REQUEST must be auto/cpu/cuda")
            if DEVICE_REQUEST == "cuda" and not environment["cuda_available"]:
                raise RuntimeError("GPU was required but CUDA is unavailable; enable a Kaggle GPU")
            DEVICE = ("cuda" if environment["cuda_available"] else "cpu") if DEVICE_REQUEST == "auto" else DEVICE_REQUEST
            environment["selected_device"] = DEVICE
            REC.write_json("environment.json", environment)
            print(json.dumps(environment, indent=2), flush=True)
            CONFIG = prepare_run_config(ROOT, RUN / "run_config.json", device=DEVICE)
        except BaseException as error:
            REC.event("environment_failed", error=str(error))
            REC.snapshot()
            raise
    ''')
    code('''
        print("CELL_4_STARTED — local preflight; no model inference", flush=True)
        verify_installed_input(ROOT, PINNED_MANIFEST)
        REC.command("oracle_preflight", [sys.executable, "-u", str(ROOT / "scripts/run_vidhoi_static.py"),
                    "--config", str(CONFIG), "--output-dir", str(RUN / "preflight"), "--preflight-only"])
    ''')
    code('''
        print("CELL_5_STARTED — CLIP static inference on train oracle pairs", flush=True)
        verify_installed_input(ROOT, PINNED_MANIFEST)
        REC.command("oracle_train_inference", [sys.executable, "-u", str(ROOT / "scripts/run_vidhoi_static.py"),
                    "--config", str(CONFIG), "--output-dir", str(RUN / "oracle_probe")])
    ''')
    code('''
        print("CELL_6_STARTED — save and download results", flush=True)
        try:
            probe_receipt = json.loads((RUN / "oracle_probe/receipt.json").read_text(encoding="utf-8"))
            probe_input = json.loads((ROOT / "assets/probe_manifest.json").read_text(encoding="utf-8"))
            if (probe_receipt.get("status") != "completed"
                    or probe_receipt.get("split") != "train"
                    or probe_receipt.get("box_source") != "oracle"
                    or probe_receipt.get("model_training_run") is not False
                    or probe_receipt.get("num_frames") != len(probe_input["frames"])):
                raise RuntimeError("Inference receipt is incomplete or does not match this train probe")
        except BaseException as error:
            REC.event("export_incomplete", error=str(error))
            REC.snapshot()
            raise
        REC.event("completed", split="train", box_source="oracle", object_label_source="oracle",
                  training_run=False, full_benchmark_evaluated=False,
                  model_inference_run=probe_receipt.get("model_inference_run"),
                  level=probe_receipt.get("level"),
                  limits="Small train probe only; no predicted detector, zero-shot split, temporal module or M2 gain.")
        RESULT_ZIP = REC.snapshot()
        from IPython.display import FileLink, display
        display(FileLink(str(RESULT_ZIP)))
        print("Download this ZIP and keep all receipts/logs, including negative results.", flush=True)
    ''')
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"isInternetEnabled": False, "language": "python"},
    }, "nbformat": 4, "nbformat_minor": 5}


def prepare(probe_manifest: str | Path, mapping: str | Path, clip_checkpoint: str | Path,
            clip_sha256: str, output: str | Path, *, checkpoint_provenance: str | Path,
            wheelhouse: str | Path | None = None, data_provenance=(), model_license=None,
            project_root: str | Path | None = None, internal_bundle: dict | None = None) -> dict:
    """Build data/code/weights ZIP and six-cell notebook; preserve existing outputs."""
    extra_sources = ()
    entrypoint = "scripts/run_vidhoi_static.py"
    bundle_name = "vidhoi_oracle_train_input.zip"
    notebook_name = "vidhoi_oracle_train_kaggle.ipynb"
    notebook_factory = _notebook
    if internal_bundle is not None:
        expected = {
            "extra_sources": ("src/temporal_hoi/inference/hoi_diagnostics.py",
                              "scripts/run_vidhoi_diagnostics.py"),
            "entrypoint": "scripts/run_vidhoi_diagnostics.py",
            "bundle_name": "vidhoi_m1_diagnostics_input.zip",
            "notebook_name": "vidhoi_m1_diagnostics_kaggle.ipynb",
        }
        if (not isinstance(internal_bundle, dict)
                or set(internal_bundle) != set(expected) | {"notebook_factory"}
                or any(internal_bundle[key] != value for key, value in expected.items())
                or not callable(internal_bundle["notebook_factory"])):
            raise ValueError("Unsupported internal bundle profile; use the diagnostic wrapper")
        extra_sources = expected["extra_sources"]
        entrypoint = expected["entrypoint"]
        bundle_name, notebook_name = expected["bundle_name"], expected["notebook_name"]
        notebook_factory = internal_bundle["notebook_factory"]
    project = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    probe_manifest, mapping = Path(probe_manifest), Path(mapping)
    clip_checkpoint, output = Path(clip_checkpoint), Path(output)
    for path, label in ((probe_manifest, "probe manifest"), (mapping, "class mapping"),
                        (clip_checkpoint, "CLIP checkpoint"),
                        (Path(checkpoint_provenance), "checkpoint provenance")):
        _plain_file(path, label)
    if output.exists():
        raise FileExistsError("Choose a new --output directory; existing evidence is preserved")
    if sha256_file(clip_checkpoint) != clip_sha256:
        raise ValueError("CLIP checkpoint SHA256 mismatch")
    if clip_checkpoint.suffix.lower() not in {".safetensors", ".pt", ".bin"}:
        raise ValueError("Use original CLIP plain tensor state_dict (.safetensors preferred)")
    if zipfile.is_zipfile(clip_checkpoint):
        with zipfile.ZipFile(clip_checkpoint) as archive:
            if any("/code/" in name or name.endswith("/constants.pkl")
                   for name in archive.namelist()):
                raise ValueError("TorchScript checkpoint unsupported; convert verified weights locally")
    probe = json.loads(probe_manifest.read_text(encoding="utf-8"))
    if (probe.get("schema_version") != "vidhoi_static_oracle_probe_v1"
            or probe.get("split") != "train" or probe.get("purpose") != "train_probe"
            or probe.get("box_source") != "oracle" or probe.get("object_label_source") != "oracle"):
        raise ValueError("Require a small train-only oracle/object-label probe manifest")
    if probe.get("mapping_sha256") != sha256_file(mapping):
        raise ValueError("Probe mapping SHA256 mismatch")
    if not isinstance(probe.get("frames"), list) or not 1 <= len(probe["frames"]) <= 64:
        raise ValueError("Require 1–64 actual train frames for this initial probe")
    provenance = json.loads(Path(checkpoint_provenance).read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Require nonempty checkpoint provenance assertions")
    if provenance.get("checkpoint_sha256", clip_sha256) != clip_sha256:
        raise ValueError("Checkpoint provenance SHA256 mismatch")
    files: dict[str, bytes | Path] = {}
    for name in SOURCE_FILES + extra_sources:
        files[name] = _plain_file(project / name, "required probe source")
    for name in GENERATED_NAMESPACES:
        files[name] = b'"""Minimal namespace for the portable static oracle train probe."""\n'
    root = probe_manifest.parent.resolve()
    frame_remap = []
    for index, frame in enumerate(probe["frames"]):
        relative = safe_member(frame["image_path"])
        image = root / relative
        if not image.resolve().is_relative_to(root):
            raise ValueError("Probe image path escapes its manifest folder")
        _plain_file(image, "probe image")
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Probe frames must be raster images, not raw videos or arbitrary files")
        if sha256_file(image) != frame.get("image_sha256"):
            raise ValueError(f"Probe image SHA256 mismatch at frame {index}")
        name = f"assets/frames/{index:04d}{image.suffix.lower()}"
        files[name] = image
        frame["image_path"] = name.removeprefix("assets/")
        frame_remap.append({"packaged_path": name, "image_sha256": frame["image_sha256"]})
    files["assets/probe_manifest.json"] = _json_bytes(probe)
    files["assets/class_mapping.json"] = mapping
    checkpoint_name = "clip_original" + clip_checkpoint.suffix.lower()
    files[f"assets/{checkpoint_name}"] = clip_checkpoint
    files["provenance/checkpoint.json"] = _json_bytes(provenance)
    files["provenance/source_probe_manifest.json"] = probe_manifest
    for index, source in enumerate(data_provenance):
        source = _plain_file(Path(source), "data provenance")
        if source.suffix.lower() not in {".json", ".txt", ".md"} or source.stat().st_size > 8*1024**2:
            raise ValueError("Data provenance must be a small public-source JSON/text/Markdown receipt")
        safe_member(source.name)
        files[f"provenance/data/{index:02d}_{source.name}"] = source
    if model_license is not None:
        license_path = _plain_file(Path(model_license), "model license")
        if license_path.stat().st_size > 1024**2:
            raise ValueError("Model license file is too large")
        files["provenance/CLIP_LICENSE.txt"] = license_path
    wheels = []
    if wheelhouse is not None:
        wheelhouse = Path(wheelhouse)
        if not wheelhouse.is_dir() or wheelhouse.is_symlink():
            raise ValueError("Wheelhouse must be an ordinary local directory")
        for wheel in sorted(wheelhouse.glob("*.whl")):
            _plain_file(wheel, "dependency wheel")
            wheel_distribution(wheel.name)
            files[f"wheelhouse/{wheel.name}"] = wheel
            wheels.append(wheel.name)
        if not wheels:
            raise ValueError("The requested wheelhouse contains no dependency wheels")
        requirements = _plain_file(wheelhouse / "requirements-offline.txt",
                                   "offline dependency requirements")
        validate_requirements(requirements.read_text(encoding="utf-8"))
        files["wheelhouse/requirements-offline.txt"] = requirements
    config = {"schema_version": "vidhoi_static_config_v1",
              "manifest": "probe_manifest.json",
              "manifest_sha256": _record(files["assets/probe_manifest.json"])["sha256"],
              "mapping": "class_mapping.json", "mapping_sha256": sha256_file(mapping),
              "checkpoint": checkpoint_name, "checkpoint_sha256": clip_sha256,
              "checkpoint_provenance": provenance, "device": "cpu", "batch_size": 8,
              "max_frames": 64, "max_pairs_per_frame": 128, "threads": 2,
              "preview_frames": min(4, len(probe["frames"])), "logit_scale": 1.0}
    files["assets/run_config.json"] = _json_bytes(config)
    scope = ("# VidHOI static oracle train probe\n\n"
             "Small train-only CLIP inference. Both boxes and object labels are ground truth.\n"
             "Not predicted detection, training, zero-shot evaluation, full VidHOI benchmark, "
             "temporal modeling or M2 retrieval improvement.\n\n"
             "Only explicitly selected frames, labels, public-source receipts, local model "
             "weights and allowlisted inference files are packaged. No private project rules, "
             "progress records, personal reports, raw dataset archive or credentials.\n"
             "Data/model rights remain subject to their source conditions; local hashes do "
             "not establish ownership or authorize public redistribution.\n"
             "Namespace-only data/evaluation __init__.py files are generated deliberately "
             "to avoid importing unrelated project modules. Other source files retain exact hashes.\n")
    files["BUNDLE_SCOPE.md"] = scope.encode("utf-8")
    if internal_bundle is not None:
        files["BUNDLE_SCOPE.md"] += (
            "\nRepresentation diagnostic: fixed full_frame, object_only and union views. "
            "Identical frames, oracle pairs, CLIP weights, prompts and evaluator. "
            "Known-positive ranks are supplementary diagnostics, not accuracy; missing "
            "labels remain unknown. No model selection or training on this viewed cohort.\n"
        ).encode("utf-8")
    records = {name: _record(value) for name, value in sorted(files.items())}
    manifest = {"schema_version": BUNDLE_SCHEMA, "dataset": "VidHOI", "split": "train",
                "purpose": "train_probe", "box_source": "oracle", "object_label_source": "oracle",
                "benchmark_evaluated": False, "training_run": False,
                "entrypoint": entrypoint, "frames": len(probe["frames"]),
                "files": records, "total_bytes": sum(row["bytes"] for row in records.values()),
                "generated_namespace_files": list(GENERATED_NAMESPACES),
                "source_probe_manifest_sha256": sha256_file(probe_manifest),
                "frame_remap": frame_remap, "bundled_wheels": wheels}
    manifest_bytes = _json_bytes(manifest)
    validate_manifest(manifest_bytes)
    if manifest["total_bytes"] > MAX_BUNDLE_BYTES:
        raise ValueError("Probe input exceeds the 2 GiB portable bundle limit")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    runtime_source = (project / "src/temporal_hoi/inference/hoi_kaggle.py").read_text(encoding="utf-8")
    notebook = notebook_factory(runtime_source, digest)
    output.mkdir(parents=True, exist_ok=False)
    bundle_path = output / bundle_name
    with zipfile.ZipFile(bundle_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, manifest_bytes)
        for name, value in sorted(files.items()):
            compression = zipfile.ZIP_STORED if name.startswith(("assets/clip_original", "wheelhouse/")) \
                else zipfile.ZIP_DEFLATED
            if isinstance(value, bytes):
                archive.writestr(name, value, compress_type=compression)
            else:
                archive.write(value, name, compress_type=compression)
    notebook_path = output / notebook_name
    notebook_path.write_bytes(_json_bytes(notebook))
    instructions = f"""# Chạy probe M1 trên Kaggle

1. **Import Notebook**: chọn `{notebook_path.name}`.
2. Tạo hoặc cập nhật **Dataset riêng tư** tên `VidHOI M1 Oracle Train Probe Input`
   chứa `{bundle_path.name}` của cùng thư mục với notebook này. Đợi xử lý hoàn tất.
   Mở notebook ở chế độ **Edit → Add Input → Your Datasets**, chọn Dataset và thêm vào.
   Trong mục Input phải thấy ZIP hoặc các file đã giải nén; chỉ Import notebook chưa gắn data.
3. Chọn GPU nếu muốn chạy nhanh, giữ notebook riêng tư, rồi **Run All**. Có 6 cell.
4. Cuối cell 6, tải ZIP kết quả có tên `oracle_train_<thời gian>_<mã>.zip`.
   Nếu một cell lỗi, giữ ZIP/log đã xuất; không sửa nhãn hoặc bỏ frame để chạy tiếp.

Gói đã có {len(probe['frames'])} frame train, nhãn/hộp GT, code và checkpoint CLIP local.
{'Đã kèm wheelhouse: Internet không cần để cài dependency.' if wheels else 'Chưa kèm wheelhouse. Nếu Kaggle thiếu dependency, cell 2 dừng rõ; có thể bật Internet và đặt ALLOW_DEPENDENCY_INTERNET=True chỉ để cài thư viện.'}
Torch, torchvision, NumPy và Pillow của Kaggle được giữ nguyên; dependency được
cài vào thư mục riêng trong working, dùng `--no-deps` và requirements đã khóa.
Không tải model/video tự động. ZIP có thể được đổi tên hoặc Kaggle giải nén;
notebook nhận bằng manifest/SHA256, không dựa vào tên Dataset.
Nếu báo `Visible files: []`, kiểm lại bước Add Input. Nếu báo không khớp manifest,
dùng đúng cặp notebook và ZIP của cùng phiên bản; không sửa hash để bỏ xác minh.
Tên hiển thị notebook đề xuất: `VidHOI M1 Static CLIP Oracle Probe`.

Đây là **probe train nhỏ, oracle boxes và oracle object labels**, chỉ inference.
Metric trên mẫu này không phải benchmark VidHOI đầy đủ, không xác nhận zero-shot,
khả năng detector predicted, học temporal, hoặc cải thiện retrieval M2.
Mỗi lần Run All dùng thư mục kết quả mới. Không đưa gói lên chế độ public nếu
chưa xác minh quyền chia sẻ media/weights; bản thân hash không cấp quyền đó.

Manifest SHA256: `{digest}`.
"""
    (output / "HUONG_DAN.md").write_text(instructions, encoding="utf-8")
    receipt = {"schema_version": "vidhoi_kaggle_build_v1", "status": "local_bundle_prepared",
               "bundle": bundle_path.name, "bundle_sha256": sha256_file(bundle_path),
               "bundle_bytes": bundle_path.stat().st_size, "manifest_sha256": digest,
               "notebook": notebook_path.name, "notebook_sha256": sha256_file(notebook_path),
               "notebook_code_cells": len(notebook["cells"]), "frames": len(probe["frames"]),
               "bundled_wheels": wheels, "files": records, "checkpoint_sha256": clip_sha256,
               "model_inference_run": False, "training_run": False, "benchmark_evaluated": False,
               "uploaded": False, "checkpoint_provenance_independently_audited": False,
               "limits": ["Input preparation is not evidence of a Kaggle run",
                          "Small train oracle probe, not full benchmark or zero-shot result"]}
    (output / "build_receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-sha256", required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--data-provenance", type=Path, action="append", default=[])
    parser.add_argument("--model-license", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.probe_manifest, args.mapping, args.clip_checkpoint, args.clip_sha256,
                      args.output, checkpoint_provenance=args.checkpoint_provenance,
                      wheelhouse=args.wheelhouse, data_provenance=args.data_provenance,
                      model_license=args.model_license)
    print(json.dumps({key: receipt[key] for key in ("status", "bundle", "bundle_bytes",
                     "frames", "notebook_code_cells", "model_inference_run", "uploaded")}))


if __name__ == "__main__":
    main()

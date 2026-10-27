"""Package three-view oracle VidHOI diagnostics without downloading or uploading."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import textwrap
from pathlib import Path

from temporal_hoi.inference.hoi_kaggle import safe_member, sha256_file

VARIANTS = ("full_frame", "object_only", "union")


def validate_diagnostic_export(run: Path, input_root: Path, *, allow_fixture: bool = False) -> dict:
    """Require all three outputs and hashes; fixture acceptance is a test API only."""
    import hashlib

    run, input_root = Path(run), Path(input_root)
    output = run / "diagnostics"
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    source = json.loads((input_root / "assets/probe_manifest.json").read_text(encoding="utf-8"))
    expected_variants = ["full_frame", "object_only", "union"]
    expected_pairs = sum(sum(entity["class_id"] == 0 for entity in frame["entities"])
                         * max(0, len(frame["entities"]) - 1) for frame in source["frames"])
    is_fixture = receipt.get("level") == "representation_diagnostic_encoder_fixture"
    if is_fixture and (not allow_fixture or receipt.get("encoder", {}).get("test_fixture") is not True):
        raise RuntimeError("A fixture receipt cannot complete a real diagnostic notebook run")
    expected_level = "representation_diagnostic_encoder_fixture" if is_fixture else \
        "representation_diagnostic_train_oracle"
    if (receipt.get("schema_version") != "vidhoi_representation_diagnostic_receipt_v1"
            or receipt.get("status") != "completed" or receipt.get("level") != expected_level
            or receipt.get("split") != "train" or receipt.get("box_source") != "oracle"
            or receipt.get("object_label_source") != "oracle"
            or receipt.get("variants") != expected_variants
            or receipt.get("num_frames") != len(source["frames"])
            or receipt.get("num_proposal_pairs") != expected_pairs
            or receipt.get("dropped_frames") != 0
            or receipt.get("dropped_proposal_pairs") != 0
            or receipt.get("official_final_split_accessed") is not False
            or receipt.get("zero_shot_benchmark_claim") is not False
            or receipt.get("model_training_run") is not False
            or receipt.get("model_inference_run") is not (not is_fixture)
            or receipt.get("benchmark_comparison_eligible") is not False
            or receipt.get("full_benchmark_reproduction_claim") is not False):
        raise RuntimeError("Incomplete or mismatched three-view diagnostic receipt")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Diagnostic receipt needs a nonempty output file inventory")
    inventory = {}
    for record in files:
        if not isinstance(record, dict):
            raise RuntimeError("Invalid diagnostic output inventory record")
        name = record.get("path")
        safe_member(name)
        if name == "receipt.json" or name in inventory:
            raise RuntimeError("Duplicate or self-referential diagnostic output record")
        path = output / name
        if (path.is_symlink() or not path.resolve().is_relative_to(output.resolve())
                or not path.is_file() or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")):
            raise RuntimeError(f"Diagnostic output hash/size mismatch: {name}")
        inventory[name] = record
    actual = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Diagnostic output symlinks are forbidden")
        if path.is_file():
            actual.add(path.relative_to(output).as_posix())
    if actual != set(inventory) | {"receipt.json"}:
        raise RuntimeError("Diagnostic output inventory differs from actual files")
    required = {"run_config.json", "prompts.json", "comparison.json", "role_symmetry.json"}
    required |= {f"{variant}/{name}" for variant in expected_variants
                 for name in ("predictions.json", "metrics.json", "known_positive_ranks.json", "receipt.json")}
    if not required <= inventory.keys():
        raise RuntimeError("Missing required diagnostic outputs")
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    packaged_config = json.loads((input_root / "assets/run_config.json").read_text(encoding="utf-8"))
    input_hashes = receipt.get("input_hashes", {})
    for name in ("manifest", "mapping", "checkpoint"):
        safe_member(packaged_config[name])
        asset = input_root / "assets" / packaged_config[name]
        if asset.is_symlink() or not asset.resolve().is_relative_to(input_root.resolve()):
            raise RuntimeError(f"Unsafe packaged diagnostic {name}")
        expected_hash = sha256_file(asset)
        if (packaged_config.get(f"{name}_sha256") != expected_hash
                or config.get(f"{name}_sha256") != expected_hash
                or input_hashes.get(name) != expected_hash
                or Path(config[name]).resolve() != asset.resolve()):
            raise RuntimeError(f"Diagnostic input identity mismatch: {name}")
    frame_ids = [f"{frame['video_id']}_{frame['frame_id']:06d}" for frame in source["frames"]]
    frame_hash = hashlib.sha256(json.dumps(frame_ids, separators=(",", ":")).encode()).hexdigest()
    if receipt.get("frame_ids_sha256") != frame_hash:
        raise RuntimeError("Diagnostic frame identity/order mismatch")
    stored_prompts = json.loads((output / "prompts.json").read_text(encoding="utf-8"))
    if (not isinstance(stored_prompts, dict)
            or any(str(int(key)) != key for key in stored_prompts)):
        raise RuntimeError("Invalid diagnostic prompt class IDs")
    prompts = {int(key): value for key, value in stored_prompts.items()}
    prompt_hash = hashlib.sha256(json.dumps(prompts, sort_keys=True,
                                            ensure_ascii=False).encode()).hexdigest()
    if (receipt.get("prompt_sha256") != prompt_hash
            or receipt.get("prompt_version") != "vidhoi_all_predicates_english_v1"):
        raise RuntimeError("Diagnostic prompt identity mismatch")
    summaries = receipt.get("variant_summaries")
    if not isinstance(summaries, dict) or set(summaries) != set(expected_variants):
        raise RuntimeError("Missing three-view variant summaries")
    for variant in expected_variants:
        details = json.loads((output / variant / "receipt.json").read_text(encoding="utf-8"))
        summary = summaries[variant]
        if (details.get("schema_version") != "vidhoi_representation_variant_receipt_v1"
                or details.get("status") != "completed"
                or details.get("variant") != variant
                or details.get("level") != expected_level
                or details.get("prompt_sha256") != prompt_hash
                or details.get("model_inference_run") is not (not is_fixture)
                or details.get("model_training_run") is not False
                or details.get("dropped_frames") != 0
                or details.get("dropped_proposal_pairs") != 0
                or summary.get("dropped_frames") != 0
                or summary.get("dropped_proposal_pairs") != 0
                or details.get("num_frames") != len(source["frames"])
                or details.get("num_proposal_pairs") != expected_pairs
                or summary.get("num_frames") != len(source["frames"])
                or summary.get("num_proposal_pairs") != expected_pairs):
            raise RuntimeError(f"Incomplete diagnostic variant: {variant}")
    return receipt


def _base_module():
    path = Path(__file__).with_name("prepare_vidhoi_kaggle.py")
    spec = importlib.util.spec_from_file_location("_vidhoi_base_packager", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Base notebook hook changed; review diagnostic bootstrap before packaging")
    return source.replace(old, new, 1)


def notebook(runtime_source: str, manifest_sha256: str) -> dict:
    """Reuse authenticated input/offline setup, with explicit diagnostic run/export."""
    result = _base_module()._notebook(runtime_source, manifest_sha256)
    first = result["cells"][0]["source"]
    first = _replace_once(first, 'BASE = WORK / "vidhoi_static_probe"',
                          'BASE = WORK / "vidhoi_m1_diagnostics"')
    first = _replace_once(first, 'RUN = BASE / ("oracle_train_" + run_id)',
                          'RUN = BASE / ("diagnostics_train_" + run_id)')
    first = _replace_once(first, "CELL_1_STARTED — VidHOI train-only oracle probe",
                          "CELL_1_STARTED — VidHOI three-view train oracle diagnostics")
    result["cells"][0]["source"] = first
    cells = result["cells"][:3]

    def add(source):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {}, "source": source,
                      "execution_count": None, "outputs": []})

    add('''
        print("CELL_4_STARTED — diagnostic input preflight", flush=True)
        try:
            verify_installed_input(ROOT, PINNED_MANIFEST)
            REC.command("diagnostic_preflight", [sys.executable, "-u",
                        str(ROOT / "scripts/run_vidhoi_diagnostics.py"), "--config", str(CONFIG),
                        "--output-dir", str(RUN / "preflight"), "--preflight-only"])
        except BaseException as error:
            REC.event("diagnostic_preflight_failed", error=str(error))
            REC.snapshot()
            raise
    ''')
    add('''
        print("CELL_5_STARTED — full_frame, object_only, union", flush=True)
        try:
            verify_installed_input(ROOT, PINNED_MANIFEST)
            REC.command("three_view_diagnostic", [sys.executable, "-u",
                        str(ROOT / "scripts/run_vidhoi_diagnostics.py"), "--config", str(CONFIG),
                        "--output-dir", str(RUN / "diagnostics")])
        except BaseException as error:
            REC.event("diagnostic_run_failed", error=str(error))
            REC.snapshot()
            raise
    ''')
    validator = inspect.getsource(validate_diagnostic_export)
    final = '''
print("CELL_6_STARTED — verify all three variants and download", flush=True)
try:
    diagnostic_receipt = validate_diagnostic_export(RUN, ROOT)
except BaseException as error:
    REC.event("export_incomplete", error=str(error))
    REC.snapshot()
    raise
REC.event("completed", level=diagnostic_receipt["level"], split="train",
          variants=diagnostic_receipt["variants"],
          model_inference_run=diagnostic_receipt["model_inference_run"], training_run=False,
          full_benchmark_evaluated=False,
          limits="Three-view train oracle diagnostic only; no predicted detector, temporal model or M2 gain.")
RESULT_ZIP = REC.snapshot()
from IPython.display import FileLink, display
print("Download from this link or the Kaggle Output files panel:", flush=True)
display(FileLink(str(RESULT_ZIP)))
'''
    add(validator + "\n\n" + final)
    result["cells"] = cells
    if len(cells) != 6:
        raise ValueError("Diagnostic notebook must have exactly six cells")
    return result


def prepare(probe_manifest, mapping, clip_checkpoint, clip_sha256, output, *,
            checkpoint_provenance, wheelhouse=None, data_provenance=(), model_license=None,
            project_root=None) -> dict:
    """Reuse the reviewed allowlist builder; add only the two diagnostic sources."""
    output = Path(output)
    hook = {
        "extra_sources": ("src/temporal_hoi/inference/hoi_diagnostics.py",
                          "scripts/run_vidhoi_diagnostics.py"),
        "entrypoint": "scripts/run_vidhoi_diagnostics.py", "notebook_factory": notebook,
        "bundle_name": "vidhoi_m1_diagnostics_input.zip",
        "notebook_name": "vidhoi_m1_diagnostics_kaggle.ipynb",
    }
    receipt = _base_module().prepare(
        probe_manifest, mapping, clip_checkpoint, clip_sha256, output,
        checkpoint_provenance=checkpoint_provenance, wheelhouse=wheelhouse,
        data_provenance=data_provenance, model_license=model_license, project_root=project_root,
        internal_bundle=hook)
    receipt["experiment"] = "vidhoi_representation_diagnostic_v1"
    receipt["variants"] = list(VARIANTS)
    (output / "build_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    guide = f'''# Chạy chẩn đoán M1 trên Kaggle

1. Import Notebook, đặt tên **VidHOI M1 Representation Diagnostics**: chọn `{receipt["notebook"]}`.
2. Add Input: tạo Dataset riêng tư **VidHOI M1 Representation Diagnostics Input**
   chứa `{receipt["bundle"]}`, đợi xử lý xong rồi gắn vào notebook.
3. Giữ notebook riêng tư, Internet Off, chọn GPU nếu có; **Save Version → Save & Run All**.
4. Cuối cell 6, tải ZIP `diagnostics_train_<thời gian>_<mã>.zip` từ link hoặc Output.

Gói chứa {receipt["frames"]} frame train, hộp/lớp vật GT, checkpoint CLIP gốc,
source và dependency offline. Không gắn input của notebook M1 cũ.
Tên ZIP có thể đổi; notebook nhận theo manifest/SHA256.

Tự động chạy đủ ba cách biểu diễn, cùng dữ liệu/cặp/prompt/model/evaluator:
- **full_frame**: toàn ảnh;
- **object_only**: vùng vật thể;
- **union**: vùng bao người và vật.

Hạng predicate chỉ chấm nhãn positive đã biết; nhãn chưa gán không tự là negative. Object-only có thể dựa vào hình dáng vật mà không phân biệt người tương tác. Ba view có số ảnh encode khác nhau; xem receipt chi phí thực.

Đây là chẩn đoán trên mẫu train nhỏ với hộp và lớp vật chuẩn. Không train,
không detector predicted, không kết luận zero-shot, không benchmark VidHOI
đầy đủ hoặc cải thiện M2. Không chọn cách biểu diễn thắng trên vài frame rồi
coi là mô hình cuối. Giữ cả kết quả âm và logs.

Notebook có 6 cell, không có câu hỏi tương tác. Mỗi lượt tạo thư mục kết quả
mới; chỉ đánh dấu hoàn tất khi đủ cả ba variant và mọi hash/count khớp.
Nếu lỗi, tải ZIP/log đã lưu. `Visible files: []` hoặc thông báo không thấy
input nghĩa là cần kiểm tra Add Input/tiến trình xử lý Dataset, không đổi tên model.
Torch/torchvision/NumPy/Pillow của Kaggle được giữ nguyên; dependency riêng
được cài offline vào working; wheelhouse hỗ trợ CPython 3.11, 3.12 và 3.13 trên Linux x86_64. Không tự tải weights/video và không publish.

Manifest SHA256: `{receipt["manifest_sha256"]}`.
'''
    (output / "HUONG_DAN.md").write_text(guide, encoding="utf-8")
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
                     "frames", "variants", "model_inference_run", "uploaded")}))


if __name__ == "__main__":
    main()

"""Build a local, allowlisted Kaggle handoff; never upload or execute a GPU job."""

import argparse
import ast
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path, PurePosixPath

from kaggle_notebook_cells import revise

DATA_FILES = {
    "configs/msrvtt_clip4clip_inference.json",
    "data/metadata/msrvtt_protocol_lock.json",
    "src/temporal_hoi/evaluation/reference/clip4clip_metrics.py.txt",
    "src/temporal_hoi/evaluation/reference/LICENSE.txt",
    "data/metadata/msrvtt_data.zip",
    "outputs/experiments/pilot/baseline/m2_sources/internvideo_dataset_utils.py",
    "outputs/experiments/pilot/baseline/m2_sources/internvideo_ret_dataset.py",
}


def allowed(name):
    """Allow only public inference code and locked metadata in the input bundle."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return False
    if name in DATA_FILES:
        return True
    if name.startswith("src/temporal_hoi/") and path.suffix == ".py":
        return True
    if name in {"scripts/run_msrvtt_baseline.py", "scripts/evaluate_msrvtt.py"}:
        return True
    return (name.startswith("external/CLIP4Clip/")
            and (path.suffix == ".py" or path.name in
                 {"LICENSE", "bpe_simple_vocab_16e6.txt.gz"}))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(context, output):
    project = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    files = {p.relative_to(context).as_posix(): p for p in context.rglob("*")
             if p.is_file() and allowed(p.relative_to(context).as_posix())}
    for name in files:
        if (project / name).is_file() and not name.startswith("external/"):
            files[name] = project / name
    for name in ("container_baseline_assets.py", "install_native_checkpoint.py",
                 "probe_packaged_model.py"):
        files[f"scripts/{name}"] = project / "scripts" / name
    if not DATA_FILES <= files.keys():
        raise ValueError("Missing locked metadata or evaluator snapshots")
    manifest = {name: sha(path) for name, path in sorted(files.items())}
    bundle = output / "baseline_kaggle_input.zip"
    with zipfile.ZipFile(bundle, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    cells = []

    def md(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

    def code(source):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {}, "source": source,
                      "execution_count": None, "outputs": []})

    md("""# M2 — CLIP4Clip native / MSR-VTT 1k trên Kaggle
Import notebook này; tạo **private Dataset** chỉ chứa `baseline_kaggle_input.zip`,
rồi Add Input vào notebook **private**. Bật GPU và Internet trong Session options.
Chạy lần lượt các cell. Không upload toàn repository hoặc Docker image.
Gói chỉ gồm code, nguồn công khai và metadata đã khóa; không có tài liệu cá nhân.

Đây là **inference checkpoint bên thứ ba**, không huấn luyện, không weights chính thức tác giả.
Epoch 4 cố định trước score. Train9k có thể đã dùng test trong quá khứ: chưa audit được.
Hai clip development chỉ đo tài nguyên vì chúng thuộc train9k của checkpoint.
M1 VidHOI chưa đủ input và không nằm trong notebook này.
Không sửa checkpoint/prompt/preprocessing theo kết quả test. Chưa xác nhận tái lập metric bài báo.
Hướng dẫn giao diện: https://www.kaggle.com/docs/notebooks""")
    code(f'''
        import os, sys, json, subprocess, zipfile, hashlib, shutil
        from pathlib import Path, PurePosixPath
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        WORK = Path("/kaggle/working")
        ROOT = WORK / "baseline"
        matches = list(Path("/kaggle/input").rglob("baseline_kaggle_input.zip"))
        assert len(matches) == 1, "Add đúng một Dataset chứa baseline_kaggle_input.zip"
        assert hashlib.sha256(matches[0].read_bytes()).hexdigest() == "{sha(bundle)}"
        assert not ROOT.exists(), "Thư mục đã có: tiếp tục cell sau hoặc tạo session mới"
        with zipfile.ZipFile(matches[0]) as z:
            names = z.namelist()
            assert len(names) == len(set(names))
            assert sum(i.file_size for i in z.infolist()) < 32 * 1024**2
            manifest = json.loads(z.read("bundle_manifest.json"))
            assert set(names) == set(manifest) | {{"bundle_manifest.json"}}
            for name in names:
                p = PurePosixPath(name)
                assert not p.is_absolute() and ".." not in p.parts and "\\\\" not in name
                data = z.read(name)
                if name in manifest:
                    assert hashlib.sha256(data).hexdigest() == manifest[name]
                dest = ROOT / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        os.chdir(ROOT)
        subprocess.run(["nvidia-smi"], check=True)
        assert shutil.disk_usage(WORK).free > 8 * 1024**3, "Cần ít nhất 8 GiB trống"
    ''')
    md("## Môi trường riêng Python 3.12\nCell cài môi trường CUDA riêng, không thay PyTorch của kernel. Lần đầu cần tải thư viện; nếu thất bại hãy giữ log lỗi.")
    code('''
        BOOT = WORK / "baseline_bootstrap"
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(BOOT),
                        "uv==0.8.22"], check=True)
        uv_env = dict(os.environ, PYTHONPATH=str(BOOT))
        def uv(*args):
            subprocess.run([sys.executable, "-m", "uv", *args], env=uv_env, check=True)
        uv("python", "install", "3.12")
        ENV = WORK / "baseline_env"
        uv("venv", "--python", "3.12", str(ENV))
        PY = str(ENV / "bin/python")
        uv("pip", "install", "--python", PY, "torch==2.6.0", "torchvision==0.21.0",
           "--index-url", "https://download.pytorch.org/whl/cu124")
        uv("pip", "install", "--python", PY, "numpy==1.26.4", "pillow==11.3.0",
           "opencv-python-headless==4.10.0.84", "ftfy==6.3.1", "regex==2024.11.6",
           "tqdm==4.67.1", "psutil==6.1.1", "pandas==2.2.3")
        RUN_ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="0")
        RESULTS = WORK / "baseline_results"
        RESULTS.mkdir(exist_ok=True)
        def run(*args):
            subprocess.run([PY, *map(str, args)], cwd=ROOT, env=RUN_ENV, check=True)
        def inline(source):
            run("-c", source)
        inline("import torch; assert torch.cuda.is_available(); "
               "assert torch.cuda.device_count()==1; print(torch.__version__, "
               "torch.cuda.get_device_name()); print(torch.ones(2,device='cuda')+1)")
        with (RESULTS / "environment.txt").open("w") as f:
            subprocess.run([PY, "-m", "pip", "freeze"], stdout=f, check=True)
    ''')
    # uv venv does not seed pip: use uv pip freeze rather than python -m pip.
    cells[-1]["source"] = cells[-1]["source"].replace(
        '[PY, "-m", "pip", "freeze"], stdout=f, check=True',
        '[sys.executable, "-m", "uv", "pip", "freeze", "--python", PY], '
        'env=uv_env, stdout=f, check=True')
    md("## Tải checkpoint và video công khai\nHash checkpoint được kiểm trước load; video lấy đúng test1k + hai clip probe qua HTTP Range, kiểm ETag/CRC/hash. Nếu nguồn đổi hoặc mạng chặn, cell dừng; không thay dữ liệu. Cache nằm trong working để thử lại trong cùng session.")
    code('''
inline("""
import sys, json
from pathlib import Path
sys.path.insert(0, 'scripts')
from install_native_checkpoint import (
    fetch_checked, CHECKPOINT_URL, CHECKPOINT_BYTES, CHECKPOINT_SHA256,
    ROOT_URL, LOG_BYTES, LOG_SHA256,
)
from container_baseline_assets import prepare_inference_media
from temporal_hoi.data.msrvtt import MSRVTTAdapter
from temporal_hoi.inference.clip4clip import verify_source
verify_source('external/CLIP4Clip')
weights = Path('weights'); weights.mkdir(exist_ok=True)
checkpoint = fetch_checked(CHECKPOINT_URL, weights/'epoch4.bin', CHECKPOINT_BYTES, CHECKPOINT_SHA256)
fetch_checked(ROOT_URL+'/ckpts/ckpt_msrvtt_retrieval_looseType/log.txt',
              weights/'training-source.log', LOG_BYTES, LOG_SHA256)
provenance = dict(source_url=CHECKPOINT_URL, checkpoint_sha256=CHECKPOINT_SHA256,
    architecture='CLIP4Clip_ViT-B32_meanP_loose_2d', official_author_checkpoint=False,
    training_regime='Third-party claimed train9k, 5 epochs, seed42; not independently audited',
    selection_rule='Fixed final epoch4 before local scores',
    limitations=['Historical test use unknown', 'Internal validation overlaps claimed train9k',
                 'Paper reproduction unverified'])
(weights/'provenance.json').write_text(json.dumps(provenance, indent=2))
adapter = MSRVTTAdapter.from_files('data/metadata/msrvtt_data.zip',
                                 'data/metadata/msrvtt_protocol_lock.json')
target = Path('data/raw/msrvtt'); target.mkdir(parents=True, exist_ok=True)
receipt = prepare_inference_media(target, adapter, cache=Path('media_cache'))
Path('/kaggle/working/baseline_results/media_receipt.json').write_text(json.dumps(receipt, indent=2))
config = json.loads(Path('configs/msrvtt_clip4clip_inference.json').read_text())
config.update(checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA256,
              checkpoint_provenance=str(weights/'provenance.json'),
              selection_fixed_before_scores=True)
Path('configs/kaggle_run.json').write_text(json.dumps(config, indent=2))
""")
CONFIG = ROOT / "configs/kaggle_run.json"
shutil.copy2(CONFIG, RESULTS / "config.json")
shutil.copy2(ROOT / "weights/provenance.json", RESULTS / "checkpoint_provenance.json")
shutil.copy2(ROOT / "data/raw/msrvtt/media_provenance.json", RESULTS / "media_provenance.json")
shutil.copy2(ROOT / "bundle_manifest.json", RESULTS / "bundle_manifest.json")
    ''')
    md("## Probe GPU hai clip — chưa chấm test\nStrict-load model và kiểm decode/encoder, RAM/VRAM/thời gian. Ước lượng 1k từ hai clip không phải thời gian benchmark đo thực. Cell dừng nếu probe lỗi.")
    code('''
        run("scripts/probe_packaged_model.py", "--config", CONFIG,
            "--output", RESULTS / "resource_probe.json")
        probe = json.loads((RESULTS / "resource_probe.json").read_text())
        print(json.dumps(probe, indent=2))
        assert probe["status"] == "native_model_resource_probe_verified"
        assert probe["estimated_1k_wall_seconds_with_factor2"] < 3600, "Ước lượng vượt budget 1 giờ; chưa chạy test"
    ''')
    md("## Final test 1.000 video\nChạy một lần khi probe đạt. Cấu hình giữ FP32/12 frame/32 token/frame batch4 và giới hạn 8 GiB RSS/1 giờ. Không giảm pool để lấy kết quả đẹp. Chạy lại cần thư mục kết quả mới; không ghi đè kết quả trước.")
    code('''
        assert probe["status"] == "native_model_resource_probe_verified"
        FINAL = RESULTS / "final_test"
        run("scripts/run_msrvtt_baseline.py", "--config", CONFIG,
            "--execute", "--allow-final-evaluation", "--output-dir", FINAL)
        result = json.loads((FINAL / "run.json").read_text())
        assert result["status"] == "completed_checkpoint_inference"
        print((FINAL / "metrics.json").read_text())
        run("scripts/evaluate_msrvtt.py", "--scores", FINAL / "scores.npz",
            "--provenance", FINAL / "provenance.json", "--checkpoint", ROOT / "weights/epoch4.bin",
            "--split", "test", "--output", RESULTS / "reference_evaluation.json")
    ''')
    md("## Lưu kết quả\nCó thể chạy cell này cả khi cell trước lỗi để lấy bằng chứng. Tải ZIP và giữ phiên bản notebook/output trước khi kết thúc session. ZIP chỉ gồm kết quả, không gói video/thư viện. Gửi ZIP kết quả về laptop để đối chiếu; không tuyên bố baseline hoàn tất nếu run.json báo failed.")
    code('''
        archive = shutil.make_archive(str(WORK / "baseline_results"), "zip", RESULTS)
        print(archive)
        from IPython.display import FileLink, display
        display(FileLink(archive))
    ''')
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    notebook = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"baseline-{i:02d}"
    notebook = revise(notebook, project, bundle)
    (output / "msrvtt_baseline_kaggle.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "handoff_manifest.json").write_text(json.dumps({
        "bundle_sha256": sha(bundle), "files": manifest, "uploaded": False,
        "gpu_execution_verified": False}, indent=2), encoding="utf-8")
    return {"output": str(output), "bundle_bytes": bundle.stat().st_size,
            "files": len(files), "code_cells": sum(c["cell_type"] == "code" for c in cells)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.context, args.output)))

"""Descriptive notebook cells with robust Kaggle inputs and streamed version logs."""

import ast
import base64
import hashlib
import json
import textwrap
import zipfile


def revise(notebook, project, bundle):
    cells = notebook["cells"]
    markdown = [
        """# CLIP4Clip meanP — MSR-VTT retrieval

Notebook đánh giá checkpoint ViT-B/32, 2D, loose meanP trên tập JSFusion test1k
đã khóa. Đây là inference checkpoint epoch4 bên thứ ba, không phải huấn luyện
hoặc weights do tác giả phát hành. Nguồn khai báo train9k/seed42/5 epoch;
lịch sử chọn model bằng test chưa kiểm chứng. Hai clip development chỉ đo tài nguyên.

Kiến trúc, BPE, RGB/bicubic short-side224/center224, lấy mẫu 1Hz rồi uniform12,
32 token và normalize→mean→normalize được đối chiếu với
[mã tác giả tại commit khóa](https://github.com/ArrowLuo/CLIP4Clip/tree/508ffa3de39ba0563a03199c440ab602a72e9b6f)
và [bài CLIP4Clip](https://arxiv.org/abs/2104.08860).
Adapter chạy FP32/Python3.12/PyTorch2.6; chưa có đối chiếu số học end-to-end
với môi trường gốc. Decode lỗi dừng thay vì tạo frame giả hoặc bỏ video.

## 1. Nhận diện và xác minh input

Cell nhận ZIP theo manifest/hash, độc lập tên file hoặc tên Dataset, và nhận cả
thư mục đã được Kaggle giải nén. Gói cũ ngày19/10 và gói mới đều được nhận diện.
Notebook nhỏ, không nhúng archive lớn vào cell code. Khi Kaggle giải nén tiếp
BPE/metadata, byte gốc được tải từ URL tác giả đã khóa và kiểm SHA256/kích thước.
Cell này cần Internet nếu archive gốc thiếu; tải có log tiến độ và timeout.
Mã thực thi hiện hành được ghi cùng hash vào output. Nguồn input chỉ chứa
code/metadata công khai. Phần này ghi log khởi tạo, dung lượng trống và trạng thái GPU.""",
        """## 2. Môi trường CUDA độc lập

Cell tạo Python3.12 riêng và cài PyTorch2.6/torchvision0.21 CUDA12.4 cùng các
dependency cố định. Kernel notebook không đổi thư viện. GPU0 là thiết bị duy nhất
được tiến trình con nhìn thấy. Phiên bản Python, driver, thư viện và phép tính CUDA
nhỏ được ghi vào output. Mỗi tiến trình có log trực tiếp và heartbeat30giây khi im lặng.""",
        """## 3. Checkpoint và media benchmark

Checkpoint epoch4 được kiểm SHA256/kích thước trước khi load tensor an toàn.
Training log công khai được giữ làm provenance; nó không chứng minh quá trình
huấn luyện đã được audit độc lập. Video được chọn đúng test1k và hai clip probe,
kiểm HTTP Range/ETag/CRC/hash. Cache tải xuống cho phép tiếp tục trong cùng session.
Mỗi nguồn sai hash hoặc clip thiếu đều dừng với lỗi cụ thể, không thay thế dữ liệu.
Trọng số, config, metadata và mã nguồn có provenance riêng.""",
        """## 4. Probe model trên GPU

Hai clip development dùng để kiểm strict-load, decode, shape encoder, RAM/VRAM
và thời gian. Cell không tính metric chất lượng validation hoặc score test.
Ước lượng thời gian cho test1k từ hai clip có hệ số dự trù2 và không phải phép đo
toàn benchmark. Budget hiện tại là 8GiB RSS/1giờ inference; probe vượt budget dừng sớm.""",
        """## 5. Inference và đối chiếu evaluator

Checkpoint và preprocessing được cố định trước khi truy cập score test.
Text/video encode được ghi tiến độ, ID video, cache hit, thời gian và bộ nhớ.
Feature lưu atomically sau từng đầu vào và chỉ dùng lại khi input/model/code/runtime
identity khớp. Mỗi lần thực thi có output mới, bảo toàn lượt trước; cache không chứa
metric để chọn cấu hình. Khả năng tiếp tục chỉ áp dụng khi working/cache còn tồn tại.

Score có trục[text,video], là cosine nhân exp(logit_scale). Kết quả gồm hai chiều
R@1/5/10, MedR và MRR phụ theo protocol. Evaluator nguồn được chạy trên cùng score;
khác biệt tie-breaking được giữ và báo riêng. FP32 và checkpoint bên thứ ba vẫn
là giới hạn đối với tuyên bố tái lập paper.""",
        """## 6. Artifact và nhật ký phiên chạy

ZIP kết quả gồm events.jsonl, console.log, môi trường, config, hash mã thực thi,
probe, score/metric/provenance và trạng thái lỗi nếu có. Snapshot được cập nhật
sau từng tiến trình và cả khi tiến trình thất bại. Cache feature nằm trong working
riêng. Notebook không thể bảo đảm lưu kịp nếu Kaggle thu hồi session hoặc hệ điều hành
kill toàn kernel; mạng, quota, driver, input hỏng vẫn có thể làm lượt chạy dừng.
Không có metric hoàn thành nếu final run thất bại.""",
    ]
    for cell, value in zip([c for c in cells if c["cell_type"] == "markdown"], markdown,
                           strict=True):
        cell["source"] = value
    codes = [c for c in cells if c["cell_type"] == "code"]
    trusted = []
    for path in (bundle, project / "outputs/experiments/msrvtt/20261021_clip4clip_baseline/baseline_kaggle_input.zip"):
        if path.is_file():
            with zipfile.ZipFile(path) as archive:
                trusted.append(hashlib.sha256(archive.read("bundle_manifest.json")).hexdigest())
    runtime = (project / "scripts/kaggle_baseline_runtime.py").read_text(encoding="utf-8")
    recovery = {}
    urls = {
        "external/CLIP4Clip/modules/bpe_simple_vocab_16e6.txt.gz":
            "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/"
            "508ffa3de39ba0563a03199c440ab602a72e9b6f/modules/bpe_simple_vocab_16e6.txt.gz",
        "data/metadata/msrvtt_data.zip":
            "https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip",
        "outputs/w03/sources/m2/msrvtt_data.zip":
            "https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip",
    }
    for path in (bundle, project / "outputs/experiments/msrvtt/20261021_clip4clip_baseline/baseline_kaggle_input.zip"):
        if not path.is_file():
            continue
        with zipfile.ZipFile(path) as archive:
            for name, url in urls.items():
                if name not in archive.namelist():
                    continue
                data = archive.read(name)
                spec = {"url": url, "size": len(data),
                        "expected_hash": hashlib.sha256(data).hexdigest()}
                if name in recovery and recovery[name] != spec:
                    raise ValueError(f"Conflicting pinned recovery asset: {name}")
                recovery[name] = spec
    # Embed current adapter scripts so the old input remains usable without another upload.
    overlays = {name: base64.b64encode((project / name).read_bytes()).decode() for name in (
        "scripts/run_msrvtt_baseline.py", "scripts/container_baseline_assets.py")}
    codes[0]["source"] = 'print("CELL_1_STARTED: kernel is executing", flush=True)\n' + runtime + "\n" + textwrap.dedent(f'''
        import os, sys, base64
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTHONUNBUFFERED"] = "1"
        WORK = Path("/kaggle/working")
        ROOT = WORK / "baseline"
        INPUT = WORK / "baseline_verified_input"
        RESULTS = WORK / "baseline_results"
        logger = LoggedCommands(RESULTS, WORK, dict(os.environ))
        logger.event("input_discovery")
        import shutil
        input_finished = threading.Event()
        def input_heartbeat():
            while not input_finished.wait(15):
                logger.event("input_heartbeat")
        threading.Thread(target=input_heartbeat, daemon=True).start()
        try:
            def fetch_asset(spec):
                return lambda: download_recovery(**spec, progress=logger.event)
            recovery_assets = {{name: fetch_asset(spec) for name, spec
                               in {json.dumps(recovery)}.items()}}
            receipt = install_input(Path("/kaggle/input"), INPUT,
                                    {json.dumps(sorted(set(trusted)))}, recovery_assets=recovery_assets,
                                    progress=logger.event)
            ROOT.mkdir(parents=True, exist_ok=True)
            manifest = json.loads((INPUT / "bundle_manifest.json").read_text())
            for name in list(manifest) + ["bundle_manifest.json"]:
                target = ROOT / name
                if not target.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError("Working path escapes root")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(INPUT / name, target)
            # Retain historical archive identity; derive current working input paths.
            aliases = {{
                "outputs/w03/sources/m2/msrvtt_data.zip": "data/metadata/msrvtt_data.zip",
                "outputs/w03/msrvtt_protocol_lock.json": "data/metadata/msrvtt_protocol_lock.json",
            }}
            alias_hashes = {{}}
            for old, new in aliases.items():
                if old in manifest and new not in manifest:
                    target = ROOT / new
                    if not target.resolve().is_relative_to(ROOT.resolve()):
                        raise ValueError("Metadata alias escapes working root")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(INPUT / old, target)
                    alias_hashes[new] = digest_bytes(target.read_bytes())
                    if alias_hashes[new] != manifest[old]:
                        raise ValueError("Derived metadata alias hash mismatch")
            (RESULTS/"input_path_aliases.json").write_text(json.dumps(alias_hashes, indent=2))
            overlays = {json.dumps(overlays)}
            for name, encoded in overlays.items():
                (ROOT / name).write_bytes(base64.b64decode(encoded))
            source_hashes = {{name: digest_bytes((ROOT/name).read_bytes()) for name in manifest}}
            (RESULTS/"executed_source_hashes.json").write_text(json.dumps(source_hashes, indent=2))
            (RESULTS/"input_receipt.json").write_text(json.dumps(receipt, indent=2))
            logger.event("input_verified", **receipt)
            free = shutil.disk_usage(WORK).free
            logger.event("disk", free_bytes=free)
            if free < 8*1024**3:
                raise RuntimeError("Need at least 8 GiB free for installation/media/results")
            logger.run("gpu_inventory", ["nvidia-smi"], timeout=60)
            os.chdir(ROOT)
        except Exception as exc:
            logger.event("input_failed", error=str(exc))
            raise
        finally:
            input_finished.set()
            logger.snapshot()
    ''')
    codes[1]["source"] = textwrap.dedent('''
        BOOT = WORK / "baseline_bootstrap"
        logger.run("bootstrap_uv", [sys.executable, "-m", "pip", "install",
                   "--target", str(BOOT), "uv==0.8.22"], timeout=600)
        uv_env = dict(os.environ, PYTHONPATH=str(BOOT), UV_PYTHON_INSTALL_DIR=str(WORK/"python"),
                      UV_CACHE_DIR=str(WORK/"uv_cache"))
        def uv(*args):
            return logger.run("uv_"+args[0], [sys.executable, "-m", "uv", *args], env=uv_env,
                              timeout=1800)
        uv("python", "install", "3.12")
        ENV = WORK / "baseline_env"
        PY = str(ENV / "bin/python")
        if not Path(PY).exists():
            uv("venv", "--python", "3.12", str(ENV))
        uv("pip", "install", "--python", PY, "torch==2.6.0", "torchvision==0.21.0",
           "--index-url", "https://download.pytorch.org/whl/cu124")
        uv("pip", "install", "--python", PY, "numpy==1.26.4", "pillow==11.3.0",
           "opencv-python-headless==4.10.0.84", "ftfy==6.3.1", "regex==2024.11.6",
           "tqdm==4.67.1", "psutil==6.1.1", "pandas==2.2.3")
        uv("pip", "check", "--python", PY)
        frozen = uv("pip", "freeze", "--python", PY)
        (RESULTS/"environment.txt").write_text(frozen)
        RUN_ENV = dict(os.environ, PYTHONPATH=str(ROOT/"src"), CUDA_VISIBLE_DEVICES="0",
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        logger.cwd = ROOT
        logger.env = RUN_ENV
        def run(*args, label="python", timeout=4500):
            return logger.run(label, [PY, "-u", *map(str, args)], timeout=timeout)
        def inline(source, label="prepare"):
            return run("-c", source, label=label)
        inline("import platform,torch,torchvision; "
               "assert platform.python_version_tuple()[:2]==('3','12'); "
               "assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; "
               "print(platform.python_version(),torch.__version__,torchvision.__version__); "
               "print(torch.cuda.get_device_name(),torch.cuda.mem_get_info()); "
               "print(torch.ones(2,device='cuda')+1)", label="cuda_preflight")
    ''')
    codes[2]["source"] = codes[2]["source"].replace(
        "selection_fixed_before_scores=True)",
        "selection_fixed_before_scores=True, "
        "feature_cache_root='/kaggle/working/baseline_feature_cache')")
    codes[2]["source"] = codes[2]["source"].replace(
        "from pathlib import Path\n", "from pathlib import Path\nprint('Verifying pinned source/checkpoint', flush=True)\n")
    codes[2]["source"] += '\nlogger.snapshot()\n'
    codes[3]["source"] = textwrap.dedent('''
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        probe_path = RESULTS / ("resource_probe_"+stamp+".json")
        run("scripts/probe_packaged_model.py", "--config", CONFIG,
            "--output", probe_path, label="resource_probe", timeout=600)
        probe = json.loads(probe_path.read_text())
        logger.event("probe_summary", **probe)
        if probe["status"] != "native_model_resource_probe_verified":
            raise RuntimeError("Resource probe failed; final test blocked")
        if probe["estimated_1k_wall_seconds_with_factor2"] >= 3600:
            raise RuntimeError("Estimated inference exceeds fixed 1-hour budget")
    ''')
    codes[4]["source"] = textwrap.dedent('''
        if probe["status"] != "native_model_resource_probe_verified":
            raise RuntimeError("Probe required before final evaluation")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        FINAL = RESULTS / ("final_test_"+stamp)
        run("scripts/run_msrvtt_baseline.py", "--config", CONFIG,
            "--execute", "--allow-final-evaluation", "--output-dir", FINAL,
            label="final_inference", timeout=3900)
        result = json.loads((FINAL/"run.json").read_text())
        if result["status"] != "completed_checkpoint_inference":
            raise RuntimeError("Final checkpoint inference incomplete")
        print((FINAL/"metrics.json").read_text(), flush=True)
        run("scripts/evaluate_msrvtt.py", "--scores", FINAL/"scores.npz",
            "--provenance", FINAL/"provenance.json", "--checkpoint", ROOT/"weights/epoch4.bin",
            "--split", "test", "--output", FINAL/"reference_evaluation.json",
            label="reference_evaluator", timeout=300)
    ''')
    codes[5]["source"] = textwrap.dedent('''
        archive = logger.snapshot()
        logger.event("artifact_ready", path=str(archive), bytes=archive.stat().st_size)
        from IPython.display import FileLink, display
        display(FileLink(str(archive)))
    ''')
    for cell in codes:
        tree = ast.parse(cell["source"])
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "inline" and isinstance(node.args[0], ast.Constant)):
                ast.parse(node.args[0].value)
    return notebook

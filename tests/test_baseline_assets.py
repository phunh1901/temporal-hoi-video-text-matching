"""Test native baseline input privacy, media and checkpoint integrity offline."""


import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest


def load_script(name):
    source = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    script_dir = str(source.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(f"test_{name}_module", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["AGENTS.md", "HUONG_DAN_TIEN_DO.md", "rules.md", "rubric.md",
                                  ".env", "reports/test.json", "docs/prd.md", "plan.md",
                                  "src/../../rules.md", "/src/x.py", "src\\x.py"])
def test_private_documents_and_escaped_paths_excluded(name):
    assert not load_script("prepare_kaggle_baseline").allowed(name)


def test_allowed_metadata_and_reference_license():
    module = load_script("prepare_kaggle_baseline")
    for name in ("src/temporal_hoi/data/msrvtt.py", "external/CLIP4Clip/LICENSE",
                 "data/metadata/msrvtt_data.zip"):
        assert module.allowed(name)


@pytest.mark.parametrize("names", [["../video0.mp4"], ["MSRVTT/video0.mp4", "other/video0.mp4"],
                                   ["video1.mp4"], ["video0.mp4", "video1.mp4"]])
def test_archive_traversal_duplicate_missing_or_unexpected_video_rejected(names):
    module = load_script("container_baseline_assets")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, b"not-decoded-fixture")
    with zipfile.ZipFile(buffer) as archive, pytest.raises(ValueError):
        module.media_members(archive, {"video0"})


def test_media_flat_mapping_and_full_coverage():
    module = load_script("container_baseline_assets")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("MSRVTT/videos/all/video0.mp4", b"data")
    with zipfile.ZipFile(buffer) as archive:
        selected, size = module.media_members(archive, {"video0"})
    assert set(selected) == {"video0"} and size == 4


def test_only_reviewed_auxiliary_sidecars_are_excluded():
    module = load_script("container_baseline_assets")
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("MSRVTT/videos/all/video0.mp4", b"video")
        archive.writestr("MSRVTT/videos/all/._video7.mp4", b"x" * 4096)
    with zipfile.ZipFile(source) as archive:
        selected, size = module.media_members(archive, {"video0"})
    assert set(selected) == {"video0"} and size == 5
    changed = io.BytesIO()
    with zipfile.ZipFile(changed, "w") as archive:
        archive.writestr("MSRVTT/videos/all/._video7.mp4", b"changed")
    with zipfile.ZipFile(changed) as archive, pytest.raises(ValueError, match="size changed"):
        module.media_members(archive, {"video0"})


def test_remote_zip_reads_exact_ranges_and_zip_crc(tmp_path, monkeypatch):
    module = load_script("container_baseline_assets")
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("all/video0.mp4", b"video-bytes" * 50)
    content = source.getvalue()
    monkeypatch.setattr(module, "MEDIA_BYTES", len(content))
    ranges = []

    def fetch(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        assert request.get_header("If-match") == module.MEDIA_ETAG
        ranges.append((start, end))
        response = io.BytesIO(content[start:end + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}",
                            "ETag": module.MEDIA_ETAG}
        return response

    monkeypatch.setattr(module.urllib.request, "urlopen", fetch)
    with zipfile.ZipFile(module.RemoteArchive()) as archive:
        assert archive.read("all/video0.mp4") == b"video-bytes" * 50
    assert len(ranges) > 1


def test_remote_zip_rejects_changed_source(monkeypatch):
    module = load_script("container_baseline_assets")
    response = io.BytesIO(b"a")
    response.status = 206
    response.headers = {"Content-Range": f"bytes 0-0/{module.MEDIA_BYTES}", "ETag": '"changed"'}
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="range/ETag"):
        module.RemoteArchive().read(1)


def test_checkpoint_hash_failure_prevents_use(tmp_path, monkeypatch):
    module = load_script("install_native_checkpoint")
    response = io.BytesIO(b"bad")
    response.status = 200
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.fetch_checked("https://example.invalid/model", tmp_path / "model", 3,
                             hashlib.sha256(b"yes").hexdigest())
    assert not (tmp_path / "model").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_failed_model_probe_never_claims_benchmark_or_quality(tmp_path, monkeypatch):
    module = load_script("probe_packaged_model")
    config = tmp_path / "config.json"
    config.write_text("{}")
    output = tmp_path / "native_probe.json"
    monkeypatch.setattr(module, "probe", lambda c: (_ for _ in ()).throw(ValueError("no GPU")))
    monkeypatch.setattr(sys, "argv", ["probe", "--config", str(config), "--output", str(output)])
    assert module.main() == 2
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["benchmark_metrics"] is None
    assert report["benchmark_inference_run"] is False


def test_download_range_and_cache_hash_are_checked(tmp_path, monkeypatch):
    module = load_script("container_baseline_assets")
    monkeypatch.setattr(module, "MEDIA_BYTES", 3)
    monkeypatch.setattr(module, "PART_BYTES", 3)

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-2/3", "ETag": module.MEDIA_ETAG}

    seen = []

    def fetch(request, timeout):
        seen.append(request)
        assert request.get_header("Range") == "bytes=0-2"
        assert request.get_header("If-match") == module.MEDIA_ETAG
        return Response(b"abc")

    monkeypatch.setattr(module.urllib.request, "urlopen", fetch)
    path = module.download_part(0, tmp_path)
    assert path.read_bytes() == b"abc"
    assert module.download_part(0, tmp_path) == path and len(seen) == 1
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Cached media part hash"):
        module.download_part(0, tmp_path)


def test_source_ignoring_range_is_rejected(tmp_path, monkeypatch):
    module = load_script("container_baseline_assets")

    class Response(io.BytesIO):
        status = 200
        headers = {}

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: Response(b"abc"))
    with pytest.raises(ValueError, match="range/ETag"):
        module.download_part(0, tmp_path)
    assert list(tmp_path.iterdir()) == []

"""Verify and prepare public MSR-VTT media for the native baseline.

The public archive's observed hash and coverage are recorded, not claimed to be
an independently published checksum. No checkpoint substitution or scores.
"""

import argparse
import concurrent.futures
import io
import json
import re
import shutil
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from temporal_hoi.data.msrvtt import MSRVTTAdapter, file_sha256

MEDIA_URL = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"
MEDIA_BYTES = 6552768292
MEDIA_ETAG = '"186934f24-5c4686e7ccdae"'
PART_BYTES = 16 * 1024**2
SOURCE_AUXILIARY_FILES = {
    f"MSRVTT/videos/all/._video{number}.mp4": 4096 for number in (7, 3, 9560, 9552)
}


class RemoteArchive(io.RawIOBase):
    """Seekable ETag-locked HTTP reader for ZIP metadata and selected entries only."""

    def __init__(self, cached_ranges=None):
        super().__init__()
        self.position = 0
        self.cached_ranges = {} if cached_ranges is None else cached_ranges
        self.cache_reads = cached_ranges is None

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else MEDIA_BYTES if whence == 2 else 0)
        if whence not in {0, 1, 2} or not 0 <= position <= MEDIA_BYTES:
            raise ValueError("Invalid remote ZIP seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = min(MEDIA_BYTES - self.position, size if size >= 0 else MEDIA_BYTES)
        if size == 0:
            return b""
        if not 0 < size <= 32 * 1024**2:
            raise ValueError("Remote ZIP read exceeds bounded range")
        start, end = self.position, self.position + size - 1
        if (start, end) in self.cached_ranges:
            self.position += size
            return self.cached_ranges[(start, end)]
        for attempt in range(4):
            try:
                request = urllib.request.Request(MEDIA_URL, headers={
                    "User-Agent": "baseline-inference-build/1", "Range": f"bytes={start}-{end}",
                    "If-Match": MEDIA_ETAG, "Accept-Encoding": "identity"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if (response.status != 206 or response.headers.get("ETag") != MEDIA_ETAG
                            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{MEDIA_BYTES}"):
                        raise ValueError("Remote ZIP range/ETag mismatch")
                    data = response.read(size + 1)
                if len(data) != size:
                    raise OSError("Incomplete remote ZIP range")
                self.position += size
                if self.cache_reads:
                    self.cached_ranges[(start, end)] = data
                return data
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)


def prepare_inference_media(target, adapter, cache=None):
    """Same official 1k pool; two fixed train-derived clips only for resource probing."""
    all_ids = set(adapter.pool("train9k").video_ids) | set(adapter.pool("test").video_ids)
    test_ids = set(adapter.pool("test").video_ids)
    probe_ids = set(adapter.pool("validation").video_ids[:2])
    expected = test_ids | probe_ids
    metadata_reader = RemoteArchive()
    with zipfile.ZipFile(metadata_reader) as archive:
        members, _ = media_members(archive, all_ids)
        names = {key: members[key].filename for key in expected}
    cache = Path(cache) if cache is not None else Path("/var/cache/temporal-hoi-media/inference-v1")
    cache.mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def extract(video_id):
        path = cache / f"{video_id}.mp4"
        receipt = cache / f"{video_id}.json"
        if path.exists() and receipt.exists():
            previous = json.loads(receipt.read_text())
            if (previous.get("source_etag") != MEDIA_ETAG
                    or path.stat().st_size != previous["bytes"]
                    or file_sha256(path) != previous["sha256"]):
                raise ValueError("Cached inference video hash mismatch")
            return previous
        if not hasattr(local, "archive"):
            local.archive = zipfile.ZipFile(RemoteArchive(metadata_reader.cached_ranges))
        member = local.archive.getinfo(names[video_id])
        temporary = None
        try:
            with local.archive.open(member) as src, tempfile.NamedTemporaryFile(
                    dir=cache, suffix=".partial", delete=False) as dest:
                temporary = Path(dest.name)
                shutil.copyfileobj(src, dest, length=1024 * 1024)
            if temporary.stat().st_size != member.file_size:
                raise ValueError("Extracted inference video size mismatch")
            record = {"video_id": video_id, "bytes": member.file_size,
                      "sha256": file_sha256(temporary), "source_crc32": member.CRC,
                      "source_member": member.filename, "source_etag": MEDIA_ETAG}
            temporary.replace(path)
            receipt.write_text(json.dumps(record) + "\n")
            return record
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        pending = [executor.submit(extract, key) for key in sorted(expected)]
        for future in concurrent.futures.as_completed(pending):
            records.append(future.result())
            if len(records) % 25 == 0 or len(records) == len(expected):
                print(f"Inference media verified: {len(records)}/{len(expected)}", flush=True)
    for video_id in sorted(expected):
        shutil.copyfile(cache / f"{video_id}.mp4", target / f"{video_id}.mp4")
    (target / "media_provenance.json").write_text(json.dumps(sorted(
        records, key=lambda row: row["video_id"]), indent=2) + "\n")
    return {"media_included": True, "media_scope": "official_test1000_plus_resource_probe2",
            "expected_video_count": len(expected), "source_expected_video_count": len(all_ids),
            "video_count": len(expected), "test_video_count": len(test_ids),
            "resource_probe_video_ids": sorted(probe_ids), "training_media_complete": False,
            "media_source_url": MEDIA_URL, "source_archive_bytes": MEDIA_BYTES,
            "source_etag": MEDIA_ETAG, "selected_video_zip_crc_verified": True,
            "excluded_source_auxiliary_files": sorted(SOURCE_AUXILIARY_FILES),
            "observed_archive_sha256": None, "whole_archive_downloaded": False,
            "decoded_media_verified": False}


def download_part(index, cache):
    """Fetch one bounded, ETag-locked byte range; retain verified parts for retry."""
    start = index * PART_BYTES
    end = min(MEDIA_BYTES, start + PART_BYTES) - 1
    size = end - start + 1
    part = cache / f"part-{index:05d}"
    receipt = cache / f"part-{index:05d}.sha256"
    if part.is_file() and receipt.is_file():
        expected = receipt.read_text().strip()
        if part.stat().st_size != size or file_sha256(part) != expected:
            raise ValueError("Cached media part hash mismatch; refuse to use it")
        return part
    for attempt in range(3):
        temporary = None
        try:
            request = urllib.request.Request(MEDIA_URL, headers={
                "User-Agent": "baseline-build/2", "Range": f"bytes={start}-{end}",
                "If-Match": MEDIA_ETAG})
            with urllib.request.urlopen(request, timeout=90) as response:
                if (response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end}/{MEDIA_BYTES}"
                        or response.headers.get("ETag") != MEDIA_ETAG):
                    raise ValueError("Media range/ETag differs from reviewed source")
                with tempfile.NamedTemporaryFile(dir=cache, prefix=f"part-{index}-",
                                                 suffix=".partial", delete=False) as dest:
                    temporary = Path(dest.name)
                    count = 0
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > size:
                            raise ValueError("Media range exceeds expected size")
                        dest.write(chunk)
                if count != size:
                    raise OSError("Incomplete media range")
            digest = file_sha256(temporary)
            temporary.replace(part)
            receipt.write_text(digest + "\n")
            return part
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()  # Only this invocation's own incomplete chunk.


def download_archive(destination, cache):
    """At most four streams, bounded chunks and hash-checked Docker build cache."""
    cache.mkdir(parents=True, exist_ok=True)
    count = (MEDIA_BYTES + PART_BYTES - 1) // PART_BYTES
    parts = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        pending = {executor.submit(download_part, index, cache): index for index in range(count)}
        for future in concurrent.futures.as_completed(pending):
            index = pending[future]
            parts[index] = future.result()
            print(f"MSR-VTT media parts verified: {len(parts)}/{count}", flush=True)
    with destination.open("xb") as dest:
        for index in range(count):
            with parts[index].open("rb") as source:
                shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
    if destination.stat().st_size != MEDIA_BYTES:
        raise ValueError("Assembled media archive size mismatch")


def media_members(archive, expected_ids):
    selected = {}
    total = 0
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                or (member.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError("Unsafe source archive path or symlink")
        if member.is_dir() or path.suffix.lower() != ".mp4":
            continue
        if member.filename in SOURCE_AUXILIARY_FILES:
            if member.file_size != SOURCE_AUXILIARY_FILES[member.filename]:
                raise ValueError("Reviewed auxiliary ZIP entry size changed")
            continue  # Four observed AppleDouble sidecars, not benchmark video IDs.
        video_id = path.stem
        if (video_id not in expected_ids or not re.fullmatch(r"video[0-9]+", video_id)
                or video_id in selected or not 0 < member.file_size < 512 * 1024**2):
            raise ValueError("Unexpected, duplicate, empty or oversized media file")
        total += member.file_size
        selected[video_id] = member
    if set(selected) != set(expected_ids) or total > 12 * 1024**3:
        raise ValueError("Media coverage mismatch or excessive archive expansion")
    return selected, total


def prepare(include_media, scope="inference"):
    root = Path("/app")
    if not root.is_dir() or Path.cwd() != root:
        raise ValueError("This asset preparation must run inside the staged /app container")
    adapter = MSRVTTAdapter.from_files(root / "data/metadata/msrvtt_data.zip",
                                       root / "data/metadata/msrvtt_protocol_lock.json")
    expected = set(adapter.pool("train9k").video_ids) | set(adapter.pool("test").video_ids)
    receipt = {"schema_version": "container_baseline_assets_v1", "media_included": False,
               "checkpoint_included": False, "benchmark_inference_run": False,
               "benchmark_metrics": None, "expected_video_count": len(expected)}
    if include_media:
        required = 24 if scope == "all" else 4
        if shutil.disk_usage(root).free < required * 1024**3:
            raise ValueError(f"Require {required} GiB free during media preparation")
        target = root / "data/raw/msrvtt"
        target.mkdir(parents=True, exist_ok=False)
        if scope == "inference":
            receipt.update(prepare_inference_media(target, adapter))
        else:
            prepare_full_media(target, expected, receipt)
    (root / "container_assets.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


def prepare_full_media(target, expected, receipt):
    """Optional complete 10k media set for future training, separate from inference."""
    with tempfile.TemporaryDirectory(prefix="msrvtt-build-") as temporary:
        source = Path(temporary) / "MSRVTT.zip"
        download_archive(source, Path("/var/cache/temporal-hoi-media/msrvtt-v1"))
        digest = file_sha256(source)
        with zipfile.ZipFile(source) as archive:
            selected, expanded = media_members(archive, expected)
            for video_id, member in selected.items():
                with archive.open(member) as src, (target / f"{video_id}.mp4").open("xb") as dest:
                    shutil.copyfileobj(src, dest, length=8 * 1024 * 1024)
        receipt.update(media_included=True, media_source_url=MEDIA_URL,
                       observed_archive_sha256=digest, archive_bytes=MEDIA_BYTES,
                       source_etag=MEDIA_ETAG,
                       media_expanded_bytes=expanded, video_count=len(selected),
                       decoded_media_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-media", type=int, choices=[0, 1], default=1)
    parser.add_argument("--media-scope", choices=["inference", "all"], default="inference")
    args = parser.parse_args()
    prepare(bool(args.include_media), args.media_scope)


if __name__ == "__main__":
    main()

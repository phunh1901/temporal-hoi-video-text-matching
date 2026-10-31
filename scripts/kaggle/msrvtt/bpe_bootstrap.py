"""Fetch the exact public tokenizer asset without embedding it in notebook source."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

BPE_NAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_BYTES = 1_356_917
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
BPE_URL = (
    "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/"
    "508ffa3de39ba0563a03199c440ab602a72e9b6f/modules/" + BPE_NAME
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(256 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path):
    if Path(path).stat().st_size != BPE_BYTES or _sha256(path) != BPE_SHA256:
        raise ValueError("Tokenizer size/SHA256 differs from the locked baseline asset")


def ensure_bpe(destination, input_root=None):
    """Return a verified cache/attached/downloaded file; never execute downloaded data."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        _verify(destination)
        return destination
    attached = sorted(Path(input_root).rglob(BPE_NAME)) if input_root else []
    attached = [p for p in attached if p.is_file()]
    if attached:
        for path in attached:
            _verify(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="bpe-", suffix=".download", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if attached:
            shutil.copyfile(attached[0], temporary)
            source = "attached_verified_asset"
        else:
            request = urllib.request.Request(BPE_URL, headers={"User-Agent": "msrvtt-kaggle/2"})
            try:
                with (
                    urllib.request.urlopen(request, timeout=30) as response,
                    temporary.open("wb") as output,
                ):
                    if response.status != 200:
                        raise ValueError("Tokenizer download needs a complete HTTP200 response")
                    received = 0
                    while chunk := response.read(min(256 * 1024, BPE_BYTES + 1 - received)):
                        received += len(chunk)
                        if received > BPE_BYTES:
                            raise ValueError("Tokenizer download exceeds pinned byte count")
                        output.write(chunk)
            except OSError as exc:
                raise RuntimeError(
                    "Không tải được tokenizer công khai. Bật Internet Kaggle, "
                    "hoặc Add Input file bpe_simple_vocab_16e6.txt.gz đúng baseline."
                ) from exc
            source = BPE_URL
        _verify(temporary)
        os.replace(temporary, destination)
        receipt = {"source": source, "bytes": BPE_BYTES, "sha256": BPE_SHA256}
        destination.with_suffix(destination.suffix + ".json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination

"""Verify and prepare the pinned jaewooo native CLIP4Clip checkpoint.

Only verified original components execute. Remote model bytes are hash checked
before weights_only loading. Synthetic CPU checks do not measure retrieval.
"""

import json
import os
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

from temporal_hoi.data.msrvtt import file_sha256

REVISION = "de15dc5686a8628d051dae0a85b80d34ed690c68"
ROOT_URL = f"https://huggingface.co/jaewooo/test/resolve/{REVISION}"
CHECKPOINT_URL = ROOT_URL + "/ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.4"
CHECKPOINT_SHA256 = "3d7e2ba508986c7347380767233b017d4ac4c9fabc135f3457de8625dc833dca"
CHECKPOINT_BYTES = 353556437
LOG_SHA256 = "e6ce285755f9c657d33f5215e1796baf094cda32a8728a2189864baa413e5a4e"
LOG_BYTES = 22286632


def fetch_checked(url, destination, size, expected_hash):
    """Reuse only an exact hash match; bound response size and own temporary files."""
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size != size or file_sha256(destination) != expected_hash:
            raise ValueError("Cached source hash/size mismatch")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        temporary = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "native-baseline-build/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 200:
                    raise ValueError("Expected complete source response")
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    count = 0
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > size:
                            raise ValueError("Source exceeds reviewed size")
                        stream.write(chunk)
                if count != size:
                    raise OSError("Incomplete checkpoint/source response")
            if file_sha256(temporary) != expected_hash:
                raise ValueError("Source SHA256 mismatch; refuse model loading")
            temporary.replace(destination)
            print(json.dumps({"verified_source": destination.name, "bytes": size,
                              "sha256": expected_hash}), flush=True)
            return destination
        except OSError:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def install():
    import torch

    from temporal_hoi.inference.clip4clip import load_component, load_model, text_tokens

    root = Path("/app")
    if Path.cwd() != root or not (root / "container_assets.json").is_file():
        raise ValueError("Run only in the prepared /app baseline container")
    cache = Path("/var/cache/temporal-hoi-checkpoints")
    checkpoint = fetch_checked(CHECKPOINT_URL, cache / "native-epoch4.bin",
                               CHECKPOINT_BYTES, CHECKPOINT_SHA256)
    log = fetch_checked(ROOT_URL + "/ckpts/ckpt_msrvtt_retrieval_looseType/log.txt",
                        cache / "training-source.log", LOG_BYTES, LOG_SHA256)
    # Record save events and configuration, never inspect scores to choose an epoch.
    saves = [line for line in log.read_text().splitlines() if "Model saved to" in line]
    expected_names = [f"pytorch_model.bin.{epoch}" for epoch in range(5)]
    if not all(any(name in line for line in saves) for name in expected_names):
        raise ValueError("Published training log does not contain the reviewed save sequence")
    target = root / "weights/clip4clip_native_third_party"
    target.mkdir(parents=True, exist_ok=False)
    destination = target / "pytorch_model.bin.4"
    shutil.copyfile(checkpoint, destination)
    request = urllib.request.Request(ROOT_URL + "/LICENSE",
                                     headers={"User-Agent": "native-baseline-build/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        license_bytes = response.read(65537)
    if len(license_bytes) > 65536 or b"MIT License" not in license_bytes:
        raise ValueError("Missing expected public repository MIT license")
    (target / "LICENSE").write_bytes(license_bytes)
    torch.set_num_threads(2)
    model = load_model(root / "external/CLIP4Clip", destination, "cpu")
    tokenizer = load_component(root / "external/CLIP4Clip",
                               "modules/tokenization_clip.py").SimpleTokenizer()
    tokens = torch.from_numpy(text_tokens(tokenizer, "a person moves"))
    with torch.inference_mode():
        image_features = model.encode_image(torch.zeros(1, 3, 224, 224))
        text_features = model.encode_text(tokens)
    if (tuple(image_features.shape) != (1, 512) or tuple(text_features.shape) != (1, 512)
            or not torch.isfinite(image_features).all() or not torch.isfinite(text_features).all()):
        raise ValueError("Native synthetic CPU encoder check failed")
    provenance = {
        "source_url": CHECKPOINT_URL, "publisher": "jaewooo/test on Hugging Face",
        "publisher_revision": REVISION, "official_author_checkpoint": False,
        "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": CHECKPOINT_BYTES,
        "architecture": "CLIP4Clip_ViT-B32_meanP_loose_2d",
        "training_regime": {
            "evidence": "Public hash-checked training log, not an independent training audit",
            "claimed_train_csv": "MSRVTT_train.9k.csv", "epochs": 5,
            "batch_size": 128, "seed": 42, "max_frames": 12, "max_words": 32,
            "lr": 0.0001, "coef_lr": 0.001,
            "training_source_code_commit_verified": False,
            "training_dataset_bytes_verified": False,
        },
        "selection_rule": "Fixed final saved epoch 4 before local retrieval scores",
        "training_log_url": ROOT_URL + "/ckpts/ckpt_msrvtt_retrieval_looseType/log.txt",
        "training_log_sha256": LOG_SHA256,
        "strict_original_model_load_verified": True,
        "synthetic_cpu_encoder_check": {"image_shape": [1, 512], "text_shape": [1, 512],
                                        "finite": True, "uses_benchmark_examples": False},
        "license_url": ROOT_URL + "/LICENSE", "license_observed_sha256": file_sha256(target / "LICENSE"),
        "limitations": [
            "Third-party replication checkpoint; not released or endorsed by original authors",
            "Published training procedure may have evaluated test during training; selection audit unavailable",
            "Claimed train9k includes the project's internal validation900; use it only for resource/function probes",
            "No claim of paper metric reproduction or independent validation quality",
        ],
        "benchmark_inference_run": False, "benchmark_metrics": None,
    }
    provenance_path = target / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    config = json.loads((root / "configs/msrvtt_clip4clip_inference.json").read_text())
    config.update(checkpoint=str(destination.relative_to(root)),
                  checkpoint_sha256=CHECKPOINT_SHA256,
                  checkpoint_provenance=str(provenance_path.relative_to(root)),
                  selection_fixed_before_scores=True)
    (root / "configs/msrvtt_clip4clip_packaged.json").write_text(json.dumps(config, indent=2) + "\n")
    assets_path = root / "container_assets.json"
    assets = json.loads(assets_path.read_text())
    if assets.get("media_scope") == "official_test1000_plus_resource_probe2":
        assets["source_expected_video_count"] = 10000
        assets["expected_video_count"] = assets["video_count"]
        if "source_zip_crc_verified" in assets:
            assets["selected_video_zip_crc_verified"] = assets.pop("source_zip_crc_verified")
    assets.update(checkpoint_included=True, checkpoint_sha256=CHECKPOINT_SHA256,
                  native_model_strict_load_verified=True, official_author_checkpoint=False,
                  synthetic_cpu_encoder_verified=True)
    assets_path.write_text(json.dumps(assets, indent=2) + "\n")
    print(json.dumps(assets), flush=True)


if __name__ == "__main__":
    if os.name != "posix":
        raise SystemExit("Install checkpoint inside Docker only")
    install()

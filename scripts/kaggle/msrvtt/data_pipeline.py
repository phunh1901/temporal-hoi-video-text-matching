"""Pinned CLIP4Clip meanP encoder, MSR-VTT manifests and resumable feature cache.

Only the original 2-D ViT/meanP architecture is accepted. No DDP, old optimizer,
automatic checkpoint guessing, or silent missing-video substitutions are used.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
import platform
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

SOURCE_REVISION = "508ffa3de39ba0563a03199c440ab602a72e9b6f"
CACHE_SCHEMA = 3


def sha256_file(path, block_size=4 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def safe_load(path):
    # Explicit weights_only avoids the PyTorch 2.6 default change and unsafe pickle.
    return torch.load(path, map_location="cpu", weights_only=True)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_file(explicit, filename, input_root, required=True):
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise FileNotFoundError(f"Không thấy tệp đã cấu hình: {candidate}")
        return candidate
    matches = sorted(Path(input_root).rglob(filename))
    matches = [p for p in matches if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    if not matches:
        raise FileNotFoundError(f"Cần thêm {filename} vào Kaggle Input hoặc đặt đường dẫn rõ ràng.")
    hashes = {sha256_file(p) for p in matches}
    if len(hashes) == 1:
        return matches[0]
    raise ValueError(
        f"Có nhiều bản {filename} khác nhau. Đặt đường dẫn rõ ràng:\n"
        + "\n".join(map(str, matches))
    )


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "video_id" not in rows[0]:
        raise ValueError(f"CSV phải có dòng dữ liệu và cột video_id: {path}")
    if any(not str(r.get("video_id", "")).strip() for r in rows):
        raise ValueError(f"CSV có video_id rỗng: {path}")
    for row in rows:
        row["video_id"] = row["video_id"].strip()
        if not row["video_id"].startswith("video") or not row["video_id"][5:].isdigit():
            raise ValueError(f"video_id phải có dạng video<ID>: {row['video_id']!r}")
    return rows


def evaluation_records(csv_path, text_id_prefix="msrvtt_jsfusion"):
    rows = read_rows(csv_path)
    if any(not r.get("sentence", "").strip() for r in rows):
        raise ValueError("TEST_CSV/VAL_CSV phải có cột sentence không rỗng.")
    records = []
    for i, row in enumerate(rows):
        record = {
            "video_id": row["video_id"],
            "text": row["sentence"],
            "text_id": f"{text_id_prefix}_{i:04d}",
        }
        if row.get("source_url", "").strip():
            record["source_url"] = row["source_url"].strip()
        records.append(record)
    return records


def source_video_urls(captions_json):
    """Read exact annotation source URLs; never infer independence from video IDs."""
    data = json.loads(Path(captions_json).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("videos"), list):
        raise ValueError("MSRVTT_data.json phải có danh sách videos để kiểm nguồn video.")
    sources = {}
    for row in data["videos"]:
        if not isinstance(row, dict) or not isinstance(row.get("video_id"), str):
            raise ValueError("videos có bản ghi hoặc video_id không hợp lệ.")
        vid, url = row["video_id"], row.get("url", "")
        if vid in sources:
            raise ValueError(f"videos có video_id trùng: {vid}")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"Thiếu URL nguồn cho {vid}; không thể kiểm source overlap.")
        sources[vid] = url
    return sources


def training_records(train_csv, captions_json, exclude_ids=()):
    train_rows = read_rows(train_csv)
    if len({r["video_id"] for r in train_rows}) != len(train_rows):
        raise ValueError("TRAIN_CSV có video_id trùng.")
    ids = {r["video_id"] for r in train_rows} - set(exclude_ids)
    candidate_index = {row["video_id"]: index for index, row in enumerate(train_rows)}
    data = json.loads(Path(captions_json).read_text(encoding="utf-8"))
    sentences = data.get("sentences", []) if isinstance(data, dict) else data
    if not isinstance(sentences, list):
        raise ValueError("Annotation phải chứa danh sách sentences.")
    records = []
    seen_sentence_ids = set()
    for annotation_row, row in enumerate(sentences):
        if not isinstance(row, dict):
            raise ValueError("Annotation caption có bản ghi không phải object.")
        if row.get("video_id") in ids:
            caption = row.get("caption", row.get("sentence", ""))
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"Caption rỗng/không phải chuỗi cho {row['video_id']}")
            sentence_id = str(row.get("sen_id", f"row_{annotation_row:06d}"))
            if sentence_id in seen_sentence_ids:
                raise ValueError(f"Annotation có sen_id trùng: {sentence_id}")
            seen_sentence_ids.add(sentence_id)
            records.append(
                {
                    "video_id": row["video_id"],
                    "text": caption,
                    "text_id": f"msrvtt_sen_{sentence_id}",
                    "candidate_index": candidate_index[row["video_id"]],
                    "annotation_row": annotation_row,
                }
            )
    absent = ids - {r["video_id"] for r in records}
    if absent or not records:
        raise ValueError(f"Thiếu caption cho {len(absent)} video train; ví dụ {sorted(absent)[:5]}")
    return records


def make_development_split(
    records,
    test_ids,
    val_csv=None,
    val_size=900,
    seed=42,
    *,
    selection_policy="fixed_epochs",
    source_urls=None,
):
    """Reserve source-group v2 dev at fixed epochs, or require external validation.

    Exact URL groups follow CSV candidate order, then stdlib seeded group shuffle
    and greedy whole-group quota. All caption rows keep annotation order. Reserving
    dev900 does not make it independent of a pretrained train9k encoder; fixed_epochs
    must not use it for epoch selection. URL checks cannot prove pretraining independence.
    """
    test_ids = set(test_ids)
    train_ids = {r["video_id"] for r in records}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Train trùng test: {sorted(overlap)[:5]}")
    if not records:
        raise ValueError("Train không được rỗng.")
    if selection_policy == "fixed_epochs":
        if val_csv:
            raise ValueError("fixed_epochs không dùng VAL_CSV; chọn policy rõ ràng.")
        if not isinstance(val_size, int) or not 0 < val_size < len(train_ids):
            raise ValueError("Quota validation phải để lại train/validation không rỗng.")
        sources = dict(source_urls or {})
        missing = train_ids - sources.keys()
        if missing:
            raise ValueError(f"Thiếu URL nguồn để tách source-group v2: {sorted(missing)[:5]}")
        ordered_ids = list(dict.fromkeys(row["video_id"] for row in records))
        if all("candidate_index" in row for row in records):
            order = {row["video_id"]: row["candidate_index"] for row in records}
            ordered_ids.sort(key=order.__getitem__)
        groups = {}
        for vid in ordered_ids:
            url = sources[vid]
            if not isinstance(url, str) or not url.strip():
                raise ValueError(f"URL nguồn rỗng cho {vid}")
            groups.setdefault(url, []).append(vid)
        shuffled = list(groups)
        random.Random(seed).shuffle(shuffled)
        reserved_ids = set()
        for url in shuffled:
            if len(groups[url]) <= val_size - len(reserved_ids):
                reserved_ids.update(groups[url])
            if len(reserved_ids) == val_size:
                break
        if len(reserved_ids) != val_size:
            raise ValueError("Không đạt quota bằng nguyên nhóm URL; không tự chia nhóm.")
        train = [row for row in records if row["video_id"] not in reserved_ids]
        reserved = [row for row in records if row["video_id"] in reserved_ids]
        return train, reserved, "sourcegroup_v2_dev_reserved_seen_by_encoder_not_used_for_selection"
    if selection_policy != "independent_validation":
        raise ValueError("SELECTION_POLICY chỉ nhận fixed_epochs hoặc independent_validation.")
    if not val_csv:
        raise ValueError("independent_validation yêu cầu VAL_CSV; không tự chia dev900.")
    val = evaluation_records(val_csv, text_id_prefix="msrvtt_validation")
    val_ids = {r["video_id"] for r in val}
    if val_ids & (train_ids | test_ids):
        raise ValueError("Validation trùng ID train/test; không cung cấp validation độc lập.")
    sources = dict(source_urls or {})
    for row in val:
        if row.get("source_url"):
            previous = sources.get(row["video_id"])
            if previous and previous != row["source_url"]:
                raise ValueError(f"URL nguồn mâu thuẫn cho {row['video_id']}")
            sources[row["video_id"]] = row["source_url"]
    missing = (train_ids | test_ids | val_ids) - sources.keys()
    if missing:
        raise ValueError(
            "Thiếu URL nguồn để kiểm validation độc lập; thêm source_url vào "
            f"VAL_CSV. Ví dụ ID thiếu: {sorted(missing)[:5]}"
        )
    train_test_urls = {sources[vid] for vid in train_ids | test_ids}
    if any(sources[vid] in train_test_urls for vid in val_ids):
        raise ValueError("Validation trùng URL nguồn train/test.")
    return records, val, "explicit_validation_disjoint_ids_and_exact_urls_pretraining_unverified"


def video_index(root, wanted_ids):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Không có VIDEO_ROOT: {root}")
    wanted = set(wanted_ids)
    index = {}
    duplicates = []
    for parent, _, filenames in os.walk(root):
        for filename in filenames:
            p = Path(parent) / filename
            if p.suffix.lower() in {".mp4", ".webm", ".avi", ".mkv", ".mov"} and p.stem in wanted:
                if p.stem in index:
                    duplicates.append(p.stem)
                else:
                    index[p.stem] = p
    if duplicates:
        raise ValueError(
            "Có nhiều tệp cho cùng video_id; đặt VIDEO_ROOT cụ thể hơn: "
            + ", ".join(sorted(set(duplicates))[:10])
        )
    missing = wanted - index.keys()
    if missing:
        raise FileNotFoundError(
            f"Thiếu {len(missing)} video, ví dụ {sorted(missing)[:10]}. "
            "Không tự bỏ video vì sẽ đổi candidate pool."
        )
    return index


def make_preprocess(size):
    # Same PIL integer Resize and CenterCrop geometry as torchvision's original
    # transforms, without importing torchvision's compiled detection operators.
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073))[:, None, None]
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711))[:, None, None]

    def transform(image):
        image = image.convert("RGB")
        width, height = image.size
        if width <= height:
            resized = (size, int(size * height / width))
        else:
            resized = (int(size * width / height), size)
        if image.size != resized:
            image = image.resize(resized, Image.Resampling.BICUBIC)
        left = int(round((image.width - size) / 2.0))
        top = int(round((image.height - size) / 2.0))
        image = image.crop((left, top, left + size, top + size))
        values = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1)
        return (values.float().div(255.0) - mean) / std

    return transform


def decode_video(path, preprocess, max_frames=12):
    """Match original feature_framerate=1, integer-fps seeks, slice_framepos=2.

    Decode every in-range one-second candidate and reject corruption. Transform
    only the up-to-12 uniformly selected candidates to keep CPU memory bounded.
    Expected EOF after the declared frame count is not a decode failure.
    """
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1:
        raise ValueError("max_frames phải là số nguyên dương.")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Không mở được video: {path}")
        fps_float = float(cap.get(cv2.CAP_PROP_FPS))
        count_float = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps_float) or not math.isfinite(count_float):
            raise ValueError(f"FPS/frame count không hợp lệ: {path}")
        fps, frame_count = int(fps_float), int(count_float)
        if fps < 1 or frame_count < 1:
            raise ValueError(f"FPS hoặc số frame bằng 0: {path}")
        duration = (frame_count + fps - 1) // fps
        # MSR-VTT clips are short. Refuse implausible headers, do not allocate huge tensors.
        if duration > 600:
            raise ValueError(f"Video dài bất thường ({duration}s), kiểm tra header: {path}")
        indices = list(range(0, frame_count, fps))
        selected = (
            np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            if len(indices) > max_frames
            else np.arange(len(indices))
        )
        selected_set = {int(i) for i in selected}
        candidates = []
        for candidate_index, frame_index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(
                    f"Lỗi decode frame {frame_index}/{frame_count}: {path}. "
                    "Không tự rút ngắn video/candidate pool."
                )
            if candidate_index in selected_set:
                candidates.append(
                    preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                )
        if not candidates:
            raise ValueError(f"Không có frame hợp lệ: {path}")
        frames = torch.stack(candidates)
        chosen = [indices[int(i)] for i in selected]
        return frames, {
            "fps": fps_float,
            "sampling_fps_integer": fps,
            "frame_count": frame_count,
            "frame_indices": chosen,
            "timestamps_seconds": [i / fps_float for i in chosen],
        }
    finally:
        cap.release()


def tokenize_records(texts, tokenizer, max_words=32):
    if not 2 <= max_words <= 77:
        raise ValueError("max_words phải trong [2,77].")
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    result = torch.zeros((len(texts), max_words), dtype=torch.long)
    for i, text in enumerate(texts):
        ids = [sot] + tokenizer.encode(text)[: max_words - 2] + [eot]
        result[i, : len(ids)] = torch.tensor(ids)
    return result


def load_encoder(checkpoint, vendor_dir, device="cuda", precision="original_fp16"):
    vendor_dir = Path(vendor_dir)
    clip_module = import_file("pinned_clip4clip_encoder", vendor_dir / "module_clip.py")
    tokenizer_module = import_file(
        "pinned_clip4clip_tokenizer", vendor_dir / "tokenization_clip.py"
    )
    state = safe_load(checkpoint)
    for wrapper in ("state_dict", "model_state_dict", "model"):
        if isinstance(state, dict) and wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
            break
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint không chứa state_dict hợp lệ.")
    if any(not isinstance(k, str) for k in state):
        raise ValueError("Tên tham số checkpoint phải là chuỗi.")
    renamed = {k.removeprefix("module."): v for k, v in state.items()}
    if len(renamed) != len(state):
        raise ValueError("Checkpoint có tên tham số trùng sau khi bỏ module. prefix.")
    state = renamed
    forbidden = (
        "transformerClip.",
        "lstm_visual.",
        "frame_position_embeddings.",
        "clip.visual.conv2.",
    )
    if any(k.startswith(forbidden) for k in state):
        raise ValueError(
            "Notebook chỉ chấp nhận checkpoint meanP/linear_patch=2d; "
            "không dùng seqTransf/seqLSTM/3d."
        )
    if any(k.startswith("clip.") for k in state):
        extras = sorted(k for k in state if not k.startswith("clip."))
        if extras:
            raise ValueError(
                "Checkpoint có module ngoài encoder; cần kiểm tra architecture: " + str(extras[:10])
            )
        state = {k[5:]: v for k, v in state.items()}
    if "visual.proj" not in state or "visual.conv2.weight" in state:
        raise ValueError("Cần checkpoint CLIP4Clip ViT meanP 2D đầy đủ.")
    # build_model infers dimensions and strict-loads every encoder parameter.
    model = clip_module.build_model(state.copy())
    if precision == "fp32" or str(device) == "cpu":
        model.float()
        # build_model creates FP16 weights; reload after .float() to retain FP32
        # checkpoint values for the explicit FP32/debug branch.
        model.load_state_dict(
            {
                k: v
                for k, v in state.items()
                if k not in {"input_resolution", "context_length", "vocab_size"}
            },
            strict=True,
        )
    elif precision != "original_fp16":
        raise ValueError("PRECISION chỉ nhận original_fp16 hoặc fp32.")
    model = model.eval().requires_grad_(False).to(device)
    tokenizer = tokenizer_module.SimpleTokenizer(str(vendor_dir / "bpe_simple_vocab_16e6.txt.gz"))
    info = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_revision": SOURCE_REVISION,
        "precision": precision,
        "vendor_sha256": {
            name: sha256_file(vendor_dir / name)
            for name in ("module_clip.py", "tokenization_clip.py", "bpe_simple_vocab_16e6.txt.gz")
        },
        "effective_weight_dtype": str(model.dtype),
        "logit_scale": float(model.logit_scale.exp().detach().cpu()),
        "embed_dim": int(model.text_projection.shape[1]),
        "resolution": int(model.visual.input_resolution),
    }
    del state
    gc.collect()
    return model, tokenizer, info


@torch.inference_mode()
def encode_in_chunks(model, values, kind, device, initial_chunk=16):
    """Retry only actual CUDA OOM, down to one frame/text. Never swallow other errors."""
    if kind not in {"image", "text"}:
        raise ValueError("kind phải là image hoặc text.")
    if not isinstance(initial_chunk, int) or isinstance(initial_chunk, bool) or initial_chunk < 1:
        raise ValueError("Batch encoder phải là số nguyên dương.")
    if not len(values):
        raise ValueError("Không encode input rỗng.")
    chunks, start, chunk_size = [], 0, initial_chunk
    while start < len(values):
        stop = min(start + chunk_size, len(values))
        try:
            inputs = values[start:stop].to(device)
            output = model.encode_image(inputs) if kind == "image" else model.encode_text(inputs)
            output = output.float().cpu()
            if not torch.isfinite(output).all() or (output.norm(dim=-1) == 0).any():
                raise ValueError(f"Encoder trả đặc trưng NaN/Inf/zero ({kind}).")
            chunks.append(output)
            del inputs, output
            start = stop
        except torch.cuda.OutOfMemoryError:
            if chunk_size == 1:
                raise RuntimeError(
                    "Không đủ VRAM cho một mẫu; đóng notebook khác hoặc chọn GPU khác."
                )
            if "inputs" in locals():
                del inputs
            if "output" in locals():
                del output
            gc.collect()
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
            print(f"Giảm batch encoder còn {chunk_size} sau CUDA OOM.")
    return torch.cat(chunks, dim=0)


def build_cache(
    records,
    index,
    model,
    tokenizer,
    model_info,
    cache_root,
    max_frames=12,
    max_words=32,
    device="cuda",
    encoder_batch=16,
    text_batch=64,
    text_shard=1024,
):
    """Per-video and per-text-shard atomic cache, reused across train/val/test."""
    for name, value in (
        ("max_frames", max_frames),
        ("encoder_batch", encoder_batch),
        ("text_batch", text_batch),
        ("text_shard", text_shard),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} phải là số nguyên dương.")
    if not records or len({row["text_id"] for row in records}) != len(records):
        raise ValueError("Records phải không rỗng và text_id phải duy nhất.")
    cache_root = Path(cache_root)
    environment = {
        "torch": str(torch.__version__),
        "opencv": str(cv2.__version__),
        "numpy": str(np.__version__),
        "python": platform.python_version(),
    }
    import ftfy
    import PIL
    import regex

    environment.update({"pillow": PIL.__version__})
    environment.update(
        {
            "ftfy": str(ftfy.__version__),
            "regex": str(regex.__version__),
            "cuda": str(torch.version.cuda),
            "device": str(device),
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        }
    )
    settings = {
        **model_info,
        "schema": CACHE_SCHEMA,
        "pipeline_sha256": sha256_file(__file__),
        "max_frames": max_frames,
        "max_words": max_words,
        "sampling": "original_integer_fps_1fps_then_uniform12",
        "preprocess": "PIL_bicubic_shortedge_centercrop_RGB_CLIP",
        "environment": environment,
    }
    namespace = cache_root / digest(settings)[:20]
    namespace.mkdir(parents=True, exist_ok=True)
    preprocess = make_preprocess(model_info["resolution"])
    video_ids = list(dict.fromkeys(r["video_id"] for r in records))
    if all("candidate_index" in row for row in records):
        order = {row["video_id"]: row["candidate_index"] for row in records}
        video_ids.sort(key=order.__getitem__)
    dim = model_info["embed_dim"]
    frames = torch.zeros((len(video_ids), max_frames, dim), dtype=torch.float32)
    mask = torch.zeros((len(video_ids), max_frames), dtype=torch.bool)
    video_metadata = []
    started = time.perf_counter()
    for i, vid in enumerate(video_ids):
        path = index[vid]
        input_sha = sha256_file(path)
        output = namespace / "videos" / f"{digest([vid, input_sha])}.pt"
        if output.exists():
            try:
                item = safe_load(output)
                if (
                    not isinstance(item, dict)
                    or item.get("input_sha256") != input_sha
                    or item.get("video_id") != vid
                ):
                    raise ValueError("Hash cache không khớp")
                features = item["features"]
                if (
                    not isinstance(features, torch.Tensor)
                    or features.ndim != 2
                    or features.shape[1] != dim
                    or not 1 <= features.shape[0] <= max_frames
                    or not features.is_floating_point()
                    or not torch.isfinite(features).all()
                    or not (features.norm(dim=-1) > 0).all()
                    or len(item["sampling"]["frame_indices"]) != len(features)
                ):
                    raise ValueError("Tensor cache không hợp lệ")
            except (
                OSError,
                EOFError,
                RuntimeError,
                ValueError,
                KeyError,
                AttributeError,
                TypeError,
                IndexError,
                pickle.UnpicklingError,
            ):
                item = None
        else:
            item = None
        if item is None:
            decoded, sampling = decode_video(path, preprocess, max_frames)
            features = encode_in_chunks(model, decoded, "image", device, encoder_batch)
            item = {
                "features": features,
                "video_id": vid,
                "input_sha256": input_sha,
                "sampling": sampling,
                "path": str(path),
            }
            atomic_torch(output, item)
        count = item["features"].shape[0]
        frames[i, :count] = item["features"]
        mask[i, :count] = True
        video_metadata.append({k: v for k, v in item.items() if k != "features"})
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(video_ids):
            print(
                f"Cache video {i + 1}/{len(video_ids)} ({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    text = torch.empty((len(records), dim), dtype=torch.float32)
    for start in range(0, len(records), text_shard):
        subset = records[start : start + text_shard]
        key = digest([r["text"] for r in subset])
        output = namespace / "texts" / f"{key}.pt"
        item = None
        if output.exists():
            try:
                candidate = safe_load(output)
                if (
                    isinstance(candidate, torch.Tensor)
                    and candidate.is_floating_point()
                    and candidate.shape == (len(subset), dim)
                    and torch.isfinite(candidate).all()
                    and (candidate.norm(dim=-1) > 0).all()
                ):
                    item = candidate
            except (
                OSError,
                EOFError,
                RuntimeError,
                ValueError,
                AttributeError,
                TypeError,
                IndexError,
                pickle.UnpicklingError,
            ):
                pass
        if item is None:
            tokens = tokenize_records([r["text"] for r in subset], tokenizer, max_words)
            item = encode_in_chunks(model, tokens, "text", device, text_batch)
            atomic_torch(output, item)
        text[start : start + len(subset)] = item
        if start == 0 or start // text_shard % 10 == 0:
            print(
                f"Cache caption {min(start + text_shard, len(records))}/{len(records)}", flush=True
            )
    metadata = {
        "settings": settings,
        "records_sha256": digest(records),
        "video_inputs_sha256": digest([(m["video_id"], m["input_sha256"]) for m in video_metadata]),
        "namespace": str(namespace),
        "seconds": time.perf_counter() - started,
        "video_metadata": video_metadata,
        "texts": [r["text"] for r in records],
    }
    metadata["fingerprint"] = digest(
        {k: metadata[k] for k in ("settings", "records_sha256", "video_inputs_sha256")}
    )
    return {
        "frames": frames,
        "mask": mask,
        "text": text,
        "video_ids": video_ids,
        "text_ids": [r["text_id"] for r in records],
        "text_video_ids": [r["video_id"] for r in records],
        "metadata": metadata,
    }

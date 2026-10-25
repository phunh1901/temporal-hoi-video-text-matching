"""Strict ViT-B/32 meanP inference using hash-pinned CLIP4Clip components.

No download, training, weight substitution or implicit data filtering occurs here.
External Python modules are executed only after every pinned file is verified.
"""

import importlib.util
from pathlib import Path

import numpy as np

from temporal_hoi.data.msrvtt import file_sha256

UPSTREAM_COMMIT = "508ffa3de39ba0563a03199c440ab602a72e9b6f"
SOURCE_HASHES = {
    "modules/module_clip.py": "6efe65dcc44503c75e33ca1c49d6c53e7441c3bc48ec707081fa55e2833c6a5b",
    "modules/tokenization_clip.py": "60af2eccd9d1208933213b57d24c9ac337ca4be935bc6550a7300b656a6f3bb5",
    "modules/bpe_simple_vocab_16e6.txt.gz": "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
    "dataloaders/rawvideo_util.py": "cb62541f12da740d9c1511519c9e4ae0785517413bd712ac32e7fe6776ba255a",
    "modules/modeling.py": "2bd870e24ee0ce9b0b5f036b1fac01adfdbb22bd392a2fe74b9c385473853e84",
    "dataloaders/dataloader_msrvtt_retrieval.py": "21963bb8d650ef23a1251615e91641e5d50eb742315bf33aba85023e6e15a334",
    "LICENSE": "7b98bfc12b332914ac5b38d8675f4cca98a1c676cf28360a606f84b59f1b2e3f",
}


def verify_source(root):
    """Check exact code/tokenizer bytes before permitting imports from this tree."""
    root = Path(root).resolve()
    for name, expected in SOURCE_HASHES.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or escaped pinned source: {name}")
        if file_sha256(path) != expected:
            raise ValueError(f"Pinned source hash mismatch: {name}")
    return dict(SOURCE_HASHES)


def load_component(root, relative_path):
    verify_source(root)
    if relative_path not in SOURCE_HASHES or not relative_path.endswith(".py"):
        raise ValueError("Only a pinned Python component may be imported")
    name = "temporal_hoi_verified_" + relative_path.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, Path(root) / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text_tokens(tokenizer, sentence, max_words=32):
    """Native start/text/end IDs, truncation and zero padding; shape [1,32]."""
    if max_words != 32 or not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Require a nonempty raw caption and locked max_words=32")
    words = ["<|startoftext|>"] + tokenizer.tokenize(sentence)
    words = words[:max_words - 1] + ["<|endoftext|>"]
    ids = tokenizer.convert_tokens_to_ids(words)
    return np.asarray([ids + [0] * (max_words - len(ids))], dtype=np.int64)


def sampled_indices(frame_count, fps, max_frames=12):
    """Match native int(fps), 1 Hz, slice_framepos=2 for fully decodable video."""
    if frame_count < 1 or not np.isfinite(fps) or fps < 1 or max_frames != 12:
        raise ValueError("Invalid video metadata or non-reference max_frames")
    all_indices = np.arange(0, frame_count, int(fps), dtype=np.int64)
    selected = (np.linspace(0, len(all_indices) - 1, max_frames, dtype=int)
                if len(all_indices) > max_frames else np.arange(len(all_indices)))
    return all_indices, all_indices[selected]


def decode_video(path, transform, guard=lambda: None):
    """Strict native sampled decode; return [T,3,224,224], seconds and indices.

    Decode every 1 Hz sample so corruption cannot silently shorten a clip. Only
    transform/store the at most 12 frames selected by the native uniform rule.
    EOF beyond the declared frame count is not treated as a sample.
    """
    import cv2
    import torch
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        indices, selected = sampled_indices(frame_count, fps)
        selected_set = set(selected.tolist())
        frames, timestamps = [], []
        for index in indices:
            guard()
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Sampled decode failed: {path}, frame {index}")
            if index in selected_set:
                timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                if (not np.isfinite(timestamp) or timestamp < 0
                        or (timestamps and timestamp <= timestamps[-1])):
                    raise ValueError("Invalid/non-increasing decoder timestamp")
                frames.append(transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
                timestamps.append(timestamp)
        tensor = torch.stack(frames)
        if tensor.shape != (len(selected), 3, 224, 224) or not torch.isfinite(tensor).all():
            raise ValueError("Unexpected transformed video shape or nonfinite pixels")
        return tensor, timestamps, selected.tolist()
    finally:
        cap.release()


def checkpoint_clip_state(checkpoint):
    """Reject non-meanP states and every non-tensor value; never load pickle code."""
    import torch

    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a native CLIP4Clip tensor state_dict")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if any(not key.startswith("clip.") for key in state):
        raise ValueError("Require native meanP state: unexpected non-clip keys; no conversion")
    clip_state = {key.removeprefix("clip."): value for key, value in state.items()}
    if any(not isinstance(value, torch.Tensor) for value in clip_state.values()):
        raise ValueError("Checkpoint state must contain tensors only")
    for key, shape in {"visual.conv1.weight": (768, 3, 32, 32),
                       "text_projection": (512, 512),
                       "positional_embedding": (77, 512),
                       "token_embedding.weight": (49408, 512)}.items():
        if key not in clip_state or tuple(clip_state[key].shape) != shape:
            raise ValueError(f"Checkpoint is not reference ViT-B/32: {key}")
    return clip_state


def load_model(root, checkpoint_path, device):
    import torch

    module = load_component(root, "modules/module_clip.py")
    state = checkpoint_clip_state(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    # meanP loose_type has only CLIP parameters; no learned temporal/cross head.
    model = module.CLIP(512, 224, 12, 768, 32, 77, 49408, 512, 8, 12, linear_patch="2d")
    for key in ("input_resolution", "context_length", "vocab_size"):
        state.pop(key, None)  # Upstream non-parameter metadata buffers.
    model.load_state_dict(state, strict=True)
    del state
    return model.float().to(device).eval()


def normalized_mean(frame_features):
    """Native frame L2 → valid-frame mean → video L2; shape [T,D] → [D]."""
    import torch

    if (frame_features.ndim != 2 or not frame_features.shape[0]
            or not torch.isfinite(frame_features).all()):
        raise ValueError("Need nonempty finite frame features")
    norms = frame_features.norm(dim=-1, keepdim=True)
    if (norms <= 0).any():
        raise ValueError("Zero frame feature")
    pooled = (frame_features / norms).mean(dim=0)
    norm = pooled.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid mean-pooled feature")
    return pooled / norm

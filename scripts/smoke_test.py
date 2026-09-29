"""Smoke test for video-text matching using OpenCLIP without fine-tuning (W01-03).

This script:
1. Opens a sample video and uniformly samples 8 frames across its duration.
2. Encodes the 8 frames using frozen OpenCLIP (ViT-B-32).
3. Computes the video representation via mean-pooling and L2 normalization (checking for non-zero norm).
4. Encodes 4 candidate sentences and computes cosine similarity scores.
5. Verifies finite numerical outputs, measures latency, records hardware/library versions,
   and saves structured outputs to outputs/w01/smoke_test.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open_clip
from PIL import Image
import torch

DEFAULT_VIDEO_PATH = "data/sample/smoke_sample.mp4"
DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"

PROMPT_CANDIDATES = [
    {
        "id": "T1_MATCH",
        "category": "video_content",
        "text": "a satellite view of planet earth rotating in space",
        "description": "Mô tả đúng ngữ cảnh của video thử nghiệm (Earth rotation in space)",
    },
    {
        "id": "T2_RULE1",
        "category": "parking_rule",
        "text": "a person parking a motorcycle on the pedestrian walkway",
        "description": "Hành vi chính 1: Đỗ xe máy trên vỉa hè",
    },
    {
        "id": "T3_RULE2",
        "category": "littering_rule",
        "text": "a person littering a trash bag on the ground",
        "description": "Hành vi chính 2: Đổ rác bừa bãi xuống đất",
    },
    {
        "id": "T4_RULE_EXT",
        "category": "soccer_rule",
        "text": "people playing soccer and kicking a ball in the yard",
        "description": "Hành vi mở rộng: Đá bóng ở sân",
    },
]


def extract_uniform_frames(video_path: str, num_frames: int = 8) -> tuple[list[Image.Image], dict[str, Any]]:
    """Uniformly sample `num_frames` from the video and return PIL images with metadata."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if total_frames > 0 else 0.0

    if total_frames < num_frames:
        cap.release()
        raise ValueError(
            f"Video has only {total_frames} frames, but {num_frames} frames are required."
        )

    # Calculate uniform indices across the timeline
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames: list[Image.Image] = []
    frame_timestamps: list[float] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr_frame = cap.read()
        if not ok or bgr_frame is None:
            cap.release()
            raise RuntimeError(f"Failed to read frame at index {idx}")

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        frames.append(pil_img)
        frame_timestamps.append(float(idx / fps))

    cap.release()

    meta = {
        "video_path": video_path,
        "fps": float(fps),
        "total_frames": total_frames,
        "resolution": f"{width}x{height}",
        "duration_seconds": round(duration_s, 3),
        "sampled_frame_indices": indices.tolist(),
        "sampled_timestamps_s": [round(t, 3) for t in frame_timestamps],
    }
    return frames, meta


def run_smoke_test(
    video_path: str = DEFAULT_VIDEO_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    output_dir: str = "outputs/w01",
) -> dict[str, Any]:
    """Execute the full smoke test pipeline."""
    t_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("TEMPORAL HOI - OPENCLIP VIDEO-TEXT MATCHING SMOKE TEST (W01-03)")
    print("=" * 70)
    print(f"Device: {device.upper()}")
    print(f"Platform: {platform.system()} {platform.release()} (Python {platform.python_version()})")
    print(f"Model: {model_name} | Pretrained Checkpoint: {pretrained}")
    print(f"Video target: {video_path}")
    print("-" * 70)

    # 1. Load model and preprocessing transform
    print("[1/5] Loading OpenCLIP model and tokenizer...")
    t_load_start = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()  # Strictly inference mode, no dropout, frozen weights
    load_time_s = time.perf_counter() - t_load_start
    print(f"      Model loaded successfully in {load_time_s:.2f}s")

    # 2. Extract 8 uniform frames
    print("[2/5] Sampling 8 uniform frames from video...")
    frames, video_meta = extract_uniform_frames(video_path, num_frames=8)
    print(
        f"      Loaded {len(frames)} frames ({video_meta['resolution']}, "
        f"{video_meta['duration_seconds']}s duration, FPS={video_meta['fps']})"
    )

    # 3. Encode image frames and compute video embedding
    print("[3/5] Encoding image frames and computing mean-pooled video embedding...")
    t_infer_start = time.perf_counter()
    image_tensors = torch.stack([preprocess(f) for f in frames]).to(device)  # shape: [8, 3, 224, 224]

    with torch.no_grad():
        frame_features = model.encode_image(image_tensors)  # shape: [8, D]
        # Frame-level L2 normalization
        frame_norms = frame_features.norm(dim=-1, keepdim=True)
        if (frame_norms < 1e-12).any():
            raise ZeroDivisionError("Detected zero-norm frame feature vector!")
        frame_features = frame_features / frame_norms

        # Baseline B0: Temporal Mean Pooling across 8 frames
        video_emb = frame_features.mean(dim=0, keepdim=True)  # shape: [1, D]

        # Video-level L2 normalization
        v_norm = video_emb.norm(dim=-1, keepdim=True)
        if (v_norm < 1e-12).any():
            raise ZeroDivisionError("Detected zero-norm video embedding vector!")
        video_emb = video_emb / v_norm  # shape: [1, D]

    embed_dim = video_emb.shape[-1]
    print(f"      Video embedding shape: {list(video_emb.shape)} (Dimension D={embed_dim})")

    # 4. Encode 4 candidate texts and compute Cosine Similarity
    print("[4/5] Encoding 4 candidate sentences and computing cosine similarity...")
    texts = [item["text"] for item in PROMPT_CANDIDATES]
    text_tokens = tokenizer(texts).to(device)  # shape: [4, 77]

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)  # shape: [4, D]
        text_norms = text_features.norm(dim=-1, keepdim=True)
        if (text_norms < 1e-12).any():
            raise ZeroDivisionError("Detected zero-norm text feature vector!")
        text_features = text_features / text_norms  # shape: [4, D]

        # Cosine similarity via dot product of L2-normalized vectors
        similarities = (video_emb @ text_features.T).squeeze(0).cpu().tolist()  # list of 4 floats

    infer_time_s = time.perf_counter() - t_infer_start

    # Verify finite numbers
    for i, s in enumerate(similarities):
        if not math.isfinite(s):
            raise ValueError(f"Score for prompt {i} is not finite: {s}")

    # 5. Display results table
    print("-" * 70)
    print(f"{'ID':<12} | {'Cosine Score':<14} | {'Category':<15} | {'Prompt Text'}")
    print("-" * 70)
    results_list = []
    for cand, score in zip(PROMPT_CANDIDATES, similarities):
        print(f"{cand['id']:<12} | {score:+.6f}      | {cand['category']:<15} | \"{cand['text']}\"")
        results_list.append(
            {
                "id": cand["id"],
                "category": cand["category"],
                "text": cand["text"],
                "description": cand["description"],
                "cosine_similarity": round(float(score), 6),
            }
        )
    print("-" * 70)

    total_time_s = time.perf_counter() - t_start

    # 6. Structured JSON summary
    report: dict[str, Any] = {
        "status": "SUCCESS",
        "task_id": "W01-03",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_architecture": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.system(),
        },
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "open_clip": open_clip.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "model_spec": {
            "model_name": model_name,
            "pretrained": pretrained,
            "embedding_dimension": embed_dim,
            "weights_frozen": True,
            "train_fine_tune": False,
        },
        "video_metadata": video_meta,
        "performance": {
            "load_time_seconds": round(load_time_s, 3),
            "inference_time_seconds": round(infer_time_s, 3),
            "total_time_seconds": round(total_time_s, 3),
            "frames_per_second_infer": round(len(frames) / infer_time_s, 2),
        },
        "verification_checks": {
            "video_opened": True,
            "valid_frame_count": len(frames) == 8,
            "all_scores_finite": True,
            "l2_normalization_checked_non_zero": True,
            "no_grad_confirmed": True,
        },
        "matching_results": results_list,
    }

    # Ensure output directory exists and save JSON
    os.makedirs(output_dir, exist_ok=True)
    out_file = Path(output_dir) / "smoke_test.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[5/5] Saved structured result to: {out_file}")
    print(f"Total execution time: {total_time_s:.2f}s (Inference: {infer_time_s:.2f}s)")
    print("=" * 70)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCLIP Video-Text Matching Smoke Test")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Path to video file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model architecture")
    parser.add_argument("--pretrained", type=str, default=DEFAULT_PRETRAINED, help="Pretrained checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/w01", help="Output directory")
    args = parser.parse_args()

    run_smoke_test(
        video_path=args.video,
        model_name=args.model,
        pretrained=args.pretrained,
        output_dir=args.output_dir,
    )

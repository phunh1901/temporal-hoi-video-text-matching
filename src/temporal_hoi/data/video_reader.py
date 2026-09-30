"""Time-based decoding with explicit failure instead of fabricated evidence frames."""

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sample_frame_indices(total_frames: int, num_samples: int = 8) -> list[int]:
    if num_samples <= 0 or total_frames < num_samples:
        raise ValueError("Insufficient distinct frames or invalid sample count")
    return np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()


class VideoReader:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(self.video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            self.close()
            raise ValueError(f"Cannot open video: {self.video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(self.fps) or self.fps <= 0 or self.total_frames <= 0:
            self.close()
            raise ValueError("Invalid video metadata; FPS is never guessed")
        self.duration_s = self.total_frames / self.fps

    def get_metadata(self):
        return dict(
            path=str(self.video_path),
            width=self.width,
            height=self.height,
            fps=self.fps,
            total_frames=self.total_frames,
            duration_s=self.duration_s,
        )

    def read_interval_frames(self, start_s, end_s, num_frames=8, max_frame_side=None):
        """Read [start, end) with decoder times; reject missing/repeated frames.

        Selection assumes constant FPS. Variable-FPS input must be converted
        to CFR first; returned timestamps are checked against that assumption.
        """
        if max_frame_side is not None and max_frame_side <= 0:
            raise ValueError("max_frame_side must be positive")
        if not all(map(math.isfinite, (start_s, end_s))) or not (
            0 <= start_s < end_s <= self.duration_s + 1e-6
        ):
            raise ValueError("Interval must lie inside video with start < end")
        first = math.ceil(start_s * self.fps - 1e-8)
        stop = min(self.total_frames, math.ceil(end_s * self.fps - 1e-8))
        indices = [first + i for i in sample_frame_indices(stop - first, num_frames)]
        frames, timestamps = [], []
        for idx in indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                raise ValueError(f"Decode failed at frame {idx}; no replacement inserted")
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(t) or abs(t - idx / self.fps) > 0.51 / self.fps:
                raise ValueError("Inconsistent decoder timestamps; convert VFR to CFR first")
            if t < start_s - 1e-6 or t >= end_s + 1e-6 or (timestamps and t <= timestamps[-1]):
                raise ValueError("Decoded timestamps outside interval or not strictly increasing")
            if max_frame_side is not None and max(bgr.shape[:2]) > max_frame_side:
                scale = max_frame_side / max(bgr.shape[:2])
                bgr = cv2.resize(
                    bgr,
                    (max(1, round(bgr.shape[1] * scale)), max(1, round(bgr.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            timestamps.append(t)
        return frames, timestamps

    def sliding_windows(self, window_size_s=4.0, stride_s=1.0, num_frames=8):
        if not all(math.isfinite(x) and x > 0 for x in (window_size_s, stride_s)):
            raise ValueError("Window size and stride must be positive and finite")
        if self.duration_s < window_size_s:
            raise ValueError("Video shorter than a full window")
        # Drop partial final windows; never pad or repeat frames.
        for step in range(math.floor((self.duration_s - window_size_s) / stride_s + 1e-8) + 1):
            start = step * stride_s
            frames, times = self.read_interval_frames(start, start + window_size_s, num_frames)
            yield start, start + window_size_s, frames, times

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

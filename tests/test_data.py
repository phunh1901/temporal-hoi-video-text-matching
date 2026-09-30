"""Regression tests for real decoding and manifest integrity."""

import cv2
import numpy as np
import pandas as pd
import pytest

from temporal_hoi.data.dataset import TemporalHOIDataset
from temporal_hoi.data.video_reader import VideoReader


def make_video(path, fps=12, count=60):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 32))
    assert writer.isOpened()
    for i in range(count):
        writer.write(np.full((32, 32, 3), i % 255, np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize("fps", [12, 24, 30])
def test_real_timestamps_and_windows(tmp_path, fps):
    with VideoReader(make_video(tmp_path / "sample.avi", fps, fps * 6)) as reader:
        frames, times = reader.read_interval_frames(0.13, 4.13)
        assert len(frames) == 8
        assert all(0.13 <= t < 4.13 for t in times)
        assert all(b > a for a, b in zip(times, times[1:]))
        assert len(list(reader.sliding_windows())) == 3
        with pytest.raises(ValueError):
            list(reader.sliding_windows(stride_s=0))
        with pytest.raises(ValueError):
            reader.read_interval_frames(0, 7)


def test_corrupt_and_short_videos(tmp_path):
    broken = tmp_path / "broken.avi"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Cannot open"):
        VideoReader(broken)
    with VideoReader(make_video(tmp_path / "short.avi", 12, 3)) as reader:
        with pytest.raises(ValueError, match="Insufficient"):
            reader.read_interval_frames(0, 0.25)
        with pytest.raises(ValueError, match="shorter"):
            list(reader.sliding_windows())


def test_decode_failure_is_not_black_frame(tmp_path, monkeypatch):
    with VideoReader(make_video(tmp_path / "valid.avi")) as reader:
        real_cap = reader.cap

        class BrokenCapture:
            def set(self, *args):
                return True

            def read(self):
                return False, None

            def release(self):
                real_cap.release()

        monkeypatch.setattr(reader, "cap", BrokenCapture())
        with pytest.raises(ValueError, match="Decode failed"):
            reader.read_interval_frames(0, 4)


def manifests(tmp_path):
    clips = pd.DataFrame(
        [
            dict(clip_id="a", session_id="s", video_path="same.mp4", behavior_id="parking"),
            dict(clip_id="b", session_id="s", video_path="same.mp4", behavior_id="parking"),
        ]
    )
    texts = pd.DataFrame(
        [
            dict(
                text_id="t",
                behavior_id="parking",
                prompt_type="positive",
                prompt_text="a parked car",
            )
        ]
    )
    splits = pd.DataFrame(
        [
            dict(clip_id="a", session_id="s", split="dev"),
            dict(clip_id="b", session_id="s", split="test"),
        ]
    )
    paths = [tmp_path / f"{name}.csv" for name in ["clips", "texts", "splits"]]
    for frame, path in zip([clips, texts, splits], paths):
        frame.to_csv(path, index=False)
    return paths, splits


def test_split_leakage_and_missing_split(tmp_path):
    paths, splits = manifests(tmp_path)
    with pytest.raises(ValueError, match="leakage"):
        TemporalHOIDataset(*paths)
    splits.iloc[:1].to_csv(paths[2], index=False)
    with pytest.raises(ValueError, match="exactly one split"):
        TemporalHOIDataset(*paths)


def test_prompt_templates_are_not_clip_labels(tmp_path):
    paths, splits = manifests(tmp_path)
    splits["split"] = "dev"
    splits.to_csv(paths[2], index=False)
    clips = pd.read_csv(paths[0])
    clips["video_path"] = str(make_video(tmp_path / "clip.avi"))
    clips["start_s"], clips["end_s"] = 0, 4
    clips["rule_id"], clips["has_violation"], clips["zone_polygon"] = "r", 0, "[]"
    clips.to_csv(paths[0], index=False)
    dataset = TemporalHOIDataset(*paths, split="dev")
    item = dataset[0]
    assert item["positive_prompts"] == []
    assert item["negative_prompts"] == []
    assert not item["matching_labels_available"]
    relevance = tmp_path / "relevance.csv"
    pd.DataFrame([dict(clip_id="a", text_id="t", is_match=0)]).to_csv(relevance, index=False)
    labeled = TemporalHOIDataset(*paths, relevance_csv=relevance)[0]
    assert not labeled["positive_prompts"]
    assert labeled["negative_prompts"][0]["text_id"] == "t"


def test_unknown_violation_is_not_negative(tmp_path):
    paths, splits = manifests(tmp_path)
    splits["split"] = "dev"
    splits.to_csv(paths[2], index=False)
    clips = pd.read_csv(paths[0])
    clips["video_path"] = str(make_video(tmp_path / "unknown.avi"))
    clips["start_s"], clips["end_s"] = 0, 4
    clips["rule_id"], clips["has_violation"], clips["zone_polygon"] = "r", float("nan"), "[]"
    clips.to_csv(paths[0], index=False)
    assert TemporalHOIDataset(*paths)[0]["has_violation"] is None


def test_resize_limits_sample_memory(tmp_path):
    with VideoReader(make_video(tmp_path / "resize.avi")) as reader:
        frames, times = reader.read_interval_frames(0, 4, max_frame_side=16)
        assert all(max(frame.size) <= 16 for frame in frames)
        assert len(times) == 8
        with pytest.raises(ValueError, match="max_frame_side"):
            reader.read_interval_frames(0, 4, max_frame_side=0)


def test_lightweight_import_does_not_load_torch():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import temporal_hoi.data; import sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

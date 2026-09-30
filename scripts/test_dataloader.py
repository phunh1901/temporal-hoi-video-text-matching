"""Development-only W02 decoding benchmark. This does not certify dataset readiness."""

import json
import time
import tracemalloc
from pathlib import Path

from PIL import Image, ImageDraw

from temporal_hoi.data.dataset import TemporalHOIDataset
from temporal_hoi.data.video_reader import VideoReader


def create_frame_strip(frames, timestamps):
    tiles = []
    for frame in frames:
        tiles.append(frame.resize((int(frame.width * 120 / frame.height), 120)))
    strip = Image.new("RGB", (sum(t.width for t in tiles), 150))
    draw = ImageDraw.Draw(strip)
    x = 0
    for tile, timestamp in zip(tiles, timestamps):
        strip.paste(tile, (x, 0))
        draw.text((x + 4, 126), f"{timestamp:.3f}s", fill="white")
        x += tile.width
    return strip


def main():
    output = Path("outputs/w02")
    previews = output / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    dataset = TemporalHOIDataset(split="dev")
    config = json.loads(Path("configs/w02_lightweight.json").read_text(encoding="utf-8"))
    records = []
    tracemalloc.start()
    started = time.perf_counter()
    for _, row in dataset.clips.iterrows():
        start = time.perf_counter()
        with VideoReader(row.video_path) as reader:
            # First full 4-second window inside the annotated interval.
            end = min(float(row.end_s), float(row.start_s) + 4)
            frames, times = reader.read_interval_frames(
                float(row.start_s), end, max_frame_side=config["max_frame_side"]
            )
        strip_path = previews / f"{row.clip_id}_dev_window.jpg"
        create_frame_strip(frames, times).save(strip_path)
        assert len(frames) == 8 and all(b > a for a, b in zip(times, times[1:]))
        records.append(
            dict(
                clip_id=row.clip_id,
                split="dev",
                timestamps=times,
                num_frames=len(frames),
                decoded_rgb_bytes=sum(frame.width * frame.height * 3 for frame in frames),
                load_time_seconds=time.perf_counter() - start,
                preview_image=str(strip_path),
            )
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Windows process peak working set includes native decoding allocations.
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in [
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            ]
        ]

    peak_rss = None
    if hasattr(ctypes, "WinDLL"):
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            peak_rss = counters.PeakWorkingSetSize
    report = dict(
        task_id="W02-03",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        status="PASS" if len(records) == config["expected_dev_clips"] else "PARTIAL",
        profile=config["profile"],
        required_dev_clips=config["expected_dev_clips"],
        max_frame_side=config["max_frame_side"],
        batch_size=1,
        num_workers=0,
        total_clips_evaluated=len(records),
        total_time_seconds=time.perf_counter() - started,
        peak_python_traced_bytes=peak,
        peak_process_working_set_bytes=peak_rss,
        memory_scope="Process peak includes imports; tracemalloc excludes native buffers",
        window_size_s=4,
        stride_s=1,
        benchmark_scope="first window per dev clip",
        all_timestamps_strictly_increasing=True,
        clips=records,
    )
    (output / "dataloader_benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=2))


if __name__ == "__main__":
    main()

"""Script to download lightweight, targeted CCTV sample videos for Temporal HOI project.

Downloads technical sample footage; action/violation labels require visual review:
1. Rule 1 (Main): Vehicle parking in restricted zone (parking.mp4)
2. Rule 2 (Main): Highway traffic / road-debris test scene (trash.mp4), NOT verified person littering
3. Rule 3 (Extension): Riding bicycle in pedestrian walkway (bicycle_pedestrian.mp4)
4. Normal baseline: Pedestrians walking normally, no violations (normal_surveillance.mp4)

Technical samples only; filenames are not verified action, violation, or license labels.
"""

import sys
import urllib.request
from pathlib import Path

import cv2
from tqdm import tqdm

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR = Path("data/raw")

VIDEO_MANIFEST = [
    {
        "id": "clip_01_parking",
        "filename": "parking_violation.mp4",
        "url": "https://raw.githubusercontent.com/M4D-AI/Highway-Incident-Detection/main/data/parking.mp4",
        "rule": "Luật 1 (Chính): Đỗ xe trong vùng cấm",
        "action": "Highway traffic and stopping candidate; violation unverified",
        "source": "Highway Incident Detection CCTV Dataset",
    },
    {
        "id": "clip_02_trash",
        "filename": "trash_dumping.mp4",
        "url": "https://raw.githubusercontent.com/M4D-AI/Highway-Incident-Detection/main/data/trash.mp4",
        "rule": "Mẫu cao tốc: không xác minh người đổ rác",
        "action": "Vehicles moving on a highway; person littering not verified",
        "source": "Highway Incident Detection CCTV Dataset",
    },
    {
        "id": "clip_03_bicycle",
        "filename": "bicycle_pedestrian.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
        "rule": "Luật 3 (Mở rộng): Đi xe đạp trong khu đi bộ",
        "action": "Person riding bicycle in pedestrian area",
        "source": "Intel sample-videos (repository CC-BY-4.0; retain attribution)",
    },
    {
        "id": "clip_04_normal",
        "filename": "normal_surveillance.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
        "rule": "Đối chứng bình thường: Không có vi phạm",
        "action": "People walking through an indoor room",
        "source": "Intel sample-videos (repository CC-BY-4.0; retain attribution)",
    },
    {
        "id": "clip_05_apartment_parking",
        "filename": "apartment_parking.mp4",
        "url": "https://raw.githubusercontent.com/bhupender0415/CarParkingDetection/master/carPark.mp4",
        "rule": "Mẫu bãi xe: nhãn vùng cấm chưa xác minh",
        "action": "Cars in marked parking spaces; residential context unverified",
        "source": "Car Parking Detection CCTV Dataset",
    },
]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """Download a file with progress bar and retry handling."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    for attempt in range(1, max_retries + 1):
        try:
            with DownloadProgressBar(
                unit="B", unit_scale=True, miniters=1, desc=output_path.name
            ) as t:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    total_size = int(resp.headers.get("Content-Length", 0))
                    t.total = total_size
                    with open(output_path, "wb") as f:
                        while chunk := resp.read(64 * 1024):
                            f.write(chunk)
                            t.update(len(chunk))
            return True
        except Exception as e:
            print(f"  [LẦN THỬ {attempt}/{max_retries}] Lỗi khi tải {output_path.name}: {e}")
            if output_path.exists():
                output_path.unlink()
            if attempt == max_retries:
                return False
            import time

            time.sleep(2)
    return False


def get_video_info(video_path: Path) -> dict:
    """Extract metadata (width, height, fps, frame count, duration) from video using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"valid": False}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "valid": True,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 1),
        "frames": frame_count,
        "duration_s": round(duration, 1),
        "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
    }


def main():
    print("=" * 70)
    print("TEMPORAL HOI - TẢI VIDEO MẪU KỸ THUẬT (W02-01)")
    print(f"Mục tiêu lưu trữ: {RAW_DIR.resolve()}")
    print("=" * 70)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    total_size_mb = 0.0

    for item in VIDEO_MANIFEST:
        dest_path = RAW_DIR / item["filename"]
        print(f"\n[*] Đang xử lý: {item['filename']}")
        print(f"    - Phân loại: {item['rule']}")
        print(f"    - Nguồn: {item['source']}")

        if dest_path.exists() and dest_path.stat().st_size > 0:
            print("    -> File đã tồn tại sẵn, bỏ qua tải lại.")
            success = True
        else:
            print(f"    -> Đang tải từ: {item['url']}")
            success = download_file(item["url"], dest_path)

        if success and dest_path.exists():
            info = get_video_info(dest_path)
            total_size_mb += info.get("size_mb", 0.0)
            results.append({**item, **info, "path": str(dest_path)})
        else:
            results.append({**item, "valid": False})

    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP VIDEO MẪU ĐÃ TẢI")
    print("=" * 70)
    print(
        f"{'File':<26} | {'Độ phân giải':<12} | {'FPS':<6} | {'Thời lượng':<10} | {'Dung lượng':<10}"
    )
    print("-" * 70)
    for r in results:
        if r.get("valid"):
            print(
                f"{r['filename']:<26} | {r['resolution']:<12} | {r['fps']:<6} | "
                f"{r['duration_s']}s{'':<6} | {r['size_mb']} MB"
            )
        else:
            print(f"{r['filename']:<26} | TẢI THẤT BẠI")
    print("-" * 70)
    print(f"TỔNG DUNG LƯỢNG: {total_size_mb:.2f} MB (Rất gọn nhẹ, không tốn tài nguyên ổ cứng)")
    print("=" * 70)


if __name__ == "__main__":
    main()


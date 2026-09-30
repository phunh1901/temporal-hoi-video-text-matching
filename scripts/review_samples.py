"""Small contact sheets and source hashes; never copy source videos."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    output = Path("outputs/w02/review")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(Path("data/raw").glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sheet = Image.new("RGB", (960, 720))
        draw = ImageDraw.Draw(sheet)
        indices = np.linspace(0, count - 1, 16, dtype=int)
        for i, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot read {path} frame {index}")
            tile = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tile.thumbnail((240, 150))
            x, y = (i % 4) * 240, (i // 4) * 180
            sheet.paste(tile, (x, y))
            draw.text((x + 3, y + 153), f"{index / fps:.2f}s", fill="white")
        records.append(
            dict(
                path=str(path),
                bytes=path.stat().st_size,
                sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                fps=fps,
                frames=count,
                duration_s=count / fps,
            )
        )
        cap.release()
        sheet.save(output / f"{path.stem}.jpg", quality=85)
    (output / "sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

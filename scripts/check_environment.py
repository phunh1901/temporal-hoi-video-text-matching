"""Check the development stack offline without downloading pretrained weights."""

import importlib
import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

# Importing the demo library must not send analytics during the check.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)

PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "open_clip": "open-clip-torch",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "pydantic": "pydantic",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "gradio": "gradio",
    "temporal_hoi": "temporal-hoi-video-text-matching",
}


def main() -> None:
    modules = {module: importlib.import_module(module) for module in PACKAGES}
    torch, np, cv2 = (modules[name] for name in ("torch", "numpy", "cv2"))

    # Verify tensor/NumPy interop, gradients and the torchvision compiled extension.
    layer = torch.nn.Linear(4, 2)
    sample = torch.from_numpy(np.ones((2, 4), dtype=np.float32))
    loss = layer(sample).square().mean()
    loss.backward()
    if layer.weight.grad is None or not torch.isfinite(layer.weight.grad).all():
        raise RuntimeError("PyTorch autograd check failed")
    keep = modules["torchvision"].ops.nms(
        torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]]),
        torch.tensor([0.9, 0.8]),
        0.5,
    )
    if keep.tolist() != [0]:
        raise RuntimeError("torchvision NMS check failed")

    # Bundled BPE tokenizer; no model download or claim of pretrained inference.
    tokens = modules["open_clip"].get_tokenizer("ViT-B-32")(["a person holding a bottle"])
    if tokens.shape != (1, 77):
        raise RuntimeError("CLIP tokenizer check failed")

    with tempfile.TemporaryDirectory(prefix="temporal-hoi-smoke-") as temporary:
        video = str(Path(temporary) / "sample.avi")
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 64))
        try:
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not create the sample video")
            for value in (0, 60, 120):
                writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        reader = cv2.VideoCapture(video)
        try:
            frames = 0
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                if frame.shape != (64, 64, 3):
                    raise RuntimeError("Video frame shape mismatch")
                frames += 1
        finally:
            reader.release()
        if frames != 3:
            raise RuntimeError(f"Video decode expected 3 frames, got {frames}")

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.system(),
                "packages": {package: version(package) for package in PACKAGES.values()},
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda_build": torch.version.cuda,
                "checks": [
                    "imports",
                    "autograd",
                    "torchvision_nms",
                    "clip_tokenizer",
                    "video_roundtrip",
                ],
                "pretrained_weights_downloaded": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Frozen CLIP oracle HOI inference on an identified, bounded VidHOI train probe.

Every annotated person proposes a directed pair with every distinct available
entity. Available entities come from oracle annotations, not a detector or an
exhaustive scene inventory. All predicate prompts are scored for every pair;
ground-truth predicates are used only by the unchanged frame-level evaluator.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from zipfile import ZipFile, is_zipfile

import numpy as np
import torch
from PIL import Image, ImageDraw

from temporal_hoi.data.vidhoi import ClassMapping, HOIPair, VidHOIFrame, identity_hash
from temporal_hoi.evaluation.hoi import evaluate_frame_hoi

CONFIG_SCHEMA = "vidhoi_static_config_v1"
MANIFEST_SCHEMA = "vidhoi_static_oracle_probe_v1"
MAPPING_SHA256 = "a74c9c6c0e1efa116eb368935aded8b5a124c3d758376391924db8e2a58368d7"
PROMPT_VERSION = "vidhoi_all_predicates_english_v1"
PREDICATE_PHRASES = {
    "lean_on": "leaning on", "watch": "watching", "above": "above",
    "next_to": "next to", "behind": "behind", "away": "moving away from",
    "towards": "moving towards", "in_front_of": "in front of", "hit": "hitting",
    "hold": "holding", "wave": "waving", "pat": "patting", "carry": "carrying",
    "point_to": "pointing to", "touch": "touching", "play(instrument)": "playing",
    "release": "releasing", "ride": "riding", "grab": "grabbing", "lift": "lifting",
    "use": "using", "press": "pressing", "inside": "inside", "caress": "caressing",
    "pull": "pulling", "get_on": "getting on", "cut": "cutting", "hug": "hugging",
    "bite": "biting", "open": "opening", "close": "closing", "throw": "throwing",
    "kick": "kicking", "drive": "driving", "get_off": "getting off", "push": "pushing",
    "wave_hand_to": "waving a hand to", "feed": "feeding", "chase": "chasing",
    "kiss": "kissing", "speak_to": "speaking to", "beneath": "beneath",
    "smell": "smelling", "clean": "cleaning", "lick": "licking", "squeeze": "squeezing",
    "shake_hand_with": "shaking hands with", "knock": "knocking on",
    "hold_hand_of": "holding the hand of", "shout_at": "shouting at",
}
OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


class Encoder(Protocol):
    metadata: dict[str, Any]

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class OracleEntity:
    track_id: str
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class OracleFrame:
    video_id: str
    source_frame_id: int
    frame_id: int
    timestamp_s: float | None
    image_path: Path
    image_sha256: str
    width: int
    height: int
    entities: tuple[OracleEntity, ...]
    ground_truth: tuple[HOIPair, ...]

    @property
    def key(self) -> str:
        return f"{self.video_id}_{self.frame_id:06d}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric data")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _hash(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} requires a lowercase SHA256")
    return value


def checked_file(path: str | Path, expected: str, name: str) -> Path:
    target = Path(path)
    _hash(expected, name)
    if not target.is_file():
        raise ValueError(f"Missing {name} file: {target}")
    if file_sha256(target) != expected:
        raise ValueError(f"{name} SHA256 mismatch")
    return target


def _track(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("track_id must be a nonempty string")
    return value


def proposal_pairs(frame: OracleFrame) -> list[tuple[OracleEntity, OracleEntity]]:
    """All person -> distinct available entity pairs, never GT-positive-only pairs."""
    entities = sorted(frame.entities, key=lambda entity: entity.track_id)
    return [(subject, obj) for subject in entities if subject.class_id == 0
            for obj in entities if subject.track_id != obj.track_id]


def build_prompts(mapping: ClassMapping, object_class: int) -> list[str]:
    if object_class not in mapping.objects:
        raise ValueError("Unknown object class for prompt construction")
    obj = mapping.objects[object_class].replace("_", " ").replace("/", " or ")
    prompts = []
    for predicate in range(len(mapping.predicates)):
        name = mapping.predicates[predicate]
        if name not in PREDICATE_PHRASES:
            raise ValueError(f"No versioned predicate prompt for {name}")
        prompts.append(f"A photo of a person {PREDICATE_PHRASES[name]} a {obj}.")
    return prompts


def load_manifest(path: Path, mapping: ClassMapping, *, max_frames: int,
                  max_pairs_per_frame: int) -> tuple[list[OracleFrame], dict]:
    """Validate all image bytes/coordinates and keep even zero-proposal frames."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in (("schema_version", MANIFEST_SCHEMA), ("split", "train"),
                          ("purpose", "train_probe"), ("box_source", "oracle"),
                          ("object_label_source", "oracle")):
        if data.get(key) != expected:
            raise ValueError(f"Manifest {key} must be {expected}")
    rows = data.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest needs a nonempty frames list")
    if len(rows) > max_frames:
        raise ValueError("Manifest exceeds max_frames; no frames were silently dropped")
    frames, keys, source_keys = [], set(), set()
    for row in rows:
        video = row.get("video_id")
        if not isinstance(video, str) or not video.strip():
            raise ValueError("video_id must be nonempty")
        source_id = _integer(row.get("source_frame_id"), "source_frame_id")
        frame_id = _integer(row.get("frame_id"), "frame_id")
        width = _integer(row.get("width"), "width", 1)
        height = _integer(row.get("height"), "height", 1)
        if "timestamp_s" not in row:
            raise ValueError("timestamp_s must be explicit; null means unknown")
        timestamp = row["timestamp_s"]
        if timestamp is not None and _finite(timestamp, "timestamp_s") < 0:
            raise ValueError("timestamp_s cannot be negative")
        raw_path = row.get("image_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("image_path must be a relative path")
        image_path = (path.parent / raw_path).resolve()
        if Path(raw_path).is_absolute() or not image_path.is_relative_to(path.parent.resolve()):
            raise ValueError("image_path escapes the manifest directory")
        checked_file(image_path, row.get("image_sha256"), "frame image")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError("Image dimensions disagree with manifest width/height")
            image.verify()
        entities = {}
        if not isinstance(row.get("entities"), list):
            raise ValueError("entities must be an explicit list")
        for entity in row["entities"]:
            track = _track(entity.get("track_id"))
            class_id = _integer(entity.get("class_id"), "entity class_id")
            if class_id not in mapping.objects or track in entities:
                raise ValueError("Unknown entity class or duplicate track_id")
            box = entity.get("bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("bbox_xyxy requires four coordinates")
            box = tuple(_finite(value, "box coordinate") for value in box)
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError("Oracle bbox_xyxy must have positive area inside original image")
            entities[track] = OracleEntity(track, class_id, box)
        if not isinstance(row.get("ground_truth"), list):
            raise ValueError("ground_truth must be an explicit list")
        labels: dict[tuple[str, str], set[int]] = {}
        for relation in row["ground_truth"]:
            sub_id = _track(relation.get("subject_track_id"))
            obj_id = _track(relation.get("object_track_id"))
            if sub_id == obj_id or sub_id not in entities or obj_id not in entities:
                raise ValueError("Ground-truth relation uses an absent or identical entity")
            if entities[sub_id].class_id != 0:
                raise ValueError("Ground-truth subject must be person")
            predicates = relation.get("predicate_ids")
            if not isinstance(predicates, list) or not predicates:
                raise ValueError("Ground-truth predicate_ids must be a nonempty list")
            for predicate in predicates:
                _integer(predicate, "predicate ID")
                if predicate not in mapping.predicates:
                    raise ValueError("Unknown predicate ID")
            labels.setdefault((sub_id, obj_id), set()).update(predicates)
        truth = tuple(HOIPair(0, entities[obj].class_id, entities[sub].bbox_xyxy,
                              entities[obj].bbox_xyxy,
                              tuple(float(index in predicates)
                                    for index in range(len(mapping.predicates))),
                              subject_track_id=sub, object_track_id=obj)
                      for (sub, obj), predicates in sorted(labels.items()))
        frame = OracleFrame(video, source_id, frame_id, timestamp, image_path,
                            row["image_sha256"], width, height, tuple(entities.values()), truth)
        source_key = (video, source_id)
        if frame.key in keys or source_key in source_keys:
            raise ValueError("Duplicate frame_id or source_frame_id identity")
        keys.add(frame.key)
        source_keys.add(source_key)
        if len(proposal_pairs(frame)) > max_pairs_per_frame:
            raise ValueError("Frame exceeds max_pairs_per_frame; no pairs were silently dropped")
        frames.append(frame)
    return frames, data


def _normalized(values: Any, rows: int, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != rows or values.shape[1] < 1:
        raise ValueError(f"{label} encoder features require shape [batch,dimension]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError(f"{label} encoder produced nonfinite or zero features")
    return values / norms


def _union(subject: OracleEntity, obj: OracleEntity) -> tuple[int, int, int, int]:
    a, b = subject.bbox_xyxy, obj.bbox_xyxy
    return (math.floor(min(a[0], b[0])), math.floor(min(a[1], b[1])),
            math.ceil(max(a[2], b[2])), math.ceil(max(a[3], b[3])))


def infer_frame(frame: OracleFrame, encoder: Encoder, text_features: dict[int, np.ndarray],
                *, batch_size: int, logit_scale: float) -> VidHOIFrame:
    pairs = proposal_pairs(frame)
    predictions = []
    with Image.open(frame.image_path) as original:
        image = original.convert("RGB")
        for offset in range(0, len(pairs), batch_size):
            batch = pairs[offset:offset + batch_size]
            crops = [image.crop(_union(subject, obj)) for subject, obj in batch]
            features = _normalized(encoder.encode_images(crops), len(batch), "image")
            for feature, (subject, obj) in zip(features, batch, strict=True):
                text = text_features[obj.class_id]
                if text.shape[1] != feature.shape[0]:
                    raise ValueError("Image and text encoder dimensions disagree")
                cosine = np.clip(text @ feature, -1.0, 1.0).astype(np.float64)
                positive_scores = np.exp(logit_scale * cosine)
                predictions.append(HOIPair(
                    0, obj.class_id, subject.bbox_xyxy, obj.bbox_xyxy,
                    tuple(float(value) for value in positive_scores),
                    subject_track_id=subject.track_id, object_track_id=obj.track_id))
    return VidHOIFrame(frame.video_id, frame.frame_id, "train", "oracle",
                       "xyxy_pixels_original_frame", frame.ground_truth,
                       tuple(predictions), frame.timestamp_s)


def safe_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    """Plain tensor state only; never auto-dispatch TorchScript or unrestricted pickle."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    elif path.suffix in {".pt", ".pth", ".bin"}:
        if is_zipfile(path):
            with ZipFile(path) as archive:
                if any("/code/" in name or name.endswith("constants.pkl")
                       for name in archive.namelist()):
                    raise ValueError("TorchScript checkpoints are not accepted; use tensor state")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    else:
        raise ValueError("Checkpoint must be safetensors or a plain tensor .pt/.pth/.bin")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                   for key, value in state.items())):
        raise ValueError("Checkpoint must contain only named tensors")
    return state


class OriginalCLIPEncoder:
    """Strict original OpenAI ViT-B/32 weights, QuickGELU, tokenizer and preprocessing."""

    def __init__(self, checkpoint: Path, *, device: str):
        import open_clip

        state = safe_checkpoint_state(checkpoint)
        print("CLIP_ARCHITECTURE_INIT: creating the architecture before strict local weight loading",
              flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=None, device="cpu", precision="fp32",
            force_quick_gelu=True, force_image_size=224,
            image_mean=OPENAI_MEAN, image_std=OPENAI_STD,
            image_interpolation="bicubic", image_resize_mode="shortest")
        self.model.load_state_dict(state, strict=True)
        print(f"CLIP_LOCAL_WEIGHTS_LOADED: strict=True tensors={len(state)} "
              f"checkpoint={checkpoint.name}; pretrained tensors replaced initialization",
              flush=True)
        del state
        self.device = device
        self.model.eval().requires_grad_(False).to(device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32", context_length=77)
        self.metadata = {
            "test_fixture": False, "architecture": "OpenAI_CLIP_ViT-B32",
            "activation": "QuickGELU", "context_length": 77, "image_size": 224,
            "mean": OPENAI_MEAN, "std": OPENAI_STD, "resize": "shortest_center_crop",
            "interpolation": "bicubic", "precision": "fp32", "device": device,
            "frozen": True, "training": False, "strict_state_dict_load": True,
            "checkpoint_format": checkpoint.suffix,
        }

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        return self.model.encode_image(batch).float().cpu().numpy()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        return self.model.encode_text(tokens).float().cpu().numpy()


def resolve_config(config: dict, config_path: Path) -> dict:
    result = dict(config)
    for name in ("manifest", "mapping", "checkpoint"):
        if name not in result or not isinstance(result[name], str) or not result[name]:
            raise ValueError(f"Configuration requires {name} path")
        result[name] = str((config_path.parent / result[name]).resolve())
    return result


def preflight(config: dict, *, test_fixture: bool = False) -> tuple[ClassMapping, list, dict]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"Configuration schema_version must be {CONFIG_SCHEMA}")
    for name, default in (("max_frames", 64), ("max_pairs_per_frame", 128),
                          ("batch_size", 8), ("threads", 2)):
        _integer(config.get(name, default), name, 1)
    _integer(config.get("preview_frames", 4), "preview_frames")
    scale = _finite(config.get("logit_scale", 1.0), "logit_scale")
    if not 0 < scale <= 100:
        raise ValueError("logit_scale must be in (0,100] for bounded positive score transform")
    if config.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    mapping_path = checked_file(config["mapping"], config.get("mapping_sha256"), "mapping")
    mapping = ClassMapping.from_json(mapping_path)
    if not test_fixture and (file_sha256(mapping_path) != MAPPING_SHA256
                             or len(mapping.objects) != 78 or len(mapping.predicates) != 50):
        raise ValueError("Real VidHOI run requires pinned canonical 78/50 mapping")
    manifest_path = checked_file(config["manifest"], config.get("manifest_sha256"), "manifest")
    frames, manifest = load_manifest(
        manifest_path, mapping, max_frames=config.get("max_frames", 64),
        max_pairs_per_frame=config.get("max_pairs_per_frame", 128))
    if manifest.get("mapping_sha256") != config["mapping_sha256"]:
        raise ValueError("Manifest mapping_sha256 disagrees with loaded mapping")
    checkpoint = checked_file(config["checkpoint"], config.get("checkpoint_sha256"), "checkpoint")
    if checkpoint.stat().st_size > config.get("max_checkpoint_bytes", 1024**3):
        raise ValueError("Checkpoint exceeds max_checkpoint_bytes")
    if not isinstance(config.get("checkpoint_provenance"), dict) or not config["checkpoint_provenance"]:
        raise ValueError("checkpoint_provenance must be a nonempty object")
    if config.get("device", "cpu") == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no device fallback")
    for class_id in {obj.class_id for frame in frames for _, obj in proposal_pairs(frame)}:
        build_prompts(mapping, class_id)
    return mapping, frames, manifest


def _peak_rss_bytes() -> int:
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (key, ctypes.c_size_t) for key in
                ("peak", "rss", "quota_peak", "quota", "page_peak", "page", "swap", "swap_peak")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters),
                                               ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                                         counters.cb):
            raise OSError("Cannot measure process peak memory")
        return int(counters.peak)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _preview(path: Path, source: OracleFrame, inferred: VidHOIFrame,
             mapping: ClassMapping) -> None:
    with Image.open(source.image_path) as original:
        image = original.convert("RGB")
        draw = ImageDraw.Draw(image)
        if inferred.predictions:
            pair = inferred.predictions[0]  # Deterministic pair; never selected by GT correctness.
            predicate = int(np.argmax(pair.predicate_values))
            draw.rectangle(pair.subject_box, outline="red", width=2)
            draw.rectangle(pair.object_box, outline="blue", width=2)
            draw.text((4, 4), f"oracle boxes; predicted: {mapping.predicates[predicate]} "
                      f"/ {mapping.objects[pair.object_class]}", fill="yellow")
        else:
            draw.text((4, 4), "oracle frame: no available person-object pair", fill="yellow")
        image.save(path)


def run_static_probe(config: dict, output_dir: str | Path, *, encoder: Encoder | None = None) -> dict:
    """Run actual inference unless an explicitly marked test encoder is injected."""
    started = time.perf_counter()
    fixture = encoder is not None
    if fixture and encoder.metadata.get("test_fixture") is not True:
        raise ValueError("Injected encoder must be explicitly marked test_fixture=True")
    mapping, frames, _ = preflight(config, test_fixture=fixture)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "run_config.json", config)
    torch.set_num_threads(config.get("threads", 2))
    device = config.get("device", "cpu")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    if encoder is None:
        encoder = OriginalCLIPEncoder(Path(config["checkpoint"]), device=device)
    load_seconds = time.perf_counter() - load_started
    batch_size = config.get("batch_size", 8)
    text_started = time.perf_counter()
    text_features, prompts_by_class = {}, {}
    object_classes = sorted({obj.class_id for frame in frames for _, obj in proposal_pairs(frame)})
    for class_id in object_classes:
        prompts = build_prompts(mapping, class_id)
        prompts_by_class[class_id] = prompts
        chunks = [_normalized(encoder.encode_texts(prompts[offset:offset + batch_size]),
                              len(prompts[offset:offset + batch_size]), "text")
                  for offset in range(0, len(prompts), batch_size)]
        text_features[class_id] = np.concatenate(chunks, axis=0)
    text_seconds = time.perf_counter() - text_started
    inference_started = time.perf_counter()
    inferred = []
    frame_stats = []
    for index, frame in enumerate(frames):
        frame_started = time.perf_counter()
        prediction = infer_frame(frame, encoder, text_features, batch_size=batch_size,
                                 logit_scale=config.get("logit_scale", 1.0))
        if device == "cuda":
            torch.cuda.synchronize()
        inferred.append(prediction)
        frame_stats.append({"frame_key": frame.key, "source_frame_id": frame.source_frame_id,
                            "image_sha256": frame.image_sha256,
                            "available_entities": len(frame.entities),
                            "proposal_pairs": len(prediction.predictions),
                            "predicate_scores": len(prediction.predictions) * len(mapping.predicates),
                            "decode_crop_encode_score_seconds": time.perf_counter() - frame_started})
        if index < config.get("preview_frames", 4):
            _preview(output / f"preview_{index:04d}.jpg", frame, prediction, mapping)
    inference_seconds = time.perf_counter() - inference_started
    evaluation_started = time.perf_counter()
    metrics = evaluate_frame_hoi(inferred)
    evaluation_seconds = time.perf_counter() - evaluation_started
    versions = {}
    for package in ("torch", "torchvision", "open-clip-torch", "numpy", "pillow", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    receipt = {
        "schema_version": "vidhoi_static_receipt_v1", "status": "completed",
        "level": "synthetic_encoder_fixture" if fixture else "oracle_train_probe_inference",
        "model_inference_run": not fixture, "model_training_run": False,
        "full_benchmark_reproduction_claim": False, "zero_shot_benchmark_claim": False,
        "benchmark_comparison_eligible": False, "official_final_split_accessed": False,
        "split": "train", "box_source": "oracle", "object_label_source": "oracle",
        "coordinate_system": "xyxy_pixels_original_frame",
        "proposal_source": "all_person_to_distinct_available_oracle_entities",
        "entity_coverage": "annotation_entities_only_not_exhaustive_scene_entities",
        "predicate_candidate_source": "all_mapping_predicates_independent_of_ground_truth_actions",
        "score_transform": "exp(fixed_logit_scale * normalized_image_text_cosine)",
        "score_is_probability": False, "proposal_subject_and_object_scores": 1.0,
        "logit_scale": config.get("logit_scale", 1.0), "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(json.dumps(prompts_by_class, sort_keys=True,
                                                     ensure_ascii=False).encode()).hexdigest(),
        "frame_ids_sha256": identity_hash([frame.key for frame in frames]),
        "input_hashes": {key: config[f"{key}_sha256"]
                         for key in ("manifest", "mapping", "checkpoint")},
        "checkpoint_provenance": config["checkpoint_provenance"],
        "checkpoint_provenance_verification": "caller_assertion_and_local_checkpoint_hash",
        "encoder": encoder.metadata, "python": platform.python_version(),
        "platform": platform.platform(), "versions": versions,
        "resources": {"model_load_seconds": load_seconds, "text_encode_seconds": text_seconds,
                      "frame_inference_seconds": inference_seconds,
                      "evaluation_seconds": evaluation_seconds,
                      "total_seconds": time.perf_counter() - started,
                      "process_peak_rss_bytes": _peak_rss_bytes(),
                      "process_peak_scope": "whole_process_lifetime_not_isolated_run_delta",
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                      if device == "cuda" else None},
        "num_frames": len(frames), "dropped_frames": 0, "dropped_proposal_pairs": 0,
        "frames_without_proposals": sum(not frame.predictions for frame in inferred),
        "num_proposal_pairs": sum(len(frame.predictions) for frame in inferred),
        "frame_statistics": frame_stats, "metrics": metrics,
    }
    _write_json(output / "prompts.json", prompts_by_class)
    _write_json(output / "predictions.json", [asdict(frame) for frame in inferred])
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "receipt.json", receipt)
    return receipt

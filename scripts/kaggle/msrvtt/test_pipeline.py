"""CPU integration tests: real pinned encoder/tokenizer, decoded videos and outputs."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import data_pipeline as dp
import numpy as np
import torch
import workflow
from PIL import Image

VENDOR = Path(__file__).parent / "vendor"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        cls.source = dp.import_file("test_pinned_clip", VENDOR / "module_clip.py")
        cls.tokenizer_source = dp.import_file(
            "test_pinned_tokenizer", VENDOR / "tokenization_clip.py"
        )
        cls.tokenizer = cls.tokenizer_source.SimpleTokenizer(
            str(VENDOR / "bpe_simple_vocab_16e6.txt.gz")
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def model(self):
        torch.manual_seed(2)
        return self.source.CLIP(32, 32, 1, 64, 16, 77, 49408, 64, 1, 1).float().eval()

    def video(self, vid, seconds=4):
        p = self.root / f"{vid}.avi"
        writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
        if not writer.isOpened():
            self.fail("OpenCV MJPG writer unavailable: integration test requires a real decoder.")
        for index in range(seconds * 5):
            a = np.zeros((48, 64, 3), dtype=np.uint8)
            a[..., 0] = (index * 9) % 255
            a[..., 1] = np.arange(64)[None, :] * 3
            a[..., 2] = np.arange(48)[:, None] * 4
            writer.write(a)
        writer.release()
        return p

    def test_preprocess_matches_original_torchvision(self):
        from torchvision import transforms

        old = transforms.Compose(
            [
                transforms.Resize(32, interpolation=Image.Resampling.BICUBIC),
                transforms.CenterCrop(32),
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )
        rng = np.random.default_rng(1)
        for shape in ((81, 48), (48, 81), (33, 32), (32, 32)):
            im = Image.fromarray(rng.integers(0, 255, (*shape, 3), dtype=np.uint8))
            torch.testing.assert_close(dp.make_preprocess(32)(im), old(im), rtol=0, atol=0)

    def test_original_sampling_and_tokenizer(self):
        p = self.video("video0")
        frames, meta = dp.decode_video(p, dp.make_preprocess(32), max_frames=3)
        self.assertEqual(meta["frame_indices"], [0, 5, 15])
        original = dp.import_file("test_rawvideo", VENDOR / "rawvideo_util.reference.py")
        oracle = original.RawVideoExtractorCV2(size=32, framerate=1).get_video_data(str(p))["video"]
        torch.testing.assert_close(
            frames, oracle[np.linspace(0, len(oracle) - 1, 3, dtype=int)], rtol=0, atol=0
        )
        captions = [
            "A person jumps into the water.",
            "hello " * 90,
            "Tiếng Việt &amp; punctuation!",
        ]
        tokens = dp.tokenize_records(captions, self.tokenizer)
        for i, text in enumerate(captions):
            words = ["<|startoftext|>"] + self.tokenizer.tokenize(text)
            words = words[:31] + ["<|endoftext|>"]
            ids = self.tokenizer.convert_tokens_to_ids(words)
            self.assertEqual(tokens[i].tolist(), ids + [0] * (32 - len(ids)))

    def test_strict_checkpoint_and_real_forward(self):
        model = self.model()
        p = self.root / "model.bin"
        dp.atomic_torch(
            p, {"state_dict": {"module.clip." + k: v for k, v in model.state_dict().items()}}
        )
        loaded, tokenizer, info = dp.load_encoder(p, VENDOR, "cpu", "fp32")
        self.assertEqual(info["embed_dim"], 32)
        tokens = dp.tokenize_records(["a video"], tokenizer)
        with torch.inference_mode():
            torch.testing.assert_close(
                loaded.encode_text(tokens), model.encode_text(tokens), rtol=0, atol=0
            )
            self.assertTrue(torch.isfinite(loaded.encode_image(torch.zeros(1, 3, 32, 32))).all())
        invalid = {"clip." + k: v for k, v in model.state_dict().items()}
        invalid["transformerClip.resblocks.0.attn.in_proj_weight"] = torch.zeros(1)
        dp.atomic_torch(p, invalid)
        with self.assertRaisesRegex(ValueError, "meanP"):
            dp.load_encoder(p, VENDOR, "cpu", "fp32")

    def test_cache_reuse_and_corruption_repair(self):
        model = self.model().requires_grad_(False)
        records = [
            {"video_id": f"video{i}", "text": f"a person {i}", "text_id": f"q{i}"} for i in range(2)
        ]
        index = {r["video_id"]: self.video(r["video_id"]) for r in records}
        info = {"resolution": 32, "embed_dim": 32, "checkpoint_sha256": "synthetic-only"}
        args = (records, index, model, self.tokenizer, info, self.root / "cache")
        first = dp.build_cache(*args, max_frames=3, device="cpu")
        second = dp.build_cache(*args, max_frames=3, device="cpu")
        torch.testing.assert_close(first["frames"], second["frames"], rtol=0, atol=0)
        self.assertEqual(first["metadata"]["fingerprint"], second["metadata"]["fingerprint"])
        namespace = Path(first["metadata"]["namespace"])
        next((namespace / "videos").glob("*.pt")).write_bytes(b"broken cache")
        next((namespace / "texts").glob("*.pt")).write_bytes(b"broken cache")
        repaired = dp.build_cache(*args, max_frames=3, device="cpu")
        torch.testing.assert_close(first["frames"], repaired["frames"], rtol=0, atol=0)
        torch.testing.assert_close(first["text"], repaired["text"], rtol=0, atol=0)
        dp.atomic_torch(self.root / "portable.pt", repaired)
        self.assertEqual(dp.safe_load(self.root / "portable.pt")["video_ids"], first["video_ids"])

    def test_zero_fps_and_missing_video_rejected(self):
        bad = self.root / "bad.mp4"
        bad.write_bytes(b"not a video")
        with self.assertRaises(ValueError):
            dp.decode_video(bad, dp.make_preprocess(32))
        with self.assertRaises(FileNotFoundError):
            dp.video_index(self.root, ["missing"])

    def write_csv(self, name, rows, fields=("video_id", "sentence")):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_csv_raw_text_ids_and_candidate_order(self):
        path = self.write_csv(
            "test.csv",
            [
                {"video_id": "video9", "sentence": "  A raw caption.  "},
                {"video_id": "video2", "sentence": "second caption"},
            ],
        )
        records = dp.evaluation_records(path)
        self.assertEqual(records[0]["text"], "  A raw caption.  ")
        self.assertEqual(records[0]["text_id"], "msrvtt_jsfusion_0000")
        model = self.model().requires_grad_(False)
        index = {r["video_id"]: self.video(r["video_id"]) for r in records}
        info = {"resolution": 32, "embed_dim": 32, "checkpoint_sha256": "synthetic"}
        cache = dp.build_cache(
            records,
            index,
            model,
            self.tokenizer,
            info,
            self.root / "cache",
            max_frames=3,
            device="cpu",
        )
        self.assertEqual(cache["video_ids"], ["video9", "video2"])

    def test_sourcegroup_split_matches_locked_algorithm_and_keeps_captions(self):
        ids = [f"video{i}" for i in (9, 2, 7, 1, 8, 3, 6, 4)]
        sources = {vid: f"https://example.invalid/source/{i // 2}" for i, vid in enumerate(ids)}
        records = [
            {"video_id": vid, "text": f"caption {j}", "text_id": f"{vid}_{j}", "candidate_index": i}
            for i, vid in enumerate(ids)
            for j in range(2)
        ]
        # Independently recorded output from the locked source-group v2 splitter.
        # Keep this test standalone; Kaggle has no project scripts/ package.
        expected_train = ["video9", "video2", "video6", "video4"]
        expected_val = ["video7", "video1", "video8", "video3"]
        train, reserved, status = dp.make_development_split(
            records, ["video100"], val_size=4, source_urls=sources
        )
        self.assertEqual(list(dict.fromkeys(r["video_id"] for r in train)), expected_train)
        self.assertEqual(list(dict.fromkeys(r["video_id"] for r in reserved)), expected_val)
        self.assertEqual(len(reserved), 8)
        self.assertIn("not_used_for_selection", status)
        self.assertFalse(
            {sources[r["video_id"]] for r in train} & {sources[r["video_id"]] for r in reserved}
        )
        with self.assertRaisesRegex(ValueError, "URL"):
            dp.make_development_split(records, ["video100"], val_size=4)

    def test_independent_validation_rejects_source_overlap(self):
        records = [{"video_id": "video1", "text": "a", "text_id": "train1"}]
        path = self.write_csv(
            "val.csv",
            [{"video_id": "video3", "sentence": "raw val", "source_url": "source_a"}],
            fields=("video_id", "sentence", "source_url"),
        )
        with self.assertRaisesRegex(ValueError, "URL"):
            dp.make_development_split(
                records,
                ["video2"],
                path,
                selection_policy="independent_validation",
                source_urls={"video1": "source_a", "video2": "source_b"},
            )

    def test_decoder_rejects_in_range_failure_even_if_frame_would_not_be_selected(self):
        from unittest.mock import patch

        class BrokenCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                return 1 if prop == cv2.CAP_PROP_FPS else 4

            def set(self, prop, value):
                self.position = value
                return True

            def read(self):
                return (
                    (False, None)
                    if self.position == 1
                    else (True, np.zeros((8, 8, 3), dtype=np.uint8))
                )

            def release(self):
                pass

        with patch.object(dp.cv2, "VideoCapture", return_value=BrokenCapture()):
            with self.assertRaisesRegex(ValueError, "decode frame 1/4"):
                dp.decode_video("synthetic.mp4", dp.make_preprocess(8), max_frames=2)

    def test_encoder_batch_and_cache_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            dp.encode_in_chunks(None, torch.ones(1), "image", "cpu", initial_chunk=0)
        with self.assertRaises(ValueError):
            dp.decode_video("never_opened", dp.make_preprocess(8), max_frames=0)

    def cache(self, offset=0):
        torch.manual_seed(7 + offset)
        f = torch.randn(3, 3, 8)
        return {
            "text": f.mean(1) + torch.randn(3, 8) * 0.03,
            "frames": f,
            "mask": torch.ones(3, 3, dtype=torch.bool),
            "video_ids": [f"v{offset + i}" for i in range(3)],
            "text_video_ids": [f"v{offset + i}" for i in range(3)],
            "text_ids": [f"q{offset + i}" for i in range(3)],
            "metadata": {"split": "test", "fingerprint": f"synthetic{offset}"},
        }

    def config(self):
        return {
            "WORK_DIR": str(self.root),
            "TAU": 0.05,
            "ALPHA": 0.25,
            "QUERY_CHUNK": 2,
            "VIDEO_CHUNK": 2,
            "RTOL": 1e-5,
            "ATOL": 1e-5,
            "EXPECTED_BASELINE_R1": None,
            "EXPECTED_R1_TOL_PP": 1e-8,
            "BASELINE_REFERENCE_NPZ": "",
            "REFERENCE_IS_LOGITS": False,
            "ALLOW_UNVERIFIED_EXPLORATORY": False,
            "REQUIRE_EXTERNAL_REFERENCE": False,
            "BOOTSTRAP_SAMPLES": 20,
            "SEED": 42,
        }

    def test_benchmark_files_gate_and_external_reference(self):
        cache = self.cache()
        cfg = self.config()
        state = {"config": cfg, "device": "cpu", "vendor_dir": str(VENDOR)}
        report, results = workflow.benchmark(state, cache)
        self.assertEqual(list(report["variants"]), ["B0", "B1", "B2"])
        self.assertTrue((self.root / "benchmark.csv").is_file())
        self.assertIn("B2-B0", report["comparisons"])
        cfg["BASELINE_REFERENCE_NPZ"] = str(self.root / "B0_scores.npz")
        report, _ = workflow.benchmark(state, cache)
        self.assertTrue(report["baseline_gate"]["external_score_parity"])
        cfg["BASELINE_REFERENCE_NPZ"] = ""
        cfg["EXPECTED_BASELINE_R1"] = -1
        report, _ = workflow.benchmark(state, cache)
        self.assertFalse(report["baseline_gate"]["reported_metric_matches"])
        # Scalar R1 is diagnostic; correctness is judged against score/rank evidence.
        self.assertEqual(list(report["variants"]), ["B0", "B1", "B2"])
        cfg["REQUIRE_EXTERNAL_REFERENCE"] = True
        report, _ = workflow.benchmark(state, cache)
        self.assertEqual(list(report["variants"]), ["B0"])
        self.assertFalse(report["baseline_gate"]["passed"])

    def test_train_and_benchmark_pipeline_and_repeat_resume(self):
        from unittest.mock import patch

        test = self.cache()
        train, val = self.cache(10), self.cache(20)
        train["metadata"]["split"] = "train"
        val["metadata"].update(split="val", encoder_training_overlap=True)
        cfg = self.config()
        cfg.update(
            RUN_TRAIN=True,
            EPOCHS=2,
            TRAIN_BATCH=2,
            LR=0.001,
            WEIGHT_DECAY=0.0001,
            TRAIN_LOGIT_SCALE=20.0,
            RESUME=True,
            ALLOW_SEEN_ENCODER_DEV=True,
        )
        state = {
            "config": cfg,
            "device": "cpu",
            "vendor_dir": str(VENDOR),
            "development_status": "synthetic_encoder_seen_dev",
        }
        report, results = workflow.benchmark(state, test)
        for attempt in range(2):
            state["model"] = self.model()
            state["tokenizer"] = self.tokenizer
            with patch.object(
                workflow,
                "cache_split",
                side_effect=lambda state, split: train if split == "train" else val,
            ):
                summary = workflow.train_and_benchmark(state, test, report, results)
            self.assertEqual(summary["variant"], "B3_trained_attention")
            self.assertTrue(summary["encoder_training_overlap"])
        self.assertTrue((self.root / "attention_head" / "final.pt").is_file())
        self.assertTrue((self.root / "trained_benchmark.json").is_file())


if __name__ == "__main__":
    unittest.main()

"""CPU synthetic checks. Run: python -m unittest discover -s notebooks/msrvtt_kaggle -p test_core.py"""

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from retrieval_core import (
    TinyAttentionHead,
    aligned_original_metrics,
    compare_runs,
    direct_meanp_reference,
    evaluate_bidirectional,
    load_attention_head,
    multi_positive_contrastive_loss,
    original_clip4clip_metrics,
    paired_bootstrap,
    score_matrix,
    train_attention_head,
    validate_cache,
)


def cache(prefix="train", seed=8, videos=5, captions=2, dim=6):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(videos, dim)).astype(np.float32)
    frames = np.repeat(centers[:, None, :], 3, axis=1)
    frames += rng.normal(scale=0.8, size=frames.shape).astype(np.float32)
    text = np.repeat(centers, captions, axis=0)
    text += rng.normal(scale=0.8, size=text.shape).astype(np.float32)
    ids = [f"{prefix}_{i}" for i in range(videos)]
    return {
        "text": torch.tensor(text),
        "frames": torch.tensor(frames),
        "mask": torch.ones(videos, 3, dtype=torch.bool),
        "video_ids": ids,
        "text_video_ids": [vid for vid in ids for _ in range(captions)],
        "text_ids": [f"{prefix}_caption_{i}" for i in range(len(text))],
        "metadata": {"split": prefix, "encoder_training_overlap": False},
    }


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.data = cache()
        self.q, self.f, self.m = [self.data[k] for k in ("text", "frames", "mask")]

    def test_b0_matches_independent_reference_and_normalizes_each_frame(self):
        # Unequal frame norms makes normalize-before-mean distinguishable.
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        f = torch.tensor([[[100.0, 0.0], [0.0, 1.0]], [[0.0, 2.0], [0.0, 5.0]]])
        m = torch.ones(2, 2, dtype=torch.bool)
        score = score_matrix(q, f, m, device="cpu", query_chunk=1, video_chunk=1)
        np.testing.assert_allclose(score, direct_meanp_reference(q, f, m), atol=1e-7)
        self.assertAlmostEqual(float(score[0, 0]), 2**-0.5, places=6)
        self.assertGreater(abs(float(score[0, 0]) - 1.0), 0.2)

    def test_b1_matches_explicit_pairwise_formula_and_masks_padding(self):
        m = self.m.clone()
        m[:, -1] = False
        modified = self.f.clone()
        modified[:, -1] = torch.tensor([100.0, -30.0, 7.0, 1.0, 2.0, 3.0])
        score = score_matrix(
            self.q, modified, m, method="B1", tau=0.17, query_chunk=3, video_chunk=2, device="cpu"
        )
        q, f = F.normalize(self.q, dim=-1), F.normalize(modified, dim=-1)
        expected = np.empty(score.shape, dtype=np.float32)
        for qi in range(len(q)):
            for vi in range(len(f)):
                valid = f[vi][m[vi]]
                w = torch.softmax((valid @ q[qi]) / 0.17, dim=0)
                pooled = F.normalize((w[:, None] * valid).sum(0), dim=0)
                expected[qi, vi] = float(q[qi] @ pooled)
        np.testing.assert_allclose(score, expected, atol=2e-7)
        before = score_matrix(self.q, self.f, m, method="B1", tau=0.17, device="cpu")
        np.testing.assert_allclose(before, score, atol=2e-7)

    def test_b0_uses_native_cpu_mean_and_numpy_product(self):
        q = self.q / self.q.norm(dim=-1, keepdim=True)
        pooled = []
        for row in self.f:
            value = (row / row.norm(dim=-1, keepdim=True)).mean(0)
            pooled.append((value / value.norm()).numpy())
        expected = q.numpy() @ np.asarray(pooled).T
        np.testing.assert_array_equal(score_matrix(self.q, self.f, self.m, device="cpu"), expected)

    def test_chunking_all_methods_and_mixture_scale(self):
        result = {}
        for method in ("B0", "B1", "B2"):
            small = score_matrix(
                self.q, self.f, self.m, method=method, query_chunk=1, video_chunk=2, device="cpu"
            )
            full = score_matrix(
                self.q,
                self.f,
                self.m,
                method=method,
                query_chunk=100,
                video_chunk=100,
                device="cpu",
            )
            np.testing.assert_allclose(small, full, atol=3e-7)
            result[method] = full
        np.testing.assert_allclose(
            result["B2"], 0.75 * result["B0"] + 0.25 * result["B1"], atol=1e-7
        )

    def test_high_temperature_limit_is_meanp(self):
        b0 = score_matrix(self.q, self.f, self.m, device="cpu")
        b1 = score_matrix(self.q, self.f, self.m, method="B1", tau=1e10, device="cpu")
        np.testing.assert_allclose(b0, b1, atol=2e-7)

    def test_zero_initialized_head_is_fixed_attention(self):
        head = TinyAttentionHead(self.q.shape[-1], tau=0.05)
        head.train()
        fixed = score_matrix(self.q, self.f, self.m, method="B2", device="cpu")
        learned = score_matrix(self.q, self.f, self.m, method="B2", head=head, device="cpu")
        np.testing.assert_allclose(fixed, learned, atol=3e-7)
        self.assertTrue(head.training)  # inference restores caller's mode

    def test_invalid_cache_fails_loudly_instead_of_deleting_candidate(self):
        bad = dict(self.data, mask=self.m.clone())
        bad["mask"][0] = False
        with self.assertRaisesRegex(ValueError, "valid"):
            validate_cache(bad)
        with self.assertRaisesRegex(ValueError, "valid"):
            score_matrix(self.q, self.f, bad["mask"], device="cpu")
        with self.assertRaises(ValueError):
            score_matrix(self.q, self.f, self.m, tau=0, device="cpu")
        zero = self.q.clone()
        zero[0] = 0
        with self.assertRaises(ValueError):
            score_matrix(zero, self.f, self.m, device="cpu")


class EvaluationTests(unittest.TestCase):
    def test_rectangular_multi_positive_ids_and_permutation_invariance(self):
        # Text order is b,a,b; video order a,b, with two b captions.
        s = np.array([[0.4, 0.9], [0.8, 0.1], [0.2, 0.95]], dtype=np.float32)
        tv, vv, ti = ["b", "a", "b"], ["a", "b"], ["b0", "a0", "b1"]
        result = evaluate_bidirectional(s, tv, vv, ti)
        self.assertEqual(result["text_to_video"]["R@1"], 100.0)
        self.assertEqual(result["video_to_text"]["R@1"], 100.0)
        shuffled = evaluate_bidirectional(
            s[[2, 0, 1]][:, [1, 0]],
            [tv[i] for i in [2, 0, 1]],
            ["b", "a"],
            [ti[i] for i in [2, 0, 1]],
        )
        self.assertEqual(result["text_to_video"], shuffled["text_to_video"])
        self.assertEqual(result["video_to_text"], shuffled["video_to_text"])

    def test_exact_ties_preserve_locked_candidate_order_both_directions(self):
        s = np.zeros((2, 2), dtype=np.float32)
        result = evaluate_bidirectional(s, ["b", "a"], ["b", "a"], ["t_b", "t_a"])
        np.testing.assert_array_equal(result["ranks_text_to_video"], [1, 2])
        np.testing.assert_array_equal(result["ranks_video_to_text"], [1, 2])
        self.assertEqual(result["ties_text_to_video"]["queries_with_top_score_tie"], 2)
        self.assertEqual(result["text_to_video"]["MRR"], 0.75)

    def test_original_evaluator_reproduces_tie_expansion(self):
        legacy = original_clip4clip_metrics(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(legacy["original_num_rank_entries"], 4)
        self.assertEqual(legacy["queries_with_ground_truth_score_ties"], 2)
        self.assertEqual(legacy["R@1"], 50.0)
        # GT is anti-diagonal; identity alignment is mandatory.
        aligned = aligned_original_metrics([[0.0, 1.0], [1.0, 0.0]], ["b", "a"], ["a", "b"])
        self.assertEqual(aligned["text_to_video"]["R@1"], 100.0)
        unavailable = aligned_original_metrics(np.zeros((3, 2)), ["a", "b", "b"], ["a", "b"])
        self.assertFalse(unavailable["available"])

    def test_paired_bootstrap_fixed_gallery_and_cluster_caption_counts(self):
        base, new = np.array([2, 1, 2]), np.array([1, 1, 1])
        result = paired_bootstrap(base, new, ["a", "a", "b"], n_bootstrap=100, seed=4)
        self.assertEqual(result["clusters"], 2)
        self.assertAlmostEqual(result["delta"]["R@1_pp"]["estimate"], 200 / 3)
        self.assertGreaterEqual(result["delta"]["R@1_pp"]["ci_low"], 50)
        self.assertLessEqual(result["delta"]["R@1_pp"]["ci_high"], 100)
        same = paired_bootstrap(base, base, ["a", "a", "b"], n_bootstrap=10)
        for metric in same["delta"].values():
            self.assertEqual(metric, {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0})

    def test_compare_runs_gain_loss_counts_and_export(self):
        base = {"ranks_text_to_video": np.array([1, 2, 3]), "ranks_video_to_text": np.array([1, 2])}
        new = {"ranks_text_to_video": np.array([2, 1, 1]), "ranks_video_to_text": np.array([2, 1])}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changes.csv"
            result = compare_runs(
                base,
                new,
                ["a", "b", "b"],
                ["a", "b"],
                ["a0", "b0", "b1"],
                n_bootstrap=10,
                output_csv=path,
            )
            self.assertEqual(result["text_to_video"]["corrected_at_1"], 2)
            self.assertEqual(result["text_to_video"]["broken_at_1"], 1)
            self.assertEqual(len(result["changed_queries"]), 5)
            self.assertIn("gt_video_id", path.read_text(encoding="utf-8"))


class TrainingTests(unittest.TestCase):
    def test_multi_positive_rectangular_loss_gradient(self):
        s = torch.tensor([[0.9, 0.1], [0.7, 0.3], [0.1, 0.8]], requires_grad=True)
        pos = torch.tensor([[True, False], [True, False], [False, True]])
        loss = multi_positive_contrastive_loss(s, pos, logit_scale=2)
        loss.backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(float(s.grad.abs().sum()), 0)
        all_positive = multi_positive_contrastive_loss(s.detach(), torch.ones_like(pos))
        self.assertAlmostEqual(float(all_positive), 0.0)

    def test_fixed_epochs_no_validation_and_exact_resume_same_schedule(self):
        train = cache()
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            full_path, resumed_path = Path(temporary) / "full", Path(temporary) / "resumed"
            full = train_attention_head(
                train, None, full_path, epochs=2, batch_size=2, device="cpu"
            )
            partial = train_attention_head(
                train,
                None,
                resumed_path,
                epochs=2,
                batch_size=2,
                device="cpu",
                max_epochs_this_call=1,
            )
            self.assertFalse(partial["training_complete"])
            self.assertIsNone(partial["checkpoint_path"])
            resumed = train_attention_head(
                train, None, resumed_path, epochs=2, batch_size=2, device="cpu", resume=True
            )
            self.assertEqual(full["history"], resumed["history"])
            self.assertEqual(full["selected_epoch"], 2)
            self.assertIsNone(full["best_val_r1"])
            self.assertNotIn("val_text_to_video", full["history"][0])
            self.assertEqual(Path(full["checkpoint_path"]).name, "final.pt")
            for key, value in full["head"].state_dict().items():
                torch.testing.assert_close(value, resumed["head"].state_dict()[key], rtol=0, atol=0)
            last = torch.load(
                resumed["last_checkpoint_path"], map_location="cpu", weights_only=True
            )
            self.assertGreater(float(last["model"]["query_delta.weight"].abs().sum()), 0)
            self.assertEqual(last["completed_epochs"], 2)
            loaded = load_attention_head(resumed["checkpoint_path"])
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(value, resumed["head"].state_dict()[key], rtol=0, atol=0)
            for epochs in (1, 3):
                with self.assertRaisesRegex(ValueError, "schedule"):
                    train_attention_head(
                        train,
                        None,
                        resumed_path,
                        epochs=epochs,
                        batch_size=2,
                        device="cpu",
                        resume=True,
                    )
            with self.assertRaises(FileExistsError):
                train_attention_head(
                    train, None, resumed_path, epochs=2, batch_size=2, device="cpu"
                )
            altered = dict(train, text=train["text"].clone())
            altered["text"][0, 0] += 0.001
            with self.assertRaisesRegex(ValueError, "cached features"):
                train_attention_head(
                    altered, None, resumed_path, epochs=2, batch_size=2, device="cpu", resume=True
                )

    def test_resume_ignores_paths_and_timing_but_binds_feature_semantics(self):
        train = cache()
        train["metadata"].update(
            settings={"max_frames": 3},
            fingerprint="stable_train",
            seconds=123.0,
            namespace="/old/cache",
            path="/old/video.mp4",
        )
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            first = train_attention_head(
                train, None, temporary, epochs=2, batch_size=2, device="cpu", max_epochs_this_call=1
            )
            rebuilt = copy.deepcopy(train)
            rebuilt["metadata"].update(seconds=0.2, namespace="/new/cache", path="/new/video.mp4")
            resumed = train_attention_head(
                rebuilt, None, temporary, epochs=2, batch_size=2, device="cpu", resume=True
            )
            self.assertEqual(first["cache_fingerprints"], resumed["cache_fingerprints"])
            changed = copy.deepcopy(rebuilt)
            changed["metadata"]["settings"] = {"max_frames": 4}
            with self.assertRaisesRegex(ValueError, "cached features"):
                train_attention_head(
                    changed, None, temporary, epochs=2, batch_size=2, device="cpu", resume=True
                )

    def test_fixed_epochs_refuses_any_validation_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "must not receive"):
                train_attention_head(cache(), cache("val"), temporary, device="cpu")

    def test_independent_validation_requires_provenance_and_disjoint_unseen_cache(self):
        train, val = cache(), cache("val", seed=19, videos=3)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "provenance"):
                train_attention_head(
                    train, val, temporary, selection_policy="independent_validation"
                )
            for invalid, message in ((train, "overlap"), (cache("test"), "test cache")):
                with self.assertRaisesRegex(ValueError, message):
                    train_attention_head(
                        train,
                        invalid,
                        temporary,
                        selection_policy="independent_validation",
                        validation_provenance="source audit fixture",
                    )
            for seen in (True, None):
                val["metadata"]["encoder_training_overlap"] = seen
                with self.assertRaisesRegex(ValueError, "absent from encoder"):
                    train_attention_head(
                        train,
                        val,
                        temporary,
                        selection_policy="independent_validation",
                        validation_provenance="source audit fixture",
                    )
            val["metadata"]["encoder_training_overlap"] = False
            result = train_attention_head(
                train,
                val,
                temporary,
                epochs=2,
                batch_size=2,
                device="cpu",
                selection_policy="independent_validation",
                validation_provenance="source audit fixture",
            )
            self.assertIn("val_text_to_video", result["history"][0])
            self.assertEqual(Path(result["checkpoint_path"]).name, "best.pt")
            self.assertIsNotNone(result["best_val_r1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Replay must restore numeric execution settings and preserve evaluator strictness."""

import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from temporal_hoi.data.vidhoi import HOIPair, VidHOIFrame
from temporal_hoi.evaluation.hoi import evaluate_frame_hoi


def _cli():
    path = Path(__file__).resolve().parents[1] / "scripts/run_vidhoi_head_study.py"
    specification = importlib.util.spec_from_file_location("_hoi_head_cli_regression", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_evaluate_restores_configured_threads_before_loading_or_scoring_cache(monkeypatch, tmp_path):
    module = _cli()
    entered = []

    class ReachedCache(Exception):
        pass

    def load_cache(*args):
        entered.append(torch.get_num_threads())
        raise ReachedCache

    monkeypatch.setattr(module, "preflight_head_data", lambda config: {})
    monkeypatch.setattr(module, "load_caches", load_cache)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        with pytest.raises(ReachedCache):
            module.evaluate_study({"threads": 2, "device": "cpu"}, tmp_path, {})
        assert entered == [2]
    finally:
        torch.set_num_threads(previous)


def test_score_tolerance_cannot_hide_a_material_ap_change_from_near_tie(monkeypatch, tmp_path):
    module = _cli()
    truth = HOIPair(0, 1, (0, 0, 4, 8), (6, 0, 8, 4), (1, 0),
                    subject_track_id="p", object_track_id="cup")
    positive_prediction = replace(truth, predicate_values=(1.0, 0.2))
    unmatched_prediction = replace(truth, predicate_values=(1.0000004, 0.2))
    correct_frame = VidHOIFrame("dev/positive", 0, "research_validation", "oracle",
                               "xyxy_pixels_original_frame", (truth,), (positive_prediction,), 0)
    unmatched_frame = VidHOIFrame("dev/unmatched", 0, "research_validation", "oracle",
                                 "xyxy_pixels_original_frame", (), (unmatched_prediction,), 0)
    persisted = [correct_frame, unmatched_frame]
    regenerated = [correct_frame, replace(unmatched_frame, predictions=(
        replace(unmatched_prediction, predicate_values=(0.9999996, 0.2)),))]
    old_metrics, new_metrics = evaluate_frame_hoi(persisted), evaluate_frame_hoi(regenerated)
    assert old_metrics["map"] == pytest.approx(0.5)
    assert new_metrics["map"] == pytest.approx(1)
    ranks = {"fixture": "rank identity deliberately unchanged"}
    for name, value in (
        ("predictions.json", [asdict(frame) for frame in persisted]),
        ("metrics.json", old_metrics), ("known_positive_ranks.json", ranks),
    ):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(module, "evaluate_head_state", lambda *args, **kwargs: {
        "frames": regenerated, "metrics": new_metrics, "known_positive_ranks": ranks})
    # A score change below the existing allclose tolerance swaps TP/FP order.
    # The evaluator replay must still reject the resulting AP difference.
    with pytest.raises(ValueError, match="metric replay mismatch"):
        module.replay_one(tmp_path, None, None, None, None, "object", "cpu")

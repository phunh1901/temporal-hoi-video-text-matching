"""Profiling must measure actual stages and explicit label-backed errors."""

import importlib.util
from pathlib import Path

import pytest

from temporal_hoi.evaluation.profiling import StageTimer, retrieval_error_groups


def test_stage_clock_counts_each_call_and_records_failed_work():
    times = iter([0, 2, 3, 7])
    timer = StageTimer(clock=lambda: next(times))
    with timer.measure("decode"):
        pass
    with pytest.raises(RuntimeError):
        with timer.measure("decode"):
            raise RuntimeError("decode failure")
    row = timer.summary()["decode"]
    assert row["seconds"] == 6
    assert row["calls"] == 2
    assert row["mean_seconds_per_call"] == 3


def test_nonmonotonic_clock_is_rejected():
    times = iter([2, 1])
    with pytest.raises(ValueError, match="monotonic"):
        with StageTimer(clock=lambda: next(times)).measure("decode"):
            pass


def test_errors_are_grouped_from_labels_and_do_not_invent_hoi_localization():
    predictions = [{"direction": "video_to_text", "query_id": "v", "split": "extension",
                    "ranked_candidates": [
                        {"candidate_id": "wrong", "rank": 1, "score": .5, "is_positive": False},
                        {"candidate_id": "right", "rank": 2, "score": .4, "is_positive": True},
                        {"candidate_id": "also_right", "rank": 3, "score": .4,
                         "is_positive": True},
                    ]}]
    result = retrieval_error_groups(predictions, {"v": "cycle"})
    assert result["groups"][0]["positive_negative_pairs"] == 2
    assert result["groups"][0]["positive_not_strictly_above_negative_pairs"] == 2
    assert result["top1_errors"][0]["first_positive_rank"] == 2
    assert result["hoi_localization_errors"]["count"] is None
    assert result["hoi_localization_errors"]["status"] == "not_run"


def test_queries_without_positive_are_excluded_not_misclassified():
    result = retrieval_error_groups([{
        "direction": "text_to_video", "query_id": "unknown",
        "ranked_candidates": [{"candidate_id": "v", "rank": 1, "score": .5,
                               "is_positive": False}],
    }])
    assert result["groups"] == []
    assert result["top1_errors"] == []
    assert len(result["excluded_unlabelled_queries"]) == 1


def test_memory_monitor_collects_end_sample_and_keeps_commit_separate(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "profile_memory_test_baseline", Path(__file__).parents[1] / "scripts/run_pilot_baseline.py"
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    monkeypatch.setattr(baseline, "memory_snapshot", lambda: {
        "supported": True, "available_physical_bytes": 100,
        "available_commit_bytes": 300, "peak_process_working_set_bytes": 70,
        "peak_process_commit_bytes": 250,
    })
    monitor = baseline.MemoryMonitor()
    monitor.start()
    result = monitor.finish()
    assert result["samples"] >= 1
    assert result["peak_process_working_set_bytes"] == 70
    assert result["peak_process_commit_bytes"] == 250

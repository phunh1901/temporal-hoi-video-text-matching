"""Independent hand-calculated expectations against pinned upstream evaluator."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_benchmark_metrics.py"
SPEC = importlib.util.spec_from_file_location("w03_metric_conformance", SCRIPT)
CONFORMANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFORMANCE)


@pytest.fixture(scope="module")
def report():
    return CONFORMANCE.build_report()


def case(report, case_id):
    return next(item for item in report["cases"] if item["id"] == case_id)


def test_pinned_source_integrity_is_enforced(tmp_path):
    changed = tmp_path / "metrics.py"
    changed.write_text("raise RuntimeError('should never be executed')", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        CONFORMANCE.load_official(changed)


def test_single_caption_matches_both_directions(report):
    forward = case(report, "singleton_unique_text_to_video")
    reverse = case(report, "singleton_unique_video_to_text")
    assert forward["internal"]["first_positive_ranks"] == [2, 3, 2]
    assert reverse["internal"]["first_positive_ranks"] == [2, 2, 2]
    assert forward["comparable_metrics_equal"] and reverse["comparable_metrics_equal"]


def test_cutoffs_are_one_based_and_inclusive(report):
    item = case(report, "cutoff_boundaries_12_candidates")
    for result in [item["official"], item["internal"]]:
        assert result["R@1"] == pytest.approx(1 / 12)
        assert result["R@5"] == pytest.approx(5 / 12)
        assert result["R@10"] == pytest.approx(10 / 12)
        assert result["MedR"] == 6.5
        assert result["first_positive_ranks"] == [1, 2, 5, 6, 10, 11, 12, 3, 4, 7, 8, 9]


def test_official_tie_can_expand_denominator_without_changing_w02_contract(report):
    item = case(report, "exact_score_tie")
    assert item["official"]["first_positive_ranks"] == [1, 2, 2]
    assert item["official"]["R@1"] == pytest.approx(1 / 3)
    assert item["internal"]["first_positive_ranks"] == [1, 2]
    assert item["internal"]["R@1"] == 0.5
    assert item["internal"]["tie_breaking"] == "candidate_order"
    assert not item["comparable_metrics_equal"]


def test_torch_tensor_even_median_is_lower_middle(report):
    item = case(report, "tensor_even_median")
    assert item["official"]["MedR"] == 1.0
    assert item["internal"]["MedR"] == 1.5
    assert item["official"]["R@1"] == item["internal"]["R@1"] == 0.5


def test_multi_caption_pool_difference_is_explicit(report):
    item = case(report, "multi_caption_video_to_text_candidate_pool")
    assert item["official_grouped_pool"]["first_positive_ranks"] == [4, 1, 1, 1]
    assert item["internal_flat_caption_pool"]["first_positive_ranks"] == [7, 1, 1, 1]
    assert item["official_grouped_pool"]["R@5"] == 1.0
    assert item["internal_flat_caption_pool"]["R@5"] == 0.75
    assert item["same_pool_comparable_metrics_equal"]
    assert case(report, "multi_caption_text_to_video")["comparable_metrics_equal"]


def test_structural_padding_is_excluded_but_finite_unknown_requires_annotation(report):
    padding = case(report, "structural_padding_filtered_before_internal_scoring")
    assert padding["internal"]["first_positive_ranks"] == [1, 1, 2, None]
    assert padding["internal"]["num_evaluated_queries"] == 3
    assert padding["comparable_metrics_equal"]
    unknown = case(report, "finite_unknown_query")
    assert unknown["official"]["R@1"] == 0.5
    assert unknown["internal"]["R@1"] == 1
    assert unknown["internal"]["num_evaluated_queries"] == 1
    assert unknown["internal"]["num_excluded_queries"] == 1


def test_report_is_json_safe_and_has_no_model_claims(report):
    json.dumps(report, allow_nan=False)
    assert report["summary"] == {
        "num_cases": 9, "equivalent": 5, "expected_protocol_differences": 4,
        "unexpected_differences": 0,
    }
    assert report["historical_w02_metric_contract_unchanged"]
    assert not report["model_quality_evaluated"] and not report["pilot_used"]
    assert all(item["conformance_passed"] for item in report["cases"])
    assert all(item["reason"] for item in report["cases"]
               if item["status"] == "expected_protocol_difference")


def test_summary_rejects_same_pool_failure_inside_expected_difference_case(monkeypatch):
    original = CONFORMANCE.comparable_equal

    def simulate_same_pool_failure(official, internal):
        if internal["first_positive_ranks"] == [4, 1, 1, 1]:
            return False
        return original(official, internal)

    monkeypatch.setattr(CONFORMANCE, "comparable_equal", simulate_same_pool_failure)
    changed = CONFORMANCE.build_report()
    assert changed["summary"]["unexpected_differences"] == 1
    assert not case(changed, "multi_caption_video_to_text_candidate_pool")["conformance_passed"]

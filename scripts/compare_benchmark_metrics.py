"""Offline W03 conformance fixtures; these are not model-quality experiments.

The upstream evaluator is used unchanged, from a pinned source snapshot. Keep
the W02 metric contract intact and report intentional protocol differences.
"""

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from temporal_hoi.evaluation import compute_retrieval_metrics

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "508ffa3de39ba0563a03199c440ab602a72e9b6f"
SOURCE_SHA256 = "103e93090de14f55d1db61e40ecf1fbc814670975bd9b9ab1fe341d0c95f0a6f"
SOURCE = ROOT / "outputs/w03/sources/clip4clip_metrics.py"
UPSTREAM_URL = f"https://github.com/ArrowLuo/CLIP4Clip/blob/{COMMIT}/metrics.py"


def load_official(source=SOURCE):
    """Verify the pinned bytes before loading the reviewed, small evaluator."""
    source = Path(source)
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"Pinned upstream evaluator hash mismatch: {actual_hash}")
    spec = importlib.util.spec_from_file_location("w03_clip4clip_official", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_official(metrics):
    """Convert percentages to fractions; preserve the upstream median convention."""
    return {
        "R@1": metrics["R1"] / 100.0,
        "R@5": metrics["R5"] / 100.0,
        "R@10": metrics["R10"] / 100.0,
        "MedR": float(metrics["MR"]),
        "MeanR": float(metrics["MeanR"]),
        **({"first_positive_ranks": [r + 1 for r in metrics["cols"]]}
           if "cols" in metrics else {}),
    }


def comparable_equal(official, internal):
    keys = ("R@1", "R@5", "R@10", "MedR")
    return all(np.isclose(official[key], internal[key]) for key in keys)


def _plain_case(official, case_id, sim, labels=None, *, reason=None):
    sim = np.asarray(sim, dtype=float)
    labels = [[i] for i in range(len(sim))] if labels is None else labels
    upstream = normalize_official(official.compute_metrics(sim))
    internal = compute_retrieval_metrics(sim, labels)
    equal = comparable_equal(upstream, internal)
    expected_equal = reason is None
    return {
        "id": case_id,
        "status": "equivalent" if expected_equal else "expected_protocol_difference",
        "input": {"scores": sim.tolist(), "positive_mapping": labels},
        "official": upstream,
        "internal": internal,
        "comparable_metrics_equal": equal,
        "expected_comparable_metrics_equal": expected_equal,
        "reason": reason,
    }


def build_report(source=SOURCE):
    official = load_official(source)
    cases = []
    sim = np.asarray([[0.8, 0.9, 0.1], [0.7, 0.6, 0.95], [0.99, 0.4, 0.5]])
    for direction, matrix in [("text_to_video", sim), ("video_to_text", sim.T)]:
        cases.append(_plain_case(official, f"singleton_unique_{direction}", matrix))

    # Each of ranks 1..12 occurs exactly once. Diagonal remains the positive.
    desired_ranks = [1, 2, 5, 6, 10, 11, 12, 3, 4, 7, 8, 9]
    cutoff_sim = np.empty((12, 12))
    for query, rank in enumerate(desired_ranks):
        order = [i for i in range(12) if i != query]
        order.insert(rank - 1, query)
        cutoff_sim[query, order] = np.arange(12, 0, -1)
    case = _plain_case(official, "cutoff_boundaries_12_candidates", cutoff_sim)
    case["hand_calculated_ranks"] = desired_ranks
    cases.append(case)

    cases.append(_plain_case(
        official, "exact_score_tie", [[0.9, 0.9], [0.8, 0.7]],
        reason=("compute_metrics returns every position equal to the diagonal score. "
                "A tie creates three ranks for two queries; W02 uses one stable "
                "candidate-order rank per query. Preserve both contracts."),
    ))

    even_tensor = np.asarray([[[0.9, 0.1]], [[0.8, 0.7]]])
    even_official = normalize_official(official.tensor_text_to_video_metrics(even_tensor))
    even_internal = compute_retrieval_metrics(even_tensor[:, 0, :], [[0], [1]])
    cases.append({
        "id": "tensor_even_median",
        "status": "expected_protocol_difference",
        "input": {"tensor": even_tensor.tolist(), "positive_mapping": [[0], [1]]},
        "official": even_official,
        "internal": even_internal,
        "comparable_metrics_equal": comparable_equal(even_official, even_internal),
        "expected_comparable_metrics_equal": False,
        "reason": ("torch.median of one-based ranks [1,2] returns 1; the internal "
                   "NumPy median returns 1.5. Plain upstream compute_metrics also "
                   "uses NumPy median; this difference is specific to the tensor branch."),
    })

    tensor = np.asarray([
        [[0.6, 0.1, 0.2, 0.3], [0.5, 0.11, 0.21, 0.31]],
        [[0.99, 0.9, 0.25, 0.35], [0.98, 0.8, 0.24, 0.34]],
        [[0.95, 0.12, 0.9, 0.36], [0.94, 0.13, 0.8, 0.37]],
        [[0.88, 0.14, 0.26, 0.9], [0.87, 0.15, 0.27, 0.8]],
    ])
    caption_scores = tensor.reshape(8, 4)
    text_labels = [[i // 2] for i in range(8)]
    text_official = normalize_official(official.tensor_text_to_video_metrics(tensor))
    text_internal = compute_retrieval_metrics(caption_scores, text_labels)
    cases.append({
        "id": "multi_caption_text_to_video",
        "status": "equivalent",
        "input": {"tensor": tensor.tolist(), "positive_mapping": text_labels},
        "official": text_official,
        "internal": text_internal,
        "comparable_metrics_equal": comparable_equal(text_official, text_internal),
        "expected_comparable_metrics_equal": True,
        "reason": "Eight caption queries each have their own source video as one positive.",
    })
    grouped_scores = official.tensor_video_to_text_sim(tensor.copy()).numpy()
    grouped_official = normalize_official(official.compute_metrics(grouped_scores))
    grouped_internal = compute_retrieval_metrics(grouped_scores, [[i] for i in range(4)])
    flat_internal = compute_retrieval_metrics(
        caption_scores.T, [[2 * i, 2 * i + 1] for i in range(4)]
    )
    cases.append({
        "id": "multi_caption_video_to_text_candidate_pool",
        "status": "expected_protocol_difference",
        "input": {"tensor": tensor.tolist()},
        "official_grouped_pool": grouped_official,
        "internal_same_grouped_pool": grouped_internal,
        "internal_flat_caption_pool": flat_internal,
        "same_pool_comparable_metrics_equal": comparable_equal(
            grouped_official, grouped_internal
        ),
        "expected_same_pool_comparable_metrics_equal": True,
        "flat_pool_comparable_metrics_equal": comparable_equal(grouped_official, flat_internal),
        "expected_flat_pool_comparable_metrics_equal": False,
        "reason": ("Official multi-sentence V-to-T max-pools captions by source video, "
                   "then ranks four groups. The flat internal pool ranks eight captions: "
                   "the first positive for video 0 moves from rank 4 to rank 7, "
                   "changing R@5 from 1 to 0.75. Use the official grouped pool "
                   "for reproduction and label the flat-caption result separately."),
    })

    padded = np.asarray([[[0.9, 0.1], [0.8, 0.2]], [[0.7, 0.6], [-np.inf, -np.inf]]])
    padded_official = normalize_official(official.tensor_text_to_video_metrics(padded))
    # Internal input validation rejects non-finite scores, so remove structural
    # padding before scoring. Keep a finite placeholder only for excluded queries.
    finite_scores = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.7, 0.6], [0, 0]])
    padded_internal = compute_retrieval_metrics(finite_scores, [[0], [0], [1], None])
    cases.append({
        "id": "structural_padding_filtered_before_internal_scoring",
        "status": "equivalent",
        "input": {"finite_scores": finite_scores.tolist(),
                  "positive_mapping": [[0], [0], [1], None],
                  "official_tensor_padding": "video 1 caption 1 = -inf in every column"},
        "official": padded_official,
        "internal": padded_internal,
        "comparable_metrics_equal": comparable_equal(padded_official, padded_internal),
        "expected_comparable_metrics_equal": True,
        "reason": ("Official tensor mask excludes non-finite diagonal padding. "
                   "Internal finite-score inputs exclude None relevance explicitly; "
                   "these agree after structural padding is handled by an adapter."),
    })
    cases.append(_plain_case(
        official, "finite_unknown_query", [[0.9, 0.1], [0.8, 0.7]], [[0], None],
        reason=("A finite upstream diagonal is assumed labeled: two queries, R@1=0.5. "
                "Internal None relevance excludes the second query: one query, R@1=1. "
                "The complete official benchmark must contain no unknown relevance; "
                "do not silently apply this internal exclusion to change its denominator."),
    ))
    for case in cases:
        checks = [
            case[key] == case[f"expected_{key}"]
            for key in (
                "comparable_metrics_equal",
                "same_pool_comparable_metrics_equal",
                "flat_pool_comparable_metrics_equal",
            )
            if key in case
        ]
        case["conformance_passed"] = bool(checks) and all(checks)
    return {
        "schema_version": 1,
        "task": "W03-04",
        "review_date": "2026-10-03",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_level": "evaluator_conformance_only",
        "model_quality_evaluated": False,
        "pilot_used": False,
        "official_source": {
            "repository": "https://github.com/ArrowLuo/CLIP4Clip",
            "commit": COMMIT,
            "path": "metrics.py",
            "url": UPSTREAM_URL,
            "snapshot": str(Path(source).relative_to(ROOT)).replace("\\", "/"),
            "sha256": SOURCE_SHA256,
            "bytes": Path(source).stat().st_size,
        },
        "environment": {"numpy": np.__version__, "torch": torch.__version__, "device": "cpu"},
        "historical_w02_metric_contract_unchanged": True,
        "summary": {
            "num_cases": len(cases),
            "equivalent": sum(case["status"] == "equivalent" for case in cases),
            "expected_protocol_differences": sum(
                case["status"] == "expected_protocol_difference" for case in cases
            ),
            "unexpected_differences": sum(
                not case["conformance_passed"]
                for case in cases
            ),
        },
        "cases": cases,
        "limits": [
            "Synthetic hand-calculated fixtures verify evaluator semantics, not model quality.",
            "MRR is internal auxiliary output; upstream does not report it.",
            "No HOI evaluator or mAP is implemented or measured by this script.",
            "Plain MSR-VTT 1k loader uses one sentence per CSV row; its default evaluator "
            "uses compute_metrics in each direction. Tensor cases audit the generic "
            "multi-sentence branch, not a change to the locked MSR-VTT 1k protocol.",
            "Upstream tensor argsort omits stable=True; tied ranks are not a portable "
            "candidate-order guarantee across Torch versions/devices.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/w03/metric_conformance.json")
    args = parser.parse_args()
    report = build_report(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    if report["summary"]["unexpected_differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Evaluate ST-HOI JSON results or verify local metrics against a pinned notebook.

No checkpoint is loaded by this evaluator. Re-scoring author results is an
evaluator check, never evidence of model inference or a zero-shot reproduction.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from temporal_hoi.data.vidhoi import (
    BoxTransform,
    ClassMapping,
    adapt_sthoi_result,
    identity_hash,
    select_frame_mask,
)
from temporal_hoi.evaluation.hoi import evaluate_frame_hoi

NOTEBOOK_SHA256 = "211fd893d3e36159395e6dba3535ed0f39dff46dfd3c9705adb592c6498b2e0b"
UPSTREAM_COMMIT = "c906d7ece7c6db423742b35859e0126373422595"
MAPPING_SHA256 = "a74c9c6c0e1efa116eb368935aded8b5a124c3d758376391924db8e2a58368d7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_cells(path: Path) -> list[ast.Module]:
    """Read only IoU and the two metric cells after enforcing the pinned source hash.

    Strip notebook magic, display-only imports, and final print statements.
    Matching/AP statements and their order stay exactly as in upstream.
    """
    if sha256(path) != NOTEBOOK_SHA256:
        raise ValueError("Upstream notebook hash mismatch; refuse execution")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if not any(marker in source for marker in ("def bbox_iou", "def voc_ap",
                                                     "tp, fp, scores, sum_gt = {}, {}, {}, {}")):
            continue
        source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (
            isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "print"
        )]
        cells.append(tree)
    if len(cells) != 3:
        raise ValueError("Pinned notebook does not contain the expected metric cells")
    return cells


def evaluate_reference(cells, rows: list[dict], class_subset: str) -> dict:
    import math

    namespace = {
        "all_results": rows, "np": np, "math": math, "tqdm": lambda iterable: iterable,
        "delete_less_than_25_instances": class_subset == "nonrare",
        "delete_more_than_25_instances": class_subset == "rare",
        "only_evaluate_temporal_predicates": False,
        "is_demo": False, "is_demo_top_10": False,
    }
    for tree in cells:
        exec(compile(tree, "<pinned ST-HOI metric cell>", "exec"), namespace)
    return {
        "map": float(namespace["mAP"]),
        "classes": [{"triplet": list(key), "gt_count": namespace["sum_gt"][key],
                     "ap": float(namespace["ap"][index])}
                    for index, key in enumerate(namespace["tp"])],
    }


def fixture_row() -> dict:
    labels = [0.0] * 50
    labels[0] = 1.0
    scores = [0.001] * 50
    scores[0] = 0.9
    return {
        "gt_boxes": [[0, 0, 0, 10, 10], [0, 20, 0, 30, 10]],
        "gt_obj_classes": [[0, 0], [0, 1]], "gt_bbox_pair_ids": [[0, 1]],
        "gt_action_labels": [labels], "proposal_boxes": [[0, 0, 0, 10, 10],
                                                         [0, 20, 0, 30, 10]],
        "proposal_classes": [[0, 0], [0, 1]], "proposal_scores": [[0, 1], [0, 1]],
        "preds_bbox_pair_ids": [[0, 1]], "preds_score": [scores],
    }


def conformance_fixtures() -> list[tuple[str, list[dict], str, float]]:
    base = fixture_row()
    wrong = copy.deepcopy(base)
    wrong["proposal_boxes"][1] = [0, 50, 0, 60, 10]
    edge = copy.deepcopy(base)
    edge["proposal_boxes"][0] = [0, 0, 0, 20, 10]
    duplicate = copy.deepcopy(base)
    duplicate["preds_bbox_pair_ids"].append([0, 1])
    duplicate["preds_score"].append([0.0001] * 50)
    duplicate["preds_score"][1][0] = 0.8
    shared = copy.deepcopy(base)
    shared["gt_action_labels"][0][1] = 1
    shared["preds_score"][0][1] = 0.8
    ties = copy.deepcopy(shared)
    ties["preds_score"][0][1] = 0.9
    floor = copy.deepcopy(base)
    floor["preds_score"][0][0] = 0
    floor["proposal_scores"][0][1] = -1
    missing = copy.deepcopy(base)
    missing["preds_bbox_pair_ids"] = []
    missing["preds_score"] = []
    crowded = copy.deepcopy(base)
    crowded["proposal_boxes"].extend([[0, 40, 0, 50, 10], [0, 60, 0, 70, 10]])
    crowded["proposal_classes"].extend([[0, 2], [0, 2]])
    crowded["proposal_scores"].extend([[0, 1], [0, 1]])
    crowded["preds_bbox_pair_ids"].extend([[0, 2], [0, 3]])
    crowded["preds_score"].extend([[1.0] * 50, [1.0] * 50])
    rare_rows = [copy.deepcopy(base) for _ in range(24)]
    other = copy.deepcopy(base)
    other["gt_action_labels"][0] = [0] * 50
    other["gt_action_labels"][0][1] = 1
    other["preds_score"][0][1] = 0.95
    rare_rows.extend(copy.deepcopy(other) for _ in range(25))
    return [
        ("perfect_single_predicate", [base], "full", 1.0),
        ("wrong_object_box", [wrong], "full", 0.0),
        ("inclusive_IoU_0_5", [edge], "full", 1.0),
        ("duplicate_prediction_consumes_GT_once", [duplicate], "full", 1.0),
        ("shared_GT_pair_across_predicates", [shared], "full", 0.5),
        ("topk_tuple_tie_predicate_descending", [ties], "full", 0.5),
        ("nonpositive_score_floor", [floor], "full", 1.0),
        ("GT_class_without_predictions", [missing], "full", 0.0),
        ("top100_before_supported_class_filter", [crowded], "full", 0.0),
        ("rare_count_24_vs_25", rare_rows, "rare", 1.0),
        ("nonrare_count_24_vs_25", rare_rows, "nonrare", 1.0),
    ]


def build_conformance_report(notebook: Path) -> dict:
    cells = reference_cells(notebook)
    mapping = ClassMapping({0: "person", 1: "cup", 2: "chair"},
                           {index: f"predicate_{index}" for index in range(50)})
    results = []
    for name, rows, subset, expected in conformance_fixtures():
        frames = [adapt_sthoi_result(row, mapping, video_id="fixture", frame_id=index,
                                     split="fixture", box_source="oracle")
                  for index, row in enumerate(rows)]
        internal = evaluate_frame_hoi(frames, class_subset=subset)
        official = evaluate_reference(cells, rows, subset)
        expected_classes = [(row["triplet"], row["gt_count"], row["ap"])
                            for row in official["classes"]]
        actual_classes = [(row["triplet"], row["gt_count"], row["ap"])
                          for row in internal["classes"]]
        equal = len(expected_classes) == len(actual_classes) and all(
            a[:2] == b[:2] and np.isclose(a[2], b[2])
            for a, b in zip(expected_classes, actual_classes, strict=True)
        )
        passed = equal and np.isclose(internal["map"], official["map"]) and np.isclose(
            internal["map"], expected
        )
        results.append({"id": name, "source": "synthetic_fixture_only", "subset": subset,
                        "expected_map": expected, "official": official,
                        "internal": internal, "conformance_passed": bool(passed)})
    return {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": "evaluator_conformance_synthetic_fixtures", "benchmark_evaluated": False,
        "model_inference_run": False, "dataset_annotation_accessed": False,
        "official_final_split_accessed": False, "benchmark_map": None,
        "upstream_commit": UPSTREAM_COMMIT, "upstream_notebook_sha256": NOTEBOOK_SHA256,
        "python": platform.python_version(), "numpy": np.__version__,
        "source_cell_edits": "magic/imports/print removed; matching and AP preserved",
        "cases": results, "summary": {"cases": len(results),
            "passed": sum(row["conformance_passed"] for row in results),
            "unexpected_differences": sum(not row["conformance_passed"] for row in results)},
    }


def evaluate_files(results_path, metadata_path, mapping_path, mask_path, box_source, subset):
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping = ClassMapping.from_json(mapping_path)
    if (len(mapping.objects) != 78 or len(mapping.predicates) != 50
            or sha256(mapping_path) != MAPPING_SHA256):
        raise ValueError("Actual ST-HOI evaluation requires the pinned canonical 78/50 mapping")
    if metadata.get("box_source") != box_source:
        raise ValueError("Metadata box_source must match the requested box source")
    fixed_before_scores = metadata.get("selection_configuration_fixed_before_scores")
    if not isinstance(fixed_before_scores, bool):
        raise ValueError("Metadata must explicitly record selection_configuration_fixed_before_scores")
    if metadata["split"] == "official_val" and not fixed_before_scores:
        raise ValueError("Official val access requires a configuration fixed before observing scores")
    if not isinstance(rows, list) or len(rows) != len(metadata["frames"]):
        raise ValueError("Every result row must have one ordered frame manifest entry")
    for path, field in ((results_path, "results_sha256"), (mapping_path, "mapping_sha256")):
        if metadata.get(field) != sha256(path):
            raise ValueError(f"Metadata {field} missing or mismatched")
    frames = []
    identity_sources = set()
    for row, info in zip(rows, metadata["frames"], strict=True):
        raw_key = row.get("orig_video_idx")
        if raw_key is not None:
            if isinstance(raw_key, list) and len(raw_key) == 1:
                raw_key = raw_key[0]
            if not isinstance(raw_key, str) or raw_key != info.get("result_frame_key"):
                raise ValueError("Result frame identity disagrees with manifest result_frame_key")
            identity_sources.add("raw_result_key_checked_against_manifest")
        else:
            identity_sources.add("caller_provided_ordered_manifest_only")
        transform = info.get("box_transform")
        if transform is None:
            raise ValueError("Actual result runs require an explicit original-to-model box transform")
        frames.append(adapt_sthoi_result(
            row, mapping, video_id=info["video_id"], frame_id=info["frame_id"],
            split=metadata["split"], box_source=box_source,
            box_transform=BoxTransform(**transform), timestamp_s=info.get("timestamp_s"),
            gt_track_ids={int(k): v for k, v in info.get("gt_track_ids", {}).items()},
            proposal_track_ids={int(k): v for k, v in info.get("proposal_track_ids", {}).items()},
        ))
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    if not isinstance(mask.get("protocol_id"), str) or not mask["protocol_id"].strip():
        raise ValueError("Frame mask protocol_id must be nonempty")
    selected = select_frame_mask(frames, mask["frame_ids"])
    expected_count = mask.get("expected_frame_count", len(mask["frame_ids"]))
    if (not isinstance(expected_count, int) or isinstance(expected_count, bool)
            or expected_count < len(mask["frame_ids"])):
        raise ValueError("expected_frame_count must be an integer >= materialized mask size")
    output = evaluate_frame_hoi(selected, class_subset=subset)
    output.update({
        "level": "provided_prediction_evaluator_check", "model_inference_run": False,
        "checkpoint_inference_claim": False, "results_sha256": sha256(results_path),
        "metadata_sha256": sha256(metadata_path), "mapping_sha256": sha256(mapping_path),
        "mask_sha256": sha256(mask_path), "mask_protocol_id": mask["protocol_id"],
        "input_frames": len(frames), "excluded_frames": len(frames) - len(selected),
        "coverage": len(selected) / expected_count,
        "coverage_denominator": "declared_expected_mask_frame_count",
        "full_benchmark_reproduction_claim": False,
        "benchmark_comparison_eligible": False,
        "benchmark_comparison_status": "provided_prediction_check_requires_external_protocol_audit",
        "mask_frame_ids_sha256": identity_hash(mask["frame_ids"]),
        "official_final_split_accessed": metadata["split"] == "official_val",
        "selection_configuration_fixed_before_scores": fixed_before_scores,
        "selection_configuration_provenance": "explicit_metadata_assertion_not_independently_verified",
        "identity_provenance": sorted(identity_sources),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--box-source", choices=("oracle", "predicted"))
    parser.add_argument("--subset", choices=("full", "rare", "nonrare"), default="full")
    parser.add_argument("--verify-upstream-notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.verify_upstream_notebook:
        output = build_conformance_report(args.verify_upstream_notebook)
    else:
        if not all((args.results, args.metadata, args.mapping, args.mask, args.box_source)):
            parser.error("Supply results, metadata, mapping, mask and box-source, or verify notebook")
        output = evaluate_files(args.results, args.metadata, args.mapping, args.mask,
                                args.box_source, args.subset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output.get("summary", {"map": output.get("map"),
                                           "num_frames": output.get("num_frames")})))
    if output.get("summary", {}).get("unexpected_differences", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""The CLI must refuse mislabeled sources, impossible coverage and unbound IDs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_vidhoi.py"
SPEC = importlib.util.spec_from_file_location("vidhoi_evaluator_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


@pytest.fixture
def input_files(tmp_path, monkeypatch):
    # Canonical mapping is separately hash-gated; this fixture isolates file gates.
    mapping = {
        "objects": {index: "person" if index == 0 else f"object_{index}" for index in range(78)},
        "predicates": {index: f"predicate_{index}" for index in range(50)},
    }
    paths = {name: tmp_path / f"{name}.json" for name in ("results", "metadata", "mapping", "mask")}
    paths["mapping"].write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setattr(CLI, "MAPPING_SHA256", CLI.sha256(paths["mapping"]))
    rows = [CLI.fixture_row()]
    rows[0]["orig_video_idx"] = ["folder/video_000016"]
    paths["results"].write_text(json.dumps(rows), encoding="utf-8")
    metadata = {"results_sha256": CLI.sha256(paths["results"]),
                "mapping_sha256": CLI.sha256(paths["mapping"]), "split": "official_val",
                "box_source": "oracle", "selection_configuration_fixed_before_scores": True,
                "frames": [{"video_id": "folder/video", "frame_id": 16,
                            "result_frame_key": "folder/video_000016", "box_transform": {}}]}
    mask = {"protocol_id": "synthetic_mask_only", "expected_frame_count": 1,
            "frame_ids": ["folder/video_000016"]}
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    paths["mask"].write_text(json.dumps(mask), encoding="utf-8")
    return paths, metadata, mask


def run(paths, box_source="oracle"):
    return CLI.evaluate_files(paths["results"], paths["metadata"], paths["mapping"],
                              paths["mask"], box_source, "full")


@pytest.mark.parametrize("count", [0, -1, 0.5, True])
def test_invalid_mask_denominator_rejected(input_files, count):
    paths, _, mask = input_files
    mask["expected_frame_count"] = count
    paths["mask"].write_text(json.dumps(mask), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_frame_count"):
        run(paths)


@pytest.mark.parametrize("protocol", ["", None, 1])
def test_mask_protocol_is_required(input_files, protocol):
    paths, _, mask = input_files
    mask["protocol_id"] = protocol
    paths["mask"].write_text(json.dumps(mask), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_id"):
        run(paths)


def test_oracle_cannot_be_relabeled_predicted(input_files):
    with pytest.raises(ValueError, match="box_source"):
        run(input_files[0], box_source="predicted")


def test_final_split_requires_explicit_fixed_selection(input_files):
    paths, metadata, _ = input_files
    metadata["selection_configuration_fixed_before_scores"] = False
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed before"):
        run(paths)


def test_manifest_does_not_override_raw_result_identity(input_files):
    paths, metadata, _ = input_files
    metadata["frames"][0]["result_frame_key"] = "folder/video_000020"
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Result frame identity"):
        run(paths)


def test_file_hash_and_original_coordinate_transform_required(input_files):
    paths, metadata, _ = input_files
    original = copy.deepcopy(metadata)
    metadata["results_sha256"] = "changed"
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="results_sha256"):
        run(paths)
    del original["frames"][0]["box_transform"]
    paths["metadata"].write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="box transform"):
        run(paths)


def test_canonical_mapping_hash_is_required_for_actual_run(input_files, monkeypatch):
    monkeypatch.setattr(CLI, "MAPPING_SHA256", "wrong")
    with pytest.raises(ValueError, match="canonical 78/50"):
        run(input_files[0])


def test_output_records_access_but_never_claims_model_inference(input_files):
    output = run(input_files[0])
    assert output["official_final_split_accessed"]
    assert output["coverage"] == 1
    assert not output["model_inference_run"] and not output["benchmark_comparison_eligible"]
    assert output["identity_provenance"] == ["raw_result_key_checked_against_manifest"]
    assert output["selection_configuration_provenance"].endswith("not_independently_verified")
    assert output["map"] == pytest.approx(1)


def test_reference_notebook_hash_failure_prevents_exec(tmp_path):
    notebook = tmp_path / "changed.ipynb"
    notebook.write_text("raise RuntimeError('must not run')", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        CLI.reference_cells(notebook)


def test_all_conformance_fixtures_have_independent_hand_calculated_expectations():
    from temporal_hoi.data.vidhoi import ClassMapping, adapt_sthoi_result
    from temporal_hoi.evaluation.hoi import evaluate_frame_hoi

    mapping = ClassMapping({0: "person", 1: "cup", 2: "chair"},
                           {index: f"predicate_{index}" for index in range(50)})
    for _, rows, subset, expected in CLI.conformance_fixtures():
        frames = [adapt_sthoi_result(row, mapping, video_id="fixture", frame_id=index,
                                     split="fixture", box_source="oracle")
                  for index, row in enumerate(rows)]
        assert evaluate_frame_hoi(frames, class_subset=subset)["map"] == pytest.approx(expected)

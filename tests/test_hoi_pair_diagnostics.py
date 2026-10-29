"""Hand-computed global budget/matching controls and bounded source-result loading."""

import copy
import hashlib
import io
import json
import struct
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from temporal_hoi.data.hoi_head_cache import _frame_record
from temporal_hoi.data.vidhoi import ClassMapping, HOIPair, VidHOIFrame
from temporal_hoi.evaluation.hoi_pair_diagnostics import (
    _numeric_npz,
    load_pair_diagnostic_source,
    pair_budget_diagnostics,
    run_pair_score_diagnostics,
    score_pair_rows,
)
from temporal_hoi.inference.hoi_static import OracleEntity, OracleFrame, file_sha256
from temporal_hoi.training.hoi_head_study import ARRAY_KEYS, run_head_study


def _pair(subject="p", obj="c1", values=(1.0, 0.0, 0.0), *, box=(6, 0, 8, 4)):
    return HOIPair(0, 1, (0, 0, 4, 8), box, values,
                    subject_track_id=subject, object_track_id=obj)


def _frame(truth, predictions):
    return VidHOIFrame("dev/v", 0, "research_validation", "oracle",
                       "xyxy_pixels_original_frame", tuple(truth), tuple(predictions), 0)


def test_top_budget_and_shared_pair_consumption_are_distinct_gates():
    truth = _pair(values=(1, 1, 0))
    prediction = _pair(values=(2.0, 3.0, 0.5))
    frame = _frame([truth], [prediction])
    full = pair_budget_diagnostics([frame])
    assert full["num_gt_positive_entries"] == 2
    assert full["num_canonical_positive_slots_survived"] == 2
    assert full["num_GT_entries_accepted_by_any_retained_prediction"] == 1
    assert full["retained_prediction_outcomes"]["shared_pair_blocked_by_other_predicate"] == 1
    assert full["records"][0]["canonical_slot_evaluator_outcome"] == "shared_pair_blocked_by_other_predicate"
    assert full["records"][1]["GT_entry_accepted_by_any_retained_prediction"]
    limited = pair_budget_diagnostics([frame], top_k=1)
    assert limited["num_canonical_positive_slots_outside_budget"] == 1
    assert limited["records"][0]["global_slot_rank"] == 2
    assert limited["records"][0]["canonical_slot_evaluator_outcome"] == "outside_top_k"
    assert limited["matching_trace_TP_count_equals_evaluator"]


def test_global_tie_order_can_exclude_positive_while_unknown_stays_unknown():
    positive = _pair(values=(1, 0, 0))
    unknown = _pair(obj="c2", values=(1, 0, 0), box=(9, 0, 11, 4))
    result = pair_budget_diagnostics([_frame([positive], [positive, unknown])], top_k=1)
    assert result["records"][0]["global_slot_rank"] == 2
    assert result["num_canonical_positive_slots_survived"] == 0
    assert result["retained_prediction_outcomes"] == {"evaluator_no_supported_GT_match": 1}
    assert result["frames"][0]["retained_unannotated_slots_not_verified_negative"] == 1
    assert result["unannotated_is_not_verified_negative"]


def test_overlapping_boxes_do_not_prove_actor_identity_correctness():
    first, second = _pair(subject="p1"), _pair(subject="p2")
    result = pair_budget_diagnostics([_frame([first, second], [first, second])])
    # Tuple ties favor proposal p2, but identical boxes cause the evaluator to match GT p1.
    assert result["num_GT_entries_accepted_by_any_retained_prediction"] == 1
    assert result["records"][1]["canonical_slot_matched_gt_index"] == 0
    assert not result["records"][1]["GT_entry_accepted_by_any_retained_prediction"]
    assert result["not_actor_correctness_accuracy"]


def test_trace_requires_canonical_identity_and_geometry():
    truth = _pair()
    with pytest.raises(ValueError, match="absent canonical"):
        pair_budget_diagnostics([_frame([truth], [_pair(obj="wrong")])])
    with pytest.raises(ValueError, match="geometry"):
        pair_budget_diagnostics([_frame([truth], [_pair(box=(7, 0, 9, 4))])])
    with pytest.raises(ValueError, match="unique explicit"):
        pair_budget_diagnostics([_frame([truth], [truth, truth])])


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _records(root):
    return [{"path": file.relative_to(root).as_posix(), "bytes": file.stat().st_size,
             "sha256": file_sha256(file)} for file in sorted(root.rglob("*")) if file.is_file()]


def _cache(role, hashes):
    mapping = ClassMapping({0: "person", 1: "cup"}, {0: "hold", 1: "open", 2: "touch"})
    positive = 1 if role == "train" else 0
    truth = _pair(values=tuple(float(index == positive) for index in range(3)))
    frame = OracleFrame(f"{role}/v", 1, 0, 0, Path("unavailable_image.png"), "0" * 64, 12, 8,
        (OracleEntity("p", 0, truth.subject_box), OracleEntity("c1", 1, truth.object_box),
         OracleEntity("c2", 1, (9, 0, 11, 4))), (truth,))
    positive_mask = np.zeros((2, 3), dtype=bool)
    positive_mask[0, positive] = True
    metadata = {"role": role, "source_split": "train", "test_fixture": True,
        "level": "synthetic_encoder_fixture", "config_sha256": "b" * 64,
        "label_policy": "positive_only_unknown", "official_final_split_accessed": False,
        "final_unseen_accessed": False, "pseudo_unseen_in_train": False, "input_hashes": hashes,
        **{name + "_sha256": hashes[name] for name in ("protocol", "mapping", "checkpoint")}}
    return {"mapping": mapping, "frames": [frame], "metadata": metadata,
        "object_features": np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.float32),
        "union_features": np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.float32),
        "object_class": np.asarray([1, 1], dtype=np.int64), "positive_mask": positive_mask,
        "frame_index": np.asarray([0, 0], dtype=np.int64),
        "subject_track_id": np.asarray(["p", "p"]), "object_track_id": np.asarray(["c1", "c2"])}


@pytest.fixture(scope="module")
def source_zip(tmp_path_factory):
    root = tmp_path_factory.mktemp("pair_source")
    study = root / "source/study"
    hashes = {name: hashlib.sha256(name.encode()).hexdigest()
              for name in ("train_manifest", "dev_manifest", "protocol", "mapping", "checkpoint")}
    train, dev = _cache("train", hashes), _cache("validation", hashes)
    texts = np.tile(np.eye(3, dtype=np.float32)[None], (2, 1, 1))
    allowed = np.ones((2, 3), dtype=bool)
    for role, cache in (("train", train), ("validation", dev)):
        directory = study / "cache" / role
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "arrays.npz", **{key: cache[key] for key in ARRAY_KEYS})
        _write(directory / "frames.json", {"mapping": asdict(cache["mapping"]),
            "frames": [_frame_record(frame) for frame in cache["frames"]], "metadata": cache["metadata"]})
        _write(directory / "receipt.json", {"schema_version": "vidhoi_head_role_cache_v1",
            "status": "completed", **cache["metadata"], "num_frames": 1, "num_pairs": 2,
            "files": _records(directory)})
    np.savez_compressed(study / "cache/text_features.npz", text_features=texts, candidate_mask=allowed)
    _write(study / "cache/receipt.json", {"schema_version": "vidhoi_head_cache_receipt_v1",
        "status": "completed", "level": "synthetic_encoder_fixture", "encoder": {"test_fixture": True},
        "model_inference_run": False, "model_training_run": False, "input_hashes": hashes,
        "config_sha256": "b" * 64, "official_final_split_accessed": False,
        "final_unseen_accessed": False, "files": _records(study / "cache")})
    run_head_study(train, dev, texts, allowed, {"test_fixture": True, "max_epochs": 1}, study / "training")
    _write(study / "evaluation/receipt.json", {"schema_version": "vidhoi_head_evaluation_audit_v1",
        "status": "completed", "official_final_split_accessed": False, "final_unseen_accessed": False,
        "input_hashes": hashes, "files": []})
    _write(study / "receipt.json", {"schema_version": "vidhoi_head_study_run_v1",
        "status": "completed", "official_final_split_accessed": False, "final_unseen_accessed": False,
        "input_hashes": hashes, "files": _records(study)})
    _write(root / "source/run_config.json", {"device": "cpu", "threads": 2,
                                             **{name + "_sha256": value for name, value in hashes.items()}})
    _write(root / "source/environment.json", {"device": "fixture_original_cpu"})
    archive = root / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as stream:
        for path in sorted((root / "source").rglob("*")):
            if path.is_file():
                stream.write(path, path.relative_to(root / "source").as_posix())
    return archive


@pytest.fixture
def source(source_zip):
    return load_pair_diagnostic_source(source_zip, file_sha256(source_zip), test_fixture=True)


def test_source_is_numeric_only_and_preserves_unknown_train_proposals(source):
    receipt = source["source_receipt"]
    assert receipt["level"] == "synthetic_cached_pair_source_fixture"
    assert receipt["all_member_crc_and_stage_inventory_hashes_verified"]
    assert not receipt["files_extracted_or_executed"]
    assert len(source["states"]) == 9
    assert source["train_cache"]["positive_mask"][1].sum() == 0
    original = score_pair_rows(source, "object", 42, "initial", [0, 1, 1])
    assert original.shape == (3, 3) and original.dtype == np.float64
    assert np.array_equal(original[0], original[1]) and np.array_equal(original[1], original[2])
    changed = copy.deepcopy(source)
    changed["train_cache"]["positive_mask"][:] = False
    assert np.array_equal(original, score_pair_rows(changed, "object", 42, "initial", [0, 1, 1]))
    with pytest.raises(ValueError, match="row indices"):
        score_pair_rows(source, "object", 42, "last", [2])
    with pytest.raises(ValueError, match="two-thread"):
        score_pair_rows(source, "object", 42, "last", [1], threads=3)


def test_eighteen_states_are_diagnostic_and_source_metrics_stay_separate(source, tmp_path):
    output = tmp_path / "diagnosed"
    recorded = source["recorded_frozen"]["object"]["metrics"]["map"]
    # Deliberately different source metric tests reporting separation, not evaluator replacement.
    source["recorded_frozen"]["object"]["metrics"]["map"] = recorded / 2
    torch.set_num_threads(3)
    receipt = run_pair_score_diagnostics(source, {"test_fixture": True,
        "source_result_sha256": source["source_receipt"]["source_result_sha256"]}, output)
    assert torch.get_num_threads() == 2
    assert receipt["diagnosed_states"] == 18 and receipt["test_fixture"]
    assert receipt["head_inference_run"] and not receipt["model_training_run"]
    assert not receipt["backbone_inference_run"]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["comparisons"]) == 9 and len(summary["aggregate"]) == 3
    first = summary["comparisons"][0]
    assert first["recorded_source_frozen_map"] == recorded / 2
    assert first["current_initial_map"] == recorded
    assert first["recorded_source_vs_current_initial_map_delta_pp"] != 0
    assert not summary["new_training_or_configuration_selection"]
    assert summary["unannotated_is_not_verified_negative"]
    assert len(list(output.glob("states/*/seed_*/*/pair_budget_diagnostic.json"))) == 18


def test_source_hash_scope_and_fixture_cannot_be_silently_replaced(source_zip, source, tmp_path):
    with pytest.raises(ValueError, match="hash/size"):
        load_pair_diagnostic_source(source_zip, "0" * 64, test_fixture=True)
    with pytest.raises(ValueError, match="fixed three-view"):
        load_pair_diagnostic_source(source_zip, file_sha256(source_zip))
    config = {"test_fixture": True, "source_result_sha256": source["source_receipt"]["source_result_sha256"],
              "protocol_sha256": "0" * 64}
    with pytest.raises(ValueError, match="protocol/mapping"):
        run_pair_score_diagnostics(source, config, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(ValueError, match="fixture/production"):
        run_pair_score_diagnostics(source, {"source_result_sha256": config["source_result_sha256"]},
                                   tmp_path / "wrong_scope")


@pytest.mark.parametrize("mutation", ["traversal", "inventory_tamper", "crc"])
def test_source_zip_path_hash_and_crc_guards(source_zip, tmp_path, mutation):
    target = tmp_path / "mutated.zip"
    with zipfile.ZipFile(source_zip) as source:
        contents = {row.filename: source.read(row) for row in source.infolist()}
    if mutation == "traversal":
        contents["../outside.json"] = b"{}"
    elif mutation == "inventory_tamper":
        contents["study/cache/train/arrays.npz"] = b"changed input cache"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    if mutation == "crc":
        with zipfile.ZipFile(target) as archive:
            info = archive.getinfo("environment.json")
        raw = bytearray(target.read_bytes())
        name_size, extra_size = struct.unpack_from("<HH", raw, info.header_offset + 26)
        offset = info.header_offset + 30 + name_size + extra_size
        raw[offset + 2] ^= 1
        target.write_bytes(raw)
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        load_pair_diagnostic_source(target, file_sha256(target), test_fixture=True)


def test_npz_objects_or_large_declared_shapes_are_rejected_before_allocation():
    raw = io.BytesIO()
    np.savez(raw, x=np.asarray([{"untrusted": "data"}], dtype=object))
    with pytest.raises(ValueError, match="Unsafe object"):
        _numeric_npz(raw.getvalue(), {"x"})
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False,
                                                 "shape": (10**12,)})
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as archive:
        archive.writestr("x.npy", header.getvalue())
    with pytest.raises(ValueError, match="oversized"):
        _numeric_npz(raw.getvalue(), {"x"})

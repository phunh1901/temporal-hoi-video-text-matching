"""Portable pair handoff guards; contract fixtures never claim real scoring or human labels."""

import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from temporal_hoi.data.hoi_pair_review import new_review_annotations, validate_review_annotations
from temporal_hoi.inference import hoi_kaggle as runtime

PROJECT = Path(__file__).resolve().parents[1]


def packager():
    spec = importlib.util.spec_from_file_location(
        "_pair_packager_fixture", PROJECT / "scripts/prepare_vidhoi_pair_diagnostics_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def records(directory):
    return [{"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
             "sha256": runtime.sha256_file(path)} for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "receipt.json"]


@pytest.fixture
def assets(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "source.zip"
    with zipfile.ZipFile(source, "x") as archive:
        archive.writestr("explicit_test_fixture.json", json.dumps({"test_fixture": True}))
    protocol, mapping = data / "protocol.json", data / "mapping.json"
    write(protocol, {"schema_version": "vidhoi_head_protocol_v1", "source_split": "train"})
    write(mapping, {"objects": {"0": "person", "1": "cup"}, "predicates": {"0": "hold"}})
    review = data / "review"
    review.mkdir()
    image = review / "context/frame.png"
    image.parent.mkdir()
    Image.new("RGB", (16, 12), (255, 0, 0)).save(image)
    manifest = review / "review_manifest.json"
    write(manifest, {"schema_version": "vidhoi_pair_review_v1", "source_split": "train",
                     "role": "train", "protocol_sha256": runtime.sha256_file(protocol),
                     "mapping_sha256": runtime.sha256_file(mapping),
                     "selection": {"role": "train", "model_scores_used": False,
                         "source_train_video_ids": ["train/a"], "excluded_dev_video_ids": ["dev/b"],
                         "excluded_final_video_ids": ["final/c"], "selected_case_ids": ["fixture_case"]},
                     "files": {"context/frame.png": {"bytes": image.stat().st_size,
                                                      "sha256": runtime.sha256_file(image)}},
                     "cases": [{"case_id": "fixture_case", "video_id": "train/a",
                         "frame_id": 0, "source_frame_id": 1, "timestamp_s": 0.0,
                         "width": 16, "height": 12, "object_track_id": "cup", "object_class": 1,
                         "object_bbox_xyxy": [12, 0, 16, 12], "predicate_id": 0, "predicate_name": "hold",
                         "actors": [{"track_id": "p1", "bbox_xyxy": [0, 0, 4, 12],
                                     "source_annotation_positive": True, "label": "unknown"},
                                    {"track_id": "p2", "bbox_xyxy": [4, 0, 8, 12],
                                     "source_annotation_positive": False, "label": "unknown"}],
                         "image_path": "context/frame.png", "image_sha256": runtime.sha256_file(image),
                         "context_frames": [{"path": "context/frame.png", "frame_id": 0,
                              "timestamp_s": 0.0, "width": 16, "height": 12,
                              "sha256": runtime.sha256_file(image)}], "video": None}]})
    annotations = data / "annotations.json"
    write(annotations, new_review_annotations(json.loads(manifest.read_text(encoding="utf-8"))))
    wheels = data / "wheelhouse"
    wheels.mkdir()
    (wheels / "requirements-offline.txt").write_text("open-clip-torch==3.3.0\n")
    (wheels / "open_clip_torch-3.3.0-py3-none-any.whl").write_bytes(
        b"Explicit packaging fixture only; never installed")
    project = tmp_path / "source"
    for name in packager().SOURCES:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if (PROJECT / name).is_file():
            shutil.copy2(PROJECT / name, path)
        else:
            path.write_text('"""Explicit source packaging fixture, never executed."""\n')
    return {"source_result": source, "protocol": protocol, "mapping": mapping,
            "review_manifest": manifest, "annotations": annotations,
            "wheelhouse": wheels, "project_root": project}


@pytest.fixture
def package(assets, tmp_path):
    directory = tmp_path / "package"
    return directory, packager().prepare(**assets, output=directory)


def test_no_clip_weights_and_seven_automatic_cells_with_manual_review(package, tmp_path):
    directory, receipt = package
    assert receipt["bundle"] == "vidhoi_m1_pair_diagnostics_input.zip"
    assert receipt["notebook"] == "vidhoi_m1_pair_diagnostics_kaggle.ipynb"
    assert receipt["review_case_count"] == 1
    assert not receipt["training_run"] and not receipt["encoder_inference_run"]
    assert not receipt["head_inference_run"] and not receipt["uploaded"]
    with zipfile.ZipFile(directory / receipt["bundle"]) as archive:
        manifest = runtime.validate_manifest(archive.read(runtime.MANIFEST_NAME))
        assert manifest["entrypoint"] == "scripts/run_vidhoi_pair_diagnostics.py"
        assert manifest["purpose"] == "head_pair_diagnostics"
        assert not any(name.endswith(".safetensors") for name in manifest["files"])
        assert manifest["source_result_member"] == "assets/source_result.bin"
        assert "assets/source_result.bin" in manifest["files"]
        assert "assets/source_result.zip" not in manifest["files"]
        assert "assets/review/context/frame.png" in manifest["files"]
        assert "assets/review/annotations.json" in manifest["files"]
        assert set(packager().SOURCES) <= set(manifest["files"])
        assert not any(name.lower().endswith(("agents.md", "rules.md", "rubric.md"))
                       or name.startswith((".git/", ".agents/", "noiquy/", "reports/"))
                       for name in manifest["files"])
    root = tmp_path / "installed"
    runtime.install_input(directory, root, receipt["manifest_sha256"])
    config = runtime.prepare_run_config(root, tmp_path / "run/config.json", device="cpu")
    values = json.loads(config.read_text(encoding="utf-8"))
    for key in packager().INPUT_KEYS:
        assert Path(values[key]).is_absolute()
        assert runtime.sha256_file(values[key]) == values[f"{key}_sha256"]
    notebook = json.loads((directory / receipt["notebook"]).read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 7
    for cell in notebook["cells"]:
        ast.parse(cell["source"])
    assert 'BASE = WORK / "vidhoi_m1_pair_diagnostics"' in notebook["cells"][0]["source"]
    assert '"pair_diagnostics_" + run_id' in notebook["cells"][0]["source"]
    for index, stage in enumerate(("preflight", "diagnose", "review"), 3):
        assert f'"--stage", "{stage}"' in notebook["cells"][index]["source"]
    assert '"--annotation-input-root", str(INPUTS)' in notebook["cells"][5]["source"]
    assert "validate_pair_export(RUN, ROOT)" in notebook["cells"][6]["source"]
    assert not notebook["metadata"]["kaggle"]["isInternetEnabled"]


@pytest.mark.parametrize("mutation", ["traversal", "hash", "extra_type", "self_reference"])
def test_review_inventory_rejects_unsafe_or_changed_assets(assets, tmp_path, mutation):
    manifest_path = assets["review_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name, value = next(iter(manifest["files"].items()))
    if mutation == "traversal":
        manifest["files"] = {"../outside.png": value}
    elif mutation == "hash":
        manifest["files"][name]["sha256"] = "0" * 64
    elif mutation == "self_reference":
        manifest["files"] = {manifest_path.name: value}
    else:
        manifest["files"] = {"AGENTS.md": value}
    write(manifest_path, manifest)
    with pytest.raises(ValueError):
        packager().prepare(**assets, output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_reviewed_source_hash_cannot_be_silently_replaced(assets, tmp_path):
    config = tmp_path / "config.json"
    write(config, {"schema_version": "vidhoi_pair_diagnostics_config_v1",
                   "source_result_sha256": "0" * 64})
    with pytest.raises(ValueError, match="SHA256 differs: source_result"):
        packager().prepare(**assets, pair_config=config, output=tmp_path / "bad")


def test_existing_evidence_preserved(assets, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        packager().prepare(**assets, output=output)
    assert (output / "user.txt").read_text() == "preserve"


def test_source_change_changes_pin(assets, tmp_path):
    first = packager().prepare(**assets, output=tmp_path / "first")
    path = assets["project_root"] / "src/temporal_hoi/evaluation/hoi_pair_diagnostics.py"
    path.write_bytes(path.read_bytes() + b"\n# Explicit source hash packaging fixture\n")
    second = packager().prepare(**assets, output=tmp_path / "second")
    assert first["manifest_sha256"] != second["manifest_sha256"]
    with pytest.raises(FileNotFoundError, match="pinned"):
        runtime.install_input(tmp_path / "second", tmp_path / "bad", first["manifest_sha256"])


@pytest.fixture
def completed(package, tmp_path):
    """Contract outputs explicitly identify synthetic diagnostics; no scores are simulated as real."""
    directory, build = package
    root = tmp_path / "installed"
    runtime.install_input(directory, root, build["manifest_sha256"])
    run = tmp_path / "run"
    config_path = runtime.prepare_run_config(root, run / "run_config.json", device="cpu")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {key: config[f"{key}_sha256"] for key in packager().INPUT_KEYS}
    output = run / "pair_diagnostics"
    diagnostic = output / "diagnostics"
    for mode in ("object", "union", "fusion"):
        for seed in (42, 43, 44):
            for state in ("initial", "last"):
                for name in ("metrics", "known_positive_ranks", "predictions", "pair_budget_diagnostic"):
                    write(diagnostic / "states" / mode / f"seed_{seed}" / state / f"{name}.json",
                          {"test_fixture": True, "purpose": "contract integrity, not score inference"})
    write(diagnostic / "source_receipt.json", {"schema_version": "vidhoi_pair_source_validation_v1",
        "status": "completed", "level": "synthetic_cached_pair_source_fixture", "test_fixture": True,
        "source_result_sha256": expected["source_result"],
        "all_member_crc_and_stage_inventory_hashes_verified": True, "num_head_states": 18,
        "files_extracted_or_executed": False, "official_final_split_accessed": False,
        "final_unseen_accessed": False, "config_sha256": "a" * 64})
    write(diagnostic / "summary.json", {"schema_version": "vidhoi_pair_score_summary_v1",
        "comparisons": [{"mode": mode, "seed": seed} for mode in ("object", "union", "fusion")
                        for seed in (42, 43, 44)],
        "aggregate": {mode: {} for mode in ("object", "union", "fusion")},
        "new_training_or_configuration_selection": False,
        "unannotated_is_not_verified_negative": True,
        "cross_environment_policy": "Explicit contract fixture, no inference results claimed"})
    write(diagnostic / "receipt.json", {"schema_version": "vidhoi_pair_score_diagnostics_v1",
        "status": "completed", "level": "synthetic_pair_score_diagnostic",
        "diagnosed_states": 18, "input_hashes": expected,
        "source_result_sha256": expected["source_result"], "modes": ["object", "union", "fusion"],
        "source_config_sha256": "a" * 64,
        "seeds": [42, 43, 44], "states": ["initial", "last"], "model_training_run": False,
        "backbone_inference_run": False, "head_inference_run": True,
        "official_final_split_accessed": False, "final_unseen_accessed": False,
        "unknown_labels_treated_as_negative": False, "files": records(diagnostic)})
    review = output / "review"
    manifest = json.loads(Path(config["review_manifest"]).read_text(encoding="utf-8"))
    labels = json.loads(Path(config["annotations"]).read_text(encoding="utf-8"))
    validated = validate_review_annotations(labels, manifest)
    write(review / "review_manifest.json", manifest)
    write(review / "vidhoi_pair_review_annotations.json", labels)
    write(review / "annotation_validation.json", validated)
    (review / "actor_review.html").write_text("Explicit test fixture only", encoding="utf-8")
    write(review / "actor_scores_after_review.json", {
        "schema_version": "vidhoi_actor_review_scores_v1",
        "states": [{"mode": m, "seed": s, "state": v,
                    "cases": [{"case_id": "fixture_case", "video_id": "train/a",
                        "predicate_id": 0, "object_track_id": "cup", "actors": [
                        {"track_id": "p1", "score": 1.0}, {"track_id": "p2", "score": 1.0}]}]}
                   for m in ("object", "union", "fusion") for s in (42, 43, 44)
                   for v in ("initial", "last")]})
    write(review / "confirmed_actor_contrasts.json", {
        "schema_version": "vidhoi_review_actor_contrasts_v1", "status": "pending_verified_contrasts",
        "unknown_labels_used_as_negative": False, "benchmark_metric": False,
        "records": [], "aggregate": {f"{m}/{s}": {"seed_evaluations": 3,
            "num_pair_comparisons_across_seeds": 0, "strict_win_fraction": None,
            "near_tie_fraction": None, "mean_margin": None}
            for m in ("object", "union", "fusion") for s in ("initial", "last")}})
    write(review / "receipt.json", {"schema_version": "vidhoi_pair_review_result_v1",
        "status": "completed", "level": "train_only_actor_review_diagnostic",
        "source_result_sha256": expected["source_result"], "input_hashes": expected,
        "review_manifest_sha256": expected["review_manifest"], **{k: validated[k] for k in
            ("manifest_fingerprint", "case_count", "actor_count", "review_labels_status", "positive_count",
             "confirmed_negative_count", "unknown_count", "eligible_contrast_case_count", "fully_labeled_case_count")},
        "annotation_source": {"kind": "packaged_blank_or_reviewed_annotations", "external": False},
        "annotation_content_sha256": hashlib.sha256(json.dumps(labels, sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "model_training_run": False, "encoder_inference_run": False, "head_inference_run": True,
        "model_scores_hidden_in_review_HTML": True, "unknown_labels_treated_as_negative": False,
        "official_final_split_accessed": False, "final_unseen_accessed": False, "files": records(review)})
    write(output / "receipt.json", {"schema_version": "vidhoi_pair_diagnostic_run_v1",
        "status": "completed", "level": "cached_feature_pair_diagnostic",
        "training_run": False, "encoder_inference_run": False, "head_inference_run": True,
        "official_final_split_accessed": False, "final_unseen_accessed": False,
        "input_hashes": expected, "diagnosed_states": 18, "review_case_count": 1,
        "confirmed_negative_count": 0, "review_labels_status": "pending_labels",
        "stage_receipts": {name: {"path": name + "/receipt.json",
                                   "sha256": runtime.sha256_file(output / name / "receipt.json"),
                                   "bytes": (output / name / "receipt.json").stat().st_size}
                           for name in ("diagnostics", "review")}, "files": records(output)})
    return run, root


def test_fixture_cannot_finish_production_notebook_but_pending_labels_valid(completed):
    run, root = completed
    with pytest.raises(RuntimeError, match="Fixture results"):
        packager().validate_pair_export(run, root)
    receipt = packager().validate_pair_export(run, root, allow_fixture=True)
    assert receipt["review_labels_status"] == "pending_labels"
    assert receipt["confirmed_negative_count"] == 0 and not receipt["training_run"]


def test_missing_initial_or_last_output_stops_export(completed):
    run, root = completed
    (run / "pair_diagnostics/diagnostics/states/fusion/seed_44/last/metrics.json").unlink()
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        packager().validate_pair_export(run, root, allow_fixture=True)


def test_changed_input_annotations_rejected_before_completion(completed):
    run, root = completed
    (root / "assets/review/annotations.json").write_text('{"test_fixture":true,"changed":true}')
    with pytest.raises(RuntimeError, match="input identity mismatch: annotations"):
        packager().validate_pair_export(run, root, allow_fixture=True)


def test_missing_state_cannot_be_hidden_by_refreshing_file_inventories(completed):
    run, root = completed
    output = run / "pair_diagnostics"
    diagnostic = output / "diagnostics"
    (diagnostic / "states/fusion/seed_44/last/metrics.json").unlink()
    child = json.loads((diagnostic / "receipt.json").read_text(encoding="utf-8"))
    child["files"] = records(diagnostic)
    write(diagnostic / "receipt.json", child)
    outer = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    outer["files"] = records(output)
    outer["stage_receipts"]["diagnostics"].update(
        sha256=runtime.sha256_file(diagnostic / "receipt.json"),
        bytes=(diagnostic / "receipt.json").stat().st_size)
    write(output / "receipt.json", outer)
    with pytest.raises(RuntimeError, match="eighteen initial/last"):
        packager().validate_pair_export(run, root, allow_fixture=True)


def test_export_failure_retains_partial_zip_and_no_completed_event(package, tmp_path, monkeypatch):
    directory, build = package
    monkeypatch.setenv("VIDHOI_INPUT_ROOT", str(directory))
    monkeypatch.setenv("VIDHOI_WORK_ROOT", str(tmp_path / "working"))
    notebook = json.loads((directory / build["notebook"]).read_text(encoding="utf-8"))
    namespace = {"__name__": "__pair_export_failure_fixture__"}
    exec(compile(notebook["cells"][0]["source"], "paircell1", "exec"), namespace)
    with pytest.raises(FileNotFoundError):
        exec(compile(notebook["cells"][6]["source"], "paircell7", "exec"), namespace)
    events = [json.loads(line) for line in
              (namespace["RUN"] / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["stage"] == "export_incomplete" for e in events)
    assert not any(e["stage"] == "completed" for e in events)
    assert list(namespace["BASE"].glob("pair_diagnostics_*.zip"))


@pytest.mark.parametrize("mode", ["legacy_missing_kernel_path", "fixed", "tampered_source"])
def test_full_export_cell_uses_verified_source_in_isolated_kernel(completed, package, tmp_path, mode):
    """Synthetic export fixtures stay explicit; the child has no repository import path."""
    run, root = completed
    directory, build = package
    notebook = json.loads((directory / build["notebook"]).read_text(encoding="utf-8"))
    source = notebook["cells"][6]["source"]
    verified_kernel_setup = ("    verify_installed_input(ROOT, PINNED_MANIFEST)\n"
                             '    sys.path.insert(0, str(ROOT / "src"))\n')
    assert source.count(verified_kernel_setup) == 1
    fixture_call = "pair_receipt = validate_pair_export(RUN, ROOT)"
    assert source.count(fixture_call) == 1
    # Only this test invocation accepts synthetic outputs; production remains strict.
    source = source.replace(fixture_call,
                            "pair_receipt = validate_pair_export(RUN, ROOT, allow_fixture=True)", 1)
    if mode == "legacy_missing_kernel_path":
        source = source.replace(verified_kernel_setup, "", 1)
    elif mode == "tampered_source":
        path = root / "src/temporal_hoi/data/hoi_pair_review.py"
        path.write_bytes(path.read_bytes() + b"\n# Explicit source tamper fixture\n")
    bootstrap_source = (PROJECT / "src/temporal_hoi/inference/hoi_kaggle.py").read_text(
        encoding="utf-8")
    payload = {"source": source, "bootstrap": bootstrap_source, "root": str(root),
               "source_run": str(run), "manifest_sha256": build["manifest_sha256"],
               "mode": mode}
    child = textwrap.dedent('''
        import importlib.util, json, shutil, sys, types
        from pathlib import Path
        payload = json.loads(sys.stdin.read())
        assert importlib.util.find_spec("temporal_hoi") is None
        namespace = {"__name__": "__isolated_synthetic_export_fixture__"}
        exec(compile(payload["bootstrap"], "embedded_stdlib_bootstrap", "exec"), namespace)
        output = Path.cwd() / "synthetic_export_run"
        recorder = namespace["RunRecorder"](output, Path.cwd())
        shutil.copytree(Path(payload["source_run"]), output, dirs_exist_ok=True)
        namespace.update(ROOT=Path(payload["root"]), RUN=output, REC=recorder,
                         PINNED_MANIFEST=payload["manifest_sha256"])
        try:
            import IPython.display as ui
        except ModuleNotFoundError:
            # UI display alone is outside this import/export regression.
            package = types.ModuleType("IPython")
            ui = types.ModuleType("IPython.display")
            class FileLink:
                def __init__(self, path):
                    self.path = path
            ui.FileLink = FileLink
            ui.display = lambda *args, **kwargs: None
            package.display = ui
            sys.modules["IPython"] = package
            sys.modules["IPython.display"] = ui
        links = []
        original_display = ui.display
        def record_display(*values, **kwargs):
            links.extend(str(value.path) for value in values)
            return original_display(*values, **kwargs)
        ui.display = record_display
        error = None
        try:
            exec(compile(payload["source"], "actual_full_export_cell", "exec"), namespace)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        events = [json.loads(line) for line in (output / "events.jsonl").read_text(
                  encoding="utf-8").splitlines()]
        module = sys.modules.get("temporal_hoi.data.hoi_pair_review")
        result = {"mode": payload["mode"], "error": error, "links": links,
                  "completed": any(event["stage"] == "completed" for event in events),
                  "export_incomplete": any(event["stage"] == "export_incomplete" for event in events),
                  "zip_exists": output.with_suffix(".zip").is_file(),
                  "review_module": str(Path(module.__file__).resolve()) if module else None}
        print("ISOLATED_EXPORT_RESULT=" + json.dumps(result))
    ''')
    isolated = tmp_path / "isolated_kernel"
    isolated.mkdir()
    result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", child],
                            input=json.dumps(payload), cwd=isolated, capture_output=True,
                            text=True, encoding="utf-8", timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line.removeprefix("ISOLATED_EXPORT_RESULT=") for line in result.stdout.splitlines()
             if line.startswith("ISOLATED_EXPORT_RESULT=")]
    assert len(lines) == 1, result.stdout + result.stderr
    checked = json.loads(lines[0])
    assert checked["zip_exists"]
    if mode == "fixed":
        assert checked["error"] is None and checked["completed"]
        assert not checked["export_incomplete"]
        assert Path(checked["review_module"]).is_relative_to(root / "src")
        assert len(checked["links"]) == 2
        assert Path(checked["links"][0]).as_posix().endswith("review/actor_review.html")
        assert checked["links"][1].endswith("synthetic_export_run.zip")
    else:
        assert not checked["completed"] and checked["export_incomplete"]
        assert checked["review_module"] is None and checked["links"] == []
        if mode == "legacy_missing_kernel_path":
            assert checked["error"]["type"] == "ModuleNotFoundError"
            assert "temporal_hoi" in checked["error"]["message"]
        else:
            assert checked["error"]["type"] == "ValueError"
            assert "Installed bundle hash/size mismatch" in checked["error"]["message"]


def refresh_review_output(run):
    output = run / "pair_diagnostics"
    review = output / "review"
    child = json.loads((review / "receipt.json").read_text(encoding="utf-8"))
    child["files"] = records(review)
    write(review / "receipt.json", child)
    outer = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    outer["confirmed_negative_count"] = child["confirmed_negative_count"]
    outer["review_labels_status"] = child["review_labels_status"]
    outer["stage_receipts"]["review"].update(sha256=runtime.sha256_file(review / "receipt.json"),
                                            bytes=(review / "receipt.json").stat().st_size)
    outer["files"] = records(output)
    write(output / "receipt.json", outer)


def test_unknown_labels_cannot_gain_metrics_even_after_refreshing_hashes(completed):
    run, root = completed
    path = run / "pair_diagnostics/review/confirmed_actor_contrasts.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["aggregate"]["union/initial"]["mean_margin"] = 10.0
    write(path, value)
    refresh_review_output(run)
    with pytest.raises(RuntimeError, match="invented metrics"):
        packager().validate_pair_export(run, root, allow_fixture=True)


def test_human_asserted_fixture_pairs_are_recomputed_before_export(completed):
    run, root = completed
    review = run / "pair_diagnostics/review"
    manifest = json.loads((review / "review_manifest.json").read_text(encoding="utf-8"))
    labels = new_review_annotations(manifest)
    labels.update(reviewer_id="explicit_synthetic_fixture_not_real_people",
                  reviewed_at_utc="2026-10-30T01:00:00Z",
                  assertions={"human_reviewed": True, "not_ai_generated_ground_truth": True,
                              "model_scores_hidden": True})
    labels["cases"][0]["actors"][0]["label"] = "positive"
    labels["cases"][0]["actors"][1].update(label="confirmed_negative",
        evidence_context_indices=[0], rationale="Explicit contract fixture with synthetic red image")
    write(review / "annotations_original.json", labels)
    write(review / "vidhoi_pair_review_annotations.json", labels)
    checked = validate_review_annotations(labels, manifest)
    write(review / "annotation_validation.json", checked)
    receipt = json.loads((review / "receipt.json").read_text(encoding="utf-8"))
    receipt.update({k: checked[k] for k in ("review_labels_status", "positive_count",
        "confirmed_negative_count", "unknown_count", "eligible_contrast_case_count", "fully_labeled_case_count")})
    receipt["annotation_source"] = {"external": True, "sha256": runtime.sha256_file(review / "annotations_original.json"),
                                    "bytes": (review / "annotations_original.json").stat().st_size}
    receipt["annotation_content_sha256"] = hashlib.sha256(json.dumps(labels, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    write(review / "receipt.json", receipt)
    comparisons = [{"mode": m, "seed": s, "state": v, "case_id": "fixture_case",
        "positive_track_id": "p1", "negative_track_id": "p2", "score_margin": 0.0,
        "strict_win": False, "exact_tie": True, "near_tie": True}
        for m in ("object", "union", "fusion") for s in (42, 43, 44) for v in ("initial", "last")]
    contrast = {"schema_version": "vidhoi_review_actor_contrasts_v1", "status": "evaluated_human_asserted_pairs",
        "records": comparisons, "near_tie_tolerance": 1e-6, "unknown_labels_used_as_negative": False,
        "benchmark_metric": False, "aggregate": {f"{m}/{s}": {"seed_evaluations": 3,
            "num_pair_comparisons_across_seeds": 3, "strict_win_fraction": 0.0,
            "near_tie_fraction": 1.0, "mean_margin": 0.0}
            for m in ("object", "union", "fusion") for s in ("initial", "last")}}
    write(review / "confirmed_actor_contrasts.json", contrast)
    refresh_review_output(run)
    assert packager().validate_pair_export(run, root, allow_fixture=True)["confirmed_negative_count"] == 1
    contrast["aggregate"]["union/last"]["strict_win_fraction"] = 1.0
    write(review / "confirmed_actor_contrasts.json", contrast)
    refresh_review_output(run)
    with pytest.raises(RuntimeError, match="aggregate differs"):
        packager().validate_pair_export(run, root, allow_fixture=True)

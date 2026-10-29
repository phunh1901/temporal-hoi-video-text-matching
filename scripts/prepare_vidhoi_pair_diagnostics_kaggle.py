"""Package cached initial/last head diagnostics and an unlabeled manual review locally."""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib.util
import inspect
import json
import textwrap
import zipfile
from pathlib import Path

from temporal_hoi.inference.hoi_kaggle import (
    MAX_BUNDLE_BYTES,
    MAX_FILES,
    safe_member,
    sha256_file,
    validate_manifest,
    validate_requirements,
    wheel_distribution,
)

BUNDLE_SCHEMA = "vidhoi_pair_diagnostics_bundle_v1"
INPUT_KEYS = ("source_result", "protocol", "mapping", "review_manifest", "annotations")
SOURCES = (
    "src/temporal_hoi/__init__.py", "src/temporal_hoi/inference/__init__.py",
    "src/temporal_hoi/data/vidhoi.py", "src/temporal_hoi/evaluation/hoi.py",
    "src/temporal_hoi/evaluation/hoi_head_audit.py",
    "src/temporal_hoi/inference/hoi_static.py", "src/temporal_hoi/inference/hoi_kaggle.py",
    "src/temporal_hoi/inference/hoi_diagnostics.py", "src/temporal_hoi/data/hoi_head_cache.py",
    "src/temporal_hoi/models/hoi_head.py", "src/temporal_hoi/training/hoi_head_study.py",
    "src/temporal_hoi/evaluation/hoi_pair_diagnostics.py",
    "src/temporal_hoi/data/hoi_pair_review.py", "scripts/run_vidhoi_pair_diagnostics.py",
)


@functools.lru_cache(maxsize=1)
def _base_module():
    spec = importlib.util.spec_from_file_location(
        "_vidhoi_pair_base_packager", Path(__file__).with_name("prepare_vidhoi_kaggle.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_bytes(value):
    return _base_module()._json_bytes(value)


def _file(path, label):
    return _base_module()._plain_file(Path(path), label)


def _record(value):
    return _base_module()._record(value)


def collect_review_files(manifest_path: Path) -> dict:
    """Keep manifest-relative image/context paths and check every declared byte."""
    manifest_path = _file(manifest_path, "locked review manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("Review manifest needs a nonempty files dictionary")
    files = {}
    root = manifest_path.parent.resolve()
    for name, record in records.items():
        safe_member(name)
        path = root / name
        if (not path.resolve().is_relative_to(root) or path.resolve() == manifest_path.resolve()
                or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".json"}):
            raise ValueError(f"Unsafe or unsupported review file: {name}")
        _file(path, "review image/context")
        if (not isinstance(record, dict) or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")):
            raise ValueError(f"Review image/context hash or size mismatch: {name}")
        files[f"assets/review/{name}"] = path
    return files


def prepare(source_result, protocol, mapping, review_manifest, output, *, annotations,
            wheelhouse, pair_config=None, project_root=None) -> dict:
    """Build a private handoff; do not run models, upload files or label negatives."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new package directory; existing evidence is preserved")
    project = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    inputs = {"source_result": _file(source_result, "completed Kaggle source result"),
              "protocol": _file(protocol, "locked development protocol"),
              "mapping": _file(mapping, "canonical numeric mapping"),
              "review_manifest": _file(review_manifest, "locked review manifest"),
              "annotations": _file(annotations, "explicit review labels, initially unknown")}
    if not zipfile.is_zipfile(inputs["source_result"]):
        raise ValueError("The source result must be the original result ZIP")
    config = ({"schema_version": "vidhoi_pair_diagnostics_config_v1", "device": "cpu",
               "threads": 2} if pair_config is None else
              json.loads(_file(pair_config, "reviewed diagnostic config").read_text(encoding="utf-8")))
    if config.get("schema_version") != "vidhoi_pair_diagnostics_config_v1":
        raise ValueError("Require vidhoi_pair_diagnostics_config_v1")
    for name, path in inputs.items():
        expected = config.get(f"{name}_sha256")
        if expected is not None and expected != sha256_file(path):
            raise ValueError(f"Reviewed diagnostic input SHA256 differs: {name}")
    files = {name: _file(project / name, "pair diagnostic source") for name in SOURCES}
    for namespace in ("data", "evaluation", "models", "training"):
        files[f"src/temporal_hoi/{namespace}/__init__.py"] = (
            b'"""Minimal namespace for cached-feature pair diagnostics."""\n')
    files.update(collect_review_files(inputs["review_manifest"]))
    # Keep the original ZIP bytes/hash, using an opaque extension for Dataset transport.
    destinations = {"source_result": "source_result.bin", "protocol": "protocol.json",
                    "mapping": "class_mapping.json", "review_manifest": "review/review_manifest.json",
                    "annotations": "review/annotations.json"}
    for name, destination in destinations.items():
        files[f"assets/{destination}"] = inputs[name]
        config[name] = destination
        config[f"{name}_sha256"] = sha256_file(inputs[name])
    files["assets/run_config.json"] = _json_bytes(config)
    wheelhouse = Path(wheelhouse)
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise ValueError("An ordinary local offline wheelhouse is required")
    requirements = _file(wheelhouse / "requirements-offline.txt", "offline dependency pins")
    validate_requirements(requirements.read_text(encoding="utf-8"))
    files["wheelhouse/requirements-offline.txt"] = requirements
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise ValueError("Offline wheels are required")
    for wheel in wheels:
        _file(wheel, "offline wheel")
        wheel_distribution(wheel.name)
        files[f"wheelhouse/{wheel.name}"] = wheel
    files["BUNDLE_SCOPE.md"] = (
        "# Cached VidHOI pair diagnostics and manual review\n\n"
        "Read only the completed upstream-train oracle development source cache. "
        "Score all initial/last heads; do not retrain or encode new CLIP features. "
        "Report cross-runtime drift. Review images/context and labels are a separate "
        "manual development audit; blanks stay unknown, never automatic negatives. "
        "No independent final benchmark, zero-shot final or M2 improvement claim. "
        "Private handoff only; byte hashes do not grant redistribution rights.\n"
    ).encode()
    records = {name: _record(value) for name, value in sorted(files.items())}
    total = sum(record["bytes"] for record in records.values())
    if len(records) > MAX_FILES or total > MAX_BUNDLE_BYTES:
        raise ValueError("Pair diagnostic bundle exceeds existing file/byte limits")
    review = json.loads(inputs["review_manifest"].read_text(encoding="utf-8"))
    manifest = {"schema_version": BUNDLE_SCHEMA, "dataset": "VidHOI", "split": "train",
                "purpose": "head_pair_diagnostics", "box_source": "oracle",
                "object_label_source": "oracle", "training_run": False,
                "benchmark_evaluated": False, "official_final_split_accessed": False,
                "final_unseen_accessed": False, "entrypoint": "scripts/run_vidhoi_pair_diagnostics.py",
                "files": records, "total_bytes": total, "review_case_count": len(review.get("cases", [])),
                "source_result_member": "assets/source_result.bin",
                "source_result_sha256": config["source_result_sha256"]}
    manifest_bytes = _json_bytes(manifest)
    validate_manifest(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    runtime = (project / "src/temporal_hoi/inference/hoi_kaggle.py").read_text(encoding="utf-8")
    notebook_data = notebook(runtime, digest)
    output.mkdir(parents=True, exist_ok=False)
    bundle_path = output / "vidhoi_m1_pair_diagnostics_input.zip"
    with zipfile.ZipFile(bundle_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bundle_manifest.json", manifest_bytes)
        for name, value in sorted(files.items()):
            compression = zipfile.ZIP_STORED if name.startswith(("assets/source_result", "wheelhouse/")) else zipfile.ZIP_DEFLATED
            if isinstance(value, bytes):
                archive.writestr(name, value, compress_type=compression)
            else:
                archive.write(value, name, compress_type=compression)
    notebook_path = output / "vidhoi_m1_pair_diagnostics_kaggle.ipynb"
    notebook_path.write_bytes(_json_bytes(notebook_data))
    receipt = {"schema_version": "vidhoi_pair_diagnostics_kaggle_build_v1",
               "status": "local_bundle_prepared", "bundle": bundle_path.name,
               "bundle_sha256": sha256_file(bundle_path), "bundle_bytes": bundle_path.stat().st_size,
               "manifest_sha256": digest, "notebook": notebook_path.name,
               "notebook_sha256": sha256_file(notebook_path), "notebook_code_cells": 7,
               "review_case_count": manifest["review_case_count"],
               "source_result_member": manifest["source_result_member"],
               "source_result_sha256": config["source_result_sha256"], "files": records,
               "training_run": False, "encoder_inference_run": False, "head_inference_run": False,
               "official_final_split_accessed": False, "final_unseen_accessed": False, "uploaded": False}
    (output / "build_receipt.json").write_bytes(_json_bytes(receipt))
    (output / "HUONG_DAN.md").write_text(instructions(receipt), encoding="utf-8")
    return receipt


def validate_pair_export(run: Path, input_root: Path, *, allow_fixture: bool = False) -> dict:
    """Require complete cached scoring and a review export; unknown is a valid label state."""
    import hashlib
    import math
    import zipfile

    from temporal_hoi.data.hoi_pair_review import validate_review_annotations

    output, input_root = Path(run) / "pair_diagnostics", Path(input_root)

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def inventory(directory, listing):
        if not isinstance(listing, list) or not listing:
            raise RuntimeError("Missing pair diagnostic output inventory")
        names = set()
        for record in listing:
            name = record.get("path")
            safe_member(name)
            path = directory / name
            if (name in names or name == "receipt.json" or path.is_symlink()
                    or not path.resolve().is_relative_to(directory.resolve()) or not path.is_file()
                    or path.stat().st_size != record.get("bytes")
                    or sha256_file(path) != record.get("sha256")):
                raise RuntimeError(f"Pair diagnostic output hash/size mismatch: {name}")
            names.add(name)
        actual = set()
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise RuntimeError("Pair diagnostic output symlinks are forbidden")
            if path.is_file():
                actual.add(path.relative_to(directory).as_posix())
        if actual != names | {"receipt.json"}:
            raise RuntimeError("Pair diagnostic inventory differs from actual files")

    config = read(Path(run) / "run_config.json")
    packaged = read(input_root / "assets/run_config.json")
    expected = {}
    for name in ("source_result", "protocol", "mapping", "review_manifest", "annotations"):
        safe_member(packaged[name])
        path = input_root / "assets" / packaged[name]
        if path.is_symlink() or not path.resolve().is_relative_to(input_root.resolve()):
            raise RuntimeError("Unsafe pair diagnostic input")
        digest = sha256_file(path)
        expected[name] = digest
        if (packaged.get(f"{name}_sha256") != digest or config.get(f"{name}_sha256") != digest
                or Path(config[name]).resolve() != path.resolve()):
            raise RuntimeError(f"Pair diagnostic input identity mismatch: {name}")
    outer = read(output / "receipt.json")
    if (outer.get("schema_version") != "vidhoi_pair_diagnostic_run_v1"
            or outer.get("status") != "completed"
            or outer.get("level") != "cached_feature_pair_diagnostic"
            or outer.get("training_run") is not False
            or outer.get("encoder_inference_run") is not False
            or outer.get("head_inference_run") is not True
            or outer.get("official_final_split_accessed") is not False
            or outer.get("final_unseen_accessed") is not False
            or outer.get("input_hashes") != expected or outer.get("diagnosed_states") != 18):
        raise RuntimeError("Incomplete or mismatched cached pair diagnostic receipt")
    inventory(output, outer.get("files"))
    stages = outer.get("stage_receipts")
    if not isinstance(stages, dict) or set(stages) != {"diagnostics", "review"}:
        raise RuntimeError("Missing diagnostics/review stage receipts")
    stage_data = {}
    for stage, record in stages.items():
        path = output / stage / "receipt.json"
        if (record.get("path") != f"{stage}/receipt.json"
                or record.get("sha256") != sha256_file(path)
                or record.get("bytes") != path.stat().st_size):
            raise RuntimeError(f"Pair stage receipt hash mismatch: {stage}")
        stage_data[stage] = read(path)
        if stage_data[stage].get("status") != "completed":
            raise RuntimeError(f"Pair stage incomplete: {stage}")
        inventory(path.parent, stage_data[stage].get("files"))
    diagnostic = stage_data["diagnostics"]
    fixture = diagnostic.get("level") == "synthetic_pair_score_diagnostic"
    if fixture and not allow_fixture:
        raise RuntimeError("Fixture results cannot complete a real pair diagnostic notebook")
    if (diagnostic.get("schema_version") != "vidhoi_pair_score_diagnostics_v1"
            or diagnostic.get("level") != ("synthetic_pair_score_diagnostic" if fixture
                                           else "cached_oracle_initial_last_pair_diagnostic")
            or diagnostic.get("diagnosed_states") != 18
            or diagnostic.get("input_hashes") != expected
            or diagnostic.get("source_result_sha256") != expected["source_result"]
            or diagnostic.get("modes") != ["object", "union", "fusion"]
            or diagnostic.get("seeds") != [42, 43, 44]
            or diagnostic.get("states") != ["initial", "last"]
            or diagnostic.get("model_training_run") is not False
            or diagnostic.get("backbone_inference_run") is not False
            or diagnostic.get("head_inference_run") is not True
            or diagnostic.get("official_final_split_accessed") is not False
            or diagnostic.get("final_unseen_accessed") is not False
            or diagnostic.get("unknown_labels_treated_as_negative") is not False):
        raise RuntimeError("Invalid cached initial/last scoring evidence")
    validated = read(output / "diagnostics/source_receipt.json")
    if (validated.get("schema_version") != "vidhoi_pair_source_validation_v1"
            or validated.get("status") != "completed" or validated.get("test_fixture") is not fixture
            or validated.get("level") != ("synthetic_cached_pair_source_fixture" if fixture
                                         else "verified_real_cached_head_study_source")
            or validated.get("source_result_sha256") != expected["source_result"]
            or validated.get("all_member_crc_and_stage_inventory_hashes_verified") is not True
            or validated.get("num_head_states") != 18
            or validated.get("files_extracted_or_executed") is not False
            or validated.get("official_final_split_accessed") is not False
            or validated.get("final_unseen_accessed") is not False
            or validated.get("config_sha256") != diagnostic.get("source_config_sha256")):
        raise RuntimeError("Missing verified source-cache and saved head provenance")
    if not fixture:
        with zipfile.ZipFile(input_root / "assets" / packaged["source_result"]) as source_zip:
            cache_bytes = source_zip.read("study/cache/receipt.json")
            training_bytes = source_zip.read("study/training/receipt.json")
            source_cache, source_training = json.loads(cache_bytes), json.loads(training_bytes)
        import hashlib

        if (validated.get("source_cache_receipt_sha256") != hashlib.sha256(cache_bytes).hexdigest()
                or validated.get("source_training_receipt_sha256") != hashlib.sha256(training_bytes).hexdigest()
                or source_cache.get("level") != "vidhoi_head_frozen_clip_feature_cache"
                or source_cache.get("encoder", {}).get("test_fixture") is not False
                or source_training.get("level") != "oracle_frozen_feature_development_head_study"
                or source_cache.get("input_hashes") != diagnostic.get("source_input_hashes")
                or source_cache.get("input_hashes", {}).get("protocol") != expected["protocol"]
                or source_cache.get("input_hashes", {}).get("mapping") != expected["mapping"]):
            raise RuntimeError("Diagnostic source cache identity differs from the original result ZIP")
    required = {f"states/{mode}/seed_{seed}/{state}/{name}.json"
                for mode in ("object", "union", "fusion") for seed in (42, 43, 44)
                for state in ("initial", "last")
                for name in ("metrics", "known_positive_ranks", "predictions", "pair_budget_diagnostic")}
    listed = {record["path"] for record in diagnostic["files"]}
    if not required <= listed:
        raise RuntimeError("Missing one of the eighteen initial/last diagnostic states")
    summary = read(output / "diagnostics/summary.json")
    comparisons = summary.get("comparisons", [])
    if (summary.get("schema_version") != "vidhoi_pair_score_summary_v1"
            or len(comparisons) != 9
            or {(row.get("mode"), row.get("seed")) for row in comparisons} != {
                (mode, seed) for mode in ("object", "union", "fusion") for seed in (42, 43, 44)}
            or set(summary.get("aggregate", {})) != {"object", "union", "fusion"}
            or summary.get("new_training_or_configuration_selection") is not False
            or summary.get("unannotated_is_not_verified_negative") is not True
            or not isinstance(summary.get("cross_environment_policy"), str)):
        raise RuntimeError("Missing complete within-environment diagnostic summary")
    review_manifest = read(input_root / "assets" / packaged["review_manifest"])
    if outer.get("review_case_count") != len(review_manifest.get("cases", [])):
        raise RuntimeError("Manual review case count differs from locked input")
    review = stage_data["review"]
    if (review.get("schema_version") != "vidhoi_pair_review_result_v1"
            or review.get("level") != "train_only_actor_review_diagnostic"
            or review.get("input_hashes") != expected
            or review.get("source_result_sha256") != expected["source_result"]
            or review.get("review_manifest_sha256") != expected["review_manifest"]
            or review.get("model_training_run") is not False
            or review.get("encoder_inference_run") is not False
            or review.get("head_inference_run") is not True
            or review.get("unknown_labels_treated_as_negative") is not False
            or review.get("model_scores_hidden_in_review_HTML") is not True
            or review.get("official_final_split_accessed") is not False
            or review.get("final_unseen_accessed") is not False):
        raise RuntimeError("Review stage scope or provenance differs")
    if read(output / "review/review_manifest.json") != review_manifest:
        raise RuntimeError("Exported review manifest differs from locked input")
    labels = read(output / "review/vidhoi_pair_review_annotations.json")
    try:
        checked = validate_review_annotations(labels, review_manifest)
    except ValueError as error:
        raise RuntimeError(f"Review annotation provenance is invalid: {error}") from error
    if read(output / "review/annotation_validation.json") != checked:
        raise RuntimeError("Review annotation validation receipt differs")
    for key in ("manifest_fingerprint", "case_count", "actor_count", "review_labels_status",
                "positive_count", "confirmed_negative_count", "unknown_count",
                "eligible_contrast_case_count", "fully_labeled_case_count"):
        if review.get(key) != checked[key]:
            raise RuntimeError(f"Review label count/status differs: {key}")
    if (outer.get("confirmed_negative_count") != checked["confirmed_negative_count"]
            or outer.get("review_labels_status") != checked["review_labels_status"]):
        raise RuntimeError("Outer review status differs from actual human label contents")
    canonical = json.dumps(labels, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != review.get("annotation_content_sha256"):
        raise RuntimeError("Actual review labels differ from their canonical identity")
    annotation_source = review.get("annotation_source", {})
    if annotation_source.get("external") is True:
        raw = output / "review/annotations_original.json"
        if (sha256_file(raw) != annotation_source.get("sha256")
                or raw.stat().st_size != annotation_source.get("bytes") or read(raw) != labels):
            raise RuntimeError("External reviewer input was not preserved verbatim")
    elif (annotation_source.get("external") is not False
          or labels != read(input_root / "assets" / packaged["annotations"])):
        raise RuntimeError("Packaged annotations were silently changed")
    scores = read(output / "review/actor_scores_after_review.json")
    states = scores.get("states", [])
    wanted = {(m, s, v) for m in ("object", "union", "fusion")
              for s in (42, 43, 44) for v in ("initial", "last")}
    if (scores.get("schema_version") != "vidhoi_actor_review_scores_v1"
            or len(states) != 18
            or {(s.get("mode"), s.get("seed"), s.get("state")) for s in states} != wanted):
        raise RuntimeError("Review actor scores lack eighteen fixed states")
    wanted_cases = {c["case_id"]: {a["track_id"] for a in c["actors"]}
                    for c in review_manifest["cases"]}
    case_metadata = {c["case_id"]: c for c in review_manifest["cases"]}
    for state in states:
        cases = state.get("cases", [])
        if len(cases) != len(wanted_cases) or {c.get("case_id") for c in cases} != set(wanted_cases):
            raise RuntimeError("Review score cases differ from annotation cases")
        for case in cases:
            locked = case_metadata[case["case_id"]]
            if any(case.get(key) != locked.get(key) for key in
                   ("video_id", "predicate_id", "object_track_id")):
                raise RuntimeError("Review actor scores refer to a different interaction query")
            actors = case.get("actors", [])
            if (len(actors) != len(wanted_cases[case["case_id"]])
                    or {a.get("track_id") for a in actors} != wanted_cases[case["case_id"]]
                    or any(type(a.get("score")) not in (int, float)
                           or not math.isfinite(a["score"]) or a["score"] < 0 for a in actors)):
                raise RuntimeError("Review actor score identity or numeric values differ")
    contrasts = read(output / "review/confirmed_actor_contrasts.json")
    if (contrasts.get("schema_version") != "vidhoi_review_actor_contrasts_v1"
            or contrasts.get("unknown_labels_used_as_negative") is not False
            or contrasts.get("benchmark_metric") is not False):
        raise RuntimeError("Actor contrast output changed unknown/benchmark semantics")
    if checked["eligible_contrast_case_count"] == 0:
        if (contrasts.get("status") != "pending_verified_contrasts" or contrasts.get("records") != []
                or any(row.get("strict_win_fraction") is not None or row.get("mean_margin") is not None
                       for row in contrasts.get("aggregate", {}).values())):
            raise RuntimeError("Missing confirmed actor contrasts cannot yield invented metrics")
    else:
        expected_comparisons = []
        human_labels = {c["case_id"]: {a["track_id"]: a["label"] for a in c["actors"]}
                        for c in labels["cases"]}
        for state in states:
            for case in state["cases"]:
                positive = [a for a in case["actors"] if human_labels[case["case_id"]][a["track_id"]] == "positive"]
                negative = [a for a in case["actors"] if human_labels[case["case_id"]][a["track_id"]] == "confirmed_negative"]
                for pos in positive:
                    for neg in negative:
                        margin = float(pos["score"] - neg["score"])
                        expected_comparisons.append({"mode": state["mode"], "seed": state["seed"],
                            "state": state["state"], "case_id": case["case_id"],
                            "positive_track_id": pos["track_id"], "negative_track_id": neg["track_id"],
                            "score_margin": margin, "strict_win": margin > 0, "exact_tie": margin == 0,
                            "near_tie": abs(margin) <= 1e-6})
        if (contrasts.get("status") != "evaluated_human_asserted_pairs"
                or contrasts.get("near_tie_tolerance") != 1e-6
                or contrasts.get("records") != expected_comparisons):
            raise RuntimeError("Confirmed contrast scores differ from actual human labels")
    aggregate = contrasts.get("aggregate", {})
    aggregate_keys = {f"{m}/{s}" for m in ("object", "union", "fusion") for s in ("initial", "last")}
    if set(aggregate) != aggregate_keys:
        raise RuntimeError("Actor contrast aggregate is incomplete")
    for key, value in aggregate.items():
        mode, state_name = key.split("/")
        rows = [r for r in contrasts["records"] if r["mode"] == mode and r["state"] == state_name]
        if value.get("seed_evaluations") != 3 or value.get("num_pair_comparisons_across_seeds") != len(rows):
            raise RuntimeError("Actor contrast aggregate count differs")
        expected_values = {
            "strict_win_fraction": sum(r["strict_win"] for r in rows) / len(rows) if rows else None,
            "near_tie_fraction": sum(r["near_tie"] for r in rows) / len(rows) if rows else None,
            "mean_margin": sum(r["score_margin"] for r in rows) / len(rows) if rows else None}
        for name, target in expected_values.items():
            actual = value.get(name)
            if ((target is None and actual is not None)
                    or (target is not None and (type(actual) not in (int, float)
                        or not math.isclose(actual, target, rel_tol=1e-12, abs_tol=1e-12)))):
                raise RuntimeError("Actor contrast aggregate differs from its verified pairs")
    return outer


def notebook(runtime_source: str, manifest_sha256: str) -> dict:
    """Seven cells: setup/deps/environment/preflight/scoring/manual review/export."""
    result = _base_module()._notebook(runtime_source, manifest_sha256)
    first = result["cells"][0]["source"]
    replacements = {
        'BASE = WORK / "vidhoi_static_probe"': 'BASE = WORK / "vidhoi_m1_pair_diagnostics"',
        'RUN = BASE / ("oracle_train_" + run_id)': 'RUN = BASE / ("pair_diagnostics_" + run_id)',
        "CELL_1_STARTED — VidHOI train-only oracle probe":
            "CELL_1_STARTED — VidHOI cached pair diagnostics and manual review",
        "INPUT_VERIFIED. Small train probe; no training, final-val or M2 retrieval.":
            "INPUT_VERIFIED. Cached head diagnostics; manual labels remain unknown until reviewed.",
    }
    for old, new in replacements.items():
        if first.count(old) != 1:
            raise ValueError("Base bootstrap changed; review pair notebook before packaging")
        first = first.replace(old, new, 1)
    result["cells"][0]["source"] = first
    cells = result["cells"][:3]

    def add(source):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {}, "source": source,
                      "execution_count": None, "outputs": []})

    for index, stage in enumerate(("preflight", "diagnose", "review"), 4):
        annotation_input = ', "--annotation-input-root", str(INPUTS)' if stage == "review" else ""
        add(f'''
            print("CELL_{index}_STARTED — pair diagnostic {stage}", flush=True)
            try:
                verify_installed_input(ROOT, PINNED_MANIFEST)
                REC.command("pair_{stage}", [sys.executable, "-u",
                            str(ROOT / "scripts/run_vidhoi_pair_diagnostics.py"),
                            "--config", str(CONFIG), "--output-dir", str(RUN / "pair_diagnostics"),
                            "--stage", "{stage}"{annotation_input}], timeout_seconds=3600)
            except BaseException as error:
                REC.event("pair_{stage}_failed", error=str(error))
                REC.snapshot()
                raise
        ''')
    validator = inspect.getsource(validate_pair_export)
    add(validator + '''

print("CELL_7_STARTED — verify cached diagnostics and export manual review", flush=True)
try:
    verify_installed_input(ROOT, PINNED_MANIFEST)
    sys.path.insert(0, str(ROOT / "src"))
    pair_receipt = validate_pair_export(RUN, ROOT)
except BaseException as error:
    REC.event("export_incomplete", error=str(error))
    REC.snapshot()
    raise
REC.event("completed", level=pair_receipt["level"], training_run=False,
          encoder_inference_run=False, head_inference_run=True,
          official_final_split_accessed=False, final_unseen_accessed=False,
          review_labels_status=pair_receipt["review_labels_status"],
          confirmed_negative_count=pair_receipt["confirmed_negative_count"],
          limits="Cached initial/last head diagnostics with explicit review status; no training, final benchmark or M2 gain.")
RESULT_ZIP = REC.snapshot()
from IPython.display import FileLink, display
review_status = json.loads((RUN / "pair_diagnostics/review/receipt.json").read_text(encoding="utf-8"))
print("REVIEW_LABELS_STATUS:", review_status["review_labels_status"],
      "positive=", review_status["positive_count"],
      "confirmed_negative=", review_status["confirmed_negative_count"],
      "unknown=", review_status["unknown_count"], flush=True)
print("Download ZIP, extract it, then open the review HTML and save reviewed labels.", flush=True)
display(FileLink(str(RUN / "pair_diagnostics/review/actor_review.html")), FileLink(str(RESULT_ZIP)))
''')
    result["cells"] = cells
    if len(cells) != 7:
        raise ValueError("Pair diagnostic notebook must have seven cells")
    return result


def instructions(receipt):
    return f'''# Chẩn đoán cặp và kiểm tra nhãn trên Kaggle

1. Import `{receipt["notebook"]}`; tên notebook **VidHOI M1 Pair Diagnostics v2**.
2. Tạo Dataset riêng tư **VidHOI M1 Pair Diagnostics Input v2** chứa
   `{receipt["bundle"]}`, đợi xử lý hoàn tất, rồi **Add Input**.
   Khi thay gói cũ bị lỗi, cập nhật cả notebook và Dataset input cùng bản này;
   bỏ Input cũ để tránh dùng nhầm phiên bản. Dataset đã mở ZIP ngoài phải có
   `assets/source_result.bin` với kích thước/hash đúng manifest.
3. Internet Off. Có thể dùng T4 như lượt trước để giảm khác biệt phần cứng;
   CPU vẫn chạy được vì chỉ dùng cache và head nhỏ, không mã hóa CLIP/học lại.
4. **Save Version → Save & Run All** chạy đủ 7 cell. Tải
   `pair_diagnostics_<thời gian>_<mã>.zip` ở cell 7 hoặc mục Output.
5. Giải nén ZIP, mở HTML trong thư mục `pair_diagnostics/review`, xem đầy đủ
   ảnh/ngữ cảnh, đánh dấu rồi tải `vidhoi_pair_review_annotations.json`.
6. Giữ đúng tên JSON này, thêm nó qua một Dataset riêng tư mới bằng **Add Input**
   vào cùng notebook, giữ nguyên Input gốc. **Save Version → Save & Run All** lần
   nữa sẽ tự nhận nhãn, kiểm tra bằng chứng và xuất kết quả review. Nếu có nhiều
   tệp cùng tên, bỏ Input nhãn cũ để tránh nhầm phiên bản. Gửi lại ZIP kết quả
   cùng JSON nhãn; lượt này vẫn không học lại mô hình.

Có {receipt["review_case_count"]} trường hợp review. Chưa chắc/không đủ ngữ cảnh
thì giữ **unknown**. Save & Run All không tự biến mục chưa đánh dấu thành negative,
không tự tạo nhãn thay người xem. Xuất review còn chờ nhãn là trạng thái hợp lệ;
không có training mới hoặc kết luận cải thiện M2 từ lượt này.

Notebook chẩn đoán đủ 3 mode × 3 seed × initial/last (18 trạng thái), giữ kết
quả gốc và báo sai khác với lượt Kaggle trước theo môi trường thực tế. Nhãn
review là kiểm tra development riêng; không thay ground truth/evaluator của
benchmark và không mở official final-val/final-unseen.

Gói đã có source result/cache/head, code, ảnh review/ngữ cảnh và wheels Linux
CPython 3.11/3.12/3.13. Không cần original CLIP weights hoặc tải thêm dataset.
File `assets/source_result.bin` chứa nguyên byte ZIP kết quả nguồn, giữ đúng
SHA256; chỉ đổi đuôi đóng gói để tránh xử lý ZIP lồng theo phần mở rộng.
Không tự nén lại thư mục kết quả vì ZIP dựng lại sẽ không có hash gốc.
Cell 7 kiểm lại input ghim và thêm thư mục source vào kernel trước khi import
bộ kiểm nhãn; PYTHONPATH của tiến trình chấm điểm không tự cấu hình kernel.
Mỗi Run All lưu kết quả mới; nếu lỗi, giữ ZIP log partial để kiểm tra.
Input manifest SHA256: `{receipt["manifest_sha256"]}`.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUT_KEYS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--pair-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.source_result, args.protocol, args.mapping, args.review_manifest,
                      args.output, annotations=args.annotations, wheelhouse=args.wheelhouse,
                      pair_config=args.pair_config)
    print(json.dumps({key: receipt[key] for key in ("status", "bundle", "bundle_bytes",
                                                  "review_case_count", "training_run", "uploaded")}))


if __name__ == "__main__":
    main()

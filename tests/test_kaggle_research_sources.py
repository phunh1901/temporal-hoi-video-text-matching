"""Public Kaggle exports preserve captured source bytes without exposing run data."""

import ast
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/kaggle/msrvtt"
SPEC = importlib.util.spec_from_file_location("public_kaggle_sources", SOURCE / "prepare_sources.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_captured_common_source_and_training_evaluation_pins_agree():
    provenance = PREPARE.verify_sources(SOURCE)
    core_hash = "94fd6527fa59cb7eae5c884444a02ff25eda753d399c821cddeddbfde6c2dd64"
    assert provenance["files"]["retrieval_core.py"]["sha256"] == core_hash
    for stage in ("pooling", "training", "evaluation"):
        assert provenance["stages"][stage]["original_source_sha256"]["retrieval_core.py"] == core_hash
    assert provenance["stages"]["pooling"]["run_train"] is False
    assert provenance["stages"]["training"]["run_train"] is True
    assert provenance["stages"]["evaluation"]["run_train"] is False


def test_restore_vendor_names_and_reject_existing_corruption(tmp_path):
    PREPARE.restore_sources(SOURCE, tmp_path / "code")
    restored = tmp_path / "code/vendor/module_clip.py"
    expected = PREPARE.verify_sources(SOURCE)["files"]["vendor/module_clip.py"]["sha256"]
    assert hashlib.sha256(restored.read_bytes()).hexdigest() == expected
    restored.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing overwrite"):
        PREPARE.restore_sources(SOURCE, tmp_path / "code")


def test_tampered_published_source_fails_before_creating_working_code(tmp_path):
    copied = tmp_path / "published"
    shutil.copytree(SOURCE, copied, ignore=shutil.ignore_patterns("__pycache__"))
    (copied / "retrieval_core.py").write_text("raise RuntimeError('must not run')", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        PREPARE.restore_sources(copied, tmp_path / "working")
    assert not (tmp_path / "working").exists()


@pytest.mark.parametrize("stage", ["pooling", "training", "evaluation"])
def test_private_inputs_missing_never_create_a_success_capture(tmp_path, stage):
    code = tmp_path / "code"
    PREPARE.restore_sources(SOURCE, code)
    with pytest.raises(FileNotFoundError, match="Attach the original input"):
        PREPARE.capture_stage(SOURCE, code, tmp_path / "run", stage, tmp_path / "inputs")
    assert not (tmp_path / "run").exists()


def test_notebooks_are_parseable_adaptations_without_outputs_or_embedded_payloads():
    for path in SOURCE.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["cells"][0]["source"].startswith("# MSR-VTT")
        assert "has not been rerun" in notebook["cells"][0]["source"]
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None and cell["outputs"] == []
            tree = ast.parse(cell["source"])
            assert not any(isinstance(node, ast.Constant) and isinstance(node.value, str)
                           and len(node.value) > 4096 for node in ast.walk(tree))
        config_cell = notebook["cells"][2]["source"]
        scope = {"base": Path("/example")}
        exec(compile(config_cell, str(path), "exec"), scope)
        assert scope["CONFIG"]["RUN_TRAIN"] is (path.name == "train_b3.ipynb")


def test_no_dataset_weights_results_or_raw_input_manifests_in_public_bundle():
    forbidden = {".csv", ".pt", ".pth", ".npz", ".npy", ".zip", ".mp4", ".gz"}
    for path in SOURCE.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        assert path.suffix not in forbidden
        assert path.name not in {"evaluation_input_manifest.json", "msrvtt_train8100_v2.manifest.json"}

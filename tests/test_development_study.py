"""Preparation must preserve existing work and never imply real experiment completion."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_cli(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "development_study_cli", ROOT / "scripts/prepare_development_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_preserves_budget_and_never_creates_fake_results(tmp_path, monkeypatch):
    cli = load_cli(monkeypatch)
    spec = json.loads((ROOT / "configs/development_head_study.json").read_text())
    spec.update(base_config=str(ROOT / spec["base_config"]),
                union_config=str(ROOT / spec["union_config"]))
    monkeypatch.setattr(cli, "alignment_readiness", lambda config: (None, {
        "status": "blocked", "blockers": ["fixture_missing_real_inputs"]}))
    output = tmp_path / "study"
    result = cli.prepare_study(spec, output)
    assert len(result["runs"]) == 6
    assert not result["benchmark_training_run"] and not result["model_selected"]
    assert not list(output.rglob("*.pt"))
    assert all(row["benchmark_metrics"] is None and row["synthetic_probe"] is None
               for row in result["runs"])
    base = json.loads(Path(spec["base_config"]).read_text())
    for run in result["runs"]:
        config = json.loads(Path(run["config"]).read_text())
        for key in ("lr", "max_epochs", "patience", "feature_batch_videos", "cache_dir",
                    "metadata_lock", "max_rss_bytes", "max_train_hours", "selection_metric"):
            assert config[key] == base[key]
    matrix = json.loads((output / "ablation_matrix.json").read_text())
    assert [row["condition"] for row in matrix["conditions"]] == ["C00", "C10", "C01", "C11"]
    assert all(row["status"] == "not_run" and row["metrics"] is None
               for row in matrix["conditions"])
    before = (output / "study_summary.json").read_bytes()
    with pytest.raises(ValueError, match="preserved"):
        cli.prepare_study(spec, output)
    assert (output / "study_summary.json").read_bytes() == before


def test_probe_inputs_fixed_across_head_and_optimizer_seed(tmp_path, monkeypatch):
    cli = load_cli(monkeypatch)
    base = json.loads((ROOT / "configs/frozen_feature_alignment.json").read_text())
    hashes = []
    for head, seed in (("linear", 42), ("linear", 43), ("mlp", 44)):
        config = {**base, "head": head, "hidden_dim": 8, "seed": seed}
        result = cli.run_synthetic_probe(config, tmp_path / f"{head}{seed}", steps=1)
        hashes.append(result["synthetic_feature_sha256"])
        assert result["completed_probe_steps"] == 1 and result["resume_extra_steps"] == 1
        assert not result["benchmark_training_run"]
    assert len(set(hashes)) == 1

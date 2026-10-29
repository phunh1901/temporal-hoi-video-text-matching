"""Synthetic transport regressions; these do not claim an actual Kaggle upload/run."""

import hashlib
import json
import zipfile

import pytest
from test_vidhoi_pair_diagnostics_kaggle import assets as assets
from test_vidhoi_pair_diagnostics_kaggle import packager

from temporal_hoi.inference import hoi_kaggle as runtime

OPAQUE_MEMBER = "assets/source_result.bin"
LEGACY_MEMBER = "assets/source_result.zip"


@pytest.fixture
def transport_bundle(assets, tmp_path):
    directory = tmp_path / "transport_package"
    receipt = packager().prepare(**assets, output=directory)
    return directory / receipt["bundle"], receipt


def _payload(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _write_expanded(root, payload):
    for name, data in payload.items():
        runtime.safe_member(name)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _simulate_extension_based_archive_expansion(root):
    """Model a service expanding .zip members and removing their archive bytes."""
    for path in sorted(root.rglob("*.zip")):
        child = path.with_suffix("")
        with zipfile.ZipFile(path) as archive:
            _write_expanded(child, {name: archive.read(name) for name in archive.namelist()})
        path.unlink()


def _encode_manifest(manifest):
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _legacy_payload(archive_path):
    payload = _payload(archive_path)
    manifest = json.loads(payload[runtime.MANIFEST_NAME])
    payload[LEGACY_MEMBER] = payload.pop(OPAQUE_MEMBER)
    manifest["files"][LEGACY_MEMBER] = manifest["files"].pop(OPAQUE_MEMBER)
    manifest.pop("source_result_member")
    config = json.loads(payload["assets/run_config.json"])
    config["source_result"] = "source_result.zip"
    payload["assets/run_config.json"] = _encode_manifest(config)
    manifest["files"]["assets/run_config.json"] = {
        "bytes": len(payload["assets/run_config.json"]),
        "sha256": hashlib.sha256(payload["assets/run_config.json"]).hexdigest(),
    }
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"].values())
    payload[runtime.MANIFEST_NAME] = _encode_manifest(manifest)
    return payload


@pytest.mark.parametrize("layout", ["archive", "expanded_with_nested_zip_processing"])
def test_opaque_source_preserves_original_zip_bytes(assets, transport_bundle, tmp_path, layout):
    archive_path, build = transport_bundle
    payload = _payload(archive_path)
    manifest = runtime.validate_manifest(payload[runtime.MANIFEST_NAME])
    original = assets["source_result"].read_bytes()
    assert manifest["source_result_member"] == OPAQUE_MEMBER
    assert OPAQUE_MEMBER in manifest["files"] and LEGACY_MEMBER not in manifest["files"]
    assert payload[OPAQUE_MEMBER] == original
    assert manifest["source_result_sha256"] == hashlib.sha256(original).hexdigest()
    if layout == "archive":
        inputs = archive_path.parent
    else:
        inputs = tmp_path / "expanded_input"
        _write_expanded(inputs, payload)
        _simulate_extension_based_archive_expansion(inputs)
        assert (inputs / OPAQUE_MEMBER).is_file()
    installed = tmp_path / "installed"
    receipt = runtime.install_input(inputs, installed, build["manifest_sha256"])
    assert receipt["format"] == ("zip" if layout == "archive" else "directory")
    assert (installed / OPAQUE_MEMBER).read_bytes() == original
    assert zipfile.is_zipfile(installed / OPAQUE_MEMBER)
    runtime.verify_installed_input(installed, build["manifest_sha256"])
    config_path = runtime.prepare_run_config(installed, tmp_path / "run/config.json", device="cpu")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["source_result"].endswith("source_result.bin")
    assert config["source_result_sha256"] == hashlib.sha256(original).hexdigest()


def test_legacy_manifest_without_member_remains_accepted(assets, transport_bundle, tmp_path):
    archive_path, _ = transport_bundle
    payload = _legacy_payload(archive_path)
    assert "source_result_member" not in runtime.validate_manifest(payload[runtime.MANIFEST_NAME])
    inputs = tmp_path / "legacy_input"
    _write_expanded(inputs, payload)
    pin = hashlib.sha256(payload[runtime.MANIFEST_NAME]).hexdigest()
    installed = tmp_path / "legacy_installed"
    runtime.install_input(inputs, installed, pin)
    assert (installed / LEGACY_MEMBER).read_bytes() == assets["source_result"].read_bytes()


def test_nested_expansion_reproduces_legacy_missing_member_failure(transport_bundle, tmp_path):
    archive_path, _ = transport_bundle
    payload = _legacy_payload(archive_path)
    inputs = tmp_path / "legacy_expanded_input"
    _write_expanded(inputs, payload)
    _simulate_extension_based_archive_expansion(inputs)
    assert not (inputs / LEGACY_MEMBER).exists()
    assert (inputs / "assets/source_result/explicit_test_fixture.json").is_file()
    pin = hashlib.sha256(payload[runtime.MANIFEST_NAME]).hexdigest()
    with pytest.raises(ValueError, match="Missing/size-mismatched expanded input: assets/source_result.zip"):
        runtime.install_input(inputs, tmp_path / "rejected", pin)
    assert not any((tmp_path / "rejected").iterdir())


@pytest.mark.parametrize("mutation", ["missing", "truncated", "changed_same_size"])
def test_missing_or_altered_opaque_source_rejected(transport_bundle, tmp_path, mutation):
    archive_path, build = transport_bundle
    inputs = tmp_path / "altered_input"
    _write_expanded(inputs, _payload(archive_path))
    source = inputs / OPAQUE_MEMBER
    original = source.read_bytes()
    if mutation == "missing":
        source.unlink()
    elif mutation == "truncated":
        source.write_bytes(original[:-1])
    else:
        source.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="(Missing/size-mismatched expanded input|Input hash/size mismatch)"):
        runtime.install_input(inputs, tmp_path / "rejected", build["manifest_sha256"])
    if mutation in {"missing", "truncated"}:
        assert not any((tmp_path / "rejected").iterdir())


@pytest.mark.parametrize("mutation", ["unapproved", "wrong_declared", "both", "no_declaration"])
def test_incompatible_or_ambiguous_source_transport_rejected(transport_bundle, mutation):
    archive_path, _ = transport_bundle
    payload = _payload(archive_path)
    manifest = json.loads(payload[runtime.MANIFEST_NAME])
    if mutation == "unapproved":
        manifest["source_result_member"] = "assets/source_result.dat"
    elif mutation == "wrong_declared":
        manifest["source_result_member"] = LEGACY_MEMBER
    elif mutation == "both":
        manifest["files"][LEGACY_MEMBER] = manifest["files"][OPAQUE_MEMBER].copy()
        manifest["total_bytes"] += manifest["files"][LEGACY_MEMBER]["bytes"]
    else:
        manifest.pop("source_result_member")
    with pytest.raises(ValueError):
        runtime.validate_manifest(_encode_manifest(manifest))

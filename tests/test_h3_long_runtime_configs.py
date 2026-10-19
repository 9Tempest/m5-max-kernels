"""CPU-only pin checks for the VAE configs used by the first-frame request."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / "integrations/h3"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "h3_long_runtime_configs", DIRECTORY / "run_long_benchmark.py"
)
long = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(long)


@pytest.fixture
def configs(tmp_path, monkeypatch):
    model, manifest = tmp_path / "model", tmp_path / "manifest.json"
    entries = []
    for component in ("video_vae", "audio_vae"):
        name = f"FL2VA/{component}/config.json"
        payload = b'{"latents_mean":[0.0],"latents_std":[1.0]}\n'
        path = model / name
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        entries.append(
            dict(
                path=name,
                type="file",
                size=len(payload),
                oid=hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
            )
        )
    manifest.write_text(json.dumps(entries))
    monkeypatch.setattr(long.base, "MODEL_MANIFEST_SHA256", long.base.sha256(manifest))
    return model, manifest, entries


def test_both_runtime_normalization_configs_are_verified_and_recorded(configs):
    model, manifest, entries = configs
    result = long.runtime_config_inventory(model, manifest)
    assert result["model_manifest_sha256"] == long.base.sha256(manifest)
    assert len(result["files"]) == 2
    for record, expected in zip(result["files"], entries, strict=True):
        assert record == dict(
            path=expected["path"],
            bytes=expected["size"],
            sha256=long.base.sha256(model / expected["path"]),
            expected=expected,
        )


@pytest.mark.parametrize("component", ["video_vae", "audio_vae"])
@pytest.mark.parametrize("field", ["latents_mean", "latents_std"])
def test_same_size_normalization_change_is_rejected(configs, component, field):
    model, manifest, _ = configs
    path = model / f"FL2VA/{component}/config.json"
    payload = path.read_bytes()
    old = b"0.0" if field == "latents_mean" else b"1.0"
    path.write_bytes(payload.replace(old, b"2.0"))
    assert path.stat().st_size == len(payload)
    with pytest.raises(ValueError, match="Git blob mismatch"):
        long.runtime_config_inventory(model, manifest)


@pytest.mark.parametrize("component", ["video_vae", "audio_vae"])
def test_missing_runtime_config_is_rejected(configs, component):
    model, manifest, _ = configs
    (model / f"FL2VA/{component}/config.json").unlink()
    with pytest.raises(FileNotFoundError):
        long.runtime_config_inventory(model, manifest)


def test_changed_model_inventory_is_rejected_before_accepting_configs(configs):
    model, manifest, _ = configs
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned model manifest"):
        long.runtime_config_inventory(model, manifest)

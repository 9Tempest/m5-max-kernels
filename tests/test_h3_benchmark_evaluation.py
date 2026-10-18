"""CPU fixtures for BF16/prepared-H256 provenance across different loader pins."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_h3_latent_evaluation import captures as captures, latent as fixture_capture


DIRECTORY = Path(__file__).resolve().parents[1] / "integrations/h3"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("h3_pair", DIRECTORY / "evaluate_benchmark_pair.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


@pytest.fixture
def paired(captures, monkeypatch):
    monkeypatch.setattr(pair, "capture", fixture_capture)
    entry = pair.catalog.load_benchmark("person-turn-v1")
    original = dict(repository="fixture", revision="pinned", model_manifest_sha256="digest")
    manifest = dict(
        schema="h3_prepared_model_v1",
        source_model=original,
        original_checkpoint_retained=True,
        policy_id=pair.POLICY_ID,
        producer=dict(
            source_sha256=fixture_capture.SOURCE_SHA256, patch_sha256=fixture_capture.PATCH_SHA256
        ),
        prepared_manifest_sha256="b" * 64,
    )
    manifest_path = captures.hadamard / "derived-checkpoint.json"
    manifest_path.write_text(json.dumps(manifest))
    digest = pair.metrics.sha256(manifest_path)
    for directory in (captures.bf16, captures.hadamard):
        path = directory / "run.json"
        run = json.loads(path.read_text())
        run.update(
            {
                key: entry[key]
                for key in (
                    "seed",
                    "prompt",
                    "width",
                    "height",
                    "seconds",
                    "fps",
                    "steps",
                    "active_dit_blocks",
                    "reuse",
                    "core_reuse",
                    "reference_rope",
                )
            }
        )
        run.update(benchmark=entry["id"], catalog=entry)
        meta_path = directory / "latents/latent-metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["seed"] = entry["seed"]
        meta_path.write_text(json.dumps(meta))
        run["latent_validation"].update(
            metadata=meta, metadata_sha256=pair.metrics.sha256(meta_path)
        )
        if directory == captures.hadamard:
            run.update(
                prepared_checkpoint=True,
                patch=dict(
                    source_sha256=pair.PREPARED_SOURCE_SHA256,
                    patch_sha256=pair.PREPARED_PATCH_SHA256,
                ),
                model=dict(
                    manifest_sha256=digest, source_model=original, prepared_manifest_sha256="b" * 64
                ),
                derived_checkpoint=dict(
                    manifest_file="derived-checkpoint.json",
                    manifest_sha256=digest,
                    source_model=original,
                    producer=manifest["producer"],
                    policy_id=pair.POLICY_ID,
                    prepared_manifest_sha256="b" * 64,
                ),
                runtime_overrides={"H3_PREPARED_DIT_MANIFEST_SHA256": "b" * 64},
                prepared_load=dict(projections=200, weight_rotation_quantization_seconds=0.0),
                measurements={"generation": {"returncode": 0, "fallback_detected": False}},
            )
        path.write_text(json.dumps(run))
    captures.media = False
    captures.media_evaluation = None
    return captures


def mutate(path, action):
    value = json.loads(path.read_text())
    action(value)
    path.write_text(json.dumps(value))


def test_approved_different_loaders_pair_successfully(paired):
    result = pair.execute(paired)
    assert result["status"] == "completed"
    assert len(result["latents"]) == 42
    assert result["latents"]["final_video"]["metrics"]["relative_l2"]["value"] == pytest.approx(0.1)
    assert result["quality_decision"] == "not_evaluated"
    assert (paired.output_dir / "bf16.npz").exists()
    with pytest.raises(FileExistsError):
        pair.execute(paired)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["patch"].update(source_sha256=fixture_capture.SOURCE_SHA256),
        lambda r: r.update(prompt="different prompt"),
        lambda r: r.update(seed=72),
        lambda r: r["model"]["source_model"].update(revision="different"),
        lambda r: r["derived_checkpoint"].update(manifest_sha256="a" * 64),
        lambda r: r["derived_checkpoint"].update(policy_id="different"),
        lambda r: r["runtime_overrides"].update(H3_PREPARED_DIT_MANIFEST_SHA256="c" * 64),
        lambda r: r["prepared_load"].update(projections=199),
        lambda r: r["prepared_load"].update(weight_rotation_quantization_seconds=1),
        lambda r: r["measurements"]["generation"].update(fallback_detected=True),
    ],
)
def test_policy_recipe_manifest_loader_mismatch_fails(paired, change):
    mutate(paired.hadamard / "run.json", change)
    result = pair.execute(paired)
    assert result["status"] == "failed"
    assert (paired.output_dir / "benchmark-evaluation.json").exists()


def test_modified_derived_manifest_is_not_trusted(paired):
    path = paired.hadamard / "derived-checkpoint.json"
    path.write_bytes(path.read_bytes() + b"\n")
    assert "SHA" in pair.execute(paired)["error"]


def test_unrelated_media_record_cannot_be_joined(paired):
    media = paired.output_dir.parent / "media.json"
    media.write_text(json.dumps(dict(status="completed", harness_sha256="wrong")))
    paired.media_evaluation = media
    assert "media join" in pair.execute(paired)["error"]

"""Small CPU fixtures for raw capture provenance and paired latent evaluation."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / "integrations/h3"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location(
    "h3_latent_evaluation", DIRECTORY / "evaluate_latents.py"
)
latent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latent)


@pytest.fixture
def captures(tmp_path, monkeypatch):
    monkeypatch.setattr(
        latent,
        "SHAPES",
        {
            "per_step_video_shape": [2, 2],
            "per_step_audio_shape": [2, 1],
            "final_video_shape": [1, 2, 2],
            "final_audio_shape": [1, 2, 1],
        },
    )
    paths = {}
    for label, mode in latent.MODES.items():
        path = tmp_path / label
        (path / "latents").mkdir(parents=True)
        meta = dict(
            schema=1,
            dtype="float32_little_endian",
            seed=42,
            completed_steps=20,
            scope="diagnostic_CPU_readback_after_completed_GPU_Euler",
            final_layout="unpacked_contiguous_VAE_input_before_decoder",
            per_step_layout="packed_model_rows_after_Euler_before_unpack",
            final_video_sigma=0,
            final_audio_sigma=0,
            int8_mlp=int(mode != "bf16"),
            int8_qkv=int(mode != "bf16"),
            int8_attention_out=int(mode != "bf16"),
            hadamard_fc2=int(mode == "hadamard-fc2"),
            finite=True,
            **latent.SHAPES,
        )
        meta_path = path / "latents/latent-metadata.json"
        meta_path.write_text(json.dumps(meta))
        tensors = []
        for name, _, shape in latent.tensor_spec():
            tensor_path = path / "latents" / name
            (
                np.ones(shape, dtype="<f4") * {"bf16": 1, "native": 1.2, "hadamard": 1.1}[label]
            ).tofile(tensor_path)
            tensors.append(
                dict(
                    file=name,
                    elements=int(np.prod(shape)),
                    sha256=latent.metrics.sha256(tensor_path),
                    finite=True,
                )
            )
        run = dict(
            status="completed",
            mode=mode,
            capture_latents=True,
            seed=42,
            benchmark="fixture",
            prompt="same prompt",
            width=512,
            height=512,
            seconds=5,
            steps=20,
            active_dit_blocks=50,
            reuse=1,
            core_reuse=1,
            reference_rope=True,
            patch=dict(patch_sha256=latent.PATCH_SHA256, source_sha256=latent.SOURCE_SHA256),
            model=dict(repository="fixture", revision="pinned", manifest_sha256="digest"),
            latent_validation=dict(
                metadata=meta, metadata_sha256=latent.metrics.sha256(meta_path), tensors=tensors
            ),
        )
        (path / "run.json").write_text(json.dumps(run))
        paths[label] = path
    return SimpleNamespace(**paths, output_dir=tmp_path / "evaluation")


def change_run(path, mutation):
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value))


def test_complete_fixture_npz_and_per_step_metrics(captures):
    result = latent.execute(captures)
    assert result["status"] == "completed"
    assert result["summary"]["video"]["hadamard_lower_relative_l2_steps"] == 20
    assert result["summary"]["audio"]["final_hadamard"]["relative_l2"]["value"] == pytest.approx(
        0.1
    )
    with np.load(captures.output_dir / "bf16.npz", allow_pickle=False) as arrays:
        assert len(arrays.files) == 42
        assert arrays["final_video"].shape == (1, 2, 2)
    assert json.loads((captures.output_dir / "latent-evaluation.json").read_text()) == result
    with pytest.raises(FileExistsError):
        latent.execute(captures)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(status="running"),
        lambda r: r.update(seed=43),
        lambda r: r.update(mode="hadamard-fc2"),
        lambda r: r["patch"].update(source_sha256="untrusted"),
        lambda r: r["latent_validation"].update(metadata_sha256="wrong"),
        lambda r: r["latent_validation"]["tensors"].pop(),
    ],
)
def test_bad_provenance_is_preserved_as_failure(captures, mutation):
    change_run(captures.native / "run.json", mutation)
    result = latent.execute(captures)
    assert result["status"] == "failed"
    assert "error" in result
    assert (captures.output_dir / "latent-evaluation.json").exists()


def test_cross_prompt_pair_rejected(captures):
    change_run(captures.native / "run.json", lambda r: r.update(prompt="different"))
    assert "recipe" in latent.execute(captures)["error"]


@pytest.mark.parametrize("kind", ["changed", "nonfinite", "extra", "truncated"])
def test_raw_tensor_integrity_shape_finite_and_extras(captures, kind):
    path = captures.native / "latents/euler-step00-video.f32"
    if kind == "extra":
        (path.parent / "extra.f32").write_bytes(b"bad")
    else:
        values = np.array([np.nan, 1, 1, 1] if kind == "nonfinite" else [2, 2, 2, 2], dtype="<f4")
        path.write_bytes(values.tobytes()[:-4] if kind == "truncated" else values.tobytes())
        if kind in ("nonfinite", "truncated"):

            def update(r):
                next(x for x in r["latent_validation"]["tensors"] if x["file"] == path.name)[
                    "sha256"
                ] = latent.metrics.sha256(path)

            change_run(captures.native / "run.json", update)
    assert latent.execute(captures)["status"] == "failed"

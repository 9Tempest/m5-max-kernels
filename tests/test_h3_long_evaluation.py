"""CPU-only first-frame, temporal-boundary and provenance checks for the 15s pair."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_h3_benchmark_evaluation import paired as paired, mutate
from test_h3_latent_evaluation import captures as captures, latent as fixture_capture


DIRECTORY = Path(__file__).resolve().parents[1] / "integrations/h3"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("h3_long_eval", DIRECTORY / "evaluate_long_pair.py")
long = importlib.util.module_from_spec(spec)
spec.loader.exec_module(long)


@pytest.fixture
def long_pair(paired, monkeypatch, tmp_path):
    monkeypatch.setattr(long, "capture", fixture_capture)
    monkeypatch.setattr(long.pair, "capture", fixture_capture)
    monkeypatch.setattr(long, "latent_shapes", lambda recipe: fixture_capture.SHAPES)
    catalog = json.loads(long.CATALOG_FILE.read_text())
    prompt = "完整中文提示词。\n".encode()
    image = b"fixture-jpeg-payload"
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    prompt_path = catalog_dir / "prompt.txt"
    image_path = catalog_dir / "image.jpg"
    prompt_path.write_bytes(prompt)
    image_path.write_bytes(image)
    monkeypatch.setattr(long, "IMAGE_SHA256", long.metrics.sha256(image_path))
    monkeypatch.setattr(long, "IMAGE_BYTES", len(image))
    monkeypatch.setattr(long, "PROMPT_SHA256", long.metrics.sha256(prompt_path))
    catalog["benchmarks"][0].update(
        prompt_file="prompt.txt",
        prompt_sha256=long.PROMPT_SHA256,
        prompt_bytes=len(prompt),
        image_sha256=long.IMAGE_SHA256,
        image_bytes=len(image),
    )
    catalog_path = catalog_dir / "long_benchmark.json"
    catalog_path.write_text(json.dumps(catalog))
    monkeypatch.setattr(long, "CATALOG_FILE", catalog_path)
    monkeypatch.setattr(long, "LONG_CATALOG_SHA256", long.metrics.sha256(catalog_path))
    entry = long.load_catalog()
    for directory in (paired.bf16, paired.hadamard):
        (directory / "prompt.txt").write_bytes(prompt)
        (directory / "conditioning-input.jpg").write_bytes(image)
        path = directory / "run.json"
        run = json.loads(path.read_text())
        run.update(long.RECIPE)
        run.update(
            benchmark=entry["id"],
            catalog=entry,
            prompt=entry["prompt"],
            prepared_checkpoint=directory == paired.hadamard,
            conditioning=dict(
                mode="first-frame",
                input_file="conditioning-input.jpg",
                sha256=long.IMAGE_SHA256,
                bytes=len(image),
                original_width=1280,
                original_height=720,
                preprocessing="upstream_first_frame_stretch_to_render_canvas",
                target_canvas=[512, 288],
            ),
            measurements={"generation": {"returncode": 0, "fallback_detected": False}},
            original_video=dict(
                width=512, height=288, frames=362, fps=24, duration_seconds=362 / 24
            ),
            exact_video=dict(width=512, height=288, frames=360, fps=24, duration_seconds=15),
        )
        meta_path = directory / "latents/latent-metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["seed"] = 83
        meta_path.write_text(json.dumps(meta))
        run["latent_validation"].update(
            metadata=meta, metadata_sha256=long.metrics.sha256(meta_path)
        )
        path.write_text(json.dumps(run))
    return paired


def test_native_362_frame_shapes_and_condition_rows():
    shapes = long.latent_shapes(long.RECIPE)
    assert shapes == dict(
        final_video_shape=[24, 107, 18, 32],
        final_audio_shape=[32, 2, 603],
        per_step_video_shape=[15408, 96],
        per_step_audio_shape=[1206, 32],
    )
    assert shapes["per_step_video_shape"][0] == 107 * 144
    assert len(long.spec(shapes)) == 42
    with pytest.raises(ValueError):
        long.latent_shapes(long.RECIPE | {"source_frames": 360})


def test_frozen_real_catalog_keeps_complete_prompt_newline():
    catalog = long.load_catalog()
    assert catalog["prompt"].endswith("\n")
    assert len(catalog["prompt"].encode()) == 2233
    assert catalog["prompt_tokens"] == 545
    assert catalog["image_prefix_tokens"] == 6


def test_complete_cpu_pair_is_immutable_and_not_quality_score(long_pair):
    result = long.execute(long_pair)
    assert result["status"] == "completed", result.get("error")
    assert result["quality_decision"] == "not_evaluated"
    assert len(result["latents"]) == 42
    assert result["latents"]["final_video"]["metrics"]["relative_l2"]["value"] == pytest.approx(0.1)
    with np.load(long_pair.output_dir / "bf16.npz", allow_pickle=False) as data:
        assert len(data.files) == 42
    with pytest.raises(FileExistsError):
        long.execute(long_pair)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(seconds=5),
        lambda r: r.update(source_frames=360),
        lambda r: r.update(exact_frames=362),
        lambda r: r.update(height=512),
        lambda r: r.update(seed=42),
        lambda r: r.update(prompt=r["prompt"].strip()),
        lambda r: r["catalog"].update(prompt="different"),
        lambda r: r["conditioning"].update(input_file="../image.jpg"),
        lambda r: r["conditioning"].update(sha256="a" * 64),
        lambda r: r["conditioning"].update(preprocessing="center_crop"),
        lambda r: r["conditioning"].update(target_canvas=[512, 512]),
        lambda r: r["conditioning"].update(original_width=720),
        lambda r: r["original_video"].update(duration_seconds=15),
        lambda r: r["exact_video"].update(frames=362),
        lambda r: r["measurements"]["generation"].update(fallback_detected=True),
        lambda r: r["patch"].update(source_sha256="wrong"),
        lambda r: r["derived_checkpoint"].update(policy_id="wrong"),
    ],
)
def test_long_recipe_source_and_conditioning_rejections(long_pair, change):
    mutate(long_pair.hadamard / "run.json", change)
    result = long.execute(long_pair)
    assert result["status"] == "failed"
    assert (long_pair.output_dir / "long-evaluation.json").exists()


@pytest.mark.parametrize(
    "filename", ["prompt.txt", "conditioning-input.jpg", "derived-checkpoint.json"]
)
def test_input_byte_changes_rejected(long_pair, filename):
    path = long_pair.hadamard / filename
    path.write_bytes(path.read_bytes() + b"x")
    assert long.execute(long_pair)["status"] == "failed"


def test_catalog_byte_change_rejected_even_if_json_equivalent(long_pair):
    long.CATALOG_FILE.write_bytes(long.CATALOG_FILE.read_bytes() + b"\n")
    assert "catalog bytes" in long.execute(long_pair)["error"]


@pytest.mark.parametrize("mode", ["bf16", "hadamard"])
def test_dynamic_shape_rejected_even_with_updated_metadata_hash(long_pair, mode):
    directory = getattr(long_pair, mode)
    path = directory / "latents/latent-metadata.json"
    meta = json.loads(path.read_text())
    meta["per_step_video_shape"] = [3, 2]
    path.write_text(json.dumps(meta))
    mutate(
        directory / "run.json",
        lambda r: r["latent_validation"].update(
            metadata=meta, metadata_sha256=long.metrics.sha256(path)
        ),
    )
    assert "shape" in long.execute(long_pair)["error"]


def test_five_second_default_capture_contract_is_not_relaxed(long_pair, monkeypatch):
    # Our tiny fixture is otherwise valid. A changed default shape is rejected;
    # the long evaluator must pass its independently validated shape explicitly.
    monkeypatch.setattr(fixture_capture, "SHAPES", {key: [100] for key in fixture_capture.SHAPES})
    with pytest.raises(ValueError, match="shape"):
        long.pair.prepared_capture(
            long_pair.hadamard,
            dict(repository="fixture", revision="pinned", manifest_sha256="digest"),
        )


@pytest.mark.parametrize("mode", ["bf16", "hadamard"])
def test_nonfinite_rejected_even_with_updated_digest(long_pair, mode):
    directory = getattr(long_pair, mode)
    path = directory / "latents/euler-step00-video.f32"
    np.array([1, 1, np.nan, 1], dtype="<f4").tofile(path)

    def update(run):
        next(entry for entry in run["latent_validation"]["tensors"] if entry["file"] == path.name)[
            "sha256"
        ] = long.metrics.sha256(path)

    mutate(directory / "run.json", update)
    assert "nonfinite" in long.execute(long_pair)["error"]


@pytest.mark.parametrize("change", [None, {"frames": 362}, {"duration_seconds": "5"}])
def test_media_join_enforces_long_contract(long_pair, monkeypatch, change):
    for directory in (long_pair.bf16, long_pair.hadamard):
        path = directory / "video-15s.mp4"
        path.write_bytes(b"fixture-video")
        mutate(
            directory / "run.json",
            lambda r: r["exact_video"].update(file=path.name, sha256=long.metrics.sha256(path)),
        )
    long_pair.media = True

    def fake_media(args):
        contract = dict(
            width=512,
            height=288,
            frames=360,
            fps="24",
            duration_seconds="15",
            audio_rate=32000,
            audio_channels=2,
        )
        contract.update(change or {})
        result = dict(
            status="completed",
            files={
                label: dict(sha256=long.metrics.sha256(path), contract=contract)
                for label, path in (("reference", args.reference), ("candidate", args.candidate))
            },
            video_rgb={"fixture": True},
            audio_pcm32k_stereo={"fixture": True},
            optional={"ssim": {"status": "completed"}},
        )
        args.output_dir.mkdir()
        long.metrics.write_json(args.output_dir / "evaluation.json", result)
        return result

    monkeypatch.setattr(long.metrics, "execute", fake_media)
    result = long.execute(long_pair)
    assert result["status"] == ("completed" if change is None else "failed")
    if change is None:
        assert result["media"]["evaluation_sha256"]


def test_bf16_fallback_is_rejected(long_pair):
    mutate(
        long_pair.bf16 / "run.json",
        lambda r: r["measurements"]["generation"].update(fallback_detected=True),
    )
    assert "fell back" in long.execute(long_pair)["error"]

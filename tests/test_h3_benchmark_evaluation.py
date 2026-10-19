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
        video = directory / "video-5s.mp4"
        video.write_bytes(b"fixture-video")
        run["exact_video"] = dict(
            file=video.name,
            sha256=pair.metrics.sha256(video),
            width=512,
            height=512,
            frames=120,
            fps=24,
            duration_seconds=5.0,
        )
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


def media_fixture(paired):
    contract = dict(
        width=512,
        height=512,
        frames=120,
        fps="24",
        duration_seconds="5",
        audio_rate=32000,
        audio_channels=2,
        decoded_audio_samples=160000,
    )
    return dict(
        status="completed",
        kind="media",
        harness_sha256=pair.metrics.sha256(Path(pair.metrics.__file__)),
        files={
            label: dict(sha256=pair.metrics.sha256(directory / "video-5s.mp4"), contract=contract)
            for label, directory in (("reference", paired.bf16), ("candidate", paired.hadamard))
        },
        video_rgb=dict(
            status="completed",
            shape=[120, 512, 512, 3],
            data_range=255.0,
            reference_dtype="uint8",
            candidate_dtype="uint8",
            reference_nonfinite=0,
            candidate_nonfinite=0,
        ),
        audio_pcm32k_stereo=dict(
            status="completed",
            shape=[160000, 2],
            data_range=2.0,
            reference_dtype="float32",
            candidate_dtype="float32",
            reference_nonfinite=0,
            candidate_nonfinite=0,
        ),
        optional={"ssim": {"status": "completed"}},
    )


def join_fixture(paired, record):
    path = paired.output_dir.parent / "joined-media.json"
    path.write_text(json.dumps(record))
    paired.media_evaluation = path
    return pair.execute(paired)


def test_valid_media_join_is_bound_to_capture_runs(paired):
    assert join_fixture(paired, media_fixture(paired))["status"] == "completed"


@pytest.mark.parametrize("requested_decode", [False, True])
def test_both_replaced_videos_cannot_reuse_original_latents(paired, monkeypatch, requested_decode):
    for directory in (paired.bf16, paired.hadamard):
        (directory / "video-5s.mp4").write_bytes(b"replacement-pair")
    record = media_fixture(paired)  # Even a matching eval for both new files must fail.
    if requested_decode:
        paired.media = True
        monkeypatch.setattr(
            pair.metrics, "execute", lambda args: pytest.fail("must reject before media decode")
        )
        result = pair.execute(paired)
    else:
        result = join_fixture(paired, record)
    assert result["status"] == "failed"
    assert "digest differs from capture run" in result["error"]


@pytest.mark.parametrize(
    "change",
    [
        {"file": "other.mp4"},
        {"file": "../video-5s.mp4"},
        {"width": 256},
        {"height": 256},
        {"frames": 96},
        {"fps": 30},
        {"duration_seconds": 4},
    ],
)
def test_both_run_records_must_declare_the_fixed_video(paired, change):
    for directory in (paired.bf16, paired.hadamard):
        mutate(directory / "run.json", lambda r: r["exact_video"].update(change))
    result = join_fixture(paired, media_fixture(paired))
    assert "recorded exact video" in result["error"]


@pytest.mark.parametrize(
    "change",
    [
        {"width": 256},
        {"height": 256},
        {"frames": 96},
        {"fps": "30"},
        {"duration_seconds": "4"},
        {"audio_rate": 48000},
        {"audio_channels": 1},
        {"decoded_audio_samples": 128000},
    ],
)
def test_both_decoded_contracts_cannot_share_a_wrong_recipe(paired, change):
    record = media_fixture(paired)
    for side in record["files"].values():
        side["contract"].update(change)
    result = join_fixture(paired, record)
    assert "decoded media differs" in result["error"]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["files"]["candidate"].update(sha256="0" * 64),
        lambda r: r["video_rgb"].update(shape=[96, 512, 512, 3]),
        lambda r: r["audio_pcm32k_stereo"].update(shape=[128000, 2]),
        lambda r: r["audio_pcm32k_stereo"].update(data_range=1.0),
        lambda r: r["video_rgb"].update(status="failed"),
        lambda r: r["audio_pcm32k_stereo"].update(candidate_nonfinite=1),
        lambda r: r["audio_pcm32k_stereo"].update(candidate_dtype="float64"),
        lambda r: r["optional"]["ssim"].update(status="not_measured"),
    ],
)
def test_joined_metrics_require_matching_hash_shape_range_and_completion(paired, change):
    record = media_fixture(paired)
    change(record)
    assert join_fixture(paired, record)["status"] == "failed"

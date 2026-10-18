"""CPU-only checks for the isolated image-conditioned 15-second recipe."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / "integrations/h3"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location(
    "h3_long_benchmark", DIRECTORY / "run_long_benchmark.py"
)
long = importlib.util.module_from_spec(spec)
spec.loader.exec_module(long)


def args(tmp_path, mode="bf16", *extra):
    return long.parser().parse_args(["--mode", mode, "--output-dir", str(tmp_path / "run"), *extra])


def test_full_prompt_and_long_temporal_geometry():
    item = long.load_benchmark()
    assert len(item["prompt"].encode()) == 2233
    assert item["prompt"].endswith("\n")
    assert hashlib.sha256(item["prompt"].encode()).hexdigest() == item["prompt_sha256"]
    assert (item["width"], item["height"], item["seed"], item["steps"]) == (512, 288, 83, 20)
    shape = long.expected_shape(item)
    assert (shape["requested_frames"], shape["aligned_frames"]) == (360, 362)
    assert (shape["video_latent_t"], shape["audio_latent_t"]) == (107, 603)
    assert (shape["video_tokens"], shape["stereo_audio_tokens"], shape["condition_video_rows"]) == (
        15408,
        1206,
        144,
    )


def test_catalog_change_unknown_id_and_alignment_rejected(tmp_path):
    path = tmp_path / "changed.json"
    path.write_bytes(long.DEFAULT.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen hash"):
        long.load_benchmark(path=path)
    with pytest.raises(ValueError, match="exactly one"):
        long.load_benchmark("other")
    with pytest.raises(ValueError, match="alignment"):
        long.expected_shape(long.load_benchmark() | {"source_frames": 360})


@pytest.mark.parametrize("mode", ["bf16", "hadamard-fc2"])
def test_command_has_actual_first_frame_complete_prompt_and_fixed_recipe(tmp_path, mode):
    item = long.load_benchmark()
    command = long.build_command(
        item,
        mode,
        tmp_path,
        tmp_path / "model",
        tmp_path / "original.mp4",
        tmp_path / "conditioning-input.jpg",
    )
    expected = {
        "-p": item["prompt"],
        "--width": "512",
        "--height": "288",
        "--seconds": "15.0",
        "--steps": "20",
        "--seed": "83",
        "--first-frame": "conditioning-input.jpg",
        "--layers": "50",
        "--reuse": "1",
        "--core-reuse": "1",
    }
    for flag, value in expected.items():
        assert command.count(flag) == 1
        assert command[command.index(flag) + 1] == value
    assert "--use-reference-rope" in command
    assert all((flag in command) == (mode == "bf16") for flag in long.base.BF16_FLAGS)


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--checkpoint-dir", "missing"],
        ["--checkpoint-dir", "missing", "--checkpoint-manifest-sha256", "bad"],
        [
            "--checkpoint-dir",
            "missing",
            "--checkpoint-manifest-sha256",
            "a" * 64,
            "--model-dir",
            "fallback",
        ],
    ],
)
def test_prepared_requires_explicit_derived_identity(tmp_path, extra):
    with pytest.raises(ValueError):
        long.configuration(args(tmp_path, "hadamard-fc2", *extra))


def test_frozen_source_choices(tmp_path):
    assert long.configuration(args(tmp_path))[1].name == "h3.c-hadamard"
    _, source, model = long.configuration(
        args(
            tmp_path,
            "hadamard-fc2",
            "--checkpoint-dir",
            str(tmp_path / "prepared"),
            "--checkpoint-manifest-sha256",
            "a" * 64,
        )
    )
    assert source.name == "h3.c-hadamard-cache"
    assert model == tmp_path / "prepared/model"
    with pytest.raises(ValueError):
        long.configuration(args(tmp_path, "bf16", "--checkpoint-dir", "wrong"))


def test_raw_conditioning_snapshot_is_not_transformed(tmp_path):
    raw = b"unaltered reference bytes\x00\xff"
    image = tmp_path / "source.jpg"
    image.write_bytes(raw)
    item = long.load_benchmark() | {
        "image_sha256": hashlib.sha256(raw).hexdigest(),
        "image_bytes": len(raw),
    }
    output = tmp_path / "output"
    output.mkdir()
    saved = long.snapshot_inputs(item, image, output)
    assert (output / saved["input_file"]).read_bytes() == raw
    assert (output / "prompt.txt").read_bytes() == item["prompt"].encode()
    assert saved["preprocessing"] == "upstream_first_frame_stretch_to_render_canvas"
    assert saved["target_canvas"] == [512, 288]
    assert json.loads((output / "inputcondition.json").read_text()) == saved
    with pytest.raises(FileExistsError):
        long.snapshot_inputs(item, image, output)


def test_bad_image_rejected_before_lock_and_failure_preserved(tmp_path, monkeypatch):
    image = tmp_path / "wrong.jpg"
    image.write_bytes(b"wrong")
    monkeypatch.setattr(long.base, "wait_for_gpu_lock", lambda *_: pytest.fail("lock must not run"))
    result = long.execute(args(tmp_path, "bf16", "--input-image", str(image)))
    assert result["status"] == "failed"
    assert "image hash" in result["error"]
    assert json.loads((tmp_path / "run/run.json").read_text()) == result


def test_source_failure_releases_lock_and_preserves_record(tmp_path, monkeypatch):
    states = []

    @contextmanager
    def lock(*_):
        states.append("acquired")
        try:
            yield
        finally:
            states.append("released")

    monkeypatch.setattr(long, "snapshot_inputs", lambda *_: {})
    monkeypatch.setattr(long.base, "wait_for_gpu_lock", lock)
    monkeypatch.setattr(long.base, "git_provenance", lambda _: {})

    def rejected(*_):
        raise ValueError("source guard fixture")

    monkeypatch.setattr(long.experiment, "validate_source", rejected)
    result = long.execute(args(tmp_path))
    assert result["status"] == "failed" and "source guard fixture" in result["error"]
    assert states == ["acquired", "released"]


def latent_fixture(directory, mode="bf16"):
    directory.mkdir()
    for step in range(20):
        for modality in ("video", "audio"):
            (directory / f"euler-step{step:02d}-{modality}.f32").touch()
    for modality in ("video", "audio"):
        (directory / f"final-{modality}.f32").touch()
    meta = dict(
        schema=1,
        scope="diagnostic_CPU_readback_after_completed_GPU_Euler",
        dtype="float32_little_endian",
        seed=83,
        completed_steps=20,
        final_video_shape=[24, 107, 18, 32],
        final_audio_shape=[32, 2, 603],
        per_step_video_shape=[15408, 96],
        per_step_audio_shape=[1206, 32],
        final_video_sigma=0,
        final_audio_sigma=0,
        final_layout="unpacked_contiguous_VAE_input_before_decoder",
        per_step_layout="packed_model_rows_after_Euler_before_unpack",
        int8_mlp=int(mode != "bf16"),
        int8_qkv=int(mode != "bf16"),
        int8_attention_out=int(mode != "bf16"),
        hadamard_fc2=int(mode == "hadamard-fc2"),
        finite=True,
    )
    (directory / "latent-metadata.json").write_text(json.dumps(meta))
    return meta


@pytest.mark.parametrize("mode", ["bf16", "hadamard-fc2"])
def test_capture_counts_exclude_first_frame_and_include_every_step(tmp_path, monkeypatch, mode):
    directory = tmp_path / "latents"
    latent_fixture(directory, mode)
    calls = []

    def finite(path, dtype, count):
        calls.append((path.name, dtype, count))
        return {"file": path.name, "elements": count, "finite": True}

    monkeypatch.setattr(long.experiment, "finite_file", finite)
    record = long.validate_latent_capture(directory, mode, long.load_benchmark())
    assert len(record["tensors"]) == 42
    assert all(
        dtype == "<f4" and count == (1479168 if "video" in name else 38592)
        for name, dtype, count in calls
    )
    (directory / "euler-step19-video.f32").unlink()
    with pytest.raises(ValueError, match="all 20"):
        long.validate_latent_capture(directory, mode, long.load_benchmark())


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 42},
        {"final_video_shape": [24, 37, 32, 32]},
        {"per_step_video_shape": [15552, 96]},
        {"dtype": "bfloat16"},
        {"final_audio_sigma": 1},
        {"scope": "wrong"},
        {"finite": False},
        {"hadamard_fc2": 1},
    ],
)
def test_capture_rejects_wrong_geometry_precision_or_scope(tmp_path, monkeypatch, change):
    directory = tmp_path / "latents"
    meta = latent_fixture(directory)
    (directory / "latent-metadata.json").write_text(json.dumps(meta | change))
    monkeypatch.setattr(
        long.experiment, "finite_file", lambda *_: pytest.fail("must reject metadata first")
    )
    with pytest.raises(ValueError, match="recipe/precision"):
        long.validate_latent_capture(directory, "bf16", long.load_benchmark())


def test_capture_checks_real_bytes_and_nonfinite(tmp_path):
    directory = tmp_path / "latents"
    latent_fixture(directory)
    with pytest.raises(ValueError, match="wrong tensor size"):
        long.validate_latent_capture(directory, "bf16", long.load_benchmark())


@pytest.mark.parametrize("bad_audio", [False, True])
def test_export_checks_native_and_exact_frames_and_audio(tmp_path, monkeypatch, bad_audio):
    calls = []
    for name in ("original.mp4", "video-15s.mp4"):
        (tmp_path / name).write_bytes(b"fixture")

    def logged(command, source, env, output, label):
        calls.append(command)
        if label.endswith("-probe"):
            frames = 362 if label == "original-probe" else 360
            probe = {
                "streams": [
                    dict(
                        codec_type="video",
                        width=512,
                        height=288,
                        avg_frame_rate="24/1",
                        nb_read_frames=str(frames),
                        duration=str(frames / 24),
                    ),
                    dict(
                        codec_type="audio",
                        channels=2,
                        channel_layout="stereo",
                        sample_rate="16000" if bad_audio else "32000",
                        duration=str(frames / 24),
                    ),
                ]
            }
            (output / f"{label}.log").write_text(json.dumps(probe))
        return {"returncode": 0}

    monkeypatch.setattr(long.base, "run_logged", logged)
    record = {"measurements": {}}
    if bad_audio:
        with pytest.raises(ValueError, match="32000 Hz"):
            long.export_video(tmp_path, tmp_path, {}, "ffmpeg", "ffprobe", record)
    else:
        long.export_video(tmp_path, tmp_path, {}, "ffmpeg", "ffprobe", record)
        trim = next(c for c in calls if c[0] == "ffmpeg")
        assert trim[trim.index("-frames:v") + 1] == "360"
        assert trim[trim.index("-t") + 1] == "15"
        assert record["original_video"]["frames"] == 362
        assert record["exact_video"]["duration_seconds"] == 15


@pytest.mark.parametrize("failure", [None, "tokens", "roundtrip", "limit", "compile"])
def test_native_tokenizer_probe_rejects_truncation_and_limits(tmp_path, monkeypatch, failure):
    model = tmp_path / "model"
    tokenizer = model / "FL2VA/tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 696 if failure == "limit" else 262144})
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "tokenizer-probe").write_bytes(b"fixture")

    def logged(command, source, env, directory, label):
        if label == "tokenizer":
            payload = dict(
                schema=1,
                prompt_bytes=2233,
                prompt_tokens=544 if failure == "tokens" else 545,
                image_prefix_tokens=6,
                roundtrip_exact=failure != "roundtrip",
            )
            (directory / "tokenizer.log").write_text(json.dumps(payload))
        return {"returncode": int(failure == "compile" and label == "tokenizer-build")}

    monkeypatch.setattr(long.base, "run_logged", logged)
    if failure:
        expected = long.base.UnsupportedError if failure == "compile" else ValueError
        with pytest.raises(expected):
            long.probe_tokenizer(long.load_benchmark(), tmp_path, model, output, {})
    else:
        record = long.probe_tokenizer(long.load_benchmark(), tmp_path, model, output, {})
        assert record["presentation_tokens"] == 697
        assert record["truncated"] is False
        assert record["native_overflow_limit"] == 167772


def test_missing_vision_component_fails_before_gpu(tmp_path):
    root = tmp_path / "FL2VA/text_encoder"
    root.mkdir(parents=True)
    (root / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    with pytest.raises(ValueError, match="vision tensors missing"):
        long.conditioning_headers(tmp_path)

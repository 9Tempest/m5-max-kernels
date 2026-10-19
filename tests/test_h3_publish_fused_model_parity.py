"""Diagnostic publication requires finite exact latents AND true media parity."""

import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest


@pytest.fixture
def publisher(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    for name in (
        "run",
        "run_fc2_experiment",
        "run_fused_campaign",
        "run_fused_experiment",
        "publish_fused_campaign",
        "publish_samples",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "fusion_parity_publisher_test", directory / "publish_fused_model_parity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, publisher, monkeypatch):
    names = sorted(
        [f"euler-step{step:02d}-{kind}.f32" for step in range(20) for kind in ("video", "audio")]
        + ["final-video.f32", "final-audio.f32"]
    )
    metadata = dict(
        dtype="float32_little_endian",
        scope="diagnostic_CPU_readback_after_completed_GPU_Euler",
        final_video_sigma=0,
        final_audio_sigma=0,
    )

    def small_capture(directory, mode, seed):
        assert mode == "hadamard-fc2" and seed == 42
        return dict(
            metadata=metadata,
            metadata_sha256="c" * 64,
            tensors=[publisher.capture.finite_file(directory / name, "<f4", 1) for name in names],
        )

    monkeypatch.setattr(publisher.capture, "validate_latent_capture", small_capture)
    directories, records = [], []
    for mode in ("none", "attention", "both"):
        directory = tmp_path / mode
        (directory / "latents").mkdir(parents=True)
        for i, name in enumerate(names):
            (directory / "latents" / name).write_bytes(struct.pack("<f", i))
        flags = dict(
            H3_HADAMARD_FC2="1",
            H3_FUSE_FC2_GATE=str(int(mode == "both")),
            H3_FUSE_ATTN_OUT_GATE=str(int(mode != "none")),
        )
        log = "\n".join(publisher.MARKERS[key] for key, value in flags.items() if value == "1")
        (directory / "generation.log").write_text(log)
        source = publisher.capture if mode == "none" else publisher.fused
        record = publisher.RECIPE | dict(
            status="completed",
            mode="hadamard-fc2",
            fusion=mode,
            seed=42,
            capture_latents=True,
            diagnostic=True,
            patch=dict(source_sha256=source.SOURCE_SHA256, patch_sha256=source.PATCH_SHA256),
            upstream=dict(
                revision=publisher.base.UPSTREAM_REVISION,
                source_sha256=source.SOURCE_SHA256,
                binary_sha256="b" * 64,
            ),
            model=dict(
                repository="MiniMaxAI/MiniMax-H3",
                revision=publisher.base.MODEL_REVISION,
                manifest_sha256=publisher.base.MODEL_MANIFEST_SHA256,
            ),
            measurements=dict(
                generation=dict(
                    returncode=0,
                    fallback_detected=False,
                    log_sha256=publisher.base.sha256(directory / "generation.log"),
                )
            ),
            runtime_overrides=flags,
            latent_validation=small_capture(directory / "latents", "hadamard-fc2", 42),
        )
        for key, file, frames in (
            ("original_video", "original.mp4", 124),
            ("exact_video", "video-5s.mp4", 120),
        ):
            (directory / file).write_bytes(b"fixture video")
            record[key] = dict(
                file=file,
                width=512,
                height=512,
                fps=24,
                frames=frames,
                sha256=publisher.base.sha256(directory / file),
            )
        if mode != "none":
            record.update(
                reference=dict(
                    run_sha256=publisher.base.sha256(directories[0] / "run.json"),
                    source_sha256=publisher.capture.SOURCE_SHA256,
                    seed=42,
                ),
                video_bytes_equal_reference=True,
                latent_parity=publisher.diagnostic.latent_parity(
                    records[0]["latent_validation"], record["latent_validation"]
                ),
            )
        (directory / "run.json").write_text(json.dumps(record))
        directories.append(directory)
        records.append(record)
    return directories, records


def save(directory, record):
    (directory / "run.json").write_text(json.dumps(record))


def test_exact_diagnostics_export_raw_bytes(tmp_path, publisher, monkeypatch):
    directories, _ = fixture(tmp_path, publisher, monkeypatch)
    result = publisher.execute(*directories, tmp_path / "public")
    assert all(entry["passed"] for entry in result["comparisons"])
    assert all(entry["export"] == "byte_identical_raw_json" for entry in result["source_records"])
    for source, entry in zip(directories, result["source_records"], strict=True):
        assert (tmp_path / "public" / entry["file"]).read_bytes() == (
            source / "run.json"
        ).read_bytes()
    assert "not this executable path" in result["prepared_checkpoint_scope"]
    with pytest.raises(FileExistsError):
        publisher.execute(*directories, tmp_path / "public")


@pytest.mark.parametrize("value", [False, None, 1])
def test_completed_status_does_not_override_false_media_parity(
    tmp_path, publisher, monkeypatch, value
):
    directories, records = fixture(tmp_path, publisher, monkeypatch)
    records[1]["video_bytes_equal_reference"] = value
    save(directories[1], records[1])
    with pytest.raises(publisher.base.NumericalError, match="video parity"):
        publisher.execute(*directories, tmp_path / "public")


def test_recomputed_video_bytes_must_match(tmp_path, publisher, monkeypatch):
    directories, records = fixture(tmp_path, publisher, monkeypatch)
    (directories[1] / "original.mp4").write_bytes(b"different")
    records[1]["original_video"]["sha256"] = publisher.base.sha256(directories[1] / "original.mp4")
    save(directories[1], records[1])
    with pytest.raises(publisher.base.NumericalError, match="video bytes"):
        publisher.execute(*directories, tmp_path / "public")


def test_nonfinite_late_tensor_rejected(tmp_path, publisher, monkeypatch):
    directories, _ = fixture(tmp_path, publisher, monkeypatch)
    (directories[2] / "latents/final-video.f32").write_bytes(struct.pack("<f", float("nan")))
    with pytest.raises(publisher.base.NumericalError, match="nonfinite"):
        publisher.execute(*directories, tmp_path / "public")


def test_single_changed_tensor_rejected_even_with_updated_manifest(
    tmp_path, publisher, monkeypatch
):
    directories, records = fixture(tmp_path, publisher, monkeypatch)
    path = directories[1] / "latents/euler-step19-video.f32"
    path.write_bytes(struct.pack("<f", 999))
    for tensor in records[1]["latent_validation"]["tensors"]:
        if tensor["file"] == path.name:
            tensor["sha256"] = publisher.base.sha256(path)
    save(directories[1], records[1])
    with pytest.raises(publisher.base.NumericalError, match="42 latent tensors"):
        publisher.execute(*directories, tmp_path / "public")


def test_private_source_uses_explicit_sanitized_summary(tmp_path, publisher, monkeypatch):
    directories, records = fixture(tmp_path, publisher, monkeypatch)
    records[1]["private_path"] = "/Users/owner/not-public"
    records[1]["upstream"]["status"] = "/Users/owner/not-public"
    save(directories[1], records[1])
    result = publisher.execute(*directories, tmp_path / "public")
    assert result["source_records"][1]["export"] == "sanitized_summary_only"
    assert not (tmp_path / "public/raw/attention-run.json").exists()
    assert "/Users/" not in json.dumps(result)


@pytest.mark.parametrize(
    "field,value", [("prepared_checkpoint", True), ("seed", 43), ("status", "running")]
)
def test_incompatible_run_rejected(tmp_path, publisher, monkeypatch, field, value):
    directories, records = fixture(tmp_path, publisher, monkeypatch)
    record = copy.deepcopy(records[1])
    record[field] = value
    save(directories[1], record)
    with pytest.raises(ValueError):
        publisher.execute(*directories, tmp_path / "public")


def test_missing_runtime_marker_rejected(tmp_path, publisher, monkeypatch):
    directories, records = fixture(tmp_path, publisher, monkeypatch)
    (directories[1] / "generation.log").write_text(publisher.MARKERS["H3_HADAMARD_FC2"])
    records[1]["measurements"]["generation"]["log_sha256"] = publisher.base.sha256(
        directories[1] / "generation.log"
    )
    save(directories[1], records[1])
    with pytest.raises(ValueError, match="marker"):
        publisher.execute(*directories, tmp_path / "public")

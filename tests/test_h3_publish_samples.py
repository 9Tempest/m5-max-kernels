"""Publication must bind samples to completed records and matching eval inputs."""

import importlib.util
import copy
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def publisher(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.delitem(sys.modules, "run", raising=False)
    spec = importlib.util.spec_from_file_location(
        "h3_publish_test", directory / "publish_samples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(tmp_path, publisher):
    run = tmp_path / "input"
    run.mkdir()
    video = run / "video-5s.mp4"
    video.write_bytes(b"fixture video, not decoded by these metadata tests")
    record = dict(
        status="completed",
        mode="bf16",
        benchmark="fixture-v1",
        prompt="Test scene",
        seed=42,
        width=512,
        height=512,
        seconds=5,
        steps=20,
        active_dit_blocks=50,
        reuse=1,
        core_reuse=1,
        reference_rope=True,
        upstream=dict(revision="source", source_sha256="digest"),
        model=dict(repository="original", revision="pinned", manifest_sha256="a" * 64),
        exact_video=dict(
            file=video.name,
            sha256=publisher.base.sha256(video),
            width=512,
            height=512,
            frames=120,
            fps=24,
            duration_seconds=5,
        ),
    )
    (run / "run.json").write_text(json.dumps(record))
    return run, record


def test_unchanged_video_is_bound_to_run_digest(tmp_path, publisher):
    directory, _ = sample(tmp_path, publisher)
    entry, video = publisher.sample_record(directory)
    assert entry["source_run_sha256"] == publisher.base.sha256(directory / "run.json")
    assert entry["video"]["sha256"] == publisher.base.sha256(video)
    assert entry["quality_decision"] == "not_evaluated"
    assert "timing_note" in entry


def test_changed_video_cannot_reuse_completed_evidence(tmp_path, publisher):
    directory, _ = sample(tmp_path, publisher)
    (directory / "video-5s.mp4").write_bytes(b"different output")
    with pytest.raises(ValueError, match="differs"):
        publisher.sample_record(directory)


@pytest.mark.parametrize("status", ["failed", "incorrect", "unsupported", "running"])
def test_failed_run_cannot_be_presented_as_sample(tmp_path, publisher, status):
    directory, record = sample(tmp_path, publisher)
    record["status"] = status
    (directory / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="completed"):
        publisher.sample_record(directory)


def test_private_absolute_paths_rejected(publisher):
    with pytest.raises(ValueError, match="private"):
        publisher.public_text_check({"path": "/Users/example/private/run.json"})


def test_gallery_escapes_prompt_html(tmp_path, publisher):
    directory, record = sample(tmp_path, publisher)
    record["prompt"] = '<script>alert("test")</script>'
    (directory / "run.json").write_text(json.dumps(record))
    entry, _ = publisher.sample_record(directory)
    page = publisher.render_gallery([entry])
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_long_landscape_sample_preserves_recipe_and_derived_identity(tmp_path, publisher):
    directory, record = sample(tmp_path, publisher)
    (directory / "video-5s.mp4").rename(directory / "video-15s.mp4")
    record.update(height=288, seconds=15, fps=24, mode="hadamard-fc2")
    record["exact_video"]["file"] = "video-15s.mp4"
    record["exact_video"].update(height=288, frames=360, duration_seconds=15)
    manifest = dict(
        schema="h3_prepared_model_v1",
        policy_id="h3_dit_int8_native_fc2_h256_v1",
        original_checkpoint_retained=True,
        source_model=dict(repository="original", revision="pinned", model_manifest_sha256="a" * 64),
        producer=dict(source_sha256="b" * 64, patch_sha256="c" * 64),
        prepared_manifest_sha256="d" * 64,
    )
    manifest_path = directory / "derived-checkpoint.json"
    manifest_path.write_text(json.dumps(manifest))
    digest = publisher.base.sha256(manifest_path)
    record["prepared_checkpoint"] = True
    record["model"] = dict(
        source_model=manifest["source_model"],
        manifest_sha256=digest,
        prepared_manifest_sha256="d" * 64,
    )
    record["derived_checkpoint"] = dict(
        manifest, manifest_file=manifest_path.name, manifest_sha256=digest
    )
    record["conditioning"] = dict(
        type="first_frame", image_sha256="a" * 64, path="/Users/private/image.jpg"
    )
    (directory / "run.json").write_text(json.dumps(record))
    entry, _ = publisher.sample_record(directory)
    assert entry["video"]["frames"] == 360
    assert entry["video"]["duration_seconds"] == 15
    assert entry["model"]["source_model"]["revision"] == "pinned"
    assert entry["model"]["producer"]["source_sha256"] == "b" * 64
    assert entry["model"]["manifest_sha256"] == digest
    assert "path" not in entry["conditioning"]
    page = publisher.render_gallery([entry])
    assert "512 × 288 · 15 s" in page


def test_duration_cannot_relabel_an_existing_video(tmp_path, publisher):
    directory, record = sample(tmp_path, publisher)
    record["seconds"] = 15
    (directory / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="duration"):
        publisher.sample_record(directory)


def test_prepared_sample_requires_local_manifest_snapshot(tmp_path, publisher):
    directory, record = sample(tmp_path, publisher)
    record["prepared_checkpoint"] = True
    record["mode"] = "hadamard-fc2"
    (directory / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="run-local"):
        publisher.sample_record(directory)


def test_prepared_manifest_cannot_change_after_run(tmp_path, publisher):
    directory, record = sample(tmp_path, publisher)
    record["prepared_checkpoint"] = True
    record["mode"] = "hadamard-fc2"
    record["derived_checkpoint"] = dict(
        manifest_file="derived-checkpoint.json", manifest_sha256="e" * 64
    )
    (directory / "derived-checkpoint.json").write_text("{}")
    (directory / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="snapshot differs"):
        publisher.sample_record(directory)


def test_prepared_h256_cannot_be_labeled_bf16(tmp_path, publisher):
    directory, record = sample(tmp_path, publisher)
    record["prepared_checkpoint"] = True
    (directory / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="H256 precision mode"):
        publisher.sample_record(directory)


def pair(tmp_path, publisher):
    directory, _ = sample(tmp_path, publisher)
    a, _ = publisher.sample_record(directory)
    b = copy.deepcopy(a)
    b.update(id="fixture-v1-seed42-hadamard-fc2", mode="hadamard-fc2", source_run_sha256="b" * 64)
    b["video"]["sha256"] = "c" * 64
    return a, b


def test_media_digest_association_must_be_unique(tmp_path, publisher):
    a, b = pair(tmp_path, publisher)
    evaluation = dict(
        kind="media",
        files=dict(
            reference=dict(sha256=a["video"]["sha256"]), candidate=dict(sha256=b["video"]["sha256"])
        ),
    )
    assert publisher.evaluation_association("media", evaluation, [a, b]) == dict(
        reference=a["id"], candidate=b["id"]
    )
    b["video"]["sha256"] = a["video"]["sha256"]
    with pytest.raises(ValueError, match="ambiguous"):
        publisher.evaluation_association("media", evaluation, [a, b])


@pytest.mark.parametrize("field", ["prompt", "seed", "steps", "conditioning"])
def test_comparison_must_use_matching_recipe_and_conditioning(tmp_path, publisher, field):
    a, b = pair(tmp_path, publisher)
    if field == "conditioning":
        b[field] = dict(image_sha256="d" * 64)
    else:
        b["recipe"][field] = "different"
    with pytest.raises(ValueError, match="different recipes"):
        publisher.validate_matched_samples([a, b])


def test_latent_evaluation_roles_must_match_precision(tmp_path, publisher):
    a, b = pair(tmp_path, publisher)
    evaluation = dict(
        sources=dict(
            bf16=dict(run_sha256=a["source_run_sha256"]),
            hadamard=dict(run_sha256=b["source_run_sha256"]),
        )
    )
    assert publisher.evaluation_association("latents", evaluation, [a, b])["hadamard"] == b["id"]
    evaluation["sources"]["bf16"]["run_sha256"] = b["source_run_sha256"]
    with pytest.raises(ValueError, match="precision"):
        publisher.evaluation_association("latents", evaluation, [a, b])


def test_latent_evaluation_requires_both_roles(tmp_path, publisher):
    a, b = pair(tmp_path, publisher)
    with pytest.raises(ValueError, match="all expected"):
        publisher.evaluation_association(
            "latents", dict(source=dict(bf16=dict(run_sha256=a["source_run_sha256"]))), [a, b]
        )

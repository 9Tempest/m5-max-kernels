"""Publication verifies raw resident evidence without launching Metal."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def publisher(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    for name in ("run", "publish_samples"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "resident_publisher_test", directory / "publish_resident.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, publisher):
    session = tmp_path / "session"
    session.mkdir()
    record = {key: None for key in publisher.KEYS}
    record.update(
        status="completed",
        mode="hadamard-fc2",
        fusion="none",
        prompt="fixture",
        benchmark="fixture-cached-v1",
        seed=42,
        width=512,
        height=512,
        steps=20,
        warmups=1,
        repetitions=3,
        source_frames=124,
        active_dit_blocks=50,
        reuse=1,
        core_reuse=1,
        use_reference_rope=True,
        warmup_seconds=180.0,
        cached_generation=dict(raw_seconds=[153.0, 152.0, 151.0], count=3, median_seconds=152.0),
        cached_denoise_callback=dict(
            raw_seconds=[123.0, 122.0, 121.0], count=3, median_seconds=122.0
        ),
        model=dict(repository="original", revision="pinned", manifest_sha256="a" * 64),
        harness=dict(revision="pinned", dirty=False),
        environment=dict(chip="Apple M5 Max"),
        all_repetition_videos_byte_identical=True,
        videos=[],
        profiles=[],
    )
    recipe = {
        key: record[key]
        for key in (
            "mode",
            "fusion",
            "prompt",
            "seed",
            "width",
            "height",
            "steps",
            "warmups",
            "repetitions",
            "reuse",
            "core_reuse",
            "use_reference_rope",
        )
    }
    recipe.update(
        event="recipe", frames=124, layers=50, profile_caveat="cached wall includes prior decode"
    )
    events = [
        recipe,
        dict(
            event="device",
            name="Apple M5 Max",
            physical_memory_bytes=128 * 2**30,
            registry_id="private-device-id",
        ),
    ]
    for index in range(4):
        stem = f"{'timed' if index else 'warmup'}-{index}"
        (session / f"{stem}.mp4").write_bytes(b"fixture-video")
        (session / f"{stem}.log").write_text("fixture profile\n")
        event = dict(
            event="generation",
            index=index,
            warmup=index == 0,
            status="completed",
            output=f"{stem}.mp4",
            profile_log=f"{stem}.log",
            wall_seconds=180.0 if index == 0 else 154.0 - index,
            denoise_callback_seconds=120.0 if index == 0 else 124.0 - index,
            cache_before=dict(
                prepared_dit=index > 0, video_decoder=index > 0, embedding_entries=int(index > 0)
            ),
        )
        events.append(event)
        record["videos"].append(
            dict(file=f"session/{stem}.mp4", sha256=publisher.base.sha256(session / f"{stem}.mp4"))
        )
        record["profiles"].append(
            dict(
                index=index,
                log_sha256=publisher.base.sha256(session / f"{stem}.log"),
                phases=[dict(wall_s=154.0)],
            )
        )
    events.append(dict(event="finished", exit_code=0))
    record["session_events"] = events
    save(tmp_path, record)
    return tmp_path / "run.json", record


def save(directory, record):
    (directory / "run.json").write_text(json.dumps(record))
    (directory / "session/session.jsonl").write_text(
        "\n".join(json.dumps(event) for event in record["session_events"]) + "\n"
    )


def test_exports_exact_event_timing_with_scope_caveat(tmp_path, publisher):
    path, record = fixture(tmp_path, publisher)
    result = publisher.summarize(path)
    assert result["cached_denoise_callback"] == record["cached_denoise_callback"]
    assert result["session_events"][0]["profile_caveat"] == "cached wall includes prior decode"
    assert "registry_id" not in result["session_events"][1]
    assert result["profiles"] == record["profiles"]
    assert "previous decode" in result["profile_interpretation"]
    assert result["source_run_sha256"] == publisher.base.sha256(path)


@pytest.mark.parametrize(
    "target", ["session/session.jsonl", "session/timed-1.mp4", "session/timed-1.log"]
)
def test_mutated_evidence_rejected(tmp_path, publisher, target):
    path, _ = fixture(tmp_path, publisher)
    changed = tmp_path / target
    changed.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="differ"):
        publisher.summarize(path)


def test_metric_cannot_substitute_native_profile_wall(tmp_path, publisher):
    path, record = fixture(tmp_path, publisher)
    record["cached_denoise_callback"].update(raw_seconds=[154.0] * 3, median_seconds=154.0)
    save(tmp_path, record)
    with pytest.raises(ValueError, match="raw timings"):
        publisher.summarize(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup", True),
        ("status", "failed"),
        ("wall_seconds", 0),
        ("denoise_callback_seconds", 200.0),
        ("output", "../escape.mp4"),
    ],
)
def test_bad_generation_contract_rejected(tmp_path, publisher, field, value):
    path, record = fixture(tmp_path, publisher)
    record["session_events"][3][field] = value
    save(tmp_path, record)
    with pytest.raises(ValueError):
        publisher.summarize(path)


def test_identity_claim_is_checked_against_files(tmp_path, publisher):
    path, record = fixture(tmp_path, publisher)
    record["all_repetition_videos_byte_identical"] = False
    save(tmp_path, record)
    with pytest.raises(ValueError, match="identity claim"):
        publisher.summarize(path)


def test_cache_must_exist_before_measured_request(tmp_path, publisher):
    path, record = fixture(tmp_path, publisher)
    record["session_events"][3]["cache_before"]["prepared_dit"] = False
    save(tmp_path, record)
    with pytest.raises(ValueError, match="caches populated"):
        publisher.summarize(path)

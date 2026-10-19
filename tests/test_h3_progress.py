"""CPU-only validation of H3 progress receipt analysis; no process or GPU launches."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / "integrations/h3"


@pytest.fixture
def analyzer(monkeypatch):
    spec = importlib.util.spec_from_file_location("h3_progress_run", DIRECTORY / "run.py")
    h3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h3)
    monkeypatch.setitem(sys.modules, "run", h3)
    spec = importlib.util.spec_from_file_location("h3_progress", DIRECTORY / "analyze_progress.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_evidence(directory, analyzer, chunks=None, times=None):
    if chunks is None:
        chunks = (
            [
                "UTF-8 日志\nh3: GPU sampler encode window is bounded; internal split disabled\n".encode()
            ]
            + [f"\rdenoise enqueue           {step:4d}/20  ".encode() for step in range(21)]
            + [
                b"\ndenoise 20/20\nh3 profile: H3 DiT                   GPU Euler denoise "
                b"wall= 50.100s encode= 1.000s wait= 49.000s root-gpu= 48.000s submissions=20\n"
            ]
        )
    if times is None:
        times = [0.1, 1.0] + [11.0 + step * 2 for step in range(20)] + [50.0]
    assert len(chunks) == len(times)
    raw = b"".join(chunks)
    events, offset = [], 0
    for chunk, elapsed in zip(chunks, times, strict=True):
        events.append(dict(byte_offset=offset, byte_count=len(chunk), elapsed_seconds=elapsed))
        offset += len(chunk)
    (directory / "generation.log").write_bytes(raw)
    (directory / "generation.events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    record = dict(
        status="completed",
        steps=20,
        diagnostics=False,
        reuse=1,
        core_reuse=1,
        active_dit_blocks=50,
        token_reduction=False,
        runtime_overrides={"H3_NAX": "qkv-attn", "H3_NAX_DIAGNOSTIC": "1"},
        upstream=dict(revision=analyzer.h3.UPSTREAM_REVISION, dirty=False),
        measurements=dict(
            generation=dict(
                log="generation.log",
                log_sha256=hashlib.sha256(raw).hexdigest(),
                returncode=0,
                fallback_detected=False,
            )
        ),
    )
    (directory / "run.json").write_text(json.dumps(record))
    return chunks, times, events, record


def save_events(directory, events):
    (directory / "generation.events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )


def test_descriptive_summary_excludes_only_first_step_and_preserves_profiles(tmp_path, analyzer):
    write_evidence(tmp_path, analyzer)
    result = analyzer.analyze(tmp_path)
    assert result["status"] == "completed"
    assert result["intervals"][0]["receipt_delta_seconds"] == 10
    assert not result["intervals"][0]["included_in_summary"]
    assert result["descriptive_summary"]["raw_seconds"] == [2] * 19
    assert result["descriptive_summary"]["median_seconds"] == 2
    assert result["descriptive_summary"]["count"] == 19
    assert result["profile_phases"][0]["wall_s"] == 50.1
    assert result["boundaries"][-1]["completed"] == 20
    assert (
        result["source_sha256"]["generation.log"]
        == hashlib.sha256((tmp_path / "generation.log").read_bytes()).hexdigest()
    )
    assert "not proven compilation-free" in " ".join(result["limitations"])


def test_coalescing_omits_all_intervals_touching_ambiguous_boundaries(tmp_path, analyzer):
    chunks, times, _, _ = write_evidence(tmp_path, analyzer)
    # Chunk 6 is completed=5 and chunk 7 is completed=6.
    chunks[6:8] = [b"".join(chunks[6:8])]
    times[6:8] = [times[7]]
    write_evidence(tmp_path, analyzer, chunks, times)
    result = analyzer.analyze(tmp_path)
    assert result["status"] == "ambiguous"
    assert [item["completed"] for item in result["boundaries"] if item["coalesced"]] == [5, 6]
    assert result["descriptive_summary"]["omitted_ambiguous_steps"] == [4, 5, 6]
    assert result["descriptive_summary"]["count"] == 16
    assert result["intervals"][5]["receipt_delta_seconds"] == 0
    assert result["intervals"][5]["timing_status"] == "coalesced_ambiguous"


def test_all_coalesced_has_no_latency_summary(tmp_path, analyzer):
    chunks, _, _, _ = write_evidence(tmp_path, analyzer)
    write_evidence(tmp_path, analyzer, [b"".join(chunks)], [50])
    result = analyzer.analyze(tmp_path)
    assert result["status"] == "ambiguous"
    assert result["descriptive_summary"]["count"] == 0
    assert result["descriptive_summary"]["median_seconds"] is None


def test_equal_timestamps_are_not_zero_step_latencies(tmp_path, analyzer):
    _, _, events, _ = write_evidence(tmp_path, analyzer)
    events[7]["elapsed_seconds"] = events[6]["elapsed_seconds"]
    save_events(tmp_path, events)
    result = analyzer.analyze(tmp_path)
    assert result["status"] == "ambiguous"
    assert result["intervals"][5]["timing_status"] == "unresolved_receipt_time"
    assert result["descriptive_summary"]["omitted_ambiguous_steps"] == [5]


def test_boundary_split_between_receipts_uses_final_numeric_byte(tmp_path, analyzer):
    chunks, times, _, _ = write_evidence(tmp_path, analyzer)
    value = chunks[11]
    split = value.index(b"/20") + 2
    chunks[11:12] = [value[:split], value[split:]]
    times[11:12] = [times[11] - 0.1, times[11]]
    write_evidence(tmp_path, analyzer, chunks, times)
    result = analyzer.analyze(tmp_path)
    assert result["boundaries"][10]["spans_receipt_blocks"]
    assert result["boundaries"][10]["receipt_block"] == 12
    assert result["descriptive_summary"]["raw_seconds"] == [2] * 19


def test_log_checksum_is_required_before_receipt_analysis(tmp_path, analyzer):
    write_evidence(tmp_path, analyzer)
    with (tmp_path / "generation.log").open("ab") as log:
        log.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        analyzer.analyze(tmp_path)


@pytest.mark.parametrize(
    "edit",
    [
        lambda events: events[0].update(byte_offset=1),
        lambda events: events[2].update(byte_offset=0),
        lambda events: events[2].update(byte_count=0),
        lambda events: events[2].update(byte_count=2.5),
        lambda events: events.pop(),
    ],
)
def test_receipt_ranges_must_contiguously_cover_exact_log(tmp_path, analyzer, edit):
    _, _, events, _ = write_evidence(tmp_path, analyzer)
    edit(events)
    save_events(tmp_path, events)
    with pytest.raises(ValueError, match="byte ranges"):
        analyzer.analyze(tmp_path)


@pytest.mark.parametrize("bad_time", [-1, 0, float("nan"), float("inf"), True, "12"])
def test_receipt_times_must_be_finite_numeric_and_monotonic(tmp_path, analyzer, bad_time):
    _, _, events, _ = write_evidence(tmp_path, analyzer)
    events[2]["elapsed_seconds"] = bad_time
    save_events(tmp_path, events)
    with pytest.raises(ValueError, match="receipt times|nonfinite JSON"):
        analyzer.analyze(tmp_path)


@pytest.mark.parametrize(
    "update",
    [
        {"status": "failed"},
        {"diagnostics": True},
        {"reuse": 2},
        {"core_reuse": 2},
        {"steps": True},
        {"token_reduction": True},
        {"runtime_overrides": {"H3_GPU_SAMPLER_WINDOW": "2"}},
    ],
)
def test_unvalidated_or_different_execution_recipe_is_rejected(tmp_path, analyzer, update):
    _, _, _, record = write_evidence(tmp_path, analyzer)
    (tmp_path / "run.json").write_text(json.dumps(record | update))
    with pytest.raises(ValueError):
        analyzer.analyze(tmp_path)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_total", "wrong_order"])
def test_progress_boundaries_must_be_complete_unique_and_ordered(tmp_path, analyzer, change):
    chunks, times, _, _ = write_evidence(tmp_path, analyzer)
    if change == "missing":
        del chunks[6]
        del times[6]
    elif change == "duplicate":
        chunks[6] += chunks[6]
    elif change == "wrong_total":
        chunks[6] = chunks[6].replace(b"/20", b"/19")
    else:
        chunks[6], chunks[7] = chunks[7], chunks[6]
    write_evidence(tmp_path, analyzer, chunks, times)
    with pytest.raises(ValueError, match="ordered denoise enqueue"):
        analyzer.analyze(tmp_path)


def test_analysis_output_never_overwrites_existing_evidence(tmp_path, analyzer, monkeypatch):
    write_evidence(tmp_path, analyzer)
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        sys, "argv", ["analyze_progress.py", str(tmp_path), "--output", str(output)]
    )
    assert analyzer.main() == 0
    original = output.read_bytes()
    assert analyzer.main() == 1
    assert output.read_bytes() == original

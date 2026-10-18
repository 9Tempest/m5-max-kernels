"""CPU mocks verify paired scheduling, timing boundaries and immutable evidence."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts/grouped_qkv_paired.py"
    spec = importlib.util.spec_from_file_location("grouped_qkv_paired_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(script, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    state = SimpleNamespace(
        locked=False,
        inside_timer=False,
        nanoseconds=0,
        prepare_calls=[],
        calls=[],
        gpu_reads=0,
        output_reads=0,
        transform_pass=True,
        quality_rejected=False,
        online_error=False,
        prepare_error=None,
        timed_calls=0,
        fail_timed_call=None,
        gpu_available=True,
    )

    class Runner:
        mode = None
        closed = False

        def run(self, mode, scope):
            assert state.locked
            assert scope == "online"
            self.mode = mode
            state.calls.append((mode, state.inside_timer))
            if state.inside_timer:
                state.timed_calls += 1
                if state.timed_calls == state.fail_timed_call:
                    raise RuntimeError("fixture GPU failure")

        def output(self, _):
            assert not state.inside_timer, "no output reads inside primary timer"
            state.output_reads += 1
            return np.array([0x3F80], dtype=np.uint16)

        @property
        def last_gpu_seconds(self):
            assert not state.inside_timer, "GPU timestamp read must follow primary timer"
            state.gpu_reads += 1
            if not state.gpu_available:
                return 0.0
            return 0.0015 if self.mode == "native" else 0.0007

        def close(self):
            assert state.locked
            self.closed = True

    runner = Runner()

    @contextmanager
    def lock():
        assert not state.locked
        state.locked = True
        try:
            yield
        finally:
            state.locked = False

    def timer():
        if not state.inside_timer:
            state.inside_timer = True
        else:
            state.nanoseconds += 2_000_000 if runner.mode == "native" else 1_000_000
            state.inside_timer = False
        return state.nanoseconds

    def prepare(*values):
        assert state.locked
        state.prepare_calls.append(values)
        if state.prepare_error is not None:
            raise state.prepare_error
        correct = script.metrics(np.ones(1), np.ones(1))
        quality = {
            mode: {stream: dict(correct) for stream in ("q", "k", "v")}
            for mode in ("native", "candidate")
        }
        if state.quality_rejected:
            quality["candidate"]["v"]["relative_l2"] = 0.2
        metadata = dict(
            transform_pass=state.transform_pass,
            per_stream_quality_error=quality,
            per_stream_implementation_error={mode: {"all": correct} for mode in quality},
            backend_provenance={"candidate_source_sha256": "CPU fixture"},
            quantization_error=correct,
        )

        def check(output):
            assert output.runner is runner
            assert not state.inside_timer
            return np.array([1.01 if state.online_error and runner.mode == "candidate" else 1.0])

        return SimpleNamespace(runner=runner), check, np.ones(1), np.ones(1), {}, metadata

    def environment():
        assert state.locked
        return {"device": "CPU fixture"}

    monkeypatch.setattr(script, "gpu_lock", lock)
    monkeypatch.setattr(script, "environment", environment)
    monkeypatch.setattr(script, "prepare", prepare)
    monkeypatch.setattr(script.time, "perf_counter_ns", timer)
    monkeypatch.setattr(
        script,
        "load",
        lambda _: {
            "cases": [
                {
                    "id": "h3.grouped_qkv.m128",
                    "op": "grouped_qkv",
                    "m": 128,
                    "k": 128,
                    "n": 1536,
                    "heads": 4,
                }
            ]
        },
    )
    args = SimpleNamespace(
        filter="m128",
        seed=29,
        repeat=4,
        warmup=2,
        tile=[64, 128, 4, 8],
        reverse=False,
        output=str(tmp_path / "paired.json"),
        k_specialization="dynamic",
        scratch="threadgroup",
        rms_order="simd",
    )
    return state, runner, args


@pytest.mark.parametrize("reverse", [False, True])
def test_one_runner_alternates_each_pair_and_records_primary_vs_gpu_time(script, fixture, reverse):
    state, runner, args = fixture
    args.reverse = reverse
    document = script.run(args)
    assert len(state.prepare_calls) == 1
    assert state.prepare_calls[0][1:] == ("candidate-qkv", "int8", 29, "normal", (64, 128, 4, 8))
    assert state.calls[:2] == [("native", False), ("candidate", False)]
    expected_warm = [
        (mode, False) for i in range(args.warmup) for mode in script.pair_order(i, reverse)
    ]
    expected_timed = [
        (mode, True) for i in range(args.repeat) for mode in script.pair_order(i, reverse)
    ]
    assert state.calls[2:] == expected_warm + expected_timed
    assert state.gpu_reads == args.repeat * 2
    assert state.output_reads == 6  # full finite checks only, before warmup/timing
    assert runner.closed and not state.locked
    assert document["measurement_complete"] and document["status"] == "ok"
    native, candidate = (document["results"][mode] for mode in ("native", "candidate"))
    assert native["timing"]["samples_ms"] == [2.0] * 4
    assert candidate["timing"]["samples_ms"] == [1.0] * 4
    assert native["gpu_command"]["samples_ms"] == [1.5] * 4
    np.testing.assert_allclose(candidate["gpu_command"]["samples_ms"], [0.7] * 4)
    assert document["paired_speed_ratio"]["median"] == 2
    assert document["paired_speed_ratio"]["ratio_of_medians"] == 2
    assert len(document["script_sha256"]) == len(document["config_sha256"]) == 64
    assert json.loads(Path(args.output).read_text()) == document


def test_per_stream_quality_rejection_still_times_both_paths(script, fixture):
    state, _, args = fixture
    state.quality_rejected = True
    document = script.run(args)
    assert document["status"] == "ok" and document["measurement_complete"]
    assert document["quality_status"] == "quality_rejected"
    assert document["results"]["native"]["quality_pass"] is True
    assert document["results"]["candidate"]["quality_pass"] is False
    assert state.timed_calls == 8


def test_failed_preparation_check_prevents_timing_and_is_saved(script, fixture):
    state, runner, args = fixture
    state.transform_pass = False
    document = script.run(args)
    assert document["status"] == "incorrect" and not document["measurement_complete"]
    assert state.calls == [] and state.gpu_reads == 0
    assert document["preparation_metadata"]["transform_pass"] is False
    assert json.loads(Path(args.output).read_text())["status"] == "incorrect"
    assert runner.closed


def test_online_validation_failure_prevents_warmup_and_timing(script, fixture):
    state, _, args = fixture
    state.online_error = True
    document = script.run(args)
    assert document["status"] == "incorrect"
    assert state.calls == [("native", False), ("candidate", False)]
    assert state.timed_calls == 0
    assert document["results"]["candidate"]["implementation_error"]["relative_l2"] > 0.008


def test_native_unavailable_is_explicit_without_fallback(script, fixture):
    state, _, args = fixture
    state.prepare_error = NotImplementedError("fixture unavailable")
    document = script.run(args)
    assert document["status"] == "unsupported"
    assert document["reason"] == "fixture unavailable"
    assert state.calls == []


def test_partial_failure_keeps_completed_pairs_and_raw_timings(script, fixture):
    state, runner, args = fixture
    state.fail_timed_call = 3
    document = script.run(args)
    assert document["status"] == "error" and not document["measurement_complete"]
    assert document["error_type"] == "RuntimeError"
    assert len(document["pairs"]) == 2  # last pair is explicitly incomplete
    assert document["paired_speed_ratio"]["samples"] == [2.0]
    assert document["results"]["native"]["timing"]["samples_ms"] == [2.0]
    assert document["results"]["candidate"]["timing"]["samples_ms"] == [1.0]
    assert runner.closed
    assert json.loads(Path(args.output).read_text())["measurement_complete"] is False


def test_unavailable_gpu_timestamps_do_not_invalidate_primary_timing(script, fixture):
    state, _, args = fixture
    state.gpu_available = False
    document = script.run(args)
    assert document["status"] == "ok"
    for result in document["results"].values():
        assert result["gpu_command"]["samples_ms"] == [None] * 4
        assert result["gpu_command"]["available_samples"] == 0
        assert "median_ms" not in result["gpu_command"]
        assert result["timing"]["available_samples"] == 4


def test_existing_results_are_immutable_before_prepare(script, fixture):
    state, _, args = fixture
    Path(args.output).write_text("existing evidence")
    with pytest.raises(FileExistsError):
        script.run(args)
    assert state.prepare_calls == []
    assert Path(args.output).read_text() == "existing evidence"


@pytest.mark.parametrize("ids", [[], ["m128", "m128-other"]])
def test_filter_requires_one_case_before_prepare(script, fixture, monkeypatch, ids):
    state, _, args = fixture
    monkeypatch.setattr(script, "load", lambda _: {"cases": [{"id": value} for value in ids]})
    with pytest.raises(ValueError, match="exactly one"):
        script.run(args)
    assert state.prepare_calls == []


def test_cli_defaults_match_requested_small_shape_crosscheck(script, monkeypatch, tmp_path):
    received = []

    def run(args):
        received.append(args)
        return dict(status="ok", measurement_complete=True)

    monkeypatch.setattr(script, "run", run)
    assert script.main(["--output", str(tmp_path / "unused.json")]) == 0
    args = received[0]
    assert (args.filter, args.seed, args.repeat, args.warmup) == ("m128", 29, 60, 30)
    assert args.tile == [64, 128, 4, 8]
    assert (args.k_specialization, args.scratch, args.rms_order) == (
        "dynamic",
        "threadgroup",
        "simd",
    )

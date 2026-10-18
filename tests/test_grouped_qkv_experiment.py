"""Grouped GPU tests require the external repository GPU lock, like the suite.

CPU protocol tests deliberately disable MLX imports and never create a device.
"""

import json
import platform
import sys

import numpy as np
import pytest

from m5kernels import engine
from m5kernels.grouped_qkv_experiment import NativeCall, NativeOutputs, _paired_metrics, prepare
from m5kernels.grouped_qkv_reference import (
    bf16_decode_reference,
    bf16_encode_reference,
    grouped_qkv_reference,
    native_int8_quantize_reference,
)
from m5kernels.numerics import metrics


def case(rows=129, rms_order="simd"):
    return dict(
        id="grouped.tail",
        op="grouped_qkv",
        m=rows,
        k=256,
        n=4 * 3 * 128,
        heads=4,
        head_dim=128,
        rope_half=48,
        rms_order=rms_order,
    )


requires_native_gpu = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64", reason="requires native Apple Metal"
)


@pytest.mark.metal
@requires_native_gpu
@pytest.mark.parametrize("backend", ["native-qkv", "candidate-qkv"])
@pytest.mark.parametrize(
    "rows,distribution,rms_order,tile",
    [
        (129, "normal", "simd", (32, 64, 4, 3)),
        (129, "zeros", "native", (64, 128, 4, 3)),
        (257, "outliers", "simd", (64, 128, 4, 3)),
        (257, "tiny", "native", (32, 64, 4, 3)),
    ],
)
def test_grouped_run_case_preserves_native_contract(backend, rows, distribution, rms_order, tile):
    result = engine.run_case(
        case(rows, rms_order),
        backend,
        "int8",
        seed=163,
        distribution=distribution,
        warmup=1,
        repeat=2,
        tile=tile,
    )
    assert result["status"] == "ok", result
    assert result["transform_pass"] and result["output_finite"] and result["online_output_finite"]
    assert result["sample_heads"] == [0, 3]
    assert len(result["sample_cols"]) == 2 * 3 * 128
    assert result["padded_rows"] == ((rows + 127) // 128) * 128
    assert result["padding_valid"] and result["packed_storage_valid"]
    assert result["rms_order"] == rms_order
    assert result["actual_zero_row_scale"] == float(np.float32(1) / np.float32(127))
    assert result["padded_row_scale"] == 1
    assert "no Hadamard" in result["quantization_policy"]
    assert result["core_dispatches"] == 1 and result["online_dispatches"] == 2
    for preparation in result["preparation_checks"].values():
        assert preparation["passed"]
        assert preparation["scale_error"]["relative_l2"] <= 1e-5
        assert preparation["max_code_difference"] <= 1
    for mode in ("native", "candidate"):
        for stream in ("q", "k", "v"):
            implementation = result["per_stream_implementation_error"][mode][stream]
            assert implementation["finite"] and implementation["relative_l2"] <= 0.008
    for stream in ("q", "k", "v"):
        paired = result["paired_full_output_error"][stream]
        assert paired["finite"] and paired["relative_l2"] <= 0.008
        assert paired["elements"] == 4 * rows * 128
    mode = "native" if backend == "native-qkv" else "candidate"
    stream_quality_pass = all(
        error["finite"] and error["relative_l2"] <= 0.05
        for error in result["per_stream_quality_error"][mode].values()
    )
    assert result["quality_pass"] == (
        stream_quality_pass and result["quality_error"]["relative_l2"] <= 0.05
    )
    assert len(result["timing"]["samples_ms"]) == 2
    for stage in ("activation_prepare", "online_activation_plus_gemm", "offline_weight_prepare"):
        assert len(result["stages"][stage]["samples_ms"]) == 2
    if distribution == "zeros":
        assert result["quality_pass"] and result["quality_error"]["relative_l2"] == 0


@pytest.mark.metal
@requires_native_gpu
@pytest.mark.parametrize("rms_order", ["simd", "native"])
def test_full_grouped_output_and_v_layout_against_independent_reference(rms_order):
    prepared = prepare(case(129, rms_order), "candidate-qkv", "int8", 167, "tiny", (32, 64, 4, 3))
    fn, extract, expected_sample, _, stages, _ = prepared
    runner = fn.runner
    try:
        aq = runner.output("input_int8")[:129].astype(np.float64)
        sa = runner.output("input_scales")[:129].copy()
        wq = runner.output("weight_int8").astype(np.float64)
        sw = runner.output("weight_scales").copy()
        decoded_a = aq * sa[:, None]
        decoded_w = (wq * sw[:, None]).reshape(4, 3, 128, 256)
        gamma_q, gamma_k, cos, sin = (
            bf16_decode_reference(runner.output(name))
            for name in ("q_gamma", "k_gamma", "cos", "sin")
        )
        reference = grouped_qkv_reference(
            decoded_a, decoded_w, gamma_q, gamma_k, cos, sin, return_projection=True
        )
        expected_v = reference["projection_bf16"][:, :, 2, :].transpose(1, 0, 2)
        np.testing.assert_array_equal(reference["outputs"][2], expected_v)
        assert np.max(np.abs(expected_v)) > 0
        assert np.max(np.abs(expected_v)) < 0.01
        for mode in ("native", "candidate"):
            runner.run(mode=mode, scope="core")
            for stream, expected in zip(("q", "k", "v"), reference["outputs"], strict=True):
                bits = runner.output(stream)
                assert bits.dtype == np.uint16 and bits.shape == (4, 129, 128)
                actual = bf16_decode_reference(bits)
                assert np.isfinite(actual).all()
                assert metrics(actual, expected)["relative_l2"] <= 0.008
        # Online packing must preserve the same full-head sampled semantics.
        online_output = stages["online_activation_plus_gemm"]()
        engine.evaluate(online_output)
        assert engine.all_finite(online_output)
        assert metrics(extract(online_output), expected_sample)["relative_l2"] <= 0.008
    finally:
        runner.close()


@pytest.mark.metal
@requires_native_gpu
@pytest.mark.parametrize("width", [256, 5376])
@pytest.mark.parametrize("tile", [(32, 64, 4, 3), (64, 128, 4, 3)])
@pytest.mark.parametrize("scratch", ["threadgroup", "device"])
def test_native_k_specialization_tail_matches_full_k_reference(width, tile, scratch):
    # K256 executes two fixed128 iterations; K5376 uses one full-K operation.
    # Tiny nonzero projections expose scale errors that RMS could otherwise hide.
    c = case(129, "native")
    c.update(k=width, k_specialization="native", scratch=scratch)
    fn, extract, expected, _, stages, metadata = prepare(
        c, "candidate-qkv", "int8", 179, "tiny", tile
    )
    try:
        assert metadata["k_specialization"] == "native"
        assert metadata["k_tile"] == (5376 if width == 5376 else 128)
        assert metadata["scratch"] == scratch
        assert metadata["padded_rows"] == 256 and metadata["padding_valid"]
        assert metadata["sample_heads"] == [0, 3]
        assert metadata["sample_rows"][0] == 0 and metadata["sample_rows"][-1] == 128
        assert metadata["transform_pass"], metadata
        assert metadata["output_stores_complete"] == {"native": True, "candidate": True}
        for stream in ("q", "k", "v"):
            paired = metadata["paired_full_output_error"][stream]
            assert paired["finite"] and paired["relative_l2"] <= 0.008
            assert paired["elements"] == 4 * 129 * 128
            for mode in ("native", "candidate"):
                error = metadata["per_stream_implementation_error"][mode][stream]
                assert error["finite"] and error["relative_l2"] <= 0.008
        # The independently sampled reference includes all128 channels of each
        # Q/K/V head and every K element, including the specialized final slice.
        for operation in (fn, stages["online_activation_plus_gemm"]):
            output = operation()
            engine.evaluate(output)
            assert engine.all_finite(output)
            assert metrics(extract(output), expected)["relative_l2"] <= 0.008
        v = bf16_decode_reference(fn.runner.output("v"))
        assert 0 < np.max(np.abs(v)) < 0.01
    finally:
        fn.runner.close()


@pytest.fixture
def no_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)


class SynchronousOutput:
    def __init__(self, events, values=None, finite=True):
        self.events = events
        self.values = np.ones(3) if values is None else values
        self.finite = finite

    def __m5_evaluate__(self):
        self.events.append("evaluate")

    def __m5_all_finite__(self):
        self.events.append("finite")
        return np.bool_(self.finite)


class SynchronousCall:
    def __init__(self, events, output):
        self.events, self.output = events, output

    def __call__(self):
        self.events.append("call")
        return self.output

    def synchronize(self):
        self.events.append("synchronize")


@pytest.mark.parametrize("finite", [False, True])
def test_evaluate_and_finite_use_custom_protocol_without_mlx(no_mlx, finite):
    events = []
    output = SynchronousOutput(events, finite=finite)
    engine.evaluate(output)
    assert engine.all_finite(output) is finite
    assert events == ["evaluate", "finite"]


def test_timed_synchronous_protocol_has_no_download_or_mlx(no_mlx, monkeypatch):
    events = []
    call = SynchronousCall(events, SynchronousOutput(events))
    times = iter([1_000_000, 2_000_000, 3_000_000, 6_000_000, 7_000_000, 12_000_000])
    monkeypatch.setattr(engine.time, "perf_counter_ns", lambda: next(times))
    result = engine.timed(call, warmup=2, repeat=3)
    assert (
        events == ["call", "evaluate"] * 2 + ["synchronize", "call", "evaluate", "synchronize"] * 3
    )
    assert result["samples_ms"] == [1.0, 3.0, 5.0]
    assert result["median_ms"] == 3.0 and result["min_ms"] == 1.0
    assert result["p10_ms"] == pytest.approx(1.4)
    assert result["p90_ms"] == pytest.approx(4.6)
    assert result["kind"] == "synchronized_host_wall"
    assert result["warmup"] == 2 and result["repeat"] == 3


def test_native_call_evaluation_does_not_read_outputs(no_mlx):
    class Runner:
        def __init__(self):
            self.calls = []

        def run(self, *, mode, scope):
            self.calls.append((mode, scope))

        def output(self, name):
            pytest.fail(f"timed evaluation downloaded {name}")

    runner = Runner()
    call = NativeCall(runner, "candidate", "online")
    engine.timed(call, warmup=1, repeat=2)
    assert runner.calls == [("candidate", "online")] * 3


@pytest.mark.parametrize("bad_stream", [None, "q", "k", "v"])
def test_native_output_finite_checks_all_three_streams(no_mlx, bad_stream):
    class Runner:
        def output(self, name):
            bits = np.full((2, 3, 128), 0x3F80, dtype=np.uint16)
            if name == bad_stream:
                bits[-1, -1, -1] = 0x7FC1
            return bits

    assert engine.all_finite(NativeOutputs(Runner())) is (bad_stream is None)


def test_per_stream_quality_rejection_is_not_hidden_by_combined_norm(no_mlx, monkeypatch):
    # Large V makes combined error tiny even though Q exceeds its quality limit.
    events = []
    actual = np.array([1.0, 1.0, 1000.0])
    original = np.array([1.1, 1.0, 1000.0])
    output = SynchronousOutput(events, values=actual)
    call = SynchronousCall(events, output)
    monkeypatch.setattr(
        engine,
        "prepare_grouped_qkv",
        lambda *args: (
            call,
            lambda value: value.values,
            actual,
            original,
            {},
            dict(mac_operations=0, transform_pass=True, component_quality_pass=False),
        ),
    )
    result = engine.run_case(case(), "candidate-qkv", "int8", warmup=0, repeat=1)
    assert result["status"] == "ok", result
    assert result["quality_error"]["relative_l2"] < 0.05
    assert not result["quality_pass"]
    assert "timing" in result


@pytest.mark.parametrize("component_pass", [False, True])
@pytest.mark.parametrize("global_pass", [False, True])
def test_component_quality_gate_only_tightens_global_gate(
    no_mlx, monkeypatch, component_pass, global_pass
):
    events = []
    actual = np.ones(3)
    original = actual.copy() if global_pass else actual * 2
    call = SynchronousCall(events, SynchronousOutput(events, values=actual))
    monkeypatch.setattr(
        engine,
        "prepare_grouped_qkv",
        lambda *args: (
            call,
            lambda value: value.values,
            actual,
            original,
            {},
            dict(mac_operations=0, transform_pass=True, component_quality_pass=component_pass),
        ),
    )
    result = engine.run_case(case(), "candidate-qkv", "int8", warmup=0, repeat=1)
    assert result["status"] == "ok", result
    assert (result["quality_error"]["relative_l2"] <= 0.05) is global_pass
    assert result["quality_pass"] is (component_pass and global_pass)


@pytest.mark.parametrize("payload", [0x7F80, 0xFF80, 0x7FC1])
@pytest.mark.parametrize("location", ["actual", "expected", "both"])
def test_paired_metrics_nonfinite_outputs_are_json_safe(no_mlx, payload, location):
    actual = np.full((3, 2, 4), 0x3F80, dtype=np.uint16)
    expected = actual.copy()
    if location in ("actual", "both"):
        actual[-1, -1, -1] = payload
    if location in ("expected", "both"):
        expected[-1, -1, -1] = payload
    result = _paired_metrics(actual, expected)
    assert not result["finite"] and not result["exact_bits"]
    assert result["elements"] == 24
    assert result["bit_mismatch_elements"] == (0 if location == "both" else 1)
    for name in ("relative_l2", "max_abs", "rmse", "cosine"):
        assert result[name] is None
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded) == result


def test_paired_metrics_match_independent_dense_metrics(no_mlx):
    expected = bf16_encode_reference(np.arange(24).reshape(3, 2, 4) - 12)
    actual = expected.copy()
    actual[0, 1, 1] = bf16_encode_reference(3.0)
    actual[-1, -1, -1] = bf16_encode_reference(-5.0)
    result = _paired_metrics(actual, expected)
    reference = metrics(bf16_decode_reference(actual), bf16_decode_reference(expected))
    assert result["finite"] and not result["exact_bits"]
    assert result["bit_mismatch_elements"] == 2 and result["elements"] == 24
    for name in ("relative_l2", "max_abs", "rmse", "cosine"):
        assert result[name] == pytest.approx(reference[name], rel=2e-15)
    json.dumps(result, allow_nan=False)


def test_paired_metrics_distinguish_signed_zero_bits_from_numerical_error(no_mlx):
    expected = np.zeros((2, 3, 4), dtype=np.uint16)
    exact = _paired_metrics(expected, expected.copy())
    assert exact["finite"] and exact["exact_bits"]
    assert exact["bit_mismatch_elements"] == 0 and exact["relative_l2"] == 0
    assert exact["cosine"] == 1
    actual = expected.copy()
    actual[-1, -1, -1] = 0x8000
    signed_zero = _paired_metrics(actual, expected)
    assert signed_zero["finite"] and not signed_zero["exact_bits"]
    assert signed_zero["bit_mismatch_elements"] == 1
    assert signed_zero["relative_l2"] == signed_zero["max_abs"] == signed_zero["rmse"] == 0
    assert signed_zero["cosine"] == 1
    json.dumps(signed_zero, allow_nan=False)


@pytest.mark.parametrize("missing_store", [None, "native", "candidate", "both"])
def test_prepare_poisons_both_paths_and_rejects_missing_unsampled_stores(
    no_mlx, monkeypatch, missing_store
):
    from m5kernels import native_qkv

    runners = []

    class CPUZeroRunner:
        """Only zero fixtures: calculate packing on CPU, synthesize zero dots."""

        def __init__(self, rows, width, heads, **kwargs):
            self.rows, self.width, self.heads = rows, width, heads
            self.buffers = {}
            self.core_modes = []
            self.metadata = {"driver": "CPU-only regression fixture"}
            runners.append(self)

        def upload(self, name, value):
            self.buffers[name] = value.copy()

        def output(self, name):
            return self.buffers[name]

        def run(self, *, mode, scope):
            if scope in ("quantize", "weight_quantize"):
                prefix = "input" if scope == "quantize" else "weight"
                values = bf16_decode_reference(self.buffers[prefix + "_bf16"])
                packed = native_int8_quantize_reference(values)
                self.buffers[prefix + "_int8"] = packed["codes"]
                self.buffers[prefix + "_scales"] = packed["scales"]
                return
            assert scope == "core"
            # Both independent correctness invocations must begin poisoned;
            # candidate must never inherit native's otherwise-correct stores.
            for stream in ("q", "k", "v"):
                np.testing.assert_array_equal(self.buffers[stream], 0x7F7F)
            self.core_modes.append(mode)
            for stream in ("q", "k", "v"):
                self.buffers[stream].fill(0)
            if missing_store in (mode, "both"):
                # Head1 is absent from sampled heads0/3; only complete-store
                # validation can catch the same omission in both kernels.
                self.buffers["v"][1, 63, 127] = 0x7F7F

    monkeypatch.setattr(native_qkv, "NativeQKVRunner", CPUZeroRunner)
    c = case(128)
    c["k"] = 128
    _, _, expected, original, _, metadata = prepare(
        c, "candidate-qkv", "int8", 173, "zeros", (32, 64, 4, 3)
    )
    assert runners[0].core_modes == ["native", "candidate"]
    assert metadata["sample_heads"] == [0, 3]
    np.testing.assert_array_equal(expected, 0)
    np.testing.assert_array_equal(original, 0)
    assert metadata["transform_pass"] is (missing_store is None)
    assert metadata["output_stores_complete"] == {
        "native": missing_store not in ("native", "both"),
        "candidate": missing_store not in ("candidate", "both"),
    }
    if missing_store == "both":
        assert metadata["paired_full_output_error"]["v"]["exact_bits"]
        assert metadata["per_stream_implementation_error"]["candidate"]["v"]["relative_l2"] == 0
    assert metadata["quantization_error"]["relative_l2"] == 0
    for stream in ("q", "k", "v"):
        assert metadata["per_stream_quantization_error"][stream]["relative_l2"] == 0
    json.dumps(metadata, allow_nan=False)

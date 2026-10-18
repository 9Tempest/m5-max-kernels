"""Matched grouped QKV comparison using native kernels and a common Metal driver."""

import numpy as np

from .grouped_qkv import generate
from .grouped_qkv_reference import (
    bf16_encode_reference,
    bf16_decode_reference,
    grouped_qkv_reference,
    native_int8_quantize_reference,
)
from .numerics import metrics, sample_indices

STREAMS = ("q", "k", "v")


class NativeOutputs:
    """Resident BF16 output handle; evaluation never downloads or converts data."""

    def __init__(self, runner):
        self.runner = runner

    def __m5_evaluate__(self):
        # NativeRunner.run already waited for completion of its command buffer.
        pass

    def __m5_all_finite__(self):
        return all(np.all((self.runner.output(s) & 0x7F80) != 0x7F80) for s in STREAMS)


class NativeCall:
    def __init__(self, runner, mode, scope):
        self.runner, self.mode, self.scope = runner, mode, scope
        self.result = NativeOutputs(runner)

    def __call__(self):
        self.runner.run(mode=self.mode, scope=self.scope)
        return self.result

    def synchronize(self):
        # No work is outstanding: each call commits and waits in the same driver.
        pass

    def capture(self, path, iterations=1):
        if iterations != 1:
            raise NotImplementedError("native grouped QKV capture supports one complete invocation")
        self.runner.capture(path, mode=self.mode, scope=self.scope)


def _paired_metrics(actual_bits, expected_bits):
    """Full-buffer comparison, bounded to one head of decoded storage at a time."""
    finite = bool(
        np.all((actual_bits & 0x7F80) != 0x7F80) and np.all((expected_bits & 0x7F80) != 0x7F80)
    )
    if not finite:
        return dict(
            finite=False,
            relative_l2=None,
            max_abs=None,
            rmse=None,
            cosine=None,
            bit_mismatch_elements=int(np.count_nonzero(actual_bits != expected_bits)),
            elements=int(actual_bits.size),
            exact_bits=False,
        )
    diff2, ref2, actual2, dot, max_abs, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    mismatch = 0
    for a, b in zip(actual_bits, expected_bits, strict=True):
        mismatch += int(np.count_nonzero(a != b))
        x, y = bf16_decode_reference(a), bf16_decode_reference(b)
        d = x - y
        diff2 += float(np.sum(d * d))
        ref2 += float(np.sum(y * y))
        actual2 += float(np.sum(x * x))
        dot += float(np.sum(x * y))
        max_abs = max(max_abs, float(np.max(np.abs(d))))
        count += x.size
    return dict(
        finite=finite,
        relative_l2=float(np.sqrt(diff2) / max(np.sqrt(ref2), 1e-30)),
        max_abs=max_abs,
        rmse=float(np.sqrt(diff2 / count)),
        cosine=float(dot / max(np.sqrt(ref2 * actual2), 1e-30)) if ref2 or actual2 else 1.0,
        bit_mismatch_elements=mismatch,
        elements=count,
        exact_bits=mismatch == 0,
    )


def _quant_check(bits, codes, scales):
    original = bf16_decode_reference(bits)
    ref = native_int8_quantize_reference(original)
    decoded = codes.astype(np.float64) * scales[:, None]
    error = metrics(decoded, ref["decoded"])
    scale_error = metrics(scales, ref["scales"])
    code_difference = int(np.max(np.abs(codes.astype(np.int16) - ref["codes"].astype(np.int16))))
    return dict(
        implementation_error=error,
        scale_error=scale_error,
        max_code_difference=code_difference,
        ideal_quantization_error=ref["quantization_error"],
        actual_total_error=metrics(decoded, original),
        passed=bool(
            error["finite"]
            and error["relative_l2"] <= 0.008
            and scale_error["finite"]
            and scale_error["relative_l2"] <= 1e-5
            and code_difference <= 1
        ),
    )


def prepare(c, backend, precision, seed, distribution, tile):
    if backend not in ("native-qkv", "candidate-qkv") or precision != "int8":
        raise NotImplementedError("matched grouped QKV uses native-qkv/candidate-qkv INT8 only")
    from .native_qkv import NativeQKVRunner

    m, k, heads = c["m"], c["k"], c["heads"]
    n, dim, half = heads * 3 * 128, c.get("head_dim", 128), c.get("rope_half", 48)
    if c["n"] != n or dim != 128 or half != 48:
        raise ValueError("native grouped QKV requires N=heads*3*128, D128 and rope_half48")
    if distribution not in ("normal", "outliers", "zeros", "tiny"):
        raise ValueError("unknown grouped QKV distribution")
    order = c.get("rms_order", "simd")
    specialization = c.get("k_specialization", "dynamic")
    scratch = c.get("scratch", "threadgroup")
    eps = c.get("epsilon", 1e-5)
    source = generate(
        tile, rms_order=order, k_specialization=specialization, input_dim=k, scratch=scratch
    )
    runner = NativeQKVRunner(m, k, heads, candidate_source=source, candidate_tile=tile, epsilon=eps)
    rng = np.random.default_rng(seed)

    def fixture(shape, scale=1.0, outlier_period=None):
        values = rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
        if distribution == "outliers" and outlier_period:
            values[..., ::outlier_period] *= 20
        if distribution == "zeros":
            values.fill(0)
        return bf16_encode_reference(values)

    x = fixture((m, k), 1e-4 if distribution == "tiny" else 1, 127)
    weights = fixture((n, k), k**-0.5, 131)
    qg = bf16_encode_reference(1 + 0.2 * rng.standard_normal(128, dtype=np.float32))
    kg = bf16_encode_reference(1 + 0.2 * rng.standard_normal(128, dtype=np.float32))
    angles = rng.uniform(-np.pi, np.pi, (m, half)).astype(np.float32)
    cos, sin = bf16_encode_reference(np.cos(angles)), bf16_encode_reference(np.sin(angles))
    for name, array in dict(
        input_bf16=x, weight_bf16=weights, q_gamma=qg, k_gamma=kg, cos=cos, sin=sin
    ).items():
        runner.upload(name, array)
    runner.run(mode="native", scope="weight_quantize")
    runner.run(mode="native", scope="quantize")
    rows = sample_indices(m)
    sampled_heads = sample_indices(heads, 2)
    cols = (sampled_heads[:, None] * 384 + np.arange(384)[None, :]).reshape(-1)
    aq, sa = runner.output("input_int8"), runner.output("input_scales")
    bq, sb = runner.output("weight_int8"), runner.output("weight_scales")
    aq_sample, sa_sample = aq[rows].copy(), sa[rows].copy()
    bq_sample, sb_sample = bq[cols].copy(), sb[cols].copy()
    prep = dict(
        activation=_quant_check(x[rows], aq_sample, sa_sample),
        weight=_quant_check(weights[cols], bq_sample, sb_sample),
    )
    padding_valid = bool(np.all(aq[m:] == 0) and np.all(sa[m:] == 1))
    storage_valid = bool(
        np.all(np.isfinite(sa))
        and np.all(sa > 0)
        and np.all(np.isfinite(sb))
        and np.all(sb > 0)
        and np.all(aq >= -127)
        and np.all(bq >= -127)
    )
    shared_args = tuple(map(bf16_decode_reference, (qg, kg, cos[rows], sin[rows])))
    ref_x = bf16_decode_reference(x[rows])
    ref_w = bf16_decode_reference(weights[cols]).reshape(len(sampled_heads), 3, 128, k)
    original = grouped_qkv_reference(ref_x, ref_w, *shared_args, epsilon=eps)
    decoded_x = aq_sample.astype(np.float64) * sa_sample[:, None]
    decoded_w = (bq_sample.astype(np.float64) * sb_sample[:, None]).reshape(
        len(sampled_heads), 3, 128, k
    )
    expected = grouped_qkv_reference(decoded_x, decoded_w, *shared_args, epsilon=eps)
    quantization_error = {
        stream: metrics(decoded, unquantized)
        for stream, decoded, unquantized in zip(STREAMS, expected, original, strict=True)
    }

    def extract_streams():
        return tuple(
            bf16_decode_reference(runner.output(s)[sampled_heads[:, None], rows[None, :], :])
            for s in STREAMS
        )

    # Poison correctness outputs independently: a missing store must not inherit
    # the other implementation's correct result. This is outside timed regions.
    sentinel = np.full((heads, m, 128), 0x7F7F, dtype=np.uint16)

    def poison_outputs():
        for stream in STREAMS:
            runner.upload(stream, sentinel)

    def stores_complete():
        return all(not np.any(runner.output(s) == 0x7F7F) for s in STREAMS)

    poison_outputs()
    runner.run(mode="native", scope="core")
    native_stores = stores_complete()
    native_bits = tuple(runner.output(s).copy() for s in STREAMS)
    native_sample = extract_streams()
    poison_outputs()
    runner.run(mode="candidate", scope="core")
    candidate_stores = stores_complete()
    candidate_sample = extract_streams()
    paired = {
        s: _paired_metrics(runner.output(s), b) for s, b in zip(STREAMS, native_bits, strict=True)
    }
    del native_bits
    per_stream = {
        mode: {
            s: metrics(value, ref) for s, value, ref in zip(STREAMS, values, expected, strict=True)
        }
        for mode, values in [("native", native_sample), ("candidate", candidate_sample)]
    }
    total_quality = {
        mode: {
            s: metrics(value, ref) for s, value, ref in zip(STREAMS, values, original, strict=True)
        }
        for mode, values in [("native", native_sample), ("candidate", candidate_sample)]
    }
    finite = NativeOutputs(runner).__m5_all_finite__()
    transformed = bool(
        finite
        and native_stores
        and candidate_stores
        and padding_valid
        and storage_valid
        and all(v["passed"] for v in prep.values())
        and all(
            v["finite"] and v["relative_l2"] <= 0.008
            for values in per_stream.values()
            for v in values.values()
        )
        and all(v["finite"] and v["relative_l2"] <= 0.008 for v in paired.values())
    )
    mode = "native" if backend == "native-qkv" else "candidate"
    selected_quality_pass = all(
        v["finite"] and v["relative_l2"] <= 0.05 for v in total_quality[mode].values()
    )

    def check(output):
        values = tuple(
            bf16_decode_reference(output.runner.output(s)[sampled_heads[:, None], rows[None, :], :])
            for s in STREAMS
        )
        return np.concatenate([value.ravel() for value in values])

    metadata = dict(
        compute_path=backend,
        execution_driver="same Objective-C Metal device/queue, buffers, command-buffer boundaries and safe compiler options",
        backend_provenance=runner.metadata,
        reference="NumPy float64 full-K projection for complete sampled Q/K/V heads, explicit BF16 projection/output",
        sample_rows=rows.tolist(),
        sample_heads=sampled_heads.tolist(),
        sample_cols=cols.tolist(),
        input_dtype="bf16",
        output_dtype="bf16",
        output_layout="three contiguous [heads,M,128] BF16 outputs",
        weight_layout="native grouped [head,Q/K/V,128,K] supplied identically to both paths",
        weight_layout_conversion_required=False,
        output_layout_conversion="included in each projection epilogue; no omitted transpose or V cost",
        projection_rounding="BF16 after FP32 per-row/per-output-channel dequantization, before Q/K RMS",
        normalization_rounding="none; FP32 gamma/RoPE before BF16 Q/K output",
        v_policy="BF16 projection only, no normalization or RoPE",
        rms_order=order,
        k_specialization=specialization,
        scratch=scratch,
        candidate_threadgroup_scratch_bytes=tile[0] * 128 * 2 if scratch == "threadgroup" else 0,
        k_tile=(5376 if k == 5376 else 128) if specialization == "native" else None,
        epsilon=eps,
        rope_table_origin="synthetic independent angles, not a model position trace",
        quantization_policy="native reciprocal-multiply W8A8, scalar quantizer, clip1, no Hadamard",
        actual_zero_row_scale=float(np.float32(1) / np.float32(127)),
        padded_row_scale=1.0,
        padded_rows=int(aq.shape[0]),
        padding_valid=padding_valid,
        packed_storage_valid=storage_valid,
        output_stores_complete=dict(native=native_stores, candidate=candidate_stores),
        preparation_checks=prep,
        paired_full_output_error=paired,
        per_stream_implementation_error=per_stream,
        per_stream_quality_error=total_quality,
        per_stream_quantization_error=quantization_error,
        quantization_error=metrics(
            np.concatenate([v.ravel() for v in expected]),
            np.concatenate([v.ravel() for v in original]),
        ),
        per_stream_quality_pass=selected_quality_pass,
        component_quality_pass=selected_quality_pass,
        transform_pass=transformed,
        main_timing_scope="resident packed activation, grouped Q/K/V projection, Q/K RMS/RoPE and all three head-major BF16 outputs",
        online_timing_scope="native BF16 activation INT8 quantization including padded rows + entire grouped QKV core",
        offline_weight_preparation_in_online=False,
        offline_weight_preparation="same native BF16-to-INT8 scalar quantizer on grouped weights",
        online_dispatches=2,
        core_dispatches=1,
        mac_operations=2 * m * n * k,
    )
    stages = dict(
        activation_prepare=NativeCall(runner, mode, "quantize"),
        online_activation_plus_gemm=NativeCall(runner, mode, "online"),
        offline_weight_prepare=NativeCall(runner, mode, "weight_quantize"),
    )
    return (
        NativeCall(runner, mode, "core"),
        check,
        np.concatenate([v.ravel() for v in expected]),
        np.concatenate([v.ravel() for v in original]),
        stages,
        metadata,
    )

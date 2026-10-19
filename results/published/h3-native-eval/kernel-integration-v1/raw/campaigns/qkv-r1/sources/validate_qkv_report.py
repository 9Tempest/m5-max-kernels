"""Strict CPU contract for qkv_validate.c; provenance belongs to the caller."""

from __future__ import annotations

import math


SCHEMA = "h3_native_qkv_v1"
LIMIT = 0.008


def _same(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"QKV report mismatch: {label}")
    if isinstance(expected, dict):
        for key in expected:
            _same(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _same(left, right, label + f"[{index}]")


def _number(value, label, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"invalid finite QKV number: {label}")
    return value


def _metric(value, count, label, gate=False):
    if not isinstance(value, dict):
        raise ValueError(f"missing QKV metric: {label}")
    _same(value.get("count"), count, label + ".count")
    _same(value.get("nonfinite"), 0, label + ".nonfinite")
    for key in ("relative_l2", "rmse", "max_abs"):
        _number(value.get(key), label + "." + key)
    if gate and value["relative_l2"] > LIMIT:
        raise ValueError(f"QKV implementation exceeds unchanged BF16 tolerance: {label}")


def validate(report, rows, seed, policy, warmups=3, repeats=10):
    """Return status after validation; numerical failures raise, raw JSON stays intact.

    M<128 has an explicit unsupported result and no GPU execution. Other requests
    must complete all independent-reference and paired contracts before timings
    can be used. This validates implementation, not model quality or promotion.
    """
    if not isinstance(report, dict) or policy not in ("native", "h256"):
        raise ValueError("invalid QKV request/report")
    _same(report.get("schema"), SCHEMA, "schema")
    _same(report.get("seed"), seed, "seed")
    if rows < 128:
        _same(report.get("status"), "unsupported", "small native shape")
        _same(report.get("rows"), rows, "unsupported rows")
        if not isinstance(report.get("reason"), str) or not report["reason"]:
            raise ValueError("unsupported QKV report lacks reason")
        if "timings" in report:
            raise ValueError("unsupported shape must not contain GPU timings")
        return "unsupported"
    _same(report.get("status"), "completed", "status")
    _same(report.get("operand_policy"), policy + "-int8", "operand policy")
    _same(
        report.get("shape"),
        dict(
            rows=rows,
            padded_rows=(rows + 127) & ~127,
            input_dim=5376,
            heads=56,
            head_dim=128,
            rope_half=48,
        ),
        "shape",
    )
    # dict equality alone admits bool/int substitutions; validate each scalar.
    for key, value in report["shape"].items():
        if type(value) is not int:
            raise ValueError("QKV shape is not integer: " + key)
    expected = "h3_qkv_project_split_int8_rope_"
    expected += "local_scales_" if rows > 2048 else ""
    _same(report.get("baseline_shader"), expected + "nax_r128_k5376_morton4", "native path")
    _same(
        report.get("baseline_selection_basis"),
        "native source default, runtime overrides rejected",
        "native path basis",
    )
    _same(report.get("candidate_shader"), "h3_m5_qkv_rope_int8_r64_swizzle8", "candidate")
    _same(report.get("candidate_tile"), [64, 128, 4, 8], "candidate tile")
    _same(report.get("baseline_slow_flags"), [0, 0, 0], "native defaults")
    _same(report.get("implementation_relative_l2_limit"), LIMIT, "tolerance")
    _same(report.get("warmups"), warmups, "warmups")
    _same(report.get("repeats"), repeats, "repeats")
    _same(report.get("first_candidate_parity"), int(seed == 29), "pair ordering")
    _number(report.get("offline_weight_quantization_seconds"), "offline W", positive=True)
    packing = report.get("packing")
    if not isinstance(packing, dict):
        raise ValueError("missing QKV packing checks")
    for key, expected in dict(
        sampled_input_rows=4,
        sampled_weight_rows=1152,
        input_errors=0,
        weight_errors=0,
        padding_errors=0,
        nonfinite_scales=0,
    ).items():
        _same(packing.get(key), expected, "packing." + key)
    for key in ("zero_output_errors", "epsilon_dominated_reference_errors"):
        _same(report.get(key), 0, key)
    paired = report.get("paired")
    if not isinstance(paired, list) or len(paired) != 3:
        raise ValueError("QKV needs all three paired streams")
    for stream, value in enumerate(paired):
        _same(value.get("stream"), stream, "paired stream")
        for key in ("native_nonfinite", "candidate_nonfinite"):
            _same(value.get(key), 0, key)
        mismatch = value.get("bit_mismatches")
        if type(mismatch) is not int or not 0 <= mismatch <= rows * 7168:
            raise ValueError("invalid QKV mismatch count")
        if stream == 2:
            _same(mismatch, 0, "V projection bit parity")
        _metric(value.get("metric"), rows * 7168, "pair")
    reference = report.get("reference")
    if not isinstance(reference, list) or len(reference) != 4:
        raise ValueError("QKV needs all four reference row classes")
    for value, row, kind in zip(
        reference,
        [0, 1, rows // 2, rows - 1],
        ["zeros", "epsilon_dominated", "normal", "outliers_tail"],
        strict=True,
    ):
        _same(value.get("row"), row, "reference row")
        _same(value.get("kind"), kind, "reference kind")
        _same(value.get("heads"), [0, 28, 55], "full reference heads")
        streams = value.get("streams")
        if not isinstance(streams, list) or len(streams) != 3:
            raise ValueError("missing reference streams")
        for stream, item in enumerate(streams):
            _same(item.get("stream"), stream, "reference stream")
            for key in (
                "native_implementation",
                "candidate_implementation",
                "native_total_error",
                "candidate_total_error",
                "quantization_error",
            ):
                _metric(item.get(key), 384, key, gate=key.endswith("implementation"))
    timings = report.get("timings")
    if not isinstance(timings, dict) or set(timings) != {
        "prepacked_qkv",
        "online_activation_plus_qkv",
    }:
        raise ValueError("missing QKV timing stage")
    for stage, dispatches in (("prepacked_qkv", 1), ("online_activation_plus_qkv", 2)):
        if set(timings[stage]) != {"native", "candidate"}:
            raise ValueError("missing paired timing path")
        for path, samples in timings[stage].items():
            if not isinstance(samples, list) or len(samples) != repeats:
                raise ValueError("missing raw QKV timings")
            for sample in samples:
                _same(sample.get("direct_dispatches"), dispatches, stage + ".dispatches")
                for key in (
                    "wall_seconds",
                    "cpu_encode_seconds",
                    "submit_wait_seconds",
                    "supplemental_gpu_seconds",
                ):
                    _number(
                        sample.get(key),
                        stage + "." + path + "." + key,
                        positive=key == "wall_seconds",
                    )
                if not math.isclose(
                    sample["wall_seconds"],
                    sample["cpu_encode_seconds"] + sample["submit_wait_seconds"],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError("QKV timing encode/wait boundaries do not sum to wall")
    _same(report.get("quality_decision"), "not_evaluated", "quality scope")
    if not isinstance(report.get("timing_scope"), str) or not report["timing_scope"]:
        raise ValueError("missing QKV timing scope")
    return "completed"

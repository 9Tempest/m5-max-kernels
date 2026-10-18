import sys

import numpy as np
import pytest

pytestmark = pytest.mark.metal
if sys.platform != "darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)
mx = pytest.importorskip("mlx.core")

from m5kernels.metal import gemm  # noqa: E402
from m5kernels.norm_hadamard import gated_residual_add  # noqa: E402
from m5kernels.numerics import metrics  # noqa: E402


def _cpu(x):
    return np.asarray(x.astype(mx.float32))


def _round_bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def _operands(mode, shape, distribution):
    rng = np.random.default_rng(229)
    m, n, k = shape
    if mode == "int8":
        limit = 127 if distribution == "outliers" else 17
        a = mx.array(rng.integers(-limit, limit + 1, size=(m, k), dtype=np.int8))
        b = mx.array(rng.integers(-limit, limit + 1, size=(n, k), dtype=np.int8))
        sa = mx.array(rng.uniform(0.01, 0.07, size=m).astype(np.float32))
        sb = mx.array(rng.uniform(0.01, 0.09, size=n).astype(np.float32))
        kwargs = dict(a_scales=sa, b_scales=sb)
    else:
        a = mx.array(rng.standard_normal((m, k), dtype=np.float32)).astype(mx.bfloat16)
        b = mx.array(rng.standard_normal((n, k), dtype=np.float32)).astype(mx.bfloat16)
        if distribution == "outliers":
            a = (a * mx.where(mx.arange(k) % 31 == 0, 20, 1)).astype(mx.bfloat16)
            b = (b * mx.where(mx.arange(k) % 47 == 0, 20, 1)).astype(mx.bfloat16)
        kwargs = {}
    if distribution == "zeros":
        a, b = mx.zeros_like(a), mx.zeros_like(b)
    residual = mx.array(0.2 * rng.standard_normal((m, n), dtype=np.float32)).astype(mx.bfloat16)
    gate = mx.array(0.3 * rng.standard_normal(n, dtype=np.float32)).astype(mx.bfloat16)
    return a, b, residual, gate, kwargs


def _check(mode, a, b, residual, gate, kwargs, gate_row_map=None):
    options = dict(tile=(32, 64, 4, 8), output_dtype=mx.bfloat16, **kwargs)
    branch = gemm(a, b, mode, **options)
    expected_staged = gated_residual_add(residual, branch, gate, gate_row_map=gate_row_map)
    actual = gemm(a, b, mode, **options, residual=residual, gate=gate, gate_row_map=gate_row_map)
    actual_cpu, staged_cpu = _cpu(actual), _cpu(expected_staged)
    assert actual.dtype == mx.bfloat16 and actual.shape == residual.shape
    assert np.isfinite(actual_cpu).all()
    if mode == "int8":
        # Integer accumulation and the two explicit BF16 boundaries should
        # reproduce the staged contract bit-for-bit, including cancellation.
        np.testing.assert_array_equal(actual_cpu, staged_cpu)
    else:
        # Floating TensorOps can change accumulator rounding after codegen;
        # do not disguise such variation as a change to the branch boundary.
        assert metrics(actual_cpu, staged_cpu)["relative_l2"] <= 0.008

    a_cpu, b_cpu = _cpu(a).astype(np.float64), _cpu(b).astype(np.float64)
    ref_branch = a_cpu @ b_cpu.T
    if mode == "int8":
        ref_branch *= (
            _cpu(kwargs["a_scales"]).astype(np.float64)[:, None]
            * _cpu(kwargs["b_scales"]).astype(np.float64)[None, :]
        )
    ref_branch = _round_bf16(ref_branch).astype(np.float64)
    selected_gate = _cpu(gate)
    if gate_row_map is not None:
        selected_gate = selected_gate[np.asarray(gate_row_map)]
    reference = _round_bf16(
        _cpu(residual).astype(np.float64) + ref_branch * selected_gate.astype(np.float64)
    )
    assert metrics(actual_cpu, reference)["relative_l2"] <= 0.008
    return actual_cpu


@pytest.mark.parametrize("mode", ["int8", "bf16"])
@pytest.mark.parametrize("shape", [(1, 17, 128), (7, 33, 256), (65, 67, 257)])
@pytest.mark.parametrize("distribution", ["normal", "outliers", "zeros"])
def test_gated_epilogue_full_reference_and_tails(mode, shape, distribution):
    a, b, residual, gate, kwargs = _operands(mode, shape, distribution)
    originals = [_cpu(x).copy() for x in (a, b, residual, gate)]
    result = _check(mode, a, b, residual, gate, kwargs)
    for before, after in zip(originals, (a, b, residual, gate), strict=True):
        np.testing.assert_array_equal(before, _cpu(after))
    if distribution == "zeros":
        np.testing.assert_array_equal(result, _cpu(residual))


@pytest.mark.parametrize("mode", ["int8", "bf16"])
@pytest.mark.parametrize("distribution", ["normal", "outliers"])
def test_gate_tables_repeated_and_boundary_rows(mode, distribution):
    a, b, residual, _, kwargs = _operands(mode, (7, 33, 257), distribution)
    rng = np.random.default_rng(331)
    gate = mx.array(0.3 * rng.standard_normal((4, 33), dtype=np.float32)).astype(mx.bfloat16)
    row_map = mx.array([3, 1, 0, 2, 3, 1, 0], dtype=mx.int32)
    table_before, map_before = _cpu(gate).copy(), np.asarray(row_map).copy()
    _check(mode, a, b, residual, gate, kwargs, row_map)
    np.testing.assert_array_equal(_cpu(gate), table_before)
    np.testing.assert_array_equal(np.asarray(row_map), map_before)


def _exact_branch_case(mode, branch_value, residual_value, gate_value):
    if mode == "int8":
        a_np, b_np = np.zeros((3, 128), np.int8), np.zeros((17, 128), np.int8)
        a_np[:, 0] = b_np[:, 0] = 1
        a, b = mx.array(a_np), mx.array(b_np)
        kwargs = dict(a_scales=mx.full((3,), branch_value), b_scales=mx.ones(17))
    else:
        a_np, b_np = np.zeros((3, 128), np.float32), np.zeros((17, 128), np.float32)
        a_np[:, 0], a_np[:, 1] = 1, branch_value - 1
        b_np[:, :2] = 1
        a, b = mx.array(a_np).astype(mx.bfloat16), mx.array(b_np).astype(mx.bfloat16)
        kwargs = {}
    residual = mx.full((3, 17), residual_value, dtype=mx.bfloat16)
    gate = mx.full((17,), gate_value, dtype=mx.bfloat16)
    output = gemm(a, b, mode, **kwargs, output_dtype=mx.bfloat16, residual=residual, gate=gate)
    staged_branch = gemm(a, b, mode, **kwargs, output_dtype=mx.bfloat16)
    staged = gated_residual_add(residual, staged_branch, gate)
    np.testing.assert_array_equal(_cpu(output), _cpu(staged))
    return _cpu(output)


@pytest.mark.parametrize("mode", ["int8", "bf16"])
def test_branch_rounding_boundary_is_preserved(mode):
    # GEMM produces a BF16 midpoint. Native branch rounding gives1.0 before
    # gate+residual. Removing that boundary would leave a nonzero2^-8 result.
    actual = _exact_branch_case(mode, 1.00390625, -1.0, 1.0)
    np.testing.assert_array_equal(actual, 0.0)


@pytest.mark.parametrize("mode", ["int8", "bf16"])
def test_gate_product_is_not_rounded_before_residual_add(mode):
    # Branch and gate are exactly representable BF16 values. Keeping their
    # product in FP32 leaves2^-14; a premature BF16 product round gives0.
    actual = _exact_branch_case(mode, 1.0078125, -1.015625, 1.0078125)
    np.testing.assert_array_equal(actual, np.float32(2**-14))


@pytest.mark.parametrize("mode", ["int8", "bf16"])
@pytest.mark.parametrize("gate_value,residual_value,expected", [(0.0, 7.5, 7.5), (-1.0, 1.0, 0.0)])
def test_zero_gate_and_exact_negative_gate_cancellation(mode, gate_value, residual_value, expected):
    actual = _exact_branch_case(mode, 1.0, residual_value, gate_value)
    np.testing.assert_array_equal(actual, expected)


def test_strided_residual_and_gate_table():
    a, b, _, _, kwargs = _operands("int8", (7, 33, 256), "normal")
    rng = np.random.default_rng(59)
    residual = mx.array(rng.standard_normal((33, 7), dtype=np.float32)).astype(mx.bfloat16).T
    gate = mx.array(rng.standard_normal((33, 3), dtype=np.float32)).astype(mx.bfloat16).T
    row_map = mx.array([2, 0, 1, 0, 2, 1, 0], dtype=mx.int32)
    _check("int8", a, b, residual, gate, kwargs, row_map)


def test_gated_epilogue_contract_rejections():
    a, b, residual, gate, scales = _operands("int8", (3, 17, 128), "normal")
    base = dict(output_dtype=mx.bfloat16, **scales)
    with pytest.raises(ValueError, match="both residual and gate"):
        gemm(a, b, "int8", **base, residual=residual)
    with pytest.raises(ValueError, match="both residual and gate"):
        gemm(a, b, "int8", **base, gate_row_map=mx.zeros(3, dtype=mx.int32))
    with pytest.raises(NotImplementedError, match="explicit BF16 output"):
        gemm(a, b, "int8", **scales, residual=residual, gate=gate)
    with pytest.raises(NotImplementedError, match="explicit BF16 output"):
        gemm(a, b, "int8", **scales, output_dtype=mx.float32, residual=residual, gate=gate)
    with pytest.raises(NotImplementedError, match="INT8 and BF16"):
        gemm(
            a.astype(mx.float16), b.astype(mx.float16), "fp16", **base, residual=residual, gate=gate
        )
    with pytest.raises(NotImplementedError, match="combining GEMM bias"):
        gemm(a, b, "int8", **base, residual=residual, gate=gate, bias=mx.zeros(17))
    with pytest.raises(ValueError, match="residual must"):
        gemm(a, b, "int8", **base, residual=residual.astype(mx.float32), gate=gate)
    with pytest.raises(ValueError, match="residual must"):
        gemm(a, b, "int8", **base, residual=residual[:2], gate=gate)
    with pytest.raises(ValueError, match="gate must have BF16"):
        gemm(a, b, "int8", **base, residual=residual, gate=gate.astype(mx.float32))
    with pytest.raises(ValueError, match="gate must have shape"):
        gemm(a, b, "int8", **base, residual=residual, gate=mx.ones((2, 17), dtype=mx.bfloat16))
    with pytest.raises(ValueError, match="gate_row_map must"):
        gemm(
            a,
            b,
            "int8",
            **base,
            residual=residual,
            gate=mx.ones((2, 17), dtype=mx.bfloat16),
            gate_row_map=mx.zeros(3),
        )
    with pytest.raises(ValueError, match="operand dtypes"):
        gemm(a.astype(mx.int32), b, "int8", **base, residual=residual, gate=gate)
    with pytest.raises(ValueError, match="float32 scales"):
        gemm(
            a,
            b,
            "int8",
            output_dtype=mx.bfloat16,
            a_scales=mx.ones(2),
            b_scales=mx.ones(17),
            residual=residual,
            gate=gate,
        )

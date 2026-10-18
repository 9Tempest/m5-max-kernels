import sys

import numpy as np
import pytest

pytestmark = pytest.mark.metal
if sys.platform != "darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)
mx = pytest.importorskip("mlx.core")

from m5kernels.metal import gemm  # noqa: E402
from m5kernels.numerics import metrics  # noqa: E402


def _operands(shape, distribution="normal"):
    m, n, k = shape
    rng = np.random.default_rng(83)
    bound = 127 if distribution == "outliers" else 17
    a = rng.integers(-bound, bound + 1, size=(m, k), dtype=np.int8)
    b = rng.integers(-bound, bound + 1, size=(n, k), dtype=np.int8)
    if distribution == "zeros":
        a.fill(0)
        b.fill(0)
    sa = rng.uniform(0.01, 0.13, size=m).astype(np.float32)
    sb = rng.uniform(0.02, 0.17, size=n).astype(np.float32)
    return a, b, sa, sb


def _expected(a, b, sa, sb, bias):
    # Integer accumulation is exact and independent of TensorOps. Scale and
    # bias application are evaluated in float64 before comparison.
    dots = a.astype(np.int64) @ b.astype(np.int64).T
    return dots.astype(np.float64) * sa[:, None] * sb[None, :] + bias.astype(np.float64)


@pytest.mark.parametrize("shape", [(1, 17, 128), (7, 33, 256), (65, 67, 257)])
@pytest.mark.parametrize("distribution", ["normal", "outliers", "zeros"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_shared_bias_full_reference_and_tails(shape, distribution, dtype):
    a, b, sa, sb = _operands(shape, distribution)
    bias = np.random.default_rng(97).uniform(-0.7, 0.7, size=shape[1]).astype(np.float32)
    output = gemm(
        mx.array(a),
        mx.array(b),
        "int8",
        (32, 64, 4, 8),
        a_scales=mx.array(sa),
        b_scales=mx.array(sb),
        bias=mx.array(bias),
        output_dtype=dtype,
    )
    actual = np.asarray(output.astype(mx.float32))
    expected = _expected(a, b, sa, sb, bias)
    assert output.dtype == dtype and np.isfinite(actual).all()
    assert metrics(actual, expected)["relative_l2"] <= (0.008 if dtype == mx.bfloat16 else 1e-6)


@pytest.mark.parametrize("distribution", ["normal", "outliers", "zeros"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_bias_table_independent_row_selection(distribution, dtype):
    a, b, sa, sb = _operands((7, 33, 257), distribution)
    bias = np.random.default_rng(107).uniform(-1.5, 1.5, size=(4, 33)).astype(np.float32)
    row_map = np.array([3, 1, 0, 2, 3, 1, 0], dtype=np.int32)
    a_gpu, b_gpu, bias_gpu, map_gpu = mx.array(a), mx.array(b), mx.array(bias), mx.array(row_map)
    output = gemm(
        a_gpu,
        b_gpu,
        "int8",
        (32, 64, 4, 8),
        a_scales=mx.array(sa),
        b_scales=mx.array(sb),
        bias=bias_gpu,
        bias_row_map=map_gpu,
        output_dtype=dtype,
    )
    actual = np.asarray(output.astype(mx.float32))
    expected = _expected(a, b, sa, sb, bias[row_map])
    assert np.isfinite(actual).all()
    assert metrics(actual, expected)["relative_l2"] <= (0.008 if dtype == mx.bfloat16 else 1e-6)
    # Projection inputs, bias table and map are reusable without mutation.
    for before, after in [(a, a_gpu), (b, b_gpu), (bias, bias_gpu), (row_map, map_gpu)]:
        np.testing.assert_array_equal(np.asarray(after), before)


def test_bias_is_added_before_bf16_output_rounding():
    a, b = np.zeros((3, 128), np.int8), np.zeros((17, 128), np.int8)
    a[:, 0] = 1
    b[:, 0] = 1
    output = gemm(
        mx.array(a),
        mx.array(b),
        "int8",
        a_scales=mx.full((3,), 1.00390625),
        b_scales=mx.ones(17),
        bias=mx.full((17,), -1.0),
        output_dtype=mx.bfloat16,
    )
    # A premature BF16 cast of 1.00390625 would round to1 before adding-1,
    # producing0. The intended FP32 addition retains exactly2^-8 instead.
    np.testing.assert_array_equal(np.asarray(output.astype(mx.float32)), np.float32(2**-8))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_zero_bias_agrees_with_unchanged_default_path(dtype):
    a, b, sa, sb = _operands((7, 33, 256), "outliers")
    args = mx.array(a), mx.array(b), "int8"
    kwargs = dict(a_scales=mx.array(sa), b_scales=mx.array(sb), output_dtype=dtype)
    baseline = gemm(*args, **kwargs)
    biased = gemm(*args, **kwargs, bias=mx.zeros(33))
    np.testing.assert_array_equal(
        np.asarray(baseline.astype(mx.float32)), np.asarray(biased.astype(mx.float32))
    )


def test_bias_contract_rejections():
    a, b = mx.zeros((3, 128), dtype=mx.int8), mx.zeros((17, 128), dtype=mx.int8)
    kwargs = dict(a_scales=mx.ones(3), b_scales=mx.ones(17))
    with pytest.raises(ValueError, match="requires a bias table"):
        gemm(a, b, "int8", **kwargs, bias_row_map=mx.zeros(3, dtype=mx.int32))
    with pytest.raises(NotImplementedError, match="INT8 only"):
        gemm(a.astype(mx.bfloat16), b.astype(mx.bfloat16), "bf16", bias=mx.zeros(17))
    with pytest.raises(ValueError, match="float32 dtype"):
        gemm(a, b, "int8", **kwargs, bias=mx.zeros(17, dtype=mx.bfloat16))
    with pytest.raises(ValueError, match=r"shape \[N\]"):
        gemm(a, b, "int8", **kwargs, bias=mx.zeros((2, 17)))
    with pytest.raises(ValueError, match=r"shape \[N\]"):
        gemm(a, b, "int8", **kwargs, bias=mx.zeros(18))
    with pytest.raises(ValueError, match="mapped bias"):
        gemm(a, b, "int8", **kwargs, bias=mx.zeros(17), bias_row_map=mx.zeros(3, dtype=mx.int32))
    with pytest.raises(ValueError, match="int32 vector"):
        gemm(a, b, "int8", **kwargs, bias=mx.zeros((2, 17)), bias_row_map=mx.zeros(3))
    with pytest.raises(ValueError, match="int32 vector"):
        gemm(
            a,
            b,
            "int8",
            **kwargs,
            bias=mx.zeros((2, 17)),
            bias_row_map=mx.zeros(2, dtype=mx.int32),
        )

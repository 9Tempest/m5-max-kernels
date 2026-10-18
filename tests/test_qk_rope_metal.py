import sys

import numpy as np
import pytest

pytestmark = pytest.mark.metal
if sys.platform != "darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)
mx = pytest.importorskip("mlx.core")

from m5kernels.metal import gemm, quantize  # noqa: E402
from m5kernels.numerics import metrics  # noqa: E402
from m5kernels.qk_rope import projection_qk_rope, qk_norm_rope  # noqa: E402
from m5kernels.qk_rope_reference import (  # noqa: E402
    bf16_round_reference,
    qk_rope_reference,
)


def _np(a):
    return np.asarray(a.astype(mx.float32))


def _tables(rng, m, rope_half=48):
    gamma = mx.array(rng.uniform(-1.5, 1.5, 128).astype(np.float32)).astype(mx.bfloat16)
    angles = rng.uniform(-4, 4, (m, rope_half)).astype(np.float32)
    cos = mx.array(np.cos(angles)).astype(mx.bfloat16)
    sin = mx.array(np.sin(angles)).astype(mx.bfloat16)
    return gamma, cos, sin


def _compare(actual, expected):
    actual_np = _np(actual)
    assert actual.dtype == mx.bfloat16
    assert np.isfinite(actual_np).all()
    assert metrics(actual_np, expected)["relative_l2"] < 0.008


@pytest.mark.parametrize("mode", ["int8", "bf16"])
@pytest.mark.parametrize(
    "shape,tile",
    [
        ((1, 128, 33), (16, 128, 4, 1)),
        ((17, 384, 96), (32, 128, 4, 8)),
        ((65, 256, 257), (64, 128, 8, 1)),
        ((129, 128, 33), (128, 128, 4, 8)),
        ((17, 384, 96), (32, 64, 4, 8)),
        ((1, 128, 5376), (128, 64, 8, 8)),
        ((65, 256, 257), (64, 64, 4, 1)),
        ((7, 128, 16), (16, 128, 8, 1)),
    ],
)
def test_fused_projection_tiles_tails_and_reference(mode, shape, tile):
    rng = np.random.default_rng(117)
    m, n, k = shape
    a = mx.array(rng.standard_normal((m, k), dtype=np.float32)).astype(mx.bfloat16)
    b = mx.array(rng.standard_normal((n, k), dtype=np.float32)).astype(mx.bfloat16)
    gamma, cos, sin = _tables(rng, m)
    scales = {}
    if mode == "int8":
        a, sa = quantize(a, "int8", tag="qk_test_a")
        b, sb = quantize(b, "int8", tag="qk_test_b")
        scales = dict(a_scales=sa, b_scales=sb)
        # Integer dot is exact, then emulate the declared FP32 dequant order.
        dot = np.asarray(a).astype(np.int64) @ np.asarray(b).astype(np.int64).T
        projected = (dot.astype(np.float32) * np.asarray(sa)[:, None]) * np.asarray(sb)[None, :]
    else:
        projected = _np(a).astype(np.float64) @ _np(b).astype(np.float64).T
    reference = qk_rope_reference(
        projected.reshape(m, n // 128, 128), _np(gamma), _np(cos), _np(sin)
    )
    fused = projection_qk_rope(a, b, gamma, cos, sin, mode, tile, **scales)
    staged_projection = gemm(a, b, mode, tile, output_dtype=mx.bfloat16, **scales)
    staged = qk_norm_rope(staged_projection, gamma, cos, sin)
    assert fused.shape == staged.shape == (n // 128, m, 128)
    _compare(fused, reference["output"])
    _compare(staged, reference["output"])
    if mode == "int8":
        # Same exact integer dot, FP32 dequant order, and postprocessor code:
        # this also catches omission of the BF16 projection boundary.
        np.testing.assert_array_equal(_np(fused), _np(staged))
    else:
        _compare(fused, _np(staged))


@pytest.mark.parametrize("mode", ["int8", "bf16"])
@pytest.mark.parametrize("kind", ["zeros", "outliers"])
def test_projection_zeros_outliers_and_input_immutability(mode, kind):
    rng = np.random.default_rng(33)
    a_np = rng.standard_normal((7, 257), dtype=np.float32)
    b_np = rng.standard_normal((384, 257), dtype=np.float32)
    if kind == "zeros":
        a_np.fill(0)
        b_np.fill(0)
    else:
        a_np[:, ::31] *= 100
        b_np[:, ::17] *= 100
    a, b = mx.array(a_np).astype(mx.bfloat16), mx.array(b_np).astype(mx.bfloat16)
    gamma, cos, sin = _tables(rng, 7)
    scales = {}
    if mode == "int8":
        a, sa = quantize(a, "int8", tag="qk_outlier_a")
        b, sb = quantize(b, "int8", tag="qk_outlier_b")
        scales = dict(a_scales=sa, b_scales=sb)
        dot = np.asarray(a).astype(np.int64) @ np.asarray(b).astype(np.int64).T
        projected = (dot.astype(np.float32) * np.asarray(sa)[:, None]) * np.asarray(sb)[None, :]
    else:
        projected = _np(a).astype(np.float64) @ _np(b).astype(np.float64).T
    inputs = [a, b, gamma, cos, sin, *scales.values()]
    saved = [_np(value).copy() for value in inputs]
    y = projection_qk_rope(a, b, gamma, cos, sin, mode, (32, 64, 4, 8), **scales)
    reference = qk_rope_reference(projected.reshape(7, 3, 128), _np(gamma), _np(cos), _np(sin))
    _compare(y, reference["output"])
    if kind == "zeros":
        np.testing.assert_array_equal(_np(y), 0)
    for value, original in zip(inputs, saved, strict=True):
        np.testing.assert_array_equal(_np(value), original)


@pytest.mark.parametrize("threads", [128, 256])
@pytest.mark.parametrize("rope_half", [1, 48, 64])
def test_standalone_native_norm_rope_reference(threads, rope_half):
    rng = np.random.default_rng(713)
    p_np = rng.standard_normal((5, 3, 128), dtype=np.float32)
    p_np[0] = 0
    p_np[1, :, ::31] *= 100
    p = mx.array(p_np.reshape(5, 384)).astype(mx.bfloat16)
    gamma, cos, sin = _tables(rng, 5, rope_half)
    y = qk_norm_rope(p, gamma, cos, sin, threads=threads)
    ref = qk_rope_reference(_np(p).reshape(5, 3, 128), _np(gamma), _np(cos), _np(sin))
    _compare(y, ref["output"])
    np.testing.assert_array_equal(_np(y)[:, 0], 0)


def test_split_half_pairs_unrotated_tail_and_head_major_layout():
    p_np = np.ones((3, 2, 128), dtype=np.float32)
    p_np[:, :, 48:96] = -1
    p_np[:, 1] *= -1
    p = mx.array(p_np.reshape(3, 256)).astype(mx.bfloat16)
    gamma = mx.ones(128, dtype=mx.bfloat16)
    cos, sin = mx.zeros((3, 48), dtype=mx.bfloat16), mx.ones((3, 48), dtype=mx.bfloat16)
    y = qk_norm_rope(p, gamma, cos, sin)
    expected = p_np.copy()
    expected[:, :, :48] = -p_np[:, :, 48:96]
    expected[:, :, 48:96] = p_np[:, :, :48]
    np.testing.assert_array_equal(_np(y), expected.transpose(1, 0, 2))


def test_no_bf16_norm_intermediate():
    # RMS numerator is exactly 131, independent of reduction ordering.
    # A nearly cancelling pair distinguishes rounding after RoPE from an
    # additional BF16 cast before the rotation, away from final midpoints.
    values = np.ones((3, 2, 128), dtype=np.float32)
    values[:, :, 127] = 2
    gamma_np = np.ones(128, dtype=np.float32)
    gamma_np[0] = 1 + 1 / 128
    p = mx.array(values.reshape(3, 256)).astype(mx.bfloat16)
    gamma = mx.array(gamma_np).astype(mx.bfloat16)
    cos = mx.full((3, 48), 181 / 256, dtype=mx.bfloat16)
    sin = mx.full((3, 48), 181 / 256, dtype=mx.bfloat16)
    native = qk_rope_reference(values, _np(gamma), _np(cos), _np(sin))
    rounded = qk_rope_reference(values, _np(gamma), _np(cos), _np(sin), norm_bf16=True)
    actual = _np(qk_norm_rope(p, gamma, cos, sin))
    np.testing.assert_array_equal(actual[:, :, 0], native["output"][:, :, 0])
    assert not np.array_equal(actual[:, :, 0], rounded["output"][:, :, 0])
    # The final output itself still has the named BF16 boundary.
    np.testing.assert_array_equal(actual, bf16_round_reference(actual))


@pytest.mark.parametrize("tile", [(16, 64, 4, 3), (32, 128, 4, 1)])
def test_tiny_nonzero_projection_preserves_dequant_scales(tile):
    # RMS nearly cancels a uniform projection-scale mistake at ordinary
    # magnitudes. Here epsilon dominates, making that mistake observable.
    rng = np.random.default_rng(137)
    m, n, k = 3, 256, 257
    a_np = rng.integers(-7, 8, size=(m, k), dtype=np.int8)
    b_np = rng.integers(-7, 8, size=(n, k), dtype=np.int8)
    a, b = mx.array(a_np), mx.array(b_np)
    sa_np = np.array([1e-4, 1e-5, 5e-5], dtype=np.float32)
    sb_np = np.full(n, 1e-3, dtype=np.float32)
    sa, sb = mx.array(sa_np), mx.array(sb_np)
    gamma, cos, sin = _tables(rng, m)
    integer_dot = a_np.astype(np.int64) @ b_np.astype(np.int64).T
    projected = (integer_dot.astype(np.float32) * sa_np[:, None]) * sb_np[None, :]
    projected = projected.reshape(m, n // 128, 128)
    assert np.max(np.mean(projected.astype(np.float64) ** 2, axis=-1)) < 1e-5 * 0.01
    reference = qk_rope_reference(projected, _np(gamma), _np(cos), _np(sin))["output"]
    wrong_scale = qk_rope_reference(projected * 2, _np(gamma), _np(cos), _np(sin))["output"]
    assert np.max(np.abs(reference)) > 0
    assert metrics(wrong_scale, reference)["relative_l2"] > 0.5

    actual = projection_qk_rope(a, b, gamma, cos, sin, "int8", tile, a_scales=sa, b_scales=sb)
    _compare(actual, reference)
    assert metrics(_np(actual), wrong_scale)["relative_l2"] > 0.4


def test_explicit_metadata_rejections():
    a, b = mx.ones((3, 17), dtype=mx.int8), mx.ones((256, 17), dtype=mx.int8)
    gamma = mx.ones(128, dtype=mx.bfloat16)
    cos, sin = mx.ones((3, 48), dtype=mx.bfloat16), mx.zeros((3, 48), dtype=mx.bfloat16)
    scales = dict(a_scales=mx.ones(3), b_scales=mx.ones(256))
    for epsilon in (0, np.nan, np.inf, 1e-100, 1e100):
        with pytest.raises(ValueError, match="epsilon"):
            projection_qk_rope(a, b, gamma, cos, sin, epsilon=epsilon, **scales)
    with pytest.raises(ValueError, match="scales"):
        projection_qk_rope(a, b, gamma, cos, sin)
    with pytest.raises(ValueError, match="whole 128-wide"):
        projection_qk_rope(a, b[:129], gamma, cos, sin, a_scales=mx.ones(3), b_scales=mx.ones(129))
    with pytest.raises(ValueError, match="gamma"):
        projection_qk_rope(a, b, gamma.astype(mx.float32), cos, sin, **scales)
    with pytest.raises(ValueError, match="matching bfloat16"):
        projection_qk_rope(a, b, gamma, cos.astype(mx.float32), sin, **scales)
    with pytest.raises(NotImplementedError, match="supported tiles"):
        projection_qk_rope(a, b, gamma, cos, sin, tile=(32, 32, 4, 1), **scales)
    with pytest.raises(NotImplementedError, match="INT8 or BF16"):
        projection_qk_rope(a, b, gamma, cos, sin, mode="fp8", **scales)
    with pytest.raises(ValueError, match="projected"):
        qk_norm_rope(mx.ones((3, 256)), gamma, cos, sin)

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.metal
if sys.platform != "darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)
mx = pytest.importorskip("mlx.core")

from m5kernels.hadamard_reference import (  # noqa: E402
    block_hadamard_reference,
    hadamard_signs,
)
from m5kernels.norm_hadamard import fused_norm_hadamard_quantize, unfused_norm  # noqa: E402
from m5kernels.norm_hadamard_reference import norm_modulate_reference  # noqa: E402
from m5kernels.numerics import quantize_reference  # noqa: E402


def rounded_reference(y, dtype):
    """CPU round-to-nearest-even at the declared post-modulation boundary."""
    y = np.asarray(y, dtype=np.float32)
    if dtype == mx.float16:
        return y.astype(np.float16).astype(np.float32)
    if dtype == mx.bfloat16:
        bits = y.view(np.uint32)
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        return rounded.view(np.float32)
    assert dtype == mx.float32
    return y


def check_reference(x, gamma, scale, shift, signs, block_size, **kwargs):
    q, s = fused_norm_hadamard_quantize(
        x, gamma, scale, shift, mx.array(signs), block_size, **kwargs
    )
    actual_q, actual_s = np.asarray(q), np.asarray(s)
    x_cpu, g_cpu = np.asarray(x.astype(mx.float32)), np.asarray(gamma.astype(mx.float32))
    s_cpu, b_cpu = np.asarray(scale.astype(mx.float32)), np.asarray(shift.astype(mx.float32))
    row_map = kwargs.get("row_map")
    if row_map is None:
        modulated = norm_modulate_reference(
            x_cpu,
            g_cpu,
            s_cpu,
            b_cpu,
            norm=kwargs.get("norm", "rms"),
            eps=kwargs.get("epsilon", 1e-5),
        )
    else:
        indices = np.asarray(row_map)
        modulated = np.concatenate(
            [
                norm_modulate_reference(
                    x_cpu[row : row + 1],
                    g_cpu,
                    s_cpu[index],
                    b_cpu[index],
                    norm=kwargs.get("norm", "rms"),
                    eps=kwargs.get("epsilon", 1e-5),
                )
                for row, index in enumerate(indices)
            ]
        )
    output_dtype = x.dtype if kwargs.get("round_output", "input") == "input" else mx.float32
    modulated = rounded_reference(modulated, output_dtype)
    rotated = block_hadamard_reference(modulated, block_size, signs)
    ref_q, ref_s = quantize_reference(rotated, "int8")
    assert actual_q.dtype == np.int8
    assert actual_s.dtype == np.float32
    assert np.isfinite(actual_s).all() and np.all(actual_s > 0)
    assert np.max(np.abs(actual_q.astype(np.int16))) <= 127
    # Near rint thresholds, FP32 summation/rsqrt differences can change one
    # code. This does not require bit equality with the float64 reference.
    assert np.max(np.abs(actual_q.astype(np.int16) - ref_q.astype(np.int16))) <= 1
    np.testing.assert_allclose(actual_s, ref_s, rtol=3e-5, atol=1e-7)
    decoded = actual_q.astype(np.float64) * actual_s[:, None]
    reference_decoded = ref_q.astype(np.float64) * ref_s[:, None]
    assert np.isfinite(decoded).all()
    error_bound = np.maximum(actual_s, ref_s) + 127 * np.abs(actual_s - ref_s)
    assert np.all(np.abs(decoded - reference_decoded) <= error_bound[:, None] + 1e-7)
    return actual_q, actual_s


@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("passes", ["one_read", "two_read"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("kind", ["normal", "zeros", "outliers"])
def test_h3_width_full_output(norm, passes, dtype, kind):
    rng = np.random.default_rng(621)
    x = rng.standard_normal((7, 5376), dtype=np.float32)
    if kind == "zeros":
        x.fill(0)
    elif kind == "outliers":
        x[:, ::137] *= 100
    gamma = rng.uniform(0.5, 1.5, size=5376).astype(np.float32)
    scale = rng.uniform(-0.3, 0.3, size=5376).astype(np.float32)
    # Nonzero shift ensures zero X is transformed rather than special-cased.
    shift = rng.uniform(-0.1, 0.1, size=5376).astype(np.float32)
    check_reference(
        mx.array(x).astype(dtype),
        mx.array(gamma),
        mx.array(scale),
        mx.array(shift),
        hadamard_signs(5376, 29),
        256,
        norm=norm,
        passes=passes,
    )


@pytest.mark.parametrize("block_size", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("norm", ["rms", "layer"])
def test_blocks_and_inactive_simd_groups(block_size, norm):
    rng = np.random.default_rng(31)
    width = block_size * 3  # Three blocks with four groups leaves one inactive.
    x = mx.array(rng.standard_normal((3, width), dtype=np.float32))
    check_reference(
        x,
        mx.ones(width),
        mx.full((width,), 0.125),
        mx.full((width,), 0.0625),
        hadamard_signs(width, 13),
        block_size,
        norm=norm,
        threads=128,
    )


@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("passes", ["one_read", "two_read"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_fp32_post_modulation_boundary(norm, passes, dtype):
    rng = np.random.default_rng(51)
    x = mx.array(rng.standard_normal((3, 768), dtype=np.float32)).astype(dtype)
    check_reference(
        x,
        mx.full((768,), 1.1),
        mx.full((768,), -0.13),
        mx.full((768,), 0.037),
        hadamard_signs(768, 41),
        256,
        norm=norm,
        passes=passes,
        round_output="fp32",
        epsilon=1e-4,
    )


@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("round_output", ["input", "fp32"])
def test_unfused_norm_reference(norm, dtype, round_output):
    rng = np.random.default_rng(79)
    x = mx.array(rng.standard_normal((3, 5376), dtype=np.float32)).astype(dtype)
    gamma = mx.array(rng.uniform(0.5, 1.5, size=5376).astype(np.float32))
    scale = mx.array(rng.uniform(-0.3, 0.3, size=5376).astype(np.float32))
    shift = mx.array(rng.uniform(-0.1, 0.1, size=5376).astype(np.float32))
    actual = unfused_norm(x, gamma, scale, shift, norm=norm, round_output=round_output)
    ref = norm_modulate_reference(
        np.asarray(x.astype(mx.float32)),
        np.asarray(gamma),
        np.asarray(scale),
        np.asarray(shift),
        norm=norm,
    )
    output_dtype = dtype if round_output == "input" else mx.float32
    ref = rounded_reference(ref, output_dtype)
    assert actual.dtype == output_dtype
    output = np.asarray(actual.astype(mx.float32))
    assert np.isfinite(output).all()
    # BF16/FP16 midpoint crossings permit one output ULP; FP32 retains a
    # strict floating implementation tolerance, with no quantization budget.
    tol = {mx.float32: 2e-6, mx.float16: 1e-3, mx.bfloat16: 8e-3}[output_dtype]
    np.testing.assert_allclose(output, ref, rtol=tol, atol=2e-7)


@pytest.mark.parametrize("round_output", ["input", "fp32"])
def test_explicit_bf16_rounding_ties(round_output):
    width = 256
    # Halfway BF16 cases distinguish a real post-modulation cast from a
    # compiler-removed cast; gamma=0 isolates this boundary from normalization.
    shift = np.resize(np.array([1.00390625, 1.01171875, -1.00390625, -1.01171875]), width)
    x = mx.zeros((3, width), dtype=mx.bfloat16)
    gamma, scale, shift_gpu = mx.zeros(width), mx.zeros(width), mx.array(shift.astype(np.float32))
    actual = unfused_norm(x, gamma, scale, shift_gpu, round_output=round_output)
    dtype = mx.bfloat16 if round_output == "input" else mx.float32
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.broadcast_to(rounded_reference(shift, dtype), x.shape),
    )
    check_reference(
        x, gamma, scale, shift_gpu, hadamard_signs(width, 3), width, round_output=round_output
    )


@pytest.mark.parametrize("passes", ["one_read", "two_read"])
def test_layernorm_constant_rows_and_zero_quantization(passes):
    x = mx.full((7, 5376), 17.0)
    q, s = check_reference(
        x,
        mx.ones(5376),
        mx.full((5376,), 0.5),
        mx.zeros(5376),
        hadamard_signs(5376, 43),
        256,
        norm="layer",
        passes=passes,
    )
    np.testing.assert_array_equal(q, 0)
    np.testing.assert_array_equal(s, 1.0)


@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("passes", ["one_read", "two_read"])
def test_modality_row_map(norm, passes):
    rng = np.random.default_rng(61)
    x = mx.array(rng.standard_normal((7, 768), dtype=np.float32)).astype(mx.bfloat16)
    gamma = mx.array(rng.uniform(0.5, 1.5, size=768).astype(np.float32))
    scale = mx.array(rng.uniform(-0.3, 0.3, size=(3, 768)).astype(np.float32))
    shift = mx.array(rng.uniform(-0.1, 0.1, size=(3, 768)).astype(np.float32))
    row_map = mx.array([2, 0, 1, 0, 2, 2, 1], dtype=mx.int32)
    check_reference(
        x,
        gamma,
        scale,
        shift,
        hadamard_signs(768, 11),
        256,
        norm=norm,
        passes=passes,
        row_map=row_map,
    )
    mapped = unfused_norm(x, gamma, scale, shift, norm=norm, row_map=row_map)
    for row, index in enumerate(np.asarray(row_map)):
        direct = unfused_norm(x[row : row + 1], gamma, scale[index], shift[index], norm=norm)
        np.testing.assert_array_equal(
            np.asarray(mapped[row].astype(mx.float32)),
            np.asarray(direct[0].astype(mx.float32)),
        )


def test_contract_rejections():
    x = mx.zeros((3, 768))
    gamma, scale, shift = mx.ones(768), mx.zeros(768), mx.zeros(768)
    signs = mx.ones(768, dtype=mx.int8)
    args = x, gamma, scale, shift, signs
    for kwargs, error, message in [
        ({"block_size": 33}, ValueError, "power of two"),
        ({"block_size": 2048}, NotImplementedError, "supported block sizes"),
        ({"block_size": 512}, NotImplementedError, "divisible"),
        ({"norm": "unknown"}, ValueError, "norm must"),
        ({"epsilon": 0}, ValueError, "positive finite"),
        ({"round_output": "bf16"}, ValueError, "round_output"),
        ({"threads": 32}, NotImplementedError, "threadgroup sizes"),
        ({"passes": "unknown"}, ValueError, "passes"),
        ({"tag": "bad-name"}, ValueError, "identifier"),
        ({"row_map": mx.zeros(3, dtype=mx.int32)}, ValueError, "mapped scale"),
    ]:
        with pytest.raises(error, match=message):
            fused_norm_hadamard_quantize(*args, **kwargs)
    with pytest.raises(ValueError, match="gamma"):
        fused_norm_hadamard_quantize(x, mx.ones(767), scale, shift, signs)
    with pytest.raises(ValueError, match="signs"):
        fused_norm_hadamard_quantize(x, gamma, scale, shift, mx.ones(768))

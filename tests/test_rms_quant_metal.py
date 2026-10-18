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
from m5kernels.rms_quant import rms_quantize  # noqa: E402


def _check(x, gamma, signs, block_size, *, epsilon=1e-5, **kwargs):
    q, s = rms_quantize(x, mx.array(gamma), mx.array(signs), block_size, epsilon=epsilon, **kwargs)
    q_cpu, s_cpu = np.asarray(q), np.asarray(s)
    x_cpu = np.asarray(x.astype(mx.float32)).astype(np.float64)
    inv_rms = 1 / np.sqrt(np.mean(x_cpu * x_cpu, axis=1, keepdims=True) + epsilon)
    # Independent float64 expression: normalize first, multiply the actual
    # FP32 gamma, then apply the independently implemented normalized H.
    normalized = (x_cpu * inv_rms) * gamma.astype(np.float64)
    rotated = block_hadamard_reference(normalized, block_size, signs)
    peak = np.max(np.abs(rotated), axis=1)
    ref_s = np.where(peak == 0, 1.0, peak / 127.0)
    ref_q = np.clip(np.rint(rotated / ref_s[:, None]), -127, 127).astype(np.int8)
    assert q_cpu.dtype == np.int8 and s_cpu.dtype == np.float32
    assert np.isfinite(s_cpu).all() and np.all(s_cpu > 0)
    assert np.max(np.abs(q_cpu.astype(np.int16))) <= 127
    # Normalizing before versus after the transform changes FP32 operation
    # order. Codes may differ by one around a quantization midpoint.
    assert np.max(np.abs(q_cpu.astype(np.int16) - ref_q.astype(np.int16))) <= 1
    np.testing.assert_allclose(s_cpu, ref_s, rtol=3e-6, atol=1e-30)
    decoded = q_cpu.astype(np.float64) * s_cpu[:, None]
    expected_decoded = ref_q.astype(np.float64) * ref_s[:, None]
    assert np.isfinite(decoded).all()
    error_bound = np.maximum(s_cpu, ref_s) + 127 * np.abs(s_cpu - ref_s)
    assert np.all(np.abs(decoded - expected_decoded) <= error_bound[:, None] * (1 + 1e-12))
    # Also check absolute quantization error directly against the unquantized
    # float64 signal, separately from decoded implementation error above.
    quant_error_bound = (0.5 * ref_s + 127 * np.abs(s_cpu - ref_s))[:, None]
    assert np.all(np.abs(decoded - rotated) <= quant_error_bound + 2e-6 * peak[:, None])
    return q_cpu, s_cpu


@pytest.mark.parametrize("block_size", [1, 256])
@pytest.mark.parametrize("fold_scale", [False, True])
@pytest.mark.parametrize("threads", [128, 256, 512])
@pytest.mark.parametrize("kind", ["normal", "zeros", "outliers", "weighted_peak", "epsilon"])
def test_reference_with_h3_width_and_plain_rms_tails(block_size, fold_scale, threads, kind):
    rng = np.random.default_rng(2021)
    width = 5376 if block_size > 1 else 769
    x = rng.standard_normal((7, width), dtype=np.float32)
    gamma = rng.uniform(-1.7, 1.7, size=width).astype(np.float32)
    if kind == "zeros":
        x.fill(0)
    elif kind == "outliers":
        x[:, ::137] *= 100
    elif kind == "weighted_peak":
        x[:, 0] = 100
        gamma[0] = 1e-3
        gamma[1::31] *= 100
        # This catches a fused reduction that mistakenly uses max(abs(X))
        # rather than the peak AFTER signed gamma and optional Hadamard.
        weighted = block_hadamard_reference(x.astype(np.float64) * gamma, block_size)
        assert np.all(np.max(np.abs(weighted), axis=1) != np.max(np.abs(x), axis=1))
    elif kind == "epsilon":
        x *= 1e-9
    q, s = _check(
        mx.array(x),
        gamma,
        hadamard_signs(width, 61),
        block_size,
        fold_scale=fold_scale,
        threads=threads,
    )
    if kind == "zeros":
        np.testing.assert_array_equal(q, 0)
        np.testing.assert_array_equal(s, 1.0)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("fold_scale", [False, True])
def test_input_dtypes_and_inactive_hadamard_groups(dtype, fold_scale):
    rng = np.random.default_rng(63)
    # Three whole blocks with four SIMD groups leaves one inactive group.
    x = mx.array(rng.standard_normal((3, 768), dtype=np.float32)).astype(dtype)
    gamma = rng.uniform(-0.7, 1.3, size=768).astype(np.float32)
    _check(x, gamma, hadamard_signs(768, 71), 256, fold_scale=fold_scale, threads=128)


@pytest.mark.parametrize("block_size", [32, 64, 128, 512, 1024])
@pytest.mark.parametrize("fold_scale", [False, True])
def test_other_block_sizes(block_size, fold_scale):
    rng = np.random.default_rng(751)
    width = 3 * block_size
    x = mx.array(rng.standard_normal((3, width), dtype=np.float32))
    gamma = rng.uniform(-1, 1, size=width).astype(np.float32)
    _check(x, gamma, hadamard_signs(width, 31), block_size, fold_scale=fold_scale)


@pytest.mark.parametrize("width", [1, 17, 255, 257, 5377])
def test_plain_rms_formula(width):
    # Integer data with gamma/signs=1 provides a simple direct formula for
    # q; inverse RMS affects only S in the folded implementation.
    values = np.resize(np.arange(-127, 128, dtype=np.float32), (3, width))
    x = mx.array(values)
    q, s = _check(x, np.ones(width, np.float32), np.ones(width, np.int8), 1)
    peak = np.max(np.abs(values), axis=1)
    expected_q = np.rint(values / peak[:, None] * 127).astype(np.int8)
    expected_s = peak / np.sqrt(np.mean(values.astype(np.float64) ** 2, axis=1) + 1e-5) / 127
    np.testing.assert_array_equal(q, expected_q)
    np.testing.assert_allclose(s, expected_s, rtol=3e-6)


@pytest.mark.parametrize("threads", [128, 256])
def test_folded_plain_rms_rounds_exact_ties_to_even(threads):
    # Peak 127 makes the unscaled quantizer step exactly one. These cases
    # must verify ties directly instead of allowing the general +/-1 bound.
    values = np.array([[127, 2.5, 3.5, -2.5, -3.5, 0, -127, 1.5]], dtype=np.float32)
    q, s = rms_quantize(
        mx.array(values),
        mx.ones(8, dtype=mx.float32),
        mx.ones(8, dtype=mx.int8),
        block_size=1,
        threads=threads,
        fold_scale=True,
    )
    np.testing.assert_array_equal(np.asarray(q), [[127, 2, 4, -2, -4, 0, -127, 2]])
    expected_scale = 1 / np.sqrt(np.mean(values.astype(np.float64) ** 2, axis=1) + 1e-5)
    np.testing.assert_allclose(np.asarray(s), expected_scale, rtol=3e-6)


@pytest.mark.parametrize("fold_scale", [False, True])
def test_zero_gamma_and_signed_gamma(fold_scale):
    x = mx.full((3, 768), 2.0)
    q, s = _check(x, np.zeros(768, np.float32), hadamard_signs(768, 9), 256, fold_scale=fold_scale)
    np.testing.assert_array_equal(q, 0)
    np.testing.assert_array_equal(s, 1.0)
    gamma = np.resize(np.array([-1.0, 0.0, 2.0], dtype=np.float32), 768)
    _check(x, gamma, hadamard_signs(768, 9), 256, fold_scale=fold_scale)


@pytest.mark.parametrize("fold_scale", [False, True])
def test_no_bf16_normalization_boundary(fold_scale):
    x = mx.ones((3, 256), dtype=mx.bfloat16)
    gamma = np.ones(256, dtype=np.float32)
    gamma[0] = 1.003
    _, s = _check(x, gamma, np.ones(256, np.int8), 1, fold_scale=fold_scale)
    # A forbidden BF16 cast after norm*gamma would round the entire row to
    # 1.0, yielding 1/127. The explicit FP32 contract retains the 1.003 peak.
    assert np.all(np.abs(s - np.float32(1 / 127)) > 1e-5)


def test_no_shift_argument_and_explicit_rejections():
    x, gamma, signs = mx.zeros((3, 768)), mx.ones(768), mx.ones(768, dtype=mx.int8)
    with pytest.raises(TypeError, match="shift"):
        rms_quantize(x, gamma, signs, shift=mx.zeros(768))
    for kwargs, error, match in [
        ({"block_size": 3}, ValueError, "power of two"),
        ({"block_size": 16}, NotImplementedError, "supported blocks"),
        ({"block_size": 512}, NotImplementedError, "divisible"),
        ({"threads": 32}, NotImplementedError, "threadgroup sizes"),
        ({"fold_scale": 1}, ValueError, "boolean"),
        ({"epsilon": 0}, ValueError, "positive finite"),
        ({"epsilon": 1e-100}, ValueError, "finite and positive in float32"),
        ({"epsilon": 1e100}, ValueError, "finite and positive in float32"),
    ]:
        with pytest.raises(error, match=match):
            rms_quantize(x, gamma, signs, **kwargs)
    with pytest.raises(ValueError, match="float32 vector"):
        rms_quantize(x, gamma.astype(mx.bfloat16), signs)
    with pytest.raises(ValueError, match="int8 vector"):
        rms_quantize(x, gamma, mx.ones(768))

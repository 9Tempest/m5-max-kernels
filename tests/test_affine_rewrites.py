"""Algebraic rewrites are precision policies; preserve the counterexamples."""

import numpy as np

from m5kernels.rms_quant_gemm import _bf16_round
from m5kernels.numerics import quantize_reference, dequantize


def test_shift_can_cross_linear_projection_before_rounding():
    rng = np.random.default_rng(29)
    x, w = rng.normal(size=(7, 17)), rng.normal(size=(11, 17))
    gamma, scale, beta = rng.normal(size=(3, 17))
    r = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    c = gamma * (1 + scale)
    np.testing.assert_allclose(
        (r * x * c + beta) @ w.T, r * ((x * c) @ w.T) + beta @ w.T, rtol=1e-12, atol=1e-12
    )


def test_bf16_boundary_prevents_constant_shift_compensation():
    a = np.array([1, 1 + 1 / 128], np.float32)
    beta = 1 / 256
    correction = _bf16_round(a + beta) - a
    np.testing.assert_array_equal(correction, [0, 1 / 128])
    assert correction[0] != correction[1]


def test_folding_gamma_into_weights_can_destroy_int8_quality():
    x = np.array([[1, 1000]], np.float32)
    gamma = np.array([1000, 1], np.float32)
    w = np.ones((1, 2), np.float32)

    def qdq(z):
        return dequantize(*quantize_reference(z, "int8"), "int8")

    original_policy = qdq(x * gamma) @ qdq(w).T
    folded_policy = qdq(x) @ qdq(w * gamma).T
    np.testing.assert_array_equal(original_policy, [[2000]])
    np.testing.assert_array_equal(folded_policy, [[0]])


def test_shift_split_quantization_can_amplify_cancellation_error():
    # Both equations are identical before quantization, but the joint path
    # quantizes only a small post-cancellation signal.
    u = np.array([[1, 1000]], np.float32)
    beta = np.array([[-0.99, -1000]], np.float32)

    def qdq(z):
        return dequantize(*quantize_reference(z, "int8"), "int8")

    target = u + beta
    joint_error = np.linalg.norm(qdq(target) - target)
    split_error = np.linalg.norm(qdq(u) + beta - target)
    assert joint_error < 1e-6
    assert split_error > 0.9

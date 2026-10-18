import math

import numpy as np
import pytest

from m5kernels.hadamard_reference import hadamard_signs
from m5kernels.rms_quant_reference import rms_quantize_reference


def dense_hadamard(block_size):
    return np.array(
        [[(-1) ** (i & j).bit_count() for j in range(block_size)] for i in range(block_size)],
        dtype=np.float64,
    ) / math.sqrt(block_size)


@pytest.mark.parametrize("block_size", [1, 4, 128])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers", "tiny"])
@pytest.mark.parametrize("signed", [False, True])
def test_folded_encoding_matches_independent_materialized_reference(
    block_size, distribution, signed
):
    rng = np.random.default_rng(59)
    width = 3 * block_size
    x = rng.normal(size=(5, width))
    gamma = rng.normal(size=width)
    gamma[0] = 0
    if distribution == "zeros":
        x.fill(0)
    elif distribution == "outliers":
        x[0, 0] *= 1000
        x[-1, -1] *= 1000
        gamma[-1] *= 100
    elif distribution == "tiny":
        x *= 1e-12
    x = np.asfortranarray(x)
    signs = hadamard_signs(width, 61) if signed else None
    originals = [a.copy() for a in (x, gamma)]
    original_signs = signs.copy() if signed else None
    dense = np.kron(np.eye(3), dense_hadamard(block_size))
    normalized = np.empty(x.shape, dtype=np.float64)
    expected_inv = np.empty(x.shape[0], dtype=np.float64)
    for i, row in enumerate(x):
        expected_inv[i] = 1.0 / math.sqrt(math.fsum(float(v) ** 2 for v in row) / width + 1e-5)
        normalized[i] = [float(value) * expected_inv[i] * gamma[j] for j, value in enumerate(row)]
    expected = (normalized * (signs if signed else 1)) @ dense
    expected_scales = np.empty(5, dtype=np.float64)
    expected_q = np.empty(x.shape, dtype=np.int8)
    for i, row in enumerate(expected):
        maximum = max(abs(float(v)) for v in row)
        expected_scales[i] = maximum / 127 if maximum else 1
        expected_q[i] = [min(127, max(-127, round(float(v) / expected_scales[i]))) for v in row]
    actual = rms_quantize_reference(x, gamma, signs, block_size)
    np.testing.assert_allclose(actual["inverse_rms"], expected_inv, rtol=3e-14, atol=0)
    np.testing.assert_allclose(actual["normalized_rotated"], expected, rtol=3e-14, atol=1e-11)
    np.testing.assert_allclose(actual["scales"], expected_scales, rtol=3e-14, atol=0)
    np.testing.assert_array_equal(actual["quantized"], expected_q)
    np.testing.assert_allclose(
        actual["dequantized"], expected_q * expected_scales[:, None], rtol=3e-14, atol=1e-11
    )
    assert actual["quantized"].dtype == np.int8
    assert actual["scales"].dtype == np.float64
    for current, before in zip((x, gamma), originals, strict=True):
        np.testing.assert_array_equal(current, before)
    if signed:
        np.testing.assert_array_equal(signs, original_signs)


@pytest.mark.parametrize("block_size", [1, 8])
def test_epsilon_changes_scales_but_never_codes(block_size):
    rng = np.random.default_rng(67)
    x = rng.normal(size=(3, 24)) * 0.01
    gamma = rng.normal(size=24)
    small_eps = rms_quantize_reference(x, gamma, block_size=block_size, eps=1e-8)
    large_eps = rms_quantize_reference(x, gamma, block_size=block_size, eps=1.0)
    np.testing.assert_array_equal(small_eps["quantized"], large_eps["quantized"])
    np.testing.assert_array_equal(small_eps["unscaled_scales"], large_eps["unscaled_scales"])
    assert np.all(small_eps["scales"] > large_eps["scales"])
    np.testing.assert_allclose(
        small_eps["scales"] / large_eps["scales"],
        small_eps["inverse_rms"] / large_eps["inverse_rms"],
        rtol=3e-15,
    )


def test_rms_statistics_use_original_input_before_gamma():
    x = np.array([[1.0, 2.0, -4.0, 8.0]])
    gamma = np.array([-100.0, 0.0, 2.0, 0.01])
    actual = rms_quantize_reference(x, gamma, block_size=2)
    expected_inv = 1 / math.sqrt((1 + 4 + 16 + 64) / 4 + 1e-5)
    np.testing.assert_allclose(actual["inverse_rms"], [expected_inv], rtol=0, atol=0)
    weighted_inv = 1 / math.sqrt(float(np.mean((x * gamma) ** 2)) + 1e-5)
    assert not np.isclose(actual["inverse_rms"][0], weighted_inv)
    expected_peak = np.max(abs((x * gamma) @ np.kron(np.eye(2), dense_hadamard(2))))
    np.testing.assert_allclose(actual["scales"], [expected_peak / 127 * expected_inv], rtol=3e-15)


@pytest.mark.parametrize("zero_gamma", [False, True])
def test_zero_rotated_rows_have_unit_output_scale(zero_gamma):
    x = np.ones((3, 8)) if zero_gamma else np.zeros((3, 8))
    gamma = np.zeros(8) if zero_gamma else -np.ones(8)
    actual = rms_quantize_reference(x, gamma, block_size=4)
    for key in ("normalized_rotated", "unscaled_rotated", "quantized", "dequantized"):
        np.testing.assert_array_equal(actual[key], np.zeros((3, 8)))
    np.testing.assert_array_equal(actual["scales"], np.ones(3))


def test_quantizer_rounds_ties_to_even_without_rms_factor():
    x = np.array([[127.0, 2.5, 3.5, -2.5, -3.5, 0.0, -127.0, 1.5]])
    actual = rms_quantize_reference(x, np.ones(8))
    np.testing.assert_array_equal(actual["quantized"], [[127, 2, 4, -2, -4, 0, -127, 2]])
    np.testing.assert_array_equal(actual["unscaled_scales"], [1.0])
    np.testing.assert_array_equal(actual["scales"], actual["inverse_rms"])


@pytest.mark.parametrize("eps", [0, -1, np.nan, np.inf, True, 1j, [1e-5]])
def test_rejects_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        rms_quantize_reference(np.ones((3, 8)), np.ones(8), eps=eps)


@pytest.mark.parametrize("shape", [(8,), (0, 8), (3, 0), (3, 2, 4)])
def test_rejects_invalid_input_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        rms_quantize_reference(np.ones(shape), np.ones(8))


@pytest.mark.parametrize("shape", [(7,), (1, 8), ()])
def test_rejects_invalid_gamma_shape(shape):
    with pytest.raises(ValueError, match="gamma"):
        rms_quantize_reference(np.ones((3, 8)), np.ones(shape))


@pytest.mark.parametrize("operand", ["x", "gamma"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1j])
def test_rejects_nonfinite_or_complex_operands(operand, invalid):
    values = dict(x=np.ones((3, 8)), gamma=np.ones(8))
    values[operand] = values[operand].astype(complex if invalid == 1j else float)
    values[operand].flat[0] = invalid
    with pytest.raises(ValueError, match="finite|real"):
        rms_quantize_reference(**values)


@pytest.mark.parametrize("block_size", [3, 16, 0, True])
def test_rejects_invalid_or_nondivisible_hadamard_block(block_size):
    with pytest.raises(ValueError, match="block_size"):
        rms_quantize_reference(np.ones((3, 8)), np.ones(8), block_size=block_size)


@pytest.mark.parametrize("signs", [np.ones(7), np.zeros(8), np.full(8, np.nan)])
def test_rejects_invalid_signs(signs):
    with pytest.raises(ValueError, match="signs"):
        rms_quantize_reference(np.ones((3, 8)), np.ones(8), signs)


def test_rejects_nonfinite_rms_intermediates():
    with pytest.raises(ValueError, match="RMS"):
        rms_quantize_reference(np.full((3, 8), np.finfo(np.float64).max), np.ones(8))

import math

import numpy as np
import pytest

from m5kernels.hadamard_reference import hadamard_signs
from m5kernels.norm_hadamard_reference import (
    norm_hadamard_quantize_reference,
    norm_modulate_reference,
)


def dense_hadamard(block_size):
    return np.array(
        [[(-1) ** (i & j).bit_count() for j in range(block_size)] for i in range(block_size)],
        dtype=np.float64,
    ) / math.sqrt(block_size)


def scalar_modulation(x, gamma, scale, shift, norm, eps):
    result = np.empty(x.shape, dtype=np.float64)
    for i, row in enumerate(x):
        mean = math.fsum(row) / len(row) if norm == "layer" else 0.0
        centered = [float(v) - mean for v in row]
        moment = math.fsum(v * v for v in centered) / len(row)
        denominator = math.sqrt(moment + eps)
        for j, value in enumerate(centered):
            result[i, j] = value / denominator * gamma[j] * (1.0 + scale[j]) + shift[j]
    return result


@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers", "constant"])
@pytest.mark.parametrize("block_size", [1, 8, 128])
def test_pipeline_against_scalar_norm_and_dense_hadamard(norm, distribution, block_size):
    rng = np.random.default_rng(41)
    width = 3 * block_size
    x = rng.normal(size=(5, width))
    if distribution == "zeros":
        x.fill(0)
    elif distribution == "outliers":
        x[:, 0] *= 1000
        x[-1, -1] *= 1000
    elif distribution == "constant":
        x[:] = rng.normal(size=(5, 1))
    x = np.asfortranarray(x)
    gamma = rng.normal(size=width)
    scale = rng.normal(size=width)
    shift = rng.normal(size=width)
    signs = hadamard_signs(width, 43)
    original = [v.copy() for v in (x, gamma, scale, shift, signs)]
    expected_modulation = scalar_modulation(x, gamma, scale, shift, norm, 1e-5)
    dense = np.kron(np.eye(3), dense_hadamard(block_size))
    expected_rotation = (expected_modulation * signs) @ dense
    expected_q = np.empty(x.shape, dtype=np.int8)
    expected_scales = np.empty(5, dtype=np.float64)
    for i, row in enumerate(expected_rotation):
        maximum = max(abs(float(value)) for value in row)
        expected_scales[i] = maximum / 127.0 if maximum else 1.0
        expected_q[i] = [max(-127, min(127, round(v / expected_scales[i]))) for v in row]
    result = norm_hadamard_quantize_reference(x, gamma, scale, shift, block_size, signs, norm=norm)
    np.testing.assert_allclose(result["modulated"], expected_modulation, rtol=3e-14, atol=1e-11)
    np.testing.assert_allclose(result["rotated"], expected_rotation, rtol=3e-14, atol=1e-11)
    np.testing.assert_allclose(result["scales"], expected_scales, rtol=3e-14, atol=1e-13)
    np.testing.assert_array_equal(result["quantized"], expected_q)
    np.testing.assert_allclose(
        result["dequantized"], expected_q * expected_scales[:, None], rtol=3e-14, atol=1e-11
    )
    assert result["quantized"].dtype == np.int8
    assert result["rotated"].dtype == np.float64
    assert result["scales"].dtype == np.float64
    for actual, before in zip((x, gamma, scale, shift, signs), original, strict=True):
        np.testing.assert_array_equal(actual, before)


@pytest.mark.parametrize("norm", ["rms", "layer"])
def test_zero_rows_use_unit_quantization_scale(norm):
    result = norm_hadamard_quantize_reference(
        np.zeros((3, 8)), np.ones(8), np.ones(8), np.zeros(8), 4, norm=norm
    )
    for name in ("modulated", "rotated", "quantized", "dequantized"):
        np.testing.assert_array_equal(result[name], np.zeros((3, 8)))
    np.testing.assert_array_equal(result["scales"], np.ones(3))


def test_constant_layernorm_row_equals_modulation_shift():
    shift = np.array([1.0, -2.0, 4.0, 8.0])
    y = norm_modulate_reference(
        np.full((3, 4), 13.0), np.arange(4), np.arange(4), shift, norm="layer"
    )
    np.testing.assert_array_equal(y, np.tile(shift, (3, 1)))


def test_zero_scale_multiplier_and_round_to_nearest_even():
    shift = np.array([127.0, 2.5, 3.5, -2.5, -3.5, 0.0, -127.0, 1.5])
    result = norm_hadamard_quantize_reference(
        np.arange(24).reshape(3, 8), np.ones(8), -np.ones(8), shift, 1
    )
    np.testing.assert_array_equal(result["modulated"], np.tile(shift, (3, 1)))
    np.testing.assert_array_equal(result["scales"], np.ones(3))
    np.testing.assert_array_equal(
        result["quantized"], np.tile([127, 2, 4, -2, -4, 0, -127, 2], (3, 1))
    )


def test_epsilon_is_inside_root_and_no_intermediate_rounding():
    x = np.array([[0.000137, 0.000211]])
    eps = 0.125
    actual = norm_modulate_reference(x, np.ones(2), np.zeros(2), np.zeros(2), eps=eps)
    expected = x / math.sqrt(math.fsum(v * v for v in x[0]) / 2 + eps)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, actual.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize("norm", ["batch", "RMS", None])
def test_rejects_unknown_normalization(norm):
    with pytest.raises(ValueError, match="norm"):
        norm_modulate_reference(np.ones((3, 8)), np.ones(8), np.zeros(8), np.zeros(8), norm=norm)


@pytest.mark.parametrize("eps", [0, -1, np.nan, np.inf, 1j, True, [1e-5]])
def test_rejects_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        norm_modulate_reference(np.ones((3, 8)), np.ones(8), np.zeros(8), np.zeros(8), eps=eps)


@pytest.mark.parametrize("name", ["x", "gamma", "scale", "shift"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1j])
def test_rejects_nonfinite_and_complex_operands(name, invalid):
    values = dict(x=np.ones((3, 8)), gamma=np.ones(8), scale=np.zeros(8), shift=np.zeros(8))
    values[name] = values[name].astype(complex if invalid == 1j else float)
    values[name].flat[0] = invalid
    with pytest.raises(ValueError, match="real|finite"):
        norm_modulate_reference(**values)


@pytest.mark.parametrize("name", ["gamma", "scale", "shift"])
@pytest.mark.parametrize("shape", [(1, 8), (7,), ()])
def test_requires_exact_per_channel_vector_shapes(name, shape):
    values = dict(x=np.ones((3, 8)), gamma=np.ones(8), scale=np.zeros(8), shift=np.zeros(8))
    values[name] = np.zeros(shape)
    with pytest.raises(ValueError, match=name):
        norm_modulate_reference(**values)


@pytest.mark.parametrize("shape", [(8,), (0, 8), (3, 0), (3, 2, 4)])
def test_rejects_invalid_input_dimensions(shape):
    with pytest.raises(ValueError, match="shape"):
        norm_modulate_reference(np.ones(shape), np.ones(8), np.zeros(8), np.zeros(8))


@pytest.mark.parametrize("block_size", [3, 16])
def test_rejects_unsupported_hadamard_blocks(block_size):
    with pytest.raises(ValueError, match="power of two|divisible"):
        norm_hadamard_quantize_reference(
            np.ones((3, 8)), np.ones(8), np.zeros(8), np.zeros(8), block_size
        )


def test_rejects_nonfinite_normalization_intermediates():
    with pytest.raises(ValueError, match="normalization"):
        norm_modulate_reference(
            np.full((3, 8), np.finfo(np.float64).max), np.ones(8), np.zeros(8), np.zeros(8)
        )

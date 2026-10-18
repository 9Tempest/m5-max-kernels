import math

import numpy as np
import pytest

from m5kernels.grouped_qkv_reference import (
    bf16_decode_reference,
    bf16_encode_reference,
    grouped_qkv_reference,
    native_int8_quantize_reference,
)


def scalar_bf16(value):
    value = float(np.float32(value))
    if value == 0:
        return value
    _, exponent = math.frexp(abs(value))
    step = math.ldexp(1.0, max(exponent - 8, -133))
    return math.copysign(round(abs(value) / step) * step, value)


def fixture(seed=149):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 17))
    weight = rng.normal(size=(2, 3, 128, 17)) / math.sqrt(17)
    q_gamma, k_gamma = rng.normal(size=(2, 128))
    angles = rng.normal(size=(3, 48))
    return [a, weight, q_gamma, k_gamma, np.cos(angles), np.sin(angles)]


@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers", "tiny"])
def test_grouped_matches_independent_scalar_dots_rms_and_pair_rotation(distribution):
    values = fixture()
    if distribution == "zeros":
        values[0].fill(0)
    elif distribution == "outliers":
        values[0][0, -1] = 100
        values[1][-1, :, -1, 0] = -100
    elif distribution == "tiny":
        values[0] *= 1e-4
    values = [bf16_decode_reference(bf16_encode_reference(value)) for value in values]
    a, weight, q_gamma, k_gamma, cos, sin = values
    originals = [v.copy() for v in values]
    result = grouped_qkv_reference(*values, return_projection=True)
    expected = [np.empty((2, 3, 128)) for _ in range(3)]
    for head in range(2):
        for stream in range(3):
            for row in range(3):
                dot = [
                    math.fsum(
                        float(a[row, k]) * float(weight[head, stream, d, k]) for k in range(17)
                    )
                    for d in range(128)
                ]
                rounded = [scalar_bf16(v) for v in dot]
                if stream == 2:
                    expected[stream][head, row] = rounded
                    continue
                gamma = q_gamma if stream == 0 else k_gamma
                inverse = 1 / math.sqrt(math.fsum(v * v for v in rounded) / 128 + 1e-5)
                normalized = [v * inverse * gamma[d] for d, v in enumerate(rounded)]
                output = normalized.copy()
                for d in range(48):
                    pair = np.array([[cos[row, d], -sin[row, d]], [sin[row, d], cos[row, d]]])
                    output[d], output[d + 48] = pair @ [normalized[d], normalized[d + 48]]
                expected[stream][head, row] = [scalar_bf16(v) for v in output]
    for actual, wanted in zip(result["outputs"], expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
        assert actual.flags.c_contiguous and actual.dtype == np.float64
    assert result["projection_bf16"].shape == (3, 2, 3, 128)
    for current, original in zip(values, originals, strict=True):
        np.testing.assert_array_equal(current, original)


def test_grouped_stream_head_layout_and_value_bypass():
    a = np.array([[1.0], [2.0], [-3.0]])
    weight = np.empty((3, 3, 128, 1))
    for h in range(3):
        for stream in range(3):
            weight[h, stream, :, 0] = 1 + h * 24 + stream * 4 + np.arange(128) / 128
    gamma_q, gamma_k = np.zeros(128), np.ones(128)
    cos, sin = np.zeros((3, 48)), np.ones((3, 48))
    q, k, v = grouped_qkv_reference(a, weight, gamma_q, gamma_k, cos, sin)
    expected_v = np.array(
        [
            [[scalar_bf16(a[r, 0] * weight[h, 2, d, 0]) for d in range(128)] for r in range(3)]
            for h in range(3)
        ]
    )
    np.testing.assert_array_equal(q, 0)
    np.testing.assert_array_equal(v, expected_v)
    assert not np.allclose(k, v)
    assert not np.array_equal(v[0], v[1])
    assert np.max(np.abs(v)) > 100


def test_gamma_vectors_are_independent_and_values_are_not_normalized():
    values = fixture()
    before = grouped_qkv_reference(*values)
    values[2] *= -1
    q_changed = grouped_qkv_reference(*values)
    np.testing.assert_array_equal(q_changed[0], -before[0])
    np.testing.assert_array_equal(q_changed[1], before[1])
    np.testing.assert_array_equal(q_changed[2], before[2])
    values[3] *= 2
    k_changed = grouped_qkv_reference(*values)
    np.testing.assert_array_equal(k_changed[0], q_changed[0])
    np.testing.assert_array_equal(k_changed[1], q_changed[1] * 2)
    np.testing.assert_array_equal(k_changed[2], q_changed[2])


def test_projection_bf16_boundary_is_visible_in_value():
    a = np.ones((1, 1))
    weight = np.full((1, 3, 128, 1), 1.00390625)
    result = grouped_qkv_reference(
        a,
        weight,
        np.ones(128),
        np.ones(128),
        np.ones((1, 48)),
        np.zeros((1, 48)),
        return_projection=True,
    )
    np.testing.assert_array_equal(result["projection_float64"], 1.00390625)
    np.testing.assert_array_equal(result["projection_bf16"], 1.0)
    np.testing.assert_array_equal(result["outputs"][2], 1.0)


def test_decoded_quantized_operands_are_not_implicitly_rounded_to_bf16():
    a = np.full((1, 1), 1.001)
    weight = np.full((1, 3, 128, 1), 1.002)
    result = grouped_qkv_reference(
        a,
        weight,
        np.ones(128),
        np.ones(128),
        np.ones((1, 48)),
        np.zeros((1, 48)),
        return_projection=True,
    )
    np.testing.assert_array_equal(result["projection_float64"], 1.001 * 1.002)


@pytest.mark.parametrize("clip", [1.0, 0.75])
def test_native_quantizer_matches_scalar_fp32_policy_and_padding(clip):
    x = np.random.default_rng(151).normal(size=(5, 17))
    x[0] = 0
    x[1, -1] = 1000
    x[2] *= 1e-8
    x = np.asfortranarray(x)
    before = x.copy()
    actual = native_int8_quantize_reference(x, padded_rows=8, clip=clip)
    expected_codes = np.zeros((8, 17), dtype=np.int8)
    expected_scales = np.ones(8, dtype=np.float32)
    for row in range(5):
        values = [scalar_bf16(v) for v in x[row]]
        peak = np.float32(max(abs(v) for v in values)) * np.float32(clip)
        inverse = np.float32(np.float32(127) / peak) if peak > 0 else np.float32(127)
        expected_scales[row] = (
            np.float32(peak / np.float32(127)) if peak > 0 else np.float32(1 / 127)
        )
        expected_codes[row] = [
            min(127, max(-127, round(float(np.float32(v) * inverse)))) for v in values
        ]
    np.testing.assert_array_equal(actual["codes"], expected_codes)
    np.testing.assert_array_equal(actual["scales"], expected_scales)
    np.testing.assert_array_equal(
        actual["decoded"], expected_codes.astype(float) * expected_scales[:, None]
    )
    assert actual["scales"][0] == np.float32(1 / 127)
    np.testing.assert_array_equal(actual["scales"][5:], 1)
    assert actual["quantization_error"]["finite"]
    assert actual["metadata"]["column_padding"] == 0
    np.testing.assert_array_equal(x, before)


def test_native_quantizer_ties_even_and_symmetric_endpoints():
    # Maximum254 gives exactly inverse0.5: odd inputs create exact half ties.
    x = np.array([[254, -254, 1, 3, 5, -1, -3, -5]], dtype=float)
    result = native_int8_quantize_reference(x)
    np.testing.assert_array_equal(result["codes"], [[127, -127, 0, 2, 2, 0, -2, -2]])
    np.testing.assert_array_equal(result["scales"], [2])


def test_bf16_payload_rounding_and_decoding_include_signed_zero():
    values = np.array([0.0, -0.0, 1.00390625, 1.01171875, -1.00390625])
    bits = bf16_encode_reference(values)
    np.testing.assert_array_equal(
        bits, np.array([0, 0x8000, 0x3F80, 0x3F82, 0xBF80], dtype=np.uint16)
    )
    decoded = bf16_decode_reference(bits)
    np.testing.assert_array_equal(decoded, [scalar_bf16(v) for v in values])
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(values))
    exceptional = bf16_decode_reference(np.array([0x7F80, 0xFF80, 0x7FC1], dtype=np.uint16))
    assert np.isposinf(exceptional[0]) and np.isneginf(exceptional[1]) and np.isnan(exceptional[2])


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1j])
def test_nonfinite_or_complex_grouped_operand_rejected(index, invalid):
    values = fixture()
    values[index] = values[index].astype(complex if invalid == 1j else float)
    values[index].flat[-1] = invalid
    with pytest.raises(ValueError, match="finite|real"):
        grouped_qkv_reference(*values)


@pytest.mark.parametrize(
    "index,value,message",
    [
        (0, np.ones((0, 17)), "activation"),
        (0, np.ones(17), "activation"),
        (1, np.ones((2, 384, 17)), "weight"),
        (1, np.ones((2, 3, 128, 16)), "K"),
        (2, np.ones((2, 128)), "gamma"),
        (3, np.ones(127), "gamma"),
        (4, np.ones((3, 96)), "cos"),
        (5, np.ones((1, 48)), "cos"),
    ],
)
def test_grouped_invalid_shapes(index, value, message):
    values = fixture()
    values[index] = value
    with pytest.raises(ValueError, match=message):
        grouped_qkv_reference(*values)


@pytest.mark.parametrize(
    "option,value",
    [("padded_rows", 2), ("padded_rows", True), ("clip", 0), ("clip", np.nan), ("clip", 1e100)],
)
def test_native_quantization_invalid_options(option, value):
    with pytest.raises(ValueError, match=option.split("_")[0]):
        native_int8_quantize_reference(np.ones((3, 17)), **{option: value})


@pytest.mark.parametrize(
    "value", [np.ones(17), np.ones((0, 17)), np.full((3, 17), np.inf), np.full((3, 17), 1j)]
)
def test_native_quantization_invalid_input(value):
    with pytest.raises(ValueError):
        native_int8_quantize_reference(value)


def test_invalid_payload_and_overflow():
    with pytest.raises(ValueError, match="uint16"):
        bf16_decode_reference(np.array([0x3F80], dtype=np.uint32))
    with pytest.raises(ValueError, match="finite"):
        bf16_encode_reference([np.inf])
    with pytest.raises(ValueError, match="finite"):
        native_int8_quantize_reference(np.full((1, 17), 2.0**-130))

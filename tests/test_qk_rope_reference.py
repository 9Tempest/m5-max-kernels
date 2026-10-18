import math

import numpy as np
import pytest

from m5kernels.qk_rope_reference import bf16_round_reference, qk_rope_reference


def scalar_bf16(value):
    """Independent arithmetic nearest-even oracle; no integer bit manipulation."""
    value = float(np.float32(value))
    if value == 0:
        return value
    _, exponent = math.frexp(abs(value))
    step = math.ldexp(1.0, max(exponent - 8, -133))
    return math.copysign(round(abs(value) / step) * step, value)


def dense_rotation(cos, sin, width):
    rotation = np.eye(width)
    half = len(cos)
    for j in range(half):
        rotation[j, j] = cos[j]
        rotation[j + half, j + half] = cos[j]
        rotation[j, j + half] = -sin[j]
        rotation[j + half, j] = sin[j]
    return rotation


@pytest.mark.parametrize("shape,half", [((1, 1, 4), 2), ((5, 3, 128), 48), ((3, 5, 127), 47)])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers", "constant"])
@pytest.mark.parametrize("rounded", [False, True])
def test_scalar_rms_and_dense_rotation(shape, half, distribution, rounded):
    rng = np.random.default_rng(103)
    projected = rng.normal(size=shape)
    if distribution == "zeros":
        projected.fill(0)
    elif distribution == "outliers":
        projected[0, 0, 0] = 1e4
        projected[-1, -1, -1] = -1e4
    elif distribution == "constant":
        projected[:] = np.arange(1, shape[1] + 1)[None, :, None]
    projected = np.asfortranarray(projected)
    gamma = rng.normal(size=shape[-1])
    gamma[-1] = 0
    angles = rng.uniform(-math.pi, math.pi, size=(shape[0], half))
    cos, sin = np.cos(angles), np.sin(angles)
    originals = [x.copy() for x in (projected, gamma, cos, sin)]
    actual = qk_rope_reference(
        projected, gamma, cos, sin, projection_bf16=rounded, output_bf16=rounded
    )
    expected_projected = np.empty(shape)
    expected_norm = np.empty(shape)
    expected_output = np.empty(shape)
    expected_inverse = np.empty(shape[:2])
    for row in range(shape[0]):
        rotation = dense_rotation(cos[row], sin[row], shape[-1])
        for head in range(shape[1]):
            values = [scalar_bf16(v) if rounded else float(v) for v in projected[row, head]]
            expected_projected[row, head] = values
            inverse = 1 / math.sqrt(math.fsum(v * v for v in values) / shape[-1] + 1e-5)
            expected_inverse[row, head] = inverse
            norm = np.array([v * inverse * gamma[j] for j, v in enumerate(values)])
            expected_norm[row, head] = norm
            output = rotation @ norm
            expected_output[row, head] = [scalar_bf16(v) if rounded else v for v in output]
    np.testing.assert_allclose(actual["projected"], expected_projected, rtol=0, atol=0)
    np.testing.assert_allclose(actual["inverse_rms"], expected_inverse, rtol=3e-15, atol=0)
    np.testing.assert_allclose(actual["normalized"], expected_norm, rtol=3e-15, atol=1e-14)
    np.testing.assert_allclose(
        actual["output"], expected_output.transpose(1, 0, 2), rtol=3e-15, atol=1e-14
    )
    assert actual["output"].flags.c_contiguous
    assert actual["output"].shape == (shape[1], shape[0], shape[2])
    for current, previous in zip((projected, gamma, cos, sin), originals, strict=True):
        np.testing.assert_array_equal(current, previous)


def test_bf16_rounding_ties_subnormals_and_negative_zero():
    values = np.array(
        [
            0.0,
            -0.0,
            1.00390625,
            1.01171875,
            -1.00390625,
            -1.01171875,
            2.0**-134,
            3 * 2.0**-134,
            -(2.0**-134),
            -3 * 2.0**-134,
        ]
    )
    expected = np.array([scalar_bf16(v) for v in values])
    actual = bf16_round_reference(values)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_array_equal(actual[2:6], [1.0, 1.015625, -1.0, -1.015625])
    assert actual.dtype == np.float64


def test_bf16_rounding_explicitly_has_fp32_producer():
    # Direct FP64->BF16 would round up; FP32 first lands exactly on the even tie.
    value = 1.00390625 + 2.0**-30
    assert bf16_round_reference(value) == 1.0


def test_bf16_rounding_preserves_fortran_and_strided_inputs():
    original = np.asfortranarray(np.random.default_rng(107).normal(size=(5, 9)))[:, ::2]
    before = original.copy()
    expected = np.array([[scalar_bf16(v) for v in row] for row in original])
    np.testing.assert_array_equal(bf16_round_reference(original), expected)
    np.testing.assert_array_equal(original, before)


def test_rms_is_per_head_over_all_channels_before_gamma():
    projected = np.array([[[1.0, 2.0, 3.0, 40.0], [10.0, 20.0, 30.0, 400.0]]])
    gamma = np.array([2.0, 3.0, 4.0, 5.0])
    result = qk_rope_reference(
        projected,
        gamma,
        np.ones((1, 1)),
        np.zeros((1, 1)),
        projection_bf16=False,
        output_bf16=False,
    )
    expected_inverse = 1 / np.sqrt(np.mean(projected**2, axis=-1) + 1e-5)
    np.testing.assert_array_equal(result["inverse_rms"], expected_inverse)
    wrong_rotary_only = 1 / np.sqrt(np.mean(projected[..., :2] ** 2, axis=-1) + 1e-5)
    wrong_after_gamma = 1 / np.sqrt(np.mean((projected * gamma) ** 2, axis=-1) + 1e-5)
    assert not np.allclose(expected_inverse, wrong_rotary_only)
    assert not np.allclose(expected_inverse, wrong_after_gamma)
    assert expected_inverse[0, 0] != expected_inverse[0, 1]


def test_split_half_pairing_direction_and_unrotated_tail():
    projected = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    result = qk_rope_reference(
        projected,
        np.ones(6),
        np.zeros((1, 2)),
        np.ones((1, 2)),
        projection_bf16=False,
        output_bf16=False,
    )
    inverse = result["inverse_rms"][0, 0]
    np.testing.assert_allclose(result["output"][0, 0], np.array([-3, -4, 1, 2, 5, 6]) * inverse)


def test_no_rope_refiner_path():
    x = np.arange(30).reshape(3, 2, 5)
    result = qk_rope_reference(x, np.ones(5), np.empty((3, 0)), np.empty((3, 0)))
    np.testing.assert_array_equal(result["rotated"], result["normalized"])
    np.testing.assert_array_equal(
        result["output"], bf16_round_reference(result["normalized"]).transpose(1, 0, 2)
    )


def test_rope_and_channel_gamma_do_not_commute():
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    gamma = np.array([2.0, 1.0, 1.0, 1.0])
    cos, sin = np.zeros((1, 2)), np.ones((1, 2))
    correct = qk_rope_reference(x, gamma, cos, sin, projection_bf16=False, output_bf16=False)[
        "output"
    ]
    rotation = dense_rotation(cos[0], sin[0], 4)
    rotated_first = (rotation @ x[0, 0])[None, None, :]
    wrong = qk_rope_reference(
        rotated_first,
        gamma,
        np.empty((1, 0)),
        np.empty((1, 0)),
        projection_bf16=False,
        output_bf16=False,
    )["output"]
    assert not np.allclose(correct, wrong)


def test_extra_norm_rounding_is_a_different_policy():
    rng = np.random.default_rng(109)
    x = rng.normal(size=(3, 2, 128))
    gamma = bf16_round_reference(rng.normal(size=128))
    angles = rng.normal(size=(3, 48))
    cos, sin = bf16_round_reference(np.cos(angles)), bf16_round_reference(np.sin(angles))
    native = qk_rope_reference(x, gamma, cos, sin)
    alternate = qk_rope_reference(x, gamma, cos, sin, norm_bf16=True)
    np.testing.assert_array_equal(
        alternate["normalized"], bf16_round_reference(native["normalized"])
    )
    assert np.any(native["output"] != alternate["output"])


def test_projection_rounding_is_observable():
    rng = np.random.default_rng(113)
    x = rng.normal(size=(3, 2, 128))
    args = (x, np.ones(128), np.ones((3, 48)), np.zeros((3, 48)))
    rounded = qk_rope_reference(*args)
    unrounded = qk_rope_reference(*args, projection_bf16=False)
    assert np.any(rounded["output"] != unrounded["output"])


@pytest.mark.parametrize("epsilon", [0, -1, True, np.nan, np.inf, "1e-5"])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        qk_rope_reference(
            np.ones((1, 1, 4)), np.ones(4), np.ones((1, 2)), np.zeros((1, 2)), epsilon=epsilon
        )


@pytest.mark.parametrize("index", [0, 1, 2, 3])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1j])
def test_nonfinite_or_complex_operands(index, invalid):
    operands = [np.ones((2, 3, 4)), np.ones(4), np.ones((2, 2)), np.zeros((2, 2))]
    operands[index] = operands[index].astype(complex if invalid == 1j else float)
    operands[index].flat[-1] = invalid
    with pytest.raises(ValueError, match="finite|real"):
        qk_rope_reference(*operands)


@pytest.mark.parametrize(
    "index,value,message",
    [
        (0, np.ones((2, 4)), "projected"),
        (0, np.ones((0, 3, 4)), "projected"),
        (0, np.ones((2, 0, 4)), "projected"),
        (0, np.ones((2, 3, 0)), "projected"),
        (1, np.ones((3, 4)), "gamma"),
        (1, np.ones(3), "gamma"),
        (2, np.ones((2, 3)), "cos"),
        (2, np.ones((1, 2)), "cos"),
        (2, np.ones(2), "cos"),
        (3, np.zeros((2, 1)), "sin"),
    ],
)
def test_invalid_shapes(index, value, message):
    operands = [np.ones((2, 3, 4)), np.ones(4), np.ones((2, 2)), np.zeros((2, 2))]
    operands[index] = value
    with pytest.raises(ValueError, match=message):
        qk_rope_reference(*operands)


@pytest.mark.parametrize("flag", ["projection_bf16", "norm_bf16", "output_bf16"])
def test_rounding_flags_are_explicit_booleans(flag):
    with pytest.raises(ValueError, match=flag):
        qk_rope_reference(
            np.ones((1, 1, 4)), np.ones(4), np.ones((1, 2)), np.zeros((1, 2)), **{flag: "false"}
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1j, 1e100, np.finfo(np.float32).max])
def test_bf16_nonfinite_and_overflow_rejected(value):
    with pytest.raises(ValueError, match="finite|real"):
        bf16_round_reference(value)


def test_float64_norm_and_rotation_overflow_rejected():
    with pytest.raises(ValueError, match="RMS"):
        qk_rope_reference(
            np.full((1, 1, 4), 1e200),
            np.ones(4),
            np.ones((1, 2)),
            np.zeros((1, 2)),
            projection_bf16=False,
            output_bf16=False,
        )
    with pytest.raises(ValueError, match="RoPE"):
        qk_rope_reference(
            np.ones((1, 1, 4)),
            np.full(4, 1e200),
            np.full((1, 2), 1e200),
            np.zeros((1, 2)),
            projection_bf16=False,
            output_bf16=False,
        )

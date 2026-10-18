import numpy as np
import pytest

from m5kernels.gated_residual_reference import gated_residual_reference


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (5, 131)])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers"])
@pytest.mark.parametrize("mapped", [False, True])
def test_matches_scalar_update_with_tails_and_input_immutability(shape, distribution, mapped):
    rng = np.random.default_rng(47)
    residual = rng.normal(size=shape)
    branch = rng.normal(size=shape)
    gate = rng.normal(size=(4, shape[1]) if mapped else shape[1])
    row_map = (np.arange(shape[0], dtype=np.int32) * 3) % 4 if mapped else None
    if distribution == "zeros":
        residual.fill(0)
        branch.fill(0)
        gate.fill(0)
    elif distribution == "outliers":
        residual[-1, -1] *= 1000
        branch[0, 0] *= 1000
        gate.flat[0] *= 1000
    residual = np.asfortranarray(residual)
    branch = np.asfortranarray(branch)
    original = [a.copy() for a in (residual, branch, gate)]
    original_map = row_map.copy() if mapped else None
    expected = np.empty(shape, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            selected = gate[row_map[i], j] if mapped else gate[j]
            expected[i, j] = float(residual[i, j]) + float(selected) * float(branch[i, j])
    actual = gated_residual_reference(residual, branch, gate, row_map=row_map)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    for current, before in zip((residual, branch, gate), original, strict=True):
        np.testing.assert_array_equal(current, before)
    if mapped:
        np.testing.assert_array_equal(row_map, original_map)


@pytest.mark.parametrize("gate_value", [0.0, 1.0, -1.0])
def test_zero_and_unit_gate(gate_value):
    residual = np.arange(21).reshape(3, 7).astype(np.float32)
    branch = np.arange(21, 42).reshape(3, 7).astype(np.float32)
    actual = gated_residual_reference(residual, branch, np.full(7, gate_value))
    np.testing.assert_array_equal(actual, residual + gate_value * branch)


def test_exact_cancellation_and_no_implicit_float32_rounding():
    branch = np.array([[2.0**40, -(2.0**40), 1.0000000001]])
    residual = np.array([[-(2.0**40), 2.0**40, -1.0]])
    actual = gated_residual_reference(residual, branch, np.ones(3))
    np.testing.assert_array_equal(actual[:, :2], np.zeros((1, 2)))
    assert actual[0, 2] == branch[0, 2] - 1.0
    assert actual[0, 2] != 0.0
    assert (branch.astype(np.float32) + residual.astype(np.float32))[0, 2] == 0.0


def test_row_map_selects_repeated_and_boundary_entries():
    residual = np.zeros((5, 7))
    branch = np.ones((5, 7))
    gate = np.arange(28).reshape(4, 7)
    row_map = np.array([3, 0, 2, 0, 3], dtype=np.int32)
    actual = gated_residual_reference(residual, branch, gate, row_map=row_map)
    np.testing.assert_array_equal(actual, gate[row_map])


@pytest.mark.parametrize("shape", [(7,), (0, 7), (3, 0), (1, 3, 7)])
def test_rejects_invalid_residual_dimensions(shape):
    with pytest.raises(ValueError, match="residual"):
        gated_residual_reference(np.ones(shape), np.ones(shape), np.ones(7))


@pytest.mark.parametrize("shape", [(7,), (1, 7), (3, 1), (3, 8)])
def test_rejects_branch_shape_mismatch(shape):
    with pytest.raises(ValueError, match="branch"):
        gated_residual_reference(np.ones((3, 7)), np.ones(shape), np.ones(7))


@pytest.mark.parametrize(
    "gate", [np.ones(6), np.ones((0, 7)), np.ones((2, 6)), 1.0, np.ones((1, 1, 7))]
)
def test_rejects_invalid_gate_shape(gate):
    with pytest.raises(ValueError, match="gate"):
        gated_residual_reference(np.ones((3, 7)), np.ones((3, 7)), gate)


def test_gate_table_requires_explicit_map():
    with pytest.raises(ValueError, match="requires row_map"):
        gated_residual_reference(np.ones((3, 7)), np.ones((3, 7)), np.ones((3, 7)))


def test_shared_gate_rejects_ignored_row_map():
    with pytest.raises(ValueError, match="only supported"):
        gated_residual_reference(
            np.ones((3, 7)), np.ones((3, 7)), np.ones(7), row_map=np.zeros(3, np.int32)
        )


@pytest.mark.parametrize("dtype", [np.int64, np.uint32, np.float32, np.float64, bool])
def test_row_map_requires_int32(dtype):
    with pytest.raises(ValueError, match="int32"):
        gated_residual_reference(
            np.ones((3, 7)), np.ones((3, 7)), np.ones((2, 7)), row_map=np.zeros(3, dtype)
        )


@pytest.mark.parametrize("row_map", [np.zeros(2, np.int32), np.zeros((3, 1), np.int32)])
def test_row_map_requires_vector_of_length_m(row_map):
    with pytest.raises(ValueError, match="length M"):
        gated_residual_reference(np.ones((3, 7)), np.ones((3, 7)), np.ones((2, 7)), row_map=row_map)


@pytest.mark.parametrize("row_map", [[0, -1, 0], [0, 2, 0], [0, 2147483647, 0]])
def test_rejects_row_map_out_of_bounds(row_map):
    with pytest.raises(ValueError, match="indices"):
        gated_residual_reference(
            np.ones((3, 7)),
            np.ones((3, 7)),
            np.ones((2, 7)),
            row_map=np.array(row_map, np.int32),
        )


@pytest.mark.parametrize("name", ["residual", "branch", "gate"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1j])
def test_rejects_nonfinite_or_complex_operands(name, invalid):
    operands = dict(residual=np.ones((3, 7)), branch=np.ones((3, 7)), gate=np.ones(7))
    operands[name] = operands[name].astype(complex if invalid == 1j else float)
    operands[name].flat[0] = invalid
    with pytest.raises(ValueError, match="finite|real"):
        gated_residual_reference(**operands)


def test_rejects_nonfinite_unselected_gate_table_entry():
    gate = np.ones((2, 7))
    gate[1, 0] = np.nan
    with pytest.raises(ValueError, match="gate must be finite"):
        gated_residual_reference(
            np.ones((3, 7)), np.ones((3, 7)), gate, row_map=np.zeros(3, np.int32)
        )


def test_rejects_nonfinite_result_from_finite_inputs():
    maximum = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="exceeds"):
        gated_residual_reference([[maximum]], [[maximum]], [1.0])

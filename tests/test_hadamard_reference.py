import numpy as np
import pytest

from m5kernels.hadamard_reference import block_hadamard_reference, hadamard_signs


def dense_hadamard(block_size):
    # Independent definition: H[i,j] = (-1)**popcount(i & j) / sqrt(B).
    return np.array(
        [[(-1) ** (i & j).bit_count() for j in range(block_size)] for i in range(block_size)],
        dtype=np.float64,
    ) / np.sqrt(block_size)


@pytest.mark.parametrize("block_size", [1, 2, 4, 32, 128])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers"])
@pytest.mark.parametrize("signed", [False, True])
def test_matches_dense_definition_with_odd_rows(block_size, distribution, signed):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(3, 3 * block_size)).astype(np.float32)
    if distribution == "zeros":
        x.fill(0)
    elif distribution == "outliers":
        x[:, ::block_size] *= 1000
    signs = hadamard_signs(x.shape[1], 17) if signed else None
    original = x.copy()
    expected_input = x.astype(np.float64) * (signs if signed else 1)
    expected = (expected_input.reshape(-1, block_size) @ dense_hadamard(block_size)).reshape(
        x.shape
    )
    actual = block_hadamard_reference(x, block_size, signs)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-11)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("block_size", [1, 8, 128])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers"])
def test_same_rotation_preserves_rectangular_gemm(block_size, distribution):
    rng = np.random.default_rng(21)
    a = rng.normal(size=(3, 3 * block_size))
    w = rng.normal(size=(5, 3 * block_size))
    if distribution == "zeros":
        a.fill(0)
        w.fill(0)
    elif distribution == "outliers":
        a[:, 0] *= 1000
        w[:, -1] *= 1000
    signs = hadamard_signs(a.shape[1], 23)
    rotated_a = block_hadamard_reference(a, block_size, signs)
    rotated_w = block_hadamard_reference(w, block_size, signs)
    np.testing.assert_allclose(rotated_a @ rotated_w.T, a @ w.T, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(rotated_a), np.linalg.norm(a), rtol=3e-15)


def test_signs_are_applied_before_hadamard_and_are_repeatable():
    x = np.array([[1.0, 2.0, 4.0, 8.0]])
    signs = np.array([1, -1, 1, -1])
    result = block_hadamard_reference(x, 4, signs)
    np.testing.assert_allclose(result, (x * signs) @ dense_hadamard(4), atol=0)
    assert not np.allclose(result, (x @ dense_hadamard(4)) * signs)
    first = hadamard_signs(256, 17)
    np.testing.assert_array_equal(first, hadamard_signs(256, 17))
    assert not np.array_equal(first, hadamard_signs(256, 18))
    assert first.dtype == np.int8
    assert set(first.tolist()) == {-1, 1}


def test_noncontiguous_inputs_and_signs_are_not_mutated():
    base = np.arange(48).reshape(3, 16)
    x = base[:, ::2]
    signs = hadamard_signs(8, 17)
    original_signs = signs.copy()
    actual = block_hadamard_reference(x, 4, signs)
    dense = np.kron(np.eye(2), dense_hadamard(4))
    np.testing.assert_allclose(actual, (x * signs) @ dense)
    np.testing.assert_array_equal(base, np.arange(48).reshape(3, 16))
    np.testing.assert_array_equal(signs, original_signs)


def test_fortran_order_input_matches_dense_definition():
    rng = np.random.default_rng(29)
    x = np.asfortranarray(rng.normal(size=(5, 768)))
    original = x.copy()
    signs = hadamard_signs(768, 17)
    assert x.flags.f_contiguous and not x.flags.c_contiguous
    dense = np.kron(np.eye(3), dense_hadamard(256))
    actual = block_hadamard_reference(x, 256, signs)
    np.testing.assert_allclose(actual, (x * signs) @ dense, rtol=3e-14, atol=1e-11)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("block_size", [0, -2, 3, 6, 2.0, True])
def test_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="block_size"):
        block_hadamard_reference(np.ones((3, 8)), block_size)


@pytest.mark.parametrize("x", [np.ones(8), np.ones((1, 2, 4)), np.ones((0, 8)), np.ones((3, 0))])
def test_rejects_invalid_dimensions(x):
    with pytest.raises(ValueError, match="shape"):
        block_hadamard_reference(x, 4)


@pytest.mark.parametrize("k", [3, 7, 9])
def test_rejects_nondivisible_reduction_width(k):
    with pytest.raises(ValueError, match="divisible"):
        block_hadamard_reference(np.ones((3, k)), 4)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1j])
def test_rejects_nonfinite_or_complex_input(bad):
    with pytest.raises(ValueError, match="finite|real"):
        block_hadamard_reference(np.array([[bad, 1, 2, 3]]), 4)


@pytest.mark.parametrize(
    "signs",
    [np.ones(3), np.ones((1, 4)), [1, 1, 0, -1], [1, 1, np.nan, 1], [1, 1, 1j, 1]],
)
def test_rejects_invalid_signs(signs):
    with pytest.raises(ValueError, match="signs"):
        block_hadamard_reference(np.ones((3, 4)), 4, signs)


@pytest.mark.parametrize("k", [0, -1, 2.0, True])
def test_sign_generator_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="K"):
        hadamard_signs(k, 17)


def test_rejects_nonfinite_output_from_finite_inputs():
    with pytest.raises(ValueError, match="output"):
        block_hadamard_reference(np.full((1, 4), np.finfo(np.float64).max), 4)

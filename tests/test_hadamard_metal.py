import sys

import numpy as np
import pytest

pytestmark = pytest.mark.metal
if sys.platform != "darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)
mx = pytest.importorskip("mlx.core")

from m5kernels.hadamard import rotate  # noqa: E402
from m5kernels.hadamard_reference import (  # noqa: E402
    block_hadamard_reference,
    hadamard_signs,
)


@pytest.mark.parametrize("block_size", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("kind", ["normal", "zeros", "outliers"])
def test_metal_rotation_full_output(block_size, kind):
    # Odd row counts and multiple blocks catch row/sign-indexing mistakes.
    x = np.random.default_rng(53).standard_normal((7, block_size * 3), dtype=np.float32)
    if kind == "zeros":
        x.fill(0)
    elif kind == "outliers":
        x[:, ::31] *= 100
    signs = hadamard_signs(x.shape[1], 71)
    actual = np.asarray(rotate(mx.array(x), block_size, mx.array(signs)))
    expected = block_hadamard_reference(x, block_size, signs)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=4e-5)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("implementation", ["metal", "mlx"])
def test_input_precision_and_noncontiguous_layout(dtype, implementation):
    rng = np.random.default_rng(19)
    # Transposition exercises required input layout conversion as well as dtype.
    x = mx.array(rng.standard_normal((768, 5), dtype=np.float32)).astype(dtype).T
    signs = hadamard_signs(x.shape[1], 23)
    expected = block_hadamard_reference(np.asarray(x.astype(mx.float32)), 256, signs)
    actual = np.asarray(rotate(x, 256, mx.array(signs), implementation=implementation))
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("block_size", [32, 256, 1024])
@pytest.mark.parametrize("implementation", ["metal", "mlx"])
def test_dot_products_preserved(block_size, implementation):
    rng = np.random.default_rng(101)
    x = rng.standard_normal((5, block_size * 2), dtype=np.float32)
    w = rng.standard_normal((9, block_size * 2), dtype=np.float32)
    x[:, 7] *= 50
    w[:, 21] *= 50
    signs = mx.array(hadamard_signs(x.shape[1], 5))
    rotated_x = np.asarray(rotate(mx.array(x), block_size, signs, implementation))
    rotated_w = np.asarray(rotate(mx.array(w), block_size, signs, implementation))
    actual = rotated_x.astype(np.float64) @ rotated_w.astype(np.float64).T
    expected = x.astype(np.float64) @ w.astype(np.float64).T
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-4)


@pytest.mark.parametrize("block_size", [0, 33, True, 32.0])
def test_invalid_block_rejected(block_size):
    with pytest.raises(ValueError, match="power of two"):
        rotate(mx.zeros((1, 64)), block_size, mx.ones(64, dtype=mx.int8))


@pytest.mark.parametrize("block_size", [16, 2048])
def test_unsupported_block_rejected(block_size):
    with pytest.raises(NotImplementedError, match="supported block sizes"):
        rotate(mx.zeros((1, 2048)), block_size, mx.ones(2048, dtype=mx.int8))


def test_no_implicit_padding():
    with pytest.raises(NotImplementedError, match="divisible"):
        rotate(mx.zeros((1, 65)), 32, mx.ones(65, dtype=mx.int8))


def test_invalid_inputs():
    with pytest.raises(ValueError, match="nonempty shape"):
        rotate(mx.zeros((64,)), 32, mx.ones(64, dtype=mx.int8))
    with pytest.raises(ValueError, match="input dtype"):
        rotate(mx.zeros((1, 64), dtype=mx.int8), 32, mx.ones(64, dtype=mx.int8))
    with pytest.raises(ValueError, match="int8 vector"):
        rotate(mx.zeros((1, 64)), 32, mx.ones(64))
    with pytest.raises(ValueError, match="int8 vector"):
        rotate(mx.zeros((1, 64)), 32, mx.ones(32, dtype=mx.int8))
    with pytest.raises(ValueError, match="unknown Hadamard implementation"):
        rotate(mx.zeros((1, 64)), 32, mx.ones(64, dtype=mx.int8), "unknown")


def test_thread_limit_rejected(monkeypatch):
    monkeypatch.setattr("m5kernels.hadamard.MAX_THREADS_PER_THREADGROUP", 64)
    with pytest.raises(NotImplementedError, match="threadgroup thread limit"):
        rotate(mx.zeros((1, 128)), 128, mx.ones(128, dtype=mx.int8))

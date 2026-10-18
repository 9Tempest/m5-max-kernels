"""Independent float64 references for randomized block Hadamard rotations.

For row-major X, this computes X D H, with a shared diagonal sign matrix D
and normalized, block-diagonal Sylvester H. Applying the same rotation to
activation and weight rows preserves X W.T before quantization.
"""

import numbers

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def hadamard_signs(k, seed):
    """Generate reproducible shared signs, independent of the global RNG state."""
    k = _positive_integer(k, "K")
    rng = np.random.Generator(np.random.PCG64(seed))
    return (2 * rng.integers(0, 2, size=k, dtype=np.int8) - 1).astype(np.int8)


def block_hadamard_reference(x, block_size, signs=None):
    """Return X D H in float64; require finite real rows and exact K divisibility.

    ``block_size`` must be a positive power of two. ``signs``, when provided,
    contains exactly K entries, each +1 or -1; otherwise D is the identity.
    The function neither pads the reduction dimension nor mutates its inputs.
    """
    block_size = _positive_integer(block_size, "block_size")
    if block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")
    x = np.asarray(x)
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("input must have nonempty shape [rows, K]")
    if np.iscomplexobj(x):
        raise ValueError("input must be real")
    if x.shape[1] % block_size:
        raise ValueError("K must be divisible by block_size")
    # Butterfly reshapes must be views of this buffer even for Fortran-order inputs.
    out = np.array(x, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(out).all():
        raise ValueError("input must be finite")
    if signs is not None:
        signs = np.asarray(signs)
        if signs.shape != (x.shape[1],) or np.iscomplexobj(signs):
            raise ValueError("signs must be a real vector of length K")
        if not np.isin(signs, [-1, 1]).all():
            raise ValueError("signs must contain only -1 and +1")
        out *= signs
    # Normalize before the butterflies to reduce intermediate overflow risk.
    out /= np.sqrt(float(block_size))
    blocks = out.reshape(-1, block_size)
    width = 1
    with np.errstate(over="ignore", invalid="ignore"):
        while width < block_size:
            groups = blocks.reshape(-1, 2 * width)
            left = groups[:, :width].copy()
            right = groups[:, width:].copy()
            groups[:, :width] = left + right
            groups[:, width:] = left - right
            width *= 2
    if not np.isfinite(out).all():
        raise ValueError("transform output exceeds finite float64 range")
    return out

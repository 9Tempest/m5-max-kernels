"""Ideal float64 references for normalization, modulation, rotation and INT8.

No intermediate BF16/FP32 rounding is implicit. A staged backend's explicit
rounding boundaries must be evaluated separately from this ideal reference.
"""

import numbers

import numpy as np

from .hadamard_reference import block_hadamard_reference


def _finite_real(value, name):
    value = np.asarray(value)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def norm_modulate_reference(x, gamma, scale, shift, *, norm="rms", eps=1e-5):
    """Return norm(X) * gamma * (1 + scale) + shift in float64.

    X has nonempty shape [rows, K]; gamma, scale and shift have shape [K].
    RMS uses the row mean square. Layer normalization centers each row and
    uses population variance. Epsilon is added inside the square root.
    There is no separate normalization bias; shift is the modulation bias.
    """
    if norm not in ("rms", "layer"):
        raise ValueError("norm must be 'rms' or 'layer'")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, numbers.Real)
        or not np.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("eps must be a positive finite real scalar")
    x = _finite_real(x, "input")
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("input must have nonempty shape [rows, K]")
    vectors = []
    for value, name in ((gamma, "gamma"), (scale, "scale"), (shift, "shift")):
        value = _finite_real(value, name)
        if value.shape != (x.shape[1],):
            raise ValueError(f"{name} must have shape [K]")
        vectors.append(value)
    gamma, scale, shift = vectors
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if norm == "layer":
            # Shift before the mean so constant rows center exactly, including
            # strided layouts, without subtracting two large near-equal means.
            centered = x - x[:, :1]
            centered = centered - np.mean(centered, axis=1, keepdims=True)
        else:
            centered = x
        moment = np.mean(centered * centered, axis=1, keepdims=True)
        normalized = centered / np.sqrt(moment + float(eps))
        modulated = normalized * gamma * (1.0 + scale) + shift
    if not np.isfinite(moment).all() or not np.isfinite(modulated).all():
        raise ValueError("normalization or modulation exceeds finite float64 range")
    return modulated


def norm_hadamard_quantize_reference(
    x, gamma, scale, shift, block_size, signs=None, *, norm="rms", eps=1e-5
):
    """Return every ideal stage of norm/modulation -> X D H -> per-row INT8.

    The returned dictionary contains float64 ``modulated`` and ``rotated``
    matrices, int8 ``quantized``, float64 row ``scales`` and ``dequantized``.
    INT8 uses symmetric [-127,127] with round-to-nearest-even, max-abs/127
    row scales and scale 1 for zero rows. Hadamard blocks are normalized,
    without padding. Shared signs are applied after modulation, before H.
    """
    modulated = norm_modulate_reference(x, gamma, scale, shift, norm=norm, eps=eps)
    rotated = block_hadamard_reference(modulated, block_size, signs)
    peak = np.max(np.abs(rotated), axis=1)
    scales = np.where(peak == 0, 1.0, peak / 127.0)
    if not np.isfinite(scales).all() or not np.all(scales > 0):
        raise ValueError("INT8 scale exceeds finite positive float64 range")
    quantized = np.clip(np.rint(rotated / scales[:, None]), -127, 127).astype(np.int8)
    dequantized = quantized.astype(np.float64) * scales[:, None]
    if not np.isfinite(dequantized).all():
        raise ValueError("INT8 dequantization exceeds finite float64 range")
    return {
        "modulated": modulated,
        "rotated": rotated,
        "quantized": quantized,
        "scales": scales,
        "dequantized": dequantized,
    }

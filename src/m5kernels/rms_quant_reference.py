"""Independent float64 reference for folding positive RMS factors into INT8 scales.

This is RMS normalization with a per-channel gamma and optional signed block
Hadamard rotation. There is no additive shift or intermediate BF16 rounding.
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


def rms_quantize_reference(x, gamma, signs=None, block_size=1, eps=1e-5):
    """Return ideal normalized/rotated values and their folded per-row INT8 encoding.

    X has nonempty shape [M,K], gamma has shape [K], and optional signs have
    shape [K] with entries +/-1. RMS statistics use original X, while the
    quantization maximum follows gamma, signs and normalized block H.

    Let Z=(X*gamma) D H and r=1/sqrt(mean(X**2)+eps). For nonzero rows,
    Q=clip(rint(Z/(amax(abs(Z))/127)),-127,127) and S=amax(abs(Z))/127*r.
    A zero row of Z uses Q=0 and S=1. Rounding is nearest-even. Positive r
    therefore affects scales but not codes; this requires no additive shift
    and no intervening dtype rounding.
    """
    if (
        isinstance(eps, bool)
        or not isinstance(eps, numbers.Real)
        or not np.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("eps must be a positive finite real scalar")
    x = _finite_real(x, "input")
    gamma = _finite_real(gamma, "gamma")
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("input must have nonempty shape [M,K]")
    if gamma.shape != (x.shape[1],):
        raise ValueError("gamma must have shape [K]")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        moment = np.mean(x * x, axis=1)
        inverse_rms = 1.0 / np.sqrt(moment + float(eps))
        weighted = x * gamma
    if (
        not np.isfinite(moment).all()
        or not np.isfinite(inverse_rms).all()
        or not np.all(inverse_rms > 0)
        or not np.isfinite(weighted).all()
    ):
        raise ValueError("RMS or gamma multiplication exceeds finite float64 range")
    unscaled_rotated = block_hadamard_reference(weighted, block_size, signs)
    peak = np.max(np.abs(unscaled_rotated), axis=1)
    nonzero = peak != 0
    unscaled_scales = np.where(nonzero, peak / 127.0, 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized_rotated = unscaled_rotated * inverse_rms[:, None]
        scales = np.where(nonzero, unscaled_scales * inverse_rms, 1.0)
    if (
        not np.isfinite(normalized_rotated).all()
        or not np.isfinite(scales).all()
        or not np.all(scales > 0)
        or not np.all(unscaled_scales > 0)
    ):
        raise ValueError("normalized values or INT8 scales exceed finite float64 range")
    quantized = np.clip(np.rint(unscaled_rotated / unscaled_scales[:, None]), -127, 127).astype(
        np.int8
    )
    dequantized = quantized.astype(np.float64) * scales[:, None]
    if not np.isfinite(dequantized).all():
        raise ValueError("INT8 dequantization exceeds finite float64 range")
    return {
        "normalized_rotated": normalized_rotated,
        "quantized": quantized,
        "scales": scales,
        "dequantized": dequantized,
        "unscaled_rotated": unscaled_rotated,
        "inverse_rms": inverse_rms,
        "unscaled_scales": unscaled_scales,
    }

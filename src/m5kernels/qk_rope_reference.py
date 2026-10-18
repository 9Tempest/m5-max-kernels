"""Independent NumPy reference for per-head Q/K RMS normalization and split RoPE.

The default rounding boundaries follow the pinned h3.c native contract, not
eager Diffusers byte parity. Arithmetic between explicit BF16 boundaries is
float64; GPU FP32 reduction order and contraction are intentionally not copied.
See docs/qk-rope-fusion.md for pinned source provenance and layout semantics.
"""

import numbers

import numpy as np


def _finite_real(value, name):
    value = np.asarray(value)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def bf16_round_reference(value):
    """Convert to FP32, then BF16 nearest-even, returning float64 decoded values.

    This explicitly models a BF16 cast from an FP32 producer. The initial FP32
    conversion is deliberate; this is not a direct float64-to-BF16 conversion.
    Finite inputs that overflow either output format are rejected. Subnormals
    and signed zeros are retained, and the input is never mutated.
    """
    value = _finite_real(value, "BF16 input")
    with np.errstate(over="ignore", invalid="ignore"):
        fp32 = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(fp32).all():
        raise ValueError("BF16 input exceeds finite FP32 range")
    bits = fp32.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    result = rounded.view(np.float32).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("BF16 rounding exceeds finite BF16 range")
    return result


def qk_rope_reference(
    projected,
    gamma,
    cos,
    sin,
    *,
    epsilon=1e-5,
    projection_bf16=True,
    norm_bf16=False,
    output_bf16=True,
):
    """Return Q or K normalization/RoPE intermediates and head-major output.

    projected[M,H,D] is the projection before its optional BF16 boundary.
    gamma[D] is shared by all heads and rows. cos/sin[M,R] contain one value
    for each split-half pair (j,j+R), with 0 <= 2R <= D. Channels 2R:D are
    normalized but unrotated. R=0 implements the text refiner's no-RoPE path.
    Gamma and trigonometric values are consumed as supplied: pass decoded BF16
    values to model native BF16 inputs, rather than rounding them implicitly.

    RMS uses every D channel of each (row,head), before gamma, with no mean
    subtraction. The native defaults round projection and final output only;
    norm_bf16=True is an explicit alternate precision experiment. It does not
    emulate eager-framework product rounding. All intermediates are float64
    row-major [M,H,D], inverse_rms is [M,H], and output is a contiguous [H,M,D]
    float64 array containing the chosen output precision's decoded values.
    """
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, numbers.Real)
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be a positive finite real scalar")
    for name, flag in (
        ("projection_bf16", projection_bf16),
        ("norm_bf16", norm_bf16),
        ("output_bf16", output_bf16),
    ):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    projected = _finite_real(projected, "projected")
    gamma = _finite_real(gamma, "gamma")
    cos = _finite_real(cos, "cos")
    sin = _finite_real(sin, "sin")
    if projected.ndim != 3 or not all(projected.shape):
        raise ValueError("projected must have nonempty shape [M,H,D]")
    rows, _, width = projected.shape
    if gamma.shape != (width,):
        raise ValueError("gamma must have shape [D]")
    if cos.ndim != 2 or cos.shape[0] != rows or cos.shape[1] * 2 > width:
        raise ValueError("cos must have shape [M,R] with 0 <= 2R <= D")
    if sin.shape != cos.shape:
        raise ValueError("sin must have the same shape as cos")
    projected = (
        bf16_round_reference(projected)
        if projection_bf16
        else np.array(projected, dtype=np.float64, order="C", copy=True)
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        moment = np.mean(projected * projected, axis=-1)
        inverse_rms = 1.0 / np.sqrt(moment + float(epsilon))
        normalized = projected * inverse_rms[..., None] * gamma
    if (
        not np.isfinite(moment).all()
        or not np.isfinite(inverse_rms).all()
        or not np.all(inverse_rms > 0)
        or not np.isfinite(normalized).all()
    ):
        raise ValueError("RMS normalization exceeds finite float64 range")
    if norm_bf16:
        normalized = bf16_round_reference(normalized)
    half = cos.shape[1]
    rotated = normalized.copy()
    lower = normalized[..., :half]
    upper = normalized[..., half : 2 * half]
    with np.errstate(over="ignore", invalid="ignore"):
        rotated[..., :half] = lower * cos[:, None, :] - upper * sin[:, None, :]
        rotated[..., half : 2 * half] = upper * cos[:, None, :] + lower * sin[:, None, :]
    if not np.isfinite(rotated).all():
        raise ValueError("RoPE exceeds finite float64 range")
    output = bf16_round_reference(rotated) if output_bf16 else rotated
    output = np.ascontiguousarray(output.transpose(1, 0, 2))
    return {
        "projected": projected,
        "inverse_rms": inverse_rms,
        "normalized": normalized,
        "rotated": rotated,
        "output": output,
    }

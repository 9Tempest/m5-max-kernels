"""One-dispatch normalization, modulation, block rotation and INT8 packing.

The modulation contract is norm(X) * gamma * (1 + scale) + shift, evaluated in
FP32, optionally rounded to X's dtype after the complete expression. Signs
are applied after this rounding and before normalized Hadamard butterflies.
"""

from functools import lru_cache
import math
from numbers import Integral, Real
from pathlib import Path
import re

import mlx.core as mx

from .hadamard import SUPPORTED_BLOCK_SIZES

SHADERS = Path(__file__).parent / "shaders"
FLOAT_DTYPES = (mx.float16, mx.bfloat16, mx.float32)


def _validate_common(x, gamma, scale, shift, norm, epsilon, round_output, threads, row_map):
    if len(x.shape) != 2 or not all(x.shape):
        raise ValueError("input must have nonempty shape [rows, K]")
    if x.dtype not in FLOAT_DTYPES:
        raise ValueError("input dtype must be float16, bfloat16 or float32")
    if gamma.shape != (x.shape[1],) or gamma.dtype not in FLOAT_DTYPES:
        raise ValueError("gamma must be a floating vector of length K")
    if row_map is None:
        if scale.shape != (x.shape[1],) or shift.shape != scale.shape:
            raise ValueError("scale and shift must be vectors of length K without row_map")
    else:
        if (
            len(scale.shape) != 2
            or not all(scale.shape)
            or scale.shape[1] != x.shape[1]
            or shift.shape != scale.shape
        ):
            raise ValueError("mapped scale and shift must have matching nonempty shape [P,K]")
        if row_map.shape != (x.shape[0],) or row_map.dtype != mx.int32:
            raise ValueError("row_map must be an int32 vector of length M")
    if scale.dtype not in FLOAT_DTYPES or shift.dtype not in FLOAT_DTYPES:
        raise ValueError("scale and shift must have floating dtype")
    if norm == "layernorm":
        norm = "layer"
    if norm not in ("rms", "layer"):
        raise ValueError("norm must be 'rms' or 'layer'")
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise ValueError("epsilon must be a positive finite real scalar")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite real scalar")
    if round_output not in ("input", "fp32"):
        raise ValueError("round_output must be 'input' or 'fp32'")
    if isinstance(threads, bool) or not isinstance(threads, Integral):
        raise ValueError("threads must be an integer")
    if threads not in (128, 256):
        raise NotImplementedError("supported threadgroup sizes are 128 and 256")
    return norm


@lru_cache(None)
def _fused_kernel(width, block_size, threads, norm, epsilon, passes, tag, mapped):
    source = (SHADERS / "norm_hadamard.metal").read_text()
    for token, value in {
        "$WIDTH": str(width),
        "$BLOCK": str(block_size),
        "$THREADS": str(threads),
        "$LAYER": "true" if norm == "layer" else "false",
        "$TWO_READ": "true" if passes == "two_read" else "false",
        "$EPSILON": repr(float(epsilon)) + "f",
        "$MOD_OFFSET": "uint(RowMap[row])*width" if mapped else "0",
    }.items():
        source = source.replace(token, value)
    return mx.fast.metal_kernel(
        name=f"norm_hadamard_int8_{width}_{block_size}_{threads}_{norm}_{passes}_{tag}_{int(mapped)}",
        input_names=["X", "Gamma", "Scale", "Shift", "Signs"] + (["RowMap"] if mapped else []),
        output_names=["Q", "S"],
        source=source,
    )


def fused_norm_hadamard_quantize(
    x,
    gamma,
    scale,
    shift,
    signs,
    block_size=256,
    *,
    norm="rms",
    epsilon=1e-5,
    round_output="input",
    threads=256,
    passes="one_read",
    tag="default",
    row_map=None,
):
    """Return symmetric per-row (INT8 codes, FP32 scales), without padding.

    X has shape [M,K]; gamma and shared +/-1 int8 signs have shape [K].
    Scale/shift are shared [K] vectors, or [P,K] tables when row_map is an
    int32[M] table-index vector. Map indices must satisfy 0 <= index < P.
    Signs, map bounds and finite values are caller preconditions, avoiding a
    hidden synchronized check in timed execution. The independent reference
    and harness validate values. FP32 intermediates must remain finite.

    Each threadgroup handles one row. Each SIMD group transforms complete
    blocks using lane shuffles and thread-local arrays. Cross-SIMD reductions
    compute row statistics and the quantization peak. ``one_read`` retains X
    in thread-local arrays; ``two_read`` reloads it after the first statistic
    reduction. RMS needs two row reductions, centered LayerNorm needs three.

    Round-to-nearest-even INT8 codes lie in [-127,127]. Scales are row peak
    divided by 127, with scale=1 for an all-zero rotated row.
    """
    norm = _validate_common(x, gamma, scale, shift, norm, epsilon, round_output, threads, row_map)
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, Integral)
        or block_size <= 0
        or block_size & (block_size - 1)
    ):
        raise ValueError("block_size must be a positive integer power of two")
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise NotImplementedError("supported block sizes are powers of two from 32 through 1024")
    if x.shape[1] % block_size:
        raise NotImplementedError("K must be divisible by block_size; padding is not implicit")
    if signs.shape != (x.shape[1],) or signs.dtype != mx.int8:
        raise ValueError("signs must be an int8 vector of length K containing only +/-1")
    if passes not in ("one_read", "two_read"):
        raise ValueError("passes must be 'one_read' or 'two_read'")
    if not isinstance(tag, str) or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tag) is None:
        raise ValueError("tag must be a valid nonempty Metal identifier")
    return tuple(
        _fused_kernel(
            x.shape[1],
            int(block_size),
            int(threads),
            norm,
            epsilon,
            passes,
            tag,
            row_map is not None,
        )(
            inputs=[x, gamma, scale, shift, signs] + ([row_map] if row_map is not None else []),
            grid=(x.shape[0] * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            template=[("O", x.dtype if round_output == "input" else mx.float32)],
            output_shapes=[x.shape, (x.shape[0],)],
            output_dtypes=[mx.int8, mx.float32],
        )
    )


@lru_cache(None)
def _norm_kernel(norm, epsilon, threads, mapped):
    source = (SHADERS / "norm_modulate.metal").read_text()
    for token, value in {
        "$THREADS": str(threads),
        "$LAYER": "true" if norm == "layer" else "false",
        "$EPSILON": repr(float(epsilon)) + "f",
        "$MOD_OFFSET": "uint(RowMap[row])*width" if mapped else "0",
    }.items():
        source = source.replace(token, value)
    return mx.fast.metal_kernel(
        name=f"norm_modulate_{norm}_{threads}_{int(mapped)}",
        input_names=["X", "Gamma", "Scale", "Shift"] + (["RowMap"] if mapped else []),
        output_names=["Y"],
        source=source,
    )


def unfused_norm(
    x,
    gamma,
    scale,
    shift,
    *,
    norm="rms",
    epsilon=1e-5,
    round_output="input",
    threads=256,
    row_map=None,
):
    """Return norm+modulation only, using the fused path's rounding contract.

    RMS follows the existing ``rms_modulate`` shader's arithmetic, with
    configurable epsilon. LayerNorm uses a centered variance calculation.
    This is a staged comparison baseline; it performs no rotation or packing.
    """
    norm = _validate_common(x, gamma, scale, shift, norm, epsilon, round_output, threads, row_map)
    dtype = x.dtype if round_output == "input" else mx.float32
    return _norm_kernel(norm, epsilon, int(threads), row_map is not None)(
        inputs=[x, gamma, scale, shift] + ([row_map] if row_map is not None else []),
        grid=(x.shape[0] * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        template=[("O", dtype)],
        output_shapes=[x.shape],
        output_dtypes=[dtype],
    )[0]

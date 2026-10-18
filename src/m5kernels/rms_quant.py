"""RMS dynamic INT8 quantization with the inverse RMS folded into row scales.

This experiment has ZERO additive shift and no BF16/FP16 boundary after RMS
normalization. A shared FP32 gamma vector may contain a previously combined
gamma * (1 + AdaLN scale). It is not interchangeable with norm+modulation
paths that include a shift or a reduced-precision intermediate.
"""

from functools import lru_cache
import math
from numbers import Integral, Real
from pathlib import Path
import re
import struct

import mlx.core as mx

BLOCK_SIZES = (1, 32, 64, 128, 256, 512, 1024)


@lru_cache(None)
def _kernel(width, block_size, threads, fold_scale, epsilon, tag):
    source = (Path(__file__).parent / "shaders" / "rms_quant.metal").read_text()
    for token, value in {
        "$WIDTH": str(width),
        "$BLOCK": str(block_size),
        "$THREADS": str(threads),
        "$FOLD": "true" if fold_scale else "false",
        "$EPSILON": repr(float(epsilon)) + "f",
        "$INDEX": ("slot*threads+tid" if block_size == 1 else "(slot*groups+sg)*block+reg*32+lane"),
    }.items():
        source = source.replace(token, value)
    return mx.fast.metal_kernel(
        name=f"rms_quant_{width}_{block_size}_{threads}_{int(fold_scale)}_{tag}",
        input_names=["X", "Gamma", "Signs"],
        output_names=["Q", "S"],
        source=source,
    )


def rms_quantize(
    x,
    gamma,
    signs,
    block_size=1,
    *,
    threads=256,
    fold_scale=True,
    epsilon=1e-5,
    tag="default",
):
    """Return per-row INT8 codes and FP32 scales for signed RMS(X)*gamma*H.

    X is BF16/FP16/FP32 [M,K], gamma is a shared FP32[K] multiplier, and
    signs is a shared +/-1 int8[K] vector. Block size 1 omits H and permits
    arbitrary K. Larger supported block sizes divide K exactly, without
    padding. All arithmetic and transform intermediates use FP32; there is
    no post-normalization cast to the input dtype and no additive shift.

    ``fold_scale=True`` computes u=(X*gamma)*D*H and reduces sum(X^2) and
    max(abs(u)) together. Codes use round(127*u/max(abs(u))); scales include
    the inverse RMS. ``False`` first computes inverse RMS, normalizes X,
    then transforms and performs a separate row-peak reduction. Both modes
    use scale=1 and q=0 when the transformed row is zero.

    The caller verifies finite inputs, +/-1 signs, and representable FP32
    intermediates/scales outside timing; this function checks metadata only.
    """
    if len(x.shape) != 2 or not all(x.shape):
        raise ValueError("input must have nonempty shape [M,K]")
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("input dtype must be float16, bfloat16 or float32")
    if gamma.shape != (x.shape[1],) or gamma.dtype != mx.float32:
        raise ValueError("gamma must be a float32 vector of length K")
    if signs.shape != (x.shape[1],) or signs.dtype != mx.int8:
        raise ValueError("signs must be an int8 vector of length K containing only +/-1")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, Integral)
        or block_size <= 0
        or block_size & (block_size - 1)
    ):
        raise ValueError("block_size must be a positive integer power of two")
    if block_size not in BLOCK_SIZES:
        raise NotImplementedError("supported blocks are 1 and powers of two from 32 through 1024")
    if x.shape[1] % block_size:
        raise NotImplementedError("K must be divisible by block_size; padding is not implicit")
    if isinstance(threads, bool) or not isinstance(threads, Integral):
        raise ValueError("threads must be an integer")
    if threads not in (128, 256, 512):
        raise NotImplementedError("supported threadgroup sizes are 128, 256 and 512")
    if not isinstance(fold_scale, bool):
        raise ValueError("fold_scale must be a boolean")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be a positive finite real scalar")
    try:
        epsilon = struct.unpack("=f", struct.pack("=f", float(epsilon)))[0]
    except OverflowError:
        raise ValueError("epsilon must remain finite and positive in float32") from None
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must remain finite and positive in float32")
    if not isinstance(tag, str) or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tag) is None:
        raise ValueError("tag must be a valid nonempty Metal identifier")
    return tuple(
        _kernel(x.shape[1], int(block_size), int(threads), fold_scale, epsilon, tag)(
            inputs=[x, gamma, signs],
            grid=(x.shape[0] * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[x.shape, (x.shape[0],)],
            output_dtypes=[mx.int8, mx.float32],
        )
    )

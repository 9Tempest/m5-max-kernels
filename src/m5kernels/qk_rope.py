"""Head-aligned Q/K projection, RMSNorm and split-half RoPE experiments.

Each call handles one Q or K projection in ordinary [N,K] weight order.
Grouped checkpoint QKV weights must be split/reordered before calling.
The native-style boundaries are BF16 projection, FP32 RMS/gamma/RoPE,
then BF16 head-major output; there is no BF16 norm intermediate.
"""

from functools import lru_cache
import math
from numbers import Integral, Real
from pathlib import Path
import struct

import mlx.core as mx

_SHADERS = Path(__file__).parent / "shaders"
_HEADER = (
    "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
)
HEAD_DIM = 128


def _epsilon(epsilon):
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be a positive finite real scalar")
    try:
        value = struct.unpack("=f", struct.pack("=f", float(epsilon)))[0]
    except OverflowError:
        raise ValueError("epsilon must remain finite and positive in float32") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("epsilon must remain finite and positive in float32")
    return value


def _validate_norm(m, n, gamma, cos, sin, epsilon):
    if m <= 0 or n <= 0 or n % HEAD_DIM:
        raise ValueError("projection must have nonempty [M,N] shape with whole 128-wide heads")
    if gamma.shape != (HEAD_DIM,) or gamma.dtype != mx.bfloat16:
        raise ValueError("gamma must be a shared bfloat16 vector of length 128")
    if len(cos.shape) != 2 or cos.shape[0] != m or not 1 <= cos.shape[1] <= HEAD_DIM // 2:
        raise ValueError("cos must have shape [M,rope_half] with rope_half in [1,64]")
    if sin.shape != cos.shape or cos.dtype != mx.bfloat16 or sin.dtype != mx.bfloat16:
        raise ValueError("cos and sin must have matching bfloat16 [M,rope_half] shapes")
    return cos.shape[1], _epsilon(epsilon)


def _validate_tile(tile):
    if not isinstance(tile, (tuple, list)) or len(tile) != 4:
        raise ValueError("tile must contain TM, TN, SIMD groups and M swizzle")
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in tile):
        raise ValueError("tile entries must be integers")
    tm, tn, groups, swizzle = map(int, tile)
    if tm not in (16, 32, 64, 128) or tn not in (64, 128) or groups not in (4, 8):
        raise NotImplementedError(
            "supported tiles have TM16/32/64/128, TN64/128 and 4/8 SIMD groups"
        )
    if swizzle <= 0:
        raise ValueError("M swizzle must be positive")
    return tm, tn, groups, swizzle


def _norm_body(read):
    source = (_SHADERS / "qk_rope_body.metal").read_text()
    return source.replace("$READ_PAIR", read.replace("dimension", "pair")).replace("$READ", read)


@lru_cache(None)
def _projection_kernel(mode, tm, tn, groups, swizzle, rope_half, epsilon):
    source = (_SHADERS / "qk_rope_projection.metal").read_text()
    replacements = {
        "$TM": str(tm),
        "$TN": str(tn),
        "$GROUPS": str(groups),
        "$SWIZZLE": str(swizzle),
        "$ROPE_HALF": str(rope_half),
        "$EPSILON": repr(epsilon) + "f",
        "$DTYPE": "int8_t" if mode == "int8" else "bfloat",
        "$ACC": "int" if mode == "int8" else "float",
        "$VALUE": "float(acc[i])*AS[row]*BS[col]" if mode == "int8" else "float(acc[i])",
        "$NORM_BODY": _norm_body("Projected[local_row*HEAD_DIM+dimension]"),
    }
    for token, value in replacements.items():
        source = source.replace(token, value)
    inputs = ["A", "B", "Gamma", "Cos", "Sin"]
    if mode == "int8":
        inputs += ["AS", "BS"]
    return mx.fast.metal_kernel(
        name=f"qk_rope_projection_{mode}_{tm}_{tn}_{groups}_{swizzle}_{rope_half}",
        input_names=inputs,
        output_names=["Y"],
        source=source,
        header=_HEADER,
    )


@lru_cache(None)
def _norm_kernel(groups, rope_half, epsilon):
    source = (_SHADERS / "qk_rope_norm.metal").read_text()
    for token, value in {
        "$GROUPS": str(groups),
        "$ROPE_HALF": str(rope_half),
        "$EPSILON": repr(epsilon) + "f",
        "$NORM_BODY": _norm_body("P[row*N+head*HEAD_DIM+dimension]"),
    }.items():
        source = source.replace(token, value)
    return mx.fast.metal_kernel(
        name=f"qk_rope_norm_{groups}_{rope_half}",
        input_names=["P", "Gamma", "Cos", "Sin"],
        output_names=["Y"],
        source=source,
    )


def projection_qk_rope(
    a,
    b,
    gamma,
    cos,
    sin,
    mode="int8",
    tile=(32, 128, 4, 1),
    *,
    a_scales=None,
    b_scales=None,
    epsilon=1e-5,
):
    """Return BF16 [heads,M,128] Q or K after projection, RMSNorm and RoPE.

    A[M,K] and B[N,K] are INT8 with FP32 per-row/per-output-channel scales,
    or BF16 when mode="bf16". Gamma is BF16[128], shared across heads.
    BF16 cos/sin[M,R] rotate pairs (d,d+R) for d<R; remaining dimensions
    are normalized but unrotated. H3 uses R=48. Input arrays are not mutated.

    One threadgroup owns a complete head. TN64 computes two sequential
    half-head matmuls; TN128 computes one. BF16 scratch is 2*TM*128 bytes.
    Finite operands and positive finite scales are caller preconditions,
    checked outside timing. Grouped QKV layouts and projection bias are
    deliberately absent from this single-projection API.
    """
    if mode not in ("int8", "bf16"):
        raise NotImplementedError("Q/K projection fusion supports INT8 or BF16 only")
    if len(a.shape) != 2 or len(b.shape) != 2 or not all(a.shape) or not all(b.shape):
        raise ValueError("A and B must be nonempty matrices")
    m, k = a.shape
    n, kb = b.shape
    if k != kb:
        raise ValueError("K mismatch")
    expected = mx.int8 if mode == "int8" else mx.bfloat16
    if a.dtype != expected or b.dtype != expected:
        raise ValueError(f"A and B must both have {mode} dtype")
    if mode == "int8":
        if k > 131071:
            raise NotImplementedError("INT32 worst-case overflow bound exceeded for signed INT8")
        if (
            a_scales is None
            or b_scales is None
            or a_scales.shape != (m,)
            or b_scales.shape != (n,)
            or a_scales.dtype != mx.float32
            or b_scales.dtype != mx.float32
        ):
            raise ValueError("INT8 requires float32 a_scales[M] and b_scales[N]")
    elif a_scales is not None or b_scales is not None:
        raise ValueError("BF16 projection does not accept quantization scales")
    rope_half, epsilon = _validate_norm(m, n, gamma, cos, sin, epsilon)
    tm, tn, groups, swizzle = _validate_tile(tile)
    inputs = [a, b, gamma, cos, sin]
    if mode == "int8":
        inputs += [a_scales, b_scales]
    return _projection_kernel(mode, tm, tn, groups, swizzle, rope_half, epsilon)(
        inputs=inputs,
        grid=(((m + tm - 1) // tm) * (n // HEAD_DIM) * groups * 32, 1, 1),
        threadgroup=(groups * 32, 1, 1),
        output_shapes=[(n // HEAD_DIM, m, HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )[0]


def qk_norm_rope(projected, gamma, cos, sin, *, epsilon=1e-5, threads=128):
    """Standalone native-style postprocess from BF16[M,N] to BF16[heads,M,128]."""
    if len(projected.shape) != 2 or projected.dtype != mx.bfloat16:
        raise ValueError("projected must be a bfloat16 [M,N] matrix")
    m, n = projected.shape
    rope_half, epsilon = _validate_norm(m, n, gamma, cos, sin, epsilon)
    if isinstance(threads, bool) or not isinstance(threads, Integral):
        raise ValueError("threads must be an integer")
    if threads not in (128, 256):
        raise NotImplementedError("standalone Q/K normalization supports 128 or 256 threads")
    groups = int(threads) // 32
    heads = n // HEAD_DIM
    return _norm_kernel(groups, rope_half, epsilon)(
        inputs=[projected, gamma, cos, sin],
        grid=(((heads * m + groups - 1) // groups) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(heads, m, HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )[0]

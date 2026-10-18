"""FP32 randomized block Hadamard rotations for activation/weight rows.

The caller supplies one shared vector of int8 signs, containing only +/-1.
The same right rotation X D H must be applied to activation and weight rows.
This preserves their product before quantization. Signs are checked for shape
and dtype, without a synchronous value check in the timed execution path.
"""

from functools import lru_cache
from numbers import Integral
from pathlib import Path

import mlx.core as mx


SUPPORTED_BLOCK_SIZES = (32, 64, 128, 256, 512, 1024)
# The shader uses one thread per element. No larger block is silently split or
# padded. Pipeline/device resource errors are propagated without a fallback.
MAX_THREADS_PER_THREADGROUP = 1024


@lru_cache(None)
def _kernel(block_size):
    source = (Path(__file__).parent / "shaders" / "hadamard.metal").read_text()
    return mx.fast.metal_kernel(
        name=f"signed_block_hadamard_{block_size}",
        input_names=["X", "Signs"],
        output_names=["Y"],
        source=source.replace("$BLOCK", str(block_size)),
    )


def rotate(x, block_size, signs, implementation="metal"):
    """Return FP32 X D H for nonempty X[rows,K], with no padding.

    H is a block-diagonal, normalized Sylvester Hadamard matrix. D is the
    diagonal matrix of the supplied +/-1 signs, applied BEFORE H. All rows use
    the same D. Inputs may be BF16, FP16 or FP32; FP32 butterfly accumulation
    is used by both the custom Metal and MLX implementations.

    Supported block sizes are powers of two from 32 through 1024, and K must
    be divisible by the block size. Input values must be finite and the FP32
    butterfly intermediates must remain representable. The harness checks
    finite outputs independently.
    """
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, Integral)
        or block_size <= 0
        or block_size & (block_size - 1)
    ):
        raise ValueError("block_size must be a positive integer power of two")
    block_size = int(block_size)
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise NotImplementedError("supported block sizes are powers of two from 32 through 1024")
    if len(x.shape) != 2 or not all(x.shape):
        raise ValueError("input must have nonempty shape [rows, K]")
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("input dtype must be float16, bfloat16 or float32")
    if x.shape[1] % block_size:
        raise NotImplementedError("K must be divisible by block_size; padding is not implicit")
    if signs.shape != (x.shape[1],) or signs.dtype != mx.int8:
        raise ValueError("signs must be an int8 vector of length K containing only +/-1")
    if implementation == "mlx":
        signed = x.astype(mx.float32) * signs
        return mx.hadamard_transform(signed.reshape(-1, block_size)).reshape(x.shape)
    if implementation != "metal":
        raise ValueError(f"unknown Hadamard implementation: {implementation}")
    if block_size > MAX_THREADS_PER_THREADGROUP:
        raise NotImplementedError("block_size exceeds the Metal threadgroup thread limit")
    return _kernel(block_size)(
        inputs=[x, signs],
        grid=(x.size, 1, 1),
        threadgroup=(block_size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )[0]

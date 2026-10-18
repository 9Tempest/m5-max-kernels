"""Independent native-policy grouped QKV references; no MLX or upstream code.

Native source provenance and precision differences are documented in
docs/grouped-qkv-native.md. Q/K normalization arithmetic is float64 between
explicit BF16 boundaries; it does not emulate a GPU RMS reduction order.
"""

from numbers import Integral, Real

import numpy as np

from .numerics import metrics
from .qk_rope_reference import bf16_round_reference, qk_rope_reference


def _finite_real(value, name):
    value = np.asarray(value)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def bf16_encode_reference(value):
    """Return uint16 BF16 payloads using FP32-producer nearest-even rounding."""
    rounded = np.asarray(bf16_round_reference(value), dtype=np.float32)
    return (rounded.view(np.uint32) >> 16).astype(np.uint16)


def bf16_decode_reference(bits):
    """Decode uint16 BF16 payloads to float64, including signed zeros/Inf/NaN.

    Nonfinite output payloads remain observable so callers can reject them in
    numerical gates. This representation helper does not imply finite inputs.
    """
    bits = np.asarray(bits)
    if bits.dtype != np.dtype(np.uint16):
        raise ValueError("BF16 payload must have uint16 dtype")
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def native_int8_quantize_reference(x, *, padded_rows=None, clip=1.0):
    """Reference the native BF16 row quantizer, including its padding policy.

    Input is converted to BF16 first. Native FP32 operations are explicit:
    peak=max(abs(row))*clip; scale=peak/127; inverse=127/peak;
    codes=clip(RNE(float32(row*inverse)),-127,127). A real all-zero row
    has scale=FP32(1/127), whereas extra padded rows have scale=1 and codes=0.
    No Hadamard transform or implicit column padding is performed.
    """
    x = _finite_real(x, "input")
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("input must have nonempty shape [M,K]")
    rows, width = x.shape
    if padded_rows is None:
        padded_rows = rows
    if isinstance(padded_rows, bool) or not isinstance(padded_rows, Integral) or padded_rows < rows:
        raise ValueError("padded_rows must be an integer at least M")
    if isinstance(clip, bool) or not isinstance(clip, Real) or not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be positive and finite")
    with np.errstate(over="ignore", under="ignore"):
        clip32 = np.float32(clip)
    if not np.isfinite(clip32) or clip32 <= 0:
        raise ValueError("clip must remain positive and finite in FP32")
    actual = bf16_round_reference(x)
    actual32 = actual.astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        peak = np.max(np.abs(actual32), axis=-1) * clip32
        nonzero = peak > 0
        denominator = np.where(nonzero, peak, np.float32(1))
        row_scales = denominator / np.float32(127)
        inverse = np.float32(127) / denominator
        scaled = actual32 * inverse[:, None]
    if (
        not np.isfinite(peak).all()
        or not np.isfinite(row_scales).all()
        or not np.all(row_scales > 0)
        or not np.isfinite(inverse).all()
        or not np.isfinite(scaled).all()
        or np.any(np.abs(scaled.astype(np.float64)) >= 2**31)
    ):
        raise ValueError("native quantization exceeds finite FP32 or INT32 conversion range")
    codes = np.zeros((padded_rows, width), dtype=np.int8)
    codes[:rows] = np.clip(np.rint(scaled), -127, 127).astype(np.int8)
    scales = np.ones(padded_rows, dtype=np.float32)
    scales[:rows] = row_scales
    decoded = codes.astype(np.float64) * scales[:, None]
    return dict(
        input_bf16=actual,
        codes=codes,
        scales=scales,
        decoded=decoded,
        quantization_error=metrics(decoded[:rows], actual),
        metadata=dict(
            policy="native_unrotated_symmetric_per_row_INT8",
            code_expression="RNE(float32(BF16_input * float32(127 / clipped_amax)))",
            scale_expression="float32(clipped_amax / 127)",
            clip=float(clip32),
            real_zero_row_scale=float(np.float32(1) / np.float32(127)),
            padded_row_scale=1.0,
            real_rows=rows,
            padded_rows=int(padded_rows),
            column_padding=0,
            code_range=[-127, 127],
            error_reference="actual rounded BF16 inputs; input conversion error excluded",
        ),
    )


def grouped_qkv_reference(
    a, weight, q_gamma, k_gamma, cos, sin, *, epsilon=1e-5, return_projection=False
):
    """Return original/decoded-operand grouped projection and native-style Q/K/V.

    A[M,K] and weight[H,3,128,K] are consumed as supplied in float64. Pass
    decoded actual BF16 operands for the original reference, or decoded INT8
    operands for the implementation reference. Weights are grouped by head,
    then Q/K/V stream, then channel; they are not three concatenated matrices.

    Both gamma vectors have shape [128]; cos/sin have shape [M,48]. Projection
    rounds FP32->BF16 before Q/K RMS/gamma/RoPE. Q/K have no intermediate norm
    BF16 round; V is only the BF16 projection. Every output is contiguous
    [H,M,128], with all values represented as float64 decoded numbers.

    return_projection=True instead returns a dict containing outputs (the
    same three-tuple), projection_float64 and projection_bf16[M,H,3,128].
    It supports full-K, complete-head sampling without changing head order.
    """
    if not isinstance(return_projection, (bool, np.bool_)):
        raise ValueError("return_projection must be boolean")
    a = _finite_real(a, "activation")
    weight = _finite_real(weight, "weight")
    if a.ndim != 2 or not all(a.shape):
        raise ValueError("activation must have nonempty shape [M,K]")
    if weight.ndim != 4 or not all(weight.shape) or weight.shape[1:3] != (3, 128):
        raise ValueError("weight must have nonempty shape [H,3,128,K]")
    if weight.shape[-1] != a.shape[-1]:
        raise ValueError("activation and weight K must match")
    cos = _finite_real(cos, "cos")
    sin = _finite_real(sin, "sin")
    if cos.shape != (a.shape[0], 48) or sin.shape != cos.shape:
        raise ValueError("native H3 cos and sin must have shape [M,48]")
    with np.errstate(over="ignore", invalid="ignore"):
        projection = (a @ weight.reshape(-1, a.shape[1]).T).reshape(a.shape[0], *weight.shape[:3])
    rounded = bf16_round_reference(projection)
    query = qk_rope_reference(rounded[:, :, 0, :], q_gamma, cos, sin, epsilon=epsilon)["output"]
    key = qk_rope_reference(rounded[:, :, 1, :], k_gamma, cos, sin, epsilon=epsilon)["output"]
    value = np.ascontiguousarray(rounded[:, :, 2, :].transpose(1, 0, 2))
    outputs = (query, key, value)
    if return_projection:
        return dict(outputs=outputs, projection_float64=projection, projection_bf16=rounded)
    return outputs

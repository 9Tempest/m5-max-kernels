"""Orthogonal input rotation before W8A8/FP8 GEMM; weight preparation is offline."""

import hashlib
import numpy as np

from .hadamard_reference import block_hadamard_reference, hadamard_signs
from .numerics import dequantize, metrics


def prepare(c, x, w, ri, ci, original, precision, tile):
    import mlx.core as mx
    from .hadamard import rotate
    from .metal import gemm, quantize

    if precision not in {"int8", "fp8"}:
        raise NotImplementedError("mpp-hadamard supports INT8 and FP8 with BF16 output")
    config = c.get("hadamard", {})
    block = config.get("block_size", 256)
    sign_seed = config.get("sign_seed")
    implementation = config.get("implementation", "metal")
    m, k = x.shape
    n = w.shape[0]
    signs_np = hadamard_signs(k, sign_seed) if sign_seed is not None else np.ones(k, np.int8)
    signs = mx.array(signs_np)
    mx.eval(signs)

    def rotation(z):
        return rotate(z, block, signs, implementation=implementation)

    # FP32 rotations feed the quantizer directly; no hidden BF16 intermediate.
    xr, wr = rotation(x), rotation(w)
    mx.eval(xr, wr)
    ref_a = block_hadamard_reference(
        np.asarray(x[mx.array(ri)].astype(mx.float32)), block, signs_np
    )
    ref_w = block_hadamard_reference(
        np.asarray(w[mx.array(ci)].astype(mx.float32)), block, signs_np
    )
    error_a = metrics(np.asarray(xr[mx.array(ri)]), ref_a)
    error_w = metrics(np.asarray(wr[mx.array(ci)]), ref_w)
    dot_error = metrics(ref_a @ ref_w.T, original)
    rotation_finite = bool(mx.all(mx.isfinite(xr)).item() and mx.all(mx.isfinite(wr)).item())
    transform_pass = (
        rotation_finite
        and all(e["finite"] and e["relative_l2"] <= 1e-5 for e in (error_a, error_w))
        and dot_error["finite"]
        and dot_error["relative_l2"] <= 1e-12
    )
    aq, sa = quantize(xr, precision, tag="hadamard_packed")
    bq, sb = quantize(wr, precision, tag="hadamard_packed")
    mx.eval(aq, sa, bq, sb)
    decoded_a = dequantize(np.asarray(aq[mx.array(ri)]), np.asarray(sa[mx.array(ri)]), precision)
    decoded_w = dequantize(np.asarray(bq[mx.array(ci)]), np.asarray(sb[mx.array(ci)]), precision)
    expected = decoded_a.astype(np.float64) @ decoded_w.astype(np.float64).T
    actual_a = np.asarray(xr[mx.array(ri)]).astype(np.float64)
    actual_w = np.asarray(wr[mx.array(ci)]).astype(np.float64)
    rotated_dot = actual_a @ actual_w.T

    def compute(q, scale):
        return gemm(q, bq, precision, tile, a_scales=scale, b_scales=sb, output_dtype=mx.bfloat16)

    def fn():
        return compute(aq, sa)

    def online_prepare():
        return quantize(rotation(x), precision, tag="hadamard_online")

    def online():
        return compute(*online_prepare())

    stages = {
        "activation_hadamard": lambda: rotation(x),
        "activation_quantization": lambda: quantize(xr, precision, tag="hadamard_stage"),
        "activation_hadamard_plus_quantization": online_prepare,
        # Preserve the harness's deployed-path key: it explicitly includes rotation here.
        "online_activation_plus_gemm": online,
    }

    def check(y):
        return np.asarray(y[mx.array(ri)[:, None], mx.array(ci)[None, :]].astype(mx.float32))

    meta = dict(
        compute_path="block_hadamard_then_tensorops_" + precision,
        input_dtype=str(x.dtype),
        rotated_dtype="mlx.core.float32",
        output_dtype="mlx.core.bfloat16",
        reference="NumPy float64 original/decoded dot products, sampled rows x columns, full K",
        sample_rows=ri.tolist(),
        sample_cols=ci.tolist(),
        quantization="symmetric_per_token_rotated_A_per_output_channel_rotated_W",
        weight_quantization_timed=False,
        weight_rotation_timed=False,
        weight_layout="N,K; identically right-rotated before quantization; transpose in matmul",
        mac_operations=2 * m * n * k,
        bytes_lower_bound=m * k + n * k + m * n * 2,
        bytes_lower_bound_scope="prequantized GEMM only; excludes rotation/quantization traffic",
        decoded_quantization_error=metrics(expected, rotated_dot),
        actual_rotation_dot_error=metrics(rotated_dot, original),
        transform_pass=transform_pass,
        transform=dict(
            kind="normalized_block_sylvester_hadamard",
            block_size=block,
            implementation=implementation,
            sign_seed=sign_seed,
            sign_rng="NumPy PCG64" if sign_seed is not None else "all +1",
            sign_sha256=hashlib.sha256(signs_np.tobytes()).hexdigest(),
            ordering="X D H; identical fixed D and H for activation and weight",
            padding=0,
            online_timing_includes="BF16 input to FP32 rotation, quantization, GEMM, BF16 output",
            activation_error=error_a,
            weight_error=error_w,
            float64_dot_preservation_error=dot_error,
            all_rotated_values_finite=rotation_finite,
            implementation_relative_l2_limit=1e-5,
            dot_preservation_relative_l2_limit=1e-12,
        ),
    )
    return fn, check, expected, original, stages, meta

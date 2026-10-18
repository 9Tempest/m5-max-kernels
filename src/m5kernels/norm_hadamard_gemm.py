"""Compare staged/fused AdaLN, block rotation and INT8 preparation before GEMM."""

import hashlib

import numpy as np

from .hadamard_reference import block_hadamard_reference, hadamard_signs
from .norm_hadamard_reference import norm_modulate_reference
from .numerics import metrics, sample_indices


def prepare(c, backend, precision, seed, distribution, tile):
    import mlx.core as mx

    from .hadamard import rotate
    from .metal import gemm, quantize
    from .norm_hadamard import fused_norm_hadamard_quantize, unfused_norm

    if (backend, precision) not in {
        ("norm-staged", "int8"),
        ("norm-fused", "int8"),
        ("mlx", "bf16"),
    }:
        raise NotImplementedError("use norm-staged/norm-fused INT8 or mlx BF16")
    m, n, k = c["m"], c["n"], c["k"]
    cfg = c.get("fusion", {})
    norm = cfg.get("norm", "rms")
    eps = cfg.get("epsilon", 1e-5)
    block = cfg.get("block_size", 256)
    threads = cfg.get("threads", 256)
    passes = cfg.get("passes", "one_read")
    rounding = cfg.get("round_output", "input")
    sign_seed = cfg.get("sign_seed")
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((m, k), dtype=np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal((n, k), dtype=np.float32)).astype(mx.bfloat16)
    if distribution == "zeros":
        x, w = mx.zeros_like(x), mx.zeros_like(w)
    elif distribution == "outliers":
        x = (x * mx.where(mx.arange(k) % 127 == 0, 20, 1)).astype(mx.bfloat16)
        w = (w * mx.where(mx.arange(k) % 131 == 0, 20, 1)).astype(mx.bfloat16)
    gamma = mx.array(1 + 0.1 * rng.standard_normal(k).astype(np.float32)).astype(mx.bfloat16)
    scale = mx.array(0.2 * rng.standard_normal(k).astype(np.float32)).astype(mx.bfloat16)
    shift = mx.array(0.1 * rng.standard_normal(k).astype(np.float32)).astype(mx.bfloat16)
    signs_np = hadamard_signs(k, sign_seed) if sign_seed is not None else np.ones(k, np.int8)
    signs = mx.array(signs_np)
    mx.eval(x, w, gamma, scale, shift, signs)
    ri, ci = sample_indices(m), sample_indices(n)

    def np32(a):
        return np.asarray(a.astype(mx.float32))

    ref_norm = norm_modulate_reference(
        np32(x[mx.array(ri)]), np32(gamma), np32(scale), np32(shift), norm=norm, eps=eps
    )
    # Explicit native-Metal boundary: completed AdaLN is rounded to BF16 once.
    rounded_ref = mx.array(ref_norm.astype(np.float32))
    if rounding == "input":
        rounded_ref = rounded_ref.astype(mx.bfloat16)
    rounded_ref = np32(rounded_ref).astype(np.float64)
    ref_w = np32(w[mx.array(ci)]).astype(np.float64)
    original = rounded_ref @ ref_w.T

    def norm_only():
        return unfused_norm(
            x, gamma, scale, shift, norm=norm, epsilon=eps, round_output=rounding, threads=threads
        )

    def check(y):
        return np32(y[mx.array(ri)[:, None], mx.array(ci)[None, :]])

    meta = dict(
        compute_path="metal_norm_mlx_bf16_gemm" if backend == "mlx" else backend,
        input_dtype="mlx.core.bfloat16",
        output_dtype="mlx.core.bfloat16",
        mac_operations=2 * m * n * k,
        reference="NumPy float64 normalization/rotation/dots; sampled rows/columns, full K",
        sample_rows=ri.tolist(),
        sample_cols=ci.tolist(),
        norm=norm,
        epsilon=eps,
        modulation="per-channel gamma, scale and shift shared by all rows",
        normalization_rounding=rounding,
        rounding_contract=(
            "FP32 norm and modulation, then one input-dtype round before rotation"
            if rounding == "input"
            else "FP32 norm, modulation and rotation; no BF16 intermediate"
        ),
        ideal_normalization_rounding_error=metrics(rounded_ref, ref_norm),
        weight_rotation_timed=False,
        weight_quantization_timed=False,
    )
    if backend == "mlx":
        normalized = norm_only()
        mx.eval(normalized)

        def fn():
            return (normalized @ w.T).astype(mx.bfloat16)

        def online():
            return (norm_only() @ w.T).astype(mx.bfloat16)

        return (
            fn,
            check,
            original,
            original,
            {"activation_prepare": norm_only, "online_activation_plus_gemm": online},
            meta,
        )

    def staged():
        return quantize(rotate(norm_only(), block, signs), "int8", tag="norm_staged")

    def fused():
        return fused_norm_hadamard_quantize(
            x,
            gamma,
            scale,
            shift,
            signs,
            block_size=block,
            norm=norm,
            epsilon=eps,
            round_output=rounding,
            threads=threads,
            passes=passes,
        )

    activation = fused if backend == "norm-fused" else staged
    aq, sa = activation()
    rotated_w = rotate(w, block, signs)
    bq, sb = quantize(rotated_w, "int8", tag="norm_offline_weight")
    mx.eval(aq, sa, bq, sb, rotated_w)
    actual_a = (
        np.asarray(aq[mx.array(ri)]).astype(np.float64) * np.asarray(sa[mx.array(ri)])[:, None]
    )
    actual_w = (
        np.asarray(bq[mx.array(ci)]).astype(np.float64) * np.asarray(sb[mx.array(ci)])[:, None]
    )
    ref_rotated = block_hadamard_reference(rounded_ref, block, signs_np)
    ref_rotated_w = block_hadamard_reference(ref_w, block, signs_np)
    ref_scale = np.max(np.abs(ref_rotated), axis=-1) / 127
    ref_scale = np.where(ref_scale == 0, 1, ref_scale)
    ref_q = np.clip(np.rint(ref_rotated / ref_scale[:, None]), -127, 127)
    ref_decoded = ref_q * ref_scale[:, None]
    ref_w_scale = np.max(np.abs(ref_rotated_w), axis=-1) / 127
    ref_w_scale = np.where(ref_w_scale == 0, 1, ref_w_scale)
    ref_w_q = np.clip(np.rint(ref_rotated_w / ref_w_scale[:, None]), -127, 127)
    ref_w_decoded = ref_w_q * ref_w_scale[:, None]
    weight_rotation_error = metrics(np32(rotated_w[mx.array(ci)]), ref_rotated_w)
    weight_error = metrics(actual_w, ref_w_decoded)
    weight_scale_error = metrics(np.asarray(sb[mx.array(ci)]), ref_w_scale)
    weight_max_code_error = float(
        np.max(np.abs(np.asarray(bq[mx.array(ci)]).astype(float) - ref_w_q))
    )
    activation_error = metrics(actual_a, ref_decoded)
    scale_error = metrics(np.asarray(sa[mx.array(ri)]), ref_scale)
    dot_error = metrics(ref_rotated @ ref_rotated_w.T, original)
    max_code_error = float(np.max(np.abs(np.asarray(aq[mx.array(ri)]).astype(float) - ref_q)))
    finite = bool(
        mx.all(mx.isfinite(sa)).item()
        and mx.all(sa > 0).item()
        and mx.all(mx.isfinite(sb)).item()
        and mx.all(sb > 0).item()
        and mx.all(mx.isfinite(rotated_w)).item()
    )
    expected = actual_a @ actual_w.T
    meta.update(
        quantization="symmetric_per_token_rotated_A_per_output_channel_rotated_W",
        transform_pass=(
            finite
            and activation_error["finite"]
            and activation_error["relative_l2"] <= 0.008
            and scale_error["finite"]
            and scale_error["relative_l2"] <= 0.008
            and max_code_error <= 1
            and weight_rotation_error["finite"]
            and weight_rotation_error["relative_l2"] <= 1e-5
            and weight_error["finite"]
            and weight_error["relative_l2"] <= 0.008
            and weight_scale_error["finite"]
            and weight_scale_error["relative_l2"] <= 1e-5
            and weight_max_code_error <= 1
            and dot_error["finite"]
            and dot_error["relative_l2"] <= 1e-12
        ),
        activation_implementation_error=activation_error,
        activation_scale_error=scale_error,
        activation_max_code_difference=max_code_error,
        activation_implementation_relative_l2_limit=0.008,
        activation_max_code_difference_limit=1,
        weight_rotation_error=weight_rotation_error,
        weight_implementation_error=weight_error,
        weight_scale_error=weight_scale_error,
        weight_max_code_difference=weight_max_code_error,
        activation_preparation_total_error=metrics(actual_a, ref_rotated),
        decoded_preparation_total_error=metrics(expected, original),
        ideal_activation_quantization_error=metrics(ref_decoded, ref_rotated),
        ideal_decoded_quantization_error=metrics(ref_decoded @ ref_w_decoded.T, original),
        transform=dict(
            kind="normalized_block_sylvester_hadamard",
            block_size=block,
            implementation=backend,
            sign_seed=sign_seed,
            sign_sha256=hashlib.sha256(signs_np.tobytes()).hexdigest(),
            float64_dot_preservation_error=dot_error,
            threads=threads,
            passes=passes,
        ),
    )

    def compute(q, s):
        return gemm(q, bq, "int8", tile, a_scales=s, b_scales=sb, output_dtype=mx.bfloat16)

    def fn():
        return compute(aq, sa)

    def online():
        return compute(*activation())

    stages = {"activation_prepare": activation, "online_activation_plus_gemm": online}
    return fn, check, expected, original, stages, meta

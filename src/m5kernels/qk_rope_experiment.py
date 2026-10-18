"""Single Q or K projection with matched staged/fused native-style postprocessing."""

import hashlib
from numbers import Integral, Real

import numpy as np

from .hadamard_reference import block_hadamard_reference
from .numerics import metrics, sample_indices
from .qk_rope_reference import bf16_round_reference, qk_rope_reference


def validate_config(c, backend, precision, distribution, tile):
    """Reject unsupported semantics before importing MLX or allocating GPU arrays."""
    if backend not in {"qk-staged", "qk-fused"} or precision != "int8":
        raise NotImplementedError("qk_rope supports qk-staged/qk-fused with INT8 only")
    if c.get("op") != "qk_rope":
        raise ValueError("operation must be qk_rope")
    for key in ("m", "n", "k"):
        if isinstance(c.get(key), bool) or not isinstance(c.get(key), Integral) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    head_dim = c.get("head_dim", 128)
    if (
        isinstance(head_dim, bool)
        or not isinstance(head_dim, Integral)
        or head_dim != 128
        or c["n"] % 128
    ):
        raise NotImplementedError("Q/K projection requires complete 128-wide heads")
    if c["k"] % 256:
        raise NotImplementedError("K must be divisible by H256; no implicit padding")
    if c["k"] > 131071:
        raise NotImplementedError("INT32 worst-case overflow bound exceeded")
    half = c.get("rope_half", 48)
    if isinstance(half, bool) or not isinstance(half, Integral) or not 1 <= half <= 64:
        raise ValueError("rope_half must be an integer in [1,64]")
    epsilon = c.get("epsilon", 1e-5)
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be positive and finite")
    with np.errstate(over="ignore", under="ignore"):
        epsilon = np.float32(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must remain positive and finite in FP32")
    if distribution not in {"normal", "outliers", "zeros"}:
        raise ValueError("distribution must be normal, outliers or zeros")
    if not isinstance(tile, (tuple, list)) or len(tile) != 4:
        raise ValueError("tile must contain TM,TN,SIMD groups,M swizzle")
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in tile):
        raise ValueError("tile entries must be integers")
    tm, tn, groups, swizzle = tile
    if tm not in (16, 32, 64, 128) or tn not in (64, 128) or groups not in (4, 8):
        raise NotImplementedError("Q/K tile supports TM16/32/64/128, TN64/128, groups4/8")
    if swizzle <= 0:
        raise ValueError("tile swizzle must be positive")
    return dict(head_dim=128, rope_half=int(half), epsilon=float(epsilon), block_size=256)


def sample_complete_heads(n, head_dim=128, count=2):
    """Return sampled head ids and every corresponding projection column."""
    if n <= 0 or head_dim <= 0 or n % head_dim or count <= 0:
        raise ValueError("sampling requires positive whole heads and positive count")
    heads = sample_indices(n // head_dim, count)
    columns = (heads[:, None] * head_dim + np.arange(head_dim)[None, :]).reshape(-1)
    return heads, columns


def _quantize64(rotated):
    peak = np.max(np.abs(rotated), axis=-1)
    scales = np.where(peak == 0, 1.0, peak / 127.0)
    codes = np.clip(np.rint(rotated / scales[:, None]), -127, 127)
    return codes, scales, codes * scales[:, None]


def prepare(c, backend, precision, seed, distribution, tile):
    cfg = validate_config(c, backend, precision, distribution, tile)
    import mlx.core as mx

    from .engine import all_finite
    from .hadamard import rotate
    from .metal import gemm, quantize
    from .qk_rope import projection_qk_rope, qk_norm_rope

    m, n, k = c["m"], c["n"], c["k"]
    dim, half, eps, block = (cfg[key] for key in ("head_dim", "rope_half", "epsilon", "block_size"))
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((m, k), dtype=np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal((n, k), dtype=np.float32) * k**-0.5).astype(mx.bfloat16)
    if distribution == "zeros":
        x, w = mx.zeros_like(x), mx.zeros_like(w)
    elif distribution == "outliers":
        x = (x * mx.where(mx.arange(k) % 127 == 0, 20, 1)).astype(mx.bfloat16)
        w = (w * mx.where(mx.arange(k) % 131 == 0, 20, 1)).astype(mx.bfloat16)
    gamma = mx.array(1 + 0.2 * rng.standard_normal(dim, dtype=np.float32)).astype(mx.bfloat16)
    angles = rng.uniform(-np.pi, np.pi, size=(m, half)).astype(np.float32)
    cos = mx.array(np.cos(angles)).astype(mx.bfloat16)
    sin = mx.array(np.sin(angles)).astype(mx.bfloat16)
    signs = mx.ones(k, dtype=mx.int8)
    mx.eval(x, w, gamma, cos, sin, signs)
    rows = sample_indices(m)
    heads, columns = sample_complete_heads(n, dim)
    rows_gpu, heads_gpu, cols_gpu = map(mx.array, (rows, heads, columns))

    def np32(value):
        return np.asarray(value.astype(mx.float32))

    def extract(output):
        # Head-major selection retains each sampled head's complete D axis.
        return np32(output[heads_gpu[:, None], rows_gpu[None, :], :]).reshape(-1)

    def activation_prepare():
        return quantize(rotate(x, block, signs), "int8", tag="qk_online")

    def offline_weight_prepare():
        return quantize(rotate(w, block, signs), "int8", tag="qk_weight")

    rotated_a, rotated_w = rotate(x, block, signs), rotate(w, block, signs)
    aq, sa = quantize(rotated_a, "int8", tag="qk_online")
    bq, sb = quantize(rotated_w, "int8", tag="qk_weight")
    mx.eval(rotated_a, rotated_w, aq, sa, bq, sb)

    def projection(q, scales):
        return gemm(q, bq, "int8", tile, a_scales=scales, b_scales=sb, output_dtype=mx.bfloat16)

    def postprocess(projected):
        return qk_norm_rope(projected, gamma, cos, sin, epsilon=eps, threads=tile[2] * 32)

    def staged(q, scales):
        return postprocess(projection(q, scales))

    def fused(q, scales):
        return projection_qk_rope(
            q, bq, gamma, cos, sin, "int8", tile, a_scales=scales, b_scales=sb, epsilon=eps
        )

    selected = fused if backend == "qk-fused" else staged
    projected = projection(aq, sa)
    staged_output = postprocess(projected)
    fused_output = fused(aq, sa)
    mx.eval(projected, staged_output, fused_output)
    parity_error = metrics(extract(fused_output), extract(staged_output))
    parity_exact = bool(mx.all(fused_output == staged_output).item())

    ref_x = np32(x[rows_gpu]).astype(np.float64)
    ref_w = np32(w[cols_gpu]).astype(np.float64)
    ref_gamma, ref_cos, ref_sin = np32(gamma), np32(cos[rows_gpu]), np32(sin[rows_gpu])

    def reference(dot):
        return qk_rope_reference(
            dot.reshape(len(rows), len(heads), dim), ref_gamma, ref_cos, ref_sin, epsilon=eps
        )["output"].reshape(-1)

    original_dot = ref_x @ ref_w.T
    original = reference(original_dot)
    a_codes, a_scales = np.asarray(aq[rows_gpu]).astype(np.float64), np.asarray(sa[rows_gpu])
    w_codes, w_scales = np.asarray(bq[cols_gpu]).astype(np.float64), np.asarray(sb[cols_gpu])
    decoded_a, decoded_w = a_codes * a_scales[:, None], w_codes * w_scales[:, None]
    decoded_dot = decoded_a @ decoded_w.T
    expected = reference(decoded_dot)

    ref_rotated_a = block_hadamard_reference(ref_x, block)
    ref_rotated_w = block_hadamard_reference(ref_w, block)
    ref_a_codes, ref_a_scales, ideal_a = _quantize64(ref_rotated_a)
    ref_w_codes, ref_w_scales, ideal_w = _quantize64(ref_rotated_w)
    rotation_a_error = metrics(np.asarray(rotated_a[rows_gpu]), ref_rotated_a)
    rotation_w_error = metrics(np.asarray(rotated_w[cols_gpu]), ref_rotated_w)
    a_scale_error, w_scale_error = metrics(a_scales, ref_a_scales), metrics(w_scales, ref_w_scales)
    a_code_diff = float(np.max(np.abs(a_codes - ref_a_codes)))
    w_code_diff = float(np.max(np.abs(w_codes - ref_w_codes)))
    a_prep_error, w_prep_error = metrics(decoded_a, ideal_a), metrics(decoded_w, ideal_w)
    dot_preservation = metrics(ref_rotated_a @ ref_rotated_w.T, original_dot)
    sampled_projection = np32(projected[rows_gpu[:, None], cols_gpu[None, :]])
    projection_error = metrics(sampled_projection, bf16_round_reference(decoded_dot))
    postprocess_error = metrics(extract(staged_output), reference(sampled_projection))
    finite = all_finite(
        (
            x,
            w,
            gamma,
            cos,
            sin,
            rotated_a,
            rotated_w,
            aq,
            sa,
            bq,
            sb,
            projected,
            staged_output,
            fused_output,
        )
    )
    positive_scales = bool(mx.all(sa > 0).item() and mx.all(sb > 0).item())

    def pass_error(error, limit):
        return error["finite"] and error["relative_l2"] <= limit

    transform_pass = bool(
        finite
        and positive_scales
        # Both INT8 paths have the same exact integer dot, FP32 dequantization
        # order and postprocessor body. Enforce the full comparison, including
        # heads and rows outside the bounded independent-reference sample.
        and parity_exact
        and all(
            pass_error(e, 1e-5)
            for e in (rotation_a_error, rotation_w_error, a_scale_error, w_scale_error)
        )
        and pass_error(dot_preservation, 1e-12)
        and all(
            pass_error(e, 0.008)
            for e in (a_prep_error, w_prep_error, parity_error, projection_error, postprocess_error)
        )
        and a_code_diff <= 1
        and w_code_diff <= 1
    )

    def fn():
        return selected(aq, sa)

    def online():
        return selected(*activation_prepare())

    meta = dict(
        compute_path=backend,
        operation_scope="one Q or K projection, not grouped QKV or attention",
        input_dtype="bf16",
        output_dtype="bf16",
        output_layout="head-major [H,M,128], contiguous",
        weight_layout="[H*128,K], ordinary single-projection order",
        reference="NumPy float64 dots with full K and every channel of sampled complete heads; explicit BF16 projection/output",
        sample_rows=rows.tolist(),
        sample_heads=heads.tolist(),
        sample_cols=columns.tolist(),
        head_dim=dim,
        rope_half=half,
        epsilon=eps,
        projection_rounding="BF16 after FP32 INT8 dequantization, before RMS",
        normalization_rounding="none; FP32 RMS/gamma/RoPE before final BF16",
        rms_scope="all128 channels per row/head; before gamma",
        rope_pairing="split-half (j,j+rope_half); remaining channels normalized but unrotated",
        rope_table_origin="synthetic independent random angles; not actual H3 position generation",
        rope_table_dtype="bf16",
        gamma_dtype="bf16 shared[128]",
        gamma_sha256=hashlib.sha256(ref_gamma.tobytes()).hexdigest(),
        main_timing_scope="resident INT8 projection plus RMS/gamma/RoPE and head-major output",
        online_timing_scope="BF16 activation H256 + INT8 packing + projection + RMS/gamma/RoPE + head-major BF16 output",
        weight_preparation_timed=True,
        weight_preparation_in_online=False,
        offline_weight_preparation="BF16 W to FP32 H256 then per-output-channel INT8 packing",
        diagnostic_stage_scope="projection and postprocess are isolated staged components, not separate fused costs",
        mac_operations=2 * m * n * k,
        all_preparation_values_finite=finite,
        positive_quantization_scales=positive_scales,
        paired_path_error=parity_error,
        paired_path_exact=parity_exact,
        paired_path_exact_required=True,
        projection_implementation_error=projection_error,
        postprocess_implementation_error=postprocess_error,
        activation_rotation_error=rotation_a_error,
        weight_rotation_error=rotation_w_error,
        activation_scale_error=a_scale_error,
        weight_scale_error=w_scale_error,
        activation_max_code_difference=a_code_diff,
        weight_max_code_difference=w_code_diff,
        activation_implementation_error=a_prep_error,
        weight_implementation_error=w_prep_error,
        ideal_activation_quantization_error=metrics(ideal_a, ref_rotated_a),
        ideal_weight_quantization_error=metrics(ideal_w, ref_rotated_w),
        ideal_decoded_quantization_error=metrics(reference(ideal_a @ ideal_w.T), original),
        decoded_preparation_total_error=metrics(expected, original),
        transform=dict(
            kind="normalized_block_sylvester_hadamard",
            block_size=block,
            sign_policy="all +1; same activation and weight transform",
            padding=0,
            float64_dot_preservation_error=dot_preservation,
        ),
        transform_pass=transform_pass,
    )
    stages = dict(
        activation_hadamard=lambda: rotate(x, block, signs),
        activation_quantization=lambda: quantize(rotated_a, "int8", tag="qk_online"),
        activation_prepare=activation_prepare,
        projection=lambda: projection(aq, sa),
        postprocess=lambda: postprocess(projected),
        online_activation_plus_gemm=online,
        offline_weight_prepare=offline_weight_prepare,
    )
    return fn, extract, expected, original, stages, meta

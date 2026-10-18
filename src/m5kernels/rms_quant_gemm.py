"""Reference-checked RMS scale-folding preparation followed by INT8 GEMM."""

import hashlib
from numbers import Integral, Real

import numpy as np

from .hadamard_reference import block_hadamard_reference, hadamard_signs
from .numerics import metrics, sample_indices
from .rms_quant_reference import rms_quantize_reference


def validate_config(c, backend, precision, distribution):
    """Validate semantics before importing MLX, including on CPU CI hosts."""
    if (backend, precision) not in {("rms-direct", "int8"), ("rms-folded", "int8")}:
        raise NotImplementedError("rms_quant_gemm supports rms-direct/rms-folded with INT8 only")
    if c.get("op") != "rms_quant_gemm":
        raise ValueError("operation must be rms_quant_gemm")
    forbidden = {"shift", "norm", "round_output"}.intersection(c)
    if forbidden:
        raise NotImplementedError(f"unsupported RMS semantics options: {sorted(forbidden)}")
    for key in ("m", "n", "k"):
        if isinstance(c.get(key), bool) or not isinstance(c.get(key), Integral) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    cfg = c.get("rms_quant", {})
    if not isinstance(cfg, dict):
        raise ValueError("rms_quant must be a configuration object")
    allowed = {"block_size", "gamma", "threads", "epsilon", "sign_seed"}
    unknown = set(cfg) - allowed
    if unknown:
        raise NotImplementedError(
            f"unsupported rms_quant options: {sorted(unknown)}; "
            "this experiment requires RMS, zero shift and no BF16 intermediate"
        )
    config = dict(block_size=1, gamma="identity", threads=256, epsilon=1e-5, sign_seed=None)
    config.update(cfg)
    block = config["block_size"]
    if isinstance(block, bool) or not isinstance(block, Integral) or block <= 0:
        raise ValueError("block_size must be a positive integer")
    if block not in (1, 256):
        raise NotImplementedError("rms_quant_gemm supports block_size 1 or 256")
    if c["k"] % block:
        raise NotImplementedError("K must be divisible by block_size; padding is not implicit")
    if config["gamma"] not in ("identity", "random"):
        raise NotImplementedError("gamma must be identity or random")
    threads = config["threads"]
    if isinstance(threads, bool) or not isinstance(threads, Integral):
        raise ValueError("threads must be an integer")
    if threads not in (128, 256, 512):
        raise NotImplementedError("supported threadgroup sizes are 128, 256 and 512")
    eps = config["epsilon"]
    if isinstance(eps, bool) or not isinstance(eps, Real) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be positive and finite")
    with np.errstate(over="ignore", under="ignore"):
        eps32 = np.float32(eps)
    if not np.isfinite(eps32) or eps32 <= 0:
        raise ValueError("epsilon must remain positive and finite in float32")
    sign_seed = config["sign_seed"]
    if sign_seed is not None and (
        isinstance(sign_seed, bool) or not isinstance(sign_seed, Integral) or sign_seed < 0
    ):
        raise ValueError("sign_seed must be a nonnegative integer or null")
    if distribution not in ("normal", "outliers", "zeros"):
        raise ValueError("distribution must be normal, outliers or zeros")
    return config


def _bf16_round(values):
    """CPU nearest-even BF16 rounding, used only for a contract diagnostic."""
    rounded = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (rounded + np.uint32(0x7FFF) + ((rounded >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).astype(np.float64)


def prepare(c, backend, precision, seed, distribution, tile):
    cfg = validate_config(c, backend, precision, distribution)
    import mlx.core as mx

    from .hadamard import rotate
    from .metal import gemm, quantize
    from .rms_quant import rms_quantize

    m, n, k = c["m"], c["n"], c["k"]
    block, threads, eps = cfg["block_size"], cfg["threads"], cfg["epsilon"]
    sign_seed = cfg["sign_seed"]
    fold = backend == "rms-folded"
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((m, k), dtype=np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal((n, k), dtype=np.float32)).astype(mx.bfloat16)
    if distribution == "zeros":
        x, w = mx.zeros_like(x), mx.zeros_like(w)
    elif distribution == "outliers":
        x = (x * mx.where(mx.arange(k) % 127 == 0, 20, 1)).astype(mx.bfloat16)
        w = (w * mx.where(mx.arange(k) % 131 == 0, 20, 1)).astype(mx.bfloat16)
    gamma_np = (
        1 + 0.2 * rng.standard_normal(k).astype(np.float32)
        if cfg["gamma"] == "random"
        else np.ones(k, np.float32)
    )
    gamma = mx.array(gamma_np)
    signs_np = hadamard_signs(k, sign_seed) if sign_seed is not None else np.ones(k, np.int8)
    signs = mx.array(signs_np)
    mx.eval(x, w, gamma, signs)
    ri, ci = sample_indices(m), sample_indices(n)
    ri_gpu, ci_gpu = mx.array(ri), mx.array(ci)

    def np32(a):
        return np.asarray(a.astype(mx.float32))

    ref_x = np32(x[ri_gpu]).astype(np.float64)
    ref_w = np32(w[ci_gpu]).astype(np.float64)
    ref_a = rms_quantize_reference(ref_x, gamma_np, signs_np, block, eps=eps)
    ref_norm = ref_x * ref_a["inverse_rms"][:, None] * gamma_np.astype(np.float64)
    original = ref_norm @ ref_w.T
    ref_rotated_w = block_hadamard_reference(ref_w, block, signs_np)
    ref_w_scale = np.max(np.abs(ref_rotated_w), axis=-1) / 127
    ref_w_scale = np.where(ref_w_scale == 0, 1, ref_w_scale)
    ref_w_q = np.clip(np.rint(ref_rotated_w / ref_w_scale[:, None]), -127, 127)
    ref_w_decoded = ref_w_q * ref_w_scale[:, None]

    def weight_transform():
        if block == 1:
            dense = w.astype(mx.float32)
            return dense * signs if sign_seed is not None else dense
        return rotate(w, block, signs)

    def offline_weight_prepare():
        return quantize(weight_transform(), "int8", tag="rms_weight_stage")

    def activation(tag="rms_online"):
        return rms_quantize(
            x,
            gamma,
            signs,
            block_size=block,
            threads=threads,
            fold_scale=fold,
            epsilon=eps,
            tag=tag,
        )

    aq, sa = activation("rms_packed")
    rotated_w = weight_transform()
    bq, sb = quantize(rotated_w, "int8", tag="rms_weight_packed")
    mx.eval(aq, sa, bq, sb, rotated_w)
    aq_cpu, sa_cpu = np.asarray(aq[ri_gpu]), np.asarray(sa[ri_gpu])
    bq_cpu, sb_cpu = np.asarray(bq[ci_gpu]), np.asarray(sb[ci_gpu])
    actual_a = aq_cpu.astype(np.float64) * sa_cpu[:, None]
    actual_w = bq_cpu.astype(np.float64) * sb_cpu[:, None]
    expected = actual_a @ actual_w.T

    activation_error = metrics(actual_a, ref_a["dequantized"])
    scale_error = metrics(sa_cpu, ref_a["scales"])
    max_code_error = float(
        np.max(np.abs(aq_cpu.astype(np.int16) - ref_a["quantized"].astype(np.int16)))
    )
    weight_rotation_error = metrics(np32(rotated_w[ci_gpu]), ref_rotated_w)
    weight_error = metrics(actual_w, ref_w_decoded)
    weight_scale_error = metrics(sb_cpu, ref_w_scale)
    weight_max_code_error = float(np.max(np.abs(bq_cpu.astype(np.float64) - ref_w_q)))
    dot_error = metrics(ref_a["normalized_rotated"] @ ref_rotated_w.T, original)
    finite = bool(
        mx.all(mx.isfinite(sa)).item()
        and mx.all(sa > 0).item()
        and mx.all(mx.isfinite(sb)).item()
        and mx.all(sb > 0).item()
        and mx.all(mx.isfinite(rotated_w)).item()
        and mx.all(aq >= -127).item()
        and mx.all(bq >= -127).item()
    )
    transform_pass = (
        finite
        and activation_error["finite"]
        and activation_error["relative_l2"] <= 0.008
        and scale_error["finite"]
        and scale_error["relative_l2"] <= 1e-5
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
    )
    rounded_norm = _bf16_round(ref_norm)
    rounded_rotated = block_hadamard_reference(rounded_norm, block, signs_np)

    def compute(q, s):
        return gemm(q, bq, "int8", tile, a_scales=s, b_scales=sb, output_dtype=mx.bfloat16)

    def fn():
        return compute(aq, sa)

    def online():
        return compute(*activation())

    def check(y):
        return np32(y[ri_gpu[:, None], ci_gpu[None, :]])

    meta = dict(
        compute_path=backend,
        input_dtype="mlx.core.bfloat16",
        output_dtype="mlx.core.bfloat16",
        preparation_dtype="mlx.core.float32",
        mac_operations=2 * m * n * k,
        reference="NumPy float64 RMS/gamma/rotation/dots; sampled rows/columns, full K",
        sample_rows=ri.tolist(),
        sample_cols=ci.tolist(),
        norm="rms",
        epsilon=eps,
        gamma=cfg["gamma"],
        gamma_sha256=hashlib.sha256(gamma_np.tobytes()).hexdigest(),
        fold_scale=fold,
        additive_shift=False,
        normalization_rounding="fp32",
        rounding_contract="FP32 RMS and gamma/rotation; no BF16 intermediate; BF16 GEMM output",
        quantization="symmetric_per_token_rotated_A_per_output_channel_rotated_W",
        weight_preparation_in_online=False,
        weight_preparation_timed=True,
        weight_preparation_timing_stage="offline_weight_prepare",
        weight_preparation_timing_scope="combined offline weight transform and INT8 packing only",
        all_preparation_values_finite=finite,
        transform_pass=bool(transform_pass),
        activation_implementation_error=activation_error,
        activation_scale_error=scale_error,
        activation_max_code_difference=max_code_error,
        activation_implementation_relative_l2_limit=0.008,
        activation_scale_relative_l2_limit=1e-5,
        activation_max_code_difference_limit=1,
        weight_rotation_error=weight_rotation_error,
        weight_implementation_error=weight_error,
        weight_scale_error=weight_scale_error,
        weight_max_code_difference=weight_max_code_error,
        activation_preparation_total_error=metrics(actual_a, ref_a["normalized_rotated"]),
        ideal_activation_quantization_error=metrics(
            ref_a["dequantized"], ref_a["normalized_rotated"]
        ),
        decoded_preparation_total_error=metrics(expected, original),
        ideal_decoded_quantization_error=metrics(ref_a["dequantized"] @ ref_w_decoded.T, original),
        bf16_intermediate_diagnostic=dict(
            diagnostic_only=True,
            contract="hypothetical BF16 round after RMS*gamma, before D*H; not executed",
            normalization_error=metrics(rounded_norm, ref_norm),
            rotated_error=metrics(rounded_rotated, ref_a["normalized_rotated"]),
            gemm_difference=metrics(rounded_norm @ ref_w.T, original),
        ),
        transform=dict(
            kind="signed_identity" if block == 1 else "normalized_block_sylvester_hadamard",
            block_size=block,
            implementation=backend,
            sign_seed=sign_seed,
            sign_rng="NumPy PCG64" if sign_seed is not None else "all +1",
            sign_sha256=hashlib.sha256(signs_np.tobytes()).hexdigest(),
            ordering="RMS(X)*gamma*D*H, W*D*H; identical D and H, zero additive shift",
            padding=0,
            threads=threads,
            float64_dot_preservation_error=dot_error,
        ),
    )
    stages = {
        "activation_prepare": activation,
        "online_activation_plus_gemm": online,
        "offline_weight_prepare": offline_weight_prepare,
    }
    return fn, check, expected, original, stages, meta

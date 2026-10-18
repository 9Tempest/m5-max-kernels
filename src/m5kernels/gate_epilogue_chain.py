"""Compare gate placement across two projections with identical BF16 boundaries."""

from numbers import Integral

import numpy as np

from .hadamard_reference import block_hadamard_reference
from .norm_hadamard_reference import norm_modulate_reference
from .numerics import metrics, sample_indices
from .rms_quant_gemm import _bf16_round


def _preparation_check(original, rotated, codes, scales, block, rows):
    """Independent sampled-row H/INT8 reference, with full contraction width."""
    reference = block_hadamard_reference(original, block)
    reference_scale = np.max(np.abs(reference), axis=-1) / 127
    reference_scale = np.where(reference_scale == 0, 1.0, reference_scale)
    reference_codes = np.clip(np.rint(reference / reference_scale[:, None]), -127, 127)
    reference_decoded = reference_codes * reference_scale[:, None]
    decoded = np.asarray(codes, dtype=np.float64) * np.asarray(scales)[:, None]
    rotation_error = metrics(rotated, reference)
    scale_error = metrics(scales, reference_scale)
    implementation_error = metrics(decoded, reference_decoded)
    code_difference = float(np.max(np.abs(np.asarray(codes, dtype=np.float64) - reference_codes)))
    return dict(
        sample_rows=rows.tolist(),
        reference="independent NumPy float64 H256 and symmetric INT8, sampled rows with full K",
        rotation_error=rotation_error,
        scale_error=scale_error,
        decoded_implementation_error=implementation_error,
        max_code_difference=code_difference,
        ideal_quantization_error=metrics(reference_decoded, reference),
        actual_quantization_total_error=metrics(decoded, reference),
        rotation_relative_l2_limit=1e-5,
        scale_relative_l2_limit=1e-5,
        decoded_implementation_relative_l2_limit=0.008,
        max_code_difference_limit=1,
        passed=bool(
            rotation_error["finite"]
            and rotation_error["relative_l2"] <= 1e-5
            and scale_error["finite"]
            and scale_error["relative_l2"] <= 1e-5
            and implementation_error["finite"]
            and implementation_error["relative_l2"] <= 0.008
            and code_difference <= 1
        ),
    )


def prepare(c, backend, precision, seed, distribution, tile):
    if backend not in {"gate-in-norm", "gate-in-gemm"} or precision != "int8":
        raise NotImplementedError("gate chain uses gate-in-norm/gate-in-gemm with INT8")
    import mlx.core as mx
    from .engine import NumericalMismatch, all_finite
    from .hadamard import rotate
    from .metal import gemm, quantize
    from .norm_hadamard import (
        fused_gated_residual_norm_hadamard_quantize,
        fused_norm_hadamard_quantize,
    )

    m, h, pk, n = c["m"], c["k"], c["previous_k"], c["n"]
    previous_tile = c.get("previous_tile")
    if previous_tile is None:
        previous_tile = tile
    if (
        not isinstance(previous_tile, (tuple, list))
        or len(previous_tile) != 4
        or any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in previous_tile)
    ):
        raise ValueError("previous_tile must contain four positive integers")
    previous_tile = tuple(int(v) for v in previous_tile)
    block, threads = 256, 256
    if h % block or pk % block:
        raise NotImplementedError("both K dimensions must divide H256; no implicit padding")
    rng = np.random.default_rng(seed)

    def array(shape, factor=1):
        return mx.array(rng.standard_normal(shape, dtype=np.float32) * factor).astype(mx.bfloat16)

    x, w0, w1 = array((m, pk)), array((h, pk), pk**-0.5), array((n, h), h**-0.5)
    residual, gate = array((m, h)), array((h,), 0.2)
    gamma, scale, shift = (
        (1 + array((h,), 0.1)).astype(mx.bfloat16),
        array((h,), 0.2),
        array((h,), 0.1),
    )
    if distribution == "outliers":
        x = (x * mx.where(mx.arange(pk) % 127 == 0, 20, 1)).astype(mx.bfloat16)
        w0 = (w0 * mx.where(mx.arange(pk) % 131 == 0, 20, 1)).astype(mx.bfloat16)
        w1 = (w1 * mx.where(mx.arange(h) % 131 == 0, 20, 1)).astype(mx.bfloat16)
        residual = (residual * mx.where(mx.arange(h) % 137 == 0, 20, 1)).astype(mx.bfloat16)
    elif distribution == "zeros":
        x, w0, w1, residual = map(mx.zeros_like, (x, w0, w1, residual))
    elif distribution != "normal":
        raise ValueError("unknown distribution")
    signs0, signs1 = mx.ones(pk, mx.int8), mx.ones(h, mx.int8)
    mx.eval(x, w0, w1, residual, gate, gamma, scale, shift, signs0, signs1)
    ri, ci, wi0 = sample_indices(m), sample_indices(n), sample_indices(h)
    ri_m, ci_m, wi0_m = mx.array(ri), mx.array(ci), mx.array(wi0)

    def np32(v):
        return np.asarray(v.astype(mx.float32))

    def input_prepare():
        return quantize(rotate(x, block, signs0), "int8", tag="chain_online")

    rotated_x = rotate(x, block, signs0)
    rotated_w0, rotated_w1 = rotate(w0, block, signs0), rotate(w1, block, signs1)
    aq, sa = quantize(rotated_x, "int8", tag="chain_packed")
    bq0, sb0 = quantize(rotated_w0, "int8", tag="chain_weight0")
    bq1, sb1 = quantize(rotated_w1, "int8", tag="chain_weight1")
    mx.eval(aq, sa, bq0, sb0, bq1, sb1, rotated_x, rotated_w0, rotated_w1)
    rotation_storage_finite = all_finite((rotated_x, rotated_w0, rotated_w1))
    packed_storage_finite = all_finite((aq, sa, bq0, sb0, bq1, sb1))
    preparation_scales_positive = all(bool(mx.all(s > 0).item()) for s in (sa, sb0, sb1))
    preparation_codes_valid = all(bool(mx.all(q >= -127).item()) for q in (aq, bq0, bq1))
    sampled_rotated_x = np32(rotated_x[ri_m]).copy()
    sampled_rotated_w0 = np32(rotated_w0[wi0_m]).copy()
    sampled_rotated_w1 = np32(rotated_w1[ci_m]).copy()
    # The reference needs only sampled rows; do not retain full FP32
    # rotations through the later CPU references or timed experiments.
    del rotated_x, rotated_w0, rotated_w1

    def previous(q, s, fused):
        return gemm(
            q,
            bq0,
            "int8",
            previous_tile,
            a_scales=s,
            b_scales=sb0,
            output_dtype=mx.bfloat16,
            **(dict(residual=residual, gate=gate) if fused else {}),
        )

    def transition(z, fused):
        if fused:
            q, s = fused_norm_hadamard_quantize(z, gamma, scale, shift, signs1, threads=threads)
            return q, s, z
        return fused_gated_residual_norm_hadamard_quantize(
            residual, z, gate, gamma, scale, shift, signs1, threads=threads
        )

    def next_projection(q, s):
        return gemm(q, bq1, "int8", tile, a_scales=s, b_scales=sb1, output_dtype=mx.bfloat16)

    fused = backend == "gate-in-gemm"
    z = previous(aq, sa, fused)
    q1, s1, rnext = transition(z, fused)
    # A second complete path checks that moving the gate preserves every
    # required BF16 boundary, independently of the float64 quality target.
    other_z = previous(aq, sa, not fused)
    other_q, other_s, other_r = transition(other_z, not fused)
    mx.eval(z, q1, s1, rnext, other_z, other_q, other_s, other_r)
    parity = dict(
        residual_exact=bool(mx.all(rnext == other_r).item()),
        activation_codes_exact=bool(mx.all(q1 == other_q).item()),
        activation_scales_exact=bool(mx.all(s1 == other_s).item()),
    )
    sampled_r = np32(rnext[ri_m]).astype(np.float64)
    actual_a = np.asarray(q1[ri_m]).astype(np.float64) * np.asarray(s1[ri_m])[:, None]
    actual_w = np.asarray(bq1[ci_m]).astype(np.float64) * np.asarray(sb1[ci_m])[:, None]
    expected = actual_a @ actual_w.T
    rx = np32(x[ri_m]).astype(np.float64)
    rw0 = np32(w0).astype(np.float64)
    rw1 = np32(w1[ci_m]).astype(np.float64)
    preparation_checks = dict(
        input_activation=_preparation_check(
            rx, sampled_rotated_x, np.asarray(aq[ri_m]), np.asarray(sa[ri_m]), block, ri
        ),
        previous_weight=_preparation_check(
            rw0[wi0], sampled_rotated_w0, np.asarray(bq0[wi0_m]), np.asarray(sb0[wi0_m]), block, wi0
        ),
        next_weight=_preparation_check(
            rw1, sampled_rotated_w1, np.asarray(bq1[ci_m]), np.asarray(sb1[ci_m]), block, ci
        ),
    )
    preparation_pass = (
        rotation_storage_finite
        and packed_storage_finite
        and preparation_scales_positive
        and preparation_codes_valid
        and all(check["passed"] for check in preparation_checks.values())
    )
    rr = np32(residual[ri_m]).astype(np.float64)
    rg = np32(gate).astype(np.float64)
    norm_args = [np32(v) for v in (gamma, scale, shift)]
    original_branch = _bf16_round(rx @ rw0.T)
    original_residual = _bf16_round(rr + original_branch * rg)
    original_norm = _bf16_round(norm_modulate_reference(original_residual, *norm_args))
    original = original_norm @ rw1.T
    decoded_x = np.asarray(aq[ri_m]).astype(np.float64) * np.asarray(sa[ri_m])[:, None]
    decoded_w0 = np.asarray(bq0).astype(np.float64) * np.asarray(sb0)[:, None]
    decoded_branch = _bf16_round(decoded_x @ decoded_w0.T)
    decoded_residual = _bf16_round(rr + decoded_branch * rg)
    residual_error = metrics(sampled_r, decoded_residual)
    # Norm/rotation reference uses actual BF16 residual, separating errors
    # in the previous GEMM from implementation of the following preparation.
    ref_norm = _bf16_round(norm_modulate_reference(sampled_r, *norm_args))
    ref_rot = block_hadamard_reference(ref_norm, block)
    ref_scale = np.max(np.abs(ref_rot), axis=-1) / 127
    ref_scale = np.where(ref_scale == 0, 1, ref_scale)
    ref_q = np.clip(np.rint(ref_rot / ref_scale[:, None]), -127, 127)
    prep_error = metrics(actual_a, ref_q * ref_scale[:, None])
    scale_error = metrics(np.asarray(s1[ri_m]), ref_scale)
    code_error = float(np.max(np.abs(np.asarray(q1[ri_m]).astype(float) - ref_q)))
    finite = all_finite((z, q1, s1, rnext, bq0, sb0, bq1, sb1)) and bool(mx.all(s1 > 0).item())
    branch_value = other_z if fused else z
    branch_error = metrics(np32(branch_value[ri_m]), decoded_branch)
    baseline_residual = np32(other_r[ri_m])

    def check(output):
        y, updated = output
        if not np.array_equal(np32(updated[ri_m]), baseline_residual):
            raise NumericalMismatch("chain residual differs from paired reference path")
        return np32(y[ri_m[:, None], ci_m[None, :]])

    def core(q, s):
        qn, sn, rn = transition(previous(q, s, fused), fused)
        return next_projection(qn, sn), rn

    def fn():
        return core(aq, sa)

    def online():
        return core(*input_prepare())

    def offline_weights():
        return (
            *quantize(rotate(w0, block, signs0), "int8", tag="chain_offline0"),
            *quantize(rotate(w1, block, signs1), "int8", tag="chain_offline1"),
        )

    meta = dict(
        compute_path=backend,
        input_dtype="bf16",
        output_dtype="bf16",
        reference="NumPy float64 full previous projection for sampled rows, next dot full K; explicit BF16 boundaries",
        mac_operations=2 * m * h * pk + 2 * m * n * h,
        sample_rows=ri.tolist(),
        sample_cols=ci.tolist(),
        branch_rounding="BF16 after INT8 dequant, before FP32 gate product",
        residual_rounding="BF16(R + float32(BF16(branch))*float32(gate)); no BF16 product round",
        normalization_rounding="completed AdaLN to BF16 before H256",
        returned_outputs=["next projection", "updated residual"],
        preceding_input_preparation="online H256 and INT8 packing",
        main_timing_scope="both projections and transition, resident preceding INT8 activation",
        previous_tile=list(previous_tile),
        next_tile=list(tile),
        online_timing_scope="input H/INT8 + previous GEMM + transition + next GEMM + residual output",
        offline_weight_preparation_in_online=False,
        preparation_checks=preparation_checks,
        preparation_rotated_storage_finite=rotation_storage_finite,
        preparation_packed_storage_finite=packed_storage_finite,
        preparation_all_scales_positive=preparation_scales_positive,
        preparation_all_codes_in_symmetric_range=preparation_codes_valid,
        preparation_pass=bool(preparation_pass),
        paired_path_parity=parity,
        previous_branch_implementation_error=branch_error,
        residual_implementation_error=residual_error,
        activation_implementation_error=prep_error,
        activation_scale_error=scale_error,
        activation_max_code_difference=code_error,
        previous_branch_quality_error=metrics(np32(branch_value[ri_m]), original_branch),
        residual_quality_error=metrics(sampled_r, original_residual),
        activation_quantization_error=metrics(actual_a, ref_rot),
        decoded_chain_quality_error=metrics(expected, original),
        transform_pass=bool(
            finite
            and preparation_pass
            and all(parity.values())
            and branch_error["relative_l2"] <= 0.008
            and residual_error["relative_l2"] <= 0.008
            and prep_error["relative_l2"] <= 0.008
            and scale_error["relative_l2"] <= 0.008
            and code_error <= 1
        ),
    )
    stages = dict(
        input_prepare=input_prepare,
        previous_projection=lambda: previous(aq, sa, fused),
        transition=lambda: transition(z, fused),
        next_projection=lambda: next_projection(q1, s1),
        online_activation_plus_gemm=online,
        offline_weight_prepare=offline_weights,
    )
    return fn, check, expected, original, stages, meta

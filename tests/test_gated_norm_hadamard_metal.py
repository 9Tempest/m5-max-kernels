import sys

import numpy as np
import pytest

pytestmark = pytest.mark.metal
if sys.platform != "darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)
mx = pytest.importorskip("mlx.core")

from m5kernels.gated_residual_reference import gated_residual_reference  # noqa: E402
from m5kernels.hadamard_reference import (  # noqa: E402
    block_hadamard_reference,
    hadamard_signs,
)
from m5kernels.norm_hadamard import (  # noqa: E402
    fused_gated_residual_norm_hadamard_quantize,
    fused_norm_hadamard_quantize,
    gated_residual_add,
)
from m5kernels.norm_hadamard_reference import norm_modulate_reference  # noqa: E402
from m5kernels.numerics import quantize_reference  # noqa: E402


def _cpu(value):
    return np.asarray(value.astype(mx.float32))


def _round(y, dtype):
    """Independent CPU emulation of the declared dtype boundary."""
    y = np.asarray(y, dtype=np.float32)
    if dtype == mx.float16:
        return y.astype(np.float16).astype(np.float32)
    if dtype == mx.bfloat16:
        bits = y.view(np.uint32)
        bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        return bits.view(np.float32)
    assert dtype == mx.float32
    return y


def _check(residual, branch, gate, gamma, scale, shift, signs, **kwargs):
    q, s, updated = fused_gated_residual_norm_hadamard_quantize(
        residual, branch, gate, gamma, scale, shift, mx.array(signs), **kwargs
    )
    q_cpu, s_cpu, updated_cpu = np.asarray(q), np.asarray(s), _cpu(updated)
    gate_map = kwargs.get("gate_row_map")
    gate_map_cpu = np.asarray(gate_map) if gate_map is not None else None
    ref_updated = _round(
        gated_residual_reference(_cpu(residual), _cpu(branch), _cpu(gate), row_map=gate_map_cpu),
        residual.dtype,
    )
    assert updated.dtype == residual.dtype and updated.shape == residual.shape
    assert np.isfinite(updated_cpu).all()
    # FP32 arithmetic can cross a reduced-precision midpoint by one ULP.
    # The explicit cancellation test below separately rejects early product
    # rounding, which would violate this contract even with small RMS error.
    tol = {mx.bfloat16: 8e-3, mx.float16: 1e-3, mx.float32: 3e-7}[residual.dtype]
    np.testing.assert_allclose(updated_cpu, ref_updated, rtol=tol, atol=1e-7)
    standalone = gated_residual_add(
        residual, branch, gate, gate_row_map=gate_map, threads=kwargs.get("threads", 256)
    )
    np.testing.assert_array_equal(updated_cpu, _cpu(standalone))

    # Check normalization/rotation/packing independently from the residual
    # update: use its verified actual rounded values as the next input.
    row_map = kwargs.get("row_map")
    norm = kwargs.get("norm", "rms")
    eps = kwargs.get("epsilon", 1e-5)
    if row_map is None:
        modulated = norm_modulate_reference(
            updated_cpu, _cpu(gamma), _cpu(scale), _cpu(shift), norm=norm, eps=eps
        )
    else:
        modulated = np.concatenate(
            [
                norm_modulate_reference(
                    updated_cpu[row : row + 1],
                    _cpu(gamma),
                    _cpu(scale[index]),
                    _cpu(shift[index]),
                    norm=norm,
                    eps=eps,
                )
                for row, index in enumerate(np.asarray(row_map))
            ]
        )
    dtype = residual.dtype if kwargs.get("round_output", "input") == "input" else mx.float32
    rotated = block_hadamard_reference(
        _round(modulated, dtype), kwargs.get("block_size", 256), signs
    )
    ref_q, ref_s = quantize_reference(rotated, "int8")
    assert q_cpu.dtype == np.int8 and s_cpu.dtype == np.float32
    assert np.isfinite(s_cpu).all() and np.all(s_cpu > 0)
    assert np.max(np.abs(q_cpu.astype(np.int16))) <= 127
    assert np.max(np.abs(q_cpu.astype(np.int16) - ref_q.astype(np.int16))) <= 1
    np.testing.assert_allclose(s_cpu, ref_s, rtol=3e-5, atol=1e-7)
    decoded = q_cpu.astype(np.float64) * s_cpu[:, None]
    expected = ref_q.astype(np.float64) * ref_s[:, None]
    error_bound = np.maximum(s_cpu, ref_s) + 127 * np.abs(s_cpu - ref_s)
    assert np.isfinite(decoded).all()
    assert np.all(np.abs(decoded - expected) <= error_bound[:, None] + 1e-7)
    return q_cpu, s_cpu, updated_cpu


@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("passes", ["one_read", "two_read"])
@pytest.mark.parametrize("threads", [128, 256])
@pytest.mark.parametrize("distribution", ["normal", "outliers"])
def test_h3_width_with_retained_residual(norm, passes, threads, distribution):
    rng = np.random.default_rng(323)
    shape = (7, 5376)
    residual = rng.standard_normal(shape, dtype=np.float32)
    branch = rng.standard_normal(shape, dtype=np.float32)
    if distribution == "outliers":
        branch[:, ::137] *= 100
        residual[:, ::251] *= 50
    residual, branch = mx.array(residual).astype(mx.bfloat16), mx.array(branch).astype(mx.bfloat16)
    gate = mx.array(rng.uniform(-0.7, 0.7, size=5376).astype(np.float32))
    gamma = mx.array(rng.uniform(0.7, 1.3, size=5376).astype(np.float32))
    scale = mx.array(rng.uniform(-0.2, 0.2, size=5376).astype(np.float32))
    shift = mx.array(rng.uniform(-0.1, 0.1, size=5376).astype(np.float32))
    before_r, before_b = _cpu(residual).copy(), _cpu(branch).copy()
    _check(
        residual,
        branch,
        gate,
        gamma,
        scale,
        shift,
        hadamard_signs(5376, 9),
        norm=norm,
        passes=passes,
        threads=threads,
    )
    np.testing.assert_array_equal(_cpu(residual), before_r)
    np.testing.assert_array_equal(_cpu(branch), before_b)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("passes", ["one_read", "two_read"])
def test_input_dtypes_fp32_modulation_and_inactive_groups(dtype, norm, passes):
    rng = np.random.default_rng(37)
    # Three H blocks with four SIMD groups exercises inactive-slot handling.
    residual = mx.array(rng.standard_normal((3, 768), dtype=np.float32)).astype(dtype)
    branch = mx.array(rng.standard_normal((3, 768), dtype=np.float32)).astype(dtype)
    gate = mx.full((768,), -0.25)
    _check(
        residual,
        branch,
        gate,
        mx.ones(768),
        mx.full((768,), 0.13),
        mx.full((768,), -0.07),
        hadamard_signs(768, 19),
        norm=norm,
        passes=passes,
        threads=128,
        round_output="fp32",
        epsilon=1e-4,
    )


@pytest.mark.parametrize("passes", ["one_read", "two_read"])
@pytest.mark.parametrize("norm", ["rms", "layer"])
def test_no_early_bf16_product_rounding(passes, norm):
    residual = mx.full((3, 768), -1.015625, dtype=mx.bfloat16)
    branch = mx.full((3, 768), 1.0078125, dtype=mx.bfloat16)
    gate = mx.full((768,), 1.0078125, dtype=mx.bfloat16)
    _, _, updated = _check(
        residual,
        branch,
        gate,
        mx.ones(768),
        mx.zeros(768),
        mx.full((768,), 0.125),
        hadamard_signs(768, 5),
        norm=norm,
        passes=passes,
    )
    # (1 + 2^-7)^2 - (1 + 2^-6) = 2^-14. Rounding the product
    # prematurely to BF16 would produce zero and silently change the block.
    np.testing.assert_array_equal(updated, np.float32(2**-14))


@pytest.mark.parametrize("case", ["zero_gate", "negative_gate", "zero_branch", "cancel"])
@pytest.mark.parametrize("norm", ["rms", "layer"])
def test_special_gate_cases(case, norm):
    rng = np.random.default_rng(911)
    residual = mx.array(rng.standard_normal((3, 768), dtype=np.float32)).astype(mx.bfloat16)
    branch = residual
    gate = mx.ones(768)
    if case == "zero_gate":
        gate = mx.zeros(768)
    elif case == "negative_gate":
        gate = mx.full((768,), -0.5)
    elif case == "zero_branch":
        branch = mx.zeros((3, 768), dtype=mx.bfloat16)
    else:
        gate = -gate
    _, _, updated = _check(
        residual,
        branch,
        gate,
        mx.ones(768),
        mx.zeros(768),
        mx.full((768,), 0.125),
        hadamard_signs(768, 7),
        norm=norm,
    )
    if case in ("zero_gate", "zero_branch"):
        np.testing.assert_array_equal(updated, _cpu(residual))
    if case == "cancel":
        np.testing.assert_array_equal(updated, 0)


@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("passes", ["one_read", "two_read"])
def test_independent_gate_and_adaln_tables(norm, passes):
    rng = np.random.default_rng(677)
    residual = mx.array(rng.standard_normal((7, 768), dtype=np.float32)).astype(mx.bfloat16)
    branch = mx.array(rng.standard_normal((7, 768), dtype=np.float32)).astype(mx.bfloat16)
    gate = mx.array(rng.uniform(-0.9, 0.9, size=(4, 768)).astype(np.float32))
    scale = mx.array(rng.uniform(-0.3, 0.3, size=(3, 768)).astype(np.float32))
    shift = mx.array(rng.uniform(-0.2, 0.2, size=(3, 768)).astype(np.float32))
    gate_map = mx.array([3, 1, 0, 1, 2, 0, 3], dtype=mx.int32)
    row_map = mx.array([2, 0, 1, 2, 0, 2, 1], dtype=mx.int32)
    q, s, _ = _check(
        residual,
        branch,
        gate,
        mx.ones(768),
        scale,
        shift,
        hadamard_signs(768, 23),
        norm=norm,
        passes=passes,
        gate_row_map=gate_map,
        row_map=row_map,
    )
    # Compare against the FAIR old fused-norm baseline after standalone gate.
    staged_r = gated_residual_add(residual, branch, gate, gate_row_map=gate_map)
    sq, ss = fused_norm_hadamard_quantize(
        staged_r,
        mx.ones(768),
        scale,
        shift,
        mx.array(hadamard_signs(768, 23)),
        norm=norm,
        passes=passes,
        row_map=row_map,
    )
    np.testing.assert_array_equal(q, np.asarray(sq))
    np.testing.assert_array_equal(s, np.asarray(ss))


def test_standalone_tail_width_and_strided_inputs():
    rng = np.random.default_rng(313)
    residual = mx.array(rng.standard_normal((769, 3), dtype=np.float32)).astype(mx.bfloat16).T
    branch = mx.array(rng.standard_normal((769, 3), dtype=np.float32)).astype(mx.bfloat16).T
    gate = mx.full((769,), -0.5)
    updated = gated_residual_add(residual, branch, gate)
    expected = _round(
        gated_residual_reference(_cpu(residual), _cpu(branch), _cpu(gate)), mx.bfloat16
    )
    np.testing.assert_array_equal(_cpu(updated), expected)


def test_gate_contract_rejections():
    residual, branch = mx.zeros((3, 768)), mx.zeros((3, 768))
    gate = mx.ones(768)
    with pytest.raises(ValueError, match="branch"):
        gated_residual_add(residual, mx.zeros((2, 768)), gate)
    with pytest.raises(ValueError, match="floating dtype"):
        gated_residual_add(residual, branch, gate.astype(mx.int8))
    with pytest.raises(ValueError, match="vector of length K"):
        gated_residual_add(residual, branch, mx.ones((2, 768)))
    with pytest.raises(ValueError, match="mapped gate"):
        gated_residual_add(residual, branch, gate, gate_row_map=mx.zeros(3, dtype=mx.int32))
    with pytest.raises(ValueError, match="int32 vector"):
        gated_residual_add(residual, branch, mx.ones((2, 768)), gate_row_map=mx.zeros(3))
    with pytest.raises(ValueError, match="int32 vector"):
        gated_residual_add(
            residual, branch, mx.ones((2, 768)), gate_row_map=mx.zeros(4, dtype=mx.int32)
        )
    with pytest.raises(NotImplementedError, match="threadgroup sizes"):
        gated_residual_add(residual, branch, gate, threads=64)

import sys

import pytest

from m5kernels.engine import run_case

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal"),
]


@pytest.mark.parametrize("distribution", ["normal", "outliers", "zeros"])
@pytest.mark.parametrize("bias_backend", ["mlx-fp32-convert", "mpp-bf16-to-fp32"])
def test_shift_split_projection_and_conditioning(distribution, bias_backend):
    pytest.importorskip("mlx.core")
    c = dict(
        id="split.tail",
        op="norm_hadamard_gemm",
        m=7,
        n=33,
        k=768,
        fusion=dict(round_output="fp32", block_size=256, sign_seed=0, bias_backend=bias_backend),
    )
    r = run_case(c, "norm-shift-split", "int8", distribution=distribution, warmup=1, repeat=3)
    assert r["status"] == "ok", r
    assert r["quality_pass"]
    assert r["shift_split"] and r["condition_prepare_timed"]
    assert r["conditioning_inclusive_implementation_error"]["relative_l2"] <= 0.008
    assert r["shift_projection_error"]["relative_l2"] <= 1e-5
    assert r["activation_scale_relative_l2_limit"] == 1e-5
    assert set(r["stages"]) == {
        "activation_prepare",
        "online_activation_plus_gemm",
        "condition_prepare",
        "online_with_conditioning",
    }


@pytest.mark.parametrize("fusion", [{}, {"norm": "layer", "round_output": "fp32"}])
def test_shift_split_rejects_incompatible_contracts(fusion):
    pytest.importorskip("mlx.core")
    c = dict(id="split.reject", op="norm_hadamard_gemm", m=3, n=17, k=256, fusion=fusion)
    r = run_case(c, "norm-shift-split", "int8", warmup=1, repeat=3)
    assert r["status"] == "unsupported"
    assert "round_output=fp32" in r["reason"]


def test_shift_split_rejects_ignored_schedule():
    pytest.importorskip("mlx.core")
    c = dict(
        id="split.schedule",
        op="norm_hadamard_gemm",
        m=3,
        n=17,
        k=256,
        fusion=dict(round_output="fp32", passes="two_read"),
    )
    r = run_case(c, "norm-shift-split", "int8", warmup=1, repeat=3)
    assert r["status"] == "unsupported"
    assert "passes=one_read" in r["reason"]


def test_shift_split_rejects_scale_bias(monkeypatch):
    pytest.importorskip("mlx.core")
    from m5kernels import rms_quant

    original = rms_quant.rms_quantize

    def corrupted(*args, **kwargs):
        q, s = original(*args, **kwargs)
        return q, s * 1.003

    monkeypatch.setattr(rms_quant, "rms_quantize", corrupted)
    c = dict(
        id="split.bad-scale",
        op="norm_hadamard_gemm",
        m=3,
        n=17,
        k=256,
        fusion=dict(round_output="fp32"),
    )
    r = run_case(c, "norm-shift-split", "int8", warmup=1, repeat=3)
    assert r["status"] == "incorrect"
    assert not r["transform_pass"]
    assert r["activation_scale_error"]["relative_l2"] > 1e-5
    assert "timing" not in r

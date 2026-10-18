import sys

import pytest

from m5kernels.engine import all_finite, run_case
from m5kernels.workloads import validate


def test_gated_projection_workload_shape_validation():
    case = dict(id="gate.test", op="gated_norm_hadamard_gemm", m=7, n=33, k=768)
    assert validate(case) == case
    with pytest.raises(ValueError, match="m"):
        validate(dict(case, m=0))


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("backend", ["mlx", "residual-separate", "residual-fused"])
def test_gated_projection_preserves_residual_output(norm, backend):
    pytest.importorskip("mlx.core")
    case = dict(
        id="gate.test",
        op="gated_norm_hadamard_gemm",
        m=7,
        n=33,
        k=768,
        fusion=dict(norm=norm, block_size=256, sign_seed=0),
    )
    result = run_case(
        case,
        backend,
        "bf16" if backend == "mlx" else "int8",
        distribution="outliers",
        warmup=1,
        repeat=3,
    )
    assert result["status"] == "ok", result
    assert result["quality_pass"]
    assert result["output_finite"] and result["online_output_finite"]
    assert result["residual_output_timed_stages"] == [
        "activation_prepare",
        "online_activation_plus_gemm",
    ]
    assert not result["residual_input_mutated"]
    if backend != "mlx":
        assert result["residual_update_exact_match"]
        assert result["residual_update_error"]["max_abs"] == 0
        assert result["transform_pass"]
        assert result["ideal_decoded_quantization_error"]["finite"]


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
def test_side_output_nonfinite_is_not_ignored():
    mx = pytest.importorskip("mlx.core")
    good = mx.array([1, 2], mx.float32)
    bad = mx.array([float("nan")], mx.float32)
    assert all_finite((good, [good]))
    assert not all_finite((good, [bad]))


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
def test_corrupted_residual_is_incorrect_with_diagnostics(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from m5kernels import norm_hadamard

    original = norm_hadamard.gated_residual_add

    def corrupt(*args, **kwargs):
        return (original(*args, **kwargs) + 1).astype(mx.bfloat16)

    monkeypatch.setattr(norm_hadamard, "gated_residual_add", corrupt)
    case = dict(id="gate.corrupt", op="gated_norm_hadamard_gemm", m=3, n=17, k=256)
    result = run_case(case, "residual-separate", "int8", warmup=1, repeat=3)
    assert result["status"] == "incorrect", result
    assert result["residual_update_error"]["max_abs"] > 0
    assert not result["residual_update_exact_match"]
    assert "timing" not in result

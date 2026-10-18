import sys

import pytest

from m5kernels.engine import run_case
from m5kernels.workloads import validate


def test_fused_workload_shape_validation():
    case = dict(id="test.norm_projection", op="norm_hadamard_gemm", m=7, n=33, k=768)
    assert validate(case) == case
    with pytest.raises(ValueError, match="k"):
        validate(dict(case, k=0))


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("backend", ["norm-staged", "norm-fused"])
def test_norm_rotation_projection_reference_contract(norm, backend):
    pytest.importorskip("mlx.core")
    case = dict(
        id="test.norm_projection",
        op="norm_hadamard_gemm",
        m=7,
        n=33,
        k=768,
        fusion=dict(norm=norm, block_size=256, sign_seed=0),
    )
    result = run_case(case, backend, "int8", distribution="outliers", warmup=1, repeat=3)
    assert result["status"] == "ok", result
    assert result["transform_pass"] and result["quality_pass"]
    assert result["online_output_finite"]
    assert result["activation_max_code_difference"] <= 1
    assert result["weight_max_code_difference"] <= 1
    assert result["weight_rotation_error"]["relative_l2"] < 1e-5
    assert result["ideal_decoded_quantization_error"]["finite"]
    assert result["decoded_preparation_total_error"]["finite"]
    assert result["stages"]["activation_prepare"]["median_ms"] > 0

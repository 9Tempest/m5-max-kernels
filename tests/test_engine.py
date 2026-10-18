import json
import sys
import pytest
from m5kernels.engine import write_result


def test_no_overwrite_evidence(tmp_path):
    p = tmp_path / "run.json"
    write_result(p, {"results": []})
    with pytest.raises(FileExistsError):
        write_result(p, {"results": [1]})
    assert json.loads(p.read_text()) == {"results": []}


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
@pytest.mark.parametrize(
    "backend,precision",
    [("mlx", "bf16"), ("mpp", "int8"), ("mpp", "fp8"), ("mx-emulated", "mxint8")],
)
def test_harness_reference_and_timing(backend, precision):
    from m5kernels.engine import run_case

    r = run_case(
        dict(id="test", op="gemm", m=32, n=64, k=128), backend, precision, warmup=1, repeat=3
    )
    assert r["status"] == "ok", r
    assert len(r["timing"]["samples_ms"]) == 3
    assert r["quality_pass"]
    if precision in {"int8", "fp8", "mxint8"}:
        assert r["online_output_finite"]
        assert r["online_implementation_error"]["relative_l2"] <= 0.008


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
@pytest.mark.parametrize("causal", [True, False])
def test_attention_reference(causal):
    from m5kernels.engine import run_case

    r = run_case(
        dict(id="test", op="attention", batch=1, heads=2, seq=65, head_dim=128, causal=causal),
        "mlx",
        "bf16",
        warmup=1,
        repeat=3,
    )
    assert r["status"] == "ok", r


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
@pytest.mark.parametrize("op", ["rms_modulate", "swiglu"])
def test_elementwise_distribution_reaches_reference(op):
    import numpy as np
    from m5kernels.engine import prepare_elementwise

    case = dict(id="test", op=op, m=2, n=256)
    references = {
        dist: prepare_elementwise(case, "mlx", "bf16", 17, dist, (256, 1, 1, 1))[2]
        for dist in ["normal", "zeros", "outliers"]
    }
    assert not np.array_equal(references["normal"], references["outliers"])
    assert np.all(references["zeros"] == references["zeros"][0, 0])
    if op == "swiglu":
        assert np.all(references["zeros"] == 0)


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
@pytest.mark.parametrize("precision", ["int8", "fp8"])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers"])
def test_hadamard_gemm_harness(precision, distribution):
    from m5kernels.engine import run_case

    case = dict(
        id="rotated",
        op="gemm",
        m=65,
        n=33,
        k=512,
        hadamard=dict(block_size=128, sign_seed=0, implementation="metal"),
    )
    r = run_case(
        case,
        "mpp-hadamard",
        precision,
        distribution=distribution,
        warmup=1,
        repeat=3,
        tile=(32, 64, 4, 8),
    )
    assert r["status"] == "ok", r
    assert r["transform_pass"] and r["online_output_finite"]
    assert r["transform"]["float64_dot_preservation_error"]["relative_l2"] <= 1e-12
    assert "activation_hadamard_plus_quantization" in r["stages"]
    assert not r["weight_rotation_timed"]
    assert not r["weight_quantization_timed"]

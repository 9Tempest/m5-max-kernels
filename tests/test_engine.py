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

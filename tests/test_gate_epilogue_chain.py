import sys
import pytest
from m5kernels.engine import run_case

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal"),
]


@pytest.mark.parametrize("distribution", ["normal", "outliers", "zeros"])
@pytest.mark.parametrize("backend", ["gate-in-norm", "gate-in-gemm"])
def test_chain_rounding_and_both_outputs(distribution, backend):
    pytest.importorskip("mlx.core")
    c = dict(id="chain.tail", op="gate_epilogue_chain", m=7, k=256, n=33, previous_k=768)
    r = run_case(c, backend, "int8", distribution=distribution, warmup=1, repeat=3)
    assert r["status"] == "ok", r
    assert r["quality_pass"]
    assert r["preparation_pass"]
    assert all(check["passed"] for check in r["preparation_checks"].values())
    assert all(r["paired_path_parity"].values())
    assert r["output_finite"] and r["online_output_finite"]

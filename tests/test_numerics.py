import numpy as np
import pytest
from m5kernels.numerics import fp8_encode, fp8_decode, quantize_reference, dequantize, metrics
from m5kernels.workloads import load, validate


def test_fp8_all_finite_encodings_roundtrip():
    codes = np.concatenate([np.arange(127, dtype=np.uint8), np.arange(128, 255, dtype=np.uint8)])
    np.testing.assert_array_equal(fp8_encode(fp8_decode(codes)), codes)


def test_fp8_endpoints_ties_and_subnormal():
    x = np.array([0, 2**-9, 2**-10, 1, 1.0625, 1.1875, 448, 1000, -448], np.float32)
    np.testing.assert_array_equal(fp8_encode(x), [0, 1, 0, 56, 56, 58, 126, 126, 254])


@pytest.mark.parametrize("mode", ["int8", "fp8", "mxint8"])
def test_quantization_zero_and_outlier(mode):
    x = np.zeros((2, 64), np.float32)
    x[1, 0] = 100
    q, s = quantize_reference(x, mode)
    y = dequantize(q, s, mode)
    assert np.isfinite(y).all()
    assert np.max(abs(y - x)) < 1
    assert np.all(y[0] == 0)


def test_mx_32_element_scale_boundaries():
    x = np.ones((1, 64), np.float32)
    x[:, 32:] = 16
    q, s = quantize_reference(x, "mxint8")
    np.testing.assert_array_equal(q, np.full((1, 64), 64, dtype=np.int8))
    np.testing.assert_array_equal(s, [[121, 125]])
    np.testing.assert_array_equal(dequantize(q, s, "mxint8"), x)


def test_metrics_reject_nonfinite():
    assert not metrics([float("nan")], [1])["finite"]
    assert metrics([0], [0])["relative_l2"] == 0
    assert metrics([1], [0])["relative_l2"] > 1


@pytest.mark.parametrize("name", ["h3", "smoke", "llm", "h3-long"])
def test_manifest(name):
    assert load(name)["schema_version"] == 1


def test_h3_attention_width_is_not_hidden_size():
    cases = {c["id"]: c for c in load("h3")["cases"]}
    assert cases["h3.attn_q.m128"]["n"] == 7168
    assert cases["h3.attn_q.m128"]["k"] == 5376
    assert cases["h3.ffn_gate_up_fused.m128"]["n"] == 28672


def test_reject_invalid_shape():
    with pytest.raises(ValueError):
        validate(dict(id="bad", op="gemm", m=-1, n=1, k=1))

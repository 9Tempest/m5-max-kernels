import sys
import numpy as np
import pytest

pytestmark = pytest.mark.metal
if sys.platform != "darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
from m5kernels.metal import quantize, gemm, rms_modulate  # noqa: E402
from m5kernels.numerics import quantize_reference, dequantize, metrics  # noqa: E402


@pytest.mark.parametrize("mode", ["int8", "fp8", "mxint8"])
@pytest.mark.parametrize("kind", ["normal", "zeros", "outliers"])
def test_gpu_quantizer_against_independent_reference(mode, kind):
    x = np.random.default_rng(41).standard_normal((7, 256), dtype=np.float32)
    if kind == "zeros":
        x.fill(0)
    if kind == "outliers":
        x[:, ::31] *= 30
    q, s = quantize(mx.array(x), mode)
    mx.eval(q, s)
    rq, rs = quantize_reference(x, mode)
    np.testing.assert_array_equal(np.asarray(q), rq)
    np.testing.assert_allclose(np.asarray(s), rs, rtol=1e-6)


@pytest.mark.parametrize("mode", ["bf16", "fp16", "int8", "fp8"])
@pytest.mark.parametrize("shape", [(1, 33, 64), (127, 65, 96), (65, 17, 257)])
def test_gemm_tail_tiles_and_k(mode, shape):
    m, n, k = shape
    rng = np.random.default_rng(5)
    dt = mx.float16 if mode == "fp16" else mx.bfloat16
    a = mx.array(rng.standard_normal((m, k), dtype=np.float32)).astype(dt)
    b = mx.array(rng.standard_normal((n, k), dtype=np.float32)).astype(dt)
    # Packed FP8 tensors require a multiple-of-128 row stride. Test rejection explicitly below.
    if mode == "fp8" and k % 128:
        pytest.skip("FP8 alignment contract tested separately")
    if mode in ["int8", "fp8"]:
        aq, sa = quantize(a, mode)
        bq, sb = quantize(b, mode)
        mx.eval(aq, sa, bq, sb)
        expected = (
            dequantize(np.asarray(aq), np.asarray(sa), mode).astype(np.float64)
            @ dequantize(np.asarray(bq), np.asarray(sb), mode).astype(np.float64).T
        )
        y = gemm(aq, bq, mode, tile=(32, 64, 4, 8), a_scales=sa, b_scales=sb, output_dtype=dt)
    else:
        expected = (
            np.asarray(a.astype(mx.float32)).astype(np.float64)
            @ np.asarray(b.astype(mx.float32)).astype(np.float64).T
        )
        y = gemm(a, b, mode, tile=(32, 64, 4, 8), output_dtype=dt)
    mx.eval(y)
    assert metrics(np.asarray(y.astype(mx.float32)), expected)["relative_l2"] < 0.008


def test_fp8_full_misaligned_output_tail():
    a = mx.ones((65, 256))
    b = mx.ones((33, 256))
    aq, sa = quantize(a, "fp8")
    bq, sb = quantize(b, "fp8")
    y = gemm(aq, bq, "fp8", a_scales=sa, b_scales=sb)
    np.testing.assert_allclose(np.asarray(y), 256, rtol=1e-5)


@pytest.mark.parametrize("n", [128, 5376, 5391])
def test_fused_rms_modulation(n):
    x = mx.random.normal((13, n)).astype(mx.bfloat16)
    w = mx.ones(n)
    s = mx.full((n,), 0.1)
    b = mx.full((n,), 0.05)
    y = rms_modulate(x, w, s, b)
    mx.eval(y)
    z = np.asarray(x.astype(mx.float32)).astype(np.float64)
    ref = z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + 1e-5) * 1.1 + 0.05
    assert metrics(np.asarray(y.astype(mx.float32)), ref)["relative_l2"] < 0.004


def test_fp8_alignment_is_explicit():
    a = mx.ones((2, 96))
    b = mx.ones((3, 96))
    aq, sa = quantize(a, "fp8")
    bq, sb = quantize(b, "fp8")
    with pytest.raises(NotImplementedError, match="stride"):
        gemm(aq, bq, "fp8", a_scales=sa, b_scales=sb)

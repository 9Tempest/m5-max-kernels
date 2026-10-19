"""CPU-only independent QKV reference and immutable native source contracts."""

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def widen(values):
    return (np.asarray(values, dtype=np.uint32) << 16).view(np.float32)


def ptr(array):
    assert array.flags.c_contiguous
    return array.ctypes.data_as(ctypes.c_void_p)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    compiler = shutil.which("clang") or shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable; CPU QKV reference cannot be checked")
    directory = tmp_path_factory.mktemp("h3-qkv-reference")
    source = directory / "wrapper.c"
    source.write_text(
        '#include "qkv_reference.h"\n'
        "uint16_t rounding(float x) { return qvr_bf16(x); }\n"
        "void project(const uint16_t*a,const uint16_t*w,const int8_t*aq,"
        "const int8_t*wq,float scale,const float*ws,unsigned k,double*o,double*d)"
        "{ qvr_project(a,w,aq,wq,scale,ws,k,o,d); }\n"
        "void post(const double*p,unsigned s,const uint16_t*g,const uint16_t*c,"
        "const uint16_t*t,double e,double*out) { qvr_postprocess(p,s,g,c,t,e,out); }\n"
        "void had(float *v,unsigned width) { qvr_h256(v,width); }\n"
    )
    library = directory / "reference.so"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "-I",
            str(ROOT / "integrations/h3"),
            str(source),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(library))
    lib.rounding.argtypes = [ctypes.c_float]
    lib.rounding.restype = ctypes.c_uint16
    lib.project.argtypes = (
        [ctypes.c_void_p] * 4
        + [ctypes.c_float, ctypes.c_void_p, ctypes.c_uint]
        + [ctypes.c_void_p] * 2
    )
    lib.post.argtypes = (
        [ctypes.c_void_p, ctypes.c_uint]
        + [ctypes.c_void_p] * 3
        + [ctypes.c_double, ctypes.c_void_p]
    )
    lib.had.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    return lib


@pytest.mark.parametrize("bits", [0, 0x80000000, 0x3F808000, 0x3F818000, 0xBF818000, 0x8000])
def test_round_to_even_boundary(reference, bits):
    value = np.array(bits, dtype=np.uint32).view(np.float32).item()
    assert reference.rounding(value) == bf16(value).item()


@pytest.mark.parametrize("seed", [17, 29])
def test_full_k_complete_head_decoded_and_original_are_separate(reference, seed):
    rng = np.random.default_rng(seed)
    k = 5376
    a, w = bf16(rng.normal(size=k)), bf16(rng.normal(size=(128, k)))
    aq = rng.integers(-127, 128, k, dtype=np.int8)
    wq = rng.integers(-127, 128, (128, k), dtype=np.int8)
    scale = np.float32(0.003)
    ws = np.linspace(0.001, 0.007, 128, dtype=np.float32)
    original, decoded = np.zeros(128), np.zeros(128)
    reference.project(*map(ptr, (a, w, aq, wq)), scale, ptr(ws), k, ptr(original), ptr(decoded))
    # An independent integer reduction proves every K term is decoded; use no
    # shared C implementation and no input quantization recipe for this oracle.
    expected = np.sum(wq.astype(np.int64) * aq, axis=1).astype(np.float64)
    expected *= float(scale) * ws.astype(np.float64)
    np.testing.assert_allclose(decoded, expected, rtol=2e-15, atol=1e-14)
    np.testing.assert_allclose(
        original, np.sum(widen(w).astype(np.float64) * widen(a), axis=1), rtol=1e-13, atol=1e-12
    )
    assert not np.allclose(decoded, original)


@pytest.mark.parametrize("distribution", ["normal", "zero", "epsilon", "outlier"])
@pytest.mark.parametrize("stream", [0, 1, 2])
def test_projection_round_before_full_head_norm_split_half_rope(reference, distribution, stream):
    rng = np.random.default_rng(29)
    projection = rng.normal(size=128)
    gamma = bf16(rng.normal(1, 0.2, 128))
    angle = rng.uniform(-3, 3, 48)
    cosine, sine = bf16(np.cos(angle)), bf16(np.sin(angle))
    if distribution == "zero":
        projection[:] = 0
    elif distribution == "epsilon":
        projection *= 1e-8
    elif distribution == "outlier":
        projection[[47, 48, 95, 96, 127]] *= 1000
    output = np.zeros(128)
    epsilon = float(np.float32(1e-5))
    reference.post(ptr(projection), stream, *map(ptr, (gamma, cosine, sine)), epsilon, ptr(output))
    rounded = widen(bf16(projection)).astype(np.float64)
    if stream == 2:
        expected = rounded
    else:
        norm = rounded / np.sqrt(np.mean(rounded**2) + epsilon) * widen(gamma)
        expected = norm.copy()
        expected[:48] = norm[:48] * widen(cosine) - norm[48:96] * widen(sine)
        expected[48:96] = norm[48:96] * widen(cosine) + norm[:48] * widen(sine)
        expected = widen(bf16(expected)).astype(np.float64)
    np.testing.assert_array_equal(output, expected)
    assert np.isfinite(output).all()
    if distribution == "epsilon" and stream != 2:
        assert 0 < np.mean(rounded**2) < epsilon
    if distribution == "zero":
        assert not np.any(output)


@pytest.mark.parametrize("width", [256, 5376])
def test_hadamard_scalar_order_against_independent_matrix(reference, width):
    rng = np.random.default_rng(17)
    original = rng.integers(-127, 128, size=width).astype(np.float32) / 16
    values = original.copy()
    matrix = np.array([[1.0]])
    for _ in range(8):
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    # These BF16 finite inputs sum exactly in FP32, so the mathematical matrix
    # is an independent oracle for sign, permutation and normalization.
    expected = (values.reshape(-1, 256).astype(np.float64) @ matrix.T / 16).astype(np.float32)
    reference.had(ptr(values), width)
    np.testing.assert_array_equal(values, expected.ravel())
    reference.had(ptr(values), width)
    np.testing.assert_allclose(
        values,
        original,
        rtol=3e-6,
        atol=3e-7,
    )


def test_shader_is_eagerly_registered_and_native_callsite_is_opt_in():
    shader = (ROOT / "integrations/h3/qkv_candidate.metal").read_text()
    overlay = (ROOT / "integrations/h3/qkv-candidate-overlay.patch").read_text()
    name = "h3_m5_qkv_rope_int8_r64_swizzle8"
    assert f"kernel void {name}(" in shader
    # One shader name in the eager creation registry, another in API lookup.
    assert overlay.count(f'@"{name}"') == 2
    assert 'getenv("H3_M5_QKV_ROPE")' in overlay
    assert 'strcmp(m5_qkv, "0")' in overlay
    assert 'strcmp(m5_qkv, "1")' in overlay
    assert "h3_gpu_m5_grouped_qkv_linear_rope_int8" in overlay
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in shader
    assert "if(stream==2) return" in shader
    assert "if(row>=args.rows) continue" in shader
    assert "bfloat rounded=bfloat" in shader

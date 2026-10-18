"""CPU-only independent checks for the native fusion reference, no Metal import."""

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
    return np.ascontiguousarray(array).ctypes.data_as(ctypes.c_void_p)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    compiler = shutil.which("clang") or shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable for CPU reference checks")
    directory = tmp_path_factory.mktemp("h3-fused-reference")
    source = directory / "wrapper.c"
    source.write_text(
        '#include "fused_reference.h"\n'
        "uint16_t rounding(float x) { return ref_bf16(x); }\n"
        "void projection(const uint16_t*a,const uint16_t*w,const int8_t*aq,"
        "const int8_t*wq,const float*as,const float*ws,size_t k,size_t n,size_t group,"
        "double*d,double*o) { ref_projection(a,w,aq,wq,as,ws,k,n,group,d,o); }\n"
        "int follow(const double*p,const uint16_t*r,const uint16_t*g,const uint16_t*n,"
        "const uint16_t*s,const uint16_t*t,size_t width,uint16_t*b,uint16_t*h,"
        "uint16_t*v,int8_t*q,float*scale) { return ref_follow(p,r,g,n,s,t,width,b,h,v,q,scale); }\n"
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
    lib.projection.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_size_t] * 3 + [ctypes.c_void_p] * 2
    lib.follow.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_size_t] + [ctypes.c_void_p] * 5
    lib.follow.restype = ctypes.c_int
    return lib


@pytest.mark.parametrize("bits", [0, 0x80000000, 0x3F808000, 0x3F818000, 0xBF818000, 0x00008000])
def test_bf16_round_to_even(reference, bits):
    value = np.array(bits, dtype=np.uint32).view(np.float32).item()
    assert reference.rounding(value) == bf16(value).item()


@pytest.mark.parametrize("group_size", [4, 12])
def test_full_k_grouped_and_row_decoded_reference(reference, group_size):
    rng = np.random.default_rng(17)
    k, n = 12, 7
    a = bf16(rng.normal(size=k))
    w = bf16(rng.normal(size=(n, k)))
    aq = rng.integers(-127, 128, k, dtype=np.int8)
    wq = rng.integers(-127, 128, (n, k), dtype=np.int8)
    scales = np.array([0.01, 0.002, 0.004][: k // group_size], dtype=np.float32)
    ws = np.linspace(0.001, 0.009, n, dtype=np.float32)
    decoded = np.empty(n, dtype=np.float64)
    original = np.empty(n, dtype=np.float64)
    reference.projection(
        *map(ptr, (a, w, aq, wq, scales, ws)), k, n, group_size, ptr(decoded), ptr(original)
    )
    expected = (wq.astype(np.float64) * ws[:, None]) @ (
        aq.astype(np.float64) * np.repeat(scales, group_size)
    )
    np.testing.assert_allclose(decoded, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(original, widen(w).astype(np.float64) @ widen(a), rtol=1e-13)


@pytest.mark.parametrize("width", [9, 5376])
@pytest.mark.parametrize("distribution", ["normal", "zero", "outlier"])
def test_complete_hidden_row_reference(reference, width, distribution):
    rng = np.random.default_rng(29)
    projection = rng.normal(size=width)
    residual, gate, norm, shift, scale = [bf16(rng.normal(size=width)) for _ in range(5)]
    gate[::3] = 0
    if distribution == "zero":
        projection[:] = 0
        residual[:] = 0
        shift[:] = 0
    elif distribution == "outlier":
        projection[-1] *= 100
    branch, hidden, adaln = [np.zeros(width, dtype=np.uint16) for _ in range(3)]
    codes, output_scale = np.zeros(width, dtype=np.int8), np.zeros(1, dtype=np.float32)
    assert reference.follow(
        *map(ptr, (projection, residual, gate, norm, shift, scale)),
        width,
        *map(ptr, (branch, hidden, adaln, codes, output_scale)),
    )
    expected_branch = bf16(projection)
    # BF16 multiplication is exact in FP32; float64 here independently models
    # fused multiply-add's single final FP32 rounding for this finite domain.
    expected_hidden = bf16(
        widen(expected_branch).astype(np.float64) * widen(gate) + widen(residual)
    )
    x = widen(expected_hidden).astype(np.float64)
    normalized = x / np.sqrt(np.mean(x * x) + float(np.float32(1e-5)))
    expected_adaln = bf16(
        normalized * widen(norm) * (1 + widen(scale).astype(np.float64)) + widen(shift)
    )
    maximum = np.max(np.abs(widen(expected_adaln)))
    expected_scale = np.float32(maximum / np.float32(127)) if maximum else np.float32(1 / 127)
    multiplier = np.float32(127) / maximum if maximum else np.float32(127)
    expected_codes = np.clip(np.rint(widen(expected_adaln) * multiplier), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(branch, expected_branch)
    np.testing.assert_array_equal(hidden, expected_hidden)
    np.testing.assert_array_equal(adaln, expected_adaln)
    np.testing.assert_array_equal(codes, expected_codes)
    assert output_scale[0] == expected_scale


def test_nonfinite_reference_is_explicit_failure(reference):
    projection = np.array([np.nan], dtype=np.float64)
    b = np.zeros(1, dtype=np.uint16)
    q, s = np.zeros(1, dtype=np.int8), np.zeros(1, dtype=np.float32)
    assert not reference.follow(ptr(projection), *[ptr(b)] * 5, 1, *[ptr(b)] * 3, ptr(q), ptr(s))

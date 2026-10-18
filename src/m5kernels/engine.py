"""Synchronized wall latency, bounded independent references, and explicit backend contracts."""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .numerics import dequantize, metrics, sample_indices
from .workloads import ROOT


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    files = sorted(
        p
        for folder in ["src", "workloads", "sources"]
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    digest = hashlib.sha256()
    for p in files:
        digest.update(str(p.relative_to(ROOT)).encode() + p.read_bytes())
    result = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        os=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=version("numpy"),
        git_commit=command(["git", "rev-parse", "HEAD"]),
        git_dirty=bool(command(["git", "status", "--porcelain"])),
        source_sha256=digest.hexdigest(),
        xcode=command(["xcodebuild", "-version"]),
        metal_compiler=command(["xcrun", "metal", "--version"]),
        thread_settings={
            k: os.environ.get(k)
            for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
        },
    )
    try:
        import mlx.core as mx

        result.update(mlx=version("mlx"), device=mx.device_info())
    except ImportError:
        result["device"] = None
    return result


def timed(fn, warmup, repeat):
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeat):
        mx.synchronize()
        start = time.perf_counter_ns()
        y = fn()
        mx.eval(y)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return dict(
        kind="synchronized_host_wall",
        samples_ms=samples,
        median_ms=statistics.median(samples),
        p10_ms=float(np.percentile(samples, 10)),
        p90_ms=float(np.percentile(samples, 90)),
        min_ms=min(samples),
        warmup=warmup,
        repeat=repeat,
    )


def _np(x):
    import mlx.core as mx

    return np.asarray(x.astype(mx.float32))


def _dequant_mlx(q, s, mode):
    import mlx.core as mx

    if mode == "mxint8":
        return (
            q.reshape(q.shape[0], -1, 32).astype(mx.float32)
            * mx.power(2.0, s.astype(mx.float32) - 127)[..., None]
        ).reshape(q.shape)
    raise ValueError(mode)


def prepare_gemm(c, backend, precision, seed, distribution, tile):
    import mlx.core as mx
    from .metal import gemm, quantize

    m, n, k = c["m"], c["n"], c["k"]
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((n, k), dtype=np.float32) / np.sqrt(k)
    if distribution == "outliers":
        a[:, ::127] *= 20
        b[:, ::131] *= 20
    elif distribution == "zeros":
        a.fill(0)
        b.fill(0)
    dtype = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}.get(
        precision, mx.bfloat16
    )
    x = mx.array(a).astype(dtype)
    w = mx.array(b).astype(dtype)
    mx.eval(x, w)
    ri = sample_indices(m)
    ci = sample_indices(n)
    ar = _np(x[mx.array(ri)])
    br = _np(w[mx.array(ci)])
    original = ar.astype(np.float64) @ br.astype(np.float64).T
    expected = original
    stages = {}
    path = ""
    quant = False
    if backend == "mlx":
        if precision not in {"bf16", "fp16", "fp32"}:
            raise NotImplementedError("MLX baseline uses floating point operands; no A8 claim")

        def fn():
            return x @ w.T

        path = "mlx.matmul"
    elif backend == "mpp":
        if precision in {"bf16", "fp16", "fp32"}:

            def fn():
                return gemm(x, w, precision, tile, output_dtype=dtype)

            path = "tensorops_float_inputs"
        elif precision in {"int8", "fp8"}:
            aq, sa = quantize(x, precision)
            bq, sb = quantize(w, precision)
            mx.eval(aq, sa, bq, sb)
            qr = dequantize(np.asarray(aq[mx.array(ri)]), np.asarray(sa[mx.array(ri)]), precision)
            wr = dequantize(np.asarray(bq[mx.array(ci)]), np.asarray(sb[mx.array(ci)]), precision)
            expected = qr.astype(np.float64) @ wr.astype(np.float64).T

            def fn():
                return gemm(aq, bq, precision, tile, a_scales=sa, b_scales=sb, output_dtype=dtype)

            def online():
                q, s = quantize(x, precision)
                return gemm(q, bq, precision, tile, a_scales=s, b_scales=sb, output_dtype=dtype)

            stages = {
                "activation_quantization": lambda: quantize(x, precision),
                "online_activation_plus_gemm": online,
            }
            path = "tensorops_int8_int32" if precision == "int8" else "tensorops_e4m3_float32"
            quant = True
        elif precision == "mxint8":
            raise NotImplementedError(
                "native scaled INT8->FP32 MPP path fails compilation on this stack; use mx-emulated"
            )
        else:
            raise ValueError(precision)
    elif backend == "mx-emulated":
        if precision != "mxint8":
            raise NotImplementedError("mx-emulated is only for MXINT8")
        aq, sa = quantize(x, "mxint8")
        bq, sb = quantize(w, "mxint8")
        mx.eval(aq, sa, bq, sb)
        qr = dequantize(np.asarray(aq[mx.array(ri)]), np.asarray(sa[mx.array(ri)]), "mxint8")
        wr = dequantize(np.asarray(bq[mx.array(ci)]), np.asarray(sb[mx.array(ci)]), "mxint8")
        expected = qr.astype(np.float64) @ wr.astype(np.float64).T
        # Scales map exactly into BF16 for this finite-value experiment; this is not native MX compute.
        wd = _dequant_mlx(bq, sb, "mxint8").astype(mx.bfloat16)
        mx.eval(wd)

        def fn():
            return _dequant_mlx(aq, sa, "mxint8").astype(mx.bfloat16) @ wd.T

        def online():
            q, s = quantize(x, "mxint8")
            return _dequant_mlx(q, s, "mxint8").astype(mx.bfloat16) @ wd.T

        stages = {
            "activation_quantization": lambda: quantize(x, "mxint8"),
            "online_activation_plus_gemm": online,
        }
        path = "software_mxint8_dequantize_then_bf16_matmul"
        quant = True
    else:
        raise ValueError(backend)

    def check(y):
        return _np(y[mx.array(ri)[:, None], mx.array(ci)[None, :]])

    meta = dict(
        compute_path=path,
        input_dtype=str(dtype),
        output_dtype=str(dtype),
        reference="NumPy float64 sampled rows x columns, full K",
        sample_rows=ri.tolist(),
        sample_cols=ci.tolist(),
        quantization="symmetric_per_token_A_per_output_channel_W"
        if quant and precision != "mxint8"
        else precision,
        weight_quantization_timed=False,
        weight_layout="N,K; transpose in matmul",
        mac_operations=2 * m * n * k,
        bytes_lower_bound=(m * k + n * k) * (1 if quant else (4 if precision == "fp32" else 2))
        + m * n * (4 if precision == "fp32" else 2),
    )
    return fn, check, expected, original, stages, meta


def prepare_attention(c, backend, precision, seed, distribution, tile):
    import mlx.core as mx
    from .metal import quantize

    if backend not in {"mlx", "qdq"}:
        raise NotImplementedError("attention backend must be mlx or qdq")
    if backend == "mlx" and precision not in {"bf16", "fp16", "fp32"}:
        raise NotImplementedError("use qdq for quantized attention sensitivity")
    if backend == "qdq" and precision not in {"int8", "fp8", "mxint8"}:
        raise NotImplementedError("qdq requires quantized precision")
    b, h, s, d = [c[k] for k in ("batch", "heads", "seq", "head_dim")]
    rng = np.random.default_rng(seed)
    dtype = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}.get(
        precision, mx.bfloat16
    )
    arrays = [
        mx.array(rng.standard_normal((b, h, s, d), dtype=np.float32)).astype(dtype)
        for _ in range(3)
    ]
    if distribution == "zeros":
        arrays = [mx.zeros_like(x) for x in arrays]
    if distribution == "outliers":
        arrays = [x * mx.where(mx.arange(d) % 31 == 0, 10, 1) for x in arrays]
    mx.eval(*arrays)
    qi = sample_indices(s, 16)

    def ref(q, k, v):
        # All B/H, selected queries, every key. No quadratic SxS reference allocation.
        q = _np(q[:, :, mx.array(qi), :]).astype(np.float64)
        k = _np(k).astype(np.float64)
        v = _np(v).astype(np.float64)
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(d)
        if c.get("causal", False):
            scores = np.where(
                np.arange(s)[None, None, None, :] <= qi[None, None, :, None], scores, -np.inf
            )
        p = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        return p @ v

    original = ref(*arrays)
    expected = original
    if backend == "qdq":
        # Sensitivity-only experiment. Quantized operands are decoded before ordinary SDPA.
        out = []
        for x in arrays:
            q, scale = quantize(x.reshape(-1, d), precision)
            mx.eval(q, scale)
            decoded = dequantize(np.asarray(q), np.asarray(scale), precision).reshape(x.shape)
            out.append(mx.array(decoded).astype(dtype))
        arrays = out
        mx.eval(*arrays)
        expected = ref(*arrays)

    def fn():
        return mx.fast.scaled_dot_product_attention(
            *arrays, scale=d**-0.5, mask="causal" if c.get("causal", False) else None
        )

    def check(y):
        return _np(y[:, :, mx.array(qi), :])

    return (
        fn,
        check,
        expected,
        original,
        {},
        dict(
            compute_path="mlx_sdpa" if backend == "mlx" else "QDQ_sensitivity_only_BF16_SDPA",
            input_dtype=str(dtype),
            output_dtype=str(dtype),
            reference="NumPy float64 selected queries, all keys, all heads",
            sample_queries=qi.tolist(),
            mac_operations=4 * b * h * s * s * d,
            bytes_lower_bound=4 * b * h * s * d * (4 if precision == "fp32" else 2),
            quantization_timed=False,
            causal=c.get("causal", False),
        ),
    )


def prepare_elementwise(c, backend, precision, seed, distribution, tile):
    import mlx.core as mx
    from .metal import rms_modulate

    if precision not in {"bf16", "fp16", "fp32"}:
        raise NotImplementedError("elementwise uses floating point")
    rng = np.random.default_rng(seed)
    m, n = c["m"], c["n"]
    dtype = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[precision]
    x = mx.array(rng.standard_normal((m, n), dtype=np.float32)).astype(dtype)
    w = mx.ones((n,), dtype=dtype)
    scale = mx.full((n,), 0.1, dtype=dtype)
    shift = mx.full((n,), 0.05, dtype=dtype)
    mx.eval(x, w, scale, shift)
    if c["op"] == "rms_modulate":

        def baseline():
            z = x.astype(mx.float32)
            r = z * mx.rsqrt(mx.mean(z * z, axis=-1, keepdims=True) + 1e-5) * w
            return (r * (1 + scale.astype(mx.float32)) + shift).astype(dtype)

        if backend == "mlx":
            fn = mx.compile(baseline)
        elif backend == "metal":

            def fn():
                return rms_modulate(x, w, scale, shift, threads=tile[0])
        else:
            raise NotImplementedError("rms backend must be mlx or metal")
        z = _np(x).astype(np.float64)
        expected = z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + 1e-5) * _np(w)
        expected = expected * (1 + _np(scale)) + _np(shift)
    else:
        if backend != "mlx":
            raise NotImplementedError("SwiGLU currently has a compiled MLX baseline")
        gate = mx.array(rng.standard_normal((m, n), dtype=np.float32)).astype(dtype)
        mx.eval(gate)
        fn = mx.compile(
            lambda: (x.astype(mx.float32) * mx.sigmoid(x.astype(mx.float32)) * gate).astype(dtype)
        )
        z = _np(x).astype(np.float64)
        expected = z / (1 + np.exp(-z)) * _np(gate)
    return (
        fn,
        _np,
        expected,
        expected,
        {},
        dict(
            compute_path=backend,
            input_dtype=str(dtype),
            output_dtype=str(dtype),
            reference="NumPy float64 full tensor",
            mac_operations=None,
            bytes_lower_bound=m * n * 4,
        ),
    )


def run_case(
    c, backend, precision, seed=17, distribution="normal", warmup=3, repeat=10, tile=(64, 64, 4, 1)
):
    import mlx.core as mx

    result = dict(
        case=c,
        backend=backend,
        precision=precision,
        seed=seed,
        distribution=distribution,
        tile=list(tile),
    )
    try:
        # Conservative budget includes references and temporary activation buffers.
        if c["op"] == "gemm":
            est = (c["m"] * c["k"] + c["n"] * c["k"] + c["m"] * c["n"]) * 24
        elif c["op"] == "attention":
            est = c["batch"] * c["heads"] * c["seq"] * c["head_dim"] * 64
        else:
            est = c["m"] * c["n"] * 48
        if est > 32 * 1024**3:
            raise NotImplementedError("estimated experiment working set >32 GiB; split workload")
        prepare = {"gemm": prepare_gemm, "attention": prepare_attention}.get(
            c["op"], prepare_elementwise
        )
        fn, check, expected, original, stages, meta = prepare(
            c, backend, precision, seed, distribution, tile
        )
        y = fn()
        mx.eval(y)
        output_finite = bool(mx.all(mx.isfinite(y)).item())
        actual = check(y)
        impl = metrics(actual, expected)
        quality = metrics(actual, original)
        tolerance = 0.0001 if precision == "fp32" else (0.001 if precision == "fp16" else 0.008)
        passed = output_finite and impl["finite"] and impl["relative_l2"] <= tolerance
        quality_limit = {
            "fp16": 0.001,
            "bf16": 0.008,
            "fp32": 0.0001,
            "int8": 0.05,
            "mxint8": 0.05,
            "fp8": 0.10,
        }[precision]
        quality_passed = quality["finite"] and quality["relative_l2"] <= quality_limit
        result.update(
            meta,
            status="ok" if passed else "incorrect",
            output_finite=output_finite,
            implementation_error=impl,
            quality_error=quality,
            implementation_relative_l2_limit=tolerance,
            quality_relative_l2_limit=quality_limit,
            quality_pass=quality_passed,
        )
        # Never promote or benchmark an incorrect implementation.
        if passed:
            result["timing"] = timed(fn, warmup, repeat)
            result["stages"] = {name: timed(f, warmup, repeat) for name, f in stages.items()}
            if meta["mac_operations"]:
                result["effective_tops"] = meta["mac_operations"] / (
                    result["timing"]["median_ms"] * 1e9
                )
        del y, fn, stages
    except NotImplementedError as e:
        result.update(status="unsupported", reason=str(e))
    except Exception as e:
        result.update(status="error", reason=str(e)[:5000])
    mx.clear_cache()
    return result


def write_result(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidental replacement of experiment evidence.
    with path.open("x") as f:
        json.dump(document, f, indent=2, allow_nan=False)
        f.write("\n")

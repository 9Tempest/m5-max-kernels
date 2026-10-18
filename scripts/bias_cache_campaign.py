"""Measure projected-shift cache construction, including input conversion.

This is per-request conditioning preparation, not reusable static weight work.
All rows here are synthetic; 60/117 correspond to planned H3 schedule table sizes.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from m5kernels.engine import environment, timed, write_result
from m5kernels.locking import gpu_lock
from m5kernels.numerics import metrics, sample_indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.repeat < 3:
        p.error("repeat >=3 required")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    import mlx.core as mx
    from m5kernels.metal import gemm

    with gpu_lock():
        rng = np.random.default_rng(args.seed)
        n, k = 7168, 5376
        w = mx.array(rng.standard_normal((n, k), dtype=np.float32)).astype(mx.bfloat16)
        beta = mx.array(0.1 * rng.standard_normal((117, k), dtype=np.float32)).astype(mx.bfloat16)
        mx.eval(w, beta)
        columns = sample_indices(n)
        wr = np.asarray(w[mx.array(columns)].astype(mx.float32)).astype(np.float64)
        doc = dict(
            schema_version=1,
            environment=environment(),
            results=[],
            campaign_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            experiment="per-request projected-shift cache construction; resident BF16 operands; FP32 output",
        )
        for rows in [1, 3, 60, 117]:
            b = beta[:rows]
            mx.eval(b)
            reference = np.asarray(b.astype(mx.float32)).astype(np.float64) @ wr.T
            for backend in ["mlx-fp32-convert", "mpp-bf16-to-fp32"]:

                def fn():
                    if backend == "mlx-fp32-convert":
                        return b.astype(mx.float32) @ w.astype(mx.float32).T
                    return gemm(b, w, "bf16", (32, 64, 4, 1), output_dtype=mx.float32)

                r = dict(
                    case=dict(id=f"bias.p{rows}", op="condition_bias", m=rows, n=n, k=k),
                    backend=backend,
                    seed=args.seed,
                    input_dtype="bf16",
                    output_dtype="fp32",
                    cache_bytes=rows * n * 4,
                    static_weight_preparation=False,
                    input_conversion_in_timing=True,
                    reference="NumPy float64 all rows, sampled columns, full K",
                )
                try:
                    output = fn()
                    mx.eval(output)
                    error = metrics(np.asarray(output[:, mx.array(columns)]), reference)
                    finite = bool(mx.all(mx.isfinite(output)).item())
                    passed = finite and error["finite"] and error["relative_l2"] <= 1e-4
                    r.update(
                        status="ok" if passed else "incorrect",
                        implementation_error=error,
                        output_finite=finite,
                        implementation_relative_l2_limit=1e-4,
                    )
                    if passed:
                        r["timing"] = timed(fn, 10, args.repeat)
                except Exception as e:
                    r.update(status="error", reason=str(e)[:5000])
                doc["results"].append(r)
                print(rows, backend, r["status"], r.get("timing", {}).get("median_ms"), flush=True)
        write_result(args.output, doc)
    return int(any(r["status"] != "ok" for r in doc["results"]))


if __name__ == "__main__":
    raise SystemExit(main())

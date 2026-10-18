"""Reproducible Xcode GPU captures and Instruments system traces."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .engine import (
    environment,
    metrics,
    prepare_gemm,
    prepare_attention,
    prepare_elementwise,
    write_result,
)
from .locking import gpu_lock
from .workloads import load


def workload(args):
    import mlx.core as mx

    cases = [c for c in load(args.suite)["cases"] if c["id"] == args.case]
    if len(cases) != 1:
        raise ValueError("select one exact case id")
    case = cases[0]
    prepare = {"gemm": prepare_gemm, "attention": prepare_attention}.get(
        case["op"], prepare_elementwise
    )
    fn, check, expected, original, stages, meta = prepare(
        case, args.backend, args.precision, args.seed, "normal", tuple(args.tile)
    )
    if args.online:
        if "online_activation_plus_gemm" not in stages:
            raise ValueError("online stage requires a quantized GEMM backend")
        fn = stages["online_activation_plus_gemm"]
    for _ in range(5):
        mx.eval(fn())
    y = fn()
    mx.eval(y)
    error = metrics(check(y), expected)
    if (
        not error["finite"]
        or error["relative_l2"]
        > (0.0001 if args.precision == "fp32" else 0.001 if args.precision == "fp16" else 0.008)
        or not mx.all(mx.isfinite(y)).item()
    ):
        raise ValueError("refusing to profile an incorrect kernel")
    return fn, dict(
        case=case,
        backend=args.backend,
        precision=args.precision,
        tile=args.tile,
        online=args.online,
        implementation_error=error,
        quality_error=metrics(check(y), original),
        **meta,
    )


def run(args):
    # Capture must be enabled before MLX initializes its Metal device.
    os.environ.setdefault("MTL_CAPTURE_ENABLED", "1")
    if args.kind == "system" and not args.worker:
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        worker = [sys.executable, "-m", "m5kernels.profile", *sys.argv[1:], "--worker"]
        # xctrace observes initialization too; worker prints a READY marker before the hot loop.
        cmd = [
            "xcrun",
            "xctrace",
            "record",
            "--template",
            "Metal System Trace",
            "--output",
            str(output),
            "--time-limit",
            f"{args.seconds + 30}s",
            "--target-stdout",
            "-",
            "--launch",
            "--",
            *worker,
        ]
        return subprocess.run(cmd, check=False).returncode
    import mlx.core as mx

    with gpu_lock():
        fn, meta = workload(args)
        doc = dict(
            schema_version=1,
            environment=environment(),
            capture=meta,
            kind=args.kind,
            profiling_overhead=True,
            not_a_benchmark_result=True,
        )
        print("M5BENCH_READY: warmed, numerically checked workload", flush=True)
        output = Path(args.output).resolve()
        if args.kind == "gpu":
            if output.exists():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            mx.synchronize()
            mx.metal.start_capture(str(output))
            try:
                # Flush any encoder created before capture, on the workload stream.
                marker = mx.fast.metal_kernel(
                    name="capture_boundary",
                    input_names=[],
                    output_names=["Y"],
                    source="Y[0] = 1;",
                )
                mx.eval(
                    marker(
                        inputs=[],
                        grid=(1, 1, 1),
                        threadgroup=(1, 1, 1),
                        output_shapes=[(1,)],
                        output_dtypes=[mx.int32],
                    )
                )
                mx.synchronize()
                for _ in range(args.iterations):
                    mx.eval(fn())
                mx.synchronize()
            finally:
                mx.metal.stop_capture()
            doc["iterations"] = args.iterations

        else:
            start = time.monotonic()
            count = 0
            while time.monotonic() - start < args.seconds:
                mx.eval(fn())
                count += 1
            mx.synchronize()
            doc["iterations"] = count
            doc["hot_loop_seconds"] = time.monotonic() - start
        write_result(str(output) + ".json", doc)
        print(json.dumps(dict(output=str(output), iterations=doc["iterations"])), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["gpu", "system"], default="gpu")
    p.add_argument("--suite", default="h3")
    p.add_argument("--case", required=True)
    p.add_argument("--backend", default="mpp")
    p.add_argument("--precision", default="int8")
    p.add_argument("--tile", nargs=4, type=int, default=[64, 64, 4, 1])
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--seconds", type=int, default=5)
    p.add_argument("--online", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.iterations < 1 or not 1 <= args.seconds <= 60:
        p.error("iterations >=1, seconds 1..60 required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

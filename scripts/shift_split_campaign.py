"""Compare native-boundary AdaLN, FP32 AdaLN and cached projected-shift policies."""

import argparse
import hashlib
import json
from pathlib import Path

from m5kernels.engine import environment, run_case, write_result
from m5kernels.locking import gpu_lock
from m5kernels.workloads import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filter", default="m4096")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--split-threads", type=int, default=256)
    p.add_argument(
        "--bias-backend",
        default="mlx-fp32-convert",
        choices=["mlx-fp32-convert", "mpp-bf16-to-fp32"],
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.repeat < 3:
        p.error("repeat >=3 required")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    cases = [c for c in load("h3-shift-split")["cases"] if args.filter in c["id"]]
    if not cases:
        p.error("no cases matched")
    with gpu_lock():
        doc = dict(
            schema_version=1,
            environment=environment(),
            results=[],
            experiment="shared P=1 conditioning; original-W projected shift + INT8 dynamic branch",
            campaign_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        configs = [("norm-fused", "input"), ("norm-fused", "fp32"), ("norm-shift-split", "fp32")]
        if args.reverse:
            configs.reverse()
        for case in cases:
            for distribution in ["normal", "outliers"]:
                for backend, rounding in configs:
                    c = dict(case, fusion=dict(case["fusion"], round_output=rounding))
                    if backend == "norm-shift-split":
                        c["fusion"].update(
                            threads=args.split_threads, bias_backend=args.bias_backend
                        )
                    r = run_case(
                        c,
                        backend,
                        "int8",
                        args.seed,
                        distribution,
                        args.warmup,
                        args.repeat,
                        (128, 64, 4, 8),
                    )
                    doc["results"].append(r)
                    print(
                        json.dumps(
                            dict(
                                case=c["id"],
                                backend=backend,
                                rounding=rounding,
                                distribution=distribution,
                                status=r["status"],
                                stages_ms={
                                    k: v["median_ms"] for k, v in r.get("stages", {}).items()
                                },
                                quality=r.get("quality_error", {}).get("relative_l2"),
                                reason=r.get("reason"),
                            )
                        ),
                        flush=True,
                    )
        write_result(args.output, doc)
    return int(any(r["status"] in {"error", "incorrect"} for r in doc["results"]))


if __name__ == "__main__":
    raise SystemExit(main())

"""Interleaved staged/fused AdaLN + Hadamard + INT8 preparation and projection."""

import argparse
import hashlib
import json
from pathlib import Path

from m5kernels.engine import environment, run_case, write_result
from m5kernels.locking import gpu_lock
from m5kernels.workloads import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="h3.norm_q.m4096,h3.norm_ffn_up.m4096")
    p.add_argument("--norms", default="rms")
    p.add_argument("--threads", default="128,256")
    p.add_argument("--passes", default="one_read,two_read")
    p.add_argument("--distributions", default="normal,outliers")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.repeat < 3:
        p.error("repeat >= 3 required")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    cases = {c["id"]: c for c in load("h3-norm-hadamard")["cases"]}
    with gpu_lock():
        doc = dict(
            schema_version=1,
            environment=environment(),
            results=[],
            experiment="native-style AdaLN BF16 boundary; H256; offline rotated W; paired inputs",
            campaign_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        for name in args.cases.split(","):
            for norm in args.norms.split(","):
                for distribution in args.distributions.split(","):
                    base = cases[name]
                    configs = [("mlx", "bf16", {}), ("norm-staged", "int8", {})]
                    configs += [
                        ("norm-fused", "int8", dict(threads=t, passes=passes))
                        for t in map(int, args.threads.split(","))
                        for passes in args.passes.split(",")
                    ]
                    for backend, precision, config in configs:
                        c = dict(base, fusion=dict(base["fusion"], norm=norm, **config))
                        r = run_case(
                            c,
                            backend,
                            precision,
                            args.seed,
                            distribution,
                            5,
                            args.repeat,
                            (128, 64, 4, 8),
                        )
                        doc["results"].append(r)
                        print(
                            json.dumps(
                                dict(
                                    case=name,
                                    norm=norm,
                                    distribution=distribution,
                                    backend=backend,
                                    config=config,
                                    status=r["status"],
                                    prep_ms=r.get("stages", {})
                                    .get("activation_prepare", {})
                                    .get("median_ms"),
                                    online_ms=r.get("stages", {})
                                    .get("online_activation_plus_gemm", {})
                                    .get("median_ms"),
                                    quality=r.get("quality_error", {}).get("relative_l2"),
                                    transform_pass=r.get("transform_pass"),
                                    reason=r.get("reason"),
                                )
                            ),
                            flush=True,
                        )
        write_result(args.output, doc)
        return int(any(r["status"] in {"error", "incorrect"} for r in doc["results"]))


if __name__ == "__main__":
    raise SystemExit(main())

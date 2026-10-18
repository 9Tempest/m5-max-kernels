"""Paired direct/folded RMS quantization with preparation and full projection timings."""

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
    p.add_argument("--distributions", default="normal,outliers")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--reverse", action="store_true", help="reverse backend order for independent rerun"
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.repeat < 3:
        p.error("repeat >=3 required")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    cases = [c for c in load("rms-quant")["cases"] if args.filter in c["id"]]
    if not cases:
        p.error("no cases matched")
    with gpu_lock():
        doc = dict(
            schema_version=1,
            environment=environment(),
            experiment="zero-shift RMS; FP32 intermediates; paired direct/folded inverse RMS scale",
            campaign_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            results=[],
        )
        backends = ["rms-direct", "rms-folded"]
        if args.reverse:
            backends.reverse()
        for c in cases:
            c = dict(c, rms_quant=dict(c["rms_quant"], threads=args.threads))
            for distribution in args.distributions.split(","):
                for backend in backends:
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
                                distribution=distribution,
                                status=r["status"],
                                prep_ms=r.get("stages", {})
                                .get("activation_prepare", {})
                                .get("median_ms"),
                                online_ms=r.get("stages", {})
                                .get("online_activation_plus_gemm", {})
                                .get("median_ms"),
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

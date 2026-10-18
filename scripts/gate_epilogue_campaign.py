"""Full two-projection chain comparison of gate placement, with immutable evidence."""

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
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.repeat < 3 or args.warmup < 1:
        p.error("repeat>=3 and warmup>=1")
    cases = [c for c in load("h3-gate-epilogue")["cases"] if args.filter in c["id"]]
    if not cases:
        p.error("no cases matched")
    with gpu_lock():
        doc = dict(
            schema_version=1,
            environment=environment(),
            results=[],
            campaign_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        backends = ["gate-in-norm", "gate-in-gemm"]
        if args.reverse:
            backends.reverse()
        for c in cases:
            for distribution in ["normal", "outliers"]:
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
                                core_ms=r.get("timing", {}).get("median_ms"),
                                stages_ms={
                                    k: v["median_ms"] for k, v in r.get("stages", {}).items()
                                },
                                parity=r.get("paired_path_parity"),
                                quality=r.get("quality_error"),
                                reason=r.get("reason"),
                            )
                        ),
                        flush=True,
                    )
        write_result(args.output, doc)
    return int(any(r["status"] in {"incorrect", "error"} for r in doc["results"]))


if __name__ == "__main__":
    raise SystemExit(main())

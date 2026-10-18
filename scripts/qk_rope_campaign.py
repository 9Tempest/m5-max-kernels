"""Serial staged/fused Q/K projection, RMSNorm and RoPE measurements."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from m5kernels.engine import environment, run_case, write_result
from m5kernels.locking import gpu_lock
from m5kernels.workloads import load


BASELINE_TILE = (128, 64, 4, 8)
DEFAULT_FUSED_TILE = (64, 128, 4, 8)


def configurations(sweep, tile=None):
    """One optimized staged baseline followed by explicit fused candidates."""
    candidates = (
        [(tm, tn, groups, 8) for tm in (32, 64, 128) for tn in (64, 128) for groups in (4, 8)]
        if sweep
        else [tuple(tile or DEFAULT_FUSED_TILE)]
    )
    return [("qk-staged", BASELINE_TILE)] + [("qk-fused", candidate) for candidate in candidates]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filter", default="m4096")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--output", required=True)
    p.add_argument("--sweep", action="store_true", help="run all 12 fused tile candidates")
    p.add_argument("--reverse", action="store_true", help="reverse baseline/candidate order")
    p.add_argument(
        "--tile",
        type=int,
        nargs=4,
        metavar=("TM", "TN", "GROUPS", "SWIZZLE"),
        help="fused candidate outside a sweep; staged baseline always uses 128 64 4 8",
    )
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.repeat < 3 or args.warmup < 1:
        p.error("repeat>=3 and warmup>=1")
    if args.seed < 0:
        p.error("seed must be nonnegative")
    if args.sweep and args.tile is not None:
        p.error("--tile only applies outside --sweep")
    cases = [c for c in load("h3-qk-rope")["cases"] if args.filter in c["id"]]
    if not cases:
        p.error("no cases matched")
    configs = configurations(args.sweep, args.tile)
    if args.reverse:
        configs.reverse()
    with gpu_lock():
        doc = dict(
            schema_version=1,
            environment=environment(),
            experiment="one Q or K projection; native-style BF16 boundaries; RMSNorm and RoPE",
            campaign_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            campaign=dict(
                suite="h3-qk-rope",
                filter=args.filter,
                seed=args.seed,
                warmup=args.warmup,
                repeat=args.repeat,
                sweep=args.sweep,
                distributions=["normal", "outliers"],
                configurations=[
                    dict(backend=backend, tile=list(tile)) for backend, tile in configs
                ],
                planned_results=len(cases) * 2 * len(configs),
                selection_only=args.sweep,
                promotion_requires="second seed and independent rerun on identical hardware",
            ),
            campaign_complete=False,
            results=[],
        )
        started = time.perf_counter()
        try:
            for case in cases:
                for distribution in ("normal", "outliers"):
                    for backend, tile in configs:
                        result = run_case(
                            case,
                            backend,
                            "int8",
                            args.seed,
                            distribution,
                            args.warmup,
                            args.repeat,
                            tile,
                        )
                        # run_case records compile failures/unsupported paths;
                        # every candidate remains in the immutable document.
                        doc["results"].append(result)
                        print(
                            json.dumps(
                                dict(
                                    case=case["id"],
                                    backend=backend,
                                    tile=list(tile),
                                    distribution=distribution,
                                    status=result["status"],
                                    core_ms=result.get("timing", {}).get("median_ms"),
                                    stages_ms={
                                        key: value["median_ms"]
                                        for key, value in result.get("stages", {}).items()
                                    },
                                    transform_pass=result.get("transform_pass"),
                                    implementation_error=result.get("implementation_error"),
                                    quality=result.get("quality_error"),
                                    quality_pass=result.get("quality_pass"),
                                    reason=result.get("reason"),
                                )
                            ),
                            flush=True,
                        )
            doc["campaign_complete"] = True
        finally:
            doc["campaign_elapsed_seconds"] = time.perf_counter() - started
            write_result(args.output, doc)
    return int(any(result["status"] in {"incorrect", "error"} for result in doc["results"]))


if __name__ == "__main__":
    raise SystemExit(main())

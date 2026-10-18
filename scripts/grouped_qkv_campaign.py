"""Serial pinned-native/candidate grouped QKV measurements on identical INT8 operands."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from m5kernels.engine import environment, run_case, write_result
from m5kernels.grouped_qkv import generate
from m5kernels.locking import gpu_lock
from m5kernels.workloads import load


# Native uses its own compact Morton dispatch; the fourth field is only a
# metadata placeholder, not a request to alter that implementation's swizzle.
NATIVE_TILE = (128, 128, 8, 1)
DEFAULT_CANDIDATE_TILE = (64, 128, 4, 8)
DISTRIBUTIONS = ("normal", "outliers", "zeros", "tiny")


def configurations(sweep, tile=None):
    """One unchanged native baseline, then one or all 12 candidate tiles."""
    candidates = (
        [(tm, tn, groups, 8) for tm in (32, 64, 128) for tn in (64, 128) for groups in (4, 8)]
        if sweep
        else [tuple(tile or DEFAULT_CANDIDATE_TILE)]
    )
    return [("native-qkv", NATIVE_TILE)] + [
        ("candidate-qkv", candidate) for candidate in candidates
    ]


def parse_distributions(value):
    distributions = [item.strip() for item in value.split(",")]
    if any(item not in DISTRIBUTIONS for item in distributions):
        raise ValueError(
            "distribution must be a comma-separated subset of " + ",".join(DISTRIBUTIONS)
        )
    if len(set(distributions)) != len(distributions):
        raise ValueError("distribution entries must be unique")
    return distributions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filter", default="m4096")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--output", required=True)
    p.add_argument("--sweep", action="store_true", help="run all 12 candidate tile configurations")
    p.add_argument("--reverse", action="store_true", help="reverse baseline/candidate order")
    p.add_argument("--rms-order", choices=("simd", "native"), default="simd")
    p.add_argument(
        "--scratch",
        choices=("threadgroup", "device"),
        default="threadgroup",
        help="candidate BF16 projection temporary storage",
    )
    p.add_argument(
        "--k-specialization",
        choices=("dynamic", "native"),
        default="dynamic",
        help="candidate runtime K, or native-style static5376/static128 reduction slices",
    )
    p.add_argument(
        "--distribution",
        default="normal,outliers",
        help="comma-separated distributions: normal,outliers,zeros,tiny",
    )
    p.add_argument(
        "--tile",
        type=int,
        nargs=4,
        metavar=("TM", "TN", "GROUPS", "SWIZZLE"),
        help="candidate outside a sweep; native kernel and dispatch remain unchanged",
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
    try:
        distributions = parse_distributions(args.distribution)
    except ValueError as e:
        p.error(str(e))
    cases = [
        dict(
            case,
            rms_order=args.rms_order,
            k_specialization=args.k_specialization,
            scratch=args.scratch,
        )
        for case in load("h3-grouped-qkv")["cases"]
        if args.filter in case["id"]
    ]
    if not cases:
        p.error("no cases matched")
    configs = configurations(args.sweep, args.tile)
    if args.reverse:
        configs.reverse()
    try:
        candidate_sources = [
            dict(
                tile=list(tile),
                rms_order=args.rms_order,
                k_specialization=args.k_specialization,
                input_dim=input_dim,
                scratch=args.scratch,
                source_sha256=hashlib.sha256(
                    generate(
                        tile,
                        rms_order=args.rms_order,
                        k_specialization=args.k_specialization,
                        input_dim=input_dim,
                        scratch=args.scratch,
                    ).encode()
                ).hexdigest(),
            )
            for backend, tile in configs
            if backend == "candidate-qkv"
            for input_dim in sorted({case["k"] for case in cases})
        ]
    except (ValueError, NotImplementedError) as e:
        p.error(str(e))
    campaign = dict(
        suite="h3-grouped-qkv",
        filter=args.filter,
        cases=cases,
        precision="int8",
        seed=args.seed,
        warmup=args.warmup,
        repeat=args.repeat,
        sweep=args.sweep,
        reverse=args.reverse,
        rms_order=args.rms_order,
        k_specialization=args.k_specialization,
        scratch=args.scratch,
        distributions=distributions,
        configurations=[dict(backend=backend, tile=list(tile)) for backend, tile in configs],
        candidate_sources=candidate_sources,
        native_tile_metadata="128x128, eight SIMD groups; slot four is a placeholder, not swizzle",
        planned_results=len(cases) * len(distributions) * len(configs),
        selection_only=args.sweep,
        promotion_requires="second seed and independent rerun on identical hardware",
    )
    with gpu_lock():
        doc = dict(
            schema_version=1,
            environment=environment(),
            experiment="grouped QKV; unchanged pinned native baseline; identical native INT8 quantization; no Hadamard",
            campaign_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            campaign_config_sha256=hashlib.sha256(
                json.dumps(campaign, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            campaign=campaign,
            campaign_complete=False,
            results=[],
        )
        started = time.perf_counter()
        try:
            for case in cases:
                for distribution in distributions:
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
                        # Preserve failures as returned by the shared engine;
                        # a failed compile never silently removes a candidate.
                        doc["results"].append(result)
                        print(
                            json.dumps(
                                dict(
                                    case=case["id"],
                                    backend=backend,
                                    tile=list(tile),
                                    rms_order=args.rms_order,
                                    k_specialization=args.k_specialization,
                                    scratch=args.scratch,
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
        except BaseException as e:
            doc["campaign_failure"] = dict(type=type(e).__name__, reason=str(e)[:5000])
            raise
        finally:
            doc["campaign_elapsed_seconds"] = time.perf_counter() - started
            write_result(args.output, doc)
    return int(any(result["status"] in {"incorrect", "error"} for result in doc["results"]))


if __name__ == "__main__":
    raise SystemExit(main())

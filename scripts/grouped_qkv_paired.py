"""Alternating native/candidate online QKV timing on one resident Metal runner."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from m5kernels.engine import (
    NumericalMismatch,
    all_finite,
    environment,
    evaluate,
    metrics,
    write_result,
)
from m5kernels.grouped_qkv_experiment import NativeCall, prepare
from m5kernels.locking import gpu_lock
from m5kernels.workloads import load

MODES = ("native", "candidate")


def pair_order(index, reverse=False):
    return MODES[::-1] if (index + int(reverse)) % 2 else MODES


def summarize(samples):
    """Preserve every raw sample, with summaries only for available values."""
    available = [value for value in samples if value is not None]
    result = dict(samples_ms=samples, available_samples=len(available), total_samples=len(samples))
    if available:
        result.update(
            median_ms=float(np.median(available)),
            p10_ms=float(np.percentile(available, 10)),
            p90_ms=float(np.percentile(available, 90)),
            min_ms=float(np.min(available)),
        )
    return result


def _passes(error, limit):
    return error["finite"] and error["relative_l2"] <= limit


def run(args):
    """Prepare once, check both online paths, and retain failures as immutable JSON."""
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.repeat < 3 or args.warmup < 1 or args.seed < 0:
        raise ValueError("repeat>=3, warmup>=1 and nonnegative seed required")
    cases = [case for case in load("h3-grouped-qkv")["cases"] if args.filter in case["id"]]
    if len(cases) != 1:
        raise ValueError("--filter must select exactly one grouped QKV case")
    case = dict(
        cases[0],
        k_specialization=args.k_specialization,
        scratch=args.scratch,
        rms_order=args.rms_order,
    )
    configuration = dict(
        suite="h3-grouped-qkv",
        case=case,
        seed=args.seed,
        distribution="normal",
        precision="int8",
        candidate_tile=list(args.tile),
        scope="online",
        warmup_per_backend=args.warmup,
        measured_pairs=args.repeat,
        reverse_initial=args.reverse,
        order="first backend alternates every pair; same runner and resident buffers throughout",
        primary_timing="synchronized host wall around NativeCall plus engine.evaluate only",
        supplemental_timing="GPU command start/end duration, read after the host timer",
        preparation_calls=1,
        offline_weight_preparation_in_online=False,
    )
    with gpu_lock():
        document = dict(
            schema_version=1,
            experiment="same-runner alternating native/candidate grouped QKV online cross-check",
            environment=environment(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(
                json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            configuration=configuration,
            status="pending",
            measurement_complete=False,
            results={},
            pairs=[],
            promotion_requires="second seed and independent rerun on identical hardware",
        )
        runner = None
        started = time.perf_counter()
        try:
            core, check, expected, original, _, metadata = prepare(
                case, "candidate-qkv", "int8", args.seed, "normal", tuple(args.tile)
            )
            runner = core.runner
            document["preparation_metadata"] = metadata
            document["transform_pass"] = bool(metadata.get("transform_pass", False))
            if not document["transform_pass"]:
                raise NumericalMismatch("preparation or paired implementation check failed")
            functions = {mode: NativeCall(runner, mode, "online") for mode in MODES}
            # Read and check outputs immediately, before the next path overwrites them.
            for mode in MODES:
                output = functions[mode]()
                evaluate(output)
                actual = check(output)
                implementation = metrics(actual, expected)
                quality = metrics(actual, original)
                finite = all_finite(output)
                stream_quality = metadata["per_stream_quality_error"][mode]
                quality_pass = bool(
                    _passes(quality, 0.05)
                    and all(_passes(error, 0.05) for error in stream_quality.values())
                )
                correct = bool(finite and _passes(implementation, 0.008))
                document["results"][mode] = dict(
                    backend=f"{mode}-qkv",
                    status="ok" if correct else "incorrect",
                    implementation_error=implementation,
                    output_finite=bool(finite),
                    per_stream_implementation_error=metadata["per_stream_implementation_error"][
                        mode
                    ],
                    quality_error=quality,
                    per_stream_quality_error=stream_quality,
                    quality_pass=quality_pass,
                    quality_status="accepted" if quality_pass else "quality_rejected",
                    implementation_relative_l2_limit=0.008,
                    quality_relative_l2_limit=0.05,
                    timing=dict(kind="synchronized_host_wall", samples_ms=[]),
                    gpu_command=dict(kind="supplemental_gpu_command_start_end", samples_ms=[]),
                )
            if any(result["status"] != "ok" for result in document["results"].values()):
                raise NumericalMismatch("online output failed independent correctness checks")
            # Both paths are warmed on the same inputs, serially, with alternating order.
            for index in range(args.warmup):
                for mode in pair_order(index, args.reverse):
                    evaluate(functions[mode]())
            for index in range(args.repeat):
                pair = dict(index=index, order=list(pair_order(index, args.reverse)))
                document["pairs"].append(pair)
                for mode in pair["order"]:
                    function = functions[mode]
                    start_ns = time.perf_counter_ns()
                    output = function()
                    evaluate(output)
                    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                    # This accessor is deliberately outside the primary timer.
                    gpu_seconds = float(runner.last_gpu_seconds)
                    gpu_ms = (
                        gpu_seconds * 1000 if np.isfinite(gpu_seconds) and gpu_seconds > 0 else None
                    )
                    result = document["results"][mode]
                    result["timing"]["samples_ms"].append(elapsed_ms)
                    result["gpu_command"]["samples_ms"].append(gpu_ms)
                    pair[f"{mode}_wall_ms"] = elapsed_ms
                    pair[f"{mode}_gpu_command_ms"] = gpu_ms
                pair["native_over_candidate_wall_ratio"] = (
                    pair["native_wall_ms"] / pair["candidate_wall_ms"]
                )
            document["measurement_complete"] = True
            document["status"] = "ok"
            document["quality_status"] = (
                "accepted"
                if all(result["quality_pass"] for result in document["results"].values())
                else "quality_rejected"
            )
        except NumericalMismatch as exc:
            document.update(status="incorrect", reason=str(exc), **exc.diagnostics)
        except NotImplementedError as exc:
            document.update(status="unsupported", reason=str(exc))
        except Exception as exc:
            document.update(status="error", error_type=type(exc).__name__, reason=str(exc)[:5000])
        except BaseException as exc:
            document.update(
                status="interrupted", error_type=type(exc).__name__, reason=str(exc)[:5000]
            )
            raise
        finally:
            for result in document["results"].values():
                for field in ("timing", "gpu_command"):
                    result[field].update(summarize(result[field]["samples_ms"]))
                result["timing"].update(warmup=args.warmup, requested_repeat=args.repeat)
                result["gpu_command"]["primary_metric"] = False
            ratios = [
                pair["native_over_candidate_wall_ratio"]
                for pair in document["pairs"]
                if "native_over_candidate_wall_ratio" in pair
            ]
            if ratios:
                document["paired_speed_ratio"] = dict(
                    definition="native synchronized wall / candidate synchronized wall; >1 favors candidate",
                    samples=ratios,
                    median=float(np.median(ratios)),
                    p10=float(np.percentile(ratios, 10)),
                    p90=float(np.percentile(ratios, 90)),
                    ratio_of_medians=(
                        document["results"]["native"]["timing"]["median_ms"]
                        / document["results"]["candidate"]["timing"]["median_ms"]
                    ),
                )
            document["elapsed_seconds"] = time.perf_counter() - started
            try:
                write_result(args.output, document)
            finally:
                if runner is not None:
                    runner.close()
        return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", default="m128", help="substring selecting exactly one case")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--repeat", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--tile", type=int, nargs=4, default=[64, 128, 4, 8])
    parser.add_argument(
        "--reverse", action="store_true", help="candidate runs first in the first pair"
    )
    parser.add_argument("--k-specialization", choices=("dynamic", "native"), default="dynamic")
    parser.add_argument("--scratch", choices=("threadgroup", "device"), default="threadgroup")
    parser.add_argument("--rms-order", choices=("simd", "native"), default="simd")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        document = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            dict(
                output=args.output,
                status=document["status"],
                measurement_complete=document["measurement_complete"],
                quality_status=document.get("quality_status"),
                paired_speed_ratio=document.get("paired_speed_ratio"),
            )
        ),
        flush=True,
    )
    return 0 if document["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

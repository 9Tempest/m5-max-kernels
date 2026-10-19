"""Pinned native gate epilogue numerical checks and paired warmed online timing.

No benchmark starts without the shared GPU lock. Each case retains its raw
driver report and log, including failures; the campaign never overwrites data.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import run as base
import run_fc2_experiment as experiment

PATCH_SHA256 = "cbd7239d98dffac8feb469caa7fdeee504a564dda51f625f95406284b2a806fe"
SOURCE_SHA256 = "83ef61ace55a0ca9d8c2735d627e8c0ca96692bfebaae2376fae3b6fd9dbef7b"
DRIVER_SHA256 = "6105dad5dd401aea70f4db1ab153b49bb6817a315c1ff314741d2e24e22bce30"
REFERENCE_SHA256 = "18e1a880800384ec20a5279f16ef6cacdee70a2b61b037cf5d04d5886e5684d7"
CHANGED_FILES = {"h3_dit.c", "h3_gpu.h", "h3_gpu.m", "h3_shaders.metal"}
CASES = tuple(itertools.product((17, 29), (2049, 9914), ("h256", "attention", "group1024")))
COUNTERS = (
    "nonfinite",
    "cpu_original_mismatches",
    "cpu_fused_mismatches",
    "hidden_bit_mismatches",
    "quantized_code_mismatches",
    "quantized_scale_bit_mismatches",
    "zero_row_errors",
    "padding_errors",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=base.ROOT / ".cache/h3.c-fused")
    p.add_argument("--model-dir", type=Path, default=base.ROOT / ".cache/models/MiniMax-H3")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    return p


def validate_source(source, provenance):
    if source.resolve() == (base.ROOT / ".cache/h3.c").resolve():
        raise ValueError("fusion experiments need a separate checkout")
    changed = set(
        filter(
            None,
            base.read_command(["git", "diff", "HEAD", "--name-only", "-z"], source).split("\0"),
        )
    )
    if (
        provenance["revision"] != base.UPSTREAM_REVISION
        or provenance["source_sha256"] != SOURCE_SHA256
        or changed != CHANGED_FILES
        or any(line.startswith("??") for line in provenance["status"].splitlines())
        or base.sha256(Path(__file__).with_name("fused.patch")) != PATCH_SHA256
    ):
        raise ValueError("runtime must exactly match the pinned combined fused.patch")
    for name, digest in (
        ("fused_validate.c", DRIVER_SHA256),
        ("fused_reference.h", REFERENCE_SHA256),
    ):
        if base.sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"frozen numerical driver source changed: {name}")
    return {"patch_sha256": PATCH_SHA256, "source_sha256": SOURCE_SHA256}


def number(value, name, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be a finite {'positive' if positive else 'nonnegative'} number"
        )
    return value


def validate_metric(metric, count, name, implementation):
    if type(metric) is not dict or type(metric.get("count")) is not int or metric["count"] != count:
        raise ValueError(f"incomplete independent reference: {name}")
    if type(metric.get("nonfinite")) is not int or metric["nonfinite"] != 0:
        raise base.NumericalError(f"nonfinite independent reference: {name}")
    if metric.get("relative_l2") is None:
        raise base.NumericalError(f"undefined independent relative error: {name}")
    for key in ("relative_l2", "rmse", "max_abs"):
        number(metric.get(key), f"{name}.{key}")
    if implementation and metric["relative_l2"] > 0.008:
        raise base.NumericalError(f"independent implementation gate failed: {name}")


def validate_case(report, seed, rows, policy, warmups, repeats):
    k = 7168 if policy == "attention" else 14336
    expected = {
        "schema": 2,
        "policy": policy,
        "rows": rows,
        "padded_rows": (rows + 127) & ~127,
        "seed": seed,
        "K": k,
        "N": 5376,
        "in_place_residual": True,
        "gate_slot": 2 if policy == "attention" else 5,
        "distinct_next_block_modulation": policy != "attention",
        "weight_key": f"blocks.0.{'attn.out_proj' if policy == 'attention' else 'mlp.fc2'}.weight",
        "scope": "real_weights_synthetic_activations_paired_online",
    }
    for name, value in expected.items():
        if type(report.get(name)) is not type(value) or report[name] != value:
            raise ValueError(f"driver recipe mismatch: {name}")
    for name in COUNTERS:
        if type(report.get(name)) is not int or report[name] < 0:
            raise ValueError(f"missing or malformed numerical counter: {name}")
        if report[name]:
            raise base.NumericalError(f"paired/finite/zero/tail contract failed: {name}")
    ref = report.get("independent_reference", {})
    if (
        ref.get("rows") != [0, rows // 2, rows - 1]
        or any(type(row) is not int for row in ref.get("rows", []))
        or ref.get("columns") != 5376
        or ref.get("full_K") != k
        or ref.get("relative_l2_implementation_limit") != 0.008
        or type(ref.get("reference_failures")) is not int
        or ref["reference_failures"] != 0
        or ref.get("passed") is not True
    ):
        raise base.NumericalError("independent complete-row/full-K reference missing or failed")
    for key in (
        "projection_implementation_vs_rounded_decoded",
        "hidden_implementation",
        "next_quantized_implementation",
        "projection_quantization_vs_original_bf16",
        "hidden_quantization_vs_original_bf16_path",
        "next_quantization_vs_original_bf16_path",
    ):
        validate_metric(ref.get(key), 3 * 5376, key, "implementation" in key)
    per_row = ref.get("per_row", [])
    if len(per_row) != 3 or [row.get("row") for row in per_row] != [0, rows // 2, rows - 1]:
        raise ValueError("independent per-row metrics must cover all selected rows")
    for row in per_row:
        for key in (
            "projection_implementation",
            "projection_quantization",
            "hidden_implementation",
            "next_quantized_implementation",
        ):
            validate_metric(row.get(key), 5376, key, "implementation" in key)
    if report.get("passed") is not True:
        raise base.NumericalError("driver numerical contract did not pass")
    timing = report.get("timing", {})
    first = "fused" if (warmups + bool(seed & 4)) & 1 else "native"
    wanted = dict(
        status="completed",
        warmups_per_variant=warmups,
        repeats_per_variant=repeats,
        scope="online_activation_pack_projection_gate_next_adaln_quant_submit_wait",
        paired_order="alternates_native_first_fused_first",
        first_measured_variant=first,
    )
    if any(type(timing.get(k)) is not type(v) or timing[k] != v for k, v in wanted.items()):
        raise ValueError("incomplete or mismatched warmed paired timing recipe")
    if timing.get("excluded") != [
        "loading",
        "offline_weight_preparation",
        "shader_pipeline_compile_and_warmup",
        "correctness_readback",
        "residual_restore",
    ]:
        raise ValueError("timing exclusions changed")
    summary = {}
    for variant in ("native", "fused"):
        samples = timing.get(variant)
        if type(samples) is not list or len(samples) != repeats:
            raise ValueError("missing raw repeated timing samples")
        for sample in samples:
            for key in (
                "wall_seconds",
                "cpu_encode_seconds",
                "submit_wait_seconds",
                "supplemental_gpu_seconds",
            ):
                number(sample.get(key), key, key in ("wall_seconds", "submit_wait_seconds"))
            if (
                abs(
                    sample["wall_seconds"]
                    - sample["cpu_encode_seconds"]
                    - sample["submit_wait_seconds"]
                )
                > 1e-8
            ):
                raise ValueError("wall/encode/wait timestamps do not form one interval")
            if type(sample.get("direct_dispatches")) is not int or sample["direct_dispatches"] != 3:
                raise ValueError("online scope must contain exactly three direct dispatches")
        summary[variant] = {
            "wall_median_seconds": statistics.median(sample["wall_seconds"] for sample in samples),
            "gpu_timestamps_available": all(
                sample["supplemental_gpu_seconds"] > 0 for sample in samples
            ),
            "raw": samples,
        }
    summary["observed_native_over_fused_wall_ratio"] = (
        summary["native"]["wall_median_seconds"] / summary["fused"]["wall_median_seconds"]
    )
    return summary


def compile_driver(source, output, env, record):
    result = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
    record["measurements"]["build"] = result
    if result["returncode"]:
        raise base.UnsupportedError("native fused source compile failed")
    driver = output / "fused-validate"
    command = [
        "clang",
        "-std=c11",
        "-O2",
        "-D_DARWIN_C_SOURCE",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(source),
        str(Path(__file__).with_name("fused_validate.c")),
        str(source / "libh3.a"),
    ]
    for framework in (
        "Foundation",
        "Metal",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
        "Accelerate",
    ):
        command.extend(("-framework", framework))
    command.extend(("-licucore", "-lm", "-o", str(driver)))
    result = base.run_logged(command, source, env, output, "driver-build")
    record["measurements"]["driver_build"] = result
    if result["returncode"]:
        raise base.UnsupportedError("private fused validation driver compile failed")
    validate_source(source, base.git_provenance(source))
    record["driver"] = {
        "binary_sha256": base.sha256(driver),
        "source_sha256": DRIVER_SHA256,
        "reference_sha256": REFERENCE_SHA256,
        "library_sha256": base.sha256(source / "libh3.a"),
    }
    return driver


def classify(error):
    return (
        "incorrect"
        if isinstance(error, base.NumericalError)
        else "unsupported"
        if isinstance(error, base.UnsupportedError)
        else "failed"
    )


def execute(args):
    output, source, model = (
        args.output_dir.resolve(),
        args.source_dir.resolve(),
        args.model_dir.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(
        schema_version=1,
        status="failed",
        started_utc=base.utc_now(),
        measurements={},
        cases=[],
        scope="real block-0 weights; synthetic activations; paired online path; not model throughput",
        quality="not evaluated",
        baseline="same frozen source and quantized weights, fusion disabled",
    )
    try:
        if not 2 <= args.warmups <= 100 or not 3 <= args.repeats <= 1000:
            raise ValueError("warmups must be 2..100 and repeats must be 3..1000")
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            record["harness"] = base.git_provenance(base.ROOT)
            record["wrapper_sha256"] = base.sha256(Path(__file__))
            record["upstream"] = base.git_provenance(source)
            record["patch"] = validate_source(source, record["upstream"])
            env, overrides = base.runtime_environment()
            env.update(overrides)
            record["runtime_overrides"] = overrides
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise base.UnsupportedError("Metal validation unavailable on this host")
            record["environment"] = dict(
                system=platform.system(),
                release=platform.release(),
                machine=platform.machine(),
                python=platform.python_version(),
                chip=base.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                memory_bytes=int(base.read_command(["sysctl", "-n", "hw.memsize"])),
                compiler=base.read_command(["clang", "--version"]),
                xcode=base.read_command(["xcodebuild", "-version"]),
            )
            if "M5" not in record["environment"]["chip"]:
                raise base.UnsupportedError("native fusion validation requires Apple M5 TensorOps")
            driver = compile_driver(source, output, env, record)
            record["model"] = base.model_inventory(
                model, Path(__file__).with_name("model-files.json")
            )
            base.write_json(
                output / "request.json",
                record
                | {"cases_requested": CASES, "warmups": args.warmups, "repeats": args.repeats},
            )
            for seed, rows, policy in CASES:
                directory = output / f"{policy}-m{rows}-seed{seed}"
                directory.mkdir()
                case = dict(seed=seed, rows=rows, policy=policy, status="failed")
                try:
                    command = [
                        str(driver),
                        str(model / "FL2VA/transformer"),
                        str(source / "h3_shaders.metal"),
                        str(rows),
                        str(seed),
                        policy,
                        str(directory / "driver.json"),
                        str(args.warmups),
                        str(args.repeats),
                    ]
                    case["process"] = base.run_logged(command, source, env, directory, "validation")
                    if case["process"]["fallback_detected"]:
                        raise base.UnsupportedError("TensorOps fallback detected")
                    path = directory / "driver.json"
                    if path.exists():
                        case["driver_report_sha256"] = base.sha256(path)
                    report = experiment.load_json(path)
                    case["summary"] = validate_case(
                        report, seed, rows, policy, args.warmups, args.repeats
                    )
                    if case["process"]["returncode"]:
                        raise ValueError(f"driver exited {case['process']['returncode']}")
                    case["status"] = "completed"
                except KeyboardInterrupt:
                    case.update(status="interrupted", error="interrupted by user")
                    raise
                except Exception as error:
                    case.update(status=classify(error), error=f"{type(error).__name__}: {error}")
                finally:
                    base.write_json(directory / "case.json", case)
                    record["cases"].append(case | {"directory": directory.name})
            validate_source(source, base.git_provenance(source))
            record["status"] = (
                "completed"
                if all(case["status"] == "completed" for case in record["cases"])
                else "failed"
            )
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupted by user")
    except Exception as error:
        record.update(status=classify(error), error=f"{type(error).__name__}: {error}")
    finally:
        record.update(
            finished_utc=base.utc_now(), harness_wall_seconds=time.perf_counter() - started
        )
        base.write_json(output / "campaign.json", record)
    return record


def main():
    args = parser().parse_args()
    try:
        record = execute(args)
    except FileExistsError:
        print("choose a new immutable output directory", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": record["status"],
                "record": base.path_label(args.output_dir / "campaign.json"),
            }
        )
    )
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

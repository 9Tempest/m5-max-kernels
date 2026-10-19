"""Immutable, serialized validation of QKV/FC1 AdaLN/H256 preparation."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import platform
import shutil
import statistics
import time

import kernel_integration as contract
import run as base
import run_fc2_experiment as records
from run_fused_campaign import classify, number, validate_metric


COUNTERS = (
    "nonfinite",
    "gate_errors",
    "padding_errors",
    "zero_errors",
    "staged_fused_code_mismatches",
    "staged_fused_scale_mismatches",
    "staged_fused_adaln_mismatches",
)


def require_device(chip):
    if "M5" not in chip:
        raise base.UnsupportedError("preparation validation requires an Apple M5 device")


def check_process_support(process, log):
    if process["fallback_detected"] or any(
        marker in log
        for marker in (
            "TensorOps unavailable",
            "cannot initialize Metal",
            "TensorOps compile failed:",
        )
    ):
        raise base.UnsupportedError("requested Metal/TensorOps path is unavailable; no fallback")


def validate(report, rows, seed, gate, distinct, warmups, repeats):
    for key, value in dict(
        schema="h3_prep_validation_v1",
        rows=rows,
        padded_rows=(rows + 127) & ~127,
        K=5376,
        seed=seed,
        gate=gate,
        distinct_maps=distinct,
    ).items():
        contract.exact(report.get(key), value, "prep " + key)
    if report.get("passed") is not True:
        raise base.NumericalError("preparation driver rejected its numerical contract")
    for name in COUNTERS:
        if type(report.get(name)) is not int or report[name] != 0:
            raise base.NumericalError("prep counter failed: " + name)
    ref = report.get("reference", {})
    for key, value in dict(
        full_width=5376, implementation_limit=0.008, rows=[0, 1, rows // 2, rows - 1]
    ).items():
        contract.exact(ref.get(key), value, "prep reference " + key)
    variants = ref.get("variants", [])
    contract.exact(
        [v.get("name") for v in variants], ["native", "staged", "fused"], "prep reference variants"
    )
    for index, variant in enumerate(variants):
        validate_metric(
            variant.get("quantization_vs_original_adaln"), 4 * 5376, "quantization", False
        )
        if index:
            for key in ("implementation_vs_cpu_quantized", "adaln_implementation_vs_fp64"):
                validate_metric(variant.get(key), 4 * 5376, key, True)
            per_row = variant.get("per_row_implementation", [])
            if len(per_row) != 4:
                raise ValueError("missing independent per-row reference")
            for metric in per_row:
                validate_metric(metric, 5376, "per-row implementation", True)
    timing = report.get("timing", {})
    for key, value in dict(
        status="completed",
        scope="preparation_only_synchronized_online_including_padding",
        warmups=warmups,
        repeats=repeats,
        rotation_seed_offset=(seed >> 2) & 1,
    ).items():
        contract.exact(timing.get(key), value, "prep timing " + key)
    variants = timing.get("variants", [])
    contract.exact(
        [v.get("name") for v in variants], ["native", "staged", "fused"], "prep timed variants"
    )
    summary = {}
    for index, variant in enumerate(variants):
        samples = variant.get("samples")
        if not isinstance(samples, list) or len(samples) != repeats:
            raise ValueError("incomplete raw timing samples")
        dispatches = (1 + int(bool(gate and distinct)), 2 + gate, 1)[index]
        for sample in samples:
            number(sample.get("wall_seconds"), "wall time", True)
            number(sample.get("supplemental_gpu_seconds"), "GPU time")
            contract.exact(sample.get("direct_dispatches"), dispatches, "dispatches")
        summary[variant["name"]] = dict(
            wall_median_seconds=statistics.median(s["wall_seconds"] for s in samples), raw=samples
        )
    summary["native_over_fused"] = (
        summary["native"]["wall_median_seconds"] / summary["fused"]["wall_median_seconds"]
    )
    summary["staged_over_fused"] = (
        summary["staged"]["wall_median_seconds"] / summary["fused"]["wall_median_seconds"]
    )
    return summary


def compile_driver(source, output, env, record):
    build = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
    record["build"] = build
    if build["returncode"]:
        raise base.UnsupportedError("native source compilation failed")
    driver = output / "prep-validate"
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
        str(output / "sources/prep_validate.c"),
        str(source / "libh3.a"),
    ]
    for name in (
        "Foundation",
        "Metal",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
        "Accelerate",
    ):
        command += ["-framework", name]
    command += ["-licucore", "-lm", "-o", str(driver)]
    record["driver_build"] = base.run_logged(command, source, env, output, "driver-build")
    if record["driver_build"]["returncode"]:
        raise base.UnsupportedError("preparation driver compilation failed")
    record["binary_sha256"] = base.sha256(driver)
    record["library_sha256"] = base.sha256(source / "libh3.a")
    return driver


def execute(args):
    output, source = args.output_dir.resolve(), args.source_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="h3_prep_campaign_v1", status="failed", started_utc=base.utc_now(), cases=[]
    )
    start = time.perf_counter()
    try:
        if not 3 <= args.warmups <= 100 or not 3 <= args.repeats <= 1000:
            raise ValueError("warmups >=3 and repeats >=3 required")
        if any(r < 17 or r > 18000 for r in args.rows) or any(
            s < 0 or s > 2**32 - 1 for s in args.seeds
        ):
            raise ValueError("rows/seeds outside driver contract")
        pin = contract.load_pin(args.source_manifest.resolve(), args.source_manifest_sha256)
        record["source_manifest_sha256"] = args.source_manifest_sha256
        record["source_manifest"] = pin
        snapshot = output / "sources"
        snapshot.mkdir()
        files = (
            "prep_validate.c",
            "fused_reference.h",
            "run_prep_campaign.py",
            "kernel_integration.py",
            "run.py",
            "run_fused_campaign.py",
            "run_fc2_experiment.py",
        )
        record["harness_files"] = {}
        for name in files:
            path = Path(__file__).with_name(name)
            shutil.copyfile(path, snapshot / name)
            record["harness_files"][name] = base.sha256(snapshot / name)
        shutil.copyfile(args.source_manifest, snapshot / "source-manifest.json")
        for name in pin["changed_files"]:
            shutil.copyfile(source / name, snapshot / name)
        for patch in pin["patches"]:
            shutil.copyfile(args.source_manifest.parent / patch["file"], snapshot / patch["file"])
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            record["upstream"] = contract.validate_source(source, pin)
            for name in pin["changed_files"]:
                if base.sha256(snapshot / name) != base.sha256(source / name):
                    raise ValueError("native source snapshot changed during capture: " + name)
            record["harness"] = base.git_provenance(base.ROOT)
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise base.UnsupportedError("Metal validation unavailable on this host")
            record["environment"] = dict(
                chip=base.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                memory_bytes=int(base.read_command(["sysctl", "-n", "hw.memsize"])),
                os=platform.platform(),
                compiler=base.read_command(["clang", "--version"]),
                xcode=base.read_command(["xcodebuild", "-version"]),
            )
            require_device(record["environment"]["chip"])
            env, overrides = base.runtime_environment()
            record["runtime_overrides"] = overrides
            driver = compile_driver(source, output, env, record)
            cases = list(itertools.product(args.seeds, args.rows, ((0, 0), (1, 0), (1, 1))))
            if args.reverse:
                cases.reverse()
            base.write_json(
                output / "request.json",
                record
                | dict(
                    cases_requested=cases,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    reverse=args.reverse,
                ),
            )
            for seed, rows, (gate, distinct) in cases:
                contract.validate_source(source, pin)
                directory = output / f"m{rows}-seed{seed}-gate{gate}-maps{distinct}"
                directory.mkdir()
                case = dict(
                    seed=seed, rows=rows, gate=gate, distinct_maps=distinct, status="failed"
                )
                try:
                    command = [
                        str(driver),
                        str(source / "h3_shaders.metal"),
                        str(rows),
                        str(seed),
                        str(gate),
                        str(distinct),
                        str(directory / "driver.json"),
                        str(args.warmups),
                        str(args.repeats),
                    ]
                    case["process"] = base.run_logged(
                        command, source, env, directory, "validation", stop_on_fallback=True
                    )
                    path = directory / "driver.json"
                    if path.exists():
                        case["driver_sha256"] = base.sha256(path)
                    check_process_support(
                        case["process"], (directory / "validation.log").read_text()
                    )
                    case["summary"] = validate(
                        records.load_json(path),
                        rows,
                        seed,
                        gate,
                        distinct,
                        args.warmups,
                        args.repeats,
                    )
                    if case["process"]["returncode"]:
                        raise ValueError("native validation process failed")
                    case["status"] = "completed"
                except Exception as error:
                    case.update(status=classify(error), error=f"{type(error).__name__}: {error}")
                finally:
                    base.write_json(directory / "case.json", case)
                    record["cases"].append(case | dict(directory=directory.name))
                print(
                    f"prep m={rows} seed={seed} gate={gate} maps={distinct}: {case['status']}",
                    flush=True,
                )
            contract.validate_source(source, pin)
            for name, digest in record["harness_files"].items():
                if base.sha256(Path(__file__).with_name(name)) != digest:
                    raise ValueError("harness changed during campaign: " + name)
            record["status"] = (
                "completed"
                if all(c["status"] == "completed" for c in record["cases"])
                else "failed"
            )
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupted")
    except Exception as error:
        record.update(status=classify(error), error=f"{type(error).__name__}: {error}")
    finally:
        record.update(finished_utc=base.utc_now(), wall_seconds=time.perf_counter() - start)
        base.write_json(output / "campaign.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=base.ROOT / ".cache/h3.c-kernel-integration"
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[17, 2049, 9914, 17455])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    args = parser.parse_args()
    result = execute(args)
    print(result["status"])
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

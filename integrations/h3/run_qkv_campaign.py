"""Pinned native-versus-custom QKV/RMS/RoPE numerical and latency campaign.

The native and candidate kernels receive identical decoded operands. Native
INT8 and H256 INT8 are separate policies. Model throughput is measured elsewhere.
"""

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
from run_fused_campaign import classify


def compile_driver(source, output, env, record):
    record["build"] = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
    if record["build"]["returncode"]:
        raise base.UnsupportedError("native compilation failed")
    driver = output / "qkv-validate"
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
        str(output / "sources/qkv_validate.c"),
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
        raise base.UnsupportedError("QKV validation driver compilation failed")
    record["binary_sha256"] = base.sha256(driver)
    record["library_sha256"] = base.sha256(source / "libh3.a")
    return driver


def execute(args):
    from validate_qkv_report import validate

    output, source, model = (
        args.output_dir.resolve(),
        args.source_dir.resolve(),
        args.model_dir.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="h3_qkv_campaign_v1", status="failed", started_utc=base.utc_now(), cases=[]
    )
    start = time.perf_counter()
    try:
        if not 3 <= args.warmups <= 100 or not 3 <= args.repeats <= 1000:
            raise ValueError("warmups >=3 and repeats >=3 required")
        if len(set(args.rows)) != len(args.rows) or len(set(args.seeds)) != len(args.seeds):
            raise ValueError("duplicate cases are forbidden")
        pin = contract.load_pin(args.source_manifest.resolve(), args.source_manifest_sha256)
        record["source_manifest_sha256"] = args.source_manifest_sha256
        record["source_manifest"] = pin
        snapshot = output / "sources"
        snapshot.mkdir()
        files = (
            "qkv_validate.c",
            "qkv_reference.h",
            "run_qkv_campaign.py",
            "validate_qkv_report.py",
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
            record["model"] = base.model_inventory(
                model, Path(__file__).with_name("model-files.json")
            )
            env, overrides = base.runtime_environment()
            record["runtime_overrides"] = overrides
            driver = compile_driver(source, output, env, record)
            cases = list(itertools.product(args.seeds, args.rows, args.policies))
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
            for seed, rows, policy in cases:
                contract.validate_source(source, pin)
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
                    case["process"] = base.run_logged(
                        command, source, env, directory, "validation", stop_on_fallback=True
                    )
                    path = directory / "driver.json"
                    if path.exists():
                        case["driver_sha256"] = base.sha256(path)
                    if case["process"]["fallback_detected"]:
                        raise base.UnsupportedError("TensorOps compilation fallback forbidden")
                    report = records.load_json(path)
                    if report.get("status") == "incorrect":
                        raise base.NumericalError("QKV driver rejected its numerical contract")
                    validated = validate(report, rows, seed, policy, args.warmups, args.repeats)
                    case["summary"] = {"status": validated}
                    if validated == "completed":
                        for stage, paths in report["timings"].items():
                            medians = {
                                name: statistics.median(x["wall_seconds"] for x in samples)
                                for name, samples in paths.items()
                            }
                            case["summary"][stage] = {
                                "wall_median_seconds": medians,
                                "native_over_candidate": medians["native"] / medians["candidate"],
                            }
                    if report.get("status") == "unsupported":
                        if rows != 17 or case["process"]["returncode"] != 2:
                            raise base.UnsupportedError("unexpected unsupported QKV shape")
                        case["status"] = "unsupported"
                    else:
                        if case["process"]["returncode"]:
                            raise ValueError("native validation process failed")
                        case["status"] = "completed"
                except Exception as error:
                    case.update(status=classify(error), error=f"{type(error).__name__}: {error}")
                finally:
                    base.write_json(directory / "case.json", case)
                    record["cases"].append(case | dict(directory=directory.name))
                print(f"qkv {policy} m={rows} seed={seed}: {case['status']}", flush=True)
            contract.validate_source(source, pin)
            for name, digest in record["harness_files"].items():
                if base.sha256(Path(__file__).with_name(name)) != digest:
                    raise ValueError("harness changed during campaign: " + name)
            successful = all(
                c["status"] == "completed"
                or (c["rows"] == 17 and c["status"] == "unsupported" and "summary" in c)
                for c in record["cases"]
            )
            record["status"] = (
                "completed_with_explicit_unsupported"
                if successful and any(c["status"] == "unsupported" for c in record["cases"])
                else "completed"
                if successful
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
    parser.add_argument("--model-dir", type=Path, default=base.ROOT / ".cache/models/MiniMax-H3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[17, 129, 2049, 9914, 17455])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    parser.add_argument(
        "--policies", nargs="+", choices=("native", "h256"), default=["native", "h256"]
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    result = execute(parser.parse_args())
    print(result["status"])
    return 0 if result["status"].startswith("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

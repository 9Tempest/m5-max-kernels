"""Reproduce native FC2 synthetic correctness or frozen-activation calibration.

All GPU work is serialized with the shared repository lock. The output directory
is immutable, each attempted case has its own record, and a rejected precision
policy never causes a checkpoint write or deletion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

import fc2_analyze
import run as base
import run_fc2_experiment as experiment


DEFAULT_SYNTHETIC_CASES = ((17, 129), (29, 257))


def synthetic_case(value):
    try:
        seed, rows = map(int, value.split(":"))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("synthetic case must be SEED:ROWS") from None
    if not 0 <= seed < 2**64 or not 128 <= rows <= 16384:
        raise argparse.ArgumentTypeError("seed must be uint64 and rows must be 128..16384")
    return seed, rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=("synthetic", "calibration"), required=True)
    p.add_argument("--capture-run", type=Path)
    p.add_argument("--synthetic-case", type=synthetic_case, action="append")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=base.ROOT / ".cache/h3.c-hadamard")
    p.add_argument("--model-dir", type=Path, default=base.ROOT / ".cache/models/MiniMax-H3")
    p.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    return p


def validate_arguments(args):
    if args.scenario == "calibration":
        if args.capture_run is None or args.synthetic_case:
            raise ValueError("calibration requires --capture-run and forbids synthetic cases")
    elif args.capture_run is not None:
        raise ValueError("synthetic scenarios do not consume a capture run")
    cases = args.synthetic_case or list(DEFAULT_SYNTHETIC_CASES)
    if len(cases) != len(set(cases)):
        raise ValueError("duplicate synthetic cases would overwrite evidence")
    return cases


def synthetic_input(seed, rows):
    """BF16 normal input, one zero row and one sparse-outlier row; full last row."""
    # Validate this helper too: direct callers do not necessarily use argparse.
    if type(seed) is not int or type(rows) is not int:
        raise ValueError("seed and rows must be integers")
    synthetic_case(f"{seed}:{rows}")
    values = np.random.default_rng(seed).normal(size=(rows, 14336)).astype(np.float32)
    values[5] = 0
    values[9, ::257] *= 40
    bits = values.view(np.uint32)
    result = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
    if not np.any(result[-1]) or np.any(result[5]):
        raise ValueError("synthetic tail/zero-row construction failed")
    return result


def validate_capture(capture):
    source = experiment.load_json(capture / "run.json")
    if (
        source.get("status") != "completed"
        or source.get("mode") != "upstream-int8"
        or source.get("capture_fc2") is not True
        or source.get("patch", {}).get("source_sha256") != experiment.SOURCE_SHA256
        or source.get("model", {}).get("revision") != base.MODEL_REVISION
    ):
        raise ValueError("calibration requires a completed pinned native INT8 capture run")
    validated = experiment.validate_fc2_capture(capture / "fc2", "upstream-int8")
    recorded = source.get("fc2_validation")
    if recorded != validated:
        raise ValueError("capture contents differ from the hashes/metadata in its run record")
    return source, validated


def compile_driver(source, output, env, record):
    built = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
    record["build"] = built
    if built["returncode"]:
        raise base.UnsupportedError("pinned native library compilation failed")
    experiment.validate_source(source, base.git_provenance(source))
    executable = output / "fc2-validate"
    source_file = Path(__file__).with_name("fc2_validate.c")
    command = [
        "clang",
        "-O2",
        "-std=c11",
        "-D_DARWIN_C_SOURCE",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(source),
        str(source_file),
        str(source / "libh3.a"),
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-framework",
        "MetalPerformanceShaders",
        "-framework",
        "MetalPerformanceShadersGraph",
        "-framework",
        "Accelerate",
        "-licucore",
        "-lm",
        "-o",
        str(executable),
    ]
    result = base.run_logged(command, base.ROOT, env, output, "build-driver")
    record["driver_build"] = result
    if result["returncode"]:
        raise base.UnsupportedError("native numerical driver compilation failed")
    record["driver"] = {
        "source_sha256": base.sha256(source_file),
        "binary_sha256": base.sha256(executable),
        "library_sha256": base.sha256(source / "libh3.a"),
    }
    return executable


def run_case(item, output, source, model, driver, env):
    case = output / item["name"]
    case.mkdir()
    record = {"schema": 1, "status": "failed", "started_utc": base.utc_now(), **item}
    if "input" in record:
        record["input"] = base.path_label(record["input"])
    try:
        if item["scenario"] == "synthetic":
            infile = case / "synthetic-input.bf16"
            synthetic_input(item["seed"], item["rows"]).tofile(infile)
        else:
            infile = Path(item["input"])
        record["input_sha256"] = base.sha256(infile)
        readback = case / "readback"
        readback.mkdir()
        command = [
            str(driver),
            str(model / "FL2VA/transformer"),
            str(item["block"]),
            str(infile),
            str(readback),
            str(item["rows"]),
            str(source / "h3_shaders.metal"),
        ]
        process = base.run_logged(command, source, env, case, "validate", True)
        record["process"] = process
        if process["fallback_detected"]:
            raise base.UnsupportedError("TensorOps fallback rejected")
        if process["returncode"]:
            raise ValueError(f"native driver exited {process['returncode']}")
        analysis = fc2_analyze.analyze(readback)
        if item["scenario"] == "synthetic":
            zero_exact = all(
                np.all(
                    fc2_analyze.bf16(readback / f"{p}.output.bf16", (item["rows"], 5376))[5] == 0
                )
                for p in fc2_analyze.POLICIES
            )
            analysis["synthetic_zero_row_exact"] = bool(zero_exact)
        else:
            zero_exact = True
        base.write_json(case / "analysis.json", analysis)
        record.update(
            analysis_sha256=base.sha256(case / "analysis.json"),
            policy_status={p: r["status"] for p, r in analysis["policies"].items()},
            total_error={
                p: r["total_error_vs_original_bf16_operands"]
                for p, r in analysis["policies"].items()
            },
            hadamard_vs_native_group1024=analysis["hadamard_vs_native_group1024"],
        )
        if not zero_exact or any(r["status"] != "passed" for r in analysis["policies"].values()):
            raise base.NumericalError("native packing/output correctness gate failed")
        record["status"] = "completed"
    except Exception as error:
        record.update(
            status="incorrect"
            if isinstance(error, base.NumericalError)
            else "unsupported"
            if isinstance(error, base.UnsupportedError)
            else "failed",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        record["finished_utc"] = base.utc_now()
        base.write_json(case / "case.json", record)
    return record


def execute(args):
    output, source, model = (
        args.output_dir.resolve(),
        args.source_dir.resolve(),
        args.model_dir.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {
        "schema": 1,
        "status": "failed",
        "scenario": args.scenario,
        "started_utc": base.utc_now(),
        "harness_sha256": base.sha256(__file__),
        "analyzer_sha256": base.sha256(Path(__file__).with_name("fc2_analyze.py")),
        "cases": [],
        "checkpoint_replacement_eligible": False,
        "timing_scope": "correctness readback only; process seconds are not benchmark latency",
    }
    try:
        synthetic_cases = validate_arguments(args)
        env, record["runtime_overrides"] = base.runtime_environment()
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            record["environment"] = {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "numpy": np.__version__,
            }
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise base.UnsupportedError("Metal GPU validation unavailable on this host")
            record["environment"].update(
                chip=base.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                memory_bytes=int(base.read_command(["sysctl", "-n", "hw.memsize"])),
                compiler=base.read_command(["clang", "--version"]),
                xcode=base.read_command(["xcodebuild", "-version"]),
            )
            if "M5" not in record["environment"]["chip"]:
                raise base.UnsupportedError("native FC2 campaign requires Apple M5 TensorOps")
            record["harness_files"] = {
                name: base.sha256(Path(__file__).with_name(name))
                for name in ("run.py", "run_fc2_experiment.py", "fc2_analyze.py", "fc2_validate.c")
            }
            record["source"] = base.git_provenance(source)
            record["patch"] = experiment.validate_source(source, record["source"])
            record["model"] = base.model_inventory(
                model, Path(__file__).with_name("model-files.json")
            )
            if args.scenario == "calibration":
                capture = args.capture_run.resolve()
                source_run, record["capture_validation"] = validate_capture(capture)
                record.update(
                    source_run_sha256=base.sha256(capture / "run.json"),
                    source_seed=source_run["seed"],
                    scope="frozen real native INT8 FC1 post-SwiGLU BF16 rows, 32 sampled output columns, full K",
                )
                plan = [
                    {
                        "name": f"fc2-step{s:02d}-block{b:02d}",
                        "scenario": "calibration",
                        "step": s,
                        "block": b,
                        "rows": 129,
                        "input": str(capture / "fc2" / f"fc2-step{s:02d}-block{b:02d}.bf16"),
                    }
                    for s in (0, 10, 19)
                    for b in (0, 24, 49)
                ]
            else:
                record.update(
                    scope="synthetic BF16 inputs and actual block0 FC2 weights",
                    synthetic_distribution="NumPy PCG64 normal; row5 zero; row9 every257th column multiplied by40; nonzero final logical row",
                )
                plan = [
                    {
                        "name": f"seed{s}-m{m}",
                        "scenario": "synthetic",
                        "seed": s,
                        "rows": m,
                        "block": 0,
                    }
                    for s, m in synthetic_cases
                ]
            record["planned_cases"] = [
                item | ({"input": base.path_label(item["input"])} if "input" in item else {})
                for item in plan
            ]
            driver = compile_driver(source, output, env, record)
            for item in plan:
                # Preserve a failing case and continue independent cases, rather
                # than dropping the rest of a numerical campaign silently.
                record["cases"].append(run_case(item, output, source, model, driver, env))
            statuses = {case["status"] for case in record["cases"]}
            record["status"] = (
                "completed"
                if statuses == {"completed"}
                else "incorrect"
                if "incorrect" in statuses
                else "unsupported"
                if statuses == {"unsupported"}
                else "failed"
            )
            if args.scenario == "calibration" and record["status"] == "completed":
                historical_strict_gate = all(
                    case["hadamard_vs_native_group1024"]["single_sample_numerics_gate_pass"]
                    for case in record["cases"]
                )
                aggregate_better = all(
                    case["hadamard_vs_native_group1024"][
                        "aggregate_l2_rmse_maxabs_all_strictly_better"
                    ]
                    for case in record["cases"]
                )
                record["historical_strict_no_row_regression_gate_pass"] = historical_strict_gate
                record["current_decision_policy"] = (
                    "User accepts aggregate improvement with explicit per-row/tail risk review; "
                    "requires full-model latent/media evidence and independent seed; "
                    "a single row regression is not an automatic veto."
                )
                record["numerics_promotion_status"] = (
                    "aggregate sample improvement; full-model and tail-risk review required"
                    if aggregate_better
                    else "mixed aggregate evidence; further review required"
                )
            else:
                record["numerics_promotion_status"] = "not established by this scenario"
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupted by user")
    except Exception as error:
        record.update(
            status="unsupported" if isinstance(error, base.UnsupportedError) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        record.update(
            finished_utc=base.utc_now(), campaign_wall_seconds=time.perf_counter() - started
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
    print(json.dumps({k: record.get(k) for k in ("status", "numerics_promotion_status", "error")}))
    # A completed experiment may correctly reject the candidate policy.
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

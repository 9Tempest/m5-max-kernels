"""Isolated capture/resident experiments for frozen QKV and AdaLN/H256 kernels.

No historical runner or source pin is modified. Source approval is explicit;
pending pins cannot execute. GPU execution is serialized through the same lock.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace

import kernel_integration as contract
import run as base
import run_fc2_experiment as capture
import run_session as session


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=contract.POLICIES, required=True)
    p.add_argument("--kind", choices=("capture", "resident"), required=True)
    p.add_argument("--implementation", choices=("fused", "staged"), default="fused")
    p.add_argument("--seed", type=int, choices=(42, 43), required=True)
    p.add_argument("--source-dir", type=Path, default=base.ROOT / ".cache/h3.c-kernel-integration")
    p.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(__file__).with_name("kernel-integration-source.json"),
    )
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--model-dir", type=Path, default=base.ROOT / ".cache/models/MiniMax-H3")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    p.add_argument(
        "--capture-adaln",
        action="store_true",
        help="extra actual-row CPU diagnostics; never resident timing",
    )
    return p


def validate_arguments(args):
    contract.flags(args.policy, args.implementation)
    if args.seed not in (42, 43):
        raise ValueError("the reviewed campaign uses seeds 42 and 43")
    if args.kind not in ("capture", "resident"):
        raise ValueError("unknown execution kind")
    if args.kind == "resident" and args.implementation != "fused":
        raise ValueError("staged preparation is diagnostic-only and cannot enter resident timing")
    if args.capture_adaln and (
        args.kind != "capture" or args.policy not in ("adaln-h-only", "combined")
    ):
        raise ValueError("AdaLN capture requires a diagnostic Hadamard input policy")


def snapshot(output, manifest, pin, capture_adaln=False):
    directory = output / "harness-snapshot"
    directory.mkdir()
    sources = {
        name: Path(__file__).with_name(name)
        for name in (
            "run_kernel_integration.py",
            "kernel_integration.py",
            "kernel_session.c",
            "run.py",
            "run_fc2_experiment.py",
            "run_session.py",
            "model-files.json",
            "UPSTREAM-LICENSE",
        )
    }
    sources["locking.py"] = base.ROOT / "src/m5kernels/locking.py"
    if capture_adaln:
        sources["adaln_capture.py"] = Path(__file__).with_name("adaln_capture.py")
        sources["fc2_analyze.py"] = Path(__file__).with_name("fc2_analyze.py")
    entries = {}
    for name, source in sources.items():
        target = directory / name
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        entries[name] = dict(
            file="harness-snapshot/" + name,
            original=base.path_label(source),
            sha256=base.sha256(target),
        )
    base.write_json(directory / "manifest.json", entries)
    directory = output / "source-pin"
    directory.mkdir()
    for source, name in [
        (manifest, "manifest.json"),
        *[(manifest.parent / item["file"], item["file"]) for item in pin["patches"]],
    ]:
        with (directory / name).open("xb") as stream:
            stream.write(source.read_bytes())
    return entries


def validate_snapshot(entries, output):
    for item in entries.values():
        if not (
            base.sha256(base.ROOT / item["original"])
            == item["sha256"]
            == base.sha256(output / item["file"])
        ):
            raise ValueError("harness changed during the experiment")


def environment():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise base.UnsupportedError("Metal validation unavailable on this host")
    value = dict(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=capture.np.__version__,
        chip=base.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        memory_bytes=int(base.read_command(["sysctl", "-n", "hw.memsize"])),
        compiler=base.read_command(["clang", "--version"]),
        xcode=base.read_command(["xcodebuild", "-version"]),
    )
    if "M5" not in value["chip"]:
        raise base.UnsupportedError("native kernel integration requires an M5 device")
    for name in ("ffmpeg", "ffprobe"):
        binary = shutil.which(name)
        if not binary:
            raise ValueError(name + " is required")
        value[name] = base.read_command([binary, "-version"]).splitlines()[0]
    return value


def build(source, output, env, record):
    result = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
    record["measurements"]["build"] = result
    if result["returncode"] or result["fallback_detected"]:
        raise base.UnsupportedError("combined source build failed")
    if record["kind"] == "capture":
        driver = output / "native-cli"
        shutil.copy2(source / "h3", driver)
    else:
        driver = output / "kernel-session"
        command = [
            "clang",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-D_DARWIN_C_SOURCE",
            "-DH3_SESSION_EXPERIMENT=1",
            "-I",
            str(source),
            str(output / "harness-snapshot/kernel_session.c"),
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
            str(driver),
        ]
        result = base.run_logged(command, source, env, output, "driver-build")
        record["measurements"]["driver_build"] = result
        if result["returncode"] or result["fallback_detected"]:
            raise base.UnsupportedError("kernel session driver build failed")
    record["driver"] = dict(
        file=driver.name,
        binary_sha256=base.sha256(driver),
        library_sha256=base.sha256(source / "libh3.a"),
    )
    return driver


def run_capture(args, source, model, output, driver, env, record):
    (output / "latents").mkdir()
    env.update(
        record["requested_runtime_flags"]
        | dict(
            H3_PROFILE="1",
            H3_LATENT_CAPTURE_DIR=str(output / "latents"),
            H3_CAPTURE_SEED=str(args.seed),
            H3_GPU_SAMPLER="1",
            H3_GPU_SAMPLER_WINDOW="1",
        )
    )
    if args.capture_adaln:
        (output / "adaln").mkdir()
        env["H3_ADALN_CAPTURE_DIR"] = str(output / "adaln")
    recipe = SimpleNamespace(
        mode="upstream-int8", width=512, height=512, seconds=5, steps=20, seed=args.seed
    )
    command = base.build_command(recipe, source, model, output / "original.mp4")
    command[0] = str(driver)
    record["runtime_overrides"] = {
        key: base.path_label(value) if key.endswith("_DIR") else value
        for key, value in env.items()
        if key.startswith("H3_")
    }
    base.write_json(output / "request.json", record | {"command": command})
    result = base.run_logged(command, source, env, output, "generation", True)
    record["measurements"]["generation"] = result
    if result["returncode"] or result["fallback_detected"]:
        raise base.UnsupportedError("native diagnostic generation failed or fell back")
    log = (output / "generation.log").read_text()
    record["policy_evidence"] = contract.validate_log(log, args.policy, args.implementation)
    record["profile"] = base.parse_profile(log)
    record["latent_validation"] = contract.validate_latents(
        output / "latents", args.policy, args.implementation, args.seed
    )
    if args.capture_adaln:
        import adaln_capture

        record["adaln_validation"] = adaln_capture.validate_capture(
            output / "adaln",
            expected_flags=contract.metadata_flags(args.policy, args.implementation)
            | {"hadamard_fc2": 1},
            expected_staged=int(args.implementation == "staged"),
            packed_rows=9914,
            expected_steps=(0, 10, 19),
            expected_blocks=(0, 24, 49),
        )
    capture.export_video(
        output, source, env, shutil.which("ffmpeg"), shutil.which("ffprobe"), record
    )


def run_resident(args, source, model, output, driver, record):
    command = [
        str(driver),
        "hadamard-fc2",
        str(model),
        str(output / "session"),
        "512",
        "512",
        "20",
        str(args.seed),
        args.policy,
    ]
    # The C driver rejects inherited H3_* variables and applies its exact flags.
    env = os.environ.copy()
    record["runtime_overrides"] = record["requested_runtime_flags"] | dict(
        H3_NAX="qkv-attn", H3_NAX_DIAGNOSTIC="1", H3_PROFILE="1"
    )
    base.write_json(output / "request.json", record | {"command": command})
    result = base.run_logged(command, source, env, output, "session-process", True)
    record["measurements"]["session_process"] = result
    if result["returncode"] or result["fallback_detected"]:
        raise base.UnsupportedError("native resident generation failed or fell back")
    path = output / "session/session.jsonl"
    events = [
        json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        for line in path.read_text().splitlines()
    ]
    generations = contract.validate_events(events, args.policy, args.seed)
    record.update(
        session_events=events,
        session_events_sha256=base.sha256(path),
        observed_runtime_flags={k: events[1][k] for k in contract.flags(args.policy)},
    )
    record["shape"] = session.shape_from_cache(
        base.temporal_shape(5, 512, 512), generations, "upstream-int8"
    )
    record["videos"], record["profiles"], record["policy_evidence"] = [], [], []
    for event in generations:
        video = output / "session" / event["output"]
        log_path = output / "session" / event["profile_log"]
        log = log_path.read_text()
        record["policy_evidence"].append(
            contract.validate_log(log, args.policy, "fused", event["index"])
        )
        result = base.run_logged(
            base.ffprobe_command(shutil.which("ffprobe"), str(video)),
            source,
            env,
            output,
            video.stem + "-probe",
        )
        if result["returncode"]:
            raise ValueError("resident output probe failed")
        value = base.validate_video(capture.load_json(output / result["log"]), 512, 512, 124)
        record["videos"].append(
            value | dict(file="session/" + video.name, sha256=base.sha256(video))
        )
        record["profiles"].append(
            dict(
                index=event["index"],
                phases=base.parse_profile(log),
                log_sha256=base.sha256(log_path),
            )
        )
    record["warmup_seconds"] = generations[0]["wall_seconds"]
    record["cached_generation"] = session.summarize([e["wall_seconds"] for e in generations[1:]])
    record["cached_denoise_callback"] = session.summarize(
        [e["denoise_callback_seconds"] for e in generations[1:]]
    )
    record["all_repetition_videos_byte_identical"] = (
        len({v["sha256"] for v in record["videos"]}) == 1
    )
    if not record["all_repetition_videos_byte_identical"]:
        raise base.NumericalError("same-policy same-seed resident outputs differ across requests")


def execute(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(
        schema=contract.SCHEMA,
        status="failed",
        kind=args.kind,
        policy=args.policy,
        implementation=args.implementation,
        seed=args.seed,
        started_utc=base.utc_now(),
        **contract.RECIPE,
        measurements={},
        capture_latents=args.kind == "capture",
        capture_adaln=args.capture_adaln,
        diagnostic=args.kind == "capture",
        prepared_checkpoint=False,
        quality_decision="not_evaluated",
        timing_contract=dict(
            capture="diagnostic extra waits/readbacks; not a performance sample",
            resident="one full warmup then three same-seed complete requests in one process",
            denoise="fresh synchronized native callbacks including online preparation and final copy/unpack; not stale cached profile wall",
            generation="includes decode and original 124-frame AV export; excludes preflight/build and exact-duration trim",
            compilation_free_e2e="not claimed: audio VAE setup repeats",
            weight_preparation="outside timed resident requests, included in warmup",
        ),
    )
    try:
        validate_arguments(args)
        pin = contract.load_pin(args.source_manifest, args.source_manifest_sha256)
        record["source_pin"] = dict(
            file="source-pin/manifest.json", sha256=args.source_manifest_sha256, manifest=pin
        )
        record["requested_runtime_flags"] = contract.flags(args.policy, args.implementation)
        env, _ = base.runtime_environment()
        record["harness_snapshot"] = snapshot(output, args.source_manifest, pin, args.capture_adaln)
        source, model = args.source_dir.resolve(), args.model_dir.resolve()
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            preflight = time.perf_counter()
            record["harness"] = base.git_provenance(base.ROOT)
            record["upstream"] = contract.validate_source(source, pin)
            record["environment"] = environment()
            record["model"] = base.model_inventory(
                model, Path(__file__).with_name("model-files.json")
            )
            driver = build(source, output, env, record)
            validate_snapshot(record["harness_snapshot"], output)
            contract.validate_source(source, pin)
            record["preflight_seconds"] = time.perf_counter() - preflight
            if args.kind == "capture":
                run_capture(args, source, model, output, driver, env, record)
            else:
                record.update(warmups=1, repetitions=3)
                run_resident(args, source, model, output, driver, record)
            contract.validate_source(source, pin)
            contract.load_pin(output / "source-pin/manifest.json", args.source_manifest_sha256)
            validate_snapshot(record["harness_snapshot"], output)
            record["status"] = "completed"
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupted by user")
    except Exception as error:
        record.update(
            status="unsupported" if isinstance(error, base.UnsupportedError) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        record.update(
            finished_utc=base.utc_now(), harness_wall_seconds=time.perf_counter() - started
        )
        base.write_json(output / "run.json", record)
    return record


if __name__ == "__main__":
    result = execute(parser().parse_args())
    print(json.dumps({key: result.get(key) for key in ("status", "error")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)

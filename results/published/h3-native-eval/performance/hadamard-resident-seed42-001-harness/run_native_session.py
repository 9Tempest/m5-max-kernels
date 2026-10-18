"""Warm resident H3 precision/fusion experiments with a pinned native runtime.

One complete warmup precedes three same-seed complete generations. The wrapper
owns the shared GPU lock, including source/model preflight and compilation.
Denoise callbacks bracket synchronized model execution. Full request latency
still includes per-request audio VAE setup; it is not kernel-only latency.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace

import run as base
import run_fc2_experiment as experiment
import run_session as session


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("bf16", "upstream-int8", "hadamard-fc2"), required=True)
    p.add_argument("--fusion", choices=("none", "fc2", "attention", "both"), default="none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-dir", type=Path, default=base.ROOT / ".cache/h3.c-hadamard")
    p.add_argument("--model-dir", type=Path, default=base.ROOT / ".cache/models/MiniMax-H3")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    return p


def validate_arguments(args):
    if not 0 <= args.seed < 2**63:
        raise ValueError("seed must be a nonnegative signed 64-bit integer")
    if args.fusion != "none" and args.mode != "hadamard-fc2":
        raise ValueError("fusion comparisons hold the Hadamard weight/activation policy fixed")


def validate_source(source, fusion):
    provenance = base.git_provenance(source)
    if fusion != "none":
        raise base.UnsupportedError("fusion source has not yet been frozen and validated")
    return provenance, experiment.validate_source(source, provenance)


def compile_driver(source, output, env, record):
    result = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
    record["measurements"]["build"] = result
    if result["returncode"]:
        raise base.UnsupportedError("native runtime compilation failed")
    source_file = Path(__file__).with_name("session.c")
    driver = output / "native-session"
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
        str(driver),
    ]
    result = base.run_logged(command, source, env, output, "driver-build")
    record["measurements"]["driver_build"] = result
    if result["returncode"]:
        raise base.UnsupportedError("resident experiment driver compilation failed")
    record["driver"] = {
        "source_sha256": base.sha256(source_file),
        "binary_sha256": base.sha256(driver),
        "library_sha256": base.sha256(source / "libh3.a"),
        "wrapper_sha256": base.sha256(Path(__file__)),
    }
    return driver


def validate_events(events, args):
    recipe = SimpleNamespace(mode=args.mode, width=512, height=512, steps=20, seed=args.seed)
    session.validate_session_recipe(events, recipe)
    recipes = [event for event in events if event.get("event") == "recipe"]
    if recipes[0].get("fusion") != args.fusion:
        raise ValueError("driver fusion policy does not match the requested experiment")
    generations = [event for event in events if event.get("event") == "generation"]
    if len(generations) != 4 or [event["index"] for event in generations] != [0, 1, 2, 3]:
        raise ValueError("expected one warmup and three measured generations")
    for event in generations:
        if event.get("status") != "completed" or event.get("warmup") is not (event["index"] == 0):
            raise ValueError("incomplete generation or mismatched warmup designation")
        if event["output"] != f"{'timed' if event['index'] else 'warmup'}-{event['index']}.mp4":
            raise ValueError("unexpected generation output path")
        if event["profile_log"] != Path(event["output"]).with_suffix(".log").name:
            raise ValueError("unexpected generation profile path")
        session.summarize([event["wall_seconds"], event["denoise_callback_seconds"]])
        if not 0 < event["denoise_callback_seconds"] <= event["wall_seconds"]:
            raise ValueError("denoise callback must be positive and contained in request wall")
        for key in ("prepared_dit", "video_decoder", "embedding_entries"):
            if not event["cache_after"].get(key) or (
                event["index"] and not event["cache_before"].get(key)
            ):
                raise ValueError("resident cache was not retained between measured requests")
    return generations


def execute(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, model = args.source_dir.resolve(), args.model_dir.resolve()
    started = time.perf_counter()
    record = {
        "schema_version": 1,
        "status": "failed",
        "started_utc": base.utc_now(),
        "benchmark": "fox-winter-t2va-cached-v1",
        "prompt": base.PROMPT,
        "mode": args.mode,
        "fusion": args.fusion,
        "seed": args.seed,
        "width": 512,
        "height": 512,
        "steps": 20,
        "source_frames": 124,
        "warmups": 1,
        "repetitions": 3,
        "active_dit_blocks": 50,
        "reuse": 1,
        "core_reuse": 1,
        "use_reference_rope": True,
        "measurements": {},
        "timing_contract": {
            "cached_generation": "h3_generate synchronized host wall including decode and 124-frame AV export; excludes one complete warmup",
            "cached_denoise_callback": "after packing/copy through synchronized GPU completion and output latent copy/unpack; warmed DiT pipelines and weights; online activation quantization/padding included",
            "compile_free_e2e": "not claimed; audio VAE context setup/compilation API and loading repeat per request",
            "offline_weight_preparation": "outside measured requests, included in warmup",
            "kernel_only": "not measured",
            "ordering": "one mode per process, three same-seed repetitions; no interleaved cross-mode stability claim",
        },
    }
    try:
        validate_arguments(args)
        _, overrides = base.runtime_environment()  # rejects all inherited H3_* controls
        env = os.environ.copy()  # driver applies its explicit recipe itself
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            preflight = time.perf_counter()
            record["harness"] = base.git_provenance(base.ROOT)
            record["harness_files"] = {
                name: base.sha256(Path(__file__).with_name(name))
                for name in (
                    "run.py",
                    "run_session.py",
                    "session.c",
                    "run_native_session.py",
                    "run_fc2_experiment.py",
                )
            }
            record["upstream"], record["patch"] = validate_source(source, args.fusion)
            record["runtime_overrides"] = overrides | {
                "H3_PROFILE": "1",
                "H3_HADAMARD_FC2": str(int(args.mode == "hadamard-fc2")),
            }
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise base.UnsupportedError("Metal validation unavailable on this host")
            record["environment"] = {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "chip": base.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                "memory_bytes": int(base.read_command(["sysctl", "-n", "hw.memsize"])),
                "compiler": base.read_command(["clang", "--version"]),
                "xcode": base.read_command(["xcodebuild", "-version"]),
            }
            if "M5" not in record["environment"]["chip"]:
                raise base.UnsupportedError("resident native experiment requires Apple M5")
            ffprobe = shutil.which("ffprobe")
            if not ffprobe or not shutil.which("ffmpeg"):
                raise ValueError("ffmpeg and ffprobe are required")
            record["environment"]["ffprobe"] = base.read_command(
                [ffprobe, "-version"]
            ).splitlines()[0]
            record["model"] = base.model_inventory(
                model, Path(__file__).with_name("model-files.json")
            )
            driver = compile_driver(source, output, env, record)
            validate_source(source, args.fusion)
            record["preflight_seconds"] = time.perf_counter() - preflight
            command = [
                str(driver),
                args.mode,
                str(model),
                str(output / "session"),
                "512",
                "512",
                "20",
                str(args.seed),
                args.fusion,
            ]
            base.write_json(output / "request.json", record | {"command": command})
            result = base.run_logged(command, source, env, output, "session-process", True)
            record["measurements"]["session_process"] = result
            if result["returncode"] or result["fallback_detected"]:
                raise base.UnsupportedError(
                    "native session failed; inspect retained per-generation logs"
                )
            events = [
                json.loads(line)
                for line in (output / "session/session.jsonl").read_text().splitlines()
            ]
            record["session_events"] = events
            generations = validate_events(events, args)
            record["shape"] = session.shape_from_cache(
                base.temporal_shape(5, 512, 512),
                generations,
                "bf16" if args.mode == "bf16" else "upstream-int8",
            )
            record["videos"], record["profiles"] = [], []
            for event in generations:
                video = output / "session" / event["output"]
                log = (output / "session" / event["profile_log"]).read_text(errors="replace")
                if "TensorOps compile failed:" in log:
                    raise base.UnsupportedError("TensorOps fallback in generation log")
                if (
                    event["index"] == 0
                    and args.mode == "hadamard-fc2"
                    and "h3 precision: FC2 plain normalized H256" not in log
                ):
                    raise base.UnsupportedError("Hadamard runtime policy marker missing")
                probe_result = base.run_logged(
                    base.ffprobe_command(ffprobe, str(video)),
                    source,
                    env,
                    output,
                    video.stem + "-probe",
                )
                if probe_result["returncode"]:
                    raise ValueError("output ffprobe failed")
                probe = json.loads((output / probe_result["log"]).read_text())
                record["videos"].append(
                    base.validate_video(probe, 512, 512, 124)
                    | {"file": "session/" + video.name, "sha256": base.sha256(video)}
                )
                record["profiles"].append(
                    {
                        "index": event["index"],
                        "phases": base.parse_profile(log),
                        "log_sha256": base.sha256(output / "session" / event["profile_log"]),
                    }
                )
            record["warmup_seconds"] = generations[0]["wall_seconds"]
            record["cached_generation"] = session.summarize(
                [event["wall_seconds"] for event in generations[1:]]
            )
            record["cached_denoise_callback"] = session.summarize(
                [event["denoise_callback_seconds"] for event in generations[1:]]
            )
            record["all_repetition_videos_byte_identical"] = (
                len({video["sha256"] for video in record["videos"]}) == 1
            )
            validate_source(source, args.fusion)
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
                "error": record.get("error"),
                "record": base.path_label(args.output_dir / "run.json"),
            }
        )
    )
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

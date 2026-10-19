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
import run_fused_campaign as fused
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
    if provenance["source_sha256"] == fused.SOURCE_SHA256:
        return provenance, fused.validate_source(source, provenance)
    if fusion != "none":
        raise base.UnsupportedError("enabled fusion requires the exact frozen fused runtime")
    return provenance, experiment.validate_source(source, provenance)


def snapshot_harness(output):
    directory = output / "harness-snapshot"
    directory.mkdir()
    names = (
        "run.py",
        "run_session.py",
        "session.c",
        "run_native_session.py",
        "run_fc2_experiment.py",
        "run_fused_campaign.py",
        "fused.patch",
        "hadamard.patch",
        "fused_validate.c",
        "fused_reference.h",
        "model-files.json",
        "UPSTREAM-LICENSE",
    )
    sources = {name: Path(__file__).with_name(name) for name in names}
    sources["locking.py"] = base.ROOT / "src/m5kernels/locking.py"
    entries = {}
    for name, source in sources.items():
        with (directory / name).open("xb") as destination:
            destination.write(source.read_bytes())
        entries[name] = {
            "file": "harness-snapshot/" + name,
            "original": base.path_label(source),
            "sha256": base.sha256(directory / name),
        }
    base.write_json(directory / "manifest.json", entries)
    return entries


def validate_snapshot(entries):
    for item in entries.values():
        if base.sha256(base.ROOT / item["original"]) != item["sha256"]:
            raise ValueError(
                "harness source changed after its immutable snapshot: " + item["original"]
            )


def compile_driver(source, output, env, record):
    result = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
    record["measurements"]["build"] = result
    if result["returncode"]:
        raise base.UnsupportedError("native runtime compilation failed")
    source_file = output / "harness-snapshot/session.c"
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
    if [event.get("event") for event in events] != [
        "recipe",
        "runtime_flags",
        "startup",
        "device",
        *["generation"] * 4,
        "finished",
    ]:
        raise ValueError("unexpected/missing runtime, startup, device or generation events")
    expected_flags = runtime_flags(args)
    if {key: events[1].get(key) for key in expected_flags} != expected_flags:
        raise ValueError("actual process fusion environment differs from the requested policy")
    if events[2].get("status") != "completed":
        raise ValueError("session model startup did not complete")
    fused.number(events[2].get("wall_seconds"), "startup wall", positive=True)
    if not isinstance(events[3].get("name"), str) or "M5" not in events[3]["name"]:
        raise base.UnsupportedError("resident driver did not report an M5 device")
    generations = [event for event in events if event.get("event") == "generation"]
    if len(generations) != 4 or [event["index"] for event in generations] != [0, 1, 2, 3]:
        raise ValueError("expected one warmup and three measured generations")
    for event in generations:
        if type(event.get("index")) is not int:
            raise ValueError("generation index must be an integer")
        if event.get("status") != "completed" or event.get("warmup") is not (event["index"] == 0):
            raise ValueError("incomplete generation or mismatched warmup designation")
        if event["output"] != f"{'timed' if event['index'] else 'warmup'}-{event['index']}.mp4":
            raise ValueError("unexpected generation output path")
        if event["profile_log"] != Path(event["output"]).with_suffix(".log").name:
            raise ValueError("unexpected generation profile path")
        for key in (
            "wall_seconds",
            "denoise_callback_seconds",
            "denoise_start_offset_seconds",
            "denoise_end_offset_seconds",
        ):
            fused.number(
                event.get(key), key, positive=key in ("wall_seconds", "denoise_callback_seconds")
            )
        if not 0 < event["denoise_callback_seconds"] <= event["wall_seconds"]:
            raise ValueError("denoise callback must be positive and contained in request wall")
        first, last = event["denoise_start_offset_seconds"], event["denoise_end_offset_seconds"]
        if (
            not 0 <= first < last <= event["wall_seconds"]
            or abs(last - first - event["denoise_callback_seconds"]) > 3e-8
        ):
            raise ValueError(
                "denoise callback offsets do not form the reported synchronized interval"
            )
        expected = dict(
            requested_seed=args.seed,
            result_seed=args.seed,
            denoise_start_phase="denoise enqueue",
            denoise_end_phase="denoise",
            denoise_total=20,
            denoise_completed=20,
            denoise_completion_events=1,
        )
        if any(
            type(event.get(key)) is not type(value) or event[key] != value
            for key, value in expected.items()
        ):
            raise ValueError(
                "generation seed or complete denoise callback boundaries differ from recipe"
            )
        for key in ("prepared_dit", "video_decoder", "embedding_entries"):
            if not event["cache_after"].get(key) or (
                event["index"] and not event["cache_before"].get(key)
            ):
                raise ValueError("resident cache was not retained between measured requests")
    return generations


def runtime_flags(args):
    return {
        "H3_HADAMARD_FC2": str(int(args.mode == "hadamard-fc2")),
        "H3_FUSE_FC2_GATE": "unset"
        if args.fusion == "none"
        else str(int(args.fusion in ("fc2", "both"))),
        "H3_FUSE_ATTN_OUT_GATE": "unset"
        if args.fusion == "none"
        else str(int(args.fusion in ("attention", "both"))),
    }


def validate_policy_log(log, index, args):
    markers = {
        "hadamard_fc2": "h3 precision: FC2 plain normalized H256",
        "fc2_gate": "h3 fusion: FC2 GEMM gate/residual epilogue",
        "attention_output_gate": "h3 fusion: attention output GEMM gate/residual epilogue",
    }
    enabled = dict(
        hadamard_fc2=args.mode == "hadamard-fc2",
        fc2_gate=args.fusion in ("fc2", "both"),
        attention_output_gate=args.fusion in ("attention", "both"),
    )
    counts = {key: log.count(marker) for key, marker in markers.items()}
    if any(counts[key] != int(index == 0 and enabled[key]) for key in markers):
        raise base.UnsupportedError(
            "runtime precision/fusion markers differ from the fixed resident recipe"
        )
    cache_hits = {
        name: log.count(marker)
        for name, marker in {
            "prepared_dit": "h3: prepared DiT cache hit",
            "video_decoder": "h3: video VAE cache hit",
            "conditioning": "h3: conditioning cache hit",
        }.items()
    }
    if any(count != int(index > 0) for count in cache_hits.values()):
        raise ValueError("generation log does not confirm exactly one hit for every resident cache")
    return {
        "index": index,
        "enabled": enabled,
        "initialization_marker_counts": counts,
        "cache_hit_counts": cache_hits,
        "evidence": "initialization markers on warmup; fixed process environment and cache hits thereafter",
    }


def execute(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, model = args.source_dir.resolve(), args.model_dir.resolve()
    started = time.perf_counter()
    record = {
        "schema_version": 2,
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
            "denoise_measurement_overhead": "includes progress callback logging/read checks; callback offsets are native monotonic timestamps, not pipe receipt times",
            "native_profile_wall_caveat": "cached DiT profile wall marks span the preceding request's decode/export; never use them as per-request denoise latency",
            "compile_free_e2e": "not claimed; audio VAE context setup/compilation API and loading repeat per request",
            "offline_weight_preparation": "outside measured requests, included in warmup",
            "kernel_only": "not measured",
            "ordering": "one mode per process, three same-seed repetitions; no interleaved cross-mode stability claim",
            "cache_policy": "precision/fusion environment is fixed for the process; upstream prepared cache keys omit these experimental environment controls",
        },
    }
    try:
        validate_arguments(args)
        record["harness_snapshot"] = snapshot_harness(output)
        _, overrides = base.runtime_environment()  # rejects all inherited H3_* controls
        env = os.environ.copy()  # driver applies its explicit recipe itself
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            preflight = time.perf_counter()
            record["harness"] = base.git_provenance(base.ROOT)
            validate_snapshot(record["harness_snapshot"])
            record["harness_files"] = {
                name: item["sha256"] for name, item in record["harness_snapshot"].items()
            }
            record["upstream"], record["patch"] = validate_source(source, args.fusion)
            record["runtime_overrides"] = (
                overrides
                | {
                    "H3_PROFILE": "1",
                }
                | {key: value for key, value in runtime_flags(args).items() if value != "unset"}
            )
            record["requested_runtime_flags"] = runtime_flags(args)
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
            validate_snapshot(record["harness_snapshot"])
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
                json.loads(
                    line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
                )
                for line in (output / "session/session.jsonl").read_text().splitlines()
            ]
            record["session_events"] = events
            generations = validate_events(events, args)
            record["observed_runtime_flags"] = {key: events[1][key] for key in runtime_flags(args)}
            record["session_events_sha256"] = base.sha256(output / "session/session.jsonl")
            record["shape"] = session.shape_from_cache(
                base.temporal_shape(5, 512, 512),
                generations,
                "bf16" if args.mode == "bf16" else "upstream-int8",
            )
            record["videos"], record["profiles"], record["policy_evidence"] = [], [], []
            for event in generations:
                video = output / "session" / event["output"]
                log = (output / "session" / event["profile_log"]).read_text(errors="replace")
                if "TensorOps compile failed:" in log:
                    raise base.UnsupportedError("TensorOps fallback in generation log")
                record["policy_evidence"].append(validate_policy_log(log, event["index"], args))
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
            validate_snapshot(record["harness_snapshot"])
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

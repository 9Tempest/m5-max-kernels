"""Record one warmup and three cached end-to-end generations with session.c.

The C driver holds the same GPU experiment flock as m5kernels.locking.gpu_lock.
Do not acquire another GPU lock around it. Audio VAE compilation API and weight
loading repeat, so these measurements must not be called compilation-free.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time

import run as h3


def summarize(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timing samples must be finite nonnegative values")
    ordered = sorted(values)

    def percentile(q):
        index = (len(ordered) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return dict(
        raw_seconds=values,
        count=len(values),
        median_seconds=statistics.median(values),
        p10_seconds=percentile(0.1),
        p90_seconds=percentile(0.9),
        percentile_method="linear interpolation; three samples are descriptive, not stability proof",
    )


def shape_from_cache(shape, generations, mode):
    # Pinned h3_cache_get_info reports tokens * (5120 BF16 values + one uint8 tag).
    # Our fixed text-only recipe has no additional video/audio conditioning storage.
    sizes = {event["cache_after"]["embedding_bytes"] for event in generations}
    if len(sizes) != 1:
        raise ValueError("same-prompt session cache shape changed between generations")
    size = sizes.pop()
    bytes_per_token = 5120 * 2 + 1
    if size <= 0 or size % bytes_per_token:
        raise ValueError("cache byte count is inconsistent with the pinned text-only layout")
    text_tokens = size // bytes_per_token
    packed = shape["video_tokens"] + shape["stereo_audio_tokens"] + text_tokens
    return shape | dict(
        text_tokens=text_tokens,
        packed_sequence_tokens=packed,
        int8_activation_arena_rows=((packed + 127) // 128) * 128
        if mode == "upstream-int8"
        else None,
        shape_evidence="text tokens derived from observed public cache bytes and pinned "
        "h3_cache_get_info formula; video/audio/layout from h3_host.c; "
        "INT8 arena allocation rounding from h3_dit.c; not a per-op tensor trace",
    )


def validate_session_recipe(events, args):
    recipes = [event for event in events if event.get("event") == "recipe"]
    if len(recipes) != 1:
        raise ValueError("session must contain exactly one driver recipe event")
    expected = dict(
        mode=args.mode,
        width=args.width,
        height=args.height,
        frames=124,
        fps=24,
        steps=args.steps,
        seed=args.seed,
        prompt=h3.PROMPT,
        layers=50,
        reuse=1,
        core_reuse=1,
        use_reference_rope=True,
        warmups=1,
        repetitions=3,
        upstream_revision=h3.UPSTREAM_REVISION,
        model_revision=h3.MODEL_REVISION,
    )
    mismatch = [
        key
        for key, value in expected.items()
        if type(recipes[0].get(key)) is not type(value) or recipes[0].get(key) != value
    ]
    if mismatch:
        raise ValueError("session driver recipe differs from request: " + ", ".join(mismatch))
    finished = [event for event in events if event.get("event") == "finished"]
    if len(finished) != 1 or finished[0] is not events[-1]:
        raise ValueError("session must end with exactly one finished event")
    if type(finished[0].get("exit_code")) is not int or finished[0]["exit_code"] != 0:
        raise ValueError("session finished event must report exit_code 0")


def execute(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, model, driver = (
        args.source_dir.resolve(),
        args.model_dir.resolve(),
        args.driver.resolve(),
    )
    started = time.perf_counter()
    record = dict(
        schema_version=1,
        benchmark="fox-winter-t2va-cached-v1",
        status="failed",
        started_utc=h3.utc_now(),
        prompt=h3.PROMPT,
        mode=args.mode,
        seed=args.seed,
        width=args.width,
        height=args.height,
        steps=args.steps,
        warmups=1,
        repetitions=3,
        active_dit_blocks=50,
        reuse=1,
        core_reuse=1,
        use_reference_rope=True,
        measurements={},
        timing_contract={
            "generation": "cached h3_generate wall time including audio/video decode and AV export",
            "compile_free_steady_state": "not measured: audio VAE calls h3_gpu_create each request",
            "profile": "cached DiT wall marks span intervening decode/export; do not interpret "
            "as per-generation denoise time",
            "denoise_callback": "after input packing/copy through synchronized GPU completion "
            "and latent output copy/unpack",
            "kernel_only": "not measured",
            "quality": "ffprobe verifies structure only; same-seed numerical/visual parity untested",
        },
    )
    try:
        if args.diagnostics:
            raise ValueError("diagnostics are only supported by the one-shot run.py harness")
        if args.gpu_wait_seconds != 0:
            raise ValueError(
                "GPU waiting is only supported by run.py; the session driver owns its lock"
            )
        if args.seconds != 5:
            raise ValueError("the session driver has a fixed 5-second request (124 source frames)")
        if args.width < 256 or args.height < 256 or args.width % 32 or args.height % 32:
            raise ValueError("width and height must be multiples of 32 and at least 256")
        if not 2 <= args.steps <= 1000 or not 0 <= args.seed < 2**64:
            raise ValueError("steps must be in [2,1000] and seed an unsigned 64-bit integer")
        record["shape"] = h3.temporal_shape(args.seconds, args.width, args.height)
        record["harness"] = h3.git_provenance(h3.ROOT)
        record["upstream"] = h3.git_provenance(source)
        if record["upstream"]["revision"] != h3.UPSTREAM_REVISION or record["upstream"]["dirty"]:
            raise ValueError("upstream must be clean at the pinned revision")
        record["driver"] = dict(
            source_sha256=h3.sha256(Path(__file__).with_name("session.c")),
            binary_sha256=h3.sha256(driver),
            library_sha256=h3.sha256(source / "libh3.a"),
            wrapper_sha256=h3.sha256(Path(__file__)),
        )
        _, overrides = h3.runtime_environment()
        record["runtime_overrides"] = overrides | {"H3_PROFILE": "1"}
        # The C driver rejects inherited H3_* controls, then applies this recipe itself.
        env = os.environ.copy()
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise h3.UnsupportedError("Metal GPU validation unavailable on this host")
        record["environment"] = dict(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            python=platform.python_version(),
            chip=h3.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            memory_bytes=int(h3.read_command(["sysctl", "-n", "hw.memsize"])),
            compiler=h3.read_command(["clang", "--version"]),
            xcode=h3.read_command(["xcodebuild", "-version"]),
        )
        if "M5" not in record["environment"]["chip"]:
            raise h3.UnsupportedError("the pinned session recipe requires an Apple M5 GPU")
        ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
        if not ffprobe or not ffmpeg:
            raise ValueError("ffmpeg and ffprobe are required")
        for name, binary in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
            record["environment"][name] = h3.read_command([binary, "-version"]).splitlines()[0]
        record["model"] = h3.model_inventory(model, args.model_manifest)
        record["preflight_seconds"] = time.perf_counter() - started
        session = output / "session"
        command = [
            os.path.relpath(driver, source),
            args.mode,
            os.path.relpath(model, source),
            os.path.relpath(session, source),
            str(args.width),
            str(args.height),
            str(args.steps),
            str(args.seed),
        ]
        h3.write_json(output / "request.json", record | {"command": command})
        result = h3.run_logged(command, source, env, output, "session-process")
        record["measurements"]["session_process"] = result
        events = [json.loads(line) for line in (session / "session.jsonl").read_text().splitlines()]
        record["session_events"] = events
        if any(
            event.get("status") == "unsupported" or event.get("event") == "unsupported"
            for event in events
        ):
            raise h3.UnsupportedError(
                "session rejected an unsupported device or TensorOps fallback"
            )
        validate_session_recipe(events, args)
        generations = [event for event in events if event.get("event") == "generation"]
        if result["returncode"] or len(generations) != 4:
            raise ValueError("session did not complete one warmup plus three timed generations")
        if [event["index"] for event in generations] != [0, 1, 2, 3] or any(
            event["status"] != "completed" or event["warmup"] != (event["index"] == 0)
            for event in generations
        ):
            raise ValueError("session generation record does not match the repetition contract")
        record["shape"] = shape_from_cache(record["shape"], generations, args.mode)
        record["videos"] = []
        record["profiles"] = []
        for event in generations:
            video = session / event["output"]
            result = h3.run_logged(
                h3.ffprobe_command(ffprobe, os.path.relpath(video, source)),
                source,
                env,
                output,
                video.stem + "-probe",
            )
            record["measurements"][video.stem + "_probe"] = result
            if result["returncode"]:
                raise ValueError("session output ffprobe failed")
            probe = json.loads((output / result["log"]).read_text())
            structure = h3.validate_video(probe, args.width, args.height, 124)
            record["videos"].append(
                structure | dict(file="session/" + video.name, sha256=h3.sha256(video))
            )
            log = (session / event["profile_log"]).read_text(errors="replace")
            record["profiles"].append(dict(index=event["index"], phases=h3.parse_profile(log)))
        record["warmup_seconds"] = generations[0]["wall_seconds"]
        record["cached_generation"] = summarize(
            [event["wall_seconds"] for event in generations[1:]]
        )
        record["cached_denoise_callback"] = summarize(
            [event["denoise_callback_seconds"] for event in generations[1:]]
        )
        record["status"] = "completed"
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupted by user")
    except Exception as error:
        record.update(
            status="unsupported" if isinstance(error, h3.UnsupportedError) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        record["artifacts"] = [
            dict(
                file=str(path.relative_to(output)),
                sha256=h3.sha256(path),
                bytes=path.stat().st_size,
            )
            for path in sorted(output.rglob("*"))
            if path.is_file()
        ]
        record.update(finished_utc=h3.utc_now(), harness_wall_seconds=time.perf_counter() - started)
        h3.write_json(output / "run.json", record)
    return record


def main():
    cli = h3.parser()
    cli.description = __doc__
    cli.add_argument("--driver", type=Path, default=h3.ROOT / ".build/h3-session")
    args = cli.parse_args()
    try:
        record = execute(args)
    except FileExistsError:
        print(
            "Output directory already exists; choose a new immutable run directory", file=sys.stderr
        )
        return 2
    print(
        json.dumps(
            dict(
                status=record["status"],
                error=record.get("error"),
                record=h3.path_label(args.output_dir / "run.json"),
            ),
            indent=2,
        )
    )
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run pinned h3.c, with explicit optional diagnostics, in an exclusive evidence directory.

This is a process-level integration measurement, not a compilation-free kernel timer.
Run with ``uv run python integrations/h3/run.py --help``. No weights are distributed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

from m5kernels.locking import gpu_lock


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
MODEL_MANIFEST_SHA256 = "7597d33ca71af128705e802150b4b39395cf20f96bddc45d2041d668145c27be"
DIAGNOSTICS_PATCH_SHA256 = "42f3c0ea22782aa71afebe3acfa399435a6d08e0091dcaddd197290036962e7f"
DIAGNOSTICS_DIT_SHA256 = "552a071ccfcbd4923599368f67154012f47bef730431f51952e14c65b9861953"
DIAGNOSTICS_SOURCE_SHA256 = "63dd83536296ceacf6240bc168dcb693dcd9f1201e152cc4b4cd9d61c69bd283"
PROMPT = (
    "A red fox walks through fresh snow in a pine forest. Medium tracking shot, "
    "natural winter light, realistic fur, soft footsteps and wind."
)
BF16_FLAGS = [
    "--use-slower-bf16-mlp",
    "--use-slower-bf16-qkv",
    "--use-slower-bf16-attention-output",
]


class UnsupportedError(ValueError):
    """The requested backend cannot be established without changing the contract."""


class NumericalError(ValueError):
    """A sampled runtime tensor contains nonfinite values."""


def gpu_wait_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("GPU wait seconds must be finite and nonnegative")
    return seconds


@contextmanager
def wait_for_gpu_lock(timeout_seconds, record):
    """Retry acquisition only; retain the acquired context across the entire body."""
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("GPU wait seconds must be finite and nonnegative")
    started = time.monotonic()
    with ExitStack() as acquired:
        while True:
            try:
                acquired.enter_context(gpu_lock())
            except ValueError as error:
                elapsed = time.monotonic() - started
                record["gpu_queue_wait_seconds"] = elapsed
                if (
                    str(error) != "another m5bench process owns the GPU experiment lock"
                    or elapsed >= timeout_seconds
                ):
                    raise
                time.sleep(min(1.0, timeout_seconds - elapsed))
            else:
                record["gpu_queue_wait_seconds"] = time.monotonic() - started
                break
        yield


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def path_label(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def read_command(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(f"metadata command failed: {command[0]}: {result.stderr.strip()}")
    return result.stdout.strip()


def git_provenance(directory):
    revision = read_command(["git", "rev-parse", "HEAD"], cwd=directory)
    status = read_command(["git", "status", "--porcelain"], cwd=directory)
    files = read_command(["git", "ls-files", "-z"], cwd=directory).split("\0")
    digest = hashlib.sha256()
    for name in sorted(filter(None, files)):
        path = directory / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + sha256(path).encode() + b"\n")
    return dict(
        revision=revision, dirty=bool(status), status=status, source_sha256=digest.hexdigest()
    )


def validate_upstream(source, provenance, diagnostics=False):
    if not diagnostics:
        if provenance["revision"] != UPSTREAM_REVISION or provenance["dirty"]:
            raise ValueError("upstream must be clean at the pinned revision")
        return None
    if source.resolve() == (ROOT / ".cache/h3.c").resolve():
        raise ValueError("diagnostics requires a separate --source-dir, such as .cache/h3.c-diag")
    patch = Path(__file__).with_name("diagnostics.patch")
    changed = read_command(["git", "diff", "HEAD", "--name-only", "-z"], cwd=source)
    if (
        provenance["revision"] != UPSTREAM_REVISION
        or changed != "h3_dit.c\0"
        or any(line.startswith("??") for line in provenance["status"].splitlines())
        or provenance["source_sha256"] != DIAGNOSTICS_SOURCE_SHA256
        or sha256(source / "h3_dit.c") != DIAGNOSTICS_DIT_SHA256
        or sha256(patch) != DIAGNOSTICS_PATCH_SHA256
    ):
        raise ValueError("diagnostic source must match only the pinned diagnostics.patch change")
    return dict(
        patch_sha256=DIAGNOSTICS_PATCH_SHA256,
        expected_source_sha256=DIAGNOSTICS_SOURCE_SHA256,
        expected_h3_dit_sha256=DIAGNOSTICS_DIT_SHA256,
    )


def temporal_shape(seconds, width, height):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    requested = int(math.floor(seconds * 24 + 0.5))
    if requested < 1 or not math.isclose(requested / 24, seconds, abs_tol=1e-9):
        raise ValueError("seconds must represent an exact positive integer frame count at 24 fps")
    aligned = max(5, requested)
    aligned += (-(aligned - 5)) % 17
    video_t = ((aligned - 5) // 17) * 5 + 2
    audio_t = int(math.floor(aligned * 40 / 24 + 0.5))
    return dict(
        fps=24,
        requested_seconds=seconds,
        requested_frames=requested,
        aligned_frames=aligned,
        aligned_seconds=aligned / 24,
        video_latent_t=video_t,
        audio_latent_t=audio_t,
        video_tokens=video_t * (width // 32) * (height // 32),
        stereo_audio_tokens=2 * audio_t,
        text_tokens=None,
        packed_sequence_tokens=None,
        shape_evidence="source-derived h3_host.c; text/packed tokens not logged by pinned CLI",
    )


def build_command(args, source, model, original):
    command = [
        "./h3",
        "--profile",
        "--info",
        "-d",
        os.path.relpath(model, source),
        "-p",
        PROMPT,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--seconds",
        str(args.seconds),
        "--steps",
        str(args.steps),
        "--layers",
        "50",
        "--reuse",
        "1",
        "--core-reuse",
        "1",
        "--use-reference-rope",
        "--seed",
        str(args.seed),
        "-o",
        os.path.relpath(original, source),
    ]
    if args.mode == "bf16":
        command.extend(BF16_FLAGS)
    elif args.mode != "upstream-int8":
        raise ValueError("unknown mode")
    return command


def runtime_environment(diagnostics=False):
    # Upstream contains many environment-controlled experimental shortcuts.
    # Inheriting any would silently alter the benchmark recipe.
    inherited = sorted(key for key in os.environ if key.startswith("H3_"))
    if inherited:
        raise ValueError("unset inherited H3_* overrides before running: " + ", ".join(inherited))
    env = os.environ.copy()
    overrides = {"H3_NAX": "qkv-attn", "H3_NAX_DIAGNOSTIC": "1"}
    if diagnostics:
        overrides["H3_DIAGNOSTICS"] = "1"
    env.update(overrides)
    return env, overrides


def parse_diagnostics(log):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")

    records = []
    for number, line in enumerate(log.splitlines(), 1):
        if "h3 diag:" not in line:
            continue
        try:
            record = json.loads(line.split("h3 diag:", 1)[1], parse_constant=reject_constant)
            json.dumps(record, allow_nan=False)  # Also rejects overflowing exponents such as 1e999.
            if (
                not isinstance(record, dict)
                or type(record.get("schema")) is not int
                or record["schema"] != 1
                or record.get("kind") not in {"geometry", "operation", "latent"}
            ):
                raise ValueError("expected a schema-1 geometry, operation or latent object")
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"malformed diagnostic record at log line {number}: {error}"
            ) from error
        records.append(record)
    return records


def validate_diagnostics(records, args, shape):
    geometries = [item for item in records if item["kind"] == "geometry"]
    latents = [item for item in records if item["kind"] == "latent"]
    if len(geometries) != 1 or not latents:
        raise ValueError("diagnostics requires exactly one geometry and at least one latent record")
    geometry = geometries[0]
    expected = dict(
        video_rows=shape["video_tokens"],
        audio_rows=shape["stereo_audio_tokens"],
        video_condition_rows=0,
        audio_condition_rows=0,
        latent_t=shape["video_latent_t"],
        latent_h=args.height // 16,
        latent_w=args.width // 16,
        audio_t_per_channel=shape["audio_latent_t"],
        active_blocks=50,
        steps=args.steps,
        hidden=5376,
        heads=56,
        head_dim=128,
        ffn=14336,
        final_heads_bf16=1,
        ssd_streaming=0,
        token_reduction=0,
        core_reuse_interval=1,
    )
    if (
        any(
            type(geometry.get(key)) is not int or geometry[key] != value
            for key, value in expected.items()
        )
        or geometry.get("spatial_rope_scale") != 1
    ):
        raise ValueError("diagnostic geometry does not match the requested recipe")
    text = geometry.get("text_rows")
    if (
        type(text) is not int
        or text <= 0
        or geometry.get("sequence") != (text + shape["video_tokens"] + shape["stereo_audio_tokens"])
    ):
        raise ValueError("diagnostic packed/text row counts are inconsistent")
    if any(
        geometry.get(key) != int(args.mode == "upstream-int8")
        for key in ("configured_int8_qkv", "configured_int8_mlp", "configured_int8_output")
    ):
        raise UnsupportedError("diagnostic projection precision does not match the requested mode")
    for sample in latents:
        if sample.get("status") != "sampled" or sample.get("pending", 0) != 0:
            raise ValueError("diagnostic latent readback failed or has pending work")
        counts = [sample.get(key) for key in ("elements", "finite", "nonfinite", "zeros")]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("diagnostic latent counts are malformed")
        elements, finite, nonfinite, zeros = counts
        if nonfinite:
            raise NumericalError("diagnostic post-Euler latent contains nonfinite values")
        if not elements or finite + nonfinite != elements or zeros > finite:
            raise ValueError("diagnostic latent counts are inconsistent")
    return dict(
        text_tokens=text,
        packed_sequence_tokens=geometry["sequence"],
        sampled_latent_records=len(latents),
        finite_scope="sampled post-Euler target latents only; no all-layer numerical parity",
    )


def parse_profile(log):
    records = []
    for line in log.splitlines():
        if "h3 profile:" not in line:
            continue
        body = line.split("h3 profile:", 1)[1].lstrip()
        heading, separator, values = body.partition("wall=")
        if not separator:
            records.append({"raw": line, "parse_status": "unrecognized"})
            continue
        known = re.fullmatch(
            r"(.*?)\s+(GPU Euler denoise|RES denoise|Euler denoise|total|load)\s*", heading
        )
        fields = (
            list(known.groups()) if known else re.split(r"\s{2,}", heading.rstrip(), maxsplit=1)
        )
        if len(fields) != 2:
            records.append({"raw": line, "parse_status": "unrecognized"})
            continue
        record = dict(component=fields[0], phase=fields[1], raw=line)
        for key, number, unit in re.findall(
            r"([\w-]+)=\s*([0-9]+(?:\.[0-9]+)?)(GiB|s)?", "wall=" + values
        ):
            record[key.replace("-", "_") + ("_" + unit if unit else "")] = (
                float(number) if "." in number else int(number)
            )
        records.append(record)
    return records


def parse_progress(log):
    return [
        dict(phase=match[0].strip(), completed=int(match[1]), total=int(match[2]))
        for match in re.findall(r"(?:^|[\r\n])([^\r\n]+?)[ \t]+(\d+)/(\d+)[ \t]*", log)
    ]


def validate_video(probe, width, height, frames):
    video = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    if len(video) != 1:
        raise ValueError("expected exactly one video stream")
    stream = video[0]
    if stream.get("width") != width or stream.get("height") != height:
        raise ValueError("video dimensions do not match the benchmark")
    if Fraction(stream.get("avg_frame_rate", "0/1")) != 24:
        raise ValueError("video is not 24 fps")
    if int(stream.get("nb_read_frames", "0")) != frames:
        raise ValueError("decoded video frame count does not match the benchmark")
    duration = float(stream.get("duration", "nan"))
    if not math.isfinite(duration) or abs(duration - frames / 24) > 0.0001:
        raise ValueError("video stream duration does not match its exact frame count")
    return dict(width=width, height=height, frames=frames, fps=24, duration_seconds=duration)


def model_inventory(model, manifest_path):
    if not model.is_dir():
        raise ValueError("model directory is missing")
    if not manifest_path.is_file():
        raise ValueError("pinned Hugging Face model manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != MODEL_MANIFEST_SHA256:
        raise ValueError("model manifest SHA256 does not match the pinned model revision")
    manifest = json.loads(manifest_bytes)
    lookup = {entry["path"]: entry for entry in manifest if entry.get("type") == "file"}
    required = {
        "FL2VA/tokenizer/tokenizer.json",
        "FL2VA/transformer/config.json",
        "FL2VA/video_vae/config.json",
        "FL2VA/audio_vae/config.json",
    }
    for component in ("text_encoder", "transformer", "video_vae/source", "audio_vae"):
        prefix = f"FL2VA/{component}/"
        weights = {
            name
            for name in lookup
            if Path(name).parent.as_posix() == prefix.rstrip("/") and name.endswith(".safetensors")
        }
        if not weights:
            raise ValueError(f"manifest has no weights for {component}")
        directory = model / prefix
        actual = (
            {
                prefix + path.name
                for path in directory.iterdir()
                if path.name.endswith(".safetensors")
            }
            if directory.is_dir()
            else set()
        )
        unexpected = actual - weights
        if unexpected:
            raise ValueError("unexpected model weight shards: " + ", ".join(sorted(unexpected)))
        required.update(weights)
    entries = []
    for name in sorted(required):
        expected = lookup.get(name)
        path = model / name
        if not expected or not path.is_file():
            raise ValueError(f"required model file is missing: {name}")
        size = path.stat().st_size
        if size != expected["size"]:
            raise ValueError(f"model file size mismatch: {name}")
        actual_hash = sha256(path)
        expected_hash = expected.get("lfs", {}).get("oid")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"model SHA256 mismatch: {name}")
        if not expected_hash:
            digest = hashlib.sha1(f"blob {size}\0".encode() + path.read_bytes()).hexdigest()
            if digest != expected["oid"]:
                raise ValueError(f"model Git blob hash mismatch: {name}")
        entries.append(dict(path=name, bytes=size, sha256=actual_hash, expected=expected))
    return dict(
        repository="MiniMaxAI/MiniMax-H3",
        revision=MODEL_REVISION,
        manifest_sha256=manifest_digest,
        files=entries,
        verification="full file digest versus pinned API manifest; performed before process timing",
        cache_note="digest verification reads weights and may warm the filesystem cache",
    )


def run_logged(command, cwd, env, output_dir, name, stop_on_fallback=False):
    started = utc_now()
    clock = time.perf_counter()
    process = None
    fallback = False
    tail = b""
    result = dict(
        command=[path_label(v) if v.startswith("/") else v for v in command], cwd=path_label(cwd)
    )
    with (output_dir / f"{name}.log").open("xb") as log:
        with (output_dir / f"{name}.events.jsonl").open("x") as events:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while True:
                    block = os.read(process.stdout.fileno(), 65536)
                    if not block:
                        break
                    offset = log.tell()
                    log.write(block)
                    log.flush()
                    events.write(
                        json.dumps(
                            dict(
                                utc=utc_now(),
                                elapsed_seconds=time.perf_counter() - clock,
                                byte_offset=offset,
                                byte_count=len(block),
                            )
                        )
                        + "\n"
                    )
                    events.flush()
                    sys.stdout.write(block.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
                    tail = (tail + block)[-131072:]
                    if stop_on_fallback and not fallback and b"TensorOps compile failed:" in tail:
                        fallback = True
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                result["returncode"] = process.wait()
            finally:
                if process and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                if process and process.stdout:
                    process.stdout.close()
    result.update(
        started_utc=started,
        finished_utc=utc_now(),
        wall_seconds=time.perf_counter() - clock,
        log=f"{name}.log",
        log_sha256=sha256(output_dir / f"{name}.log"),
        fallback_detected=fallback,
    )
    return result


def ffprobe_command(ffprobe, video):
    return [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video),
    ]


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mode", choices=["bf16", "upstream-int8"], required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--source-dir", type=Path, default=ROOT / ".cache/h3.c")
    cli.add_argument(
        "--diagnostics",
        action="store_true",
        help="require the exact diagnostic patch in a separate source checkout",
    )
    cli.add_argument("--model-dir", type=Path, default=ROOT / ".cache/models/MiniMax-H3")
    cli.add_argument(
        "--model-manifest", type=Path, default=Path(__file__).with_name("model-files.json")
    )
    cli.add_argument("--seconds", type=float, default=5.0)
    cli.add_argument("--steps", type=int, default=20)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--width", type=int, default=512)
    cli.add_argument("--height", type=int, default=512)
    cli.add_argument(
        "--gpu-wait-seconds",
        type=gpu_wait_seconds,
        default=0.0,
        help="wait up to this many seconds for the GPU lock after model preflight (default: 0)",
    )
    return cli


def execute(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, model = args.source_dir.resolve(), args.model_dir.resolve()
    started = time.perf_counter()
    record = dict(
        schema_version=1,
        status="failed",
        started_utc=utc_now(),
        benchmark="fox-winter-t2va-v1",
        prompt=PROMPT,
        seed=args.seed,
        mode=args.mode,
        diagnostics=args.diagnostics,
        gpu_wait_timeout_seconds=args.gpu_wait_seconds,
        width=args.width,
        height=args.height,
        steps=args.steps,
        active_dit_blocks=50,
        reuse=1,
        core_reuse=1,
        token_reduction=False,
        use_reference_rope=True,
        internal_canvas=[args.width, args.height],
        environment=dict(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            python=platform.python_version(),
        ),
        measurements={},
        timing_contract={
            "process": "one CLI invocation including runtime shader compilation, loading, "
            "weight preparation, text encoding, denoising, VAE and media export",
            "compile_free_steady_state": "not measured",
            "offline_weight_preparation": "included in upstream DiT load; not separately exposed",
            "online_activation_quantization": "included in DiT denoise; not separately exposed",
            "kernel_only": "not measured",
            "gpu_queue_wait": "after model preflight; excluded from generation process timing",
            "profile": "upstream context counters; total overlaps phase marks; do not sum both",
            "quality": "video structure checked; numerical/visual/audio quality not established",
            "diagnostics": "host timestamps, logging and latent scans add overhead; host_api_ms "
            "is CPU encoding/API time, not GPU execution time; keep separate from baseline latency"
            if args.diagnostics
            else "disabled",
        },
        precision_contract={
            "storage": "original checkpoint weights; no persisted quantized checkpoint",
            "execution": (
                "BF16 overrides for DiT QKV, MLP and attention-output projections; "
                "remaining upstream mixed floating-point path"
                if args.mode == "bf16"
                else "upstream on-load INT8 DiT QKV, MLP and attention-output projections; "
                "remaining upstream mixed floating-point path"
            ),
            "dispatch_evidence": "pinned source selection and required M5; no per-op runtime trace",
            "implementation_error": "not measured",
            "quantization_error": "not measured",
        },
    )
    try:
        if args.width < 256 or args.height < 256 or args.width % 32 or args.height % 32:
            raise ValueError("width and height must be multiples of 32 and at least 256")
        if not 2 <= args.steps <= 1000 or not 0 <= args.seed < 2**64:
            raise ValueError("steps must be in [2,1000] and seed an unsigned 64-bit integer")
        record["shape"] = temporal_shape(args.seconds, args.width, args.height)
        record["harness"] = git_provenance(ROOT)
        record["runner_sha256"] = sha256(Path(__file__))
        record["upstream"] = git_provenance(source)
        record["diagnostic_patch"] = validate_upstream(source, record["upstream"], args.diagnostics)
        record["upstream"]["binary_sha256"] = sha256(source / "h3")
        record["upstream"]["repository"] = "https://github.com/antirez/h3.c"
        env, record["runtime_overrides"] = (
            runtime_environment(True) if args.diagnostics else runtime_environment()
        )
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise UnsupportedError("Metal GPU validation unavailable on this host")
        record["environment"]["chip"] = read_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        record["environment"]["memory_bytes"] = int(read_command(["sysctl", "-n", "hw.memsize"]))
        record["environment"]["compiler"] = read_command(["clang", "--version"])
        record["environment"]["xcode"] = read_command(["xcodebuild", "-version"])
        if "M5" not in record["environment"]["chip"]:
            raise UnsupportedError("this pinned TensorOps benchmark requires an Apple M5 GPU")
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise ValueError("ffmpeg and ffprobe are required")
        record["environment"]["ffmpeg"] = read_command([ffmpeg, "-version"]).splitlines()[0]
        record["environment"]["ffprobe"] = read_command([ffprobe, "-version"]).splitlines()[0]
        record["model"] = model_inventory(model, args.model_manifest)
        record["preflight_seconds"] = time.perf_counter() - started
        original, exact = output / "original.mp4", output / "video-5s.mp4"
        if args.seconds != 5:
            exact = output / "video-exact.mp4"
        command = build_command(args, source, model, original)
        write_json(
            output / "request.json",
            record | {"command": [path_label(v) if v.startswith("/") else v for v in command]},
        )
        with wait_for_gpu_lock(args.gpu_wait_seconds, record):
            measurement = run_logged(command, source, env, output, "generation", True)
            record["measurements"]["generation"] = measurement
            log = (output / "generation.log").read_text(errors="replace")
            record["profile"] = parse_profile(log)
            record["progress"] = parse_progress(log)
            record["diagnostic_records"] = parse_diagnostics(log)
            record["shape_log_lines"] = [
                line
                for line in log.splitlines()
                if re.search(r"(?:tokens|sequence|layout|rows)\s*[:=]", line, re.IGNORECASE)
            ]
            if measurement["fallback_detected"]:
                raise UnsupportedError("upstream TensorOps compilation failed; fallback rejected")
            if measurement["returncode"]:
                raise ValueError(f"upstream exited with code {measurement['returncode']}")
            if args.diagnostics:
                record["diagnostic_validation"] = validate_diagnostics(
                    record["diagnostic_records"], args, record["shape"]
                )
            for name, video, frames in (
                ("original", original, record["shape"]["aligned_frames"]),
                ("exact", exact, record["shape"]["requested_frames"]),
            ):
                if name == "exact":
                    trim = [
                        ffmpeg,
                        "-nostdin",
                        "-n",
                        "-i",
                        os.path.relpath(original, source),
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a?",
                        "-frames:v",
                        str(frames),
                        "-t",
                        str(args.seconds),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-movflags",
                        "+faststart",
                        os.path.relpath(exact, source),
                    ]
                    result = run_logged(trim, source, env, output, "trim")
                    record["measurements"]["trim"] = result
                    if result["returncode"]:
                        raise ValueError("exact-duration export failed")
                result = run_logged(
                    ffprobe_command(ffprobe, os.path.relpath(video, source)),
                    source,
                    env,
                    output,
                    name + "-probe",
                )
                record["measurements"][name + "_probe"] = result
                if result["returncode"]:
                    raise ValueError(f"{name} video ffprobe failed")
                probe = json.loads((output / (name + "-probe.log")).read_text())
                record[name + "_video"] = validate_video(probe, args.width, args.height, frames)
                record[name + "_video"].update(file=video.name, sha256=sha256(video))
        record["status"] = "completed"
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupted by user")
    except Exception as error:
        record.update(
            status="incorrect"
            if isinstance(error, NumericalError)
            else "unsupported"
            if isinstance(error, UnsupportedError)
            else "failed",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        record.update(finished_utc=utc_now(), harness_wall_seconds=time.perf_counter() - started)
        write_json(output / "run.json", record)
    return record


def main():
    args = parser().parse_args()
    try:
        record = execute(args)
    except FileExistsError:
        print(
            "Output directory already exists; choose a new immutable run directory", file=sys.stderr
        )
        return 2
    print(
        json.dumps(
            {
                "status": record["status"],
                "error": record.get("error"),
                "record": path_label(args.output_dir / "run.json"),
            },
            indent=2,
        )
    )
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

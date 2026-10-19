"""Run the pinned imported INT8/H256 checkpoint and optionally verify replay parity.

This is separate from the frozen BF16/INT8/H256 experiment runner. It never
weakens that runner's original-checkpoint guard. No weight preparation occurs
inside the imported native DiT load path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

import derived_checkpoint as derived
import prepared_checkpoint as prepared
import run as base
import run_fc2_experiment as experiment


PATCH_SHA256 = "1cad08a5bddc2dfa37c29d2c00551fe467a032d358d1287072f93bfd9dc74c67"
SOURCE_SHA256 = "054ec73eb454e44dc75d71bf83fa53a7129e5bd1e30fb304236f545f92f9cd7e"


def validate_source(source, provenance):
    changed = set(
        filter(
            None,
            base.read_command(["git", "diff", "HEAD", "--name-only", "-z"], source).split("\0"),
        )
    )
    if (
        source.resolve()
        in {(base.ROOT / ".cache/h3.c").resolve(), (base.ROOT / ".cache/h3.c-hadamard").resolve()}
        or provenance["revision"] != base.UPSTREAM_REVISION
        or provenance["source_sha256"] != SOURCE_SHA256
        or changed != experiment.CHANGED_FILES
        or any(line.startswith("??") for line in provenance["status"].splitlines())
        or base.sha256(Path(__file__).with_name("prepared-checkpoint.patch")) != PATCH_SHA256
    ):
        raise ValueError("prepared runtime must exactly match the pinned prepared-checkpoint.patch")
    return dict(source_sha256=SOURCE_SHA256, patch_sha256=PATCH_SHA256)


def validate_load_marker(log):
    matches = re.findall(
        r"h3 prepared load: projections=(\d+) verified_payload_load_wall_seconds=([0-9.]+) weight_rotation_quantization_seconds=([0-9.]+)",
        log,
    )
    if len(matches) != 1 or int(matches[0][0]) != 200 or float(matches[0][2]) != 0:
        raise ValueError(
            "imported runtime must load exactly 200 prepared projections without requantization"
        )
    if "h3 precision: FC2 plain normalized H256" not in log:
        raise base.UnsupportedError("Hadamard FC2 runtime policy marker missing")
    seconds = float(matches[0][1])
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid verified payload load time")
    return dict(
        projections=200,
        verified_payload_load_wall_seconds=seconds,
        weight_rotation_quantization_seconds=0.0,
    )


def verify_parity(record, reference_directory):
    reference_path = reference_directory / "run.json"
    reference = experiment.load_json(reference_path)
    for key in (
        "status",
        "mode",
        "seed",
        "prompt",
        "width",
        "height",
        "seconds",
        "steps",
        "active_dit_blocks",
        "reuse",
        "core_reuse",
        "reference_rope",
        "capture_latents",
    ):
        expected = "completed" if key == "status" else record[key]
        if reference.get(key) != expected:
            raise ValueError(f"import parity reference recipe differs: {key}")
    if reference.get("patch", {}).get("source_sha256") != experiment.SOURCE_SHA256:
        raise ValueError("import parity reference must use the frozen on-load H256 producer")
    verified = experiment.validate_latent_capture(
        reference_directory / "latents", "hadamard-fc2", record["seed"]
    )
    if verified != reference.get("latent_validation"):
        raise ValueError("parity reference latent files differ from immutable record")
    candidate_tensors = record["latent_validation"]["tensors"]
    reference_tensors = verified["tensors"]
    comparisons = [
        dict(
            file=c["file"],
            candidate_sha256=c["sha256"],
            reference_sha256=r["sha256"],
            byte_exact=c == r,
        )
        for c, r in zip(candidate_tensors, reference_tensors, strict=True)
    ]
    media = []
    for key in ("original_video", "exact_video"):
        expected = reference[key]
        if base.sha256(reference_directory / expected["file"]) != expected["sha256"]:
            raise ValueError("parity reference media differs from immutable record")
        media.append(
            dict(
                file=record[key]["file"],
                candidate_sha256=record[key]["sha256"],
                reference_sha256=expected["sha256"],
                byte_exact=record[key]["sha256"] == expected["sha256"],
            )
        )
    return dict(
        reference_run_sha256=base.sha256(reference_path),
        latent_comparisons=comparisons,
        media_comparisons=media,
        all_latents_byte_exact=all(x["byte_exact"] for x in comparisons),
        all_media_byte_exact=all(x["byte_exact"] for x in media),
        scope="20 packed Euler video/audio states, final unpacked VAE-input latents and both MP4 files; replay parity, not model quality",
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--checkpoint-manifest-sha256", required=True)
    p.add_argument("--source-dir", type=Path, default=base.ROOT / ".cache/h3.c-hadamard-cache")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture-latents", action="store_true")
    p.add_argument("--parity-reference", type=Path)
    p.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    return p


def execute(args):
    output, source, checkpoint = (
        args.output_dir.resolve(),
        args.source_dir.resolve(),
        args.checkpoint_dir.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(
        schema_version=1,
        status="failed",
        started_utc=base.utc_now(),
        benchmark="fox-winter-t2va-v1",
        mode="hadamard-fc2",
        prepared_checkpoint=True,
        seed=args.seed,
        prompt=base.PROMPT,
        width=512,
        height=512,
        seconds=5,
        steps=20,
        active_dit_blocks=50,
        reuse=1,
        core_reuse=1,
        reference_rope=True,
        capture_latents=args.capture_latents,
        diagnostic=args.capture_latents,
        measurements={},
        quality="not evaluated",
        original_checkpoint_retained=True,
    )
    record["timing_contract"] = dict(
        scope="diagnostic captured generation"
        if args.capture_latents
        else "single full process including verified prepared loading",
        gpu_lock="preflight, compilation, complete process, capture and media verification",
        offline_weight_preparation="not in this run; see checkpoint prepare.json",
        prepared_reload="reported separately from native load marker",
        online_activation_preparation="native quantization; FC2 adds H256, included in denoise",
        resident_compile_free_latency="not measured",
        kernel_only="not measured",
    )
    try:
        if not 0 <= args.seed < 2**63 or not prepared.is_sha(args.checkpoint_manifest_sha256):
            raise ValueError(
                "valid nonnegative signed-64 seed and lowercase manifest SHA256 required"
            )
        if args.parity_reference and not args.capture_latents:
            raise ValueError("parity reference requires all latent captures")
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            preflight = time.perf_counter()
            record["harness"] = base.git_provenance(base.ROOT)
            record["harness_files"] = {
                name: base.sha256(Path(__file__).with_name(name))
                for name in (
                    "run_prepared_checkpoint.py",
                    "derived_checkpoint.py",
                    "prepared_checkpoint.py",
                    "run_fc2_experiment.py",
                    "run.py",
                    "prepared-checkpoint.patch",
                )
            }
            record["upstream"] = base.git_provenance(source)
            record["patch"] = validate_source(source, record["upstream"])
            completed = experiment.load_json(checkpoint / "derive.json")
            if (
                completed.get("status") != "completed"
                or completed.get("derived_manifest_sha256") != args.checkpoint_manifest_sha256
            ):
                raise ValueError(
                    "a completed derived checkpoint with the explicit manifest hash is required"
                )
            record["model"] = derived.verify_derived(checkpoint, args.checkpoint_manifest_sha256)
            env, overrides = base.runtime_environment()
            overrides.update(
                H3_HADAMARD_FC2="1",
                H3_PREPARED_DIT_CACHE=str(checkpoint / "weights/manifest.json"),
                H3_PREPARED_DIT_MANIFEST_SHA256=record["model"]["prepared_manifest_sha256"],
            )
            if args.capture_latents:
                (output / "latents").mkdir()
                overrides.update(
                    H3_LATENT_CAPTURE_DIR=str(output / "latents"),
                    H3_CAPTURE_SEED=str(args.seed),
                    H3_GPU_SAMPLER="1",
                    H3_GPU_SAMPLER_WINDOW="1",
                )
            env.update(overrides)
            record["runtime_overrides"] = {
                k: base.path_label(v) if k.endswith(("_DIR", "_CACHE")) else v
                for k, v in overrides.items()
            }
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise base.UnsupportedError("native Metal validation unavailable on this host")
            record["environment"] = dict(
                system=platform.system(),
                release=platform.release(),
                machine=platform.machine(),
                python=platform.python_version(),
                numpy=np.__version__,
                chip=base.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                memory_bytes=int(base.read_command(["sysctl", "-n", "hw.memsize"])),
                compiler=base.read_command(["clang", "--version"]),
                xcode=base.read_command(["xcodebuild", "-version"]),
            )
            if "M5" not in record["environment"]["chip"]:
                raise base.UnsupportedError("prepared runtime requires Apple M5 TensorOps")
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise ValueError("FFmpeg and FFprobe are required")
            record["environment"].update(
                ffmpeg=base.read_command([ffmpeg, "-version"]).splitlines()[0],
                ffprobe=base.read_command([ffprobe, "-version"]).splitlines()[0],
            )
            build = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
            record["measurements"]["build"] = build
            if build["returncode"]:
                raise base.UnsupportedError("prepared native runtime build failed")
            validate_source(source, base.git_provenance(source))
            record["upstream"].update(
                binary_sha256=base.sha256(source / "h3"),
                library_sha256=base.sha256(source / "libh3.a"),
            )
            record["preflight_seconds"] = time.perf_counter() - preflight
            recipe = SimpleNamespace(
                mode="upstream-int8", width=512, height=512, seconds=5.0, steps=20, seed=args.seed
            )
            command = base.build_command(
                recipe, source, checkpoint / "model", output / "original.mp4"
            )
            record["shape"] = base.temporal_shape(5, 512, 512)
            base.write_json(output / "request.json", record | {"command": command})
            process = base.run_logged(command, source, env, output, "generation", True)
            record["measurements"]["generation"] = process
            log = (output / "generation.log").read_text(errors="replace")
            record["profile"], record["progress"] = (
                base.parse_profile(log),
                base.parse_progress(log),
            )
            if process["fallback_detected"]:
                raise base.UnsupportedError("TensorOps compilation failed; fallback rejected")
            if process["returncode"]:
                raise ValueError(f"prepared native generation exited {process['returncode']}")
            record["prepared_load"] = validate_load_marker(log)
            if args.capture_latents:
                record["latent_validation"] = experiment.validate_latent_capture(
                    output / "latents", "hadamard-fc2", args.seed
                )
            experiment.export_video(output, source, env, ffmpeg, ffprobe, record)
            if args.parity_reference:
                record["replay_parity"] = verify_parity(record, args.parity_reference.resolve())
                if (
                    not record["replay_parity"]["all_latents_byte_exact"]
                    or not record["replay_parity"]["all_media_byte_exact"]
                ):
                    raise base.NumericalError(
                        "prepared checkpoint replay differs from on-load H256 reference; evidence retained"
                    )
        record["status"] = "completed"
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupted by user")
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
    print(json.dumps({key: record.get(key) for key in ("status", "error", "replay_parity")}))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

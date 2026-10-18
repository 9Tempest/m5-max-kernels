"""Pinned native H3 FC2 precision experiment; GPU runs are serialized end to end.

Capture runs are diagnostics, not performance samples. The original checkpoint
is read-only; this runner never exports or deletes model weights.
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

import numpy as np

import run as base


PATCH_SHA256 = "84065e6139c842aee2cb6d037a6c72ecfa9895e07b1696e9bf638bde3a5a8d1e"
SOURCE_SHA256 = "833076975eeaaf14b0a9db01f6237d7de77930034222b1dd3b85cca7721f04ec"
CHANGED_FILES = {"h3_dit.c", "h3_gpu.h", "h3_gpu.m", "h3_shaders.metal"}
ROOT = base.ROOT


def validate_source(source, provenance):
    if source.resolve() == (ROOT / ".cache/h3.c").resolve():
        raise ValueError("FC2 experiments require a separate patched checkout")
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
        or base.sha256(Path(__file__).with_name("hadamard.patch")) != PATCH_SHA256
    ):
        raise ValueError("FC2 source must exactly match the pinned hadamard.patch")
    return {"patch_sha256": PATCH_SHA256, "source_sha256": SOURCE_SHA256}


def finite_file(path, dtype, elements):
    if path.stat().st_size != elements * np.dtype(dtype).itemsize:
        raise ValueError(f"wrong tensor size: {path.name}")
    values = np.memmap(path, dtype=dtype, mode="r")
    for start in range(0, elements, 1024 * 1024):
        part = values[start : start + 1024 * 1024]
        finite = np.all((part & 0x7F80) != 0x7F80) if dtype == "<u2" else np.isfinite(part).all()
        if not finite:
            raise base.NumericalError(f"nonfinite captured tensor: {path.name}")
    return {"file": path.name, "elements": elements, "sha256": base.sha256(path), "finite": True}


def load_json(path):
    value = json.loads(
        path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
    )
    json.dumps(value, allow_nan=False)
    return value


def validate_fc2_capture(directory, mode):
    expected = {
        f"fc2-step{s:02d}-block{b:02d}.{suffix}"
        for s in (0, 10, 19)
        for b in (0, 24, 49)
        for suffix in ("bf16", "json")
    }
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError("FC2 capture must contain exactly steps 0/10/19 × blocks 0/24/49")
    records = []
    for step in (0, 10, 19):
        for block in (0, 24, 49):
            prefix = f"fc2-step{step:02d}-block{block:02d}"
            metadata = load_json(directory / f"{prefix}.json")
            wanted = dict(
                schema=1,
                step=step,
                block=block,
                packed_rows=9914,
                width=14336,
                int8_qkv=1,
                int8_attention_out=1,
                hadamard_fc2=int(mode == "hadamard-fc2"),
                nonfinite=0,
            )
            if any(type(metadata.get(k)) is not int or metadata[k] != v for k, v in wanted.items()):
                raise ValueError("FC2 metadata does not match the fixed benchmark recipe")
            selected, cursor = [], 0
            for segment in metadata["segments"]:
                first, stop = segment["start"], segment["stop"]
                if (
                    type(first) is not int
                    or type(stop) is not int
                    or first != cursor
                    or not first <= stop <= 9914
                ):
                    raise ValueError("malformed contiguous modality segments")
                cursor = stop
                if stop > first:
                    selected.extend(
                        i
                        for i in (first, first + (stop - first) // 2, stop - 1)
                        if i not in selected
                    )
            if cursor != 9914:
                raise ValueError("incomplete packed modality coverage")
            for j in range(1, 8):
                value = (9913 * j) // 8
                if value not in selected:
                    selected.append(value)
            if metadata.get("row_ids") != selected or metadata.get("sample_rows") != len(selected):
                raise ValueError("FC2 sampled row IDs do not match the source selection rule")
            records.append(
                {
                    "metadata": metadata,
                    "tensor": finite_file(
                        directory / f"{prefix}.bf16", "<u2", len(selected) * 14336
                    ),
                    "metadata_sha256": base.sha256(directory / f"{prefix}.json"),
                }
            )
    return records


def validate_latent_capture(directory, mode, seed):
    expected = {f"euler-step{s:02d}-{kind}.f32" for s in range(20) for kind in ("video", "audio")}
    expected |= {"final-video.f32", "final-audio.f32", "latent-metadata.json"}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError("latent capture needs all 20 Euler states and final unpacked video/audio")
    meta = load_json(directory / "latent-metadata.json")
    wanted = dict(
        schema=1,
        seed=seed,
        completed_steps=20,
        final_video_shape=[24, 37, 32, 32],
        final_audio_shape=[32, 2, 207],
        per_step_video_shape=[9472, 96],
        per_step_audio_shape=[414, 32],
        int8_mlp=int(mode != "bf16"),
        int8_qkv=int(mode != "bf16"),
        int8_attention_out=int(mode != "bf16"),
        hadamard_fc2=int(mode == "hadamard-fc2"),
        finite=True,
    )
    if any(meta.get(k) != v or type(meta.get(k)) is not type(v) for k, v in wanted.items()):
        raise ValueError("latent metadata does not match the fixed recipe/precision")
    if (
        meta.get("final_layout") != "unpacked_contiguous_VAE_input_before_decoder"
        or meta.get("per_step_layout") != "packed_model_rows_after_Euler_before_unpack"
    ):
        raise ValueError("unknown captured latent layout")
    records = []
    for name in sorted(expected - {"latent-metadata.json"}):
        count = 24 * 37 * 32 * 32 if "video" in name else 32 * 2 * 207
        records.append(finite_file(directory / name, "<f4", count))
    return {
        "metadata": meta,
        "metadata_sha256": base.sha256(directory / "latent-metadata.json"),
        "tensors": records,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("bf16", "upstream-int8", "hadamard-fc2"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=ROOT / ".cache/h3.c-hadamard")
    p.add_argument("--model-dir", type=Path, default=ROOT / ".cache/models/MiniMax-H3")
    p.add_argument("--capture-fc2", action="store_true")
    p.add_argument("--capture-latents", action="store_true")
    p.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    return p


def export_video(output, source, env, ffmpeg, ffprobe, record):
    original, exact = output / "original.mp4", output / "video-5s.mp4"
    for name, video, frames in (("original", original, 124), ("exact", exact, 120)):
        if name == "exact":
            command = [
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
                "120",
                "-t",
                "5",
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
            result = base.run_logged(command, source, env, output, "trim")
            record["measurements"]["trim"] = result
            if result["returncode"]:
                raise ValueError("exact five-second video export failed")
        result = base.run_logged(
            base.ffprobe_command(ffprobe, os.path.relpath(video, source)),
            source,
            env,
            output,
            name + "-probe",
        )
        record["measurements"][name + "_probe"] = result
        if result["returncode"]:
            raise ValueError(f"{name} ffprobe failed")
        probe = load_json(output / (name + "-probe.log"))
        record[name + "_video"] = base.validate_video(probe, 512, 512, frames) | {
            "file": video.name,
            "sha256": base.sha256(video),
        }


def execute(args):
    output, source, model = (
        args.output_dir.resolve(),
        args.source_dir.resolve(),
        args.model_dir.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    diagnostic = args.capture_fc2 or args.capture_latents
    record = dict(
        schema_version=1,
        status="failed",
        started_utc=base.utc_now(),
        benchmark="fox-winter-t2va-v1",
        mode=args.mode,
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
        diagnostic=diagnostic,
        capture_fc2=args.capture_fc2,
        capture_latents=args.capture_latents,
        measurements={},
        quality="not evaluated",
        checkpoint_replacement_eligible=False,
    )
    record["timing_contract"] = {
        "scope": "diagnostic capture with extra waits/readbacks; not a performance sample"
        if diagnostic
        else "single complete process invocation; includes shader setup, model loading and offline weight preparation",
        "gpu_lock": "covers source/model preflight, compilation, generation, capture checks and media validation",
        "model_preflight": "full pinned weight digests read before process timing; warms file cache",
        "resident_compile_free_latency": "not measured",
        "kernel_only": "not measured",
        "offline_preparation": "FC2 H256/INT8 weight preparation included in DiT load",
        "online_preparation": "FC2 H256/INT8 activation preparation included in denoise",
    }
    try:
        if not 0 <= args.seed < 2**63:
            raise ValueError("seed must fit a nonnegative signed 64-bit integer")
        if args.mode == "bf16" and args.capture_fc2:
            raise ValueError(
                "FC2 capture describes native INT8 FC1 trajectories; incompatible with BF16 mode"
            )
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            preflight = time.perf_counter()
            record["harness"] = base.git_provenance(ROOT)
            record["harness_files"] = {
                name: base.sha256(Path(__file__).with_name(name))
                for name in (
                    "run_fc2_experiment.py",
                    "run.py",
                    "fc2_validate.c",
                    "fc2_analyze.py",
                    "hadamard.patch",
                )
            }
            record["upstream"] = base.git_provenance(source)
            record["patch"] = validate_source(source, record["upstream"])
            env, overrides = base.runtime_environment()
            overrides.update(H3_HADAMARD_FC2=str(int(args.mode == "hadamard-fc2")))
            if args.capture_fc2:
                (output / "fc2").mkdir()
                overrides["H3_FC2_CAPTURE_DIR"] = str(output / "fc2")
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
                k: base.path_label(v) if k.endswith("_DIR") else v for k, v in overrides.items()
            }
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise base.UnsupportedError("Metal validation unavailable on this host")
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
                raise base.UnsupportedError("FC2 experiment requires Apple M5 TensorOps")
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise ValueError("ffmpeg and ffprobe are required")
            record["environment"].update(
                ffmpeg=base.read_command([ffmpeg, "-version"]).splitlines()[0],
                ffprobe=base.read_command([ffprobe, "-version"]).splitlines()[0],
            )
            # Rebuild from the pinned source, avoiding a stale/external binary.
            compiled = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
            record["measurements"]["build"] = compiled
            if compiled["returncode"]:
                raise base.UnsupportedError("native source compile failed")
            validate_source(source, base.git_provenance(source))
            record["upstream"].update(
                binary_sha256=base.sha256(source / "h3"),
                library_sha256=base.sha256(source / "libh3.a"),
                repository="https://github.com/antirez/h3.c",
            )
            driver = ROOT / ".build/h3-fc2-validate"
            record["numerical_driver"] = {
                "path": base.path_label(driver),
                "binary_sha256": base.sha256(driver) if driver.is_file() else None,
                "used_in_generation": False,
            }
            record["model"] = base.model_inventory(
                model, Path(__file__).with_name("model-files.json")
            )
            record["preflight_seconds"] = time.perf_counter() - preflight
            recipe = SimpleNamespace(
                mode="bf16" if args.mode == "bf16" else "upstream-int8",
                width=512,
                height=512,
                seconds=5.0,
                steps=20,
                seed=args.seed,
            )
            command = base.build_command(recipe, source, model, output / "original.mp4")
            record["shape"] = base.temporal_shape(5, 512, 512)
            base.write_json(output / "request.json", record | {"command": command})
            result = base.run_logged(command, source, env, output, "generation", True)
            record["measurements"]["generation"] = result
            log = (output / "generation.log").read_text(errors="replace")
            record["profile"] = base.parse_profile(log)
            record["progress"] = base.parse_progress(log)
            if result["fallback_detected"]:
                raise base.UnsupportedError("TensorOps compilation failed; fallback rejected")
            if result["returncode"]:
                raise ValueError(f"native generation exited {result['returncode']}")
            if args.mode == "hadamard-fc2" and "h3 precision: FC2 plain normalized H256" not in log:
                raise base.UnsupportedError("Hadamard FC2 runtime policy marker missing")
            if args.capture_fc2:
                record["fc2_validation"] = validate_fc2_capture(output / "fc2", args.mode)
            if args.capture_latents:
                record["latent_validation"] = validate_latent_capture(
                    output / "latents", args.mode, args.seed
                )
            export_video(output, source, env, ffmpeg, ffprobe, record)
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

"""Run a frozen diversity prompt with original BF16 or a verified imported H256 model.

No mutable prompt override and no checkpoint fallback. Existing runners remain
unchanged. The common GPU lock covers preflight/build/generation/media checks.
"""

import argparse
import json
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace

import benchmark_catalog as catalog
import derived_checkpoint as derived
import run as base
import run_fc2_experiment as experiment
import run_prepared_checkpoint as imported


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-id", required=True)
    p.add_argument("--catalog", type=Path, default=catalog.DEFAULT)
    p.add_argument("--mode", choices=("bf16", "hadamard-fc2"), required=True)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--checkpoint-manifest-sha256")
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--capture-latents", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def configuration(args):
    benchmark = catalog.load_benchmark(args.benchmark_id, args.catalog)
    prepared = args.mode == "hadamard-fc2"
    if prepared:
        if not args.checkpoint_dir or not args.checkpoint_manifest_sha256 or args.model_dir:
            raise ValueError(
                "H256 requires --checkpoint-dir and explicit manifest SHA; no --model-dir fallback"
            )
        if not imported.prepared.is_sha(args.checkpoint_manifest_sha256):
            raise ValueError("invalid derived checkpoint SHA256")
        source = args.source_dir or base.ROOT / ".cache/h3.c-hadamard-cache"
        model = args.checkpoint_dir.resolve() / "model"
    else:
        if args.checkpoint_dir or args.checkpoint_manifest_sha256:
            raise ValueError("BF16 requires the original model, not a prepared checkpoint")
        source = args.source_dir or base.ROOT / ".cache/h3.c-hadamard"
        model = args.model_dir or base.ROOT / ".cache/models/MiniMax-H3"
    return benchmark, source.resolve(), model.resolve()


def build_command(benchmark, mode, source, model, original):
    recipe = SimpleNamespace(
        mode="bf16" if mode == "bf16" else "upstream-int8",
        width=benchmark["width"],
        height=benchmark["height"],
        seconds=float(benchmark["seconds"]),
        steps=benchmark["steps"],
        seed=benchmark["seed"],
    )
    command = base.build_command(recipe, source, model, original)
    if command.count("-p") != 1:
        raise ValueError("upstream command prompt flag changed")
    command[command.index("-p") + 1] = benchmark["prompt"]
    return command


def execute(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(
        schema_version=1,
        status="failed",
        measurements={},
        mode=args.mode,
        quality="not evaluated",
        original_checkpoint_retained=True,
        started_utc=base.utc_now(),
        capture_latents=args.capture_latents,
        diagnostic=args.capture_latents,
        prepared_checkpoint=args.mode == "hadamard-fc2",
    )
    try:
        benchmark, source, model = configuration(args)
        record.update(
            {
                key: benchmark[key]
                for key in (
                    "seed",
                    "prompt",
                    "width",
                    "height",
                    "seconds",
                    "fps",
                    "steps",
                    "active_dit_blocks",
                    "reuse",
                    "core_reuse",
                    "reference_rope",
                )
            }
        )
        record.update(benchmark=benchmark["id"], catalog=benchmark)
        record["timing_contract"] = dict(
            scope="diagnostic capture with extra waits/readbacks; not a performance sample"
            if args.capture_latents
            else "single complete process including startup/loading/export",
            gpu_lock="preflight/build/generation/capture/media validation",
            model_preflight="full weight verification before process timing warms file cache",
            offline_weight_preparation="separate prepared checkpoint export"
            if args.mode != "bf16"
            else "none",
            online_activation_preparation="included in generation; H256 FC2 for prepared mode",
            resident_compile_free_latency="not measured",
        )
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            preflight = time.perf_counter()
            record["harness"] = base.git_provenance(base.ROOT)
            record["harness_files"] = {
                name: base.sha256(Path(__file__).with_name(name))
                for name in (
                    "run_benchmark.py",
                    "benchmark_catalog.py",
                    "benchmarks.json",
                    "run.py",
                    "run_fc2_experiment.py",
                    "run_prepared_checkpoint.py",
                    "derived_checkpoint.py",
                    "prepared_checkpoint.py",
                )
            }
            record["upstream"] = base.git_provenance(source)
            source_guard = (
                experiment.validate_source if args.mode == "bf16" else imported.validate_source
            )
            record["patch"] = source_guard(source, record["upstream"])
            env, overrides = base.runtime_environment()
            overrides["H3_HADAMARD_FC2"] = str(int(args.mode == "hadamard-fc2"))
            if args.mode == "bf16":
                record["model"] = base.model_inventory(
                    model, Path(__file__).with_name("model-files.json")
                )
            else:
                checkpoint = args.checkpoint_dir.resolve()
                complete = experiment.load_json(checkpoint / "derive.json")
                if (
                    complete.get("status") != "completed"
                    or complete.get("derived_manifest_sha256") != args.checkpoint_manifest_sha256
                ):
                    raise ValueError(
                        "derived checkpoint must be completed with the explicit manifest hash"
                    )
                record["model"] = derived.verify_derived(
                    checkpoint, args.checkpoint_manifest_sha256
                )
                manifest = experiment.load_json(checkpoint / "derived-checkpoint.json")
                record["model"]["source_model"] = manifest["source_model"]
                record["derived_checkpoint"] = {
                    "manifest_file": "derived-checkpoint.json",
                    "manifest_sha256": args.checkpoint_manifest_sha256,
                    "source_model": manifest["source_model"],
                    "producer": manifest["producer"],
                    "policy_id": manifest["policy_id"],
                    "prepared_manifest_sha256": manifest["prepared_manifest_sha256"],
                }
                with (output / "derived-checkpoint.json").open("xb") as saved:
                    saved.write((checkpoint / "derived-checkpoint.json").read_bytes())
                if (
                    base.sha256(output / "derived-checkpoint.json")
                    != args.checkpoint_manifest_sha256
                ):
                    raise ValueError("derived manifest changed after full checkpoint verification")
                overrides.update(
                    H3_PREPARED_DIT_CACHE=str(checkpoint / "weights/manifest.json"),
                    H3_PREPARED_DIT_MANIFEST_SHA256=record["model"]["prepared_manifest_sha256"],
                )
            if args.capture_latents:
                (output / "latents").mkdir()
                overrides.update(
                    H3_LATENT_CAPTURE_DIR=str(output / "latents"),
                    H3_CAPTURE_SEED=str(benchmark["seed"]),
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
                chip=base.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                memory_bytes=int(base.read_command(["sysctl", "-n", "hw.memsize"])),
                compiler=base.read_command(["clang", "--version"]),
                xcode=base.read_command(["xcodebuild", "-version"]),
            )
            if "M5" not in record["environment"]["chip"]:
                raise base.UnsupportedError("benchmark requires Apple M5 TensorOps")
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise ValueError("FFmpeg and FFprobe required")
            record["environment"].update(
                ffmpeg=base.read_command([ffmpeg, "-version"]).splitlines()[0],
                ffprobe=base.read_command([ffprobe, "-version"]).splitlines()[0],
            )
            build = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
            record["measurements"]["build"] = build
            if build["returncode"]:
                raise base.UnsupportedError("native source compile failed")
            source_guard(source, base.git_provenance(source))
            record["upstream"].update(
                binary_sha256=base.sha256(source / "h3"),
                library_sha256=base.sha256(source / "libh3.a"),
            )
            record["preflight_seconds"] = time.perf_counter() - preflight
            command = build_command(benchmark, args.mode, source, model, output / "original.mp4")
            record["shape"] = base.temporal_shape(5, 512, 512)
            # New prompts have different text lengths: do not transplant fox's 9914 rows.
            record["shape"]["shape_evidence"] += "; new prompt text/total packed rows untraced"
            base.write_json(output / "request.json", record | {"command": command})
            result = base.run_logged(command, source, env, output, "generation", True)
            record["measurements"]["generation"] = result
            log = (output / "generation.log").read_text(errors="replace")
            record["profile"], record["progress"] = (
                base.parse_profile(log),
                base.parse_progress(log),
            )
            if result["fallback_detected"]:
                raise base.UnsupportedError("TensorOps compilation failed; fallback rejected")
            if result["returncode"]:
                raise ValueError(f"generation exited {result['returncode']}")
            if args.mode == "hadamard-fc2":
                record["prepared_load"] = imported.validate_load_marker(log)
            if args.capture_latents:
                record["latent_validation"] = experiment.validate_latent_capture(
                    output / "latents", args.mode, benchmark["seed"]
                )
                record["observed_latent_shapes"] = {
                    k: v
                    for k, v in record["latent_validation"]["metadata"].items()
                    if k.endswith("_shape")
                }
            experiment.export_video(output, source, env, ffmpeg, ffprobe, record)
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


if __name__ == "__main__":
    result = execute(parser().parse_args())
    print(json.dumps({key: result.get(key) for key in ("status", "error", "benchmark", "mode")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)

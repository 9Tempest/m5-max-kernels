"""One-shot native fusion diagnostics with exact trajectory parity to frozen H256.

This adds Euler readbacks and synchronization, so it is not a latency benchmark.
Use run_native_session.py for warm resident end-to-end performance measurements.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import run as base
import run_fc2_experiment as experiment
import run_fused_campaign as fusion


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fusion", choices=("none", "fc2", "attention", "both"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=base.ROOT / ".cache/h3.c-fused")
    p.add_argument("--model-dir", type=Path, default=base.ROOT / ".cache/models/MiniMax-H3")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    return p


def validate_reference(directory, seed):
    record = experiment.load_json(directory / "run.json")
    if (
        record.get("status") != "completed"
        or record.get("mode") != "hadamard-fc2"
        or record.get("fusion", "none") != "none"
        or record.get("capture_latents") is not True
        or type(record.get("seed")) is not int
        or record["seed"] != seed
        or record.get("model", {}).get("revision") != base.MODEL_REVISION
        or record.get("patch", {}).get("source_sha256") != experiment.SOURCE_SHA256
    ):
        raise ValueError(
            "reference must be a completed same-seed original frozen H256 latent capture"
        )
    expected = dict(
        prompt=base.PROMPT,
        width=512,
        height=512,
        seconds=5,
        steps=20,
        active_dit_blocks=50,
        reuse=1,
        core_reuse=1,
        reference_rope=True,
    )
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("reference generation recipe mismatch")
    validated = experiment.validate_latent_capture(directory / "latents", "hadamard-fc2", seed)
    if record.get("latent_validation") != validated:
        raise ValueError("reference latent files differ from their immutable run record")
    return record, validated


def latent_parity(reference, candidate):
    expected = {entry["file"]: entry for entry in reference["tensors"]}
    actual = {entry["file"]: entry for entry in candidate["tensors"]}
    if len(expected) != 42 or set(expected) != set(actual):
        raise ValueError("trajectory parity requires all 40 Euler and two final latent tensors")
    rows = []
    for name in sorted(expected):
        left, right = expected[name], actual[name]
        if (
            left["elements"] != right["elements"]
            or left["finite"] is not True
            or right["finite"] is not True
        ):
            raise base.NumericalError("latent layout/finite contract differs")
        rows.append(
            {
                "file": name,
                "elements": left["elements"],
                "reference_sha256": left["sha256"],
                "candidate_sha256": right["sha256"],
                "byte_equal": left["sha256"] == right["sha256"],
            }
        )
    return {
        "contract": "exact bytes for every packed Euler state and final unpacked VAE input",
        "passed": all(row["byte_equal"] for row in rows),
        "tensors": rows,
    }


def execute(args):
    output, source, model = (
        args.output_dir.resolve(),
        args.source_dir.resolve(),
        args.model_dir.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(
        schema_version=1,
        status="failed",
        started_utc=base.utc_now(),
        mode="hadamard-fc2",
        fusion=args.fusion,
        seed=args.seed,
        prompt=base.PROMPT,
        benchmark="fox-winter-t2va-v1",
        width=512,
        height=512,
        seconds=5,
        steps=20,
        active_dit_blocks=50,
        reuse=1,
        core_reuse=1,
        reference_rope=True,
        diagnostic=True,
        capture_latents=True,
        capture_fc2=False,
        quality="not evaluated",
        measurements={},
        timing_contract="diagnostic extra waits/readbacks; not a performance sample",
    )
    try:
        if not 0 <= args.seed < 2**63:
            raise ValueError("seed must fit nonnegative signed 64-bit")
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            record["harness"] = base.git_provenance(base.ROOT)
            record["harness_files"] = {
                name: base.sha256(Path(__file__).with_name(name))
                for name in (
                    "run_fused_experiment.py",
                    "run_fused_campaign.py",
                    "run_fc2_experiment.py",
                    "run.py",
                    "fused.patch",
                )
            }
            record["upstream"] = base.git_provenance(source)
            record["patch"] = fusion.validate_source(source, record["upstream"])
            env, overrides = base.runtime_environment()
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise base.UnsupportedError("Metal validation unavailable on this host")
            record["environment"] = dict(
                system=platform.system(),
                release=platform.release(),
                machine=platform.machine(),
                python=platform.python_version(),
                numpy=experiment.np.__version__,
                chip=base.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                memory_bytes=int(base.read_command(["sysctl", "-n", "hw.memsize"])),
                compiler=base.read_command(["clang", "--version"]),
                xcode=base.read_command(["xcodebuild", "-version"]),
            )
            if "M5" not in record["environment"]["chip"]:
                raise base.UnsupportedError("native gate fusion requires Apple M5 TensorOps")
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise ValueError("ffmpeg and ffprobe are required")
            reference, reference_latents = validate_reference(
                args.reference_run.resolve(), args.seed
            )
            record["reference"] = {
                "run_sha256": base.sha256(args.reference_run / "run.json"),
                "source_sha256": reference["patch"]["source_sha256"],
                "seed": args.seed,
            }
            result = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
            record["measurements"]["build"] = result
            if result["returncode"]:
                raise base.UnsupportedError("native fused runtime build failed")
            fusion.validate_source(source, base.git_provenance(source))
            record["upstream"]["binary_sha256"] = base.sha256(source / "h3")
            record["model"] = base.model_inventory(
                model, Path(__file__).with_name("model-files.json")
            )
            (output / "latents").mkdir()
            overrides.update(
                H3_HADAMARD_FC2="1",
                H3_FUSE_FC2_GATE=str(int(args.fusion in ("fc2", "both"))),
                H3_FUSE_ATTN_OUT_GATE=str(int(args.fusion in ("attention", "both"))),
                H3_LATENT_CAPTURE_DIR=str(output / "latents"),
                H3_CAPTURE_SEED=str(args.seed),
                H3_GPU_SAMPLER="1",
                H3_GPU_SAMPLER_WINDOW="1",
            )
            env.update(overrides)
            record["runtime_overrides"] = {
                k: base.path_label(v) if k.endswith("_DIR") else v for k, v in overrides.items()
            }
            recipe = SimpleNamespace(
                mode="upstream-int8", width=512, height=512, seconds=5.0, steps=20, seed=args.seed
            )
            command = base.build_command(recipe, source, model, output / "original.mp4")
            base.write_json(output / "request.json", record | {"command": command})
            result = base.run_logged(command, source, env, output, "generation", True)
            record["measurements"]["generation"] = result
            log = (output / "generation.log").read_text(errors="replace")
            if result["fallback_detected"]:
                raise base.UnsupportedError("TensorOps fallback rejected")
            if result["returncode"]:
                raise ValueError(f"generation exited {result['returncode']}")
            if "h3 precision: FC2 plain normalized H256" not in log:
                raise base.UnsupportedError("H256 precision marker missing")
            for enabled, marker in (
                (args.fusion in ("fc2", "both"), "h3 fusion: FC2 GEMM gate/residual epilogue"),
                (
                    args.fusion in ("attention", "both"),
                    "h3 fusion: attention output GEMM gate/residual epilogue",
                ),
            ):
                if (marker in log) != enabled:
                    raise base.UnsupportedError(
                        "runtime fusion marker disagrees with requested flags"
                    )
            record["profile"] = base.parse_profile(log)
            record["latent_validation"] = experiment.validate_latent_capture(
                output / "latents", "hadamard-fc2", args.seed
            )
            record["latent_parity"] = latent_parity(reference_latents, record["latent_validation"])
            experiment.export_video(output, source, env, ffmpeg, ffprobe, record)
            record["video_bytes_equal_reference"] = (
                record["exact_video"]["sha256"] == reference["exact_video"]["sha256"]
            )
            fusion.validate_source(source, base.git_provenance(source))
            if not record["latent_parity"]["passed"]:
                raise base.NumericalError("fusion changed at least one latent trajectory tensor")
        record["status"] = "completed"
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupted by user")
    except Exception as error:
        record.update(status=fusion.classify(error), error=f"{type(error).__name__}: {error}")
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
            {"status": record["status"], "record": base.path_label(args.output_dir / "run.json")}
        )
    )
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

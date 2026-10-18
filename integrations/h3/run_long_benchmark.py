"""Run the frozen 15-second first-frame benchmark with BF16 or prepared H256.

No mutable prompt override and no checkpoint fallback. Existing runners remain
unchanged. The common GPU lock covers preflight/build/generation/media checks.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace

import derived_checkpoint as derived
import run as base
import run_fc2_experiment as experiment
import run_prepared_checkpoint as imported


DEFAULT = Path(__file__).with_name("long_benchmark.json")
CATALOG_SHA256 = "ee6fb7632e4b65858c9f0ec2944f5ae2a95eb85ae0e3219e308fb6fd1295e21a"


def load_benchmark(benchmark_id="deepseek-whale-maid-15s-v1", path=DEFAULT):
    if base.sha256(path) != CATALOG_SHA256:
        raise ValueError("long benchmark catalog differs from the frozen hash")
    catalog = experiment.load_json(path)
    matches = [entry for entry in catalog["benchmarks"] if entry["id"] == benchmark_id]
    if catalog.get("schema_version") != 1 or len(matches) != 1:
        raise ValueError("benchmark ID must identify exactly one frozen catalog entry")
    entry = matches[0]
    prompt_path = DEFAULT.parent / entry["prompt_file"]
    payload = prompt_path.read_bytes()
    if base.sha256(prompt_path) != entry["prompt_sha256"] or len(payload) != entry["prompt_bytes"]:
        raise ValueError("full UTF8 prompt differs from the frozen hash/bytes")
    prompt = payload.decode("utf-8")
    if "\0" in prompt:
        raise ValueError("embedded NUL in prompt would truncate the native argument")
    return (
        catalog["recipe"]
        | entry
        | dict(prompt=prompt, catalog_sha256=CATALOG_SHA256, catalog_suite=catalog["suite"])
    )


def snapshot_inputs(benchmark, image, output):
    payload = image.read_bytes()
    if len(payload) != benchmark["image_bytes"] or base.sha256(image) != benchmark["image_sha256"]:
        raise ValueError("first-frame input differs from the frozen image hash/bytes")
    with (output / "conditioning-input.jpg").open("xb") as saved:
        saved.write(payload)
    if base.sha256(output / "conditioning-input.jpg") != benchmark["image_sha256"]:
        raise ValueError("first-frame input changed while snapshotting")
    with (output / "prompt.txt").open("xb") as saved:
        saved.write(benchmark["prompt"].encode("utf-8"))
    conditioning = dict(
        mode="first-frame",
        type="first_frame",
        input_file="conditioning-input.jpg",
        sha256=benchmark["image_sha256"],
        image_sha256=benchmark["image_sha256"],
        bytes=benchmark["image_bytes"],
        original_width=benchmark["image_width"],
        original_height=benchmark["image_height"],
        width=benchmark["width"],
        height=benchmark["height"],
        preprocessing="upstream_first_frame_stretch_to_render_canvas",
        target_canvas=[benchmark["width"], benchmark["height"]],
    )
    base.write_json(output / "inputcondition.json", conditioning)
    return conditioning


def expected_shape(benchmark):
    shape = base.temporal_shape(benchmark["seconds"], benchmark["width"], benchmark["height"])
    if (
        shape["aligned_frames"] != benchmark["source_frames"]
        or shape["requested_frames"] != benchmark["exact_frames"]
    ):
        raise ValueError("upstream temporal alignment differs from frozen recipe")
    shape["condition_video_rows"] = (benchmark["width"] // 32) * (benchmark["height"] // 32)
    return shape


def conditioning_headers(model):
    """Header-only coverage of every native first-frame vision/encoder tensor."""
    root = model / "FL2VA/text_encoder"
    index = experiment.load_json(root / "model.safetensors.index.json")["weight_map"]
    vision = {
        "model.visual.patch_embed.proj.weight": [1152, 3, 2, 16, 16],
        "model.visual.patch_embed.proj.bias": [1152],
        "model.visual.pos_embed.weight": [2304, 1152],
    }
    for block in range(27):
        prefix = f"model.visual.blocks.{block}."
        for layer, shape in (
            ("norm1", [1152]),
            ("norm2", [1152]),
            ("attn.qkv", [3456, 1152]),
            ("attn.proj", [1152, 1152]),
            ("mlp.linear_fc1", [4304, 1152]),
            ("mlp.linear_fc2", [1152, 4304]),
        ):
            vision[prefix + layer + ".weight"] = shape
            vision[prefix + layer + ".bias"] = [shape[0]]
    for prefix, width in [("model.visual.merger.", 1152)] + [
        (f"model.visual.deepstack_merger_list.{i}.", 4608) for i in range(3)
    ]:
        for layer, shape in (
            ("norm", [width]),
            ("linear_fc1", [4608, 4608]),
            ("linear_fc2", [5120, 4608]),
        ):
            vision[prefix + layer + ".weight"] = shape
            vision[prefix + layer + ".bias"] = [shape[0]]
    if not vision.keys() <= index.keys():
        raise ValueError("first-frame Qwen vision tensors missing from index")
    headers = {}
    for name in sorted({index[key] for key in vision}):
        with derived.contained_file(root, name).open("rb") as handle:
            headers[name] = derived.read_header(handle)[0]
    for key, shape in vision.items():
        entry = headers[index[key]].get(key, {})
        if entry.get("shape") != shape or entry.get("dtype") != "BF16":
            raise ValueError(f"first-frame vision tensor shape/dtype mismatch: {key}")
    vae = {}

    def norm(prefix, channels):
        vae[prefix + ".weight"] = [channels]
        vae[prefix + ".bias"] = [channels]

    def conv(prefix, inputs, outputs, kernel):
        vae[prefix + ".weight"] = [outputs, inputs, kernel, kernel, kernel]
        vae[prefix + ".bias"] = [outputs]

    conv("encoder.conv_in", 3, 128, 3)
    previous = 128
    for level, channels in enumerate((128, 256, 256, 512, 512, 1024)):
        for block in range(2):
            inputs = channels if block else previous
            prefix = f"encoder.down.{level}.block.{block}."
            norm(prefix + "norm1", inputs)
            conv(prefix + "conv1", inputs, channels, 3)
            norm(prefix + "norm2", channels)
            conv(prefix + "conv2", channels, channels, 3)
            if inputs != channels:
                conv(prefix + "nin_shortcut", inputs, channels, 1)
        if level < 4:
            conv(f"encoder.down.{level}.downsample.conv", channels, channels, 3)
        previous = channels
    norm("encoder.norm_out", 1024)
    conv("encoder.conv_out", 1024, 48, 3)
    conv("quant_conv", 48, 48, 1)
    with (model / "FL2VA/video_vae/source/model.safetensors").open("rb") as handle:
        header = derived.read_header(handle)[0]
    for key, shape in vae.items():
        entry = header.get(key, {})
        if entry.get("shape") != shape or entry.get("dtype") != "F32":
            raise ValueError(f"first-frame VAE encoder tensor shape/dtype mismatch: {key}")
    return dict(
        vision_tensor_count=len(vision),
        vision_dtype="BF16",
        vision_shards=sorted(headers),
        vae_encoder_tensor_count=len(vae),
        vae_encoder_storage="F32",
        header_only=True,
        coverage="all tensors loaded by pinned h3_vision_encoder.c and h3_video_encoder.c",
    )


def probe_tokenizer(benchmark, source, model, output, env):
    driver = Path(__file__).with_name("tokenizer_probe.c").resolve()
    binary = output / "tokenizer-probe"
    command = [
        "clang",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(source),
        str(driver),
        str(source / "h3_tokenizer.o"),
        "-framework",
        "Foundation",
        "-licucore",
        "-o",
        str(binary),
    ]
    build = base.run_logged(command, source, env, output, "tokenizer-build")
    if build["returncode"]:
        raise base.UnsupportedError("CPU native tokenizer probe compile failed")
    tokenizer = model / "FL2VA/tokenizer/tokenizer.json"
    result = base.run_logged(
        [str(binary), str(tokenizer), str(output / "prompt.txt")], source, env, output, "tokenizer"
    )
    if result["returncode"]:
        raise ValueError("native tokenizer failed or full prompt roundtrip changed")
    tokenization = experiment.load_json(output / "tokenizer.log")
    for key in ("prompt_bytes", "prompt_tokens", "image_prefix_tokens"):
        if type(tokenization.get(key)) is not int or tokenization[key] != benchmark[key]:
            raise ValueError("native tokenizer count changed; prompt truncation is forbidden")
    if tokenization.get("roundtrip_exact") is not True:
        raise ValueError("native tokenizer did not roundtrip the entire prompt")
    vision_tokens = (benchmark["height"] // 32) * (benchmark["width"] // 32)
    total = tokenization["prompt_tokens"] + tokenization["image_prefix_tokens"] + vision_tokens + 2
    config = experiment.load_json(model / "FL2VA/tokenizer/tokenizer_config.json")
    native_limit = (2**32 - 1) // 25600
    if total > min(config["model_max_length"], native_limit):
        raise ValueError(
            "complete image/text presentation exceeds tokenizer/native limits; refusing truncation"
        )
    return tokenization | dict(
        presentation_tokens=total,
        vision_tokens=vision_tokens,
        vision_boundary_tokens=2,
        native_overflow_limit=native_limit,
        configured_model_max_length=config["model_max_length"],
        tokenizer_sha256=base.sha256(tokenizer),
        config_sha256=base.sha256(model / "FL2VA/tokenizer/tokenizer_config.json"),
        binary_sha256=base.sha256(binary),
        driver_sha256=base.sha256(driver),
        prompt_sha256=benchmark["prompt_sha256"],
        truncated=False,
        build=build,
        process=result,
    )


def validate_latent_capture(directory, mode, benchmark):
    shape = expected_shape(benchmark)
    expected = {
        f"euler-step{s:02d}-{kind}.f32"
        for s in range(benchmark["steps"])
        for kind in ("video", "audio")
    }
    expected |= {"final-video.f32", "final-audio.f32", "latent-metadata.json"}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError("latent capture needs all 20 Euler states and final unpacked video/audio")
    meta = experiment.load_json(directory / "latent-metadata.json")
    wanted = dict(
        schema=1,
        scope="diagnostic_CPU_readback_after_completed_GPU_Euler",
        dtype="float32_little_endian",
        seed=benchmark["seed"],
        completed_steps=benchmark["steps"],
        final_video_shape=[
            24,
            shape["video_latent_t"],
            benchmark["height"] // 16,
            benchmark["width"] // 16,
        ],
        final_audio_shape=[32, 2, shape["audio_latent_t"]],
        per_step_video_shape=[shape["video_tokens"], 96],
        per_step_audio_shape=[shape["stereo_audio_tokens"], 32],
        int8_mlp=int(mode != "bf16"),
        int8_qkv=int(mode != "bf16"),
        int8_attention_out=int(mode != "bf16"),
        hadamard_fc2=int(mode == "hadamard-fc2"),
        finite=True,
        final_video_sigma=0,
        final_audio_sigma=0,
    )
    if any(meta.get(k) != v or type(meta.get(k)) is not type(v) for k, v in wanted.items()):
        raise ValueError("latent metadata does not match the 15-second recipe/precision")
    if (
        meta.get("final_layout") != "unpacked_contiguous_VAE_input_before_decoder"
        or meta.get("per_step_layout") != "packed_model_rows_after_Euler_before_unpack"
    ):
        raise ValueError("unknown captured latent layout")
    records = []
    for name in sorted(expected - {"latent-metadata.json"}):
        count = math.prod(wanted["final_video_shape" if "video" in name else "final_audio_shape"])
        records.append(experiment.finite_file(directory / name, "<f4", count))
    return dict(
        metadata=meta,
        metadata_sha256=base.sha256(directory / "latent-metadata.json"),
        tensors=records,
    )


def export_video(output, source, env, ffmpeg, ffprobe, record):
    original, exact = output / "original.mp4", output / "video-15s.mp4"
    for name, video, frames in (("original", original, 362), ("exact", exact, 360)):
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
                "0:a:0",
                "-frames:v",
                "360",
                "-t",
                "15",
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
                raise ValueError("exact fifteen-second video export failed")
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
        probe = experiment.load_json(output / (name + "-probe.log"))
        audio = [
            stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"
        ]
        if (
            len(audio) != 1
            or audio[0].get("channels") != 2
            or audio[0].get("sample_rate") != "32000"
            or audio[0].get("channel_layout") != "stereo"
        ):
            raise ValueError("expected one 32000 Hz stereo generated audio stream")
        duration = float(audio[0].get("duration", "nan"))
        if not math.isfinite(duration) or abs(duration - frames / 24) > 0.05:
            raise ValueError("generated audio duration differs from the video")
        record[name + "_video"] = base.validate_video(probe, 512, 288, frames) | dict(
            file=video.name,
            sha256=base.sha256(video),
            audio_channels=2,
            audio_duration_seconds=duration,
        )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-id", default="deepseek-whale-maid-15s-v1")
    p.add_argument("--catalog", type=Path, default=DEFAULT)
    p.add_argument(
        "--input-image",
        type=Path,
        default=base.ROOT / ".cache/h3-inputs/deepseek-whale-maid-v1.jpg",
    )
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
    benchmark = load_benchmark(args.benchmark_id, args.catalog)
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


def build_command(benchmark, mode, source, model, original, first_frame):
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
    command.extend(["--first-frame", os.path.relpath(first_frame, source)])
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
                    "source_frames",
                    "exact_frames",
                )
            }
        )
        record.update(benchmark=benchmark["id"], catalog=benchmark)
        record["conditioning"] = snapshot_inputs(benchmark, args.input_image, output)
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
                    "run_long_benchmark.py",
                    "long_benchmark.json",
                    "tokenizer_probe.c",
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
            record["conditioning_components"] = conditioning_headers(model)
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
            record["tokenization"] = probe_tokenizer(benchmark, source, model, output, env)
            record["preflight_seconds"] = time.perf_counter() - preflight
            command = build_command(
                benchmark,
                args.mode,
                source,
                model,
                output / "original.mp4",
                output / record["conditioning"]["input_file"],
            )
            record["shape"] = expected_shape(benchmark)
            record["shape"]["text_tokens"] = record["tokenization"]["presentation_tokens"]
            record["shape"]["packed_sequence_tokens"] = (
                record["shape"]["video_tokens"]
                + record["shape"]["stereo_audio_tokens"]
                + record["shape"]["condition_video_rows"]
                + record["shape"]["text_tokens"]
            )
            record["shape"]["shape_evidence"] += (
                "; full native CPU tokenizer roundtrip and pinned first-frame layout; total rows source-derived, not runtime-traced"
            )
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
                record["latent_validation"] = validate_latent_capture(
                    output / "latents", args.mode, benchmark
                )
                record["observed_latent_shapes"] = {
                    k: v
                    for k, v in record["latent_validation"]["metadata"].items()
                    if k.endswith("_shape")
                }
            export_video(output, source, env, ffmpeg, ffprobe, record)
            if (
                base.sha256(output / "conditioning-input.jpg") != benchmark["image_sha256"]
                or base.sha256(output / "prompt.txt") != benchmark["prompt_sha256"]
            ):
                raise ValueError("run-local conditioning input changed during generation")
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

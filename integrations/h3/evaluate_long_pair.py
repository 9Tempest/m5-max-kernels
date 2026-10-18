"""Strict CPU evaluation for the pinned 15-second first-frame-conditioned pair."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluate as metrics
import evaluate_benchmark_pair as pair
import evaluate_latents as capture


IMAGE_SHA256 = "b1b4c9330f8e27fd95c770e610210b366ad30933a651f71d1e39401bf63cf24d"
IMAGE_BYTES = 208682
PROMPT_SHA256 = "fccc21a2e05295b2920af3b6a3a58e720d7d8f5f15f2e057986faecd4f95f0cc"
LONG_CATALOG_SHA256 = "ee6fb7632e4b65858c9f0ec2944f5ae2a95eb85ae0e3219e308fb6fd1295e21a"
CATALOG_FILE = Path(__file__).with_name("long_benchmark.json")
RECIPE = dict(
    width=512,
    height=288,
    seconds=15,
    fps=24,
    seed=83,
    steps=20,
    source_frames=362,
    exact_frames=360,
    active_dit_blocks=50,
    reuse=1,
    core_reuse=1,
    reference_rope=True,
)


def latent_shapes(recipe):
    width, height, frames, fps = (recipe[k] for k in ("width", "height", "source_frames", "fps"))
    if width % 32 or height % 32 or frames < 5 or (frames - 5) % 17 or fps != 24:
        raise ValueError("unsupported native temporal/spatial contract")
    video_t = (frames - 5) // 17 * 5 + 2
    audio_t = math.floor(frames * 40 / fps + 0.5)
    return dict(
        final_video_shape=[24, video_t, height // 16, width // 16],
        final_audio_shape=[32, 2, audio_t],
        per_step_video_shape=[video_t * (height // 32) * (width // 32), 96],
        per_step_audio_shape=[audio_t * 2, 32],
    )


def spec(shapes):
    return [
        (
            f"euler-step{step:02d}-{kind}.f32",
            f"euler_step{step:02d}_{kind}",
            shapes[f"per_step_{kind}_shape"],
        )
        for step in range(20)
        for kind in ("video", "audio")
    ] + [
        (f"final-{kind}.f32", f"final_{kind}", shapes[f"final_{kind}_shape"])
        for kind in ("video", "audio")
    ]


def load_catalog():
    """Bind the entire saved catalog, prompt bytes and reference image identity."""
    if metrics.sha256(CATALOG_FILE) != LONG_CATALOG_SHA256:
        raise ValueError("long benchmark catalog bytes differ from the approved hash")
    catalog = capture.read_json(CATALOG_FILE)
    if catalog.get("schema_version") != 1 or len(catalog.get("benchmarks", [])) != 1:
        raise ValueError("invalid long benchmark catalog schema")
    entry = catalog["benchmarks"][0]
    prompt_path = CATALOG_FILE.parent / entry["prompt_file"]
    if (
        entry.get("prompt_sha256") != PROMPT_SHA256
        or metrics.sha256(prompt_path) != PROMPT_SHA256
        or prompt_path.stat().st_size != entry.get("prompt_bytes")
        or entry.get("image_sha256") != IMAGE_SHA256
        or entry.get("image_bytes") != IMAGE_BYTES
    ):
        raise ValueError("catalog prompt/image identity differs")
    result = dict(
        **catalog["recipe"],
        **entry,
        prompt=prompt_path.read_text(),
        catalog_sha256=LONG_CATALOG_SHA256,
        catalog_suite=catalog["suite"],
    )
    if any(result.get(k) != v or type(result.get(k)) is not type(v) for k, v in RECIPE.items()):
        raise ValueError("catalog recipe differs from approved long contract")
    return result


def validate_recipe(reference_dir, candidate_dir):
    runs = [
        capture.read_json(directory / "run.json") for directory in (reference_dir, candidate_dir)
    ]
    catalog = load_catalog()
    identities = []
    for directory, run in zip((reference_dir, candidate_dir), runs, strict=True):
        if run.get("catalog") != catalog or run.get("benchmark") != catalog["id"]:
            raise ValueError("long benchmark catalog entry differs")
        if any(run.get(k) != v or type(run.get(k)) is not type(v) for k, v in RECIPE.items()):
            raise ValueError("long recipe/seed/geometry/frame alignment differs")
        prompt_path = directory / "prompt.txt"
        if (
            metrics.sha256(prompt_path) != PROMPT_SHA256
            or run.get("prompt") != prompt_path.read_text()
        ):
            raise ValueError("full Chinese prompt bytes/text differ")
        conditioning = run.get("conditioning", {})
        if (
            conditioning.get("mode") != "first-frame"
            or conditioning.get("input_file") != "conditioning-input.jpg"
        ):
            raise ValueError("exact first-frame conditioning input is required")
        image_path = directory / conditioning["input_file"]
        if (
            conditioning.get("sha256"),
            conditioning.get("bytes"),
            metrics.sha256(image_path),
            image_path.stat().st_size,
        ) != (IMAGE_SHA256, IMAGE_BYTES, IMAGE_SHA256, IMAGE_BYTES):
            raise ValueError("first-frame image bytes/digest changed")
        identity = {
            key: conditioning.get(key)
            for key in (
                "mode",
                "sha256",
                "bytes",
                "original_width",
                "original_height",
                "preprocessing",
                "target_canvas",
            )
        }
        if identity != dict(
            mode="first-frame",
            sha256=IMAGE_SHA256,
            bytes=IMAGE_BYTES,
            original_width=1280,
            original_height=720,
            preprocessing="upstream_first_frame_stretch_to_render_canvas",
            target_canvas=[512, 288],
        ):
            raise ValueError("first-frame preprocessing/canvas identity differs")
        identities.append(identity)
        if run.get("status") != "completed" or run.get("capture_latents") is not True:
            raise ValueError("completed long latent captures are required")
        for key, frames in (("original_video", 362), ("exact_video", 360)):
            media = run.get(key, {})
            if (media.get("width"), media.get("height"), media.get("frames"), media.get("fps")) != (
                512,
                288,
                frames,
                24,
            ):
                raise ValueError("reported long video structure differs")
            if not math.isclose(media.get("duration_seconds", -1), frames / 24, abs_tol=1e-5):
                raise ValueError("reported long video duration differs")
    return runs, identities[0], latent_shapes(RECIPE)


def bf16_capture(directory, run, shapes):
    if run.get("mode") != "bf16" or run.get("prepared_checkpoint") is not False:
        raise ValueError("reference must use original BF16")
    if run.get("patch") != dict(
        source_sha256=capture.SOURCE_SHA256, patch_sha256=capture.PATCH_SHA256
    ):
        raise ValueError("reference must use the approved BF16 observation source")
    process = run.get("measurements", {}).get("generation", {})
    if process.get("returncode") != 0 or process.get("fallback_detected") is not False:
        raise ValueError("BF16 generation failed or fell back")
    model = {
        key: run.get("model", {}).get(key) for key in ("repository", "revision", "manifest_sha256")
    }
    if not all(isinstance(value, str) and value for value in model.values()):
        raise ValueError("original model identity missing")
    validation = run.get("latent_validation", {})
    path = directory / "latents/latent-metadata.json"
    meta = capture.read_json(path)
    if meta != validation.get("metadata") or metrics.sha256(path) != validation.get(
        "metadata_sha256"
    ):
        raise ValueError("BF16 latent metadata changed")
    wanted = dict(
        schema=1,
        dtype="float32_little_endian",
        seed=83,
        completed_steps=20,
        scope="diagnostic_CPU_readback_after_completed_GPU_Euler",
        final_layout="unpacked_contiguous_VAE_input_before_decoder",
        per_step_layout="packed_model_rows_after_Euler_before_unpack",
        final_video_sigma=0,
        final_audio_sigma=0,
        int8_mlp=0,
        int8_qkv=0,
        int8_attention_out=0,
        hadamard_fc2=0,
        finite=True,
        **shapes,
    )
    if any(meta.get(k) != v or type(meta.get(k)) is not type(v) for k, v in wanted.items()):
        raise ValueError("BF16 latent dynamic shape/boundary/seed/precision differs")
    specs = spec(shapes)
    expected = {name for name, _, _ in specs}
    entries = validation.get("tensors", [])
    indexed = {entry["file"]: entry for entry in entries}
    if (
        {p.name for p in (directory / "latents").iterdir()} != expected | {"latent-metadata.json"}
        or set(indexed) != expected
        or len(entries) != len(expected)
    ):
        raise ValueError("BF16 needs exactly 20 steps and final video/audio")
    arrays, evidence = {}, []
    for name, key, shape in specs:
        path, count = directory / "latents" / name, math.prod(shape)
        digest = metrics.sha256(path)
        if (
            indexed[name] != dict(file=name, elements=count, sha256=digest, finite=True)
            or path.stat().st_size != count * 4
        ):
            raise ValueError(f"BF16 tensor digest/size differs: {name}")
        value = np.fromfile(path, dtype="<f4").reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite BF16 latent: {name}")
        arrays[key] = value
        evidence.append(dict(file=name, shape=shape, sha256=digest, finite=True))
    return arrays, dict(
        run_sha256=metrics.sha256(directory / "run.json"),
        model=model,
        patch=run["patch"],
        metadata=meta,
        metadata_sha256=validation["metadata_sha256"],
        tensors=evidence,
    )


def execute(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema_version=1,
        status="failed",
        quality_decision="not_evaluated",
        evaluator_sha256=metrics.sha256(Path(__file__)),
        metric_sha256=metrics.sha256(Path(metrics.__file__)),
        prepared_validator_sha256=metrics.sha256(Path(pair.__file__)),
        latent_helper_sha256=metrics.sha256(Path(capture.__file__)),
        scope="paired 15s first-frame-conditioned captured latents",
        data_range=1.0,
        data_range_note="latent PSNR convention only; emphasize SNR and relative L2",
    )
    try:
        input_run_hashes = {
            label: metrics.sha256(directory / "run.json")
            for label, directory in (("bf16", args.bf16), ("hadamard", args.hadamard))
        }
        runs, conditioning, shapes = validate_recipe(args.bf16, args.hadamard)
        record.update(
            recipe=RECIPE, conditioning=conditioning, shapes=shapes, catalog=runs[0]["catalog"]
        )
        a, original = bf16_capture(args.bf16, runs[0], shapes)
        b, prepared = pair.prepared_capture(args.hadamard, original["model"], shapes=shapes)
        record["sources"] = dict(bf16=original, hadamard=prepared)
        for label, arrays in (("bf16", a), ("hadamard", b)):
            with (out / f"{label}.npz").open("xb") as stream:
                np.savez(stream, **arrays)
            record["sources"][label]["npz_sha256"] = metrics.sha256(out / f"{label}.npz")
        record["latents"] = capture.compare_arrays(a, b)
        if any(value["status"] != "completed" for value in record["latents"].values()):
            raise ValueError("long latent numerical comparison failed")
        if args.media:
            videos = []
            for directory, run in zip((args.bf16, args.hadamard), runs, strict=True):
                name = run["exact_video"]["file"]
                if Path(name).name != name:
                    raise ValueError("exact video must be local to the run")
                path = directory / name
                if metrics.sha256(path) != run["exact_video"]["sha256"]:
                    raise ValueError("exact video digest differs from capture run")
                videos.append(path)
            media = metrics.execute(
                SimpleNamespace(
                    reference=videos[0],
                    candidate=videos[1],
                    kind="media",
                    data_range=None,
                    audio_data_range=2.0,
                    ssim=True,
                    lpips=False,
                    vbench=False,
                    output_dir=out / "media",
                )
            )
            if media["status"] != "completed":
                raise ValueError("strict long media evaluation failed")
            for label, path in zip(("reference", "candidate"), videos, strict=True):
                evidence = media["files"][label]
                contract = evidence["contract"]
                expected = dict(
                    width=512,
                    height=288,
                    frames=360,
                    fps="24",
                    duration_seconds="15",
                    audio_rate=32000,
                    audio_channels=2,
                )
                if evidence["sha256"] != metrics.sha256(path) or any(
                    contract.get(k) != value for k, value in expected.items()
                ):
                    raise ValueError("decoded media differs from the approved 15s recipe")
            record["media"] = dict(
                evaluation_sha256=metrics.sha256(out / "media/evaluation.json"),
                video_rgb=media["video_rgb"],
                audio_pcm32k_stereo=media["audio_pcm32k_stereo"],
                ssim=media["optional"]["ssim"],
            )
        for label, directory in (("bf16", args.bf16), ("hadamard", args.hadamard)):
            if not (
                metrics.sha256(directory / "run.json")
                == record["sources"][label]["run_sha256"]
                == input_run_hashes[label]
            ):
                raise ValueError("input run changed during evaluation")
        record["status"] = "completed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    metrics.write_json(out / "long-evaluation.json", record)
    return record


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--bf16", type=Path, required=True)
    cli.add_argument("--hadamard", type=Path, required=True)
    cli.add_argument("--media", action="store_true")
    cli.add_argument("--output-dir", type=Path, required=True)
    result = execute(cli.parse_args())
    print(json.dumps({key: result.get(key) for key in ("status", "error")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)

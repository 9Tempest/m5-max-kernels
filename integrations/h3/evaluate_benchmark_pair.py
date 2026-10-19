"""CPU paired latent/media evaluation across approved BF16 and prepared-H256 loaders."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import benchmark_catalog as catalog
import evaluate as metrics
import evaluate_latents as capture


PREPARED_SOURCE_SHA256 = "054ec73eb454e44dc75d71bf83fa53a7129e5bd1e30fb304236f545f92f9cd7e"
PREPARED_PATCH_SHA256 = "1cad08a5bddc2dfa37c29d2c00551fe467a032d358d1287072f93bfd9dc74c67"
POLICY_ID = "h3_dit_int8_native_fc2_h256_v1"


def prepared_capture(directory, original_identity, *, shapes=None):
    # Default remains the frozen five-second contract. The separate long-pair
    # evaluator supplies shapes only after validating its independent recipe.
    shapes = capture.SHAPES if shapes is None else shapes
    run = capture.read_json(directory / "run.json")
    if (
        run.get("status"),
        run.get("mode"),
        run.get("prepared_checkpoint"),
        run.get("capture_latents"),
    ) != ("completed", "hadamard-fc2", True, True):
        raise ValueError("candidate must be a completed prepared-H256 latent capture")
    if run.get("patch") != dict(
        source_sha256=PREPARED_SOURCE_SHA256, patch_sha256=PREPARED_PATCH_SHA256
    ):
        raise ValueError("candidate must use the approved prepared loader source/patch")
    declaration = run.get("derived_checkpoint", {})
    if declaration.get("manifest_file") != "derived-checkpoint.json":
        raise ValueError("derived manifest snapshot is required")
    manifest_path = directory / "derived-checkpoint.json"
    digest = metrics.sha256(manifest_path)
    model = run.get("model", {})
    if digest != declaration.get("manifest_sha256") or digest != model.get("manifest_sha256"):
        raise ValueError("derived manifest SHA differs from verified generation record")
    manifest = capture.read_json(manifest_path)
    wanted_source = {
        "repository": original_identity["repository"],
        "revision": original_identity["revision"],
        "model_manifest_sha256": original_identity["manifest_sha256"],
    }
    if any(
        value != wanted_source
        for value in (
            manifest.get("source_model"),
            declaration.get("source_model"),
            model.get("source_model"),
        )
    ):
        raise ValueError("prepared checkpoint does not derive from the paired original model")
    if (
        manifest.get("schema") != "h3_prepared_model_v1"
        or manifest.get("original_checkpoint_retained") is not True
    ):
        raise ValueError("invalid derived model preservation/schema contract")
    if manifest.get("policy_id") != POLICY_ID or declaration.get("policy_id") != POLICY_ID:
        raise ValueError("unknown prepared checkpoint precision policy")
    producer = manifest.get("producer", {})
    if (producer.get("source_sha256"), producer.get("patch_sha256")) != (
        capture.SOURCE_SHA256,
        capture.PATCH_SHA256,
    ):
        raise ValueError("prepared weights have an unapproved producer")
    if declaration.get("producer") != producer:
        raise ValueError("derived producer snapshot differs")
    prepared_digest = manifest.get("prepared_manifest_sha256")
    if (
        not isinstance(prepared_digest, str)
        or len(prepared_digest) != 64
        or any(
            value != prepared_digest
            for value in (
                declaration.get("prepared_manifest_sha256"),
                model.get("prepared_manifest_sha256"),
                run.get("runtime_overrides", {}).get("H3_PREPARED_DIT_MANIFEST_SHA256"),
            )
        )
    ):
        raise ValueError("prepared payload manifest binding differs")
    load = run.get("prepared_load", {})
    if load.get("projections") != 200 or load.get("weight_rotation_quantization_seconds") != 0:
        raise ValueError(
            "candidate did not verify the 200-projection import without requantization"
        )
    process = run.get("measurements", {}).get("generation", {})
    if process.get("returncode") != 0 or process.get("fallback_detected") is not False:
        raise ValueError("candidate generation failed or used a fallback")
    validation = run.get("latent_validation", {})
    meta_path = directory / "latents/latent-metadata.json"
    meta = capture.read_json(meta_path)
    if meta != validation.get("metadata") or metrics.sha256(meta_path) != validation.get(
        "metadata_sha256"
    ):
        raise ValueError("candidate latent metadata changed")
    wanted = dict(
        schema=1,
        dtype="float32_little_endian",
        seed=run["seed"],
        completed_steps=20,
        scope="diagnostic_CPU_readback_after_completed_GPU_Euler",
        final_layout="unpacked_contiguous_VAE_input_before_decoder",
        per_step_layout="packed_model_rows_after_Euler_before_unpack",
        final_video_sigma=0,
        final_audio_sigma=0,
        int8_mlp=1,
        int8_qkv=1,
        int8_attention_out=1,
        hadamard_fc2=1,
        finite=True,
        **shapes,
    )
    if any(meta.get(k) != v or type(meta.get(k)) is not type(v) for k, v in wanted.items()):
        raise ValueError("candidate latent metadata boundary/precision/seed/shape differs")
    specs = [
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
    expected = {name for name, _, _ in specs}
    files = {p.name for p in (directory / "latents").iterdir()}
    entries = validation.get("tensors", [])
    indexed = {item["file"]: item for item in entries}
    if (
        files != expected | {"latent-metadata.json"}
        or set(indexed) != expected
        or len(entries) != len(expected)
    ):
        raise ValueError("candidate requires exactly 20 steps and final video/audio")
    arrays, evidence = {}, []
    for name, key, shape in specs:
        path, count = directory / "latents" / name, math.prod(shape)
        digest = metrics.sha256(path)
        if (
            indexed[name] != dict(file=name, elements=count, sha256=digest, finite=True)
            or path.stat().st_size != count * 4
        ):
            raise ValueError(f"candidate tensor digest/size changed: {name}")
        value = np.fromfile(path, dtype="<f4").reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite candidate latent: {name}")
        arrays[key] = value
        evidence.append(dict(file=name, key=key, shape=shape, sha256=digest, finite=True))
    return arrays, dict(
        run_sha256=metrics.sha256(directory / "run.json"),
        model=wanted_source,
        loader_patch=run["patch"],
        derived_manifest_sha256=metrics.sha256(manifest_path),
        prepared_manifest_sha256=prepared_digest,
        metadata_sha256=metrics.sha256(meta_path),
        metadata=meta,
        tensors=evidence,
    )


def validate_pair_recipe(reference, candidate):
    entry = catalog.load_benchmark(reference.get("benchmark"))
    wanted = {
        key: entry[key]
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
    for run in (reference, candidate):
        if run.get("benchmark") != entry["id"] or run.get("catalog") != entry:
            raise ValueError("benchmark catalog entry/hash differs")
        if any(
            run.get(key) != value or type(run.get(key)) is not type(value)
            for key, value in wanted.items()
        ):
            raise ValueError("paired prompt/seed/geometry/recipe differs")
    return entry


def recorded_video(directory, run):
    """Bind the media to the generation record, before decoding or joining it."""
    video = run.get("exact_video", {})
    expected = dict(file="video-5s.mp4", width=512, height=512, frames=120, fps=24)
    if (
        any(video.get(key) != value for key, value in expected.items())
        or video.get("duration_seconds") != 5
    ):
        raise ValueError("recorded exact video differs from the fixed five-second recipe")
    path = directory / expected["file"]
    if metrics.sha256(path) != video.get("sha256"):
        raise ValueError("exact video digest differs from capture run")
    return path


def validate_media_join(media, videos, runs):
    if (
        media.get("status") != "completed"
        or media.get("kind") != "media"
        or media.get("harness_sha256") != metrics.sha256(Path(metrics.__file__))
    ):
        raise ValueError("media join requires a completed current strict evaluator record")
    expected = dict(
        width=512,
        height=512,
        frames=120,
        fps="24",
        duration_seconds="5",
        audio_rate=32000,
        audio_channels=2,
        decoded_audio_samples=160000,
    )
    contracts = []
    for key, path, run in zip(("reference", "candidate"), videos, runs, strict=True):
        evidence = media["files"][key]
        contract = evidence["contract"]
        if not (evidence.get("sha256") == metrics.sha256(path) == run["exact_video"]["sha256"]):
            raise ValueError("media evaluation is not for the recorded paired videos")
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError("decoded media differs from the fixed five-second recipe")
        contracts.append(contract)
    if contracts[0] != contracts[1]:
        raise ValueError("joined media timelines/audio/color contracts differ")
    for key, shape, data_range, dtype in (
        ("video_rgb", [120, 512, 512, 3], 255.0, "uint8"),
        ("audio_pcm32k_stereo", [160000, 2], 2.0, "float32"),
    ):
        value = media[key]
        if (
            value.get("status") != "completed"
            or value.get("shape") != shape
            or value.get("data_range") != data_range
            or value.get("reference_dtype") != dtype
            or value.get("candidate_dtype") != dtype
            or value.get("reference_nonfinite") != 0
            or value.get("candidate_nonfinite") != 0
        ):
            raise ValueError("joined RGB/PCM numerical contract differs")
    if media.get("optional", {}).get("ssim", {}).get("status") != "completed":
        raise ValueError("paired media requires completed RGB SSIM")


def execute(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema_version=1,
        status="failed",
        quality_decision="not_evaluated",
        scope="matched BF16 versus verified prepared-H256 captured latents",
        data_range=1.0,
        data_range_note="latent PSNR convention, not a quality score",
        evaluator_sha256=metrics.sha256(Path(__file__)),
        numerical_metric_sha256=metrics.sha256(Path(metrics.__file__)),
    )
    try:
        input_run_hashes = {
            label: metrics.sha256(directory / "run.json")
            for label, directory in (("bf16", args.bf16), ("hadamard", args.hadamard))
        }
        reference_run = capture.read_json(args.bf16 / "run.json")
        candidate_run = capture.read_json(args.hadamard / "run.json")
        record["benchmark"] = validate_pair_recipe(reference_run, candidate_run)
        a, original = capture.validate_capture(args.bf16, "bf16")
        b, prepared = prepared_capture(args.hadamard, original["model"])
        record["sources"] = {"bf16": original, "hadamard": prepared}
        for label, arrays in (("bf16", a), ("hadamard", b)):
            with (out / f"{label}.npz").open("xb") as stream:
                np.savez(stream, **arrays)
            record["sources"][label]["npz_sha256"] = metrics.sha256(out / f"{label}.npz")
        record["latents"] = capture.compare_arrays(a, b)
        if any(value["status"] != "completed" for value in record["latents"].values()):
            raise ValueError("latent numerical comparison failed")
        if args.media or args.media_evaluation:
            runs = (reference_run, candidate_run)
            videos = [
                recorded_video(directory, run)
                for directory, run in zip((args.bf16, args.hadamard), runs, strict=True)
            ]
            if args.media:
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
                media_path = out / "media/evaluation.json"
            else:
                media_path = args.media_evaluation
                media = capture.read_json(media_path)
            validate_media_join(media, videos, runs)
            record["media"] = {
                "evaluation_sha256": metrics.sha256(media_path),
                "video_rgb": media["video_rgb"],
                "audio_pcm32k_stereo": media["audio_pcm32k_stereo"],
                "ssim": media["optional"]["ssim"],
            }
        for label, directory in (("bf16", args.bf16), ("hadamard", args.hadamard)):
            if not (
                metrics.sha256(directory / "run.json")
                == record["sources"][label]["run_sha256"]
                == input_run_hashes[label]
            ):
                raise ValueError("input run record changed during evaluation")
        record["status"] = "completed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    metrics.write_json(out / "benchmark-evaluation.json", record)
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--hadamard", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--media", action="store_true")
    group.add_argument("--media-evaluation", type=Path)
    return p


if __name__ == "__main__":
    result = execute(parser().parse_args())
    print(json.dumps({key: result.get(key) for key in ("status", "error")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)

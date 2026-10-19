"""CPU-only, provenance-checked paired Euler/final latent evaluation."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

import evaluate as metrics


SOURCE_SHA256 = "833076975eeaaf14b0a9db01f6237d7de77930034222b1dd3b85cca7721f04ec"
PATCH_SHA256 = "84065e6139c842aee2cb6d037a6c72ecfa9895e07b1696e9bf638bde3a5a8d1e"
SHAPES = {
    "per_step_video_shape": [9472, 96],
    "per_step_audio_shape": [414, 32],
    "final_video_shape": [24, 37, 32, 32],
    "final_audio_shape": [32, 2, 207],
}
RECIPE_KEYS = (
    "benchmark",
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
)
MODES = {"bf16": "bf16", "native": "upstream-int8", "hadamard": "hadamard-fc2"}


def read_json(path):
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {value}")
        ),
    )


def tensor_spec():
    for step in range(20):
        for kind in ("video", "audio"):
            yield (
                f"euler-step{step:02d}-{kind}.f32",
                f"euler_step{step:02d}_{kind}",
                SHAPES[f"per_step_{kind}_shape"],
            )
    for kind in ("video", "audio"):
        yield f"final-{kind}.f32", f"final_{kind}", SHAPES[f"final_{kind}_shape"]


def validate_capture(directory, mode):
    run_path = directory / "run.json"
    run = read_json(run_path)
    if (
        run.get("status") != "completed"
        or run.get("mode") != mode
        or run.get("capture_latents") is not True
    ):
        raise ValueError("capture run must be completed and match the requested precision mode")
    expected_recipe = {
        "width": 512,
        "height": 512,
        "seconds": 5,
        "steps": 20,
        "active_dit_blocks": 50,
        "reuse": 1,
        "core_reuse": 1,
        "reference_rope": True,
    }
    if any(run.get(k) != v or type(run.get(k)) is not type(v) for k, v in expected_recipe.items()):
        raise ValueError(
            "capture recipe differs from 512-square 5s full 20-step reference-RoPE contract"
        )
    if (
        type(run.get("seed")) is not int
        or not isinstance(run.get("prompt"), str)
        or not run["prompt"]
    ):
        raise ValueError("capture requires an explicit seed and prompt")
    if run.get("patch") != {"patch_sha256": PATCH_SHA256, "source_sha256": SOURCE_SHA256}:
        raise ValueError("capture producer differs from the pinned observation/FC2 patch")
    latent = directory / "latents"
    meta_path = latent / "latent-metadata.json"
    meta = read_json(meta_path)
    validation = run.get("latent_validation", {})
    if validation.get("metadata") != meta or validation.get("metadata_sha256") != metrics.sha256(
        meta_path
    ):
        raise ValueError("latent metadata differs from immutable run record")
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
        int8_mlp=int(mode != "bf16"),
        int8_qkv=int(mode != "bf16"),
        int8_attention_out=int(mode != "bf16"),
        hadamard_fc2=int(mode == "hadamard-fc2"),
        finite=True,
        **SHAPES,
    )
    for key, value in wanted.items():
        if meta.get(key) != value or type(meta.get(key)) is not type(value):
            raise ValueError(f"latent metadata contract differs: {key}")
    specs = list(tensor_spec())
    expected = {name for name, _, _ in specs}
    if {p.name for p in latent.iterdir()} != expected | {"latent-metadata.json"}:
        raise ValueError("latent directory must contain exactly 20 steps and final video/audio")
    records = validation.get("tensors", [])
    indexed = {entry["file"]: entry for entry in records}
    if len(records) != len(expected) or set(indexed) != expected:
        raise ValueError("immutable run tensor inventory differs")
    arrays, evidence = {}, []
    for name, key, shape in specs:
        path, count = latent / name, math.prod(shape)
        digest = metrics.sha256(path)
        entry = indexed[name]
        if entry != {"file": name, "elements": count, "sha256": digest, "finite": True}:
            raise ValueError(f"captured tensor differs from immutable inventory: {name}")
        if path.stat().st_size != count * 4:
            raise ValueError(f"wrong raw float32 size: {name}")
        array = np.fromfile(path, dtype="<f4").reshape(shape)
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite captured tensor: {name}")
        arrays[key] = array
        evidence.append(dict(key=key, source_file=name, shape=shape, sha256=digest, finite=True))
    model = run.get("model", {})
    model_identity = {key: model.get(key) for key in ("repository", "revision", "manifest_sha256")}
    if not all(isinstance(x, str) and x for x in model_identity.values()):
        raise ValueError("missing original model identity")
    provenance = dict(
        run_sha256=metrics.sha256(run_path),
        recipe={k: run.get(k) for k in RECIPE_KEYS},
        model=model_identity,
        patch=run["patch"],
        metadata_sha256=metrics.sha256(meta_path),
        metadata=meta,
        tensors=evidence,
    )
    return arrays, provenance


def compare_arrays(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError("capture keys differ")
    return {key: metrics.numeric_metrics(reference[key], candidate[key], 1.0) for key in reference}


def summarize(native, hadamard):
    summary = {}
    for kind in ("video", "audio"):
        states = []
        for step in range(20):
            key = f"euler_step{step:02d}_{kind}"
            a, b = native[key]["metrics"], hadamard[key]["metrics"]
            states.append(
                dict(
                    step=step,
                    native_relative_l2=a["relative_l2"],
                    hadamard_relative_l2=b["relative_l2"],
                    native_snr_db=a["snr_db"],
                    hadamard_snr_db=b["snr_db"],
                )
            )
        comparable = [
            s
            for s in states
            if s["native_relative_l2"]["status"] == "finite"
            and s["hadamard_relative_l2"]["status"] == "finite"
        ]
        summary[kind] = dict(
            per_step=states,
            finite_comparable_steps=len(comparable),
            hadamard_lower_relative_l2_steps=sum(
                s["hadamard_relative_l2"]["value"] < s["native_relative_l2"]["value"]
                for s in comparable
            ),
            final_native=native[f"final_{kind}"]["metrics"],
            final_hadamard=hadamard[f"final_{kind}"]["metrics"],
        )
    return summary


def execute(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(
        schema_version=1,
        status="failed",
        source={},
        evaluator_sha256=metrics.sha256(Path(__file__)),
        metric_implementation_sha256=metrics.sha256(Path(metrics.__file__)),
        scope="matched packed post-Euler states and unpacked final VAE inputs",
        data_range=1.0,
        data_range_note="convention only; latent PSNR is not model quality",
        quality_decision="not_evaluated",
        source_revision=metrics.source_provenance(),
    )
    try:
        captured = {}
        for label, mode in MODES.items():
            directory = getattr(args, label)
            arrays, provenance = validate_capture(directory, mode)
            record["source"][label] = provenance
            captured[label] = arrays
            with (out / f"{label}.npz").open("xb") as stream:
                np.savez(stream, **arrays)
            record["source"][label]["npz_sha256"] = metrics.sha256(out / f"{label}.npz")
        baseline = record["source"]["bf16"]
        for label in ("native", "hadamard"):
            for key in ("recipe", "model", "patch"):
                if record["source"][label][key] != baseline[key]:
                    raise ValueError(f"paired captures differ in {key}: {label}")
        record["comparisons"] = {
            "bf16_vs_native": compare_arrays(captured["bf16"], captured["native"]),
            "bf16_vs_hadamard": compare_arrays(captured["bf16"], captured["hadamard"]),
        }
        if any(
            m["status"] != "completed" for c in record["comparisons"].values() for m in c.values()
        ):
            raise ValueError("a numerical comparison failed")
        record["summary"] = summarize(
            record["comparisons"]["bf16_vs_native"], record["comparisons"]["bf16_vs_hadamard"]
        )
        for label in MODES:
            if (
                metrics.sha256(getattr(args, label) / "run.json")
                != record["source"][label]["run_sha256"]
            ):
                raise ValueError("source run record changed during evaluation")
        record["status"] = "completed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["evaluation_wall_seconds"] = time.perf_counter() - started
    metrics.write_json(out / "latent-evaluation.json", record)
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        p.add_argument(f"--{mode}", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


if __name__ == "__main__":
    result = execute(parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "error": result.get("error"),
                "summary": result.get("summary"),
            },
            allow_nan=False,
        )
    )
    raise SystemExit(0 if result["status"] == "completed" else 1)

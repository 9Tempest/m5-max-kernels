"""CPU-only strict drift reports for new kernel policies and frozen BF16/H256 captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluate as metrics
import evaluate_benchmark_pair as media_binding
import evaluate_latents as historical
import kernel_evidence as provenance
import kernel_integration as contract
import run as base
import run_fc2_experiment as capture


SCHEMA = "h3_kernel_integration_evaluation_v1"
MODEL = dict(
    repository="MiniMaxAI/MiniMax-H3",
    revision=base.MODEL_REVISION,
    manifest_sha256=base.MODEL_MANIFEST_SHA256,
)


def original_capture(directory, mode, seed):
    record = capture.load_json(directory / "run.json")
    for key, value in contract.RECIPE.items():
        if key not in ("mode", "fps", "source_frames"):
            contract.exact(record.get(key), value, "historical recipe " + key)
    contract.exact(record.get("seed"), seed, "historical seed")
    arrays, evidence = historical.validate_capture(directory, mode)
    contract.exact(evidence["model"], MODEL, "historical original model")
    return record, arrays, evidence


def candidate_capture(directory, expected_pin, seed):
    record_path = directory / "run.json"
    initial = metrics.sha256(record_path)
    record = capture.load_json(record_path)
    for key, value in contract.RECIPE.items():
        contract.exact(record.get(key), value, "candidate recipe " + key)
    for key, value in dict(
        schema=contract.SCHEMA,
        kind="capture",
        status="completed",
        seed=seed,
        capture_latents=True,
        diagnostic=True,
        prepared_checkpoint=False,
    ).items():
        contract.exact(record.get(key), value, "candidate " + key)
    policy, implementation = record.get("policy"), record.get("implementation")
    flags = contract.flags(policy, implementation)
    contract.exact(record.get("requested_runtime_flags"), flags, "requested policy flags")
    for key, value in flags.items():
        contract.exact(
            record.get("runtime_overrides", {}).get(key), value, "generation flag " + key
        )
    evidence = provenance.producer(directory, record, expected_pin)
    process = record.get("measurements", {}).get("generation", {})
    contract.exact(process.get("returncode"), 0, "generation success")
    contract.exact(process.get("fallback_detected"), False, "generation fallback")
    contract.exact(process.get("log"), "generation.log", "generation log name")
    log_path = directory / "generation.log"
    contract.exact(metrics.sha256(log_path), process.get("log_sha256"), "generation log SHA")
    contract.validate_log(log_path.read_text(), policy, implementation)
    validation = contract.validate_latents(directory / "latents", policy, implementation, seed)
    contract.exact(validation, record.get("latent_validation"), "capture file inventories")
    arrays = {}
    for step in range(20):
        for kind in ("video", "audio"):
            arrays[f"euler_step{step:02d}_{kind}"] = np.fromfile(
                directory / f"latents/euler-step{step:02d}-{kind}.f32", dtype="<f4"
            ).reshape(contract.SHAPES[f"per_step_{kind}_shape"])
    for kind in ("video", "audio"):
        arrays["final_" + kind] = np.fromfile(
            directory / f"latents/final-{kind}.f32", dtype="<f4"
        ).reshape(contract.SHAPES[f"final_{kind}_shape"])
    contract.exact(metrics.sha256(record_path), initial, "candidate run unchanged")
    return (
        record,
        arrays,
        dict(
            run_sha256=initial,
            policy=policy,
            implementation=implementation,
            seed=seed,
            **evidence,
            metadata_sha256=validation["metadata_sha256"],
            tensors=validation["tensors"],
        ),
    )


def compare_with_tail(reference, candidate):
    result = historical.compare_arrays(reference, candidate)
    for key, value in result.items():
        if value["status"] != "completed":
            raise ValueError("nonfinite or invalid latent comparison")
        error = np.abs(candidate[key].astype(np.float64) - reference[key].astype(np.float64))
        value["absolute_error_quantiles"] = {
            name: float(np.quantile(error, q))
            for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("p999", 0.999))
        }
    return result


def media_pair(reference_dir, candidate_dir, reference_run, candidate_run, output):
    videos = [
        media_binding.recorded_video(path, run)
        for path, run in ((reference_dir, reference_run), (candidate_dir, candidate_run))
    ]
    record = metrics.execute(
        SimpleNamespace(
            reference=videos[0],
            candidate=videos[1],
            kind="media",
            data_range=None,
            audio_data_range=2.0,
            ssim=True,
            lpips=False,
            vbench=False,
            output_dir=output,
        )
    )
    media_binding.validate_media_join(record, videos, (reference_run, candidate_run))
    return dict(
        evaluation_sha256=metrics.sha256(output / "evaluation.json"),
        video_rgb=record["video_rgb"],
        audio_pcm32k_stereo=record["audio_pcm32k_stereo"],
        ssim=record["optional"]["ssim"],
    )


def execute(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema=SCHEMA,
        status="failed",
        quality_decision="not_evaluated",
        evaluator_sha256=metrics.sha256(Path(__file__)),
        dependencies={
            name: metrics.sha256(Path(__file__).with_name(name))
            for name in (
                "kernel_integration.py",
                "kernel_evidence.py",
                "evaluate.py",
                "evaluate_latents.py",
                "evaluate_benchmark_pair.py",
                "run_fc2_experiment.py",
                "run.py",
            )
        },
        data_range=1.0,
        scope="BF16-relative numeric/structural drift, not perceptual ground truth; latent PSNR range1 is a convention",
        comparisons={},
        sources={},
    )
    try:
        directories = [args.bf16, args.old_hadamard, *args.run_dir]
        hashes = {str(path.resolve()): metrics.sha256(path / "run.json") for path in directories}
        if len(hashes) != len(directories) or not args.run_dir:
            raise ValueError("distinct BF16, historical H and at least one candidate are required")
        seed = capture.load_json(args.bf16 / "run.json").get("seed")
        if seed not in (42, 43):
            raise ValueError("evaluation requires seed42 or43")
        bf, a, evidence = original_capture(args.bf16, "bf16", seed)
        old, h, old_evidence = original_capture(args.old_hadamard, "hadamard-fc2", seed)
        record.update(
            seed=seed,
            source_pin_sha256=args.source_manifest_sha256,
            sources=dict(bf16=evidence, old_hadamard=old_evidence),
        )
        record["comparisons"]["old-hadamard"] = dict(against_bf16=compare_with_tail(a, h))
        if args.media:
            record["comparisons"]["old-hadamard"]["media"] = media_pair(
                args.bf16, args.old_hadamard, bf, old, output / "old-hadamard-media"
            )
        values = {}
        for directory in args.run_dir:
            run, arrays, evidence = candidate_capture(directory, args.source_manifest_sha256, seed)
            key = run["policy"] + "-" + run["implementation"]
            if key in values:
                raise ValueError("duplicate policy/implementation candidate")
            values[key] = arrays
            record["sources"][key] = evidence
            value = dict(
                against_bf16=compare_with_tail(a, arrays),
                against_old_hadamard=compare_with_tail(h, arrays),
            )
            if args.media:
                value["media"] = media_pair(
                    args.bf16, directory, bf, run, output / (key + "-media")
                )
            record["comparisons"][key] = value
        if "baseline-fused" in values:
            record["baseline_replay_exact"] = all(
                np.array_equal(values["baseline-fused"][key], h[key]) for key in h
            )
        record["same_policy_implementation"] = {}
        for policy in ("adaln-h-only", "combined"):
            staged, fused = policy + "-staged", policy + "-fused"
            if staged in values and fused in values:
                record["same_policy_implementation"][policy] = compare_with_tail(
                    values[staged], values[fused]
                )
        for directory in directories:
            contract.exact(
                metrics.sha256(directory / "run.json"),
                hashes[str(directory.resolve())],
                "input run remained immutable",
            )
        record["status"] = "completed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    metrics.write_json(output / "kernel-evaluation.json", record)
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--old-hadamard", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, action="append", required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--media", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    return p


if __name__ == "__main__":
    result = execute(parser().parse_args())
    print(json.dumps({key: result.get(key) for key in ("status", "error")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)

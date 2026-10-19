"""Verify and export one matched none/attention resident pair; never runs Metal.

The two inputs are completed run_native_session.py raw run.json records with
their original local evidence. Publication summaries alone are insufficient.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import publish_resident as resident
import publish_samples as public
import run as base
import run_native_session as native
import run_session as session


SOURCE_SHA256 = "83ef61ace55a0ca9d8c2735d627e8c0ca96692bfebaae2376fae3b6fd9dbef7b"
PATCH_SHA256 = "cbd7239d98dffac8feb469caa7fdeee504a564dda51f625f95406284b2a806fe"
HARNESS_SHA256 = {
    "run.py": "4af33663e29de566716a373d5b5a03f4796c157a092f21aaf6d8df7bda8c82e7",
    "run_session.py": "3ea762071e36f4bb047181ce799bee0e48ca7efad05a4436f93c695d589fa31b",
    "session.c": "52a52de9acdcf3d778d5c5fb0560c3eae0c848d89cbd3320e49d43eeb1448c2c",
    "run_native_session.py": "643c410c12ca3a96f85757c95f69e0e43d19808a878ee2312521cdbcc7a70b2f",
    "run_fc2_experiment.py": "99b7ef56a9e43f9a2fe62e2e50b94f30c4a3f44819a2201d4eb0e575cab0a562",
    "run_fused_campaign.py": "8eade2abe2665794481e758b4a09b6641efe66ca1d4bff11a575aa370a4be380",
    "fused.patch": PATCH_SHA256,
    "hadamard.patch": "84065e6139c842aee2cb6d037a6c72ecfa9895e07b1696e9bf638bde3a5a8d1e",
    "fused_validate.c": "6105dad5dd401aea70f4db1ab153b49bb6817a315c1ff314741d2e24e22bce30",
    "fused_reference.h": "18e1a880800384ec20a5279f16ef6cacdee70a2b61b037cf5d04d5886e5684d7",
    "model-files.json": "7597d33ca71af128705e802150b4b39395cf20f96bddc45d2041d668145c27be",
    "UPSTREAM-LICENSE": "663e51edcd9a3a8f858018ce6924ad2895a24c269c26c39ad75325bcd0f7e5bc",
    "locking.py": "a9957d5f9f82424c15d76b7d76b701676dcb47112d6d13a01697cb719556eff4",
}
RECIPE = {
    "benchmark": "fox-winter-t2va-cached-v1",
    "prompt": base.PROMPT,
    "mode": "hadamard-fc2",
    "width": 512,
    "height": 512,
    "steps": 20,
    "source_frames": 124,
    "warmups": 1,
    "repetitions": 3,
    "active_dit_blocks": 50,
    "reuse": 1,
    "core_reuse": 1,
    "use_reference_rope": True,
}
MODEL = {
    "repository": "MiniMaxAI/MiniMax-H3",
    "revision": base.MODEL_REVISION,
    "manifest_sha256": HARNESS_SHA256["model-files.json"],
}
ENVIRONMENT_KEYS = (
    "system",
    "release",
    "machine",
    "python",
    "chip",
    "memory_bytes",
    "compiler",
    "xcode",
    "ffprobe",
)


def exact(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"resident mismatch: {label}")


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("resident process needs timezone-aware UTC timestamps")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("resident process needs timezone-aware UTC timestamps")
    return result.timestamp()


def local_file(directory, relative):
    """Evidence is confined to this run, including when a symlink is present."""
    path = directory / relative
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("resident evidence escapes its run directory")
    return path


def validate_snapshot(directory, record):
    exact(record.get("harness_files"), HARNESS_SHA256, "pinned runtime harness hashes")
    entries = record.get("harness_snapshot")
    if not isinstance(entries, dict) or set(entries) != set(HARNESS_SHA256):
        raise ValueError("resident immutable harness snapshot is incomplete")
    manifest = public.load_json(local_file(directory, "harness-snapshot/manifest.json"))
    exact(manifest, entries, "run-local snapshot manifest")
    for name, digest in HARNESS_SHA256.items():
        entry = entries[name]
        exact(entry.get("file"), "harness-snapshot/" + name, "snapshot filename")
        exact(entry.get("sha256"), digest, "snapshot declared hash")
        exact(base.sha256(local_file(directory, entry["file"])), digest, "snapshot file hash")
    driver = record.get("driver", {})
    exact(driver.get("source_sha256"), HARNESS_SHA256["session.c"], "native driver source")
    exact(driver.get("wrapper_sha256"), HARNESS_SHA256["run_native_session.py"], "driver wrapper")
    exact(
        base.sha256(local_file(directory, "native-session")),
        driver.get("binary_sha256"),
        "driver binary",
    )
    if not isinstance(driver.get("library_sha256"), str) or len(driver["library_sha256"]) != 64:
        raise ValueError("missing recorded native library digest")


def validate_process(directory, record, generations):
    process = record.get("measurements", {}).get("session_process", {})
    exact(process.get("returncode"), 0, "session return code")
    exact(process.get("fallback_detected"), False, "session fallback")
    exact(process.get("log"), "session-process.log", "session process log name")
    exact(
        base.sha256(local_file(directory, process["log"])),
        process.get("log_sha256"),
        "session process log",
    )
    start, end = (timestamp(process.get(key)) for key in ("started_utc", "finished_utc"))
    wall = process.get("wall_seconds")
    if (
        type(wall) not in (int, float)
        or not math.isfinite(wall)
        or end <= start
        or wall < sum(event["wall_seconds"] for event in generations)
        or abs(end - start - wall) > 0.1
    ):
        raise ValueError("session process interval or elapsed time is inconsistent")
    previous = None
    for event in generations:
        epoch = event.get("started_unix_seconds")
        if type(epoch) is not int or not start - 1 < epoch < end:
            raise ValueError("generation start does not lie within the session process")
        # Driver start timestamps have one-second precision; request wall is monotonic.
        if epoch + event["wall_seconds"] > end or (
            previous and epoch < previous[0] + math.floor(previous[1])
        ):
            raise ValueError("resident generation intervals overlap or leave the process")
        previous = (epoch, event["wall_seconds"])
    return {
        key: process[key] for key in ("started_utc", "finished_utc", "wall_seconds", "log_sha256")
    }


def validate_builds(directory, record):
    """Match compiler flags while allowing each run's snapshot/output paths."""
    measurements = record.get("measurements", {})
    for name, stem in (("build", "build"), ("driver_build", "driver-build")):
        item = measurements.get(name, {})
        exact(item.get("returncode"), 0, name + " return code")
        exact(item.get("fallback_detected"), False, name + " fallback")
        exact(item.get("log"), stem + ".log", name + " log name")
        exact(
            base.sha256(local_file(directory, item["log"])),
            item.get("log_sha256"),
            name + " log hash",
        )
    exact(measurements["build"].get("command"), ["make", "-B", "-j8"], "native build command")
    command = measurements["driver_build"].get("command")
    expected = [
        "clang",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-D_DARWIN_C_SOURCE",
        "-DH3_SESSION_EXPERIMENT=1",
        "-I",
        "<source>",
        "<run>/harness-snapshot/session.c",
        "<source>/libh3.a",
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-framework",
        "MetalPerformanceShaders",
        "-framework",
        "MetalPerformanceShadersGraph",
        "-framework",
        "Accelerate",
        "-licucore",
        "-lm",
        "-o",
        "<run>/native-session",
    ]
    if not isinstance(command, list) or len(command) != len(expected):
        raise ValueError("driver compile command differs from pinned flags")
    normalized = command.copy()
    source = (base.ROOT / command[9]).resolve()
    for index, path in (
        (10, directory / "harness-snapshot/session.c"),
        (11, source / "libh3.a"),
        (25, directory / "native-session"),
    ):
        exact(
            (base.ROOT / command[index]).resolve(),
            path.resolve(),
            "driver compile input/output binding",
        )
        normalized[index] = expected[index]
    normalized[9] = "<source>"
    exact(normalized, expected, "driver compiler flags")
    for item in (measurements["build"], measurements["driver_build"]):
        exact((base.ROOT / item["cwd"]).resolve(), source, "compiler working directory")
    return {"make": ["make", "-B", "-j8"], "driver": normalized}


def validate_run(path, fusion):
    initial_digest = base.sha256(path)
    record = public.load_json(path)
    exact(record.get("schema_version"), 2, "raw resident schema")
    for key, expected in RECIPE.items():
        exact(record.get(key), expected, key)
    exact(record.get("fusion"), fusion, "fusion arm")
    seed = record.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("resident seed must be a nonnegative signed 64-bit integer")
    if (
        record.get("prepared_checkpoint")
        or record.get("capture_latents")
        or record.get("diagnostic")
    ):
        raise ValueError("resident comparison excludes prepared-loader and capture experiments")
    exact(
        record.get("all_repetition_videos_byte_identical"),
        True,
        "all four original videos identical",
    )
    exact(
        record.get("patch"),
        {"patch_sha256": PATCH_SHA256, "source_sha256": SOURCE_SHA256},
        "frozen fused patch",
    )
    exact(record.get("upstream", {}).get("source_sha256"), SOURCE_SHA256, "frozen fused source")
    exact(record.get("upstream", {}).get("revision"), base.UPSTREAM_REVISION, "upstream revision")
    validate_snapshot(path.parent, record)
    compilation = validate_builds(path.parent, record)
    for event in record.get("session_events", []):
        if event.get("event") == "generation":
            for field in ("output", "profile_log"):
                name = event.get(field)
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError("unsafe resident evidence filename")
                local_file(path.parent, "session/" + name)
    local_file(path.parent, "session/session.jsonl")
    # Recheck raw event/log/media bytes through the existing schema-2 exporter.
    summary = resident.summarize(path)
    exact(summary["model"], MODEL, "original model identity")
    exact(
        record.get("session_events_sha256"),
        summary["session_events_sha256"],
        "required event digest",
    )
    args = SimpleNamespace(mode="hadamard-fc2", fusion=fusion, seed=seed)
    generations = native.validate_events(record["session_events"], args)
    flags = native.runtime_flags(args)
    exact(
        record["session_events"][1], {"event": "runtime_flags", **flags}, "exact process flag event"
    )
    for key in ("requested_runtime_flags", "observed_runtime_flags"):
        exact(record.get(key), flags, key)
    expected_overrides = {
        "H3_NAX": "qkv-attn",
        "H3_NAX_DIAGNOSTIC": "1",
        "H3_PROFILE": "1",
        **{key: value for key, value in flags.items() if value != "unset"},
    }
    exact(record.get("runtime_overrides"), expected_overrides, "only permitted runtime controls")
    policies = []
    for event, profile, video in zip(
        generations, record["profiles"], record["videos"], strict=True
    ):
        log = local_file(path.parent, "session/" + event["profile_log"]).read_text()
        if "TensorOps compile failed:" in log:
            raise ValueError("TensorOps fallback in resident generation log")
        policies.append(native.validate_policy_log(log, event["index"], args))
        exact(profile["phases"], base.parse_profile(log), "profile extracted from immutable log")
        for key, value in {"width": 512, "height": 512, "frames": 124, "fps": 24}.items():
            exact(video.get(key), value, "original video " + key)
        if abs(video.get("duration_seconds", 0) - 124 / 24) > 1e-5:
            raise ValueError("resident comparison requires original 124-frame videos")
        local_file(path.parent, "session/" + event["output"])
    exact(record.get("policy_evidence"), policies, "precision/fusion/cache log evidence")
    expected_shape = session.shape_from_cache(
        base.temporal_shape(5, 512, 512), generations, "upstream-int8"
    )
    exact(record.get("shape"), expected_shape, "shape from observed resident cache")
    process = validate_process(path.parent, record, generations)
    environment = record.get("environment", {})
    if any(key not in environment for key in ENVIRONMENT_KEYS):
        raise ValueError("resident environment/toolchain provenance is incomplete")
    exact(environment["system"], "Darwin", "Metal host OS")
    exact(environment["machine"], "arm64", "Metal host architecture")
    device = record["session_events"][3]
    exact(device["name"], environment["chip"], "device model")
    exact(device["physical_memory_bytes"], environment["memory_bytes"], "device memory")
    exact(base.sha256(path), initial_digest, "raw record changed during verification")
    exact(summary["source_run_sha256"], initial_digest, "summarized raw record digest")
    return record, summary, process, compilation


def metric_pair(baseline, candidate):
    def stats(values):
        return {
            "raw_seconds": values,
            "count": len(values),
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "range_seconds": max(values) - min(values),
        }

    a, b = stats(baseline), stats(candidate)
    return {
        "baseline": a,
        "candidate": b,
        "baseline_median_divided_by_candidate": a["median_seconds"] / b["median_seconds"],
        "median_seconds_saved": a["median_seconds"] - b["median_seconds"],
        "median_reduction_percent": 100 * (1 - b["median_seconds"] / a["median_seconds"]),
        "interpretation": "ratio/difference of separate three-sample medians; not paired per-repetition ratios, confidence intervals or statistical significance",
    }


def compare(baseline_path, candidate_path):
    for name in ("run.py", "run_session.py", "run_native_session.py", "run_fused_campaign.py"):
        exact(
            base.sha256(Path(__file__).with_name(name)),
            HARNESS_SHA256[name],
            "validator dependency pin",
        )
    b, bs, bp, bc = validate_run(baseline_path, "none")
    c, cs, cp, cc = validate_run(candidate_path, "attention")
    exact(cc, bc, "cross-arm build commands")
    for key in ("seed", "shape", "timing_contract", "harness_files", "environment"):
        exact(c[key], b[key], "cross-arm " + key)
    exact(cs["model"], bs["model"], "cross-arm model")
    exact(cs["video_sha256"], bs["video_sha256"], "cross-arm original video SHA256")
    # UTC process bounds exclude overlapping wrapper queue waits/preflight.
    b0, b1 = timestamp(bp["started_utc"]), timestamp(bp["finished_utc"])
    c0, c1 = timestamp(cp["started_utc"]), timestamp(cp["finished_utc"])
    if not (b1 <= c0 or c1 <= b0):
        raise ValueError("baseline and candidate session process intervals overlap")
    result = {
        "schema_version": 1,
        "status": "completed",
        "kind": "matched_resident_none_attention_pair",
        "recipe": RECIPE | {"seed": b["seed"]},
        "shape": b["shape"],
        "model": bs["model"],
        "source_sha256": SOURCE_SHA256,
        "patch_sha256": PATCH_SHA256,
        "runtime_harness_sha256": HARNESS_SHA256,
        "environment": bs["environment"],
        "compiler": b["environment"]["compiler"].splitlines()[0],
        "timing_contract": b["timing_contract"],
        "compilation_commands": bc,
        "validation": {
            "same_runtime_source_and_pinned_harness": True,
            "same_recipe_model_environment": True,
            "full_warmup_plus_three_requests_each": True,
            "raw_events_callbacks_logs_caches_media_verified": True,
            "all_eight_original_videos_byte_identical": True,
            "original_video_sha256": bs["video_sha256"][0],
            "nonoverlapping_session_process_intervals": True,
            "permitted_policy_difference": "H256 fixed; none leaves both gate flags unset, attention explicitly sets FC2=0 and ATTN_OUT=1",
            "git_provenance": "aggregate checkout revision/dirty state may differ due documentation; all runtime snapshot files are pinned and equal",
            "binary_provenance": "each retained executable is hashed against its run record; source/toolchain match is required, archive/executable hashes are recorded separately",
        },
        "arms": {
            "baseline": {
                "fusion": "none",
                "summary": "baseline.json",
                "source_run_sha256": bs["source_run_sha256"],
                "process_interval": bp,
                "driver": bs["driver"],
            },
            "candidate": {
                "fusion": "attention",
                "summary": "candidate.json",
                "source_run_sha256": cs["source_run_sha256"],
                "process_interval": cp,
                "driver": cs["driver"],
            },
        },
        "execution_order": ["baseline", "candidate"] if b1 <= c0 else ["candidate", "baseline"],
        "cached_generation": metric_pair(
            b["cached_generation"]["raw_seconds"], c["cached_generation"]["raw_seconds"]
        ),
        "cached_denoise_callback": metric_pair(
            b["cached_denoise_callback"]["raw_seconds"], c["cached_denoise_callback"]["raw_seconds"]
        ),
        "claims": {
            "stable_promotion": "not_established",
            "new_best": "not_claimed",
            "statistical_significance": "not_evaluated",
            "independent_replication": "not_established_by_three_requests_within_one_process",
            "compilation_free_end_to_end": "not_claimed",
            "quality": "same-seed byte-identical output only; not official-pipeline or perceptual acceptance",
        },
        "limitations": [
            "One complete warmup and three same-seed timed requests per arm; one sequential process per policy, no interleaving within process.",
            "Generation includes decode and original 124-frame AV export; repeated audio VAE setup prevents a compilation-free end-to-end claim.",
            "Denoise uses synchronized native callback offsets, including online preparation and callback overhead; cached native profile wall is not substituted.",
            "No latent capture is enabled. Shared-lock scheduling plus nonoverlap checks prevent these two arms overlapping; absence of unrelated external GPU load is not proven.",
            "A second seed with reversed process order and independent repeated comparisons is still needed before stable performance promotion.",
        ],
        "exporter_files_sha256": {
            name: base.sha256(Path(__file__).with_name(name))
            for name in (
                "compare_resident.py",
                "publish_resident.py",
                "publish_samples.py",
                "run_native_session.py",
                "run_session.py",
                "run.py",
            )
        },
    }
    public.public_text_check(result)
    return result, bs, cs


def readme(result):
    lines = [
        "# Matched resident none / attention comparison",
        "",
        f"Seed {result['recipe']['seed']}; H256 FC2 fixed; the same frozen fused runtime with attention-output gate fusion disabled/enabled.",
        "",
        "Each arm completes one full warmup and three timed requests. All eight original 124-frame videos are byte-identical. Raw timing events, callback boundaries, policy/cache log markers, source snapshots and media checksums are verified before export.",
        "",
        "| Synchronized host measurement | Baseline median (s) | Attention median (s) | Baseline / attention | Median seconds saved |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("cached_generation", "Complete cached generation"),
        ("cached_denoise_callback", "Denoise callback"),
    ):
        metric = result[key]
        lines.append(
            f"| {label} | {metric['baseline']['median_seconds']:.6f} | {metric['candidate']['median_seconds']:.6f} | {metric['baseline_median_divided_by_candidate']:.6f}× | {metric['median_seconds_saved']:.6f} |"
        )
    lines += [
        "",
        "Raw samples, ranges and exact process ordering are in [comparison.json](comparison.json). [Baseline evidence](baseline.json) and [attention evidence](candidate.json) are sanitized schema-2 summaries, not byte-identical copies of the raw run records. Original raw-record SHA256s are retained; no model weights or private paths are exported.",
        "",
        *result["limitations"],
        "",
        "These descriptive ratios do not establish statistical significance, independent replication or a new best result.",
        "",
    ]
    return "\n".join(lines)


def export(baseline_path, candidate_path, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        result, baseline, candidate = compare(baseline_path, candidate_path)
        base.write_json(output / "baseline.json", baseline)
        base.write_json(output / "candidate.json", candidate)
        for arm in result["arms"].values():
            arm["summary_sha256"] = base.sha256(output / arm["summary"])
        with (output / "README.md").open("x") as handle:
            handle.write(readme(result))
    except Exception as error:
        reason = (
            str(error)
            if isinstance(error, ValueError)
            else "required local evidence could not be validated"
        )
        try:
            public.public_text_check(reason)
        except ValueError:
            reason = "local evidence validation failed; private path details omitted"
        result = {
            "schema_version": 1,
            "status": "rejected",
            "error_type": type(error).__name__,
            "reason": reason,
        }
    base.write_json(output / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.baseline_run, args.candidate_run, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "reason") if key in result}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify two-seed, reversed-order resident integration experiments on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import compare_resident as prior
import kernel_evidence as evidence
import kernel_integration as contract
import run as base
import run_fc2_experiment as capture
import run_session as session


SCHEMA = "h3_kernel_resident_comparison_v1"
ORDER = {42: list(contract.POLICIES), 43: list(reversed(contract.POLICIES))}


def read_process(directory, record, generations):
    return prior.validate_process(directory, record, generations)


def validate_builds(directory, record):
    measurements = record.get("measurements", {})
    for name, stem in (("build", "build"), ("driver_build", "driver-build")):
        value = measurements.get(name, {})
        contract.exact(value.get("returncode"), 0, name + " success")
        contract.exact(value.get("fallback_detected"), False, name + " fallback")
        contract.exact(value.get("log"), stem + ".log", name + " log")
        contract.exact(
            base.sha256(evidence.local_file(directory, value["log"])),
            value.get("log_sha256"),
            name + " log SHA",
        )
    contract.exact(measurements["build"].get("command"), ["make", "-B", "-j8"], "native build")
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
        "<run>/harness-snapshot/kernel_session.c",
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
        "<run>/kernel-session",
    ]
    if not isinstance(command, list) or len(command) != len(expected):
        raise ValueError("resident compiler command differs")
    source = (base.ROOT / command[9]).resolve()
    normalized = command.copy()
    for index, target in (
        (10, directory / "harness-snapshot/kernel_session.c"),
        (11, source / "libh3.a"),
        (25, directory / "kernel-session"),
    ):
        contract.exact(
            (base.ROOT / command[index]).resolve(), target.resolve(), "compiler file binding"
        )
        normalized[index] = expected[index]
    normalized[9] = "<source>"
    contract.exact(normalized, expected, "compiler options")
    for name in ("build", "driver_build"):
        contract.exact((base.ROOT / measurements[name]["cwd"]).resolve(), source, "build cwd")
    return dict(make=["make", "-B", "-j8"], driver=normalized)


def validate_run(directory, pin):
    path = directory / "run.json"
    initial = base.sha256(path)
    run = capture.load_json(path)
    for key, value in contract.RECIPE.items():
        contract.exact(run.get(key), value, "resident recipe " + key)
    for key, value in dict(
        schema=contract.SCHEMA,
        kind="resident",
        status="completed",
        implementation="fused",
        capture_latents=False,
        capture_adaln=False,
        diagnostic=False,
        prepared_checkpoint=False,
        warmups=1,
        repetitions=3,
        all_repetition_videos_byte_identical=True,
    ).items():
        contract.exact(run.get(key), value, "resident " + key)
    policy, seed = run.get("policy"), run.get("seed")
    if type(seed) is not int or seed not in (42, 43):
        raise ValueError("resident campaign requires seeds42/43")
    flags = contract.flags(policy)
    provenance = evidence.producer(directory, run, pin)
    for key in ("requested_runtime_flags", "observed_runtime_flags"):
        contract.exact(run.get(key), flags, key)
    contract.exact(
        run.get("runtime_overrides"),
        flags | dict(H3_NAX="qkv-attn", H3_NAX_DIAGNOSTIC="1", H3_PROFILE="1"),
        "only permitted runtime flags",
    )
    compilation = validate_builds(directory, run)
    events_path = evidence.local_file(directory, "session/session.jsonl")
    contract.exact(base.sha256(events_path), run.get("session_events_sha256"), "raw event SHA")
    events = [
        json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        for line in events_path.read_text().splitlines()
    ]
    contract.exact(events, run.get("session_events"), "raw event values")
    generations = contract.validate_events(events, policy, seed)
    policies, videos = [], []
    for event, profile, video in zip(generations, run["profiles"], run["videos"], strict=True):
        log_path = evidence.local_file(directory, "session/" + event["profile_log"])
        contract.exact(base.sha256(log_path), profile.get("log_sha256"), "request profile SHA")
        log = log_path.read_text()
        contract.exact(profile.get("index"), event["index"], "profile index")
        contract.exact(profile.get("phases"), base.parse_profile(log), "parsed profile")
        policies.append(contract.validate_log(log, policy, "fused", event["index"]))
        name = "session/" + event["output"]
        contract.exact(video.get("file"), name, "video filename")
        contract.exact(
            base.sha256(evidence.local_file(directory, name)), video.get("sha256"), "video SHA"
        )
        probe = capture.load_json(
            evidence.local_file(directory, Path(event["output"]).stem + "-probe.log")
        )
        value = base.validate_video(probe, 512, 512, 124)
        contract.exact(
            video, value | dict(file=name, sha256=video["sha256"]), "probed resident geometry"
        )
        videos.append(video["sha256"])
    if len(set(videos)) != 1:
        raise ValueError("same-seed resident outputs differ within the process")
    contract.exact(run.get("policy_evidence"), policies, "observed runtime policy")
    contract.exact(
        run.get("shape"),
        session.shape_from_cache(base.temporal_shape(5, 512, 512), generations, "upstream-int8"),
        "runtime cache shape",
    )
    process = read_process(directory, run, generations)
    environment = run.get("environment", {})
    if any(key not in environment for key in prior.ENVIRONMENT_KEYS):
        raise ValueError("incomplete resident environment")
    contract.exact(environment["system"], "Darwin", "Metal OS")
    contract.exact(environment["machine"], "arm64", "Metal architecture")
    contract.exact(events[3].get("name"), environment["chip"], "device model")
    contract.exact(
        events[3].get("physical_memory_bytes"), environment["memory_bytes"], "device memory"
    )
    timing = {}
    for name, key in (
        ("cached_generation", "wall_seconds"),
        ("cached_denoise_callback", "denoise_callback_seconds"),
    ):
        timing[name] = session.summarize([item[key] for item in generations[1:]])
        contract.exact(run.get(name), timing[name], "raw timing summary")
    contract.exact(base.sha256(path), initial, "run record unchanged")
    return dict(
        run_sha256=initial,
        policy=policy,
        seed=seed,
        producer=provenance,
        environment=environment,
        compilation=compilation,
        process=process,
        video_sha256=videos,
        timing=timing,
        timing_contract=run["timing_contract"],
        shape=run["shape"],
        warmup_seconds=generations[0]["wall_seconds"],
    )


def compare_records(records, policies=None):
    policies = list(contract.POLICIES) if policies is None else list(policies)
    if (
        len(policies) < 2
        or len(set(policies)) != len(policies)
        or "baseline" not in policies
        or not set(policies) <= set(contract.POLICIES)
    ):
        raise ValueError("a unique baseline plus candidate policy set is required")
    indexed = {(item["seed"], item["policy"]): item for item in records}
    expected = {(seed, policy) for seed in (42, 43) for policy in policies}
    if len(records) != 2 * len(policies) or set(indexed) != expected:
        raise ValueError(
            "exactly one resident process per requested policy and each seed is required"
        )
    reference = records[0]
    for item in records[1:]:
        for key in ("environment", "compilation", "timing_contract", "shape"):
            contract.exact(item[key], reference[key], "matched " + key)
        for key in ("source_pin_sha256", "source_sha256", "model", "harness"):
            contract.exact(
                item["producer"][key], reference["producer"][key], "matched producer " + key
            )
    ordered = sorted(records, key=lambda r: prior.timestamp(r["process"]["started_utc"]))
    for first, second in zip(ordered, ordered[1:]):
        if prior.timestamp(first["process"]["finished_utc"]) > prior.timestamp(
            second["process"]["started_utc"]
        ):
            raise ValueError("resident processes overlap")
    for seed, wanted in ORDER.items():
        contract.exact(
            [r["policy"] for r in ordered if r["seed"] == seed],
            [policy for policy in wanted if policy in policies],
            "counterbalanced seed order",
        )
    pairs = {}
    for left, right in (
        ("baseline", "qkv-only"),
        ("baseline", "adaln-h-only"),
        ("baseline", "combined"),
        ("qkv-only", "combined"),
        ("adaln-h-only", "combined"),
    ):
        if left not in policies or right not in policies:
            continue
        pair = {}
        for seed in (42, 43):
            a, b = indexed[seed, left], indexed[seed, right]
            pair[str(seed)] = {
                key: prior.metric_pair(
                    a["timing"][key]["raw_seconds"], b["timing"][key]["raw_seconds"]
                )
                for key in ("cached_generation", "cached_denoise_callback")
            }
            pair[str(seed)]["same_seed_outputs_byte_identical"] = (
                a["video_sha256"][0] == b["video_sha256"][0]
            )
        pair["both_seeds_complete_request_benefit"] = all(
            pair[str(s)]["cached_generation"]["median_seconds_saved"] > 0 for s in (42, 43)
        )
        pair["neither_seed_denoise_regresses"] = all(
            pair[str(s)]["cached_denoise_callback"]["median_seconds_saved"] >= 0 for s in (42, 43)
        )
        pairs[left + "__" + right] = pair
    return dict(
        source_pin_sha256=reference["producer"]["source_pin_sha256"],
        source_sha256=reference["producer"]["source_sha256"],
        model=reference["producer"]["model"],
        environment=reference["environment"],
        arms=records,
        comparisons=pairs,
        compared_policies=policies,
        execution_order=[dict(seed=r["seed"], policy=r["policy"]) for r in ordered],
        limitations=[
            "Two separate seeds/processes, one warmup and three timed requests per policy; descriptive medians, not confidence intervals.",
            "Complete requests include decode/export and repeated audio VAE setup; not compilation-free end-to-end.",
            "Nonoverlap and shared lock do not rule out unrelated external device load.",
            "Cross-policy byte identity is descriptive only; SIMD RMS may legitimately change BF16 rounding.",
            "Performance alone never promotes a default; independent numerical gates and explicit aggregate/tail quality review remain required.",
        ],
    )


def execute(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema=SCHEMA,
        status="failed",
        stable_promotion="not_evaluated",
        evaluator_sha256=base.sha256(__file__),
    )
    try:
        records = [
            validate_run(directory, args.source_manifest_sha256) for directory in args.run_dir
        ]
        result.update(compare_records(records, args.policies))
        for directory, record in zip(args.run_dir, records, strict=True):
            contract.exact(
                base.sha256(directory / "run.json"), record["run_sha256"], "inputs immutable"
            )
        result["status"] = "completed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    base.write_json(output / "resident-comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--policies", nargs="+", choices=contract.POLICIES)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = execute(parser.parse_args())
    print(json.dumps({key: result.get(key) for key in ("status", "error")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)

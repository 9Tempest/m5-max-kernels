"""Export selected immutable resident performance evidence without local paths."""

import argparse
import json
import math
from pathlib import Path
import statistics

import publish_samples as public
import run as base


KEYS = (
    "benchmark",
    "prompt",
    "mode",
    "fusion",
    "seed",
    "width",
    "height",
    "steps",
    "source_frames",
    "warmups",
    "repetitions",
    "active_dit_blocks",
    "reuse",
    "core_reuse",
    "use_reference_rope",
    "timing_contract",
    "harness_files",
    "upstream",
    "patch",
    "driver",
    "shape",
    "warmup_seconds",
    "cached_generation",
    "cached_denoise_callback",
    "all_repetition_videos_byte_identical",
    "started_utc",
    "finished_utc",
)


def validate_evidence(path, record):
    events_path = path.parent / "session/session.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    if events != record.get("session_events"):
        raise ValueError("resident session events differ from completed run")
    digest = base.sha256(events_path)
    if record.get("session_events_sha256", digest) != digest:
        raise ValueError("resident session event checksum differs")
    recipes = [event for event in events if event.get("event") == "recipe"]
    if len(recipes) != 1 or events[-1] != {"event": "finished", "exit_code": 0}:
        raise ValueError("resident recipe or successful completion event missing")
    recipe = recipes[0]
    for key in (
        "mode",
        "fusion",
        "prompt",
        "seed",
        "width",
        "height",
        "steps",
        "warmups",
        "repetitions",
        "reuse",
        "core_reuse",
        "use_reference_rope",
    ):
        if recipe.get(key) != record[key]:
            raise ValueError(f"resident event recipe differs: {key}")
    if (
        recipe.get("frames") != record["source_frames"]
        or recipe.get("layers") != record["active_dit_blocks"]
    ):
        raise ValueError("resident frame/layer recipe differs")
    generations = [event for event in events if event.get("event") == "generation"]
    count = record["warmups"] + record["repetitions"]
    if (
        len(generations) != count
        or len(record["videos"]) != count
        or len(record["profiles"]) != count
    ):
        raise ValueError("resident generation/video/profile count differs")
    for index, (event, video, profile) in enumerate(
        zip(generations, record["videos"], record["profiles"], strict=True)
    ):
        warmup = index < record["warmups"]
        if (
            event.get("index") != index
            or event.get("warmup") is not warmup
            or event.get("status") != "completed"
        ):
            raise ValueError("resident generation order/status differs")
        wall, denoise = event.get("wall_seconds"), event.get("denoise_callback_seconds")
        if (
            any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (wall, denoise)
            )
            or not 0 < denoise <= wall
        ):
            raise ValueError("resident callback must lie within positive request wall")
        if not warmup:
            offset = index - record["warmups"]
            if (
                wall != record["cached_generation"]["raw_seconds"][offset]
                or denoise != record["cached_denoise_callback"]["raw_seconds"][offset]
            ):
                raise ValueError("resident raw timings differ from original generation events")
            if any(
                not event.get("cache_before", {}).get(key)
                for key in ("prepared_dit", "video_decoder", "embedding_entries")
            ):
                raise ValueError("resident timed request does not have all caches populated")
        if "requested_seed" in event and (
            event["requested_seed"] != record["seed"] or event.get("result_seed") != record["seed"]
        ):
            raise ValueError("resident result seed differs from recipe")
        video_name, log_name = event.get("output"), event.get("profile_log")
        if not all(
            isinstance(name, str) and Path(name).name == name for name in (video_name, log_name)
        ):
            raise ValueError("unsafe resident evidence filename")
        if (
            video.get("file") != "session/" + video_name
            or base.sha256(path.parent / "session" / video_name) != video["sha256"]
        ):
            raise ValueError("resident video differs from immutable source record")
        if (
            profile.get("index") != index
            or base.sha256(path.parent / "session" / log_name) != profile["log_sha256"]
        ):
            raise ValueError("resident profile log differs from immutable source record")
    if record["warmup_seconds"] != generations[0]["wall_seconds"]:
        raise ValueError("resident warmup time differs from original event")
    identical = len({video["sha256"] for video in record["videos"]}) == 1
    if record["all_repetition_videos_byte_identical"] is not identical:
        raise ValueError("resident video identity claim differs from actual digests")
    # Only the device model and capacity are public; never registry/serial IDs.
    public_events = [
        {key: event[key] for key in ("event", "name", "physical_memory_bytes") if key in event}
        if event.get("event") == "device"
        else event
        for event in events
    ]
    return public_events, digest


def summarize(path):
    record = public.load_json(path)
    if (
        record.get("status") != "completed"
        or type(record.get("warmups")) is not int
        or record["warmups"] != 1
    ):
        raise ValueError("resident measurement must complete a full warmup")
    for key in ("cached_generation", "cached_denoise_callback"):
        metric = record[key]
        times = metric["raw_seconds"]
        if len(times) < 3 or len(times) != record["repetitions"] or metric["count"] != len(times):
            raise ValueError("resident measurement needs at least three complete repetitions")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in times):
            raise ValueError("invalid resident timing")
        if metric["median_seconds"] != statistics.median(times):
            raise ValueError("resident median differs from raw times")
    events, event_digest = validate_evidence(path, record)
    result = {key: record[key] for key in KEYS}
    result.update(
        schema_version=2,
        status="completed",
        source_run_sha256=base.sha256(path),
        exporter_sha256=base.sha256(__file__),
        session_events=events,
        session_events_sha256=event_digest,
        profiles=record["profiles"],
        profile_interpretation="Native cached DiT profile wall spans time since the previous denoise mark and can include the previous decode; use cached_denoise_callback for request-local synchronized host wall. root_gpu_s is not total GPU time. These are host intervals, not kernel-only or compilation-free end-to-end latency.",
    )
    result["model"] = public.model_identity(record, path.parent)
    for key in ("runtime_overrides", "observed_runtime_flags", "policy_evidence"):
        if key in record:
            result[key] = record[key]
    result["harness"] = {key: record["harness"][key] for key in ("revision", "dirty")}
    result["environment"] = {
        key: value
        for key, value in record["environment"].items()
        if key
        in ("system", "release", "machine", "python", "chip", "memory_bytes", "xcode", "ffprobe")
    }
    result["video_sha256"] = [entry["sha256"] for entry in record["videos"]]
    public.public_text_check(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base.write_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

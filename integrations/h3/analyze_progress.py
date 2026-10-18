"""Describe unmodified H3 denoise progress intervals from immutable pipe-receipt evidence.

These are first-step-excluded host wall estimates, not compilation-free or GPU
kernel timings. No model execution or profiling occurs in this analyzer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import run as h3


def strict_json(data):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    value = json.loads(data, parse_constant=reject)
    json.dumps(value, allow_nan=False)
    return value


def receipt_blocks(data, log_size):
    blocks, offset, previous_time = [], 0, -1.0
    for line in data.splitlines():
        event = strict_json(line)
        if not isinstance(event, dict):
            raise ValueError("receipt event must be an object")
        start, count, elapsed = (
            event.get("byte_offset"),
            event.get("byte_count"),
            event.get("elapsed_seconds"),
        )
        if type(start) is not int or type(count) is not int or count <= 0 or start != offset:
            raise ValueError("receipt byte ranges must be positive, contiguous, and start at zero")
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed < previous_time
        ):
            raise ValueError("receipt times must be finite, nonnegative, and nondecreasing")
        if elapsed < 0:
            raise ValueError("receipt times must be finite, nonnegative, and nondecreasing")
        blocks.append(dict(index=len(blocks), start=start, end=start + count, elapsed=elapsed))
        offset, previous_time = start + count, elapsed
    if not blocks or offset != log_size:
        raise ValueError("receipt byte ranges must cover the complete generation log exactly")
    return blocks


def analyze(directory):
    directory = Path(directory)
    names = ("run.json", "generation.log", "generation.events.jsonl")
    files = {name: (directory / name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    run = strict_json(files["run.json"])
    if not isinstance(run, dict) or run.get("status") != "completed":
        raise ValueError("analysis requires a completed run")
    measurements = run.get("measurements")
    generation = measurements.get("generation") if isinstance(measurements, dict) else None
    if not isinstance(generation, dict):
        raise ValueError("generation process evidence must be an object")
    if (
        generation.get("log") != "generation.log"
        or generation.get("log_sha256") != hashes["generation.log"]
        or generation.get("returncode") != 0
        or generation.get("fallback_detected") is not False
    ):
        raise ValueError("generation log checksum or successful-process evidence does not match")
    upstream = run.get("upstream", {})
    if (
        not isinstance(upstream, dict)
        or upstream.get("revision") != h3.UPSTREAM_REVISION
        or upstream.get("dirty") is not False
        or run.get("diagnostics", False) is not False
        or run.get("runtime_overrides") != {"H3_NAX": "qkv-attn", "H3_NAX_DIAGNOSTIC": "1"}
        or run.get("reuse") != 1
        or run.get("core_reuse") != 1
        or run.get("active_dit_blocks") != 50
        or run.get("token_reduction") is not False
    ):
        raise ValueError("requires the pinned unmodified, full-model, default-window recipe")
    steps = run.get("steps")
    if type(steps) is not int or not 2 <= steps <= 1000:
        raise ValueError("invalid denoising step count")
    log = files["generation.log"]
    if b"h3: GPU sampler encode window is bounded; internal split disabled" not in log:
        raise ValueError("log does not confirm the default synchronized GPU sampler window")
    profiles = h3.parse_profile(log.decode("utf-8", errors="replace"))
    if (
        sum(
            item.get("component") == "H3 DiT" and item.get("phase") == "GPU Euler denoise"
            for item in profiles
        )
        != 1
    ):
        raise ValueError("log must contain exactly one GPU Euler denoise phase profile")
    blocks = receipt_blocks(files["generation.events.jsonl"], len(log))
    matches = list(
        re.finditer(rb"(?:^|[\r\n])denoise enqueue[ \t]+([0-9]+)/([0-9]+)(?=[ \t\r\n]|$)", log)
    )
    if [int(match[1]) for match in matches] != list(range(steps + 1)) or any(
        int(match[2]) != steps for match in matches
    ):
        raise ValueError("expected exactly one ordered denoise enqueue boundary for 0..steps")
    boundaries = []
    block_index = 0
    for match in matches:
        # Use the block containing the final numeric byte; UTF-8 decoding never changes offsets.
        end = match.end(2)
        while blocks[block_index]["end"] < end:
            block_index += 1
        block = blocks[block_index]
        boundaries.append(
            dict(
                completed=int(match[1]),
                total=steps,
                numeric_end_byte=end,
                receipt_block=block_index,
                receipt_elapsed_seconds=block["elapsed"],
                spans_receipt_blocks=match.start(1) < block["start"],
            )
        )
    counts = Counter(boundary["receipt_block"] for boundary in boundaries)
    for boundary in boundaries:
        boundary["coalesced"] = counts[boundary["receipt_block"]] > 1
    intervals = []
    for step, (before, after) in enumerate(zip(boundaries, boundaries[1:])):
        delta = after["receipt_elapsed_seconds"] - before["receipt_elapsed_seconds"]
        ambiguous = before["coalesced"] or after["coalesced"] or delta <= 0
        intervals.append(
            dict(
                step=step,
                from_completed=step,
                to_completed=step + 1,
                receipt_delta_seconds=delta,
                timing_status="coalesced_ambiguous"
                if before["coalesced"] or after["coalesced"]
                else "unresolved_receipt_time"
                if delta <= 0
                else "receipt_estimate",
                excluded_first_step=step == 0,
                included_in_summary=step != 0 and not ambiguous,
            )
        )
    raw = [item["receipt_delta_seconds"] for item in intervals if item["included_in_summary"]]
    return dict(
        schema_version=1,
        status="ambiguous"
        if any(item["timing_status"] != "receipt_estimate" for item in intervals)
        else "completed",
        measurement="first-step-excluded denoising progress receipt intervals",
        source_directory=h3.path_label(directory),
        source_sha256=hashes,
        analyzer_sha256=h3.sha256(Path(__file__)),
        helper_sha256=h3.sha256(Path(h3.__file__)),
        recipe={
            key: run.get(key) for key in ("benchmark", "mode", "seed", "width", "height", "steps")
        },
        boundaries=boundaries,
        intervals=intervals,
        descriptive_summary=dict(
            expected_count=steps - 1,
            count=len(raw),
            raw_seconds=raw,
            median_seconds=statistics.median(raw) if raw else None,
            omitted_ambiguous_steps=[
                item["step"]
                for item in intervals
                if item["step"] != 0 and not item["included_in_summary"]
            ],
        ),
        profile_phases=profiles,
        limitations=[
            "Receipt timestamps are recorded after pipe read and local log flush; they are not producer or device timestamps.",
            "Differences include CPU encoding, synchronization, dispatch and observer scheduling/I/O delay; delay error is not bounded.",
            "Step 0 is excluded descriptively. Remaining steps are not proven compilation-free and are not independent repeated identical inputs.",
            "Coalesced boundaries and intervals touching them are ambiguous, excluded from the summary; zero receipt deltas are not zero execution time.",
            "Completed enqueue steps follow synchronization for this fixed recipe; the final latent readback/unpack follows enqueue completion and is excluded.",
            "Profile phase totals overlap. Root-GPU counters and receipt intervals are distinct; neither provides per-operation device attribution.",
        ],
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run_dir", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    try:
        result = analyze(args.run_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        h3.write_json(args.output, result)
    except (ValueError, OSError) as error:
        print(f"progress analysis rejected: {error}", file=sys.stderr)
        return 1
    print(json.dumps(dict(status=result["status"], output=h3.path_label(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay a .gputrace and export only workload counters, without machine paths."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from m5kernels.engine import write_result
from m5kernels.locking import gpu_lock


def objects(text):
    """gpudebug mixes startup messages, compact JSON and indented JSON."""
    decoder = json.JSONDecoder()
    pos = 0
    result = []
    while (pos := text.find("{", pos)) >= 0:
        value, length = decoder.raw_decode(text[pos:])
        result.append(value)
        pos += length
    return result


def call(*args):
    run = subprocess.run(
        ["xcrun", "gpudebug", *args, "--json"], capture_output=True, text=True, timeout=180
    )
    if run.returncode or "error:" in run.stdout.lower():
        raise RuntimeError(run.stderr + run.stdout)
    return run.stdout


def collect(trace, expected):
    started = call("-t", str(trace), "-c", "status", "--timeout", "300")
    session = re.search(r"Session (\d+) created", started)
    if not session:
        raise RuntimeError("No GPU debugger session was created")
    session = session[1]
    try:
        status = objects(started)[-1]
        count = int(re.search(r"(\d+) dispatches", status["summary"])[1])
        if expected is not None and count != expected:
            raise ValueError(f"Capture has {count} dispatches, expected {expected}")
        result = objects(call("-s", session, "-c", "profile run --gpu-state default --exec serial"))
        if not result[-1].get("success"):
            raise RuntimeError(result)
        timeline = objects(call("-s", session, "-c", "info /performance/timeline"))[-1]
        listing = objects(call("-s", session, "-c", "go /performance/shaders", "-c", "list --all"))[
            -1
        ]
        shaders = []
        for shader in listing["children"]:
            detail = objects(
                call("-s", session, "-c", f"info /performance/shaders/{shader['name']}")
            )[-1]
            shaders.append(dict(function=shader["values"][0]["value"], counters=detail))
        call("-s", session, "-c", "profile embed")
        return dict(
            schema_version=1,
            trace_basename=trace.name,
            workload=json.loads(Path(str(trace) + ".json").read_text()),
            device=status["device"],
            capture_summary=status["summary"],
            profiling_overhead=True,
            timing_scope="serial GPU replay, default GPU state; not benchmark wall latency",
            timeline=timeline,
            shaders=shaders,
        )
    finally:
        call("--terminate", session)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--expect-dispatches", type=int)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with gpu_lock():
        write_result(args.output, collect(args.trace, args.expect_dispatches))


if __name__ == "__main__":
    main()

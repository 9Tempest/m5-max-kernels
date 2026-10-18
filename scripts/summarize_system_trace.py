"""Summarize target-only GPU intervals exported by xctrace; publish no process IDs."""

import argparse
from collections import defaultdict
import xml.etree.ElementTree as ET

from m5kernels.engine import write_result


def union_ns(intervals):
    end = -1
    total = 0
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def summarize(path, pid):
    root = ET.parse(path).getroot()
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}

    def resolve(e):
        return ids[e.get("ref")] if e.get("ref") else e

    groups = defaultdict(list)
    for node in root.iter("node"):
        schema = node.find("schema")
        if schema is None or schema.get("name") != "metal-gpu-intervals":
            continue
        columns = [col.findtext("mnemonic") for col in schema.findall("col")]
        for row in node.findall("row"):
            values = dict(zip(columns, map(resolve, row), strict=True))
            process = values["process"]
            if process.findtext("pid") != str(pid):
                continue
            start = int(values["start"].text)
            duration = int(values["duration"].text)
            # Channel names are a fixed GPU category, never process/event labels.
            channel = values["channel-name"].get("fmt", "unknown")
            if channel not in {"Compute", "Blit", "Render", "Vertex", "Fragment"}:
                channel = "other"
            groups[channel].append((start, start + duration))
    if not groups:
        raise ValueError("No GPU intervals for the selected target PID")
    all_intervals = [x for values in groups.values() for x in values]
    span = max(b for a, b in all_intervals) - min(a for a, b in all_intervals)
    active = union_ns(all_intervals)
    return dict(
        schema_version=1,
        source="xctrace Metal System Trace / metal-gpu-intervals",
        scope="target process, entire launch including setup/warmup; duplicate/nested intervals unioned",
        profiling_overhead=True,
        interval_count=len(all_intervals),
        observed_span_ms=span / 1e6,
        gpu_active_union_ms=active / 1e6,
        observed_busy_fraction=active / span if span else 0,
        channels={
            name: dict(interval_count=len(v), gpu_active_union_ms=union_ns(v) / 1e6)
            for name, v in groups.items()
        },
        warning="Not a per-kernel latency or hot-loop-only utilization measurement",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xml")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    write_result(args.output, summarize(args.xml, args.pid))


if __name__ == "__main__":
    main()

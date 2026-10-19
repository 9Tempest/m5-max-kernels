"""Describe verified resident arms without claiming complete promotion evidence."""

import argparse
import itertools
import json
from pathlib import Path

import compare_kernel_resident as comparison
import kernel_integration as contract
import run as base


def summarize(records):
    if len(records) < 2:
        raise ValueError("at least two verified resident arms are required")
    indexed = {(r["seed"], r["policy"]): r for r in records}
    if len(indexed) != len(records):
        raise ValueError("duplicate resident arm")
    reference = records[0]
    for item in records[1:]:
        for key in ("environment", "compilation", "timing_contract", "shape"):
            contract.exact(item[key], reference[key], "matched " + key)
        for key in ("source_pin_sha256", "source_sha256", "model", "harness"):
            contract.exact(item["producer"][key], reference["producer"][key], "matched " + key)
    ordered = sorted(records, key=lambda r: comparison.prior.timestamp(r["process"]["started_utc"]))
    for a, b in zip(ordered, ordered[1:]):
        if comparison.prior.timestamp(a["process"]["finished_utc"]) > comparison.prior.timestamp(
            b["process"]["started_utc"]
        ):
            raise ValueError("resident GPU intervals overlap")
    pairs = []
    for seed in sorted({r["seed"] for r in records}):
        policies = [p for p in contract.POLICIES if (seed, p) in indexed]
        for left, right in itertools.combinations(policies, 2):
            a, b = indexed[seed, left], indexed[seed, right]
            pairs.append(
                dict(
                    seed=seed,
                    baseline=left,
                    candidate=right,
                    metrics={
                        key: comparison.prior.metric_pair(
                            a["timing"][key]["raw_seconds"], b["timing"][key]["raw_seconds"]
                        )
                        for key in ("cached_generation", "cached_denoise_callback")
                    },
                )
            )
    if not pairs:
        raise ValueError("no same-seed comparison exists")
    return dict(
        arms=records,
        pairs=pairs,
        promotion_eligible=False,
        scope="Descriptive matched resident samples; no complete counterbalanced promotion campaign is asserted.",
        limitations=[
            "Three timed requests after one full warmup per arm; complete requests include repeated audio VAE setup.",
            "Within-session medians do not establish significance, independent replication or perceptual quality.",
            "Default promotion requires the separate strict two-seed evidence and review workflow.",
        ],
    )


def execute(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema="h3_kernel_resident_descriptive_v1",
        status="failed",
        promotion_eligible=False,
        evaluator_sha256=base.sha256(__file__),
    )
    try:
        records = [comparison.validate_run(p, args.source_manifest_sha256) for p in args.run_dir]
        report.update(summarize(records), status="completed")
        for directory, record in zip(args.run_dir, records, strict=True):
            contract.exact(
                base.sha256(directory / "run.json"), record["run_sha256"], "input immutable"
            )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    base.write_json(args.output_dir / "resident-summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = execute(parser.parse_args())
    print(json.dumps({key: result.get(key) for key in ("status", "error")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)

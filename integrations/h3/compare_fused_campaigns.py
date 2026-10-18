"""Compare two independently recorded, pinned native fusion campaigns on CPU."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

import publish_fused_campaign as publisher
import run as base


def compare(first, second):
    runs = [publisher.collect(directory)[0] for directory in (first, second)]
    if runs[0]["source_campaign_sha256"] == runs[1]["source_campaign_sha256"]:
        raise ValueError("an independent rerun cannot reuse the same campaign record")
    raw = [publisher.read_public(directory / "campaign.json") for directory in (first, second)]
    if datetime.fromisoformat(raw[0]["finished_utc"]) > datetime.fromisoformat(
        raw[1]["started_utc"]
    ):
        raise ValueError("campaign intervals overlap or are out of order")
    for key in (
        "environment",
        "model",
        "patch",
        "wrapper_sha256",
        "warmups_per_variant",
        "repeats_per_variant",
    ):
        if runs[0][key] != runs[1][key]:
            raise ValueError(f"paired campaign contract differs: {key}")
    for key in ("source_sha256", "reference_sha256", "binary_sha256"):
        if not all(
            re.fullmatch(r"[0-9a-f]{64}", run["driver"].get(key, "")) for run in runs
        ) or runs[0]["driver"].get(key) != runs[1]["driver"].get(key):
            raise ValueError(f"paired driver identity differs: {key}")
    if not all(
        re.fullmatch(r"[0-9a-f]{64}", run["driver"].get("library_sha256", "")) for run in runs
    ):
        raise ValueError("paired library archive identity missing")
    if raw[0].get("runtime_overrides") != raw[1].get("runtime_overrides"):
        raise ValueError("paired runtime environment overrides differ")
    cases = []
    for a, b in zip(runs[0]["cases"], runs[1]["cases"], strict=True):
        for key in (
            "directory",
            "seed",
            "rows",
            "padded_rows",
            "policy",
            "K",
            "N",
            "weight_key",
            "timing_recipe",
        ):
            if a[key] != b[key]:
                raise ValueError(f"paired case contract differs: {key}")
        observations = []
        for run, case in zip(runs, (a, b), strict=True):
            timing = case["timing"]
            native = timing["native"]["wall_median_seconds"]
            fused = timing["fused"]["wall_median_seconds"]
            observations.append(
                dict(
                    campaign_id=run["campaign_id"],
                    driver_report_sha256=case["driver_report_sha256"],
                    case_report_sha256=case["case_report_sha256"],
                    native_median_seconds=native,
                    fused_median_seconds=fused,
                    native_over_fused_ratio=native / fused,
                    fused_wall_reduction_percent=100 * (1 - fused / native),
                    native_raw=timing["native"]["raw"],
                    fused_raw=timing["fused"]["raw"],
                )
            )
        ratios = [item["native_over_fused_ratio"] for item in observations]
        cases.append(
            {
                key: a[key]
                for key in (
                    "directory",
                    "seed",
                    "rows",
                    "padded_rows",
                    "policy",
                    "K",
                    "N",
                    "weight_key",
                )
            }
            | dict(
                observed_lower_median_in_both_campaigns=all(ratio > 1 for ratio in ratios),
                observed_higher_median_in_both_campaigns=all(ratio < 1 for ratio in ratios),
                numerical_contract="both campaigns pass independent references and exact paired hidden/codes/scales",
                reference_metrics_identical_between_campaigns=a["independent_reference"]
                == b["independent_reference"],
                observations=observations,
            )
        )
    groups = []
    for policy, rows in sorted({(case["policy"], case["rows"]) for case in cases}):
        selected = [case for case in cases if (case["policy"], case["rows"]) == (policy, rows)]
        observations = [item for case in selected for item in case["observations"]]
        ratios = [item["native_over_fused_ratio"] for item in observations]
        groups.append(
            dict(
                policy=policy,
                rows=rows,
                seeds=[case["seed"] for case in selected],
                cases_with_lower_median_in_both=sum(
                    case["observed_lower_median_in_both_campaigns"] for case in selected
                ),
                matched_seed_cases=len(selected),
                native_over_fused_ratio_min=min(ratios),
                native_over_fused_ratio_max=max(ratios),
                lower_median_all_seeds_both_campaigns=all(ratio > 1 for ratio in ratios),
                statistical_significance="not assessed; descriptive medians, not a significance test",
            )
        )
    result = dict(
        schema_version=1,
        status="completed",
        analyzer_sha256=base.sha256(__file__),
        scope="two serial native operator campaigns; real block-0 weights with synthetic activations; not a model speed measurement",
        campaigns=[
            dict(
                id=run["campaign_id"],
                source_campaign_sha256=run["source_campaign_sha256"],
                source_request_sha256=run["source_request_sha256"],
                started_utc=original["started_utc"],
                finished_utc=original["finished_utc"],
                driver=run["driver"],
                harness=run["harness"],
            )
            for run, original in zip(runs, raw, strict=True)
        ],
        shared_environment=runs[0]["environment"],
        shared_model=runs[0]["model"],
        shared_patch=runs[0]["patch"],
        shared_wrapper_sha256=runs[0]["wrapper_sha256"],
        warmups_per_variant=runs[0]["warmups_per_variant"],
        repeats_per_variant=runs[0]["repeats_per_variant"],
        library_archive_sha_equal=runs[0]["driver"]["library_sha256"]
        == runs[1]["driver"]["library_sha256"],
        linked_driver_binary_sha_equal=True,
        provenance_note="Library archive hashes are retained separately. The linked executable, source, reference and wrapper hashes must match; matching source alone is insufficient for this comparison.",
        numerical_cases_passed=2 * len(cases),
        groups=groups,
        cases=cases,
        limits=[
            "No statistical significance or model throughput claim",
            "No all-layer or real-timestep coverage",
            "Final block endpoint and both-flags buffer lifetimes require model checks",
            "No default enablement follows from this comparison",
        ],
    )
    publisher.public.public_text_check(result)
    return result


def readme(result):
    first, second = [run["id"] for run in result["campaigns"]]
    archive_comparison = "match" if result["library_archive_sha_equal"] else "differ"
    lines = [
        "# Independent native fusion campaign comparison",
        "",
        f"This joins `{first}` and `{second}` after revalidating all {result['numerical_cases_passed']} raw case/driver reports. Both campaigns cover two seeds (17/29), logical rows 2049/9914, and H256 FC2, group1024 FC2 and attention output. All independent numerical and exact paired output contracts pass.",
        "",
        "[Machine-readable comparison, raw timing samples and source hashes](comparison.json). The original published campaign JSON is unchanged. Each ratio is native wall median / fused wall median; a ratio above one means a lower fused median in that observation, not a statistical significance result.",
        "",
        "| Policy | Rows | Seed | First native / fused ms | First ratio | Second native / fused ms | Second ratio | Lower fused median in both? |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in result["cases"]:
        a, b = case["observations"]
        lines.append(
            f"| {case['policy']} | {case['rows']} | {case['seed']} | {a['native_median_seconds'] * 1000:.4f} / {a['fused_median_seconds'] * 1000:.4f} | {a['native_over_fused_ratio']:.5f} | {b['native_median_seconds'] * 1000:.4f} / {b['fused_median_seconds'] * 1000:.4f} | {b['native_over_fused_ratio']:.5f} | {'yes' if case['observed_lower_median_in_both_campaigns'] else 'no'} |"
        )
    lines += [
        "",
        f"The recorded device/software environment, model identity, runtime patch, driver/reference sources, wrapper and linked executable match. Library archive hashes {archive_comparison} and remain visible; the comparison requires byte-identical linked executable hashes. Each path has the same warmup/repetition counts and alternating order. The campaign intervals do not overlap.",
        "",
        "CPU encode, submit/wait and supplemental command-buffer GPU timings remain separate in every raw sample. Timing includes online activation packing/rotation, native layout/padding, GEMM/gate and next AdaLN/packing. Residual restoration, correctness readback, prior compilation and offline weight preparation are excluded. Both paths have three timed dispatches.",
        "",
        "Read gains at the matched policy and shape. A group1024 FC2 gain is not evidence for H256 FC2, and two block-0 operator campaigns do not validate real timestep maps, all 50 blocks, final endpoints or complete generation performance. Whole-model exact parity and matched resident requests remain separate evidence.",
        "",
    ]
    return "\n".join(lines)


def execute(first, second, output):
    result = compare(first, second)
    output.mkdir(parents=True, exist_ok=False)
    base.write_json(output / "comparison.json", result)
    with (output / "README.md").open("x") as handle:
        handle.write(readme(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.first_dir, args.second_dir, args.output_dir)
    print(
        json.dumps(
            dict(status=result["status"], numerical_cases_passed=result["numerical_cases_passed"])
        )
    )

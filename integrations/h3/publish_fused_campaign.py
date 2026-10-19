"""CPU-only export of pinned fusion correctness and raw paired timing evidence."""

import argparse
import json
from pathlib import Path
import re

import publish_samples as public
import run as base
import run_fused_campaign as fused


def read_public(path):
    value = public.load_json(path)
    public.public_text_check(value)
    serialized = json.dumps(value)
    if re.search(r"(?i)(registry_id|serial_number|thread_id|api_key|access_token)", serialized):
        raise ValueError("raw fusion JSON contains a private identifier or credential field")
    if re.search(r"(?i)[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", serialized):
        raise ValueError("raw fusion JSON contains an unreviewed unique identifier")
    return value


def collect(directory):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", directory.name):
        raise ValueError("campaign directory must have a plain publication identifier")
    campaign = read_public(directory / "campaign.json")
    request = read_public(directory / "request.json")
    expected_patch = dict(source_sha256=fused.SOURCE_SHA256, patch_sha256=fused.PATCH_SHA256)
    if campaign.get("status") != "completed" or campaign.get("patch") != expected_patch:
        raise ValueError("only a completed pinned fusion campaign can be exported")
    if (
        campaign["upstream"].get("revision") != base.UPSTREAM_REVISION
        or campaign["upstream"].get("source_sha256") != fused.SOURCE_SHA256
        or campaign["driver"].get("source_sha256") != fused.DRIVER_SHA256
        or campaign["driver"].get("reference_sha256") != fused.REFERENCE_SHA256
    ):
        raise ValueError("fusion runtime/driver/reference source identity differs")
    for key in ("patch", "upstream", "driver", "model", "wrapper_sha256"):
        if request.get(key) != campaign.get(key):
            raise ValueError(f"fusion request/campaign identity differs: {key}")
    if request.get("cases_requested") != [list(case) for case in fused.CASES]:
        raise ValueError("fusion request does not cover the twelve frozen cases")
    warmups, repeats = request["warmups"], request["repeats"]
    if type(warmups) is not int or type(repeats) is not int or warmups < 2 or repeats < 3:
        raise ValueError("fusion campaign needs warmups and repeated paired timings")
    cases = campaign.get("cases", [])
    if [(case.get("seed"), case.get("rows"), case.get("policy")) for case in cases] != list(
        fused.CASES
    ):
        raise ValueError("fusion campaign is missing, duplicating or reordering cases")
    paths = [Path("campaign.json"), Path("request.json")]
    summaries = []
    for embedded in cases:
        seed, rows, policy = embedded["seed"], embedded["rows"], embedded["policy"]
        name = f"{policy}-m{rows}-seed{seed}"
        if embedded.get("directory") != name:
            raise ValueError("fusion case directory differs from recipe")
        case_path, driver_path = Path(name) / "case.json", Path(name) / "driver.json"
        case, report = read_public(directory / case_path), read_public(directory / driver_path)
        if embedded != case | {"directory": name} or case.get("status") != "completed":
            raise ValueError("fusion raw case differs from completed campaign")
        if case.get("driver_report_sha256") != base.sha256(directory / driver_path):
            raise ValueError("fusion raw driver checksum differs")
        if (
            case["process"].get("returncode") != 0
            or case["process"].get("fallback_detected") is not False
        ):
            raise ValueError("fusion process failed or used a fallback")
        summary = fused.validate_case(report, seed, rows, policy, warmups, repeats)
        if summary != case.get("summary"):
            raise ValueError("fusion stored summary differs from validated raw timings")
        paths.extend((case_path, driver_path))
        summaries.append(
            dict(
                directory=name,
                seed=seed,
                rows=rows,
                padded_rows=report["padded_rows"],
                policy=policy,
                K=report["K"],
                N=report["N"],
                weight_key=report["weight_key"],
                numerical_counters={key: report[key] for key in fused.COUNTERS},
                independent_reference=report["independent_reference"],
                timing=summary,
                timing_recipe={
                    key: report["timing"][key]
                    for key in (
                        "warmups_per_variant",
                        "repeats_per_variant",
                        "paired_order",
                        "first_measured_variant",
                    )
                },
                driver_report_sha256=case["driver_report_sha256"],
                case_report_sha256=base.sha256(directory / case_path),
            )
        )
    result = dict(
        schema_version=2,
        status="completed",
        campaign_id=directory.name,
        source_campaign_sha256=base.sha256(directory / "campaign.json"),
        source_request_sha256=base.sha256(directory / "request.json"),
        exporter_sha256=base.sha256(__file__),
        created_utc=base.utc_now(),
        scope="one campaign of real block-0 weights with synthetic activations; paired online operator paths",
        conclusions=dict(
            independent_numerical_contract=f"passed all {len(summaries)} cases",
            independent_rerun="not assessed by this single-campaign export",
            whole_model_parity="not assessed by this operator-only export",
            end_to_end_acceleration="not measured by this operator-only export",
            default_enablement=False,
        ),
        timing_contract=dict(
            primary="synchronized host wall; warmed alternating-order repetitions; counts recorded separately",
            cpu_encode="command begin through encode completion",
            submit_wait="submission through command-buffer completion",
            supplemental_gpu="command-buffer duration; not individual shader attribution",
            included="online activation packing/rotation, projection/gate, next AdaLN/INT8 packing; native layout and padding",
            excluded="loading, offline weight preparation, prior pipeline compile/warmup, correctness readback and CPU residual restore",
            dispatches="three per path: activation packing, GEMM and next AdaLN/packing; residual restore occurs before timing",
        ),
        warmups_per_variant=warmups,
        repeats_per_variant=repeats,
        environment=campaign["environment"],
        upstream=campaign["upstream"],
        patch=campaign["patch"],
        driver=campaign["driver"],
        harness=campaign["harness"],
        wrapper_sha256=campaign["wrapper_sha256"],
        model={
            key: campaign["model"][key] for key in ("repository", "revision", "manifest_sha256")
        },
        cases=summaries,
        raw_files=[
            dict(
                file="raw/" + path.as_posix(),
                sha256=base.sha256(directory / path),
                bytes=(directory / path).stat().st_size,
            )
            for path in paths
        ],
    )
    public.public_text_check(result)
    return result, paths


def readme(result):
    lines = [
        f"# Native gate epilogue {result['campaign_id']}",
        "",
        f"All {len(result['cases'])} cases in this package pass independent full-K selected-row numerical checks, full-output finite/zero/tail checks, and exact paired hidden/codes/scales checks. This is one campaign using real block-0 weights and synthetic activations, including zero and sparse-outlier rows. This package does not assess independent reruns, full-model trajectory parity or matched resident performance; those require separate evidence. Neither flag is enabled by default by this experiment.",
        "",
        "[Summary and complete raw timing samples](summary.json). Every linked JSON in `raw/` is copied byte-for-byte, with checksums in the summary. No weights, activation arrays, executables or private logs are included.",
        "",
        "| Policy | Logical / padded rows | Seed | Native median ms | Fused median ms | Native / fused |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        timing = case["timing"]
        lines.append(
            f"| {case['policy']} | {case['rows']} / {case['padded_rows']} | {case['seed']} | {timing['native']['wall_median_seconds'] * 1000:.4f} | {timing['fused']['wall_median_seconds'] * 1000:.4f} | {timing['observed_native_over_fused_wall_ratio']:.5f} |"
        )
    lines += ["", "Observed ratio ranges within this campaign (native / fused):", ""]
    for policy in sorted({case["policy"] for case in result["cases"]}):
        ratios = [
            case["timing"]["observed_native_over_fused_wall_ratio"]
            for case in result["cases"]
            if case["policy"] == policy
        ]
        lines.append(
            f"- `{policy}`: {min(ratios):.5f}–{max(ratios):.5f}; fused median lower in {sum(ratio > 1 for ratio in ratios)}/{len(ratios)} cases, higher in {sum(ratio < 1 for ratio in ratios)}/{len(ratios)}, exactly equal in {sum(ratio == 1 for ratio in ratios)}/{len(ratios)}."
        )
    first_variants = sorted(
        {
            (case["seed"], case["timing_recipe"]["first_measured_variant"])
            for case in result["cases"]
        }
    )
    ordering = "; ".join(f"seed {seed}: {variant} first" for seed, variant in first_variants)
    lines += [
        "",
        "These are operator-path observations from one campaign, not an end-to-end speedup or a selected backend. Raw samples retain timing variation; independent reruns and complete model checks are needed before adoption.",
        "",
        f"Each variant has {result['warmups_per_variant']} warmups and {result['repeats_per_variant']} alternating-order measured repetitions. First measured variants from the driver records: {ordering}. Wall, CPU encode, submit/wait and supplemental command-buffer GPU intervals are retained separately. CPU residual restoration is outside timing. Both paths contain three GPU dispatches: activation quantization/rotation, projection, and next AdaLN/quantization. The native backend already fuses gate/AdaLN/quantization, so moving the gate to GEMM does not remove a dispatch in this measured chain.",
        "",
        "The independent reference uses all K terms and all 5,376 channels for rows 0/middle/last. Quantization drift from original BF16 operands is separate from decoded-operand implementation error. Full-output exact paired checks cover all rows, but this does not validate all 50 blocks, real timestep maps, both flags together or the final block endpoint. See the [integration contract](../../../../integrations/h3/fused.md).",
        "",
    ]
    return "\n".join(lines)


def execute(directory, output):
    result, paths = collect(directory)
    output.mkdir(parents=True, exist_ok=False)
    for path, entry in zip(paths, result["raw_files"], strict=True):
        target = output / entry["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        public.copy_exclusive(directory / path, target)
        if base.sha256(target) != entry["sha256"]:
            raise ValueError("fusion source changed after validation; partial output retained")
    base.write_json(output / "summary.json", result)
    with (output / "README.md").open("x") as handle:
        handle.write(readme(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.campaign_dir, args.output_dir)
    print(json.dumps(dict(status=result["status"], cases=len(result["cases"]))))

"""CPU source/record/locking contracts; these tests never initialize Metal."""

from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest


@pytest.fixture
def campaign(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    for name in ("run", "run_fc2_experiment"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "h3_fused_campaign_test", directory / "run_fused_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(count):
    return dict(count=count, nonfinite=0, relative_l2=0.0, rmse=0.0, max_abs=0.0)


def valid_report(campaign, seed=17, rows=9914, policy="h256", warmups=3, repeats=3):
    report = dict(
        schema=2,
        policy=policy,
        rows=rows,
        padded_rows=(rows + 127) & ~127,
        seed=seed,
        K=7168 if policy == "attention" else 14336,
        N=5376,
        in_place_residual=True,
        gate_slot=2 if policy == "attention" else 5,
        distinct_next_block_modulation=policy != "attention",
        weight_key=f"blocks.0.{'attn.out_proj' if policy == 'attention' else 'mlp.fc2'}.weight",
        scope="real_weights_synthetic_activations_paired_online",
        passed=True,
    )
    report.update({name: 0 for name in campaign.COUNTERS})
    ref = dict(
        rows=[0, rows // 2, rows - 1],
        columns=5376,
        full_K=report["K"],
        relative_l2_implementation_limit=0.008,
        reference_failures=0,
        passed=True,
    )
    for key in (
        "projection_implementation_vs_rounded_decoded",
        "hidden_implementation",
        "next_quantized_implementation",
        "projection_quantization_vs_original_bf16",
        "hidden_quantization_vs_original_bf16_path",
        "next_quantization_vs_original_bf16_path",
    ):
        ref[key] = metric(3 * 5376)
    ref["per_row"] = [
        dict(
            row=r,
            **{
                key: metric(5376)
                for key in (
                    "projection_implementation",
                    "projection_quantization",
                    "hidden_implementation",
                    "next_quantized_implementation",
                )
            },
        )
        for r in ref["rows"]
    ]
    report["independent_reference"] = ref
    sample = dict(
        wall_seconds=0.01,
        cpu_encode_seconds=0.001,
        submit_wait_seconds=0.009,
        supplemental_gpu_seconds=0.008,
        direct_dispatches=3,
    )
    report["timing"] = dict(
        status="completed",
        warmups_per_variant=warmups,
        repeats_per_variant=repeats,
        scope="online_activation_pack_projection_gate_next_adaln_quant_submit_wait",
        paired_order="alternates_native_first_fused_first",
        first_measured_variant="fused" if (warmups + bool(seed & 4)) & 1 else "native",
        excluded=[
            "loading",
            "offline_weight_preparation",
            "shader_pipeline_compile_and_warmup",
            "correctness_readback",
            "residual_restore",
        ],
        native=[copy.deepcopy(sample) for _ in range(repeats)],
        fused=[copy.deepcopy(sample) for _ in range(repeats)],
    )
    return report


def test_new_shader_dispatches_are_registered_in_tensorops_branch():
    patch = (Path(__file__).parents[1] / "integrations/h3/fused.patch").read_text()
    host = patch.split("diff --git a/h3_gpu.m b/h3_gpu.m", 1)[1].split("diff --git", 1)[0]
    shaders = patch.split("diff --git a/h3_shaders.metal b/h3_shaders.metal", 1)[1]
    declared = set(re.findall(r"^\+kernel void (h3_\w+(?:_gate|_pre_gated))\(", shaders, re.M))
    registered = set(re.findall(r'^\+\s*\[names addObject:@"(h3_\w+)"\];', host, re.M))
    required = {
        "h3_linear_int8_nax_r128_full_k14336_gate",
        "h3_linear_int8_grouped_local_nax_r128x64_gate",
        "h3_linear_int8_local_scales_nax_r128_gate",
        "h3_adaln_quantize_int8_pre_gated",
    }
    assert declared == required
    assert required <= registered
    registry = host.split(" if (gpu.tensorOpsEnabled) {", 1)[1].split("@@", 1)[0]
    assert all(f'[names addObject:@"{name}"];' in registry for name in required)


def test_frozen_driver_header_and_patch_hashes(campaign):
    directory = Path(campaign.__file__).parent
    for name, digest in (
        ("fused.patch", campaign.PATCH_SHA256),
        ("fused_validate.c", campaign.DRIVER_SHA256),
        ("fused_reference.h", campaign.REFERENCE_SHA256),
    ):
        assert campaign.base.sha256(directory / name) == digest


def test_runtime_provenance_rejects_extra_changed_files(campaign, monkeypatch, tmp_path):
    source = tmp_path / "runtime"
    provenance = dict(
        revision=campaign.base.UPSTREAM_REVISION,
        source_sha256=campaign.SOURCE_SHA256,
        status=" M h3_dit.c\n M h3_gpu.h\n M h3_gpu.m\n M h3_shaders.metal",
    )
    monkeypatch.setattr(
        campaign.base, "read_command", lambda *args: "\0".join(sorted(campaign.CHANGED_FILES))
    )
    assert campaign.validate_source(source, provenance)
    for field, value in (
        ("source_sha256", "0" * 64),
        ("revision", "other"),
        ("status", provenance["status"] + "\n?? unexpected.c"),
    ):
        with pytest.raises(ValueError):
            campaign.validate_source(source, provenance | {field: value})
    monkeypatch.setattr(
        campaign.base,
        "read_command",
        lambda *args: "\0".join(sorted(campaign.CHANGED_FILES | {"h3.c"})),
    )
    with pytest.raises(ValueError):
        campaign.validate_source(source, provenance)


def test_all_twelve_case_recipes_and_order(campaign):
    assert len(campaign.CASES) == 12
    for seed, rows, policy in campaign.CASES:
        report = valid_report(campaign, seed, rows, policy)
        summary = campaign.validate_case(report, seed, rows, policy, 3, 3)
        assert summary["observed_native_over_fused_wall_ratio"] == 1


@pytest.mark.parametrize(
    "counter",
    [
        "nonfinite",
        "cpu_original_mismatches",
        "cpu_fused_mismatches",
        "hidden_bit_mismatches",
        "quantized_code_mismatches",
        "quantized_scale_bit_mismatches",
        "zero_row_errors",
        "padding_errors",
    ],
)
def test_any_full_output_contract_failure_rejects(campaign, counter):
    report = valid_report(campaign)
    report[counter] = 1
    with pytest.raises(campaign.base.NumericalError):
        campaign.validate_case(report, 17, 9914, "h256", 3, 3)


@pytest.mark.parametrize("value", [None, float("nan"), -1, True])
def test_bad_raw_time_rejected(campaign, value):
    report = valid_report(campaign)
    report["timing"]["fused"][0]["wall_seconds"] = value
    with pytest.raises(ValueError):
        campaign.validate_case(report, 17, 9914, "h256", 3, 3)


def test_one_independent_row_cannot_hide_in_aggregate(campaign):
    report = valid_report(campaign)
    report["independent_reference"]["per_row"][-1]["hidden_implementation"]["relative_l2"] = 0.009
    with pytest.raises(campaign.base.NumericalError, match="gate failed"):
        campaign.validate_case(report, 17, 9914, "h256", 3, 3)


def test_quantization_drift_is_reported_separately(campaign):
    report = valid_report(campaign)
    report["independent_reference"]["projection_quantization_vs_original_bf16"]["relative_l2"] = 0.2
    assert campaign.validate_case(report, 17, 9914, "h256", 3, 3)


def test_missing_repeats_or_dispatch_scope_rejected(campaign):
    for mutation in (
        lambda r: r["timing"]["native"].pop(),
        lambda r: r["timing"]["fused"][0].update(direct_dispatches=2),
    ):
        report = valid_report(campaign)
        mutation(report)
        with pytest.raises(ValueError):
            campaign.validate_case(report, 17, 9914, "h256", 3, 3)


def test_campaign_continues_after_failure_and_holds_lock(campaign, tmp_path, monkeypatch):
    held = {"value": False}

    @contextmanager
    def lock(*args):
        held["value"] = True
        try:
            yield
        finally:
            held["value"] = False

    monkeypatch.setattr(campaign.base, "wait_for_gpu_lock", lock)
    monkeypatch.setattr(campaign.base, "git_provenance", lambda _: {})
    monkeypatch.setattr(campaign, "validate_source", lambda *args: {})
    monkeypatch.setattr(campaign.base, "runtime_environment", lambda: ({}, {}))
    monkeypatch.setattr(campaign.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(campaign.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        campaign.base,
        "read_command",
        lambda cmd: "128" if cmd[-1] == "hw.memsize" else "Apple M5 Max",
    )

    def preflight(*args):
        assert held["value"]
        return {}

    monkeypatch.setattr(campaign.base, "model_inventory", preflight)
    monkeypatch.setattr(campaign, "compile_driver", lambda *args: tmp_path / "private-driver")
    monkeypatch.setattr(campaign, "CASES", ((17, 2049, "h256"), (29, 9914, "attention")))

    def logged(command, source, env, directory, name):
        assert held["value"]
        report = valid_report(campaign, int(command[4]), int(command[3]), command[5])
        if report["seed"] == 17:
            report["nonfinite"] = 1
        Path(command[6]).write_text(json.dumps(report))
        return {"returncode": 2 if report["seed"] == 17 else 0, "fallback_detected": False}

    monkeypatch.setattr(campaign.base, "run_logged", logged)
    args = campaign.parser().parse_args(["--output-dir", str(tmp_path / "new"), "--repeats", "3"])
    result = campaign.execute(args)
    assert not held["value"]
    assert [case["status"] for case in result["cases"]] == ["incorrect", "completed"]
    assert (args.output_dir / "h256-m2049-seed17/driver.json").is_file()
    assert result["status"] == "failed"
    with pytest.raises(FileExistsError):
        campaign.execute(args)


def test_invalid_options_preserve_failure_without_gpu(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign.base, "wait_for_gpu_lock", lambda *args: pytest.fail("must fail before GPU lock")
    )
    args = campaign.parser().parse_args(["--output-dir", str(tmp_path / "new"), "--repeats", "1"])
    result = campaign.execute(args)
    assert result["status"] == "failed"
    assert (args.output_dir / "campaign.json").exists()

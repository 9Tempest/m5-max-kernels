"""CPU-only rejection tests for the native preparation evidence gate."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def module(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    for name in ("run", "run_fc2_experiment", "run_fused_campaign", "kernel_integration"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "prep_campaign_test", directory / "run_prep_campaign.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metric(count):
    return dict(count=count, nonfinite=0, relative_l2=0.001, rmse=0.002, max_abs=0.004)


def report(module):
    result = dict(
        schema="h3_prep_validation_v1",
        rows=17,
        padded_rows=128,
        K=5376,
        seed=17,
        gate=1,
        distinct_maps=1,
        passed=True,
    )
    result.update(dict.fromkeys(module.COUNTERS, 0))
    variants = []
    timings = []
    for index, name in enumerate(("native", "staged", "fused")):
        variant = dict(name=name, quantization_vs_original_adaln=metric(4 * 5376))
        if index:
            variant.update(
                implementation_vs_cpu_quantized=metric(4 * 5376),
                adaln_implementation_vs_fp64=metric(4 * 5376),
                per_row_implementation=[metric(5376) for _ in range(4)],
            )
        variants.append(variant)
        timings.append(
            dict(
                name=name,
                samples=[
                    dict(
                        wall_seconds=0.01,
                        supplemental_gpu_seconds=0.009,
                        direct_dispatches=(2, 3, 1)[index],
                    )
                    for _ in range(3)
                ],
            )
        )
    result["reference"] = dict(
        full_width=5376, implementation_limit=0.008, rows=[0, 1, 8, 16], variants=variants
    )
    result["timing"] = dict(
        status="completed",
        scope="preparation_only_synchronized_online_including_padding",
        warmups=3,
        repeats=3,
        rotation_seed_offset=0,
        variants=timings,
    )
    return result


def validate(module, value):
    return module.validate(value, 17, 17, 1, 1, 3, 3)


def test_valid_independent_and_paired_evidence(module):
    assert validate(module, report(module))["native_over_fused"] == 1


@pytest.mark.parametrize(
    "counter",
    (
        "nonfinite",
        "gate_errors",
        "padding_errors",
        "zero_errors",
        "staged_fused_code_mismatches",
        "staged_fused_scale_mismatches",
        "staged_fused_adaln_mismatches",
    ),
)
def test_rejects_each_contract_failure(module, counter):
    value = report(module)
    value[counter] = 1
    with pytest.raises(ValueError):
        validate(module, value)


@pytest.mark.parametrize(
    "field,bad",
    (
        ("relative_l2", 0.009),
        ("nonfinite", 1),
        ("count", 5375),
        ("relative_l2", float("nan")),
        ("max_abs", float("inf")),
    ),
)
def test_per_row_reference_cannot_hide_behind_aggregate(module, field, bad):
    value = report(module)
    value["reference"]["variants"][2]["per_row_implementation"][3][field] = bad
    with pytest.raises(ValueError):
        validate(module, value)


def test_quantization_error_is_not_implementation_error(module):
    value = report(module)
    value["reference"]["variants"][2]["quantization_vs_original_adaln"]["relative_l2"] = 0.12
    assert validate(module, value)


@pytest.mark.parametrize(
    "change", ("missing_tail", "missing_sample", "dispatch", "negative_wall", "false_pass")
)
def test_rejects_incomplete_or_changed_scope(module, change):
    value = copy.deepcopy(report(module))
    if change == "missing_tail":
        value["reference"]["rows"][-1] = 15
    elif change == "missing_sample":
        value["timing"]["variants"][2]["samples"].pop()
    elif change == "dispatch":
        value["timing"]["variants"][2]["samples"][0]["direct_dispatches"] = 2
    elif change == "negative_wall":
        value["timing"]["variants"][2]["samples"][0]["wall_seconds"] = -1
    else:
        value["passed"] = False
    with pytest.raises(ValueError):
        validate(module, value)


def test_non_m5_is_unsupported_not_passing(module):
    with pytest.raises(module.base.UnsupportedError):
        module.require_device("Apple M4 Max")
    module.require_device("Apple M5 Max")


@pytest.mark.parametrize(
    "message",
    ("TensorOps unavailable", "cannot initialize Metal", "TensorOps compile failed: unavailable"),
)
def test_missing_backend_is_classified_before_parsing_empty_report(module, message):
    with pytest.raises(module.base.UnsupportedError):
        module.check_process_support(dict(returncode=1, fallback_detected=False), message)

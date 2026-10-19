"""CPU-only rejection fixtures for the independent native QKV evidence contract."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "qkv_report_contract",
    Path(__file__).resolve().parents[1] / "integrations/h3/validate_qkv_report.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metric(count=384):
    return dict(count=count, nonfinite=0, relative_l2=0.001, rmse=0.02, max_abs=0.1)


def fixture(rows=9914, seed=17, policy="native"):
    sample = dict(
        wall_seconds=0.1,
        cpu_encode_seconds=0.03,
        submit_wait_seconds=0.07,
        supplemental_gpu_seconds=0.06,
        direct_dispatches=1,
    )
    value = dict(
        schema=MODULE.SCHEMA,
        status="completed",
        seed=seed,
        operand_policy=policy + "-int8",
        shape=dict(
            rows=rows,
            padded_rows=(rows + 127) & ~127,
            input_dim=5376,
            heads=56,
            head_dim=128,
            rope_half=48,
        ),
        baseline_shader="h3_qkv_project_split_int8_rope_"
        + ("local_scales_" if rows > 2048 else "")
        + "nax_r128_k5376_morton4",
        baseline_selection_basis="native source default, runtime overrides rejected",
        candidate_shader="h3_m5_qkv_rope_int8_r64_swizzle8",
        candidate_tile=[64, 128, 4, 8],
        baseline_slow_flags=[0, 0, 0],
        implementation_relative_l2_limit=0.008,
        offline_weight_quantization_seconds=0.1,
        warmups=3,
        repeats=10,
        first_candidate_parity=int(seed == 29),
        zero_output_errors=0,
        epsilon_dominated_reference_errors=0,
        packing=dict(
            sampled_input_rows=4,
            sampled_weight_rows=1152,
            input_errors=0,
            weight_errors=0,
            padding_errors=0,
            nonfinite_scales=0,
        ),
        paired=[
            dict(
                stream=s,
                bit_mismatches=0,
                native_nonfinite=0,
                candidate_nonfinite=0,
                metric=metric(rows * 7168),
            )
            for s in range(3)
        ],
        reference=[
            dict(
                row=r,
                kind=k,
                heads=[0, 28, 55],
                streams=[
                    dict(
                        stream=s,
                        **{
                            name: metric()
                            for name in (
                                "native_implementation",
                                "candidate_implementation",
                                "native_total_error",
                                "candidate_total_error",
                                "quantization_error",
                            )
                        },
                    )
                    for s in range(3)
                ],
            )
            for r, k in zip(
                [0, 1, rows // 2, rows - 1],
                ["zeros", "epsilon_dominated", "normal", "outliers_tail"],
                strict=True,
            )
        ],
        timings={},
        timing_scope="synchronized, already AdaLN input; excludes offline weights",
        quality_decision="not_evaluated",
    )
    for name, dispatches in (("prepacked_qkv", 1), ("online_activation_plus_qkv", 2)):
        value["timings"][name] = {
            p: [dict(sample, direct_dispatches=dispatches) for _ in range(10)]
            for p in ("native", "candidate")
        }
    return value


@pytest.mark.parametrize("rows", [129, 2049, 9914, 17455])
@pytest.mark.parametrize("seed", [17, 29])
@pytest.mark.parametrize("policy", ["native", "h256"])
def test_complete_reports(rows, seed, policy):
    assert MODULE.validate(fixture(rows, seed, policy), rows, seed, policy) == "completed"


def test_small_shape_explicitly_unsupported_without_timings():
    report = dict(
        schema=MODULE.SCHEMA,
        status="unsupported",
        rows=17,
        seed=17,
        reason="native API requires M>=128",
    )
    assert MODULE.validate(report, 17, 17, "native") == "unsupported"
    report["timings"] = {}
    with pytest.raises(ValueError):
        MODULE.validate(report, 17, 17, "native")


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "incorrect"),
        ("implementation_relative_l2_limit", 0.009),
        ("operand_policy", "h256-int8"),
        ("warmups", False),
        ("zero_output_errors", 1),
        ("epsilon_dominated_reference_errors", 1),
        ("quality_decision", "passed"),
        ("offline_weight_quantization_seconds", float("nan")),
        ("baseline_shader", "staged_fake_baseline"),
    ],
)
def test_reject_incorrect_or_changed_contract(key, value):
    report = fixture()
    report[key] = value
    with pytest.raises(ValueError):
        MODULE.validate(report, 9914, 17, "native")


@pytest.mark.parametrize(
    "change",
    [
        "missing_tail",
        "missing_stream",
        "missing_head",
        "threshold",
        "nan",
        "null",
        "nonfinite",
        "count",
        "V_diff",
        "padding",
        "missing_time",
        "bad_dispatch",
        "bad_wall",
        "bad_gpu",
        "false_int",
    ],
)
def test_reject_incomplete_numerics_or_timing(change):
    report = fixture()
    implementation = report["reference"][3]["streams"][0]["candidate_implementation"]
    sample = report["timings"]["online_activation_plus_qkv"]["candidate"][0]
    if change == "missing_tail":
        report["reference"].pop()
    elif change == "missing_stream":
        report["reference"][0]["streams"].pop()
    elif change == "missing_head":
        report["reference"][0]["heads"].pop()
    elif change == "threshold":
        implementation["relative_l2"] = 0.00800001
    elif change == "nan":
        implementation["relative_l2"] = float("nan")
    elif change == "null":
        implementation["relative_l2"] = None
    elif change == "nonfinite":
        implementation["nonfinite"] = 1
    elif change == "count":
        implementation["count"] = 383
    elif change == "V_diff":
        report["paired"][2]["bit_mismatches"] = 1
    elif change == "padding":
        report["packing"]["padding_errors"] = 1
    elif change == "missing_time":
        report["timings"]["prepacked_qkv"]["native"].pop()
    elif change == "bad_dispatch":
        sample["direct_dispatches"] = 1
    elif change == "bad_wall":
        sample["wall_seconds"] = 0.2
    elif change == "bad_gpu":
        sample["supplemental_gpu_seconds"] = float("inf")
    elif change == "false_int":
        report["paired"][0]["native_nonfinite"] = False
    with pytest.raises(ValueError):
        MODULE.validate(report, 9914, 17, "native")


def test_qk_bits_and_quantization_error_are_not_implementation_gates():
    report = fixture()
    report["paired"][0]["bit_mismatches"] = 128
    report["reference"][3]["streams"][0]["quantization_error"]["relative_l2"] = 0.03
    snapshot = deepcopy(report)
    assert MODULE.validate(report, 9914, 17, "native") == "completed"
    assert report == snapshot

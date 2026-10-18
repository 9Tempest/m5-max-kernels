import sys

import numpy as np
import pytest

from m5kernels.rms_quant_gemm import _bf16_round, prepare, validate_config


def _case(block_size=1, **config):
    return dict(
        id="test.rms_quant_projection",
        op="rms_quant_gemm",
        m=7,
        n=33,
        k=769 if block_size == 1 else 768,
        rms_quant=dict(block_size=block_size, gamma="random", sign_seed=23, **config),
    )


def test_explicit_config_defaults_and_input_immutability():
    case = dict(id="test.rms", op="rms_quant_gemm", m=3, n=7, k=17)
    config = validate_config(case, "rms-folded", "int8", "normal")
    assert config == dict(block_size=1, gamma="identity", threads=256, epsilon=1e-5, sign_seed=None)
    assert "rms_quant" not in case


@pytest.mark.parametrize("backend", ["rms-direct", "rms-folded"])
@pytest.mark.parametrize("block_size", [1, 256])
def test_512_threads_supported_for_h3_width_and_plain_tails(backend, block_size):
    case = _case(block_size, threads=512)
    if block_size == 256:
        case["k"] = 5376
    config = validate_config(case, backend, "int8", "normal")
    assert config["threads"] == 512
    assert config["block_size"] == block_size


@pytest.mark.parametrize(
    "option,value",
    [
        ("shift", 0),
        ("shift", 0.1),
        ("norm", "layer"),
        ("norm", "rms"),
        ("round_output", "input"),
        ("round_output", "fp32"),
        ("unexpected", True),
    ],
)
def test_unknown_or_incompatible_semantics_are_not_silently_ignored(option, value):
    case = _case()
    case["rms_quant"][option] = value
    with pytest.raises(NotImplementedError, match="unsupported rms_quant options"):
        # Rejection precedes MLX import and allocation, so CPU CI checks it.
        prepare(case, "rms-folded", "int8", 17, "normal", (32, 64, 4, 8))


@pytest.mark.parametrize("option", ["shift", "norm", "round_output"])
def test_top_level_semantics_options_are_rejected(option):
    case = _case()
    case[option] = "unsupported"
    with pytest.raises(NotImplementedError, match="semantics options"):
        validate_config(case, "rms-direct", "int8", "normal")


@pytest.mark.parametrize("eps", [0, -1, np.nan, np.inf, 1e-100, 1e100, True])
def test_epsilon_must_be_representable_positive_float32(eps):
    case = _case(epsilon=eps)
    with pytest.raises(ValueError, match="epsilon"):
        validate_config(case, "rms-direct", "int8", "normal")


@pytest.mark.parametrize(
    "backend,precision", [("mlx", "bf16"), ("rms-folded", "bf16"), ("unknown", "int8")]
)
def test_backend_precision_rejection(backend, precision):
    with pytest.raises(NotImplementedError, match="INT8 only"):
        validate_config(_case(), backend, precision, "normal")


def test_padding_and_gamma_policy_are_explicit():
    case = _case(256)
    case["k"] = 769
    with pytest.raises(NotImplementedError, match="padding"):
        validate_config(case, "rms-folded", "int8", "normal")
    case = _case()
    case["rms_quant"]["gamma"] = "unknown"
    with pytest.raises(NotImplementedError, match="gamma"):
        validate_config(case, "rms-folded", "int8", "normal")


def test_cpu_bf16_diagnostic_rounding_at_ties():
    actual = _bf16_round(np.array([1.00390625, 1.01171875, -1.00390625, -1.01171875]))
    np.testing.assert_array_equal(actual, [1.0, 1.015625, -1.0, -1.015625])


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
@pytest.mark.parametrize("backend", ["rms-direct", "rms-folded"])
@pytest.mark.parametrize("block_size", [1, 256])
@pytest.mark.parametrize("distribution", ["normal", "outliers", "zeros"])
def test_rms_quant_projection_reference_and_timing(backend, block_size, distribution):
    pytest.importorskip("mlx.core")
    from m5kernels.engine import run_case

    result = run_case(
        _case(block_size),
        backend,
        "int8",
        distribution=distribution,
        warmup=1,
        repeat=3,
        tile=(32, 64, 4, 8),
    )
    assert result["status"] == "ok", result
    assert result["transform_pass"] and result["online_output_finite"]
    assert result["activation_max_code_difference"] <= 1
    assert result["activation_scale_error"]["relative_l2"] <= 1e-5
    assert result["activation_implementation_error"]["relative_l2"] <= 0.008
    assert result["weight_rotation_error"]["relative_l2"] <= 1e-5
    assert result["weight_max_code_difference"] <= 1
    assert result["weight_scale_error"]["relative_l2"] <= 1e-5
    assert result["ideal_activation_quantization_error"]["finite"]
    assert result["ideal_decoded_quantization_error"]["finite"]
    assert result["decoded_preparation_total_error"]["finite"]
    assert result["quality_pass"] == (
        result["quality_error"]["relative_l2"] <= result["quality_relative_l2_limit"]
    )
    assert result["bf16_intermediate_diagnostic"]["diagnostic_only"]
    assert result["bf16_intermediate_diagnostic"]["gemm_difference"]["finite"]
    assert result["normalization_rounding"] == "fp32" and not result["additive_shift"]
    assert result["stages"]["activation_prepare"]["median_ms"] > 0
    assert result["stages"]["online_activation_plus_gemm"]["median_ms"] > 0
    assert result["stages"]["offline_weight_prepare"]["median_ms"] > 0
    assert result["weight_preparation_timed"] and not result["weight_preparation_in_online"]
    if distribution == "zeros":
        assert result["quality_error"]["relative_l2"] == 0


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
@pytest.mark.parametrize("block_size", [1, 256])
def test_direct_and_folded_use_identical_inputs_and_semantic_reference(block_size):
    pytest.importorskip("mlx.core")
    case = _case(block_size)
    direct = prepare(case, "rms-direct", "int8", 53, "normal", (32, 64, 4, 8))
    folded = prepare(case, "rms-folded", "int8", 53, "normal", (32, 64, 4, 8))
    np.testing.assert_array_equal(direct[3], folded[3])
    assert direct[5]["gamma_sha256"] == folded[5]["gamma_sha256"]
    assert direct[5]["transform"]["sign_sha256"] == folded[5]["transform"]["sign_sha256"]
    assert direct[5]["fold_scale"] is False and folded[5]["fold_scale"] is True


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
def test_unsupported_shift_survives_as_harness_result():
    pytest.importorskip("mlx.core")
    from m5kernels.engine import run_case

    case = _case(256, shift=0.1)
    result = run_case(case, "rms-folded", "int8", warmup=1, repeat=3)
    assert result["status"] == "unsupported", result
    assert "shift" in result["reason"]

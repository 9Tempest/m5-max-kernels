import sys

import numpy as np
import pytest

from m5kernels.qk_rope_experiment import prepare, sample_complete_heads, validate_config


def case(**changes):
    result = dict(id="qk.tail", op="qk_rope", m=7, n=384, k=256, head_dim=128, rope_half=48)
    result.update(changes)
    return result


def test_complete_head_sampling_includes_boundaries():
    heads, columns = sample_complete_heads(7168)
    np.testing.assert_array_equal(heads, [0, 55])
    np.testing.assert_array_equal(columns[:128], np.arange(128))
    np.testing.assert_array_equal(columns[128:], np.arange(7040, 7168))
    assert columns.size == 256


def test_small_projection_samples_every_available_head():
    heads, columns = sample_complete_heads(128)
    np.testing.assert_array_equal(heads, [0])
    np.testing.assert_array_equal(columns, np.arange(128))


@pytest.mark.parametrize("backend", ["qk-staged", "qk-fused"])
@pytest.mark.parametrize("tile", [(16, 64, 4, 1), (32, 128, 8, 7), (128, 128, 4, 8)])
def test_config_is_explicit_and_immutable(backend, tile):
    c = case()
    before = c.copy()
    cfg = validate_config(c, backend, "int8", "normal", tile)
    assert cfg == dict(head_dim=128, rope_half=48, epsilon=float(np.float32(1e-5)), block_size=256)
    assert c == before


@pytest.mark.parametrize(
    "backend,precision", [("mlx", "bf16"), ("qk-fused", "bf16"), ("unknown", "int8")]
)
def test_unsupported_backend_before_gpu_import(backend, precision):
    with pytest.raises(NotImplementedError, match="INT8 only"):
        prepare(case(), backend, precision, 17, "normal", (32, 128, 4, 1))


@pytest.mark.parametrize(
    "changes,error,message",
    [
        ({"op": "gemm"}, ValueError, "operation"),
        ({"m": 0}, ValueError, "positive"),
        ({"n": True}, ValueError, "positive"),
        ({"k": 1.5}, ValueError, "positive"),
        ({"k": 257}, NotImplementedError, "padding"),
        ({"k": 131072}, NotImplementedError, "overflow"),
        ({"n": 129}, NotImplementedError, "complete"),
        ({"head_dim": 64}, NotImplementedError, "complete"),
        ({"rope_half": 65}, ValueError, "rope_half"),
        ({"rope_half": True}, ValueError, "rope_half"),
        ({"epsilon": 0}, ValueError, "epsilon"),
        ({"epsilon": np.nan}, ValueError, "epsilon"),
        ({"epsilon": 1e-100}, ValueError, "epsilon"),
        ({"epsilon": 1e100}, ValueError, "epsilon"),
    ],
)
def test_rejected_shapes_and_semantics(changes, error, message):
    with pytest.raises(error, match=message):
        prepare(case(**changes), "qk-fused", "int8", 17, "normal", (32, 128, 4, 1))


@pytest.mark.parametrize(
    "tile", [(32, 128, 4), (32, 128, 4, 0), (True, 128, 4, 1), (32, 128, 4, 1.5)]
)
def test_invalid_tiles(tile):
    with pytest.raises(ValueError, match="tile"):
        validate_config(case(), "qk-fused", "int8", "normal", tile)


def test_unsupported_tile_and_distribution_are_explicit():
    with pytest.raises(NotImplementedError, match="tile"):
        validate_config(case(), "qk-fused", "int8", "normal", (32, 32, 4, 1))
    with pytest.raises(ValueError, match="distribution"):
        validate_config(case(), "qk-fused", "int8", "unsupported", (32, 128, 4, 1))


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
@pytest.mark.parametrize("backend", ["qk-staged", "qk-fused"])
@pytest.mark.parametrize("distribution", ["normal", "outliers", "zeros"])
def test_complete_online_projection_norm_rope_smoke(backend, distribution):
    pytest.importorskip("mlx.core")
    from m5kernels.engine import run_case

    result = run_case(
        case(),
        backend,
        "int8",
        seed=131,
        distribution=distribution,
        warmup=1,
        repeat=3,
        tile=(16, 64, 4, 3),
    )
    assert result["status"] == "ok", result
    assert result["transform_pass"] and result["quality_pass"]
    assert result["output_finite"] and result["online_output_finite"]
    assert result["sample_heads"] == [0, 2]
    assert len(result["sample_cols"]) == 256
    assert result["paired_path_exact_required"] and result["paired_path_exact"]
    assert result["paired_path_error"]["relative_l2"] <= 0.008
    assert result["projection_implementation_error"]["relative_l2"] <= 0.008
    assert result["postprocess_implementation_error"]["relative_l2"] <= 0.008
    assert result["ideal_decoded_quantization_error"]["finite"]
    assert result["decoded_preparation_total_error"]["finite"]
    assert result["weight_preparation_timed"] and not result["weight_preparation_in_online"]
    assert result["stages"]["activation_prepare"]["median_ms"] > 0
    assert result["stages"]["postprocess"]["median_ms"] > 0
    assert result["stages"]["online_activation_plus_gemm"]["median_ms"] > 0
    assert result["stages"]["offline_weight_prepare"]["median_ms"] > 0
    if distribution == "zeros":
        assert result["quality_error"]["relative_l2"] == 0


@pytest.mark.metal
@pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
def test_corruption_in_unsampled_head_blocks_timing(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from m5kernels import qk_rope
    from m5kernels.engine import run_case

    original = qk_rope.projection_qk_rope

    def corrupt_middle_head(*args, **kwargs):
        output = original(*args, **kwargs)
        mask = mx.arange(output.shape[0])[:, None, None] == 1
        return mx.where(mask, -output, output)

    monkeypatch.setattr(qk_rope, "projection_qk_rope", corrupt_middle_head)
    result = run_case(case(), "qk-fused", "int8", seed=131, warmup=1, repeat=3, tile=(16, 64, 4, 3))
    assert result["sample_heads"] == [0, 2]
    # The corruption is finite and invisible to the sampled comparisons.
    assert result["output_finite"] and result["online_output_finite"]
    assert result["paired_path_error"]["relative_l2"] == 0
    assert result["implementation_error"]["relative_l2"] <= 0.008
    assert result["paired_path_exact_required"] and not result["paired_path_exact"]
    assert not result["transform_pass"]
    assert result["status"] == "incorrect"
    assert "timing" not in result and "stages" not in result

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "fc2_analyze", Path(__file__).parents[1] / "integrations/h3/fc2_analyze.py"
)
fc2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fc2)


def bf16_bits(x):
    bits = np.asarray(x, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def rounded(x):
    return (bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def quant(x, hadamard, padded=None):
    x = fc2.hadamard_reference(x, np.float32) if hadamard else x
    peak = np.max(np.abs(x), axis=1)
    scale = np.where(peak > 0, peak / 127, 1 if hadamard else 1 / 127).astype(np.float32)
    q = np.rint(
        x / scale[:, None]
        if hadamard
        else x * (np.where(peak > 0, 127 / np.where(peak > 0, peak, 1), 127))[:, None]
    )
    q = np.clip(q, -127, 127).astype(np.int8)
    if padded:
        q = np.pad(q, ((0, padded - len(q)), (0, 0)))
        scale = np.pad(scale, (0, padded - len(scale)), constant_values=1)
    return q, scale


def fixture(d):
    rng = np.random.default_rng(43)
    m, p, k, n, c, count = 129, 256, 1024, 8, 4, 3
    x = np.zeros((m, k), np.float32)
    x[:count] = rounded(rng.normal(size=(count, k)))
    x[1, 17] = 100
    w = rounded(rng.normal(size=(c, k)))
    columns = np.array([0, 1, 6, 7], "<u4")
    meta = dict(rows=m, padded_rows=p, K=k, N=n, sample_columns=c, sample_rows=count)
    (d / "driver.json").write_text(json.dumps(meta))
    bf16_bits(x).tofile(d / "input.bf16")
    bf16_bits(w).tofile(d / "weight-samples.bf16")
    columns.tofile(d / "columns.u32")
    for tag in fc2.POLICIES:
        h = tag == "hadamard-row"
        aq, sa = quant(x, h, p)
        wq, sw = quant(w, h)
        y = np.zeros((m, n), np.float32)
        y[:, columns] = (aq[:m].astype(np.float64) * sa[:m, None]) @ (
            wq.astype(np.float64) * sw[:, None]
        ).T
        bf16_bits(y).tofile(d / f"{tag}.output.bf16")
        aq.tofile(d / f"{tag}.activation.i8")
        sa.tofile(d / f"{tag}.activation-scales.f32")
        wq.tofile(d / f"{tag}.weight-samples.i8")
        sw.tofile(d / f"{tag}.weight-scales.f32")


def test_reference_preserves_full_dot_and_is_involution():
    rng = np.random.default_rng(29)
    x, w = rng.normal(size=(5, 1024)), rng.normal(size=(3, 1024))
    x[0] = 0
    x[1, 17] = 10000
    hx, hw = fc2.hadamard_reference(x), fc2.hadamard_reference(w)
    np.testing.assert_allclose(hx @ hw.T, x @ w.T, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(fc2.hadamard_reference(hx), x, rtol=1e-11, atol=1e-11)


def test_reference_rejects_unsupported_or_nonfinite():
    with pytest.raises(ValueError):
        fc2.hadamard_reference(np.zeros((2, 255)))
    with pytest.raises(ValueError):
        fc2.hadamard_reference(np.full((2, 256), np.inf))


def test_analyzer_separates_implementation_and_quantization(tmp_path):
    fixture(tmp_path)
    result = fc2.analyze(tmp_path)
    for record in result["policies"].values():
        assert record["status"] == "passed"
        assert record["zero_rows_exact"] and record["padded_rows_codes_zero_scales_one"]
        assert record["implementation_error_vs_actual_decoded_operands"]["relative_l2"] < 0.004
        assert record["quantization_error_vs_original_bf16_operands"]["relative_l2"] > 0.001
    assert result["checkpoint_replacement_eligible"] is False


@pytest.mark.parametrize("failure", ["padding", "weight_scale", "output_nan", "wrong_codes"])
def test_analyzer_detects_corruption(tmp_path, failure):
    fixture(tmp_path)
    tag = "hadamard-row"
    if failure == "padding":
        path = tmp_path / f"{tag}.activation.i8"
        data = np.fromfile(path, np.int8)
        data[-1] = 1
    elif failure == "weight_scale":
        path = tmp_path / f"{tag}.weight-scales.f32"
        data = np.fromfile(path, np.float32)
        data[0] *= 2
    elif failure == "output_nan":
        path = tmp_path / f"{tag}.output.bf16"
        data = np.fromfile(path, np.uint16)
        data[2] = 0x7FC0  # Unsampled output column still must be finite.
    else:
        path = tmp_path / f"{tag}.activation.i8"
        data = np.fromfile(path, np.int8)
        data[0] = 127 if data[0] != 127 else -127
    data.tofile(path)
    assert fc2.analyze(tmp_path)["policies"][tag]["status"] == "incorrect"

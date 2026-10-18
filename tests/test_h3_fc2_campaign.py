import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


INTEGRATION = Path(__file__).parents[1] / "integrations/h3"
sys.path.insert(0, str(INTEGRATION))
SPEC = importlib.util.spec_from_file_location("fc2_campaign", INTEGRATION / "fc2_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
sys.path.remove(str(INTEGRATION))


@pytest.mark.parametrize(
    "value", ["17", "17:129:2", "-1:129", "17:127", "17:16385", f"{2**64}:129"]
)
def test_bad_synthetic_case_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        campaign.synthetic_case(value)


def test_scenarios_reject_conflicting_inputs():
    args = campaign.parser().parse_args(["--scenario", "calibration", "--output-dir", "unused"])
    with pytest.raises(ValueError, match="requires"):
        campaign.validate_arguments(args)
    args = campaign.parser().parse_args(
        ["--scenario", "synthetic", "--capture-run", "unused", "--output-dir", "unused"]
    )
    with pytest.raises(ValueError, match="do not consume"):
        campaign.validate_arguments(args)
    args = campaign.parser().parse_args(
        [
            "--scenario",
            "synthetic",
            "--synthetic-case",
            "17:129",
            "--synthetic-case",
            "17:129",
            "--output-dir",
            "unused",
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        campaign.validate_arguments(args)


def test_synthetic_rows_deterministic_finite_zero_and_nonzero_tail():
    a, b = campaign.synthetic_input(17, 129), campaign.synthetic_input(17, 129)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (129, 14336) and a.dtype == np.dtype("<u2")
    assert not np.any(a[5])
    assert np.any(a[-1])
    assert np.all((a & 0x7F80) != 0x7F80)
    values = (a.astype(np.uint32) << 16).view(np.float32)
    assert abs(values[9, ::257]).max() > 40


def test_capture_hash_binding_rejects_changed_files(tmp_path, monkeypatch):
    source = dict(
        status="completed",
        mode="upstream-int8",
        capture_fc2=True,
        patch={"source_sha256": campaign.experiment.SOURCE_SHA256},
        model={"revision": campaign.base.MODEL_REVISION},
        fc2_validation=[{"hash": "old"}],
    )
    (tmp_path / "run.json").write_text(json.dumps(source))
    monkeypatch.setattr(
        campaign.experiment, "validate_fc2_capture", lambda *args: [{"hash": "new"}]
    )
    with pytest.raises(ValueError, match="differ"):
        campaign.validate_capture(tmp_path)


def test_driver_failure_preserves_case_and_partial_readback(tmp_path, monkeypatch):
    infile = tmp_path / "input.bf16"
    np.zeros((16, 14336), "<u2").tofile(infile)
    monkeypatch.setattr(
        campaign.base, "run_logged", lambda *args: {"returncode": 1, "fallback_detected": True}
    )
    item = dict(name="failed-case", scenario="calibration", input=str(infile), rows=129, block=0)
    result = campaign.run_case(item, tmp_path, tmp_path, tmp_path, tmp_path / "driver", {})
    assert result["status"] == "unsupported"
    assert (tmp_path / "failed-case/readback").is_dir()
    saved = json.loads((tmp_path / "failed-case/case.json").read_text())
    assert saved["status"] == "unsupported"
    assert saved["process"]["returncode"] == 1


def test_invalid_campaign_writes_failure_without_gpu(tmp_path):
    args = campaign.parser().parse_args(
        ["--scenario", "calibration", "--output-dir", str(tmp_path / "out")]
    )
    result = campaign.execute(args)
    assert result["status"] == "failed" and not result["cases"]
    assert (tmp_path / "out/campaign.json").is_file()

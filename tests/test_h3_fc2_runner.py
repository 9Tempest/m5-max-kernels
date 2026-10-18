import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


INTEGRATION = Path(__file__).parents[1] / "integrations/h3"
sys.path.insert(0, str(INTEGRATION))
SPEC = importlib.util.spec_from_file_location("fc2_runner", INTEGRATION / "run_fc2_experiment.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
sys.path.remove(str(INTEGRATION))


def test_finite_file_checks_bf16_exponent_and_size(tmp_path):
    p = tmp_path / "x.bf16"
    np.array([0, 0x3F80, 0xBF80], "<u2").tofile(p)
    assert runner.finite_file(p, "<u2", 3)["finite"]
    with pytest.raises(ValueError, match="size"):
        runner.finite_file(p, "<u2", 2)
    np.array([0, 0x7F80, 0xBF80], "<u2").tofile(p)
    with pytest.raises(runner.base.NumericalError):
        runner.finite_file(p, "<u2", 3)


def test_finite_file_checks_float_nan(tmp_path):
    p = tmp_path / "x.f32"
    np.array([0, -1, np.nan], "<f4").tofile(p)
    with pytest.raises(runner.base.NumericalError):
        runner.finite_file(p, "<f4", 3)


def test_missing_capture_is_failure(tmp_path):
    with pytest.raises(ValueError, match="exactly steps"):
        runner.validate_fc2_capture(tmp_path, "upstream-int8")
    with pytest.raises(ValueError, match="all 20"):
        runner.validate_latent_capture(tmp_path, "bf16", 42)


def test_source_guard_rejects_other_patch_and_untracked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.base, "read_command", lambda *args: "\0".join(sorted(runner.CHANGED_FILES))
    )
    monkeypatch.setattr(runner.base, "sha256", lambda *args: runner.PATCH_SHA256)
    provenance = dict(
        revision=runner.base.UPSTREAM_REVISION,
        source_sha256=runner.SOURCE_SHA256,
        status=" M h3_dit.c",
    )
    assert runner.validate_source(tmp_path, provenance)["source_sha256"] == runner.SOURCE_SHA256
    with pytest.raises(ValueError, match="exactly"):
        runner.validate_source(tmp_path, provenance | {"source_sha256": "wrong"})
    with pytest.raises(ValueError, match="exactly"):
        runner.validate_source(tmp_path, provenance | {"status": "?? injected.metal"})


def test_recipe_has_no_arbitrary_steps_or_dimensions():
    args = runner.parser().parse_args(
        ["--mode", "hadamard-fc2", "--output-dir", "unused", "--seed", "43"]
    )
    assert args.seed == 43
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["--mode", "bf16", "--output-dir", "unused", "--steps", "2"])


def test_bf16_fc2_capture_rejected_before_gpu(tmp_path):
    args = runner.parser().parse_args(
        ["--mode", "bf16", "--output-dir", str(tmp_path / "out"), "--capture-fc2"]
    )
    result = runner.execute(args)
    assert result["status"] == "failed"
    assert "incompatible with BF16" in result["error"]
    assert not result["measurements"]

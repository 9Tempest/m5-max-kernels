"""CPU tests for complete native fusion trajectory comparison."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def runner(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    for name in ("run", "run_fc2_experiment", "run_fused_campaign"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "h3_fused_experiment_test", directory / "run_fused_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latents():
    names = [f"euler-step{s:02d}-{kind}.f32" for s in range(20) for kind in ("video", "audio")]
    names += ["final-video.f32", "final-audio.f32"]
    return {
        "tensors": [
            dict(file=name, elements=100, finite=True, sha256=f"{i:064x}")
            for i, name in enumerate(names)
        ]
    }


def test_requires_every_euler_and_final_tensor(runner):
    reference = latents()
    result = runner.latent_parity(reference, copy.deepcopy(reference))
    assert result["passed"] and len(result["tensors"]) == 42
    candidate = copy.deepcopy(reference)
    candidate["tensors"].pop()
    with pytest.raises(ValueError, match="all 40 Euler"):
        runner.latent_parity(reference, candidate)


def test_any_single_step_difference_is_retained(runner):
    reference = latents()
    candidate = copy.deepcopy(reference)
    candidate["tensors"][17]["sha256"] = "f" * 64
    result = runner.latent_parity(reference, candidate)
    assert not result["passed"]
    assert sum(not row["byte_equal"] for row in result["tensors"]) == 1


@pytest.mark.parametrize("field,value", [("elements", 99), ("finite", False)])
def test_finite_and_layout_preconditions(runner, field, value):
    reference = latents()
    candidate = copy.deepcopy(reference)
    candidate["tensors"][0][field] = value
    with pytest.raises(runner.base.NumericalError):
        runner.latent_parity(reference, candidate)


@pytest.mark.parametrize(
    "field,value",
    [("seed", 43), ("mode", "upstream-int8"), ("status", "failed"), ("fusion", "fc2")],
)
def test_incompatible_reference_rejected(runner, monkeypatch, tmp_path, field, value):
    record = dict(
        status="completed",
        mode="hadamard-fc2",
        seed=42,
        capture_latents=True,
        model={"revision": runner.base.MODEL_REVISION},
        patch={"source_sha256": runner.experiment.SOURCE_SHA256},
    )
    record[field] = value
    monkeypatch.setattr(runner.experiment, "load_json", lambda _: record)
    with pytest.raises(ValueError, match="same-seed"):
        runner.validate_reference(tmp_path, 42)


def test_invalid_seed_writes_failure_without_gpu(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner.base, "wait_for_gpu_lock", lambda *args: pytest.fail("GPU must not run")
    )
    args = runner.parser().parse_args(
        [
            "--fusion",
            "both",
            "--seed",
            "-1",
            "--reference-run",
            str(tmp_path / "ref"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    result = runner.execute(args)
    assert result["status"] == "failed"
    assert (args.output_dir / "run.json").exists()
    with pytest.raises(FileExistsError):
        runner.execute(args)

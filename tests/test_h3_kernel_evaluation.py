"""Small CPU arrays only; no model loading or media subprocesses."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/h3"))
import evaluate_kernel_integration as evaluation
import kernel_evidence as evidence


def test_tail_metrics_and_nonzero_allowed_qkv_drift():
    a = {"final_video": np.array([1, 2, 3, 4], dtype=np.float32)}
    b = {"final_video": a["final_video"] + np.float32(1e-4)}
    result = evaluation.compare_with_tail(a, b)["final_video"]
    assert result["status"] == "completed"
    assert result["metrics"]["relative_l2"]["value"] > 0
    assert result["absolute_error_quantiles"]["p999"] > 0


def test_exact_pair_retains_json_safe_infinity_status():
    value = {"final_audio": np.ones(4, dtype=np.float32)}
    result = evaluation.compare_with_tail(value, value)["final_audio"]
    assert result["metrics"]["snr_db"]["status"] == "positive_infinity"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_never_produces_successful_report(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        evaluation.compare_with_tail({"a": np.ones(2)}, {"a": np.array([1, bad])})


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    contract = evaluation.contract
    directory = tmp_path / "candidate"
    (directory / "latents").mkdir(parents=True)
    shapes = {key: [2] for key in contract.SHAPES}
    monkeypatch.setattr(contract, "SHAPES", shapes)
    for step in range(20):
        for kind in ("video", "audio"):
            np.ones(2, dtype="<f4").tofile(directory / f"latents/euler-step{step:02d}-{kind}.f32")
    for kind in ("video", "audio"):
        np.ones(2, dtype="<f4").tofile(directory / f"latents/final-{kind}.f32")
    log = directory / "generation.log"
    log.write_text(
        "h3 kernel integration: m5_qkv_rope=1 hadamard_qkv=0 hadamard_fc1=0\nh3 precision: FC2 plain normalized H256\n"
    )
    validation = dict(metadata_sha256="a" * 64, tensors=[])
    monkeypatch.setattr(contract, "validate_latents", lambda *_: copy.deepcopy(validation))
    monkeypatch.setattr(
        evidence, "producer", lambda *_: dict(model=evaluation.MODEL, source_sha256="b" * 64)
    )
    value = dict(
        **contract.RECIPE,
        schema=contract.SCHEMA,
        kind="capture",
        status="completed",
        seed=42,
        capture_latents=True,
        diagnostic=True,
        prepared_checkpoint=False,
        policy="qkv-only",
        implementation="fused",
        requested_runtime_flags=contract.flags("qkv-only"),
        runtime_overrides=contract.flags("qkv-only"),
        latent_validation=validation,
        measurements=dict(
            generation=dict(
                returncode=0,
                fallback_detected=False,
                log="generation.log",
                log_sha256=evaluation.metrics.sha256(log),
            )
        ),
    )
    (directory / "run.json").write_text(json.dumps(value))
    return directory, value


def test_candidate_reads_all_twenty_steps_and_final(candidate):
    directory, _ = candidate
    _, arrays, result = evaluation.candidate_capture(directory, "c" * 64, 42)
    assert len(arrays) == 42
    assert result["policy"] == "qkv-only"


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 43),
        ("width", 256),
        ("reference_rope", False),
        ("prepared_checkpoint", True),
        ("diagnostic", False),
    ],
)
def test_capture_recipe_binding(candidate, field, value):
    directory, record = candidate
    record[field] = value
    (directory / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        evaluation.candidate_capture(directory, "c" * 64, 42)


def test_capture_flags_bind_runtime_not_just_label(candidate):
    directory, record = candidate
    record["runtime_overrides"]["H3_M5_QKV_ROPE"] = "0"
    (directory / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="generation flag"):
        evaluation.candidate_capture(directory, "c" * 64, 42)


def test_generation_log_substitution_fails(candidate):
    directory, _ = candidate
    (directory / "generation.log").write_text("replacement")
    with pytest.raises(ValueError, match="log SHA"):
        evaluation.candidate_capture(directory, "c" * 64, 42)


def test_evidence_confined_even_via_symlink(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (tmp_path / "outside").write_text("evidence")
    (directory / "link").symlink_to(tmp_path / "outside")
    for name in ("../outside", "link", str(tmp_path / "outside")):
        with pytest.raises(ValueError):
            evidence.local_file(directory, name)


def test_historical_reference_requires_same_prompt_and_seed(tmp_path, monkeypatch):
    value = dict(**evaluation.contract.RECIPE, seed=42)
    value["prompt"] = "different trajectory"
    (tmp_path / "run.json").write_text(json.dumps(value))
    monkeypatch.setattr(
        evaluation.historical, "validate_capture", lambda *_: pytest.fail("reject before arrays")
    )
    with pytest.raises(ValueError, match="prompt"):
        evaluation.original_capture(tmp_path, "bf16", 42)

import importlib.util
from pathlib import Path
import sys

import pytest


INTEGRATION = Path(__file__).parents[1] / "integrations/h3"
sys.path.insert(0, str(INTEGRATION))
SPEC = importlib.util.spec_from_file_location(
    "run_prepared_checkpoint", INTEGRATION / "run_prepared_checkpoint.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
sys.path.remove(str(INTEGRATION))


def marker(count=200, seconds="2.345", quant="0"):
    return (
        "h3 precision: FC2 plain normalized H256\n"
        f"h3 prepared load: projections={count} verified_payload_load_wall_seconds={seconds} weight_rotation_quantization_seconds={quant}\n"
    )


def test_prepared_load_marker_requires_full_count_no_requantization():
    result = runner.validate_load_marker(marker())
    assert result["projections"] == 200
    assert result["verified_payload_load_wall_seconds"] == 2.345
    assert result["weight_rotation_quantization_seconds"] == 0


@pytest.mark.parametrize(
    "log", ["", marker(count=199), marker(quant="0.0001"), marker(seconds="0"), marker() + marker()]
)
def test_partial_missing_and_ambiguous_runtime_loads_rejected(log):
    with pytest.raises(ValueError):
        runner.validate_load_marker(log)


def test_source_guard_rejects_wrong_source_and_untracked_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.base, "read_command", lambda *a: "\0".join(sorted(runner.experiment.CHANGED_FILES))
    )
    monkeypatch.setattr(runner.base, "sha256", lambda *a: runner.PATCH_SHA256)
    provenance = dict(
        revision=runner.base.UPSTREAM_REVISION,
        source_sha256=runner.SOURCE_SHA256,
        status=" M h3_dit.c",
    )
    runner.validate_source(tmp_path, provenance)
    with pytest.raises(ValueError):
        runner.validate_source(tmp_path, provenance | {"source_sha256": "0" * 64})
    with pytest.raises(ValueError):
        runner.validate_source(tmp_path, provenance | {"status": "?? injected.metal"})


def test_bad_manifest_hash_fails_before_gpu_lock_and_records_failure(tmp_path, monkeypatch):
    def forbid(*args):
        raise AssertionError("invalid arguments must not acquire GPU lock")

    monkeypatch.setattr(runner.base, "wait_for_gpu_lock", forbid)
    args = runner.parser().parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--checkpoint-manifest-sha256",
            "bad",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    result = runner.execute(args)
    assert result["status"] == "failed" and "manifest SHA256" in result["error"]
    assert (tmp_path / "run/run.json").is_file()


def test_parity_requires_capture_before_gpu_lock(tmp_path, monkeypatch):
    def forbid(*args):
        raise AssertionError("invalid arguments must not acquire GPU lock")

    monkeypatch.setattr(runner.base, "wait_for_gpu_lock", forbid)
    args = runner.parser().parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--checkpoint-manifest-sha256",
            "0" * 64,
            "--parity-reference",
            "unused",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    result = runner.execute(args)
    assert result["status"] == "failed" and "requires all latent" in result["error"]

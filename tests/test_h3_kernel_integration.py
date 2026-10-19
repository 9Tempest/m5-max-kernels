"""CPU contracts only; this module never launches a native executable."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_h3_native_session import native as native, valid_events


DIRECTORY = Path(__file__).resolve().parents[1] / "integrations/h3"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "kernel_runner_test", DIRECTORY / "run_kernel_integration.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
contract = runner.contract


@pytest.fixture
def pinned(tmp_path):
    patch = tmp_path / "combined.patch"
    patch.write_text("fixture only\n")
    value = dict(
        schema="h3_kernel_integration_source_v1",
        status="frozen",
        upstream_revision=runner.base.UPSTREAM_REVISION,
        source_sha256="a" * 64,
        changed_files=["h3_dit.c", "h3_gpu.h", "h3_gpu.m", "h3_shaders.metal"],
        patches=[dict(file=patch.name, sha256=runner.base.sha256(patch))],
    )
    path = tmp_path / "source.json"
    path.write_text(json.dumps(value))
    return path, runner.base.sha256(path), value


def arguments(tmp_path, **overrides):
    args = runner.parser().parse_args(
        [
            "--policy",
            "baseline",
            "--kind",
            "capture",
            "--seed",
            "42",
            "--source-manifest-sha256",
            "0" * 64,
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )
    vars(args).update(overrides)
    return args


@pytest.mark.parametrize("policy,expected", list(contract.POLICIES.items()))
def test_four_unconfounded_policies(policy, expected):
    actual = contract.flags(policy)
    assert tuple(int(actual[key]) for key in contract.NEW_FLAGS) == expected
    assert actual["H3_HADAMARD_FC2"] == "1"
    assert actual["H3_FUSE_FC2_GATE"] == actual["H3_FUSE_ATTN_OUT_GATE"] == "0"
    assert actual["H3_HADAMARD_PREP_STAGED"] == "0"


@pytest.mark.parametrize("policy", ["baseline", "qkv-only"])
def test_staged_has_no_meaning_without_input_hadamard(policy):
    with pytest.raises(ValueError, match="only defined"):
        contract.flags(policy, "staged")


def test_staged_reference_never_enters_performance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.base, "wait_for_gpu_lock", lambda *_: pytest.fail("GPU must not be reached")
    )
    args = arguments(tmp_path, policy="combined", kind="resident", implementation="staged")
    result = runner.execute(args)
    assert result["status"] == "failed"
    assert "diagnostic-only" in result["error"]
    assert (args.output_dir / "run.json").exists()
    with pytest.raises(FileExistsError):
        runner.execute(args)


def test_pending_template_cannot_reach_gpu(tmp_path, monkeypatch, pinned):
    monkeypatch.setattr(
        runner.base, "wait_for_gpu_lock", lambda *_: pytest.fail("GPU must not be reached")
    )
    args = arguments(tmp_path)
    args.source_manifest, _, value = pinned
    value["status"] = "pending"
    args.source_manifest.write_text(json.dumps(value))
    args.source_manifest_sha256 = runner.base.sha256(args.source_manifest)
    result = runner.execute(args)
    assert result["status"] == "failed"
    assert "frozen" in result["error"]


def test_source_pin_requires_exact_bytes_and_overlay(pinned):
    path, digest, value = pinned
    assert contract.load_pin(path, digest) == value
    with pytest.raises(ValueError, match="approved SHA"):
        contract.load_pin(path, "b" * 64)
    (path.parent / "combined.patch").write_text("modified\n")
    with pytest.raises(ValueError, match="overlay digest"):
        contract.load_pin(path, digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "pending"),
        ("source_sha256", None),
        ("upstream_revision", "other"),
        ("patches", []),
        ("changed_files", ["h3_dit.c"]),
    ],
)
def test_incomplete_or_changed_source_pin_fails(pinned, field, value):
    path, _, manifest = pinned
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        contract.load_pin(path, runner.base.sha256(path))


def kernel_events(native, policy="combined"):
    events = valid_events(native, SimpleNamespace(mode="hadamard-fc2", fusion="none", seed=42))
    events[0].pop("fusion")
    events[0].update(policy=policy, implementation="fused")
    events[1] = dict(event="runtime_flags", **contract.flags(policy))
    return events


@pytest.mark.parametrize("policy", contract.POLICIES)
def test_driver_actual_flags_complete_callbacks_and_cache(native, policy):
    assert len(contract.validate_events(kernel_events(native, policy), policy, 42)) == 4


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e[1].update(H3_HADAMARD_QKV="0"),
        lambda e: e[1].update(H3_HADAMARD_PREP_STAGED="1"),
        lambda e: e[1].update(H3_FUSE_ATTN_OUT_GATE="1"),
        lambda e: e[0].update(policy="baseline"),
        lambda e: e[0].update(implementation="staged"),
        lambda e: e[5].update(denoise_callback_seconds=float("nan")),
        lambda e: e[5].update(denoise_end_offset_seconds=151.0),
        lambda e: e[5].update(result_seed=43),
        lambda e: e[5].update(denoise_completion_events=2),
        lambda e: e[5]["cache_before"].update(prepared_dit=False),
        lambda e: e[-1].update(exit_code=1),
    ],
)
def test_driver_contract_failures(native, mutate):
    events = kernel_events(native)
    mutate(events)
    with pytest.raises((ValueError, TypeError)):
        contract.validate_events(events, "combined", 42)


def test_precise_native_initialization_marker_is_required():
    marker = "h3 kernel integration: m5_qkv_rope=1 hadamard_qkv=1 hadamard_fc1=1\n"
    log = marker + "h3 precision: FC2 plain normalized H256\n"
    assert contract.validate_log(log, "combined", "fused")["initialization"]
    for bad in (
        "",
        log.replace("m5_qkv_rope=1", "m5_qkv_rope=0"),
        log + marker,
        log + "TensorOps compile failed:",
        log + "h3 fusion:",
    ):
        with pytest.raises((ValueError, runner.base.UnsupportedError)):
            contract.validate_log(bad, "combined", "fused")


@pytest.mark.parametrize(
    "key", ["m5_qkv_rope", "hadamard_qkv", "hadamard_fc1", "hadamard_prep_staged"]
)
def test_actual_latent_policy_is_type_checked(tmp_path, monkeypatch, key):
    metadata = contract.metadata_flags("combined", "staged") | dict(
        dtype="float32_little_endian",
        scope="diagnostic_CPU_readback_after_completed_GPU_Euler",
        final_video_sigma=0,
        final_audio_sigma=0,
    )
    result = {"metadata": metadata}
    monkeypatch.setattr(
        contract.capture, "validate_latent_capture", lambda *_: copy.deepcopy(result)
    )
    assert contract.validate_latents(tmp_path, "combined", "staged", 42) == result
    metadata[key] = str(metadata[key])
    with pytest.raises(ValueError, match="latent"):
        contract.validate_latents(tmp_path, "combined", "staged", 42)

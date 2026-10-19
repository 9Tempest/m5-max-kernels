"""CPU contracts for resident native sessions; never launches Metal."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def native(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    for name in ("run", "run_session", "run_fc2_experiment", "run_fused_campaign"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "h3_native_session_test", directory / "run_native_session.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args_for(native, tmp_path, *extra):
    return native.parser().parse_args(
        ["--mode", "hadamard-fc2", "--output-dir", str(tmp_path / "run"), *extra]
    )


def valid_events(native, args):
    recipe = dict(
        event="recipe",
        mode=args.mode,
        width=512,
        height=512,
        frames=124,
        fps=24,
        steps=20,
        seed=args.seed,
        prompt=native.base.PROMPT,
        layers=50,
        reuse=1,
        core_reuse=1,
        use_reference_rope=True,
        warmups=1,
        repetitions=3,
        upstream_revision=native.base.UPSTREAM_REVISION,
        model_revision=native.base.MODEL_REVISION,
        fusion=args.fusion,
    )
    events = [
        recipe,
        dict(event="runtime_flags", **native.runtime_flags(args)),
        dict(event="startup", status="completed", wall_seconds=0.01),
        dict(event="device", name="Apple M5 Max", physical_memory_bytes=128 * 2**30),
    ]
    for index in range(4):
        name = f"{'timed' if index else 'warmup'}-{index}"
        events.append(
            dict(
                event="generation",
                index=index,
                warmup=index == 0,
                status="completed",
                output=name + ".mp4",
                profile_log=name + ".log",
                wall_seconds=150.0,
                denoise_callback_seconds=120.0,
                requested_seed=args.seed,
                result_seed=args.seed,
                denoise_start_phase="denoise enqueue",
                denoise_end_phase="denoise",
                denoise_total=20,
                denoise_completed=20,
                denoise_completion_events=1,
                denoise_start_offset_seconds=1.0,
                denoise_end_offset_seconds=121.0,
                cache_before=dict(
                    prepared_dit=index > 0,
                    video_decoder=index > 0,
                    embedding_entries=int(index > 0),
                ),
                cache_after=dict(prepared_dit=True, video_decoder=True, embedding_entries=1),
            )
        )
    return events + [dict(event="finished", exit_code=0)]


def test_accepts_complete_resident_contract(native, tmp_path):
    args = args_for(native, tmp_path)
    assert len(native.validate_events(valid_events(native, args), args)) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("denoise_callback_seconds", None),
        ("denoise_callback_seconds", float("nan")),
        ("denoise_callback_seconds", 151),
        ("denoise_callback_seconds", 0),
        ("wall_seconds", float("inf")),
        ("warmup", True),
        ("output", "../outside.mp4"),
        ("profile_log", "../outside.log"),
        ("status", "unsupported"),
    ],
)
def test_rejects_invalid_generation(native, tmp_path, field, value):
    args = args_for(native, tmp_path)
    events = valid_events(native, args)
    events[5][field] = value
    with pytest.raises((ValueError, TypeError)):
        native.validate_events(events, args)


def test_requires_resident_cache_and_exact_fusion(native, tmp_path):
    args = args_for(native, tmp_path)
    events = valid_events(native, args)
    bad_cache = copy.deepcopy(events)
    bad_cache[5]["cache_before"]["prepared_dit"] = False
    with pytest.raises(ValueError, match="cache"):
        native.validate_events(bad_cache, args)
    events[0]["fusion"] = "both"
    with pytest.raises(ValueError, match="fusion policy"):
        native.validate_events(events, args)


def test_disallows_confounded_fusion_comparison_before_gpu(native, tmp_path):
    args = args_for(native, tmp_path, "--mode", "bf16", "--fusion", "both")
    result = native.execute(args)
    assert result["status"] == "failed"
    assert "policy fixed" in result["error"]
    assert (args.output_dir / "run.json").exists()
    with pytest.raises(FileExistsError):
        native.execute(args)


def test_rejects_negative_seed_before_gpu(native, tmp_path):
    args = args_for(native, tmp_path, "--seed", "-1")
    result = native.execute(args)
    assert result["status"] == "failed"
    assert "seed" in result["error"]


def test_rejects_inherited_overrides(native, tmp_path, monkeypatch):
    monkeypatch.setenv("H3_FUSE_FC2_GATE", "1")
    result = native.execute(args_for(native, tmp_path))
    assert result["status"] == "failed"
    assert "H3_FUSE_FC2_GATE" in result["error"]


@pytest.mark.parametrize("fusion", ["none", "fc2", "attention", "both"])
def test_frozen_fusion_source_accepted_including_disabled_control(
    native, tmp_path, monkeypatch, fusion
):
    provenance = dict(source_sha256=native.fused.SOURCE_SHA256)
    monkeypatch.setattr(native.base, "git_provenance", lambda _: provenance)
    monkeypatch.setattr(native.fused, "validate_source", lambda source, value: {"checked": True})
    assert native.validate_source(tmp_path, fusion) == (provenance, {"checked": True})


def test_original_frozen_h_source_still_default_none(native, tmp_path, monkeypatch):
    provenance = dict(source_sha256=native.experiment.SOURCE_SHA256)
    monkeypatch.setattr(native.base, "git_provenance", lambda _: provenance)
    monkeypatch.setattr(
        native.experiment, "validate_source", lambda source, value: {"original": True}
    )
    assert native.validate_source(tmp_path, "none") == (provenance, {"original": True})
    with pytest.raises(native.base.UnsupportedError, match="exact frozen"):
        native.validate_source(tmp_path, "both")
    assert (
        native.parser()
        .parse_args(["--mode", "hadamard-fc2", "--output-dir", "unused"])
        .source_dir.name
        == "h3.c-hadamard"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("requested_seed", 43),
        ("result_seed", 43),
        ("result_seed", True),
        ("denoise_start_phase", "denoise"),
        ("denoise_end_phase", "denoise enqueue"),
        ("denoise_total", 19),
        ("denoise_completed", 19),
        ("denoise_completion_events", 2),
        ("denoise_start_offset_seconds", -1),
        ("denoise_end_offset_seconds", 151),
        ("denoise_end_offset_seconds", 120),
        ("denoise_start_offset_seconds", True),
    ],
)
def test_seed_and_native_callback_boundaries_are_verified(native, tmp_path, field, value):
    args = args_for(native, tmp_path)
    events = valid_events(native, args)
    events[5][field] = value
    with pytest.raises(ValueError):
        native.validate_events(events, args)


def test_actual_process_flags_must_match_recipe(native, tmp_path):
    args = args_for(native, tmp_path, "--fusion", "both")
    events = valid_events(native, args)
    events[1]["H3_FUSE_FC2_GATE"] = "0"
    with pytest.raises(ValueError, match="actual process"):
        native.validate_events(events, args)


def test_init_markers_and_cache_hits_prove_resident_policy(native, tmp_path):
    args = args_for(native, tmp_path, "--fusion", "both")
    warmup = "\n".join(
        [
            "h3 precision: FC2 plain normalized H256",
            "h3 fusion: FC2 GEMM gate/residual epilogue",
            "h3 fusion: attention output GEMM gate/residual epilogue",
        ]
    )
    observed = native.validate_policy_log(warmup, 0, args)
    assert observed["initialization_marker_counts"] == dict(
        hadamard_fc2=1, fc2_gate=1, attention_output_gate=1
    )
    hits = "\n".join(
        ["h3: prepared DiT cache hit", "h3: video VAE cache hit", "h3: conditioning cache hit"]
    )
    assert native.validate_policy_log(hits, 1, args)["cache_hit_counts"] == dict(
        prepared_dit=1, video_decoder=1, conditioning=1
    )
    for log, index in (("", 0), (warmup, 1), (hits, 0), (hits.replace("video VAE", "missing"), 1)):
        with pytest.raises(ValueError):
            native.validate_policy_log(log, index, args)
    none = args_for(native, tmp_path)
    with pytest.raises(native.base.UnsupportedError):
        native.validate_policy_log(warmup, 0, none)


def test_exact_harness_snapshot_is_exclusive_and_bound(native, tmp_path, monkeypatch):
    entries = native.snapshot_harness(tmp_path)
    native.validate_snapshot(entries)
    original = Path(native.__file__).with_name("session.c").read_bytes()
    assert (tmp_path / "harness-snapshot/session.c").read_bytes() == original
    assert len(entries) >= 12
    with pytest.raises(FileExistsError):
        native.snapshot_harness(tmp_path)
    real_hash = native.base.sha256
    monkeypatch.setattr(
        native.base,
        "sha256",
        lambda path: "changed" if Path(path).name == "session.c" else real_hash(path),
    )
    with pytest.raises(ValueError, match="changed after"):
        native.validate_snapshot(entries)

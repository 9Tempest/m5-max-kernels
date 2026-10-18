"""CPU-only checks for the evidence wrapper; these never launch H3 or Metal."""

import importlib.util
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


MODULE = Path(__file__).resolve().parents[1] / "integrations/h3/run.py"
spec = importlib.util.spec_from_file_location("h3_integration", MODULE)
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)


def arguments(tmp_path, *extra):
    return h3.parser().parse_args(
        [
            "--mode",
            "bf16",
            "--output-dir",
            str(tmp_path / "run"),
            *extra,
        ]
    )


def test_five_second_alignment_keeps_original_and_exact_target_distinct():
    shape = h3.temporal_shape(5, 512, 512)
    assert shape["requested_frames"] == 120
    assert shape["aligned_frames"] == 124
    assert shape["video_latent_t"] == 37
    assert shape["video_tokens"] == 9472
    assert shape["stereo_audio_tokens"] == 414
    assert shape["text_tokens"] is None
    assert shape["packed_sequence_tokens"] is None


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), 0.01])
def test_invalid_durations_are_not_silently_rounded(seconds):
    with pytest.raises(ValueError):
        h3.temporal_shape(seconds, 512, 512)


def test_modes_have_fixed_full_model_recipe(tmp_path):
    args = arguments(tmp_path)
    command = h3.build_command(args, tmp_path, tmp_path, tmp_path / "original.mp4")
    assert command[command.index("--layers") + 1] == "50"
    assert command[command.index("--reuse") + 1] == "1"
    assert command[command.index("--core-reuse") + 1] == "1"
    assert "--use-reference-rope" in command
    assert all(flag in command for flag in h3.BF16_FLAGS)
    assert "--token-reduction" not in command
    args.mode = "upstream-int8"
    command = h3.build_command(args, tmp_path, tmp_path, tmp_path / "original.mp4")
    assert all(flag not in command for flag in h3.BF16_FLAGS)
    assert "--use-int8-row-fc2" not in command


def test_inherited_runtime_shortcuts_are_rejected(monkeypatch):
    monkeypatch.setenv("H3_TOKEN_REDUCTION", "1")
    with pytest.raises(ValueError, match="H3_TOKEN_REDUCTION"):
        h3.runtime_environment()


def test_profile_preserves_overlapping_phases_and_source_counters():
    log = (
        "h3 profile: H3 DiT                   load           wall=  10.001s encode= 0.800s "
        "wait=   2.000s root-gpu= 1.900s peak= 30.123GiB alloc= 45.333GiB "
        "submissions=122 direct=700 linear=20 conv=0 attention=2\n"
        "h3 profile: H3 DiT                   GPU Euler denoise wall=  40.500s "
        "root-gpu= 37.000s submissions=1000\n"
        "h3 profile: resident video VAE decoder total          wall=   8.000s\n"
    )
    records = h3.parse_profile(log)
    assert len(records) == 3
    assert records[0]["component"] == "H3 DiT"
    assert records[0]["phase"] == "load"
    assert records[0]["wall_s"] == 10.001
    assert records[0]["peak_GiB"] == 30.123
    assert records[0]["submissions"] == 122
    assert records[1]["phase"] == "GPU Euler denoise"
    assert records[2]["component"] == "resident video VAE decoder"
    assert records[2]["phase"] == "total"
    assert h3.parse_profile("h3 profile: unavailable") == [
        {"raw": "h3 profile: unavailable", "parse_status": "unrecognized"}
    ]


def test_progress_parses_carriage_return_updates():
    progress = h3.parse_progress("\rtext encoder                 0/50  \rtext encoder 50/50\n")
    assert [item["completed"] for item in progress] == [0, 50]


def valid_probe():
    return {
        "streams": [
            dict(
                codec_type="video",
                width=512,
                height=512,
                avg_frame_rate="24/1",
                nb_read_frames="120",
                duration="5.000000",
            )
        ]
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 256),
        ("height", 256),
        ("avg_frame_rate", "30/1"),
        ("nb_read_frames", "124"),
        ("duration", "5.166667"),
        ("duration", "nan"),
    ],
)
def test_bad_export_structure_is_rejected(field, value):
    probe = valid_probe()
    probe["streams"][0][field] = value
    with pytest.raises(ValueError):
        h3.validate_video(probe, 512, 512, 120)


def test_correct_exact_five_second_video_structure():
    assert h3.validate_video(valid_probe(), 512, 512, 120)["duration_seconds"] == 5


def test_raw_records_are_exclusive(tmp_path):
    path = tmp_path / "raw.json"
    h3.write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        h3.write_json(path, {"overwritten": True})
    assert json.loads(path.read_text()) == {"original": True}


def test_bad_request_retains_failure_without_launching_gpu(tmp_path):
    args = arguments(tmp_path, "--width", "255")
    record = h3.execute(args)
    assert record["status"] == "failed"
    assert "multiples of 32" in record["error"]
    assert json.loads((args.output_dir / "run.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        h3.execute(args)


def test_non_metal_host_is_unavailable_never_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        h3, "git_provenance", lambda _: dict(revision=h3.UPSTREAM_REVISION, dirty=False)
    )
    monkeypatch.setattr(h3, "sha256", lambda _: "digest")
    monkeypatch.setattr(h3, "runtime_environment", lambda: ({}, {}))
    monkeypatch.setattr(h3.platform, "system", lambda: "Linux")
    record = h3.execute(arguments(tmp_path))
    assert record["status"] == "unsupported"
    assert "unavailable" in record["error"]
    assert not record["measurements"]


def test_missing_models_fail_before_hashing_or_runtime(tmp_path):
    with pytest.raises(ValueError, match="model directory is missing"):
        h3.model_inventory(tmp_path / "missing", tmp_path / "manifest.json")


def test_truncated_model_is_rejected_before_runtime(tmp_path, monkeypatch):
    entries = []
    paths = ["FL2VA/transformer/config.json", "FL2VA/tokenizer/tokenizer.json"] + [
        f"FL2VA/{component}/model.safetensors"
        for component in ("text_encoder", "transformer", "video_vae/source", "audio_vae")
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"truncated")
        entries.append(dict(type="file", path=name, size=999))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    monkeypatch.setattr(h3, "MODEL_MANIFEST_SHA256", h3.sha256(manifest))
    with pytest.raises(ValueError, match="size mismatch"):
        h3.model_inventory(tmp_path, manifest)


@pytest.fixture
def model_fixture(tmp_path, monkeypatch):
    paths = ["FL2VA/transformer/config.json", "FL2VA/tokenizer/tokenizer.json"] + [
        f"FL2VA/{component}/model.safetensors"
        for component in ("text_encoder", "transformer", "video_vae/source", "audio_vae")
    ]
    entries = []
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b"test data"
        path.write_bytes(data)
        entries.append(
            dict(
                type="file",
                path=name,
                size=len(data),
                oid=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
            )
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    monkeypatch.setattr(h3, "MODEL_MANIFEST_SHA256", h3.sha256(manifest))
    return tmp_path, manifest


@pytest.mark.parametrize(
    "component", ["text_encoder", "transformer", "video_vae/source", "audio_vae"]
)
def test_unlisted_runtime_shards_cannot_shadow_verified_weights(model_fixture, component):
    model, manifest = model_fixture
    assert len(h3.model_inventory(model, manifest)["files"]) == 6
    (model / "FL2VA" / component / "000-shadow.safetensors").write_bytes(b"unlisted weights")
    with pytest.raises(ValueError, match="unexpected model weight shards"):
        h3.model_inventory(model, manifest)


def test_arbitrary_manifest_cannot_be_stamped_as_the_pinned_revision(model_fixture):
    model, manifest = model_fixture
    replacement = model / "replacement.json"
    replacement.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest SHA256"):
        h3.model_inventory(model, replacement)


def test_checked_in_model_manifest_matches_revision_pin():
    assert h3.sha256(MODULE.with_name("model-files.json")) == h3.MODEL_MANIFEST_SHA256


def test_cpu_subprocess_failure_preserves_bytes_and_timestamps(tmp_path):
    command = [sys.executable, "-c", "import sys; print('failure detail'); sys.exit(3)"]
    result = h3.run_logged(command, tmp_path, os.environ.copy(), tmp_path, "cpu")
    assert result["returncode"] == 3
    assert (tmp_path / "cpu.log").read_bytes() == b"failure detail\n"
    events = [json.loads(line) for line in (tmp_path / "cpu.events.jsonl").read_text().splitlines()]
    assert sum(event["byte_count"] for event in events) == 15
    assert all(event["elapsed_seconds"] >= 0 for event in events)
    assert result["log_sha256"] == h3.sha256(tmp_path / "cpu.log")


def test_upstream_compile_fallback_is_explicit(tmp_path):
    command = [sys.executable, "-c", "print('h3: TensorOps compile failed: example', flush=True)"]
    result = h3.run_logged(command, tmp_path, os.environ.copy(), tmp_path, "fallback", True)
    assert result["fallback_detected"]
    assert "compile failed" in (tmp_path / "fallback.log").read_text()


@pytest.fixture
def session_wrapper(monkeypatch):
    monkeypatch.setitem(sys.modules, "run", h3)
    spec = importlib.util.spec_from_file_location("h3_session", MODULE.with_name("run_session.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_summary_reports_raw_samples_and_interpolated_percentiles(session_wrapper):
    summary = session_wrapper.summarize([1.0, 2.0, 5.0])
    assert summary["raw_seconds"] == [1.0, 2.0, 5.0]
    assert summary["median_seconds"] == 2
    assert summary["p10_seconds"] == pytest.approx(1.2)
    assert summary["p90_seconds"] == pytest.approx(4.4)
    with pytest.raises(ValueError):
        session_wrapper.summarize([1, float("nan"), 2])


def test_session_bad_recipe_retains_failure_before_driver_launch(tmp_path, session_wrapper):
    args = arguments(tmp_path, "--seconds", "2")
    args.driver = tmp_path / "driver-does-not-exist"
    record = session_wrapper.execute(args)
    assert record["status"] == "failed"
    assert "fixed 5-second" in record["error"]
    assert json.loads((args.output_dir / "run.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        session_wrapper.execute(args)


def test_session_rejects_diagnostics_before_driver_or_gpu(tmp_path, monkeypatch, session_wrapper):
    args = arguments(tmp_path, "--diagnostics")
    args.driver = tmp_path / "driver-does-not-exist"

    def unexpected_preflight(*_args, **_kwargs):
        pytest.fail("session diagnostics must be rejected before preflight or process launch")

    monkeypatch.setattr(h3, "git_provenance", unexpected_preflight)
    monkeypatch.setattr(h3, "run_logged", unexpected_preflight)
    record = session_wrapper.execute(args)
    assert record["status"] == "failed"
    assert "diagnostics are only supported by the one-shot" in record["error"]
    assert record["measurements"] == {}
    assert json.loads((args.output_dir / "run.json").read_text())["status"] == "failed"


def test_session_recovers_text_shape_from_observed_public_cache_bytes(session_wrapper):
    generations = [dict(cache_after=dict(embedding_bytes=82 * 10241))] * 4
    shape = session_wrapper.shape_from_cache(
        h3.temporal_shape(5, 512, 512), generations, "upstream-int8"
    )
    assert shape["text_tokens"] == 82
    assert shape["packed_sequence_tokens"] == 9968
    assert shape["int8_activation_arena_rows"] == 9984
    with pytest.raises(ValueError, match="inconsistent"):
        session_wrapper.shape_from_cache(
            shape, [dict(cache_after=dict(embedding_bytes=82 * 10241 + 1))], "upstream-int8"
        )


def session_events(args):
    return [
        dict(
            event="recipe",
            mode=args.mode,
            width=args.width,
            height=args.height,
            frames=124,
            fps=24,
            steps=args.steps,
            seed=args.seed,
            prompt=h3.PROMPT,
            layers=50,
            reuse=1,
            core_reuse=1,
            use_reference_rope=True,
            warmups=1,
            repetitions=3,
            upstream_revision=h3.UPSTREAM_REVISION,
            model_revision=h3.MODEL_REVISION,
        ),
        dict(event="finished", exit_code=0),
    ]


def test_session_recipe_accepts_matching_driver_contract(tmp_path, session_wrapper):
    args = arguments(tmp_path)
    session_wrapper.validate_session_recipe(session_events(args), args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "upstream-int8"),
        ("width", 256),
        ("height", 256),
        ("frames", 120),
        ("fps", 30),
        ("steps", 4),
        ("seed", 43),
        ("prompt", "different prompt"),
        ("layers", 45),
        ("reuse", 2),
        ("core_reuse", 4),
        ("use_reference_rope", False),
        ("warmups", 0),
        ("repetitions", 1),
        ("upstream_revision", "other"),
        ("model_revision", "other"),
        ("seed", 42.0),
        ("use_reference_rope", 1),
    ],
)
def test_session_recipe_mismatch_is_rejected(tmp_path, session_wrapper, field, value):
    args = arguments(tmp_path)
    events = session_events(args)
    events[0][field] = value
    with pytest.raises(ValueError, match="recipe differs"):
        session_wrapper.validate_session_recipe(events, args)


@pytest.mark.parametrize("recipes", [0, 2])
def test_session_requires_exactly_one_recipe(tmp_path, session_wrapper, recipes):
    args = arguments(tmp_path)
    recipe, finished = session_events(args)
    with pytest.raises(ValueError, match="exactly one driver recipe"):
        session_wrapper.validate_session_recipe([recipe] * recipes + [finished], args)


@pytest.mark.parametrize("case", ["missing", "duplicate", "failure", "not-last", "boolean"])
def test_session_requires_one_successful_terminal_event(tmp_path, session_wrapper, case):
    args = arguments(tmp_path)
    events = session_events(args)
    if case == "missing":
        events.pop()
    elif case == "duplicate":
        events.append(events[-1].copy())
    elif case == "failure":
        events[-1]["exit_code"] = 1
    elif case == "not-last":
        events.append(dict(event="generation"))
    else:
        events[-1]["exit_code"] = False
    with pytest.raises(ValueError, match="finished event"):
        session_wrapper.validate_session_recipe(events, args)


@pytest.fixture
def simulated_lock_clock(monkeypatch):
    state = dict(now=0.0, sleeps=[], attempts=0, acquired=False, releases=0)
    monkeypatch.setattr(h3.time, "monotonic", lambda: state["now"])

    def sleep(seconds):
        assert not state["acquired"]
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(h3.time, "sleep", sleep)
    return state


def test_gpu_queue_wait_acquires_once_and_holds_through_body(monkeypatch, simulated_lock_clock):
    state = simulated_lock_clock

    @contextmanager
    def lock():
        state["attempts"] += 1
        if state["attempts"] <= 2:
            raise ValueError("another m5bench process owns the GPU experiment lock")
        state["acquired"] = True
        try:
            yield
        finally:
            state["acquired"] = False
            state["releases"] += 1

    monkeypatch.setattr(h3, "gpu_lock", lock)
    record = {}
    with h3.wait_for_gpu_lock(5, record):
        assert state["acquired"] and state["releases"] == 0
        assert record["gpu_queue_wait_seconds"] == 2
        state["now"] += 10  # Simulated generation/media work occurs after acquisition.
        assert state["acquired"]
    assert state["attempts"] == 3
    assert state["releases"] == 1
    assert record["gpu_queue_wait_seconds"] == 2


@pytest.mark.parametrize("timeout,sleeps", [(0, []), (2.5, [1.0, 1.0, 0.5])])
def test_gpu_queue_timeout_is_bounded_and_preserves_busy_error(
    monkeypatch, simulated_lock_clock, timeout, sleeps
):
    state = simulated_lock_clock

    @contextmanager
    def busy():
        state["attempts"] += 1
        raise ValueError("another m5bench process owns the GPU experiment lock")
        yield

    monkeypatch.setattr(h3, "gpu_lock", busy)
    record = {}
    with pytest.raises(ValueError, match="another m5bench process"):
        with h3.wait_for_gpu_lock(timeout, record):
            pytest.fail("cannot enter the body without owning the GPU lock")
    assert state["sleeps"] == sleeps
    assert record["gpu_queue_wait_seconds"] == timeout


def test_gpu_queue_does_not_retry_execution_value_errors(monkeypatch, simulated_lock_clock):
    state = simulated_lock_clock

    @contextmanager
    def lock():
        state["attempts"] += 1
        state["acquired"] = True
        try:
            yield
        finally:
            state["acquired"] = False
            state["releases"] += 1

    monkeypatch.setattr(h3, "gpu_lock", lock)
    with pytest.raises(ValueError, match="another m5bench process"):
        with h3.wait_for_gpu_lock(5, {}):
            # Even the same text from execution must escape rather than retry.
            raise ValueError("another m5bench process owns the GPU experiment lock")
    assert state["attempts"] == state["releases"] == 1
    assert state["sleeps"] == []


def test_gpu_queue_does_not_retry_other_acquisition_errors(monkeypatch, simulated_lock_clock):
    state = simulated_lock_clock

    @contextmanager
    def broken():
        raise ValueError("different lock failure")
        yield

    monkeypatch.setattr(h3, "gpu_lock", broken)
    with pytest.raises(ValueError, match="different lock failure"):
        with h3.wait_for_gpu_lock(5, {}):
            pytest.fail("must not acquire")
    assert state["sleeps"] == []


def test_session_rejects_nonzero_gpu_wait_before_launch(tmp_path, session_wrapper):
    args = arguments(tmp_path, "--gpu-wait-seconds", "5")
    args.driver = tmp_path / "driver-does-not-exist"
    record = session_wrapper.execute(args)
    assert record["status"] == "failed"
    assert "session driver owns its lock" in record["error"]
    assert record["measurements"] == {}

"""Independent evidence fixtures and failure gates for resident publication."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys

import pytest


@pytest.fixture
def compare(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    for name in (
        "run",
        "run_session",
        "run_native_session",
        "run_fused_campaign",
        "publish_samples",
        "publish_resident",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "resident_comparison_test", directory / "compare_resident.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def save(path, record, module):
    events = path.parent / "session/session.jsonl"
    events.write_text("\n".join(json.dumps(e) for e in record["session_events"]) + "\n")
    record["session_events_sha256"] = module.base.sha256(events)
    path.write_text(json.dumps(record))


def fixture(directory, module, fusion, start):
    directory.mkdir()
    (directory / "session").mkdir()
    snapshot = directory / "harness-snapshot"
    snapshot.mkdir()
    source = Path(module.__file__).parent
    entries = {}
    for name in module.HARNESS_SHA256:
        original = (
            source / name
            if name != "locking.py"
            else source.parents[1] / "src/m5kernels/locking.py"
        )
        shutil.copyfile(original, snapshot / name)
        entries[name] = dict(
            file="harness-snapshot/" + name,
            original="fixture/" + name,
            sha256=module.base.sha256(snapshot / name),
        )
    (snapshot / "manifest.json").write_text(json.dumps(entries))
    # Different executable bytes are permitted: each is bound to its own digest.
    (directory / "native-session").write_bytes(b"fixture-executable-" + fusion.encode())
    record = dict(
        schema_version=2,
        status="completed",
        benchmark="fox-winter-t2va-cached-v1",
        prompt="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind.",
        mode="hadamard-fc2",
        fusion=fusion,
        seed=42,
        width=512,
        height=512,
        steps=20,
        source_frames=124,
        warmups=1,
        repetitions=3,
        active_dit_blocks=50,
        reuse=1,
        core_reuse=1,
        use_reference_rope=True,
        started_utc=stamp(start - 10),
        finished_utc=stamp(start + 700),
        timing_contract=dict(
            cached_generation="decode/export included",
            cached_denoise_callback="synchronized callback",
            compile_free_e2e="not claimed",
        ),
        harness=dict(revision="document-revision-" + fusion, dirty=fusion == "attention"),
        harness_snapshot=entries,
        harness_files={k: v["sha256"] for k, v in entries.items()},
        upstream=dict(
            revision="8974cc055ea9c02fcd14cc27dfda3e1027c05153",
            dirty=True,
            status="M h3_dit.c",
            source_sha256=module.SOURCE_SHA256,
        ),
        patch=dict(source_sha256=module.SOURCE_SHA256, patch_sha256=module.PATCH_SHA256),
        model=dict(
            repository="MiniMaxAI/MiniMax-H3",
            revision="42ed227ee7df40d41602854ae760620d6eb651fe",
            manifest_sha256="7597d33ca71af128705e802150b4b39395cf20f96bddc45d2041d668145c27be",
        ),
        driver=dict(
            source_sha256=entries["session.c"]["sha256"],
            wrapper_sha256=entries["run_native_session.py"]["sha256"],
            binary_sha256=module.base.sha256(directory / "native-session"),
            library_sha256=("a" if fusion == "none" else "b") * 64,
        ),
        environment=dict(
            system="Darwin",
            release="27.0.0",
            machine="arm64",
            python="3.12.15",
            chip="Apple M5 Max",
            memory_bytes=128 * 2**30,
            compiler="Apple clang test\nInstalledDir: /Applications/Xcode.app",
            xcode="Xcode 27.1",
            ffprobe="ffprobe 9.0.2",
        ),
        all_repetition_videos_byte_identical=True,
        videos=[],
        profiles=[],
        policy_evidence=[],
        measurements={},
    )
    flags = dict(
        H3_HADAMARD_FC2="1",
        H3_FUSE_FC2_GATE="unset" if fusion == "none" else "0",
        H3_FUSE_ATTN_OUT_GATE="unset" if fusion == "none" else "1",
    )
    record.update(
        requested_runtime_flags=flags.copy(),
        observed_runtime_flags=flags.copy(),
        runtime_overrides=dict(
            H3_NAX="qkv-attn",
            H3_NAX_DIAGNOSTIC="1",
            H3_PROFILE="1",
            **{k: v for k, v in flags.items() if v != "unset"},
        ),
    )
    recipe = {
        k: record[k]
        for k in (
            "mode",
            "fusion",
            "prompt",
            "seed",
            "width",
            "height",
            "steps",
            "warmups",
            "repetitions",
            "reuse",
            "core_reuse",
            "use_reference_rope",
        )
    }
    recipe.update(
        event="recipe",
        frames=124,
        layers=50,
        fps=24,
        upstream_revision=record["upstream"]["revision"],
        model_revision=record["model"]["revision"],
    )
    events = [
        recipe,
        dict(event="runtime_flags", **flags),
        dict(event="startup", status="completed", wall_seconds=0.25),
        dict(
            event="device",
            name="Apple M5 Max",
            physical_memory_bytes=128 * 2**30,
            registry_id="private-device-id",
        ),
    ]
    walls = [180.0, 153.0, 152.0, 151.0] if fusion == "none" else [179.0, 151.0, 150.0, 149.0]
    denoises = [125.0, 123.0, 122.0, 121.0] if fusion == "none" else [124.0, 121.0, 120.0, 119.0]
    elapsed = 1
    for i, (wall, denoise) in enumerate(zip(walls, denoises, strict=True)):
        stem = f"{'warmup' if i == 0 else 'timed'}-{i}"
        (directory / f"session/{stem}.mp4").write_bytes(b"identical-fixture-video")
        markers = (
            ["h3 precision: FC2 plain normalized H256"]
            + (
                ["h3 fusion: attention output GEMM gate/residual epilogue"]
                if fusion == "attention"
                else []
            )
            if i == 0
            else [
                "h3: prepared DiT cache hit",
                "h3: video VAE cache hit",
                "h3: conditioning cache hit",
            ]
        )
        log = "\n".join(markers) + "\n"
        (directory / f"session/{stem}.log").write_text(log)
        events.append(
            dict(
                event="generation",
                index=i,
                warmup=i == 0,
                status="completed",
                started_unix_seconds=start + elapsed,
                output=stem + ".mp4",
                profile_log=stem + ".log",
                wall_seconds=wall,
                denoise_callback_seconds=denoise,
                requested_seed=42,
                result_seed=42,
                denoise_start_phase="denoise enqueue",
                denoise_end_phase="denoise",
                denoise_total=20,
                denoise_completed=20,
                denoise_completion_events=1,
                denoise_start_offset_seconds=1.0,
                denoise_end_offset_seconds=1.0 + denoise,
                cache_before=dict(
                    prepared_dit=int(i > 0),
                    video_decoder=int(i > 0),
                    embedding_entries=int(i > 0),
                    embedding_bytes=286748 if i else 0,
                ),
                cache_after=dict(
                    prepared_dit=1, video_decoder=1, embedding_entries=1, embedding_bytes=286748
                ),
            )
        )
        elapsed += int(wall) + 1
        record["videos"].append(
            dict(
                file=f"session/{stem}.mp4",
                sha256=module.base.sha256(directory / f"session/{stem}.mp4"),
                width=512,
                height=512,
                frames=124,
                fps=24,
                duration_seconds=5.166667,
            )
        )
        record["profiles"].append(
            dict(
                index=i, log_sha256=module.base.sha256(directory / f"session/{stem}.log"), phases=[]
            )
        )
        record["policy_evidence"].append(
            dict(
                index=i,
                enabled=dict(
                    hadamard_fc2=True, fc2_gate=False, attention_output_gate=fusion == "attention"
                ),
                initialization_marker_counts=dict(
                    hadamard_fc2=int(i == 0),
                    fc2_gate=0,
                    attention_output_gate=int(i == 0 and fusion == "attention"),
                ),
                cache_hit_counts=dict(
                    prepared_dit=int(i > 0), video_decoder=int(i > 0), conditioning=int(i > 0)
                ),
                evidence="initialization markers on warmup; fixed process environment and cache hits thereafter",
            )
        )
    events.append(dict(event="finished", exit_code=0))
    record["session_events"] = events
    record["shape"] = dict(
        fps=24,
        requested_seconds=5,
        requested_frames=120,
        aligned_frames=124,
        aligned_seconds=124 / 24,
        video_latent_t=37,
        audio_latent_t=207,
        video_tokens=9472,
        stereo_audio_tokens=414,
        text_tokens=28,
        packed_sequence_tokens=9914,
        int8_activation_arena_rows=9984,
        shape_evidence="text tokens derived from observed public cache bytes and pinned h3_cache_get_info formula; video/audio/layout from h3_host.c; INT8 arena allocation rounding from h3_dit.c; not a per-op tensor trace",
    )
    record["warmup_seconds"] = walls[0]
    for name, values in (
        ("cached_generation", walls[1:]),
        ("cached_denoise_callback", denoises[1:]),
    ):
        record[name] = dict(raw_seconds=values, count=3, median_seconds=statistics.median(values))
    for key, stem, command in (
        ("build", "build", ["make", "-B", "-j8"]),
        (
            "driver_build",
            "driver-build",
            [
                "clang",
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-D_DARWIN_C_SOURCE",
                "-DH3_SESSION_EXPERIMENT=1",
                "-I",
                ".cache/h3.c-fused",
                str(snapshot / "session.c"),
                ".cache/h3.c-fused/libh3.a",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-framework",
                "MetalPerformanceShaders",
                "-framework",
                "MetalPerformanceShadersGraph",
                "-framework",
                "Accelerate",
                "-licucore",
                "-lm",
                "-o",
                str(directory / "native-session"),
            ],
        ),
        ("session_process", "session-process", ["fixture"]),
    ):
        (directory / f"{stem}.log").write_text("fixture process\n")
        record["measurements"][key] = dict(
            command=command,
            cwd=".cache/h3.c-fused",
            returncode=0,
            fallback_detected=False,
            log=stem + ".log",
            log_sha256=module.base.sha256(directory / f"{stem}.log"),
            started_utc=stamp(start),
            finished_utc=stamp(start + elapsed + 1),
            wall_seconds=float(elapsed + 1),
        )
    path = directory / "run.json"
    save(path, record, module)
    return path, record


@pytest.fixture
def pair(tmp_path, compare):
    return fixture(tmp_path / "baseline", compare, "none", 1800000000), fixture(
        tmp_path / "candidate", compare, "attention", 1800001000
    )


def test_success_descriptive_statistics_and_sanitized_schema(tmp_path, compare, pair):
    (bp, b), (cp, c) = pair
    result = compare.export(bp, cp, tmp_path / "published")
    assert result["status"] == "completed", result
    assert result["cached_generation"]["median_seconds_saved"] == 2.0
    assert result["cached_generation"]["baseline_median_divided_by_candidate"] == 152 / 150
    assert result["cached_denoise_callback"]["baseline"]["range_seconds"] == 2.0
    assert result["execution_order"] == ["baseline", "candidate"]
    assert result["claims"]["stable_promotion"] == "not_established"
    assert b["driver"]["binary_sha256"] != c["driver"]["binary_sha256"]
    for name in ("baseline", "candidate"):
        output = (tmp_path / f"published/{name}.json").read_text()
        assert json.loads(output)["schema_version"] == 2
        assert "private-device-id" not in output and str(tmp_path) not in output
    with pytest.raises(FileExistsError):
        compare.export(bp, cp, tmp_path / "published")


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "upstream-int8"),
        ("fusion", "both"),
        ("seed", 43),
        ("steps", 19),
        ("active_dit_blocks", 49),
        ("source_frames", 120),
        ("repetitions", 4),
        ("capture_latents", True),
        ("prepared_checkpoint", True),
        ("all_repetition_videos_byte_identical", False),
    ],
)
def test_recipe_and_precision_drift_rejected(compare, pair, field, value):
    (bp, _), (cp, c) = pair
    c[field] = value
    save(cp, c, compare)
    with pytest.raises(ValueError):
        compare.compare(bp, cp)


def test_same_unpinned_source_in_both_arms_rejected(compare, pair):
    for p, r in pair:
        r["patch"]["source_sha256"] = r["upstream"]["source_sha256"] = "9" * 64
        save(p, r, compare)
    with pytest.raises(ValueError, match="frozen fused"):
        compare.compare(pair[0][0], pair[1][0])


@pytest.mark.parametrize(
    "target",
    [
        "harness-snapshot/session.c",
        "native-session",
        "session/timed-1.mp4",
        "session/timed-1.log",
        "session/session.jsonl",
        "session-process.log",
    ],
)
def test_actual_evidence_corruption_rejected(compare, pair, target):
    (bp, _), (cp, _) = pair
    (cp.parent / target).write_bytes(b"corrupted\n")
    with pytest.raises(ValueError):
        compare.compare(bp, cp)


@pytest.mark.parametrize(
    "field,value",
    [
        ("denoise_completed", 19),
        ("denoise_completion_events", 2),
        ("denoise_start_phase", "denoise"),
        ("denoise_end_offset_seconds", 2.0),
        ("result_seed", 43),
        ("warmup", True),
    ],
)
def test_callback_completeness_rejected_after_event_rehash(compare, pair, field, value):
    (bp, _), (cp, c) = pair
    c["session_events"][5][field] = value
    save(cp, c, compare)
    with pytest.raises(ValueError):
        compare.compare(bp, cp)


def test_native_stale_profile_wall_cannot_replace_callback(compare, pair):
    (bp, _), (cp, c) = pair
    c["cached_denoise_callback"].update(raw_seconds=[200.0, 200.0, 200.0], median_seconds=200.0)
    save(cp, c, compare)
    with pytest.raises(ValueError, match="raw timings"):
        compare.compare(bp, cp)


def test_fake_cache_policy_evidence_cannot_override_actual_log(compare, pair):
    (bp, _), (cp, c) = pair
    p = cp.parent / "session/timed-1.log"
    p.write_text("h3: prepared DiT cache hit\n")
    c["profiles"][1]["log_sha256"] = compare.base.sha256(p)
    save(cp, c, compare)
    with pytest.raises(ValueError, match="every resident cache"):
        compare.compare(bp, cp)


@pytest.mark.parametrize(
    "field", ["runtime_overrides", "observed_runtime_flags", "requested_runtime_flags"]
)
def test_control_changes_rejected(compare, pair, field):
    (bp, _), (cp, c) = pair
    c[field]["H3_FUSE_FC2_GATE"] = "1"
    save(cp, c, compare)
    with pytest.raises(ValueError):
        compare.compare(bp, cp)


def test_cross_arm_video_gate_even_when_each_arm_internally_identical(compare, pair):
    (bp, _), (cp, c) = pair
    for v in c["videos"]:
        p = cp.parent / v["file"]
        p.write_bytes(b"different-identical-candidate")
        v["sha256"] = compare.base.sha256(p)
    save(cp, c, compare)
    with pytest.raises(ValueError, match="cross-arm original video"):
        compare.compare(bp, cp)


def test_runtime_snapshot_pin_cannot_be_changed_in_both_declarations(compare, pair):
    for p, r in pair:
        r["harness_files"]["session.c"] = "9" * 64
        r["harness_snapshot"]["session.c"]["sha256"] = "9" * 64
        save(p, r, compare)
    with pytest.raises(ValueError, match="pinned runtime harness"):
        compare.compare(pair[0][0], pair[1][0])


@pytest.mark.parametrize(
    "field,value",
    [("compiler", "other clang"), ("xcode", "other Xcode"), ("memory_bytes", 64 * 2**30)],
)
def test_environment_differences_rejected(compare, pair, field, value):
    (bp, _), (cp, c) = pair
    c["environment"][field] = value
    save(cp, c, compare)
    with pytest.raises(ValueError):
        compare.compare(bp, cp)


def test_changed_compile_flags_rejected_even_if_both_arms_match(compare, pair):
    for p, r in pair:
        r["measurements"]["driver_build"]["command"][2] = "-Ofast"
        save(p, r, compare)
    with pytest.raises(ValueError, match="compiler flags"):
        compare.compare(pair[0][0], pair[1][0])


def test_overlapping_processes_rejected_but_wrapper_queue_overlap_allowed(tmp_path, compare, pair):
    (bp, b), (cp, c) = pair
    b["finished_utc"] = c["finished_utc"]
    save(bp, b, compare)
    assert compare.compare(bp, cp)[0]["status"] == "completed"
    op, _ = fixture(tmp_path / "overlap", compare, "attention", 1800000300)
    with pytest.raises(ValueError, match="process intervals overlap"):
        compare.compare(bp, op)


def test_reversed_process_order_supported(tmp_path, compare, pair):
    (bp, _), _ = pair
    cp, _ = fixture(tmp_path / "earlier", compare, "attention", 1799999000)
    assert compare.compare(bp, cp)[0]["execution_order"] == ["candidate", "baseline"]


def test_rejected_export_retained_without_overwriting_inputs(tmp_path, compare, pair):
    (bp, _), (cp, c) = pair
    c["fusion"] = "fc2"
    save(cp, c, compare)
    before = bp.read_bytes(), cp.read_bytes()
    r = compare.export(bp, cp, tmp_path / "rejected")
    assert (
        r["status"] == "rejected"
        and json.loads((tmp_path / "rejected/comparison.json").read_text()) == r
    )
    assert before == (bp.read_bytes(), cp.read_bytes())
    assert not (tmp_path / "rejected/baseline.json").exists()

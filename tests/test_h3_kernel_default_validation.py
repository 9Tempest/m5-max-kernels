"""CPU evidence rejection fixtures; no media decoding, model loading or GPU work."""

from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/h3"))
import decide_kernel_defaults as defaults


PIN = "a" * 64


def numeric(shape, dtype="float32", data_range=1.0, case="drift", tails=False):
    reference = np.array([1, 2], dtype=dtype)
    candidate = np.array([2, 3], dtype=dtype)
    if case in ("exact", "zero"):
        candidate = reference.copy()
    if case in ("zero", "zero-reference"):
        reference[:] = 0
    if case == "zero":
        candidate[:] = 0
    result = defaults.evaluation.metrics.numeric_metrics(reference, candidate, data_range)
    result.update(shape=shape, elements=math.prod(shape))
    if tails:
        result["absolute_error_quantiles"] = dict.fromkeys(
            ("p50", "p95", "p99", "p999"),
            result["metrics"]["maximum_absolute_error"]["value"],
        )
    return result


def media():
    color = dict.fromkeys(
        (
            "pix_fmt",
            "color_range",
            "color_space",
            "color_transfer",
            "color_primaries",
            "chroma_location",
            "sample_aspect_ratio",
        )
    )
    color["pix_fmt"] = "yuv420p"
    contract = dict(
        width=512,
        height=512,
        fps="24",
        frames=120,
        duration_seconds="5",
        audio_rate=32000,
        audio_channels=2,
        decoded_audio_samples=160000,
        video_pts=[str(Fraction(i, 24)) for i in range(120)],
        audio_pts=["0"],
        audio_frame_samples=[160000],
        color_metadata=color,
        video_frame_color_metadata=[color] * 120,
    )
    return dict(
        schema_version=1,
        status="completed",
        kind="media",
        harness_sha256=defaults.base.sha256(Path(defaults.evaluation.metrics.__file__)),
        files={
            side: dict(
                name="video-5s.mp4", sha256=digest * 64, bytes=100, contract=deepcopy(contract)
            )
            for side, digest in (("reference", "b"), ("candidate", "c"))
        },
        video_rgb=numeric([120, 512, 512, 3], "uint8", 255.0),
        audio_pcm32k_stereo=numeric([160000, 2], "float32", 2.0),
        audio_channels={side: numeric([160000], "float32", 2.0) for side in ("left", "right")},
        optional=dict(
            ssim=dict(
                status="completed",
                backend="ffmpeg ssim, planar RGB gbrp",
                mean=dict(status="finite", value=0.9),
                per_frame=[0.9] * 120,
            )
        ),
    )


@pytest.fixture
def quality(tmp_path):
    report = dict(
        schema=defaults.evaluation.SCHEMA,
        status="completed",
        seed=42,
        source_pin_sha256=PIN,
        evaluator_sha256=defaults.base.sha256(Path(defaults.evaluation.__file__)),
        data_range=1.0,
        baseline_replay_exact=True,
        dependencies={
            name: defaults.base.sha256(Path(defaults.__file__).with_name(name))
            for name in (
                "kernel_integration.py",
                "kernel_evidence.py",
                "evaluate.py",
                "evaluate_latents.py",
                "evaluate_benchmark_pair.py",
                "run_fc2_experiment.py",
                "run.py",
            )
        },
        sources={
            "qkv-only-fused": dict(
                implementation="fused",
                policy="qkv-only",
                seed=42,
                model=defaults.evidence.MODEL,
                source_pin_sha256=PIN,
                run_sha256="d" * 64,
                source_sha256="e" * 64,
                metadata_sha256="f" * 64,
            )
        },
        comparisons={},
    )
    for label in ("old-hadamard", "baseline-fused", "qkv-only-fused"):
        raw = tmp_path / (label + "-media") / "evaluation.json"
        raw.parent.mkdir()
        decoded = media()
        raw.write_text(json.dumps(decoded))
        report["comparisons"][label] = dict(
            against_bf16={
                key: numeric(shape, tails=True)
                for _, key, shape in defaults.evaluation.historical.tensor_spec()
            },
            media=dict(
                evaluation_sha256=defaults.base.sha256(raw),
                video_rgb=decoded["video_rgb"],
                audio_pcm32k_stereo=decoded["audio_pcm32k_stereo"],
                ssim=decoded["optional"]["ssim"],
            ),
        )
    directories = []
    for role in ("bf16", "old_hadamard", "baseline-fused", "qkv-only-fused"):
        directory = tmp_path / (role + "-run")
        directory.mkdir()
        raw = dict(
            **defaults.contract.RECIPE,
            status="completed",
            seed=42,
            model=defaults.evidence.MODEL,
            exact_video=dict(
                file="video-5s.mp4",
                width=512,
                height=512,
                frames=120,
                fps=24,
                duration_seconds=5.0,
                sha256=("b" if role == "bf16" else "c") * 64,
            ),
        )
        raw["mode"] = "bf16" if role == "bf16" else "hadamard-fc2"
        if role not in ("bf16", "old_hadamard"):
            policy = role.removesuffix("-fused")
            raw.update(
                kind="capture",
                policy=policy,
                implementation="fused",
                capture_latents=True,
                diagnostic=True,
                prepared_checkpoint=False,
                requested_runtime_flags=defaults.contract.flags(policy),
                source_pin=dict(sha256=PIN),
                upstream=dict(source_sha256="e" * 64),
                latent_validation=dict(metadata_sha256="f" * 64),
            )
        (directory / "run.json").write_text(json.dumps(raw))
        source = report["sources"].setdefault(
            role, dict(model=defaults.evidence.MODEL, seed=42, source_pin_sha256=PIN)
        )
        source["run_sha256"] = defaults.base.sha256(directory / "run.json")
        if role == "baseline-fused":
            source.update(
                policy="baseline",
                implementation="fused",
                source_sha256="e" * 64,
                metadata_sha256="f" * 64,
            )
        directories.append(directory)
    path = tmp_path / "kernel-evaluation.json"
    path.write_text(json.dumps(report))
    return path, report, defaults.capture_run_index(directories)


def test_complete_numeric_evidence_is_accepted_but_not_perceptual(quality):
    path, _, inputs = quality
    value = defaults.quality_evaluation(path, "qkv-only", PIN, inputs)
    assert value["seed"] == 42
    assert "not perceptual" in value["scope"]


@pytest.mark.parametrize("case", ["exact", "zero", "zero-reference", "drift"])
def test_legal_zero_error_and_zero_reference_statuses_are_retained(case):
    value = numeric([2], case=case, tails=True)
    defaults.numeric_result(value, [2], "float32", 1.0, tails=True)
    if case == "exact":
        assert value["metrics"]["snr_db"]["status"] == "positive_infinity"
        assert value["metrics"]["snr_db"]["value"] is None
    elif case == "zero":
        assert value["metrics"]["relative_l2"]["status"] == "undefined"
    elif case == "zero-reference":
        assert value["metrics"]["snr_db"]["status"] == "negative_infinity"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "null",
        "nan",
        "fake_infinity",
        "fake_zero",
        "bad_shape",
        "nonfinite",
        "tail_none",
        "tail_order",
        "tail_max",
    ],
)
def test_invalid_metrics_cannot_pass_by_status_or_key_presence(mutation):
    value = numeric([2], tails=True)
    if mutation == "missing":
        value.pop("metrics")
    elif mutation == "null":
        value["metrics"]["rmse"] = None
    elif mutation == "nan":
        value["metrics"]["rmse"]["value"] = float("nan")
    elif mutation == "fake_infinity":
        value["metrics"]["snr_db"] = dict(
            status="positive_infinity", value=None, reason="zero error"
        )
    elif mutation == "fake_zero":
        value["exactly_equal"] = True
    elif mutation == "bad_shape":
        value["shape"] = [True, 2]
    elif mutation == "nonfinite":
        value["candidate_nonfinite"] = 1
    elif mutation == "tail_none":
        value["absolute_error_quantiles"]["p99"] = None
    elif mutation == "tail_order":
        value["absolute_error_quantiles"]["p50"] = 2
    elif mutation == "tail_max":
        value["absolute_error_quantiles"] = dict.fromkeys(value["absolute_error_quantiles"], 2)
    with pytest.raises(ValueError):
        defaults.numeric_result(value, [2], "float32", 1.0, tails=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "no_rgb",
        "no_pcm",
        "no_ssim",
        "ssim_short",
        "ssim_nan",
        "ssim_mean",
        "short_audio",
        "bad_pts",
        "different_color",
        "zero_bytes",
    ],
)
def test_incomplete_media_is_not_a_quality_pass(mutation):
    value = media()
    if mutation == "no_rgb":
        value.pop("video_rgb")
    elif mutation == "no_pcm":
        value.pop("audio_pcm32k_stereo")
    elif mutation == "no_ssim":
        value["optional"].pop("ssim")
    elif mutation == "ssim_short":
        value["optional"]["ssim"]["per_frame"].pop()
    elif mutation == "ssim_nan":
        value["optional"]["ssim"]["per_frame"][0] = float("nan")
    elif mutation == "ssim_mean":
        value["optional"]["ssim"]["mean"]["value"] = 0.8
    elif mutation == "short_audio":
        value["files"]["candidate"]["contract"]["audio_frame_samples"] = [159999]
    elif mutation == "bad_pts":
        value["files"]["candidate"]["contract"]["video_pts"][1] = "0"
    elif mutation == "different_color":
        value["files"]["candidate"]["contract"]["color_metadata"]["color_range"] = "tv"
    elif mutation == "zero_bytes":
        value["files"]["candidate"]["bytes"] = 0
    with pytest.raises(ValueError):
        defaults.media_result(value)


@pytest.mark.parametrize(
    "mutation", ["dependency", "seed", "policy", "pin", "source_sha", "missing_tail"]
)
def test_quality_sources_and_dependencies_are_required(quality, mutation):
    path, value, inputs = quality
    source = value["sources"]["qkv-only-fused"]
    if mutation == "dependency":
        value["dependencies"]["evaluate.py"] = "0" * 64
    elif mutation == "seed":
        source["seed"] = 43
    elif mutation == "policy":
        source["policy"] = "combined"
    elif mutation == "pin":
        source["source_pin_sha256"] = "0" * 64
    elif mutation == "source_sha":
        source["source_sha256"] = None
    elif mutation == "missing_tail":
        value["comparisons"]["qkv-only-fused"]["against_bf16"].pop("final_audio")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        defaults.quality_evaluation(path, "qkv-only", PIN, inputs)


def test_original_null_metrics_and_absent_media_reproducer_is_rejected(quality):
    path, value, inputs = quality
    for item in value["comparisons"].values():
        for array in item["against_bf16"].values():
            array["metrics"] = {k: None for k in array["metrics"]}
        item["media"].pop("video_rgb")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        defaults.quality_evaluation(path, "qkv-only", PIN, inputs)


@pytest.mark.parametrize(
    "mutation", ["a_start_bad", "b_end_bad", "a_reverse", "b_reverse", "overlap", "not_utc"]
)
def test_reruns_require_two_valid_nonoverlapping_utc_intervals(monkeypatch, mutation):
    a = dict(
        campaign_sha256="a" * 64,
        environment={},
        harness_files={},
        source_sha256="c" * 64,
        reverse=False,
        cases=[1, 2],
        started_utc="2026-10-18T10:00:00+00:00",
        finished_utc="2026-10-18T11:00:00+00:00",
    )
    b = a | dict(
        campaign_sha256="b" * 64,
        reverse=True,
        cases=[2, 1],
        started_utc="2026-10-18T12:00:00+00:00",
        finished_utc="2026-10-18T13:00:00+00:00",
    )
    if mutation == "a_start_bad":
        a["started_utc"] = "not a timestamp"
    elif mutation == "b_end_bad":
        b["finished_utc"] = "not a timestamp"
    elif mutation == "a_reverse":
        a["started_utc"] = b["finished_utc"]
    elif mutation == "b_reverse":
        b["finished_utc"] = a["finished_utc"]
    elif mutation == "overlap":
        b["started_utc"] = "2026-10-18T10:30:00+00:00"
    elif mutation == "not_utc":
        b["finished_utc"] = "2026-10-18T13:00:00-07:00"
    items = iter([a, b])
    monkeypatch.setattr(defaults, "operator_campaign", lambda *_: next(items))
    with pytest.raises(ValueError):
        defaults.operator_replicates([Path("a"), Path("b")], "qkv", PIN)


def test_quality_requires_raw_capture_run_manifests(quality):
    path, _, _ = quality
    with pytest.raises(ValueError, match="raw capture manifests"):
        defaults.quality_evaluation(path, "qkv-only", PIN)


@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_media_sha_must_match_corresponding_raw_capture_run(quality, side):
    path, report, inputs = quality
    label = "qkv-only-fused"
    raw = path.parent / (label + "-media") / "evaluation.json"
    decoded = json.loads(raw.read_text())
    decoded["files"][side]["sha256"] = "1" * 64
    raw.write_text(json.dumps(decoded))
    report["comparisons"][label]["media"]["evaluation_sha256"] = defaults.base.sha256(raw)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="run join"):
        defaults.quality_evaluation(path, "qkv-only", PIN, inputs)


def test_capture_binding_rejects_missing_or_changed_run(quality):
    path, report, inputs = quality
    key = report["sources"]["qkv-only-fused"]["run_sha256"]
    entry = inputs.pop(key)
    with pytest.raises(ValueError, match="bound raw capture run"):
        defaults.quality_evaluation(path, "qkv-only", PIN, inputs)
    inputs[key] = entry
    entry["path"].write_text("{}")
    with pytest.raises(ValueError, match="unchanged"):
        defaults.quality_evaluation(path, "qkv-only", PIN, inputs)


@pytest.mark.parametrize("field", ["source_sha256", "metadata_sha256"])
def test_quality_source_digests_must_join_bound_raw_run(quality, field):
    path, report, inputs = quality
    report["sources"]["qkv-only-fused"][field] = "1" * 64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="join"):
        defaults.quality_evaluation(path, "qkv-only", PIN, inputs)


@pytest.fixture
def performance_binding(tmp_path):
    arms, evaluations, directories = [], [], []
    for seed in (42, 43):
        bindings = []
        for index, role in enumerate(("bf16", "old_hadamard", "baseline", "qkv-only")):
            directory = tmp_path / f"capture-{seed}-{role}"
            directory.mkdir()
            video_sha = f"{seed:02x}{index:02x}" * 16
            raw = dict(
                status="completed",
                seed=seed,
                policy=role,
                model=deepcopy(defaults.evidence.MODEL),
                upstream=dict(source_sha256="e" * 64),
                original_video=dict(
                    file="original.mp4",
                    width=512,
                    height=512,
                    frames=124,
                    fps=24,
                    duration_seconds=5.166667,
                    sha256=video_sha,
                ),
            )
            path = directory / "run.json"
            path.write_text(json.dumps(raw))
            bindings.append(
                dict(
                    role=role if index < 2 else role + "-fused",
                    run_sha256=defaults.base.sha256(path),
                )
            )
            directories.append(directory)
            if index >= 2:
                arms.append(
                    dict(
                        seed=seed,
                        policy=role,
                        run_sha256=f"{seed:02x}{index + 4:02x}" * 16,
                        producer=dict(
                            source_sha256="e" * 64, model=deepcopy(defaults.evidence.MODEL)
                        ),
                        video_sha256=[video_sha] * 4,
                    )
                )
        evaluations.append(dict(seed=seed, source_sha256="e" * 64, capture_inputs=bindings))
    return arms, evaluations, defaults.capture_run_index(directories)


def update_bound_capture(evaluations, captures, mutate):
    """Keep the run SHA honest while independently damaging the reported contract."""
    binding = evaluations[0]["capture_inputs"][-1]
    entry = captures.pop(binding["run_sha256"])
    mutate(entry["record"])
    entry["path"].write_text(json.dumps(entry["record"]))
    binding["run_sha256"] = defaults.base.sha256(entry["path"])
    captures[binding["run_sha256"]] = entry


def test_resident_quality_binding_records_all_four_original_exports(performance_binding):
    arms, evaluations, captures = performance_binding
    result = defaults.performance_capture_binding(arms, evaluations, captures, "qkv-only")
    assert len(result["joins"]) == 4
    assert all(item["matching_generations"] == 4 for item in result["joins"])
    assert all(item["original_video"]["frames"] == 124 for item in result["joins"])
    assert "does not establish latent exactness" in result["scope"]
    # Native ffprobe serializes 31/6 to six decimals; either finite representation is valid.
    update_bound_capture(
        evaluations,
        captures,
        lambda raw: raw["original_video"].update(duration_seconds=124 / 24),
    )
    defaults.performance_capture_binding(arms, evaluations, captures, "qkv-only")


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", None),
        ("sha256", "a" * 63),
        ("file", "video-5s.mp4"),
        ("width", 256),
        ("height", 256),
        ("frames", 120),
        ("fps", 24.0),
        ("duration_seconds", 5.0),
        ("duration_seconds", float("nan")),
        ("duration_seconds", float("inf")),
        ("duration_seconds", None),
        ("duration_seconds", True),
        ("duration_seconds", "5.166667"),
    ],
)
def test_capture_original_geometry_and_digest_are_required(performance_binding, field, value):
    arms, evaluations, captures = performance_binding
    update_bound_capture(
        evaluations, captures, lambda raw: raw["original_video"].update({field: value})
    )
    with pytest.raises(ValueError):
        defaults.performance_capture_binding(arms, evaluations, captures, "qkv-only")


@pytest.mark.parametrize("damage", ["absent", "digest_absent", "seed", "policy", "source", "model"])
def test_original_video_must_come_from_the_matching_bound_capture(performance_binding, damage):
    arms, evaluations, captures = performance_binding

    def mutate(raw):
        if damage == "absent":
            raw.pop("original_video")
        elif damage == "digest_absent":
            raw["original_video"].pop("sha256")
        elif damage in ("seed", "policy"):
            raw[damage] = 43 if damage == "seed" else "baseline"
        elif damage == "source":
            raw["upstream"]["source_sha256"] = "d" * 64
        else:
            raw["model"]["revision"] = "different"

    update_bound_capture(evaluations, captures, mutate)
    with pytest.raises(ValueError):
        defaults.performance_capture_binding(arms, evaluations, captures, "qkv-only")


@pytest.mark.parametrize("damage", ["one_different", "all_different", "short", "extra", "not_list"])
def test_every_resident_export_must_match_capture(performance_binding, damage):
    arms, evaluations, captures = performance_binding
    videos = arms[-1]["video_sha256"]
    if damage == "one_different":
        videos[-1] = "0" * 64
    elif damage == "all_different":
        videos[:] = ["0" * 64] * 4
    elif damage == "short":
        videos.pop()
    elif damage == "extra":
        videos.append(videos[-1])
    else:
        arms[-1]["video_sha256"] = videos[0]
    with pytest.raises(ValueError):
        defaults.performance_capture_binding(arms, evaluations, captures, "qkv-only")


@pytest.mark.parametrize("damage", ["missing_arm", "duplicate_arm", "missing_seed", "changed_run"])
def test_video_binding_requires_complete_immutable_two_seed_evidence(performance_binding, damage):
    arms, evaluations, captures = performance_binding
    if damage == "missing_arm":
        arms.pop()
    elif damage == "duplicate_arm":
        arms[-1] = arms[0]
    elif damage == "missing_seed":
        evaluations.pop()
    else:
        next(iter(captures.values()))["path"].write_text("{}")
        # Damage a capture that is actually used as a resident arm.
        binding = evaluations[0]["capture_inputs"][-1]
        captures[binding["run_sha256"]]["path"].write_text("{}")
    with pytest.raises(ValueError):
        defaults.performance_capture_binding(arms, evaluations, captures, "qkv-only")


@pytest.mark.parametrize("mismatch", [False, True])
def test_decision_executes_video_binding_and_fails_closed(
    tmp_path, monkeypatch, performance_binding, mismatch
):
    arms, evaluations, captures = performance_binding
    if mismatch:
        arms[-1]["video_sha256"][-1] = "0" * 64
    monkeypatch.setattr(defaults.resident, "validate_run", lambda path, _: arms[int(path.name)])
    monkeypatch.setattr(
        defaults.resident,
        "compare_records",
        lambda *_: dict(
            source_sha256="e" * 64,
            comparisons={
                "baseline__qkv-only": dict(
                    both_seeds_complete_request_benefit=True,
                    neither_seed_denoise_regresses=True,
                )
            },
        ),
    )
    monkeypatch.setattr(defaults, "operator_replicates", lambda *_: [])
    monkeypatch.setattr(
        defaults, "quality_evaluation", lambda path, *_: evaluations[int(path.name)]
    )
    monkeypatch.setattr(defaults, "review_quality", lambda *_: True)
    review = tmp_path / "review.json"
    review.write_text("{}")
    result = defaults.execute(
        SimpleNamespace(
            output_dir=tmp_path / "decision",
            policy="qkv-only",
            source_manifest_sha256=PIN,
            resident_run_dir=[Path(str(i)) for i in range(4)],
            qkv_campaign=[],
            capture_run_dir=[entry["path"].parent for entry in captures.values()],
            evaluation=[Path("0"), Path("1")],
            quality_review=review,
        )
    )
    assert result["status"] == ("failed" if mismatch else "completed")
    assert result["eligible_for_default"] is not mismatch
    if mismatch:
        assert "original video SHA binding" in result["error"]
    else:
        assert len(result["performance_quality_video_binding"]["joins"]) == 4

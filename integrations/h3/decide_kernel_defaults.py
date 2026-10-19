"""Fail-closed default eligibility from raw implementation, quality and resident evidence.

This emits an immutable recommendation; it never mutates runtime defaults.
Explicit per-seed aggregate/tail review is required because scalar drift is not
perceptual quality and no universal SNR threshold is scientifically justified.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import compare_kernel_resident as resident
import evaluate_kernel_integration as evaluation
import kernel_evidence as evidence
import kernel_integration as contract
import run as base
import run_fc2_experiment as capture
import run_prep_campaign as prep


SCHEMA = "h3_kernel_default_decision_v1"
REVIEW_FIELDS = (
    "per_step_latents",
    "final_video_latent",
    "final_audio_latent",
    "decoded_rgb",
    "decoded_pcm",
    "absolute_error_tails",
    "bf16_relative_material_regression",
)


def operator_campaign(directory, kind, pin_digest):
    """Revalidate raw native reports, never trust a summary's pass Boolean."""
    from validate_qkv_report import validate as validate_qkv

    path = directory / "campaign.json"
    initial = base.sha256(path)
    record = capture.load_json(path)
    contract.exact(record.get("schema"), "h3_" + kind + "_campaign_v1", "operator schema")
    if record.get("status") not in ("completed", "completed_with_explicit_unsupported"):
        raise ValueError("operator campaign is incomplete or incorrect")
    source = directory / "sources/source-manifest.json"
    pin = contract.load_pin(source, pin_digest)
    contract.exact(record.get("source_manifest_sha256"), pin_digest, "operator source digest")
    contract.exact(record.get("source_manifest"), pin, "operator source snapshot")
    contract.exact(
        record.get("upstream", {}).get("source_sha256"),
        pin["source_sha256"],
        "operator native source",
    )
    request = capture.load_json(directory / "request.json")
    for key in ("source_manifest_sha256", "source_manifest", "harness_files", "environment"):
        contract.exact(request.get(key), record.get(key), "operator request " + key)
    expected_files = {
        "kernel_integration.py",
        "run.py",
        "run_fused_campaign.py",
        "run_fc2_experiment.py",
    } | (
        {"prep_validate.c", "fused_reference.h", "run_prep_campaign.py"}
        if kind == "prep"
        else {"qkv_validate.c", "qkv_reference.h", "run_qkv_campaign.py", "validate_qkv_report.py"}
    )
    contract.exact(
        set(record.get("harness_files", {})), expected_files, "complete operator source snapshots"
    )
    for name, digest in record.get("harness_files", {}).items():
        if Path(name).name != name or not contract.digest(digest):
            raise ValueError("unsafe operator snapshot")
        contract.exact(base.sha256(directory / "sources" / name), digest, "operator source file")
    binary = directory / ("prep-validate" if kind == "prep" else "qkv-validate")
    contract.exact(base.sha256(binary), record.get("binary_sha256"), "operator binary")
    if kind == "qkv":
        contract.exact(
            {key: record.get("model", {}).get(key) for key in evidence.MODEL},
            evidence.MODEL,
            "operator original model",
        )
    for name, stem in (("build", "build"), ("driver_build", "driver-build")):
        build = record.get(name, {})
        contract.exact(build.get("returncode"), 0, "operator compilation")
        contract.exact(build.get("fallback_detected"), False, "operator compile fallback")
        contract.exact(build.get("log"), stem + ".log", "operator build log")
        contract.exact(
            base.sha256(directory / build["log"]), build.get("log_sha256"), "operator build log SHA"
        )
    seen, supported = [], []
    for case in record.get("cases", []):
        name = case.get("directory")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("operator case directory must be local")
        case_dir = directory / name
        contract.exact(
            capture.load_json(case_dir / "case.json"),
            {key: value for key, value in case.items() if key != "directory"},
            "case snapshot",
        )
        process = case.get("process", {})
        contract.exact(process.get("fallback_detected"), False, "operator fallback")
        contract.exact(process.get("log"), "validation.log", "operator log name")
        contract.exact(
            base.sha256(case_dir / "validation.log"),
            process.get("log_sha256"),
            "operator process log",
        )
        raw = case_dir / "driver.json"
        contract.exact(base.sha256(raw), case.get("driver_sha256"), "raw native report digest")
        report = capture.load_json(raw)
        seed, rows = case.get("seed"), case.get("rows")
        if kind == "prep":
            key = (seed, rows, case.get("gate"), case.get("distinct_maps"))
            summary = prep.validate(
                report, rows, seed, key[2], key[3], request["warmups"], request["repeats"]
            )
            contract.exact(case.get("summary"), summary, "prep recomputed numerical summary")
            status = "completed"
            seen.append([seed, rows, [key[2], key[3]]])
        else:
            key = (seed, rows, case.get("policy"))
            status = validate_qkv(
                report, rows, seed, key[2], request["warmups"], request["repeats"]
            )
            seen.append(list(key))
        contract.exact(case.get("status"), status, "validated operator status")
        contract.exact(
            process.get("returncode"), 2 if status == "unsupported" else 0, "operator exit code"
        )
        if status == "unsupported" and not (kind == "qkv" and rows == 17):
            raise ValueError("unsupported deployed operator shape")
        if status == "completed":
            supported.append(key)
    contract.exact(seen, request.get("cases_requested"), "complete requested operator cases")
    if len(supported) != len(set(supported)):
        raise ValueError("duplicate operator cases")
    if kind == "prep":
        expected = {
            (seed, rows, gate, maps)
            for seed, rows, (gate, maps) in itertools.product(
                (17, 29), (17, 2049, 9914, 17455), ((0, 0), (1, 0), (1, 1))
            )
        }
    else:
        expected = set(itertools.product((17, 29), (129, 2049, 9914, 17455), ("native", "h256")))
    if not expected <= set(supported):
        raise ValueError("operator campaign lacks required seeds, shapes or precision policies")
    contract.exact(base.sha256(path), initial, "operator raw record unchanged")
    return dict(
        campaign_sha256=initial,
        kind=kind,
        source_sha256=pin["source_sha256"],
        source_pin_sha256=pin_digest,
        cases=seen,
        environment=record["environment"],
        started_utc=record["started_utc"],
        finished_utc=record["finished_utc"],
        reverse=request.get("reverse"),
        harness_files=record["harness_files"],
        supported_cases=len(supported),
        revalidator_sha256=base.sha256(
            Path(__file__).with_name(
                "run_prep_campaign.py" if kind == "prep" else "validate_qkv_report.py"
            )
        ),
    )


def operator_replicates(directories, kind, pin):
    if len(directories) != 2:
        raise ValueError("two independently rerun operator campaigns are required")
    a, b = [operator_campaign(path, kind, pin) for path in directories]
    if a["campaign_sha256"] == b["campaign_sha256"]:
        raise ValueError("the same operator evidence is not independent replication")
    for key in ("environment", "harness_files", "source_sha256"):
        contract.exact(a[key], b[key], "operator rerun " + key)
    contract.exact(a["reverse"], False, "first operator order")
    contract.exact(b["reverse"], True, "second operator order")
    contract.exact(a["cases"], list(reversed(b["cases"])), "operator reversed cases")
    bounds = []
    for value in (a, b):
        start, finish = (
            resident.prior.timestamp(value[key]) for key in ("started_utc", "finished_utc")
        )
        if not start < finish:
            raise ValueError("operator rerun must have a positive UTC interval")
        bounds.append((start, finish))
    if bounds[0][1] > bounds[1][0]:
        raise ValueError("operator reruns overlap")
    return [a, b]


def finite(value, label, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("invalid finite quality value: " + label)
    return value


def metric_score(item, label, expected="finite", nonnegative=False):
    if not isinstance(item, dict):
        raise ValueError("missing structured quality metric: " + label)
    contract.exact(item.get("status"), expected, label + " status")
    if expected == "finite":
        return finite(item.get("value"), label, nonnegative)
    contract.exact(item.get("value"), None, label + " nonfinite representation")
    if not isinstance(item.get("reason"), str) or not item["reason"]:
        raise ValueError("nonfinite theoretical metric requires a reason: " + label)
    return None


def numeric_result(value, shape, dtype, data_range, tails=False):
    """Validate the emitted numeric schema, including legitimate zero-reference cases."""
    if not isinstance(value, dict):
        raise ValueError("missing numerical quality result")
    for key, expected in dict(
        status="completed",
        shape=shape,
        elements=math.prod(shape),
        reference_dtype=dtype,
        candidate_dtype=dtype,
        data_range=data_range,
        accumulation_dtype="float64",
        reference_nonfinite=0,
        candidate_nonfinite=0,
    ).items():
        contract.exact(value.get(key), expected, "quality numerical " + key)
    if any(type(v) is not int for v in value["shape"]):
        raise ValueError("quality dimensions must be integers")
    exact = value.get("exactly_equal")
    if type(exact) is not bool:
        raise ValueError("exact equality evidence is required")
    ranges = []
    for key in ("reference_min_max", "candidate_min_max"):
        bounds = value.get(key)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("missing finite numerical range")
        low, high = (finite(v, key) for v in bounds)
        if low > high:
            raise ValueError("reversed numerical range")
        ranges.append(bounds)
    metrics = value.get("metrics", {})
    contract.exact(
        set(metrics),
        {"rmse", "maximum_absolute_error", "snr_db", "psnr_db", "relative_l2"},
        "metric set",
    )
    rmse = metric_score(metrics["rmse"], "RMSE", nonnegative=True)
    maximum = metric_score(metrics["maximum_absolute_error"], "maximum error", nonnegative=True)
    if rmse > maximum or exact != (maximum == 0) or (maximum > 0 and rmse == 0):
        raise ValueError("inconsistent exact/error metrics")
    zero_reference = ranges[0] == [0, 0]
    if exact:
        contract.exact(ranges[0], ranges[1], "equal inputs must share their range")
        metric_score(metrics["psnr_db"], "PSNR", "positive_infinity")
        metric_score(
            metrics["snr_db"], "SNR", "undefined" if zero_reference else "positive_infinity"
        )
        if zero_reference:
            metric_score(metrics["relative_l2"], "relative L2", "undefined")
        else:
            contract.exact(
                metric_score(metrics["relative_l2"], "relative L2", nonnegative=True),
                0.0,
                "zero relative error",
            )
    else:
        psnr = metric_score(metrics["psnr_db"], "PSNR")
        if not math.isclose(psnr, 20 * math.log10(data_range / rmse), abs_tol=1e-8, rel_tol=1e-10):
            raise ValueError("PSNR disagrees with data range/RMSE")
        if zero_reference:
            metric_score(metrics["snr_db"], "SNR", "negative_infinity")
            metric_score(metrics["relative_l2"], "relative L2", "positive_infinity")
        else:
            snr = metric_score(metrics["snr_db"], "SNR")
            relative = metric_score(metrics["relative_l2"], "relative L2", nonnegative=True)
            if relative == 0 or not math.isclose(
                snr, -20 * math.log10(relative), abs_tol=1e-8, rel_tol=1e-10
            ):
                raise ValueError("SNR disagrees with relative L2")
    if tails:
        quantiles = value.get("absolute_error_quantiles", {})
        contract.exact(set(quantiles), {"p50", "p95", "p99", "p999"}, "error quantiles")
        ordered = [finite(quantiles[k], k, True) for k in ("p50", "p95", "p99", "p999")]
        if ordered != sorted(ordered) or ordered[-1] > maximum:
            raise ValueError("invalid ordered absolute-error tails")


def media_result(decoded):
    for key, expected in dict(
        schema_version=1,
        status="completed",
        kind="media",
        harness_sha256=base.sha256(Path(evaluation.metrics.__file__)),
    ).items():
        contract.exact(decoded.get(key), expected, "strict media " + key)
    files = decoded.get("files", {})
    contract.exact(set(files), {"reference", "candidate"}, "paired media files")
    contracts = []
    for side in ("reference", "candidate"):
        item = files[side]
        if (
            not contract.digest(item.get("sha256"))
            or type(item.get("bytes")) is not int
            or item["bytes"] <= 0
        ):
            raise ValueError("media input byte identity is missing")
        contract.exact(item.get("name"), "video-5s.mp4", "media filename")
        value = item.get("contract", {})
        for key, expected in dict(
            width=512,
            height=512,
            fps="24",
            frames=120,
            duration_seconds="5",
            audio_rate=32000,
            audio_channels=2,
            decoded_audio_samples=160000,
        ).items():
            contract.exact(value.get(key), expected, "media geometry " + key)
        contract.exact(
            value.get("video_pts"), [str(Fraction(i, 24)) for i in range(120)], "video timestamps"
        )
        counts, points = value.get("audio_frame_samples"), value.get("audio_pts")
        if (
            not isinstance(counts, list)
            or not counts
            or any(type(n) is not int or n <= 0 for n in counts)
            or sum(counts) != 160000
        ):
            raise ValueError("audio frame samples do not cover exactly five seconds")
        expected_pts, consumed = [], 0
        for count in counts:
            expected_pts.append(str(Fraction(consumed, 32000)))
            consumed += count
        contract.exact(points, expected_pts, "audio timestamps")
        color = value.get("color_metadata")
        color_keys = {
            "pix_fmt",
            "color_range",
            "color_space",
            "color_transfer",
            "color_primaries",
            "chroma_location",
            "sample_aspect_ratio",
        }
        if not isinstance(color, dict) or set(color) != color_keys:
            raise ValueError("complete media color metadata required")
        contract.exact(
            value.get("video_frame_color_metadata"), [color] * 120, "frame color metadata"
        )
        contracts.append(value)
    contract.exact(contracts[0], contracts[1], "complete paired media contract")
    numeric_result(decoded.get("video_rgb"), [120, 512, 512, 3], "uint8", 255.0)
    numeric_result(decoded.get("audio_pcm32k_stereo"), [160000, 2], "float32", 2.0)
    channels = decoded.get("audio_channels", {})
    contract.exact(set(channels), {"left", "right"}, "audio channel metrics")
    for value in channels.values():
        numeric_result(value, [160000], "float32", 2.0)
    ssim = decoded.get("optional", {}).get("ssim", {})
    contract.exact(ssim.get("status"), "completed", "RGB SSIM")
    contract.exact(ssim.get("backend"), "ffmpeg ssim, planar RGB gbrp", "RGB SSIM backend")
    mean = metric_score(ssim.get("mean"), "SSIM mean")
    frames = ssim.get("per_frame")
    if (
        not isinstance(frames, list)
        or len(frames) != 120
        or any(not -1 <= finite(v, "SSIM frame") <= 1 for v in frames)
    ):
        raise ValueError("complete finite per-frame SSIM required")
    if not math.isclose(mean, sum(frames) / 120, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("SSIM mean differs from raw frame scores")
    return {side: files[side]["sha256"] for side in ("reference", "candidate")}


def capture_run_index(directories):
    """Read immutable small run manifests only, never media or model payloads."""
    indexed = {}
    for directory in directories:
        path = Path(directory) / "run.json"
        digest = base.sha256(path)
        if digest in indexed:
            raise ValueError("capture run manifests must be distinct")
        value = capture.load_json(path)
        contract.exact(base.sha256(path), digest, "capture run remained unchanged")
        indexed[digest] = dict(path=path, record=value)
    if not indexed:
        raise ValueError("raw capture run manifests are required for media/source binding")
    return indexed


def capture_media_source(source, role, seed, pin, indexed):
    digest = source.get("run_sha256")
    if not contract.digest(digest) or digest not in indexed:
        raise ValueError("quality source lacks a bound raw capture run: " + role)
    entry = indexed[digest]
    contract.exact(base.sha256(entry["path"]), digest, "bound capture run unchanged")
    value = entry["record"]
    for key, expected in dict(status="completed", seed=seed).items():
        contract.exact(value.get(key), expected, "bound capture " + key)
    for key, expected in contract.RECIPE.items():
        if key not in ("mode", "fps", "source_frames"):
            contract.exact(value.get(key), expected, "bound capture recipe " + key)
    model = {key: value.get("model", {}).get(key) for key in evidence.MODEL}
    contract.exact(model, evidence.MODEL, "bound original checkpoint")
    contract.exact(source.get("model"), model, "quality/raw checkpoint join")
    contract.exact(
        value.get("mode"), "bf16" if role == "bf16" else "hadamard-fc2", "bound capture mode"
    )
    if role not in ("bf16", "old_hadamard"):
        policy = role.removesuffix("-fused")
        for key, expected in dict(
            kind="capture",
            policy=policy,
            implementation="fused",
            capture_latents=True,
            diagnostic=True,
            prepared_checkpoint=False,
            requested_runtime_flags=contract.flags(policy),
        ).items():
            contract.exact(value.get(key), expected, "bound candidate " + key)
        contract.exact(value.get("source_pin", {}).get("sha256"), pin, "bound candidate source pin")
        contract.exact(source.get("seed"), seed, "source seed")
        contract.exact(source.get("policy"), policy, "source policy")
        contract.exact(source.get("source_pin_sha256"), pin, "source pin")
        contract.exact(source.get("implementation"), "fused", "source implementation")
        for key in ("source_sha256", "metadata_sha256"):
            if not contract.digest(source.get(key)):
                raise ValueError("bound candidate source digest is missing: " + key)
        contract.exact(
            source.get("source_sha256"),
            value.get("upstream", {}).get("source_sha256"),
            "quality/raw native source join",
        )
        contract.exact(
            source.get("metadata_sha256"),
            value.get("latent_validation", {}).get("metadata_sha256"),
            "quality/raw latent metadata join",
        )
    video = value.get("exact_video", {})
    for key, expected in dict(
        file="video-5s.mp4", width=512, height=512, frames=120, fps=24, duration_seconds=5.0
    ).items():
        contract.exact(video.get(key), expected, "bound exact video " + key)
    if not contract.digest(video.get("sha256")):
        raise ValueError("bound capture lacks exact video digest")
    return dict(run_sha256=digest, video_sha256=video["sha256"], role=role)


def quality_evaluation(path, policy, pin, capture_runs=None):
    initial = base.sha256(path)
    report = capture.load_json(path)
    for key, value in dict(
        schema=evaluation.SCHEMA,
        status="completed",
        source_pin_sha256=pin,
        evaluator_sha256=base.sha256(Path(evaluation.__file__)),
        data_range=1.0,
    ).items():
        contract.exact(report.get(key), value, "quality evidence " + key)
    dependencies = {
        name: base.sha256(Path(__file__).with_name(name))
        for name in (
            "kernel_integration.py",
            "kernel_evidence.py",
            "evaluate.py",
            "evaluate_latents.py",
            "evaluate_benchmark_pair.py",
            "run_fc2_experiment.py",
            "run.py",
        )
    }
    contract.exact(report.get("dependencies"), dependencies, "quality evaluator dependencies")
    seed = report.get("seed")
    contract.exact(report.get("baseline_replay_exact"), True, "old FC2-H baseline control replay")
    if type(seed) is not int or seed not in (42, 43):
        raise ValueError("quality evidence must use seed42/43")
    source = report.get("sources", {}).get(policy + "-fused", {})
    for key, expected in dict(
        implementation="fused",
        policy=policy,
        seed=seed,
        model=evidence.MODEL,
        source_pin_sha256=pin,
    ).items():
        contract.exact(source.get(key), expected, "candidate quality source " + key)
    for key in ("run_sha256", "source_sha256", "metadata_sha256"):
        if not contract.digest(source.get(key)):
            raise ValueError("missing candidate source digest: " + key)
    if not capture_runs:
        raise ValueError("raw capture manifests are required to bind quality/media inputs")
    roles = ("bf16", "old_hadamard", "baseline-fused", policy + "-fused")
    bindings = {
        role: capture_media_source(
            report.get("sources", {}).get(role, {}), role, seed, pin, capture_runs
        )
        for role in roles
    }
    for label in ("old-hadamard", "baseline-fused", policy + "-fused"):
        value = report.get("comparisons", {}).get(label, {})
        arrays = value.get("against_bf16", {})
        expected = {key: shape for _, key, shape in evaluation.historical.tensor_spec()}
        contract.exact(set(arrays), set(expected), "twenty-step and final quality coverage")
        for key, shape in expected.items():
            numeric_result(arrays[key], shape, "float32", 1.0, tails=True)
        media = value.get("media", {})
        raw = path.parent / (label + "-media") / "evaluation.json"
        digest = base.sha256(raw)
        contract.exact(digest, media.get("evaluation_sha256"), "strict media evidence SHA")
        decoded = capture.load_json(raw)
        media_inputs = media_result(decoded)
        role = "old_hadamard" if label == "old-hadamard" else label
        contract.exact(
            media_inputs["reference"],
            bindings["bf16"]["video_sha256"],
            "media reference/raw BF16 run join",
        )
        contract.exact(
            media_inputs["candidate"],
            bindings[role]["video_sha256"],
            "media candidate/raw capture run join",
        )
        for key in ("video_rgb", "audio_pcm32k_stereo"):
            contract.exact(decoded.get(key), media.get(key), "joined media numerical results")
        contract.exact(decoded["optional"]["ssim"], media.get("ssim"), "joined RGB SSIM")
        contract.exact(base.sha256(raw), digest, "strict media evidence unchanged")
    contract.exact(base.sha256(path), initial, "quality evaluation unchanged")
    return dict(
        seed=seed,
        evaluation_sha256=initial,
        candidate_run_sha256=source["run_sha256"],
        source_sha256=source["source_sha256"],
        source_pin_sha256=pin,
        policy=policy,
        capture_inputs=list(bindings.values()),
        scope="Numeric aggregate/tail acceptance only; not perceptual quality certification",
    )


def performance_capture_binding(performance_runs, evaluations, capture_runs, policy):
    """Join validated resident exports to the captures used for numerical review."""
    quality = {item["seed"]: item for item in evaluations}
    arms = {(item["seed"], item["policy"]): item for item in performance_runs}
    expected = {(seed, role) for seed in (42, 43) for role in ("baseline", policy)}
    if len(evaluations) != 2 or set(quality) != {42, 43}:
        raise ValueError("video binding requires both quality seeds")
    if len(performance_runs) != 4 or set(arms) != expected:
        raise ValueError("video binding requires both resident policies and seeds")
    joins = []
    for seed, role in sorted(expected):
        arm = arms[seed, role]
        bindings = quality[seed]["capture_inputs"]
        matches = [item for item in bindings if item.get("role") == role + "-fused"]
        if len(matches) != 1:
            raise ValueError("resident policy lacks a unique reviewed capture")
        digest = matches[0].get("run_sha256")
        if not contract.digest(digest) or digest not in capture_runs:
            raise ValueError("resident policy lacks a bound raw capture run")
        entry = capture_runs[digest]
        contract.exact(base.sha256(entry["path"]), digest, "performance capture run unchanged")
        raw = entry["record"]
        for key, value in dict(seed=seed, policy=role, status="completed").items():
            contract.exact(raw.get(key), value, "performance capture " + key)
        contract.exact(
            arm["producer"]["source_sha256"],
            raw.get("upstream", {}).get("source_sha256"),
            "performance capture native source",
        )
        contract.exact(
            arm["producer"]["model"],
            {key: raw.get("model", {}).get(key) for key in evidence.MODEL},
            "performance capture original model",
        )
        video = raw.get("original_video")
        if not isinstance(video, dict):
            raise ValueError("capture lacks original video evidence")
        for key, value in dict(
            file="original.mp4", width=512, height=512, frames=124, fps=24
        ).items():
            contract.exact(video.get(key), value, "capture original video " + key)
        duration = finite(video.get("duration_seconds"), "original video duration")
        # Match run.validate_video's native 124-frame duration contract, not the 5s trim.
        if abs(duration - 124 / 24) > 0.0001:
            raise ValueError("capture original video duration differs from its frame count")
        if not contract.digest(video.get("sha256")):
            raise ValueError("capture original video digest is missing or invalid")
        videos = arm.get("video_sha256")
        if not isinstance(videos, list) or len(videos) != 4:
            raise ValueError("four resident original video digests are required")
        contract.exact(videos, [video["sha256"]] * 4, "resident/capture original video SHA binding")
        joins.append(
            dict(
                seed=seed,
                policy=role,
                resident_run_sha256=arm["run_sha256"],
                capture_run_sha256=digest,
                original_video=video,
                resident_video_sha256=videos,
                matching_generations=4,
            )
        )
    return dict(
        joins=joins,
        scope="Four validated original 124-frame resident exports match the reviewed same-seed/policy capture by SHA256. Video byte identity does not establish latent exactness.",
    )


def review_quality(review, evaluations, policy, pin):
    for key, expected in dict(
        schema="h3_kernel_quality_review_v1", policy=policy, source_pin_sha256=pin
    ).items():
        contract.exact(review.get(key), expected, "explicit quality review " + key)
    indexed = {value["seed"]: value for value in evaluations}
    if len(evaluations) != 2 or set(indexed) != {42, 43}:
        raise ValueError("quality requires both seeds")
    reviews = review.get("reviews", [])
    if len(reviews) != 2 or {item.get("seed") for item in reviews} != {42, 43}:
        raise ValueError("explicit per-seed quality decisions are required")
    accepted = True
    for value in reviews:
        contract.exact(
            value.get("evaluation_sha256"),
            indexed[value["seed"]]["evaluation_sha256"],
            "review bound to exact quality evidence",
        )
        if not isinstance(value.get("reviewer"), str) or not value["reviewer"].strip():
            raise ValueError("quality reviewer must be named")
        if value.get("decision") not in ("acceptable", "rejected", "not_evaluated"):
            raise ValueError("unknown quality decision")
        fields = REVIEW_FIELDS + (
            ("activation_quantization_aggregate_and_tail",)
            if policy in ("adaln-h-only", "combined")
            else ()
        )
        for field in fields:
            if (
                not isinstance(value.get("findings", {}).get(field), str)
                or not value["findings"][field].strip()
            ):
                raise ValueError("explicit aggregate/tail findings are required: " + field)
        accepted &= value["decision"] == "acceptable"
    return accepted


def activation_evidence(paths, evaluations):
    """Bind sampled real-operand precision evidence; do not call it GEMM proof."""
    import adaln_capture

    indexed = {item["candidate_run_sha256"]: item for item in evaluations}
    if len(paths) != 2 or len(indexed) != 2:
        raise ValueError("two real-activation analyses are required")
    result, seen = [], set()
    for path in paths:
        report = capture.load_json(path)
        contract.exact(
            report.get("analyzer_sha256"),
            base.sha256(Path(adaln_capture.__file__)),
            "actual activation analyzer",
        )
        contract.exact(
            report.get("reference_helper_sha256"),
            base.sha256(Path(__file__).with_name("fc2_analyze.py")),
            "activation reference helper",
        )
        run_sha = report.get("source_run_sha256")
        if run_sha not in indexed or run_sha in seen:
            raise ValueError(
                "actual activation evidence does not bind each candidate seed exactly once"
            )
        seen.add(run_sha)
        cases = report.get("cases", [])
        wanted = set(itertools.product(("qkv", "fc1"), (0, 10, 19), (0, 24, 49)))
        actual = {
            (
                c.get("metadata", {}).get("family"),
                c.get("metadata", {}).get("step"),
                c.get("metadata", {}).get("block"),
            )
            for c in cases
        }
        if len(cases) != 18 or actual != wanted:
            raise ValueError("actual activation evidence lacks full requested sample coverage")
        for case in cases:
            preparation = case.get("preparation", {})
            contract.exact(preparation.get("status"), "passed", "actual preparation reference")
            contract.exact(
                preparation.get("relative_l2_limit"), 0.008, "unrelaxed actual preparation limit"
            )
            for key in (
                "gate_error_vs_independent_fp64_then_bf16",
                "adaln_error_vs_independent_fp64_then_bf16",
            ):
                value = preparation.get(key, {}).get("relative_l2")
                contract.positive(value, "actual independent reference", allow_zero=True)
                if value > 0.008:
                    raise ValueError("actual independent preparation reference failed")
            contract.exact(
                preparation.get("activation_pack_implementation", {}).get("pass"),
                True,
                "actual H256 activation packing",
            )
            precision = case.get("projection_precision", {})
            contract.exact(
                precision.get("rotation_check_pass"),
                True,
                "sampled original-basis dot-product invariance",
            )
            for key in ("plain_row_int8_quantization_error", "h256_row_int8_quantization_error"):
                for metric in ("relative_l2", "rmse", "max_absolute"):
                    contract.positive(
                        precision.get(key, {}).get(metric), "same-X/W quantization", allow_zero=True
                    )
        result.append(
            dict(
                analysis_sha256=base.sha256(path),
                source_run_sha256=run_sha,
                seed=indexed[run_sha]["seed"],
                case_count=18,
                scope="Actual H-trajectory rows; sampled output channels/full K. Quantization evidence is separate from GPU GEMM implementation proof.",
            )
        )
    return result


def decide(policy, performance, quality_accepted, implementation_verified):
    pair = performance["comparisons"]["baseline__" + policy]
    gates = dict(
        implementation=implementation_verified,
        explicit_two_seed_quality=quality_accepted,
        two_seed_e2e_benefit=pair["both_seeds_complete_request_benefit"],
        denoise_no_regression=pair["neither_seed_denoise_regresses"],
    )
    eligible = all(type(value) is bool and value for value in gates.values())
    return dict(
        policy=policy,
        gates=gates,
        eligible_for_default=eligible,
        decision="eligible_for_default" if eligible else "keep_experimental",
        recommended_flags=contract.flags(policy) if eligible else None,
        runtime_defaults_modified=False,
        quality_review_scope="Numeric aggregate/tail acceptance; not perceptual quality certification",
        scope="Pinned model/source, M5 host, fixed two-seed five-second recipe; descriptive evidence, not universal quality or statistical significance.",
    )


def execute(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema=SCHEMA,
        status="failed",
        decision="keep_experimental",
        eligible_for_default=False,
        runtime_defaults_modified=False,
        evaluator_sha256=base.sha256(__file__),
    )
    try:
        policy = args.policy
        pin = args.source_manifest_sha256
        performance_runs = [resident.validate_run(path, pin) for path in args.resident_run_dir]
        performance = resident.compare_records(performance_runs, ("baseline", policy))
        record["performance"] = performance
        operators = {}
        if policy in ("qkv-only", "combined"):
            operators["qkv"] = operator_replicates(args.qkv_campaign, "qkv", pin)
        if policy in ("adaln-h-only", "combined"):
            operators["prep"] = operator_replicates(args.prep_campaign, "prep", pin)
        record["implementation_evidence"] = operators
        capture_inputs = capture_run_index(args.capture_run_dir)
        evaluations = [
            quality_evaluation(path, policy, pin, capture_inputs) for path in args.evaluation
        ]
        for value in evaluations:
            contract.exact(
                value["source_sha256"], performance["source_sha256"], "quality and timing source"
            )
        record["performance_quality_video_binding"] = performance_capture_binding(
            performance_runs, evaluations, capture_inputs, policy
        )
        if policy in ("adaln-h-only", "combined"):
            record["actual_activation_evidence"] = activation_evidence(
                args.adaln_analysis, evaluations
            )
        review = capture.load_json(args.quality_review)
        record.update(
            quality_evidence=evaluations,
            quality_review=review,
            quality_review_sha256=base.sha256(args.quality_review),
        )
        accepted = review_quality(review, evaluations, policy, pin)
        for digest, entry in capture_inputs.items():
            contract.exact(
                base.sha256(entry["path"]), digest, "raw capture run unchanged during decision"
            )
        record.update(decide(policy, performance, accepted, True), status="completed")
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    base.write_json(output / "default-decision.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("qkv-only", "adaln-h-only", "combined"), required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--resident-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--qkv-campaign", type=Path, action="append", default=[])
    parser.add_argument("--prep-campaign", type=Path, action="append", default=[])
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument(
        "--capture-run-dir",
        type=Path,
        action="append",
        required=True,
        help="raw BF16, old-H, baseline and candidate run directories for both seeds; only run.json is read",
    )
    parser.add_argument("--adaln-analysis", type=Path, action="append", default=[])
    parser.add_argument("--quality-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = execute(parser.parse_args())
    print(json.dumps({key: result.get(key) for key in ("status", "decision", "error")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)

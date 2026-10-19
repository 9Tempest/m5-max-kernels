"""Independent contracts for the QKV/RMS/RoPE and AdaLN/H256 integration study."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import run as base
import run_fc2_experiment as capture
import run_session as session


SCHEMA = "h3_kernel_integration_run_v1"
POLICIES = {
    "baseline": (0, 0, 0),
    "qkv-only": (1, 0, 0),
    "adaln-h-only": (0, 1, 1),
    "combined": (1, 1, 1),
}
NEW_FLAGS = ("H3_M5_QKV_ROPE", "H3_HADAMARD_QKV", "H3_HADAMARD_FC1")
RECIPE = dict(
    prompt=base.PROMPT,
    mode="hadamard-fc2",
    width=512,
    height=512,
    seconds=5,
    fps=24,
    steps=20,
    source_frames=124,
    active_dit_blocks=50,
    reuse=1,
    core_reuse=1,
    reference_rope=True,
)
SHAPES = dict(
    final_video_shape=[24, 37, 32, 32],
    final_audio_shape=[32, 2, 207],
    per_step_video_shape=[9472, 96],
    per_step_audio_shape=[414, 32],
)


def exact(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("kernel integration mismatch: " + label)


def digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def flags(policy, implementation="fused"):
    if policy not in POLICIES or implementation not in ("fused", "staged"):
        raise ValueError("unknown kernel policy or preparation implementation")
    if implementation == "staged" and policy not in ("adaln-h-only", "combined"):
        raise ValueError("staged preparation is only defined for Hadamard input policies")
    return dict(zip(NEW_FLAGS, map(str, POLICIES[policy]), strict=True)) | dict(
        H3_HADAMARD_FC2="1",
        H3_FUSE_FC2_GATE="0",
        H3_FUSE_ATTN_OUT_GATE="0",
        H3_HADAMARD_PREP_STAGED=str(int(implementation == "staged")),
    )


def metadata_flags(policy, implementation):
    controls = flags(policy, implementation)
    return dict(
        m5_qkv_rope=int(controls["H3_M5_QKV_ROPE"]),
        hadamard_qkv=int(controls["H3_HADAMARD_QKV"]),
        hadamard_fc1=int(controls["H3_HADAMARD_FC1"]),
        hadamard_prep_staged=int(controls["H3_HADAMARD_PREP_STAGED"]),
    )


def load_pin(path, expected_sha256):
    if not digest(expected_sha256) or base.sha256(path) != expected_sha256:
        raise ValueError("source manifest differs from the explicit approved SHA256")
    value = capture.load_json(path)
    exact(value.get("schema"), "h3_kernel_integration_source_v1", "source manifest schema")
    exact(value.get("status"), "frozen", "source manifest must be frozen")
    exact(value.get("upstream_revision"), base.UPSTREAM_REVISION, "upstream revision")
    if not digest(value.get("source_sha256")):
        raise ValueError("source digest is not frozen")
    changed = value.get("changed_files")
    if (
        not isinstance(changed, list)
        or len(changed) != len(set(changed))
        or set(changed) != {"h3_dit.c", "h3_gpu.h", "h3_gpu.m", "h3_shaders.metal"}
    ):
        raise ValueError("unexpected native changed-file contract")
    if not isinstance(value.get("patches"), list) or not value["patches"]:
        raise ValueError("reviewed overlay digests are required")
    seen = set()
    for item in value["patches"]:
        name = item.get("file")
        if not isinstance(name, str) or Path(name).name != name or name in seen:
            raise ValueError("overlay filenames must be unique local basenames")
        seen.add(name)
        if not digest(item.get("sha256")) or base.sha256(path.parent / name) != item["sha256"]:
            raise ValueError("overlay digest differs from source manifest")
    return value


def validate_source(source, pin):
    provenance = base.git_provenance(source)
    changed = set(
        filter(
            None,
            base.read_command(["git", "diff", "HEAD", "--name-only", "-z"], source).split("\0"),
        )
    )
    if (
        provenance["revision"] != pin["upstream_revision"]
        or provenance["source_sha256"] != pin["source_sha256"]
        or changed != set(pin["changed_files"])
        or any(line.startswith("??") for line in provenance["status"].splitlines())
    ):
        raise ValueError("combined native source differs from the reviewed source pin")
    return provenance


def validate_latents(directory, policy, implementation, seed):
    result = capture.validate_latent_capture(directory, "hadamard-fc2", seed)
    expected = metadata_flags(policy, implementation) | dict(
        dtype="float32_little_endian",
        scope="diagnostic_CPU_readback_after_completed_GPU_Euler",
        final_video_sigma=0,
        final_audio_sigma=0,
    )
    for key, value in expected.items():
        exact(result["metadata"].get(key), value, "latent " + key)
    return result


def validate_log(log, policy, implementation, index=0):
    values = POLICIES[policy]
    marker = "h3 kernel integration: m5_qkv_rope=%d hadamard_qkv=%d hadamard_fc1=%d" % values
    lines = [line for line in log.splitlines() if line.startswith("h3 kernel integration:")]
    exact(lines, [marker] if index == 0 else [], "observed native integration policy")
    if "TensorOps compile failed:" in log or "h3 fusion:" in log:
        raise base.UnsupportedError("fallback or unrelated gate fusion is forbidden")
    exact(log.count("h3 precision: FC2 plain normalized H256"), int(index == 0), "FC2 policy")
    if index:
        for marker in ("prepared DiT", "video VAE", "conditioning"):
            exact(log.count(f"h3: {marker} cache hit"), 1, "retained " + marker + " cache")
    return dict(policy=policy, implementation=implementation, initialization=index == 0)


def positive(value, label, allow_zero=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError("invalid finite timing: " + label)


def validate_events(events, policy, seed):
    session.validate_session_recipe(
        events, SimpleNamespace(mode="hadamard-fc2", width=512, height=512, steps=20, seed=seed)
    )
    exact(
        [e.get("event") for e in events],
        ["recipe", "runtime_flags", "startup", "device", *["generation"] * 4, "finished"],
        "session event order",
    )
    exact(events[0].get("policy"), policy, "driver policy")
    exact(events[0].get("implementation"), "fused", "resident implementation")
    exact(events[1], {"event": "runtime_flags", **flags(policy)}, "actual process environment")
    exact(events[2].get("status"), "completed", "startup")
    positive(events[2].get("wall_seconds"), "startup")
    if "M5" not in str(events[3].get("name")):
        raise base.UnsupportedError("resident execution requires an observed M5 device")
    generations = events[4:8]
    for index, event in enumerate(generations):
        name = f"{'warmup' if index == 0 else 'timed'}-{index}.mp4"
        for key, value in dict(
            index=index,
            warmup=index == 0,
            status="completed",
            output=name,
            profile_log=Path(name).with_suffix(".log").name,
            requested_seed=seed,
            result_seed=seed,
            denoise_start_phase="denoise enqueue",
            denoise_end_phase="denoise",
            denoise_total=20,
            denoise_completed=20,
            denoise_completion_events=1,
        ).items():
            exact(event.get(key), value, "generation " + key)
        for key in (
            "wall_seconds",
            "denoise_callback_seconds",
            "denoise_start_offset_seconds",
            "denoise_end_offset_seconds",
        ):
            positive(event.get(key), key, allow_zero=key.endswith("offset_seconds"))
        first, last = event["denoise_start_offset_seconds"], event["denoise_end_offset_seconds"]
        if (
            not 0 <= first < last <= event["wall_seconds"]
            or abs(last - first - event["denoise_callback_seconds"]) > 3e-8
        ):
            raise ValueError("denoise callback interval is not contained in the request")
        for key in ("prepared_dit", "video_decoder", "embedding_entries"):
            if not event.get("cache_after", {}).get(key) or (
                index and not event.get("cache_before", {}).get(key)
            ):
                raise ValueError("required resident cache was not retained")
    return generations

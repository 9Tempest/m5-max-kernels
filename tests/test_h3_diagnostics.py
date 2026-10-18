"""CPU-only checks for explicitly identified H3 diagnostic runs; never launch Metal."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "integrations/h3/run.py"
spec = importlib.util.spec_from_file_location("h3_diagnostics", MODULE)
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)


def arguments(tmp_path):
    return h3.parser().parse_args(
        [
            "--mode",
            "bf16",
            "--diagnostics",
            "--source-dir",
            str(tmp_path / "source"),
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )


def geometry():
    return dict(
        schema=1,
        kind="geometry",
        sequence=9914,
        text_rows=28,
        video_rows=9472,
        audio_rows=414,
        video_condition_rows=0,
        audio_condition_rows=0,
        latent_t=37,
        latent_h=32,
        latent_w=32,
        audio_t_per_channel=207,
        hidden=5376,
        heads=56,
        head_dim=128,
        ffn=14336,
        active_blocks=50,
        steps=20,
        configured_int8_qkv=0,
        configured_int8_mlp=0,
        configured_int8_output=0,
        nax_mlp_available=0,
        final_heads_bf16=1,
        spatial_rope_scale=1,
        ssd_streaming=0,
        token_reduction=0,
        core_reuse_interval=1,
    )


def latent():
    return dict(
        schema=1,
        kind="latent",
        step=0,
        modality="video",
        status="sampled",
        boundary="post_euler",
        elements=3,
        finite=3,
        nonfinite=0,
        zeros=1,
    )


def log_lines(records):
    return "\n".join("h3 diag: " + json.dumps(record) for record in records)


def test_patch_digest_matches_the_explicit_pin():
    assert h3.sha256(MODULE.with_name("diagnostics.patch")) == h3.DIAGNOSTICS_PATCH_SHA256


@pytest.fixture
def pinned_diagnostic(monkeypatch, tmp_path):
    provenance = dict(
        revision=h3.UPSTREAM_REVISION,
        dirty=True,
        status="M h3_dit.c",
        source_sha256=h3.DIAGNOSTICS_SOURCE_SHA256,
    )
    monkeypatch.setattr(h3, "read_command", lambda *args, **kwargs: "h3_dit.c\0")
    monkeypatch.setattr(
        h3,
        "sha256",
        lambda path: (
            h3.DIAGNOSTICS_DIT_SHA256
            if Path(path).name == "h3_dit.c"
            else h3.DIAGNOSTICS_PATCH_SHA256
        ),
    )
    return tmp_path / "source", provenance


def test_exact_diagnostic_source_is_accepted_but_not_as_baseline(pinned_diagnostic):
    source, provenance = pinned_diagnostic
    assert h3.validate_upstream(source, provenance, True)["expected_source_sha256"] == (
        h3.DIAGNOSTICS_SOURCE_SHA256
    )
    with pytest.raises(ValueError, match="clean at the pinned"):
        h3.validate_upstream(source, provenance)
    with pytest.raises(ValueError, match="separate --source-dir"):
        h3.validate_upstream(h3.ROOT / ".cache/h3.c", provenance, True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "another revision"),
        ("source_sha256", "modified source"),
        ("status", "M h3_dit.c\n?? unrecorded.c"),
    ],
)
def test_diagnostic_provenance_rejects_unpinned_inputs(pinned_diagnostic, field, value):
    source, provenance = pinned_diagnostic
    with pytest.raises(ValueError, match="pinned diagnostics.patch"):
        h3.validate_upstream(source, provenance | {field: value}, True)


@pytest.mark.parametrize("changed", ["", "h3_gpu.m\0", "h3_dit.c\0h3_gpu.m\0"])
def test_diagnostic_diff_must_touch_only_dit(pinned_diagnostic, monkeypatch, changed):
    source, provenance = pinned_diagnostic
    monkeypatch.setattr(h3, "read_command", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError, match="pinned diagnostics.patch"):
        h3.validate_upstream(source, provenance, True)


@pytest.mark.parametrize("bad_file", ["diagnostics.patch", "h3_dit.c"])
def test_patch_and_changed_file_are_verified(pinned_diagnostic, monkeypatch, bad_file):
    source, provenance = pinned_diagnostic
    original = h3.sha256
    monkeypatch.setattr(
        h3, "sha256", lambda path: "wrong" if Path(path).name == bad_file else original(path)
    )
    with pytest.raises(ValueError, match="pinned diagnostics.patch"):
        h3.validate_upstream(source, provenance, True)


def test_diagnostic_environment_is_opt_in_and_rejects_inherited_controls(monkeypatch):
    for key in list(h3.os.environ):
        if key.startswith("H3_"):
            monkeypatch.delenv(key)
    assert "H3_DIAGNOSTICS" not in h3.runtime_environment()[0]
    env, overrides = h3.runtime_environment(True)
    assert env["H3_DIAGNOSTICS"] == overrides["H3_DIAGNOSTICS"] == "1"
    monkeypatch.setenv("H3_DIAGNOSTICS", "1")
    with pytest.raises(ValueError, match="unset inherited"):
        h3.runtime_environment(True)


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        '{"schema":1,"kind":"unknown"}',
        '{"schema":1,"kind":"latent","value":NaN}',
        '{"schema":1,"kind":"latent","value":Infinity}',
        '{"schema":1,"kind":"latent","value":-Infinity}',
        '{"schema":1,"kind":"latent","nested":{"value":1e999}}',
    ],
)
def test_malformed_or_nonfinite_json_is_rejected_without_coercion(payload):
    with pytest.raises(ValueError, match="malformed diagnostic record"):
        h3.parse_diagnostics("normal log\nh3 diag: " + payload)


def test_observed_geometry_and_latent_scope_are_preserved(tmp_path):
    records = [geometry(), latent()]
    parsed = h3.parse_diagnostics("ordinary log\r" + log_lines(records))
    assert parsed == records
    result = h3.validate_diagnostics(parsed, arguments(tmp_path), h3.temporal_shape(5, 512, 512))
    assert result["packed_sequence_tokens"] == 9914
    assert result["text_tokens"] == 28
    assert "no all-layer numerical parity" in result["finite_scope"]


@pytest.mark.parametrize(
    "records", [[], [geometry()], [latent()], [geometry(), geometry(), latent()]]
)
def test_diagnostics_requires_geometry_and_latent_evidence(tmp_path, records):
    with pytest.raises(ValueError, match="requires exactly one geometry"):
        h3.validate_diagnostics(records, arguments(tmp_path), h3.temporal_shape(5, 512, 512))


@pytest.mark.parametrize("update", [{"video_rows": 1}, {"sequence": 1}, {"text_rows": 0}])
def test_observed_geometry_must_match_request(tmp_path, update):
    with pytest.raises(ValueError, match="geometry|row counts"):
        h3.validate_diagnostics(
            [geometry() | update, latent()], arguments(tmp_path), h3.temporal_shape(5, 512, 512)
        )


def test_projection_fallback_is_unsupported(tmp_path):
    with pytest.raises(h3.UnsupportedError, match="precision"):
        h3.validate_diagnostics(
            [geometry() | {"configured_int8_mlp": 1}, latent()],
            arguments(tmp_path),
            h3.temporal_shape(5, 512, 512),
        )


@pytest.mark.parametrize(
    "sample,status",
    [
        (latent() | {"nonfinite": 1, "finite": 2}, "incorrect"),
        (latent() | {"status": "read_failed"}, "failed"),
        (latent() | {"pending": 1}, "failed"),
    ],
)
def test_runtime_diagnostic_failure_is_retained_without_gpu(tmp_path, monkeypatch, sample, status):
    args = arguments(tmp_path)
    monkeypatch.setattr(h3, "git_provenance", lambda _: {})
    monkeypatch.setattr(h3, "validate_upstream", lambda *args: {})
    monkeypatch.setattr(h3, "sha256", lambda _: "digest")
    monkeypatch.setattr(h3, "runtime_environment", lambda *args: ({}, {}))
    monkeypatch.setattr(h3.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(h3.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        h3,
        "read_command",
        lambda command, **kwargs: "128" if command[-1] == "hw.memsize" else "Apple M5 Max",
    )
    monkeypatch.setattr(h3.shutil, "which", lambda binary: binary)
    monkeypatch.setattr(h3, "model_inventory", lambda *args: {})
    monkeypatch.setattr(h3, "gpu_lock", nullcontext)

    def fake_process(command, source, env, output, name, stop_on_fallback):
        assert name == "generation"
        (output / "generation.log").write_text(log_lines([geometry(), sample]))
        return dict(returncode=0, fallback_detected=False)

    monkeypatch.setattr(h3, "run_logged", fake_process)
    result = h3.execute(args)
    assert result["status"] == status
    assert result["diagnostics"] is True
    assert result["diagnostic_records"][-1] == sample
    assert json.loads((args.output_dir / "run.json").read_text())["status"] == status

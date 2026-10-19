import copy
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


INTEGRATION = Path(__file__).parents[1] / "integrations/h3"
sys.path.insert(0, str(INTEGRATION))
SPEC = importlib.util.spec_from_file_location(
    "prepared_checkpoint", INTEGRATION / "prepared_checkpoint.py"
)
prepared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepared)
sys.path.remove(str(INTEGRATION))


def manifest_fixture():
    manifest = {
        "schema": 1,
        "policy_id": prepared.POLICY_ID,
        "source_model": {
            "repository": "MiniMaxAI/MiniMax-H3",
            "revision": prepared.base.MODEL_REVISION,
            "model_manifest_sha256": prepared.base.MODEL_MANIFEST_SHA256,
        },
        "producer": {
            "upstream_revision": prepared.base.UPSTREAM_REVISION,
            "source_sha256": prepared.experiment.SOURCE_SHA256,
            "patch_sha256": prepared.experiment.PATCH_SHA256,
        },
        "projections": [],
    }
    for block in range(50):
        for family in prepared.FAMILIES:
            entry = prepared.projection_spec(block, family)
            rows, columns = entry["rows"], entry["columns"]
            entry.update(
                original_tensor_sha256="a" * 64,
                original_source_file_sha256="b" * 64,
                original_source_file="FL2VA/transformer/model-00001.safetensors",
                original_dtype="BF16",
                original_shape=[rows, columns],
                original_data_offsets=[0, rows * columns * 2],
                original_file_offset=1024,
            )
            for key, suffix, size, dtype, layout in (
                ("codes", "codes.i8", rows * columns, "int8", "N,K_row_major"),
                ("scales", "scales.f32", rows * 4, "float32-le", "N"),
            ):
                entry[key] = dict(
                    file=f"block{block:02d}-{family}.{suffix}",
                    bytes=size,
                    dtype=dtype,
                    layout=layout,
                    sha256="c" * 64,
                )
            manifest["projections"].append(entry)
    return manifest


def test_complete_manifest_with_explicit_mixed_policy():
    manifest = manifest_fixture()
    assert prepared.validate_manifest_schema(manifest) is manifest
    assert manifest["projections"][3]["weight_transform"] == "normalized_plain_H256_FP32"
    assert manifest["projections"][2]["weight_transform"] == "identity"


@pytest.mark.parametrize(
    "field,value",
    [
        ("block", True),
        ("columns", 14335),
        ("weight_transform", "signed_H256"),
        ("original_tensor_sha256", "missing"),
        ("original_dtype", "F16"),
        ("original_source_file", "../../outside.safetensors"),
        ("original_data_offsets", [0, 2]),
    ],
)
def test_manifest_rejects_wrong_shape_transform_and_provenance(field, value):
    manifest = manifest_fixture()
    manifest["projections"][3][field] = value
    with pytest.raises(ValueError):
        prepared.validate_manifest_schema(manifest)


def test_manifest_rejects_partial_reordered_and_unpinned_producer():
    original = manifest_fixture()
    for transform in (
        lambda m: m["projections"].pop(),
        lambda m: m["projections"].reverse(),
        lambda m: m["producer"].update(source_sha256="0" * 64),
        lambda m: m["projections"][0]["codes"].update(file="../codes.i8"),
        lambda m: m["projections"][0]["scales"].update(layout="group1024"),
    ):
        candidate = copy.deepcopy(original)
        transform(candidate)
        with pytest.raises(ValueError):
            prepared.validate_manifest_schema(candidate)


def payload_fixture(directory, values, kind):
    path = directory / ("codes.i8" if kind == "codes" else "scales.f32")
    values.tofile(path)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=prepared.base.sha256(path))


def test_payload_digest_and_length_prevent_loading_changed_weights(tmp_path):
    payload = payload_fixture(tmp_path, np.array([-127, 0, 127], dtype=np.int8), "codes")
    prepared.validate_payload(payload, tmp_path, "codes")
    (tmp_path / payload["file"]).write_bytes(bytes([1, 0, 127]))
    with pytest.raises(ValueError, match="SHA256"):
        prepared.validate_payload(payload, tmp_path, "codes")
    (tmp_path / payload["file"]).write_bytes(b"\0")
    with pytest.raises(ValueError, match="size"):
        prepared.validate_payload(payload, tmp_path, "codes")


@pytest.mark.parametrize(
    "values,kind",
    [
        (np.array([-128, 0], dtype=np.int8), "codes"),
        (np.array([1, np.nan], dtype="<f4"), "scales"),
        (np.array([1, np.inf], dtype="<f4"), "scales"),
        (np.array([1, 0], dtype="<f4"), "scales"),
        (np.array([1, -1], dtype="<f4"), "scales"),
    ],
)
def test_payload_rejects_invalid_numerics_even_with_matching_hash(tmp_path, values, kind):
    payload = payload_fixture(tmp_path, values, kind)
    with pytest.raises(prepared.base.NumericalError):
        prepared.validate_payload(payload, tmp_path, kind)


def test_positive_finite_scales_and_symmetric_codes_are_accepted(tmp_path):
    for values, kind in (
        (np.array([0, -127, 127], dtype=np.int8), "codes"),
        (np.array([1, 0.001], dtype="<f4"), "scales"),
    ):
        payload = payload_fixture(tmp_path, values, kind)
        prepared.validate_payload(payload, tmp_path, kind)


@pytest.mark.parametrize("failure", ["native_export", "partial_completion", "native_compile"])
def test_offline_export_failure_retains_evidence_without_publishing_manifest(
    tmp_path, monkeypatch, failure
):
    source, output = tmp_path / "source", tmp_path / "checkpoint"
    source.mkdir()
    (source / "libh3.a").write_bytes(b"CPU fixture, not a native library")
    monkeypatch.setattr(prepared.base, "wait_for_gpu_lock", lambda *args: nullcontext())
    monkeypatch.setattr(prepared.base, "runtime_environment", lambda: ({}, {}))
    monkeypatch.setattr(prepared.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(prepared.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        prepared.base,
        "read_command",
        lambda command: "1" if command[-1] == "hw.memsize" else "Apple M5 CPU fixture",
    )
    monkeypatch.setattr(prepared.base, "git_provenance", lambda *args: {})
    monkeypatch.setattr(prepared.experiment, "validate_source", lambda *args: {})
    monkeypatch.setattr(prepared.base, "model_inventory", lambda *args: {"files": []})
    monkeypatch.setattr(prepared, "original_tensor_inventory", lambda *args: {})

    def fake_process(command, cwd, env, destination, name, *args):
        (destination / (name + ".log")).write_text("intentional CPU failure fixture\n")
        result = dict(returncode=0, fallback_detected=False)
        if name == "build" and failure == "native_compile":
            result["returncode"] = 1
        if name == "build-exporter":
            (destination / "prepared-export").write_bytes(b"fixture executable was not run")
        if name == "export":
            (destination / "weights/partial.codes.i8").write_bytes(b"partial diagnostic")
            if failure == "native_export":
                result["returncode"] = 1
            else:
                prepared.base.write_json(
                    destination / "weights/export-complete.json",
                    dict(first_block=0, blocks=1, projections=4),
                )
        return result

    monkeypatch.setattr(prepared.base, "run_logged", fake_process)
    args = prepared.parser().parse_args(["--source-dir", str(source), "--output-dir", str(output)])
    result = prepared.execute(args)
    assert result["status"] == ("unsupported" if failure == "native_compile" else "failed")
    assert (output / "prepare.json").is_file()
    assert not (output / "weights/manifest.json").exists()
    assert (output / "build.log").is_file()
    if failure != "native_compile":
        assert (output / "weights/partial.codes.i8").read_bytes() == b"partial diagnostic"

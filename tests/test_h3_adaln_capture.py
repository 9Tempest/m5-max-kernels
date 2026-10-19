import importlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/h3"))
capture = importlib.import_module("adaln_capture")


def test_index_absent_from_native_inventory_requires_upstream_pin(tmp_path, monkeypatch):
    import hashlib

    data = b'{"weight_map":{}}'
    entry = dict(
        path="FL2VA/transformer/model.safetensors.index.json",
        type="file",
        size=len(data),
        oid=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([entry]))
    monkeypatch.setattr(capture.base, "MODEL_MANIFEST_SHA256", capture.sha256(manifest))
    assert capture.verify_index(data, {}, manifest).startswith(
        "pinned_upstream_git_blob_at_analysis_time"
    )
    with pytest.raises(ValueError, match="Git blob"):
        capture.verify_index(data + b" ", {}, manifest)
    manifest.write_text("[]")
    with pytest.raises(ValueError, match="inventory pin"):
        capture.verify_index(data, {}, manifest)


def bits(value):
    data = np.asarray(value, np.float32).view(np.uint32)
    return ((data + 0x7FFF + ((data >> 16) & 1)) >> 16).astype("<u2")


def fixture(directory, monkeypatch, *, family="qkv", gated=False):
    # Keep scientific tests small; production capture contract remains K5376.
    monkeypatch.setattr(capture, "WIDTH", 256)
    directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(71)
    shape = (16, 256)
    values = {
        k: capture.bf16_round(rng.normal(size=shape))
        for k in ("residual", "branch", "gate", "shift", "scale")
    }
    values["residual"][0] = 0
    values["residual"][1, 255] = capture.bf16_round([10000])[0]
    values["shift"][0] = 0
    values["branch"][0] = 0
    values["gamma"] = capture.bf16_round(rng.uniform(0.5, 1.5, 256))
    if not gated:
        values["branch"][:] = 0
        values["gate"][:] = 0
    values["gated"] = capture.bf16_round(
        values["residual"].astype(np.float64) + values["branch"].astype(np.float64) * values["gate"]
    )
    x = values["gated"].astype(np.float64)
    inv = 1 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-5)
    values["adaln"] = capture.bf16_round(
        x * inv * values["gamma"] * (1 + values["scale"].astype(np.float64)) + values["shift"]
    )
    transformed = capture.hadamard_reference(values["adaln"], np.float32)
    peak = np.max(abs(transformed), axis=1)
    values["scales"] = np.where(peak > 0, peak / np.float32(127), 1).astype("<f4")
    values["codes"] = np.rint(transformed / values["scales"][:, None]).astype(np.int8)
    values["gate-map"] = np.arange(16, dtype="<u4") % 3 if gated else np.zeros(16, "<u4")
    values["norm-map"] = (np.arange(16, dtype="<u4") + 1) % 3
    stem = f"{family}-step00-block00"
    meta = dict(
        schema=1,
        scope="diagnostic_with_extra_gpu_waits",
        family=family,
        step=0,
        block=0,
        packed_rows=17,
        sample_rows=16,
        width=256,
        slots=6,
        gate_slot=0 if family == "qkv" else 2,
        shift_slot=0 if family == "qkv" else 3,
        scale_slot=1 if family == "qkv" else 4,
        apply_gate=int(gated),
        staged=0,
        hadamard_qkv=int(family == "qkv"),
        hadamard_fc1=int(family == "fc1"),
        hadamard_fc2=1,
        epsilon=1e-5,
        row_ids=list(range(15)) + [16],
        segments=[dict(kind=0, start=0, stop=17)],
        input_layout="selected_rows_full_K_BF16_little_endian",
        modulation_layout="gate_shift_scale_gathered_per_sample_using_independent_maps",
        gamma_shape=[256],
        codes_layout="selected_rows_full_K_signed_INT8",
        scales_layout="one_FP32_little_endian_per_sample",
        zero_row_scale=1,
        nonfinite=0,
    )
    path = directory / f"{stem}.json"
    path.write_text(json.dumps(meta))
    for suffix in capture.SUFFIXES:
        value = values[suffix.split(".")[0]]
        if suffix.endswith(".bf16"):
            value = bits(value)
        value.tofile(directory / f"{stem}-{suffix}")
    return path, meta, values


def flags(meta):
    return {k: meta[k] for k in capture.FLAGS}


def validate(directory, meta, **kwargs):
    return capture.validate_capture(
        directory, expected_flags=flags(meta), expected_steps=(0,), expected_blocks=(0,), **kwargs
    )


@pytest.mark.parametrize("family,gated", [("qkv", False), ("fc1", True)])
def test_real_layout_contract_and_independent_references(tmp_path, monkeypatch, family, gated):
    path, meta, _ = fixture(tmp_path, monkeypatch, family=family, gated=gated)
    result = validate(tmp_path, meta, packed_rows=17)
    assert result["case_count"] == 1
    assert len(result["files"]) == 13
    actual_meta, values, _ = capture.read_case(path)
    reference = capture.analyze_preparation(actual_meta, values)
    assert reference["status"] == "passed"
    assert reference["gate_exact_bf16_mismatches"] == 0
    assert reference["adaln_exact_bf16_mismatches"] == 0
    assert reference["activation_quantization_error_in_original_basis"]["relative_l2"] > 0
    assert reference["activation_pack_implementation"]["pass"]


@pytest.mark.parametrize(
    "update",
    [
        {"schema": True},
        {"hadamard_qkv": 0},
        {"width": 255},
        {"epsilon": 0},
        {"gate_slot": True},
        {"apply_gate": 1},
        {"row_ids": list(range(16))},  # Omits actual final row 16.
        {"row_ids": [0] * 16},
        {"segments": [{"kind": 0, "start": 1, "stop": 17}]},
        {"scales_layout": "FP16"},
        {"scope": "performance"},
    ],
)
def test_reject_corrupted_geometry_policy_or_scope(tmp_path, monkeypatch, update):
    path, meta, _ = fixture(tmp_path, monkeypatch)
    path.write_text(json.dumps(meta | update))
    with pytest.raises(ValueError):
        capture.read_case(path)


@pytest.mark.parametrize("kind", ["nan", "minus128", "scalezero", "zerorow", "size", "pregate"])
def test_reject_nonfinite_invalid_code_and_zero_row_corruption(tmp_path, monkeypatch, kind):
    path, _, values = fixture(tmp_path, monkeypatch)
    stem = path.stem
    if kind == "nan":
        values["gamma"][15] = np.nan
        bits(values["gamma"]).tofile(tmp_path / f"{stem}-gamma.bf16")
    elif kind == "minus128":
        values["codes"][3, 7] = -128
        values["codes"].tofile(tmp_path / f"{stem}-codes.i8")
    elif kind == "scalezero":
        values["scales"][5] = 0
        values["scales"].tofile(tmp_path / f"{stem}-scales.f32")
    elif kind == "zerorow":
        values["codes"][0, 11] = 1
        values["codes"].tofile(tmp_path / f"{stem}-codes.i8")
    elif kind == "pregate":
        values["gate-map"][7] = 2
        values["gate-map"].tofile(tmp_path / f"{stem}-gate-map.u32")
    else:
        (tmp_path / f"{stem}-branch.bf16").write_bytes(b"short")
    with pytest.raises(ValueError):
        capture.read_case(path)


def test_implementation_corruption_is_not_hidden_by_quantization_error(tmp_path, monkeypatch):
    _, meta, values = fixture(tmp_path, monkeypatch)
    values["codes"][3, 7] = 100
    result = capture.analyze_preparation(meta, values)
    assert result["status"] == "incorrect"
    assert result["gate_error_vs_independent_fp64_then_bf16"]["relative_l2"] == 0
    assert not result["activation_pack_implementation"]["pass"]


def test_directory_rejects_extra_case_and_staged_mismatch(tmp_path, monkeypatch):
    _, meta, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="preparation path"):
        validate(tmp_path, meta, expected_staged=1)
    with pytest.raises(ValueError, match="packed rows"):
        validate(tmp_path, meta, packed_rows=18)
    (tmp_path / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected"):
        validate(tmp_path, meta)


def test_channel_sampling_covers_complete_heads_and_paired_fc1_channels():
    q, info = capture.selected_columns("qkv")
    assert len(q) == 3 * 3 * 128
    assert len(set(q)) == len(q)
    assert info["heads"] == [0, 28, 55]
    for head in info["heads"]:
        assert set(range(head * 384, (head + 1) * 384)) <= set(q)
    f, _ = capture.selected_columns("fc1")
    np.testing.assert_array_equal(f[128:] - f[:128], 14336)
    assert f[0] == 0 and f[-1] == 28671


def test_projection_simulation_explicitly_separates_gpu_implementation(tmp_path, monkeypatch):
    _, _, values = fixture(tmp_path, monkeypatch)
    weight = capture.bf16_round(np.random.default_rng(89).normal(size=(8, 256)))
    result = capture.projection_precision(values, weight)
    assert result["rotation_check_pass"]
    assert result["projection_implementation_status"] == "not measured by this capture"
    assert len(result["per_row_plain_quantization_error"]) == 16
    assert len(result["per_row_h256_quantization_error"]) == 16
    assert sum(result[f"h256_per_row_l2_{k}"] for k in ("better", "equal", "worse")) == 16
    assert result["plain_row_int8_quantization_error"]["relative_l2"] > 0


def model_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "WIDTH", 256)
    directory = tmp_path / "FL2VA/transformer"
    directory.mkdir(parents=True)
    name = "blocks.0.mlp.fc1.weight"
    size = 28672 * 256 * 2
    header = json.dumps(
        {name: dict(dtype="BF16", shape=[28672, 256], data_offsets=[0, size])}
    ).encode()
    shard = directory / "weights.safetensors"
    with shard.open("wb") as file:
        file.write(struct.pack("<Q", len(header)))
        file.write(header)
        file.truncate(8 + len(header) + size)
    index = directory / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {name: shard.name}}))
    record = dict(
        repository="MiniMaxAI/MiniMax-H3",
        revision=capture.base.MODEL_REVISION,
        manifest_sha256=capture.base.MODEL_MANIFEST_SHA256,
        files=[
            dict(
                path="FL2VA/transformer/weights.safetensors",
                bytes=shard.stat().st_size,
                sha256="1" * 64,
            ),
            dict(
                path="FL2VA/transformer/model.safetensors.index.json",
                bytes=index.stat().st_size,
                sha256=capture.sha256(index),
            ),
        ],
    )
    return record, index, shard


def test_header_memmap_sampler_records_no_false_full_hash_verification(tmp_path, monkeypatch):
    record, _, _ = model_fixture(tmp_path, monkeypatch)
    values, provenance = capture.read_weight_samples(tmp_path, record, 0, "fc1")
    assert values.shape == (256, 256)
    assert np.all(values == 0)
    assert not provenance["source_shard_digest_reverified_by_this_analyzer"]
    assert provenance["run_declared_source_file_sha256"] == "1" * 64


@pytest.mark.parametrize("kind", ["index", "identity", "size", "shape", "path"])
def test_sampler_rejects_wrong_checkpoint_identity_index_and_header(tmp_path, monkeypatch, kind):
    record, index, shard = model_fixture(tmp_path, monkeypatch)
    if kind == "identity":
        record["revision"] = "wrong"
    elif kind == "index":
        index.write_text("{}")
    elif kind == "size":
        record["files"][0]["bytes"] -= 1
    elif kind == "path":
        index.write_text(
            json.dumps({"weight_map": {"blocks.0.mlp.fc1.weight": "../weights.safetensors"}})
        )
    else:
        with shard.open("r+b") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            header = file.read(length).replace(b"BF16", b"FP32")
            file.seek(8)
            file.write(header)
    with pytest.raises(ValueError):
        capture.read_weight_samples(tmp_path, record, 0, "fc1")


def test_cli_preserves_failed_analysis_and_never_overwrites(tmp_path, monkeypatch):
    output = tmp_path / "failure.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adaln_capture.py",
            str(tmp_path / "missing"),
            "--hadamard-qkv",
            "1",
            "--hadamard-fc1",
            "1",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError):
        capture.main()
    assert json.loads(output.read_text())["status"] == "failed"
    before = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        capture.main()
    assert output.read_bytes() == before

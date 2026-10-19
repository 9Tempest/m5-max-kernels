import hashlib
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest


INTEGRATION = Path(__file__).parents[1] / "integrations/h3"
sys.path.insert(0, str(INTEGRATION))
SPEC = importlib.util.spec_from_file_location(
    "derived_checkpoint", INTEGRATION / "derived_checkpoint.py"
)
derived = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(derived)
sys.path.remove(str(INTEGRATION))


def shard(path):
    # BF16 bit patterns include signed zero and a noncanonical NaN payload: the
    # untouched tensor must be copied as bytes, never converted through float.
    data = {
        "norm": bytes.fromhex("00800000017f"),
        "projection": b"\x01\x02" * 6,
        "bias": bytes.fromhex("0000803f000000c0"),
    }
    header = {"__metadata__": {"format": "pt"}}
    position = 0
    for name, values in data.items():
        header[name] = dict(
            dtype="F32" if name == "bias" else "BF16",
            shape=[2] if name == "bias" else [3] if name == "norm" else [2, 3],
            data_offsets=[position, position + len(values)],
        )
        position += len(values)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(data.values()))
    return data


def test_streamed_rewrite_preserves_bytes_keys_shape_dtype_and_provenance(tmp_path):
    source, target = tmp_path / "original.safetensors", tmp_path / "model/new.safetensors"
    data = shard(source)
    original_bytes = source.read_bytes()
    result = derived.rewrite_shard(
        source,
        target,
        {"projection": hashlib.sha256(data["projection"]).hexdigest()},
        hashlib.sha256(original_bytes).hexdigest(),
    )
    assert source.read_bytes() == original_bytes
    with target.open("rb") as file:
        header, tensors, prefix = derived.read_header(file)
        assert [name for name, _ in tensors] == ["norm", "bias"]
        assert header["norm"]["dtype"] == "BF16" and header["norm"]["shape"] == [3]
        assert header["bias"]["data_offsets"] == [6, 14]
        assert header["__metadata__"] == {"format": "pt"}
        assert file.read() == data["norm"] + data["bias"]
    assert result["sha256"] == derived.base.sha256(target)
    assert result["derived_header_bytes"] == len(prefix)
    assert (
        result["retained_tensors"][0]["tensor_sha256"] == hashlib.sha256(data["norm"]).hexdigest()
    )
    assert len(result["removed_tensors"]) == 1


def test_rewrite_rejects_export_digest_mismatch_and_retains_failed_output(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    shard(source)
    with pytest.raises(ValueError, match="tensor SHA256"):
        derived.rewrite_shard(source, target, {"projection": "0" * 64}, derived.base.sha256(source))
    assert target.is_file() and source.is_file()


def test_rewrite_rejects_changed_original_and_does_not_overwrite(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    data = shard(source)
    removed = {"projection": hashlib.sha256(data["projection"]).hexdigest()}
    with pytest.raises(ValueError, match="original shard"):
        derived.rewrite_shard(source, target, removed, "0" * 64)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        derived.rewrite_shard(source, target, removed, derived.base.sha256(source))
    assert target.read_bytes() == before


def test_hardlink_preserves_original_content_and_independent_path(tmp_path):
    source, target = tmp_path / "source", tmp_path / "model/target"
    source.write_bytes(b"stable original checkpoint")
    result = derived.preserve_file(source, target, derived.base.sha256(source))
    assert result["storage"] == "hardlink"
    assert source.stat().st_ino == target.stat().st_ino
    assert source.read_bytes() == target.read_bytes()
    with pytest.raises(FileExistsError):
        derived.preserve_file(source, target, derived.base.sha256(source))


@pytest.mark.parametrize("name", ["../outside", "/absolute", "", "model/../../outside"])
def test_unsafe_inventory_path_is_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        derived.contained_file(tmp_path, name)


def test_header_rejects_duplicate_keys_and_overlapping_tensor_ranges(tmp_path):
    path = tmp_path / "bad"
    for header in (
        b'{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}',
        b'{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"y":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}',
    ):
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
        with path.open("rb") as file, pytest.raises(ValueError):
            derived.read_header(file)


def test_import_rejects_outer_manifest_digest_before_reading_model(tmp_path):
    (tmp_path / "derived-checkpoint.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest SHA256"):
        derived.verify_derived(tmp_path, "0" * 64)
    assert not (tmp_path / "model").exists()


def test_failed_export_cannot_become_a_derived_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate"
    checkpoint.mkdir()
    derived.base.write_json(checkpoint / "prepare.json", {"status": "failed"})
    monkeypatch.setattr(derived.base, "wait_for_gpu_lock", lambda *args: nullcontext())
    args = SimpleNamespace(
        checkpoint_dir=checkpoint, model_dir=tmp_path / "original", gpu_wait_seconds=0
    )
    record = derived.execute(args)
    assert (
        record["status"] == "failed" and "successful immutable prepared export" in record["error"]
    )
    assert (checkpoint / "derive.json").is_file()
    assert not (checkpoint / "model").exists()
    assert not (checkpoint / "derived-checkpoint.json").exists()

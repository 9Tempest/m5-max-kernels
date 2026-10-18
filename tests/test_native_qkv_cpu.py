"""No Metal device creation: host contract, provenance, and C ABI adapter tests."""

import ctypes as ct
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from m5kernels import native_qkv as nq


@pytest.mark.parametrize(
    "rows,k,local,full,mode,padded",
    [
        (128, 128, False, False, 7, 128),
        (2048, 5376, False, True, 7, 2048),
        (2049, 5376, True, True, 5, 2176),
        (9914, 7168, True, False, 5, 9984),
    ],
)
def test_native_dispatch_defaults(rows, k, local, full, mode, padded):
    config = nq.native_dispatch(rows, k, 56)
    assert config["local_scales"] is local
    assert config["full_k"] is full
    assert config["rms_mode"] == mode
    assert config["padded_rows"] == padded
    assert config["threadgroups"] == (padded // 128 * 56 * 3, 1, 1)
    assert config["threads_per_threadgroup"] == (256, 1, 1)
    assert ("local_scales" in config["entry_point"]) is local
    assert ("k5376" in config["entry_point"]) is full
    assert config["quantizer"].endswith("_scalar")
    assert config["actual_zero_row_scale"] == np.float32(1) / np.float32(127)
    assert config["padded_zero_row_scale"] == 1
    assert struct.calcsize(config["argument_format"]) == 28
    # C struct prefix must be exactly the pinned native seven-field argument ABI.
    c = nq._Config(*config["argument_values"], padded, 123, 128)
    assert bytes(c)[:28] == struct.pack("6If", *config["argument_values"])
    assert ct.sizeof(c) == 40


@pytest.mark.parametrize(
    "rows,k,heads",
    [(127, 128, 4), (128, 129, 4), (128, 128, 3), (128, 133248, 4), (2**32 - 1, 128, 4)],
)
def test_native_unsupported_shapes(rows, k, heads):
    with pytest.raises(NotImplementedError):
        nq.native_dispatch(rows, k, heads)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, 2**32, None])
def test_shape_metadata_rejects_malformed(value):
    with pytest.raises(ValueError):
        nq.native_dispatch(value, 128, 4)


@pytest.mark.parametrize("epsilon", [0, -1, np.nan, np.inf, 1e-100, 1e100, True, [], "bad"])
def test_epsilon_contract(epsilon):
    with pytest.raises(ValueError):
        nq.native_dispatch(128, 128, 4, epsilon=epsilon)


def test_env_flags_do_not_change_declared_default(monkeypatch):
    before = nq.native_dispatch(4096, 5376, 56)
    for name in nq._FLAGS:
        monkeypatch.setenv(name, "1")
    assert nq.native_dispatch(4096, 5376, 56) == before


def test_exact_shared_buffer_shapes_and_dtypes():
    specs = nq.buffer_specs(129, 256, 4)
    assert specs["input_bf16"] == (11, (129, 256), np.dtype(np.uint16))
    assert specs["input_int8"] == (0, (256, 256), np.dtype(np.int8))
    assert specs["weight_bf16"] == (12, (1536, 256), np.dtype(np.uint16))
    assert specs["weight_int8"] == (1, (1536, 256), np.dtype(np.int8))
    assert specs["input_scales"] == (2, (256,), np.dtype(np.float32))
    assert specs["q"] == (4, (4, 129, 128), np.dtype(np.uint16))
    assert specs["k"][0] == 5 and specs["v"][0] == 6
    assert specs["cos"][1] == (129, 48)


@pytest.mark.parametrize("name", ["input_bf16", "q_gamma", "cos", "weight_int8", "input_scales"])
def test_upload_accepts_only_exact_contract(name):
    specs = nq.buffer_specs(128, 128, 4)
    _, shape, dtype = specs[name]
    valid = np.ones(shape, dtype=dtype)
    nq.validate_upload(specs, name, valid)
    with pytest.raises(ValueError, match="requires ndarray"):
        nq.validate_upload(specs, name, valid.astype(np.float64))
    with pytest.raises(ValueError, match="requires ndarray"):
        nq.validate_upload(specs, name, valid.reshape(-1)[:1])
    noncontiguous = np.repeat(valid, 2, axis=-1)[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        nq.validate_upload(specs, name, noncontiguous)


@pytest.mark.parametrize("bits", [0x7F80, 0xFF80, 0x7FC1])
def test_nonfinite_bf16_rejected(bits):
    specs = nq.buffer_specs(128, 128, 4)
    value = np.zeros((128,), np.uint16)
    value[37] = bits
    with pytest.raises(ValueError, match="nonfinite BF16"):
        nq.validate_upload(specs, "q_gamma", value)


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_scale_contract(scale):
    specs = nq.buffer_specs(128, 128, 4)
    value = np.ones((128,), np.float32)
    value[17] = scale
    with pytest.raises(ValueError, match="finite and positive"):
        nq.validate_upload(specs, "input_scales", value)


def test_minimum_int8_code_rejected():
    specs = nq.buffer_specs(128, 128, 4)
    codes = np.zeros((128, 128), np.int8)
    codes[3, 7] = -128
    with pytest.raises(ValueError, match="symmetric"):
        nq.validate_upload(specs, "input_int8", codes)


@pytest.fixture
def runner():
    obj = nq.NativeQKVRunner.__new__(nq.NativeQKVRunner)
    obj._handle = 17
    obj.specs = nq.buffer_specs(128, 128, 4)
    obj._initialized = set()
    calls = []
    buffers = {}

    def upload(handle, slot, ptr, length):
        calls.append(("upload", handle, slot, length))
        buffers[slot] = ct.create_string_buffer(ct.string_at(ptr, length), length)
        return 1

    def output(handle, slot, length):
        calls.append(("output", handle, slot))
        length._obj.value = ct.sizeof(buffers[slot])
        return ct.addressof(buffers[slot])

    def run(handle, mode, scope):
        calls.append(("run", handle, mode, scope))
        return 1

    obj._lib = SimpleNamespace(
        m5_qkv_run=run,
        m5_qkv_upload=upload,
        m5_qkv_buffer=output,
        m5_qkv_destroy=lambda handle: calls.append(("destroy", handle)),
        m5_qkv_last_gpu_seconds=lambda handle: 0.001,
        m5_qkv_capture=lambda handle, path, mode, scope: (
            calls.append(("capture", mode, scope)) or 1
        ),
    )
    yield obj, calls
    obj.close()


def test_run_has_no_output_read_or_copy(runner):
    obj, calls = runner
    obj._initialized = set(obj.specs)
    assert obj.run("candidate", "online") is obj
    assert calls == [("run", 17, 1, 1)]
    assert obj.last_gpu_seconds == 0.001
    assert {"q", "k", "v", "input_int8", "input_scales"} <= obj._initialized


def test_stage_scopes_and_missing_resident_inputs(runner):
    obj, calls = runner
    with pytest.raises(RuntimeError, match="not initialized"):
        obj.run("native", "online")
    assert calls == []
    obj._initialized.add("weight_bf16")
    obj.run(scope="weight_quantize")
    assert calls == [("run", 17, 0, 3)]
    assert {"weight_int8", "weight_scales"} <= obj._initialized
    obj._initialized.add("input_bf16")
    obj.run(scope="quantize")
    assert calls[-1] == ("run", 17, 0, 2)
    assert "q" not in obj._initialized


def test_upload_invalidates_stale_quantization_and_read_is_shared(runner):
    obj, calls = runner
    obj._initialized = set(obj.specs)
    original = np.full((128, 128), 0x3F80, dtype=np.uint16)
    obj.upload("input_bf16", original)
    assert "input_int8" not in obj._initialized
    assert "input_scales" not in obj._initialized
    assert np.all(original == 0x3F80)
    view = obj.output("input_bf16")
    assert view.shape == original.shape and view.dtype == np.uint16
    assert not view.flags.writeable
    np.testing.assert_array_equal(view, original)
    another = obj.output("input_bf16")
    assert np.shares_memory(view, another)
    assert len([call for call in calls if call[0] == "output"]) == 2
    assert obj.output("input_bf16").base is not None


def test_invalid_dispatch_rejected_before_c_call(runner):
    obj, calls = runner
    with pytest.raises(ValueError):
        obj.run(mode="fallback")
    with pytest.raises(ValueError):
        obj.run(scope="everything")
    assert calls == []


def test_close_idempotent_and_blocks_use(runner):
    obj, calls = runner
    obj.close()
    obj.close()
    assert calls == [("destroy", 17)]
    with pytest.raises(RuntimeError, match="closed"):
        obj.run()


def test_capture_refuses_overwrite_and_uses_same_run_scope(runner, tmp_path):
    obj, calls = runner
    obj._initialized = set(obj.specs)
    existing = tmp_path / "existing.gputrace"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        obj.capture(existing)
    assert calls == []
    obj.capture(tmp_path / "new.gputrace", mode="candidate", scope="online")
    assert calls == [("capture", 1, 1)]


def test_verify_upstream_requires_exact_pinned_bytes(monkeypatch, tmp_path):
    content = {name: name.encode() for name in ("h3_shaders.metal", "h3_gpu.m", "LICENSE")}
    for name, data in content.items():
        (tmp_path / name).write_bytes(data)

    def git(args):
        return (
            (nq.UPSTREAM_COMMIT + "\n").encode()
            if args[-1] == "HEAD"
            else content[args[-1].split(":")[1]]
        )

    monkeypatch.setattr(nq.subprocess, "check_output", git)
    result = nq.verify_upstream(tmp_path)
    assert result["commit"] == nq.UPSTREAM_COMMIT
    assert (
        result["source_sha256"]["h3_shaders.metal"]
        == hashlib.sha256(content["h3_shaders.metal"]).hexdigest()
    )
    (tmp_path / "h3_shaders.metal").write_text("modified")
    with pytest.raises(ValueError, match="local modifications"):
        nq.verify_upstream(tmp_path)


def test_verify_upstream_rejects_other_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(nq.subprocess, "check_output", lambda _: b"other-revision\n")
    with pytest.raises(ValueError, match="HEAD must be"):
        nq.verify_upstream(tmp_path)


def test_nonmac_build_is_explicitly_unsupported(monkeypatch):
    monkeypatch.setattr(nq.platform, "system", lambda: "Linux")
    with pytest.raises(NotImplementedError, match="macOS arm64"):
        nq.build_bridge()


def test_constructor_binds_shared_abi_without_device_in_cpu_test(monkeypatch, tmp_path):
    seen = {}

    def create(native_path, candidate_path, native_entry, candidate_entry, config):
        seen.update(
            native_path=native_path,
            candidate_source=Path(candidate_path.decode()).read_text(),
            native_entry=native_entry,
            candidate_entry=candidate_entry,
            config=config._obj,
        )
        return 23

    library = SimpleNamespace(
        m5_qkv_create=create,
        m5_qkv_metadata=lambda _: b'{"device":"fake CPU fixture"}',
        m5_qkv_destroy=lambda handle: seen.update(destroyed=handle),
    )
    monkeypatch.setattr(nq, "verify_upstream", lambda _: {"commit": nq.UPSTREAM_COMMIT})
    monkeypatch.setattr(nq, "build_bridge", lambda _: (tmp_path / "fake.dylib", {}))
    monkeypatch.setattr(nq, "_bind", lambda _: library)
    monkeypatch.setenv("H3_INT8_VECTOR_QUANT", "1")
    with nq.NativeQKVRunner(
        129,
        128,
        4,
        candidate_source="// CPU fixture; deliberately never compiled",
        candidate_tile=(64, 64, 4, 8),
        upstream_root=tmp_path,
        cache_dir=tmp_path,
    ) as obj:
        assert seen["config"].padded_rows == 256
        assert seen["config"].candidate_groups == 36
        assert seen["config"].candidate_threads == 128
        assert seen["config"].rms_mode == 7
        assert seen["candidate_entry"] == b"grouped_qkv"
        assert seen["native_entry"] == b"h3_qkv_project_split_int8_rope_nax_r128_morton4"
        assert obj.metadata["device"] == "fake CPU fixture"
        assert "H3_INT8_VECTOR_QUANT" in obj.metadata["ignored_environment_flags_present"]
        assert obj.metadata["native_dispatch"]["quantizer"] == nq.QUANTIZER
        assert obj.metadata["native_dispatch"]["threadgroups"] == (24, 1, 1)
        assert obj.metadata["candidate_dispatch"]["threadgroups"] == (36, 1, 1)
    assert seen["destroyed"] == 23


def test_bridge_source_contains_no_vendored_shader_body():
    source = Path(nq.BRIDGE_SOURCE).read_text()
    assert "kernel void" not in source
    assert "MTLMathModeSafe" in source
    # The Python-hosted runtime otherwise defaults below TensorOps' language version.
    assert "options.languageVersion = MTLLanguageVersion4_1;" in source
    assert '"Metal 4.1 (explicit, both libraries)"' in source
    assert "newLibraryWithSource:source" in source

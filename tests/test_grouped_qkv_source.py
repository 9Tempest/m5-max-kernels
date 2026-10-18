import hashlib
import importlib.util
from pathlib import Path
import struct
import sys

import pytest

from m5kernels.grouped_qkv import dispatch_config, generate, pack_arguments


@pytest.mark.parametrize("tm", [32, 64, 128])
@pytest.mark.parametrize("tn", [64, 128])
@pytest.mark.parametrize("groups", [4, 8])
@pytest.mark.parametrize("rms_order", ["simd", "native"])
@pytest.mark.parametrize("scratch", ["threadgroup", "device"])
@pytest.mark.parametrize(
    "k_specialization,input_dim,k_tile",
    [("dynamic", 5376, None), ("native", 5376, 5376), ("native", 256, 128)],
)
def test_all_supported_sources_are_fully_instantiated(
    tm, tn, groups, rms_order, scratch, k_specialization, input_dim, k_tile
):
    tile = (tm, tn, groups, 8)
    source = generate(
        tile,
        rms_order=rms_order,
        k_specialization=k_specialization,
        input_dim=input_dim,
        scratch=scratch,
    )
    assert "$" not in source
    assert source.count("kernel void grouped_qkv(") == 1
    assert f"TM={tm}, TN={tn}, GROUPS={groups}, SWIZZLE=8" in source
    assert f"NATIVE_RMS={'true' if rms_order == 'native' else 'false'}" in source
    assert f"#if {int(k_specialization == 'native')}" in source
    assert f"#define GROUPED_QKV_DEVICE_SCRATCH {int(scratch == 'device')}" in source
    config = dispatch_config(
        17,
        input_dim,
        56,
        tile,
        rms_order=rms_order,
        k_specialization=k_specialization,
        scratch=scratch,
    )
    assert config["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert config["threadgroup_memory_bytes"] == (tm * 128 * 2 if scratch == "threadgroup" else 0)
    assert config["scratch"] == scratch
    assert config["k_specialization"] == k_specialization
    assert config["k_tile"] == k_tile
    if k_tile is not None:
        assert f"constexpr uint K_TILE={k_tile};" in source


def test_dynamic_default_is_independent_of_optional_k_value():
    assert generate() == generate(k_specialization="dynamic", input_dim=5376)
    assert generate() == generate(k_specialization="dynamic", input_dim=17)
    assert dispatch_config(17, 17, 4)["k_tile"] is None


def test_threadgroup_scratch_remains_default_without_changing_native_abi():
    assert generate() == generate(scratch="threadgroup")
    shared = dispatch_config(129, 5376, 56)
    device = dispatch_config(129, 5376, 56, scratch="device")
    assert shared["scratch"] == "threadgroup"
    assert shared["source_sha256"] != device["source_sha256"]
    for key in ("buffer_bindings", "buffer_shapes", "threadgroups", "threads_per_threadgroup"):
        assert shared[key] == device[key]
    assert pack_arguments(shared) == pack_arguments(device)


@pytest.mark.parametrize("scratch", ["unknown", "global", "", None, True])
def test_invalid_scratch_is_rejected_before_driver_use(scratch):
    with pytest.raises(ValueError, match="scratch"):
        generate(scratch=scratch)
    with pytest.raises(ValueError, match="scratch"):
        dispatch_config(129, 5376, 56, scratch=scratch)


@pytest.mark.parametrize("input_dim,k_tile", [(128, 128), (256, 128), (5376, 5376), (7168, 128)])
def test_native_k_policy_has_exact_full_or_128_wide_reduction(input_dim, k_tile):
    config = dispatch_config(129, input_dim, 4, k_specialization="native")
    assert config["k_tile"] == k_tile
    assert input_dim % k_tile == 0
    source = generate(k_specialization="native", input_dim=input_dim)
    assert "tensor<device int8_t,dextents<int32_t,2>,tensor_inline>" in source
    assert "matmul2d_descriptor::mode::multiply_accumulate" in source
    assert "if(acc.is_valid_element(i)) acc[i]=0;" in source
    assert "for(uint k=0;k<args.input_dim;k+=K_TILE)" in source


@pytest.mark.parametrize("input_dim", [None, 0, True, -128, 1.5])
def test_native_k_requires_positive_integer_input_dim(input_dim):
    with pytest.raises(ValueError, match="input_dim"):
        generate(k_specialization="native", input_dim=input_dim)


@pytest.mark.parametrize("input_dim", [1, 127, 129, 5375, 5377, 131072])
def test_native_k_rejects_unsupported_tail_or_overflow(input_dim):
    with pytest.raises(NotImplementedError):
        generate(k_specialization="native", input_dim=input_dim)
    with pytest.raises(NotImplementedError):
        dispatch_config(129, input_dim, 4, k_specialization="native")


@pytest.mark.parametrize("rows", [1, 17, 128, 129, 4096, 9914])
def test_native_argument_and_buffer_abi(rows):
    config = dispatch_config(rows, 5376, 56)
    packed = pack_arguments(config)
    assert len(packed) == struct.calcsize("6If") == config["argument_nbytes"] == 28
    fields = struct.unpack("6If", packed)
    assert fields[:5] == (rows, 5376, 56, 128, 48)
    assert fields[5] == (7 if rows <= 2048 else 5)
    assert fields[6] == pytest.approx(1e-5, rel=1e-7)
    assert config["argument_buffer_index"] == 11
    assert config["buffer_bindings"] == {
        "A": 0,
        "W": 1,
        "AS": 2,
        "BS": 3,
        "Q": 4,
        "K": 5,
        "V": 6,
        "QGamma": 7,
        "KGamma": 8,
        "Cos": 9,
        "Sin": 10,
    }
    padded = (rows + 127) // 128 * 128
    assert config["padded_rows"] == padded
    assert config["buffer_shapes"]["A"] == (padded, 5376)
    assert config["buffer_shapes"]["AS"] == (padded,)
    assert config["buffer_shapes"]["W"] == (56, 3, 128, 5376)
    assert config["buffer_shapes"]["BS"] == (56, 3, 128)
    for stream in ("Q", "K", "V"):
        assert config["buffer_shapes"][stream] == (56, rows, 128)
    assert config["threadgroups"] == (((rows + 63) // 64) * 56 * 3, 1, 1)
    assert config["threads_per_threadgroup"] == (128, 1, 1)


@pytest.mark.parametrize("rows", [1, 33, 65, 129, 257, 9914])
@pytest.mark.parametrize("heads", [1, 3, 56])
@pytest.mark.parametrize("tm", [32, 64, 128])
def test_swizzled_launch_covers_each_row_tile_and_grouped_stream_once(rows, heads, tm):
    config = dispatch_config(rows, 256, heads, (tm, 64, 4, 8))
    row_tiles = (rows + tm - 1) // tm
    column_tiles = heads * 3
    visited = []
    for group_id in range(config["threadgroups"][0]):
        band, inner = divmod(group_id, column_tiles * 8)
        start = band * 8
        height = min(8, row_tiles - start)
        grouped_column, local_row_tile = divmod(inner, height)
        head, stream = divmod(grouped_column, 3)
        visited.append((start + local_row_tile, head, stream))
    expected = {
        (row, head, stream)
        for row in range(row_tiles)
        for head in range(heads)
        for stream in range(3)
    }
    assert len(visited) == len(expected)
    assert set(visited) == expected


@pytest.mark.parametrize("flags", [1, 3, 5, 7, True])
def test_native_head_major_flags_preserved(flags):
    assert dispatch_config(17, 256, 4, head_major=flags)["argument_values"][5] == int(flags)


@pytest.mark.parametrize("epsilon", [0, -1, float("nan"), float("inf"), 1e-100, 1e100, True])
def test_invalid_epsilon_is_rejected_before_driver_use(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        dispatch_config(17, 256, 4, epsilon=epsilon)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"rows": 0}, ValueError),
        ({"rows": True}, ValueError),
        ({"heads": -1}, ValueError),
        ({"input_dim": 131072}, NotImplementedError),
        ({"head_dim": 64}, NotImplementedError),
        ({"rope_half": 64}, NotImplementedError),
        ({"head_major": 0}, NotImplementedError),
        ({"head_major": 2}, NotImplementedError),
        ({"head_major": 9}, NotImplementedError),
        ({"head_major": 1.0}, NotImplementedError),
        ({"rows": 2**31}, ValueError),
        ({"heads": 2**31}, ValueError),
        ({"tile": (16, 128, 4, 8)}, NotImplementedError),
        ({"tile": (32, 32, 4, 8)}, NotImplementedError),
        ({"tile": (32, 128, 2, 8)}, NotImplementedError),
        ({"tile": (32, 128, 4, 0)}, ValueError),
        ({"tile": (True, 128, 4, 8)}, ValueError),
        ({"rms_order": "unknown"}, ValueError),
        ({"k_specialization": "unknown"}, ValueError),
    ],
)
def test_explicit_contract_rejections(kwargs, error):
    config = dict(rows=17, input_dim=256, heads=4)
    config.update(kwargs)
    with pytest.raises(error):
        dispatch_config(**config)


def test_module_and_source_generation_do_not_require_mlx(monkeypatch):
    # Import this module under a fresh name with MLX deliberately unavailable.
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    path = Path(__file__).parents[1] / "src" / "m5kernels" / "grouped_qkv.py"
    spec = importlib.util.spec_from_file_location("grouped_qkv_cpu_only", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.dispatch_config(17, 256, 4)["entry_point"] == "grouped_qkv"
    assert "kernel void grouped_qkv" in module.generate()

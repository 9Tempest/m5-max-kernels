"""CPU-only source/config generation for the standalone grouped-QKV kernel.

The candidate consumes the pinned native runtime's packed INT8 buffers and
returns all three BF16 head-major streams. This module neither quantizes
operands nor creates a Metal device; native and candidate sources are loaded
by the same external driver with identical compiler options.
"""

import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import struct

ENTRY_POINT = "grouped_qkv"
ARGUMENT_FORMAT = "6If"
HEAD_DIM = 128
ROPE_HALF = 48
STREAMS = 3
INPUT_ROW_ALIGNMENT = 128
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_SOURCE = Path(__file__).parent / "shaders" / "grouped_qkv.metal"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _tile(tile):
    if not isinstance(tile, (tuple, list)) or len(tile) != 4:
        raise ValueError("tile must contain TM, TN, SIMD groups and M swizzle")
    tm, tn, groups, swizzle = (
        _positive_integer(value, name)
        for value, name in zip(tile, ("TM", "TN", "SIMD groups", "M swizzle"), strict=True)
    )
    if tm not in (32, 64, 128) or tn not in (64, 128) or groups not in (4, 8):
        raise NotImplementedError("supported tiles use TM32/64/128, TN64/128 and 4/8 SIMD groups")
    if swizzle > _INT32_MAX:
        raise ValueError("M swizzle exceeds the supported integer range")
    return tm, tn, groups, swizzle


def _rms_order(rms_order):
    if rms_order not in ("simd", "native"):
        raise ValueError("rms_order must be 'simd' or 'native'")
    return rms_order


def _scratch(scratch):
    if scratch not in ("threadgroup", "device"):
        raise ValueError("scratch must be 'threadgroup' or 'device'")
    return scratch


def _k_specialization(mode, input_dim):
    if mode not in ("dynamic", "native"):
        raise ValueError("k_specialization must be 'dynamic' or 'native'")
    if input_dim is not None:
        input_dim = _positive_integer(input_dim, "input_dim")
        if input_dim > 131071:
            raise NotImplementedError("INT32 worst-case overflow bound exceeded for signed INT8")
    if mode == "dynamic":
        return None
    if input_dim is None:
        raise ValueError("native K specialization requires input_dim")
    if input_dim % 128:
        raise NotImplementedError("native K specialization requires input_dim divisible by 128")
    return 5376 if input_dim == 5376 else 128


def _epsilon(epsilon):
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be positive and finite")
    try:
        result = struct.unpack("f", struct.pack("f", float(epsilon)))[0]
    except OverflowError:
        raise ValueError("epsilon must remain positive and finite in FP32") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError("epsilon must remain positive and finite in FP32")
    return result


def generate(
    tile=(64, 128, 4, 8),
    *,
    rms_order="simd",
    k_specialization="dynamic",
    input_dim=None,
    scratch="threadgroup",
):
    """Return complete original MSL with stable entry point ``grouped_qkv``.

    TN64 executes two half-head products within one threadgroup; TN128
    executes one complete-head product. ``native`` uses channel-ordered
    128-element FMA accumulation for RMS, while ``simd`` reduces four
    channels per lane. Both preserve the same BF16 operation boundaries.
    Use the same safe Metal math options as the unchanged native source.

    ``k_specialization='native'`` requires the launch's ``input_dim`` and
    follows the pinned native TensorOps reduction policy: a static full-K
    slice for 5376, otherwise static 128-wide multiply-accumulate slices.
    Other positive K multiples of 128 are supported; tails are rejected.
    The default ``dynamic`` retains the runtime-K multiply path.

    ``scratch='device'`` stores rounded Q/K projection intermediates in
    their final output buffers, with a device-memory threadgroup barrier
    before RMS/RoPE. It needs no threadgroup scratch. The default retains
    the BF16 projection in threadgroup storage. V always stores directly.
    """
    tm, tn, groups, swizzle = _tile(tile)
    order = _rms_order(rms_order)
    k_tile = _k_specialization(k_specialization, input_dim)
    scratch = _scratch(scratch)
    source = _SOURCE.read_text()
    for token, value in {
        "$TM": str(tm),
        "$TN": str(tn),
        "$GROUPS": str(groups),
        "$SWIZZLE": str(swizzle),
        "$NATIVE_RMS": "true" if order == "native" else "false",
        "$NATIVE_K": "1" if k_tile is not None else "0",
        "$K_TILE": str(k_tile or 128),
        "$DEVICE_SCRATCH": "1" if scratch == "device" else "0",
    }.items():
        source = source.replace(token, value)
    return source


def dispatch_config(
    rows,
    input_dim,
    heads,
    tile=(64, 128, 4, 8),
    *,
    head_dim=128,
    rope_half=48,
    head_major=None,
    epsilon=1e-5,
    rms_order="simd",
    k_specialization="dynamic",
    scratch="threadgroup",
):
    """Describe a launch and the exact native buffer/argument ABI, without GPU work.

    A and activation scales include zero-padded rows to a multiple of 128;
    outputs contain only logical rows. W/weight scales use grouped order
    [head,Q/K/V,128,K], and Q/K/V each use contiguous [head,row,128].

    ``head_major`` is the native flag word. Its low bit must be one; bits
    1/2 are accepted native optimization flags, not candidate layout modes.
    The candidate's RMS order is selected independently at compilation.
    By default the native flags are 7 through 2048 rows and 5 above that.
    K specialization is independent of RMS order and the native flag word.
    """
    rows = _positive_integer(rows, "rows")
    input_dim = _positive_integer(input_dim, "input_dim")
    heads = _positive_integer(heads, "heads")
    head_dim = _positive_integer(head_dim, "head_dim")
    rope_half = _positive_integer(rope_half, "rope_half")
    if head_dim != HEAD_DIM or rope_half != ROPE_HALF:
        raise NotImplementedError("grouped H3 QKV requires head_dim=128 and rope_half=48")
    if input_dim > 131071:
        raise NotImplementedError("INT32 worst-case overflow bound exceeded for signed INT8")
    tm, tn, groups, swizzle = _tile(tile)
    order = _rms_order(rms_order)
    k_tile = _k_specialization(k_specialization, input_dim)
    scratch = _scratch(scratch)
    if head_major is None:
        head_major = 7 if rows <= 2048 else 5
    if isinstance(head_major, bool):
        head_major = int(head_major)
    if not isinstance(head_major, Integral) or head_major not in (1, 3, 5, 7):
        raise NotImplementedError(
            "head_major must set bit0, with only native flag bits1/2 optional"
        )
    epsilon = _epsilon(epsilon)
    padded_rows = (rows + INPUT_ROW_ALIGNMENT - 1) // INPUT_ROW_ALIGNMENT * INPUT_ROW_ALIGNMENT
    columns = heads * STREAMS * HEAD_DIM
    row_tiles = (rows + tm - 1) // tm
    group_count = row_tiles * heads * STREAMS
    if padded_rows > _INT32_MAX or columns > _INT32_MAX:
        raise ValueError("matrix extents exceed the signed 32-bit TensorOps range")
    if group_count > _UINT32_MAX or heads * STREAMS * swizzle > _UINT32_MAX:
        raise ValueError("dispatch or swizzle indexing exceeds the unsigned 32-bit range")
    values = (rows, input_dim, heads, HEAD_DIM, ROPE_HALF, int(head_major), epsilon)
    shapes = {
        "A": (padded_rows, input_dim),
        "W": (heads, STREAMS, HEAD_DIM, input_dim),
        "AS": (padded_rows,),
        "BS": (heads, STREAMS, HEAD_DIM),
        "Q": (heads, rows, HEAD_DIM),
        "K": (heads, rows, HEAD_DIM),
        "V": (heads, rows, HEAD_DIM),
        "QGamma": (HEAD_DIM,),
        "KGamma": (HEAD_DIM,),
        "Cos": (rows, ROPE_HALF),
        "Sin": (rows, ROPE_HALF),
    }
    return dict(
        entry_point=ENTRY_POINT,
        tile=(tm, tn, groups, swizzle),
        rms_order=order,
        k_specialization=k_specialization,
        k_tile=k_tile,
        scratch=scratch,
        threadgroups=(group_count, 1, 1),
        threads_per_threadgroup=(groups * 32, 1, 1),
        argument_values=values,
        argument_format=ARGUMENT_FORMAT,
        argument_nbytes=struct.calcsize(ARGUMENT_FORMAT),
        argument_buffer_index=11,
        padded_rows=padded_rows,
        threadgroup_memory_bytes=tm * HEAD_DIM * 2 if scratch == "threadgroup" else 0,
        buffer_shapes=shapes,
        buffer_bindings={name: index for index, name in enumerate(shapes)},
        output_layout="three separate contiguous BF16 [heads,rows,128] buffers",
        weight_layout="INT8 [heads,3,128,input_dim], streams Q,K,V",
        source_sha256=hashlib.sha256(
            generate(
                tile,
                rms_order=order,
                k_specialization=k_specialization,
                input_dim=input_dim,
                scratch=scratch,
            ).encode()
        ).hexdigest(),
    )


def pack_arguments(config):
    """Pack six uint32 fields and one FP32 epsilon for buffer index 11."""
    return struct.pack(config["argument_format"], *config["argument_values"])

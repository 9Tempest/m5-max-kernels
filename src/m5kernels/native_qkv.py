"""Shared Objective-C/Metal host for pinned h3.c and experimental grouped QKV.

Importing this module, validating a shape, checking provenance, or compiling the
host library does not create a Metal device. Constructing ``NativeQKVRunner``
does, and must run under the repository's GPU lock. Both projections use the
same resident buffers, command queue, safe math options, native scalar quantizer,
and BF16 boundaries. ``run`` submits once and waits; it never reads output data.

Upstream shaders are loaded unchanged from a verified local checkout. h3.c is
MIT licensed, copyright 2026 Salvatore Sanfilippo; the host source includes its
notice for the upstream dispatch selection reproduced there. No shader is
vendored and no source/runtime files in the upstream checkout are modified.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
from numbers import Real
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np

UPSTREAM_COMMIT = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
UPSTREAM_URL = "https://github.com/antirez/h3.c"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SOURCE = Path(__file__).parent / "native" / "qkv_runner.m"
QUANTIZER = "h3_quantize_bf16_int8_rows_scalar"
_MODES = {"native": 0, "candidate": 1}
_SCOPES = {"core": 0, "online": 1, "quantize": 2, "weight_quantize": 3}
_FLAGS = (
    "H3_NAX",
    "H3_INT8_VECTOR_QUANT",
    "H3_DISABLE_FUSED_INT8_QKV_ROPE",
    "H3_DISABLE_QKV_LOCAL_SCALES",
    "H3_DISABLE_QKV_FULL_K",
    "H3_DISABLE_VECTOR_QKV_RMS",
    "H3_DISABLE_VECTOR_QKV_ROPE",
)


class NativeQKVError(RuntimeError):
    """Explicit compilation, allocation, or dispatch failure; no fallback."""


class _Config(ct.Structure):
    _fields_ = [
        (name, ct.c_uint32)
        for name in ("rows", "input_dim", "heads", "head_dim", "rope_half", "rms_mode")
    ] + [
        ("epsilon", ct.c_float),
        ("padded_rows", ct.c_uint32),
        ("candidate_groups", ct.c_uint32),
        ("candidate_threads", ct.c_uint32),
    ]


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0 or value > np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit a positive uint32")
    return int(value)


def native_dispatch(rows, input_dim, heads, *, head_dim=128, rope_half=48, epsilon=1e-5):
    """CPU-only default h3_gpu.m dispatch policy, with external H3 flags ignored."""
    rows = _positive_int(rows, "rows")
    input_dim = _positive_int(input_dim, "input_dim")
    heads = _positive_int(heads, "heads")
    head_dim = _positive_int(head_dim, "head_dim")
    rope_half = _positive_int(rope_half, "rope_half")
    if rows < 128 or input_dim % 128 or heads % 4 or head_dim != 128 or rope_half != 48:
        raise NotImplementedError(
            "native grouped QKV requires M>=128, K%128=0, H%4=0, D128, RoPE48"
        )
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise ValueError("epsilon must be a finite positive FP32 scalar")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("epsilon must be a finite positive FP32 scalar") from exc
    if not np.isfinite(epsilon) or epsilon < float(np.finfo(np.float32).smallest_subnormal):
        raise ValueError("epsilon must be a finite positive FP32 scalar")
    if epsilon > float(np.finfo(np.float32).max):
        raise ValueError("epsilon must be representable in FP32")
    if input_dim > np.iinfo(np.int32).max // (127 * 127):
        raise NotImplementedError("INT8 reduction may overflow INT32")
    padded = (rows + 127) // 128 * 128
    # Upstream shader indexing uses uint, including flattened strides.
    if max(padded * input_dim, heads * 3 * 128 * input_dim, heads * rows * 128) > 2**32 - 1:
        raise NotImplementedError("native flattened indexing exceeds uint32")
    local = rows > 2048
    full_k = input_dim == 5376
    suffix = "local_scales_" if local else ""
    suffix += "nax_r128_"
    suffix += "k5376_" if full_k else ""
    pipeline = f"h3_qkv_project_split_int8_rope_{suffix}morton4"
    rms_mode = 5 if local else 7
    return {
        "entry_point": pipeline,
        "threadgroups": (padded // 128 * heads * 3, 1, 1),
        "threads_per_threadgroup": (256, 1, 1),
        "argument_values": (
            rows,
            input_dim,
            heads,
            head_dim,
            rope_half,
            rms_mode,
            float(np.float32(epsilon)),
        ),
        "argument_format": "6If",
        "padded_rows": padded,
        "local_scales": local,
        "full_k": full_k,
        "rms_mode": rms_mode,
        "quantizer": QUANTIZER,
        "quantizer_clip": 1.0,
        "quantizer_threads": 256,
        "actual_zero_row_scale": float(np.float32(1) / np.float32(127)),
        "padded_zero_row_scale": 1.0,
        "output_layout": "head_major",
    }


def buffer_specs(rows, input_dim, heads):
    """Map name to (C slot, exact NumPy shape, dtype); BF16 is uint16 bits."""
    native = native_dispatch(rows, input_dim, heads)
    padded = native["padded_rows"]
    weights = heads * 3 * 128
    raw = [
        ("input_int8", (padded, input_dim), np.int8),
        ("weight_int8", (weights, input_dim), np.int8),
        ("input_scales", (padded,), np.float32),
        ("weight_scales", (weights,), np.float32),
        ("q", (heads, rows, 128), np.uint16),
        ("k", (heads, rows, 128), np.uint16),
        ("v", (heads, rows, 128), np.uint16),
        ("q_gamma", (128,), np.uint16),
        ("k_gamma", (128,), np.uint16),
        ("cos", (rows, 48), np.uint16),
        ("sin", (rows, 48), np.uint16),
        ("input_bf16", (rows, input_dim), np.uint16),
        ("weight_bf16", (weights, input_dim), np.uint16),
    ]
    return {name: (index, shape, np.dtype(dtype)) for index, (name, shape, dtype) in enumerate(raw)}


def validate_upload(specs, name, array):
    """Validate resident data without conversion, reshaping, or GPU work."""
    if name not in specs:
        raise ValueError(f"unknown resident buffer {name!r}")
    _, shape, dtype = specs[name]
    if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
        raise ValueError(f"{name} requires ndarray shape {shape}, dtype {dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C contiguous")
    if dtype == np.uint16:
        if np.any((array & np.uint16(0x7F80)) == np.uint16(0x7F80)):
            raise ValueError(f"{name} contains nonfinite BF16 bits")
    elif dtype == np.float32:
        if not np.all(np.isfinite(array)) or not np.all(array > 0):
            raise ValueError(f"{name} scales must be finite and positive")
    elif np.any(array == -128):
        raise ValueError(f"{name} must use symmetric codes [-127,127]")


def verify_upstream(upstream_root=None):
    """Verify HEAD and exact shader/host/license bytes against the pinned commit."""
    root = Path(upstream_root) if upstream_root is not None else REPOSITORY_ROOT / ".cache/h3.c"
    if not root.is_dir():
        raise NotImplementedError(
            f"local h3.c checkout required at pinned commit {UPSTREAM_COMMIT}"
        )
    try:
        revision = (
            subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
        )
        if revision != UPSTREAM_COMMIT:
            raise ValueError(f"h3.c HEAD must be {UPSTREAM_COMMIT}, got {revision}")
        hashes = {}
        for filename in ("h3_shaders.metal", "h3_gpu.m", "LICENSE"):
            expected = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{revision}:{filename}"]
            )
            actual = (root / filename).read_bytes()
            if actual != expected:
                raise ValueError(f"pinned h3.c {filename} has local modifications")
            hashes[filename] = hashlib.sha256(actual).hexdigest()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NotImplementedError("cannot read pinned local h3.c checkout") from exc
    return {
        "repository": UPSTREAM_URL,
        "commit": revision,
        "source_sha256": hashes,
        "license": "MIT; Copyright (c) 2026 Salvatore Sanfilippo",
        "shader_policy": "load verified upstream source bytes unchanged",
    }


def build_bridge(cache_dir=None):
    """Compile the C ABI host only. Does not load Metal or create a device."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise NotImplementedError("native QKV host requires macOS arm64 with the Metal SDK")
    try:
        compiler = subprocess.check_output(["xcrun", "--find", "clang"], text=True).strip()
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NotImplementedError("Xcode clang/SDK required for native QKV bridge") from exc
    source = BRIDGE_SOURCE.read_bytes()
    digest = hashlib.sha256(source + version.encode() + sdk.encode()).hexdigest()
    cache = Path(cache_dir) if cache_dir is not None else REPOSITORY_ROOT / ".cache/native-qkv"
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"qkv_runner-{digest[:24]}.dylib"
    start = time.perf_counter()
    cache_hit = output.exists()
    if not cache_hit:
        fd, temporary = tempfile.mkstemp(prefix="qkv-runner-", suffix=".dylib", dir=cache)
        os.close(fd)
        try:
            result = subprocess.run(
                [
                    compiler,
                    "-O2",
                    "-fobjc-arc",
                    "-dynamiclib",
                    "-isysroot",
                    sdk,
                    "-framework",
                    "Foundation",
                    "-framework",
                    "Metal",
                    str(BRIDGE_SOURCE),
                    "-o",
                    temporary,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                raise NativeQKVError(f"native QKV host compilation failed:\n{result.stderr}")
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return output, {
        "bridge_source_sha256": hashlib.sha256(source).hexdigest(),
        "bridge_compiler": version,
        "bridge_build_seconds": time.perf_counter() - start,
        "bridge_build_cache_hit": cache_hit,
    }


def _bind(path):
    lib = ct.CDLL(str(path))
    signatures = {
        "m5_qkv_error": (ct.c_char_p, []),
        "m5_qkv_create": (ct.c_void_p, [ct.c_char_p] * 4 + [ct.POINTER(_Config)]),
        "m5_qkv_destroy": (None, [ct.c_void_p]),
        "m5_qkv_metadata": (ct.c_char_p, [ct.c_void_p]),
        "m5_qkv_buffer": (ct.c_void_p, [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint64)]),
        "m5_qkv_upload": (ct.c_int, [ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.c_uint64]),
        "m5_qkv_run": (ct.c_int, [ct.c_void_p, ct.c_uint32, ct.c_uint32]),
        "m5_qkv_capture": (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_uint32, ct.c_uint32]),
        "m5_qkv_last_gpu_seconds": (ct.c_double, [ct.c_void_p]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(lib, name)
        function.restype = result
        function.argtypes = arguments
    return lib


class NativeQKVRunner:
    """Resident common-host comparison. Not thread safe; use the GPU lock.

    ``output`` exposes read-only shared storage, without a GPU-to-CPU copy. Views
    remain valid until ``close`` and change after subsequent dispatches; call
    ``copy()`` outside timing to retain a result. BF16 arrays are raw uint16 bits.
    Projection weights have flattened [head,QKV,channel,K] order.
    """

    def __init__(
        self,
        rows,
        input_dim,
        heads,
        *,
        candidate_source=None,
        candidate_path=None,
        candidate_entry_point="grouped_qkv",
        candidate_tile=(64, 128, 4, 8),
        upstream_root=None,
        epsilon=1e-5,
        cache_dir=None,
    ):
        self._handle = None
        native = native_dispatch(rows, input_dim, heads, epsilon=epsilon)
        self.specs = buffer_specs(rows, input_dim, heads)
        if candidate_source is not None and candidate_path is not None:
            raise ValueError("provide candidate_source or candidate_path, not both")
        if not isinstance(candidate_entry_point, str) or not candidate_entry_point.isidentifier():
            raise ValueError("candidate_entry_point must be an identifier")
        if not isinstance(candidate_tile, (tuple, list)) or len(candidate_tile) != 4:
            raise ValueError("candidate_tile must contain TM,TN,SIMD groups,swizzle")
        tile = tuple(_positive_int(v, "candidate_tile value") for v in candidate_tile)
        from .grouped_qkv import dispatch_config

        candidate = dispatch_config(rows, input_dim, heads, tile, epsilon=epsilon)
        candidate_groups = candidate["threadgroups"][0]
        groups = tile[2]
        provenance = verify_upstream(upstream_root)
        upstream = (
            Path(upstream_root) if upstream_root is not None else REPOSITORY_ROOT / ".cache/h3.c"
        )
        cache = Path(cache_dir) if cache_dir is not None else REPOSITORY_ROOT / ".cache/native-qkv"
        if candidate_path is None:
            if candidate_source is None:
                from .grouped_qkv import generate

                candidate_source = generate(tile=tile)
            if not isinstance(candidate_source, str) or not candidate_source.strip():
                raise ValueError("candidate_source must be nonempty Metal source")
            cache.mkdir(parents=True, exist_ok=True)
            source_bytes = candidate_source.encode()
            candidate_path = cache / f"candidate-{hashlib.sha256(source_bytes).hexdigest()}.metal"
            if candidate_path.exists() and candidate_path.read_bytes() != source_bytes:
                raise NativeQKVError("candidate cache content does not match its source hash")
            if not candidate_path.exists():
                candidate_path.write_bytes(source_bytes)
        else:
            candidate_path = Path(candidate_path)
            source_bytes = candidate_path.read_bytes()
        library, build_meta = build_bridge(cache)
        self._lib = _bind(library)
        config = _Config(
            *native["argument_values"], native["padded_rows"], candidate_groups, groups * 32
        )
        self._handle = self._lib.m5_qkv_create(
            os.fsencode(upstream / "h3_shaders.metal"),
            os.fsencode(candidate_path),
            native["entry_point"].encode(),
            candidate_entry_point.encode(),
            ct.byref(config),
        )
        if not self._handle:
            self._raise_error()
        self._initialized = set()
        self.metadata = {
            **json.loads(self._lib.m5_qkv_metadata(self._handle)),
            **build_meta,
            "upstream": provenance,
            "native_dispatch": native,
            "candidate_dispatch": {
                "entry_point": candidate_entry_point,
                "tile": tile,
                "threadgroups": (candidate_groups, 1, 1),
                "threads_per_threadgroup": (groups * 32, 1, 1),
                "argument_values": native["argument_values"],
                "argument_format": "6If",
            },
            "candidate_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "environment_policy": "fixed upstream defaults; all external H3 feature flags ignored",
            "ignored_environment_flags_present": [name for name in _FLAGS if name in os.environ],
            "timing_scope": "synchronized host wall; one command buffer; no output reads",
            "online_dispatches": 2,
            "core_dispatches": 1,
            "weight_layout": "[head,QKV,128,K] flattened to [heads*384,K]",
            "output_layout": "three BF16 [heads,rows,128] buffers",
            "input_and_weight_precision": "BF16 -> native scalar symmetric INT8 + FP32 row scales",
        }

    def _raise_error(self):
        message = self._lib.m5_qkv_error().decode(errors="replace")
        if message.startswith("UNSUPPORTED:"):
            raise NotImplementedError(message)
        raise NativeQKVError(message)

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("native QKV runner is closed")

    def upload(self, name, array):
        self._require_open()
        validate_upload(self.specs, name, array)
        slot = self.specs[name][0]
        if not self._lib.m5_qkv_upload(self._handle, slot, array.ctypes.data, array.nbytes):
            self._raise_error()
        self._initialized.add(name)
        if name == "input_bf16":
            self._initialized.difference_update(("input_int8", "input_scales"))
        elif name == "weight_bf16":
            self._initialized.difference_update(("weight_int8", "weight_scales"))
        return self

    def output(self, name):
        self._require_open()
        if name not in self.specs:
            raise ValueError(f"unknown resident buffer {name!r}")
        if name not in self._initialized:
            raise RuntimeError(f"resident buffer {name} has not been initialized")
        slot, shape, dtype = self.specs[name]
        size = ct.c_uint64()
        address = self._lib.m5_qkv_buffer(self._handle, slot, ct.byref(size))
        if not address:
            self._raise_error()
        if size.value != int(np.prod(shape)) * dtype.itemsize:
            raise NativeQKVError("C/Python resident buffer size mismatch")
        buffer = (ct.c_uint8 * size.value).from_address(address)
        # Retain the runner while the view exists; explicit close still invalidates views.
        buffer._runner = self
        result = np.ctypeslib.as_array(buffer).view(dtype).reshape(shape)
        result.flags.writeable = False
        return result

    def _validate_run(self, mode, scope):
        self._require_open()
        if mode not in _MODES or scope not in _SCOPES:
            raise ValueError(
                "mode must be native/candidate; scope core/online/quantize/weight_quantize"
            )
        if scope == "weight_quantize":
            required = {"weight_bf16"}
        elif scope == "quantize":
            required = {"input_bf16"}
        else:
            required = {"weight_int8", "weight_scales", "q_gamma", "k_gamma", "cos", "sin"}
            required |= {"input_bf16"} if scope == "online" else {"input_int8", "input_scales"}
        missing = required - self._initialized
        if missing:
            raise RuntimeError(f"resident inputs not initialized: {', '.join(sorted(missing))}")

    def _mark_run(self, scope):
        if scope == "weight_quantize":
            self._initialized.update(("weight_int8", "weight_scales"))
        elif scope in ("online", "quantize"):
            self._initialized.update(("input_int8", "input_scales"))
        if scope in ("core", "online"):
            self._initialized.update(("q", "k", "v"))

    def run(self, mode="native", scope="online"):
        self._validate_run(mode, scope)
        if not self._lib.m5_qkv_run(self._handle, _MODES[mode], _SCOPES[scope]):
            self._raise_error()
        self._mark_run(scope)
        return self

    def capture(self, path, mode="native", scope="online"):
        self._validate_run(mode, scope)
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f"refusing to overwrite capture {path}")
        if not path.parent.is_dir():
            raise ValueError("capture parent directory must exist")
        if not self._lib.m5_qkv_capture(
            self._handle, os.fsencode(path), _MODES[mode], _SCOPES[scope]
        ):
            self._raise_error()
        self._mark_run(scope)
        return self

    @property
    def last_gpu_seconds(self):
        self._require_open()
        return self._lib.m5_qkv_last_gpu_seconds(self._handle)

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.m5_qkv_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


NativeRunner = NativeQKVRunner

"""Validate native AdaLN/H256 captures and analyze frozen operands on CPU.

The light validator hashes only the small capture files. The optional projection
analysis reads selected original BF16 weight rows, never creates a GPU context,
and does not certify the actual GPU weight pack or GEMM implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from fc2_analyze import hadamard_reference, metrics, pack_check
import run as base


WIDTH = 5376
SAMPLE_ROWS = 16
FLAGS = ("hadamard_qkv", "hadamard_fc1", "hadamard_fc2")
BF16_ROWS = ("residual", "branch", "gate", "shift", "scale", "gated", "adaln")
SUFFIXES = tuple(f"{name}.bf16" for name in BF16_ROWS) + (
    "gamma.bf16",
    "codes.i8",
    "scales.f32",
    "gate-map.u32",
    "norm-map.u32",
)
IMPLEMENTATION_RELATIVE_L2_LIMIT = 0.008


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {name}")
    return value


def _read(path, dtype, shape):
    path = Path(path)
    count = int(np.prod(shape))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"capture payload must be a regular file: {path.name}")
    if path.stat().st_size != count * np.dtype(dtype).itemsize:
        raise ValueError(f"capture payload size mismatch: {path.name}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def _decode(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_round(value):
    """RNE BF16 boundary from a finite FP64/FP32 reference value."""
    value = np.asarray(value, np.float32)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite or FP32-overflowing BF16 reference")
    bits = value.view(np.uint32)
    return _decode(((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16))


def read_case(path):
    """Read one case, enforcing its geometry, finite values and file contract."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("capture metadata must be a regular file")
    meta = json.loads(path.read_text())
    if not isinstance(meta, dict) or type(meta.get("schema")) is not int or meta["schema"] != 1:
        raise ValueError("unknown AdaLN capture schema")
    if meta.get("scope") != "diagnostic_with_extra_gpu_waits":
        raise ValueError("capture must declare diagnostic waits")
    family = meta.get("family")
    if family not in ("qkv", "fc1"):
        raise ValueError("unknown capture family")
    step = _int(meta.get("step"), 0, 19, "step")
    block = _int(meta.get("block"), 0, 49, "block")
    stem = f"{family}-step{step:02d}-block{block:02d}"
    if path.name != f"{stem}.json":
        raise ValueError("metadata identity disagrees with filename")
    rows = _int(meta.get("packed_rows"), SAMPLE_ROWS, 100000, "packed rows")
    for key, expected in {
        "sample_rows": SAMPLE_ROWS,
        "width": WIDTH,
        "slots": 6,
        "zero_row_scale": 1,
        "nonfinite": 0,
    }.items():
        if type(meta.get(key)) is not int or meta[key] != expected:
            raise ValueError(f"invalid capture {key}")
    for flag in (*FLAGS, "apply_gate", "staged"):
        _int(meta.get(flag), 0, 1, flag)
    if meta[f"hadamard_{family}"] != 1:
        raise ValueError("capture family policy is disabled")
    expected_slots = (0 if block == 0 else 5, 0, 1) if family == "qkv" else (2, 3, 4)
    slot_names = ("gate_slot", "shift_slot", "scale_slot")
    if (
        any(type(meta.get(k)) is not int for k in slot_names)
        or tuple(meta[k] for k in slot_names) != expected_slots
    ):
        raise ValueError("capture modulation slots disagree with native family")
    if family == "qkv" and block == 0 and meta["apply_gate"] != 0:
        raise ValueError("first QKV cannot apply a preceding block gate")
    if type(meta.get("epsilon")) not in (int, float) or meta["epsilon"] != 1e-5:
        raise ValueError("capture epsilon mismatch")
    ids = meta.get("row_ids")
    if not isinstance(ids, list) or len(ids) != SAMPLE_ROWS:
        raise ValueError("exactly 16 selected row IDs required")
    for row in ids:
        _int(row, 0, rows - 1, "row ID")
    if len(set(ids)) != SAMPLE_ROWS:
        raise ValueError("duplicate selected row IDs")
    segments = meta.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("packed modality segments required")
    stop = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("invalid packed segment")
        _int(segment.get("kind"), 0, 255, "segment kind")
        start = _int(segment.get("start"), 0, rows, "segment start")
        end = _int(segment.get("stop"), start, rows, "segment stop")
        if start != stop:
            raise ValueError("packed segments must be a contiguous partition")
        if end > start and not {start, start + (end - start) // 2, end - 1} <= set(ids):
            raise ValueError("capture misses a modality boundary or middle row")
        stop = end
    if stop != rows:
        raise ValueError("packed segments do not cover all rows")
    layouts = {
        "input_layout": "selected_rows_full_K_BF16_little_endian",
        "modulation_layout": "gate_shift_scale_gathered_per_sample_using_independent_maps",
        "gamma_shape": [WIDTH],
        "codes_layout": "selected_rows_full_K_signed_INT8",
        "scales_layout": "one_FP32_little_endian_per_sample",
    }
    if any(meta.get(k) != v for k, v in layouts.items()):
        raise ValueError("capture layout mismatch")
    values = {}
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}}
    for suffix in SUFFIXES:
        payload = path.parent / f"{stem}-{suffix}"
        if suffix.endswith(".bf16"):
            shape = (WIDTH,) if suffix == "gamma.bf16" else (SAMPLE_ROWS, WIDTH)
            value = _decode(_read(payload, "<u2", shape))
        elif suffix == "codes.i8":
            value = _read(payload, "i1", (SAMPLE_ROWS, WIDTH))
            if np.any(value == -128):
                raise ValueError("INT8 codes outside symmetric [-127,127] contract")
        else:
            dtype = "<f4" if suffix == "scales.f32" else "<u4"
            value = _read(payload, dtype, (SAMPLE_ROWS,))
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite capture payload: {suffix}")
        values[suffix.split(".")[0]] = value
        files[payload.name] = {"bytes": payload.stat().st_size, "sha256": sha256(payload)}
    if np.any(values["scales"] <= 0):
        raise ValueError("quant scales must be positive")
    if not meta["apply_gate"]:
        if any(np.any(values[key]) for key in ("branch", "gate", "gate-map")):
            raise ValueError("pregated capture must zero unused gate operands")
        if not np.array_equal(values["gated"], values["residual"]):
            raise ValueError("pregated residual was changed")
    zero = ~np.any(values["adaln"], axis=1)
    if np.any(values["codes"][zero]) or np.any(values["scales"][zero] != 1):
        raise ValueError("zero AdaLN rows must pack to codes zero and scale one")
    return meta, values, files


def validate_capture(
    directory,
    *,
    expected_flags,
    expected_steps=(0, 10, 19),
    expected_blocks=(0, 24, 49),
    packed_rows=None,
    expected_staged=0,
):
    """Light structural/finite validation; no matrix products or weight reads."""
    d = Path(directory)
    if not d.is_dir() or d.is_symlink():
        raise ValueError("capture directory missing or symlinked")
    for flag in FLAGS:
        _int(expected_flags.get(flag), 0, 1, flag)
    _int(expected_staged, 0, 1, "expected staged mode")
    for label, selection, maximum in (
        ("steps", expected_steps, 19),
        ("blocks", expected_blocks, 49),
    ):
        if not selection or len(set(selection)) != len(selection):
            raise ValueError(f"expected {label} must be nonempty and unique")
        for item in selection:
            _int(item, 0, maximum, f"expected {label}")
    if packed_rows is not None:
        _int(packed_rows, SAMPLE_ROWS, 100000, "expected packed rows")
    families = [f for f in ("qkv", "fc1") if expected_flags[f"hadamard_{f}"]]
    if not families:
        raise ValueError("at least one H family must be enabled")
    cases, files, expected_names = [], {}, set()
    for family in families:
        for step in expected_steps:
            for block in expected_blocks:
                path = d / f"{family}-step{step:02d}-block{block:02d}.json"
                meta, _, payloads = read_case(path)
                if any(meta[k] != expected_flags[k] for k in FLAGS):
                    raise ValueError("capture flags do not match run policy")
                if meta["staged"] != expected_staged:
                    raise ValueError("capture preparation path mismatch")
                if packed_rows is not None and meta["packed_rows"] != packed_rows:
                    raise ValueError("capture packed rows do not match generation")
                cases.append(meta)
                files.update(payloads)
                expected_names.update(payloads)
    if {p.name for p in d.iterdir()} != expected_names:
        raise ValueError("capture directory contains missing or unexpected files")
    return {
        "schema": 1,
        "scope": "16 selected real rows per step/block/family; full K; diagnostic waits",
        "verifier_sha256": sha256(__file__),
        "reference_helper_sha256": sha256(Path(__file__).with_name("fc2_analyze.py")),
        "cases": cases,
        "case_count": len(cases),
        "files": files,
        "numerical_reference_evaluated": False,
    }


def analyze_preparation(meta, values):
    """Independent FP64 gate/AdaLN reference, followed by independent H packing."""
    residual = values["residual"].astype(np.float64)
    gate_reference = (
        bf16_round(residual + values["branch"].astype(np.float64) * values["gate"])
        if meta["apply_gate"]
        else values["residual"]
    )
    # Use actual gated BF16 as this stage's input to separate errors by stage.
    gated = values["gated"].astype(np.float64)
    inv = 1 / np.sqrt(np.mean(gated * gated, axis=1, keepdims=True) + meta["epsilon"])
    adaln_float64 = (
        gated * inv * values["gamma"].astype(np.float64) * (1 + values["scale"].astype(np.float64))
        + values["shift"]
    )
    adaln_reference = bf16_round(adaln_float64)
    gate_error = metrics(values["gated"], gate_reference)
    norm_error = metrics(values["adaln"], adaln_reference)
    packing = pack_check(values["adaln"], values["codes"], values["scales"][:, None], hadamard=True)
    decoded = values["codes"].astype(np.float64) * values["scales"][:, None]
    original_basis = hadamard_reference(decoded)
    passed = (
        gate_error["relative_l2"] <= IMPLEMENTATION_RELATIVE_L2_LIMIT
        and norm_error["relative_l2"] <= IMPLEMENTATION_RELATIVE_L2_LIMIT
        and packing["pass"]
    )
    return {
        "status": "passed" if passed else "incorrect",
        "gate_error_vs_independent_fp64_then_bf16": gate_error,
        "gate_exact_bf16_mismatches": int(np.count_nonzero(values["gated"] != gate_reference)),
        "adaln_error_vs_independent_fp64_then_bf16": norm_error,
        "adaln_exact_bf16_mismatches": int(np.count_nonzero(values["adaln"] != adaln_reference)),
        "relative_l2_limit": IMPLEMENTATION_RELATIVE_L2_LIMIT,
        "activation_pack_implementation": packing,
        "activation_quantization_error_in_original_basis": metrics(original_basis, values["adaln"]),
        "limits": "FP64 reference has a different reduction tree; exact equality belongs to the paired native staged/fused GPU test. Gathered condition vectors are captured; the complete condition table is not.",
    }


def selected_columns(family):
    if family == "qkv":
        heads = (0, 28, 55)
        columns = [
            h * 384 + stream * 128 + c for h in heads for stream in range(3) for c in range(128)
        ]
        return np.asarray(columns, np.int64), {
            "layout": "head,stream(Q,K,V),128",
            "heads": list(heads),
        }
    if family == "fc1":
        paired = np.linspace(0, 14335, 128, dtype=np.int64)
        return np.concatenate((paired, paired + 14336)), {
            "layout": "gate[14336],up[14336]",
            "paired_channel_ids": paired.tolist(),
        }
    raise ValueError("unknown projection family")


def verify_index(index_bytes, recorded, manifest_path=None):
    """The native loader reads shard headers, so its inventory may omit the index.

    In that case verify the CPU analyzer's index against the pinned upstream
    Git-blob digest now. This is analysis-time provenance, not a claim that the
    native generation consumed or verified this index.
    """
    digest = hashlib.sha256(index_bytes).hexdigest()
    if recorded:
        if recorded.get("sha256") != digest:
            raise ValueError("model index differs from run inventory")
        return "completed_run_inventory"
    manifest_path = manifest_path or Path(__file__).with_name("model-files.json")
    manifest_bytes = Path(manifest_path).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != base.MODEL_MANIFEST_SHA256:
        raise ValueError("model inventory pin differs")
    matches = [
        entry
        for entry in json.loads(manifest_bytes)
        if entry.get("path") == "FL2VA/transformer/model.safetensors.index.json"
        and entry.get("type") == "file"
    ]
    if len(matches) != 1:
        raise ValueError("pinned inventory lacks a unique transformer index")
    expected = matches[0]
    blob = hashlib.sha1(f"blob {len(index_bytes)}\0".encode() + index_bytes).hexdigest()
    if expected.get("size") != len(index_bytes) or expected.get("oid") != blob:
        raise ValueError("model index differs from the pinned upstream Git blob")
    return "pinned_upstream_git_blob_at_analysis_time_not_used_by_native_loader"


def read_weight_samples(model, model_record, block, family):
    """Header/index checks plus selected-row memmap; no whole-shard rehash."""
    _int(block, 0, 49, "block")
    columns, layout = selected_columns(family)
    suffix, rows = ("attn.qkv_proj.weight", 21504) if family == "qkv" else ("mlp.fc1.weight", 28672)
    tensor_name = f"blocks.{block}.{suffix}"
    model = Path(model)
    index_path = model / "FL2VA/transformer/model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    shard = index.get("weight_map", {}).get(tensor_name)
    if (
        not isinstance(shard, str)
        or Path(shard).name != shard
        or not shard.endswith(".safetensors")
    ):
        raise ValueError("projection is missing from index or has an unsafe shard path")
    relative = f"FL2VA/transformer/{shard}"
    if any(
        model_record.get(k) != v
        for k, v in {
            "repository": "MiniMaxAI/MiniMax-H3",
            "revision": base.MODEL_REVISION,
            "manifest_sha256": base.MODEL_MANIFEST_SHA256,
        }.items()
    ):
        raise ValueError("original checkpoint identity differs from pinned BF16 model")
    records = {entry["path"]: entry for entry in model_record["files"]}
    if len(records) != len(model_record["files"]) or relative not in records:
        raise ValueError("missing/duplicate run model inventory entry")
    index_record = records.get("FL2VA/transformer/model.safetensors.index.json", {})
    index_verification = verify_index(index_bytes, index_record)
    path = model / relative
    before = path.stat()
    if before.st_size != records[relative]["bytes"]:
        raise ValueError("original weight shard size differs from run inventory")
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated weight header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 100 * 1024 * 1024:
            raise ValueError("invalid weight header length")
        header_bytes = file.read(length)
    header = json.loads(header_bytes)
    tensor = header.get(tensor_name, {})
    offsets = tensor.get("data_offsets")
    if (
        tensor.get("dtype") != "BF16"
        or tensor.get("shape") != [rows, WIDTH]
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(x) is not int for x in offsets)
        or not 0 <= offsets[0] < offsets[1] <= before.st_size - 8 - length
        or offsets[1] - offsets[0] != rows * WIDTH * 2
    ):
        raise ValueError("original BF16 projection tensor contract mismatch")
    mapped = np.memmap(
        path, mode="r", dtype="<u2", offset=8 + length + offsets[0], shape=(rows, WIDTH)
    )
    bits = np.asarray(mapped[columns]).copy()
    del mapped
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("original weight shard changed while sampling")
    weight = _decode(bits)
    if not np.isfinite(weight).all():
        raise ValueError("nonfinite original weight sample")
    provenance = {
        "tensor": tensor_name,
        "original_dtype": "BF16",
        "original_shape": [rows, WIDTH],
        "data_offsets": offsets,
        "source_file": relative,
        "run_declared_source_file_sha256": records[relative]["sha256"],
        "source_shard_digest_reverified_by_this_analyzer": False,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "index_verification": index_verification,
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "sample_bf16_bytes_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
        "selected_output_columns": columns.tolist(),
        "sampling": layout,
    }
    return weight, provenance


def _quantize_reference(value, hadamard):
    value = hadamard_reference(value, np.float32) if hadamard else np.asarray(value, np.float32)
    peak = np.max(np.abs(value), axis=1)
    divisor = np.where(peak > 0, peak, np.float32(1))
    scales = np.where(
        peak > 0, peak / np.float32(127), 1 if hadamard else np.float32(1 / 127)
    ).astype(np.float32)
    normalized = (
        value / scales[:, None] if hadamard else value * (np.float32(127) / divisor)[:, None]
    )
    codes = np.clip(np.rint(normalized), -127, 127).astype(np.int8)
    return codes.astype(np.float64) * scales[:, None]


def projection_precision(values, weight):
    """Same frozen post-AdaLN BF16 X/W, both policies quantized on CPU."""
    x = values["adaln"]
    reference = x.astype(np.float64) @ weight.astype(np.float64).T
    rotated = hadamard_reference(x) @ hadamard_reference(weight).T
    dot = metrics(rotated, reference)
    outputs = {}
    for name, hadamard in (("plain_row_int8", False), ("h256_row_int8", True)):
        decoded = _quantize_reference(x, hadamard) @ _quantize_reference(weight, hadamard).T
        outputs[name] = decoded
    plain, h = outputs["plain_row_int8"], outputs["h256_row_int8"]
    per_plain = [metrics(a, b) for a, b in zip(plain, reference)]
    per_h = [metrics(a, b) for a, b in zip(h, reference)]
    actual_x = values["codes"].astype(np.float64) * values["scales"][:, None]
    mixed = actual_x @ _quantize_reference(weight, True).T
    plain_norm = np.linalg.norm(plain - reference, axis=1)
    h_norm = np.linalg.norm(h - reference, axis=1)
    return {
        "scope": "CPU quantization simulation on identical frozen BF16 post-AdaLN X and selected original BF16 W rows, full K. No GPU GEMM output or actual GPU weight codes were captured; this is precision evidence, not GPU weight/GEMM implementation validation.",
        "reference": "independent NumPy float64 original BF16 product before Q/K RMS/RoPE or FC1 SwiGLU",
        "float64_rotation_dot_preservation": dot,
        "rotation_check_pass": bool(dot["relative_l2"] <= 1e-12),
        "plain_row_int8_quantization_error": metrics(plain, reference),
        "h256_row_int8_quantization_error": metrics(h, reference),
        "per_row_plain_quantization_error": per_plain,
        "per_row_h256_quantization_error": per_h,
        "h256_per_row_l2_better": int(np.count_nonzero(h_norm < plain_norm)),
        "h256_per_row_l2_equal": int(np.count_nonzero(h_norm == plain_norm)),
        "h256_per_row_l2_worse": int(np.count_nonzero(h_norm > plain_norm)),
        "actual_GPU_activation_CPU_weight_mixed_product_error": metrics(mixed, reference),
        "projection_implementation_status": "not measured by this capture",
        "quality_decision": "not evaluated",
    }


def analyze(directory, *, expected_flags, model=None, run_record=None, expected_staged=0):
    verified = validate_capture(
        directory, expected_flags=expected_flags, expected_staged=expected_staged
    )
    if (model is None) != (run_record is None):
        raise ValueError("projection analysis needs both original model and completed run record")
    run_sha, model_record = None, None
    if run_record is not None:
        run_record = Path(run_record)
        run_sha = sha256(run_record)
        run_data = json.loads(run_record.read_text())
        if run_data.get("status") != "completed" or not isinstance(run_data.get("model"), dict):
            raise ValueError("completed run with original model inventory required")
        model_record = run_data["model"]
        recorded = run_data.get("adaln_validation", {})
        if recorded.get("files") != verified["files"] or recorded.get("cases") != verified["cases"]:
            raise ValueError("capture files/cases are not bound to the supplied completed run")
    cases = []
    for meta in verified["cases"]:
        path = (
            Path(directory)
            / f"{meta['family']}-step{meta['step']:02d}-block{meta['block']:02d}.json"
        )
        current, values, _ = read_case(path)
        if current != meta:
            raise ValueError("capture metadata changed during analysis")
        case = {"metadata": meta, "preparation": analyze_preparation(meta, values)}
        if model_record is not None:
            weight, provenance = read_weight_samples(
                model, model_record, meta["block"], meta["family"]
            )
            case["weight_sample"] = provenance
            case["projection_precision"] = projection_precision(values, weight)
        cases.append(case)
    # Bind all readbacks after the computation as well; retain source captures.
    for name, info in verified["files"].items():
        if sha256(Path(directory) / name) != info["sha256"]:
            raise ValueError("capture changed during analysis")
    if run_record is not None and sha256(run_record) != run_sha:
        raise ValueError("run record changed during analysis")
    return {
        "schema": 1,
        "analyzer_sha256": sha256(__file__),
        "reference_helper_sha256": sha256(Path(__file__).with_name("fc2_analyze.py")),
        "capture": verified,
        "source_run_sha256": run_sha,
        "cases": cases,
        "all_preparation_checks_pass": all(c["preparation"]["status"] == "passed" for c in cases),
        "all_projection_rotation_checks_pass": (
            all(c["projection_precision"]["rotation_check_pass"] for c in cases)
            if model_record is not None
            else None
        ),
        "promotion_established": False,
        "limitations": "Selected rows/channels cannot establish full-model numerical or media quality. Original shard hashes are inherited from the completed run, not rehashed here. GPU weight-pack and GEMM implementation require the independent GPU driver.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--hadamard-qkv", type=int, choices=(0, 1), required=True)
    parser.add_argument("--hadamard-fc1", type=int, choices=(0, 1), required=True)
    parser.add_argument("--hadamard-fc2", type=int, choices=(0, 1), default=1)
    parser.add_argument("--staged", type=int, choices=(0, 1), default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--run-record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    flags = {k: getattr(args, k) for k in FLAGS}
    try:
        result = analyze(
            args.directory,
            expected_flags=flags,
            expected_staged=args.staged,
            model=args.model,
            run_record=args.run_record,
        )
    except Exception as exc:
        result = {
            "schema": 1,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "analyzer_sha256": sha256(__file__),
        }
        with args.output.open("x") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
        raise
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
